import os
import argparse
import torch
import librosa
import numpy as np
import pandas as pd

from huggingface_hub import login
from transformers import AutoProcessor, WhisperForConditionalGeneration

from config import Config, set_seed
from utils import (
    pad_last_batch,
    custom_collate_fn_padding,
    EarlyStopping,
    save_inference_results,
    compute_metrics
)
from train import (
    train_one_epoch,
    validate_one_epoch,
    initialize_weights
)

from dataset import CustomAudioTextDataset
from model import SpeechEmbeddingExtractor, TextEmbeddingExtractor, Multimodal

import tensorflow as tf
import kagglehub

import torch.nn as nn
import torch.optim as optim
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoTokenizer, AutoModel


#########################################################################################################
# ASR
#########################################################################################################
def run_asr_decode():
    """
    Run ASR Model to decode speech to text (Whisper model-based)
    """
    print("[MODE: decode] Running ASR ...")
    login(token=Config.HUGGINGFACE_TOKEN)
    processor = AutoProcessor.from_pretrained(Config.ASR_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(Config.ASR_MODEL)

    for csv_path in [Config.TRAIN_CSV_PATH, Config.VALID_CSV_PATH, Config.TEST_CSV_PATH]:
        if not os.path.exists(csv_path):
            print(f"[ERROR] {csv_path} not found.")
            continue
        print(f"[INFO] Decoding {csv_path} ...")

        df = pd.read_csv(csv_path)
        transcriptions = []

        for i, row in df.iterrows():
            audio_path = row['speech']
            if not os.path.exists(audio_path):
                print(f"[ERROR] {audio_path} not found.")
                continue
                
            waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
            inputs = processor(waveform, sampling_rate=sr, return_tensors="pt")
            input_features = inputs.input_features.to(Config.DEVICE)

            with torch.no_grad():
                generated_ids = model.generate(input_features)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            transcriptions.append(transcription)

        df[Config.TEXT_COLUMN_NAME] = transcriptions
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Updated '{Config.TEXT_COLUMN_NAME}' column in {csv_path}")

#########################################################################################################
# extract embeddings
#########################################################################################################
def run_extract_embeddings():
    """
    extract speech (TRILLsson, Wav2vec2) and text (roberta) embeddings and save
    """
    print("[MODE: embed] Extracting embeddings ...")

    # TRILLsson
    trillsson_path = kagglehub.model_download("google/trillsson/tensorFlow2/5")
    print("[INFO] TRILLsson model path: ", trillsson_path)
    trillsson_model = tf.keras.models.load_model(trillsson_path)
    trillsson_model.trainable = False
    trillsson_embedding_model = tf.keras.Model(
        inputs = trillsson_model.input,
        outputs = trillsson_model.get_layer('tf.reshape_1').output
    )

    def extract_trillsson_embedding(waveform):
        w_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
        w_tf = tf.expand_dims(w_tf, axis=0)
        output = trillsson_embedding_model(w_tf)
        return output.numpy()[0]
    
    # Wav2Vec2
    w2v2_name = "facebook/wav2vec2-large-xlsr-53"
    w2v2_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(w2v2_name)
    w2v2_model = Wav2Vec2Model.from_pretrained(w2v2_name)
    for param in w2v2_model.parameters():
        param.requires_grad = False

    def extract_w2v2_embedding(waveform):
        inputs = w2v2_feature_extractor(waveform, return_tensors='pt', sampling_rate=16000)
        with torch.no_grad():
            outputs = w2v2_model(**inputs)
        return outputs.last_hidden_state.squeeze(0).cpu().numpy()
    
    # Roberta
    roberta_name = "klue/roberta-base"
    roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_name)
    roberta_model = AutoModel.from_pretrained(roberta_name)
    for param in roberta_model.parameters():
        param.requires_grad = False

    def extract_text_embedding(text):
        if not isinstance(text, str) or text.strip() == "":
            return np.zeros((Config.MAX_TEXT_LENGTH, 768), dtype=np.float32)
        
        inputs = roberta_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=Config.MAX_TEXT_LENGTH,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = roberta_model(**inputs)
        return outputs.last_hidden_state.squeeze(0).cpu().numpy()
    
    def process_dataset(csv_path, emb_dir):
        if not os.path.exists(csv_path):
            print(f"[ERROR] {csv_path} not found.")
            return
        
        os.makedirs(emb_dir, exist_ok=True)
        df = pd.read_csv(csv_path)

        trillsson_paths, w2v2_paths, text_paths = [], [], []

        for i, row in df.iterrows():
            audio_file = row['speech']
            filename = audio_file.split('/')[-1].split('.')[0]
            asr_text = row.get(Config.TEXT_COLUMN_NAME, "")

            trill_path = os.path.join(emb_dir, f'trill_{filename}.npy')
            w2v2_path = os.path.join(emb_dir, f'w2v2_{filename}.npy')
            text_path = os.path.join(emb_dir, f'text_{filename}.npy')

            if os.path.exists(audio_file):
                waveform, _ = librosa.load(audio_file, sr=16000, mono=True)
                if not os.path.exists(trill_path):
                    t_emb = extract_trillsson_embedding(waveform)
                    np.save(trill_path, t_emb)
                    w_emb = extract_w2v2_embedding(waveform)
                    np.save(w2v2_path, w_emb)
                else:
                    trill_path = None
                    w2v2_path = None

            if not os.path.exists(text_path) and (asr_text is not None):
                text_emb = extract_text_embedding(asr_text)
                np.save(text_path, text_emb)

            trillsson_paths.append(trill_path)
            w2v2_paths.append(w2v2_path)
            text_paths.append(text_path)

        df['trillsson_path'] = trillsson_paths
        df['w2v2_path'] = w2v2_paths
        df[Config.TEXT_COLUMN_NAME] = text_paths
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Updated CSV with embedding paths => {csv_path}")

    process_dataset(Config.TRAIN_CSV_PATH, Config.TRAIN_EMB_DIR)
    process_dataset(Config.VALID_CSV_PATH, Config.VALID_EMB_DIR)
    process_dataset(Config.TEST_CSV_PATH, Config.TEST_EMB_DIR)

#########################################################################################################
# Train
#########################################################################################################
def run_train():
    print("[MODE: train] Training ...")

    if not os.path.exists(Config.TRAIN_CSV_PATH) or not os.path.exists(Config.VALID_CSV_PATH):
        print("[ERROR] Train or Valid CSV not found.")
        return
    
    train_dataset = CustomAudioTextDataset(
        csv_path = Config.TRAIN_CSV_PATH,
        text_column = Config.TEXT_COLUMN_NAME,
        **Config.DATASET_PARAMS
    )
    valid_dataset = CustomAudioTextDataset(
        csv_path = Config.VALID_CSV_PATH,
        text_column = Config.TEXT_COLUMN_NAME,
        **Config.DATASET_PARAMS
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn = lambda batch: custom_collate_fn_padding(pad_last_batch(batch, Config.BATCH_SIZE))
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn = lambda batch: custom_collate_fn_padding(pad_last_batch(batch, Config.BATCH_SIZE))
    )

    for seed in Config.SEEDS:
        set_seed(seed)
        print(f"[INFO] Training with seed {seed}")

        speech_extractor = SpeechEmbeddingExtractor(
            use_speech=Config.DATASET_PARAMS['use_speech'],
            use_trillsson=Config.DATASET_PARAMS['use_trillsson'],
            use_w2v2=Config.DATASET_PARAMS['use_w2v2'],
            hidden_dim=512
        ).to(Config.DEVICE)
        text_extractor = TextEmbeddingExtractor(
            use_text=Config.DATASET_PARAMS['use_text'],
            hidden_dim=512
        ).to(Config.DEVICE)

        model = Multimodal(
            speech_extractor=speech_extractor,
            text_extractor=text_extractor,
            embed_dim=512,
            hidden_dim=512
        ).to(Config.DEVICE)

        model.apply(initialize_weights)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
        early_stopping = EarlyStopping(patience=Config.PATIENCE, delta=0.0)

        best_loss = float('inf')
        for epoch in range(Config.NUM_EPOCHS):
            print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
            train_loss, train_scc = train_one_epoch(
                model, train_loader, optimizer, criterion, Config.DEVICE, clip_value=Config.CLIP_VALUE
            )
            valid_loss, valid_scc = validate_one_epoch(
                model. valid_loader, criterion, Config.DEVICE
            )

            print(f"Train Loss: {train_loss:.4f} | Train SCC: {train_scc:.4f}")
            print(f"Valid Loss: {valid_loss:.4f} | Valid SCC: {valid_scc:.4f}")

            if valid_loss < best_loss:
                best_loss = valid_loss
                ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"model_seed{seed}.pt")
                os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)

            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("[INFO] Early stopping")
                break

    print("[INFO] Training completed.")

#########################################################################################################
# Inference
#########################################################################################################
def run_inference():
    print("[MODE: inference] Inference ...")
    if not os.path.exists(Config.TEST_CSV_PATH):
        print("[ERROR] Test CSV not found.")
        return
    
    test_dataset = CustomAudioTextDataset(
        csv_path = Config.TEST_CSV_PATH,
        text_column = Config.TEXT_COLUMN_NAME,
        **Config.DATASET_PARAMS
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn = lambda batch: custom_collate_fn_padding(pad_last_batch(batch, Config.BATCH_SIZE))
    )

    y_pred_total = torch.zeros(len(test_dataset), dtype=torch.float32, device=Config.DEVICE)
    y_true_total = torch.zeros(len(test_dataset), dtype=torch.float32, device=Config.DEVICE)

    def create_infer_model():
        speech_extractor = SpeechEmbeddingExtractor(
            use_speech=Config.DATASET_PARAMS['use_speech'],
            use_trillsson=Config.DATASET_PARAMS['use_trillsson'],
            use_w2v2=Config.DATASET_PARAMS['use_w2v2'],
            hidden_dim=512
        ).to(Config.DEVICE)
        text_extractor = TextEmbeddingExtractor(
            use_text=Config.DATASET_PARAMS['use_text'],
            hidden_dim=512
        ).to(Config.DEVICE)
        model_infer = Multimodal(
            speech_extractor=speech_extractor,
            text_extractor=text_extractor,
            embed_dim=512,
            hidden_dim=512
        ).to(Config.DEVICE)
        return model_infer
    
    start_idx = 0
    for i, seed in enumerate(Config.SEEDS):
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"model_seed{seed}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[ERROR] {ckpt_path} not found.")
            continue

        model_infer = create_infer_model()
        model_infer.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))
        model_infer.eval()

        batch_start = 0
        with torch.no_grad():
            for batch in test_loader:
                (trill_pad, trill_mask, w2v2_pad, w2v2_mask, text_pad, text_mask), labels = batch
                bsz = labels.size(0)

                if trill_pad is not None: trill_pad = trill_pad.to(Config.DEVICE)
                if trill_mask is not None: trill_mask = trill_mask.to(Config.DEVICE)
                if w2v2_pad is not None: w2v2_pad = w2v2_pad.to(Config.DEVICE)
                if w2v2_mask is not None: w2v2_mask = w2v2_mask.to(Config.DEVICE)
                if text_pad is not None: text_pad = text_pad.to(Config.DEVICE)
                if text_mask is not None: text_mask = text_mask.to(Config.DEVICE)
                y_true = labels.to(Config.DEVICE)

                y_pred = model_infer(
                    trill_pad, trill_mask, w2v2_pad, w2v2_mask, text_pad, text_mask
                )

                y_pred_total[batch_start:batch_start+bsz] += y_pred
                if i == 0:
                    y_true_total[batch_start:batch_start+bsz] = y_true

                batch_start += bsz

    num_seeds = len(Config.SEEDS)
    if num_seeds == 0:
        print("[ERROR] No checkpoint found.")
        return
    
    y_pred_total /= num_seeds
    y_pred_total = y_pred_total.cpu()
    y_true_total = y_true_total.cpu()

    test_scc, test_pvalue = compute_metrics(y_true_total, y_pred_total)
    print(f"Test SCC: {test_scc:.4f} | p-value: {test_pvalue:.4f}")

    save_inference_results(
        y_true_total, y_pred_total, test_scc, test_pvalue, Config.OUTPUT_JSON
    )

#########################################################################################################
# Main
#########################################################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        help='Choose between "decode", "embed", "train", "inference", "all"')
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    mode = args.mode.lower()
    if mode == 'decode':
        run_asr_decode()
    elif mode == 'embed':
        run_extract_embeddings()
    elif mode == 'train':
        run_train()
    elif mode == 'inference':
        run_inference()
    elif mode == 'all':
        run_asr_decode()
        run_extract_embeddings()
        run_train()
        run_inference()
    else:
        print("[ERROR] Invalid mode. Choose between 'decode', 'embed', 'train', 'inference', 'all'.")

if __name__ == "__main__":
    main()