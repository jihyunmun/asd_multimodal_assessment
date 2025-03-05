import os
import librosa
import numpy as np
import pandas as pd
import torch

import tensorflow as tf
import kagglehub

from transformers import (
    Wav2Vec2FeatureExtractor, Wav2Vec2Model,
    AutoTokenizer, AutoModel
)

from config import Config

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