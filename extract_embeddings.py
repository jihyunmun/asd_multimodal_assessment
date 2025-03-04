import os
import pandas as pd
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoTokenizer, AutoModel
import tensorflow as tf
import kagglehub
import numpy as np
import librosa
from config import (TRAIN_CSV, VALID_CSV, TEST_CSV, TRAIN_EMB_DIR, VALID_EMB_DIR, TEST_EMB_DIR, MAX_TEXT_LENGTH)


########################################
# Load Models
########################################
# Trillsson
trillsson_path = kagglehub.model_download("google/trillsson/tensorFlow2/5")
print("Path to Trillsson model files:", trillsson_path)
trillsson_model = tf.keras.models.load_model(trillsson_path)
trillsson_model.trainable = False

trillsson_embedding_model = tf.keras.Model(
    inputs=trillsson_model.input,
    outputs=trillsson_model.get_layer('tf.reshape_1').output
)

# Wav2Vec2
w2v2_name = 'facebook/wav2vec2-large-xlsr-53'
w2v2_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    w2v2_name, 
    cache_dir='/data5/jihyeon1202/asd_mm/cache/'
)
w2v2_model = Wav2Vec2Model.from_pretrained(
    w2v2_name, 
    cache_dir='/data5/jihyeon1202/asd_mm/cache/'
)
for param in w2v2_model.parameters():
    param.requires_grad = False

# klue/roberta-base
roberta_name = 'klue/roberta-base'

roberta_tokenizer = AutoTokenizer.from_pretrained(
    roberta_name,
    cache_dir='/data5/jihyeon1202/cache/'
)
roberta_model = AutoModel.from_pretrained(
    roberta_name,
    cache_dir='/data5/jihyeon1202/cache/'
)
for param in roberta_model.parameters():
    param.requires_grad = False

########################################
# Define embedding extraction functions
########################################
def extract_trillsson_embedding(waveform):
    waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
    waveform_tf = tf.expand_dims(waveform_tf, axis=0)
    output = trillsson_embedding_model(waveform_tf)
    return output.numpy()[0]

def extract_w2v2_embedding(waveform):
    inputs = w2v2_feature_extractor(waveform, return_tensors='pt', padding=True)
    with torch.no_grad():
        outputs = w2v2_model(**inputs)
    return outputs.last_hidden_state.squeeze(0).cpu().numpy()

def extract_text_embedding(text):
    inputs = roberta_tokenizer(
        text, 
        truncation=True,
        padding='max_length',
        max_length=MAX_TEXT_LENGTH,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    return outputs.last_hidden_state.squeeze(0).cpu().numpy()

########################################
# Define dataset processing function
########################################
def process_dataset(dataset_type, csv_path, emb_dir):
    df = pd.read_csv(csv_path)

    trillsson_paths = []
    w2v2_paths = []
    text_paths = []

    os.makedirs(emb_dir, exist_ok=True)

    for i, row in df.iterrows():
        audio_file = row['speech']
        text = row['transcription']

        print(f"[{dataset_type}][{i}] Processing audio file: {audio_file}")
        waveform, _ = librosa.load(audio_file, sr=16000, mono=True)

        trillsson_path = os.path.join(emb_dir, f"trillsson_{i}.npy")
        w2v2_path = os.path.join(emb_dir, f"w2v2_{i}.npy")
        text_path = os.path.join(emb_dir, f"text_{i}.npy")

        trillsson_paths.append(trillsson_path)
        w2v2_paths.append(w2v2_path)
        text_paths.append(text_path)

        if not os.path.exists(trillsson_path):
            trillsson_embedding = extract_trillsson_embedding(waveform)
            w2v2_embedding = extract_w2v2_embedding(waveform)
            text_embedding = extract_text_embedding(text)

            np.save(trillsson_path, trillsson_embedding)
            np.save(w2v2_path, w2v2_embedding)
            np.save(text_path, text_embedding)

    df['trillsson_path'] = trillsson_paths
    df['w2v2_path'] = w2v2_paths
    df['text_path'] = text_paths

    df.to_csv(csv_path, index=False)
    print(f"[{dataset_type}] Saved new columns (trillsson_path, w2v2_path, text_path) into {csv_path}")

########################################
# Main
########################################
def main():
    datasets = {
        "train": {"csv": TRAIN_CSV, "emb_dir": TRAIN_EMB_DIR},
        "valid": {"csv": VALID_CSV, "emb_dir": VALID_EMB_DIR},
        "test": {"csv": TEST_CSV, "emb_dir": TEST_EMB_DIR}
    }

    for dataset_type, paths in datasets.items():
        process_dataset(dataset_type, paths['csv'], paths['emb_dir'])


if __name__ == '__main__':
    main()