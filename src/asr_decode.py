import os
import librosa
import torch
import pandas as pd
from huggingface_hub import login
from transformers import AutoProcessor, WhisperForConditionalGeneration

from config import Config

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
