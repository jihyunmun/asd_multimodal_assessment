import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from config import Config, set_seed
from utils import (
    pad_last_batch,
    custom_collate_fn_padding,
    EarlyStopping,
    save_inference_results,
    compute_metrics
)
from dataset import CustomAudioTextDataset
from model import SpeechEmbeddingExtractor, TextEmbeddingExtractor, Multimodal
from train import (
    train_one_epoch,
    validate_one_epoch,
    initialize_weights
)

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