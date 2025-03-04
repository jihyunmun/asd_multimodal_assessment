import os
import torch
import torch.nn as nn
import torch.optim as optim

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

def main():
    os.envirom['CUDA_VISIBLE_DEVICES'] = '0'
    device = Config.DEVICE

    train_csv  = Config.TRAIN_CSV_PATH
    valid_csv  = Config.VALID_CSV_PATH
    test_csv   = Config.TEST_CSV_PATH

    print("[INFO] Loading datasets...")
    train_dataset = CustomAudioTextDataset(
        csv_path = train_csv,
        text_column = Config.TEXT_COLUMN_NAME,
        **Config.DATASET_PARAMS
    )
    valid_dataset = CustomAudioTextDataset(
        csv_path = valid_csv,
        text_column = Config.TEXT_COLUMN_NAME,
        **Config.DATASET_PARAMS
    )
    test_dataset = CustomAudioTextDataset(
        csv_path = test_csv,
        text_column = Config.TEXT_COLUMN_NAME,
        **Config.DATASET_PARAMS
    )

    for seed in Config.SEEDS:
        print('*'*50)
        print(f"Start Training with Seed {seed}")
        print('*'*50)
        set_seed(seed)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn = lambda batch: custom_collate_fn_padding(
                pad_last_batch(batch, Config.BATCH_SIZE)
            )
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn = lambda batch: custom_collate_fn_padding(
                pad_last_batch(batch, Config.BATCH_SIZE)
            )
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn = lambda batch: custom_collate_fn_padding(
                pad_last_batch(batch, Config.BATCH_SIZE)
            )
        )

        # ---------------------------
        # Prepare Model
        # ---------------------------
        speech_extractor = SpeechEmbeddingExtractor(
            use_speech = Config.DATASET_PARAMS['use_speech'],
            use_trillsson = Config.DATASET_PARAMS['use_trillsson'],
            use_w2v2 = Config.DATASET_PARAMS['use_w2v2'],
            hidden_dim = 512
        ).to(device)

        text_extractor = TextEmbeddingExtractor(
            use_text = Config.DATASET_PARAMS['use_text'],
            hidden_dim = 512
        ).to(device)

        model = Multimodal(
            speech_extractor=speech_extractor,
            text_extractor=text_extractor,
            embed_dim=512,
            hidden_dim=512
        ).to(device)

        model.apply(initialize_weights)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
        early_stopper = EarlyStopping(patience=Config.PATIENCE, delta=0)
        best_loss = float('inf')

        # ---------------------------
        # Train & Validate
        # ---------------------------
        for epoch in range(Config.NUM_EPOCHS):
            print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
            train_loss, train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion, device, Config.CLIP_VALUE
            )
            valid_loss, valid_metrics = validate_one_epoch(
                model, valid_loader, criterion, device
            )

            print(f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
            print(f"Train SCC: {train_metrics} | Valid SCC: {valid_metrics}")

            if valid_loss < best_loss:
                best_loss = valid_loss
                checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"best_model_seed_{seed}.pth")
                torch.save(model.state_dict(), checkpoint_path)

            early_stopper(valid_loss, model)
            if early_stopper.early_stop:
                print("[INFO] Early Stopping")
                break

    y_pred_total = torch.zeros(len(test_dataset)).to(device)
    y_true_total = torch.zeros(len(test_dataset)).to(device)

    for i, seed in enumerate(Config.SEEDS):
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"best_model_seed_{seed}.pth")

        speech_extractor = SpeechEmbeddingExtractor(
            use_speech = Config.DATASET_PARAMS['use_speech'],
            use_trillsson = Config.DATASET_PARAMS['use_trillsson'],
            use_w2v2 = Config.DATASET_PARAMS['use_w2v2'],
            hidden_dim = 512
        ).to(device)
        text_extractor = TextEmbeddingExtractor(
            use_text = Config.DATASET_PARAMS['use_text'],
            hidden_dim = 512
        ).to(device)
        model_infer = Multimodal(
            speech_extractor=speech_extractor,
            text_extractor=text_extractor,
            embed_dim=512,
            hidden_dim=512
        ).to(device)

        model_infer.load_state_dict(torch.load(ckpt_path, map_location=device))
        model_infer.eval()

        start_idx = 0
        with torch.no_grad():
            for batch in test_loader:
                (trill_pad, trill_mask, w2v2_pad, w2v2_mask, text_pad, text_mask), labels = batch
                bsz = labels.size(0)

                if trill_pad is not None:
                    trill_pad = trill_pad.to(device)
                if trill_mask is not None:
                    trill_mask = trill_mask.to(device)
                if w2v2_pad is not None:
                    w2v2_pad = w2v2_pad.to(device)
                if w2v2_mask is not None:
                    w2v2_mask = w2v2_mask.to(device)
                if text_pad is not None:
                    text_pad = text_pad.to(device)
                if text_mask is not None:
                    text_mask = text_mask.to(device)

                y_true = labels.to(device)
                y_pred = model_infer(
                    trill_pad, trill_mask,
                    w2v2_pad, w2v2_mask,
                    text_pad, text_mask
                ).squeeze(-1)

                y_pred_total[start_idx:start_idx+bsz] += y_pred

                if i == 0:
                    y_true_total[start_idx:start_idx+bsz] = y_true
                start_idx += bsz

    y_pred_total /= len(Config.SEEDS)

    y_pred_total = y_pred_total.cpu()
    y_true_total = y_true_total.cpu()

    test_scc, test_pvalue = compute_metrics(y_true_total, y_pred_total)
    print(f"Test SCC: {test_scc:.4f} | Test P-value: {test_pvalue:.4f}")

    save_inference_results(
        y_true_total, y_pred_total,
        test_scc, test_pvalue,
        Config.OUTPUT_JSON
    )

if __name__ == '__main__':
    main()