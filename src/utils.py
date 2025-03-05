import torch
import random
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
import json
from scipy.stats import spearmanr

def pad_embeddings_and_create_mask(embedding_list):
    if all(e is None for e in embedding_list):
        return None, None
    
    padded = pad_sequence(embedding_list, batch_first=True, padding_value=0.0)
    batch_size = padded.size(0)
    max_len = padded.size(1)

    lengths = torch.tensor([e.size(0) for e in embedding_list], dtype=torch.long)
    # mask: True for padding, False for non-padding
    mask = torch.arange(max_len).unsqueeze(0).expand(batch_size, max_len)
    mask = mask >= lengths.unsqueeze(1)

    return padded, mask

def pad_last_batch(batch, batch_size):
    current_size = len(batch)
    if current_size < batch_size:
        pad_size = batch_size - current_size
        for _ in range(pad_size):
            i = random.randint(0, current_size - 1)
            sample = batch[i]
            batch.append(sample)
    return batch

def custom_collate_fn_padding(batch):
    """
    batch: [((trill_emb, w2v2_emb, text_emb), label), ...]
    1) trill_list, w2v2_list, text_list, label_list
    2) create (padded, mask)
    3) label_list -> tensor
    4) ((trill_pad, trill_mask), (w2v2_pad, w2v2_mask), (text_pad, text_mask)), labels
    """
    trill_list, w2v2_list, text_list, label_list = [], [], [], []

    for (trill_emb, w2v2_emb, text_emb), label in batch:
        trill_list.append(trill_emb)
        w2v2_list.append(w2v2_emb)
        text_list.append(text_emb)
        label_list.append(label)

    labels = torch.stack(label_list).float()

    trill_pad, trill_mask = pad_embeddings_and_create_mask(trill_list)
    w2v2_pad, w2v2_mask = pad_embeddings_and_create_mask(w2v2_list)
    text_pad, text_mask = pad_embeddings_and_create_mask(text_list)

    return ((trill_pad, trill_mask), (w2v2_pad, w2v2_mask), (text_pad, text_mask)), labels

def compute_metrics(y_true, y_pred):
    scc, pvalue = spearmanr(y_true, y_pred)
    return (scc, pvalue)

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0

def save_inference_results(y_true, y_pred, spearman_corr, pvalue, output_path):
    y_pred_list = y_pred.tolist()
    y_true_list = y_true.tolist()

    results = {
        "test_metrics": {
            "Spearman Correlation Coefficient": spearman_corr,
            "p-value": float(pvalue)
        },
        "predictions": [
            {"index": i, "prediction": float(pred), "label": float(label)}
            for i, (pred, label) in enumerate
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Test results saved to {output_path}")