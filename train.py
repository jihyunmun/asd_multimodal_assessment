# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from utils import compute_metrics
from tqdm import tqdm

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def train_one_epoch(model, data_loader, optimizer, criterion, device, clip_value=1.0):

    model.train()
    total_loss = 0
    total_samples = 0
    all_preds, all_labels = [], []

    pbar = tqdm(data_loader, desc='Training')
    for i, batch in enumerate(pbar):
        (trill_pad, trill_mask, w2v2_pad, w2v2_mask, text_pad, text_mask), labels = batch

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

        optimizer.zero_grad()

        y_pred = model(
            trill_pad, trill_mask,
            w2v2_pad, w2v2_mask,
            text_pad, text_mask
        )

        loss = criterion(y_pred.squeeze(-1), y_true)
        loss.backward()

        clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        batch_size = y_true.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_preds.append(y_pred.detach().cpu().squeeze(-1))
        all_labels.append(y_true.detach().cpu())

        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / total_samples
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    train_scc, _ = compute_metrics(all_labels, all_preds)

    return avg_loss, train_scc

def validate_one_epoch(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Validating')
        for i, batch in enumerate(pbar):
            (trill_pad, trill_mask, w2v2_pad, w2v2_mask, text_pad, text_mask), labels = batch

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

            y_pred = model(
                trill_pad, trill_mask,
                w2v2_pad, w2v2_mask,
                text_pad, text_mask
            )

            loss = criterion(y_pred.squeeze(-1), y_true)

            batch_size = y_true.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(y_pred.detach().cpu().squeeze(-1))
            all_labels.append(y_true.detach().cpu())

            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / total_samples
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        val_scc, _ = compute_metrics(all_labels, all_preds)

    return avg_loss, val_scc