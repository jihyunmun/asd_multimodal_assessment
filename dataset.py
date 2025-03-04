import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch.nn.functional as F

class CustomAudioTextDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 text_column: str,
                 use_speech: bool = True,
                 use_text: bool = True,
                 use_trillsson: bool = True,
                 use_w2v2: bool = True,
                 preload_to_ram: bool = False):
        """
        Args:
            csv_path: path to CSV file
            text_column: target text column's name in the CSV file
            use_speech: whether to use audio embeddings (Trillsson, W2V2) overall
            use_text: whether to use text embeddings overall
            use_trillsson: whether to use TRILLsson embeddings
            use_w2v2: whether to use W2V2 embeddings
            preload_to_ram: whether to preload to RAM
        """

        self.data = pd.read_csv(csv_path)
        self.use_speech = use_speech
        self.use_text = use_text
        self.use_trillsson = use_trillsson
        self.use_w2v2 = use_w2v2
        self.preload_to_ram = preload_to_ram
        self.text_column = text_column

        # (1) Construct required columns
        required_cols = []

        if self.use_speech:
            if self.use_trillsson:
                required_cols.append('trillsson_path')
            if self.use_w2v2:
                required_cols.append('w2v2_path')

        if self.use_text:
            required_cols.append(text_column)

        if len(required_cols) > 0:
            self.data.dropna(subset=required_cols, inplace=True)

            # remove empty strings / 'None'
            for col in required_cols:
                self.data = self.data[self.data[col].apply(lambda x: isinstance(x, str) and x != '')]
                self.data = self.data[self.data[col] != 'None']

        ## remove NaN in label column
        self.data.dropna(subset=['label'], inplace=True)

        # (2) Preload to RAM
        self.preloaded_data = []

        if self.preload_to_ram:
            for i in range(len(self.data)):
                row = self.data.iloc[i]

                trill_emb = None
                w2v2_emb = None
                text_emb = None

                # load audio embeddings
                if self.use_speech:
                    if self.use_trillsson and 'trillsson_path' in row:
                        trill_emb = np.load(row['trillsson_path'])
                        trill_emb = torch.tensor(trill_emb, dtype=torch.float32)
                        trill_emb = self.normalize_embedding(trill_emb)

                    if self.use_w2v2 and 'w2v2_path' in row:
                        w2v2_emb = np.load(row['w2v2_path'])
                        w2v2_emb = torch.tensor(w2v2_emb, dtype=torch.float32)
                        w2v2_emb = self.normalize_embedding(w2v2_emb)

                    # interpolate trill_emb to match w2v2_emb's sequence length
                    if (trill_emb is not None) and (w2v2_emb is not None):
                        T_trill = trill_emb.size(0)
                        T_w2v2 = w2v2_emb.size(0)
                        if T_trill != T_w2v2:
                            trill_emb = trill_emb.unsqueeze(0).transpose(1, 2)
                            trill_emb = F.interpolate(trill_emb, size=T_w2v2, mode='linear')
                            trill_emb = trill_emb.transpose(1, 2).squeeze(0)

                # load text embeddings
                if self.use_text:
                    if text_column in row:
                        text_emb = np.load(row[text_column])
                        text_emb = torch.tensor(text_emb, dtype=torch.float32)
                        text_emb = self.normalize_embedding(text_emb)


                label = torch.tensor(row['label'], dtype=torch.float32)
                self.preloaded_data.append(((trill_emb, w2v2_emb, text_emb), label))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.preload_to_ram:
            return self.preloaded_data[idx]
        
        # if not preloaded, load embeddings on-the-fly
        row = self.data.iloc[idx]

        trill_emb = None
        w2v2_emb = None
        text_emb = None

        if self.use_speech:
            if self.use_trillsson and 'trillsson_path' in row:
                trill_emb = np.load(row['trillsson_path'])
                trill_emb = torch.tensor(trill_emb, dtype=torch.float32)
                trill_emb = self.normalize_embedding(trill_emb)
            if self.use_w2v2 and 'w2v2_path' in row:
                w2v2_emb = np.load(row['w2v2_path'])
                w2v2_emb = torch.tensor(w2v2_emb, dtype=torch.float32)
                w2v2_emb = self.normalize_embedding(w2v2_emb)

            if (trill_emb is not None) and (w2v2_emb is not None):
                T_trill = trill_emb.size(0)
                T_w2v2 = w2v2_emb.size(0)
                if T_trill != T_w2v2:
                    trill_emb = trill_emb.unsqueeze(0).transpose(1, 2)
                    trill_emb = F.interpolate(trill_emb, size=T_w2v2, mode='linear')
                    trill_emb = trill_emb.transpose(1, 2).squeeze(0)

        if self.use_text:
            if self.text_column in row:
                text_emb = np.load(row[self.text_column])
                text_emb = torch.tensor(text_emb, dtype=torch.float32)
                text_emb = self.normalize_embedding(text_emb)

        label = torch.tensor(row['label'], dtype=torch.float32)

        return ((trill_emb, w2v2_emb, text_emb), label)
    
    @staticmethod
    def normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
        """
        normalize (T, D) shaped embeddings along the time axis
        to avoid NaN, use unbiased=False and clamp
        """
        if embedding.dim() == 2:
            T, D = embedding.size()
            if T == 0:
                return embedding
            mean = embedding.mean(dim=0, keepdim=True) # (T, D) -> (1, D)
            std = embedding.std(dim=0, keepdim=True, unbiased=False)
            std = torch.clamp(std, min=1e-9)
            embedding = (embedding - mean) / std
        return embedding