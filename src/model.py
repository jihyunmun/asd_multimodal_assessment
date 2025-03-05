import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################################################
# Define Embedding Extractors
###############################################################################################################
class SpeechEmbeddingExtractor(nn.Module):
    def __init__(self,
                 use_speech=True,
                 use_trillsson=True,
                 use_w2v2=True,
                 input_dim=1024,
                 hidden_dim=512,
                 num_heads=4):
        
        super().__init__()
        self.use_speech = use_speech
        self.use_trillsson = use_trillsson
        self.use_w2v2 = use_w2v2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if self.use_trillsson:
            self.trill_proj = nn.Linear(self.input_dim, self.hidden_dim)
        if self.use_w2v2:
            self.w2v2_proj = nn.Linear(self.input_dim, self.hidden_dim)

        self.mha = nn.MultiheadAttention(
            embed_dim = hidden_dim,
            num_heads = num_heads,
            batch_first = False,
            dropout = 0.1
        )

        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.Dropout(0.1)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, trill_padded, trill_mask, w2v2_padded, w2v2_mask):
        """
        (1) Projection: embed_dim -> hidden_dim
        (2) Cross-attention
        (3) Final speech embedding
        """
        if not self.use_speech:
            return None
        
        if not (self.use_trillsson or self.use_w2v2):
            return None
        
        trill_emb = None
        w2v2_emb = None

        # (1) Projection
        if self.use_trillsson and trill_padded is not None:
            trill_emb = self.trill_proj(trill_padded)

        if self.use_w2v2 and w2v2_padded is not None:
            w2v2_emb = self.w2v2_proj(w2v2_padded)

        # (2) Cross-attention
        if (trill_emb is not None) and (w2v2_emb is not None):
            # w2v2: query, trillsson: key, value
            w2v2_t = w2v2_emb.transpose(0, 1) # (T, B, D)
            trill_t = trill_emb.transpose(0, 1) # (T, B, D)

            attn_out, _ = self.mha(
                query = w2v2_t,
                key = trill_t,
                value = trill_t,
                need_weights = False
            )
            x = self.norm1(attn_out + w2v2_t)
            ff = self.feedforward(x)
            x = self.norm2(ff + x)
            speech_out = x.transpose(0, 1)
            return speech_out
        
        elif trill_emb is not None:
            return trill_emb
        
        elif w2v2_emb is not None:
            return w2v2_emb
        

class TextEmbeddingExtractor(nn.Module):
    def __init__(self, use_text=True, input_dim=768, hidden_dim=512):
        super().__init__()
        self.use_text = use_text
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forwad(self, text_padded):
        if not self.use_text:
            return None
        if text_padded is None:
            return None
        
        text = self.projection(text_padded)
        text = self.norm(text)
        return text
    
###############################################################################################################
# Define Multimodal Assessment Model
###############################################################################################################
class Multimodal(nn.Module):
    def __init__(self,
                 speech_extractor: SpeechEmbeddingExtractor,
                 text_extractor: TextEmbeddingExtractor,
                 embed_dim=512,
                 hidden_dim=512,
                 num_heads=4):
        super().__init__()
        self.speech_extractor = speech_extractor
        self.text_extractor = text_extractor

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # co-attention
        self.multihead_attn_speech = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.multihead_attn_text = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        # LayerNorm
        self.norm_speech_1 = nn.LayerNorm(embed_dim)
        self.norm_speech_2 = nn.LayerNorm(hidden_dim)
        self.norm_text_1 = nn.LayerNorm(embed_dim)
        self.norm_text_2 = nn.LayerNorm(hidden_dim)

        # Feedforward
        self.ffn_speech = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.1)
        )
        self.ffn_text = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.1)
        )

        # Fusion & head
        self.fusion_linear = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self,
                trill_padded, trill_mask,
                w2v2_padded, w2v2_mask,
                text_padded, text_mask):
        """
        (1) Extract embeddings
        (2) Co-attention
        (3) Masked Mean Pooling
        (4) Fusion & Regression
        """

        # (1) Extract embeddings
        speech_emb = self.speech_extractor(trill_padded, trill_mask, w2v2_padded, w2v2_mask) # [B, T, D]

        if self.speech_extractor.use_speech:
            if self.speech_extractor.use_trillsson and self.speech_extractor.use_w2v2:
                speech_mask = w2v2_mask
            elif self.speech_extractor.use_trillsson:
                speech_mask = trill_mask
            elif self.speech_extractor.use_w2v2:
                speech_mask = w2v2_mask
            else:
                speech_mask = None
        else:
            speech_mask = None

        text_emb = self.text_extractor(text_padded) # [B, T, D]

        if speech_emb is not None and speech_mask is None:
            B, T_sp, _ = speech_emb.size()
            speech_mask = torch.zeros((B, T_sp), dtype=torch.bool, device=speech_emb.device)
        
        if text_emb is not None and text_mask is None:
            B, T_tx, _ = text_emb.size()
            text_mask = torch.zeros((B, T_tx), dtype=torch.bool, device=text_emb.device)

        # (2) Co-attention
        ## (a) speech_emb = None, text_emb = None -> return None
        ## (b) speech_emb = None, text_emb != None -> return text_emb after normalization
        ## (c) speech_emb != None, text_emb = None -> return speech_emb after normalization
        ## (d) speech_emb != None, text_emb != None -> return co-attended embeddings
        if speech_emb is None and text_emb is None:
            return None

        elif speech_emb is None:
            text_out = self.norm_text_1(text_emb)
            text_ffn_out = self.ffn_text(text_out)
            text_out = self.norm_text_2(text_out + text_ffn_out)
            speech_out = None

        elif text_emb is None:
            speech_out = self.norm_speech_1(speech_emb)
            speech_ffn_out = self.ffn_speech(speech_out)
            speech_out = self.norm_speech_2(speech_out + speech_ffn_out)
            text_out = None

        else:
            # text-conditioned speech embedding
            speech_from_text, _ = self.multihead_attn_speech(
                speech_emb, text_emb, text_emb,
                key_padding_mask=text_mask
            )
            speech_out = self.norm_speech_1(speech_emb + speech_from_text)
            speech_ffn_out = self.ffn_speech(speech_out)
            speech_out = self.norm_speech_2(speech_out + speech_ffn_out)

            # speech-conditioned text embedding
            text_from_speech, _ = self.multihead_attn_text(
                text_emb, speech_emb, speech_emb,
                key_padding_mask=speech_mask
            )
            text_out = self.norm_text_1(text_emb + text_from_speech)
            text_ffn_out = self.ffn_text(text_out)
            text_out = self.norm_text_2(text_out + text_ffn_out)

        # (3) Masked Mean Pooling
        speech_rep = None
        text_rep = None

        if speech_out is not None:
            valid_speech = (~speech_mask).unsqueeze(-1).float() # [B, T, 1]
            masked_speech = speech_out * valid_speech
            speech_lengths = valid_speech.sum(dim=1, keepdim=True).clamp(min=1e-9) # [B, 1, 1]
            speech_rep = masked_speech.sum(dim=1) / speech_lengths.squeeze(2) # [B, D]

        if text_out is not None:
            valid_text = (~text_mask).unsqueeze(-1).float()
            masked_text = text_out * valid_text
            text_lengths = valid_text.sum(dim=1, keepdim=True).clamp(min=1e-9)
            text_rep = masked_text.sum(dim=1) / text_lengths.squeeze(2)

        # (4) Fusion & Regression
        if speech_rep is not None and text_rep is not None:
            fusion = torch.cat([speech_rep, text_rep], dim=-1) # [B, 2D]
            fusion = self.fusion_linear(fusion) # [B, D]
        elif speech_rep is not None:
            fusion_one = nn.Linear(self.hidden_dim, self.hidden_dim).to(speech_rep.device)
            fusion = F.relu(fusion_one(speech_rep))
        elif text_rep is not None:
            fusion_one = nn.Linear(self.hidden_dim, self.hidden_dim).to(text_rep.device)
            fusion = F.relu(fusion_one(text_rep))
        else:
            return None
        
        y_hat = self.reg_head(fusion)
        return y_hat