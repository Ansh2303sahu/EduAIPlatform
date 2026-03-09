
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalFusionNet(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.embed_dim = 384
        self.table_dim = 64
        self.modalities = 4  # text, ocr, audio, table

        # project table 64 -> 384 for attention
        self.table_proj = nn.Linear(self.table_dim, self.embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=4,
            batch_first=True,
        )
        self.attn_ln = nn.LayerNorm(self.embed_dim)

        # gating network -> per-sample modality weights
        self.gate = nn.Sequential(
            nn.Linear(self.modalities * self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.modalities),
        )

        # attended 4*384 + raw table64
        fusion_dim = (self.modalities * self.embed_dim) + self.table_dim  # 1536 + 64 = 1600
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, text_emb, ocr_emb, audio_emb, table_emb, mask):
        """
        text/ocr/audio: [B,384]
        table: [B,64]
        mask: [B,4] bool True=present
        """
        B = text_emb.size(0)

        table_tok = self.table_proj(table_emb)  # [B,384]
        tokens = torch.stack([text_emb, ocr_emb, audio_emb, table_tok], dim=1)  # [B,4,384]

        # key_padding_mask: True means "ignore"
        key_padding_mask = ~mask

        attn_out, _ = self.attn(tokens, tokens, tokens, key_padding_mask=key_padding_mask)
        attn_out = self.attn_ln(attn_out + tokens)  # residual + LN

        pooled = attn_out.reshape(B, -1)            # [B,4*384]
        # gating (compute in fp32 to avoid fp16 overflow issues)
        gate_logits = self.gate(pooled).float()         # [B,4] fp32
        neg_inf = torch.finfo(gate_logits.dtype).min
        gate_logits = gate_logits.masked_fill((~mask).bool(), neg_inf)

        gate_w = torch.softmax(gate_logits, dim=1).to(attn_out.dtype)  # back to fp16/fp32 as needed

        gated = attn_out * gate_w.unsqueeze(-1)     # [B,4,384]
        gated_flat = gated.reshape(B, -1)           # [B,1536]

        fusion_vec = torch.cat([gated_flat, table_emb], dim=1)  # [B,1600]
        return self.classifier(fusion_vec)
