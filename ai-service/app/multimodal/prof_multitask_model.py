import torch
import torch.nn as nn
import torch.nn.functional as F

class ProfMultitaskMultimodalNet(nn.Module):
    def __init__(self, head_dims: dict[str, int], dropout: float = 0.3):
        super().__init__()
        self.embed_dim = 384
        self.table_dim = 64
        self.modalities = 4

        self.table_proj = nn.Linear(self.table_dim, self.embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=4,
            batch_first=True,
        )
        self.attn_ln = nn.LayerNorm(self.embed_dim)

        self.gate = nn.Sequential(
            nn.Linear(self.modalities * self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.modalities),
        )

        fusion_dim = (self.modalities * self.embed_dim) + self.table_dim  # 4*384 + 64 = 1600
        self.backbone = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.heads = nn.ModuleDict({
            name: nn.Linear(256, n_classes) for name, n_classes in head_dims.items()
        })

    def forward(self, text_emb, ocr_emb, audio_emb, table_emb, mask):
        B = text_emb.size(0)

        table_tok = self.table_proj(table_emb)  # [B,384]
        tokens = torch.stack([text_emb, ocr_emb, audio_emb, table_tok], dim=1)  # [B,4,384]

        key_padding_mask = ~mask  # True=ignore
        attn_out, _ = self.attn(tokens, tokens, tokens, key_padding_mask=key_padding_mask)
        attn_out = self.attn_ln(attn_out + tokens)

        pooled = attn_out.reshape(B, -1)
        gate_logits = self.gate(pooled).masked_fill(~mask, -1e9)
        gate_w = F.softmax(gate_logits, dim=1)

        gated = attn_out * gate_w.unsqueeze(-1)
        gated_flat = gated.reshape(B, -1)

        fusion_vec = torch.cat([gated_flat, table_emb], dim=1)
        h = self.backbone(fusion_vec)

        out = {name: head(h) for name, head in self.heads.items()}
        return out