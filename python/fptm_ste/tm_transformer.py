import torch
import torch.nn as nn
import torch.nn.functional as F
from .tm import FuzzyPatternTM_STE
from .deep_tm import DeepTMNetwork


class TM_TransformerBlock(nn.Module):
    """
    Transformer block where the FFN is replaced with a TM-based layer.
    Uses standard MultiheadAttention (not Swin windows) for simplicity.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        n_clauses: int = 200,
        tm_tau: float = 0.5,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # TM replaces FFN
        self.tm = FuzzyPatternTM_STE(d_model, n_clauses, d_model, tau=tm_tau)
        self.gate = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x: torch.Tensor, attn_mask=None, use_ste: bool = True):
        # Self-attention
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # TM feedforward
        B, T, C = x.shape
        flat = torch.sigmoid(x).reshape(B * T, C)
        logits, _ = self.tm(flat, use_ste=use_ste)  # [B*T, C]
        logits = logits.view(B, T, C)
        gate = torch.sigmoid(self.gate)
        x = self.norm2(x + self.dropout(gate * logits + (1 - gate) * torch.sigmoid(logits)))
        return x


class DeepTMTransformerBlock(nn.Module):
    """Transformer block that swaps the FFN with a DeepTMNetwork stack."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        n_clauses: int = 200,
        tm_tau: float = 0.5,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        hidden_dims = [dim_feedforward, d_model]
        self.deep_tm = DeepTMNetwork(
            input_dim=d_model,
            hidden_dims=hidden_dims,
            n_classes=d_model,
            n_clauses=n_clauses,
            dropout=dropout,
            tau=tm_tau,
            input_shape=None,
            auto_expand_grayscale=False,
            allow_channel_reduce=False,
        )
        self.gate = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x: torch.Tensor, attn_mask=None, use_ste: bool = True):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))

        B, T, C = x.shape
        flat = torch.sigmoid(x).reshape(B * T, C)
        logits, _ = self.deep_tm(flat, use_ste=use_ste)
        logits = logits.view(B, T, C)
        gate = torch.sigmoid(self.gate)
        x = self.norm2(x + self.dropout(gate * logits + (1 - gate) * torch.sigmoid(logits)))
        return x


class UnifiedTMTransformer(nn.Module):
    """
    Simple encoder-only transformer built from TM_TransformerBlock blocks.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 4,
        max_len: int = 512,
        n_clauses: int = 200,
        *,
        backend: str = "ste",
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        tm_tau: float = 0.5,
    ):
        super().__init__()
        backend_key = backend.lower()
        if backend_key == "ste":
            block_cls = TM_TransformerBlock
        elif backend_key in {"deep", "deeptm"}:
            block_cls = DeepTMTransformerBlock
        else:
            raise ValueError(f"Unsupported transformer TM backend '{backend}'. Use 'ste' or 'deeptm'.")

        self.backend = backend_key
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [
                block_cls(
                    d_model,
                    n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    n_clauses=n_clauses,
                    tm_tau=tm_tau,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, use_ste: bool = True):
        # x: [B, T]
        B, T = x.shape
        h = self.embed(x) + self.pos[:, :T, :]
        for layer in self.layers:
            h = layer(h, use_ste=use_ste)
        h = self.norm(h)
        return self.head(h)


