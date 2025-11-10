import torch
import torch.nn as nn
import torch.nn.functional as F


class TM_Attention_Oracle(nn.Module):
    """
    Attention over TM clauses and scale logits to boost accuracy.
    """
    def __init__(self, n_scales: int, n_classes: int, d_model: int = 256, n_heads: int = 4):
        super().__init__()
        self.n_scales = n_scales
        self.n_classes = n_classes

        self.scale_attn = nn.MultiheadAttention(embed_dim=n_classes, num_heads=n_heads, batch_first=True)
        self.clause_proj = nn.Linear(1, n_classes)  # project clause outputs per clause to class space
        self.clause_attn = nn.MultiheadAttention(embed_dim=n_classes, num_heads=n_heads, batch_first=True)

        self.gate = nn.Parameter(torch.tensor([0.5]))

    def forward(self, scale_logits_list, clause_outputs_list):
        """
        scale_logits_list: list of [B, C] per scale
        clause_outputs_list: list of [B, n_clauses] per scale
        """
        B = scale_logits_list[0].size(0)
        S = len(scale_logits_list)

        scales = torch.stack(scale_logits_list, dim=1)  # [B, S, C]
        # Self-attend scales
        scale_enh, _ = self.scale_attn(scales, scales, scales)

        # Concatenate clause outputs across scales and attend
        clause_feats = []
        for clauses in clause_outputs_list:
            # [B, n_clauses] -> [B, n_clauses, 1] -> [B, n_clauses, C]
            clause_feats.append(self.clause_proj(clauses.unsqueeze(-1)))
        clause_cat = torch.cat(clause_feats, dim=1)  # [B, sum(n_clauses), C]
        clause_enh, _ = self.clause_attn(clause_cat, clause_cat, clause_cat)
        clause_pool = clause_enh.mean(dim=1)  # [B, C]

        gate = torch.sigmoid(self.gate)
        fused = gate * scale_enh.mean(dim=1) + (1 - gate) * clause_pool
        return fused  # [B, C]


