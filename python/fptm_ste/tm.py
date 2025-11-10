import torch
import torch.nn as nn
import torch.nn.functional as F


class FuzzyPatternTM_STE(nn.Module):
    """
    Differentiable Fuzzy-Pattern TM with Straight-Through Estimator (STE).

    - ta_include_pos:  [clauses_half, features]
    - ta_include_neg:  [clauses_half, features]
    - ta_include_pos_inv: [clauses_half, features]
    - ta_include_neg_inv: [clauses_half, features]
    """

    def __init__(self, n_features: int, n_clauses: int, n_classes: int, tau: float = 0.5):
        super().__init__()
        self.n_features = n_features
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.tau = tau

        half = n_clauses // 2
        # Parameters as logits -> probabilities via sigmoid
        self.ta_pos = nn.Parameter(torch.randn(half, n_features) * 0.05)
        self.ta_neg = nn.Parameter(torch.randn(half, n_features) * 0.05)
        self.ta_pos_inv = nn.Parameter(torch.randn(half, n_features) * 0.05)
        self.ta_neg_inv = nn.Parameter(torch.randn(half, n_features) * 0.05)

        # Voting weights per clause per class
        self.voting = nn.Parameter(torch.randn(n_clauses, n_classes) * 0.1)

    @staticmethod
    def _ste_binary(p: torch.Tensor, tau: float) -> torch.Tensor:
        with torch.no_grad():
            hard = (torch.sigmoid(p) > tau).float()
        # Straight-through
        return hard + (torch.sigmoid(p) - hard).detach()

    def _clause_products(self, x: torch.Tensor, use_ste: bool = True):
        """
        x: [batch, features] in [0,1]
        Returns:
            pos_prod: [batch, clauses_half]
            neg_prod: [batch, clauses_half]
            clause_outputs: [batch, n_clauses]
        """
        B, F_ = x.shape
        assert F_ == self.n_features

        if use_ste:
            p_pos = self._ste_binary(self.ta_pos, self.tau)
            p_neg = self._ste_binary(self.ta_neg, self.tau)
            p_pos_inv = self._ste_binary(self.ta_pos_inv, self.tau)
            p_neg_inv = self._ste_binary(self.ta_neg_inv, self.tau)
        else:
            p_pos = torch.sigmoid(self.ta_pos)
            p_neg = torch.sigmoid(self.ta_neg)
            p_pos_inv = torch.sigmoid(self.ta_pos_inv)
            p_neg_inv = torch.sigmoid(self.ta_neg_inv)

        p_pos = p_pos.clamp(0.0, 1.0)
        p_neg = p_neg.clamp(0.0, 1.0)
        p_pos_inv = p_pos_inv.clamp(0.0, 1.0)
        p_neg_inv = p_neg_inv.clamp(0.0, 1.0)

        x_neg = 1.0 - x

        # Approximate product t-norm with exponential of summed penalties for speed and stability.
        scale = 4.0 / self.n_features
        pos_score = scale * (torch.matmul(x_neg, p_pos.t()) + torch.matmul(x, p_pos_inv.t()))  # [B, half]
        neg_score = scale * (torch.matmul(x_neg, p_neg.t()) + torch.matmul(x, p_neg_inv.t()))  # [B, half]

        pos_prod = torch.exp(-torch.clamp(pos_score, min=0.0, max=10.0))
        neg_prod = torch.exp(-torch.clamp(neg_score, min=0.0, max=10.0))

        clause_outputs = torch.cat([pos_prod, neg_prod], dim=1)  # [B, n_clauses]
        return pos_prod, neg_prod, clause_outputs

    def forward(self, x: torch.Tensor, use_ste: bool = True):
        """
        x: [batch, features] in [0,1]
        Returns:
            logits: [batch, n_classes]
            clause_outputs: [batch, n_clauses]
        """
        _, _, clause_outputs = self._clause_products(x, use_ste=use_ste)
        logits = clause_outputs @ self.voting
        return logits, clause_outputs

    @torch.no_grad()
    def discretize(self, threshold: float = 0.5):
        """
        Returns a Python dict with included literals per clause bank, suitable
        for JSON export.
        """
        def mask(p):
            return (torch.sigmoid(p) >= threshold)

        pos = mask(self.ta_pos).cpu().numpy()  # [half, F]
        neg = mask(self.ta_neg).cpu().numpy()
        pos_inv = mask(self.ta_pos_inv).cpu().numpy()
        neg_inv = mask(self.ta_neg_inv).cpu().numpy()

        def to_lists(arr):
            # Convert [half, F] boolean to list-of-indices-per-clause (1-based)
            out = []
            for r in arr:
                idxs = (r.nonzero()[0] + 1).tolist()
                out.append(idxs)
            return out

        return {
            "positive": to_lists(pos),
            "negative": to_lists(neg),
            "positive_inv": to_lists(pos_inv),
            "negative_inv": to_lists(neg_inv),
            "clauses_num": self.n_clauses
        }


