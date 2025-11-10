from typing import Optional

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



class FuzzyPatternTMFPTM(nn.Module):
    """Julia-style FPTM with STE gradients.

    Mimics the LF-based clause voting used in `src/FuzzyPatternTM.jl` while
    keeping the differentiable straight-through masks from the STE variant.
    """

    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        tau: float = 0.5,
        lf: int = 4,
        literal_budget: Optional[int] = None,
        vote_clamp: Optional[float] = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.tau = tau
        self.lf = lf
        self.literal_budget = literal_budget
        self.vote_clamp = vote_clamp

        half = n_clauses // 2
        self.ta_pos = nn.Parameter(torch.randn(half, n_features) * 0.05)
        self.ta_neg = nn.Parameter(torch.randn(half, n_features) * 0.05)
        self.ta_pos_inv = nn.Parameter(torch.randn(half, n_features) * 0.05)
        self.ta_neg_inv = nn.Parameter(torch.randn(half, n_features) * 0.05)
        # Votes are ±1 style; keep them learnable but small.
        self.voting = nn.Parameter(torch.randn(n_clauses, n_classes) * 0.05)

    @staticmethod
    def _ste_mask(p: torch.Tensor, tau: float, use_ste: bool) -> torch.Tensor:
        probs = torch.sigmoid(p)
        hard = (probs >= tau).float()
        if not use_ste:
            return hard
        return hard + (probs - hard).detach()

    def _clause_capacity(self, mask_pos: torch.Tensor, mask_inv: torch.Tensor) -> torch.Tensor:
        included = mask_pos.sum(dim=1) + mask_inv.sum(dim=1)
        if self.literal_budget is not None:
            included = torch.minimum(included, torch.as_tensor(float(self.literal_budget), device=included.device))
        if self.lf > 0:
            included = torch.minimum(included, torch.as_tensor(float(self.lf), device=included.device))
        return included.unsqueeze(0)  # [1, half]

    @staticmethod
    def _straight_relu(x: torch.Tensor) -> torch.Tensor:
        clamped = torch.relu(x)
        return clamped + (x - clamped).detach()

    def _strength(self, x: torch.Tensor, mask_pos: torch.Tensor, mask_inv: torch.Tensor) -> torch.Tensor:
        capacity = self._clause_capacity(mask_pos, mask_inv)
        mismatch = F.linear(1.0 - x, mask_pos) + F.linear(x, mask_inv)
        raw = capacity - mismatch
        return self._straight_relu(raw)

    def forward(self, x: torch.Tensor, use_ste: bool = True):
        mask_pos = self._ste_mask(self.ta_pos, self.tau, use_ste)
        mask_neg = self._ste_mask(self.ta_neg, self.tau, use_ste)
        mask_pos_inv = self._ste_mask(self.ta_pos_inv, self.tau, use_ste)
        mask_neg_inv = self._ste_mask(self.ta_neg_inv, self.tau, use_ste)

        pos_strength = self._strength(x, mask_pos, mask_pos_inv)
        neg_strength = self._strength(x, mask_neg, mask_neg_inv)

        clause_votes = torch.cat([pos_strength, -neg_strength], dim=1)
        if self.vote_clamp is not None:
            clause_votes = clause_votes.clamp(-self.vote_clamp, self.vote_clamp)
        logits = clause_votes @ self.voting
        return logits, clause_votes

    @torch.no_grad()
    def discretize(self, threshold: float = 0.5):
        def mask(p):
            return (torch.sigmoid(p) >= threshold)

        pos = mask(self.ta_pos).cpu().numpy()
        neg = mask(self.ta_neg).cpu().numpy()
        pos_inv = mask(self.ta_pos_inv).cpu().numpy()
        neg_inv = mask(self.ta_neg_inv).cpu().numpy()

        def to_lists(arr):
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
            "clauses_num": self.n_clauses,
            "lf": self.lf,
            "literal_budget": self.literal_budget,
        }


