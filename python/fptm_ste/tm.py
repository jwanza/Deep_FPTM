from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _adjust_channels_batch(
    x: torch.Tensor,
    expected_channels: int,
    auto_expand_grayscale: bool,
    allow_channel_reduce: bool,
) -> torch.Tensor:
    actual_channels = x.shape[1]
    if actual_channels == expected_channels:
        return x
    if actual_channels == 1 and expected_channels > 1 and auto_expand_grayscale:
        return x.repeat(1, expected_channels, 1, 1)
    if actual_channels > 1 and expected_channels == 1 and allow_channel_reduce:
        return x.mean(dim=1, keepdim=True)
    raise ValueError(
        "Cannot adjust channel count from "
        f"{actual_channels} to {expected_channels}. Enable auto expansion for grayscale"
        " or allow channel reduction if appropriate."
    )


def _resize_spatial(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    current_h, current_w = x.shape[2:]
    if current_h == height and current_w == width:
        return x
    mode = "bilinear" if x.shape[1] > 1 else "nearest"
    return F.interpolate(x, size=(height, width), mode=mode, align_corners=False)


def prepare_tm_input(
    x: torch.Tensor,
    *,
    n_features: int,
    input_shape: Optional[Tuple[int, int, int]] = None,
    auto_expand_grayscale: bool = False,
    allow_channel_reduce: bool = True,
) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected input tensor, received {type(x).__name__}.")

    if input_shape is None:
        if x.dim() == 2:
            if x.shape[1] != n_features:
                raise ValueError(
                    f"Expected feature dimension {n_features}, got {x.shape[1]}."
                )
            return x
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() == 4:
            flat = x.reshape(x.shape[0], -1)
            if flat.shape[1] != n_features:
                raise ValueError(
                    "Input tensor does not match the expected flattened dimension "
                    f"({flat.shape[1]} vs {n_features})."
                )
            return flat
        if x.dim() == 1:
            if x.numel() != n_features:
                raise ValueError(
                    f"Expected {n_features} features, received tensor with {x.numel()} elements."
                )
            return x.unsqueeze(0)
        raise ValueError(
            "Unsupported input shape. Provide a flattened tensor or specify input_shape for image data."
        )

    expected_c, expected_h, expected_w = input_shape
    total_expected = expected_c * expected_h * expected_w
    if total_expected != n_features:
        raise ValueError(
            "input_shape product does not match n_features: "
            f"{total_expected} (from {input_shape}) vs {n_features}."
        )

    original_dim = x.dim()
    if original_dim == 1:
        x = x.unsqueeze(0)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4 and x.dim() != 2:
        raise ValueError(
            "Unsupported input tensor shape. Expected flattenable tensor or 3/4D image tensor."
        )

    if x.dim() == 2:
        if x.shape[1] != n_features:
            raise ValueError(
                f"Expected flattened tensor with {n_features} features, got {x.shape[1]}."
            )
        return x

    x = _adjust_channels_batch(x, expected_c, auto_expand_grayscale, allow_channel_reduce)
    x = _resize_spatial(x, expected_h, expected_w)
    flat = x.reshape(x.shape[0], -1)
    if flat.shape[1] != n_features:
        raise ValueError(
            "Flattened tensor does not match expected feature dimension "
            f"({flat.shape[1]} vs {n_features})."
        )
    return flat


class FuzzyPatternTM_STE(nn.Module):
    """
    Differentiable Fuzzy-Pattern TM with Straight-Through Estimator (STE).

    - ta_include_pos:  [clauses_half, features]
    - ta_include_neg:  [clauses_half, features]
    - ta_include_pos_inv: [clauses_half, features]
    - ta_include_neg_inv: [clauses_half, features]
    """

    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        tau: float = 0.5,
        *,
        input_shape: Optional[Tuple[int, int, int]] = None,
        auto_expand_grayscale: bool = False,
        allow_channel_reduce: bool = True,
        clause_dropout: float = 0.0,
        literal_dropout: float = 0.0,
        clause_bias_init: float = 0.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.tau = tau
        self.input_shape = tuple(input_shape) if input_shape is not None else None
        self.auto_expand_grayscale = auto_expand_grayscale
        self.allow_channel_reduce = allow_channel_reduce
        self.clause_dropout = clause_dropout
        self.literal_dropout = literal_dropout
        self.clause_bias = nn.Parameter(torch.full((n_clauses,), clause_bias_init, dtype=torch.float32))
        if self.input_shape is not None:
            product = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
            if product != self.n_features:
                raise ValueError(
                    "input_shape product does not equal n_features: "
                    f"{product} vs {self.n_features}."
                )

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

    def _clause_products(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
        *,
        already_flat: bool = False,
    ):
        if not already_flat:
            x = prepare_tm_input(
                x,
                n_features=self.n_features,
                input_shape=self.input_shape,
                auto_expand_grayscale=self.auto_expand_grayscale,
                allow_channel_reduce=self.allow_channel_reduce,
            )
        B, F_ = x.shape
        assert F_ == self.n_features

        if self.training and self.literal_dropout > 0.0:
            x = F.dropout(x, p=self.literal_dropout, training=True)

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
        if self.training and self.clause_dropout > 0.0:
            clause_outputs = F.dropout(clause_outputs, p=self.clause_dropout, training=True)
        return pos_prod, neg_prod, clause_outputs

    def forward(self, x: torch.Tensor, use_ste: bool = True):
        """
        x: [batch, features] in [0,1] or image tensor convertible to flattened form.
        Returns:
            logits: [batch, n_classes]
            clause_outputs: [batch, n_clauses]
        """
        flat_x = prepare_tm_input(
            x,
            n_features=self.n_features,
            input_shape=self.input_shape,
            auto_expand_grayscale=self.auto_expand_grayscale,
            allow_channel_reduce=self.allow_channel_reduce,
        )
        _, _, clause_outputs = self._clause_products(flat_x, use_ste=use_ste, already_flat=True)
        logits = (clause_outputs + self.clause_bias.view(1, -1)) @ self.voting
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
        *,
        input_shape: Optional[Tuple[int, int, int]] = None,
        auto_expand_grayscale: bool = False,
        allow_channel_reduce: bool = True,
        lf: int = 4,
        literal_budget: Optional[int] = None,
        vote_clamp: Optional[float] = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.tau = tau
        self.input_shape = tuple(input_shape) if input_shape is not None else None
        self.auto_expand_grayscale = auto_expand_grayscale
        self.allow_channel_reduce = allow_channel_reduce
        if self.input_shape is not None:
            product = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
            if product != self.n_features:
                raise ValueError(
                    "input_shape product does not equal n_features: "
                    f"{product} vs {self.n_features}."
                )
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
        x = prepare_tm_input(
            x,
            n_features=self.n_features,
            input_shape=self.input_shape,
            auto_expand_grayscale=self.auto_expand_grayscale,
            allow_channel_reduce=self.allow_channel_reduce,
        )
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


