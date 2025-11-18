from typing import Dict, Optional, Tuple

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


def _ste_ternary(logits: torch.Tensor, band: float, temperature: float) -> torch.Tensor:
    """
    Straight-through ternary quantizer mapping logits to {-1, 0, +1}.

    Args:
        logits: Arbitrary tensor of logits.
        band: Non-negative margin defining the neutral zone around zero.
        temperature: Positive temperature controlling the slope of the soft surrogate.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive for STE ternary quantization.")
    soft = torch.tanh(logits / temperature)
    if band < 0:
        raise ValueError("band must be non-negative.")
    with torch.no_grad():
        hard = torch.zeros_like(logits)
        if band == 0:
            hard = torch.sign(logits)
        else:
            hard = torch.where(logits > band, torch.ones_like(logits), hard)
            hard = torch.where(logits < -band, -torch.ones_like(logits), hard)
    return hard + (soft - soft.detach())


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
        use_bitpack: bool = True,
        bitpack_threshold: float = 0.5,
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
        self.use_bitpack = use_bitpack
        self.bitpack_threshold = bitpack_threshold

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

        sig_pos = torch.sigmoid(self.ta_pos)
        sig_neg = torch.sigmoid(self.ta_neg)
        sig_pos_inv = torch.sigmoid(self.ta_pos_inv)
        sig_neg_inv = torch.sigmoid(self.ta_neg_inv)

        hard_pos_zero = sig_pos >= self.tau
        hard_neg_zero = sig_neg >= self.tau
        hard_pos_one = sig_pos_inv >= self.tau
        hard_neg_one = sig_neg_inv >= self.tau

        if use_ste:
            p_pos = self._ste_binary(self.ta_pos, self.tau)
            p_neg = self._ste_binary(self.ta_neg, self.tau)
            p_pos_inv = self._ste_binary(self.ta_pos_inv, self.tau)
            p_neg_inv = self._ste_binary(self.ta_neg_inv, self.tau)
        else:
            p_pos = sig_pos
            p_neg = sig_neg
            p_pos_inv = sig_pos_inv
            p_neg_inv = sig_neg_inv

        p_pos = p_pos.clamp(0.0, 1.0)
        p_neg = p_neg.clamp(0.0, 1.0)
        p_pos_inv = p_pos_inv.clamp(0.0, 1.0)
        p_neg_inv = p_neg_inv.clamp(0.0, 1.0)

        x_neg = 1.0 - x

        # Approximate product t-norm with exponential of summed penalties for speed and stability.
        scale = 4.0 / self.n_features
        pos_score = scale * (torch.matmul(x_neg, p_pos.t()) + torch.matmul(x, p_pos_inv.t()))  # [B, half]
        neg_score = scale * (torch.matmul(x_neg, p_neg.t()) + torch.matmul(x, p_neg_inv.t()))  # [B, half]

        pos_soft = torch.exp(-torch.clamp(pos_score, min=0.0, max=10.0))
        neg_soft = torch.exp(-torch.clamp(neg_score, min=0.0, max=10.0))

        pos_prod = pos_soft
        neg_prod = neg_soft

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
        return x + (clamped - x).detach()

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



class FuzzyPatternTM_STCM(nn.Module):
    """
    Setun–Ternary Clause Machine (STCM).

    Each clause bank uses a single ternary mask per feature, mapping logits to {-1, 0, +1}
    via a straight-through estimator. Positive values require x_i=1, negative values
    require x_i=0, and zeros ignore the feature. Two clause banks (positive / negative)
    share this compact representation while supporting both capacity−mismatch and
    product-style fuzzy operators.
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
        clause_dropout: float = 0.0,
        literal_dropout: float = 0.0,
        clause_bias_init: float = 0.0,
        operator: str = "capacity",
        ternary_voting: bool = False,
        ternary_band: float = 0.0,
        ste_temperature: float = 1.0,
    ):
        super().__init__()
        if n_clauses <= 0 or n_features <= 0:
            raise ValueError("n_features and n_clauses must both be positive.")
        if n_clauses % 2 != 0:
            raise ValueError("n_clauses must be even so clause banks split evenly.")
        self.n_features = n_features
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.tau = tau
        self.input_shape = tuple(input_shape) if input_shape is not None else None
        self.auto_expand_grayscale = auto_expand_grayscale
        self.allow_channel_reduce = allow_channel_reduce
        self.lf = lf
        self.literal_budget = literal_budget
        self.vote_clamp = vote_clamp
        self.clause_dropout = clause_dropout
        self.literal_dropout = literal_dropout
        self.operator = operator
        if operator not in {"capacity", "product"}:
            raise ValueError("operator must be either 'capacity' or 'product'.")
        self.ternary_voting = ternary_voting
        self.ternary_band = ternary_band
        self.ste_temperature = ste_temperature
        self.product_scale = 4.0 / self.n_features

        if self.input_shape is not None:
            product = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
            if product != self.n_features:
                raise ValueError(
                    "input_shape product does not equal n_features: "
                    f"{product} vs {self.n_features}."
                )

        self.clause_bias = nn.Parameter(
            torch.full((n_clauses,), clause_bias_init, dtype=torch.float32)
        )

        half = n_clauses // 2
        if half == 0:
            raise ValueError("n_clauses must be at least 2 for STCM.")

        self.pos_logits = nn.Parameter(torch.randn(half, n_features) * 0.05)
        self.neg_logits = nn.Parameter(torch.randn(half, n_features) * 0.05)

        if ternary_voting:
            self.vote_logits = nn.Parameter(torch.randn(n_clauses, n_classes) * 0.1)
            self.voting = None
        else:
            self.voting = nn.Parameter(torch.randn(n_clauses, n_classes) * 0.1)
            self.vote_logits = None

    def _mask_from_logits(self, logits: torch.Tensor, use_ste: bool) -> torch.Tensor:
        if use_ste:
            return _ste_ternary(logits, self.ternary_band, self.ste_temperature)
        return torch.tanh(logits / self.ste_temperature)

    def _split_masks(self, mask: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        temp = max(self.ste_temperature, 1e-6)
        soft_pos = torch.sigmoid(logits / temp)
        hard_pos = torch.clamp(mask, min=0.0)
        pos = hard_pos + (soft_pos - soft_pos.detach())

        soft_inv = torch.sigmoid(-logits / temp)
        hard_inv = torch.clamp(-mask, min=0.0)
        inv = hard_inv + (soft_inv - soft_inv.detach())
        return pos, inv

    def _capacity_strength(self, x: torch.Tensor, mask_pos: torch.Tensor, mask_inv: torch.Tensor) -> torch.Tensor:
        capacity = self._clause_capacity(mask_pos, mask_inv)
        mismatch = F.linear(1.0 - x, mask_pos) + F.linear(x, mask_inv)
        raw = capacity - mismatch
        return self._straight_relu(raw)

    def _product_strength(self, x: torch.Tensor, mask_pos: torch.Tensor, mask_inv: torch.Tensor) -> torch.Tensor:
        penalties = F.linear(1.0 - x, mask_pos) + F.linear(x, mask_inv)
        scaled = torch.clamp(penalties * self.product_scale, min=0.0, max=10.0)
        return torch.exp(-scaled)

    def _strength(self, x: torch.Tensor, mask_pos: torch.Tensor, mask_inv: torch.Tensor) -> torch.Tensor:
        mask_pos = mask_pos.to(x.dtype)
        mask_inv = mask_inv.to(x.dtype)
        if self.operator == "capacity":
            return self._capacity_strength(x, mask_pos, mask_inv)
        return self._product_strength(x, mask_pos, mask_inv)

    def _clause_outputs(self, x: torch.Tensor, use_ste: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_pos = self._mask_from_logits(self.pos_logits, use_ste)
        mask_neg = self._mask_from_logits(self.neg_logits, use_ste)
        pos_pos, pos_inv = self._split_masks(mask_pos, self.pos_logits)
        neg_pos, neg_inv = self._split_masks(mask_neg, self.neg_logits)
        pos_pos, pos_inv = self._enforce_literal_budget(pos_pos, pos_inv)
        neg_pos, neg_inv = self._enforce_literal_budget(neg_pos, neg_inv)

        pos_strength = self._strength(x, pos_pos, pos_inv)
        neg_strength = self._strength(x, neg_pos, neg_inv)
        clause_votes = torch.cat([pos_strength, -neg_strength], dim=1)

        if self.vote_clamp is not None:
            clause_votes = clause_votes.clamp(-self.vote_clamp, self.vote_clamp)
        if self.training and self.clause_dropout > 0.0:
            clause_votes = F.dropout(clause_votes, p=self.clause_dropout, training=True)
        return pos_strength, neg_strength, clause_votes

    def _clause_capacity(self, mask_pos: torch.Tensor, mask_inv: torch.Tensor) -> torch.Tensor:
        included = mask_pos.sum(dim=1) + mask_inv.sum(dim=1)
        included = self._apply_literal_constraints(included)
        return included.unsqueeze(0)

    @staticmethod
    def _straight_relu(x: torch.Tensor) -> torch.Tensor:
        clamped = torch.relu(x)
        return x + (clamped - x).detach()

    def _apply_literal_constraints(self, included: torch.Tensor) -> torch.Tensor:
        if self.literal_budget is not None:
            limit = torch.as_tensor(float(self.literal_budget), device=included.device)
            included = torch.minimum(included, limit)
        if self.lf > 0:
            limit = torch.as_tensor(float(self.lf), device=included.device)
            included = torch.minimum(included, limit)
        return included

    def _literal_limit_value(self) -> Optional[float]:
        if self.literal_budget is not None:
            return float(self.literal_budget)
        if self.lf > 0:
            return float(self.lf)
        return None

    def _enforce_literal_budget(self, mask_pos: torch.Tensor, mask_inv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        limit = self._literal_limit_value()
        if limit is None:
            return mask_pos, mask_inv
        total = mask_pos.sum(dim=1, keepdim=True) + mask_inv.sum(dim=1, keepdim=True)
        target = torch.full_like(total, limit)
        denom = torch.clamp(total, min=1e-6)
        scale = torch.minimum(torch.ones_like(total), target / denom)
        mask_pos = mask_pos * scale
        mask_inv = mask_inv * scale
        return mask_pos, mask_inv

    def forward(self, x: torch.Tensor, use_ste: bool = True):
        """
        Args:
            x: Tensor in [0,1] with shape [batch, features] or image tensor convertible
               via prepare_tm_input.
            use_ste: Whether to use STE-based ternary masks (default True). If False,
                     hard ternary masks (no gradient) are used.
        """
        flat_x = prepare_tm_input(
            x,
            n_features=self.n_features,
            input_shape=self.input_shape,
            auto_expand_grayscale=self.auto_expand_grayscale,
            allow_channel_reduce=self.allow_channel_reduce,
        )
        if self.training and self.literal_dropout > 0.0:
            flat_x = F.dropout(flat_x, p=self.literal_dropout, training=True)

        _, _, clause_outputs = self._clause_outputs(flat_x, use_ste)
        biased = clause_outputs + self.clause_bias.view(1, -1)
        voting = self._voting_matrix(use_ste)
        logits = biased @ voting
        return logits, clause_outputs

    def _voting_matrix(self, use_ste: bool) -> torch.Tensor:
        if self.ternary_voting:
            assert self.vote_logits is not None
            return _ste_ternary(self.vote_logits, self.ternary_band, self.ste_temperature)
        assert self.voting is not None
        return self.voting

    def extra_repr(self) -> str:
        return (
            f"features={self.n_features}, clauses={self.n_clauses}, classes={self.n_classes}, "
            f"operator='{self.operator}', ternary_voting={self.ternary_voting}"
        )

    @torch.no_grad()
    def discretize(self, threshold: float = 0.0):
        def hard_mask(logits: torch.Tensor) -> torch.Tensor:
            thr = threshold if threshold > 0 else self.ternary_band
            hard = torch.zeros_like(logits)
            hard = torch.where(logits >= thr, torch.ones_like(logits), hard)
            hard = torch.where(logits <= -thr, -torch.ones_like(logits), hard)
            return hard

        pos_mask = hard_mask(self.pos_logits)
        neg_mask = hard_mask(self.neg_logits)

        def to_lists(mask: torch.Tensor, predicate) -> list:
            out = []
            for row in predicate(mask).cpu():
                idxs = (row.nonzero(as_tuple=False).view(-1) + 1).tolist()
                out.append(idxs)
            return out

        pos_required = to_lists(pos_mask, lambda t: t > 0)
        pos_inverse = to_lists(pos_mask, lambda t: t < 0)
        neg_required = to_lists(neg_mask, lambda t: t > 0)
        neg_inverse = to_lists(neg_mask, lambda t: t < 0)

        return {
            "positive": pos_required,
            "positive_inv": pos_inverse,
            "negative": neg_required,
            "negative_inv": neg_inverse,
            "clauses_num": self.n_clauses,
            "operator": self.operator,
            "ternary_voting": self.ternary_voting,
        }

