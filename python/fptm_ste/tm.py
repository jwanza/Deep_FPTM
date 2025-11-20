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

    def prune(self, threshold: float = 0.1):
        """
        Prunes low-magnitude literal weights to zero.
        """
        with torch.no_grad():
            for param in [self.ta_pos, self.ta_neg, self.ta_pos_inv, self.ta_neg_inv]:
                if 0 < threshold < 1:
                    logit_thresh = torch.log(torch.tensor(threshold / (1 - threshold)))
                    mask = param > logit_thresh
                    param.data = torch.where(mask, param.data, torch.tensor(-10.0, device=param.device))

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
        skip_norm: bool = False,
    ):
        if not skip_norm and not already_flat:
            x = prepare_tm_input(
                x,
                n_features=self.n_features,
                input_shape=self.input_shape,
                auto_expand_grayscale=self.auto_expand_grayscale,
                allow_channel_reduce=self.allow_channel_reduce,
            )
        # Skip assert for speed if skip_norm is True

        if self.training and self.literal_dropout > 0.0:
            x = F.dropout(x, p=self.literal_dropout, training=True)

        # Use cached sigmoids/masks if not training? No, weights change.
        # But we can batch the sigmoid/STE calculation for all 4 parameters.
        # Concatenate weights first: [pos, neg, pos_inv, neg_inv]
        # Shape: [4 * half, n_features]
        
        all_logits = torch.cat([self.ta_pos, self.ta_neg, self.ta_pos_inv, self.ta_neg_inv], dim=0)
        
        if use_ste:
             all_p = self._ste_binary(all_logits, self.tau)
        else:
             all_p = torch.sigmoid(all_logits)
             
        all_p = all_p.clamp(0.0, 1.0)
        
        # Split back
        half = self.n_clauses // 2
        p_pos = all_p[0:half]
        p_neg = all_p[half:2*half]
        p_pos_inv = all_p[2*half:3*half]
        p_neg_inv = all_p[3*half:4*half]
        
        x_neg = 1.0 - x
        X_combined = torch.cat([x_neg, x], dim=1) # [B, 2F]
        
        # W_pos = [p_pos, p_pos_inv] -> [half, 2F] (concatenation along feature dim)
        W_pos = torch.cat([p_pos, p_pos_inv], dim=1)
        W_neg = torch.cat([p_neg, p_neg_inv], dim=1)
        
        # W_total = [W_pos; W_neg] -> [n_clauses, 2F]
        W_total = torch.cat([W_pos, W_neg], dim=0)
        
        scale = 4.0 / self.n_features
        
        # The Big MatMul
        scores = F.linear(X_combined, W_total) * scale
        
        pos_score = scores[:, :half]
        neg_score = scores[:, half:]
        
        pos_soft = torch.exp(-torch.clamp(pos_score, min=0.0, max=10.0))
        neg_soft = torch.exp(-torch.clamp(neg_score, min=0.0, max=10.0))

        pos_prod = pos_soft
        neg_prod = neg_soft

        clause_outputs = torch.cat([pos_prod, neg_prod], dim=1)  # [B, n_clauses]
        if self.training and self.clause_dropout > 0.0:
            clause_outputs = F.dropout(clause_outputs, p=self.clause_dropout, training=True)
        return pos_prod, neg_prod, clause_outputs

    def forward(self, x: torch.Tensor, use_ste: bool = True, skip_norm: bool = False):
        """
        x: [batch, features]
        skip_norm: If True, assumes x is already flattened and normalized to [0,1] with correct dim.
        """
        if not skip_norm:
             flat_x = prepare_tm_input(
                x,
                n_features=self.n_features,
                input_shape=self.input_shape,
                auto_expand_grayscale=self.auto_expand_grayscale,
                allow_channel_reduce=self.allow_channel_reduce,
            )
        else:
             flat_x = x
             
        _, _, clause_outputs = self._clause_products(flat_x, use_ste=use_ste, already_flat=True, skip_norm=True)
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
        T: float = 1.0,
        *,
        input_shape: Optional[Tuple[int, int, int]] = None,
        auto_expand_grayscale: bool = False,
        allow_channel_reduce: bool = True,
        lf: int = 4,
        literal_budget: Optional[int] = None,
        vote_clamp: Optional[float] = None,
        team_per_class: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.tau = tau
        self.T = T
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
        self.team_per_class = team_per_class

        half = n_clauses // 2
        # Initialize to -2.0 to encourage sparsity (prob ~0.12) instead of 0.0 (prob ~0.5)
        self.ta_pos = nn.Parameter(torch.randn(half, n_features) * 0.05 + 0.1)
        self.ta_neg = nn.Parameter(torch.randn(half, n_features) * 0.05 + 0.1)
        self.ta_pos_inv = nn.Parameter(torch.randn(half, n_features) * 0.05 + 0.1)
        self.ta_neg_inv = nn.Parameter(torch.randn(half, n_features) * 0.05 + 0.1)
        
        if self.team_per_class:
            if half % n_classes != 0:
                 raise ValueError(f"Half clauses ({half}) must be divisible by n_classes ({n_classes}) for team split.")
            self.voting = None
        else:
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
        # Use Leaky ReLU with a very small negative slope to allow gradient flow for "dead" clauses
        clamped = F.leaky_relu(x, negative_slope=0.01)
        return x + (clamped - x).detach()

    def _strength(self, x: torch.Tensor, mask_pos: torch.Tensor, mask_inv: torch.Tensor) -> torch.Tensor:
        capacity = self._clause_capacity(mask_pos, mask_inv)
        mismatch = F.linear(1.0 - x, mask_pos) + F.linear(x, mask_inv)
        raw = capacity - mismatch
        return self._straight_relu(raw)

    def forward(self, x: torch.Tensor, use_ste: bool = True, skip_norm: bool = False):
        if not skip_norm:
            x = prepare_tm_input(
                x,
                n_features=self.n_features,
                input_shape=self.input_shape,
                auto_expand_grayscale=self.auto_expand_grayscale,
                allow_channel_reduce=self.allow_channel_reduce,
            )
            
        # Fuse mask calculation
        # [ta_pos; ta_neg; ta_pos_inv; ta_neg_inv] -> [4*half, F]
        all_logits = torch.cat([self.ta_pos, self.ta_neg, self.ta_pos_inv, self.ta_neg_inv], dim=0)
        all_masks = self._ste_mask(all_logits, self.tau, use_ste)
        
        half = self.n_clauses // 2
        mask_pos = all_masks[0:half]
        mask_neg = all_masks[half:2*half]
        mask_pos_inv = all_masks[2*half:3*half]
        mask_neg_inv = all_masks[3*half:4*half]
        
        # Enforce capacity constraints?
        # _strength calls _clause_capacity then _straight_relu(capacity - mismatch)
        # mismatch = (1-x)@mask_pos.t() + x@mask_inv.t()
        
        # We can fuse mismatch calculation!
        # X_combined = [1-x, x] -> [B, 2F]
        # W_pos = [mask_pos, mask_pos_inv] -> [half, 2F]
        # W_neg = [mask_neg, mask_neg_inv] -> [half, 2F]
        # W_total = [W_pos; W_neg] -> [n_clauses, 2F]
        
        x_neg = 1.0 - x
        X_combined = torch.cat([x_neg, x], dim=1)
        
        W_pos = torch.cat([mask_pos, mask_pos_inv], dim=1)
        W_neg = torch.cat([mask_neg, mask_neg_inv], dim=1)
        W_total = torch.cat([W_pos, W_neg], dim=0)
        
        mismatch = F.linear(X_combined, W_total) # [B, n_clauses]
        
        # Capacity = sum(mask_pos + mask_inv)
        # W_total rows are [mask_pos, mask_pos_inv] etc.
        # So sum(row) is exactly included capacity!
        
        capacity = W_total.sum(dim=1).unsqueeze(0) # [1, n_clauses]
        
        # Apply literal budget constraints to capacity
        if self.literal_budget is not None:
            limit = torch.as_tensor(float(self.literal_budget), device=capacity.device)
            capacity = torch.minimum(capacity, limit)
        if self.lf > 0:
            limit = torch.as_tensor(float(self.lf), device=capacity.device)
            capacity = torch.minimum(capacity, limit)
            
        # Strength = ReLU(capacity - mismatch)
        raw = capacity - mismatch
        strength = self._straight_relu(raw)
        
        pos_strength = strength[:, :half]
        neg_strength = strength[:, half:]

        clause_votes = torch.cat([pos_strength, -neg_strength], dim=1)
        if self.team_per_class:
            # clause_votes: [B, n_clauses] where n_clauses = half_pos + half_neg
            # We assume half_pos is interleaved or concatenated?
            # In __init__, ta_pos and ta_neg have size `half`.
            # _strength returns [B, half].
            # torch.cat([pos, -neg], dim=1) -> [B, half + half].
            #
            # We want to split into n_classes groups.
            # We enforced half % n_classes == 0.
            # k_half = half // n_classes
            #
            # Pos clauses for class c are indices [c*k_half : (c+1)*k_half] in pos_strength.
            # Neg clauses for class c are indices [c*k_half : (c+1)*k_half] in neg_strength.
            #
            # Reshape pos_strength to [B, n_classes, k_half]
            # Reshape neg_strength to [B, n_classes, k_half]
            
            B = clause_votes.shape[0]
            half = self.n_clauses // 2
            k_half = half // self.n_classes
            
            pos_votes = pos_strength.view(B, self.n_classes, k_half).sum(dim=2)
            neg_votes = neg_strength.view(B, self.n_classes, k_half).sum(dim=2)
            
            # Net score per class = Sum(Pos) - Sum(Neg)
            # Note that clause_votes contained -neg_strength, but here we work with raw strengths
            logits = pos_votes - neg_votes
        else:
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

    def prune(self, threshold: float = 0.1):
        """
        Prunes low-magnitude literal weights to zero.
        """
        with torch.no_grad():
            for param in [self.pos_logits, self.neg_logits]:
                if threshold > 0:
                    mask = torch.abs(param) < threshold
                    param.data.masked_fill_(mask, 0.0)

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
        if hasattr(self, 'operator_impl') and self.operator_impl is not None:
            matches = self._match_triplet(x, mask_pos, mask_inv)
            return self.operator_impl(*matches)
        if self.operator == "capacity":
            return self._capacity_strength(x, mask_pos, mask_inv)
        return self._product_strength(x, mask_pos, mask_inv)

    def _clause_outputs(self, x: torch.Tensor, use_ste: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Batch calculate masks
        # Stack logits: [4*half, n_features]
        # Note: self.pos_logits is [half, F], self.neg_logits is [half, F]
        # Masks required: pos_pos, pos_inv, neg_pos, neg_inv
        # STCM only stores pos_logits and neg_logits.
        # _split_masks splits one logit set into (pos, inv).
        
        # We can stack pos_logits and neg_logits -> [2*half, F]
        # Calculate mask -> [2*half, F]
        # Then split -> [4*half, F] effectively? No, split logic is inside _split_masks.
        
        # Let's optimize _split_masks first.
        # It does sigmoid(logits) and sigmoid(-logits).
        # sigmoid(-x) = 1 - sigmoid(x).
        
        all_logits = torch.cat([self.pos_logits, self.neg_logits], dim=0) # [2*half, F]
        
        temp = max(self.ste_temperature, 1e-6)
        
        if use_ste:
             mask_all = _ste_ternary(all_logits, self.ternary_band, self.ste_temperature)
        else:
             mask_all = torch.tanh(all_logits / self.ste_temperature)
             
        # Split back
        half = self.n_clauses // 2
        mask_pos = mask_all[:half]
        mask_neg = mask_all[half:]
        
        # Now _split_masks logic:
        # hard_pos = clamp(mask, min=0)
        # hard_inv = clamp(-mask, min=0)
        # soft_pos = sigmoid(logits/temp)
        # soft_inv = sigmoid(-logits/temp)
        
        # Optimize soft calculation:
        soft_all = torch.sigmoid(all_logits / temp)
        soft_pos_all = soft_all
        soft_inv_all = 1.0 - soft_all # sigmoid(-x) == 1 - sigmoid(x)
        
        # Optimize hard calculation:
        hard_pos_all = torch.clamp(mask_all, min=0.0)
        hard_inv_all = torch.clamp(-mask_all, min=0.0)
        
        # Apply STE-like pass-through if needed (mask_all already has it if use_ste? No, _split_masks adds another layer?)
        # _mask_from_logits returns values in [-1, 1].
        # _split_masks splits -1 -> inv=1, 1 -> pos=1.
        # And adds soft gradients.
        
        pos_all = hard_pos_all + (soft_pos_all - soft_pos_all.detach())
        inv_all = hard_inv_all + (soft_inv_all - soft_inv_all.detach())
        
        # Split into pos/neg halves
        pos_pos = pos_all[:half]
        pos_inv = inv_all[:half]
        neg_pos = pos_all[half:]
        neg_inv = inv_all[half:]
        
        # Enforce budget
        # We can optimize enforce_budget later if needed, it's elementwise/sum.
        pos_pos, pos_inv = self._enforce_literal_budget(pos_pos, pos_inv)
        neg_pos, neg_inv = self._enforce_literal_budget(neg_pos, neg_inv)
        
        # Calculate strengths
        # Similar to STE optimization: Fuse into single linear call?
        # _strength uses either capacity or product.
        # Capacity: relu(capacity - mismatch)
        # Mismatch = (1-x) @ mask_pos.t() + x @ mask_inv.t()
        
        # We can reuse the [1-x, x] concatenation trick!
        
        x_neg = 1.0 - x
        X_combined = torch.cat([x_neg, x], dim=1) # [B, 2F]
        
        # Weights:
        # W_pos = [mask_pos, mask_inv] (concatenated along F dim)
        # W_neg = [mask_neg, mask_neg_inv]
        
        # But wait, mask_pos is [half, F].
        # We need [mask_pos; mask_inv] -> [half, 2F].
        
        W_pos = torch.cat([pos_pos, pos_inv], dim=1)
        W_neg = torch.cat([neg_pos, neg_inv], dim=1)
        W_total = torch.cat([W_pos, W_neg], dim=0) # [n_clauses, 2F]
        
        # Mismatches / Penalties
        # For Capacity: mismatch = X_combined @ W_total.t()
        # For Product: penalties = X_combined @ W_total.t()
        # Both are just linear projection!
        
        raw_outputs = F.linear(X_combined, W_total) # [B, n_clauses]
        
        if self.operator == "capacity":
            # Need capacity per clause
            # capacity = sum(mask_pos + mask_inv)
            # W_total is [mask_pos, mask_inv] per row.
            # So sum(W_total, dim=1) is exactly capacity!
            capacity = W_total.sum(dim=1).unsqueeze(0) # [1, n_clauses]
            
            # Apply literal budget constraints to capacity (moved here from mask)
            capacity = self._apply_literal_constraints(capacity)

            # Raw strength = capacity - mismatch
            # mismatch = raw_outputs
            strength = self._straight_relu(capacity - raw_outputs)
            
        else: # Product
            # strength = exp(-clamp(raw_outputs * scale))
            scaled = torch.clamp(raw_outputs * self.product_scale, min=0.0, max=10.0)
            strength = torch.exp(-scaled)
            
        # Split back to pos/neg strength for voting
        pos_strength = strength[:, :half]
        neg_strength = strength[:, half:]
        
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
        clamped = F.leaky_relu(x, negative_slope=0.01)
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

    def forward(self, x: torch.Tensor, use_ste: bool = True, skip_norm: bool = False):
        """
        Args:
            x: Tensor in [0,1] with shape [batch, features] or image tensor convertible
               via prepare_tm_input.
            use_ste: Whether to use STE-based ternary masks (default True). If False,
                     hard ternary masks (no gradient) are used.
            skip_norm: If True, bypass prepare_tm_input checks.
        """
        if not skip_norm:
            flat_x = prepare_tm_input(
                x,
                n_features=self.n_features,
                input_shape=self.input_shape,
                auto_expand_grayscale=self.auto_expand_grayscale,
                allow_channel_reduce=self.allow_channel_reduce,
            )
        else:
            flat_x = x
            
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

