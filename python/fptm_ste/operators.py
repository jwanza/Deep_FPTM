from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class _BaseTernaryOperator(nn.Module):
    """Base class for ternary clause operators."""

    name: str = "base"

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


# =============================================================================
# Classical Fuzzy Logic T-Norms
# =============================================================================


class LukasiewiczTNorm(_BaseTernaryOperator):
    """
    Lukasiewicz t-norm (bounded product).

    T(a, b) = max(0, a + b - 1)

    This is the strongest t-norm that is continuous. It represents a "strong 
    conjunction" where both inputs must be high for a high output. The bounded
    nature provides good gradient properties near the boundary.

    For clause matching: High strength only when most literals are satisfied.
    """

    name = "lukasiewicz"

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) < 2:
            raise ValueError("LukasiewiczTNorm expects at least two inputs.")
        result = inputs[0]
        for inp in inputs[1:]:
            result = torch.clamp(result + inp - 1.0, min=0.0)
        return result


class GodelTNorm(_BaseTernaryOperator):
    """
    Gödel t-norm (minimum).

    T(a, b) = min(a, b)

    The weakest t-norm, representing a "weak conjunction" that takes the
    minimum evidence. This is idempotent: T(a, a) = a.

    For clause matching: Clause strength limited by the worst-matching literal.
    Good for strict pattern matching where all literals matter equally.
    """

    name = "godel"

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if not inputs:
            raise ValueError("GodelTNorm expects at least one input.")
        result = inputs[0]
        for inp in inputs[1:]:
            result = torch.minimum(result, inp)
        return result


class HamacherProduct(_BaseTernaryOperator):
    """
    Hamacher product t-norm.

    T(a, b) = (a * b) / (a + b - a*b + eps)

    A smooth alternative to the product t-norm with better gradient properties
    near zero. Approaches the product t-norm as values increase.

    For clause matching: Smooth multiplicative interaction with stability.
    """

    name = "hamacher"

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) < 2:
            raise ValueError("HamacherProduct expects at least two inputs.")
        result = inputs[0]
        for inp in inputs[1:]:
            numerator = result * inp
            denominator = result + inp - result * inp + self.eps
            result = numerator / denominator
        return result


class YagerTNorm(_BaseTernaryOperator):
    """
    Yager t-norm (parameterized family).

    T(a, b) = 1 - min(1, ((1-a)^p + (1-b)^p)^(1/p))

    Parameterized by p > 0:
    - p → 0: Approaches drastic product (very strict)
    - p = 1: Equals Lukasiewicz t-norm
    - p → ∞: Approaches Gödel (minimum) t-norm

    For clause matching: Tunable strictness for clause evaluation.
    """

    name = "yager"

    def __init__(self, p: float = 2.0, eps: float = 1e-8):
        super().__init__()
        self.p = max(0.01, p)  # Prevent division by zero
        self.eps = eps

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) < 2:
            raise ValueError("YagerTNorm expects at least two inputs.")
        
        # Compute (1-a)^p for all inputs
        neg_powers = [(1.0 - inp).clamp(min=0.0).pow(self.p) for inp in inputs]
        
        # Sum them up
        total = neg_powers[0]
        for np in neg_powers[1:]:
            total = total + np
        
        # Take p-th root and subtract from 1
        root = total.pow(1.0 / self.p)
        result = 1.0 - torch.minimum(torch.ones_like(root), root)
        return result


class DrasticProduct(_BaseTernaryOperator):
    """
    Drastic product t-norm (strictest t-norm).

    T(a, b) = a if b == 1
            = b if a == 1
            = 0 otherwise

    The strictest t-norm - requires at least one input to be exactly 1.
    Uses soft approximation for differentiability.

    For clause matching: Very strict - only perfect matches contribute.
    """

    name = "drastic"

    def __init__(self, sharpness: float = 10.0, eps: float = 1e-8):
        super().__init__()
        self.sharpness = sharpness
        self.eps = eps

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) < 2:
            raise ValueError("DrasticProduct expects at least two inputs.")
        
        result = inputs[0]
        for inp in inputs[1:]:
            # Soft approximation: weight by how close each is to 1
            weight_a = torch.sigmoid((result - 0.99) * self.sharpness)
            weight_b = torch.sigmoid((inp - 0.99) * self.sharpness)
            
            # If a ≈ 1, use b; if b ≈ 1, use a; else ≈ 0
            result = weight_a * inp + weight_b * result * (1 - weight_a) + \
                     (1 - weight_a) * (1 - weight_b) * self.eps
        return result


class EinsteinProduct(_BaseTernaryOperator):
    """
    Einstein product t-norm.

    T(a, b) = (a * b) / (2 - (a + b - a*b))

    Similar to Hamacher but with different denominator. Provides smooth
    gradients and is associative.

    For clause matching: Alternative smooth product with different curvature.
    """

    name = "einstein"

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) < 2:
            raise ValueError("EinsteinProduct expects at least two inputs.")
        
        result = inputs[0]
        for inp in inputs[1:]:
            numerator = result * inp
            denominator = 2.0 - (result + inp - result * inp) + self.eps
            result = numerator / denominator
        return result


class NilpotentMinimum(_BaseTernaryOperator):
    """
    Nilpotent minimum t-norm.

    T(a, b) = min(a, b) if a + b > 1
            = 0 otherwise

    Combines aspects of Gödel and drastic product. Activates only when
    sum exceeds 1, then takes minimum.

    For clause matching: Moderate strictness with threshold behavior.
    """

    name = "nilpotent"

    def __init__(self, sharpness: float = 10.0):
        super().__init__()
        self.sharpness = sharpness

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) < 2:
            raise ValueError("NilpotentMinimum expects at least two inputs.")
        
        result = inputs[0]
        for inp in inputs[1:]:
            min_val = torch.minimum(result, inp)
            # Soft threshold at a + b > 1
            gate = torch.sigmoid((result + inp - 1.0) * self.sharpness)
            result = gate * min_val
        return result


class BoundedDifference(_BaseTernaryOperator):
    """
    Bounded difference (s-norm as t-norm complement).

    Uses the relationship between t-norms and s-norms to create
    a bounded version of the difference operator.

    T(a, b) = max(0, min(a, b) - (1 - max(a, b)))

    For clause matching: Emphasizes agreement between literals.
    """

    name = "bounded_diff"

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) < 2:
            raise ValueError("BoundedDifference expects at least two inputs.")
        
        result = inputs[0]
        for inp in inputs[1:]:
            min_val = torch.minimum(result, inp)
            max_val = torch.maximum(result, inp)
            result = torch.clamp(min_val - (1.0 - max_val), min=0.0)
        return result


# =============================================================================
# Adaptive and Learnable Operators
# =============================================================================


class ParameterizedTNorm(_BaseTernaryOperator):
    """
    Parameterized t-norm family (Schweizer-Sklar).

    T_p(a, b) = (max(0, a^p + b^p - 1))^(1/p)  for p != 0
              = a * b                           for p = 0

    The parameter p is learnable, allowing the network to discover
    the optimal t-norm for the task.

    - p → -∞: Approaches drastic product
    - p = -1: Hamacher-like
    - p = 0: Product t-norm
    - p = 1: Lukasiewicz
    - p → +∞: Approaches Gödel (minimum)
    """

    name = "parameterized"

    def __init__(self, init_p: float = 1.0, learnable: bool = True, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        if learnable:
            self.p = nn.Parameter(torch.tensor(init_p))
        else:
            self.register_buffer("p", torch.tensor(init_p))

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) < 2:
            raise ValueError("ParameterizedTNorm expects at least two inputs.")
        
        p = self.p
        
        # Handle p ≈ 0 case (product t-norm)
        if torch.abs(p) < 0.01:
            result = inputs[0]
            for inp in inputs[1:]:
                result = result * inp
            return result
        
        # General case
        result = inputs[0].clamp(min=self.eps)
        for inp in inputs[1:]:
            inp_clamped = inp.clamp(min=self.eps)
            inner = result.pow(p) + inp_clamped.pow(p) - 1.0
            inner_clamped = torch.clamp(inner, min=self.eps)
            result = inner_clamped.pow(1.0 / p)
        
        return result.clamp(0.0, 1.0)


class SoftMinMax(_BaseTernaryOperator):
    """
    Soft interpolation between min and max (weighted quasi-arithmetic mean).

    T(a, b) = α * min(a, b) + (1-α) * a * b

    Where α is learnable. This allows smooth interpolation between
    Gödel (α=1) and product (α=0) t-norms.

    For clause matching: Adaptive strictness learned from data.
    """

    name = "softminmax"

    def __init__(self, init_alpha: float = 0.5, learnable: bool = True):
        super().__init__()
        # Use logit for unconstrained optimization
        init_logit = torch.log(torch.tensor(init_alpha / (1 - init_alpha + 1e-8)))
        if learnable:
            self.alpha_logit = nn.Parameter(init_logit)
        else:
            self.register_buffer("alpha_logit", init_logit)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) < 2:
            raise ValueError("SoftMinMax expects at least two inputs.")
        
        alpha = self.alpha
        result = inputs[0]
        
        for inp in inputs[1:]:
            min_val = torch.minimum(result, inp)
            prod_val = result * inp
            result = alpha * min_val + (1 - alpha) * prod_val
        
        return result


class TernaryQuantumAND(_BaseTernaryOperator):
    """
    Quantum-inspired ternary AND (TQAND).

    Normalizes the multiplicative interaction of three inputs to stay numerically
    stable while preserving relative magnitudes.
    """

    name = "tqand"

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) < 3:
            raise ValueError("TernaryQuantumAND expects at least three inputs.")
        a, b, c = inputs[:3]
        denom = torch.sqrt(torch.clamp(a**2 + b**2 + c**2, min=self.eps))
        return (a * b * c) / denom


class TernaryXOR(_BaseTernaryOperator):
    """
    Ternary XOR (TXOR).

    Emphasizes disagreement between two inputs using a quadratic remainder that
    stays inside the [0, 1] interval.
    """

    name = "txor"

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) < 2:
            raise ValueError("TernaryXOR expects at least two inputs.")
        a, b = inputs[:2]
        diff_sq = (a - b) ** 2
        return torch.remainder(diff_sq, 3.0) / 3.0


class TernaryMajority(_BaseTernaryOperator):
    """
    Ternary majority gate (TMAJ).

    Returns the median across all inputs which corresponds to the balanced
    ternary majority vote.
    """

    name = "tmaj"

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if not inputs:
            raise ValueError("TernaryMajority expects at least one input.")
        stacked = torch.stack(inputs)
        return torch.median(stacked, dim=0).values


# =============================================================================
# Adaptive Operator Mixing
# =============================================================================


class AdaptiveOperatorMixer(nn.Module):
    """
    Learnable mixture of multiple fuzzy operators.
    
    This module learns to combine outputs from multiple operators using
    attention-like weights. The mixing can be:
    - Global: same weights for all clauses/positions
    - Per-clause: different weights per clause  
    - Per-position: different weights per spatial position
    
    The learned combination allows the network to discover the optimal
    operator blend for the task at hand.
    
    Args:
        operators: List of operator names to mix
        n_clauses: Number of clauses (for per-clause mixing)
        mixing_type: 'global', 'per_clause', or 'attention'
        temperature: Softmax temperature for weight computation
        init_uniform: If True, initialize weights uniformly
    """
    
    name = "adaptive_mixer"
    
    def __init__(
        self,
        operators: Optional[List[str]] = None,
        n_clauses: int = 1,
        mixing_type: str = "global",
        temperature: float = 1.0,
        init_uniform: bool = True,
    ):
        super().__init__()
        
        # Default to a diverse set of operators
        if operators is None:
            operators = ["godel", "lukasiewicz", "hamacher", "softminmax"]
        
        self.operator_names = operators
        self.n_operators = len(operators)
        self.n_clauses = n_clauses
        self.mixing_type = mixing_type
        self.temperature = temperature
        
        # Build operator instances
        self.operators = nn.ModuleList()
        for name in operators:
            op = build_ternary_operator(name)
            if op is None:
                raise ValueError(f"Unknown operator: {name}")
            self.operators.append(op)
        
        # Initialize mixing weights
        if mixing_type == "global":
            weight_shape = (self.n_operators,)
        elif mixing_type == "per_clause":
            weight_shape = (n_clauses, self.n_operators)
        elif mixing_type == "attention":
            # For attention, we use a small network to compute weights
            self.attention_net = nn.Sequential(
                nn.Linear(2, 16),  # 2 inputs (pos_match, inv_match stats)
                nn.ReLU(),
                nn.Linear(16, self.n_operators),
            )
            weight_shape = None
        else:
            raise ValueError(f"Unknown mixing_type: {mixing_type}")
        
        if weight_shape is not None:
            if init_uniform:
                init_val = torch.zeros(*weight_shape)
            else:
                init_val = torch.randn(*weight_shape) * 0.1
            self.mixing_logits = nn.Parameter(init_val)
        else:
            self.mixing_logits = None
    
    def get_mixing_weights(
        self,
        batch_size: int,
        device: torch.device,
        inputs: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        """Compute mixing weights based on mixing type."""
        if self.mixing_type == "attention" and inputs is not None:
            # Compute input statistics for attention
            a, b = inputs[:2]
            stats = torch.stack([
                a.mean(dim=-1),  # Mean of first input
                b.mean(dim=-1),  # Mean of second input
            ], dim=-1)  # [batch, 2]
            
            logits = self.attention_net(stats)  # [batch, n_operators]
            weights = torch.softmax(logits / self.temperature, dim=-1)
            return weights
        
        if self.mixing_logits is None:
            # Fallback to uniform
            return torch.ones(self.n_operators, device=device) / self.n_operators
        
        weights = torch.softmax(self.mixing_logits / self.temperature, dim=-1)
        
        if self.mixing_type == "global":
            # Expand to match batch size
            weights = weights.unsqueeze(0).expand(batch_size, -1)
        elif self.mixing_type == "per_clause":
            # [n_clauses, n_operators] -> need to handle batch
            weights = weights.unsqueeze(0).expand(batch_size, -1, -1)
        
        return weights
    
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply mixture of operators to inputs.
        
        Args:
            *inputs: Input tensors (typically 2 for t-norms)
            
        Returns:
            Mixed operator output
        """
        if len(inputs) < 2:
            raise ValueError("AdaptiveOperatorMixer expects at least two inputs.")
        
        batch_size = inputs[0].shape[0]
        device = inputs[0].device
        
        # Compute individual operator outputs
        op_outputs = []
        for op in self.operators:
            out = op(*inputs)
            op_outputs.append(out)
        
        # Stack: [n_operators, batch, ...]
        stacked = torch.stack(op_outputs, dim=0)
        
        # Get mixing weights
        weights = self.get_mixing_weights(batch_size, device, inputs)
        
        # Apply mixing
        if self.mixing_type == "per_clause" and stacked.dim() > 2:
            # [n_operators, batch, n_clauses] * [batch, n_clauses, n_operators]
            weights = weights.permute(0, 2, 1)  # [batch, n_operators, n_clauses]
            stacked = stacked.permute(1, 0, 2)  # [batch, n_operators, n_clauses]
            result = (stacked * weights).sum(dim=1)  # [batch, n_clauses]
        else:
            # [n_operators, batch, ...] with [batch, n_operators]
            # Expand weights to match stacked shape
            weight_shape = [1] * stacked.dim()
            weight_shape[0] = self.n_operators
            weight_shape[1] = batch_size
            weights_expanded = weights.t().view(*weight_shape[:2], *([1] * (stacked.dim() - 2)))
            result = (stacked * weights_expanded).sum(dim=0)
        
        return result
    
    def get_operator_contributions(self) -> Dict[str, float]:
        """Get the learned contribution of each operator."""
        if self.mixing_logits is None:
            return {name: 1.0 / len(self.operator_names) for name in self.operator_names}
        
        with torch.no_grad():
            if self.mixing_type == "global":
                weights = torch.softmax(self.mixing_logits / self.temperature, dim=-1)
                return {
                    name: float(weights[i])
                    for i, name in enumerate(self.operator_names)
                }
            elif self.mixing_type == "per_clause":
                weights = torch.softmax(self.mixing_logits / self.temperature, dim=-1)
                mean_weights = weights.mean(dim=0)
                return {
                    name: float(mean_weights[i])
                    for i, name in enumerate(self.operator_names)
                }
        
        return {name: 0.0 for name in self.operator_names}


class EnsembleOperator(nn.Module):
    """
    Ensemble of operators with learnable gating.
    
    Similar to AdaptiveOperatorMixer but uses a more sophisticated
    gating mechanism based on input characteristics.
    """
    
    name = "ensemble"
    
    def __init__(
        self,
        operators: Optional[List[str]] = None,
        hidden_dim: int = 32,
        use_residual: bool = True,
    ):
        super().__init__()
        
        if operators is None:
            operators = ["godel", "lukasiewicz", "hamacher"]
        
        self.operator_names = operators
        self.n_operators = len(operators)
        self.use_residual = use_residual
        
        # Build operators
        self.operators = nn.ModuleList()
        for name in operators:
            op = build_ternary_operator(name)
            if op is None:
                raise ValueError(f"Unknown operator: {name}")
            self.operators.append(op)
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_operators),
            nn.Softmax(dim=-1),
        )
        
        # Optional output projection
        if use_residual:
            self.output_proj = nn.Linear(1, 1, bias=False)
            nn.init.ones_(self.output_proj.weight)
    
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) < 2:
            raise ValueError("EnsembleOperator expects at least two inputs.")
        
        a, b = inputs[0], inputs[1]
        batch_size = a.shape[0]
        
        # Compute gate inputs (statistics of inputs)
        gate_input = torch.stack([
            a.mean(dim=-1),
            b.mean(dim=-1),
        ], dim=-1)  # [batch, 2]
        
        # Get gating weights
        gate_weights = self.gate(gate_input)  # [batch, n_operators]
        
        # Compute operator outputs
        outputs = []
        for op in self.operators:
            outputs.append(op(*inputs))
        stacked = torch.stack(outputs, dim=-1)  # [batch, ..., n_operators]
        
        # Apply gating
        if stacked.dim() == 3:
            # [batch, features, n_operators]
            gate_weights = gate_weights.unsqueeze(1)  # [batch, 1, n_operators]
        
        result = (stacked * gate_weights).sum(dim=-1)
        
        # Residual connection with product
        if self.use_residual:
            residual = a * b
            result = result + 0.1 * residual
        
        return result


_REGISTERED_OPERATORS = {
    cls.name: cls for cls in (
        # Original operators
        TernaryQuantumAND,
        TernaryXOR,
        TernaryMajority,
        # Classical fuzzy t-norms
        LukasiewiczTNorm,
        GodelTNorm,
        HamacherProduct,
        YagerTNorm,
        DrasticProduct,
        EinsteinProduct,
        NilpotentMinimum,
        BoundedDifference,
        # Learnable operators
        ParameterizedTNorm,
        SoftMinMax,
        # Adaptive mixing
        AdaptiveOperatorMixer,
        EnsembleOperator,
    )
}


def available_ternary_operators() -> Tuple[str, ...]:
    """Returns the tuple of operator identifiers that can be requested."""

    return tuple(sorted(_REGISTERED_OPERATORS))


def build_ternary_operator(name: Optional[str]) -> Optional[_BaseTernaryOperator]:
    """
    Returns an instantiated operator module if ``name`` refers to a custom ternary
    operator. ``None`` means the caller should fall back to the legacy capacity /
    product logic implemented inside the STCM module.
    """

    if name is None:
        return None
    key = name.lower()
    cls = _REGISTERED_OPERATORS.get(key)
    if cls is None:
        return None
    return cls()

