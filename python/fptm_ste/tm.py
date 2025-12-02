from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .operators import build_ternary_operator, available_ternary_operators


# =============================================================================
# Clause Memory Bank
# =============================================================================


class ClauseMemoryBank(nn.Module):
    """
    Persistent clause memory bank with EMA updates.
    
    This module maintains a memory bank of representative clause patterns
    that can be read from and written to during forward passes. The memory
    enables context-aware clause decisions by allowing clauses to access
    patterns learned from previous samples.
    
    Key features:
    - EMA-updated memory slots for stable pattern storage
    - Attention-based memory read for context retrieval
    - Optional memory write for pattern accumulation
    - Temperature-controlled attention for sharp/soft retrieval
    
    Args:
        n_slots: Number of memory slots
        clause_dim: Dimension of clause representations
        key_dim: Dimension of keys for memory access (default: clause_dim)
        ema_decay: EMA decay rate for memory updates
        temperature: Temperature for attention computation
        learnable_keys: Whether to use learnable key projections
    """
    
    def __init__(
        self,
        n_slots: int,
        clause_dim: int,
        key_dim: Optional[int] = None,
        ema_decay: float = 0.99,
        temperature: float = 1.0,
        learnable_keys: bool = True,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.clause_dim = clause_dim
        self.key_dim = key_dim if key_dim is not None else clause_dim
        self.ema_decay = ema_decay
        self.temperature = temperature
        
        # Memory bank (not a parameter, updated via EMA)
        self.register_buffer("memory", torch.randn(n_slots, clause_dim) * 0.02)
        self.register_buffer("memory_keys", torch.randn(n_slots, self.key_dim) * 0.02)
        
        # Key projections for queries
        if learnable_keys:
            self.query_proj = nn.Linear(clause_dim, self.key_dim)
            self.value_proj = nn.Linear(clause_dim, clause_dim)
        else:
            self.query_proj = None
            self.value_proj = None
        
        # Output projection to combine memory with clause
        self.output_gate = nn.Sequential(
            nn.Linear(clause_dim * 2, clause_dim),
            nn.Sigmoid(),
        )
    
    def read(self, clause_outputs: torch.Tensor) -> torch.Tensor:
        """
        Read from memory using clause outputs as queries.
        
        Args:
            clause_outputs: [batch, n_clauses, clause_dim] or [batch, n_clauses]
            
        Returns:
            Memory-enhanced clause outputs [batch, n_clauses, clause_dim]
        """
        # Handle 2D inputs - use a separate variable to avoid modifying input
        if clause_outputs.dim() == 2:
            x = clause_outputs.unsqueeze(-1)
            squeeze_output = True
        else:
            x = clause_outputs
            squeeze_output = False
        
        batch_size, n_clauses, _ = x.shape
        
        # Project queries
        if self.query_proj is not None:
            queries = self.query_proj(x)  # [batch, n_clauses, key_dim]
        else:
            queries = x
        
        # Compute attention over memory
        # keys: [n_slots, key_dim] -> [1, n_slots, key_dim]
        # Detach memory to avoid in-place modification issues during EMA update
        keys = self.memory_keys.detach().unsqueeze(0)
        
        # Attention: [batch, n_clauses, n_slots]
        attn_logits = torch.bmm(queries, keys.transpose(-2, -1).expand(batch_size, -1, -1))
        attn_logits = attn_logits / (self.key_dim ** 0.5 * self.temperature)
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        # Retrieve memory content: [batch, n_clauses, clause_dim]
        # Detach memory to avoid in-place modification issues
        memory_expanded = self.memory.detach().unsqueeze(0).expand(batch_size, -1, -1)
        retrieved = torch.bmm(attn_weights, memory_expanded)
        
        # Gate combination
        combined = torch.cat([x, retrieved], dim=-1)
        gate = self.output_gate(combined)
        output = x + gate * (retrieved - x)
        
        if squeeze_output:
            output = output.squeeze(-1)
        
        return output
    
    @torch.no_grad()
    def write(self, clause_outputs: torch.Tensor, update_keys: bool = True) -> None:
        """
        Write to memory using EMA update.
        
        Args:
            clause_outputs: [batch, n_clauses, clause_dim]
            update_keys: Whether to also update memory keys
        """
        if clause_outputs.dim() == 2:
            clause_outputs = clause_outputs.unsqueeze(-1)
        
        # Aggregate clause outputs
        batch_mean = clause_outputs.mean(dim=0)  # [n_clauses, clause_dim]
        
        # Subsample or pad to match n_slots
        n_clauses = batch_mean.shape[0]
        if n_clauses >= self.n_slots:
            # Take first n_slots
            update = batch_mean[:self.n_slots]
        else:
            # Pad with existing memory
            update = torch.cat([
                batch_mean,
                self.memory[n_clauses:],
            ], dim=0)
        
        # EMA update - use copy_ with computed value to avoid in-place issues
        new_memory = self.ema_decay * self.memory + (1 - self.ema_decay) * update
        self.memory.copy_(new_memory)
        
        if update_keys and self.query_proj is not None:
            # Update keys based on new memory
            with torch.enable_grad():
                new_keys = self.query_proj(self.memory.detach())
            new_memory_keys = self.ema_decay * self.memory_keys + (1 - self.ema_decay) * new_keys.detach()
            self.memory_keys.copy_(new_memory_keys)
    
    def forward(
        self,
        clause_outputs: torch.Tensor,
        update_memory: bool = True,
    ) -> torch.Tensor:
        """
        Read from memory and optionally update it.
        
        Args:
            clause_outputs: [batch, n_clauses, clause_dim]
            update_memory: Whether to write to memory (only during training)
            
        Returns:
            Memory-enhanced clause outputs
        """
        # Read from memory
        output = self.read(clause_outputs)
        
        # Update memory during training
        if update_memory and self.training:
            self.write(clause_outputs)
        
        return output


class ClauseMemoryAttention(nn.Module):
    """
    Memory-augmented attention for clauses.
    
    Combines ClauseMemoryBank with multi-head attention for more
    sophisticated memory access patterns.
    """
    
    def __init__(
        self,
        clause_dim: int,
        n_slots: int = 64,
        n_heads: int = 4,
        ema_decay: float = 0.99,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = clause_dim // n_heads
        
        # Memory bank
        self.memory = ClauseMemoryBank(
            n_slots=n_slots,
            clause_dim=clause_dim,
            ema_decay=ema_decay,
        )
        
        # Multi-head attention for memory access
        self.q_proj = nn.Linear(clause_dim, clause_dim)
        self.k_proj = nn.Linear(clause_dim, clause_dim)
        self.v_proj = nn.Linear(clause_dim, clause_dim)
        self.out_proj = nn.Linear(clause_dim, clause_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(clause_dim)
    
    def forward(
        self,
        clause_outputs: torch.Tensor,
        update_memory: bool = True,
    ) -> torch.Tensor:
        """
        Apply memory-augmented attention.
        
        Args:
            clause_outputs: [batch, n_clauses, clause_dim]
            update_memory: Whether to update memory
            
        Returns:
            Memory-enhanced outputs [batch, n_clauses, clause_dim]
        """
        # First, enhance with memory bank
        memory_enhanced = self.memory(clause_outputs, update_memory=update_memory)
        
        # Then apply self-attention
        batch_size, n_clauses, _ = clause_outputs.shape
        
        q = self.q_proj(memory_enhanced)
        k = self.k_proj(memory_enhanced)
        v = self.v_proj(memory_enhanced)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, n_clauses, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, n_clauses, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, n_clauses, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, n_clauses, -1)
        out = self.out_proj(out)
        
        # Residual connection and layer norm
        return self.norm(clause_outputs + out)


# =============================================================================
# Advanced Voting Mechanisms
# =============================================================================


class AttentionVoting(nn.Module):
    """
    Attention-based voting mechanism for clauses.
    
    Replaces fixed voting weights with input-dependent attention,
    allowing the model to dynamically weight clause contributions
    based on the input pattern.
    
    Args:
        n_clauses: Number of clauses
        n_classes: Number of output classes
        n_heads: Number of attention heads
        use_input_context: Whether to use input features for attention
        temperature: Softmax temperature
    """
    
    def __init__(
        self,
        n_clauses: int,
        n_classes: int,
        n_heads: int = 4,
        use_input_context: bool = True,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.n_heads = n_heads
        self.use_input_context = use_input_context
        self.temperature = temperature
        
        # Query: learnable class queries
        self.class_queries = nn.Parameter(torch.randn(1, n_classes, n_heads) * 0.02)
        
        # Key: project clause outputs
        self.clause_key = nn.Linear(1, n_heads)
        
        # Value: clause-to-class contribution (like traditional voting weights)
        self.clause_value = nn.Linear(1, n_classes)
        
        # Optional input context
        if use_input_context:
            self.context_proj = nn.Linear(n_clauses, n_heads * n_classes)
        
        self.dropout = nn.Dropout(dropout)
        
        # Output scaling
        self.output_scale = nn.Parameter(torch.ones(1))
    
    def forward(
        self,
        clause_outputs: torch.Tensor,
        input_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute attention-weighted voting.
        
        Args:
            clause_outputs: [batch, n_clauses] clause strength outputs
            input_features: [batch, n_features] original input (optional)
            
        Returns:
            [batch, n_classes] class logits
        """
        batch_size = clause_outputs.shape[0]
        
        # Expand clause outputs for projection
        clause_expanded = clause_outputs.unsqueeze(-1)  # [batch, n_clauses, 1]
        
        # Keys from clauses: [batch, n_clauses, n_heads]
        keys = self.clause_key(clause_expanded)
        
        # Values (contributions to each class): [batch, n_clauses, n_classes]
        values = self.clause_value(clause_expanded) * clause_expanded
        
        # Queries: class queries, optionally modulated by input context
        # [batch, n_classes, n_heads]
        queries = self.class_queries.expand(batch_size, -1, -1)
        
        if self.use_input_context and input_features is not None:
            # Use clause outputs as context
            context = self.context_proj(clause_outputs)  # [batch, n_heads * n_classes]
            context = context.view(batch_size, self.n_classes, self.n_heads)
            queries = queries + 0.1 * context
        
        # Compute attention: [batch, n_classes, n_clauses]
        # Each class attends to all clauses
        attn = torch.bmm(queries, keys.transpose(-2, -1))  # [batch, n_classes, n_clauses]
        attn = attn / (self.n_heads ** 0.5 * self.temperature)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted combination: [batch, n_classes]
        # For each class, sum over attended clause values
        logits = torch.bmm(attn, values)  # [batch, n_classes, n_classes]
        logits = logits.diagonal(dim1=-2, dim2=-1)  # [batch, n_classes]
        
        return logits * self.output_scale


class HierarchicalVoting(nn.Module):
    """
    Hierarchical voting with super-clauses.
    
    Two-level hierarchy:
    1. Clauses vote within "super-clauses" (local consensus)
    2. Super-clauses vote for classes (global decision)
    
    This creates an intermediate abstraction layer that can capture
    higher-level patterns.
    
    Args:
        n_clauses: Number of clauses
        n_classes: Number of output classes
        n_super_clauses: Number of super-clause groups
        aggregation: 'sum', 'mean', 'max', or 'attention'
    """
    
    def __init__(
        self,
        n_clauses: int,
        n_classes: int,
        n_super_clauses: int = 16,
        aggregation: str = "attention",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.n_super_clauses = n_super_clauses
        self.aggregation = aggregation
        
        # Clause to super-clause assignment (soft or hard)
        self.clause_to_super = nn.Linear(1, n_super_clauses)
        
        # Super-clause to class voting
        self.super_to_class = nn.Linear(n_super_clauses, n_classes)
        
        # If using attention aggregation
        if aggregation == "attention":
            self.local_attn = nn.Linear(1, n_super_clauses)
            self.global_attn = nn.Linear(n_super_clauses, n_classes)
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, clause_outputs: torch.Tensor) -> torch.Tensor:
        """
        Hierarchical voting.
        
        Args:
            clause_outputs: [batch, n_clauses] clause strength outputs
            
        Returns:
            [batch, n_classes] class logits
        """
        batch_size = clause_outputs.shape[0]
        clause_expanded = clause_outputs.unsqueeze(-1)  # [batch, n_clauses, 1]
        
        # Level 1: Clauses -> Super-clauses
        # Soft assignment weights: [batch, n_clauses, n_super_clauses]
        assignment = self.clause_to_super(clause_expanded)
        assignment = F.softmax(assignment / self.temperature, dim=-1)
        assignment = self.dropout(assignment)
        
        # Weighted aggregation to super-clauses
        if self.aggregation == "attention":
            # Use attention for weighted sum
            attn_weights = self.local_attn(clause_expanded)  # [batch, n_clauses, n_super_clauses]
            attn_weights = F.softmax(attn_weights / self.temperature, dim=1)
            super_outputs = (attn_weights * clause_expanded * assignment).sum(dim=1)
        elif self.aggregation == "sum":
            weighted = clause_expanded * assignment
            super_outputs = weighted.sum(dim=1)  # [batch, n_super_clauses]
        elif self.aggregation == "mean":
            weighted = clause_expanded * assignment
            counts = assignment.sum(dim=1).clamp(min=1)
            super_outputs = weighted.sum(dim=1) / counts
        elif self.aggregation == "max":
            weighted = clause_expanded * assignment
            super_outputs = weighted.max(dim=1).values
        
        # Level 2: Super-clauses -> Classes
        if self.aggregation == "attention":
            # Attention-weighted voting
            global_attn = self.global_attn(super_outputs)  # [batch, n_classes]
            global_attn = F.softmax(global_attn / self.temperature, dim=-1)
            # Use super_outputs as features for final projection
            logits = self.super_to_class(super_outputs * global_attn.mean(dim=-1, keepdim=True))
        else:
            logits = self.super_to_class(super_outputs)
        
        return logits


class ProbabilisticVoting(nn.Module):
    """
    Probabilistic voting with uncertainty estimation.
    
    Models clause-to-class contributions as distributions, enabling
    uncertainty quantification in predictions.
    
    Args:
        n_clauses: Number of clauses
        n_classes: Number of output classes
        n_samples: Monte Carlo samples for uncertainty
    """
    
    def __init__(
        self,
        n_clauses: int,
        n_classes: int,
        n_samples: int = 10,
    ):
        super().__init__()
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.n_samples = n_samples
        
        # Mean and log-variance of voting weights
        self.voting_mean = nn.Parameter(torch.randn(n_clauses, n_classes) * 0.1)
        self.voting_logvar = nn.Parameter(torch.zeros(n_clauses, n_classes) - 2.0)
        
        # Temperature for output
        self.temperature = nn.Parameter(torch.ones(1))
    
    def _sample_weights(self, n_samples: int = None) -> torch.Tensor:
        """Sample voting weights from distribution."""
        if n_samples is None:
            n_samples = self.n_samples
        
        std = torch.exp(0.5 * self.voting_logvar)
        eps = torch.randn(n_samples, *self.voting_mean.shape, device=self.voting_mean.device)
        return self.voting_mean + eps * std
    
    def forward(
        self,
        clause_outputs: torch.Tensor,
        return_uncertainty: bool = False,
    ):
        """
        Probabilistic voting with optional uncertainty.
        
        Args:
            clause_outputs: [batch, n_clauses] clause strength outputs
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            logits or (logits, uncertainty)
        """
        if self.training or not return_uncertainty:
            # Use mean during training or when uncertainty not needed
            weights = self.voting_mean
            logits = torch.mm(clause_outputs, weights) / self.temperature
            
            if return_uncertainty:
                # Approximate uncertainty from weight variance
                var = torch.exp(self.voting_logvar)
                uncertainty = torch.mm(clause_outputs ** 2, var)
                return logits, uncertainty
            return logits
        
        # Monte Carlo sampling for uncertainty
        all_logits = []
        sampled_weights = self._sample_weights(self.n_samples)
        
        for i in range(self.n_samples):
            weights = sampled_weights[i]
            logits = torch.mm(clause_outputs, weights) / self.temperature
            all_logits.append(logits)
        
        # Stack and compute statistics
        stacked = torch.stack(all_logits, dim=0)  # [n_samples, batch, n_classes]
        mean_logits = stacked.mean(dim=0)
        uncertainty = stacked.var(dim=0)  # Epistemic uncertainty
        
        return mean_logits, uncertainty
    
    def kl_divergence(self) -> torch.Tensor:
        """KL divergence from prior N(0, 1) for regularization."""
        mean = self.voting_mean
        logvar = self.voting_logvar
        var = torch.exp(logvar)
        
        kl = 0.5 * (var + mean ** 2 - 1 - logvar).sum()
        return kl


class ConfidenceWeightedVoting(nn.Module):
    """
    Voting weighted by clause confidence.
    
    Clauses with higher confidence (stronger match) get more weight
    in the voting process.
    """
    
    def __init__(
        self,
        n_clauses: int,
        n_classes: int,
        confidence_type: str = "softmax",  # 'softmax', 'sigmoid', or 'sparsemax'
    ):
        super().__init__()
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.confidence_type = confidence_type
        
        # Base voting weights
        self.voting = nn.Parameter(torch.randn(n_clauses, n_classes) * 0.1)
        
        # Confidence projection
        self.confidence_proj = nn.Linear(n_clauses, n_clauses)
        
        # Temperature
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, clause_outputs: torch.Tensor) -> torch.Tensor:
        """
        Confidence-weighted voting.
        
        Args:
            clause_outputs: [batch, n_clauses] clause strength outputs
            
        Returns:
            [batch, n_classes] class logits
        """
        # Compute clause confidence
        confidence_logits = self.confidence_proj(clause_outputs)
        
        if self.confidence_type == "softmax":
            confidence = F.softmax(confidence_logits / self.temperature, dim=-1)
        elif self.confidence_type == "sigmoid":
            confidence = torch.sigmoid(confidence_logits / self.temperature)
        else:  # sparsemax-like approximation
            confidence = F.softmax(confidence_logits / (self.temperature * 0.1), dim=-1)
        
        # Weight clause outputs by confidence
        weighted_outputs = clause_outputs * confidence
        
        # Vote
        logits = torch.mm(weighted_outputs, self.voting)
        
        return logits


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

    def get_masks(self, use_ste: bool = True):
        """
        Returns the literal inclusion masks for the positive and negative clause banks.
        
        Returns:
            Tuple of (p_pos, p_neg, p_pos_inv, p_neg_inv) where:
            - p_pos: [half, n_features] - positive literals for positive clauses
            - p_neg: [half, n_features] - positive literals for negative clauses
            - p_pos_inv: [half, n_features] - inverted literals for positive clauses
            - p_neg_inv: [half, n_features] - inverted literals for negative clauses
        """
        half = self.n_clauses // 2
        all_logits = torch.cat([self.ta_pos, self.ta_neg, self.ta_pos_inv, self.ta_neg_inv], dim=0)
        
        if use_ste:
            all_p = self._ste_binary(all_logits, self.tau)
        else:
            all_p = torch.sigmoid(all_logits)
            
        all_p = all_p.clamp(0.0, 1.0)
        
        p_pos = all_p[0:half]
        p_neg = all_p[half:2*half]
        p_pos_inv = all_p[2*half:3*half]
        p_neg_inv = all_p[3*half:4*half]
        
        return p_pos, p_neg, p_pos_inv, p_neg_inv

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
        
        # Support for extended fuzzy operators
        valid_operators = {"capacity", "product"} | set(available_ternary_operators())
        if operator not in valid_operators:
            raise ValueError(
                f"operator must be one of {sorted(valid_operators)}, got '{operator}'."
            )
        
        # Build custom operator module if not using built-in capacity/product
        self.operator_impl = None
        if operator not in {"capacity", "product"}:
            self.operator_impl = build_ternary_operator(operator)
            if self.operator_impl is not None:
                # Register as submodule for proper parameter tracking
                self.add_module("_operator_impl", self.operator_impl)
        
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

    def _match_scores(self, x: torch.Tensor, mask_pos: torch.Tensor, mask_inv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute match scores for positive and inverted literals.
        
        Returns:
            pos_match: How well x matches positive literal requirements (high x where mask_pos is active)
            inv_match: How well x matches inverted literal requirements (low x where mask_inv is active)
        """
        # Positive literals: x should be high where mask_pos is active
        # Match score = x * mask_pos normalized by number of active literals
        pos_count = mask_pos.sum(dim=1, keepdim=True).clamp(min=1.0)
        pos_match = F.linear(x, mask_pos) / pos_count.t()
        
        # Inverted literals: x should be low (1-x high) where mask_inv is active
        inv_count = mask_inv.sum(dim=1, keepdim=True).clamp(min=1.0)
        inv_match = F.linear(1.0 - x, mask_inv) / inv_count.t()
        
        return pos_match, inv_match

    def _strength(self, x: torch.Tensor, mask_pos: torch.Tensor, mask_inv: torch.Tensor) -> torch.Tensor:
        mask_pos = mask_pos.to(x.dtype)
        mask_inv = mask_inv.to(x.dtype)
        
        # Check for custom fuzzy operator
        if hasattr(self, 'operator_impl') and self.operator_impl is not None:
            # Compute match scores for the fuzzy operator
            pos_match, inv_match = self._match_scores(x, mask_pos, mask_inv)
            # Apply fuzzy t-norm to combine match scores
            combined = self.operator_impl(pos_match, inv_match)
            return combined
        
        # Built-in operators
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

    def get_masks(self, use_ste: bool = True):
        """
        Returns the split ternary masks for the positive and negative clause banks.
        
        Returns:
            Tuple of (pos_pos, neg_pos, pos_inv, neg_inv) where:
            - pos_pos: [half, n_features] - positive literals (x=1) for positive clauses
            - neg_pos: [half, n_features] - positive literals (x=1) for negative clauses
            - pos_inv: [half, n_features] - inverted literals (x=0) for positive clauses
            - neg_inv: [half, n_features] - inverted literals (x=0) for negative clauses
        """
        half = self.n_clauses // 2
        all_logits = torch.cat([self.pos_logits, self.neg_logits], dim=0)
        
        temp = max(self.ste_temperature, 1e-6)
        
        if use_ste:
            mask_all = _ste_ternary(all_logits, self.ternary_band, self.ste_temperature)
        else:
            mask_all = torch.tanh(all_logits / self.ste_temperature)
        
        mask_pos = mask_all[:half]
        mask_neg = mask_all[half:]
        
        soft_all = torch.sigmoid(all_logits / temp)
        soft_pos_all = soft_all
        soft_inv_all = 1.0 - soft_all
        
        hard_pos_all = torch.clamp(mask_all, min=0.0)
        hard_inv_all = torch.clamp(-mask_all, min=0.0)
        
        pos_all = hard_pos_all + (soft_pos_all - soft_pos_all.detach())
        inv_all = hard_inv_all + (soft_inv_all - soft_inv_all.detach())
        
        pos_pos = pos_all[:half]
        pos_inv = inv_all[:half]
        neg_pos = pos_all[half:]
        neg_inv = inv_all[half:]
        
        pos_pos, pos_inv = self._enforce_literal_budget(pos_pos, pos_inv)
        neg_pos, neg_inv = self._enforce_literal_budget(neg_pos, neg_inv)
        
        return pos_pos, neg_pos, pos_inv, neg_inv

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

