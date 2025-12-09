"""
Low-Rank Adaptation (LoRA) for Tsetlin Machines.

This module implements LoRA-style adaptation for efficient fine-tuning
of TM models on new tasks while keeping base weights frozen.

Key idea: Instead of modifying all weights, learn low-rank updates:
W' = W + BA, where B ∈ R^{d×r} and A ∈ R^{r×k} with r << min(d, k)

Benefits:
1. Parameter efficient: Only r*(d+k) params instead of d*k
2. No forgetting: Base weights are frozen
3. Mergeable: Can merge LoRA into base weights for inference
4. Composable: Multiple LoRA adapters can be combined

References:
- Hu et al. (2021): LoRA: Low-Rank Adaptation of Large Language Models
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tm import FuzzyPatternTM_STCM, FuzzyPatternTM_STE


# =============================================================================
# Core LoRA Layer
# =============================================================================


class LoRALayer(nn.Module):
    """
    Low-rank adaptation layer.
    
    Implements the low-rank update W' = W + (α/r) * BA where:
    - W: Original frozen weight matrix
    - B: Low-rank down-projection (initialized to zero)
    - A: Low-rank up-projection (initialized from normal distribution)
    - α: Scaling factor
    - r: Rank
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of the low-rank matrices (r)
        alpha: Scaling factor (default: rank)
        dropout: Dropout rate for LoRA path
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank
        
        # Low-rank matrices
        # A: [rank, in_features] - initialized with Kaiming
        # B: [out_features, rank] - initialized with zeros
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # For merged weights
        self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LoRA output.
        
        Args:
            x: Input tensor [..., in_features]
            
        Returns:
            LoRA update [..., out_features]
        """
        # LoRA path: x @ A^T @ B^T * scaling
        lora_out = x @ self.lora_A.t() @ self.lora_B.t()
        lora_out = self.dropout(lora_out) * self.scaling
        return lora_out
    
    def get_delta_weight(self) -> torch.Tensor:
        """
        Get the full delta weight matrix for merging.
        
        Returns:
            Delta weight [out_features, in_features]
        """
        return (self.lora_B @ self.lora_A) * self.scaling
    
    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, alpha={self.alpha}"


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    Wraps a standard linear layer and adds LoRA adaptation.
    Original weights can be frozen or trainable.
    
    Args:
        linear: Original linear layer
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout
        freeze_base: Whether to freeze original weights
    """
    
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 4,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.linear = linear
        self.freeze_base = freeze_base
        
        # Freeze base weights if requested
        if freeze_base:
            for param in linear.parameters():
                param.requires_grad = False
        
        # Add LoRA layer
        self.lora = LoRALayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with LoRA adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        if self.merged:
            return self.linear(x)
        
        return self.linear(x) + self.lora(x)
    
    def merge_weights(self) -> None:
        """Merge LoRA weights into base weights for faster inference."""
        if self.merged:
            return
        
        delta = self.lora.get_delta_weight()
        self.linear.weight.data += delta
        self.merged = True
    
    def unmerge_weights(self) -> None:
        """Unmerge LoRA weights from base weights."""
        if not self.merged:
            return
        
        delta = self.lora.get_delta_weight()
        self.linear.weight.data -= delta
        self.merged = False


# =============================================================================
# LoRA Adapters for TM Components
# =============================================================================


class LoRAClauseAdapter(nn.Module):
    """
    LoRA adapter for TM clause weights.
    
    Adapts the clause masks (positive and negative logits) using
    low-rank updates while keeping original masks frozen.
    
    Args:
        n_clauses: Number of clauses
        n_features: Number of input features
        rank: LoRA rank
        alpha: LoRA scaling factor
        adapt_positive: Adapt positive clause masks
        adapt_negative: Adapt negative clause masks
    """
    
    def __init__(
        self,
        n_clauses: int,
        n_features: int,
        rank: int = 4,
        alpha: Optional[float] = None,
        adapt_positive: bool = True,
        adapt_negative: bool = True,
    ):
        super().__init__()
        self.n_clauses = n_clauses
        self.n_features = n_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank
        
        half = n_clauses // 2
        
        # LoRA for positive clause masks
        if adapt_positive:
            self.pos_lora_A = nn.Parameter(torch.zeros(rank, n_features))
            self.pos_lora_B = nn.Parameter(torch.zeros(half, rank))
            nn.init.kaiming_uniform_(self.pos_lora_A, a=math.sqrt(5))
        else:
            self.pos_lora_A = None
            self.pos_lora_B = None
        
        # LoRA for negative clause masks
        if adapt_negative:
            self.neg_lora_A = nn.Parameter(torch.zeros(rank, n_features))
            self.neg_lora_B = nn.Parameter(torch.zeros(half, rank))
            nn.init.kaiming_uniform_(self.neg_lora_A, a=math.sqrt(5))
        else:
            self.neg_lora_A = None
            self.neg_lora_B = None
    
    def get_pos_delta(self) -> Optional[torch.Tensor]:
        """Get delta for positive clause masks."""
        if self.pos_lora_A is None:
            return None
        return (self.pos_lora_B @ self.pos_lora_A) * self.scaling
    
    def get_neg_delta(self) -> Optional[torch.Tensor]:
        """Get delta for negative clause masks."""
        if self.neg_lora_A is None:
            return None
        return (self.neg_lora_B @ self.neg_lora_A) * self.scaling
    
    def forward(
        self,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply LoRA adaptation to clause logits.
        
        Args:
            pos_logits: Positive clause logits [half, n_features]
            neg_logits: Negative clause logits [half, n_features]
            
        Returns:
            (adapted_pos_logits, adapted_neg_logits)
        """
        adapted_pos = pos_logits
        adapted_neg = neg_logits
        
        if self.pos_lora_A is not None:
            adapted_pos = pos_logits + self.get_pos_delta()
        
        if self.neg_lora_A is not None:
            adapted_neg = neg_logits + self.get_neg_delta()
        
        return adapted_pos, adapted_neg


class LoRAVotingAdapter(nn.Module):
    """
    LoRA adapter for TM voting weights.
    
    Adapts the clause-to-class voting matrix using low-rank updates.
    
    Args:
        n_clauses: Number of clauses
        n_classes: Number of output classes
        rank: LoRA rank
        alpha: LoRA scaling factor
    """
    
    def __init__(
        self,
        n_clauses: int,
        n_classes: int,
        rank: int = 4,
        alpha: Optional[float] = None,
    ):
        super().__init__()
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, n_clauses))
        self.lora_B = nn.Parameter(torch.zeros(n_classes, rank))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def get_delta(self) -> torch.Tensor:
        """Get delta for voting weights."""
        return (self.lora_B @ self.lora_A) * self.scaling
    
    def forward(self, voting: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adaptation to voting weights.
        
        Args:
            voting: Original voting weights [n_clauses, n_classes]
            
        Returns:
            Adapted voting weights [n_clauses, n_classes]
        """
        delta = self.get_delta()  # [n_classes, n_clauses]
        return voting + delta.t()  # [n_clauses, n_classes]


# =============================================================================
# LoRA-Wrapped TM Models
# =============================================================================


class LoRAClauseMachine(nn.Module):
    """
    Tsetlin Machine with LoRA adaptation.
    
    Wraps a base TM and adds LoRA adapters for efficient fine-tuning.
    Base weights are frozen; only LoRA parameters are trainable.
    
    Args:
        base_tm: Base Tsetlin Machine model
        rank: LoRA rank for all adapters
        alpha: LoRA scaling factor
        adapt_clauses: Whether to adapt clause masks
        adapt_voting: Whether to adapt voting weights
    """
    
    def __init__(
        self,
        base_tm: Union[FuzzyPatternTM_STCM, FuzzyPatternTM_STE],
        rank: int = 4,
        alpha: Optional[float] = None,
        adapt_clauses: bool = True,
        adapt_voting: bool = True,
    ):
        super().__init__()
        self.base_tm = base_tm
        self.rank = rank
        
        # Freeze base model
        for param in base_tm.parameters():
            param.requires_grad = False
        
        # Add LoRA adapters
        if adapt_clauses:
            self.clause_adapter = LoRAClauseAdapter(
                n_clauses=base_tm.n_clauses,
                n_features=base_tm.n_features,
                rank=rank,
                alpha=alpha,
            )
        else:
            self.clause_adapter = None
        
        if adapt_voting:
            voting_shape = base_tm.voting.shape if hasattr(base_tm, 'voting') and base_tm.voting is not None else None
            if voting_shape is not None:
                self.voting_adapter = LoRAVotingAdapter(
                    n_clauses=base_tm.n_clauses,
                    n_classes=base_tm.n_classes,
                    rank=rank,
                    alpha=alpha,
                )
            else:
                self.voting_adapter = None
        else:
            self.voting_adapter = None
        
        self.merged = False
    
    @property
    def n_clauses(self) -> int:
        return self.base_tm.n_clauses
    
    @property
    def n_classes(self) -> int:
        return self.base_tm.n_classes
    
    @property
    def n_features(self) -> int:
        return self.base_tm.n_features
    
    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
        skip_norm: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with LoRA adaptation.
        
        Args:
            x: Input tensor
            use_ste: Use STE for base TM
            skip_norm: Skip input normalization
            
        Returns:
            (logits, clause_outputs)
        """
        if self.merged:
            return self.base_tm(x, use_ste=use_ste, skip_norm=skip_norm)
        
        # Get base clause outputs
        logits, clauses = self.base_tm(x, use_ste=use_ste, skip_norm=skip_norm)
        
        # Apply clause adaptation if present
        if self.clause_adapter is not None:
            # For STCM, we adapt the logits directly
            if hasattr(self.base_tm, 'pos_logits'):
                pos_delta = self.clause_adapter.get_pos_delta()
                neg_delta = self.clause_adapter.get_neg_delta()
                
                if pos_delta is not None or neg_delta is not None:
                    # Re-run with adapted logits
                    # This is a simplified version - full implementation would
                    # modify the computation
                    pass
        
        # Apply voting adaptation if present
        if self.voting_adapter is not None and hasattr(self.base_tm, 'voting'):
            adapted_voting = self.voting_adapter(self.base_tm.voting)
            
            # Recompute logits with adapted voting
            if hasattr(self.base_tm, 'clause_bias'):
                biased = clauses + self.base_tm.clause_bias.view(1, -1)
            else:
                biased = clauses
            logits = biased @ adapted_voting
        
        return logits, clauses
    
    def merge_weights(self) -> None:
        """Merge LoRA weights into base model for inference."""
        if self.merged:
            return
        
        with torch.no_grad():
            # Merge voting adapter
            if self.voting_adapter is not None and hasattr(self.base_tm, 'voting'):
                delta = self.voting_adapter.get_delta()
                self.base_tm.voting.data += delta.t()
            
            # Merge clause adapter
            if self.clause_adapter is not None:
                if hasattr(self.base_tm, 'pos_logits'):
                    pos_delta = self.clause_adapter.get_pos_delta()
                    neg_delta = self.clause_adapter.get_neg_delta()
                    
                    if pos_delta is not None:
                        self.base_tm.pos_logits.data += pos_delta
                    if neg_delta is not None:
                        self.base_tm.neg_logits.data += neg_delta
        
        self.merged = True
    
    def unmerge_weights(self) -> None:
        """Unmerge LoRA weights from base model."""
        if not self.merged:
            return
        
        with torch.no_grad():
            # Unmerge voting adapter
            if self.voting_adapter is not None and hasattr(self.base_tm, 'voting'):
                delta = self.voting_adapter.get_delta()
                self.base_tm.voting.data -= delta.t()
            
            # Unmerge clause adapter
            if self.clause_adapter is not None:
                if hasattr(self.base_tm, 'pos_logits'):
                    pos_delta = self.clause_adapter.get_pos_delta()
                    neg_delta = self.clause_adapter.get_neg_delta()
                    
                    if pos_delta is not None:
                        self.base_tm.pos_logits.data -= pos_delta
                    if neg_delta is not None:
                        self.base_tm.neg_logits.data -= neg_delta
        
        self.merged = False
    
    def get_lora_params(self) -> List[nn.Parameter]:
        """Get only LoRA parameters (for optimizer)."""
        params = []
        
        if self.clause_adapter is not None:
            if self.clause_adapter.pos_lora_A is not None:
                params.extend([self.clause_adapter.pos_lora_A, self.clause_adapter.pos_lora_B])
            if self.clause_adapter.neg_lora_A is not None:
                params.extend([self.clause_adapter.neg_lora_A, self.clause_adapter.neg_lora_B])
        
        if self.voting_adapter is not None:
            params.extend([self.voting_adapter.lora_A, self.voting_adapter.lora_B])
        
        return params
    
    def lora_param_count(self) -> int:
        """Count number of LoRA parameters."""
        return sum(p.numel() for p in self.get_lora_params())
    
    def base_param_count(self) -> int:
        """Count number of base parameters."""
        return sum(p.numel() for p in self.base_tm.parameters())


# =============================================================================
# Multi-Task LoRA
# =============================================================================


class MultiTaskLoRAClauseMachine(nn.Module):
    """
    Multi-task LoRA for Tsetlin Machines.
    
    Maintains separate LoRA adapters for each task, allowing
    task-specific adaptation while sharing the base model.
    
    Args:
        base_tm: Base Tsetlin Machine model
        rank: LoRA rank for all adapters
        alpha: LoRA scaling factor
    """
    
    def __init__(
        self,
        base_tm: Union[FuzzyPatternTM_STCM, FuzzyPatternTM_STE],
        rank: int = 4,
        alpha: Optional[float] = None,
    ):
        super().__init__()
        self.base_tm = base_tm
        self.rank = rank
        self.alpha = alpha
        
        # Freeze base model
        for param in base_tm.parameters():
            param.requires_grad = False
        
        # Task-specific adapters
        self.task_adapters: Dict[str, nn.ModuleDict] = nn.ModuleDict()
        
        self.current_task: Optional[str] = None
    
    def add_task(self, task_id: str) -> None:
        """
        Add a new task with its own LoRA adapters.
        
        Args:
            task_id: Unique task identifier
        """
        if task_id in self.task_adapters:
            return
        
        adapters = nn.ModuleDict({
            "clause": LoRAClauseAdapter(
                n_clauses=self.base_tm.n_clauses,
                n_features=self.base_tm.n_features,
                rank=self.rank,
                alpha=self.alpha,
            ),
            "voting": LoRAVotingAdapter(
                n_clauses=self.base_tm.n_clauses,
                n_classes=self.base_tm.n_classes,
                rank=self.rank,
                alpha=self.alpha,
            ),
        })
        
        self.task_adapters[task_id] = adapters
        self.current_task = task_id
    
    def set_task(self, task_id: str) -> None:
        """Set the current active task."""
        if task_id not in self.task_adapters:
            raise ValueError(f"Task {task_id} not found. Call add_task first.")
        self.current_task = task_id
    
    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[str] = None,
        use_ste: bool = True,
        skip_norm: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with task-specific LoRA.
        
        Args:
            x: Input tensor
            task_id: Task to use (default: current_task)
            use_ste: Use STE for base TM
            skip_norm: Skip input normalization
            
        Returns:
            (logits, clause_outputs)
        """
        task = task_id or self.current_task
        
        # Base forward
        logits, clauses = self.base_tm(x, use_ste=use_ste, skip_norm=skip_norm)
        
        if task is None or task not in self.task_adapters:
            return logits, clauses
        
        adapters = self.task_adapters[task]
        
        # Apply voting adaptation
        if "voting" in adapters and hasattr(self.base_tm, 'voting'):
            adapted_voting = adapters["voting"](self.base_tm.voting)
            
            if hasattr(self.base_tm, 'clause_bias'):
                biased = clauses + self.base_tm.clause_bias.view(1, -1)
            else:
                biased = clauses
            logits = biased @ adapted_voting
        
        return logits, clauses
    
    def get_task_params(self, task_id: str) -> List[nn.Parameter]:
        """Get LoRA parameters for a specific task."""
        if task_id not in self.task_adapters:
            return []
        
        params = []
        for adapter in self.task_adapters[task_id].values():
            params.extend(adapter.parameters())
        return params


# =============================================================================
# Utility Functions
# =============================================================================


def merge_lora_weights(
    model: LoRAClauseMachine,
    copy: bool = True,
) -> Union[FuzzyPatternTM_STCM, FuzzyPatternTM_STE]:
    """
    Merge LoRA weights into base model and return merged model.
    
    Args:
        model: LoRA-wrapped model
        copy: If True, return a copy; if False, modify in place
        
    Returns:
        Merged base model
    """
    if copy:
        import copy as copy_module
        model = copy_module.deepcopy(model)
    
    model.merge_weights()
    return model.base_tm


def add_lora_to_module(
    module: nn.Module,
    rank: int = 4,
    alpha: Optional[float] = None,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Add LoRA adapters to linear layers in a module.
    
    Args:
        module: Module to adapt
        rank: LoRA rank
        alpha: LoRA scaling factor
        target_modules: List of module names to adapt (None = all linear)
        
    Returns:
        Module with LoRA adapters
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if target_modules is None or name in target_modules:
                setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
        else:
            add_lora_to_module(child, rank, alpha, target_modules)
    
    return module


def count_lora_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and LoRA parameters in a model.
    
    Args:
        model: Model with LoRA adapters
        
    Returns:
        (total_params, lora_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total, trainable




