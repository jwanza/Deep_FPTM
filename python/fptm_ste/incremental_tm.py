"""
Incremental Tsetlin Machine (IncrementalSTCM)

This module implements true incremental learning for STCM, mirroring the Julia
FuzzyPatternTM implementation's key mechanisms:

1. Tsetlin Automaton State Machine: Discrete states (0-255) with gradual transitions
2. Probability-Gated Updates: Clauses update with probability based on margin
3. Sparse Random Exploration: Random feature probing when clauses don't match
4. Type I/II Feedback: Controlled reinforcement and suppression
5. Per-Sample Training: Optional sample-level updates for true incremental learning

Reference: FuzzyPatternTM.jl (src/FuzzyPatternTM.jl)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tm import FuzzyPatternTM_STCM, prepare_tm_input, _ste_ternary


@dataclass
class IncrementalConfig:
    """Configuration for incremental learning behavior."""
    
    # Automaton state machine
    states_num: int = 256  # Total number of automaton states
    include_limit: int = 128  # Threshold for literal inclusion
    
    # Feedback control (matching Julia T, S parameters)
    T: float = 15.0  # Threshold for vote summation
    S: float = 10.0  # Sparsity denominator (s = n_features / S)
    
    # Literal budget (matching Julia L, LF)
    L: int = 16  # Max literals per clause
    LF: int = 4  # Early termination threshold
    
    # Update behavior
    use_probabilistic_updates: bool = True  # Gate updates by probability
    use_sparse_exploration: bool = True  # Random feature exploration
    exploration_decay: float = 0.999  # Decay for exploration rate
    
    # EMA stabilization
    use_ema: bool = True
    ema_decay: float = 0.995  # Higher = more stable
    
    # Gradient blending (for hybrid training)
    gradient_blend: float = 0.3  # 0=pure incremental, 1=pure gradient


class TsetlinAutomaton(nn.Module):
    """
    Tsetlin Automaton state machine for literal inclusion decisions.
    
    Maintains integer states in [0, states_num-1] for each literal position.
    Literals are included when state >= include_limit.
    
    State transitions:
    - Reinforce: state += 1 (clamped to max)
    - Suppress: state -= 1 (clamped to min)
    
    This provides stable, gradual pattern consolidation unlike
    continuous gradient updates.
    """
    
    def __init__(
        self,
        n_clauses: int,
        n_features: int,
        states_num: int = 256,
        include_limit: int = 128,
    ):
        super().__init__()
        self.n_clauses = n_clauses
        self.n_features = n_features
        self.states_num = states_num
        self.include_limit = include_limit
        
        half = n_clauses // 2
        
        # Initialize states just below include threshold (like Julia)
        # Shape: [half, n_features] for each of 4 banks
        initial_state = include_limit - 1
        
        # Register as buffers (not parameters, updated manually)
        self.register_buffer('pos_states', torch.full((half, n_features), initial_state, dtype=torch.int16))
        self.register_buffer('neg_states', torch.full((half, n_features), initial_state, dtype=torch.int16))
        self.register_buffer('pos_inv_states', torch.full((half, n_features), initial_state, dtype=torch.int16))
        self.register_buffer('neg_inv_states', torch.full((half, n_features), initial_state, dtype=torch.int16))
        
        # EMA shadow for state transitions
        self.register_buffer('_update_count', torch.tensor(0, dtype=torch.long))
        
    def get_inclusion_masks(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert automaton states to binary inclusion masks.
        
        Returns:
            Tuple of (pos_mask, neg_mask, pos_inv_mask, neg_inv_mask)
            Each is [half, n_features] with 1.0 for included literals
        """
        return (
            (self.pos_states >= self.include_limit).float(),
            (self.neg_states >= self.include_limit).float(),
            (self.pos_inv_states >= self.include_limit).float(),
            (self.neg_inv_states >= self.include_limit).float(),
        )
    
    def get_soft_masks(self, temperature: float = 1.0) -> Tuple[torch.Tensor, ...]:
        """
        Get soft (differentiable) masks based on distance from threshold.
        
        Uses sigmoid with temperature to create gradient-friendly masks
        while maintaining alignment with discrete automaton states.
        """
        def soft_mask(states):
            # Distance from threshold, scaled by temperature
            distance = (states.float() - self.include_limit) / temperature
            return torch.sigmoid(distance)
        
        return (
            soft_mask(self.pos_states),
            soft_mask(self.neg_states),
            soft_mask(self.pos_inv_states),
            soft_mask(self.neg_inv_states),
        )
    
    @torch.no_grad()
    def reinforce(
        self,
        bank: str,
        clause_idx: int,
        feature_mask: torch.Tensor,
        amount: int = 1,
    ) -> None:
        """
        Reinforce (strengthen) specific literals.
        
        Args:
            bank: One of 'pos', 'neg', 'pos_inv', 'neg_inv'
            clause_idx: Index of clause to update
            feature_mask: Boolean mask of features to reinforce
            amount: Amount to increase state (default 1)
        """
        states = getattr(self, f'{bank}_states')
        mask = feature_mask.bool()
        states[clause_idx, mask] = torch.clamp(
            states[clause_idx, mask] + amount,
            max=self.states_num - 1
        )
        
    @torch.no_grad()
    def suppress(
        self,
        bank: str,
        clause_idx: int,
        feature_mask: torch.Tensor,
        amount: int = 1,
    ) -> None:
        """
        Suppress (weaken) specific literals.
        
        Args:
            bank: One of 'pos', 'neg', 'pos_inv', 'neg_inv'
            clause_idx: Index of clause to update
            feature_mask: Boolean mask of features to suppress
            amount: Amount to decrease state (default 1)
        """
        states = getattr(self, f'{bank}_states')
        mask = feature_mask.bool()
        states[clause_idx, mask] = torch.clamp(
            states[clause_idx, mask] - amount,
            min=0
        )
    
    @torch.no_grad()
    def batch_apply_updates(
        self,
        bank: str,
        increments: Optional[torch.Tensor] = None,
        decrements: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Apply batched updates to automaton states.
        
        Args:
            bank: One of 'pos', 'neg', 'pos_inv', 'neg_inv'
            increments: Tensor [half, features] of counts to add
            decrements: Tensor [half, features] of counts to subtract
        """
        states = getattr(self, f'{bank}_states')
        
        if increments is not None:
            states.add_(increments)
            
        if decrements is not None:
            states.sub_(decrements)
            
        # Clamp to valid range
        states.clamp_(0, self.states_num - 1)

    @torch.no_grad()
    def batch_sparse_explore(
        self,
        bank: str,
        explore_counts: torch.Tensor,
        n_positions: int,
    ) -> None:
        """
        Batched sparse exploration.
        
        Args:
            bank: Bank to explore
            explore_counts: Tensor [half] of exploration events per clause
            n_positions: Number of random positions per event
        """
        states = getattr(self, f'{bank}_states')
        half, n_features = states.shape
        
        # We need to decrement random positions.
        # Total decrements per clause = explore_counts * n_positions
        total_decrements = explore_counts * n_positions
        
        # Optimization: Instead of exact loop, use probabilistic decay
        # Probability of any feature being decayed = total_decrements / n_features
        # This approximates the "random selection" over the batch
        
        probs = (total_decrements.float() / n_features).clamp(0.0, 1.0)
        
        # Generate mask: 1 where we should decrement
        # Shape [half, n_features]
        # We broadcast probs [half] to [half, n_features]
        decay_mask = torch.rand_like(states, dtype=torch.float) < probs.unsqueeze(1)
        
        # Apply decay only where state > 0
        active_mask = states > 0
        mask = decay_mask & active_mask
        
        if mask.any():
            states.sub_(mask.long())


class IncrementalSTCM(nn.Module):
    """
    Incremental Setun-Ternary Clause Machine with true incremental learning.
    
    This implementation combines:
    1. Tsetlin Automaton for stable state transitions
    2. Differentiable path for gradient-based fine-tuning
    3. Probability-gated feedback matching Julia behavior
    4. Sparse exploration for escaping local optima
    
    The key insight is that Julia's TM achieves stable incremental learning
    through discrete automaton states and probabilistic clause updates,
    not continuous gradient descent.
    
    Args:
        n_features: Number of input features
        n_clauses: Number of clauses (must be even)
        n_classes: Number of output classes
        config: IncrementalConfig with learning parameters
        tau: STE threshold (for gradient path)
        operator: 'capacity' or 'product' clause evaluation
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        config: Optional[IncrementalConfig] = None,
        tau: float = 0.5,
        *,
        input_shape: Optional[Tuple[int, int, int]] = None,
        operator: str = "capacity",
        clause_dropout: float = 0.0,
        literal_dropout: float = 0.0,
    ):
        super().__init__()
        
        if n_clauses % 2 != 0:
            raise ValueError("n_clauses must be even")
            
        self.config = config or IncrementalConfig()
        self.n_features = n_features
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.tau = tau
        self.input_shape = tuple(input_shape) if input_shape else None
        self.operator = operator
        self.clause_dropout = clause_dropout
        self.literal_dropout = literal_dropout
        
        half = n_clauses // 2
        
        # Tsetlin Automaton for discrete state tracking
        self.automaton = TsetlinAutomaton(
            n_clauses=n_clauses,
            n_features=n_features,
            states_num=self.config.states_num,
            include_limit=self.config.include_limit,
        )
        
        # Differentiable parameters (for gradient path)
        # These shadow the automaton states for hybrid training
        self.pos_logits = nn.Parameter(torch.zeros(half, n_features))
        self.neg_logits = nn.Parameter(torch.zeros(half, n_features))
        
        # Voting weights
        self.voting = nn.Parameter(torch.randn(n_clauses, n_classes) * 0.1)
        self.clause_bias = nn.Parameter(torch.zeros(n_clauses))
        
        # EMA shadow parameters
        if self.config.use_ema:
            self.register_buffer('ema_pos_logits', torch.zeros(half, n_features))
            self.register_buffer('ema_neg_logits', torch.zeros(half, n_features))
            self.register_buffer('ema_voting', torch.zeros(n_clauses, n_classes))
        
        # Compute sparsity parameter (s = n_features / S)
        self.s = max(1, int(round(n_features / self.config.S)))
        
        # Product operator scale
        self.product_scale = 4.0 / n_features
        
        # Training state
        self._current_sample_idx = 0
        self._exploration_rate = 1.0
        self._last_clause_activity: Optional[torch.Tensor] = None
        self._feedback_stats: Dict[str, float] = {}
        
        # Sync automaton states to differentiable parameters
        self._sync_automaton_to_params()
    
    def _sync_automaton_to_params(self) -> None:
        """
        Synchronize automaton states to differentiable parameters.
        
        Maps discrete states [0, 255] to logits such that:
        - state < include_limit -> negative logit (excluded)
        - state >= include_limit -> positive logit (included)
        """
        with torch.no_grad():
            pos_mask, neg_mask, pos_inv_mask, neg_inv_mask = self.automaton.get_inclusion_masks()
            
            # Convert masks to ternary logits
            # Positive: require x=1, Negative: require x=0
            # Combined: logit > 0 means require 1, logit < 0 means require 0
            pos_ternary = pos_mask - pos_inv_mask  # +1 for require 1, -1 for require 0, 0 for don't care
            neg_ternary = neg_mask - neg_inv_mask
            
            # Scale to create proper logit distribution
            self.pos_logits.data = pos_ternary * 2.0
            self.neg_logits.data = neg_ternary * 2.0
    
    def _sync_params_to_automaton(self) -> None:
        """
        Synchronize differentiable parameters back to automaton states.
        
        Used after gradient updates to maintain consistency.
        """
        with torch.no_grad():
            half = self.n_clauses // 2
            
            for bank_idx, (logits, inv_bank) in enumerate([
                (self.pos_logits, 'pos_inv'),
                (self.neg_logits, 'neg_inv'),
            ]):
                bank = 'pos' if bank_idx == 0 else 'neg'
                
                for j in range(half):
                    for i in range(self.n_features):
                        logit = logits[j, i].item()
                        
                        # Update automaton based on logit sign and magnitude
                        if logit > self.tau:
                            # Want to include as positive literal
                            states = getattr(self.automaton, f'{bank}_states')
                            if states[j, i] < self.config.include_limit:
                                states[j, i] = min(self.config.states_num - 1, states[j, i] + 1)
                        elif logit < -self.tau:
                            # Want to include as inverted literal
                            states = getattr(self.automaton, f'{inv_bank}_states')
                            if states[j, i] < self.config.include_limit:
                                states[j, i] = min(self.config.states_num - 1, states[j, i] + 1)
    
    def _get_masks(self, use_automaton: bool = True) -> Tuple[torch.Tensor, ...]:
        """
        Get literal inclusion masks.
        
        Args:
            use_automaton: If True, use discrete automaton states.
                          If False, use differentiable logits.
        """
        if use_automaton:
            return self.automaton.get_inclusion_masks()
        else:
            # STE-style differentiable masks from logits
            half = self.n_clauses // 2
            
            pos_ternary = _ste_ternary(self.pos_logits, 0.0, 1.0)
            neg_ternary = _ste_ternary(self.neg_logits, 0.0, 1.0)
            
            return (
                torch.clamp(pos_ternary, min=0.0),  # pos_pos
                torch.clamp(neg_ternary, min=0.0),  # neg_pos
                torch.clamp(-pos_ternary, min=0.0),  # pos_inv
                torch.clamp(-neg_ternary, min=0.0),  # neg_inv
            )
    
    def _clause_capacity(self, mask_pos: torch.Tensor, mask_inv: torch.Tensor) -> torch.Tensor:
        """Compute clause capacity (number of included literals)."""
        included = mask_pos.sum(dim=1) + mask_inv.sum(dim=1)
        
        # Apply LF constraint
        if self.config.LF > 0:
            included = torch.minimum(
                included,
                torch.tensor(float(self.config.LF), device=included.device)
            )
        
        # Apply L constraint
        if self.config.L > 0:
            included = torch.minimum(
                included,
                torch.tensor(float(self.config.L), device=included.device)
            )
            
        return included.unsqueeze(0)
    
    def _clause_strength(
        self,
        x: torch.Tensor,
        mask_pos: torch.Tensor,
        mask_inv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute clause strength using capacity-mismatch formulation.
        
        This matches Julia's check_clause behavior:
        - Count mismatches (x[i]=0 where pos[i]=1, x[i]=1 where inv[i]=1)
        - Strength = max(0, capacity - mismatch)
        """
        # Optimized projection: W_eff = mask_pos - mask_inv
        W_eff = mask_pos - mask_inv
        projection = F.linear(x, W_eff)  # [B, half]
        mismatch_bias = mask_pos.sum(dim=1).unsqueeze(0)  # [1, half]
        mismatch = mismatch_bias - projection  # [B, half]
        
        if self.operator == "capacity":
            capacity = self._clause_capacity(mask_pos, mask_inv)
            raw = capacity - mismatch
            # Leaky ReLU for gradient flow
            return F.leaky_relu(raw, negative_slope=0.01)
        else:  # product
            scaled = torch.clamp(mismatch * self.product_scale, min=0.0, max=10.0)
            return torch.exp(-scaled)
    
    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
        skip_norm: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, features] or image tensor
            use_ste: Use STE for gradient path (True) or pure automaton (False)
            skip_norm: Skip input normalization
            
        Returns:
            (logits, clause_outputs) tuple
        """
        if not skip_norm:
            x = prepare_tm_input(
                x,
                n_features=self.n_features,
                input_shape=self.input_shape,
            )
        
        if self.training and self.literal_dropout > 0:
            x = F.dropout(x, p=self.literal_dropout, training=True)
        
        half = self.n_clauses // 2
        
        # Get masks (automaton for inference, differentiable for training)
        if use_ste and self.training:
            pos_pos, neg_pos, pos_inv, neg_inv = self._get_masks(use_automaton=False)
        else:
            pos_pos, neg_pos, pos_inv, neg_inv = self._get_masks(use_automaton=True)
        
        # Compute clause strengths
        pos_strength = self._clause_strength(x, pos_pos, pos_inv)
        neg_strength = self._clause_strength(x, neg_pos, neg_inv)
        
        # Combine: positive clauses add, negative clauses subtract
        clause_outputs = torch.cat([pos_strength, -neg_strength], dim=1)
        
        if self.training and self.clause_dropout > 0:
            clause_outputs = F.dropout(clause_outputs, p=self.clause_dropout, training=True)
        
        # Store for feedback computation
        self._last_clause_activity = clause_outputs.detach().abs().mean(dim=0)
        
        # Voting
        biased = clause_outputs + self.clause_bias.view(1, -1)
        
        # Use EMA voting if available and in eval mode
        if self.config.use_ema and not self.training:
            voting = self.ema_voting
        else:
            voting = self.voting
            
        logits = biased @ voting
        
        return logits, clause_outputs
    
    @torch.no_grad()
    def incremental_feedback(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        clause_outputs: torch.Tensor,
        logits: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Apply incremental Tsetlin Machine feedback (Vectorized).
        
        This implements the Julia feedback mechanism efficiently:
        1. Compute update probability based on vote margin
        2. Vectorized Type I/II feedback updates
        3. Batched sparse exploration
        
        Args:
            x: Input features [batch, features] (assumed binary/float in [0,1])
            y: Target labels [batch] (ignored in this simplified single-bank implementation
               except for potential future class-specific routing)
            clause_outputs: Clause activation outputs [batch, n_clauses]
            logits: Classification logits [batch, n_classes]
            
        Returns:
            Dict with feedback statistics
        """
        batch_size = x.shape[0]
        half = self.n_clauses // 2
        device = x.device
        
        # --- 1. Vote & Probability Calculation ---
        
        # Compute vote for 'correct' class concept
        # (Assuming clauses 0..half are positive, half..end are negative)
        pos_sum = clause_outputs[:, :half].sum(dim=1)
        neg_sum = clause_outputs[:, half:].abs().sum(dim=1)
        vote = pos_sum - neg_sum
        
        # Clamp vote to [-T, T]
        T = self.config.T
        vote = torch.clamp(vote, min=-T, max=T)
        
        # Update probabilities
        # Positive feedback (for correct class prediction)
        update_prob_pos = (T - vote) / (2 * T)
        # Negative feedback (for incorrect class prediction / negative clauses)
        update_prob_neg = (T + vote) / (2 * T)
        
        # --- 2. Gating Masks ---
        
        # Random gating based on probabilities
        if self.config.use_probabilistic_updates:
            # Generate random values [B, half]
            rand_pos = torch.rand(batch_size, half, device=device)
            rand_neg = torch.rand(batch_size, half, device=device)
            
            mask_pos_gate = rand_pos < update_prob_pos.unsqueeze(1)
            mask_neg_gate = rand_neg < update_prob_neg.unsqueeze(1)
        else:
            mask_pos_gate = torch.ones(batch_size, half, dtype=torch.bool, device=device)
            mask_neg_gate = torch.ones(batch_size, half, dtype=torch.bool, device=device)
            
        # Clause activity masks
        pos_clause_active = clause_outputs[:, :half] > 0
        neg_clause_active = clause_outputs[:, half:].abs() > 0
        
        # Combined Update Masks: Gate AND Active
        M_pos = mask_pos_gate & pos_clause_active      # [B, half]
        M_neg = mask_neg_gate & neg_clause_active      # [B, half]
        
        # --- 3. Type I Feedback (Positive Clauses) ---
        
        # Ensure x is binary-like for matrix mult
        x_float = (x > 0.5).float()
        x_inv_float = 1.0 - x_float
        
        M_pos_float = M_pos.float()
        
        # Calculate increments (Reinforce)
        # Reinforce 'pos' where x=1: sum over batch(M_pos[b,j] * x[b,k])
        inc_pos = M_pos_float.T @ x_float         # [half, features]
        # Reinforce 'pos_inv' where x=0
        inc_pos_inv = M_pos_float.T @ x_inv_float # [half, features]
        
        # Apply Literal Budget Constraint to Reinforcements
        # Only reinforce if clause literal count <= L
        with torch.no_grad():
            pos_mask, _, pos_inv_mask, _ = self.automaton.get_inclusion_masks()
            n_literals = pos_mask.sum(dim=1) + pos_inv_mask.sum(dim=1)
            budget_mask = (n_literals <= self.config.L).float().unsqueeze(1) # [half, 1]
            
            inc_pos *= budget_mask
            inc_pos_inv *= budget_mask
            
        # Calculate decrements (Suppress)
        # Suppress 'pos' where x=0 (mismatch)
        # Condition (state < limit) is handled by masking the decrements later
        dec_pos = M_pos_float.T @ x_inv_float
        # Suppress 'pos_inv' where x=1
        dec_pos_inv = M_pos_float.T @ x_float
        
        # Apply Type I Updates
        # Note: Suppress only if state < limit (handled by masking in call or here)
        # We handle it here for efficiency
        
        def apply_conditional_suppress(bank, dec_counts):
            states = getattr(self.automaton, f'{bank}_states')
            below_limit = states < self.config.include_limit
            # Only apply decrements where state < limit
            valid_dec = dec_counts * below_limit.float()
            # If any valid decrements, apply
            if valid_dec.sum() > 0:
                self.automaton.batch_apply_updates(bank, decrements=valid_dec.long())

        self.automaton.batch_apply_updates('pos', increments=inc_pos.long())
        apply_conditional_suppress('pos', dec_pos)
        
        self.automaton.batch_apply_updates('pos_inv', increments=inc_pos_inv.long())
        apply_conditional_suppress('pos_inv', dec_pos_inv)
        
        # --- 4. Sparse Exploration (Positive Clauses) ---
        
        if self.config.use_sparse_exploration:
            # Explore if gated BUT NOT active
            M_explore = mask_pos_gate & (~pos_clause_active)
            explore_counts = M_explore.sum(dim=0) # [half]
            
            if explore_counts.sum() > 0:
                self.automaton.batch_sparse_explore('pos', explore_counts, self.s)
                self.automaton.batch_sparse_explore('pos_inv', explore_counts, self.s)
        
        # --- 5. Type II Feedback (Negative Clauses) ---
        
        M_neg_float = M_neg.float()
        
        # Reinforce 'neg' where x=0 (to exclude 0s)
        inc_neg = M_neg_float.T @ x_inv_float
        # Reinforce 'neg_inv' where x=1 (to exclude 1s)
        inc_neg_inv = M_neg_float.T @ x_float
        
        self.automaton.batch_apply_updates('neg', increments=inc_neg.long())
        self.automaton.batch_apply_updates('neg_inv', increments=inc_neg_inv.long())
        
        # --- 6. Stats & Cleanup ---
        
        # Decay exploration rate
        self._exploration_rate *= self.config.exploration_decay
        
        # Sync automaton to parameters for gradient path
        self._sync_automaton_to_params()
        
        stats = {
            'type1_updates': M_pos.float().sum().item(),
            'type2_updates': M_neg.float().sum().item(),
            'explorations': M_explore.float().sum().item() if self.config.use_sparse_exploration else 0,
            'clauses_updated': (M_pos.float().sum() + M_neg.float().sum()).item(),
        }
        
        self._feedback_stats = stats
        return stats
    
    def update_ema(self) -> None:
        """Update EMA shadow parameters."""
        if not self.config.use_ema:
            return
            
        decay = self.config.ema_decay
        with torch.no_grad():
            self.ema_pos_logits.mul_(decay).add_(self.pos_logits.data, alpha=1 - decay)
            self.ema_neg_logits.mul_(decay).add_(self.neg_logits.data, alpha=1 - decay)
            self.ema_voting.mul_(decay).add_(self.voting.data, alpha=1 - decay)
    
    def get_feedback_stats(self) -> Dict[str, float]:
        """Return last feedback statistics."""
        return self._feedback_stats.copy()
    
    def clause_activity(self) -> Optional[torch.Tensor]:
        """Return last clause activity tensor."""
        return self._last_clause_activity
    
    def extra_repr(self) -> str:
        return (
            f"features={self.n_features}, clauses={self.n_clauses}, "
            f"classes={self.n_classes}, T={self.config.T}, S={self.config.S}, "
            f"L={self.config.L}, LF={self.config.LF}"
        )


def incremental_train_step(
    model: IncrementalSTCM,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: Optional[torch.optim.Optimizer] = None,
    use_gradient: bool = True,
    gradient_weight: float = 0.5,
    clip_grad: float = 1.0,
) -> Dict[str, float]:
    """
    Hybrid incremental training step.
    
    Combines:
    1. Incremental automaton feedback (Julia-style)
    2. Gradient-based optimization (for voting weights)
    
    Args:
        model: IncrementalSTCM model
        x: Input batch [batch, features]
        y: Labels [batch]
        optimizer: Optional optimizer for gradient updates
        use_gradient: Whether to also apply gradient updates
        gradient_weight: Weight for gradient loss (0-1)
        clip_grad: Gradient clipping value
        
    Returns:
        Dict with training statistics
    """
    model.train()
    
    stats = {'loss': 0.0, 'accuracy': 0.0}
    
    # Forward pass
    logits, clause_outputs = model(x, use_ste=use_gradient)
    
    # Compute loss for gradient path
    if use_gradient and optimizer is not None:
        optimizer.zero_grad()
        loss = F.cross_entropy(logits, y)
        loss.backward()
        
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        # Scale gradients by gradient_weight
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param.grad *= gradient_weight
        
        optimizer.step()
        stats['loss'] = loss.item()
    
    # Apply incremental feedback (the key part!)
    feedback_stats = model.incremental_feedback(x, y, clause_outputs.detach(), logits.detach())
    stats.update(feedback_stats)
    
    # Update EMA
    model.update_ema()
    
    # Compute accuracy
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        stats['accuracy'] = (preds == y).float().mean().item()
    
    return stats


def incremental_train_epoch(
    model: IncrementalSTCM,
    dataloader,
    optimizer: Optional[torch.optim.Optimizer] = None,
    use_gradient: bool = True,
    gradient_weight: float = 0.5,
    device: torch.device = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Train for one epoch with incremental learning.
    
    Args:
        model: IncrementalSTCM model
        dataloader: Training data loader
        optimizer: Optional optimizer
        use_gradient: Use gradient updates
        gradient_weight: Gradient weight
        device: Device to use
        verbose: Print progress
        
    Returns:
        Epoch statistics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.train()
    
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    total_type1 = 0
    total_type2 = 0
    total_explore = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        stats = incremental_train_step(
            model, data, target,
            optimizer=optimizer,
            use_gradient=use_gradient,
            gradient_weight=gradient_weight,
        )
        
        batch_size = target.size(0)
        total_loss += stats['loss'] * batch_size
        total_acc += stats['accuracy'] * batch_size
        total_samples += batch_size
        total_type1 += stats.get('type1_updates', 0)
        total_type2 += stats.get('type2_updates', 0)
        total_explore += stats.get('explorations', 0)
        
        if verbose and batch_idx % 50 == 0:
            print(f"Batch {batch_idx}: loss={stats['loss']:.4f}, acc={stats['accuracy']:.4f}")
    
    return {
        'loss': total_loss / max(1, total_samples),
        'accuracy': total_acc / max(1, total_samples),
        'type1_updates': total_type1,
        'type2_updates': total_type2,
        'explorations': total_explore,
    }


class IncrementalDeepTM(nn.Module):
    """
    Deep TM Network with incremental learning layers.
    
    Stacks IncrementalSTCM layers with residual connections.
    Uses incremental feedback throughout the network.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        n_classes: int,
        n_clauses: int = 100,
        config: Optional[IncrementalConfig] = None,
        dropout: float = 0.1,
        tau: float = 0.5,
        *,
        input_shape: Optional[Tuple[int, int, int]] = None,
    ):
        super().__init__()
        
        self.config = config or IncrementalConfig()
        self.input_dim = input_dim
        self.input_shape = tuple(input_shape) if input_shape else None
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()
        
        prev = input_dim
        for h in hidden_dims:
            layer = IncrementalSTCM(
                n_features=prev,
                n_clauses=n_clauses,
                n_classes=h,
                config=self.config,
                tau=tau,
            )
            self.layers.append(layer)
            self.norms.append(nn.LayerNorm(h))
            self.residuals.append(
                nn.Linear(prev, h, bias=False) if prev != h else nn.Identity()
            )
            prev = h
        
        # Classifier head
        self.classifier = IncrementalSTCM(
            n_features=prev,
            n_clauses=n_clauses,
            n_classes=n_classes,
            config=self.config,
            tau=tau,
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through all layers."""
        
        # Flatten if needed
        if self.input_shape is not None:
            x = prepare_tm_input(x, n_features=self.input_dim, input_shape=self.input_shape)
        elif x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Process through layers
        for layer, norm, res in zip(self.layers, self.norms, self.residuals):
            identity = res(x)
            logits, _ = layer(x, use_ste=use_ste, skip_norm=True)
            x = norm(self.dropout(torch.sigmoid(logits)) + identity)
        
        # Final classifier
        logits, clauses = self.classifier(x, use_ste=use_ste, skip_norm=True)
        
        return logits, clauses
    
    def incremental_feedback_all_layers(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """
        Apply incremental feedback to all layers.
        
        Runs forward pass and applies feedback at each layer.
        """
        stats = {}
        
        # Prepare input
        if self.input_shape is not None:
            x = prepare_tm_input(x, n_features=self.input_dim, input_shape=self.input_shape)
        elif x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Forward and feedback through layers
        for i, (layer, norm, res) in enumerate(zip(self.layers, self.norms, self.residuals)):
            identity = res(x)
            logits, clause_outputs = layer(x, use_ste=False, skip_norm=True)
            
            # Apply feedback at this layer (using final labels as guidance)
            layer_stats = layer.incremental_feedback(x, y, clause_outputs, logits)
            stats[f'layer_{i}'] = layer_stats
            
            x = norm(self.dropout(torch.sigmoid(logits)) + identity)
        
        # Classifier feedback
        logits, clause_outputs = self.classifier(x, use_ste=False, skip_norm=True)
        classifier_stats = self.classifier.incremental_feedback(x, y, clause_outputs, logits)
        stats['classifier'] = classifier_stats
        
        return stats
    
    def update_all_ema(self) -> None:
        """Update EMA for all layers."""
        for layer in self.layers:
            layer.update_ema()
        self.classifier.update_ema()
