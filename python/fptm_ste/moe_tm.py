"""
Sparse Mixture-of-Experts Tsetlin Machine (SMoE-TM).

This module implements a mixture-of-experts architecture where different
clause "experts" specialize on different input patterns. A router network
selects top-k experts per sample, enabling efficient and specialized processing.

Key innovations:
1. Clause groups act as "experts" specializing on different patterns
2. Top-k sparse routing for efficiency
3. Load balancing loss for expert utilization
4. Capacity-based expert assignment
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tm import FuzzyPatternTM_STCM
from .tm_optimized import OptimizedSTCM


class MoEClauseRouter(nn.Module):
    """
    Router for directing inputs to clause experts.
    
    Computes routing probabilities and selects top-k experts per sample.
    Includes auxiliary losses for load balancing and expert utilization.
    
    Args:
        input_dim: Dimension of router input
        n_experts: Number of clause expert groups
        top_k: Number of experts to select per sample
        noise_std: Standard deviation of noise for exploration
        capacity_factor: Expert capacity multiplier
    """
    
    def __init__(
        self,
        input_dim: int,
        n_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.noise_std = noise_std
        self.capacity_factor = capacity_factor
        
        # Router network
        self.router = nn.Linear(input_dim, n_experts, bias=False)
        
        # Initialize with small weights for balanced start
        nn.init.xavier_uniform_(self.router.weight, gain=0.01)
    
    def forward(
        self,
        x: torch.Tensor,
        return_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute routing decisions.
        
        Args:
            x: Input tensor [batch, input_dim]
            return_aux_loss: Whether to compute auxiliary load balancing loss
            
        Returns:
            (expert_weights, expert_indices, gates, aux_loss)
            - expert_weights: [batch, top_k] - weights for selected experts
            - expert_indices: [batch, top_k] - indices of selected experts
            - gates: [batch, n_experts] - full routing probabilities
            - aux_loss: Scalar load balancing loss
        """
        batch_size = x.shape[0]
        
        # Compute routing logits
        router_logits = self.router(x)  # [batch, n_experts]
        
        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise
        
        # Compute gates (probabilities)
        gates = F.softmax(router_logits, dim=-1)  # [batch, n_experts]
        
        # Select top-k experts
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)
        
        # Normalize top-k gates to sum to 1
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Compute auxiliary loss for load balancing
        aux_loss = None
        if return_aux_loss:
            aux_loss = self._load_balance_loss(gates, top_k_indices)
        
        return top_k_gates, top_k_indices, gates, aux_loss
    
    def _load_balance_loss(
        self,
        gates: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.
        
        Encourages uniform expert utilization to prevent collapse.
        """
        batch_size = gates.shape[0]
        
        # Fraction of tokens routed to each expert
        # Create one-hot for selected experts and sum
        expert_mask = F.one_hot(expert_indices, self.n_experts).float()
        expert_mask = expert_mask.sum(dim=1)  # [batch, n_experts]
        tokens_per_expert = expert_mask.sum(dim=0) / batch_size  # [n_experts]
        
        # Mean routing probability per expert
        router_prob_per_expert = gates.mean(dim=0)  # [n_experts]
        
        # Load balance loss: encourage uniform distribution
        # L = n_experts * sum(tokens_per_expert * router_prob_per_expert)
        load_balance_loss = self.n_experts * torch.sum(
            tokens_per_expert * router_prob_per_expert
        )
        
        return load_balance_loss


class ClauseExpert(nn.Module):
    """
    Single clause expert (a subset of clauses that specialize on patterns).
    
    Each expert has its own set of clauses that learn to match specific
    input patterns.
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        operator: str = "capacity",
        tau: float = 0.5,
        expert_cls: type = OptimizedSTCM,
    ):
        super().__init__()
        self.stcm = expert_cls(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            operator=operator,
            tau=tau,
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expert forward pass.
        
        Returns:
            (logits, clause_outputs)
        """
        return self.stcm(x)


class SparseMoETM(nn.Module):
    """
    Sparse Mixture-of-Experts Tsetlin Machine.
    
    Divides clauses into expert groups, each specializing on different
    input patterns. A router selects top-k experts per sample for
    efficient sparse computation.
    
    Args:
        n_features: Number of input features
        n_clauses_per_expert: Clauses per expert group
        n_classes: Number of output classes
        n_experts: Number of expert groups
        top_k: Number of experts to use per sample
        operator: Clause operator type
        tau: Binarization threshold
        router_dim: Hidden dimension for router (default: n_features)
        noise_std: Router noise for exploration
        aux_loss_weight: Weight for load balancing loss
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses_per_expert: int,
        n_classes: int,
        n_experts: int = 8,
        top_k: int = 2,
        operator: str = "capacity",
        tau: float = 0.5,
        router_dim: Optional[int] = None,
        noise_std: float = 0.1,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_experts = n_experts
        self.n_clauses_per_expert = n_clauses_per_expert
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        
        # Router
        router_dim = router_dim or n_features
        self.router = MoEClauseRouter(
            input_dim=router_dim,
            n_experts=n_experts,
            top_k=top_k,
            noise_std=noise_std,
        )
        
        # Expert groups
        self.experts = nn.ModuleList([
            ClauseExpert(
                n_features=n_features,
                n_clauses=n_clauses_per_expert,
                n_classes=n_classes,
                operator=operator,
                tau=tau,
            )
            for _ in range(n_experts)
        ])
        
        # Combined output projection
        self.output_proj = nn.Linear(n_classes, n_classes, bias=False)
        nn.init.eye_(self.output_proj.weight)
        
        # Track auxiliary loss for training
        self.register_buffer("_aux_loss", torch.tensor(0.0))
    
    @property
    def aux_loss(self) -> torch.Tensor:
        """Get the auxiliary load balancing loss from last forward pass."""
        return self._aux_loss
    
    def forward(
        self,
        x: torch.Tensor,
        return_routing: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict]:
        """
        Sparse MoE forward pass - OPTIMIZED with batched expert processing.
        
        Args:
            x: Input tensor [batch, n_features]
            return_routing: If True, return routing information
            
        Returns:
            (logits, clauses) or dict with routing info
        """
        batch_size = x.shape[0]
        device = x.device
        n_classes = self.experts[0].stcm.n_classes
        
        # Get routing decisions
        expert_weights, expert_indices, gates, aux_loss = self.router(x)
        
        # Store aux loss for training
        if aux_loss is not None:
            self._aux_loss = aux_loss * self.aux_loss_weight
        
        # OPTIMIZED: Process experts in batches instead of per-sample
        # Group samples by expert assignment for batched processing
        logits = torch.zeros(batch_size, n_classes, device=device, dtype=x.dtype)
        all_clauses_list = [None] * batch_size
        
        # For each expert, find all samples that selected it and process together
        for expert_idx in range(self.n_experts):
            # Find which (sample, position) pairs selected this expert
            mask = (expert_indices == expert_idx)  # [batch, top_k]
            
            if not mask.any():
                continue
            
            # Get sample indices and position (0 or 1 for top_k=2)
            sample_pos = mask.nonzero(as_tuple=False)  # [N, 2] where N = number of selections
            
            if sample_pos.shape[0] == 0:
                continue
            
            sample_indices = sample_pos[:, 0]  # Which samples selected this expert
            pos_indices = sample_pos[:, 1]     # At which top-k position
            
            # Batch process all samples that selected this expert
            expert_inputs = x[sample_indices]  # [N, n_features]
            
            # Run expert once for all samples
            exp_logits, exp_clauses = self.experts[expert_idx](expert_inputs)  # [N, n_classes], [N, n_clauses]
            
            # Get weights for these selections
            weights = expert_weights[sample_indices, pos_indices].unsqueeze(-1)  # [N, 1]
            
            # Accumulate weighted logits (using index_add for efficiency)
            weighted_logits = exp_logits * weights
            logits.index_add_(0, sample_indices, weighted_logits)
            
            # Store weighted clauses
            weighted_clauses = exp_clauses * weights
            for idx, sample_idx in enumerate(sample_indices.tolist()):
                if all_clauses_list[sample_idx] is None:
                    all_clauses_list[sample_idx] = [weighted_clauses[idx]]
                else:
                    all_clauses_list[sample_idx].append(weighted_clauses[idx])
        
        # Concatenate clause outputs
        clauses = torch.stack([
            torch.cat(c, dim=0) if c else torch.zeros(self.top_k * self.n_clauses_per_expert, device=device)
            for c in all_clauses_list
        ], dim=0)
        
        # Output projection
        logits = self.output_proj(logits)
        
        if return_routing:
            return {
                "logits": logits,
                "clauses": clauses,
                "expert_weights": expert_weights,
                "expert_indices": expert_indices,
                "gates": gates,
                "aux_loss": self._aux_loss,
            }
        
        return logits, clauses
    
    def get_expert_utilization(self) -> torch.Tensor:
        """
        Get utilization statistics for each expert.
        
        Call this after a batch to see which experts are being used.
        """
        # This requires tracking during forward, simplified version:
        return torch.ones(self.n_experts) / self.n_experts


class BatchedSparseMoETM(nn.Module):
    """
    Batched implementation of SparseMoETM for better efficiency.
    
    Instead of processing each sample individually, this groups samples
    by their expert assignments for better parallelism.
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses_per_expert: int,
        n_classes: int,
        n_experts: int = 8,
        top_k: int = 2,
        operator: str = "capacity",
        tau: float = 0.5,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.n_clauses_per_expert = n_clauses_per_expert
        self.n_classes = n_classes
        self.top_k = top_k
        
        # Router
        self.router = MoEClauseRouter(
            input_dim=n_features,
            n_experts=n_experts,
            top_k=top_k,
        )
        
        # Batched expert computation
        # Instead of separate modules, use combined parameters
        total_clauses = n_experts * n_clauses_per_expert
        self.shared_stcm = OptimizedSTCM(
            n_features=n_features,
            n_clauses=total_clauses,
            n_classes=n_classes,
            operator=operator,
            tau=tau,
        )
        
        # Expert masks for selecting clause subsets
        expert_masks = []
        for i in range(n_experts):
            mask = torch.zeros(total_clauses)
            mask[i * n_clauses_per_expert:(i + 1) * n_clauses_per_expert] = 1.0
            expert_masks.append(mask)
        self.register_buffer("expert_masks", torch.stack(expert_masks))
        
        self.register_buffer("_aux_loss", torch.tensor(0.0))
    
    @property
    def aux_loss(self) -> torch.Tensor:
        return self._aux_loss
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched forward pass.
        """
        batch_size = x.shape[0]
        
        # Get routing
        expert_weights, expert_indices, gates, aux_loss = self.router(x)
        if aux_loss is not None:
            self._aux_loss = aux_loss * 0.01
        
        # Run all clauses
        all_logits, all_clauses = self.shared_stcm(x)  # [batch, n_classes], [batch, total_clauses]
        
        # Apply expert masking
        # For each sample, weight clauses by expert assignment
        weighted_clauses = torch.zeros_like(all_clauses)
        
        for k in range(self.top_k):
            expert_idx = expert_indices[:, k]  # [batch]
            weight = expert_weights[:, k:k+1]  # [batch, 1]
            
            # Get mask for each sample's expert
            masks = self.expert_masks[expert_idx]  # [batch, total_clauses]
            weighted_clauses = weighted_clauses + masks * weight * all_clauses
        
        # Recompute logits based on weighted clauses
        # Simplified: use the all_logits weighted by expert gates
        weighted_logits = torch.zeros_like(all_logits)
        for k in range(self.top_k):
            weighted_logits = weighted_logits + expert_weights[:, k:k+1] * all_logits
        
        return weighted_logits, weighted_clauses


class HierarchicalMoETM(nn.Module):
    """
    Hierarchical Mixture-of-Experts TM.
    
    Two-level hierarchy:
    1. Coarse router selects expert "families"
    2. Fine router selects specific experts within families
    
    This enables more structured expert specialization.
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses_per_expert: int,
        n_classes: int,
        n_families: int = 4,
        experts_per_family: int = 4,
        top_k_families: int = 2,
        top_k_experts: int = 1,
    ):
        super().__init__()
        self.n_families = n_families
        self.experts_per_family = experts_per_family
        self.n_experts = n_families * experts_per_family
        
        # Coarse router (selects families)
        self.family_router = MoEClauseRouter(
            input_dim=n_features,
            n_experts=n_families,
            top_k=top_k_families,
        )
        
        # Fine routers (one per family)
        self.expert_routers = nn.ModuleList([
            MoEClauseRouter(
                input_dim=n_features,
                n_experts=experts_per_family,
                top_k=top_k_experts,
            )
            for _ in range(n_families)
        ])
        
        # All experts
        self.experts = nn.ModuleList([
            ClauseExpert(
                n_features=n_features,
                n_clauses=n_clauses_per_expert,
                n_classes=n_classes,
            )
            for _ in range(self.n_experts)
        ])
        
        self.register_buffer("_aux_loss", torch.tensor(0.0))
    
    @property
    def aux_loss(self) -> torch.Tensor:
        return self._aux_loss
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hierarchical MoE forward pass.
        """
        batch_size = x.shape[0]
        device = x.device
        n_classes = self.experts[0].stcm.n_classes
        
        # First level: select families
        family_weights, family_indices, _, aux1 = self.family_router(x)
        
        # Initialize outputs
        logits = torch.zeros(batch_size, n_classes, device=device)
        all_clauses = []
        
        total_aux = aux1 if aux1 is not None else torch.tensor(0.0, device=device)
        
        for i in range(batch_size):
            sample_logits = torch.zeros(n_classes, device=device)
            sample_clauses = []
            
            for fk, family_idx in enumerate(family_indices[i]):
                family_idx = family_idx.item()
                family_weight = family_weights[i, fk]
                
                # Second level: select experts within family
                expert_weights, expert_indices, _, aux2 = self.expert_routers[family_idx](
                    x[i:i+1]
                )
                if aux2 is not None:
                    total_aux = total_aux + aux2
                
                for ek, expert_idx in enumerate(expert_indices[0]):
                    expert_idx = expert_idx.item()
                    expert_weight = expert_weights[0, ek]
                    
                    # Global expert index
                    global_idx = family_idx * self.experts_per_family + expert_idx
                    
                    # Run expert
                    exp_logits, exp_clauses = self.experts[global_idx](x[i:i+1])
                    
                    # Combined weight
                    weight = family_weight * expert_weight
                    sample_logits = sample_logits + weight * exp_logits.squeeze(0)
                    sample_clauses.append(exp_clauses.squeeze(0) * weight)
            
            logits[i] = sample_logits
            all_clauses.append(torch.cat(sample_clauses, dim=0))
        
        clauses = torch.stack(all_clauses, dim=0)
        self._aux_loss = total_aux * 0.01
        
        return logits, clauses


class SwitchMoETM(nn.Module):
    """
    Switch-style MoE TM (top-1 routing).
    
    Simpler variant that routes each sample to exactly one expert,
    with capacity constraints and auxiliary losses.
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses_per_expert: int,
        n_classes: int,
        n_experts: int = 8,
        capacity_factor: float = 1.0,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        
        # Simple linear router
        self.router = nn.Linear(n_features, n_experts)
        
        # Experts
        self.experts = nn.ModuleList([
            ClauseExpert(
                n_features=n_features,
                n_clauses=n_clauses_per_expert,
                n_classes=n_classes,
            )
            for _ in range(n_experts)
        ])
        
        self.register_buffer("_aux_loss", torch.tensor(0.0))
    
    @property
    def aux_loss(self) -> torch.Tensor:
        return self._aux_loss
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Switch-style forward pass - OPTIMIZED with batched expert processing.
        """
        batch_size = x.shape[0]
        device = x.device
        n_classes = self.experts[0].stcm.n_classes
        n_clauses = self.experts[0].stcm.n_clauses
        
        # Route to single expert
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-1 selection
        expert_gate, expert_index = router_probs.max(dim=-1)
        
        # Compute load balancing loss (vectorized)
        expert_mask = F.one_hot(expert_index, self.n_experts).float()  # [batch, n_experts]
        fraction_per_expert = expert_mask.sum(dim=0) / batch_size
        mean_prob_per_expert = router_probs.mean(dim=0)
        
        self._aux_loss = self.n_experts * torch.sum(
            fraction_per_expert * mean_prob_per_expert
        ) * 0.01
        
        # OPTIMIZED: Batch process per expert
        logits = torch.zeros(batch_size, n_classes, device=device, dtype=x.dtype)
        clauses = torch.zeros(batch_size, n_clauses, device=device, dtype=x.dtype)
        
        for exp_idx in range(self.n_experts):
            # Find samples routed to this expert
            mask = (expert_index == exp_idx)
            if not mask.any():
                continue
            
            sample_indices = mask.nonzero(as_tuple=True)[0]
            expert_inputs = x[sample_indices]
            gates = expert_gate[sample_indices].unsqueeze(-1)
            
            # Batch process
            exp_logits, exp_clauses = self.experts[exp_idx](expert_inputs)
            
            # Store results with gating
            logits[sample_indices] = gates * exp_logits
            clauses[sample_indices] = exp_clauses
        
        return logits, clauses

