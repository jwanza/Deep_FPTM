import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List

from .tm import prepare_tm_input

class ProbabilisticLogicLayer(nn.Module):
    """
    Probabilistic Logic Layer (PLU-based STCM variant).
    
    A fully differentiable Tsetlin Machine layer using Gumbel-Softmax to 
    learn discrete structural decisions.
    
    Instead of 'soft weights' (tanh), this learns the PROBABILITY of a literal
    being:
    1. Included as Positive (require 1)
    2. Included as Negative (require 0)
    3. Excluded (Wildcard)
    
    This matches the discrete nature of Tsetlin Machines while maintaining
    high-quality gradient flow via the Gumbel-Softmax relaxation.
    """
    
    def __init__(
        self, 
        n_features: int,
        n_clauses: int, 
        n_classes: int,
        temperature: float = 1.0, 
        learnable_temp: bool = True,
        input_shape: Optional[Tuple[int, int, int]] = None,
        auto_expand_grayscale: bool = False,
        allow_channel_reduce: bool = True,
        clause_dropout: float = 0.0,
        literal_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.auto_expand_grayscale = auto_expand_grayscale
        self.allow_channel_reduce = allow_channel_reduce
        self.clause_dropout = clause_dropout
        self.literal_dropout = literal_dropout
        
        # Logits for [Include_Pos, Include_Neg, Exclude]
        # Initialize with bias towards 'Exclude' for sparsity (Tsetlin-like)
        self.logits = nn.Parameter(torch.randn(n_clauses, n_features, 3))
        with torch.no_grad():
            self.logits[:, :, 2] += 5.0  # Strong bias towards exclusion
            
        # Temperature for Gumbel-Softmax
        self.temperature = nn.Parameter(torch.tensor(temperature)) if learnable_temp else temperature
        
        # Voting weights for classification
        self.voting = nn.Parameter(torch.randn(n_clauses, n_classes) * 0.1)
        self.clause_bias = nn.Parameter(torch.zeros(n_clauses))

    def get_ternary_weights(self, hard: bool = False):
        """
        Get effective ternary weights {-1, 0, 1}.
        """
        if self.training and not hard:
            # Differentiable sampling
            # [N, D, 3]
            soft_decisions = F.gumbel_softmax(self.logits, tau=self.temperature, hard=True, dim=-1)
        else:
            # Hard argmax
            indices = self.logits.argmax(dim=-1)
            soft_decisions = F.one_hot(indices, num_classes=3).float()
            
        # Map to weights:
        # Index 0 (Pos) -> +1
        # Index 1 (Neg) -> -1
        # Index 2 (Exclude) -> 0
        w_pos = soft_decisions[..., 0]
        w_neg = soft_decisions[..., 1]
        
        # Result is [N, D]
        return w_pos - w_neg

    def forward(
        self, 
        x: torch.Tensor, 
        skip_norm: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [Batch, In_Dims] or Image
            
        Returns:
            logits: [Batch, N_Classes]
            clause_outputs: [Batch, N_Clauses]
        """
        if not skip_norm:
            x = prepare_tm_input(
                x,
                n_features=self.n_features,
                input_shape=self.input_shape,
                auto_expand_grayscale=self.auto_expand_grayscale,
                allow_channel_reduce=self.allow_channel_reduce,
            )
            
        if self.training and self.literal_dropout > 0:
            x = F.dropout(x, p=self.literal_dropout, training=True)

        # w_effective: [N_Clauses, In_Dims]
        # hard=False during training ensures gradients flow through probabilities
        # even if we sample a "hard" decision via Gumbel-Straight-Through
        w_effective = self.get_ternary_weights(hard=False if self.training else True)
        
        # Ensure fuzzy boolean input [0, 1]
        x_bool = torch.sigmoid(x) if x.min() < 0 or x.max() > 1 else x
        
        # Differentiable Tsetlin Logic:
        # We want to penalize Mismatches.
        # w=1, x=1 -> Match (+1)
        # w=-1, x=0 -> Match (-1 * -1 = +1 with bipolar X)
        # x_bipolar = 2*x - 1  (0->-1, 1->1)
        
        x_bipolar = 2 * x_bool - 1
        
        # Dot product: [B, N]
        match_score = F.linear(x_bipolar, w_effective)
        
        # Capacity: [N]
        # Sum of absolute weights (number of active literals)
        capacity = w_effective.abs().sum(dim=1)
        
        # A clause evaluates to TRUE if match_score == capacity.
        # dist = capacity - match_score (always >= 0 for boolean logic)
        dist = (capacity.unsqueeze(0) - match_score).clamp(min=0.0)
        
        # Gaussian/Exponential kernel for AND gate
        # exp(-dist) -> 1.0 when dist=0, decays quickly
        clause_activation = torch.exp(-dist * 0.1)
        
        if self.training and self.clause_dropout > 0:
            clause_activation = F.dropout(clause_activation, p=self.clause_dropout, training=True)
            
        # Voting
        logits = (clause_activation + self.clause_bias.unsqueeze(0)) @ self.voting
        
        return logits, clause_activation
        
    def get_sparsity(self) -> float:
        """Get sparsity (fraction of Excluded literals)."""
        with torch.no_grad():
            # Index 2 is Exclude
            indices = self.logits.argmax(dim=-1)
            sparsity = (indices == 2).float().mean().item()
            return sparsity

    @torch.no_grad()
    def discretize(self, threshold: float = 0.5):
        """
        Returns a Python dict with included literals per clause, suitable for export.
        """
        w_effective = self.get_ternary_weights(hard=True).cpu() # [N, D]
        
        pos_literals = []
        neg_literals = []
        
        for i in range(self.n_clauses):
            row = w_effective[i]
            # 1-based indices
            p_idxs = ((row == 1).nonzero(as_tuple=False).view(-1) + 1).tolist()
            n_idxs = ((row == -1).nonzero(as_tuple=False).view(-1) + 1).tolist()
            pos_literals.append(p_idxs)
            neg_literals.append(n_idxs)
            
        return {
            "positive": pos_literals,  # List of lists
            "negative": neg_literals,  # List of lists (inverted requirements)
            "clauses_num": self.n_clauses,
            "sparsity": self.get_sparsity()
        }
