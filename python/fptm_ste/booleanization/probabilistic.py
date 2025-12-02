"""
Probabilistic Literal Clause Machine.

Instead of hard binary values, this module represents each literal as
a probability distribution, preserving uncertainty from continuous inputs.

Key Innovation:
Binary TMs lose the uncertainty inherent in continuous values. A feature
at 0.49 vs 0.51 becomes completely different (0 vs 1), yet they're almost
identical. Probabilistic literals preserve this uncertainty.

Architecture:
1. Feature → Distribution: Map continuous to Beta/Bernoulli distribution
2. Probabilistic Clause Evaluation: Expected clause strength
3. Uncertainty Propagation: Track uncertainty through voting
4. Interpretable Rules: Extract rules with confidence intervals

Benefits:
- Preserves uncertainty from continuous features
- Provides confidence estimates
- Maintains interpretability (rules with probabilities)
- Smooth gradients everywhere
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tm import FuzzyPatternTM_STCM, prepare_tm_input


# =============================================================================
# Distributional Literal
# =============================================================================


class DistributionalLiteral(nn.Module):
    """
    Represents a literal as a probability distribution.
    
    Instead of x → {0, 1}, maps x → P(literal=1|x).
    This preserves uncertainty about the literal's truth value.
    
    Args:
        n_features: Number of input features
        distribution: Type of distribution ('bernoulli', 'beta', 'gaussian')
        temperature: Temperature for soft decisions
    """
    
    def __init__(
        self,
        n_features: int,
        distribution: str = "bernoulli",
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.distribution = distribution
        self.temperature = temperature
        
        # Learnable threshold and spread parameters
        self.threshold = nn.Parameter(torch.full((n_features,), 0.5))
        self.spread = nn.Parameter(torch.ones(n_features))
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert continuous features to probability distributions.
        
        Args:
            x: Continuous features [batch, n_features] in [0, 1]
            
        Returns:
            (probability, uncertainty) both [batch, n_features]
        """
        if self.distribution == "bernoulli":
            # Use sigmoid to get probability
            logits = (x - self.threshold) / (self.temperature * self.spread.clamp(min=0.1))
            prob = torch.sigmoid(logits)
            
            # Uncertainty = entropy of Bernoulli
            # H = -p*log(p) - (1-p)*log(1-p)
            eps = 1e-7
            entropy = -prob * torch.log(prob + eps) - (1 - prob) * torch.log(1 - prob + eps)
            uncertainty = entropy / math.log(2)  # Normalize to [0, 1]
        
        elif self.distribution == "beta":
            # Parameters of Beta distribution
            # alpha > 1 and beta > 1 for unimodal
            alpha = 1 + F.softplus(x - self.threshold)
            beta = 1 + F.softplus(self.threshold - x)
            
            # Mean of Beta
            prob = alpha / (alpha + beta)
            
            # Variance-based uncertainty
            var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            uncertainty = var * 4  # Scale to [0, 1]
        
        elif self.distribution == "gaussian":
            # Gaussian CDF centered at threshold
            z = (x - self.threshold) / (self.spread.clamp(min=0.1) * self.temperature)
            prob = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
            
            # Uncertainty is high near threshold
            uncertainty = torch.exp(-z ** 2 / 2) / math.sqrt(2 * math.pi)
        
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
        
        return prob, uncertainty


# =============================================================================
# Uncertainty-Aware Voting
# =============================================================================


class UncertaintyAwareVoting(nn.Module):
    """
    Voting mechanism that considers uncertainty.
    
    Clauses with high uncertainty contribute less to the final
    prediction, allowing the model to be more conservative when unsure.
    
    Args:
        n_clauses: Number of clauses
        n_classes: Number of output classes
        uncertainty_discount: How much to discount uncertain clauses
    """
    
    def __init__(
        self,
        n_clauses: int,
        n_classes: int,
        uncertainty_discount: float = 0.5,
    ):
        super().__init__()
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.uncertainty_discount = uncertainty_discount
        
        # Voting weights
        self.voting = nn.Parameter(torch.randn(n_clauses, n_classes) * 0.1)
    
    def forward(
        self,
        clause_probs: torch.Tensor,
        clause_uncertainty: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Uncertainty-weighted voting.
        
        Args:
            clause_probs: Clause activation probabilities [batch, n_clauses]
            clause_uncertainty: Clause uncertainties [batch, n_clauses]
            
        Returns:
            (logits, prediction_uncertainty)
        """
        # Discount clauses by uncertainty
        confidence = 1 - self.uncertainty_discount * clause_uncertainty
        weighted_probs = clause_probs * confidence
        
        # Vote
        logits = weighted_probs @ self.voting
        
        # Propagate uncertainty to predictions
        # Use uncertainty-weighted voting magnitude
        uncertainty_weights = clause_uncertainty @ self.voting.abs()
        prediction_uncertainty = uncertainty_weights / (self.voting.abs().sum(dim=0) + 1e-6)
        
        return logits, prediction_uncertainty


# =============================================================================
# Probabilistic Literal Clause Machine
# =============================================================================


class ProbabilisticLiteralClauseMachine(nn.Module):
    """
    Clause Machine with Probabilistic Literals.
    
    Each feature is represented as a probability distribution rather
    than a hard binary value, preserving uncertainty throughout.
    
    Args:
        n_features: Number of input features
        n_clauses: Number of clauses
        n_classes: Number of output classes
        distribution: Distribution type for literals
        temperature: Temperature for probabilistic computations
        uncertainty_discount: Discount factor for uncertain clauses
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        distribution: str = "bernoulli",
        temperature: float = 1.0,
        uncertainty_discount: float = 0.5,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        
        # Distributional literals
        self.literals = DistributionalLiteral(
            n_features=n_features,
            distribution=distribution,
            temperature=temperature,
        )
        
        # Clause weights (similar to TM but operating on probabilities)
        half = n_clauses // 2
        self.pos_weights = nn.Parameter(torch.randn(half, n_features) * 0.1)
        self.neg_weights = nn.Parameter(torch.randn(half, n_features) * 0.1)
        
        # Inverted literal weights
        self.pos_inv_weights = nn.Parameter(torch.randn(half, n_features) * 0.1)
        self.neg_inv_weights = nn.Parameter(torch.randn(half, n_features) * 0.1)
        
        # Uncertainty-aware voting
        self.voting = UncertaintyAwareVoting(
            n_clauses=n_clauses,
            n_classes=n_classes,
            uncertainty_discount=uncertainty_discount,
        )
    
    def _clause_mask(self, weights: torch.Tensor) -> torch.Tensor:
        """Convert weights to inclusion probabilities."""
        return torch.sigmoid(weights)
    
    def _probabilistic_clause_strength(
        self,
        prob: torch.Tensor,
        uncertainty: torch.Tensor,
        pos_mask: torch.Tensor,
        inv_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute probabilistic clause strength.
        
        Uses expected value computation for clause evaluation.
        
        Args:
            prob: Literal probabilities [batch, n_features]
            uncertainty: Literal uncertainties [batch, n_features]
            pos_mask: Positive literal mask [half, n_features]
            inv_mask: Inverted literal mask [half, n_features]
            
        Returns:
            (clause_strength, clause_uncertainty)
        """
        # Probability of matching positive literals
        # P(match) = prod(P(lit=1)^mask * P(lit=0)^(1-mask))
        # In log domain for stability
        
        eps = 1e-7
        log_prob = torch.log(prob.clamp(min=eps))
        log_neg_prob = torch.log((1 - prob).clamp(min=eps))
        
        # Weighted log probabilities for positive literals
        # [batch, half] = [batch, n_features] @ [n_features, half]
        pos_log_strength = F.linear(log_prob, pos_mask) + F.linear(log_neg_prob, inv_mask)
        
        # Convert back from log
        clause_strength = torch.exp(pos_log_strength.clamp(min=-20))
        
        # Propagate uncertainty
        # Uncertainty increases with clause size (more literals = more uncertainty)
        weighted_uncertainty = F.linear(uncertainty, pos_mask + inv_mask)
        clause_size = (pos_mask + inv_mask).sum(dim=1, keepdim=True).clamp(min=1)
        clause_uncertainty = weighted_uncertainty / clause_size.t()
        
        return clause_strength, clause_uncertainty
    
    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict]:
        """
        Probabilistic forward pass.
        
        Args:
            x: Input features [batch, n_features]
            return_uncertainty: Return uncertainty estimates
            
        Returns:
            (logits, clause_probs) or dict with uncertainty
        """
        # Prepare input
        x_flat = prepare_tm_input(x, n_features=self.n_features)
        
        # Get literal distributions
        prob, uncertainty = self.literals(x_flat)
        
        # Get clause masks
        pos_mask = self._clause_mask(self.pos_weights)
        neg_mask = self._clause_mask(self.neg_weights)
        pos_inv_mask = self._clause_mask(self.pos_inv_weights)
        neg_inv_mask = self._clause_mask(self.neg_inv_weights)
        
        # Compute probabilistic clause strengths
        pos_strength, pos_uncert = self._probabilistic_clause_strength(
            prob, uncertainty, pos_mask, pos_inv_mask
        )
        neg_strength, neg_uncert = self._probabilistic_clause_strength(
            prob, uncertainty, neg_mask, neg_inv_mask
        )
        
        # Combine clauses (positive contribute positively, negative contribute negatively)
        clause_probs = torch.cat([pos_strength, -neg_strength], dim=1)
        clause_uncertainty = torch.cat([pos_uncert, neg_uncert], dim=1)
        
        # Voting with uncertainty
        logits, pred_uncertainty = self.voting(clause_probs, clause_uncertainty)
        
        if return_uncertainty:
            return {
                "logits": logits,
                "clause_probs": clause_probs,
                "clause_uncertainty": clause_uncertainty,
                "prediction_uncertainty": pred_uncertainty,
                "literal_prob": prob,
                "literal_uncertainty": uncertainty,
            }
        
        return logits, clause_probs
    
    def get_interpretable_rules(
        self,
        threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Extract human-readable rules with confidence.
        
        Args:
            threshold: Threshold for including literal in rule
            
        Returns:
            List of rule dictionaries with literals and confidence
        """
        rules = []
        
        for clause_type, (weights, inv_weights) in [
            ("positive", (self.pos_weights, self.pos_inv_weights)),
            ("negative", (self.neg_weights, self.neg_inv_weights)),
        ]:
            masks = self._clause_mask(weights).detach()
            inv_masks = self._clause_mask(inv_weights).detach()
            
            for clause_idx in range(masks.shape[0]):
                rule = {
                    "type": clause_type,
                    "clause_idx": clause_idx,
                    "positive_literals": [],
                    "negative_literals": [],
                }
                
                for feat_idx in range(masks.shape[1]):
                    pos_prob = masks[clause_idx, feat_idx].item()
                    inv_prob = inv_masks[clause_idx, feat_idx].item()
                    
                    if pos_prob > threshold:
                        rule["positive_literals"].append({
                            "feature": feat_idx,
                            "confidence": pos_prob,
                        })
                    if inv_prob > threshold:
                        rule["negative_literals"].append({
                            "feature": feat_idx,
                            "confidence": inv_prob,
                        })
                
                if rule["positive_literals"] or rule["negative_literals"]:
                    rules.append(rule)
        
        return rules

