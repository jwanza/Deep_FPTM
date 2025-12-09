"""
Enhanced STCM module that layers clause-aware feedback heuristics on top of the
baseline FuzzyPatternTM_STCM without altering the existing implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .tm import FuzzyPatternTM_STCM, prepare_tm_input


@dataclass
class FeedbackStats:
    """Container tracking tensors needed for auxiliary feedback losses."""

    pos_strength: torch.Tensor
    neg_strength: torch.Tensor
    clause_outputs: torch.Tensor
    voting_matrix: torch.Tensor
    logits: torch.Tensor


class EnhancedSTCM(FuzzyPatternTM_STCM):
    """
    Drop-in replacement for STCM with differentiable approximations of TM
    feedback Types I–III plus clause-diagnostic hooks.
    """

    def __init__(
        self,
        *args,
        clause_feedback_weight: float = 0.1,
        vote_regularizer_weight: float = 0.01,
        pos_margin: float = 0.4,
        neg_margin: float = 0.1,
        activity_threshold: float = 0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.clause_feedback_weight = clause_feedback_weight
        self.vote_regularizer_weight = vote_regularizer_weight
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.activity_threshold = activity_threshold
        self._feedback_cache: Optional[FeedbackStats] = None
        self._latest_clause_activity: Optional[torch.Tensor] = None

    # --------------------------------------------------------------------- #
    # Forward + diagnostics
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor, use_ste: bool = True, skip_norm: bool = False):
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

        pos_strength, neg_strength, clause_outputs = self._clause_outputs(flat_x, use_ste)
        biased = clause_outputs + self.clause_bias.view(1, -1)
        voting = self._voting_matrix(use_ste)
        logits = biased @ voting

        if self.training:
            self._feedback_cache = FeedbackStats(
                pos_strength=pos_strength,
                neg_strength=neg_strength,
                clause_outputs=clause_outputs,
                voting_matrix=voting,
                logits=logits,
            )
        else:
            self._feedback_cache = None

        self._latest_clause_activity = clause_outputs.detach().abs().mean(dim=0)
        return logits, clause_outputs

    # --------------------------------------------------------------------- #
    # Public helpers
    # --------------------------------------------------------------------- #
    def has_feedback(self) -> bool:
        return self._feedback_cache is not None

    def pop_feedback_cache(self) -> None:
        """Release cached tensors to avoid holding graphs longer than needed."""
        self._feedback_cache = None

    def clause_activity(self) -> Optional[torch.Tensor]:
        return self._latest_clause_activity

    # --------------------------------------------------------------------- #
    # Loss utilities
    # --------------------------------------------------------------------- #
    def extra_feedback_losses(
        self,
        target: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute additional loss terms that mimic TM feedback behaviours.
        Returns a dict so callers can log individual components.
        """
        if not self.training or self._feedback_cache is None:
            device = target.device if isinstance(target, torch.Tensor) else torch.device("cpu")
            zero = torch.tensor(0.0, device=device)
            return {"clause_feedback": zero, "vote_regularizer": zero}

        cache = self._feedback_cache
        logits = logits if logits is not None else cache.logits

        clause_term = self._clause_feedback_loss(target, logits, cache)
        vote_term = self._vote_gate_regularizer(cache)
        return {
            "clause_feedback": clause_term,
            "vote_regularizer": vote_term,
        }

    def _clause_feedback_loss(
        self,
        target: torch.Tensor,
        logits: torch.Tensor,
        cache: FeedbackStats,
    ) -> torch.Tensor:
        if self.clause_feedback_weight <= 0:
            return torch.zeros(1, device=logits.device, dtype=logits.dtype)

        pos_mean = cache.pos_strength.mean(dim=1)
        neg_mean = cache.neg_strength.mean(dim=1)
        gap = pos_mean - neg_mean

        true_scores = logits.gather(1, target.view(-1, 1)).squeeze(1)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(1, target.view(-1, 1), False)
        alt_scores = logits.masked_fill(~mask, torch.finfo(logits.dtype).min).max(dim=1).values
        confidence = torch.sigmoid(true_scores - alt_scores)

        type_i = F.relu(self.pos_margin - gap)
        type_ii = F.relu(neg_mean - self.neg_margin)

        loss = (confidence.detach() * type_i + (1.0 - confidence.detach()) * type_ii).mean()
        return loss * self.clause_feedback_weight

    def _vote_gate_regularizer(self, cache: FeedbackStats) -> torch.Tensor:
        if self.vote_regularizer_weight <= 0:
            return torch.zeros(1, device=cache.logits.device, dtype=cache.logits.dtype)

        activity = cache.clause_outputs.detach().abs().mean(dim=0)
        target_mag = torch.where(
            activity >= self.activity_threshold,
            torch.ones_like(activity),
            torch.zeros_like(activity),
        )

        votes = self._vote_parameters()
        # Average magnitude per clause across classes to align with activity statistic
        clause_vote_mag = votes.abs().mean(dim=1)
        reg = F.mse_loss(clause_vote_mag, target_mag)
        return reg * self.vote_regularizer_weight

    def _vote_parameters(self) -> torch.Tensor:
        if self.ternary_voting:
            assert self.vote_logits is not None
            return self.vote_logits
        assert self.voting is not None
        return self.voting


