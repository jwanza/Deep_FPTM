import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parents[1] / "joel" / "OmniEmbedding"))

from fptm_ste import EnhancedSTCM, FuzzyPatternTM_STCM
from fptm_ste.moe_tm import SparseMoETM
from fptm_ste.trainers import ClauseMetricScheduler


def test_enhanced_stcm_feedback_losses_require_training():
    model = EnhancedSTCM(
        n_features=16,
        n_clauses=32,
        n_classes=2,
        clause_feedback_weight=0.5,
        vote_regularizer_weight=0.1,
        ternary_voting=True,
    )
    model.train()
    x = torch.rand(6, 16)
    y = torch.tensor([0, 1, 0, 1, 0, 1])

    logits, _ = model(x)
    losses = model.extra_feedback_losses(y, logits)
    clause_loss = losses["clause_feedback"]
    vote_loss = losses["vote_regularizer"]

    assert clause_loss.requires_grad
    assert vote_loss.requires_grad
    assert clause_loss.item() >= 0
    assert vote_loss.item() >= 0


def test_clause_metric_scheduler_tightens_and_relaxes():
    base = EnhancedSTCM(n_features=8, n_clauses=16, n_classes=2, clause_feedback_weight=0.0, vote_regularizer_weight=0.0)
    base.train()
    x = torch.ones(4, 8)
    base(x)

    scheduler = ClauseMetricScheduler(base)

    # Force high activity
    base._latest_clause_activity = torch.ones(base.n_clauses)
    initial_tau = base.tau
    scheduler.step()
    tightened_tau = base.tau
    assert tightened_tau >= initial_tau

    # Force inactivity
    base._latest_clause_activity = torch.zeros(base.n_clauses)
    scheduler.step()
    assert base.tau <= tightened_tau


def test_sparse_moe_entropy_regularizer_produces_aux_loss():
    n_features = 12
    model = SparseMoETM(
        n_features=n_features,
        n_clauses_per_expert=10,
        n_classes=2,
        n_experts=3,
        top_k=2,
        use_clause_stats=True,
        entropy_weight=0.05,
    )
    model.train()
    x = torch.rand(5, n_features)
    logits, clauses = model(x)

    assert logits.shape[0] == 5
    assert clauses.shape[0] == 5
    aux = model.aux_loss
    assert aux is not None
    assert torch.is_tensor(aux)

