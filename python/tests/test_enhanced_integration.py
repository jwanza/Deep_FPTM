import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parents[1] / "joel" / "OmniEmbedding"))

from fptm_ste import FuzzyPatternTM_STCM, EnhancedSTCM
from fptm_ste.trainers import ClauseMetricScheduler
from tm_training import train_tm_model, evaluate_tm_model


def build_noisy_dataset():
    torch.manual_seed(321)
    X = torch.rand(160, 12)
    logits = X[:, :4].sum(dim=1) - X[:, 4:8].sum(dim=1)
    y = (logits > 0).long()

    train_X, test_X = X[:120], X[120:]
    train_y, test_y = y[:120].clone(), y[120:].clone()

    noise_idx = torch.randperm(train_y.numel())[:30]
    train_y[noise_idx] = 1 - train_y[noise_idx]

    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    return train_dataset, test_dataset


def train_and_eval(model, train_dataset, test_dataset, epochs, scheduler=None, seed=0):
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    torch.manual_seed(seed)
    trained = train_tm_model(
        model,
        train_loader,
        epochs=epochs,
        lr=0.003,
        anneal_tau=True,
        test_loader=None,
        report_every=epochs + 1,
        clause_metric_scheduler=scheduler,
        device_override=torch.device("cpu"),
    )
    acc = evaluate_tm_model(trained, test_loader, use_ste=True, device_override=torch.device("cpu"))
    return acc


def test_enhanced_stcm_beats_baseline_on_toy_problem():
    train_dataset, test_dataset = build_noisy_dataset()

    baseline = FuzzyPatternTM_STCM(
        n_features=12,
        n_clauses=48,
        n_classes=2,
        tau=0.5,
        operator="capacity",
        clause_dropout=0.3,
        literal_dropout=0.35,
    )
    baseline_acc = train_and_eval(baseline, train_dataset, test_dataset, epochs=10, seed=1337)

    enhanced = EnhancedSTCM(
        n_features=12,
        n_clauses=48,
        n_classes=2,
        tau=0.5,
        operator="capacity",
        clause_dropout=0.1,
        literal_dropout=0.1,
        ternary_voting=True,
        clause_feedback_weight=0.05,
        vote_regularizer_weight=0.01,
    )
    scheduler = ClauseMetricScheduler(enhanced)
    enhanced_acc = train_and_eval(enhanced, train_dataset, test_dataset, epochs=10, scheduler=scheduler, seed=1337)

    assert enhanced_acc >= baseline_acc + 0.05

