import pytest

from fptm_ste import FuzzyPatternTM_STCM
from fptm_ste.trainers import TauLiteralScheduleConfig, TauLiteralScheduler


def test_tau_literal_scheduler_updates_model_state():
    model = FuzzyPatternTM_STCM(
        n_features=8,
        n_clauses=10,
        n_classes=2,
        literal_budget=12,
    )
    cfg = TauLiteralScheduleConfig(
        tau_start=0.9,
        tau_end=0.3,
        literal_start=12,
        literal_end=4,
        total_epochs=4,
        warmup_epochs=0,
        mode="linear",
    )
    scheduler = TauLiteralScheduler(cfg)

    scheduler.apply(model, epoch=0)
    assert model.tau == pytest.approx(0.9, rel=1e-3)
    assert model.literal_budget == pytest.approx(12.0, rel=1e-3)

    scheduler.apply(model, epoch=4)
    assert model.tau == pytest.approx(0.3, rel=1e-3)
    assert model.literal_budget == pytest.approx(4.0, rel=1e-3)


def test_tau_literal_scheduler_respects_warmup():
    model = FuzzyPatternTM_STCM(n_features=6, n_clauses=8, n_classes=2)
    cfg = TauLiteralScheduleConfig(tau_start=0.8, tau_end=0.2, warmup_epochs=2, total_epochs=6)
    scheduler = TauLiteralScheduler(cfg)

    scheduler.apply(model, epoch=1)
    assert model.tau == pytest.approx(0.8, rel=1e-3)

    scheduler.apply(model, epoch=3)
    assert model.tau < 0.8

