import math
import torch.nn as nn

from fptm_ste.deep_ctm import DeepCTMNetwork
from fptm_ste.tests.run_mnist_equiv import (
    _interp_schedule,
    _apply_ctm_schedule,
    DistillationStageScheduler,
)
from fptm_ste.trainers import ClauseCurriculumScheduler


def test_interp_schedule_linear_reaches_end():
    sched = {"type": "linear", "start": 0.5, "end": 0.1, "epochs": 4}
    vals = [_interp_schedule(e, sched) for e in range(4)]
    assert math.isclose(vals[0], 0.4, rel_tol=1e-4)
    assert math.isclose(vals[-1], 0.1, rel_tol=1e-4)


def test_apply_ctm_schedule_updates_tau():
    model = DeepCTMNetwork(
        in_channels=1,
        image_size=(8, 8),
        num_classes=2,
        channels=[4],
        kernels=[3],
        strides=[1],
        pools=[1],
        clauses_per_block=[16],
        head_clauses=32,
        tau=0.5,
        dropout=0.0,
        conv_core_backend="tm",
    )
    sched = {"tau": {"type": "linear", "start": 0.5, "end": 0.3, "epochs": 2}}
    updates = _apply_ctm_schedule(model, 0, sched)
    assert "tau" in updates
    assert math.isclose(model.tau, updates["tau"])
    assert math.isclose(updates["tau"], 0.4, rel_tol=1e-4)


class _DummyTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lf = 16
        self.ternary_band = 0.5
        self.ste_temperature = 2.0
        self.clause_dropout = 0.0


def test_clause_curriculum_scheduler_updates_attributes():
    model = _DummyTM()
    scheduler = ClauseCurriculumScheduler(
        model,
        total_epochs=10,
        lf_schedule=(32, 4),
        band_schedule=(1.0, 0.1),
        temp_schedule=(2.0, 0.5),
        clause_warmup_epochs=5,
        schedule_type="linear",
    )
    scheduler.step(epoch=5)
    values = scheduler.get_current_values()
    assert 4 <= values["lf"] <= 32
    assert 0.1 <= values["ternary_band"] <= 1.0
    assert 0.5 <= values["temperature"] <= 2.0
    # Clause warmup should set dropout below 0.5 during warmup window
    assert model.clause_dropout <= 0.5


def test_distillation_stage_scheduler_scales_weights_and_tau():
    scheduler = DistillationStageScheduler(
        stage_epoch=2,
        total_epochs=4,
        base_weights={"teacher": 0.4, "self": 0.1},
        base_temps={"teacher": 2.0, "self": 1.0},
        weight_scales={"teacher": 2.0, "self": 0.5},
        temp_scales={"teacher": 0.5, "self": 1.0},
        base_tau=0.6,
        target_tau=0.2,
    )
    weights, temps, tau = scheduler.compute(1)
    assert weights["teacher"] == 0.4
    assert temps["teacher"] == 2.0
    assert tau is None

    weights, temps, tau = scheduler.compute(3)
    assert math.isclose(weights["teacher"], 0.8)
    assert math.isclose(weights["self"], 0.05)
    assert math.isclose(temps["teacher"], 1.0)
    assert tau is not None and 0.2 <= tau <= 0.6

