import math

from fptm_ste.deep_ctm import DeepCTMNetwork
from fptm_ste.tests.run_mnist_equiv import _interp_schedule, _apply_ctm_schedule


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

