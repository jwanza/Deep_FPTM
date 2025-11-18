import math
from typing import Dict

import torch
from torch.utils.data import DataLoader, TensorDataset

from fptm_ste.tm_transformer import UnifiedTMTransformer
from fptm_ste.tests.run_mnist_equiv import (
    DistillationStageScheduler,
    TeacherClauseProjector,
    TeacherFeatureCollector,
    TeacherHintAdapter,
    collect_clause_outputs,
    compute_student_clause_token_distribution,
    infer_clause_dimensions,
    warm_start_clauses_with_teacher,
)


def _make_tiny_transformer() -> UnifiedTMTransformer:
    return UnifiedTMTransformer(
        num_classes=3,
        architecture="vit",
        backend="ste",
        image_size=(16, 16),
        in_channels=1,
        patch_size=4,
        embed_dim=16,
        depths=1,
        num_heads=2,
        mlp_ratio=2.0,
        tm_clauses=32,
        tm_tau=0.5,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    )


def test_teacher_feature_collector_captures_vit_blocks():
    model = _make_tiny_transformer()
    collector = TeacherFeatureCollector(model)
    x = torch.randn(2, 1, 16, 16)
    logits, features, logits_for_loss = collector.collect(x, retain_logits_grad=True)

    assert logits.shape == (2, 3)
    assert logits_for_loss.requires_grad
    assert "patch_embed" in features
    assert "block_1" in features
    assert isinstance(features["block_1"], torch.Tensor)


def test_teacher_feature_collector_detaches_logits_when_requested():
    model = _make_tiny_transformer()
    collector = TeacherFeatureCollector(model)
    x = torch.randn(2, 1, 16, 16)
    _, _, logits_for_loss = collector.collect(x, retain_logits_grad=False)
    assert not logits_for_loss.requires_grad


def test_teacher_hint_adapter_projects_to_class_logits():
    device = torch.device("cpu")
    adapter = TeacherHintAdapter(num_classes=5, device=device)
    feature_map = {
        "block_1": torch.randn(2, 6, 12),
        "patch_embed": torch.randn(2, 9, 12),
    }
    outputs = adapter(feature_map)

    assert set(outputs.keys()).issubset(feature_map.keys())
    for tensor in outputs.values():
        assert tensor.shape == (2, 5)


def test_teacher_clause_projector_matches_clause_dimensions():
    device = torch.device("cpu")
    clause_dims: Dict[str, int] = {"block_1": 7}
    projector = TeacherClauseProjector(clause_dims, device=device)
    feature_map = {"block_1": torch.randn(3, 4, 10)}

    outputs = projector(feature_map)
    assert "block_1" in outputs
    assert outputs["block_1"].shape == (3, 7)


def test_collect_clause_outputs_provides_raw_activations():
    model = _make_tiny_transformer()
    x = torch.randn(2, 1, 16, 16)
    model(x, use_ste=True)
    clause_map = collect_clause_outputs(model)

    assert "block_1" in clause_map
    clause_tensor = clause_map["block_1"]
    assert clause_tensor.dim() == 3


def test_infer_clause_dimensions_matches_tm_blocks():
    model = _make_tiny_transformer()
    clause_dims = infer_clause_dimensions(model)
    assert clause_dims == {"block_1": 32}


def test_student_clause_distribution_is_normalised():
    clauses = torch.rand(2, 4, 6)
    distribution = compute_student_clause_token_distribution(clauses, drop_first_token=False)
    sums = distribution.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_distillation_scheduler_scales_after_stage_transition():
    scheduler = DistillationStageScheduler(
        stage_epoch=2,
        total_epochs=6,
        base_weights={"teacher": 1.0},
        base_temps={"teacher": 2.0},
        weight_scales={"teacher": 0.5},
        temp_scales={"teacher": 0.25},
        base_tau=0.4,
        target_tau=0.9,
    )
    weights_stage1, temps_stage1, tau_stage1 = scheduler.compute(epoch=0)
    weights_stage2, temps_stage2, tau_stage2 = scheduler.compute(epoch=3)

    assert math.isclose(weights_stage1["teacher"], 1.0, rel_tol=1e-6)
    assert math.isclose(temps_stage1["teacher"], 2.0, rel_tol=1e-6)
    assert tau_stage1 is None

    assert math.isclose(weights_stage2["teacher"], 0.5, rel_tol=1e-6)
    assert math.isclose(temps_stage2["teacher"], 0.5, rel_tol=1e-6)
    assert tau_stage2 is not None
    assert 0.4 < tau_stage2 <= 0.9


def test_clause_warm_start_updates_parameters():
    model = _make_tiny_transformer()
    collector = TeacherFeatureCollector(model)
    x = torch.randn(8, 1, 16, 16)
    y = torch.randint(0, 3, (8,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    before = [param.clone() for param in model.parameters()]
    warm_start_clauses_with_teacher(
        model,
        collector,
        loader,
        device=torch.device("cpu"),
        batches=1,
        lr=1e-2,
        temperature=1.5,
    )
    after = list(model.parameters())
    changed = any(not torch.allclose(p0, p1) for p0, p1 in zip(before, after))
    assert changed
