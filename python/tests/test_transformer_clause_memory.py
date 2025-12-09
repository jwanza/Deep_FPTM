import pytest
import torch

from fptm_ste.tm_transformer import UnifiedTMTransformer


@pytest.mark.parametrize("backend", ["stcm", "ste"])
def test_transformer_reports_clause_memory_metrics(backend: str):
    model = UnifiedTMTransformer(
        num_classes=3,
        architecture="vit",
        backend=backend,
        image_size=(8, 8),
        in_channels=3,
        patch_size=4,
        embed_dim=16,
        depths=1,
        num_heads=2,
        mlp_ratio=2.0,
        tm_clauses=12,
        clause_memory_slots=4,
    )
    x = torch.rand(2, 3, 8, 8)
    model.forward(x, use_ste=False, collect_diagnostics=True)
    metrics = model.consume_clause_metrics()
    assert metrics, "expected clause metrics when diagnostics enabled"
    assert any("memory_mean" in key for key in metrics.keys())



