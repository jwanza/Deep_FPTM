import torch
import pytest

from fptm_ste.multires_tm import SpatialTMScaleConfig, SpatialTMEnsemble


def test_spatial_tm_ensemble_forward():
    configs = [
        SpatialTMScaleConfig(in_channels=8, image_size=16, patch_size=4, n_clauses=4),
        SpatialTMScaleConfig(in_channels=16, image_size=8, patch_size=2, n_clauses=6),
    ]
    ensemble = SpatialTMEnsemble(configs, n_classes=3, attention_heads=2, attention_dim=32)
    feats = [
        torch.randn(2, cfg.in_channels, cfg.image_size, cfg.image_size)
        for cfg in configs
    ]
    logits, aux, fused = ensemble(feats)
    assert logits.shape == (2, 3)
    assert len(aux) == len(configs)
    assert len(fused) == len(configs)


def test_spatial_tm_cross_scale_attention_aligns_shapes():
    configs = [
        SpatialTMScaleConfig(in_channels=4, image_size=32, patch_size=4, n_clauses=8),
        SpatialTMScaleConfig(in_channels=8, image_size=16, patch_size=4, n_clauses=12),
        SpatialTMScaleConfig(in_channels=16, image_size=8, patch_size=2, n_clauses=16),
    ]
    ensemble = SpatialTMEnsemble(configs, n_classes=5, attention_heads=1, attention_dim=16)
    feats = [torch.randn(1, cfg.in_channels, cfg.image_size, cfg.image_size) for cfg in configs]
    logits, aux, fused = ensemble(feats, use_ste=False)
    assert logits.shape == (1, 5)
    assert all(a.shape == (1, 5) for a in aux)
    for cfg, fmap in zip(configs, fused):
        assert fmap.shape[2:] == (cfg.image_size // cfg.patch_size, cfg.image_size // cfg.patch_size)


def test_spatial_tm_raises_on_scale_mismatch():
    cfg = SpatialTMScaleConfig(in_channels=4, image_size=16, patch_size=4, n_clauses=4)
    ensemble = SpatialTMEnsemble([cfg], n_classes=2)
    with pytest.raises(ValueError):
        ensemble([])

