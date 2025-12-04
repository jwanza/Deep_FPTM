import torch
import pytest

from fptm_ste.backbones import (
    UniversalBackboneFactory,
    get_backbone_normalization,
)

try:  # pragma: no cover - optional dependency
    import timm  # noqa: F401

    HAS_TIMM = True
except Exception:  # pragma: no cover - optional dependency
    HAS_TIMM = False


def test_simple_backbone_factory_outputs_scales():
    adapter = UniversalBackboneFactory.create(
        backbone_type="simple",
        num_scales=3,
        in_channels=1,
        base_channels=8,
    )
    features = adapter(torch.randn(2, 1, 64, 64))
    assert len(features) == 3
    meta = adapter.metadata()
    assert meta.backbone_type == "simple"
    assert len(meta.channels) == 3
    assert len(meta.reductions) == 3
    assert meta.channels == adapter.get_output_channels()
    assert meta.reductions == adapter.get_reduction_factors()


def test_backbone_normalization_channel_dims():
    mean, std = get_backbone_normalization("swin", input_channels=1)
    assert len(mean) == 1
    assert len(std) == 1
    rgb_mean, rgb_std = get_backbone_normalization("resnet", input_channels=3)
    assert len(rgb_mean) == 3
    assert len(rgb_std) == 3


def test_swin_backbone_metadata_matches_channels():
    adapter = UniversalBackboneFactory.create(
        backbone_type="swin",
        backbone_variant="tiny",
        num_scales=2,
        pretrained=False,
        input_size=224,
    )
    feats = adapter(torch.randn(1, 3, 224, 224))
    assert len(feats) == 2
    meta = adapter.metadata()
    assert meta.backbone_type == "swin"
    assert meta.variant == "tiny"
    assert meta.num_scales == 2
    assert meta.channels[:2] == [f.shape[1] for f in feats]


@pytest.mark.skipif(not HAS_TIMM, reason="timm unavailable")
def test_timm_resnet_backbone_reports_correct_reductions():
    adapter = UniversalBackboneFactory.create(
        backbone_type="resnet",
        backbone_variant="18",
        num_scales=3,
        pretrained=False,
    )
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        feats = adapter(x)
    meta = adapter.metadata()
    assert len(feats) == 3
    assert meta.channels[:3] == [f.shape[1] for f in feats]
    assert meta.reductions[:3] == adapter.get_reduction_factors()
    assert all(r > 0 for r in meta.reductions)


@pytest.mark.skipif(not HAS_TIMM, reason="timm unavailable")
def test_timm_vit_backbone_handles_out_indices():
    adapter = UniversalBackboneFactory.create(
        backbone_type="vit",
        backbone_variant="vit_base_patch16_224",
        num_scales=1,
        pretrained=False,
        out_indices=(0,),
    )
    feats = adapter(torch.randn(2, 3, 224, 224))
    assert len(feats) == 1
    meta = adapter.metadata()
    assert meta.channels[0] == feats[0].shape[1]
    assert meta.reductions[0] >= 1


