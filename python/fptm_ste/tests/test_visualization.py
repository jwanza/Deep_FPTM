from pathlib import Path

import pytest
import torch

pytest.importorskip("matplotlib")

from fptm_ste.tm_transformer import UnifiedTMTransformer
from fptm_ste.visualization import generate_transformer_overlay


def test_generate_transformer_overlay(tmp_path: Path):
    model = UnifiedTMTransformer(
        num_classes=5,
        architecture="vit",
        backend="ste",
        image_size=(32, 32),
        in_channels=3,
        patch_size=4,
        embed_dim=32,
        depths=1,
        num_heads=2,
        mlp_ratio=2.0,
        tm_clauses=16,
    )
    image = torch.rand(3, 32, 32)
    class_names = [f"class_{i}" for i in range(5)]
    output_path = tmp_path / "overlay.png"
    generate_transformer_overlay(
        model,
        image,
        label=0,
        class_names=class_names,
        architecture="vit",
        patch_size=4,
        image_size=(32, 32),
        mean_std=None,
        output_path=output_path,
        device=torch.device("cpu"),
        topk=3,
    )
    assert output_path.exists()

