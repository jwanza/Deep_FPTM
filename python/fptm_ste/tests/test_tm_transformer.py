import torch
from fptm_ste import TM_TransformerBlock, UnifiedTMTransformer


def test_tm_transformer_block():
    block = TM_TransformerBlock(d_model=64, n_heads=4, n_clauses=50)
    x = torch.rand(2, 16, 64)
    y = block(x, use_ste=True)
    assert y.shape == x.shape


def test_unified_tm_transformer_vit():
    model = UnifiedTMTransformer(
        num_classes=10,
        architecture="vit",
        backend="ste",
        image_size=(32, 32),
        in_channels=3,
        patch_size=4,
        embed_dim=64,
        depths=2,
        num_heads=2,
        mlp_ratio=2.0,
        tm_clauses=64,
    )
    x = torch.rand(2, 3, 32, 32, requires_grad=True)
    logits = model(x, use_ste=True)
    assert logits.shape == (2, 10)
    logits.sum().backward()


def test_unified_tm_transformer_swin_deeptm():
    model = UnifiedTMTransformer(
        num_classes=10,
        architecture="swin",
        backend="deeptm",
        image_size=(32, 32),
        in_channels=3,
        patch_size=4,
        embed_dim=(32, 64),
        depths=(1, 1),
        num_heads=(2, 4),
        mlp_ratio=(2.0, 2.0),
        tm_clauses=(32, 32),
        window_size=4,
    )
    x = torch.rand(2, 3, 32, 32, requires_grad=True)
    logits = model(x, use_ste=True)
    assert logits.shape == (2, 10)
    logits.sum().backward()


