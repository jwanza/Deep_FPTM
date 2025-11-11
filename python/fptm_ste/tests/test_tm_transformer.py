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


def test_unified_tm_transformer_vit_diagnostics():
    model = UnifiedTMTransformer(
        num_classes=10,
        architecture="vit",
        backend="ste",
        image_size=(32, 32),
        in_channels=3,
        patch_size=4,
        embed_dim=32,
        depths=2,
        num_heads=2,
        mlp_ratio=2.0,
        tm_clauses=32,
    )
    x = torch.rand(1, 3, 32, 32)
    logits, diagnostics = model(x, use_ste=True, collect_diagnostics=True)
    assert logits.shape == (1, 10)
    expected = {
        "patch_embed",
        "block_1",
        "block_1_head1",
        "block_1_head2",
        "block_2",
        "block_2_head1",
        "block_2_head2",
        "pre_head",
        "final_decision",
    }
    assert set(diagnostics.keys()) == expected
    for key, value in diagnostics.items():
        assert value.shape == (1, 10), key


def test_unified_tm_transformer_swin_diagnostics():
    model = UnifiedTMTransformer(
        num_classes=10,
        architecture="swin",
        backend="ste",
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
    x = torch.rand(1, 3, 32, 32)
    logits, diagnostics = model(x, use_ste=True, collect_diagnostics=True)
    assert logits.shape == (1, 10)
    expected = {
        "patch_embed",
        "stage1_block1",
        "stage1_block1_head1",
        "stage1_block1_head2",
        "stage1_out",
        "stage2_block1",
        "stage2_block1_head1",
        "stage2_block1_head2",
        "stage2_block1_head3",
        "stage2_block1_head4",
        "stage2_out",
        "pre_head",
        "final_decision",
    }
    assert set(diagnostics.keys()) == expected
    for key, value in diagnostics.items():
        assert value.shape == (1, 10), key


