import pytest
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





@pytest.mark.parametrize("gate_type", ["linear", "geglu", "swiglu"])
def test_tm_feedforward_split_gates(gate_type):
    model = UnifiedTMTransformer(
        num_classes=10,
        architecture="vit",
        backend="ste",
        image_size=(32, 32),
        in_channels=3,
        patch_size=4,
        embed_dim=32,
        depths=1,
        num_heads=2,
        mlp_ratio=2.0,
        tm_clauses=32,
        ff_gate=gate_type,
    )
    x = torch.rand(2, 3, 32, 32, requires_grad=True)
    logits = model(x, use_ste=True)
    assert logits.shape == (2, 10)
    logits.sum().backward()


def test_tm_feedforward_tm_gate():
    model = UnifiedTMTransformer(
        num_classes=10,
        architecture="vit",
        backend="ste",
        image_size=(32, 32),
        in_channels=3,
        patch_size=4,
        embed_dim=32,
        depths=1,
        num_heads=2,
        mlp_ratio=2.0,
        tm_clauses=32,
        ff_gate="tm",
    )
    x = torch.rand(2, 3, 32, 32, requires_grad=True)
    logits = model(x, use_ste=True)
    assert logits.shape == (2, 10)
    logits.sum().backward()


def test_tm_feedforward_deeptm_gate():
    model = UnifiedTMTransformer(
        num_classes=10,
        architecture="vit",
        backend="deeptm",
        image_size=(32, 32),
        in_channels=3,
        patch_size=4,
        embed_dim=32,
        depths=1,
        num_heads=2,
        mlp_ratio=2.0,
        tm_clauses=32,
        ff_gate="deeptm",
    )
    x = torch.rand(2, 3, 32, 32, requires_grad=True)
    logits = model(x, use_ste=True)
    assert logits.shape == (2, 10)
    logits.sum().backward()


def test_unified_tm_transformer_clause_dropout():
    model = UnifiedTMTransformer(
        num_classes=10,
        architecture="vit",
        backend="ste",
        image_size=(32, 32),
        in_channels=3,
        patch_size=4,
        embed_dim=32,
        depths=1,
        num_heads=2,
        mlp_ratio=2.0,
        tm_clauses=32,
        clause_dropout=0.25,
        literal_dropout=0.1,
        clause_bias_init=0.05,
    )
    model.train()
    x = torch.rand(2, 3, 32, 32, requires_grad=True)
    logits = model(x, use_ste=True)
    reg = model.pop_regularization_loss()
    loss = logits.sum()
    if reg is not None:
        loss = loss + reg
    loss.backward()


def test_unified_tm_transformer_sparsity_penalty():
    model = UnifiedTMTransformer(
        num_classes=10,
        architecture="vit",
        backend="ste",
        image_size=(32, 32),
        in_channels=3,
        patch_size=4,
        embed_dim=32,
        depths=1,
        num_heads=2,
        mlp_ratio=2.0,
        tm_clauses=32,
        ff_sparsity_weight=0.1,
    )
    model.train()
    x = torch.rand(2, 3, 32, 32, requires_grad=True)
    logits = model(x, use_ste=True)
    reg = model.pop_regularization_loss()
    assert reg is not None
    (logits.sum() + reg).backward()


@pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm", "scalenorm"])
def test_unified_tm_transformer_norm_variants(norm_type):
    model = UnifiedTMTransformer(
        num_classes=10,
        architecture="vit",
        backend="ste",
        image_size=(32, 32),
        in_channels=3,
        patch_size=4,
        embed_dim=32,
        depths=1,
        num_heads=2,
        mlp_ratio=2.0,
        tm_clauses=32,
        norm_type=norm_type,
    )
    x = torch.rand(2, 3, 32, 32)
    logits = model(x, use_ste=True)
    assert logits.shape == (2, 10)


def test_unified_tm_transformer_clause_metrics_capture():
    model = UnifiedTMTransformer(
        num_classes=4,
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
    model(image.unsqueeze(0), use_ste=True, collect_diagnostics=True)
    metrics = model.consume_clause_metrics()
    assert "block_1_clause_mean" in metrics
    assert isinstance(metrics["block_1_clause_mean"], torch.Tensor)


def test_unified_tm_transformer_clause_specialization_adjusts_gains():
    model = UnifiedTMTransformer(
        num_classes=4,
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
    before = model.blocks[0].attn.head_gains.clone()
    metrics = {"block_1_head_mean": [0.2, 0.8]}
    model.apply_clause_head_specialization(metrics, smoothing=1.0)
    after = model.blocks[0].attn.head_gains
    assert not torch.allclose(before, after)


def test_unified_tm_transformer_feature_mix():
    model = UnifiedTMTransformer(
        num_classes=10,
        architecture="vit",
        backend="ste",
        image_size=(32, 32),
        in_channels=3,
        patch_size=4,
        embed_dim=32,
        depths=1,
        num_heads=2,
        mlp_ratio=2.0,
        tm_clauses=32,
        ff_mix_type="linear_depthwise",
        ff_bitwise_mix=True,
    )
    x = torch.rand(2, 3, 32, 32)
    logits = model(x, use_ste=True)
    assert logits.shape == (2, 10)


def test_unified_tm_transformer_learnable_tau():
    model = UnifiedTMTransformer(
        num_classes=10,
        architecture="vit",
        backend="ste",
        image_size=(32, 32),
        in_channels=3,
        patch_size=4,
        embed_dim=32,
        depths=1,
        num_heads=2,
        mlp_ratio=2.0,
        tm_clauses=32,
        learnable_tau=True,
        tau_ema_beta=0.9,
    )
    x = torch.rand(2, 3, 32, 32)
    logits = model(x, use_ste=True)
    assert logits.shape == (2, 10)
    reg = model.pop_regularization_loss()
    assert reg is None or reg.shape == ()


def test_unified_tm_transformer_clause_attention():
    model = UnifiedTMTransformer(
        num_classes=10,
        architecture="vit",
        backend="ste",
        image_size=(32, 32),
        in_channels=3,
        patch_size=4,
        embed_dim=32,
        depths=1,
        num_heads=2,
        mlp_ratio=2.0,
        tm_clauses=32,
        clause_attention=True,
        clause_routing=True,
    )
    x = torch.rand(2, 3, 32, 32)
    logits = model(x, use_ste=True)
    assert logits.shape == (2, 10)


def test_unified_tm_transformer_relative_pos_learned():
    model = UnifiedTMTransformer(
        num_classes=10,
        architecture="vit",
        backend="ste",
        image_size=(32, 32),
        in_channels=3,
        patch_size=4,
        embed_dim=32,
        depths=1,
        num_heads=2,
        mlp_ratio=2.0,
        tm_clauses=32,
        relative_position_type="learned",
    )
    assert model.blocks[0].attn.relative_position_bias is not None
    x = torch.rand(2, 3, 32, 32)
    logits = model(x, use_ste=True)
    assert logits.shape == (2, 10)


def test_unified_tm_transformer_relative_pos_rotary():
    model = UnifiedTMTransformer(
        num_classes=10,
        architecture="vit",
        backend="ste",
        image_size=(32, 32),
        in_channels=3,
        patch_size=4,
        embed_dim=32,
        depths=1,
        num_heads=2,
        mlp_ratio=2.0,
        tm_clauses=32,
        relative_position_type="rotary",
    )
    assert model.blocks[0].attn.rotary_emb is not None
    x = torch.rand(2, 3, 32, 32)
    logits = model(x, use_ste=True)
    assert logits.shape == (2, 10)


def test_unified_tm_transformer_continuous_bypass():
    model = UnifiedTMTransformer(
        num_classes=10,
        architecture="vit",
        backend="ste",
        image_size=(32, 32),
        in_channels=3,
        patch_size=4,
        embed_dim=32,
        depths=1,
        num_heads=2,
        mlp_ratio=2.0,
        tm_clauses=32,
        continuous_bypass=True,
        bypass_scale=0.5,
    )
    x = torch.rand(2, 3, 32, 32)
    logits = model(x, use_ste=True)
    assert logits.shape == (2, 10)
