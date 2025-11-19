import torch

from fptm_ste.deep_ctm import DeepCTMNetwork
from fptm_ste.tm import FuzzyPatternTM_STCM


def test_deepctm_stem_and_linear_mix_forward_backward():
    model = DeepCTMNetwork(
        in_channels=3,
        image_size=(32, 32),
        num_classes=10,
        channels=[8, 16],
        kernels=[3, 3],
        strides=[1, 1],
        pools=[1, 1],
        clauses_per_block=[32, 32],
        head_clauses=64,
        tau=0.5,
        dropout=0.1,
        conv_core_backend="tm",
        use_stem=True,
        stem_channels=16,
        mix_type="linear",
    )
    model.train()
    x = torch.rand(4, 3, 32, 32)
    logits, _ = model(x, use_ste=False)
    assert logits.shape == (4, 10)
    loss = logits.mean()
    loss.backward()
    # ensure parameters received gradients
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_deepctm_hybrid_heads_mix_summary():
    model = DeepCTMNetwork(
        in_channels=3,
        image_size=(16, 16),
        num_classes=5,
        channels=[8],
        kernels=[3],
        strides=[1],
        pools=[1],
        clauses_per_block=[16],
        head_clauses=32,
        tau=0.5,
        dropout=0.0,
        conv_core_backend="tm",
        head_type="deeptm",
        head_hidden_dims=[16],
        head_linear=True,
        head_linear_hidden=12,
        head_linear_dropout=0.1,
        head_attention=True,
        head_attention_heads=2,
        head_attention_dim=8,
        head_attention_dropout=0.0,
        head_mix_init_tm=1.0,
        head_mix_init_linear=0.3,
        head_mix_init_attention=0.3,
    )
    model.train()
    x = torch.randn(2, 3, 16, 16)
    logits, diagnostics = model(x, use_ste=False, collect_diagnostics=True)
    assert logits.shape == (2, 5)
    assert "head_tm" in diagnostics and "head_linear" in diagnostics and "head_attention" in diagnostics
    summary = model.head_mix_summary()
    assert set(summary.keys()) == {"tm", "linear", "attention"}
    assert all(weight > 0 for weight in summary.values())


def test_deepctm_deepstcm_head_forward():
    model = DeepCTMNetwork(
        in_channels=1,
        image_size=(8, 8),
        num_classes=3,
        channels=[4],
        kernels=[3],
        strides=[1],
        pools=[1],
        clauses_per_block=[8],
        head_clauses=16,
        tau=0.4,
        dropout=0.0,
        conv_core_backend="stcm",
        layer_cls=FuzzyPatternTM_STCM,
        head_type="deepstcm",
        head_hidden_dims=[6],
        stcm_operator="capacity",
        stcm_ternary_voting=True,
        stcm_ternary_band=0.05,
        stcm_ste_temperature=1.0,
    )
    model.eval()
    x = torch.rand(1, 1, 8, 8)
    logits, _ = model(x, use_ste=False)
    assert logits.shape == (1, 3)

