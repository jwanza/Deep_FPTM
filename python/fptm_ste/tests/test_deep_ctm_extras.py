import torch

from fptm_ste.deep_ctm import DeepCTMNetwork


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

