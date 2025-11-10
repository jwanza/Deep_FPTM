import torch

from fptm_ste.swin_pyramid_tm import SwinLikePyramidTM, window_partition, window_reverse


def test_window_partition_roundtrip():
    B, H, W, C = 2, 8, 8, 16
    x = torch.rand(B, H, W, C)
    ws = 4
    windows = window_partition(x, ws)
    restored = window_reverse(windows, ws, H, W)
    assert torch.allclose(x, restored)


def test_swin_like_pyramid_forward_backward():
    model = SwinLikePyramidTM()
    x = torch.rand(2, 1, 28, 28, requires_grad=True)
    logits, stage_logits, clause_outputs = model(x, use_ste=True)
    assert logits.shape == (2, 10)
    assert len(stage_logits) == len(clause_outputs)
    loss = logits.sum()
    loss.backward()
    assert x.grad is not None and torch.all(torch.isfinite(x.grad))
