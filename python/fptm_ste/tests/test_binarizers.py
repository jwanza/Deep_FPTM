import torch
from fptm_ste.binarizers import SwinDualBinarizer, CNNSingleBinarizer
from fptm_ste.booleanization.learnable import LearnableBinarizer


def test_swin_dual_binarizer_shapes():
    B, C, H, W = 2, 64, 14, 14
    x = torch.randn(B, C, H, W)  # zero-centered
    binarizer = SwinDualBinarizer(in_channels=C, num_thresholds=8, init_temperature=1.0)
    y = binarizer(x, use_discrete=False)
    assert y.shape == (B, 16, H, W)  # 2T channels

    # Discrete
    y_hard = binarizer(x, use_discrete=True)
    assert ((y_hard == 0) | (y_hard == 1)).all()


def test_cnn_single_binarizer_shapes():
    B, C, H, W = 2, 32, 28, 28
    x = torch.rand(B, C, H, W)  # nonnegative
    binarizer = CNNSingleBinarizer(in_channels=C, num_thresholds=8, init_temperature=1.0)
    y = binarizer(x, use_discrete=False)
    assert y.shape == (B, 8, H, W)

    # Discrete
    y_hard = binarizer(x, use_discrete=True)
    assert ((y_hard == 0) | (y_hard == 1)).all()





def test_learnable_binarizer_modes():
    x = torch.randn(1, 4, 8, 8)
    dual = LearnableBinarizer(in_channels=4, num_thresholds=2, init_temperature=0.5, mode="dual")
    single = LearnableBinarizer(in_channels=4, num_thresholds=2, init_temperature=0.5, mode="single")
    y_dual = dual(x)
    y_single = single(torch.relu(x))
    assert y_dual.shape[1] == 4
    assert y_single.shape[1] == 2


def test_learnable_binarizer_temperature_anneals():
    binarizer = LearnableBinarizer(in_channels=2, num_thresholds=2, init_temperature=1.0)
    assert binarizer.temperature == 1.0
    binarizer.anneal_temperature(0.2)
    assert 0.01 <= binarizer.temperature <= 0.2
    binarizer.anneal_temperature(-5.0)
    assert binarizer.temperature >= 0.01


def test_learnable_binarizer_tracks_running_stats_and_gradients():
    torch.manual_seed(0)
    binarizer = LearnableBinarizer(in_channels=3, num_thresholds=4, init_temperature=0.7, mode="dual")
    binarizer.train()
    x = torch.randn(2, 3, 6, 6, requires_grad=True)
    y = binarizer(x, use_discrete=False)
    loss = y.mean()
    loss.backward()
    assert x.grad is not None
    assert binarizer.running_mean.abs().max() < 10.0
    assert binarizer.running_std.min() >= 0.01
