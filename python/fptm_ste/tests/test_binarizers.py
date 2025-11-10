import torch
from fptm_ste.binarizers import SwinDualBinarizer, CNNSingleBinarizer


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


