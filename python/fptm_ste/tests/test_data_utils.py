import pytest
import torch

from fptm_ste.utils.data import ChannelAdjust, adjust_channels


def test_adjust_channels_expands_grayscale_to_rgb():
    tensor = torch.randn(1, 5, 5)
    adjusted = adjust_channels(tensor, 3)
    assert adjusted.shape == (3, 5, 5)
    assert torch.allclose(adjusted[0], tensor[0])
    assert torch.allclose(adjusted[1], tensor[0])
    assert torch.allclose(adjusted[2], tensor[0])


def test_adjust_channels_reduces_rgb_to_grayscale():
    tensor = torch.randn(3, 6, 6)
    adjusted = adjust_channels(tensor, 1)
    assert adjusted.shape == (1, 6, 6)
    mean_slice = tensor.mean(dim=0, keepdim=True)
    assert torch.allclose(adjusted, mean_slice)


def test_channel_adjust_respects_expand_flag():
    tensor = torch.randn(1, 4, 4)
    transform = ChannelAdjust(target_channels=3, allow_expand=False)
    with pytest.raises(ValueError):
        transform(tensor)

