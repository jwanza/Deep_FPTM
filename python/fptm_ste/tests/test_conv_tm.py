import torch
import torch.nn.functional as F

from fptm_ste.conv_tm import ConvSTE2d, ConvSTCM2d
from fptm_ste.deep_ctm import DeepCTMNetwork
from fptm_ste.tm import FuzzyPatternTM_STE, FuzzyPatternTM_STCM


def _shape_smoke_convste2d():
    x = torch.rand(2, 3, 32, 32)
    layer = ConvSTE2d(3, 16, kernel_size=3, stride=1, padding=1, n_clauses=64, tau=0.5)
    y = layer(x, use_ste=True)
    assert y.shape == (2, 16, 32, 32)
    y.mean().backward()


def _shape_smoke_convstcm2d():
    x = torch.rand(2, 1, 28, 28)
    layer = ConvSTCM2d(1, 8, kernel_size=5, stride=1, padding=2, n_clauses=64, tau=0.5, operator="capacity")
    y = layer(x, use_ste=False)
    assert y.shape == (2, 8, 28, 28)
    y.mean().backward()


def _smoke_deepctm():
    x = torch.rand(4, 1, 28, 28)
    model = DeepCTMNetwork(
        in_channels=1,
        image_size=(28, 28),
        num_classes=10,
        channels=[16, 32],
        kernels=[5, 3],
        strides=[1, 1],
        pools=[2, 2],
        clauses_per_block=[64, 64],
        head_clauses=128,
        tau=0.5,
        dropout=0.1,
        conv_core_backend="tm",
        layer_cls=FuzzyPatternTM_STE,
    )
    logits, _ = model(x, use_ste=True)
    assert logits.shape == (4, 10)
    loss = F.cross_entropy(logits, torch.randint(0, 10, (4,)))
    loss.backward()


def _smoke_deepcstcm():
    x = torch.rand(4, 3, 32, 32)
    model = DeepCTMNetwork(
        in_channels=3,
        image_size=(32, 32),
        num_classes=10,
        channels=[16, 32],
        kernels=[3, 3],
        strides=[1, 1],
        pools=[2, 2],
        clauses_per_block=[64, 64],
        head_clauses=128,
        tau=0.5,
        dropout=0.1,
        conv_core_backend="stcm",
        layer_cls=FuzzyPatternTM_STCM,
        stcm_operator="capacity",
        stcm_ternary_voting=False,
        stcm_ternary_band=0.05,
        stcm_ste_temperature=1.0,
    )
    logits, _ = model(x, use_ste=False)
    assert logits.shape == (4, 10)
    loss = F.cross_entropy(logits, torch.randint(0, 10, (4,)))
    loss.backward()


