import types
from torchvision import transforms
from fptm_ste.tests.run_mnist_equiv import _build_transform


def test_build_transform_injects_normalize():
    mean = (0.4, 0.5, 0.6)
    std = (0.2, 0.3, 0.4)
    t = _build_transform(
        transforms.ToTensor(),
        target_size=None,
        apply_randaugment=False,
        randaugment_n=2,
        randaugment_m=9,
        add_normalize=(mean, std),
    )
    assert isinstance(t, transforms.Compose)
    # last should be Normalize
    assert any(isinstance(step, transforms.Normalize) for step in t.transforms), "Normalize not injected"
    norms = [step for step in t.transforms if isinstance(step, transforms.Normalize)]
    assert tuple(norms[-1].mean) == mean
    assert tuple(norms[-1].std) == std


