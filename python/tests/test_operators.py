import torch

from fptm_ste.operators import (
    TernaryMajority,
    TernaryQuantumAND,
    TernaryXOR,
    available_ternary_operators,
    build_ternary_operator,
)


def test_tqand_matches_reference_formula():
    op = TernaryQuantumAND()
    a = torch.tensor([[1.0, 0.5], [0.2, 0.8]])
    b = torch.tensor([[0.5, 0.5], [0.9, 0.1]])
    c = torch.tensor([[0.7, 0.3], [0.4, 0.4]])
    expected = (a * b * c) / torch.sqrt(torch.clamp(a**2 + b**2 + c**2, min=1e-8))
    torch.testing.assert_close(op(a, b, c), expected)


def test_txor_outputs_are_bounded_and_symmetric():
    op = TernaryXOR()
    a = torch.tensor([[0.0, 1.0, 0.5]])
    b = torch.tensor([[1.0, 0.0, 0.2]])
    forward = op(a, b)
    reverse = op(b, a)
    torch.testing.assert_close(forward, reverse)
    assert torch.all(forward >= 0.0)
    assert torch.all(forward <= 1.0)


def test_tmaj_matches_median_computation():
    op = TernaryMajority()
    tensors = [
        torch.tensor([[0.0, 1.0, 0.2]]),
        torch.tensor([[1.0, 0.0, 0.4]]),
        torch.tensor([[0.5, 0.5, 0.6]]),
    ]
    expected = torch.median(torch.stack(tensors), dim=0).values
    torch.testing.assert_close(op(*tensors), expected)


def test_operator_registry_instantiates_modules():
    registry = available_ternary_operators()
    assert {"tqand", "txor", "tmaj"}.issubset(set(registry))
    for name in registry:
        module = build_ternary_operator(name)
        assert module is not None
    assert build_ternary_operator("capacity") is None

