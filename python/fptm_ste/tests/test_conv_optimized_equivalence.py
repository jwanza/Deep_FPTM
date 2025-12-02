"""Equivalence tests for ConvTM2dOptimized vs ConvTM2d base implementation."""
import torch
import torch.nn.functional as F
from fptm_ste.conv_tm import ConvTM2d, ConvTM2dOptimized
from fptm_ste.tm import FuzzyPatternTM_STE, FuzzyPatternTM_STCM


def test_convste_optimized_vs_base_forward():
    """Test that ConvTM2dOptimized produces identical forward outputs to ConvTM2d for STE."""
    torch.manual_seed(42)
    
    # Create base and optimized versions with same parameters
    base = ConvTM2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
        n_clauses=64,
        tau=0.5,
        core_backend="tm",
        layer_cls=FuzzyPatternTM_STE,
    )
    
    optimized = ConvTM2dOptimized(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
        n_clauses=64,
        tau=0.5,
        core_backend="tm",
        layer_cls=FuzzyPatternTM_STE,
    )
    
    # Copy weights from base to optimized
    optimized.core.load_state_dict(base.core.state_dict())
    
    # Test input
    x = torch.rand(2, 3, 32, 32)
    
    # Forward pass
    with torch.no_grad():
        y_base = base(x, use_ste=True)
        y_opt = optimized(x, use_ste=True)
    
    # Check equivalence
    assert y_base.shape == y_opt.shape, f"Shape mismatch: {y_base.shape} != {y_opt.shape}"
    
    max_diff = (y_base - y_opt).abs().max().item()
    mean_diff = (y_base - y_opt).abs().mean().item()
    
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    
    assert torch.allclose(y_base, y_opt, atol=1e-5, rtol=1e-4), \
        f"Outputs differ: max_diff={max_diff}, mean_diff={mean_diff}"
    
    print("✓ ConvSTE2d: Optimized matches base (forward)")


def test_convstcm_optimized_vs_base_forward():
    """Test that ConvTM2dOptimized produces identical forward outputs to ConvTM2d for STCM."""
    torch.manual_seed(42)
    
    # Create base and optimized versions with same parameters
    base = ConvTM2d(
        in_channels=1,
        out_channels=8,
        kernel_size=5,
        stride=1,
        padding=2,
        n_clauses=64,
        tau=0.5,
        core_backend="stcm",
        layer_cls=FuzzyPatternTM_STCM,
        operator="capacity",
        ternary_voting=False,
        ternary_band=0.1,
        ste_temperature=1.0,
    )
    
    optimized = ConvTM2dOptimized(
        in_channels=1,
        out_channels=8,
        kernel_size=5,
        stride=1,
        padding=2,
        n_clauses=64,
        tau=0.5,
        core_backend="stcm",
        layer_cls=FuzzyPatternTM_STCM,
        operator="capacity",
        ternary_voting=False,
        ternary_band=0.1,
        ste_temperature=1.0,
    )
    
    # Copy weights from base to optimized
    optimized.core.load_state_dict(base.core.state_dict())
    
    # Test input
    x = torch.rand(2, 1, 28, 28)
    
    # Forward pass
    with torch.no_grad():
        y_base = base(x, use_ste=True)
        y_opt = optimized(x, use_ste=True)
    
    # Check equivalence
    assert y_base.shape == y_opt.shape, f"Shape mismatch: {y_base.shape} != {y_opt.shape}"
    
    max_diff = (y_base - y_opt).abs().max().item()
    mean_diff = (y_base - y_opt).abs().mean().item()
    
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    
    assert torch.allclose(y_base, y_opt, atol=1e-5, rtol=1e-4), \
        f"Outputs differ: max_diff={max_diff}, mean_diff={mean_diff}"
    
    print("✓ ConvSTCM2d: Optimized matches base (forward)")


def test_convste_optimized_vs_base_backward():
    """Test that ConvTM2dOptimized produces identical gradients to ConvTM2d for STE."""
    torch.manual_seed(42)
    
    # Create base and optimized versions with same parameters
    base = ConvTM2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
        n_clauses=64,
        tau=0.5,
        core_backend="tm",
        layer_cls=FuzzyPatternTM_STE,
    )
    
    optimized = ConvTM2dOptimized(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
        n_clauses=64,
        tau=0.5,
        core_backend="tm",
        layer_cls=FuzzyPatternTM_STE,
    )
    
    # Copy weights from base to optimized
    optimized.core.load_state_dict(base.core.state_dict())
    
    # Test input
    x_base = torch.rand(2, 3, 32, 32, requires_grad=True)
    x_opt = x_base.clone().detach().requires_grad_(True)
    
    # Forward pass
    y_base = base(x_base, use_ste=True)
    y_opt = optimized(x_opt, use_ste=True)
    
    # Backward pass with same loss
    loss_base = y_base.mean()
    loss_opt = y_opt.mean()
    
    loss_base.backward()
    loss_opt.backward()
    
    # Check gradient equivalence
    assert x_base.grad is not None and x_opt.grad is not None
    
    grad_max_diff = (x_base.grad - x_opt.grad).abs().max().item()
    grad_mean_diff = (x_base.grad - x_opt.grad).abs().mean().item()
    
    print(f"  Max gradient difference: {grad_max_diff:.2e}")
    print(f"  Mean gradient difference: {grad_mean_diff:.2e}")
    
    assert torch.allclose(x_base.grad, x_opt.grad, atol=1e-5, rtol=1e-4), \
        f"Gradients differ: max_diff={grad_max_diff}, mean_diff={grad_mean_diff}"
    
    print("✓ ConvSTE2d: Optimized matches base (backward)")


def test_convstcm_optimized_vs_base_backward():
    """Test that ConvTM2dOptimized produces identical gradients to ConvTM2d for STCM."""
    torch.manual_seed(42)
    
    # Create base and optimized versions with same parameters
    base = ConvTM2d(
        in_channels=1,
        out_channels=8,
        kernel_size=5,
        stride=1,
        padding=2,
        n_clauses=64,
        tau=0.5,
        core_backend="stcm",
        layer_cls=FuzzyPatternTM_STCM,
        operator="capacity",
        ternary_voting=False,
        ternary_band=0.1,
        ste_temperature=1.0,
    )
    
    optimized = ConvTM2dOptimized(
        in_channels=1,
        out_channels=8,
        kernel_size=5,
        stride=1,
        padding=2,
        n_clauses=64,
        tau=0.5,
        core_backend="stcm",
        layer_cls=FuzzyPatternTM_STCM,
        operator="capacity",
        ternary_voting=False,
        ternary_band=0.1,
        ste_temperature=1.0,
    )
    
    # Copy weights from base to optimized
    optimized.core.load_state_dict(base.core.state_dict())
    
    # Test input
    x_base = torch.rand(2, 1, 28, 28, requires_grad=True)
    x_opt = x_base.clone().detach().requires_grad_(True)
    
    # Forward pass
    y_base = base(x_base, use_ste=True)
    y_opt = optimized(x_opt, use_ste=True)
    
    # Backward pass with same loss
    loss_base = y_base.mean()
    loss_opt = y_opt.mean()
    
    loss_base.backward()
    loss_opt.backward()
    
    # Check gradient equivalence
    assert x_base.grad is not None and x_opt.grad is not None
    
    grad_max_diff = (x_base.grad - x_opt.grad).abs().max().item()
    grad_mean_diff = (x_base.grad - x_opt.grad).abs().mean().item()
    
    print(f"  Max gradient difference: {grad_max_diff:.2e}")
    print(f"  Mean gradient difference: {grad_mean_diff:.2e}")
    
    assert torch.allclose(x_base.grad, x_opt.grad, atol=1e-5, rtol=1e-4), \
        f"Gradients differ: max_diff={grad_max_diff}, mean_diff={grad_mean_diff}"
    
    print("✓ ConvSTCM2d: Optimized matches base (backward)")


if __name__ == "__main__":
    print("\n=== Testing ConvTM2dOptimized Equivalence ===")
    
    print("\n[1/4] ConvSTE2d Forward Pass:")
    test_convste_optimized_vs_base_forward()
    
    print("\n[2/4] ConvSTCM2d Forward Pass:")
    test_convstcm_optimized_vs_base_forward()
    
    print("\n[3/4] ConvSTE2d Backward Pass:")
    test_convste_optimized_vs_base_backward()
    
    print("\n[4/4] ConvSTCM2d Backward Pass:")
    test_convstcm_optimized_vs_base_backward()
    
    print("\n✅ All equivalence tests passed!")
    print("ConvTM2dOptimized produces identical results to ConvTM2d base implementation.")

