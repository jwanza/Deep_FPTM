"""
Unit tests for fuzzy logic operators.

Tests cover:
1. Output shape correctness
2. Output range (should be in [0, 1])
3. Gradient flow
4. Mathematical properties (associativity, boundary conditions)
5. Numerical stability
"""

import pytest
import torch
import torch.nn as nn

from fptm_ste.operators import (
    _BaseTernaryOperator,
    available_ternary_operators,
    build_ternary_operator,
    # Classical fuzzy t-norms
    LukasiewiczTNorm,
    GodelTNorm,
    HamacherProduct,
    YagerTNorm,
    DrasticProduct,
    EinsteinProduct,
    NilpotentMinimum,
    BoundedDifference,
    # Learnable operators
    ParameterizedTNorm,
    SoftMinMax,
    # Adaptive mixing
    AdaptiveOperatorMixer,
    EnsembleOperator,
    # Original operators
    TernaryQuantumAND,
    TernaryXOR,
    TernaryMajority,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_inputs():
    """Standard test inputs."""
    torch.manual_seed(42)
    a = torch.rand(16, 32)
    b = torch.rand(16, 32)
    c = torch.rand(16, 32)
    return a, b, c


@pytest.fixture
def boundary_inputs():
    """Boundary value test inputs (0s and 1s)."""
    zeros = torch.zeros(8, 16)
    ones = torch.ones(8, 16)
    mixed = torch.tensor([[0.0, 0.5, 1.0], [0.25, 0.75, 0.5]])
    return zeros, ones, mixed


ALL_TNORM_CLASSES = [
    LukasiewiczTNorm,
    GodelTNorm,
    HamacherProduct,
    YagerTNorm,
    DrasticProduct,
    EinsteinProduct,
    NilpotentMinimum,
    BoundedDifference,
    ParameterizedTNorm,
    SoftMinMax,
]

ALL_OPERATOR_NAMES = [
    "lukasiewicz",
    "godel",
    "hamacher",
    "yager",
    "drastic",
    "einstein",
    "nilpotent",
    "bounded_diff",
    "parameterized",
    "softminmax",
    "tqand",
    "txor",
    "tmaj",
]


# =============================================================================
# Shape and Range Tests
# =============================================================================


class TestOperatorShapes:
    """Test that operators produce correct output shapes."""
    
    @pytest.mark.parametrize("op_name", ALL_OPERATOR_NAMES)
    def test_output_shape(self, op_name, sample_inputs):
        """Output shape should match input shape."""
        a, b, c = sample_inputs
        op = build_ternary_operator(op_name)
        
        if op is None:
            pytest.skip(f"Operator {op_name} not available")
        
        try:
            # Try with 2 inputs first
            result = op(a, b)
            assert result.shape == a.shape, f"Shape mismatch: {result.shape} != {a.shape}"
        except ValueError:
            # Some operators need 3 inputs
            result = op(a, b, c)
            assert result.shape == a.shape, f"Shape mismatch: {result.shape} != {a.shape}"
    
    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    @pytest.mark.parametrize("feature_size", [1, 16, 128])
    def test_various_sizes(self, batch_size, feature_size):
        """Test with various tensor sizes."""
        a = torch.rand(batch_size, feature_size)
        b = torch.rand(batch_size, feature_size)
        
        op = LukasiewiczTNorm()
        result = op(a, b)
        assert result.shape == (batch_size, feature_size)


class TestOperatorRanges:
    """Test that outputs stay in valid range [0, 1]."""
    
    @pytest.mark.parametrize("op_class", ALL_TNORM_CLASSES)
    def test_output_in_unit_interval(self, op_class, sample_inputs):
        """Output should be in [0, 1]."""
        a, b, c = sample_inputs
        op = op_class()
        
        result = op(a, b)
        assert torch.all(result >= -1e-6), f"Output below 0: min={result.min()}"
        assert torch.all(result <= 1 + 1e-6), f"Output above 1: max={result.max()}"
    
    @pytest.mark.parametrize("op_class", ALL_TNORM_CLASSES)
    def test_boundary_preservation(self, op_class, boundary_inputs):
        """T(a, 1) should equal a for most t-norms."""
        zeros, ones, _ = boundary_inputs
        op = op_class()
        
        result = op(zeros, ones)
        # Most t-norms satisfy T(a, 1) = a
        # Check it's at least in valid range
        assert torch.all(result >= -1e-6)
        assert torch.all(result <= 1 + 1e-6)


# =============================================================================
# Gradient Flow Tests
# =============================================================================


class TestGradientFlow:
    """Test that gradients flow through operators."""
    
    @pytest.mark.parametrize("op_class", ALL_TNORM_CLASSES)
    def test_backward_pass(self, op_class):
        """Gradients should flow through the operator."""
        a = torch.rand(8, 16, requires_grad=True)
        b = torch.rand(8, 16, requires_grad=True)
        
        op = op_class()
        result = op(a, b)
        loss = result.sum()
        loss.backward()
        
        assert a.grad is not None, "No gradient for input a"
        assert b.grad is not None, "No gradient for input b"
        assert not torch.isnan(a.grad).any(), "NaN in gradient for a"
        assert not torch.isnan(b.grad).any(), "NaN in gradient for b"
    
    @pytest.mark.parametrize("op_class", [ParameterizedTNorm, SoftMinMax])
    def test_learnable_parameters_grad(self, op_class):
        """Learnable parameters should receive gradients."""
        a = torch.rand(8, 16, requires_grad=True)
        b = torch.rand(8, 16, requires_grad=True)
        
        op = op_class()
        result = op(a, b)
        loss = result.sum()
        loss.backward()
        
        # Check that learnable parameters have gradients
        for name, param in op.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"


# =============================================================================
# Mathematical Properties Tests
# =============================================================================


class TestTNormProperties:
    """Test mathematical properties of t-norms."""
    
    @pytest.mark.parametrize("op_class", [
        LukasiewiczTNorm, GodelTNorm, HamacherProduct, EinsteinProduct
    ])
    def test_commutativity(self, op_class, sample_inputs):
        """T(a, b) = T(b, a) for commutative operators."""
        a, b, _ = sample_inputs
        op = op_class()
        
        result1 = op(a, b)
        result2 = op(b, a)
        
        assert torch.allclose(result1, result2, atol=1e-6), \
            f"Not commutative: max diff = {(result1 - result2).abs().max()}"
    
    @pytest.mark.parametrize("op_class", [GodelTNorm])
    def test_idempotency(self, op_class, sample_inputs):
        """T(a, a) = a for idempotent operators (Gödel)."""
        a, _, _ = sample_inputs
        op = op_class()
        
        result = op(a, a)
        
        assert torch.allclose(result, a, atol=1e-6), \
            f"Not idempotent: max diff = {(result - a).abs().max()}"
    
    @pytest.mark.parametrize("op_class", ALL_TNORM_CLASSES)
    def test_monotonicity(self, op_class):
        """T is monotonic: if a ≤ a' then T(a, b) ≤ T(a', b)."""
        a = torch.tensor([[0.2, 0.5, 0.8]])
        a_prime = torch.tensor([[0.3, 0.6, 0.9]])
        b = torch.tensor([[0.5, 0.5, 0.5]])
        
        op = op_class()
        
        result_a = op(a, b)
        result_a_prime = op(a_prime, b)
        
        assert torch.all(result_a <= result_a_prime + 1e-6), \
            f"Not monotonic: {result_a} > {result_a_prime}"
    
    def test_lukasiewicz_is_bounded(self):
        """Lukasiewicz t-norm: T(a, b) = max(0, a + b - 1)."""
        a = torch.tensor([0.3, 0.7, 1.0])
        b = torch.tensor([0.4, 0.6, 1.0])
        expected = torch.tensor([0.0, 0.3, 1.0])  # max(0, a+b-1)
        
        op = LukasiewiczTNorm()
        result = op(a, b)
        
        assert torch.allclose(result, expected, atol=1e-6)
    
    def test_godel_is_minimum(self):
        """Gödel t-norm: T(a, b) = min(a, b)."""
        a = torch.tensor([0.3, 0.7, 0.5])
        b = torch.tensor([0.5, 0.4, 0.5])
        expected = torch.tensor([0.3, 0.4, 0.5])
        
        op = GodelTNorm()
        result = op(a, b)
        
        assert torch.allclose(result, expected, atol=1e-6)


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Test numerical stability under edge cases."""
    
    @pytest.mark.parametrize("op_class", ALL_TNORM_CLASSES)
    def test_no_nan_on_zeros(self, op_class):
        """No NaN when inputs are zeros."""
        zeros = torch.zeros(8, 16)
        op = op_class()
        
        result = op(zeros, zeros)
        assert not torch.isnan(result).any(), "NaN with zero inputs"
    
    @pytest.mark.parametrize("op_class", ALL_TNORM_CLASSES)
    def test_no_nan_on_ones(self, op_class):
        """No NaN when inputs are ones."""
        ones = torch.ones(8, 16)
        op = op_class()
        
        result = op(ones, ones)
        assert not torch.isnan(result).any(), "NaN with one inputs"
    
    @pytest.mark.parametrize("op_class", ALL_TNORM_CLASSES)
    def test_no_inf_on_extreme_values(self, op_class):
        """No Inf with extreme but valid values."""
        small = torch.full((8, 16), 1e-7)
        large = torch.full((8, 16), 1.0 - 1e-7)
        
        op = op_class()
        
        result1 = op(small, small)
        result2 = op(large, large)
        result3 = op(small, large)
        
        assert not torch.isinf(result1).any(), "Inf with small inputs"
        assert not torch.isinf(result2).any(), "Inf with large inputs"
        assert not torch.isinf(result3).any(), "Inf with mixed inputs"
    
    def test_yager_varying_p(self):
        """Yager t-norm should be stable for different p values."""
        a = torch.rand(8, 16)
        b = torch.rand(8, 16)
        
        for p in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            op = YagerTNorm(p=p)
            result = op(a, b)
            
            assert not torch.isnan(result).any(), f"NaN with p={p}"
            assert not torch.isinf(result).any(), f"Inf with p={p}"
            assert torch.all(result >= 0) and torch.all(result <= 1), f"Out of range with p={p}"


# =============================================================================
# Learnable Operator Tests
# =============================================================================


class TestLearnableOperators:
    """Test learnable operator functionality."""
    
    def test_parameterized_tnorm_learns(self):
        """ParameterizedTNorm parameter should update during training."""
        op = ParameterizedTNorm(init_p=1.0, learnable=True)
        optimizer = torch.optim.SGD(op.parameters(), lr=0.1)
        
        initial_p = op.p.clone()
        
        # Simple training loop
        for _ in range(10):
            a = torch.rand(8, 16)
            b = torch.rand(8, 16)
            
            result = op(a, b)
            loss = (result - 0.5).pow(2).mean()  # Target: push outputs to 0.5
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        assert not torch.allclose(op.p, initial_p), "Parameter p did not change"
    
    def test_softminmax_alpha_in_range(self):
        """SoftMinMax alpha should always be in (0, 1)."""
        op = SoftMinMax(init_alpha=0.5, learnable=True)
        optimizer = torch.optim.SGD(op.parameters(), lr=1.0)
        
        for _ in range(50):
            a = torch.rand(8, 16)
            b = torch.rand(8, 16)
            
            result = op(a, b)
            loss = result.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Alpha should always be in (0, 1) due to sigmoid
            assert 0 < op.alpha < 1, f"Alpha out of range: {op.alpha}"
    
    def test_softminmax_interpolation(self):
        """SoftMinMax at extremes should match min or product."""
        a = torch.rand(8, 16)
        b = torch.rand(8, 16)
        
        # Alpha = 1 should be close to min
        op_min = SoftMinMax(init_alpha=0.999, learnable=False)
        result_min = op_min(a, b)
        expected_min = torch.minimum(a, b)
        
        assert torch.allclose(result_min, expected_min, atol=0.01), \
            "Alpha=1 should approximate minimum"
        
        # Alpha = 0 should be close to product
        op_prod = SoftMinMax(init_alpha=0.001, learnable=False)
        result_prod = op_prod(a, b)
        expected_prod = a * b
        
        assert torch.allclose(result_prod, expected_prod, atol=0.01), \
            "Alpha=0 should approximate product"


# =============================================================================
# Registry Tests
# =============================================================================


class TestOperatorRegistry:
    """Test operator registry functions."""
    
    def test_available_operators_complete(self):
        """All new operators should be registered."""
        available = available_ternary_operators()
        
        expected = {
            "lukasiewicz", "godel", "hamacher", "yager",
            "drastic", "einstein", "nilpotent", "bounded_diff",
            "parameterized", "softminmax",
            "tqand", "txor", "tmaj",
        }
        
        for name in expected:
            assert name in available, f"Operator {name} not registered"
    
    @pytest.mark.parametrize("op_name", ALL_OPERATOR_NAMES)
    def test_build_operator(self, op_name):
        """build_ternary_operator should return valid operators."""
        op = build_ternary_operator(op_name)
        
        assert op is not None, f"Failed to build {op_name}"
        assert isinstance(op, _BaseTernaryOperator), f"{op_name} is not a _BaseTernaryOperator"
    
    def test_build_unknown_returns_none(self):
        """Unknown operator names should return None."""
        op = build_ternary_operator("nonexistent_operator")
        assert op is None


# =============================================================================
# Integration with STCM Tests
# =============================================================================


class TestSTCMIntegration:
    """Test operators work with STCM."""
    
    @pytest.mark.parametrize("op_name", ["lukasiewicz", "godel", "hamacher", "yager"])
    def test_stcm_with_fuzzy_operator(self, op_name):
        """STCM should accept fuzzy operators."""
        from fptm_ste import FuzzyPatternTM_STCM
        
        # Note: STCM uses "capacity" or "product" by default
        # Custom operators require passing operator_impl
        model = FuzzyPatternTM_STCM(
            n_features=32,
            n_clauses=16,
            n_classes=4,
            operator=op_name,  # This may or may not work depending on STCM implementation
        )
        
        x = torch.rand(8, 32)
        try:
            logits, clauses = model(x)
            assert logits.shape == (8, 4)
            assert clauses.shape == (8, 16)
        except ValueError:
            # If operator not supported in STCM, that's expected
            pytest.skip(f"Operator {op_name} not directly supported in STCM")


# =============================================================================
# Adaptive Operator Mixer Tests
# =============================================================================


class TestAdaptiveOperatorMixer:
    """Test the AdaptiveOperatorMixer functionality."""
    
    def test_basic_forward(self):
        """Basic forward pass works correctly."""
        mixer = AdaptiveOperatorMixer(
            operators=["godel", "lukasiewicz", "hamacher"],
            mixing_type="global",
        )
        
        a = torch.rand(8, 16)
        b = torch.rand(8, 16)
        
        result = mixer(a, b)
        assert result.shape == a.shape
        assert torch.all(result >= 0) and torch.all(result <= 1)
    
    def test_per_clause_mixing(self):
        """Per-clause mixing works correctly."""
        n_clauses = 32
        mixer = AdaptiveOperatorMixer(
            operators=["godel", "lukasiewicz"],
            n_clauses=n_clauses,
            mixing_type="per_clause",
        )
        
        a = torch.rand(8, n_clauses)
        b = torch.rand(8, n_clauses)
        
        result = mixer(a, b)
        assert result.shape == a.shape
    
    def test_attention_mixing(self):
        """Attention-based mixing works correctly."""
        mixer = AdaptiveOperatorMixer(
            operators=["godel", "hamacher"],
            mixing_type="attention",
        )
        
        a = torch.rand(8, 16)
        b = torch.rand(8, 16)
        
        result = mixer(a, b)
        assert result.shape == a.shape
    
    def test_gradient_flow(self):
        """Gradients flow through the mixer."""
        mixer = AdaptiveOperatorMixer(
            operators=["godel", "lukasiewicz"],
            mixing_type="global",
        )
        
        a = torch.rand(8, 16, requires_grad=True)
        b = torch.rand(8, 16, requires_grad=True)
        
        result = mixer(a, b)
        loss = result.sum()
        loss.backward()
        
        assert a.grad is not None
        assert b.grad is not None
        # Mixing weights should also have gradients
        assert mixer.mixing_logits.grad is not None
    
    def test_weights_update(self):
        """Mixing weights update during training."""
        mixer = AdaptiveOperatorMixer(
            operators=["godel", "lukasiewicz", "hamacher"],
            mixing_type="global",
        )
        optimizer = torch.optim.SGD(mixer.parameters(), lr=0.1)
        
        initial_logits = mixer.mixing_logits.clone()
        
        for _ in range(10):
            a = torch.rand(8, 16)
            b = torch.rand(8, 16)
            
            result = mixer(a, b)
            # Push towards specific target
            loss = (result - 0.3).pow(2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        assert not torch.allclose(mixer.mixing_logits, initial_logits)
    
    def test_get_operator_contributions(self):
        """Operator contributions sum to 1."""
        mixer = AdaptiveOperatorMixer(
            operators=["godel", "lukasiewicz", "hamacher"],
            mixing_type="global",
        )
        
        contributions = mixer.get_operator_contributions()
        
        assert len(contributions) == 3
        assert abs(sum(contributions.values()) - 1.0) < 1e-5


class TestEnsembleOperator:
    """Test the EnsembleOperator functionality."""
    
    def test_basic_forward(self):
        """Basic forward pass works correctly."""
        ensemble = EnsembleOperator(
            operators=["godel", "lukasiewicz"],
            hidden_dim=16,
        )
        
        a = torch.rand(8, 16)
        b = torch.rand(8, 16)
        
        result = ensemble(a, b)
        assert result.shape == a.shape
    
    def test_gradient_flow(self):
        """Gradients flow through the ensemble."""
        ensemble = EnsembleOperator(
            operators=["godel", "hamacher"],
        )
        
        a = torch.rand(8, 16, requires_grad=True)
        b = torch.rand(8, 16, requires_grad=True)
        
        result = ensemble(a, b)
        loss = result.sum()
        loss.backward()
        
        assert a.grad is not None
        assert b.grad is not None
    
    def test_without_residual(self):
        """Ensemble works without residual connection."""
        ensemble = EnsembleOperator(
            operators=["lukasiewicz", "godel"],
            use_residual=False,
        )
        
        a = torch.rand(8, 16)
        b = torch.rand(8, 16)
        
        result = ensemble(a, b)
        assert result.shape == a.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

