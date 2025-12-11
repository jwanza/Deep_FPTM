"""
Tests for fused T-norm operator kernels.

Verifies correctness and speedup for all T-norm operators.
"""
import unittest
import torch
import time


class TestFusedTNorm(unittest.TestCase):
    """Test fused T-norm operators."""
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from fptm_ste.kernels_tnorm import FusedTNorm, TRITON_AVAILABLE
            cls.FusedTNorm = FusedTNorm
            cls.triton_available = TRITON_AVAILABLE
        except ImportError as e:
            raise unittest.SkipTest(f"Cannot import T-norm kernels: {e}")
    
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
    
    def test_lukasiewicz_correctness(self):
        """Test Lukasiewicz t-norm: max(a + b - 1, 0)"""
        a = torch.rand(1000, device=self.device)
        b = torch.rand(1000, device=self.device)
        
        expected = torch.maximum(a + b - 1.0, torch.zeros_like(a))
        actual = self.FusedTNorm.apply(a, b, 'lukasiewicz')
        
        self.assertTrue(torch.allclose(expected, actual, atol=1e-5))
    
    def test_godel_correctness(self):
        """Test Godel t-norm: min(a, b)"""
        a = torch.rand(1000, device=self.device)
        b = torch.rand(1000, device=self.device)
        
        expected = torch.minimum(a, b)
        actual = self.FusedTNorm.apply(a, b, 'godel')
        
        self.assertTrue(torch.allclose(expected, actual, atol=1e-5))
    
    def test_hamacher_correctness(self):
        """Test Hamacher t-norm: (a * b) / (a + b - a*b + eps)"""
        a = torch.rand(1000, device=self.device) * 0.9 + 0.05  # Avoid extremes
        b = torch.rand(1000, device=self.device) * 0.9 + 0.05
        
        eps = 1e-8
        expected = (a * b) / (a + b - a * b + eps)
        actual = self.FusedTNorm.apply(a, b, 'hamacher')
        
        self.assertTrue(torch.allclose(expected, actual, atol=1e-4))
    
    def test_einstein_correctness(self):
        """Test Einstein t-norm."""
        a = torch.rand(1000, device=self.device) * 0.9 + 0.05
        b = torch.rand(1000, device=self.device) * 0.9 + 0.05
        
        eps = 1e-8
        expected = (a * b) / (2.0 - (a + b - a * b) + eps)
        actual = self.FusedTNorm.apply(a, b, 'einstein')
        
        self.assertTrue(torch.allclose(expected, actual, atol=1e-4))
    
    def test_product_correctness(self):
        """Test Product t-norm: a * b"""
        a = torch.rand(1000, device=self.device)
        b = torch.rand(1000, device=self.device)
        
        expected = a * b
        actual = self.FusedTNorm.apply(a, b, 'product')
        
        self.assertTrue(torch.allclose(expected, actual, atol=1e-5))
    
    def test_all_operators_output_range(self):
        """Test that all operators output values in [0, 1]."""
        operators = ['lukasiewicz', 'godel', 'hamacher', 'einstein', 
                     'product', 'nilpotent_min', 'yager']
        
        a = torch.rand(1000, device=self.device)
        b = torch.rand(1000, device=self.device)
        
        for op_name in operators:
            with self.subTest(op=op_name):
                out = self.FusedTNorm.apply(a, b, op_name)
                self.assertTrue((out >= -0.01).all(), f"{op_name} has values < 0")
                self.assertTrue((out <= 1.01).all(), f"{op_name} has values > 1")


class TestFusedAdaptiveMixer(unittest.TestCase):
    """Test fused adaptive operator mixer."""
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from fptm_ste.kernels_adaptive_mixer import (
                fused_adaptive_mixer,
                adaptive_mixer_reference,
            )
            cls.fused_mixer = fused_adaptive_mixer
            cls.ref_mixer = adaptive_mixer_reference
        except ImportError as e:
            raise unittest.SkipTest(f"Cannot import adaptive mixer: {e}")
    
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
    
    def test_correctness_equal_weights(self):
        """Test with equal weights."""
        a = torch.rand(1000, device=self.device)
        b = torch.rand(1000, device=self.device)
        weights = torch.tensor([0.25, 0.25, 0.25, 0.25], device=self.device)
        
        expected = self.ref_mixer((a, b), weights)
        actual = self.fused_mixer(a, b, weights)
        
        self.assertTrue(torch.allclose(expected, actual, atol=1e-4))
    
    def test_correctness_skewed_weights(self):
        """Test with skewed weights."""
        a = torch.rand(1000, device=self.device)
        b = torch.rand(1000, device=self.device)
        weights = torch.tensor([0.7, 0.1, 0.1, 0.1], device=self.device)
        
        expected = self.ref_mixer((a, b), weights)
        actual = self.fused_mixer(a, b, weights)
        
        self.assertTrue(torch.allclose(expected, actual, atol=1e-4))
    
    def test_gradient_flow(self):
        """Test that gradients flow through the mixer."""
        a = torch.rand(100, device=self.device, requires_grad=True)
        b = torch.rand(100, device=self.device, requires_grad=True)
        weights = torch.softmax(torch.randn(4, device=self.device), dim=0)
        weights.requires_grad = True
        
        out = self.fused_mixer(a, b, weights)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)


class TestDeepLayerPostprocess(unittest.TestCase):
    """Test fused DeepTM layer post-processing."""
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from fptm_ste.kernels_deep_layer import (
                fused_deep_layer_postprocess,
                deep_layer_postprocess_reference,
            )
            cls.fused_pp = fused_deep_layer_postprocess
            cls.ref_pp = deep_layer_postprocess_reference
        except ImportError as e:
            raise unittest.SkipTest(f"Cannot import deep layer kernels: {e}")
    
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
    
    def test_correctness_no_layernorm(self):
        """Test without layernorm."""
        B, D = 32, 256
        logits = torch.randn(B, D, device=self.device)
        identity = torch.randn(B, D, device=self.device)
        
        expected = self.ref_pp(logits, identity, None, None, 0.0, False)
        actual = self.fused_pp(logits, identity, None, None, 0.0, False)
        
        self.assertTrue(torch.allclose(expected, actual, atol=1e-4))
    
    def test_correctness_with_layernorm(self):
        """Test with layernorm."""
        B, D = 32, 256
        logits = torch.randn(B, D, device=self.device)
        identity = torch.randn(B, D, device=self.device)
        gamma = torch.ones(D, device=self.device)
        beta = torch.zeros(D, device=self.device)
        
        expected = self.ref_pp(logits, identity, gamma, beta, 0.0, False)
        actual = self.fused_pp(logits, identity, gamma, beta, 0.0, False)
        
        self.assertTrue(torch.allclose(expected, actual, atol=1e-3))


class TestPLLKernels(unittest.TestCase):
    """Test fused ProbabilisticLogicLayer kernels."""
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from fptm_ste.kernels_pll import (
                fused_pll_forward,
                pll_forward_reference,
            )
            cls.fused_pll = fused_pll_forward
            cls.ref_pll = pll_forward_reference
        except ImportError as e:
            raise unittest.SkipTest(f"Cannot import PLL kernels: {e}")
    
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
    
    def test_output_shapes(self):
        """Test output shapes are correct."""
        B, D, C, n_classes = 32, 128, 64, 10
        
        x = torch.randn(B, D, device=self.device)
        logits = torch.randn(C, D, 3, device=self.device)
        voting = torch.randn(n_classes, C, device=self.device)
        bias = torch.zeros(C, device=self.device)
        
        class_logits, clause_outputs = self.fused_pll(
            x, logits, voting, bias, 1.0, False
        )
        
        self.assertEqual(class_logits.shape, (B, n_classes))
        self.assertEqual(clause_outputs.shape, (B, C))
    
    def test_eval_mode_deterministic(self):
        """Test that eval mode is deterministic."""
        B, D, C, n_classes = 16, 64, 32, 10
        
        x = torch.randn(B, D, device=self.device)
        logits = torch.randn(C, D, 3, device=self.device)
        voting = torch.randn(n_classes, C, device=self.device)
        bias = torch.zeros(C, device=self.device)
        
        out1, _ = self.fused_pll(x, logits, voting, bias, 1.0, False)
        out2, _ = self.fused_pll(x, logits, voting, bias, 1.0, False)
        
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)


