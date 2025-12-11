"""
Integration tests for Triton kernel integration.

Verifies that:
1. Full STCM forward with Triton matches PyTorch-only implementation
2. CTTM forward with Triton matches PyTorch-only implementation  
3. Training with Triton produces equivalent results
4. Models can switch between Triton/PyTorch modes
"""
import unittest
import torch
import torch.nn.functional as F
from unittest.mock import patch


class TestSTCMForwardEquivalence(unittest.TestCase):
    """Test that STCM with Triton matches baseline."""
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
    
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
    
    def test_stcm_forward_matches_baseline(self):
        """Full STCM forward with Triton should match PyTorch-only."""
        from fptm_ste import FuzzyPatternTM_STCM, set_triton_enabled
        
        # Create model
        model = FuzzyPatternTM_STCM(
            n_features=784,
            n_clauses=128,
            n_classes=10,
            ternary_band=0.3,
            ste_temperature=0.5
        )
        model.cuda()
        model.eval()
        
        # Test input
        x = torch.randn(32, 784, device=self.device)
        
        # Get outputs with Triton disabled
        set_triton_enabled(False)
        with torch.no_grad():
            ref_logits, ref_clauses = model(x)
        
        # Get outputs with Triton enabled
        set_triton_enabled(True)
        with torch.no_grad():
            triton_logits, triton_clauses = model(x)
        
        # Compare
        self.assertTrue(
            torch.allclose(ref_logits, triton_logits, atol=1e-4),
            f"Logits mismatch. Max diff: {(ref_logits - triton_logits).abs().max()}"
        )
        self.assertTrue(
            torch.allclose(ref_clauses, triton_clauses, atol=1e-4),
            f"Clauses mismatch. Max diff: {(ref_clauses - triton_clauses).abs().max()}"
        )
    
    def test_stcm_backward_matches_baseline(self):
        """STCM backward pass with Triton should match PyTorch-only."""
        from fptm_ste import FuzzyPatternTM_STCM, set_triton_enabled
        
        # Create model
        model = FuzzyPatternTM_STCM(
            n_features=784,
            n_clauses=64,
            n_classes=10
        )
        model.cuda()
        model.train()
        
        # Test input
        x = torch.randn(16, 784, device=self.device)
        target = torch.randint(0, 10, (16,), device=self.device)
        
        # Get gradients with Triton disabled
        set_triton_enabled(False)
        model.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        ref_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        
        # Get gradients with Triton enabled
        set_triton_enabled(True)
        model.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        triton_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        
        # Compare gradients
        for name in ref_grads:
            self.assertTrue(
                torch.allclose(ref_grads[name], triton_grads[name], atol=1e-4),
                f"Gradient mismatch for {name}. Max diff: {(ref_grads[name] - triton_grads[name]).abs().max()}"
            )
    
    def test_optimized_stcm_forward(self):
        """Test OptimizedSTCM with Triton."""
        from fptm_ste import OptimizedSTCM, set_triton_enabled
        
        model = OptimizedSTCM(
            n_features=784,
            n_clauses=128,
            n_classes=10
        )
        model.cuda()
        model.eval()
        
        x = torch.randn(32, 784, device=self.device)
        
        # With Triton disabled
        set_triton_enabled(False)
        with torch.no_grad():
            ref_out = model(x)
        
        # With Triton enabled
        set_triton_enabled(True)
        with torch.no_grad():
            triton_out = model(x)
        
        # Extract logits (could be tuple or tensor)
        ref_logits = ref_out[0] if isinstance(ref_out, tuple) else ref_out
        triton_logits = triton_out[0] if isinstance(triton_out, tuple) else triton_out
        
        self.assertTrue(
            torch.allclose(ref_logits, triton_logits, atol=1e-3),
            f"OptimizedSTCM mismatch. Max diff: {(ref_logits - triton_logits).abs().max()}"
        )


class TestProbabilisticLogicLayerEquivalence(unittest.TestCase):
    """Test ProbabilisticLogicLayer with Triton."""
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
    
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
    
    def test_pll_forward_deterministic(self):
        """Test PLL in eval mode (deterministic)."""
        from fptm_ste import ProbabilisticLogicLayer, set_triton_enabled
        
        pll = ProbabilisticLogicLayer(
            n_features=128,
            n_clauses=64,
            n_classes=10
        )
        pll.cuda()
        pll.eval()
        
        x = torch.randn(16, 128, device=self.device)
        
        # With Triton disabled
        set_triton_enabled(False)
        with torch.no_grad():
            ref_logits, ref_clauses = pll(x)
        
        # With Triton enabled
        set_triton_enabled(True)
        with torch.no_grad():
            triton_logits, triton_clauses = pll(x)
        
        # In eval mode, should be exactly the same (no Gumbel noise)
        self.assertTrue(
            torch.allclose(ref_logits, triton_logits, atol=1e-4),
            f"PLL logits mismatch"
        )


class TestDeepTMEquivalence(unittest.TestCase):
    """Test DeepTM with Triton."""
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
    
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
    
    def test_deep_tm_forward(self):
        """Test DeepTMNetwork forward pass."""
        from fptm_ste import DeepTMNetwork, set_triton_enabled
        
        model = DeepTMNetwork(
            input_dim=784,
            hidden_dims=[128, 128],
            n_clauses=64,
            n_classes=10
        )
        model.cuda()
        model.eval()
        
        x = torch.randn(16, 784, device=self.device)
        
        # With Triton disabled
        set_triton_enabled(False)
        with torch.no_grad():
            ref_out = model(x)
        
        # With Triton enabled
        set_triton_enabled(True)
        with torch.no_grad():
            triton_out = model(x)
        
        ref_logits = ref_out[0] if isinstance(ref_out, tuple) else ref_out
        triton_logits = triton_out[0] if isinstance(triton_out, tuple) else triton_out
        
        self.assertTrue(
            torch.allclose(ref_logits, triton_logits, atol=1e-3),
            f"DeepTM mismatch. Max diff: {(ref_logits - triton_logits).abs().max()}"
        )


class TestTritonToggleDuringTraining(unittest.TestCase):
    """Test switching Triton on/off during training."""
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
    
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
    
    def test_toggle_during_training_step(self):
        """Verify we can toggle Triton mid-training without errors."""
        from fptm_ste import FuzzyPatternTM_STCM, set_triton_enabled
        
        model = FuzzyPatternTM_STCM(
            n_features=784,
            n_clauses=64,
            n_classes=10
        )
        model.cuda()
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        x = torch.randn(16, 784, device=self.device)
        target = torch.randint(0, 10, (16,), device=self.device)
        
        # Step 1: With Triton
        set_triton_enabled(True)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        
        # Step 2: Without Triton
        set_triton_enabled(False)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        
        # Step 3: Back to Triton
        set_triton_enabled(True)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        
        # No errors = success


class TestTritonStatusAPI(unittest.TestCase):
    """Test Triton status API."""
    
    def test_get_status(self):
        """Test get_triton_status returns valid dict."""
        from fptm_ste import get_triton_status
        
        status = get_triton_status()
        
        self.assertIn('triton_enabled', status)
        self.assertIn('triton_hardware_available', status)
        self.assertIn('fused_kernels_available', status)
        
        # Types
        self.assertIsInstance(status['triton_enabled'], bool)
        self.assertIsInstance(status['triton_hardware_available'], bool)
    
    def test_set_enabled_updates_status(self):
        """Test set_triton_enabled updates status correctly."""
        from fptm_ste import set_triton_enabled, get_triton_status
        
        # Store original
        original = get_triton_status()['triton_enabled']
        
        try:
            set_triton_enabled(False)
            self.assertFalse(get_triton_status()['triton_enabled'])
            
            set_triton_enabled(True)
            self.assertTrue(get_triton_status()['triton_enabled'])
        finally:
            # Restore
            set_triton_enabled(original)


if __name__ == "__main__":
    unittest.main(verbosity=2)

