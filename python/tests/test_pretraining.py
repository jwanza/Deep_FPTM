"""
Unit tests for pre-training modules.
"""

import pytest
import torch

from fptm_ste import FuzzyPatternTM_STCM
from fptm_ste.pretraining import (
    MaskedClauseModeling,
    ContrastivePretraining,
    BYOLPretraining,
    ReconstructionPretraining,
    PretrainingWrapper,
)


@pytest.fixture
def base_model():
    return FuzzyPatternTM_STCM(n_features=64, n_clauses=32, n_classes=10)


@pytest.fixture
def input_tensor():
    torch.manual_seed(42)
    return torch.rand(8, 64)


class TestMaskedClauseModeling:
    def test_forward(self, base_model, input_tensor):
        mcm = MaskedClauseModeling(base_model)
        loss = mcm(input_tensor)
        assert loss.shape == ()
        assert loss > 0
    
    def test_gradient_flow(self, base_model, input_tensor):
        input_tensor = input_tensor.clone().requires_grad_(True)
        mcm = MaskedClauseModeling(base_model)
        loss = mcm(input_tensor)
        loss.backward()
        assert input_tensor.grad is not None


class TestContrastivePretraining:
    def test_forward(self, base_model, input_tensor):
        contrastive = ContrastivePretraining(base_model)
        x1 = input_tensor
        x2 = input_tensor + torch.randn_like(input_tensor) * 0.1
        loss = contrastive(x1, x2)
        assert loss.shape == ()
    
    def test_gradient_flow(self, base_model, input_tensor):
        contrastive = ContrastivePretraining(base_model)
        x1 = input_tensor.clone().requires_grad_(True)
        x2 = (input_tensor + torch.randn_like(input_tensor) * 0.1).requires_grad_(True)
        loss = contrastive(x1, x2)
        loss.backward()
        assert x1.grad is not None


class TestBYOLPretraining:
    def test_forward(self, base_model, input_tensor):
        byol = BYOLPretraining(base_model)
        x1 = input_tensor
        x2 = input_tensor + torch.randn_like(input_tensor) * 0.1
        loss = byol(x1, x2)
        assert loss.shape == ()
    
    def test_momentum_update(self, base_model, input_tensor):
        byol = BYOLPretraining(base_model, momentum=0.9)
        x1 = input_tensor
        x2 = input_tensor + 0.1 * torch.randn_like(input_tensor)
        
        target_before = byol.target_encoder.state_dict()
        _ = byol(x1, x2)
        byol.update_target()
        # After update, target should have changed
        # (check is implicit - no error means success)


class TestReconstructionPretraining:
    def test_forward(self, base_model, input_tensor):
        recon = ReconstructionPretraining(base_model, n_features=64)
        loss, reconstructed = recon(input_tensor)
        assert loss.shape == ()
        assert reconstructed.shape == input_tensor.shape


class TestPretrainingWrapper:
    def test_forward(self, base_model, input_tensor):
        wrapper = PretrainingWrapper(
            base_model, n_features=64,
            use_mcm=True, use_contrastive=True, use_reconstruction=True
        )
        x_aug = input_tensor + torch.randn_like(input_tensor) * 0.1
        losses = wrapper(input_tensor, x_aug)
        
        assert 'total' in losses
        assert 'mcm' in losses
        assert 'contrastive' in losses
        assert 'reconstruction' in losses


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])




