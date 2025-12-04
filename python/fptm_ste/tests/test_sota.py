"""
Unit tests for SOTA Hybrid TM architecture.
"""

import unittest
import torch
import torch.nn as nn
from fptm_ste.sota_hybrid import SotaHybridTM
from fptm_ste.backbones import UniversalBackboneFactory

class TestSotaHybridTM(unittest.TestCase):
    
    def test_backbone_integration(self):
        """Verify backbone factory creates correct models."""
        print("\nTesting Backbone Integration...")
        
        # Swin
        model = SotaHybridTM(n_classes=10, backbone="swin_tiny", pretrained=False)
        self.assertEqual(model.num_scales, 4)
        print("  ✅ Swin-Tiny created")
        
        # ResNet (using Universal factory implicitly)
        # Use underscore to separate type and variant for robustness
        model = SotaHybridTM(n_classes=10, backbone="resnet_18", pretrained=False)
        self.assertEqual(model.num_scales, 4) # ResNet typically has 4 stages
        print("  ✅ ResNet18 created")
        
    def test_dual_sigmoid_logic(self):
        """Verify Swin backbone triggers Dual-Sigmoid binarizer (2x channels)."""
        print("\nTesting Dual-Sigmoid Logic...")
        
        model = SotaHybridTM(n_classes=10, backbone="swin_tiny", pretrained=False)
        
        # Check first adapter
        adapter = model.adapters[0]
        self.assertEqual(adapter.mode, "dual")
        self.assertEqual(adapter.output_channels, adapter.num_thresholds * 2)
        print("  ✅ Swin uses Dual-Sigmoid (2x channels)")
        
        # Check ResNet
        model_resnet = SotaHybridTM(n_classes=10, backbone="resnet_18", pretrained=False)
        adapter_resnet = model_resnet.adapters[0]
        self.assertEqual(adapter_resnet.mode, "single")
        self.assertEqual(adapter_resnet.output_channels, adapter_resnet.num_thresholds)
        print("  ✅ ResNet uses Single-Sigmoid")
        
    def test_forward_pass_shape(self):
        """Verify forward pass returns correct shape."""
        print("\nTesting Forward Pass Shape...")
        
        B, C, H, W = 2, 3, 224, 224
        x = torch.randn(B, C, H, W)
        
        model = SotaHybridTM(n_classes=10, backbone="swin_tiny", pretrained=False)
        output = model(x)
        
        self.assertEqual(output.shape, (B, 10))
        print(f"  ✅ Output shape correct: {output.shape}")
        
    def test_gradient_flow(self):
        """Verify gradients flow all the way to backbone."""
        print("\nTesting Gradient Flow...")
        
        B, C, H, W = 2, 3, 224, 224
        x = torch.randn(B, C, H, W)
        y = torch.tensor([0, 1])
        
        model = SotaHybridTM(n_classes=10, backbone="swin_tiny", pretrained=False)
        
        # Enable gradients for backbone (ensure not frozen)
        for p in model.backbone.parameters():
            p.requires_grad = True
            
        logits = model(x, use_ste=True)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        
        # Check backbone gradients
        # Pick a random parameter from the first layer/stage
        backbone_param = list(model.backbone.parameters())[0]
        self.assertIsNotNone(backbone_param.grad)
        self.assertNotEqual(backbone_param.grad.abs().sum().item(), 0.0)
        print("  ✅ Gradients reached backbone")
        
        # Check TM head gradients
        tm_param = list(model.heads[0].parameters())[0]
        self.assertIsNotNone(tm_param.grad)
        print("  ✅ Gradients reached TM Head")

if __name__ == "__main__":
    unittest.main()

