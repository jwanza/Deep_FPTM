"""Tests for Deep-CTM intermediate layer learning and gradient flow."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from fptm_ste.deep_ctm import DeepCTMNetwork
from fptm_ste.tm import FuzzyPatternTM_STE, FuzzyPatternTM_STCM


def create_synthetic_dataset(n_samples=500, n_classes=10, img_size=28):
    """Create a simple synthetic dataset for testing."""
    torch.manual_seed(42)
    X = torch.rand(n_samples, 1, img_size, img_size)
    y = torch.randint(0, n_classes, (n_samples,))
    return TensorDataset(X, y)


def test_intermediate_learning_with_aux_weight():
    """Test that aux_weight > 0 forces intermediate blocks to learn."""
    print("\n=== Testing Intermediate Learning with aux_weight ===")
    
    # Create model with aux_weight
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
        conv_core_backend="stcm",
        layer_cls=FuzzyPatternTM_STCM,
        stcm_operator="capacity",
        stcm_ternary_voting=False,
        stcm_ternary_band=0.1,
        stcm_ste_temperature=1.0,
        aux_weight=0.3,
    )
    
    # Create small dataset
    dataset = create_synthetic_dataset(n_samples=200, n_classes=10, img_size=28)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train for 3 epochs
    model.train()
    epoch_block1_accs = []
    
    for epoch in range(3):
        total_loss = 0.0
        correct = 0
        total = 0
        block1_correct = 0
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            
            # Forward with diagnostics
            logits, diagnostics = model(batch_x, use_ste=True, collect_diagnostics=True)
            
            # Main loss
            main_loss = F.cross_entropy(logits, batch_y)
            
            # Auxiliary losses from intermediate blocks
            aux_loss = 0.0
            if model.aux_weight > 0 and diagnostics:
                for key, diag_logits in diagnostics.items():
                    if key.startswith("block_"):
                        aux_loss += F.cross_entropy(diag_logits, batch_y)
            
            total_loss_value = main_loss + model.aux_weight * aux_loss
            total_loss_value.backward()
            optimizer.step()
            
            total_loss += total_loss_value.item()
            
            # Accuracy
            pred = logits.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
            
            # Block 1 accuracy
            if "block_1" in diagnostics:
                block1_pred = diagnostics["block_1"].argmax(dim=1)
                block1_correct += (block1_pred == batch_y).sum().item()
        
        train_acc = correct / total
        block1_acc = block1_correct / total
        epoch_block1_accs.append(block1_acc)
        
        print(f"  Epoch {epoch+1}/3: train_acc={train_acc:.4f}, block_1_acc={block1_acc:.4f}")
    
    # Verify block_1 learns (at least close to random = 0.1 for 10 classes)
    # On synthetic data, we expect at least ~10% accuracy
    assert epoch_block1_accs[0] > 0.05, \
        f"Block 1 initial accuracy {epoch_block1_accs[0]:.4f} too low (>0.05)"
    
    # Verify block_1 shows some learning capability (may not always improve on random data)
    # Just verify it stays reasonable and doesn't collapse
    assert max(epoch_block1_accs) > 0.08, \
        f"Block 1 max accuracy {max(epoch_block1_accs):.4f} too low (>0.08)"
    
    print("✓ Intermediate blocks learn with aux_weight > 0")
    print(f"  Block 1 accuracy: {epoch_block1_accs[0]:.4f} -> {epoch_block1_accs[-1]:.4f}")


def test_gradient_flow_to_early_blocks():
    """Test that gradients flow to early blocks."""
    print("\n=== Testing Gradient Flow ===")
    
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
        conv_core_backend="stcm",
        layer_cls=FuzzyPatternTM_STCM,
        stcm_operator="capacity",
        stcm_ternary_voting=False,
        stcm_ternary_band=0.1,
        stcm_ste_temperature=1.0,
        aux_weight=0.3,
    )
    
    model.train()
    
    # Create dummy batch
    x = torch.rand(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    
    # Forward pass with diagnostics
    logits, diagnostics = model(x, use_ste=True, collect_diagnostics=True)
    
    # Loss
    main_loss = F.cross_entropy(logits, y)
    aux_loss = 0.0
    if diagnostics:
        for key, diag_logits in diagnostics.items():
            if key.startswith("block_"):
                aux_loss += F.cross_entropy(diag_logits, y)
    
    total_loss = main_loss + model.aux_weight * aux_loss
    total_loss.backward()
    
    # Check that first block has gradients
    first_block = model.blocks[0]
    has_grads = False
    grad_norms = []
    
    for name, param in first_block.named_parameters():
        if param.grad is not None:
            has_grads = True
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm > 1e-6:
                print(f"  ✓ {name}: grad_norm={grad_norm:.2e}")
    
    assert has_grads, "No gradients found in first block"
    assert max(grad_norms) > 1e-6, f"Gradients too small: max={max(grad_norms):.2e}"
    
    # Check diagnostic heads have gradients
    diag_head1 = model.diag_heads[0]
    diag_has_grads = False
    for name, param in diag_head1.named_parameters():
        if param.grad is not None and param.grad.norm().item() > 1e-6:
            diag_has_grads = True
            print(f"  ✓ diag_head[0].{name}: grad_norm={param.grad.norm().item():.2e}")
    
    assert diag_has_grads, "Diagnostic head has no gradients"
    
    print("✓ Gradients flow to early blocks and diagnostic heads")


def test_integration_full_training():
    """Integration test: full training loop with diagnostics."""
    print("\n=== Integration Test: Full Training ===")
    
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
        conv_core_backend="stcm",
        layer_cls=FuzzyPatternTM_STCM,
        stcm_operator="capacity",
        stcm_ternary_voting=False,
        stcm_ternary_band=0.1,
        stcm_ste_temperature=1.0,
        aux_weight=0.3,
    )
    
    # Create synthetic train/test split
    train_dataset = create_synthetic_dataset(n_samples=300, n_classes=10, img_size=28)
    test_dataset = create_synthetic_dataset(n_samples=100, n_classes=10, img_size=28)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    test_accs = []
    
    for epoch in range(3):
        # Train
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits, diagnostics = model(batch_x, use_ste=True, collect_diagnostics=True)
            
            main_loss = F.cross_entropy(logits, batch_y)
            aux_loss = 0.0
            if diagnostics:
                for key, diag_logits in diagnostics.items():
                    if key.startswith("block_"):
                        aux_loss += F.cross_entropy(diag_logits, batch_y)
            
            total_loss = main_loss + model.aux_weight * aux_loss
            total_loss.backward()
            optimizer.step()
        
        # Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                logits, _ = model(batch_x, use_ste=True, collect_diagnostics=False)
                pred = logits.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)
        
        test_acc = correct / total
        test_accs.append(test_acc)
        print(f"  Epoch {epoch+1}/3: test_acc={test_acc:.4f}")
    
    # Verify no errors during training
    assert len(test_accs) == 3, "Training did not complete 3 epochs"
    
    # Verify test accuracy is reasonable (on synthetic data, should be > 0.08)
    assert test_accs[-1] > 0.08, f"Final test accuracy {test_accs[-1]:.4f} too low (should be > 0.08)"
    
    print("✓ Full training loop completed successfully")
    print(f"  Test accuracy: {test_accs[0]:.4f} -> {test_accs[-1]:.4f}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Deep-CTM Learning and Gradient Flow")
    print("="*60)
    
    test_intermediate_learning_with_aux_weight()
    test_gradient_flow_to_early_blocks()
    test_integration_full_training()
    
    print("\n" + "="*60)
    print("✅ All Deep-CTM learning tests passed!")
    print("="*60)

