"""
CIFAR-10 regression tests for Triton kernel integration.

Verifies that CIFAR-10 accuracy is maintained when using Triton kernels.
"""
import unittest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class TestCIFARRegression(unittest.TestCase):
    """Verify CIFAR-10 accuracy doesn't degrade with Triton kernels."""
    
    BASELINE_ACCURACY = 0.50  # Conservative baseline for quick test (without full training)
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        
        # Load CIFAR-10 test set
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        cls.test_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
        cls.test_loader = DataLoader(cls.test_dataset, batch_size=256, shuffle=False)
        
        cls.train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        cls.train_loader = DataLoader(cls.train_dataset, batch_size=128, shuffle=True)
    
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
    
    def _train_epoch(self, model, optimizer, n_batches=50):
        """Train for a limited number of batches."""
        model.train()
        total_loss = 0
        for i, (x, y) in enumerate(self.train_loader):
            if i >= n_batches:
                break
            x = x.view(-1, 3 * 32 * 32).to(self.device)
            y = y.to(self.device)
            
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / min(n_batches, len(self.train_loader))
    
    def _evaluate(self, model):
        """Evaluate model on test set."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.view(-1, 3 * 32 * 32).to(self.device)
                y = y.to(self.device)
                logits, _ = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        return correct / total
    
    def test_stcm_trains_on_cifar(self):
        """Test STCM training on CIFAR-10."""
        from fptm_ste import OptimizedSTCM, set_triton_enabled
        
        set_triton_enabled(True)
        
        model = OptimizedSTCM(
            n_features=3 * 32 * 32,  # CIFAR-10 flattened
            n_clauses=256,
            n_classes=10
        )
        model.to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train briefly
        for epoch in range(2):
            loss = self._train_epoch(model, optimizer, n_batches=30)
            print(f"Epoch {epoch+1}: loss={loss:.4f}")
        
        # Evaluate
        accuracy = self._evaluate(model)
        print(f"CIFAR-10 accuracy with Triton: {accuracy:.4f}")
        
        # Should be learning something (better than 10% random)
        self.assertGreater(accuracy, 0.15, 
            f"Expected > 15% accuracy, got {accuracy:.2%}")
    
    def test_triton_vs_pytorch_cifar(self):
        """Compare Triton vs PyTorch on CIFAR-10."""
        from fptm_ste import OptimizedSTCM, set_triton_enabled
        
        # Train with Triton
        torch.manual_seed(42)
        set_triton_enabled(True)
        
        model_triton = OptimizedSTCM(
            n_features=3 * 32 * 32,
            n_clauses=128,
            n_classes=10
        )
        model_triton.to(self.device)
        
        optimizer = torch.optim.Adam(model_triton.parameters(), lr=1e-3)
        for _ in range(2):
            self._train_epoch(model_triton, optimizer, n_batches=20)
        
        acc_triton = self._evaluate(model_triton)
        
        # Train with PyTorch
        torch.manual_seed(42)
        set_triton_enabled(False)
        
        model_pytorch = OptimizedSTCM(
            n_features=3 * 32 * 32,
            n_clauses=128,
            n_classes=10
        )
        model_pytorch.to(self.device)
        
        optimizer = torch.optim.Adam(model_pytorch.parameters(), lr=1e-3)
        for _ in range(2):
            self._train_epoch(model_pytorch, optimizer, n_batches=20)
        
        acc_pytorch = self._evaluate(model_pytorch)
        
        # Re-enable Triton
        set_triton_enabled(True)
        
        print(f"CIFAR Triton accuracy: {acc_triton:.4f}")
        print(f"CIFAR PyTorch accuracy: {acc_pytorch:.4f}")
        print(f"Difference: {abs(acc_triton - acc_pytorch):.4f}")
        
        # Should be within 10% of each other
        self.assertLess(
            abs(acc_triton - acc_pytorch),
            0.10,
            f"Accuracy difference too large: {abs(acc_triton - acc_pytorch):.2%}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)






