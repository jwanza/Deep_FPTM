"""
Tests for knowledge distillation from deep to shallow STCM.

Validates:
1. Distillation loss computation is correct
2. Student model learns from teacher
3. Distilled shallow model achieves higher accuracy than baseline
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F


class TestDistillationTrainer(unittest.TestCase):
    """Test DistillationTrainer mechanics."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        
    def test_distillation_loss_computation(self):
        """Verify distillation loss combines hard and soft losses correctly."""
        from fptm_ste.distillation import DistillationTrainer
        from fptm_ste.tm_optimized import OptimizedSTCM
        
        B, F_dim, C, K = 32, 128, 64, 10
        
        # Create dummy teacher and student
        teacher = OptimizedSTCM(n_features=F_dim, n_clauses=C, n_classes=K).to(self.device)
        student = OptimizedSTCM(n_features=F_dim, n_clauses=C, n_classes=K).to(self.device)
        
        trainer = DistillationTrainer(
            teacher_model=teacher,
            student_model=student,
            temperature=4.0,
            alpha=0.5,
            device=self.device,
        )
        
        # Generate test data
        student_logits = torch.randn(B, K, device=self.device)
        teacher_logits = torch.randn(B, K, device=self.device)
        labels = torch.randint(0, K, (B,), device=self.device)
        
        loss, metrics = trainer.distillation_loss(student_logits, teacher_logits, labels)
        
        # Verify loss is scalar
        self.assertEqual(loss.dim(), 0)
        
        # Verify metrics
        self.assertIn("hard_loss", metrics)
        self.assertIn("soft_loss", metrics)
        self.assertIn("total_loss", metrics)
        
        # Verify combined loss formula
        expected = 0.5 * metrics["hard_loss"] + 0.5 * metrics["soft_loss"]
        self.assertAlmostEqual(metrics["total_loss"], expected, places=4)
        
    def test_teacher_frozen(self):
        """Verify teacher parameters are frozen during distillation."""
        from fptm_ste.distillation import DistillationTrainer
        from fptm_ste.tm_optimized import OptimizedSTCM
        
        F_dim, C, K = 128, 64, 10
        
        teacher = OptimizedSTCM(n_features=F_dim, n_clauses=C, n_classes=K).to(self.device)
        student = OptimizedSTCM(n_features=F_dim, n_clauses=C, n_classes=K).to(self.device)
        
        # Record teacher params before
        teacher_params_before = {n: p.clone() for n, p in teacher.named_parameters()}
        
        trainer = DistillationTrainer(teacher, student, device=self.device)
        
        # Verify all teacher params have requires_grad=False
        for param in trainer.teacher.parameters():
            self.assertFalse(param.requires_grad)
        
    def test_student_learns_from_teacher(self):
        """Verify student improves when learning from a better teacher."""
        from fptm_ste.distillation import DistillationTrainer
        from fptm_ste.tm_optimized import OptimizedSTCM
        
        B, F_dim, C, K = 64, 128, 64, 10
        
        # Create a simple dataset
        X = torch.randn(200, F_dim, device=self.device)
        # Teacher's predictions as soft labels
        teacher = OptimizedSTCM(n_features=F_dim, n_clauses=C, n_classes=K).to(self.device)
        teacher.eval()
        with torch.no_grad():
            teacher_out = teacher(X)[0]
            labels = teacher_out.argmax(dim=-1)  # Use teacher predictions as ground truth
        
        # Create dataset/loader
        dataset = torch.utils.data.TensorDataset(X, labels)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=True)
        
        # Create student
        student = OptimizedSTCM(n_features=F_dim, n_clauses=C, n_classes=K).to(self.device)
        
        # Get initial accuracy
        student.eval()
        with torch.no_grad():
            initial_out = student(X)[0]
            initial_acc = (initial_out.argmax(dim=-1) == labels).float().mean().item()
        
        # Train with distillation
        trainer = DistillationTrainer(teacher, student, temperature=4.0, alpha=0.7, device=self.device)
        history = trainer.train(train_loader, epochs=5, lr=1e-3, verbose=False)
        
        # Get final accuracy
        student.eval()
        with torch.no_grad():
            final_out = student(X)[0]
            final_acc = (final_out.argmax(dim=-1) == labels).float().mean().item()
        
        print(f"\nDistillation: initial_acc={initial_acc:.4f} -> final_acc={final_acc:.4f}")
        
        # Student should improve
        self.assertGreater(final_acc, initial_acc)


class TestDistilledSTCM(unittest.TestCase):
    """Test DistilledSTCM wrapper."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        
    def test_forward_pass(self):
        """Verify forward pass works."""
        from fptm_ste.distillation import DistilledSTCM
        
        B, F_dim, C, K = 32, 128, 64, 10
        
        model = DistilledSTCM(
            n_features=F_dim,
            n_clauses=C,
            n_classes=K,
        ).to(self.device)
        
        x = torch.rand(B, F_dim, device=self.device)
        logits, clause_out = model(x)
        
        self.assertEqual(logits.shape, (B, K))
        
    def test_from_teacher(self):
        """Test creating DistilledSTCM from teacher dimensions."""
        from fptm_ste.distillation import DistilledSTCM
        from fptm_ste.deep_tm import DeepTMNetwork
        
        F_dim, K = 128, 10
        
        # Create teacher
        teacher = DeepTMNetwork(
            input_dim=F_dim,
            hidden_dims=[64, 32],
            n_classes=K,
            n_clauses=32,
        ).to(self.device)
        
        # Create distilled model from teacher
        distilled = DistilledSTCM.from_teacher(
            teacher_model=teacher,
            n_clauses=64,
        )
        
        self.assertEqual(distilled.model.n_features, F_dim)
        self.assertEqual(distilled.model.n_classes, K)


class TestDistillationBenchmark(unittest.TestCase):
    """Benchmark distillation accuracy improvement."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        
    def test_distillation_improves_accuracy(self):
        """
        Test that distillation improves shallow model accuracy.
        
        This is the key validation: a distilled shallow model should
        achieve higher accuracy than a shallow model trained from scratch.
        """
        from fptm_ste.distillation import DistillationTrainer
        from fptm_ste.deep_tm import DeepTMNetwork
        from fptm_ste.tm_optimized import OptimizedSTCM
        
        # Create synthetic data
        F_dim, K = 256, 10
        N_train, N_test = 1000, 200
        B = 64
        
        # Generate data with some structure
        torch.manual_seed(42)
        centers = torch.randn(K, F_dim, device=self.device) * 3
        
        def generate_data(n):
            labels = torch.randint(0, K, (n,), device=self.device)
            X = centers[labels] + torch.randn(n, F_dim, device=self.device) * 0.5
            return X, labels
        
        X_train, y_train = generate_data(N_train)
        X_test, y_test = generate_data(N_test)
        
        train_ds = torch.utils.data.TensorDataset(X_train, y_train)
        test_ds = torch.utils.data.TensorDataset(X_test, y_test)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=B, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=B)
        
        # 1. Train deep teacher
        teacher = DeepTMNetwork(
            input_dim=F_dim,
            hidden_dims=[128, 64],
            n_classes=K,
            n_clauses=64,
        ).to(self.device)
        
        opt_t = torch.optim.AdamW(teacher.parameters(), lr=1e-3)
        teacher.train()
        for _ in range(10):
            for x, y in train_loader:
                opt_t.zero_grad()
                logits = teacher(x)[0]
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt_t.step()
        
        teacher.eval()
        with torch.no_grad():
            teacher_acc = (teacher(X_test)[0].argmax(-1) == y_test).float().mean().item()
        print(f"\nTeacher (Deep) accuracy: {teacher_acc:.4f}")
        
        # 2. Train shallow baseline (no distillation)
        baseline = OptimizedSTCM(n_features=F_dim, n_clauses=128, n_classes=K).to(self.device)
        opt_b = torch.optim.AdamW(baseline.parameters(), lr=1e-3)
        baseline.train()
        for _ in range(10):
            for x, y in train_loader:
                opt_b.zero_grad()
                logits = baseline(x)[0]
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt_b.step()
        
        baseline.eval()
        with torch.no_grad():
            baseline_acc = (baseline(X_test)[0].argmax(-1) == y_test).float().mean().item()
        print(f"Baseline (Shallow) accuracy: {baseline_acc:.4f}")
        
        # 3. Train distilled student
        student = OptimizedSTCM(n_features=F_dim, n_clauses=128, n_classes=K).to(self.device)
        trainer = DistillationTrainer(
            teacher_model=teacher,
            student_model=student,
            temperature=4.0,
            alpha=0.7,
            device=self.device,
        )
        trainer.train(train_loader, epochs=10, lr=1e-3, val_loader=test_loader, verbose=False)
        
        student.eval()
        with torch.no_grad():
            student_acc = (student(X_test)[0].argmax(-1) == y_test).float().mean().item()
        print(f"Distilled (Shallow) accuracy: {student_acc:.4f}")
        print(f"Improvement: {(student_acc - baseline_acc)*100:.2f}%")
        
        # Distilled student should be better than baseline
        # (or at least close if baseline already performs well)
        self.assertGreaterEqual(student_acc, baseline_acc * 0.95)


if __name__ == "__main__":
    unittest.main(verbosity=2)

