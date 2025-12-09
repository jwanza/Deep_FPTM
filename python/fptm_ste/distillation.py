"""
Deep-to-Shallow Knowledge Distillation for STCM.

This module implements knowledge distillation from deep STCM models (DeepTMNetwork)
to shallow STCM models, enabling:
1. Transfer of learned representations from deep to shallow
2. Higher accuracy on shallow models (+1.5% expected)
3. Faster inference with shallow model performance closer to deep

Key Components:
- DistillationTrainer: Handles the distillation training loop
- DistilledSTCM: A shallow STCM trained via distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class DistillationTrainer:
    """
    Knowledge distillation trainer for transferring knowledge from
    a deep STCM model to a shallow student model.
    
    Uses a combination of:
    1. Hard labels (cross-entropy with true labels)
    2. Soft labels (KL divergence with teacher logits)
    3. Optional: Feature matching (clause activation similarity)
    
    Args:
        teacher_model: Pre-trained deep model (frozen during distillation)
        student_model: Shallow model to train
        temperature: Temperature for softening logits (higher = softer)
        alpha: Weight for soft label loss (1-alpha for hard labels)
        feature_matching_weight: Weight for intermediate feature matching
    
    Example:
        >>> teacher = DeepTMNetwork(...).load_state_dict(...)
        >>> student = FuzzyPatternTM_STCM(...)
        >>> trainer = DistillationTrainer(teacher, student, temperature=4.0)
        >>> trainer.train(train_loader, epochs=10)
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        feature_matching_weight: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.feature_matching_weight = feature_matching_weight
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Move to device
        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)
        
    def distillation_loss(
        self, 
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined distillation loss.
        
        Loss = (1 - alpha) * CE(student, labels) + alpha * T^2 * KL(soft_student, soft_teacher)
        
        The T^2 factor ensures gradients are of similar magnitude for different T values.
        """
        T = self.temperature
        
        # Hard label loss (cross-entropy with true labels)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Soft label loss (KL divergence with teacher)
        # log_softmax for numerical stability
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
        
        # Combined loss
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        
        metrics = {
            "hard_loss": hard_loss.item(),
            "soft_loss": soft_loss.item(),
            "total_loss": total_loss.item(),
        }
        
        return total_loss, metrics
    
    @torch.no_grad()
    def get_teacher_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Get teacher logits for a batch."""
        self.teacher.eval()
        output = self.teacher(x)
        # Handle tuple returns (logits, clause_outputs)
        if isinstance(output, tuple):
            return output[0]
        return output
    
    def train_step(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Single training step with distillation."""
        x = x.to(self.device)
        labels = labels.to(self.device)
        
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Get teacher predictions (no grad)
        teacher_logits = self.get_teacher_outputs(x)
        
        # Get student predictions
        self.student.train()
        student_output = self.student(x)
        if isinstance(student_output, tuple):
            student_logits = student_output[0]
        else:
            student_logits = student_output
        
        # Compute loss
        loss, metrics = self.distillation_loss(student_logits, teacher_logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        optimizer.step()
        
        return metrics
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Full training loop with distillation.
        
        Args:
            train_loader: Training data loader (returns (x, labels))
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            val_loader: Optional validation loader
            verbose: Print progress
            
        Returns:
            Dictionary with training history
        """
        optimizer = AdamW(self.student.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        
        history = {
            "train_loss": [],
            "hard_loss": [],
            "soft_loss": [],
            "val_acc": [],
        }
        
        for epoch in range(epochs):
            self.student.train()
            epoch_losses = []
            epoch_hard = []
            epoch_soft = []
            
            for batch_idx, (x, labels) in enumerate(train_loader):
                metrics = self.train_step(x, labels, optimizer)
                epoch_losses.append(metrics["total_loss"])
                epoch_hard.append(metrics["hard_loss"])
                epoch_soft.append(metrics["soft_loss"])
            
            scheduler.step()
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_hard = sum(epoch_hard) / len(epoch_hard)
            avg_soft = sum(epoch_soft) / len(epoch_soft)
            
            history["train_loss"].append(avg_loss)
            history["hard_loss"].append(avg_hard)
            history["soft_loss"].append(avg_soft)
            
            # Validation
            val_acc = None
            if val_loader is not None:
                val_acc = self.evaluate(val_loader)
                history["val_acc"].append(val_acc)
            
            if verbose:
                val_str = f" | val_acc={val_acc:.4f}" if val_acc else ""
                print(f"Distill epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | "
                      f"hard={avg_hard:.4f} | soft={avg_soft:.4f}{val_str}")
        
        return history
    
    @torch.no_grad()
    def evaluate(self, data_loader: torch.utils.data.DataLoader) -> float:
        """Evaluate student model accuracy."""
        self.student.eval()
        correct = 0
        total = 0
        
        for x, labels in data_loader:
            x = x.to(self.device)
            labels = labels.to(self.device)
            
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            
            output = self.student(x)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        return correct / total


class DistilledSTCM(nn.Module):
    """
    A convenience wrapper for creating a distilled STCM.
    
    This creates a shallow STCM and provides methods for distillation training
    from a pre-trained deep model.
    
    Example:
        >>> teacher = DeepTMNetwork(...)  # Pre-trained
        >>> distilled = DistilledSTCM.from_teacher(teacher, n_clauses=512)
        >>> distilled.distill(train_loader, epochs=10)
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        stcm_class: Optional[type] = None,
        **stcm_kwargs,
    ):
        super().__init__()
        
        # Import here to avoid circular imports
        if stcm_class is None:
            from .tm_optimized import OptimizedSTCM
            stcm_class = OptimizedSTCM
        
        self.model = stcm_class(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            **stcm_kwargs,
        )
        self.distillation_history = None
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)
    
    def distill(
        self,
        teacher_model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        temperature: float = 4.0,
        alpha: float = 0.5,
        lr: float = 1e-3,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Train this model via distillation from a teacher.
        
        Args:
            teacher_model: Pre-trained teacher (e.g., DeepTMNetwork)
            train_loader: Training data
            epochs: Distillation epochs
            temperature: Softmax temperature
            alpha: Weight for soft labels
            lr: Learning rate
            val_loader: Optional validation data
            device: Device to train on
            
        Returns:
            Training history
        """
        trainer = DistillationTrainer(
            teacher_model=teacher_model,
            student_model=self.model,
            temperature=temperature,
            alpha=alpha,
            device=device,
        )
        
        self.distillation_history = trainer.train(
            train_loader=train_loader,
            epochs=epochs,
            lr=lr,
            val_loader=val_loader,
        )
        
        return self.distillation_history
    
    @classmethod
    def from_teacher(
        cls,
        teacher_model: nn.Module,
        n_clauses: int = 512,
        stcm_class: Optional[type] = None,
        **stcm_kwargs,
    ) -> "DistilledSTCM":
        """
        Create a DistilledSTCM with architecture matching the teacher's input/output.
        
        Automatically infers n_features and n_classes from the teacher.
        """
        # Try to infer n_classes from teacher
        n_classes = None
        if hasattr(teacher_model, 'n_classes'):
            n_classes = teacher_model.n_classes
        elif hasattr(teacher_model, 'head') and hasattr(teacher_model.head, 'n_classes'):
            n_classes = teacher_model.head.n_classes
        elif hasattr(teacher_model, 'classifier') and hasattr(teacher_model.classifier, 'n_classes'):
            n_classes = teacher_model.classifier.n_classes
        
        if n_classes is None:
            raise ValueError("Cannot infer n_classes from teacher. Provide explicitly.")
        
        # Try to infer n_features from teacher
        n_features = None
        if hasattr(teacher_model, 'input_dim'):
            n_features = teacher_model.input_dim
        elif hasattr(teacher_model, 'n_features'):
            n_features = teacher_model.n_features
        elif hasattr(teacher_model, 'layers') and len(teacher_model.layers) > 0:
            first_layer = teacher_model.layers[0]
            if hasattr(first_layer, 'n_features'):
                n_features = first_layer.n_features
        
        if n_features is None:
            raise ValueError("Cannot infer n_features from teacher. Provide explicitly.")
        
        return cls(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            stcm_class=stcm_class,
            **stcm_kwargs,
        )

