"""
Continual Learning Module for Tsetlin Machines.

This module implements state-of-the-art continual learning methods adapted
for Tsetlin Machines, enabling learning across multiple tasks without
catastrophic forgetting.

Implemented Methods:
1. EWC (Elastic Weight Consolidation) - Regularization-based
2. SI (Synaptic Intelligence) - Online importance estimation
3. MAS (Memory Aware Synapses) - Gradient-free importance
4. GEM (Gradient Episodic Memory) - Constrained optimization
5. PackNet - Architecture-based pruning and freezing
6. Experience Replay - Rehearsal-based
7. Progressive Networks - Expandable architecture

References:
- Kirkpatrick et al. (2017): Overcoming catastrophic forgetting in neural networks
- Zenke et al. (2017): Continual Learning Through Synaptic Intelligence
- Aljundi et al. (2018): Memory Aware Synapses
- Lopez-Paz & Ranzato (2017): Gradient Episodic Memory
- Mallya & Lazebnik (2018): PackNet
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .tm import FuzzyPatternTM_STCM, FuzzyPatternTM_STE


# =============================================================================
# Base Continual Learning Wrapper
# =============================================================================


class ContinualLearningWrapper(nn.Module):
    """
    Base class for continual learning methods.
    
    Wraps a base TM model and adds continual learning capabilities.
    Subclasses implement specific methods (EWC, SI, etc.).
    
    Args:
        base_model: Base Tsetlin Machine model
        lamb: Regularization strength for importance-weighted penalties
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        lamb: float = 1.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.lamb = lamb
        self.current_task = 0
        
        # Store parameter names for importance tracking
        self._param_names = [
            name for name, _ in base_model.named_parameters()
            if _.requires_grad
        ]
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through base model."""
        return self.base_model(x, **kwargs)
    
    def penalty(self) -> torch.Tensor:
        """Compute continual learning penalty. Override in subclasses."""
        return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def consolidate_task(self, dataloader: Optional[DataLoader] = None) -> None:
        """Consolidate after task completion. Override in subclasses."""
        self.current_task += 1
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get trainable parameters (for methods that freeze params)."""
        return [p for p in self.base_model.parameters() if p.requires_grad]


# =============================================================================
# Elastic Weight Consolidation (EWC)
# =============================================================================


class EWCClauseMachine(ContinualLearningWrapper):
    """
    Elastic Weight Consolidation for Tsetlin Machines.
    
    Computes Fisher Information Matrix to identify important parameters
    for previous tasks, then penalizes changes to those parameters.
    
    Key formula:
    L_EWC = L_task + (λ/2) * Σ_i F_i * (θ_i - θ*_i)^2
    
    where F_i is the Fisher information and θ*_i are the optimal parameters.
    
    Args:
        base_model: Base TM model
        lamb: EWC regularization strength
        fisher_n_samples: Number of samples for Fisher estimation
        online: Use online EWC (running average of Fisher)
        gamma: Decay factor for online EWC
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        lamb: float = 1000.0,
        fisher_n_samples: Optional[int] = None,
        online: bool = False,
        gamma: float = 1.0,
    ):
        super().__init__(base_model, lamb)
        self.fisher_n_samples = fisher_n_samples
        self.online = online
        self.gamma = gamma
        
        # Storage for Fisher matrices and optimal parameters
        self.fisher: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        
        # For online EWC
        self.running_fisher: Dict[str, torch.Tensor] = {}
    
    def compute_fisher_information(
        self,
        dataloader: DataLoader,
        n_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diagonal Fisher Information Matrix.
        
        Uses empirical Fisher (gradient of log-likelihood) which is
        equivalent to expected gradient squared under model distribution.
        
        Args:
            dataloader: DataLoader for computing Fisher
            n_samples: Maximum samples to use (None = all)
            
        Returns:
            Dict mapping parameter names to Fisher diagonal
        """
        n_samples = n_samples or self.fisher_n_samples
        
        fisher = {name: torch.zeros_like(param)
                  for name, param in self.base_model.named_parameters()
                  if param.requires_grad}
        
        self.base_model.train()
        n_used = 0
        
        for batch in dataloader:
            if n_samples is not None and n_used >= n_samples:
                break
            
            x, y = batch[0], batch[1]
            if next(self.parameters()).is_cuda:
                x, y = x.cuda(), y.cuda()
            
            self.base_model.zero_grad()
            
            # Forward pass
            logits, _ = self.base_model(x)
            
            # Use log-likelihood (negative cross-entropy)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Sample from model distribution for true Fisher
            # Or use labels for empirical Fisher
            loss = F.nll_loss(log_probs, y)
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.base_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
            
            n_used += x.shape[0]
        
        # Normalize by number of samples
        for name in fisher:
            fisher[name] /= max(n_used, 1)
        
        return fisher
    
    def consolidate_task(self, dataloader: DataLoader) -> None:
        """
        Consolidate task by computing and storing Fisher and optimal params.
        
        For online EWC, maintains a running average of Fisher matrices.
        
        Args:
            dataloader: DataLoader for the task just completed
        """
        # Compute Fisher for current task
        current_fisher = self.compute_fisher_information(dataloader)
        
        # Store optimal parameters
        self.optimal_params = {
            name: param.clone().detach()
            for name, param in self.base_model.named_parameters()
            if param.requires_grad
        }
        
        if self.online:
            # Online EWC: running average of Fisher
            for name in current_fisher:
                if name in self.running_fisher:
                    self.running_fisher[name] = (
                        self.gamma * self.running_fisher[name] +
                        current_fisher[name]
                    )
                else:
                    self.running_fisher[name] = current_fisher[name].clone()
            self.fisher = self.running_fisher
        else:
            # Standard EWC: store Fisher for each task
            self.fisher = current_fisher
        
        self.current_task += 1
    
    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty: Σ_i F_i * (θ_i - θ*_i)^2
        
        Returns:
            Scalar EWC penalty
        """
        if len(self.fisher) == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for name, param in self.base_model.named_parameters():
            if name in self.fisher and name in self.optimal_params:
                diff = param - self.optimal_params[name]
                penalty += (self.fisher[name] * diff ** 2).sum()
        
        return (self.lamb / 2) * penalty


# =============================================================================
# Synaptic Intelligence (SI)
# =============================================================================


class SynapticIntelligenceClause(ContinualLearningWrapper):
    """
    Synaptic Intelligence for Tsetlin Machines.
    
    Tracks parameter importance online during training by measuring
    how much each parameter contributes to loss reduction.
    
    Key formula:
    ω_i = Σ_t g_i(t) * Δθ_i(t) / (Δθ_i^2 + ξ)
    
    where g_i is gradient and Δθ_i is parameter change.
    
    Args:
        base_model: Base TM model
        lamb: SI regularization strength
        xi: Damping factor for importance
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        lamb: float = 1.0,
        xi: float = 1e-3,
    ):
        super().__init__(base_model, lamb)
        self.xi = xi
        
        # Importance weights (omega)
        self.omega: Dict[str, torch.Tensor] = {}
        
        # Running sum of gradient * delta
        self.omega_sum: Dict[str, torch.Tensor] = {}
        
        # Parameter values at task start
        self.init_params: Dict[str, torch.Tensor] = {}
        
        # Previous parameter values (for delta computation)
        self.prev_params: Dict[str, torch.Tensor] = {}
        
        # Running sum of delta squared
        self.delta_squared: Dict[str, torch.Tensor] = {}
        
        self._initialize_tracking()
    
    def _initialize_tracking(self) -> None:
        """Initialize tracking variables for SI."""
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                self.omega_sum[name] = torch.zeros_like(param)
                self.prev_params[name] = param.clone().detach()
                self.init_params[name] = param.clone().detach()
                self.delta_squared[name] = torch.zeros_like(param)
    
    def update_omega(self) -> None:
        """
        Update running importance after each optimizer step.
        
        Call this after optimizer.step() during training.
        """
        for name, param in self.base_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Delta = current - previous
                delta = param.detach() - self.prev_params[name]
                
                # Accumulate gradient * delta
                self.omega_sum[name] += (-param.grad.data * delta)
                
                # Update previous params
                self.prev_params[name] = param.clone().detach()
    
    def consolidate_task(self, dataloader: Optional[DataLoader] = None) -> None:
        """
        Consolidate task by computing final importance weights.
        
        Args:
            dataloader: Not used for SI (importance tracked online)
        """
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                # Total parameter change over task
                delta_task = param.detach() - self.init_params[name]
                
                # Importance = accumulated gradient*delta / (delta^2 + xi)
                importance = self.omega_sum[name] / (delta_task ** 2 + self.xi)
                
                # Accumulate importance across tasks
                if name in self.omega:
                    self.omega[name] += importance.clamp(min=0)
                else:
                    self.omega[name] = importance.clamp(min=0)
                
                # Reset for next task
                self.omega_sum[name] = torch.zeros_like(param)
                self.init_params[name] = param.clone().detach()
                self.prev_params[name] = param.clone().detach()
        
        self.current_task += 1
    
    def penalty(self) -> torch.Tensor:
        """
        Compute SI penalty: Σ_i ω_i * (θ_i - θ*_i)^2
        
        Returns:
            Scalar SI penalty
        """
        if len(self.omega) == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for name, param in self.base_model.named_parameters():
            if name in self.omega and name in self.init_params:
                diff = param - self.init_params[name]
                penalty += (self.omega[name] * diff ** 2).sum()
        
        return self.lamb * penalty


# =============================================================================
# Memory Aware Synapses (MAS)
# =============================================================================


class MemoryAwareSynapsesClause(ContinualLearningWrapper):
    """
    Memory Aware Synapses for Tsetlin Machines.
    
    Computes importance without labels by measuring sensitivity
    of network output to parameter perturbations.
    
    Key formula:
    Ω_i = (1/N) * Σ_n |∂F(x_n)/∂θ_i|
    
    where F(x) is the output function (no labels needed).
    
    Args:
        base_model: Base TM model
        lamb: MAS regularization strength
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        lamb: float = 1.0,
    ):
        super().__init__(base_model, lamb)
        
        # Importance weights
        self.omega: Dict[str, torch.Tensor] = {}
        
        # Reference parameters
        self.ref_params: Dict[str, torch.Tensor] = {}
    
    def compute_importance(
        self,
        dataloader: DataLoader,
        n_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute importance based on output sensitivity.
        
        Unlike EWC, this doesn't require labels - just input data.
        
        Args:
            dataloader: DataLoader for importance computation
            n_samples: Maximum samples to use
            
        Returns:
            Dict mapping parameter names to importance
        """
        importance = {
            name: torch.zeros_like(param)
            for name, param in self.base_model.named_parameters()
            if param.requires_grad
        }
        
        self.base_model.train()
        n_used = 0
        
        for batch in dataloader:
            if n_samples is not None and n_used >= n_samples:
                break
            
            x = batch[0]
            if next(self.parameters()).is_cuda:
                x = x.cuda()
            
            self.base_model.zero_grad()
            
            # Forward pass
            logits, _ = self.base_model(x)
            
            # Use L2 norm of output as objective
            # This measures output sensitivity without labels
            output_norm = logits.norm(p=2)
            output_norm.backward()
            
            # Accumulate absolute gradients
            for name, param in self.base_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    importance[name] += param.grad.data.abs()
            
            n_used += x.shape[0]
        
        # Normalize
        for name in importance:
            importance[name] /= max(n_used, 1)
        
        return importance
    
    def consolidate_task(self, dataloader: DataLoader) -> None:
        """
        Consolidate task by computing importance and storing reference params.
        
        Args:
            dataloader: DataLoader for importance computation
        """
        # Compute importance
        current_importance = self.compute_importance(dataloader)
        
        # Accumulate importance
        for name in current_importance:
            if name in self.omega:
                self.omega[name] += current_importance[name]
            else:
                self.omega[name] = current_importance[name].clone()
        
        # Store reference parameters
        self.ref_params = {
            name: param.clone().detach()
            for name, param in self.base_model.named_parameters()
            if param.requires_grad
        }
        
        self.current_task += 1
    
    def penalty(self) -> torch.Tensor:
        """
        Compute MAS penalty: Σ_i Ω_i * (θ_i - θ*_i)^2
        
        Returns:
            Scalar MAS penalty
        """
        if len(self.omega) == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for name, param in self.base_model.named_parameters():
            if name in self.omega and name in self.ref_params:
                diff = param - self.ref_params[name]
                penalty += (self.omega[name] * diff ** 2).sum()
        
        return self.lamb * penalty


# =============================================================================
# Experience Replay Buffer
# =============================================================================


class ExperienceReplayBuffer:
    """
    Experience replay buffer with reservoir sampling.
    
    Maintains a fixed-size buffer of samples from previous tasks
    using reservoir sampling for uniform distribution.
    
    Args:
        max_size: Maximum buffer size
        device: Device for stored tensors
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        device: str = "cpu",
        capacity: Optional[int] = None,
    ):
        # Accept legacy keyword `capacity` used by tests; falls back to max_size.
        self.max_size = capacity if capacity is not None else max_size
        self.device = device
        
        self.buffer_x: List[torch.Tensor] = []
        self.buffer_y: List[torch.Tensor] = []
        self.n_seen = 0

    def add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        """
        Add samples to buffer using reservoir sampling.
        
        Args:
            x: Input samples [batch, features]
            y: Labels [batch]
        """
        # Normalize shapes for single sample inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)

        batch_size = x.shape[0]
        
        for i in range(batch_size):
            self.n_seen += 1
            
            if len(self.buffer_x) < self.max_size:
                # Buffer not full, just add
                self.buffer_x.append(x[i].detach().cpu())
                self.buffer_y.append(y[i].detach().cpu())
            else:
                # Reservoir sampling: replace with probability max_size/n_seen
                replace_idx = torch.randint(0, self.n_seen, (1,)).item()
                if replace_idx < self.max_size:
                    self.buffer_x[replace_idx] = x[i].detach().cpu()
                    self.buffer_y[replace_idx] = y[i].detach().cpu()
    
    def sample(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch from the buffer.
        
        Args:
            batch_size: Number of samples to return
            
        Returns:
            (x, y) tuple of sampled data
        """
        if len(self.buffer_x) == 0:
            return None, None
        
        indices = torch.randint(0, len(self.buffer_x), (batch_size,))
        
        x = torch.stack([self.buffer_x[i] for i in indices]).to(self.device)
        y = torch.stack([self.buffer_y[i] for i in indices]).to(self.device)
        
        return x, y
    
    def __len__(self) -> int:
        return len(self.buffer_x)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer_x = []
        self.buffer_y = []
        self.n_seen = 0


class EWCWrapper(EWCClauseMachine):
    """
    Thin wrapper to match legacy keyword `lambda_` used in tests.
    """

    def __init__(self, base_model: nn.Module, lambda_: float = 1000.0, **kwargs):
        super().__init__(base_model, lamb=lambda_, **kwargs)
        # Legacy compatibility: expose `model` attribute used in tests.
        self.model = self.base_model

    def compute_fisher(self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 64):
        """Legacy helper expected by tests; wraps compute_fisher_information."""
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)))
        fisher = self.compute_fisher_information(loader)
        # Mirror consolidate_task side effects expected by tests
        self.fisher = fisher
        self.optimal_params = {
            name: param.clone().detach()
            for name, param in self.base_model.named_parameters()
            if param.requires_grad
        }
        return fisher

    def consolidate(self, dataloader: Optional[DataLoader] = None):
        """Legacy alias; if no dataloader is provided, just bump task counter."""
        if dataloader is not None:
            return self.consolidate_task(dataloader)
        # Already populated fisher/optimal_params via compute_fisher
        self.current_task += 1
        return None

    def ewc_penalty(self) -> torch.Tensor:
        """Legacy alias for penalty()."""
        return self.penalty()


class ReplayAugmentedTrainer:
    """
    Training helper that augments batches with replay samples.
    
    Args:
        model: Model to train
        buffer: Experience replay buffer
        replay_batch_size: Number of replay samples per batch
        replay_weight: Weight for replay samples in loss
    """
    
    def __init__(
        self,
        model: nn.Module,
        buffer: ExperienceReplayBuffer,
        replay_batch_size: int = 16,
        replay_weight: float = 1.0,
    ):
        self.model = model
        self.buffer = buffer
        self.replay_batch_size = replay_batch_size
        self.replay_weight = replay_weight
    
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: Callable = F.cross_entropy,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single training step with replay augmentation.
        
        Args:
            x: Current task input
            y: Current task labels
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            (total_loss, task_loss)
        """
        self.model.train()
        optimizer.zero_grad()
        
        # Forward on current task
        logits, _ = self.model(x)
        task_loss = criterion(logits, y)
        
        # Forward on replay if buffer not empty
        replay_loss = torch.tensor(0.0, device=x.device)
        if len(self.buffer) > 0:
            replay_x, replay_y = self.buffer.sample(self.replay_batch_size)
            if replay_x is not None:
                replay_logits, _ = self.model(replay_x)
                replay_loss = criterion(replay_logits, replay_y)
        
        # Combined loss
        total_loss = task_loss + self.replay_weight * replay_loss
        
        total_loss.backward()
        optimizer.step()
        
        # Add current samples to buffer
        self.buffer.add(x, y)
        
        return total_loss, task_loss


# =============================================================================
# Gradient Episodic Memory (GEM)
# =============================================================================


class GradientEpisodicMemory(ContinualLearningWrapper):
    """
    Gradient Episodic Memory for Tsetlin Machines.
    
    Stores representative samples for each task and projects
    gradients to not increase loss on previous tasks.
    
    Key idea: Project gradient g to g' such that:
    g' · g_t ≥ 0 for all previous task gradients g_t
    
    Args:
        base_model: Base TM model
        memory_per_task: Samples to store per task
        margin: Margin for constraint satisfaction
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        memory_per_task: int = 256,
        margin: float = 0.0,
    ):
        super().__init__(base_model, lamb=0.0)  # No penalty for GEM
        self.memory_per_task = memory_per_task
        self.margin = margin
        
        # Memory for each task
        self.task_memory: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        
        # Reference gradients for each task
        self.ref_gradients: Dict[int, Dict[str, torch.Tensor]] = {}
    
    def store_task_memory(
        self,
        task_id: int,
        dataloader: DataLoader,
    ) -> None:
        """
        Store representative samples for a task.
        
        Args:
            task_id: ID of the task
            dataloader: DataLoader for the task
        """
        all_x, all_y = [], []
        
        for batch in dataloader:
            all_x.append(batch[0])
            all_y.append(batch[1])
            if sum(x.shape[0] for x in all_x) >= self.memory_per_task:
                break
        
        x = torch.cat(all_x, dim=0)[:self.memory_per_task]
        y = torch.cat(all_y, dim=0)[:self.memory_per_task]
        
        self.task_memory[task_id] = (x.detach().cpu(), y.detach().cpu())
    
    def _compute_gradient_on_task(
        self,
        task_id: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute gradient on stored task samples."""
        if task_id not in self.task_memory:
            return {}
        
        x, y = self.task_memory[task_id]
        device = next(self.parameters()).device
        x, y = x.to(device), y.to(device)
        
        self.base_model.zero_grad()
        logits, _ = self.base_model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        
        gradients = {}
        for name, param in self.base_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradients[name] = param.grad.clone()
        
        return gradients
    
    def project_gradient(
        self,
        current_grad: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Project gradient to satisfy GEM constraints.
        
        Uses quadratic programming to find the closest gradient
        that doesn't increase loss on previous tasks.
        
        Args:
            current_grad: Current task gradient
            
        Returns:
            Projected gradient
        """
        if self.current_task == 0:
            return current_grad
        
        # Compute reference gradients for all previous tasks
        ref_grads = []
        for task_id in range(self.current_task):
            task_grad = self._compute_gradient_on_task(task_id)
            if task_grad:
                ref_grads.append(task_grad)
        
        if not ref_grads:
            return current_grad
        
        # Check constraints
        violated = []
        for i, ref in enumerate(ref_grads):
            dot_product = sum(
                (current_grad[name] * ref[name]).sum()
                for name in current_grad if name in ref
            )
            if dot_product < self.margin:
                violated.append(i)
        
        if not violated:
            return current_grad
        
        # Simple projection: for each violated constraint, project out
        # the violating component
        projected = {k: v.clone() for k, v in current_grad.items()}
        
        for i in violated:
            ref = ref_grads[i]
            
            # Compute dot product
            dot = sum(
                (projected[name] * ref[name]).sum()
                for name in projected if name in ref
            )
            
            # Compute ref norm squared
            ref_norm_sq = sum(
                (ref[name] ** 2).sum()
                for name in ref
            ) + 1e-8
            
            # Project
            for name in projected:
                if name in ref:
                    projected[name] -= (dot / ref_norm_sq) * ref[name]
        
        return projected
    
    def train_step_with_gem(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        """
        Training step with GEM gradient projection.
        
        Args:
            x: Input batch
            y: Labels
            optimizer: Optimizer
            
        Returns:
            Loss value
        """
        self.base_model.train()
        optimizer.zero_grad()
        
        # Forward and backward on current task
        logits, _ = self.base_model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        
        # Get current gradients
        current_grad = {}
        for name, param in self.base_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                current_grad[name] = param.grad.clone()
        
        # Project gradient
        projected = self.project_gradient(current_grad)
        
        # Apply projected gradient
        for name, param in self.base_model.named_parameters():
            if name in projected:
                param.grad = projected[name]
        
        optimizer.step()
        
        return loss
    
    def consolidate_task(self, dataloader: DataLoader) -> None:
        """Store memory for completed task."""
        self.store_task_memory(self.current_task, dataloader)
        self.current_task += 1


# =============================================================================
# PackNet
# =============================================================================


class PackNetClause(ContinualLearningWrapper):
    """
    PackNet for Tsetlin Machines.
    
    After each task, prunes a fraction of parameters and freezes
    the important ones. New tasks use the remaining capacity.
    
    Args:
        base_model: Base TM model
        prune_fraction: Fraction of free parameters to prune per task
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        prune_fraction: float = 0.5,
    ):
        super().__init__(base_model, lamb=0.0)
        self.prune_fraction = prune_fraction
        
        # Binary masks indicating which params are still available
        self.available_masks: Dict[str, torch.Tensor] = {}
        
        # Masks for each task
        self.task_masks: Dict[int, Dict[str, torch.Tensor]] = {}
        
        self._initialize_masks()
    
    def _initialize_masks(self) -> None:
        """Initialize all parameters as available."""
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                self.available_masks[name] = torch.ones_like(param)
    
    def _get_importance(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, torch.Tensor]:
        """Compute parameter importance for pruning."""
        importance = {
            name: torch.zeros_like(param)
            for name, param in self.base_model.named_parameters()
            if param.requires_grad
        }
        
        self.base_model.train()
        
        for batch in dataloader:
            x, y = batch[0], batch[1]
            if next(self.parameters()).is_cuda:
                x, y = x.cuda(), y.cuda()
            
            self.base_model.zero_grad()
            logits, _ = self.base_model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            
            for name, param in self.base_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    importance[name] += (param * param.grad).abs()
        
        return importance
    
    def prune_and_freeze(
        self,
        dataloader: DataLoader,
    ) -> None:
        """
        Prune unimportant parameters and freeze important ones.
        
        Args:
            dataloader: DataLoader for importance computation
        """
        # Compute importance
        importance = self._get_importance(dataloader)
        
        # For each parameter, find threshold to keep (1-prune_fraction)
        task_mask = {}
        
        for name in importance:
            if name not in self.available_masks:
                continue
            
            # Only consider available parameters
            avail = self.available_masks[name]
            imp = importance[name] * avail
            
            # Flatten and find threshold
            flat_imp = imp[avail.bool()].flatten()
            
            if flat_imp.numel() == 0:
                task_mask[name] = torch.zeros_like(avail)
                continue
            
            # Keep top (1 - prune_fraction)
            n_keep = int(flat_imp.numel() * (1 - self.prune_fraction))
            if n_keep == 0:
                n_keep = 1
            
            threshold = torch.topk(flat_imp, n_keep).values[-1]
            
            # Create mask: important params for this task
            mask = (imp >= threshold).float() * avail
            task_mask[name] = mask
            
            # Update available mask: remove used params
            self.available_masks[name] = avail * (1 - mask)
        
        self.task_masks[self.current_task] = task_mask
    
    def masked_forward(
        self,
        x: torch.Tensor,
        task_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with task-specific masks.
        
        Args:
            x: Input tensor
            task_id: Task to use masks for (None = use all)
            
        Returns:
            (logits, clause_outputs)
        """
        # Apply masks to parameters
        with torch.no_grad():
            original_params = {}
            for name, param in self.base_model.named_parameters():
                original_params[name] = param.data.clone()
                
                if task_id is not None and task_id in self.task_masks:
                    if name in self.task_masks[task_id]:
                        param.data *= self.task_masks[task_id][name]
                else:
                    # Use union of all task masks
                    mask = torch.zeros_like(param)
                    for tid in self.task_masks:
                        if name in self.task_masks[tid]:
                            mask = torch.max(mask, self.task_masks[tid][name])
                    param.data *= mask
        
        # Forward
        result = self.base_model(x)
        
        # Restore parameters
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                param.data = original_params[name]
        
        return result
    
    def masked_train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        """
        Training step that only updates available parameters.
        
        Args:
            x: Input batch
            y: Labels
            optimizer: Optimizer
            
        Returns:
            Loss value
        """
        self.base_model.train()
        optimizer.zero_grad()
        
        logits, _ = self.base_model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        
        # Zero out gradients for frozen parameters
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if param.grad is not None and name in self.available_masks:
                    param.grad *= self.available_masks[name]
        
        optimizer.step()
        
        return loss
    
    def consolidate_task(self, dataloader: DataLoader) -> None:
        """Prune and freeze after task completion."""
        self.prune_and_freeze(dataloader)
        self.current_task += 1


# =============================================================================
# Progressive Networks
# =============================================================================


class ProgressiveClauseNetwork(nn.Module):
    """
    Progressive Network for Tsetlin Machines.
    
    Adds a new "column" (TM module) for each new task, with lateral
    connections to previous columns. Previous columns are frozen.
    
    Args:
        base_tm_fn: Function to create a new TM column
        lateral_dim: Dimension for lateral connections
    """
    
    def __init__(
        self,
        base_tm_fn: Callable[[], nn.Module],
        lateral_dim: Optional[int] = None,
    ):
        super().__init__()
        self.base_tm_fn = base_tm_fn
        self.lateral_dim = lateral_dim
        
        # Columns (one per task)
        self.columns = nn.ModuleList()
        
        # Lateral adapters from previous columns
        self.lateral_adapters = nn.ModuleList()
        
        self.current_task = 0
    
    def add_task(self) -> None:
        """Add a new column for a new task."""
        # Create new column
        new_column = self.base_tm_fn()
        self.columns.append(new_column)
        
        # Create lateral connections from all previous columns
        if len(self.columns) > 1:
            lateral_list = nn.ModuleList()
            for i in range(len(self.columns) - 1):
                prev_col = self.columns[i]
                if hasattr(prev_col, 'n_clauses'):
                    in_dim = prev_col.n_clauses
                else:
                    in_dim = self.lateral_dim or 64
                
                if hasattr(new_column, 'n_features'):
                    out_dim = new_column.n_features
                else:
                    out_dim = in_dim
                
                adapter = nn.Linear(in_dim, out_dim)
                lateral_list.append(adapter)
            
            self.lateral_adapters.append(lateral_list)
        
        # Freeze previous columns
        for i in range(len(self.columns) - 1):
            for param in self.columns[i].parameters():
                param.requires_grad = False
        
        self.current_task = len(self.columns) - 1
    
    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[int] = None,
        use_ste: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a specific task.
        
        Args:
            x: Input tensor
            task_id: Task to run (None = current task)
            use_ste: Use STE for TM
            
        Returns:
            (logits, clause_outputs)
        """
        if task_id is None:
            task_id = self.current_task
        
        if task_id >= len(self.columns):
            raise ValueError(f"Task {task_id} not yet created")
        
        # Run all columns up to task_id
        all_clauses = []
        
        for col_idx in range(task_id + 1):
            col_input = x
            
            # Add lateral connections from previous columns
            if col_idx > 0:
                lateral_contrib = torch.zeros_like(x)
                for prev_idx in range(col_idx):
                    prev_clauses = all_clauses[prev_idx]
                    adapter = self.lateral_adapters[col_idx - 1][prev_idx]
                    lateral_contrib = lateral_contrib + adapter(prev_clauses)
                
                col_input = col_input + lateral_contrib
            
            # Run column
            logits, clauses = self.columns[col_idx](col_input, use_ste=use_ste)
            all_clauses.append(clauses)
        
        # Use output from target task's column
        return logits, all_clauses[task_id]
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get parameters trainable for current task."""
        params = list(self.columns[self.current_task].parameters())
        
        if self.current_task > 0:
            params.extend(self.lateral_adapters[self.current_task - 1].parameters())
        
        return params


# =============================================================================
# Combined Continual Learning Pipeline
# =============================================================================


class ContinualLearningPipeline:
    """
    High-level pipeline for continual learning experiments.
    
    Combines a base model with a continual learning method and
    provides convenient training/evaluation interfaces.
    
    Args:
        base_model: Base TM model
        method: CL method ('ewc', 'si', 'mas', 'gem', 'packnet', 'replay')
        **method_kwargs: Arguments for the CL method
    """
    
    METHODS = {
        "ewc": EWCClauseMachine,
        "si": SynapticIntelligenceClause,
        "mas": MemoryAwareSynapsesClause,
        "gem": GradientEpisodicMemory,
        "packnet": PackNetClause,
    }
    
    def __init__(
        self,
        base_model: nn.Module,
        method: str = "ewc",
        use_replay: bool = False,
        replay_buffer_size: int = 1000,
        **method_kwargs,
    ):
        self.method_name = method
        self.use_replay = use_replay
        
        # Create CL-wrapped model
        if method in self.METHODS:
            self.model = self.METHODS[method](base_model, **method_kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Optional replay buffer
        if use_replay:
            device = "cuda" if next(base_model.parameters()).is_cuda else "cpu"
            self.replay_buffer = ExperienceReplayBuffer(
                max_size=replay_buffer_size,
                device=device,
            )
        else:
            self.replay_buffer = None
        
        self.task_accuracies: Dict[int, List[float]] = defaultdict(list)
    
    def train_task(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        lr: float = 0.001,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Train on a single task.
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            epochs: Number of epochs
            lr: Learning rate
            verbose: Print progress
            
        Returns:
            Dict with training metrics
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        metrics = {"train_loss": [], "val_acc": []}
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            
            for batch in train_loader:
                x, y = batch[0], batch[1]
                if next(self.model.parameters()).is_cuda:
                    x, y = x.cuda(), y.cuda()
                
                optimizer.zero_grad()
                
                # Forward
                logits, _ = self.model(x)
                task_loss = F.cross_entropy(logits, y)
                
                # CL penalty
                cl_penalty = self.model.penalty()
                
                # Replay loss
                replay_loss = torch.tensor(0.0, device=x.device)
                if self.replay_buffer and len(self.replay_buffer) > 0:
                    replay_x, replay_y = self.replay_buffer.sample(32)
                    if replay_x is not None:
                        replay_logits, _ = self.model(replay_x)
                        replay_loss = F.cross_entropy(replay_logits, replay_y)
                
                # Total loss
                loss = task_loss + cl_penalty + replay_loss
                loss.backward()
                
                # SI update
                if isinstance(self.model, SynapticIntelligenceClause):
                    self.model.update_omega()
                
                optimizer.step()
                
                # Add to replay buffer
                if self.replay_buffer:
                    self.replay_buffer.add(x, y)
                
                epoch_loss += loss.item()
                n_batches += 1
            
            metrics["train_loss"].append(epoch_loss / n_batches)
            
            # Validation
            if val_loader:
                acc = self.evaluate(val_loader)
                metrics["val_acc"].append(acc)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/n_batches:.4f} - Val Acc: {acc:.4f}")
        
        return metrics
    
    def finish_task(self, dataloader: DataLoader) -> None:
        """
        Called after finishing a task for consolidation.
        
        Args:
            dataloader: DataLoader for the completed task
        """
        self.model.consolidate_task(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """
        Evaluate model accuracy.
        
        Args:
            dataloader: Test data
            
        Returns:
            Accuracy
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch[0], batch[1]
                if next(self.model.parameters()).is_cuda:
                    x, y = x.cuda(), y.cuda()
                
                logits, _ = self.model(x)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.shape[0]
        
        return correct / total
    
    def evaluate_all_tasks(
        self,
        task_loaders: List[DataLoader],
    ) -> Dict[str, float]:
        """
        Evaluate on all seen tasks.
        
        Args:
            task_loaders: List of test loaders, one per task
            
        Returns:
            Dict with per-task and average accuracy
        """
        results = {}
        
        for task_id, loader in enumerate(task_loaders):
            acc = self.evaluate(loader)
            results[f"task_{task_id}"] = acc
            self.task_accuracies[task_id].append(acc)
        
        results["average"] = sum(results.values()) / len(results)
        
        return results




