"""
Sharpness-Aware Minimization (SAM) Optimizer.

SAM seeks parameters that lie in neighborhoods with uniformly low loss,
which correlates with better generalization.

Key Innovation:
Standard optimizers find low loss points that may be in sharp valleys.
SAM finds points where the loss is low even when parameters are perturbed,
leading to flatter minima and better generalization.

Algorithm:
1. Compute gradient at current position
2. Take a step in the gradient direction (perturbation)
3. Compute gradient at perturbed position
4. Update using perturbed gradient

Benefits:
- Better generalization
- More robust to noise
- Works with any base optimizer
- Minimal computational overhead (2x gradient)

References:
- Foret et al. (2020): Sharpness-Aware Minimization for Efficiently Improving Generalization
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple

import torch
from torch.optim import Optimizer


class SAM(Optimizer):
    """
    Sharpness-Aware Minimization Optimizer.
    
    Wraps a base optimizer and modifies the update to seek flat minima.
    
    Args:
        params: Model parameters to optimize
        base_optimizer: Base optimizer class (e.g., torch.optim.SGD)
        rho: Size of the perturbation neighborhood
        adaptive: Use adaptive rho based on parameter scale
        **kwargs: Arguments passed to base optimizer
    
    Example:
        >>> model = MyModel()
        >>> optimizer = SAM(model.parameters(), torch.optim.SGD, lr=0.1, rho=0.05)
        >>> for x, y in dataloader:
        ...     loss = criterion(model(x), y)
        ...     loss.backward()
        ...     optimizer.first_step(zero_grad=True)
        ...     criterion(model(x), y).backward()
        ...     optimizer.second_step(zero_grad=True)
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base_optimizer: type,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ):
        if rho < 0.0:
            raise ValueError(f"Invalid rho: {rho}. Should be >= 0")
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        
        # Create base optimizer with same param groups
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        
        # Store original parameters
        self.state["step"] = 0
    
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """
        Compute and apply perturbation (ε = ρ * ∇L / ||∇L||).
        
        This moves parameters to a perturbed position where we'll
        compute the second gradient.
        
        Args:
            zero_grad: Whether to zero gradients after step
        """
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Store original params
                self.state[p]["old_p"] = p.data.clone()
                
                # Compute perturbation
                if group["adaptive"]:
                    # Scale by parameter magnitude
                    e_w = (torch.abs(p) * p.grad) * scale
                else:
                    e_w = p.grad * scale
                
                # Apply perturbation
                p.add_(e_w)
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """
        Restore original parameters and apply the actual update.
        
        The gradient at the perturbed position is used to update
        the original parameters.
        
        Args:
            zero_grad: Whether to zero gradients after step
        """
        # Restore original parameters
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        
        # Apply base optimizer update
        self.base_optimizer.step()
        self.state["step"] += 1
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> torch.Tensor:
        """
        Full SAM step (convenience method).
        
        For manual control, use first_step() and second_step().
        
        Args:
            closure: Closure that reevaluates the model and returns the loss
            
        Returns:
            Loss value
        """
        if closure is None:
            raise ValueError("SAM requires closure for full step")
        
        # First forward-backward
        closure = torch.enable_grad()(closure)
        loss = closure()
        
        # First step (perturbation)
        self.first_step(zero_grad=True)
        
        # Second forward-backward at perturbed position
        closure()
        
        # Second step (actual update)
        self.second_step(zero_grad=True)
        
        return loss
    
    def _grad_norm(self) -> torch.Tensor:
        """Compute total gradient norm across all parameters."""
        shared_device = self.param_groups[0]["params"][0].device
        
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) + 1e-12) * p.grad).norm(p=2).to(shared_device)
                if group["adaptive"]
                else p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        
        return norm
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state."""
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class GSAM(SAM):
    """
    Gradient SAM - Variant that uses gradient alignment.
    
    Instead of just using the gradient at the perturbed point,
    GSAM uses a gradient that is more aligned with the original gradient.
    
    This can lead to more stable training.
    
    Args:
        params: Model parameters
        base_optimizer: Base optimizer class
        rho: Perturbation radius
        alpha: Gradient alignment coefficient
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base_optimizer: type,
        rho: float = 0.05,
        alpha: float = 0.5,
        **kwargs,
    ):
        super().__init__(params, base_optimizer, rho=rho, **kwargs)
        self.alpha = alpha
    
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Store original gradient in addition to SAM perturbation."""
        # Store original gradients
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    self.state[p]["orig_grad"] = p.grad.clone()
        
        # Apply SAM perturbation
        super().first_step(zero_grad=zero_grad)
    
    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Use gradient alignment before base optimizer step."""
        # Restore original parameters
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                p.data = self.state[p]["old_p"]
                
                # Align gradient with original
                orig_grad = self.state[p]["orig_grad"]
                perturbed_grad = p.grad
                
                # Projection of perturbed gradient onto original gradient
                dot = (orig_grad * perturbed_grad).sum()
                orig_norm_sq = (orig_grad ** 2).sum() + 1e-12
                projection = dot / orig_norm_sq
                
                # Aligned gradient
                aligned = projection * orig_grad
                
                # Combine
                p.grad = self.alpha * orig_grad + (1 - self.alpha) * aligned
        
        # Apply base optimizer update
        self.base_optimizer.step()
        self.state["step"] += 1
        
        if zero_grad:
            self.zero_grad()


class LookSAM(SAM):
    """
    LookSAM - SAM with periodic sharpness computation.
    
    Instead of computing sharpness at every step, LookSAM computes
    it periodically, reducing computational overhead.
    
    Args:
        params: Model parameters
        base_optimizer: Base optimizer class
        rho: Perturbation radius
        k: Compute sharpness every k steps
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base_optimizer: type,
        rho: float = 0.05,
        k: int = 5,
        **kwargs,
    ):
        super().__init__(params, base_optimizer, rho=rho, **kwargs)
        self.k = k
        self.sam_direction: dict = {}
    
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Use cached direction or compute new one."""
        step = self.state.get("step", 0)
        
        if step % self.k == 0:
            # Compute and store new SAM direction
            grad_norm = self._grad_norm()
            
            for group in self.param_groups:
                scale = group["rho"] / (grad_norm + 1e-12)
                
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    
                    # Store direction
                    direction = p.grad * scale
                    self.sam_direction[id(p)] = direction
                    
                    # Store original and perturb
                    self.state[p]["old_p"] = p.data.clone()
                    p.add_(direction)
        else:
            # Use cached direction
            for group in self.param_groups:
                for p in group["params"]:
                    if id(p) not in self.sam_direction:
                        continue
                    
                    self.state[p]["old_p"] = p.data.clone()
                    p.add_(self.sam_direction[id(p)])
        
        if zero_grad:
            self.zero_grad()


def create_sam_optimizer(
    model: torch.nn.Module,
    base_optimizer: str = "adam",
    lr: float = 0.001,
    rho: float = 0.05,
    adaptive: bool = False,
    **kwargs,
) -> SAM:
    """
    Convenience function to create SAM optimizer.
    
    Args:
        model: Model to optimize
        base_optimizer: Name of base optimizer ('adam', 'sgd', 'adamw')
        lr: Learning rate
        rho: SAM perturbation radius
        adaptive: Use adaptive SAM
        **kwargs: Additional optimizer arguments
        
    Returns:
        SAM optimizer
    """
    optimizers = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }
    
    if base_optimizer.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {base_optimizer}")
    
    base_class = optimizers[base_optimizer.lower()]
    
    return SAM(
        model.parameters(),
        base_class,
        lr=lr,
        rho=rho,
        adaptive=adaptive,
        **kwargs,
    )

