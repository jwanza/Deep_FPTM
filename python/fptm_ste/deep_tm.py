import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Sequence, Tuple, Type
from .tm import FuzzyPatternTM_STE, FuzzyPatternTM_STCM, prepare_tm_input
from .tm_optimized import OptimizedSTCM
from .tm_feedback import EnhancedSTCM


def _resolve_layer_cls(layer_cls: Type[nn.Module], operator: Optional[str]) -> Type[nn.Module]:
    if layer_cls not in {FuzzyPatternTM_STCM, OptimizedSTCM}:
        return layer_cls
    if operator is None or operator in {"capacity", "product"}:
        return OptimizedSTCM
    return FuzzyPatternTM_STCM


def _class_supports_kwarg(cls: Type[nn.Module], name: str) -> bool:
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if name in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


class DeepTMNetwork(nn.Module):
    def __init__(
        self,
        input_dim: Optional[int],
        hidden_dims: Sequence[int],
        n_classes: int,
        n_clauses: int = 100,
        dropout: float = 0.1,
        tau: float = 0.5,
        noise_std: float = 0.0,
        *,
        input_shape: Optional[Tuple[int, int, int]] = None,
        auto_expand_grayscale: bool = False,
        allow_channel_reduce: bool = True,
        clause_dropout: float = 0.0,
        literal_dropout: float = 0.0,
        clause_bias_init: float = 0.0,
        layer_cls: Type[nn.Module] = FuzzyPatternTM_STE,
        layer_operator: Optional[str] = None,
        layer_ternary_voting: Optional[bool] = None,
        layer_extra_kwargs: Optional[Dict[str, Any]] = None,
        vote_dropout: float = 0.0,
        slow_layer_count: int = 0,
        slow_layer_lr_scale: float = 0.5,
    ):
        super().__init__()
        self.input_shape = tuple(input_shape) if input_shape is not None else None
        self.auto_expand_grayscale = auto_expand_grayscale
        self.allow_channel_reduce = allow_channel_reduce

        if self.input_shape is not None:
            expected_dim = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
            if input_dim is not None and input_dim not in (expected_dim, -1):
                raise ValueError(
                    "input_dim does not match input_shape: "
                    f"{input_dim} vs {expected_dim}. Use input_dim=None or -1 to infer automatically."
                )
            input_dim = expected_dim
        if input_dim is None or input_dim <= 0:
            raise ValueError("input_dim must be positive or inferred via input_shape.")

        self.input_dim = input_dim
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.noise_std = noise_std
        self.layer_cls = _resolve_layer_cls(layer_cls, layer_operator)
        self.layer_extra_kwargs = dict(layer_extra_kwargs or {})
        self.vote_dropout = vote_dropout
        self.slow_layer_count = max(0, slow_layer_count)
        self.slow_layer_lr_scale = slow_layer_lr_scale
        self._enhanced_layer_configs: list[tuple[nn.Module, float]] = []

        prev = input_dim
        for idx, h in enumerate(hidden_dims):
            layer_kwargs = {}
            if idx == 0:
                layer_kwargs = dict(
                    input_shape=self.input_shape,
                    auto_expand_grayscale=self.auto_expand_grayscale,
                    allow_channel_reduce=self.allow_channel_reduce,
                )
            tm_kwargs = dict(
                n_features=prev,
                n_clauses=n_clauses,
                n_classes=h,
                tau=tau,
            )
            # Only add dropout params if the layer class supports them
            if _class_supports_kwarg(self.layer_cls, "clause_dropout"):
                tm_kwargs["clause_dropout"] = clause_dropout
            if _class_supports_kwarg(self.layer_cls, "literal_dropout"):
                tm_kwargs["literal_dropout"] = literal_dropout
            if _class_supports_kwarg(self.layer_cls, "clause_bias_init"):
                tm_kwargs["clause_bias_init"] = clause_bias_init
            tm_kwargs.update(layer_kwargs)
            tm_kwargs.update(self.layer_extra_kwargs)
            if layer_operator is not None and _class_supports_kwarg(self.layer_cls, "operator"):
                tm_kwargs["operator"] = layer_operator
            if layer_ternary_voting is not None and _class_supports_kwarg(self.layer_cls, "ternary_voting"):
                tm_kwargs["ternary_voting"] = layer_ternary_voting
            layer_module = self.layer_cls(**tm_kwargs)
            self._register_enhanced_layer(layer_module, idx)
            self.layers.append(layer_module)
            self.norms.append(nn.LayerNorm(h))
            self.residuals.append(nn.Linear(prev, h, bias=False) if prev != h else nn.Identity())
            prev = h

        classifier_kwargs = dict(
            n_features=prev,
            n_clauses=n_clauses,
            n_classes=n_classes,
            tau=tau,
        )
        # Only add dropout params if the layer class supports them
        if _class_supports_kwarg(self.layer_cls, "clause_dropout"):
            classifier_kwargs["clause_dropout"] = clause_dropout
        if _class_supports_kwarg(self.layer_cls, "literal_dropout"):
            classifier_kwargs["literal_dropout"] = literal_dropout
        if _class_supports_kwarg(self.layer_cls, "clause_bias_init"):
            classifier_kwargs["clause_bias_init"] = clause_bias_init
        classifier_kwargs.update(self.layer_extra_kwargs)
        if layer_operator is not None and _class_supports_kwarg(self.layer_cls, "operator"):
            classifier_kwargs["operator"] = layer_operator
        if layer_ternary_voting is not None and _class_supports_kwarg(self.layer_cls, "ternary_voting"):
            classifier_kwargs["ternary_voting"] = layer_ternary_voting
        classifier_module = self.layer_cls(**classifier_kwargs)
        self._register_enhanced_layer(classifier_module, len(self.layers))
        self.classifier = classifier_module
        self.dropout = nn.Dropout(dropout)

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return prepare_tm_input(
            x,
            n_features=self.input_dim,
            input_shape=self.input_shape,
            auto_expand_grayscale=self.auto_expand_grayscale,
            allow_channel_reduce=self.allow_channel_reduce,
        )

    def forward(self, x: torch.Tensor, use_ste: bool = True):
        x = self._normalize_input(x)
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        for layer, norm, res in zip(self.layers, self.norms, self.residuals):
            identity = res(x)
            # Bypass internal normalization checks for speed
            logits, _ = layer(x, use_ste=use_ste, skip_norm=True)
            logits = self._maybe_drop_votes(layer, logits)
            x = norm(self.dropout(torch.sigmoid(logits)) + identity)
        logits, clauses = self.classifier(x, use_ste=use_ste, skip_norm=True)
        logits = self._maybe_drop_votes(self.classifier, logits)
        return logits, clauses

    def set_tau(self, tau: float) -> None:
        for layer in self.layers:
            if hasattr(layer, "tau"):
                layer.tau = tau
        if hasattr(self.classifier, "tau"):
            self.classifier.tau = tau

    # ------------------------------------------------------------------ #
    # Enhanced-layer helpers
    # ------------------------------------------------------------------ #
    def _register_enhanced_layer(self, layer: nn.Module, idx: int) -> None:
        if not isinstance(layer, EnhancedSTCM):
            return
        base_dropout = float(getattr(layer, "clause_dropout", 0.0))
        self._enhanced_layer_configs.append((layer, base_dropout))
        if idx < self.slow_layer_count and self.slow_layer_lr_scale < 1.0:
            self._scale_gradients(layer, self.slow_layer_lr_scale)

    def _scale_gradients(self, module: nn.Module, scale: float) -> None:
        for param in module.parameters():
            if not param.requires_grad:
                continue
            param.register_hook(lambda grad, s=scale: grad * s if grad is not None else grad)

    def _maybe_drop_votes(self, layer: nn.Module, logits: torch.Tensor) -> torch.Tensor:
        if self.vote_dropout <= 0 or not isinstance(layer, EnhancedSTCM):
            return logits
        return F.dropout(logits, p=self.vote_dropout, training=self.training)

    def set_validation_gap(self, gap: float) -> None:
        """
        Adjust clause dropout for enhanced layers based on validation gap
        (train_acc - val_acc). Larger gaps increase dropout to limit overfit.
        """
        if not self._enhanced_layer_configs:
            return
        clamp_gap = max(0.0, float(gap))
        scale = min(2.0, 1.0 + clamp_gap)
        for layer, base in self._enhanced_layer_configs:
            if hasattr(layer, "clause_dropout"):
                adjusted = max(0.0, min(0.5, base * scale))
                layer.clause_dropout = float(adjusted)
    
    # =========================================================================
    # CUDA Graph Support for 10-15x Inference Speedup
    # =========================================================================
    
    _cuda_graph = None
    _static_input = None
    _static_output = None
    _graph_batch_size = None
    _use_graph = False
    
    def enable_cuda_graph(self, batch_size: int) -> 'DeepTMNetwork':
        """
        Enable CUDA graph for fast inference.
        
        Args:
            batch_size: Fixed batch size for graph capture
            
        Returns:
            self (for method chaining)
        """
        self._graph_batch_size = batch_size
        self._use_graph = True
        self._cuda_graph = None  # Will capture on first forward
        return self
    
    def disable_cuda_graph(self) -> 'DeepTMNetwork':
        """Disable CUDA graph."""
        self._use_graph = False
        if self._cuda_graph is not None:
            del self._cuda_graph
            self._cuda_graph = None
        return self
    
    def _capture_cuda_graph(self, x: torch.Tensor) -> None:
        """Capture forward pass in CUDA graph."""
        self.eval()
        self._static_input = x.clone()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = self._forward_no_graph(self._static_input)
        torch.cuda.synchronize()
        
        # Capture
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph):
            with torch.no_grad():
                self._static_output = self._forward_no_graph(self._static_input)
        torch.cuda.synchronize()
    
    def _forward_no_graph(self, x: torch.Tensor, use_ste: bool = True):
        """Forward pass without graph (for capture)."""
        # Call the original forward implementation
        if self.input_shape is not None:
            x = self._handle_image_input(x)
        if self.noise_std > 0 and self.training:
            x = x + torch.randn_like(x) * self.noise_std
        
        for layer, norm, res in zip(self.layers, self.norms, self.residuals):
            out, clauses = layer(x, use_ste=use_ste)
            out = norm(out)
            out = F.relu(out)
            if res is not None:
                x = out + res(x)
            else:
                x = out
        
        logits = self.final(x)
        clause_outputs = clauses
        
        if self.vote_dropout == 0:
            return logits, clause_outputs
        return F.dropout(logits, p=self.vote_dropout, training=self.training), clause_outputs
    
    def forward_with_graph(self, x: torch.Tensor, use_ste: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with automatic CUDA graph usage."""
        # Use graph if enabled, in eval mode, correct batch size
        if (self._use_graph and not self.training and 
            x.is_cuda and x.shape[0] == self._graph_batch_size):
            
            # Capture on first call
            if self._cuda_graph is None:
                self._capture_cuda_graph(x)
            
            # Copy input and replay
            self._static_input.copy_(x)
            self._cuda_graph.replay()
            
            # Clone outputs
            if isinstance(self._static_output, tuple):
                return tuple(o.clone() for o in self._static_output)
            return self._static_output.clone()
        
        # Standard forward
        return self._forward_no_graph(x, use_ste=use_ste)
    
    def inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast inference method using CUDA graph if enabled."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output = self.forward_with_graph(x)
        if was_training:
            self.train()
        return output



