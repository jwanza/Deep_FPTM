import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Sequence, Tuple, Type
from .tm import FuzzyPatternTM_STE, prepare_tm_input


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
        self.layer_cls = layer_cls
        self.layer_extra_kwargs = dict(layer_extra_kwargs or {})

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
                clause_dropout=clause_dropout,
                literal_dropout=literal_dropout,
                clause_bias_init=clause_bias_init,
                **layer_kwargs,
            )
            tm_kwargs.update(self.layer_extra_kwargs)
            if layer_operator is not None and _class_supports_kwarg(layer_cls, "operator"):
                tm_kwargs["operator"] = layer_operator
            if layer_ternary_voting is not None and _class_supports_kwarg(layer_cls, "ternary_voting"):
                tm_kwargs["ternary_voting"] = layer_ternary_voting
            self.layers.append(layer_cls(**tm_kwargs))
            self.norms.append(nn.LayerNorm(h))
            self.residuals.append(nn.Linear(prev, h, bias=False) if prev != h else nn.Identity())
            prev = h

        classifier_kwargs = dict(
            n_features=prev,
            n_clauses=n_clauses * 2,
            n_classes=n_classes,
            tau=tau,
            clause_dropout=clause_dropout,
            literal_dropout=literal_dropout,
            clause_bias_init=clause_bias_init,
        )
        classifier_kwargs.update(self.layer_extra_kwargs)
        if layer_operator is not None and _class_supports_kwarg(layer_cls, "operator"):
            classifier_kwargs["operator"] = layer_operator
        if layer_ternary_voting is not None and _class_supports_kwarg(layer_cls, "ternary_voting"):
            classifier_kwargs["ternary_voting"] = layer_ternary_voting
        self.classifier = layer_cls(**classifier_kwargs)
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
            logits, _ = layer(x, use_ste=use_ste)
            x = norm(self.dropout(torch.sigmoid(logits)) + identity)
        logits, clauses = self.classifier(x, use_ste=use_ste)
        return logits, clauses

    def set_tau(self, tau: float) -> None:
        for layer in self.layers:
            if hasattr(layer, "tau"):
                layer.tau = tau
        if hasattr(self.classifier, "tau"):
            self.classifier.tau = tau



