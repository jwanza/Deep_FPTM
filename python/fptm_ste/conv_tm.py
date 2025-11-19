from typing import Optional, Sequence, Tuple, Type
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tm import FuzzyPatternTM_STE, FuzzyPatternTM_STCM
from .deep_tm import DeepTMNetwork


def _class_supports_kwarg(cls: Type[nn.Module], name: str) -> bool:
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if name in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


class PatchDeepTMCore(nn.Module):
    """
    Small DeepTM-like core applied to flattened patches, weight-shared across positions.
    Supports passing STCM-specific kwargs into the internal layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        n_classes: int,
        n_clauses: int = 128,
        dropout: float = 0.1,
        tau: float = 0.5,
        *,
        input_shape: Optional[Tuple[int, int, int]] = None,
        layer_cls: Type[nn.Module] = FuzzyPatternTM_STE,
        layer_operator: Optional[str] = None,
        layer_ternary_voting: Optional[bool] = None,
        clause_dropout: float = 0.0,
        literal_dropout: float = 0.0,
        clause_bias_init: float = 0.0,
        layer_extra_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.input_shape = tuple(input_shape) if input_shape is not None else None
        self.input_dim = int(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.layer_cls = layer_cls
        prev = self.input_dim
        layer_extra_kwargs = dict(layer_extra_kwargs or {})

        # Hidden layers
        for idx, h in enumerate(hidden_dims):
            layer_kwargs = {}
            if idx == 0:
                layer_kwargs = dict(
                    input_shape=self.input_shape,
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
            if layer_operator is not None and _class_supports_kwarg(layer_cls, "operator"):
                tm_kwargs["operator"] = layer_operator
            if layer_ternary_voting is not None and _class_supports_kwarg(layer_cls, "ternary_voting"):
                tm_kwargs["ternary_voting"] = layer_ternary_voting
            # Pass-through any extra kwargs the layer might accept (e.g., ternary_band / ste_temperature for STCM)
            for k, v in layer_extra_kwargs.items():
                if _class_supports_kwarg(layer_cls, k):
                    tm_kwargs[k] = v
            self.layers.append(layer_cls(**tm_kwargs))
            self.norms.append(nn.LayerNorm(h))
            self.residuals.append(nn.Linear(prev, h, bias=False) if prev != h else nn.Identity())
            prev = h

        # Classifier
        classifier_kwargs = dict(
            n_features=prev,
            n_clauses=n_clauses * 2,
            n_classes=n_classes,
            tau=tau,
            clause_dropout=clause_dropout,
            literal_dropout=literal_dropout,
            clause_bias_init=clause_bias_init,
        )
        if layer_operator is not None and _class_supports_kwarg(layer_cls, "operator"):
            classifier_kwargs["operator"] = layer_operator
        if layer_ternary_voting is not None and _class_supports_kwarg(layer_cls, "ternary_voting"):
            classifier_kwargs["ternary_voting"] = layer_ternary_voting
        for k, v in layer_extra_kwargs.items():
            if _class_supports_kwarg(layer_cls, k):
                classifier_kwargs[k] = v
        self.classifier = layer_cls(**classifier_kwargs)

    def forward(self, x: torch.Tensor, use_ste: bool = True):
        # x: [batch, features]
        for layer, norm, res in zip(self.layers, self.norms, self.residuals):
            identity = res(x)
            logits, _ = layer(x, use_ste=use_ste)
            x = norm(self.dropout(torch.sigmoid(logits)) + identity)
        logits, clauses = self.classifier(x, use_ste=use_ste)
        return logits, clauses


class ConvTM2d(nn.Module):
    """
    Convolutional TM wrapper: shares a TM (or small DeepTM) core across spatial patches.
    Supports both STE and STCM cores and optional deep per-patch cores.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        *,
        n_clauses: int = 128,
        tau: float = 0.5,
        core_backend: str = "tm",  # {"tm","deeptm"} for STE; {"stcm","deepstcm"} for STCM
        core_hidden_dims: Optional[Sequence[int]] = None,
        clause_dropout: float = 0.0,
        literal_dropout: float = 0.0,
        clause_bias_init: float = 0.0,
        layer_cls: Type[nn.Module] = FuzzyPatternTM_STE,
        operator: Optional[str] = None,
        ternary_voting: Optional[bool] = None,
        ternary_band: float = 0.0,
        ste_temperature: float = 1.0,
        im2col_chunk: Optional[int] = None,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = int(kernel_size[0]), int(kernel_size[1])
        if isinstance(stride, int):
            sh, sw = stride, stride
        else:
            sh, sw = int(stride[0]), int(stride[1])
        if isinstance(padding, int):
            ph, pw = padding, padding
        else:
            ph, pw = int(padding[0]), int(padding[1])
        if isinstance(dilation, int):
            dh, dw = dilation, dilation
        else:
            dh, dw = int(dilation[0]), int(dilation[1])

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (kh, kw)
        self.stride = (sh, sw)
        self.padding = (ph, pw)
        self.dilation = (dh, dw)
        self.n_clauses = int(n_clauses)
        self.tau = float(tau)
        self.core_backend = (core_backend or "tm").lower()
        self.im2col_chunk = None if im2col_chunk is None else int(im2col_chunk)

        self.unfold = nn.Unfold(kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        n_features = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        input_shape = (self.in_channels, self.kernel_size[0], self.kernel_size[1])

        # Build core
        layer_extra_kwargs = {}
        if layer_cls is FuzzyPatternTM_STCM or layer_cls == FuzzyPatternTM_STCM:
            layer_extra_kwargs.update({"ternary_band": float(ternary_band), "ste_temperature": float(ste_temperature)})

        backend = self.core_backend
        if backend in {"tm", "stcm"}:
            tm_kwargs = dict(
                n_features=n_features,
                n_clauses=self.n_clauses,
                n_classes=self.out_channels,
                tau=self.tau,
                input_shape=input_shape,
                clause_dropout=clause_dropout,
                literal_dropout=literal_dropout,
                clause_bias_init=clause_bias_init,
            )
            if operator is not None and _class_supports_kwarg(layer_cls, "operator"):
                tm_kwargs["operator"] = operator
            if ternary_voting is not None and _class_supports_kwarg(layer_cls, "ternary_voting"):
                tm_kwargs["ternary_voting"] = ternary_voting
            for k, v in layer_extra_kwargs.items():
                if _class_supports_kwarg(layer_cls, k):
                    tm_kwargs[k] = v
            self.core = layer_cls(**tm_kwargs)
        elif backend in {"deeptm", "deepstcm"}:
            hidden = list(core_hidden_dims or [max(64, self.out_channels)])
            self.core = PatchDeepTMCore(
                input_dim=n_features,
                hidden_dims=hidden,
                n_classes=self.out_channels,
                n_clauses=self.n_clauses,
                dropout=0.1,
                tau=self.tau,
                input_shape=input_shape,
                layer_cls=layer_cls,
                layer_operator=operator,
                layer_ternary_voting=ternary_voting,
                clause_dropout=clause_dropout,
                literal_dropout=literal_dropout,
                clause_bias_init=clause_bias_init,
                layer_extra_kwargs=layer_extra_kwargs,
            )
        else:
            raise ValueError(f"Unknown core_backend '{core_backend}'.")

    def _out_hw(self, H: int, W: int) -> Tuple[int, int]:
        kh, kw = self.kernel_size
        ph, pw = self.padding
        dh, dw = self.dilation
        sh, sw = self.stride
        Ho = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        Wo = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        return int(Ho), int(Wo)

    def forward(self, x: torch.Tensor, use_ste: bool = True) -> torch.Tensor:
        # x: [B, C, H, W] -> patches: [B, F, L]
        B, _, H, W = x.shape
        patches = self.unfold(x)
        B_, F_, L = patches.shape
        assert B_ == B

        # Evaluate core over patches, optionally in chunks to limit memory
        if self.im2col_chunk is None or self.im2col_chunk <= 0 or L <= self.im2col_chunk:
            flat = patches.transpose(1, 2).reshape(B * L, F_)
            logits, _ = self.core(flat, use_ste=use_ste)  # [B*L, C_out]
            out = logits.view(B, L, self.out_channels).transpose(1, 2)  # [B, C_out, L]
        else:
            pieces = []
            step = self.im2col_chunk
            pt = patches.transpose(1, 2)  # [B, L, F]
            for start in range(0, L, step):
                end = min(start + step, L)
                chunk = pt[:, start:end, :].reshape(B * (end - start), F_)
                logits, _ = self.core(chunk, use_ste=use_ste)
                pieces.append(logits.view(B, end - start, self.out_channels))
            out = torch.cat(pieces, dim=1).transpose(1, 2)  # [B, C_out, L]

        Ho, Wo = self._out_hw(H, W)
        return out.view(B, self.out_channels, Ho, Wo)


class ConvSTE2d(ConvTM2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        *,
        n_clauses: int = 128,
        tau: float = 0.5,
        core_backend: str = "tm",
        core_hidden_dims: Optional[Sequence[int]] = None,
        clause_dropout: float = 0.0,
        literal_dropout: float = 0.0,
        clause_bias_init: float = 0.0,
        im2col_chunk: Optional[int] = None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            n_clauses=n_clauses,
            tau=tau,
            core_backend=core_backend,
            core_hidden_dims=core_hidden_dims,
            clause_dropout=clause_dropout,
            literal_dropout=literal_dropout,
            clause_bias_init=clause_bias_init,
            layer_cls=FuzzyPatternTM_STE,
            operator=None,
            ternary_voting=None,
            ternary_band=0.0,
            ste_temperature=1.0,
            im2col_chunk=im2col_chunk,
        )


class ConvSTCM2d(ConvTM2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        *,
        n_clauses: int = 128,
        tau: float = 0.5,
        core_backend: str = "stcm",
        core_hidden_dims: Optional[Sequence[int]] = None,
        clause_dropout: float = 0.0,
        literal_dropout: float = 0.0,
        clause_bias_init: float = 0.0,
        operator: str = "capacity",
        ternary_voting: bool = False,
        ternary_band: float = 0.05,
        ste_temperature: float = 1.0,
        im2col_chunk: Optional[int] = None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            n_clauses=n_clauses,
            tau=tau,
            core_backend=core_backend,
            core_hidden_dims=core_hidden_dims,
            clause_dropout=clause_dropout,
            literal_dropout=literal_dropout,
            clause_bias_init=clause_bias_init,
            layer_cls=FuzzyPatternTM_STCM,
            operator=operator,
            ternary_voting=ternary_voting,
            ternary_band=ternary_band,
            ste_temperature=ste_temperature,
            im2col_chunk=im2col_chunk,
        )




class ConvTM2dOptimized(ConvTM2d):
    """
    Optimized Convolutional TM that uses F.conv2d instead of Unfold+MatMul.
    Significantly faster and more memory efficient for STE/STCM backends.
    """

    def forward(self, x: torch.Tensor, use_ste: bool = True) -> torch.Tensor:
        # Fallback for deep backends or unsupported cores
        if self.core_backend not in {"tm", "stcm"} or not hasattr(self.core, "get_masks"):
            return super().forward(x, use_ste=use_ste)

        # Retrieve masks from core
        masks = self.core.get_masks(use_ste=use_ste)
        
        C_in = self.in_channels
        K_h, K_w = self.kernel_size
        
        # Helper to reshape [Half, Features] -> [Half, C_in, K_h, K_w]
        def reshape_mask(m):
            # n_features = C_in * K_h * K_w
            return m.view(-1, C_in, K_h, K_w)

        clause_outputs = None

        if isinstance(self.core, FuzzyPatternTM_STE):
            p_pos, p_neg, p_pos_inv, p_neg_inv = [reshape_mask(m) for m in masks]
            
            # Score = scale * (conv(x, inv - pos) + sum(pos))
            scale = 4.0 / self.core.n_features
            
            # Positive Bank
            w_pos = p_pos_inv - p_pos
            b_pos = p_pos.sum(dim=(1, 2, 3))
            conv_pos = F.conv2d(x, w_pos, bias=b_pos, stride=self.stride, padding=self.padding, dilation=self.dilation)
            pos_score = scale * conv_pos
            pos_prod = torch.exp(-torch.clamp(pos_score, min=0.0, max=10.0))
            
            # Negative Bank
            w_neg = p_neg_inv - p_neg
            b_neg = p_neg.sum(dim=(1, 2, 3))
            conv_neg = F.conv2d(x, w_neg, bias=b_neg, stride=self.stride, padding=self.padding, dilation=self.dilation)
            neg_score = scale * conv_neg
            neg_prod = torch.exp(-torch.clamp(neg_score, min=0.0, max=10.0))
            
            clause_outputs = torch.cat([pos_prod, neg_prod], dim=1)

        elif isinstance(self.core, FuzzyPatternTM_STCM):
            pos_pos_flat, neg_pos_flat, pos_inv_flat, neg_inv_flat = masks
            
            def compute_strength(m_pos_flat, m_inv_flat):
                # Capacity constraint is calculated on flat vectors
                capacity = self.core._clause_capacity(m_pos_flat, m_inv_flat).squeeze(0) # [Half]
                
                m_pos = reshape_mask(m_pos_flat)
                m_inv = reshape_mask(m_inv_flat)
                
                # mismatch = conv(x, mask_inv - mask_pos) + sum(mask_pos)
                w = m_inv - m_pos
                b = m_pos.sum(dim=(1, 2, 3))
                
                mismatch = F.conv2d(x, w, bias=b, stride=self.stride, padding=self.padding, dilation=self.dilation)
                
                if self.core.operator == "capacity":
                    # raw = capacity - mismatch
                    raw = capacity.view(1, -1, 1, 1) - mismatch
                    return self.core._straight_relu(raw)
                elif self.core.operator == "product":
                     scaled = torch.clamp(mismatch * self.core.product_scale, min=0.0, max=10.0)
                     return torch.exp(-scaled)
                else:
                    raw = capacity.view(1, -1, 1, 1) - mismatch
                    return self.core._straight_relu(raw)

            pos_strength = compute_strength(pos_pos_flat, pos_inv_flat)
            neg_strength = compute_strength(neg_pos_flat, neg_inv_flat)
            
            clause_outputs = torch.cat([pos_strength, -neg_strength], dim=1)
            
            if self.core.vote_clamp is not None:
                clause_outputs = clause_outputs.clamp(-self.core.vote_clamp, self.core.vote_clamp)

        else:
             return super().forward(x, use_ste=use_ste)

        if self.training and self.core.clause_dropout > 0.0:
             clause_outputs = F.dropout(clause_outputs, p=self.core.clause_dropout, training=True)
             
        # Voting
        # self.core.voting is [n_clauses, n_classes]
        # We need [n_classes, n_clauses, 1, 1] for conv
        
        voting_weight = self.core._voting_matrix(use_ste) if hasattr(self.core, "_voting_matrix") else self.core.voting
        w_vote = voting_weight.t().unsqueeze(2).unsqueeze(3) # [n_classes, n_clauses, 1, 1]
        
        # Clause bias
        if hasattr(self.core, "clause_bias"):
             clause_outputs = clause_outputs + self.core.clause_bias.view(1, -1, 1, 1)
        
        logits = F.conv2d(clause_outputs, w_vote)
        return logits
