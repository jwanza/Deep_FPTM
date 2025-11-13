import math
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .tm import FuzzyPatternTM_STE
from .deep_tm import DeepTMNetwork
from .swin_tm import DropPath, WindowAttention, InstrumentedMultiheadAttention
from .tm_positional import RelativePositionBias2D, RotaryEmbedding
from .swin_pyramid_tm import PatchMerging, window_partition, window_reverse


def _to_sequence(value: Union[int, float, Sequence[Union[int, float]]], length: int) -> Tuple[Union[int, float], ...]:
    if isinstance(value, Sequence):
        if len(value) != length:
            raise ValueError(f"Expected sequence of length {length}, got {len(value)}.")
        return tuple(value)
    return tuple(value for _ in range(length))


def _expand_vit_clauses(tm_clauses: Union[int, Sequence[int]], depth: int) -> List[int]:
    if isinstance(tm_clauses, Sequence) and not isinstance(tm_clauses, (str, bytes)):
        clause_list = [int(max(1, c)) for c in tm_clauses]
        if len(clause_list) != depth:
            if len(clause_list) == 0:
                clause_list = [64] * depth
            else:
                clause_list = (clause_list + [clause_list[-1]])[:depth]
    else:
        clause_list = [int(max(1, tm_clauses))] * depth
    return clause_list

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.mean(x.pow(2), dim=-1, keepdim=True)
        denom = torch.sqrt(norm + self.eps)
        shape = [1] * (x.dim() - 1) + [-1]
        return x * self.weight.view(*shape) / denom


class ScaleNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True)
        return self.scale * x / (norm + self.eps)


def build_norm(norm_type: str, dim: int) -> nn.Module:
    key = norm_type.lower()
    if key in {"layer", "layernorm", "ln"}:
        return nn.LayerNorm(dim)
    if key in {"rms", "rmsnorm"}:
        return RMSNorm(dim)
    if key in {"scale", "scalenorm"}:
        return ScaleNorm(dim)
    raise ValueError(f"Unsupported norm type '{norm_type}'.")


def _init_linear(layer: nn.Linear, *, gain: float = 1.0) -> None:
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)



class PatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, *, flatten: bool = True) -> None:
        super().__init__()
        self.flatten = flatten
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten:
            x = x.view(B, H, W, C)
        return x


class TMFeedForward(nn.Module):
    """Feed-forward module backed by TM / DeepTM with advanced gating."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        backend: str,
        n_clauses: int,
        tm_tau: float,
        dropout: float,
        gate_type: str = "none",
        gate_activation: str = "sigmoid",
        norm_type: str = "layernorm",
        clause_dropout: float = 0.0,
        literal_dropout: float = 0.0,
        sparsity_weight: float = 0.0,
        clause_bias_init: float = 0.0,
        mix_type: str = "none",
        bitwise_mix: bool = False,
        learnable_tau: bool = False,
        tau_min: float = 0.05,
        tau_max: float = 0.95,
        tau_ema_beta: Optional[float] = None,
        clause_attention: bool = False,
        clause_routing: bool = False,
        continuous_bypass: bool = False,
        bypass_scale: float = 1.0,
        use_flash_attention: bool = False,
        use_residual_attention: bool = False,
    ) -> None:
        super().__init__()
        self.backend = backend.lower()
        self.hidden_dim = hidden_dim
        self.output_dim = dim
        self.gate_type = gate_type.lower()
        self.gate_activation = gate_activation.lower()
        self.mix_type = mix_type.lower()
        self.bitwise_mix = bitwise_mix
        self.learnable_tau = learnable_tau
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_ema_beta = tau_ema_beta
        self.clause_attention = clause_attention
        self.clause_routing = clause_routing
        self.continuous_bypass = continuous_bypass
        self.norm = build_norm(norm_type, dim)
        self._requires_split_gate = self.gate_type in {"linear", "geglu", "swiglu"}
        proj_dim = hidden_dim * 2 if self._requires_split_gate else hidden_dim
        self.proj_in = nn.Linear(dim, proj_dim)
        _init_linear(self.proj_in)
        self.activation = nn.GELU()
        self.clause_dropout = clause_dropout
        self.literal_dropout = literal_dropout
        self.sparsity_weight = sparsity_weight
        self._sparsity_penalty: Optional[torch.Tensor] = None
        self.pre_linear: Optional[nn.Linear] = None
        self.post_linear: Optional[nn.Linear] = None
        if "linear" in self.mix_type:
            self.pre_linear = nn.Linear(dim, dim)
            self.post_linear = nn.Linear(dim, dim)
            _init_linear(self.pre_linear)
            _init_linear(self.post_linear)
        self.depthwise: Optional[nn.Conv1d] = None
        if "depthwise" in self.mix_type:
            self.depthwise = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.bitwise_linear: Optional[nn.Linear] = None
        if self.bitwise_mix:
            self.bitwise_linear = nn.Linear(hidden_dim * 2, hidden_dim)
            _init_linear(self.bitwise_linear)
        if self.learnable_tau:
            init_tau = float(max(min(tm_tau, tau_max), tau_min))
            init = torch.log(torch.tensor(init_tau) / (1 - torch.tensor(init_tau)))
            self.tau_param = nn.Parameter(init)
        else:
            self.register_parameter("tau_param", None)
        if self.tau_ema_beta is not None:
            self.register_buffer("tau_ema", torch.tensor(tm_tau))
        else:
            self.register_buffer("tau_ema", None)
        self.clause_attn_proj: Optional[nn.Module] = None
        if self.clause_attention:
            self.clause_attn_proj = nn.Sequential(
                nn.LayerNorm(n_clauses),
                nn.Linear(n_clauses, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
            for layer in self.clause_attn_proj:
                if isinstance(layer, nn.Linear):
                    _init_linear(layer)
        self.clause_gate: Optional[nn.Module] = None
        if self.clause_routing:
            self.clause_gate = nn.Sequential(
                nn.LayerNorm(n_clauses),
                nn.Linear(n_clauses, dim),
                nn.Sigmoid(),
            )
            for layer in self.clause_gate:
                if isinstance(layer, nn.Linear):
                    _init_linear(layer)
        self.bypass: Optional[nn.Module] = None
        if self.continuous_bypass:
            self.bypass = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
            for layer in self.bypass:
                if isinstance(layer, nn.Linear):
                    _init_linear(layer)
            self.bypass_gain = nn.Parameter(torch.tensor(bypass_scale))
        else:
            self.register_parameter("bypass_gain", None)
        self.last_clause_outputs: Optional[torch.Tensor] = None
        if self.backend == "ste":
            self.core = FuzzyPatternTM_STE(
                hidden_dim,
                n_clauses,
                dim,
                tau=tm_tau,
                clause_dropout=clause_dropout,
                literal_dropout=literal_dropout,
                clause_bias_init=clause_bias_init,
            )
        elif self.backend in {"deep", "deeptm"}:
            self.core = DeepTMNetwork(
                input_dim=hidden_dim,
                hidden_dims=[hidden_dim],
                n_classes=dim,
                n_clauses=n_clauses,
                dropout=dropout,
                tau=tm_tau,
                input_shape=None,
                auto_expand_grayscale=False,
                allow_channel_reduce=False,
                clause_dropout=clause_dropout,
                literal_dropout=literal_dropout,
                clause_bias_init=clause_bias_init,
            )
        else:
            raise ValueError(f"Unsupported TM backend '{backend}'.")
        self.gate_core: Optional[nn.Module] = None
        if self.gate_type == "tm":
            self.gate_core = FuzzyPatternTM_STE(
                hidden_dim,
                n_clauses,
                hidden_dim,
                tau=tm_tau,
                clause_dropout=0.0,
                literal_dropout=0.0,
                clause_bias_init=0.0,
            )
        elif self.gate_type == "deeptm":
            self.gate_core = DeepTMNetwork(
                input_dim=hidden_dim,
                hidden_dims=[hidden_dim],
                n_classes=hidden_dim,
                n_clauses=n_clauses,
                dropout=dropout,
                tau=tm_tau,
                input_shape=None,
                auto_expand_grayscale=False,
                allow_channel_reduce=False,
                clause_dropout=0.0,
                literal_dropout=0.0,
                clause_bias_init=0.0,
            )
        self.dropout = nn.Dropout(dropout)

    def _apply_gate_activation(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.gate_activation == "sigmoid":
            return torch.sigmoid(tensor)
        if self.gate_activation == "tanh":
            return torch.tanh(tensor)
        if self.gate_activation == "relu":
            return F.relu(tensor)
        raise ValueError(f"Unsupported gate activation '{self.gate_activation}'.")

    def _update_tau(self, tau: float) -> None:
        if hasattr(self.core, "set_tau"):
            self.core.set_tau(tau)
        elif hasattr(self.core, "tau"):
            self.core.tau = tau
        if self.gate_core is not None:
            if hasattr(self.gate_core, "set_tau"):
                self.gate_core.set_tau(tau)
            elif hasattr(self.gate_core, "tau"):
                self.gate_core.tau = tau

    def forward(self, x: torch.Tensor, use_ste: bool = True) -> torch.Tensor:
        B, T, C = x.shape
        y = self.norm(x)
        if self.pre_linear is not None:
            y = self.pre_linear(y)
        if self.depthwise is not None:
            y = self.depthwise(y.transpose(1, 2)).transpose(1, 2)
        proj = self.proj_in(y)
        if self._requires_split_gate:
            value, gate = proj.chunk(2, dim=-1)
            if self.gate_type == "swiglu":
                value = F.silu(value)
                gate = torch.sigmoid(gate)
            elif self.gate_type == "geglu":
                value = F.gelu(value)
                gate = torch.sigmoid(gate)
            else:
                value = self.activation(value)
                gate = self._apply_gate_activation(gate)
            fused = value * gate
        else:
            fused = self.activation(proj)
            if self.gate_type in {"tm", "deeptm"}:
                if self.gate_core is None:
                    raise RuntimeError("TM-based gate requested but gate_core is uninitialized.")
                gate_input = torch.sigmoid(fused.reshape(B * T, -1))
                gate_logits, _ = self.gate_core(gate_input, use_ste=use_ste)
                gate = self._apply_gate_activation(gate_logits.view(B, T, -1))
                fused = fused * gate
        if self.bitwise_linear is not None:
            concat = torch.cat([fused, 1.0 - fused], dim=-1)
            fused = self.bitwise_linear(concat)
        if self.learnable_tau or self.tau_ema_beta is not None:
            tau_val = None
            if self.learnable_tau:
                tau_val = torch.sigmoid(self.tau_param)
                tau_val = self.tau_min + (self.tau_max - self.tau_min) * tau_val
            if self.tau_ema_beta is not None:
                source = tau_val if tau_val is not None else torch.tensor(getattr(self.core, 'tau', 0.5), device=fused.device)
                self.tau_ema = self.tau_ema * self.tau_ema_beta + source.detach() * (1 - self.tau_ema_beta)
                tau_val = self.tau_ema
            if tau_val is not None:
                self._update_tau(float(tau_val.detach().item()))
        bypass_out = None
        if self.bypass is not None:
            bypass_out = self.bypass(y)
        flat = fused.reshape(B * T, -1)
        logits, clause_outputs = self.core(flat, use_ste=use_ste)
        if self.sparsity_weight > 0.0:
            self._sparsity_penalty = clause_outputs.abs().mean() * self.sparsity_weight
        else:
            self._sparsity_penalty = None
        self.last_clause_outputs = clause_outputs.view(B, T, -1).detach()
        logits = logits.view(B, T, C)
        if self.clause_attn_proj is not None:
            clause_ctx = self.clause_attn_proj(clause_outputs).view(B, T, C)
            logits = logits + clause_ctx
        if self.clause_gate is not None:
            gate = self.clause_gate(clause_outputs).view(B, T, C)
            logits = logits * gate
        if bypass_out is not None:
            gain = torch.tanh(self.bypass_gain) if self.bypass_gain is not None else 1.0
            logits = logits + gain * bypass_out
        if self.post_linear is not None:
            logits = self.post_linear(logits)
        return self.dropout(logits)

    @property
    def sparsity_penalty(self) -> Optional[torch.Tensor]:
        return self._sparsity_penalty

class TMEncoderBlock(nn.Module):
    """ViT-style encoder block with TM/DeepTM feed-forward."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float,
        backend: str,
        n_clauses: int,
        tm_tau: float,
        drop: float,
        attn_drop: float,
        drop_path: float,
        gate_type: str = "none",
        gate_activation: str = "sigmoid",
        norm_type: str = "layernorm",
        layerscale_init: float = 1e-4,
        enable_layerscale: bool = True,
        clause_dropout: float = 0.0,
        literal_dropout: float = 0.0,
        sparsity_weight: float = 0.0,
        clause_bias_init: float = 0.0,
        mix_type: str = "none",
        bitwise_mix: bool = False,
        learnable_tau: bool = False,
        tau_min: float = 0.05,
        tau_max: float = 0.95,
        tau_ema_beta: Optional[float] = None,
        clause_attention: bool = False,
        clause_routing: bool = False,
        continuous_bypass: bool = False,
        bypass_scale: float = 1.0,
        use_flash_attention: bool = False,
        use_residual_attention: bool = False,
        grid_size: Optional[Tuple[int, int]] = None,
        include_cls_token: bool = False,
        relative_position_type: str = "none",
    ) -> None:
        super().__init__()
        self.norm1 = build_norm(norm_type, dim)
        rpb = None
        rotary = None
        position_type = relative_position_type.lower()
        if position_type not in {"none", "learned", "rotary"}:
            raise ValueError(f"Unknown relative position type '{relative_position_type}'.")
        if position_type in {"learned", "rotary"} and grid_size is None:
            raise ValueError("grid_size must be provided when using relative positional encodings.")
        if position_type == "learned" and grid_size is not None:
            rpb = RelativePositionBias2D(num_heads, grid_size, has_cls_token=include_cls_token)
        if position_type == "rotary" and grid_size is not None:
            seq_len = grid_size[0] * grid_size[1] + (1 if include_cls_token else 0)
            rotary = RotaryEmbedding(dim // num_heads, max_seq_len=seq_len)
        self.attn = InstrumentedMultiheadAttention(
            dim,
            num_heads,
            attn_drop,
            use_flash=use_flash_attention,
            relative_position_bias=rpb,
            rotary_emb=rotary,
        )
        self.attn_drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = TMFeedForward(
            dim,
            hidden_dim,
            backend=backend,
            n_clauses=n_clauses,
            tm_tau=tm_tau,
            dropout=drop,
            gate_type=gate_type,
            gate_activation=gate_activation,
            norm_type=norm_type,
            clause_dropout=clause_dropout,
            literal_dropout=literal_dropout,
            sparsity_weight=sparsity_weight,
            clause_bias_init=clause_bias_init,
            mix_type=mix_type,
            bitwise_mix=bitwise_mix,
            learnable_tau=learnable_tau,
            tau_min=tau_min,
            tau_max=tau_max,
            tau_ema_beta=tau_ema_beta,
            clause_attention=clause_attention,
            clause_routing=clause_routing,
            continuous_bypass=continuous_bypass,
            bypass_scale=bypass_scale,
        )
        self.norm2 = build_norm(norm_type, dim)
        self.num_heads = num_heads
        self.enable_layerscale = enable_layerscale
        self.use_residual_attention = use_residual_attention
        self.last_attention: Optional[torch.Tensor] = None
        self.last_head_context: Optional[torch.Tensor] = None
        if enable_layerscale:
            self.attn_scale = nn.Parameter(torch.ones(dim))
            self.tm_scale = nn.Parameter(torch.ones(dim) * layerscale_init)
        else:
            self.register_buffer('attn_scale', None)
            self.register_buffer('tm_scale', None)

    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
        *,
        collect_diagnostics: bool = False,
        attn_residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_in = self.norm1(x)
        attn_out, head_context, _ = self.attn(attn_in)
        attn_out = self.attn_drop(attn_out)
        if self.use_residual_attention and attn_residual is not None:
            attn_out = attn_out + attn_residual
        if self.enable_layerscale and self.attn_scale is not None:
            attn_out = attn_out * self.attn_scale.view(1, 1, -1)
        self.last_attention = attn_out
        self.last_head_context = head_context if collect_diagnostics else None
        x = x + self.drop_path(attn_out)
        ff_input = self.norm2(x)
        ff_out = self.ffn(ff_input, use_ste=use_ste)
        if self.enable_layerscale and self.tm_scale is not None:
            ff_out = ff_out * self.tm_scale.view(1, 1, -1)
        x = x + self.drop_path(ff_out)
        if collect_diagnostics:
            return x, head_context
        return x

class TMSwinBlock(nn.Module):
    """Windowed attention block with TM feed-forward."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        *,
        mlp_ratio: float,
        backend: str,
        n_clauses: int,
        tm_tau: float,
        drop: float,
        attn_drop: float,
        drop_path: float,
        gate_type: str = "none",
        gate_activation: str = "sigmoid",
        norm_type: str = "layernorm",
        layerscale_init: float = 1e-4,
        enable_layerscale: bool = True,
        clause_dropout: float = 0.0,
        literal_dropout: float = 0.0,
        sparsity_weight: float = 0.0,
        clause_bias_init: float = 0.0,
        mix_type: str = "none",
        bitwise_mix: bool = False,
        learnable_tau: bool = False,
        tau_min: float = 0.05,
        tau_max: float = 0.95,
        tau_ema_beta: Optional[float] = None,
        clause_attention: bool = False,
        clause_routing: bool = False,
        continuous_bypass: bool = False,
        bypass_scale: float = 1.0,
        use_flash_attention: bool = False,
        use_residual_attention: bool = False,
        relative_position_type: str = "none",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = build_norm(norm_type, dim)
        position_type = relative_position_type.lower()
        if position_type not in {"none", "learned", "rotary"}:
            raise ValueError(f"Unknown relative position type '{relative_position_type}'.")
        rpb = None
        rotary = None
        if position_type == "learned":
            rpb = RelativePositionBias2D(num_heads, (window_size, window_size), has_cls_token=False)
        if position_type == "rotary":
            rotary = RotaryEmbedding(dim // num_heads, max_seq_len=window_size * window_size)
        self.attn = WindowAttention(
            dim,
            num_heads,
            attn_drop,
            use_flash=use_flash_attention,
            relative_position_bias=rpb,
            rotary_emb=rotary,
        )
        self.attn_drop = nn.Dropout(drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = TMFeedForward(
            dim,
            hidden_dim,
            backend=backend,
            n_clauses=n_clauses,
            tm_tau=tm_tau,
            dropout=drop,
            gate_type=gate_type,
            gate_activation=gate_activation,
            norm_type=norm_type,
            clause_dropout=clause_dropout,
            literal_dropout=literal_dropout,
            sparsity_weight=sparsity_weight,
            clause_bias_init=clause_bias_init,
            mix_type=mix_type,
            bitwise_mix=bitwise_mix,
            learnable_tau=learnable_tau,
            tau_min=tau_min,
            tau_max=tau_max,
            tau_ema_beta=tau_ema_beta,
            clause_attention=clause_attention,
            clause_routing=clause_routing,
            continuous_bypass=continuous_bypass,
            bypass_scale=bypass_scale,
        )
        self.norm2 = build_norm(norm_type, dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.enable_layerscale = enable_layerscale
        self.use_residual_attention = use_residual_attention
        self.last_attention: Optional[torch.Tensor] = None
        self.last_head_tokens: Optional[torch.Tensor] = None
        self.last_hw: Optional[Tuple[int, int]] = None
        if enable_layerscale:
            self.attn_scale = nn.Parameter(torch.ones(dim))
            self.tm_scale = nn.Parameter(torch.ones(dim) * layerscale_init)
        else:
            self.register_buffer('attn_scale', None)
            self.register_buffer('tm_scale', None)

    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        use_ste: bool = True,
        *,
        collect_diagnostics: bool = False,
        attn_residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, int, Optional[torch.Tensor]]:
        B, L, C = x.shape
        assert L == H * W, "Unexpected token count for Swin block"
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        H_pad, W_pad = x.shape[1:3]

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = window_partition(x, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_out, attn_weights, head_context = self.attn(windows)
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
        head_dim = self.attn.attn.head_dim
        head_context = head_context.view(-1, self.window_size, self.window_size, self.attn.attn.num_heads, head_dim)
        head_context_flat = head_context.view(-1, self.window_size, self.window_size, self.attn.attn.num_heads * head_dim)
        head_context_merged = window_reverse(head_context_flat, self.window_size, H_pad, W_pad)
        head_context_merged = head_context_merged.view(B, H_pad, W_pad, self.attn.attn.num_heads, head_dim)
        if self.shift_size > 0:
            attn_out = torch.roll(attn_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            head_context_merged = torch.roll(head_context_merged, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = window_reverse(attn_out, self.window_size, H_pad, W_pad)
        if pad_h or pad_w:
            x = x[:, :H, :W, :]
            head_context_merged = head_context_merged[:, :H, :W, :, :]
        x = x.reshape(B, H * W, C)
        head_tokens = head_context_merged.reshape(B, H * W, self.attn.attn.num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        attn_proj = self.attn_drop(x)
        if self.use_residual_attention and attn_residual is not None:
            attn_proj = attn_proj + attn_residual
        if self.enable_layerscale and self.attn_scale is not None:
            attn_proj = attn_proj * self.attn_scale.view(1, 1, -1)
        self.last_attention = attn_proj
        self.last_head_tokens = head_tokens if collect_diagnostics else None
        self.last_hw = (H, W)
        x = shortcut + self.drop_path1(attn_proj)

        ff_out = self.ffn(self.norm2(x), use_ste=use_ste)
        if self.enable_layerscale and self.tm_scale is not None:
            ff_out = ff_out * self.tm_scale.view(1, 1, -1)
        x = x + self.drop_path2(ff_out)
        return x, H, W, head_tokens if collect_diagnostics else None

class TMSwinStage(nn.Module):
    """Stack of Swin blocks followed by optional patch merging."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        *,
        backend: str,
        mlp_ratio: float,
        n_clauses: int,
        tm_tau: float,
        drop: float,
        attn_drop: float,
        drop_path: Sequence[float],
        downsample: bool,
        use_checkpoint: bool,
        gate_type: str = "none",
        gate_activation: str = "sigmoid",
        norm_type: str = "layernorm",
        layerscale_init: float = 1e-4,
        enable_layerscale: bool = True,
        clause_dropout: float = 0.0,
        literal_dropout: float = 0.0,
        sparsity_weight: float = 0.0,
        clause_bias_init: float = 0.0,
        mix_type: str = "none",
        bitwise_mix: bool = False,
        learnable_tau: bool = False,
        tau_min: float = 0.05,
        tau_max: float = 0.95,
        tau_ema_beta: Optional[float] = None,
        clause_attention: bool = False,
        clause_routing: bool = False,
        continuous_bypass: bool = False,
        bypass_scale: float = 1.0,
        use_flash_attention: bool = False,
        use_residual_attention: bool = False,
        relative_position_type: str = "none",
    ) -> None:
        super().__init__()
        blocks = []
        for idx in range(depth):
            shift = 0 if idx % 2 == 0 else window_size // 2
            blocks.append(
                TMSwinBlock(
                    dim,
                    num_heads,
                    window_size,
                    shift,
                    mlp_ratio=mlp_ratio,
                    backend=backend,
                    n_clauses=n_clauses,
                    tm_tau=tm_tau,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[idx],
                    gate_type=gate_type,
                    gate_activation=gate_activation,
                    norm_type=norm_type,
                    layerscale_init=layerscale_init,
                    enable_layerscale=enable_layerscale,
                    clause_dropout=clause_dropout,
                    literal_dropout=literal_dropout,
                    sparsity_weight=sparsity_weight,
                    clause_bias_init=clause_bias_init,
                    mix_type=mix_type,
                    bitwise_mix=bitwise_mix,
                    learnable_tau=learnable_tau,
                    tau_min=tau_min,
                    tau_max=tau_max,
                    tau_ema_beta=tau_ema_beta,
                    clause_attention=clause_attention,
                    clause_routing=clause_routing,
                    continuous_bypass=continuous_bypass,
                    bypass_scale=bypass_scale,
                    use_flash_attention=use_flash_attention,
                    use_residual_attention=use_residual_attention,
                    relative_position_type=relative_position_type,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.downsample = PatchMerging(dim, dim * 2) if downsample else None
        self.use_checkpoint = use_checkpoint
        self.use_residual_attention = use_residual_attention

    def forward(self, x: torch.Tensor, H: int, W: int, use_ste: bool = True,
                *, collect_diagnostics: bool = False,
                record_callback: Optional[Callable[[str, torch.Tensor, int, int, Optional[torch.Tensor]], None]] = None,
                stage_index: int = 0) -> Tuple[torch.Tensor, int, int]:
        attn_residual = None
        for block_idx, block in enumerate(self.blocks):
            if self.use_checkpoint and self.training and x.requires_grad:
                residual = attn_residual if self.use_residual_attention else None

                def _forward(inp, blk=block, h=H, w=W, res=residual):  # type: ignore
                    out, _, _, _ = blk(inp, h, w, use_ste=use_ste, collect_diagnostics=False, attn_residual=res)
                    return out

                x = checkpoint(_forward, x)
                head_tokens = None
            else:
                x, H, W, head_tokens = block(
                    x,
                    H,
                    W,
                    use_ste=use_ste,
                    collect_diagnostics=collect_diagnostics,
                    attn_residual=attn_residual if self.use_residual_attention else None,
                )
            if self.use_residual_attention:
                attn_residual = block.last_attention
            if collect_diagnostics and record_callback is not None:
                record_callback(f"stage{stage_index + 1}_block{block_idx + 1}", x, H, W, head_tokens)
        if collect_diagnostics and record_callback is not None:
            record_callback(f"stage{stage_index + 1}_out", x, H, W, None)
        if self.downsample is not None:
            x = x.reshape(-1, H, W, x.shape[-1])
            x = self.downsample(x)
            H, W = x.shape[1], x.shape[2]
            x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x, H, W

class UnifiedTMTransformer(nn.Module):
    """Vision transformer with TM/DeepTM feed-forward options."""

    def __init__(
        self,
        *,
        num_classes: int,
        architecture: str = "vit",
        backend: str = "ste",
        image_size: Tuple[int, int] = (32, 32),
        in_channels: int = 3,
        patch_size: int = 4,
        embed_dim: Union[int, Sequence[int]] = 96,
        depths: Union[int, Sequence[int]] = 4,
        num_heads: Union[int, Sequence[int]] = 3,
        mlp_ratio: Union[float, Sequence[float]] = 4.0,
        tm_clauses: Union[int, Sequence[int]] = 256,
        tm_tau: float = 0.5,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        drop_path_schedule: str = "linear",
        window_size: int = 7,
        use_cls_token: bool = True,
        pool: str = "cls",
        grad_checkpoint: bool = False,
        ff_gate: str = "none",
        ff_gate_activation: str = "sigmoid",
        layerscale_init: float = 1e-4,
        use_layerscale: bool = True,
        clause_dropout: float = 0.0,
        literal_dropout: float = 0.0,
        ff_sparsity_weight: float = 0.0,
        clause_bias_init: float = 0.0,
        norm_type: str = "layernorm",
        ff_mix_type: str = "none",
        ff_bitwise_mix: bool = False,
        learnable_tau: bool = False,
        tau_min: float = 0.05,
        tau_max: float = 0.95,
        tau_ema_beta: Optional[float] = None,
        clause_attention: bool = False,
        clause_routing: bool = False,
        continuous_bypass: bool = False,
        bypass_scale: float = 1.0,
        use_flash_attention: bool = False,
        use_residual_attention: bool = False,
        relative_position_type: str = "none",
    ) -> None:
        super().__init__()
        self.architecture = architecture.lower()
        self.backend = backend.lower()
        self.num_classes = num_classes
        self.pool = pool
        self.grad_checkpoint = grad_checkpoint
        self.ff_gate = ff_gate.lower()
        self.ff_gate_activation = ff_gate_activation.lower()
        self.layerscale_init = layerscale_init
        self.use_layerscale = use_layerscale
        self.clause_dropout = clause_dropout
        self.literal_dropout = literal_dropout
        self.ff_sparsity_weight = ff_sparsity_weight
        self.clause_bias_init = clause_bias_init
        self.norm_type = norm_type
        self.ff_mix_type = ff_mix_type
        self.ff_bitwise_mix = ff_bitwise_mix
        self.learnable_tau = learnable_tau
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_ema_beta = tau_ema_beta
        self.clause_attention = clause_attention
        self.clause_routing = clause_routing
        self.continuous_bypass = continuous_bypass
        self.bypass_scale = bypass_scale
        self.use_flash_attention = use_flash_attention
        self.use_residual_attention = use_residual_attention
        self.latest_clause_metrics: Dict[str, torch.Tensor] = {}
        self.relative_position_type = relative_position_type.lower()
        if self.relative_position_type not in {"none", "learned", "rotary"}:
            raise ValueError(f"Unknown relative positional encoding '{relative_position_type}'.")
        self._pending_regularization: Optional[torch.Tensor] = None
        schedule = drop_path_schedule.lower()

        self.component_dims: Dict[str, int] = {}
        self.component_order: List[str] = []
        self.diagnostic_heads: nn.ModuleDict = nn.ModuleDict()
        self.attn_head_heads: nn.ModuleDict = nn.ModuleDict()

        if self.architecture not in {"vit", "swin"}:
            raise ValueError("architecture must be 'vit' or 'swin'.")

        if self.architecture == "vit":
            self.patch_embed = PatchEmbed(in_channels, int(embed_dim), patch_size, flatten=True)
            grid_h = image_size[0] // patch_size
            grid_w = image_size[1] // patch_size
            num_patches = grid_h * grid_w
            grid_size = (grid_h, grid_w)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, int(embed_dim))) if use_cls_token else None
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + (1 if self.cls_token is not None else 0), int(embed_dim)))
            self.pos_drop = nn.Dropout(drop_rate)
            depth = depths if isinstance(depths, int) else sum(depths)
            head = num_heads if isinstance(num_heads, int) else num_heads[0]
            mlp = mlp_ratio if isinstance(mlp_ratio, (int, float)) else mlp_ratio[0]
            clause_list = _expand_vit_clauses(tm_clauses, depth)
            if schedule == "cosine":
                steps = torch.linspace(0.0, 1.0, depth)
                drop_path = (drop_path_rate * 0.5 * (1 - torch.cos(torch.pi * steps))).tolist()
            else:
                drop_path = torch.linspace(0, drop_path_rate, depth).tolist()
            self.blocks = nn.ModuleList(
                [
                    TMEncoderBlock(
                        int(embed_dim),
                        head,
                        mlp_ratio=mlp,
                        backend=self.backend,
                        n_clauses=int(clause_list[i]),
                        tm_tau=tm_tau,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path[i],
                        gate_type=self.ff_gate,
                        gate_activation=self.ff_gate_activation,
                        norm_type=self.norm_type,
                        layerscale_init=self.layerscale_init,
                        enable_layerscale=self.use_layerscale,
                        clause_dropout=self.clause_dropout,
                        literal_dropout=self.literal_dropout,
                        sparsity_weight=self.ff_sparsity_weight,
                        clause_bias_init=self.clause_bias_init,
                        mix_type=self.ff_mix_type,
                        bitwise_mix=self.ff_bitwise_mix,
                        learnable_tau=self.learnable_tau,
                        tau_min=self.tau_min,
                        tau_max=self.tau_max,
                        tau_ema_beta=self.tau_ema_beta,
                        clause_attention=self.clause_attention,
                        clause_routing=self.clause_routing,
                        continuous_bypass=self.continuous_bypass,
                        bypass_scale=self.bypass_scale,
                        use_flash_attention=self.use_flash_attention,
                        use_residual_attention=self.use_residual_attention,
                        grid_size=grid_size,
                        include_cls_token=self.cls_token is not None,
                        relative_position_type=self.relative_position_type,
                    )
                    for i in range(depth)
                ]
            )
            self.norm = nn.LayerNorm(int(embed_dim))
            self.head = nn.Linear(int(embed_dim), num_classes)
            _init_linear(self.head)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            if self.cls_token is not None:
                nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.component_order = ["patch_embed"] + [f"block_{i + 1}" for i in range(depth)] + ["pre_head"]
            for name in self.component_order:
                self.component_dims[name] = int(embed_dim)
            head_dim = int(embed_dim) // head
            for block_idx in range(depth):
                for head_idx in range(head):
                    key = f"block_{block_idx + 1}_head{head_idx + 1}"
                    layer = nn.Linear(head_dim, num_classes)
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                    self.attn_head_heads[key] = layer
        else:
            if isinstance(depths, int):
                raise ValueError("depths must be a sequence for Swin architecture.")
            stage_depths = tuple(depths)
            stage_heads = _to_sequence(num_heads, len(stage_depths))
            if isinstance(embed_dim, Sequence):
                stage_dims = tuple(embed_dim)
            else:
                stage_dims = tuple(int(embed_dim) * (2 ** i) for i in range(len(stage_depths)))
            stage_mlp = _to_sequence(mlp_ratio, len(stage_depths))
            stage_clauses = _to_sequence(tm_clauses, len(stage_depths))
            total_blocks = sum(stage_depths)
            if schedule == "cosine":
                steps = torch.linspace(0.0, 1.0, total_blocks)
                drop_path = (drop_path_rate * 0.5 * (1 - torch.cos(torch.pi * steps))).tolist()
            else:
                drop_path = torch.linspace(0, drop_path_rate, total_blocks).tolist()
            self.patch_embed = PatchEmbed(in_channels, stage_dims[0], patch_size, flatten=False)
            stages = []
            dp_offset = 0
            for idx, depth in enumerate(stage_depths):
                stage = TMSwinStage(
                    dim=stage_dims[idx],
                    depth=depth,
                    num_heads=stage_heads[idx],
                    window_size=window_size,
                    backend=self.backend,
                    mlp_ratio=stage_mlp[idx],
                    n_clauses=stage_clauses[idx],
                    tm_tau=tm_tau,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path[dp_offset : dp_offset + depth],
                    downsample=(idx < len(stage_depths) - 1),
                    use_checkpoint=grad_checkpoint,
                    gate_type=self.ff_gate,
                    gate_activation=self.ff_gate_activation,
                    norm_type=self.norm_type,
                    layerscale_init=self.layerscale_init,
                    enable_layerscale=self.use_layerscale,
                    clause_dropout=self.clause_dropout,
                    literal_dropout=self.literal_dropout,
                    sparsity_weight=self.ff_sparsity_weight,
                    clause_bias_init=self.clause_bias_init,
                    mix_type=self.ff_mix_type,
                    bitwise_mix=self.ff_bitwise_mix,
                    learnable_tau=self.learnable_tau,
                    tau_min=self.tau_min,
                    tau_max=self.tau_max,
                    tau_ema_beta=self.tau_ema_beta,
                    clause_attention=self.clause_attention,
                    clause_routing=self.clause_routing,
                    continuous_bypass=self.continuous_bypass,
                    bypass_scale=self.bypass_scale,
                    use_flash_attention=self.use_flash_attention,
                    use_residual_attention=self.use_residual_attention,
                    relative_position_type=self.relative_position_type,
                )
                stages.append(stage)
                dp_offset += depth
            self.stages = nn.ModuleList(stages)
            self.norm = nn.LayerNorm(stage_dims[-1])
            self.head = nn.Linear(stage_dims[-1], num_classes)
            component_names: List[str] = ["patch_embed"]
            self.component_dims["patch_embed"] = stage_dims[0]
            for idx, depth in enumerate(stage_depths):
                dim = stage_dims[idx]
                for block_idx in range(depth):
                    name = f"stage{idx + 1}_block{block_idx + 1}"
                    component_names.append(name)
                    self.component_dims[name] = dim
                stage_out_name = f"stage{idx + 1}_out"
                component_names.append(stage_out_name)
                self.component_dims[stage_out_name] = dim
            component_names.append("pre_head")
            self.component_dims["pre_head"] = stage_dims[-1]
            self.component_order = component_names
            for idx, depth in enumerate(stage_depths):
                head_dim = stage_dims[idx] // stage_heads[idx]
                for block_idx in range(depth):
                    for head_idx in range(stage_heads[idx]):
                        key = f"stage{idx + 1}_block{block_idx + 1}_head{head_idx + 1}"
                        layer = nn.Linear(head_dim, num_classes)
                        nn.init.trunc_normal_(layer.weight, std=0.02)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                        self.attn_head_heads[key] = layer

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

        for name in self.component_order:
            dim = self.component_dims[name]
            head = nn.Linear(dim, num_classes)
            nn.init.trunc_normal_(head.weight, std=0.02)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
            self.diagnostic_heads[name] = head

    def _pool_vit_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Pool tokens for ViT architecture."""
        if self.pool == "cls":
            return x[:, 0]
        return x.mean(dim=1)

    @staticmethod
    def _pool_swin_tokens(tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)

    def _pool_attention_heads(self, head_tokens: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "vit" and self.pool == "cls" and head_tokens.size(2) > 0:
            return head_tokens[:, :, 0, :]
        return head_tokens.mean(dim=2)

    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
        *,
        collect_diagnostics: bool = False,
        return_features: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor],
    ]:
        if x.dim() != 4:
            raise ValueError("UnifiedTMTransformer expects image tensor of shape (B, C, H, W).")

        diagnostics: Dict[str, torch.Tensor] = {}
        collecting = collect_diagnostics
        reg_loss: Optional[torch.Tensor] = None
        clause_metrics: Dict[str, torch.Tensor] = {}

        if self.architecture == "vit":
            x = self.patch_embed(x)
            if hasattr(self, "cls_token") and self.cls_token is not None:
                cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed[:, : x.size(1), :]
            x = self.pos_drop(x)
            if collecting and "patch_embed" in self.diagnostic_heads:
                diagnostics["patch_embed"] = self.diagnostic_heads["patch_embed"](self._pool_vit_tokens(x))
            attn_residual = None
            for idx, block in enumerate(self.blocks):
                component_key = f"block_{idx + 1}"
                use_ckpt = self.grad_checkpoint and self.training and x.requires_grad and not collecting
                if use_ckpt:
                    residual = attn_residual if self.use_residual_attention else None
                    def _block_forward(inp: torch.Tensor, blk: TMEncoderBlock = block) -> torch.Tensor:
                        return blk(
                            inp,
                            use_ste=use_ste,
                            collect_diagnostics=False,
                            attn_residual=residual,
                        )
                    x = checkpoint(_block_forward, x)
                    head_context = None
                else:
                    result = block(
                        x,
                        use_ste=use_ste,
                        collect_diagnostics=collecting,
                        attn_residual=attn_residual if self.use_residual_attention else None,
                    )
                    if collecting:
                        x, head_context = result  # type: ignore[misc]
                    else:
                        x = result  # type: ignore[assignment]
                        head_context = None
                if self.use_residual_attention:
                    attn_residual = block.last_attention
                if collecting and component_key in self.diagnostic_heads:
                    diagnostics[component_key] = self.diagnostic_heads[component_key](self._pool_vit_tokens(x))
                    if head_context is not None:
                        head_feats = self._pool_attention_heads(head_context, mode="vit")
                        for head_idx in range(head_feats.shape[1]):
                            head_key = f"{component_key}_head{head_idx + 1}"
                            if head_key in self.attn_head_heads:
                                diagnostics[head_key] = self.attn_head_heads[head_key](head_feats[:, head_idx, :])
                if collecting:
                    clause_out = getattr(block.ffn, "last_clause_outputs", None)
                    if clause_out is not None:
                        clause_mean = clause_out.abs().mean(dim=(0, 1)).detach()
                        clause_metrics[f"{component_key}_clause_mean"] = clause_mean.cpu()
                        head_chunks = torch.chunk(clause_mean, block.num_heads)
                        head_means = torch.tensor(
                            [chunk.mean().item() for chunk in head_chunks],
                            dtype=clause_mean.dtype,
                        )
                        clause_metrics[f"{component_key}_head_mean"] = head_means.cpu()
                penalty = getattr(block.ffn, "sparsity_penalty", None)
                if penalty is not None:
                    reg_loss = penalty if reg_loss is None else reg_loss + penalty
            x = self.norm(x)
            if collecting and "pre_head" in self.diagnostic_heads:
                diagnostics["pre_head"] = self.diagnostic_heads["pre_head"](self._pool_vit_tokens(x))
            feats = self._pool_vit_tokens(x)
            logits = self.head(feats)
            self._pending_regularization = reg_loss
            if collecting:
                self.latest_clause_metrics = clause_metrics
            else:
                self.latest_clause_metrics = {}
            if collecting:
                diagnostics["final_decision"] = logits
                if return_features:
                    return logits, diagnostics, feats
                return logits, diagnostics
            if return_features:
                return logits, feats
            return logits

        tokens = self.patch_embed(x)
        B, H, W, C = tokens.shape
        tokens = tokens.reshape(B, H * W, C)
        if collecting and "patch_embed" in self.diagnostic_heads:
            diagnostics["patch_embed"] = self.diagnostic_heads["patch_embed"](self._pool_swin_tokens(tokens))

        for stage_idx, stage in enumerate(self.stages):
            if collecting:
                def stage_record(name: str, comps: torch.Tensor, h: int, w: int, head_tokens: Optional[torch.Tensor]) -> None:
                    if name in self.diagnostic_heads:
                        diagnostics[name] = self.diagnostic_heads[name](self._pool_swin_tokens(comps))
                    if head_tokens is not None:
                        head_feats = self._pool_attention_heads(head_tokens, mode="swin")
                        for head_idx in range(head_feats.shape[1]):
                            head_key = f"{name}_head{head_idx + 1}"
                            if head_key in self.attn_head_heads:
                                diagnostics[head_key] = self.attn_head_heads[head_key](head_feats[:, head_idx, :])
            else:
                stage_record = None
            tokens, H, W = stage(
                tokens,
                H,
                W,
                use_ste=use_ste,
                collect_diagnostics=collecting,
                record_callback=stage_record,
                stage_index=stage_idx,
            )
            if collecting:
                for block_idx, block in enumerate(stage.blocks):
                    clause_out = getattr(block.ffn, "last_clause_outputs", None)
                    if clause_out is not None:
                        clause_mean = clause_out.abs().mean(dim=(0, 1)).detach()
                        key = f"stage{stage_idx + 1}_block{block_idx + 1}"
                        clause_metrics[key + "_clause_mean"] = clause_mean.cpu()
                        head_chunks = torch.chunk(clause_mean, block.attn.attn.num_heads)
                        head_means = torch.tensor(
                            [chunk.mean().item() for chunk in head_chunks],
                            dtype=clause_mean.dtype,
                        )
                        clause_metrics[key + "_head_mean"] = head_means.cpu()
            for block in stage.blocks:
                penalty = getattr(block.ffn, "sparsity_penalty", None)
                if penalty is not None:
                    reg_loss = penalty if reg_loss is None else reg_loss + penalty

        x = self.norm(tokens)
        if collecting and "pre_head" in self.diagnostic_heads:
            diagnostics["pre_head"] = self.diagnostic_heads["pre_head"](self._pool_swin_tokens(x))
        feats = self._pool_swin_tokens(x)
        logits = self.head(feats)
        self._pending_regularization = reg_loss
        if collecting:
            self.latest_clause_metrics = clause_metrics
        else:
            self.latest_clause_metrics = {}
        if collecting:
            diagnostics["final_decision"] = logits
            if return_features:
                return logits, diagnostics, feats
            return logits, diagnostics
        if return_features:
            return logits, feats
        return logits

    def consume_clause_metrics(self) -> Dict[str, torch.Tensor]:
        metrics = {key: value.clone() for key, value in self.latest_clause_metrics.items()}
        self.latest_clause_metrics = {}
        return metrics

    def apply_clause_head_specialization(
        self,
        clause_metrics: Dict[str, List[float]],
        smoothing: float = 0.5,
    ) -> None:
        smoothing = float(max(0.0, min(1.0, smoothing)))
        if self.architecture == "vit":
            for idx, block in enumerate(self.blocks):
                key = f"block_{idx + 1}_head_mean"
                if key not in clause_metrics:
                    continue
                values = torch.tensor(
                    clause_metrics[key],
                    dtype=block.attn.head_gains.dtype,
                    device=block.attn.head_gains.device,
                )
                if values.numel() != block.attn.num_heads:
                    continue
                norm = values / (values.mean() + 1e-6)
                gains = (1.0 - smoothing) + smoothing * norm
                block.attn.set_head_gains(gains)
        else:
            for stage_idx, stage in enumerate(self.stages):
                for block_idx, block in enumerate(stage.blocks):
                    key = f"stage{stage_idx + 1}_block{block_idx + 1}_head_mean"
                    if key not in clause_metrics:
                        continue
                    values = torch.tensor(
                        clause_metrics[key],
                        dtype=block.attn.attn.head_gains.dtype,
                        device=block.attn.attn.head_gains.device,
                    )
                    if values.numel() != block.attn.attn.num_heads:
                        continue
                    norm = values / (values.mean() + 1e-6)
                    gains = (1.0 - smoothing) + smoothing * norm
                    block.attn.set_head_gains(gains)

    def layer_param_groups(self) -> List[List[nn.Parameter]]:
        groups: List[List[nn.Parameter]] = []
        seen: Set[int] = set()

        def add_params(params) -> None:
            bucket: List[nn.Parameter] = []
            if params is None:
                return
            for p in params:
                if p is None:
                    continue
                if not isinstance(p, nn.Parameter):
                    continue
                if id(p) in seen:
                    continue
                seen.add(id(p))
                bucket.append(p)
            if bucket:
                groups.append(bucket)

        if self.architecture == "vit":
            add_params(self.patch_embed.parameters())
            add_params([self.pos_embed])
            if self.cls_token is not None:
                add_params([self.cls_token])
            for block in self.blocks:
                add_params(block.parameters())
            add_params(self.norm.parameters())
            add_params(self.head.parameters())
        else:
            add_params(self.patch_embed.parameters())
            for stage in self.stages:
                for block in stage.blocks:
                    add_params(block.parameters())
                if stage.downsample is not None:
                    add_params(stage.downsample.parameters())
            add_params(self.norm.parameters())
            add_params(self.head.parameters())

        add_params(self.diagnostic_heads.parameters())
        add_params(self.attn_head_heads.parameters())

        remaining = [p for p in self.parameters() if id(p) not in seen]
        if remaining:
            groups.append(remaining)
        return groups

    def pop_regularization_loss(self) -> Optional[torch.Tensor]:
        reg = self._pending_regularization
        self._pending_regularization = None
        return reg


class TM_TransformerBlock(TMEncoderBlock):
    """Backwards-compatible alias of the legacy token block."""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int = 512, dropout: float = 0.1, n_clauses: int = 200):
        super().__init__(
            dim=d_model,
            num_heads=n_heads,
            mlp_ratio=dim_feedforward / d_model,
            backend="ste",
            n_clauses=n_clauses,
            tm_tau=0.5,
            drop=dropout,
            attn_drop=dropout,
            drop_path=0.0,
        )

    def forward(self, x: torch.Tensor, use_ste: bool = True) -> torch.Tensor:
        return super().forward(x, use_ste=use_ste)


