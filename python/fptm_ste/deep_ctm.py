import math
from typing import Dict, Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .tm import FuzzyPatternTM_STE, FuzzyPatternTM_STCM
from .conv_tm import ConvSTE2d, ConvSTCM2d
from .deep_tm import DeepTMNetwork


def _softplus_inverse(value: float) -> float:
    value = max(1e-6, float(value))
    return math.log(math.exp(value) - 1.0)


class DeepCTMNetwork(nn.Module):
    """
    Deep convolutional TM network.

    Stacks ConvTM2d blocks with residual connections and normalization.
    Supports two backends:
      - STE (layer_cls=FuzzyPatternTM_STE), conv_core_backend in {"tm","deeptm"}
      - STCM (layer_cls=FuzzyPatternTM_STCM), conv_core_backend in {"stcm","deepstcm"}
    """

    def __init__(
        self,
        *,
        in_channels: int,
        image_size: Tuple[int, int],
        num_classes: int,
        channels: Sequence[int],
        kernels: Sequence[int],
        strides: Sequence[int],
        pools: Sequence[int],
        clauses_per_block: Sequence[int],
        head_clauses: int = 512,
        tau: float = 0.5,
        dropout: float = 0.1,
        conv_core_backend: str = "tm",
        core_hidden_dims: Optional[Sequence[int]] = None,
        clause_dropout: float = 0.0,
        literal_dropout: float = 0.0,
        clause_bias_init: float = 0.0,
        aux_weight: float = 0.0,
        mix_type: str = "none",
        use_stem: bool = False,
        stem_channels: Optional[int] = None,
        head_type: str = "auto",
        head_hidden_dims: Optional[Sequence[int]] = None,
        head_linear: bool = False,
        head_linear_hidden: Optional[int] = None,
        head_linear_dropout: float = 0.0,
        head_attention: bool = False,
        head_attention_dim: Optional[int] = None,
        head_attention_heads: int = 4,
        head_attention_dropout: float = 0.0,
        head_mix_init_tm: float = 1.0,
        head_mix_init_linear: float = 0.2,
        head_mix_init_attention: float = 0.2,
        # STCM specific
        stcm_operator: str = "capacity",
        stcm_ternary_voting: bool = False,
        stcm_ternary_band: float = 0.05,
        stcm_ste_temperature: float = 1.0,
        # Select STE vs STCM
        layer_cls: Type[nn.Module] = FuzzyPatternTM_STE,
    ):
        super().__init__()
        assert len(channels) == len(kernels) == len(strides) == len(pools) == len(clauses_per_block), "Per-block configs must be same length."
        self.in_channels = int(in_channels)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.num_classes = int(num_classes)
        self.channels = list(int(c) for c in channels)
        self.kernels = list(int(k) for k in kernels)
        self.strides = list(int(s) for s in strides)
        self.pools = list(int(p) for p in pools)
        self.clauses_per_block = list(int(c) for c in clauses_per_block)
        self.head_clauses = int(head_clauses)
        self.tau = float(tau)
        self.dropout_p = float(dropout)
        self.conv_core_backend = (conv_core_backend or "tm").lower()
        self.core_hidden_dims = list(core_hidden_dims) if core_hidden_dims is not None else None
        self.layer_cls = layer_cls
        self.use_stem = bool(use_stem)
        self.stem_channels = int(stem_channels) if stem_channels is not None else (int(channels[0]) if channels else in_channels)
        self.mix_type = (mix_type or "none").lower()
        resolved_head_type = head_type.lower() if head_type else "auto"
        if resolved_head_type == "auto":
            resolved_head_type = "stcm" if (layer_cls is FuzzyPatternTM_STCM or layer_cls == FuzzyPatternTM_STCM) else "tm"
        self.head_type = resolved_head_type
        self.head_hidden_dims = list(int(h) for h in head_hidden_dims) if head_hidden_dims is not None else None
        self.head_linear_enabled = bool(head_linear)
        self.head_linear_hidden = int(head_linear_hidden) if head_linear_hidden is not None else None
        self.head_linear_dropout = float(head_linear_dropout)
        self.head_attention_enabled = bool(head_attention)
        self.head_attention_dim = int(head_attention_dim) if head_attention_dim is not None else None
        self.head_attention_heads = int(max(1, head_attention_heads))
        self.head_attention_dropout = float(head_attention_dropout)
        self.head_mix_inits = dict(
            tm=float(head_mix_init_tm),
            linear=float(head_mix_init_linear),
            attention=float(head_mix_init_attention),
        )

        blocks: nn.ModuleList = nn.ModuleList()
        norms: nn.ModuleList = nn.ModuleList()
        proj: nn.ModuleList = nn.ModuleList()
        pools_modules: nn.ModuleList = nn.ModuleList()
        mix_modules: nn.ModuleList = nn.ModuleList()

        C_in = self.in_channels
        H, W = self.image_size
        if self.use_stem:
            stem_c = self.stem_channels
            self.stem = nn.Sequential(
                nn.Conv2d(C_in, stem_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(stem_c),
                nn.GELU(),
                nn.Conv2d(stem_c, stem_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(stem_c),
                nn.GELU(),
            )
            C_in = stem_c
        else:
            self.stem = None
        for idx, (C_out, K, S, P, nC) in enumerate(zip(self.channels, self.kernels, self.strides, self.pools, self.clauses_per_block)):
            if layer_cls is FuzzyPatternTM_STE or layer_cls == FuzzyPatternTM_STE:
                conv = ConvSTE2d(
                    C_in,
                    C_out,
                    kernel_size=K,
                    stride=S,
                    padding=K // 2 if K > 1 else 0,
                    n_clauses=nC,
                    tau=self.tau,
                    core_backend=self.conv_core_backend if self.conv_core_backend in {"tm", "deeptm"} else "tm",
                    core_hidden_dims=self.core_hidden_dims,
                    clause_dropout=clause_dropout,
                    literal_dropout=literal_dropout,
                    clause_bias_init=clause_bias_init,
                )
            else:
                conv = ConvSTCM2d(
                    C_in,
                    C_out,
                    kernel_size=K,
                    stride=S,
                    padding=K // 2 if K > 1 else 0,
                    n_clauses=nC,
                    tau=self.tau,
                    core_backend=self.conv_core_backend if self.conv_core_backend in {"stcm", "deepstcm"} else "stcm",
                    core_hidden_dims=self.core_hidden_dims,
                    clause_dropout=clause_dropout,
                    literal_dropout=literal_dropout,
                    clause_bias_init=clause_bias_init,
                    operator=stcm_operator,
                    ternary_voting=stcm_ternary_voting,
                    ternary_band=stcm_ternary_band,
                    ste_temperature=stcm_ste_temperature,
                )
            blocks.append(conv)
            norms.append(nn.BatchNorm2d(C_out))
            proj.append(nn.Conv2d(C_in, C_out, kernel_size=1, stride=S, bias=False) if (C_in != C_out or S != 1) else nn.Identity())
            pools_modules.append(nn.AvgPool2d(kernel_size=P, stride=P) if P and P > 1 else nn.Identity())
            if self.mix_type != "none":
                mix_modules.append(self._build_mixer_module(C_out))

            # update spatial dims and channels
            H = (H + (K // 2) * 2 - K) // S + 1
            W = (W + (K // 2) * 2 - K) // S + 1
            if P and P > 1:
                H = H // P
                W = W // P
            C_in = C_out

        self.blocks = blocks
        self.norms = norms
        self.projs = proj
        self.pools = pools_modules
        self.mixers = mix_modules if self.mix_type != "none" else nn.ModuleList()
        self.dropout = nn.Dropout2d(self.dropout_p)
        self.final_channels = C_in
        self.final_hw = (max(1, H), max(1, W))
        # Diagnostic heads for per-block classification (for logging/monitoring)
        self.diag_heads = nn.ModuleList([nn.Linear(c, self.num_classes) for c in self.channels])
        self.aux_weight = float(aux_weight)

        # Head classifier setup
        self.tm_head_module: Optional[nn.Module] = None
        self.tm_head_returns_clauses = False
        self.linear_head_module: Optional[nn.Module] = self._build_linear_head(
            hidden_dim=self.head_linear_hidden, dropout=self.head_linear_dropout
        ) if self.head_linear_enabled else None
        attn_modules = self._build_attention_head(
            attention_dim=self.head_attention_dim,
            num_heads=self.head_attention_heads,
            dropout=self.head_attention_dropout,
        ) if self.head_attention_enabled else None
        if attn_modules is not None:
            (
                self.attn_proj_in,
                self.attn_cls_token,
                self.attn_module,
                self.attn_norm,
                self.attn_mlp,
                self.attn_out_proj,
            ) = attn_modules
        else:
            self.attn_proj_in = None
            self.attn_cls_token = None
            self.attn_module = None
            self.attn_norm = None
            self.attn_mlp = None
            self.attn_out_proj = None
        self.tm_head_module, self.tm_head_returns_clauses = self._build_tm_head(
            head_type=self.head_type,
            clause_dropout=clause_dropout,
            literal_dropout=literal_dropout,
            clause_bias_init=clause_bias_init,
            stcm_operator=stcm_operator,
            stcm_ternary_voting=stcm_ternary_voting,
            stcm_ternary_band=stcm_ternary_band,
            stcm_ste_temperature=stcm_ste_temperature,
        )
        self.head_mix_params = nn.ParameterDict()
        self._latest_head_mix: Dict[str, float] = {}
        if self.tm_head_module is not None:
            self.head_mix_params["tm"] = nn.Parameter(
                torch.tensor(_softplus_inverse(max(1e-3, self.head_mix_inits["tm"]))), requires_grad=True
            )
        if self.linear_head_module is not None:
            self.head_mix_params["linear"] = nn.Parameter(
                torch.tensor(_softplus_inverse(max(1e-3, self.head_mix_inits["linear"]))), requires_grad=True
            )
        if self.attn_module is not None:
            self.head_mix_params["attention"] = nn.Parameter(
                torch.tensor(_softplus_inverse(max(1e-3, self.head_mix_inits["attention"]))), requires_grad=True
            )
        self.classifier = self._resolve_classifier_export()

    def forward(self, x: torch.Tensor, use_ste: bool = True, collect_diagnostics: bool = False):
        # x: [B, C, H, W]
        if self.stem is not None:
            x = self.stem(x)
        diagnostics = {} if collect_diagnostics else None
        for block_idx in range(len(self.blocks)):
            conv = self.blocks[block_idx]
            norm = self.norms[block_idx]
            proj = self.projs[block_idx]
            pool = self.pools[block_idx]
            mixer = self.mixers[block_idx] if (self.mix_type != "none" and block_idx < len(self.mixers)) else None
            identity = proj(x)
            logits = conv(x, use_ste=use_ste)
            x = norm(self.dropout(torch.sigmoid(logits)) + identity)
            if mixer is not None:
                x = x + mixer(x)
            if collect_diagnostics:
                pooled = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # [B, C_i]
                diag = self.diag_heads[block_idx](pooled)  # [B, num_classes]
                diagnostics[f"block_{block_idx + 1}"] = diag
            x = pool(x)
        # Global average pool to [B, C]
        feature_map = x
        pooled = F.adaptive_avg_pool2d(feature_map, 1).squeeze(-1).squeeze(-1)
        head_logits: Dict[str, torch.Tensor] = {}
        clauses = None
        if self.tm_head_module is not None:
            tm_logits, clauses = self.tm_head_module(pooled, use_ste=use_ste) if self.tm_head_returns_clauses else (self.tm_head_module(pooled), None)
            head_logits["tm"] = tm_logits
        if self.linear_head_module is not None:
            head_logits["linear"] = self.linear_head_module(pooled)
        if self.attn_module is not None and self.attn_proj_in is not None and self.attn_out_proj is not None:
            head_logits["attention"] = self._forward_attention_head(feature_map)
        total_logits = None
        self._latest_head_mix = {}
        for name, logits_val in head_logits.items():
            if name in self.head_mix_params:
                weight = F.softplus(self.head_mix_params[name])
            else:
                weight = logits_val.new_tensor(1.0)
            self._latest_head_mix[name] = float(weight.detach().cpu())
            weighted = logits_val * weight
            total_logits = weighted if total_logits is None else total_logits + weighted
        if total_logits is None:
            raise RuntimeError("DeepCTMNetwork has no active head outputs.")
        if collect_diagnostics:
            for name, logits_val in head_logits.items():
                diagnostics[f"head_{name}"] = logits_val
        if collect_diagnostics:
            return total_logits, diagnostics
        return total_logits, clauses

    def set_tau(self, tau: float) -> None:
        self.tau = float(tau)
        for module in self.modules():
            if hasattr(module, "tau"):
                try:
                    setattr(module, "tau", self.tau)
                except Exception:
                    pass

    def set_stcm_band(self, band: float) -> None:
        value = float(band)
        for module in self.modules():
            if hasattr(module, "ternary_band"):
                try:
                    module.ternary_band = value
                except Exception:
                    continue

    def set_stcm_temperature(self, temp: float) -> None:
        value = float(temp)
        for module in self.modules():
            if hasattr(module, "ste_temperature"):
                try:
                    module.ste_temperature = value
                except Exception:
                    continue

    def _build_mixer_module(self, channels: int) -> nn.Module:
        if self.mix_type == "linear":
            return nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            )
        if self.mix_type == "depthwise":
            return nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            )
        if self.mix_type == "linear_depthwise":
            return nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            )
        raise ValueError(f"Unsupported mix_type '{self.mix_type}'")

    def _build_tm_head(
        self,
        *,
        head_type: str,
        clause_dropout: float,
        literal_dropout: float,
        clause_bias_init: float,
        stcm_operator: str,
        stcm_ternary_voting: bool,
        stcm_ternary_band: float,
        stcm_ste_temperature: float,
    ) -> Tuple[Optional[nn.Module], bool]:
        head_type = head_type.lower()
        if head_type == "none":
            return None, False
        if head_type in {"tm", "stcm"}:
            classifier_cls = FuzzyPatternTM_STE if head_type == "tm" else FuzzyPatternTM_STCM
            head_kwargs = dict(
                n_features=self.final_channels,
                n_clauses=max(2, self.head_clauses),
                n_classes=self.num_classes,
                tau=self.tau,
                clause_dropout=clause_dropout,
                literal_dropout=literal_dropout,
                clause_bias_init=clause_bias_init,
            )
            if classifier_cls is FuzzyPatternTM_STCM:
                head_kwargs.update(
                    dict(
                        operator=stcm_operator,
                        ternary_voting=stcm_ternary_voting,
                        ternary_band=stcm_ternary_band,
                        ste_temperature=stcm_ste_temperature,
                    )
                )
            return classifier_cls(**head_kwargs), True
        if head_type in {"deeptm", "deepstcm"}:
            layer_cls = FuzzyPatternTM_STE if head_type == "deeptm" else FuzzyPatternTM_STCM
            hidden_dims = self.head_hidden_dims or [max(self.final_channels, self.num_classes * 2)]
            layer_extra_kwargs: Dict[str, float] = {}
            layer_operator = None
            layer_ternary_voting = None
            if layer_cls is FuzzyPatternTM_STCM:
                layer_operator = stcm_operator
                layer_ternary_voting = stcm_ternary_voting
                layer_extra_kwargs = dict(ternary_band=stcm_ternary_band, ste_temperature=stcm_ste_temperature)
            head = DeepTMNetwork(
                input_dim=self.final_channels,
                hidden_dims=hidden_dims,
                n_classes=self.num_classes,
                n_clauses=max(2, self.head_clauses),
                dropout=self.dropout_p,
                tau=self.tau,
                clause_dropout=clause_dropout,
                literal_dropout=literal_dropout,
                clause_bias_init=clause_bias_init,
                layer_cls=layer_cls,
                layer_operator=layer_operator,
                layer_ternary_voting=layer_ternary_voting,
                layer_extra_kwargs=layer_extra_kwargs if layer_extra_kwargs else None,
            )
            return head, True
        raise ValueError(f"Unsupported head_type '{head_type}'")

    def _build_linear_head(self, hidden_dim: Optional[int], dropout: float) -> Optional[nn.Module]:
        hidden = hidden_dim if hidden_dim is not None else max(self.final_channels, self.num_classes * 2)
        layers = [
            nn.LayerNorm(self.final_channels),
            nn.Linear(self.final_channels, hidden),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, self.num_classes))
        return nn.Sequential(*layers)

    def _build_attention_head(
        self,
        attention_dim: Optional[int],
        num_heads: int,
        dropout: float,
    ) -> Optional[Tuple[nn.Module, nn.Parameter, nn.Module, nn.Module, nn.Module, nn.Module]]:
        attn_dim = attention_dim if attention_dim is not None else self.final_channels
        proj_in = nn.Linear(self.final_channels, attn_dim) if attn_dim != self.final_channels else nn.Identity()
        cls_token = nn.Parameter(torch.zeros(1, 1, attn_dim))
        init.trunc_normal_(cls_token, std=0.02)
        attn = nn.MultiheadAttention(attn_dim, num_heads, dropout=dropout, batch_first=True)
        norm = nn.LayerNorm(attn_dim)
        mlp = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim * 2, attn_dim),
        )
        out_proj = nn.Linear(attn_dim, self.num_classes)
        return proj_in, cls_token, attn, norm, mlp, out_proj

    def _forward_attention_head(self, feature_map: torch.Tensor) -> torch.Tensor:
        if self.attn_module is None or self.attn_proj_in is None or self.attn_cls_token is None or self.attn_out_proj is None:
            raise RuntimeError("Attention head requested but not initialized.")
        B = feature_map.shape[0]
        tokens = feature_map.flatten(2).transpose(1, 2)  # [B, HW, C]
        tokens = self.attn_proj_in(tokens)
        cls_tokens = self.attn_cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        attn_out, _ = self.attn_module(tokens, tokens, tokens, need_weights=False)
        cls_out = self.attn_norm(attn_out[:, 0])
        cls_out = cls_out + self.attn_mlp(cls_out)
        return self.attn_out_proj(cls_out)

    def head_mix_summary(self) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        for name, param in self.head_mix_params.items():
            summary[name] = float(F.softplus(param).detach().cpu())
        return summary

    def _resolve_classifier_export(self) -> Optional[nn.Module]:
        if hasattr(self.tm_head_module, "discretize"):
            return self.tm_head_module
        if (
            self.tm_head_module is not None
            and hasattr(self.tm_head_module, "classifier")
            and hasattr(self.tm_head_module.classifier, "discretize")
        ):
            return self.tm_head_module.classifier
        return None


