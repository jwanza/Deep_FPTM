from typing import Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tm import FuzzyPatternTM_STE, FuzzyPatternTM_STCM
from .conv_tm import ConvSTE2d, ConvSTCM2d


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
        # Diagnostic heads for per-block classification (for logging/monitoring)
        self.diag_heads = nn.ModuleList([nn.Linear(c, self.num_classes) for c in self.channels])
        self.aux_weight = float(aux_weight)

        # Head classifier over pooled features
        classifier_cls = FuzzyPatternTM_STE if (layer_cls is FuzzyPatternTM_STE or layer_cls == FuzzyPatternTM_STE) else FuzzyPatternTM_STCM
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
                dict(operator=stcm_operator, ternary_voting=stcm_ternary_voting, ternary_band=stcm_ternary_band, ste_temperature=stcm_ste_temperature)
            )
        self.classifier = classifier_cls(**head_kwargs)

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
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        logits, clauses = self.classifier(x, use_ste=use_ste)
        if collect_diagnostics:
            return logits, diagnostics
        return logits, clauses

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


