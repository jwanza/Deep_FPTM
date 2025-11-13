from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:
    import timm
except ImportError:  # pragma: no cover - timm is an optional dependency
    timm = None

from .tm import FuzzyPatternTM_STE
from .binarizers import SwinDualBinarizer, CNNSingleBinarizer
from .swin_pyramid_tm import PatchMerging, window_partition, window_reverse
from .tm_positional import RelativePositionBias2D, RotaryEmbedding, apply_rotary_pos_emb


class InstrumentedMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        *,
        use_flash: bool = False,
        relative_position_bias: Optional[RelativePositionBias2D] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.use_flash = use_flash
        self.register_buffer("head_gains", torch.ones(num_heads), persistent=False)
        self.relative_position_bias = relative_position_bias
        self.rotary_emb = rotary_emb

    def set_head_gains(self, gains: torch.Tensor) -> None:
        if gains.numel() != self.num_heads:
            raise ValueError(f"Expected {self.num_heads} gains, received {gains.numel()}.")
        gains = gains.to(self.qkv.weight.device, dtype=self.head_gains.dtype)
        with torch.no_grad():
            self.head_gains.copy_(gains)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rotary_emb is not None:
            sin, cos = self.rotary_emb.get_sin_cos(T, device=x.device, dtype=x.dtype)
            q, k = apply_rotary_pos_emb(q, k, sin, cos)
        dropout_p = self.attn_dropout.p if self.training else 0.0
        can_flash = (
            self.use_flash
            and self.relative_position_bias is None
            and x.is_cuda
            and hasattr(F, "scaled_dot_product_attention")
        )
        if can_flash:
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=dropout_p,
                is_causal=False,
            )
            attn_weights = None
        else:
            attn_scores = torch.matmul(q * self.scale, k.transpose(-2, -1))
            if self.relative_position_bias is not None:
                bias = self.relative_position_bias(device=x.device, dtype=attn_scores.dtype)
                if bias.shape[-1] != attn_scores.shape[-1]:
                    raise ValueError(
                        f"Relative bias shape mismatch: expected {attn_scores.shape[-1]}, got {bias.shape[-1]}"
                    )
                attn_scores = attn_scores + bias.unsqueeze(0)
            attn_weights = attn_scores.softmax(dim=-1)
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)
            context = torch.matmul(attn_weights, v)
        context = context * self.head_gains.view(1, self.num_heads, 1, 1)
        context_merge = context.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(context_merge)
        return out, context, attn_weights if not can_flash else None


class _SwinBackbone(nn.Module):
    """Flexible Swin Transformer backbone with optional channel projection."""

    SUPPORTED_VARIANTS: Dict[str, Tuple[int, List[int], List[int], int]] = {
        # variant: (embed_dim, depths, num_heads, window_size)
        "tiny": (96, [2, 2, 6, 2], [3, 6, 12, 24], 7),
        "small": (96, [2, 2, 18, 2], [3, 6, 12, 24], 7),
        "base": (128, [2, 2, 18, 2], [4, 8, 16, 32], 7),
        "large": (192, [2, 2, 18, 2], [6, 12, 24, 48], 7),
    }

    def __init__(
        self,
        variant: str = "tiny",
        pretrained: bool = True,
        num_scales: int = 4,
        input_size: int = 224,
        output_channels: Optional[Sequence[int]] = None,
        freeze_stages: int = 0,
        use_checkpoint: bool = False,
        drop_path_rate: float = 0.2,
        use_timm: bool = True,
        version: str = "v1",
        timm_model_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported Swin variant '{variant}'. Supported: {list(self.SUPPORTED_VARIANTS)}")
        if num_scales < 1 or num_scales > 4:
            raise ValueError("num_scales must be between 1 and 4.")

        self.variant = variant
        self.version = version
        self.timm_model_name = timm_model_name
        self.num_scales = num_scales
        self.input_size = input_size
        self.use_timm = False

        if use_timm:
            self.use_timm = self._create_from_timm(
                variant=variant,
                pretrained=pretrained,
                input_size=input_size,
                version=version,
                timm_model_name=timm_model_name,
                drop_path=drop_path_rate,
            )

        if not self.use_timm:
            raise RuntimeError(
                "timm.backend unavailable. Install timm >=0.9.0 to use Swin backbones or disable use_timm."
            )

        if hasattr(self.swin, "feature_info") and self.swin.feature_info is not None:
            self.native_channels = [
                self.swin.feature_info.info[i]["num_chs"] for i in range(num_scales)
            ]
            self._feature_info = self.swin.feature_info
        else:  # pragma: no cover - timm always exposes feature_info, but keep fallback
            embed_dim, _, _, _ = self.SUPPORTED_VARIANTS[variant]
            self.native_channels = [embed_dim * (2 ** i) for i in range(num_scales)]
            self._feature_info = None

        if output_channels is not None:
            if len(output_channels) != num_scales:
                raise ValueError("output_channels length must match num_scales.")
            self.projections = nn.ModuleList(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
                    for in_ch, out_ch in zip(self.native_channels, output_channels)
                ]
            )
            self._output_channels = list(output_channels)
        else:
            self.projections = None
            self._output_channels = list(self.native_channels)

        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)

        if use_checkpoint:
            self._enable_gradient_checkpointing()

    def _create_from_timm(
        self,
        variant: str,
        pretrained: bool,
        input_size: int,
        version: str,
        timm_model_name: Optional[str],
        drop_path: float,
    ) -> bool:
        if timm is None:
            return False
        try:
            model_name = self._resolve_timm_model(variant, input_size, version, timm_model_name)
            try:
                self.swin = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=tuple(range(self.num_scales)),
                    img_size=input_size,
                    pretrained_cfg_overlay={"drop_path_rate": drop_path},
                )
            except TypeError:
                self.swin = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=tuple(range(self.num_scales)),
                    img_size=input_size,
                )
                if hasattr(self.swin, "drop_path_rate"):
                    self.swin.drop_path_rate = drop_path
            self.use_timm = True
            return True
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[SwinBackbone] Failed to create timm model ({exc}).")
            return False

    def _resolve_timm_model(
        self,
        variant: str,
        input_size: int,
        version: str,
        override: Optional[str],
    ) -> str:
        if override:
            return override
        if version not in {"v1", "v2", "s3"}:
            raise ValueError(f"Unknown Swin version '{version}'. Use 'v1', 'v2', or 's3'.")

        if version == "v1":
            if input_size >= 384:
                name_map = {
                    "tiny": "swin_tiny_patch4_window12_384",
                    "small": "swin_small_patch4_window12_384",
                    "base": "swin_base_patch4_window12_384",
                    "large": "swin_large_patch4_window12_384",
                }
            else:
                name_map = {
                    "tiny": "swin_tiny_patch4_window7_224",
                    "small": "swin_small_patch4_window7_224",
                    "base": "swin_base_patch4_window7_224",
                    "large": "swin_large_patch4_window7_224",
                }
        elif version == "v2":
            if input_size <= 256:
                name_map = {
                    "tiny": "swinv2_tiny_window8_256",
                    "small": "swinv2_small_window8_256",
                    "base": "swinv2_base_window8_256",
                    "large": "swinv2_large_window12_192",
                }
            else:
                name_map = {
                    "tiny": "swinv2_tiny_window16_256",
                    "small": "swinv2_small_window16_256",
                    "base": "swinv2_base_window12to16_192to256",
                    "large": "swinv2_large_window12to16_192to256",
                }
        else:  # s3
            name_map = {
                "tiny": "swin_s3_tiny_224",
                "small": "swin_s3_small_224",
                "base": "swin_s3_base_224",
                "large": "swin_s3_base_224",
            }
        return name_map[variant]

    def _freeze_stages(self, num_stages: int) -> None:
        if not self.use_timm:
            return

        if hasattr(self.swin, "patch_embed"):
            for param in self.swin.patch_embed.parameters():
                param.requires_grad = False

        for i in range(min(num_stages, 4)):
            stage_name = f"layers_{i}"
            if hasattr(self.swin, stage_name):
                stage = getattr(self.swin, stage_name)
                for param in stage.parameters():
                    param.requires_grad = False

    def _enable_gradient_checkpointing(self) -> None:
        if self.use_timm and hasattr(self.swin, "set_grad_checkpointing"):
            self.swin.set_grad_checkpointing(True)

    @property
    def feature_info(self):
        if self.use_timm:
            return getattr(self.swin, "feature_info", None)
        return getattr(self, "_feature_info", None)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.shape[1] == 1:  # grayscale → RGB
            x = x.repeat(1, 3, 1, 1)

        features = self.swin(x)

        outputs = []
        for idx, feat in enumerate(features[: self.num_scales]):
            if feat.dim() == 4 and feat.shape[1] != feat.shape[-1]:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            if self.projections is not None:
                feat = self.projections[idx](feat)
            outputs.append(feat)
        return outputs

    def get_output_channels(self) -> List[int]:
        return list(self._output_channels)


class _SwinBackboneWithFPN(nn.Module):
    """Wraps Swin backbone with a lightweight FPN."""

    def __init__(
        self,
        variant: str = "tiny",
        pretrained: bool = True,
        num_scales: int = 4,
        input_size: int = 224,
        output_channels: Optional[Sequence[int]] = None,
        fpn_channels: int = 256,
        **kwargs,
    ) -> None:
        super().__init__()
        self.backbone = _SwinBackbone(
            variant=variant,
            pretrained=pretrained,
            num_scales=num_scales,
            input_size=input_size,
            output_channels=None,
            **kwargs,
        )

        native_channels = self.backbone.native_channels
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_ch, fpn_channels, kernel_size=1) for in_ch in native_channels]
        )
        self.output_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(32, fpn_channels),
                    nn.GELU(),
                )
                for _ in range(num_scales)
            ]
        )

        if output_channels is not None:
            if len(output_channels) != num_scales:
                raise ValueError("output_channels length must match num_scales.")
            self.projections = nn.ModuleList(
                [
                    nn.Conv2d(fpn_channels, out_ch, kernel_size=1)
                    for out_ch in output_channels
                ]
            )
            self._output_channels = list(output_channels)
        else:
            self.projections = None
            self._output_channels = [fpn_channels] * num_scales

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.backbone(x)
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, feats)]

        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        outputs = [
            out_conv(lateral)
            for out_conv, lateral in zip(self.output_convs, laterals)
        ]

        if self.projections is not None:
            outputs = [
                proj(feat) for proj, feat in zip(self.projections, outputs)
            ]
        return outputs

    def get_output_channels(self) -> List[int]:
        return list(self._output_channels)


class SwinFeatureExtractor(nn.Module):
    """High-level Swin feature extractor with optional FPN."""

    def __init__(
        self,
        variant: str = "tiny",
        pretrained: bool = True,
        num_scales: int = 4,
        input_size: int = 224,
        output_channels: Optional[Sequence[int]] = None,
        freeze_stages: int = 0,
        freeze: bool = False,
        use_checkpoint: bool = False,
        drop_path_rate: float = 0.2,
        use_fpn: bool = False,
        fpn_channels: int = 256,
        use_timm: bool = True,
        version: str = "v1",
        timm_model_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        backbone_kwargs = dict(
            variant=variant,
            pretrained=pretrained,
            num_scales=num_scales,
            input_size=input_size,
            output_channels=output_channels,
            freeze_stages=freeze_stages,
            use_checkpoint=use_checkpoint,
            drop_path_rate=drop_path_rate,
            use_timm=use_timm,
            version=version,
            timm_model_name=timm_model_name,
        )

        if use_fpn:
            self.backbone = _SwinBackboneWithFPN(
                fpn_channels=fpn_channels,
                **backbone_kwargs,
            )
        else:
            self.backbone = _SwinBackbone(**backbone_kwargs)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)

    def get_output_channels(self) -> List[int]:
        return self.backbone.get_output_channels()



class CrossScaleSpatialAttention(nn.Module):
    """Multi-head attention operating over backbone feature maps."""

    def __init__(self, dims: Sequence[int], num_heads: int = 4, proj_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.num_scales = len(dims)
        self.proj_dim = proj_dim
        self.projections = nn.ModuleList([nn.Conv2d(dim, proj_dim, kernel_size=1) for dim in dims])
        self.attn = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(proj_dim)
        self.output = nn.ModuleList([nn.Conv2d(proj_dim, dim, kernel_size=1) for dim in dims])

    def forward(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if not features:
            return []
        bsz = features[0].shape[0]
        min_h = min(f.shape[2] for f in features)
        min_w = min(f.shape[3] for f in features)

        aligned = []
        shapes = []
        for feat, proj in zip(features, self.projections):
            shapes.append(feat.shape)
            if feat.shape[2:] != (min_h, min_w):
                feat = F.adaptive_avg_pool2d(feat, (min_h, min_w))
            feat = proj(feat)  # (B, proj_dim, H, W)
            aligned.append(feat.flatten(2).transpose(1, 2))  # (B, HW, proj_dim)

        combined = torch.cat(aligned, dim=1)
        attended, _ = self.attn(combined, combined, combined)
        attended = self.norm(attended + combined)

        splits = torch.split(attended, [a.shape[1] for a in aligned], dim=1)
        outputs: List[torch.Tensor] = []
        for split, out_proj, (b, c, h, w) in zip(splits, self.output, shapes):
            split = split.transpose(1, 2).reshape(bsz, self.proj_dim, min_h, min_w)
            split = out_proj(split)
            if (h, w) != (min_h, min_w):
                split = F.interpolate(split, size=(h, w), mode="bilinear", align_corners=False)
            outputs.append(split)
        return outputs


class LearnableScaleAttention(nn.Module):
    """Learns per-scale weights using feature statistics and logits."""

    def __init__(
        self,
        num_scales: int,
        feature_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.temperature = nn.Parameter(torch.tensor(1.0))
        stats_dim = num_scales * 4  # entropy, max-logit, mean, std
        input_dim = stats_dim + num_scales * feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_scales),
        )
        self.bias = nn.Parameter(torch.zeros(num_scales))
        self.register_buffer("ema_accuracy", torch.ones(num_scales) * 0.5)
        self.register_buffer("ema_confidence", torch.ones(num_scales) * 0.5)
        self.register_buffer("ema_calibration", torch.ones(num_scales))
        self.momentum = 0.05
        self._pending_update: Optional[Dict[str, torch.Tensor]] = None

    def _compute_stats(self, features: List[torch.Tensor], logits: List[torch.Tensor]) -> torch.Tensor:
        stats = []
        proj_feats = []
        for feat, logit in zip(features, logits):
            probs = F.softmax(logit, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            max_logit = logit.max(dim=-1)[0]
            feat_mean = feat.mean(dim=-1)
            feat_std = feat.std(dim=-1)
            stats.append(torch.stack([entropy, max_logit, feat_mean, feat_std], dim=-1))
            proj_feats.append(feat)
        stats_tensor = torch.cat(stats, dim=-1)
        features_tensor = torch.cat(proj_feats, dim=-1)
        return torch.cat([stats_tensor, features_tensor], dim=-1)

    def forward(self, features: List[torch.Tensor], logits: List[torch.Tensor]) -> torch.Tensor:
        stats = self._compute_stats(features, logits)
        attn_logits = self.mlp(stats) + self.bias.unsqueeze(0)
        attn_logits = torch.clamp(attn_logits, -10.0, 10.0)
        temp = torch.clamp(self.temperature.abs(), 0.1, 5.0)
        weights = F.softmax(attn_logits / temp, dim=-1)

        if self.training:
            stacked = torch.stack(logits, dim=1)
            with torch.no_grad():
                ensemble = stacked.mean(dim=1)
                pseudo_labels = ensemble.argmax(dim=-1)
                confidences = F.softmax(stacked, dim=-1).max(dim=-1)[0]
                correct = (stacked.argmax(dim=-1) == pseudo_labels.unsqueeze(1)).float()
                self._pending_update = {
                    "correct": correct.detach(),
                    "confidence": confidences.detach(),
                }

        return weights

    def update_ema_after_step(self):
        if not self.training or self._pending_update is None:
            return
        correct = self._pending_update["correct"].mean(dim=0)
        confidence = self._pending_update["confidence"].mean(dim=0)
        calibration = correct / (confidence + 1e-6)
        self.ema_accuracy.mul_(1 - self.momentum).add_(correct, alpha=self.momentum)
        self.ema_confidence.mul_(1 - self.momentum).add_(confidence, alpha=self.momentum)
        self.ema_calibration.mul_(1 - self.momentum).add_(calibration, alpha=self.momentum)
        self._pending_update = None

    def extra_repr(self) -> str:
        return f"num_scales={self.num_scales}, temperature={self.temperature.item():.3f}"


class AntiCollapseGate(nn.Module):
    """Prevents ensemble weights from collapsing to a single expert."""

    def __init__(self, num_heads: int, min_fraction: float = 0.05, entropy_weight: float = 0.01):
        super().__init__()
        self.num_heads = num_heads
        self.min_fraction = min_fraction
        self.entropy_weight = entropy_weight
        self.bias = nn.Parameter(torch.zeros(num_heads))

    def forward(self, weights: torch.Tensor, training: bool = False) -> Tuple[torch.Tensor, Dict[str, float], Optional[torch.Tensor]]:
        logits = torch.log(torch.clamp(weights, 1e-6, 1.0)) + self.bias
        adjusted = F.softmax(logits, dim=-1)
        if self.min_fraction > 0:
            adjusted = (1 - self.min_fraction * self.num_heads) * adjusted + self.min_fraction
        stats = {
            "mean": adjusted.mean().item(),
            "std": adjusted.std().item(),
            "min": adjusted.min().item(),
            "max": adjusted.max().item(),
        }
        entropy = None
        if training and self.entropy_weight > 0:
            entropy = -(adjusted * torch.log(adjusted + 1e-8)).sum(dim=-1).mean() * self.entropy_weight
        return adjusted, stats, entropy


class MultiScaleTMEnsemble(nn.Module):
    """
    Flexible TM mapping with learnable binarizers.
    - Allows any number of TM heads (N >= 1)
    - Each head can consume one or more backbone stages (by index), with repeat allowed
    - For Swin/ViT: use dual-channel binarizer; for CNN/ResNet: single-channel
    """
    def __init__(self,
                 feature_dims: list[int],
                 n_classes: int,
                 head_configs: list[dict],
                 backbone_type: str = "swin",
                 init_temperature: float = 1.0,
                 stage_resolutions: Optional[Sequence[int]] = None,
                 use_spatial_attention: bool = False,
                 spatial_attention_heads: int = 4,
                 spatial_attention_dim: int = 256,
                 use_learnable_scale_attention: bool = False,
                 scale_attention_hidden: int = 128,
                 use_anti_collapse_gate: bool = False,
                 gate_entropy_weight: float = 0.01):
        """
        head_configs: list of dicts, each like:
          { "stages": [0,1], "n_clauses": 200, "binarizer": "auto", "thresholds": 16, "pool": "gap" }
        """
        super().__init__()
        self.feature_dims = feature_dims
        self.n_classes = n_classes
        self.backbone_type = backbone_type
        self.heads = nn.ModuleList()
        self.gate = nn.Parameter(torch.ones(len(head_configs)) / max(1, len(head_configs)))
        self.stage_resolutions = list(stage_resolutions) if stage_resolutions is not None else None

        resolution_to_stage = {}
        if self.stage_resolutions is not None:
            for idx, res in enumerate(self.stage_resolutions):
                resolution_to_stage[res] = idx

        self.spatial_attention = None
        if use_spatial_attention:
            self.spatial_attention = CrossScaleSpatialAttention(
                dims=feature_dims,
                num_heads=spatial_attention_heads,
                proj_dim=spatial_attention_dim,
            )

        self.scale_attention = None
        self.scale_feature_dim = max(32, scale_attention_hidden // 2)
        if use_learnable_scale_attention:
            self.scale_attention = LearnableScaleAttention(
                num_scales=len(head_configs),
                feature_dim=self.scale_feature_dim,
                hidden_dim=scale_attention_hidden,
            )

        self.anti_collapse_gate = None
        if use_anti_collapse_gate:
            self.anti_collapse_gate = AntiCollapseGate(
                num_heads=len(head_configs),
                entropy_weight=gate_entropy_weight,
            )
        self._entropy_loss: Optional[torch.Tensor] = None
        self.last_attention_weights: Optional[torch.Tensor] = None
        self.last_gate_stats: Dict[str, float] = {}

        for cfg in head_configs:
            stages: Optional[Sequence[int]] = cfg.get("stages")
            if stages is None:
                if "stage" in cfg:
                    stages = [cfg["stage"]]
                elif "resolutions" in cfg and resolution_to_stage:
                    stages = [resolution_to_stage[r] for r in cfg["resolutions"]]
                elif "resolution" in cfg and resolution_to_stage:
                    stages = [resolution_to_stage[cfg["resolution"]]]
                else:
                    stages = [len(self.feature_dims) - 1]
            stages = list(stages)

            for s in stages:
                if s < 0 or s >= len(self.feature_dims):
                    raise ValueError(f"Stage index {s} out of bounds for feature dims length {len(self.feature_dims)}")

            n_clauses = cfg.get("n_clauses", 200)
            num_thresholds = cfg.get("thresholds", 16)
            pool = cfg.get("pool", "gap")
            binarizer = cfg.get("binarizer", "auto")

            # Build head: stage adapters (binarizers) + projector + TM
            adapters = nn.ModuleList()
            fused_dim = 0
            for s in stages:
                in_ch = feature_dims[s]
                if binarizer == "auto":
                    if backbone_type in ["swin", "vit"]:
                        adapter = SwinDualBinarizer(in_ch, num_thresholds, init_temperature, backbone_type=backbone_type)
                    else:
                        adapter = CNNSingleBinarizer(in_ch, num_thresholds, init_temperature, backbone_type=backbone_type)
                elif binarizer == "dual":
                    adapter = SwinDualBinarizer(in_ch, num_thresholds, init_temperature, backbone_type=backbone_type)
                else:
                    adapter = CNNSingleBinarizer(in_ch, num_thresholds, init_temperature, backbone_type=backbone_type)
                adapters.append(adapter)
                fused_dim += adapter.output_channels

            projector = nn.Sequential(
                nn.AdaptiveAvgPool2d(1) if pool == "gap" else nn.AdaptiveMaxPool2d(1),
                nn.Flatten(),
                nn.Linear(fused_dim, max(32, fused_dim // 2)),
                nn.GELU(),
                nn.Linear(max(32, fused_dim // 2), max(16, fused_dim // 4)),
                nn.Sigmoid()
            )

            tm = FuzzyPatternTM_STE(max(16, fused_dim // 4), n_clauses, n_classes, tau=0.5)
            head = nn.Module()
            head.stages = adapters
            head.projector = projector
            head.tm = tm
            if use_learnable_scale_attention:
                head.feature_proj = nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(max(32, fused_dim // 2), self.scale_feature_dim),
                )
            else:
                head.feature_proj = nn.Identity()
            # store cfg as plain attribute
            head.cfg = dict(cfg, stages=stages)
            self.heads.append(head)

    def set_temperature(self, t: float):
        for head in self.heads:
            for adapter in head.stages:
                if hasattr(adapter, "set_temperature"):
                    adapter.set_temperature(t)

    def anneal_binarizers(self, factor: float = 0.95):
        for head in self.heads:
            for adapter in head.stages:
                if hasattr(adapter, "anneal_temperature"):
                    adapter.anneal_temperature(factor)

    def reset_binarizers(self):
        for head in self.heads:
            for adapter in head.stages:
                if hasattr(adapter, "reset_running_stats"):
                    adapter.reset_running_stats()

    def forward(self, multi_scale_features: list[torch.Tensor], use_ste: bool = True):
        logits_list, clause_list = [], []
        head_features: List[torch.Tensor] = []

        feature_source = multi_scale_features
        if self.spatial_attention is not None:
            feature_source = self.spatial_attention(multi_scale_features)

        for head in self.heads:
            adapters = head.stages
            selected_feats = [feature_source[s] for s in head.cfg["stages"]]
            feats: List[torch.Tensor] = []
            for adapter, feat_map in zip(adapters, selected_feats):
                b = adapter(feat_map, use_discrete=not self.training)
                b = F.adaptive_avg_pool2d(b, 1)  # unify spatial dims to 1x1
                feats.append(b)
            fused = torch.cat(feats, dim=1)  # [B, sum(thresh), 1, 1]
            x = head.projector(fused)
            logits, clauses = head.tm(x, use_ste=use_ste)
            logits_list.append(logits)
            clause_list.append(clauses)
            head_features.append(head.feature_proj(x))

        stacked = torch.stack(logits_list, dim=1)  # [B, H, C]

        if self.scale_attention is not None:
            weights = self.scale_attention(head_features, logits_list)
        else:
            base_weights = F.softmax(self.gate, dim=0)
            weights = base_weights.unsqueeze(0).expand(stacked.shape[0], -1)

        gate_stats: Dict[str, float] = {}
        entropy_loss: Optional[torch.Tensor] = None
        if self.anti_collapse_gate is not None:
            weights, gate_stats, entropy_loss = self.anti_collapse_gate(weights, training=self.training)

        final_logits = torch.sum(stacked * weights.unsqueeze(-1), dim=1)

        self._entropy_loss = entropy_loss
        self.last_attention_weights = weights.detach()
        self.last_gate_stats = gate_stats
        return final_logits, logits_list, clause_list

    def attention_entropy_loss(self) -> Optional[torch.Tensor]:
        return self._entropy_loss

    def update_attention_ema(self):
        if self.scale_attention is not None:
            self.scale_attention.update_ema_after_step()

    def get_gate_diagnostics(self) -> Dict[str, float]:
        return dict(self.last_gate_stats)


# ---------------------------------------------------------------------------
# TM-native Swin backbone replacements


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


@dataclass
class SwinTMStageConfig:
    embed_dim: int
    depth: int
    num_heads: int
    window_size: int = 7
    shift_size: int = 0
    tm_hidden_dim: int = 256
    tm_clauses: int = 256
    head_clauses: int = 128
    dropout: float = 0.0
    drop_path: float = 0.0
    tau: float = 0.5
    clause_pool: str = "mean"
    drop_path_schedule: Optional[Tuple[float, ...]] = None


def build_swin_stage_configs(
    preset: str,
    *,
    tm_hidden_dim: int = 256,
    tm_clauses: int = 256,
    head_clauses: int = 128,
    tau: float = 0.5,
    window_size: int = 7,
    dropout: float = 0.0,
    drop_path: float = 0.0,
    clause_pool: str = "mean",
) -> Tuple[SwinTMStageConfig, ...]:
    preset = preset.lower()
    if preset not in {"tiny", "small", "base", "large"}:
        raise ValueError(f"Unknown Swin TM preset '{preset}'")
    embed_dims = {
        "tiny": (96, 192, 384, 768),
        "small": (96, 192, 384, 768),
        "base": (128, 256, 512, 1024),
        "large": (192, 384, 768, 1536),
    }[preset]
    depths = {
        "tiny": (2, 2, 6, 2),
        "small": (2, 2, 18, 2),
        "base": (2, 2, 18, 2),
        "large": (2, 2, 18, 2),
    }[preset]
    heads = {
        "tiny": (3, 6, 12, 24),
        "small": (3, 6, 12, 24),
        "base": (4, 8, 16, 32),
        "large": (6, 12, 24, 48),
    }[preset]
    configs: List[SwinTMStageConfig] = []
    shift_cycle = (0, window_size // 2)
    total_blocks = sum(depths)
    if drop_path > 0:
        drop_path_rates = torch.linspace(0, drop_path, total_blocks).tolist()
    else:
        drop_path_rates = [0.0 for _ in range(total_blocks)]
    rate_index = 0
    for idx, (dim, depth, head) in enumerate(zip(embed_dims, depths, heads)):
        stage_rates = tuple(drop_path_rates[rate_index : rate_index + depth])
        rate_index += depth
        configs.append(
            SwinTMStageConfig(
                embed_dim=dim,
                depth=depth,
                num_heads=head,
                window_size=window_size,
                shift_size=shift_cycle[idx % 2],
                tm_hidden_dim=tm_hidden_dim,
                tm_clauses=tm_clauses,
                head_clauses=head_clauses,
                dropout=dropout,
                drop_path=stage_rates[-1] if stage_rates else drop_path,
                tau=tau,
                clause_pool=clause_pool,
                drop_path_schedule=stage_rates,
            )
        )
    return tuple(configs)


class WindowAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        *,
        use_flash: bool = False,
        relative_position_bias: Optional[RelativePositionBias2D] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        self.attn = InstrumentedMultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            use_flash=use_flash,
            relative_position_bias=relative_position_bias,
            rotary_emb=rotary_emb,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, context, weights = self.attn(x)
        weights_mean = weights.mean(dim=0) if weights is not None else None
        return attn_out, weights_mean, context

    def set_head_gains(self, gains: torch.Tensor) -> None:
        self.attn.set_head_gains(gains)


class WindowTMFeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        window_size: int,
        tm_hidden_dim: int,
        tm_clauses: int,
        tau: float,
        dropout: float,
        clause_pool: str,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.norm = nn.LayerNorm(embed_dim)
        in_dim = window_size * window_size * embed_dim
        self.pre_tm = nn.Sequential(nn.Linear(in_dim, tm_hidden_dim), nn.Sigmoid())
        self.tm = FuzzyPatternTM_STE(tm_hidden_dim, tm_clauses, embed_dim, tau=tau)
        self.post_tm = nn.Sequential(nn.Linear(embed_dim, in_dim), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.clause_pool = clause_pool
        self.last_clause_outputs: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, *, use_ste: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W, C = x.shape
        x_norm = self.norm(x)
        windows = window_partition(x_norm, self.window_size)
        windows_flat = windows.reshape(-1, self.window_size * self.window_size * C)
        tm_in = self.pre_tm(windows_flat)
        logits, clause_outputs = self.tm(tm_in, use_ste=use_ste)
        self.last_clause_outputs = clause_outputs.detach()
        window_out = self.post_tm(logits).view(-1, self.window_size, self.window_size, C)
        merged = window_reverse(window_out, self.window_size, H, W)
        gate = torch.sigmoid(self.gate)
        fused = gate * merged + (1.0 - gate) * torch.sigmoid(merged)
        delta = self.dropout(fused)
        if self.clause_pool == "mean":
            summary = clause_outputs.mean(dim=1)
        elif self.clause_pool == "max":
            summary = clause_outputs.max(dim=1)[0]
        else:
            summary = clause_outputs
        return delta, summary


class SwinTMBlock(nn.Module):
    def __init__(self, config: SwinTMStageConfig) -> None:
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = WindowAttention(config.embed_dim, config.num_heads)
        self.drop_path1 = DropPath(config.drop_path)
        self.window_ffn = WindowTMFeedForward(
            embed_dim=config.embed_dim,
            window_size=config.window_size,
            tm_hidden_dim=config.tm_hidden_dim,
            tm_clauses=config.tm_clauses,
            tau=config.tau,
            dropout=config.dropout,
            clause_pool=config.clause_pool,
        )
        self.drop_path2 = DropPath(config.drop_path)

    def forward(self, x: torch.Tensor, H: int, W: int, *, use_ste: bool = True):
        B, L, C = x.shape
        window_size = self.config.window_size
        h = self.norm1(x).view(B, H, W, C)
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h or pad_w:
            h = h.permute(0, 3, 1, 2)
            h = F.pad(h, (0, pad_w, 0, pad_h))
            h = h.permute(0, 2, 3, 1)
        H_pad, W_pad = H + pad_h, W + pad_w
        shift = self.config.shift_size if window_size > self.config.shift_size else 0
        if shift > 0:
            h = torch.roll(h, shifts=(-shift, -shift), dims=(1, 2))
        windows = window_partition(h, window_size).view(-1, window_size * window_size, C)
        attn_out, attn_weights, head_context = self.attn(windows)
        attn_out = attn_out.view(-1, window_size, window_size, C)
        head_dim = self.attn.attn.head_dim
        head_context = head_context.view(-1, window_size, window_size, self.config.num_heads, head_dim)
        head_context_flat = head_context.view(-1, window_size, window_size, self.config.num_heads * head_dim)
        head_context_merged = window_reverse(head_context_flat, window_size, H_pad, W_pad)
        head_context_merged = head_context_merged.view(B, H_pad, W_pad, self.config.num_heads, head_dim)
        shifted = window_reverse(attn_out, window_size, H_pad, W_pad)
        if shift > 0:
            shifted = torch.roll(shifted, shifts=(shift, shift), dims=(1, 2))
            head_context_merged = torch.roll(head_context_merged, shifts=(shift, shift), dims=(1, 2))
        if pad_h or pad_w:
            shifted = shifted[:, :H, :W, :]
            head_context_merged = head_context_merged[:, :H, :W, :, :]
        shifted_tokens = shifted.reshape(B, H * W, C)
        head_context_tokens = head_context_merged.reshape(B, H * W, self.config.num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        x = x + self.drop_path1(shifted_tokens)
        y = x.view(B, H, W, C)
        if pad_h or pad_w:
            y = y.permute(0, 3, 1, 2)
            y = F.pad(y, (0, pad_w, 0, pad_h))
            y = y.permute(0, 2, 3, 1)
        delta, clause_summary = self.window_ffn(y, use_ste=use_ste)
        if pad_h or pad_w:
            delta = delta[:, :H, :W, :]
        delta_tokens = delta.reshape(B, H * W, C)
        x = x + self.drop_path2(delta_tokens)
        return x, clause_summary, attn_weights, head_context_tokens


class SwinTMStage(nn.Module):
    def __init__(self, config: SwinTMStageConfig, downsample: bool, num_classes: int) -> None:
        super().__init__()
        self.config = config
        blocks = []
        schedule = config.drop_path_schedule or tuple([config.drop_path] * config.depth)
        for i in range(config.depth):
            shift = config.shift_size if (i % 2 == 1) else 0
            block_cfg = SwinTMStageConfig(
                embed_dim=config.embed_dim,
                depth=1,
                num_heads=config.num_heads,
                window_size=config.window_size,
                shift_size=shift,
                tm_hidden_dim=config.tm_hidden_dim,
                tm_clauses=config.tm_clauses,
                head_clauses=config.head_clauses,
                dropout=config.dropout,
                drop_path=schedule[i],
                tau=config.tau,
                clause_pool=config.clause_pool,
            )
            blocks.append(SwinTMBlock(block_cfg))
        self.blocks = nn.ModuleList(blocks)
        self.patch_merge = PatchMerging(config.embed_dim, config.embed_dim * 2) if downsample else None
        self.head_norm = nn.LayerNorm(config.embed_dim)
        self.head_proj = nn.Sequential(nn.Linear(config.embed_dim, config.tm_hidden_dim), nn.Sigmoid())
        self.head_tm = FuzzyPatternTM_STE(config.tm_hidden_dim, config.head_clauses, num_classes, tau=config.tau)

    def forward(self, x: torch.Tensor, H: int, W: int, *, use_ste: bool = True,
                collect_diagnostics: bool = False,
                record_callback: Optional[Callable[[str, torch.Tensor, int, int, Optional[torch.Tensor]], None]] = None,
                stage_index: int = 0):
        clause_summaries: List[torch.Tensor] = []
        attn_summaries: List[torch.Tensor] = []
        for block_idx, block in enumerate(self.blocks):
            x, summary, attn, head_tokens = block(x, H, W, use_ste=use_ste)
            clause_summaries.append(summary)
            attn_summaries.append(attn)
            if collect_diagnostics and record_callback is not None:
                record_callback(f"stage{stage_index + 1}_block{block_idx + 1}", x, H, W, head_tokens)
        B, L, C = x.shape
        feat = self.head_norm(x).mean(dim=1)
        head_in = self.head_proj(feat)
        logits, clauses = self.head_tm(head_in, use_ste=use_ste)
        if collect_diagnostics and record_callback is not None:
            record_callback(f"stage{stage_index + 1}_out", x, H, W, None)
        if self.patch_merge is not None:
            x = x.view(B, H, W, C)
            x = self.patch_merge(x)
            H, W = x.shape[1], x.shape[2]
            x = x.view(B, H * W, -1)
        return x, H, W, logits, clauses, clause_summaries, attn_summaries


class SwinTM(nn.Module):
    def __init__(
        self,
        stage_configs: Optional[Sequence[SwinTMStageConfig]] = None,
        *,
        preset: str = "tiny",
        num_classes: int = 10,
        in_channels: int = 3,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        if stage_configs is None:
            stage_configs = build_swin_stage_configs(preset)
        self.stage_configs = tuple(stage_configs)
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_embed = nn.Conv2d(in_channels, self.stage_configs[0].embed_dim, kernel_size=4, stride=4)
        stages = []
        for idx, cfg in enumerate(self.stage_configs):
            downsample = idx < len(self.stage_configs) - 1
            stages.append(SwinTMStage(cfg, downsample, num_classes))
        self.stages = nn.ModuleList(stages)
        final_dim = self.stage_configs[-1].embed_dim
        self.final_norm = nn.LayerNorm(final_dim)
        self.final_proj = nn.Sequential(nn.Linear(final_dim, self.stage_configs[-1].tm_hidden_dim), nn.Sigmoid())
        self.final_tm = FuzzyPatternTM_STE(
            self.stage_configs[-1].tm_hidden_dim,
            self.stage_configs[-1].head_clauses,
            num_classes,
            tau=self.stage_configs[-1].tau,
        )
        self.last_stage_logits: List[torch.Tensor] = []
        self.last_clause_summaries: List[List[torch.Tensor]] = []
        self.last_attention_weights: List[List[torch.Tensor]] = []
        self.last_scale_fused_logits: Optional[torch.Tensor] = None
        self.last_clause_attention_logits: Optional[torch.Tensor] = None
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, *, use_ste: bool = True):
        if x.dim() == 2:
            H, W = self.image_size
            C = self.patch_embed.in_channels
            x = x.view(x.shape[0], C, H, W)
        feats = self.patch_embed(x)
        B, C, H, W = feats.shape
        feats = feats.view(B, C, H * W).transpose(1, 2)
        stage_logits: List[torch.Tensor] = []
        clause_lists: List[torch.Tensor] = []
        summary_lists: List[List[torch.Tensor]] = []
        attn_lists: List[List[torch.Tensor]] = []
        for stage in self.stages:
            feats, H, W, s_logits, clauses, summaries, attn_summary = stage(feats, H, W, use_ste=use_ste)
            stage_logits.append(s_logits)
            clause_lists.append(clauses)
            summary_lists.append(summaries)
            attn_lists.append(attn_summary)
        pooled = self.final_norm(feats).mean(dim=1)
        final_in = self.final_proj(pooled)
        final_logits, final_clauses = self.final_tm(final_in, use_ste=use_ste)
        self.last_stage_logits = stage_logits + [final_logits]
        self.last_clause_summaries = summary_lists
        self.last_attention_weights = attn_lists
        self.last_scale_fused_logits = None
        self.last_clause_attention_logits = None
        return final_logits, stage_logits, clause_lists, final_clauses


