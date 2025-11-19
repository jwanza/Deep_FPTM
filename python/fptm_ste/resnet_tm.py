from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tm import FuzzyPatternTM_STE, FuzzyPatternTM_STCM


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


@dataclass
class ResNetTMConfig:
    layers: Sequence[int]
    channels: Sequence[int]
    tm_hidden: int = 256
    tm_clauses: int = 128
    tau: float = 0.5
    tau_schedule: Optional[Sequence[float]] = None
    tm_cls: Union[Type[nn.Module], str] = FuzzyPatternTM_STE
    tm_kwargs: Dict[str, Any] = field(default_factory=dict)


class BasicTMBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        tm_hidden: int,
        tm_clauses: int,
        tau: float,
        downsample: Optional[nn.Module] = None,
        tm_cls: Union[Type[nn.Module], str] = FuzzyPatternTM_STE,
        tm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.pre_tm = nn.Sequential(nn.Linear(out_channels, tm_hidden), nn.Sigmoid())

        if isinstance(tm_cls, str):
             if tm_cls == "FuzzyPatternTM_STCM":
                 tm_cls = FuzzyPatternTM_STCM
             else:
                 tm_cls = FuzzyPatternTM_STE
        
        kwargs = tm_kwargs or {}
        self.tm = tm_cls(tm_hidden, tm_clauses, out_channels, tau=tau, **kwargs)
        self.post_tm = nn.Sequential(nn.Linear(out_channels, out_channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor, *, use_ste: bool = True) -> torch.Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        B, C, H, W = out.shape
        out_flat = out.permute(0, 2, 3, 1).contiguous().view(-1, C)
        tm_in = self.pre_tm(out_flat)
        logits, _ = self.tm(tm_in, use_ste=use_ste)
        logits = logits.view(B, H, W, C)
        out = self.post_tm(logits).permute(0, 3, 1, 2).contiguous()

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return F.relu(out, inplace=True)


class BottleneckTMBlock(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        tm_hidden: int,
        tm_clauses: int,
        tau: float,
        downsample: Optional[nn.Module] = None,
        tm_cls: Union[Type[nn.Module], str] = FuzzyPatternTM_STE,
        tm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        width = out_channels
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.pre_tm = nn.Sequential(nn.Linear(out_channels * self.expansion, tm_hidden), nn.Sigmoid())

        if isinstance(tm_cls, str):
             if tm_cls == "FuzzyPatternTM_STCM":
                 tm_cls = FuzzyPatternTM_STCM
             else:
                 tm_cls = FuzzyPatternTM_STE
        
        kwargs = tm_kwargs or {}
        self.tm = tm_cls(tm_hidden, tm_clauses, out_channels * self.expansion, tau=tau, **kwargs)
        self.post_tm = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, *, use_ste: bool = True) -> torch.Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        B, C, H, W = out.shape
        out_flat = out.permute(0, 2, 3, 1).contiguous().view(-1, C)
        tm_in = self.pre_tm(out_flat)
        logits, _ = self.tm(tm_in, use_ste=use_ste)
        logits = logits.view(B, H, W, C)
        out = self.post_tm(logits).permute(0, 3, 1, 2).contiguous()

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return F.relu(out, inplace=True)


class ResNetTM(nn.Module):
    def __init__(
        self,
        config: ResNetTMConfig,
        num_classes: int = 10,
        in_channels: int = 3,
        block: Callable[..., nn.Module] = BasicTMBlock,
        input_norm: Optional[str] = None,
    ) -> None:
        super().__init__()
        if len(config.channels) != len(config.layers):
            raise ValueError("channels and layers must have the same length")
        self.config = config
        self.stage_taus = self._build_stage_taus(config)
        self.final_tau = self.stage_taus[-1]
        stem_out_channels = config.channels[0]
        self.input_norm = self._build_input_norm(input_norm, in_channels)
        self.in_channels = stem_out_channels
        self.conv1 = nn.Conv2d(in_channels, stem_out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stages = nn.ModuleList()
        self.stage_dims: List[int] = []
        for idx, (planes, blocks) in enumerate(zip(config.channels, config.layers)):
            stride = 1 if idx == 0 else 2
            layer, out_dim = self._make_layer(block, planes, blocks, stride, config, self.stage_taus[idx])
            self.stages.append(layer)
            self.stage_dims.append(out_dim)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.stage_proj = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, config.tm_hidden), nn.Sigmoid()) for dim in self.stage_dims
        )
        
        tm_cls = config.tm_cls
        if isinstance(tm_cls, str):
             if tm_cls == "FuzzyPatternTM_STCM":
                 tm_cls = FuzzyPatternTM_STCM
             else:
                 tm_cls = FuzzyPatternTM_STE
        
        tm_kwargs = config.tm_kwargs or {}

        self.stage_heads = nn.ModuleList(
            tm_cls(config.tm_hidden, config.tm_clauses, num_classes, tau=tau, **tm_kwargs)
            for tau in self.stage_taus
        )
        self.final_proj = nn.Sequential(nn.Linear(self.stage_dims[-1], config.tm_hidden), nn.Sigmoid())
        self.final_tm = tm_cls(config.tm_hidden, config.tm_clauses, num_classes, tau=self.final_tau, **tm_kwargs)
        self.last_stage_logits: List[torch.Tensor] = []
        self.last_stage_clauses: List[torch.Tensor] = []
        self._reset_parameters()

    def _build_stage_taus(self, config: ResNetTMConfig) -> List[float]:
        if config.tau_schedule is None:
            return [config.tau for _ in config.layers]
        tau_schedule = list(config.tau_schedule)
        if len(tau_schedule) != len(config.layers):
            raise ValueError("tau_schedule length must match number of layers")
        return [float(t) for t in tau_schedule]

    @staticmethod
    def _build_input_norm(norm_type: Optional[str], channels: int) -> nn.Module:
        if norm_type is None or norm_type.lower() == "none":
            return nn.Identity()
        norm = norm_type.lower()
        if norm == "batchnorm":
            return nn.BatchNorm2d(channels)
        if norm == "layernorm":
            return nn.GroupNorm(1, channels)
        if norm == "instancenorm":
            return nn.InstanceNorm2d(channels, affine=True)
        raise ValueError(f"Unknown input_norm '{norm_type}'")

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_layer(
        self,
        block: Callable[..., nn.Module],
        planes: int,
        blocks: int,
        stride: int,
        config: ResNetTMConfig,
        tau: float,
    ) -> Tuple[nn.Sequential, int]:
        downsample = None
        out_channels = planes * block.expansion
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(
            block(
                self.in_channels,
                planes,
                stride,
                config.tm_hidden,
                config.tm_clauses,
                tau,
                downsample,
                tm_cls=config.tm_cls,
                tm_kwargs=config.tm_kwargs,
            )
        )
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes, 1, config.tm_hidden, config.tm_clauses, tau, 
                                tm_cls=config.tm_cls, tm_kwargs=config.tm_kwargs))
        return nn.Sequential(*layers), out_channels

    def forward(self, x: torch.Tensor, *, use_ste: bool = True):
        x = self.input_norm(x)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        stage_logits: List[torch.Tensor] = []
        stage_clauses: List[torch.Tensor] = []

        for idx, stage in enumerate(self.stages):
            x = stage(x)
            pooled = self.avgpool(x).flatten(1)
            proj = self.stage_proj[idx](pooled)
            logits, clauses = self.stage_heads[idx](proj, use_ste=use_ste)
            stage_logits.append(logits)
            stage_clauses.append(clauses)

        final_proj = self.final_proj(self.avgpool(x).flatten(1))
        final_logits, final_clauses = self.final_tm(final_proj, use_ste=use_ste)
        self.last_stage_logits = stage_logits + [final_logits]
        self.last_stage_clauses = stage_clauses + [final_clauses]
        return final_logits, stage_logits, stage_clauses, final_clauses


def resnet_tm18(
    num_classes: int = 10,
    *,
    in_channels: int = 3,
    tm_hidden: int = 256,
    tm_clauses: int = 128,
    tau: float = 0.5,
    tau_schedule: Optional[Sequence[float]] = None,
    input_norm: Optional[str] = None,
    tm_cls: Union[Type[nn.Module], str] = FuzzyPatternTM_STE,
    tm_kwargs: Optional[Dict[str, Any]] = None,
) -> ResNetTM:
    config = ResNetTMConfig(
        layers=(2, 2, 2, 2),
        channels=(64, 128, 256, 512),
        tm_hidden=tm_hidden,
        tm_clauses=tm_clauses,
        tau=tau,
        tau_schedule=tau_schedule,
        tm_cls=tm_cls,
        tm_kwargs=tm_kwargs or {},
    )
    return ResNetTM(
        config,
        num_classes=num_classes,
        in_channels=in_channels,
        block=BasicTMBlock,
        input_norm=input_norm,
    )


def resnet_tm34(
    num_classes: int = 10,
    *,
    in_channels: int = 3,
    tm_hidden: int = 256,
    tm_clauses: int = 128,
    tau: float = 0.5,
    tau_schedule: Optional[Sequence[float]] = None,
    input_norm: Optional[str] = None,
    tm_cls: Union[Type[nn.Module], str] = FuzzyPatternTM_STE,
    tm_kwargs: Optional[Dict[str, Any]] = None,
) -> ResNetTM:
    config = ResNetTMConfig(
        layers=(3, 4, 6, 3),
        channels=(64, 128, 256, 512),
        tm_hidden=tm_hidden,
        tm_clauses=tm_clauses,
        tau=tau,
        tau_schedule=tau_schedule,
        tm_cls=tm_cls,
        tm_kwargs=tm_kwargs or {},
    )
    return ResNetTM(
        config,
        num_classes=num_classes,
        in_channels=in_channels,
        block=BasicTMBlock,
        input_norm=input_norm,
    )


def resnet_tm50(
    num_classes: int = 10,
    *,
    in_channels: int = 3,
    tm_hidden: int = 256,
    tm_clauses: int = 128,
    tau: float = 0.5,
    tau_schedule: Optional[Sequence[float]] = None,
    input_norm: Optional[str] = None,
    tm_cls: Union[Type[nn.Module], str] = FuzzyPatternTM_STE,
    tm_kwargs: Optional[Dict[str, Any]] = None,
) -> ResNetTM:
    config = ResNetTMConfig(
        layers=(3, 4, 6, 3),
        channels=(64, 128, 256, 512),
        tm_hidden=tm_hidden,
        tm_clauses=tm_clauses,
        tau=tau,
        tau_schedule=tau_schedule,
        tm_cls=tm_cls,
        tm_kwargs=tm_kwargs or {},
    )
    return ResNetTM(
        config,
        num_classes=num_classes,
        in_channels=in_channels,
        block=BottleneckTMBlock,
        input_norm=input_norm,
    )


def resnet_tm101(
    num_classes: int = 10,
    *,
    in_channels: int = 3,
    tm_hidden: int = 256,
    tm_clauses: int = 128,
    tau: float = 0.5,
    tau_schedule: Optional[Sequence[float]] = None,
    input_norm: Optional[str] = None,
    tm_cls: Union[Type[nn.Module], str] = FuzzyPatternTM_STE,
    tm_kwargs: Optional[Dict[str, Any]] = None,
) -> ResNetTM:
    config = ResNetTMConfig(
        layers=(3, 4, 23, 3),
        channels=(64, 128, 256, 512),
        tm_hidden=tm_hidden,
        tm_clauses=tm_clauses,
        tau=tau,
        tau_schedule=tau_schedule,
        tm_cls=tm_cls,
        tm_kwargs=tm_kwargs or {},
    )
    return ResNetTM(
        config,
        num_classes=num_classes,
        in_channels=in_channels,
        block=BottleneckTMBlock,
        input_norm=input_norm,
    )
