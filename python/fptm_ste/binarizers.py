import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinDualBinarizer(nn.Module):
    """
    Dual-sigmoid learnable binarizer for Swin/ViT-like zero-centered features.
    - Reduces channels to T thresholds via 1x1 conv with learnable bias
    - Produces two channels per threshold:
      * negative channel: sigmoid on (-x - threshold)
      * positive channel: sigmoid on ( x - threshold)
    - STE hard binarization option
    - Cosine-annealed temperature scheduling supported via setter
    """
    def __init__(
        self,
        in_channels: int,
        num_thresholds: int = 16,
        init_temperature: float = 1.0,
        backbone_type: str = "swin",
    ):
        super().__init__()
        self.num_thresholds = num_thresholds
        self.temperature = init_temperature
        self.backbone_type = backbone_type

        self.reduce = nn.Conv2d(in_channels, num_thresholds, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.reduce.weight, gain=0.5)
        nn.init.zeros_(self.reduce.bias)

        # Running stats to place thresholds
        self.register_buffer('running_mean', torch.zeros(1, num_thresholds, 1, 1))
        self.register_buffer('running_std', torch.ones(1, num_thresholds, 1, 1))
        self.register_buffer('initialized', torch.tensor(False))

        # Offsets in std units
        init_offsets = torch.linspace(-1.0, 1.0, num_thresholds).view(1, num_thresholds, 1, 1)
        self.threshold_offsets = nn.Parameter(init_offsets)

    @staticmethod
    def _ste(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hard = (x > 0.5).float()
        return hard + (x - hard).detach()

    def set_temperature(self, t: float):
        self.temperature = float(t)

    def anneal_temperature(self, value: float):
        self.temperature = max(0.01, float(value))

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_std.fill_(1.0)
        self.initialized.fill_(False)

    @property
    def output_channels(self) -> int:
        return self.num_thresholds * 2

    def forward(self, x: torch.Tensor, use_discrete: bool = False) -> torch.Tensor:
        # x: [B, C, H, W], unbounded (zero-centered)
        x_r = self.reduce(x.float())  # [B, T, H, W]

        if not self.initialized and self.training:
            with torch.no_grad():
                self.running_mean.copy_(x_r.mean(dim=[0, 2, 3], keepdim=True))
                self.running_std.copy_(x_r.std(dim=[0, 2, 3], keepdim=True).clamp(min=0.1))
                self.initialized.copy_(torch.tensor(True))

        if self.training:
            with torch.no_grad():
                m = x_r.mean(dim=[0, 2, 3], keepdim=True)
                s = x_r.std(dim=[0, 2, 3], keepdim=True).clamp(min=0.1)
                momentum = 0.05
                self.running_mean.mul_(1 - momentum).add_(m, alpha=momentum)
                self.running_std.mul_(1 - momentum).add_(s, alpha=momentum)
                self.running_mean.clamp_(-10.0, 10.0)
                self.running_std.clamp_(0.01, 10.0)

        thresholds = (self.running_mean + self.threshold_offsets * self.running_std).clamp(-10.0, 10.0)
        temp = self.temperature + 1e-8

        # Dual channels
        diff_neg = torch.clamp((-(x_r) - thresholds) / temp * 5.0, -20.0, 20.0)
        diff_pos = torch.clamp((( x_r) - thresholds) / temp * 5.0, -20.0, 20.0)
        p_neg = torch.sigmoid(diff_neg)
        p_pos = torch.sigmoid(diff_pos)
        p = torch.cat([p_neg, p_pos], dim=1)  # [B, 2T, H, W]

        if use_discrete or not self.training:
            with torch.no_grad():
                return (p > 0.5).float()
        return p


class CNNSingleBinarizer(nn.Module):
    """
    Single-sigmoid learnable binarizer for nonnegative (ReLU) feature maps.
    - Reduces channels to T with 1x1 conv and bias
    - Applies sigmoid((x - threshold)/T) with STE option
    """
    def __init__(
        self,
        in_channels: int,
        num_thresholds: int = 16,
        init_temperature: float = 1.0,
        backbone_type: str = "cnn",
    ):
        super().__init__()
        self.num_thresholds = num_thresholds
        self.temperature = init_temperature
        self.backbone_type = backbone_type

        self.reduce = nn.Conv2d(in_channels, num_thresholds, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.reduce.weight, gain=0.5)
        nn.init.zeros_(self.reduce.bias)

        self.register_buffer('running_mean', torch.zeros(1, num_thresholds, 1, 1))
        self.register_buffer('running_std', torch.ones(1, num_thresholds, 1, 1))
        self.register_buffer('initialized', torch.tensor(False))

        init_offsets = torch.linspace(-1.0, 1.0, num_thresholds).view(1, num_thresholds, 1, 1)
        self.threshold_offsets = nn.Parameter(init_offsets)

    @staticmethod
    def _ste(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hard = (x > 0.5).float()
        return hard + (x - hard).detach()

    def set_temperature(self, t: float):
        self.temperature = float(t)

    def anneal_temperature(self, value: float):
        self.temperature = max(0.01, float(value))

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_std.fill_(1.0)
        self.initialized.fill_(False)

    @property
    def output_channels(self) -> int:
        return self.num_thresholds

    def forward(self, x: torch.Tensor, use_discrete: bool = False) -> torch.Tensor:
        # x: [B, C, H, W], expected nonnegative
        x_r = self.reduce(x.float())

        if not self.initialized and self.training:
            with torch.no_grad():
                self.running_mean.copy_(x_r.mean(dim=[0, 2, 3], keepdim=True))
                self.running_std.copy_(x_r.std(dim=[0, 2, 3], keepdim=True).clamp(min=0.1))
                self.initialized.copy_(torch.tensor(True))

        if self.training:
            with torch.no_grad():
                m = x_r.mean(dim=[0, 2, 3], keepdim=True)
                s = x_r.std(dim=[0, 2, 3], keepdim=True).clamp(min=0.1)
                momentum = 0.05
                self.running_mean.mul_(1 - momentum).add_(m, alpha=momentum)
                self.running_std.mul_(1 - momentum).add_(s, alpha=momentum)
                self.running_mean.clamp_(-10.0, 10.0)
                self.running_std.clamp_(0.01, 10.0)

        thresholds = (self.running_mean + self.threshold_offsets * self.running_std).clamp(-10.0, 10.0)
        temp = self.temperature + 1e-8

        diff = torch.clamp(((x_r - thresholds) / temp) * 5.0, -20.0, 20.0)
        p = torch.sigmoid(diff)  # [B, T, H, W]

        if use_discrete or not self.training:
            with torch.no_grad():
                return (p > 0.5).float()
        return p


