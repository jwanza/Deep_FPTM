import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .tm import FuzzyPatternTM_STE
from .binarizers import SwinDualBinarizer, CNNSingleBinarizer


class SwinFeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "swin_tiny_patch4_window7_224", pretrained: bool = True, freeze: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor):
        # Returns list of [B, C, H, W] features at multiple scales
        feats = self.backbone(x)
        return feats


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
                 init_temperature: float = 1.0):
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

        for cfg in head_configs:
            stages = cfg.get("stages", [0])
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
                        adapters.append(SwinDualBinarizer(in_ch, num_thresholds, init_temperature))
                        fused_dim += num_thresholds * 2
                    else:
                        adapters.append(CNNSingleBinarizer(in_ch, num_thresholds, init_temperature))
                        fused_dim += num_thresholds
                elif binarizer == "dual":
                    adapters.append(SwinDualBinarizer(in_ch, num_thresholds, init_temperature))
                    fused_dim += num_thresholds * 2
                else:
                    adapters.append(CNNSingleBinarizer(in_ch, num_thresholds, init_temperature))
                    fused_dim += num_thresholds

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
            # store cfg as plain attribute
            head.cfg = cfg
            self.heads.append(head)

    def set_temperature(self, t: float):
        for head in self.heads:
            for adapter in head.stages:
                if hasattr(adapter, "set_temperature"):
                    adapter.set_temperature(t)

    def forward(self, multi_scale_features: list[torch.Tensor], use_ste: bool = True):
        logits_list, clause_list = [], []
        for head in self.heads:
            adapters = head.stages
            feats = []
            for adapter, feat_map in zip(adapters, [multi_scale_features[s] for s in head.cfg["stages"]]):
                b = adapter(feat_map, use_discrete=not self.training)
                b = F.adaptive_avg_pool2d(b, 1)  # unify spatial dims to 1x1
                feats.append(b)
            fused = torch.cat(feats, dim=1)  # [B, sum(thresh), 1, 1]
            x = head.projector(fused)
            logits, clauses = head.tm(x, use_ste=use_ste)
            logits_list.append(logits)
            clause_list.append(clauses)

        weights = F.softmax(self.gate, dim=0)
        stacked = torch.stack(logits_list, dim=1)  # [B, H, C]
        final_logits = torch.sum(stacked * weights.view(1, -1, 1), dim=1)
        return final_logits, logits_list, clause_list


