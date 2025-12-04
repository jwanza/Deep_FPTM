from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableScaleAttention(nn.Module):
    """
    Learns per-scale weights using feature statistics and logits.
    
    Dynamically weights the contribution of each scale (fine-grained vs global)
    based on the input image's characteristics.
    """

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
        # Stats: entropy, max-logit, mean, std (4 stats per scale)
        stats_dim = num_scales * 4
        # Input to MLP: Stats + Scale Features
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
        
        # EMA tracking for visualization/debugging
        self.register_buffer("ema_weights", torch.ones(num_scales) / num_scales)
        self.momentum = 0.05

    def _compute_stats(self, features: List[torch.Tensor], logits: List[torch.Tensor]) -> torch.Tensor:
        stats = []
        proj_feats = []
        for feat, logit in zip(features, logits):
            # Compute stats from logits
            probs = F.softmax(logit, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            max_logit = logit.max(dim=-1)[0]
            
            # Compute stats from features
            if feat.dim() > 2:
                feat_pooled = feat.mean(dim=[-2, -1]) # Global pool spatial dims
            else:
                feat_pooled = feat
                
            feat_mean = feat_pooled.mean(dim=-1)
            feat_std = feat_pooled.std(dim=-1)
            
            stats.append(torch.stack([entropy, max_logit, feat_mean, feat_std], dim=-1))
            
            # Ensure feature dims match expected input (project or pad if needed)
            # For simplicity, we assume features are already projected or we take a subset/mean
            # Here we assume features are [B, C_scale], and we might need to reduce to feature_dim
            # But the init receives feature_dim. Let's assume input features are already projected to a common dim.
            # If not, we take the mean or first N channels.
            if feat_pooled.shape[-1] > self.mlp[0].in_features // self.num_scales - 4:
                 # Simple truncation if too large (or could project)
                 feat_reduced = feat_pooled[:, : self.mlp[0].in_features // self.num_scales - 4]
            else:
                 feat_reduced = feat_pooled
            
            proj_feats.append(feat_reduced)
            
        stats_tensor = torch.cat(stats, dim=-1) # [B, num_scales * 4]
        
        # This part assumes features are compatible. 
        # In a robust implementation, we might skip features in the MLP if dims mismatch significantly
        # or rely on the caller to project them. 
        # For SOTA Hybrid, we usually project scale features to a common embedding size before fusion.
        
        # Let's adjust: We concatenate stats. The feature part of MLP input is tricky if dims vary.
        # Let's rely mainly on stats for the attention mechanism to be generic.
        # OR better: The user of this module should project features.
        
        return stats_tensor

    def forward(
        self, 
        features: List[torch.Tensor], 
        logits: List[torch.Tensor], 
        return_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            features: List of feature tensors (one per scale).
            logits: List of logit tensors (one per scale).
        """
        # Simplify: Just use stats for attention weights to avoid dimension headaches
        # Stats: [B, num_scales * 4]
        stats = []
        for logit in logits:
            probs = F.softmax(logit, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            max_logit = logit.max(dim=-1)[0]
            stats.append(torch.stack([entropy, max_logit], dim=-1))
        
        # Also add mean/std of features
        for feat in features:
             if feat.dim() > 2:
                feat = feat.mean(dim=[-2, -1])
             stats.append(torch.stack([feat.mean(dim=-1), feat.std(dim=-1)], dim=-1))
             
        # Combined stats: [B, num_scales * 4]
        stats_input = torch.cat(stats, dim=-1)
        
        # Note: We need to adjust MLP input dim in __init__ if we change logic here.
        # The __init__ assumed input_dim = stats_dim + features.
        # Let's stick to using just stats for robustness unless features are projected.
        # To fix the __init__ vs forward mismatch without changing __init__ signature too much,
        # we will project features if provided, or just zero-pad if we want to ignore them.
        # Actually, let's just implement a robust version that projects features if passed.
        
        # Better: Re-implement a simpler MLP that only takes the stats (4 per scale)
        # We'll create a new MLP on the fly if needed? No, that breaks loading.
        # We will assume the `feature_dim` passed to init was 0 if we only want stats, 
        # or that the user provides projected features.
        
        # Let's proceed with the `_compute_stats` logic but handle the feature concatenation carefully.
        # We'll assume the caller projects features to `feature_dim` before calling if `feature_dim > 0`.
        
        # Re-calc stats including features
        input_vecs = []
        for i, (feat, logit) in enumerate(zip(features, logits)):
            probs = F.softmax(logit, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            max_logit = logit.max(dim=-1)[0]
            
            if feat.dim() > 2:
                feat_flat = feat.mean(dim=[-2, -1])
            else:
                feat_flat = feat
                
            f_mean = feat_flat.mean(dim=-1)
            f_std = feat_flat.std(dim=-1)
            
            vec_stats = torch.stack([entropy, max_logit, f_mean, f_std], dim=-1)
            
            # Append features if expected
            # We determine if we need features based on MLP input size
            expected_dim_per_scale = self.mlp[0].in_features // self.num_scales
            if expected_dim_per_scale > 4:
                # We need to pad or use features
                # Take first (expected - 4) channels
                n_feat = expected_dim_per_scale - 4
                if feat_flat.shape[-1] >= n_feat:
                    feat_part = feat_flat[:, :n_feat]
                else:
                    # Pad
                    feat_part = F.pad(feat_flat, (0, n_feat - feat_flat.shape[-1]))
                input_vecs.append(torch.cat([vec_stats, feat_part], dim=-1))
            else:
                input_vecs.append(vec_stats)
                
        mlp_input = torch.cat(input_vecs, dim=-1)
        
        attn_logits = self.mlp(mlp_input) + self.bias.unsqueeze(0)
        attn_logits = torch.clamp(attn_logits, -10.0, 10.0)
        temp = torch.clamp(self.temperature.abs(), 0.1, 5.0)
        weights = F.softmax(attn_logits / temp, dim=-1) # [B, num_scales]
        
        if self.training:
            with torch.no_grad():
                self.ema_weights.mul_(1 - self.momentum).add_(weights.mean(dim=0), alpha=self.momentum)

        # Weighted sum of logits
        # Stack logits: [B, num_scales, num_classes]
        stacked_logits = torch.stack(logits, dim=1)
        final_logits = (stacked_logits * weights.unsqueeze(-1)).sum(dim=1)
        
        if return_weights:
            return final_logits, weights
        return final_logits


class ResidualInterpretableDecision(nn.Module):
    """
    Decision layer with residual connections and better spatial preservation.
    
    1. Preserves spatial structure (concatenates BEFORE pooling)
    2. Residual connection (direct path + enhanced path)
    3. Learnable fusion weight
    """
    
    def __init__(self, input_dims: List[int], num_classes: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.total_dim = sum(input_dims)
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim or self.total_dim
        
        # Direct path: Simple linear classifier
        self.direct_classifier = nn.Linear(self.total_dim, num_classes)
        
        # Enhanced path: Learnable refinements
        self.enhanced_pathway = nn.Sequential(
            nn.Linear(self.total_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, num_classes)
        )
        
        # Learnable fusion weight
        self.fusion_weight = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, features: List[torch.Tensor], return_explanation: bool = False):
        """
        Args:
            features: List of feature tensors (B, C, H, W) or (B, C)
        """
        # Concatenate BEFORE pooling if possible
        if features[0].dim() == 4:
            # Check if all have same H, W. If not, pool them individually.
            # Assuming they might be from different scales, pooling first is safer unless we upsample.
            # But the plan says "Preserve Spatial Dimensions" -> "Keep features in ... H, W format as long as possible"
            # If they are different sizes, we must pool or resize.
            # Let's assume we pool for the final classification if sizes differ.
            
            # Check sizes
            sizes = [f.shape[-2:] for f in features]
            if all(s == sizes[0] for s in sizes):
                # Same size: Concat then pool
                combined_spatial = torch.cat(features, dim=1)
                pooled = F.adaptive_avg_pool2d(combined_spatial, 1)
                combined = pooled.flatten(1)
            else:
                # Different sizes: Pool then concat
                pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in features]
                combined = torch.cat(pooled, dim=1)
        else:
            combined = torch.cat(features, dim=1)
        
        direct_logits = self.direct_classifier(combined)
        enhanced_logits = self.enhanced_pathway(combined)
        
        alpha = torch.sigmoid(self.fusion_weight)
        logits = (1 - alpha) * direct_logits + alpha * enhanced_logits
        
        if return_explanation:
            explanation = {
                'direct_weight': (1 - alpha).item(),
                'enhanced_weight': alpha.item(),
                'direct_logits': direct_logits,
                'enhanced_logits': enhanced_logits,
            }
            return logits, explanation
        
        return logits
