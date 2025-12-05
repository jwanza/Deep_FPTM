"""
Benchmark Script for SOTA Hybrid TM.

Compares Accuracy, Throughput, and FLOPs/Params against baselines.
"""

import time
import torch
import torch.nn as nn
from fptm_ste.sota_hybrid import SotaHybridTM
from fptm_ste.backbones import UniversalBackboneFactory

try:
    from fptm_ste.tm import FuzzyPatternTM_STCM
except ImportError:
    pass # Might not be needed for direct comparison if we rely on SotaHybridTM

def benchmark_model(model, input_shape=(1, 3, 224, 224), iterations=100, device="cuda"):
    model.eval()
    model.to(device)
    x = torch.randn(input_shape).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
            
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x)
            
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iterations
    throughput = input_shape[0] / avg_time
    
    params = sum(p.numel() for p in model.parameters())
    return avg_time * 1000, throughput, params

def run_benchmarks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmarks on {device}...\n")
    
    configs = [
        ("Swin-Tiny (Pure)", "swin_tiny", False),
        ("ResNet-18 (Pure)", "resnet18", False),
        ("SOTA Hybrid (Swin-Tiny)", "swin_tiny", True),
        ("SOTA Hybrid (ResNet-18)", "resnet18", True),
    ]
    
    print(f"{'Model':<30} | {'Params (M)':<10} | {'Latency (ms)':<12} | {'Throughput (img/s)':<20}")
    print("-" * 80)
    
    for name, backbone, is_hybrid in configs:
        if is_hybrid:
            model = SotaHybridTM(
                n_classes=100, 
                backbone=backbone, 
                pretrained=False,
                n_clauses_base=256
            )
        else:
            # Create standard backbone + linear head
            # Split manually for Robustness
            if "_" in backbone:
                bb_type = backbone.split("_")[0]
                bb_variant = backbone.split("_", 1)[1]
            else:
                # Handle resnet18 case (no underscore)
                if "resnet" in backbone:
                    bb_type = "resnet"
                    bb_variant = backbone.replace("resnet", "")
                else:
                    bb_type = backbone
                    bb_variant = "base"
                    
            bb = UniversalBackboneFactory.create(
                backbone_type=bb_type,
                backbone_variant=bb_variant,
                pretrained=False
            )
            # Simple head
            meta = bb.metadata()
            out_dim = meta.channels[-1]
            head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_dim, 100)
            )
            model = nn.Sequential(bb, head) # Just wraps list output? No, needs adapter.
            
            # Simple wrapper for list output of backbone
            class Wrapper(nn.Module):
                def __init__(self, bb, head):
                    super().__init__()
                    self.bb = bb
                    self.head = head
                def forward(self, x):
                    feats = self.bb(x)
                    return self.head(feats[-1])
            model = Wrapper(bb, head)

        latency, throughput, params = benchmark_model(model, device=device)
        print(f"{name:<30} | {params/1e6:<10.2f} | {latency:<12.2f} | {throughput:<20.0f}")

if __name__ == "__main__":
    run_benchmarks()

