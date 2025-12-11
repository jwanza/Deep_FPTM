#!/usr/bin/env python3
"""
Test ConvSTCM2d: Single layer and Deep ResNet-style architectures on MNIST.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import argparse
from typing import List, Optional

from fptm_ste.conv_tm import ConvSTCM2d
from fptm_ste.tm import FuzzyPatternTM_STCM


class SingleConvSTCMClassifier(nn.Module):
    """
    Minimal architecture:
      ConvSTCM2d -> AdaptiveAvgPool -> Flatten -> Linear head
    """
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 10,
        n_clauses: int = 512,
        kernel_size: int = 5,
        out_channels: int = 64,
        operator: str = "capacity",
        ternary_voting: bool = True,
        ternary_band: float = 0.01,
        ste_temperature: float = 1.0,
        ste_gradient_mode: str = "gated_linear",
    ):
        super().__init__()
        self.conv = ConvSTCM2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,  # Same padding
            n_clauses=n_clauses,
            tau=0.5,
            core_backend="stcm",
            operator=operator,
            ternary_voting=ternary_voting,
            ternary_band=ternary_band,
            ste_temperature=ste_temperature,
            ste_gradient_mode=ste_gradient_mode,
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.head = nn.Linear(out_channels * 4 * 4, n_classes)
    
    def forward(self, x, use_ste: bool = True):
        x = self.conv(x, use_ste=use_ste)
        x = torch.sigmoid(x)  # Convert logits to [0,1] features
        x = self.pool(x)
        x = self.flatten(x)
        logits = self.head(x)
        return logits


class ConvSTCMBlock(nn.Module):
    """
    ConvSTCM2d block with optional residual connection.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_clauses: int,
        kernel_size: int = 3,
        stride: int = 1,
        operator: str = "capacity",
        ternary_voting: bool = True,
        ternary_band: float = 0.01,
        ste_gradient_mode: str = "gated_linear",
        use_residual: bool = False,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.conv = ConvSTCM2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            n_clauses=n_clauses,
            tau=0.5,
            core_backend="stcm",
            operator=operator,
            ternary_voting=ternary_voting,
            ternary_band=ternary_band,
            ste_temperature=1.0,
            ste_gradient_mode=ste_gradient_mode,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.1)
        
        # Residual projection if dimensions change
        if use_residual:
            if in_channels != out_channels or stride != 1:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.residual = nn.Identity()
    
    def forward(self, x, use_ste: bool = True):
        out = self.conv(x, use_ste=use_ste)
        out = self.bn(out)
        out = torch.sigmoid(out)  # Activation to [0,1] for next STCM layer
        out = self.dropout(out)
        
        if self.use_residual:
            identity = self.residual(x)
            # Scale residual to match sigmoid range
            out = out + 0.1 * torch.sigmoid(identity)
        
        return out


class DeepConvSTCMResNet(nn.Module):
    """
    Deep architecture using stacked ConvSTCM2d blocks.
    
    Architecture:
      Stem (Conv2d) -> [ConvSTCMBlock x N] -> AdaptiveAvgPool -> Linear
    
    Supports configurable depth, channels, and clauses per layer.
    """
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 10,
        channels: List[int] = [64, 128, 256],
        clauses: List[int] = [512, 1024, 2048],
        kernels: List[int] = [5, 3, 3],
        strides: List[int] = [1, 2, 2],
        operator: str = "capacity",
        ternary_voting: bool = True,
        ternary_band: float = 0.01,
        ste_gradient_mode: str = "gated_linear",
        use_residual: bool = False,
    ):
        super().__init__()
        assert len(channels) == len(clauses) == len(kernels) == len(strides), \
            "channels, clauses, kernels, strides must have same length"
        
        self.n_layers = len(channels)
        self.ternary_band = ternary_band
        
        # Stem: Regular conv to expand channels and map to [0,1] range
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.Sigmoid(),  # Map to [0,1] for STCM input
        )
        
        # Build ConvSTCM blocks
        self.blocks = nn.ModuleList()
        prev_ch = channels[0]
        for i, (ch, cl, k, s) in enumerate(zip(channels, clauses, kernels, strides)):
            block = ConvSTCMBlock(
                in_channels=prev_ch,
                out_channels=ch,
                n_clauses=cl,
                kernel_size=k,
                stride=s,
                operator=operator,
                ternary_voting=ternary_voting,
                ternary_band=ternary_band,
                ste_gradient_mode=ste_gradient_mode,
                use_residual=use_residual,
            )
            self.blocks.append(block)
            prev_ch = ch
        
        # Classifier head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.head = nn.Linear(channels[-1], n_classes)
    
    def set_ternary_band(self, band: float):
        """Update ternary band for all STCM blocks."""
        self.ternary_band = band
        for block in self.blocks:
            if hasattr(block.conv, 'core') and hasattr(block.conv.core, 'ternary_band'):
                block.conv.core.ternary_band = band
    
    def forward(self, x, use_ste: bool = True, return_layer_accs: bool = False):
        x = self.stem(x)
        
        layer_outputs = []
        for block in self.blocks:
            x = block(x, use_ste=use_ste)
            if return_layer_accs:
                layer_outputs.append(x)
        
        x = self.pool(x)
        x = self.flatten(x)
        logits = self.head(x)
        
        if return_layer_accs:
            return logits, layer_outputs
        return logits


class HybridCNNSTCM(nn.Module):
    """
    Hybrid: Regular CNN backbone + ConvSTCM2d feature layer + Linear classifier.
    This leverages CNN's generalization + STCM's ternary feature extraction.
    """
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 10,
        cnn_channels: List[int] = [32, 64],
        stcm_out_channels: int = 128,
        stcm_clauses: int = 1024,
        stcm_kernel: int = 3,
        operator: str = "capacity",
        ternary_voting: bool = True,
        ternary_band: float = 0.01,
        ste_gradient_mode: str = "gated_linear",
    ):
        super().__init__()
        
        # CNN backbone
        layers = []
        prev_ch = in_channels
        for i, ch in enumerate(cnn_channels):
            layers.extend([
                nn.Conv2d(prev_ch, ch, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            prev_ch = ch
        self.backbone = nn.Sequential(*layers)
        
        # Calculate feature map size after backbone (MNIST 28x28 -> 7x7 after 2 pools)
        self.feature_size = 28 // (2 ** len(cnn_channels))
        
        # Single ConvSTCM2d feature extractor (NOT classifier)
        self.stcm_layer = ConvSTCM2d(
            in_channels=cnn_channels[-1],
            out_channels=stcm_out_channels,  # Feature channels, not classes
            kernel_size=stcm_kernel,
            stride=1,
            padding=stcm_kernel // 2,
            n_clauses=stcm_clauses,
            tau=0.5,
            core_backend="stcm",
            operator=operator,
            ternary_voting=ternary_voting,
            ternary_band=ternary_band,
            ste_temperature=1.0,
            ste_gradient_mode=ste_gradient_mode,
        )
        
        # Classifier head
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.head = nn.Linear(stcm_out_channels * 2 * 2, n_classes)
    
    def set_ternary_band(self, band: float):
        if hasattr(self.stcm_layer, 'core') and hasattr(self.stcm_layer.core, 'ternary_band'):
            self.stcm_layer.core.ternary_band = band
    
    def forward(self, x, use_ste: bool = True):
        # CNN features
        x = self.backbone(x)
        # Normalize to [0,1] for STCM
        x = torch.sigmoid(x)
        # STCM feature extraction
        x = self.stcm_layer(x, use_ste=use_ste)
        x = torch.sigmoid(x)  # Activate STCM output
        # Pool and classify
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x)


class SingleConvSTCMDirectClassifier(nn.Module):
    """
    Alternative: ConvSTCM2d outputs directly to n_classes, then global pool.
    This tests whether the STCM voting directly works as classifier.
    """
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 10,
        n_clauses: int = 1024,
        kernel_size: int = 7,
        operator: str = "capacity",
        ternary_voting: bool = True,
        ternary_band: float = 0.01,
        ste_temperature: float = 1.0,
        ste_gradient_mode: str = "gated_linear",
    ):
        super().__init__()
        self.conv = ConvSTCM2d(
            in_channels=in_channels,
            out_channels=n_classes,  # Direct to classes
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            n_clauses=n_clauses,
            tau=0.5,
            core_backend="stcm",
            operator=operator,
            ternary_voting=ternary_voting,
            ternary_band=ternary_band,
            ste_temperature=ste_temperature,
            ste_gradient_mode=ste_gradient_mode,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x, use_ste: bool = True):
        x = self.conv(x, use_ste=use_ste)
        x = self.pool(x)
        return x.view(x.size(0), -1)


def get_mnist_loaders(batch_size=512):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, band_value=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Update ternary band if provided
    if band_value is not None and hasattr(model, 'conv') and hasattr(model.conv, 'core'):
        model.conv.core.ternary_band = band_value
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data, use_ste=True)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data, use_ste=True)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description="ConvSTCM2d Test (Single & Deep)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_clauses", type=int, default=512)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--out_channels", type=int, default=64)
    parser.add_argument("--operator", type=str, default="capacity")
    parser.add_argument("--ternary_voting", action="store_true", default=True)
    parser.add_argument("--ternary_band", type=float, default=0.01)
    parser.add_argument("--ste_gradient_mode", type=str, default="gated_linear")
    parser.add_argument("--band_schedule", action="store_true", help="Use band scheduling 0.0->0.05")
    parser.add_argument("--direct_classifier", action="store_true", help="Use direct classification (no Linear head)")
    # Deep ResNet options
    parser.add_argument("--deep", action="store_true", help="Use DeepConvSTCMResNet instead of single layer")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of ConvSTCM blocks for deep model")
    parser.add_argument("--channels", type=str, default="64,128,256", help="Comma-separated channel counts per layer")
    parser.add_argument("--clauses", type=str, default="512,1024,2048", help="Comma-separated clause counts per layer")
    parser.add_argument("--kernels", type=str, default="5,3,3", help="Comma-separated kernel sizes per layer")
    parser.add_argument("--strides", type=str, default="1,2,2", help="Comma-separated strides per layer")
    # Hybrid CNN + STCM
    parser.add_argument("--hybrid", action="store_true", help="Use HybridCNNSTCM (CNN backbone + STCM head)")
    parser.add_argument("--cnn_channels", type=str, default="32,64", help="Comma-separated CNN backbone channels")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_loader, test_loader = get_mnist_loaders(args.batch_size)
    
    if args.hybrid:
        # Hybrid CNN + STCM
        cnn_channels = [int(x) for x in args.cnn_channels.split(',')]
        model = HybridCNNSTCM(
            in_channels=1,
            n_classes=10,
            cnn_channels=cnn_channels,
            stcm_clauses=args.n_clauses,
            stcm_kernel=args.kernel_size,
            operator=args.operator,
            ternary_voting=args.ternary_voting,
            ternary_band=args.ternary_band if not args.band_schedule else 0.0,
            ste_gradient_mode=args.ste_gradient_mode,
        ).to(device)
        print(f"Model: HybridCNNSTCM (CNN backbone: {cnn_channels} + STCM head: {args.n_clauses} clauses)")
    elif args.deep:
        # Parse layer configs
        channels = [int(x) for x in args.channels.split(',')]
        clauses = [int(x) for x in args.clauses.split(',')]
        kernels = [int(x) for x in args.kernels.split(',')]
        strides = [int(x) for x in args.strides.split(',')]
        
        model = DeepConvSTCMResNet(
            in_channels=1,
            n_classes=10,
            channels=channels,
            clauses=clauses,
            kernels=kernels,
            strides=strides,
            operator=args.operator,
            ternary_voting=args.ternary_voting,
            ternary_band=args.ternary_band if not args.band_schedule else 0.0,
            ste_gradient_mode=args.ste_gradient_mode,
        ).to(device)
        print(f"Model: DeepConvSTCMResNet ({len(channels)} layers)")
        print(f"  Channels: {channels}")
        print(f"  Clauses:  {clauses}")
        print(f"  Kernels:  {kernels}")
        print(f"  Strides:  {strides}")
    elif args.direct_classifier:
        model = SingleConvSTCMDirectClassifier(
            in_channels=1,
            n_classes=10,
            n_clauses=args.n_clauses,
            kernel_size=args.kernel_size,
            operator=args.operator,
            ternary_voting=args.ternary_voting,
            ternary_band=args.ternary_band if not args.band_schedule else 0.0,
            ste_temperature=1.0,
            ste_gradient_mode=args.ste_gradient_mode,
        ).to(device)
        print(f"Model: SingleConvSTCMDirectClassifier (kernel={args.kernel_size}, clauses={args.n_clauses})")
    else:
        model = SingleConvSTCMClassifier(
            in_channels=1,
            n_classes=10,
            n_clauses=args.n_clauses,
            kernel_size=args.kernel_size,
            out_channels=args.out_channels,
            operator=args.operator,
            ternary_voting=args.ternary_voting,
            ternary_band=args.ternary_band if not args.band_schedule else 0.0,
            ste_temperature=1.0,
            ste_gradient_mode=args.ste_gradient_mode,
        ).to(device)
        print(f"Model: SingleConvSTCMClassifier (kernel={args.kernel_size}, out_channels={args.out_channels}, clauses={args.n_clauses})")
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Gradient mode: {args.ste_gradient_mode}, Band schedule: {args.band_schedule}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(args.epochs):
        start = time.time()
        
        # Band scheduling: 0.0 -> 0.05 over epochs
        if args.band_schedule:
            band = 0.0 + (0.05 - 0.0) * min(epoch / (args.epochs - 1), 1.0)
            if hasattr(model, 'set_ternary_band'):
                model.set_ternary_band(band)
        else:
            band = args.ternary_band
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, band if not args.deep else None)
        test_acc = evaluate(model, test_loader, device)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        print(f"Epoch {epoch+1:2d}/{args.epochs} | Loss: {train_loss:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Best: {best_acc:.2f}% | Band: {band:.4f} | Time: {time.time()-start:.1f}s")
    
    print(f"\n=== Final Best Test Accuracy: {best_acc:.2f}% ===")


if __name__ == "__main__":
    main()

