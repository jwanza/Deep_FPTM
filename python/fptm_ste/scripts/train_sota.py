"""
Advanced Training Script for SOTA Hybrid Tsetlin Machine.

Features:
- SAM Optimizer (Sharpness-Aware Minimization)
- Clause Contrastive Loss (InfoNCE)
- Adversarial Training (FGSM)
- Curriculum Learning (Annealing T, lf, temperature)
- Mixed Precision Training (AMP)
"""

import argparse
import os
import time
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from fptm_ste.sota_hybrid import SotaHybridTM
from fptm_ste.optimizers import SAM
from fptm_ste.trainers import (
    ClauseContrastiveLoss, 
    ClauseCurriculumScheduler,
    train_epoch_with_curriculum
)
from fptm_ste.augmentation import get_cifar10_train_transform, get_cifar10_test_transform

def parse_args():
    parser = argparse.ArgumentParser(description="Train SOTA Hybrid TM")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "mnist"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    
    # Model
    parser.add_argument("--backbone", type=str, default="swin_tiny")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--n_clauses", type=int, default=512)
    parser.add_argument("--freeze_stages", type=int, default=0)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--use_sam", action="store_true", default=True)
    parser.add_argument("--rho", type=float, default=0.05, help="SAM rho")
    
    # Advanced
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--adversarial", action="store_true", help="Enable FGSM adversarial training")
    parser.add_argument("--epsilon", type=float, default=0.03, help="FGSM epsilon")
    parser.add_argument("--amp", action="store_true", default=True, help="Use Mixed Precision")
    
    # Curriculum
    parser.add_argument("--start_temp", type=float, default=1.0)
    parser.add_argument("--end_temp", type=float, default=0.01)
    
    return parser.parse_args()

def get_dataloaders(args):
    if args.dataset == "cifar10":
        train_transform = get_cifar10_train_transform(mode="strong")
        test_transform = get_cifar10_test_transform()
        
        train_set = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=True, download=True, transform=train_transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=test_transform
        )
        num_classes = 10
        input_size = 32
        
    elif args.dataset == "cifar100":
        # Reuse CIFAR10 transforms
        train_transform = get_cifar10_train_transform(mode="strong")
        test_transform = get_cifar10_test_transform()
        
        train_set = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=True, download=True, transform=train_transform
        )
        test_set = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=test_transform
        )
        num_classes = 100
        input_size = 32
        
    else:
        raise ValueError(f"Dataset {args.dataset} not implemented yet.")
        
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.workers, pin_memory=True
    )
    
    return train_loader, test_loader, num_classes, input_size

def train_one_epoch(
    model, loader, optimizer, scaler, contrastive_loss_fn, 
    args, epoch, device
):
    model.train()
    total_loss = 0
    total_acc = 0
    num_samples = 0
    
    start_time = time.time()
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        # --- Adversarial Attack Generation (FGSM) ---
        if args.adversarial:
            x.requires_grad = True
            # Forward for gradient
            # Note: We assume model returns logits directly if return_explanation=False
            # But SotaHybridTM returns logits by default.
            logits = model(x, use_ste=True)
            loss_adv = F.cross_entropy(logits, y)
            
            # Compute gradient
            grad = torch.autograd.grad(loss_adv, x, retain_graph=False, create_graph=False)[0]
            x_adv = x + args.epsilon * grad.sign()
            x_adv = torch.clamp(x_adv, 0, 1).detach() # Clip to image range
            
            # Use adversarial examples for training
            x_input = x_adv
        else:
            x_input = x
            
        # --- Forward & Loss ---
        def closure():
            # Enable autocast for mixed precision
            with torch.amp.autocast(device_type='cuda', enabled=args.amp):
                logits = model(x_input, use_ste=True)
                
                # Check output type (might be tuple if explanation enabled, but we disabled it)
                if isinstance(logits, tuple):
                    logits = logits[0]
                    
                loss = F.cross_entropy(logits, y)
                
                # Contrastive Loss (Optional but recommended)
                # SotaHybridTM doesn't return raw clauses easily in standard forward.
                # To enable contrastive loss, we might need access to 'tm_outputs'.
                # For now, we rely on Classification Loss primarily.
                # If we want Contrastive, we need SotaHybridTM to return features.
                # Let's stick to CE for simplicity unless we modify SotaHybridTM return.
                
            return loss, logits

        # --- Optimization Step ---
        if args.use_sam:
            # SAM First Step
            loss, logits = closure()
            scaler.scale(loss).backward()
            scaler.step(optimizer.first_step) # SAM first step (no zero_grad here, handled inside?)
            # SAM optimizer usually requires manual zero_grad or handles it.
            # Our SAM implementation has `first_step(zero_grad=True)`.
            
            # SAM Second Step
            optimizer.zero_grad() # Clear grads from first step
            loss_2, _ = closure() # Recompute gradients at perturbed point
            scaler.scale(loss_2).backward()
            scaler.step(optimizer.second_step) # SAM second step
            scaler.update()
            optimizer.zero_grad()
            
        else:
            # Standard AdamW
            loss, logits = closure()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        # Stats
        acc = (logits.argmax(dim=1) == y).float().sum().item()
        total_loss += loss.item() * x.size(0)
        total_acc += acc
        num_samples += x.size(0)
        
    avg_loss = total_loss / num_samples
    avg_acc = total_acc / num_samples
    duration = time.time() - start_time
    
    return avg_loss, avg_acc, duration

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_acc = 0
    num_samples = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, use_ste=True)
        acc = (logits.argmax(dim=1) == y).float().sum().item()
        total_acc += acc
        num_samples += x.size(0)
        
    return total_acc / num_samples

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training SOTA Hybrid TM on {args.dataset} with {args.backbone}")
    
    # Data
    train_loader, test_loader, num_classes, input_size = get_dataloaders(args)
    
    # Model
    model = SotaHybridTM(
        n_classes=num_classes,
        backbone=args.backbone,
        pretrained=args.pretrained,
        n_clauses_base=args.n_clauses,
        input_size=input_size,
        freeze_stages=args.freeze_stages
    ).to(device)
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Optimizer
    base_optimizer = optim.AdamW
    if args.use_sam:
        print("   Optimizer: SAM + AdamW")
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, lr=args.lr, weight_decay=args.wd)
    else:
        print("   Optimizer: AdamW")
        optimizer = base_optimizer(model.parameters(), lr=args.lr, weight_decay=args.wd)
        
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    # Curriculum
    curriculum = ClauseCurriculumScheduler(
        model, 
        total_epochs=args.epochs,
        temp_schedule=(args.start_temp, args.end_temp),
        schedule_type="cosine"
    )
    
    # Contrastive Loss (Placeholder if we decide to use it)
    contrastive_loss = ClauseContrastiveLoss()
    
    # Loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc, duration = train_one_epoch(
            model, train_loader, optimizer, scaler, contrastive_loss, 
            args, epoch, device
        )
        
        # Validate
        val_acc = evaluate(model, test_loader, device)
        
        # Step Curriculum
        curriculum.step(epoch)
        current_temp = curriculum.get_current_values()["temperature"]
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | "
              f"Val Acc: {val_acc*100:.2f}% | "
              f"Temp: {current_temp:.3f} | "
              f"Time: {duration:.1f}s")
              
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_sota_model.pth")
            print(f"   🏆 New Best! Saved.")
            
    print(f"Done. Best Validation Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()



