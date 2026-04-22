#!/usr/bin/env python3
"""
Training script for Parallel DC-FPTM with both-wrong awareness
Implements complete training pipeline with all advanced features
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms

import argparse
import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('..')

from fptm.models.dc_fptm_parallel import ParallelDCFPTM, create_parallel_dc_fptm


class ParallelTrainer:
    """
    Advanced trainer for parallel DC-FPTM with:
    - Both-wrong focused training
    - Complementary path training
    - Performance tracking
    - Adaptive strategies
    """
    
    def __init__(
        self,
        model: ParallelDCFPTM,
        device: str = 'cuda',
        mixed_precision: bool = True
    ):
        self.model = model
        self.device = device
        self.mixed_precision = mixed_precision
        
        if mixed_precision:
            self.scaler = GradScaler()
        
        # Track statistics
        self.stats = {
            'both_wrong_cases': [],
            'disagreement_cases': [],
            'correction_success': [],
            'path_accuracies': {'discrete': [], 'continuous': [], 'ensemble': []},
            'confidence_evolution': []
        }
        
        # Loss components
        self.ce_loss = nn.CrossEntropyLoss()
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        epoch: int,
        args
    ) -> Dict:
        """
        Train for one epoch with all advanced features
        """
        
        self.model.train()
        
        epoch_stats = {
            'loss': 0,
            'acc': 0,
            'both_wrong_count': 0,
            'disagreement_count': 0,
            'correction_count': 0
        }
        
        total_samples = 0
        both_wrong_samples = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.size(0)
            
            # ============================================================
            # Forward pass with all information
            # ============================================================
            
            if self.mixed_precision:
                with autocast():
                    output_dict = self.model(data, return_all=True)
            else:
                output_dict = self.model(data, return_all=True)
            
            final_out = output_dict['final']
            discrete_out = output_dict['discrete']
            continuous_out = output_dict['continuous']
            both_wrong_prob = output_dict['both_wrong_prob']
            
            # ============================================================
            # Identify special cases
            # ============================================================
            
            discrete_pred = discrete_out.argmax(dim=1)
            continuous_pred = continuous_out.argmax(dim=1)
            final_pred = final_out.argmax(dim=1)
            
            discrete_correct = (discrete_pred == target)
            continuous_correct = (continuous_pred == target)
            both_wrong = (~discrete_correct) & (~continuous_correct)
            disagreement = (discrete_pred != continuous_pred)
            
            # Store both-wrong cases for focused training
            if both_wrong.any():
                both_wrong_indices = both_wrong.nonzero(as_tuple=True)[0]
                both_wrong_samples.append({
                    'data': data[both_wrong_indices],
                    'target': target[both_wrong_indices],
                    'discrete_out': discrete_out[both_wrong_indices],
                    'continuous_out': continuous_out[both_wrong_indices]
                })
                epoch_stats['both_wrong_count'] += both_wrong.sum().item()
            
            epoch_stats['disagreement_count'] += disagreement.sum().item()
            
            # ============================================================
            # Compute losses
            # ============================================================
            
            # 1. Standard classification loss
            classification_loss = self.ce_loss(final_out, target)
            
            # 2. Path-specific losses
            discrete_loss = self.ce_loss(discrete_out, target)
            continuous_loss = self.ce_loss(continuous_out, target)
            
            # 3. Complementary training loss (encourage different errors)
            if args.complementary_training:
                # Diversity loss - encourage different predictions when wrong
                discrete_probs = F.softmax(discrete_out, dim=-1)
                continuous_probs = F.softmax(continuous_out, dim=-1)
                
                # KL divergence (want it to be high for wrong samples)
                kl_div = F.kl_div(
                    discrete_probs.log(),
                    continuous_probs.detach(),
                    reduction='none'
                ).sum(dim=-1)
                
                # Only apply to wrong samples
                wrong_mask = ~(final_pred == target)
                diversity_loss = -kl_div[wrong_mask].mean() if wrong_mask.any() else 0
            else:
                diversity_loss = 0
            
            # 4. Both-wrong penalty
            both_wrong_penalty = 0
            if both_wrong.any():
                # Extra penalty when both are wrong
                both_wrong_penalty = (
                    discrete_loss * both_wrong.float() +
                    continuous_loss * both_wrong.float()
                ).mean()
            
            # 5. Confidence calibration loss
            confidence = output_dict.get('confidence', 0.5)
            final_correct = (final_pred == target).float()
            
            # Want confidence to match accuracy
            calibration_loss = F.mse_loss(
                torch.tensor([confidence], device=self.device),
                final_correct.mean().unsqueeze(0)
            )
            
            # ============================================================
            # Total loss
            # ============================================================
            
            total_loss = (
                classification_loss +
                0.2 * (discrete_loss + continuous_loss) +
                0.1 * diversity_loss +
                0.3 * both_wrong_penalty +
                0.1 * calibration_loss
            )
            
            # ============================================================
            # Backward pass
            # ============================================================
            
            optimizer.zero_grad()
            
            if self.mixed_precision:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            
            # ============================================================
            # Update statistics
            # ============================================================
            
            epoch_stats['loss'] += total_loss.item() * batch_size
            epoch_stats['acc'] += final_correct.sum().item()
            total_samples += batch_size
            
            # Store features for nearest neighbor verification
            self.model.store_features(data, target)
            
            # ============================================================
            # Logging
            # ============================================================
            
            if batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {total_loss.item():.6f}\t'
                      f'Acc: {100. * final_correct.mean():.2f}%\t'
                      f'Both Wrong: {both_wrong.sum().item()}/{batch_size}')
        
        # ============================================================
        # Focused training on both-wrong cases
        # ============================================================
        
        if both_wrong_samples and args.focus_both_wrong:
            self.focused_both_wrong_training(
                both_wrong_samples, optimizer, epoch, args
            )
        
        # Normalize statistics
        epoch_stats['loss'] /= total_samples
        epoch_stats['acc'] /= total_samples
        
        return epoch_stats
    
    def focused_both_wrong_training(
        self,
        both_wrong_samples: list,
        optimizer: optim.Optimizer,
        epoch: int,
        args
    ):
        """
        Extra training focused on cases where both models fail
        """
        
        print(f"\nFocused training on {len(both_wrong_samples)} both-wrong batches...")
        
        for sample_batch in both_wrong_samples:
            data = sample_batch['data']
            target = sample_batch['target']
            
            # Forward with correction mechanisms emphasized
            output_dict = self.model(data, return_all=True, use_correction=True)
            
            # Strong supervision for correction
            correction_loss = self.ce_loss(output_dict['final'], target)
            
            # Also train third opinion network specifically
            third_out = self.model.third_opinion(data)
            third_loss = self.ce_loss(third_out, target)
            
            total_loss = correction_loss + 0.5 * third_loss
            
            optimizer.zero_grad()
            
            if self.mixed_precision:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
    
    def evaluate(
        self,
        test_loader: DataLoader,
        epoch: int,
        args
    ) -> Dict:
        """
        Comprehensive evaluation with detailed analysis
        """
        
        self.model.eval()
        
        test_stats = {
            'loss': 0,
            'acc': 0,
            'discrete_acc': 0,
            'continuous_acc': 0,
            'both_right': 0,
            'one_right': 0,
            'both_wrong': 0,
            'corrected': 0,
            'confidence_avg': 0
        }
        
        total_samples = 0
        
        # Class-wise statistics
        class_correct = {i: 0 for i in range(self.model.num_classes)}
        class_total = {i: 0 for i in range(self.model.num_classes)}
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.size(0)
                
                # Get all outputs
                output_dict = self.model(data, return_all=True)
                
                final_out = output_dict['final']
                discrete_out = output_dict['discrete']
                continuous_out = output_dict['continuous']
                both_wrong_prob = output_dict['both_wrong_prob']
                confidence = output_dict['confidence']
                
                # Compute accuracies
                final_pred = final_out.argmax(dim=1)
                discrete_pred = discrete_out.argmax(dim=1)
                continuous_pred = continuous_out.argmax(dim=1)
                
                final_correct = (final_pred == target)
                discrete_correct = (discrete_pred == target)
                continuous_correct = (continuous_pred == target)
                
                # Categorize outcomes
                both_right = discrete_correct & continuous_correct
                one_right = (discrete_correct & ~continuous_correct) | (~discrete_correct & continuous_correct)
                both_wrong = ~discrete_correct & ~continuous_correct
                
                # Check correction success
                corrected = both_wrong & final_correct
                
                # Update statistics
                test_stats['loss'] += F.cross_entropy(final_out, target).item() * batch_size
                test_stats['acc'] += final_correct.sum().item()
                test_stats['discrete_acc'] += discrete_correct.sum().item()
                test_stats['continuous_acc'] += continuous_correct.sum().item()
                test_stats['both_right'] += both_right.sum().item()
                test_stats['one_right'] += one_right.sum().item()
                test_stats['both_wrong'] += both_wrong.sum().item()
                test_stats['corrected'] += corrected.sum().item()
                test_stats['confidence_avg'] += confidence * batch_size
                
                total_samples += batch_size
                
                # Class-wise statistics
                for i in range(batch_size):
                    label = target[i].item()
                    class_total[label] += 1
                    if final_correct[i]:
                        class_correct[label] += 1
        
        # Normalize
        for key in test_stats:
            if key != 'confidence_avg':
                test_stats[key] /= total_samples
        
        # Compute class-wise accuracies
        class_acc = {
            cls: (class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0)
            for cls in range(self.model.num_classes)
        }
        
        # Find problem classes
        problem_classes = [
            cls for cls, acc in class_acc.items() if acc < test_stats['acc'] - 0.1
        ]
        
        # Print detailed results
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Test Results:")
        print(f"{'='*60}")
        print(f"Overall Accuracy: {test_stats['acc']*100:.2f}%")
        print(f"Discrete Path: {test_stats['discrete_acc']*100:.2f}%")
        print(f"Continuous Path: {test_stats['continuous_acc']*100:.2f}%")
        print(f"-"*60)
        print(f"Both Right: {test_stats['both_right']*100:.2f}%")
        print(f"One Right: {test_stats['one_right']*100:.2f}%")
        print(f"Both Wrong: {test_stats['both_wrong']*100:.2f}%")
        print(f"Corrected (of both wrong): {test_stats['corrected']/max(test_stats['both_wrong'], 1e-8)*100:.2f}%")
        print(f"-"*60)
        print(f"Average Confidence: {test_stats['confidence_avg']:.3f}")
        
        if problem_classes:
            print(f"Problem Classes: {problem_classes}")
        
        print(f"{'='*60}")
        
        test_stats['class_acc'] = class_acc
        test_stats['problem_classes'] = problem_classes
        
        return test_stats
    
    def analyze_failure_modes(
        self,
        test_loader: DataLoader
    ) -> Dict:
        """
        Detailed analysis of when and why the model fails
        """
        
        self.model.eval()
        
        failure_analysis = {
            'both_wrong_samples': [],
            'high_confidence_errors': [],
            'disagreement_errors': [],
            'correction_failures': []
        }
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output_dict = self.model(data, return_all=True)
                
                final_pred = output_dict['final'].argmax(dim=1)
                discrete_pred = output_dict['discrete'].argmax(dim=1)
                continuous_pred = output_dict['continuous'].argmax(dim=1)
                confidence = output_dict['confidence']
                
                # Identify failures
                final_wrong = (final_pred != target)
                discrete_wrong = (discrete_pred != target)
                continuous_wrong = (continuous_pred != target)
                both_wrong = discrete_wrong & continuous_wrong
                
                # High confidence errors
                high_conf_errors = final_wrong & (confidence > 0.9)
                
                # Disagreement errors
                disagreement = (discrete_pred != continuous_pred)
                disagreement_errors = final_wrong & disagreement
                
                # Store samples for analysis
                if both_wrong.any():
                    indices = both_wrong.nonzero(as_tuple=True)[0]
                    for idx in indices[:5]:  # Store up to 5 samples
                        failure_analysis['both_wrong_samples'].append({
                            'data': data[idx].cpu(),
                            'target': target[idx].item(),
                            'discrete_pred': discrete_pred[idx].item(),
                            'continuous_pred': continuous_pred[idx].item(),
                            'final_pred': final_pred[idx].item()
                        })
        
        return failure_analysis


def get_data_loaders(args):
    """
    Get data loaders for specified dataset
    """
    
    # Data augmentation for training
    if args.dataset in ['cifar10', 'cifar100', 'svhn']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_transform = train_transform
    
    # Load dataset
    if args.dataset == 'mnist':
        train_dataset = datasets.MNIST(
            'data', train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.MNIST(
            'data', train=False, transform=test_transform
        )
    elif args.dataset == 'fashionmnist':
        train_dataset = datasets.FashionMNIST(
            'data', train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.FashionMNIST(
            'data', train=False, transform=test_transform
        )
    elif args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            'data', train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            'data', train=False, transform=test_transform
        )
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            'data', train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR100(
            'data', train=False, transform=test_transform
        )
    elif args.dataset == 'svhn':
        train_dataset = datasets.SVHN(
            'data', split='train', download=True, transform=train_transform
        )
        test_dataset = datasets.SVHN(
            'data', split='test', download=True, transform=test_transform
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def plot_results(history: Dict, save_path: str):
    """
    Plot training history and analysis
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Accuracy over time
    axes[0, 0].plot(history['train_acc'], label='Train')
    axes[0, 0].plot(history['test_acc'], label='Test')
    axes[0, 0].plot(history['discrete_acc'], label='Discrete', linestyle='--')
    axes[0, 0].plot(history['continuous_acc'], label='Continuous', linestyle='--')
    axes[0, 0].set_title('Accuracy Evolution')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss over time
    axes[0, 1].plot(history['train_loss'], label='Train')
    axes[0, 1].plot(history['test_loss'], label='Test')
    axes[0, 1].set_title('Loss Evolution')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Both-wrong rate
    axes[0, 2].plot(history['both_wrong_rate'], label='Both Wrong', color='red')
    axes[0, 2].plot(history['correction_rate'], label='Corrected', color='green')
    axes[0, 2].set_title('Both-Wrong Detection & Correction')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Rate')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Outcome distribution
    outcomes = ['Both Right', 'One Right', 'Both Wrong']
    values = [
        history['both_right_rate'][-1] if history['both_right_rate'] else 0,
        history['one_right_rate'][-1] if history['one_right_rate'] else 0,
        history['both_wrong_rate'][-1] if history['both_wrong_rate'] else 0
    ]
    axes[1, 0].bar(outcomes, values, color=['green', 'yellow', 'red'])
    axes[1, 0].set_title('Final Outcome Distribution')
    axes[1, 0].set_ylabel('Percentage')
    
    # Confidence distribution
    if 'confidence_history' in history:
        axes[1, 1].hist(history['confidence_history'][-1], bins=20, alpha=0.7)
        axes[1, 1].set_title('Confidence Distribution')
        axes[1, 1].set_xlabel('Confidence')
        axes[1, 1].set_ylabel('Count')
    
    # Class-wise accuracy
    if 'class_acc' in history and history['class_acc']:
        class_acc = history['class_acc'][-1]
        axes[1, 2].bar(range(len(class_acc)), list(class_acc.values()))
        axes[1, 2].set_title('Class-wise Accuracy')
        axes[1, 2].set_xlabel('Class')
        axes[1, 2].set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Results plotted and saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Parallel DC-FPTM')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashionmnist', 'cifar10', 'cifar100', 'svhn'],
                        help='Dataset to use')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=100,
                        help='Test batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Advanced training options
    parser.add_argument('--complementary_training', action='store_true',
                        help='Use complementary training (encourage different errors)')
    parser.add_argument('--focus_both_wrong', action='store_true',
                        help='Extra training on both-wrong cases')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Logging arguments
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Batch logging interval')
    parser.add_argument('--save_dir', type=str, default='checkpoints_parallel',
                        help='Directory to save checkpoints')
    
    # Analysis arguments
    parser.add_argument('--analyze_failures', action='store_true',
                        help='Perform detailed failure analysis')
    parser.add_argument('--plot_results', action='store_true',
                        help='Plot training results')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir) / args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(args)
    
    # Create model
    print("Creating Parallel DC-FPTM model...")
    model = create_parallel_dc_fptm(args.dataset, device=str(device))
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Create trainer
    trainer = ParallelTrainer(model, device=str(device), mixed_precision=args.mixed_precision)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'discrete_acc': [],
        'continuous_acc': [],
        'both_right_rate': [],
        'one_right_rate': [],
        'both_wrong_rate': [],
        'correction_rate': [],
        'confidence_history': [],
        'class_acc': []
    }
    
    best_acc = 0
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_stats = trainer.train_epoch(train_loader, optimizer, epoch, args)
        
        # Evaluate
        test_stats = trainer.evaluate(test_loader, epoch, args)
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_stats['loss'])
        history['train_acc'].append(train_stats['acc'])
        history['test_loss'].append(test_stats['loss'])
        history['test_acc'].append(test_stats['acc'])
        history['discrete_acc'].append(test_stats['discrete_acc'])
        history['continuous_acc'].append(test_stats['continuous_acc'])
        history['both_right_rate'].append(test_stats['both_right'])
        history['one_right_rate'].append(test_stats['one_right'])
        history['both_wrong_rate'].append(test_stats['both_wrong'])
        history['correction_rate'].append(
            test_stats['corrected'] / max(test_stats['both_wrong'], 1e-8)
        )
        history['confidence_history'].append(test_stats['confidence_avg'])
        history['class_acc'].append(test_stats.get('class_acc', {}))
        
        # Save best model
        if test_stats['acc'] > best_acc:
            best_acc = test_stats['acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'history': history
            }, save_dir / 'best_model.pth')
            print(f"Saved best model with accuracy: {best_acc*100:.2f}%")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_stats['loss']:.4f}, Acc: {train_stats['acc']*100:.2f}%")
        print(f"  Test Loss: {test_stats['loss']:.4f}, Acc: {test_stats['acc']*100:.2f}%")
        print(f"  Best Acc: {best_acc*100:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  Time: {epoch_time:.2f}s")
        print("="*60)
    
    # Final analysis
    if args.analyze_failures:
        print("\nPerforming failure analysis...")
        failure_analysis = trainer.analyze_failure_modes(test_loader)
        
        # Save analysis
        with open(save_dir / 'failure_analysis.json', 'w') as f:
            # Convert tensors to lists for JSON serialization
            for key in failure_analysis:
                if isinstance(failure_analysis[key], list):
                    for item in failure_analysis[key]:
                        if 'data' in item and hasattr(item['data'], 'tolist'):
                            item['data'] = item['data'].tolist()
            json.dump(failure_analysis, f, indent=2)
        
        print(f"Failure analysis saved to {save_dir / 'failure_analysis.json'}")
    
    # Plot results
    if args.plot_results:
        plot_results(history, str(save_dir / 'training_results.png'))
    
    # Save final history
    with open(save_dir / 'training_history.json', 'w') as f:
        # Convert numpy/torch values to Python types for JSON
        json_history = {}
        for key, value in history.items():
            if isinstance(value, list) and len(value) > 0:
                if hasattr(value[0], 'item'):
                    json_history[key] = [v.item() for v in value]
                elif isinstance(value[0], np.ndarray):
                    json_history[key] = [v.tolist() for v in value]
                else:
                    json_history[key] = value
            else:
                json_history[key] = value
        json.dump(json_history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_acc*100:.2f}%")
    print(f"Results saved to {save_dir}")


if __name__ == '__main__':
    main()
