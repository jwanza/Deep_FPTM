"""
Universal Vision Training Script - The ONE script for all vision datasets
Supports: MNIST, Fashion-MNIST, CIFAR-10/100, SVHN with full memory optimization
Includes all features: Julia params, attention, smart preprocessing, CPU staging
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import time
import argparse
import sys
import os
import gc
from contextlib import nullcontext

sys.path.append('.')

from fptm.models.fptm_conv_julia import FPTMConvJulia
from fptm.utils import set_seed

# Optional import - only needed for binary feature preprocessing
try:
    from smart_preprocessor import SmartPreprocessor
    HAS_SMART_PREPROCESSOR = True
except ImportError:
    HAS_SMART_PREPROCESSOR = False
    SmartPreprocessor = None


class CPUCachedDataset(Dataset):
    """Dataset that keeps data in CPU memory and streams to GPU."""
    
    def __init__(self, data, labels=None, pin_memory=True):
        """Keep data on CPU, optionally pin memory for faster transfer."""
        # Ensure data is on CPU
        if torch.is_tensor(data) and data.is_cuda:
            data = data.cpu()
        if labels is not None and torch.is_tensor(labels) and labels.is_cuda:
            labels = labels.cpu()
        
        # Pin memory for faster GPU transfer if available
        if pin_memory and torch.cuda.is_available():
            self.data = data.pin_memory() if torch.is_tensor(data) else torch.tensor(data).pin_memory()
            self.labels = labels.pin_memory() if labels is None else (
                labels.pin_memory() if torch.is_tensor(labels) else torch.tensor(labels).pin_memory()
            )
        else:
            self.data = data if torch.is_tensor(data) else torch.tensor(data)
            self.labels = labels if labels is None else (
                labels if torch.is_tensor(labels) else torch.tensor(labels)
            )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx], idx


class UniversalVisionFPTM(nn.Module):
    """
    Universal FPTM for all vision datasets with memory optimization.
    """
    def __init__(self, 
                 input_channels=1,
                 image_size=28,
                 num_classes=10,
                 num_clauses=512,
                 patch_size=4,
                 attention_heads=0,
                 automata_states=50,
                 continuous_mode=False,
                 normalize_mode="minmax",
                 use_color_channels=False,
                 # Julia parameters
                 T=100,
                 s=3.0,
                 L=16,
                 lf=200,
                 include_limit=128,
                 use_julia_eval=False,
                 use_julia_kernels=False,
                 # Memory optimization
                 gradient_checkpointing=False):
        super().__init__()
        
        self.continuous_mode = continuous_mode
        self.input_channels = input_channels
        self.use_color_channels = use_color_channels
        self.gradient_checkpointing = gradient_checkpointing
        
        # Main FPTM configuration
        fptm_kwargs = {
            'image_size': image_size,
            'patch_size': patch_size,
            'num_clauses': num_clauses,
            'num_classes': num_classes,
            'attention_heads': attention_heads,
            'epsilon': 1e-6,
            'automata_states': automata_states,
            'T': T,
            's': s,
            'L': L,
            'lf': lf,
            'include_limit': include_limit,
            'use_julia_eval': use_julia_eval,
            'use_julia_kernels': use_julia_kernels
        }
        
        if continuous_mode:
            fptm_kwargs['normalize_mode'] = normalize_mode
            if use_color_channels and input_channels > 1:
                # Process each color channel separately
                self.channel_processors = nn.ModuleList([
                    FPTMConvJulia(in_channels=1, **{**fptm_kwargs, 'num_clauses': num_clauses // input_channels})
                    for _ in range(input_channels)
                ])
                # Fusion layer
                self.fusion = nn.Sequential(
                    nn.Linear(num_classes * input_channels, num_classes * 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(num_classes * 2, num_classes)
                )
                self.fptm = None
            else:
                # Single channel or grayscale
                self.channel_processors = None
                self.fusion = None
                self.fptm = FPTMConvJulia(
                    in_channels=1 if not use_color_channels and input_channels > 1 else input_channels,
                    **fptm_kwargs
                )
        else:
            # Binary mode
            fptm_kwargs['normalize_mode'] = "none"
            self.channel_mixer = nn.Conv2d(input_channels, 1, kernel_size=1) if input_channels > 1 else None
            self.fptm = FPTMConvJulia(in_channels=1 if input_channels > 1 else input_channels, **fptm_kwargs)
    
    def forward(self, x):
        """Forward with optional gradient checkpointing."""
        if self.gradient_checkpointing and self.training:
            import torch.utils.checkpoint as cp
            return cp.checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x):
        """Actual forward implementation."""
        if self.continuous_mode:
            if self.channel_processors is not None:
                # Process each channel
                outputs = []
                for i, processor in enumerate(self.channel_processors):
                    channel_input = x[:, i:i+1, :, :]
                    if x.shape[1] == 1:  # Grayscale expanded
                        channel_input = x
                    outputs.append(processor(channel_input))
                combined = torch.cat(outputs, dim=1)
                return self.fusion(combined)
            else:
                # Convert RGB to grayscale if needed
                if not self.use_color_channels and x.shape[1] > 1:
                    x = transforms.functional.rgb_to_grayscale(x, num_output_channels=1)
                return self.fptm(x)
        else:
            # Binary mode
            if self.channel_mixer is not None:
                x = self.channel_mixer(x)
            return self.fptm(x)
    
    @torch.no_grad()
    def reinforce(self, x, y_true, y_pred, s=None):
        """Reinforcement learning step."""
        if s is None:
            s = 3.0
            
        if self.continuous_mode and self.channel_processors is not None:
            for i, processor in enumerate(self.channel_processors):
                channel_input = x[:, i:i+1, :, :] if x.shape[1] > 1 else x
                processor.reinforce(channel_input, y_true, y_pred, s)
        elif self.fptm is not None:
            if not self.continuous_mode and self.channel_mixer is not None:
                x = self.channel_mixer(x)
            elif not self.use_color_channels and x.shape[1] > 1:
                x = transforms.functional.rgb_to_grayscale(x, num_output_channels=1)
            self.fptm.reinforce(x, y_true, y_pred, s)


def memory_cleanup(interval=10, batch_idx=0):
    """Clean GPU memory at intervals."""
    if batch_idx % interval == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if batch_idx % (interval * 5) == 0:
                torch.cuda.synchronize()


def train_epoch(model, train_loader, optimizer, criterion, device, epoch,
                gradient_accumulation_steps=1, mixed_precision=False,
                memory_cleanup_interval=10, verbose=True):
    """Memory-optimized training epoch."""
    model.train()
    
    # Mixed precision setup
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    autocast_ctx = torch.cuda.amp.autocast if mixed_precision else nullcontext
    
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()
    
    start_time = time.time()
    
    for batch_idx, (x, y) in enumerate(train_loader):
        # Move to GPU only when needed (non-blocking for speed)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        # Forward pass with optional mixed precision
        with autocast_ctx():
            logits = model(x)
            loss = criterion(logits, y) / gradient_accumulation_steps
        
        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights after gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Reinforcement learning (less frequent to save memory)
        if batch_idx % (gradient_accumulation_steps * 3) == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                model.reinforce(x, y, preds)
        
        # Statistics
        with torch.no_grad():
            running_loss += loss.item() * gradient_accumulation_steps
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
        
        # Memory cleanup
        if memory_cleanup_interval > 0:
            memory_cleanup(memory_cleanup_interval, batch_idx)
            # Delete tensors explicitly at intervals
            if batch_idx % memory_cleanup_interval == 0:
                del x, y, logits, loss
        
        # Progress reporting
        if verbose and batch_idx % 100 == 0 and batch_idx > 0:
            if torch.cuda.is_available():
                mem_gb = torch.cuda.memory_allocated(device) / 1024**3
                max_mem_gb = torch.cuda.max_memory_reserved(device) / 1024**3
                print(f"  Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss={running_loss/(batch_idx+1):.3f}, "
                      f"Acc={100*correct/total:.1f}%, "
                      f"Mem={mem_gb:.1f}/{max_mem_gb:.1f}GB")
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    epoch_time = time.time() - start_time
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    return train_loss, train_acc, epoch_time


def evaluate(model, test_loader, device, mixed_precision=False):
    """Memory-efficient evaluation."""
    model.eval()
    
    autocast_ctx = torch.cuda.amp.autocast if mixed_precision else nullcontext
    
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with autocast_ctx():
                logits = model(x)
                loss = criterion(logits, y)
            
            running_loss += loss.item()
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
            
            # Periodic cleanup
            if batch_idx % 20 == 0:
                del x, y, logits, loss
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    return test_loss, test_acc


def get_dataset(dataset_name, data_dir='./data', augmentation=False):
    """Get dataset with appropriate transforms."""
    
    # Define transforms for different datasets
    if dataset_name in ['fashionmnist', 'mnist']:
        input_channels = 1
        image_size = 28
        num_classes = 10
        
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip() if augmentation else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        if dataset_name == 'fashionmnist':
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform_train)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, transform=transform_test)
        else:  # mnist
            train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform_train)
            test_dataset = datasets.MNIST(data_dir, train=False, transform=transform_test)
    
    # =============================================================================
    # MedMNIST 2D Datasets (28×28)
    # =============================================================================
    elif dataset_name in ['pathmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 
                          'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist',
                          'organamnist', 'organcmnist', 'organsmnist', 'chestmnist']:
        import medmnist
        from medmnist import INFO
        from torch.utils.data import Dataset
        import torch
        
        # Wrapper to squeeze MedMNIST labels from (N, 1) to (N,)
        class MedMNISTWrapper(Dataset):
            def __init__(self, dataset, task='single-label'):
                self.dataset = dataset
                self.task = task
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                img, label = self.dataset[idx]
                # For single-label tasks, squeeze (1,) -> scalar
                if self.task != 'multi-label, binary-class':
                    # Convert numpy array to tensor if needed, then squeeze
                    if isinstance(label, np.ndarray):
                        label = torch.from_numpy(label).squeeze().long()
                    else:
                        label = label.squeeze().long()
                return img, label
        
        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info['python_class'])
        
        image_size = 28
        input_channels = info['n_channels']
        num_classes = len(info['label'])
        task = info['task']
        
        # Grayscale datasets (n_channels=1)
        if input_channels == 1:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip() if augmentation else transforms.Lambda(lambda x: x),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        # RGB datasets (n_channels=3)
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip() if augmentation else transforms.Lambda(lambda x: x),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        # Special handling for multi-label classification (chestmnist)
        if task == 'multi-label, binary-class':
            # ChestMNIST requires as_rgb=False to preserve multi-label format
            train_dataset = DataClass(split='train', transform=transform_train, download=True, root=data_dir, as_rgb=False)
            val_dataset = DataClass(split='val', transform=transform_test, download=True, root=data_dir, as_rgb=False)
            test_dataset = DataClass(split='test', transform=transform_test, download=True, root=data_dir, as_rgb=False)
            # Wrap to handle label format
            train_dataset = MedMNISTWrapper(train_dataset, task=task)
            val_dataset = MedMNISTWrapper(val_dataset, task=task)
            test_dataset = MedMNISTWrapper(test_dataset, task=task)
            # For multi-label, we need to merge train+val for training
            from torch.utils.data import ConcatDataset
            train_dataset = ConcatDataset([train_dataset, val_dataset])
        else:
            # Standard binary-class or multi-class datasets
            train_dataset = DataClass(split='train', transform=transform_train, download=True, root=data_dir)
            val_dataset = DataClass(split='val', transform=transform_test, download=True, root=data_dir)
            test_dataset = DataClass(split='test', transform=transform_test, download=True, root=data_dir)
            # Wrap to squeeze labels: (N, 1) -> (N,)
            train_dataset = MedMNISTWrapper(train_dataset, task=task)
            val_dataset = MedMNISTWrapper(val_dataset, task=task)
            test_dataset = MedMNISTWrapper(test_dataset, task=task)
            # Merge train+val for better training
            from torch.utils.data import ConcatDataset
            train_dataset = ConcatDataset([train_dataset, val_dataset])
        
        print(f"📊 {dataset_name.upper()}: {info['task']} | {input_channels}ch | {num_classes} classes | {info['n_samples']}")
    
    # =============================================================================
    # MedMNIST 3D Datasets (28×28×28)
    # =============================================================================
    elif dataset_name in ['adrenalmnist3d', 'fracturemnist3d', 'nodulemnist3d', 
                          'organmnist3d', 'synapsemnist3d', 'vesselmnist3d']:
        import medmnist
        from medmnist import INFO
        from torch.utils.data import Dataset
        import torch
        
        # Wrapper to squeeze MedMNIST labels from (N, 1) to (N,)
        class MedMNISTWrapper(Dataset):
            def __init__(self, dataset, task='single-label'):
                self.dataset = dataset
                self.task = task
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                img, label = self.dataset[idx]
                # For single-label tasks, squeeze (1,) -> scalar
                if self.task != 'multi-label, binary-class':
                    # Convert numpy array to tensor if needed, then squeeze
                    if isinstance(label, np.ndarray):
                        label = torch.from_numpy(label).squeeze().long()
                    else:
                        label = label.squeeze().long()
                return img, label
        
        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info['python_class'])
        
        image_size = 28  # 3D volumes are 28×28×28
        input_channels = 28  # Treat depth as channels for 2D processing
        num_classes = len(info['label'])
        task = info['task']
        
        # Transform for 3D data: convert (28,28,28) -> (28,28,28) as channels
        # We'll treat each 28×28 slice along the depth dimension as a separate channel
        def transform_3d_to_2d(x):
            # x is PIL Image with shape (28, 28, 28)
            # Convert to tensor and reshape
            x = torch.from_numpy(np.array(x)).float()
            if x.ndim == 3:
                # Already (28, 28, 28)
                x = x.permute(2, 0, 1)  # (28, 28, 28) -> channels first
            elif x.ndim == 4:
                # (1, 28, 28, 28) -> squeeze first dim
                x = x.squeeze(0).permute(2, 0, 1)
            # Normalize
            x = (x - x.mean()) / (x.std() + 1e-8)
            return x
        
        transform_train = transforms.Compose([
            transforms.Lambda(transform_3d_to_2d),
        ])
        transform_test = transforms.Compose([
            transforms.Lambda(transform_3d_to_2d),
        ])
        
        train_dataset = DataClass(split='train', transform=transform_train, download=True, root=data_dir)
        val_dataset = DataClass(split='val', transform=transform_test, download=True, root=data_dir)
        test_dataset = DataClass(split='test', transform=transform_test, download=True, root=data_dir)
        
        # Wrap to squeeze labels: (N, 1) -> (N,)
        train_dataset = MedMNISTWrapper(train_dataset, task=task)
        val_dataset = MedMNISTWrapper(val_dataset, task=task)
        test_dataset = MedMNISTWrapper(test_dataset, task=task)
        
        # Merge train+val
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset([train_dataset, val_dataset])
        
        print(f"📊 {dataset_name.upper()} (3D): {info['task']} | {input_channels}ch (depth) | {num_classes} classes | {info['n_samples']}")
    
    elif dataset_name in ['cifar10', 'cifar100']:
        input_channels = 3
        image_size = 32
        num_classes = 10 if dataset_name == 'cifar10' else 100
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4) if augmentation else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip() if augmentation else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        if dataset_name == 'cifar10':
            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
            test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform_test)
        else:
            train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train)
            test_dataset = datasets.CIFAR100(data_dir, train=False, transform=transform_test)
    
    elif dataset_name == 'svhn':
        input_channels = 3
        image_size = 32
        num_classes = 10
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4) if augmentation else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        
        train_dataset = datasets.SVHN(data_dir, split='train', download=True, transform=transform_train)
        test_dataset = datasets.SVHN(data_dir, split='test', transform=transform_test)
    
    elif dataset_name == 'stl10':
        input_channels = 3
        image_size = 96
        num_classes = 10
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=12) if augmentation else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip() if augmentation else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
        ])
        
        train_dataset = datasets.STL10(data_dir, split='train', download=True, transform=transform_train)
        test_dataset = datasets.STL10(data_dir, split='test', transform=transform_test)
    
    elif dataset_name == 'tinyimagenet':
        input_channels = 3
        image_size = 64
        num_classes = 200
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8) if augmentation else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip() if augmentation else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet stats
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # Tiny ImageNet requires manual download
        # Expected structure: data_dir/tiny-imagenet-200/{train,val}/
        import os
        train_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'train')
        val_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'val')
        
        if not os.path.exists(train_dir):
            raise ValueError(f"Tiny ImageNet not found. Please download from http://cs231n.stanford.edu/tiny-imagenet-200.zip and extract to {data_dir}")
        
        train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dataset = datasets.ImageFolder(val_dir, transform=transform_test)
    
    elif dataset_name == 'food101':
        input_channels = 3
        image_size = 224
        num_classes = 101
        
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224) if augmentation else transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip() if augmentation else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        train_dataset = datasets.Food101(data_dir, split='train', download=True, transform=transform_train)
        test_dataset = datasets.Food101(data_dir, split='test', download=True, transform=transform_test)
    
    elif dataset_name == 'stanforddogs':
        input_channels = 3
        image_size = 224
        num_classes = 120
        
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224) if augmentation else transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip() if augmentation else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # Stanford Dogs requires manual download
        import os
        images_dir = os.path.join(data_dir, 'stanford-dogs', 'Images')
        if not os.path.exists(images_dir):
            raise ValueError(f"Stanford Dogs not found. Please download from http://vision.stanford.edu/aditya86/ImageNetDogs/ and extract to {data_dir}/stanford-dogs/")
        
        # Use ImageFolder with train/test split
        from torch.utils.data import random_split
        full_dataset = datasets.ImageFolder(images_dir, transform=transform_train)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        
        # Apply test transform to test_dataset
        test_dataset.dataset.transform = transform_test
    
    elif dataset_name == 'caltech256':
        input_channels = 3
        image_size = 224
        num_classes = 257
        
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224) if augmentation else transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip() if augmentation else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # Caltech256 in torchvision
        full_dataset = datasets.Caltech256(data_dir, download=True, transform=transform_train)
        
        # Split 80/20 train/test
        from torch.utils.data import random_split
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        
        # Apply test transform
        test_dataset.dataset.transform = transform_test
    
    elif dataset_name == 'imagenet':
        input_channels = 3
        image_size = 224
        num_classes = 1000
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224) if augmentation else transforms.Resize(256),
            transforms.CenterCrop(224) if not augmentation else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip() if augmentation else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # ImageNet requires manual download
        import os
        train_dir = os.path.join(data_dir, 'imagenet', 'train')
        val_dir = os.path.join(data_dir, 'imagenet', 'val')
        
        if not os.path.exists(train_dir):
            raise ValueError(f"ImageNet not found. Please download ImageNet ILSVRC2012 and extract to {data_dir}/imagenet/")
        
        train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dataset = datasets.ImageFolder(val_dir, transform=transform_test)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, test_dataset, input_channels, image_size, num_classes


def get_dataset_defaults(dataset_name):
    """Get optimal default configurations based on Julia examples"""
    configs = {
        'mnist': {
            'num_clauses': 200,       # Julia: 20 per class × 10 classes
            'T': 20,
            'automata_states': 100,   # Julia states_num=256, we use half
            'L': 150,
            'lf': 75,
            'include_limit': 90,      # Adjusted for our state range (90% of 100)
            's': 3.9,                # Scaled from Julia S=200 (see note below)
            'batch_size': 256,
            'lr': 0.01,
            'epochs': 50
        },
        'fashionmnist': {
            'num_clauses': 200,       # Julia: 20 per class × 10 classes  
            'T': 100,
            'automata_states': 128,   # Julia states_num=256, we use half
            'L': 200,
            'lf': 200,
            'include_limit': 115,     # Adjusted for our state range (90% of 128)
            's': 4.5,
            'batch_size': 128,
            'lr': 0.005,
            'epochs': 100
        },
        'cifar10': {
            'num_clauses': 200,       # Julia: 20 per class × 10 classes
            'T': 1000,
            'automata_states': 128,   # Julia states_num=256, we use half
            'L': 4000,
            'lf': 4000,
            'include_limit': 115,     # Adjusted for our state range
            's': 5.0,
            'batch_size': 64,
            'lr': 0.005,
            'epochs': 200
        },
        'cifar100': {
            'num_clauses': 2000,      # 20 per class × 100 classes
            'T': 1000,
            'automata_states': 128,   # Julia states_num=256, we use half
            'L': 4000,
            'lf': 4000,
            'include_limit': 115,     # Adjusted for our state range
            's': 5.0,
            'batch_size': 32,
            'lr': 0.005,
            'epochs': 300
        },
        'svhn': {
            'num_clauses': 200,       # Similar to MNIST
            'T': 50,
            'automata_states': 100,   # More reasonable
            'L': 200,
            'lf': 100,
            'include_limit': 150,     # Adjusted for automata_states
            's': 3.5,
            'batch_size': 128,
            'lr': 0.01,
            'epochs': 100
        }
    }
    return configs.get(dataset_name, configs['fashionmnist'])

def main():
    parser = argparse.ArgumentParser(description='Universal Vision Training Script')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='fashionmnist',
                       choices=['mnist', 'fashionmnist', 'cifar10', 'cifar100', 'svhn', 
                               'stl10', 'tinyimagenet', 'food101', 'stanforddogs', 
                               'caltech256', 'imagenet'],
                       help='Dataset to train on')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--use_optimal', action='store_true',
                       help='Use optimal Julia-based configurations for the dataset')
    
    # Model architecture
    parser.add_argument('--num_clauses', type=int, default=512, help='Number of clauses')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size')
    parser.add_argument('--attention_heads', type=int, default=0, help='Number of attention heads')
    parser.add_argument('--automata_states', type=int, default=8, help='Automata states (8 works well!)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    # Julia parameters (Tsetlin Machine specific)
    parser.add_argument('--T', type=int, default=100, help='Decision threshold')
    parser.add_argument('--s', type=float, default=3.5, help='Reinforcement strength') 
    parser.add_argument('--L', type=int, default=20, help='Learning sensitivity')
    parser.add_argument('--lf', type=int, default=200, help='Leakage factor (Julia)')
    parser.add_argument('--include_limit', type=int, default=None, help='Include limit (auto-set if None)')
    parser.add_argument('--use_julia_eval', action='store_true', help='Use Julia-style evaluation')
    parser.add_argument('--use_julia_kernels', action='store_true', help='Use Julia vision kernels')
    
    # Feature mode
    parser.add_argument('--continuous', action='store_true', help='Use continuous mode')
    parser.add_argument('--normalize_mode', type=str, default='minmax',
                       choices=['none', 'minmax', 'global'])
    parser.add_argument('--use_color_channels', action='store_true',
                       help='Process color channels separately')
    
    # Binary mode preprocessing (when not using --continuous)
    parser.add_argument('--num_thresholds', type=int, default=16, help='Number of thresholds for binary mode')
    parser.add_argument('--include_edges', action='store_true', help='Include edge features')
    parser.add_argument('--include_inverted', action='store_true', default=True, help='Include inverted features')
    
    # Memory optimization
    parser.add_argument('--gradient_accumulation', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Use gradient checkpointing')
    parser.add_argument('--memory_cleanup_interval', type=int, default=10, help='Memory cleanup interval (0=disable)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='Pin memory for faster GPU transfer')
    parser.add_argument('--max_gpu_memory_fraction', type=float, default=0.9, help='Maximum GPU memory fraction to use')
    
    # Other
    parser.add_argument('--augmentation', action='store_true', help='Use data augmentation')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Apply optimal configurations if requested
    if args.use_optimal:
        optimal = get_dataset_defaults(args.dataset)
        print(f"\n📚 Loading Julia-optimal configurations for {args.dataset.upper()}")
        
        # Create a set of command line arguments that were explicitly set
        import sys
        explicit_args = set()
        for arg in sys.argv[1:]:
            if arg.startswith('--') and not arg == '--use_optimal':
                arg_name = arg[2:].replace('-', '_').split('=')[0]
                explicit_args.add(arg_name)
        
        # Apply optimal values for non-explicit arguments
        for key, value in optimal.items():
            if hasattr(args, key) and key not in explicit_args:
                setattr(args, key, value)
                
        print(f"   Applied: clauses={args.num_clauses}, states={args.automata_states}, T={args.T}")
        print(f"   L={args.L}, lf={args.lf}, s={args.s}, epochs={args.epochs}")
    
    # Set default include_limit if not provided  
    if args.include_limit is None:
        if args.use_optimal and 'include_limit' in get_dataset_defaults(args.dataset):
            args.include_limit = get_dataset_defaults(args.dataset)['include_limit']
        else:
            args.include_limit = args.automata_states + 1
            
    # Set random seed
    set_seed(args.seed)
    
    # Device and memory setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(args.max_gpu_memory_fraction)
        # Enable TF32 for faster training on A100/4090
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    print("\n" + "="*70)
    print("🌍 UNIVERSAL VISION TRAINING -", args.dataset.upper())
    print("="*70)
    print(f"Mode: {'CONTINUOUS' if args.continuous else 'BINARY'}")
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"        num_clauses={args.num_clauses}, patch_size={args.patch_size}")
    print(f"        attention_heads={args.attention_heads}, automata_states={args.automata_states}")
    
    if not args.continuous:
        print(f"        thresholds={args.num_thresholds}, edges={args.include_edges}, inverted={args.include_inverted}")
    
    if args.use_julia_eval or args.use_julia_kernels:
        print(f"Julia: eval={args.use_julia_eval}, kernels={args.use_julia_kernels}")
        print(f"       T={args.T}, s={args.s}, L={args.L}, lf={args.lf}")
    
    print(f"Memory: grad_accum={args.gradient_accumulation}, mixed_prec={args.mixed_precision}")
    print(f"        workers={args.num_workers}, cleanup_interval={args.memory_cleanup_interval}")
    print("="*70)
    
    # Load dataset
    print(f"\n📊 Loading {args.dataset.upper()} dataset...")
    
    if args.continuous:
        # Use raw images for continuous mode
        train_dataset, test_dataset, input_channels, image_size, num_classes = get_dataset(
            args.dataset, args.data_dir, args.augmentation
        )
        print(f"   Using raw images: {input_channels}×{image_size}×{image_size}")
    else:
        # Use preprocessed binary features
        print(f"   Creating binary features with {args.num_thresholds} thresholds...")
        
        # Check if SmartPreprocessor is available
        if not HAS_SMART_PREPROCESSOR:
            raise ImportError(
                "SmartPreprocessor is required for binary feature preprocessing but is not installed.\n"
                "Either:\n"
                "  1. Use --no_binary --no_conv (continuous mode with raw images), OR\n"
                "  2. Install smart_preprocessor module, OR\n"
                "  3. Use MedMNIST datasets which support raw image mode"
            )
        
        # Get original dataset for labels
        orig_train, orig_test, orig_channels, image_size, num_classes = get_dataset(
            args.dataset, args.data_dir, augmentation=False
        )
        
        # Create smart preprocessor
        preprocessor = SmartPreprocessor(args.dataset)
        
        # Get or create preprocessed features (kept on CPU!)
        train_data = preprocessor.get_or_create_preprocessed(
            'train',
            num_thresholds=args.num_thresholds,
            include_edges=args.include_edges,
            include_inverted=args.include_inverted
        )
        
        test_data = preprocessor.get_or_create_preprocessed(
            'test',
            num_thresholds=args.num_thresholds,
            include_edges=args.include_edges,
            include_inverted=args.include_inverted
        )
        
        # Extract features and labels from the dictionary
        if isinstance(train_data, dict):
            train_features = train_data['features']
            train_labels = train_data['labels']
        else:
            # If it's already a tensor, get labels separately
            train_features = train_data
            train_labels = torch.tensor([orig_train[i][1] for i in range(len(orig_train))])
            
        if isinstance(test_data, dict):
            test_features = test_data['features']
            test_labels = test_data['labels']
        else:
            # If it's already a tensor, get labels separately
            test_features = test_data
            test_labels = torch.tensor([orig_test[i][1] for i in range(len(orig_test))])
        
        # Create CPU-cached datasets
        train_dataset = CPUCachedDataset(train_features, train_labels, pin_memory=args.pin_memory)
        test_dataset = CPUCachedDataset(test_features, test_labels, pin_memory=args.pin_memory)
        
        # Calculate input channels for binary mode
        input_channels = args.num_thresholds * (2 if args.include_inverted else 1)
        if args.include_edges:
            input_channels += 2  # Sobel x and y
        
        print(f"   Binary features: {input_channels} channels")
    
    print(f"   Train: {len(train_dataset):,} samples")
    print(f"   Test: {len(test_dataset):,} samples")
    print(f"   Classes: {num_classes}")
    
    # Create data loaders with memory optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,  # Larger batch for evaluation
        shuffle=False,
        num_workers=args.num_workers // 2 if args.num_workers > 1 else 0,
        pin_memory=args.pin_memory and torch.cuda.is_available()
    )
    
    # Auto-set include_limit if not specified
    if args.include_limit is None:
        args.include_limit = args.automata_states + 1  # Just above midpoint
    
    # Create model
    print(f"\nCreating Universal Vision FPTM...")
    model = UniversalVisionFPTM(
        input_channels=input_channels,
        image_size=image_size,
        num_classes=num_classes,
        num_clauses=args.num_clauses,
        patch_size=args.patch_size,
        attention_heads=args.attention_heads,
        automata_states=args.automata_states,
        continuous_mode=args.continuous,
        normalize_mode=args.normalize_mode if args.continuous else "none",
        use_color_channels=args.use_color_channels,
        # Julia parameters
        T=args.T,
        s=args.s,
        L=args.L,
        lf=args.lf,
        include_limit=args.include_limit,
        use_julia_eval=args.use_julia_eval,
        use_julia_kernels=args.use_julia_kernels,
        # Memory optimization
        gradient_checkpointing=args.gradient_checkpointing
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {total_mem:.1f}GB (using up to {args.max_gpu_memory_fraction*100:.0f}%)")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    best_val_acc = 0
    first_epoch_time = None
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc, epoch_time = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            gradient_accumulation_steps=args.gradient_accumulation,
            mixed_precision=args.mixed_precision,
            memory_cleanup_interval=args.memory_cleanup_interval,
            verbose=args.verbose
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(model, test_loader, device, args.mixed_precision)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Track speedup
        if epoch == 1:
            first_epoch_time = epoch_time
        speedup = first_epoch_time / epoch_time if first_epoch_time else 1.0
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': vars(args),
                'accuracy': val_acc
            }, 'best_model_universal.pt')
            marker = "🔥"
        else:
            marker = ""
        
        # Print progress
        print(f"[{epoch:3d}/{args.epochs}] "
              f"Train: {train_loss:.3f}/{train_acc:.1f}% | "
              f"Val: {val_loss:.3f}/{val_acc:.1f}% | "
              f"LR: {current_lr:.5f} | "
              f"Time: {epoch_time:.1f}s | "
              f"Speed: {speedup:.2f}x {marker}")
        
        # Memory stats every 5 epochs
        if torch.cuda.is_available() and epoch % 5 == 0:
            mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"  Peak GPU memory: {mem_gb:.2f}GB")
            torch.cuda.reset_peak_memory_stats(device)
    
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("-"*70)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Training speedup: {speedup:.2f}x")
    print(f"Model saved to: best_model_universal.pt")
    print("="*70)


if __name__ == "__main__":
    main()