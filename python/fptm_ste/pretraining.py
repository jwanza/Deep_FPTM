"""
Self-Supervised Pre-Training for Tsetlin Machines.

This module implements pre-training strategies for TM models:
1. Masked Clause Modeling (MCM): Predict masked clause outputs
2. Contrastive Pre-Training: SimCLR/BYOL-style clause representation learning
3. Reconstruction Pre-Training: Reconstruct input from clause representations

These methods help TMs learn better clause representations before fine-tuning.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tm import FuzzyPatternTM_STCM


class MaskedClauseModeling(nn.Module):
    """
    Masked Clause Modeling (MCM) pre-training.
    
    Similar to BERT's masked language modeling, but for clause outputs.
    During pre-training, some clause outputs are masked and the model
    learns to predict them from the unmasked clauses.
    
    Args:
        base_model: STCM model to pre-train
        mask_ratio: Fraction of clauses to mask
        prediction_head: Type of prediction head
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        mask_ratio: float = 0.15,
        prediction_head: str = "mlp",
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.base_model = base_model
        self.mask_ratio = mask_ratio
        
        # Get clause dimension from model
        if hasattr(base_model, 'n_clauses'):
            n_clauses = base_model.n_clauses
        else:
            n_clauses = 128  # Default
        
        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.randn(1, 1) * 0.02)
        
        # Prediction head
        if prediction_head == "mlp":
            self.prediction_head = nn.Sequential(
                nn.Linear(n_clauses, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_clauses),
            )
        elif prediction_head == "linear":
            self.prediction_head = nn.Linear(n_clauses, n_clauses)
        else:
            raise ValueError(f"Unknown prediction_head: {prediction_head}")
        
        # Context encoder for masked prediction
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=1,  # Each clause is a scalar
                nhead=1,
                dim_feedforward=64,
                batch_first=True,
            ),
            num_layers=2,
        )
    
    def create_mask(
        self,
        batch_size: int,
        n_clauses: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create random mask for clauses."""
        mask = torch.rand(batch_size, n_clauses, device=device) < self.mask_ratio
        # Ensure at least one clause is masked per sample
        mask[:, 0] = mask[:, 0] | (mask.sum(dim=1) == 0)
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        return_predictions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass for MCM pre-training.
        
        Args:
            x: Input tensor [batch, n_features]
            return_predictions: Whether to return predictions dict
            
        Returns:
            loss or (loss, predictions_dict)
        """
        # Get clause outputs from base model
        output = self.base_model(x)
        if isinstance(output, tuple):
            _, clause_outputs = output
        else:
            clause_outputs = output
        
        batch_size, n_clauses = clause_outputs.shape
        device = clause_outputs.device
        
        # Create mask
        mask = self.create_mask(batch_size, n_clauses, device)
        
        # Apply mask (replace masked positions with mask token)
        clause_expanded = clause_outputs.unsqueeze(-1)  # [batch, n_clauses, 1]
        
        # Create masked version without in-place operation
        mask_expanded = mask.unsqueeze(-1)  # [batch, n_clauses, 1]
        mask_token_expanded = self.mask_token.expand_as(clause_expanded)
        masked_clauses = torch.where(mask_expanded, mask_token_expanded, clause_expanded)
        
        # Encode context
        context = self.context_encoder(masked_clauses)
        context = context.squeeze(-1)  # [batch, n_clauses]
        
        # Predict masked clauses
        predictions = self.prediction_head(context)
        
        # Compute loss only on masked positions
        loss = F.mse_loss(
            predictions[mask],
            clause_outputs[mask],
        )
        
        if return_predictions:
            return loss, {
                "predictions": predictions,
                "targets": clause_outputs,
                "mask": mask,
            }
        
        return loss


class ContrastivePretraining(nn.Module):
    """
    Contrastive pre-training for TM clause representations.
    
    Uses SimCLR-style contrastive learning to learn clause representations
    that are similar for augmented views of the same input.
    
    Args:
        base_model: STCM model to pre-train
        projection_dim: Dimension of projection head output
        temperature: Temperature for InfoNCE loss
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.temperature = temperature
        
        # Get clause dimension
        if hasattr(base_model, 'n_clauses'):
            n_clauses = base_model.n_clauses
        else:
            n_clauses = 128
        
        # Projection head (following SimCLR)
        self.projection = nn.Sequential(
            nn.Linear(n_clauses, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )
    
    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        """Get clause representations and project them."""
        output = self.base_model(x)
        if isinstance(output, tuple):
            _, clause_outputs = output
        else:
            clause_outputs = output
        
        # Project
        z = self.projection(clause_outputs)
        z = F.normalize(z, dim=1)
        
        return z
    
    def info_nce_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE loss between two sets of representations."""
        batch_size = z1.shape[0]
        device = z1.device
        
        # Concatenate representations
        z = torch.cat([z1, z2], dim=0)  # [2*batch, dim]
        
        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # [2*batch, 2*batch]
        
        # Create labels (positive pairs are i and i+batch)
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, float('-inf'))
        
        # Cross-entropy loss
        loss = F.cross_entropy(sim, labels)
        
        return loss
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for contrastive pre-training.
        
        Args:
            x1: First augmented view [batch, n_features]
            x2: Second augmented view [batch, n_features]
            
        Returns:
            Contrastive loss
        """
        z1 = self.get_representations(x1)
        z2 = self.get_representations(x2)
        
        loss = self.info_nce_loss(z1, z2)
        
        return loss


class BYOLPretraining(nn.Module):
    """
    BYOL-style pre-training for TM (Bootstrap Your Own Latent).
    
    Uses a momentum encoder and predictor to learn without negative pairs.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        projection_dim: int = 256,
        hidden_dim: int = 512,
        momentum: float = 0.99,
    ):
        super().__init__()
        self.momentum = momentum
        
        # Online network
        self.online_encoder = base_model
        self.online_projector = self._make_projector(base_model, projection_dim, hidden_dim)
        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        
        # Target network (momentum updated)
        self.target_encoder = self._copy_encoder(base_model)
        self.target_projector = self._make_projector(base_model, projection_dim, hidden_dim)
        
        # Stop gradients for target
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
    
    def _get_n_clauses(self, model: nn.Module) -> int:
        if hasattr(model, 'n_clauses'):
            return model.n_clauses
        return 128
    
    def _make_projector(
        self,
        model: nn.Module,
        projection_dim: int,
        hidden_dim: int,
    ) -> nn.Module:
        n_clauses = self._get_n_clauses(model)
        return nn.Sequential(
            nn.Linear(n_clauses, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
        )
    
    def _copy_encoder(self, encoder: nn.Module) -> nn.Module:
        """Create a copy of the encoder."""
        import copy
        return copy.deepcopy(encoder)
    
    @torch.no_grad()
    def update_target(self):
        """Update target network with momentum."""
        for online_params, target_params in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_params.data = (
                self.momentum * target_params.data +
                (1 - self.momentum) * online_params.data
            )
        
        for online_params, target_params in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters(),
        ):
            target_params.data = (
                self.momentum * target_params.data +
                (1 - self.momentum) * online_params.data
            )
    
    def _get_clauses(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        output = model(x)
        if isinstance(output, tuple):
            return output[1]
        return output
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for BYOL pre-training.
        
        Args:
            x1: First augmented view
            x2: Second augmented view
            
        Returns:
            BYOL loss
        """
        # Online network forward
        online_clauses1 = self._get_clauses(self.online_encoder, x1)
        online_clauses2 = self._get_clauses(self.online_encoder, x2)
        
        online_z1 = self.online_projector(online_clauses1)
        online_z2 = self.online_projector(online_clauses2)
        
        online_p1 = self.online_predictor(online_z1)
        online_p2 = self.online_predictor(online_z2)
        
        # Target network forward (no gradients)
        with torch.no_grad():
            target_clauses1 = self._get_clauses(self.target_encoder, x1)
            target_clauses2 = self._get_clauses(self.target_encoder, x2)
            
            target_z1 = self.target_projector(target_clauses1)
            target_z2 = self.target_projector(target_clauses2)
        
        # BYOL loss: predict target from online
        loss = (
            self._regression_loss(online_p1, target_z2) +
            self._regression_loss(online_p2, target_z1)
        ) / 2
        
        return loss
    
    def _regression_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Normalized MSE loss."""
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        return 2 - 2 * (pred * target).sum(dim=-1).mean()


class ReconstructionPretraining(nn.Module):
    """
    Reconstruction-based pre-training.
    
    Trains the TM to reconstruct the input from clause representations,
    encouraging clauses to capture meaningful patterns.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        n_features: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.base_model = base_model
        
        n_clauses = base_model.n_clauses if hasattr(base_model, 'n_clauses') else 128
        
        # Decoder to reconstruct input
        self.decoder = nn.Sequential(
            nn.Linear(n_clauses, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
            nn.Sigmoid(),  # Assume input is normalized to [0, 1]
        )
    
    def forward(
        self,
        x: torch.Tensor,
        noise_std: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for reconstruction pre-training.
        
        Args:
            x: Input tensor [batch, n_features]
            noise_std: Noise to add to clause outputs (denoising)
            
        Returns:
            (reconstruction_loss, reconstructed_input)
        """
        # Get clause outputs
        output = self.base_model(x)
        if isinstance(output, tuple):
            _, clause_outputs = output
        else:
            clause_outputs = output
        
        # Optionally add noise (denoising autoencoder)
        if noise_std > 0 and self.training:
            clause_outputs = clause_outputs + torch.randn_like(clause_outputs) * noise_std
        
        # Reconstruct input
        reconstructed = self.decoder(clause_outputs)
        
        # Reconstruction loss
        loss = F.mse_loss(reconstructed, x)
        
        return loss, reconstructed


class PretrainingWrapper(nn.Module):
    """
    Wrapper that combines multiple pre-training objectives.
    
    Allows training with multiple self-supervised losses simultaneously.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        n_features: int,
        use_mcm: bool = True,
        use_contrastive: bool = True,
        use_reconstruction: bool = False,
        mcm_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        reconstruction_weight: float = 0.5,
    ):
        super().__init__()
        self.base_model = base_model
        
        self.use_mcm = use_mcm
        self.use_contrastive = use_contrastive
        self.use_reconstruction = use_reconstruction
        
        self.mcm_weight = mcm_weight
        self.contrastive_weight = contrastive_weight
        self.reconstruction_weight = reconstruction_weight
        
        if use_mcm:
            self.mcm = MaskedClauseModeling(base_model)
        
        if use_contrastive:
            self.contrastive = ContrastivePretraining(base_model)
        
        if use_reconstruction:
            self.reconstruction = ReconstructionPretraining(base_model, n_features)
    
    def forward(
        self,
        x: torch.Tensor,
        x_aug: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with all active pre-training objectives.
        
        Args:
            x: Input tensor
            x_aug: Augmented view (for contrastive)
            
        Returns:
            Dict with loss components
        """
        losses = {}
        total_loss = 0.0
        
        if self.use_mcm:
            mcm_loss = self.mcm(x)
            losses['mcm'] = mcm_loss
            total_loss = total_loss + self.mcm_weight * mcm_loss
        
        if self.use_contrastive and x_aug is not None:
            contrastive_loss = self.contrastive(x, x_aug)
            losses['contrastive'] = contrastive_loss
            total_loss = total_loss + self.contrastive_weight * contrastive_loss
        
        if self.use_reconstruction:
            recon_loss, _ = self.reconstruction(x)
            losses['reconstruction'] = recon_loss
            total_loss = total_loss + self.reconstruction_weight * recon_loss
        
        losses['total'] = total_loss
        
        return losses


def pretrain_tm(
    model: nn.Module,
    dataloader,
    epochs: int = 100,
    lr: float = 1e-3,
    pretraining_type: str = "mcm",
    device: torch.device = None,
    verbose: bool = True,
) -> Dict:
    """
    Utility function to pre-train a TM model.
    
    Args:
        model: TM model to pre-train
        dataloader: Data loader (no labels needed)
        epochs: Number of pre-training epochs
        lr: Learning rate
        pretraining_type: 'mcm', 'contrastive', or 'combined'
        device: Device to use
        verbose: Print progress
        
    Returns:
        Dict with training history
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Create pre-training wrapper
    if pretraining_type == "mcm":
        pretrain_model = MaskedClauseModeling(model)
    elif pretraining_type == "contrastive":
        pretrain_model = ContrastivePretraining(model)
    else:
        # Get n_features from dataloader
        sample = next(iter(dataloader))[0]
        n_features = sample.shape[-1] if sample.dim() > 1 else sample.shape[0]
        pretrain_model = PretrainingWrapper(
            model, n_features,
            use_mcm=True,
            use_contrastive=True,
        )
    
    pretrain_model = pretrain_model.to(device)
    optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=lr)
    
    history = {'loss': []}
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            
            # Flatten if needed
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            
            optimizer.zero_grad()
            
            if pretraining_type == "contrastive":
                # Create augmented view (simple noise)
                x_aug = x + torch.randn_like(x) * 0.1
                loss = pretrain_model(x, x_aug)
            elif isinstance(pretrain_model, PretrainingWrapper):
                x_aug = x + torch.randn_like(x) * 0.1
                losses = pretrain_model(x, x_aug)
                loss = losses['total']
            else:
                loss = pretrain_model(x)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        history['loss'].append(avg_loss)
        
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch + 1}/{epochs}: loss = {avg_loss:.4f}")
    
    return history

