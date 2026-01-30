"""Decoder implementation for reconstructing observables from latent state."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ParticleLevelDecoder(nn.Module):
    """Particle-level decoder that predicts residual updates to particle embeddings.
    
    Architecture:
        Δh_i = MLP([z || h_i || c])
        h_i^pred = h_i + Δh_i
    
    The decoder takes the latent state z, individual particle embeddings h_i,
    and conditioning c, and predicts residual updates to each particle embedding.
    
    MLP architecture: [D_latent + D_transformer + D_cond] → [256, 128] → D_transformer
    
    Requirements:
        - Validates: Requirement 4.3
        - Predicts residual updates to particle embeddings
        - Conditions on latent state, particle state, and beam parameters
    """
    
    def __init__(self,
                 latent_dim: int,
                 particle_dim: int,
                 conditioning_dim: int,
                 hidden_dims: Optional[list] = None):
        """Initialize particle-level decoder.
        
        Args:
            latent_dim: Dimension of latent space (D_latent)
            particle_dim: Dimension of particle embeddings (D_transformer)
            conditioning_dim: Dimension of conditioning vector (D_cond)
            hidden_dims: Hidden layer dimensions for MLP (default: [256, 128])
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.particle_dim = particle_dim
        self.conditioning_dim = conditioning_dim
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        # MLP for predicting residual updates
        # Input: [z || h_i || c]
        mlp_input_dim = latent_dim + particle_dim + conditioning_dim
        
        # Build MLP layers
        layers = []
        prev_dim = mlp_input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Output layer: predict residual with same dimension as particle embedding
        layers.append(nn.Linear(prev_dim, particle_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self,
                z: torch.Tensor,
                x: torch.Tensor,
                batch: torch.Tensor,
                conditioning: torch.Tensor) -> torch.Tensor:
        """Predict updated particle embeddings.
        
        Args:
            z: Latent state (batch_size, latent_dim)
            x: Particle embeddings (N, particle_dim)
            batch: Batch assignment for each particle (N,)
            conditioning: Beam parameters (batch_size, conditioning_dim)
        
        Returns:
            Updated particle embeddings h_i^pred = h_i + Δh_i (N, particle_dim)
        """
        # Expand latent state and conditioning to match each particle
        # z is (batch_size, latent_dim), need to map to (N, latent_dim)
        z_expanded = z[batch]  # (N, latent_dim)
        
        # Expand conditioning similarly
        conditioning_expanded = conditioning[batch]  # (N, conditioning_dim)
        
        # Concatenate [z || h_i || c] for each particle
        mlp_input = torch.cat([z_expanded, x, conditioning_expanded], dim=1)  # (N, latent_dim + particle_dim + cond_dim)
        
        # Predict residual update
        delta_h = self.mlp(mlp_input)  # (N, particle_dim)
        
        # Apply residual connection
        h_pred = x + delta_h  # (N, particle_dim)
        
        return h_pred


class ObservableDecoder(nn.Module):
    """Observable decoder that predicts coarse-grained observables from latent state.
    
    Architecture:
        observables = MLP([z || c])
    
    Predicts 5 scalar observables:
        - Beam width: σ_x, σ_y ∈ ℝ²
        - Energy spread: σ_E ∈ ℝ
        - Emittance: ε_x, ε_y ∈ ℝ²
    
    MLP architecture: [D_latent + D_cond] → [128, 64] → 5
    
    Requirements:
        - Validates: Requirements 4.3, 4.4
        - Predicts coarse-grained observables from latent state
        - Outputs beam width, energy spread, and emittance
    """
    
    def __init__(self,
                 latent_dim: int,
                 conditioning_dim: int,
                 num_observables: int = 5,
                 hidden_dims: Optional[list] = None):
        """Initialize observable decoder.
        
        Args:
            latent_dim: Dimension of latent space (D_latent)
            conditioning_dim: Dimension of conditioning vector (D_cond)
            num_observables: Number of observables to predict (default: 5)
            hidden_dims: Hidden layer dimensions for MLP (default: [128, 64])
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.conditioning_dim = conditioning_dim
        self.num_observables = num_observables
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # MLP for predicting observables
        # Input: [z || c]
        mlp_input_dim = latent_dim + conditioning_dim
        
        # Build MLP layers
        layers = []
        prev_dim = mlp_input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Output layer: predict observables
        layers.append(nn.Linear(prev_dim, num_observables))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self,
                z: torch.Tensor,
                conditioning: torch.Tensor) -> torch.Tensor:
        """Predict coarse-grained observables.
        
        Args:
            z: Latent state (batch_size, latent_dim)
            conditioning: Beam parameters (batch_size, conditioning_dim)
        
        Returns:
            Observables (batch_size, num_observables)
            Contains: [σ_x, σ_y, σ_E, ε_x, ε_y]
        """
        # Concatenate latent state with conditioning
        mlp_input = torch.cat([z, conditioning], dim=1)  # (batch_size, latent_dim + cond_dim)
        
        # Predict observables
        observables = self.mlp(mlp_input)  # (batch_size, num_observables)
        
        return observables


class Decoder(nn.Module):
    """Combined decoder with both particle-level and observable prediction heads.
    
    This module combines:
        1. ParticleLevelDecoder: Predicts residual updates to particle embeddings
        2. ObservableDecoder: Predicts coarse-grained observables
    
    Requirements:
        - Validates: Requirements 4.3, 4.4
        - Reconstructs observables from latent state
        - Provides both particle-level and coarse-grained predictions
    """
    
    def __init__(self,
                 latent_dim: int,
                 particle_dim: int,
                 conditioning_dim: int,
                 num_observables: int = 5):
        """Initialize combined decoder.
        
        Args:
            latent_dim: Dimension of latent space (D_latent)
            particle_dim: Dimension of particle embeddings (D_transformer)
            conditioning_dim: Dimension of conditioning vector (D_cond)
            num_observables: Number of observables to predict (default: 5)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.particle_dim = particle_dim
        self.conditioning_dim = conditioning_dim
        self.num_observables = num_observables
        
        # Particle-level decoder
        self.particle_decoder = ParticleLevelDecoder(
            latent_dim=latent_dim,
            particle_dim=particle_dim,
            conditioning_dim=conditioning_dim
        )
        
        # Observable decoder
        self.observable_decoder = ObservableDecoder(
            latent_dim=latent_dim,
            conditioning_dim=conditioning_dim,
            num_observables=num_observables
        )
    
    def forward(self,
                z: torch.Tensor,
                x: torch.Tensor,
                batch: torch.Tensor,
                conditioning: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent state to particle predictions and observables.
        
        Args:
            z: Latent state (batch_size, latent_dim)
            x: Particle embeddings (N, particle_dim)
            batch: Batch assignment for each particle (N,)
            conditioning: Beam parameters (batch_size, conditioning_dim)
        
        Returns:
            Tuple of:
                - particle_pred: Updated particle embeddings (N, particle_dim)
                - observables: Coarse-grained predictions (batch_size, num_observables)
        """
        # Particle-level predictions
        particle_pred = self.particle_decoder(z, x, batch, conditioning)
        
        # Observable predictions
        observables = self.observable_decoder(z, conditioning)
        
        return particle_pred, observables
