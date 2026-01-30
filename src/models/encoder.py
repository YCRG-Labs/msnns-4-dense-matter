"""Encoder implementation for mapping particle-level to latent representations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from typing import Optional


class GlobalPooling(nn.Module):
    """Global pooling operations for aggregating particle-level information.
    
    Implements three types of pooling:
    - Mean pooling: z_mean = (1/N) Σ_i h_i
    - Max pooling: z_max = max_i h_i
    - Sum pooling: z_sum = Σ_i h_i
    
    The pooled features are concatenated: z_pool = [z_mean || z_max || z_sum]
    
    Requirements:
        - Validates: Requirement 4.2
        - Aggregates particle-level information to beam-level representation
        - Maintains permutation invariance through symmetric operations
    """
    
    def __init__(self):
        """Initialize global pooling module."""
        super().__init__()
    
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply global pooling to particle embeddings.
        
        Args:
            x: Particle embeddings (N, D_transformer)
            batch: Batch assignment for each particle (N,). If None, assumes single batch.
            
        Returns:
            Pooled features (batch_size, 3 * D_transformer)
            Concatenation of [mean || max || sum] pooling
        """
        if batch is None:
            # Single batch case
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        # Get number of batches
        batch_size = int(batch.max().item()) + 1
        
        # Mean pooling: (1/N) Σ_i h_i
        # Use scatter with mean reduction
        z_mean = scatter(x, batch, dim=0, dim_size=batch_size, reduce='mean')  # (batch_size, D)
        
        # Max pooling: max_i h_i
        # Use scatter with max reduction
        z_max = scatter(x, batch, dim=0, dim_size=batch_size, reduce='max')  # (batch_size, D)
        
        # Sum pooling: Σ_i h_i
        # Use scatter with sum reduction
        z_sum = scatter(x, batch, dim=0, dim_size=batch_size, reduce='add')  # (batch_size, D)
        
        # Concatenate all pooling results
        z_pool = torch.cat([z_mean, z_max, z_sum], dim=1)  # (batch_size, 3*D)
        
        return z_pool


class Encoder(nn.Module):
    """Encoder for mapping particle-level representations to latent space.
    
    Architecture:
        1. Global pooling (mean, max, sum) of particle embeddings
        2. Concatenate with conditioning vector (density, energy, material)
        3. MLP to produce latent state
    
    The encoder maps from particle-level to collective-level representation:
        z = MLP(GlobalPool(h_1, ..., h_N) || c)
    
    Requirements:
        - Validates: Requirements 4.1, 4.2, 4.5
        - Projects particle representations to low-dimensional latent space
        - Uses global pooling to aggregate particle-level information
        - Conditions on beam parameters (density, energy, material)
    """
    
    def __init__(self, 
                 input_dim: int,
                 latent_dim: int,
                 conditioning_dim: int,
                 hidden_dims: Optional[list] = None):
        """Initialize encoder.
        
        Args:
            input_dim: Dimension of particle embeddings (D_transformer)
            latent_dim: Dimension of latent space (D_latent)
            conditioning_dim: Dimension of conditioning vector (D_cond)
            hidden_dims: Hidden layer dimensions for MLP (default: [512, 256, 128])
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.conditioning_dim = conditioning_dim
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        # Global pooling module
        self.global_pooling = GlobalPooling()
        
        # MLP for encoding to latent space
        # Input: 3 * input_dim (from pooling) + conditioning_dim
        mlp_input_dim = 3 * input_dim + conditioning_dim
        
        # Build MLP layers
        layers = []
        prev_dim = mlp_input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, 
                x: torch.Tensor,
                batch: Optional[torch.Tensor] = None,
                conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode particle embeddings to latent space.
        
        Args:
            x: Particle embeddings (N, input_dim)
            batch: Batch assignment for each particle (N,)
            conditioning: Beam parameters (batch_size, conditioning_dim)
                Contains: [density, energy, material_encoding]
            
        Returns:
            Latent collective state (batch_size, latent_dim)
        """
        # Apply global pooling
        z_pool = self.global_pooling(x, batch)  # (batch_size, 3 * input_dim)
        
        # Concatenate with conditioning if provided
        if conditioning is not None:
            # Ensure conditioning has correct batch size
            batch_size = z_pool.shape[0]
            if conditioning.shape[0] != batch_size:
                raise ValueError(
                    f"Conditioning batch size ({conditioning.shape[0]}) "
                    f"does not match pooled features batch size ({batch_size})"
                )
            
            # Concatenate pooled features with conditioning
            z_input = torch.cat([z_pool, conditioning], dim=1)  # (batch_size, 3*input_dim + cond_dim)
        else:
            # No conditioning provided
            z_input = z_pool
        
        # Pass through MLP to get latent state
        z = self.mlp(z_input)  # (batch_size, latent_dim)
        
        return z
