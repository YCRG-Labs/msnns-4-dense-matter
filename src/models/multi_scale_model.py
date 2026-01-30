"""Multi-scale model integrating GNN, Transformer, Encoder, Neural ODE, and Decoder."""

import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass

from .gnn import GNNLayer
from .transformer import TransformerLayer
from .encoder import Encoder
from .neural_ode import NeuralODE
from .decoder import Decoder


@dataclass
class ModelConfig:
    """Configuration for MultiScaleModel.
    
    Attributes:
        # Input dimensions
        input_dim: Dimension of input node features (8 + num_species)
        num_species: Number of particle species (default: 5 for Si, Fe, W, Cu, Al)
        
        # GNN parameters
        gnn_hidden_dim: Hidden dimension for GNN layers
        gnn_num_layers: Number of GNN message passing layers
        gnn_cutoff_radius: Spatial cutoff radius for edge construction (nm)
        
        # Transformer parameters
        transformer_hidden_dim: Hidden dimension for transformer
        transformer_num_heads: Number of attention heads
        transformer_dropout: Dropout probability
        
        # Encoder parameters
        encoder_hidden_dims: Hidden layer dimensions for encoder MLP
        latent_dim: Dimension of latent space
        
        # Neural ODE parameters
        ode_hidden_dims: Hidden layer dimensions for dynamics MLP
        ode_solver: ODE solver method
        ode_rtol: Relative tolerance for ODE solver
        ode_atol: Absolute tolerance for ODE solver
        
        # Decoder parameters
        num_observables: Number of coarse-grained observables to predict
        
        # Conditioning
        conditioning_dim: Dimension of conditioning vector (2 + num_species for density, energy, material)
        
        # Ablation study flags (Requirement 20.3)
        enable_gnn: Enable/disable GNN component
        enable_transformer: Enable/disable Transformer component
        enable_neural_ode: Enable/disable Neural ODE component
    """
    # Input dimensions
    input_dim: int = 13  # 3 (pos) + 3 (mom) + 1 (charge) + 1 (mass) + 5 (species)
    num_species: int = 5
    
    # GNN parameters
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 4
    gnn_cutoff_radius: float = 5.0
    
    # Transformer parameters
    transformer_hidden_dim: int = 128
    transformer_num_heads: int = 8
    transformer_dropout: float = 0.1
    
    # Encoder parameters
    encoder_hidden_dims: Optional[list] = None
    latent_dim: int = 32
    
    # Neural ODE parameters
    ode_hidden_dims: Optional[list] = None
    ode_solver: str = 'dopri5'
    ode_rtol: float = 1e-3
    ode_atol: float = 1e-4
    
    # Decoder parameters
    num_observables: int = 5
    
    # Conditioning
    conditioning_dim: int = 7  # 2 (density, energy) + 5 (material one-hot)
    
    # Ablation study flags (Requirement 20.3)
    enable_gnn: bool = True
    enable_transformer: bool = True
    enable_neural_ode: bool = True
    
    def __post_init__(self):
        """Set default values for optional parameters."""
        if self.encoder_hidden_dims is None:
            self.encoder_hidden_dims = [512, 256, 128]
        if self.ode_hidden_dims is None:
            self.ode_hidden_dims = [256, 256]


class MultiScaleModel(nn.Module):
    """Multi-scale neural network for collective phenomena discovery.
    
    Integrates five hierarchical components:
        1. GNN: Captures local particle interactions
        2. Transformer: Captures long-range correlations
        3. Encoder: Maps to latent collective state
        4. Neural ODE: Evolves collective dynamics
        5. Decoder: Reconstructs observables
    
    Forward pass:
        Input: GraphData with particle states and conditioning
        1. GNN: h^(gnn) = GNN(x, pos, batch)
        2. Transformer: h^(trans) = Transformer(h^(gnn), pos, batch)
        3. Encoder: z(t_0) = Encoder(h^(trans), batch, conditioning)
        4. Neural ODE: z(t_1) = NeuralODE(z(t_0), t_span, conditioning)
        5. Decoder: (particle_pred, observables) = Decoder(z(t_1), h^(trans), batch, conditioning)
    
    Requirements:
        - Validates: Requirements 4.5, 20.1
        - Chains GNN → Transformer → Encoder → Neural ODE → Decoder
        - Returns dictionary with particle predictions, observables, and latent states
        - Supports modular architecture for experimentation
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize multi-scale model.
        
        Args:
            config: Model configuration with all hyperparameters
        """
        super().__init__()
        
        self.config = config
        
        # 1. GNN Layer (or skip connection if disabled)
        if config.enable_gnn:
            self.gnn = GNNLayer(
                input_dim=config.input_dim,
                hidden_dim=config.gnn_hidden_dim,
                num_layers=config.gnn_num_layers,
                cutoff_radius=config.gnn_cutoff_radius
            )
        else:
            # Identity mapping: project input to gnn_hidden_dim
            self.gnn = nn.Linear(config.input_dim, config.gnn_hidden_dim)
        
        # 2. Transformer Layer (or skip connection if disabled)
        if config.enable_transformer:
            self.transformer = TransformerLayer(
                input_dim=config.gnn_hidden_dim,
                hidden_dim=config.transformer_hidden_dim,
                num_heads=config.transformer_num_heads,
                dropout=config.transformer_dropout
            )
        else:
            # Identity mapping: project gnn output to transformer_hidden_dim
            if config.gnn_hidden_dim != config.transformer_hidden_dim:
                self.transformer = nn.Linear(config.gnn_hidden_dim, config.transformer_hidden_dim)
            else:
                self.transformer = nn.Identity()
        
        # 3. Encoder
        self.encoder = Encoder(
            input_dim=config.transformer_hidden_dim,
            latent_dim=config.latent_dim,
            conditioning_dim=config.conditioning_dim,
            hidden_dims=config.encoder_hidden_dims
        )
        
        # 4. Neural ODE (or identity if disabled)
        if config.enable_neural_ode:
            self.neural_ode = NeuralODE(
                latent_dim=config.latent_dim,
                conditioning_dim=config.conditioning_dim,
                hidden_dims=config.ode_hidden_dims,
                solver=config.ode_solver,
                rtol=config.ode_rtol,
                atol=config.ode_atol
            )
        else:
            # Identity mapping: return input latent state unchanged
            self.neural_ode = nn.Identity()
        
        # 5. Decoder
        self.decoder = Decoder(
            latent_dim=config.latent_dim,
            particle_dim=config.transformer_hidden_dim,
            conditioning_dim=config.conditioning_dim,
            num_observables=config.num_observables
        )
    
    def forward(self,
                data,
                t_span: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through multi-scale model.
        
        Args:
            data: GraphData object with attributes:
                - x: Node features (N, input_dim)
                - pos: Particle positions (N, 3)
                - batch: Batch assignment (N,)
                - density: Beam density (batch_size,)
                - energy: Beam energy (batch_size,)
                - material: Material encoding (batch_size, num_species)
            t_span: Time interval [t_start, t_end] (2,)
        
        Returns:
            Dictionary with keys:
                - 'particle_pred': Predicted particle embeddings (N, transformer_hidden_dim)
                - 'observables': Predicted coarse-grained observables (batch_size, num_observables)
                - 'latent_z0': Initial latent state (batch_size, latent_dim)
                - 'latent_z1': Evolved latent state (batch_size, latent_dim)
        
        Raises:
            ValueError: If NaN or Inf detected in forward pass
        """
        # Extract data attributes
        x = data.x
        pos = data.pos
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else None
        
        # Check for NaN/Inf in inputs
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("NaN or Inf detected in input features")
        if torch.isnan(pos).any() or torch.isinf(pos).any():
            raise ValueError("NaN or Inf detected in input positions")
        if torch.isnan(t_span).any() or torch.isinf(t_span).any():
            raise ValueError("NaN or Inf detected in time span")
        
        # Determine batch size
        if batch is not None:
            batch_size = int(batch.max().item()) + 1
        else:
            batch_size = 1
            # Create batch tensor for single batch
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        # Construct conditioning vector from data
        # Conditioning: [density, energy, material_encoding]
        if hasattr(data, 'density') and hasattr(data, 'energy') and hasattr(data, 'material'):
            # Ensure density and energy are 2D (batch_size, 1)
            density = data.density.view(-1, 1) if data.density.dim() == 1 else data.density
            energy = data.energy.view(-1, 1) if data.energy.dim() == 1 else data.energy
            
            # Ensure material is 2D
            material = data.material if data.material.dim() == 2 else data.material.unsqueeze(0)
            
            # Concatenate to form conditioning vector
            conditioning = torch.cat([density, energy, material], dim=1)  # (batch_size, conditioning_dim)
        else:
            # If conditioning not provided, create dummy conditioning
            conditioning = torch.zeros(batch_size, self.config.conditioning_dim, 
                                      device=x.device, dtype=x.dtype)
        
        # 1. GNN: Process local interactions (or skip if disabled)
        if self.config.enable_gnn:
            h_gnn = self.gnn(x, pos, batch)  # (N, gnn_hidden_dim)
        else:
            # Apply linear projection
            h_gnn = self.gnn(x)  # (N, gnn_hidden_dim)
        
        # Check for NaN/Inf after GNN
        if torch.isnan(h_gnn).any() or torch.isinf(h_gnn).any():
            raise ValueError("NaN or Inf detected after GNN layer")
        
        # 2. Transformer: Capture long-range correlations (or skip if disabled)
        if self.config.enable_transformer:
            h_trans = self.transformer(h_gnn, pos, batch)  # (N, transformer_hidden_dim)
        else:
            # Apply identity or linear projection
            h_trans = self.transformer(h_gnn)  # (N, transformer_hidden_dim)
        
        # Check for NaN/Inf after Transformer
        if torch.isnan(h_trans).any() or torch.isinf(h_trans).any():
            raise ValueError("NaN or Inf detected after Transformer layer")
        
        # 3. Encoder: Map to latent space
        z0 = self.encoder(h_trans, batch, conditioning)  # (batch_size, latent_dim)
        
        # Check for NaN/Inf after Encoder
        if torch.isnan(z0).any() or torch.isinf(z0).any():
            raise ValueError("NaN or Inf detected after Encoder")
        
        # 4. Neural ODE: Evolve latent dynamics (or skip if disabled)
        if self.config.enable_neural_ode:
            z1 = self.neural_ode(z0, t_span, conditioning)  # (batch_size, latent_dim)
        else:
            # Identity: no time evolution
            z1 = z0
        
        # Check for NaN/Inf after Neural ODE
        if torch.isnan(z1).any() or torch.isinf(z1).any():
            raise ValueError("NaN or Inf detected after Neural ODE integration")
        
        # 5. Decoder: Reconstruct observables
        particle_pred, observables = self.decoder(z1, h_trans, batch, conditioning)
        
        # Check for NaN/Inf in outputs
        if torch.isnan(particle_pred).any() or torch.isinf(particle_pred).any():
            raise ValueError("NaN or Inf detected in particle predictions")
        if torch.isnan(observables).any() or torch.isinf(observables).any():
            raise ValueError("NaN or Inf detected in observable predictions")
        
        # Return all outputs
        return {
            'particle_pred': particle_pred,
            'observables': observables,
            'latent_z0': z0,
            'latent_z1': z1
        }
    
    def encode(self, data) -> torch.Tensor:
        """Encode particle data to latent space without time evolution.
        
        Useful for analysis and hypothesis generation.
        
        Args:
            data: GraphData object
        
        Returns:
            Latent state z (batch_size, latent_dim)
        """
        x = data.x
        pos = data.pos
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else None
        
        # Determine batch size
        if batch is not None:
            batch_size = int(batch.max().item()) + 1
        else:
            batch_size = 1
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        # Construct conditioning
        if hasattr(data, 'density') and hasattr(data, 'energy') and hasattr(data, 'material'):
            density = data.density.view(-1, 1) if data.density.dim() == 1 else data.density
            energy = data.energy.view(-1, 1) if data.energy.dim() == 1 else data.energy
            material = data.material if data.material.dim() == 2 else data.material.unsqueeze(0)
            conditioning = torch.cat([density, energy, material], dim=1)
        else:
            conditioning = torch.zeros(batch_size, self.config.conditioning_dim,
                                      device=x.device, dtype=x.dtype)
        
        # Process through GNN and Transformer
        if self.config.enable_gnn:
            h_gnn = self.gnn(x, pos, batch)
        else:
            h_gnn = self.gnn(x)
        
        if self.config.enable_transformer:
            h_trans = self.transformer(h_gnn, pos, batch)
        else:
            h_trans = self.transformer(h_gnn)
        
        # Encode to latent space
        z = self.encoder(h_trans, batch, conditioning)
        
        return z
    
    def decode(self, z: torch.Tensor, data) -> Dict[str, torch.Tensor]:
        """Decode latent state to observables without encoding.
        
        Useful for analysis and hypothesis generation.
        
        Args:
            z: Latent state (batch_size, latent_dim)
            data: GraphData object (for particle embeddings and conditioning)
        
        Returns:
            Dictionary with 'particle_pred' and 'observables'
        """
        x = data.x
        pos = data.pos
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else None
        
        # Determine batch size
        if batch is not None:
            batch_size = int(batch.max().item()) + 1
        else:
            batch_size = 1
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        # Construct conditioning
        if hasattr(data, 'density') and hasattr(data, 'energy') and hasattr(data, 'material'):
            density = data.density.view(-1, 1) if data.density.dim() == 1 else data.density
            energy = data.energy.view(-1, 1) if data.energy.dim() == 1 else data.energy
            material = data.material if data.material.dim() == 2 else data.material.unsqueeze(0)
            conditioning = torch.cat([density, energy, material], dim=1)
        else:
            conditioning = torch.zeros(batch_size, self.config.conditioning_dim,
                                      device=z.device, dtype=z.dtype)
        
        # Get particle embeddings (need to process through GNN and Transformer)
        if self.config.enable_gnn:
            h_gnn = self.gnn(x, pos, batch)
        else:
            h_gnn = self.gnn(x)
        
        if self.config.enable_transformer:
            h_trans = self.transformer(h_gnn, pos, batch)
        else:
            h_trans = self.transformer(h_gnn)
        
        # Decode
        particle_pred, observables = self.decoder(z, h_trans, batch, conditioning)
        
        return {
            'particle_pred': particle_pred,
            'observables': observables
        }
    
    def get_component(self, component_name: str) -> nn.Module:
        """Get a specific component for ablation studies.
        
        Args:
            component_name: Name of component ('gnn', 'transformer', 'encoder', 'neural_ode', 'decoder')
        
        Returns:
            The requested component module
        
        Raises:
            ValueError: If component_name is invalid
        """
        components = {
            'gnn': self.gnn,
            'transformer': self.transformer,
            'encoder': self.encoder,
            'neural_ode': self.neural_ode,
            'decoder': self.decoder
        }
        
        if component_name not in components:
            raise ValueError(
                f"Invalid component name '{component_name}'. "
                f"Must be one of: {list(components.keys())}"
            )
        
        return components[component_name]
    
    def is_component_enabled(self, component_name: str) -> bool:
        """Check if a component is enabled.
        
        Args:
            component_name: Name of component ('gnn', 'transformer', 'neural_ode')
        
        Returns:
            True if component is enabled, False otherwise
        
        Raises:
            ValueError: If component_name is invalid or not ablatable
        """
        ablatable_components = {
            'gnn': self.config.enable_gnn,
            'transformer': self.config.enable_transformer,
            'neural_ode': self.config.enable_neural_ode
        }
        
        if component_name not in ablatable_components:
            raise ValueError(
                f"Invalid or non-ablatable component name '{component_name}'. "
                f"Ablatable components are: {list(ablatable_components.keys())}"
            )
        
        return ablatable_components[component_name]
    
    def get_ablation_config(self) -> Dict[str, bool]:
        """Get current ablation configuration.
        
        Returns:
            Dictionary mapping component names to enabled status
        """
        return {
            'gnn': self.config.enable_gnn,
            'transformer': self.config.enable_transformer,
            'neural_ode': self.config.enable_neural_ode
        }
