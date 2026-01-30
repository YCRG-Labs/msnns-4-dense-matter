"""Neural ODE implementation for modeling collective dynamics in latent space."""

import torch
import torch.nn as nn
from typing import Optional, List
import math
import logging


logger = logging.getLogger(__name__)


class DynamicsFunction(nn.Module):
    """Dynamics function for Neural ODE with time encoding.
    
    Implements: f_θ(z, c, t) = MLP([z || c || sin(ωt) || cos(ωt)])
    
    The dynamics function models collective dynamics in latent space, separating
    fast particle collisions from slow collective oscillations. Time encoding
    captures periodic phenomena.
    
    Architecture:
        Input: [z || c || sin(ωt) || cos(ωt)]
        Hidden layers: [256, 256]
        Output: dz/dt (same dimension as z)
    
    Requirements:
        - Validates: Requirement 3.1
        - Models learned dynamics in latent space
        - Conditions on beam parameters (density, energy, material)
        - Uses time encoding to capture periodic phenomena
    """
    
    def __init__(self,
                 latent_dim: int,
                 conditioning_dim: int,
                 hidden_dims: Optional[List[int]] = None,
                 time_encoding_freq: Optional[float] = None):
        """Initialize dynamics function.
        
        Args:
            latent_dim: Dimension of latent space (D_latent)
            conditioning_dim: Dimension of conditioning vector (D_cond)
            hidden_dims: Hidden layer dimensions (default: [256, 256])
            time_encoding_freq: Frequency ω for time encoding (default: 2π/1000)
                If None, uses ω = 2π / T_max where T_max = 1000 fs
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.conditioning_dim = conditioning_dim
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        # Time encoding frequency: ω = 2π / T_max
        if time_encoding_freq is None:
            T_max = 1000.0  # Maximum trajectory time in fs
            self.omega = 2 * math.pi / T_max
        else:
            self.omega = time_encoding_freq
        
        # MLP architecture: [D_latent + D_cond + 2] → hidden_dims → D_latent
        # Input: z (latent_dim) + c (conditioning_dim) + sin(ωt) + cos(ωt) (2)
        mlp_input_dim = latent_dim + conditioning_dim + 2
        
        # Build MLP layers
        layers = []
        prev_dim = mlp_input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer: dz/dt has same dimension as z
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Store conditioning for use in forward pass
        self.conditioning = None
    
    def set_conditioning(self, conditioning: torch.Tensor):
        """Set conditioning vector for the dynamics function.
        
        This is called before ODE integration to provide beam parameters
        that remain constant during integration.
        
        Args:
            conditioning: Beam parameters (batch_size, conditioning_dim)
        """
        self.conditioning = conditioning
    
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute dz/dt at time t.
        
        This is the function that will be integrated by the ODE solver.
        
        Args:
            t: Current time (scalar tensor)
            z: Current latent state (batch_size, latent_dim)
            
        Returns:
            dz/dt: Time derivative of latent state (batch_size, latent_dim)
        """
        if self.conditioning is None:
            raise RuntimeError(
                "Conditioning must be set before calling forward. "
                "Call set_conditioning() first."
            )
        
        # Ensure t is a scalar
        if t.dim() > 0:
            t = t.squeeze()
        
        # Compute time encoding: [sin(ωt), cos(ωt)]
        sin_t = torch.sin(self.omega * t)
        cos_t = torch.cos(self.omega * t)
        
        # Expand time encoding to match batch size
        batch_size = z.shape[0]
        time_encoding = torch.stack([sin_t, cos_t]).unsqueeze(0).expand(batch_size, -1)
        time_encoding = time_encoding.to(z.device)
        
        # Concatenate: [z || c || sin(ωt) || cos(ωt)]
        mlp_input = torch.cat([z, self.conditioning, time_encoding], dim=1)
        
        # Compute dz/dt
        dzdt = self.mlp(mlp_input)
        
        return dzdt


class NeuralODE(nn.Module):
    """Neural ODE for modeling collective dynamics in latent space.
    
    Uses torchdiffeq to integrate the dynamics function:
        dz/dt = f_θ(z, c, t)
        z(t_1) = ODESolve(f_θ, z(t_0), t_0, t_1)
    
    The Neural ODE separates fast particle collisions from slow collective
    oscillations by modeling dynamics in a low-dimensional latent space.
    
    Requirements:
        - Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5
        - Evolves latent state according to learned dynamics
        - Conditions on beam density, energy, and material properties
        - Integrates over specified time intervals
        - Uses adaptive ODE solvers for numerical stability
    """
    
    def __init__(self,
                 latent_dim: int,
                 conditioning_dim: int,
                 hidden_dims: Optional[List[int]] = None,
                 time_encoding_freq: Optional[float] = None,
                 solver: str = 'dopri5',
                 rtol: float = 1e-3,
                 atol: float = 1e-4):
        """Initialize Neural ODE.
        
        Args:
            latent_dim: Dimension of latent space (D_latent)
            conditioning_dim: Dimension of conditioning vector (D_cond)
            hidden_dims: Hidden layer dimensions for dynamics MLP (default: [256, 256])
            time_encoding_freq: Frequency for time encoding (default: 2π/1000)
            solver: ODE solver method (default: 'dopri5' - Dormand-Prince 5th order)
            rtol: Relative tolerance for ODE solver (default: 1e-3)
            atol: Absolute tolerance for ODE solver (default: 1e-4)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.conditioning_dim = conditioning_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        
        # Dynamics function
        self.dynamics = DynamicsFunction(
            latent_dim=latent_dim,
            conditioning_dim=conditioning_dim,
            hidden_dims=hidden_dims,
            time_encoding_freq=time_encoding_freq
        )
    
    def forward(self,
                z0: torch.Tensor,
                t_span: torch.Tensor,
                conditioning: torch.Tensor) -> torch.Tensor:
        """Integrate latent dynamics from t0 to t1.
        
        Uses fallback solvers if primary solver fails:
        1. dopri5 (Dormand-Prince 5th order) - default
        2. rk4 (4th order Runge-Kutta) - fallback
        3. euler (Euler method) - last resort
        
        Args:
            z0: Initial latent state (batch_size, latent_dim)
            t_span: Time interval [t_start, t_end] (2,)
            conditioning: Beam parameters (batch_size, conditioning_dim)
                Contains: [density, energy, material_encoding]
            
        Returns:
            z1: Evolved latent state at t_end (batch_size, latent_dim)
        
        Raises:
            RuntimeError: If all solvers fail
        """
        # Validate inputs
        if z0.dim() != 2:
            raise ValueError(f"z0 must be 2D (batch_size, latent_dim), got shape {z0.shape}")
        
        if t_span.shape != (2,):
            raise ValueError(f"t_span must have shape (2,), got shape {t_span.shape}")
        
        if conditioning.shape[0] != z0.shape[0]:
            raise ValueError(
                f"Conditioning batch size ({conditioning.shape[0]}) "
                f"does not match z0 batch size ({z0.shape[0]})"
            )
        
        if conditioning.shape[1] != self.conditioning_dim:
            raise ValueError(
                f"Conditioning dimension ({conditioning.shape[1]}) "
                f"does not match expected dimension ({self.conditioning_dim})"
            )
        
        # Set conditioning for dynamics function
        self.dynamics.set_conditioning(conditioning)
        
        # Import torchdiffeq (lazy import to avoid dependency issues)
        try:
            from torchdiffeq import odeint_adjoint as odeint
        except ImportError:
            raise ImportError(
                "torchdiffeq is required for Neural ODE. "
                "Install with: pip install torchdiffeq"
            )
        
        # Try solvers in order: dopri5 -> rk4 -> euler
        solvers = [self.solver, 'rk4', 'euler']
        tolerances = [
            (self.rtol, self.atol),
            (1e-2, 1e-3),  # Relaxed tolerances for rk4
            (1e-1, 1e-2)   # Very relaxed for euler
        ]
        
        last_error = None
        
        for solver, (rtol, atol) in zip(solvers, tolerances):
            try:
                # Integrate ODE
                # odeint returns (num_times, batch_size, latent_dim)
                z_trajectory = odeint(
                    self.dynamics,
                    z0,
                    t_span,
                    method=solver,
                    rtol=rtol,
                    atol=atol
                )
                
                # Return final state z(t_end)
                z1 = z_trajectory[-1]  # (batch_size, latent_dim)
                
                # Check for NaN/Inf in result
                if torch.isnan(z1).any() or torch.isinf(z1).any():
                    raise RuntimeError(f"NaN or Inf detected in ODE solution with solver {solver}")
                
                # Log if we had to use a fallback solver
                if solver != self.solver:
                    logger.warning(
                        f"Primary solver '{self.solver}' failed, "
                        f"successfully used fallback solver '{solver}'"
                    )
                
                return z1
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"ODE integration failed with solver '{solver}': {e}. "
                    f"Trying next fallback solver..."
                )
                continue
        
        # All solvers failed
        raise RuntimeError(
            f"All ODE solvers failed. Last error: {last_error}. "
            "This indicates severe numerical instability. "
            "Try reducing time step, checking input data, or adjusting model parameters."
        )
    
    def integrate_trajectory(self,
                            z0: torch.Tensor,
                            times: torch.Tensor,
                            conditioning: torch.Tensor) -> torch.Tensor:
        """Integrate latent dynamics over multiple time points.
        
        This is useful for analyzing latent trajectories and extracting
        collective mode properties. Uses fallback solvers if primary fails.
        
        Args:
            z0: Initial latent state (batch_size, latent_dim)
            times: Time points to evaluate (num_times,)
            conditioning: Beam parameters (batch_size, conditioning_dim)
            
        Returns:
            z_trajectory: Latent states at all time points (num_times, batch_size, latent_dim)
        
        Raises:
            RuntimeError: If all solvers fail
        """
        # Set conditioning for dynamics function
        self.dynamics.set_conditioning(conditioning)
        
        # Import torchdiffeq
        try:
            from torchdiffeq import odeint_adjoint as odeint
        except ImportError:
            raise ImportError(
                "torchdiffeq is required for Neural ODE. "
                "Install with: pip install torchdiffeq"
            )
        
        # Try solvers in order: dopri5 -> rk4 -> euler
        solvers = [self.solver, 'rk4', 'euler']
        tolerances = [
            (self.rtol, self.atol),
            (1e-2, 1e-3),  # Relaxed tolerances for rk4
            (1e-1, 1e-2)   # Very relaxed for euler
        ]
        
        last_error = None
        
        for solver, (rtol, atol) in zip(solvers, tolerances):
            try:
                # Integrate ODE over all time points
                z_trajectory = odeint(
                    self.dynamics,
                    z0,
                    times,
                    method=solver,
                    rtol=rtol,
                    atol=atol
                )
                
                # Check for NaN/Inf in result
                if torch.isnan(z_trajectory).any() or torch.isinf(z_trajectory).any():
                    raise RuntimeError(f"NaN or Inf detected in ODE trajectory with solver {solver}")
                
                # Log if we had to use a fallback solver
                if solver != self.solver:
                    logger.warning(
                        f"Primary solver '{self.solver}' failed for trajectory integration, "
                        f"successfully used fallback solver '{solver}'"
                    )
                
                return z_trajectory
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"ODE trajectory integration failed with solver '{solver}': {e}. "
                    f"Trying next fallback solver..."
                )
                continue
        
        # All solvers failed
        raise RuntimeError(
            f"All ODE solvers failed for trajectory integration. Last error: {last_error}. "
            "This indicates severe numerical instability."
        )
