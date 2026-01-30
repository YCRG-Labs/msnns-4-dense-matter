"""Graph Neural Network (GNN) layer implementation for particle interactions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.utils import scatter
from typing import Optional


def construct_edges(pos: torch.Tensor, 
                    cutoff_radius: float,
                    batch: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct graph edges based on spatial cutoff radius.
    
    Creates edges between particles within cutoff_radius distance of each other.
    Computes edge features including relative positions, distances, and relative momenta.
    
    Args:
        pos: Particle positions (N, 3)
        cutoff_radius: Spatial cutoff for edge construction (nm)
        batch: Batch assignment for each node (N,). If None, assumes single batch.
        
    Returns:
        edge_index: Edge connectivity (2, E)
        edge_attr: Edge features (E, 7) containing:
            - Relative position (3): r_j - r_i
            - Distance (1): ||r_j - r_i||
            - Placeholder for relative momentum (3): will be computed in forward pass
    
    Requirements:
        - Validates: Requirements 1.1, 1.2
        - Property 5: Edge Construction Correctness
        - Property 6: Relative Position Encoding
    """
    # Construct edges using radius_graph from PyTorch Geometric
    # This finds all pairs (i, j) where ||pos[j] - pos[i]|| <= cutoff_radius
    edge_index = radius_graph(pos, r=cutoff_radius, batch=batch, loop=False)
    
    # Compute edge features
    row, col = edge_index
    
    # Relative position: r_j - r_i (translation invariant)
    rel_pos = pos[col] - pos[row]  # (E, 3)
    
    # Distance: ||r_j - r_i||
    distance = torch.norm(rel_pos, dim=1, keepdim=True)  # (E, 1)
    
    # Combine edge features
    # Note: Relative momentum will be added in the message passing layer
    # when node features (which include momentum) are available
    edge_attr = torch.cat([rel_pos, distance], dim=1)  # (E, 4)
    
    return edge_index, edge_attr


class EdgeMLP(nn.Module):
    """MLP for computing edge messages.
    
    Takes concatenated node features and edge features as input,
    outputs message vectors.
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, message_dim: int):
        """Initialize edge MLP.
        
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden layer dimension
            message_dim: Output message dimension
        """
        super().__init__()
        
        # Input: [h_i || h_j || e_ij]
        input_dim = 2 * node_dim + edge_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )
    
    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Compute edge messages.
        
        Args:
            h_i: Source node features (E, node_dim)
            h_j: Target node features (E, node_dim)
            edge_attr: Edge features (E, edge_dim)
            
        Returns:
            messages: Edge messages (E, message_dim)
        """
        # Concatenate node and edge features
        x = torch.cat([h_i, h_j, edge_attr], dim=1)
        return self.mlp(x)


class NodeMLP(nn.Module):
    """MLP for updating node features.
    
    Takes concatenated node features and aggregated messages as input,
    outputs node updates.
    """
    
    def __init__(self, node_dim: int, message_dim: int, hidden_dim: int, output_dim: int):
        """Initialize node MLP.
        
        Args:
            node_dim: Dimension of node features
            message_dim: Dimension of aggregated messages
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
        """
        super().__init__()
        
        # Input: [h_i || m_i]
        input_dim = node_dim + message_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, h: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """Update node features.
        
        Args:
            h: Node features (N, node_dim)
            messages: Aggregated messages (N, message_dim)
            
        Returns:
            updates: Node updates (N, output_dim)
        """
        x = torch.cat([h, messages], dim=1)
        return self.mlp(x)


class MessagePassingLayer(MessagePassing):
    """Single message passing layer for GNN.
    
    Implements message passing with:
    - Edge message computation using relative positions and node features
    - Sum aggregation over neighbors
    - Node update using aggregated messages
    
    Requirements:
        - Validates: Requirements 1.2, 1.3
        - Property 6: Relative Position Encoding
        - Property 7: Message Aggregation Completeness
    """
    
    def __init__(self, node_dim: int, hidden_dim: int, message_dim: int):
        """Initialize message passing layer.
        
        Args:
            node_dim: Dimension of node features
            hidden_dim: Hidden layer dimension for MLPs
            message_dim: Dimension of message vectors
        """
        super().__init__(aggr='add', node_dim=0)  # Sum aggregation, node_dim=0 means use first dimension
        
        self._node_dim = node_dim
        self.message_dim = message_dim
        
        # Edge features: rel_pos (3) + distance (1) + rel_momentum (3) = 7
        edge_dim = 7
        
        # MLP for computing edge messages
        self.edge_mlp = EdgeMLP(node_dim, edge_dim, hidden_dim, message_dim)
        
        # MLP for updating nodes
        self.node_mlp = NodeMLP(node_dim, message_dim, hidden_dim, node_dim)
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """Forward pass of message passing layer.
        
        Args:
            x: Node features (N, node_dim) containing positions, momenta, charges, masses, species
            pos: Particle positions (N, 3) - used for computing relative momentum
            edge_index: Edge connectivity (2, E)
            edge_attr: Edge features (E, 4) - rel_pos (3) + distance (1)
            
        Returns:
            Node updates (N, node_dim)
        """
        # Extract momentum from node features
        # Assuming node features are: [pos (3), momentum (3), charge (1), mass (1), species_onehot (K)]
        # For now, we'll extract momentum from positions 3:6
        momentum = x[:, 3:6]  # (N, 3)
        
        # Compute relative momentum for edges
        row, col = edge_index
        rel_momentum = momentum[col] - momentum[row]  # (E, 3)
        
        # Augment edge features with relative momentum
        edge_attr_full = torch.cat([edge_attr, rel_momentum], dim=1)  # (E, 7)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr_full)
        
        return out
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Compute messages for each edge.
        
        Args:
            x_i: Source node features (E, node_dim)
            x_j: Target node features (E, node_dim)
            edge_attr: Edge features (E, 7)
            
        Returns:
            messages: Edge messages (E, message_dim)
        """
        return self.edge_mlp(x_i, x_j, edge_attr)
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Update node features using aggregated messages.
        
        Args:
            aggr_out: Aggregated messages (N, message_dim)
            x: Original node features (N, node_dim)
            
        Returns:
            Node updates (N, node_dim)
        """
        return self.node_mlp(x, aggr_out)


class GNNLayer(nn.Module):
    """Graph Neural Network layer with multiple message passing layers and residual connections.
    
    Stacks L message passing layers with residual connections and layer normalization.
    
    Requirements:
        - Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7
        - Property 1: Translation Invariance
        - Property 2: Permutation Invariance
        - Property 5: Edge Construction Correctness
        - Property 6: Relative Position Encoding
        - Property 7: Message Aggregation Completeness
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 4,
                 cutoff_radius: float = 5.0):
        """Initialize GNN layer.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden representations
            num_layers: Number of message passing layers (default: 4)
            cutoff_radius: Spatial cutoff for edge construction in nm (default: 5.0)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cutoff_radius = cutoff_radius
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, 
                x: torch.Tensor,
                pos: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through GNN layer.
        
        Args:
            x: Node features (N, input_dim)
            pos: Particle positions (N, 3)
            batch: Batch assignment for each node (N,)
            
        Returns:
            Node embeddings after message passing (N, hidden_dim)
        """
        # Construct edges based on spatial cutoff
        edge_index, edge_attr = construct_edges(pos, self.cutoff_radius, batch)
        
        # Project input features to hidden dimension
        h = self.input_proj(x)  # (N, hidden_dim)
        
        # Apply message passing layers with residual connections
        for i, (mp_layer, layer_norm) in enumerate(zip(self.mp_layers, self.layer_norms)):
            # Message passing
            h_update = mp_layer(h, pos, edge_index, edge_attr)
            
            # Residual connection
            h = h + h_update
            
            # Layer normalization
            h = layer_norm(h)
        
        return h
