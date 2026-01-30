"""Transformer layer implementation for capturing long-range correlations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class RelativePositionEncodingMLP(nn.Module):
    """MLP that maps relative positions to encoding vectors.
    
    Maps relative position vectors (r_j - r_i) to encoding vectors that are
    added to attention scores to maintain translation invariance.
    
    Requirements:
        - Validates: Requirement 2.2
        - Maintains translation invariance through relative position encoding
    """
    
    def __init__(self, hidden_dim: int, num_heads: int):
        """Initialize relative position encoding MLP.
        
        Args:
            hidden_dim: Dimension of transformer hidden state
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # MLP: R^3 -> R^(hidden_dim / num_heads)
        # Maps relative position to encoding per head
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.head_dim)
        )
    
    def forward(self, rel_pos: torch.Tensor) -> torch.Tensor:
        """Compute relative position encodings.
        
        Args:
            rel_pos: Relative positions (N, N, 3) where rel_pos[i, j] = pos[j] - pos[i]
            
        Returns:
            Relative position encodings (N, N, head_dim)
        """
        # Flatten spatial dimensions for MLP
        N = rel_pos.shape[0]
        rel_pos_flat = rel_pos.reshape(-1, 3)  # (N*N, 3)
        
        # Apply MLP
        encoding_flat = self.mlp(rel_pos_flat)  # (N*N, head_dim)
        
        # Reshape back
        encoding = encoding_flat.reshape(N, N, self.head_dim)  # (N, N, head_dim)
        
        return encoding


class MultiHeadSelfAttentionWithRelativeEncoding(nn.Module):
    """Multi-head self-attention with relative position encoding.
    
    Implements attention mechanism with relative position encodings added to
    attention scores before softmax to maintain translation invariance.
    
    Attention formula:
        Q = x W_Q, K = x W_K, V = x W_V
        A_ij = softmax((Q_i · K_j + RPE(r_i, r_j)) / sqrt(d_k))
        output_i = sum_j A_ij V_j
    
    Requirements:
        - Validates: Requirements 2.1, 2.2, 2.3
        - Captures all-to-all particle interactions
        - Maintains translation invariance through relative position encoding
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """Initialize multi-head self-attention.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of transformer hidden state
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)
        self.v_proj = nn.Linear(input_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Relative position encoding
        self.rel_pos_encoder = RelativePositionEncodingMLP(hidden_dim, num_heads)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of multi-head self-attention.
        
        Args:
            x: Input features (N, input_dim)
            pos: Particle positions (N, 3) for relative encoding
            batch: Batch assignment (N,) - if provided, attention is computed within batches
            
        Returns:
            Output features (N, hidden_dim)
        """
        N = x.shape[0]
        
        # Handle batched computation
        if batch is not None:
            # Process each batch separately
            unique_batches = torch.unique(batch)
            outputs = []
            
            for b in unique_batches:
                mask = batch == b
                x_b = x[mask]
                pos_b = pos[mask]
                
                # Compute attention for this batch
                out_b = self._compute_attention(x_b, pos_b)
                outputs.append(out_b)
            
            # Concatenate results
            output = torch.zeros(N, self.hidden_dim, device=x.device, dtype=x.dtype)
            idx = 0
            for b in unique_batches:
                mask = batch == b
                n_b = mask.sum()
                output[mask] = outputs[unique_batches.tolist().index(b.item())]
            
            return output
        else:
            # Single batch
            return self._compute_attention(x, pos)
    
    def _compute_attention(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Compute attention for a single batch.
        
        Args:
            x: Input features (N, input_dim)
            pos: Particle positions (N, 3)
            
        Returns:
            Output features (N, hidden_dim)
        """
        N = x.shape[0]
        
        # Compute Q, K, V
        Q = self.q_proj(x)  # (N, hidden_dim)
        K = self.k_proj(x)  # (N, hidden_dim)
        V = self.v_proj(x)  # (N, hidden_dim)
        
        # Reshape for multi-head attention
        # (N, hidden_dim) -> (N, num_heads, head_dim)
        Q = Q.view(N, self.num_heads, self.head_dim)
        K = K.view(N, self.num_heads, self.head_dim)
        V = V.view(N, self.num_heads, self.head_dim)
        
        # Compute relative positions: rel_pos[i, j] = pos[j] - pos[i]
        rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)  # (N, N, 3)
        
        # Compute relative position encodings for each head
        # We'll compute for one head and reuse (since MLP output is per-head)
        rel_pos_encoding = self.rel_pos_encoder(rel_pos)  # (N, N, head_dim)
        
        # Compute attention scores for each head
        # Q: (N, num_heads, head_dim)
        # K: (N, num_heads, head_dim)
        # Attention: (N, num_heads, N)
        
        # Compute Q·K^T for each head
        # (N, num_heads, head_dim) @ (N, num_heads, head_dim).T
        # -> (num_heads, N, N)
        attn_scores = torch.einsum('ihd,jhd->hij', Q, K) / self.scale  # (num_heads, N, N)
        
        # Add relative position encoding to attention scores
        # rel_pos_encoding: (N, N, head_dim)
        # We need to add this to each head's attention scores
        # For each head h: attn_scores[h, i, j] += Q[i, h] · rel_pos_encoding[i, j]
        rel_attn = torch.einsum('ihd,ijd->hij', Q, rel_pos_encoding) / self.scale  # (num_heads, N, N)
        attn_scores = attn_scores + rel_attn
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # (num_heads, N, N)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # attn_weights: (num_heads, N, N)
        # V: (N, num_heads, head_dim)
        # output: (N, num_heads, head_dim)
        output = torch.einsum('hij,jhd->ihd', attn_weights, V)  # (N, num_heads, head_dim)
        
        # Concatenate heads
        output = output.reshape(N, self.hidden_dim)  # (N, hidden_dim)
        
        # Output projection
        output = self.out_proj(output)
        
        return output


class FeedForwardNetwork(nn.Module):
    """Feed-forward network with residual connections.
    
    Implements: FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2
    
    Requirements:
        - Validates: Requirement 2.4
    """
    
    def __init__(self, hidden_dim: int, ffn_dim: Optional[int] = None, dropout: float = 0.1):
        """Initialize feed-forward network.
        
        Args:
            hidden_dim: Dimension of hidden state
            ffn_dim: Dimension of FFN hidden layer (default: 4 * hidden_dim)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()
        
        if ffn_dim is None:
            ffn_dim = 4 * hidden_dim
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FFN.
        
        Args:
            x: Input features (N, hidden_dim)
            
        Returns:
            Output features (N, hidden_dim)
        """
        return self.ffn(x)


class TransformerLayer(nn.Module):
    """Transformer layer for capturing long-range correlations.
    
    Combines multi-head self-attention with relative position encoding,
    feed-forward network, layer normalization, and residual connections.
    
    Architecture:
        h' = LayerNorm(h + Attention(h, pos))
        h'' = LayerNorm(h' + FFN(h'))
    
    Requirements:
        - Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5
        - Property 1: Translation Invariance (through relative position encoding)
        - Property 2: Permutation Equivariance
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """Initialize transformer layer.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of transformer hidden state
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Input projection (if input_dim != hidden_dim)
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()
        
        # Multi-head self-attention with relative encoding
        self.attention = MultiHeadSelfAttentionWithRelativeEncoding(
            hidden_dim, hidden_dim, num_heads, dropout
        )
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(hidden_dim, dropout=dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                x: torch.Tensor,
                pos: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer layer.
        
        Args:
            x: Input features (N, input_dim)
            pos: Particle positions (N, 3) for relative encoding
            batch: Batch assignment (N,)
            
        Returns:
            Transformed features with long-range correlations (N, hidden_dim)
        """
        # Project input to hidden dimension
        h = self.input_proj(x)
        
        # Self-attention with residual connection and layer norm
        attn_out = self.attention(h, pos, batch)
        h = self.norm1(h + self.dropout(attn_out))
        
        # Feed-forward with residual connection and layer norm
        ffn_out = self.ffn(h)
        h = self.norm2(h + self.dropout(ffn_out))
        
        return h
