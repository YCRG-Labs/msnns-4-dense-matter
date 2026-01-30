"""Neural network models for multi-scale physics-informed learning."""

from .gnn import (
    construct_edges,
    EdgeMLP,
    NodeMLP,
    MessagePassingLayer,
    GNNLayer
)
from .encoder import (
    GlobalPooling,
    Encoder
)
from .neural_ode import (
    DynamicsFunction,
    NeuralODE
)

__all__ = [
    'construct_edges',
    'EdgeMLP',
    'NodeMLP',
    'MessagePassingLayer',
    'GNNLayer',
    'GlobalPooling',
    'Encoder',
    'DynamicsFunction',
    'NeuralODE'
]
