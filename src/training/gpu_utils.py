"""GPU utilities for training infrastructure.

This module provides utilities for:
- CUDA device management
- Efficient batching for graphs
- GPU memory monitoring
- Distributed training support
- Mixed-precision training support

Requirements:
    - Validates: Requirements 16.1, 16.2, 16.3, 16.4, 16.5
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import Data, Batch
from typing import List, Optional, Dict, Tuple, Union
import warnings
import os


class DeviceManager:
    """Manages CUDA devices and provides device utilities.
    
    Requirements:
        - Validates: Requirement 16.1 (CUDA support)
    """
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        """Initialize device manager.
        
        Args:
            device: Device to use ('cuda', 'cpu', or specific device like 'cuda:0')
                   If None, automatically selects best available device
        """
        if device is None:
            self.device = self._auto_select_device()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self._log_device_info()
    
    def _auto_select_device(self) -> torch.device:
        """Automatically select best available device.
        
        Returns:
            torch.device: Best available device
        """
        if torch.cuda.is_available():
            # Select GPU with most free memory
            device_count = torch.cuda.device_count()
            if device_count > 1:
                # Find GPU with most free memory
                max_free_memory = 0
                best_device = 0
                
                for i in range(device_count):
                    torch.cuda.set_device(i)
                    free_memory = torch.cuda.get_device_properties(i).total_memory - \
                                 torch.cuda.memory_allocated(i)
                    if free_memory > max_free_memory:
                        max_free_memory = free_memory
                        best_device = i
                
                return torch.device(f'cuda:{best_device}')
            else:
                return torch.device('cuda:0')
        else:
            warnings.warn("CUDA not available, using CPU")
            return torch.device('cpu')
    
    def _log_device_info(self):
        """Log information about selected device."""
        print(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            device_idx = self.device.index if self.device.index is not None else 0
            props = torch.cuda.get_device_properties(device_idx)
            print(f"  Device name: {props.name}")
            print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
    
    def move_to_device(self, data):
        """Move data to device.
        
        Args:
            data: Data to move (tensor, module, or PyG Data/Batch)
        
        Returns:
            Data on device
        """
        if isinstance(data, (torch.Tensor, nn.Module)):
            return data.to(self.device)
        elif isinstance(data, (Data, Batch)):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self.move_to_device(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self.move_to_device(item) for item in data)
        else:
            return data
    
    def get_device(self) -> torch.device:
        """Get current device.
        
        Returns:
            torch.device: Current device
        """
        return self.device
    
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available.
        
        Returns:
            bool: True if CUDA is available
        """
        return torch.cuda.is_available()
    
    def get_device_count(self) -> int:
        """Get number of available CUDA devices.
        
        Returns:
            int: Number of CUDA devices
        """
        return torch.cuda.device_count() if torch.cuda.is_available() else 0


class GraphBatcher:
    """Efficient batching for graph data with variable-size particle configurations.
    
    Requirements:
        - Validates: Requirement 16.2 (efficient batching)
        - Use PyTorch Geometric batching for graphs
        - Handle variable-size particle configurations
    """
    
    def __init__(self, follow_batch: Optional[List[str]] = None):
        """Initialize graph batcher.
        
        Args:
            follow_batch: List of attributes to create batch assignment for
        """
        self.follow_batch = follow_batch or []
    
    def collate(self, data_list: List[Data]) -> Batch:
        """Collate list of PyG Data objects into a batch.
        
        Args:
            data_list: List of PyG Data objects
        
        Returns:
            Batch: Batched data
        
        Requirements:
            - Validates: Requirement 16.2
            - Use PyTorch Geometric batching for graphs
            - Handle variable-size particle configurations
        """
        if len(data_list) == 0:
            raise ValueError("Cannot collate empty list")
        
        # Use PyTorch Geometric's Batch.from_data_list
        # This automatically handles variable-size graphs
        batch = Batch.from_data_list(data_list, follow_batch=self.follow_batch)
        
        return batch
    
    def separate(self, batch: Batch) -> List[Data]:
        """Separate a batch back into individual Data objects.
        
        Args:
            batch: Batched data
        
        Returns:
            List of individual Data objects
        """
        return batch.to_data_list()
    
    def get_batch_size(self, batch: Batch) -> int:
        """Get number of graphs in batch.
        
        Args:
            batch: Batched data
        
        Returns:
            Number of graphs in batch
        """
        if hasattr(batch, 'num_graphs'):
            return batch.num_graphs
        elif hasattr(batch, 'batch'):
            return batch.batch.max().item() + 1
        else:
            return 1


class GPUMemoryMonitor:
    """Monitor GPU memory usage and provide warnings.
    
    Requirements:
        - Validates: Requirement 16.4 (GPU memory monitoring)
        - Check GPU memory usage and provide warnings
        - Implement automatic batch size adjustment
    """
    
    def __init__(self,
                 device: torch.device,
                 warning_threshold: float = 0.9,
                 critical_threshold: float = 0.95):
        """Initialize GPU memory monitor.
        
        Args:
            device: CUDA device to monitor
            warning_threshold: Fraction of memory usage to trigger warning (0-1)
            critical_threshold: Fraction of memory usage to trigger critical warning (0-1)
        """
        self.device = device
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
        if device.type != 'cuda':
            warnings.warn("GPU memory monitoring only works with CUDA devices")
            self.enabled = False
        else:
            self.enabled = True
            self.device_idx = device.index if device.index is not None else 0
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics.
        
        Returns:
            Dictionary with memory statistics (in GB)
        """
        if not self.enabled:
            return {}
        
        torch.cuda.synchronize(self.device)
        
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
        free = total - allocated
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'total': total,
            'utilization': allocated / total if total > 0 else 0.0
        }
    
    def check_memory(self, verbose: bool = False) -> Tuple[bool, str]:
        """Check memory usage and return warning if needed.
        
        Args:
            verbose: If True, always print memory stats
        
        Returns:
            Tuple of (is_critical, message)
        """
        if not self.enabled:
            return False, ""
        
        stats = self.get_memory_stats()
        utilization = stats['utilization']
        
        message = ""
        is_critical = False
        
        if utilization >= self.critical_threshold:
            message = (f"CRITICAL: GPU memory usage at {utilization*100:.1f}% "
                      f"({stats['allocated']:.2f}/{stats['total']:.2f} GB). "
                      f"Consider reducing batch size or model size.")
            is_critical = True
            warnings.warn(message)
        elif utilization >= self.warning_threshold:
            message = (f"WARNING: GPU memory usage at {utilization*100:.1f}% "
                      f"({stats['allocated']:.2f}/{stats['total']:.2f} GB)")
            warnings.warn(message)
        elif verbose:
            message = (f"GPU memory: {stats['allocated']:.2f}/{stats['total']:.2f} GB "
                      f"({utilization*100:.1f}%)")
            print(message)
        
        return is_critical, message
    
    def suggest_batch_size(self,
                          current_batch_size: int,
                          target_utilization: float = 0.8) -> int:
        """Suggest optimal batch size based on current memory usage.
        
        Args:
            current_batch_size: Current batch size
            target_utilization: Target memory utilization (0-1)
        
        Returns:
            Suggested batch size
        
        Requirements:
            - Validates: Requirement 16.4 (automatic batch size adjustment)
        """
        if not self.enabled:
            return current_batch_size
        
        stats = self.get_memory_stats()
        current_utilization = stats['utilization']
        
        if current_utilization == 0:
            return current_batch_size
        
        # Estimate batch size that would achieve target utilization
        # Assumes linear relationship between batch size and memory
        suggested_batch_size = int(current_batch_size * target_utilization / current_utilization)
        
        # Ensure at least batch size of 1
        suggested_batch_size = max(1, suggested_batch_size)
        
        # Don't suggest increasing batch size if already near target
        if current_utilization > target_utilization * 0.9:
            suggested_batch_size = min(suggested_batch_size, current_batch_size)
        
        return suggested_batch_size
    
    def clear_cache(self):
        """Clear GPU cache to free up memory."""
        if self.enabled:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)


class DistributedTrainingManager:
    """Manager for distributed training across multiple GPUs.
    
    Requirements:
        - Validates: Requirement 16.3 (distributed training)
        - Add support for multi-GPU training with DistributedDataParallel
    """
    
    def __init__(self,
                 backend: str = 'nccl',
                 init_method: str = 'env://'):
        """Initialize distributed training manager.
        
        Args:
            backend: Backend for distributed training ('nccl', 'gloo', 'mpi')
            init_method: Initialization method for process group
        """
        self.backend = backend
        self.init_method = init_method
        self.is_initialized = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
    
    def setup(self):
        """Setup distributed training environment.
        
        Requirements:
            - Validates: Requirement 16.3
            - Add support for multi-GPU training with DistributedDataParallel
        """
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, distributed training disabled")
            return
        
        # Check if running in distributed mode
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            # Initialize process group
            dist.init_process_group(
                backend=self.backend,
                init_method=self.init_method,
                rank=self.rank,
                world_size=self.world_size
            )
            
            # Set device for this process
            torch.cuda.set_device(self.local_rank)
            
            self.is_initialized = True
            
            if self.is_main_process():
                print(f"Distributed training initialized:")
                print(f"  Backend: {self.backend}")
                print(f"  World size: {self.world_size}")
                print(f"  Rank: {self.rank}")
                print(f"  Local rank: {self.local_rank}")
        else:
            if torch.cuda.device_count() > 1:
                warnings.warn(
                    f"Multiple GPUs detected ({torch.cuda.device_count()}) but "
                    "distributed training not initialized. Set RANK and WORLD_SIZE "
                    "environment variables to enable distributed training."
                )
    
    def cleanup(self):
        """Cleanup distributed training environment."""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
    
    def wrap_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        """Wrap model with DistributedDataParallel.
        
        Args:
            model: Model to wrap
            device: Device to place model on
        
        Returns:
            Wrapped model (DDP if distributed, otherwise original)
        """
        if not self.is_initialized:
            return model
        
        # Move model to device
        model = model.to(device)
        
        # Wrap with DDP
        model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False
        )
        
        return model
    
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0).
        
        Returns:
            True if main process
        """
        return self.rank == 0
    
    def get_rank(self) -> int:
        """Get process rank.
        
        Returns:
            Process rank
        """
        return self.rank
    
    def get_world_size(self) -> int:
        """Get world size (total number of processes).
        
        Returns:
            World size
        """
        return self.world_size
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_initialized:
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """All-reduce operation across all processes.
        
        Args:
            tensor: Tensor to reduce
            op: Reduction operation
        
        Returns:
            Reduced tensor
        """
        if self.is_initialized:
            dist.all_reduce(tensor, op=op)
        return tensor


class MixedPrecisionTrainer:
    """Mixed-precision training support using automatic mixed precision (AMP).
    
    Requirements:
        - Validates: Requirement 16.5 (mixed-precision training)
        - Add support for automatic mixed precision (AMP)
    """
    
    def __init__(self,
                 enabled: bool = True,
                 init_scale: float = 2.**16,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 2000):
        """Initialize mixed-precision trainer.
        
        Args:
            enabled: Whether to enable mixed-precision training
            init_scale: Initial loss scale
            growth_factor: Factor to grow loss scale
            backoff_factor: Factor to reduce loss scale on overflow
            growth_interval: Number of steps between loss scale increases
        """
        self.enabled = enabled and torch.cuda.is_available()
        
        if self.enabled:
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval
            )
            print("Mixed-precision training enabled (AMP)")
        else:
            self.scaler = None
            if enabled and not torch.cuda.is_available():
                warnings.warn("Mixed-precision training requires CUDA, disabled")
    
    def context(self):
        """Get autocast context for forward pass.
        
        Returns:
            Autocast context manager
        
        Requirements:
            - Validates: Requirement 16.5
            - Add support for automatic mixed precision (AMP)
        """
        if self.enabled:
            return autocast()
        else:
            # Return dummy context manager that does nothing
            from contextlib import nullcontext
            return nullcontext()
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass.
        
        Args:
            loss: Loss tensor
        
        Returns:
            Scaled loss
        """
        if self.enabled:
            return self.scaler.scale(loss)
        else:
            return loss
    
    def step(self, optimizer: torch.optim.Optimizer):
        """Perform optimizer step with gradient scaling.
        
        Args:
            optimizer: Optimizer
        """
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients before gradient clipping.
        
        Args:
            optimizer: Optimizer
        """
        if self.enabled:
            self.scaler.unscale_(optimizer)
    
    def get_scale(self) -> float:
        """Get current loss scale.
        
        Returns:
            Current loss scale
        """
        if self.enabled:
            return self.scaler.get_scale()
        else:
            return 1.0
    
    def state_dict(self) -> Dict:
        """Get state dict for checkpointing.
        
        Returns:
            State dict
        """
        if self.enabled:
            return self.scaler.state_dict()
        else:
            return {}
    
    def load_state_dict(self, state_dict: Dict):
        """Load state dict from checkpoint.
        
        Args:
            state_dict: State dict
        """
        if self.enabled and state_dict:
            self.scaler.load_state_dict(state_dict)


def setup_training_environment(
    device: Optional[str] = None,
    distributed: bool = False,
    mixed_precision: bool = False,
    seed: Optional[int] = None
) -> Tuple[DeviceManager, Optional[DistributedTrainingManager], Optional[MixedPrecisionTrainer]]:
    """Setup complete training environment with GPU support.
    
    Args:
        device: Device to use (None for auto-select)
        distributed: Whether to enable distributed training
        mixed_precision: Whether to enable mixed-precision training
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (device_manager, distributed_manager, mixed_precision_trainer)
    """
    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # Setup device manager
    device_manager = DeviceManager(device)
    
    # Setup distributed training
    distributed_manager = None
    if distributed:
        distributed_manager = DistributedTrainingManager()
        distributed_manager.setup()
    
    # Setup mixed-precision training
    mixed_precision_trainer = None
    if mixed_precision:
        mixed_precision_trainer = MixedPrecisionTrainer(enabled=True)
    
    return device_manager, distributed_manager, mixed_precision_trainer
