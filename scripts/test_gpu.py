"""Quick GPU test script."""

import torch
import sys

print("="*60)
print("GPU Configuration Test")
print("="*60)
print()

# Check PyTorch
print(f"PyTorch version: {torch.__version__}")
print()

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print()
    
    # Check GPUs
    gpu_count = torch.cuda.device_count()
    print(f"GPU count: {gpu_count}")
    print()
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  VRAM: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi-Processors: {props.multi_processor_count}")
        print()
    
    # Test GPU computation
    print("Testing GPU computation...")
    try:
        # Create tensors on GPU
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        
        # Perform computation
        z = torch.matmul(x, y)
        
        # Synchronize
        torch.cuda.synchronize()
        
        print("✅ GPU computation successful!")
        print()
        
        # Memory info
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        print(f"GPU Memory:")
        print(f"  Allocated: {allocated:.1f} MB")
        print(f"  Reserved: {reserved:.1f} MB")
        print()
        
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        print()
    
    # Recommended settings
    print("="*60)
    print("Recommended Training Settings")
    print("="*60)
    print()
    
    vram_gb = props.total_memory / 1024**3
    
    if vram_gb >= 20:
        batch_size = 32
        print(f"Your GPU has {vram_gb:.0f}GB VRAM - Excellent!")
        print(f"Recommended batch size: {batch_size}")
    elif vram_gb >= 12:
        batch_size = 16
        print(f"Your GPU has {vram_gb:.0f}GB VRAM - Great!")
        print(f"Recommended batch size: {batch_size}")
    elif vram_gb >= 8:
        batch_size = 8
        print(f"Your GPU has {vram_gb:.0f}GB VRAM - Good!")
        print(f"Recommended batch size: {batch_size}")
    else:
        batch_size = 4
        print(f"Your GPU has {vram_gb:.0f}GB VRAM - Adequate")
        print(f"Recommended batch size: {batch_size}")
    
    print()
    print("Training command:")
    print(f"python scripts/train_pretrain.py \\")
    print(f"    --train-data data/synthetic_small \\")
    print(f"    --epochs 50 \\")
    print(f"    --batch-size {batch_size} \\")
    print(f"    --device cuda \\")
    print(f"    --checkpoint-dir checkpoints/gpu_training")
    print()
    
    # Estimate training time
    if vram_gb >= 20:
        time_est = "15-30 minutes"
    elif vram_gb >= 12:
        time_est = "20-40 minutes"
    elif vram_gb >= 8:
        time_est = "30-60 minutes"
    else:
        time_est = "45-90 minutes"
    
    print(f"Estimated training time: {time_est}")
    print()
    
    print("="*60)
    print("✅ GPU is ready for training!")
    print("="*60)
    
else:
    print("❌ CUDA not available")
    print()
    print("Possible issues:")
    print("1. No NVIDIA GPU detected")
    print("2. CUDA drivers not installed")
    print("3. PyTorch installed without CUDA support")
    print()
    print("To install PyTorch with CUDA:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print()
    sys.exit(1)
