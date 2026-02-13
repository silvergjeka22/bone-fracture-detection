"""
Device management utilities.

This module provides utilities for detecting and managing compute devices
(CPU, CUDA, Apple Silicon MPS).
"""

import torch


def get_device(prefer_mps=True, prefer_cuda=True):
    """
    Automatically detect and return the best available device.
    
    Args:
        prefer_mps: Prefer Apple Silicon MPS if available
        prefer_cuda: Prefer CUDA if available
        
    Returns:
        torch.device: The selected device
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif prefer_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon MPS device")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device


def move_to_device(obj, device):
    """
    Move a tensor, model, or collection to the specified device.
    
    Args:
        obj: Tensor, model, list, tuple, or dict to move
        device: Target device
        
    Returns:
        Object moved to the device
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item, device) for item in obj)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    else:
        return obj


def print_device_info():
    """Print information about available devices."""
    print("=" * 70)
    print("DEVICE INFORMATION")
    print("=" * 70)
    
    # CPU
    print(f"CPU: Available")
    
    # CUDA
    if torch.cuda.is_available():
        print(f"CUDA: Available")
        print(f"  Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA: Not available")
    
    # MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        print("MPS (Apple Silicon): Available")
    else:
        print("MPS (Apple Silicon): Not available")
    
    print("=" * 70)
