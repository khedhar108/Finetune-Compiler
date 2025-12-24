"""
GPU Memory utilities for the AI Compiler Core Engine.

Provides functions for monitoring and managing GPU memory.
"""

from __future__ import annotations

import gc
from typing import Any

from engine.utils.logging import get_logger, console

logger = get_logger(__name__)


def get_gpu_memory() -> dict[str, Any]:
    """
    Get current GPU memory usage.
    
    Returns:
        Dictionary with memory statistics or empty dict if no GPU
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {"available": False, "message": "No CUDA GPU available"}
        
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        total = props.total_memory
        free = total - reserved
        
        return {
            "available": True,
            "device_name": props.name,
            "device_index": device,
            "total_gb": total / (1024**3),
            "allocated_gb": allocated / (1024**3),
            "reserved_gb": reserved / (1024**3),
            "free_gb": free / (1024**3),
            "utilization_percent": (reserved / total) * 100,
        }
    except ImportError:
        return {"available": False, "message": "PyTorch not installed"}
    except Exception as e:
        return {"available": False, "message": str(e)}


def clear_gpu_memory() -> None:
    """
    Clear GPU memory cache.
    
    This forces garbage collection and clears the CUDA cache.
    """
    try:
        import torch
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU memory cache cleared")
    except ImportError:
        pass


def log_gpu_memory(prefix: str = "") -> None:
    """
    Log current GPU memory usage.
    
    Args:
        prefix: Optional prefix for the log message
    """
    mem = get_gpu_memory()
    
    if not mem.get("available", False):
        logger.info(f"{prefix}GPU: {mem.get('message', 'Not available')}")
        return
    
    msg = (
        f"{prefix}GPU Memory: "
        f"{mem['allocated_gb']:.2f}GB allocated / "
        f"{mem['reserved_gb']:.2f}GB reserved / "
        f"{mem['total_gb']:.2f}GB total "
        f"({mem['utilization_percent']:.1f}% used)"
    )
    logger.info(msg)


def print_gpu_info() -> None:
    """Print GPU information in a formatted way."""
    mem = get_gpu_memory()
    
    if not mem.get("available", False):
        console.print(f"[yellow]GPU: {mem.get('message', 'Not available')}[/yellow]")
        return
    
    console.print(f"[bold cyan]GPU: {mem['device_name']}[/bold cyan]")
    console.print(f"  Total Memory: {mem['total_gb']:.2f} GB")
    console.print(f"  Allocated: {mem['allocated_gb']:.2f} GB")
    console.print(f"  Reserved: {mem['reserved_gb']:.2f} GB")
    console.print(f"  Free: {mem['free_gb']:.2f} GB")
    console.print(f"  Utilization: {mem['utilization_percent']:.1f}%")


def check_memory_requirements(
    model_size_gb: float,
    quantization: str = "4bit",
    buffer_factor: float = 1.5,
) -> tuple[bool, str]:
    """
    Check if GPU has enough memory for training.
    
    Args:
        model_size_gb: Approximate model size in GB
        quantization: Quantization mode (none, 4bit, 8bit)
        buffer_factor: Safety buffer multiplier
        
    Returns:
        Tuple of (can_train, message)
    """
    mem = get_gpu_memory()
    
    if not mem.get("available", False):
        return False, "No GPU available"
    
    # Estimate memory requirements based on quantization
    if quantization == "4bit":
        required_gb = model_size_gb * 0.25 * buffer_factor
    elif quantization == "8bit":
        required_gb = model_size_gb * 0.5 * buffer_factor
    else:
        required_gb = model_size_gb * buffer_factor
    
    free_gb = mem["free_gb"]
    
    if free_gb >= required_gb:
        return True, f"Sufficient memory: {free_gb:.2f}GB free, ~{required_gb:.2f}GB required"
    else:
        return False, f"Insufficient memory: {free_gb:.2f}GB free, ~{required_gb:.2f}GB required"
