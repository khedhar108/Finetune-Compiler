"""
Cache management utilities.

Configure and manage HuggingFace model cache location.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from engine.utils.logging import get_logger, print_info, print_success

logger = get_logger(__name__)


def get_cache_dir() -> Path:
    """
    Get current HuggingFace cache directory.
    
    Returns:
        Path to cache directory
    """
    # Check environment variables in order of priority
    cache_dir = (
        os.environ.get("HF_HOME") or
        os.environ.get("HUGGINGFACE_HUB_CACHE") or
        os.environ.get("TRANSFORMERS_CACHE") or
        Path.home() / ".cache" / "huggingface" / "hub"
    )
    return Path(cache_dir)


def set_cache_dir(path: str) -> bool:
    """
    Set HuggingFace cache directory.
    
    Args:
        path: Path to use as cache directory
        
    Returns:
        True if successful, False otherwise
    """
    cache_path = Path(path)
    
    try:
        # Create directory if it doesn't exist
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables
        os.environ["HF_HOME"] = str(cache_path)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_path / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(cache_path)
        
        print_success(f"Cache directory set to: {cache_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to set cache directory: {e}")
        return False


def get_cache_size() -> dict:
    """
    Get cache directory size information.
    
    Returns:
        Dict with size info
    """
    cache_dir = get_cache_dir()
    
    if not cache_dir.exists():
        return {
            "path": str(cache_dir),
            "exists": False,
            "size_bytes": 0,
            "size_gb": 0,
            "model_count": 0,
        }
    
    total_size = 0
    model_count = 0
    
    for item in cache_dir.iterdir():
        if item.is_dir() and item.name.startswith("models--"):
            model_count += 1
            for f in item.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size
    
    return {
        "path": str(cache_dir),
        "exists": True,
        "size_bytes": total_size,
        "size_gb": round(total_size / (1024**3), 2),
        "model_count": model_count,
    }


def list_cached_models() -> list:
    """
    List all cached models.
    
    Returns:
        List of model names
    """
    cache_dir = get_cache_dir()
    
    if not cache_dir.exists():
        return []
    
    models = []
    for item in cache_dir.iterdir():
        if item.is_dir() and item.name.startswith("models--"):
            # Convert folder name to model name
            # models--unsloth--Llama-3.2-1B -> unsloth/Llama-3.2-1B
            name = item.name.replace("models--", "").replace("--", "/")
            models.append(name)
    
    return models


def clear_model_cache(model_name: str) -> bool:
    """
    Clear cache for a specific model.
    
    Args:
        model_name: Model name (e.g., "unsloth/Llama-3.2-1B")
        
    Returns:
        True if successful
    """
    import shutil
    
    cache_dir = get_cache_dir()
    # Convert model name to folder name
    folder_name = f"models--{model_name.replace('/', '--')}"
    model_cache = cache_dir / folder_name
    
    if model_cache.exists():
        shutil.rmtree(model_cache)
        print_success(f"Cleared cache for: {model_name}")
        return True
    
    return False


def get_cache_status() -> str:
    """Get formatted cache status for display."""
    info = get_cache_size()
    
    if not info["exists"]:
        return f"📁 Cache: Not created yet\n📍 Location: {info['path']}"
    
    return (
        f"📁 Cache: {info['size_gb']} GB ({info['model_count']} models)\n"
        f"📍 Location: {info['path']}"
    )
