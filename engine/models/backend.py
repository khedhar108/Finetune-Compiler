"""
Model Backend Selector.

Determines whether to use HuggingFace or Unsloth backend.
"""

from __future__ import annotations

import os
import platform
from enum import Enum
from typing import Literal

from engine.models.unsloth_loader import is_unsloth_available

class BackendType(str, Enum):
    HUGGINGFACE = "huggingface"
    UNSLOTH = "unsloth"


def get_optimal_backend(model_name: str) -> BackendType:
    """
    Determine the optimal backend for the current environment and model.
    
    Args:
        model_name: Name of the model to check compatibility
        
    Returns:
        BackendType (unsloth or huggingface)
    """
    # 1. Check if Unsloth is installed
    if not is_unsloth_available():
        return BackendType.HUGGINGFACE
    
    # 2. Check Platform (Unsloth works best on Linux/WSL)
    # On Windows native, we might prefer HF unless we're sure
    # But if it's installed, we assume user wants it.
    
    # 3. Check Model Compatibility
    # Unsloth supports Llama, Mistral, Gemma, Yi, Qwen, DeepSeek, etc.
    # It does NOT support Whisper, generic BERT, etc.
    model_name_lower = model_name.lower()
    supported_architectures = [
        "llama", "mistral", "gemma", "yi", "qwen", "deepseek", "tinyllama"
    ]
    
    is_supported = any(arch in model_name_lower for arch in supported_architectures)
    
    if is_supported:
        return BackendType.UNSLOTH
    
    return BackendType.HUGGINGFACE
