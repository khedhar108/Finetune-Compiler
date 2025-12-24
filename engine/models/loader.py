"""
Model loading utilities for the AI Compiler Core Engine.

Supports loading models with various quantization options.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from engine.utils.logging import get_logger, print_info, print_success, print_error
from engine.utils.memory import log_gpu_memory, clear_gpu_memory

logger = get_logger(__name__)


def get_device_map() -> str:
    """
    Get the appropriate device map for model loading.
    
    Returns:
        Device map string
    """
    if torch.cuda.is_available():
        return "auto"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_quantization_config(
    quantization: Literal["none", "4bit", "8bit"],
    compute_dtype: torch.dtype = torch.float16,
) -> BitsAndBytesConfig | None:
    """
    Get the quantization configuration.
    
    Args:
        quantization: Quantization mode
        compute_dtype: Compute dtype for quantized layers
        
    Returns:
        BitsAndBytesConfig or None
    """
    if quantization == "none":
        return None
    
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    
    if quantization == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    raise ValueError(f"Unsupported quantization: {quantization}")


def get_torch_dtype(dtype: str) -> torch.dtype:
    """
    Convert dtype string to torch dtype.
    
    Args:
        dtype: Dtype string
        
    Returns:
        torch.dtype
    """
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    
    return dtype_map.get(dtype, "auto")


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = False,
    **kwargs: Any,
) -> PreTrainedTokenizer:
    """
    Load a tokenizer from HuggingFace.
    
    Args:
        model_name: Model name or path
        trust_remote_code: Whether to trust remote code
        **kwargs: Additional arguments for AutoTokenizer
        
    Returns:
        Loaded tokenizer
    """
    print_info(f"Loading tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print_success("Tokenizer loaded")
    return tokenizer


def load_model(
    model_name: str,
    quantization: Literal["none", "4bit", "8bit"] = "4bit",
    max_seq_length: int = 2048,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    device_map: str | None = None,
    **kwargs: Any,
) -> PreTrainedModel:
    """
    Load a model with optional quantization.
    
    Args:
        model_name: Model name or path
        quantization: Quantization mode (none, 4bit, 8bit)
        max_seq_length: Maximum sequence length
        dtype: Model dtype
        trust_remote_code: Whether to trust remote code
        device_map: Device map for multi-GPU
        **kwargs: Additional arguments for AutoModelForCausalLM
        
    Returns:
        Loaded model
    """
    print_info(f"Loading model: {model_name}")
    print_info(f"Quantization: {quantization}, Max seq length: {max_seq_length}")
    
    # Log GPU memory before loading
    log_gpu_memory("Before loading: ")
    
    # Get device map
    if device_map is None:
        device_map = get_device_map()
    
    # Get quantization config
    quant_config = get_quantization_config(
        quantization,
        compute_dtype=torch.float16 if dtype == "float16" else torch.bfloat16,
    )
    
    # Get torch dtype
    torch_dtype = get_torch_dtype(dtype)
    
    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        
        print_success(f"Model loaded: {model_name}")
        log_gpu_memory("After loading: ")
        
        return model
        
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        clear_gpu_memory()
        raise


def prepare_model_for_training(
    model: PreTrainedModel,
    gradient_checkpointing: bool = True,
) -> PreTrainedModel:
    """
    Prepare a model for training.
    
    Args:
        model: Model to prepare
        gradient_checkpointing: Whether to enable gradient checkpointing
        
    Returns:
        Prepared model
    """
    # Enable gradient checkpointing
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # Enable input gradients for LoRA
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    return model
