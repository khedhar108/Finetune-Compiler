"""
Unsloth integration utilities.

Provides helper functions to detect and use Unsloth when available.
"""

from __future__ import annotations

import sys
from typing import Any

from engine.utils.logging import get_logger, print_info, print_success, print_warning

logger = get_logger(__name__)


def is_unsloth_available() -> bool:
    """
    Check if Unsloth is available and can be used.
    
    Returns:
        True if Unsloth is installed and on a supported platform
    """
    # Unsloth only works on Linux
    if not sys.platform.startswith('linux'):
        return False
    
    try:
        import unsloth
        return True
    except ImportError:
        return False


def get_unsloth_model(
    model_name: str,
    max_seq_length: int = 2048,
    dtype: str = "auto",
    load_in_4bit: bool = True,
) -> tuple[Any, Any]:
    """
    Load a model using Unsloth for optimized training.
    
    Args:
        model_name: Model name (preferably unsloth/* models)
        max_seq_length: Maximum sequence length
        dtype: Data type
        load_in_4bit: Whether to use 4-bit quantization
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from unsloth import FastLanguageModel
    
    print_info(f"Loading with Unsloth: {model_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None if dtype == "auto" else dtype,
        load_in_4bit=load_in_4bit,
    )
    
    print_success("Model loaded with Unsloth optimizations")
    
    return model, tokenizer


def apply_unsloth_lora(
    model: Any,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> Any:
    """
    Apply LoRA using Unsloth's optimized implementation.
    
    Args:
        model: Model loaded with Unsloth
        r: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        target_modules: Target modules (Unsloth has good defaults)
        
    Returns:
        Model with LoRA applied
    """
    from unsloth import FastLanguageModel
    
    if target_modules is None:
        # Unsloth default targets
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    
    print_info(f"Applying Unsloth LoRA: r={r}, alpha={alpha}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth optimized
        random_state=42,
    )
    
    print_success("Unsloth LoRA applied")
    
    return model


def get_training_mode() -> str:
    """
    Get the training mode based on available optimizations.
    
    Returns:
        "unsloth" if Unsloth is available, "peft" otherwise
    """
    if is_unsloth_available():
        print_success("🚀 Unsloth detected - using optimized training (2-5x faster)")
        return "unsloth"
    else:
        print_warning("⚠️ Using standard PEFT (Unsloth requires Linux + GPU)")
        return "peft"
