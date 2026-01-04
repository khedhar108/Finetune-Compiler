"""
Unsloth model loader for high-performance training.
Wrapper around FastLanguageModel.
"""

from __future__ import annotations

from typing import Any, Tuple

from engine.utils.logging import get_logger, print_info, print_success, print_error
from engine.utils.memory import log_gpu_memory

logger = get_logger(__name__)

# Try importing Unsloth
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    FastLanguageModel = None


def is_unsloth_available() -> bool:
    """Check if Unsloth is available."""
    return UNSLOTH_AVAILABLE


def load_unsloth_model(
    model_name: str,
    max_seq_length: int = 2048,
    dtype: str | None = None,
    load_in_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0,
    target_modules: list[str] | None = None,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """
    Load a model using Unsloth's FastLanguageModel.
    
    Args:
        model_name: Model name or path
        max_seq_length: Maximum sequence length
        dtype: Data type (None for auto)
        load_in_4bit: Whether to use 4-bit quantization
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout (must be 0 for Unsloth)
        target_modules: Target modules for LoRA
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if not UNSLOTH_AVAILABLE:
        raise ImportError("Unsloth is not installed. Please run: uv sync --extra unsloth")

    print_info(f"Loading Unsloth model: {model_name}")
    log_gpu_memory("Before Unsloth load: ")

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto detection
            load_in_4bit=load_in_4bit,
            **kwargs,
        )

        print_info("Applying Unsloth LoRA adapters...")
        
        # Unsloth handles PEFT internally for optimization
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout, # Optimized to 0
            bias="none",
            # Use gradient checkpointing "unsloth" for long context
            use_gradient_checkpointing="unsloth", 
            random_state=3407,
        )

        print_success(f"Unsloth model loaded and adapted: {model_name}")
        log_gpu_memory("After Unsloth load: ")
        
        return model, tokenizer

    except Exception as e:
        print_error(f"Failed to load Unsloth model: {e}")
        raise
