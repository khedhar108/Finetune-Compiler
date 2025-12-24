"""
LoRA adapter utilities for the AI Compiler Core Engine.

Provides functions for creating, applying, saving, and loading LoRA adapters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from transformers import PreTrainedModel

from engine.utils.logging import get_logger, print_info, print_success
from engine.utils.memory import log_gpu_memory

logger = get_logger(__name__)


def create_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
    bias: Literal["none", "all", "lora_only"] = "none",
    task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    """
    Create a LoRA configuration.
    
    Args:
        r: LoRA rank
        alpha: LoRA alpha (scaling factor)
        dropout: LoRA dropout
        target_modules: Modules to apply LoRA to
        bias: Bias training mode
        task_type: Task type (CAUSAL_LM, SEQ_CLS, etc.)
        
    Returns:
        LoraConfig instance
    """
    if target_modules is None:
        # Default target modules for most LLMs
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # Map task type string to TaskType enum
    task_type_map = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_CLS": TaskType.SEQ_CLS,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        "TOKEN_CLS": TaskType.TOKEN_CLS,
    }
    
    peft_task_type = task_type_map.get(task_type, TaskType.CAUSAL_LM)
    
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=peft_task_type,
    )
    
    print_info(f"Created LoRA config: r={r}, alpha={alpha}, dropout={dropout}")
    
    return config


def apply_lora(
    model: PreTrainedModel,
    lora_config: LoraConfig,
    prepare_for_kbit: bool = True,
) -> PeftModel:
    """
    Apply LoRA to a model.
    
    Args:
        model: Base model
        lora_config: LoRA configuration
        prepare_for_kbit: Whether to prepare model for k-bit training
        
    Returns:
        PeftModel with LoRA applied
    """
    print_info("Applying LoRA to model...")
    log_gpu_memory("Before LoRA: ")
    
    # Prepare model for k-bit training if using quantization
    if prepare_for_kbit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    
    # Apply LoRA
    peft_model = get_peft_model(model, lora_config)
    
    # Log trainable parameters
    trainable_params, all_params = get_trainable_params(peft_model)
    trainable_percent = 100 * trainable_params / all_params
    
    print_success(
        f"LoRA applied: {trainable_params:,} trainable params "
        f"({trainable_percent:.2f}% of {all_params:,} total)"
    )
    log_gpu_memory("After LoRA: ")
    
    return peft_model


def get_trainable_params(model: PreTrainedModel | PeftModel) -> tuple[int, int]:
    """
    Get the number of trainable and total parameters.
    
    Args:
        model: Model to analyze
        
    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return trainable_params, all_params


def save_adapter(
    model: PeftModel,
    output_path: str | Path,
    save_config: bool = True,
) -> None:
    """
    Save a LoRA adapter to disk.
    
    Args:
        model: PeftModel with LoRA
        output_path: Output directory
        save_config: Whether to save the config
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print_info(f"Saving adapter to: {output_path}")
    
    model.save_pretrained(output_path)
    
    print_success(f"Adapter saved to: {output_path}")


def load_adapter(
    base_model: PreTrainedModel,
    adapter_path: str | Path,
    is_trainable: bool = False,
) -> PeftModel:
    """
    Load a LoRA adapter and apply it to a base model.
    
    Args:
        base_model: Base model to apply adapter to
        adapter_path: Path to saved adapter
        is_trainable: Whether to make the adapter trainable
        
    Returns:
        PeftModel with adapter loaded
    """
    adapter_path = Path(adapter_path)
    
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    
    print_info(f"Loading adapter from: {adapter_path}")
    
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=is_trainable,
    )
    
    print_success("Adapter loaded")
    
    return peft_model


def merge_and_unload(
    model: PeftModel,
) -> PreTrainedModel:
    """
    Merge LoRA weights into the base model and unload adapter.
    
    This creates a standalone model without the LoRA adapter layer.
    
    Args:
        model: PeftModel with LoRA
        
    Returns:
        Base model with merged weights
    """
    print_info("Merging LoRA weights into base model...")
    
    merged_model = model.merge_and_unload()
    
    print_success("LoRA weights merged")
    
    return merged_model
