"""
Export utilities for the AI Compiler Core Engine.
Handles conversion to GGUF and other formats using Unsloth.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Literal

from engine.utils.logging import get_logger, print_info, print_success, print_error, print_warning
from engine.utils.unsloth import is_unsloth_available

logger = get_logger(__name__)

def export_to_gguf(
    model_path: str,
    output_path: str | None = None,
    quantization: str = "q4_k_m",
) -> bool:
    """
    Export a model to GGUF format using Unsloth.
    
    Args:
        model_path: Path to the trained model (adapter or merged)
        output_path: Output file path (e.g. model.gguf)
        quantization: GGUF quantization method (q4_k_m, q8_0, f16)
        
    Returns:
        True if successful, False otherwise
    """
    print_info(f"Starting GGUF Export for: {model_path}")
    
    if not is_unsloth_available():
        print_error("❌ GGUF export requires Unsloth (Linux/Colab only).")
        print_info("Install with: uv sync --extra unsloth")
        return False

    try:
        from unsloth import FastLanguageModel
        
        # Load the model
        print_info("Loading model for export...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = 2048, # Standard for export
            dtype = None,
            load_in_4bit = True,
        )
        
        # Define output filename
        if output_path is None:
            output_path = f"{str(Path(model_path).name)}-{quantization}.gguf"
            
        print_info(f"Converting to GGUF ({quantization})...")
        print_info("This may take a few minutes...")
        
        # Save to GGUF
        model.save_pretrained_gguf(
            output_path,
            tokenizer,
            quantization_method = quantization
        )
        
        print_success(f"✅ Export Complete: {output_path}")
        print_info("You can now run this model with Ollama:")
        print_info(f"  > ollama create my-model -f Modelfile")
        print_info(f"  > FROM ./{output_path}")
        
        return True
        
    except Exception as e:
        print_error(f"Export failed: {e}")
        return False
