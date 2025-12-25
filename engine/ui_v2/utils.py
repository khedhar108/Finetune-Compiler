"""
Utility functions for AI Compiler UI v2.
"""

import os
from engine.utils.memory import get_gpu_memory

def check_hf_token() -> str:
    """Check if HuggingFace token is set."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return f"✅ Token found: {token[:8]}..."
    return "⚠️ No token set. Set HF_TOKEN environment variable for private models."


def save_hf_token(token: str) -> str:
    """Save HuggingFace token to environment."""
    if token and token.startswith("hf_"):
        os.environ["HF_TOKEN"] = token
        return f"✅ Token saved: {token[:8]}..."
    return "❌ Invalid token. Must start with 'hf_'"


def get_gpu_status() -> str:
    """Get GPU status with clear recommendation."""
    try:
        stats = get_gpu_memory()
        if stats.get("available", False):
            gpu_name = stats['device_name']
            total = stats['total_gb']
            used = stats.get('reserved_gb', 0)
            free = stats.get('free_gb', total)
            
            # Determine if it's Colab T4 or local GPU
            is_t4 = "T4" in gpu_name
            is_colab = "T4" in gpu_name and total > 14  # T4 has ~15GB
            
            status = f"🎮 **{gpu_name}**\n"
            status += f"📊 Memory: {used:.1f}GB / {total:.1f}GB ({free:.1f}GB free)\n"
            
            if is_colab:
                status += "☁️ Colab T4 - Great for training!\n"
                status += "✅ Unsloth available (2-5x faster)"
            elif "RTX" in gpu_name or "GeForce" in gpu_name:
                status += f"🖥️ Local GPU - Ready to train!\n"
                status += "⚠️ Unsloth: Windows not supported, using PEFT (works fine)"
            else:
                status += "✅ Ready to train"
            
            return status
        return "❌ No GPU available - Use Colab for training"
    except:
        return "⚠️ Unable to check GPU - Check if PyTorch is installed"


def build_config(
    model_name, quantization, max_seq_length,
    lora_r, lora_alpha,
    data_source, data_path, data_format,
    epochs, batch_size, learning_rate,
    output_dir,
) -> dict:
    """Build configuration from wizard inputs."""
    return {
        "project": {
            "name": "wizard-training",
            "output_dir": output_dir,
            "seed": 42,
        },
        "data": {
            "source": data_source,
            "path": data_path,
            "format": data_format,
            "test_split": 0.1,
        },
        "model": {
            "name": model_name,
            "task": "asr" if "whisper" in model_name.lower() else "causal_lm",
            "quantization": quantization,
            "max_seq_length": max_seq_length,
        },
        "lora": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": 4,
            "learning_rate": learning_rate,
            "fp16": True,
            "gradient_checkpointing": True,
        },
        "logging": {
            "log_steps": 5,
            "save_steps": 100,
            "save_total_limit": 3,
        },
    }
