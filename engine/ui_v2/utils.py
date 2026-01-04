"""
Utility functions for AI Compiler UI v2.
"""

import os
from typing import Dict, List, Any, Optional
from engine.utils.memory import get_gpu_memory


# ============================================================================
# Dataset Analysis & Format Detection
# ============================================================================

def analyze_dataset(dataset_id: str, split: str = "train") -> Dict[str, Any]:
    """
    Analyze a HuggingFace dataset and detect its format.
    
    Returns:
        dict with keys:
        - columns: list of column names
        - modalities: detected modalities (text, audio, image)
        - suggested_format: best format for training
        - sample_rows: first 3 rows as preview
        - available_splits: list of splits in this dataset
        - used_split: which split was actually loaded
        - error: error message if any
    """
    try:
        from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
        
        # Step 1: Get available splits for this dataset
        try:
            available_splits = get_dataset_split_names(dataset_id)
        except Exception:
            # Fallback: try default config
            available_splits = ["train"]  # Assume train exists
        
        # Step 2: Choose which split to load
        if split in available_splits:
            use_split = split
        elif available_splits:
            use_split = available_splits[0]  # Fallback to first available
        else:
            use_split = "train"  # Last resort default
        
        # Step 3: Load a small sample (first 5 rows)
        ds = load_dataset(dataset_id, split=f"{use_split}[:5]", trust_remote_code=True)
        
        columns = ds.column_names
        features = ds.features
        
        # Detect modalities based on column names and types
        modalities = []
        format_hints = []
        
        # Check for audio columns
        audio_cols = [c for c in columns if any(k in c.lower() for k in ["audio", "speech", "wav", "mp3"])]
        if audio_cols or any("Audio" in str(features[c]) for c in columns):
            modalities.append("audio")
        
        # Check for image columns
        image_cols = [c for c in columns if any(k in c.lower() for k in ["image", "img", "photo", "picture"])]
        if image_cols or any("Image" in str(features[c]) for c in columns):
            modalities.append("image")
        
        # Check for text format patterns
        col_lower = [c.lower() for c in columns]
        
        if "instruction" in col_lower and ("output" in col_lower or "response" in col_lower):
            format_hints.append("alpaca")
            modalities.append("text") if "text" not in modalities else None
        
        if "messages" in col_lower:
            # Check if it's ChatML format
            try:
                sample_msg = ds[0].get("messages", [])
                if isinstance(sample_msg, list) and len(sample_msg) > 0:
                    if isinstance(sample_msg[0], dict) and "role" in sample_msg[0]:
                        format_hints.append("chatml")
                        modalities.append("text") if "text" not in modalities else None
            except:
                pass
        
        if any(t in col_lower for t in ["transcription", "sentence", "text", "transcript"]):
            if "audio" in modalities:
                format_hints.append("audio")  # ASR format
            else:
                format_hints.append("completion")
                modalities.append("text") if "text" not in modalities else None
        
        # Determine suggested format
        if "audio" in format_hints:
            suggested_format = "audio"
        elif "chatml" in format_hints:
            suggested_format = "chatml"
        elif "alpaca" in format_hints:
            suggested_format = "alpaca"
        else:
            suggested_format = "completion"
        
        # Build sample preview (first 3 rows)
        sample_rows = []
        for i in range(min(3, len(ds))):
            row = {}
            for col in columns[:5]:  # Limit to 5 columns for preview
                val = ds[i][col]
                # Truncate long values
                if isinstance(val, str):
                    row[col] = val[:100] + "..." if len(val) > 100 else val
                elif isinstance(val, dict):
                    row[col] = str(val)[:80] + "..."
                elif isinstance(val, list):
                    row[col] = f"[List: {len(val)} items]"
                elif hasattr(val, 'shape'):  # Audio/Image arrays
                    row[col] = f"[{type(val).__name__}: {val.shape if hasattr(val, 'shape') else 'binary'}]"
                else:
                    row[col] = str(val)[:50]
            sample_rows.append(row)
        
        return {
            "columns": columns,
            "modalities": list(set(modalities)) if modalities else ["text"],
            "suggested_format": suggested_format,
            "sample_rows": sample_rows,
            "num_rows": len(ds),
            "available_splits": available_splits,
            "used_split": use_split,
            "error": None
        }
        
    except Exception as e:
        return {
            "columns": [],
            "modalities": [],
            "suggested_format": "alpaca",
            "sample_rows": [],
            "num_rows": 0,
            "error": str(e)
        }


def format_preview_table(sample_rows: List[Dict]) -> str:
    """Convert sample rows to markdown table for display."""
    if not sample_rows:
        return "*No preview available*"
    
    columns = list(sample_rows[0].keys())
    
    # Build markdown table
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    
    rows = []
    for row in sample_rows:
        row_str = "| " + " | ".join(str(row.get(c, ""))[:40] for c in columns) + " |"
        rows.append(row_str)
    
    return "\n".join([header, separator] + rows)


def check_hf_token() -> str:
    """Check if HuggingFace token is set (environment or disk)."""
    # Check environment first
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    # Check disk (standard HuggingFace location)
    if not token:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_path):
            try:
                with open(token_path, "r") as f:
                    token = f.read().strip()
                if token:
                    os.environ["HF_TOKEN"] = token  # Also set in env for this session
            except Exception:
                pass
    
    if token:
        return f"✅ Token found: {token[:8]}... (persisted)"
    return "⚠️ No token set. Add your HuggingFace token below for private datasets."


def save_hf_token(token: str) -> str:
    """Save HuggingFace token to environment AND disk for persistence."""
    if not token or not token.strip():
        return "❌ Please enter a token."
    
    token = token.strip()
    
    if not token.startswith("hf_"):
        return "❌ Invalid token format. Must start with 'hf_'"
    
    # Set in environment for current session
    os.environ["HF_TOKEN"] = token
    
    # Persist to disk (standard HuggingFace location)
    try:
        token_dir = os.path.expanduser("~/.cache/huggingface")
        os.makedirs(token_dir, exist_ok=True)
        token_path = os.path.join(token_dir, "token")
        with open(token_path, "w") as f:
            f.write(token)
        return f"✅ Token saved and persisted: {token[:8]}...\n\n(Stored at `~/.cache/huggingface/token` - will auto-load next time)"
    except Exception as e:
        return f"⚠️ Token set for this session only (disk save failed: {e})"


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
    output_dir, train_split,
    trust_remote_code=False,
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
            "train_ratio": train_split / 100.0,  # Convert % to ratio
        },
        "model": {
            "name": model_name,
            "task": "asr" if "whisper" in model_name.lower() else "causal_lm",
            "quantization": quantization,
            "max_seq_length": max_seq_length,
            "trust_remote_code": trust_remote_code,
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
