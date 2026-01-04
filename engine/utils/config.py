"""
Configuration management for the AI Compiler Core Engine.

This module provides JSON-based configuration loading, validation, and merging.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ProjectConfig(BaseModel):
    """Project-level configuration."""

    name: str = Field(default="my-finetune", description="Project name")
    output_dir: str = Field(default="./output", description="Output directory")
    seed: int = Field(default=42, description="Random seed for reproducibility")


class DataConfig(BaseModel):
    """Data loading configuration."""

    source: Literal["huggingface", "csv", "json", "gdrive"] = Field(
        default="csv", description="Data source type"
    )
    path: str = Field(description="Dataset path or HuggingFace dataset name")
    format: Literal["alpaca", "chatml", "completion", "custom", "audio"] = Field(
        default="alpaca", description="Prompt format"
    )
    test_split: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Test split ratio"
    )
    train_ratio: float = Field(
        default=1.0, ge=0.01, le=1.0, description="Dataset usage ratio (0.01 to 1.0)"
    )
    max_samples: int | None = Field(
        default=None, description="Maximum samples to use (None for all)"
    )
    text_column: str = Field(default="text", description="Column name for text")
    
    # Column mapping for custom formats
    instruction_column: str = Field(default="instruction", description="Instruction column")
    input_column: str = Field(default="input", description="Input column")
    output_column: str = Field(default="output", description="Output column")
    
    # Audio/ASR specific columns
    audio_column: str = Field(default="audio", description="Column for audio file paths (ASR)")
    transcription_column: str = Field(default="transcription", description="Column for transcriptions (ASR)")


class ModelConfig(BaseModel):
    """Model loading configuration."""

    name: str = Field(
        default="unsloth/Llama-3.2-1B", description="Model name or path"
    )
    task: Literal["causal_lm", "seq2seq", "asr", "tts"] = Field(
        default="causal_lm", description="Model task type (causal_lm, seq2seq, asr, tts)"
    )
    quantization: Literal["none", "4bit", "8bit"] = Field(
        default="4bit", description="Quantization mode"
    )
    max_seq_length: int = Field(
        default=2048, ge=128, le=8192, description="Maximum sequence length"
    )
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = Field(
        default="auto", description="Model dtype"
    )
    trust_remote_code: bool = Field(
        default=False, description="Trust remote code from HuggingFace"
    )


class LoRAConfig(BaseModel):
    """LoRA adapter configuration."""

    r: int = Field(default=16, ge=4, le=256, description="LoRA rank")
    alpha: int = Field(default=32, ge=8, le=512, description="LoRA alpha")
    dropout: float = Field(
        default=0.05, ge=0.0, le=0.5, description="LoRA dropout"
    )
    target_modules: list[str] = Field(
        default=["q_proj", "v_proj", "k_proj", "o_proj"],
        description="Target modules for LoRA",
    )
    bias: Literal["none", "all", "lora_only"] = Field(
        default="none", description="Bias training mode"
    )
    task_type: str = Field(default="CAUSAL_LM", description="Task type")


class TrainingConfig(BaseModel):
    """Training loop configuration."""

    epochs: int = Field(default=3, ge=1, le=100, description="Number of epochs")
    batch_size: int = Field(default=4, ge=1, le=64, description="Batch size")
    gradient_accumulation_steps: int = Field(
        default=4, ge=1, le=64, description="Gradient accumulation steps"
    )
    learning_rate: float = Field(
        default=2e-4, ge=1e-6, le=1e-2, description="Learning rate"
    )
    weight_decay: float = Field(
        default=0.01, ge=0.0, le=0.5, description="Weight decay"
    )
    warmup_ratio: float = Field(
        default=0.03, ge=0.0, le=0.5, description="Warmup ratio"
    )
    fp16: bool = Field(default=True, description="Use FP16 mixed precision")
    bf16: bool = Field(default=False, description="Use BF16 mixed precision")
    gradient_checkpointing: bool = Field(
        default=True, description="Enable gradient checkpointing"
    )
    max_grad_norm: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Max gradient norm"
    )
    optim: str = Field(default="adamw_8bit", description="Optimizer")


class LoggingConfig(BaseModel):
    """Logging and checkpointing configuration."""

    log_steps: int = Field(default=10, ge=1, description="Log every N steps")
    save_steps: int = Field(default=100, ge=1, description="Save every N steps")
    eval_steps: int = Field(default=100, ge=1, description="Evaluate every N steps")
    save_total_limit: int = Field(
        default=3, ge=1, le=20, description="Max checkpoints to keep"
    )
    logging_dir: str = Field(default="./logs", description="Logging directory")
    report_to: list[str] = Field(
        default=["none"], description="Reporting integrations"
    )


class Config(BaseModel):
    """
    Main configuration class for the AI Compiler Core Engine.
    
    This class combines all configuration sections and provides
    loading, validation, and merging functionality.
    """

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        """Load configuration from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create configuration from a dictionary."""
        return cls.model_validate(data)

    def to_json(self, path: str | Path) -> None:
        """Save configuration to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary."""
        return self.model_dump()

    def merge_with(self, overrides: dict[str, Any]) -> "Config":
        """
        Create a new config by merging this config with overrides.
        
        Args:
            overrides: Dictionary of values to override
            
        Returns:
            New Config instance with merged values
        """
        current = self.model_dump()
        
        def deep_merge(base: dict, updates: dict) -> dict:
            result = base.copy()
            for key, value in updates.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged = deep_merge(current, overrides)
        return Config.model_validate(merged)


def load_config(path: str | Path) -> Config:
    """
    Load configuration from a JSON file.
    
    This is a convenience function that wraps Config.from_json().
    
    Args:
        path: Path to the JSON configuration file
        
    Returns:
        Loaded and validated Config instance
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValidationError: If the configuration is invalid
    """
    return Config.from_json(path)


def get_default_config() -> dict[str, Any]:
    """
    Get the default configuration as a dictionary.
    
    Returns:
        Dictionary with all default values
    """
    return {
        "project": ProjectConfig().model_dump(),
        "data": {
            "source": "csv",
            "path": "./data/train.csv",
            "format": "alpaca",
        },
        "model": ModelConfig().model_dump(),
        "lora": LoRAConfig().model_dump(),
        "training": TrainingConfig().model_dump(),
        "logging": LoggingConfig().model_dump(),
    }
