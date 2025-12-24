"""
Training orchestration for the AI Compiler Core Engine.

Provides the main Trainer class that coordinates data loading,
model setup, and training loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import (
    TrainingArguments,
    Trainer as HFTrainer,
    DataCollatorForLanguageModeling,
)

from engine.utils.config import Config
from engine.utils.logging import (
    get_logger,
    print_banner,
    print_success,
    print_error,
    print_info,
    log_config,
)
from engine.utils.memory import log_gpu_memory, clear_gpu_memory
from engine.data import load_dataset_from_source, format_dataset
from engine.data.loader import split_dataset
from engine.models.loader import load_model, load_tokenizer, prepare_model_for_training
from engine.models.adapters import create_lora_config, apply_lora, save_adapter

logger = get_logger(__name__)


class Trainer:
    """
    Main trainer class for fine-tuning LLMs.
    
    This class orchestrates:
    - Dataset loading and preprocessing
    - Model and tokenizer loading
    - LoRA adapter setup
    - Training loop execution
    - Checkpoint saving
    
    Example:
        config = Config.from_json("config.json")
        trainer = Trainer(config)
        trainer.train()
    """
    
    def __init__(self, config: Config) -> None:
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
    
    def setup_data(self) -> None:
        """Load and preprocess the dataset."""
        print_info("Setting up data...")
        
        # Load dataset
        dataset = load_dataset_from_source(
            source=self.config.data.source,
            path=self.config.data.path,
        )
        
        # Limit samples if specified
        if self.config.data.max_samples:
            dataset = dataset.select(range(min(len(dataset), self.config.data.max_samples)))
            print_info(f"Limited to {len(dataset)} samples")
        
        # Format dataset
        dataset = format_dataset(
            dataset,
            format_type=self.config.data.format,
            instruction_col=self.config.data.instruction_column,
            input_col=self.config.data.input_column,
            output_col=self.config.data.output_column,
            text_col=self.config.data.text_column,
        )
        
        # Tokenize
        print_info("Tokenizing dataset...")
        
        def tokenize_fn(examples: dict[str, list]) -> dict[str, list]:
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.model.max_seq_length,
                padding=False,
            )
        
        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        # Split into train/eval
        if self.config.data.test_split > 0:
            self.train_dataset, self.eval_dataset = split_dataset(
                tokenized,
                test_size=self.config.data.test_split,
                seed=self.config.project.seed,
            )
            print_info(f"Train: {len(self.train_dataset)}, Eval: {len(self.eval_dataset)}")
        else:
            self.train_dataset = tokenized
            self.eval_dataset = None
            print_info(f"Train: {len(self.train_dataset)} (no eval split)")
    
    def setup_model(self) -> None:
        """Load model and tokenizer, apply LoRA."""
        print_info("Setting up model...")
        
        # Load tokenizer
        self.tokenizer = load_tokenizer(
            self.config.model.name,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        
        # Load model
        self.model = load_model(
            model_name=self.config.model.name,
            quantization=self.config.model.quantization,
            max_seq_length=self.config.model.max_seq_length,
            dtype=self.config.model.dtype,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        
        # Prepare for training
        self.model = prepare_model_for_training(
            self.model,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
        )
        
        # Apply LoRA
        lora_config = create_lora_config(
            r=self.config.lora.r,
            alpha=self.config.lora.alpha,
            dropout=self.config.lora.dropout,
            target_modules=self.config.lora.target_modules,
            bias=self.config.lora.bias,
            task_type=self.config.lora.task_type,
        )
        
        self.model = apply_lora(
            self.model,
            lora_config,
            prepare_for_kbit=self.config.model.quantization != "none",
        )
    
    def get_training_args(self, resume_from: str | None = None) -> TrainingArguments:
        """Create HuggingFace TrainingArguments."""
        output_dir = Path(self.config.project.output_dir)
        
        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.training.epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_ratio=self.config.training.warmup_ratio,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            max_grad_norm=self.config.training.max_grad_norm,
            optim=self.config.training.optim,
            logging_steps=self.config.logging.log_steps,
            save_steps=self.config.logging.save_steps,
            eval_steps=self.config.logging.eval_steps if self.eval_dataset else None,
            evaluation_strategy="steps" if self.eval_dataset else "no",
            save_total_limit=self.config.logging.save_total_limit,
            logging_dir=self.config.logging.logging_dir,
            report_to=self.config.logging.report_to,
            seed=self.config.project.seed,
            remove_unused_columns=False,
            resume_from_checkpoint=resume_from,
        )
    
    def train(self, resume_from: str | None = None) -> None:
        """
        Run the training loop.
        
        Args:
            resume_from: Optional checkpoint path to resume from
        """
        print_banner()
        log_config(self.config.to_dict(), title="Training Configuration")
        log_gpu_memory("Initial: ")
        
        try:
            # Setup
            self.setup_model()
            self.setup_data()
            
            # Create training arguments
            training_args = self.get_training_args(resume_from)
            
            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Causal LM, not masked LM
            )
            
            # Create HuggingFace Trainer
            trainer = HFTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                data_collator=data_collator,
            )
            
            # Train
            print_info("Starting training...")
            trainer.train(resume_from_checkpoint=resume_from)
            
            # Save final model
            output_dir = Path(self.config.project.output_dir)
            adapter_dir = output_dir / "adapter_model"
            save_adapter(self.model, adapter_dir)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(output_dir)
            
            # Save config
            self.config.to_json(output_dir / "config.json")
            
            print_success(f"Training complete! Model saved to: {output_dir}")
            
        except Exception as e:
            print_error(f"Training failed: {e}")
            clear_gpu_memory()
            raise
        
        finally:
            log_gpu_memory("Final: ")
