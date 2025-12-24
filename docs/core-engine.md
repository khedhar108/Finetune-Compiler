# 🔧 Core Engine Architecture

> The heart of the AI Compiler Platform - a modular, efficient LLM fine-tuning engine.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI LAYER                               │
│  ai-compile train | evaluate | export | serve                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CONFIG MANAGER                             │
│  Load JSON → Validate → Merge Defaults → Distribute to Modules  │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   DATA MODULE   │ │  MODEL MODULE   │ │ TRAINING MODULE │
│                 │ │                 │ │                 │
│ • Loader        │ │ • Loader        │ │ • Trainer       │
│ • Preprocessor  │ │ • Adapters      │ │ • Callbacks     │
│ • Formats       │ │ • Registry      │ │ • Strategies    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION MODULE                           │
│  Metrics (Accuracy, BLEU, Perplexity) → Validation → Reports    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                   │
│  Checkpoints | Adapters | Merged Models | GGUF | Model Cards    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Specifications

### 1. Data Module (`engine/data/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `loader.py` | Load datasets from various sources | `load_hf()`, `load_csv()`, `load_gdrive()` |
| `preprocessor.py` | Tokenization & text processing | `tokenize()`, `truncate()`, `pad()` |
| `formats.py` | Prompt template formatting | `alpaca_format()`, `chatml_format()` |

**Supported Data Sources:**
```python
# HuggingFace Hub
{"source": "huggingface", "path": "tatsu-lab/alpaca"}

# Local CSV
{"source": "csv", "path": "./data/train.csv"}

# Local JSON/JSONL
{"source": "json", "path": "./data/train.jsonl"}

# Google Drive
{"source": "gdrive", "file_id": "1ABC123..."}
```

**Supported Formats:**
```python
# Alpaca (instruction-tuning)
{"instruction": "...", "input": "...", "output": "..."}

# ChatML (conversational)
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

# Completion (simple text)
{"text": "..."}
```

---

### 2. Model Module (`engine/models/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `loader.py` | Load base models with quantization | `load_model()`, `get_device_map()` |
| `adapters.py` | LoRA/QLoRA configuration & application | `create_lora()`, `apply_lora()`, `save_adapter()` |
| `registry.py` | Supported models catalog | `get_model_info()`, `list_models()` |

**Quantization Options:**
```json
{
  "quantization": "none",     // Full precision (needs 16GB+)
  "quantization": "8bit",     // 8-bit (needs 8GB+)
  "quantization": "4bit"      // 4-bit QLoRA (needs 4GB+)
}
```

**LoRA Configuration:**
```json
{
  "lora": {
    "r": 16,              // Rank (8-64 typical)
    "alpha": 32,          // Alpha (usually 2x rank)
    "dropout": 0.05,      // Dropout for regularization
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
  }
}
```

---

### 3. Training Module (`engine/training/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `trainer.py` | Training loop orchestration | `train()`, `resume()`, `save_checkpoint()` |
| `callbacks.py` | Custom training callbacks | `LoggingCallback`, `MemoryCallback` |
| `strategies.py` | Training strategy selection | `FullFineTune`, `LoRAStrategy`, `QLoRAStrategy` |

**Training Strategies:**

| Strategy | VRAM Required | Speed | Quality |
|----------|---------------|-------|---------|
| `full` | 24GB+ | Slow | Best |
| `lora` | 8GB+ | Fast | Good |
| `qlora` | 4GB+ | Medium | Good |

**Memory Optimization Techniques:**
- Gradient checkpointing (trade compute for memory)
- FP16/BF16 mixed precision
- Gradient accumulation (effective larger batches)
- Flash Attention 2 (when available)

---

### 4. Evaluation Module (`engine/evaluation/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `metrics.py` | Compute evaluation metrics | `accuracy()`, `bleu()`, `perplexity()` |
| `validator.py` | Validation during training | `validate()`, `early_stop_check()` |

**Supported Metrics:**
- **Text Generation**: Perplexity, BLEU, ROUGE
- **Classification**: Accuracy, F1, Precision, Recall
- **Custom**: User-defined metric functions

---

### 5. Utils Module (`engine/utils/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `config.py` | JSON config handling | `load()`, `validate()`, `merge()` |
| `logging.py` | Structured logging | `setup_logger()`, `log_metrics()` |
| `memory.py` | GPU memory utilities | `get_gpu_memory()`, `clear_cache()` |
| `unsloth.py` | Unsloth integration | `is_unsloth_available()`, `get_unsloth_model()` |

---

### 6. Unsloth Integration (Auto-Optimized Training)

The engine automatically detects if Unsloth is available and uses it for **2-5x faster training**.

**Auto-Detection Logic:**
```python
# engine/utils/unsloth.py
def is_unsloth_available() -> bool:
    # Only works on Linux with GPU
    if not sys.platform.startswith('linux'):
        return False
    try:
        import unsloth
        return True
    except ImportError:
        return False
```

**Environment Compatibility:**

| Environment | OS | Unsloth | Fallback |
|-------------|-----|---------|----------|
| Windows | Windows | ❌ | PEFT |
| Google Colab | Linux | ✅ | - |
| Linux Server | Linux | ✅ | - |
| VS Code + Colab Extension | Linux (remote) | ✅ | - |

**Performance Comparison:**

| Setting | Standard PEFT | With Unsloth |
|---------|---------------|--------------|
| Training speed | 1x | 2-5x faster |
| Memory usage | 100% | ~50% |
| LoRA rank | 8 | 16 |
| Target modules | 2 | 7 |

**Installation:**
```bash
# On Linux/Colab, install with Unsloth extra
uv sync --extra unsloth

# On Windows, standard install (PEFT fallback)
uv sync
```

## CLI Commands

```bash
# Train a model
ai-compile train --config config.json

# Resume training from checkpoint
ai-compile train --config config.json --resume ./checkpoints/step-1000

# Evaluate a model
ai-compile evaluate --model ./output --dataset test.csv --metrics accuracy,bleu

# Export to different formats
ai-compile export --model ./output --format gguf
ai-compile export --model ./output --format merged
ai-compile export --model ./output --format adapter

# Quick inference test
ai-compile infer --model ./output --prompt "What is machine learning?"
```

---

## Configuration Schema

See [configs/default.json](../core-engine/configs/default.json) for full schema.

```json
{
  "project": {
    "name": "my-finetune",
    "output_dir": "./output",
    "seed": 42
  },
  "data": {
    "source": "csv",
    "path": "./data/train.csv",
    "format": "alpaca",
    "test_split": 0.1,
    "max_samples": null
  },
  "model": {
    "name": "unsloth/Llama-3.2-1B",
    "quantization": "4bit",
    "max_seq_length": 2048
  },
  "lora": {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
  },
  "training": {
    "epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "fp16": true,
    "gradient_checkpointing": true,
    "max_grad_norm": 1.0
  },
  "logging": {
    "log_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
    "save_total_limit": 3
  }
}
```

---

## File Output Structure

After training completes:

```
output/
├── adapter_model/           # LoRA adapter weights
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── checkpoints/             # Training checkpoints
│   ├── checkpoint-100/
│   └── checkpoint-200/
├── logs/                    # Training logs
│   └── training.log
├── config.json              # Used configuration
├── training_args.json       # HuggingFace TrainingArguments
└── README.md                # Auto-generated model card
```

---

## Related Documents

- [Vision](./vision.md)
- [Deployment Guide](./deployment.md)
- [Business Use Cases](./business-use-cases.md)

---

## Official References & Documentation

| Library | Purpose | Link |
|---------|---------|------|
| **Transformers** | Model loading & training | [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers/index) |
| **Datasets** | Data loading & processing | [huggingface.co/docs/datasets](https://huggingface.co/docs/datasets/index) |
| **PEFT** | LoRA/QLoRA adapters | [github.com/huggingface/peft](https://github.com/huggingface/peft) |
| **Unsloth** | Optimized training | [docs.unsloth.ai](https://docs.unsloth.ai/) |
| **bitsandbytes** | Quantization (4-bit/8-bit) | [github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) |
| **Typer** | CLI framework | [typer.tiangolo.com](https://typer.tiangolo.com/) |
| **Evaluate** | Metrics library | [huggingface.co/docs/evaluate](https://huggingface.co/docs/evaluate) |
| **MLflow** | Experiment tracking | [mlflow.org](https://mlflow.org/) |
| **Gdown** | Google Drive downloads | [github.com/wkentaro/gdown](https://github.com/wkentaro/gdown) |
| **LangGraph** | Agent workflows | [github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) |
| **LangFuse** | LLM observability | [langfuse.com](https://www.langfuse.com/) |
| **UV** | Fast Python package manager | [docs.astral.sh/uv](https://docs.astral.sh/uv/) |
