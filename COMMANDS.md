# Command Log

This file tracks all terminal commands for the AI Compiler.

---

## Quick Start

```bash
# Install
uv sync                          # Basic
uv sync --extra ui               # With Gradio UI
uv sync --all-extras             # Everything

# Verify
uv run ai-compile info
```

---

## Training Commands

```bash
# Train with default config
uv run train

# Train with specific config
uv run ai-compile train --config configs/default.json

# Train with auto-resume (default)
uv run ai-compile train --config configs/default.json

# Manual resume from checkpoint
uv run ai-compile train --config configs/default.json --resume ./output/checkpoint-100

# Disable auto-resume
uv run ai-compile train --config configs/default.json --no-auto-resume

# Dry run (validate config only)
uv run ai-compile train --config configs/default.json --dry-run

# Colab-optimized training
uv run train-colab

# ASR training
uv run train-asr
```

---

## Inference Commands

```bash
# Single prompt
uv run ai-compile infer --model ./output --prompt "What is AI?"

# Interactive chat mode
uv run ai-compile infer --model ./output --interactive

# With options
uv run ai-compile infer --model ./output --prompt "Hello" \
    --temperature 0.7 \
    --max-tokens 256 \
    --format alpaca

# From HuggingFace model
uv run ai-compile infer --model username/my-model --prompt "Test"
```

---

## Deploy Commands

```bash
# Deploy to HuggingFace Hub
uv run ai-compile deploy --model ./output --repo username/my-model

# Deploy as private
uv run ai-compile deploy --model ./output --repo username/my-model --private

# With token
uv run ai-compile deploy --model ./output --repo username/my-model --token hf_xxxxx
```

---

## UI Commands

```bash
# Classic UI (port 7860)
uv run ui

# Wizard UI (port 7862)
uv run ui2

# With share link
uv run ui-share
uv run ui2-share

# Custom port
uv run ai-compile ui --port 8080
```

---

## Development Commands

```bash
# Run tests
uv run test

# Linting
uv run lint

# Formatting
uv run format

# Type checking
uv run typecheck

# All checks
uv run lint && uv run typecheck && uv run test
```

---

## Data Preparation

```bash
# Download from Google Drive
uv run python scripts/download_gdrive.py \
    --folder-id FOLDER_ID \
    --output-dir data/

# Prepare audio dataset
uv run python scripts/prepare_audio_dataset.py \
    --input-csv data/raw.csv \
    --output-dir data/prepared
```

---

## Environment Variables

```bash
# HuggingFace tokens
$env:HF_TOKEN = "hf_xxxxx"           # General (read)
$env:HF_READ_TOKEN = "hf_xxxxx"      # Read-only
$env:HF_WRITE_TOKEN = "hf_xxxxx"     # Write (for deploy)

# Or login interactively
uv run huggingface-cli login
```

---

## Utility Commands

```bash
# Show system info
uv run info

# Generate default config
uv run init

# Show version
uv run ai-compile --version

# Show help
uv run ai-compile --help
uv run ai-compile train --help
uv run ai-compile infer --help
```

---

## Notes

- All commands assume you're in the `ai-compiler/` directory
- UV automatically manages the virtual environment
- Use `uv run` prefix to run commands in the virtual environment
- Checkpoints are auto-saved and auto-resumed
