# 🚀 FTune - Easy Model Fine-Tuning

> No-code LLM fine-tuning with a beautiful wizard UI. Train your own AI models in minutes!

## Installation

```bash
# Basic install
uv sync

# With UI (Gradio web interface)
uv sync --extra ui

# With Unsloth (2-5x faster, Linux/Colab only)
uv sync --extra unsloth

# Full install (dev + ui + unsloth)
uv sync --all-extras
```

## 🚀 Run on Colab (Recommended for GPU)

Use Google Colab's free T4 GPU for faster training with Unsloth.

### Quick Start (3 commands)

```bash
# In Colab notebook:
!git clone https://github.com/khedhar108/Finetune-Compiler.git
%cd Finetune-Compiler
!pip install uv -q && uv sync --extra ui --extra unsloth
!uv run ftune --share
```

Then **click the `gradio.live` link** to open the Wizard UI.

### Using the Notebook

1. Open `notebooks/vscode_colab_launcher.ipynb` in Colab
2. Run all cells
3. Click the public URL

> ⚠️ **Note:** Don't stop Cell 4 - the UI server runs there. Link expires in 1 week.

## Quick Commands

| Command | Description |
|---------|-------------|
| `uv run ftune` | 🎯 **Launch FTune Wizard UI** (Simple & Default) |
| `uv run ftune-cli train ...` | Train using the CLI |
| `uv run ftune-cli infer ...` | Test your fine-tuned model |

### UI Options

| Command | Port | Description |
|---------|------|-------------|
| `uv run ftune` | 7862 | **FTune Wizard** (step-by-step) ✨ |
| `uv run ftune-ui` | 7862 | Shortcut for Wizard UI |


## Full CLI Commands

```bash
# Training
uv run ftune-cli train --config configs/default.json
uv run ftune-cli train --config my_config.json --resume ./checkpoint

# Inference (test your model)
uv run ftune-cli infer --model ./output --prompt "What is AI?"
uv run ftune-cli infer --model ./output --interactive

# Evaluation
uv run ai-compile evaluate --model ./output --dataset test.csv

# Export
uv run ai-compile export --model ./output --format gguf

# Deploy to HuggingFace
uv run ai-compile deploy --model ./output --repo your-username/model-name

# UI
uv run ai-compile ui      # Classic
uv run ai-compile ui2     # Wizard

# Info
uv run ai-compile info
```

## Data Preparation

```bash
# Download from Google Drive
uv run python scripts/download_gdrive.py --folder-id YOUR_ID --output-dir data/

# Prepare audio for ASR (convert to 16kHz WAV)
uv run python scripts/prepare_audio_dataset.py \
    --input-csv data/raw.csv \
    --output-dir data/prepared
```

## Core Commands (Fallback)

If shortcuts don't work, use these direct commands:

```bash
# Install UV (first time only)
# Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/Mac:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync                          # Basic
uv sync --extra ui               # With Gradio UI
uv sync --extra unsloth          # With Unsloth (Linux only)
uv sync --dev                    # With dev tools

# Run CLI directly
uv run python -m engine.cli.main train --config configs/default.json
uv run python -m engine.cli.main info
uv run python -m engine.cli.main ui

# Run tests directly
uv run python -m pytest tests/ -v

# Run linting directly
uv run python -m ruff check engine/
uv run python -m mypy engine/ --ignore-missing-imports

# Launch UI directly (without CLI)
uv run python -c "from engine.ui import launch_ui; launch_ui()"
```

## Project Structure

```
ai-compiler/
├── engine/              # Core modules
│   ├── data/            # Data loading
│   ├── models/          # Model loading, LoRA
│   ├── training/        # Training loop
│   ├── evaluation/      # Metrics
│   ├── inference/       # Model testing
│   ├── ui/              # Gradio Classic UI
│   ├── ui_v2/           # Gradio Wizard UI
│   ├── cli/             # Typer CLI
│   └── utils/           # Config, logging, checkpoint
├── configs/             # JSON configs
├── scripts/             # Utility scripts
├── tests/               # Test suite
├── docs/                # Documentation
└── pyproject.toml       # All configs
```

## Tips & Requirements

### Disk Space

| Component | Size |
|-----------|------|
| Base model | 2-8 GB (cached once) |
| **LoRA adapter (output)** | 10-50 MB only! |
| Checkpoints | Auto-cleaned |

### Custom Cache Location

By default, models are cached in `C:\Users\{you}\.cache\huggingface\`. 
To use a different drive:

```powershell
# Windows - set before running
$env:HF_HOME = "D:\models\cache"

# Or add to your profile permanently:
[System.Environment]::SetEnvironmentVariable("HF_HOME", "D:\models\cache", "User")
```

```bash
# Linux/Mac
export HF_HOME="/path/to/cache"
```

### Running Environments

| Environment | GPU | Notes |
|-------------|-----|-------|
| **Local Windows** | Your NVIDIA GPU | Needs ~10 GB disk |
| **Google Colab** | Free T4 GPU | Uses cloud storage |
| **VS Code + Colab** | Free T4 GPU | Best of both |

### Unsloth (2-5x Faster)

```bash
# Linux/Colab only
uv sync --extra unsloth

# Windows - auto-falls back to PEFT (works fine)
uv sync
```

### Common Issues

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce `batch_size` to 1-2 |
| Slow on Windows | Normal without Unsloth |
| Model not found | Check path or HF login |
| UI won't open | Run `uv sync --extra ui` |

### 🛑 Stopping the Server

If you need to stop all running FTune instances (e.g., if files are blocked), run:

```bash
uv run kill-ftune
```

### Resume Training

Training auto-resumes from checkpoints:
```bash
# Interrupted? Just run again:
uv run train
# → Auto-resumes from last checkpoint!
```

## Documentation

- [**Quick Start**](docs/quick-start.md) ← 5-minute demo
- [**Getting Started**](docs/getting-started.md) ← Full guide
- [**ASR Medical Guide**](docs/asr-medical-guide.md) ← Whisper for medical transcription
- [Architecture](docs/core-engine.md)
- [**Frontend Architecture (v2)**](docs/ui-architecture.md)
- [Commands Reference](COMMANDS.md)

## License

MIT

