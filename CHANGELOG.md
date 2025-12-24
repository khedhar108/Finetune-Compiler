# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Core Engine Modules**
  - `engine/data/` - Data loading (HuggingFace, CSV, JSON, GDrive)
  - `engine/data/formats.py` - Prompt formatting (Alpaca, ChatML, completion)
  - `engine/models/loader.py` - Model loading with 4-bit/8-bit quantization
  - `engine/models/adapters.py` - LoRA/QLoRA adapter management (PEFT)
  - `engine/training/trainer.py` - Training orchestration with HuggingFace Trainer
  - `engine/evaluation/metrics.py` - Perplexity and accuracy metrics
  - `engine/utils/config.py` - Pydantic-based JSON configuration system
  - `engine/utils/logging.py` - Rich console logging with progress bars
  - `engine/utils/memory.py` - GPU memory monitoring utilities
  - `engine/cli/main.py` - Typer CLI (train, evaluate, export, init, info)

- **Configuration Files**
  - `configs/default.json` - Default training configuration
  - `configs/colab_optimized.json` - Optimized for Google Colab T4 GPU
  - `configs/asr_whisper.json` - Example for ASR/Whisper fine-tuning

- **Utility Scripts**
  - `scripts/prepare_audio_dataset.py` - Convert audio to 16kHz WAV for ASR
  - `scripts/download_gdrive.py` - Download datasets from Google Drive

- **Documentation**
  - `docs/core-engine.md` - Architecture and module specifications
  - `docs/deployment.md` - Local, cloud, and HuggingFace deployment guides
  - `docs/resources.md` - Library references and links

- **ASR Support**
  - Added `task` field to model config (causal_lm, seq2seq, asr, tts)
  - Added `audio_column` and `transcription_column` for audio datasets
  - Added `audio` format type for ASR data

- **Unsloth Integration**
  - Added `engine/utils/unsloth.py` for auto-detection
  - Auto-installs Unsloth on Linux/Colab environments
  - 2-5x faster training with optimized LoRA
  - Updated notebook with environment auto-detection

- **Visual UI (Gradio)**
  - Added `engine/ui/app.py` - visual interface for compiler
  - Config builder with sliders and dropdowns
  - Live training logs streaming
  - GPU memory monitor
  - Added `ui` CLI command: `ai-compile ui`
  - Gradio as optional dependency: `uv sync --extra ui`

- **Wizard UI v2 (Enhanced)**
  - Added `engine/ui_v2/app.py` - step-by-step wizard interface
  - 4-step flow: Model → Data → Train → Deploy
  - Collapsible logs panel
  - HuggingFace token input & dataset quick-select
  - Progress bar with percentage
  - One-click deploy to HuggingFace Hub
  - Added `ui2` CLI command: `ai-compile ui2`

- **HuggingFace Integration**
  - Added `engine/utils/huggingface.py` for Hub integration
  - Read/Write token management
  - `upload_to_hub()` - deploy models to Hub
  - `load_from_hub()` - download models
  - Added `deploy` CLI command: `ai-compile deploy --model ./output --repo username/model`
  - Returns model URL for immediate use

- **Inference Engine**
  - Added `engine/inference/` module for running trained models
  - `InferenceEngine` class for easy model loading
  - Streaming text generation
  - Interactive chat mode
  - Added `infer` CLI command: `ai-compile infer --model ./output --interactive`

- **Checkpoint & Resume**
  - Added `engine/utils/checkpoint.py` for checkpoint management
  - Training lock mechanism (prevents re-computation)
  - Auto-resume from last checkpoint
  - Checkpoint info saving (step, epoch, loss)
  - Added `--auto-resume` flag to train command

- **Testing**
  - Test scaffolding with pytest (test_config.py, test_data.py, test_models.py)

### Changed
- Restructured project from `ai-compiler-platform/core-engine/` to `ai-compiler/`
- Separated business docs to `medical-asr/docs/` (app-specific)
- Moved vision and roadmap to root `voice-ai/` folder

### Technical Details
- Python 3.10+ required
- UV package manager for dependency management
- 70 packages installed (PyTorch 2.9.1, Transformers 4.57.3, PEFT 0.18.0)
- Type checking with mypy configured
- Linting with ruff configured

---

## [0.1.0] - 2024-12-21

### Added
- Initial release of AI Compiler Core Engine
- CLI tool `ai-compile` with commands: train, evaluate, export, init, info
- Support for fine-tuning on 4-8GB GPUs using LoRA/QLoRA
- JSON-based configuration system
- Colab quickstart notebook

[Unreleased]: https://github.com/username/ai-compiler/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/username/ai-compiler/releases/tag/v0.1.0

