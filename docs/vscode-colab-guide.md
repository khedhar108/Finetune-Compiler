# 🚀 VS Code + Colab T4 Quick Setup

Run the AI Compiler Wizard UI on Colab's Tesla T4 GPU.

---

## Quick Start

### 1. Connect to Colab T4
```
Ctrl + Shift + P → "Colab: Connect to Colab Runtime" → T4 GPU
```

### 2. Open Notebook
Open `notebooks/vscode_colab_launcher.ipynb` and run all cells.

**OR** run these commands in terminal:

```bash
# Clone from GitHub
git clone https://github.com/khedhar108/Finetune-Compiler.git
cd Finetune-Compiler

# Install
pip install uv -q
uv sync --extra ui --extra unsloth

# Launch UI
uv run ai-compile ui2 --share
```

### 3. Click the Link
Look for: `Running on public URL: https://xxxxx.gradio.live`

---

## What You Get

| Feature | Status |
|---------|--------|
| Tesla T4 GPU (16GB) | ✅ |
| Unsloth (2-5x faster) | ✅ |
| Wizard UI v2 | ✅ |

---

## Commands Reference

```bash
# Check GPU
nvidia-smi

# Run info
uv run ai-compile info

# Launch Classic UI
uv run ai-compile ui --share

# Launch Wizard UI
uv run ai-compile ui2 --share

# Train
uv run ai-compile train --config config.json
```
