# 🚀 VS Code + Colab T4 Quick Setup

Run the AI Compiler Wizard UI on Colab's Tesla T4 GPU from VS Code.

---

## Prerequisites

1. **VS Code** with Google Colab extension installed
2. **Google Account** for Colab access

---

## Step-by-Step Guide

### Step 1: Connect VS Code to Colab

```
Ctrl + Shift + P  →  "Colab: Connect to Colab Runtime"  →  Select "T4 GPU"
```

Wait for the status bar to show: `Colab: Connected`

---

### Step 2: Upload ai-compiler Folder

**Option A: Via VS Code Explorer**
1. In VS Code's Explorer, right-click on the Colab workspace
2. Upload the `ai-compiler` folder

**Option B: Via Terminal**
```bash
# In VS Code terminal (connected to Colab)
cd /content

# If you have it on Google Drive:
cp -r /content/drive/MyDrive/path/to/ai-compiler /content/

# OR upload zip and extract:
unzip ai-compiler.zip -d /content/
```

---

### Step 3: Run Setup

**Option A: Use Dedicated Notebook (Recommended)**
1. Open `research/notebooks/vscode_colab_launcher.ipynb` in VS Code
2. Run the cells one by one to Check GPU, Install, and Launch UI.

**Option B: Use Setup Script**
1. Open Terminal in VS Code
2. Run:
```bash
cd /content/ai-compiler
python scripts/colab_vscode_setup.py
```

OR run these commands individually:

```bash
# 1. Install UV
pip install uv -q

# 2. Install dependencies with UI and Unsloth
uv sync --extra ui --extra unsloth

# 3. Verify
uv run ai-compile info

# 4. Launch UI (with share link)
uv run ai-compile ui2 --share
```

---

### Step 4: Open the UI

After running the UI command, you'll see:
```
Running on public URL: https://xxxxx.gradio.live
```

**Click that link** to open the Wizard UI!

---

## What You Get

| Feature | Status |
|---------|--------|
| Tesla T4 GPU (16GB) | ✅ Available |
| Unsloth (2-5x faster) | ✅ Enabled |
| Wizard UI v2 | ✅ Running |
| Training | ✅ Ready |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No GPU" | Reconnect: `Ctrl+Shift+P` → Colab → T4 |
| UI won't start | Run `uv sync --extra ui` first |
| Can't find folder | Check you're in `/content/ai-compiler` |

---

## Quick Commands Reference

```bash
# Check GPU
nvidia-smi

# Check installation
uv run ai-compile info

# Run Classic UI
uv run ai-compile ui --share

# Run Wizard UI v2
uv run ai-compile ui2 --share

# Train directly
uv run ai-compile train --config config.json
```
