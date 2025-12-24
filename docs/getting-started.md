# 🚀 Getting Started Guide

> A complete beginner's guide to using the AI Compiler.

---

## What Does This Compiler Do?

Think of it like this:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Pre-trained     │     │  Your Data      │     │ Your Custom     │
│ AI Model        │  +  │  (examples)     │  =  │ AI Model        │
│ (knows general) │     │  (your domain)  │     │ (knows your     │
│                 │     │                 │     │  specific task) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Example:** A general AI model + Medical transcription examples = Medical AI that understands medicine names perfectly.

---

## Prerequisites (One-Time Setup)

### Step 1: Install UV (Package Manager)

**Windows** - Open PowerShell and run:
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Mac/Linux** - Open Terminal and run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Open Terminal in Project Folder

1. Open the `ai-compiler` folder in VS Code
2. Press `Ctrl + `` (backtick) to open terminal
3. You should see: `PS D:\...\ai-compiler>`

### Step 3: Install Dependencies

```bash
# Basic install (CLI only)
uv sync

# Full install (with visual UI)
uv sync --extra ui
```

Wait for it to finish (takes 2-5 minutes first time).

---

## Test That Everything Works

Run this command:

```bash
uv run ai-compile info
```

You should see something like:
```
╔═══════════════════════════════════════════════════════════════╗
║                    🚀 AI COMPILER CORE                        ║
╚═══════════════════════════════════════════════════════════════╝

Version: 0.1.0
PyTorch: 2.9.1
Transformers: 4.57.3
PEFT: 0.18.0
```

✅ If you see this, you're ready!

---

## Quick Test with Visual UI

### Option 1: Classic UI (Simple)
```bash
uv run ai-compile ui
```
Opens at `http://localhost:7860` - tabs for Config, Training, Test.

### Option 2: Wizard UI v2 (Recommended)
```bash
uv run ai-compile ui2
```
Opens at `http://localhost:7862` with step-by-step flow:

```
Step 1: Model  →  Step 2: Data  →  Step 3: Train  →  Step 4: Deploy
```

Features:
- 🎯 Progressive steps (can't skip ahead)
- 📊 Collapsible logs panel
- 🔐 HuggingFace token input
- 📤 One-click deploy to Hub

---

## Your First Training (Step-by-Step)

### Option A: Visual UI (Easiest)

1. Run: `uv run ai-compile ui`
2. Open `http://localhost:7860` in browser
3. Set these values:
   - Model: `unsloth/Llama-3.2-1B`
   - Quantization: `4bit`
   - Epochs: `1`
   - Dataset Path: `./data/train.csv`
4. Click "Start Training"
5. Watch the logs!

### Option B: Command Line

#### Step 1: Create Sample Data

Create a file `data/train.csv`:

```csv
instruction,input,output
"What is paracetamol?","","Paracetamol is a common pain reliever and fever reducer."
"What is the dosage for amoxicillin?","","The typical adult dosage is 500mg every 8 hours."
"Define hypertension.","","Hypertension is high blood pressure, typically above 140/90 mmHg."
```

#### Step 2: Run Training

```bash
uv run ai-compile train --config configs/default.json
```

#### Step 3: Wait and Watch

You'll see output like:
```
[Step 10] loss: 2.34 | lr: 2e-4 | GPU: 4.2GB
[Step 20] loss: 1.89 | lr: 2e-4 | GPU: 4.1GB
...
✅ Training completed!
```

**What do these mean?**
- `loss`: How wrong the model is (lower = better)
- `lr`: Learning rate (how fast it learns)
- `GPU`: Memory usage

---

## HuggingFace Dataset Examples

Here are ready-to-use datasets you can try:

### Example 1: Alpaca Dataset (General Q&A)

```json
{
  "data": {
    "source": "huggingface",
    "path": "tatsu-lab/alpaca",
    "format": "alpaca"
  },
  "model": {
    "name": "unsloth/Llama-3.2-1B",
    "quantization": "4bit"
  }
}
```

### Example 2: Medical Q&A

```json
{
  "data": {
    "source": "huggingface",
    "path": "medalpaca/medical_meadow_medical_flashcards",
    "format": "alpaca"
  },
  "model": {
    "name": "unsloth/Llama-3.2-1B"
  }
}
```

### Example 3: Code Generation

```json
{
  "data": {
    "source": "huggingface",
    "path": "sahil2801/CodeAlpaca-20k",
    "format": "alpaca"
  },
  "model": {
    "name": "unsloth/Llama-3.2-1B"
  }
}
```

### Example 4: Customer Support

```json
{
  "data": {
    "source": "huggingface",
    "path": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    "format": "custom",
    "instruction_column": "instruction",
    "output_column": "response"
  }
}
```

---

## Understanding Evaluation Metrics

### What We Measure:

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **Loss** | How wrong the model is | Lower is better (< 1.0) |
| **Perplexity** | How "surprised" the model is | Lower is better (< 10) |
| **Accuracy** | % of correct answers | Higher is better (> 80%) |

### How to Read Training Output:

```
[Step 100] loss: 0.85 ← Getting better!
[Step 200] loss: 0.42 ← Even better!
[Step 300] loss: 0.38 ← Almost done!
```

If loss stops decreasing, training is complete.

---

## Is This Compiler Universal?

**Yes!** You can use it for:

| Use Case | Model Type | Config |
|----------|------------|--------|
| Text Q&A | LLM (Llama, Mistral) | `task: causal_lm` |
| Speech-to-Text | Whisper | `task: asr` |
| Translation | mT5, NLLB | `task: seq2seq` |
| Summarization | BART, T5 | `task: seq2seq` |

---

## ASR (Speech-to-Text) Usage

### Step 1: Prepare Audio Data

Create a CSV file `data/audio_dataset.csv`:

```csv
audio_path,transcription
./audio/sample1.wav,"Paracetamol 500mg twice daily"
./audio/sample2.wav,"Blood pressure 120 over 80"
./audio/sample3.wav,"Amoxicillin 250mg for 7 days"
```

### Step 2: Convert Audio to 16kHz WAV

```bash
uv run python scripts/prepare_audio_dataset.py \
    --input-csv data/audio_dataset.csv \
    --output-dir data/prepared
```

### Step 3: Train Whisper Model

```bash
uv run ai-compile train --config configs/asr_whisper.json
```

### Step 4: Test Your Model

```bash
cd ../medical-asr/testing-ui
python app.py
```

Open `http://localhost:7861` and record audio to test!

---

## After Training: What Next?

### Your trained model is in: `./output/`

```
output/
├── adapter_model/          ← Your trained weights
├── tokenizer_config.json   ← For loading
└── README.md               ← Model info
```

---

## Deploying to HuggingFace Hub

### Step 1: Get HuggingFace Tokens

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create two tokens:

| Token Type | Permission | Use For |
|------------|------------|---------|
| **Read** | Read | Downloading private datasets |
| **Write** | Write | Uploading your model |

### Step 2: Deploy via CLI

```bash
# Basic deploy
uv run ai-compile deploy --model ./output --repo your-username/my-model

# Deploy as private
uv run ai-compile deploy --model ./output --repo your-username/my-model --private

# With token (if not set in environment)
uv run ai-compile deploy --model ./output --repo your-username/my-model --token hf_xxxxx
```

### Step 3: Deploy via Wizard UI

1. Run: `uv run ui2`
2. Go to Step 4: Deploy
3. Enter your Write Token
4. Enter repository name: `your-username/model-name`
5. Click "🚀 Deploy to HuggingFace Hub"
6. Copy the returned URL!

### Step 4: Use Your Model

Your model is now at:
```
https://huggingface.co/your-username/my-model
```

Load it in your app:
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-username/my-model")
---

## Test Your Trained Model

After training, test your model immediately:

```bash
# Single question
uv run ai-compile infer --model ./output --prompt "What is paracetamol?"

# Interactive chat mode
uv run ai-compile infer --model ./output --interactive
# Type questions, get responses, type 'exit' to quit
```

---

## Disk Space & Storage

### How Much Space Do I Need?

| Component | Size | Notes |
|-----------|------|-------|
| Base model (downloaded) | 2-8 GB | Cached, downloaded once |
| **LoRA adapter (your output)** | 10-50 MB | Very small! |
| Checkpoints | 10-50 MB each | Auto-cleaned |
| Merged model (optional) | 2-8 GB | Only if you merge |

**LoRA is efficient!** It only saves the *changes*, not the whole model.

### Where Are Models Cached?

| OS | Cache Path |
|----|------------|
| **Windows** | `C:\Users\{username}\.cache\huggingface\hub\` |
| **Linux/Mac** | `~/.cache/huggingface/hub/` |
| **Colab** | `/root/.cache/huggingface/hub/` |

**To clear cache (free up space):**
```bash
# Show cache size
du -sh ~/.cache/huggingface/hub/

# Clear specific model (careful!)
rm -rf ~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B
```

### Change Cache Location (Use Different Drive)

If your C: drive is full, change the cache location:

**Windows (PowerShell):**
```powershell
# Set for current session
$env:HF_HOME = "D:\models\cache"

# Set permanently (add to PowerShell profile)
[System.Environment]::SetEnvironmentVariable("HF_HOME", "D:\models\cache", "User")

# Then restart terminal and run
uv run train
```

**Linux/Mac:**
```bash
export HF_HOME="/mnt/external/models"
```

**In Python code:**
```python
import os
os.environ["HF_HOME"] = "D:\\models\\cache"

# Then import transformers
from transformers import AutoModelForCausalLM
```

### Where Is Data Stored?

| Environment | Location | Notes |
|-------------|----------|-------|
| **Windows (local)** | Your disk | Need ~10 GB free |
| **Colab (browser)** | Google's cloud | Uses their storage |
| **Colab Extension (VS Code)** | Google's cloud | Uses their storage |

---

## Running on Different Environments

### Option 1: Local Windows (Your PC)

```bash
# If you have NVIDIA GPU:
uv sync
uv run train
```

### Option 2: Google Colab (Browser)

1. Open `research/notebooks/colab_quickstart.ipynb`
2. Upload to Colab
3. Connect to T4 GPU runtime
4. Run cells

### Option 3: VS Code + Colab Extension

1. Install "Google Colab" VS Code extension
2. Connect to Colab runtime (T4 GPU)
3. Training runs on Google's GPU
4. Your local disk is NOT used for heavy files

### Unsloth (Linux/Colab Only)

```bash
# On Colab/Linux - installs Unsloth for 2-5x faster training
uv sync --extra unsloth

# On Windows - automatically falls back to PEFT (works fine)
uv sync
```

---

## Summary: Quick Reference

```bash
# First time setup
uv sync --extra ui

# Test installation
uv run ai-compile info

# Launch visual UI
uv run ai-compile ui2

# Train
uv run train

# Test your model
uv run ai-compile infer --model ./output --interactive

# Deploy
uv run ai-compile deploy --model ./output --repo username/my-model
```

---

**Need more help?** Check the [Documentation](core-engine.md)
