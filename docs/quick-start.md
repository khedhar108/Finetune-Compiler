# ⚡ Quick Start Demo

> Train and deploy a model in under 5 minutes!

---

## Prerequisites

```bash
cd ai-compiler
uv sync --extra ui
```

---

## Option A: Visual UI (Easiest)

```bash
uv run ui2
```

1. Open `http://localhost:7862`
2. **Step 1**: Select `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (smallest, fastest)
3. **Step 2**: Keep default dataset `tatsu-lab/alpaca`
4. **Step 3**: Set Epochs = 1, click **Start Training**
5. **Step 4**: Deploy to HuggingFace!

---

## Option B: One Command (CLI)

```bash
# Train with default config (TinyLlama + Alpaca)
uv run train

# Test your model
uv run ai-compile infer --model ./output --prompt "What is AI?"

# Deploy to HuggingFace
uv run ai-compile deploy --model ./output --repo YOUR-USERNAME/my-first-model
```

---

## What Happens?

```
Step 1: Download TinyLlama (1.1B params) ← ~2 GB, cached for future
Step 2: Download Alpaca dataset ← 52K examples
Step 3: Train for 1 epoch ← ~5-10 minutes
Step 4: Save to ./output/ ← LoRA adapter (~30 MB)
Step 5: Deploy to HuggingFace ← Get URL!
```

---

## Use Your Model

After deploying, you get a URL like:
```
https://huggingface.co/YOUR-USERNAME/my-first-model
```

Use it in Python:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("YOUR-USERNAME/my-first-model")
tokenizer = AutoTokenizer.from_pretrained("YOUR-USERNAME/my-first-model")

# Generate
inputs = tokenizer("What is machine learning?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Use `TinyLlama` + batch_size=1 |
| No GPU | Use Google Colab |
| Slow download | Normal for first time |

---

## Next Steps

- [Full Guide](getting-started.md) - Detailed explanations
- [ASR Medical Guide](asr-medical-guide.md) - For speech-to-text
- [Deploy Guide](getting-started.md#deploying-to-huggingface-hub) - HuggingFace deployment
