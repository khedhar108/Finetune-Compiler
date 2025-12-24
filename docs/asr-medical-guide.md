# 🎤 ASR Medical Guide

> Fine-tune Whisper for medical transcription and deploy to HuggingFace.

---

## Your Goal

```
Medical Audio → Fine-tuned Whisper → HuggingFace → medical-asr Backend
    │                   │                │                │
    │                   │                │                └─ Better transcription!
    │                   │                └─ Get model URL
    │                   └─ Train on medical terms
    └─ Record samples with medicine names
```

---

## Step 1: Prepare Medical Audio Data

### Create Dataset CSV

Create `data/medical_audio.csv`:

```csv
audio_path,transcription
./audio/sample1.wav,"Paracetamol 500 milligrams twice daily after meals"
./audio/sample2.wav,"Blood pressure reading 120 over 80 millimeters of mercury"
./audio/sample3.wav,"Prescribe Amoxicillin 250 milligrams for 7 days"
./audio/sample4.wav,"Patient has type 2 diabetes with HbA1c of 7.5 percent"
./audio/sample5.wav,"Administer Metformin 500 milligrams before breakfast"
```

### Record Audio Samples

- Record yourself saying medical terms
- Save as `.wav` files in `audio/` folder
- Keep each recording 3-10 seconds
- Minimum 50-100 samples for good results

### Convert to 16kHz WAV (Required)

```bash
uv run python scripts/prepare_audio_dataset.py \
    --input-csv data/medical_audio.csv \
    --output-dir data/prepared
```

---

## Step 2: Train Whisper

### Option A: Visual UI

```bash
uv run ui2
```

1. **Step 1 (Model)**: Select `openai/whisper-small`
2. **Step 2 (Data)**: 
   - Source: `csv`
   - Path: `./data/prepared/dataset.csv`
   - Format: `audio`
3. **Step 3 (Train)**: 
   - Epochs: 3
   - Click **Start Training**
4. **Step 4 (Deploy)**: Enter HuggingFace token and deploy

### Option B: CLI

```bash
# Create ASR config
uv run ai-compile init --task asr --output configs/medical_asr.json

# Edit configs/medical_asr.json:
{
  "model": {
    "name": "openai/whisper-small",
    "task_type": "asr"
  },
  "data": {
    "source": "csv",
    "path": "./data/prepared/dataset.csv",
    "format": "audio",
    "audio_column": "audio_path",
    "text_column": "transcription"
  },
  "training": {
    "epochs": 3,
    "batch_size": 4
  }
}

# Train
uv run ai-compile train --config configs/medical_asr.json

# Deploy
uv run ai-compile deploy --model ./output --repo YOUR-USERNAME/medical-whisper
```

---

## Step 3: Deploy to HuggingFace

```bash
uv run ai-compile deploy \
    --model ./output \
    --repo YOUR-USERNAME/medical-whisper-v1

# Output:
# ✅ Successfully deployed!
# 📎 Model URL: https://huggingface.co/YOUR-USERNAME/medical-whisper-v1
```

---

## Step 4: Use in medical-asr Backend

### Update medical-asr/testing-ui/app.py

```python
from transformers import pipeline

# Before (generic Whisper):
# transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# After (your fine-tuned model):
transcriber = pipeline(
    "automatic-speech-recognition", 
    model="YOUR-USERNAME/medical-whisper-v1"
)

def transcribe(audio_path):
    result = transcriber(audio_path)
    return result["text"]
```

### Or Load Manually

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# Load your fine-tuned model
processor = WhisperProcessor.from_pretrained("YOUR-USERNAME/medical-whisper-v1")
model = WhisperForConditionalGeneration.from_pretrained("YOUR-USERNAME/medical-whisper-v1")

# Transcribe
audio, sr = librosa.load("patient_audio.wav", sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
generated_ids = model.generate(inputs.input_features)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(transcription)
# Output: "Paracetamol 500 milligrams twice daily"
```

---

## Complete Workflow Summary

```bash
# 1. Prepare data
cd ai-compiler
# Record audio samples
# Create CSV with audio_path,transcription

# 2. Convert audio
uv run python scripts/prepare_audio_dataset.py \
    --input-csv data/medical_audio.csv \
    --output-dir data/prepared

# 3. Train
uv run ai-compile train --config configs/asr_whisper.json

# 4. Test locally
uv run ai-compile infer --model ./output --prompt "./audio/test.wav"

# 5. Deploy
uv run ai-compile deploy --model ./output --repo YOUR-USERNAME/medical-whisper

# 6. Use in medical-asr
cd ../medical-asr
# Update app.py with your model URL
python testing-ui/app.py
```

---

## Tips for Medical ASR

### Data Quality

| Tip | Why |
|-----|-----|
| Clear audio | Better transcription |
| Varied speakers | Generalization |
| Include medicine names | Domain adaptation |
| 100+ samples | Minimum for good results |

### Model Selection

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `whisper-small` | 244M | Good | Fast |
| `whisper-medium` | 769M | Better | Medium |
| `whisper-large-v3` | 1.55B | Best | Slow |

**Recommendation:** Start with `whisper-small`, upgrade if needed.

### Common Medical Terms to Include

```
Medications: Paracetamol, Amoxicillin, Metformin, Insulin
Dosages: milligrams, milliliters, twice daily, after meals
Vitals: blood pressure, heart rate, temperature, oxygen saturation
Conditions: diabetes, hypertension, asthma, infection
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Use `whisper-small` + batch_size=2 |
| Poor transcription | Add more training samples |
| Slow inference | Use smaller model |
| Audio not loading | Ensure 16kHz WAV format |

---

## Need Help?

- [Full Documentation](getting-started.md)
- [Quick Start Demo](quick-start.md)
- [Medical ASR Testing UI](../../medical-asr/testing-ui/README.md)
