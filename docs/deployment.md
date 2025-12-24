# 🚀 Deployment Guide

> From fine-tuned model to production inference - step by step.

---

## Deployment Options Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    FINE-TUNED MODEL                             │
│              (LoRA Adapter or Merged Weights)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  LOCAL/EDGE   │    │    CLOUD      │    │  HF SPACES    │
│               │    │               │    │               │
│ • Ollama      │    │ • RunPod      │    │ • Gradio      │
│ • llama.cpp   │    │ • Modal       │    │ • Streamlit   │
│ • vLLM        │    │ • Replicate   │    │ • Docker      │
│ • Text Gen UI │    │ • AWS/GCP     │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR BACKEND SERVICE                         │
│     FastAPI / Flask / Node.js / Any HTTP client                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Export Your Model

After training, export in the desired format:

```bash
# Export as merged model (adapter + base = single model)
ai-compile export --model ./output --format merged --output ./models/my-model

# Export as GGUF (for Ollama/llama.cpp)
ai-compile export --model ./output --format gguf --quantization q4_k_m --output ./models/my-model.gguf

# Export adapter only (smallest, needs base model at inference)
ai-compile export --model ./output --format adapter --output ./models/my-adapter
```

**Export Formats:**

| Format | Size | Use Case | Inference Speed |
|--------|------|----------|-----------------|
| `adapter` | ~50MB | Development, quick testing | Fast (with base cached) |
| `merged` | 2-14GB | Full deployment | Fast |
| `gguf` | 1-8GB | CPU/Edge deployment | Very Fast |

---

## Step 2: Choose Deployment Target

### Option A: Local Deployment (Free)

#### Using Ollama (Recommended for beginners)

```bash
# 1. Install Ollama (https://ollama.ai)
# 2. Create Modelfile
cat > Modelfile << EOF
FROM ./my-model.gguf
SYSTEM "You are a helpful medical assistant."
PARAMETER temperature 0.7
EOF

# 3. Create model in Ollama
ollama create my-medical-model -f Modelfile

# 4. Run inference
ollama run my-medical-model "What are symptoms of diabetes?"

# 5. Use via API (runs on localhost:11434)
curl http://localhost:11434/api/generate -d '{
  "model": "my-medical-model",
  "prompt": "What are symptoms of diabetes?"
}'
```

#### Using vLLM (High throughput)

```bash
# Install vLLM
pip install vllm

# Run OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
    --model ./models/my-model \
    --port 8000

# Use via OpenAI SDK
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "my-model", "prompt": "Hello", "max_tokens": 100}'
```

#### Using Text Generation WebUI

```bash
# Clone and setup
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
./start_linux.sh  # or start_windows.bat

# Place model in models/ folder, access via browser
```

---

### Option B: HuggingFace Spaces (Free tier available)

#### Step 2B.1: Push to Hub

```bash
# Login to HuggingFace
huggingface-cli login

# Push adapter (or full model)
ai-compile push --model ./output --repo username/my-medical-model
```

#### Step 2B.2: Create Gradio Space

Create `app.py`:

```python
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
base_model = "unsloth/Llama-3.2-1B"
adapter = "username/my-medical-model"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)

def generate(prompt, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

demo = gr.Interface(
    fn=generate,
    inputs=["text", gr.Slider(50, 500, value=256, label="Max Tokens")],
    outputs="text",
    title="Medical Assistant"
)

demo.launch()
```

#### Step 2B.3: Deploy to Space

```bash
# Create Space on HuggingFace Hub
# Upload app.py and requirements.txt
# Select "Gradio" SDK
# Done! Access at https://huggingface.co/spaces/username/my-app
```

---

### Option C: Cloud Deployment (Paid)

#### RunPod Serverless

```python
# handler.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def handler(event):
    prompt = event["input"]["prompt"]
    # Load and run inference
    # Return result
    return {"output": generated_text}
```

#### Modal

```python
# app.py
import modal

stub = modal.Stub("medical-llm")

@stub.function(gpu="T4", image=modal.Image.debian_slim().pip_install("transformers", "peft", "torch"))
def generate(prompt: str) -> str:
    # Load model and generate
    return result

@stub.local_entrypoint()
def main():
    result = generate.remote("What is diabetes?")
    print(result)
```

---

## Step 3: Integrate with Backend

### FastAPI Example

```python
# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import httpx

app = FastAPI()

# Point to your deployed model
OLLAMA_URL = "http://localhost:11434/api/generate"
# or HF Inference API
# HF_URL = "https://api-inference.huggingface.co/models/username/model"

class Query(BaseModel):
    prompt: str
    max_tokens: int = 256

class Response(BaseModel):
    response: str

@app.post("/ask", response_model=Response)
async def ask_medical_question(query: Query):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            OLLAMA_URL,
            json={"model": "my-medical-model", "prompt": query.prompt}
        )
        return Response(response=response.json()["response"])

# Run with: uvicorn main:app --reload
```

### Node.js/Express Example

```javascript
// backend/server.js
const express = require('express');
const axios = require('axios');

const app = express();
app.use(express.json());

const OLLAMA_URL = 'http://localhost:11434/api/generate';

app.post('/ask', async (req, res) => {
    const { prompt } = req.body;
    
    const response = await axios.post(OLLAMA_URL, {
        model: 'my-medical-model',
        prompt: prompt
    });
    
    res.json({ response: response.data.response });
});

app.listen(3000, () => console.log('Server running on port 3000'));
```

---

## Step 4: Frontend Integration

```javascript
// frontend/api.js
async function askQuestion(prompt) {
    const response = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
    });
    return response.json();
}

// React component
function MedicalChat() {
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');
    
    const handleSubmit = async () => {
        const result = await askQuestion(question);
        setAnswer(result.response);
    };
    
    return (
        <div>
            <input value={question} onChange={e => setQuestion(e.target.value)} />
            <button onClick={handleSubmit}>Ask</button>
            <p>{answer}</p>
        </div>
    );
}
```

---

## Deployment Comparison

| Method | Cost | Setup Time | Scalability | Best For |
|--------|------|------------|-------------|----------|
| Ollama (local) | Free | 10 min | Single user | Development, demos |
| vLLM (local) | Free | 20 min | Multi-user | Local production |
| HF Spaces | Free tier | 30 min | Limited | Public demos |
| RunPod | $0.20/hr | 1 hr | Auto-scale | Production |
| Modal | Pay-per-use | 1 hr | Auto-scale | Production |
| AWS/GCP | Varies | 2+ hrs | Custom | Enterprise |

---

## Security Considerations

> [!CAUTION]
> **For production deployments:**
> - Add authentication (API keys, JWT)
> - Rate limit requests
> - Sanitize inputs (prompt injection prevention)
> - Log requests for audit
> - Use HTTPS

---

## Related Documents

- [Core Engine](./core-engine.md)
- [Business Use Cases](./business-use-cases.md)
- [UI Guide](./ui-guide.md)
