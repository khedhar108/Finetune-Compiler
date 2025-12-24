# 📚 Resources & References

> Official documentation and tools used in the AI Compiler Platform.

---

## Core Engine Libraries

| Library | Purpose | Documentation |
|---------|---------|---------------|
| **Transformers** | Model loading & training | [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers/index) |
| **Datasets** | Data loading & processing | [huggingface.co/docs/datasets](https://huggingface.co/docs/datasets/index) |
| **PEFT** | LoRA/QLoRA adapters | [github.com/huggingface/peft](https://github.com/huggingface/peft) |
| **Unsloth** | Optimized training | [docs.unsloth.ai](https://docs.unsloth.ai/) |
| **bitsandbytes** | 4-bit/8-bit quantization | [github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) |
| **Accelerate** | Multi-GPU training | [huggingface.co/docs/accelerate](https://huggingface.co/docs/accelerate) |
| **Typer** | CLI framework | [typer.tiangolo.com](https://typer.tiangolo.com/) |
| **Evaluate** | Metrics library | [huggingface.co/docs/evaluate](https://huggingface.co/docs/evaluate) |
| **UV** | Fast Python package manager | [docs.astral.sh/uv](https://docs.astral.sh/uv/) |

---

## Audio Processing Libraries

| Library | Purpose | Documentation |
|---------|---------|---------------|
| **Librosa** | Audio analysis, resampling | [librosa.org](https://librosa.org/) |
| **torchaudio** | PyTorch audio processing | [pytorch.org/audio](https://pytorch.org/audio/) |
| **soundfile** | Read/write audio files | [pysoundfile.readthedocs.io](https://pysoundfile.readthedocs.io/) |
| **noisereduce** | Noise reduction | [github.com/timsainb/noisereduce](https://github.com/timsainb/noisereduce) |
| **Silero VAD** | Voice activity detection | [github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad) |

---

## Audio Segmentation

| Tool | Purpose | Documentation |
|------|---------|---------------|
| **SAM Audio** | Audio segmentation (Meta) | [github.com/facebookresearch/sam-audio](https://github.com/facebookresearch/sam-audio) |
| **Blog Post** | SAM Audio announcement | [ai.meta.com/blog/sam-audio](https://ai.meta.com/blog/sam-audio/) |

---

## ASR / Speech Models

| Model | Purpose | Documentation |
|-------|---------|---------------|
| **Whisper** | Speech-to-text | [github.com/openai/whisper](https://github.com/openai/whisper) |
| **Wav2Vec2** | Speech recognition | [huggingface.co/facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) |
| **Whisper Fine-tuning** | Custom ASR | [huggingface.co/blog/fine-tune-whisper](https://huggingface.co/blog/fine-tune-whisper) |

---

## AI Frameworks (Backend)

| Framework | Purpose | Documentation |
|-----------|---------|---------------|
| **LangChain** | LLM application framework | [langchain.com](https://www.langchain.com/) |
| **LangGraph** | Multi-agent workflows | [github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) |
| **LangFuse** | LLM observability | [langfuse.com](https://www.langfuse.com/) |

---

## Logging & Experiment Tracking

| Tool | Purpose | Documentation |
|------|---------|---------------|
| **MLflow** | Experiment tracking | [mlflow.org](https://mlflow.org/) |
| **Weights & Biases** | ML experiment tracking | [wandb.ai](https://wandb.ai/) |
| **Gdown** | Google Drive downloads | [github.com/wkentaro/gdown](https://github.com/wkentaro/gdown) |

---

## Deployment

| Tool | Purpose | Documentation |
|------|---------|---------------|
| **Ollama** | Local LLM deployment | [ollama.ai](https://ollama.ai/) |
| **vLLM** | High-throughput inference | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |
| **llama.cpp** | CPU/edge deployment | [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) |
| **HuggingFace Hub** | Model hosting | [huggingface.co/docs/hub](https://huggingface.co/docs/hub) |

---

## UI Frameworks

| Framework | Purpose | Documentation |
|-----------|---------|---------------|
| **Gradio** | ML web UI (development) | [gradio.app](https://www.gradio.app/) |
| **Next.js** | React framework (production) | [nextjs.org](https://nextjs.org/) |
| **FastAPI** | Python API framework | [fastapi.tiangolo.com](https://fastapi.tiangolo.com/) |

---

## Related Documents

- [Core Engine](./core-engine.md)
- [Audio Preprocessing](./audio-preprocessing.md)
- [Deployment Guide](./deployment.md)
