"""
AI Compiler Wizard UI v2.

A step-by-step wizard interface for training and deploying models.
Features:
- Progressive steps (Model → Data → Train → Deploy)
- Collapsible logs panel
- HuggingFace integration
- One-click deploy
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Generator

try:
    import gradio as gr
except ImportError:
    raise ImportError(
        "Gradio is required for UI. Install with: uv sync --extra ui"
    )

from engine.utils.config import get_default_config
from engine.utils.memory import get_gpu_memory


# ============ Constants ============

# Models organized by category with recommendations
MODELS_BY_CATEGORY = {
    "🚀 Quick Start (Recommended)": [
        ("unsloth/Llama-3.2-1B", "Best for beginners, fast training"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Very small, quick experiments"),
    ],
    "💬 Text Generation / Chat": [
        ("unsloth/Llama-3.2-3B", "Good balance of speed and quality"),
        ("microsoft/phi-2", "Strong reasoning, 2.7B params"),
        ("mistralai/Mistral-7B-v0.1", "High quality, needs more GPU"),
        ("THUDM/glm-4-9b-chat", "GLM4, multilingual"),
        ("Qwen/Qwen2-1.5B", "Qwen2, good for chat"),
    ],
    "📝 Text Classification": [
        ("microsoft/deberta-v3-base", "Best for classification"),
        ("roberta-base", "Good all-rounder"),
        ("distilbert-base-uncased", "Fast, lightweight"),
    ],
    "🎤 Speech-to-Text (ASR)": [
        ("openai/whisper-small", "Fast, good accuracy"),
        ("openai/whisper-medium", "Better accuracy, more GPU"),
        ("openai/whisper-large-v3", "Best accuracy, needs lots of GPU"),
    ],
    "🌐 Translation / Multilingual": [
        ("google/mt5-small", "Multilingual T5"),
        ("facebook/nllb-200-distilled-600M", "200 languages"),
    ],
}

# Flatten for dropdown
MODELS = []
for category, models in MODELS_BY_CATEGORY.items():
    for model_name, desc in models:
        MODELS.append(f"{model_name}")

# Model recommendations lookup
MODEL_INFO = {
    model: desc 
    for category, models in MODELS_BY_CATEGORY.items() 
    for model, desc in models
}

DATA_SOURCES = ["huggingface", "csv", "json", "gdrive"]
FORMATS = ["alpaca", "chatml", "completion", "audio"]
QUANTIZATION = ["4bit", "8bit", "none"]

# Datasets with descriptions
HUGGINGFACE_DATASETS = {
    "📚 General Q&A": [
        ("tatsu-lab/alpaca", "52K instruction-following examples"),
        ("databricks/dolly-15k", "15K high-quality examples"),
    ],
    "🏥 Medical": [
        ("medalpaca/medical_meadow_medical_flashcards", "Medical flashcards"),
        ("medmcqa", "Medical MCQ dataset"),
    ],
    "💻 Code": [
        ("sahil2801/CodeAlpaca-20k", "20K code instructions"),
        ("iamtarun/python_code_instructions_18k_alpaca", "Python code"),
    ],
    "📞 Customer Support": [
        ("bitext/Bitext-customer-support-llm-chatbot-training-dataset", "Customer support"),
    ],
}

# Flatten datasets for dropdown
DATASET_CHOICES = []
for category, datasets in HUGGINGFACE_DATASETS.items():
    for ds_name, desc in datasets:
        DATASET_CHOICES.append(f"{ds_name}")


# ============ Training Manager ============

class TrainingManager:
    """Manages training subprocess and logs."""
    
    def __init__(self):
        self.process = None
        self.logs = []
        self.is_running = False
        self.progress = 0
    
    def start(self, config_path: str):
        self.logs = ["🚀 Starting training...\n"]
        self.is_running = True
        self.progress = 0
        
        self.process = subprocess.Popen(
            ["uv", "run", "ai-compile", "train", "--config", config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        def read_output():
            for line in self.process.stdout:
                self.logs.append(line)
                # Parse progress from logs
                if "Step" in line:
                    try:
                        step = int(line.split("Step")[1].split("]")[0].strip())
                        self.progress = min(step, 100)
                    except:
                        pass
            self.is_running = False
            self.logs.append("\n✅ Training complete!")
        
        thread = threading.Thread(target=read_output, daemon=True)
        thread.start()
    
    def stop(self):
        if self.process:
            self.process.terminate()
            self.is_running = False
            self.logs.append("\n⏹ Training stopped by user.")
    
    def get_logs(self) -> str:
        return "".join(self.logs[-50:])
    
    def get_progress(self) -> int:
        return self.progress


training_manager = TrainingManager()


# ============ Helper Functions ============

def check_hf_token() -> str:
    """Check if HuggingFace token is set."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return f"✅ Token found: {token[:8]}..."
    return "⚠️ No token set. Set HF_TOKEN environment variable for private models."


def save_hf_token(token: str) -> str:
    """Save HuggingFace token to environment."""
    if token and token.startswith("hf_"):
        os.environ["HF_TOKEN"] = token
        return f"✅ Token saved: {token[:8]}..."
    return "❌ Invalid token. Must start with 'hf_'"


def get_gpu_status() -> str:
    """Get GPU status with clear recommendation."""
    try:
        stats = get_gpu_memory()
        if stats.get("available", False):
            gpu_name = stats['device_name']
            total = stats['total_gb']
            used = stats.get('reserved_gb', 0)
            free = stats.get('free_gb', total)
            
            # Determine if it's Colab T4 or local GPU
            is_t4 = "T4" in gpu_name
            is_colab = "T4" in gpu_name and total > 14  # T4 has ~15GB
            
            status = f"🎮 **{gpu_name}**\n"
            status += f"📊 Memory: {used:.1f}GB / {total:.1f}GB ({free:.1f}GB free)\n"
            
            if is_colab:
                status += "☁️ Colab T4 - Great for training!\n"
                status += "✅ Unsloth available (2-5x faster)"
            elif "RTX" in gpu_name or "GeForce" in gpu_name:
                status += f"🖥️ Local GPU - Ready to train!\n"
                status += "⚠️ Unsloth: Windows not supported, using PEFT (works fine)"
            else:
                status += "✅ Ready to train"
            
            return status
        return "❌ No GPU available - Use Colab for training"
    except:
        return "⚠️ Unable to check GPU - Check if PyTorch is installed"


def build_config(
    model_name, quantization, max_seq_length,
    lora_r, lora_alpha,
    data_source, data_path, data_format,
    epochs, batch_size, learning_rate,
    output_dir,
) -> dict:
    """Build configuration from wizard inputs."""
    return {
        "project": {
            "name": "wizard-training",
            "output_dir": output_dir,
            "seed": 42,
        },
        "data": {
            "source": data_source,
            "path": data_path,
            "format": data_format,
            "test_split": 0.1,
        },
        "model": {
            "name": model_name,
            "task": "asr" if "whisper" in model_name.lower() else "causal_lm",
            "quantization": quantization,
            "max_seq_length": max_seq_length,
        },
        "lora": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": 4,
            "learning_rate": learning_rate,
            "fp16": True,
            "gradient_checkpointing": True,
        },
        "logging": {
            "log_steps": 5,
            "save_steps": 100,
            "save_total_limit": 3,
        },
    }


# ============ Wizard Steps ============

def step1_model():
    """Step 1: Model Selection."""
    with gr.Column():
        gr.Markdown("## Step 1: Select Model")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("Choose a pre-trained model to fine-tune.")
                
                model_name = gr.Dropdown(
                    choices=MODELS,
                    value=MODELS[0],
                    label="Base Model",
                    info="Smaller models train faster, larger models are smarter",
                )
                
                model_recommendation = gr.Textbox(
                    value=MODEL_INFO.get(MODELS[0], ""),
                    label="💡 Recommendation",
                    interactive=False,
                )
                
                quantization = gr.Radio(
                    choices=QUANTIZATION,
                    value="4bit",
                    label="Quantization",
                    info="4-bit uses least memory (recommended)",
                )
                
                max_seq_length = gr.Slider(
                    minimum=256,
                    maximum=4096,
                    value=2048,
                    step=256,
                    label="Max Sequence Length",
                    info="Longer = more context, more memory",
                )
                
                with gr.Row():
                    lora_r = gr.Slider(4, 64, value=16, step=4, label="LoRA Rank")
                    lora_alpha = gr.Slider(8, 128, value=32, step=8, label="LoRA Alpha")
                
                gpu_status = gr.Textbox(value=get_gpu_status(), label="GPU Status", interactive=False)
            
            # Help panel on the right
            with gr.Column(scale=1):
                with gr.Accordion("📖 Quick Guide", open=True):
                    gr.Markdown("""
**Which model to choose?**

| Use Case | Recommended |
|----------|-------------|
| 🚀 First time | `Llama-3.2-1B` |
| 💬 Chat | `Mistral-7B` or `phi-2` |
| 📝 Classification | `deberta-v3-base` |
| 🎤 ASR | `whisper-small` |

**Quantization:**
- `4bit` = Least memory (start here!)
- `8bit` = More accuracy
- `none` = Full precision (needs lots of RAM)

**LoRA Settings:**
- Default values work for most cases
- Higher rank = more capacity but slower
                    """)
        
        def update_recommendation(model):
            return MODEL_INFO.get(model, "No info available")
        
        model_name.change(update_recommendation, [model_name], [model_recommendation])
        
    return model_name, quantization, max_seq_length, lora_r, lora_alpha, gpu_status


def step2_data():
    """Step 2: Data Configuration."""
    with gr.Column():
        gr.Markdown("## Step 2: Configure Data")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("Select your training data source.")
                
                data_source = gr.Radio(
                    choices=DATA_SOURCES,
                    value="huggingface",
                    label="Data Source",
                )
                
                with gr.Row():
                    data_path = gr.Textbox(
                        value="tatsu-lab/alpaca",
                        label="Dataset Path / HuggingFace ID",
                        placeholder="Enter path or HuggingFace dataset name",
                    )
                    
                    hf_examples = gr.Dropdown(
                        choices=DATASET_CHOICES,
                        label="Quick Select (HuggingFace)",
                        info="Pre-filled examples",
                    )
                
                data_format = gr.Radio(
                    choices=FORMATS,
                    value="alpaca",
                    label="Data Format",
                    info="alpaca = instruction/output, chatml = chat format",
                )
                
                gr.Markdown("### 🔑 HuggingFace Token")
                gr.Markdown("*For private datasets or gated models (like Llama)*")
                
                with gr.Row():
                    hf_token_input = gr.Textbox(
                        label="Token",
                        placeholder="hf_xxxxxxxxxxxxxxxx",
                        type="password",
                    )
                    hf_token_btn = gr.Button("Save", size="sm")
                    hf_token_status = gr.Textbox(
                        value=check_hf_token(),
                        label="Status",
                        interactive=False,
                    )
            
            # Help panel on the right
            with gr.Column(scale=1):
                with gr.Accordion("📖 Quick Guide", open=True):
                    gr.Markdown("""
**Getting Data:**

1. **HuggingFace** (easiest)
   - Just enter dataset name
   - Example: `tatsu-lab/alpaca`

2. **CSV file**
   - Create `data/train.csv`
   - Columns: `instruction`, `output`

**Need a token?**
1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a token
3. Paste above and click Save

**Format Guide:**
- `alpaca` = instruction/output pairs
- `chatml` = chat messages
- `audio` = for ASR models
                    """)
                
                with gr.Accordion("🔗 Quick Links", open=False):
                    gr.Markdown("""
- [Get HuggingFace Token](https://huggingface.co/settings/tokens)
- [Browse Datasets](https://huggingface.co/datasets)
- [Alpaca Format Guide](https://github.com/tatsu-lab/stanford_alpaca)
                    """)
        
        def update_path(example):
            return example if example else "tatsu-lab/alpaca"
        
        hf_examples.change(update_path, [hf_examples], [data_path])
        hf_token_btn.click(save_hf_token, [hf_token_input], [hf_token_status])
        
    return data_source, data_path, data_format


def step3_training():
    """Step 3: Training Configuration & Execution."""
    with gr.Column():
        gr.Markdown("## Step 3: Train Model")
        
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                    batch_size = gr.Slider(1, 16, value=4, step=1, label="Batch Size")
                    learning_rate = gr.Number(value=2e-4, label="Learning Rate")
                
                output_dir = gr.Textbox(value="./output", label="Output Directory")
                
                # Debug options
                with gr.Row():
                    debug_mode = gr.Checkbox(
                        label="🐛 Debug Mode",
                        value=False,
                        info="Show detailed logs, stack traces, and save log file",
                    )
                    save_logs = gr.Checkbox(
                        label="💾 Save Logs",
                        value=True,
                        info="Save training logs to output/training.log",
                    )
                
                with gr.Row():
                    start_btn = gr.Button("▶ Start Training", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹ Stop", variant="stop", size="lg")
                
                progress_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Progress",
                    interactive=False,
                )
                
                with gr.Accordion("📜 Training Logs", open=False):
                    logs_output = gr.Textbox(
                        label="",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                        elem_id="logs-panel",
                    )
            
            # Help panel
            with gr.Column(scale=1):
                with gr.Accordion("📖 Quick Guide", open=True):
                    gr.Markdown("""
**Training Tips:**

- Start with **1 epoch** to test
- If "CUDA out of memory": reduce batch_size
- Loss should decrease over time

**Debug Mode:**
- ☑ Shows full error stack traces
- ☑ Logs GPU memory at each step
- ☑ Saves detailed log file
- ☑ Shows model loading details

**Output Directory:**
- `./output/` = relative to ai-compiler
- Checkpoints saved automatically
- Resume if interrupted
                    """)
        
    return epochs, batch_size, learning_rate, output_dir, debug_mode, save_logs, start_btn, stop_btn, progress_bar, logs_output


def step4_deploy():
    """Step 4: Deploy Model."""
    with gr.Column():
        gr.Markdown("## Step 4: Deploy Model")
        gr.Markdown("Your model is ready! Configure tokens and deploy.")
        
        # Token Section
        gr.Markdown("### 🔑 HuggingFace Tokens")
        gr.Markdown("""
        | Token Type | Purpose | How to Get |
        |------------|---------|------------|
        | **Read Token** | Download private datasets, gated models | [Get here](https://huggingface.co/settings/tokens) |
        | **Write Token** | Upload models to Hub | [Create with write access](https://huggingface.co/settings/tokens) |
        """)
        
        with gr.Row():
            with gr.Column():
                read_token_input = gr.Textbox(
                    label="🔒 Read Token (for private data)",
                    placeholder="hf_xxxxxxxxxxxxx",
                    type="password",
                    info="Use for: Private datasets, Llama, gated models",
                )
                read_token_btn = gr.Button("Save Read Token", size="sm")
                read_token_status = gr.Textbox(value="", label="Status", interactive=False)
            
            with gr.Column():
                write_token_input = gr.Textbox(
                    label="✍️ Write Token (for upload)",
                    placeholder="hf_xxxxxxxxxxxxx",
                    type="password",
                    info="Use for: Uploading models, creating repos",
                )
                write_token_btn = gr.Button("Save Write Token", size="sm")
                write_token_status = gr.Textbox(value="", label="Status", interactive=False)
        
        gr.Markdown("---")
        
        # Deploy Section
        gr.Markdown("### 📤 Deploy to HuggingFace Hub")
        
        model_path = gr.Textbox(value="./output", label="Trained Model Path")
        
        with gr.Row():
            hf_repo_name = gr.Textbox(
                label="Repository Name",
                placeholder="your-username/my-fine-tuned-model",
                info="Format: username/model-name",
            )
            private_repo = gr.Checkbox(label="Private Repository", value=False)
        
        hf_deploy_btn = gr.Button("🚀 Deploy to HuggingFace Hub", variant="primary", size="lg")
        
        with gr.Row():
            hf_deploy_status = gr.Textbox(label="Status", interactive=False, scale=2)
            hf_model_url = gr.Textbox(label="📎 Model URL (copy this!)", interactive=False, scale=2)
        
        gr.Markdown("---")
        
        # Local Export Section
        gr.Markdown("### 💻 Local Export")
        with gr.Row():
            export_format = gr.Radio(
                choices=["adapter", "merged", "gguf"],
                value="adapter",
                label="Export Format",
                info="adapter = small, merged = full model, gguf = for Ollama",
            )
            export_btn = gr.Button("Export Model")
            export_status = gr.Textbox(label="Status", interactive=False)
        
        gr.Markdown("---")
        
        # Test Section
        gr.Markdown("### 🧪 Test Your Deployed Model")
        with gr.Row():
            load_model_id = gr.Textbox(
                label="HuggingFace Model ID",
                placeholder="your-username/model-name",
                info="Enter the URL from above or any HuggingFace model",
            )
            load_btn = gr.Button("Load Model")
        
        test_prompt = gr.Textbox(label="Test Prompt", lines=3, placeholder="Enter a prompt to test your model...")
        test_output = gr.Textbox(label="Model Output", lines=5, interactive=False)
        test_btn = gr.Button("🚀 Generate", variant="primary")
        
        # Token handlers
        def save_read_token(token):
            from engine.utils.huggingface import set_read_token
            if set_read_token(token):
                return f"✅ Read token saved ({token[:8]}...)"
            return "❌ Invalid token (must start with hf_)"
        
        def save_write_token(token):
            from engine.utils.huggingface import set_write_token
            if set_write_token(token):
                return f"✅ Write token saved ({token[:8]}...)"
            return "❌ Invalid token (must start with hf_)"
        
        read_token_btn.click(save_read_token, [read_token_input], [read_token_status])
        write_token_btn.click(save_write_token, [write_token_input], [write_token_status])
        
    return (model_path, hf_repo_name, private_repo, hf_deploy_btn, hf_deploy_status, hf_model_url,
            export_format, export_btn, export_status,
            load_model_id, load_btn, test_prompt, test_output, test_btn)


# ============ Main App ============

def create_wizard_app() -> gr.Blocks:
    """Create the wizard-style Gradio app."""
    
    with gr.Blocks(
        title="AI Compiler v2",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 20px; }
        #logs-panel { font-family: monospace; font-size: 12px; }
        .step-indicator { display: flex; justify-content: center; gap: 20px; margin: 20px 0; }
        """
    ) as app:
        
        gr.Markdown(
            """
            # 🚀 AI Compiler v2
            ### Visual LLM Fine-Tuning Wizard
            """,
            elem_classes=["main-header"]
        )
        
        # Wizard tabs as steps
        with gr.Tabs() as tabs:
            with gr.Tab("1️⃣ Model", id=0):
                model_name, quantization, max_seq_length, lora_r, lora_alpha, gpu_status = step1_model()
                next1 = gr.Button("Next: Configure Data →", variant="primary")
            
            with gr.Tab("2️⃣ Data", id=1):
                data_source, data_path, data_format = step2_data()
                with gr.Row():
                    back2 = gr.Button("← Back")
                    next2 = gr.Button("Next: Train →", variant="primary")
            
            with gr.Tab("3️⃣ Train", id=2):
                (epochs, batch_size, learning_rate, output_dir, debug_mode, save_logs,
                 start_btn, stop_btn, progress_bar, logs_output) = step3_training()
                with gr.Row():
                    back3 = gr.Button("← Back")
                    next3 = gr.Button("Next: Deploy →", variant="primary")
            
            with gr.Tab("4️⃣ Deploy", id=3):
                (model_path, hf_repo_name, private_repo, hf_deploy_btn, hf_deploy_status, hf_model_url,
                 export_format, export_btn, export_status,
                 load_model_id, load_btn, test_prompt, test_output, test_btn) = step4_deploy()
                back4 = gr.Button("← Back to Training")
        
        # Tab navigation
        next1.click(lambda: gr.Tabs(selected=1), None, tabs)
        back2.click(lambda: gr.Tabs(selected=0), None, tabs)
        next2.click(lambda: gr.Tabs(selected=2), None, tabs)
        back3.click(lambda: gr.Tabs(selected=1), None, tabs)
        next3.click(lambda: gr.Tabs(selected=3), None, tabs)
        back4.click(lambda: gr.Tabs(selected=2), None, tabs)
        
        # Training handlers
        def start_training(
            model_name, quantization, max_seq_length, lora_r, lora_alpha,
            data_source, data_path, data_format,
            epochs, batch_size, learning_rate, output_dir,
            debug_mode, save_logs
        ):
            config = build_config(
                model_name, quantization, max_seq_length, lora_r, lora_alpha,
                data_source, data_path, data_format,
                epochs, batch_size, learning_rate, output_dir
            )
            
            # Add debug flag to config
            config_path = Path(output_dir) / "wizard_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            
            # Start with debug flag if enabled
            cmd = ["uv", "run", "ai-compile", "train", "--config", str(config_path)]
            if debug_mode:
                cmd.append("--dry-run")  # For now, just validate in debug
            
            training_manager.start(str(config_path))
            
            # Log debug info
            if debug_mode:
                training_manager.logs.append("🐛 DEBUG MODE ENABLED\n")
                training_manager.logs.append(f"Config: {config_path}\n")
                training_manager.logs.append(f"Model: {model_name}\n")
                training_manager.logs.append(f"Data: {data_path}\n")
            
            # Save logs to file if enabled
            if save_logs:
                log_file = Path(output_dir) / "training.log"
                training_manager.logs.append(f"📝 Logs will be saved to: {log_file}\n")
            
            # Stream updates
            while training_manager.is_running:
                yield training_manager.get_logs(), training_manager.get_progress()
                time.sleep(0.5)
            
            # Save final logs
            if save_logs:
                log_file = Path(output_dir) / "training.log"
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(training_manager.get_logs())
                training_manager.logs.append(f"\n💾 Logs saved to: {log_file}")
            
            yield training_manager.get_logs(), 100

        
        start_btn.click(
            start_training,
            inputs=[
                model_name, quantization, max_seq_length, lora_r, lora_alpha,
                data_source, data_path, data_format,
                epochs, batch_size, learning_rate, output_dir,
                debug_mode, save_logs
            ],
            outputs=[logs_output, progress_bar],
        )
        
        stop_btn.click(
            lambda: (training_manager.stop(), training_manager.get_logs()),
            outputs=[logs_output],
        )
        
        # Deploy handlers
        def deploy_to_hf(model_path, repo_name, private):
            """Deploy model to HuggingFace Hub."""
            from engine.utils.huggingface import upload_to_hub, validate_repo_name
            
            # Validate inputs
            if not repo_name:
                return "❌ Please enter a repository name", ""
            
            is_valid, msg = validate_repo_name(repo_name)
            if not is_valid:
                return f"❌ {msg}", ""
            
            # Upload
            result = upload_to_hub(
                model_path=model_path,
                repo_name=repo_name,
                private=private,
            )
            
            if result["success"]:
                return f"✅ Successfully deployed!", result["url"]
            else:
                return result["error"], ""
        
        hf_deploy_btn.click(
            deploy_to_hf,
            [model_path, hf_repo_name, private_repo],
            [hf_deploy_status, hf_model_url]
        )
        
        def export_model(model_path, format):
            return f"📦 Exporting to {format}... (use CLI: ai-compile export --format {format})"
        
        export_btn.click(export_model, [model_path, export_format], [export_status])
        
        gr.Markdown("---\n*AI Compiler v2 | Wizard Interface*")
    
    return app


def launch_wizard_ui(
    share: bool = False,
    server_port: int = 7862,
    server_name: str = "127.0.0.1",
):
    """Launch the wizard UI."""
    app = create_wizard_app()
    app.launch(share=share, server_port=server_port, server_name=server_name)


if __name__ == "__main__":
    launch_wizard_ui()
