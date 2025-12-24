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
    
    # Status constants
    STATUS_IDLE = "idle"
    STATUS_RUNNING = "running"
    STATUS_SUCCESS = "success"
    STATUS_FAILED = "failed"
    
    def __init__(self):
        self.process = None
        self.logs = []
        self.is_running = False
        self.progress = 0
        self.status = self.STATUS_IDLE
        self.final_loss = None
        self.error_message = None
    
    def start(self, config_path: str):
        self.logs = ["🚀 Starting training...\n"]
        self.is_running = True
        self.progress = 0
        self.status = self.STATUS_RUNNING
        self.final_loss = None
        self.error_message = None
        
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
                # Parse loss
                if "loss" in line.lower():
                    try:
                        import re
                        match = re.search(r'loss[:\s]+([0-9.]+)', line.lower())
                        if match:
                            self.final_loss = float(match.group(1))
                    except:
                        pass
                # Check for errors
                if "error" in line.lower() or "exception" in line.lower():
                    self.error_message = line.strip()
            
            # Determine final status
            exit_code = self.process.wait()
            self.is_running = False
            
            if exit_code == 0:
                self.status = self.STATUS_SUCCESS
                self.logs.append("\n✅ Training complete!")
            else:
                self.status = self.STATUS_FAILED
                self.logs.append(f"\n❌ Training failed (exit code: {exit_code})")
        
        thread = threading.Thread(target=read_output, daemon=True)
        thread.start()
    
    def stop(self):
        if self.process:
            self.process.terminate()
            self.is_running = False
            self.status = self.STATUS_FAILED
            self.logs.append("\n⏹ Training stopped by user.")
    
    def get_logs(self) -> str:
        # Return ALL logs, not just last 50
        return "".join(self.logs)
    
    def get_progress(self) -> int:
        return self.progress
    
    def get_status(self) -> str:
        return self.status
    
    def get_status_html(self) -> str:
        """Get status as styled HTML."""
        if self.status == self.STATUS_IDLE:
            return "⏳ **Waiting to start...**"
        elif self.status == self.STATUS_RUNNING:
            return "🔄 **Training in progress...**"
        elif self.status == self.STATUS_SUCCESS:
            loss_str = f" | Loss: {self.final_loss:.4f}" if self.final_loss else ""
            return f"✅ **Training Complete!**{loss_str}"
        else:  # FAILED
            return f"❌ **Training Failed**"


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
    """Step 1: Model Selection (Redesigned)."""
    with gr.Column(elem_classes=["premium-card"]):
        with gr.Row():
            # Left: Main Selection
            with gr.Column(scale=2):
                gr.Markdown("### 🧠 Select Base Model", elem_classes=["main-header"])
                gr.Markdown("Choose a pre-trained foundation model to fine-tune.", elem_classes=["mono-text"])
                
                model_name = gr.Dropdown(
                    choices=MODELS,
                    value=MODELS[0],
                    label="",
                    info="Select architecture",
                    elem_id="model-dropdown",
                    interactive=True,
                )
                
                # Recommendation Card
                with gr.Group():
                    model_recommendation = gr.Textbox(
                        value=MODEL_INFO.get(MODELS[0], ""),
                        label="💡 AI Recommendation",
                        interactive=False,
                        lines=2,
                        show_label=True,
                    )

                # Advanced Settings in Accordion
                with gr.Accordion("⚙️ Advanced Configuration", open=False, elem_classes=["transparent-accordion"]):
                    gr.Markdown("Optimization & LoRA Parameters")
                    with gr.Row():
                        quantization = gr.Radio(
                            choices=QUANTIZATION,
                            value="4bit",
                            label="Quantization",
                            info="Storage precision",
                        )
                        max_seq_length = gr.Slider(256, 4096, value=2048, step=256, label="Context Length")
                    
                    with gr.Row():
                        lora_r = gr.Slider(4, 128, value=16, step=4, label="LoRA Rank (Capacity)")
                        lora_alpha = gr.Slider(8, 256, value=32, step=8, label="LoRA Alpha (Scaling)")

                # GPU Status Card (Always visible)
                gpu_status = gr.Textbox(
                    value=get_gpu_status(),
                    label="🎮 GPU Resources",
                    interactive=False,
                    lines=3,
                    elem_classes=["mono-text"]
                )

            # Right: Quick Guide
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("#### 📖 Quick Guide")
                    gr.Markdown("""
                    **1. Choose Model**
                    - `Llama-3.2-1B`: Best for beginners
                    - `Mistral-7B`: High quality chat
                    - `Whisper`: Speech-to-Text
                    
                    **2. Quantization**
                    - `4bit`: Best for 90% of cases (low VRAM)
                    - `8bit`: Higher precision
                    
                    **3. LoRA Rank**
                    - `16`: Standard fine-tuning
                    - `64+`: Complex reasoning tasks
                    """, elem_classes=["mono-text"])
        
        def update_recommendation(model):
            return MODEL_INFO.get(model, "No info available")
        
        model_name.change(update_recommendation, [model_name], [model_recommendation])
        
    return model_name, quantization, max_seq_length, lora_r, lora_alpha, gpu_status


def step2_data():
    """Step 2: Data Configuration (Redesigned with Cards)."""
    with gr.Column(elem_classes=["premium-card"]):
        gr.Markdown("### 📂 Data & Privacy", elem_classes=["main-header"])
        
        with gr.Row():
            # Left: Data Selection
            with gr.Column(scale=2):
                gr.Markdown("#### Select Data Source", elem_classes=["mono-text"])
                
                data_source = gr.Radio(
                    choices=DATA_SOURCES,
                    value="huggingface",
                    label="",
                    info="Choose where your training data comes from",
                    elem_classes=["premium-radio"],
                )
                
                with gr.Group():
                    data_path = gr.Textbox(
                        value="tatsu-lab/alpaca",
                        label="Source Path / ID",
                        placeholder="e.g., tatsu-lab/alpaca or data/train.csv",
                        info="HuggingFace dataset ID or local path",
                    )
                    
                    hf_examples = gr.Dropdown(
                        choices=DATASET_CHOICES,
                        label="⚡ Quick Select",
                        info="Pre-validated datasets for fine-tuning",
                        interactive=True,
                    )
                
                data_format = gr.Radio(
                    choices=FORMATS,
                    value="alpaca",
                    label="Data Format",
                    info="Ensure your data matches this schema",
                )
                
                # Authentication Card
                with gr.Accordion("� Authentication (Private Data)", open=False, elem_classes=["premium-card", "transparent-accordion"]):
                    gr.Markdown("Required for gated models (Llama 3) or private datasets.")
                    with gr.Row():
                        hf_token_input = gr.Textbox(
                            label="HuggingFace Token",
                            placeholder="hf_...",
                            type="password",
                            scale=3,
                        )
                        hf_token_btn = gr.Button("Validate", size="sm", scale=1, variant="secondary")
                    
                    hf_token_status = gr.Textbox(
                        value=check_hf_token(),
                        label="",
                        interactive=False,
                        elem_classes=["mono-text"]
                    )
            
            # Right: Guide
            with gr.Column(scale=1):
                gr.Markdown("#### � Format Guide", elem_classes=["mono-text"])
                gr.Markdown("""
                **Alpaca (Standard)**
                ```json
                {"instruction": "...", "output": "..."}
                ```
                
                **ChatML (Conversation)**
                ```json
                {"messages": [{"role": "user", "content": "..."}]}
                ```
                """, elem_classes=["mono-text"])
        
        def update_path(example):
            return example if example else "tatsu-lab/alpaca"
        
        hf_examples.change(update_path, [hf_examples], [data_path])
        hf_token_btn.click(save_hf_token, [hf_token_input], [hf_token_status])
        
    return data_source, data_path, data_format


def step3_training():
    """Step 3: Training Dashboard (Redesigned)."""
    with gr.Column(elem_classes=["premium-card"]):
        gr.Markdown("### 🚀 Training Dashboard", elem_classes=["main-header"])
        
        with gr.Row():
            # Left: Controls
            with gr.Column(scale=2):
                # Stats Row (Mini Cards)
                with gr.Row():
                    epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs", info="Passes through data")
                    batch_size = gr.Slider(1, 16, value=4, step=1, label="Batch Size", info="Items per step")
                    learning_rate = gr.Number(value=2e-4, label="Learning Rate", info="Step size")
                
                output_dir = gr.Textbox(value="./output", label="Artifact Output Path")
                
                with gr.Accordion("🛠️ Developer Tools", open=False, elem_classes=["transparent-accordion"]):
                    with gr.Row():
                        debug_mode = gr.Checkbox(label="Verbose Debug Logging", value=False)
                        save_logs = gr.Checkbox(label="Save Logs to File", value=True)
                
                # Big Action Buttons
                with gr.Row():
                    start_btn = gr.Button("▶ START TRAINING", variant="primary", scale=2, elem_classes=["primary-btn"])
                    stop_btn = gr.Button("⏹ ABORT", variant="stop", scale=1)
                
                # Progress
                progress_bar = gr.Slider(0, 100, 0, label="Training Progress", interactive=False)
                
                # Status Indicator (3 states: idle, running, success/failed)
                training_status = gr.Markdown(
                    value="<div class='status-idle'>⏳ Waiting to start...</div>",
                    elem_id="training-status"
                )

            # Right: Live Terminal
            with gr.Column(scale=2):
                gr.Markdown("#### 💻 Live Terminal", elem_classes=["mono-text"])
                logs_output = gr.Textbox(
                    label="",
                    lines=25,
                    max_lines=None,  # No limit on stored lines
                    interactive=False,
                    elem_id="logs-panel",
                    placeholder="Waiting for training to start...",
                    show_label=False,
                )
        
    return epochs, batch_size, learning_rate, output_dir, debug_mode, save_logs, start_btn, stop_btn, progress_bar, logs_output, training_status



# ============ Step 4: Deploy (Netlify-Style) ============

def step4_deploy():
    """Step 4: Deployment Control Center (Netlify-Style)."""
    
    # Status Banner (Dynamic)
    deploy_status_banner = gr.Markdown(
        value="<div class='deploy-banner badge-error'>⚠️ Training not completed. Deployment disabled.</div>",
        visible=True
    )
    
    with gr.Row():
        # Left: Production Environment
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("""
                <div class="deploy-header">
                    <span style="font-weight:600">☁️ Production Environment</span>
                    <span class="badge badge-idle">OFFLINE</span>
                </div>
                """, elem_classes=["deploy-box-header"])
                
                with gr.Column(elem_classes=["deploy-box-body"]):
                    gr.Markdown("**Destination: HuggingFace Hub**", elem_classes=["mono-text"])
                    
                    # Auth Section
                    with gr.Accordion("🔑 API Authentication", open=False, elem_classes=["transparent-accordion"]):
                        with gr.Row():
                            read_token_input = gr.Textbox(label="Read Token", placeholder="hf_...", type="password")
                            read_token_btn = gr.Button("Save", size="sm")
                        read_token_status = gr.Textbox(label="", interactive=False, visible=False)
                        
                        with gr.Row():
                            write_token_input = gr.Textbox(label="Write Token", placeholder="hf_...", type="password")
                            write_token_btn = gr.Button("Save", size="sm")
                        write_token_status = gr.Textbox(label="", interactive=False, visible=False)

                    gr.Markdown("---")
                    
                    model_path = gr.Textbox(value="./output", label="Artifact Path", interactive=False)
                    
                    hf_repo_name = gr.Textbox(
                        label="Target Repository",
                        placeholder="username/my-awesome-model",
                        info="e.g. khedhar108/voice-assistant-v1"
                    )
                    
                    private_repo = gr.Checkbox(label="Private Repository", value=True)
                    
                    # Deploy Button (Disabled by default)
                    hf_deploy_btn = gr.Button("🚀 Deploy to Production", variant="primary", interactive=False)
                    hf_deploy_status = gr.Textbox(label="Deployment Log", interactive=False, lines=3)
                    hf_model_url = gr.Markdown(visible=False)

        # Right: Preview & Sandbox
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("""
                <div class="deploy-header">
                    <span style="font-weight:600">⚡ Preview & Export</span>
                    <span class="badge badge-success">READY</span>
                </div>
                """, elem_classes=["deploy-box-header"])
                
                with gr.Column(elem_classes=["deploy-box-body"]):
                    gr.Markdown("#### 📦 Export Artifacts", elem_classes=["mono-text"])
                    with gr.Row():
                        export_format = gr.Radio(
                            choices=["adapter", "merged", "gguf"],
                            value="adapter",
                            label="Format",
                            container=False
                        )
                        export_btn = gr.Button("Download", size="sm")
                    export_status = gr.Textbox(label="", interactive=False)
                    
                    gr.Markdown("---")
                    gr.Markdown("#### 🧪 Live Sandbox", elem_classes=["mono-text"])
                    
                    load_model_id = gr.Textbox(
                        label="Model Adapter ID",
                        placeholder="./output",
                        value="./output"
                    )
                    load_btn = gr.Button("Reload Model", size="sm", variant="secondary")
                    
                    test_prompt = gr.Textbox(label="Test Prompt", lines=2, placeholder="Type your query here...")
                    test_btn = gr.Button("Generate Preview", variant="primary", size="sm")
                    test_output = gr.Textbox(label="Output", lines=3, interactive=False)

    # Handlers (Auth)
    def save_read_token(token):
        from engine.utils.huggingface import set_read_token
        if set_read_token(token): return "✅ Saved"
        return "❌ Invalid"
    
    def save_write_token(token):
        from engine.utils.huggingface import set_write_token
        if set_write_token(token): return "✅ Saved"
        return "❌ Invalid"
    
    read_token_btn.click(save_read_token, [read_token_input], [read_token_status])
    write_token_btn.click(save_write_token, [write_token_input], [write_token_status])
    
    return (deploy_status_banner, model_path, hf_repo_name, private_repo, hf_deploy_btn, hf_deploy_status, hf_model_url,
            export_format, export_btn, export_status,
            load_model_id, load_btn, test_prompt, test_output, test_btn)



# ============ Main App ============

def create_wizard_app() -> gr.Blocks:
    """Create the wizard-style Gradio app."""
    
    with gr.Blocks(
        title="AI Compiler v2",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        css="""
        /* Global Dark Theme Overrides */
        body, .gradio-container { background-color: #0f172a; color: #e2e8f0; }
        
        /* Premium Card Styling */
        .premium-card {
            background: rgba(30, 41, 59, 0.7) !important;
            border: 1px solid rgba(148, 163, 184, 0.1) !important;
            border-radius: 12px !important;
            padding: 20px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
            backdrop-filter: blur(10px);
        }
        
        /* Header */
        .main-header h1 {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem !important;
            font-weight: 800 !important;
            margin-bottom: 0.5rem;
        }
        
        /* Buttons */
        .primary-btn {
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
            border: none !important;
            color: white !important;
            font-weight: 600 !important;
            transition: all 0.2s ease;
        }
        .primary-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 15px -3px rgba(124, 58, 237, 0.3);
        }
        
        /* Step Indicator */
        .step-indicator {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 30px;
        }
        
        /* Utility */
        .mono-text { font-family: 'JetBrains Mono', monospace; font-size: 13px; }
        .success-text { color: #4ade80 !important; font-weight: 600; }
        .error-text { color: #f87171 !important; font-weight: 600; }
        
        /* Custom Accordion */
        .transparent-accordion {
            background: transparent !important;
            border: none !important;
        }
        
        /* Scrollable Logs Panel */
        #logs-panel textarea {
            max-height: 500px !important;
            overflow-y: auto !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 12px !important;
            background: #1e293b !important;
            color: #e2e8f0 !important;
        }
        
        /* Status Indicator Styles */
        .status-idle { color: #94a3b8; }
        .status-success { color: #4ade80; font-weight: bold; }
        .status-failed { color: #f87171; font-weight: bold; }
        
        /* Blinking Animation for In-Progress */
        .status-running {
            color: #fbbf24;
            font-weight: bold;
            animation: blink 1s ease-in-out infinite;
        }
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        
        /* Boxed Card for Deploy */
        .deploy-card {
            background: rgba(30, 41, 59, 0.9) !important;
            border: 1px solid rgba(99, 102, 241, 0.3) !important;
            border-radius: 10px !important;
            padding: 16px !important;
            margin-bottom: 12px !important;
        }
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
                 start_btn, stop_btn, progress_bar, logs_output, training_status) = step3_training()
                with gr.Row():
                    back3 = gr.Button("← Back")
                    next3 = gr.Button("Next: Deploy →", variant="primary")
            
            with gr.Tab("4️⃣ Deploy", id=3):
                (deploy_status_banner, model_path, hf_repo_name, private_repo, hf_deploy_btn, hf_deploy_status, hf_model_url,
                 export_format, export_btn, export_status,
                 load_model_id, load_btn, test_prompt, test_output, test_btn) = step4_deploy()
                back4 = gr.Button("← Back to Training")
        
        # Tab navigation
        next1.click(lambda: gr.Tabs(selected=1), None, tabs)
        back2.click(lambda: gr.Tabs(selected=0), None, tabs)
        next2.click(lambda: gr.Tabs(selected=2), None, tabs)
        back3.click(lambda: gr.Tabs(selected=1), None, tabs)
        # Logic: Check status when moving to Deploy tab
        def check_deploy_status():
            from engine.ui_v2.app import TrainingManager, training_manager
            
            if training_manager.status == TrainingManager.STATUS_SUCCESS:
                return (
                    gr.Markdown("<div class='deploy-banner badge-live'>✅ Training Successful. Ready to Deploy.</div>", visible=True),
                    gr.Button(interactive=True),
                    gr.Tabs(selected=3)
                )
            elif training_manager.status == TrainingManager.STATUS_RUNNING:
                return (
                    gr.Markdown("<div class='deploy-banner badge-building'>⚠️ Training in progress... Please wait.</div>", visible=True),
                    gr.Button(interactive=False),
                    gr.Tabs(selected=3)
                )
            else:
                return (
                    gr.Markdown("<div class='deploy-banner badge-error'>⚠️ Training not completed. Deployment disabled.</div>", visible=True),
                    gr.Button(interactive=False),
                    gr.Tabs(selected=3)
                )

        next3.click(
            check_deploy_status,
            inputs=None,
            outputs=[deploy_status_banner, hf_deploy_btn, tabs]
        )
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
            
            # Stream updates with status
            while training_manager.is_running:
                status_html = f"<div class='status-running'>🔄 Training in progress... ({training_manager.get_progress()}%)</div>"
                yield training_manager.get_logs(), training_manager.get_progress(), status_html
                time.sleep(0.5)
            
            # Final status
            if training_manager.status == TrainingManager.STATUS_SUCCESS:
                final_status = f"<div class='status-success'>✅ Training Complete! Loss: {training_manager.final_loss or 'N/A'}</div>"
            else:
                final_status = f"<div class='status-failed'>❌ Training Failed</div>"
            
            # Save final logs
            if save_logs:
                log_file = Path(output_dir) / "training.log"
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(training_manager.get_logs())
                training_manager.logs.append(f"\n💾 Logs saved to: {log_file}")
            
            yield training_manager.get_logs(), 100, final_status

        
        start_btn.click(
            start_training,
            inputs=[
                model_name, quantization, max_seq_length, lora_r, lora_alpha,
                data_source, data_path, data_format,
                epochs, batch_size, learning_rate, output_dir,
                debug_mode, save_logs
            ],
            outputs=[logs_output, progress_bar, training_status],
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
