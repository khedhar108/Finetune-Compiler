"""
AI Compiler Gradio UI.

A visual interface for configuring, training, and monitoring LLM fine-tuning.
Runs the compiler as a subprocess - no performance impact on training.
"""

from __future__ import annotations

import json
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
from engine.utils.memory import get_gpu_info, get_memory_stats


# Available models
MODELS = [
    "unsloth/Llama-3.2-1B",
    "unsloth/Llama-3.2-3B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2",
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Llama-2-7b-hf",
]

QUANTIZATION_OPTIONS = ["4bit", "8bit", "none"]
FORMAT_OPTIONS = ["alpaca", "chatml", "completion", "custom"]


class TrainingProcess:
    """Manages the training subprocess."""
    
    def __init__(self):
        self.process = None
        self.logs = []
        self.is_running = False
    
    def start(self, config_path: str) -> None:
        """Start training subprocess."""
        self.logs = []
        self.is_running = True
        
        self.process = subprocess.Popen(
            ["uv", "run", "ai-compile", "train", "--config", config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        # Read output in background thread
        def read_output():
            for line in self.process.stdout:
                self.logs.append(line.strip())
            self.is_running = False
        
        thread = threading.Thread(target=read_output, daemon=True)
        thread.start()
    
    def stop(self) -> None:
        """Stop training subprocess."""
        if self.process:
            self.process.terminate()
            self.is_running = False
    
    def get_logs(self) -> str:
        """Get all logs as string."""
        return "\n".join(self.logs[-100:])  # Last 100 lines


# Global training process
training_process = TrainingProcess()


def build_config(
    model_name: str,
    quantization: str,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dataset_path: str,
    data_format: str,
    output_dir: str,
) -> dict:
    """Build configuration from UI inputs."""
    return {
        "project": {
            "name": "ui-training",
            "output_dir": output_dir,
            "seed": 42,
        },
        "data": {
            "source": "csv",
            "path": dataset_path,
            "format": data_format,
            "test_split": 0.1,
        },
        "model": {
            "name": model_name,
            "task": "causal_lm",
            "quantization": quantization,
            "max_seq_length": max_seq_length,
        },
        "lora": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
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


def start_training(
    model_name, quantization, max_seq_length,
    lora_r, lora_alpha, lora_dropout,
    epochs, batch_size, learning_rate,
    dataset_path, data_format, output_dir,
) -> Generator[str, None, None]:
    """Start training and stream logs."""
    
    # Build config
    config = build_config(
        model_name, quantization, max_seq_length,
        lora_r, lora_alpha, lora_dropout,
        epochs, batch_size, learning_rate,
        dataset_path, data_format, output_dir,
    )
    
    # Save config
    config_path = Path(output_dir) / "ui_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    yield f"📝 Config saved to {config_path}\n"
    yield "🚀 Starting training...\n\n"
    
    # Start training
    training_process.start(str(config_path))
    
    # Stream logs
    last_log_count = 0
    while training_process.is_running:
        if len(training_process.logs) > last_log_count:
            new_logs = training_process.logs[last_log_count:]
            last_log_count = len(training_process.logs)
            yield "\n".join(new_logs) + "\n"
        time.sleep(0.5)
    
    # Final logs
    yield "\n✅ Training complete!"


def stop_training() -> str:
    """Stop training."""
    training_process.stop()
    return "⏹ Training stopped."


def get_gpu_status() -> str:
    """Get GPU status string."""
    try:
        stats = get_memory_stats()
        if stats["available"]:
            return (
                f"🎮 GPU: {stats['device_name']}\n"
                f"📊 Memory: {stats['used_gb']:.1f}GB / {stats['total_gb']:.1f}GB "
                f"({stats['percent_used']:.0f}%)"
            )
        else:
            return "❌ No GPU available"
    except Exception as e:
        return f"⚠️ GPU check failed: {e}"


def create_app() -> gr.Blocks:
    """Create the Gradio app."""
    
    with gr.Blocks(
        title="AI Compiler",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 20px; }
        .gpu-status { font-family: monospace; }
        """
    ) as app:
        
        gr.Markdown(
            """
            # 🚀 AI Compiler
            ### Visual LLM Fine-Tuning Interface
            """,
            elem_classes=["main-title"]
        )
        
        with gr.Tabs():
            # Tab 1: Configuration
            with gr.Tab("🔧 Configuration"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Model Settings")
                        model_name = gr.Dropdown(
                            choices=MODELS,
                            value=MODELS[0],
                            label="Model",
                        )
                        quantization = gr.Dropdown(
                            choices=QUANTIZATION_OPTIONS,
                            value="4bit",
                            label="Quantization",
                        )
                        max_seq_length = gr.Slider(
                            minimum=256,
                            maximum=4096,
                            value=2048,
                            step=256,
                            label="Max Sequence Length",
                        )
                    
                    with gr.Column():
                        gr.Markdown("### LoRA Settings")
                        lora_r = gr.Slider(
                            minimum=4,
                            maximum=64,
                            value=16,
                            step=4,
                            label="LoRA Rank (r)",
                        )
                        lora_alpha = gr.Slider(
                            minimum=8,
                            maximum=128,
                            value=32,
                            step=8,
                            label="LoRA Alpha",
                        )
                        lora_dropout = gr.Slider(
                            minimum=0.0,
                            maximum=0.3,
                            value=0.05,
                            step=0.01,
                            label="LoRA Dropout",
                        )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Training Settings")
                        epochs = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Epochs",
                        )
                        batch_size = gr.Slider(
                            minimum=1,
                            maximum=16,
                            value=4,
                            step=1,
                            label="Batch Size",
                        )
                        learning_rate = gr.Number(
                            value=2e-4,
                            label="Learning Rate",
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Data Settings")
                        dataset_path = gr.Textbox(
                            value="./data/train.csv",
                            label="Dataset Path",
                        )
                        data_format = gr.Dropdown(
                            choices=FORMAT_OPTIONS,
                            value="alpaca",
                            label="Data Format",
                        )
                        output_dir = gr.Textbox(
                            value="./output",
                            label="Output Directory",
                        )
            
            # Tab 2: Training
            with gr.Tab("📊 Training"):
                with gr.Row():
                    start_btn = gr.Button("▶ Start Training", variant="primary")
                    stop_btn = gr.Button("⏹ Stop", variant="stop")
                
                gpu_status = gr.Textbox(
                    value=get_gpu_status(),
                    label="GPU Status",
                    interactive=False,
                    elem_classes=["gpu-status"],
                )
                
                refresh_gpu = gr.Button("🔄 Refresh GPU Status")
                
                logs_output = gr.Textbox(
                    label="Training Logs",
                    lines=20,
                    max_lines=30,
                    interactive=False,
                )
                
                # Event handlers
                start_btn.click(
                    fn=start_training,
                    inputs=[
                        model_name, quantization, max_seq_length,
                        lora_r, lora_alpha, lora_dropout,
                        epochs, batch_size, learning_rate,
                        dataset_path, data_format, output_dir,
                    ],
                    outputs=logs_output,
                )
                
                stop_btn.click(fn=stop_training, outputs=logs_output)
                refresh_gpu.click(fn=get_gpu_status, outputs=gpu_status)
            
            # Tab 3: Test
            with gr.Tab("🧪 Test"):
                gr.Markdown("### Test Your Fine-Tuned Model")
                
                model_path = gr.Textbox(
                    value="./output",
                    label="Model Path",
                )
                
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3,
                )
                
                generate_btn = gr.Button("🚀 Generate", variant="primary")
                
                output_text = gr.Textbox(
                    label="Output",
                    lines=10,
                    interactive=False,
                )
                
                gr.Markdown(
                    """
                    > **Note:** Use `ai-compile infer` for advanced inference options.
                    """
                )
        
        gr.Markdown(
            """
            ---
            *Powered by AI Compiler | [Documentation](./docs/)*
            """
        )
    
    return app


def launch_ui(
    share: bool = False,
    server_port: int = 7860,
    server_name: str = "127.0.0.1",
) -> None:
    """
    Launch the Gradio UI.
    
    Args:
        share: Create a public share link
        server_port: Port to run on
        server_name: Host to bind to
    """
    app = create_app()
    app.launch(
        share=share,
        server_port=server_port,
        server_name=server_name,
    )


if __name__ == "__main__":
    launch_ui()
