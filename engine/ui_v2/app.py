"""
AI Compiler Wizard UI v2.

A step-by-step wizard interface for training and deploying models.
Refactored for modularity.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

try:
    import gradio as gr
except ImportError:
    raise ImportError(
        "Gradio is required for UI. Install with: uv sync --extra ui"
    )

from engine.ui_v2.consts import UI_CSS
from engine.ui_v2.manager import TrainingManager, training_manager
from engine.ui_v2.utils import build_config
from engine.ui_v2.steps.model import step1_model
from engine.ui_v2.steps.data import step2_data
from engine.ui_v2.steps.train import step3_training
from engine.ui_v2.steps.deploy import step4_deploy


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
        css=UI_CSS
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
