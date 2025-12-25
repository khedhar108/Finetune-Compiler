"""
Step 3: Training Dashboard.
"""

import gradio as gr

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
