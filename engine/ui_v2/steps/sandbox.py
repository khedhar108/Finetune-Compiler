"""
Step 5: Live Sandbox.
"""

import gradio as gr
from engine.ui_v2.components import CustomDropdown

def step5_sandbox():
    """Step 5: Interactive Sandbox for Testing Models."""
    
    gr.Markdown("## 🧪 Live Sandbox")
    gr.Markdown("Test your trained model immediately in this interactive playground.")
    
    with gr.Row():
        # Left: Settings
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### ⚙️ Model Settings")
                
                model_path_input = gr.Textbox(
                    label="Trained Model Path", 
                    value="./output",
                    placeholder="e.g. ./output or ./checkpoints/checkpoint-500",
                    info="Folder containing adapter_config.json (Default: ./output)"
                )
                
                with gr.Row():
                    quantization = CustomDropdown(
                        choices=["None", "4bit", "8bit"],
                        value="4bit",
                        label="Quantization"
                    )
                    device = CustomDropdown(
                        choices=["auto", "cuda", "cpu"],
                        value="auto",
                        label="Device"
                    )
                
                load_btn = gr.Button("🔄 Load Model", variant="secondary")
                load_status = gr.Textbox(label="Status", interactive=False, max_lines=1)
        
        # Right: Chat
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### 💬 Chat Interface")
                
                chatbot = gr.Chatbot(label="Conversation", height=400)
                msg = gr.Textbox(label="Your Message", placeholder="Type here...", lines=2)
                
                with gr.Row():
                    clear = gr.Button("🗑️ Clear")
                    submit = gr.Button("📤 Send", variant="primary")
                    
    # Return all components needed for wiring in app.py
    return (model_path_input, quantization, device, load_btn, load_status,
            chatbot, msg, clear, submit)
