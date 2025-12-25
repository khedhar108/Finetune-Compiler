"""
Step 2: Data Configuration.
"""

import gradio as gr
from engine.ui_v2.consts import DATA_SOURCES, DATASET_CHOICES, FORMATS
from engine.ui_v2.utils import check_hf_token, save_hf_token

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
                    info="Choose your data origin",
                    elem_classes=["premium-radio", "radio-cards"],
                    container=False, 
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
                with gr.Accordion("🔒 Authentication (Private Data)", open=False, elem_classes=["premium-card", "transparent-accordion"]):
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
                gr.Markdown("#### 📋 Format Guide", elem_classes=["mono-text"])
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
