"""
Step 1: Model Selection.
"""

import gradio as gr
from engine.ui_v2.consts import MODELS, MODEL_INFO, QUANTIZATION
from engine.ui_v2.utils import get_gpu_status
from engine.utils.huggingface import search_models

def step1_model():
    """Step 1: Model Selection (Premium Redesign)."""
    with gr.Column(elem_classes=["premium-card"]):
        gr.Markdown("### 🧠 Model Foundation", elem_classes=["main-header"])
        
        with gr.Row():
            # LEFT COLUMN: Configuration (70%)
            with gr.Column(scale=7):
                
                # 1. Architecture Selection
                gr.Markdown("#### Architecture Selection", elem_classes=["section-header"])
                
                # Search input (full width, triggers on Enter)
                search_query = gr.Textbox(
                    placeholder="Search Hub (e.g. 'mistral') - Press Enter to search",
                    show_label=False,
                    autofocus=True
                )
                
                # Model dropdown (full width)
                model_name = gr.Dropdown(
                    choices=MODELS,
                    value=MODELS[0],
                    label="Select Model",
                    interactive=True,
                    allow_custom_value=True
                )
                
                # Current Selection (inline helper)
                model_recommendation = gr.Markdown(
                    f"💡 *{MODEL_INFO.get(MODELS[0], 'Custom Model')}*",
                    elem_classes=["helper-text"]
                )
                
                gr.Markdown("---")
                
                # 2. Optimization Settings
                gr.Markdown("#### Optimization Strategy", elem_classes=["section-header"])
                quantization = gr.Radio(
                    choices=QUANTIZATION,
                    value="4bit",
                    label="Quantization",
                    info="Memory vs Precision trade-off",
                    elem_classes=["radio-cards"]
                )
               
                max_seq_length = gr.Slider(
                    256, 4096, 
                    value=2048, 
                    step=256, 
                    label="Context Window", 
                    info="Max tokens per sequence"
                )

                # 3. Advanced Settings
                with gr.Accordion("🛠️ Advanced LoRA Configuration", open=False):
                    with gr.Row():
                        lora_r = gr.Slider(4, 128, value=16, step=4, label="LoRA Rank (r)")
                        lora_alpha = gr.Slider(8, 256, value=32, step=8, label="LoRA Alpha")

            # RIGHT COLUMN: Status Only (30%)
            with gr.Column(scale=3):
                gr.Markdown("#### System Status", elem_classes=["section-header"])
                search_status = gr.Markdown("Ready to search...", elem_classes=["status-idle"])
                
                gpu_status = gr.Textbox(
                    value=get_gpu_status(),
                    show_label=False, 
                    interactive=False,
                    lines=4,
                    elem_classes=["console-output"]
                )

    # --- Event Handlers ---

    def update_recommendation(model):
        info = MODEL_INFO.get(model, 'Custom model from HuggingFace Hub')
        return f"💡 *{info}*"

    def perform_search(query):
        if not query or len(query) < 2:
            return gr.update(), "⚠️ Enter 2+ chars"
        
        try:
            raw_results = search_models(query, limit=20)
        except Exception:
            return gr.update(), "❌ Connection Error"
        
        # Strict filtering
        results = [m for m in raw_results if query.lower() in m.lower()]
        
        if not results:
            return gr.update(), f"❌ No models found for '{query}'"
        
        # Update choices
        new_choices = list(sorted(list(set(MODELS + results))))
        first_match = results[0]
        
        return gr.update(choices=new_choices, value=first_match), f"✅ Found {len(results)} matches"

    # Wire up events
    model_name.change(update_recommendation, [model_name], [model_recommendation])
    search_query.submit(perform_search, [search_query], [model_name, search_status])
    
    return model_name, quantization, max_seq_length, lora_r, lora_alpha, gpu_status
