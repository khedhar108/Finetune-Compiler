"""
Step 1: Model Selection.
"""

import gradio as gr
from engine.ui_v2.consts import MODELS, MODEL_INFO, QUANTIZATION
from engine.ui_v2.utils import get_gpu_status
from engine.utils.huggingface import search_models
from engine.models.backend import get_optimal_backend, BackendType
from engine.ui_v2.components import CustomDropdown, create_info_box
from engine.ui_v2.consts import HELP_CONTENT

def step1_model():
    """Step 1: Model Selection (Premium Redesign)."""
    with gr.Column(elem_classes=["premium-card"]):
        gr.Markdown("### 🧠 Model Foundation", elem_classes=["main-header"])
        
        # Grid Layout: 2/3 Main Content, 1/3 Sidebar
        with gr.Row(elem_classes=["cols-2-1"]):
            
            # --- MAIN COLUMN ---
            with gr.Column():
                
                # 1. Architecture Selection
                gr.Markdown("#### Architecture Selection", elem_classes=["section-header"])
                
                # Search input (enters triggers search)
                search_query = gr.Textbox(
                    placeholder="Search Hub (e.g. 'mistral') - Press Enter to search",
                    show_label=False,
                    autofocus=True,
                    elem_classes=["gr-input"]
                )
                
                # Model dropdown
                model_name = CustomDropdown(
                    choices=MODELS,
                    value=MODELS[0],
                    label="Select Model",
                    interactive=True,
                    allow_custom_value=True,
                    elem_id="model-dropdown"
                )
                
                # Inline Helper & Status
                with gr.Row(elem_classes=["status-container"]):
                    model_recommendation = gr.Markdown(
                        f"💡 *{MODEL_INFO.get(MODELS[0], 'Custom Model')}*",
                        elem_classes=["helper-text"]
                    )
                    backend_status = gr.Markdown(
                        value="Checking acceleration...",
                        elem_classes=["badge-idle"]
                    )
                
                # Performance Savings Card (Dynamic)
                performance_stats = gr.HTML(
                    value="",
                    visible=False,
                    elem_classes=["content-box"]
                )
                
                # 2. Optimization Settings (Horizontal)
                gr.Markdown("#### Optimization Strategy", elem_classes=["section-header"])
                quantization = gr.Radio(
                    choices=QUANTIZATION,
                    value="4bit",
                    label="Quantization",
                    info="Memory vs Precision trade-off",
                    elem_classes=["radio-cards"]
                )
                create_info_box("Quantization", HELP_CONTENT["quantization"])
                
                max_seq_length = gr.Slider(
                    256, 4096, 
                    value=2048, 
                    step=256, 
                    label="Context Window", 
                    info="Max tokens per sequence"
                )
                create_info_box("Context Window", HELP_CONTENT["context_window"])

            # --- SIDEBAR COLUMN ---
            with gr.Column(elem_classes=["sticky-column", "sidebar-box"]):
                gr.Markdown("#### System Status", elem_classes=["section-header"])
                
                search_status = gr.Markdown("Ready to search...", elem_classes=["mono-text"])
                
                gpu_status = gr.Textbox(
                    value=get_gpu_status(),
                    show_label=False, 
                    interactive=False,
                    lines=8,
                    elem_classes=["console-output"]
                )

                gr.Markdown("#### Advanced Config", elem_classes=["section-header"])
                with gr.Accordion("LoRA Parameters", open=True):
                    lora_r = gr.Slider(4, 128, value=16, step=4, label="LoRA Rank (r)")
                    create_info_box("LoRA Rank", HELP_CONTENT["lora_rank"])
                    
                    lora_alpha = gr.Slider(8, 256, value=32, step=8, label="LoRA Alpha")
                    create_info_box("LoRA Alpha", HELP_CONTENT["lora_alpha"])

    # --- Event Handlers ---

    def update_recommendation(model):
        # 1. Get Description
        info = MODEL_INFO.get(model, 'Custom model from HuggingFace Hub')
        desc_text = f"💡 *{info}*"
        
        # 2. Check Backend
        backend = get_optimal_backend(model)
        
        perf_html = ""
        is_visible = False
        
        if backend == BackendType.UNSLOTH:
            status_text = "⚡ **Acceleration Logic:** **Unsloth (High Performance)**"
            # Estimate Savings
            perf_html = """
            <div style="display: flex; gap: 20px; margin-top: 10px;">
                <div style="background: rgba(74, 222, 128, 0.1); padding: 10px; border-radius: 8px; border: 1px solid rgba(74, 222, 128, 0.2); width: 100%;">
                    <div style="font-size: 12px; color: #4ade80; font-weight: bold;">📉 VRAM USAGE</div>
                    <div style="font-size: 18px; color: #e2e8f0; font-weight: 800;">-70% SAVED</div>
                    <div style="font-size: 11px; color: #94a3b8;">Fits on T4 / RTX 3060</div>
                </div>
                <div style="background: rgba(99, 102, 241, 0.1); padding: 10px; border-radius: 8px; border: 1px solid rgba(99, 102, 241, 0.2); width: 100%;">
                    <div style="font-size: 12px; color: #818cf8; font-weight: bold;">🚀 TRAINING SPEED</div>
                    <div style="font-size: 18px; color: #e2e8f0; font-weight: 800;">2x - 5x FASTER</div>
                    <div style="font-size: 11px; color: #94a3b8;">Via Triton Kernels</div>
                </div>
            </div>
            """
            is_visible = True
        else:
            status_text = "🐢 **Acceleration Logic:** **Standard (HuggingFace)**"
            is_visible = False
            
        return desc_text, status_text, gr.update(value=perf_html, visible=is_visible)

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
    model_name.change(update_recommendation, [model_name], [model_recommendation, backend_status, performance_stats])
    search_query.submit(perform_search, [search_query], [model_name, search_status])
    
    return model_name, quantization, max_seq_length, lora_r, lora_alpha, gpu_status
