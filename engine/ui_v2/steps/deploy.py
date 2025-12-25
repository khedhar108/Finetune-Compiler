"""
Step 4: Deployment Control Center.
"""

import gradio as gr
from engine.utils.huggingface import set_read_token, set_write_token

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
        if set_read_token(token): return "✅ Saved"
        return "❌ Invalid"
    
    def save_write_token(token):
        if set_write_token(token): return "✅ Saved"
        return "❌ Invalid"
    
    read_token_btn.click(save_read_token, [read_token_input], [read_token_status])
    write_token_btn.click(save_write_token, [write_token_input], [write_token_status])
    
    return (deploy_status_banner, model_path, hf_repo_name, private_repo, hf_deploy_btn, hf_deploy_status, hf_model_url,
            export_format, export_btn, export_status,
            load_model_id, load_btn, test_prompt, test_output, test_btn)
