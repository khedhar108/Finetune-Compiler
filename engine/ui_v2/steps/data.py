"""
Step 2: Data Configuration.
"""

import os
import gradio as gr
from engine.ui_v2.consts import DATA_SOURCES, DATASET_CHOICES, FORMATS, FORMAT_INFO, QUICK_TIPS_CONTENT
from engine.ui_v2.utils import check_hf_token, save_hf_token, analyze_dataset, format_preview_table
from engine.ui_v2.components import CustomDropdown

def step2_data():
    """Step 2: Data Configuration (Redesigned with Cards)."""
    with gr.Column(elem_classes=["premium-card"]):
        gr.Markdown("### 📂 Data & Privacy", elem_classes=["main-header"])
        
        with gr.Row(elem_classes=["cols-2-1"]):
            
            # --- MAIN COLUMN ---
            with gr.Column():
                gr.Markdown("#### Source Configuration", elem_classes=["section-header"])
                
                # Data Source Selection (Horizontal)
                data_source = gr.Radio(
                    choices=DATA_SOURCES,
                    value="huggingface",
                    label="Data Origin",
                    info="Where is your dataset located?",
                    elem_classes=["radio-cards"],
                    container=False, 
                )
                
                gr.Markdown("#### Dataset Details", elem_classes=["section-header"])
                with gr.Group():
                    # 1. Text Input (for HuggingFace IDs or GDrive)
                    data_path = gr.Textbox(
                        value="tatsu-lab/alpaca",
                        label="Source Path / ID",
                        placeholder="e.g., tatsu-lab/alpaca or data/train.csv",
                        info="HuggingFace ID or Local File Path",
                        elem_classes=["gr-input"],
                        visible=True
                    )
                    
                    # 2. File Upload (for CSV/JSON)
                    file_upload = gr.File(
                        label="Upload Local Dataset",
                        file_types=[".csv", ".json", ".jsonl"],
                        file_count="single",
                        visible=False,
                        type="filepath",
                        elem_classes=["premium-card"]
                    )
                    
                    with gr.Row():
                        hf_examples = CustomDropdown(
                            choices=DATASET_CHOICES,
                            label="⚡ Quick Select",
                            info="Pre-validated datasets",
                            interactive=True,
                            visible=True
                        )
                        tips_btn = gr.Button("💡 Quick Tips", size="sm", variant="secondary")
                
                # Dataset Preview Accordion (appears after dataset selection)
                with gr.Accordion("📊 Dataset Preview", open=False, visible=False) as preview_accordion:
                    preview_status = gr.Markdown("*Analyzing dataset...*")
                    preview_table = gr.Markdown("*Loading preview...*")
                    detected_badge = gr.Markdown("**Detected Format:** Analyzing...")
                    analyze_btn = gr.Button("🔄 Refresh Analysis", size="sm", variant="secondary")
                    
                    # --- Package Install Section ---
                    with gr.Accordion("📦 Install Missing Package", open=False):
                        gr.Markdown("If dataset requires a package (e.g., `torchcodec`), install it here:")
                        with gr.Row():
                            pkg_input = gr.Textbox(
                                label="Package Name",
                                placeholder="torchcodec",
                                scale=3
                            )
                            pkg_install_btn = gr.Button("⬇️ Install", size="sm", variant="primary", scale=1)
                        pkg_status = gr.Markdown("")

                # Authentication Card
                gr.Markdown("#### Access Control", elem_classes=["section-header"])
                with gr.Accordion("🔒 Authentication (Private Data)", open=False):
                    gr.Markdown("Required for gated models (Llama 3) or private datasets.", elem_classes=["helper-text"])
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
            
            # --- SIDEBAR COLUMN ---
            with gr.Column(elem_classes=["sticky-column", "sidebar-box"]):
                gr.Markdown("#### data Strategy", elem_classes=["section-header"])
                
                train_split = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Dataset Usage %",
                    info="Start with 50% for faster experiments"
                )
                
                data_format = gr.Radio(
                    choices=FORMATS,
                    value="alpaca",
                    label="Data Format",
                    info="Schema type",
                    elem_classes=["radio-cards"]
                )
                
                gr.Markdown("#### Format Guide", elem_classes=["section-header"])
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
                
                # --- System Status Box ---
                with gr.Accordion("📦 System Status", open=False):
                    # Get current cache path (check saved config first)
                    config_path = os.path.expanduser("~/.cache/huggingface/ftune_config.json")
                    default_cache = os.path.expanduser("~/.cache/huggingface")
                    
                    current_cache = default_cache
                    if os.path.exists(config_path):
                        try:
                            import json
                            with open(config_path, 'r') as f:
                                cfg = json.load(f)
                                current_cache = cfg.get("cache_path", default_cache)
                                # Set env var on load
                                os.environ["HF_HOME"] = current_cache
                        except:
                            pass
                    
                    gr.Markdown("**📂 Dataset Cache Location:**")
                    gr.Markdown("*Choose a drive with enough space (datasets can be 1-50GB)*", elem_classes=["helper-text"])
                    
                    with gr.Row():
                        cache_path_input = gr.Textbox(
                            value=current_cache,
                            label="Cache Folder",
                            placeholder="D:\\HuggingFaceCache",
                            scale=3
                        )
                        cache_browse_btn = gr.Button("📁 Browse", size="sm", scale=1)
                        cache_save_btn = gr.Button("💾 Save", size="sm", variant="primary", scale=1)
                    
                    cache_status = gr.Markdown("")
                    
                    gr.Markdown("---")
                    
                    system_status_md = gr.Markdown(f"""
**🔑 Token (Read):**
{check_hf_token()}

**📤 Token (Write):**
*Same as read token - used for model uploads*

**💡 Tip:** Datasets are cached locally. Token auto-persisted to disk.
                    """)
        
        # --- Quick Tips Drawer (Fixed Position) ---
        with gr.Column(visible=False, elem_classes=["drawer-content"], elem_id="tips-drawer") as tips_drawer:
            with gr.Row(elem_classes=["drawer-header"]):
                gr.Markdown("## 💡 Quick Tips")
                close_drawer_btn = gr.Button("✕", elem_classes=["drawer-close-btn"], size="sm")
            
            # Use enhanced tips content from consts
            gr.Markdown(QUICK_TIPS_CONTENT)
        


        def update_path_from_picker(example):
            return example if example else "tatsu-lab/alpaca"
            
        def update_path_from_upload(file):
            if file is not None:
                return file.name  # Absolute path of uploaded file
            return gr.update()
        
        def install_package(pkg_name):
            """Install a package by spawning a separate PowerShell window.
            
            Why: The current Python process locks .venv files. Running uv add 
            in the same process fails on Windows. We spawn a new PowerShell 
            that runs independently.
            """
            import subprocess
            import sys
            
            if not pkg_name or not pkg_name.strip():
                return "❌ Please enter a package name."
            
            pkg_name = pkg_name.strip()
            
            # Security: Only allow alphanumeric, hyphen, underscore, brackets
            import re
            if not re.match(r'^[a-zA-Z0-9_\-\[\]]+$', pkg_name):
                return "❌ Invalid package name format."
            
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            
            # Windows: Suggest manual installation to avoid locking issues
            if sys.platform == "win32":
                return f"""### ⚠️ Windows: Action Required
                
To ensure safe installation without file locking errors, please follows these steps:

1. **Stop the server**: Press `Ctrl+C` in your terminal (or run `uv run kill-ftune`).
2. **Install the package**:
   ```bash
   uv add {pkg_name}
   ```
   *(This ensures it installs ONLY in the virtual environment)*

3. **Restart the server**:
   ```bash
   uv run ftune
   ```
"""

            # Linux/Mac: Run in background (safer on POSIX)
            try:
                subprocess.Popen(["uv", "add", pkg_name], cwd=project_root)
                return f"""## ✅ Installation Started
                
Running: `uv add {pkg_name}`

> Please check your terminal for progress. You may need to restart the server manually if modules are not found.
"""
            except Exception as e:
                return f"❌ Error starting installation: {e}"
        
        def show_loading():
            """Show loading state immediately when analysis starts."""
            return (
                gr.update(visible=True, open=True),  # Show and open accordion
                "⏳ **Loading dataset from HuggingFace...**\n\nThis may take a few seconds for large datasets.",
                "*Fetching preview data...*",
                "**Detecting format...**",
                gr.update()  # No format change yet
            )
        
        def analyze_and_preview(dataset_id, data_src):
            """Analyze dataset and return preview + detected format."""
            if data_src != "huggingface" or not dataset_id:
                return (
                    gr.update(visible=False),  # preview_accordion
                    "*Select a HuggingFace dataset to preview*",  # preview_status
                    "",  # preview_table
                    "",  # detected_badge
                    gr.update()  # data_format (no change)
                )
            
            try:
                result = analyze_dataset(dataset_id)
                
                if result["error"]:
                    error_msg = result["error"]
                    
                    # Detect authentication errors specifically
                    auth_keywords = ["401", "gated", "authentication", "login", "private", "access denied", "unauthorized"]
                    is_auth_error = any(kw in error_msg.lower() for kw in auth_keywords)
                    
                    if is_auth_error:
                        # Build dataset-specific URL for accepting conditions
                        dataset_url = f"https://huggingface.co/datasets/{dataset_id}"
                        return (
                            gr.update(visible=True),
                            "🔒 **Gated Dataset - Access Required**",
                            f"""
This dataset requires you to **accept conditions** on HuggingFace before access is granted.

**Step 1: Accept Conditions**
1. Visit: [{dataset_id}]({dataset_url})
2. Click **"Agree and access repository"** button
3. Fill in required fields (name, email, purpose)
4. Submit and wait for approval (usually instant)

**Step 2: Add Token (if not already done)**
1. Scroll up to **Access Control** section
2. Enter your HuggingFace token (`hf_...`)
3. Click **Validate**

**Step 3: Retry**
1. Click **🔄 Refresh Analysis** below

---
**Get a token:** [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
                            """,
                            f"**Status:** 🔐 Go to [{dataset_id}]({dataset_url}) and click 'Agree'",
                            gr.update()
                        )
                    else:
                        return (
                            gr.update(visible=True),
                            f"⚠️ **Error loading dataset**",
                            f"```\n{error_msg}\n```\n\n**Possible causes:**\n- Dataset ID is incorrect\n- Network connection issue\n- Dataset structure not supported",
                            "",
                            gr.update()
                        )
                
                # Build preview table
                preview_md = format_preview_table(result["sample_rows"])
                
                # Build detection badge with modality info
                modalities = result["modalities"]
                modality_str = ", ".join(modalities)
                fmt = result["suggested_format"]
                fmt_info = FORMAT_INFO.get(fmt, {})
                
                # Get split info
                used_split = result.get("used_split", "train")
                available_splits = result.get("available_splits", ["train"])
                splits_str = ", ".join(available_splits)
                
                # Clarify audio format includes transcription
                if fmt == "audio" and len(modalities) > 1:
                    badge = f"**Detected:** {fmt_info.get('icon', '')} **{fmt_info.get('name', fmt)}** (Audio + Transcription) | Modalities: `{modality_str}` | Split: `{used_split}`"
                else:
                    badge = f"**Detected:** {fmt_info.get('icon', '')} **{fmt_info.get('name', fmt)}** | Modalities: `{modality_str}` | Split: `{used_split}`"
                
                # Status message with split info
                cols = ", ".join(result["columns"][:5])
                status = f"✅ **Columns found:** `{cols}` ({len(result['columns'])} total)\n\n📂 **Available splits:** `{splits_str}` (using: `{used_split}`)"
                
                return (
                    gr.update(visible=True),  # Show preview accordion
                    status,  # preview_status
                    preview_md,  # preview_table
                    badge,  # detected_badge
                    gr.update(value=fmt)  # Auto-select format
                )
            except Exception as e:
                return (
                    gr.update(visible=True),
                    f"❌ **Unexpected Error**",
                    f"```\n{str(e)}\n```",
                    "",
                    gr.update()
                )

        def toggle_inputs(source):
            if source == "huggingface":
                return {
                    data_path: gr.update(visible=True, label="HuggingFace Repo ID", value="tatsu-lab/alpaca"),
                    file_upload: gr.update(visible=False),
                    hf_examples: gr.update(visible=True)
                }
            elif source in ["csv", "json"]:
                return {
                    data_path: gr.update(visible=True, label="Local File Path", value="", placeholder="Upload a file below or type path"),
                    file_upload: gr.update(visible=True),
                    hf_examples: gr.update(visible=False)
                }
            else: # gdrive
                return {
                    data_path: gr.update(visible=True, label="Google Drive Folder ID", value=""),
                    file_upload: gr.update(visible=False),
                    hf_examples: gr.update(visible=False)
                }
        
        # Logic Wiring
        data_source.change(toggle_inputs, [data_source], [data_path, file_upload, hf_examples])
        file_upload.upload(update_path_from_upload, [file_upload], [data_path])
        
        # Token save with status refresh
        def save_and_refresh_status(token):
            """Save token and return both the save status and updated system status."""
            from engine.ui_v2.utils import save_hf_token, check_hf_token
            save_result = save_hf_token(token)
            updated_status = f"""
**🔑 Token (Read):**
{check_hf_token()}

**📤 Token (Write):**
*Same as read token - used for model uploads*

**💡 Tip:** Datasets are cached locally. Token auto-persisted to disk.
            """
            return save_result, updated_status
        
        def save_cache_location(path):
            """Save cache location to config and set HF_HOME."""
            import json
            
            if not path or not path.strip():
                return "❌ Please enter a folder path."
            
            path = path.strip()
            
            # Validate path exists or can be created
            try:
                os.makedirs(path, exist_ok=True)
                os.makedirs(os.path.join(path, "datasets"), exist_ok=True)
            except Exception as e:
                return f"❌ Cannot create folder: {e}"
            
            # Set environment variable for this session
            os.environ["HF_HOME"] = path
            os.environ["HF_DATASETS_CACHE"] = os.path.join(path, "datasets")
            
            # Persist to config file
            try:
                config_path = os.path.expanduser("~/.cache/huggingface/ftune_config.json")
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                
                config = {}
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                
                config["cache_path"] = path
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                return f"""✅ **Cache location saved!**

📂 New path: `{path}`

> ⚠️ **Restart FTune** for changes to fully take effect.
"""
            except Exception as e:
                return f"⚠️ Set for this session only (save failed: {e})"
        
        def browse_cache_folder():
            """Open a folder browser dialog and return selected path."""
            import subprocess
            import sys
            
            if sys.platform == "win32":
                # Use PowerShell to open folder browser
                ps_script = '''
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = "Select Cache Folder for HuggingFace Datasets"
$dialog.ShowNewFolderButton = $true
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    Write-Output $dialog.SelectedPath
}
'''
                try:
                    result = subprocess.run(
                        ["powershell", "-Command", ps_script],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    selected = result.stdout.strip()
                    if selected:
                        return selected
                except:
                    pass
            
            return gr.update()  # No change if failed
        
        hf_token_btn.click(
            fn=save_and_refresh_status,
            inputs=[hf_token_input],
            outputs=[hf_token_status, system_status_md]
        )
        
        # Cache location handlers
        cache_save_btn.click(
            fn=save_cache_location,
            inputs=[cache_path_input],
            outputs=[cache_status]
        )
        
        cache_browse_btn.click(
            fn=browse_cache_folder,
            inputs=[],
            outputs=[cache_path_input]
        )
        
        # Auto-analyze when dataset path is submitted (Enter key)
        # First show loading, then analyze
        data_path.submit(
            fn=show_loading,
            inputs=None,
            outputs=[preview_accordion, preview_status, preview_table, detected_badge, data_format]
        ).then(
            fn=analyze_and_preview,
            inputs=[data_path, data_source],
            outputs=[preview_accordion, preview_status, preview_table, detected_badge, data_format]
        )
        
        # Chain: Quick Select → Update Path → Show Loading → Analyze
        # This ensures user sees immediate feedback while dataset loads
        hf_examples.change(
            fn=update_path_from_picker,
            inputs=[hf_examples],
            outputs=[data_path]
        ).then(
            fn=show_loading,
            inputs=None,
            outputs=[preview_accordion, preview_status, preview_table, detected_badge, data_format]
        ).then(
            fn=analyze_and_preview,
            inputs=[data_path, data_source],
            outputs=[preview_accordion, preview_status, preview_table, detected_badge, data_format]
        )
        
        # Manual refresh button
        analyze_btn.click(
            fn=analyze_and_preview,
            inputs=[data_path, data_source],
            outputs=[preview_accordion, preview_status, preview_table, detected_badge, data_format]
        )
        
        # Package install button
        pkg_install_btn.click(
            fn=install_package,
            inputs=[pkg_input],
            outputs=[pkg_status]
        )
        
        # Drawer Events with JS Force (Robust Fix)
        fn_open = "() => { const el = document.getElementById('tips-drawer'); if(el) { el.style.display = 'flex'; el.classList.remove('hidden'); } }"
        fn_close = "() => { const el = document.getElementById('tips-drawer'); if(el) { el.style.display = 'none'; el.classList.add('hidden'); } }"
        
        tips_btn.click(
            fn=lambda: gr.update(visible=True), 
            inputs=None, 
            outputs=tips_drawer,
            js=fn_open
        )
        
        close_drawer_btn.click(
            fn=lambda: gr.update(visible=False), 
            inputs=None, 
            outputs=tips_drawer,
            js=fn_close
        )
        
    return data_source, data_path, data_format, train_split
