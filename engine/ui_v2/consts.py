"""
Constants for AI Compiler UI v2.
"""

# Help/Info Content for UI Toggles
HELP_CONTENT = {
    "quantization": """
**What is it?**
Reduces the precision of model weights to save memory.

**Why change it?**
- **4bit (Recommended):** Lowest memory usage. Allows running 7B models on free Colab T4 GPUs.
- **8bit:** Higher memory usage, slightly better accuracy.
- **None:** Full precision. Requires massive 80GB+ GPUs (A100).
""",
    "context_window": """
**What is it?**
The amount of text (tokens) the model can "see" at once.

**How to choose:**
- **2048 (Standard):** Good for most chat/instruction tasks.
- **1024:** Saves VRAM if you are running out of memory.
- **4096+:** Needed for summarizing long documents (requires more memory).
""",
    "lora_rank": """
**What is it?**
The "capacity" or "intelligence" of the fine-tuning adapter.

**Recommended Values:**
- **16 (Standard):** Best balance of quality and speed.
- **8:** Faster, good for simple pattern matching.
- **64:** For very complex logic (requires more VRAM).
""",
    "lora_alpha": """
**What is it?**
A scaling factor that controls how strongly the new training overrides the base model.

**Rule of Thumb:**
Set Alpha to **2x the Rank**.
- If Rank = 16, set Alpha = 32.
- If Rank = 64, set Alpha = 128.
"""
}

# Models organized by category with recommendations
MODELS_BY_CATEGORY = {
    "🚀 Quick Start (Recommended)": [
        ("unsloth/Llama-3.2-1B-bnb-4bit", "Best for beginners, 4-bit fast training"),
        ("unsloth/llama-3-8b-bnb-4bit", "Powerful 8B model, 4-bit optimized"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Very small, quick experiments"),
    ],
    "⚡ High Performance (Unsloth Optimized)": [
        ("unsloth/mistral-7b-v0.3-bnb-4bit", "Mistral v0.3 4-bit, 2x faster"),
        ("unsloth/gemma-7b-bnb-4bit", "Google Gemma 7B, 4-bit"),
        ("unsloth/yi-6b-bnb-4bit", "Yi 6B, very fast"),
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
        ("google/medasr", "Specialized for medical dictation"),
    ],
    "🔊 Text-to-Speech (TTS)": [
        ("microsoft/speecht5_tts", "General purpose TTS"),
        ("suno/bark", "Expressive, multi-speaker TTS"),
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

# Format metadata for auto-detection and UI display
FORMAT_INFO = {
    "alpaca": {
        "icon": "📝",
        "name": "Alpaca (Instruction-Response)",
        "desc": "Standard instruction-following format with instruction, input, and output fields.",
        "columns": ["instruction", "output"],
        "optional_columns": ["input"],
        "example": '{"instruction": "Summarize this", "input": "Long text...", "output": "Summary..."}',
        "use_case": "Q&A, summarization, instruction-following tasks"
    },
    "chatml": {
        "icon": "💬",
        "name": "ChatML (Conversation)",
        "desc": "Multi-turn conversation format with role-based messages.",
        "columns": ["messages"],
        "optional_columns": [],
        "example": '{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}',
        "use_case": "Chatbots, multi-turn dialogues"
    },
    "completion": {
        "icon": "📄",
        "name": "Completion (Plain Text)",
        "desc": "Simple text generation without structured fields.",
        "columns": ["text"],
        "optional_columns": [],
        "example": '{"text": "Once upon a time..."}',
        "use_case": "Text generation, story writing, plain text tasks"
    },
    "audio": {
        "icon": "🎤",
        "name": "Audio-Text (ASR)",
        "desc": "Speech-to-text: Audio files paired with transcriptions. Used for training speech recognition models.",
        "columns": ["audio", "transcription"],
        "optional_columns": ["sentence", "text", "transcript"],
        "example": '{"audio": <audio_bytes>, "transcription": "Hello world"}',
        "use_case": "Speech recognition, medical dictation, voice assistants. The 'audio' format automatically handles BOTH audio and text columns."
    },
    "image": {
        "icon": "🖼️",
        "name": "Image-Text (VQA/Caption)",
        "desc": "Image with text caption or question-answer pairs.",
        "columns": ["image", "caption"],
        "optional_columns": ["question", "answer"],
        "example": '{"image": <image_bytes>, "caption": "A cat sitting on a couch"}',
        "use_case": "Image captioning, visual Q&A"
    },
}

# Quick Tips Content (Enhanced for beginners)
QUICK_TIPS_CONTENT = """
### 🎯 Getting Started

**Step 1: Choose your data source**
- **HuggingFace** → Best for beginners. Thousands of ready-to-use datasets.
- **CSV/JSON** → Use your own local data files.
- **Google Drive** → For large datasets stored in cloud.

---

### 📊 Understanding Data Formats

| Format | When to Use | Example |
|--------|-------------|---------|
| 📝 **Alpaca** | Q&A, instructions | `{"instruction": "...", "output": "..."}` |
| 💬 **ChatML** | Chatbots, dialogues | `{"messages": [{role, content}]}` |
| 📄 **Completion** | Plain text generation | `{"text": "..."}` |
| 🎤 **Audio** | Speech recognition | `{"audio": ..., "transcription": "..."}` |

---

### 🔍 Auto-Detection

When you select a dataset, the system will:
1. **Analyze** the column structure
2. **Detect** the format automatically
3. **Show preview** of first 3 rows

You can always override the auto-detected format if needed.

---

### ⚠️ Common Issues

| Problem | Solution |
|---------|----------|
| **Private dataset** | Add your HuggingFace token in Authentication |
| **Slow loading** | Reduce "Dataset Usage %" for faster training |
| **Wrong format** | Manually select the correct format |

---

### 💡 Pro Tips

1. **Start small**: Use 10-20% of data for initial experiments
2. **Check preview**: Always verify the data looks correct
3. **Audio datasets**: Ensure `transcription` or `sentence` column exists
"""


# Datasets with descriptions
HUGGINGFACE_DATASETS = {
    "🏥 Medical / Clinical ASR (Audio + Text)": [
        ("ekacare/eka-medical-asr-evaluation-dataset", "Real Indian medical speech (English/Hindi)"),
        ("united-we-care/United-Syn-Med", "Synthetic clinical speech (100k+ rows)"),
        ("google/fleurs", "Multi-language (Medical subsets available via slicing)"),
    ],
    "🇮🇳 Indian Languages (ASR / TTS)": [
        ("ai4bharat/IndicVoices", "Natural speech, 22 Indian languages"),
        ("mozilla-foundation/common_voice_11_0", "Hindi subset available"),
    ],
    "💊 Medical Lexicons (Text Only)": [
        ("drowsyng/medicines-dataset", "Indian e-pharmacy catalogue (Kaggle)"),
        ("united-we-care/United-MedSyn", "Medical terminology & dialogs"),
    ],
    "📚 General Q&A": [
        ("tatsu-lab/alpaca", "52K instruction-following examples"),
        ("databricks/dolly-15k", "15K high-quality examples"),
    ],
    "💻 Code": [
        ("sahil2801/CodeAlpaca-20k", "20K code instructions"),
        ("iamtarun/python_code_instructions_18k_alpaca", "Python code"),
    ],
    "📞 Customer Support": [
        ("bitext/Bitext-customer-support-llm-chatbot-training-dataset", "Customer support"),
    ],
}

# Flatten datasets for dropdown (Label, Value)
DATASET_CHOICES = []
for category, datasets in HUGGINGFACE_DATASETS.items():
    for ds_name, desc in datasets:
        # Format: "Category: Dataset (Description)"
        label = f"{category}: {ds_name} | {desc}"
        value = ds_name
        DATASET_CHOICES.append((label, value))

UI_CSS = """
/* ==========================================================================
   1. GLOBAL RESET & TYPOGRAPHY (DENSITY RELAXATION)
   ========================================================================== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

* { box-sizing: border-box; }

body, .gradio-container { 
    background-color: #0b0f19; 
    color: #e2e8f0; 
    font-family: 'Inter', sans-serif;
    font-size: 16px !important; /* INCREASED from 14px */
    line-height: 1.6 !important;
    margin: 0; padding: 0;
}

/* ==========================================================================
   2. LAYOUT UTILITIES
   ========================================================================== */
.cols-2-1 {
    display: grid !important;
    grid-template-columns: 2fr 1fr !important;
    gap: 32px !important; /* Increased gap */
    align-items: start !important;
}

@media (max-width: 768px) {
    .cols-2-1 { grid-template-columns: 1fr !important; }
}

.sticky-column {
    position: sticky !important;
    top: 32px !important;
    height: fit-content !important;
}

/* ==========================================================================
   3. COMPONENTS: CARDS
   ========================================================================== */
.premium-card {
    background: rgba(30, 41, 59, 0.4) !important;
    border: 1px solid rgba(148, 163, 184, 0.1) !important;
    border-radius: 16px !important; /* Softer corners */
    padding: 32px !important; /* More breathing room */
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 8px 32px -4px rgba(0, 0, 0, 0.2) !important;
    margin-bottom: 24px !important;
}

.sidebar-box {
    background: rgba(30, 41, 59, 0.3) !important;
    border: 1px solid rgba(148, 163, 184, 0.05) !important;
    border-radius: 16px !important;
    padding: 24px !important;
}

/* ==========================================================================
   4. INPUTS & DROPDOWNS (AVANT-GARDE STYLE)
   ========================================================================== */

/* General Logic for ALL Inputs to relax density */
input, textarea, select, .gr-input, .gr-box, .gr-check-radio, .custom-dropdown .wrap {
    font-size: 16px !important;
    padding: 12px 16px !important; /* Larger touch targets */
    border-radius: 8px !important;
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    color: white !important;
    transition: all 0.2s ease !important;
}

/* Force Height on Text Inputs & Dropdown containers */
.gr-input label input, .custom-dropdown .wrap, .custom-dropdown .wrap input {
    min-height: 48px !important;
    display: flex !important;
    align-items: center !important;
}

input:focus, textarea:focus, .custom-dropdown .wrap:focus-within {
    border-color: #818cf8 !important; /* Indigo-400 */
    box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.15) !important;
    background-color: #252f45 !important;
}

/* Custom Dropdown Specifics */
.custom-dropdown {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}

.custom-dropdown .wrap {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    cursor: pointer !important;
    width: 100% !important;
}

.custom-dropdown .wrap input {
    width: 100% !important;
    flex: 1 !important;
}

.custom-dropdown svg { /* Chevron */
    fill: #94a3b8 !important; 
    width: 20px !important; 
    height: 20px !important;
}

/* The Options List (Popup) */
ul.options, ul[role="listbox"] {
    background-color: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    box-shadow: 0 20px 40px rgba(0,0,0,0.6) !important;
    z-index: 5000 !important;
    padding: 8px !important;
}

li.item {
    padding: 12px 16px !important; /* Relaxed list items */
    font-size: 15px !important;
    border-radius: 6px !important;
    margin-bottom: 2px !important;
    cursor: pointer !important;
    color: #cbd5e1 !important;
}

li.item:hover, li.item.selected {
    background: linear-gradient(90deg, rgba(99, 102, 241, 0.15) 0%, transparent 100%) !important;
    color: #fff !important;
}

/* ==========================================================================
   5. BUTTONS & INTERACTIVE
   ========================================================================== */
.primary-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 12px 24px !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
}
.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5) !important;
}

/* ==========================================================================
   6. DRAWER ANIMATION (Slide-in)
   ========================================================================== */
/* Animation Keyframes */
@keyframes slideInRight {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

.drawer-content {
    position: fixed !important;
    top: 0 !important;
    right: 0 !important;
    height: 100vh !important;
    width: 450px !important;
    max-width: 90vw !important;
    
    background: rgba(10, 10, 20, 0.95) !important;
    backdrop-filter: blur(24px) saturate(180%) !important;
    border-left: 1px solid rgba(255, 255, 255, 0.08) !important;
    
    padding: 32px !important;
    z-index: 2147483647 !important;
    box-shadow: -30px 0 80px rgba(0,0,0,0.8) !important;
    
    display: flex; /* REMOVED !important so Gradio can toggle display:none */
    flex-direction: column !important;
    
    animation: slideInRight 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
}

/* Force hide when Gradio applies hidden class */
.drawer-content.hidden, .drawer-content.hide, .drawer-content[style*="display: none"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    transform: translateX(100%) !important;
    pointer-events: none !important;
}

.drawer-header {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    padding-bottom: 24px !important;
    margin-bottom: 24px !important;
    border-bottom: 1px solid rgba(255,255,255,0.1) !important;
}
.drawer-header h2 { font-size: 1.5rem; font-weight: 700; color: #fff; margin: 0; }

.drawer-close-btn {
    background: transparent !important;
    border: none !important;
    color: #94a3b8 !important;
    font-size: 1.5rem !important;
    padding: 8px !important;
}

/* ==========================================================================
   7. UTILITIES
   ========================================================================== */
.main-header h1 {
    font-size: 3rem !important; /* Larger Title */
    margin-bottom: 16px !important;
    letter-spacing: -1px !important;
}
.section-header p, .helper-text { font-size: 15px !important; color: #94a3b8 !important; }
.badge-live { background: rgba(74, 222, 128, 0.15); color: #4ade80; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; letter-spacing: 0.5px; text-transform: uppercase; }

/* Custom Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #475569; border-radius: 3px; }
"""
