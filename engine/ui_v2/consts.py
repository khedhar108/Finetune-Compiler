"""
Constants for AI Compiler UI v2.
"""

# Models organized by category with recommendations
MODELS_BY_CATEGORY = {
    "🚀 Quick Start (Recommended)": [
        ("unsloth/Llama-3.2-1B", "Best for beginners, fast training"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Very small, quick experiments"),
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

# Datasets with descriptions
HUGGINGFACE_DATASETS = {
    "📚 General Q&A": [
        ("tatsu-lab/alpaca", "52K instruction-following examples"),
        ("databricks/dolly-15k", "15K high-quality examples"),
    ],
    "🏥 Medical": [
        ("medalpaca/medical_meadow_medical_flashcards", "Medical flashcards"),
        ("medmcqa", "Medical MCQ dataset"),
    ],
    "💻 Code": [
        ("sahil2801/CodeAlpaca-20k", "20K code instructions"),
        ("iamtarun/python_code_instructions_18k_alpaca", "Python code"),
    ],
    "📞 Customer Support": [
        ("bitext/Bitext-customer-support-llm-chatbot-training-dataset", "Customer support"),
    ],
}

# Flatten datasets for dropdown
DATASET_CHOICES = []
for category, datasets in HUGGINGFACE_DATASETS.items():
    for ds_name, desc in datasets:
        DATASET_CHOICES.append(f"{ds_name}")

UI_CSS = """
/* Global Fonts & Reset */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');

body, .gradio-container { 
    background-color: #0b0f19; 
    color: #e2e8f0; 
    font-family: 'Inter', sans-serif;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #1e293b; 
}
::-webkit-scrollbar-thumb {
    background: #475569; 
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #64748b; 
}

/* Premium Card Styling */
.premium-card {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 16px;
    padding: 24px !important;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

/* Header Gradient */
.main-header h1 {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #a78bfa 0%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}

/* Inputs & Dropdowns */
input, textarea, select, .gr-input {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: white !important;
    transition: all 0.2s;
}
input:focus, textarea:focus, select:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.2) !important;
}

/* Buttons */
.primary-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
    transition: all 0.2s ease;
}
.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(124, 58, 237, 0.4);
}

/* Match button height to textbox */
button.secondary {
    min-height: 42px !important;
    height: 42px !important;
}

/* Sticky Right Column */
.sticky-column {
    position: sticky;
    top: 20px;
    height: fit-content;
}

/* Radio Cards (Step 2) */
.radio-cards .wrap {
    display: grid !important;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}
.radio-cards label {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    transition: all 0.2s;
}
.radio-cards label.selected {
    border-color: #818cf8 !important;
    background: #312e81 !important;
}

/* Badge Styles */
.badge-live { background: rgba(74, 222, 128, 0.1); color: #4ade80; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; border: 1px solid rgba(74, 222, 128, 0.2); }
.badge-building { background: rgba(251, 191, 36, 0.1); color: #fbbf24; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; border: 1px solid rgba(251, 191, 36, 0.2); }
.badge-error { background: rgba(248, 113, 113, 0.1); color: #f87171; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; border: 1px solid rgba(248, 113, 113, 0.2); }
.badge-idle { background: rgba(148, 163, 184, 0.1); color: #94a3b8; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }

/* Utility */
.mono-text { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #cbd5e1; }
.helper-text { font-size: 12px; color: #94a3b8; margin-top: 4px; }

/* Section Headers */
.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    color: #94a3b8;
    margin-bottom: 8px !important;
    margin-top: 16px !important;
}
.section-header:first-child {
    margin-top: 0 !important;
}

/* Info Display Box */
.info-display textarea {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 8px !important;
    padding: 12px !important;
    font-size: 13px !important;
    color: #e2e8f0 !important;
}

/* Console Output */
.console-output textarea {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: #f87171 !important;
}

/* Status Animations */
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(74, 222, 128, 0); }
    100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
}

/* Layout Stability */
.status-container {
    min-height: 28px;
    margin-top: 8px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
}

/* Fix Dropdown Spacing */
#model-dropdown {
    margin-top: 12px !important;
}

/* Boxed Layout Design */
.content-box {
    background: rgba(30, 41, 59, 0.4) !important;
    border: 1px solid rgba(148, 163, 184, 0.1) !important;
    border-radius: 12px !important;
    padding: 20px 24px !important;
    margin-bottom: 16px !important;
}

/* Ensure inner content has breathing room */
.content-box > div {
    padding: 0 !important;
}
.content-box .prose {
    padding: 8px 0 !important;
}
.content-box p {
    margin: 8px 0 !important;
    line-height: 1.5;
}

/* Fix Dropdown Overlay Issue */
.content-box .wrap {
    position: relative;
    z-index: 100;
}
.content-box ul[role="listbox"],
div[data-testid="dropdown"] ul {
    z-index: 9999 !important;
    position: absolute !important;
}

.sidebar-box {
    background: rgba(30, 41, 59, 0.6);
    border-left: 1px solid rgba(148, 163, 184, 0.1);
    height: 100%;
    padding: 20px !important;
    border-radius: 0 12px 12px 0;
}
"""

