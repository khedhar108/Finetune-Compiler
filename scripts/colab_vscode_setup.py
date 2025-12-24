"""
🚀 AI Compiler - VS Code + Colab T4 Setup
==========================================

This script is designed for running the AI Compiler Wizard UI on Colab's T4 GPU
when connected via the VS Code Colab extension.

HOW TO USE:
-----------
1. Connect VS Code to Colab (Ctrl+Shift+P → "Colab: Connect to Colab Runtime" → T4)
2. Upload your ai-compiler folder to Colab
3. Run this script: python scripts/colab_vscode_setup.py

OR copy each section into separate Colab cells.
"""

# ============================================================
# CELL 1: Environment Check
# ============================================================
print("=" * 60)
print("🔍 CHECKING ENVIRONMENT")
print("=" * 60)

import os
import sys
import subprocess

# Check if in Colab
IN_COLAB = 'google.colab' in sys.modules
IS_LINUX = sys.platform.startswith('linux')

# Check GPU
try:
    gpu_info = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader']
    ).decode().strip()
    HAS_GPU = True
    GPU_NAME = gpu_info.split(',')[0].strip()
    GPU_MEMORY = gpu_info.split(',')[1].strip()
except:
    HAS_GPU = False
    GPU_NAME = "None"
    GPU_MEMORY = "0 MiB"

print(f"📍 In Colab: {IN_COLAB}")
print(f"🐧 Linux: {IS_LINUX}")
print(f"🎮 GPU: {GPU_NAME}")
print(f"💾 GPU Memory: {GPU_MEMORY}")
print(f"🚀 Can use Unsloth: {IS_LINUX and HAS_GPU}")
print("=" * 60)

if HAS_GPU and "T4" in GPU_NAME:
    print("\n✅ Tesla T4 detected! Ready for fast training with Unsloth!")
elif HAS_GPU:
    print(f"\n✅ {GPU_NAME} detected! Ready for training!")
else:
    print("\n❌ No GPU detected. Please enable GPU in Colab runtime.")
    print("   Go to: Runtime → Change runtime type → GPU")

# ============================================================
# CELL 2: Install Dependencies
# ============================================================
print("\n" + "=" * 60)
print("📦 INSTALLING DEPENDENCIES")
print("=" * 60)

# Install UV package manager
os.system("pip install uv -q")

# Get current directory
cwd = os.getcwd()
print(f"📂 Current directory: {cwd}")

# Check if we're in ai-compiler folder
if os.path.exists("pyproject.toml"):
    print("✅ Found pyproject.toml - in ai-compiler folder")
elif os.path.exists("ai-compiler/pyproject.toml"):
    os.chdir("ai-compiler")
    print("✅ Changed to ai-compiler folder")
else:
    print("⚠️ Not in ai-compiler folder. Please upload or navigate to it.")
    print("   Expected structure: /content/ai-compiler/")

# Install with UI extra
print("\n📦 Installing AI Compiler with UI dependencies...")
if IS_LINUX and HAS_GPU:
    os.system("uv sync --extra ui --extra unsloth")
else:
    os.system("uv sync --extra ui")

print("\n✅ Installation complete!")

# ============================================================
# CELL 3: Verify Installation
# ============================================================
print("\n" + "=" * 60)
print("✅ VERIFYING INSTALLATION")
print("=" * 60)

os.system("uv run ai-compile info")

# ============================================================
# CELL 4: Launch Wizard UI
# ============================================================
print("\n" + "=" * 60)
print("🚀 LAUNCHING WIZARD UI")
print("=" * 60)
print("🔗 A public URL will appear below - click it to open the UI!")
print("   The UI will run on the T4 GPU for fast training.")
print("=" * 60 + "\n")

# Run UI with share link (required for Colab)
os.system("uv run ai-compile ui2 --share")
