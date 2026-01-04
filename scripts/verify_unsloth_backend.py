
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.getcwd())

from engine.models.backend import get_optimal_backend, BackendType, is_unsloth_available

def test_backend_selection():
    print("Testing Backend Selection Logic...")
    
    # 1. Test Availability Check
    available = is_unsloth_available()
    print(f"   Unsloth Available: {available} (Expected: False on standard Windows setup)")
    
    # 2. Test Model Logic (Assuming Unsloth is FALSE for this test)
    # We patch is_unsloth_available to return True to test the logic
    with patch("engine.models.backend.is_unsloth_available", return_value=True):
        print("\n   [Mock] Simulating Unsloth Installed:")
        
        # Llama -> Unsloth
        backend = get_optimal_backend("meta-llama/Llama-3-8b")
        print(f"   - Llama-3 model -> {backend.value} (Expected: unsloth)")
        if backend != BackendType.UNSLOTH:
            print("❌ Failed Llama check")
            return False

        # Mistral -> Unsloth
        backend = get_optimal_backend("mistralai/Mistral-7B-v0.1")
        print(f"   - Mistral model -> {backend.value} (Expected: unsloth)")
        
        # Whisper -> HF (Not supported by Unsloth yet)
        backend = get_optimal_backend("openai/whisper-large-v3")
        print(f"   - Whisper model -> {backend.value} (Expected: huggingface)")
        if backend != BackendType.HUGGINGFACE:
            print("❌ Failed Whisper check")
            return False

    with patch("engine.models.backend.is_unsloth_available", return_value=False):
        print("\n   [Mock] Simulating Unsloth NOT Installed:")
        backend = get_optimal_backend("meta-llama/Llama-3-8b")
        print(f"   - Llama-3 model -> {backend.value} (Expected: huggingface)")
        if backend != BackendType.HUGGINGFACE:
            print("❌ Failed fallback check")
            return False
            
    return True

def test_trainer_integration():
    print("\nTesting Trainer Integration (Import Safety)...")
    try:
        from engine.training.trainer import Trainer
        print("✅ Trainer imported successfully (no syntax errors or missing imports)")
    except Exception as e:
        print(f"❌ Trainer import failed: {e}")
        return False
    return True

if __name__ == "__main__":
    if test_backend_selection() and test_trainer_integration():
        print("\n✨ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n💀 SOME TESTS FAILED")
        sys.exit(1)
