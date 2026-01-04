
import sys
import os
from unittest.mock import MagicMock

sys.path.append(os.getcwd())
sys.modules["gradio"] = MagicMock()

def test_features():
    print("Testing Feature Integration...")
    
    # 1. Verify Dataset Choices Tuple Structure
    print("1. Checking DATASET_CHOICES...")
    try:
        from engine.ui_v2.consts import DATASET_CHOICES
        if not DATASET_CHOICES:
            print("❌ DATASET_CHOICES is empty")
            return False
            
        first_item = DATASET_CHOICES[0]
        if not isinstance(first_item, tuple) or len(first_item) != 2:
            print(f"❌ DATASET_CHOICES item format incorrect. Expected tuple (label, value), got: {type(first_item)}")
            return False
        
        print(f"   ✅ Tuple format confirmed: {first_item[0][:30]}... -> {first_item[1]}")
        
        # Check for specific medical datasets
        found_med = any("ekacare" in x[1] for x in DATASET_CHOICES)
        if not found_med:
            print("❌ EkaCare dataset not found in choices")
            return False
        print("   ✅ Medical datasets present.")
        
    except Exception as e:
        print(f"❌ Error importing consts: {e}")
        return False

    # 2. Verify Config Handling
    print("\n2. Checking Config & Build Logic...")
    try:
        from engine.ui_v2.utils import build_config
        from engine.utils.config import Config
        
        # Mock inputs
        conf_dict = build_config(
            model_name="test", quantization="4bit", max_seq_length=1024,
            lora_r=16, lora_alpha=32,
            data_source="huggingface", data_path="test/path", data_format="alpaca",
            epochs=1, batch_size=1, learning_rate=2e-4,
            output_dir="./out", train_split=50 # 50% slider value
        )
        
        if conf_dict["data"]["train_ratio"] != 0.5:
             print(f"❌ build_config failed to convert train_split 50 -> 0.5. Got: {conf_dict['data'].get('train_ratio')}")
             return False
        print("   ✅ build_config correctly sets train_ratio=0.5")
        
        # Validate against Pydantic model
        cfg = Config.model_validate(conf_dict)
        if cfg.data.train_ratio != 0.5:
            print("❌ Config model validation failed to retain train_ratio")
            return False
        print("   ✅ Config model validation passed.")

    except Exception as e:
        print(f"❌ Error in config test: {e}")
        return False

    print("\n✨ ALL TESTS PASSED")
    return True

if __name__ == "__main__":
    if test_features():
        sys.exit(0)
    else:
        sys.exit(1)
