
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

def test_alpaca_format():
    print("Testing alpaca_format with extra args...")
    try:
        from engine.data.formats import alpaca_format
        example = {"instruction": "Hello", "input": "", "output": "World"}
        # This calls with extra 'text_col' argument which was causing the crash
        result = alpaca_format(example, text_col="dummy_value")
        print("✅ alpaca_format accepted extra args successfully")
        print(f"   Result: {result}")
    except TypeError as e:
        print(f"❌ alpaca_format failed: {e}")
        return False
    except Exception as e:
        print(f"❌ alpaca_format failed with unexpected error: {e}")
        return False
    return True

def test_progress_parsing():
    print("\nTesting progress parsing logic...")
    
    # Mock Manager class logic
    class MockManager:
        def __init__(self):
            self.progress = 0
            self.final_loss = None
            self.logs = []
            
    manager = MockManager()
    
    # Test line
    line = "[PROGRESS] current_step=50 total_steps=200 loss=1.234"
    
    # Paste the exact logic from manager.py
    if "[PROGRESS]" in line:
        try:
            parts = line.strip().split()
            data = {}
            for part in parts[1:]:
                key, val = part.split("=")
                data[key] = val
            
            current = int(data.get("current_step", 0))
            total = int(data.get("total_steps", 1))
            loss = float(data.get("loss", 0.0))
            
            if total > 0:
                manager.progress = int((current / total) * 100)
            
            if loss > 0:
                manager.final_loss = loss
        except Exception as e:
            print(f"Parsing failed: {e}")

    print(f"   Input: {line}")
    print(f"   Parsed Progress: {manager.progress}% (Expected: 25%)")
    print(f"   Parsed Loss: {manager.final_loss} (Expected: 1.234)")
    
    if manager.progress == 25 and manager.final_loss == 1.234:
        print("✅ Progress parsing verified")
        return True
    else:
        print("❌ Progress parsing failed")
        return False

if __name__ == "__main__":
    success_format = test_alpaca_format()
    success_parsing = test_progress_parsing()
    
    if success_format and success_parsing:
        print("\n✨ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n💀 SOME TESTS FAILED")
        sys.exit(1)
