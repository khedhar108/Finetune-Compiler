
import sys
import os

def verify_ui_classes():
    print("🔍 Verifying UI Refactor...")
    
    files_to_check = {
        "engine/ui_v2/steps/model.py": ["cols-2-1", "sticky-column", "premium-card"],
        "engine/ui_v2/steps/data.py": ["cols-2-1", "sticky-column", "premium-card", "drawer-content"]
    }
    
    success = True
    
    for relative_path, classes in files_to_check.items():
        if not os.path.exists(relative_path):
            print(f"❌ File not found: {relative_path}")
            success = False
            continue
            
        with open(relative_path, "r", encoding="utf-8") as f:
            content = f.read()
            print(f"📄 Checking {relative_path}...")
            for cls in classes:
                if cls in content:
                    print(f"   ✅ Found class '{cls}'")
                else:
                    print(f"   ❌ Missing class '{cls}'")
                    success = False
                    
    return success

if __name__ == "__main__":
    if verify_ui_classes():
        print("\n✨ UI REFACTOR VERIFIED")
        sys.exit(0)
    else:
        print("\n❌ SOME CLASSES MISSING")
        sys.exit(1)
