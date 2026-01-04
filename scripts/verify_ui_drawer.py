
import sys
import os

sys.path.append(os.getcwd())

def verify_drawer():
    print("🔍 Verifying Drawer Implementation...")
    
    # 1. Check CSS
    try:
        from engine.ui_v2.consts import UI_CSS
        if ".drawer-content" in UI_CSS:
            print("   ✅ CSS class '.drawer-content' found in UI_CSS.")
        else:
            print("   ❌ CSS class '.drawer-content' NOT found in UI_CSS.")
            return False
            
        if "right: 0 !important" in UI_CSS:
            print("   ✅ CSS positioning 'right: 0' found.")
        else:
            print("   ❌ CSS positioning NOT found.")
            return False
            
    except Exception as e:
        print(f"   ❌ Error checking CSS: {e}")
        return False

    # 2. Check UI File Content (Static Analysis)
    try:
        with open("engine/ui_v2/steps/data.py", "r", encoding="utf-8") as f:
            content = f.read()
            
        if 'elem_classes=["drawer-content"]' in content:
            print("   ✅ UI Component with drawer class found in data.py.")
        else:
            print("   ❌ UI Component with drawer class NOT found.")
            return False
            
        if 'tips_btn.click' in content:
             print("   ✅ Logic for drawer toggle found.")
        else:
             print("   ❌ Logic for drawer toggle NOT found.")
             return False

    except Exception as e:
        print(f"   ❌ Error checking UI file: {e}")
        return False

    print("\n✨ DRAWER VERIFICATION PASSED")
    return True

if __name__ == "__main__":
    if verify_drawer():
        sys.exit(0)
    else:
        sys.exit(1)
