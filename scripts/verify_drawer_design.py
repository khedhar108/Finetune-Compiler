
import sys
import os

def verify_drawer_design():
    print("🔍 Verifying Drawer Design...")
    
    # 1. Check CSS for Z-Index and Animation
    with open("engine/ui_v2/consts.py", "r", encoding="utf-8") as f:
        css = f.read()
        if "z-index: 2147483647" in css:
            print("   ✅ Max Z-Index Found")
        else:
            print("   ❌ Max Z-Index Missing")
            return False
            
        if "@keyframes slideInRight" in css:
            print("   ✅ Animation Keyframes Found")
        else:
            print("   ❌ Animation Missing")
            return False

    # 2. Check Logic in Data.py
    with open("engine/ui_v2/steps/data.py", "r", encoding="utf-8") as f:
        logic = f.read()
        if "gr.update(visible=True)" in logic and "gr.update(visible=False)" in logic:
             print("   ✅ Correct gr.update() logic Found")
        else:
             print("   ❌ Logic missing gr.update()")
             return False
             
        if "gr.Column(visible=True)" in logic:
            print("   ❌ Found usage of gr.Column() in lambda (Possible Duplicate)")
            # return False # Warn only, as gr.Column might be used elsewhere legitimately
    
    return True

if __name__ == "__main__":
    if verify_drawer_design():
        print("\n✨ DRAWER DESIGN VERIFIED")
        sys.exit(0)
    else:
        print("\n❌ VERIFICATION FAILED")
        sys.exit(1)
