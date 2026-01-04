import sys
try:
    import torchcodec
    print("SUCCESS: torchcodec imported successfully.")
except Exception as e:
    print(f"FAILURE: Could not import torchcodec. Error: {e}")
    sys.exit(1)
