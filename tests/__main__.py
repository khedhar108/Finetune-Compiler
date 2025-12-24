"""
Entry point for running tests as a module.

Usage:
    uv run python -m tests
"""

from tests import quick_check, run_all_tests
import sys

if __name__ == "__main__":
    print("=" * 50)
    print("🧪 AI Compiler Test Suite")
    print("=" * 50)
    
    # Quick check first
    if quick_check() != 0:
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("Running pytest...")
    print("=" * 50 + "\n")
    
    # Run full tests
    sys.exit(run_all_tests())
