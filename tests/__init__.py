"""
AI Compiler Test Suite
======================

Run all tests:
    uv run pytest tests/ -v

Run specific test file:
    uv run pytest tests/test_ui_v2.py -v

Pre-push validation (recommended):
    uv run python -m tests

This will:
1. Run import checks
2. Run all pytest tests (if pytest installed)
"""

import sys


def run_all_tests():
    """Run all tests and return exit code."""
    try:
        import pytest
    except ImportError:
        print("⚠️  pytest not installed. Run: uv sync --extra dev")
        print("    Skipping pytest tests...")
        return 0
    
    # Run with verbose output
    exit_code = pytest.main([
        "tests/",
        "-v",
        "--tb=short",
        "-x",
    ])
    
    return exit_code


def quick_check():
    """Quick import check without running full tests."""
    print("🔍 Quick Import Check...")
    
    errors = []
    
    # Check core imports
    try:
        from engine.ui_v2 import app
        print("  ✅ engine.ui_v2.app")
    except ImportError as e:
        errors.append(f"  ❌ engine.ui_v2.app: {e}")
    
    try:
        from engine.cli import main
        print("  ✅ engine.cli.main")
    except ImportError as e:
        errors.append(f"  ❌ engine.cli.main: {e}")
    
    try:
        from engine.utils import huggingface
        print("  ✅ engine.utils.huggingface")
    except ImportError as e:
        errors.append(f"  ❌ engine.utils.huggingface: {e}")
    
    if errors:
        print("\n❌ Import errors found:")
        for e in errors:
            print(e)
        return 1
    
    print("\n✅ All core imports OK!")
    return 0
