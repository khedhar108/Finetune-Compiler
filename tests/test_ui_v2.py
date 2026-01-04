"""
UI v2 Smoke Tests

Run before pushing: uv run pytest tests/test_ui_v2.py -v

These tests verify:
1. All step functions can be called without errors
2. All step functions return the expected number of components
3. The Gradio app can be created successfully

Note: These tests require Gradio to be installed (uv sync --extra ui)
"""

import pytest

# Skip all tests if Gradio is not installed
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not GRADIO_AVAILABLE,
    reason="Gradio not installed. Run: uv sync --extra ui"
)


def test_step1_model_returns_correct_count():
    """Step 1 should return exactly 6 components."""
    from engine.ui_v2.app import step1_model
    
    with gr.Blocks():
        result = step1_model()
    
    assert len(result) == 6, f"step1_model should return 6 items, got {len(result)}"


def test_step2_data_returns_correct_count():
    """Step 2 should return exactly 3 components."""
    from engine.ui_v2.app import step2_data
    
    with gr.Blocks():
        result = step2_data()
    
    assert len(result) == 3, f"step2_data should return 3 items, got {len(result)}"


def test_step3_training_returns_correct_count():
    """Step 3 should return exactly 11 components (added training_status)."""
    from engine.ui_v2.app import step3_training
    
    with gr.Blocks():
        result = step3_training()
    
    assert len(result) == 11, f"step3_training should return 11 items, got {len(result)}"


def test_step4_deploy_returns_correct_count():
    """Step 4 should return exactly 10 components (removed sandbox)."""
    from engine.ui_v2.app import step4_deploy
    
    with gr.Blocks():
        result = step4_deploy()
    
    assert len(result) == 10, f"step4_deploy should return 10 items, got {len(result)}"


def test_step5_sandbox_returns_correct_count():
    """Step 5 should return exactly 9 components."""
    from engine.ui_v2.app import step5_sandbox
    
    with gr.Blocks():
        result = step5_sandbox()
    
    assert len(result) == 9, f"step5_sandbox should return 9 items, got {len(result)}"


def test_wizard_app_creates_successfully():
    """The full wizard app should create without errors."""
    from engine.ui_v2.app import create_wizard_app
    
    app = create_wizard_app()
    
    assert app is not None, "create_wizard_app should return a Gradio Blocks object"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
