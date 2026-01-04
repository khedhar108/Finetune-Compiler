"""
Tests for dataset format detection and preview.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock

# Mock Gradio before importing modules that depend on it
sys.modules['gradio'] = MagicMock()
sys.modules['gr'] = MagicMock()


class TestAnalyzeDataset:
    """Tests for the analyze_dataset function."""
    
    def test_alpaca_format_detection(self):
        """Test that Alpaca format is detected from instruction/output columns."""
        from engine.ui_v2.utils import analyze_dataset
        
        # Mock dataset with Alpaca columns
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["instruction", "input", "output"]
        mock_dataset.features = {"instruction": "str", "input": "str", "output": "str"}
        mock_dataset.__len__ = lambda self: 3
        mock_dataset.__getitem__ = lambda self, i: {
            "instruction": f"Instruction {i}",
            "input": f"Input {i}",
            "output": f"Output {i}"
        }
        
        with patch("engine.ui_v2.utils.load_dataset", return_value=mock_dataset):
            result = analyze_dataset("tatsu-lab/alpaca")
            
            assert result["suggested_format"] == "alpaca"
            assert "text" in result["modalities"]
            assert "instruction" in result["columns"]
            assert result["error"] is None
    
    def test_chatml_format_detection(self):
        """Test that ChatML format is detected from messages column."""
        from engine.ui_v2.utils import analyze_dataset
        
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["messages"]
        mock_dataset.features = {"messages": "list"}
        mock_dataset.__len__ = lambda self: 2
        mock_dataset.__getitem__ = lambda self, i: {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }
        mock_dataset.get = lambda key, default=None: mock_dataset[0].get(key, default)
        
        with patch("engine.ui_v2.utils.load_dataset", return_value=mock_dataset):
            result = analyze_dataset("some/chat-dataset")
            
            assert result["suggested_format"] == "chatml"
            assert result["error"] is None
    
    def test_audio_format_detection(self):
        """Test that Audio format is detected from audio + transcription columns."""
        from engine.ui_v2.utils import analyze_dataset
        
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["audio", "transcription", "language"]
        mock_dataset.features = {
            "audio": "Audio(sampling_rate=16000)",
            "transcription": "str",
            "language": "str"
        }
        mock_dataset.__len__ = lambda self: 2
        mock_dataset.__getitem__ = lambda self, i: {
            "audio": {"array": [0.1, 0.2], "sampling_rate": 16000},
            "transcription": "Hello world",
            "language": "en"
        }
        
        with patch("engine.ui_v2.utils.load_dataset", return_value=mock_dataset):
            result = analyze_dataset("ekacare/eka-medical-asr-evaluation-dataset")
            
            assert result["suggested_format"] == "audio"
            assert "audio" in result["modalities"]
            assert result["error"] is None
    
    def test_error_handling(self):
        """Test that errors are handled gracefully."""
        from engine.ui_v2.utils import analyze_dataset
        
        with patch("engine.ui_v2.utils.load_dataset", side_effect=Exception("Dataset not found")):
            result = analyze_dataset("nonexistent/dataset")
            
            assert result["error"] is not None
            assert "not found" in result["error"].lower()
            assert result["suggested_format"] == "alpaca"  # Default fallback


class TestFormatPreviewTable:
    """Tests for the format_preview_table function."""
    
    def test_empty_rows(self):
        """Test handling of empty sample rows."""
        from engine.ui_v2.utils import format_preview_table
        
        result = format_preview_table([])
        assert "*No preview available*" in result
    
    def test_markdown_table_generation(self):
        """Test that markdown table is generated correctly."""
        from engine.ui_v2.utils import format_preview_table
        
        sample_rows = [
            {"col1": "value1", "col2": "value2"},
            {"col1": "value3", "col2": "value4"}
        ]
        
        result = format_preview_table(sample_rows)
        
        assert "| col1 | col2 |" in result
        assert "| --- | --- |" in result
        assert "value1" in result
        assert "value3" in result


class TestFormatInfo:
    """Tests for FORMAT_INFO constant."""
    
    def test_all_formats_have_required_keys(self):
        """Test that all formats have the required metadata keys."""
        from engine.ui_v2.consts import FORMAT_INFO
        
        required_keys = ["icon", "name", "desc", "columns", "example", "use_case"]
        
        for fmt_name, fmt_data in FORMAT_INFO.items():
            for key in required_keys:
                assert key in fmt_data, f"Format '{fmt_name}' missing key '{key}'"
    
    def test_formats_match_choices(self):
        """Test that FORMAT_INFO covers all FORMATS choices."""
        from engine.ui_v2.consts import FORMAT_INFO, FORMATS
        
        for fmt in FORMATS:
            assert fmt in FORMAT_INFO, f"Format '{fmt}' not in FORMAT_INFO"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
