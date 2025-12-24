"""
AI Compiler Core Engine

A modular, resource-efficient LLM fine-tuning engine.
"""

__version__ = "0.1.0"
__author__ = "Pradeep"

from engine.utils.config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
