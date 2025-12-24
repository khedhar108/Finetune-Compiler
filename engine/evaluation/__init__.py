"""
Evaluation module for the AI Compiler Core Engine.
"""

from engine.evaluation.metrics import (
    compute_perplexity,
    compute_accuracy,
)

__all__ = [
    "compute_perplexity",
    "compute_accuracy",
]
