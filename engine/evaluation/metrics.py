"""
Evaluation metrics for the AI Compiler Core Engine.

Provides functions for computing various evaluation metrics.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from engine.utils.logging import get_logger, print_info

logger = get_logger(__name__)


def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: list[str],
    max_length: int = 2048,
    batch_size: int = 4,
) -> float:
    """
    Compute perplexity on a list of texts.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        texts: List of texts to evaluate
        max_length: Maximum sequence length
        batch_size: Batch size for evaluation
        
    Returns:
        Perplexity score (lower is better)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(device)
            
            outputs = model(**encodings, labels=encodings["input_ids"])
            
            # Get loss per token
            loss = outputs.loss
            num_tokens = encodings["attention_mask"].sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity


def compute_accuracy(
    predictions: list[str],
    references: list[str],
    normalize: bool = True,
) -> float:
    """
    Compute exact match accuracy.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        normalize: Whether to normalize texts before comparison
        
    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")
    
    if not predictions:
        return 0.0
    
    correct = 0
    
    for pred, ref in zip(predictions, references):
        if normalize:
            pred = pred.strip().lower()
            ref = ref.strip().lower()
        
        if pred == ref:
            correct += 1
    
    return correct / len(predictions)


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: Any,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        eval_dataset: Evaluation dataset
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of metric names to values
    """
    if metrics is None:
        metrics = ["perplexity"]
    
    print_info(f"Evaluating with metrics: {metrics}")
    
    results = {}
    
    # Get texts from dataset
    if hasattr(eval_dataset, "to_list"):
        texts = [ex.get("text", "") for ex in eval_dataset.to_list()]
    else:
        texts = [ex.get("text", "") for ex in eval_dataset]
    
    if "perplexity" in metrics:
        perplexity = compute_perplexity(model, tokenizer, texts)
        results["perplexity"] = perplexity
        print_info(f"Perplexity: {perplexity:.2f}")
    
    return results
