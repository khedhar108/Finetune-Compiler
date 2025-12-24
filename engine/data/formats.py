"""
Dataset formatting utilities for the AI Compiler Core Engine.

Supports various prompt formats:
- Alpaca (instruction, input, output)
- ChatML (messages array)
- Completion (simple text)
- Custom templates
"""

from __future__ import annotations

from typing import Any, Callable

from datasets import Dataset

from engine.utils.logging import get_logger, print_info

logger = get_logger(__name__)


# ========== Prompt Templates ==========

ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_NO_INPUT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""

CHATML_TEMPLATE = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>"""


# ========== Format Functions ==========

def alpaca_format(
    example: dict[str, Any],
    instruction_col: str = "instruction",
    input_col: str = "input",
    output_col: str = "output",
) -> dict[str, str]:
    """
    Format a single example in Alpaca style.
    
    Args:
        example: Dataset row
        instruction_col: Column name for instruction
        input_col: Column name for input
        output_col: Column name for output
        
    Returns:
        Dictionary with 'text' key containing formatted prompt
    """
    instruction = example.get(instruction_col, "")
    input_text = example.get(input_col, "")
    output = example.get(output_col, "")
    
    if input_text and input_text.strip():
        text = ALPACA_TEMPLATE.format(
            instruction=instruction,
            input=input_text,
            output=output,
        )
    else:
        text = ALPACA_NO_INPUT_TEMPLATE.format(
            instruction=instruction,
            output=output,
        )
    
    return {"text": text}


def chatml_format(
    example: dict[str, Any],
    user_col: str = "user",
    assistant_col: str = "assistant",
    messages_col: str = "messages",
) -> dict[str, str]:
    """
    Format a single example in ChatML style.
    
    Args:
        example: Dataset row
        user_col: Column name for user message
        assistant_col: Column name for assistant message
        messages_col: Column name for messages array
        
    Returns:
        Dictionary with 'text' key containing formatted prompt
    """
    # Check if messages array format
    if messages_col in example and example[messages_col]:
        messages = example[messages_col]
        text_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        text = "\n".join(text_parts)
    else:
        # Simple user/assistant format
        user = example.get(user_col, "")
        assistant = example.get(assistant_col, "")
        
        text = CHATML_TEMPLATE.format(
            user=user,
            assistant=assistant,
        )
    
    return {"text": text}


def completion_format(
    example: dict[str, Any],
    text_col: str = "text",
) -> dict[str, str]:
    """
    Format a single example as simple text completion.
    
    Args:
        example: Dataset row
        text_col: Column name for text
        
    Returns:
        Dictionary with 'text' key
    """
    return {"text": example.get(text_col, "")}


def custom_format(
    example: dict[str, Any],
    template: str,
) -> dict[str, str]:
    """
    Format a single example using a custom template.
    
    Args:
        example: Dataset row
        template: Format string with placeholders matching column names
        
    Returns:
        Dictionary with 'text' key containing formatted prompt
    """
    try:
        text = template.format(**example)
    except KeyError as e:
        raise KeyError(f"Template placeholder not found in example: {e}")
    
    return {"text": text}


# ========== Main Formatting Function ==========

def format_dataset(
    dataset: Dataset,
    format_type: str = "alpaca",
    **kwargs: Any,
) -> Dataset:
    """
    Format a dataset according to the specified format type.
    
    Args:
        dataset: Dataset to format
        format_type: Format type (alpaca, chatml, completion, custom)
        **kwargs: Additional arguments for the format function
        
    Returns:
        Formatted dataset with 'text' column
    """
    format_type = format_type.lower()
    
    formatters: dict[str, Callable] = {
        "alpaca": alpaca_format,
        "chatml": chatml_format,
        "completion": completion_format,
    }
    
    if format_type == "custom":
        if "template" not in kwargs:
            raise ValueError("Custom format requires 'template' argument")
        
        template = kwargs.pop("template")
        format_fn = lambda ex: custom_format(ex, template)
    elif format_type in formatters:
        format_fn = lambda ex: formatters[format_type](ex, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format_type}. Supported: {list(formatters.keys())}")
    
    print_info(f"Formatting dataset with {format_type} format...")
    
    formatted = dataset.map(format_fn, remove_columns=dataset.column_names)
    
    logger.info(f"Formatted {len(formatted)} examples")
    
    return formatted


def get_format_columns(format_type: str) -> list[str]:
    """
    Get the required column names for a format type.
    
    Args:
        format_type: Format type
        
    Returns:
        List of required column names
    """
    columns = {
        "alpaca": ["instruction", "input", "output"],
        "chatml": ["messages"],  # or user, assistant
        "completion": ["text"],
    }
    
    return columns.get(format_type.lower(), [])
