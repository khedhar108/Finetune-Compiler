"""
Inference utilities for trained models.

Load and run inference on fine-tuned models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Generator

from engine.utils.logging import get_logger, print_info, print_success

logger = get_logger(__name__)


def load_model_for_inference(
    model_path: str,
    device: str = "auto",
    quantization: Optional[str] = None,
):
    """
    Load a trained model for inference.
    
    Args:
        model_path: Path to model or HuggingFace model ID
        device: Device to load on ('auto', 'cuda', 'cpu')
        quantization: Optional quantization ('4bit', '8bit')
        
    Returns:
        Tuple of (model, tokenizer)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print_info(f"Loading model from: {model_path}")
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optional quantization
    model_kwargs = {
        "device_map": device if device == "auto" else None,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }
    
    if quantization == "4bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quantization == "8bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    
    if device != "auto" and quantization is None:
        model = model.to(device)
    
    model.eval()
    print_success(f"Model loaded on {device}")
    
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    stream: bool = False,
) -> str | Generator[str, None, None]:
    """
    Generate text from a prompt.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        do_sample: Whether to sample (False = greedy)
        stream: Whether to stream output
        
    Returns:
        Generated text or generator if streaming
    """
    import torch
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    if stream:
        return _stream_generate(
            model, tokenizer, inputs,
            max_new_tokens, temperature, top_p, do_sample
        )
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove prompt from output
    if generated.startswith(prompt):
        generated = generated[len(prompt):].strip()
    
    return generated


def _stream_generate(
    model, tokenizer, inputs,
    max_new_tokens, temperature, top_p, do_sample
) -> Generator[str, None, None]:
    """Stream generation token by token."""
    import torch
    from transformers import TextIteratorStreamer
    from threading import Thread
    
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    for text in streamer:
        yield text


def chat_format_prompt(
    instruction: str,
    system_prompt: str = "You are a helpful assistant.",
    format_type: str = "alpaca",
) -> str:
    """
    Format a prompt for chat-style models.
    
    Args:
        instruction: User instruction
        system_prompt: System prompt
        format_type: Format type ('alpaca', 'chatml', 'llama')
        
    Returns:
        Formatted prompt
    """
    if format_type == "alpaca":
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
    
    elif format_type == "chatml":
        return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
    
    elif format_type == "llama":
        return f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{instruction} [/INST]"""
    
    else:
        return instruction


class InferenceEngine:
    """
    High-level inference engine for easy model usage.
    
    Example:
        engine = InferenceEngine("./output")
        response = engine.generate("What is machine learning?")
        print(response)
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        quantization: Optional[str] = None,
        prompt_format: str = "alpaca",
    ):
        self.model_path = model_path
        self.prompt_format = prompt_format
        self.model, self.tokenizer = load_model_for_inference(
            model_path, device, quantization
        )
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        format_prompt: bool = True,
    ) -> str:
        """Generate response for a prompt."""
        if format_prompt:
            prompt = chat_format_prompt(prompt, format_type=self.prompt_format)
        
        return generate_text(
            self.model, self.tokenizer, prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    
    def stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        format_prompt: bool = True,
    ) -> Generator[str, None, None]:
        """Stream response for a prompt."""
        if format_prompt:
            prompt = chat_format_prompt(prompt, format_type=self.prompt_format)
        
        return generate_text(
            self.model, self.tokenizer, prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
