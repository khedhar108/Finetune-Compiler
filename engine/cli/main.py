"""
CLI interface for the AI Compiler Core Engine.

Provides commands for training, evaluation, and export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from engine import __version__
from engine.utils.config import Config, load_config, get_default_config
from engine.utils.logging import (
    print_banner,
    print_success,
    print_error,
    print_info,
    print_warning,
    log_config,
)
from engine.utils.memory import print_gpu_info
import os
import sys

# Inject local FFmpeg path if it exists
project_root = Path(__file__).resolve().parent.parent.parent
local_ffmpeg = project_root / ".venv" / "ffmpeg" / "bin"
if local_ffmpeg.exists():
    os.environ["PATH"] = str(local_ffmpeg) + os.pathsep + os.environ["PATH"]
    if sys.platform == "win32":
        try:
            os.add_dll_directory(str(local_ffmpeg))
        except Exception:
            pass

# Create Typer app
app = typer.Typer(
    name="ai-compile",
    help="AI Compiler Core - Modular LLM Fine-Tuning Engine",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=False,
)

console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold cyan]AI Compiler Core[/bold cyan] version {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    ctx: typer.Context = typer.Option(None, hidden=True),
) -> None:
    """
    AI Compiler Core - Modular LLM Fine-Tuning Engine
    
    Fine-tune LLMs with LoRA/QLoRA on limited hardware.
    """
    # If no subcommand is invoked, launch the UI
    if ctx.invoked_subcommand is None and not version:
        try:
            from engine.ui_v2 import launch_wizard_ui
            print_banner()
            # Default to port 7862 for Wizard UI
            launch_wizard_ui(server_port=7862)
        except ImportError:
            print_error("UI dependencies not installed.")
            print_info("Install with: uv sync --extra ui")
            raise typer.Exit(1)


@app.command()
def train(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to JSON configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    resume: Optional[Path] = typer.Option(
        None,
        "--resume",
        "-r",
        help="Resume from checkpoint directory",
    ),
    auto_resume: bool = typer.Option(
        True,
        "--auto-resume/--no-auto-resume",
        help="Automatically resume from last checkpoint if available",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry_run",
        help="Validate config without training",
    ),
) -> None:
    """
    Train a model with the specified configuration.
    
    Features:
    - Auto-resume from last checkpoint (--auto-resume)
    - Checkpoint locking (prevents re-computation)
    - Manual resume (--resume ./checkpoint-100)
    
    Example:
        ai-compile train --config configs/default.json
        ai-compile train --config configs/default.json --resume ./output/checkpoint-100
    """
    print_banner()
    
    try:
        # Load configuration
        print_info(f"Loading configuration: {config}")
        cfg = load_config(config)
        
        # Display configuration
        log_config(cfg.to_dict(), title="Training Configuration")
        
        # Show GPU info
        print_gpu_info()
        
        if dry_run:
            print_success("Configuration validated successfully (dry run)")
            raise typer.Exit(0)
        
        # Check for checkpoint/resume
        from engine.utils.checkpoint import CheckpointManager
        checkpoint_mgr = CheckpointManager(cfg.project.output_dir)
        
        resume_path = None
        
        # Priority: explicit resume > auto-resume
        if resume:
            resume_path = str(resume)
            print_info(f"Resuming from specified checkpoint: {resume_path}")
        elif auto_resume and checkpoint_mgr.can_resume():
            resume_info = checkpoint_mgr.get_resume_info()
            resume_path = resume_info["checkpoint_path"]
            print_info(f"Auto-resuming from last checkpoint: {resume_path}")
            print_info(f"  Last step: {resume_info['last_step']}, Loss: {resume_info['last_loss']:.4f}")
        
        # Acquire lock
        if not checkpoint_mgr.acquire_lock():
            print_warning("Another training session may be in progress.")
            if not typer.confirm("Continue anyway?"):
                raise typer.Exit(0)
        
        try:
            # Import training module
            from engine.training import Trainer
            
            # Create trainer and run
            trainer = Trainer(cfg)
            trainer.train(resume_from=resume_path)
            
            print_success("Training completed!")
            
        finally:
            # Always release lock
            checkpoint_mgr.release_lock()
        
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Training failed: {e}")
        raise typer.Exit(1)


@app.command()
def infer(
    model: Path = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to trained model or HuggingFace model ID",
    ),
    prompt: str = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Prompt to generate from",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Start interactive chat mode",
    ),
    max_tokens: int = typer.Option(
        256,
        "--max-tokens",
        help="Maximum tokens to generate",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Sampling temperature (0.0 = deterministic)",
    ),
    format_type: str = typer.Option(
        "alpaca",
        "--format",
        "-f",
        help="Prompt format (alpaca, chatml, llama)",
    ),
) -> None:
    """
    Run inference on a trained model.
    
    Examples:
        ai-compile infer --model ./output --prompt "What is AI?"
        ai-compile infer --model ./output --interactive
        ai-compile infer --model username/model --prompt "Hello"
    """
    print_banner()
    
    try:
        from engine.inference import InferenceEngine
        
        model_path = str(model) if model.exists() else str(model)
        
        print_info(f"Loading model: {model_path}")
        engine = InferenceEngine(
            model_path=model_path,
            prompt_format=format_type,
        )
        
        if interactive:
            # Interactive chat mode
            console.print("\n[bold green]Interactive Mode[/bold green]")
            console.print("[dim]Type 'exit' or 'quit' to end[/dim]\n")
            
            while True:
                try:
                    user_input = console.input("[bold cyan]You:[/bold cyan] ")
                    
                    if user_input.lower() in ["exit", "quit", "q"]:
                        break
                    
                    if not user_input.strip():
                        continue
                    
                    console.print("[bold green]AI:[/bold green] ", end="")
                    
                    # Stream response
                    for chunk in engine.stream(user_input, max_tokens=max_tokens, temperature=temperature):
                        console.print(chunk, end="")
                    console.print()
                    
                except KeyboardInterrupt:
                    break
            
            console.print("\n[dim]Goodbye![/dim]")
            
        elif prompt:
            # Single prompt mode
            print_info(f"Generating response...")
            
            response = engine.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            console.print(f"\n[bold green]Response:[/bold green]\n{response}")
            
        else:
            print_error("Please provide --prompt or use --interactive mode")
            raise typer.Exit(1)
            
    except Exception as e:
        print_error(f"Inference failed: {e}")
        raise typer.Exit(1)


@app.command()
def evaluate(
    model: Path = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to trained model or adapter",
        exists=True,
        resolve_path=True,
    ),
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to evaluation dataset",
        exists=True,
        resolve_path=True,
    ),
    metrics: str = typer.Option(
        "perplexity",
        "--metrics",
        help="Comma-separated list of metrics",
    ),
) -> None:
    """
    Evaluate a trained model.
    
    Example:
        ai-compile evaluate --model ./output --dataset test.csv --metrics accuracy,bleu
    """
    print_banner()
    
    print_info(f"Evaluating model: {model}")
    print_info(f"Dataset: {dataset}")
    print_info(f"Metrics: {metrics}")
    
    # TODO: Implement evaluation
    print_warning("Evaluation not yet implemented")


@app.command()
def export(
    model: Path = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to trained model or adapter",
        exists=True,
        resolve_path=True,
    ),
    format: str = typer.Option(
        "adapter",
        "--format",
        "-f",
        help="Export format: adapter, merged, gguf",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path (defaults to model path with format suffix)",
    ),
    quantization: Optional[str] = typer.Option(
        None,
        "--quantization",
        "-q",
        help="GGUF quantization (q4_k_m, q5_k_m, etc.)",
    ),
) -> None:
    """
    Export a trained model to different formats.
    
    Example:
        ai-compile export --model ./output --format gguf --quantization q4_k_m
    """
    print_banner()
    
    print_info(f"Exporting model: {model}")
    print_info(f"Format: {format}")
    
    if output:
        print_info(f"Output: {output}")
    
    # Implement export logic
    if format.lower() == "gguf":
        from engine.models.export import export_to_gguf
        
        qt = quantization if quantization else "q4_k_m"  # Default to balanced
        success = export_to_gguf(
            model_path=str(model),
            output_path=str(output) if output else None,
            quantization=qt
        )
        
        if not success:
            raise typer.Exit(code=1)
            
    else:
        # Placeholder for other formats (adapter, merged)
        # For now, just copy the files or use standard PEFT merge
        print_warning(f"Format '{format}' not yet implemented in CLI. Only 'gguf' is fully supported via Unsloth.")


@app.command()
def init(
    output: Path = typer.Option(
        Path("./config.json"),
        "--output",
        "-o",
        help="Output path for config file",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing file",
    ),
) -> None:
    """
    Generate a default configuration file.
    
    Example:
        ai-compile init --output my_config.json
    """
    import json
    
    if output.exists() and not force:
        print_error(f"File already exists: {output}. Use --force to overwrite.")
        raise typer.Exit(1)
    
    default_config = get_default_config()
    
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=2)
    
    print_success(f"Default configuration saved to: {output}")


@app.command()
def info() -> None:
    """
    Display system and GPU information.
    """
    print_banner()
    
    console.print(f"[bold]Version:[/bold] {__version__}")
    console.print()
    
    print_gpu_info()
    
    console.print()
    console.print("[bold]Installed packages:[/bold]")
    
    try:
        import torch
        console.print(f"  PyTorch: {torch.__version__}")
        console.print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            console.print(f"  CUDA version: {torch.version.cuda}")
    except ImportError:
        console.print("  [red]PyTorch not installed[/red]")
    
    try:
        import transformers
        console.print(f"  Transformers: {transformers.__version__}")
    except ImportError:
        console.print("  [red]Transformers not installed[/red]")
    
    try:
        import peft
        console.print(f"  PEFT: {peft.__version__}")
    except ImportError:
        console.print("  [red]PEFT not installed[/red]")


@app.command()
def ui(
    share: bool = typer.Option(
        False,
        "--share",
        "-s",
        help="Create a public share link",
    ),
    port: int = typer.Option(
        7860,
        "--port",
        "-p",
        help="Port to run the UI on",
    ),
) -> None:
    """
    Launch the visual UI for training and configuration.
    
    Example:
        ai-compile ui
        ai-compile ui --share
        ai-compile ui --port 8080
    """
    try:
        from engine.ui import launch_ui
        
        print_banner()
        print_info(f"Starting UI on http://127.0.0.1:{port}")
        
        if share:
            print_info("Creating public share link...")
        
        launch_ui(share=share, server_port=port)
        
    except ImportError:
        print_error("UI dependencies not installed.")
        print_info("Install with: uv sync --extra ui")
        raise typer.Exit(1)


@app.command()
def deploy(
    model: Path = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to trained model or adapter",
        exists=True,
        resolve_path=True,
    ),
    repo: str = typer.Option(
        ...,
        "--repo",
        "-r",
        help="HuggingFace repository name (format: username/model-name)",
    ),
    private: bool = typer.Option(
        False,
        "--private",
        "-p",
        help="Make the repository private",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        "-t",
        help="HuggingFace write token (or set HF_WRITE_TOKEN env var)",
    ),
) -> None:
    """
    Deploy a trained model to HuggingFace Hub.
    
    Example:
        ai-compile deploy --model ./output --repo username/my-model
        ai-compile deploy --model ./output --repo username/my-model --private
    """
    print_banner()
    
    try:
        from engine.utils.huggingface import upload_to_hub, set_write_token, validate_repo_name
        
        # Validate repo name
        is_valid, msg = validate_repo_name(repo)
        if not is_valid:
            print_error(f"Invalid repository name: {msg}")
            raise typer.Exit(1)
        
        # Set token if provided
        if token:
            set_write_token(token)
        
        print_info(f"Deploying model: {model}")
        print_info(f"Repository: {repo}")
        print_info(f"Private: {private}")
        
        # Upload
        result = upload_to_hub(
            model_path=str(model),
            repo_name=repo,
            private=private,
        )
        
        if result["success"]:
            print_success("Model deployed successfully!")
            console.print(f"\n[bold green]📎 Model URL:[/bold green] {result['url']}")
            console.print("\n[dim]Use this URL in your applications![/dim]")
        else:
            print_error(result["error"])
            raise typer.Exit(1)
            
    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        print_info("Run: uv sync")
        raise typer.Exit(1)


@app.command()
def ui2(
    share: bool = typer.Option(
        False,
        "--share",
        "-s",
        help="Create a public share link",
    ),
    port: int = typer.Option(
        7862,
        "--port",
        "-p",
        help="Port to run the UI on",
    ),
) -> None:
    """
    Launch the Wizard UI v2 (step-by-step interface).
    
    Enhanced UI with:
    - Step-by-step wizard flow
    - Collapsible logs panel
    - HuggingFace integration
    - One-click deploy
    
    Example:
        ai-compile ui2
        ai-compile ui2 --share
    """
    try:
        from engine.ui_v2 import launch_wizard_ui
        
        print_banner()
        print_info(f"Starting Wizard UI v2 on http://127.0.0.1:{port}")
        
        if share:
            print_info("Creating public share link...")
        
        launch_wizard_ui(share=share, server_port=port)
        
    except ImportError:
        print_error("UI dependencies not installed.")
        print_info("Install with: uv sync --extra ui")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
