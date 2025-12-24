"""
Logging utilities for the AI Compiler Core Engine.

Provides structured logging with Rich console output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table


# Global console instance
console = Console()

# Logger cache
_loggers: dict[str, logging.Logger] = {}


def setup_logger(
    name: str = "ai-compiler",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    rich_tracebacks: bool = True,
) -> logging.Logger:
    """
    Set up a logger with Rich console handler.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for file logging
        rich_tracebacks: Whether to use Rich tracebacks
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Rich console handler
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=rich_tracebacks,
        show_time=True,
        show_path=False,
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rich_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
    
    # Cache the logger
    _loggers[name] = logger
    
    return logger


def get_logger(name: str = "ai-compiler") -> logging.Logger:
    """
    Get or create a logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


def log_config(config: dict[str, Any], title: str = "Configuration") -> None:
    """
    Log configuration as a formatted table.
    
    Args:
        config: Configuration dictionary
        title: Table title
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Section", style="cyan")
    table.add_column("Parameter", style="green")
    table.add_column("Value", style="yellow")
    
    for section, params in config.items():
        if isinstance(params, dict):
            for key, value in params.items():
                table.add_row(section, key, str(value))
        else:
            table.add_row("", section, str(params))
    
    console.print(table)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """
    Log metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number
    """
    step_str = f"[Step {step}] " if step is not None else ""
    metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    console.print(f"{step_str}{metrics_str}", style="blue")


def create_progress() -> Progress:
    """
    Create a Rich progress bar for training.
    
    Returns:
        Configured Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )


def print_banner() -> None:
    """Print the AI Compiler banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                    🚀 AI COMPILER CORE                        ║
║           Modular LLM Fine-Tuning Engine                      ║
╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"✅ {message}", style="bold green")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"❌ {message}", style="bold red")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"⚠️  {message}", style="bold yellow")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"ℹ️  {message}", style="bold blue")
