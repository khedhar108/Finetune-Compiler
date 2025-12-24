"""
Data loading utilities for the AI Compiler Core Engine.

Supports loading from various sources:
- HuggingFace Hub
- Local CSV files
- Local JSON/JSONL files
- Google Drive
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset as hf_load_dataset

from engine.utils.logging import get_logger, print_info, print_success, print_error

logger = get_logger(__name__)


def load_dataset_from_source(
    source: str,
    path: str,
    **kwargs: Any,
) -> Dataset:
    """
    Load a dataset from the specified source.
    
    Args:
        source: Source type (huggingface, csv, json, gdrive)
        path: Path or identifier for the dataset
        **kwargs: Additional arguments passed to the loader
        
    Returns:
        Loaded dataset
        
    Raises:
        ValueError: If source is not supported
    """
    source = source.lower()
    
    loaders = {
        "huggingface": load_huggingface,
        "hf": load_huggingface,
        "csv": load_csv,
        "json": load_json,
        "jsonl": load_json,
        "gdrive": load_gdrive,
    }
    
    if source not in loaders:
        raise ValueError(f"Unsupported source: {source}. Supported: {list(loaders.keys())}")
    
    logger.info(f"Loading dataset from {source}: {path}")
    dataset = loaders[source](path, **kwargs)
    logger.info(f"Loaded {len(dataset)} samples")
    
    return dataset


def load_csv(
    path: str,
    split: str | None = None,
    **kwargs: Any,
) -> Dataset:
    """
    Load a dataset from a CSV file.
    
    Args:
        path: Path to the CSV file
        split: Optional split to select
        **kwargs: Additional arguments for load_dataset
        
    Returns:
        Loaded dataset
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    print_info(f"Loading CSV: {path}")
    
    dataset = hf_load_dataset(
        "csv",
        data_files=str(path),
        split=split or "train",
        **kwargs,
    )
    
    print_success(f"Loaded {len(dataset)} rows from CSV")
    return dataset


def load_json(
    path: str,
    split: str | None = None,
    **kwargs: Any,
) -> Dataset:
    """
    Load a dataset from a JSON or JSONL file.
    
    Args:
        path: Path to the JSON/JSONL file
        split: Optional split to select
        **kwargs: Additional arguments for load_dataset
        
    Returns:
        Loaded dataset
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    print_info(f"Loading JSON: {path}")
    
    dataset = hf_load_dataset(
        "json",
        data_files=str(path),
        split=split or "train",
        **kwargs,
    )
    
    print_success(f"Loaded {len(dataset)} rows from JSON")
    return dataset


def load_huggingface(
    dataset_name: str,
    split: str = "train",
    subset: str | None = None,
    **kwargs: Any,
) -> Dataset:
    """
    Load a dataset from HuggingFace Hub.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        split: Split to load (default: train)
        subset: Optional subset/config name
        **kwargs: Additional arguments for load_dataset
        
    Returns:
        Loaded dataset
    """
    print_info(f"Loading from HuggingFace: {dataset_name}")
    
    if subset:
        dataset = hf_load_dataset(dataset_name, subset, split=split, **kwargs)
    else:
        dataset = hf_load_dataset(dataset_name, split=split, **kwargs)
    
    print_success(f"Loaded {len(dataset)} samples from HuggingFace")
    return dataset


def load_gdrive(
    file_id: str,
    output_path: str = "./data/gdrive_download",
    file_type: str = "csv",
    **kwargs: Any,
) -> Dataset:
    """
    Load a dataset from Google Drive.
    
    Args:
        file_id: Google Drive file ID
        output_path: Local path to save the downloaded file
        file_type: Type of file (csv, json)
        **kwargs: Additional arguments for the loader
        
    Returns:
        Loaded dataset
    """
    try:
        import gdown
    except ImportError:
        raise ImportError("gdown is required for Google Drive downloads. Install with: pip install gdown")
    
    print_info(f"Downloading from Google Drive: {file_id}")
    
    # Construct download URL
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add extension if not present
    if not output_path.suffix:
        output_path = output_path.with_suffix(f".{file_type}")
    
    # Download file
    gdown.download(url, str(output_path), quiet=False)
    
    print_success(f"Downloaded to: {output_path}")
    
    # Load based on file type
    if file_type == "csv":
        return load_csv(str(output_path), **kwargs)
    elif file_type in ("json", "jsonl"):
        return load_json(str(output_path), **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def split_dataset(
    dataset: Dataset,
    test_size: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Split a dataset into train and test sets.
    
    Args:
        dataset: Dataset to split
        test_size: Fraction for test set (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1")
    
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    
    return split["train"], split["test"]
