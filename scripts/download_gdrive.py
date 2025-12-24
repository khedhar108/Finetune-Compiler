#!/usr/bin/env python3
"""
Google Drive Dataset Download Script

Downloads files from Google Drive for dataset preparation.
Supports both single files and folders.

Usage:
    # Download single file
    uv run python scripts/download_gdrive.py \\
        --file-id 1ABC123xyz \\
        --output data/downloaded.csv

    # Download folder
    uv run python scripts/download_gdrive.py \\
        --folder-id 1XYZ789abc \\
        --output-dir data/gdrive_download
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import gdown
    from rich.console import Console
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: uv pip install gdown rich")
    sys.exit(1)


console = Console()


def download_file(file_id: str, output_path: str, quiet: bool = False) -> str:
    """
    Download a single file from Google Drive.
    
    Args:
        file_id: Google Drive file ID
        output_path: Local output path
        quiet: Suppress progress output
        
    Returns:
        Path to downloaded file
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    
    console.print(f"[cyan]Downloading file: {file_id}[/cyan]")
    
    output = gdown.download(url, output_path, quiet=quiet)
    
    if output:
        console.print(f"[green]✅ Downloaded to: {output}[/green]")
        return output
    else:
        raise RuntimeError("Download failed")


def download_folder(folder_id: str, output_dir: str, quiet: bool = False) -> str:
    """
    Download a folder from Google Drive.
    
    Args:
        folder_id: Google Drive folder ID
        output_dir: Local output directory
        quiet: Suppress progress output
        
    Returns:
        Path to output directory
    """
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    
    console.print(f"[cyan]Downloading folder: {folder_id}[/cyan]")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    gdown.download_folder(url, output=output_dir, quiet=quiet)
    
    console.print(f"[green]✅ Downloaded to: {output_dir}[/green]")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download files from Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download single file
  python download_gdrive.py --file-id 1ABC123 --output data/file.csv
  
  # Download folder
  python download_gdrive.py --folder-id 1XYZ789 --output-dir data/audio
  
How to get file/folder ID:
  1. Open file/folder in Google Drive
  2. Copy the ID from the URL:
     - File: https://drive.google.com/file/d/[FILE_ID]/view
     - Folder: https://drive.google.com/drive/folders/[FOLDER_ID]
        """,
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file-id",
        help="Google Drive file ID",
    )
    group.add_argument(
        "--folder-id",
        help="Google Drive folder ID",
    )
    
    parser.add_argument(
        "--output",
        help="Output path for file download",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/gdrive_download",
        help="Output directory for folder download (default: ./data/gdrive_download)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    try:
        if args.file_id:
            if not args.output:
                console.print("[red]Error: --output required for file download[/red]")
                sys.exit(1)
            download_file(args.file_id, args.output, args.quiet)
        else:
            download_folder(args.folder_id, args.output_dir, args.quiet)
            
        console.print()
        console.print("[bold cyan]Next step:[/bold cyan]")
        console.print("  uv run python scripts/prepare_audio_dataset.py --help")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
