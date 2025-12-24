#!/usr/bin/env python3
"""
Audio Dataset Preparation Script

Converts raw audio files (MP3, M4A, OGG, etc.) to 16kHz WAV format
required for ASR model fine-tuning.

Usage:
    uv run python scripts/prepare_audio_dataset.py \\
        --input-csv data/raw_dataset.csv \\
        --output-dir data/prepared \\
        --audio-column audio_path \\
        --transcription-column transcription

Input CSV format:
    audio_path,transcription
    /path/to/audio1.mp3,"Hello world"
    /path/to/audio2.m4a,"How are you"

Output:
    - Converted WAV files in output directory
    - New CSV with updated audio paths
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    import librosa
    import soundfile as sf
    import pandas as pd
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: uv pip install librosa soundfile pandas rich")
    sys.exit(1)


console = Console()


def convert_audio(
    input_path: str,
    output_path: str,
    target_sr: int = 16000,
    normalize: bool = True,
) -> bool:
    """
    Convert audio file to 16kHz mono WAV.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output WAV file
        target_sr: Target sample rate (default: 16000)
        normalize: Whether to normalize amplitude
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load audio (librosa handles many formats)
        audio, _ = librosa.load(input_path, sr=target_sr, mono=True)
        
        # Normalize amplitude to [-1, 1]
        if normalize:
            audio = librosa.util.normalize(audio)
        
        # Save as WAV
        sf.write(output_path, audio, target_sr)
        
        return True
    except Exception as e:
        console.print(f"[red]Error converting {input_path}: {e}[/red]")
        return False


def prepare_dataset(
    input_csv: str,
    output_dir: str,
    audio_column: str = "audio_path",
    transcription_column: str = "transcription",
    target_sr: int = 16000,
    skip_existing: bool = True,
) -> str:
    """
    Prepare audio dataset for ASR fine-tuning.
    
    Converts all audio files to 16kHz WAV format and creates
    a new CSV with updated paths.
    
    Args:
        input_csv: Path to input CSV
        output_dir: Output directory for converted files
        audio_column: Column name for audio paths
        transcription_column: Column name for transcriptions
        target_sr: Target sample rate
        skip_existing: Skip already converted files
        
    Returns:
        Path to prepared CSV
    """
    # Load CSV
    console.print(f"[cyan]Loading dataset: {input_csv}[/cyan]")
    df = pd.read_csv(input_csv)
    
    if audio_column not in df.columns:
        raise ValueError(f"Audio column '{audio_column}' not found in CSV")
    if transcription_column not in df.columns:
        raise ValueError(f"Transcription column '{transcription_column}' not found in CSV")
    
    console.print(f"[green]Found {len(df)} samples[/green]")
    
    # Create output directory
    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert audio files
    new_paths = []
    successful = 0
    failed = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Converting audio...", total=len(df))
        
        for idx, row in df.iterrows():
            input_path = row[audio_column]
            output_path = audio_dir / f"{idx:06d}.wav"
            
            # Skip if exists
            if skip_existing and output_path.exists():
                new_paths.append(str(output_path))
                progress.update(task, advance=1)
                continue
            
            # Convert
            if convert_audio(input_path, str(output_path), target_sr):
                new_paths.append(str(output_path))
                successful += 1
            else:
                new_paths.append("")  # Mark as failed
                failed += 1
            
            progress.update(task, advance=1)
    
    # Update DataFrame
    df[audio_column] = new_paths
    
    # Remove failed conversions
    df = df[df[audio_column] != ""]
    
    # Save prepared CSV
    output_csv = output_dir / "prepared_dataset.csv"
    df.to_csv(output_csv, index=False)
    
    # Summary
    console.print()
    console.print("[bold green]✅ Dataset preparation complete![/bold green]")
    console.print(f"   Successful: {successful}")
    console.print(f"   Failed: {failed}")
    console.print(f"   Output CSV: {output_csv}")
    console.print(f"   Audio files: {audio_dir}")
    
    return str(output_csv)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare audio dataset for ASR fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python prepare_audio_dataset.py --input-csv data/raw.csv --output-dir data/prepared
  
  # Custom column names
  python prepare_audio_dataset.py \\
      --input-csv data/raw.csv \\
      --output-dir data/prepared \\
      --audio-column file_path \\
      --transcription-column text
        """,
    )
    
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to input CSV with audio paths and transcriptions",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for prepared dataset",
    )
    parser.add_argument(
        "--audio-column",
        default="audio_path",
        help="Column name for audio file paths (default: audio_path)",
    )
    parser.add_argument(
        "--transcription-column",
        default="transcription",
        help="Column name for transcriptions (default: transcription)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-convert existing files",
    )
    
    args = parser.parse_args()
    
    try:
        output_csv = prepare_dataset(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            audio_column=args.audio_column,
            transcription_column=args.transcription_column,
            target_sr=args.sample_rate,
            skip_existing=not args.no_skip_existing,
        )
        
        console.print()
        console.print("[bold cyan]Next step:[/bold cyan]")
        console.print(f"  uv run ai-compile train --config configs/asr_whisper.json")
        console.print(f"  # Make sure to update 'data.path' to: {output_csv}")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
