
import os
import sys
import shutil
import urllib.request
import zipfile
from pathlib import Path

# Configuration
FFMPEG_URL = "https://github.com/GyanD/codexffmpeg/releases/download/2025-01-01-git-8d4bd796ba/ffmpeg-2025-01-01-git-8d4bd796ba-full_build-shared.7z"
# Using a zip fallback if 7z isn't easily handled without external libs, 
# but GyanD only provides 7z for release builds usually. 
# Let's use a known zip compatible release or handle 7z.
# Actually, standard python lib doesn't support 7z. 
# Switching to a zip source for shared build if available, or asking user to install 7zip support?
# Wait, for automation we need standard tools.
# Let's use BtbN builds which provide zip format for shared builds.
FFMPEG_ZIP_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip"

PROJECT_ROOT = Path(__file__).parent.parent
VENV_DIR = PROJECT_ROOT / ".venv"
TARGET_DIR = VENV_DIR / "ffmpeg"

def setup_ffmpeg():
    if not VENV_DIR.exists():
        print(f"Error: Virtual environment not found at {VENV_DIR}")
        print("Please run 'uv sync' first.")
        sys.exit(1)

    if (TARGET_DIR / "bin" / "avcodec-61.dll").exists() or (TARGET_DIR / "bin" / "avcodec-60.dll").exists(): 
        # Checking for common versions
        print(f"FFmpeg already appears to be installed in {TARGET_DIR}")
        return

    print(f"Downloading FFmpeg from {FFMPEG_ZIP_URL}...")
    zip_path = VENV_DIR / "ffmpeg.zip"
    
    try:
        # Download
        urllib.request.urlretrieve(FFMPEG_ZIP_URL, zip_path)
        print("Download complete. Extracting...")

        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(VENV_DIR)
        
        # Rename/Move to final location
        # The zip usually contains a single top-level folder like "ffmpeg-master-latest-win64-gpl-shared"
        extracted_dirs = [d for d in VENV_DIR.iterdir() if d.is_dir() and "ffmpeg" in d.name and d != TARGET_DIR]
        
        if extracted_dirs:
            # Assume the first new folder is the right one
            source_dir = extracted_dirs[0]
            if TARGET_DIR.exists():
                shutil.rmtree(TARGET_DIR)
            source_dir.rename(TARGET_DIR)
            print(f"FFmpeg installed to {TARGET_DIR}")
        else:
            print("Error: Could not find extracted folder.")

    except Exception as e:
        print(f"Failed to setup FFmpeg: {e}")
    finally:
        # Cleanup
        if zip_path.exists():
            zip_path.unlink()

if __name__ == "__main__":
    setup_ffmpeg()
