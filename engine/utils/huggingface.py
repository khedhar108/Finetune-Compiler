"""
HuggingFace Hub integration utilities.

Handles authentication, model upload, and deployment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from engine.utils.logging import get_logger, print_info, print_success, print_error, print_warning

logger = get_logger(__name__)


# ============ Token Management ============

def get_read_token() -> Optional[str]:
    """Get HuggingFace read token from environment."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_READ_TOKEN")


def get_write_token() -> Optional[str]:
    """Get HuggingFace write token from environment."""
    return os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")


def set_read_token(token: str) -> bool:
    """
    Set HuggingFace read token.
    
    Args:
        token: HuggingFace token (should start with hf_)
        
    Returns:
        True if valid and set, False otherwise
    """
    if not token or not token.startswith("hf_"):
        return False
    os.environ["HF_READ_TOKEN"] = token
    os.environ["HF_TOKEN"] = token  # Also set default
    return True


def set_write_token(token: str) -> bool:
    """
    Set HuggingFace write token.
    
    Args:
        token: HuggingFace token with write permissions
        
    Returns:
        True if valid and set, False otherwise
    """
    if not token or not token.startswith("hf_"):
        return False
    os.environ["HF_WRITE_TOKEN"] = token
    return True


def check_token_status() -> dict:
    """
    Check status of HuggingFace tokens.
    
    Returns:
        Dict with read_token and write_token status
    """
    read_token = get_read_token()
    write_token = get_write_token()
    
    return {
        "read_token": f"✅ Set ({read_token[:8]}...)" if read_token else "❌ Not set",
        "write_token": f"✅ Set ({write_token[:8]}...)" if write_token else "❌ Not set",
        "can_download": bool(read_token),
        "can_upload": bool(write_token),
    }


# ============ Model Upload ============

def upload_to_hub(
    model_path: str,
    repo_name: str,
    private: bool = False,
    commit_message: str = "Upload fine-tuned model",
) -> dict:
    """
    Upload a trained model to HuggingFace Hub.
    
    Args:
        model_path: Path to the trained model directory
        repo_name: Repository name (format: username/model-name)
        private: Whether to make the repo private
        commit_message: Commit message for the upload
        
    Returns:
        Dict with success status and URL
    """
    try:
        from huggingface_hub import HfApi, create_repo
        
        write_token = get_write_token()
        if not write_token:
            return {
                "success": False,
                "error": "❌ Write token not set. Please add your write token.",
                "url": None,
            }
        
        model_path = Path(model_path)
        if not model_path.exists():
            return {
                "success": False,
                "error": f"❌ Model path not found: {model_path}",
                "url": None,
            }
        
        api = HfApi(token=write_token)
        
        # Validate token permissions first
        try:
            user_info = api.whoami(token=write_token)
            print_info(f"Authenticated as: {user_info.get('name', 'Unknown')}")
        except Exception as e:
            return {
                "success": False,
                "error": f"❌ Authentication failed: Invalid token. ({str(e)})",
                "url": None,
            }

        # Create repository if it doesn't exist
        print_info(f"Creating repository: {repo_name}")
        try:
            create_repo(
                repo_id=repo_name,
                token=write_token,
                private=private,
                exist_ok=True,
            )
        except Exception as e:
            if "403" in str(e):
                return {
                    "success": False,
                    "error": "❌ Permission Denied: Your token is READ-ONLY. Please provide a WRITE token.",
                    "url": None,
                }
            logger.warning(f"Repo creation note: {e}")
        
        # Upload all files
        print_info(f"Uploading model files from {model_path}")
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_name,
            commit_message=commit_message,
        )
        
        # Generate URL
        url = f"https://huggingface.co/{repo_name}"
        
        print_success(f"Model uploaded successfully!")
        print_info(f"URL: {url}")
        
        return {
            "success": True,
            "error": None,
            "url": url,
            "repo_name": repo_name,
        }
        
    except ImportError:
        return {
            "success": False,
            "error": "❌ huggingface_hub not installed. Run: pip install huggingface_hub",
            "url": None,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"❌ Upload failed: {str(e)}",
            "url": None,
        }


def load_from_hub(
    repo_name: str,
    local_dir: Optional[str] = None,
) -> dict:
    """
    Load a model from HuggingFace Hub.
    
    Args:
        repo_name: Repository name (format: username/model-name)
        local_dir: Optional local directory to download to
        
    Returns:
        Dict with success status and local path
    """
    try:
        from huggingface_hub import snapshot_download
        
        read_token = get_read_token()
        
        print_info(f"Downloading model: {repo_name}")
        
        local_path = snapshot_download(
            repo_id=repo_name,
            token=read_token,
            local_dir=local_dir,
        )
        
        print_success(f"Model downloaded to: {local_path}")
        
        return {
            "success": True,
            "error": None,
            "local_path": local_path,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"❌ Download failed: {str(e)}",
            "local_path": None,
        }


# ============ Validation ============

def validate_repo_name(repo_name: str) -> tuple[bool, str]:
    """
    Validate HuggingFace repository name.
    
    Args:
        repo_name: Repository name to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not repo_name:
        return False, "Repository name cannot be empty"
    
    if "/" not in repo_name:
        return False, "Format should be: username/model-name"
    
    parts = repo_name.split("/")
    if len(parts) != 2:
        return False, "Format should be: username/model-name"
    
    username, model_name = parts
    if not username or not model_name:
        return False, "Both username and model name are required"
    
    # Check for invalid characters
    invalid_chars = set(' !@#$%^&*()+=[]{}|;:\'",<>?')
    if any(c in repo_name for c in invalid_chars):
        return False, "Repository name contains invalid characters"
    
    return True, "✅ Valid repository name"


# ============ Search ============

def search_models(query: str, limit: int = 20) -> list[str]:
    """
    Search for models on Hugging Face Hub.
    
    Args:
        query: Search query
        limit: Maximum results to return
        
    Returns:
        List of model IDs
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        models = api.list_models(
            search=query,
            limit=limit,
            sort="downloads",
            direction=-1,
            filter="text-generation"  # Filter for text gen models primarily
        )
        
        return [model.id for model in models]
        
    except Exception as e:
        logger.warning(f"Model search failed: {e}")
        return []

