"""
Checkpoint management utilities.

Handles saving, loading, and locking checkpoints for resume training.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from engine.utils.logging import get_logger, print_info, print_warning, print_success

logger = get_logger(__name__)


# Lock file name
LOCK_FILE = ".training.lock"
CHECKPOINT_INFO = "checkpoint_info.json"


class CheckpointManager:
    """
    Manages training checkpoints with locking mechanism.
    
    Prevents re-computation by:
    1. Creating a lock file when training starts
    2. Saving checkpoint info (step, epoch, loss)
    3. Allowing resume from last checkpoint
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.lock_path = self.output_dir / LOCK_FILE
        self.info_path = self.output_dir / CHECKPOINT_INFO
        
    def is_locked(self) -> bool:
        """Check if training is currently in progress."""
        return self.lock_path.exists()
    
    def acquire_lock(self) -> bool:
        """
        Acquire training lock.
        
        Returns:
            True if lock acquired, False if already locked
        """
        if self.is_locked():
            lock_info = self._read_lock()
            print_warning(f"Training already in progress since {lock_info.get('started_at', 'unknown')}")
            return False
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        lock_info = {
            "started_at": datetime.now().isoformat(),
            "pid": os.getpid(),
            "status": "running",
        }
        
        with open(self.lock_path, "w") as f:
            json.dump(lock_info, f, indent=2)
        
        print_info("Training lock acquired")
        return True
    
    def release_lock(self, status: str = "completed") -> None:
        """Release training lock."""
        if self.lock_path.exists():
            self.lock_path.unlink()
            print_info(f"Training lock released (status: {status})")
    
    def _read_lock(self) -> dict:
        """Read lock file info."""
        try:
            with open(self.lock_path) as f:
                return json.load(f)
        except:
            return {}
    
    def save_checkpoint_info(
        self,
        step: int,
        epoch: int,
        loss: float,
        checkpoint_path: str,
    ) -> None:
        """
        Save checkpoint information for resume.
        
        Args:
            step: Current training step
            epoch: Current epoch
            loss: Current loss value
            checkpoint_path: Path to checkpoint directory
        """
        info = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "checkpoint_path": checkpoint_path,
            "saved_at": datetime.now().isoformat(),
        }
        
        with open(self.info_path, "w") as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Checkpoint info saved: step={step}, epoch={epoch}, loss={loss:.4f}")
    
    def get_checkpoint_info(self) -> Optional[dict]:
        """
        Get last checkpoint information.
        
        Returns:
            Dict with checkpoint info or None if no checkpoint
        """
        if not self.info_path.exists():
            return None
        
        try:
            with open(self.info_path) as f:
                return json.load(f)
        except:
            return None
    
    def get_resume_checkpoint(self) -> Optional[str]:
        """
        Get checkpoint path for resuming.
        
        Returns:
            Path to checkpoint directory or None
        """
        info = self.get_checkpoint_info()
        if info and info.get("checkpoint_path"):
            checkpoint_path = Path(info["checkpoint_path"])
            if checkpoint_path.exists():
                return str(checkpoint_path)
        
        # Also check for HuggingFace Trainer checkpoints
        checkpoints = list(self.output_dir.glob("checkpoint-*"))
        if checkpoints:
            # Return latest checkpoint
            latest = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
            return str(latest)
        
        return None
    
    def can_resume(self) -> bool:
        """Check if training can be resumed."""
        return self.get_resume_checkpoint() is not None
    
    def get_resume_info(self) -> dict:
        """
        Get information about resumable training.
        
        Returns:
            Dict with resume status and info
        """
        checkpoint = self.get_resume_checkpoint()
        info = self.get_checkpoint_info()
        
        return {
            "can_resume": checkpoint is not None,
            "checkpoint_path": checkpoint,
            "last_step": info.get("step") if info else None,
            "last_epoch": info.get("epoch") if info else None,
            "last_loss": info.get("loss") if info else None,
            "saved_at": info.get("saved_at") if info else None,
        }
    
    def clean_checkpoints(self, keep_last: int = 3) -> None:
        """
        Clean old checkpoints, keeping only the last N.
        
        Args:
            keep_last: Number of checkpoints to keep
        """
        checkpoints = sorted(
            self.output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )
        
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                shutil.rmtree(checkpoint)
                logger.info(f"Removed old checkpoint: {checkpoint}")


def get_checkpoint_manager(output_dir: str) -> CheckpointManager:
    """Create a checkpoint manager for the given output directory."""
    return CheckpointManager(output_dir)


def check_resume_available(output_dir: str) -> dict:
    """
    Check if training can be resumed.
    
    Args:
        output_dir: Output directory to check
        
    Returns:
        Dict with resume status and checkpoint info
    """
    manager = CheckpointManager(output_dir)
    return manager.get_resume_info()
