"""
Training Manager for AI Compiler UI v2.
"""

import subprocess
import threading
import time

class TrainingManager:
    """Manages training subprocess and logs."""
    
    # Status constants
    STATUS_IDLE = "idle"
    STATUS_RUNNING = "running"
    STATUS_SUCCESS = "success"
    STATUS_FAILED = "failed"
    
    def __init__(self):
        self.process = None
        self.logs = []
        self.is_running = False
        self.progress = 0
        self.status = self.STATUS_IDLE
        self.final_loss = None
        self.error_message = None
    
    def start(self, config_path: str):
        self.logs = ["🚀 Starting training...\n"]
        self.is_running = True
        self.progress = 0
        self.status = self.STATUS_RUNNING
        self.final_loss = None
        self.error_message = None
        
        self.process = subprocess.Popen(
            ["uv", "run", "ai-compile", "train", "--config", config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        def read_output():
            for line in self.process.stdout:
                self.logs.append(line)
                # Parse progress from logs
                if "Step" in line:
                    try:
                        step = int(line.split("Step")[1].split("]")[0].strip())
                        self.progress = min(step, 100)
                    except:
                        pass
                # Parse loss
                if "loss" in line.lower():
                    try:
                        import re
                        match = re.search(r'loss[:\s]+([0-9.]+)', line.lower())
                        if match:
                            self.final_loss = float(match.group(1))
                    except:
                        pass
                # Check for errors
                if "error" in line.lower() or "exception" in line.lower():
                    self.error_message = line.strip()
            
            # Determine final status
            exit_code = self.process.wait()
            self.is_running = False
            
            if exit_code == 0:
                self.status = self.STATUS_SUCCESS
                self.logs.append("\n✅ Training complete!")
            else:
                self.status = self.STATUS_FAILED
                self.logs.append(f"\n❌ Training failed (exit code: {exit_code})")
        
        thread = threading.Thread(target=read_output, daemon=True)
        thread.start()
    
    def stop(self):
        if self.process:
            self.process.terminate()
            self.is_running = False
            self.status = self.STATUS_FAILED
            self.logs.append("\n⏹ Training stopped by user.")
    
    def get_logs(self) -> str:
        # Return ALL logs, not just last 50
        return "".join(self.logs)
    
    def get_progress(self) -> int:
        return self.progress
    
    def get_status(self) -> str:
        return self.status
    
    def get_status_html(self) -> str:
        """Get status as styled HTML."""
        if self.status == self.STATUS_IDLE:
            return "⏳ **Waiting to start...**"
        elif self.status == self.STATUS_RUNNING:
            return "🔄 **Training in progress...**"
        elif self.status == self.STATUS_SUCCESS:
            loss_str = f" | Loss: {self.final_loss:.4f}" if self.final_loss else ""
            return f"✅ **Training Complete!**{loss_str}"
        else:  # FAILED
            return f"❌ **Training Failed**"


training_manager = TrainingManager()
