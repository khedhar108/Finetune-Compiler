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
                # Parse structured progress
                if "[PROGRESS]" in line:
                    try:
                        # [PROGRESS] current_step=10 total_steps=100 loss=0.5
                        parts = line.strip().split()
                        data = {}
                        for part in parts[1:]:  # Skip [PROGRESS]
                            key, val = part.split("=")
                            data[key] = val
                        
                        current = int(data.get("current_step", 0))
                        total = int(data.get("total_steps", 1))
                        loss = float(data.get("loss", 0.0))
                        
                        if total > 0:
                            self.progress = int((current / total) * 100)
                        
                        if loss > 0:
                            self.final_loss = loss
                            
                    except Exception as e:
                        # self.logs.append(f"\nDebug: Error parsing progress: {e}\n")
                        pass

                # Fallback: Parse standard HuggingFace logs
                elif "'loss':" in line or '"loss":' in line:
                     # {'loss': 2.5029, 'learning_rate': 0.0002, 'epoch': 0.01}
                     try:
                         # Try to extract loss from dict-like string
                         import ast
                         # Find dict part
                         idx_start = line.find("{")
                         idx_end = line.rfind("}")
                         if idx_start != -1 and idx_end != -1:
                             log_dict = ast.literal_eval(line[idx_start:idx_end+1])
                             if "loss" in log_dict:
                                 self.final_loss = float(log_dict["loss"])
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
