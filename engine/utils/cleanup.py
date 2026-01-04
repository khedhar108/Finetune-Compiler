import psutil
import os
import signal
import sys

def main():
    print("🔍 Searching for 'ftune' related processes...")
    killed_count = 0
    
    current_pid = os.getpid()
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Skip the current process
            if proc.info['pid'] == current_pid:
                continue
                
            cmdline = proc.info['cmdline']
            if cmdline:
                # Check if 'ftune' is in the command line args
                # and it's a python process or the exe wrapper
                cmd_str = ' '.join(cmdline)
                if 'ftune' in cmd_str and ('python' in proc.info['name'] or 'ftune' in proc.info['name']):
                    print(f"🛑 Killing Process ID: {proc.info['pid']} - {cmd_str[:100]}...")
                    proc.kill()
                    killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if killed_count > 0:
        print(f"✅ Successfully terminated {killed_count} ftune instance(s).")
    else:
        print("✨ No running ftune instances found.")

if __name__ == "__main__":
    main()
