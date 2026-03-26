import subprocess
import sys
import os

def start_app():
    ui_path = os.path.join("app", "ui.py")
    
    print("Starting AI Research Copilot UI...")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", ui_path])
    except KeyboardInterrupt:
        print("\n Shutting down...")
    except FileNotFoundError:
        print("Error: Streamlit is not installed or app/ui.py is missing.")

if __name__ == "__main__":
    start_app()