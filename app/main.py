import subprocess
import sys

if __name__ == "__main__":
    sys.exit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", "app/ui/streamlit_app.py"]
        )
    )
