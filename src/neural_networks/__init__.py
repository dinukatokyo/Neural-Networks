"""Customer churn teaching studio."""
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """The installed neural-networks command opens the Streamlit studio."""
    root = Path(__file__).resolve().parents[2]
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(root / "src/neural_networks/app.py")],
        cwd=root,
    ))
