"""
This script ensures that the project root is in the path.
"""

from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).parent.parent.absolute()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

DATA_DIR = PROJECT_DIR / "experiments" / "data"
RESULTS_DIR = PROJECT_DIR / "experiments" / "results"
MANUSCRIPT_DIR = PROJECT_DIR / "ms"
