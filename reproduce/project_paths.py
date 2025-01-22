"""
    This script ensures that the project root is in the path.
    This makes the following import possible:

    >>> import project_path # Import this first!
    >>> from slisemap.slisemap import Slisemap
    >>> from experiment.data import get_autompg
"""

from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).parent.parent.absolute()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

DATA_DIR = PROJECT_DIR / "reproduce" / "data"
RESULTS_DIR = PROJECT_DIR / "reproduce" / "results"
MANUSCRIPT_DIR = PROJECT_DIR / "ms"
