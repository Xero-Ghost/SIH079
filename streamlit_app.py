import sys
from pathlib import Path
import runpy

# Ensure root directory is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Launch main dashboard app
runpy.run_path(str(ROOT / "app" / "dashboard.py"), run_name="__main__")
