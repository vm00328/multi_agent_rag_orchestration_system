import sys
from pathlib import Path

# in order for Python to find our modules when running tests, we need to add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))
