"""Pytest configuration: add src/ to the import path."""

import sys
from pathlib import Path

# Add the implementation root so that `import src.xxx` works
_impl_root = Path(__file__).resolve().parent.parent
if str(_impl_root) not in sys.path:
    sys.path.insert(0, str(_impl_root))
