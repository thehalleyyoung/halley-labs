"""
Python front-end: AST → SSA IR translation.

Handles Python 3.8–3.12 syntax, truthiness coercions,
isinstance/is None guards, comprehension desugaring, and exception flow.
"""

from src.python_frontend.compiler import PythonSSACompiler
from src.python_frontend.guard_extractor import PythonGuardExtractor
from src.python_frontend.ssa_passes import PythonSSAPasses

__all__ = ["PythonSSACompiler", "PythonGuardExtractor", "PythonSSAPasses"]
