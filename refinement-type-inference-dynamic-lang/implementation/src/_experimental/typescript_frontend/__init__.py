"""
TypeScript front-end: TS AST → SSA IR translation.

Handles TypeScript type guards, discriminated unions, optional chaining,
nullish coalescing, and strict/loose equality semantics.
"""

from src.typescript_frontend.compiler import TypeScriptSSACompiler
from src.typescript_frontend.guard_extractor import TypeScriptGuardExtractor

__all__ = ["TypeScriptSSACompiler", "TypeScriptGuardExtractor"]
