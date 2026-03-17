"""DSL module: tokenizer, parser, AST, type checker, compiler, formatter.

Provides a complete pipeline for processing the RegSynth regulatory DSL.
"""

from regsynth_py.dsl.tokenizer import tokenize, Token, TokenType
from regsynth_py.dsl.parser import parse, ParseError
from regsynth_py.dsl.ast_nodes import Program
from regsynth_py.dsl.type_checker import TypeChecker
from regsynth_py.dsl.compiler import Compiler
from regsynth_py.dsl.formatter import Formatter

__all__ = [
    "tokenize", "Token", "TokenType",
    "parse", "ParseError",
    "Program", "TypeChecker", "Compiler", "Formatter",
]
