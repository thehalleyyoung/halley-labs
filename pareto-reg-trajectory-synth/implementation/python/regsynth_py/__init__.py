"""RegSynth: Regulatory compliance synthesis toolkit.

Provides a Python frontend for regulatory DSL parsing, jurisdiction database,
visualization, report generation, benchmarking, and analysis.
"""

__version__ = "0.1.0"
__author__ = "RegSynth Team"

from regsynth_py.dsl.tokenizer import tokenize, Token, TokenType
from regsynth_py.dsl.parser import parse, ParseError
from regsynth_py.dsl.ast_nodes import Program, ObligationDecl, JurisdictionDecl
from regsynth_py.dsl.type_checker import TypeChecker
from regsynth_py.dsl.compiler import Compiler
from regsynth_py.jurisdiction.database import JurisdictionDB
from regsynth_py.analysis.cost_analysis import CostAnalyzer
from regsynth_py.analysis.risk_analysis import RiskAnalyzer
from regsynth_py.analysis.coverage_analysis import CoverageAnalyzer

__all__ = [
    "__version__",
    "tokenize", "Token", "TokenType",
    "parse", "ParseError",
    "Program", "ObligationDecl", "JurisdictionDecl",
    "TypeChecker", "Compiler",
    "JurisdictionDB",
    "CostAnalyzer", "RiskAnalyzer", "CoverageAnalyzer",
]
