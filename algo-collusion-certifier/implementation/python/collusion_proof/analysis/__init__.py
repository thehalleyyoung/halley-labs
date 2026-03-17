"""Analysis module for collusion detection and certification."""

try:
    from collusion_proof.analysis.composite_test import CompositeTest
    from collusion_proof.analysis.game_config import GameConfig
    from collusion_proof.analysis.test_result import TestResult
    from collusion_proof.analysis.verdict import Verdict
    from collusion_proof.analysis.collusion_premium import CollusionPremiumResult
    from collusion_proof.analysis.price_dispersion import PriceDispersionAnalyzer
    from collusion_proof.analysis.structural_break import StructuralBreakDetector
except ImportError:
    pass
