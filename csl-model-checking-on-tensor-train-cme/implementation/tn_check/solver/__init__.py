"""
Dense reference solver for ground-truth CME computation.

Provides exact solutions via sparse/dense matrix exponentiation
for comparison with TT-compressed results.
"""

from tn_check.solver.dense_reference import DenseReferenceSolver

__all__ = ["DenseReferenceSolver"]
