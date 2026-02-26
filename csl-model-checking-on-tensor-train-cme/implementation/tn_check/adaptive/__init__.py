"""
Adaptive rank controller for TT-compressed time evolution.

Implements greedy-doubling with per-bond singular-value monitoring:
if error exceeds 2*target, double all bond dimensions and re-integrate.
"""

from tn_check.adaptive.controller import AdaptiveRankController

__all__ = ["AdaptiveRankController"]
