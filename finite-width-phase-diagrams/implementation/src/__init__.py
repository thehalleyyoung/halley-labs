"""Finite-Width Phase Diagram system for neural networks.

Computes phase diagrams predicting lazy-to-rich training transitions
using NTK theory, spectral bifurcation analysis, and empirical calibration.

Primary API (see api.py):
    detect_regime, compute_phase_diagram, predict_phase_boundary,
    recommend_training_regime
"""

__version__ = "0.2.0"
