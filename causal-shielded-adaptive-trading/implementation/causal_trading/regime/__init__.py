"""
Regime inference module for Causal-Shielded Adaptive Trading.

Provides Bayesian nonparametric regime detection via Sticky HDP-HMM,
online regime tracking, transition matrix estimation, and full
posterior computation over latent market regimes.
"""

from .sticky_hdp_hmm import StickyHDPHMM
from .regime_detection import BayesianRegimeDetector
from .transition_matrix import TransitionMatrixEstimator
from .online_tracker import OnlineRegimeTracker
from .regime_posterior import RegimePosterior
from .student_t_emission import StudentTEmission, EmissionModelSelector

__all__ = [
    "StickyHDPHMM",
    "BayesianRegimeDetector",
    "TransitionMatrixEstimator",
    "OnlineRegimeTracker",
    "RegimePosterior",
    "StudentTEmission",
    "EmissionModelSelector",
]
