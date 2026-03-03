"""
Privacy amplification package for DP-Forge.

This package implements state-of-the-art privacy amplification techniques
for differential privacy:

1. **Shuffling**: Erlingsson-Feldman-Mironov model with Balle et al. tight
   analysis. Amplifies local DP to central DP via random shuffling.

2. **Subsampling (RDP)**: Mironov-Talwar-Zhang 2019 tight subsampling bounds
   for Rényi DP. Handles both Poisson and fixed-size subsampling.

3. **Random Check-in**: Balle et al. stochastic participation model. Amplifies
   privacy under random check-in protocols.

4. **Amplified CEGIS**: Extends the CEGIS synthesis loop to jointly optimize
   mechanism parameters and amplification protocol parameters.

Key Classes
-----------
- :class:`ShuffleAmplifier` — Shuffled model amplification (shuffling.py)
- :class:`SubsamplingRDPAmplifier` — Tight subsampling RDP bounds (subsampling_rdp.py)
- :class:`RandomCheckInAmplifier` — Random check-in amplification (random_check_in.py)
- :class:`AmplifiedCEGISEngine` — CEGIS with amplification (amplified_cegis.py)

Key Functions
-------------
- :func:`shuffle_amplification_bound` — Compute shuffled amplification
- :func:`poisson_subsampled_rdp` — Poisson subsampling RDP
- :func:`fixed_subsampled_rdp` — Fixed-size subsampling RDP
- :func:`random_checkin_amplification` — Random check-in bounds

Design Philosophy
-----------------
All amplification bounds are **numerically sound**: they use interval
arithmetic or conservative approximations to ensure computed bounds never
underestimate privacy loss. Integration with dp_forge.rdp allows seamless
composition of amplified mechanisms.

Usage Example
-------------
>>> from dp_forge.amplification import ShuffleAmplifier
>>> amplifier = ShuffleAmplifier(n_users=1000)
>>> eps_central, delta_central = amplifier.amplify(
...     epsilon_local=2.0, delta_local=0.0
... )
>>> print(f"Amplified: ε={eps_central:.3f}, δ={delta_central:.6f}")
"""

from dp_forge.amplification.amplified_cegis import (
    AmplifiedCEGISEngine,
    AmplifiedSynthesisResult,
    amplified_synthesize,
)
from dp_forge.amplification.random_check_in import (
    RandomCheckInAmplifier,
    random_checkin_amplification,
)
from dp_forge.amplification.shuffling import (
    ShuffleAmplifier,
    shuffle_amplification_bound,
    optimal_local_epsilon,
    minimum_n_for_amplification,
)
from dp_forge.amplification.subsampling_rdp import (
    SubsamplingRDPAmplifier,
    poisson_subsampled_rdp,
    fixed_subsampled_rdp,
    optimal_subsampling_rate,
)

__all__ = [
    # Shuffling
    "ShuffleAmplifier",
    "shuffle_amplification_bound",
    "optimal_local_epsilon",
    "minimum_n_for_amplification",
    # Subsampling RDP
    "SubsamplingRDPAmplifier",
    "poisson_subsampled_rdp",
    "fixed_subsampled_rdp",
    "optimal_subsampling_rate",
    # Random check-in
    "RandomCheckInAmplifier",
    "random_checkin_amplification",
    # Amplified CEGIS
    "AmplifiedCEGISEngine",
    "AmplifiedSynthesisResult",
    "amplified_synthesize",
]
