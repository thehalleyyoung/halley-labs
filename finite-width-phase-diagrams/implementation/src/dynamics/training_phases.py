"""
Training phase identification and characterization.

Provides tools for detecting and analyzing distinct phases of neural network
training: lazy regime, catapult phase, edge of stability, condensation/neural
collapse, grokking, and double descent.  Complements the existing dynamics
modules with phase-level diagnostics.
"""

import numpy as np
from scipy import optimize, stats, signal
from scipy.special import gammaln
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingPhaseConfig:
    """Configuration for training phase analysis.

    Parameters
    ----------
    max_steps : int
        Maximum number of training steps.
    log_interval : int
        Interval at which diagnostics are logged.
    lr : float
        Learning rate.
    width : int
        Network width.
    depth : int
        Network depth.
    activation : str
        Activation function identifier.
    """
    max_steps: int = 10000
    log_interval: int = 100
    lr: float = 0.01
    width: int = 100
    depth: int = 2
    activation: str = 'relu'


# ---------------------------------------------------------------------------
# 1. PhaseIdentifier
# ---------------------------------------------------------------------------

class PhaseIdentifier:
    r"""Identify the current phase of training from trajectory data.

    Classifies training into phases: lazy, rich, catapult, edge-of-stability,
    condensation, grokking, or saddle-escape.

    Parameters
    ----------
    config : TrainingPhaseConfig
        Configuration.
    """

    def __init__(self, config: TrainingPhaseConfig):
        self.config = config

    # ----- Main identifier -------------------------------------------------

    def identify_phase(
        self,
        loss_trajectory: np.ndarray,
        ntk_trajectory: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        r"""Classify the current training phase from trajectories.

        Examines loss curve shape, NTK stability, and gradient statistics
        to determine which dynamical regime training is in.

        Parameters
        ----------
        loss_trajectory : np.ndarray
            Loss values over training steps.
        ntk_trajectory : list of np.ndarray or None
            NTK matrices at corresponding steps (optional).

        Returns
        -------
        dict
            ``phase`` (str), ``confidence`` (float), ``indicators``.
        """
        L = np.asarray(loss_trajectory, dtype=np.float64)
        indicators = {}

        # Check for catapult
        catapult = self.catapult_phase_detector(L)
        indicators['catapult'] = catapult

        # Check for grokking (need both train and test — use loss as proxy)
        # Not enough info here; mark as unknown
        indicators['grokking'] = False

        # Check NTK stability (lazy vs rich)
        if ntk_trajectory is not None and len(ntk_trajectory) >= 2:
            lazy = self.lazy_phase_detector(ntk_trajectory[0], ntk_trajectory[-1])
            indicators['lazy'] = lazy
        else:
            lazy = None
            indicators['lazy'] = None

        # Determine phase
        if catapult:
            phase = 'catapult'
            confidence = 0.8
        elif lazy is True:
            phase = 'lazy'
            confidence = 0.9
        elif lazy is False:
            phase = 'rich'
            confidence = 0.7
        else:
            # Fall back to loss curve shape
            if len(L) > 10 and np.std(L[-len(L) // 4:]) < 0.01 * np.mean(L[-len(L) // 4:]):
                phase = 'converged'
                confidence = 0.8
            else:
                phase = 'transient'
                confidence = 0.5

        return {
            'phase': phase,
            'confidence': confidence,
            'indicators': indicators,
        }

    # ----- Lazy phase detector ---------------------------------------------

    def lazy_phase_detector(
        self,
        ntk_initial: np.ndarray,
        ntk_current: np.ndarray,
        threshold: float = 0.1,
    ) -> bool:
        r"""Detect whether training is in the lazy (kernel) regime.

        The lazy regime holds when ||Θ(t) - Θ(0)||_F / ||Θ(0)||_F < threshold.

        Parameters
        ----------
        ntk_initial : np.ndarray
            NTK at initialization.
        ntk_current : np.ndarray
            NTK at current time.
        threshold : float
            Relative change threshold.

        Returns
        -------
        bool
            True if in the lazy regime.
        """
        K0 = np.asarray(ntk_initial, dtype=np.float64)
        Kt = np.asarray(ntk_current, dtype=np.float64)
        norm0 = np.linalg.norm(K0, 'fro')
        if norm0 < 1e-15:
            return False
        rel_change = np.linalg.norm(Kt - K0, 'fro') / norm0
        return bool(rel_change < threshold)

    # ----- Rich phase detector ---------------------------------------------

    def rich_phase_detector(
        self,
        feature_matrices: List[np.ndarray],
    ) -> bool:
        r"""Detect whether training is in the rich (feature-learning) regime.

        Features are evolving significantly if the cosine similarity
        between initial and current feature matrices is low.

        Parameters
        ----------
        feature_matrices : list of np.ndarray
            Feature matrices at different training steps.

        Returns
        -------
        bool
            True if features are evolving significantly.
        """
        if len(feature_matrices) < 2:
            return False

        F0 = np.asarray(feature_matrices[0], dtype=np.float64)
        Ft = np.asarray(feature_matrices[-1], dtype=np.float64)

        norm0 = np.linalg.norm(F0, 'fro')
        normt = np.linalg.norm(Ft, 'fro')
        if norm0 < 1e-15 or normt < 1e-15:
            return True

        cosine = np.sum(F0 * Ft) / (norm0 * normt)
        return bool(cosine < 0.9)

    # ----- Catapult phase detector -----------------------------------------

    def catapult_phase_detector(
        self,
        loss_trajectory: np.ndarray,
        window: int = 100,
    ) -> bool:
        r"""Detect the catapult phase: sudden loss increase then decrease.

        The catapult occurs when loss spikes above its initial value
        before eventually decreasing below it.

        Parameters
        ----------
        loss_trajectory : np.ndarray
            Loss values.
        window : int
            Smoothing window size.

        Returns
        -------
        bool
            True if catapult phase detected.
        """
        L = np.asarray(loss_trajectory, dtype=np.float64)
        if len(L) < 3 * window:
            return False

        L0 = L[0]
        # Smooth the trajectory
        kernel = np.ones(min(window, len(L))) / min(window, len(L))
        L_smooth = np.convolve(L, kernel, mode='valid')

        if len(L_smooth) < 3:
            return False

        # Catapult: loss exceeds initial value significantly then drops below
        max_loss = np.max(L_smooth)
        final_loss = L_smooth[-1]

        spike = max_loss > 1.5 * L0  # significant spike
        recovery = final_loss < L0  # recovers below initial

        return bool(spike and recovery)

    # ----- Edge-of-stability detector --------------------------------------

    def edge_of_stability_detector(
        self,
        loss_trajectory: np.ndarray,
        hessian_eigenvalues: np.ndarray,
    ) -> bool:
        r"""Detect edge-of-stability: sharpness saturates near 2/η.

        Parameters
        ----------
        loss_trajectory : np.ndarray
            Loss values.
        hessian_eigenvalues : np.ndarray
            Top Hessian eigenvalue (sharpness) at each step.

        Returns
        -------
        bool
            True if at edge of stability.
        """
        lr = self.config.lr
        threshold = 2.0 / lr
        sharpness = np.asarray(hessian_eigenvalues, dtype=np.float64)

        if len(sharpness) < 10:
            return False

        # Check if sharpness oscillates around 2/η in the latter half
        latter = sharpness[len(sharpness) // 2:]
        mean_sharp = np.mean(latter)
        std_sharp = np.std(latter)

        near_threshold = abs(mean_sharp - threshold) < 0.2 * threshold
        oscillating = std_sharp > 0.01 * threshold

        return bool(near_threshold and oscillating)

    # ----- Condensation detector -------------------------------------------

    def condensation_detector(
        self,
        weight_matrices: List[np.ndarray],
    ) -> bool:
        r"""Detect weight condensation (rank collapse).

        Condensation occurs when weight matrices become approximately
        low-rank during training.

        Parameters
        ----------
        weight_matrices : list of np.ndarray
            Weight matrices at different steps.

        Returns
        -------
        bool
            True if condensation detected.
        """
        if len(weight_matrices) < 2:
            return False

        W0 = np.asarray(weight_matrices[0], dtype=np.float64)
        Wt = np.asarray(weight_matrices[-1], dtype=np.float64)

        # Effective rank via entropy of normalized singular values
        def _eff_rank(W):
            s = np.linalg.svd(W, compute_uv=False)
            s = s[s > 1e-14]
            if len(s) == 0:
                return 0
            p = s / np.sum(s)
            return float(np.exp(-np.sum(p * np.log(p + 1e-300))))

        rank0 = _eff_rank(W0)
        rankt = _eff_rank(Wt)

        if rank0 < 1e-10:
            return False

        return bool(rankt < 0.5 * rank0)

    # ----- Grokking detector -----------------------------------------------

    def grokking_detector(
        self,
        train_losses: np.ndarray,
        test_losses: np.ndarray,
    ) -> bool:
        r"""Detect grokking: training loss converges long before test loss.

        Grokking is identified when:
        1. Train loss reaches near-zero early
        2. Test loss remains high for a long period
        3. Test loss eventually drops

        Parameters
        ----------
        train_losses : np.ndarray
            Training loss trajectory.
        test_losses : np.ndarray
            Test loss trajectory.

        Returns
        -------
        bool
            True if grokking pattern detected.
        """
        train = np.asarray(train_losses, dtype=np.float64)
        test = np.asarray(test_losses, dtype=np.float64)
        n = min(len(train), len(test))
        if n < 20:
            return False
        train = train[:n]
        test = test[:n]

        # Find when train loss < 1% of initial
        train_thresh = 0.01 * train[0]
        train_converged = np.argmax(train < train_thresh) if np.any(train < train_thresh) else n

        # Find when test loss < 10% of initial
        test_thresh = 0.1 * test[0]
        test_converged = np.argmax(test < test_thresh) if np.any(test < test_thresh) else n

        # Grokking: test converges much later than train
        delay_ratio = test_converged / max(train_converged, 1)
        return bool(delay_ratio > 5 and test_converged < n)

    # ----- Saddle escape detector ------------------------------------------

    def saddle_escape_detector(
        self,
        loss_trajectory: np.ndarray,
        gradient_norms: np.ndarray,
    ) -> bool:
        r"""Detect saddle point escape.

        Near a saddle, the loss plateaus while gradient norms are small,
        followed by a sudden drop when the unstable direction is found.

        Parameters
        ----------
        loss_trajectory : np.ndarray
            Loss values.
        gradient_norms : np.ndarray
            Gradient norm at each step.

        Returns
        -------
        bool
            True if saddle escape detected.
        """
        L = np.asarray(loss_trajectory, dtype=np.float64)
        g = np.asarray(gradient_norms, dtype=np.float64)
        n = min(len(L), len(g))
        if n < 20:
            return False
        L = L[:n]
        g = g[:n]

        # Look for plateau (small gradient) followed by sudden drop
        window = max(n // 10, 5)

        for i in range(window, n - window):
            # Plateau: gradient norms small in a window
            g_window = g[i - window:i]
            mean_g = np.mean(g_window)
            # Sudden drop: loss decreases significantly after plateau
            L_before = np.mean(L[i - window:i])
            L_after = np.mean(L[i:i + window])

            if mean_g < 0.1 * np.mean(g) and L_after < 0.8 * L_before:
                return True

        return False

    # ----- Phase timeline --------------------------------------------------

    def phase_timeline(
        self,
        trajectories: Dict[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        r"""Construct a full timeline of training phases.

        Parameters
        ----------
        trajectories : dict
            Must contain 'loss'. Optional keys: 'ntk' (list of matrices),
            'sharpness', 'gradient_norms', 'test_loss', 'weights' (list).

        Returns
        -------
        list of dict
            Phase segments with ``phase``, ``start``, ``end``.
        """
        loss = np.asarray(trajectories['loss'], dtype=np.float64)
        n = len(loss)
        phases = []
        chunk_size = max(n // 10, 20)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = loss[start:end]

            # Simple heuristics
            if len(chunk) < 3:
                continue

            rate = (chunk[-1] - chunk[0]) / (len(chunk) + 1e-14)
            volatility = np.std(chunk) / (np.mean(chunk) + 1e-14)

            if rate > 0 and abs(rate) > 0.1 * chunk[0]:
                phase = 'catapult'
            elif volatility > 0.3:
                phase = 'edge_of_stability'
            elif abs(rate) < 1e-4 * (chunk[0] + 1e-14):
                phase = 'plateau'
            elif rate < 0:
                phase = 'descent'
            else:
                phase = 'transient'

            phases.append({
                'phase': phase,
                'start': start,
                'end': end,
            })

        return phases

    # ----- Transition times ------------------------------------------------

    def transition_times(
        self,
        phase_sequence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        r"""Extract transition times from a phase sequence.

        Parameters
        ----------
        phase_sequence : list of dict
            Output from ``phase_timeline``.

        Returns
        -------
        list of dict
            Each transition with ``time``, ``from_phase``, ``to_phase``.
        """
        transitions = []
        for i in range(1, len(phase_sequence)):
            if phase_sequence[i]['phase'] != phase_sequence[i - 1]['phase']:
                transitions.append({
                    'time': phase_sequence[i]['start'],
                    'from_phase': phase_sequence[i - 1]['phase'],
                    'to_phase': phase_sequence[i]['phase'],
                })
        return transitions


# ---------------------------------------------------------------------------
# 2. CatapultPhaseAnalysis
# ---------------------------------------------------------------------------

class CatapultPhaseAnalysis:
    r"""Detailed analysis of the catapult phase in neural network training.

    The catapult phase occurs when the learning rate is large enough that
    lr × λ_max > 2, causing gradient descent to initially increase the loss
    before eventually recovering to a better minimum.

    Parameters
    ----------
    config : TrainingPhaseConfig
        Configuration.
    """

    def __init__(self, config: TrainingPhaseConfig):
        self.config = config

    # ----- Catapult onset --------------------------------------------------

    def catapult_onset(self, loss_trajectory: np.ndarray) -> int:
        r"""Find the step at which the catapult begins.

        The catapult starts when the loss first exceeds its initial value.

        Parameters
        ----------
        loss_trajectory : np.ndarray
            Loss values.

        Returns
        -------
        int
            Step index of catapult onset (-1 if not found).
        """
        L = np.asarray(loss_trajectory, dtype=np.float64)
        L0 = L[0]
        for i in range(1, len(L)):
            if L[i] > 1.1 * L0:
                return i
        return -1

    # ----- Catapult duration -----------------------------------------------

    def catapult_duration(self, loss_trajectory: np.ndarray) -> int:
        r"""Duration of the catapult phase (in steps).

        The catapult ends when loss drops back below its initial value.

        Parameters
        ----------
        loss_trajectory : np.ndarray
            Loss values.

        Returns
        -------
        int
            Number of steps in the catapult phase (0 if none).
        """
        L = np.asarray(loss_trajectory, dtype=np.float64)
        onset = self.catapult_onset(L)
        if onset < 0:
            return 0

        L0 = L[0]
        for i in range(onset, len(L)):
            if L[i] < L0:
                return i - onset
        return len(L) - onset

    # ----- Catapult magnitude ----------------------------------------------

    def catapult_magnitude(self, loss_trajectory: np.ndarray) -> float:
        r"""Peak loss during the catapult relative to initial loss.

        Parameters
        ----------
        loss_trajectory : np.ndarray
            Loss values.

        Returns
        -------
        float
            max(L) / L(0) during catapult.
        """
        L = np.asarray(loss_trajectory, dtype=np.float64)
        onset = self.catapult_onset(L)
        if onset < 0:
            return 1.0
        L0 = L[0]
        if L0 < 1e-14:
            return float(np.max(L))
        return float(np.max(L[onset:]) / L0)

    # ----- Catapult condition ----------------------------------------------

    def catapult_condition(
        self,
        ntk_eigenvalues: np.ndarray,
        lr: float,
    ) -> Dict[str, Any]:
        r"""Check the catapult condition: η λ_max > 2.

        When the product of learning rate and top NTK eigenvalue exceeds 2,
        discrete gradient descent becomes unstable (catapult).

        Parameters
        ----------
        ntk_eigenvalues : np.ndarray
            Eigenvalues of the NTK.
        lr : float
            Learning rate.

        Returns
        -------
        dict
            ``catapult_expected``, ``lr_lambda_max``, ``lambda_max``.
        """
        eigs = np.asarray(ntk_eigenvalues, dtype=np.float64)
        lam_max = float(eigs[-1]) if len(eigs) > 0 else 0.0
        product = lr * lam_max

        return {
            'catapult_expected': product > 2.0,
            'lr_lambda_max': float(product),
            'lambda_max': lam_max,
            'critical_lr': float(2.0 / lam_max) if lam_max > 1e-14 else np.inf,
        }

    # ----- Post-catapult regime --------------------------------------------

    def post_catapult_regime(
        self,
        ntk_before: np.ndarray,
        ntk_after: np.ndarray,
    ) -> Dict[str, Any]:
        r"""Characterize the regime after the catapult.

        Compares the NTK before and after the catapult to determine whether
        training has moved from lazy to rich regime.

        Parameters
        ----------
        ntk_before : np.ndarray
            NTK before catapult.
        ntk_after : np.ndarray
            NTK after catapult.

        Returns
        -------
        dict
            ``kernel_change``, ``regime`` ('lazy' or 'rich'),
            ``spectral_change``.
        """
        K0 = np.asarray(ntk_before, dtype=np.float64)
        K1 = np.asarray(ntk_after, dtype=np.float64)

        norm0 = np.linalg.norm(K0, 'fro')
        rel_change = np.linalg.norm(K1 - K0, 'fro') / max(norm0, 1e-14)

        eigs0 = np.linalg.eigvalsh(K0)
        eigs1 = np.linalg.eigvalsh(K1)
        spectral_change = float(np.linalg.norm(eigs1 - eigs0) / (np.linalg.norm(eigs0) + 1e-14))

        regime = 'rich' if rel_change > 0.1 else 'lazy'

        return {
            'kernel_change': float(rel_change),
            'regime': regime,
            'spectral_change': spectral_change,
        }

    # ----- Catapult phase boundary -----------------------------------------

    def catapult_phase_boundary(
        self,
        width_range: np.ndarray,
        lr_range: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        r"""Compute (width, lr) phase boundary for the catapult.

        The catapult occurs when η λ_max(Θ) > 2.  For random NTKs in the
        infinite-width limit, λ_max ~ O(N/d) giving η_c ~ 2d/N.

        Parameters
        ----------
        width_range : np.ndarray
            Network widths.
        lr_range : np.ndarray
            Learning rates.

        Returns
        -------
        dict
            ``width_grid``, ``lr_grid``, ``catapult_map`` (2D bool array).
        """
        widths = np.asarray(width_range, dtype=np.float64)
        lrs = np.asarray(lr_range, dtype=np.float64)
        catapult_map = np.zeros((len(widths), len(lrs)), dtype=bool)

        for i, N in enumerate(widths):
            # Approximate λ_max ~ c * N for some constant c ~ σ_w²
            lam_max = self.config.depth * N * 0.01  # rough estimate
            for j, lr in enumerate(lrs):
                catapult_map[i, j] = lr * lam_max > 2.0

        return {
            'width_grid': widths,
            'lr_grid': lrs,
            'catapult_map': catapult_map,
        }

    # ----- Catapult as symmetry breaking -----------------------------------

    def catapult_as_symmetry_breaking(
        self,
        weight_trajectory: List[np.ndarray],
    ) -> Dict[str, Any]:
        r"""Analyze the catapult as symmetry breaking of the kernel regime.

        During the catapult, the weight-space trajectory breaks the
        approximate permutation symmetry of the kernel regime.

        Parameters
        ----------
        weight_trajectory : list of np.ndarray
            Weight matrices at successive steps.

        Returns
        -------
        dict
            ``symmetry_breaking_measure``, ``singular_value_change``.
        """
        if len(weight_trajectory) < 2:
            return {'symmetry_breaking_measure': 0.0, 'singular_value_change': 0.0}

        W0 = np.asarray(weight_trajectory[0], dtype=np.float64)
        Wt = np.asarray(weight_trajectory[-1], dtype=np.float64)

        s0 = np.linalg.svd(W0, compute_uv=False)
        st = np.linalg.svd(Wt, compute_uv=False)

        # Symmetry breaking: deviation from uniform singular values
        s0_norm = s0 / (np.sum(s0) + 1e-14)
        st_norm = st / (np.sum(st) + 1e-14)

        # Entropy decrease = symmetry breaking
        H0 = -np.sum(s0_norm * np.log(s0_norm + 1e-300))
        Ht = -np.sum(st_norm * np.log(st_norm + 1e-300))

        n_sv = min(len(s0), len(st))
        sv_change = float(np.linalg.norm(s0[:n_sv] - st[:n_sv]) / (np.linalg.norm(s0[:n_sv]) + 1e-14))

        return {
            'symmetry_breaking_measure': float(H0 - Ht),
            'singular_value_change': sv_change,
        }

    # ----- Progressive sharpening -----------------------------------------

    def progressive_sharpening(
        self,
        hessian_top_eigs: np.ndarray,
        steps: np.ndarray,
    ) -> Dict[str, Any]:
        r"""Track progressive sharpening: λ_max increasing toward 2/η.

        Parameters
        ----------
        hessian_top_eigs : np.ndarray
            Top Hessian eigenvalue at each logged step.
        steps : np.ndarray
            Step numbers.

        Returns
        -------
        dict
            ``sharpness``, ``steps``, ``threshold``, ``time_to_threshold``.
        """
        sharpness = np.asarray(hessian_top_eigs, dtype=np.float64)
        steps = np.asarray(steps, dtype=np.float64)
        threshold = 2.0 / self.config.lr

        # Find when sharpness first reaches threshold
        exceeded = np.where(sharpness >= threshold)[0]
        time_to_threshold = int(steps[exceeded[0]]) if len(exceeded) > 0 else -1

        # Sharpening rate (slope of sharpness vs step)
        if len(steps) > 1:
            coeffs = np.polyfit(steps, sharpness, 1)
            rate = float(coeffs[0])
        else:
            rate = 0.0

        return {
            'sharpness': sharpness,
            'steps': steps,
            'threshold': float(threshold),
            'time_to_threshold': time_to_threshold,
            'sharpening_rate': rate,
        }


# ---------------------------------------------------------------------------
# 3. EdgeOfStabilityAnalysis
# ---------------------------------------------------------------------------

class EdgeOfStabilityAnalysis:
    r"""Analysis of the edge-of-stability phenomenon.

    At the edge of stability (EoS), the top Hessian eigenvalue (sharpness)
    saturates near 2/η, and the loss non-monotonically decreases while
    oscillating.

    Parameters
    ----------
    config : TrainingPhaseConfig
        Configuration.
    """

    def __init__(self, config: TrainingPhaseConfig):
        self.config = config
        self.threshold = 2.0 / config.lr

    # ----- Sharpness trajectory --------------------------------------------

    def sharpness_trajectory(
        self,
        hessian_top_eig_trajectory: np.ndarray,
    ) -> Dict[str, Any]:
        r"""Track sharpness = λ_max of the Hessian over training.

        Parameters
        ----------
        hessian_top_eig_trajectory : np.ndarray
            Top eigenvalue at each step.

        Returns
        -------
        dict
            ``sharpness``, ``mean``, ``std``, ``above_threshold_fraction``.
        """
        s = np.asarray(hessian_top_eig_trajectory, dtype=np.float64)
        above = np.sum(s >= self.threshold) / len(s) if len(s) > 0 else 0.0

        return {
            'sharpness': s,
            'mean': float(np.mean(s)),
            'std': float(np.std(s)),
            'above_threshold_fraction': float(above),
            'threshold': self.threshold,
        }

    # ----- EoS onset -------------------------------------------------------

    def eos_onset(
        self,
        sharpness_trajectory: np.ndarray,
        lr: float,
    ) -> int:
        r"""Find the step when sharpness first reaches 2/η.

        Parameters
        ----------
        sharpness_trajectory : np.ndarray
            Sharpness values.
        lr : float
            Learning rate.

        Returns
        -------
        int
            Step index of EoS onset (-1 if not reached).
        """
        s = np.asarray(sharpness_trajectory, dtype=np.float64)
        threshold = 2.0 / lr
        indices = np.where(s >= threshold)[0]
        return int(indices[0]) if len(indices) > 0 else -1

    # ----- EoS oscillation analysis ----------------------------------------

    def eos_oscillation_analysis(
        self,
        loss_trajectory: np.ndarray,
        sharpness_trajectory: np.ndarray,
    ) -> Dict[str, Any]:
        r"""Characterize oscillations at the edge of stability.

        At EoS, both loss and sharpness oscillate with correlated dynamics.

        Parameters
        ----------
        loss_trajectory : np.ndarray
            Loss values.
        sharpness_trajectory : np.ndarray
            Sharpness values.

        Returns
        -------
        dict
            ``loss_oscillation_amplitude``, ``sharpness_oscillation_amplitude``,
            ``correlation``, ``dominant_frequency``.
        """
        L = np.asarray(loss_trajectory, dtype=np.float64)
        S = np.asarray(sharpness_trajectory, dtype=np.float64)
        n = min(len(L), len(S))
        L, S = L[:n], S[:n]

        if n < 10:
            return {
                'loss_oscillation_amplitude': 0.0,
                'sharpness_oscillation_amplitude': 0.0,
                'correlation': 0.0,
                'dominant_frequency': 0.0,
            }

        # Detrend
        L_detrend = L - np.polyval(np.polyfit(np.arange(n), L, 1), np.arange(n))
        S_detrend = S - np.polyval(np.polyfit(np.arange(n), S, 1), np.arange(n))

        loss_amp = float(np.std(L_detrend))
        sharp_amp = float(np.std(S_detrend))

        # Correlation
        if loss_amp > 1e-14 and sharp_amp > 1e-14:
            corr = float(np.corrcoef(L_detrend, S_detrend)[0, 1])
        else:
            corr = 0.0

        # Dominant frequency via FFT
        fft_L = np.fft.rfft(L_detrend)
        freqs = np.fft.rfftfreq(n)
        power = np.abs(fft_L) ** 2
        if len(power) > 1:
            # Skip DC component
            dom_idx = np.argmax(power[1:]) + 1
            dom_freq = float(freqs[dom_idx])
        else:
            dom_freq = 0.0

        return {
            'loss_oscillation_amplitude': loss_amp,
            'sharpness_oscillation_amplitude': sharp_amp,
            'correlation': corr,
            'dominant_frequency': dom_freq,
        }

    # ----- EoS effective dynamics ------------------------------------------

    def eos_effective_dynamics(
        self,
        trajectory: np.ndarray,
        lr: float,
    ) -> Dict[str, Any]:
        r"""Effective dynamics at the edge of stability.

        At EoS, the discrete update w_{t+1} = w_t - η ∇L differs
        fundamentally from gradient flow, leading to implicit
        regularization toward flatter minima.

        Parameters
        ----------
        trajectory : np.ndarray, shape (T, d)
            Weight trajectory.
        lr : float
            Learning rate.

        Returns
        -------
        dict
            ``effective_step_size``, ``oscillation_period``,
            ``drift_direction``.
        """
        W = np.asarray(trajectory, dtype=np.float64)
        if W.ndim == 1:
            W = W.reshape(-1, 1)
        T, d = W.shape

        if T < 3:
            return {
                'effective_step_size': 0.0,
                'oscillation_period': 0,
                'drift_direction': np.zeros(d),
            }

        # Effective step sizes
        steps = np.linalg.norm(np.diff(W, axis=0), axis=1)
        eff_step = float(np.mean(steps))

        # Drift direction (average displacement)
        total_drift = W[-1] - W[0]
        drift_norm = np.linalg.norm(total_drift)
        drift_dir = total_drift / (drift_norm + 1e-14)

        # Oscillation period from autocorrelation of steps
        if len(steps) > 10:
            acf = np.correlate(steps - np.mean(steps), steps - np.mean(steps), mode='full')
            acf = acf[len(acf) // 2:]
            acf = acf / (acf[0] + 1e-14)
            # First peak after zero crossing
            zero_crossings = np.where(np.diff(np.sign(acf)))[0]
            if len(zero_crossings) >= 2:
                period = int(zero_crossings[1] - zero_crossings[0])
            else:
                period = 0
        else:
            period = 0

        return {
            'effective_step_size': eff_step,
            'oscillation_period': period,
            'drift_direction': drift_dir,
        }

    # ----- EoS phase boundary ---------------------------------------------

    def eos_phase_boundary(
        self,
        width_range: np.ndarray,
        lr_range: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        r"""Compute (width, lr) boundary for the EoS regime.

        EoS requires η λ_max ≈ 2, which depends on both width and lr.

        Parameters
        ----------
        width_range : np.ndarray
            Network widths.
        lr_range : np.ndarray
            Learning rates.

        Returns
        -------
        dict
            ``width_grid``, ``lr_grid``, ``eos_map`` (2D bool array).
        """
        widths = np.asarray(width_range, dtype=np.float64)
        lrs = np.asarray(lr_range, dtype=np.float64)
        eos_map = np.zeros((len(widths), len(lrs)), dtype=bool)

        for i, N in enumerate(widths):
            lam_max = self.config.depth * N * 0.01
            for j, lr in enumerate(lrs):
                product = lr * lam_max
                # EoS when product is close to 2 (within factor 2)
                eos_map[i, j] = 1.0 < product < 4.0

        return {
            'width_grid': widths,
            'lr_grid': lrs,
            'eos_map': eos_map,
        }

    # ----- GD vs continuous GF at EoS --------------------------------------

    def eos_connection_to_gd(
        self,
        continuous_trajectory: np.ndarray,
        discrete_trajectory: np.ndarray,
    ) -> Dict[str, float]:
        r"""Compare gradient descent with continuous gradient flow at EoS.

        Measures how far discrete GD deviates from continuous GF,
        quantifying the role of discretization effects.

        Parameters
        ----------
        continuous_trajectory : np.ndarray, shape (T, d)
            Continuous gradient flow trajectory.
        discrete_trajectory : np.ndarray, shape (T, d)
            Discrete gradient descent trajectory.

        Returns
        -------
        dict
            ``max_deviation``, ``mean_deviation``, ``final_deviation``.
        """
        Wc = np.asarray(continuous_trajectory, dtype=np.float64)
        Wd = np.asarray(discrete_trajectory, dtype=np.float64)
        T = min(len(Wc), len(Wd))
        Wc, Wd = Wc[:T], Wd[:T]

        deviations = np.linalg.norm(Wc - Wd, axis=-1) if Wc.ndim > 1 else np.abs(Wc - Wd)

        return {
            'max_deviation': float(np.max(deviations)),
            'mean_deviation': float(np.mean(deviations)),
            'final_deviation': float(deviations[-1]) if len(deviations) > 0 else 0.0,
        }

    # ----- Sharpness reduction mechanism -----------------------------------

    def sharpness_reduction_mechanism(
        self,
        weight_trajectory: List[np.ndarray],
        hessian_fn: Callable[[np.ndarray], np.ndarray],
    ) -> Dict[str, Any]:
        r"""Analyze how sharpness is reduced at EoS.

        At EoS, gradient descent implicitly moves weights toward regions
        of lower curvature.

        Parameters
        ----------
        weight_trajectory : list of np.ndarray
            Weight vectors at successive steps.
        hessian_fn : callable
            Function w → H(w) returning the Hessian matrix.

        Returns
        -------
        dict
            ``sharpness_values``, ``curvature_reduction_rate``.
        """
        sharpness = []
        for w in weight_trajectory:
            H = hessian_fn(np.asarray(w, dtype=np.float64))
            top_eig = float(np.max(np.linalg.eigvalsh(H)))
            sharpness.append(top_eig)

        sharpness = np.array(sharpness)
        if len(sharpness) > 1:
            rate = float(np.polyfit(np.arange(len(sharpness)), sharpness, 1)[0])
        else:
            rate = 0.0

        return {
            'sharpness_values': sharpness,
            'curvature_reduction_rate': rate,
        }

    # ----- Implicit regularization at EoS ----------------------------------

    def implicit_regularization_at_eos(
        self,
        weight_trajectory: List[np.ndarray],
    ) -> Dict[str, Any]:
        r"""Identify what is implicitly regularized at EoS.

        Tracks the trace of the Hessian (sharpness), weight norm, and
        effective rank as implicit regularization measures.

        Parameters
        ----------
        weight_trajectory : list of np.ndarray
            Weight vectors at successive steps.

        Returns
        -------
        dict
            ``weight_norms``, ``effective_ranks``, ``trends``.
        """
        norms = []
        eff_ranks = []

        for w in weight_trajectory:
            w = np.asarray(w, dtype=np.float64)
            norms.append(float(np.linalg.norm(w)))

            if w.ndim >= 2:
                s = np.linalg.svd(w, compute_uv=False)
                s = s[s > 1e-14]
                if len(s) > 0:
                    p = s / np.sum(s)
                    eff_ranks.append(float(np.exp(-np.sum(p * np.log(p + 1e-300)))))
                else:
                    eff_ranks.append(0.0)
            else:
                eff_ranks.append(float(len(w)))

        norms = np.array(norms)
        eff_ranks = np.array(eff_ranks)

        trends = {}
        if len(norms) > 1:
            trends['norm_trend'] = 'decreasing' if norms[-1] < norms[0] else 'increasing'
            trends['rank_trend'] = 'decreasing' if eff_ranks[-1] < eff_ranks[0] else 'stable'
        else:
            trends['norm_trend'] = 'unknown'
            trends['rank_trend'] = 'unknown'

        return {
            'weight_norms': norms,
            'effective_ranks': eff_ranks,
            'trends': trends,
        }


# ---------------------------------------------------------------------------
# 4. CondensationAnalysis
# ---------------------------------------------------------------------------

class CondensationAnalysis:
    r"""Analysis of weight condensation and neural collapse during training.

    Weight condensation occurs when hidden-layer weight matrices
    become approximately low-rank, concentrating learned features
    into a small subspace.  Neural collapse is the terminal phase
    where features and classifiers converge to a simplex ETF structure.

    Parameters
    ----------
    config : TrainingPhaseConfig
        Configuration.
    """

    def __init__(self, config: TrainingPhaseConfig):
        self.config = config

    # ----- Singular value evolution ----------------------------------------

    def singular_value_evolution(
        self,
        weight_matrices_trajectory: List[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        r"""Track singular value decomposition over training.

        Parameters
        ----------
        weight_matrices_trajectory : list of np.ndarray
            Weight matrices at successive steps.

        Returns
        -------
        dict
            ``singular_values`` (list of arrays), ``steps``.
        """
        sv_list = []
        for W in weight_matrices_trajectory:
            s = np.linalg.svd(np.asarray(W, dtype=np.float64), compute_uv=False)
            sv_list.append(s)

        return {
            'singular_values': sv_list,
            'steps': np.arange(len(sv_list)),
        }

    # ----- Effective rank trajectory ---------------------------------------

    def effective_rank_trajectory(
        self,
        weight_matrices_trajectory: List[np.ndarray],
    ) -> np.ndarray:
        r"""Track effective rank of weight matrices over training.

        Effective rank = exp(spectral entropy).

        Parameters
        ----------
        weight_matrices_trajectory : list of np.ndarray
            Weight matrices at successive steps.

        Returns
        -------
        np.ndarray
            Effective rank at each step.
        """
        ranks = []
        for W in weight_matrices_trajectory:
            s = np.linalg.svd(np.asarray(W, dtype=np.float64), compute_uv=False)
            s = s[s > 1e-14]
            if len(s) == 0:
                ranks.append(0.0)
                continue
            p = s / np.sum(s)
            ranks.append(float(np.exp(-np.sum(p * np.log(p + 1e-300)))))
        return np.array(ranks)

    # ----- Condensation onset ----------------------------------------------

    def condensation_onset(
        self,
        effective_ranks: np.ndarray,
    ) -> int:
        r"""Find when the effective rank starts dropping significantly.

        Parameters
        ----------
        effective_ranks : np.ndarray
            Effective rank trajectory.

        Returns
        -------
        int
            Step index of condensation onset (-1 if not found).
        """
        r = np.asarray(effective_ranks, dtype=np.float64)
        if len(r) < 5:
            return -1

        r0 = r[0]
        threshold = 0.8 * r0  # 20% drop
        for i in range(1, len(r)):
            if r[i] < threshold:
                return i
        return -1

    # ----- Condensed subspace ----------------------------------------------

    def condensed_subspace(
        self,
        weight_matrix: np.ndarray,
        threshold: float = 0.99,
    ) -> Dict[str, Any]:
        r"""Identify the condensed subspace of a weight matrix.

        The condensed subspace contains the top-k singular vectors
        capturing ≥ threshold fraction of total variance.

        Parameters
        ----------
        weight_matrix : np.ndarray
            Weight matrix.
        threshold : float
            Fraction of variance to capture (default 0.99).

        Returns
        -------
        dict
            ``rank``, ``explained_variance``, ``top_directions``.
        """
        W = np.asarray(weight_matrix, dtype=np.float64)
        U, s, Vt = np.linalg.svd(W, full_matrices=False)

        total_var = np.sum(s ** 2)
        if total_var < 1e-14:
            return {'rank': 0, 'explained_variance': 0.0, 'top_directions': np.array([])}

        cumvar = np.cumsum(s ** 2) / total_var
        k = int(np.searchsorted(cumvar, threshold) + 1)
        k = min(k, len(s))

        return {
            'rank': k,
            'explained_variance': float(cumvar[k - 1]) if k > 0 else 0.0,
            'top_directions': Vt[:k],
        }

    # ----- Condensation order parameter ------------------------------------

    def condensation_order_parameter(
        self,
        weight_matrix: np.ndarray,
        threshold: float = 0.01,
    ) -> float:
        r"""Condensation order parameter: fraction of variance in top-k SVs.

        Parameters
        ----------
        weight_matrix : np.ndarray
            Weight matrix.
        threshold : float
            Singular value threshold (relative to max) for defining
            "significant" directions.

        Returns
        -------
        float
            Fraction of variance in significant SVs.
        """
        W = np.asarray(weight_matrix, dtype=np.float64)
        s = np.linalg.svd(W, compute_uv=False)
        if len(s) == 0 or s[0] < 1e-14:
            return 0.0

        mask = s > threshold * s[0]
        return float(np.sum(s[mask] ** 2) / np.sum(s ** 2))

    # ----- Condensation phase boundary -------------------------------------

    def condensation_phase_boundary(
        self,
        width_range: np.ndarray,
        depth_range: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        r"""Compute (width, depth) boundary for condensation.

        Condensation is more likely with greater depth (many layers
        compound rank reduction) and smaller width.

        Parameters
        ----------
        width_range : np.ndarray
            Network widths.
        depth_range : np.ndarray
            Network depths.

        Returns
        -------
        dict
            ``width_grid``, ``depth_grid``, ``condensation_map``.
        """
        widths = np.asarray(width_range, dtype=np.float64)
        depths = np.asarray(depth_range, dtype=np.float64)
        cond_map = np.zeros((len(widths), len(depths)), dtype=bool)

        for i, N in enumerate(widths):
            for j, L in enumerate(depths):
                # Heuristic: condensation when depth/width ratio is large
                ratio = L / N
                cond_map[i, j] = ratio > 0.5

        return {
            'width_grid': widths,
            'depth_grid': depths,
            'condensation_map': cond_map,
        }

    # ----- Neural collapse analysis ----------------------------------------

    def neural_collapse_analysis(
        self,
        feature_matrices: List[np.ndarray],
        labels: np.ndarray,
    ) -> Dict[str, Any]:
        r"""Analyze neural collapse (NC) in terminal phase of training.

        NC has four properties:
        1. Within-class variability → 0 (NC1)
        2. Class means form a simplex ETF (NC2)
        3. Classifier aligns with class means (NC3)
        4. Nearest class center becomes optimal (NC4)

        Parameters
        ----------
        feature_matrices : list of np.ndarray
            Feature matrices at successive training steps.
        labels : np.ndarray
            Class labels.

        Returns
        -------
        dict
            ``nc1_variability``, ``nc2_etf_distance``, ``class_means``.
        """
        labels = np.asarray(labels, dtype=int)
        classes = np.unique(labels)
        K = len(classes)

        results = []
        for features in feature_matrices:
            H = np.asarray(features, dtype=np.float64)

            # Compute class means
            class_means = np.array([
                np.mean(H[labels == c], axis=0) for c in classes
            ])
            global_mean = np.mean(H, axis=0)

            # NC1: within-class variability
            Sw = np.zeros((H.shape[1], H.shape[1]))
            for c in classes:
                H_c = H[labels == c] - class_means[np.where(classes == c)[0][0]]
                Sw += H_c.T @ H_c
            Sw /= len(labels)

            # Between-class variability
            centered_means = class_means - global_mean
            Sb = centered_means.T @ centered_means / K

            # NC1 metric: tr(Sw Sb^{-1}) / K  (should → 0)
            try:
                nc1 = float(np.trace(Sw @ np.linalg.pinv(Sb)) / K)
            except np.linalg.LinAlgError:
                nc1 = np.inf

            # NC2: distance to simplex ETF
            nc2 = self.simplex_etf_distance(class_means, global_mean)

            results.append({
                'nc1_variability': nc1,
                'nc2_etf_distance': nc2,
            })

        return {
            'nc1_variability': [r['nc1_variability'] for r in results],
            'nc2_etf_distance': [r['nc2_etf_distance'] for r in results],
            'class_means': class_means if len(feature_matrices) > 0 else None,
            'n_classes': K,
        }

    # ----- Simplex ETF distance --------------------------------------------

    def simplex_etf_distance(
        self,
        class_means: np.ndarray,
        global_mean: np.ndarray,
    ) -> float:
        r"""Distance from class means to a simplex equiangular tight frame.

        The simplex ETF has the property that the cosine similarity between
        any two centered class means equals -1/(K-1).

        Parameters
        ----------
        class_means : np.ndarray, shape (K, d)
            Class mean vectors.
        global_mean : np.ndarray, shape (d,)
            Global mean vector.

        Returns
        -------
        float
            Frobenius distance to the ideal ETF Gram matrix.
        """
        K = class_means.shape[0]
        centered = class_means - global_mean

        # Normalize
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-14)
        normalized = centered / norms

        # Gram matrix of cosine similarities
        G = normalized @ normalized.T

        # Ideal ETF Gram matrix: I - (1/(K-1)) (I - (1/K) 11^T)
        # Simplifies to: G_ideal_{ij} = δ_{ij} + (1-δ_{ij}) × (-1/(K-1))
        if K <= 1:
            return 0.0
        G_ideal = np.eye(K) - np.ones((K, K)) / (K - 1)
        np.fill_diagonal(G_ideal, 1.0)

        return float(np.linalg.norm(G - G_ideal, 'fro') / K)


# ---------------------------------------------------------------------------
# 5. DoubleDescentAnalysis
# ---------------------------------------------------------------------------

class DoubleDescentAnalysis:
    r"""Analysis of the double descent phenomenon.

    Double descent occurs when test error exhibits a second descent after
    the interpolation threshold where training error first reaches zero.
    This class provides tools for detecting and explaining this phenomenon.

    Parameters
    ----------
    config : TrainingPhaseConfig
        Configuration.
    """

    def __init__(self, config: TrainingPhaseConfig):
        self.config = config

    # ----- Interpolation threshold -----------------------------------------

    def interpolation_threshold(
        self,
        train_loss_vs_params: Dict[int, float],
    ) -> Optional[int]:
        r"""Find N* where training loss first drops to ≈ 0.

        Parameters
        ----------
        train_loss_vs_params : dict
            Mapping {n_params: train_loss}.

        Returns
        -------
        int or None
            Interpolation threshold N*.
        """
        items = sorted(train_loss_vs_params.items())
        for n_params, loss in items:
            if loss < 1e-6:
                return n_params
        return None

    # ----- Test loss curve -------------------------------------------------

    def test_loss_curve(
        self,
        n_params_range: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        r"""Compute test loss vs. model size (number of parameters).

        Fits kernel ridge regression with NTK-like kernels of varying
        effective rank to simulate varying model complexity.

        Parameters
        ----------
        n_params_range : np.ndarray
            Array of model sizes (number of parameters).
        X_train, y_train : np.ndarray
            Training data.
        X_test, y_test : np.ndarray
            Test data.

        Returns
        -------
        dict
            ``n_params``, ``train_loss``, ``test_loss``.
        """
        n_params_arr = np.asarray(n_params_range, dtype=int)
        n_train = X_train.shape[0]
        d = X_train.shape[1]

        train_losses = np.empty(len(n_params_arr))
        test_losses = np.empty(len(n_params_arr))

        for i, p in enumerate(n_params_arr):
            # Simulate model of complexity p by using rank-p random features
            rng = np.random.default_rng(42)
            W = rng.normal(size=(d, p)) / np.sqrt(p)

            Phi_train = X_train @ W
            if self.config.activation == 'relu':
                Phi_train = np.maximum(Phi_train, 0)
            elif self.config.activation == 'tanh':
                Phi_train = np.tanh(Phi_train)

            Phi_test = X_test @ W
            if self.config.activation == 'relu':
                Phi_test = np.maximum(Phi_test, 0)
            elif self.config.activation == 'tanh':
                Phi_test = np.tanh(Phi_test)

            # Ridge regression with small regularization
            lam = 1e-6
            K = Phi_train @ Phi_train.T + lam * np.eye(n_train)
            try:
                alpha = np.linalg.solve(K, y_train)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, y_train, rcond=None)[0]

            y_pred_train = Phi_train @ (Phi_train.T @ alpha)
            y_pred_test = Phi_test @ (Phi_train.T @ alpha)

            train_losses[i] = float(np.mean((y_pred_train - y_train) ** 2))
            test_losses[i] = float(np.mean((y_pred_test - y_test) ** 2))

        return {
            'n_params': n_params_arr,
            'train_loss': train_losses,
            'test_loss': test_losses,
        }

    # ----- Epoch-wise double descent ---------------------------------------

    def epoch_wise_double_descent(
        self,
        test_losses_by_epoch: np.ndarray,
        n_params: int,
    ) -> Dict[str, Any]:
        r"""Detect epoch-wise double descent.

        Test loss may first decrease, then increase (overfitting),
        then decrease again (second descent) as training continues.

        Parameters
        ----------
        test_losses_by_epoch : np.ndarray
            Test loss at each epoch.
        n_params : int
            Number of model parameters.

        Returns
        -------
        dict
            ``has_double_descent``, ``first_minimum``, ``peak``,
            ``second_minimum``.
        """
        L = np.asarray(test_losses_by_epoch, dtype=np.float64)
        if len(L) < 10:
            return {
                'has_double_descent': False,
                'first_minimum': None,
                'peak': None,
                'second_minimum': None,
            }

        # Find local minima and maxima
        minima = []
        maxima = []
        for i in range(1, len(L) - 1):
            if L[i] < L[i - 1] and L[i] < L[i + 1]:
                minima.append(i)
            if L[i] > L[i - 1] and L[i] > L[i + 1]:
                maxima.append(i)

        has_dd = (len(minima) >= 2 and len(maxima) >= 1
                  and maxima[0] > minima[0])

        return {
            'has_double_descent': has_dd,
            'first_minimum': int(minima[0]) if len(minima) > 0 else None,
            'peak': int(maxima[0]) if len(maxima) > 0 else None,
            'second_minimum': int(minima[1]) if len(minima) > 1 else None,
            'n_params': n_params,
        }

    # ----- Spectral explanation --------------------------------------------

    def spectral_explanation(
        self,
        ntk_eigenvalues: np.ndarray,
        n_train: int,
        noise_var: float,
    ) -> Dict[str, Any]:
        r"""Explain double descent from the NTK spectrum.

        The peak in test error near n = p occurs because the smallest
        eigenvalue of K approaches zero, amplifying noise.

        Parameters
        ----------
        ntk_eigenvalues : np.ndarray
            NTK eigenvalues.
        n_train : int
            Number of training samples.
        noise_var : float
            Noise variance.

        Returns
        -------
        dict
            ``condition_number``, ``min_eigenvalue``,
            ``noise_amplification``, ``explanation``.
        """
        eigs = np.sort(np.asarray(ntk_eigenvalues, dtype=np.float64))
        pos_eigs = eigs[eigs > 1e-14]

        if len(pos_eigs) == 0:
            return {
                'condition_number': np.inf,
                'min_eigenvalue': 0.0,
                'noise_amplification': np.inf,
                'explanation': 'Singular kernel — at interpolation threshold.',
            }

        kappa = float(pos_eigs[-1] / pos_eigs[0])
        noise_amp = float(noise_var * np.sum(1.0 / pos_eigs ** 2) / n_train)

        if kappa > 1e6:
            expl = 'Near interpolation threshold: condition number very large.'
        elif kappa > 100:
            expl = 'Moderately ill-conditioned: noise amplification significant.'
        else:
            expl = 'Well-conditioned: away from interpolation threshold.'

        return {
            'condition_number': kappa,
            'min_eigenvalue': float(pos_eigs[0]),
            'noise_amplification': noise_amp,
            'explanation': expl,
        }

    # ----- Bias-variance decomposition -------------------------------------

    def bias_variance_decomposition(
        self,
        predictions_by_init: np.ndarray,
        y_true: np.ndarray,
    ) -> Dict[str, float]:
        r"""Decompose test error into bias² + variance.

        E[(f̂ - y)²] = (E[f̂] - y)² + Var[f̂] + σ²_noise

        Parameters
        ----------
        predictions_by_init : np.ndarray, shape (n_inits, n_test)
            Predictions from different random initializations.
        y_true : np.ndarray, shape (n_test,)
            True test labels.

        Returns
        -------
        dict
            ``bias_squared``, ``variance``, ``total_error``.
        """
        preds = np.asarray(predictions_by_init, dtype=np.float64)
        y = np.asarray(y_true, dtype=np.float64).ravel()

        mean_pred = np.mean(preds, axis=0)
        bias_sq = float(np.mean((mean_pred - y) ** 2))
        variance = float(np.mean(np.var(preds, axis=0)))
        total = bias_sq + variance

        return {
            'bias_squared': bias_sq,
            'variance': variance,
            'total_error': total,
        }

    # ----- Optimal regularization ------------------------------------------

    def optimal_regularization(
        self,
        ntk_eigenvalues: np.ndarray,
        n_train: int,
        noise_var: float,
    ) -> float:
        r"""Optimal ridge regularization λ* from the NTK spectrum.

        λ* minimizes the expected test error
            R(λ) = bias²(λ) + variance(λ)

        For kernel ridge regression with eigenvalues {λ_k}:
            R(λ) = Σ_k [λ²/(λ_k+λ)² c_k² + σ² λ_k/(λ_k+λ)²/n]

        Parameters
        ----------
        ntk_eigenvalues : np.ndarray
            NTK eigenvalues.
        n_train : int
            Training set size.
        noise_var : float
            Noise variance.

        Returns
        -------
        float
            Optimal ridge parameter λ*.
        """
        eigs = np.asarray(ntk_eigenvalues, dtype=np.float64)
        eigs = eigs[eigs > 1e-14]

        if len(eigs) == 0:
            return 1.0

        def _risk(log_lam):
            lam = np.exp(log_lam)
            # Variance component
            var = noise_var * np.sum(eigs / (eigs + lam) ** 2) / n_train
            # Bias component (assume signal ~ eigenvalue strength)
            bias = np.sum(lam ** 2 / (eigs + lam) ** 2 * eigs)
            return var + bias

        result = optimize.minimize_scalar(
            _risk,
            bounds=(np.log(1e-8), np.log(1e4)),
            method='bounded',
        )
        return float(np.exp(result.x))

    # ----- Benign overfitting condition ------------------------------------

    def benign_overfitting_condition(
        self,
        ntk_eigenvalues: np.ndarray,
    ) -> Dict[str, Any]:
        r"""Check conditions for benign overfitting.

        Benign overfitting occurs when:
        1. The effective rank of the kernel is large (many small eigenvalues)
        2. The signal lies mostly in the top eigenspace
        3. The tail eigenvalues decay slowly enough to absorb noise

        The condition requires Σ_{k>k*} λ_k >> k* σ² for benign interpolation.

        Parameters
        ----------
        ntk_eigenvalues : np.ndarray
            NTK eigenvalues.

        Returns
        -------
        dict
            ``is_benign``, ``effective_rank``, ``tail_sum``,
            ``spectral_decay_rate``.
        """
        eigs = np.sort(np.asarray(ntk_eigenvalues, dtype=np.float64))[::-1]
        N = len(eigs)

        if N < 5:
            return {
                'is_benign': False,
                'effective_rank': float(N),
                'tail_sum': 0.0,
                'spectral_decay_rate': 0.0,
            }

        # Effective rank
        pos_eigs = eigs[eigs > 1e-14]
        if len(pos_eigs) == 0:
            return {
                'is_benign': False,
                'effective_rank': 0.0,
                'tail_sum': 0.0,
                'spectral_decay_rate': 0.0,
            }
        p = pos_eigs / np.sum(pos_eigs)
        eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-300))))

        # Tail sum (eigenvalues beyond effective rank)
        k_star = max(int(eff_rank), 1)
        tail_sum = float(np.sum(eigs[k_star:])) if k_star < N else 0.0

        # Spectral decay rate
        log_eigs = np.log(pos_eigs + 1e-14)
        log_idx = np.log(np.arange(1, len(pos_eigs) + 1))
        if len(log_eigs) > 2:
            decay_rate = -float(np.polyfit(log_idx, log_eigs, 1)[0])
        else:
            decay_rate = 0.0

        # Benign overfitting heuristic: slow decay (rate < 1) + high eff rank
        is_benign = decay_rate < 1.0 and eff_rank > 0.3 * N

        return {
            'is_benign': is_benign,
            'effective_rank': eff_rank,
            'tail_sum': tail_sum,
            'spectral_decay_rate': decay_rate,
        }

    # ----- (n, p) phase diagram --------------------------------------------

    def phase_diagram_n_p(
        self,
        n_range: np.ndarray,
        p_range: np.ndarray,
        X_fn: Callable[[int], np.ndarray],
        y_fn: Callable[[int, np.ndarray], np.ndarray],
    ) -> Dict[str, np.ndarray]:
        r"""Compute (n, p) phase diagram for double descent.

        Classifies each (n_train, n_params) point as:
        - underparameterized (p < n): classical bias-variance
        - interpolation threshold (p ≈ n): peak error
        - overparameterized (p > n): potential benign overfitting

        Parameters
        ----------
        n_range : np.ndarray
            Training set sizes.
        p_range : np.ndarray
            Number of parameters.
        X_fn : callable
            Function n → X generating n data points.
        y_fn : callable
            Function (n, X) → y generating targets.

        Returns
        -------
        dict
            ``n_grid``, ``p_grid``, ``phase_labels``, ``test_errors``.
        """
        ns = np.asarray(n_range, dtype=int)
        ps = np.asarray(p_range, dtype=int)
        phase_labels = np.empty((len(ns), len(ps)), dtype=int)
        test_errors = np.empty((len(ns), len(ps)))

        for i, n in enumerate(ns):
            X = X_fn(int(n))
            y = y_fn(int(n), X)

            for j, p in enumerate(ps):
                ratio = p / max(n, 1)
                if ratio < 0.8:
                    phase_labels[i, j] = 0  # underparameterized
                elif ratio < 1.2:
                    phase_labels[i, j] = 1  # interpolation threshold
                else:
                    phase_labels[i, j] = 2  # overparameterized

                # Rough error estimate from ratio
                if ratio < 1:
                    # Bias-dominated
                    test_errors[i, j] = (1 - ratio) ** 2
                elif abs(ratio - 1) < 0.1:
                    # Peak
                    test_errors[i, j] = 10.0 / (abs(ratio - 1) + 0.01)
                else:
                    # Overparameterized: decreasing with p
                    test_errors[i, j] = 1.0 / ratio

        return {
            'n_grid': ns,
            'p_grid': ps,
            'phase_labels': phase_labels,
            'test_errors': test_errors,
        }
