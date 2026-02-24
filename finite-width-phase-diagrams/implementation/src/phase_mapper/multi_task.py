"""
Multi-task phase diagrams and transfer learning phases.

Analyzes how multiple tasks interact through shared neural network representations,
identifying phase boundaries between cooperative and competitive regimes, and
characterizing transfer learning dynamics in terms of NTK overlap and spectral geometry.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize
from scipy.linalg import eigh, svd
from scipy.spatial.distance import cosine


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultiTaskConfig:
    """Configuration for multi-task phase diagram analysis.

    Attributes:
        n_tasks: Number of tasks sharing the network.
        task_similarities: (n_tasks, n_tasks) symmetric matrix of pairwise task
            similarities in [0, 1].  Diagonal entries are 1.
        task_weights: Per-task loss weights (length n_tasks).
        width: Network hidden-layer width N.
        depth: Network depth L.
        activation: Activation function identifier (e.g. ``'relu'``, ``'tanh'``).
    """

    n_tasks: int = 2
    task_similarities: np.ndarray = field(default_factory=lambda: np.eye(2))
    task_weights: np.ndarray = field(default_factory=lambda: np.ones(2) / 2)
    width: int = 256
    depth: int = 3
    activation: str = "relu"


# ---------------------------------------------------------------------------
# Task Interference Analyzer
# ---------------------------------------------------------------------------

class TaskInterferenceAnalyzer:
    """Quantifies interference and alignment between tasks via NTK and gradient geometry.

    In the NTK picture every task induces a kernel K_i.  Interference between
    tasks i and j is governed by the overlap of their NTK matrices and the
    alignment of the per-task gradients in parameter space.
    """

    def __init__(self, config: MultiTaskConfig) -> None:
        """Initialise with a multi-task configuration.

        Args:
            config: A ``MultiTaskConfig`` instance describing the multi-task setup.
        """
        self.config = config

    # -- pairwise NTK overlap --------------------------------------------------

    def compute_task_ntk_overlap(
        self, ntk_task_i: np.ndarray, ntk_task_j: np.ndarray
    ) -> float:
        """Normalised trace overlap between two task NTK matrices.

        .. math::
            \\text{overlap} = \\frac{\\operatorname{tr}(K_i K_j)}
                              {\\|K_i\\|_F \\, \\|K_j\\|_F}

        Args:
            ntk_task_i: (n_i, n_i) NTK matrix for task i.
            ntk_task_j: (n_j, n_j) NTK matrix for task j.  Must be the same
                shape as *ntk_task_i* (evaluated on shared data or aligned).

        Returns:
            Scalar overlap in [-1, 1].
        """
        numerator = np.trace(ntk_task_i @ ntk_task_j)
        norm_i = np.linalg.norm(ntk_task_i, "fro")
        norm_j = np.linalg.norm(ntk_task_j, "fro")
        if norm_i < 1e-12 or norm_j < 1e-12:
            return 0.0
        return float(numerator / (norm_i * norm_j))

    # -- gradient alignment ----------------------------------------------------

    def task_gradient_alignment(
        self, grad_task_i: np.ndarray, grad_task_j: np.ndarray
    ) -> float:
        """Cosine similarity between two task gradient vectors.

        Args:
            grad_task_i: Flattened gradient vector for task i (length P).
            grad_task_j: Flattened gradient vector for task j (length P).

        Returns:
            Cosine similarity in [-1, 1].
        """
        gi = grad_task_i.ravel()
        gj = grad_task_j.ravel()
        norm_i = np.linalg.norm(gi)
        norm_j = np.linalg.norm(gj)
        if norm_i < 1e-12 or norm_j < 1e-12:
            return 0.0
        return float(np.dot(gi, gj) / (norm_i * norm_j))

    # -- full interference matrix ----------------------------------------------

    def interference_matrix(self, task_ntks: List[np.ndarray]) -> np.ndarray:
        """Build the full pairwise interference matrix.

        ``M[i, j]`` is the NTK overlap between tasks i and j.  Positive values
        indicate beneficial (cooperative) interaction; negative values indicate
        harmful (competitive) interaction.

        Args:
            task_ntks: List of (n, n) NTK matrices, one per task.

        Returns:
            (n_tasks, n_tasks) interference matrix.
        """
        n = len(task_ntks)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                overlap = self.compute_task_ntk_overlap(task_ntks[i], task_ntks[j])
                M[i, j] = overlap
                M[j, i] = overlap
        return M

    # -- filter positive / negative transfer pairs -----------------------------

    def positive_transfer_pairs(
        self, interference_mat: np.ndarray
    ) -> List[Tuple[int, int, float]]:
        """Return task pairs with positive (beneficial) interference.

        Args:
            interference_mat: (n_tasks, n_tasks) interference matrix.

        Returns:
            List of ``(i, j, M_ij)`` for every upper-triangular pair where
            ``M_ij > 0``.
        """
        pairs: List[Tuple[int, int, float]] = []
        n = interference_mat.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if interference_mat[i, j] > 0:
                    pairs.append((i, j, float(interference_mat[i, j])))
        return pairs

    def negative_transfer_pairs(
        self, interference_mat: np.ndarray
    ) -> List[Tuple[int, int, float]]:
        """Return task pairs with negative (harmful) interference.

        Args:
            interference_mat: (n_tasks, n_tasks) interference matrix.

        Returns:
            List of ``(i, j, M_ij)`` where ``M_ij < 0``.
        """
        pairs: List[Tuple[int, int, float]] = []
        n = interference_mat.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if interference_mat[i, j] < 0:
                    pairs.append((i, j, float(interference_mat[i, j])))
        return pairs

    # -- aggregated interference -----------------------------------------------

    def total_interference(
        self, interference_mat: np.ndarray, weights: np.ndarray
    ) -> float:
        """Weighted sum of pairwise interference values.

        .. math::
            I_{\\text{total}} = \\sum_{i < j} w_i w_j M_{ij}

        Args:
            interference_mat: (n_tasks, n_tasks) interference matrix.
            weights: Per-task weights (length n_tasks).

        Returns:
            Scalar total interference.
        """
        n = interference_mat.shape[0]
        total = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                total += weights[i] * weights[j] * interference_mat[i, j]
        return float(total)

    # -- multi-gradient conflict -----------------------------------------------

    def gradient_conflict_angle(self, gradients: List[np.ndarray]) -> np.ndarray:
        """Pairwise cosine-similarity matrix among multiple task gradients.

        A negative entry ``(i, j)`` signals a gradient conflict between tasks
        i and j: a step that reduces loss for task i increases it for task j.

        Args:
            gradients: List of P-dimensional gradient vectors, one per task.

        Returns:
            (n_tasks, n_tasks) matrix of cosine similarities.
        """
        n = len(gradients)
        cos_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                c = self.task_gradient_alignment(gradients[i], gradients[j])
                cos_mat[i, j] = c
                cos_mat[j, i] = c
        return cos_mat

    # -- Pareto stationarity ---------------------------------------------------

    def pareto_stationarity(
        self, gradients: List[np.ndarray], weights: np.ndarray
    ) -> Tuple[bool, float]:
        """Check approximate Pareto stationarity of a multi-objective point.

        A point is Pareto stationary if there exist non-negative coefficients
        α_i such that Σ α_i g_i = 0.  We solve the minimum-norm element in
        the convex hull of the gradients via a small QP.

        Args:
            gradients: List of P-dimensional task gradient vectors.
            weights: Current task weights (used as initial guess).

        Returns:
            ``(is_stationary, residual_norm)`` — *is_stationary* is ``True``
            when the residual norm is below ``1e-6``.
        """
        G = np.stack([g.ravel() for g in gradients])  # (T, P)
        T = G.shape[0]

        # Minimise ||G^T alpha||^2 s.t. alpha >= 0, sum(alpha) = 1
        GGT = G @ G.T  # (T, T)

        def objective(alpha: np.ndarray) -> float:
            return float(alpha @ GGT @ alpha)

        def grad_obj(alpha: np.ndarray) -> np.ndarray:
            return 2.0 * GGT @ alpha

        constraints = [{"type": "eq", "fun": lambda a: np.sum(a) - 1.0}]
        bounds = [(0.0, None)] * T
        alpha0 = weights / weights.sum()
        res = optimize.minimize(
            objective, alpha0, jac=grad_obj, bounds=bounds, constraints=constraints,
            method="SLSQP",
        )
        residual_norm = float(np.sqrt(max(res.fun, 0.0)))
        return residual_norm < 1e-6, residual_norm

    # -- task dominance --------------------------------------------------------

    def task_dominance_score(
        self, ntk_eigenvalues_per_task: List[np.ndarray]
    ) -> np.ndarray:
        """Score indicating how much each task dominates the shared NTK spectrum.

        The score for task t is the fraction of total spectral mass (sum of
        eigenvalues) belonging to that task.

        Args:
            ntk_eigenvalues_per_task: List of 1-D arrays of eigenvalues, one
                per task.

        Returns:
            (n_tasks,) array of dominance scores summing to 1.
        """
        masses = np.array([np.sum(np.abs(ev)) for ev in ntk_eigenvalues_per_task])
        total = masses.sum()
        if total < 1e-15:
            return np.ones(len(masses)) / len(masses)
        return masses / total


# ---------------------------------------------------------------------------
# Multi-Task Phase Boundary
# ---------------------------------------------------------------------------

class MultiTaskPhaseBoundary:
    """Locates phase boundaries in the (similarity, width) plane for multi-task learning.

    The key insight is that at large width (lazy / NTK regime) tasks decouple
    because the kernel is fixed at initialisation, whereas at finite width
    (rich / feature-learning regime) tasks interact through shared learned
    representations.  The boundary between cooperative and competitive
    regimes depends on the similarity between tasks and the network width.
    """

    def __init__(self, config: MultiTaskConfig) -> None:
        self.config = config
        self._analyzer = TaskInterferenceAnalyzer(config)

    # -- cooperative / competitive boundary ------------------------------------

    def cooperative_competitive_boundary(
        self,
        similarity_range: np.ndarray,
        width_range: np.ndarray,
    ) -> np.ndarray:
        """Phase boundary separating cooperative and competitive regimes.

        For each width N we estimate the critical similarity s*(N) above which
        multi-task training is cooperative (positive transfer) and below which
        it is competitive (negative transfer).

        The critical similarity scales as  s* ~ 1 / sqrt(N)  in the simplest
        mean-field model.

        Args:
            similarity_range: 1-D array of similarity values to probe.
            width_range: 1-D array of widths to probe.

        Returns:
            (len(width_range),) array of critical similarity thresholds.
        """
        critical = np.zeros(len(width_range))
        for idx, N in enumerate(width_range):
            # Mean-field estimate: interference ~ (1-s) - s / sqrt(N)
            # Boundary at interference = 0 → s* = 1 / (1 + 1/sqrt(N))
            critical[idx] = 1.0 / (1.0 + 1.0 / np.sqrt(max(N, 1)))
        return critical

    # -- optimal task weighting ------------------------------------------------

    def optimal_task_weighting(self, task_ntks: List[np.ndarray]) -> np.ndarray:
        """Find task weights that minimise total pairwise interference.

        Solves  min_w Σ_{i<j} w_i w_j M_{ij}  subject to  w >= 0, Σ w_i = 1.

        Args:
            task_ntks: List of NTK matrices, one per task.

        Returns:
            (n_tasks,) optimal weight vector.
        """
        M = self._analyzer.interference_matrix(task_ntks)
        T = M.shape[0]

        def obj(w: np.ndarray) -> float:
            return self._analyzer.total_interference(M, w)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, None)] * T
        w0 = np.ones(T) / T
        res = optimize.minimize(obj, w0, bounds=bounds, constraints=constraints,
                                method="SLSQP")
        return res.x / res.x.sum()

    # -- spectral task analysis ------------------------------------------------

    def spectral_task_analysis(
        self,
        combined_ntk: np.ndarray,
        per_task_ntks: List[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Spectral view of how tasks share the combined NTK eigenspace.

        For each task, we compute the projection of its NTK onto the
        eigenvectors of the combined NTK and measure the alignment with each
        eigenmode.

        Args:
            combined_ntk: (n, n) NTK of the multi-task model.
            per_task_ntks: List of (n, n) per-task NTK matrices.

        Returns:
            Dictionary with keys ``'eigenvalues'``, ``'per_task_projections'``
            (list of 1-D arrays of projection weights), and
            ``'alignment_scores'`` (list of scalars).
        """
        eigenvalues, eigenvectors = eigh(combined_ntk)
        idx_sort = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx_sort]
        eigenvectors = eigenvectors[:, idx_sort]

        projections: List[np.ndarray] = []
        alignment_scores: List[float] = []
        for K_t in per_task_ntks:
            proj = np.array([
                float(v @ K_t @ v) for v in eigenvectors.T
            ])
            projections.append(proj)
            score = float(np.sum(proj * eigenvalues) / (
                np.linalg.norm(proj) * np.linalg.norm(eigenvalues) + 1e-15
            ))
            alignment_scores.append(score)

        return {
            "eigenvalues": eigenvalues,
            "per_task_projections": projections,
            "alignment_scores": np.array(alignment_scores),
        }

    # -- (similarity, width) phase diagram -------------------------------------

    def phase_diagram_similarity_width(
        self,
        similarities: np.ndarray,
        widths: np.ndarray,
        X_tasks: List[np.ndarray],
        y_tasks: List[np.ndarray],
    ) -> np.ndarray:
        """Compute a grid of interference values over (similarity, width).

        For each (s, N) point we synthesise NTK matrices with the prescribed
        similarity (using a simple rank-1 perturbation model) and compute the
        total interference.

        Args:
            similarities: 1-D array of similarity values.
            widths: 1-D array of widths.
            X_tasks: List of (n_samples, d) input arrays, one per task.
            y_tasks: List of (n_samples,) target arrays, one per task.

        Returns:
            (len(similarities), len(widths)) matrix of interference values.
        """
        n_tasks = len(X_tasks)
        grid = np.zeros((len(similarities), len(widths)))

        for si, s in enumerate(similarities):
            for wi, N in enumerate(widths):
                # Construct synthetic NTKs: K_t = base_K + s * shared + (1-s) * task_specific
                d = X_tasks[0].shape[1]
                base = np.eye(X_tasks[0].shape[0]) * N
                shared_part = X_tasks[0] @ X_tasks[0].T / d
                ntks = []
                for t in range(n_tasks):
                    task_part = X_tasks[t] @ X_tasks[t].T / d
                    K_t = base + s * shared_part + (1 - s) * task_part
                    ntks.append(K_t)
                M = self._analyzer.interference_matrix(ntks)
                w = self.config.task_weights[:n_tasks]
                if len(w) < n_tasks:
                    w = np.ones(n_tasks) / n_tasks
                grid[si, wi] = self._analyzer.total_interference(M, w)

        return grid

    # -- critical similarity threshold -----------------------------------------

    def critical_similarity_threshold(self, width: int) -> float:
        """Similarity below which tasks compete at a given width.

        Uses the mean-field scaling  s* = 1 / (1 + 1/sqrt(N)).

        Args:
            width: Network width N.

        Returns:
            Critical similarity s* in (0, 1).
        """
        return 1.0 / (1.0 + 1.0 / np.sqrt(max(width, 1)))

    # -- task capacity ---------------------------------------------------------

    def task_capacity(
        self, width: int, n_tasks_range: np.ndarray
    ) -> np.ndarray:
        """Maximum-cooperation score as a function of the number of tasks.

        At a given width N the effective per-task capacity drops as
        ~ N / T (where T is the number of tasks).  The cooperation score
        is positive when per-task capacity exceeds a threshold.

        Args:
            width: Network width N.
            n_tasks_range: 1-D array of task counts to evaluate.

        Returns:
            (len(n_tasks_range),) array of cooperation scores (positive means
            cooperative, negative means overloaded).
        """
        scores = np.zeros(len(n_tasks_range))
        for i, T in enumerate(n_tasks_range):
            per_task_cap = width / max(T, 1)
            # Threshold is depth-dependent
            threshold = self.config.depth * np.log(self.config.depth + 1)
            scores[i] = per_task_cap - threshold
        return scores

    # -- interference scaling with width ---------------------------------------

    def interference_scaling_with_width(
        self, widths: np.ndarray, task_ntks: List[np.ndarray]
    ) -> np.ndarray:
        """How total interference scales with network width.

        We rescale NTK matrices as K(N) = (N_0/N) * K_0 (mean-field scaling)
        and evaluate the interference at each width.

        Args:
            widths: 1-D array of widths.
            task_ntks: Baseline NTK matrices (at width N_0 = config.width).

        Returns:
            (len(widths),) array of total interference values.
        """
        N0 = self.config.width
        interf = np.zeros(len(widths))
        for idx, N in enumerate(widths):
            scale = N0 / max(N, 1)
            scaled = [K * scale for K in task_ntks]
            M = self._analyzer.interference_matrix(scaled)
            interf[idx] = self._analyzer.total_interference(
                M, self.config.task_weights[: len(task_ntks)]
            )
        return interf

    # -- lazy regime: tasks independent ----------------------------------------

    def lazy_regime_task_independence(
        self, width: int, task_ntks: List[np.ndarray]
    ) -> Dict[str, float]:
        """Quantify task independence in the lazy (NTK) regime.

        In the infinite-width lazy regime the NTK is fixed, so tasks are
        essentially independent — each task is solved in its own function-space
        RKHS.  At finite width, residual coupling is O(1/N).

        Args:
            width: Network width N.
            task_ntks: Per-task NTK matrices (at reference width).

        Returns:
            Dictionary with ``'independence_score'`` in [0,1] (1 = fully
            independent) and ``'coupling_strength'`` ≈ O(1/N).
        """
        M = self._analyzer.interference_matrix(task_ntks)
        off_diag = M - np.diag(np.diag(M))
        coupling = float(np.mean(np.abs(off_diag)))
        # In the lazy regime coupling scales as 1/width
        lazy_coupling = coupling / max(width, 1) * self.config.width
        independence = max(0.0, 1.0 - lazy_coupling)
        return {
            "independence_score": independence,
            "coupling_strength": lazy_coupling,
        }

    # -- rich regime: tasks interact through features --------------------------

    def rich_regime_task_interaction(
        self,
        width: int,
        task_ntks: List[np.ndarray],
        lr: float,
        steps: int,
    ) -> Dict[str, float]:
        """Estimate task interaction strength in the rich (feature-learning) regime.

        In the rich regime the NTK evolves during training, causing tasks to
        interact through shared feature updates.  The interaction strength
        scales as  ~ lr * steps / sqrt(N).

        Args:
            width: Network width N.
            task_ntks: Per-task NTK matrices.
            lr: Learning rate η.
            steps: Number of training steps.

        Returns:
            Dictionary with ``'interaction_strength'``, ``'feature_change'``,
            and ``'is_cooperative'`` flag.
        """
        M = self._analyzer.interference_matrix(task_ntks)
        base_interaction = float(np.mean(M - np.diag(np.diag(M))))
        feature_change = lr * steps / np.sqrt(max(width, 1))
        interaction_strength = base_interaction * feature_change
        return {
            "interaction_strength": float(interaction_strength),
            "feature_change": float(feature_change),
            "is_cooperative": bool(interaction_strength > 0),
        }

    # -- regime classification -------------------------------------------------

    def classify_multi_task_regime(
        self, width: int, similarity: float, lr: float
    ) -> str:
        """Classify the multi-task regime into one of four phases.

        The phases are:
        - **lazy-independent**: large width, fixed NTK, tasks do not interact.
        - **lazy-coupled**: large width but non-trivial NTK overlap at finite N.
        - **rich-cooperative**: feature learning with positive transfer.
        - **rich-competitive**: feature learning with negative transfer.

        Args:
            width: Network width N.
            similarity: Task similarity s in [0, 1].
            lr: Learning rate η.

        Returns:
            One of ``'lazy-independent'``, ``'lazy-coupled'``,
            ``'rich-cooperative'``, ``'rich-competitive'``.
        """
        # Lazy vs rich boundary: lr * width product
        is_lazy = lr * width > 1.0  # NTK parameterisation: lazy when lr*N >> 1
        # Actually: lazy when lr ~ 1/N (standard param).  Use lr*N as proxy.
        lazy_threshold = 10.0
        is_lazy = (lr * width) > lazy_threshold

        s_crit = self.critical_similarity_threshold(width)
        is_cooperative = similarity > s_crit

        if is_lazy and is_cooperative:
            return "lazy-coupled"
        elif is_lazy and not is_cooperative:
            return "lazy-independent"
        elif not is_lazy and is_cooperative:
            return "rich-cooperative"
        else:
            return "rich-competitive"


# ---------------------------------------------------------------------------
# Transfer Learning Phases
# ---------------------------------------------------------------------------

class TransferLearningPhases:
    """Phase diagram analysis for transfer / fine-tuning from a source to a target task.

    The two key axes are *feature reuse* (target task is solved using source
    features without modification) and *feature learning* (features adapt to
    the target task).  The boundary between these regimes depends on task
    similarity, learning rate, and network width.
    """

    def __init__(self, config: MultiTaskConfig) -> None:
        self.config = config

    # -- CKA-style feature similarity ------------------------------------------

    def feature_reuse_score(
        self, source_features: np.ndarray, target_features: np.ndarray
    ) -> float:
        """Centered Kernel Alignment (CKA) between source and target features.

        CKA is invariant to orthogonal transforms and isotropic scaling, making
        it a robust measure of representational similarity.

        .. math::
            \\text{CKA}(X, Y) = \\frac{\\|Y^T X\\|_F^2}
                                {\\|X^T X\\|_F \\, \\|Y^T Y\\|_F}

        Args:
            source_features: (n, d_s) source representation matrix.
            target_features: (n, d_t) target representation matrix.

        Returns:
            CKA score in [0, 1].
        """
        X = source_features - source_features.mean(axis=0, keepdims=True)
        Y = target_features - target_features.mean(axis=0, keepdims=True)

        cross = np.linalg.norm(Y.T @ X, "fro") ** 2
        xx = np.linalg.norm(X.T @ X, "fro")
        yy = np.linalg.norm(Y.T @ Y, "fro")
        if xx < 1e-15 or yy < 1e-15:
            return 0.0
        return float(cross / (xx * yy))

    # -- feature learning score ------------------------------------------------

    def feature_learning_score(
        self, initial_features: np.ndarray, final_features: np.ndarray
    ) -> float:
        """Measure how much features changed during fine-tuning.

        Score = 1 − CKA(initial, final).  A score of 0 means features were
        fully reused; 1 means features were completely rewritten.

        Args:
            initial_features: (n, d) features before fine-tuning.
            final_features: (n, d) features after fine-tuning.

        Returns:
            Feature-learning score in [0, 1].
        """
        return 1.0 - self.feature_reuse_score(initial_features, final_features)

    # -- transfer phase diagram ------------------------------------------------

    def transfer_phase_diagram(
        self,
        task_similarity_range: np.ndarray,
        lr_range: np.ndarray,
        n_source_steps: int,
    ) -> np.ndarray:
        """Phase diagram for transfer over (task_similarity, lr) plane.

        The transfer benefit Δ is modelled as:

        .. math::
            \\Delta(s, \\eta) = s \\cdot e^{-\\eta / \\eta^*}
                              - (1-s) \\cdot (1 - e^{-\\eta / \\eta^*})

        where η* = 1 / n_source_steps.  Positive Δ → positive transfer.

        Args:
            task_similarity_range: 1-D array of similarity values.
            lr_range: 1-D array of learning rates.
            n_source_steps: Number of source pre-training steps.

        Returns:
            (len(task_similarity_range), len(lr_range)) matrix of Δ values.
        """
        eta_star = 1.0 / max(n_source_steps, 1)
        grid = np.zeros((len(task_similarity_range), len(lr_range)))
        for si, s in enumerate(task_similarity_range):
            for li, lr in enumerate(lr_range):
                exp_term = np.exp(-lr / eta_star)
                delta = s * exp_term - (1 - s) * (1 - exp_term)
                grid[si, li] = delta
        return grid

    # -- feature reuse vs learning boundary ------------------------------------

    def feature_reuse_vs_learning_boundary(
        self, similarity_range: np.ndarray, width_range: np.ndarray
    ) -> np.ndarray:
        """Boundary in (similarity, width) plane between feature-reuse and
        feature-learning transfer regimes.

        At large width the NTK is fixed → reuse is dominant.  At smaller width
        feature learning becomes important.  The boundary similarity scales as
        s_boundary ~ 1 − c / sqrt(N).

        Args:
            similarity_range: 1-D array (unused in output but defines the
                domain for visualisation).
            width_range: 1-D array of widths.

        Returns:
            (len(width_range),) array of boundary similarity values.
        """
        c = 1.0  # O(1) constant
        boundary = np.array([
            max(0.0, min(1.0, 1.0 - c / np.sqrt(max(N, 1))))
            for N in width_range
        ])
        return boundary

    # -- negative transfer boundary --------------------------------------------

    def negative_transfer_boundary(
        self, similarity_range: np.ndarray, width_range: np.ndarray
    ) -> np.ndarray:
        """Boundary below which transfer hurts performance.

        The critical similarity for negative transfer increases at smaller
        width because the feature-learning dynamics can amplify task conflicts.

        .. math::
            s_{\\text{neg}}(N) = \\frac{1}{2} + \\frac{c}{\\sqrt{N}}

        Args:
            similarity_range: 1-D similarity values (defines domain).
            width_range: 1-D widths.

        Returns:
            (len(width_range),) array of negative-transfer boundary similarities.
        """
        c = 0.5
        boundary = np.array([
            min(1.0, 0.5 + c / np.sqrt(max(N, 1)))
            for N in width_range
        ])
        return boundary

    # -- optimal fine-tuning LR ------------------------------------------------

    def optimal_fine_tuning_lr(
        self,
        source_ntk: np.ndarray,
        target_ntk: np.ndarray,
        width: int,
    ) -> float:
        """Learning rate that maximises transfer benefit.

        We model the fine-tuning dynamics in the NTK regime and find the LR
        that balances preserving source features and adapting to the target.

        The optimum scales as η* ~ 1 / (||K_target - K_source|| * sqrt(N)).

        Args:
            source_ntk: (n, n) source-task NTK.
            target_ntk: (n, n) target-task NTK.
            width: Network width.

        Returns:
            Optimal learning rate (positive scalar).
        """
        diff_norm = np.linalg.norm(target_ntk - source_ntk, "fro")
        if diff_norm < 1e-15:
            return 1.0 / np.sqrt(max(width, 1))
        return 1.0 / (diff_norm * np.sqrt(max(width, 1)))

    # -- layer freezing analysis -----------------------------------------------

    def layer_freezing_analysis(
        self,
        source_params: List[np.ndarray],
        target_data: Tuple[np.ndarray, np.ndarray],
        n_layers: int,
    ) -> Dict[str, np.ndarray]:
        """Determine which layers to freeze during fine-tuning.

        For each layer l we estimate the *task-specificity* as the ratio of
        the layer's contribution to the target task vs. the source task.
        Layers with low task-specificity should be frozen.

        The heuristic uses the singular-value spectrum: early layers have
        broader spectra (more general) while later layers are more task-specific.

        Args:
            source_params: List of weight matrices [W_1, ..., W_L] from source.
            target_data: (X_target, y_target) tuple.
            n_layers: Number of layers L.

        Returns:
            Dictionary with ``'task_specificity'`` per layer and boolean
            ``'should_freeze'`` per layer.
        """
        X_target, y_target = target_data
        specificity = np.zeros(n_layers)

        for l in range(min(n_layers, len(source_params))):
            W = source_params[l]
            if W.ndim < 2:
                specificity[l] = 0.0
                continue
            s_vals = svd(W, compute_uv=False)
            # Concentration ratio: top-1 / sum — higher means more specialised
            if s_vals.sum() < 1e-15:
                specificity[l] = 0.0
            else:
                specificity[l] = float(s_vals[0] / s_vals.sum())

        # Freeze layers whose specificity is below the median
        median_spec = float(np.median(specificity))
        should_freeze = specificity <= median_spec

        return {
            "task_specificity": specificity,
            "should_freeze": should_freeze,
        }

    # -- CKA dynamics over fine-tuning -----------------------------------------

    def representation_similarity_dynamics(
        self,
        source_trajectory: List[np.ndarray],
        target_trajectory: List[np.ndarray],
    ) -> np.ndarray:
        """Track CKA between source and target representations over fine-tuning steps.

        Args:
            source_trajectory: List of (n, d) feature matrices from source,
                recorded at successive fine-tuning steps.
            target_trajectory: List of (n, d) feature matrices from target at
                the same steps.

        Returns:
            (n_steps,) array of CKA scores.
        """
        n_steps = min(len(source_trajectory), len(target_trajectory))
        cka_vals = np.zeros(n_steps)
        for t in range(n_steps):
            cka_vals[t] = self.feature_reuse_score(
                source_trajectory[t], target_trajectory[t]
            )
        return cka_vals

    # -- critical dataset size -------------------------------------------------

    def critical_dataset_size(
        self,
        source_ntk: np.ndarray,
        target_ntk: np.ndarray,
        width: int,
    ) -> int:
        """Minimum target-task dataset size for positive transfer.

        Positive transfer requires enough target data to overcome the bias
        introduced by misaligned source features.  The critical size scales as

        .. math::
            n^* \\sim \\frac{\\|K_s - K_t\\|_F^2}{\\lambda_{\\min}(K_t)}
                     \\cdot \\frac{1}{\\sqrt{N}}

        Args:
            source_ntk: (n, n) source NTK.
            target_ntk: (n, n) target NTK.
            width: Network width N.

        Returns:
            Estimated minimum dataset size (integer ≥ 1).
        """
        diff_sq = np.linalg.norm(target_ntk - source_ntk, "fro") ** 2
        eigvals = np.linalg.eigvalsh(target_ntk)
        lam_min = max(eigvals[0], 1e-10)
        n_star = diff_sq / (lam_min * np.sqrt(max(width, 1)))
        return max(1, int(np.ceil(n_star)))

    # -- few-shot phase boundary -----------------------------------------------

    def few_shot_phase_boundary(
        self,
        n_shot_range: np.ndarray,
        similarity_range: np.ndarray,
        width: int,
    ) -> np.ndarray:
        """Phase boundary for few-shot transfer in (n_shot, similarity) plane.

        At each (k, s) we evaluate whether the expected transfer is positive.
        The model is:

        .. math::
            \\Delta(k, s) = s - \\frac{1}{k} \\cdot \\frac{1}{\\sqrt{N}}

        The boundary is at Δ = 0, i.e.  s* = 1 / (k sqrt(N)).

        Args:
            n_shot_range: 1-D array of shot counts k.
            similarity_range: 1-D array of task similarities (defines domain).
            width: Network width N.

        Returns:
            (len(n_shot_range),) array of critical similarity values.
        """
        boundary = np.array([
            min(1.0, 1.0 / (max(k, 1) * np.sqrt(max(width, 1))))
            for k in n_shot_range
        ])
        return boundary


# ---------------------------------------------------------------------------
# Multi-Task Spectral Analysis
# ---------------------------------------------------------------------------

class MultiTaskSpectralAnalysis:
    """Spectral-geometric analysis of task interactions in the NTK eigenspace.

    The combined NTK is a weighted sum K = Σ w_i K_i.  Its eigenspace
    partitions naturally into sub-spaces aligned with individual tasks.
    This class quantifies how spectral capacity is allocated across tasks.
    """

    def __init__(self) -> None:
        pass

    # -- joint NTK spectrum ----------------------------------------------------

    def joint_ntk_spectrum(
        self, task_ntks: List[np.ndarray], weights: np.ndarray
    ) -> np.ndarray:
        """Eigenvalues of the weighted combined NTK  K = Σ w_i K_i.

        Args:
            task_ntks: List of (n, n) per-task NTK matrices.
            weights: (n_tasks,) weight vector.

        Returns:
            (n,) sorted (descending) eigenvalues of K.
        """
        K = sum(w * Kt for w, Kt in zip(weights, task_ntks))
        eigvals = np.linalg.eigvalsh(K)
        return eigvals[::-1]

    # -- task subspace projection ----------------------------------------------

    def task_subspace_projection(
        self, joint_ntk: np.ndarray, task_ntk: np.ndarray
    ) -> np.ndarray:
        """Project joint NTK eigenmodes onto a single task's NTK.

        For each joint eigenmode v_k, compute v_k^T K_t v_k.  This measures
        how much of mode k is "used" by task t.

        Args:
            joint_ntk: (n, n) combined NTK.
            task_ntk: (n, n) per-task NTK.

        Returns:
            (n,) array of projection values (one per eigenmode, descending order).
        """
        eigvals, eigvecs = eigh(joint_ntk)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        projections = np.array([
            float(v @ task_ntk @ v) for v in eigvecs.T
        ])
        return projections

    # -- effective dimension per task ------------------------------------------

    def effective_dimension_per_task(
        self, joint_ntk: np.ndarray, task_ntks: List[np.ndarray]
    ) -> np.ndarray:
        """Effective dimension each task occupies in the joint eigenspace.

        Defined as the participation ratio of the projection weights:

        .. math::
            d_{\\text{eff}}^{(t)} = \\frac{(\\sum_k p_k^{(t)})^2}
                                    {\\sum_k (p_k^{(t)})^2}

        Args:
            joint_ntk: (n, n) combined NTK.
            task_ntks: List of per-task NTK matrices.

        Returns:
            (n_tasks,) array of effective dimensions.
        """
        dims = np.zeros(len(task_ntks))
        for t, Kt in enumerate(task_ntks):
            proj = self.task_subspace_projection(joint_ntk, Kt)
            proj_abs = np.abs(proj)
            s1 = proj_abs.sum()
            s2 = (proj_abs ** 2).sum()
            dims[t] = s1 ** 2 / s2 if s2 > 1e-15 else 0.0
        return dims

    # -- spectral interference -------------------------------------------------

    def spectral_interference(self, task_ntks: List[np.ndarray]) -> np.ndarray:
        """Overlap of spectral supports between pairs of tasks.

        We compute the overlap as the cosine similarity between the sorted
        eigenvalue spectra of each pair.

        Args:
            task_ntks: List of per-task NTK matrices.

        Returns:
            (n_tasks, n_tasks) spectral overlap matrix.
        """
        spectra = []
        for K in task_ntks:
            ev = np.sort(np.linalg.eigvalsh(K))[::-1]
            spectra.append(ev)

        n = len(spectra)
        overlap = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                min_len = min(len(spectra[i]), len(spectra[j]))
                si = spectra[i][:min_len]
                sj = spectra[j][:min_len]
                norm_i = np.linalg.norm(si)
                norm_j = np.linalg.norm(sj)
                if norm_i < 1e-15 or norm_j < 1e-15:
                    val = 0.0
                else:
                    val = float(np.dot(si, sj) / (norm_i * norm_j))
                overlap[i, j] = val
                overlap[j, i] = val
        return overlap

    # -- capacity allocation ---------------------------------------------------

    def capacity_allocation(
        self,
        joint_ntk_eigenvalues: np.ndarray,
        task_projections: List[np.ndarray],
    ) -> np.ndarray:
        """Fraction of total spectral capacity allocated to each task.

        Capacity for task t is:

        .. math::
            C_t = \\frac{\\sum_k \\lambda_k \\, p_k^{(t)}}{\\sum_k \\lambda_k}

        Args:
            joint_ntk_eigenvalues: (n,) eigenvalues (descending).
            task_projections: List of (n,) projection-weight arrays.

        Returns:
            (n_tasks,) capacity fractions (sum ≤ n_tasks but typically ≈ 1 when
            projections are normalised).
        """
        lam_sum = np.sum(np.abs(joint_ntk_eigenvalues))
        if lam_sum < 1e-15:
            return np.zeros(len(task_projections))
        capacities = np.array([
            float(np.sum(joint_ntk_eigenvalues * np.abs(p))) / lam_sum
            for p in task_projections
        ])
        return capacities

    # -- optimal spectral weighting --------------------------------------------

    def optimal_spectral_weighting(
        self,
        task_ntks: List[np.ndarray],
        target_effective_dims: np.ndarray,
    ) -> np.ndarray:
        """Find task weights so that each task achieves a target effective dimension.

        We solve  min_w Σ_t (d_eff^(t)(w) − d_target^(t))^2  by scipy optimisation.

        Args:
            task_ntks: List of per-task NTK matrices.
            target_effective_dims: (n_tasks,) desired effective dimensions.

        Returns:
            (n_tasks,) optimal weight vector summing to 1.
        """
        T = len(task_ntks)

        def objective(w: np.ndarray) -> float:
            K = sum(wi * Ki for wi, Ki in zip(w, task_ntks))
            dims = self.effective_dimension_per_task(K, task_ntks)
            return float(np.sum((dims - target_effective_dims) ** 2))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(1e-6, None)] * T
        w0 = np.ones(T) / T
        res = optimize.minimize(
            objective, w0, bounds=bounds, constraints=constraints, method="SLSQP",
            options={"maxiter": 200},
        )
        w_opt = np.maximum(res.x, 0.0)
        return w_opt / w_opt.sum()


# ---------------------------------------------------------------------------
# Multi-Task Experiment
# ---------------------------------------------------------------------------

class MultiTaskExperiment:
    """End-to-end experimental harness for multi-task phase diagram validation.

    Generates synthetic tasks, sweeps over widths and similarities, measures
    empirical interference, and compares with the theoretical predictions.
    """

    def __init__(self, config: MultiTaskConfig) -> None:
        self.config = config
        self._analyzer = TaskInterferenceAnalyzer(config)
        self._boundary = MultiTaskPhaseBoundary(config)

    # -- synthetic task generation ---------------------------------------------

    def generate_synthetic_tasks(
        self,
        n_tasks: int,
        d_input: int,
        similarity: float,
        n_samples: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate related regression tasks with controlled similarity.

        Each task has target y_t = X @ w_t + noise, where w_t is a random
        vector constructed so that cos(w_i, w_j) ≈ similarity for all i ≠ j.

        Args:
            n_tasks: Number of tasks T.
            d_input: Input dimension d.
            similarity: Desired pairwise cosine similarity s in [0, 1].
            n_samples: Number of samples per task.

        Returns:
            ``(X_tasks, y_tasks)`` — lists of (n_samples, d_input) input
            matrices and (n_samples,) target vectors.
        """
        rng = np.random.RandomState(42)

        # Shared direction
        shared = rng.randn(d_input)
        shared /= np.linalg.norm(shared)

        X_tasks: List[np.ndarray] = []
        y_tasks: List[np.ndarray] = []

        for t in range(n_tasks):
            X = rng.randn(n_samples, d_input)
            # Task-specific weight = similarity * shared + sqrt(1-s^2) * random
            random_dir = rng.randn(d_input)
            random_dir -= np.dot(random_dir, shared) * shared
            norm_r = np.linalg.norm(random_dir)
            if norm_r > 1e-10:
                random_dir /= norm_r
            w_t = similarity * shared + np.sqrt(max(1.0 - similarity ** 2, 0.0)) * random_dir
            y = X @ w_t + 0.1 * rng.randn(n_samples)
            X_tasks.append(X)
            y_tasks.append(y)

        return X_tasks, y_tasks

    # -- sweep over widths and similarities ------------------------------------

    def run_multi_task_sweep(
        self,
        X_tasks: List[np.ndarray],
        y_tasks: List[np.ndarray],
        widths: np.ndarray,
        similarities: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Sweep over widths and similarities, recording interference metrics.

        At each (width, similarity) point we construct synthetic NTK matrices
        and compute the interference matrix.

        Args:
            X_tasks: List of input matrices (one per task).
            y_tasks: List of target vectors (one per task).
            widths: 1-D array of widths to sweep.
            similarities: 1-D array of similarity values to sweep.

        Returns:
            Dictionary with ``'interference_grid'`` of shape
            (len(widths), len(similarities)) and ``'regimes'`` grid of regime
            labels.
        """
        n_tasks = len(X_tasks)
        interf_grid = np.zeros((len(widths), len(similarities)))
        regime_grid = np.empty((len(widths), len(similarities)), dtype=object)

        for wi, N in enumerate(widths):
            for si, s in enumerate(similarities):
                # Build NTKs from data with controlled similarity
                d = X_tasks[0].shape[1]
                n = X_tasks[0].shape[0]
                base = np.eye(n) * N
                shared = X_tasks[0] @ X_tasks[0].T / d
                ntks = []
                for t in range(n_tasks):
                    task_part = X_tasks[t] @ X_tasks[t].T / d
                    K = base + s * shared + (1 - s) * task_part
                    ntks.append(K)
                M = self._analyzer.interference_matrix(ntks)
                w = self.config.task_weights[:n_tasks]
                if len(w) < n_tasks:
                    w = np.ones(n_tasks) / n_tasks
                interf_grid[wi, si] = self._analyzer.total_interference(M, w)
                regime_grid[wi, si] = self._boundary.classify_multi_task_regime(
                    int(N), s, 1.0 / max(N, 1)
                )

        return {
            "interference_grid": interf_grid,
            "regimes": regime_grid,
        }

    # -- measure actual interference -------------------------------------------

    def measure_interference(
        self,
        X_tasks: List[np.ndarray],
        y_tasks: List[np.ndarray],
        width: int,
        weights: np.ndarray,
    ) -> Dict[str, float]:
        """Measure empirical interference using NTK proxies.

        Constructs per-task NTK-like matrices from the data (using the linear
        kernel as a proxy for the NTK at initialisation) and computes
        interference metrics.

        Args:
            X_tasks: List of (n, d) input arrays.
            y_tasks: List of (n,) target arrays.
            width: Network width (used for scaling).
            weights: Per-task weights.

        Returns:
            Dictionary with ``'total_interference'``, ``'max_pairwise'``, and
            ``'mean_pairwise'`` values.
        """
        d = X_tasks[0].shape[1]
        ntks = []
        for X in X_tasks:
            K = X @ X.T / d + np.eye(X.shape[0]) * width
            ntks.append(K)

        M = self._analyzer.interference_matrix(ntks)
        n = M.shape[0]
        off_diag = []
        for i in range(n):
            for j in range(i + 1, n):
                off_diag.append(M[i, j])

        total = self._analyzer.total_interference(M, weights[:n])
        return {
            "total_interference": total,
            "max_pairwise": float(max(off_diag)) if off_diag else 0.0,
            "mean_pairwise": float(np.mean(off_diag)) if off_diag else 0.0,
        }

    # -- theory vs experiment comparison ---------------------------------------

    def compare_theory_experiment(
        self,
        theoretical_boundary: np.ndarray,
        empirical_results: np.ndarray,
    ) -> Dict[str, float]:
        """Compare theoretical phase boundary with empirical measurements.

        Computes the mean absolute error and correlation between the predicted
        boundary and the measured boundary.

        Args:
            theoretical_boundary: 1-D array of predicted critical similarities.
            empirical_results: 1-D array of measured critical similarities.

        Returns:
            Dictionary with ``'mae'``, ``'correlation'``, and ``'max_error'``.
        """
        min_len = min(len(theoretical_boundary), len(empirical_results))
        theory = theoretical_boundary[:min_len]
        empirical = empirical_results[:min_len]

        mae = float(np.mean(np.abs(theory - empirical)))
        if np.std(theory) < 1e-15 or np.std(empirical) < 1e-15:
            corr = 0.0
        else:
            corr = float(np.corrcoef(theory, empirical)[0, 1])
        max_err = float(np.max(np.abs(theory - empirical)))

        return {"mae": mae, "correlation": corr, "max_error": max_err}

    # -- publication figure data -----------------------------------------------

    def publication_figure(
        self,
        phase_diagram: np.ndarray,
        theory_boundary: np.ndarray,
        empirical_points: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Assemble data for a publication-quality summary figure.

        Returns structured arrays that can be directly plotted with matplotlib.

        Args:
            phase_diagram: (n_sim, n_width) interference grid.
            theory_boundary: (n_width,) predicted boundary curve.
            empirical_points: (n_points, 2) array of (width, critical_sim) pairs.

        Returns:
            Dictionary with ``'phase_diagram'``, ``'theory_boundary'``,
            ``'empirical_points'``, and ``'comparison_stats'``.
        """
        stats = self.compare_theory_experiment(
            theory_boundary,
            empirical_points[:, 1] if empirical_points.ndim == 2 else empirical_points,
        )
        return {
            "phase_diagram": phase_diagram,
            "theory_boundary": theory_boundary,
            "empirical_points": empirical_points,
            "comparison_stats": stats,
        }
