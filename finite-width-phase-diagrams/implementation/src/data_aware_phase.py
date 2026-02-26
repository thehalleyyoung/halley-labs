"""Data-aware phase analysis for neural networks.

Extends the architecture-only mean-field theory with dataset-dependent
corrections based on kernel-task spectral alignment. The key insight
(Bordelon & Pehlevy, 2020) is that the spectral alignment between the
Neural Tangent Kernel and the target function governs lazy-to-rich
transitions and shifts the effective phase boundary.

The alignment score κ = Σ(y_k² · λ_k) / (||y||² · Σλ_k) measures how
well the target function concentrates along the top kernel eigendirections.
Well-aligned tasks (κ → 1) train easily even in the ordered phase, while
poorly aligned tasks require near-critical initialization.

Usage:
    >>> analyzer = DataAwarePhaseAnalyzer()
    >>> result = analyzer.data_aware_analysis(model, train_loader)
    >>> print(result.spectral.adjusted_phase)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn
    from torch import Tensor

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from mean_field_theory import ArchitectureSpec, MeanFieldAnalyzer
except ImportError:
    from .mean_field_theory import ArchitectureSpec, MeanFieldAnalyzer

try:
    from pytorch_integration import analyze_model, detect_architecture, extract_ntk
except ImportError:
    from .pytorch_integration import analyze_model, detect_architecture, extract_ntk


def _require_torch() -> None:
    """Raise ImportError if PyTorch is not available."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for data_aware_phase. "
            "Install with: pip install phase-diagrams[torch]"
        )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SpectralAnalysis:
    """Results of spectral analysis of kernel-task alignment.

    Attributes:
        alignment_score: The κ metric measuring how well the target
            concentrates along the top kernel eigendirections. Values
            close to 1 indicate strong alignment; close to 0 indicates
            the target lives in the kernel's null space.
        effective_dimension: Participation ratio of kernel eigenvalues,
            p = (Σλ_k)² / Σλ_k², measuring the effective number of
            active kernel directions.
        eigenvalue_decay_rate: Power-law exponent β of the eigenvalue
            spectrum λ_k ~ k^{-β}. Steeper decay (larger β) means the
            kernel concentrates on fewer directions.
        label_concentration: Fraction of label energy in the top-k
            eigendirections where k = effective_dimension.
        adjusted_chi_1: Data-corrected susceptibility χ_eff that
            accounts for kernel-task alignment.
        adjusted_phase: Data-aware phase classification (one of
            "ordered", "critical", "chaotic").
        explanation: Human-readable explanation of the analysis.
    """

    alignment_score: float = 0.0
    effective_dimension: float = 0.0
    eigenvalue_decay_rate: float = 0.0
    label_concentration: float = 0.0
    adjusted_chi_1: float = 0.0
    adjusted_phase: str = "unknown"
    explanation: str = ""


@dataclass
class DataAwareReport:
    """Combined architecture and data-aware analysis report.

    Attributes:
        spectral: Spectral analysis results with alignment metrics.
        architecture_phase: Phase from architecture-only mean-field theory.
        architecture_chi_1: Raw susceptibility from mean-field analysis.
        kernel_eigenvalues: Eigenvalues of the NTK or empirical kernel.
        target_projections: Projections of the target onto kernel
            eigenvectors (y_k coefficients).
        width: Network width used in the analysis.
        depth: Network depth used in the analysis.
    """

    spectral: SpectralAnalysis = field(default_factory=SpectralAnalysis)
    architecture_phase: str = "unknown"
    architecture_chi_1: float = 0.0
    kernel_eigenvalues: Optional[NDArray] = None
    target_projections: Optional[NDArray] = None
    width: int = 0
    depth: int = 0


# ---------------------------------------------------------------------------
# Core analysis class
# ---------------------------------------------------------------------------


class DataAwarePhaseAnalyzer:
    """Analyzer that combines mean-field theory with data-dependent corrections.

    The standard mean-field approach classifies networks into ordered,
    critical, or chaotic phases based on architecture alone. This analyzer
    augments that with spectral alignment between the NTK and the target
    function, shifting phase boundaries according to the alignment score.

    Args:
        alpha: O(1) constant governing the strength of the data-dependent
            correction. Default 1.0 follows the Bordelon & Pehlevy scaling.
        nystrom_threshold: Number of parameters above which the Nyström
            approximation is used instead of full Jacobian NTK computation.
        nystrom_rank: Rank of the Nyström approximation.
        top_k_fraction: Fraction of eigenvalues considered "top-k" for
            the label concentration metric. Default 0.1 (top 10%).
        mf_tolerance: Tolerance for mean-field fixed-point iteration.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        nystrom_threshold: int = 100_000,
        nystrom_rank: int = 100,
        top_k_fraction: float = 0.1,
        mf_tolerance: float = 1e-8,
    ) -> None:
        self.alpha = alpha
        self.nystrom_threshold = nystrom_threshold
        self.nystrom_rank = nystrom_rank
        self.top_k_fraction = top_k_fraction
        self.mf_analyzer = MeanFieldAnalyzer(tolerance=mf_tolerance)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_kernel_alignment(
        self,
        model: Any,
        data_loader: Any,
        n_samples: int = 200,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Compute NTK alignment with target labels.

        Uses the existing ``extract_ntk`` for small models and switches
        to Nyström approximation for large ones.  Returns kernel
        eigenvalues, eigenvectors, and projected target labels.

        Args:
            model: A ``torch.nn.Module`` instance.
            data_loader: A PyTorch ``DataLoader`` yielding ``(inputs, targets)``
                tuples or plain ``inputs`` tensors.
            n_samples: Maximum number of samples to use for the kernel
                computation.

        Returns:
            A 3-tuple ``(eigenvalues, eigenvectors, targets)`` where
            *eigenvalues* is a 1-D array sorted in descending order,
            *eigenvectors* are the corresponding column vectors, and
            *targets* is the collected label vector (one-hot for
            classification, raw for regression).

        Raises:
            ImportError: If PyTorch is not installed.
            ValueError: If the data loader yields no data.
        """
        _require_torch()

        inputs, targets = self._collect_samples(data_loader, n_samples)
        if inputs.shape[0] == 0:
            raise ValueError("Data loader yielded no samples.")

        n_params = sum(p.numel() for p in model.parameters())
        method = "nystrom" if n_params > self.nystrom_threshold else "auto"

        model.eval()
        with torch.no_grad():
            ntk_result = extract_ntk(
                model,
                inputs,
                method=method,
                nystrom_rank=self.nystrom_rank,
            )

        if ntk_result.eigenvalues is not None and len(ntk_result.eigenvalues) > 0:
            eigenvalues = np.sort(ntk_result.eigenvalues)[::-1]
        else:
            # Fallback: eigendecompose the kernel matrix ourselves
            eigenvalues, _ = self._safe_eigh(ntk_result.kernel_matrix)

        # Full eigenvectors from the kernel matrix for projections
        if ntk_result.kernel_matrix is not None:
            eigenvalues_full, eigenvectors = self._safe_eigh(ntk_result.kernel_matrix)
        else:
            # No kernel matrix available — construct a diagonal proxy
            eigenvalues_full = eigenvalues
            eigenvectors = np.eye(len(eigenvalues))

        target_np = self._targets_to_numpy(targets)

        return eigenvalues_full, eigenvectors, target_np

    def spectral_task_alignment(
        self,
        kernel_eigenvalues: NDArray,
        target_in_eigenbasis: NDArray,
    ) -> SpectralAnalysis:
        """Compute the spectral alignment score and related metrics.

        The alignment score is defined as::

            κ = Σ(y_k² · λ_k) / (||y||² · Σλ_k)

        where *y_k* are projections of the target onto the kernel
        eigenvectors and *λ_k* are the corresponding eigenvalues.

        Args:
            kernel_eigenvalues: 1-D array of eigenvalues in descending
                order.
            target_in_eigenbasis: 1-D array of target projections onto
                the eigenvectors (y_k = vₖᵀ y).

        Returns:
            A :class:`SpectralAnalysis` dataclass with the computed
            metrics (``adjusted_chi_1`` and ``adjusted_phase`` are left
            at their defaults; use :meth:`adjusted_phase_boundary` to
            fill them).

        Raises:
            ValueError: If input arrays are empty or have mismatched
                lengths.
        """
        eigenvalues = np.asarray(kernel_eigenvalues, dtype=np.float64)
        y_k = np.asarray(target_in_eigenbasis, dtype=np.float64)

        if eigenvalues.size == 0 or y_k.size == 0:
            raise ValueError("Eigenvalues and target projections must be non-empty.")

        # Truncate to the shorter length if sizes differ (e.g. Nyström)
        min_len = min(len(eigenvalues), len(y_k))
        eigenvalues = eigenvalues[:min_len]
        y_k = y_k[:min_len]

        # Clamp negative eigenvalues to zero (numerical noise)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        y_norm_sq = float(np.sum(y_k ** 2))
        lambda_sum = float(np.sum(eigenvalues))

        # Edge case: zero-norm target or zero-trace kernel
        if y_norm_sq < 1e-30 or lambda_sum < 1e-30:
            return SpectralAnalysis(
                alignment_score=0.0,
                effective_dimension=0.0,
                eigenvalue_decay_rate=0.0,
                label_concentration=0.0,
                explanation="Degenerate kernel or zero target; alignment undefined.",
            )

        # κ: alignment score
        alignment = float(np.sum(y_k ** 2 * eigenvalues)) / (y_norm_sq * lambda_sum)
        alignment = float(np.clip(alignment, 0.0, 1.0))

        # Participation ratio (effective dimension)
        lambda_sq_sum = float(np.sum(eigenvalues ** 2))
        if lambda_sq_sum > 1e-30:
            eff_dim = (lambda_sum ** 2) / lambda_sq_sum
        else:
            eff_dim = 0.0

        # Eigenvalue decay rate via log-log regression
        decay_rate = self._fit_power_law(eigenvalues)

        # Label concentration in top-k eigendirections
        k = max(1, int(self.top_k_fraction * len(eigenvalues)))
        top_energy = float(np.sum(y_k[:k] ** 2))
        label_conc = top_energy / y_norm_sq if y_norm_sq > 1e-30 else 0.0

        explanation = (
            f"Alignment κ={alignment:.4f}. "
            f"Effective dimension={eff_dim:.1f}/{len(eigenvalues)}. "
            f"Eigenvalue decay β={decay_rate:.2f}. "
            f"Top-{k} label concentration={label_conc:.2%}."
        )

        return SpectralAnalysis(
            alignment_score=alignment,
            effective_dimension=eff_dim,
            eigenvalue_decay_rate=decay_rate,
            label_concentration=label_conc,
            explanation=explanation,
        )

    def adjusted_phase_boundary(
        self,
        alignment_score: float,
        chi_1: float,
        width: int,
        depth: int,
    ) -> Tuple[float, str]:
        """Shift the effective phase boundary based on alignment.

        Following Bordelon & Pehlevy (2020), the effective susceptibility
        that governs trainability is::

            χ_eff = χ_1 · (1 + α · (1 - κ) · log(N) / N)

        where *N* is the width, *κ* is the alignment score, and *α* is
        an O(1) constant.  Well-aligned tasks (κ → 1) have χ_eff ≈ χ_1;
        poorly aligned tasks push toward the ordered phase.

        Args:
            alignment_score: Spectral alignment κ ∈ [0, 1].
            chi_1: Raw susceptibility from mean-field analysis.
            width: Network width *N*.
            depth: Network depth *L*.

        Returns:
            A 2-tuple ``(chi_eff, phase)`` where *chi_eff* is the
            adjusted susceptibility and *phase* is one of ``"ordered"``,
            ``"critical"``, or ``"chaotic"``.
        """
        kappa = float(np.clip(alignment_score, 0.0, 1.0))
        N = max(width, 1)

        # Data-dependent correction
        correction = 1.0 + self.alpha * (1.0 - kappa) * math.log(N) / N
        chi_eff = chi_1 * correction

        # Depth-dependent tightening: deeper nets are more sensitive
        depth_factor = 1.0 + 0.01 * max(depth - 10, 0)
        chi_eff *= depth_factor

        phase = self._classify_phase(chi_eff)

        return chi_eff, phase

    def data_aware_analysis(
        self,
        model: Any,
        data_loader: Any,
        input_shape: Optional[Tuple[int, ...]] = None,
        n_samples: int = 200,
    ) -> DataAwareReport:
        """Full pipeline combining architecture and data-dependent analysis.

        1. Detect architecture and run mean-field analysis.
        2. Compute NTK and spectral alignment with the target.
        3. Adjust the phase boundary using the alignment score.

        Args:
            model: A ``torch.nn.Module`` instance.
            data_loader: A PyTorch ``DataLoader`` yielding ``(inputs, targets)``
                or plain ``inputs`` tensors. If targets are absent,
                only architecture-based analysis is returned.
            input_shape: Optional input shape ``(C, H, W)`` or ``(D,)`` for
                mean-field analysis. If *None*, inferred from the first
                batch of the data loader.
            n_samples: Number of samples for kernel computation.

        Returns:
            A :class:`DataAwareReport` combining architecture-only and
            data-aware analysis.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        _require_torch()

        # -- Step 1: Architecture detection & mean-field analysis ----------
        arch_info = detect_architecture(model)
        depth = arch_info.depth
        widths = arch_info.widths
        width = max(widths) if widths else 1

        arch_spec = self._arch_info_to_spec(arch_info)
        mf_report = self.mf_analyzer.analyze(arch_spec)

        arch_chi_1 = mf_report.chi_1
        arch_phase = mf_report.phase

        report = DataAwareReport(
            architecture_phase=arch_phase,
            architecture_chi_1=arch_chi_1,
            width=width,
            depth=depth,
        )

        # -- Step 2: NTK and spectral alignment ---------------------------
        try:
            eigenvalues, eigenvectors, targets = self.compute_kernel_alignment(
                model, data_loader, n_samples=n_samples
            )
        except (ValueError, RuntimeError) as exc:
            # No usable data — fall back to architecture-only analysis
            report.spectral = SpectralAnalysis(
                adjusted_chi_1=arch_chi_1,
                adjusted_phase=arch_phase,
                explanation=f"Data-aware analysis unavailable: {exc}. "
                f"Falling back to architecture-only phase={arch_phase}.",
            )
            return report

        # Handle case where targets are missing (unsupervised loader)
        if targets is None or targets.size == 0:
            report.spectral = SpectralAnalysis(
                adjusted_chi_1=arch_chi_1,
                adjusted_phase=arch_phase,
                explanation="No target labels found; using architecture-only analysis.",
            )
            report.kernel_eigenvalues = eigenvalues
            return report

        # Project targets onto eigenbasis
        target_projections = self._project_onto_eigenbasis(targets, eigenvectors)

        # -- Step 3: Spectral analysis ------------------------------------
        spectral = self.spectral_task_alignment(eigenvalues, target_projections)

        # -- Step 4: Adjust phase boundary --------------------------------
        chi_eff, adjusted_phase = self.adjusted_phase_boundary(
            spectral.alignment_score, arch_chi_1, width, depth
        )

        spectral.adjusted_chi_1 = chi_eff
        spectral.adjusted_phase = adjusted_phase

        # Enrich explanation
        phase_shift = ""
        if adjusted_phase != arch_phase:
            phase_shift = (
                f" Phase shifted from '{arch_phase}' to '{adjusted_phase}' "
                f"due to data alignment."
            )
        spectral.explanation += (
            f" Architecture χ₁={arch_chi_1:.4f}, effective χ_eff={chi_eff:.4f}."
            f"{phase_shift}"
        )

        report.spectral = spectral
        report.kernel_eigenvalues = eigenvalues
        report.target_projections = target_projections

        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_samples(
        data_loader: Any,
        n_samples: int,
    ) -> Tuple[Any, Optional[Any]]:
        """Collect up to *n_samples* from a PyTorch DataLoader.

        Supports loaders that yield ``(inputs, targets)`` or plain
        ``inputs`` tensors.

        Returns:
            ``(inputs_tensor, targets_tensor)`` where *targets_tensor*
            may be ``None`` if the loader has no labels.
        """
        all_inputs: list = []
        all_targets: list = []
        has_targets = True
        collected = 0

        for batch in data_loader:
            if collected >= n_samples:
                break

            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
                all_targets.append(y)
            else:
                x = batch if not isinstance(batch, (list, tuple)) else batch[0]
                has_targets = False

            all_inputs.append(x)
            collected += x.shape[0]

        if not all_inputs:
            # Return empty tensors
            return torch.empty(0), None

        inputs = torch.cat(all_inputs, dim=0)[:n_samples]
        targets = None
        if has_targets and all_targets:
            targets = torch.cat(all_targets, dim=0)[:n_samples]

        return inputs, targets

    @staticmethod
    def _targets_to_numpy(targets: Optional[Any]) -> Optional[NDArray]:
        """Convert targets tensor to a 1-D numpy array.

        For multi-class targets, one-hot encodes and flattens.  For
        regression targets, simply flattens.
        """
        if targets is None:
            return None

        t = targets.detach().cpu()

        # Integer labels → one-hot → flatten
        if t.dtype in (torch.long, torch.int, torch.int32, torch.int16):
            n_classes = int(t.max().item()) + 1
            if n_classes <= 1:
                # Single-class edge case: use raw values
                return t.float().numpy().ravel()
            one_hot = torch.zeros(t.shape[0], n_classes)
            one_hot.scatter_(1, t.unsqueeze(1), 1.0)
            return one_hot.numpy().ravel()

        return t.float().numpy().ravel()

    @staticmethod
    def _safe_eigh(matrix: Optional[NDArray]) -> Tuple[NDArray, NDArray]:
        """Symmetric eigendecomposition with numerical safeguards.

        Returns eigenvalues in *descending* order with matching
        eigenvectors.
        """
        if matrix is None:
            return np.array([]), np.array([[]])

        K = np.asarray(matrix, dtype=np.float64)
        # Symmetrise
        K = 0.5 * (K + K.T)

        eigenvalues, eigenvectors = np.linalg.eigh(K)

        # Descending order
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvalues[idx], eigenvectors[:, idx]

    @staticmethod
    def _project_onto_eigenbasis(
        targets: NDArray,
        eigenvectors: NDArray,
    ) -> NDArray:
        """Project target vector onto eigenbasis.

        If the target dimension does not match the number of
        eigenvectors, truncate or pad as needed.
        """
        y = np.asarray(targets, dtype=np.float64).ravel()
        V = np.asarray(eigenvectors, dtype=np.float64)

        n_eig = V.shape[1]
        n_y = len(y)

        if n_y > V.shape[0]:
            y = y[: V.shape[0]]
        elif n_y < V.shape[0]:
            y = np.pad(y, (0, V.shape[0] - n_y))

        return V.T @ y  # shape (n_eig,)

    @staticmethod
    def _fit_power_law(eigenvalues: NDArray) -> float:
        """Fit a power-law decay λ_k ~ k^{-β} via log-log regression.

        Returns the exponent β (positive for decaying spectra).
        """
        pos = eigenvalues[eigenvalues > 1e-30]
        if len(pos) < 2:
            return 0.0

        log_k = np.log(np.arange(1, len(pos) + 1))
        log_lam = np.log(pos)

        # Least-squares fit: log_lam = -β · log_k + const
        A = np.vstack([log_k, np.ones_like(log_k)]).T
        result = np.linalg.lstsq(A, log_lam, rcond=None)
        slope = result[0][0]

        return float(-slope)  # β is positive for decay

    @staticmethod
    def _classify_phase(chi_eff: float) -> str:
        """Classify phase from effective susceptibility.

        Uses standard mean-field thresholds with a narrow critical
        window around χ₁ = 1.
        """
        if chi_eff < 0.95:
            return "ordered"
        elif chi_eff > 1.05:
            return "chaotic"
        else:
            return "critical"

    @staticmethod
    def _arch_info_to_spec(arch_info: Any) -> ArchitectureSpec:
        """Convert an ``ArchitectureInfo`` to an ``ArchitectureSpec``.

        Fills in reasonable defaults for fields not present in the
        architecture info (e.g., weight scale, activation).
        """
        width = max(arch_info.widths) if arch_info.widths else 1000
        depth = arch_info.depth if arch_info.depth else 1

        activation = "relu"
        has_residual = getattr(arch_info, "has_residual", False)
        has_batchnorm = getattr(arch_info, "has_normalization", False)

        # Edge-of-chaos initialisation heuristic: σ_w = √(2/fan_in)
        sigma_w = math.sqrt(2.0 / width) * math.sqrt(width)  # ≈ √2

        return ArchitectureSpec(
            depth=depth,
            width=width,
            activation=activation,
            sigma_w=sigma_w,
            sigma_b=0.0,
            has_residual=has_residual,
            has_batchnorm=has_batchnorm,
        )
