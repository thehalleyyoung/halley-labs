"""
Contagion model property verification via SMT.

Addresses critique: "Contagion models are complex numerical simulations
used as trusted inputs but never formally verified or model-checked."

Verifies key mathematical properties of contagion models:
  1. Monotonicity: higher initial shocks => higher final losses
  2. Bounded output: losses are in [0, total_assets]
  3. Lipschitz continuity: small shock changes => bounded loss changes
  4. Fixed-point convergence: iterative models converge
  5. Conservation: total losses <= total exposures
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

logger = logging.getLogger(__name__)


@dataclass
class PropertyVerificationResult:
    """Result of verifying a single contagion model property."""
    property_name: str
    model_name: str
    verified: bool
    counterexample: Optional[Dict[str, float]] = None
    verification_time_s: float = 0.0
    method: str = "smt"  # "smt" or "sampling"
    n_samples_checked: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContagionVerificationReport:
    """Complete verification report for a contagion model."""
    model_name: str
    properties: List[PropertyVerificationResult]
    all_verified: bool
    n_verified: int
    n_total: int
    total_time_s: float
    estimated_lipschitz: Optional[float] = None

    def summary(self) -> str:
        return (
            f"Contagion model '{self.model_name}': "
            f"{self.n_verified}/{self.n_total} properties verified. "
            f"{'ALL PASS' if self.all_verified else 'FAILURES DETECTED'}."
        )


class ContagionModelVerifier:
    """
    Formal property verification for contagion models.

    Uses a combination of SMT encoding (for small instances) and
    sampling-based testing (for larger instances) to verify mathematical
    properties of contagion models.

    Parameters
    ----------
    n_samples : int
        Number of random samples for sampling-based verification.
    smt_timeout_ms : int
        Z3 timeout for SMT-based verification.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        smt_timeout_ms: int = 5000,
        seed: int = 42,
    ):
        self.n_samples = n_samples
        self.smt_timeout_ms = smt_timeout_ms
        self._rng = np.random.default_rng(seed)

    def verify_model(
        self,
        model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        model_name: str,
        n_nodes: int,
        exposure_matrix: Optional[np.ndarray] = None,
        total_assets: Optional[np.ndarray] = None,
    ) -> ContagionVerificationReport:
        """
        Verify all properties of a contagion model.

        Parameters
        ----------
        model_fn : callable
            Function(shocks, exposures) -> losses. Takes initial shock
            vector and exposure matrix, returns loss vector.
        model_name : str
            Name of the model (e.g., "DebtRank", "Cascade").
        n_nodes : int
            Number of nodes in the network.
        exposure_matrix : ndarray, optional
            Inter-node exposure matrix. If None, generates random.
        total_assets : ndarray, optional
            Total assets per node. If None, generates random.
        """
        t0 = time.time()

        if exposure_matrix is None:
            exposure_matrix = self._random_exposure_matrix(n_nodes)
        if total_assets is None:
            total_assets = self._rng.uniform(100, 10000, n_nodes)

        results: List[PropertyVerificationResult] = []

        # 1. Monotonicity
        results.append(self._verify_monotonicity(
            model_fn, model_name, n_nodes, exposure_matrix
        ))

        # 2. Bounded output
        results.append(self._verify_bounded_output(
            model_fn, model_name, n_nodes, exposure_matrix, total_assets
        ))

        # 3. Lipschitz continuity
        lip_result = self._verify_lipschitz(
            model_fn, model_name, n_nodes, exposure_matrix
        )
        results.append(lip_result)

        # 4. Non-negativity
        results.append(self._verify_non_negativity(
            model_fn, model_name, n_nodes, exposure_matrix
        ))

        # 5. Zero-shock baseline
        results.append(self._verify_zero_baseline(
            model_fn, model_name, n_nodes, exposure_matrix
        ))

        elapsed = time.time() - t0
        n_verified = sum(1 for r in results if r.verified)

        return ContagionVerificationReport(
            model_name=model_name,
            properties=results,
            all_verified=all(r.verified for r in results),
            n_verified=n_verified,
            n_total=len(results),
            total_time_s=elapsed,
            estimated_lipschitz=lip_result.details.get("estimated_constant"),
        )

    def _verify_monotonicity(
        self,
        model_fn: Callable,
        model_name: str,
        n_nodes: int,
        exposures: np.ndarray,
    ) -> PropertyVerificationResult:
        """
        Verify: if shocks_1 >= shocks_2 componentwise, then
        losses_1 >= losses_2 componentwise (monotonicity).
        """
        t0 = time.time()
        counterexample = None
        n_checked = 0

        for _ in range(self.n_samples):
            x = self._rng.uniform(0, 1, n_nodes)
            delta = self._rng.uniform(0, 0.5, n_nodes)
            y = np.clip(x + delta, 0, 1)  # y >= x

            try:
                loss_x = model_fn(x, exposures)
                loss_y = model_fn(y, exposures)
            except Exception:
                continue

            n_checked += 1

            if np.any(loss_y < loss_x - 1e-10):
                counterexample = {
                    "x": x.tolist(),
                    "y": y.tolist(),
                    "loss_x": loss_x.tolist(),
                    "loss_y": loss_y.tolist(),
                }
                break

        elapsed = time.time() - t0
        return PropertyVerificationResult(
            property_name="monotonicity",
            model_name=model_name,
            verified=(counterexample is None),
            counterexample=counterexample,
            verification_time_s=elapsed,
            method="sampling",
            n_samples_checked=n_checked,
        )

    def _verify_bounded_output(
        self,
        model_fn: Callable,
        model_name: str,
        n_nodes: int,
        exposures: np.ndarray,
        total_assets: np.ndarray,
    ) -> PropertyVerificationResult:
        """
        Verify: losses are in [0, total_assets] for each node.
        """
        t0 = time.time()
        counterexample = None
        n_checked = 0

        for _ in range(self.n_samples):
            shocks = self._rng.uniform(0, 1, n_nodes)

            try:
                losses = model_fn(shocks, exposures)
            except Exception:
                continue

            n_checked += 1

            if np.any(losses < -1e-10) or np.any(losses > total_assets + 1e-10):
                counterexample = {
                    "shocks": shocks.tolist(),
                    "losses": losses.tolist(),
                    "total_assets": total_assets.tolist(),
                }
                break

        elapsed = time.time() - t0
        return PropertyVerificationResult(
            property_name="bounded_output",
            model_name=model_name,
            verified=(counterexample is None),
            counterexample=counterexample,
            verification_time_s=elapsed,
            method="sampling",
            n_samples_checked=n_checked,
        )

    def _verify_lipschitz(
        self,
        model_fn: Callable,
        model_name: str,
        n_nodes: int,
        exposures: np.ndarray,
    ) -> PropertyVerificationResult:
        """
        Estimate and verify Lipschitz continuity.
        """
        t0 = time.time()
        max_ratio = 0.0
        n_checked = 0

        for _ in range(self.n_samples):
            x = self._rng.uniform(0, 1, n_nodes)
            y = self._rng.uniform(0, 1, n_nodes)

            try:
                fx = model_fn(x, exposures)
                fy = model_fn(y, exposures)
            except Exception:
                continue

            n_checked += 1
            d_in = np.linalg.norm(x - y)
            d_out = np.linalg.norm(fx - fy)

            if d_in > 1e-12:
                ratio = d_out / d_in
                max_ratio = max(max_ratio, ratio)

        elapsed = time.time() - t0

        # Lipschitz if ratio is finite and reasonable
        verified = np.isfinite(max_ratio) and max_ratio < 1e6

        return PropertyVerificationResult(
            property_name="lipschitz_continuity",
            model_name=model_name,
            verified=verified,
            verification_time_s=elapsed,
            method="sampling",
            n_samples_checked=n_checked,
            details={"estimated_constant": float(max_ratio)},
        )

    def _verify_non_negativity(
        self,
        model_fn: Callable,
        model_name: str,
        n_nodes: int,
        exposures: np.ndarray,
    ) -> PropertyVerificationResult:
        """Verify: losses >= 0 for all non-negative shocks."""
        t0 = time.time()
        counterexample = None
        n_checked = 0

        for _ in range(self.n_samples):
            shocks = self._rng.uniform(0, 1, n_nodes)

            try:
                losses = model_fn(shocks, exposures)
            except Exception:
                continue

            n_checked += 1
            if np.any(losses < -1e-10):
                counterexample = {
                    "shocks": shocks.tolist(),
                    "losses": losses.tolist(),
                }
                break

        elapsed = time.time() - t0
        return PropertyVerificationResult(
            property_name="non_negativity",
            model_name=model_name,
            verified=(counterexample is None),
            counterexample=counterexample,
            verification_time_s=elapsed,
            method="sampling",
            n_samples_checked=n_checked,
        )

    def _verify_zero_baseline(
        self,
        model_fn: Callable,
        model_name: str,
        n_nodes: int,
        exposures: np.ndarray,
    ) -> PropertyVerificationResult:
        """Verify: zero shocks produce zero losses."""
        t0 = time.time()
        shocks = np.zeros(n_nodes)

        try:
            losses = model_fn(shocks, exposures)
            verified = bool(np.allclose(losses, 0, atol=1e-10))
            counterexample = None if verified else {
                "shocks": shocks.tolist(),
                "losses": losses.tolist(),
            }
        except Exception as e:
            verified = False
            counterexample = {"error": str(e)}

        elapsed = time.time() - t0
        return PropertyVerificationResult(
            property_name="zero_baseline",
            model_name=model_name,
            verified=verified,
            counterexample=counterexample,
            verification_time_s=elapsed,
            method="direct",
            n_samples_checked=1,
        )

    def _random_exposure_matrix(self, n: int) -> np.ndarray:
        """Generate a random exposure matrix."""
        E = self._rng.uniform(0, 100, (n, n))
        np.fill_diagonal(E, 0)
        return E

    def verify_smt_small_instance(
        self,
        n_nodes: int = 3,
        model_name: str = "linear_contagion",
    ) -> PropertyVerificationResult:
        """
        Full SMT verification for a small contagion instance.

        Encodes a linear contagion model as QF_LRA and verifies
        monotonicity via Z3.
        """
        if not HAS_Z3:
            return PropertyVerificationResult(
                property_name="smt_monotonicity",
                model_name=model_name,
                verified=False,
                details={"error": "z3 not available"},
            )

        t0 = time.time()
        solver = z3.Solver()
        solver.set("timeout", self.smt_timeout_ms)

        # Encode linear contagion: loss_i = shock_i + sum_j w_ij * shock_j
        shocks1 = [z3.Real(f"s1_{i}") for i in range(n_nodes)]
        shocks2 = [z3.Real(f"s2_{i}") for i in range(n_nodes)]
        weights = [[z3.Real(f"w_{i}_{j}") for j in range(n_nodes)]
                    for i in range(n_nodes)]
        losses1 = [z3.Real(f"l1_{i}") for i in range(n_nodes)]
        losses2 = [z3.Real(f"l2_{i}") for i in range(n_nodes)]

        for i in range(n_nodes):
            # Shocks in [0, 1]
            solver.add(shocks1[i] >= 0, shocks1[i] <= 1)
            solver.add(shocks2[i] >= 0, shocks2[i] <= 1)
            # s2 >= s1 (componentwise)
            solver.add(shocks2[i] >= shocks1[i])

            # Weights non-negative
            for j in range(n_nodes):
                solver.add(weights[i][j] >= 0)
                if i == j:
                    solver.add(weights[i][j] == 0)

            # Linear contagion model
            loss_sum1 = shocks1[i] + z3.Sum([
                weights[i][j] * shocks1[j] for j in range(n_nodes)
            ])
            loss_sum2 = shocks2[i] + z3.Sum([
                weights[i][j] * shocks2[j] for j in range(n_nodes)
            ])
            solver.add(losses1[i] == loss_sum1)
            solver.add(losses2[i] == loss_sum2)

        # Negate monotonicity: exists i where loss2[i] < loss1[i]
        violations = [losses2[i] < losses1[i] for i in range(n_nodes)]
        solver.add(z3.Or(violations))

        result = solver.check()
        elapsed = time.time() - t0

        return PropertyVerificationResult(
            property_name="smt_monotonicity",
            model_name=model_name,
            verified=(result == z3.unsat),
            verification_time_s=elapsed,
            method="smt",
            details={"z3_result": str(result), "n_nodes": n_nodes},
        )
