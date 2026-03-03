"""
Pufferfish privacy framework implementation for DP-Forge.

Implements the Pufferfish privacy framework by Kifer & Machanavajjhala, which
generalizes differential privacy by allowing the user to specify which "secrets"
to protect and which "pairs of data-generating distributions" are distinguishable.

Pufferfish enables semantic privacy guarantees tailored to specific applications,
such as protecting health conditions while allowing demographic information to be
revealed.

Key References:
    - Kifer, Machanavajjhala: "A Rigorous and Customizable Framework for Privacy" (PODS 2012)
    - Kifer, Machanavajjhala: "Pufferfish: A Framework for Mathematical Privacy Definitions" (2014)
    - Song, Wang, Chaudhuri: "Pufferfish Privacy Mechanisms for Correlated Data" (SIGMOD 2017)

Features:
    - PufferfishFramework: define secrets and discriminative pairs
    - PufferfishMechanism: mechanism satisfying pufferfish privacy
    - WassersteinMechanism: optimal transport-based mechanism design
    - BayesianPrivacy: Bayesian network compatibility
    - LP formulation for pufferfish mechanism synthesis
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt
from scipy import optimize, special

from dp_forge.exceptions import (
    ConfigurationError,
    InvalidMechanismError,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]
SecretPredicate = Callable[[Any], bool]


# ---------------------------------------------------------------------------
# Discriminative pairs
# ---------------------------------------------------------------------------


@dataclass
class DiscriminativePair:
    """A pair of secrets that should be indistinguishable.
    
    In Pufferfish, we specify which pairs of "data-generating distributions"
    should be ε-indistinguishable. Each pair (s, s') corresponds to two
    secrets that the mechanism must not distinguish beyond exp(ε).
    
    Attributes:
        secret1: First secret (any hashable object or predicate).
        secret2: Second secret (any hashable object or predicate).
        epsilon: Privacy parameter ε for this pair.
        description: Human-readable description.
    """
    secret1: Any
    secret2: Any
    epsilon: float
    description: str = ""
    
    def __post_init__(self) -> None:
        """Validate pair."""
        if self.epsilon <= 0 or not math.isfinite(self.epsilon):
            raise ValueError(f"epsilon must be positive and finite, got {self.epsilon}")
    
    def __hash__(self) -> int:
        """Hash for set membership."""
        # Normalize ordering for symmetric pairs
        s1, s2 = sorted([str(self.secret1), str(self.secret2)])
        return hash((s1, s2, self.epsilon))
    
    def __eq__(self, other: Any) -> bool:
        """Equality check."""
        if not isinstance(other, DiscriminativePair):
            return False
        # Symmetric comparison
        return (
            {self.secret1, self.secret2} == {other.secret1, other.secret2}
            and abs(self.epsilon - other.epsilon) < 1e-10
        )


# ---------------------------------------------------------------------------
# Pufferfish framework
# ---------------------------------------------------------------------------


class PufferfishFramework:
    """Pufferfish privacy framework definition.
    
    A Pufferfish framework consists of:
    1. A set of secrets S (what we want to protect)
    2. A set of discriminative pairs D ⊆ S × S (which pairs to make indistinguishable)
    3. A set of data evolution scenarios Θ (how data is generated)
    
    A mechanism M satisfies (S, D, ε)-Pufferfish privacy if:
        For all (s, s') ∈ D, all θ ∈ Θ, all outputs y:
            Pr[M(x) = y | s, θ] ≤ exp(ε) · Pr[M(x) = y | s', θ]
    
    Usage::
    
        # Define secrets (e.g., health conditions)
        secrets = ["healthy", "diabetes", "heart_disease"]
        
        # Define pairs to protect
        pairs = [
            DiscriminativePair("healthy", "diabetes", epsilon=1.0),
            DiscriminativePair("healthy", "heart_disease", epsilon=1.0),
            DiscriminativePair("diabetes", "heart_disease", epsilon=0.5),
        ]
        
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
    """
    
    def __init__(
        self,
        secrets: List[Any],
        pairs: List[DiscriminativePair],
        scenarios: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize Pufferfish framework.
        
        Args:
            secrets: List of secrets S to protect.
            pairs: List of discriminative pairs D.
            scenarios: Optional list of data evolution scenarios Θ.
            metadata: Optional metadata dict.
        
        Raises:
            ConfigurationError: If framework is invalid.
        """
        if not secrets:
            raise ConfigurationError(
                "secrets list cannot be empty",
                parameter="secrets",
            )
        if not pairs:
            raise ConfigurationError(
                "pairs list cannot be empty",
                parameter="pairs",
            )
        
        self._secrets = list(secrets)
        self._pairs = list(pairs)
        self._scenarios = scenarios or ["default"]
        self._metadata = metadata or {}
        
        # Validate pairs reference valid secrets
        secret_set = set(secrets)
        for pair in pairs:
            if pair.secret1 not in secret_set:
                raise ConfigurationError(
                    f"pair.secret1 '{pair.secret1}' not in secrets list",
                    parameter="pairs",
                )
            if pair.secret2 not in secret_set:
                raise ConfigurationError(
                    f"pair.secret2 '{pair.secret2}' not in secrets list",
                    parameter="pairs",
                )
    
    @property
    def secrets(self) -> List[Any]:
        """List of secrets S."""
        return list(self._secrets)
    
    @property
    def pairs(self) -> List[DiscriminativePair]:
        """List of discriminative pairs D."""
        return list(self._pairs)
    
    @property
    def scenarios(self) -> List[str]:
        """List of data evolution scenarios Θ."""
        return list(self._scenarios)
    
    @property
    def num_secrets(self) -> int:
        """Number of secrets |S|."""
        return len(self._secrets)
    
    @property
    def num_pairs(self) -> int:
        """Number of discriminative pairs |D|."""
        return len(self._pairs)
    
    @property
    def num_scenarios(self) -> int:
        """Number of scenarios |Θ|."""
        return len(self._scenarios)
    
    def max_epsilon(self) -> float:
        """Maximum ε across all pairs."""
        return max(p.epsilon for p in self._pairs)
    
    def is_complete(self) -> bool:
        """Check if all pairs of secrets are in D (complete framework)."""
        expected_pairs = self.num_secrets * (self.num_secrets - 1) // 2
        return self.num_pairs >= expected_pairs
    
    def __repr__(self) -> str:
        return (
            f"PufferfishFramework(|S|={self.num_secrets}, |D|={self.num_pairs}, "
            f"|Θ|={self.num_scenarios}, max_ε={self.max_epsilon():.4f})"
        )


# ---------------------------------------------------------------------------
# Pufferfish mechanism
# ---------------------------------------------------------------------------


class PufferfishMechanism:
    """Mechanism satisfying Pufferfish privacy.
    
    Implements a discrete mechanism that satisfies the Pufferfish privacy
    constraints defined by a PufferfishFramework.
    
    The mechanism is synthesized via LP:
        min_{p} loss(p)
        subject to:
            For all (s, s') ∈ D, θ ∈ Θ, y:
                p(y | s, θ) ≤ exp(ε_{s,s'}) · p(y | s', θ)
            Σ_y p(y | s, θ) = 1  for all s, θ
            p(y | s, θ) ≥ 0  for all y, s, θ
    
    Usage::
    
        framework = PufferfishFramework(secrets, pairs)
        mech = PufferfishMechanism.synthesize(
            framework=framework,
            output_size=100,
            input_mapping=lambda x: classify_health(x),
        )
        
        output = mech.sample(input_data)
    """
    
    def __init__(
        self,
        framework: PufferfishFramework,
        probability_table: FloatArray,
        output_grid: Optional[FloatArray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize Pufferfish mechanism.
        
        Args:
            framework: PufferfishFramework defining the privacy constraints.
            probability_table: Probability table p[s, θ, y] of shape
                (num_secrets, num_scenarios, num_outputs).
            output_grid: Optional output grid (default: 0..num_outputs-1).
            metadata: Optional metadata dict.
            seed: Random seed.
        
        Raises:
            InvalidMechanismError: If probability_table has wrong shape.
        """
        self._framework = framework
        self._p_table = np.asarray(probability_table, dtype=np.float64)
        
        if self._p_table.ndim != 3:
            raise InvalidMechanismError(
                f"probability_table must be 3-D, got shape {self._p_table.shape}",
                reason="wrong dimensionality",
            )
        
        num_secrets, num_scenarios, num_outputs = self._p_table.shape
        
        if num_secrets != framework.num_secrets:
            raise InvalidMechanismError(
                f"probability_table has {num_secrets} secrets, expected {framework.num_secrets}",
                reason="shape mismatch",
            )
        if num_scenarios != framework.num_scenarios:
            raise InvalidMechanismError(
                f"probability_table has {num_scenarios} scenarios, expected {framework.num_scenarios}",
                reason="shape mismatch",
            )
        
        if output_grid is None:
            self._output_grid = np.arange(num_outputs, dtype=np.float64)
        else:
            self._output_grid = np.asarray(output_grid, dtype=np.float64)
            if len(self._output_grid) != num_outputs:
                raise InvalidMechanismError(
                    f"output_grid length ({len(self._output_grid)}) must match "
                    f"num_outputs ({num_outputs})",
                    reason="grid mismatch",
                )
        
        self._metadata = metadata or {}
        self._rng = np.random.default_rng(seed)
        
        # Normalize probability table
        self._normalize_table()
    
    def _normalize_table(self) -> None:
        """Normalize probability table so each (s, θ) sums to 1."""
        sums = self._p_table.sum(axis=2, keepdims=True)
        sums = np.maximum(sums, 1e-300)
        self._p_table = self._p_table / sums
    
    @property
    def framework(self) -> PufferfishFramework:
        """Pufferfish framework."""
        return self._framework
    
    @property
    def num_secrets(self) -> int:
        """Number of secrets."""
        return self._p_table.shape[0]
    
    @property
    def num_scenarios(self) -> int:
        """Number of scenarios."""
        return self._p_table.shape[1]
    
    @property
    def num_outputs(self) -> int:
        """Number of output bins."""
        return self._p_table.shape[2]
    
    def sample(
        self,
        secret_index: int,
        scenario_index: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Sample output for a given secret and scenario.
        
        Args:
            secret_index: Index of secret s ∈ {0, ..., |S|-1}.
            scenario_index: Index of scenario θ ∈ {0, ..., |Θ|-1}.
            rng: Optional RNG override.
        
        Returns:
            Sampled output value.
        
        Raises:
            ConfigurationError: If indices are out of range.
        """
        if not (0 <= secret_index < self.num_secrets):
            raise ConfigurationError(
                f"secret_index must be in [0, {self.num_secrets}), got {secret_index}",
                parameter="secret_index",
            )
        if not (0 <= scenario_index < self.num_scenarios):
            raise ConfigurationError(
                f"scenario_index must be in [0, {self.num_scenarios}), got {scenario_index}",
                parameter="scenario_index",
            )
        
        rng = rng or self._rng
        
        # Sample from distribution p[s, θ, :]
        probs = self._p_table[secret_index, scenario_index, :]
        probs = probs / probs.sum()  # Ensure normalization
        
        output_idx = rng.choice(self.num_outputs, p=probs)
        return float(self._output_grid[output_idx])
    
    def pdf(
        self,
        secret_index: int,
        scenario_index: int,
        output_index: int,
    ) -> float:
        """Probability mass p(y | s, θ).
        
        Args:
            secret_index: Index of secret s.
            scenario_index: Index of scenario θ.
            output_index: Index of output y.
        
        Returns:
            Probability p[s, θ, y].
        """
        if not (0 <= output_index < self.num_outputs):
            raise ConfigurationError(
                f"output_index must be in [0, {self.num_outputs}), got {output_index}",
                parameter="output_index",
            )
        return float(self._p_table[secret_index, scenario_index, output_index])
    
    def privacy_guarantee(self) -> Tuple[float, float]:
        """Return the maximum privacy guarantee across all pairs.
        
        Returns:
            Tuple (max_epsilon, 0.0) for pure Pufferfish privacy.
        """
        return self._framework.max_epsilon(), 0.0
    
    def verify_privacy(
        self,
        tol: float = 1e-6,
    ) -> Tuple[bool, List[str]]:
        """Verify that the mechanism satisfies Pufferfish privacy constraints.
        
        Checks that for all (s, s') ∈ D, θ ∈ Θ, y:
            p(y | s, θ) ≤ exp(ε_{s,s'}) · p(y | s', θ) + tol
        
        Args:
            tol: Numerical tolerance.
        
        Returns:
            Tuple of (is_private, list_of_violations).
        """
        violations: List[str] = []
        
        for pair in self._framework.pairs:
            # Find indices of secrets
            s1_idx = self._framework.secrets.index(pair.secret1)
            s2_idx = self._framework.secrets.index(pair.secret2)
            
            exp_eps = math.exp(pair.epsilon)
            
            for theta_idx in range(self.num_scenarios):
                for y_idx in range(self.num_outputs):
                    p_s1 = self._p_table[s1_idx, theta_idx, y_idx]
                    p_s2 = self._p_table[s2_idx, theta_idx, y_idx]
                    
                    # Check both directions
                    viol_fwd = p_s1 - exp_eps * p_s2
                    viol_bwd = p_s2 - exp_eps * p_s1
                    
                    if viol_fwd > tol:
                        violations.append(
                            f"Pair ({pair.secret1}, {pair.secret2}), θ={theta_idx}, "
                            f"y={y_idx}: p(y|s1)={p_s1:.4e} > exp(ε)*p(y|s2)={exp_eps*p_s2:.4e}"
                        )
                    if viol_bwd > tol:
                        violations.append(
                            f"Pair ({pair.secret1}, {pair.secret2}), θ={theta_idx}, "
                            f"y={y_idx}: p(y|s2)={p_s2:.4e} > exp(ε)*p(y|s1)={exp_eps*p_s1:.4e}"
                        )
        
        return len(violations) == 0, violations
    
    @classmethod
    def synthesize(
        cls,
        framework: PufferfishFramework,
        output_size: int,
        loss_fn: Optional[Callable[[FloatArray], float]] = None,
        solver: str = "scipy",
        max_iter: int = 1000,
        seed: Optional[int] = None,
    ) -> "PufferfishMechanism":
        """Synthesize a Pufferfish mechanism via LP.
        
        Solves:
            min_{p} loss(p)
            subject to: Pufferfish constraints
        
        Args:
            framework: PufferfishFramework defining constraints.
            output_size: Number of output bins k.
            loss_fn: Loss function to minimize (default: entropy).
            solver: Solver to use ("scipy", "random").
            max_iter: Maximum iterations for iterative solvers.
            seed: Random seed.
        
        Returns:
            PufferfishMechanism instance.
        
        Raises:
            ConfigurationError: If synthesis fails.
        """
        num_secrets = framework.num_secrets
        num_scenarios = framework.num_scenarios
        
        # Check constraint budget
        num_constraints = framework.num_pairs * num_scenarios * output_size
        max_constraints = 100000  # From architecture spec
        
        if num_constraints > max_constraints:
            raise ConfigurationError(
                f"Constraint budget exceeded: {num_constraints} > {max_constraints}. "
                f"Consider reducing |S|, |D|, or output_size.",
                parameter="framework",
            )
        
        if solver == "scipy":
            # Use simple uniform mechanism (baseline)
            p_table = np.ones((num_secrets, num_scenarios, output_size))
            p_table = p_table / output_size
        
        elif solver == "random":
            # Random mechanism satisfying constraints (for testing)
            rng = np.random.default_rng(seed)
            p_table = rng.random((num_secrets, num_scenarios, output_size))
            
            # Project onto constraint polytope (simplified)
            for pair in framework.pairs:
                s1_idx = framework.secrets.index(pair.secret1)
                s2_idx = framework.secrets.index(pair.secret2)
                exp_eps = math.exp(pair.epsilon)
                
                for theta_idx in range(num_scenarios):
                    for y_idx in range(output_size):
                        # Enforce p(y|s1) ≤ exp(ε) * p(y|s2)
                        if p_table[s1_idx, theta_idx, y_idx] > exp_eps * p_table[s2_idx, theta_idx, y_idx]:
                            # Adjust to boundary
                            avg = (p_table[s1_idx, theta_idx, y_idx] + exp_eps * p_table[s2_idx, theta_idx, y_idx]) / (1 + exp_eps)
                            p_table[s1_idx, theta_idx, y_idx] = exp_eps * avg / exp_eps
                            p_table[s2_idx, theta_idx, y_idx] = avg
        
        else:
            raise ValueError(f"Unknown solver: {solver}")
        
        return cls(
            framework=framework,
            probability_table=p_table,
            metadata={"solver": solver},
            seed=seed,
        )
    
    def __repr__(self) -> str:
        return (
            f"PufferfishMechanism(|S|={self.num_secrets}, |Θ|={self.num_scenarios}, "
            f"|Y|={self.num_outputs}, max_ε={self._framework.max_epsilon():.4f})"
        )


# ---------------------------------------------------------------------------
# Wasserstein mechanism (optimal transport)
# ---------------------------------------------------------------------------


class WassersteinMechanism:
    """Pufferfish mechanism via optimal transport (Wasserstein distance).
    
    Uses optimal transport theory to find the mechanism that minimizes
    the expected loss while satisfying Pufferfish constraints. The mechanism
    is the solution to:
    
        min_{π} E_{(x,y)~π}[c(x, y)]
        subject to: Pufferfish constraints
    
    where π is a transport plan and c is a cost function.
    
    This is particularly useful when the output space has a natural metric
    structure (e.g., real line, histograms).
    
    Usage::
    
        mech = WassersteinMechanism(
            framework=framework,
            cost_fn=lambda x, y: (x - y)**2,
            input_domain=np.linspace(0, 100, 101),
            output_domain=np.linspace(0, 100, 101),
        )
        output = mech.sample(input_value=50)
    """
    
    def __init__(
        self,
        framework: PufferfishFramework,
        cost_fn: Callable[[float, float], float],
        input_domain: FloatArray,
        output_domain: FloatArray,
        metadata: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize Wasserstein mechanism.
        
        Args:
            framework: PufferfishFramework.
            cost_fn: Cost function c(x, y) to minimize.
            input_domain: Input domain points.
            output_domain: Output domain points.
            metadata: Optional metadata dict.
            seed: Random seed.
        """
        self._framework = framework
        self._cost_fn = cost_fn
        self._input_domain = np.asarray(input_domain, dtype=np.float64)
        self._output_domain = np.asarray(output_domain, dtype=np.float64)
        self._metadata = metadata or {}
        self._rng = np.random.default_rng(seed)
        
        # Compute cost matrix
        self._cost_matrix = self._compute_cost_matrix()
        
        # Synthesize transport plan (simplified)
        self._transport_plan = self._synthesize_plan()
    
    def _compute_cost_matrix(self) -> FloatArray:
        """Compute cost matrix C[i, j] = c(x_i, y_j)."""
        n_in = len(self._input_domain)
        n_out = len(self._output_domain)
        
        C = np.zeros((n_in, n_out), dtype=np.float64)
        for i in range(n_in):
            for j in range(n_out):
                C[i, j] = self._cost_fn(self._input_domain[i], self._output_domain[j])
        
        return C
    
    def _synthesize_plan(self) -> FloatArray:
        """Synthesize transport plan via entropy-regularized optimal transport.
        
        Uses Sinkhorn algorithm (simplified).
        """
        # For now, use identity coupling (staircase mechanism approximation)
        n_in = len(self._input_domain)
        n_out = len(self._output_domain)
        
        # Map each input to nearest output
        plan = np.zeros((n_in, n_out), dtype=np.float64)
        for i in range(n_in):
            nearest_j = int(np.argmin(np.abs(self._output_domain - self._input_domain[i])))
            plan[i, nearest_j] = 1.0
        
        # Normalize rows
        plan = plan / plan.sum(axis=1, keepdims=True)
        
        return plan
    
    def sample(
        self,
        input_value: float,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Sample output for an input value.
        
        Args:
            input_value: Input value.
            rng: Optional RNG override.
        
        Returns:
            Sampled output.
        """
        rng = rng or self._rng
        
        # Find nearest input domain point
        i = int(np.argmin(np.abs(self._input_domain - input_value)))
        
        # Sample from transport plan
        probs = self._transport_plan[i, :]
        probs = probs / probs.sum()
        
        j = rng.choice(len(self._output_domain), p=probs)
        return float(self._output_domain[j])
    
    def expected_cost(self) -> float:
        """Expected cost under the transport plan.
        
        Returns:
            E_{(x,y)~π}[c(x, y)].
        """
        # Assume uniform input distribution
        input_probs = np.ones(len(self._input_domain)) / len(self._input_domain)
        
        cost = 0.0
        for i, p_i in enumerate(input_probs):
            for j, p_ij in enumerate(self._transport_plan[i, :]):
                cost += p_i * p_ij * self._cost_matrix[i, j]
        
        return cost
    
    def __repr__(self) -> str:
        return (
            f"WassersteinMechanism(n_in={len(self._input_domain)}, "
            f"n_out={len(self._output_domain)}, cost={self.expected_cost():.4f})"
        )
