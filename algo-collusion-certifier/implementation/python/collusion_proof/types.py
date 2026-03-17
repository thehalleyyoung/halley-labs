"""Domain types for the CollusionProof certification system.

Provides Pydantic models, enums, and dataclasses for representing market
games, pricing trajectories, statistical test results, and certification
verdicts used throughout the collusion-detection pipeline.
"""

import enum
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

# ── Primitive Type Aliases ───────────────────────────────────────────────────

Price = float
Quantity = float
Profit = float


# ── Enums ────────────────────────────────────────────────────────────────────


class MarketType(str, enum.Enum):
    """Type of strategic interaction in the market."""

    BERTRAND = "bertrand"
    COURNOT = "cournot"
    DIFFERENTIATED = "differentiated"
    AUCTION = "auction"
    SEQUENTIAL = "sequential"


class DemandSystem(str, enum.Enum):
    """Functional form of the demand curve."""

    LINEAR = "linear"
    LOGIT = "logit"
    CES = "ces"
    ALMOST_IDEAL = "almost_ideal"


class OracleAccessLevel(str, enum.Enum):
    """Degree of algorithmic transparency available to the detector."""

    PASSIVE = "passive"
    CHECKPOINT = "checkpoint"
    REWIND = "rewind"


class EvaluationMode(str, enum.Enum):
    """How thorough the evaluation suite should be."""

    SMOKE = "smoke"
    STANDARD = "standard"
    FULL = "full"
    ADVERSARIAL = "adversarial"


class Verdict(str, enum.Enum):
    """Final classification of observed market behaviour."""

    COMPETITIVE = "competitive"
    SUSPICIOUS = "suspicious"
    COLLUSIVE = "collusive"
    INCONCLUSIVE = "inconclusive"

    @property
    def is_harmful(self) -> bool:
        return self in (Verdict.COLLUSIVE, Verdict.SUSPICIOUS)


class ScenarioCategory(str, enum.Enum):
    """Category tag for benchmark scenarios."""

    COLLUSIVE = "collusive"
    COMPETITIVE = "competitive"
    BOUNDARY = "boundary"
    ADVERSARIAL = "adversarial"


class TestTier(str, enum.Enum):
    """Hierarchical tier in the sequential testing procedure."""

    TIER1_PRICE_LEVEL = "tier1_price_level"
    TIER2_CORRELATION = "tier2_correlation"
    TIER3_PUNISHMENT = "tier3_punishment"
    TIER4_COUNTERFACTUAL = "tier4_counterfactual"

    @property
    def order(self) -> int:
        return int(self.value[4])


class NullHypothesis(str, enum.Enum):
    """Null hypothesis for each testing tier."""

    COMPETITIVE_EQUILIBRIUM = "competitive_eq"
    INDEPENDENT_PLAY = "independent_play"
    NO_PUNISHMENT = "no_punishment"
    NO_SUPRACOMPETITIVE = "no_supracompetitive"


# ── Core Data Models (Pydantic) ─────────────────────────────────────────────


class PlayerAction(BaseModel):
    """Single action taken by a player in one round."""

    player_id: int
    round_num: int
    price: Optional[Price] = None
    quantity: Optional[Quantity] = None
    profit: Optional[Profit] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("price")
    @classmethod
    def price_non_negative(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("price must be >= 0")
        return v

    @field_validator("quantity")
    @classmethod
    def quantity_non_negative(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("quantity must be >= 0")
        return v

    def __repr__(self) -> str:
        parts = [f"player={self.player_id}", f"round={self.round_num}"]
        if self.price is not None:
            parts.append(f"p={self.price:.4f}")
        if self.quantity is not None:
            parts.append(f"q={self.quantity:.4f}")
        if self.profit is not None:
            parts.append(f"π={self.profit:.4f}")
        return f"PlayerAction({', '.join(parts)})"


class MarketOutcome(BaseModel):
    """Outcome of a single market round."""

    round_num: int
    actions: List[PlayerAction]
    market_price: Optional[Price] = None
    total_quantity: Optional[Quantity] = None
    consumer_surplus: Optional[float] = None
    producer_surplus: Optional[float] = None

    @field_validator("market_price")
    @classmethod
    def market_price_non_negative(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("market_price must be >= 0")
        return v

    @field_validator("total_quantity")
    @classmethod
    def total_quantity_non_negative(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("total_quantity must be >= 0")
        return v

    @property
    def prices(self) -> List[Price]:
        """All non-None player prices in this round."""
        return [a.price for a in self.actions if a.price is not None]

    @property
    def profits(self) -> List[Profit]:
        """All non-None player profits in this round."""
        return [a.profit for a in self.actions if a.profit is not None]

    @property
    def num_players(self) -> int:
        return len(self.actions)

    @property
    def mean_price(self) -> Optional[float]:
        p = self.prices
        return float(np.mean(p)) if p else None

    @property
    def total_welfare(self) -> Optional[float]:
        cs = self.consumer_surplus
        ps = self.producer_surplus
        if cs is not None and ps is not None:
            return cs + ps
        return None

    def __repr__(self) -> str:
        mp = f", mean_p={self.mean_price:.4f}" if self.mean_price is not None else ""
        return f"MarketOutcome(round={self.round_num}, n={self.num_players}{mp})"


class PriceTrajectory(BaseModel):
    """Complete price trajectory across all rounds for all players."""

    outcomes: List[MarketOutcome]
    num_players: int
    num_rounds: int
    market_type: MarketType = MarketType.BERTRAND

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def check_lengths(self) -> "PriceTrajectory":
        if len(self.outcomes) != self.num_rounds:
            raise ValueError(
                f"Expected {self.num_rounds} outcomes, got {len(self.outcomes)}"
            )
        return self

    # ── matrix helpers ───────────────────────────────────────────────────

    def get_prices_matrix(self) -> np.ndarray:
        """Return an (num_rounds x num_players) price matrix.  Missing values
        are filled with NaN."""
        mat = np.full((self.num_rounds, self.num_players), np.nan)
        for outcome in self.outcomes:
            for action in outcome.actions:
                if action.price is not None and action.player_id < self.num_players:
                    mat[outcome.round_num, action.player_id] = action.price
        return mat

    def get_profits_matrix(self) -> np.ndarray:
        """Return an (num_rounds x num_players) profit matrix."""
        mat = np.full((self.num_rounds, self.num_players), np.nan)
        for outcome in self.outcomes:
            for action in outcome.actions:
                if action.profit is not None and action.player_id < self.num_players:
                    mat[outcome.round_num, action.player_id] = action.profit
        return mat

    def get_player_prices(self, player_id: int) -> List[float]:
        """Prices set by *player_id* across all rounds (NaN for missing)."""
        prices: List[float] = []
        for outcome in self.outcomes:
            found = False
            for action in outcome.actions:
                if action.player_id == player_id and action.price is not None:
                    prices.append(action.price)
                    found = True
                    break
            if not found:
                prices.append(float("nan"))
        return prices

    def get_round_prices(self, round_num: int) -> List[float]:
        """Prices from every player in a given round."""
        if 0 <= round_num < len(self.outcomes):
            return self.outcomes[round_num].prices
        return []

    def tail(self, n: int) -> "PriceTrajectory":
        """Return a new trajectory containing only the last *n* rounds."""
        n = min(n, self.num_rounds)
        sliced = self.outcomes[-n:]
        return PriceTrajectory(
            outcomes=sliced,
            num_players=self.num_players,
            num_rounds=n,
            market_type=self.market_type,
        )

    def convergence_window(self, threshold: float = 0.01) -> Optional[int]:
        """Return the first round after which all subsequent per-round mean
        prices vary by less than *threshold* from their running mean.  Returns
        ``None`` if the series never converges."""
        means = [o.mean_price for o in self.outcomes]
        if any(m is None for m in means):
            return None
        arr = np.array(means, dtype=float)
        if len(arr) < 2:
            return None
        final_mean = float(np.mean(arr[-max(1, len(arr) // 10) :]))
        for i in range(len(arr)):
            if np.all(np.abs(arr[i:] - final_mean) < threshold):
                return i
        return None

    def __repr__(self) -> str:
        return (
            f"PriceTrajectory(players={self.num_players}, "
            f"rounds={self.num_rounds}, type={self.market_type.value})"
        )


# ── Configuration Models ─────────────────────────────────────────────────────


class GameConfig(BaseModel):
    """Configuration for a repeated game environment."""

    num_players: int = Field(ge=2, le=100)
    num_rounds: int = Field(ge=10, le=1_000_000)
    num_actions: int = Field(ge=2, le=1000, default=10)
    market_type: MarketType = MarketType.BERTRAND
    demand_system: DemandSystem = DemandSystem.LINEAR
    discount_factor: float = Field(ge=0.0, le=1.0, default=0.95)
    marginal_cost: float = Field(ge=0.0, default=1.0)
    monopoly_price: Optional[float] = None
    nash_price: Optional[float] = None
    demand_intercept: float = Field(ge=0.0, default=10.0)
    demand_slope: float = Field(gt=0.0, default=1.0)
    noise_std: float = Field(ge=0.0, default=0.0)
    differentiation: float = Field(ge=0.0, le=1.0, default=0.0)

    @model_validator(mode="after")
    def compute_equilibrium_prices(self) -> "GameConfig":
        """Fill in Nash and monopoly prices for linear Bertrand when omitted."""
        if self.demand_system == DemandSystem.LINEAR:
            if self.nash_price is None:
                if self.market_type == MarketType.BERTRAND:
                    self.nash_price = self.marginal_cost
                elif self.market_type == MarketType.COURNOT:
                    n = self.num_players
                    self.nash_price = (
                        self.demand_intercept + n * self.marginal_cost
                    ) / (n + 1)
            if self.monopoly_price is None:
                self.monopoly_price = (
                    self.demand_intercept + self.marginal_cost
                ) / 2.0
        return self

    @property
    def price_range(self) -> Tuple[float, float]:
        """Theoretical price range from Nash to monopoly."""
        lo = self.nash_price if self.nash_price is not None else self.marginal_cost
        hi = (
            self.monopoly_price
            if self.monopoly_price is not None
            else self.demand_intercept
        )
        return (lo, hi)

    def __repr__(self) -> str:
        return (
            f"GameConfig(n={self.num_players}, T={self.num_rounds}, "
            f"{self.market_type.value}/{self.demand_system.value})"
        )


class AlgorithmConfig(BaseModel):
    """Configuration for a pricing algorithm under test."""

    name: str
    algorithm_type: str = "q_learning"
    learning_rate: float = Field(ge=0.0, le=1.0, default=0.1)
    discount_factor: float = Field(ge=0.0, le=1.0, default=0.95)
    epsilon: float = Field(ge=0.0, le=1.0, default=0.1)
    epsilon_decay: float = Field(ge=0.0, le=1.0, default=0.999)
    epsilon_min: float = Field(ge=0.0, le=1.0, default=0.01)
    memory_length: int = Field(ge=0, default=1)
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def epsilon_min_le_epsilon(self) -> "AlgorithmConfig":
        if self.epsilon_min > self.epsilon:
            raise ValueError("epsilon_min must be <= epsilon")
        return self

    def effective_epsilon(self, step: int) -> float:
        """Return the ε-greedy exploration rate at the given training step."""
        decayed = self.epsilon * (self.epsilon_decay ** step)
        return max(decayed, self.epsilon_min)

    def __repr__(self) -> str:
        return f"AlgorithmConfig(name={self.name!r}, type={self.algorithm_type!r})"


# ── Statistical Results ──────────────────────────────────────────────────────


class ConfidenceInterval(BaseModel):
    """A confidence interval with level and bounds."""

    lower: float
    upper: float
    level: float = Field(ge=0.0, le=1.0, default=0.95)
    point_estimate: Optional[float] = None
    method: str = "normal"

    @model_validator(mode="after")
    def lower_le_upper(self) -> "ConfidenceInterval":
        if self.lower > self.upper:
            raise ValueError(
                f"lower ({self.lower}) must be <= upper ({self.upper})"
            )
        return self

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2.0

    def contains(self, value: float) -> bool:
        """Return True if *value* lies within the interval (inclusive)."""
        return self.lower <= value <= self.upper

    def __repr__(self) -> str:
        pe = f", est={self.point_estimate:.4f}" if self.point_estimate is not None else ""
        return (
            f"CI({self.level:.0%}: [{self.lower:.4f}, {self.upper:.4f}]{pe})"
        )


class HypothesisTestResult(BaseModel):
    """Result of a single hypothesis test."""

    test_name: str
    null_hypothesis: str
    alternative: str = "two-sided"
    test_statistic: float
    p_value: float = Field(ge=0.0, le=1.0)
    reject: bool
    alpha: float = Field(ge=0.0, le=1.0, default=0.05)
    confidence_interval: Optional[ConfidenceInterval] = None
    effect_size: Optional[float] = None
    power: Optional[float] = None
    sample_size: int = 0
    method: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("power")
    @classmethod
    def power_in_unit_interval(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("power must be in [0, 1]")
        return v

    @property
    def is_significant(self) -> bool:
        return self.p_value < self.alpha

    def __repr__(self) -> str:
        sig = "reject" if self.reject else "fail-to-reject"
        return (
            f"HypothesisTestResult({self.test_name!r}, "
            f"p={self.p_value:.4g}, {sig})"
        )


class TestResult(BaseModel):
    """Result of a composite test battery for one tier."""

    tier: TestTier
    null_hypothesis: NullHypothesis
    test_results: List[HypothesisTestResult]
    combined_p_value: Optional[float] = None
    combined_reject: bool = False
    alpha_spent: float = 0.0
    evidence_strength: float = Field(ge=0.0, le=1.0, default=0.0)
    details: Dict[str, Any] = Field(default_factory=dict)

    @property
    def num_rejections(self) -> int:
        return sum(1 for t in self.test_results if t.reject)

    @property
    def rejection_rate(self) -> float:
        if not self.test_results:
            return 0.0
        return self.num_rejections / len(self.test_results)

    @property
    def min_p_value(self) -> Optional[float]:
        if not self.test_results:
            return None
        return min(t.p_value for t in self.test_results)

    def __repr__(self) -> str:
        rej = "REJECT" if self.combined_reject else "ACCEPT"
        return (
            f"TestResult({self.tier.value}, {rej}, "
            f"tests={len(self.test_results)}, "
            f"rejections={self.num_rejections})"
        )


class CollusionPremiumResult(BaseModel):
    """Result of collusion premium computation."""

    premium: float
    premium_ci: ConfidenceInterval
    collusion_index: float = Field(ge=0.0, le=1.0)
    collusion_index_ci: ConfidenceInterval
    nash_price: float
    monopoly_price: float
    observed_price: float
    absolute_margin: float
    demand_robust: bool = True
    bootstrap_samples: int = 1000
    details: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("nash_price")
    @classmethod
    def nash_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("nash_price must be >= 0")
        return v

    @field_validator("monopoly_price")
    @classmethod
    def monopoly_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("monopoly_price must be >= 0")
        return v

    @model_validator(mode="after")
    def nash_le_monopoly(self) -> "CollusionPremiumResult":
        if self.nash_price > self.monopoly_price:
            raise ValueError("nash_price must be <= monopoly_price")
        return self

    @property
    def price_gap(self) -> float:
        """Monopoly price minus Nash price."""
        return self.monopoly_price - self.nash_price

    @property
    def relative_premium(self) -> Optional[float]:
        """Premium as a fraction of the Nash price (None if Nash is zero)."""
        if self.nash_price == 0:
            return None
        return self.premium / self.nash_price

    def __repr__(self) -> str:
        return (
            f"CollusionPremium(idx={self.collusion_index:.3f}, "
            f"premium={self.premium:.4f})"
        )


class DetectionResult(BaseModel):
    """Overall detection result combining all tiers."""

    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    tier_results: List[TestResult]
    collusion_premium: Optional[CollusionPremiumResult] = None
    evidence_summary: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)

    @property
    def highest_rejected_tier(self) -> Optional[TestTier]:
        """Return the highest tier whose null was rejected, if any."""
        rejected = [tr.tier for tr in self.tier_results if tr.combined_reject]
        if not rejected:
            return None
        return max(rejected, key=lambda t: t.order)

    @property
    def num_tiers_rejected(self) -> int:
        return sum(1 for tr in self.tier_results if tr.combined_reject)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionResult":
        return cls.model_validate(data)

    def __repr__(self) -> str:
        return (
            f"DetectionResult({self.verdict.value}, "
            f"conf={self.confidence:.2f}, "
            f"tiers_rejected={self.num_tiers_rejected}/{len(self.tier_results)})"
        )


class CertificateSummary(BaseModel):
    """Summary suitable for regulatory submission."""

    system_id: str
    evaluation_date: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    collusion_index: Optional[float] = None
    tier_verdicts: Dict[str, Verdict] = Field(default_factory=dict)
    evidence_chain: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    methodology_version: str = "1.0"

    @field_validator("collusion_index")
    @classmethod
    def collusion_index_bounded(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("collusion_index must be in [0, 1]")
        return v

    @classmethod
    def from_detection(
        cls,
        system_id: str,
        result: DetectionResult,
    ) -> "CertificateSummary":
        """Build a certificate directly from a :class:`DetectionResult`."""
        tier_verdicts: Dict[str, Verdict] = {}
        for tr in result.tier_results:
            tier_verdicts[tr.tier.value] = (
                Verdict.COLLUSIVE if tr.combined_reject else Verdict.COMPETITIVE
            )

        ci = (
            result.collusion_premium.collusion_index
            if result.collusion_premium
            else None
        )

        return cls(
            system_id=system_id,
            evaluation_date=datetime.now(timezone.utc).isoformat(),
            verdict=result.verdict,
            confidence=result.confidence,
            collusion_index=ci,
            tier_verdicts=tier_verdicts,
            evidence_chain=[result.evidence_summary] if result.evidence_summary else [],
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def __repr__(self) -> str:
        return (
            f"Certificate({self.system_id!r}, {self.verdict.value}, "
            f"conf={self.confidence:.2f})"
        )


class EvidenceBundle(BaseModel):
    """Bundle of evidence supporting a verdict."""

    price_trajectory: Optional[PriceTrajectory] = None
    test_results: List[TestResult] = Field(default_factory=list)
    collusion_premium: Optional[CollusionPremiumResult] = None
    visualizations: Dict[str, str] = Field(default_factory=dict)
    narrative: str = ""

    model_config = {"arbitrary_types_allowed": True}

    @property
    def num_tests(self) -> int:
        return sum(len(tr.test_results) for tr in self.test_results)

    @property
    def overall_rejection_rate(self) -> float:
        total = self.num_tests
        if total == 0:
            return 0.0
        rejected = sum(
            1
            for tr in self.test_results
            for ht in tr.test_results
            if ht.reject
        )
        return rejected / total

    def __repr__(self) -> str:
        has_traj = self.price_trajectory is not None
        return (
            f"EvidenceBundle(tiers={len(self.test_results)}, "
            f"tests={self.num_tests}, trajectory={has_traj})"
        )


# ── Scenario and Evaluation Types ────────────────────────────────────────────


class ScenarioConfig(BaseModel):
    """Configuration for an evaluation scenario."""

    scenario_id: str
    name: str
    category: ScenarioCategory
    game_config: GameConfig
    algorithm_configs: List[AlgorithmConfig]
    expected_verdict: Verdict
    description: str = ""
    difficulty: float = Field(ge=0.0, le=1.0, default=0.5)

    @model_validator(mode="after")
    def algo_count_matches_players(self) -> "ScenarioConfig":
        if len(self.algorithm_configs) != self.game_config.num_players:
            raise ValueError(
                f"Number of algorithm configs ({len(self.algorithm_configs)}) "
                f"must equal num_players ({self.game_config.num_players})"
            )
        return self

    def __repr__(self) -> str:
        return (
            f"Scenario({self.scenario_id!r}, {self.category.value}, "
            f"expect={self.expected_verdict.value})"
        )


class BenchmarkResult(BaseModel):
    """Result of running a single benchmark scenario."""

    scenario_id: str
    expected_verdict: Verdict
    actual_verdict: Verdict
    correct: bool
    detection_result: DetectionResult
    runtime_seconds: float = 0.0

    @model_validator(mode="after")
    def validate_correct_flag(self) -> "BenchmarkResult":
        should_be = self.expected_verdict == self.actual_verdict
        if self.correct != should_be:
            self.correct = should_be
        return self

    @property
    def is_type_i_error(self) -> bool:
        """False positive: expected competitive but flagged as collusive."""
        return (
            self.expected_verdict == Verdict.COMPETITIVE
            and self.actual_verdict == Verdict.COLLUSIVE
        )

    @property
    def is_type_ii_error(self) -> bool:
        """False negative: expected collusive but classified as competitive."""
        return (
            self.expected_verdict == Verdict.COLLUSIVE
            and self.actual_verdict == Verdict.COMPETITIVE
        )

    def __repr__(self) -> str:
        mark = "✓" if self.correct else "✗"
        return (
            f"BenchmarkResult({self.scenario_id!r} {mark}, "
            f"expected={self.expected_verdict.value}, "
            f"actual={self.actual_verdict.value})"
        )


class EvaluationSummary(BaseModel):
    """Summary of a full evaluation run across many scenarios."""

    mode: EvaluationMode
    total_scenarios: int
    correct: int
    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    type_i_error_rate: float = Field(ge=0.0, le=1.0)
    type_ii_error_rate: float = Field(ge=0.0, le=1.0)
    power: float = Field(ge=0.0, le=1.0)
    results: List[BenchmarkResult] = Field(default_factory=list)
    runtime_seconds: float = 0.0

    @classmethod
    def from_results(
        cls, mode: EvaluationMode, results: List[BenchmarkResult]
    ) -> "EvaluationSummary":
        """Compute all aggregate metrics from a list of benchmark results."""
        total = len(results)
        correct = sum(1 for r in results if r.correct)

        # True/false positives/negatives (positive = collusive verdict)
        tp = sum(
            1
            for r in results
            if r.expected_verdict == Verdict.COLLUSIVE
            and r.actual_verdict == Verdict.COLLUSIVE
        )
        fp = sum(
            1
            for r in results
            if r.expected_verdict == Verdict.COMPETITIVE
            and r.actual_verdict == Verdict.COLLUSIVE
        )
        fn = sum(
            1
            for r in results
            if r.expected_verdict == Verdict.COLLUSIVE
            and r.actual_verdict == Verdict.COMPETITIVE
        )

        n_competitive = sum(
            1 for r in results if r.expected_verdict == Verdict.COMPETITIVE
        )
        n_collusive = sum(
            1 for r in results if r.expected_verdict == Verdict.COLLUSIVE
        )

        accuracy = correct / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        type_i = fp / n_competitive if n_competitive else 0.0
        type_ii = fn / n_collusive if n_collusive else 0.0
        power = 1.0 - type_ii

        runtime = sum(r.runtime_seconds for r in results)

        return cls(
            mode=mode,
            total_scenarios=total,
            correct=correct,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            type_i_error_rate=type_i,
            type_ii_error_rate=type_ii,
            power=power,
            results=results,
            runtime_seconds=runtime,
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def __repr__(self) -> str:
        return (
            f"EvaluationSummary({self.mode.value}, "
            f"acc={self.accuracy:.2%}, F1={self.f1_score:.2%}, "
            f"power={self.power:.2%}, n={self.total_scenarios})"
        )

    def __str__(self) -> str:
        lines = [
            f"=== Evaluation Summary ({self.mode.value}) ===",
            f"  Scenarios : {self.total_scenarios}",
            f"  Correct   : {self.correct}",
            f"  Accuracy  : {self.accuracy:.2%}",
            f"  Precision : {self.precision:.2%}",
            f"  Recall    : {self.recall:.2%}",
            f"  F1 Score  : {self.f1_score:.2%}",
            f"  Type I    : {self.type_i_error_rate:.2%}",
            f"  Type II   : {self.type_ii_error_rate:.2%}",
            f"  Power     : {self.power:.2%}",
            f"  Runtime   : {self.runtime_seconds:.1f}s",
        ]
        return "\n".join(lines)
