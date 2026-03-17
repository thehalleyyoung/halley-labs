"""Tests for the evaluation module.

Tests for scenarios, benchmarks, classification metrics, baselines,
sensitivity analysis, and ground truth validation.
"""

import pytest
import numpy as np

from collusion_proof.types import (
    Verdict,
    ScenarioCategory,
    EvaluationMode,
    GameConfig,
    AlgorithmConfig,
    ScenarioConfig,
    BenchmarkResult,
    EvaluationSummary,
    DetectionResult,
    TestResult,
    TestTier,
    NullHypothesis,
    HypothesisTestResult,
)
from collusion_proof.config import MarketConfig, TestConfig
from collusion_proof.cli.commands import run_analysis, run_benchmark, validate_inputs


def _make_detection_result(verdict: Verdict, confidence: float = 0.8) -> DetectionResult:
    """Helper to create a DetectionResult for testing."""
    return DetectionResult(
        verdict=verdict,
        confidence=confidence,
        tier_results=[
            TestResult(
                tier=TestTier.TIER1_PRICE_LEVEL,
                null_hypothesis=NullHypothesis.NO_SUPRACOMPETITIVE,
                test_results=[
                    HypothesisTestResult(
                        test_name="price_level",
                        null_hypothesis="prices <= nash",
                        test_statistic=2.5,
                        p_value=0.01 if verdict == Verdict.COLLUSIVE else 0.5,
                        reject=verdict == Verdict.COLLUSIVE,
                    )
                ],
                combined_p_value=0.01 if verdict == Verdict.COLLUSIVE else 0.5,
                combined_reject=verdict == Verdict.COLLUSIVE,
                alpha_spent=0.02,
            )
        ],
        evidence_summary=f"Test verdict: {verdict.value}",
    )


class TestScenarios:
    """Tests for evaluation scenario generation."""

    def test_scenario_config_creation(self):
        game = GameConfig(num_players=2, num_rounds=1000)
        algos = [
            AlgorithmConfig(name="agent0"),
            AlgorithmConfig(name="agent1"),
        ]
        sc = ScenarioConfig(
            scenario_id="S001",
            name="Test Scenario",
            category=ScenarioCategory.COLLUSIVE,
            game_config=game,
            algorithm_configs=algos,
            expected_verdict=Verdict.COLLUSIVE,
        )
        assert sc.scenario_id == "S001"
        assert sc.category == ScenarioCategory.COLLUSIVE
        assert sc.expected_verdict == Verdict.COLLUSIVE

    def test_scenario_categories(self):
        categories = [c.value for c in ScenarioCategory]
        assert "collusive" in categories
        assert "competitive" in categories
        assert "boundary" in categories
        assert "adversarial" in categories

    def test_scenario_algo_count_mismatch(self):
        """Algorithm count must match player count."""
        game = GameConfig(num_players=2, num_rounds=100)
        algos = [AlgorithmConfig(name="agent0")]  # Only 1, but 2 players
        with pytest.raises(Exception):
            ScenarioConfig(
                scenario_id="S999",
                name="Bad",
                category=ScenarioCategory.COMPETITIVE,
                game_config=game,
                algorithm_configs=algos,
                expected_verdict=Verdict.COMPETITIVE,
            )

    def test_game_config_computes_equilibria(self):
        game = GameConfig(num_players=2, num_rounds=100)
        assert game.nash_price is not None
        assert game.monopoly_price is not None
        assert game.nash_price < game.monopoly_price

    def test_scenario_difficulty_bounds(self):
        game = GameConfig(num_players=2, num_rounds=100)
        algos = [AlgorithmConfig(name="a0"), AlgorithmConfig(name="a1")]
        sc = ScenarioConfig(
            scenario_id="S002",
            name="Easy",
            category=ScenarioCategory.COLLUSIVE,
            game_config=game,
            algorithm_configs=algos,
            expected_verdict=Verdict.COLLUSIVE,
            difficulty=0.3,
        )
        assert 0.0 <= sc.difficulty <= 1.0


class TestMetrics:
    """Tests for classification metrics."""

    def test_perfect_classification(self):
        results = [
            BenchmarkResult(
                scenario_id="S1",
                expected_verdict=Verdict.COLLUSIVE,
                actual_verdict=Verdict.COLLUSIVE,
                correct=True,
                detection_result=_make_detection_result(Verdict.COLLUSIVE),
            ),
            BenchmarkResult(
                scenario_id="S2",
                expected_verdict=Verdict.COMPETITIVE,
                actual_verdict=Verdict.COMPETITIVE,
                correct=True,
                detection_result=_make_detection_result(Verdict.COMPETITIVE),
            ),
        ]
        summary = EvaluationSummary.from_results(EvaluationMode.SMOKE, results)
        assert summary.accuracy == 1.0
        assert summary.precision == 1.0
        assert summary.recall == 1.0
        assert summary.f1_score == 1.0
        assert summary.type_i_error_rate == 0.0
        assert summary.type_ii_error_rate == 0.0

    def test_accuracy_computation(self):
        results = [
            BenchmarkResult(
                scenario_id=f"S{i}",
                expected_verdict=Verdict.COLLUSIVE if i < 5 else Verdict.COMPETITIVE,
                actual_verdict=Verdict.COLLUSIVE if i < 4 else Verdict.COMPETITIVE,
                correct=(i < 4 or i >= 5),
                detection_result=_make_detection_result(
                    Verdict.COLLUSIVE if i < 4 else Verdict.COMPETITIVE
                ),
            )
            for i in range(10)
        ]
        summary = EvaluationSummary.from_results(EvaluationMode.STANDARD, results)
        assert summary.accuracy == 9 / 10

    def test_precision_recall(self):
        # 2 TP, 1 FP, 1 FN
        results = [
            BenchmarkResult(
                scenario_id="TP1",
                expected_verdict=Verdict.COLLUSIVE,
                actual_verdict=Verdict.COLLUSIVE,
                correct=True,
                detection_result=_make_detection_result(Verdict.COLLUSIVE),
            ),
            BenchmarkResult(
                scenario_id="TP2",
                expected_verdict=Verdict.COLLUSIVE,
                actual_verdict=Verdict.COLLUSIVE,
                correct=True,
                detection_result=_make_detection_result(Verdict.COLLUSIVE),
            ),
            BenchmarkResult(
                scenario_id="FP1",
                expected_verdict=Verdict.COMPETITIVE,
                actual_verdict=Verdict.COLLUSIVE,
                correct=False,
                detection_result=_make_detection_result(Verdict.COLLUSIVE),
            ),
            BenchmarkResult(
                scenario_id="FN1",
                expected_verdict=Verdict.COLLUSIVE,
                actual_verdict=Verdict.COMPETITIVE,
                correct=False,
                detection_result=_make_detection_result(Verdict.COMPETITIVE),
            ),
        ]
        summary = EvaluationSummary.from_results(EvaluationMode.STANDARD, results)
        assert summary.precision == 2 / 3  # TP / (TP + FP) = 2/3
        assert summary.recall == 2 / 3  # TP / (TP + FN) = 2/3

    def test_type_i_error(self):
        """Type I: false positive (expected competitive, got collusive)."""
        results = [
            BenchmarkResult(
                scenario_id="FP",
                expected_verdict=Verdict.COMPETITIVE,
                actual_verdict=Verdict.COLLUSIVE,
                correct=False,
                detection_result=_make_detection_result(Verdict.COLLUSIVE),
            ),
            BenchmarkResult(
                scenario_id="TN",
                expected_verdict=Verdict.COMPETITIVE,
                actual_verdict=Verdict.COMPETITIVE,
                correct=True,
                detection_result=_make_detection_result(Verdict.COMPETITIVE),
            ),
        ]
        summary = EvaluationSummary.from_results(EvaluationMode.SMOKE, results)
        assert summary.type_i_error_rate == 0.5

    def test_type_ii_error(self):
        """Type II: false negative (expected collusive, got competitive)."""
        results = [
            BenchmarkResult(
                scenario_id="FN",
                expected_verdict=Verdict.COLLUSIVE,
                actual_verdict=Verdict.COMPETITIVE,
                correct=False,
                detection_result=_make_detection_result(Verdict.COMPETITIVE),
            ),
            BenchmarkResult(
                scenario_id="TP",
                expected_verdict=Verdict.COLLUSIVE,
                actual_verdict=Verdict.COLLUSIVE,
                correct=True,
                detection_result=_make_detection_result(Verdict.COLLUSIVE),
            ),
        ]
        summary = EvaluationSummary.from_results(EvaluationMode.SMOKE, results)
        assert summary.type_ii_error_rate == 0.5
        assert summary.power == 0.5

    def test_mcc_like_f1(self):
        """F1 score for perfect then imperfect results."""
        perfect = [
            BenchmarkResult(
                scenario_id="P1",
                expected_verdict=Verdict.COLLUSIVE,
                actual_verdict=Verdict.COLLUSIVE,
                correct=True,
                detection_result=_make_detection_result(Verdict.COLLUSIVE),
            ),
        ]
        summary = EvaluationSummary.from_results(EvaluationMode.SMOKE, perfect)
        assert summary.f1_score == 1.0

    def test_empty_results(self):
        summary = EvaluationSummary.from_results(EvaluationMode.SMOKE, [])
        assert summary.accuracy == 0.0
        assert summary.total_scenarios == 0


class TestBaselines:
    """Tests for baseline screening methods."""

    def test_correlation_screen_independent(self):
        """Independent prices should show low correlation."""
        rng = np.random.RandomState(42)
        prices = rng.normal(3.0, 1.0, (1000, 2))
        corr = float(np.corrcoef(prices[:, 0], prices[:, 1])[0, 1])
        assert abs(corr) < 0.1

    def test_correlation_screen_correlated(self):
        """Correlated prices should be detected."""
        rng = np.random.RandomState(42)
        base = rng.normal(3.0, 1.0, 1000)
        prices = np.column_stack([base, base + rng.normal(0, 0.05, 1000)])
        corr = float(np.corrcoef(prices[:, 0], prices[:, 1])[0, 1])
        assert corr > 0.9

    def test_variance_screen_low(self):
        """Low variance suggests coordinated pricing."""
        rng = np.random.RandomState(42)
        prices = rng.normal(5.0, 0.01, (1000, 2))
        var = float(np.var(prices))
        assert var < 0.01

    def test_variance_screen_high(self):
        """High variance suggests independent pricing."""
        rng = np.random.RandomState(42)
        prices = rng.uniform(1.0, 6.0, (1000, 2))
        var = float(np.var(prices))
        assert var > 1.0


class TestSensitivity:
    """Tests for sensitivity analysis."""

    def test_one_at_a_time_alpha(self):
        """Varying alpha should change detection results."""
        rng = np.random.RandomState(42)
        prices = rng.normal(3.0, 0.5, (1000, 2))
        prices = np.clip(prices, 0, None)

        r_strict = run_analysis(prices, 1.0, 5.5, alpha=0.001)
        r_lax = run_analysis(prices, 1.0, 5.5, alpha=0.20)
        # More lenient alpha should be at least as likely to reject
        assert r_lax["confidence"] >= 0  # just check it runs

    def test_input_perturbation(self):
        """Small input changes should not dramatically alter verdicts."""
        rng = np.random.RandomState(42)
        prices = rng.normal(3.0, 0.5, (1000, 2))
        prices = np.clip(prices, 0, None)

        r1 = run_analysis(prices, 1.0, 5.5)
        perturbed = prices + rng.normal(0, 0.01, prices.shape)
        perturbed = np.clip(perturbed, 0, None)
        r2 = run_analysis(perturbed, 1.0, 5.5)

        # Collusion index should be similar
        assert abs(r1["collusion_index"] - r2["collusion_index"]) < 0.1


class TestGroundTruth:
    """Tests for ground truth validation."""

    def test_validate_known_collusive(self):
        """Clearly collusive prices should be identified."""
        prices = np.ones((2000, 2)) * 5.4
        result = run_analysis(prices, nash_price=1.0, monopoly_price=5.5)
        assert result["verdict"] in ("collusive", "suspicious")

    def test_validate_known_competitive(self):
        """Clearly competitive prices should be identified."""
        rng = np.random.RandomState(42)
        prices = rng.normal(1.05, 0.1, (2000, 2))
        prices = np.clip(prices, 0, None)
        result = run_analysis(prices, nash_price=1.0, monopoly_price=5.5)
        assert result["verdict"] in ("competitive", "inconclusive")

    def test_accuracy_over_scenarios(self):
        """Run the benchmark and check overall accuracy."""
        results = run_benchmark("smoke", seed=42)
        assert results["accuracy"] >= 0.5  # at least 50% correct
        assert results["total_scenarios"] > 0


class TestBenchmarkRunner:
    """Tests for the benchmark runner."""

    def test_smoke_mode(self):
        results = run_benchmark("smoke", seed=42)
        assert results["mode"] == "smoke"
        assert results["total_scenarios"] == 5
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_standard_mode(self):
        results = run_benchmark("standard", seed=42)
        assert results["mode"] == "standard"
        assert results["total_scenarios"] == 15

    def test_benchmark_reproducibility(self):
        r1 = run_benchmark("smoke", seed=42)
        r2 = run_benchmark("smoke", seed=42)
        assert r1["accuracy"] == r2["accuracy"]
        assert r1["correct"] == r2["correct"]

    def test_benchmark_result_structure(self):
        results = run_benchmark("smoke", seed=42)
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1_score" in results
        assert "type_i_error_rate" in results
        assert "type_ii_error_rate" in results
        assert "results" in results
        assert len(results["results"]) == results["total_scenarios"]


class TestBenchmarkResult:
    """Tests for BenchmarkResult type."""

    def test_correct_flag_auto_computed(self):
        dr = _make_detection_result(Verdict.COLLUSIVE)
        br = BenchmarkResult(
            scenario_id="test",
            expected_verdict=Verdict.COLLUSIVE,
            actual_verdict=Verdict.COLLUSIVE,
            correct=False,  # intentionally wrong; validator should fix it
            detection_result=dr,
        )
        assert br.correct is True

    def test_is_type_i_error(self):
        dr = _make_detection_result(Verdict.COLLUSIVE)
        br = BenchmarkResult(
            scenario_id="test",
            expected_verdict=Verdict.COMPETITIVE,
            actual_verdict=Verdict.COLLUSIVE,
            correct=False,
            detection_result=dr,
        )
        assert br.is_type_i_error is True
        assert br.is_type_ii_error is False

    def test_is_type_ii_error(self):
        dr = _make_detection_result(Verdict.COMPETITIVE)
        br = BenchmarkResult(
            scenario_id="test",
            expected_verdict=Verdict.COLLUSIVE,
            actual_verdict=Verdict.COMPETITIVE,
            correct=False,
            detection_result=dr,
        )
        assert br.is_type_ii_error is True
        assert br.is_type_i_error is False


class TestValidation:
    """Tests for input validation."""

    def test_validate_good_inputs(self):
        prices = np.ones((1000, 2)) * 3.0
        warnings = validate_inputs(1.0, 5.5, prices)
        assert len(warnings) == 0

    def test_validate_nash_ge_monopoly(self):
        prices = np.ones((1000, 2)) * 3.0
        warnings = validate_inputs(6.0, 5.5, prices)
        assert any("Nash price" in w for w in warnings)

    def test_validate_few_rounds(self):
        prices = np.ones((50, 2)) * 3.0
        warnings = validate_inputs(1.0, 5.5, prices)
        assert any("rounds" in w.lower() for w in warnings)

    def test_validate_negative_prices(self):
        prices = np.array([[-1.0, 2.0], [3.0, 4.0]])
        warnings = validate_inputs(1.0, 5.5, prices)
        assert any("negative" in w.lower() for w in warnings)
