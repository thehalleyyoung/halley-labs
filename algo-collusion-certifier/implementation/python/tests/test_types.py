"""Tests for domain types."""

import pytest
import numpy as np
from pydantic import ValidationError

from collusion_proof.types import (
    PlayerAction,
    MarketOutcome,
    PriceTrajectory,
    GameConfig,
    AlgorithmConfig,
    ConfidenceInterval,
    HypothesisTestResult,
    TestResult,
    CollusionPremiumResult,
    DetectionResult,
    CertificateSummary,
    EvidenceBundle,
    ScenarioConfig,
    BenchmarkResult,
    EvaluationSummary,
    Verdict,
    MarketType,
    DemandSystem,
    OracleAccessLevel,
    EvaluationMode,
    ScenarioCategory,
    TestTier,
    NullHypothesis,
)


class TestPlayerAction:
    def test_creation(self):
        action = PlayerAction(player_id=0, round_num=5, price=3.5, quantity=2.0, profit=5.0)
        assert action.player_id == 0
        assert action.round_num == 5
        assert action.price == 3.5
        assert action.quantity == 2.0
        assert action.profit == 5.0

    def test_optional_fields(self):
        action = PlayerAction(player_id=1, round_num=0)
        assert action.price is None
        assert action.quantity is None
        assert action.profit is None
        assert action.metadata == {}

    def test_validation_negative_price(self):
        with pytest.raises(ValidationError):
            PlayerAction(player_id=0, round_num=0, price=-1.0)

    def test_validation_negative_quantity(self):
        with pytest.raises(ValidationError):
            PlayerAction(player_id=0, round_num=0, quantity=-5.0)

    def test_metadata(self):
        action = PlayerAction(
            player_id=0, round_num=0, metadata={"algorithm": "q_learning"}
        )
        assert action.metadata["algorithm"] == "q_learning"

    def test_repr(self):
        action = PlayerAction(player_id=0, round_num=1, price=2.5)
        r = repr(action)
        assert "player=0" in r
        assert "round=1" in r


class TestMarketOutcome:
    def _make_outcome(self, round_num=0, prices=(3.0, 3.5)):
        actions = [
            PlayerAction(player_id=i, round_num=round_num, price=p)
            for i, p in enumerate(prices)
        ]
        return MarketOutcome(round_num=round_num, actions=actions)

    def test_prices_property(self):
        outcome = self._make_outcome(prices=(2.0, 3.0))
        assert outcome.prices == [2.0, 3.0]

    def test_num_players(self):
        outcome = self._make_outcome(prices=(1.0, 2.0, 3.0))
        assert outcome.num_players == 3

    def test_mean_price(self):
        outcome = self._make_outcome(prices=(2.0, 4.0))
        assert outcome.mean_price == 3.0

    def test_mean_price_empty(self):
        outcome = MarketOutcome(
            round_num=0,
            actions=[PlayerAction(player_id=0, round_num=0)],
        )
        assert outcome.mean_price is None

    def test_total_welfare(self):
        outcome = MarketOutcome(
            round_num=0,
            actions=[PlayerAction(player_id=0, round_num=0, price=3.0)],
            consumer_surplus=5.0,
            producer_surplus=3.0,
        )
        assert outcome.total_welfare == 8.0

    def test_total_welfare_none(self):
        outcome = self._make_outcome()
        assert outcome.total_welfare is None


class TestPriceTrajectory:
    def _make_trajectory(self, n_rounds=10, n_players=2):
        outcomes = []
        for r in range(n_rounds):
            actions = [
                PlayerAction(player_id=p, round_num=r, price=float(r + p + 1))
                for p in range(n_players)
            ]
            outcomes.append(MarketOutcome(round_num=r, actions=actions))
        return PriceTrajectory(
            outcomes=outcomes, num_players=n_players, num_rounds=n_rounds
        )

    def test_price_matrix(self):
        traj = self._make_trajectory(5, 2)
        mat = traj.get_prices_matrix()
        assert mat.shape == (5, 2)
        assert mat[0, 0] == 1.0  # round 0, player 0
        assert mat[0, 1] == 2.0  # round 0, player 1
        assert mat[4, 0] == 5.0  # round 4, player 0

    def test_profits_matrix(self):
        traj = self._make_trajectory(3, 2)
        mat = traj.get_profits_matrix()
        assert mat.shape == (3, 2)
        assert np.all(np.isnan(mat))  # no profits set

    def test_player_prices(self):
        traj = self._make_trajectory(5, 2)
        p0 = traj.get_player_prices(0)
        assert len(p0) == 5
        assert p0[0] == 1.0

    def test_round_prices(self):
        traj = self._make_trajectory(5, 3)
        rp = traj.get_round_prices(2)
        assert len(rp) == 3
        assert rp[0] == 3.0  # round 2, player 0: r + p + 1 = 2+0+1

    def test_tail(self):
        traj = self._make_trajectory(10, 2)
        tail = traj.tail(3)
        assert tail.num_rounds == 3
        assert len(tail.outcomes) == 3

    def test_length_mismatch(self):
        with pytest.raises(Exception):
            PriceTrajectory(
                outcomes=[], num_players=2, num_rounds=5
            )


class TestGameConfig:
    def test_defaults(self):
        config = GameConfig(num_players=2, num_rounds=100)
        assert config.num_actions == 10
        assert config.discount_factor == 0.95
        assert config.marginal_cost == 1.0

    def test_computed_prices_bertrand(self):
        config = GameConfig(num_players=2, num_rounds=100, market_type=MarketType.BERTRAND)
        # Bertrand Nash price = marginal_cost for symmetric
        assert config.nash_price == config.marginal_cost

    def test_computed_prices_monopoly(self):
        config = GameConfig(num_players=2, num_rounds=100)
        # monopoly_price = (a + c) / 2 = (10 + 1) / 2 = 5.5
        assert abs(config.monopoly_price - 5.5) < 1e-10

    def test_price_range(self):
        config = GameConfig(num_players=2, num_rounds=100)
        lo, hi = config.price_range
        assert lo < hi

    def test_validation_min_players(self):
        with pytest.raises(ValidationError):
            GameConfig(num_players=1, num_rounds=100)

    def test_validation_min_rounds(self):
        with pytest.raises(ValidationError):
            GameConfig(num_players=2, num_rounds=5)

    def test_cournot_nash(self):
        config = GameConfig(
            num_players=3, num_rounds=100, market_type=MarketType.COURNOT
        )
        assert config.nash_price is not None
        assert config.nash_price > config.marginal_cost


class TestConfidenceInterval:
    def test_width(self):
        ci = ConfidenceInterval(lower=1.0, upper=3.0)
        assert ci.width == 2.0

    def test_midpoint(self):
        ci = ConfidenceInterval(lower=1.0, upper=3.0)
        assert ci.midpoint == 2.0

    def test_contains(self):
        ci = ConfidenceInterval(lower=1.0, upper=3.0)
        assert ci.contains(2.0)
        assert ci.contains(1.0)
        assert ci.contains(3.0)
        assert not ci.contains(0.5)
        assert not ci.contains(3.5)

    def test_lower_must_be_le_upper(self):
        with pytest.raises(ValidationError):
            ConfidenceInterval(lower=5.0, upper=2.0)

    def test_point_estimate(self):
        ci = ConfidenceInterval(lower=1.0, upper=3.0, point_estimate=2.0)
        assert ci.point_estimate == 2.0

    def test_level(self):
        ci = ConfidenceInterval(lower=1.0, upper=3.0, level=0.99)
        assert ci.level == 0.99


class TestHypothesisTestResult:
    def test_creation(self):
        result = HypothesisTestResult(
            test_name="price_test",
            null_hypothesis="H0: price <= nash",
            test_statistic=2.5,
            p_value=0.01,
            reject=True,
        )
        assert result.test_name == "price_test"
        assert result.p_value == 0.01
        assert result.reject is True

    def test_is_significant(self):
        result = HypothesisTestResult(
            test_name="test",
            null_hypothesis="H0",
            test_statistic=1.0,
            p_value=0.03,
            reject=True,
            alpha=0.05,
        )
        assert result.is_significant is True

    def test_not_significant(self):
        result = HypothesisTestResult(
            test_name="test",
            null_hypothesis="H0",
            test_statistic=0.5,
            p_value=0.5,
            reject=False,
        )
        assert result.is_significant is False

    def test_power_validation(self):
        with pytest.raises(ValidationError):
            HypothesisTestResult(
                test_name="test",
                null_hypothesis="H0",
                test_statistic=1.0,
                p_value=0.5,
                reject=False,
                power=1.5,
            )


class TestVerdict:
    def test_enum_values(self):
        assert Verdict.COMPETITIVE.value == "competitive"
        assert Verdict.SUSPICIOUS.value == "suspicious"
        assert Verdict.COLLUSIVE.value == "collusive"
        assert Verdict.INCONCLUSIVE.value == "inconclusive"

    def test_is_harmful(self):
        assert Verdict.COLLUSIVE.is_harmful is True
        assert Verdict.SUSPICIOUS.is_harmful is True
        assert Verdict.COMPETITIVE.is_harmful is False
        assert Verdict.INCONCLUSIVE.is_harmful is False


class TestTestTier:
    def test_order(self):
        assert TestTier.TIER1_PRICE_LEVEL.order == 1
        assert TestTier.TIER2_CORRELATION.order == 2
        assert TestTier.TIER3_PUNISHMENT.order == 3
        assert TestTier.TIER4_COUNTERFACTUAL.order == 4


class TestScenarioConfig:
    def test_creation(self):
        game = GameConfig(num_players=2, num_rounds=100)
        algos = [AlgorithmConfig(name="a1"), AlgorithmConfig(name="a2")]
        sc = ScenarioConfig(
            scenario_id="test_01",
            name="Test Scenario",
            category=ScenarioCategory.COLLUSIVE,
            game_config=game,
            algorithm_configs=algos,
            expected_verdict=Verdict.COLLUSIVE,
            difficulty=0.7,
        )
        assert sc.scenario_id == "test_01"
        assert sc.category == ScenarioCategory.COLLUSIVE
        assert sc.difficulty == 0.7


class TestDetectionResult:
    def test_highest_rejected_tier(self):
        tier1 = TestResult(
            tier=TestTier.TIER1_PRICE_LEVEL,
            null_hypothesis=NullHypothesis.NO_SUPRACOMPETITIVE,
            test_results=[],
            combined_reject=True,
        )
        tier2 = TestResult(
            tier=TestTier.TIER2_CORRELATION,
            null_hypothesis=NullHypothesis.INDEPENDENT_PLAY,
            test_results=[],
            combined_reject=True,
        )
        tier3 = TestResult(
            tier=TestTier.TIER3_PUNISHMENT,
            null_hypothesis=NullHypothesis.NO_PUNISHMENT,
            test_results=[],
            combined_reject=False,
        )
        dr = DetectionResult(
            verdict=Verdict.SUSPICIOUS,
            confidence=0.7,
            tier_results=[tier1, tier2, tier3],
        )
        assert dr.highest_rejected_tier == TestTier.TIER2_CORRELATION
        assert dr.num_tiers_rejected == 2

    def test_no_rejections(self):
        tier1 = TestResult(
            tier=TestTier.TIER1_PRICE_LEVEL,
            null_hypothesis=NullHypothesis.NO_SUPRACOMPETITIVE,
            test_results=[],
            combined_reject=False,
        )
        dr = DetectionResult(
            verdict=Verdict.COMPETITIVE,
            confidence=0.9,
            tier_results=[tier1],
        )
        assert dr.highest_rejected_tier is None
        assert dr.num_tiers_rejected == 0


class TestCertificateSummary:
    def test_from_detection(self):
        tier1 = TestResult(
            tier=TestTier.TIER1_PRICE_LEVEL,
            null_hypothesis=NullHypothesis.NO_SUPRACOMPETITIVE,
            test_results=[],
            combined_reject=True,
        )
        dr = DetectionResult(
            verdict=Verdict.COLLUSIVE,
            confidence=0.95,
            tier_results=[tier1],
            evidence_summary="Strong evidence of collusion.",
        )
        cert = CertificateSummary.from_detection("SYS-001", dr)
        assert cert.system_id == "SYS-001"
        assert cert.verdict == Verdict.COLLUSIVE
        assert cert.confidence == 0.95
        assert "Strong evidence" in cert.evidence_chain[0]


class TestEvidenceBundle:
    def test_num_tests(self):
        ht = HypothesisTestResult(
            test_name="t1", null_hypothesis="H0",
            test_statistic=1.0, p_value=0.01, reject=True,
        )
        tr = TestResult(
            tier=TestTier.TIER1_PRICE_LEVEL,
            null_hypothesis=NullHypothesis.NO_SUPRACOMPETITIVE,
            test_results=[ht, ht],
        )
        bundle = EvidenceBundle(test_results=[tr])
        assert bundle.num_tests == 2

    def test_rejection_rate(self):
        ht_reject = HypothesisTestResult(
            test_name="t1", null_hypothesis="H0",
            test_statistic=3.0, p_value=0.001, reject=True,
        )
        ht_accept = HypothesisTestResult(
            test_name="t2", null_hypothesis="H0",
            test_statistic=0.5, p_value=0.5, reject=False,
        )
        tr = TestResult(
            tier=TestTier.TIER1_PRICE_LEVEL,
            null_hypothesis=NullHypothesis.NO_SUPRACOMPETITIVE,
            test_results=[ht_reject, ht_accept],
        )
        bundle = EvidenceBundle(test_results=[tr])
        assert bundle.overall_rejection_rate == 0.5
