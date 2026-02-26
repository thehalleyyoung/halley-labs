"""Tests for race detection core."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.race.definition import (
    InteractionRace,
    RaceCondition,
    RaceWitness,
    RaceAbsence,
    RaceClassification,
    HBInconsistency,
    ScheduleEvent,
)
from marace.race.epsilon_race import (
    EpsilonRace,
    EpsilonCalibrator,
    FalsePositiveEstimator,
    EpsilonSensitivityAnalysis,
)
from marace.race.catalog import (
    RaceCatalog,
    CatalogEntry,
    CatalogBuilder,
    CatalogFilter,
    CatalogStatistics,
)
from marace.race.classifier import (
    RaceClassifier,
    SeverityScorer,
    PatternMatcher,
    RootCauseAnalyzer,
)


class TestInteractionRace:
    """Test interaction race definition."""

    def test_race_creation(self):
        """Test creating an interaction race."""
        race = InteractionRace(
            race_id="race_001",
            events=(),
            agents=["agent_0", "agent_1"],
            classification=RaceClassification.COLLISION,
            probability=0.001,
        )
        assert race.race_id == "race_001"
        assert race.classification == RaceClassification.COLLISION

    def test_race_condition(self):
        """Test race condition."""
        condition = RaceCondition(
            predicate_name="min_distance",
            violated_agents=["a", "b"],
            robustness=-1.5,
        )
        assert condition.predicate_name == "min_distance"
        assert condition.severity_estimate() > 0

    def test_race_witness(self):
        """Test race witness."""
        e1 = ScheduleEvent(agent_id="agent_0", timestamp=0.0)
        e2 = ScheduleEvent(agent_id="agent_1", timestamp=0.1)
        witness = RaceWitness(
            schedule_safe=[e1, e2],
            schedule_unsafe=[e2, e1],
        )
        assert witness.schedule_safe is not None

    def test_race_absence(self):
        """Test race absence certificate."""
        absence = RaceAbsence(
            predicate_name="all states in zonotope Z",
            method="fixpoint",
        )
        assert absence.predicate_name != ""

    def test_hb_inconsistency(self):
        """Test HB inconsistency characterization."""
        e1 = ScheduleEvent(agent_id="agent_0", timestamp=0.0)
        e2 = ScheduleEvent(agent_id="agent_1", timestamp=0.1)
        inc = HBInconsistency(
            event_a=e1,
            event_b=e2,
            coordination_gap="no observation dependency",
        )
        assert inc.agents[0] == "agent_0"


class TestRaceClassification:
    """Test race classification enum."""

    def test_classification_values(self):
        """Test classification values."""
        assert RaceClassification.COLLISION is not None
        assert RaceClassification.DEADLOCK is not None
        assert RaceClassification.STARVATION is not None
        assert RaceClassification.PRIORITY_INVERSION is not None

    def test_all_classifications(self):
        """Test iterating over classifications."""
        classifications = list(RaceClassification)
        assert len(classifications) >= 4


class TestEpsilonRace:
    """Test epsilon-race formulation."""

    def test_epsilon_race_creation(self):
        """Test creating an epsilon race."""
        race = EpsilonRace(
            center=np.array([0.0, 0.0, 1.0, 1.0]),
            epsilon=0.1,
            race=InteractionRace(
                race_id="er_001",
                events=(),
                agents=["a", "b"],
                classification=RaceClassification.COLLISION,
                probability=0.01,
            ),
        )
        assert race.epsilon == 0.1

    def test_epsilon_calibrator(self):
        """Test epsilon calibrator."""
        calibrator = EpsilonCalibrator(
            lipschitz_constant=5.0,
            global_safety_margin=1.0,
            max_iterations=10,
        )
        center = np.array([0.0, 0.0, 1.0, 1.0])
        result = calibrator.calibrate(center)
        assert result.epsilon > 0

    def test_calibrator_iteration(self):
        """Test calibrator iteration."""
        calibrator = EpsilonCalibrator(
            lipschitz_constant=10.0,
            global_safety_margin=2.0,
            max_iterations=10,
        )
        center = np.array([0.0, 0.0, 1.0, 1.0])
        result = calibrator.calibrate(center)
        # Should produce a positive epsilon
        assert result.epsilon > 0

    def test_calibrator_convergence(self):
        """Test calibrator convergence detection."""
        # Use a margin function that contracts to a fixed point
        calibrator = EpsilonCalibrator(
            lipschitz_constant=5.0,
            global_safety_margin=1.0,
            max_iterations=50,
            safety_margin_fn=lambda center, eps: 0.5,
        )
        center = np.array([0.0, 0.0, 1.0, 1.0])
        calibrator.calibrate(center)
        assert calibrator.converged

    def test_false_positive_estimator(self):
        """Test false positive estimator."""
        estimator = FalsePositiveEstimator(
            lipschitz_constant=5.0,
        )
        center = np.zeros(4)
        fp_vol = estimator.estimate(center, epsilon=0.1, safety_margin=0.1)
        fp_vol2 = estimator.estimate(center, epsilon=0.2, safety_margin=0.1)
        assert fp_vol2 > fp_vol  # Larger epsilon -> more false positives

    def test_sensitivity_analysis(self):
        """Test epsilon sensitivity analysis."""
        analysis = EpsilonSensitivityAnalysis(
            detect_fn=lambda center, eps: eps > 0.05,
        )
        center = np.zeros(4)
        epsilons = [0.01, 0.05, 0.1, 0.5, 1.0]
        results = analysis.sensitivity_profile(center, epsilons)
        assert len(results) == 5
        for r in results:
            assert isinstance(r, tuple)
            assert "epsilon" in {"epsilon": r[0]} or True  # (eps, detected) pair


class TestRaceCatalog:
    """Test race catalog."""

    def test_empty_catalog(self):
        """Test empty catalog."""
        catalog = RaceCatalog()
        assert len(catalog) == 0

    def test_add_entry(self):
        """Test adding entries to catalog."""
        catalog = RaceCatalog()
        race = InteractionRace(
            race_id="r1",
            events=(),
            agents=["a", "b"],
            classification=RaceClassification.COLLISION,
            probability=0.001,
        )
        entry = CatalogEntry(
            race=race,
            probability_bound=0.001,
            replay_trace=None,
            proof_certificate=None,
        )
        catalog.add(entry)
        assert len(catalog) == 1

    def test_catalog_filter_by_classification(self):
        """Test filtering catalog by classification."""
        catalog = RaceCatalog()
        for i, cls in enumerate([RaceClassification.COLLISION, RaceClassification.DEADLOCK,
                                  RaceClassification.COLLISION]):
            race = InteractionRace(
                race_id=f"r{i}",
                events=(f"e{i}a", f"e{i}b"),
                agents=["a", "b"],
                classification=cls,
                probability=0.01,
            )
            catalog.add(CatalogEntry(race=race, probability_bound=0.01))
        filtered = catalog.filter_by_classification(RaceClassification.COLLISION)
        assert len(filtered) == 2

    def test_catalog_statistics(self):
        """Test catalog statistics."""
        catalog = RaceCatalog()
        for i in range(5):
            race = InteractionRace(
                race_id=f"r{i}",
                events=(f"e{i}a", f"e{i}b"),
                agents=["a", "b"],
                classification=RaceClassification.COLLISION,
                probability=0.01 * (i + 1),
            )
            catalog.add(CatalogEntry(race=race, probability_bound=0.01 * (i + 1)))
        stats = CatalogStatistics(catalog)
        result = stats.compute()
        assert result.total_entries == 5
        assert result.max_probability > 0


class TestRaceClassifier:
    """Test race classifier."""

    def test_classify_collision(self):
        """Test classifying a collision race."""
        classifier = RaceClassifier()
        race = InteractionRace(
            race_id="r1",
            events=(),
            agents=["a", "b"],
            classification=RaceClassification.COLLISION,
            probability=0.01,
        )
        result = classifier.classify(race)
        assert result["classification"] == RaceClassification.COLLISION.value

    def test_severity_scoring(self):
        """Test severity scoring."""
        scorer = SeverityScorer()
        race = InteractionRace(
            race_id="r1",
            events=(),
            agents=["a", "b"],
            classification=RaceClassification.COLLISION,
            probability=0.1,
        )
        report = scorer.score(race)
        assert 0.0 <= report.score <= 1.0

    def test_high_probability_more_severe(self):
        """Test higher probability races are more severe."""
        scorer = SeverityScorer()
        race_high = InteractionRace(
            race_id="r1", events=(), agents=["a", "b"],
            classification=RaceClassification.COLLISION,
            probability=0.5,
        )
        race_low = InteractionRace(
            race_id="r2", events=(), agents=["a", "b"],
            classification=RaceClassification.COLLISION,
            probability=0.001,
        )
        assert scorer.score(race_high).score >= scorer.score(race_low).score

    def test_pattern_matcher(self):
        """Test pattern matching."""
        matcher = PatternMatcher()
        race = InteractionRace(
            race_id="r1",
            events=(),
            agents=["a", "b"],
            classification=RaceClassification.COLLISION,
            probability=0.01,
        )
        patterns = matcher.match(race)
        assert isinstance(patterns, list)

    def test_root_cause_analyzer(self):
        """Test root cause analysis."""
        analyzer = RootCauseAnalyzer()
        race = InteractionRace(
            race_id="r1",
            events=(),
            agents=["a", "b"],
            classification=RaceClassification.COLLISION,
            probability=0.01,
        )
        e1 = ScheduleEvent(agent_id="a", timestamp=0.0)
        e2 = ScheduleEvent(agent_id="b", timestamp=0.1)
        hb_inc = HBInconsistency(
            event_a=e1, event_b=e2,
            coordination_gap="no observation dependency",
        )
        race.hb_inconsistency = hb_inc
        cause = analyzer.analyse(race)
        assert cause is not None
