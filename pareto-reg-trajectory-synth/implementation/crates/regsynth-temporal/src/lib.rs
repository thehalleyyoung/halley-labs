//! Temporal model for the RegSynth regulatory compliance engine.
//!
//! Models how regulations evolve over time with phase-in dates,
//! amendments, and multi-jurisdictional timelines.

pub mod bisimulation;
pub mod conflict_detector;
pub mod constraint_evolution;
pub mod incremental;
pub mod phase_in;
pub mod temporal_diff;
pub mod timeline;
pub mod transition_system;
pub mod version_lattice;

pub use bisimulation::{BisimulationRelation, compute_bisimulation, quotient_system, refine_partition};
pub use conflict_detector::{
    ConflictCertificate, ConflictResolution, Relaxation, TcgEdgeKind, TcgNode,
    TemporalConflictDetector, TemporalConstraintGraph,
};
pub use constraint_evolution::ConstraintEvolution;
pub use incremental::IncrementalUpdate;
pub use phase_in::{eu_ai_act_schedule, GracePeriod, Milestone, PhaseInOperator, PhaseInSchedule, SunsetOperator};
pub use temporal_diff::TemporalDiff;
pub use timeline::{Timeline, TimelineEvent};
pub use transition_system::{RegulatoryTransitionSystem, Transition, TransitionLabel};
pub use version_lattice::{VersionLattice, VersionNode, VersionOrdering};

use chrono::NaiveDate;
use regsynth_types::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

// ---------------------------------------------------------------------------
// Shared types used across every sub-module
// ---------------------------------------------------------------------------

/// Unique identifier for an obligation within the temporal system.
pub type ObligationId = String;

/// Unique identifier for a state in the transition system.
pub type StateId = String;

/// A concrete regulatory obligation with temporal metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Obligation {
    pub id: ObligationId,
    pub kind: ObligationKind,
    pub jurisdiction: Jurisdiction,
    pub interval: TemporalInterval,
    pub description: String,
    pub article_ref: Option<ArticleRef>,
    pub risk_level: Option<RiskLevel>,
    pub grade: FormalizabilityGrade,
}

impl Obligation {
    pub fn new(
        id: impl Into<String>,
        kind: ObligationKind,
        jurisdiction: Jurisdiction,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            jurisdiction,
            interval: TemporalInterval::unbounded(),
            description: description.into(),
            article_ref: None,
            risk_level: None,
            grade: FormalizabilityGrade::F1,
        }
    }

    pub fn with_interval(mut self, interval: TemporalInterval) -> Self {
        self.interval = interval;
        self
    }

    pub fn with_article_ref(mut self, article_ref: ArticleRef) -> Self {
        self.article_ref = Some(article_ref);
        self
    }

    pub fn with_risk_level(mut self, risk_level: RiskLevel) -> Self {
        self.risk_level = Some(risk_level);
        self
    }

    pub fn with_grade(mut self, grade: FormalizabilityGrade) -> Self {
        self.grade = grade;
        self
    }

    /// Check if this obligation is active at the given date.
    pub fn is_active_at(&self, date: &NaiveDate) -> bool {
        date_in_interval(date, &self.interval)
    }
}

impl fmt::Display for Obligation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} {} ({})",
            self.id, self.kind, self.description, self.jurisdiction
        )
    }
}

/// A regulatory state: a set of active obligations at a point in time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegulatoryState {
    pub id: StateId,
    pub obligations: BTreeSet<ObligationId>,
    pub timestamp: Option<NaiveDate>,
}

impl RegulatoryState {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            obligations: BTreeSet::new(),
            timestamp: None,
        }
    }

    pub fn with_obligations(id: impl Into<String>, obligations: BTreeSet<ObligationId>) -> Self {
        Self {
            id: id.into(),
            obligations,
            timestamp: None,
        }
    }

    pub fn with_timestamp(mut self, timestamp: NaiveDate) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    pub fn add_obligation(&mut self, id: ObligationId) {
        self.obligations.insert(id);
    }

    pub fn remove_obligation(&mut self, id: &str) -> bool {
        self.obligations.remove(id)
    }

    /// Two states are obligation-equivalent if they share the same active set.
    pub fn obligation_equivalent(&self, other: &RegulatoryState) -> bool {
        self.obligations == other.obligations
    }

    pub fn obligation_count(&self) -> usize {
        self.obligations.len()
    }
}

impl fmt::Display for RegulatoryState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "State({}, {} obligations)",
            self.id,
            self.obligations.len()
        )
    }
}

/// Events that cause regulatory state transitions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegulatoryEvent {
    Amendment {
        description: String,
        date: NaiveDate,
    },
    PhaseIn {
        milestone: String,
        date: NaiveDate,
    },
    Sunset {
        description: String,
        date: NaiveDate,
    },
    Repeal {
        description: String,
        date: NaiveDate,
    },
    GracePeriodStart {
        description: String,
        date: NaiveDate,
    },
    GracePeriodEnd {
        description: String,
        date: NaiveDate,
    },
}

impl RegulatoryEvent {
    pub fn date(&self) -> NaiveDate {
        match self {
            Self::Amendment { date, .. }
            | Self::PhaseIn { date, .. }
            | Self::Sunset { date, .. }
            | Self::Repeal { date, .. }
            | Self::GracePeriodStart { date, .. }
            | Self::GracePeriodEnd { date, .. } => *date,
        }
    }

    pub fn description(&self) -> &str {
        match self {
            Self::Amendment { description, .. }
            | Self::Sunset { description, .. }
            | Self::Repeal { description, .. }
            | Self::GracePeriodStart { description, .. }
            | Self::GracePeriodEnd { description, .. } => description,
            Self::PhaseIn { milestone, .. } => milestone,
        }
    }
}

impl fmt::Display for RegulatoryEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Amendment { description, date } => {
                write!(f, "Amendment: {} ({})", description, date)
            }
            Self::PhaseIn { milestone, date } => {
                write!(f, "PhaseIn: {} ({})", milestone, date)
            }
            Self::Sunset { description, date } => {
                write!(f, "Sunset: {} ({})", description, date)
            }
            Self::Repeal { description, date } => {
                write!(f, "Repeal: {} ({})", description, date)
            }
            Self::GracePeriodStart { description, date } => {
                write!(f, "GracePeriodStart: {} ({})", description, date)
            }
            Self::GracePeriodEnd { description, date } => {
                write!(f, "GracePeriodEnd: {} ({})", description, date)
            }
        }
    }
}

/// Errors in the temporal model.
#[derive(Debug, Clone, thiserror::Error)]
pub enum TemporalError {
    #[error("State not found: {0}")]
    StateNotFound(String),
    #[error("Invalid transition: {0}")]
    InvalidTransition(String),
    #[error("Version not found: {0}")]
    VersionNotFound(String),
    #[error("Merge conflict: {0}")]
    MergeConflict(String),
    #[error("Invalid timeline: {0}")]
    InvalidTimeline(String),
    #[error("Temporal conflict: {0}")]
    TemporalConflict(String),
}

/// Check whether a date falls within a temporal interval.
pub fn date_in_interval(date: &NaiveDate, interval: &TemporalInterval) -> bool {
    let after_start = interval.start.map_or(true, |s| *date >= s);
    let before_end = interval.end.map_or(true, |e| *date <= e);
    after_start && before_end
}

/// Helper: construct a `NaiveDate` from y/m/d (panics on invalid input).
pub(crate) fn ymd(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).expect("invalid date literal")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obligation_creation() {
        let obl = Obligation::new(
            "obl-1",
            ObligationKind::Obligation,
            Jurisdiction::new("EU"),
            "Test obligation",
        );
        assert_eq!(obl.id, "obl-1");
        assert_eq!(obl.kind, ObligationKind::Obligation);
        assert_eq!(obl.grade, FormalizabilityGrade::F1);
    }

    #[test]
    fn test_obligation_active_at() {
        let obl = Obligation::new(
            "obl-1",
            ObligationKind::Obligation,
            Jurisdiction::new("EU"),
            "Test",
        )
        .with_interval(TemporalInterval::new(
            Some(ymd(2025, 1, 1)),
            Some(ymd(2025, 12, 31)),
        ));
        assert!(obl.is_active_at(&ymd(2025, 6, 15)));
        assert!(!obl.is_active_at(&ymd(2024, 6, 15)));
        assert!(!obl.is_active_at(&ymd(2026, 1, 1)));
    }

    #[test]
    fn test_obligation_builder() {
        let obl = Obligation::new(
            "obl-2",
            ObligationKind::Prohibition,
            Jurisdiction::new("US"),
            "No subliminal AI",
        )
        .with_risk_level(RiskLevel::Unacceptable)
        .with_grade(FormalizabilityGrade::F3)
        .with_article_ref(ArticleRef {
            framework: "EU-AI-Act".into(),
            article: "5".into(),
            paragraph: Some("1(a)".into()),
        });
        assert_eq!(obl.risk_level, Some(RiskLevel::Unacceptable));
        assert_eq!(obl.grade, FormalizabilityGrade::F3);
        assert!(obl.article_ref.is_some());
    }

    #[test]
    fn test_state_obligation_equivalence() {
        let mut s1 = RegulatoryState::new("s1");
        s1.add_obligation("obl-1".into());
        s1.add_obligation("obl-2".into());

        let mut s2 = RegulatoryState::new("s2");
        s2.add_obligation("obl-2".into());
        s2.add_obligation("obl-1".into());

        assert!(s1.obligation_equivalent(&s2));
    }

    #[test]
    fn test_state_remove_obligation() {
        let mut s = RegulatoryState::new("s");
        s.add_obligation("a".into());
        s.add_obligation("b".into());
        assert!(s.remove_obligation("a"));
        assert_eq!(s.obligation_count(), 1);
        assert!(!s.remove_obligation("a"));
    }

    #[test]
    fn test_regulatory_event_date() {
        let event = RegulatoryEvent::Amendment {
            description: "Test".into(),
            date: ymd(2025, 3, 1),
        };
        assert_eq!(event.date(), ymd(2025, 3, 1));
    }

    #[test]
    fn test_date_in_interval_bounded() {
        let interval = TemporalInterval::new(Some(ymd(2025, 1, 1)), Some(ymd(2025, 12, 31)));
        assert!(date_in_interval(&ymd(2025, 6, 15), &interval));
        assert!(date_in_interval(&ymd(2025, 1, 1), &interval));
        assert!(date_in_interval(&ymd(2025, 12, 31), &interval));
        assert!(!date_in_interval(&ymd(2024, 12, 31), &interval));
    }

    #[test]
    fn test_date_in_interval_unbounded() {
        assert!(date_in_interval(&ymd(2025, 6, 15), &TemporalInterval::unbounded()));
    }

    #[test]
    fn test_date_in_half_bounded() {
        let left = TemporalInterval::new(Some(ymd(2025, 1, 1)), None);
        assert!(date_in_interval(&ymd(2099, 1, 1), &left));
        assert!(!date_in_interval(&ymd(2024, 12, 31), &left));

        let right = TemporalInterval::new(None, Some(ymd(2025, 12, 31)));
        assert!(date_in_interval(&ymd(2020, 1, 1), &right));
        assert!(!date_in_interval(&ymd(2026, 1, 1), &right));
    }

    #[test]
    fn test_event_display() {
        let e = RegulatoryEvent::PhaseIn {
            milestone: "M1".into(),
            date: ymd(2025, 2, 2),
        };
        let s = format!("{}", e);
        assert!(s.contains("PhaseIn"));
        assert!(s.contains("M1"));
    }

    #[test]
    fn test_state_display() {
        let mut st = RegulatoryState::new("test-state");
        st.add_obligation("obl-1".into());
        st.add_obligation("obl-2".into());
        let d = format!("{}", st);
        assert!(d.contains("test-state"));
        assert!(d.contains("2 obligations"));
    }
}
