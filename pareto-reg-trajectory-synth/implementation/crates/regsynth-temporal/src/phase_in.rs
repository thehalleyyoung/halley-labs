use crate::{ObligationId, ymd};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// A single milestone in a phase-in schedule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Milestone {
    pub date: NaiveDate,
    pub label: String,
    pub obligations: BTreeSet<ObligationId>,
}

impl Milestone {
    pub fn new(date: NaiveDate, label: impl Into<String>, obligations: BTreeSet<ObligationId>) -> Self {
        Self { date, label: label.into(), obligations }
    }
}

impl PartialOrd for Milestone {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Milestone {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.date.cmp(&other.date).then_with(|| self.label.cmp(&other.label))
    }
}

/// A phase-in schedule: an ordered sequence of milestones for a regulatory framework.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseInSchedule {
    pub framework: String,
    milestones: Vec<Milestone>,
}

impl PhaseInSchedule {
    pub fn new(framework: impl Into<String>) -> Self {
        Self {
            framework: framework.into(),
            milestones: Vec::new(),
        }
    }

    pub fn add_milestone(&mut self, milestone: Milestone) {
        self.milestones.push(milestone);
        self.milestones.sort();
    }

    pub fn milestones(&self) -> &[Milestone] {
        &self.milestones
    }

    /// Returns the union of obligations from all milestones whose date <= `date`.
    pub fn obligations_active_at(&self, date: &NaiveDate) -> BTreeSet<ObligationId> {
        let mut active = BTreeSet::new();
        for m in &self.milestones {
            if m.date <= *date {
                active.extend(m.obligations.iter().cloned());
            }
        }
        active
    }

    /// Returns the next milestone strictly after `date`, if any.
    pub fn next_milestone(&self, date: &NaiveDate) -> Option<&Milestone> {
        self.milestones.iter().find(|m| m.date > *date)
    }

    /// Returns the most recent milestone at or before `date`, if any.
    pub fn current_milestone(&self, date: &NaiveDate) -> Option<&Milestone> {
        self.milestones.iter().rev().find(|m| m.date <= *date)
    }

    /// Returns the sorted list of dates where transitions happen.
    pub fn compute_transition_points(&self) -> Vec<NaiveDate> {
        self.milestones.iter().map(|m| m.date).collect()
    }

    pub fn len(&self) -> usize {
        self.milestones.len()
    }

    pub fn is_empty(&self) -> bool {
        self.milestones.is_empty()
    }

    /// Total number of distinct obligations across all milestones.
    pub fn total_obligation_count(&self) -> usize {
        let all: BTreeSet<_> = self.milestones.iter()
            .flat_map(|m| m.obligations.iter().cloned())
            .collect();
        all.len()
    }
}

/// Operator that wraps a PhaseInSchedule for querying activation status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseInOperator {
    pub schedule: PhaseInSchedule,
}

impl PhaseInOperator {
    pub fn new(schedule: PhaseInSchedule) -> Self {
        Self { schedule }
    }

    /// Returns obligations active at the given date.
    pub fn active_at(&self, date: &NaiveDate) -> BTreeSet<ObligationId> {
        self.schedule.obligations_active_at(date)
    }

    /// Returns the date of the next activation milestone after `date`, if any.
    pub fn next_activation(&self, date: &NaiveDate) -> Option<NaiveDate> {
        self.schedule.next_milestone(date).map(|m| m.date)
    }
}

/// A grace period for a specific obligation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GracePeriod {
    pub obligation_id: ObligationId,
    pub start: NaiveDate,
    pub end: NaiveDate,
    pub description: String,
}

impl GracePeriod {
    pub fn new(obligation_id: impl Into<String>, start: NaiveDate, end: NaiveDate) -> Self {
        Self {
            obligation_id: obligation_id.into(),
            start,
            end,
            description: String::new(),
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Returns true if the given date falls within the grace period (inclusive).
    pub fn is_in_grace(&self, date: &NaiveDate) -> bool {
        *date >= self.start && *date <= self.end
    }

    /// Duration of the grace period in days.
    pub fn duration_days(&self) -> i64 {
        (self.end - self.start).num_days()
    }
}

/// Operator that manages sunset (expiry) dates for obligations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SunsetOperator {
    sunsets: Vec<(ObligationId, NaiveDate)>,
}

impl SunsetOperator {
    pub fn new() -> Self {
        Self { sunsets: Vec::new() }
    }

    pub fn add_sunset(&mut self, obligation_id: impl Into<String>, date: NaiveDate) {
        self.sunsets.push((obligation_id.into(), date));
    }

    /// Remove obligations that have been sunset by the given date.
    pub fn apply(&self, obligations: &BTreeSet<ObligationId>, date: &NaiveDate) -> BTreeSet<ObligationId> {
        let sunset_ids: BTreeSet<&str> = self.sunsets.iter()
            .filter(|(_, d)| *date >= *d)
            .map(|(id, _)| id.as_str())
            .collect();
        obligations.iter()
            .filter(|o| !sunset_ids.contains(o.as_str()))
            .cloned()
            .collect()
    }

    /// Returns all sunset entries.
    pub fn sunset_dates(&self) -> &[(ObligationId, NaiveDate)] {
        &self.sunsets
    }

    /// Check if a specific obligation has been sunset by the given date.
    pub fn is_sunset(&self, obligation_id: &str, date: &NaiveDate) -> bool {
        self.sunsets.iter().any(|(id, d)| id == obligation_id && *date >= *d)
    }
}

impl Default for SunsetOperator {
    fn default() -> Self {
        Self::new()
    }
}

/// Constructs the EU AI Act phase-in schedule with 4 milestones.
pub fn eu_ai_act_schedule() -> PhaseInSchedule {
    let mut schedule = PhaseInSchedule::new("EU-AI-Act");

    let m1_obligations: BTreeSet<ObligationId> = [
        "prohibited-subliminal",
        "prohibited-social-scoring",
        "prohibited-biometric-categorization",
    ].iter().map(|s| s.to_string()).collect();
    schedule.add_milestone(Milestone::new(
        ymd(2025, 2, 2),
        "Prohibited AI Practices",
        m1_obligations,
    ));

    let m2_obligations: BTreeSet<ObligationId> = [
        "gpai-transparency",
        "gpai-copyright",
        "gpai-systemic-risk",
    ].iter().map(|s| s.to_string()).collect();
    schedule.add_milestone(Milestone::new(
        ymd(2025, 8, 2),
        "GPAI Obligations",
        m2_obligations,
    ));

    let m3_obligations: BTreeSet<ObligationId> = [
        "high-risk-conformity",
        "high-risk-risk-management",
        "high-risk-data-governance",
        "high-risk-transparency-users",
        "high-risk-human-oversight",
    ].iter().map(|s| s.to_string()).collect();
    schedule.add_milestone(Milestone::new(
        ymd(2026, 8, 2),
        "High-Risk AI Obligations",
        m3_obligations,
    ));

    let m4_obligations: BTreeSet<ObligationId> = [
        "full-market-surveillance",
        "full-penalties",
        "full-reporting",
    ].iter().map(|s| s.to_string()).collect();
    schedule.add_milestone(Milestone::new(
        ymd(2027, 8, 2),
        "Full Enforcement",
        m4_obligations,
    ));

    schedule
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ymd;

    #[test]
    fn test_milestone_ordering() {
        let m1 = Milestone::new(ymd(2025, 1, 1), "First", BTreeSet::new());
        let m2 = Milestone::new(ymd(2026, 1, 1), "Second", BTreeSet::new());
        assert!(m1 < m2);
    }

    #[test]
    fn test_phase_in_schedule_sorted() {
        let mut sched = PhaseInSchedule::new("test");
        let m2 = Milestone::new(ymd(2026, 1, 1), "Second", BTreeSet::new());
        let m1 = Milestone::new(ymd(2025, 1, 1), "First", BTreeSet::new());
        sched.add_milestone(m2);
        sched.add_milestone(m1);
        assert_eq!(sched.milestones()[0].label, "First");
        assert_eq!(sched.milestones()[1].label, "Second");
    }

    #[test]
    fn test_obligations_active_at() {
        let sched = eu_ai_act_schedule();
        let before = ymd(2025, 1, 1);
        assert!(sched.obligations_active_at(&before).is_empty());

        let after_m1 = ymd(2025, 3, 1);
        let active = sched.obligations_active_at(&after_m1);
        assert!(active.contains("prohibited-subliminal"));
        assert!(active.contains("prohibited-social-scoring"));
        assert!(!active.contains("gpai-transparency"));

        let after_m2 = ymd(2025, 9, 1);
        let active2 = sched.obligations_active_at(&after_m2);
        assert!(active2.contains("prohibited-subliminal"));
        assert!(active2.contains("gpai-transparency"));
        assert!(!active2.contains("high-risk-conformity"));
    }

    #[test]
    fn test_all_milestones_active() {
        let sched = eu_ai_act_schedule();
        let far_future = ymd(2030, 1, 1);
        let active = sched.obligations_active_at(&far_future);
        assert_eq!(active.len(), sched.total_obligation_count());
        assert_eq!(sched.total_obligation_count(), 14);
    }

    #[test]
    fn test_next_milestone() {
        let sched = eu_ai_act_schedule();
        let m = sched.next_milestone(&ymd(2025, 3, 1)).unwrap();
        assert_eq!(m.label, "GPAI Obligations");
        assert!(sched.next_milestone(&ymd(2028, 1, 1)).is_none());
    }

    #[test]
    fn test_current_milestone() {
        let sched = eu_ai_act_schedule();
        assert!(sched.current_milestone(&ymd(2024, 1, 1)).is_none());
        let cur = sched.current_milestone(&ymd(2025, 5, 1)).unwrap();
        assert_eq!(cur.label, "Prohibited AI Practices");
        let cur2 = sched.current_milestone(&ymd(2026, 9, 1)).unwrap();
        assert_eq!(cur2.label, "High-Risk AI Obligations");
    }

    #[test]
    fn test_transition_points() {
        let sched = eu_ai_act_schedule();
        let points = sched.compute_transition_points();
        assert_eq!(points.len(), 4);
        assert_eq!(points[0], ymd(2025, 2, 2));
        assert_eq!(points[3], ymd(2027, 8, 2));
    }

    #[test]
    fn test_schedule_len() {
        let sched = eu_ai_act_schedule();
        assert_eq!(sched.len(), 4);
        assert!(!sched.is_empty());
        let empty = PhaseInSchedule::new("empty");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_phase_in_operator() {
        let sched = eu_ai_act_schedule();
        let op = PhaseInOperator::new(sched);
        let active = op.active_at(&ymd(2025, 9, 1));
        assert!(active.contains("gpai-transparency"));
        let next = op.next_activation(&ymd(2025, 3, 1));
        assert_eq!(next, Some(ymd(2025, 8, 2)));
    }

    #[test]
    fn test_grace_period() {
        let gp = GracePeriod::new("obl-1", ymd(2025, 1, 1), ymd(2025, 6, 30))
            .with_description("Six month grace period");
        assert!(gp.is_in_grace(&ymd(2025, 3, 15)));
        assert!(gp.is_in_grace(&ymd(2025, 1, 1)));
        assert!(gp.is_in_grace(&ymd(2025, 6, 30)));
        assert!(!gp.is_in_grace(&ymd(2024, 12, 31)));
        assert!(!gp.is_in_grace(&ymd(2025, 7, 1)));
        assert_eq!(gp.duration_days(), 180);
        assert_eq!(gp.description, "Six month grace period");
    }

    #[test]
    fn test_sunset_operator() {
        let mut sunset = SunsetOperator::new();
        sunset.add_sunset("obl-A", ymd(2026, 1, 1));
        sunset.add_sunset("obl-B", ymd(2027, 1, 1));

        let mut obligations = BTreeSet::new();
        obligations.insert("obl-A".to_string());
        obligations.insert("obl-B".to_string());
        obligations.insert("obl-C".to_string());

        let result = sunset.apply(&obligations, &ymd(2025, 6, 1));
        assert_eq!(result.len(), 3);

        let result = sunset.apply(&obligations, &ymd(2026, 6, 1));
        assert_eq!(result.len(), 2);
        assert!(!result.contains("obl-A"));
        assert!(result.contains("obl-B"));

        let result = sunset.apply(&obligations, &ymd(2027, 6, 1));
        assert_eq!(result.len(), 1);
        assert!(result.contains("obl-C"));
    }

    #[test]
    fn test_sunset_is_sunset() {
        let mut sunset = SunsetOperator::new();
        sunset.add_sunset("obl-X", ymd(2026, 1, 1));
        assert!(!sunset.is_sunset("obl-X", &ymd(2025, 12, 31)));
        assert!(sunset.is_sunset("obl-X", &ymd(2026, 1, 1)));
        assert!(sunset.is_sunset("obl-X", &ymd(2027, 1, 1)));
        assert!(!sunset.is_sunset("obl-Y", &ymd(2027, 1, 1)));
    }

    #[test]
    fn test_sunset_dates() {
        let mut sunset = SunsetOperator::new();
        sunset.add_sunset("a", ymd(2025, 1, 1));
        sunset.add_sunset("b", ymd(2026, 1, 1));
        assert_eq!(sunset.sunset_dates().len(), 2);
    }
}
