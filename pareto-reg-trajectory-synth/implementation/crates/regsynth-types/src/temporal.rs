use std::fmt;

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

/// Timestamp used throughout the system, backed by chrono UTC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Timestamp(pub DateTime<Utc>);

impl Timestamp {
    pub fn now() -> Self {
        Self(Utc::now())
    }

    pub fn from_utc(dt: DateTime<Utc>) -> Self {
        Self(dt)
    }

    pub fn from_ymd(year: i32, month: u32, day: u32) -> Option<Self> {
        NaiveDate::from_ymd_opt(year, month, day).map(|d| {
            Self(DateTime::from_naive_utc_and_offset(
                d.and_hms_opt(0, 0, 0).unwrap(),
                Utc,
            ))
        })
    }

    pub fn inner(&self) -> DateTime<Utc> {
        self.0
    }

    pub fn days_until(&self, other: &Timestamp) -> i64 {
        (other.0 - self.0).num_days()
    }

    pub fn add_days(&self, days: i64) -> Self {
        Self(self.0 + chrono::Duration::days(days))
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.format("%Y-%m-%d"))
    }
}

/// A closed temporal interval [start, end].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalInterval {
    pub start: Timestamp,
    pub end: Timestamp,
}

impl TemporalInterval {
    /// An interval that contains all possible timestamps.
    pub fn always() -> Self {
        use chrono::DateTime;
        Self {
            start: Timestamp(DateTime::<Utc>::MIN_UTC),
            end: Timestamp(DateTime::<Utc>::MAX_UTC),
        }
    }

    pub fn new(start: Timestamp, end: Timestamp) -> Result<Self, String> {
        if start > end {
            return Err(format!("start {} is after end {}", start, end));
        }
        Ok(Self { start, end })
    }

    /// Unchecked constructor for known-valid intervals.
    pub fn new_unchecked(start: Timestamp, end: Timestamp) -> Self {
        Self { start, end }
    }

    pub fn duration_days(&self) -> i64 {
        self.start.days_until(&self.end)
    }

    pub fn contains_timestamp(&self, ts: &Timestamp) -> bool {
        ts >= &self.start && ts <= &self.end
    }

    /// Check whether a Unix timestamp (seconds since epoch) falls within this interval.
    pub fn contains(&self, unix_secs: i64) -> bool {
        use chrono::DateTime;
        let dt = DateTime::from_timestamp(unix_secs, 0)
            .unwrap_or(DateTime::<Utc>::MIN_UTC);
        self.contains_timestamp(&Timestamp(dt))
    }

    pub fn contains_interval(&self, other: &TemporalInterval) -> bool {
        self.start <= other.start && self.end >= other.end
    }

    /// Whether two intervals overlap.
    pub fn overlaps(&self, other: &TemporalInterval) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    /// Compute the intersection of two intervals if they overlap.
    pub fn intersection(&self, other: &TemporalInterval) -> Option<TemporalInterval> {
        if !self.overlaps(other) {
            return None;
        }
        let start = if self.start > other.start {
            self.start
        } else {
            other.start
        };
        let end = if self.end < other.end {
            self.end
        } else {
            other.end
        };
        Some(TemporalInterval { start, end })
    }

    /// Compute the union (convex hull) of two intervals.
    pub fn hull(&self, other: &TemporalInterval) -> TemporalInterval {
        let start = if self.start < other.start {
            self.start
        } else {
            other.start
        };
        let end = if self.end > other.end {
            self.end
        } else {
            other.end
        };
        TemporalInterval { start, end }
    }

    /// Split an interval at a given timestamp.
    pub fn split_at(&self, ts: &Timestamp) -> Option<(TemporalInterval, TemporalInterval)> {
        if !self.contains_timestamp(ts) || *ts == self.start || *ts == self.end {
            return None;
        }
        Some((
            TemporalInterval {
                start: self.start,
                end: *ts,
            },
            TemporalInterval {
                start: *ts,
                end: self.end,
            },
        ))
    }

    /// Whether this interval precedes another with no overlap.
    pub fn before(&self, other: &TemporalInterval) -> bool {
        self.end < other.start
    }

    /// Whether this interval follows another with no overlap.
    pub fn after(&self, other: &TemporalInterval) -> bool {
        self.start > other.end
    }
}

impl fmt::Display for TemporalInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.start, self.end)
    }
}

/// Builder for TemporalInterval.
pub struct TemporalIntervalBuilder {
    start: Option<Timestamp>,
    end: Option<Timestamp>,
}

impl TemporalIntervalBuilder {
    pub fn new() -> Self {
        Self {
            start: None,
            end: None,
        }
    }

    pub fn start_ymd(mut self, y: i32, m: u32, d: u32) -> Self {
        self.start = Timestamp::from_ymd(y, m, d);
        self
    }

    pub fn end_ymd(mut self, y: i32, m: u32, d: u32) -> Self {
        self.end = Timestamp::from_ymd(y, m, d);
        self
    }

    pub fn start(mut self, ts: Timestamp) -> Self {
        self.start = Some(ts);
        self
    }

    pub fn end(mut self, ts: Timestamp) -> Self {
        self.end = Some(ts);
        self
    }

    pub fn build(self) -> Result<TemporalInterval, String> {
        let start = self.start.ok_or("start is required")?;
        let end = self.end.ok_or("end is required")?;
        TemporalInterval::new(start, end)
    }
}

/// An event on a regulatory timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    pub name: String,
    pub timestamp: Timestamp,
    pub description: String,
    pub event_type: TimelineEventType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimelineEventType {
    /// Regulation published / adopted
    Enacted,
    /// Enforcement begins for some provisions
    PartialEnforcement,
    /// Full enforcement begins
    FullEnforcement,
    /// Grace period expires
    GracePeriodEnd,
    /// Regulation repealed / sunset
    Sunset,
    /// Amendment effective
    Amendment,
    /// Deadline for a specific compliance milestone
    Milestone,
}

impl fmt::Display for TimelineEventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Enacted => "Enacted",
            Self::PartialEnforcement => "Partial Enforcement",
            Self::FullEnforcement => "Full Enforcement",
            Self::GracePeriodEnd => "Grace Period End",
            Self::Sunset => "Sunset",
            Self::Amendment => "Amendment",
            Self::Milestone => "Milestone",
        };
        write!(f, "{}", s)
    }
}

impl fmt::Display for TimelineEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} - {} ({})", self.timestamp, self.event_type, self.name, self.description)
    }
}

/// A phased enforcement schedule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseInSchedule {
    pub name: String,
    pub phases: Vec<PhaseInPhase>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseInPhase {
    pub phase_name: String,
    pub interval: TemporalInterval,
    pub compliance_fraction: f64,
    pub description: String,
}

impl Eq for PhaseInPhase {}

impl PhaseInSchedule {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            phases: Vec::new(),
        }
    }

    pub fn add_phase(
        &mut self,
        phase_name: impl Into<String>,
        interval: TemporalInterval,
        compliance_fraction: f64,
        description: impl Into<String>,
    ) {
        self.phases.push(PhaseInPhase {
            phase_name: phase_name.into(),
            interval,
            compliance_fraction: compliance_fraction.clamp(0.0, 1.0),
            description: description.into(),
        });
    }

    /// Get the compliance fraction required at a given timestamp.
    pub fn compliance_at(&self, ts: &Timestamp) -> f64 {
        for phase in self.phases.iter().rev() {
            if phase.interval.contains_timestamp(ts) {
                return phase.compliance_fraction;
            }
        }
        // Before first phase: 0, after last phase: last phase's fraction
        if let Some(last) = self.phases.last() {
            if *ts > last.interval.end {
                return last.compliance_fraction;
            }
        }
        0.0
    }

    pub fn total_duration_days(&self) -> i64 {
        if self.phases.is_empty() {
            return 0;
        }
        let start = self
            .phases
            .iter()
            .map(|p| p.interval.start)
            .min()
            .unwrap();
        let end = self.phases.iter().map(|p| p.interval.end).max().unwrap();
        start.days_until(&end)
    }

    pub fn is_complete_at(&self, ts: &Timestamp) -> bool {
        self.compliance_at(ts) >= 1.0
    }

    /// Validate: phases should not overlap and compliance should be monotonic.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        for i in 0..self.phases.len() {
            if self.phases[i].compliance_fraction < 0.0
                || self.phases[i].compliance_fraction > 1.0
            {
                errors.push(format!(
                    "Phase '{}' has invalid compliance fraction {}",
                    self.phases[i].phase_name, self.phases[i].compliance_fraction
                ));
            }
            for j in (i + 1)..self.phases.len() {
                if self.phases[i].interval.overlaps(&self.phases[j].interval) {
                    errors.push(format!(
                        "Phases '{}' and '{}' overlap",
                        self.phases[i].phase_name, self.phases[j].phase_name
                    ));
                }
            }
        }
        errors
    }
}

impl fmt::Display for PhaseInSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PhaseIn({}, {} phases)", self.name, self.phases.len())
    }
}

/// Temporal bound for obligation applicability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalBound {
    Before(Timestamp),
    After(Timestamp),
    During(TemporalInterval),
    Between(Timestamp, Timestamp),
    Unbounded,
}

impl TemporalBound {
    /// Check whether a timestamp satisfies this bound.
    pub fn satisfied_by(&self, ts: &Timestamp) -> bool {
        match self {
            Self::Before(deadline) => ts < deadline,
            Self::After(start) => ts >= start,
            Self::During(interval) => interval.contains_timestamp(ts),
            Self::Between(start, end) => ts >= start && ts <= end,
            Self::Unbounded => true,
        }
    }

    /// Intersect two bounds.
    pub fn intersect(&self, other: &TemporalBound) -> TemporalBound {
        match (self, other) {
            (Self::Unbounded, b) => b.clone(),
            (a, Self::Unbounded) => a.clone(),
            (Self::After(a), Self::Before(b)) | (Self::Before(b), Self::After(a)) => {
                if a <= b {
                    Self::Between(*a, *b)
                } else {
                    // empty interval; represent as Between with inverted bounds
                    Self::Between(*a, *a) // degenerate point
                }
            }
            (Self::After(a), Self::After(b)) => Self::After(if *a > *b { *a } else { *b }),
            (Self::Before(a), Self::Before(b)) => Self::Before(if *a < *b { *a } else { *b }),
            (Self::During(i1), Self::During(i2)) => {
                match i1.intersection(i2) {
                    Some(inter) => Self::During(inter),
                    None => {
                        // empty; use degenerate
                        let t = if i1.end < i2.start { i1.end } else { i2.end };
                        Self::Between(t, t)
                    }
                }
            }
            _ => self.clone(), // conservative fallback
        }
    }
}

impl fmt::Display for TemporalBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Before(ts) => write!(f, "before {}", ts),
            Self::After(ts) => write!(f, "after {}", ts),
            Self::During(iv) => write!(f, "during {}", iv),
            Self::Between(a, b) => write!(f, "between {} and {}", a, b),
            Self::Unbounded => write!(f, "unbounded"),
        }
    }
}

/// Temporal operators used in obligation specifications.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalOperator {
    /// Must hold at all times in the interval.
    Always(TemporalInterval),
    /// Must hold at some time in the interval.
    Eventually(TemporalInterval),
    /// p holds until q becomes true.
    Until {
        condition: String,
        deadline: Timestamp,
    },
    /// Gradual phase-in with a schedule.
    PhaseIn(PhaseInSchedule),
    /// Active until sunset date.
    Sunset(Timestamp),
}

impl TemporalOperator {
    /// Whether the operator is currently active at the given time.
    pub fn is_active_at(&self, ts: &Timestamp) -> bool {
        match self {
            Self::Always(interval) => interval.contains_timestamp(ts),
            Self::Eventually(interval) => interval.contains_timestamp(ts),
            Self::Until { deadline, .. } => *ts <= *deadline,
            Self::PhaseIn(schedule) => schedule.compliance_at(ts) > 0.0,
            Self::Sunset(sunset) => *ts <= *sunset,
        }
    }

    /// Deadline (if any) associated with this operator.
    pub fn deadline(&self) -> Option<Timestamp> {
        match self {
            Self::Always(interval) => Some(interval.end),
            Self::Eventually(interval) => Some(interval.end),
            Self::Until { deadline, .. } => Some(*deadline),
            Self::PhaseIn(schedule) => schedule.phases.last().map(|p| p.interval.end),
            Self::Sunset(ts) => Some(*ts),
        }
    }
}

impl fmt::Display for TemporalOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Always(iv) => write!(f, "□{}", iv),
            Self::Eventually(iv) => write!(f, "◇{}", iv),
            Self::Until {
                condition,
                deadline,
            } => write!(f, "U({}, {})", condition, deadline),
            Self::PhaseIn(s) => write!(f, "PhaseIn({})", s.name),
            Self::Sunset(ts) => write!(f, "Sunset({})", ts),
        }
    }
}

/// An enforcement milestone with a deadline and description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementMilestone {
    pub name: String,
    pub deadline: Timestamp,
    pub description: String,
    pub penalty_on_miss: Option<String>,
    pub is_hard_deadline: bool,
}

impl EnforcementMilestone {
    pub fn is_overdue(&self, now: &Timestamp) -> bool {
        *now > self.deadline
    }

    pub fn days_remaining(&self, now: &Timestamp) -> i64 {
        now.days_until(&self.deadline)
    }
}

impl fmt::Display for EnforcementMilestone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.is_hard_deadline { "HARD" } else { "soft" };
        write!(f, "[{}] {} by {} - {}", kind, self.name, self.deadline, self.description)
    }
}

/// Sort milestones by deadline.
pub fn sort_milestones(milestones: &mut [EnforcementMilestone]) {
    milestones.sort_by(|a, b| a.deadline.cmp(&b.deadline));
}

/// Find the next upcoming milestone.
pub fn next_milestone<'a>(
    milestones: &'a [EnforcementMilestone],
    now: &Timestamp,
) -> Option<&'a EnforcementMilestone> {
    milestones
        .iter()
        .filter(|m| m.deadline >= *now)
        .min_by_key(|m| m.deadline)
}

/// Compute the overlap fraction of interval `a` that is covered by `b`.
pub fn overlap_fraction(a: &TemporalInterval, b: &TemporalInterval) -> f64 {
    let a_days = a.duration_days() as f64;
    if a_days <= 0.0 {
        return 0.0;
    }
    match a.intersection(b) {
        Some(inter) => inter.duration_days() as f64 / a_days,
        None => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(y: i32, m: u32, d: u32) -> Timestamp {
        Timestamp::from_ymd(y, m, d).unwrap()
    }

    fn iv(y1: i32, m1: u32, d1: u32, y2: i32, m2: u32, d2: u32) -> TemporalInterval {
        TemporalInterval::new(ts(y1, m1, d1), ts(y2, m2, d2)).unwrap()
    }

    #[test]
    fn test_timestamp_ordering() {
        let a = ts(2024, 1, 1);
        let b = ts(2025, 6, 15);
        assert!(a < b);
        assert_eq!(a.days_until(&b), 531);
    }

    #[test]
    fn test_interval_creation() {
        let interval = iv(2024, 1, 1, 2025, 12, 31);
        assert!(interval.duration_days() > 700);
    }

    #[test]
    fn test_interval_invalid() {
        let result = TemporalInterval::new(ts(2025, 1, 1), ts(2024, 1, 1));
        assert!(result.is_err());
    }

    #[test]
    fn test_interval_contains() {
        let interval = iv(2024, 1, 1, 2024, 12, 31);
        assert!(interval.contains_timestamp(&ts(2024, 6, 15)));
        assert!(!interval.contains_timestamp(&ts(2025, 1, 1)));
    }

    #[test]
    fn test_interval_overlap() {
        let a = iv(2024, 1, 1, 2024, 6, 30);
        let b = iv(2024, 3, 1, 2024, 9, 30);
        assert!(a.overlaps(&b));
        let inter = a.intersection(&b).unwrap();
        assert_eq!(inter.start, ts(2024, 3, 1));
        assert_eq!(inter.end, ts(2024, 6, 30));
    }

    #[test]
    fn test_interval_no_overlap() {
        let a = iv(2024, 1, 1, 2024, 3, 31);
        let b = iv(2024, 6, 1, 2024, 9, 30);
        assert!(!a.overlaps(&b));
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn test_interval_hull() {
        let a = iv(2024, 1, 1, 2024, 3, 31);
        let b = iv(2024, 6, 1, 2024, 9, 30);
        let hull = a.hull(&b);
        assert_eq!(hull.start, ts(2024, 1, 1));
        assert_eq!(hull.end, ts(2024, 9, 30));
    }

    #[test]
    fn test_interval_split() {
        let interval = iv(2024, 1, 1, 2024, 12, 31);
        let (left, right) = interval.split_at(&ts(2024, 6, 15)).unwrap();
        assert_eq!(left.end, ts(2024, 6, 15));
        assert_eq!(right.start, ts(2024, 6, 15));
    }

    #[test]
    fn test_temporal_bound() {
        let bound = TemporalBound::After(ts(2024, 6, 1));
        assert!(bound.satisfied_by(&ts(2024, 7, 1)));
        assert!(!bound.satisfied_by(&ts(2024, 5, 1)));
    }

    #[test]
    fn test_temporal_bound_intersect() {
        let a = TemporalBound::After(ts(2024, 3, 1));
        let b = TemporalBound::Before(ts(2024, 9, 1));
        let inter = a.intersect(&b);
        assert!(inter.satisfied_by(&ts(2024, 6, 1)));
        assert!(!inter.satisfied_by(&ts(2024, 1, 1)));
    }

    #[test]
    fn test_phase_in_schedule() {
        let mut schedule = PhaseInSchedule::new("EU AI Act Phase-In");
        schedule.add_phase(
            "Phase 1: Prohibited",
            iv(2024, 8, 1, 2025, 2, 1),
            0.3,
            "Prohibited AI practices",
        );
        schedule.add_phase(
            "Phase 2: High-Risk",
            iv(2025, 2, 2, 2026, 8, 1),
            0.7,
            "High-risk AI systems",
        );
        schedule.add_phase(
            "Phase 3: Full",
            iv(2026, 8, 2, 2027, 8, 2),
            1.0,
            "Full compliance",
        );

        assert!((schedule.compliance_at(&ts(2024, 10, 1)) - 0.3).abs() < 1e-9);
        assert!((schedule.compliance_at(&ts(2025, 6, 1)) - 0.7).abs() < 1e-9);
        assert!(schedule.is_complete_at(&ts(2027, 1, 1)));
        assert!(!schedule.is_complete_at(&ts(2025, 1, 1)));
        assert!(schedule.validate().is_empty());
    }

    #[test]
    fn test_temporal_operator_active() {
        let op = TemporalOperator::Sunset(ts(2030, 1, 1));
        assert!(op.is_active_at(&ts(2025, 1, 1)));
        assert!(!op.is_active_at(&ts(2031, 1, 1)));
    }

    #[test]
    fn test_enforcement_milestone() {
        let m = EnforcementMilestone {
            name: "Conformity Assessment".into(),
            deadline: ts(2025, 8, 1),
            description: "Complete conformity assessment for high-risk AI".into(),
            penalty_on_miss: Some("Up to €15M fine".into()),
            is_hard_deadline: true,
        };
        assert!(m.is_overdue(&ts(2025, 9, 1)));
        assert!(!m.is_overdue(&ts(2025, 7, 1)));
        assert!(m.days_remaining(&ts(2025, 7, 1)) > 0);
    }

    #[test]
    fn test_overlap_fraction() {
        let a = iv(2024, 1, 1, 2024, 12, 31);
        let b = iv(2024, 7, 1, 2025, 6, 30);
        let frac = overlap_fraction(&a, &b);
        assert!(frac > 0.4 && frac < 0.6);
    }

    #[test]
    fn test_serialization() {
        let interval = iv(2024, 1, 1, 2025, 12, 31);
        let json = serde_json::to_string(&interval).unwrap();
        let deser: TemporalInterval = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.start, interval.start);
        assert_eq!(deser.end, interval.end);
    }
}
