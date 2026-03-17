use serde::{Deserialize, Serialize};
use chrono::NaiveDate;

use crate::roadmap::TaskStatus;

// ─── Milestone Status ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MilestoneStatus {
    Upcoming,
    InProgress,
    Achieved,
    Missed,
}

impl std::fmt::Display for MilestoneStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Upcoming => write!(f, "Upcoming"),
            Self::InProgress => write!(f, "In Progress"),
            Self::Achieved => write!(f, "Achieved"),
            Self::Missed => write!(f, "Missed"),
        }
    }
}

// ─── Milestone ──────────────────────────────────────────────────────────────

/// A regulatory milestone with a deadline and related tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Milestone {
    pub id: String,
    pub name: String,
    pub deadline: NaiveDate,
    pub status: MilestoneStatus,
    pub description: String,
    pub related_task_ids: Vec<String>,
    /// Regulation this milestone belongs to (e.g. "EU AI Act").
    #[serde(default)]
    pub regulation: Option<String>,
}

impl Milestone {
    pub fn new(id: impl Into<String>, name: impl Into<String>, deadline: NaiveDate) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            deadline,
            status: MilestoneStatus::Upcoming,
            description: String::new(),
            related_task_ids: Vec::new(),
            regulation: None,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn with_regulation(mut self, reg: impl Into<String>) -> Self {
        self.regulation = Some(reg.into());
        self
    }

    pub fn with_task(mut self, task_id: impl Into<String>) -> Self {
        self.related_task_ids.push(task_id.into());
        self
    }

    /// Days remaining until the deadline from the given date. Negative means overdue.
    pub fn days_remaining(&self, as_of: NaiveDate) -> i64 {
        (self.deadline - as_of).num_days()
    }

    /// Whether this milestone is at risk (deadline within `warning_days` and not achieved).
    pub fn is_at_risk(&self, as_of: NaiveDate, warning_days: i64) -> bool {
        self.status != MilestoneStatus::Achieved
            && self.days_remaining(as_of) >= 0
            && self.days_remaining(as_of) <= warning_days
    }
}

// ─── Status Report ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusReport {
    pub milestones: Vec<Milestone>,
    pub on_track_count: usize,
    pub at_risk_count: usize,
    pub missed_count: usize,
    pub achieved_count: usize,
    pub report_date: NaiveDate,
    pub next_deadline: Option<NaiveDate>,
}

// ─── Milestone Tracker ──────────────────────────────────────────────────────

/// Tracks regulatory milestones and computes status relative to task progress.
pub struct MilestoneTracker {
    pub milestones: Vec<Milestone>,
    /// Number of days before a deadline at which a milestone becomes "at risk".
    pub warning_window_days: i64,
}

impl MilestoneTracker {
    pub fn new() -> Self {
        Self { milestones: Vec::new(), warning_window_days: 30 }
    }

    pub fn with_warning_window(mut self, days: i64) -> Self {
        self.warning_window_days = days;
        self
    }

    pub fn add_milestone(&mut self, milestone: Milestone) {
        self.milestones.push(milestone);
        self.milestones.sort_by_key(|m| m.deadline);
    }

    /// Populate with standard EU AI Act milestones.
    pub fn add_eu_ai_act_milestones(&mut self) {
        let milestones = [
            ("eu-aia-prohibited", "EU AI Act: Prohibited AI Practices", "2025-02-02",
             "Ban on prohibited AI practices takes effect (Art. 5)"),
            ("eu-aia-gpai-codes", "EU AI Act: GPAI Codes of Practice", "2025-05-02",
             "Codes of practice for general-purpose AI models due"),
            ("eu-aia-governance", "EU AI Act: Governance Structure", "2025-08-02",
             "AI Office and governance structures operational"),
            ("eu-aia-gpai-obligations", "EU AI Act: GPAI Obligations", "2026-08-02",
             "Obligations for GPAI model providers apply (Ch. V)"),
            ("eu-aia-high-risk", "EU AI Act: High-Risk AI Systems", "2027-08-02",
             "Full obligations for high-risk AI systems (Annex III)"),
            ("eu-aia-annex-i", "EU AI Act: Annex I Systems", "2027-08-02",
             "Obligations for AI systems in Annex I EU legislation"),
        ];

        for (id, name, date_str, desc) in milestones {
            let deadline = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
                .unwrap_or_else(|_| NaiveDate::from_ymd_opt(2027, 8, 2).unwrap());
            self.add_milestone(
                Milestone::new(id, name, deadline)
                    .with_description(desc)
                    .with_regulation("EU AI Act"),
            );
        }
    }

    /// Update milestone statuses based on associated task progress.
    /// A milestone is:
    /// - `Achieved` if all related tasks are `Completed`.
    /// - `InProgress` if any related task is `InProgress`.
    /// - `Missed` if deadline has passed and not all tasks are complete.
    /// - `Upcoming` otherwise.
    pub fn update_statuses(
        &mut self,
        as_of: NaiveDate,
        task_statuses: &std::collections::HashMap<String, TaskStatus>,
    ) {
        for milestone in &mut self.milestones {
            if milestone.related_task_ids.is_empty() {
                // No associated tasks — status based on date only
                if milestone.status == MilestoneStatus::Achieved {
                    continue;
                }
                if milestone.deadline < as_of {
                    milestone.status = MilestoneStatus::Missed;
                } else {
                    milestone.status = MilestoneStatus::Upcoming;
                }
                continue;
            }

            let all_complete = milestone.related_task_ids.iter().all(|tid| {
                task_statuses.get(tid).copied() == Some(TaskStatus::Completed)
            });

            let any_in_progress = milestone.related_task_ids.iter().any(|tid| {
                task_statuses.get(tid).copied() == Some(TaskStatus::InProgress)
            });

            if all_complete {
                milestone.status = MilestoneStatus::Achieved;
            } else if milestone.deadline < as_of {
                milestone.status = MilestoneStatus::Missed;
            } else if any_in_progress {
                milestone.status = MilestoneStatus::InProgress;
            } else {
                milestone.status = MilestoneStatus::Upcoming;
            }
        }
    }

    /// Generate a status report as of the given date.
    pub fn status_report(&self, as_of: NaiveDate) -> StatusReport {
        let mut on_track = 0;
        let mut at_risk = 0;
        let mut missed = 0;
        let mut achieved = 0;

        for m in &self.milestones {
            match m.status {
                MilestoneStatus::Achieved => {
                    achieved += 1;
                    on_track += 1;
                }
                MilestoneStatus::Missed => missed += 1,
                MilestoneStatus::InProgress | MilestoneStatus::Upcoming => {
                    if m.deadline < as_of {
                        at_risk += 1;
                    } else if m.is_at_risk(as_of, self.warning_window_days) {
                        at_risk += 1;
                    } else {
                        on_track += 1;
                    }
                }
            }
        }

        let next_deadline = self.milestones.iter()
            .filter(|m| m.status != MilestoneStatus::Achieved && m.deadline >= as_of)
            .map(|m| m.deadline)
            .min();

        StatusReport {
            milestones: self.milestones.clone(),
            on_track_count: on_track,
            at_risk_count: at_risk,
            missed_count: missed,
            achieved_count: achieved,
            report_date: as_of,
            next_deadline,
        }
    }

    /// Return milestones whose deadlines fall within the next `days` from `as_of`.
    pub fn upcoming_within(&self, as_of: NaiveDate, days: i64) -> Vec<&Milestone> {
        self.milestones
            .iter()
            .filter(|m| {
                m.status != MilestoneStatus::Achieved
                    && m.deadline >= as_of
                    && m.days_remaining(as_of) <= days
            })
            .collect()
    }

    /// Return all milestones that have been missed as of the given date.
    pub fn missed_milestones(&self, as_of: NaiveDate) -> Vec<&Milestone> {
        self.milestones
            .iter()
            .filter(|m| m.status == MilestoneStatus::Missed || (m.deadline < as_of && m.status != MilestoneStatus::Achieved))
            .collect()
    }
}

impl Default for MilestoneTracker {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn test_milestone_days_remaining() {
        let m = Milestone::new("m1", "Test", date(2025, 6, 15));
        assert_eq!(m.days_remaining(date(2025, 6, 10)), 5);
        assert_eq!(m.days_remaining(date(2025, 6, 20)), -5);
    }

    #[test]
    fn test_milestone_at_risk() {
        let m = Milestone::new("m1", "Test", date(2025, 6, 15));
        assert!(m.is_at_risk(date(2025, 6, 10), 30));
        assert!(!m.is_at_risk(date(2025, 1, 1), 30));
    }

    #[test]
    fn test_add_eu_ai_act_milestones() {
        let mut tracker = MilestoneTracker::new();
        tracker.add_eu_ai_act_milestones();
        assert!(tracker.milestones.len() >= 6);
        assert!(tracker.milestones.iter().all(|m| m.regulation == Some("EU AI Act".into())));
        // Should be sorted by deadline
        for w in tracker.milestones.windows(2) {
            assert!(w[0].deadline <= w[1].deadline);
        }
    }

    #[test]
    fn test_update_statuses_all_complete() {
        let mut tracker = MilestoneTracker::new();
        tracker.add_milestone(
            Milestone::new("m1", "Milestone", date(2025, 12, 31))
                .with_task("t1")
                .with_task("t2"),
        );

        let mut task_statuses = HashMap::new();
        task_statuses.insert("t1".into(), TaskStatus::Completed);
        task_statuses.insert("t2".into(), TaskStatus::Completed);

        tracker.update_statuses(date(2025, 6, 1), &task_statuses);
        assert_eq!(tracker.milestones[0].status, MilestoneStatus::Achieved);
    }

    #[test]
    fn test_update_statuses_missed() {
        let mut tracker = MilestoneTracker::new();
        tracker.add_milestone(
            Milestone::new("m1", "Milestone", date(2025, 1, 1))
                .with_task("t1"),
        );

        let mut task_statuses = HashMap::new();
        task_statuses.insert("t1".into(), TaskStatus::NotStarted);

        tracker.update_statuses(date(2025, 6, 1), &task_statuses);
        assert_eq!(tracker.milestones[0].status, MilestoneStatus::Missed);
    }

    #[test]
    fn test_update_statuses_in_progress() {
        let mut tracker = MilestoneTracker::new();
        tracker.add_milestone(
            Milestone::new("m1", "Milestone", date(2025, 12, 31))
                .with_task("t1"),
        );

        let mut task_statuses = HashMap::new();
        task_statuses.insert("t1".into(), TaskStatus::InProgress);

        tracker.update_statuses(date(2025, 6, 1), &task_statuses);
        assert_eq!(tracker.milestones[0].status, MilestoneStatus::InProgress);
    }

    #[test]
    fn test_status_report() {
        let mut tracker = MilestoneTracker::new();
        tracker.add_milestone(Milestone::new("m1", "Done", date(2025, 1, 1)));
        tracker.milestones[0].status = MilestoneStatus::Achieved;

        tracker.add_milestone(Milestone::new("m2", "Future", date(2026, 1, 1)));
        tracker.add_milestone(Milestone::new("m3", "Missed", date(2024, 1, 1)));

        let report = tracker.status_report(date(2025, 6, 1));
        assert_eq!(report.achieved_count, 1);
        assert_eq!(report.on_track_count, 2); // achieved + future
        assert_eq!(report.missed_count, 0);
        assert!(report.next_deadline.is_some());
    }

    #[test]
    fn test_upcoming_within() {
        let mut tracker = MilestoneTracker::new();
        tracker.add_milestone(Milestone::new("m1", "Soon", date(2025, 6, 10)));
        tracker.add_milestone(Milestone::new("m2", "Later", date(2026, 1, 1)));

        let upcoming = tracker.upcoming_within(date(2025, 6, 1), 30);
        assert_eq!(upcoming.len(), 1);
        assert_eq!(upcoming[0].id, "m1");
    }

    #[test]
    fn test_milestone_status_display() {
        assert_eq!(MilestoneStatus::Upcoming.to_string(), "Upcoming");
        assert_eq!(MilestoneStatus::Achieved.to_string(), "Achieved");
        assert_eq!(MilestoneStatus::Missed.to_string(), "Missed");
    }

    #[test]
    fn test_sorted_insertion() {
        let mut tracker = MilestoneTracker::new();
        tracker.add_milestone(Milestone::new("late", "Late", date(2027, 1, 1)));
        tracker.add_milestone(Milestone::new("early", "Early", date(2025, 1, 1)));
        assert_eq!(tracker.milestones[0].id, "early");
        assert_eq!(tracker.milestones[1].id, "late");
    }
}
