//! Schedule optimization for resolving polypharmacy conflicts.
//!
//! Provides constraint-satisfaction based schedule optimization that adjusts
//! medication administration times to maximize temporal separation between
//! interacting drugs while respecting prescribed dosing frequencies.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use guardpharma_conflict_detect::{ConfirmedConflict, ConflictSeverity};
use guardpharma_pk_model::{DrugDatabase, DrugPkEntry, OneCompartmentModel};
use guardpharma_types::DrugId;

/// Convert a conflict-detect DrugId to a types DrugId.
fn to_types_drug_id(id: &guardpharma_conflict_detect::DrugId) -> DrugId {
    DrugId::new(id.as_str())
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Time of day specification (24-hour clock).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct TimeOfDay {
    /// Hour (0–23).
    pub hour: u8,
    /// Minute (0–59).
    pub minute: u8,
}

impl TimeOfDay {
    /// Create a new time of day.
    pub fn new(hour: u8, minute: u8) -> Self {
        assert!(hour < 24, "Hour must be 0–23");
        assert!(minute < 60, "Minute must be 0–59");
        TimeOfDay { hour, minute }
    }

    /// Convert to fractional hours since midnight.
    pub fn to_hours(&self) -> f64 {
        self.hour as f64 + self.minute as f64 / 60.0
    }

    /// Create from fractional hours since midnight.
    pub fn from_hours(hours: f64) -> Self {
        let h = hours.rem_euclid(24.0);
        let hour = h.floor() as u8;
        let minute = ((h - hour as f64) * 60.0).round() as u8;
        TimeOfDay {
            hour: hour.min(23),
            minute: minute.min(59),
        }
    }

    /// Compute the forward distance in hours to another time (wrapping at 24h).
    pub fn hours_until(&self, other: &TimeOfDay) -> f64 {
        let diff = other.to_hours() - self.to_hours();
        if diff < 0.0 {
            diff + 24.0
        } else {
            diff
        }
    }

    /// Compute the minimum circular distance in hours.
    pub fn circular_distance(&self, other: &TimeOfDay) -> f64 {
        let fwd = self.hours_until(other);
        fwd.min(24.0 - fwd)
    }
}

impl fmt::Display for TimeOfDay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:02}:{:02}", self.hour, self.minute)
    }
}

impl Default for TimeOfDay {
    fn default() -> Self {
        TimeOfDay { hour: 8, minute: 0 }
    }
}

/// A flexibility window around a preferred administration time.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FlexibilityWindow {
    /// Earliest acceptable time.
    pub earliest: TimeOfDay,
    /// Latest acceptable time.
    pub latest: TimeOfDay,
    /// Preferred time.
    pub preferred: TimeOfDay,
}

impl FlexibilityWindow {
    /// Create a new flexibility window.
    pub fn new(earliest: TimeOfDay, latest: TimeOfDay, preferred: TimeOfDay) -> Self {
        FlexibilityWindow {
            earliest,
            latest,
            preferred,
        }
    }

    /// Create a window centered on the preferred time ± margin_minutes.
    pub fn symmetric(preferred: TimeOfDay, margin_minutes: u32) -> Self {
        let margin_hours = margin_minutes as f64 / 60.0;
        let early = (preferred.to_hours() - margin_hours).rem_euclid(24.0);
        let late = (preferred.to_hours() + margin_hours).rem_euclid(24.0);
        FlexibilityWindow {
            earliest: TimeOfDay::from_hours(early),
            latest: TimeOfDay::from_hours(late),
            preferred,
        }
    }

    /// Check if a time falls within this window.
    pub fn contains(&self, time: &TimeOfDay) -> bool {
        let t = time.to_hours();
        let e = self.earliest.to_hours();
        let l = self.latest.to_hours();
        if e <= l {
            t >= e && t <= l
        } else {
            // Wraps around midnight
            t >= e || t <= l
        }
    }

    /// Width of the window in hours.
    pub fn width_hours(&self) -> f64 {
        self.earliest.hours_until(&self.latest)
    }

    /// Deviation from preferred time in hours.
    pub fn deviation_from_preferred(&self, time: &TimeOfDay) -> f64 {
        self.preferred.circular_distance(time)
    }
}

impl Default for FlexibilityWindow {
    fn default() -> Self {
        FlexibilityWindow::symmetric(TimeOfDay::new(8, 0), 60)
    }
}

/// Dose schedule for a single medication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoseSchedule {
    /// Drug identifier.
    pub drug_id: DrugId,
    /// Planned times of administration within a 24-hour day.
    pub times_of_day: Vec<TimeOfDay>,
    /// Interval between doses in hours.
    pub interval_hours: f64,
    /// Flexibility window around each administration time (in minutes).
    pub flexibility_window_minutes: u32,
    /// Flexibility windows for each dose time.
    pub flexibility_windows: Vec<FlexibilityWindow>,
}

impl DoseSchedule {
    /// Create a new dose schedule with evenly-spaced times.
    pub fn new(drug_id: DrugId, doses_per_day: u32, start_hour: u8) -> Self {
        let interval = if doses_per_day > 0 {
            24.0 / doses_per_day as f64
        } else {
            24.0
        };
        let mut times = Vec::new();
        for i in 0..doses_per_day {
            let h = (start_hour as f64 + i as f64 * interval).rem_euclid(24.0);
            times.push(TimeOfDay::from_hours(h));
        }
        times.sort();
        let windows = times
            .iter()
            .map(|t| FlexibilityWindow::symmetric(*t, 60))
            .collect();
        DoseSchedule {
            drug_id,
            times_of_day: times,
            interval_hours: interval,
            flexibility_window_minutes: 60,
            flexibility_windows: windows,
        }
    }

    /// Create with specific times.
    pub fn with_times(drug_id: DrugId, times: Vec<TimeOfDay>) -> Self {
        let interval = if times.len() > 1 {
            24.0 / times.len() as f64
        } else {
            24.0
        };
        let windows = times
            .iter()
            .map(|t| FlexibilityWindow::symmetric(*t, 60))
            .collect();
        DoseSchedule {
            drug_id,
            times_of_day: times,
            interval_hours: interval,
            flexibility_window_minutes: 60,
            flexibility_windows: windows,
        }
    }

    /// Number of doses per day.
    pub fn doses_per_day(&self) -> usize {
        self.times_of_day.len()
    }

    /// Get the minimum separation between this schedule and another.
    pub fn min_separation_hours(&self, other: &DoseSchedule) -> f64 {
        let mut min_sep = f64::MAX;
        for t1 in &self.times_of_day {
            for t2 in &other.times_of_day {
                let sep = t1.circular_distance(t2);
                if sep < min_sep {
                    min_sep = sep;
                }
            }
        }
        min_sep
    }
}

/// A constraint on the schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleConstraint {
    /// First drug in the constraint pair.
    pub drug_a: DrugId,
    /// Second drug in the constraint pair.
    pub drug_b: DrugId,
    /// Minimum required separation in hours.
    pub min_separation_hours: f64,
    /// Reason for this constraint.
    pub reason: String,
    /// Priority weight (higher = more important).
    pub priority: f64,
}

impl ScheduleConstraint {
    /// Create a new constraint.
    pub fn new(drug_a: DrugId, drug_b: DrugId, min_sep: f64, reason: &str) -> Self {
        ScheduleConstraint {
            drug_a,
            drug_b,
            min_separation_hours: min_sep,
            reason: reason.to_string(),
            priority: 1.0,
        }
    }

    /// Set priority weight.
    pub fn with_priority(mut self, priority: f64) -> Self {
        self.priority = priority;
        self
    }

    /// Check if a schedule assignment satisfies this constraint.
    pub fn is_satisfied(&self, schedules: &HashMap<DrugId, DoseSchedule>) -> bool {
        if let (Some(sa), Some(sb)) = (schedules.get(&self.drug_a), schedules.get(&self.drug_b)) {
            sa.min_separation_hours(sb) >= self.min_separation_hours
        } else {
            true
        }
    }

    /// Compute the violation amount (0 if satisfied, positive if violated).
    pub fn violation(&self, schedules: &HashMap<DrugId, DoseSchedule>) -> f64 {
        if let (Some(sa), Some(sb)) = (schedules.get(&self.drug_a), schedules.get(&self.drug_b)) {
            let actual_sep = sa.min_separation_hours(sb);
            (self.min_separation_hours - actual_sep).max(0.0)
        } else {
            0.0
        }
    }
}

/// An optimized schedule with rationale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedSchedule {
    /// Individual drug schedules.
    pub schedules: Vec<DoseSchedule>,
    /// Rationale for each scheduling decision.
    pub rationale: Vec<String>,
}

impl OptimizedSchedule {
    /// Create a new empty optimized schedule.
    pub fn new() -> Self {
        OptimizedSchedule {
            schedules: Vec::new(),
            rationale: Vec::new(),
        }
    }

    /// Add a drug schedule with rationale.
    pub fn add(&mut self, schedule: DoseSchedule, rationale: &str) {
        self.schedules.push(schedule);
        self.rationale.push(rationale.to_string());
    }

    /// Get schedule for a specific drug.
    pub fn get_schedule(&self, drug_id: &DrugId) -> Option<&DoseSchedule> {
        self.schedules.iter().find(|s| s.drug_id == *drug_id)
    }

    /// Compute the minimum separation between any two drug schedules.
    pub fn min_pairwise_separation(&self) -> f64 {
        let mut min_sep = f64::MAX;
        for i in 0..self.schedules.len() {
            for j in (i + 1)..self.schedules.len() {
                let sep = self.schedules[i].min_separation_hours(&self.schedules[j]);
                if sep < min_sep {
                    min_sep = sep;
                }
            }
        }
        if min_sep == f64::MAX {
            0.0
        } else {
            min_sep
        }
    }

    /// Build a lookup map from drug ID to schedule.
    pub fn to_map(&self) -> HashMap<DrugId, DoseSchedule> {
        self.schedules
            .iter()
            .map(|s| (s.drug_id.clone(), s.clone()))
            .collect()
    }
}

impl Default for OptimizedSchedule {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of schedule optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleResult {
    /// Original schedules before optimization.
    pub original_schedule: Vec<DoseSchedule>,
    /// Optimized schedules after optimization.
    pub optimized_schedule: OptimizedSchedule,
    /// Conflicts resolved by the schedule change.
    pub conflicts_resolved: Vec<String>,
    /// Conflicts remaining after the schedule change.
    pub conflicts_remaining: Vec<String>,
    /// Overall improvement score (0.0 = no improvement, 1.0 = perfect).
    pub improvement_score: f64,
    /// Constraints that were used.
    pub constraints_used: Vec<ScheduleConstraint>,
    /// Total constraint violation before optimization.
    pub violation_before: f64,
    /// Total constraint violation after optimization.
    pub violation_after: f64,
}

impl ScheduleResult {
    /// Whether the optimization resolved all conflicts.
    pub fn all_resolved(&self) -> bool {
        self.conflicts_remaining.is_empty()
    }

    /// Number of conflicts resolved.
    pub fn num_resolved(&self) -> usize {
        self.conflicts_resolved.len()
    }

    /// Number of conflicts remaining.
    pub fn num_remaining(&self) -> usize {
        self.conflicts_remaining.len()
    }
}

// ---------------------------------------------------------------------------
// Greedy Scheduler
// ---------------------------------------------------------------------------

/// Simple greedy scheduler that places drugs to maximize pairwise separation.
#[derive(Debug, Clone)]
pub struct GreedyScheduler {
    /// Waking hours start.
    pub wake_hour: u8,
    /// Waking hours end.
    pub sleep_hour: u8,
    /// Minimum acceptable separation in hours.
    pub min_separation: f64,
    /// Time resolution for placement attempts in minutes.
    pub resolution_minutes: u32,
}

impl GreedyScheduler {
    /// Create a new greedy scheduler with default waking hours (6am–10pm).
    pub fn new() -> Self {
        GreedyScheduler {
            wake_hour: 6,
            sleep_hour: 22,
            min_separation: 2.0,
            resolution_minutes: 30,
        }
    }

    /// Set waking hours.
    pub fn with_waking_hours(mut self, wake: u8, sleep: u8) -> Self {
        self.wake_hour = wake;
        self.sleep_hour = sleep;
        self
    }

    /// Set minimum separation requirement.
    pub fn with_min_separation(mut self, hours: f64) -> Self {
        self.min_separation = hours;
        self
    }

    /// Generate candidate times within waking hours at the given resolution.
    fn candidate_times(&self) -> Vec<TimeOfDay> {
        let mut times = Vec::new();
        let step = self.resolution_minutes;
        let mut current_min = self.wake_hour as u32 * 60;
        let end_min = self.sleep_hour as u32 * 60;
        while current_min <= end_min {
            let h = (current_min / 60) as u8;
            let m = (current_min % 60) as u8;
            if h < 24 {
                times.push(TimeOfDay::new(h, m));
            }
            current_min += step;
        }
        times
    }

    /// Find the best time slot for a single dose, maximizing minimum separation
    /// from already-placed doses.
    fn best_single_slot(
        &self,
        placed: &[TimeOfDay],
        candidates: &[TimeOfDay],
    ) -> TimeOfDay {
        if placed.is_empty() {
            // Place first dose in the morning
            return TimeOfDay::new(self.wake_hour, 0);
        }
        let mut best_time = candidates[0];
        let mut best_min_sep = 0.0_f64;
        for &candidate in candidates {
            let min_sep = placed
                .iter()
                .map(|p| p.circular_distance(&candidate))
                .fold(f64::MAX, f64::min);
            if min_sep > best_min_sep {
                best_min_sep = min_sep;
                best_time = candidate;
            }
        }
        best_time
    }

    /// Schedule multiple doses per day for a drug, maximizing internal separation.
    fn schedule_drug_doses(
        &self,
        doses_per_day: usize,
        placed_others: &[TimeOfDay],
        candidates: &[TimeOfDay],
    ) -> Vec<TimeOfDay> {
        let mut result = Vec::new();
        let mut all_placed: Vec<TimeOfDay> = placed_others.to_vec();

        for _ in 0..doses_per_day {
            let best = self.best_single_slot(&all_placed, candidates);
            result.push(best);
            all_placed.push(best);
        }

        result.sort();
        result
    }

    /// Run greedy scheduling for all drug schedules with constraints.
    pub fn optimize(
        &self,
        schedules: &[DoseSchedule],
        constraints: &[ScheduleConstraint],
    ) -> OptimizedSchedule {
        let candidates = self.candidate_times();
        let mut result = OptimizedSchedule::new();
        let mut all_placed: Vec<(DrugId, TimeOfDay)> = Vec::new();

        // Sort drugs by number of constraints (most constrained first).
        let mut drug_order: Vec<(usize, usize)> = schedules
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let num_constraints = constraints
                    .iter()
                    .filter(|c| c.drug_a == s.drug_id || c.drug_b == s.drug_id)
                    .count();
                (i, num_constraints)
            })
            .collect();
        drug_order.sort_by(|a, b| b.1.cmp(&a.1));

        for (idx, _constraint_count) in &drug_order {
            let sched = &schedules[*idx];
            let doses_per_day = sched.doses_per_day().max(1);

            // Collect placed times for interacting drugs only.
            let interacting_drugs: Vec<&DrugId> = constraints
                .iter()
                .filter_map(|c| {
                    if c.drug_a == sched.drug_id {
                        Some(&c.drug_b)
                    } else if c.drug_b == sched.drug_id {
                        Some(&c.drug_a)
                    } else {
                        None
                    }
                })
                .collect();

            let relevant_placed: Vec<TimeOfDay> = all_placed
                .iter()
                .filter(|(id, _)| interacting_drugs.contains(&id))
                .map(|(_, t)| *t)
                .collect();

            let times = self.schedule_drug_doses(doses_per_day, &relevant_placed, &candidates);
            let new_sched = DoseSchedule::with_times(sched.drug_id.clone(), times.clone());

            let rationale = if relevant_placed.is_empty() {
                format!(
                    "{}: scheduled {} dose(s) with default placement",
                    sched.drug_id,
                    doses_per_day
                )
            } else {
                let min_sep = relevant_placed
                    .iter()
                    .flat_map(|p| times.iter().map(move |t| p.circular_distance(t)))
                    .fold(f64::MAX, f64::min);
                format!(
                    "{}: scheduled {} dose(s), min separation {:.1}h from interacting drugs",
                    sched.drug_id, doses_per_day, min_sep
                )
            };

            for t in &times {
                all_placed.push((sched.drug_id.clone(), *t));
            }
            result.add(new_sched, &rationale);
        }

        result
    }
}

impl Default for GreedyScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Constraint Scheduler (CSP with backtracking)
// ---------------------------------------------------------------------------

/// Full CSP solver using backtracking with constraint propagation.
#[derive(Debug, Clone)]
pub struct ConstraintScheduler {
    /// Waking hours start.
    pub wake_hour: u8,
    /// Waking hours end.
    pub sleep_hour: u8,
    /// Time resolution for variable domains (minutes).
    pub resolution_minutes: u32,
    /// Maximum number of backtracks before giving up.
    pub max_backtracks: usize,
}

impl ConstraintScheduler {
    /// Create a new constraint scheduler.
    pub fn new() -> Self {
        ConstraintScheduler {
            wake_hour: 6,
            sleep_hour: 22,
            resolution_minutes: 30,
            max_backtracks: 10_000,
        }
    }

    /// Set waking hours.
    pub fn with_waking_hours(mut self, wake: u8, sleep: u8) -> Self {
        self.wake_hour = wake;
        self.sleep_hour = sleep;
        self
    }

    /// Set resolution.
    pub fn with_resolution(mut self, minutes: u32) -> Self {
        self.resolution_minutes = minutes;
        self
    }

    /// Set max backtracks.
    pub fn with_max_backtracks(mut self, max: usize) -> Self {
        self.max_backtracks = max;
        self
    }

    /// Generate the domain of possible times within waking hours.
    fn domain(&self) -> Vec<TimeOfDay> {
        let mut times = Vec::new();
        let step = self.resolution_minutes;
        let mut current_min = self.wake_hour as u32 * 60;
        let end_min = self.sleep_hour as u32 * 60;
        while current_min <= end_min {
            let h = (current_min / 60) as u8;
            let m = (current_min % 60) as u8;
            if h < 24 {
                times.push(TimeOfDay::new(h, m));
            }
            current_min += step;
        }
        times
    }

    /// Build CSP variables: one variable per (drug, dose_index).
    fn build_variables(
        &self,
        schedules: &[DoseSchedule],
    ) -> Vec<CspVariable> {
        let mut vars = Vec::new();
        for sched in schedules {
            let n = sched.doses_per_day().max(1);
            for dose_idx in 0..n {
                vars.push(CspVariable {
                    drug_id: sched.drug_id.clone(),
                    dose_index: dose_idx,
                    domain: self.domain(),
                    assigned: None,
                });
            }
        }
        vars
    }

    /// Check if an assignment is consistent with all constraints.
    fn is_consistent(
        assignment: &[Option<TimeOfDay>],
        variables: &[CspVariable],
        constraints: &[ScheduleConstraint],
        var_index: usize,
    ) -> bool {
        let var = &variables[var_index];
        let time = match assignment[var_index] {
            Some(t) => t,
            None => return true,
        };

        // Check inter-drug constraints.
        for c in constraints {
            let is_a = var.drug_id == c.drug_a;
            let is_b = var.drug_id == c.drug_b;
            if !is_a && !is_b {
                continue;
            }
            let other_drug = if is_a { &c.drug_b } else { &c.drug_a };

            for (j, other_var) in variables.iter().enumerate() {
                if j == var_index || other_var.drug_id != *other_drug {
                    continue;
                }
                if let Some(other_time) = assignment[j] {
                    let sep = time.circular_distance(&other_time);
                    if sep < c.min_separation_hours {
                        return false;
                    }
                }
            }
        }

        // Check intra-drug constraints: same drug doses must be spread out.
        for (j, other_var) in variables.iter().enumerate() {
            if j == var_index || other_var.drug_id != var.drug_id {
                continue;
            }
            if let Some(other_time) = assignment[j] {
                let sep = time.circular_distance(&other_time);
                // Same drug doses should be at least interval/2 apart.
                let n_doses = variables
                    .iter()
                    .filter(|v| v.drug_id == var.drug_id)
                    .count();
                let min_internal = if n_doses > 1 {
                    24.0 / n_doses as f64 * 0.5
                } else {
                    0.0
                };
                if sep < min_internal {
                    return false;
                }
            }
        }

        true
    }

    /// Forward checking: prune domains of unassigned variables.
    fn forward_check(
        domains: &mut Vec<Vec<TimeOfDay>>,
        assignment: &[Option<TimeOfDay>],
        variables: &[CspVariable],
        constraints: &[ScheduleConstraint],
        var_index: usize,
    ) -> bool {
        let var = &variables[var_index];
        let time = match assignment[var_index] {
            Some(t) => t,
            None => return true,
        };

        for (j, other_var) in variables.iter().enumerate() {
            if j == var_index || assignment[j].is_some() {
                continue;
            }

            // Find applicable constraints.
            let min_sep = constraints
                .iter()
                .filter(|c| {
                    (var.drug_id == c.drug_a && other_var.drug_id == c.drug_b)
                        || (var.drug_id == c.drug_b && other_var.drug_id == c.drug_a)
                })
                .map(|c| c.min_separation_hours)
                .fold(0.0_f64, f64::max);

            if min_sep > 0.0 {
                domains[j].retain(|&candidate| {
                    time.circular_distance(&candidate) >= min_sep
                });
                if domains[j].is_empty() {
                    return false;
                }
            }
        }

        true
    }

    /// Select the next unassigned variable (MRV heuristic).
    fn select_variable(
        assignment: &[Option<TimeOfDay>],
        domains: &[Vec<TimeOfDay>],
    ) -> Option<usize> {
        let mut best_idx = None;
        let mut best_domain_size = usize::MAX;

        for (i, val) in assignment.iter().enumerate() {
            if val.is_none() && domains[i].len() < best_domain_size {
                best_domain_size = domains[i].len();
                best_idx = Some(i);
            }
        }

        best_idx
    }

    /// Backtracking search with constraint propagation.
    fn backtrack(
        &self,
        assignment: &mut Vec<Option<TimeOfDay>>,
        domains: &Vec<Vec<TimeOfDay>>,
        variables: &[CspVariable],
        constraints: &[ScheduleConstraint],
        backtracks: &mut usize,
    ) -> bool {
        if *backtracks >= self.max_backtracks {
            return false;
        }

        let var_idx = match Self::select_variable(assignment, domains) {
            Some(idx) => idx,
            None => return true, // All assigned.
        };

        let domain_copy = domains[var_idx].clone();
        for &candidate in &domain_copy {
            assignment[var_idx] = Some(candidate);

            if Self::is_consistent(assignment, variables, constraints, var_idx) {
                let mut new_domains = domains.clone();
                if Self::forward_check(
                    &mut new_domains,
                    assignment,
                    variables,
                    constraints,
                    var_idx,
                ) {
                    if self.backtrack(assignment, &new_domains, variables, constraints, backtracks) {
                        return true;
                    }
                }
            }

            *backtracks += 1;
            assignment[var_idx] = None;
        }

        false
    }

    /// Compute the objective value: sum of minimum pairwise separations weighted
    /// by constraint priority.
    fn objective_value(
        assignment: &[Option<TimeOfDay>],
        variables: &[CspVariable],
        constraints: &[ScheduleConstraint],
    ) -> f64 {
        let mut total = 0.0;
        for c in constraints {
            let times_a: Vec<TimeOfDay> = variables
                .iter()
                .enumerate()
                .filter(|(_, v)| v.drug_id == c.drug_a)
                .filter_map(|(i, _)| assignment[i])
                .collect();
            let times_b: Vec<TimeOfDay> = variables
                .iter()
                .enumerate()
                .filter(|(_, v)| v.drug_id == c.drug_b)
                .filter_map(|(i, _)| assignment[i])
                .collect();

            let min_sep = times_a
                .iter()
                .flat_map(|a| times_b.iter().map(move |b| a.circular_distance(b)))
                .fold(f64::MAX, f64::min);

            if min_sep < f64::MAX {
                total += min_sep * c.priority;
            }
        }
        total
    }

    /// Run the CSP solver.
    pub fn optimize(
        &self,
        schedules: &[DoseSchedule],
        constraints: &[ScheduleConstraint],
    ) -> OptimizedSchedule {
        let variables = self.build_variables(schedules);
        let domains: Vec<Vec<TimeOfDay>> = variables.iter().map(|v| v.domain.clone()).collect();
        let mut assignment: Vec<Option<TimeOfDay>> = vec![None; variables.len()];
        let mut backtracks = 0;

        let solved = self.backtrack(
            &mut assignment,
            &domains,
            &variables,
            constraints,
            &mut backtracks,
        );

        let mut result = OptimizedSchedule::new();

        if solved {
            let mut drug_times: HashMap<DrugId, Vec<TimeOfDay>> = HashMap::new();
            for (i, var) in variables.iter().enumerate() {
                if let Some(time) = assignment[i] {
                    drug_times
                        .entry(var.drug_id.clone())
                        .or_default()
                        .push(time);
                }
            }

            for sched in schedules {
                if let Some(times) = drug_times.get(&sched.drug_id) {
                    let mut sorted_times = times.clone();
                    sorted_times.sort();
                    let new_sched = DoseSchedule::with_times(sched.drug_id.clone(), sorted_times);
                    let rationale = format!(
                        "{}: CSP solution found after {} backtracks",
                        sched.drug_id, backtracks
                    );
                    result.add(new_sched, &rationale);
                }
            }
        } else {
            // Fall back to greedy if CSP fails.
            let greedy = GreedyScheduler::new()
                .with_waking_hours(self.wake_hour, self.sleep_hour);
            result = greedy.optimize(schedules, constraints);
            result
                .rationale
                .push(format!("CSP solver failed after {} backtracks, fell back to greedy", backtracks));
        }

        result
    }
}

impl Default for ConstraintScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal CSP variable representation.
#[derive(Debug, Clone)]
struct CspVariable {
    drug_id: DrugId,
    dose_index: usize,
    domain: Vec<TimeOfDay>,
    assigned: Option<TimeOfDay>,
}

// ---------------------------------------------------------------------------
// Schedule Optimizer (top-level)
// ---------------------------------------------------------------------------

/// Top-level schedule optimizer that selects between greedy and CSP approaches.
#[derive(Debug, Clone)]
pub struct ScheduleOptimizer {
    /// Use CSP solver when the number of constraint pairs <= this threshold.
    pub csp_threshold: usize,
    /// Greedy scheduler configuration.
    pub greedy: GreedyScheduler,
    /// Constraint scheduler configuration.
    pub csp: ConstraintScheduler,
    /// PK database for computing interaction-aware separations.
    pk_db: Option<DrugDatabase>,
}

impl ScheduleOptimizer {
    /// Create a new schedule optimizer.
    pub fn new() -> Self {
        ScheduleOptimizer {
            csp_threshold: 20,
            greedy: GreedyScheduler::new(),
            csp: ConstraintScheduler::new(),
            pk_db: None,
        }
    }

    /// Attach a PK database for PK-informed constraint generation.
    pub fn with_pk_database(mut self, db: DrugDatabase) -> Self {
        self.pk_db = Some(db);
        self
    }

    /// Set the CSP threshold.
    pub fn with_csp_threshold(mut self, threshold: usize) -> Self {
        self.csp_threshold = threshold;
        self
    }

    /// Extract scheduling constraints from confirmed conflicts.
    pub fn extract_constraints(
        &self,
        conflicts: &[ConfirmedConflict],
    ) -> Vec<ScheduleConstraint> {
        let mut constraints = Vec::new();

        for conflict in conflicts {
            let min_sep = match conflict.severity {
                ConflictSeverity::Critical => 8.0,
                ConflictSeverity::Major => 4.0,
                ConflictSeverity::Moderate => 2.0,
                ConflictSeverity::Minor => 1.0,
            };

            // Adjust based on PK data if available.
            let drug_a = conflict.drugs.first().map(to_types_drug_id).unwrap_or_else(|| DrugId::new(""));
            let drug_b = conflict.drugs.get(1).map(to_types_drug_id).unwrap_or_else(|| DrugId::new(""));
            let adjusted_sep = if let Some(ref db) = self.pk_db {
                self.pk_informed_separation(
                    &drug_a,
                    &drug_b,
                    min_sep,
                    db,
                )
            } else {
                min_sep
            };

            let priority = match conflict.severity {
                ConflictSeverity::Critical => 4.0,
                ConflictSeverity::Major => 3.0,
                ConflictSeverity::Moderate => 2.0,
                ConflictSeverity::Minor => 1.0,
            };

            constraints.push(
                ScheduleConstraint::new(
                    drug_a,
                    drug_b,
                    adjusted_sep,
                    &format!("Conflict: {:?}", conflict.interaction_type),
                )
                .with_priority(priority),
            );
        }

        constraints
    }

    /// Compute PK-informed separation based on half-lives and Tmax.
    fn pk_informed_separation(
        &self,
        drug_a: &DrugId,
        drug_b: &DrugId,
        base_sep: f64,
        db: &DrugDatabase,
    ) -> f64 {
        let tmax_a = db
            .get_drug(drug_a)
            .map(|e| e.tmax)
            .unwrap_or(1.5);

        let tmax_b = db
            .get_drug(drug_b)
            .map(|e| e.tmax)
            .unwrap_or(1.5);

        // Aim for separation such that one drug's peak doesn't overlap with
        // the other drug's absorption/peak phase.
        let pk_sep = tmax_a + tmax_b;
        base_sep.max(pk_sep).min(12.0)
    }

    /// Build default schedules from the conflict list.
    fn build_default_schedules(
        &self,
        conflicts: &[ConfirmedConflict],
    ) -> Vec<DoseSchedule> {
        let mut drug_ids: Vec<DrugId> = Vec::new();
        for c in conflicts {
            for d in &c.drugs {
                let did = to_types_drug_id(d);
                if !drug_ids.contains(&did) {
                    drug_ids.push(did);
                }
            }
        }
        drug_ids
            .into_iter()
            .map(|id| DoseSchedule::new(id, 1, 8))
            .collect()
    }

    /// Optimize schedules for the given conflicts and medications.
    pub fn optimize(
        &self,
        conflicts: &[ConfirmedConflict],
        schedules: &[DoseSchedule],
    ) -> ScheduleResult {
        let constraints = self.extract_constraints(conflicts);
        let input_schedules = if schedules.is_empty() {
            self.build_default_schedules(conflicts)
        } else {
            schedules.to_vec()
        };

        // Compute violation before optimization.
        let original_map: HashMap<DrugId, DoseSchedule> = input_schedules
            .iter()
            .map(|s| (s.drug_id.clone(), s.clone()))
            .collect();
        let violation_before: f64 = constraints
            .iter()
            .map(|c| c.violation(&original_map) * c.priority)
            .sum();

        // Choose solver based on problem size.
        let num_constraint_pairs = constraints.len();
        let optimized = if num_constraint_pairs <= self.csp_threshold {
            self.csp.optimize(&input_schedules, &constraints)
        } else {
            self.greedy.optimize(&input_schedules, &constraints)
        };

        // Compute violation after optimization.
        let opt_map = optimized.to_map();
        let violation_after: f64 = constraints
            .iter()
            .map(|c| c.violation(&opt_map) * c.priority)
            .sum();

        // Categorize conflicts.
        let mut resolved = Vec::new();
        let mut remaining = Vec::new();
        for c in &constraints {
            if c.is_satisfied(&opt_map) {
                resolved.push(format!(
                    "{} ↔ {}: separation met (≥{:.1}h)",
                    c.drug_a, c.drug_b, c.min_separation_hours
                ));
            } else {
                remaining.push(format!(
                    "{} ↔ {}: separation not met (need {:.1}h)",
                    c.drug_a, c.drug_b, c.min_separation_hours
                ));
            }
        }

        let improvement = if violation_before > 0.0 {
            1.0 - (violation_after / violation_before)
        } else {
            1.0
        };

        ScheduleResult {
            original_schedule: input_schedules,
            optimized_schedule: optimized,
            conflicts_resolved: resolved,
            conflicts_remaining: remaining,
            improvement_score: improvement.max(0.0).min(1.0),
            constraints_used: constraints,
            violation_before,
            violation_after,
        }
    }

    /// Check if schedule changes resolve the concentration overlap for a conflict.
    pub fn check_overlap_resolution(
        &self,
        conflict: &ConfirmedConflict,
        schedule_a: &DoseSchedule,
        schedule_b: &DoseSchedule,
        db: &DrugDatabase,
    ) -> bool {
        let drug_a_id = conflict.drugs.first().map(to_types_drug_id).unwrap_or_else(|| DrugId::new(""));
        let drug_b_id = conflict.drugs.get(1).map(to_types_drug_id).unwrap_or_else(|| DrugId::new(""));
        let entry_a = match db.get_drug(&drug_a_id) {
            Some(e) => e,
            None => return false,
        };
        let entry_b = match db.get_drug(&drug_b_id) {
            Some(e) => e,
            None => return false,
        };

        let tmax_a = entry_a.tmax;
        let tmax_b = entry_b.tmax;
        let half_life_a = entry_a.half_life;
        let half_life_b = entry_b.half_life;

        // Check if peak windows overlap after rescheduling.
        for ta in &schedule_a.times_of_day {
            for tb in &schedule_b.times_of_day {
                let sep = ta.circular_distance(tb);
                let peak_a_start = tmax_a * 0.5;
                let peak_a_end = tmax_a + half_life_a * 0.5;
                let peak_b_start = tmax_b * 0.5;
                let peak_b_end = tmax_b + half_life_b * 0.5;

                let a_window_end = peak_a_end;
                let b_window_start = sep + peak_b_start;

                // If drug B's peak starts after drug A's peak ends, no overlap.
                if b_window_start < a_window_end {
                    return false;
                }
            }
        }

        true
    }
}

impl Default for ScheduleOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use guardpharma_conflict_detect::{ConflictType, InteractionMechanism};
    use guardpharma_types::{CypEnzyme, InhibitionType, Severity};

    fn make_conflict(a: &str, b: &str, severity: Severity) -> ConfirmedConflict {
        ConfirmedConflict::new(
            DrugId::new(a),
            DrugId::new(b),
            ConflictType::CypInhibition {
                enzyme: CypEnzyme::CYP3A4,
                inhibition_type: InhibitionType::Competitive,
            },
            severity,
        )
    }

    #[test]
    fn test_time_of_day_basic() {
        let t = TimeOfDay::new(14, 30);
        assert_eq!(t.to_hours(), 14.5);
        assert_eq!(t.to_string(), "14:30");
    }

    #[test]
    fn test_time_of_day_from_hours() {
        let t = TimeOfDay::from_hours(14.5);
        assert_eq!(t.hour, 14);
        assert_eq!(t.minute, 30);
    }

    #[test]
    fn test_time_circular_distance() {
        let a = TimeOfDay::new(2, 0);
        let b = TimeOfDay::new(22, 0);
        assert!((a.circular_distance(&b) - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_flexibility_window() {
        let pref = TimeOfDay::new(8, 0);
        let w = FlexibilityWindow::symmetric(pref, 60);
        assert!(w.contains(&TimeOfDay::new(8, 30)));
        assert!(w.contains(&TimeOfDay::new(7, 30)));
        assert!(!w.contains(&TimeOfDay::new(6, 0)));
    }

    #[test]
    fn test_dose_schedule_separation() {
        let s1 = DoseSchedule::with_times(DrugId::new("a"), vec![TimeOfDay::new(8, 0)]);
        let s2 = DoseSchedule::with_times(DrugId::new("b"), vec![TimeOfDay::new(14, 0)]);
        assert!((s1.min_separation_hours(&s2) - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_schedule_constraint_satisfaction() {
        let c = ScheduleConstraint::new(
            DrugId::new("a"),
            DrugId::new("b"),
            4.0,
            "test",
        );
        let mut map = HashMap::new();
        map.insert(
            DrugId::new("a"),
            DoseSchedule::with_times(DrugId::new("a"), vec![TimeOfDay::new(8, 0)]),
        );
        map.insert(
            DrugId::new("b"),
            DoseSchedule::with_times(DrugId::new("b"), vec![TimeOfDay::new(14, 0)]),
        );
        assert!(c.is_satisfied(&map));
    }

    #[test]
    fn test_schedule_constraint_violation() {
        let c = ScheduleConstraint::new(
            DrugId::new("a"),
            DrugId::new("b"),
            4.0,
            "test",
        );
        let mut map = HashMap::new();
        map.insert(
            DrugId::new("a"),
            DoseSchedule::with_times(DrugId::new("a"), vec![TimeOfDay::new(8, 0)]),
        );
        map.insert(
            DrugId::new("b"),
            DoseSchedule::with_times(DrugId::new("b"), vec![TimeOfDay::new(9, 0)]),
        );
        assert!(!c.is_satisfied(&map));
        assert!(c.violation(&map) > 0.0);
    }

    #[test]
    fn test_greedy_scheduler_basic() {
        let scheduler = GreedyScheduler::new();
        let schedules = vec![
            DoseSchedule::new(DrugId::new("a"), 1, 8),
            DoseSchedule::new(DrugId::new("b"), 1, 8),
        ];
        let constraints = vec![ScheduleConstraint::new(
            DrugId::new("a"),
            DrugId::new("b"),
            4.0,
            "test",
        )];
        let result = scheduler.optimize(&schedules, &constraints);
        assert_eq!(result.schedules.len(), 2);
    }

    #[test]
    fn test_csp_scheduler_two_drugs() {
        let scheduler = ConstraintScheduler::new().with_resolution(60);
        let schedules = vec![
            DoseSchedule::new(DrugId::new("a"), 1, 8),
            DoseSchedule::new(DrugId::new("b"), 1, 8),
        ];
        let constraints = vec![ScheduleConstraint::new(
            DrugId::new("a"),
            DrugId::new("b"),
            4.0,
            "test",
        )];
        let result = scheduler.optimize(&schedules, &constraints);
        assert_eq!(result.schedules.len(), 2);
        let map = result.to_map();
        let sep = map[&DrugId::new("a")].min_separation_hours(&map[&DrugId::new("b")]);
        assert!(sep >= 4.0, "Expected separation >= 4.0, got {}", sep);
    }

    #[test]
    fn test_schedule_optimizer_full() {
        let optimizer = ScheduleOptimizer::new();
        let conflicts = vec![make_conflict("drug_a", "drug_b", Severity::Major)];
        let schedules = vec![
            DoseSchedule::new(DrugId::new("drug_a"), 1, 8),
            DoseSchedule::new(DrugId::new("drug_b"), 1, 8),
        ];
        let result = optimizer.optimize(&conflicts, &schedules);
        assert!(result.improvement_score >= 0.0);
    }

    #[test]
    fn test_schedule_optimizer_three_drugs() {
        let optimizer = ScheduleOptimizer::new();
        let conflicts = vec![
            make_conflict("a", "b", Severity::Major),
            make_conflict("b", "c", Severity::Moderate),
        ];
        let schedules = vec![
            DoseSchedule::new(DrugId::new("a"), 1, 8),
            DoseSchedule::new(DrugId::new("b"), 2, 8),
            DoseSchedule::new(DrugId::new("c"), 1, 8),
        ];
        let result = optimizer.optimize(&conflicts, &schedules);
        assert!(!result.optimized_schedule.schedules.is_empty());
    }

    #[test]
    fn test_optimized_schedule_min_separation() {
        let mut opt = OptimizedSchedule::new();
        opt.add(
            DoseSchedule::with_times(DrugId::new("a"), vec![TimeOfDay::new(8, 0)]),
            "test",
        );
        opt.add(
            DoseSchedule::with_times(DrugId::new("b"), vec![TimeOfDay::new(14, 0)]),
            "test",
        );
        let sep = opt.min_pairwise_separation();
        assert!((sep - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_extract_constraints_from_conflicts() {
        let optimizer = ScheduleOptimizer::new();
        let conflicts = vec![
            make_conflict("a", "b", Severity::Major),
            make_conflict("c", "d", Severity::Minor),
        ];
        let constraints = optimizer.extract_constraints(&conflicts);
        assert_eq!(constraints.len(), 2);
        assert!(constraints[0].min_separation_hours > constraints[1].min_separation_hours);
    }

    #[test]
    fn test_multiple_doses_per_day() {
        let scheduler = GreedyScheduler::new();
        let schedules = vec![
            DoseSchedule::new(DrugId::new("a"), 3, 8),
            DoseSchedule::new(DrugId::new("b"), 2, 8),
        ];
        let constraints = vec![ScheduleConstraint::new(
            DrugId::new("a"),
            DrugId::new("b"),
            2.0,
            "test",
        )];
        let result = scheduler.optimize(&schedules, &constraints);
        let sched_a = result.get_schedule(&DrugId::new("a")).unwrap();
        assert_eq!(sched_a.doses_per_day(), 3);
    }
}
