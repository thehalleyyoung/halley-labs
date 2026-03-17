//! Temporal spacing recommendations for drug administration.
//!
//! Uses pharmacokinetic profiles to recommend optimal temporal separation
//! between interacting drugs, maximising the window where peak plasma
//! concentrations do not overlap.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::{
    ConfirmedConflict, ConflictSeverity, DrugId, InteractionType, PkDatabase, PkProfile,
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Temporal constraint between two drugs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConstraint {
    pub drug_a: DrugId,
    pub drug_b: DrugId,
    pub min_separation_hours: f64,
    pub ideal_separation_hours: f64,
    pub reason: String,
}

impl TemporalConstraint {
    pub fn new(drug_a: DrugId, drug_b: DrugId, min_sep: f64, ideal_sep: f64, reason: &str) -> Self {
        TemporalConstraint {
            drug_a,
            drug_b,
            min_separation_hours: min_sep,
            ideal_separation_hours: ideal_sep,
            reason: reason.to_string(),
        }
    }

    /// Whether a given separation satisfies the minimum.
    pub fn is_satisfied(&self, separation_hours: f64) -> bool {
        separation_hours >= self.min_separation_hours
    }

    /// How much of the ideal separation is achieved.
    pub fn satisfaction_ratio(&self, separation_hours: f64) -> f64 {
        if self.ideal_separation_hours <= 0.0 {
            return 1.0;
        }
        (separation_hours / self.ideal_separation_hours).clamp(0.0, 1.0)
    }
}

/// A flexibility window around a recommended time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlexibilityWindow {
    pub earliest_hour: f64,
    pub latest_hour: f64,
    pub preferred_hour: f64,
}

impl FlexibilityWindow {
    pub fn new(preferred: f64, margin_hours: f64) -> Self {
        FlexibilityWindow {
            earliest_hour: (preferred - margin_hours).rem_euclid(24.0),
            latest_hour: (preferred + margin_hours).rem_euclid(24.0),
            preferred_hour: preferred.rem_euclid(24.0),
        }
    }

    /// Width of the window in hours.
    pub fn width_hours(&self) -> f64 {
        let diff = self.latest_hour - self.earliest_hour;
        if diff < 0.0 {
            diff + 24.0
        } else {
            diff
        }
    }

    /// Whether a time (hour of day) falls within this window.
    pub fn contains(&self, hour: f64) -> bool {
        let h = hour.rem_euclid(24.0);
        if self.earliest_hour <= self.latest_hour {
            h >= self.earliest_hour && h <= self.latest_hour
        } else {
            h >= self.earliest_hour || h <= self.latest_hour
        }
    }
}

/// Recommended timing for a single drug.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugTiming {
    pub drug_id: DrugId,
    pub recommended_hours: Vec<f64>,
    pub flexibility: Vec<FlexibilityWindow>,
    pub notes: Vec<String>,
}

/// Overall temporal recommendation for a set of drugs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalRecommendation {
    pub drug_timings: Vec<DrugTiming>,
    pub constraints: Vec<TemporalConstraint>,
    pub overall_feasibility: f64,
    pub schedule_description: String,
    pub warnings: Vec<String>,
}

impl TemporalRecommendation {
    /// Get timing for a specific drug.
    pub fn timing_for(&self, drug_id: &DrugId) -> Option<&DrugTiming> {
        self.drug_timings.iter().find(|t| t.drug_id == *drug_id)
    }

    /// Minimum separation achieved between any interacting pair.
    pub fn min_achieved_separation(&self) -> f64 {
        let mut min_sep = f64::MAX;
        for c in &self.constraints {
            let time_a = self
                .drug_timings
                .iter()
                .find(|t| t.drug_id == c.drug_a)
                .and_then(|t| t.recommended_hours.first().copied());
            let time_b = self
                .drug_timings
                .iter()
                .find(|t| t.drug_id == c.drug_b)
                .and_then(|t| t.recommended_hours.first().copied());
            if let (Some(a), Some(b)) = (time_a, time_b) {
                let sep = circular_distance(a, b);
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
}

/// Simulated concentration time-course for evaluating temporal separation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatedSchedule {
    pub drug_id: DrugId,
    pub admin_times: Vec<f64>,
    pub concentration_at_times: Vec<(f64, f64)>,
    pub peak_time_h: f64,
    pub peak_concentration: f64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Circular distance on a 24-hour clock.
fn circular_distance(a: f64, b: f64) -> f64 {
    let diff = (a.rem_euclid(24.0) - b.rem_euclid(24.0)).abs();
    diff.min(24.0 - diff)
}

/// Compute the ideal separation based on PK parameters of two drugs.
fn pk_ideal_separation(pk_a: &PkProfile, pk_b: &PkProfile) -> f64 {
    // Goal: separate peaks. The ideal separation should ensure drug A's
    // concentration has fallen significantly before drug B reaches its peak,
    // and vice versa.
    let peak_window_a = pk_a.tmax_hours + pk_a.half_life_hours * 0.5;
    let peak_window_b = pk_b.tmax_hours + pk_b.half_life_hours * 0.5;
    // The minimum separation so that administration of B happens after A's
    // peak window closes, and B hasn't peaked by the time A's next dose.
    let sep = (peak_window_a + pk_b.tmax_hours).min(12.0);
    sep.max(peak_window_b)
}

/// Compute a severity-based minimum separation when no PK data is available.
fn severity_based_separation(severity: &ConflictSeverity) -> f64 {
    match severity {
        ConflictSeverity::Critical => 8.0,
        ConflictSeverity::Major => 4.0,
        ConflictSeverity::Moderate => 2.0,
        ConflictSeverity::Minor => 1.0,
    }
}

// ---------------------------------------------------------------------------
// TemporalRecommender
// ---------------------------------------------------------------------------

/// Engine that recommends optimal administration times for interacting drugs.
#[derive(Debug, Clone)]
pub struct TemporalRecommender {
    pk_db: PkDatabase,
    /// Waking hours (start, end) in fractional hours.
    wake_window: (f64, f64),
    /// Resolution for candidate time slots (hours).
    resolution_h: f64,
    /// Default flexibility margin in hours.
    flexibility_margin_h: f64,
}

impl TemporalRecommender {
    pub fn new(pk_db: PkDatabase) -> Self {
        TemporalRecommender {
            pk_db,
            wake_window: (6.0, 22.0),
            resolution_h: 0.5,
            flexibility_margin_h: 1.0,
        }
    }

    pub fn with_wake_window(mut self, start: f64, end: f64) -> Self {
        self.wake_window = (start.clamp(0.0, 23.0), end.clamp(1.0, 24.0));
        self
    }

    pub fn with_resolution(mut self, hours: f64) -> Self {
        self.resolution_h = hours.max(0.25);
        self
    }

    /// Recommend timing for drugs involved in the given conflicts.
    pub fn recommend_timing(
        &self,
        conflicts: &[ConfirmedConflict],
    ) -> TemporalRecommendation {
        // Collect unique drug ids.
        let mut drug_ids: Vec<DrugId> = Vec::new();
        for c in conflicts {
            if !drug_ids.contains(&c.drug_a) {
                drug_ids.push(c.drug_a.clone());
            }
            if !drug_ids.contains(&c.drug_b) {
                drug_ids.push(c.drug_b.clone());
            }
        }

        // Build constraints from conflicts + PK data.
        let constraints = self.build_constraints(conflicts);

        // Generate candidate time slots within waking hours.
        let candidates = self.candidate_times();

        // Greedy placement: place each drug to maximise minimum separation
        // from already-placed interacting drugs.
        let mut placements: HashMap<DrugId, f64> = HashMap::new();

        // Sort drugs by number of constraints (most constrained first).
        let mut ordered: Vec<(DrugId, usize)> = drug_ids
            .iter()
            .map(|id| {
                let n = constraints
                    .iter()
                    .filter(|c| c.drug_a == *id || c.drug_b == *id)
                    .count();
                (id.clone(), n)
            })
            .collect();
        ordered.sort_by(|a, b| b.1.cmp(&a.1));

        for (drug_id, _) in &ordered {
            // Collect placed times for interacting drugs.
            let interacting: Vec<f64> = constraints
                .iter()
                .filter_map(|c| {
                    if c.drug_a == *drug_id {
                        placements.get(&c.drug_b).copied()
                    } else if c.drug_b == *drug_id {
                        placements.get(&c.drug_a).copied()
                    } else {
                        None
                    }
                })
                .collect();

            let best_time = if interacting.is_empty() {
                // First drug: place at a reasonable morning time.
                self.wake_window.0 + 2.0
            } else {
                // Find candidate maximising minimum separation from interacting drugs.
                let mut best = candidates[0];
                let mut best_min_sep = 0.0_f64;
                for &cand in &candidates {
                    let min_sep = interacting
                        .iter()
                        .map(|&t| circular_distance(cand, t))
                        .fold(f64::MAX, f64::min);
                    if min_sep > best_min_sep {
                        best_min_sep = min_sep;
                        best = cand;
                    }
                }
                best
            };

            placements.insert(drug_id.clone(), best_time);
        }

        // Build recommendation.
        let mut drug_timings: Vec<DrugTiming> = Vec::new();
        let mut warnings: Vec<String> = Vec::new();

        for drug_id in &drug_ids {
            let time = placements.get(drug_id).copied().unwrap_or(8.0);
            let flex = FlexibilityWindow::new(time, self.flexibility_margin_h);
            let mut notes = Vec::new();

            if let Some(pk) = self.pk_db.get(drug_id) {
                notes.push(format!(
                    "t½ = {:.1}h, Tmax = {:.1}h",
                    pk.half_life_hours, pk.tmax_hours
                ));
            }

            drug_timings.push(DrugTiming {
                drug_id: drug_id.clone(),
                recommended_hours: vec![time],
                flexibility: vec![flex],
                notes,
            });
        }

        // Check constraint satisfaction.
        let mut feasibility_sum = 0.0;
        let constraint_count = constraints.len().max(1);
        for c in &constraints {
            let sep = match (placements.get(&c.drug_a), placements.get(&c.drug_b)) {
                (Some(&a), Some(&b)) => circular_distance(a, b),
                _ => 0.0,
            };
            let ratio = c.satisfaction_ratio(sep);
            feasibility_sum += ratio;

            if !c.is_satisfied(sep) {
                warnings.push(format!(
                    "{} ↔ {}: achieved {:.1}h separation, need {:.1}h minimum",
                    c.drug_a, c.drug_b, sep, c.min_separation_hours
                ));
            }
        }
        let overall_feasibility = feasibility_sum / constraint_count as f64;

        let description = self.format_schedule(&drug_timings);

        TemporalRecommendation {
            drug_timings,
            constraints,
            overall_feasibility,
            schedule_description: description,
            warnings,
        }
    }

    /// Build temporal constraints from conflicts using PK data where available.
    fn build_constraints(&self, conflicts: &[ConfirmedConflict]) -> Vec<TemporalConstraint> {
        let mut constraints = Vec::new();
        for conflict in conflicts {
            let pk_a = self.pk_db.get(&conflict.drug_a);
            let pk_b = self.pk_db.get(&conflict.drug_b);

            let (min_sep, ideal_sep) = match (pk_a, pk_b) {
                (Some(a), Some(b)) => {
                    let ideal = pk_ideal_separation(a, b);
                    let severity_min = severity_based_separation(&conflict.severity);
                    (severity_min.max(a.tmax_hours + b.tmax_hours), ideal)
                }
                _ => {
                    let s = severity_based_separation(&conflict.severity);
                    (s, s * 1.5)
                }
            };

            let reason = format!(
                "{} interaction between {} and {}",
                conflict.interaction_type, conflict.drug_a, conflict.drug_b
            );

            constraints.push(TemporalConstraint::new(
                conflict.drug_a.clone(),
                conflict.drug_b.clone(),
                min_sep.min(12.0),
                ideal_sep.min(12.0),
                &reason,
            ));
        }
        constraints
    }

    /// Generate candidate time slots within the waking window.
    fn candidate_times(&self) -> Vec<f64> {
        let mut times = Vec::new();
        let mut t = self.wake_window.0;
        while t <= self.wake_window.1 {
            times.push(t);
            t += self.resolution_h;
        }
        if times.is_empty() {
            times.push(8.0);
        }
        times
    }

    /// Simulate a separated schedule and return concentration profiles.
    pub fn simulate_separated_schedule(
        &self,
        drug_id: &DrugId,
        admin_times: &[f64],
    ) -> SimulatedSchedule {
        let pk = self.pk_db.get(drug_id);
        let ke = pk.map(|p| p.elimination_rate()).unwrap_or(0.1);
        let tmax = pk.map(|p| p.tmax_hours).unwrap_or(1.5);

        // Simple one-compartment oral model approximation.
        let ka = if tmax > 0.0 {
            // Approximate absorption rate: ka ≈ 2.5 / tmax for oral drugs.
            2.5 / tmax
        } else {
            1.0
        };

        let mut concentration_points: Vec<(f64, f64)> = Vec::new();
        let mut peak_conc = 0.0_f64;
        let mut peak_time = 0.0_f64;

        // Sample concentration over 24 hours.
        let dt = 0.25;
        let mut t = 0.0;
        while t <= 24.0 {
            let mut total_conc = 0.0;
            for &admin in admin_times {
                let elapsed = t - admin;
                if elapsed >= 0.0 {
                    // Bateman equation: C(t) = (F*D*ka / (Vd * (ka-ke))) * (e^(-ke*t) - e^(-ka*t))
                    let c = if (ka - ke).abs() > 1e-6 {
                        (ka / (ka - ke)) * ((-ke * elapsed).exp() - (-ka * elapsed).exp())
                    } else {
                        ka * elapsed * (-ke * elapsed).exp()
                    };
                    total_conc += c.max(0.0);
                }
            }
            concentration_points.push((t, total_conc));
            if total_conc > peak_conc {
                peak_conc = total_conc;
                peak_time = t;
            }
            t += dt;
        }

        SimulatedSchedule {
            drug_id: drug_id.clone(),
            admin_times: admin_times.to_vec(),
            concentration_at_times: concentration_points,
            peak_time_h: peak_time,
            peak_concentration: peak_conc,
        }
    }

    fn format_schedule(&self, timings: &[DrugTiming]) -> String {
        let mut parts: Vec<String> = timings
            .iter()
            .map(|t| {
                let hours_str: Vec<String> = t
                    .recommended_hours
                    .iter()
                    .map(|h| {
                        let hh = *h as u32;
                        let mm = ((*h - hh as f64) * 60.0).round() as u32;
                        format!("{:02}:{:02}", hh, mm)
                    })
                    .collect();
                format!("{}: {}", t.drug_id, hours_str.join(", "))
            })
            .collect();
        parts.sort();
        parts.join("; ")
    }
}

impl Default for TemporalRecommender {
    fn default() -> Self {
        Self::new(PkDatabase::new())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    fn make_pk_db() -> PkDatabase {
        PkDatabase::demo()
    }

    fn make_conflict(a: &str, b: &str, severity: ConflictSeverity) -> ConfirmedConflict {
        ConfirmedConflict::new(
            DrugId::new(a),
            DrugId::new(b),
            InteractionType::CypInhibition {
                enzyme: "CYP3A4".to_string(),
            },
            severity,
        )
    }

    #[test]
    fn test_circular_distance_basic() {
        assert!((circular_distance(8.0, 14.0) - 6.0).abs() < 0.01);
        assert!((circular_distance(2.0, 22.0) - 4.0).abs() < 0.01);
        assert!((circular_distance(0.0, 12.0) - 12.0).abs() < 0.01);
    }

    #[test]
    fn test_temporal_constraint_satisfaction() {
        let c = TemporalConstraint::new(
            DrugId::new("a"),
            DrugId::new("b"),
            4.0,
            6.0,
            "test",
        );
        assert!(c.is_satisfied(5.0));
        assert!(!c.is_satisfied(3.0));
        assert!((c.satisfaction_ratio(3.0) - 0.5).abs() < 0.01);
        assert!((c.satisfaction_ratio(6.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_flexibility_window_basic() {
        let w = FlexibilityWindow::new(8.0, 1.0);
        assert!(w.contains(8.0));
        assert!(w.contains(8.5));
        assert!(w.contains(7.5));
        assert!(!w.contains(6.0));
    }

    #[test]
    fn test_flexibility_window_midnight_wrap() {
        let w = FlexibilityWindow::new(23.5, 1.0);
        assert!(w.contains(23.5));
        assert!(w.contains(0.0));
        assert!(w.contains(0.4));
    }

    #[test]
    fn test_recommend_timing_two_drugs() {
        let pk_db = make_pk_db();
        let recommender = TemporalRecommender::new(pk_db);
        let conflict = make_conflict("atorvastatin", "omeprazole", ConflictSeverity::Moderate);
        let rec = recommender.recommend_timing(&[conflict]);
        assert_eq!(rec.drug_timings.len(), 2);
        assert!(rec.overall_feasibility > 0.0);

        let sep = rec.min_achieved_separation();
        assert!(sep >= 1.5, "Expected separation >= 1.5h, got {:.1}h", sep);
    }

    #[test]
    fn test_recommend_timing_three_drugs() {
        let pk_db = make_pk_db();
        let recommender = TemporalRecommender::new(pk_db);
        let conflicts = vec![
            make_conflict("atorvastatin", "omeprazole", ConflictSeverity::Moderate),
            make_conflict("omeprazole", "warfarin", ConflictSeverity::Major),
        ];
        let rec = recommender.recommend_timing(&conflicts);
        assert_eq!(rec.drug_timings.len(), 3);
        assert!(!rec.schedule_description.is_empty());
    }

    #[test]
    fn test_pk_ideal_separation() {
        let pk_a = PkProfile::new(DrugId::new("a"), 4.0, 1.5);
        let pk_b = PkProfile::new(DrugId::new("b"), 2.0, 1.0);
        let ideal = pk_ideal_separation(&pk_a, &pk_b);
        assert!(ideal >= 2.0, "Ideal separation should be >= 2h, got {:.1}h", ideal);
        assert!(ideal <= 12.0);
    }

    #[test]
    fn test_simulate_separated_schedule() {
        let pk_db = make_pk_db();
        let recommender = TemporalRecommender::new(pk_db);
        let sim = recommender.simulate_separated_schedule(
            &DrugId::new("atorvastatin"),
            &[8.0],
        );
        assert!(!sim.concentration_at_times.is_empty());
        assert!(sim.peak_concentration > 0.0);
        // Peak should be close to tmax (1.5h) after admin at 8.0.
        assert!(
            (sim.peak_time_h - 9.5).abs() < 2.0,
            "Peak at {:.1}h, expected near 9.5h",
            sim.peak_time_h
        );
    }

    #[test]
    fn test_simulate_multiple_doses() {
        let pk_db = make_pk_db();
        let recommender = TemporalRecommender::new(pk_db);
        let sim = recommender.simulate_separated_schedule(
            &DrugId::new("metoprolol"),
            &[8.0, 20.0],
        );
        assert!(sim.peak_concentration > 0.0);
        // With two doses the peak should be higher or at least match single dose.
        let sim_single = recommender.simulate_separated_schedule(
            &DrugId::new("metoprolol"),
            &[8.0],
        );
        assert!(sim.peak_concentration >= sim_single.peak_concentration * 0.9);
    }

    #[test]
    fn test_severity_based_separation_ordering() {
        assert!(
            severity_based_separation(&ConflictSeverity::Critical)
                > severity_based_separation(&ConflictSeverity::Major)
        );
        assert!(
            severity_based_separation(&ConflictSeverity::Major)
                > severity_based_separation(&ConflictSeverity::Moderate)
        );
        assert!(
            severity_based_separation(&ConflictSeverity::Moderate)
                > severity_based_separation(&ConflictSeverity::Minor)
        );
    }

    #[test]
    fn test_no_conflicts_empty_recommendation() {
        let recommender = TemporalRecommender::new(PkDatabase::new());
        let rec = recommender.recommend_timing(&[]);
        assert!(rec.drug_timings.is_empty());
        assert!(rec.constraints.is_empty());
    }

    #[test]
    fn test_recommendation_has_timing_for_each_drug() {
        let pk_db = make_pk_db();
        let recommender = TemporalRecommender::new(pk_db);
        let conflict = make_conflict("warfarin", "ibuprofen", ConflictSeverity::Major);
        let rec = recommender.recommend_timing(&[conflict]);
        assert!(rec.timing_for(&DrugId::new("warfarin")).is_some());
        assert!(rec.timing_for(&DrugId::new("ibuprofen")).is_some());
    }

    #[test]
    fn test_custom_wake_window() {
        let pk_db = make_pk_db();
        let recommender = TemporalRecommender::new(pk_db).with_wake_window(7.0, 21.0);
        let conflict = make_conflict("atorvastatin", "warfarin", ConflictSeverity::Moderate);
        let rec = recommender.recommend_timing(&[conflict]);
        for timing in &rec.drug_timings {
            for &h in &timing.recommended_hours {
                assert!(h >= 7.0 && h <= 21.0, "Time {:.1} outside wake window", h);
            }
        }
    }
}
