// Safety envelope types for the SafeStep deployment planner.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::constraint::ConstraintEvaluation;
use crate::graph::ClusterState;
use crate::identifiers::{ConstraintId, EnvelopeId, StateId};

// ─── EnvelopeMembership ─────────────────────────────────────────────────

/// Membership status of a state with respect to the safety envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnvelopeMembership {
    /// State is strictly inside the envelope: both forward and backward reachability hold.
    Inside,
    /// State is outside the envelope: forward or backward reachability fails.
    Outside,
    /// State is on the boundary of the envelope.
    Boundary,
    /// Point of no return: forward reachability holds but backward does not.
    PNR,
}

impl EnvelopeMembership {
    pub fn is_safe_for_deployment(&self) -> bool {
        matches!(self, Self::Inside | Self::Boundary)
    }

    pub fn is_pnr(&self) -> bool {
        matches!(self, Self::PNR)
    }

    pub fn allows_rollback(&self) -> bool {
        matches!(self, Self::Inside | Self::Boundary)
    }
}

impl fmt::Display for EnvelopeMembership {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inside => write!(f, "inside"),
            Self::Outside => write!(f, "outside"),
            Self::Boundary => write!(f, "boundary"),
            Self::PNR => write!(f, "PNR"),
        }
    }
}

// ─── ReachabilityResult ─────────────────────────────────────────────────

/// Forward/backward reachability information for a state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReachabilityResult {
    pub state: ClusterState,
    pub state_id: StateId,
    pub forward_reachable: bool,
    pub backward_reachable: bool,
    pub forward_distance: Option<usize>,
    pub backward_distance: Option<usize>,
    pub membership: EnvelopeMembership,
}

impl ReachabilityResult {
    pub fn new(state: ClusterState, forward: bool, backward: bool) -> Self {
        let state_id = state.state_id();
        let membership = match (forward, backward) {
            (true, true) => EnvelopeMembership::Inside,
            (true, false) => EnvelopeMembership::PNR,
            (false, true) => EnvelopeMembership::Outside,
            (false, false) => EnvelopeMembership::Outside,
        };
        Self {
            state,
            state_id,
            forward_reachable: forward,
            backward_reachable: backward,
            forward_distance: None,
            backward_distance: None,
            membership,
        }
    }

    pub fn with_forward_distance(mut self, dist: usize) -> Self {
        self.forward_distance = Some(dist);
        self
    }

    pub fn with_backward_distance(mut self, dist: usize) -> Self {
        self.backward_distance = Some(dist);
        self
    }

    pub fn total_distance(&self) -> Option<usize> {
        match (self.forward_distance, self.backward_distance) {
            (Some(f), Some(b)) => Some(f + b),
            _ => None,
        }
    }

    pub fn is_on_shortest_path(&self, optimal_length: usize) -> bool {
        self.total_distance()
            .map(|d| d == optimal_length)
            .unwrap_or(false)
    }
}

impl fmt::Display for ReachabilityResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.state, self.membership)?;
        if let Some(fd) = self.forward_distance {
            write!(f, " (fwd={})", fd)?;
        }
        if let Some(bd) = self.backward_distance {
            write!(f, " (bwd={})", bd)?;
        }
        Ok(())
    }
}

// ─── PointOfNoReturn ────────────────────────────────────────────────────

/// A point-of-no-return state with witness information explaining why rollback is blocked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointOfNoReturn {
    pub state: ClusterState,
    pub state_id: StateId,
    pub forward_distance_to_target: Option<usize>,
    pub witness: StuckWitness,
    pub severity: PNRSeverity,
    pub notes: Vec<String>,
}

impl PointOfNoReturn {
    pub fn new(state: ClusterState, witness: StuckWitness) -> Self {
        let state_id = state.state_id();
        let severity = if witness.blocking_constraints.is_empty() {
            PNRSeverity::Structural
        } else {
            PNRSeverity::ConstraintBlocked
        };
        Self {
            state,
            state_id,
            forward_distance_to_target: None,
            witness,
            severity,
            notes: Vec::new(),
        }
    }

    pub fn with_forward_distance(mut self, dist: usize) -> Self {
        self.forward_distance_to_target = Some(dist);
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }
}

impl fmt::Display for PointOfNoReturn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PNR {}: {}", self.state, self.severity)?;
        if let Some(dist) = self.forward_distance_to_target {
            write!(f, " ({} steps to target)", dist)?;
        }
        write!(f, " — {}", self.witness)
    }
}

/// Severity of a PNR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PNRSeverity {
    /// Structural: no backward path exists in the graph at all.
    Structural,
    /// Constraint-blocked: backward paths exist but all violate constraints.
    ConstraintBlocked,
    /// Resource-blocked: backward paths require more resources than available.
    ResourceBlocked,
}

impl fmt::Display for PNRSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Structural => write!(f, "structural"),
            Self::ConstraintBlocked => write!(f, "constraint-blocked"),
            Self::ResourceBlocked => write!(f, "resource-blocked"),
        }
    }
}

// ─── StuckWitness ───────────────────────────────────────────────────────

/// Witness explaining why a state is a PNR (why rollback is blocked).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StuckWitness {
    /// The minimal set of constraints blocking retreat.
    pub blocking_constraints: Vec<ConstraintId>,
    /// The services whose versions cannot be downgraded.
    pub irreversible_services: Vec<usize>,
    /// Human-readable explanation.
    pub explanation: String,
    /// Attempted rollback paths and why they fail.
    pub failed_paths: Vec<FailedPath>,
}

impl StuckWitness {
    pub fn structural(services: Vec<usize>, explanation: impl Into<String>) -> Self {
        Self {
            blocking_constraints: Vec::new(),
            irreversible_services: services,
            explanation: explanation.into(),
            failed_paths: Vec::new(),
        }
    }

    pub fn constraint_blocked(
        constraints: Vec<ConstraintId>,
        explanation: impl Into<String>,
    ) -> Self {
        Self {
            blocking_constraints: constraints,
            irreversible_services: Vec::new(),
            explanation: explanation.into(),
            failed_paths: Vec::new(),
        }
    }

    pub fn with_failed_path(mut self, path: FailedPath) -> Self {
        self.failed_paths.push(path);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.blocking_constraints.is_empty() && self.irreversible_services.is_empty()
    }
}

impl fmt::Display for StuckWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.explanation)?;
        if !self.blocking_constraints.is_empty() {
            write!(f, " (blocked by: ")?;
            for (i, c) in self.blocking_constraints.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", c)?;
            }
            write!(f, ")")?;
        }
        if !self.irreversible_services.is_empty() {
            write!(f, " (irreversible: {:?})", self.irreversible_services)?;
        }
        Ok(())
    }
}

/// A failed backward path from a PNR state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedPath {
    pub path_states: Vec<ClusterState>,
    pub failure_point: usize,
    pub violated_constraints: Vec<ConstraintId>,
    pub description: String,
}

impl FailedPath {
    pub fn new(
        states: Vec<ClusterState>,
        failure_point: usize,
        description: impl Into<String>,
    ) -> Self {
        Self {
            path_states: states,
            failure_point,
            violated_constraints: Vec::new(),
            description: description.into(),
        }
    }

    pub fn with_violated(mut self, constraint_id: ConstraintId) -> Self {
        self.violated_constraints.push(constraint_id);
        self
    }
}

impl fmt::Display for FailedPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Failed at step {} of {}: {}",
            self.failure_point,
            self.path_states.len(),
            self.description,
        )
    }
}

// ─── SafetyEnvelope ─────────────────────────────────────────────────────

/// The rollback safety envelope: set of states from which both forward
/// (to target) and backward (to start) reachability hold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyEnvelope {
    pub id: EnvelopeId,
    pub start_state: ClusterState,
    pub target_state: ClusterState,
    /// States classified by membership.
    pub memberships: Vec<(ClusterState, EnvelopeMembership)>,
    /// Points of no return with witness information.
    pub pnr_states: Vec<PointOfNoReturn>,
    /// Overall statistics.
    pub stats: EnvelopeStats,
}

impl SafetyEnvelope {
    pub fn new(start: ClusterState, target: ClusterState) -> Self {
        Self {
            id: EnvelopeId::generate(),
            start_state: start,
            target_state: target,
            memberships: Vec::new(),
            pnr_states: Vec::new(),
            stats: EnvelopeStats::default(),
        }
    }

    pub fn add_membership(&mut self, state: ClusterState, membership: EnvelopeMembership) {
        match membership {
            EnvelopeMembership::Inside => self.stats.inside_count += 1,
            EnvelopeMembership::Outside => self.stats.outside_count += 1,
            EnvelopeMembership::Boundary => self.stats.boundary_count += 1,
            EnvelopeMembership::PNR => self.stats.pnr_count += 1,
        }
        self.memberships.push((state, membership));
    }

    pub fn add_pnr(&mut self, pnr: PointOfNoReturn) {
        self.pnr_states.push(pnr);
    }

    pub fn query(&self, state: &ClusterState) -> EnvelopeMembership {
        self.memberships
            .iter()
            .find(|(s, _)| s == state)
            .map(|(_, m)| *m)
            .unwrap_or(EnvelopeMembership::Outside)
    }

    pub fn is_inside(&self, state: &ClusterState) -> bool {
        self.query(state) == EnvelopeMembership::Inside
    }

    pub fn is_pnr(&self, state: &ClusterState) -> bool {
        self.query(state) == EnvelopeMembership::PNR
    }

    pub fn envelope_states(&self) -> Vec<&ClusterState> {
        self.memberships
            .iter()
            .filter(|(_, m)| m.is_safe_for_deployment())
            .map(|(s, _)| s)
            .collect()
    }

    pub fn total_classified(&self) -> usize {
        self.memberships.len()
    }

    /// Compute statistics from current memberships.
    pub fn recompute_stats(&mut self) {
        self.stats = EnvelopeStats::default();
        for (_, m) in &self.memberships {
            match m {
                EnvelopeMembership::Inside => self.stats.inside_count += 1,
                EnvelopeMembership::Outside => self.stats.outside_count += 1,
                EnvelopeMembership::Boundary => self.stats.boundary_count += 1,
                EnvelopeMembership::PNR => self.stats.pnr_count += 1,
            }
        }
        self.stats.total_classified = self.memberships.len();
        self.stats.envelope_size =
            self.stats.inside_count + self.stats.boundary_count;
        if self.stats.total_classified > 0 {
            self.stats.envelope_ratio = self.stats.envelope_size as f64
                / self.stats.total_classified as f64;
        }
    }
}

impl fmt::Display for SafetyEnvelope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Envelope({}): {} inside, {} boundary, {} PNR, {} outside",
            self.id,
            self.stats.inside_count,
            self.stats.boundary_count,
            self.stats.pnr_count,
            self.stats.outside_count,
        )
    }
}

// ─── EnvelopeStats ──────────────────────────────────────────────────────

/// Statistics about the safety envelope.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnvelopeStats {
    pub inside_count: usize,
    pub outside_count: usize,
    pub boundary_count: usize,
    pub pnr_count: usize,
    pub total_classified: usize,
    pub envelope_size: usize,
    pub envelope_ratio: f64,
    pub max_forward_distance: Option<usize>,
    pub max_backward_distance: Option<usize>,
    pub computation_time_ms: u64,
}

impl EnvelopeStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_computation_time(mut self, ms: u64) -> Self {
        self.computation_time_ms = ms;
        self
    }
}

impl fmt::Display for EnvelopeStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Envelope: {}/{} states ({:.1}%), {} PNRs",
            self.envelope_size,
            self.total_classified,
            self.envelope_ratio * 100.0,
            self.pnr_count,
        )?;
        if self.computation_time_ms > 0 {
            write!(f, " [{}ms]", self.computation_time_ms)?;
        }
        Ok(())
    }
}

// ─── EnvelopeQuery ──────────────────────────────────────────────────────

/// Interface for querying envelope membership.
pub trait EnvelopeQuery: Send + Sync {
    fn membership(&self, state: &ClusterState) -> EnvelopeMembership;
    fn is_safe(&self, state: &ClusterState) -> bool {
        self.membership(state).is_safe_for_deployment()
    }
}

impl EnvelopeQuery for SafetyEnvelope {
    fn membership(&self, state: &ClusterState) -> EnvelopeMembership {
        self.query(state)
    }
}

/// A simple envelope query backed by a set of safe states.
pub struct SetEnvelopeQuery {
    inside: hashbrown::HashSet<ClusterState>,
    pnr: hashbrown::HashSet<ClusterState>,
}

impl SetEnvelopeQuery {
    pub fn new(
        inside: hashbrown::HashSet<ClusterState>,
        pnr: hashbrown::HashSet<ClusterState>,
    ) -> Self {
        Self { inside, pnr }
    }
}

impl EnvelopeQuery for SetEnvelopeQuery {
    fn membership(&self, state: &ClusterState) -> EnvelopeMembership {
        if self.inside.contains(state) {
            EnvelopeMembership::Inside
        } else if self.pnr.contains(state) {
            EnvelopeMembership::PNR
        } else {
            EnvelopeMembership::Outside
        }
    }
}

// ─── EnvelopeAnalysis ───────────────────────────────────────────────────

/// Analysis of a plan's trajectory through the envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvelopeTrajectory {
    pub steps: Vec<TrajectoryPoint>,
    pub first_pnr_index: Option<usize>,
    pub first_outside_index: Option<usize>,
    pub is_fully_inside: bool,
    pub has_pnr: bool,
}

impl EnvelopeTrajectory {
    pub fn new(steps: Vec<TrajectoryPoint>) -> Self {
        let first_pnr_index = steps
            .iter()
            .position(|p| p.membership == EnvelopeMembership::PNR);
        let first_outside_index = steps
            .iter()
            .position(|p| p.membership == EnvelopeMembership::Outside);
        let is_fully_inside = steps
            .iter()
            .all(|p| p.membership.is_safe_for_deployment());
        let has_pnr = first_pnr_index.is_some();
        Self {
            steps,
            first_pnr_index,
            first_outside_index,
            is_fully_inside,
            has_pnr,
        }
    }

    pub fn safe_prefix_length(&self) -> usize {
        self.steps
            .iter()
            .take_while(|p| p.membership.is_safe_for_deployment())
            .count()
    }
}

impl fmt::Display for EnvelopeTrajectory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Trajectory ({} points", self.steps.len())?;
        if self.is_fully_inside {
            write!(f, ", fully inside")?;
        }
        if self.has_pnr {
            write!(f, ", PNR at step {}", self.first_pnr_index.unwrap())?;
        }
        write!(f, ")")
    }
}

/// A single point in an envelope trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    pub step_index: usize,
    pub state: ClusterState,
    pub membership: EnvelopeMembership,
    pub constraint_evaluations: Vec<ConstraintEvaluation>,
}

impl TrajectoryPoint {
    pub fn new(
        step_index: usize,
        state: ClusterState,
        membership: EnvelopeMembership,
    ) -> Self {
        Self {
            step_index,
            state,
            membership,
            constraint_evaluations: Vec::new(),
        }
    }

    pub fn with_evaluations(mut self, evals: Vec<ConstraintEvaluation>) -> Self {
        self.constraint_evaluations = evals;
        self
    }

    pub fn has_violations(&self) -> bool {
        self.constraint_evaluations
            .iter()
            .any(|e| e.status.is_violated())
    }
}

// ─── Envelope computation request/result ────────────────────────────────

/// Request to compute an envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvelopeRequest {
    pub start_state: ClusterState,
    pub target_state: ClusterState,
    pub max_states: usize,
    pub timeout_ms: u64,
    pub include_witnesses: bool,
    pub include_distances: bool,
}

impl EnvelopeRequest {
    pub fn new(start: ClusterState, target: ClusterState) -> Self {
        Self {
            start_state: start,
            target_state: target,
            max_states: 100_000,
            timeout_ms: 30_000,
            include_witnesses: true,
            include_distances: false,
        }
    }

    pub fn with_max_states(mut self, max: usize) -> Self {
        self.max_states = max;
        self
    }

    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }
}

/// Result of an envelope computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvelopeResult {
    pub envelope: SafetyEnvelope,
    pub is_complete: bool,
    pub computation_time_ms: u64,
    pub states_explored: usize,
    pub notes: Vec<String>,
}

impl EnvelopeResult {
    pub fn new(envelope: SafetyEnvelope, time_ms: u64) -> Self {
        Self {
            envelope,
            is_complete: true,
            computation_time_ms: time_ms,
            states_explored: 0,
            notes: Vec::new(),
        }
    }

    pub fn with_explored(mut self, count: usize) -> Self {
        self.states_explored = count;
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn partial(mut self) -> Self {
        self.is_complete = false;
        self
    }
}

impl fmt::Display for EnvelopeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.envelope)?;
        if !self.is_complete {
            write!(f, " [PARTIAL]")?;
        }
        write!(f, " ({}ms, {} explored)", self.computation_time_ms, self.states_explored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::version::VersionIndex;

    fn make_state(versions: &[u32]) -> ClusterState {
        ClusterState::new(
            &versions
                .iter()
                .map(|&v| VersionIndex(v))
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn test_envelope_membership() {
        assert!(EnvelopeMembership::Inside.is_safe_for_deployment());
        assert!(EnvelopeMembership::Boundary.is_safe_for_deployment());
        assert!(!EnvelopeMembership::Outside.is_safe_for_deployment());
        assert!(!EnvelopeMembership::PNR.is_safe_for_deployment());
        assert!(EnvelopeMembership::PNR.is_pnr());
        assert!(EnvelopeMembership::Inside.allows_rollback());
        assert!(!EnvelopeMembership::PNR.allows_rollback());
    }

    #[test]
    fn test_reachability_result() {
        let state = make_state(&[0, 1]);
        let rr = ReachabilityResult::new(state.clone(), true, true);
        assert_eq!(rr.membership, EnvelopeMembership::Inside);

        let rr_pnr = ReachabilityResult::new(state.clone(), true, false);
        assert_eq!(rr_pnr.membership, EnvelopeMembership::PNR);

        let rr_out = ReachabilityResult::new(state, false, false);
        assert_eq!(rr_out.membership, EnvelopeMembership::Outside);
    }

    #[test]
    fn test_reachability_distances() {
        let state = make_state(&[0, 0]);
        let rr = ReachabilityResult::new(state, true, true)
            .with_forward_distance(3)
            .with_backward_distance(2);
        assert_eq!(rr.total_distance(), Some(5));
        assert!(rr.is_on_shortest_path(5));
        assert!(!rr.is_on_shortest_path(4));
    }

    #[test]
    fn test_point_of_no_return() {
        let state = make_state(&[1, 0]);
        let witness =
            StuckWitness::structural(vec![0], "Service 0 cannot be downgraded");
        let pnr = PointOfNoReturn::new(state.clone(), witness)
            .with_forward_distance(2)
            .with_note("schema migration applied");
        assert_eq!(pnr.severity, PNRSeverity::Structural);
        assert_eq!(pnr.forward_distance_to_target, Some(2));
        let s = pnr.to_string();
        assert!(s.contains("PNR"));
    }

    #[test]
    fn test_stuck_witness_constraint() {
        let witness = StuckWitness::constraint_blocked(
            vec![ConstraintId::new("api-compat")],
            "API compatibility prevents downgrade",
        );
        assert!(!witness.is_empty());
        let s = witness.to_string();
        assert!(s.contains("api-compat"));
    }

    #[test]
    fn test_stuck_witness_empty() {
        let witness = StuckWitness {
            blocking_constraints: Vec::new(),
            irreversible_services: Vec::new(),
            explanation: String::new(),
            failed_paths: Vec::new(),
        };
        assert!(witness.is_empty());
    }

    #[test]
    fn test_safety_envelope() {
        let start = make_state(&[0, 0]);
        let target = make_state(&[1, 1]);
        let mut env = SafetyEnvelope::new(start.clone(), target.clone());

        env.add_membership(start.clone(), EnvelopeMembership::Inside);
        env.add_membership(make_state(&[1, 0]), EnvelopeMembership::Inside);
        env.add_membership(make_state(&[0, 1]), EnvelopeMembership::PNR);
        env.add_membership(target.clone(), EnvelopeMembership::Inside);

        assert!(env.is_inside(&start));
        assert!(env.is_pnr(&make_state(&[0, 1])));
        assert_eq!(env.envelope_states().len(), 3);
    }

    #[test]
    fn test_envelope_query() {
        let start = make_state(&[0, 0]);
        let target = make_state(&[1, 1]);
        let mut env = SafetyEnvelope::new(start.clone(), target);
        env.add_membership(start.clone(), EnvelopeMembership::Inside);

        let m: &dyn EnvelopeQuery = &env;
        assert!(m.is_safe(&start));
        assert!(!m.is_safe(&make_state(&[9, 9])));
    }

    #[test]
    fn test_set_envelope_query() {
        let mut inside = hashbrown::HashSet::new();
        inside.insert(make_state(&[0, 0]));
        inside.insert(make_state(&[1, 1]));
        let mut pnr = hashbrown::HashSet::new();
        pnr.insert(make_state(&[1, 0]));

        let q = SetEnvelopeQuery::new(inside, pnr);
        assert_eq!(
            q.membership(&make_state(&[0, 0])),
            EnvelopeMembership::Inside
        );
        assert_eq!(
            q.membership(&make_state(&[1, 0])),
            EnvelopeMembership::PNR
        );
        assert_eq!(
            q.membership(&make_state(&[0, 1])),
            EnvelopeMembership::Outside
        );
    }

    #[test]
    fn test_envelope_stats() {
        let start = make_state(&[0, 0]);
        let target = make_state(&[1, 1]);
        let mut env = SafetyEnvelope::new(start.clone(), target.clone());
        env.add_membership(start, EnvelopeMembership::Inside);
        env.add_membership(make_state(&[1, 0]), EnvelopeMembership::Inside);
        env.add_membership(make_state(&[0, 1]), EnvelopeMembership::PNR);
        env.add_membership(target, EnvelopeMembership::Inside);
        env.recompute_stats();

        assert_eq!(env.stats.inside_count, 3);
        assert_eq!(env.stats.pnr_count, 1);
        assert_eq!(env.stats.total_classified, 4);
        assert_eq!(env.stats.envelope_size, 3);
    }

    #[test]
    fn test_envelope_trajectory() {
        let points = vec![
            TrajectoryPoint::new(0, make_state(&[0, 0]), EnvelopeMembership::Inside),
            TrajectoryPoint::new(1, make_state(&[1, 0]), EnvelopeMembership::Inside),
            TrajectoryPoint::new(2, make_state(&[1, 1]), EnvelopeMembership::PNR),
        ];
        let traj = EnvelopeTrajectory::new(points);
        assert!(!traj.is_fully_inside);
        assert!(traj.has_pnr);
        assert_eq!(traj.first_pnr_index, Some(2));
        assert_eq!(traj.safe_prefix_length(), 2);
    }

    #[test]
    fn test_envelope_trajectory_fully_inside() {
        let points = vec![
            TrajectoryPoint::new(0, make_state(&[0, 0]), EnvelopeMembership::Inside),
            TrajectoryPoint::new(1, make_state(&[1, 0]), EnvelopeMembership::Inside),
        ];
        let traj = EnvelopeTrajectory::new(points);
        assert!(traj.is_fully_inside);
        assert!(!traj.has_pnr);
    }

    #[test]
    fn test_failed_path() {
        let fp = FailedPath::new(
            vec![make_state(&[1, 0]), make_state(&[0, 0])],
            1,
            "constraint violated at step 1",
        )
        .with_violated(ConstraintId::new("c1"));
        assert_eq!(fp.violated_constraints.len(), 1);
        let s = fp.to_string();
        assert!(s.contains("step 1"));
    }

    #[test]
    fn test_envelope_request() {
        let req = EnvelopeRequest::new(make_state(&[0, 0]), make_state(&[1, 1]))
            .with_max_states(500)
            .with_timeout(5000);
        assert_eq!(req.max_states, 500);
        assert_eq!(req.timeout_ms, 5000);
    }

    #[test]
    fn test_envelope_result() {
        let env = SafetyEnvelope::new(make_state(&[0, 0]), make_state(&[1, 1]));
        let result = EnvelopeResult::new(env, 100)
            .with_explored(50)
            .with_note("partial exploration");
        assert!(result.is_complete);
        assert_eq!(result.computation_time_ms, 100);
        assert_eq!(result.states_explored, 50);
    }

    #[test]
    fn test_envelope_result_partial() {
        let env = SafetyEnvelope::new(make_state(&[0, 0]), make_state(&[1, 1]));
        let result = EnvelopeResult::new(env, 5000).partial();
        assert!(!result.is_complete);
    }

    #[test]
    fn test_trajectory_point_violations() {
        let eval = ConstraintEvaluation::violated(
            ConstraintId::new("c1"),
            crate::constraint::ConstraintStrength::Hard,
            "bad",
        );
        let point = TrajectoryPoint::new(0, make_state(&[0, 0]), EnvelopeMembership::Outside)
            .with_evaluations(vec![eval]);
        assert!(point.has_violations());
    }

    #[test]
    fn test_pnr_severity_display() {
        assert_eq!(PNRSeverity::Structural.to_string(), "structural");
        assert_eq!(
            PNRSeverity::ConstraintBlocked.to_string(),
            "constraint-blocked"
        );
    }

    #[test]
    fn test_envelope_display() {
        let start = make_state(&[0, 0]);
        let target = make_state(&[1, 1]);
        let mut env = SafetyEnvelope::new(start.clone(), target);
        env.add_membership(start, EnvelopeMembership::Inside);
        let s = env.to_string();
        assert!(s.contains("Envelope"));
        assert!(s.contains("1 inside"));
    }
}
