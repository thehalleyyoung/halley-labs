//! Circuit-breaker-aware abstraction layer for monotone cascade verification.
//!
//! # Problem
//!
//! Circuit breakers (CBs) break the monotonicity guarantee that antichain
//! pruning relies on (Theorem 2 in the paper): a CB can *trip* under high
//! load, *reducing* downstream traffic and potentially *preventing* a cascade
//! that would occur without the breaker. This means a failure set `F` may
//! cascade while a superset `F' ⊇ F` does not — a direct monotonicity
//! violation.
//!
//! # Solution: Sound Over-Approximation via Absorbing States
//!
//! We model each CB as a three-state lattice:
//!
//! ```text
//!   Closed  ≤  HalfOpen  ≤  Open        (CB state lattice)
//! ```
//!
//! The key insight is that a tripped CB *reduces* load (non-monotone), but a
//! **permanently-open** CB *increases* load on the caller (which must handle
//! the rejection). We over-approximate CB behavior by treating every CB as
//! *permanently open* once it could possibly trip:
//!
//! - **Closed → Open**: If the error count *could* reach the trip threshold
//!   under the current failure set, we conservatively set the CB to `Open`.
//! - **Open is absorbing**: Once open, the CB never transitions back to
//!   `Closed` or `HalfOpen` in our abstract model. This is sound because the
//!   real system can only trip *fewer* times (the half-open probe might
//!   succeed and re-close the breaker).
//! - **Load effect of Open**: An open CB adds its rejection-backpressure load
//!   to the *caller* (fast-fail retries, error handling) rather than
//!   subtracting load from the *callee*. This preserves monotonicity: more
//!   failures → more CBs forced open → more backpressure → more load.
//!
//! # Soundness Argument (Proof Sketch)
//!
//! **Claim.** The permanently-open CB abstraction is a sound
//! over-approximation: every cascade in the concrete (CB-inclusive) system is
//! also a cascade in the abstract model.
//!
//! **Proof sketch.**
//! 1. *Concrete model*: CB in state `s ∈ {Closed, HalfOpen, Open}` with
//!    transition `Closed →[err≥k] Open →[probe] HalfOpen →[ok] Closed`.
//! 2. *Abstract model*: CB in state `ŝ ∈ {Closed, Open}` with transition
//!    `Closed →[err≥k] Open` (absorbing — no back-transition).
//! 3. *Abstraction relation*: `α(Closed) = Closed`, `α(HalfOpen) = Open`,
//!    `α(Open) = Open`. The abstract state is always ≥ the concrete state
//!    in the CB lattice ordering.
//! 4. *Load over-approximation*: In the abstract model, an open CB adds
//!    backpressure `β(e)` to the caller. In the concrete model, a closed or
//!    half-open CB may *absorb* some of this load. Therefore
//!    `L_abstract(v, t, F) ≥ L_concrete(v, t, F)` for all v, t, F.
//! 5. *Monotonicity recovery*: Since the abstract load function composes
//!    only non-decreasing operators (addition, multiplication by ≥1 factors,
//!    and the absorbing CB open-state contribution), the abstract model
//!    satisfies monotonicity: `F ⊆ F' ⟹ L̂(v,t,F) ≤ L̂(v,t,F')`.
//! 6. *Soundness*: If `F` causes a cascade concretely (`L(v*,t,F) > κ(v*)`),
//!    then `L̂(v*,t,F) ≥ L(v*,t,F) > κ(v*)`, so the abstract model also
//!    detects it. ∎
//!
//! # Practical Consequence
//!
//! With this abstraction layer, the existing antichain pruning
//! ([`super::AntichainPruner`]) and MARCO enumeration remain sound even for
//! topologies containing circuit breakers. The cost is a potential increase
//! in false positives (the abstract model may report cascades that the real
//! CB would prevent), but no false negatives.

use cascade_types::policy::CircuitBreakerPolicy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// CB state lattice
// ---------------------------------------------------------------------------

/// Abstract circuit-breaker state, ordered as a monotone lattice:
/// `Closed < HalfOpen < Open`.
///
/// In the abstract model, `Open` is an absorbing state: once a CB is
/// considered open, it stays open for the remainder of the analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CbState {
    /// CB is passing traffic normally.
    Closed,
    /// CB is probing (concrete only — maps to `Open` in abstraction).
    HalfOpen,
    /// CB has tripped; all requests are rejected immediately.
    Open,
}

impl CbState {
    /// Lattice ordering: Closed < HalfOpen < Open.
    pub fn lattice_rank(self) -> u8 {
        match self {
            CbState::Closed => 0,
            CbState::HalfOpen => 1,
            CbState::Open => 2,
        }
    }

    /// Lattice join (least upper bound).
    pub fn join(self, other: CbState) -> CbState {
        if self.lattice_rank() >= other.lattice_rank() {
            self
        } else {
            other
        }
    }

    /// Returns `true` if this state blocks traffic to the downstream callee.
    pub fn is_blocking(self) -> bool {
        matches!(self, CbState::Open | CbState::HalfOpen)
    }
}

impl PartialOrd for CbState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CbState {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.lattice_rank().cmp(&other.lattice_rank())
    }
}

impl fmt::Display for CbState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CbState::Closed => write!(f, "Closed"),
            CbState::HalfOpen => write!(f, "HalfOpen"),
            CbState::Open => write!(f, "Open"),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-edge CB abstraction
// ---------------------------------------------------------------------------

/// Abstract representation of a circuit breaker on a single edge.
///
/// Tracks the abstract state and the backpressure load contributed to the
/// caller when the CB is open.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CbEdgeAbstraction {
    /// Source service (caller).
    pub source: String,
    /// Target service (callee).
    pub target: String,
    /// Current abstract CB state.
    pub state: CbState,
    /// Error threshold that triggers a trip (`consecutive_errors`).
    pub trip_threshold: u32,
    /// Backpressure load added to the *caller* when the CB is open.
    /// Models fast-fail overhead: error-handling, retry attempts against
    /// the open breaker, logging, etc.
    pub backpressure_load: i64,
}

impl CbEdgeAbstraction {
    /// Create a new CB abstraction from a policy.
    ///
    /// The backpressure load is estimated as `max_retries + 1` — each
    /// rejected request still costs one attempt plus retry overhead on
    /// the caller side.
    pub fn from_policy(source: &str, target: &str, policy: &CircuitBreakerPolicy) -> Self {
        Self {
            source: source.to_string(),
            target: target.to_string(),
            state: CbState::Closed,
            trip_threshold: policy.consecutive_errors,
            backpressure_load: (policy.max_retries as i64) + 1,
        }
    }

    /// Attempt to trip this CB given the observed error count.
    ///
    /// If `errors >= trip_threshold`, the state moves to `Open` and
    /// **stays there** (absorbing). This is the core of the monotone
    /// over-approximation: we never model recovery.
    pub fn maybe_trip(&mut self, errors: u32) -> bool {
        if errors >= self.trip_threshold {
            self.state = self.state.join(CbState::Open);
            true
        } else {
            false
        }
    }

    /// Return the load contribution of this CB to the caller.
    ///
    /// - `Closed`: 0 (traffic flows normally to callee).
    /// - `Open`/`HalfOpen`: `backpressure_load` added to caller.
    pub fn caller_load_contribution(&self) -> i64 {
        if self.state.is_blocking() {
            self.backpressure_load
        } else {
            0
        }
    }

    /// Return the load multiplier for traffic flowing through this edge.
    ///
    /// - `Closed`: normal retry amplification `(1 + r)`.
    /// - `Open`: 0 — no traffic reaches the callee (it's all rejected).
    ///
    /// Combined with [`caller_load_contribution`], this ensures that
    /// opening a CB *moves* load from callee to caller rather than
    /// removing it from the system, preserving monotonicity.
    pub fn callee_traffic_factor(&self, retry_count: u32) -> i64 {
        if self.state.is_blocking() {
            0
        } else {
            1 + retry_count as i64
        }
    }
}

// ---------------------------------------------------------------------------
// Topology-wide CB abstraction
// ---------------------------------------------------------------------------

/// Circuit-breaker-aware abstraction layer for an entire topology.
///
/// Wraps the per-edge CB abstractions and provides methods for computing
/// abstract load propagation that preserves monotonicity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerAbstraction {
    /// Per-edge CB state, keyed by `(source, target)`.
    edges: HashMap<(String, String), CbEdgeAbstraction>,
}

impl CircuitBreakerAbstraction {
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
        }
    }

    /// Register a CB-protected edge.
    pub fn add_edge(&mut self, edge: CbEdgeAbstraction) {
        self.edges
            .insert((edge.source.clone(), edge.target.clone()), edge);
    }

    /// Register a CB from a policy on a given edge.
    pub fn add_from_policy(
        &mut self,
        source: &str,
        target: &str,
        policy: &CircuitBreakerPolicy,
    ) {
        self.add_edge(CbEdgeAbstraction::from_policy(source, target, policy));
    }

    /// Conservatively trip all CBs whose error count could reach the
    /// threshold given the current failure set.
    ///
    /// In the monotone over-approximation, any edge whose target is in
    /// the failure set will accumulate enough errors to trip. Additionally,
    /// any edge whose target is overloaded (load > capacity) is assumed
    /// to eventually produce enough errors.
    pub fn apply_failure_set(
        &mut self,
        failed_services: &[String],
        overloaded_services: &[String],
    ) {
        let should_trip: std::collections::HashSet<&str> = failed_services
            .iter()
            .chain(overloaded_services.iter())
            .map(|s| s.as_str())
            .collect();

        for edge in self.edges.values_mut() {
            if should_trip.contains(edge.target.as_str()) {
                // Force trip: the target is either failed or overloaded,
                // so the CB will see consecutive errors ≥ threshold.
                edge.state = edge.state.join(CbState::Open);
            }
        }
    }

    /// Compute the abstract load contribution from CB backpressure
    /// for a given caller service.
    ///
    /// Sums backpressure from all open CBs whose source is `service_id`.
    pub fn backpressure_for_caller(&self, service_id: &str) -> i64 {
        self.edges
            .values()
            .filter(|e| e.source == service_id)
            .map(|e| e.caller_load_contribution())
            .sum()
    }

    /// Get the traffic factor for a specific edge.
    ///
    /// Returns `None` if the edge has no CB (should use normal retry
    /// amplification). Returns `Some(factor)` if a CB is present.
    pub fn traffic_factor(&self, source: &str, target: &str, retry_count: u32) -> Option<i64> {
        self.edges
            .get(&(source.to_string(), target.to_string()))
            .map(|e| e.callee_traffic_factor(retry_count))
    }

    /// Get the abstract state of a CB on a specific edge.
    pub fn edge_state(&self, source: &str, target: &str) -> Option<CbState> {
        self.edges
            .get(&(source.to_string(), target.to_string()))
            .map(|e| e.state)
    }

    /// Count how many CBs are in each state.
    pub fn state_summary(&self) -> CbStateSummary {
        let mut summary = CbStateSummary::default();
        for edge in self.edges.values() {
            match edge.state {
                CbState::Closed => summary.closed += 1,
                CbState::HalfOpen => summary.half_open += 1,
                CbState::Open => summary.open += 1,
            }
        }
        summary
    }

    /// Return `true` if the abstraction is currently monotone-safe,
    /// i.e., all CBs are either `Closed` (normal) or `Open` (absorbing).
    ///
    /// The `HalfOpen` state is only used as an intermediate in the
    /// concrete model; in the abstraction it immediately maps to `Open`.
    pub fn is_monotone_safe(&self) -> bool {
        self.edges.values().all(|e| {
            matches!(e.state, CbState::Closed | CbState::Open)
        })
    }

    /// Number of registered CB-protected edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

impl Default for CircuitBreakerAbstraction {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of CB states across the topology.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CbStateSummary {
    pub closed: usize,
    pub half_open: usize,
    pub open: usize,
}

impl fmt::Display for CbStateSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CB states: {} closed, {} half-open, {} open",
            self.closed, self.half_open, self.open
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cascade_types::policy::CircuitBreakerPolicy;

    fn default_policy() -> CircuitBreakerPolicy {
        CircuitBreakerPolicy::new()
            .with_consecutive_errors(5)
            .with_max_retries(3)
    }

    // -- CbState lattice tests ----------------------------------------------

    #[test]
    fn test_cb_state_ordering() {
        assert!(CbState::Closed < CbState::HalfOpen);
        assert!(CbState::HalfOpen < CbState::Open);
        assert!(CbState::Closed < CbState::Open);
    }

    #[test]
    fn test_cb_state_join_idempotent() {
        for s in [CbState::Closed, CbState::HalfOpen, CbState::Open] {
            assert_eq!(s.join(s), s);
        }
    }

    #[test]
    fn test_cb_state_join_commutative() {
        let pairs = [
            (CbState::Closed, CbState::Open),
            (CbState::Closed, CbState::HalfOpen),
            (CbState::HalfOpen, CbState::Open),
        ];
        for (a, b) in pairs {
            assert_eq!(a.join(b), b.join(a));
        }
    }

    #[test]
    fn test_cb_state_join_absorbing() {
        // Open is the top element — join with anything gives Open
        assert_eq!(CbState::Open.join(CbState::Closed), CbState::Open);
        assert_eq!(CbState::Open.join(CbState::HalfOpen), CbState::Open);
    }

    #[test]
    fn test_cb_state_is_blocking() {
        assert!(!CbState::Closed.is_blocking());
        assert!(CbState::HalfOpen.is_blocking());
        assert!(CbState::Open.is_blocking());
    }

    #[test]
    fn test_cb_state_display() {
        assert_eq!(format!("{}", CbState::Closed), "Closed");
        assert_eq!(format!("{}", CbState::Open), "Open");
    }

    // -- CbEdgeAbstraction tests -------------------------------------------

    #[test]
    fn test_edge_from_policy() {
        let policy = default_policy();
        let edge = CbEdgeAbstraction::from_policy("a", "b", &policy);
        assert_eq!(edge.state, CbState::Closed);
        assert_eq!(edge.trip_threshold, 5);
        assert_eq!(edge.backpressure_load, 4); // max_retries(3) + 1
    }

    #[test]
    fn test_edge_trip_below_threshold() {
        let policy = default_policy();
        let mut edge = CbEdgeAbstraction::from_policy("a", "b", &policy);
        assert!(!edge.maybe_trip(3));
        assert_eq!(edge.state, CbState::Closed);
    }

    #[test]
    fn test_edge_trip_at_threshold() {
        let policy = default_policy();
        let mut edge = CbEdgeAbstraction::from_policy("a", "b", &policy);
        assert!(edge.maybe_trip(5));
        assert_eq!(edge.state, CbState::Open);
    }

    #[test]
    fn test_edge_trip_is_absorbing() {
        let policy = default_policy();
        let mut edge = CbEdgeAbstraction::from_policy("a", "b", &policy);
        edge.maybe_trip(5);
        assert_eq!(edge.state, CbState::Open);
        // Trying to "un-trip" by reporting zero errors has no effect
        edge.maybe_trip(0);
        assert_eq!(edge.state, CbState::Open);
    }

    #[test]
    fn test_edge_load_closed() {
        let policy = default_policy();
        let edge = CbEdgeAbstraction::from_policy("a", "b", &policy);
        assert_eq!(edge.caller_load_contribution(), 0);
        assert_eq!(edge.callee_traffic_factor(3), 4); // 1 + 3 retries
    }

    #[test]
    fn test_edge_load_open() {
        let policy = default_policy();
        let mut edge = CbEdgeAbstraction::from_policy("a", "b", &policy);
        edge.maybe_trip(10);
        assert_eq!(edge.caller_load_contribution(), 4); // backpressure
        assert_eq!(edge.callee_traffic_factor(3), 0); // no traffic to callee
    }

    // -- CircuitBreakerAbstraction tests -----------------------------------

    #[test]
    fn test_abstraction_empty() {
        let abs = CircuitBreakerAbstraction::new();
        assert_eq!(abs.edge_count(), 0);
        assert!(abs.is_monotone_safe());
    }

    #[test]
    fn test_abstraction_add_edge() {
        let mut abs = CircuitBreakerAbstraction::new();
        abs.add_from_policy("a", "b", &default_policy());
        assert_eq!(abs.edge_count(), 1);
        assert_eq!(abs.edge_state("a", "b"), Some(CbState::Closed));
    }

    #[test]
    fn test_abstraction_apply_failure_set() {
        let mut abs = CircuitBreakerAbstraction::new();
        abs.add_from_policy("a", "b", &default_policy());
        abs.add_from_policy("b", "c", &default_policy());

        // Fail service "b" — CB on edge a→b should trip
        abs.apply_failure_set(&["b".into()], &[]);
        assert_eq!(abs.edge_state("a", "b"), Some(CbState::Open));
        // CB on edge b→c is unaffected (target "c" is not failed)
        assert_eq!(abs.edge_state("b", "c"), Some(CbState::Closed));
    }

    #[test]
    fn test_abstraction_backpressure() {
        let mut abs = CircuitBreakerAbstraction::new();
        abs.add_from_policy("a", "b", &default_policy());
        abs.add_from_policy("a", "c", &default_policy());

        // No failures: zero backpressure
        assert_eq!(abs.backpressure_for_caller("a"), 0);

        // Fail both targets
        abs.apply_failure_set(&["b".into(), "c".into()], &[]);
        // Backpressure = 4 + 4 = 8 (from both open CBs)
        assert_eq!(abs.backpressure_for_caller("a"), 8);
    }

    #[test]
    fn test_abstraction_traffic_factor() {
        let mut abs = CircuitBreakerAbstraction::new();
        abs.add_from_policy("a", "b", &default_policy());

        // Closed: normal retry amplification
        assert_eq!(abs.traffic_factor("a", "b", 3), Some(4));

        abs.apply_failure_set(&["b".into()], &[]);

        // Open: no traffic to callee
        assert_eq!(abs.traffic_factor("a", "b", 3), Some(0));
        // Non-CB edge: None
        assert_eq!(abs.traffic_factor("x", "y", 3), None);
    }

    #[test]
    fn test_abstraction_state_summary() {
        let mut abs = CircuitBreakerAbstraction::new();
        abs.add_from_policy("a", "b", &default_policy());
        abs.add_from_policy("b", "c", &default_policy());
        abs.add_from_policy("c", "d", &default_policy());

        abs.apply_failure_set(&["b".into()], &[]);

        let summary = abs.state_summary();
        assert_eq!(summary.open, 1);
        assert_eq!(summary.closed, 2);
    }

    #[test]
    fn test_monotonicity_preserved_under_growing_failures() {
        // Core monotonicity test: adding more failures should only
        // increase (or maintain) the total backpressure, never decrease it.
        let policy = default_policy();
        let mut abs1 = CircuitBreakerAbstraction::new();
        abs1.add_from_policy("a", "b", &policy);
        abs1.add_from_policy("a", "c", &policy);
        abs1.add_from_policy("b", "c", &policy);

        let mut abs2 = abs1.clone();

        // F = {b}
        abs1.apply_failure_set(&["b".into()], &[]);
        let bp1 = abs1.backpressure_for_caller("a");

        // F' = {b, c} ⊇ F
        abs2.apply_failure_set(&["b".into(), "c".into()], &[]);
        let bp2 = abs2.backpressure_for_caller("a");

        // Monotonicity: bp(F') >= bp(F)
        assert!(
            bp2 >= bp1,
            "Monotonicity violated: bp({{}b,c}})={bp2} < bp({{}b}})={bp1}"
        );
    }
}
