//! Algebraic property verification — checks that Axioms A1–A4 hold for a
//! given code region, enabling the protocol-aware merge operator.
//!
//! - A1: Finite outcomes (|C| ≤ 350, |V| ≤ 6, |E| ≤ 30)
//! - A2: Lattice preferences (partial order on cipher suites)
//! - A3: Monotonic progression (handshake phases form acyclic DAG)
//! - A4: Deterministic selection (given fixed offered sets, selection is unique)

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

use petgraph::graph::DiGraph;
use petgraph::visit::Dfs;
use petgraph::Direction;
use serde::{Deserialize, Serialize};

use negsyn_types::{
    CipherSuite, HandshakePhase, MergeConfig, MergeError, NegotiationState, ProtocolVersion,
    SymbolicState, SymbolicValue,
};

use crate::lattice::SecurityLattice;

// ---------------------------------------------------------------------------
// Property violations
// ---------------------------------------------------------------------------

/// Detailed description of an algebraic property violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertyViolation {
    FiniteOutcomeExceeded {
        category: OutcomeCategory,
        actual: usize,
        limit: usize,
    },
    LatticeViolation {
        suite_a: u16,
        suite_b: u16,
        description: String,
    },
    MonotonicityViolation {
        from_phase: String,
        to_phase: String,
        description: String,
    },
    DeterminismViolation {
        offered_set: Vec<u16>,
        selections: Vec<u16>,
        description: String,
    },
}

impl fmt::Display for PropertyViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FiniteOutcomeExceeded {
                category,
                actual,
                limit,
            } => write!(
                f,
                "A1 violated: {:?} count {} exceeds limit {}",
                category, actual, limit
            ),
            Self::LatticeViolation {
                suite_a,
                suite_b,
                description,
            } => write!(
                f,
                "A2 violated: suites 0x{:04x}, 0x{:04x}: {}",
                suite_a, suite_b, description
            ),
            Self::MonotonicityViolation {
                from_phase,
                to_phase,
                description,
            } => write!(
                f,
                "A3 violated: {} -> {}: {}",
                from_phase, to_phase, description
            ),
            Self::DeterminismViolation {
                offered_set,
                selections,
                description,
            } => write!(
                f,
                "A4 violated: offered {:?}, got selections {:?}: {}",
                offered_set, selections, description
            ),
        }
    }
}

/// Category of outcome being counted for A1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutcomeCategory {
    CipherSuites,
    Versions,
    Extensions,
}

// ---------------------------------------------------------------------------
// Fallback strategy for when properties fail
// ---------------------------------------------------------------------------

/// What to do when a property check fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FallbackAction {
    /// Use generic veritesting (no protocol-specific optimization).
    GenericVeritesting,
    /// Fork the states instead of merging.
    Fork,
    /// Attempt to decompose the region into smaller pieces.
    Decompose,
    /// Summarize the region abstractly.
    Summarize,
}

// ---------------------------------------------------------------------------
// Individual checkers
// ---------------------------------------------------------------------------

/// A1: Checks that outcome sets are bounded within limits.
pub struct FiniteOutcomeChecker {
    max_ciphers: usize,
    max_versions: usize,
    max_extensions: usize,
}

impl FiniteOutcomeChecker {
    pub fn new(config: &MergeConfig) -> Self {
        Self {
            max_ciphers: config.max_cipher_outcomes as usize,
            max_versions: config.max_version_outcomes as usize,
            max_extensions: config.max_extension_outcomes as usize,
        }
    }

    /// Check that the number of possible cipher suite outcomes is bounded.
    pub fn check_cipher_bound(&self, offered: &BTreeSet<u16>) -> Result<(), PropertyViolation> {
        if offered.len() > self.max_ciphers {
            return Err(PropertyViolation::FiniteOutcomeExceeded {
                category: OutcomeCategory::CipherSuites,
                actual: offered.len(),
                limit: self.max_ciphers,
            });
        }
        Ok(())
    }

    /// Check that the number of version outcomes is bounded.
    pub fn check_version_bound(
        &self,
        versions: &[ProtocolVersion],
    ) -> Result<(), PropertyViolation> {
        let unique: BTreeSet<_> = versions.iter().collect();
        if unique.len() > self.max_versions {
            return Err(PropertyViolation::FiniteOutcomeExceeded {
                category: OutcomeCategory::Versions,
                actual: unique.len(),
                limit: self.max_versions,
            });
        }
        Ok(())
    }

    /// Check that the number of extension outcomes is bounded.
    pub fn check_extension_bound(&self, extension_count: usize) -> Result<(), PropertyViolation> {
        if extension_count > self.max_extensions {
            return Err(PropertyViolation::FiniteOutcomeExceeded {
                category: OutcomeCategory::Extensions,
                actual: extension_count,
                limit: self.max_extensions,
            });
        }
        Ok(())
    }

    /// Full A1 check across all outcome categories.
    pub fn check_all(
        &self,
        offered_ciphers: &BTreeSet<u16>,
        versions: &[ProtocolVersion],
        extension_count: usize,
    ) -> Vec<PropertyViolation> {
        let mut violations = Vec::new();
        if let Err(v) = self.check_cipher_bound(offered_ciphers) {
            violations.push(v);
        }
        if let Err(v) = self.check_version_bound(versions) {
            violations.push(v);
        }
        if let Err(v) = self.check_extension_bound(extension_count) {
            violations.push(v);
        }
        violations
    }
}

/// A2: Verifies that cipher suite preferences form a lattice.
pub struct LatticeChecker {
    lattice: SecurityLattice,
}

impl LatticeChecker {
    pub fn new(lattice: SecurityLattice) -> Self {
        Self { lattice }
    }

    pub fn from_standard() -> Self {
        Self {
            lattice: SecurityLattice::from_standard_registry(),
        }
    }

    /// Check that every pair of suites in the offered set has a join and meet.
    pub fn check_lattice_property(
        &self,
        offered: &BTreeSet<u16>,
    ) -> Vec<PropertyViolation> {
        let mut violations = Vec::new();
        let ids: Vec<u16> = offered.iter().copied().collect();

        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let a = ids[i];
                let b = ids[j];

                // Check that both are registered
                if self.lattice.profile(a).is_none() {
                    violations.push(PropertyViolation::LatticeViolation {
                        suite_a: a,
                        suite_b: b,
                        description: format!("Suite 0x{:04x} not registered in lattice", a),
                    });
                    continue;
                }
                if self.lattice.profile(b).is_none() {
                    violations.push(PropertyViolation::LatticeViolation {
                        suite_a: a,
                        suite_b: b,
                        description: format!("Suite 0x{:04x} not registered in lattice", b),
                    });
                    continue;
                }

                // Check join existence
                if self.lattice.join(a, b).is_none() {
                    violations.push(PropertyViolation::LatticeViolation {
                        suite_a: a,
                        suite_b: b,
                        description: "No join (least upper bound) exists".to_string(),
                    });
                }

                // Check meet existence
                if self.lattice.meet(a, b).is_none() {
                    violations.push(PropertyViolation::LatticeViolation {
                        suite_a: a,
                        suite_b: b,
                        description: "No meet (greatest lower bound) exists".to_string(),
                    });
                }
            }
        }

        violations
    }

    /// Check transitivity: if a ≤ b and b ≤ c, then a ≤ c.
    pub fn check_transitivity(&self, offered: &BTreeSet<u16>) -> Vec<PropertyViolation> {
        let mut violations = Vec::new();
        let ids: Vec<u16> = offered.iter().copied().collect();

        for i in 0..ids.len() {
            for j in 0..ids.len() {
                if i == j {
                    continue;
                }
                for k in 0..ids.len() {
                    if j == k || i == k {
                        continue;
                    }
                    let a_le_b = self.lattice.is_at_least_as_strong(ids[j], ids[i]);
                    let b_le_c = self.lattice.is_at_least_as_strong(ids[k], ids[j]);
                    if a_le_b && b_le_c {
                        let a_le_c = self.lattice.is_at_least_as_strong(ids[k], ids[i]);
                        if !a_le_c {
                            violations.push(PropertyViolation::LatticeViolation {
                                suite_a: ids[i],
                                suite_b: ids[k],
                                description: format!(
                                    "Transitivity violation: 0x{:04x} ≤ 0x{:04x} ≤ 0x{:04x} but not 0x{:04x} ≤ 0x{:04x}",
                                    ids[i], ids[j], ids[k], ids[i], ids[k]
                                ),
                            });
                        }
                    }
                }
            }
        }

        violations
    }

    pub fn lattice(&self) -> &SecurityLattice {
        &self.lattice
    }
}

/// A3: Checks that handshake phase progression is monotonic (acyclic DAG).
pub struct MonotonicityChecker {
    phase_graph: DiGraph<HandshakePhase, ()>,
    phase_indices: HashMap<HandshakePhase, petgraph::graph::NodeIndex>,
}

impl MonotonicityChecker {
    pub fn new() -> Self {
        let mut graph = DiGraph::new();
        let mut indices = HashMap::new();

        // Add all phases as nodes
        for phase in HandshakePhase::all_phases() {
            let idx = graph.add_node(*phase);
            indices.insert(*phase, idx);
        }

        // Add standard progression edges
        let edges = [
            (HandshakePhase::Initial, HandshakePhase::ClientHello),
            (HandshakePhase::ClientHello, HandshakePhase::ServerHello),
            (HandshakePhase::ServerHello, HandshakePhase::Certificate),
            (HandshakePhase::Certificate, HandshakePhase::KeyExchange),
            (HandshakePhase::KeyExchange, HandshakePhase::ChangeCipherSpec),
            (HandshakePhase::ChangeCipherSpec, HandshakePhase::Finished),
            (HandshakePhase::Finished, HandshakePhase::ApplicationData),
            // Alert can come from any phase
            (HandshakePhase::ClientHello, HandshakePhase::Alert),
            (HandshakePhase::ServerHello, HandshakePhase::Alert),
            (HandshakePhase::Certificate, HandshakePhase::Alert),
            (HandshakePhase::KeyExchange, HandshakePhase::Alert),
            // Renegotiation
            (HandshakePhase::ApplicationData, HandshakePhase::Renegotiation),
            (HandshakePhase::Renegotiation, HandshakePhase::ClientHello),
        ];

        for (from, to) in &edges {
            let from_idx = indices[from];
            let to_idx = indices[to];
            graph.add_edge(from_idx, to_idx, ());
        }

        Self {
            phase_graph: graph,
            phase_indices: indices,
        }
    }

    /// Check that the phase transition DAG is acyclic (ignoring renegotiation).
    pub fn check_acyclicity(&self) -> Vec<PropertyViolation> {
        let mut violations = Vec::new();

        // Build a copy without renegotiation edges for acyclicity check
        let mut check_graph = DiGraph::new();
        let mut check_indices = HashMap::new();

        for phase in HandshakePhase::all_phases() {
            if *phase == HandshakePhase::Renegotiation {
                continue;
            }
            let idx = check_graph.add_node(*phase);
            check_indices.insert(*phase, idx);
        }

        for edge in self.phase_graph.edge_indices() {
            let (src, dst) = self.phase_graph.edge_endpoints(edge).unwrap();
            let src_phase = self.phase_graph[src];
            let dst_phase = self.phase_graph[dst];

            if src_phase == HandshakePhase::Renegotiation
                || dst_phase == HandshakePhase::Renegotiation
            {
                continue;
            }

            if let (Some(&si), Some(&di)) =
                (check_indices.get(&src_phase), check_indices.get(&dst_phase))
            {
                check_graph.add_edge(si, di, ());
            }
        }

        // Check for cycles using topological sort
        if petgraph::algo::is_cyclic_directed(&check_graph) {
            violations.push(PropertyViolation::MonotonicityViolation {
                from_phase: "unknown".to_string(),
                to_phase: "unknown".to_string(),
                description: "Phase transition graph contains a cycle (excluding renegotiation)"
                    .to_string(),
            });
        }

        violations
    }

    /// Check that a specific transition is valid (forward progression).
    pub fn check_transition(
        &self,
        from: HandshakePhase,
        to: HandshakePhase,
    ) -> Result<(), PropertyViolation> {
        let from_idx = self.phase_indices.get(&from);
        let to_idx = self.phase_indices.get(&to);

        match (from_idx, to_idx) {
            (Some(&fi), Some(&ti)) => {
                // Check if there's a direct edge
                if self.phase_graph.contains_edge(fi, ti) {
                    Ok(())
                } else {
                    // Check if there's a path
                    let mut dfs = Dfs::new(&self.phase_graph, fi);
                    while let Some(visited) = dfs.next(&self.phase_graph) {
                        if visited == ti {
                            return Ok(());
                        }
                    }
                    Err(PropertyViolation::MonotonicityViolation {
                        from_phase: format!("{:?}", from),
                        to_phase: format!("{:?}", to),
                        description: "No valid forward path between phases".to_string(),
                    })
                }
            }
            _ => Err(PropertyViolation::MonotonicityViolation {
                from_phase: format!("{:?}", from),
                to_phase: format!("{:?}", to),
                description: "Unknown phase".to_string(),
            }),
        }
    }

    /// Verify that a sequence of observed phases follows monotonic progression.
    pub fn check_sequence(&self, phases: &[HandshakePhase]) -> Vec<PropertyViolation> {
        let mut violations = Vec::new();
        for window in phases.windows(2) {
            if let Err(v) = self.check_transition(window[0], window[1]) {
                violations.push(v);
            }
        }
        violations
    }

    /// Check that observed phase ordering indices are non-decreasing (with
    /// tolerance for renegotiation).
    pub fn check_order_indices(&self, phases: &[HandshakePhase]) -> Vec<PropertyViolation> {
        let mut violations = Vec::new();
        let mut prev_idx = 0u8;

        for (i, phase) in phases.iter().enumerate() {
            let idx = phase.order_index();
            if idx < prev_idx && *phase != HandshakePhase::Renegotiation
                && *phase != HandshakePhase::ClientHello
            {
                violations.push(PropertyViolation::MonotonicityViolation {
                    from_phase: format!("{:?}({})", phases[i.saturating_sub(1)], prev_idx),
                    to_phase: format!("{:?}({})", phase, idx),
                    description: "Phase order index decreased".to_string(),
                });
            }
            prev_idx = idx;
        }

        violations
    }
}

impl Default for MonotonicityChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// A4: Verifies that cipher selection is deterministic given fixed inputs.
pub struct DeterminismChecker {
    observed_selections: HashMap<BTreeSet<u16>, BTreeSet<u16>>,
}

impl DeterminismChecker {
    pub fn new() -> Self {
        Self {
            observed_selections: HashMap::new(),
        }
    }

    /// Record an observed selection: given offered set, a selection was made.
    pub fn record_selection(&mut self, offered: BTreeSet<u16>, selected: u16) {
        self.observed_selections
            .entry(offered)
            .or_default()
            .insert(selected);
    }

    /// Check if all observed selections are deterministic (unique per offered set).
    pub fn check_determinism(&self) -> Vec<PropertyViolation> {
        let mut violations = Vec::new();

        for (offered, selections) in &self.observed_selections {
            if selections.len() > 1 {
                violations.push(PropertyViolation::DeterminismViolation {
                    offered_set: offered.iter().copied().collect(),
                    selections: selections.iter().copied().collect(),
                    description: format!(
                        "Multiple selections ({}) observed for the same offered set",
                        selections.len()
                    ),
                });
            }
        }

        violations
    }

    /// Check determinism for a specific offered set against an expected selection.
    pub fn check_specific(
        &self,
        offered: &BTreeSet<u16>,
        expected: u16,
    ) -> Result<(), PropertyViolation> {
        if let Some(selections) = self.observed_selections.get(offered) {
            if selections.len() == 1 && selections.contains(&expected) {
                Ok(())
            } else if selections.len() > 1 {
                Err(PropertyViolation::DeterminismViolation {
                    offered_set: offered.iter().copied().collect(),
                    selections: selections.iter().copied().collect(),
                    description: format!("Expected 0x{:04x} but multiple selections exist", expected),
                })
            } else {
                Err(PropertyViolation::DeterminismViolation {
                    offered_set: offered.iter().copied().collect(),
                    selections: selections.iter().copied().collect(),
                    description: format!(
                        "Expected 0x{:04x} but observed {:?}",
                        expected, selections
                    ),
                })
            }
        } else {
            Ok(()) // No prior observations, assume valid
        }
    }

    /// Clear all recorded observations.
    pub fn clear(&mut self) {
        self.observed_selections.clear();
    }

    pub fn observation_count(&self) -> usize {
        self.observed_selections.len()
    }
}

impl Default for DeterminismChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Composite property checker
// ---------------------------------------------------------------------------

/// Checks all four algebraic properties (A1–A4) for a code region.
pub struct PropertyChecker {
    finite_checker: FiniteOutcomeChecker,
    lattice_checker: LatticeChecker,
    monotonicity_checker: MonotonicityChecker,
    determinism_checker: DeterminismChecker,
    config: MergeConfig,
}

/// Result of a full property check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyCheckResult {
    pub a1_satisfied: bool,
    pub a2_satisfied: bool,
    pub a3_satisfied: bool,
    pub a4_satisfied: bool,
    pub violations: Vec<PropertyViolation>,
    pub recommended_fallback: Option<FallbackAction>,
}

impl PropertyCheckResult {
    pub fn all_satisfied(&self) -> bool {
        self.a1_satisfied && self.a2_satisfied && self.a3_satisfied && self.a4_satisfied
    }

    pub fn merge_eligible(&self) -> bool {
        // A1 and A3 are hard requirements; A2 and A4 can be worked around
        self.a1_satisfied && self.a3_satisfied
    }
}

impl PropertyChecker {
    pub fn new(config: MergeConfig) -> Self {
        Self {
            finite_checker: FiniteOutcomeChecker::new(&config),
            lattice_checker: LatticeChecker::from_standard(),
            monotonicity_checker: MonotonicityChecker::new(),
            determinism_checker: DeterminismChecker::new(),
            config,
        }
    }

    pub fn with_lattice(mut self, lattice: SecurityLattice) -> Self {
        self.lattice_checker = LatticeChecker::new(lattice);
        self
    }

    /// Run all property checks against observed states.
    pub fn check_all(
        &self,
        states: &[&SymbolicState],
    ) -> PropertyCheckResult {
        let mut violations = Vec::new();

        // A1: Finite outcomes
        let a1_violations = self.check_a1(states);
        let a1_ok = a1_violations.is_empty();
        violations.extend(a1_violations);

        // A2: Lattice preferences
        let a2_violations = self.check_a2(states);
        let a2_ok = a2_violations.is_empty();
        violations.extend(a2_violations);

        // A3: Monotonic progression
        let a3_violations = self.check_a3(states);
        let a3_ok = a3_violations.is_empty();
        violations.extend(a3_violations);

        // A4: Determinism
        let a4_violations = self.check_a4(states);
        let a4_ok = a4_violations.is_empty();
        violations.extend(a4_violations);

        let fallback = self.select_fallback(&violations, a1_ok, a2_ok, a3_ok, a4_ok);

        PropertyCheckResult {
            a1_satisfied: a1_ok,
            a2_satisfied: a2_ok,
            a3_satisfied: a3_ok,
            a4_satisfied: a4_ok,
            violations,
            recommended_fallback: fallback,
        }
    }

    fn check_a1(&self, states: &[&SymbolicState]) -> Vec<PropertyViolation> {
        let mut all_ciphers = BTreeSet::new();
        let mut all_versions = Vec::new();
        let mut max_extensions = 0;

        for state in states {
            all_ciphers.extend(state.negotiation.offered_ciphers.iter().map(|c| c.iana_id));
            all_versions.extend(state.negotiation.version.iter().cloned());
            max_extensions = max_extensions.max(state.negotiation.extensions.len());
        }

        self.finite_checker
            .check_all(&all_ciphers, &all_versions, max_extensions)
    }

    fn check_a2(&self, states: &[&SymbolicState]) -> Vec<PropertyViolation> {
        let mut all_ciphers = BTreeSet::new();
        for state in states {
            all_ciphers.extend(state.negotiation.offered_ciphers.iter().map(|c| c.iana_id));
        }
        self.lattice_checker.check_lattice_property(&all_ciphers)
    }

    fn check_a3(&self, states: &[&SymbolicState]) -> Vec<PropertyViolation> {
        // Check pairwise phase ordering
        let phases: Vec<HandshakePhase> = states
            .iter()
            .map(|s| s.negotiation.phase)
            .collect();
        let mut violations = self.monotonicity_checker.check_acyclicity();
        violations.extend(self.monotonicity_checker.check_order_indices(&phases));
        violations
    }

    fn check_a4(&self, states: &[&SymbolicState]) -> Vec<PropertyViolation> {
        let mut checker = DeterminismChecker::new();
        for state in states {
            if let Some(ref selected) = state.negotiation.selected_cipher {
                let offered_ids: BTreeSet<u16> = state.negotiation.offered_ciphers.iter().map(|c| c.iana_id).collect();
                checker.record_selection(offered_ids, selected.iana_id);
            }
        }
        checker.check_determinism()
    }

    fn select_fallback(
        &self,
        violations: &[PropertyViolation],
        a1_ok: bool,
        a2_ok: bool,
        a3_ok: bool,
        a4_ok: bool,
    ) -> Option<FallbackAction> {
        if a1_ok && a2_ok && a3_ok && a4_ok {
            return None;
        }

        if !a1_ok {
            return Some(FallbackAction::Decompose);
        }
        if !a3_ok {
            return Some(FallbackAction::Fork);
        }
        if !a4_ok {
            return Some(FallbackAction::GenericVeritesting);
        }
        if !a2_ok {
            return Some(FallbackAction::GenericVeritesting);
        }

        Some(FallbackAction::Fork)
    }

    pub fn finite_checker(&self) -> &FiniteOutcomeChecker {
        &self.finite_checker
    }

    pub fn lattice_checker(&self) -> &LatticeChecker {
        &self.lattice_checker
    }

    pub fn monotonicity_checker(&self) -> &MonotonicityChecker {
        &self.monotonicity_checker
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use negsyn_types::{Extension, NegotiationState};

    fn make_state(phase: HandshakePhase, ciphers: &[u16]) -> SymbolicState {
        let mut neg = NegotiationState::new(phase, ProtocolVersion::Tls12);
        neg.offered_ciphers = ciphers.iter().copied().collect();
        SymbolicState::new(1, 0x1000, neg)
    }

    #[test]
    fn test_finite_outcome_check_pass() {
        let config = MergeConfig::default();
        let checker = FiniteOutcomeChecker::new(&config);
        let ciphers: BTreeSet<u16> = (0..100).collect();
        assert!(checker.check_cipher_bound(&ciphers).is_ok());
    }

    #[test]
    fn test_finite_outcome_check_fail() {
        let mut config = MergeConfig::default();
        config.max_cipher_outcomes = 10;
        let checker = FiniteOutcomeChecker::new(&config);
        let ciphers: BTreeSet<u16> = (0..20).collect();
        assert!(checker.check_cipher_bound(&ciphers).is_err());
    }

    #[test]
    fn test_lattice_check_standard_suites() {
        let checker = LatticeChecker::from_standard();
        let offered: BTreeSet<u16> = [0x002F, 0xC02F, 0x009C].into();
        let violations = checker.check_lattice_property(&offered);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_lattice_check_unknown_suite() {
        let checker = LatticeChecker::from_standard();
        let offered: BTreeSet<u16> = [0x002F, 0xFFFF].into();
        let violations = checker.check_lattice_property(&offered);
        assert!(!violations.is_empty());
    }

    #[test]
    fn test_lattice_transitivity() {
        let checker = LatticeChecker::from_standard();
        let offered: BTreeSet<u16> = [0x002F, 0x0035, 0xC02F].into();
        let violations = checker.check_transitivity(&offered);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_monotonicity_acyclicity() {
        let checker = MonotonicityChecker::new();
        let violations = checker.check_acyclicity();
        assert!(violations.is_empty());
    }

    #[test]
    fn test_monotonicity_valid_transition() {
        let checker = MonotonicityChecker::new();
        assert!(checker
            .check_transition(HandshakePhase::ClientHello, HandshakePhase::ServerHello)
            .is_ok());
    }

    #[test]
    fn test_monotonicity_valid_sequence() {
        let checker = MonotonicityChecker::new();
        let phases = vec![
            HandshakePhase::Initial,
            HandshakePhase::ClientHello,
            HandshakePhase::ServerHello,
            HandshakePhase::Certificate,
            HandshakePhase::KeyExchange,
        ];
        let violations = checker.check_sequence(&phases);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_determinism_check_pass() {
        let mut checker = DeterminismChecker::new();
        let offered: BTreeSet<u16> = [0x002F, 0xC02F].into();
        checker.record_selection(offered.clone(), 0xC02F);
        checker.record_selection(offered, 0xC02F); // same selection
        let violations = checker.check_determinism();
        assert!(violations.is_empty());
    }

    #[test]
    fn test_determinism_check_fail() {
        let mut checker = DeterminismChecker::new();
        let offered: BTreeSet<u16> = [0x002F, 0xC02F].into();
        checker.record_selection(offered.clone(), 0xC02F);
        checker.record_selection(offered, 0x002F); // different!
        let violations = checker.check_determinism();
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn test_property_checker_all_pass() {
        let config = MergeConfig::default();
        let checker = PropertyChecker::new(config);

        let s1 = make_state(HandshakePhase::ClientHello, &[0x002F, 0xC02F]);
        let s2 = make_state(HandshakePhase::ClientHello, &[0x002F, 0xC02F]);
        let states: Vec<&SymbolicState> = vec![&s1, &s2];

        let result = checker.check_all(&states);
        assert!(result.a1_satisfied);
        assert!(result.a3_satisfied);
        assert!(result.recommended_fallback.is_none() || result.all_satisfied());
    }

    #[test]
    fn test_property_checker_merge_eligible() {
        let config = MergeConfig::default();
        let checker = PropertyChecker::new(config);

        let s1 = make_state(HandshakePhase::ClientHello, &[0x002F]);
        let states: Vec<&SymbolicState> = vec![&s1];

        let result = checker.check_all(&states);
        assert!(result.merge_eligible());
    }

    #[test]
    fn test_fallback_on_large_cipher_set() {
        let mut config = MergeConfig::default();
        config.max_cipher_outcomes = 2;
        let checker = PropertyChecker::new(config);

        let s = make_state(HandshakePhase::ClientHello, &[0x002F, 0xC02F, 0x009C]);
        let states: Vec<&SymbolicState> = vec![&s];

        let result = checker.check_all(&states);
        assert!(!result.a1_satisfied);
        assert!(result.recommended_fallback.is_some());
    }

    #[test]
    fn test_order_indices_check() {
        let checker = MonotonicityChecker::new();
        let phases = vec![
            HandshakePhase::ClientHello,
            HandshakePhase::ServerHello,
            HandshakePhase::Certificate,
        ];
        let violations = checker.check_order_indices(&phases);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_violation_display() {
        let v = PropertyViolation::FiniteOutcomeExceeded {
            category: OutcomeCategory::CipherSuites,
            actual: 500,
            limit: 350,
        };
        let s = format!("{}", v);
        assert!(s.contains("A1 violated"));
        assert!(s.contains("500"));
    }
}
