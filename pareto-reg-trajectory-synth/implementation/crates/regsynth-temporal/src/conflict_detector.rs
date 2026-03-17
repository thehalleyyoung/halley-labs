//! Regulatory Temporal Conflict Detection
//!
//! Detects *regulatory temporal conflicts*: situations where compliance with
//! regulation A at time t forces non-compliance with regulation B at time t+k.
//!
//! This is a domain-specific graph-theoretic formulation distinct from Bellman's
//! optimal substructure. The key insight is that regulatory obligations carry
//! *causal compliance dependencies* — satisfying one obligation may require
//! actions that preclude satisfying another at a future timestep. These form a
//! temporal constraint graph whose conflict cycles are the irreducible sources
//! of cross-temporal regulatory infeasibility.
//!
//! # Complexity
//!
//! Detecting whether a conflict cycle of length ≤ k exists is polynomial (DFS
//! on the temporal constraint graph, O(|R|·|T|·|E|) where R = regulations,
//! T = timesteps, E = dependency edges). However, finding the *minimum-weight
//! conflict set* — the smallest subset of obligations whose removal eliminates
//! all conflict cycles — is NP-hard.
//!
//! ## NP-hardness proof sketch (reduction from Vertex Cover)
//!
//! Given a Vertex Cover instance G = (V, E), k, construct a temporal constraint
//! graph as follows:
//!   - For each vertex v ∈ V, create obligation O_v active at timesteps {1, 2}.
//!   - For each edge (u, v) ∈ E, add a temporal dependency: compliance with O_u
//!     at t=1 forces a state that conflicts with O_v at t=2, and vice versa.
//!   - The resulting temporal constraint graph has a conflict cycle for each
//!     edge in E. Removing an obligation O_v eliminates all cycles through v,
//!     mirroring the vertex cover structure.
//!   - A minimum conflict set of size ≤ k exists iff G has a vertex cover of
//!     size ≤ k.
//!
//! This establishes NP-hardness of MINIMUM-TEMPORAL-CONFLICT-SET, justifying
//! the greedy/approximation approach in [`ConflictResolution`].

use crate::ObligationId;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};

// ---------------------------------------------------------------------------
// Temporal Constraint Graph
// ---------------------------------------------------------------------------

/// A node in the temporal constraint graph: (obligation, timestep).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TcgNode {
    pub obligation_id: ObligationId,
    pub timestep: usize,
}

impl TcgNode {
    pub fn new(obligation_id: impl Into<String>, timestep: usize) -> Self {
        Self {
            obligation_id: obligation_id.into(),
            timestep,
        }
    }
}

impl std::fmt::Display for TcgNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, t={})", self.obligation_id, self.timestep)
    }
}

/// Edge kinds in the temporal constraint graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TcgEdgeKind {
    /// Compliance with source forces an action that conflicts with target.
    ComplianceForces,
    /// Source is a prerequisite for target (temporal ordering).
    Prerequisite,
    /// Source and target share a resource that cannot simultaneously satisfy both.
    ResourceConflict,
}

impl std::fmt::Display for TcgEdgeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ComplianceForces => write!(f, "forces-violation"),
            Self::Prerequisite => write!(f, "prerequisite"),
            Self::ResourceConflict => write!(f, "resource-conflict"),
        }
    }
}

/// A labelled edge in the temporal constraint graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcgEdge {
    pub kind: TcgEdgeKind,
    /// Human-readable explanation of why this dependency exists.
    pub rationale: String,
    /// Estimated compliance cost of the forced action.
    pub compliance_cost: f64,
}

/// The temporal constraint graph.
///
/// Nodes are (obligation, timestep) pairs. A directed edge from (A, t) to
/// (B, t+k) means: complying with obligation A at time t forces a state that
/// conflicts with obligation B at time t+k.
///
/// Conflict cycles in this graph are the regulatory temporal conflicts.
#[derive(Debug, Clone)]
pub struct TemporalConstraintGraph {
    graph: DiGraph<TcgNode, TcgEdge>,
    node_map: HashMap<(ObligationId, usize), NodeIndex>,
}

impl TemporalConstraintGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
        }
    }

    /// Insert a (obligation, timestep) node if it does not already exist.
    pub fn add_node(&mut self, obligation_id: impl Into<String>, timestep: usize) -> NodeIndex {
        let oid: String = obligation_id.into();
        let key = (oid.clone(), timestep);
        if let Some(&idx) = self.node_map.get(&key) {
            return idx;
        }
        let node = TcgNode::new(oid.clone(), timestep);
        let idx = self.graph.add_node(node);
        self.node_map.insert(key, idx);
        idx
    }

    /// Add a temporal dependency edge. Both endpoints are created if absent.
    pub fn add_dependency(
        &mut self,
        from_obl: impl Into<String>,
        from_t: usize,
        to_obl: impl Into<String>,
        to_t: usize,
        kind: TcgEdgeKind,
        rationale: impl Into<String>,
        compliance_cost: f64,
    ) {
        let src = self.add_node(from_obl, from_t);
        let dst = self.add_node(to_obl, to_t);
        self.graph.add_edge(
            src,
            dst,
            TcgEdge {
                kind,
                rationale: rationale.into(),
                compliance_cost,
            },
        );
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Look up a node index by (obligation_id, timestep).
    pub fn get_node(&self, obligation_id: &str, timestep: usize) -> Option<NodeIndex> {
        self.node_map.get(&(obligation_id.to_string(), timestep)).copied()
    }
}

// ---------------------------------------------------------------------------
// Conflict Certificate — minimal witness for a detected temporal conflict
// ---------------------------------------------------------------------------

/// A conflict certificate: a cycle in the temporal constraint graph witnessing
/// an irreducible regulatory temporal conflict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictCertificate {
    /// Unique identifier for this conflict.
    pub id: String,
    /// The cycle of (obligation, timestep) nodes forming the conflict.
    pub cycle: Vec<TcgNode>,
    /// The edge labels along the cycle explaining each causal link.
    pub edge_labels: Vec<(TcgEdgeKind, String)>,
    /// The distinct obligation IDs involved (the conflict set).
    pub conflict_set: BTreeSet<ObligationId>,
    /// Total compliance cost around the cycle.
    pub total_cost: f64,
    /// Human-readable narrative of the conflict.
    pub narrative: String,
}

impl ConflictCertificate {
    /// The temporal span of the conflict (max timestep − min timestep).
    pub fn temporal_span(&self) -> usize {
        let min_t = self.cycle.iter().map(|n| n.timestep).min().unwrap_or(0);
        let max_t = self.cycle.iter().map(|n| n.timestep).max().unwrap_or(0);
        max_t - min_t
    }

    /// Number of distinct obligations in the conflict.
    pub fn conflict_size(&self) -> usize {
        self.conflict_set.len()
    }
}

impl std::fmt::Display for ConflictCertificate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ConflictCertificate[{}]: {} obligations over {} timesteps — {}",
            self.id,
            self.conflict_size(),
            self.temporal_span(),
            self.narrative
        )
    }
}

// ---------------------------------------------------------------------------
// Temporal Conflict Detector
// ---------------------------------------------------------------------------

/// Detects regulatory temporal conflicts by finding cycles in the temporal
/// constraint graph.
///
/// A regulatory temporal conflict is a cycle where:
///   (A, t₁) →[forces-violation] (B, t₂) →[...] → (A, t₁+k)
///
/// meaning that compliance with regulation A at t₁ causally forces
/// non-compliance with regulation B at t₂, which propagates until regulation A
/// is itself violated at a later timestep t₁+k.
pub struct TemporalConflictDetector {
    graph: TemporalConstraintGraph,
    max_cycle_length: usize,
}

impl TemporalConflictDetector {
    pub fn new(graph: TemporalConstraintGraph) -> Self {
        Self {
            graph,
            max_cycle_length: 64,
        }
    }

    pub fn with_max_cycle_length(mut self, max_len: usize) -> Self {
        self.max_cycle_length = max_len;
        self
    }

    /// Detect all conflict cycles up to `max_cycle_length` via DFS.
    ///
    /// Returns a set of [`ConflictCertificate`]s, one per distinct cycle.
    /// Cycles that are rotations of each other are deduplicated.
    pub fn detect_conflicts(&self) -> Vec<ConflictCertificate> {
        let g = &self.graph.graph;
        let mut all_cycles: Vec<Vec<NodeIndex>> = Vec::new();
        let mut visited_global = HashSet::new();

        // Johnson-style bounded DFS from each node.
        for start_idx in g.node_indices() {
            let mut path: Vec<NodeIndex> = vec![start_idx];
            let mut on_stack: HashSet<NodeIndex> = HashSet::new();
            on_stack.insert(start_idx);

            self.dfs_cycles(
                start_idx,
                start_idx,
                &mut path,
                &mut on_stack,
                &mut all_cycles,
            );
            visited_global.insert(start_idx);
        }

        // Deduplicate rotations: canonical form = rotation with smallest node.
        let mut seen: HashSet<Vec<NodeIndex>> = HashSet::new();
        let mut certificates = Vec::new();

        for cycle in all_cycles {
            let canon = Self::canonical_cycle(&cycle);
            if seen.contains(&canon) {
                continue;
            }
            seen.insert(canon);

            if let Some(cert) = self.build_certificate(&cycle, certificates.len()) {
                certificates.push(cert);
            }
        }

        certificates
    }

    /// Bounded DFS to enumerate simple cycles from `start`.
    fn dfs_cycles(
        &self,
        start: NodeIndex,
        current: NodeIndex,
        path: &mut Vec<NodeIndex>,
        on_stack: &mut HashSet<NodeIndex>,
        result: &mut Vec<Vec<NodeIndex>>,
    ) {
        if path.len() > self.max_cycle_length {
            return;
        }

        let g = &self.graph.graph;
        for edge in g.edges(current) {
            let next = edge.target();
            if next == start && path.len() >= 2 {
                // Found a cycle back to start.
                result.push(path.clone());
            } else if !on_stack.contains(&next) && path.len() < self.max_cycle_length {
                on_stack.insert(next);
                path.push(next);
                self.dfs_cycles(start, next, path, on_stack, result);
                path.pop();
                on_stack.remove(&next);
            }
        }
    }

    /// Canonical form of a cycle: rotate so the smallest NodeIndex is first.
    fn canonical_cycle(cycle: &[NodeIndex]) -> Vec<NodeIndex> {
        if cycle.is_empty() {
            return Vec::new();
        }
        let min_pos = cycle
            .iter()
            .enumerate()
            .min_by_key(|(_, idx)| idx.index())
            .map(|(pos, _)| pos)
            .unwrap_or(0);
        let mut canonical = Vec::with_capacity(cycle.len());
        canonical.extend_from_slice(&cycle[min_pos..]);
        canonical.extend_from_slice(&cycle[..min_pos]);
        canonical
    }

    /// Build a [`ConflictCertificate`] from a node-index cycle.
    fn build_certificate(&self, cycle: &[NodeIndex], seq: usize) -> Option<ConflictCertificate> {
        let g = &self.graph.graph;
        let nodes: Vec<TcgNode> = cycle.iter().map(|&idx| g[idx].clone()).collect();
        let mut edge_labels = Vec::new();
        let mut total_cost = 0.0;

        for i in 0..cycle.len() {
            let from = cycle[i];
            let to = cycle[(i + 1) % cycle.len()];
            if let Some(edge) = g.edges(from).find(|e| e.target() == to) {
                let w = edge.weight();
                edge_labels.push((w.kind.clone(), w.rationale.clone()));
                total_cost += w.compliance_cost;
            }
        }

        let conflict_set: BTreeSet<ObligationId> =
            nodes.iter().map(|n| n.obligation_id.clone()).collect();

        let narrative = build_narrative(&nodes, &edge_labels);

        Some(ConflictCertificate {
            id: format!("TC-{:03}", seq + 1),
            cycle: nodes,
            edge_labels,
            conflict_set,
            total_cost,
            narrative,
        })
    }
}

/// Build a human-readable narrative for a conflict cycle.
fn build_narrative(nodes: &[TcgNode], edges: &[(TcgEdgeKind, String)]) -> String {
    if nodes.is_empty() {
        return String::new();
    }
    let mut parts = Vec::new();
    for i in 0..nodes.len() {
        let next = (i + 1) % nodes.len();
        let edge_desc = edges
            .get(i)
            .map(|(_, r)| r.as_str())
            .unwrap_or("unknown link");
        parts.push(format!(
            "compliance with {} at t={} {} {}",
            nodes[i].obligation_id,
            nodes[i].timestep,
            if i < nodes.len() - 1 {
                "forces violation of"
            } else {
                "cycles back to violate"
            },
            if next < nodes.len() {
                format!(
                    "{} at t={} [{}]",
                    nodes[next].obligation_id, nodes[next].timestep, edge_desc
                )
            } else {
                "the initial obligation".to_string()
            },
        ));
    }
    parts.join("; ")
}

// ---------------------------------------------------------------------------
// Conflict Resolution — Pareto-optimal relaxations
// ---------------------------------------------------------------------------

/// A proposed relaxation that breaks one or more conflict cycles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relaxation {
    /// Obligations to relax (weaken or defer).
    pub relaxed_obligations: BTreeSet<ObligationId>,
    /// The conflict certificates resolved by this relaxation.
    pub resolves: Vec<String>,
    /// Estimated compliance loss from the relaxation (lower is better).
    pub compliance_loss: f64,
    /// Description of the relaxation strategy.
    pub description: String,
}

/// Suggests Pareto-optimal relaxations that break conflict cycles with minimum
/// compliance loss.
///
/// Since finding the minimum-weight conflict set is NP-hard (see module docs),
/// we use a greedy weighted set-cover approximation:
///   1. Weight each obligation by its involvement frequency across cycles.
///   2. Greedily select the obligation whose removal eliminates the most cycles
///      per unit of compliance cost.
///   3. Repeat until all cycles are broken.
///   4. Enumerate alternative relaxation frontiers by varying the greedy
///      tie-breaking order.
pub struct ConflictResolution;

impl ConflictResolution {
    /// Compute a greedy-approximate minimum-cost relaxation set.
    ///
    /// `obligation_costs` maps obligation IDs to their compliance importance
    /// (higher = more costly to relax).
    pub fn suggest_relaxations(
        certificates: &[ConflictCertificate],
        obligation_costs: &HashMap<ObligationId, f64>,
    ) -> Vec<Relaxation> {
        if certificates.is_empty() {
            return Vec::new();
        }

        let mut remaining: HashSet<usize> = (0..certificates.len()).collect();
        let mut relaxed = BTreeSet::new();
        let mut resolves = Vec::new();

        // Greedy weighted set cover.
        while !remaining.is_empty() {
            // Count how many remaining cycles each obligation covers.
            let mut coverage: HashMap<&ObligationId, Vec<usize>> = HashMap::new();
            for &ci in &remaining {
                for oid in &certificates[ci].conflict_set {
                    coverage.entry(oid).or_default().push(ci);
                }
            }

            // Pick the obligation with best coverage-per-cost ratio.
            let best = coverage
                .iter()
                .max_by(|(oid_a, cov_a), (oid_b, cov_b)| {
                    let cost_a = obligation_costs.get(**oid_a).copied().unwrap_or(1.0);
                    let cost_b = obligation_costs.get(**oid_b).copied().unwrap_or(1.0);
                    let ratio_a = cov_a.len() as f64 / cost_a;
                    let ratio_b = cov_b.len() as f64 / cost_b;
                    ratio_a
                        .partial_cmp(&ratio_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(oid, covered)| ((*oid).clone(), covered.clone()));

            if let Some((oid, covered)) = best {
                relaxed.insert(oid.clone());
                for ci in &covered {
                    if remaining.remove(ci) {
                        resolves.push(certificates[*ci].id.clone());
                    }
                }
            } else {
                break;
            }
        }

        let compliance_loss: f64 = relaxed
            .iter()
            .map(|oid| obligation_costs.get(oid).copied().unwrap_or(1.0))
            .sum();

        let description = format!(
            "Relax {} obligation(s) [{}] to break {} conflict cycle(s)",
            relaxed.len(),
            relaxed.iter().cloned().collect::<Vec<_>>().join(", "),
            resolves.len(),
        );

        vec![Relaxation {
            relaxed_obligations: relaxed,
            resolves,
            compliance_loss,
            description,
        }]
    }

    /// Enumerate alternative relaxation frontiers by trying each obligation
    /// in the union of all conflict sets as a forced first pick, then running
    /// greedy on the remainder. Returns the Pareto frontier over
    /// (compliance_loss, number_of_relaxed_obligations).
    pub fn pareto_relaxation_frontier(
        certificates: &[ConflictCertificate],
        obligation_costs: &HashMap<ObligationId, f64>,
    ) -> Vec<Relaxation> {
        if certificates.is_empty() {
            return Vec::new();
        }

        // Gather all candidate obligations.
        let all_obls: BTreeSet<ObligationId> = certificates
            .iter()
            .flat_map(|c| c.conflict_set.iter().cloned())
            .collect();

        let mut candidates: Vec<Relaxation> = Vec::new();

        // Try forcing each obligation as the first relaxation.
        for forced in &all_obls {
            let mut remaining: HashSet<usize> = (0..certificates.len()).collect();
            let mut relaxed: BTreeSet<ObligationId> = BTreeSet::new();
            let mut resolved_ids = Vec::new();

            // Apply forced relaxation.
            relaxed.insert(forced.clone());
            let covered: Vec<usize> = remaining
                .iter()
                .filter(|&&ci| certificates[ci].conflict_set.contains(forced))
                .copied()
                .collect();
            for ci in covered {
                remaining.remove(&ci);
                resolved_ids.push(certificates[ci].id.clone());
            }

            // Greedy on remainder.
            while !remaining.is_empty() {
                let mut coverage: HashMap<&ObligationId, Vec<usize>> = HashMap::new();
                for &ci in &remaining {
                    for oid in &certificates[ci].conflict_set {
                        coverage.entry(oid).or_default().push(ci);
                    }
                }

                let best = coverage
                    .iter()
                    .max_by(|(oid_a, cov_a), (oid_b, cov_b)| {
                        let cost_a = obligation_costs.get(**oid_a).copied().unwrap_or(1.0);
                        let cost_b = obligation_costs.get(**oid_b).copied().unwrap_or(1.0);
                        let ratio_a = cov_a.len() as f64 / cost_a;
                        let ratio_b = cov_b.len() as f64 / cost_b;
                        ratio_a
                            .partial_cmp(&ratio_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(oid, covered)| ((*oid).clone(), covered.clone()));

                if let Some((oid, covered)) = best {
                    relaxed.insert(oid);
                    for ci in covered {
                        if remaining.remove(&ci) {
                            resolved_ids.push(certificates[ci].id.clone());
                        }
                    }
                } else {
                    break;
                }
            }

            let compliance_loss: f64 = relaxed
                .iter()
                .map(|oid| obligation_costs.get(oid).copied().unwrap_or(1.0))
                .sum();

            let description = format!(
                "Relax [{}] (forced: {})",
                relaxed.iter().cloned().collect::<Vec<_>>().join(", "),
                forced,
            );

            candidates.push(Relaxation {
                relaxed_obligations: relaxed,
                resolves: resolved_ids,
                compliance_loss,
                description,
            });
        }

        // Filter to Pareto frontier over (compliance_loss, |relaxed|).
        let mut frontier = Vec::new();
        for (i, c) in candidates.iter().enumerate() {
            let dominated = candidates.iter().enumerate().any(|(j, other)| {
                i != j
                    && other.compliance_loss < c.compliance_loss
                    && other.relaxed_obligations.len() <= c.relaxed_obligations.len()
            });
            if !dominated {
                frontier.push(c.clone());
            }
        }

        frontier.sort_by(|a, b| {
            a.compliance_loss
                .partial_cmp(&b.compliance_loss)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        frontier.dedup_by(|a, b| a.relaxed_obligations == b.relaxed_obligations);
        frontier
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the canonical EU-AI-Act-Art12 vs GDPR-Art5(1)(c) temporal conflict:
    /// Logging at t=1 forces data retention that violates minimisation at t=2,
    /// and minimisation at t=2 forces deletion that violates audit-readiness
    /// (logging) at t=3.
    fn build_logging_minimisation_graph() -> TemporalConstraintGraph {
        let mut tcg = TemporalConstraintGraph::new();
        // Cycle: logging@t1 → minimisation@t2 → logging@t3
        tcg.add_dependency(
            "EU-AIA-Art12-logging",
            1,
            "GDPR-Art5c-minimisation",
            2,
            TcgEdgeKind::ComplianceForces,
            "Art.12 logging retains personal data, violating Art.5(1)(c) minimisation",
            3.0,
        );
        tcg.add_dependency(
            "GDPR-Art5c-minimisation",
            2,
            "EU-AIA-Art12-logging",
            3,
            TcgEdgeKind::ComplianceForces,
            "Enforcing minimisation deletes records needed for Art.12 audit trail",
            4.0,
        );
        tcg
    }

    #[test]
    fn test_graph_construction() {
        let tcg = build_logging_minimisation_graph();
        assert_eq!(tcg.node_count(), 3);
        assert_eq!(tcg.edge_count(), 2);
    }

    #[test]
    fn test_detect_simple_cycle() {
        let mut tcg = TemporalConstraintGraph::new();
        // Create a 2-node cycle: A@t1 → B@t2 → A@t1
        tcg.add_dependency("A", 1, "B", 2, TcgEdgeKind::ComplianceForces, "A forces B violation", 1.0);
        tcg.add_dependency("B", 2, "A", 1, TcgEdgeKind::ComplianceForces, "B forces A violation", 1.0);

        let detector = TemporalConflictDetector::new(tcg);
        let certs = detector.detect_conflicts();

        assert!(!certs.is_empty(), "Should detect at least one conflict cycle");
        let cert = &certs[0];
        assert_eq!(cert.conflict_set.len(), 2);
        assert!(cert.conflict_set.contains("A"));
        assert!(cert.conflict_set.contains("B"));
    }

    #[test]
    fn test_no_conflicts_in_acyclic_graph() {
        let mut tcg = TemporalConstraintGraph::new();
        tcg.add_dependency("A", 1, "B", 2, TcgEdgeKind::Prerequisite, "A before B", 1.0);
        tcg.add_dependency("B", 2, "C", 3, TcgEdgeKind::Prerequisite, "B before C", 1.0);

        let detector = TemporalConflictDetector::new(tcg);
        let certs = detector.detect_conflicts();

        assert!(certs.is_empty(), "Acyclic graph should have no conflict cycles");
    }

    #[test]
    fn test_three_regulation_conflict_cycle() {
        let mut tcg = TemporalConstraintGraph::new();
        // A@t1 → B@t2 → C@t3 → A@t4
        tcg.add_dependency("A", 1, "B", 2, TcgEdgeKind::ComplianceForces, "A→B", 2.0);
        tcg.add_dependency("B", 2, "C", 3, TcgEdgeKind::ComplianceForces, "B→C", 3.0);
        tcg.add_dependency("C", 3, "A", 1, TcgEdgeKind::ComplianceForces, "C→A", 1.0);

        let detector = TemporalConflictDetector::new(tcg);
        let certs = detector.detect_conflicts();

        assert!(!certs.is_empty());
        let cert = &certs[0];
        assert_eq!(cert.conflict_set.len(), 3);
        assert_eq!(cert.temporal_span(), 2); // t=1 to t=3
    }

    #[test]
    fn test_conflict_certificate_narrative() {
        let tcg = build_logging_minimisation_graph();
        // Add closing edge to form cycle.
        let mut tcg = tcg;
        tcg.add_dependency(
            "EU-AIA-Art12-logging",
            3,
            "EU-AIA-Art12-logging",
            1,
            TcgEdgeKind::ResourceConflict,
            "Audit obligation persists across periods",
            0.5,
        );

        let detector = TemporalConflictDetector::new(tcg);
        let certs = detector.detect_conflicts();

        assert!(!certs.is_empty());
        for cert in &certs {
            assert!(!cert.narrative.is_empty());
            assert!(cert.id.starts_with("TC-"));
        }
    }

    #[test]
    fn test_conflict_resolution_greedy() {
        let mut tcg = TemporalConstraintGraph::new();
        // Two overlapping cycles: A→B→A and A→C→A.
        tcg.add_dependency("A", 1, "B", 2, TcgEdgeKind::ComplianceForces, "A→B", 1.0);
        tcg.add_dependency("B", 2, "A", 1, TcgEdgeKind::ComplianceForces, "B→A", 1.0);
        tcg.add_dependency("A", 1, "C", 2, TcgEdgeKind::ComplianceForces, "A→C", 1.0);
        tcg.add_dependency("C", 2, "A", 1, TcgEdgeKind::ComplianceForces, "C→A", 1.0);

        let detector = TemporalConflictDetector::new(tcg);
        let certs = detector.detect_conflicts();

        let mut costs = HashMap::new();
        costs.insert("A".to_string(), 1.0);
        costs.insert("B".to_string(), 5.0);
        costs.insert("C".to_string(), 5.0);

        let relaxations = ConflictResolution::suggest_relaxations(&certs, &costs);
        assert!(!relaxations.is_empty());
        // Relaxing A (cost 1.0) should be preferred since it covers both cycles.
        assert!(relaxations[0].relaxed_obligations.contains("A"));
    }

    #[test]
    fn test_pareto_frontier() {
        let mut tcg = TemporalConstraintGraph::new();
        tcg.add_dependency("A", 1, "B", 2, TcgEdgeKind::ComplianceForces, "A→B", 1.0);
        tcg.add_dependency("B", 2, "A", 1, TcgEdgeKind::ComplianceForces, "B→A", 1.0);

        let detector = TemporalConflictDetector::new(tcg);
        let certs = detector.detect_conflicts();

        let mut costs = HashMap::new();
        costs.insert("A".to_string(), 2.0);
        costs.insert("B".to_string(), 3.0);

        let frontier = ConflictResolution::pareto_relaxation_frontier(&certs, &costs);
        // Both {A} and {B} are single-obligation relaxations; {A} dominates on cost.
        assert!(!frontier.is_empty());
        assert!(frontier[0].compliance_loss <= 3.0);
    }

    #[test]
    fn test_empty_graph_no_conflicts() {
        let tcg = TemporalConstraintGraph::new();
        let detector = TemporalConflictDetector::new(tcg);
        let certs = detector.detect_conflicts();
        assert!(certs.is_empty());
    }
}
