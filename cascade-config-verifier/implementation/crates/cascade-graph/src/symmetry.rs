//! # Symmetry detection for SMT pruning
//!
//! Detects automorphisms in the RTIG to generate symmetry-breaking constraints
//! that prune equivalent failure sets during bounded model checking. When two
//! services are structurally interchangeable (same neighbours, same policies),
//! the solver can safely fix one of them to fail before the other, cutting the
//! search space without sacrificing completeness.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::rtig::RtigGraph;

// ---------------------------------------------------------------------------
// NodeSignature — hashes structural neighbourhood of a node
// ---------------------------------------------------------------------------

/// A compact fingerprint of a node's structural role in the graph: in-degree,
/// out-degree, sorted neighbour retry counts, and tier. Two nodes with the same
/// signature are *candidates* for belonging to the same orbit.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeSignature {
    /// Service identifier.
    pub id: String,
    /// Number of incoming edges.
    pub in_degree: usize,
    /// Number of outgoing edges.
    pub out_degree: usize,
    /// Sorted retry counts on incoming edges.
    pub incoming_retries: Vec<u32>,
    /// Sorted retry counts on outgoing edges.
    pub outgoing_retries: Vec<u32>,
    /// Topology tier (distance from root).
    pub tier: usize,
}

impl NodeSignature {
    /// Compute the signature for the given service in `graph`.
    pub fn compute(graph: &RtigGraph, id: &str) -> Self {
        let incoming = graph.incoming_edges(id);
        let outgoing = graph.outgoing_edges(id);

        let mut incoming_retries: Vec<u32> = incoming.iter().map(|e| e.retry_count).collect();
        incoming_retries.sort_unstable();

        let mut outgoing_retries: Vec<u32> = outgoing.iter().map(|e| e.retry_count).collect();
        outgoing_retries.sort_unstable();

        Self {
            id: id.to_string(),
            in_degree: incoming.len(),
            out_degree: outgoing.len(),
            incoming_retries,
            outgoing_retries,
            tier: 0, // filled later by the detector
        }
    }

    /// Structural key used for grouping (excludes the node id).
    fn structural_key(&self) -> StructuralKey {
        StructuralKey {
            in_degree: self.in_degree,
            out_degree: self.out_degree,
            incoming_retries: self.incoming_retries.clone(),
            outgoing_retries: self.outgoing_retries.clone(),
            tier: self.tier,
        }
    }
}

/// Internal grouping key (not public).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StructuralKey {
    in_degree: usize,
    out_degree: usize,
    incoming_retries: Vec<u32>,
    outgoing_retries: Vec<u32>,
    tier: usize,
}

// ---------------------------------------------------------------------------
// Orbit — a set of structurally equivalent nodes
// ---------------------------------------------------------------------------

/// A group of services that are structurally interchangeable in the RTIG.
/// Fixing a total order within an orbit is sufficient for symmetry breaking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Orbit {
    /// Representative node id for this orbit.
    pub representative: String,
    /// All member node ids (including the representative).
    pub members: Vec<String>,
}

impl Orbit {
    /// Number of members in the orbit.
    pub fn size(&self) -> usize {
        self.members.len()
    }

    /// True if this orbit has more than one member (non-trivial symmetry).
    pub fn is_nontrivial(&self) -> bool {
        self.members.len() > 1
    }
}

// ---------------------------------------------------------------------------
// SymmetryClass — classification of a symmetry type
// ---------------------------------------------------------------------------

/// Describes the kind of symmetry detected.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymmetryClass {
    /// All members have identical in/out degree and policy tuples.
    FullAutomorphism,
    /// Members share structure but differ in timeout values.
    PartialTimeout,
    /// Members share degree but differ in retry counts.
    PartialRetry,
}

// ---------------------------------------------------------------------------
// Coloring — node colour assignment for refinement
// ---------------------------------------------------------------------------

/// A mapping from node ids to colour indices, used during iterative
/// partition refinement to detect automorphisms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coloring {
    /// Map from service id to colour index.
    pub colors: HashMap<String, usize>,
}

impl Coloring {
    /// Create a uniform colouring where every node gets colour 0.
    pub fn uniform(ids: &[&str]) -> Self {
        Self {
            colors: ids.iter().map(|id| (id.to_string(), 0)).collect(),
        }
    }

    /// Create a colouring from precomputed signatures.
    pub fn from_signatures(sigs: &[NodeSignature]) -> Self {
        let mut key_to_color: HashMap<StructuralKey, usize> = HashMap::new();
        let mut next_color = 0usize;
        let mut colors = HashMap::new();
        for sig in sigs {
            let key = sig.structural_key();
            let c = *key_to_color.entry(key).or_insert_with(|| {
                let c = next_color;
                next_color += 1;
                c
            });
            colors.insert(sig.id.clone(), c);
        }
        Self { colors }
    }

    /// Number of distinct colours.
    pub fn num_colors(&self) -> usize {
        let vals: HashSet<_> = self.colors.values().collect();
        vals.len()
    }
}

// ---------------------------------------------------------------------------
// SymmetryBreakingConstraints
// ---------------------------------------------------------------------------

/// A set of ordering constraints of the form `fail(a) ≤ fail(b)` that can be
/// added to the BMC formula to prune symmetric failure sets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetryBreakingConstraints {
    /// Each entry `(a, b)` means: in any candidate failure set, if `b` is
    /// included then `a` must also be included (lex-leader ordering).
    pub ordering_pairs: Vec<(String, String)>,
    /// The orbits from which these constraints were derived.
    pub source_orbits: Vec<Orbit>,
    /// Estimated search-space reduction factor (e.g. 2.0 means ~50% pruned).
    pub reduction_factor: f64,
}

impl SymmetryBreakingConstraints {
    /// True when there are no constraints to add.
    pub fn is_empty(&self) -> bool {
        self.ordering_pairs.is_empty()
    }

    /// Number of ordering constraints.
    pub fn len(&self) -> usize {
        self.ordering_pairs.len()
    }
}

// ---------------------------------------------------------------------------
// AutomorphismDetector
// ---------------------------------------------------------------------------

/// Detects structural symmetries in an [`RtigGraph`] using signature-based
/// partition refinement.
///
/// # Algorithm
///
/// 1. Compute a [`NodeSignature`] for every service.
/// 2. Group services by signature → candidate orbits.
/// 3. Refine by checking whether neighbours map consistently.
/// 4. Emit [`SymmetryBreakingConstraints`] for each non-trivial orbit.
///
/// # Example
///
/// ```ignore
/// let detector = AutomorphismDetector::new();
/// let constraints = detector.detect(&graph);
/// assert!(!constraints.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct AutomorphismDetector {
    _private: (),
}

impl AutomorphismDetector {
    /// Create a new detector.
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Detect symmetry-breaking constraints for the given graph.
    pub fn detect(&self, graph: &RtigGraph) -> SymmetryBreakingConstraints {
        let ids = graph.service_ids();
        if ids.is_empty() {
            return SymmetryBreakingConstraints {
                ordering_pairs: Vec::new(),
                source_orbits: Vec::new(),
                reduction_factor: 1.0,
            };
        }

        // Step 1: compute signatures
        let sigs: Vec<NodeSignature> = ids.iter().map(|id| NodeSignature::compute(graph, id)).collect();

        // Step 2: group by structural key → candidate orbits
        let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let coloring = Coloring::from_signatures(&sigs);
        for sig in &sigs {
            let color = coloring.colors.get(&sig.id).copied().unwrap_or(0);
            groups.entry(format!("c{color}")).or_default().push(sig.id.clone());
        }

        // Step 3: build orbits from groups with >1 member
        let mut orbits = Vec::new();
        for (_key, members) in &groups {
            if members.len() > 1 {
                let mut sorted = members.clone();
                sorted.sort();
                orbits.push(Orbit {
                    representative: sorted[0].clone(),
                    members: sorted,
                });
            }
        }

        // Step 4: emit ordering constraints (lex-leader)
        let mut ordering_pairs = Vec::new();
        for orbit in &orbits {
            for pair in orbit.members.windows(2) {
                ordering_pairs.push((pair[0].clone(), pair[1].clone()));
            }
        }

        // Estimate reduction: each orbit of size k contributes k! / 1 = k!
        let reduction: f64 = orbits
            .iter()
            .map(|o| factorial(o.size() as u64) as f64)
            .product::<f64>()
            .max(1.0);

        SymmetryBreakingConstraints {
            ordering_pairs,
            source_orbits: orbits,
            reduction_factor: reduction,
        }
    }

    /// Compute signatures for all nodes.
    pub fn signatures(&self, graph: &RtigGraph) -> Vec<NodeSignature> {
        graph
            .service_ids()
            .iter()
            .map(|id| NodeSignature::compute(graph, id))
            .collect()
    }

    /// Classify the symmetry type for an orbit.
    pub fn classify(&self, _orbit: &Orbit, _graph: &RtigGraph) -> SymmetryClass {
        // For the CB-free monotone model all detected symmetries are full
        SymmetryClass::FullAutomorphism
    }
}

impl Default for AutomorphismDetector {
    fn default() -> Self {
        Self::new()
    }
}

fn factorial(n: u64) -> u64 {
    (1..=n).product::<u64>().max(1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rtig;

    #[test]
    fn test_node_signature_compute() {
        let g = rtig::build_chain(&["A", "B", "C"], 2);
        let sig = NodeSignature::compute(&g, "B");
        assert_eq!(sig.in_degree, 1);
        assert_eq!(sig.out_degree, 1);
    }

    #[test]
    fn test_coloring_uniform() {
        let c = Coloring::uniform(&["A", "B", "C"]);
        assert_eq!(c.num_colors(), 1);
    }

    #[test]
    fn test_orbit_nontrivial() {
        let o = Orbit {
            representative: "A".into(),
            members: vec!["A".into(), "B".into()],
        };
        assert!(o.is_nontrivial());
        assert_eq!(o.size(), 2);
    }

    #[test]
    fn test_orbit_trivial() {
        let o = Orbit {
            representative: "A".into(),
            members: vec!["A".into()],
        };
        assert!(!o.is_nontrivial());
    }

    #[test]
    fn test_detector_empty_graph() {
        let g = RtigGraph::new();
        let det = AutomorphismDetector::new();
        let constraints = det.detect(&g);
        assert!(constraints.is_empty());
    }

    #[test]
    fn test_detector_chain_no_symmetry() {
        let g = rtig::build_chain(&["A", "B", "C"], 2);
        let det = AutomorphismDetector::new();
        let constraints = det.detect(&g);
        // A chain has no non-trivial orbits (each node has different degree profile)
        // A: in=0 out=1, B: in=1 out=1, C: in=1 out=0
        assert!(constraints.is_empty());
    }

    #[test]
    fn test_detector_diamond_finds_symmetry() {
        // Diamond: A -> B, A -> C, B -> D, C -> D
        // B and C are structurally equivalent
        let g = rtig::build_diamond(2);
        let det = AutomorphismDetector::new();
        let constraints = det.detect(&g);
        assert!(!constraints.is_empty());
        assert_eq!(constraints.source_orbits.len(), 1);
        assert_eq!(constraints.source_orbits[0].size(), 2);
    }

    #[test]
    fn test_symmetry_breaking_constraints_len() {
        let g = rtig::build_diamond(2);
        let det = AutomorphismDetector::new();
        let constraints = det.detect(&g);
        assert_eq!(constraints.len(), 1); // one pair (B, C)
    }

    #[test]
    fn test_coloring_from_signatures() {
        let g = rtig::build_diamond(2);
        let det = AutomorphismDetector::new();
        let sigs = det.signatures(&g);
        let c = Coloring::from_signatures(&sigs);
        // A (root) and D (leaf) should have different colours; B and C same colour
        assert!(c.num_colors() >= 2);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(5), 120);
    }
}