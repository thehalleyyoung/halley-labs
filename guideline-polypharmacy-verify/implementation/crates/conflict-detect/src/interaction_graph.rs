//! Drug interaction graph construction and analysis using petgraph.
//!
//! Builds a directed graph where nodes are drugs and edges represent
//! interactions.  Provides clique detection, topological sorting by
//! severity, cascade analysis, and summary analytics.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};

use crate::types::{
    ConflictSeverity, ConfirmedConflict, DrugId, DrugInfo, InteractionType, SafetyVerdict,
};

// ---------------------------------------------------------------------------
// Node / Edge types
// ---------------------------------------------------------------------------

/// A node in the interaction graph representing a single drug.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugNode {
    pub drug_id: DrugId,
    pub name: String,
    pub therapeutic_class: String,
    pub half_life_hours: f64,
    pub cyp_enzymes: Vec<String>,
}

impl DrugNode {
    pub fn from_drug_info(info: &DrugInfo) -> Self {
        Self {
            drug_id: info.id.clone(),
            name: info.name.clone(),
            therapeutic_class: info.therapeutic_class.clone(),
            half_life_hours: info.half_life_hours,
            cyp_enzymes: info.cyp_enzymes.clone(),
        }
    }
}

impl fmt::Display for DrugNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name, self.drug_id)
    }
}

/// An edge in the interaction graph representing an interaction between two drugs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEdge {
    pub interaction_type: InteractionType,
    pub severity: ConflictSeverity,
    pub confidence: f64,
    pub description: String,
}

impl InteractionEdge {
    pub fn new(
        interaction_type: InteractionType,
        severity: ConflictSeverity,
        confidence: f64,
    ) -> Self {
        let description = interaction_type.description();
        Self {
            interaction_type,
            severity,
            confidence,
            description,
        }
    }

    /// Numeric weight combining severity and confidence.
    pub fn weight(&self) -> f64 {
        self.severity.numeric_score() * self.confidence
    }
}

impl fmt::Display for InteractionEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}] (conf={:.0}%)",
            self.description,
            self.severity,
            self.confidence * 100.0
        )
    }
}

// ---------------------------------------------------------------------------
// InteractionChain
// ---------------------------------------------------------------------------

/// A chain (path) of cascading drug interactions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionChain {
    pub drugs: Vec<DrugId>,
    pub interactions: Vec<InteractionType>,
    pub cumulative_severity: f64,
    pub length: usize,
}

impl InteractionChain {
    /// Aggregate severity from individual interactions.
    pub fn compute_severity(interactions: &[InteractionEdge]) -> f64 {
        interactions
            .iter()
            .map(|e| e.weight())
            .sum::<f64>()
            .min(10.0)
    }

    /// Return the maximum individual severity in the chain.
    pub fn max_severity(&self) -> ConflictSeverity {
        ConflictSeverity::from_score(self.cumulative_severity)
    }
}

// ---------------------------------------------------------------------------
// VisualizationData
// ---------------------------------------------------------------------------

/// Serializable snapshot of the graph for external visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationData {
    pub nodes: Vec<VisualizationNode>,
    pub edges: Vec<VisualizationEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationNode {
    pub id: String,
    pub label: String,
    pub group: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationEdge {
    pub source: String,
    pub target: String,
    pub label: String,
    pub weight: f64,
}

// ---------------------------------------------------------------------------
// GraphAnalytics
// ---------------------------------------------------------------------------

/// Summary statistics for an interaction graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAnalytics {
    pub node_count: usize,
    pub edge_count: usize,
    pub density: f64,
    pub max_in_degree: usize,
    pub max_out_degree: usize,
    pub avg_degree: f64,
    pub connected_components: usize,
    pub degree_distribution: HashMap<usize, usize>,
    pub most_connected_drug: Option<DrugId>,
    pub severity_histogram: HashMap<String, usize>,
}

// ---------------------------------------------------------------------------
// CascadeAnalyzer
// ---------------------------------------------------------------------------

/// Analyzes cascading (transitive) drug interactions.
pub struct CascadeAnalyzer<'a> {
    graph: &'a InteractionGraph,
    max_depth: usize,
}

impl<'a> CascadeAnalyzer<'a> {
    pub fn new(graph: &'a InteractionGraph, max_depth: usize) -> Self {
        Self { graph, max_depth }
    }

    /// Find all interaction chains starting from a given drug.
    pub fn find_chains_from(&self, drug_id: &DrugId) -> Vec<InteractionChain> {
        let start = match self.graph.node_index(drug_id) {
            Some(idx) => idx,
            None => return Vec::new(),
        };

        let mut chains = Vec::new();
        let mut stack: Vec<(NodeIndex, Vec<DrugId>, Vec<InteractionEdge>, HashSet<NodeIndex>)> =
            Vec::new();

        let mut visited = HashSet::new();
        visited.insert(start);
        stack.push((
            start,
            vec![drug_id.clone()],
            Vec::new(),
            visited,
        ));

        while let Some((current, path, edges, vis)) = stack.pop() {
            if path.len() > 1 {
                let cum_sev = InteractionChain::compute_severity(&edges);
                chains.push(InteractionChain {
                    drugs: path.clone(),
                    interactions: edges.iter().map(|e| e.interaction_type.clone()).collect(),
                    cumulative_severity: cum_sev,
                    length: path.len() - 1,
                });
            }

            if path.len() - 1 >= self.max_depth {
                continue;
            }

            for edge_ref in self.graph.inner.edges_directed(current, Direction::Outgoing) {
                let target = edge_ref.target();
                if vis.contains(&target) {
                    continue;
                }
                let target_node = &self.graph.inner[target];
                let mut new_path = path.clone();
                new_path.push(target_node.drug_id.clone());
                let mut new_edges = edges.clone();
                new_edges.push(edge_ref.weight().clone());
                let mut new_vis = vis.clone();
                new_vis.insert(target);
                stack.push((target, new_path, new_edges, new_vis));
            }
        }

        chains.sort_by(|a, b| {
            b.cumulative_severity
                .partial_cmp(&a.cumulative_severity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        chains
    }

    /// Find all chains across all drugs, sorted by cumulative severity.
    pub fn find_all_chains(&self) -> Vec<InteractionChain> {
        let mut all = Vec::new();
        let drug_ids: Vec<DrugId> = self
            .graph
            .inner
            .node_indices()
            .map(|ni| self.graph.inner[ni].drug_id.clone())
            .collect();
        for did in &drug_ids {
            all.extend(self.find_chains_from(did));
        }
        // Deduplicate by sorting drugs in each chain
        let mut seen: HashSet<String> = HashSet::new();
        all.retain(|chain| {
            let mut key_parts: Vec<&str> = chain.drugs.iter().map(|d| d.as_str()).collect();
            key_parts.sort();
            let key = key_parts.join(",");
            seen.insert(key)
        });
        all.sort_by(|a, b| {
            b.cumulative_severity
                .partial_cmp(&a.cumulative_severity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all
    }
}

// ---------------------------------------------------------------------------
// InteractionGraph
// ---------------------------------------------------------------------------

/// Directed graph of drug interactions.
pub struct InteractionGraph {
    pub inner: DiGraph<DrugNode, InteractionEdge>,
    node_map: HashMap<DrugId, NodeIndex>,
}

impl InteractionGraph {
    /// Create an empty interaction graph.
    pub fn new() -> Self {
        Self {
            inner: DiGraph::new(),
            node_map: HashMap::new(),
        }
    }

    /// Build a graph from a set of confirmed conflicts.
    pub fn build_from_conflicts(
        drugs: &[DrugInfo],
        conflicts: &[ConfirmedConflict],
    ) -> Self {
        let mut graph = Self::new();

        // Add all drug nodes
        for drug in drugs {
            graph.add_drug(drug);
        }

        // Add edges for each conflict
        for conflict in conflicts {
            if conflict.drugs.len() >= 2 {
                let a = &conflict.drugs[0];
                let b = &conflict.drugs[1];
                graph.add_interaction(
                    a,
                    b,
                    InteractionEdge::new(
                        conflict.interaction_type.clone(),
                        conflict.severity,
                        conflict.confidence,
                    ),
                );
            }
        }

        graph
    }

    /// Add a drug node. Returns the node index (existing if already present).
    pub fn add_drug(&mut self, info: &DrugInfo) -> NodeIndex {
        if let Some(&idx) = self.node_map.get(&info.id) {
            return idx;
        }
        let node = DrugNode::from_drug_info(info);
        let idx = self.inner.add_node(node);
        self.node_map.insert(info.id.clone(), idx);
        idx
    }

    /// Add a drug node from raw fields.
    pub fn add_drug_node(&mut self, node: DrugNode) -> NodeIndex {
        if let Some(&idx) = self.node_map.get(&node.drug_id) {
            return idx;
        }
        let drug_id = node.drug_id.clone();
        let idx = self.inner.add_node(node);
        self.node_map.insert(drug_id, idx);
        idx
    }

    /// Add a directed interaction edge from drug_a to drug_b.
    pub fn add_interaction(
        &mut self,
        from: &DrugId,
        to: &DrugId,
        edge: InteractionEdge,
    ) -> bool {
        match (self.node_map.get(from), self.node_map.get(to)) {
            (Some(&a), Some(&b)) => {
                self.inner.add_edge(a, b, edge);
                true
            }
            _ => false,
        }
    }

    /// Look up a node index by DrugId.
    pub fn node_index(&self, drug_id: &DrugId) -> Option<NodeIndex> {
        self.node_map.get(drug_id).copied()
    }

    /// Number of drug nodes.
    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Number of interaction edges.
    pub fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// Get all interactions involving a specific drug.
    pub fn interactions_for(&self, drug_id: &DrugId) -> Vec<(&DrugNode, &InteractionEdge)> {
        let idx = match self.node_map.get(drug_id) {
            Some(&i) => i,
            None => return Vec::new(),
        };
        let mut results = Vec::new();
        for edge_ref in self.inner.edges_directed(idx, Direction::Outgoing) {
            let target = &self.inner[edge_ref.target()];
            results.push((target, edge_ref.weight()));
        }
        for edge_ref in self.inner.edges_directed(idx, Direction::Incoming) {
            let source = &self.inner[edge_ref.source()];
            results.push((source, edge_ref.weight()));
        }
        results
    }

    /// Find interaction cliques (fully connected subgraphs).
    /// Uses a Bron–Kerbosch-style approach on the undirected view.
    pub fn find_interaction_cliques(&self, min_size: usize) -> Vec<Vec<DrugId>> {
        let n = self.inner.node_count();
        if n == 0 {
            return Vec::new();
        }

        // Build adjacency set (undirected)
        let mut adj: HashMap<NodeIndex, HashSet<NodeIndex>> = HashMap::new();
        for ni in self.inner.node_indices() {
            adj.insert(ni, HashSet::new());
        }
        for edge_ref in self.inner.edge_references() {
            adj.get_mut(&edge_ref.source())
                .unwrap()
                .insert(edge_ref.target());
            adj.get_mut(&edge_ref.target())
                .unwrap()
                .insert(edge_ref.source());
        }

        let mut cliques: Vec<Vec<NodeIndex>> = Vec::new();
        let all_nodes: Vec<NodeIndex> = self.inner.node_indices().collect();

        self.bron_kerbosch(
            &adj,
            Vec::new(),
            all_nodes.clone(),
            Vec::new(),
            &mut cliques,
            min_size,
        );

        cliques
            .into_iter()
            .map(|c| {
                let mut ids: Vec<DrugId> = c
                    .iter()
                    .map(|&ni| self.inner[ni].drug_id.clone())
                    .collect();
                ids.sort();
                ids
            })
            .collect()
    }

    fn bron_kerbosch(
        &self,
        adj: &HashMap<NodeIndex, HashSet<NodeIndex>>,
        r: Vec<NodeIndex>,
        mut p: Vec<NodeIndex>,
        mut x: Vec<NodeIndex>,
        results: &mut Vec<Vec<NodeIndex>>,
        min_size: usize,
    ) {
        if p.is_empty() && x.is_empty() {
            if r.len() >= min_size {
                results.push(r);
            }
            return;
        }

        // Pivot: choose the vertex in P ∪ X with most connections to P
        let pivot = p
            .iter()
            .chain(x.iter())
            .max_by_key(|&&v| {
                adj.get(&v)
                    .map_or(0, |neighbors| {
                        p.iter().filter(|&&u| neighbors.contains(&u)).count()
                    })
            })
            .copied();

        let pivot_neighbors: HashSet<NodeIndex> = pivot
            .and_then(|pv| adj.get(&pv))
            .cloned()
            .unwrap_or_default();

        let candidates: Vec<NodeIndex> = p
            .iter()
            .filter(|v| !pivot_neighbors.contains(v))
            .copied()
            .collect();

        for v in candidates {
            let v_neighbors = adj.get(&v).cloned().unwrap_or_default();
            let new_r: Vec<NodeIndex> = {
                let mut nr = r.clone();
                nr.push(v);
                nr
            };
            let new_p: Vec<NodeIndex> = p.iter().filter(|&&u| v_neighbors.contains(&u)).copied().collect();
            let new_x: Vec<NodeIndex> = x.iter().filter(|&&u| v_neighbors.contains(&u)).copied().collect();

            self.bron_kerbosch(adj, new_r, new_p, new_x, results, min_size);

            p.retain(|&u| u != v);
            x.push(v);
        }
    }

    /// Sort drugs by the severity of their worst interaction (descending).
    pub fn topological_sort_by_severity(&self) -> Vec<(DrugId, ConflictSeverity)> {
        let mut drug_severity: HashMap<DrugId, ConflictSeverity> = HashMap::new();

        for ni in self.inner.node_indices() {
            let drug_id = &self.inner[ni].drug_id;
            let mut max_sev = ConflictSeverity::Minor;

            for edge_ref in self.inner.edges_directed(ni, Direction::Outgoing) {
                if edge_ref.weight().severity > max_sev {
                    max_sev = edge_ref.weight().severity;
                }
            }
            for edge_ref in self.inner.edges_directed(ni, Direction::Incoming) {
                if edge_ref.weight().severity > max_sev {
                    max_sev = edge_ref.weight().severity;
                }
            }

            drug_severity.insert(drug_id.clone(), max_sev);
        }

        let mut sorted: Vec<(DrugId, ConflictSeverity)> = drug_severity.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        sorted
    }

    /// Compute summary analytics for the graph.
    pub fn analytics(&self) -> GraphAnalytics {
        let nc = self.inner.node_count();
        let ec = self.inner.edge_count();

        let density = if nc > 1 {
            ec as f64 / (nc as f64 * (nc as f64 - 1.0))
        } else {
            0.0
        };

        let mut max_in = 0usize;
        let mut max_out = 0usize;
        let mut degree_dist: HashMap<usize, usize> = HashMap::new();
        let mut total_degree = 0usize;
        let mut most_connected: Option<(DrugId, usize)> = None;

        for ni in self.inner.node_indices() {
            let in_deg = self
                .inner
                .edges_directed(ni, Direction::Incoming)
                .count();
            let out_deg = self
                .inner
                .edges_directed(ni, Direction::Outgoing)
                .count();
            let deg = in_deg + out_deg;
            total_degree += deg;

            if in_deg > max_in {
                max_in = in_deg;
            }
            if out_deg > max_out {
                max_out = out_deg;
            }

            *degree_dist.entry(deg).or_insert(0) += 1;

            match &most_connected {
                Some((_, best)) if deg > *best => {
                    most_connected = Some((self.inner[ni].drug_id.clone(), deg));
                }
                None => {
                    most_connected = Some((self.inner[ni].drug_id.clone(), deg));
                }
                _ => {}
            }
        }

        let avg_degree = if nc > 0 {
            total_degree as f64 / nc as f64
        } else {
            0.0
        };

        // Connected components (undirected BFS)
        let components = self.count_connected_components();

        // Severity histogram
        let mut sev_hist: HashMap<String, usize> = HashMap::new();
        for edge_ref in self.inner.edge_references() {
            *sev_hist
                .entry(edge_ref.weight().severity.label().to_string())
                .or_insert(0) += 1;
        }

        GraphAnalytics {
            node_count: nc,
            edge_count: ec,
            density,
            max_in_degree: max_in,
            max_out_degree: max_out,
            avg_degree,
            connected_components: components,
            degree_distribution: degree_dist,
            most_connected_drug: most_connected.map(|(id, _)| id),
            severity_histogram: sev_hist,
        }
    }

    fn count_connected_components(&self) -> usize {
        let mut visited: HashSet<NodeIndex> = HashSet::new();
        let mut components = 0usize;

        for ni in self.inner.node_indices() {
            if visited.contains(&ni) {
                continue;
            }
            components += 1;
            let mut queue = VecDeque::new();
            queue.push_back(ni);
            visited.insert(ni);

            while let Some(current) = queue.pop_front() {
                for edge_ref in self.inner.edges_directed(current, Direction::Outgoing) {
                    if visited.insert(edge_ref.target()) {
                        queue.push_back(edge_ref.target());
                    }
                }
                for edge_ref in self.inner.edges_directed(current, Direction::Incoming) {
                    if visited.insert(edge_ref.source()) {
                        queue.push_back(edge_ref.source());
                    }
                }
            }
        }

        components
    }

    /// Export the graph for visualization.
    pub fn to_visualization_data(&self) -> VisualizationData {
        let nodes: Vec<VisualizationNode> = self
            .inner
            .node_indices()
            .map(|ni| {
                let n = &self.inner[ni];
                VisualizationNode {
                    id: n.drug_id.to_string(),
                    label: n.name.clone(),
                    group: n.therapeutic_class.clone(),
                }
            })
            .collect();

        let edges: Vec<VisualizationEdge> = self
            .inner
            .edge_references()
            .map(|er| {
                let src = &self.inner[er.source()];
                let tgt = &self.inner[er.target()];
                VisualizationEdge {
                    source: src.drug_id.to_string(),
                    target: tgt.drug_id.to_string(),
                    label: er.weight().severity.label().to_string(),
                    weight: er.weight().weight(),
                }
            })
            .collect();

        VisualizationData { nodes, edges }
    }

    /// Get the neighbors (drugs that interact with) a given drug.
    pub fn neighbors(&self, drug_id: &DrugId) -> Vec<DrugId> {
        let idx = match self.node_map.get(drug_id) {
            Some(&i) => i,
            None => return Vec::new(),
        };
        let mut result = HashSet::new();
        for edge_ref in self.inner.edges_directed(idx, Direction::Outgoing) {
            result.insert(self.inner[edge_ref.target()].drug_id.clone());
        }
        for edge_ref in self.inner.edges_directed(idx, Direction::Incoming) {
            result.insert(self.inner[edge_ref.source()].drug_id.clone());
        }
        let mut v: Vec<DrugId> = result.into_iter().collect();
        v.sort();
        v
    }

    /// Find the shortest path between two drugs using BFS on the undirected
    /// view.  Returns the path of drug IDs, or `None` if not connected.
    pub fn shortest_path(&self, from: &DrugId, to: &DrugId) -> Option<Vec<DrugId>> {
        let start = self.node_map.get(from).copied()?;
        let end = self.node_map.get(to).copied()?;
        if start == end {
            return Some(vec![from.clone()]);
        }

        let mut visited: HashSet<NodeIndex> = HashSet::new();
        let mut parent: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let mut queue = VecDeque::new();

        visited.insert(start);
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            let neighbors_iter = self
                .inner
                .edges_directed(current, Direction::Outgoing)
                .map(|e| e.target())
                .chain(
                    self.inner
                        .edges_directed(current, Direction::Incoming)
                        .map(|e| e.source()),
                );

            for next in neighbors_iter {
                if visited.insert(next) {
                    parent.insert(next, current);
                    if next == end {
                        // Reconstruct path
                        let mut path = vec![end];
                        let mut cur = end;
                        while let Some(&p) = parent.get(&cur) {
                            path.push(p);
                            cur = p;
                            if cur == start {
                                break;
                            }
                        }
                        path.reverse();
                        return Some(
                            path.iter()
                                .map(|&ni| self.inner[ni].drug_id.clone())
                                .collect(),
                        );
                    }
                    queue.push_back(next);
                }
            }
        }

        None
    }
}

impl Default for InteractionGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for InteractionGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InteractionGraph")
            .field("nodes", &self.inner.node_count())
            .field("edges", &self.inner.edge_count())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Dosage, AdministrationRoute, GuidelineId, MedicationRecord, PatientId,
                       SafetyCertificate, VerificationResult, VerificationTier};

    fn drug(id: &str, name: &str, class: &str, cyps: &[&str]) -> DrugInfo {
        DrugInfo {
            id: DrugId::new(id),
            name: name.to_string(),
            therapeutic_class: class.to_string(),
            cyp_enzymes: cyps.iter().map(|s| s.to_string()).collect(),
            half_life_hours: 10.0,
            bioavailability: 0.8,
            protein_binding: 0.5,
            therapeutic_index: None,
        }
    }

    fn make_conflict(
        id_a: &str,
        id_b: &str,
        itype: InteractionType,
        sev: ConflictSeverity,
    ) -> ConfirmedConflict {
        ConfirmedConflict {
            id: format!("c-{}-{}", id_a, id_b),
            drugs: vec![DrugId::new(id_a), DrugId::new(id_b)],
            interaction_type: itype,
            severity: sev,
            verdict: SafetyVerdict::PossiblyUnsafe,
            mechanism_description: "test".to_string(),
            evidence_tier: VerificationTier::Tier2ModelCheck,
            counter_example: None,
            confidence: 0.9,
            clinical_recommendation: "test".to_string(),
            affected_parameters: vec![],
            guideline_references: vec![],
        }
    }

    #[test]
    fn test_empty_graph() {
        let g = InteractionGraph::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
        let analytics = g.analytics();
        assert_eq!(analytics.density, 0.0);
    }

    #[test]
    fn test_add_drug_idempotent() {
        let mut g = InteractionGraph::new();
        let d = drug("a", "DrugA", "ClassX", &[]);
        let idx1 = g.add_drug(&d);
        let idx2 = g.add_drug(&d);
        assert_eq!(idx1, idx2);
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_add_interaction() {
        let mut g = InteractionGraph::new();
        let da = drug("a", "A", "X", &["3A4"]);
        let db = drug("b", "B", "X", &["3A4"]);
        g.add_drug(&da);
        g.add_drug(&db);
        let added = g.add_interaction(
            &DrugId::new("a"),
            &DrugId::new("b"),
            InteractionEdge::new(
                InteractionType::CypInhibition {
                    enzyme: "3A4".to_string(),
                },
                ConflictSeverity::Major,
                0.85,
            ),
        );
        assert!(added);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_build_from_conflicts() {
        let drugs = vec![
            drug("a", "A", "X", &[]),
            drug("b", "B", "Y", &[]),
            drug("c", "C", "Z", &[]),
        ];
        let conflicts = vec![
            make_conflict(
                "a",
                "b",
                InteractionType::CypInhibition {
                    enzyme: "3A4".to_string(),
                },
                ConflictSeverity::Major,
            ),
            make_conflict(
                "b",
                "c",
                InteractionType::QtProlongation,
                ConflictSeverity::Critical,
            ),
        ];

        let g = InteractionGraph::build_from_conflicts(&drugs, &conflicts);
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_neighbors() {
        let drugs = vec![
            drug("a", "A", "X", &[]),
            drug("b", "B", "Y", &[]),
            drug("c", "C", "Z", &[]),
        ];
        let conflicts = vec![
            make_conflict("a", "b", InteractionType::RenalCompetition, ConflictSeverity::Minor),
            make_conflict("a", "c", InteractionType::AbsorptionAlteration, ConflictSeverity::Moderate),
        ];
        let g = InteractionGraph::build_from_conflicts(&drugs, &conflicts);
        let neighbors = g.neighbors(&DrugId::new("a"));
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&DrugId::new("b")));
        assert!(neighbors.contains(&DrugId::new("c")));
    }

    #[test]
    fn test_topological_sort_by_severity() {
        let drugs = vec![
            drug("a", "A", "X", &[]),
            drug("b", "B", "Y", &[]),
            drug("c", "C", "Z", &[]),
        ];
        let conflicts = vec![
            make_conflict(
                "a",
                "b",
                InteractionType::CypInhibition {
                    enzyme: "2D6".to_string(),
                },
                ConflictSeverity::Minor,
            ),
            make_conflict(
                "b",
                "c",
                InteractionType::QtProlongation,
                ConflictSeverity::Critical,
            ),
        ];
        let g = InteractionGraph::build_from_conflicts(&drugs, &conflicts);
        let sorted = g.topological_sort_by_severity();

        // b and c should appear first (Critical interaction)
        assert!(!sorted.is_empty());
        let top_sev = sorted[0].1;
        assert_eq!(top_sev, ConflictSeverity::Critical);
    }

    #[test]
    fn test_find_cliques_triangle() {
        let mut g = InteractionGraph::new();
        let da = drug("a", "A", "X", &[]);
        let db = drug("b", "B", "X", &[]);
        let dc = drug("c", "C", "X", &[]);
        g.add_drug(&da);
        g.add_drug(&db);
        g.add_drug(&dc);

        let edge = || {
            InteractionEdge::new(
                InteractionType::PharmacodynamicSynergy,
                ConflictSeverity::Moderate,
                0.8,
            )
        };
        g.add_interaction(&DrugId::new("a"), &DrugId::new("b"), edge());
        g.add_interaction(&DrugId::new("b"), &DrugId::new("a"), edge());
        g.add_interaction(&DrugId::new("b"), &DrugId::new("c"), edge());
        g.add_interaction(&DrugId::new("c"), &DrugId::new("b"), edge());
        g.add_interaction(&DrugId::new("a"), &DrugId::new("c"), edge());
        g.add_interaction(&DrugId::new("c"), &DrugId::new("a"), edge());

        let cliques = g.find_interaction_cliques(3);
        assert!(
            !cliques.is_empty(),
            "Should find the 3-clique among a, b, c"
        );
        assert_eq!(cliques[0].len(), 3);
    }

    #[test]
    fn test_analytics() {
        let drugs = vec![
            drug("a", "A", "X", &[]),
            drug("b", "B", "Y", &[]),
        ];
        let conflicts = vec![make_conflict(
            "a",
            "b",
            InteractionType::ProteinBindingDisplacement,
            ConflictSeverity::Moderate,
        )];
        let g = InteractionGraph::build_from_conflicts(&drugs, &conflicts);
        let a = g.analytics();
        assert_eq!(a.node_count, 2);
        assert_eq!(a.edge_count, 1);
        assert!(a.density > 0.0);
        assert_eq!(a.connected_components, 1);
    }

    #[test]
    fn test_cascade_analyzer() {
        let drugs = vec![
            drug("a", "A", "X", &[]),
            drug("b", "B", "Y", &[]),
            drug("c", "C", "Z", &[]),
        ];
        let conflicts = vec![
            make_conflict(
                "a",
                "b",
                InteractionType::CypInhibition {
                    enzyme: "3A4".to_string(),
                },
                ConflictSeverity::Major,
            ),
            make_conflict(
                "b",
                "c",
                InteractionType::CypInduction {
                    enzyme: "2C9".to_string(),
                },
                ConflictSeverity::Moderate,
            ),
        ];
        let g = InteractionGraph::build_from_conflicts(&drugs, &conflicts);
        let analyzer = CascadeAnalyzer::new(&g, 5);
        let chains = analyzer.find_chains_from(&DrugId::new("a"));
        assert!(!chains.is_empty(), "Should find cascading chain a → b → c");
        // At least one chain of length 2
        assert!(chains.iter().any(|c| c.length == 2));
    }

    #[test]
    fn test_shortest_path() {
        let drugs = vec![
            drug("a", "A", "X", &[]),
            drug("b", "B", "Y", &[]),
            drug("c", "C", "Z", &[]),
        ];
        let conflicts = vec![
            make_conflict("a", "b", InteractionType::RenalCompetition, ConflictSeverity::Minor),
            make_conflict("b", "c", InteractionType::AbsorptionAlteration, ConflictSeverity::Minor),
        ];
        let g = InteractionGraph::build_from_conflicts(&drugs, &conflicts);
        let path = g.shortest_path(&DrugId::new("a"), &DrugId::new("c"));
        assert!(path.is_some());
        let p = path.unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!(p[0], DrugId::new("a"));
        assert_eq!(p[2], DrugId::new("c"));
    }

    #[test]
    fn test_visualization_data() {
        let drugs = vec![
            drug("a", "A", "X", &[]),
            drug("b", "B", "Y", &[]),
        ];
        let conflicts = vec![make_conflict(
            "a",
            "b",
            InteractionType::QtProlongation,
            ConflictSeverity::Critical,
        )];
        let g = InteractionGraph::build_from_conflicts(&drugs, &conflicts);
        let viz = g.to_visualization_data();
        assert_eq!(viz.nodes.len(), 2);
        assert_eq!(viz.edges.len(), 1);
        assert!(viz.edges[0].weight > 0.0);
    }

    #[test]
    fn test_connected_components_disconnected() {
        let mut g = InteractionGraph::new();
        g.add_drug(&drug("a", "A", "X", &[]));
        g.add_drug(&drug("b", "B", "Y", &[]));
        g.add_drug(&drug("c", "C", "Z", &[]));
        // No edges → 3 components
        let a = g.analytics();
        assert_eq!(a.connected_components, 3);
    }

    #[test]
    fn test_interaction_edge_weight() {
        let edge = InteractionEdge::new(
            InteractionType::CypInhibition {
                enzyme: "3A4".to_string(),
            },
            ConflictSeverity::Critical,
            0.95,
        );
        assert!((edge.weight() - 10.0 * 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_interaction_chain_max_severity() {
        let chain = InteractionChain {
            drugs: vec![DrugId::new("a"), DrugId::new("b")],
            interactions: vec![InteractionType::QtProlongation],
            cumulative_severity: 9.0,
            length: 1,
        };
        assert_eq!(chain.max_severity(), ConflictSeverity::Critical);
    }
}
