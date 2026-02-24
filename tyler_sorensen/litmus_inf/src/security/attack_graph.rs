//! Attack graph construction and analysis for GPU security in LITMUS∞.
//!
//! Provides attack graph construction, path enumeration, vulnerability scoring,
//! attack surface analysis, mitigation recommendations, and threat modeling.

use std::collections::{HashMap, HashSet, BTreeMap, VecDeque};
use std::fmt;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// Severity and Scoring
// ---------------------------------------------------------------------------

/// Severity levels for vulnerabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SeverityLevel {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

impl SeverityLevel {
    /// Numeric score for the severity level.
    pub fn score(&self) -> f64 {
        match self {
            Self::Info => 0.0,
            Self::Low => 2.5,
            Self::Medium => 5.0,
            Self::High => 7.5,
            Self::Critical => 10.0,
        }
    }

    /// From a numeric score.
    pub fn from_score(score: f64) -> Self {
        if score >= 9.0 { Self::Critical }
        else if score >= 7.0 { Self::High }
        else if score >= 4.0 { Self::Medium }
        else if score >= 1.0 { Self::Low }
        else { Self::Info }
    }
}

impl fmt::Display for SeverityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// CVSS-like vulnerability score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityScore {
    /// Base score (0.0 - 10.0).
    pub base: f64,
    /// Temporal score modifier.
    pub temporal: f64,
    /// Environmental score modifier.
    pub environmental: f64,
    /// Attack vector (network, adjacent, local, physical).
    pub attack_vector: AttackVector,
    /// Attack complexity.
    pub attack_complexity: AttackComplexity,
    /// Impact on confidentiality.
    pub confidentiality_impact: ImpactLevel,
    /// Impact on integrity.
    pub integrity_impact: ImpactLevel,
    /// Impact on availability.
    pub availability_impact: ImpactLevel,
}

impl VulnerabilityScore {
    /// Create a new score.
    pub fn new(base: f64) -> Self {
        Self {
            base: base.clamp(0.0, 10.0),
            temporal: 1.0,
            environmental: 1.0,
            attack_vector: AttackVector::Local,
            attack_complexity: AttackComplexity::Low,
            confidentiality_impact: ImpactLevel::None,
            integrity_impact: ImpactLevel::None,
            availability_impact: ImpactLevel::None,
        }
    }

    /// Compute the overall score.
    pub fn overall(&self) -> f64 {
        (self.base * self.temporal * self.environmental).clamp(0.0, 10.0)
    }

    /// Get the severity level.
    pub fn severity(&self) -> SeverityLevel {
        SeverityLevel::from_score(self.overall())
    }

    /// Set temporal modifier.
    pub fn with_temporal(mut self, t: f64) -> Self {
        self.temporal = t.clamp(0.0, 1.0);
        self
    }

    /// Set environmental modifier.
    pub fn with_environmental(mut self, e: f64) -> Self {
        self.environmental = e.clamp(0.0, 1.0);
        self
    }

    /// Set attack vector.
    pub fn with_vector(mut self, v: AttackVector) -> Self {
        self.attack_vector = v;
        self
    }

    /// Set complexity.
    pub fn with_complexity(mut self, c: AttackComplexity) -> Self {
        self.attack_complexity = c;
        self
    }

    /// Set impacts.
    pub fn with_impacts(mut self, c: ImpactLevel, i: ImpactLevel, a: ImpactLevel) -> Self {
        self.confidentiality_impact = c;
        self.integrity_impact = i;
        self.availability_impact = a;
        self
    }
}

impl fmt::Display for VulnerabilityScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CVSS {:.1} ({})", self.overall(), self.severity())
    }
}

/// Attack vector categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttackVector {
    Network,
    Adjacent,
    Local,
    Physical,
}

/// Attack complexity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttackComplexity {
    Low,
    High,
}

/// Impact level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImpactLevel {
    None,
    Low,
    High,
}

// ---------------------------------------------------------------------------
// Attack Node
// ---------------------------------------------------------------------------

/// Type of attack node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttackNodeType {
    /// Entry point into the attack.
    EntryPoint,
    /// Memory access that can be exploited.
    MemoryAccess,
    /// Race condition between threads.
    RaceCondition,
    /// Information leak (side channel).
    InformationLeak,
    /// Privilege escalation.
    Escalation,
    /// Exit point (attack succeeds).
    ExitPoint,
    /// Fence bypass vulnerability.
    FenceBypass,
    /// Scope violation (accessing outside scope).
    ScopeViolation,
}

impl fmt::Display for AttackNodeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EntryPoint => write!(f, "ENTRY"),
            Self::MemoryAccess => write!(f, "MEM_ACCESS"),
            Self::RaceCondition => write!(f, "RACE"),
            Self::InformationLeak => write!(f, "INFO_LEAK"),
            Self::Escalation => write!(f, "ESCALATION"),
            Self::ExitPoint => write!(f, "EXIT"),
            Self::FenceBypass => write!(f, "FENCE_BYPASS"),
            Self::ScopeViolation => write!(f, "SCOPE_VIOLATION"),
        }
    }
}

/// A node in the attack graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackNode {
    /// Unique node identifier.
    pub id: usize,
    /// Type of attack step.
    pub node_type: AttackNodeType,
    /// Human-readable description.
    pub description: String,
    /// Severity of this step.
    pub severity: SeverityLevel,
    /// Associated memory address (if applicable).
    pub address: Option<u64>,
    /// Thread ID involved.
    pub thread_id: Option<usize>,
    /// GPU scope (if applicable).
    pub scope: Option<String>,
    /// Tags for categorization.
    pub tags: Vec<String>,
}

impl AttackNode {
    /// Create a new attack node.
    pub fn new(id: usize, node_type: AttackNodeType, description: &str) -> Self {
        Self {
            id,
            node_type,
            description: description.to_string(),
            severity: SeverityLevel::Medium,
            address: None,
            thread_id: None,
            scope: None,
            tags: Vec::new(),
        }
    }

    /// Set severity.
    pub fn with_severity(mut self, severity: SeverityLevel) -> Self {
        self.severity = severity;
        self
    }

    /// Set address.
    pub fn with_address(mut self, addr: u64) -> Self {
        self.address = Some(addr);
        self
    }

    /// Set thread.
    pub fn with_thread(mut self, tid: usize) -> Self {
        self.thread_id = Some(tid);
        self
    }

    /// Set scope.
    pub fn with_scope(mut self, scope: &str) -> Self {
        self.scope = Some(scope.to_string());
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Check if this is an entry node.
    pub fn is_entry(&self) -> bool { self.node_type == AttackNodeType::EntryPoint }
    /// Check if this is an exit node.
    pub fn is_exit(&self) -> bool { self.node_type == AttackNodeType::ExitPoint }
}

impl fmt::Display for AttackNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} ({}) - {}", self.id, self.node_type, self.severity, self.description)
    }
}

// ---------------------------------------------------------------------------
// Attack Edge
// ---------------------------------------------------------------------------

/// An edge in the attack graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackEdge {
    /// Source node ID.
    pub source: usize,
    /// Target node ID.
    pub target: usize,
    /// Preconditions for this transition.
    pub preconditions: Vec<String>,
    /// Postconditions after this transition.
    pub postconditions: Vec<String>,
    /// Probability of successful transition (0.0 - 1.0).
    pub probability: f64,
    /// Description of the attack step.
    pub description: String,
    /// Estimated cost/effort.
    pub cost: f64,
}

impl AttackEdge {
    /// Create a new edge.
    pub fn new(source: usize, target: usize) -> Self {
        Self {
            source,
            target,
            preconditions: Vec::new(),
            postconditions: Vec::new(),
            probability: 1.0,
            description: String::new(),
            cost: 1.0,
        }
    }

    /// Set probability.
    pub fn with_probability(mut self, p: f64) -> Self {
        self.probability = p.clamp(0.0, 1.0);
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Add a precondition.
    pub fn with_precondition(mut self, pre: &str) -> Self {
        self.preconditions.push(pre.to_string());
        self
    }

    /// Add a postcondition.
    pub fn with_postcondition(mut self, post: &str) -> Self {
        self.postconditions.push(post.to_string());
        self
    }

    /// Set cost.
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl fmt::Display for AttackEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {} (p={:.2}, cost={:.1})", self.source, self.target, self.probability, self.cost)?;
        if !self.description.is_empty() {
            write!(f, " [{}]", self.description)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Attack Graph
// ---------------------------------------------------------------------------

/// Attack graph using adjacency list representation.
#[derive(Debug, Clone)]
pub struct AttackGraph {
    /// Nodes in the graph.
    nodes: Vec<AttackNode>,
    /// Adjacency list: adj[source] = [(target_idx, edge)].
    adj: Vec<Vec<(usize, AttackEdge)>>,
    /// Reverse adjacency list for backward traversal.
    rev_adj: Vec<Vec<(usize, usize)>>,
}

impl AttackGraph {
    /// Create a new empty attack graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            adj: Vec::new(),
            rev_adj: Vec::new(),
        }
    }

    /// Add a node. Returns the node ID.
    pub fn add_node(&mut self, node: AttackNode) -> usize {
        let id = self.nodes.len();
        let mut n = node;
        n.id = id;
        self.nodes.push(n);
        self.adj.push(Vec::new());
        self.rev_adj.push(Vec::new());
        id
    }

    /// Add an edge.
    pub fn add_edge(&mut self, edge: AttackEdge) {
        let src = edge.source;
        let tgt = edge.target;
        assert!(src < self.nodes.len() && tgt < self.nodes.len());
        let edge_idx = self.adj[src].len();
        self.adj[src].push((tgt, edge));
        self.rev_adj[tgt].push((src, edge_idx));
    }

    /// Get a node by ID.
    pub fn node(&self, id: usize) -> Option<&AttackNode> {
        self.nodes.get(id)
    }

    /// Get all nodes.
    pub fn nodes(&self) -> &[AttackNode] {
        &self.nodes
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get edges from a node.
    pub fn edges_from(&self, node: usize) -> &[(usize, AttackEdge)] {
        &self.adj[node]
    }

    /// Get all edges.
    pub fn all_edges(&self) -> Vec<&AttackEdge> {
        self.adj.iter().flat_map(|edges| edges.iter().map(|(_, e)| e)).collect()
    }

    /// Total number of edges.
    pub fn edge_count(&self) -> usize {
        self.adj.iter().map(|edges| edges.len()).sum()
    }

    /// Get entry nodes.
    pub fn entry_nodes(&self) -> Vec<usize> {
        self.nodes.iter()
            .filter(|n| n.is_entry())
            .map(|n| n.id)
            .collect()
    }

    /// Get exit nodes.
    pub fn exit_nodes(&self) -> Vec<usize> {
        self.nodes.iter()
            .filter(|n| n.is_exit())
            .map(|n| n.id)
            .collect()
    }

    /// Check if a path exists from source to target.
    pub fn path_exists(&self, source: usize, target: usize) -> bool {
        if source == target { return true; }
        let mut visited = vec![false; self.nodes.len()];
        let mut queue = VecDeque::new();
        queue.push_back(source);
        visited[source] = true;

        while let Some(node) = queue.pop_front() {
            for &(next, _) in &self.adj[node] {
                if next == target { return true; }
                if !visited[next] {
                    visited[next] = true;
                    queue.push_back(next);
                }
            }
        }
        false
    }

    /// Find shortest path from source to target.
    pub fn shortest_path(&self, source: usize, target: usize) -> Option<Vec<usize>> {
        if source == target { return Some(vec![source]); }
        let mut visited = vec![false; self.nodes.len()];
        let mut parent = vec![usize::MAX; self.nodes.len()];
        let mut queue = VecDeque::new();
        queue.push_back(source);
        visited[source] = true;

        while let Some(node) = queue.pop_front() {
            for &(next, _) in &self.adj[node] {
                if !visited[next] {
                    visited[next] = true;
                    parent[next] = node;
                    if next == target {
                        let mut path = vec![target];
                        let mut current = target;
                        while current != source {
                            current = parent[current];
                            path.push(current);
                        }
                        path.reverse();
                        return Some(path);
                    }
                    queue.push_back(next);
                }
            }
        }
        None
    }

    /// Get neighbors of a node.
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        self.adj[node].iter().map(|&(tgt, _)| tgt).collect()
    }

    /// Get predecessors of a node.
    pub fn predecessors(&self, node: usize) -> Vec<usize> {
        self.rev_adj[node].iter().map(|&(src, _)| src).collect()
    }

    /// Get the maximum severity of any node.
    pub fn max_severity(&self) -> SeverityLevel {
        self.nodes.iter()
            .map(|n| n.severity)
            .max()
            .unwrap_or(SeverityLevel::Info)
    }
}

impl Default for AttackGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AttackGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Attack Graph ({} nodes, {} edges)", self.node_count(), self.edge_count())?;
        for node in &self.nodes {
            writeln!(f, "  {}", node)?;
        }
        for (src, edges) in self.adj.iter().enumerate() {
            for (tgt, edge) in edges {
                writeln!(f, "  {} -> {}: {}", src, tgt, edge.description)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Attack Graph Builder
// ---------------------------------------------------------------------------

/// Builds attack graphs from program events.
#[derive(Debug, Clone)]
pub struct AttackGraphBuilder {
    graph: AttackGraph,
    /// Memory access log: address -> [(thread, op_type, event_id)].
    access_log: HashMap<u64, Vec<(usize, String, usize)>>,
}

impl AttackGraphBuilder {
    pub fn new() -> Self {
        Self {
            graph: AttackGraph::new(),
            access_log: HashMap::new(),
        }
    }

    /// Add an entry point.
    pub fn add_entry(&mut self, description: &str) -> usize {
        self.graph.add_node(
            AttackNode::new(0, AttackNodeType::EntryPoint, description)
                .with_severity(SeverityLevel::Info)
        )
    }

    /// Add an exit point.
    pub fn add_exit(&mut self, description: &str) -> usize {
        self.graph.add_node(
            AttackNode::new(0, AttackNodeType::ExitPoint, description)
                .with_severity(SeverityLevel::Critical)
        )
    }

    /// Record a memory access.
    pub fn record_access(&mut self, addr: u64, thread: usize, op_type: &str, event_id: usize) {
        self.access_log.entry(addr).or_default()
            .push((thread, op_type.to_string(), event_id));
    }

    /// Add a vulnerability node.
    pub fn add_vulnerability(
        &mut self, node_type: AttackNodeType, description: &str,
        severity: SeverityLevel,
    ) -> usize {
        self.graph.add_node(
            AttackNode::new(0, node_type, description).with_severity(severity)
        )
    }

    /// Connect two nodes.
    pub fn connect(&mut self, source: usize, target: usize, description: &str, probability: f64) {
        self.graph.add_edge(
            AttackEdge::new(source, target)
                .with_description(description)
                .with_probability(probability)
        );
    }

    /// Detect race conditions from the access log.
    pub fn detect_races(&mut self) -> Vec<usize> {
        let mut race_nodes = Vec::new();

        for (addr, accesses) in &self.access_log {
            let mut writers: Vec<usize> = Vec::new();
            let mut readers: Vec<usize> = Vec::new();

            for (thread, op, _) in accesses {
                match op.as_str() {
                    "write" | "store" => writers.push(*thread),
                    "read" | "load" => readers.push(*thread),
                    _ => {}
                }
            }

            // Check for write-write races.
            let unique_writers: HashSet<_> = writers.iter().collect();
            if unique_writers.len() > 1 {
                let node_id = self.graph.add_node(
                    AttackNode::new(0, AttackNodeType::RaceCondition,
                        &format!("Write-write race on address 0x{:x}", addr))
                        .with_severity(SeverityLevel::High)
                        .with_address(*addr)
                );
                race_nodes.push(node_id);
            }

            // Check for read-write races.
            if !writers.is_empty() && !readers.is_empty() {
                let writer_threads: HashSet<_> = writers.iter().collect();
                let reader_threads: HashSet<_> = readers.iter().collect();
                let cross_thread = writer_threads.iter()
                    .any(|w| reader_threads.iter().any(|r| *w != *r));

                if cross_thread {
                    let node_id = self.graph.add_node(
                        AttackNode::new(0, AttackNodeType::RaceCondition,
                            &format!("Read-write race on address 0x{:x}", addr))
                            .with_severity(SeverityLevel::Medium)
                            .with_address(*addr)
                    );
                    race_nodes.push(node_id);
                }
            }
        }

        race_nodes
    }

    /// Build the attack graph.
    pub fn build(self) -> AttackGraph {
        self.graph
    }
}

impl Default for AttackGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Attack Path Enumerator
// ---------------------------------------------------------------------------

/// Enumerates attack paths in an attack graph.
#[derive(Debug, Clone)]
pub struct AttackPathEnumerator {
    max_paths: usize,
    max_depth: usize,
}

/// An attack path from entry to exit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackPath {
    /// Sequence of node IDs.
    pub nodes: Vec<usize>,
    /// Total probability (product of edge probabilities).
    pub probability: f64,
    /// Total cost.
    pub cost: f64,
    /// Maximum severity along the path.
    pub max_severity: SeverityLevel,
}

impl AttackPath {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            probability: 1.0,
            cost: 0.0,
            max_severity: SeverityLevel::Info,
        }
    }

    /// Length of the path.
    pub fn length(&self) -> usize {
        self.nodes.len()
    }

    /// Risk score (probability * severity).
    pub fn risk_score(&self) -> f64 {
        self.probability * self.max_severity.score()
    }
}

impl Default for AttackPath {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for AttackPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Path [{}] (p={:.4}, cost={:.1}, severity={})",
            self.nodes.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(" -> "),
            self.probability, self.cost, self.max_severity)
    }
}

impl AttackPathEnumerator {
    pub fn new(max_paths: usize, max_depth: usize) -> Self {
        Self { max_paths, max_depth }
    }

    /// Find all paths from entry to exit nodes.
    pub fn enumerate_all(&self, graph: &AttackGraph) -> Vec<AttackPath> {
        let entries = graph.entry_nodes();
        let exits: HashSet<usize> = graph.exit_nodes().into_iter().collect();
        let mut all_paths = Vec::new();

        for entry in entries {
            let mut visited = vec![false; graph.node_count()];
            let mut current_path = AttackPath::new();
            current_path.nodes.push(entry);

            self.dfs_paths(
                graph, entry, &exits, &mut visited,
                &mut current_path, &mut all_paths,
            );

            if all_paths.len() >= self.max_paths {
                break;
            }
        }

        all_paths
    }

    fn dfs_paths(
        &self,
        graph: &AttackGraph,
        node: usize,
        exits: &HashSet<usize>,
        visited: &mut Vec<bool>,
        current: &mut AttackPath,
        results: &mut Vec<AttackPath>,
    ) {
        if results.len() >= self.max_paths {
            return;
        }
        if current.nodes.len() > self.max_depth {
            return;
        }

        if exits.contains(&node) && current.nodes.len() > 1 {
            let severity = current.nodes.iter()
                .filter_map(|&n| graph.node(n))
                .map(|n| n.severity)
                .max()
                .unwrap_or(SeverityLevel::Info);
            let mut path = current.clone();
            path.max_severity = severity;
            results.push(path);
            return;
        }

        visited[node] = true;

        for &(next, ref edge) in graph.edges_from(node) {
            if !visited[next] {
                current.nodes.push(next);
                let old_prob = current.probability;
                let old_cost = current.cost;
                current.probability *= edge.probability;
                current.cost += edge.cost;

                self.dfs_paths(graph, next, exits, visited, current, results);

                current.probability = old_prob;
                current.cost = old_cost;
                current.nodes.pop();
            }
        }

        visited[node] = false;
    }

    /// Find k-shortest paths using BFS with priority.
    pub fn k_shortest_paths(&self, graph: &AttackGraph, k: usize) -> Vec<AttackPath> {
        let mut all_paths = self.enumerate_all(graph);
        all_paths.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal));
        all_paths.truncate(k);
        all_paths
    }

    /// Find the highest-risk paths.
    pub fn highest_risk_paths(&self, graph: &AttackGraph, k: usize) -> Vec<AttackPath> {
        let mut all_paths = self.enumerate_all(graph);
        all_paths.sort_by(|a, b| b.risk_score().partial_cmp(&a.risk_score()).unwrap_or(std::cmp::Ordering::Equal));
        all_paths.truncate(k);
        all_paths
    }
}

// ---------------------------------------------------------------------------
// Attack Surface Analyzer
// ---------------------------------------------------------------------------

/// Analyzes the attack surface of a program.
#[derive(Debug, Clone)]
pub struct AttackSurfaceAnalyzer;

/// Attack surface metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackSurface {
    /// Number of entry points.
    pub entry_points: usize,
    /// Number of exit points.
    pub exit_points: usize,
    /// Number of reachable vulnerability nodes.
    pub reachable_vulnerabilities: usize,
    /// Total number of attack paths.
    pub total_paths: usize,
    /// Maximum attack depth.
    pub max_depth: usize,
    /// Average path length.
    pub avg_path_length: f64,
    /// Maximum risk score.
    pub max_risk: f64,
    /// Nodes by severity.
    pub severity_distribution: HashMap<String, usize>,
    /// Shared memory exposure (GPU-specific).
    pub shared_memory_exposure: usize,
    /// Cross-scope edges.
    pub cross_scope_edges: usize,
}

impl AttackSurfaceAnalyzer {
    pub fn new() -> Self { Self }

    /// Analyze the attack surface of a graph.
    pub fn analyze(&self, graph: &AttackGraph) -> AttackSurface {
        let entry_points = graph.entry_nodes().len();
        let exit_points = graph.exit_nodes().len();

        // Find reachable vulnerabilities from entry nodes.
        let entries = graph.entry_nodes();
        let mut reachable = HashSet::new();
        for entry in &entries {
            let mut visited = vec![false; graph.node_count()];
            let mut queue = VecDeque::new();
            queue.push_back(*entry);
            visited[*entry] = true;

            while let Some(node) = queue.pop_front() {
                let n = &graph.nodes()[node];
                if matches!(n.node_type,
                    AttackNodeType::RaceCondition | AttackNodeType::InformationLeak |
                    AttackNodeType::FenceBypass | AttackNodeType::ScopeViolation |
                    AttackNodeType::Escalation
                ) {
                    reachable.insert(node);
                }
                for &(next, _) in graph.edges_from(node) {
                    if !visited[next] {
                        visited[next] = true;
                        queue.push_back(next);
                    }
                }
            }
        }

        // Enumerate paths for metrics.
        let enumerator = AttackPathEnumerator::new(1000, 20);
        let paths = enumerator.enumerate_all(graph);

        let max_depth = paths.iter().map(|p| p.length()).max().unwrap_or(0);
        let avg_len = if paths.is_empty() { 0.0 }
            else { paths.iter().map(|p| p.length() as f64).sum::<f64>() / paths.len() as f64 };
        let max_risk = paths.iter().map(|p| p.risk_score()).fold(0.0f64, f64::max);

        let mut severity_dist = HashMap::new();
        for node in graph.nodes() {
            *severity_dist.entry(node.severity.to_string()).or_insert(0) += 1;
        }

        let shared_mem = graph.nodes().iter()
            .filter(|n| n.tags.contains(&"shared_memory".to_string()))
            .count();

        let cross_scope = graph.all_edges().iter()
            .filter(|e| {
                let src_scope = graph.node(e.source).and_then(|n| n.scope.as_ref());
                let tgt_scope = graph.node(e.target).and_then(|n| n.scope.as_ref());
                src_scope.is_some() && tgt_scope.is_some() && src_scope != tgt_scope
            })
            .count();

        AttackSurface {
            entry_points,
            exit_points,
            reachable_vulnerabilities: reachable.len(),
            total_paths: paths.len(),
            max_depth,
            avg_path_length: avg_len,
            max_risk,
            severity_distribution: severity_dist,
            shared_memory_exposure: shared_mem,
            cross_scope_edges: cross_scope,
        }
    }
}

impl Default for AttackSurfaceAnalyzer {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for AttackSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Attack Surface Analysis:")?;
        writeln!(f, "  Entry points: {}", self.entry_points)?;
        writeln!(f, "  Exit points: {}", self.exit_points)?;
        writeln!(f, "  Reachable vulnerabilities: {}", self.reachable_vulnerabilities)?;
        writeln!(f, "  Total paths: {}", self.total_paths)?;
        writeln!(f, "  Max depth: {}", self.max_depth)?;
        writeln!(f, "  Avg path length: {:.1}", self.avg_path_length)?;
        writeln!(f, "  Max risk: {:.2}", self.max_risk)?;
        writeln!(f, "  Shared memory exposure: {}", self.shared_memory_exposure)?;
        writeln!(f, "  Cross-scope edges: {}", self.cross_scope_edges)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Mitigation Engine
// ---------------------------------------------------------------------------

/// Type of mitigation action.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MitigationType {
    /// Insert a memory fence.
    FenceInsertion { fence_type: String, position: String },
    /// Restrict memory access scope.
    ScopeRestriction { from_scope: String, to_scope: String },
    /// Change access pattern.
    AccessPatternChange { description: String },
    /// Add synchronization primitive.
    AddSynchronization { sync_type: String },
    /// Use constant-time implementation.
    ConstantTime { operation: String },
    /// Apply padding to prevent cache attacks.
    CachePadding { size: usize },
}

impl fmt::Display for MitigationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FenceInsertion { fence_type, position } =>
                write!(f, "Insert {} fence at {}", fence_type, position),
            Self::ScopeRestriction { from_scope, to_scope } =>
                write!(f, "Restrict scope {} -> {}", from_scope, to_scope),
            Self::AccessPatternChange { description } =>
                write!(f, "Change access: {}", description),
            Self::AddSynchronization { sync_type } =>
                write!(f, "Add sync: {}", sync_type),
            Self::ConstantTime { operation } =>
                write!(f, "Constant-time: {}", operation),
            Self::CachePadding { size } =>
                write!(f, "Cache padding: {} bytes", size),
        }
    }
}

/// A mitigation recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mitigation {
    /// Type of mitigation.
    pub mitigation_type: MitigationType,
    /// Vulnerability node IDs this addresses.
    pub addresses_nodes: Vec<usize>,
    /// Estimated cost (performance overhead, 0.0 - 1.0).
    pub cost: f64,
    /// Estimated effectiveness (0.0 - 1.0).
    pub effectiveness: f64,
    /// Description.
    pub description: String,
}

impl Mitigation {
    pub fn new(mt: MitigationType) -> Self {
        Self {
            mitigation_type: mt,
            addresses_nodes: Vec::new(),
            cost: 0.5,
            effectiveness: 0.5,
            description: String::new(),
        }
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost.clamp(0.0, 1.0);
        self
    }

    pub fn with_effectiveness(mut self, eff: f64) -> Self {
        self.effectiveness = eff.clamp(0.0, 1.0);
        self
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn for_nodes(mut self, nodes: Vec<usize>) -> Self {
        self.addresses_nodes = nodes;
        self
    }

    /// Cost-effectiveness ratio (higher is better).
    pub fn cost_effectiveness(&self) -> f64 {
        if self.cost <= 0.0 { return self.effectiveness * 100.0; }
        self.effectiveness / self.cost
    }
}

impl fmt::Display for Mitigation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (cost={:.2}, effectiveness={:.2}, ratio={:.2})",
            self.mitigation_type, self.cost, self.effectiveness, self.cost_effectiveness())
    }
}

/// Engine for recommending mitigations.
#[derive(Debug, Clone)]
pub struct MitigationEngine {
    mitigations: Vec<Mitigation>,
}

impl MitigationEngine {
    pub fn new() -> Self {
        Self { mitigations: Vec::new() }
    }

    /// Generate mitigations for an attack graph.
    pub fn generate_mitigations(&mut self, graph: &AttackGraph) -> Vec<Mitigation> {
        self.mitigations.clear();

        for node in graph.nodes() {
            match node.node_type {
                AttackNodeType::RaceCondition => {
                    self.mitigations.push(
                        Mitigation::new(MitigationType::FenceInsertion {
                            fence_type: "full".to_string(),
                            position: format!("before access at node {}", node.id),
                        })
                        .with_cost(0.3)
                        .with_effectiveness(0.9)
                        .with_description("Insert fence to prevent race condition")
                        .for_nodes(vec![node.id])
                    );
                    self.mitigations.push(
                        Mitigation::new(MitigationType::AddSynchronization {
                            sync_type: "mutex".to_string(),
                        })
                        .with_cost(0.5)
                        .with_effectiveness(1.0)
                        .with_description("Add mutex synchronization")
                        .for_nodes(vec![node.id])
                    );
                }
                AttackNodeType::InformationLeak => {
                    self.mitigations.push(
                        Mitigation::new(MitigationType::ConstantTime {
                            operation: "memory access".to_string(),
                        })
                        .with_cost(0.4)
                        .with_effectiveness(0.85)
                        .with_description("Use constant-time operations")
                        .for_nodes(vec![node.id])
                    );
                    self.mitigations.push(
                        Mitigation::new(MitigationType::CachePadding { size: 64 })
                        .with_cost(0.1)
                        .with_effectiveness(0.6)
                        .with_description("Apply cache line padding")
                        .for_nodes(vec![node.id])
                    );
                }
                AttackNodeType::ScopeViolation => {
                    if let Some(ref scope) = node.scope {
                        self.mitigations.push(
                            Mitigation::new(MitigationType::ScopeRestriction {
                                from_scope: scope.clone(),
                                to_scope: "workgroup".to_string(),
                            })
                            .with_cost(0.2)
                            .with_effectiveness(0.95)
                            .with_description("Restrict memory scope")
                            .for_nodes(vec![node.id])
                        );
                    }
                }
                AttackNodeType::FenceBypass => {
                    self.mitigations.push(
                        Mitigation::new(MitigationType::FenceInsertion {
                            fence_type: "system".to_string(),
                            position: format!("at node {}", node.id),
                        })
                        .with_cost(0.6)
                        .with_effectiveness(0.95)
                        .with_description("Insert stronger fence")
                        .for_nodes(vec![node.id])
                    );
                }
                _ => {}
            }
        }

        // Sort by cost-effectiveness.
        self.mitigations.sort_by(|a, b| {
            b.cost_effectiveness().partial_cmp(&a.cost_effectiveness())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.mitigations.clone()
    }

    /// Get the top k mitigations by cost-effectiveness.
    pub fn top_mitigations(&self, k: usize) -> Vec<&Mitigation> {
        self.mitigations.iter().take(k).collect()
    }

    /// Get total cost of all recommended mitigations.
    pub fn total_cost(&self) -> f64 {
        self.mitigations.iter().map(|m| m.cost).sum()
    }
}

impl Default for MitigationEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Attack Pattern Database
// ---------------------------------------------------------------------------

/// Known attack patterns for GPU programs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttackPattern {
    /// Race condition on shared memory.
    SharedMemoryRace,
    /// Cache timing side channel.
    CacheTimingLeak,
    /// Warp shuffle information leak.
    WarpShuffleLeak,
    /// Rowhammer-like bit flip attack.
    RowhammerAttack,
    /// Covert channel through contention.
    ContentionCovertChannel,
    /// Speculative execution leak.
    SpeculativeExecutionLeak,
    /// Memory scope escape.
    ScopeEscape,
    /// Uninitialized memory read.
    UninitializedRead,
    /// Double-free or use-after-free.
    UseAfterFree,
    /// Integer overflow in address calculation.
    AddressOverflow,
}

impl AttackPattern {
    pub fn all() -> Vec<Self> {
        vec![
            Self::SharedMemoryRace, Self::CacheTimingLeak,
            Self::WarpShuffleLeak, Self::RowhammerAttack,
            Self::ContentionCovertChannel, Self::SpeculativeExecutionLeak,
            Self::ScopeEscape, Self::UninitializedRead,
            Self::UseAfterFree, Self::AddressOverflow,
        ]
    }

    pub fn severity(&self) -> SeverityLevel {
        match self {
            Self::SharedMemoryRace => SeverityLevel::High,
            Self::CacheTimingLeak => SeverityLevel::Medium,
            Self::WarpShuffleLeak => SeverityLevel::Medium,
            Self::RowhammerAttack => SeverityLevel::Critical,
            Self::ContentionCovertChannel => SeverityLevel::Low,
            Self::SpeculativeExecutionLeak => SeverityLevel::High,
            Self::ScopeEscape => SeverityLevel::Critical,
            Self::UninitializedRead => SeverityLevel::Medium,
            Self::UseAfterFree => SeverityLevel::Critical,
            Self::AddressOverflow => SeverityLevel::High,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::SharedMemoryRace => "Race condition on GPU shared memory",
            Self::CacheTimingLeak => "Cache timing side channel leaking data",
            Self::WarpShuffleLeak => "Information leak via warp shuffle operations",
            Self::RowhammerAttack => "Bit flip attack via repeated memory access",
            Self::ContentionCovertChannel => "Covert channel through resource contention",
            Self::SpeculativeExecutionLeak => "Data leak via speculative execution",
            Self::ScopeEscape => "Memory access escaping intended scope",
            Self::UninitializedRead => "Reading uninitialized GPU memory",
            Self::UseAfterFree => "Accessing freed GPU memory",
            Self::AddressOverflow => "Integer overflow in address computation",
        }
    }
}

impl fmt::Display for AttackPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// Database of known attack patterns.
#[derive(Debug, Clone)]
pub struct AttackPatternDB {
    patterns: Vec<AttackPattern>,
}

impl AttackPatternDB {
    /// Create a database with all known patterns.
    pub fn new() -> Self {
        Self { patterns: AttackPattern::all() }
    }

    /// Match patterns against an attack graph.
    pub fn match_patterns(&self, graph: &AttackGraph) -> Vec<PatternMatch> {
        let mut matches = Vec::new();

        for pattern in &self.patterns {
            let matched_nodes = self.find_matching_nodes(graph, pattern);
            if !matched_nodes.is_empty() {
                matches.push(PatternMatch {
                    pattern: pattern.clone(),
                    matched_nodes,
                    confidence: 0.8,
                });
            }
        }

        matches
    }

    fn find_matching_nodes(&self, graph: &AttackGraph, pattern: &AttackPattern) -> Vec<usize> {
        let mut nodes = Vec::new();

        for node in graph.nodes() {
            let matches = match pattern {
                AttackPattern::SharedMemoryRace =>
                    node.node_type == AttackNodeType::RaceCondition && node.tags.contains(&"shared_memory".to_string()),
                AttackPattern::CacheTimingLeak =>
                    node.node_type == AttackNodeType::InformationLeak && node.tags.contains(&"cache".to_string()),
                AttackPattern::WarpShuffleLeak =>
                    node.node_type == AttackNodeType::InformationLeak && node.tags.contains(&"warp_shuffle".to_string()),
                AttackPattern::ScopeEscape =>
                    node.node_type == AttackNodeType::ScopeViolation,
                AttackPattern::UninitializedRead =>
                    node.node_type == AttackNodeType::MemoryAccess && node.tags.contains(&"uninitialized".to_string()),
                _ => {
                    // Generic matching by node type.
                    match (pattern, node.node_type) {
                        (AttackPattern::RowhammerAttack, AttackNodeType::MemoryAccess) =>
                            node.tags.contains(&"repeated_access".to_string()),
                        (AttackPattern::ContentionCovertChannel, AttackNodeType::InformationLeak) =>
                            node.tags.contains(&"contention".to_string()),
                        (AttackPattern::SpeculativeExecutionLeak, AttackNodeType::InformationLeak) =>
                            node.tags.contains(&"speculative".to_string()),
                        (AttackPattern::UseAfterFree, AttackNodeType::MemoryAccess) =>
                            node.tags.contains(&"use_after_free".to_string()),
                        (AttackPattern::AddressOverflow, AttackNodeType::MemoryAccess) =>
                            node.tags.contains(&"overflow".to_string()),
                        _ => false,
                    }
                }
            };

            if matches {
                nodes.push(node.id);
            }
        }

        nodes
    }

    /// Get all patterns.
    pub fn patterns(&self) -> &[AttackPattern] {
        &self.patterns
    }
}

impl Default for AttackPatternDB {
    fn default() -> Self {
        Self::new()
    }
}

/// A match of a pattern against the attack graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    pub pattern: AttackPattern,
    pub matched_nodes: Vec<usize>,
    pub confidence: f64,
}

impl fmt::Display for PatternMatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}) matched at {} nodes (confidence={:.2})",
            self.pattern, self.pattern.severity(),
            self.matched_nodes.len(), self.confidence)
    }
}

// ---------------------------------------------------------------------------
// Threat Model
// ---------------------------------------------------------------------------

/// Attacker capability model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThreatModel {
    /// Can observe memory access patterns and timing.
    PassiveObserver,
    /// Can actively manipulate memory contents.
    ActiveManipulator,
    /// Co-resident tenant on the same GPU.
    CoResidentTenant,
}

impl ThreatModel {
    /// Can the attacker read shared memory?
    pub fn can_read_shared_memory(&self) -> bool {
        matches!(self, Self::ActiveManipulator | Self::CoResidentTenant)
    }

    /// Can the attacker write shared memory?
    pub fn can_write_shared_memory(&self) -> bool {
        matches!(self, Self::ActiveManipulator)
    }

    /// Can the attacker observe timing?
    pub fn can_observe_timing(&self) -> bool {
        true // All threat models can observe timing.
    }

    /// Can the attacker co-locate code?
    pub fn can_colocate(&self) -> bool {
        matches!(self, Self::CoResidentTenant)
    }
}

impl fmt::Display for ThreatModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PassiveObserver => write!(f, "Passive Observer"),
            Self::ActiveManipulator => write!(f, "Active Manipulator"),
            Self::CoResidentTenant => write!(f, "Co-Resident Tenant"),
        }
    }
}

/// Analyzes which attacks are feasible under a given threat model.
#[derive(Debug, Clone)]
pub struct AttackFeasibilityAnalyzer {
    threat_model: ThreatModel,
}

impl AttackFeasibilityAnalyzer {
    pub fn new(threat_model: ThreatModel) -> Self {
        Self { threat_model }
    }

    /// Check if a pattern is feasible under the current threat model.
    pub fn is_feasible(&self, pattern: &AttackPattern) -> bool {
        match pattern {
            AttackPattern::SharedMemoryRace =>
                self.threat_model.can_read_shared_memory(),
            AttackPattern::CacheTimingLeak =>
                self.threat_model.can_observe_timing(),
            AttackPattern::WarpShuffleLeak =>
                self.threat_model.can_colocate(),
            AttackPattern::RowhammerAttack =>
                self.threat_model.can_write_shared_memory(),
            AttackPattern::ContentionCovertChannel =>
                self.threat_model.can_observe_timing(),
            AttackPattern::SpeculativeExecutionLeak =>
                self.threat_model.can_observe_timing(),
            AttackPattern::ScopeEscape =>
                self.threat_model.can_colocate(),
            AttackPattern::UninitializedRead =>
                self.threat_model.can_read_shared_memory(),
            AttackPattern::UseAfterFree =>
                self.threat_model.can_read_shared_memory(),
            AttackPattern::AddressOverflow =>
                self.threat_model.can_write_shared_memory(),
        }
    }

    /// Filter patterns to only feasible ones.
    pub fn feasible_patterns(&self, patterns: &[AttackPattern]) -> Vec<AttackPattern> {
        patterns.iter()
            .filter(|p| self.is_feasible(p))
            .cloned()
            .collect()
    }

    /// Analyze attack graph feasibility.
    pub fn analyze_graph(&self, graph: &AttackGraph) -> FeasibilityReport {
        let total_nodes = graph.node_count();
        let total_edges = graph.edge_count();

        let feasible_nodes: Vec<usize> = graph.nodes().iter()
            .filter(|n| self.is_node_feasible(n))
            .map(|n| n.id)
            .collect();

        let total_paths = AttackPathEnumerator::new(100, 15)
            .enumerate_all(graph);
        let feasible_paths: Vec<AttackPath> = total_paths.iter()
            .filter(|path| path.nodes.iter().all(|&n| {
                graph.node(n).map_or(true, |node| self.is_node_feasible(node))
            }))
            .cloned()
            .collect();

        FeasibilityReport {
            threat_model: self.threat_model,
            total_nodes,
            feasible_nodes: feasible_nodes.len(),
            total_paths: total_paths.len(),
            feasible_paths: feasible_paths.len(),
            max_feasible_risk: feasible_paths.iter()
                .map(|p| p.risk_score())
                .fold(0.0f64, f64::max),
        }
    }

    fn is_node_feasible(&self, node: &AttackNode) -> bool {
        match node.node_type {
            AttackNodeType::EntryPoint | AttackNodeType::ExitPoint => true,
            AttackNodeType::MemoryAccess =>
                self.threat_model.can_read_shared_memory(),
            AttackNodeType::RaceCondition =>
                self.threat_model.can_read_shared_memory(),
            AttackNodeType::InformationLeak =>
                self.threat_model.can_observe_timing(),
            AttackNodeType::ScopeViolation =>
                self.threat_model.can_colocate(),
            AttackNodeType::FenceBypass =>
                self.threat_model.can_colocate(),
            AttackNodeType::Escalation =>
                self.threat_model.can_write_shared_memory(),
        }
    }
}

/// Feasibility analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeasibilityReport {
    pub threat_model: ThreatModel,
    pub total_nodes: usize,
    pub feasible_nodes: usize,
    pub total_paths: usize,
    pub feasible_paths: usize,
    pub max_feasible_risk: f64,
}

impl fmt::Display for FeasibilityReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Feasibility Report ({})", self.threat_model)?;
        writeln!(f, "  Nodes: {}/{} feasible", self.feasible_nodes, self.total_nodes)?;
        writeln!(f, "  Paths: {}/{} feasible", self.feasible_paths, self.total_paths)?;
        writeln!(f, "  Max risk: {:.2}", self.max_feasible_risk)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_simple_graph() -> AttackGraph {
        let mut graph = AttackGraph::new();
        let entry = graph.add_node(AttackNode::new(0, AttackNodeType::EntryPoint, "Start"));
        let race = graph.add_node(
            AttackNode::new(0, AttackNodeType::RaceCondition, "Shared mem race")
                .with_severity(SeverityLevel::High)
        );
        let leak = graph.add_node(
            AttackNode::new(0, AttackNodeType::InformationLeak, "Cache timing")
                .with_severity(SeverityLevel::Medium)
        );
        let exit = graph.add_node(AttackNode::new(0, AttackNodeType::ExitPoint, "Data leaked"));

        graph.add_edge(AttackEdge::new(entry, race).with_probability(0.8));
        graph.add_edge(AttackEdge::new(race, leak).with_probability(0.6));
        graph.add_edge(AttackEdge::new(leak, exit).with_probability(0.9));
        graph.add_edge(AttackEdge::new(entry, leak).with_probability(0.3));
        graph
    }

    // -- SeverityLevel tests --

    #[test]
    fn test_severity_ordering() {
        assert!(SeverityLevel::Critical > SeverityLevel::High);
        assert!(SeverityLevel::High > SeverityLevel::Medium);
        assert!(SeverityLevel::Medium > SeverityLevel::Low);
        assert!(SeverityLevel::Low > SeverityLevel::Info);
    }

    #[test]
    fn test_severity_from_score() {
        assert_eq!(SeverityLevel::from_score(9.5), SeverityLevel::Critical);
        assert_eq!(SeverityLevel::from_score(7.5), SeverityLevel::High);
        assert_eq!(SeverityLevel::from_score(5.0), SeverityLevel::Medium);
        assert_eq!(SeverityLevel::from_score(2.0), SeverityLevel::Low);
        assert_eq!(SeverityLevel::from_score(0.0), SeverityLevel::Info);
    }

    // -- VulnerabilityScore tests --

    #[test]
    fn test_vuln_score() {
        let score = VulnerabilityScore::new(8.0);
        assert_eq!(score.severity(), SeverityLevel::High);
        assert_eq!(score.overall(), 8.0);
    }

    #[test]
    fn test_vuln_score_with_modifiers() {
        let score = VulnerabilityScore::new(8.0)
            .with_temporal(0.5)
            .with_environmental(0.8);
        assert!((score.overall() - 3.2).abs() < 0.01);
    }

    // -- AttackGraph tests --

    #[test]
    fn test_graph_creation() {
        let graph = build_simple_graph();
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.edge_count(), 4);
    }

    #[test]
    fn test_graph_entry_exit() {
        let graph = build_simple_graph();
        assert_eq!(graph.entry_nodes().len(), 1);
        assert_eq!(graph.exit_nodes().len(), 1);
    }

    #[test]
    fn test_graph_path_exists() {
        let graph = build_simple_graph();
        assert!(graph.path_exists(0, 3));
        assert!(!graph.path_exists(3, 0));
    }

    #[test]
    fn test_graph_shortest_path() {
        let graph = build_simple_graph();
        let path = graph.shortest_path(0, 3).unwrap();
        assert!(path.len() >= 2);
        assert_eq!(path[0], 0);
        assert_eq!(*path.last().unwrap(), 3);
    }

    #[test]
    fn test_graph_neighbors() {
        let graph = build_simple_graph();
        let neighbors = graph.neighbors(0);
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_graph_predecessors() {
        let graph = build_simple_graph();
        let preds = graph.predecessors(3);
        assert_eq!(preds.len(), 1);
    }

    #[test]
    fn test_graph_max_severity() {
        let graph = build_simple_graph();
        assert_eq!(graph.max_severity(), SeverityLevel::High);
    }

    // -- AttackGraphBuilder tests --

    #[test]
    fn test_builder() {
        let mut builder = AttackGraphBuilder::new();
        let entry = builder.add_entry("Start");
        let exit = builder.add_exit("End");
        builder.connect(entry, exit, "direct", 1.0);
        let graph = builder.build();
        assert_eq!(graph.node_count(), 2);
        assert!(graph.path_exists(entry, exit));
    }

    #[test]
    fn test_builder_race_detection() {
        let mut builder = AttackGraphBuilder::new();
        builder.record_access(0x100, 0, "write", 0);
        builder.record_access(0x100, 1, "read", 1);
        let races = builder.detect_races();
        assert!(!races.is_empty());
    }

    // -- AttackPathEnumerator tests --

    #[test]
    fn test_path_enumeration() {
        let graph = build_simple_graph();
        let enumerator = AttackPathEnumerator::new(100, 10);
        let paths = enumerator.enumerate_all(&graph);
        assert!(paths.len() >= 2); // Two paths: entry->race->leak->exit and entry->leak->exit
    }

    #[test]
    fn test_k_shortest_paths() {
        let graph = build_simple_graph();
        let enumerator = AttackPathEnumerator::new(100, 10);
        let paths = enumerator.k_shortest_paths(&graph, 1);
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn test_highest_risk_paths() {
        let graph = build_simple_graph();
        let enumerator = AttackPathEnumerator::new(100, 10);
        let paths = enumerator.highest_risk_paths(&graph, 2);
        assert!(!paths.is_empty());
        if paths.len() >= 2 {
            assert!(paths[0].risk_score() >= paths[1].risk_score());
        }
    }

    // -- AttackSurfaceAnalyzer tests --

    #[test]
    fn test_attack_surface() {
        let graph = build_simple_graph();
        let analyzer = AttackSurfaceAnalyzer::new();
        let surface = analyzer.analyze(&graph);
        assert_eq!(surface.entry_points, 1);
        assert_eq!(surface.exit_points, 1);
        assert!(surface.reachable_vulnerabilities >= 2);
        assert!(surface.total_paths >= 2);
    }

    // -- MitigationEngine tests --

    #[test]
    fn test_mitigation_generation() {
        let graph = build_simple_graph();
        let mut engine = MitigationEngine::new();
        let mitigations = engine.generate_mitigations(&graph);
        assert!(!mitigations.is_empty());
    }

    #[test]
    fn test_mitigation_cost_effectiveness() {
        let m = Mitigation::new(MitigationType::FenceInsertion {
            fence_type: "full".to_string(),
            position: "before store".to_string(),
        }).with_cost(0.2).with_effectiveness(0.8);
        assert_eq!(m.cost_effectiveness(), 4.0);
    }

    // -- AttackPatternDB tests --

    #[test]
    fn test_pattern_db() {
        let db = AttackPatternDB::new();
        assert_eq!(db.patterns().len(), 10);
    }

    #[test]
    fn test_pattern_match() {
        let mut graph = AttackGraph::new();
        graph.add_node(
            AttackNode::new(0, AttackNodeType::RaceCondition, "race")
                .with_tag("shared_memory")
        );
        let db = AttackPatternDB::new();
        let matches = db.match_patterns(&graph);
        assert!(matches.iter().any(|m| m.pattern == AttackPattern::SharedMemoryRace));
    }

    // -- ThreatModel tests --

    #[test]
    fn test_threat_model_passive() {
        let tm = ThreatModel::PassiveObserver;
        assert!(tm.can_observe_timing());
        assert!(!tm.can_read_shared_memory());
        assert!(!tm.can_write_shared_memory());
    }

    #[test]
    fn test_threat_model_active() {
        let tm = ThreatModel::ActiveManipulator;
        assert!(tm.can_read_shared_memory());
        assert!(tm.can_write_shared_memory());
    }

    #[test]
    fn test_threat_model_coresident() {
        let tm = ThreatModel::CoResidentTenant;
        assert!(tm.can_colocate());
        assert!(tm.can_read_shared_memory());
        assert!(!tm.can_write_shared_memory());
    }

    // -- AttackFeasibilityAnalyzer tests --

    #[test]
    fn test_feasibility_passive() {
        let analyzer = AttackFeasibilityAnalyzer::new(ThreatModel::PassiveObserver);
        assert!(analyzer.is_feasible(&AttackPattern::CacheTimingLeak));
        assert!(!analyzer.is_feasible(&AttackPattern::SharedMemoryRace));
    }

    #[test]
    fn test_feasibility_active() {
        let analyzer = AttackFeasibilityAnalyzer::new(ThreatModel::ActiveManipulator);
        assert!(analyzer.is_feasible(&AttackPattern::SharedMemoryRace));
        assert!(analyzer.is_feasible(&AttackPattern::RowhammerAttack));
    }

    #[test]
    fn test_feasibility_filter() {
        let analyzer = AttackFeasibilityAnalyzer::new(ThreatModel::PassiveObserver);
        let feasible = analyzer.feasible_patterns(&AttackPattern::all());
        assert!(!feasible.is_empty());
        // Passive observer can only do timing-based attacks.
        assert!(feasible.contains(&AttackPattern::CacheTimingLeak));
    }

    #[test]
    fn test_feasibility_report() {
        let graph = build_simple_graph();
        let analyzer = AttackFeasibilityAnalyzer::new(ThreatModel::ActiveManipulator);
        let report = analyzer.analyze_graph(&graph);
        assert!(report.feasible_nodes > 0);
    }
}
