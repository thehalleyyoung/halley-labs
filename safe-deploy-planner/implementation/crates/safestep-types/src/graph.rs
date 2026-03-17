// Version-product graph and state types for the SafeStep deployment planner.

use std::collections::VecDeque;
use std::fmt;
use std::hash::{Hash, Hasher};

use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use smallvec::SmallVec;

use crate::error::{Result, SafeStepError};
use crate::identifiers::StateId;
use crate::version::{VersionIndex, VersionSet};

/// A cluster state is a vector of version indices, one per service.
/// ClusterState[i] = the current version index of service i.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClusterState {
    versions: SmallVec<[VersionIndex; 8]>,
}

impl ClusterState {
    /// Create from a slice of version indices.
    pub fn new(versions: &[VersionIndex]) -> Self {
        Self {
            versions: SmallVec::from_slice(versions),
        }
    }

    /// Create an all-zeros state (first version of each service).
    pub fn initial(num_services: usize) -> Self {
        Self {
            versions: SmallVec::from_elem(VersionIndex(0), num_services),
        }
    }

    /// Number of services.
    pub fn num_services(&self) -> usize {
        self.versions.len()
    }

    /// Get the version index for a service.
    pub fn get(&self, service_idx: usize) -> VersionIndex {
        self.versions[service_idx]
    }

    /// Set the version index for a service.
    pub fn set(&mut self, service_idx: usize, version: VersionIndex) {
        self.versions[service_idx] = version;
    }

    /// Create a new state by upgrading one service.
    pub fn with_upgrade(&self, service_idx: usize, new_version: VersionIndex) -> Self {
        let mut new = self.clone();
        new.versions[service_idx] = new_version;
        new
    }

    /// Return all services as a slice.
    pub fn as_slice(&self) -> &[VersionIndex] {
        &self.versions
    }

    /// Hamming distance: number of services at different versions.
    pub fn hamming_distance(&self, other: &ClusterState) -> usize {
        self.versions
            .iter()
            .zip(other.versions.iter())
            .filter(|(a, b)| a != b)
            .count()
    }

    /// Check if this state is monotonically >= another (no service downgraded).
    pub fn is_monotone_from(&self, other: &ClusterState) -> bool {
        self.versions
            .iter()
            .zip(other.versions.iter())
            .all(|(a, b)| a.0 >= b.0)
    }

    /// Services that differ between this state and another.
    pub fn differing_services(&self, other: &ClusterState) -> Vec<usize> {
        self.versions
            .iter()
            .zip(other.versions.iter())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| i)
            .collect()
    }

    /// Compute a content-addressed state ID.
    pub fn state_id(&self) -> StateId {
        let mut hasher = Sha256::new();
        for v in &self.versions {
            hasher.update(v.0.to_le_bytes());
        }
        let hash = hasher.finalize();
        StateId::new(hex::encode(&hash[..16]))
    }

    /// Compact integer encoding of this state given max version counts per service.
    pub fn to_flat_index(&self, version_counts: &[u32]) -> u64 {
        let mut idx = 0u64;
        let mut multiplier = 1u64;
        for (i, vi) in self.versions.iter().enumerate() {
            idx += vi.0 as u64 * multiplier;
            multiplier *= version_counts[i] as u64;
        }
        idx
    }

    /// Decode from flat index.
    pub fn from_flat_index(mut idx: u64, version_counts: &[u32]) -> Self {
        let mut versions = SmallVec::new();
        for &count in version_counts {
            let c = count as u64;
            versions.push(VersionIndex((idx % c) as u32));
            idx /= c;
        }
        Self { versions }
    }
}

impl Hash for ClusterState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for v in &self.versions {
            v.hash(state);
        }
    }
}

impl fmt::Debug for ClusterState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "State[")?;
        for (i, v) in self.versions.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", v.0)?;
        }
        write!(f, "]")
    }
}

impl fmt::Display for ClusterState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, v) in self.versions.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "v{}", v.0)?;
        }
        write!(f, ")")
    }
}

// ─── GraphEdge ──────────────────────────────────────────────────────────

/// A labeled edge in the version-product graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphEdge {
    pub service_idx: usize,
    pub from_version: VersionIndex,
    pub to_version: VersionIndex,
}

impl GraphEdge {
    pub fn new(service_idx: usize, from: VersionIndex, to: VersionIndex) -> Self {
        Self {
            service_idx,
            from_version: from,
            to_version: to,
        }
    }

    /// Is this a forward (upgrade) transition?
    pub fn is_upgrade(&self) -> bool {
        self.to_version.0 > self.from_version.0
    }

    /// Is this a backward (downgrade) transition?
    pub fn is_downgrade(&self) -> bool {
        self.to_version.0 < self.from_version.0
    }

    /// The version step size (positive for upgrades, negative for downgrades).
    pub fn step_size(&self) -> i64 {
        self.to_version.0 as i64 - self.from_version.0 as i64
    }
}

impl fmt::Display for GraphEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "svc[{}]: {} -> {}",
            self.service_idx, self.from_version, self.to_version
        )
    }
}

// ─── Safety predicate trait ──────────────────────────────────────────────

/// Trait for checking if a cluster state is safe (satisfies all invariants).
pub trait SafetyPredicate: Send + Sync {
    fn is_safe(&self, state: &ClusterState) -> bool;
    fn check_safety(&self, state: &ClusterState) -> Result<()> {
        if self.is_safe(state) {
            Ok(())
        } else {
            Err(SafeStepError::constraint_violation(format!(
                "State {} violates safety predicate",
                state
            )))
        }
    }
}

/// A safety predicate that always returns true (for testing).
pub struct TruePredicate;

impl SafetyPredicate for TruePredicate {
    fn is_safe(&self, _state: &ClusterState) -> bool {
        true
    }
}

/// A safety predicate based on a closure.
pub struct ClosurePredicate<F: Fn(&ClusterState) -> bool + Send + Sync> {
    pred: F,
}

impl<F: Fn(&ClusterState) -> bool + Send + Sync> ClosurePredicate<F> {
    pub fn new(pred: F) -> Self {
        Self { pred }
    }
}

impl<F: Fn(&ClusterState) -> bool + Send + Sync> SafetyPredicate for ClosurePredicate<F> {
    fn is_safe(&self, state: &ClusterState) -> bool {
        (self.pred)(state)
    }
}

// ─── Transition constraints ──────────────────────────────────────────────

/// Constraint on allowed transitions.
pub trait TransitionConstraint: Send + Sync {
    fn is_allowed(&self, from: &ClusterState, edge: &GraphEdge, to: &ClusterState) -> bool;
}

/// Only allows monotone (non-downgrading) transitions.
pub struct MonotoneFilter;

impl TransitionConstraint for MonotoneFilter {
    fn is_allowed(&self, _from: &ClusterState, edge: &GraphEdge, _to: &ClusterState) -> bool {
        edge.to_version.0 >= edge.from_version.0
    }
}

/// Only allows single-step version transitions (increment or decrement by 1).
pub struct SingleStepFilter;

impl TransitionConstraint for SingleStepFilter {
    fn is_allowed(&self, _from: &ClusterState, edge: &GraphEdge, _to: &ClusterState) -> bool {
        let diff = (edge.to_version.0 as i64 - edge.from_version.0 as i64).unsigned_abs();
        diff == 1
    }
}

/// Combines multiple transition constraints (all must pass).
pub struct CompositeFilter {
    filters: Vec<Box<dyn TransitionConstraint>>,
}

impl CompositeFilter {
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    pub fn add(mut self, filter: Box<dyn TransitionConstraint>) -> Self {
        self.filters.push(filter);
        self
    }
}

impl Default for CompositeFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl TransitionConstraint for CompositeFilter {
    fn is_allowed(&self, from: &ClusterState, edge: &GraphEdge, to: &ClusterState) -> bool {
        self.filters
            .iter()
            .all(|f| f.is_allowed(from, edge, to))
    }
}

// ─── Adjacency oracle ───────────────────────────────────────────────────

/// Lazy adjacency computation for implicit graphs.
pub trait AdjacencyOracle: Send + Sync {
    /// Return all valid neighbor states from a given state.
    fn neighbors(&self, state: &ClusterState) -> Vec<(GraphEdge, ClusterState)>;
}

/// Default adjacency oracle: all single-service transitions.
pub struct DefaultAdjacencyOracle {
    version_counts: Vec<u32>,
    safety: Box<dyn SafetyPredicate>,
    filter: Option<Box<dyn TransitionConstraint>>,
}

impl DefaultAdjacencyOracle {
    pub fn new(
        version_counts: Vec<u32>,
        safety: Box<dyn SafetyPredicate>,
    ) -> Self {
        Self {
            version_counts,
            safety,
            filter: None,
        }
    }

    pub fn with_filter(mut self, filter: Box<dyn TransitionConstraint>) -> Self {
        self.filter = Some(filter);
        self
    }
}

impl AdjacencyOracle for DefaultAdjacencyOracle {
    fn neighbors(&self, state: &ClusterState) -> Vec<(GraphEdge, ClusterState)> {
        let mut result = Vec::new();
        for (svc_idx, &max_ver) in self.version_counts.iter().enumerate() {
            let current = state.get(svc_idx);
            for ver in 0..max_ver {
                let vi = VersionIndex(ver);
                if vi == current {
                    continue;
                }
                let edge = GraphEdge::new(svc_idx, current, vi);
                let new_state = state.with_upgrade(svc_idx, vi);
                if let Some(ref filter) = self.filter {
                    if !filter.is_allowed(state, &edge, &new_state) {
                        continue;
                    }
                }
                if self.safety.is_safe(&new_state) {
                    result.push((edge, new_state));
                }
            }
        }
        result
    }
}

// ─── VersionProductGraph ─────────────────────────────────────────────────

/// The explicit version-product graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionProductGraph {
    /// Number of services.
    pub num_services: usize,
    /// Version counts per service (size of each service's version set).
    pub version_counts: Vec<u32>,
    /// Service names for display.
    pub service_names: Vec<String>,
    /// Adjacency list: state -> edges.
    adjacency: HashMap<ClusterState, Vec<(GraphEdge, ClusterState)>>,
    /// Set of safe states.
    safe_states: HashSet<ClusterState>,
    /// Total states explored.
    explored_count: usize,
}

impl VersionProductGraph {
    pub fn new(version_sets: &[VersionSet]) -> Self {
        let num_services = version_sets.len();
        let version_counts: Vec<u32> = version_sets.iter().map(|vs| vs.len() as u32).collect();
        let service_names: Vec<String> = version_sets
            .iter()
            .map(|vs| vs.service_name().to_string())
            .collect();
        Self {
            num_services,
            version_counts,
            service_names,
            adjacency: HashMap::new(),
            safe_states: HashSet::new(),
            explored_count: 0,
        }
    }

    /// Total number of possible states in the product graph.
    pub fn total_state_space_size(&self) -> u64 {
        self.version_counts
            .iter()
            .map(|&c| c as u64)
            .product()
    }

    /// Number of explored states.
    pub fn explored_count(&self) -> usize {
        self.explored_count
    }

    /// Number of known safe states.
    pub fn safe_state_count(&self) -> usize {
        self.safe_states.len()
    }

    /// Add a safe state and its adjacency.
    pub fn add_state(
        &mut self,
        state: ClusterState,
        neighbors: Vec<(GraphEdge, ClusterState)>,
    ) {
        self.safe_states.insert(state.clone());
        self.adjacency.insert(state, neighbors);
        self.explored_count += 1;
    }

    /// Check if a state has been explored.
    pub fn is_explored(&self, state: &ClusterState) -> bool {
        self.adjacency.contains_key(state)
    }

    /// Check if a state is known to be safe.
    pub fn is_safe(&self, state: &ClusterState) -> bool {
        self.safe_states.contains(state)
    }

    /// Get neighbors of a state.
    pub fn neighbors(&self, state: &ClusterState) -> &[(GraphEdge, ClusterState)] {
        self.adjacency
            .get(state)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// BFS exploration from a start state.
    pub fn explore_bfs(
        &mut self,
        start: &ClusterState,
        oracle: &dyn AdjacencyOracle,
        max_states: usize,
    ) {
        let mut queue = VecDeque::new();
        if !self.is_explored(start) {
            let nbrs = oracle.neighbors(start);
            self.add_state(start.clone(), nbrs.clone());
            for (_, neighbor) in &nbrs {
                queue.push_back(neighbor.clone());
            }
        }

        while let Some(state) = queue.pop_front() {
            if self.explored_count >= max_states {
                break;
            }
            if self.is_explored(&state) {
                continue;
            }
            let nbrs = oracle.neighbors(&state);
            for (_, neighbor) in &nbrs {
                if !self.is_explored(neighbor) {
                    queue.push_back(neighbor.clone());
                }
            }
            self.add_state(state, nbrs);
        }
    }

    /// Forward reachability: states reachable from `start` via safe transitions.
    pub fn forward_reachable(&self, start: &ClusterState) -> HashSet<ClusterState> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(start.clone());
        queue.push_back(start.clone());

        while let Some(state) = queue.pop_front() {
            for (_, neighbor) in self.neighbors(&state) {
                if !visited.contains(neighbor) {
                    visited.insert(neighbor.clone());
                    queue.push_back(neighbor.clone());
                }
            }
        }
        visited
    }

    /// Backward reachability: states from which `target` is reachable.
    pub fn backward_reachable(&self, target: &ClusterState) -> HashSet<ClusterState> {
        // Build reverse adjacency
        let mut reverse: HashMap<ClusterState, Vec<ClusterState>> = HashMap::new();
        for (state, neighbors) in &self.adjacency {
            for (_, neighbor) in neighbors {
                reverse
                    .entry(neighbor.clone())
                    .or_default()
                    .push(state.clone());
            }
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(target.clone());
        queue.push_back(target.clone());

        while let Some(state) = queue.pop_front() {
            if let Some(preds) = reverse.get(&state) {
                for pred in preds {
                    if !visited.contains(pred) {
                        visited.insert(pred.clone());
                        queue.push_back(pred.clone());
                    }
                }
            }
        }
        visited
    }

    /// Find shortest path from start to goal using BFS.
    pub fn shortest_path(
        &self,
        start: &ClusterState,
        goal: &ClusterState,
    ) -> Option<Vec<(GraphEdge, ClusterState)>> {
        if start == goal {
            return Some(Vec::new());
        }

        let mut visited: HashSet<ClusterState> = HashSet::new();
        let mut queue: VecDeque<(ClusterState, Vec<(GraphEdge, ClusterState)>)> = VecDeque::new();
        visited.insert(start.clone());
        queue.push_back((start.clone(), Vec::new()));

        while let Some((current, path)) = queue.pop_front() {
            for (edge, neighbor) in self.neighbors(&current) {
                if visited.contains(neighbor) {
                    continue;
                }
                let mut new_path = path.clone();
                new_path.push((edge.clone(), neighbor.clone()));
                if neighbor == goal {
                    return Some(new_path);
                }
                visited.insert(neighbor.clone());
                queue.push_back((neighbor.clone(), new_path));
            }
        }
        None
    }

    /// Compute the set of states in the rollback safety envelope.
    /// A state is in the envelope if both forward-to-target and backward-to-start are possible.
    pub fn safety_envelope(
        &self,
        start: &ClusterState,
        target: &ClusterState,
    ) -> HashSet<ClusterState> {
        let forward = self.forward_reachable(start);
        let backward = self.backward_reachable(start);
        let to_target = self.backward_reachable(target);

        // States from which we can reach target AND from which we can retreat to start
        forward
            .intersection(&to_target)
            .filter(|s| backward.iter().any(|x| x == *s))
            .cloned()
            .collect()
    }

    /// Find points of no return: states reachable from start, from which start is NOT reachable.
    pub fn points_of_no_return(
        &self,
        start: &ClusterState,
        target: &ClusterState,
    ) -> Vec<ClusterState> {
        let forward = self.forward_reachable(start);
        let backward = self.backward_reachable(start);
        let to_target = self.backward_reachable(target);

        forward
            .iter()
            .filter(|s| !backward.iter().any(|x| x == *s) && to_target.iter().any(|x| x == *s))
            .cloned()
            .collect()
    }

    /// Enumerate all states in the graph.
    pub fn all_states(&self) -> impl Iterator<Item = &ClusterState> {
        self.adjacency.keys()
    }

    /// Get graph statistics.
    pub fn stats(&self) -> GraphStats {
        let total_edges: usize = self.adjacency.values().map(|v| v.len()).sum();
        GraphStats {
            num_services: self.num_services,
            version_counts: self.version_counts.clone(),
            total_state_space: self.total_state_space_size(),
            explored_states: self.explored_count,
            safe_states: self.safe_states.len(),
            total_edges,
            avg_branching_factor: if self.explored_count > 0 {
                total_edges as f64 / self.explored_count as f64
            } else {
                0.0
            },
        }
    }
}

/// Statistics about the version-product graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub num_services: usize,
    pub version_counts: Vec<u32>,
    pub total_state_space: u64,
    pub explored_states: usize,
    pub safe_states: usize,
    pub total_edges: usize,
    pub avg_branching_factor: f64,
}

impl fmt::Display for GraphStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Graph: {} services, {} total states, {} explored, {} safe, {} edges (avg branching {:.2})",
            self.num_services,
            self.total_state_space,
            self.explored_states,
            self.safe_states,
            self.total_edges,
            self.avg_branching_factor,
        )
    }
}

// ─── Service dependency graph utilities ─────────────────────────────────

/// Estimate the treewidth of a service dependency graph.
/// Uses the greedy minimum-degree heuristic (upper bound).
pub fn estimate_treewidth(adjacency: &[Vec<usize>]) -> usize {
    let n = adjacency.len();
    if n == 0 {
        return 0;
    }

    let mut adj: Vec<HashSet<usize>> = adjacency
        .iter()
        .map(|nbrs| nbrs.iter().copied().collect())
        .collect();
    let mut eliminated = vec![false; n];
    let mut treewidth = 0;

    for _ in 0..n {
        // Find the vertex with minimum degree
        let min_v = (0..n)
            .filter(|&v| !eliminated[v])
            .min_by_key(|&v| adj[v].iter().filter(|&&u| !eliminated[u]).count())
            .unwrap();

        let neighbors: Vec<usize> = adj[min_v]
            .iter()
            .copied()
            .filter(|&u| !eliminated[u])
            .collect();
        let degree = neighbors.len();
        treewidth = treewidth.max(degree);

        // Make neighbors into a clique
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let u = neighbors[i];
                let v = neighbors[j];
                adj[u].insert(v);
                adj[v].insert(u);
            }
        }

        eliminated[min_v] = true;
    }

    treewidth
}

/// Build a symmetric adjacency list from directed dependency edges.
pub fn build_undirected_adjacency(
    num_services: usize,
    edges: &[(usize, usize)],
) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); num_services];
    for &(u, v) in edges {
        if u < num_services && v < num_services {
            if !adj[u].contains(&v) {
                adj[u].push(v);
            }
            if !adj[v].contains(&u) {
                adj[v].push(u);
            }
        }
    }
    adj
}

// ─── State space size estimation ─────────────────────────────────────────

/// Estimate the number of reachable safe states without full enumeration.
/// Uses random sampling.
pub fn estimate_safe_state_count(
    version_counts: &[u32],
    predicate: &dyn SafetyPredicate,
    num_samples: usize,
) -> f64 {
    let total = version_counts.iter().map(|&c| c as u64).product::<u64>();
    if total == 0 || num_samples == 0 {
        return 0.0;
    }

    let mut safe_count = 0u64;
    // Deterministic sampling using a linear congruential generator
    let mut rng_state: u64 = 0x12345678;
    for _ in 0..num_samples {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let sample_idx = rng_state % total;
        let state = ClusterState::from_flat_index(sample_idx, version_counts);
        if predicate.is_safe(&state) {
            safe_count += 1;
        }
    }

    (safe_count as f64 / num_samples as f64) * total as f64
}

/// Information about a state in the context of a plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateInfo {
    pub state: ClusterState,
    pub state_id: StateId,
    pub is_safe: bool,
    pub forward_reachable_count: Option<usize>,
    pub backward_reachable_count: Option<usize>,
    pub in_envelope: bool,
    pub is_pnr: bool,
}

impl StateInfo {
    pub fn new(state: ClusterState, is_safe: bool) -> Self {
        let state_id = state.state_id();
        Self {
            state,
            state_id,
            is_safe,
            forward_reachable_count: None,
            backward_reachable_count: None,
            in_envelope: false,
            is_pnr: false,
        }
    }
}

/// Metadata about graph construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    pub construction_time_ms: u64,
    pub memory_estimate_bytes: u64,
    pub treewidth_estimate: Option<usize>,
    pub is_complete: bool,
    pub exploration_depth: usize,
}

impl GraphMetadata {
    pub fn new() -> Self {
        Self {
            construction_time_ms: 0,
            memory_estimate_bytes: 0,
            treewidth_estimate: None,
            is_complete: false,
            exploration_depth: 0,
        }
    }
}

impl Default for GraphMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for graph exploration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationConfig {
    pub max_states: usize,
    pub max_depth: usize,
    pub monotone_only: bool,
    pub single_step_only: bool,
    pub timeout_ms: u64,
}

impl ExplorationConfig {
    pub fn new() -> Self {
        Self {
            max_states: 100_000,
            max_depth: 100,
            monotone_only: true,
            single_step_only: true,
            timeout_ms: 30_000,
        }
    }

    pub fn with_max_states(mut self, max: usize) -> Self {
        self.max_states = max;
        self
    }

    pub fn with_monotone(mut self, monotone: bool) -> Self {
        self.monotone_only = monotone;
        self
    }
}

impl Default for ExplorationConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_version_sets() -> Vec<VersionSet> {
        use crate::version::Version;
        let mut vs1 = VersionSet::new("svc-a");
        vs1.insert(Version::new(1, 0, 0));
        vs1.insert(Version::new(2, 0, 0));
        let mut vs2 = VersionSet::new("svc-b");
        vs2.insert(Version::new(1, 0, 0));
        vs2.insert(Version::new(1, 1, 0));
        vs2.insert(Version::new(2, 0, 0));
        vec![vs1, vs2]
    }

    #[test]
    fn test_cluster_state_new() {
        let state = ClusterState::new(&[VersionIndex(0), VersionIndex(1)]);
        assert_eq!(state.num_services(), 2);
        assert_eq!(state.get(0), VersionIndex(0));
        assert_eq!(state.get(1), VersionIndex(1));
    }

    #[test]
    fn test_cluster_state_initial() {
        let state = ClusterState::initial(3);
        assert_eq!(state.num_services(), 3);
        assert_eq!(state.get(0), VersionIndex(0));
        assert_eq!(state.get(2), VersionIndex(0));
    }

    #[test]
    fn test_cluster_state_upgrade() {
        let s1 = ClusterState::new(&[VersionIndex(0), VersionIndex(0)]);
        let s2 = s1.with_upgrade(0, VersionIndex(1));
        assert_eq!(s2.get(0), VersionIndex(1));
        assert_eq!(s2.get(1), VersionIndex(0));
        assert_eq!(s1.get(0), VersionIndex(0)); // original unchanged
    }

    #[test]
    fn test_cluster_state_hamming() {
        let s1 = ClusterState::new(&[VersionIndex(0), VersionIndex(0), VersionIndex(0)]);
        let s2 = ClusterState::new(&[VersionIndex(1), VersionIndex(0), VersionIndex(2)]);
        assert_eq!(s1.hamming_distance(&s2), 2);
    }

    #[test]
    fn test_cluster_state_monotone() {
        let s1 = ClusterState::new(&[VersionIndex(0), VersionIndex(1)]);
        let s2 = ClusterState::new(&[VersionIndex(1), VersionIndex(1)]);
        let s3 = ClusterState::new(&[VersionIndex(0), VersionIndex(0)]);
        assert!(s2.is_monotone_from(&s1));
        assert!(!s3.is_monotone_from(&s1));
    }

    #[test]
    fn test_cluster_state_flat_index() {
        let counts = vec![3, 4];
        let state = ClusterState::new(&[VersionIndex(2), VersionIndex(3)]);
        let idx = state.to_flat_index(&counts);
        let recovered = ClusterState::from_flat_index(idx, &counts);
        assert_eq!(state, recovered);
    }

    #[test]
    fn test_cluster_state_id() {
        let s1 = ClusterState::new(&[VersionIndex(0), VersionIndex(1)]);
        let s2 = ClusterState::new(&[VersionIndex(0), VersionIndex(1)]);
        let s3 = ClusterState::new(&[VersionIndex(1), VersionIndex(0)]);
        assert_eq!(s1.state_id(), s2.state_id());
        assert_ne!(s1.state_id(), s3.state_id());
    }

    #[test]
    fn test_graph_edge() {
        let e = GraphEdge::new(0, VersionIndex(1), VersionIndex(3));
        assert!(e.is_upgrade());
        assert!(!e.is_downgrade());
        assert_eq!(e.step_size(), 2);

        let e2 = GraphEdge::new(0, VersionIndex(3), VersionIndex(1));
        assert!(!e2.is_upgrade());
        assert!(e2.is_downgrade());
        assert_eq!(e2.step_size(), -2);
    }

    #[test]
    fn test_version_product_graph_new() {
        let vsets = make_version_sets();
        let graph = VersionProductGraph::new(&vsets);
        assert_eq!(graph.num_services, 2);
        assert_eq!(graph.total_state_space_size(), 6); // 2 * 3
    }

    #[test]
    fn test_version_product_graph_explore() {
        let vsets = make_version_sets();
        let mut graph = VersionProductGraph::new(&vsets);
        let start = ClusterState::initial(2);

        let oracle = DefaultAdjacencyOracle::new(
            graph.version_counts.clone(),
            Box::new(TruePredicate),
        )
        .with_filter(Box::new(MonotoneFilter));

        graph.explore_bfs(&start, &oracle, 100);
        assert!(graph.explored_count() > 0);
        assert!(graph.is_safe(&start));
    }

    #[test]
    fn test_forward_reachability() {
        let vsets = make_version_sets();
        let mut graph = VersionProductGraph::new(&vsets);
        let start = ClusterState::initial(2);

        let oracle = DefaultAdjacencyOracle::new(
            graph.version_counts.clone(),
            Box::new(TruePredicate),
        )
        .with_filter(Box::new(MonotoneFilter));

        graph.explore_bfs(&start, &oracle, 100);
        let reachable = graph.forward_reachable(&start);
        assert!(reachable.contains(&start));
        // Target state should be reachable
        let target = ClusterState::new(&[VersionIndex(1), VersionIndex(2)]);
        assert!(reachable.contains(&target));
    }

    #[test]
    fn test_shortest_path() {
        let vsets = make_version_sets();
        let mut graph = VersionProductGraph::new(&vsets);
        let start = ClusterState::initial(2);
        let target = ClusterState::new(&[VersionIndex(1), VersionIndex(2)]);

        let oracle = DefaultAdjacencyOracle::new(
            graph.version_counts.clone(),
            Box::new(TruePredicate),
        )
        .with_filter(Box::new(MonotoneFilter));

        graph.explore_bfs(&start, &oracle, 100);
        let path = graph.shortest_path(&start, &target);
        assert!(path.is_some());
        let path = path.unwrap();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_shortest_path_same_state() {
        let vsets = make_version_sets();
        let graph = VersionProductGraph::new(&vsets);
        let start = ClusterState::initial(2);
        let path = graph.shortest_path(&start, &start);
        assert!(path.is_some());
        assert!(path.unwrap().is_empty());
    }

    #[test]
    fn test_monotone_filter() {
        let filter = MonotoneFilter;
        let s1 = ClusterState::new(&[VersionIndex(0)]);
        let s2 = ClusterState::new(&[VersionIndex(1)]);
        let up = GraphEdge::new(0, VersionIndex(0), VersionIndex(1));
        let down = GraphEdge::new(0, VersionIndex(1), VersionIndex(0));
        assert!(filter.is_allowed(&s1, &up, &s2));
        assert!(!filter.is_allowed(&s2, &down, &s1));
    }

    #[test]
    fn test_single_step_filter() {
        let filter = SingleStepFilter;
        let s = ClusterState::new(&[VersionIndex(0)]);
        let e1 = GraphEdge::new(0, VersionIndex(0), VersionIndex(1));
        let e2 = GraphEdge::new(0, VersionIndex(0), VersionIndex(2));
        assert!(filter.is_allowed(&s, &e1, &s));
        assert!(!filter.is_allowed(&s, &e2, &s));
    }

    #[test]
    fn test_composite_filter() {
        let filter = CompositeFilter::new()
            .add(Box::new(MonotoneFilter))
            .add(Box::new(SingleStepFilter));
        let s = ClusterState::new(&[VersionIndex(0)]);
        let e_ok = GraphEdge::new(0, VersionIndex(0), VersionIndex(1));
        let e_skip = GraphEdge::new(0, VersionIndex(0), VersionIndex(2));
        let e_down = GraphEdge::new(0, VersionIndex(1), VersionIndex(0));
        assert!(filter.is_allowed(&s, &e_ok, &s));
        assert!(!filter.is_allowed(&s, &e_skip, &s));
        assert!(!filter.is_allowed(&s, &e_down, &s));
    }

    #[test]
    fn test_treewidth_empty() {
        assert_eq!(estimate_treewidth(&[]), 0);
    }

    #[test]
    fn test_treewidth_path() {
        // Path: 0-1-2-3
        let adj = vec![vec![1], vec![0, 2], vec![1, 3], vec![2]];
        let tw = estimate_treewidth(&adj);
        assert_eq!(tw, 1);
    }

    #[test]
    fn test_treewidth_clique() {
        // K4 clique: treewidth = 3
        let adj = vec![
            vec![1, 2, 3],
            vec![0, 2, 3],
            vec![0, 1, 3],
            vec![0, 1, 2],
        ];
        let tw = estimate_treewidth(&adj);
        assert_eq!(tw, 3);
    }

    #[test]
    fn test_build_undirected_adjacency() {
        let adj = build_undirected_adjacency(3, &[(0, 1), (1, 2)]);
        assert!(adj[0].contains(&1));
        assert!(adj[1].contains(&0));
        assert!(adj[1].contains(&2));
        assert!(adj[2].contains(&1));
        assert!(!adj[0].contains(&2));
    }

    #[test]
    fn test_graph_stats() {
        let vsets = make_version_sets();
        let mut graph = VersionProductGraph::new(&vsets);
        let start = ClusterState::initial(2);

        let oracle = DefaultAdjacencyOracle::new(
            graph.version_counts.clone(),
            Box::new(TruePredicate),
        );
        graph.explore_bfs(&start, &oracle, 100);

        let stats = graph.stats();
        assert_eq!(stats.num_services, 2);
        assert!(stats.explored_states > 0);
        let s = stats.to_string();
        assert!(s.contains("Graph:"));
    }

    #[test]
    fn test_cluster_state_display() {
        let s = ClusterState::new(&[VersionIndex(0), VersionIndex(2)]);
        assert_eq!(s.to_string(), "(v0, v2)");
    }

    #[test]
    fn test_differing_services() {
        let s1 = ClusterState::new(&[VersionIndex(0), VersionIndex(1), VersionIndex(2)]);
        let s2 = ClusterState::new(&[VersionIndex(0), VersionIndex(3), VersionIndex(2)]);
        assert_eq!(s1.differing_services(&s2), vec![1]);
    }

    #[test]
    fn test_state_info() {
        let state = ClusterState::initial(2);
        let info = StateInfo::new(state.clone(), true);
        assert!(info.is_safe);
        assert!(!info.is_pnr);
    }

    #[test]
    fn test_exploration_config() {
        let cfg = ExplorationConfig::new()
            .with_max_states(500)
            .with_monotone(false);
        assert_eq!(cfg.max_states, 500);
        assert!(!cfg.monotone_only);
    }

    #[test]
    fn test_estimate_safe_state_count() {
        let counts = vec![3, 3];
        let count = estimate_safe_state_count(&counts, &TruePredicate, 100);
        // All states are safe with TruePredicate, so should be close to 9
        assert!((count - 9.0).abs() < 1.0);
    }

    #[test]
    fn test_backward_reachable() {
        let vsets = make_version_sets();
        let mut graph = VersionProductGraph::new(&vsets);
        let start = ClusterState::initial(2);

        let oracle = DefaultAdjacencyOracle::new(
            graph.version_counts.clone(),
            Box::new(TruePredicate),
        );
        graph.explore_bfs(&start, &oracle, 100);

        let target = ClusterState::new(&[VersionIndex(1), VersionIndex(2)]);
        let back = graph.backward_reachable(&target);
        // Start should be able to reach target, so start should be in backward reachable
        assert!(back.contains(&start));
    }
}
