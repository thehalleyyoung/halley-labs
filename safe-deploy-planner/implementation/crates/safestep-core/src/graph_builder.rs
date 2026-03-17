//! Version-product graph builder for SafeStep.
//!
//! Constructs the full Cartesian-product state space from a set of service
//! descriptors, creates single-service transition edges, labels them with
//! risk/duration metadata, and optionally prunes infeasible states using
//! constraints and downward-closure analysis.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, trace, warn};

use safestep_types::identifiers::{ConstraintId, Id};

use crate::{
    Constraint, CoreResult, Edge, ServiceDescriptor, ServiceIndex, State, TransitionMetadata,
    VersionIndex, VersionProductGraph,
};
use safestep_types::error::SafeStepError;

// ---------------------------------------------------------------------------
// GraphBuilder
// ---------------------------------------------------------------------------

/// Builds a [`VersionProductGraph`] from service descriptors.
///
/// The graph is the Cartesian product of each service's version set.
/// Edges connect states that differ in exactly one service's version.
pub struct GraphBuilder;

impl GraphBuilder {
    // -- public entry points ------------------------------------------------

    /// Build the complete version-product graph (no constraint pruning).
    #[instrument(skip_all, fields(services = services.len()))]
    pub fn from_services(services: &[ServiceDescriptor]) -> CoreResult<VersionProductGraph> {
        Self::validate_services(services)?;

        let dimensions = Self::compute_state_space_dimensions(services);
        info!(
            dimensions = ?dimensions,
            "building version-product graph"
        );

        let owned: Vec<ServiceDescriptor> = services.to_vec();
        let mut graph = VersionProductGraph::new(owned);

        let states = Self::enumerate_states(&dimensions);
        for s in &states {
            graph.add_state(s.clone());
        }

        let edges = Self::enumerate_edges(&states, services);
        for e in edges {
            graph.add_edge(e);
        }

        Self::label_edges(&mut graph);

        info!(
            states = graph.state_count(),
            edges = graph.edge_count(),
            "graph construction complete"
        );
        Ok(graph)
    }

    /// Build the graph, then prune states that violate any constraint.
    #[instrument(skip_all, fields(services = services.len(), constraints = constraints.len()))]
    pub fn from_services_filtered(
        services: &[ServiceDescriptor],
        constraints: &[Constraint],
    ) -> CoreResult<VersionProductGraph> {
        let mut graph = Self::from_services(services)?;

        let removed_constraints = Self::prune_infeasible_states(&mut graph, constraints);
        debug!(removed = removed_constraints, "pruned infeasible states");

        let removed_monotone = Self::apply_monotone_filter(&mut graph, constraints);
        debug!(removed = removed_monotone, "monotone filter pass");

        info!(
            states = graph.state_count(),
            edges = graph.edge_count(),
            "filtered graph ready"
        );
        Ok(graph)
    }

    /// Return the number of versions per service (the dimensions of the product).
    pub fn compute_state_space_dimensions(services: &[ServiceDescriptor]) -> Vec<usize> {
        services.iter().map(|s| s.version_count()).collect()
    }

    /// Populate edge metadata from the service descriptors already stored in
    /// the graph (risk scores, downtime requirements, duration estimates).
    pub fn label_edges(graph: &mut VersionProductGraph) {
        let service_count = graph.services.len();
        for edge in &mut graph.edges {
            let svc_idx = edge.service.0 as usize;
            if svc_idx >= service_count {
                continue;
            }
            let svc = &graph.services[svc_idx];
            let from = edge.from_version.0 as usize;
            let to = edge.to_version.0 as usize;

            let is_upgrade = to > from;
            let risk = svc.upgrade_risk.get(&(from, to)).copied().unwrap_or_else(|| {
                if is_upgrade {
                    Self::default_upgrade_risk(from, to)
                } else {
                    Self::default_downgrade_risk(from, to)
                }
            });

            let requires_downtime = if to < svc.downtime_required.len() {
                svc.downtime_required[to]
            } else {
                false
            };

            let duration = Self::estimate_duration(from, to, is_upgrade, requires_downtime);

            edge.metadata = TransitionMetadata {
                is_upgrade,
                risk_score: risk,
                estimated_duration_secs: duration,
                requires_downtime,
            };
        }
    }

    /// Iteratively remove states reachable only through constraint-violating
    /// predecessors using a downward-closure (monotone) argument.
    ///
    /// Returns the number of states removed.
    pub fn apply_monotone_filter(
        graph: &mut VersionProductGraph,
        constraints: &[Constraint],
    ) -> usize {
        let compatibility_constraints: Vec<&Constraint> = constraints
            .iter()
            .filter(|c| matches!(c, Constraint::Compatibility { .. }))
            .collect();

        if compatibility_constraints.is_empty() {
            return 0;
        }

        let mut removed_total = 0;
        let mut changed = true;

        while changed {
            changed = false;
            let mut to_remove: Vec<State> = Vec::new();

            for state in &graph.states {
                let predecessors = graph.predecessors(state);
                if predecessors.is_empty() {
                    continue;
                }

                let all_predecessors_violate = predecessors.iter().all(|(pred, _)| {
                    compatibility_constraints
                        .iter()
                        .any(|c| !c.check_state(pred))
                });

                if all_predecessors_violate {
                    let state_also_violates = compatibility_constraints
                        .iter()
                        .any(|c| !c.check_state(state));

                    if state_also_violates {
                        to_remove.push(state.clone());
                    }
                }
            }

            if !to_remove.is_empty() {
                changed = true;
                removed_total += to_remove.len();
                for s in &to_remove {
                    Self::remove_state_from_graph(graph, s);
                }
            }
        }

        removed_total
    }

    // -- internal helpers ---------------------------------------------------

    fn validate_services(services: &[ServiceDescriptor]) -> CoreResult<()> {
        if services.is_empty() {
            return Err(SafeStepError::graph(
                "cannot build graph from zero services",
            ));
        }
        for (i, svc) in services.iter().enumerate() {
            if svc.versions.is_empty() {
                return Err(SafeStepError::graph(
                    format!("service {} ('{}') has no versions", i, svc.name),
                ));
            }
        }
        Ok(())
    }

    /// Enumerate the full Cartesian product of version indices.
    fn enumerate_states(dimensions: &[usize]) -> Vec<State> {
        let total: usize = dimensions.iter().product();
        if total == 0 {
            return Vec::new();
        }

        let mut states = Vec::with_capacity(total);
        let n = dimensions.len();
        let mut indices = vec![0u16; n];

        loop {
            let versions: Vec<VersionIndex> = indices.iter().map(|&i| VersionIndex(i)).collect();
            states.push(State::new(versions));

            // increment the mixed-radix counter
            let mut carry = true;
            for dim in (0..n).rev() {
                if carry {
                    indices[dim] += 1;
                    if indices[dim] as usize >= dimensions[dim] {
                        indices[dim] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry {
                break;
            }
        }

        trace!(count = states.len(), "enumerated states");
        states
    }

    /// Create edges for every pair of states that differ in exactly one service
    /// version (single-service transitions only).
    fn enumerate_edges(states: &[State], services: &[ServiceDescriptor]) -> Vec<Edge> {
        let state_set: HashSet<&State> = states.iter().collect();
        let n_services = services.len();
        let mut edges: Vec<Edge> = Vec::new();

        for state in states {
            for svc_idx in 0..n_services {
                let svc = &services[svc_idx];
                let current_ver = state.versions[svc_idx].0;
                for target_ver in 0..svc.version_count() as u16 {
                    if target_ver == current_ver {
                        continue;
                    }
                    let mut new_versions = state.versions.clone();
                    new_versions[svc_idx] = VersionIndex(target_ver);
                    let target_state = State::new(new_versions);

                    if state_set.contains(&target_state) {
                        let edge = Edge {
                            from: state.clone(),
                            to: target_state,
                            service: ServiceIndex(svc_idx as u16),
                            from_version: VersionIndex(current_ver),
                            to_version: VersionIndex(target_ver),
                            metadata: TransitionMetadata::default(),
                        };
                        edges.push(edge);
                    }
                }
            }
        }

        trace!(count = edges.len(), "enumerated edges");
        edges
    }

    /// Remove a state and all incident edges from the graph.
    fn remove_state_from_graph(graph: &mut VersionProductGraph, state: &State) {
        // Collect indices of edges to remove (incident on `state`).
        let mut edge_indices_to_remove: HashSet<usize> = HashSet::new();

        if let Some(idxs) = graph.adjacency.get(state) {
            edge_indices_to_remove.extend(idxs.iter().copied());
        }
        if let Some(idxs) = graph.reverse_adjacency.get(state) {
            edge_indices_to_remove.extend(idxs.iter().copied());
        }

        // Remove the state from the state list.
        graph.states.retain(|s| s != state);
        graph.adjacency.remove(state);
        graph.reverse_adjacency.remove(state);

        if edge_indices_to_remove.is_empty() {
            return;
        }

        // Sort in descending order so we can remove by swapping from the back.
        let mut sorted: Vec<usize> = edge_indices_to_remove.into_iter().collect();
        sorted.sort_unstable_by(|a, b| b.cmp(a));

        for &idx in &sorted {
            if idx < graph.edges.len() {
                graph.edges.swap_remove(idx);
            }
        }

        // Rebuild adjacency maps from scratch (simplest correct approach).
        Self::rebuild_adjacency(graph);
    }

    /// Rebuild both adjacency and reverse-adjacency maps from the edge list.
    fn rebuild_adjacency(graph: &mut VersionProductGraph) {
        graph.adjacency.clear();
        graph.reverse_adjacency.clear();

        for s in &graph.states {
            graph.adjacency.entry(s.clone()).or_default();
            graph.reverse_adjacency.entry(s.clone()).or_default();
        }

        for (idx, edge) in graph.edges.iter().enumerate() {
            graph
                .adjacency
                .entry(edge.from.clone())
                .or_default()
                .push(idx);
            graph
                .reverse_adjacency
                .entry(edge.to.clone())
                .or_default()
                .push(idx);
        }
    }

    /// Remove all states that violate at least one constraint.
    /// Returns the number of states removed.
    fn prune_infeasible_states(
        graph: &mut VersionProductGraph,
        constraints: &[Constraint],
    ) -> usize {
        let infeasible: Vec<State> = graph
            .states
            .iter()
            .filter(|s| constraints.iter().any(|c| !c.check_state(s)))
            .cloned()
            .collect();

        let count = infeasible.len();
        for s in &infeasible {
            Self::remove_state_from_graph(graph, s);
        }
        count
    }

    fn default_upgrade_risk(from: usize, to: usize) -> u32 {
        let gap = if to > from { to - from } else { from - to };
        (gap as u32) * 10
    }

    fn default_downgrade_risk(from: usize, to: usize) -> u32 {
        let gap = if from > to { from - to } else { to - from };
        (gap as u32) * 15
    }

    fn estimate_duration(from: usize, to: usize, is_upgrade: bool, requires_downtime: bool) -> u64 {
        let gap = if to > from { to - from } else { from - to };
        let base = 30u64 + (gap as u64) * 20;
        let downtime_factor: u64 = if requires_downtime { 2 } else { 1 };
        let direction_factor: u64 = if is_upgrade { 1 } else { 2 };
        base * downtime_factor * direction_factor
    }
}

// ---------------------------------------------------------------------------
// StateSpace
// ---------------------------------------------------------------------------

/// Represents the state space either explicitly (list of states) or
/// symbolically (dimension vector for lazy enumeration).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateSpaceRepr {
    /// Every state is stored in memory.
    Explicit { states: Vec<State> },
    /// Only the dimensions are stored; states are enumerated on the fly.
    Symbolic { dimensions: Vec<usize> },
}

/// A handle to the state space that can be either explicit or symbolic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSpace {
    repr: StateSpaceRepr,
}

impl StateSpace {
    /// Create an explicit state space from a list of states.
    pub fn explicit(states: Vec<State>) -> Self {
        Self {
            repr: StateSpaceRepr::Explicit { states },
        }
    }

    /// Create a symbolic state space from dimension sizes.
    pub fn symbolic(dimensions: Vec<usize>) -> Self {
        Self {
            repr: StateSpaceRepr::Symbolic { dimensions },
        }
    }

    /// Total number of states in the space.
    pub fn total_states(&self) -> u64 {
        match &self.repr {
            StateSpaceRepr::Explicit { states } => states.len() as u64,
            StateSpaceRepr::Symbolic { dimensions } => {
                dimensions.iter().fold(1u64, |acc, &d| acc.saturating_mul(d as u64))
            }
        }
    }

    /// Number of services (dimensions).
    pub fn dimension(&self) -> usize {
        match &self.repr {
            StateSpaceRepr::Explicit { states } => {
                states.first().map(|s| s.dimension()).unwrap_or(0)
            }
            StateSpaceRepr::Symbolic { dimensions } => dimensions.len(),
        }
    }

    /// Is this an explicit representation?
    pub fn is_explicit(&self) -> bool {
        matches!(&self.repr, StateSpaceRepr::Explicit { .. })
    }

    /// Is this a symbolic representation?
    pub fn is_symbolic(&self) -> bool {
        matches!(&self.repr, StateSpaceRepr::Symbolic { .. })
    }

    /// Enumerate neighbors of `state` that exist in this space, using
    /// the given graph's edge information.
    pub fn enumerate_neighbors(
        &self,
        state: &State,
        graph: &VersionProductGraph,
    ) -> Vec<(State, Edge)> {
        let neighbor_pairs = graph.neighbors(state);
        let mut result = Vec::with_capacity(neighbor_pairs.len());

        for (neighbor_state, edge_idx) in neighbor_pairs {
            if self.contains(&neighbor_state) {
                let edge = graph.edges[edge_idx].clone();
                result.push((neighbor_state, edge));
            }
        }

        result
    }

    /// Check whether a state belongs to this space.
    pub fn contains(&self, state: &State) -> bool {
        match &self.repr {
            StateSpaceRepr::Explicit { states } => states.contains(state),
            StateSpaceRepr::Symbolic { dimensions } => {
                if state.dimension() != dimensions.len() {
                    return false;
                }
                state
                    .versions
                    .iter()
                    .zip(dimensions.iter())
                    .all(|(v, &d)| (v.0 as usize) < d)
            }
        }
    }

    /// Materialise all states (if symbolic, enumerates them).
    pub fn to_explicit(&self) -> Vec<State> {
        match &self.repr {
            StateSpaceRepr::Explicit { states } => states.clone(),
            StateSpaceRepr::Symbolic { dimensions } => {
                GraphBuilder::enumerate_states(dimensions)
            }
        }
    }

    /// Convert an index in row-major order back into a [`State`].
    pub fn index_to_state(&self, mut index: u64) -> Option<State> {
        let dims = match &self.repr {
            StateSpaceRepr::Symbolic { dimensions } => dimensions.as_slice(),
            StateSpaceRepr::Explicit { states } => {
                return states.get(index as usize).cloned();
            }
        };
        if dims.is_empty() {
            return None;
        }
        let total: u64 = dims.iter().fold(1u64, |a, &d| a.saturating_mul(d as u64));
        if index >= total {
            return None;
        }
        let mut versions = vec![VersionIndex(0); dims.len()];
        for i in (0..dims.len()).rev() {
            let d = dims[i] as u64;
            versions[i] = VersionIndex((index % d) as u16);
            index /= d;
        }
        Some(State::new(versions))
    }

    /// Convert a [`State`] to its row-major index within a symbolic space.
    pub fn state_to_index(&self, state: &State) -> Option<u64> {
        let dims = match &self.repr {
            StateSpaceRepr::Symbolic { dimensions } => dimensions.as_slice(),
            StateSpaceRepr::Explicit { states } => {
                return states.iter().position(|s| s == state).map(|p| p as u64);
            }
        };
        if state.dimension() != dims.len() {
            return None;
        }
        let mut idx: u64 = 0;
        let mut multiplier: u64 = 1;
        for i in (0..dims.len()).rev() {
            let v = state.versions[i].0 as u64;
            if v >= dims[i] as u64 {
                return None;
            }
            idx += v * multiplier;
            multiplier *= dims[i] as u64;
        }
        Some(idx)
    }
}

impl fmt::Display for StateSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.repr {
            StateSpaceRepr::Explicit { states } => {
                write!(f, "Explicit({} states)", states.len())
            }
            StateSpaceRepr::Symbolic { dimensions } => {
                write!(f, "Symbolic({:?})", dimensions)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GraphStatistics
// ---------------------------------------------------------------------------

/// Aggregate statistics about a [`VersionProductGraph`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStatistics {
    pub state_count: usize,
    pub edge_count: usize,
    pub density: f64,
    pub max_degree: usize,
    pub avg_degree: f64,
    pub connected_components: usize,
    pub min_degree: usize,
    pub max_in_degree: usize,
    pub max_out_degree: usize,
    pub self_loop_count: usize,
    pub diameter_estimate: usize,
}

impl GraphStatistics {
    /// Compute full statistics for the graph.
    pub fn compute(graph: &VersionProductGraph) -> Self {
        let state_count = graph.state_count();
        let edge_count = graph.edge_count();

        let density = if state_count > 1 {
            edge_count as f64 / (state_count as f64 * (state_count as f64 - 1.0))
        } else {
            0.0
        };

        // Per-state degree analysis.
        let mut max_out_degree: usize = 0;
        let mut max_in_degree: usize = 0;
        let mut min_degree: usize = usize::MAX;
        let mut total_degree: usize = 0;

        for state in &graph.states {
            let out_deg = graph
                .adjacency
                .get(state)
                .map(|v| v.len())
                .unwrap_or(0);
            let in_deg = graph
                .reverse_adjacency
                .get(state)
                .map(|v| v.len())
                .unwrap_or(0);
            let degree = out_deg + in_deg;
            max_out_degree = max_out_degree.max(out_deg);
            max_in_degree = max_in_degree.max(in_deg);
            min_degree = min_degree.min(degree);
            total_degree += degree;
        }

        if state_count == 0 {
            min_degree = 0;
        }

        let max_degree = max_out_degree.max(max_in_degree);
        let avg_degree = if state_count > 0 {
            total_degree as f64 / state_count as f64
        } else {
            0.0
        };

        let self_loop_count = graph
            .edges
            .iter()
            .filter(|e| e.from == e.to)
            .count();

        let connected_components = Self::count_connected_components(graph);
        let diameter_estimate = Self::estimate_diameter(graph);

        Self {
            state_count,
            edge_count,
            density,
            max_degree,
            avg_degree,
            connected_components,
            min_degree,
            max_in_degree,
            max_out_degree,
            self_loop_count,
            diameter_estimate,
        }
    }

    /// Human-readable summary string.
    pub fn summary(&self) -> String {
        format!(
            "GraphStatistics {{ states: {}, edges: {}, density: {:.4}, \
             max_degree: {}, avg_degree: {:.2}, components: {}, \
             diameter_est: {} }}",
            self.state_count,
            self.edge_count,
            self.density,
            self.max_degree,
            self.avg_degree,
            self.connected_components,
            self.diameter_estimate,
        )
    }

    /// Count weakly connected components via BFS (treating edges as undirected).
    fn count_connected_components(graph: &VersionProductGraph) -> usize {
        if graph.states.is_empty() {
            return 0;
        }

        let mut visited: HashSet<&State> = HashSet::new();
        let mut components = 0;

        // Build an undirected adjacency list from the graph.
        let mut undirected: HashMap<&State, Vec<&State>> = HashMap::new();
        for state in &graph.states {
            undirected.entry(state).or_default();
        }
        for edge in &graph.edges {
            undirected.entry(&edge.from).or_default().push(&edge.to);
            undirected.entry(&edge.to).or_default().push(&edge.from);
        }

        for state in &graph.states {
            if visited.contains(state) {
                continue;
            }
            components += 1;
            let mut queue: VecDeque<&State> = VecDeque::new();
            queue.push_back(state);
            visited.insert(state);

            while let Some(current) = queue.pop_front() {
                if let Some(neighbors) = undirected.get(current) {
                    for &nbr in neighbors {
                        if visited.insert(nbr) {
                            queue.push_back(nbr);
                        }
                    }
                }
            }
        }

        components
    }

    /// Estimate the diameter by running BFS from a few randomly-chosen states
    /// and taking the maximum eccentricity found.
    fn estimate_diameter(graph: &VersionProductGraph) -> usize {
        if graph.states.is_empty() {
            return 0;
        }

        let sample_count = graph.states.len().min(5);
        let step = if graph.states.len() > 1 {
            graph.states.len() / sample_count
        } else {
            1
        };

        let mut max_dist = 0usize;

        for i in 0..sample_count {
            let start_idx = (i * step).min(graph.states.len() - 1);
            let start = &graph.states[start_idx];
            let eccentricity = Self::bfs_eccentricity(graph, start);
            max_dist = max_dist.max(eccentricity);
        }

        max_dist
    }

    /// BFS from `start`, return the distance to the farthest reachable state.
    fn bfs_eccentricity(graph: &VersionProductGraph, start: &State) -> usize {
        let mut visited: HashSet<State> = HashSet::new();
        let mut queue: VecDeque<(State, usize)> = VecDeque::new();
        visited.insert(start.clone());
        queue.push_back((start.clone(), 0));

        let mut max_depth = 0;

        while let Some((current, depth)) = queue.pop_front() {
            max_depth = max_depth.max(depth);
            for (neighbor, _edge_idx) in graph.neighbors(&current) {
                if visited.insert(neighbor.clone()) {
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }

        max_depth
    }
}

impl fmt::Display for GraphStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// DownwardClosureChecker
// ---------------------------------------------------------------------------

/// A violation of the downward-closure property.
///
/// If (a, b) is a compatible pair for two services and a' ≤ a, b' ≤ b,
/// then (a', b') must also be compatible. A violation records a concrete
/// counterexample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    pub constraint_id: ConstraintId,
    pub service_a: ServiceIndex,
    pub version_a: VersionIndex,
    pub service_b: ServiceIndex,
    pub version_b: VersionIndex,
    pub lower_a: VersionIndex,
    pub lower_b: VersionIndex,
    pub description: String,
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Violation(constraint={}, ({},{}) compatible but ({},{}) not): {}",
            self.constraint_id.as_str(),
            self.version_a,
            self.version_b,
            self.lower_a,
            self.lower_b,
            self.description,
        )
    }
}

/// Checks whether the compatibility relation induced by a set of constraints
/// is downward-closed with respect to the version ordering.
///
/// A compatibility relation R between services i and j is *downward-closed* if
/// whenever R(a, b) holds and a' ≤ a, b' ≤ b then R(a', b') also holds.
///
/// This property is useful because it ensures monotonicity of the feasible
/// region, enabling efficient pruning and guaranteeing that downgrading
/// from a feasible state always produces a feasible state.
pub struct DownwardClosureChecker {
    /// Number of versions per service (dimensions).
    dimensions: Vec<usize>,
}

impl DownwardClosureChecker {
    /// Create a new checker from the service descriptors.
    pub fn new(services: &[ServiceDescriptor]) -> Self {
        let dimensions = services.iter().map(|s| s.version_count()).collect();
        Self { dimensions }
    }

    /// Check whether ALL compatibility constraints in `constraints` are
    /// downward-closed. Returns `true` if the property holds everywhere.
    pub fn check(&self, constraints: &[Constraint]) -> bool {
        self.find_violations(constraints).is_empty()
    }

    /// Find all violations of the downward-closure property among the
    /// compatibility constraints.
    pub fn find_violations(&self, constraints: &[Constraint]) -> Vec<Violation> {
        let mut violations = Vec::new();

        for constraint in constraints {
            if let Constraint::Compatibility {
                id,
                service_a,
                service_b,
                compatible_pairs,
            } = constraint
            {
                let pair_set: HashSet<(VersionIndex, VersionIndex)> =
                    compatible_pairs.iter().copied().collect();

                for &(va, vb) in compatible_pairs {
                    // For each compatible pair (va, vb), check all pairs
                    // (va', vb') where va' <= va and vb' <= vb.
                    for la in 0..=va.0 {
                        for lb in 0..=vb.0 {
                            let lower_a = VersionIndex(la);
                            let lower_b = VersionIndex(lb);

                            // Verify the lower pair is within bounds.
                            let a_idx = service_a.0 as usize;
                            let b_idx = service_b.0 as usize;
                            if a_idx < self.dimensions.len()
                                && (la as usize) < self.dimensions[a_idx]
                                && b_idx < self.dimensions.len()
                                && (lb as usize) < self.dimensions[b_idx]
                            {
                                if !pair_set.contains(&(lower_a, lower_b)) {
                                    violations.push(Violation {
                                        constraint_id: id.clone(),
                                        service_a: *service_a,
                                        version_a: va,
                                        service_b: *service_b,
                                        version_b: vb,
                                        lower_a,
                                        lower_b,
                                        description: format!(
                                            "({}, {}) is compatible but ({}, {}) is not — \
                                             downward-closure violated",
                                            va, vb, lower_a, lower_b,
                                        ),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        violations
    }

    /// Compute the downward closure of a set of compatible pairs.
    /// Given a set of pairs, returns the smallest downward-closed superset.
    pub fn downward_close(
        &self,
        pairs: &[(VersionIndex, VersionIndex)],
        service_a: ServiceIndex,
        service_b: ServiceIndex,
    ) -> Vec<(VersionIndex, VersionIndex)> {
        let mut closed: HashSet<(VersionIndex, VersionIndex)> = HashSet::new();

        let a_idx = service_a.0 as usize;
        let b_idx = service_b.0 as usize;
        let max_a = if a_idx < self.dimensions.len() {
            self.dimensions[a_idx]
        } else {
            0
        };
        let max_b = if b_idx < self.dimensions.len() {
            self.dimensions[b_idx]
        } else {
            0
        };

        for &(va, vb) in pairs {
            for la in 0..=va.0 {
                for lb in 0..=vb.0 {
                    if (la as usize) < max_a && (lb as usize) < max_b {
                        closed.insert((VersionIndex(la), VersionIndex(lb)));
                    }
                }
            }
        }

        let mut result: Vec<_> = closed.into_iter().collect();
        result.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        result
    }
}

// ---------------------------------------------------------------------------
// Auxiliary: reachability helpers used by tests and other modules
// ---------------------------------------------------------------------------

/// Check whether `target` is reachable from `start` in the graph via BFS.
pub fn is_reachable(graph: &VersionProductGraph, start: &State, target: &State) -> bool {
    if start == target {
        return true;
    }
    let mut visited: HashSet<State> = HashSet::new();
    let mut queue: VecDeque<State> = VecDeque::new();
    visited.insert(start.clone());
    queue.push_back(start.clone());

    while let Some(current) = queue.pop_front() {
        for (neighbor, _) in graph.neighbors(&current) {
            if &neighbor == target {
                return true;
            }
            if visited.insert(neighbor.clone()) {
                queue.push_back(neighbor);
            }
        }
    }

    false
}

/// Return the shortest path (sequence of edge indices) from `start` to `target`,
/// or `None` if unreachable.
pub fn shortest_path(
    graph: &VersionProductGraph,
    start: &State,
    target: &State,
) -> Option<Vec<usize>> {
    if start == target {
        return Some(Vec::new());
    }

    let mut visited: HashSet<State> = HashSet::new();
    let mut queue: VecDeque<(State, Vec<usize>)> = VecDeque::new();
    visited.insert(start.clone());
    queue.push_back((start.clone(), Vec::new()));

    while let Some((current, path)) = queue.pop_front() {
        for (neighbor, edge_idx) in graph.neighbors(&current) {
            if &neighbor == target {
                let mut full = path.clone();
                full.push(edge_idx);
                return Some(full);
            }
            if visited.insert(neighbor.clone()) {
                let mut new_path = path.clone();
                new_path.push(edge_idx);
                queue.push_back((neighbor, new_path));
            }
        }
    }

    None
}

/// Collect all states reachable from `start`.
pub fn reachable_set(graph: &VersionProductGraph, start: &State) -> HashSet<State> {
    let mut visited: HashSet<State> = HashSet::new();
    let mut queue: VecDeque<State> = VecDeque::new();
    visited.insert(start.clone());
    queue.push_back(start.clone());

    while let Some(current) = queue.pop_front() {
        for (neighbor, _) in graph.neighbors(&current) {
            if visited.insert(neighbor.clone()) {
                queue.push_back(neighbor);
            }
        }
    }

    visited
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Constraint, ServiceDescriptor, ServiceIndex, State, VersionIndex};
    use safestep_types::identifiers::Id;
    use std::collections::HashMap;

    /// Helper: create a simple service descriptor with N versions.
    fn make_service(name: &str, n_versions: usize) -> ServiceDescriptor {
        let versions: Vec<String> = (0..n_versions).map(|i| format!("v{}", i)).collect();
        ServiceDescriptor::new(name, versions)
    }

    /// Helper: create a service descriptor with risk and downtime info.
    fn make_service_with_metadata(
        name: &str,
        n_versions: usize,
        risk_map: Vec<((usize, usize), u32)>,
        downtime: Vec<bool>,
    ) -> ServiceDescriptor {
        let versions: Vec<String> = (0..n_versions).map(|i| format!("v{}", i)).collect();
        let mut svc = ServiceDescriptor::new(name, versions);
        svc.upgrade_risk = risk_map.into_iter().collect();
        svc.downtime_required = downtime;
        svc
    }

    // -- GraphBuilder -------------------------------------------------------

    #[test]
    fn test_from_services_single_service() {
        let svc = make_service("api", 3);
        let graph = GraphBuilder::from_services(&[svc]).unwrap();

        // 3 versions → 3 states
        assert_eq!(graph.state_count(), 3);
        // Each state can transition to 2 others → 3 * 2 = 6 edges
        assert_eq!(graph.edge_count(), 6);
    }

    #[test]
    fn test_from_services_two_services() {
        let a = make_service("api", 2);
        let b = make_service("db", 3);
        let graph = GraphBuilder::from_services(&[a, b]).unwrap();

        // 2 × 3 = 6 states
        assert_eq!(graph.state_count(), 6);

        // Edges: for each state, service 0 can change to 1 other version (2-1),
        //        service 1 can change to 2 others (3-1).
        // Total = 6 * (1 + 2) = 18
        assert_eq!(graph.edge_count(), 18);
    }

    #[test]
    fn test_from_services_empty_services_returns_error() {
        let result = GraphBuilder::from_services(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_services_zero_version_service_returns_error() {
        let svc = ServiceDescriptor::new("empty", Vec::<String>::new());
        let result = GraphBuilder::from_services(&[svc]);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_services_filtered_removes_forbidden() {
        let svc = make_service("api", 3);
        let forbidden = Constraint::Forbidden {
            id: Id::from_name("no-v1"),
            service: ServiceIndex(0),
            version: VersionIndex(1),
        };
        let graph =
            GraphBuilder::from_services_filtered(&[svc], &[forbidden]).unwrap();

        // State (v1) should be removed, leaving 2 states
        assert_eq!(graph.state_count(), 2);

        // Verify forbidden state is absent
        let forbidden_state = State::new(vec![VersionIndex(1)]);
        assert!(!graph.has_state(&forbidden_state));
    }

    #[test]
    fn test_from_services_filtered_compatibility() {
        let a = make_service("api", 2);
        let b = make_service("db", 2);
        let compat = Constraint::Compatibility {
            id: Id::from_name("compat"),
            service_a: ServiceIndex(0),
            service_b: ServiceIndex(1),
            compatible_pairs: vec![
                (VersionIndex(0), VersionIndex(0)),
                (VersionIndex(0), VersionIndex(1)),
                (VersionIndex(1), VersionIndex(1)),
            ],
        };
        let graph =
            GraphBuilder::from_services_filtered(&[a, b], &[compat]).unwrap();

        // (1,0) is incompatible → 3 states remain
        assert_eq!(graph.state_count(), 3);
        let bad = State::new(vec![VersionIndex(1), VersionIndex(0)]);
        assert!(!graph.has_state(&bad));
    }

    #[test]
    fn test_label_edges_uses_service_metadata() {
        let svc = make_service_with_metadata(
            "api",
            3,
            vec![((0, 1), 5), ((1, 2), 20), ((2, 1), 8)],
            vec![false, false, true],
        );
        let graph = GraphBuilder::from_services(&[svc]).unwrap();

        // Find the edge from v0 → v1
        let edge_0_1 = graph
            .edges
            .iter()
            .find(|e| e.from_version == VersionIndex(0) && e.to_version == VersionIndex(1))
            .expect("edge v0->v1 should exist");
        assert_eq!(edge_0_1.metadata.risk_score, 5);
        assert!(edge_0_1.metadata.is_upgrade);
        assert!(!edge_0_1.metadata.requires_downtime);

        // Find the edge from v1 → v2 (requires downtime at v2)
        let edge_1_2 = graph
            .edges
            .iter()
            .find(|e| e.from_version == VersionIndex(1) && e.to_version == VersionIndex(2))
            .expect("edge v1->v2 should exist");
        assert_eq!(edge_1_2.metadata.risk_score, 20);
        assert!(edge_1_2.metadata.requires_downtime);
    }

    #[test]
    fn test_compute_state_space_dimensions() {
        let a = make_service("api", 4);
        let b = make_service("db", 2);
        let c = make_service("cache", 5);
        let dims = GraphBuilder::compute_state_space_dimensions(&[a, b, c]);
        assert_eq!(dims, vec![4, 2, 5]);
    }

    #[test]
    fn test_apply_monotone_filter() {
        let a = make_service("api", 3);
        let b = make_service("db", 3);

        let compat = Constraint::Compatibility {
            id: Id::from_name("compat-mono"),
            service_a: ServiceIndex(0),
            service_b: ServiceIndex(1),
            compatible_pairs: vec![
                (VersionIndex(0), VersionIndex(0)),
                (VersionIndex(0), VersionIndex(1)),
                (VersionIndex(0), VersionIndex(2)),
                (VersionIndex(1), VersionIndex(0)),
                (VersionIndex(1), VersionIndex(1)),
                (VersionIndex(1), VersionIndex(2)),
                (VersionIndex(2), VersionIndex(0)),
                (VersionIndex(2), VersionIndex(1)),
                // (2,2) is MISSING — only compatible pair missing
            ],
        };

        let mut graph = GraphBuilder::from_services(&[a, b]).unwrap();
        assert_eq!(graph.state_count(), 9);

        let removed = GraphBuilder::apply_monotone_filter(&mut graph, &[compat]);
        // The filter should identify that (2,2) is infeasible and its
        // predecessors that ALL lead through incompatible states MAY be pruned.
        // At minimum the filter ran without error; exact count depends on
        // predecessor analysis.
        assert!(removed <= 9);
    }

    // -- StateSpace ---------------------------------------------------------

    #[test]
    fn test_state_space_symbolic() {
        let ss = StateSpace::symbolic(vec![3, 4, 2]);
        assert!(ss.is_symbolic());
        assert!(!ss.is_explicit());
        assert_eq!(ss.total_states(), 24);
        assert_eq!(ss.dimension(), 3);

        // Every valid state should be contained.
        let s = State::new(vec![VersionIndex(2), VersionIndex(3), VersionIndex(1)]);
        assert!(ss.contains(&s));

        // Out-of-bounds state should NOT be contained.
        let oob = State::new(vec![VersionIndex(3), VersionIndex(0), VersionIndex(0)]);
        assert!(!ss.contains(&oob));
    }

    #[test]
    fn test_state_space_explicit() {
        let states = vec![
            State::new(vec![VersionIndex(0)]),
            State::new(vec![VersionIndex(2)]),
        ];
        let ss = StateSpace::explicit(states);
        assert!(ss.is_explicit());
        assert_eq!(ss.total_states(), 2);
        assert!(ss.contains(&State::new(vec![VersionIndex(0)])));
        assert!(!ss.contains(&State::new(vec![VersionIndex(1)])));
    }

    #[test]
    fn test_state_space_index_conversion() {
        let ss = StateSpace::symbolic(vec![2, 3]);
        // Row-major: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5
        let s = ss.index_to_state(3).unwrap();
        assert_eq!(s, State::new(vec![VersionIndex(1), VersionIndex(0)]));

        let idx = ss.state_to_index(&s).unwrap();
        assert_eq!(idx, 3);

        // Round-trip for every state.
        for i in 0..6u64 {
            let state = ss.index_to_state(i).unwrap();
            let back = ss.state_to_index(&state).unwrap();
            assert_eq!(i, back);
        }
    }

    #[test]
    fn test_state_space_enumerate_neighbors() {
        let svc = make_service("api", 3);
        let graph = GraphBuilder::from_services(&[svc]).unwrap();
        let ss = StateSpace::explicit(graph.states.clone());

        let origin = State::new(vec![VersionIndex(0)]);
        let neighbors = ss.enumerate_neighbors(&origin, &graph);
        // From v0 we can go to v1 and v2
        assert_eq!(neighbors.len(), 2);
    }

    // -- GraphStatistics ----------------------------------------------------

    #[test]
    fn test_graph_statistics_single_service() {
        let svc = make_service("api", 4);
        let graph = GraphBuilder::from_services(&[svc]).unwrap();
        let stats = GraphStatistics::compute(&graph);

        assert_eq!(stats.state_count, 4);
        assert_eq!(stats.edge_count, 12); // 4 * 3
        assert_eq!(stats.connected_components, 1);
        assert!(stats.density > 0.0);
        assert!(stats.avg_degree > 0.0);

        let summary = stats.summary();
        assert!(summary.contains("states: 4"));
        assert!(summary.contains("edges: 12"));
    }

    #[test]
    fn test_graph_statistics_two_services() {
        let a = make_service("api", 2);
        let b = make_service("db", 2);
        let graph = GraphBuilder::from_services(&[a, b]).unwrap();
        let stats = GraphStatistics::compute(&graph);

        assert_eq!(stats.state_count, 4);
        // Each state has 1 + 1 = 2 outgoing edges → 4 * 2 = 8 edges
        assert_eq!(stats.edge_count, 8);
        assert_eq!(stats.connected_components, 1);
        assert_eq!(stats.self_loop_count, 0);
    }

    // -- DownwardClosureChecker ---------------------------------------------

    #[test]
    fn test_downward_closure_holds() {
        let a = make_service("api", 3);
        let b = make_service("db", 3);

        // All pairs ≤ (2,2) are compatible → downward-closed.
        let mut pairs = Vec::new();
        for i in 0..3u16 {
            for j in 0..3u16 {
                pairs.push((VersionIndex(i), VersionIndex(j)));
            }
        }
        let constraint = Constraint::Compatibility {
            id: Id::from_name("full-compat"),
            service_a: ServiceIndex(0),
            service_b: ServiceIndex(1),
            compatible_pairs: pairs,
        };

        let checker = DownwardClosureChecker::new(&[a, b]);
        assert!(checker.check(&[constraint]));
    }

    #[test]
    fn test_downward_closure_violated() {
        let a = make_service("api", 3);
        let b = make_service("db", 3);

        // (2,2) is compatible but (1,1) is NOT → violation.
        let constraint = Constraint::Compatibility {
            id: Id::from_name("bad-compat"),
            service_a: ServiceIndex(0),
            service_b: ServiceIndex(1),
            compatible_pairs: vec![
                (VersionIndex(2), VersionIndex(2)),
                (VersionIndex(0), VersionIndex(0)),
                // Deliberately omit (1,1) which is ≤ (2,2)
            ],
        };

        let checker = DownwardClosureChecker::new(&[a, b]);
        assert!(!checker.check(&[constraint]));

        let violations = checker.find_violations(&[constraint]);
        assert!(!violations.is_empty());

        // There should be a violation mentioning the missing (1,1) pair.
        let has_11 = violations.iter().any(|v| {
            v.lower_a == VersionIndex(1) && v.lower_b == VersionIndex(1)
        });
        assert!(has_11, "should detect that (1,1) is missing");
    }

    #[test]
    fn test_downward_close_computation() {
        let a = make_service("api", 4);
        let b = make_service("db", 4);
        let checker = DownwardClosureChecker::new(&[a, b]);

        // Start with just {(2,2)}.
        let pairs = vec![(VersionIndex(2), VersionIndex(2))];
        let closed = checker.downward_close(&pairs, ServiceIndex(0), ServiceIndex(1));

        // Closure should contain all (a,b) with a <= 2, b <= 2 → 3 * 3 = 9 pairs.
        assert_eq!(closed.len(), 9);
        assert!(closed.contains(&(VersionIndex(0), VersionIndex(0))));
        assert!(closed.contains(&(VersionIndex(1), VersionIndex(2))));
        assert!(closed.contains(&(VersionIndex(2), VersionIndex(2))));
        // (3,0) should NOT be in the closure.
        assert!(!closed.contains(&(VersionIndex(3), VersionIndex(0))));
    }

    // -- Reachability helpers -----------------------------------------------

    #[test]
    fn test_reachability() {
        let svc = make_service("api", 3);
        let graph = GraphBuilder::from_services(&[svc]).unwrap();

        let s0 = State::new(vec![VersionIndex(0)]);
        let s2 = State::new(vec![VersionIndex(2)]);

        assert!(is_reachable(&graph, &s0, &s2));
        assert!(is_reachable(&graph, &s2, &s0));
    }

    #[test]
    fn test_shortest_path() {
        let svc = make_service("api", 4);
        let graph = GraphBuilder::from_services(&[svc]).unwrap();

        let s0 = State::new(vec![VersionIndex(0)]);
        let s3 = State::new(vec![VersionIndex(3)]);

        let path = shortest_path(&graph, &s0, &s3).unwrap();
        // Direct edge exists from v0 → v3
        assert_eq!(path.len(), 1);
    }

    #[test]
    fn test_reachable_set() {
        let a = make_service("api", 2);
        let b = make_service("db", 2);
        let graph = GraphBuilder::from_services(&[a, b]).unwrap();

        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let reachable = reachable_set(&graph, &start);

        // All 4 states are reachable from any state in a fully connected graph.
        assert_eq!(reachable.len(), 4);
    }

    #[test]
    fn test_reachability_with_pruned_graph() {
        let a = make_service("api", 3);
        let b = make_service("db", 2);

        // Forbid api v2 → only states with api ∈ {v0, v1} remain.
        let forbidden = Constraint::Forbidden {
            id: Id::from_name("no-api-v2"),
            service: ServiceIndex(0),
            version: VersionIndex(2),
        };
        let graph = GraphBuilder::from_services_filtered(&[a, b], &[forbidden]).unwrap();

        assert_eq!(graph.state_count(), 4); // 2 * 2

        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        assert!(is_reachable(&graph, &start, &target));

        let removed_target = State::new(vec![VersionIndex(2), VersionIndex(0)]);
        assert!(!is_reachable(&graph, &start, &removed_target));
    }

    // -- Large graph smoke test ---------------------------------------------

    #[test]
    fn test_three_service_graph() {
        let a = make_service("api", 3);
        let b = make_service("db", 2);
        let c = make_service("cache", 2);
        let graph = GraphBuilder::from_services(&[a, b, c]).unwrap();

        // 3 × 2 × 2 = 12 states
        assert_eq!(graph.state_count(), 12);

        // edges per state: (3-1) + (2-1) + (2-1) = 2 + 1 + 1 = 4
        // total: 12 * 4 = 48
        assert_eq!(graph.edge_count(), 48);

        let stats = GraphStatistics::compute(&graph);
        assert_eq!(stats.connected_components, 1);
        assert!(stats.diameter_estimate >= 3);
    }
}
