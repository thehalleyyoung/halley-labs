//! Product PTA construction.
//!
//! Builds the synchronised parallel composition of multiple individual-drug
//! PTAs.  Edges that reference shared variables (enzyme activities) are
//! synchronised; independent edges are interleaved.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    ActionLabel, AtomicPredicate, Clock, ClockId, Edge, EdgeId, Location,
    LocationId, LocationKind, ModelCheckerError, PTA, Predicate, Result,
    Update, Variable, VariableId, VariableKind,
};

// ---------------------------------------------------------------------------
// ProductLocation
// ---------------------------------------------------------------------------

/// A location in the product PTA is a tuple of individual-PTA locations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProductLocation {
    /// Component location IDs, one per input PTA (indexed in order).
    pub components: Vec<LocationId>,
    /// Unique ID assigned in the product.
    pub product_id: LocationId,
    /// Derived name (e.g. "(idle, absorbing)").
    pub name: String,
    /// Derived kind – Error if *any* component is Error, Initial if *all*
    /// components are Initial, otherwise Normal.
    pub kind: LocationKind,
}

impl fmt::Display for ProductLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

// ---------------------------------------------------------------------------
// ProductEdge
// ---------------------------------------------------------------------------

/// An edge in the product PTA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductEdge {
    pub id: EdgeId,
    pub source: LocationId,
    pub target: LocationId,
    pub guard: Predicate,
    pub action: ActionLabel,
    pub updates: Vec<Update>,
    /// Which component PTA(s) participate in this edge.
    pub participants: Vec<usize>,
    /// Original edge ids (one per participant).
    pub origin_edges: Vec<EdgeId>,
}

// ---------------------------------------------------------------------------
// ProductPTA
// ---------------------------------------------------------------------------

/// The product (parallel composition) of several PTAs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductPTA {
    pub locations: Vec<ProductLocation>,
    pub edges: Vec<ProductEdge>,
    pub clocks: Vec<Clock>,
    pub variables: Vec<Variable>,
    pub initial_location: LocationId,
    pub initial_variable_values: Vec<f64>,
    /// Names of the component PTAs.
    pub component_names: Vec<String>,
    /// Shared variable IDs (enzyme activities).
    pub shared_variables: HashSet<VariableId>,
    /// Mapping from product location id → component location tuple.
    pub location_map: HashMap<Vec<LocationId>, LocationId>,
}

impl ProductPTA {
    pub fn num_locations(&self) -> usize {
        self.locations.len()
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn num_components(&self) -> usize {
        self.component_names.len()
    }

    /// Whether any product location is an error location.
    pub fn has_error_locations(&self) -> bool {
        self.locations.iter().any(|l| l.kind == LocationKind::Error)
    }

    /// Get all error locations.
    pub fn error_locations(&self) -> Vec<&ProductLocation> {
        self.locations.iter().filter(|l| l.kind == LocationKind::Error).collect()
    }

    /// Outgoing edges from a product location.
    pub fn outgoing_edges(&self, loc: LocationId) -> Vec<&ProductEdge> {
        self.edges.iter().filter(|e| e.source == loc).collect()
    }

    /// Get a product location by its component tuple.
    pub fn get_product_location(&self, components: &[LocationId]) -> Option<&ProductLocation> {
        self.location_map
            .get(components)
            .and_then(|pid| self.locations.get(*pid))
    }
}

// ---------------------------------------------------------------------------
// SharedVariableAnalysis
// ---------------------------------------------------------------------------

/// Analysis of which variables are shared between PTAs.
#[derive(Debug, Clone)]
pub struct SharedVariableAnalysis {
    /// For each variable name, the set of PTA indices that reference it.
    pub variable_usage: HashMap<String, BTreeSet<usize>>,
    /// Variable names shared by ≥2 PTAs.
    pub shared_names: HashSet<String>,
    /// Variable names private to a single PTA.
    pub private_names: HashMap<String, usize>,
}

impl SharedVariableAnalysis {
    /// Analyse a set of PTAs to determine variable sharing.
    pub fn analyse(ptas: &[PTA]) -> Self {
        let mut variable_usage: HashMap<String, BTreeSet<usize>> = HashMap::new();

        for (idx, pta) in ptas.iter().enumerate() {
            for var in &pta.variables {
                variable_usage
                    .entry(var.name.clone())
                    .or_default()
                    .insert(idx);
            }
        }

        let mut shared_names = HashSet::new();
        let mut private_names = HashMap::new();

        for (name, users) in &variable_usage {
            if users.len() >= 2 {
                shared_names.insert(name.clone());
            } else if let Some(&single) = users.iter().next() {
                private_names.insert(name.clone(), single);
            }
        }

        Self { variable_usage, shared_names, private_names }
    }

    /// Number of shared variables.
    pub fn num_shared(&self) -> usize {
        self.shared_names.len()
    }

    /// Number of private variables.
    pub fn num_private(&self) -> usize {
        self.private_names.len()
    }

    /// Whether two PTAs share any variables.
    pub fn share_variables(&self, a: usize, b: usize) -> bool {
        for users in self.variable_usage.values() {
            if users.contains(&a) && users.contains(&b) {
                return true;
            }
        }
        false
    }

    /// Get the set of enzymes that are shared.
    pub fn shared_enzyme_names(&self) -> Vec<&str> {
        self.shared_names.iter().map(|s| s.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// ProductBuilder
// ---------------------------------------------------------------------------

/// Builds the full product PTA from a set of component PTAs.
#[derive(Debug)]
pub struct ProductBuilder {
    max_states: usize,
}

impl ProductBuilder {
    pub fn new(max_states: usize) -> Self {
        Self { max_states }
    }

    /// Build the synchronised product of the given PTAs.
    pub fn build_product(&self, ptas: &[PTA]) -> Result<ProductPTA> {
        if ptas.is_empty() {
            return Err(ModelCheckerError::PtaConstruction(
                "cannot build product of zero PTAs".into(),
            ));
        }

        if ptas.len() == 1 {
            return self.singleton_product(&ptas[0]);
        }

        let estimated = compute_product_size(ptas);
        if estimated > self.max_states {
            return Err(ModelCheckerError::StateSpaceExplosion {
                num_states: estimated,
                limit: self.max_states,
            });
        }

        let analysis = SharedVariableAnalysis::analyse(ptas);

        // Build unified variable and clock lists.
        let (unified_vars, var_remap, initial_vals) =
            self.unify_variables(ptas, &analysis);
        let (unified_clocks, clock_remap) = self.unify_clocks(ptas);

        // Enumerate product locations via BFS.
        let (locations, location_map) = self.enumerate_locations(ptas)?;

        // Build product edges.
        let edges = self.build_edges(
            ptas,
            &locations,
            &location_map,
            &analysis,
            &var_remap,
            &clock_remap,
        );

        let initial_components: Vec<LocationId> =
            ptas.iter().map(|p| p.initial_location).collect();
        let initial_location = *location_map
            .get(&initial_components)
            .ok_or_else(|| {
                ModelCheckerError::PtaConstruction("initial location not in product".into())
            })?;

        let shared_variables: HashSet<VariableId> = unified_vars
            .iter()
            .filter(|v| {
                analysis.shared_names.contains(&v.name)
                    || v.kind == VariableKind::EnzymeActivity
            })
            .map(|v| v.id)
            .collect();

        Ok(ProductPTA {
            locations,
            edges,
            clocks: unified_clocks,
            variables: unified_vars,
            initial_location,
            initial_variable_values: initial_vals,
            component_names: ptas.iter().map(|p| p.name.clone()).collect(),
            shared_variables,
            location_map,
        })
    }

    /// Degenerate product: a single PTA wrapped as a ProductPTA.
    fn singleton_product(&self, pta: &PTA) -> Result<ProductPTA> {
        let locations: Vec<ProductLocation> = pta
            .locations
            .iter()
            .map(|l| ProductLocation {
                components: vec![l.id],
                product_id: l.id,
                name: format!("({})", l.name),
                kind: l.kind,
            })
            .collect();

        let location_map: HashMap<Vec<LocationId>, LocationId> = locations
            .iter()
            .map(|l| (l.components.clone(), l.product_id))
            .collect();

        let edges: Vec<ProductEdge> = pta
            .edges
            .iter()
            .map(|e| ProductEdge {
                id: e.id,
                source: e.source,
                target: e.target,
                guard: e.guard.clone(),
                action: e.action.clone(),
                updates: e.updates.clone(),
                participants: vec![0],
                origin_edges: vec![e.id],
            })
            .collect();

        Ok(ProductPTA {
            locations,
            edges,
            clocks: pta.clocks.clone(),
            variables: pta.variables.clone(),
            initial_location: pta.initial_location,
            initial_variable_values: pta.initial_variable_values.clone(),
            component_names: vec![pta.name.clone()],
            shared_variables: pta.shared_variable_ids(),
            location_map,
        })
    }

    /// Unify variables across PTAs, merging shared ones.
    fn unify_variables(
        &self,
        ptas: &[PTA],
        analysis: &SharedVariableAnalysis,
    ) -> (Vec<Variable>, Vec<HashMap<VariableId, VariableId>>, Vec<f64>) {
        let mut unified: Vec<Variable> = Vec::new();
        let mut name_to_id: HashMap<String, VariableId> = HashMap::new();
        let mut remap: Vec<HashMap<VariableId, VariableId>> = vec![HashMap::new(); ptas.len()];
        let mut initial_vals: Vec<f64> = Vec::new();

        for (pta_idx, pta) in ptas.iter().enumerate() {
            for var in &pta.variables {
                if let Some(&existing_id) = name_to_id.get(&var.name) {
                    // Shared variable – widen bounds.
                    let existing = &mut unified[existing_id];
                    existing.lower_bound = existing.lower_bound.min(var.lower_bound);
                    existing.upper_bound = existing.upper_bound.max(var.upper_bound);
                    remap[pta_idx].insert(var.id, existing_id);
                } else {
                    let new_id = unified.len();
                    unified.push(Variable {
                        id: new_id,
                        name: var.name.clone(),
                        kind: var.kind,
                        lower_bound: var.lower_bound,
                        upper_bound: var.upper_bound,
                    });
                    let init_val = pta
                        .initial_variable_values
                        .get(var.id)
                        .copied()
                        .unwrap_or(0.0);
                    initial_vals.push(init_val);
                    name_to_id.insert(var.name.clone(), new_id);
                    remap[pta_idx].insert(var.id, new_id);
                }
            }
        }

        (unified, remap, initial_vals)
    }

    /// Unify clocks across PTAs (all clocks are private → rename to avoid
    /// collisions).
    fn unify_clocks(
        &self,
        ptas: &[PTA],
    ) -> (Vec<Clock>, Vec<HashMap<ClockId, ClockId>>) {
        let mut unified: Vec<Clock> = Vec::new();
        let mut remap: Vec<HashMap<ClockId, ClockId>> = vec![HashMap::new(); ptas.len()];

        for (pta_idx, pta) in ptas.iter().enumerate() {
            for clk in &pta.clocks {
                let new_id = unified.len();
                unified.push(Clock {
                    id: new_id,
                    name: format!("{}_{}", pta.name, clk.name),
                });
                remap[pta_idx].insert(clk.id, new_id);
            }
        }

        (unified, remap)
    }

    /// Enumerate all reachable product locations via BFS.
    fn enumerate_locations(
        &self,
        ptas: &[PTA],
    ) -> Result<(Vec<ProductLocation>, HashMap<Vec<LocationId>, LocationId>)> {
        let mut locations: Vec<ProductLocation> = Vec::new();
        let mut map: HashMap<Vec<LocationId>, LocationId> = HashMap::new();
        let mut queue: VecDeque<Vec<LocationId>> = VecDeque::new();

        let initial: Vec<LocationId> = ptas.iter().map(|p| p.initial_location).collect();
        let pid = 0;
        let kind = derive_product_kind(ptas, &initial);
        locations.push(ProductLocation {
            components: initial.clone(),
            product_id: pid,
            name: format_tuple(ptas, &initial),
            kind,
        });
        map.insert(initial.clone(), pid);
        queue.push_back(initial);

        while let Some(current) = queue.pop_front() {
            if locations.len() > self.max_states {
                return Err(ModelCheckerError::StateSpaceExplosion {
                    num_states: locations.len(),
                    limit: self.max_states,
                });
            }

            // Generate all successors.
            let successors = self.compute_successors(ptas, &current);
            for succ in successors {
                if !map.contains_key(&succ) {
                    let new_id = locations.len();
                    let kind = derive_product_kind(ptas, &succ);
                    locations.push(ProductLocation {
                        components: succ.clone(),
                        product_id: new_id,
                        name: format_tuple(ptas, &succ),
                        kind,
                    });
                    map.insert(succ.clone(), new_id);
                    queue.push_back(succ);
                }
            }
        }

        Ok((locations, map))
    }

    /// Compute all successor location tuples from a given product location.
    fn compute_successors(&self, ptas: &[PTA], current: &[LocationId]) -> Vec<Vec<LocationId>> {
        let mut successors = Vec::new();

        // Interleaved moves: each component moves independently.
        for (idx, pta) in ptas.iter().enumerate() {
            for edge in pta.outgoing_edges(current[idx]) {
                let mut next = current.to_vec();
                next[idx] = edge.target;
                successors.push(next);
            }
        }

        // Synchronised moves: two components that share a sync label move together.
        for i in 0..ptas.len() {
            for j in (i + 1)..ptas.len() {
                let edges_i = ptas[i].outgoing_edges(current[i]);
                let edges_j = ptas[j].outgoing_edges(current[j]);

                for ei in &edges_i {
                    for ej in &edges_j {
                        if can_synchronize(&ei.action, &ej.action) {
                            let mut next = current.to_vec();
                            next[i] = ei.target;
                            next[j] = ej.target;
                            successors.push(next);
                        }
                    }
                }
            }
        }

        successors
    }

    /// Build all product edges.
    fn build_edges(
        &self,
        ptas: &[PTA],
        locations: &[ProductLocation],
        location_map: &HashMap<Vec<LocationId>, LocationId>,
        analysis: &SharedVariableAnalysis,
        var_remap: &[HashMap<VariableId, VariableId>],
        clock_remap: &[HashMap<ClockId, ClockId>],
    ) -> Vec<ProductEdge> {
        let mut edges: Vec<ProductEdge> = Vec::new();
        let mut next_id: EdgeId = 0;

        for loc in locations {
            let current = &loc.components;

            // Interleaved edges.
            for (idx, pta) in ptas.iter().enumerate() {
                for edge in pta.outgoing_edges(current[idx]) {
                    let mut target = current.clone();
                    target[idx] = edge.target;

                    if let Some(&target_pid) = location_map.get(&target) {
                        let remapped_guard =
                            remap_predicate(&edge.guard, &var_remap[idx], &clock_remap[idx]);
                        let remapped_updates =
                            remap_updates(&edge.updates, &var_remap[idx], &clock_remap[idx]);

                        edges.push(ProductEdge {
                            id: next_id,
                            source: loc.product_id,
                            target: target_pid,
                            guard: remapped_guard,
                            action: edge.action.clone(),
                            updates: remapped_updates,
                            participants: vec![idx],
                            origin_edges: vec![edge.id],
                        });
                        next_id += 1;
                    }
                }
            }

            // Synchronised edges.
            for i in 0..ptas.len() {
                for j in (i + 1)..ptas.len() {
                    let edges_i = ptas[i].outgoing_edges(current[i]);
                    let edges_j = ptas[j].outgoing_edges(current[j]);

                    for ei in &edges_i {
                        for ej in &edges_j {
                            if can_synchronize(&ei.action, &ej.action) {
                                let mut target = current.clone();
                                target[i] = ei.target;
                                target[j] = ej.target;

                                if let Some(&target_pid) = location_map.get(&target) {
                                    let guard_i = remap_predicate(
                                        &ei.guard,
                                        &var_remap[i],
                                        &clock_remap[i],
                                    );
                                    let guard_j = remap_predicate(
                                        &ej.guard,
                                        &var_remap[j],
                                        &clock_remap[j],
                                    );
                                    let combined_guard = guard_i.and(guard_j);

                                    let mut combined_updates = remap_updates(
                                        &ei.updates,
                                        &var_remap[i],
                                        &clock_remap[i],
                                    );
                                    combined_updates.extend(remap_updates(
                                        &ej.updates,
                                        &var_remap[j],
                                        &clock_remap[j],
                                    ));

                                    edges.push(ProductEdge {
                                        id: next_id,
                                        source: loc.product_id,
                                        target: target_pid,
                                        guard: combined_guard,
                                        action: ei.action.clone(),
                                        updates: combined_updates,
                                        participants: vec![i, j],
                                        origin_edges: vec![ei.id, ej.id],
                                    });
                                    next_id += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        edges
    }
}

// ---------------------------------------------------------------------------
// PartialProductBuilder
// ---------------------------------------------------------------------------

/// Builds the product incrementally (on-demand), only exploring locations
/// as they are requested.
#[derive(Debug)]
pub struct PartialProductBuilder {
    max_states: usize,
    explored: HashMap<Vec<LocationId>, LocationId>,
    locations: Vec<ProductLocation>,
    edges: Vec<ProductEdge>,
    next_edge_id: EdgeId,
}

impl PartialProductBuilder {
    pub fn new(max_states: usize) -> Self {
        Self {
            max_states,
            explored: HashMap::new(),
            locations: Vec::new(),
            edges: Vec::new(),
            next_edge_id: 0,
        }
    }

    /// Register the initial product location.
    pub fn register_initial(&mut self, ptas: &[PTA]) -> Result<LocationId> {
        let initial: Vec<LocationId> = ptas.iter().map(|p| p.initial_location).collect();
        self.register_location(ptas, &initial)
    }

    /// Register (or look up) a product location from a component tuple.
    pub fn register_location(
        &mut self,
        ptas: &[PTA],
        components: &[LocationId],
    ) -> Result<LocationId> {
        if let Some(&pid) = self.explored.get(components) {
            return Ok(pid);
        }

        if self.locations.len() >= self.max_states {
            return Err(ModelCheckerError::StateSpaceExplosion {
                num_states: self.locations.len(),
                limit: self.max_states,
            });
        }

        let pid = self.locations.len();
        let kind = derive_product_kind(ptas, components);
        self.locations.push(ProductLocation {
            components: components.to_vec(),
            product_id: pid,
            name: format_tuple(ptas, components),
            kind,
        });
        self.explored.insert(components.to_vec(), pid);
        Ok(pid)
    }

    /// Expand all outgoing transitions from the given product location.
    pub fn expand(
        &mut self,
        ptas: &[PTA],
        product_loc: LocationId,
        var_remap: &[HashMap<VariableId, VariableId>],
        clock_remap: &[HashMap<ClockId, ClockId>],
    ) -> Result<Vec<LocationId>> {
        let components = self.locations[product_loc].components.clone();
        let mut new_locations = Vec::new();

        // Interleaved edges.
        for (idx, pta) in ptas.iter().enumerate() {
            for edge in pta.outgoing_edges(components[idx]) {
                let mut target = components.clone();
                target[idx] = edge.target;
                let target_pid = self.register_location(ptas, &target)?;
                new_locations.push(target_pid);

                let remapped_guard =
                    remap_predicate(&edge.guard, &var_remap[idx], &clock_remap[idx]);
                let remapped_updates =
                    remap_updates(&edge.updates, &var_remap[idx], &clock_remap[idx]);

                self.edges.push(ProductEdge {
                    id: self.next_edge_id,
                    source: product_loc,
                    target: target_pid,
                    guard: remapped_guard,
                    action: edge.action.clone(),
                    updates: remapped_updates,
                    participants: vec![idx],
                    origin_edges: vec![edge.id],
                });
                self.next_edge_id += 1;
            }
        }

        Ok(new_locations)
    }

    /// How many locations have been explored so far.
    pub fn explored_count(&self) -> usize {
        self.locations.len()
    }

    /// Convert to a full ProductPTA.
    pub fn into_product(
        self,
        ptas: &[PTA],
        unified_clocks: Vec<Clock>,
        unified_vars: Vec<Variable>,
        initial_vals: Vec<f64>,
        shared_variables: HashSet<VariableId>,
    ) -> ProductPTA {
        let initial: Vec<LocationId> = ptas.iter().map(|p| p.initial_location).collect();
        let initial_location = self.explored.get(&initial).copied().unwrap_or(0);

        ProductPTA {
            locations: self.locations,
            edges: self.edges,
            clocks: unified_clocks,
            variables: unified_vars,
            initial_location,
            initial_variable_values: initial_vals,
            component_names: ptas.iter().map(|p| p.name.clone()).collect(),
            shared_variables,
            location_map: self.explored,
        }
    }
}

// ---------------------------------------------------------------------------
// ProductStateExplorer
// ---------------------------------------------------------------------------

/// Explores the product state space one step at a time, suitable for
/// on-the-fly algorithms.
#[derive(Debug)]
pub struct ProductStateExplorer {
    /// Current component location tuple.
    pub current: Vec<LocationId>,
    /// Current clock valuations.
    pub clocks: Vec<f64>,
    /// Current variable valuations.
    pub variables: Vec<f64>,
    /// Trace of visited (component-location tuples, edge label).
    pub trace: Vec<(Vec<LocationId>, ActionLabel)>,
    /// Maximum exploration depth.
    pub max_depth: usize,
}

impl ProductStateExplorer {
    pub fn new(ptas: &[PTA], max_depth: usize) -> Self {
        let initial: Vec<LocationId> = ptas.iter().map(|p| p.initial_location).collect();
        let num_clocks: usize = ptas.iter().map(|p| p.clocks.len()).sum();
        let all_vars: Vec<f64> = ptas
            .iter()
            .flat_map(|p| p.initial_variable_values.iter().copied())
            .collect();

        Self {
            current: initial,
            clocks: vec![0.0; num_clocks],
            variables: all_vars,
            trace: Vec::new(),
            max_depth,
        }
    }

    /// Available interleaved edges from the current state.
    pub fn available_edges<'a>(&self, ptas: &'a [PTA]) -> Vec<(usize, &'a Edge)> {
        let mut result = Vec::new();
        for (idx, pta) in ptas.iter().enumerate() {
            for edge in pta.outgoing_edges(self.current[idx]) {
                if edge.guard.evaluate(&self.clocks, &self.variables) {
                    result.push((idx, edge));
                }
            }
        }
        result
    }

    /// Take an interleaved step: component `idx` fires the given edge.
    pub fn step(&mut self, idx: usize, edge: &Edge) {
        self.current[idx] = edge.target;
        self.apply_updates(&edge.updates);
        self.trace.push((self.current.clone(), edge.action.clone()));
    }

    /// Let time elapse by `dt` hours (advance all clocks).
    pub fn elapse(&mut self, dt: f64) {
        for c in &mut self.clocks {
            *c += dt;
        }
    }

    /// Apply a list of updates to clocks and variables.
    fn apply_updates(&mut self, updates: &[Update]) {
        for u in updates {
            match u {
                Update::ClockReset(c) => {
                    if let Some(v) = self.clocks.get_mut(*c) {
                        *v = 0.0;
                    }
                }
                Update::VarAssign { var, value } => {
                    if let Some(v) = self.variables.get_mut(*var) {
                        *v = *value;
                    }
                }
                Update::VarIncrement { var, delta } => {
                    if let Some(v) = self.variables.get_mut(*var) {
                        *v += delta;
                    }
                }
                Update::VarScale { var, factor } => {
                    if let Some(v) = self.variables.get_mut(*var) {
                        *v *= factor;
                    }
                }
            }
        }
    }

    /// Current exploration depth.
    pub fn depth(&self) -> usize {
        self.trace.len()
    }

    /// Whether the maximum depth has been reached.
    pub fn at_max_depth(&self) -> bool {
        self.trace.len() >= self.max_depth
    }
}

// ---------------------------------------------------------------------------
// OnTheFlyProduct
// ---------------------------------------------------------------------------

/// Lazy product construction used during model checking: locations and edges
/// are generated only when first encountered.
#[derive(Debug)]
pub struct OnTheFlyProduct {
    partial: PartialProductBuilder,
    analysis: Option<SharedVariableAnalysis>,
}

impl OnTheFlyProduct {
    pub fn new(max_states: usize) -> Self {
        Self {
            partial: PartialProductBuilder::new(max_states),
            analysis: None,
        }
    }

    /// Initialise with a set of PTAs.
    pub fn init(&mut self, ptas: &[PTA]) -> Result<LocationId> {
        self.analysis = Some(SharedVariableAnalysis::analyse(ptas));
        self.partial.register_initial(ptas)
    }

    /// Get the successor locations from a product location (expanding lazily).
    pub fn successors(
        &mut self,
        ptas: &[PTA],
        loc: LocationId,
        var_remap: &[HashMap<VariableId, VariableId>],
        clock_remap: &[HashMap<ClockId, ClockId>],
    ) -> Result<Vec<LocationId>> {
        self.partial.expand(ptas, loc, var_remap, clock_remap)
    }

    /// Current number of explored locations.
    pub fn explored(&self) -> usize {
        self.partial.explored_count()
    }

    /// Whether a location is an error location.
    pub fn is_error(&self, loc: LocationId) -> bool {
        self.partial
            .locations
            .get(loc)
            .map_or(false, |l| l.kind == LocationKind::Error)
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Estimate the product state-space size (upper bound).
pub fn compute_product_size(ptas: &[PTA]) -> usize {
    ptas.iter().map(|p| p.num_locations()).product()
}

/// Determine whether two action labels should synchronise.
pub fn can_synchronize(a: &ActionLabel, b: &ActionLabel) -> bool {
    match (a, b) {
        (ActionLabel::Sync(la), ActionLabel::Sync(lb)) => la == lb,
        (
            ActionLabel::InhibitEnzyme { enzyme: ea, .. },
            ActionLabel::InhibitEnzyme { enzyme: eb, .. },
        ) => ea == eb,
        (
            ActionLabel::InduceEnzyme { enzyme: ea, .. },
            ActionLabel::InduceEnzyme { enzyme: eb, .. },
        ) => ea == eb,
        (
            ActionLabel::InhibitEnzyme { enzyme: ea, .. },
            ActionLabel::InduceEnzyme { enzyme: eb, .. },
        ) => ea == eb,
        (
            ActionLabel::InduceEnzyme { enzyme: ea, .. },
            ActionLabel::InhibitEnzyme { enzyme: eb, .. },
        ) => ea == eb,
        _ => false,
    }
}

/// Derive the kind of a product location from its components.
fn derive_product_kind(ptas: &[PTA], components: &[LocationId]) -> LocationKind {
    let mut any_error = false;
    let mut all_initial = true;
    let mut any_urgent = false;

    for (idx, &loc_id) in components.iter().enumerate() {
        if let Some(loc) = ptas[idx].location(loc_id) {
            match loc.kind {
                LocationKind::Error => any_error = true,
                LocationKind::Initial => {}
                LocationKind::Urgent => any_urgent = true,
                _ => all_initial = false,
            }
            if loc.kind != LocationKind::Initial {
                all_initial = false;
            }
        }
    }

    if any_error {
        LocationKind::Error
    } else if any_urgent {
        LocationKind::Urgent
    } else if all_initial {
        LocationKind::Initial
    } else {
        LocationKind::Normal
    }
}

/// Format a product location tuple as a string.
fn format_tuple(ptas: &[PTA], components: &[LocationId]) -> String {
    let names: Vec<String> = components
        .iter()
        .enumerate()
        .map(|(idx, &lid)| {
            ptas[idx]
                .location(lid)
                .map(|l| l.name.clone())
                .unwrap_or_else(|| format!("?{lid}"))
        })
        .collect();
    format!("({})", names.join(", "))
}

/// Remap variable and clock IDs in a predicate.
fn remap_predicate(
    pred: &Predicate,
    var_map: &HashMap<VariableId, VariableId>,
    clock_map: &HashMap<ClockId, ClockId>,
) -> Predicate {
    let conjuncts: Vec<AtomicPredicate> = pred
        .conjuncts
        .iter()
        .map(|ap| match ap {
            AtomicPredicate::ClockLeq { clock, bound } => AtomicPredicate::ClockLeq {
                clock: clock_map.get(clock).copied().unwrap_or(*clock),
                bound: *bound,
            },
            AtomicPredicate::ClockGeq { clock, bound } => AtomicPredicate::ClockGeq {
                clock: clock_map.get(clock).copied().unwrap_or(*clock),
                bound: *bound,
            },
            AtomicPredicate::ClockEq { clock, value } => AtomicPredicate::ClockEq {
                clock: clock_map.get(clock).copied().unwrap_or(*clock),
                value: *value,
            },
            AtomicPredicate::VarLeq { var, bound } => AtomicPredicate::VarLeq {
                var: var_map.get(var).copied().unwrap_or(*var),
                bound: *bound,
            },
            AtomicPredicate::VarGeq { var, bound } => AtomicPredicate::VarGeq {
                var: var_map.get(var).copied().unwrap_or(*var),
                bound: *bound,
            },
            AtomicPredicate::VarInRange { var, lo, hi } => AtomicPredicate::VarInRange {
                var: var_map.get(var).copied().unwrap_or(*var),
                lo: *lo,
                hi: *hi,
            },
            AtomicPredicate::BoolConst(b) => AtomicPredicate::BoolConst(*b),
        })
        .collect();
    Predicate { conjuncts }
}

/// Remap variable and clock IDs in a list of updates.
fn remap_updates(
    updates: &[Update],
    var_map: &HashMap<VariableId, VariableId>,
    clock_map: &HashMap<ClockId, ClockId>,
) -> Vec<Update> {
    updates
        .iter()
        .map(|u| match u {
            Update::ClockReset(c) => {
                Update::ClockReset(clock_map.get(c).copied().unwrap_or(*c))
            }
            Update::VarAssign { var, value } => Update::VarAssign {
                var: var_map.get(var).copied().unwrap_or(*var),
                value: *value,
            },
            Update::VarIncrement { var, delta } => Update::VarIncrement {
                var: var_map.get(var).copied().unwrap_or(*var),
                delta: *delta,
            },
            Update::VarScale { var, factor } => Update::VarScale {
                var: var_map.get(var).copied().unwrap_or(*var),
                factor: *factor,
            },
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{make_test_pta, CypEnzyme, CypMetabolismRoute, DosingSchedule, DrugId, PkParameters, PtaBuilder};

    fn two_drug_ptas() -> (PTA, PTA) {
        let pta_a = make_test_pta("drugA", 500.0, false);
        let pta_b = make_test_pta("drugB", 250.0, false);
        (pta_a, pta_b)
    }

    #[test]
    fn test_compute_product_size() {
        let (a, b) = two_drug_ptas();
        let size = compute_product_size(&[a.clone(), b.clone()]);
        assert_eq!(size, a.num_locations() * b.num_locations());
    }

    #[test]
    fn test_singleton_product() {
        let pta = make_test_pta("metformin", 500.0, true);
        let builder = ProductBuilder::new(10_000);
        let product = builder.build_product(&[pta.clone()]).unwrap();
        assert_eq!(product.num_locations(), pta.num_locations());
        assert_eq!(product.num_components(), 1);
    }

    #[test]
    fn test_product_two_ptas() {
        let (a, b) = two_drug_ptas();
        let builder = ProductBuilder::new(100_000);
        let product = builder.build_product(&[a, b]).unwrap();
        assert!(product.num_locations() > 1);
        assert!(!product.edges.is_empty());
    }

    #[test]
    fn test_product_initial_location() {
        let (a, b) = two_drug_ptas();
        let builder = ProductBuilder::new(100_000);
        let product = builder.build_product(&[a, b]).unwrap();
        let init_loc = &product.locations[product.initial_location];
        assert_eq!(init_loc.kind, LocationKind::Initial);
    }

    #[test]
    fn test_product_error_propagation() {
        let a = make_test_pta("drugA", 500.0, true); // has toxic/error loc
        let b = make_test_pta("drugB", 250.0, false);
        let builder = ProductBuilder::new(100_000);
        let product = builder.build_product(&[a, b]).unwrap();
        assert!(product.has_error_locations());
    }

    #[test]
    fn test_state_space_explosion_guard() {
        let (a, b) = two_drug_ptas();
        let builder = ProductBuilder::new(1); // tiny limit
        let result = builder.build_product(&[a, b]);
        assert!(result.is_err());
    }

    #[test]
    fn test_shared_variable_analysis() {
        let mut a = make_test_pta("drugA", 500.0, false);
        let mut b = make_test_pta("drugB", 250.0, false);

        // Add a shared enzyme variable to both.
        a.variables.push(Variable {
            id: a.variables.len(),
            name: "cyp3a4_activity".into(),
            kind: VariableKind::EnzymeActivity,
            lower_bound: 0.0,
            upper_bound: 2.0,
        });
        b.variables.push(Variable {
            id: b.variables.len(),
            name: "cyp3a4_activity".into(),
            kind: VariableKind::EnzymeActivity,
            lower_bound: 0.0,
            upper_bound: 2.0,
        });

        let analysis = SharedVariableAnalysis::analyse(&[a, b]);
        assert!(analysis.shared_names.contains("cyp3a4_activity"));
        assert_eq!(analysis.num_shared(), 1);
        assert!(analysis.share_variables(0, 1));
    }

    #[test]
    fn test_can_synchronize_sync() {
        let a = ActionLabel::Sync("cyp3a4".into());
        let b = ActionLabel::Sync("cyp3a4".into());
        assert!(can_synchronize(&a, &b));

        let c = ActionLabel::Sync("cyp2d6".into());
        assert!(!can_synchronize(&a, &c));
    }

    #[test]
    fn test_can_synchronize_enzyme() {
        let a = ActionLabel::InhibitEnzyme {
            enzyme: "CYP3A4".into(),
            drug: "drugA".into(),
        };
        let b = ActionLabel::InhibitEnzyme {
            enzyme: "CYP3A4".into(),
            drug: "drugB".into(),
        };
        assert!(can_synchronize(&a, &b));
    }

    #[test]
    fn test_can_synchronize_different_actions() {
        let a = ActionLabel::Administer { drug: "drugA".into() };
        let b = ActionLabel::Administer { drug: "drugB".into() };
        assert!(!can_synchronize(&a, &b)); // administer is local
    }

    #[test]
    fn test_partial_product_builder() {
        let (a, b) = two_drug_ptas();
        let ptas = [a, b];
        let mut partial = PartialProductBuilder::new(10_000);
        let init = partial.register_initial(&ptas).unwrap();
        assert_eq!(init, 0);
        assert_eq!(partial.explored_count(), 1);
    }

    #[test]
    fn test_product_state_explorer() {
        let pta = make_test_pta("aspirin", 100.0, false);
        let explorer = ProductStateExplorer::new(&[pta.clone()], 10);
        assert_eq!(explorer.current, vec![pta.initial_location]);
        assert!(!explorer.at_max_depth());
    }

    #[test]
    fn test_on_the_fly_product_init() {
        let ptas = vec![make_test_pta("drugA", 500.0, false)];
        let mut otf = OnTheFlyProduct::new(10_000);
        let init = otf.init(&ptas).unwrap();
        assert_eq!(init, 0);
        assert_eq!(otf.explored(), 1);
    }

    #[test]
    fn test_remap_predicate() {
        let pred = Predicate::from_conjuncts(vec![
            AtomicPredicate::VarLeq { var: 0, bound: 5.0 },
            AtomicPredicate::ClockGeq { clock: 0, bound: 1.0 },
        ]);
        let mut var_map = HashMap::new();
        var_map.insert(0, 3);
        let mut clock_map = HashMap::new();
        clock_map.insert(0, 7);

        let remapped = remap_predicate(&pred, &var_map, &clock_map);
        match &remapped.conjuncts[0] {
            AtomicPredicate::VarLeq { var, .. } => assert_eq!(*var, 3),
            _ => panic!("expected VarLeq"),
        }
        match &remapped.conjuncts[1] {
            AtomicPredicate::ClockGeq { clock, .. } => assert_eq!(*clock, 7),
            _ => panic!("expected ClockGeq"),
        }
    }

    #[test]
    fn test_product_location_display() {
        let loc = ProductLocation {
            components: vec![0, 1],
            product_id: 0,
            name: "(idle, absorbing)".into(),
            kind: LocationKind::Normal,
        };
        assert_eq!(format!("{loc}"), "(idle, absorbing)");
    }
}
