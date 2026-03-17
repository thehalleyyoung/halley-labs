//! SMT variable management.
//!
//! Provides the core variable infrastructure for SMT encoding: sorted
//! variables, variable stores, time-indexed copies for bounded model
//! checking, and symbol tables mapping PTA names to SMT identifiers.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════
// VariableId
// ═══════════════════════════════════════════════════════════════════════════

/// Opaque, cheap-to-copy identifier for an SMT variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VariableId(pub u32);

impl VariableId {
    /// Construct from a raw index (intended for store use only).
    pub fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw numeric value.
    pub fn raw(self) -> u32 {
        self.0
    }
}

impl fmt::Display for VariableId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SmtSort
// ═══════════════════════════════════════════════════════════════════════════

/// SMT-LIB2 sorts (types) for variables.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SmtSort {
    Bool,
    Int,
    Real,
    BitVec(usize),
    Array(Box<SmtSort>, Box<SmtSort>),
    Enum(Vec<String>),
}

impl SmtSort {
    pub fn bitvec(width: usize) -> Self {
        SmtSort::BitVec(width)
    }

    pub fn array(index: SmtSort, element: SmtSort) -> Self {
        SmtSort::Array(Box::new(index), Box::new(element))
    }

    pub fn enum_sort(variants: Vec<String>) -> Self {
        SmtSort::Enum(variants)
    }

    /// True for numeric sorts (Int or Real).
    pub fn is_numeric(&self) -> bool {
        matches!(self, SmtSort::Int | SmtSort::Real)
    }

    /// True for boolean sort.
    pub fn is_bool(&self) -> bool {
        matches!(self, SmtSort::Bool)
    }

    /// SMT-LIB2 sort name.
    pub fn to_smtlib2(&self) -> String {
        match self {
            SmtSort::Bool => "Bool".to_string(),
            SmtSort::Int => "Int".to_string(),
            SmtSort::Real => "Real".to_string(),
            SmtSort::BitVec(w) => format!("(_ BitVec {})", w),
            SmtSort::Array(i, e) => format!("(Array {} {})", i.to_smtlib2(), e.to_smtlib2()),
            SmtSort::Enum(variants) => {
                let vs: Vec<_> = variants.iter().map(|v| v.as_str()).collect();
                format!("(Enum {})", vs.join(" "))
            }
        }
    }

    /// Returns the bit-width for bitvectors, None otherwise.
    pub fn bitvec_width(&self) -> Option<usize> {
        match self {
            SmtSort::BitVec(w) => Some(*w),
            _ => None,
        }
    }

    /// Returns the enum variants if this is an Enum sort.
    pub fn enum_variants(&self) -> Option<&[String]> {
        match self {
            SmtSort::Enum(vs) => Some(vs),
            _ => None,
        }
    }
}

impl fmt::Display for SmtSort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_smtlib2())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SmtVariable
// ═══════════════════════════════════════════════════════════════════════════

/// An SMT variable with its sort and optional time-step index.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SmtVariable {
    pub id: VariableId,
    pub name: String,
    pub sort: SmtSort,
    pub time_step: Option<usize>,
}

impl SmtVariable {
    pub fn new(id: VariableId, name: &str, sort: SmtSort) -> Self {
        Self {
            id,
            name: name.to_string(),
            sort,
            time_step: None,
        }
    }

    pub fn with_time_step(mut self, step: usize) -> Self {
        self.time_step = Some(step);
        self
    }

    /// The fully-qualified SMT name including time index.
    pub fn qualified_name(&self) -> String {
        match self.time_step {
            Some(t) => format!("{}_t{}", self.name, t),
            None => self.name.clone(),
        }
    }

    /// SMT-LIB2 declaration string.
    pub fn to_declare_const(&self) -> String {
        format!("(declare-const {} {})", self.qualified_name(), self.sort.to_smtlib2())
    }

    /// True if this variable is boolean-sorted.
    pub fn is_bool(&self) -> bool {
        self.sort.is_bool()
    }

    /// True if this variable is numerically sorted.
    pub fn is_numeric(&self) -> bool {
        self.sort.is_numeric()
    }
}

impl fmt::Display for SmtVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.qualified_name(), self.sort)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TimeIndexedVariable
// ═══════════════════════════════════════════════════════════════════════════

/// A variable that has time-indexed copies (one per step in BMC).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeIndexedVariable {
    /// The base variable name (without time suffix).
    pub base_name: String,
    /// Sort of the variable.
    pub sort: SmtSort,
    /// Map from time step to VariableId.
    pub instances: HashMap<usize, VariableId>,
    /// The base (un-indexed) VariableId.
    pub base_id: VariableId,
}

impl TimeIndexedVariable {
    pub fn new(base_name: &str, sort: SmtSort, base_id: VariableId) -> Self {
        Self {
            base_name: base_name.to_string(),
            sort,
            instances: HashMap::new(),
            base_id,
        }
    }

    /// Get the variable id for a specific time step.
    pub fn at_step(&self, step: usize) -> Option<VariableId> {
        self.instances.get(&step).copied()
    }

    /// Register a variable id for a time step.
    pub fn add_step(&mut self, step: usize, id: VariableId) {
        self.instances.insert(step, id);
    }

    /// The name at a given step.
    pub fn name_at_step(&self, step: usize) -> String {
        format!("{}_t{}", self.base_name, step)
    }

    /// All steps for which instances exist, sorted.
    pub fn steps(&self) -> Vec<usize> {
        let mut s: Vec<_> = self.instances.keys().copied().collect();
        s.sort();
        s
    }

    /// Number of time-indexed instances.
    pub fn num_instances(&self) -> usize {
        self.instances.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// VariableStore
// ═══════════════════════════════════════════════════════════════════════════

/// Central store for all SMT variables used in an encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableStore {
    variables: Vec<SmtVariable>,
    name_index: HashMap<String, VariableId>,
    time_indexed: HashMap<String, TimeIndexedVariable>,
    next_id: u32,
}

impl VariableStore {
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            name_index: HashMap::new(),
            time_indexed: HashMap::new(),
            next_id: 0,
        }
    }

    fn alloc_id(&mut self) -> VariableId {
        let id = VariableId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Create a new variable with the given name and sort.
    pub fn create_variable(&mut self, name: &str, sort: SmtSort) -> VariableId {
        if let Some(&existing) = self.name_index.get(name) {
            return existing;
        }
        let id = self.alloc_id();
        let var = SmtVariable::new(id, name, sort);
        self.name_index.insert(name.to_string(), id);
        self.variables.push(var);
        id
    }

    /// Create a boolean variable.
    pub fn create_bool(&mut self, name: &str) -> VariableId {
        self.create_variable(name, SmtSort::Bool)
    }

    /// Create an integer variable.
    pub fn create_int(&mut self, name: &str) -> VariableId {
        self.create_variable(name, SmtSort::Int)
    }

    /// Create a real-valued variable.
    pub fn create_real(&mut self, name: &str) -> VariableId {
        self.create_variable(name, SmtSort::Real)
    }

    /// Create a bitvector variable of given width.
    pub fn create_bitvec(&mut self, name: &str, width: usize) -> VariableId {
        self.create_variable(name, SmtSort::BitVec(width))
    }

    /// Create a state variable (integer encoding of location).
    pub fn create_state_var(&mut self, name: &str) -> VariableId {
        self.create_variable(name, SmtSort::Int)
    }

    /// Create a concentration variable (real-valued).
    pub fn create_concentration_var(&mut self, name: &str) -> VariableId {
        self.create_variable(name, SmtSort::Real)
    }

    /// Create a clock variable (real-valued, non-negative).
    pub fn create_clock_var(&mut self, name: &str) -> VariableId {
        self.create_variable(name, SmtSort::Real)
    }

    /// Create an enzyme activity variable (real-valued in [0,∞)).
    pub fn create_enzyme_var(&mut self, name: &str) -> VariableId {
        self.create_variable(name, SmtSort::Real)
    }

    /// Create a time-indexed variable and its instance at `step`.
    pub fn create_time_indexed(
        &mut self,
        base_name: &str,
        sort: SmtSort,
        step: usize,
    ) -> VariableId {
        let step_name = format!("{}_t{}", base_name, step);
        let id = self.create_variable(&step_name, sort.clone());

        let entry = self.time_indexed
            .entry(base_name.to_string())
            .or_insert_with(|| {
                let base_id = self.create_variable(base_name, sort.clone());
                TimeIndexedVariable::new(base_name, sort, base_id)
            });
        entry.add_step(step, id);

        id
    }

    /// Get a variable by id.
    pub fn get(&self, id: VariableId) -> Option<&SmtVariable> {
        self.variables.get(id.0 as usize)
    }

    /// Look up a variable by name.
    pub fn get_by_name(&self, name: &str) -> Option<&SmtVariable> {
        self.name_index.get(name).and_then(|&id| self.get(id))
    }

    /// Look up a variable id by name.
    pub fn id_by_name(&self, name: &str) -> Option<VariableId> {
        self.name_index.get(name).copied()
    }

    /// Get a time-indexed variable set.
    pub fn get_time_indexed(&self, base_name: &str) -> Option<&TimeIndexedVariable> {
        self.time_indexed.get(base_name)
    }

    /// Get variable id at a specific time step.
    pub fn id_at_step(&self, base_name: &str, step: usize) -> Option<VariableId> {
        let step_name = format!("{}_t{}", base_name, step);
        self.id_by_name(&step_name)
    }

    /// Total number of variables.
    pub fn len(&self) -> usize {
        self.variables.len()
    }

    /// True if no variables have been created.
    pub fn is_empty(&self) -> bool {
        self.variables.is_empty()
    }

    /// Iterate over all variables.
    pub fn iter(&self) -> impl Iterator<Item = &SmtVariable> {
        self.variables.iter()
    }

    /// All variable names.
    pub fn names(&self) -> Vec<&str> {
        self.variables.iter().map(|v| v.name.as_str()).collect()
    }

    /// Generate all SMT-LIB2 declarations.
    pub fn to_declarations(&self) -> Vec<String> {
        self.variables.iter().map(|v| v.to_declare_const()).collect()
    }

    /// All time-indexed base names.
    pub fn time_indexed_names(&self) -> Vec<&str> {
        self.time_indexed.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for VariableStore {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// VariableFactory
// ═══════════════════════════════════════════════════════════════════════════

/// Factory that produces time-indexed copies of state, clock, and
/// concentration variables for each BMC step.
#[derive(Debug, Clone)]
pub struct VariableFactory {
    /// Base names and sorts for location/state variables.
    state_vars: Vec<(String, SmtSort)>,
    /// Base names for clock variables.
    clock_vars: Vec<String>,
    /// Base names for concentration variables.
    conc_vars: Vec<String>,
    /// Base names for enzyme variables.
    enzyme_vars: Vec<String>,
    /// Extra auxiliary variables.
    aux_vars: Vec<(String, SmtSort)>,
}

impl VariableFactory {
    pub fn new() -> Self {
        Self {
            state_vars: Vec::new(),
            clock_vars: Vec::new(),
            conc_vars: Vec::new(),
            enzyme_vars: Vec::new(),
            aux_vars: Vec::new(),
        }
    }

    pub fn add_state_var(&mut self, name: &str, sort: SmtSort) {
        self.state_vars.push((name.to_string(), sort));
    }

    pub fn add_clock_var(&mut self, name: &str) {
        self.clock_vars.push(name.to_string());
    }

    pub fn add_concentration_var(&mut self, name: &str) {
        self.conc_vars.push(name.to_string());
    }

    pub fn add_enzyme_var(&mut self, name: &str) {
        self.enzyme_vars.push(name.to_string());
    }

    pub fn add_aux_var(&mut self, name: &str, sort: SmtSort) {
        self.aux_vars.push((name.to_string(), sort));
    }

    /// Instantiate variables for steps `0..=bound` into the store.
    pub fn instantiate(&self, store: &mut VariableStore, bound: usize) {
        for step in 0..=bound {
            // Location variable
            store.create_time_indexed("loc", SmtSort::Int, step);

            for (name, sort) in &self.state_vars {
                store.create_time_indexed(name, sort.clone(), step);
            }

            for name in &self.clock_vars {
                store.create_time_indexed(name, SmtSort::Real, step);
            }

            for name in &self.conc_vars {
                store.create_time_indexed(name, SmtSort::Real, step);
            }

            for name in &self.enzyme_vars {
                store.create_time_indexed(name, SmtSort::Real, step);
            }

            for (name, sort) in &self.aux_vars {
                store.create_time_indexed(name, sort.clone(), step);
            }

            // Transition selector for step (not for the last step)
            if step < bound {
                store.create_time_indexed("trans", SmtSort::Int, step);
            }
        }
    }

    /// Number of base variables (before time indexing).
    pub fn num_base_vars(&self) -> usize {
        1 + self.state_vars.len()
            + self.clock_vars.len()
            + self.conc_vars.len()
            + self.enzyme_vars.len()
            + self.aux_vars.len()
    }

    /// Estimated total variables for a given bound.
    pub fn estimated_total_vars(&self, bound: usize) -> usize {
        let per_step = self.num_base_vars();
        let total_steps = bound + 1;
        // +bound for transition selectors
        per_step * total_steps + bound
    }
}

impl Default for VariableFactory {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SymbolTable
// ═══════════════════════════════════════════════════════════════════════════

/// Maps PTA variable names to SMT variable identifiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolTable {
    /// Location encoding: LocationId -> integer index.
    location_encoding: HashMap<String, i64>,
    /// Reverse map: integer index -> LocationId.
    location_decoding: HashMap<i64, String>,
    /// Clock variable names -> base SMT variable names.
    clock_map: HashMap<String, String>,
    /// Concentration variable names -> base SMT variable names.
    concentration_map: HashMap<String, String>,
    /// State variable names -> base SMT variable names.
    state_map: HashMap<String, String>,
    /// Enzyme variable names -> base SMT variable names.
    enzyme_map: HashMap<String, String>,
    /// Edge id -> integer index.
    edge_encoding: HashMap<String, i64>,
    /// Reverse map: integer index -> edge id.
    edge_decoding: HashMap<i64, String>,
    /// Stutter transition index.
    stutter_index: i64,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            location_encoding: HashMap::new(),
            location_decoding: HashMap::new(),
            clock_map: HashMap::new(),
            concentration_map: HashMap::new(),
            state_map: HashMap::new(),
            enzyme_map: HashMap::new(),
            edge_encoding: HashMap::new(),
            edge_decoding: HashMap::new(),
            stutter_index: -1,
        }
    }

    /// Register a location and assign it an integer encoding.
    pub fn register_location(&mut self, loc_id: &str, index: i64) {
        self.location_encoding.insert(loc_id.to_string(), index);
        self.location_decoding.insert(index, loc_id.to_string());
    }

    /// Register a clock variable mapping.
    pub fn register_clock(&mut self, pta_name: &str, smt_name: &str) {
        self.clock_map.insert(pta_name.to_string(), smt_name.to_string());
    }

    /// Register a concentration variable mapping.
    pub fn register_concentration(&mut self, pta_name: &str, smt_name: &str) {
        self.concentration_map.insert(pta_name.to_string(), smt_name.to_string());
    }

    /// Register a state variable mapping.
    pub fn register_state(&mut self, pta_name: &str, smt_name: &str) {
        self.state_map.insert(pta_name.to_string(), smt_name.to_string());
    }

    /// Register an enzyme variable mapping.
    pub fn register_enzyme(&mut self, pta_name: &str, smt_name: &str) {
        self.enzyme_map.insert(pta_name.to_string(), smt_name.to_string());
    }

    /// Register an edge and assign it an integer encoding.
    pub fn register_edge(&mut self, edge_id: &str, index: i64) {
        self.edge_encoding.insert(edge_id.to_string(), index);
        self.edge_decoding.insert(index, edge_id.to_string());
    }

    /// Set the stutter transition index.
    pub fn set_stutter_index(&mut self, index: i64) {
        self.stutter_index = index;
    }

    /// Get the integer encoding of a location.
    pub fn location_index(&self, loc_id: &str) -> Option<i64> {
        self.location_encoding.get(loc_id).copied()
    }

    /// Get the location name from an integer encoding.
    pub fn location_name(&self, index: i64) -> Option<&str> {
        self.location_decoding.get(&index).map(|s| s.as_str())
    }

    /// Get the SMT base name for a clock variable.
    pub fn clock_smt_name(&self, pta_name: &str) -> Option<&str> {
        self.clock_map.get(pta_name).map(|s| s.as_str())
    }

    /// Get the SMT base name for a concentration variable.
    pub fn concentration_smt_name(&self, pta_name: &str) -> Option<&str> {
        self.concentration_map.get(pta_name).map(|s| s.as_str())
    }

    /// Get the SMT base name for a state variable.
    pub fn state_smt_name(&self, pta_name: &str) -> Option<&str> {
        self.state_map.get(pta_name).map(|s| s.as_str())
    }

    /// Get the SMT base name for an enzyme variable.
    pub fn enzyme_smt_name(&self, pta_name: &str) -> Option<&str> {
        self.enzyme_map.get(pta_name).map(|s| s.as_str())
    }

    /// Get the integer encoding of an edge.
    pub fn edge_index(&self, edge_id: &str) -> Option<i64> {
        self.edge_encoding.get(edge_id).copied()
    }

    /// Get the edge name from an integer encoding.
    pub fn edge_name(&self, index: i64) -> Option<&str> {
        self.edge_decoding.get(&index).map(|s| s.as_str())
    }

    /// The stutter transition index.
    pub fn stutter_index(&self) -> i64 {
        self.stutter_index
    }

    /// Number of registered locations.
    pub fn num_locations(&self) -> usize {
        self.location_encoding.len()
    }

    /// Number of registered edges.
    pub fn num_edges(&self) -> usize {
        self.edge_encoding.len()
    }

    /// All registered location names (sorted by index).
    pub fn location_names_sorted(&self) -> Vec<(&str, i64)> {
        let mut pairs: Vec<_> = self.location_encoding
            .iter()
            .map(|(k, &v)| (k.as_str(), v))
            .collect();
        pairs.sort_by_key(|&(_, v)| v);
        pairs
    }

    /// All registered edge names (sorted by index).
    pub fn edge_names_sorted(&self) -> Vec<(&str, i64)> {
        let mut pairs: Vec<_> = self.edge_encoding
            .iter()
            .map(|(k, &v)| (k.as_str(), v))
            .collect();
        pairs.sort_by_key(|&(_, v)| v);
        pairs
    }

    /// All clock variable PTA names.
    pub fn clock_names(&self) -> Vec<&str> {
        self.clock_map.keys().map(|s| s.as_str()).collect()
    }

    /// All concentration variable PTA names.
    pub fn concentration_names(&self) -> Vec<&str> {
        self.concentration_map.keys().map(|s| s.as_str()).collect()
    }

    /// All state variable PTA names.
    pub fn state_names(&self) -> Vec<&str> {
        self.state_map.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers for building symbol tables from a PTA
// ═══════════════════════════════════════════════════════════════════════════

use crate::pta::PTA;

/// Build a complete symbol table and variable factory from a PTA.
pub fn build_symbol_table_and_factory(pta: &PTA) -> (SymbolTable, VariableFactory) {
    let mut table = SymbolTable::new();
    let mut factory = VariableFactory::new();

    // Register locations
    for (i, loc) in pta.locations.iter().enumerate() {
        table.register_location(&loc.id.0, i as i64);
    }

    // Register edges
    for (i, edge) in pta.edges.iter().enumerate() {
        table.register_edge(&edge.id.0, i as i64);
    }
    // Stutter is one past the last edge
    table.set_stutter_index(pta.edges.len() as i64);

    // Register and create clock variables
    for clock in &pta.clocks {
        let smt_name = format!("clk_{}", clock.name);
        table.register_clock(&clock.name, &smt_name);
        factory.add_clock_var(&smt_name);
    }

    // Register and create concentration variables
    for conc in &pta.concentration_vars {
        let smt_name = format!("conc_{}", conc.drug_name.to_lowercase().replace(' ', "_"));
        table.register_concentration(&conc.name, &smt_name);
        factory.add_concentration_var(&smt_name);
    }

    // Register state variables
    for sv in &pta.state_vars {
        let smt_name = format!("sv_{}", sv.name);
        let sort = match sv.kind {
            crate::pta::StateVariableKind::Bool => SmtSort::Bool,
            crate::pta::StateVariableKind::Int => SmtSort::Int,
            crate::pta::StateVariableKind::Real => SmtSort::Real,
        };
        table.register_state(&sv.name, &smt_name);
        factory.add_state_var(&smt_name, sort);
    }

    (table, factory)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_id_display() {
        let id = VariableId(42);
        assert_eq!(format!("{}", id), "v42");
    }

    #[test]
    fn test_smt_sort_smtlib2() {
        assert_eq!(SmtSort::Bool.to_smtlib2(), "Bool");
        assert_eq!(SmtSort::Int.to_smtlib2(), "Int");
        assert_eq!(SmtSort::Real.to_smtlib2(), "Real");
        assert_eq!(SmtSort::BitVec(32).to_smtlib2(), "(_ BitVec 32)");
        assert_eq!(
            SmtSort::array(SmtSort::Int, SmtSort::Real).to_smtlib2(),
            "(Array Int Real)"
        );
    }

    #[test]
    fn test_smt_sort_properties() {
        assert!(SmtSort::Bool.is_bool());
        assert!(!SmtSort::Bool.is_numeric());
        assert!(SmtSort::Int.is_numeric());
        assert!(SmtSort::Real.is_numeric());
        assert_eq!(SmtSort::BitVec(16).bitvec_width(), Some(16));
    }

    #[test]
    fn test_variable_store_basics() {
        let mut store = VariableStore::new();
        let a = store.create_bool("a");
        let b = store.create_int("b");
        let c = store.create_real("c");

        assert_eq!(store.len(), 3);
        assert_eq!(store.get(a).unwrap().name, "a");
        assert_eq!(store.get(b).unwrap().sort, SmtSort::Int);
        assert!(store.get(c).unwrap().is_numeric());
    }

    #[test]
    fn test_variable_store_dedup() {
        let mut store = VariableStore::new();
        let a1 = store.create_bool("x");
        let a2 = store.create_bool("x");
        assert_eq!(a1, a2);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_variable_store_lookup() {
        let mut store = VariableStore::new();
        store.create_bool("flag");
        store.create_real("conc");

        assert!(store.get_by_name("flag").is_some());
        assert!(store.get_by_name("missing").is_none());
        assert!(store.id_by_name("conc").is_some());
    }

    #[test]
    fn test_time_indexed_variable() {
        let mut store = VariableStore::new();
        let id0 = store.create_time_indexed("x", SmtSort::Real, 0);
        let id1 = store.create_time_indexed("x", SmtSort::Real, 1);
        let id2 = store.create_time_indexed("x", SmtSort::Real, 2);

        assert_ne!(id0, id1);
        assert_ne!(id1, id2);

        let tiv = store.get_time_indexed("x").unwrap();
        assert_eq!(tiv.num_instances(), 3);
        assert_eq!(tiv.at_step(0), Some(id0));
        assert_eq!(tiv.at_step(1), Some(id1));
        assert_eq!(tiv.steps(), vec![0, 1, 2]);
    }

    #[test]
    fn test_variable_declarations() {
        let mut store = VariableStore::new();
        store.create_bool("p");
        store.create_real("x");

        let decls = store.to_declarations();
        assert_eq!(decls.len(), 2);
        assert!(decls[0].contains("Bool"));
        assert!(decls[1].contains("Real"));
    }

    #[test]
    fn test_variable_factory_instantiation() {
        let mut factory = VariableFactory::new();
        factory.add_clock_var("clk_x");
        factory.add_concentration_var("conc_warfarin");

        let mut store = VariableStore::new();
        factory.instantiate(&mut store, 3);

        // loc_t0..loc_t3, clk_x_t0..clk_x_t3, conc_warfarin_t0..conc_warfarin_t3,
        // trans_t0..trans_t2, plus base names
        assert!(store.id_at_step("loc", 0).is_some());
        assert!(store.id_at_step("loc", 3).is_some());
        assert!(store.id_at_step("clk_x", 2).is_some());
        assert!(store.id_at_step("conc_warfarin", 1).is_some());
    }

    #[test]
    fn test_variable_factory_estimation() {
        let mut factory = VariableFactory::new();
        factory.add_clock_var("x");
        factory.add_concentration_var("c");
        // base vars: loc + x + c = 3
        assert_eq!(factory.num_base_vars(), 3);
        // bound=2: 3 vars * 3 steps + 2 trans = 11
        assert_eq!(factory.estimated_total_vars(2), 11);
    }

    #[test]
    fn test_symbol_table_locations() {
        let mut table = SymbolTable::new();
        table.register_location("l0", 0);
        table.register_location("l1", 1);
        table.register_location("l2", 2);

        assert_eq!(table.location_index("l0"), Some(0));
        assert_eq!(table.location_name(1), Some("l1"));
        assert_eq!(table.num_locations(), 3);
    }

    #[test]
    fn test_symbol_table_clocks() {
        let mut table = SymbolTable::new();
        table.register_clock("x", "clk_x");
        assert_eq!(table.clock_smt_name("x"), Some("clk_x"));
        assert_eq!(table.clock_smt_name("y"), None);
    }

    #[test]
    fn test_symbol_table_edges() {
        let mut table = SymbolTable::new();
        table.register_edge("e0", 0);
        table.register_edge("e1", 1);
        table.set_stutter_index(2);

        assert_eq!(table.edge_index("e0"), Some(0));
        assert_eq!(table.edge_name(1), Some("e1"));
        assert_eq!(table.stutter_index(), 2);
    }

    #[test]
    fn test_symbol_table_sorted() {
        let mut table = SymbolTable::new();
        table.register_location("l2", 2);
        table.register_location("l0", 0);
        table.register_location("l1", 1);

        let sorted = table.location_names_sorted();
        assert_eq!(sorted[0].0, "l0");
        assert_eq!(sorted[1].0, "l1");
        assert_eq!(sorted[2].0, "l2");
    }

    #[test]
    fn test_smtvariable_qualified_name() {
        let v = SmtVariable::new(VariableId(0), "x", SmtSort::Real);
        assert_eq!(v.qualified_name(), "x");

        let v2 = v.clone().with_time_step(3);
        assert_eq!(v2.qualified_name(), "x_t3");
    }

    #[test]
    fn test_enum_sort() {
        let s = SmtSort::enum_sort(vec!["A".into(), "B".into(), "C".into()]);
        assert!(s.enum_variants().is_some());
        assert_eq!(s.enum_variants().unwrap().len(), 3);
    }
}
