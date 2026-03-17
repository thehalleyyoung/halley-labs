//! Library-level contract management.
//!
//! A [`ContractLibrary`] collects all contracts for a binary or library and
//! supports dependency-aware composition, whole-library bounds, and
//! pre-built profiles for common cryptographic libraries.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::composition::{self, CompositionError, WholeLibraryBound};
use crate::contract::{ContractStrength, LeakageContract};
use shared_types::FunctionId;

// ---------------------------------------------------------------------------
// Contract summary
// ---------------------------------------------------------------------------

/// Compact summary of a single contract (for listing / indexing).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractSummary {
    /// Function name.
    pub function_name: String,
    /// Function identifier.
    pub function_id: FunctionId,
    /// Worst-case bound in bits.
    pub worst_case_bits: f64,
    /// Whether constant-time.
    pub is_constant_time: bool,
    /// Contract strength.
    pub strength: ContractStrength,
    /// Number of cache sets touched by the transformer.
    pub touched_sets: usize,
}

impl ContractSummary {
    /// Build from a full contract.
    pub fn from_contract(c: &LeakageContract) -> Self {
        Self {
            function_name: c.function_name.clone(),
            function_id: c.function_id.clone(),
            worst_case_bits: c.worst_case_bits(),
            is_constant_time: c.is_constant_time(),
            strength: c.strength,
            touched_sets: c.touched_sets(),
        }
    }
}

impl fmt::Display for ContractSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.2} bits ({}{})",
            self.function_name,
            self.worst_case_bits,
            self.strength.label(),
            if self.is_constant_time {
                ", CT"
            } else {
                ""
            }
        )
    }
}

// ---------------------------------------------------------------------------
// Dependency graph
// ---------------------------------------------------------------------------

/// Call-graph dependencies among functions in a library.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyGraph {
    /// Adjacency list: caller → set of callees.
    pub edges: BTreeMap<String, BTreeSet<String>>,
    /// Reverse edges: callee → set of callers.
    pub reverse_edges: BTreeMap<String, BTreeSet<String>>,
}

impl DependencyGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            edges: BTreeMap::new(),
            reverse_edges: BTreeMap::new(),
        }
    }

    /// Add a call edge.
    pub fn add_edge(&mut self, caller: &str, callee: &str) {
        self.edges
            .entry(caller.to_string())
            .or_default()
            .insert(callee.to_string());
        self.reverse_edges
            .entry(callee.to_string())
            .or_default()
            .insert(caller.to_string());
    }

    /// Return the callees of a function.
    pub fn callees(&self, function: &str) -> Vec<String> {
        self.edges
            .get(function)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Return the callers of a function.
    pub fn callers(&self, function: &str) -> Vec<String> {
        self.reverse_edges
            .get(function)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Topological sort of the call graph (leaves first).
    pub fn topological_order(&self) -> Vec<String> {
        let mut all_nodes: BTreeSet<String> = BTreeSet::new();
        for (k, vs) in &self.edges {
            all_nodes.insert(k.clone());
            for v in vs {
                all_nodes.insert(v.clone());
            }
        }

        let mut in_degree: BTreeMap<String, usize> = BTreeMap::new();
        for node in &all_nodes {
            in_degree.insert(node.clone(), 0);
        }
        for callees in self.edges.values() {
            for callee in callees {
                *in_degree.entry(callee.clone()).or_default() += 1;
            }
        }

        let mut queue: Vec<String> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(n, _)| n.clone())
            .collect();
        queue.sort();

        let mut order = Vec::new();
        while let Some(node) = queue.pop() {
            order.push(node.clone());
            if let Some(callees) = self.edges.get(&node) {
                for callee in callees {
                    if let Some(d) = in_degree.get_mut(callee) {
                        *d = d.saturating_sub(1);
                        if *d == 0 {
                            queue.push(callee.clone());
                            queue.sort();
                        }
                    }
                }
            }
        }

        order
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        let mut all: BTreeSet<&String> = BTreeSet::new();
        for (k, vs) in &self.edges {
            all.insert(k);
            for v in vs {
                all.insert(v);
            }
        }
        all.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|s| s.len()).sum()
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Library bound
// ---------------------------------------------------------------------------

/// A leakage bound for a library entry point computed by composing along
/// the call graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryBound {
    /// Entry-point function name.
    pub entry_point: String,
    /// Worst-case leakage in bits when calling the entry point.
    pub total_bits: f64,
    /// Per-function breakdown along the call chain.
    pub breakdown: BTreeMap<String, f64>,
    /// Whether the bound is provably sound.
    pub is_sound: bool,
    /// Call depth from entry to deepest callee.
    pub call_depth: usize,
}

impl LibraryBound {
    /// Create a trivial bound (single function, no callees).
    pub fn leaf(contract: &LeakageContract) -> Self {
        let mut breakdown = BTreeMap::new();
        breakdown.insert(contract.function_name.clone(), contract.worst_case_bits());
        Self {
            entry_point: contract.function_name.clone(),
            total_bits: contract.worst_case_bits(),
            breakdown,
            is_sound: contract.is_sound(),
            call_depth: 0,
        }
    }
}

impl fmt::Display for LibraryBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.2} bits (depth={}, sound={})",
            self.entry_point, self.total_bits, self.call_depth, self.is_sound
        )
    }
}

// ---------------------------------------------------------------------------
// Crypto library profile
// ---------------------------------------------------------------------------

/// Pre-built contract profile for a well-known cryptographic library.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoLibraryProfile {
    /// Library name (e.g. "OpenSSL 3.1", "libsodium 1.0.19").
    pub name: String,
    /// Library version string.
    pub version: String,
    /// Expected constant-time functions.
    pub expected_ct_functions: Vec<String>,
    /// Known leaky functions and their expected bounds.
    pub known_leaky: BTreeMap<String, f64>,
    /// Notes / caveats.
    pub notes: Vec<String>,
}

impl CryptoLibraryProfile {
    /// Create an empty profile.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            expected_ct_functions: Vec::new(),
            known_leaky: BTreeMap::new(),
            notes: Vec::new(),
        }
    }

    /// Pre-built profile for OpenSSL AES.
    pub fn openssl_aes() -> Self {
        let mut p = Self::new("OpenSSL", "3.x");
        p.expected_ct_functions = vec![
            "aes_encrypt".into(),
            "aes_decrypt".into(),
            "AES_set_encrypt_key".into(),
        ];
        p.notes.push("Assumes AES-NI is available".into());
        p
    }

    /// Check whether a contract matches expectations for a function in
    /// this profile.
    pub fn check_function(&self, contract: &LeakageContract) -> Option<String> {
        if self
            .expected_ct_functions
            .contains(&contract.function_name)
        {
            if !contract.is_constant_time() {
                return Some(format!(
                    "{} expected constant-time but has {:.2} bits leakage",
                    contract.function_name,
                    contract.worst_case_bits()
                ));
            }
        }
        if let Some(&expected) = self.known_leaky.get(&contract.function_name) {
            if contract.worst_case_bits() > expected + 0.01 {
                return Some(format!(
                    "{} leaks {:.2} bits, expected ≤ {:.2}",
                    contract.function_name,
                    contract.worst_case_bits(),
                    expected
                ));
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Contract library
// ---------------------------------------------------------------------------

/// Top-level container for all contracts in a library / binary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractLibrary {
    /// Library name.
    pub name: String,
    /// All contracts keyed by function name.
    pub contracts: BTreeMap<String, LeakageContract>,
    /// Call-graph dependencies.
    pub dependencies: DependencyGraph,
    /// Optional crypto-library profile for validation.
    pub profile: Option<CryptoLibraryProfile>,
}

impl ContractLibrary {
    /// Create an empty library.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            contracts: BTreeMap::new(),
            dependencies: DependencyGraph::new(),
            profile: None,
        }
    }

    /// Add a contract.
    pub fn add_contract(&mut self, contract: LeakageContract) {
        self.contracts
            .insert(contract.function_name.clone(), contract);
    }

    /// Look up a contract by function name.
    pub fn get(&self, name: &str) -> Option<&LeakageContract> {
        self.contracts.get(name)
    }

    /// Number of contracts.
    pub fn len(&self) -> usize {
        self.contracts.len()
    }

    /// Whether the library is empty.
    pub fn is_empty(&self) -> bool {
        self.contracts.is_empty()
    }

    /// Get summaries for all contracts.
    pub fn summaries(&self) -> Vec<ContractSummary> {
        self.contracts
            .values()
            .map(ContractSummary::from_contract)
            .collect()
    }

    /// Compute a whole-library bound by composing all contracts
    /// sequentially along the call graph from a given entry point.
    pub fn compute_bound(
        &self,
        entry_point: &str,
    ) -> Result<LibraryBound, CompositionError> {
        let contract = self.contracts.get(entry_point).ok_or_else(|| {
            CompositionError::PreconditionMismatch {
                reason: format!("entry point '{}' not found", entry_point),
            }
        })?;

        let callees = self.dependencies.callees(entry_point);
        if callees.is_empty() {
            return Ok(LibraryBound::leaf(contract));
        }

        let mut total = contract.worst_case_bits();
        let mut breakdown = BTreeMap::new();
        breakdown.insert(entry_point.to_string(), contract.worst_case_bits());
        let mut max_depth = 0;
        let mut is_sound = contract.is_sound();

        for callee_name in &callees {
            if let Some(callee_contract) = self.contracts.get(callee_name) {
                let callee_bound = self.compute_bound(callee_name)?;
                total += callee_bound.total_bits;
                breakdown.extend(callee_bound.breakdown);
                max_depth = max_depth.max(callee_bound.call_depth + 1);
                is_sound = is_sound && callee_bound.is_sound;
            }
        }

        Ok(LibraryBound {
            entry_point: entry_point.to_string(),
            total_bits: total,
            breakdown,
            is_sound,
            call_depth: max_depth,
        })
    }

    /// Compute the whole-library bound (sum of all function bounds).
    pub fn whole_library_bound(&self) -> WholeLibraryBound {
        let mut bound = WholeLibraryBound::zero();
        bound.strategy = "sum-all".into();
        for (name, c) in &self.contracts {
            bound.add_contribution(name, c.worst_case_bits());
            if !c.is_sound() {
                bound.is_sound = false;
            }
        }
        bound
    }

    /// Set a crypto library profile.
    pub fn with_profile(mut self, profile: CryptoLibraryProfile) -> Self {
        self.profile = Some(profile);
        self
    }
}

impl fmt::Display for ContractLibrary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ContractLibrary '{}' ({} contracts)", self.name, self.len())?;
        for summary in self.summaries() {
            writeln!(f, "  {}", summary)?;
        }
        Ok(())
    }
}
