//! Abstract program states mapping program locations to abstract values.
//!
//! [`AbstractState`] models the abstract value of a single program point
//! (registers, flags, memory), while [`AbstractEnvironment`] maps every
//! program point in a function to its corresponding abstract state.

use std::collections::BTreeMap;

use serde::{Serialize, Deserialize};
use indexmap::IndexMap;

use shared_types::{Interval, SecurityLevel};

// ---------------------------------------------------------------------------
// AbstractState
// ---------------------------------------------------------------------------

/// An abstract state at a single program point.
///
/// Tracks abstract values for registers, memory locations, and processor
/// flags.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbstractState {
    /// Abstract values for general-purpose registers, keyed by register id.
    pub registers: BTreeMap<u32, Interval>,
    /// Abstract values for memory locations, keyed by virtual address.
    pub memory: BTreeMap<u64, Interval>,
    /// Security labels for tracked locations (registers + memory).
    pub labels: BTreeMap<String, SecurityLevel>,
    /// Whether this state is reachable (`false` ⇒ bottom).
    pub reachable: bool,
}

impl AbstractState {
    /// Create a new unreachable (bottom) state.
    pub fn unreachable() -> Self {
        Self {
            registers: BTreeMap::new(),
            memory: BTreeMap::new(),
            labels: BTreeMap::new(),
            reachable: false,
        }
    }

    /// Create a new reachable state with all registers and memory set to ⊤.
    pub fn top_state() -> Self {
        Self {
            registers: BTreeMap::new(),
            memory: BTreeMap::new(),
            labels: BTreeMap::new(),
            reachable: true,
        }
    }

    /// Look up the abstract interval for a register.
    pub fn get_register(&self, reg: u32) -> Option<&Interval> {
        self.registers.get(&reg)
    }

    /// Set the abstract interval for a register.
    pub fn set_register(&mut self, reg: u32, value: Interval) {
        self.registers.insert(reg, value);
    }

    /// Look up the abstract interval for a memory address.
    pub fn get_memory(&self, addr: u64) -> Option<&Interval> {
        self.memory.get(&addr)
    }

    /// Set the abstract interval for a memory address.
    pub fn set_memory(&mut self, addr: u64, value: Interval) {
        self.memory.insert(addr, value);
    }

    /// Assign a security label to a named location.
    pub fn set_label(&mut self, location: impl Into<String>, level: SecurityLevel) {
        self.labels.insert(location.into(), level);
    }
}

impl Default for AbstractState {
    fn default() -> Self {
        Self::unreachable()
    }
}

// ---------------------------------------------------------------------------
// AbstractEnvironment
// ---------------------------------------------------------------------------

/// Maps every program point (identified by virtual address) in a function
/// to its [`AbstractState`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbstractEnvironment {
    /// Per-address abstract states.
    pub states: IndexMap<u64, AbstractState>,
    /// Human-readable label for this environment (e.g., function name).
    pub label: String,
}

impl AbstractEnvironment {
    /// Create a new empty environment.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            states: IndexMap::new(),
            label: label.into(),
        }
    }

    /// Insert or overwrite the abstract state at the given address.
    pub fn set(&mut self, address: u64, state: AbstractState) {
        self.states.insert(address, state);
    }

    /// Retrieve the abstract state at the given address.
    pub fn get(&self, address: u64) -> Option<&AbstractState> {
        self.states.get(&address)
    }

    /// Retrieve a mutable reference to the abstract state at the given address.
    pub fn get_mut(&mut self, address: u64) -> Option<&mut AbstractState> {
        self.states.get_mut(&address)
    }

    /// Return the number of tracked program points.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Returns `true` if no program points are tracked.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Join another environment into this one point-wise.
    pub fn join_with(&mut self, other: &AbstractEnvironment) {
        for (&addr, other_state) in &other.states {
            match self.states.get(&addr) {
                Some(_existing) => {
                    // In a full implementation this would perform a lattice
                    // join of the two abstract states.
                }
                None => {
                    self.states.insert(addr, other_state.clone());
                }
            }
        }
    }
}

impl Default for AbstractEnvironment {
    fn default() -> Self {
        Self::new("")
    }
}
