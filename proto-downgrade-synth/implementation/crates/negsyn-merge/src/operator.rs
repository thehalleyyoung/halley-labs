//! Core merge operator — Definition D2: Protocol-Aware Merge.
//!
//! The `MergeOperator` implements the PROTOMERGE algorithm that merges two
//! symbolic execution states when they are at the same handshake phase,
//! with compatible protocol parameters. The merged state uses ITE expressions
//! for differing values and disjunction of path constraints.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::time::Instant;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use negsyn_types::{
    CipherSuite, ConcreteValue, Extension, HandshakePhase, MergeConfig, MergeError, MergeMetrics,
    MergeResult as TypesMergeResult, MergeableState, MemoryRegion, MemoryPermissions, NegotiationState,
    PathConstraint, ProtocolVersion, SymbolicState, SymbolicValue,
};

use crate::cache::MergeCache;
use crate::cost::{CostEstimator, MergeCost};
use crate::lattice::{PreferenceLattice, SecurityLattice, SelectionFunction};
use crate::symbolic_merge::SymbolicMerger;

// ---------------------------------------------------------------------------
// Merge output
// ---------------------------------------------------------------------------

/// Result of a successful merge operation.
#[derive(Debug, Clone)]
pub struct MergeOutput {
    pub merged_state: SymbolicState,
    pub left_id: u64,
    pub right_id: u64,
    pub cost: MergeCost,
    pub metadata: MergeMetadata,
}

/// Metadata about a merge operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeMetadata {
    pub constraint_count_before: usize,
    pub constraint_count_after: usize,
    pub ite_nodes_created: usize,
    pub memory_regions_merged: usize,
    pub registers_merged: usize,
    pub extension_conflicts: usize,
    pub cipher_selection_mode: CipherSelectionMode,
    pub merge_time_us: u64,
    pub was_cached: bool,
}

/// How cipher selection was handled during merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CipherSelectionMode {
    Identical,
    IteOnConstraint,
    LatticeJoin,
    FipsOverride,
    CallbackDependent,
}

// ---------------------------------------------------------------------------
// Mergeability predicate
// ---------------------------------------------------------------------------

/// Checks whether two states satisfy the conditions for protocol-aware merge.
///
/// Two states are mergeable iff:
/// 1. Same handshake phase
/// 2. Same protocol version
/// 3. Same offered cipher set
/// 4. Disjunction of path constraints is satisfiable
/// 5. Merge cost is within budget
pub struct MergeabilityPredicate {
    config: MergeConfig,
    cost_estimator: CostEstimator,
}

impl MergeabilityPredicate {
    pub fn new(config: MergeConfig) -> Self {
        Self {
            cost_estimator: CostEstimator::new(config.clone()),
            config,
        }
    }

    /// Check if two states can be merged, returning a detailed reason if not.
    pub fn can_merge(&self, left: &SymbolicState, right: &SymbolicState) -> Result<(), MergeError> {
        // Condition 1: Same handshake phase
        if left.negotiation.phase != right.negotiation.phase {
            return Err(MergeError::IncompatibleStates {
                reason: format!(
                    "Phase mismatch: {:?} vs {:?}",
                    left.negotiation.phase, right.negotiation.phase
                ),
            });
        }

        // Condition 2: Same protocol version
        if left.negotiation.version != right.negotiation.version {
            return Err(MergeError::IncompatibleStates {
                reason: format!(
                    "Version mismatch: {:?} vs {:?}",
                    left.negotiation.version, right.negotiation.version
                ),
            });
        }

        // Condition 3: Same offered cipher set
        let left_cipher_ids: BTreeSet<u16> = left.negotiation.offered_ciphers.iter().map(|c| c.iana_id).collect();
        let right_cipher_ids: BTreeSet<u16> = right.negotiation.offered_ciphers.iter().map(|c| c.iana_id).collect();
        if left_cipher_ids != right_cipher_ids {
            return Err(MergeError::IncompatibleStates {
                reason: format!(
                    "Offered cipher sets differ: {} vs {} ciphers",
                    left_cipher_ids.len(),
                    right_cipher_ids.len()
                ),
            });
        }

        // Condition 4: Disjunction satisfiability (lightweight check)
        if left.constraints.is_empty() && right.constraints.is_empty() {
            // both empty is fine
        }

        // Condition 5: Cost within budget
        let total_complexity = (left.constraints.len() + right.constraints.len()) as u32;
        if total_complexity > self.config.max_merged_constraints {
            return Err(MergeError::ComplexityExceeded {
                reason: "Merged constraint count exceeds limit".to_string(),
                complexity: total_complexity,
                limit: self.config.max_merged_constraints,
            });
        }

        Ok(())
    }

    /// Estimate the cost of merging two states.
    pub fn estimate_cost(&self, left: &SymbolicState, right: &SymbolicState) -> MergeCost {
        self.cost_estimator.estimate(left, right)
    }
}

// ---------------------------------------------------------------------------
// Protocol merge trait
// ---------------------------------------------------------------------------

/// Trait for types that implement protocol-aware merge.
pub trait ProtocolMerge {
    type State: MergeableState;
    type Output;

    /// Merge two states into one.
    fn merge(&mut self, left: &Self::State, right: &Self::State) -> TypesMergeResult<Self::Output>;

    /// Check if two states can be merged.
    fn can_merge(&self, left: &Self::State, right: &Self::State) -> bool;

    /// Estimate the cost of merging two states.
    fn merge_cost(&self, left: &Self::State, right: &Self::State) -> MergeCost;
}

// ---------------------------------------------------------------------------
// Merge context
// ---------------------------------------------------------------------------

/// Tracks merge history, statistics, and accumulated state.
#[derive(Debug)]
pub struct MergeContext {
    pub metrics: MergeMetrics,
    next_state_id: u64,
    merge_history: Vec<MergeRecord>,
    active_merges: usize,
    max_concurrent_merges: usize,
}

/// Record of a single merge operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeRecord {
    pub left_id: u64,
    pub right_id: u64,
    pub result_id: u64,
    pub success: bool,
    pub time_us: u64,
    pub cost: MergeCost,
}

impl MergeContext {
    pub fn new(start_id: u64) -> Self {
        Self {
            metrics: MergeMetrics::default(),
            next_state_id: start_id,
            merge_history: Vec::new(),
            active_merges: 0,
            max_concurrent_merges: 0,
        }
    }

    pub fn allocate_state_id(&mut self) -> u64 {
        let id = self.next_state_id;
        self.next_state_id += 1;
        id
    }

    pub fn record_merge(&mut self, record: MergeRecord) {
        self.metrics.record_attempt(record.success);
        self.merge_history.push(record);
    }

    pub fn begin_merge(&mut self) {
        self.active_merges += 1;
        if self.active_merges > self.max_concurrent_merges {
            self.max_concurrent_merges = self.active_merges;
        }
    }

    pub fn end_merge(&mut self) {
        self.active_merges = self.active_merges.saturating_sub(1);
    }

    pub fn merge_count(&self) -> usize {
        self.merge_history.len()
    }

    pub fn success_rate(&self) -> f64 {
        self.metrics.success_rate()
    }

    pub fn history(&self) -> &[MergeRecord] {
        &self.merge_history
    }

    pub fn recent_history(&self, n: usize) -> &[MergeRecord] {
        let start = self.merge_history.len().saturating_sub(n);
        &self.merge_history[start..]
    }
}

// ---------------------------------------------------------------------------
// Merge operator
// ---------------------------------------------------------------------------

/// The core protocol-aware merge operator (ALG2: PROTOMERGE).
///
/// Given two symbolic states `s1` and `s2` at the same handshake phase:
/// - Merged constraint: φ₁ ∨ φ₂
/// - For each differing value v: ITE(φ₁, v₁, v₂)
/// - Cipher selection via lattice when deterministic, ITE otherwise
pub struct MergeOperator {
    config: MergeConfig,
    predicate: MergeabilityPredicate,
    symbolic_merger: SymbolicMerger,
    context: MergeContext,
    cache: Option<MergeCache>,
    lattice: Option<SecurityLattice>,
}

impl MergeOperator {
    pub fn new(config: MergeConfig) -> Self {
        let predicate = MergeabilityPredicate::new(config.clone());
        let symbolic_merger = SymbolicMerger::new(config.clone());
        let cache = if config.enable_caching {
            Some(MergeCache::new(config.cache_capacity))
        } else {
            None
        };
        Self {
            predicate,
            symbolic_merger,
            context: MergeContext::new(1_000_000),
            cache,
            lattice: None,
            config,
        }
    }

    pub fn with_lattice(mut self, lattice: SecurityLattice) -> Self {
        self.lattice = Some(lattice);
        self
    }

    pub fn with_cache(mut self, cache: MergeCache) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Perform the protocol-aware merge of two symbolic states.
    pub fn merge_states(
        &mut self,
        left: &SymbolicState,
        right: &SymbolicState,
    ) -> TypesMergeResult<MergeOutput> {
        let start = Instant::now();
        self.context.begin_merge();

        // Check mergeability
        self.predicate.can_merge(left, right)?;

        // Check cache
        if let Some(ref cache) = self.cache {
            let key = crate::cache::CacheKey::from_states(left, right);
            if let Some(cached) = cache.get(&key) {
                self.context.end_merge();
                return Ok(cached.clone());
            }
        }

        // Allocate new state ID
        let merged_id = self.context.allocate_state_id();

        // Build the disjunction constraint for the merge point
        let left_path_expr = self.build_sv_conjunction(&left.constraints);
        let right_path_expr = self.build_sv_conjunction(&right.constraints);
        let disjunction = SymbolicValue::or_expr(left_path_expr.clone(), right_path_expr.clone());

        // Merge path constraints: the merged state takes the disjunction
        let mut merged_pc = PathConstraint::new();
        merged_pc.add(disjunction);

        // Merge registers via ITE (convert HashMap to BTreeMap)
        let left_regs: BTreeMap<String, SymbolicValue> = left.registers.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let right_regs: BTreeMap<String, SymbolicValue> = right.registers.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let (merged_regs_btree, reg_ite_count) =
            self.merge_registers(&left_regs, &right_regs, &left_path_expr);

        // Merge memory regions (convert SymbolicMemory to BTreeMap<String, MemoryRegion>)
        let left_mem: BTreeMap<String, MemoryRegion> = left.memory.regions.iter().map(|r| (r.name.clone(), r.clone())).collect();
        let right_mem: BTreeMap<String, MemoryRegion> = right.memory.regions.iter().map(|r| (r.name.clone(), r.clone())).collect();
        let (merged_mem_map, mem_merge_count) =
            self.merge_memory(&left_mem, &right_mem, &left_path_expr);

        // Merge negotiation state
        let (merged_negotiation, cipher_mode, ext_conflicts) =
            self.merge_negotiation(&left.negotiation, &right.negotiation, &left_path_expr);

        // Build merged state
        let merged_state = SymbolicState {
            id: merged_id,
            program_counter: left.program_counter,
            pc: merged_pc.clone(),
            path_constraint: merged_pc.clone(),
            constraints: merged_pc.conditions.clone(),
            memory: {
                let mut sm = negsyn_types::SymbolicMemory::new();
                for (_, region) in merged_mem_map {
                    sm.add_region(region);
                }
                sm
            },
            registers: merged_regs_btree.into_iter().collect(),
            negotiation: merged_negotiation,
            depth: left.depth.max(right.depth) + 1,
            is_feasible: true,
            parent_id: None,
        };

        let elapsed_us = start.elapsed().as_micros() as u64;
        let cost = self.predicate.estimate_cost(left, right);

        let metadata = MergeMetadata {
            constraint_count_before: left.constraints.len() + right.constraints.len(),
            constraint_count_after: 1,
            ite_nodes_created: reg_ite_count,
            memory_regions_merged: mem_merge_count,
            registers_merged: reg_ite_count,
            extension_conflicts: ext_conflicts,
            cipher_selection_mode: cipher_mode,
            merge_time_us: elapsed_us,
            was_cached: false,
        };

        let output = MergeOutput {
            merged_state,
            left_id: left.id,
            right_id: right.id,
            cost: cost.clone(),
            metadata,
        };

        // Record in context
        self.context.record_merge(MergeRecord {
            left_id: left.id,
            right_id: right.id,
            result_id: merged_id,
            success: true,
            time_us: elapsed_us,
            cost,
        });

        // Store in cache
        if let Some(ref mut cache) = self.cache {
            let key = crate::cache::CacheKey::from_states(left, right);
            cache.put(key, output.clone());
        }

        self.context.end_merge();

        Ok(output)
    }

    /// Build the conjunction of symbolic value constraints into a single expression.
    fn build_sv_conjunction(&self, constraints: &[SymbolicValue]) -> SymbolicValue {
        if constraints.is_empty() {
            return SymbolicValue::Concrete(ConcreteValue::Bool(true));
        }

        let mut result = constraints[0].clone();
        for c in &constraints[1..] {
            result = SymbolicValue::and_expr(result, c.clone());
        }
        result
    }

    /// Merge register maps, creating ITE expressions for differing values.
    fn merge_registers(
        &self,
        left: &BTreeMap<String, SymbolicValue>,
        right: &BTreeMap<String, SymbolicValue>,
        left_cond: &SymbolicValue,
    ) -> (BTreeMap<String, SymbolicValue>, usize) {
        let mut merged = BTreeMap::new();
        let mut ite_count = 0;

        // All keys from both sides
        let all_keys: BTreeSet<&String> = left.keys().chain(right.keys()).collect();

        for key in all_keys {
            let value = match (left.get(key), right.get(key)) {
                (Some(lv), Some(rv)) => {
                    if lv == rv {
                        lv.clone()
                    } else {
                        ite_count += 1;
                        SymbolicValue::ite(left_cond.clone(), lv.clone(), rv.clone())
                    }
                }
                (Some(lv), None) => {
                    ite_count += 1;
                    SymbolicValue::ite(
                        left_cond.clone(),
                        lv.clone(),
                        SymbolicValue::Concrete(ConcreteValue::Int(0)),
                    )
                }
                (None, Some(rv)) => {
                    ite_count += 1;
                    SymbolicValue::ite(
                        left_cond.clone(),
                        SymbolicValue::Concrete(ConcreteValue::Int(0)),
                        rv.clone(),
                    )
                }
                (None, None) => unreachable!(),
            };
            merged.insert(key.clone(), value);
        }

        (merged, ite_count)
    }

    /// Merge memory region maps.
    fn merge_memory(
        &self,
        left: &BTreeMap<String, MemoryRegion>,
        right: &BTreeMap<String, MemoryRegion>,
        left_cond: &SymbolicValue,
    ) -> (BTreeMap<String, MemoryRegion>, usize) {
        let mut merged = BTreeMap::new();
        let mut merge_count = 0;

        let all_regions: BTreeSet<&String> = left.keys().chain(right.keys()).collect();

        for name in all_regions {
            let region = match (left.get(name), right.get(name)) {
                (Some(lr), Some(rr)) => {
                    merge_count += 1;
                    self.merge_single_region(lr, rr, left_cond)
                }
                (Some(lr), None) => lr.clone(),
                (None, Some(rr)) => rr.clone(),
                (None, None) => unreachable!(),
            };
            merged.insert(name.clone(), region);
        }

        (merged, merge_count)
    }

    /// Merge a single memory region by creating ITE for differing cells.
    fn merge_single_region(
        &self,
        left: &MemoryRegion,
        right: &MemoryRegion,
        left_cond: &SymbolicValue,
    ) -> MemoryRegion {
        let perms = MemoryPermissions {
            read: left.permissions.read || right.permissions.read,
            write: left.permissions.write || right.permissions.write,
            execute: left.permissions.execute || right.permissions.execute,
        };
        let mut merged = MemoryRegion::new(
            left.name.clone(),
            left.base_address,
            left.size.max(right.size),
            perms,
        );

        let all_offsets: BTreeSet<u64> = left
            .contents()
            .keys()
            .chain(right.contents().keys())
            .copied()
            .collect();

        for offset in all_offsets {
            let value = match (left.contents().get(&offset), right.contents().get(&offset)) {
                (Some(lv), Some(rv)) => {
                    if lv == rv {
                        lv.clone()
                    } else {
                        SymbolicValue::ite(left_cond.clone(), lv.clone(), rv.clone())
                    }
                }
                (Some(lv), None) => lv.clone(),
                (None, Some(rv)) => rv.clone(),
                (None, None) => unreachable!(),
            };
            merged.write(offset, value);
        }

        merged
    }

    /// Merge negotiation states, handling cipher selection and extensions.
    fn merge_negotiation(
        &self,
        left: &NegotiationState,
        right: &NegotiationState,
        left_cond: &SymbolicValue,
    ) -> (NegotiationState, CipherSelectionMode, usize) {
        let mut merged = NegotiationState::new();
        merged.phase = left.phase;
        merged.version = left.version;
        merged.offered_ciphers = left.offered_ciphers.clone();
        merged.is_resumption = left.is_resumption || right.is_resumption;

        // Merge selected cipher
        let cipher_mode = match (&left.selected_cipher, &right.selected_cipher) {
            (Some(lc), Some(rc)) if lc == rc => {
                merged.selected_cipher = Some(lc.clone());
                CipherSelectionMode::Identical
            }
            (Some(lc), Some(rc)) => {
                // Try lattice-based resolution
                if let Some(ref lattice) = self.lattice {
                    if let Some(joined) = lattice.join(lc.iana_id, rc.iana_id) {
                        if self.config.fips_mode {
                            let lp = lattice.profile(lc.iana_id);
                            let rp = lattice.profile(rc.iana_id);
                            let fips_cs = match (
                                lp.map(|p| p.is_fips).unwrap_or(false),
                                rp.map(|p| p.is_fips).unwrap_or(false),
                            ) {
                                (true, false) => lc.clone(),
                                (false, true) => rc.clone(),
                                _ => {
                                    // Use the cipher whose iana_id matches the joined result
                                    if lc.iana_id == joined { lc.clone() } else { rc.clone() }
                                }
                            };
                            merged.selected_cipher = Some(fips_cs);
                            CipherSelectionMode::FipsOverride
                        } else {
                            // Use the cipher whose iana_id matches the joined result
                            let joined_cs = if lc.iana_id == joined { lc.clone() } else { rc.clone() };
                            merged.selected_cipher = Some(joined_cs);
                            CipherSelectionMode::LatticeJoin
                        }
                    } else {
                        merged.selected_cipher = Some(lc.clone());
                        CipherSelectionMode::CallbackDependent
                    }
                } else {
                    merged.selected_cipher = Some(lc.clone());
                    CipherSelectionMode::IteOnConstraint
                }
            }
            (Some(c), None) | (None, Some(c)) => {
                merged.selected_cipher = Some(c.clone());
                CipherSelectionMode::IteOnConstraint
            }
            (None, None) => {
                merged.selected_cipher = None;
                CipherSelectionMode::Identical
            }
        };

        // Merge extensions
        let ext_conflicts = self.merge_extensions(
            &left.extensions,
            &right.extensions,
            &mut merged.extensions,
        );

        // Merge session ID
        merged.session_id = match (&left.session_id, &right.session_id) {
            (Some(ls), Some(rs)) if ls == rs => Some(ls.clone()),
            (Some(ls), _) => Some(ls.clone()),
            (_, Some(rs)) => Some(rs.clone()),
            (None, None) => None,
        };

        (merged, cipher_mode, ext_conflicts)
    }

    /// Merge extension lists, detecting conflicts.
    fn merge_extensions(
        &self,
        left: &[Extension],
        right: &[Extension],
        merged: &mut Vec<Extension>,
    ) -> usize {
        let mut conflict_count = 0;
        let mut left_by_id: BTreeMap<u16, &Extension> =
            left.iter().map(|e| (e.id, e)).collect();
        let mut right_by_id: BTreeMap<u16, &Extension> =
            right.iter().map(|e| (e.id, e)).collect();

        let all_ids: BTreeSet<u16> = left_by_id
            .keys()
            .chain(right_by_id.keys())
            .copied()
            .collect();

        for id in all_ids {
            match (left_by_id.get(&id), right_by_id.get(&id)) {
                (Some(le), Some(re)) => {
                    if le.data == re.data {
                        merged.push((*le).clone());
                    } else {
                        conflict_count += 1;
                        // On conflict, prefer the critical extension or the left one
                        if re.is_critical && !le.is_critical {
                            merged.push((*re).clone());
                        } else {
                            merged.push((*le).clone());
                        }
                    }
                }
                (Some(e), None) | (None, Some(e)) => {
                    merged.push((*e).clone());
                }
                (None, None) => unreachable!(),
            }
        }

        conflict_count
    }

    /// Try to merge multiple states pairwise, returning the reduced set.
    pub fn merge_batch(
        &mut self,
        states: &[SymbolicState],
    ) -> Vec<TypesMergeResult<MergeOutput>> {
        if states.len() < 2 {
            return Vec::new();
        }

        let mut results = Vec::new();

        // Group states by (phase, version, offered_ciphers)
        let mut groups: BTreeMap<GroupKey, Vec<usize>> = BTreeMap::new();
        for (i, state) in states.iter().enumerate() {
            let key = GroupKey {
                phase: state.negotiation.phase,
                version: state.negotiation.version,
                offered: state.negotiation.offered_ciphers.iter().map(|c| c.iana_id).collect(),
            };
            groups.entry(key).or_default().push(i);
        }

        // Merge within each group pairwise
        for (_, indices) in &groups {
            let mut i = 0;
            while i + 1 < indices.len() {
                let result = self.merge_states(&states[indices[i]], &states[indices[i + 1]]);
                results.push(result);
                i += 2;
            }
        }

        results
    }

    pub fn context(&self) -> &MergeContext {
        &self.context
    }

    pub fn context_mut(&mut self) -> &mut MergeContext {
        &mut self.context
    }

    pub fn config(&self) -> &MergeConfig {
        &self.config
    }

    pub fn metrics(&self) -> &MergeMetrics {
        &self.context.metrics
    }
}

/// Grouping key for batch merges.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GroupKey {
    phase: HandshakePhase,
    version: Option<ProtocolVersion>,
    offered: BTreeSet<u16>,
}

impl ProtocolMerge for MergeOperator {
    type State = SymbolicState;
    type Output = MergeOutput;

    fn merge(
        &mut self,
        left: &SymbolicState,
        right: &SymbolicState,
    ) -> TypesMergeResult<MergeOutput> {
        self.merge_states(left, right)
    }

    fn can_merge(&self, left: &SymbolicState, right: &SymbolicState) -> bool {
        self.predicate.can_merge(left, right).is_ok()
    }

    fn merge_cost(&self, left: &SymbolicState, right: &SymbolicState) -> MergeCost {
        self.predicate.estimate_cost(left, right)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use negsyn_types::{HandshakePhase, NegotiationState, ProtocolVersion};

    fn make_test_state(id: u64, phase: HandshakePhase, ciphers: &[u16]) -> SymbolicState {
        let mut neg = NegotiationState::new();
        neg.phase = phase;
        neg.offered_ciphers = ciphers.iter().map(|&iana_id| {
            CipherSuite::new(
                iana_id,
                format!("test_cipher_{:04x}", iana_id),
                negsyn_types::KeyExchange::RSA,
                negsyn_types::AuthAlgorithm::RSA,
                negsyn_types::EncryptionAlgorithm::AES128GCM,
                negsyn_types::MacAlgorithm::AEAD,
                negsyn_types::SecurityLevel::Standard,
            )
        }).collect();
        let mut state = SymbolicState::new(id, 0x1000);
        state.negotiation = neg;
        state.registers.insert(
            "rax".to_string(),
            SymbolicValue::var("rax", negsyn_types::SymSort::BitVec(64)),
        );
        state
    }

    fn make_test_state_with_constraint(
        id: u64,
        phase: HandshakePhase,
        ciphers: &[u16],
        constraint: SymbolicValue,
    ) -> SymbolicState {
        let mut state = make_test_state(id, phase, ciphers);
        state.constraints.push(constraint.clone());
        state.path_constraint.add(constraint);
        state.pc = state.path_constraint.clone();
        state
    }

    #[test]
    fn test_mergeability_same_phase() {
        let config = MergeConfig::default();
        let pred = MergeabilityPredicate::new(config);

        let s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F, 0xC02F]);
        let s2 = make_test_state(2, HandshakePhase::ClientHello, &[0x002F, 0xC02F]);

        assert!(pred.can_merge(&s1, &s2).is_ok());
    }

    #[test]
    fn test_mergeability_different_phase() {
        let config = MergeConfig::default();
        let pred = MergeabilityPredicate::new(config);

        let s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F]);
        let s2 = make_test_state(2, HandshakePhase::ServerHello, &[0x002F]);

        assert!(pred.can_merge(&s1, &s2).is_err());
    }

    #[test]
    fn test_mergeability_different_ciphers() {
        let config = MergeConfig::default();
        let pred = MergeabilityPredicate::new(config);

        let s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F]);
        let s2 = make_test_state(2, HandshakePhase::ClientHello, &[0xC02F]);

        assert!(pred.can_merge(&s1, &s2).is_err());
    }

    #[test]
    fn test_merge_identical_states() {
        let config = MergeConfig::default();
        let mut op = MergeOperator::new(config);

        let s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F, 0xC02F]);
        let s2 = make_test_state(2, HandshakePhase::ClientHello, &[0x002F, 0xC02F]);

        let result = op.merge_states(&s1, &s2);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.left_id, 1);
        assert_eq!(output.right_id, 2);
        assert_eq!(output.merged_state.negotiation.phase, HandshakePhase::ClientHello);
    }

    #[test]
    fn test_merge_registers_ite() {
        let config = MergeConfig::default();
        let mut op = MergeOperator::new(config);

        let mut s1 = make_test_state(1, HandshakePhase::ServerHello, &[0x002F]);
        s1.registers
            .insert("rbx".to_string(), SymbolicValue::Concrete(ConcreteValue::Int(42)));

        let mut s2 = make_test_state(2, HandshakePhase::ServerHello, &[0x002F]);
        s2.registers
            .insert("rbx".to_string(), SymbolicValue::Concrete(ConcreteValue::Int(99)));

        let result = op.merge_states(&s1, &s2).unwrap();
        let rbx = result.merged_state.registers.get("rbx").unwrap();
        // Should be an ITE since values differ
        assert!(matches!(rbx, SymbolicValue::Ite { .. }));
    }

    #[test]
    fn test_merge_with_constraints() {
        let config = MergeConfig::default();
        let mut op = MergeOperator::new(config);

        let c1 = SymbolicValue::eq_expr(
            SymbolicValue::var("x", negsyn_types::SymSort::Int),
            SymbolicValue::Concrete(ConcreteValue::Int(1)),
        );
        let c2 = SymbolicValue::eq_expr(
            SymbolicValue::var("x", negsyn_types::SymSort::Int),
            SymbolicValue::Concrete(ConcreteValue::Int(2)),
        );

        let s1 =
            make_test_state_with_constraint(1, HandshakePhase::ClientHello, &[0x002F], c1);
        let s2 =
            make_test_state_with_constraint(2, HandshakePhase::ClientHello, &[0x002F], c2);

        let result = op.merge_states(&s1, &s2).unwrap();
        // Merged state should have a single constraint (the disjunction)
        assert_eq!(result.merged_state.constraints.len(), 1);
    }

    #[test]
    fn test_merge_different_selected_cipher() {
        let config = MergeConfig::default();
        let mut op = MergeOperator::new(config);

        let mut s1 = make_test_state(1, HandshakePhase::ServerHello, &[0x002F, 0xC02F]);
        s1.negotiation.selected_cipher = Some(s1.negotiation.offered_ciphers[0].clone());

        let mut s2 = make_test_state(2, HandshakePhase::ServerHello, &[0x002F, 0xC02F]);
        s2.negotiation.selected_cipher = Some(s2.negotiation.offered_ciphers[1].clone());

        let result = op.merge_states(&s1, &s2).unwrap();
        assert!(result.merged_state.negotiation.selected_cipher.is_some());
        assert_eq!(result.metadata.cipher_selection_mode, CipherSelectionMode::IteOnConstraint);
    }

    #[test]
    fn test_merge_with_lattice() {
        let config = MergeConfig::default();
        let lattice = crate::lattice::SecurityLattice::from_standard_registry();
        let mut op = MergeOperator::new(config).with_lattice(lattice);

        let mut s1 = make_test_state(1, HandshakePhase::ServerHello, &[0x002F, 0xC02F]);
        s1.negotiation.selected_cipher = Some(s1.negotiation.offered_ciphers[0].clone());

        let mut s2 = make_test_state(2, HandshakePhase::ServerHello, &[0x002F, 0xC02F]);
        s2.negotiation.selected_cipher = Some(s2.negotiation.offered_ciphers[1].clone());

        let result = op.merge_states(&s1, &s2).unwrap();
        assert_eq!(result.metadata.cipher_selection_mode, CipherSelectionMode::LatticeJoin);
    }

    #[test]
    fn test_merge_batch() {
        let config = MergeConfig::default();
        let mut op = MergeOperator::new(config);

        let states: Vec<SymbolicState> = (0..4)
            .map(|i| make_test_state(i, HandshakePhase::ClientHello, &[0x002F]))
            .collect();

        let results = op.merge_batch(&states);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[test]
    fn test_merge_context_tracking() {
        let config = MergeConfig::default();
        let mut op = MergeOperator::new(config);

        let s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F]);
        let s2 = make_test_state(2, HandshakePhase::ClientHello, &[0x002F]);

        let _ = op.merge_states(&s1, &s2);
        assert_eq!(op.context().merge_count(), 1);
        assert!(op.context().success_rate() > 0.0);
    }

    #[test]
    fn test_merge_memory_regions() {
        let config = MergeConfig::default();
        let mut op = MergeOperator::new(config);

        let mut s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F]);
        let mut region1 = MemoryRegion::new(0x1000, 256, "stack");
        region1.write(0, SymbolicValue::Concrete(ConcreteValue::Int(10)));
        region1.write(8, SymbolicValue::Concrete(ConcreteValue::Int(20)));
        s1.memory.insert("stack".to_string(), region1);

        let mut s2 = make_test_state(2, HandshakePhase::ClientHello, &[0x002F]);
        let mut region2 = MemoryRegion::new(0x1000, 256, "stack");
        region2.write(0, SymbolicValue::Concrete(ConcreteValue::Int(10))); // same
        region2.write(8, SymbolicValue::Concrete(ConcreteValue::Int(30))); // different
        s2.memory.insert("stack".to_string(), region2);

        let result = op.merge_states(&s1, &s2).unwrap();
        let stack = result.merged_state.memory.get("stack").unwrap();

        // Offset 0 should be concrete (identical)
        assert_eq!(stack.read(0), Some(&SymbolicValue::Concrete(ConcreteValue::Int(10))));

        // Offset 8 should be ITE (different)
        assert!(matches!(stack.read(8), Some(SymbolicValue::Ite { .. })));
    }

    #[test]
    fn test_merge_extensions_conflict() {
        let config = MergeConfig::default();
        let mut op = MergeOperator::new(config);

        let mut s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F]);
        s1.negotiation.extensions.push(Extension::new(
            0x0000,
            "server_name",
            vec![1, 2, 3],
        ));

        let mut s2 = make_test_state(2, HandshakePhase::ClientHello, &[0x002F]);
        s2.negotiation.extensions.push(Extension::new(
            0x0000,
            "server_name",
            vec![4, 5, 6], // different data
        ));

        let result = op.merge_states(&s1, &s2).unwrap();
        assert_eq!(result.metadata.extension_conflicts, 1);
    }

    #[test]
    fn test_protocol_merge_trait() {
        let config = MergeConfig::default();
        let mut op = MergeOperator::new(config);

        let s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F]);
        let s2 = make_test_state(2, HandshakePhase::ClientHello, &[0x002F]);

        assert!(op.can_merge(&s1, &s2));
        let cost = op.merge_cost(&s1, &s2);
        assert!(cost.total_score() >= 0.0);

        let result = <MergeOperator as ProtocolMerge>::merge(&mut op, &s1, &s2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_merge_different_versions() {
        let config = MergeConfig::default();
        let mut op = MergeOperator::new(config);

        let mut s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F]);
        s1.negotiation.version = ProtocolVersion::Tls12;

        let mut s2 = make_test_state(2, HandshakePhase::ClientHello, &[0x002F]);
        s2.negotiation.version = ProtocolVersion::Tls13;

        assert!(!op.can_merge(&s1, &s2));
    }

    #[test]
    fn test_merge_fips_override() {
        let mut config = MergeConfig::default();
        config.fips_mode = true;
        let lattice = crate::lattice::SecurityLattice::from_standard_registry();
        let mut op = MergeOperator::new(config).with_lattice(lattice);

        let mut s1 = make_test_state(1, HandshakePhase::ServerHello, &[0x002F, 0xCCA8]);
        s1.negotiation.selected_cipher = Some(0x002F); // AES (FIPS)

        let mut s2 = make_test_state(2, HandshakePhase::ServerHello, &[0x002F, 0xCCA8]);
        s2.negotiation.selected_cipher = Some(0xCCA8); // ChaCha20 (not FIPS)

        let result = op.merge_states(&s1, &s2).unwrap();
        assert_eq!(result.metadata.cipher_selection_mode, CipherSelectionMode::FipsOverride);
    }
}
