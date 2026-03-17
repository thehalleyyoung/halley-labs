// Proof traces, resolution chains, proof checking, and UNSAT certificates.

use crate::variable::{Literal, LiteralVec};
use std::collections::{HashMap, HashSet};

// ── ProofStep ─────────────────────────────────────────────────────────────────

/// A single step in a proof trace.
#[derive(Debug, Clone)]
pub enum ProofStep {
    /// An original input clause.
    Input {
        id: u32,
        literals: LiteralVec,
    },
    /// A decision was made.
    Decision {
        literal: Literal,
        level: u32,
    },
    /// A literal was propagated.
    Propagation {
        literal: Literal,
        reason: u32, // clause id
        level: u32,
    },
    /// A conflict was detected.
    Conflict {
        clause_id: u32,
    },
    /// A clause was learned via resolution.
    Learn {
        id: u32,
        literals: LiteralVec,
        resolution: ResolutionChain,
    },
    /// A clause was deleted.
    Delete {
        id: u32,
    },
    /// Backtrack to a decision level.
    Backtrack {
        level: u32,
    },
}

impl ProofStep {
    pub fn is_input(&self) -> bool {
        matches!(self, ProofStep::Input { .. })
    }
    pub fn is_learn(&self) -> bool {
        matches!(self, ProofStep::Learn { .. })
    }
    pub fn is_delete(&self) -> bool {
        matches!(self, ProofStep::Delete { .. })
    }
}

// ── ResolutionChain ───────────────────────────────────────────────────────────

/// A chain of resolution steps that derives a clause.
#[derive(Debug, Clone)]
pub struct ResolutionChain {
    /// Sequence of (clause_id, pivot_variable) pairs.
    /// Resolution starts from the first clause and resolves with each
    /// subsequent clause on the pivot variable.
    pub steps: Vec<ResolutionStep>,
}

/// Single resolution step.
#[derive(Debug, Clone)]
pub struct ResolutionStep {
    /// The clause to resolve with.
    pub clause_id: u32,
    /// The pivot variable used for resolution.
    pub pivot: Literal,
}

impl ResolutionChain {
    pub fn new() -> Self {
        ResolutionChain { steps: Vec::new() }
    }

    pub fn add_step(&mut self, clause_id: u32, pivot: Literal) {
        self.steps.push(ResolutionStep { clause_id, pivot });
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Perform the resolution chain given a mapping from clause ids to their literals.
    /// Returns the resulting clause.
    pub fn resolve(&self, clauses: &HashMap<u32, LiteralVec>) -> Option<LiteralVec> {
        if self.steps.is_empty() {
            return None;
        }

        let first_id = self.steps[0].clause_id;
        let mut current: HashSet<Literal> = clauses.get(&first_id)?.iter().cloned().collect();

        for step in &self.steps[1..] {
            let other = clauses.get(&step.clause_id)?;
            let pivot_pos = step.pivot;
            let pivot_neg = step.pivot.negated();

            // Remove the pivot from both sides.
            current.remove(&pivot_pos);
            current.remove(&pivot_neg);

            // Add literals from the other clause (except the pivot).
            for &lit in other {
                if lit != pivot_pos && lit != pivot_neg {
                    current.insert(lit);
                }
            }
        }

        Some(current.into_iter().collect())
    }
}

impl Default for ResolutionChain {
    fn default() -> Self {
        Self::new()
    }
}

// ── ProofTrace ────────────────────────────────────────────────────────────────

/// Records solver decisions for proof certificates.
#[derive(Debug, Clone)]
pub struct ProofTrace {
    steps: Vec<ProofStep>,
    /// Mapping from clause id to its literals (for verification).
    clause_map: HashMap<u32, LiteralVec>,
    next_id: u32,
}

impl ProofTrace {
    pub fn new() -> Self {
        ProofTrace {
            steps: Vec::new(),
            clause_map: HashMap::new(),
            next_id: 0,
        }
    }

    /// Record an input clause.
    pub fn add_input(&mut self, literals: LiteralVec) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.clause_map.insert(id, literals.clone());
        self.steps.push(ProofStep::Input { id, literals });
        id
    }

    /// Record a decision.
    pub fn add_decision(&mut self, literal: Literal, level: u32) {
        self.steps.push(ProofStep::Decision { literal, level });
    }

    /// Record a propagation.
    pub fn add_propagation(&mut self, literal: Literal, reason: u32, level: u32) {
        self.steps.push(ProofStep::Propagation {
            literal,
            reason,
            level,
        });
    }

    /// Record a conflict.
    pub fn add_conflict(&mut self, clause_id: u32) {
        self.steps.push(ProofStep::Conflict { clause_id });
    }

    /// Record a learned clause.
    pub fn add_learned(&mut self, literals: LiteralVec, resolution: ResolutionChain) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.clause_map.insert(id, literals.clone());
        self.steps.push(ProofStep::Learn {
            id,
            literals,
            resolution,
        });
        id
    }

    /// Record a clause deletion.
    pub fn add_deletion(&mut self, id: u32) {
        self.clause_map.remove(&id);
        self.steps.push(ProofStep::Delete { id });
    }

    /// Record a backtrack.
    pub fn add_backtrack(&mut self, level: u32) {
        self.steps.push(ProofStep::Backtrack { level });
    }

    /// Get all proof steps.
    pub fn steps(&self) -> &[ProofStep] {
        &self.steps
    }

    /// Get the clause literals by id.
    pub fn get_clause(&self, id: u32) -> Option<&LiteralVec> {
        self.clause_map.get(&id)
    }

    /// Number of steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Number of input clauses.
    pub fn input_count(&self) -> usize {
        self.steps.iter().filter(|s| s.is_input()).count()
    }

    /// Number of learned clauses.
    pub fn learned_count(&self) -> usize {
        self.steps.iter().filter(|s| s.is_learn()).count()
    }
}

impl Default for ProofTrace {
    fn default() -> Self {
        Self::new()
    }
}

// ── UnsatCertificate ──────────────────────────────────────────────────────────

/// Certificate of unsatisfiability.
#[derive(Debug, Clone)]
pub struct UnsatCertificate {
    /// The minimal UNSAT core (subset of input clause ids).
    pub core: Vec<u32>,
    /// Resolution proof that derives the empty clause from the core.
    pub resolution_proof: Vec<(u32, LiteralVec, ResolutionChain)>,
}

impl UnsatCertificate {
    pub fn new(core: Vec<u32>) -> Self {
        UnsatCertificate {
            core,
            resolution_proof: Vec::new(),
        }
    }

    /// Add a resolution step to the proof.
    pub fn add_proof_step(
        &mut self,
        id: u32,
        result: LiteralVec,
        chain: ResolutionChain,
    ) {
        self.resolution_proof.push((id, result, chain));
    }

    /// Check if the proof derives the empty clause.
    pub fn derives_empty(&self) -> bool {
        self.resolution_proof
            .last()
            .map_or(false, |(_, lits, _)| lits.is_empty())
    }

    /// Size of the UNSAT core.
    pub fn core_size(&self) -> usize {
        self.core.len()
    }

    /// Number of resolution steps in the proof.
    pub fn proof_length(&self) -> usize {
        self.resolution_proof.len()
    }
}

// ── ProofChecker ──────────────────────────────────────────────────────────────

/// Verifies proof traces for correctness.
#[derive(Debug)]
pub struct ProofChecker {
    clause_map: HashMap<u32, LiteralVec>,
    errors: Vec<String>,
}

impl ProofChecker {
    pub fn new() -> Self {
        ProofChecker {
            clause_map: HashMap::new(),
            errors: Vec::new(),
        }
    }

    /// Verify a complete proof trace. Returns true if the proof is valid.
    pub fn verify(&mut self, trace: &ProofTrace) -> bool {
        self.clause_map.clear();
        self.errors.clear();

        for step in trace.steps() {
            match step {
                ProofStep::Input { id, literals } => {
                    self.clause_map.insert(*id, literals.clone());
                }
                ProofStep::Learn {
                    id,
                    literals,
                    resolution,
                } => {
                    // Verify the resolution chain derives the learned clause.
                    if let Some(derived) = resolution.resolve(&self.clause_map) {
                        let derived_set: HashSet<Literal> = derived.into_iter().collect();
                        let expected_set: HashSet<Literal> = literals.iter().cloned().collect();

                        if !derived_set.is_subset(&expected_set) {
                            self.errors.push(format!(
                                "Learned clause {} has literals not in resolution result",
                                id
                            ));
                        }
                    } else if !resolution.is_empty() {
                        self.errors.push(format!(
                            "Resolution chain for clause {} references unknown clauses",
                            id
                        ));
                    }

                    self.clause_map.insert(*id, literals.clone());
                }
                ProofStep::Delete { id } => {
                    self.clause_map.remove(id);
                }
                _ => {
                    // Decision, propagation, conflict, backtrack steps are informational.
                }
            }
        }

        self.errors.is_empty()
    }

    /// Verify an UNSAT certificate.
    pub fn verify_unsat_certificate(
        &mut self,
        cert: &UnsatCertificate,
        input_clauses: &HashMap<u32, LiteralVec>,
    ) -> bool {
        self.errors.clear();

        // Check that all core clauses exist.
        for &core_id in &cert.core {
            if !input_clauses.contains_key(&core_id) {
                self.errors
                    .push(format!("Core clause {} not found in input", core_id));
            }
        }

        // Build clause map from input + proof steps.
        let mut clause_map = input_clauses.clone();
        for (id, result, chain) in &cert.resolution_proof {
            if let Some(derived) = chain.resolve(&clause_map) {
                let derived_set: HashSet<Literal> = derived.into_iter().collect();
                let expected_set: HashSet<Literal> = result.iter().cloned().collect();

                if !derived_set.is_subset(&expected_set) {
                    self.errors.push(format!(
                        "Resolution step {} produces unexpected literals",
                        id
                    ));
                }
            }
            clause_map.insert(*id, result.clone());
        }

        // Check that the proof derives the empty clause.
        if !cert.derives_empty() {
            self.errors
                .push("Proof does not derive the empty clause".into());
        }

        self.errors.is_empty()
    }

    /// Get errors found during verification.
    pub fn errors(&self) -> &[String] {
        &self.errors
    }
}

impl Default for ProofChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    fn lit(v: i32) -> Literal {
        Literal::from_dimacs(v)
    }

    #[test]
    fn test_proof_trace_basic() {
        let mut trace = ProofTrace::new();
        let c0 = trace.add_input(smallvec![lit(1), lit(2)]);
        let c1 = trace.add_input(smallvec![lit(-1), lit(2)]);
        trace.add_decision(lit(1), 1);
        trace.add_propagation(lit(2), c0, 1);

        assert_eq!(trace.len(), 4);
        assert_eq!(trace.input_count(), 2);
    }

    #[test]
    fn test_proof_trace_learned() {
        let mut trace = ProofTrace::new();
        let c0 = trace.add_input(smallvec![lit(1), lit(2)]);
        let c1 = trace.add_input(smallvec![lit(-1), lit(2)]);

        let mut chain = ResolutionChain::new();
        chain.add_step(c0, lit(1));
        chain.add_step(c1, lit(1));
        let c2 = trace.add_learned(smallvec![lit(2)], chain);

        assert_eq!(trace.learned_count(), 1);
        assert_eq!(trace.get_clause(c2).unwrap().len(), 1);
    }

    #[test]
    fn test_proof_trace_deletion() {
        let mut trace = ProofTrace::new();
        let c0 = trace.add_input(smallvec![lit(1)]);
        trace.add_deletion(c0);
        assert!(trace.get_clause(c0).is_none());
    }

    #[test]
    fn test_resolution_chain_simple() {
        // Resolve (1 ∨ 2) with (¬1 ∨ 3) on pivot 1 → (2 ∨ 3)
        let mut clauses = HashMap::new();
        clauses.insert(0, smallvec![lit(1), lit(2)]);
        clauses.insert(1, smallvec![lit(-1), lit(3)]);

        let mut chain = ResolutionChain::new();
        chain.add_step(0, lit(1));
        chain.add_step(1, lit(1));

        let result = chain.resolve(&clauses).unwrap();
        let result_set: HashSet<Literal> = result.into_iter().collect();
        assert!(result_set.contains(&lit(2)));
        assert!(result_set.contains(&lit(3)));
        assert!(!result_set.contains(&lit(1)));
        assert!(!result_set.contains(&lit(-1)));
    }

    #[test]
    fn test_resolution_chain_empty_result() {
        // Resolve (1) with (¬1) on pivot 1 → empty clause.
        let mut clauses = HashMap::new();
        clauses.insert(0, smallvec![lit(1)]);
        clauses.insert(1, smallvec![lit(-1)]);

        let mut chain = ResolutionChain::new();
        chain.add_step(0, lit(1));
        chain.add_step(1, lit(1));

        let result = chain.resolve(&clauses).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_resolution_chain_multi_step() {
        // (1 ∨ 2) resolve with (¬1 ∨ 3) → (2 ∨ 3)
        // (2 ∨ 3) resolve with (¬2 ∨ 4) → (3 ∨ 4)
        let mut clauses = HashMap::new();
        clauses.insert(0, smallvec![lit(1), lit(2)]);
        clauses.insert(1, smallvec![lit(-1), lit(3)]);
        clauses.insert(2, smallvec![lit(-2), lit(4)]);

        let mut chain = ResolutionChain::new();
        chain.add_step(0, lit(1));
        chain.add_step(1, lit(1));
        chain.add_step(2, lit(2));

        let result = chain.resolve(&clauses).unwrap();
        let result_set: HashSet<Literal> = result.into_iter().collect();
        assert!(result_set.contains(&lit(3)));
        assert!(result_set.contains(&lit(4)));
    }

    #[test]
    fn test_proof_checker_valid() {
        let mut trace = ProofTrace::new();
        let c0 = trace.add_input(smallvec![lit(1), lit(2)]);
        let c1 = trace.add_input(smallvec![lit(-1), lit(2)]);

        let mut chain = ResolutionChain::new();
        chain.add_step(c0, lit(1));
        chain.add_step(c1, lit(1));
        trace.add_learned(smallvec![lit(2)], chain);

        let mut checker = ProofChecker::new();
        assert!(checker.verify(&trace));
        assert!(checker.errors().is_empty());
    }

    #[test]
    fn test_proof_checker_empty_trace() {
        let trace = ProofTrace::new();
        let mut checker = ProofChecker::new();
        assert!(checker.verify(&trace));
    }

    #[test]
    fn test_unsat_certificate() {
        let mut cert = UnsatCertificate::new(vec![0, 1]);

        let mut chain = ResolutionChain::new();
        chain.add_step(0, lit(1));
        chain.add_step(1, lit(1));
        cert.add_proof_step(2, smallvec![], chain);

        assert!(cert.derives_empty());
        assert_eq!(cert.core_size(), 2);
        assert_eq!(cert.proof_length(), 1);
    }

    #[test]
    fn test_unsat_certificate_does_not_derive_empty() {
        let cert = UnsatCertificate::new(vec![0]);
        assert!(!cert.derives_empty());
    }

    #[test]
    fn test_verify_unsat_certificate() {
        let mut input = HashMap::new();
        input.insert(0, smallvec![lit(1)]);
        input.insert(1, smallvec![lit(-1)]);

        let mut cert = UnsatCertificate::new(vec![0, 1]);
        let mut chain = ResolutionChain::new();
        chain.add_step(0, lit(1));
        chain.add_step(1, lit(1));
        cert.add_proof_step(2, smallvec![], chain);

        let mut checker = ProofChecker::new();
        assert!(checker.verify_unsat_certificate(&cert, &input));
    }

    #[test]
    fn test_proof_step_queries() {
        let step = ProofStep::Input {
            id: 0,
            literals: smallvec![lit(1)],
        };
        assert!(step.is_input());
        assert!(!step.is_learn());
        assert!(!step.is_delete());
    }

    #[test]
    fn test_proof_trace_backtrack() {
        let mut trace = ProofTrace::new();
        trace.add_decision(lit(1), 1);
        trace.add_backtrack(0);
        assert_eq!(trace.len(), 2);
    }

    #[test]
    fn test_resolution_chain_default() {
        let chain = ResolutionChain::default();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
    }
}
