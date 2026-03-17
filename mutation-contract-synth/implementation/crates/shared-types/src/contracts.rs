//! Contract and specification types for the MutSpec system.
//!
//! Contracts represent the preconditions, postconditions, and invariants
//! synthesised from mutation analysis.  A [`Specification`] aggregates
//! multiple contracts for a function.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::formula::Formula;
use crate::operators::MutantId;

// ---------------------------------------------------------------------------
// ContractClause
// ---------------------------------------------------------------------------

/// A single clause in a contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContractClause {
    /// Precondition: must hold on entry.
    Requires(Formula),
    /// Postcondition: must hold on exit.
    Ensures(Formula),
    /// Loop invariant (included for completeness, not used in loop-free programs).
    Invariant(Formula),
}

impl ContractClause {
    pub fn requires(f: Formula) -> Self {
        ContractClause::Requires(f)
    }

    pub fn ensures(f: Formula) -> Self {
        ContractClause::Ensures(f)
    }

    pub fn invariant(f: Formula) -> Self {
        ContractClause::Invariant(f)
    }

    pub fn formula(&self) -> &Formula {
        match self {
            ContractClause::Requires(f)
            | ContractClause::Ensures(f)
            | ContractClause::Invariant(f) => f,
        }
    }

    pub fn formula_mut(&mut self) -> &mut Formula {
        match self {
            ContractClause::Requires(f)
            | ContractClause::Ensures(f)
            | ContractClause::Invariant(f) => f,
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            ContractClause::Requires(_) => "requires",
            ContractClause::Ensures(_) => "ensures",
            ContractClause::Invariant(_) => "invariant",
        }
    }

    pub fn is_requires(&self) -> bool {
        matches!(self, ContractClause::Requires(_))
    }

    pub fn is_ensures(&self) -> bool {
        matches!(self, ContractClause::Ensures(_))
    }

    pub fn is_invariant(&self) -> bool {
        matches!(self, ContractClause::Invariant(_))
    }

    /// JML-style annotation string.
    pub fn to_jml(&self) -> String {
        match self {
            ContractClause::Requires(f) => format!("//@ requires {};", f),
            ContractClause::Ensures(f) => format!("//@ ensures {};", f),
            ContractClause::Invariant(f) => format!("//@ loop_invariant {};", f),
        }
    }
}

impl fmt::Display for ContractClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContractClause::Requires(formula) => write!(f, "requires {formula}"),
            ContractClause::Ensures(formula) => write!(f, "ensures {formula}"),
            ContractClause::Invariant(formula) => write!(f, "invariant {formula}"),
        }
    }
}

// ---------------------------------------------------------------------------
// ContractStrength
// ---------------------------------------------------------------------------

/// Qualitative assessment of how strong a contract is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum ContractStrength {
    /// The contract kills all non-equivalent mutants.
    Strongest,
    /// The contract kills a significant fraction of mutants.
    Adequate,
    /// The contract kills some mutants but is incomplete.
    Weak,
    /// The contract is trivially true (kills nothing).
    Trivial,
}

impl ContractStrength {
    pub fn from_kill_ratio(ratio: f64) -> Self {
        if ratio >= 1.0 {
            ContractStrength::Strongest
        } else if ratio >= 0.8 {
            ContractStrength::Adequate
        } else if ratio > 0.0 {
            ContractStrength::Weak
        } else {
            ContractStrength::Trivial
        }
    }

    pub fn is_adequate_or_better(&self) -> bool {
        matches!(
            self,
            ContractStrength::Strongest | ContractStrength::Adequate
        )
    }

    pub fn name(&self) -> &'static str {
        match self {
            ContractStrength::Strongest => "strongest",
            ContractStrength::Adequate => "adequate",
            ContractStrength::Weak => "weak",
            ContractStrength::Trivial => "trivial",
        }
    }
}

impl fmt::Display for ContractStrength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// SynthesisTier
// ---------------------------------------------------------------------------

/// The synthesis strategy tier that produced a contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SynthesisTier {
    /// Tier 1: lattice walk over abstract interpretation domains.
    Tier1LatticeWalk,
    /// Tier 2: template-based enumeration.
    Tier2Template,
    /// Tier 3: fallback heuristic/SMT-guided search.
    Tier3Fallback,
}

impl SynthesisTier {
    pub fn name(&self) -> &'static str {
        match self {
            SynthesisTier::Tier1LatticeWalk => "Tier1-LatticeWalk",
            SynthesisTier::Tier2Template => "Tier2-Template",
            SynthesisTier::Tier3Fallback => "Tier3-Fallback",
        }
    }

    pub fn tier_number(&self) -> u8 {
        match self {
            SynthesisTier::Tier1LatticeWalk => 1,
            SynthesisTier::Tier2Template => 2,
            SynthesisTier::Tier3Fallback => 3,
        }
    }

    /// Returns the next fallback tier, if any.
    pub fn fallback(&self) -> Option<SynthesisTier> {
        match self {
            SynthesisTier::Tier1LatticeWalk => Some(SynthesisTier::Tier2Template),
            SynthesisTier::Tier2Template => Some(SynthesisTier::Tier3Fallback),
            SynthesisTier::Tier3Fallback => None,
        }
    }

    /// Parse from tier number.
    pub fn from_number(n: u8) -> Option<SynthesisTier> {
        match n {
            1 => Some(SynthesisTier::Tier1LatticeWalk),
            2 => Some(SynthesisTier::Tier2Template),
            3 => Some(SynthesisTier::Tier3Fallback),
            _ => None,
        }
    }
}

impl fmt::Display for SynthesisTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// ContractProvenance
// ---------------------------------------------------------------------------

/// Tracks which mutants contributed to a contract clause.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContractProvenance {
    /// Mutants that this clause was designed to kill.
    pub targeted_mutants: Vec<MutantId>,
    /// The synthesis tier that produced this clause.
    pub tier: SynthesisTier,
    /// Number of solver queries used.
    pub solver_queries: u32,
    /// Time in milliseconds to synthesise this clause.
    pub synthesis_time_ms: f64,
}

impl ContractProvenance {
    pub fn new(tier: SynthesisTier) -> Self {
        Self {
            targeted_mutants: Vec::new(),
            tier,
            solver_queries: 0,
            synthesis_time_ms: 0.0,
        }
    }

    pub fn with_mutant(mut self, id: MutantId) -> Self {
        self.targeted_mutants.push(id);
        self
    }

    pub fn with_mutants(mut self, ids: Vec<MutantId>) -> Self {
        self.targeted_mutants.extend(ids);
        self
    }

    pub fn with_solver_queries(mut self, n: u32) -> Self {
        self.solver_queries = n;
        self
    }

    pub fn with_time(mut self, ms: f64) -> Self {
        self.synthesis_time_ms = ms;
        self
    }

    pub fn mutant_count(&self) -> usize {
        self.targeted_mutants.len()
    }
}

impl fmt::Display for ContractProvenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tier={}, mutants={}, queries={}, time={:.1}ms",
            self.tier,
            self.targeted_mutants.len(),
            self.solver_queries,
            self.synthesis_time_ms
        )
    }
}

// ---------------------------------------------------------------------------
// Contract
// ---------------------------------------------------------------------------

/// A contract for a single function, consisting of multiple clauses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Contract {
    pub function_name: String,
    pub clauses: Vec<ContractClause>,
    pub provenance: Vec<ContractProvenance>,
    pub strength: ContractStrength,
    pub verified: bool,
}

impl Contract {
    pub fn new(function_name: impl Into<String>) -> Self {
        Self {
            function_name: function_name.into(),
            clauses: Vec::new(),
            provenance: Vec::new(),
            strength: ContractStrength::Trivial,
            verified: false,
        }
    }

    pub fn add_clause(&mut self, clause: ContractClause) {
        self.clauses.push(clause);
    }

    pub fn add_provenance(&mut self, prov: ContractProvenance) {
        self.provenance.push(prov);
    }

    pub fn set_strength(&mut self, strength: ContractStrength) {
        self.strength = strength;
    }

    pub fn mark_verified(&mut self) {
        self.verified = true;
    }

    pub fn preconditions(&self) -> Vec<&Formula> {
        self.clauses
            .iter()
            .filter_map(|c| match c {
                ContractClause::Requires(f) => Some(f),
                _ => None,
            })
            .collect()
    }

    pub fn postconditions(&self) -> Vec<&Formula> {
        self.clauses
            .iter()
            .filter_map(|c| match c {
                ContractClause::Ensures(f) => Some(f),
                _ => None,
            })
            .collect()
    }

    pub fn invariants(&self) -> Vec<&Formula> {
        self.clauses
            .iter()
            .filter_map(|c| match c {
                ContractClause::Invariant(f) => Some(f),
                _ => None,
            })
            .collect()
    }

    pub fn clause_count(&self) -> usize {
        self.clauses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }

    /// Total number of mutants targeted across all provenance entries.
    pub fn total_targeted_mutants(&self) -> usize {
        self.provenance.iter().map(|p| p.mutant_count()).sum()
    }

    /// Render as JML annotations.
    pub fn to_jml(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("// Contract for {}", self.function_name));
        for clause in &self.clauses {
            lines.push(clause.to_jml());
        }
        lines.join("\n")
    }

    /// Merge another contract's clauses into this one.
    pub fn merge_from(&mut self, other: &Contract) {
        self.clauses.extend(other.clauses.iter().cloned());
        self.provenance.extend(other.provenance.iter().cloned());
    }
}

impl fmt::Display for Contract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Contract for `{}` [{}{}]:",
            self.function_name,
            self.strength,
            if self.verified { ", verified" } else { "" }
        )?;
        for clause in &self.clauses {
            writeln!(f, "  {clause}")?;
        }
        for prov in &self.provenance {
            writeln!(f, "  provenance: {prov}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Specification
// ---------------------------------------------------------------------------

/// A specification wrapping multiple contracts for a program.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Specification {
    pub contracts: Vec<Contract>,
    pub program_name: Option<String>,
}

impl Specification {
    pub fn new() -> Self {
        Self {
            contracts: Vec::new(),
            program_name: None,
        }
    }

    pub fn with_program_name(mut self, name: impl Into<String>) -> Self {
        self.program_name = Some(name.into());
        self
    }

    pub fn add_contract(&mut self, contract: Contract) {
        self.contracts.push(contract);
    }

    pub fn contract_for(&self, function_name: &str) -> Option<&Contract> {
        self.contracts
            .iter()
            .find(|c| c.function_name == function_name)
    }

    pub fn contract_for_mut(&mut self, function_name: &str) -> Option<&mut Contract> {
        self.contracts
            .iter_mut()
            .find(|c| c.function_name == function_name)
    }

    pub fn is_empty(&self) -> bool {
        self.contracts.is_empty()
    }

    pub fn function_count(&self) -> usize {
        self.contracts.len()
    }

    pub fn total_clauses(&self) -> usize {
        self.contracts.iter().map(|c| c.clause_count()).sum()
    }

    pub fn overall_strength(&self) -> ContractStrength {
        if self.contracts.is_empty() {
            return ContractStrength::Trivial;
        }
        self.contracts
            .iter()
            .map(|c| c.strength)
            .max()
            .unwrap_or(ContractStrength::Trivial)
    }

    pub fn all_verified(&self) -> bool {
        !self.contracts.is_empty() && self.contracts.iter().all(|c| c.verified)
    }

    /// Render the full specification as JML.
    pub fn to_jml(&self) -> String {
        self.contracts
            .iter()
            .map(|c| c.to_jml())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

impl Default for Specification {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Specification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref name) = self.program_name {
            writeln!(f, "Specification for {name}")?;
        }
        writeln!(
            f,
            "{} contracts, {} total clauses, overall strength: {}",
            self.function_count(),
            self.total_clauses(),
            self.overall_strength()
        )?;
        for contract in &self.contracts {
            writeln!(f)?;
            write!(f, "{contract}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Verification status types
// ---------------------------------------------------------------------------

/// Result of verifying a contract clause against the program.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VerificationResult {
    /// The clause holds for all inputs.
    Valid,
    /// A counterexample was found.
    Invalid { counterexample: String },
    /// Verification timed out.
    Unknown { reason: String },
}

impl VerificationResult {
    pub fn is_valid(&self) -> bool {
        matches!(self, VerificationResult::Valid)
    }

    pub fn is_invalid(&self) -> bool {
        matches!(self, VerificationResult::Invalid { .. })
    }
}

impl fmt::Display for VerificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerificationResult::Valid => write!(f, "VALID"),
            VerificationResult::Invalid { counterexample } => {
                write!(f, "INVALID (counterexample: {counterexample})")
            }
            VerificationResult::Unknown { reason } => write!(f, "UNKNOWN ({reason})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::{Formula, Predicate, Term};

    fn sample_formula() -> Formula {
        Formula::Atom(Predicate {
            relation: crate::formula::Relation::Gt,
            left: Term::Var("x".into()),
            right: Term::Const(0),
        })
    }

    #[test]
    fn test_contract_clause_requires() {
        let clause = ContractClause::requires(sample_formula());
        assert!(clause.is_requires());
        assert!(!clause.is_ensures());
        assert_eq!(clause.kind_name(), "requires");
    }

    #[test]
    fn test_contract_clause_ensures() {
        let clause = ContractClause::ensures(sample_formula());
        assert!(clause.is_ensures());
        assert_eq!(clause.kind_name(), "ensures");
    }

    #[test]
    fn test_contract_clause_jml() {
        let clause = ContractClause::requires(Formula::True);
        let jml = clause.to_jml();
        assert!(jml.starts_with("//@"));
        assert!(jml.contains("requires"));
    }

    #[test]
    fn test_contract_clause_display() {
        let clause = ContractClause::ensures(Formula::True);
        let s = clause.to_string();
        assert!(s.contains("ensures"));
    }

    #[test]
    fn test_contract_clause_formula_access() {
        let mut clause = ContractClause::requires(Formula::True);
        assert_eq!(clause.formula(), &Formula::True);
        *clause.formula_mut() = Formula::False;
        assert_eq!(clause.formula(), &Formula::False);
    }

    #[test]
    fn test_contract_strength_from_ratio() {
        assert_eq!(
            ContractStrength::from_kill_ratio(1.0),
            ContractStrength::Strongest
        );
        assert_eq!(
            ContractStrength::from_kill_ratio(0.9),
            ContractStrength::Adequate
        );
        assert_eq!(
            ContractStrength::from_kill_ratio(0.5),
            ContractStrength::Weak
        );
        assert_eq!(
            ContractStrength::from_kill_ratio(0.0),
            ContractStrength::Trivial
        );
    }

    #[test]
    fn test_contract_strength_adequate_or_better() {
        assert!(ContractStrength::Strongest.is_adequate_or_better());
        assert!(ContractStrength::Adequate.is_adequate_or_better());
        assert!(!ContractStrength::Weak.is_adequate_or_better());
        assert!(!ContractStrength::Trivial.is_adequate_or_better());
    }

    #[test]
    fn test_contract_strength_ordering() {
        assert!(ContractStrength::Strongest < ContractStrength::Adequate);
        assert!(ContractStrength::Adequate < ContractStrength::Weak);
    }

    #[test]
    fn test_synthesis_tier() {
        assert_eq!(SynthesisTier::Tier1LatticeWalk.tier_number(), 1);
        assert_eq!(SynthesisTier::Tier2Template.tier_number(), 2);
        assert_eq!(SynthesisTier::Tier3Fallback.tier_number(), 3);
    }

    #[test]
    fn test_synthesis_tier_fallback() {
        assert_eq!(
            SynthesisTier::Tier1LatticeWalk.fallback(),
            Some(SynthesisTier::Tier2Template)
        );
        assert_eq!(
            SynthesisTier::Tier2Template.fallback(),
            Some(SynthesisTier::Tier3Fallback)
        );
        assert_eq!(SynthesisTier::Tier3Fallback.fallback(), None);
    }

    #[test]
    fn test_synthesis_tier_from_number() {
        assert_eq!(
            SynthesisTier::from_number(1),
            Some(SynthesisTier::Tier1LatticeWalk)
        );
        assert_eq!(SynthesisTier::from_number(4), None);
    }

    #[test]
    fn test_synthesis_tier_display() {
        assert!(SynthesisTier::Tier1LatticeWalk
            .to_string()
            .contains("Tier1"));
    }

    #[test]
    fn test_contract_provenance() {
        let prov = ContractProvenance::new(SynthesisTier::Tier1LatticeWalk)
            .with_solver_queries(5)
            .with_time(42.0);
        assert_eq!(prov.mutant_count(), 0);
        assert_eq!(prov.solver_queries, 5);
        let s = prov.to_string();
        assert!(s.contains("Tier1"));
        assert!(s.contains("42.0"));
    }

    #[test]
    fn test_contract_provenance_with_mutants() {
        let id1 = MutantId::new();
        let id2 = MutantId::new();
        let prov =
            ContractProvenance::new(SynthesisTier::Tier2Template).with_mutants(vec![id1, id2]);
        assert_eq!(prov.mutant_count(), 2);
    }

    #[test]
    fn test_contract_basic() {
        let mut contract = Contract::new("foo");
        assert!(contract.is_empty());
        contract.add_clause(ContractClause::requires(Formula::True));
        contract.add_clause(ContractClause::ensures(sample_formula()));
        assert_eq!(contract.clause_count(), 2);
        assert_eq!(contract.preconditions().len(), 1);
        assert_eq!(contract.postconditions().len(), 1);
        assert_eq!(contract.invariants().len(), 0);
    }

    #[test]
    fn test_contract_strength() {
        let mut contract = Contract::new("bar");
        contract.set_strength(ContractStrength::Adequate);
        assert_eq!(contract.strength, ContractStrength::Adequate);
    }

    #[test]
    fn test_contract_verified() {
        let mut contract = Contract::new("baz");
        assert!(!contract.verified);
        contract.mark_verified();
        assert!(contract.verified);
    }

    #[test]
    fn test_contract_jml() {
        let mut contract = Contract::new("f");
        contract.add_clause(ContractClause::requires(Formula::True));
        contract.add_clause(ContractClause::ensures(Formula::True));
        let jml = contract.to_jml();
        assert!(jml.contains("requires"));
        assert!(jml.contains("ensures"));
    }

    #[test]
    fn test_contract_merge() {
        let mut c1 = Contract::new("f");
        c1.add_clause(ContractClause::requires(Formula::True));
        let mut c2 = Contract::new("f");
        c2.add_clause(ContractClause::ensures(Formula::True));
        c1.merge_from(&c2);
        assert_eq!(c1.clause_count(), 2);
    }

    #[test]
    fn test_contract_display() {
        let mut contract = Contract::new("g");
        contract.add_clause(ContractClause::ensures(Formula::True));
        contract.set_strength(ContractStrength::Strongest);
        contract.mark_verified();
        let s = contract.to_string();
        assert!(s.contains("g"));
        assert!(s.contains("strongest"));
        assert!(s.contains("verified"));
    }

    #[test]
    fn test_specification_basic() {
        let mut spec = Specification::new().with_program_name("test_prog");
        assert!(spec.is_empty());
        let contract = Contract::new("f");
        spec.add_contract(contract);
        assert_eq!(spec.function_count(), 1);
        assert!(spec.contract_for("f").is_some());
        assert!(spec.contract_for("g").is_none());
    }

    #[test]
    fn test_specification_overall_strength() {
        let spec = Specification::new();
        assert_eq!(spec.overall_strength(), ContractStrength::Trivial);

        let mut spec2 = Specification::new();
        let mut c = Contract::new("f");
        c.set_strength(ContractStrength::Adequate);
        spec2.add_contract(c);
        assert_eq!(spec2.overall_strength(), ContractStrength::Adequate);
    }

    #[test]
    fn test_specification_all_verified() {
        let mut spec = Specification::new();
        assert!(!spec.all_verified());

        let mut c = Contract::new("f");
        c.mark_verified();
        spec.add_contract(c);
        assert!(spec.all_verified());
    }

    #[test]
    fn test_specification_jml() {
        let mut spec = Specification::new();
        let mut c = Contract::new("f");
        c.add_clause(ContractClause::requires(Formula::True));
        spec.add_contract(c);
        let jml = spec.to_jml();
        assert!(jml.contains("requires"));
    }

    #[test]
    fn test_specification_display() {
        let mut spec = Specification::new().with_program_name("prog");
        let c = Contract::new("f");
        spec.add_contract(c);
        let s = spec.to_string();
        assert!(s.contains("prog"));
        assert!(s.contains("1 contracts"));
    }

    #[test]
    fn test_specification_contract_for_mut() {
        let mut spec = Specification::new();
        let c = Contract::new("f");
        spec.add_contract(c);
        let cm = spec.contract_for_mut("f").unwrap();
        cm.add_clause(ContractClause::requires(Formula::True));
        assert_eq!(spec.total_clauses(), 1);
    }

    #[test]
    fn test_verification_result() {
        assert!(VerificationResult::Valid.is_valid());
        assert!(!VerificationResult::Valid.is_invalid());
        let inv = VerificationResult::Invalid {
            counterexample: "x=0".into(),
        };
        assert!(inv.is_invalid());
        let unk = VerificationResult::Unknown {
            reason: "timeout".into(),
        };
        assert!(!unk.is_valid());
    }

    #[test]
    fn test_verification_result_display() {
        assert_eq!(VerificationResult::Valid.to_string(), "VALID");
        let inv = VerificationResult::Invalid {
            counterexample: "x=0".into(),
        };
        assert!(inv.to_string().contains("x=0"));
    }

    #[test]
    fn test_contract_total_targeted_mutants() {
        let mut c = Contract::new("f");
        c.add_provenance(
            ContractProvenance::new(SynthesisTier::Tier1LatticeWalk)
                .with_mutants(vec![MutantId::new(), MutantId::new()]),
        );
        c.add_provenance(
            ContractProvenance::new(SynthesisTier::Tier2Template).with_mutant(MutantId::new()),
        );
        assert_eq!(c.total_targeted_mutants(), 3);
    }
}
