use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use crate::obligation::ObligationId;
use crate::cost::CostVector;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StrategyId(pub String);

impl StrategyId {
    pub fn new(id: impl Into<String>) -> Self { StrategyId(id.into()) }
    pub fn as_str(&self) -> &str { &self.0 }
}

impl fmt::Display for StrategyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.0) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceChoice {
    Comply,
    Waive,
    Defer,
    PartialComply(u8),
}

impl ComplianceChoice {
    pub fn is_compliant(&self) -> bool {
        matches!(self, ComplianceChoice::Comply)
    }

    pub fn compliance_level(&self) -> f64 {
        match self {
            ComplianceChoice::Comply => 1.0,
            ComplianceChoice::PartialComply(p) => *p as f64 / 100.0,
            ComplianceChoice::Defer => 0.0,
            ComplianceChoice::Waive => 0.0,
        }
    }
}

impl fmt::Display for ComplianceChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComplianceChoice::Comply => write!(f, "COMPLY"),
            ComplianceChoice::Waive => write!(f, "WAIVE"),
            ComplianceChoice::Defer => write!(f, "DEFER"),
            ComplianceChoice::PartialComply(p) => write!(f, "PARTIAL({}%)", p),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyAssignment {
    pub assignments: HashMap<String, ComplianceChoice>,
}

impl StrategyAssignment {
    pub fn new() -> Self {
        StrategyAssignment { assignments: HashMap::new() }
    }

    pub fn set(&mut self, obligation_id: &ObligationId, choice: ComplianceChoice) {
        self.assignments.insert(obligation_id.0.clone(), choice);
    }

    pub fn get(&self, obligation_id: &ObligationId) -> Option<&ComplianceChoice> {
        self.assignments.get(&obligation_id.0)
    }

    pub fn is_compliant(&self, obligation_id: &ObligationId) -> bool {
        self.assignments.get(&obligation_id.0)
            .map(|c| c.is_compliant())
            .unwrap_or(false)
    }

    pub fn compliance_count(&self) -> usize {
        self.assignments.values().filter(|c| c.is_compliant()).count()
    }

    pub fn total_count(&self) -> usize {
        self.assignments.len()
    }

    pub fn coverage(&self) -> f64 {
        if self.assignments.is_empty() { return 0.0; }
        self.compliance_count() as f64 / self.total_count() as f64
    }

    pub fn as_bitvec(&self, obligation_ids: &[ObligationId]) -> Vec<bool> {
        obligation_ids.iter().map(|id| self.is_compliant(id)).collect()
    }

    pub fn from_bitvec(obligation_ids: &[ObligationId], bits: &[bool]) -> Self {
        let mut sa = StrategyAssignment::new();
        for (id, &bit) in obligation_ids.iter().zip(bits.iter()) {
            sa.set(id, if bit { ComplianceChoice::Comply } else { ComplianceChoice::Waive });
        }
        sa
    }

    pub fn diff(&self, other: &StrategyAssignment) -> StrategyDiff {
        let mut changed = Vec::new();
        let all_keys: HashSet<&String> = self.assignments.keys().chain(other.assignments.keys()).collect();
        for key in all_keys {
            let old_choice = self.assignments.get(key).copied();
            let new_choice = other.assignments.get(key).copied();
            if old_choice != new_choice {
                changed.push(StrategyChange {
                    obligation_id: ObligationId::new(key.clone()),
                    old_choice,
                    new_choice,
                });
            }
        }
        StrategyDiff { changes: changed }
    }

    pub fn merge(&self, other: &StrategyAssignment) -> StrategyAssignment {
        let mut merged = self.clone();
        for (k, v) in &other.assignments {
            merged.assignments.insert(k.clone(), *v);
        }
        merged
    }
}

impl Default for StrategyAssignment {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyChange {
    pub obligation_id: ObligationId,
    pub old_choice: Option<ComplianceChoice>,
    pub new_choice: Option<ComplianceChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyDiff {
    pub changes: Vec<StrategyChange>,
}

impl StrategyDiff {
    pub fn num_changes(&self) -> usize { self.changes.len() }

    pub fn is_empty(&self) -> bool { self.changes.is_empty() }

    pub fn newly_compliant(&self) -> Vec<&ObligationId> {
        self.changes.iter()
            .filter(|c| c.new_choice.map(|nc| nc.is_compliant()).unwrap_or(false)
                && !c.old_choice.map(|oc| oc.is_compliant()).unwrap_or(false))
            .map(|c| &c.obligation_id)
            .collect()
    }

    pub fn newly_waived(&self) -> Vec<&ObligationId> {
        self.changes.iter()
            .filter(|c| c.old_choice.map(|oc| oc.is_compliant()).unwrap_or(false)
                && !c.new_choice.map(|nc| nc.is_compliant()).unwrap_or(false))
            .map(|c| &c.obligation_id)
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStrategy {
    pub id: StrategyId,
    pub name: String,
    pub description: String,
    pub assignment: StrategyAssignment,
    pub cost: CostVector,
    pub metadata: HashMap<String, String>,
}

impl ComplianceStrategy {
    pub fn new(id: StrategyId, name: &str, assignment: StrategyAssignment, cost: CostVector) -> Self {
        ComplianceStrategy {
            id, name: name.to_string(),
            description: String::new(),
            assignment, cost,
            metadata: HashMap::new(),
        }
    }

    pub fn coverage(&self) -> f64 { self.assignment.coverage() }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    pub fn dominates(&self, other: &ComplianceStrategy) -> bool {
        self.cost.dominates(&other.cost)
    }

    pub fn transition_cost(&self, other: &ComplianceStrategy) -> usize {
        self.assignment.diff(&other.assignment).num_changes()
    }
}

impl fmt::Display for ComplianceStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Strategy '{}' (coverage={:.1}%, {})", self.name, self.coverage() * 100.0, self.cost)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyVariable {
    pub obligation_id: ObligationId,
    pub variable_name: String,
    pub is_hard: bool,
    pub weight: f64,
}

impl StrategyVariable {
    pub fn new(obligation_id: ObligationId, var_name: &str, is_hard: bool, weight: f64) -> Self {
        StrategyVariable {
            obligation_id, variable_name: var_name.to_string(), is_hard, weight,
        }
    }

    pub fn hard(obligation_id: ObligationId, var_name: &str) -> Self {
        Self::new(obligation_id, var_name, true, 1.0)
    }

    pub fn soft(obligation_id: ObligationId, var_name: &str, weight: f64) -> Self {
        Self::new(obligation_id, var_name, false, weight)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySpace {
    pub variables: Vec<StrategyVariable>,
    pub hard_constraint_ids: HashSet<String>,
}

impl StrategySpace {
    pub fn new() -> Self {
        StrategySpace { variables: Vec::new(), hard_constraint_ids: HashSet::new() }
    }

    pub fn add_variable(&mut self, var: StrategyVariable) {
        if var.is_hard {
            self.hard_constraint_ids.insert(var.obligation_id.0.clone());
        }
        self.variables.push(var);
    }

    pub fn dimension(&self) -> usize { self.variables.len() }

    pub fn hard_count(&self) -> usize { self.variables.iter().filter(|v| v.is_hard).count() }

    pub fn soft_count(&self) -> usize { self.variables.iter().filter(|v| !v.is_hard).count() }

    pub fn total_weight(&self) -> f64 { self.variables.iter().map(|v| v.weight).sum() }

    pub fn validate_assignment(&self, assignment: &StrategyAssignment) -> StrategyValidation {
        let mut violations = Vec::new();
        let mut covered = 0;
        let mut total = 0;

        for var in &self.variables {
            total += 1;
            match assignment.get(&var.obligation_id) {
                Some(choice) if choice.is_compliant() => { covered += 1; }
                Some(_) if var.is_hard => {
                    violations.push(format!("Hard constraint {} not satisfied", var.obligation_id));
                }
                None if var.is_hard => {
                    violations.push(format!("Hard constraint {} has no assignment", var.obligation_id));
                }
                _ => {}
            }
        }

        let hard_satisfied = violations.is_empty();
        let is_valid = hard_satisfied;
        StrategyValidation {
            is_valid,
            violations,
            coverage: if total > 0 { covered as f64 / total as f64 } else { 0.0 },
            hard_satisfied,
        }
    }
}

impl Default for StrategySpace {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyValidation {
    pub is_valid: bool,
    pub violations: Vec<String>,
    pub coverage: f64,
    pub hard_satisfied: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_assignment() {
        let mut sa = StrategyAssignment::new();
        sa.set(&ObligationId::new("o1"), ComplianceChoice::Comply);
        sa.set(&ObligationId::new("o2"), ComplianceChoice::Waive);
        assert!(sa.is_compliant(&ObligationId::new("o1")));
        assert!(!sa.is_compliant(&ObligationId::new("o2")));
        assert_eq!(sa.coverage(), 0.5);
    }

    #[test]
    fn test_strategy_diff() {
        let mut s1 = StrategyAssignment::new();
        s1.set(&ObligationId::new("o1"), ComplianceChoice::Comply);
        s1.set(&ObligationId::new("o2"), ComplianceChoice::Waive);

        let mut s2 = StrategyAssignment::new();
        s2.set(&ObligationId::new("o1"), ComplianceChoice::Comply);
        s2.set(&ObligationId::new("o2"), ComplianceChoice::Comply);

        let diff = s1.diff(&s2);
        assert_eq!(diff.num_changes(), 1);
    }

    #[test]
    fn test_bitvec_roundtrip() {
        let ids = vec![ObligationId::new("a"), ObligationId::new("b"), ObligationId::new("c")];
        let bits = vec![true, false, true];
        let sa = StrategyAssignment::from_bitvec(&ids, &bits);
        let back = sa.as_bitvec(&ids);
        assert_eq!(bits, back);
    }

    #[test]
    fn test_strategy_space_validation() {
        let mut space = StrategySpace::new();
        space.add_variable(StrategyVariable::hard(ObligationId::new("h1"), "x_h1"));
        space.add_variable(StrategyVariable::soft(ObligationId::new("s1"), "x_s1", 0.5));

        let mut good = StrategyAssignment::new();
        good.set(&ObligationId::new("h1"), ComplianceChoice::Comply);
        good.set(&ObligationId::new("s1"), ComplianceChoice::Waive);
        assert!(space.validate_assignment(&good).is_valid);

        let mut bad = StrategyAssignment::new();
        bad.set(&ObligationId::new("h1"), ComplianceChoice::Waive);
        assert!(!space.validate_assignment(&bad).is_valid);
    }
}
