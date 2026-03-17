use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::SmtExpr;

/// Encodes regulatory obligations into SMT/constraint expressions.
/// Implements the translation function τ from the typed obligation algebra
/// to quantifier-free SMT formulas.
#[derive(Debug, Clone)]
pub struct ObligationEncoder {
    variable_counter: usize,
    provenance: HashMap<String, ObligationProvenance>,
    jurisdiction_weights: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObligationProvenance {
    pub variable_name: String,
    pub obligation_id: String,
    pub article_ref: String,
    pub jurisdiction: String,
    pub obligation_type: String,
    pub is_hard: bool,
}

#[derive(Debug, Clone)]
pub struct EncodedObligation {
    pub constraint_expr: SmtExpr,
    pub variable_name: String,
    pub is_hard: bool,
    pub weight: f64,
    pub provenance: ObligationProvenance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObligationKind {
    Obligation,
    Permission,
    Prohibition,
}

#[derive(Debug, Clone)]
pub struct RawObligation {
    pub id: String,
    pub kind: ObligationKind,
    pub jurisdiction: String,
    pub article_ref: String,
    pub description: String,
    pub is_binding: bool,
    pub risk_weight: f64,
    pub conditions: Vec<ConditionSpec>,
    pub exemptions: Vec<String>,
    pub cross_refs: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ConditionSpec {
    pub condition_var: String,
    pub required_value: bool,
}

impl ObligationEncoder {
    pub fn new() -> Self {
        ObligationEncoder {
            variable_counter: 0,
            provenance: HashMap::new(),
            jurisdiction_weights: HashMap::new(),
        }
    }

    pub fn set_jurisdiction_weight(&mut self, jurisdiction: &str, weight: f64) {
        self.jurisdiction_weights.insert(jurisdiction.to_string(), weight);
    }

    fn next_var(&mut self, prefix: &str) -> String {
        self.variable_counter += 1;
        format!("{}_{}", prefix, self.variable_counter)
    }

    fn compliance_var(&mut self, obligation_id: &str) -> String {
        format!("comply_{}", obligation_id.replace("::", "_").replace("-", "_"))
    }

    pub fn encode_obligation(&mut self, obl: &RawObligation) -> EncodedObligation {
        let var_name = self.compliance_var(&obl.id);
        let base_expr = match obl.kind {
            ObligationKind::Obligation => {
                // OBL: the compliance variable must be true
                SmtExpr::Var(var_name.clone(), crate::SmtSort::Bool)
            }
            ObligationKind::Permission => {
                // PERM: the compliance variable may be true (no constraint by itself)
                SmtExpr::BoolLit(true)
            }
            ObligationKind::Prohibition => {
                // PROH: the compliance variable must be false (or the prohibition variable true)
                SmtExpr::Not(Box::new(SmtExpr::Var(format!("action_{}", obl.id.replace("::", "_").replace("-", "_")), crate::SmtSort::Bool)))
            }
        };

        // Apply conditions: if condition_var then base_expr
        let conditioned_expr = if obl.conditions.is_empty() {
            base_expr
        } else {
            let condition_conj = self.encode_conditions(&obl.conditions);
            SmtExpr::Implies(Box::new(condition_conj), Box::new(base_expr))
        };

        // Apply exemptions: base_expr OR any_exemption_applies
        let final_expr = if obl.exemptions.is_empty() {
            conditioned_expr
        } else {
            let exemption_disj = self.encode_exemptions(&obl.exemptions);
            SmtExpr::Or(vec![conditioned_expr, exemption_disj])
        };

        let is_hard = obl.is_binding && obl.kind != ObligationKind::Permission;
        let jur_weight = self.jurisdiction_weights.get(&obl.jurisdiction).copied().unwrap_or(1.0);
        let weight = obl.risk_weight * jur_weight;

        let prov = ObligationProvenance {
            variable_name: var_name.clone(),
            obligation_id: obl.id.clone(),
            article_ref: obl.article_ref.clone(),
            jurisdiction: obl.jurisdiction.clone(),
            obligation_type: format!("{:?}", obl.kind),
            is_hard,
        };
        self.provenance.insert(var_name.clone(), prov.clone());

        EncodedObligation {
            constraint_expr: final_expr,
            variable_name: var_name,
            is_hard,
            weight,
            provenance: prov,
        }
    }

    fn encode_conditions(&self, conditions: &[ConditionSpec]) -> SmtExpr {
        let cond_exprs: Vec<SmtExpr> = conditions.iter().map(|c| {
            if c.required_value {
                SmtExpr::Var(c.condition_var.clone(), crate::SmtSort::Bool)
            } else {
                SmtExpr::Not(Box::new(SmtExpr::Var(c.condition_var.clone(), crate::SmtSort::Bool)))
            }
        }).collect();

        if cond_exprs.len() == 1 {
            cond_exprs.into_iter().next().unwrap()
        } else {
            SmtExpr::And(cond_exprs)
        }
    }

    fn encode_exemptions(&self, exemptions: &[String]) -> SmtExpr {
        let exempt_exprs: Vec<SmtExpr> = exemptions.iter()
            .map(|e| SmtExpr::Var(format!("exempt_{}", e.replace("::", "_").replace("-", "_")), crate::SmtSort::Bool))
            .collect();

        if exempt_exprs.len() == 1 {
            exempt_exprs.into_iter().next().unwrap()
        } else {
            SmtExpr::Or(exempt_exprs)
        }
    }

    /// τ(O₁ ⊗ O₂) = τ(O₁) ∧ τ(O₂)  [conjunction]
    pub fn encode_conjunction(&self, left: &SmtExpr, right: &SmtExpr) -> SmtExpr {
        SmtExpr::And(vec![left.clone(), right.clone()])
    }

    /// τ(O₁ ⊕ O₂) = τ(O₁) ∨ τ(O₂)  [disjunction]
    pub fn encode_disjunction(&self, left: &SmtExpr, right: &SmtExpr) -> SmtExpr {
        SmtExpr::Or(vec![left.clone(), right.clone()])
    }

    /// τ(O₁ ▷ O₂) = ite(priority, τ(O₁), τ(O₂))  [jurisdictional override]
    pub fn encode_jurisdictional_override(&mut self, priority: &SmtExpr, fallback: &SmtExpr, priority_jur: &str) -> SmtExpr {
        let priority_var = format!("priority_{}", priority_jur);
        SmtExpr::Ite(
            Box::new(SmtExpr::Var(priority_var, crate::SmtSort::Bool)),
            Box::new(priority.clone()),
            Box::new(fallback.clone()),
        )
    }

    /// τ(O₁ ⊘ O₂) = τ(O₁) ∧ ¬τ(O₂)  [exception]
    pub fn encode_exception(&self, base: &SmtExpr, exemption: &SmtExpr) -> SmtExpr {
        SmtExpr::And(vec![
            base.clone(),
            SmtExpr::Not(Box::new(exemption.clone())),
        ])
    }

    /// Encode a set of obligations into a combined constraint system
    pub fn encode_obligation_set(&mut self, obligations: &[RawObligation]) -> EncodedObligationSet {
        let mut hard_constraints = Vec::new();
        let mut soft_constraints = Vec::new();
        let mut all_encoded = Vec::new();

        for obl in obligations {
            let encoded = self.encode_obligation(obl);
            if encoded.is_hard {
                hard_constraints.push(encoded.clone());
            } else {
                soft_constraints.push(encoded.clone());
            }
            all_encoded.push(encoded);
        }

        // Build cross-reference constraints
        let cross_ref_constraints = self.encode_cross_references(obligations);

        EncodedObligationSet {
            hard_constraints,
            soft_constraints,
            cross_ref_constraints,
            all_encoded,
            provenance: self.provenance.clone(),
        }
    }

    fn encode_cross_references(&self, obligations: &[RawObligation]) -> Vec<SmtExpr> {
        let mut constraints = Vec::new();
        let obl_map: HashMap<&str, &RawObligation> = obligations.iter()
            .map(|o| (o.id.as_str(), o))
            .collect();

        for obl in obligations {
            for cross_ref in &obl.cross_refs {
                if obl_map.contains_key(cross_ref.as_str()) {
                    // If obligation A references B, compliance with A implies awareness of B
                    let var_a = format!("comply_{}", obl.id.replace("::", "_").replace("-", "_"));
                    let var_b = format!("aware_{}", cross_ref.replace("::", "_").replace("-", "_"));
                    constraints.push(SmtExpr::Implies(
                        Box::new(SmtExpr::Var(var_a, crate::SmtSort::Bool)),
                        Box::new(SmtExpr::Var(var_b, crate::SmtSort::Bool)),
                    ));
                }
            }
        }
        constraints
    }

    pub fn get_provenance(&self, var_name: &str) -> Option<&ObligationProvenance> {
        self.provenance.get(var_name)
    }

    pub fn all_provenance(&self) -> &HashMap<String, ObligationProvenance> {
        &self.provenance
    }

    pub fn variable_count(&self) -> usize {
        self.variable_counter
    }
}

impl Default for ObligationEncoder {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone)]
pub struct EncodedObligationSet {
    pub hard_constraints: Vec<EncodedObligation>,
    pub soft_constraints: Vec<EncodedObligation>,
    pub cross_ref_constraints: Vec<SmtExpr>,
    pub all_encoded: Vec<EncodedObligation>,
    pub provenance: HashMap<String, ObligationProvenance>,
}

impl EncodedObligationSet {
    pub fn hard_formula(&self) -> SmtExpr {
        let exprs: Vec<SmtExpr> = self.hard_constraints.iter()
            .map(|e| e.constraint_expr.clone())
            .chain(self.cross_ref_constraints.iter().cloned())
            .collect();
        if exprs.is_empty() { SmtExpr::BoolLit(true) }
        else if exprs.len() == 1 { exprs.into_iter().next().unwrap() }
        else { SmtExpr::And(exprs) }
    }

    pub fn soft_clauses(&self) -> Vec<(SmtExpr, f64)> {
        self.soft_constraints.iter()
            .map(|e| (e.constraint_expr.clone(), e.weight))
            .collect()
    }

    pub fn total_hard(&self) -> usize { self.hard_constraints.len() }
    pub fn total_soft(&self) -> usize { self.soft_constraints.len() }
    pub fn total(&self) -> usize { self.all_encoded.len() }

    pub fn total_soft_weight(&self) -> f64 {
        self.soft_constraints.iter().map(|e| e.weight).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_obl(id: &str, kind: ObligationKind, binding: bool) -> RawObligation {
        RawObligation {
            id: id.to_string(),
            kind,
            jurisdiction: "EU".to_string(),
            article_ref: format!("Art.{}", id),
            description: format!("Test obligation {}", id),
            is_binding: binding,
            risk_weight: 1.0,
            conditions: Vec::new(),
            exemptions: Vec::new(),
            cross_refs: Vec::new(),
        }
    }

    #[test]
    fn test_encode_obligation() {
        let mut encoder = ObligationEncoder::new();
        let obl = make_obl("obl1", ObligationKind::Obligation, true);
        let encoded = encoder.encode_obligation(&obl);
        assert!(encoded.is_hard);
        assert!(matches!(encoded.constraint_expr, SmtExpr::Var(..)));
    }

    #[test]
    fn test_encode_prohibition() {
        let mut encoder = ObligationEncoder::new();
        let obl = make_obl("proh1", ObligationKind::Prohibition, true);
        let encoded = encoder.encode_obligation(&obl);
        assert!(encoded.is_hard);
        assert!(matches!(encoded.constraint_expr, SmtExpr::Not(_)));
    }

    #[test]
    fn test_encode_permission() {
        let mut encoder = ObligationEncoder::new();
        let obl = make_obl("perm1", ObligationKind::Permission, false);
        let encoded = encoder.encode_obligation(&obl);
        assert!(!encoded.is_hard);
    }

    #[test]
    fn test_conjunction() {
        let encoder = ObligationEncoder::new();
        let a = SmtExpr::Var("x".to_string(), crate::SmtSort::Bool);
        let b = SmtExpr::Var("y".to_string(), crate::SmtSort::Bool);
        let result = encoder.encode_conjunction(&a, &b);
        assert!(matches!(result, SmtExpr::And(_)));
    }

    #[test]
    fn test_obligation_set() {
        let mut encoder = ObligationEncoder::new();
        let obls = vec![
            make_obl("h1", ObligationKind::Obligation, true),
            make_obl("h2", ObligationKind::Prohibition, true),
            make_obl("s1", ObligationKind::Permission, false),
        ];
        let set = encoder.encode_obligation_set(&obls);
        assert_eq!(set.total_hard(), 2);
        assert_eq!(set.total_soft(), 1);
    }

    #[test]
    fn test_conditions() {
        let mut encoder = ObligationEncoder::new();
        let mut obl = make_obl("cond1", ObligationKind::Obligation, true);
        obl.conditions.push(ConditionSpec {
            condition_var: "is_high_risk".to_string(),
            required_value: true,
        });
        let encoded = encoder.encode_obligation(&obl);
        assert!(matches!(encoded.constraint_expr, SmtExpr::Implies(_, _)));
    }

    #[test]
    fn test_provenance_tracking() {
        let mut encoder = ObligationEncoder::new();
        let obl = make_obl("tracked1", ObligationKind::Obligation, true);
        let encoded = encoder.encode_obligation(&obl);
        let prov = encoder.get_provenance(&encoded.variable_name);
        assert!(prov.is_some());
        assert_eq!(prov.unwrap().obligation_id, "tracked1");
    }
}
