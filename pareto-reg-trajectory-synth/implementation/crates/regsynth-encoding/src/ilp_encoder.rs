use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::obligation_encoder::{RawObligation, ObligationKind};

/// ILP encoder: translates regulatory obligations to ILP (Integer Linear Programming) model.
/// Binary decision variables for compliance choices, linear constraints for requirements,
/// multi-objective encoding via epsilon-constraint method.
#[derive(Debug, Clone)]
pub struct IlpEncoder {
    variables: Vec<IlpVariable>,
    constraints: Vec<LinearConstraint>,
    objectives: Vec<LinearObjective>,
    var_index: HashMap<String, usize>,
    next_aux: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IlpVariable {
    pub name: String,
    pub var_type: IlpVarType,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub obligation_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IlpVarType {
    Binary,
    Integer,
    Continuous,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearConstraint {
    pub name: String,
    pub coefficients: Vec<(String, f64)>,
    pub sense: ConstraintSense,
    pub rhs: f64,
    pub is_hard: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintSense {
    LessEqual,
    GreaterEqual,
    Equal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearObjective {
    pub name: String,
    pub coefficients: Vec<(String, f64)>,
    pub sense: ObjectiveSense,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectiveSense {
    Minimize,
    Maximize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IlpModel {
    pub variables: Vec<IlpVariable>,
    pub constraints: Vec<LinearConstraint>,
    pub objectives: Vec<LinearObjective>,
}

impl IlpEncoder {
    pub fn new() -> Self {
        IlpEncoder {
            variables: Vec::new(),
            constraints: Vec::new(),
            objectives: Vec::new(),
            var_index: HashMap::new(),
            next_aux: 0,
        }
    }

    fn add_binary_var(&mut self, name: &str, obligation_id: Option<&str>) -> usize {
        let idx = self.variables.len();
        self.var_index.insert(name.to_string(), idx);
        self.variables.push(IlpVariable {
            name: name.to_string(),
            var_type: IlpVarType::Binary,
            lower_bound: 0.0,
            upper_bound: 1.0,
            obligation_id: obligation_id.map(|s| s.to_string()),
        });
        idx
    }

    fn add_continuous_var(&mut self, name: &str, lb: f64, ub: f64) -> usize {
        let idx = self.variables.len();
        self.var_index.insert(name.to_string(), idx);
        self.variables.push(IlpVariable {
            name: name.to_string(),
            var_type: IlpVarType::Continuous,
            lower_bound: lb,
            upper_bound: ub,
            obligation_id: None,
        });
        idx
    }

    fn aux_var(&mut self) -> String {
        self.next_aux += 1;
        let name = format!("aux_{}", self.next_aux);
        self.add_binary_var(&name, None);
        name
    }

    fn compliance_var_name(obl_id: &str) -> String {
        format!("x_{}", obl_id.replace("::", "_").replace("-", "_"))
    }

    pub fn encode_obligations(&mut self, obligations: &[RawObligation]) -> IlpModel {
        // Create binary decision variable for each obligation
        for obl in obligations {
            let var_name = Self::compliance_var_name(&obl.id);
            self.add_binary_var(&var_name, Some(&obl.id));
        }

        // Encode hard constraints: binding obligations must be satisfied
        for obl in obligations {
            if obl.is_binding && obl.kind == ObligationKind::Obligation {
                let var_name = Self::compliance_var_name(&obl.id);
                self.constraints.push(LinearConstraint {
                    name: format!("hard_{}", obl.id),
                    coefficients: vec![(var_name, 1.0)],
                    sense: ConstraintSense::GreaterEqual,
                    rhs: 1.0,
                    is_hard: true,
                });
            }
        }

        // Encode prohibitions: prohibited actions must be 0
        for obl in obligations {
            if obl.is_binding && obl.kind == ObligationKind::Prohibition {
                let var_name = Self::compliance_var_name(&obl.id);
                self.constraints.push(LinearConstraint {
                    name: format!("proh_{}", obl.id),
                    coefficients: vec![(var_name, 1.0)],
                    sense: ConstraintSense::LessEqual,
                    rhs: 0.0,
                    is_hard: true,
                });
            }
        }

        // Encode conditional implications as linear constraints
        for obl in obligations {
            for (i, cond) in obl.conditions.iter().enumerate() {
                let obl_var = Self::compliance_var_name(&obl.id);
                let cond_var = cond.condition_var.clone();
                // If condition is true, obligation must be complied with:
                // obl_var >= cond_var (if required_value is true)
                if cond.required_value {
                    self.constraints.push(LinearConstraint {
                        name: format!("cond_{}_{}", obl.id, i),
                        coefficients: vec![(obl_var, 1.0), (cond_var, -1.0)],
                        sense: ConstraintSense::GreaterEqual,
                        rhs: 0.0,
                        is_hard: obl.is_binding,
                    });
                }
            }
        }

        // Cross-reference constraints
        for obl in obligations {
            for cross_ref in &obl.cross_refs {
                let var_a = Self::compliance_var_name(&obl.id);
                let var_b = Self::compliance_var_name(cross_ref);
                if self.var_index.contains_key(&var_b) {
                    // comply(A) implies aware(B): x_A <= x_B + (1 - x_A) => 2*x_A - x_B <= 1
                    self.constraints.push(LinearConstraint {
                        name: format!("xref_{}_{}", obl.id, cross_ref),
                        coefficients: vec![(var_a, 2.0), (var_b, -1.0)],
                        sense: ConstraintSense::LessEqual,
                        rhs: 1.0,
                        is_hard: false,
                    });
                }
            }
        }

        self.build_model()
    }

    /// Build cost objectives for 4-dimensional optimization
    pub fn add_cost_objectives(&mut self, obligations: &[RawObligation], cost_data: &ObligationCostData) {
        // Objective 1: Minimize implementation cost
        let impl_coeffs: Vec<(String, f64)> = obligations.iter()
            .map(|o| {
                let var = Self::compliance_var_name(&o.id);
                let cost = cost_data.implementation_cost.get(&o.id).copied().unwrap_or(100_000.0);
                (var, cost)
            })
            .collect();
        self.objectives.push(LinearObjective {
            name: "implementation_cost".to_string(),
            coefficients: impl_coeffs,
            sense: ObjectiveSense::Minimize,
        });

        // Objective 2: Minimize time to compliance (weighted by urgency)
        let time_coeffs: Vec<(String, f64)> = obligations.iter()
            .map(|o| {
                let var = Self::compliance_var_name(&o.id);
                let time = cost_data.time_to_comply.get(&o.id).copied().unwrap_or(6.0);
                (var, time)
            })
            .collect();
        self.objectives.push(LinearObjective {
            name: "time_to_compliance".to_string(),
            coefficients: time_coeffs,
            sense: ObjectiveSense::Minimize,
        });

        // Objective 3: Minimize residual risk (non-compliance penalty)
        let risk_coeffs: Vec<(String, f64)> = obligations.iter()
            .map(|o| {
                let var = Self::compliance_var_name(&o.id);
                let risk = cost_data.residual_risk.get(&o.id).copied().unwrap_or(0.5);
                // Negate because x=1 means comply (reduce risk), x=0 means waive (incur risk)
                (var, -risk)
            })
            .collect();
        self.objectives.push(LinearObjective {
            name: "residual_risk".to_string(),
            coefficients: risk_coeffs,
            sense: ObjectiveSense::Minimize,
        });

        // Objective 4: Minimize operational burden
        let burden_coeffs: Vec<(String, f64)> = obligations.iter()
            .map(|o| {
                let var = Self::compliance_var_name(&o.id);
                let burden = cost_data.operational_burden.get(&o.id).copied().unwrap_or(10_000.0);
                (var, burden)
            })
            .collect();
        self.objectives.push(LinearObjective {
            name: "operational_burden".to_string(),
            coefficients: burden_coeffs,
            sense: ObjectiveSense::Minimize,
        });
    }

    /// Epsilon-constraint encoding: optimize one objective with bounds on others
    pub fn epsilon_constraint_model(&self, primary_objective: usize, bounds: &[(usize, f64)]) -> IlpModel {
        let mut model = self.build_model();

        // Add epsilon constraints for non-primary objectives
        for &(obj_idx, bound) in bounds {
            if obj_idx < self.objectives.len() && obj_idx != primary_objective {
                let obj = &self.objectives[obj_idx];
                model.constraints.push(LinearConstraint {
                    name: format!("eps_bound_{}", obj.name),
                    coefficients: obj.coefficients.clone(),
                    sense: if obj.sense == ObjectiveSense::Minimize { ConstraintSense::LessEqual } else { ConstraintSense::GreaterEqual },
                    rhs: bound,
                    is_hard: true,
                });
            }
        }

        // Set primary objective
        if primary_objective < self.objectives.len() {
            model.objectives = vec![self.objectives[primary_objective].clone()];
        }

        model
    }

    fn build_model(&self) -> IlpModel {
        IlpModel {
            variables: self.variables.clone(),
            constraints: self.constraints.clone(),
            objectives: self.objectives.clone(),
        }
    }

    pub fn variable_count(&self) -> usize { self.variables.len() }
    pub fn constraint_count(&self) -> usize { self.constraints.len() }
    pub fn hard_constraint_count(&self) -> usize { self.constraints.iter().filter(|c| c.is_hard).count() }
}

impl Default for IlpEncoder {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone, Default)]
pub struct ObligationCostData {
    pub implementation_cost: HashMap<String, f64>,
    pub time_to_comply: HashMap<String, f64>,
    pub residual_risk: HashMap<String, f64>,
    pub operational_burden: HashMap<String, f64>,
}

impl IlpModel {
    pub fn to_lp_format(&self) -> String {
        let mut lp = String::new();

        // Objective
        if let Some(obj) = self.objectives.first() {
            let sense = match obj.sense {
                ObjectiveSense::Minimize => "Minimize",
                ObjectiveSense::Maximize => "Maximize",
            };
            lp.push_str(&format!("{}\n  obj: ", sense));
            let terms: Vec<String> = obj.coefficients.iter()
                .map(|(var, coeff)| {
                    if *coeff >= 0.0 { format!("+ {} {}", coeff, var) }
                    else { format!("- {} {}", coeff.abs(), var) }
                })
                .collect();
            lp.push_str(&terms.join(" "));
            lp.push('\n');
        }

        // Constraints
        lp.push_str("Subject To\n");
        for c in &self.constraints {
            lp.push_str(&format!("  {}: ", c.name));
            let terms: Vec<String> = c.coefficients.iter()
                .map(|(var, coeff)| {
                    if *coeff >= 0.0 { format!("+ {} {}", coeff, var) }
                    else { format!("- {} {}", coeff.abs(), var) }
                })
                .collect();
            lp.push_str(&terms.join(" "));
            let sense_str = match c.sense {
                ConstraintSense::LessEqual => "<=",
                ConstraintSense::GreaterEqual => ">=",
                ConstraintSense::Equal => "=",
            };
            lp.push_str(&format!(" {} {}\n", sense_str, c.rhs));
        }

        // Bounds and variable types
        lp.push_str("Bounds\n");
        for v in &self.variables {
            if v.var_type != IlpVarType::Binary {
                lp.push_str(&format!("  {} <= {} <= {}\n", v.lower_bound, v.name, v.upper_bound));
            }
        }

        // Binary section
        let binary_vars: Vec<&str> = self.variables.iter()
            .filter(|v| v.var_type == IlpVarType::Binary)
            .map(|v| v.name.as_str())
            .collect();
        if !binary_vars.is_empty() {
            lp.push_str("Binary\n");
            for v in binary_vars { lp.push_str(&format!("  {}\n", v)); }
        }

        // Integer section
        let int_vars: Vec<&str> = self.variables.iter()
            .filter(|v| v.var_type == IlpVarType::Integer)
            .map(|v| v.name.as_str())
            .collect();
        if !int_vars.is_empty() {
            lp.push_str("General\n");
            for v in int_vars { lp.push_str(&format!("  {}\n", v)); }
        }

        lp.push_str("End\n");
        lp
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        let var_names: std::collections::HashSet<&str> = self.variables.iter().map(|v| v.name.as_str()).collect();

        for c in &self.constraints {
            for (var, _) in &c.coefficients {
                if !var_names.contains(var.as_str()) {
                    errors.push(format!("Constraint '{}' references unknown variable '{}'", c.name, var));
                }
            }
        }

        for v in &self.variables {
            if v.lower_bound > v.upper_bound {
                errors.push(format!("Variable '{}' has lb > ub", v.name));
            }
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obligation_encoder::ConditionSpec;

    fn make_obl(id: &str, kind: ObligationKind, binding: bool) -> RawObligation {
        RawObligation {
            id: id.to_string(), kind, jurisdiction: "EU".to_string(),
            article_ref: format!("Art.{}", id), description: format!("Test {}", id),
            is_binding: binding, risk_weight: 1.0, conditions: Vec::new(),
            exemptions: Vec::new(), cross_refs: Vec::new(),
        }
    }

    #[test]
    fn test_ilp_encoding() {
        let mut encoder = IlpEncoder::new();
        let obls = vec![
            make_obl("obl1", ObligationKind::Obligation, true),
            make_obl("obl2", ObligationKind::Permission, false),
        ];
        let model = encoder.encode_obligations(&obls);
        assert_eq!(model.variables.len(), 2);
        assert!(model.constraints.iter().any(|c| c.is_hard));
    }

    #[test]
    fn test_lp_format() {
        let mut encoder = IlpEncoder::new();
        let obls = vec![make_obl("a", ObligationKind::Obligation, true)];
        let model = encoder.encode_obligations(&obls);
        let lp = model.to_lp_format();
        assert!(lp.contains("Binary"));
        assert!(lp.contains("End"));
    }

    #[test]
    fn test_model_validation() {
        let mut encoder = IlpEncoder::new();
        let obls = vec![make_obl("v1", ObligationKind::Obligation, true)];
        let model = encoder.encode_obligations(&obls);
        assert!(model.validate().is_ok());
    }
}
