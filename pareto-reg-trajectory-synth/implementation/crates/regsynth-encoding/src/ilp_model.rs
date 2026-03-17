// ilp_model.rs — Builder API, validation, and LP-format export for `IlpModel`.

use crate::{
    IlpConstraint, IlpConstraintType, IlpModel, IlpObjective, IlpVariable, ObjectiveSense,
    Provenance,
};
use std::collections::HashSet;
use std::fmt;

// ─── IlpVar (convenience wrapper) ──────────────────────────────────────────

/// High-level variable descriptor used by the builder API before being lowered
/// into `IlpVariable`.
#[derive(Debug, Clone, PartialEq)]
pub enum IlpVarKind {
    /// Continuous variable in `[lower, upper]`.
    Continuous { lower: f64, upper: f64 },
    /// Binary variable (0 or 1).
    Binary,
    /// General integer variable in `[lower, upper]`.
    Integer { lower: f64, upper: f64 },
}

impl IlpVarKind {
    pub fn to_variable(&self, name: impl Into<String>) -> IlpVariable {
        let name = name.into();
        match self {
            IlpVarKind::Continuous { lower, upper } => IlpVariable {
                name,
                lower_bound: *lower,
                upper_bound: *upper,
                is_integer: false,
                is_binary: false,
            },
            IlpVarKind::Binary => IlpVariable {
                name,
                lower_bound: 0.0,
                upper_bound: 1.0,
                is_integer: true,
                is_binary: true,
            },
            IlpVarKind::Integer { lower, upper } => IlpVariable {
                name,
                lower_bound: *lower,
                upper_bound: *upper,
                is_integer: true,
                is_binary: false,
            },
        }
    }
}

// ─── Validation result ──────────────────────────────────────────────────────

/// Problems discovered during model validation.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationError {
    pub kind: ValidationErrorKind,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationErrorKind {
    DuplicateVariable,
    UndefinedVariable,
    EmptyObjective,
    InvalidBounds,
    EmptyConstraint,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}] {}", self.kind, self.message)
    }
}

// ─── IlpModel builder methods ───────────────────────────────────────────────

impl IlpModel {
    /// Create an empty model with a minimisation objective.
    pub fn new_minimise() -> Self {
        Self {
            variables: Vec::new(),
            constraints: Vec::new(),
            objective: IlpObjective {
                sense: ObjectiveSense::Minimize,
                coefficients: Vec::new(),
                constant: 0.0,
            },
        }
    }

    /// Create an empty model with a maximisation objective.
    pub fn new_maximise() -> Self {
        Self {
            variables: Vec::new(),
            constraints: Vec::new(),
            objective: IlpObjective {
                sense: ObjectiveSense::Maximize,
                coefficients: Vec::new(),
                constant: 0.0,
            },
        }
    }

    // ── variable management ─────────────────────────────────────────────

    /// Add a variable to the model and return its index.
    pub fn add_variable(&mut self, var: IlpVariable) -> usize {
        let idx = self.variables.len();
        self.variables.push(var);
        idx
    }

    /// Convenience: add a binary decision variable.
    pub fn add_binary_variable(&mut self, name: impl Into<String>) -> usize {
        self.add_variable(IlpVarKind::Binary.to_variable(name))
    }

    /// Convenience: add a continuous variable with the given bounds.
    pub fn add_continuous_variable(
        &mut self,
        name: impl Into<String>,
        lower: f64,
        upper: f64,
    ) -> usize {
        self.add_variable(IlpVarKind::Continuous { lower, upper }.to_variable(name))
    }

    /// Convenience: add a general-integer variable with the given bounds.
    pub fn add_integer_variable(
        &mut self,
        name: impl Into<String>,
        lower: f64,
        upper: f64,
    ) -> usize {
        self.add_variable(IlpVarKind::Integer { lower, upper }.to_variable(name))
    }

    /// Look up a variable by name.
    pub fn find_variable(&self, name: &str) -> Option<&IlpVariable> {
        self.variables.iter().find(|v| v.name == name)
    }

    // ── constraints ─────────────────────────────────────────────────────

    /// Add a constraint and return its index.
    pub fn add_constraint(&mut self, constraint: IlpConstraint) -> usize {
        let idx = self.constraints.len();
        self.constraints.push(constraint);
        idx
    }

    /// Helper to build and add a constraint from parts.
    pub fn add_linear_constraint(
        &mut self,
        id: impl Into<String>,
        coefficients: Vec<(String, f64)>,
        sense: IlpConstraintType,
        rhs: f64,
        provenance: Option<Provenance>,
    ) -> usize {
        self.add_constraint(IlpConstraint {
            id: id.into(),
            coefficients,
            constraint_type: sense,
            rhs,
            provenance,
        })
    }

    // ── objective ───────────────────────────────────────────────────────

    /// Set (replace) the model objective.
    pub fn set_objective(&mut self, objective: IlpObjective) {
        self.objective = objective;
    }

    /// Add a term to the current objective.
    pub fn add_objective_term(&mut self, var_name: impl Into<String>, coefficient: f64) {
        self.objective
            .coefficients
            .push((var_name.into(), coefficient));
    }

    /// Set the objective constant.
    pub fn set_objective_constant(&mut self, constant: f64) {
        self.objective.constant = constant;
    }

    // ── statistics ──────────────────────────────────────────────────────

    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    pub fn binary_variable_count(&self) -> usize {
        self.variables.iter().filter(|v| v.is_binary).count()
    }

    pub fn integer_variable_count(&self) -> usize {
        self.variables
            .iter()
            .filter(|v| v.is_integer && !v.is_binary)
            .count()
    }

    pub fn continuous_variable_count(&self) -> usize {
        self.variables.iter().filter(|v| !v.is_integer).count()
    }

    // ── validation ──────────────────────────────────────────────────────

    /// Validate the model, returning any detected problems.
    pub fn validate(&self) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        let mut seen_names: HashSet<&str> = HashSet::new();

        // Duplicate variable names.
        for v in &self.variables {
            if !seen_names.insert(&v.name) {
                errors.push(ValidationError {
                    kind: ValidationErrorKind::DuplicateVariable,
                    message: format!("Duplicate variable name: {}", v.name),
                });
            }
        }

        // Invalid bounds.
        for v in &self.variables {
            if v.lower_bound > v.upper_bound {
                errors.push(ValidationError {
                    kind: ValidationErrorKind::InvalidBounds,
                    message: format!(
                        "Variable {} has lower_bound ({}) > upper_bound ({})",
                        v.name, v.lower_bound, v.upper_bound
                    ),
                });
            }
        }

        // Constraints referencing undefined variables.
        for c in &self.constraints {
            if c.coefficients.is_empty() {
                errors.push(ValidationError {
                    kind: ValidationErrorKind::EmptyConstraint,
                    message: format!("Constraint {} has no coefficients", c.id),
                });
            }
            for (var_name, _) in &c.coefficients {
                if !seen_names.contains(var_name.as_str()) {
                    errors.push(ValidationError {
                        kind: ValidationErrorKind::UndefinedVariable,
                        message: format!(
                            "Constraint {} references undefined variable {}",
                            c.id, var_name
                        ),
                    });
                }
            }
        }

        // Objective referencing undefined variables.
        for (var_name, _) in &self.objective.coefficients {
            if !seen_names.contains(var_name.as_str()) {
                errors.push(ValidationError {
                    kind: ValidationErrorKind::UndefinedVariable,
                    message: format!("Objective references undefined variable {}", var_name),
                });
            }
        }

        errors
    }

    // ── LP format export ────────────────────────────────────────────────

    /// Produce an LP-format string (compatible with CPLEX / Gurobi / GLPK).
    pub fn to_lp_format_string(&self) -> String {
        let mut out = String::new();

        // Objective
        let sense_str = match self.objective.sense {
            ObjectiveSense::Minimize => "Minimize",
            ObjectiveSense::Maximize => "Maximize",
        };
        out.push_str(&format!("\\\\  LP model generated by regsynth-encoding\n"));
        out.push_str(&format!("{}\n", sense_str));
        out.push_str("  obj: ");
        out.push_str(&format_linear_expr(&self.objective.coefficients, self.objective.constant));
        out.push('\n');

        // Constraints
        out.push_str("Subject To\n");
        for c in &self.constraints {
            out.push_str(&format!("  {}: ", c.id));
            out.push_str(&format_linear_expr(&c.coefficients, 0.0));
            out.push_str(&format!(" {} {}\n", c.constraint_type, c.rhs));
        }

        // Bounds
        out.push_str("Bounds\n");
        for v in &self.variables {
            if v.is_binary {
                // Binaries are declared in the Binary section.
                continue;
            }
            if v.lower_bound == f64::NEG_INFINITY && v.upper_bound == f64::INFINITY {
                out.push_str(&format!("  {} free\n", v.name));
            } else if v.lower_bound == f64::NEG_INFINITY {
                out.push_str(&format!("  -inf <= {} <= {}\n", v.name, v.upper_bound));
            } else if v.upper_bound == f64::INFINITY {
                out.push_str(&format!("  {} <= {} <= +inf\n", v.lower_bound, v.name));
            } else {
                out.push_str(&format!(
                    "  {} <= {} <= {}\n",
                    v.lower_bound, v.name, v.upper_bound
                ));
            }
        }

        // General integers
        let generals: Vec<&IlpVariable> = self
            .variables
            .iter()
            .filter(|v| v.is_integer && !v.is_binary)
            .collect();
        if !generals.is_empty() {
            out.push_str("General\n");
            for v in generals {
                out.push_str(&format!("  {}\n", v.name));
            }
        }

        // Binaries
        let binaries: Vec<&IlpVariable> = self.variables.iter().filter(|v| v.is_binary).collect();
        if !binaries.is_empty() {
            out.push_str("Binary\n");
            for v in binaries {
                out.push_str(&format!("  {}\n", v.name));
            }
        }

        out.push_str("End\n");
        out
    }

    /// Merge another model into this one. Constraints, variables, and objective
    /// terms are appended; the objective sense of `self` is preserved.
    pub fn merge(&mut self, other: &IlpModel) {
        let existing: HashSet<String> =
            self.variables.iter().map(|v| v.name.clone()).collect();
        for v in &other.variables {
            if !existing.contains(&v.name) {
                self.variables.push(v.clone());
            }
        }
        self.constraints
            .extend(other.constraints.iter().cloned());
        self.objective
            .coefficients
            .extend(other.objective.coefficients.iter().cloned());
        self.objective.constant += other.objective.constant;
    }
}

// ─── private formatting helper ──────────────────────────────────────────────

fn format_linear_expr(coefficients: &[(String, f64)], constant: f64) -> String {
    let mut parts = Vec::new();
    for (i, (var, coeff)) in coefficients.iter().enumerate() {
        if i == 0 {
            if *coeff == 1.0 {
                parts.push(var.clone());
            } else if *coeff == -1.0 {
                parts.push(format!("- {}", var));
            } else {
                parts.push(format!("{} {}", coeff, var));
            }
        } else if *coeff > 0.0 {
            if *coeff == 1.0 {
                parts.push(format!("+ {}", var));
            } else {
                parts.push(format!("+ {} {}", coeff, var));
            }
        } else if *coeff < 0.0 {
            if *coeff == -1.0 {
                parts.push(format!("- {}", var));
            } else {
                parts.push(format!("- {} {}", -coeff, var));
            }
        }
    }
    if constant != 0.0 {
        if parts.is_empty() {
            parts.push(format!("{}", constant));
        } else if constant > 0.0 {
            parts.push(format!("+ {}", constant));
        } else {
            parts.push(format!("- {}", -constant));
        }
    }
    if parts.is_empty() {
        "0".into()
    } else {
        parts.join(" ")
    }
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_model() -> IlpModel {
        let mut m = IlpModel::new_minimise();
        m.add_binary_variable("x1");
        m.add_binary_variable("x2");
        m.add_continuous_variable("cost", 0.0, 1000.0);
        m.add_linear_constraint(
            "c1",
            vec![("x1".into(), 1.0), ("x2".into(), 1.0)],
            IlpConstraintType::Ge,
            1.0,
            None,
        );
        m.add_linear_constraint(
            "c2",
            vec![("cost".into(), 1.0), ("x1".into(), -50.0)],
            IlpConstraintType::Le,
            200.0,
            None,
        );
        m.add_objective_term("cost", 1.0);
        m
    }

    #[test]
    fn test_variable_counts() {
        let m = sample_model();
        assert_eq!(m.variable_count(), 3);
        assert_eq!(m.binary_variable_count(), 2);
        assert_eq!(m.continuous_variable_count(), 1);
        assert_eq!(m.constraint_count(), 2);
    }

    #[test]
    fn test_validate_clean() {
        let m = sample_model();
        assert!(m.validate().is_empty());
    }

    #[test]
    fn test_validate_duplicate_variable() {
        let mut m = IlpModel::new_minimise();
        m.add_binary_variable("x");
        m.add_binary_variable("x");
        let errors = m.validate();
        assert!(errors
            .iter()
            .any(|e| e.kind == ValidationErrorKind::DuplicateVariable));
    }

    #[test]
    fn test_validate_undefined_variable_in_constraint() {
        let mut m = IlpModel::new_minimise();
        m.add_binary_variable("x");
        m.add_linear_constraint(
            "c1",
            vec![("y".into(), 1.0)],
            IlpConstraintType::Le,
            5.0,
            None,
        );
        let errors = m.validate();
        assert!(errors
            .iter()
            .any(|e| e.kind == ValidationErrorKind::UndefinedVariable));
    }

    #[test]
    fn test_validate_invalid_bounds() {
        let mut m = IlpModel::new_minimise();
        m.add_variable(IlpVariable {
            name: "bad".into(),
            lower_bound: 10.0,
            upper_bound: 5.0,
            is_integer: false,
            is_binary: false,
        });
        let errors = m.validate();
        assert!(errors
            .iter()
            .any(|e| e.kind == ValidationErrorKind::InvalidBounds));
    }

    #[test]
    fn test_lp_format_contains_sections() {
        let m = sample_model();
        let lp = m.to_lp_format_string();
        assert!(lp.contains("Minimize"));
        assert!(lp.contains("Subject To"));
        assert!(lp.contains("Bounds"));
        assert!(lp.contains("Binary"));
        assert!(lp.contains("End"));
        assert!(lp.contains("x1"));
        assert!(lp.contains("cost"));
    }

    #[test]
    fn test_lp_format_constraint_output() {
        let m = sample_model();
        let lp = m.to_lp_format_string();
        // c1: x1 + x2 >= 1
        assert!(lp.contains("c1:"));
        assert!(lp.contains(">= 1"));
    }

    #[test]
    fn test_merge() {
        let mut m1 = IlpModel::new_minimise();
        m1.add_binary_variable("a");
        m1.add_objective_term("a", 1.0);

        let mut m2 = IlpModel::new_minimise();
        m2.add_binary_variable("b");
        m2.add_linear_constraint(
            "c",
            vec![("b".into(), 1.0)],
            IlpConstraintType::Le,
            1.0,
            None,
        );
        m2.add_objective_term("b", 2.0);

        m1.merge(&m2);
        assert_eq!(m1.variable_count(), 2);
        assert_eq!(m1.constraint_count(), 1);
        assert_eq!(m1.objective.coefficients.len(), 2);
    }

    #[test]
    fn test_find_variable() {
        let m = sample_model();
        assert!(m.find_variable("x1").is_some());
        assert!(m.find_variable("nonexistent").is_none());
    }

    #[test]
    fn test_new_maximise() {
        let m = IlpModel::new_maximise();
        assert_eq!(m.objective.sense, ObjectiveSense::Maximize);
    }

    #[test]
    fn test_format_linear_expr_negative_coeffs() {
        let s = format_linear_expr(
            &[("x".into(), -3.0), ("y".into(), 2.0)],
            0.0,
        );
        assert!(s.contains("- 3 x"));
        assert!(s.contains("+ 2 y"));
    }

    #[test]
    fn test_empty_objective_format() {
        let s = format_linear_expr(&[], 0.0);
        assert_eq!(s, "0");
    }
}
