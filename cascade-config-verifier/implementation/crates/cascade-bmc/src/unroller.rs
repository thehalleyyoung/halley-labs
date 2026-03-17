//! BMC Unrolling: constructs the unrolled transition system formula.

use cascade_graph::rtig::RtigGraph;
use cascade_types::service::ServiceId;
use cascade_types::smt::{SmtConstraint, SmtExpr, SmtFormula, SmtSort, SmtVariable};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::encoder::BmcEncoder;

// ---------------------------------------------------------------------------
// UnrolledFormula
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnrolledFormula {
    pub constraints: Vec<SmtConstraint>,
    pub variables: Vec<SmtVariable>,
    pub depth: usize,
    pub metadata: UnrollingMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnrollingMetadata {
    pub service_count: usize,
    pub edge_count: usize,
    pub total_variables: usize,
    pub total_constraints: usize,
    pub cone_applied: bool,
    pub cone_services: Vec<String>,
}

impl UnrolledFormula {
    pub fn new(depth: usize) -> Self {
        Self {
            constraints: Vec::new(),
            variables: Vec::new(),
            depth,
            metadata: UnrollingMetadata {
                service_count: 0,
                edge_count: 0,
                total_variables: 0,
                total_constraints: 0,
                cone_applied: false,
                cone_services: Vec::new(),
            },
        }
    }

    pub fn add_constraint(&mut self, c: SmtConstraint) {
        self.constraints.push(c);
        self.metadata.total_constraints = self.constraints.len();
    }

    pub fn add_variable(&mut self, v: SmtVariable) {
        self.variables.push(v);
        self.metadata.total_variables = self.variables.len();
    }

    pub fn merge(&mut self, other: UnrolledFormula) {
        self.constraints.extend(other.constraints);
        self.variables.extend(other.variables);
        self.metadata.total_constraints = self.constraints.len();
        self.metadata.total_variables = self.variables.len();
    }
}

// ---------------------------------------------------------------------------
// BmcUnroller
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BmcUnroller {
    pub graph: RtigGraph,
}

impl BmcUnroller {
    pub fn new(graph: RtigGraph) -> Self {
        Self { graph }
    }

    /// Full unrolling to the given depth.
    pub fn unroll(&self, depth: usize, failure_budget: usize) -> UnrolledFormula {
        let mut encoder = BmcEncoder::new(self.graph.clone(), depth, failure_budget);
        let mut formula = SmtFormula {
            declarations: Vec::new(),
            constraints: Vec::new(),
        };

        let init = encoder.encode_initial_conditions(&mut formula);
        let mut constraints = init;

        for t in 0..depth {
            let trans = encoder.encode_transition_relation(&mut formula, t);
            constraints.extend(trans);
        }

        let budget = encoder.encode_failure_budget(failure_budget);
        constraints.push(budget);

        let mut result = UnrolledFormula::new(depth);
        result.variables = formula.declarations;
        result.constraints = constraints;
        result.metadata.service_count = self.graph.service_count();
        result.metadata.edge_count = self.graph.edge_count();
        result.metadata.total_variables = result.variables.len();
        result.metadata.total_constraints = result.constraints.len();
        result
    }

    /// Unroll with cone-of-influence reduction: only include services reachable
    /// backwards from the target.
    pub fn unroll_with_cone_of_influence(
        &self,
        target: &str,
        depth: usize,
        failure_budget: usize,
    ) -> UnrolledFormula {
        let cone = ConeOfInfluence::compute_cone(&self.graph, target);
        let cone_ids: Vec<ServiceId> = cone.iter().map(|s| ServiceId::from(s.as_str())).collect();
        let subgraph = self.graph.subgraph(&cone_ids);

        let mut encoder = BmcEncoder::new(subgraph.clone(), depth, failure_budget);
        let mut formula = SmtFormula {
            declarations: Vec::new(),
            constraints: Vec::new(),
        };

        let init = encoder.encode_initial_conditions(&mut formula);
        let mut constraints = init;

        for t in 0..depth {
            let trans = encoder.encode_transition_relation(&mut formula, t);
            constraints.extend(trans);
        }

        let budget = encoder.encode_failure_budget(failure_budget);
        constraints.push(budget);

        let mut result = UnrolledFormula::new(depth);
        result.variables = formula.declarations;
        result.constraints = constraints;
        result.metadata.service_count = subgraph.service_count();
        result.metadata.edge_count = subgraph.edge_count();
        result.metadata.total_variables = result.variables.len();
        result.metadata.total_constraints = result.constraints.len();
        result.metadata.cone_applied = true;
        result.metadata.cone_services = cone.into_iter().collect();
        result
    }

    /// Incrementally extend an existing unrolling by one step.
    pub fn incremental_unroll(
        &self,
        existing: &UnrolledFormula,
        failure_budget: usize,
    ) -> UnrolledFormula {
        let new_depth = existing.depth + 1;
        let t = existing.depth;

        let mut encoder = BmcEncoder::new(self.graph.clone(), new_depth, failure_budget);
        let mut formula = SmtFormula {
            declarations: Vec::new(),
            constraints: Vec::new(),
        };

        // We only need to encode the new transition step t -> t+1
        // First, we need to register existing variables (simplified: re-encode init for variable
        // declarations, but only add the new transition)
        let _ = encoder.encode_initial_conditions(&mut formula);
        for step in 0..t {
            let _ = encoder.encode_transition_relation(&mut formula, step);
        }

        let new_trans = encoder.encode_transition_relation(&mut formula, t);

        let mut result = existing.clone();
        result.depth = new_depth;

        // Add only the new transition constraints and variables
        for c in new_trans {
            result.add_constraint(c);
        }

        // Add new variables that were declared
        let existing_var_names: HashSet<String> = result.variables.iter().map(|v| v.name.clone()).collect();
        for v in formula.declarations {
            if !existing_var_names.contains(&v.name) {
                result.add_variable(v);
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// ConeOfInfluence
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ConeOfInfluence;

impl ConeOfInfluence {
    /// Compute the set of services that can influence the target service.
    /// This is the backward reachable set from the target.
    pub fn compute_cone(graph: &RtigGraph, target: &str) -> HashSet<String> {
        graph.reverse_reachable(target)
    }

    /// Prune variables from a formula that are not in the cone of influence.
    pub fn prune_irrelevant_variables(
        formula: &UnrolledFormula,
        cone: &HashSet<String>,
    ) -> UnrolledFormula {
        let mut pruned = UnrolledFormula::new(formula.depth);

        // Keep variables whose name contains a service in the cone
        for var in &formula.variables {
            let keep = cone.iter().any(|svc| var.name.contains(svc));
            if keep {
                pruned.add_variable(var.clone());
            }
        }

        // Keep constraints that only reference variables in the cone
        let var_names: HashSet<String> = pruned.variables.iter().map(|v| v.name.clone()).collect();
        for constraint in &formula.constraints {
            let vars_in_constraint = collect_vars_from_expr(&constraint.expr);
            let all_in_cone = vars_in_constraint.iter().all(|v| {
                var_names.contains(v) || v.starts_with("failed_")
            });
            if all_in_cone || vars_in_constraint.is_empty() {
                pruned.add_constraint(constraint.clone());
            }
        }

        pruned.metadata = formula.metadata.clone();
        pruned.metadata.cone_applied = true;
        pruned.metadata.total_variables = pruned.variables.len();
        pruned.metadata.total_constraints = pruned.constraints.len();
        pruned
    }
}

fn collect_vars_from_expr(expr: &SmtExpr) -> Vec<String> {
    let mut vars = Vec::new();
    collect_vars_recursive(expr, &mut vars);
    vars
}

fn collect_vars_recursive(expr: &SmtExpr, vars: &mut Vec<String>) {
    match expr {
        SmtExpr::Var(v) => vars.push(v.clone()),
        SmtExpr::IntConst(_) | SmtExpr::BoolConst(_) | SmtExpr::RealConst(_) => {}
        SmtExpr::Not(e) => collect_vars_recursive(e, vars),
        SmtExpr::And(es) | SmtExpr::Or(es) => {
            for e in es {
                collect_vars_recursive(e, vars);
            }
        }
        SmtExpr::Implies(a, b) | SmtExpr::Eq(a, b)
        | SmtExpr::Lt(a, b) | SmtExpr::Le(a, b)
        | SmtExpr::Gt(a, b) | SmtExpr::Ge(a, b)
        | SmtExpr::Add(a, b) | SmtExpr::Sub(a, b)
        | SmtExpr::Mul(a, b) | SmtExpr::Div(a, b) => {
            collect_vars_recursive(a, vars);
            collect_vars_recursive(b, vars);
        }
        SmtExpr::Ite(c, t, e) => {
            collect_vars_recursive(c, vars);
            collect_vars_recursive(t, vars);
            collect_vars_recursive(e, vars);
        }
    }
}

// ---------------------------------------------------------------------------
// DepthBoundComputer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DepthBoundComputer;

impl DepthBoundComputer {
    /// Compute the completeness bound: d* = diameter × max_retries.
    /// Any cascade of length > d* must traverse a cycle and is thus subsumed.
    pub fn compute_completeness_bound(graph: &RtigGraph) -> usize {
        let diameter = graph.diameter();
        let max_retries = graph.max_retries() as usize;
        // d* = diameter * (1 + max_retries) to account for retry-induced
        // re-traversals of paths
        diameter * (1 + max_retries)
    }

    /// Target-specific bound: only consider paths that reach the target.
    pub fn compute_tight_bound(graph: &RtigGraph, target: &str) -> usize {
        let longest = graph.longest_path_to(target);
        let max_retries = graph.max_retries() as usize;
        longest * (1 + max_retries)
    }

    /// Compute a per-service depth bound based on the maximum path length
    /// to each service weighted by retry counts.
    pub fn compute_per_service_bounds(graph: &RtigGraph) -> HashMap<String, usize> {
        let mut bounds = HashMap::new();
        for sid in graph.service_ids() {
            let longest = graph.longest_path_to(sid);
            // Accumulate max retry along the longest path
            let max_retries = graph.incoming_edges(sid)
                .iter()
                .map(|e| e.retry_count as usize)
                .max()
                .unwrap_or(0);
            bounds.insert(sid.to_owned(), longest * (1 + max_retries));
        }
        bounds
    }
}

// ---------------------------------------------------------------------------
// UnrollingOptimizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct UnrollingOptimizer;

impl UnrollingOptimizer {
    /// Merge equivalent states across timesteps.
    /// If state[v,t] = state[v,t-1] for all v, the formula can skip that step.
    pub fn optimize(formula: &UnrolledFormula) -> UnrolledFormula {
        // Conservative optimization: remove constraints with trivially true expressions
        let mut optimized = UnrolledFormula::new(formula.depth);
        optimized.variables = formula.variables.clone();

        for constraint in &formula.constraints {
            if !Self::is_trivially_true(&constraint.expr) {
                optimized.add_constraint(constraint.clone());
            }
        }

        optimized.metadata = formula.metadata.clone();
        optimized.metadata.total_constraints = optimized.constraints.len();
        optimized
    }

    fn is_trivially_true(expr: &SmtExpr) -> bool {
        match expr {
            SmtExpr::BoolConst(true) => true,
            SmtExpr::Eq(a, b) if a == b => true,
            SmtExpr::Le(a, b) if a == b => true,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// VariableReduction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct VariableReduction;

impl VariableReduction {
    /// Eliminate variables via equality propagation.
    /// If we find `x = expr` where expr has no x, replace all occurrences of x.
    pub fn reduce(formula: &UnrolledFormula) -> UnrolledFormula {
        let mut equalities: HashMap<String, SmtExpr> = HashMap::new();

        // Find simple equalities: name = expr
        for constraint in &formula.constraints {
            if let SmtExpr::Eq(lhs, rhs) = &constraint.expr {
                if let SmtExpr::Var(name) = lhs.as_ref() {
                    let rhs_vars = collect_vars_from_expr(rhs);
                    if !rhs_vars.contains(name) {
                        equalities.insert(name.clone(), *rhs.clone());
                    }
                }
                if let SmtExpr::Var(name) = rhs.as_ref() {
                    let lhs_vars = collect_vars_from_expr(lhs);
                    if !lhs_vars.contains(name) {
                        equalities.insert(name.clone(), *lhs.clone());
                    }
                }
            }
        }

        if equalities.is_empty() {
            return formula.clone();
        }

        let mut reduced = UnrolledFormula::new(formula.depth);

        // Remove eliminated variables
        let eliminated: HashSet<String> = equalities.keys().cloned().collect();
        for var in &formula.variables {
            if !eliminated.contains(&var.name) {
                reduced.add_variable(var.clone());
            }
        }

        // Substitute in constraints
        for constraint in &formula.constraints {
            // Skip the equality constraints we used for elimination
            if let SmtExpr::Eq(lhs, _) = &constraint.expr {
                if let SmtExpr::Var(name) = lhs.as_ref() {
                    if eliminated.contains(name) {
                        continue;
                    }
                }
            }
            let mut new_expr = constraint.expr.clone();
            for (var, replacement) in &equalities {
                new_expr = substitute_expr(&new_expr, var, replacement);
            }
            reduced.add_constraint(SmtConstraint {
                name: constraint.name.clone(),
                expr: new_expr,
                source: None,
            });
        }

        reduced.metadata = formula.metadata.clone();
        reduced.metadata.total_variables = reduced.variables.len();
        reduced.metadata.total_constraints = reduced.constraints.len();
        reduced
    }
}

fn substitute_expr(expr: &SmtExpr, var: &str, replacement: &SmtExpr) -> SmtExpr {
    match expr {
        SmtExpr::Var(v) if v == var => replacement.clone(),
        SmtExpr::Var(_) | SmtExpr::IntConst(_) | SmtExpr::BoolConst(_) | SmtExpr::RealConst(_) => expr.clone(),
        SmtExpr::Not(e) => SmtExpr::Not(Box::new(substitute_expr(e, var, replacement))),
        SmtExpr::And(es) => SmtExpr::And(es.iter().map(|e| substitute_expr(e, var, replacement)).collect()),
        SmtExpr::Or(es) => SmtExpr::Or(es.iter().map(|e| substitute_expr(e, var, replacement)).collect()),
        SmtExpr::Implies(a, b) => SmtExpr::Implies(
            Box::new(substitute_expr(a, var, replacement)),
            Box::new(substitute_expr(b, var, replacement)),
        ),
        SmtExpr::Eq(a, b) => SmtExpr::Eq(
            Box::new(substitute_expr(a, var, replacement)),
            Box::new(substitute_expr(b, var, replacement)),
        ),
        SmtExpr::Lt(a, b) => SmtExpr::Lt(
            Box::new(substitute_expr(a, var, replacement)),
            Box::new(substitute_expr(b, var, replacement)),
        ),
        SmtExpr::Le(a, b) => SmtExpr::Le(
            Box::new(substitute_expr(a, var, replacement)),
            Box::new(substitute_expr(b, var, replacement)),
        ),
        SmtExpr::Add(a, b) => SmtExpr::Add(
            Box::new(substitute_expr(a, var, replacement)),
            Box::new(substitute_expr(b, var, replacement)),
        ),
        SmtExpr::Sub(a, b) => SmtExpr::Sub(
            Box::new(substitute_expr(a, var, replacement)),
            Box::new(substitute_expr(b, var, replacement)),
        ),
        SmtExpr::Mul(a, b) => SmtExpr::Mul(
            Box::new(substitute_expr(a, var, replacement)),
            Box::new(substitute_expr(b, var, replacement)),
        ),
        SmtExpr::Div(a, b) => SmtExpr::Div(
            Box::new(substitute_expr(a, var, replacement)),
            Box::new(substitute_expr(b, var, replacement)),
        ),
        SmtExpr::Gt(a, b) => SmtExpr::Gt(
            Box::new(substitute_expr(a, var, replacement)),
            Box::new(substitute_expr(b, var, replacement)),
        ),
        SmtExpr::Ge(a, b) => SmtExpr::Ge(
            Box::new(substitute_expr(a, var, replacement)),
            Box::new(substitute_expr(b, var, replacement)),
        ),
        SmtExpr::Ite(c, t, e) => SmtExpr::Ite(
            Box::new(substitute_expr(c, var, replacement)),
            Box::new(substitute_expr(t, var, replacement)),
            Box::new(substitute_expr(e, var, replacement)),
        ),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cascade_graph::rtig::{DependencyEdgeInfo, RtigGraph, ServiceNode};

    fn simple_graph() -> RtigGraph {
        let mut g = RtigGraph::new();
        g.add_service(ServiceNode::new("gateway", 1000).with_baseline_load(100));
        g.add_service(ServiceNode::new("auth", 500).with_baseline_load(50));
        g.add_service(ServiceNode::new("db", 200).with_baseline_load(30));
        g.add_edge(DependencyEdgeInfo::new("gateway", "auth").with_retry_count(3));
        g.add_edge(DependencyEdgeInfo::new("auth", "db").with_retry_count(2));
        g
    }

    #[test]
    fn test_basic_unroll() {
        let g = simple_graph();
        let unroller = BmcUnroller::new(g);
        let formula = unroller.unroll(3, 1);
        assert_eq!(formula.depth, 3);
        assert!(formula.variables.len() > 0);
        assert!(formula.constraints.len() > 0);
    }

    #[test]
    fn test_unroll_depth_zero() {
        let g = simple_graph();
        let unroller = BmcUnroller::new(g);
        let formula = unroller.unroll(0, 1);
        assert_eq!(formula.depth, 0);
        // Should have initial conditions + budget only
        assert!(formula.constraints.len() > 0);
    }

    #[test]
    fn test_cone_of_influence() {
        let g = simple_graph();
        let cone = ConeOfInfluence::compute_cone(&g, "db");
        assert!(cone.contains("db"));
        assert!(cone.contains("auth"));
        assert!(cone.contains("gateway"));
    }

    #[test]
    fn test_cone_of_influence_leaf() {
        let g = simple_graph();
        let cone = ConeOfInfluence::compute_cone(&g, "gateway");
        assert_eq!(cone.len(), 1);
        assert!(cone.contains("gateway"));
    }

    #[test]
    fn test_unroll_with_cone() {
        let g = simple_graph();
        let unroller = BmcUnroller::new(g);
        let formula = unroller.unroll_with_cone_of_influence("db", 2, 1);
        assert!(formula.metadata.cone_applied);
        assert!(formula.metadata.cone_services.len() > 0);
    }

    #[test]
    fn test_incremental_unroll() {
        let g = simple_graph();
        let unroller = BmcUnroller::new(g);
        let base = unroller.unroll(2, 1);
        let extended = unroller.incremental_unroll(&base, 1);
        assert_eq!(extended.depth, 3);
        assert!(extended.constraints.len() > base.constraints.len());
    }

    #[test]
    fn test_depth_bound_completeness() {
        let g = simple_graph();
        let bound = DepthBoundComputer::compute_completeness_bound(&g);
        // diameter=2, max_retries=3 => 2*(1+3) = 8
        assert_eq!(bound, 8);
    }

    #[test]
    fn test_depth_bound_tight() {
        let g = simple_graph();
        let bound = DepthBoundComputer::compute_tight_bound(&g, "db");
        // longest_path_to(db)=2, max_retries=3 => 2*(1+3) = 8
        assert_eq!(bound, 8);
    }

    #[test]
    fn test_depth_bound_gateway() {
        let g = simple_graph();
        let bound = DepthBoundComputer::compute_tight_bound(&g, "gateway");
        // longest_path_to(gateway)=0 => 0
        assert_eq!(bound, 0);
    }

    #[test]
    fn test_per_service_bounds() {
        let g = simple_graph();
        let bounds = DepthBoundComputer::compute_per_service_bounds(&g);
        assert!(bounds.contains_key("db"));
        assert!(bounds.contains_key("gateway"));
    }

    #[test]
    fn test_unrolling_optimizer_trivial() {
        let mut formula = UnrolledFormula::new(1);
        formula.add_constraint(SmtConstraint {
            name: Some("trivial".to_owned()),
            expr: SmtExpr::BoolConst(true),
            source: None,
        });
        formula.add_constraint(SmtConstraint {
            name: Some("real".to_owned()),
            expr: SmtExpr::Gt(
                Box::new(SmtExpr::Var("x".into())),
                Box::new(SmtExpr::IntConst(0)),
            ),
            source: None,
        });
        let optimized = UnrollingOptimizer::optimize(&formula);
        assert_eq!(optimized.constraints.len(), 1);
    }

    #[test]
    fn test_variable_reduction() {
        let mut formula = UnrolledFormula::new(1);
        formula.add_variable(SmtVariable::new("x".into(), SmtSort::Int));
        formula.add_variable(SmtVariable::new("y".into(), SmtSort::Int));
        // x = 5
        formula.add_constraint(SmtConstraint {
            name: Some("eq".into()),
            expr: SmtExpr::Eq(Box::new(SmtExpr::Var("x".into())), Box::new(SmtExpr::IntConst(5))),
            source: None,
        });
        // y > x
        formula.add_constraint(SmtConstraint {
            name: Some("bound".into()),
            expr: SmtExpr::Gt(
                Box::new(SmtExpr::Var("y".into())),
                Box::new(SmtExpr::Var("x".into())),
            ),
            source: None,
        });
        let reduced = VariableReduction::reduce(&formula);
        // x should be eliminated
        assert!(reduced.variables.len() < formula.variables.len());
    }

    #[test]
    fn test_unrolled_formula_merge() {
        let mut f1 = UnrolledFormula::new(1);
        f1.add_variable(SmtVariable::new("a".into(), SmtSort::Int));
        let mut f2 = UnrolledFormula::new(2);
        f2.add_variable(SmtVariable::new("b".into(), SmtSort::Int));
        f1.merge(f2);
        assert_eq!(f1.variables.len(), 2);
    }

    #[test]
    fn test_collect_vars_from_expr() {
        let expr = SmtExpr::Add(
            Box::new(SmtExpr::Var("x".into())),
            Box::new(SmtExpr::Var("y".into())),
        );
        let vars = collect_vars_from_expr(&expr);
        assert!(vars.contains(&"x".to_string()));
        assert!(vars.contains(&"y".to_string()));
    }

    #[test]
    fn test_substitute_expr() {
        let expr = SmtExpr::Add(
            Box::new(SmtExpr::Var("x".into())),
            Box::new(SmtExpr::IntConst(1)),
        );
        let result = substitute_expr(&expr, "x", &SmtExpr::IntConst(5));
        assert_eq!(result, SmtExpr::Add(
            Box::new(SmtExpr::IntConst(5)),
            Box::new(SmtExpr::IntConst(1)),
        ));
    }
}
