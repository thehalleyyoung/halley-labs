//! BMC QF_LIA Encoding for cascade verification.
//!
//! Encodes the cascade-failure analysis problem as a quantifier-free linear
//! integer arithmetic (QF_LIA) formula suitable for SMT solving.

use cascade_graph::rtig::{DependencyEdgeInfo, RtigGraph, ServiceNode};
use cascade_types::service::ServiceHealth;
use cascade_types::smt::{SmtConstraint, SmtExpr, SmtFormula, SmtSort, SmtVariable};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

// ---------------------------------------------------------------------------
// SmtVariableMap – maps SMT variable names back to services and timesteps
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SmtVariableMap {
    entries: IndexMap<String, VarMapEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarMapEntry {
    pub service: String,
    pub attribute: VarAttribute,
    pub time_step: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VarAttribute {
    State,
    Load,
    Retry,
    Timeout,
    Failed,
    Degraded,
    BackoffDelay,
}

impl SmtVariableMap {
    pub fn new() -> Self {
        Self { entries: IndexMap::new() }
    }

    pub fn insert(&mut self, var_name: String, entry: VarMapEntry) {
        self.entries.insert(var_name, entry);
    }

    pub fn get(&self, var_name: &str) -> Option<&VarMapEntry> {
        self.entries.get(var_name)
    }

    pub fn service_vars(&self, service: &str) -> Vec<(&String, &VarMapEntry)> {
        self.entries
            .iter()
            .filter(|(_, e)| e.service == service)
            .collect()
    }

    pub fn timestep_vars(&self, t: usize) -> Vec<(&String, &VarMapEntry)> {
        self.entries
            .iter()
            .filter(|(_, e)| e.time_step == Some(t))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Variable naming helpers
// ---------------------------------------------------------------------------

fn state_var(service: &str, t: usize) -> String {
    format!("state_{service}_{t}")
}

fn load_var(service: &str, t: usize) -> String {
    format!("load_{service}_{t}")
}

fn retry_var(service: &str, t: usize) -> String {
    format!("retry_{service}_{t}")
}

fn timeout_var(service: &str, t: usize) -> String {
    format!("timeout_{service}_{t}")
}

fn failed_var(service: &str) -> String {
    format!("failed_{service}")
}

fn backoff_var(service: &str, t: usize) -> String {
    format!("backoff_{service}_{t}")
}

// State encoding: 0 = healthy, 1 = degraded, 2 = unavailable
fn health_to_int(h: ServiceHealth) -> i64 {
    match h {
        ServiceHealth::Healthy => 0,
        ServiceHealth::Degraded => 1,
        ServiceHealth::Unavailable => 2,
    }
}

// ---------------------------------------------------------------------------
// BmcEncoder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BmcEncoder {
    pub graph: RtigGraph,
    pub depth_bound: usize,
    pub failure_budget: usize,
    var_map: SmtVariableMap,
}

impl BmcEncoder {
    pub fn new(graph: RtigGraph, depth_bound: usize, failure_budget: usize) -> Self {
        Self {
            graph,
            depth_bound,
            failure_budget,
            var_map: SmtVariableMap::new(),
        }
    }

    pub fn variable_map(&self) -> &SmtVariableMap {
        &self.var_map
    }

    /// Register a variable in both the formula and the variable map.
    fn register_var(
        &mut self,
        formula: &mut SmtFormula,
        name: &str,
        sort: SmtSort,
        service: &str,
        attr: VarAttribute,
        time_step: Option<usize>,
    ) {
        formula.add_declaration(SmtVariable::new(name.to_owned(), sort));
        self.var_map.insert(
            name.to_owned(),
            VarMapEntry {
                service: service.to_owned(),
                attribute: attr,
                time_step,
            },
        );
    }

    // -----------------------------------------------------------------------
    // Initial conditions  (t = 0)
    // -----------------------------------------------------------------------

    pub fn encode_initial_conditions(&mut self, formula: &mut SmtFormula) -> Vec<SmtConstraint> {
        let mut constraints = Vec::new();
        let service_ids: Vec<String> = self.graph.service_ids().iter().map(|s| s.to_string()).collect();

        for sid in &service_ids {
            let node = self.graph.service(sid).unwrap().clone();
            let f_var = failed_var(sid);
            self.register_var(formula, &f_var, SmtSort::Int, sid, VarAttribute::Failed, None);
            // failed_v ∈ {0, 1}
            constraints.push(SmtConstraint {
                name: Some(format!("failed_range_{sid}")),
                expr: SmtExpr::And(vec![
                    SmtExpr::Le(Box::new(SmtExpr::IntConst(0)), Box::new(SmtExpr::Var(f_var.clone()))),
                    SmtExpr::Le(Box::new(SmtExpr::Var(f_var.clone())), Box::new(SmtExpr::IntConst(1))),
                ]),
                source: None,
            });

            let s_var = state_var(sid, 0);
            self.register_var(formula, &s_var, SmtSort::Int, sid, VarAttribute::State, Some(0));

            let l_var = load_var(sid, 0);
            self.register_var(formula, &l_var, SmtSort::Int, sid, VarAttribute::Load, Some(0));

            let r_var = retry_var(sid, 0);
            self.register_var(formula, &r_var, SmtSort::Int, sid, VarAttribute::Retry, Some(0));

            let t_var = timeout_var(sid, 0);
            self.register_var(formula, &t_var, SmtSort::Int, sid, VarAttribute::Timeout, Some(0));

            let bk_var = backoff_var(sid, 0);
            self.register_var(formula, &bk_var, SmtSort::Int, sid, VarAttribute::BackoffDelay, Some(0));

            // If failed: state=2 (unavailable), load=0
            // If healthy: state=0 (healthy), load=baseline_load
            let baseline = node.baseline_load as i64;
            let retry_budget = node.retry_budget as i64;
            let timeout_ms = node.timeout_ms as i64;

            // state[v,0] = ite(failed[v]=1, 2, 0)
            constraints.push(SmtConstraint {
                name: Some(format!("init_state_{sid}")),
                expr: SmtExpr::Eq(
                    Box::new(SmtExpr::Var(s_var.clone())),
                    Box::new(SmtExpr::Ite(
                        Box::new(SmtExpr::Eq(
                            Box::new(SmtExpr::Var(f_var.clone())),
                            Box::new(SmtExpr::IntConst(1)),
                        )),
                        Box::new(SmtExpr::IntConst(health_to_int(ServiceHealth::Unavailable))),
                        Box::new(SmtExpr::IntConst(health_to_int(ServiceHealth::Healthy))),
                    )),
                ),
source: None,
            });

            // load[v,0] = ite(failed[v]=1, 0, baseline)
            constraints.push(SmtConstraint {
                name: Some(format!("init_load_{sid}")),
                expr: SmtExpr::Eq(
                    Box::new(SmtExpr::Var(l_var)),
                    Box::new(SmtExpr::Ite(
                        Box::new(SmtExpr::Eq(
                            Box::new(SmtExpr::Var(f_var.clone())),
                            Box::new(SmtExpr::IntConst(1)),
                        )),
                        Box::new(SmtExpr::IntConst(0)),
                        Box::new(SmtExpr::IntConst(baseline)),
                    )),
                ),
source: None,
            });

            // retry[v,0] = retry_budget
            constraints.push(SmtConstraint {
                name: Some(format!("init_retry_{sid}")),
                expr: SmtExpr::Eq(
                    Box::new(SmtExpr::Var(r_var)),
                    Box::new(SmtExpr::IntConst(retry_budget)),
                ),
                source: None,
            });

            // timeout[v,0] = timeout_ms
            constraints.push(SmtConstraint {
                name: Some(format!("init_timeout_{sid}")),
                expr: SmtExpr::Eq(
                    Box::new(SmtExpr::Var(t_var)),
                    Box::new(SmtExpr::IntConst(timeout_ms)),
                ),
                source: None,
            });

            // backoff[v,0] = 0
            constraints.push(SmtConstraint {
                name: Some(format!("init_backoff_{sid}")),
                expr: SmtExpr::Eq(
                    Box::new(SmtExpr::Var(bk_var)),
                    Box::new(SmtExpr::IntConst(0)),
                ),
                source: None,
            });
        }

        constraints
    }

    // -----------------------------------------------------------------------
    // Transition relation  (t -> t+1)
    // -----------------------------------------------------------------------

    pub fn encode_transition_relation(
        &mut self,
        formula: &mut SmtFormula,
        time_step: usize,
    ) -> Vec<SmtConstraint> {
        let mut constraints = Vec::new();
        let t = time_step;
        let t1 = t + 1;
        let service_ids: Vec<String> = self.graph.service_ids().iter().map(|s| s.to_string()).collect();

        for sid in &service_ids {
            let node = self.graph.service(sid).unwrap().clone();
            let capacity = node.capacity as i64;

            // Declare t+1 variables
            let s_var_next = state_var(sid, t1);
            self.register_var(formula, &s_var_next, SmtSort::Int, sid, VarAttribute::State, Some(t1));
            let l_var_next = load_var(sid, t1);
            self.register_var(formula, &l_var_next, SmtSort::Int, sid, VarAttribute::Load, Some(t1));
            let r_var_next = retry_var(sid, t1);
            self.register_var(formula, &r_var_next, SmtSort::Int, sid, VarAttribute::Retry, Some(t1));
            let t_var_next = timeout_var(sid, t1);
            self.register_var(formula, &t_var_next, SmtSort::Int, sid, VarAttribute::Timeout, Some(t1));
            let bk_var_next = backoff_var(sid, t1);
            self.register_var(formula, &bk_var_next, SmtSort::Int, sid, VarAttribute::BackoffDelay, Some(t1));

            // ---- Load propagation ----
            // load[v,t+1] = baseline + sum over predecessors of load[pred,t] * (1 + retry_factor)
            // We encode retry_factor as integer: amplification = 1 + retry_count
            let incoming = self.graph.incoming_edges(sid);
            let baseline = node.baseline_load as i64;

            if incoming.is_empty() {
                // No predecessors: load stays at baseline (or 0 if failed)
                let f_var = failed_var(sid);
                constraints.push(SmtConstraint {
                    name: Some(format!("trans_load_{sid}_{t1}")),
                    expr: SmtExpr::Eq(
                        Box::new(SmtExpr::Var(l_var_next.clone())),
                        Box::new(SmtExpr::Ite(
                            Box::new(SmtExpr::Eq(
                                Box::new(SmtExpr::Var(f_var)),
                                Box::new(SmtExpr::IntConst(1)),
                            )),
                            Box::new(SmtExpr::IntConst(0)),
                            Box::new(SmtExpr::IntConst(baseline)),
                        )),
                    ),
source: None,
                });
            } else {
                // load[v,t+1] = baseline + sum_pred(load[pred,t] * amplification)
                // where amplification = 1 + retry_count  (integer approximation)
                let mut load_sum_terms: Vec<SmtExpr> = vec![SmtExpr::IntConst(baseline)];
                for edge in &incoming {
                    let pred_id = edge.source.as_str();
                    let pred_load = load_var(pred_id, t);
                    let amplification = 1 + edge.retry_count as i64;
                    // load[pred,t] * amplification
                    // But only if predecessor is not unavailable (state < 2)
                    let pred_state = state_var(pred_id, t);
                    let pred_retry = retry_var(pred_id, t);

                    // Effective load from pred: ite(retry[pred,t] > 0 && state[pred,t] < 2,
                    //   load[pred,t] * amplification,
                    //   load[pred,t])
                    let amplified = SmtExpr::Mul(
                        Box::new(SmtExpr::Var(pred_load.clone())),
                        Box::new(SmtExpr::IntConst(amplification)),
                    );
                    let base_load = SmtExpr::Var(pred_load.clone());
                    let effective = SmtExpr::Ite(
                        Box::new(SmtExpr::And(vec![
                            SmtExpr::Gt(
                                Box::new(SmtExpr::Var(pred_retry)),
                                Box::new(SmtExpr::IntConst(0)),
                            ),
                            SmtExpr::Lt(
                                Box::new(SmtExpr::Var(pred_state)),
                                Box::new(SmtExpr::IntConst(2)),
                            ),
                        ])),
                        Box::new(amplified),
                        Box::new(base_load),
                    );
                    load_sum_terms.push(effective);
                }

                let load_sum = if load_sum_terms.len() == 1 {
                    load_sum_terms.pop().unwrap()
                } else {
                    SmtExpr::Add(
                        Box::new(load_sum_terms[0].clone()),
                        Box::new(self.sum_exprs(&load_sum_terms[1..])),
                    )
                };

                // If this service is failed, load is 0
                let f_var = failed_var(sid);
                constraints.push(SmtConstraint {
                    name: Some(format!("trans_load_{sid}_{t1}")),
                    expr: SmtExpr::Eq(
                        Box::new(SmtExpr::Var(l_var_next.clone())),
                        Box::new(SmtExpr::Ite(
                            Box::new(SmtExpr::Eq(
                                Box::new(SmtExpr::Var(f_var)),
                                Box::new(SmtExpr::IntConst(1)),
                            )),
                            Box::new(SmtExpr::IntConst(0)),
                            Box::new(load_sum),
                        )),
                    ),
source: None,
                });
            }

            // ---- Timeout decrement ----
            // timeout[v,t+1] = max(timeout[v,t] - elapsed, 0)
            // We model elapsed as a fixed step duration (1000ms per step)
            let elapsed = 1000i64;
            let timeout_current = timeout_var(sid, t);
            constraints.push(SmtConstraint {
                name: Some(format!("trans_timeout_{sid}_{t1}")),
                expr: SmtExpr::Eq(
                    Box::new(SmtExpr::Var(t_var_next.clone())),
                    Box::new(SmtExpr::Ite(
                        Box::new(SmtExpr::Gt(
                            Box::new(SmtExpr::Var(timeout_current.clone())),
                            Box::new(SmtExpr::IntConst(elapsed)),
                        )),
                        Box::new(SmtExpr::Sub(
                            Box::new(SmtExpr::Var(timeout_current)),
                            Box::new(SmtExpr::IntConst(elapsed)),
                        )),
                        Box::new(SmtExpr::IntConst(0)),
                    )),
                ),
source: None,
            });

            // ---- State degradation ----
            // if load[v,t+1] > capacity then state = 2 (unavailable)
            // elif load[v,t+1] > capacity * 0.8 then state = 1 (degraded)
            // else state = state[v,t]  (stays the same or improves)
            let threshold_80 = (capacity * 80) / 100;
            let f_var = failed_var(sid);
            constraints.push(SmtConstraint {
                name: Some(format!("trans_state_{sid}_{t1}")),
                expr: SmtExpr::Eq(
                    Box::new(SmtExpr::Var(s_var_next.clone())),
                    Box::new(SmtExpr::Ite(
                        Box::new(SmtExpr::Eq(
                            Box::new(SmtExpr::Var(f_var)),
                            Box::new(SmtExpr::IntConst(1)),
                        )),
                        Box::new(SmtExpr::IntConst(2)),
                        Box::new(SmtExpr::Ite(
                            Box::new(SmtExpr::Gt(
                                Box::new(SmtExpr::Var(l_var_next.clone())),
                                Box::new(SmtExpr::IntConst(capacity)),
                            )),
                            Box::new(SmtExpr::IntConst(2)),
                            Box::new(SmtExpr::Ite(
                                Box::new(SmtExpr::Gt(
                                    Box::new(SmtExpr::Var(l_var_next.clone())),
                                    Box::new(SmtExpr::IntConst(threshold_80)),
                                )),
                                Box::new(SmtExpr::IntConst(1)),
                                Box::new(SmtExpr::IntConst(0)),
                            )),
                        )),
                    )),
                ),
source: None,
            });

            // ---- Retry exhaustion ----
            // retry[v,t+1] = ite(state[v,t] >= 1, max(retry[v,t] - 1, 0), retry[v,t])
            let retry_current = retry_var(sid, t);
            let state_current = state_var(sid, t);
            constraints.push(SmtConstraint {
                name: Some(format!("trans_retry_{sid}_{t1}")),
                expr: SmtExpr::Eq(
                    Box::new(SmtExpr::Var(r_var_next.clone())),
                    Box::new(SmtExpr::Ite(
                        Box::new(SmtExpr::Ge(
                            Box::new(SmtExpr::Var(state_current)),
                            Box::new(SmtExpr::IntConst(1)),
                        )),
                        Box::new(SmtExpr::Ite(
                            Box::new(SmtExpr::Gt(
                                Box::new(SmtExpr::Var(retry_current.clone())),
                                Box::new(SmtExpr::IntConst(0)),
                            )),
                            Box::new(SmtExpr::Sub(
                                Box::new(SmtExpr::Var(retry_current)),
                                Box::new(SmtExpr::IntConst(1)),
                            )),
                            Box::new(SmtExpr::IntConst(0)),
                        )),
                        Box::new(SmtExpr::Var(retry_var(sid, t))),
                    )),
                ),
source: None,
            });

            // ---- Backoff delay encoding ----
            // backoff[v,t+1] = ite(retry[v,t+1] < retry[v,t], backoff_base * 2^(budget - retry), 0)
            // Simplified: backoff[v,t+1] = ite(retry decremented, backoff[v,t] + backoff_base, 0)
            let backoff_base = self.graph.incoming_edges(sid)
                .first()
                .map(|_e| 1000i64)
                .unwrap_or(100);
            let bk_current = backoff_var(sid, t);
            constraints.push(SmtConstraint {
                name: Some(format!("trans_backoff_{sid}_{t1}")),
                expr: SmtExpr::Eq(
                    Box::new(SmtExpr::Var(bk_var_next)),
                    Box::new(SmtExpr::Ite(
                        Box::new(SmtExpr::Lt(
                            Box::new(SmtExpr::Var(r_var_next)),
                            Box::new(SmtExpr::Var(retry_var(sid, t))),
                        )),
                        Box::new(SmtExpr::Add(
                            Box::new(SmtExpr::Var(bk_current)),
                            Box::new(SmtExpr::IntConst(backoff_base)),
                        )),
                        Box::new(SmtExpr::IntConst(0)),
                    )),
                ),
source: None,
            });

            // ---- Non-negativity for load ----
            constraints.push(SmtConstraint {
                name: Some(format!("nonneg_load_{sid}_{t1}")),
                expr: SmtExpr::Le(
                    Box::new(SmtExpr::IntConst(0)),
                    Box::new(SmtExpr::Var(l_var_next)),
                ),
                source: None,
            });
        }

        constraints
    }

    // -----------------------------------------------------------------------
    // Cascade property: exists t such that load[target,t] > capacity
    // -----------------------------------------------------------------------

    pub fn encode_cascade_property(
        &self,
        target_service: &str,
        threshold: i64,
    ) -> SmtConstraint {
        let mut disjuncts = Vec::new();
        for t in 0..=self.depth_bound {
            disjuncts.push(SmtExpr::Gt(
                Box::new(SmtExpr::Var(load_var(target_service, t))),
                Box::new(SmtExpr::IntConst(threshold)),
            ));
        }
        SmtConstraint {
            name: Some(format!("cascade_property_{target_service}")),
            expr: SmtExpr::Or(disjuncts),
            source: None,
        }
    }

    // -----------------------------------------------------------------------
    // Failure budget: sum of failed[v] <= k
    // -----------------------------------------------------------------------

    pub fn encode_failure_budget(&self, budget_k: usize) -> SmtConstraint {
        let service_ids: Vec<String> = self.graph.service_ids().iter().map(|s| s.to_string()).collect();
        if service_ids.is_empty() {
            return SmtConstraint {
                name: Some("failure_budget_empty".to_owned()),
                expr: SmtExpr::BoolConst(true),
                source: None,
            };
        }
        let sum = self.sum_exprs(
            &service_ids
                .iter()
                .map(|sid| SmtExpr::Var(failed_var(sid)))
                .collect::<Vec<_>>(),
        );
        SmtConstraint {
            name: Some(format!("failure_budget_{budget_k}")),
            expr: SmtExpr::Le(Box::new(sum), Box::new(SmtExpr::IntConst(budget_k as i64))),
            source: None,
        }
    }

    // -----------------------------------------------------------------------
    // Full BMC formula assembly
    // -----------------------------------------------------------------------

    pub fn encode_full_bmc_formula(
        &mut self,
        target_service: &str,
    ) -> SmtFormula {
        let mut formula = SmtFormula {
            declarations: Vec::new(),
            constraints: Vec::new(),
        };

        // 1. Initial conditions
        let init = self.encode_initial_conditions(&mut formula);
        for c in init {
            formula.constraints.push(c);
        }

        // 2. Transition relations for each time step
        for t in 0..self.depth_bound {
            let trans = self.encode_transition_relation(&mut formula, t);
            for c in trans {
                formula.constraints.push(c);
            }
        }

        // 3. Failure budget constraint
        let budget = self.encode_failure_budget(self.failure_budget);
        formula.constraints.push(budget);

        // 4. Cascade property (what we want to check SAT for)
        let capacity = self.graph.service(target_service)
            .map(|n| n.capacity as i64)
            .unwrap_or(100);
        let prop = self.encode_cascade_property(target_service, capacity);
        formula.constraints.push(prop);

        formula
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn sum_exprs(&self, exprs: &[SmtExpr]) -> SmtExpr {
        match exprs.len() {
            0 => SmtExpr::IntConst(0),
            1 => exprs[0].clone(),
            _ => {
                let mid = exprs.len() / 2;
                SmtExpr::Add(
                    Box::new(self.sum_exprs(&exprs[..mid])),
                    Box::new(self.sum_exprs(&exprs[mid..])),
                )
            }
        }
    }

    /// Extract which services are marked as failed from an SMT model.
    pub fn extract_failed_services(&self, assignments: &HashMap<String, i64>) -> Vec<String> {
        let mut failed = Vec::new();
        for sid in self.graph.service_ids() {
            let var = failed_var(sid);
            if assignments.get(&var).copied() == Some(1) {
                failed.push(sid.to_owned());
            }
        }
        failed
    }

    /// Extract load values across all timesteps for a service.
    pub fn extract_load_trace(
        &self,
        service: &str,
        assignments: &HashMap<String, i64>,
    ) -> Vec<(usize, i64)> {
        let mut trace = Vec::new();
        for t in 0..=self.depth_bound {
            let var = load_var(service, t);
            if let Some(&val) = assignments.get(&var) {
                trace.push((t, val));
            }
        }
        trace
    }

    /// Extract state values across all timesteps for a service.
    pub fn extract_state_trace(
        &self,
        service: &str,
        assignments: &HashMap<String, i64>,
    ) -> Vec<(usize, i64)> {
        let mut trace = Vec::new();
        for t in 0..=self.depth_bound {
            let var = state_var(service, t);
            if let Some(&val) = assignments.get(&var) {
                trace.push((t, val));
            }
        }
        trace
    }

    /// Count the total number of SMT variables generated.
    pub fn total_variable_count(&self) -> usize {
        self.var_map.len()
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
    fn test_variable_naming() {
        assert_eq!(state_var("auth", 0), "state_auth_0");
        assert_eq!(load_var("db", 3), "load_db_3");
        assert_eq!(failed_var("gateway"), "failed_gateway");
    }

    #[test]
    fn test_health_to_int() {
        assert_eq!(health_to_int(ServiceHealth::Healthy), 0);
        assert_eq!(health_to_int(ServiceHealth::Degraded), 1);
        assert_eq!(health_to_int(ServiceHealth::Unavailable), 2);
    }

    #[test]
    fn test_encode_initial_conditions() {
        let g = simple_graph();
        let mut encoder = BmcEncoder::new(g, 3, 1);
        let mut formula = SmtFormula { declarations: Vec::new(), constraints: Vec::new() };
        let constraints = encoder.encode_initial_conditions(&mut formula);
        // 3 services × 6 constraints each (range, state, load, retry, timeout, backoff) = 18
        assert!(constraints.len() >= 18);
        assert!(!formula.declarations.is_empty());
    }

    #[test]
    fn test_encode_transition_relation() {
        let g = simple_graph();
        let mut encoder = BmcEncoder::new(g, 3, 1);
        let mut formula = SmtFormula { declarations: Vec::new(), constraints: Vec::new() };
        let _ = encoder.encode_initial_conditions(&mut formula);
        let trans = encoder.encode_transition_relation(&mut formula, 0);
        // Each service gets load, timeout, state, retry, backoff, nonneg constraints
        assert!(trans.len() >= 15);
    }

    #[test]
    fn test_encode_cascade_property() {
        let g = simple_graph();
        let encoder = BmcEncoder::new(g, 3, 1);
        let prop = encoder.encode_cascade_property("db", 200);
        assert_eq!(prop.name, "cascade_property_db");
        // Should be a disjunction over 4 timesteps (0..=3)
        if let SmtExpr::Or(ref disjuncts) = prop.expr {
            assert_eq!(disjuncts.len(), 4);
        } else {
            panic!("Expected Or expression");
        }
    }

    #[test]
    fn test_encode_failure_budget() {
        let g = simple_graph();
        let encoder = BmcEncoder::new(g, 3, 1);
        let budget = encoder.encode_failure_budget(1);
        assert_eq!(budget.name, "failure_budget_1");
    }

    #[test]
    fn test_full_formula_generation() {
        let g = simple_graph();
        let mut encoder = BmcEncoder::new(g, 2, 1);
        let formula = encoder.encode_full_bmc_formula("db");
        assert!(!formula.declarations.is_empty());
        assert!(!formula.constraints.is_empty());
        // Should have: init + transitions × depth + budget + property
        assert!(formula.constraints.len() > 20);
    }

    #[test]
    fn test_variable_map_tracking() {
        let g = simple_graph();
        let mut encoder = BmcEncoder::new(g, 2, 1);
        let _ = encoder.encode_full_bmc_formula("db");
        let map = encoder.variable_map();
        assert!(!map.is_empty());
        // Check that we can find load variables for db
        let db_vars = map.service_vars("db");
        assert!(db_vars.len() > 0);
    }

    #[test]
    fn test_extract_failed_services() {
        let g = simple_graph();
        let encoder = BmcEncoder::new(g, 2, 1);
        let mut assignments = HashMap::new();
        assignments.insert("failed_auth".to_owned(), 1);
        assignments.insert("failed_gateway".to_owned(), 0);
        assignments.insert("failed_db".to_owned(), 0);
        let failed = encoder.extract_failed_services(&assignments);
        assert_eq!(failed, vec!["auth"]);
    }

    #[test]
    fn test_extract_load_trace() {
        let g = simple_graph();
        let encoder = BmcEncoder::new(g, 2, 1);
        let mut assignments = HashMap::new();
        assignments.insert("load_db_0".to_owned(), 30);
        assignments.insert("load_db_1".to_owned(), 150);
        assignments.insert("load_db_2".to_owned(), 300);
        let trace = encoder.extract_load_trace("db", &assignments);
        assert_eq!(trace.len(), 3);
        assert_eq!(trace[0], (0, 30));
        assert_eq!(trace[2], (2, 300));
    }

    #[test]
    fn test_var_attribute_variants() {
        assert_ne!(VarAttribute::State, VarAttribute::Load);
        assert_ne!(VarAttribute::Retry, VarAttribute::Timeout);
    }

    #[test]
    fn test_smt_variable_map_operations() {
        let mut map = SmtVariableMap::new();
        assert!(map.is_empty());
        map.insert("load_a_0".to_owned(), VarMapEntry {
            service: "a".to_owned(),
            attribute: VarAttribute::Load,
            time_step: Some(0),
        });
        assert_eq!(map.len(), 1);
        assert!(map.get("load_a_0").is_some());
        let ts_vars = map.timestep_vars(0);
        assert_eq!(ts_vars.len(), 1);
    }

    #[test]
    fn test_sum_exprs_helper() {
        let g = simple_graph();
        let encoder = BmcEncoder::new(g, 2, 1);
        let result = encoder.sum_exprs(&[SmtExpr::IntConst(1), SmtExpr::IntConst(2), SmtExpr::IntConst(3)]);
        // Should produce a balanced tree of Adds
        match result {
            SmtExpr::Add(_, _) => {} // correct structure
            _ => panic!("Expected Add expression"),
        }
    }

    #[test]
    fn test_single_service_encoding() {
        let mut g = RtigGraph::new();
        g.add_service(ServiceNode::new("lonely", 100).with_baseline_load(10));
        let mut encoder = BmcEncoder::new(g, 1, 1);
        let formula = encoder.encode_full_bmc_formula("lonely");
        assert!(!formula.declarations.is_empty());
    }

    #[test]
    fn test_large_depth_formula() {
        let g = simple_graph();
        let mut encoder = BmcEncoder::new(g, 10, 2);
        let formula = encoder.encode_full_bmc_formula("db");
        assert!(formula.declarations.len() > 50);
        assert!(formula.constraints.len() > 100);
    }

    #[test]
    fn test_backoff_variable_generation() {
        let g = simple_graph();
        let mut encoder = BmcEncoder::new(g, 2, 1);
        let _ = encoder.encode_full_bmc_formula("db");
        let map = encoder.variable_map();
        let has_backoff = map.entries.keys().any(|k| k.starts_with("backoff_"));
        assert!(has_backoff);
    }
}
