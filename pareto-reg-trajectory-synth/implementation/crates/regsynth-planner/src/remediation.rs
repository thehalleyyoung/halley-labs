use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

// ─── Remediation Kind ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RemediationKind {
    TechnicalControl,
    ProcessChange,
    PolicyUpdate,
    Training,
    Documentation,
    ThirdPartyAudit,
}

impl std::fmt::Display for RemediationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TechnicalControl => write!(f, "Technical Control"),
            Self::ProcessChange => write!(f, "Process Change"),
            Self::PolicyUpdate => write!(f, "Policy Update"),
            Self::Training => write!(f, "Training"),
            Self::Documentation => write!(f, "Documentation"),
            Self::ThirdPartyAudit => write!(f, "Third-Party Audit"),
        }
    }
}

// ─── Remediation Option ─────────────────────────────────────────────────────

/// A candidate remediation action that can cover one or more obligations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationOption {
    pub id: String,
    pub kind: RemediationKind,
    pub description: String,
    pub estimated_cost: f64,
    pub estimated_days: u32,
    pub obligation_ids: Vec<String>,
    /// Effectiveness score in [0, 1] — higher means more effective.
    pub effectiveness: f64,
    /// Priority score for tie-breaking (higher = preferred).
    #[serde(default)]
    pub priority: u32,
}

impl RemediationOption {
    pub fn cost_per_obligation(&self) -> f64 {
        if self.obligation_ids.is_empty() {
            self.estimated_cost
        } else {
            self.estimated_cost / self.obligation_ids.len() as f64
        }
    }

    /// Weighted cost: estimated_cost / effectiveness, penalizing low-effectiveness options.
    pub fn weighted_cost(&self) -> f64 {
        if self.effectiveness > 0.0 {
            self.estimated_cost / self.effectiveness
        } else {
            f64::MAX
        }
    }
}

// ─── Hitting Set Result ─────────────────────────────────────────────────────

/// The result of a minimum-weight hitting set computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HittingSetResult {
    pub selected_option_ids: Vec<String>,
    pub total_cost: f64,
    pub total_days: u32,
    pub covered_obligations: Vec<String>,
    pub uncovered_obligations: Vec<String>,
    pub coverage_ratio: f64,
}

// ─── Remediation Engine ─────────────────────────────────────────────────────

/// Engine for suggesting remediations and computing minimum-weight hitting sets
/// over the obligation-to-remediation bipartite graph.
pub struct RemediationEngine {
    pub options: Vec<RemediationOption>,
}

impl RemediationEngine {
    pub fn new() -> Self {
        Self { options: Vec::new() }
    }

    pub fn add_option(&mut self, opt: RemediationOption) {
        self.options.push(opt);
    }

    /// Return all options that cover at least one of the given obligations.
    pub fn suggest_for_obligations(&self, obligation_ids: &[String]) -> Vec<&RemediationOption> {
        let target: HashSet<&str> = obligation_ids.iter().map(|s| s.as_str()).collect();
        self.options
            .iter()
            .filter(|opt| opt.obligation_ids.iter().any(|o| target.contains(o.as_str())))
            .collect()
    }

    /// Greedy minimum-cost covering: pick cheapest options first until all obligations
    /// are covered (classic greedy weighted set cover).
    pub fn minimum_cost_covering(&self, obligation_ids: &[String]) -> Vec<&RemediationOption> {
        let mut uncovered: HashSet<&str> = obligation_ids.iter().map(|s| s.as_str()).collect();
        let mut selected: Vec<&RemediationOption> = Vec::new();

        while !uncovered.is_empty() {
            // Pick the option with the best cost-effectiveness that covers at least one uncovered obligation
            let best = self.options.iter()
                .filter(|opt| opt.obligation_ids.iter().any(|o| uncovered.contains(o.as_str())))
                .min_by(|a, b| {
                    let a_new = a.obligation_ids.iter().filter(|o| uncovered.contains(o.as_str())).count();
                    let b_new = b.obligation_ids.iter().filter(|o| uncovered.contains(o.as_str())).count();
                    let a_ratio = if a_new > 0 { a.estimated_cost / a_new as f64 } else { f64::MAX };
                    let b_ratio = if b_new > 0 { b.estimated_cost / b_new as f64 } else { f64::MAX };
                    a_ratio.partial_cmp(&b_ratio).unwrap_or(std::cmp::Ordering::Equal)
                });

            match best {
                Some(opt) => {
                    for o in &opt.obligation_ids {
                        uncovered.remove(o.as_str());
                    }
                    selected.push(opt);
                }
                None => break, // No option can cover remaining obligations
            }
        }

        selected
    }

    /// Compute a minimum-weight hitting set result.
    ///
    /// A hitting set selects remediation options such that every obligation is
    /// "hit" (covered) by at least one selected option. This uses a greedy
    /// approximation weighted by `weighted_cost()`.
    pub fn minimum_weight_hitting_set(&self, obligation_ids: &[String]) -> HittingSetResult {
        let mut uncovered: HashSet<&str> = obligation_ids.iter().map(|s| s.as_str()).collect();
        let total_count = uncovered.len();
        let mut selected_ids = Vec::new();
        let mut total_cost = 0.0;
        let mut total_days = 0u32;
        let mut covered: HashSet<String> = HashSet::new();

        // Build index: obligation → options that cover it
        let mut obl_to_options: HashMap<&str, Vec<usize>> = HashMap::new();
        for (idx, opt) in self.options.iter().enumerate() {
            for o in &opt.obligation_ids {
                obl_to_options.entry(o.as_str()).or_default().push(idx);
            }
        }

        let mut used: HashSet<usize> = HashSet::new();

        while !uncovered.is_empty() {
            // Score each unused option by: new_coverage / weighted_cost
            let best_idx = self.options.iter().enumerate()
                .filter(|(idx, _)| !used.contains(idx))
                .filter_map(|(idx, opt)| {
                    let new_coverage = opt.obligation_ids.iter()
                        .filter(|o| uncovered.contains(o.as_str()))
                        .count();
                    if new_coverage == 0 {
                        return None;
                    }
                    let score = new_coverage as f64 / opt.weighted_cost().max(1e-9);
                    Some((idx, score))
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx);

            match best_idx {
                Some(idx) => {
                    let opt = &self.options[idx];
                    used.insert(idx);
                    selected_ids.push(opt.id.clone());
                    total_cost += opt.estimated_cost;
                    total_days = total_days.max(opt.estimated_days); // parallel execution

                    for o in &opt.obligation_ids {
                        uncovered.remove(o.as_str());
                        covered.insert(o.clone());
                    }
                }
                None => break,
            }
        }

        let uncovered_list: Vec<String> = uncovered.iter().map(|s| s.to_string()).collect();
        let coverage_ratio = if total_count > 0 {
            covered.len() as f64 / total_count as f64
        } else {
            1.0
        };

        HittingSetResult {
            selected_option_ids: selected_ids,
            total_cost,
            total_days,
            covered_obligations: covered.into_iter().collect(),
            uncovered_obligations: uncovered_list,
            coverage_ratio,
        }
    }

    /// Filter options by kind.
    pub fn options_by_kind(&self, kind: RemediationKind) -> Vec<&RemediationOption> {
        self.options.iter().filter(|o| o.kind == kind).collect()
    }

    /// Return the total cost if all options were selected.
    pub fn total_remediation_cost(&self) -> f64 {
        self.options.iter().map(|o| o.estimated_cost).sum()
    }

    /// Rank all options by weighted cost (ascending — cheapest first).
    pub fn ranked_by_cost(&self) -> Vec<&RemediationOption> {
        let mut sorted: Vec<&RemediationOption> = self.options.iter().collect();
        sorted.sort_by(|a, b| {
            a.weighted_cost().partial_cmp(&b.weighted_cost()).unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }
}

impl Default for RemediationEngine {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_option(id: &str, cost: f64, obls: &[&str], effectiveness: f64) -> RemediationOption {
        RemediationOption {
            id: id.into(),
            kind: RemediationKind::TechnicalControl,
            description: format!("Option {}", id),
            estimated_cost: cost,
            estimated_days: 10,
            obligation_ids: obls.iter().map(|s| s.to_string()).collect(),
            effectiveness,
            priority: 0,
        }
    }

    #[test]
    fn test_suggest_for_obligations() {
        let mut engine = RemediationEngine::new();
        engine.add_option(make_option("r1", 100.0, &["o1", "o2"], 0.9));
        engine.add_option(make_option("r2", 200.0, &["o3"], 0.8));

        let suggestions = engine.suggest_for_obligations(&["o1".into()]);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].id, "r1");
    }

    #[test]
    fn test_minimum_cost_covering_basic() {
        let mut engine = RemediationEngine::new();
        engine.add_option(make_option("r1", 100.0, &["o1", "o2"], 0.9));
        engine.add_option(make_option("r2", 50.0, &["o1"], 0.8));
        engine.add_option(make_option("r3", 60.0, &["o2"], 0.8));

        // r1 covers both for 100, while r2+r3 covers both for 110 → pick r1
        let result = engine.minimum_cost_covering(&["o1".into(), "o2".into()]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "r1");
    }

    #[test]
    fn test_minimum_cost_covering_uncoverable() {
        let mut engine = RemediationEngine::new();
        engine.add_option(make_option("r1", 100.0, &["o1"], 0.9));

        let result = engine.minimum_cost_covering(&["o1".into(), "o2".into()]);
        assert_eq!(result.len(), 1); // Can only cover o1
    }

    #[test]
    fn test_hitting_set_full_coverage() {
        let mut engine = RemediationEngine::new();
        engine.add_option(make_option("r1", 100.0, &["o1", "o2"], 0.9));
        engine.add_option(make_option("r2", 200.0, &["o2", "o3"], 0.8));

        let hs = engine.minimum_weight_hitting_set(&["o1".into(), "o2".into(), "o3".into()]);
        assert_eq!(hs.coverage_ratio, 1.0);
        assert!(hs.uncovered_obligations.is_empty());
        assert_eq!(hs.selected_option_ids.len(), 2);
    }

    #[test]
    fn test_hitting_set_partial_coverage() {
        let mut engine = RemediationEngine::new();
        engine.add_option(make_option("r1", 100.0, &["o1"], 0.9));

        let hs = engine.minimum_weight_hitting_set(&["o1".into(), "o2".into()]);
        assert!(hs.coverage_ratio < 1.0);
        assert_eq!(hs.uncovered_obligations.len(), 1);
    }

    #[test]
    fn test_cost_per_obligation() {
        let opt = make_option("r1", 300.0, &["o1", "o2", "o3"], 0.9);
        assert!((opt.cost_per_obligation() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_cost() {
        let opt = make_option("r1", 100.0, &["o1"], 0.5);
        assert!((opt.weighted_cost() - 200.0).abs() < 1e-9);
    }

    #[test]
    fn test_ranked_by_cost() {
        let mut engine = RemediationEngine::new();
        engine.add_option(make_option("expensive", 1000.0, &["o1"], 0.5));
        engine.add_option(make_option("cheap", 10.0, &["o2"], 1.0));
        let ranked = engine.ranked_by_cost();
        assert_eq!(ranked[0].id, "cheap");
    }

    #[test]
    fn test_options_by_kind() {
        let mut engine = RemediationEngine::new();
        engine.add_option(RemediationOption {
            id: "r1".into(),
            kind: RemediationKind::Training,
            description: "Training".into(),
            estimated_cost: 50.0,
            estimated_days: 5,
            obligation_ids: vec!["o1".into()],
            effectiveness: 0.8,
            priority: 0,
        });
        engine.add_option(make_option("r2", 100.0, &["o2"], 0.9));

        assert_eq!(engine.options_by_kind(RemediationKind::Training).len(), 1);
        assert_eq!(engine.options_by_kind(RemediationKind::TechnicalControl).len(), 1);
        assert_eq!(engine.options_by_kind(RemediationKind::Documentation).len(), 0);
    }

    #[test]
    fn test_remediation_kind_display() {
        assert_eq!(RemediationKind::TechnicalControl.to_string(), "Technical Control");
        assert_eq!(RemediationKind::ThirdPartyAudit.to_string(), "Third-Party Audit");
    }

    #[test]
    fn test_total_remediation_cost() {
        let mut engine = RemediationEngine::new();
        engine.add_option(make_option("r1", 100.0, &["o1"], 0.9));
        engine.add_option(make_option("r2", 200.0, &["o2"], 0.8));
        assert!((engine.total_remediation_cost() - 300.0).abs() < 1e-9);
    }
}
