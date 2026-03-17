//! Pluggable repair strategies and strategy comparison.
//!
//! Each strategy implements the [`RepairStrategy`] trait to produce a set of
//! [`RepairPlan`]s from a collection of risky paths. The [`compare_strategies`]
//! function runs all built-in strategies and recommends the best one.

use super::synthesizer::{ParameterBounds, RiskyPathInfo};
use super::{RepairAction, RepairPlan};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A pluggable repair strategy.
pub trait RepairStrategy {
    /// Synthesize zero or more repair plans for the given risks.
    fn synthesize(
        &self,
        risks: &[RiskyPathInfo],
        adj: &[(String, String, u32, u64)],
        bounds: &ParameterBounds,
    ) -> Vec<RepairPlan>;

    /// Human-readable name for this strategy.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Strategy implementations
// ---------------------------------------------------------------------------

/// Minimize the total number of parameter changes.
///
/// For each risky path, find the single edge whose retry reduction yields the
/// greatest amplification decrease and change only that edge.
#[derive(Debug, Clone, Default)]
pub struct MinimalChangeStrategy;

impl RepairStrategy for MinimalChangeStrategy {
    fn name(&self) -> &str {
        "MinimalChange"
    }

    fn synthesize(
        &self,
        risks: &[RiskyPathInfo],
        adj: &[(String, String, u32, u64)],
        bounds: &ParameterBounds,
    ) -> Vec<RepairPlan> {
        let mut plans = Vec::new();

        for risk in risks.iter().filter(|r| r.has_amplification_violation()) {
            let edges = collect_path_edges(&risk.path, adj);
            if edges.is_empty() {
                continue;
            }

            // Find the edge with the highest retry count.
            if let Some(worst) = edges.iter().max_by_key(|e| e.2) {
                let needed = min_retry_for_threshold(
                    &risk.path,
                    adj,
                    &worst.0,
                    &worst.1,
                    risk.threshold,
                    bounds,
                );
                if needed < worst.2 {
                    let mut plan = RepairPlan { feasible: true, ..Default::default() };
                    plan.add_action(RepairAction::reduce_retries(
                        &worst.0, &worst.1, worst.2, needed,
                    ));
                    plans.push(plan);
                }
            }
        }

        for risk in risks.iter().filter(|r| r.has_timeout_violation()) {
            if let Some(plan) = fix_timeout_greedy(&risk.path, adj, risk.deadline_ms, bounds) {
                plans.push(plan);
            }
        }

        plans
    }
}

/// Minimize total deviation from current parameter values by spreading
/// changes evenly across all edges on the path.
#[derive(Debug, Clone, Default)]
pub struct MinimalDeviationStrategy;

impl RepairStrategy for MinimalDeviationStrategy {
    fn name(&self) -> &str {
        "MinimalDeviation"
    }

    fn synthesize(
        &self,
        risks: &[RiskyPathInfo],
        adj: &[(String, String, u32, u64)],
        bounds: &ParameterBounds,
    ) -> Vec<RepairPlan> {
        let mut plans = Vec::new();

        for risk in risks.iter().filter(|r| r.has_amplification_violation()) {
            let edges = collect_path_edges(&risk.path, adj);
            if edges.is_empty() {
                continue;
            }

            let n = edges.len() as f64;
            let per_edge_max_factor = risk.threshold.powf(1.0 / n);

            let mut plan = RepairPlan { feasible: true, ..Default::default() };
            for (src, tgt, retries, _) in &edges {
                let current_factor = 1.0 + *retries as f64;
                if current_factor > per_edge_max_factor {
                    let new_retries =
                        ((per_edge_max_factor - 1.0).floor() as u32).max(bounds.min_retry);
                    if new_retries < *retries {
                        plan.add_action(RepairAction::reduce_retries(
                            src, tgt, *retries, new_retries,
                        ));
                    }
                }
            }
            if !plan.is_empty() {
                plans.push(plan);
            }
        }

        for risk in risks.iter().filter(|r| r.has_timeout_violation()) {
            if let Some(plan) = fix_timeout_proportional(&risk.path, adj, risk.deadline_ms, bounds)
            {
                plans.push(plan);
            }
        }

        plans
    }
}

/// Fix the worst risks first, then move to less severe ones.
#[derive(Debug, Clone, Default)]
pub struct GreedyStrategy;

impl RepairStrategy for GreedyStrategy {
    fn name(&self) -> &str {
        "Greedy"
    }

    fn synthesize(
        &self,
        risks: &[RiskyPathInfo],
        adj: &[(String, String, u32, u64)],
        bounds: &ParameterBounds,
    ) -> Vec<RepairPlan> {
        // Sort risks by severity (amplification ratio or timeout excess).
        let mut sorted_risks: Vec<&RiskyPathInfo> = risks.iter().collect();
        sorted_risks.sort_by(|a, b| {
            let severity_a = severity_score(a);
            let severity_b = severity_score(b);
            severity_b
                .partial_cmp(&severity_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut plans = Vec::new();
        let mut modified_adj = adj.to_vec();

        for risk in sorted_risks {
            if risk.has_amplification_violation() {
                let current_amp = path_amplification(&risk.path, &modified_adj);
                if current_amp <= risk.threshold {
                    continue; // already fixed by a prior repair
                }
                if let Some(plan) =
                    greedy_amplification_fix(&risk.path, &modified_adj, risk.threshold, bounds)
                {
                    apply_plan_to_adj_mut(&plan, &mut modified_adj);
                    plans.push(plan);
                }
            }
            if risk.has_timeout_violation() {
                let current_to = path_timeout(&risk.path, &modified_adj);
                if current_to <= risk.deadline_ms {
                    continue;
                }
                if let Some(plan) =
                    fix_timeout_greedy(&risk.path, &modified_adj, risk.deadline_ms, bounds)
                {
                    apply_plan_to_adj_mut(&plan, &mut modified_adj);
                    plans.push(plan);
                }
            }
        }

        plans
    }
}

/// Uniformly reduce all retry counts on risky paths by a fixed percentage.
#[derive(Debug, Clone)]
pub struct UniformReductionStrategy {
    pub reduction_pct: f64,
}

impl UniformReductionStrategy {
    pub fn new(reduction_pct: f64) -> Self {
        Self {
            reduction_pct: reduction_pct.clamp(0.0, 1.0),
        }
    }
}

impl RepairStrategy for UniformReductionStrategy {
    fn name(&self) -> &str {
        "UniformReduction"
    }

    fn synthesize(
        &self,
        risks: &[RiskyPathInfo],
        adj: &[(String, String, u32, u64)],
        bounds: &ParameterBounds,
    ) -> Vec<RepairPlan> {
        let mut plan = RepairPlan { feasible: true, ..Default::default() };
        let mut seen = std::collections::HashSet::new();

        for risk in risks.iter().filter(|r| r.has_amplification_violation()) {
            let edges = collect_path_edges(&risk.path, adj);
            for (src, tgt, retries, _) in &edges {
                let key = (src.clone(), tgt.clone());
                if seen.contains(&key) {
                    continue;
                }
                seen.insert(key);

                let reduction = (*retries as f64 * self.reduction_pct).ceil() as u32;
                let new_retries = retries.saturating_sub(reduction).max(bounds.min_retry);
                if new_retries < *retries {
                    plan.add_action(RepairAction::reduce_retries(src, tgt, *retries, new_retries));
                }
            }
        }

        if plan.is_empty() {
            Vec::new()
        } else {
            vec![plan]
        }
    }
}

/// Reset all parameters on risky paths to safe defaults
/// (retry=3, timeout=5000ms).
#[derive(Debug, Clone, Default)]
pub struct NaiveDefaultStrategy;

impl NaiveDefaultStrategy {
    const DEFAULT_RETRY: u32 = 3;
    const DEFAULT_TIMEOUT_MS: u64 = 5000;
}

impl RepairStrategy for NaiveDefaultStrategy {
    fn name(&self) -> &str {
        "NaiveDefault"
    }

    fn synthesize(
        &self,
        risks: &[RiskyPathInfo],
        adj: &[(String, String, u32, u64)],
        _bounds: &ParameterBounds,
    ) -> Vec<RepairPlan> {
        let mut plan = RepairPlan { feasible: true, ..Default::default() };
        let mut seen = std::collections::HashSet::new();

        for risk in risks {
            let edges = collect_path_edges(&risk.path, adj);
            for (src, tgt, retries, timeout) in &edges {
                let key = (src.clone(), tgt.clone());
                if seen.contains(&key) {
                    continue;
                }
                seen.insert(key);

                if *retries != Self::DEFAULT_RETRY {
                    plan.add_action(RepairAction::reduce_retries(
                        src,
                        tgt,
                        *retries,
                        Self::DEFAULT_RETRY,
                    ));
                }
                if *timeout != Self::DEFAULT_TIMEOUT_MS {
                    plan.add_action(RepairAction::adjust_timeout(
                        src,
                        tgt,
                        *timeout,
                        Self::DEFAULT_TIMEOUT_MS,
                    ));
                }
            }
        }

        if plan.is_empty() {
            Vec::new()
        } else {
            vec![plan]
        }
    }
}

// ---------------------------------------------------------------------------
// Strategy comparison
// ---------------------------------------------------------------------------

/// Results of running multiple strategies and comparing them.
#[derive(Debug, Clone)]
pub struct StrategyComparison {
    /// `(strategy_name, plans)` pairs.
    pub results: Vec<(String, Vec<RepairPlan>)>,
    /// The recommended strategy name.
    pub recommendation: String,
}

/// Run all built-in strategies and recommend the best one.
pub fn compare_strategies(
    risks: &[RiskyPathInfo],
    adj: &[(String, String, u32, u64)],
    bounds: &ParameterBounds,
) -> StrategyComparison {
    let strategies: Vec<Box<dyn RepairStrategy>> = vec![
        Box::new(MinimalChangeStrategy),
        Box::new(MinimalDeviationStrategy),
        Box::new(GreedyStrategy),
        Box::new(UniformReductionStrategy::new(0.5)),
        Box::new(NaiveDefaultStrategy),
    ];

    let mut results: Vec<(String, Vec<RepairPlan>)> = Vec::new();
    for strategy in &strategies {
        let plans = strategy.synthesize(risks, adj, bounds);
        results.push((strategy.name().to_string(), plans));
    }

    // Score each strategy: prefer lowest total deviation among feasible plans.
    let mut best_name = "MinimalChange".to_string();
    let mut best_score = f64::MAX;

    for (name, plans) in &results {
        let feasible_plans: Vec<&RepairPlan> = plans.iter().filter(|p| p.feasible).collect();
        if feasible_plans.is_empty() {
            continue;
        }
        let total_deviation: f64 = feasible_plans.iter().map(|p| p.total_deviation).sum();
        let change_count: usize = feasible_plans.iter().map(|p| p.actions.len()).sum();
        // Combined score: deviation + small penalty per change.
        let score = total_deviation + change_count as f64 * 0.1;
        if score < best_score {
            best_score = score;
            best_name = name.clone();
        }
    }

    StrategyComparison {
        results,
        recommendation: best_name,
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn collect_path_edges(
    path: &[String],
    adj: &[(String, String, u32, u64)],
) -> Vec<(String, String, u32, u64)> {
    let mut out = Vec::new();
    for w in path.windows(2) {
        if let Some(e) = adj.iter().find(|e| e.0 == w[0] && e.1 == w[1]) {
            out.push(e.clone());
        }
    }
    out
}

fn path_amplification(path: &[String], adj: &[(String, String, u32, u64)]) -> f64 {
    let mut product = 1.0_f64;
    for w in path.windows(2) {
        if let Some(e) = adj.iter().find(|e| e.0 == w[0] && e.1 == w[1]) {
            product *= 1.0 + e.2 as f64;
        }
    }
    product
}

fn path_timeout(path: &[String], adj: &[(String, String, u32, u64)]) -> u64 {
    let mut total = 0u64;
    for w in path.windows(2) {
        if let Some(e) = adj.iter().find(|e| e.0 == w[0] && e.1 == w[1]) {
            total = total.saturating_add(e.3);
        }
    }
    total
}

fn severity_score(risk: &RiskyPathInfo) -> f64 {
    let amp_excess = if risk.has_amplification_violation() {
        risk.amplification / risk.threshold.max(1.0)
    } else {
        0.0
    };
    let to_excess = if risk.has_timeout_violation() {
        risk.timeout_ms as f64 / risk.deadline_ms.max(1) as f64
    } else {
        0.0
    };
    amp_excess + to_excess
}

fn min_retry_for_threshold(
    path: &[String],
    adj: &[(String, String, u32, u64)],
    src: &str,
    tgt: &str,
    threshold: f64,
    bounds: &ParameterBounds,
) -> u32 {
    for r in 0..=bounds.max_retry {
        let amp = path_amplification_with_override(path, adj, src, tgt, r);
        if amp <= threshold {
            return r.max(bounds.min_retry);
        }
    }
    bounds.min_retry
}

fn path_amplification_with_override(
    path: &[String],
    adj: &[(String, String, u32, u64)],
    override_src: &str,
    override_tgt: &str,
    new_retry: u32,
) -> f64 {
    let mut product = 1.0_f64;
    for w in path.windows(2) {
        if w[0] == override_src && w[1] == override_tgt {
            product *= 1.0 + new_retry as f64;
        } else if let Some(e) = adj.iter().find(|e| e.0 == w[0] && e.1 == w[1]) {
            product *= 1.0 + e.2 as f64;
        }
    }
    product
}

fn greedy_amplification_fix(
    path: &[String],
    adj: &[(String, String, u32, u64)],
    threshold: f64,
    bounds: &ParameterBounds,
) -> Option<RepairPlan> {
    let edges = collect_path_edges(path, adj);
    if edges.is_empty() {
        return None;
    }

    let mut local_edges = edges.clone();
    let mut plan = RepairPlan { feasible: true, ..Default::default() };

    for _ in 0..100 {
        let amp: f64 = local_edges.iter().map(|e| 1.0 + e.2 as f64).product();
        if amp <= threshold {
            break;
        }

        // Reduce the edge with the highest retry count.
        if let Some(idx) = local_edges
            .iter()
            .enumerate()
            .filter(|(_, e)| e.2 > bounds.min_retry)
            .max_by_key(|(_, e)| e.2)
            .map(|(i, _)| i)
        {
            local_edges[idx].2 -= 1;
        } else {
            plan.feasible = false;
            break;
        }
    }

    for (orig, modified) in edges.iter().zip(local_edges.iter()) {
        if modified.2 != orig.2 {
            plan.add_action(RepairAction::reduce_retries(
                &orig.0, &orig.1, orig.2, modified.2,
            ));
        }
    }

    if plan.is_empty() { None } else { Some(plan) }
}

fn fix_timeout_greedy(
    path: &[String],
    adj: &[(String, String, u32, u64)],
    deadline_ms: u64,
    bounds: &ParameterBounds,
) -> Option<RepairPlan> {
    let edges = collect_path_edges(path, adj);
    if edges.is_empty() {
        return None;
    }

    let current_sum: u64 = edges.iter().map(|e| e.3).sum();
    if current_sum <= deadline_ms {
        return None;
    }

    let excess = current_sum - deadline_ms;
    let mut local_edges = edges.clone();
    let mut remaining = excess;

    // Trim from largest timeouts first.
    let mut indices: Vec<usize> = (0..local_edges.len()).collect();
    indices.sort_by(|a, b| local_edges[*b].3.cmp(&local_edges[*a].3));

    for &i in &indices {
        if remaining == 0 {
            break;
        }
        let reducible = local_edges[i].3.saturating_sub(bounds.min_timeout_ms);
        let reduce_by = remaining.min(reducible);
        if reduce_by > 0 {
            local_edges[i].3 -= reduce_by;
            remaining -= reduce_by;
        }
    }

    let mut plan = RepairPlan {
        feasible: remaining == 0,
        ..Default::default()
    };

    for (orig, modified) in edges.iter().zip(local_edges.iter()) {
        if modified.3 != orig.3 {
            plan.add_action(RepairAction::adjust_timeout(
                &orig.0, &orig.1, orig.3, modified.3,
            ));
        }
    }

    if plan.is_empty() { None } else { Some(plan) }
}

fn fix_timeout_proportional(
    path: &[String],
    adj: &[(String, String, u32, u64)],
    deadline_ms: u64,
    bounds: &ParameterBounds,
) -> Option<RepairPlan> {
    let edges = collect_path_edges(path, adj);
    if edges.is_empty() {
        return None;
    }

    let current_sum: u64 = edges.iter().map(|e| e.3).sum();
    if current_sum <= deadline_ms {
        return None;
    }

    let ratio = deadline_ms as f64 / current_sum as f64;
    let mut plan = RepairPlan { feasible: true, ..Default::default() };

    for (src, tgt, _, timeout) in &edges {
        let new_timeout = ((*timeout as f64) * ratio).round() as u64;
        let clamped = new_timeout.max(bounds.min_timeout_ms);
        if clamped != *timeout {
            plan.add_action(RepairAction::adjust_timeout(src, tgt, *timeout, clamped));
        }
    }

    if plan.is_empty() { None } else { Some(plan) }
}

fn apply_plan_to_adj_mut(plan: &RepairPlan, adj: &mut [(String, String, u32, u64)]) {
    for action in &plan.actions {
        if let Some((ref src, ref tgt)) = action.edge {
            if let Some(e) = adj.iter_mut().find(|e| &e.0 == src && &e.1 == tgt) {
                match &action.action_type {
                    super::RepairActionType::ReduceRetries { to, .. } => e.2 = *to,
                    super::RepairActionType::AdjustTimeout { to_ms, .. } => e.3 = *to_ms,
                    _ => {}
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_bounds() -> ParameterBounds {
        ParameterBounds {
            min_retry: 0,
            max_retry: 10,
            min_timeout_ms: 100,
            max_timeout_ms: 60_000,
        }
    }

    fn sample_adj() -> Vec<(String, String, u32, u64)> {
        vec![
            ("A".into(), "B".into(), 5, 3000),
            ("B".into(), "C".into(), 4, 4000),
            ("C".into(), "D".into(), 3, 5000),
        ]
    }

    fn amp_risk() -> RiskyPathInfo {
        RiskyPathInfo {
            path: vec!["A".into(), "B".into(), "C".into(), "D".into()],
            amplification: 120.0,
            timeout_ms: 12000,
            threshold: 30.0,
            deadline_ms: 20000,
        }
    }

    fn timeout_risk() -> RiskyPathInfo {
        RiskyPathInfo {
            path: vec!["A".into(), "B".into(), "C".into()],
            amplification: 1.0,
            timeout_ms: 7000,
            threshold: 1000.0,
            deadline_ms: 5000,
        }
    }

    #[test]
    fn test_minimal_change_produces_plan() {
        let strategy = MinimalChangeStrategy;
        let plans = strategy.synthesize(&[amp_risk()], &sample_adj(), &default_bounds());
        assert!(!plans.is_empty());
        // Should produce the fewest possible changes.
        for plan in &plans {
            assert!(plan.actions.len() <= 2);
        }
    }

    #[test]
    fn test_minimal_deviation_produces_plan() {
        let strategy = MinimalDeviationStrategy;
        let plans = strategy.synthesize(&[amp_risk()], &sample_adj(), &default_bounds());
        assert!(!plans.is_empty());
    }

    #[test]
    fn test_greedy_fixes_worst_first() {
        let strategy = GreedyStrategy;
        let plans = strategy.synthesize(&[amp_risk()], &sample_adj(), &default_bounds());
        assert!(!plans.is_empty());
        for plan in &plans {
            assert!(plan.feasible);
        }
    }

    #[test]
    fn test_uniform_reduction() {
        let strategy = UniformReductionStrategy::new(0.5);
        let plans = strategy.synthesize(&[amp_risk()], &sample_adj(), &default_bounds());
        assert!(!plans.is_empty());
        // All edges on the path should have been reduced.
        let total_actions: usize = plans.iter().map(|p| p.actions.len()).sum();
        assert!(total_actions >= 2);
    }

    #[test]
    fn test_naive_default_resets() {
        let strategy = NaiveDefaultStrategy;
        let plans = strategy.synthesize(&[amp_risk()], &sample_adj(), &default_bounds());
        assert!(!plans.is_empty());
        for plan in &plans {
            for action in &plan.actions {
                if let super::super::RepairActionType::ReduceRetries { to, .. } =
                    &action.action_type
                {
                    assert_eq!(*to, 3);
                }
            }
        }
    }

    #[test]
    fn test_compare_strategies_returns_all() {
        let cmp = compare_strategies(&[amp_risk()], &sample_adj(), &default_bounds());
        assert_eq!(cmp.results.len(), 5);
        assert!(!cmp.recommendation.is_empty());
    }

    #[test]
    fn test_compare_strategies_empty_risks() {
        let cmp = compare_strategies(&[], &sample_adj(), &default_bounds());
        assert_eq!(cmp.results.len(), 5);
        // All strategies should return empty plans for no risks.
        for (_name, plans) in &cmp.results {
            assert!(plans.is_empty());
        }
    }

    #[test]
    fn test_timeout_fix_via_minimal_change() {
        let strategy = MinimalChangeStrategy;
        let plans = strategy.synthesize(&[timeout_risk()], &sample_adj(), &default_bounds());
        assert!(!plans.is_empty());
    }

    #[test]
    fn test_strategy_names_unique() {
        let strategies: Vec<Box<dyn RepairStrategy>> = vec![
            Box::new(MinimalChangeStrategy),
            Box::new(MinimalDeviationStrategy),
            Box::new(GreedyStrategy),
            Box::new(UniformReductionStrategy::new(0.5)),
            Box::new(NaiveDefaultStrategy),
        ];
        let names: Vec<&str> = strategies.iter().map(|s| s.name()).collect();
        let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
        assert_eq!(names.len(), unique.len());
    }

    #[test]
    fn test_combined_risks() {
        let strategy = GreedyStrategy;
        let risks = vec![amp_risk(), timeout_risk()];
        let plans = strategy.synthesize(&risks, &sample_adj(), &default_bounds());
        assert!(!plans.is_empty());
    }

    #[test]
    fn test_no_change_when_safe() {
        let safe_risk = RiskyPathInfo {
            path: vec!["A".into(), "B".into()],
            amplification: 2.0,
            timeout_ms: 1000,
            threshold: 100.0,
            deadline_ms: 10000,
        };
        let strategy = MinimalChangeStrategy;
        let plans = strategy.synthesize(&[safe_risk], &sample_adj(), &default_bounds());
        assert!(plans.is_empty());
    }
}
