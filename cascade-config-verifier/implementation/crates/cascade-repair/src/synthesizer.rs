//! Repair synthesis engine for cascade configuration issues.
//!
//! Given a set of risky paths (amplification or timeout violations) and the
//! graph adjacency, the [`RepairSynthesizer`] computes a minimal-cost
//! [`RepairPlan`] that brings all paths back within safe thresholds.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Instant;

use super::{RepairAction, RepairPlan};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Objective function for repair synthesis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepairObjective {
    /// Minimize the number of parameter changes.
    MinimizeChanges,
    /// Minimize total deviation from current values.
    MinimizeDeviation,
    /// Maximize post-repair robustness margin.
    MaximizeRobustness,
    /// Balanced trade-off across all objectives.
    Balanced,
}

/// Allowed parameter bounds for repair synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterBounds {
    pub min_retry: u32,
    pub max_retry: u32,
    pub min_timeout_ms: u64,
    pub max_timeout_ms: u64,
}

impl Default for ParameterBounds {
    fn default() -> Self {
        Self {
            min_retry: 0,
            max_retry: 10,
            min_timeout_ms: 100,
            max_timeout_ms: 60_000,
        }
    }
}

/// Statistics collected during a synthesis run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RepairStatistics {
    pub synthesis_time_ms: u64,
    pub solver_calls: u32,
    pub repairs_found: usize,
}

/// A risky path detected during analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskyPathInfo {
    /// Ordered sequence of service names along the path.
    pub path: Vec<String>,
    /// Current retry-amplification product along the path.
    pub amplification: f64,
    /// Current cumulative timeout along the path (ms).
    pub timeout_ms: u64,
    /// Maximum allowed amplification (threshold).
    pub threshold: f64,
    /// Maximum allowed cumulative timeout (deadline).
    pub deadline_ms: u64,
}

impl RiskyPathInfo {
    pub fn has_amplification_violation(&self) -> bool {
        self.amplification > self.threshold && self.threshold > 0.0
    }

    pub fn has_timeout_violation(&self) -> bool {
        self.timeout_ms > self.deadline_ms && self.deadline_ms > 0
    }
}

/// Result of a synthesis run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairResult {
    pub primary_repair: Option<RepairPlan>,
    pub alternatives: Vec<RepairPlan>,
    pub statistics: RepairStatistics,
}

// ---------------------------------------------------------------------------
// RepairSynthesizer
// ---------------------------------------------------------------------------

/// Synthesizes minimal-cost repair plans for cascade risk violations.
#[derive(Debug, Clone)]
pub struct RepairSynthesizer {
    pub bounds: ParameterBounds,
}

impl RepairSynthesizer {
    pub fn new(bounds: ParameterBounds) -> Self {
        Self { bounds }
    }

    // -- helpers ----------------------------------------------------------

    /// Look up the edge `(src, tgt)` in the adjacency list.
    fn find_edge<'a>(
        src: &str,
        tgt: &str,
        adj: &'a [(String, String, u32, u64)],
    ) -> Option<&'a (String, String, u32, u64)> {
        adj.iter().find(|e| e.0 == src && e.1 == tgt)
    }

    /// Compute the amplification product along a path given the adjacency.
    fn path_amplification(path: &[String], adj: &[(String, String, u32, u64)]) -> f64 {
        if path.len() < 2 {
            return 1.0;
        }
        let mut product = 1.0_f64;
        for w in path.windows(2) {
            if let Some(e) = Self::find_edge(&w[0], &w[1], adj) {
                product *= 1.0 + e.2 as f64;
            }
        }
        product
    }

    /// Compute the cumulative timeout along a path.
    fn path_timeout(path: &[String], adj: &[(String, String, u32, u64)]) -> u64 {
        if path.len() < 2 {
            return 0;
        }
        let mut total = 0u64;
        for w in path.windows(2) {
            if let Some(e) = Self::find_edge(&w[0], &w[1], adj) {
                total = total.saturating_add(e.3);
            }
        }
        total
    }

    /// Compute amplification with a modified retry count for one edge.
    fn path_amplification_with_override(
        path: &[String],
        adj: &[(String, String, u32, u64)],
        override_src: &str,
        override_tgt: &str,
        new_retry: u32,
    ) -> f64 {
        if path.len() < 2 {
            return 1.0;
        }
        let mut product = 1.0_f64;
        for w in path.windows(2) {
            if w[0] == override_src && w[1] == override_tgt {
                product *= 1.0 + new_retry as f64;
            } else if let Some(e) = Self::find_edge(&w[0], &w[1], adj) {
                product *= 1.0 + e.2 as f64;
            }
        }
        product
    }

    /// Compute timeout with a modified timeout for one edge.
    fn path_timeout_with_override(
        path: &[String],
        adj: &[(String, String, u32, u64)],
        override_src: &str,
        override_tgt: &str,
        new_timeout: u64,
    ) -> u64 {
        if path.len() < 2 {
            return 0;
        }
        let mut total = 0u64;
        for w in path.windows(2) {
            if w[0] == override_src && w[1] == override_tgt {
                total = total.saturating_add(new_timeout);
            } else if let Some(e) = Self::find_edge(&w[0], &w[1], adj) {
                total = total.saturating_add(e.3);
            }
        }
        total
    }

    // -- public API -------------------------------------------------------

    /// Top-level synthesis entry point.
    pub fn synthesize(
        &self,
        risks: &[RiskyPathInfo],
        adjacency: &[(String, String, u32, u64)],
        objective: &RepairObjective,
    ) -> RepairResult {
        let start = Instant::now();
        let mut solver_calls = 0u32;
        let mut plans: Vec<RepairPlan> = Vec::new();

        // Attempt an amplification fix for every amplification violation.
        for risk in risks.iter().filter(|r| r.has_amplification_violation()) {
            solver_calls += 1;
            if let Some(plan) =
                self.synthesize_amplification_fix(&risk.path, adjacency, risk.threshold)
            {
                plans.push(plan);
            }
        }

        // Attempt a timeout fix for every timeout violation.
        for risk in risks.iter().filter(|r| r.has_timeout_violation()) {
            solver_calls += 1;
            if let Some(plan) =
                self.synthesize_timeout_fix(&risk.path, adjacency, risk.deadline_ms)
            {
                plans.push(plan);
            }
        }

        // Merge plans into a single primary repair.
        let primary = Self::merge_plans(&plans);

        // Generate alternatives depending on objective.
        let mut alternatives = Vec::new();
        match objective {
            RepairObjective::MinimizeChanges => {
                // Already the primary strategy; offer a deviation-based alt.
                if let Some(alt) = self.deviation_alternative(risks, adjacency) {
                    solver_calls += 1;
                    alternatives.push(alt);
                }
            }
            RepairObjective::MinimizeDeviation => {
                if let Some(alt) = self.changes_alternative(risks, adjacency) {
                    solver_calls += 1;
                    alternatives.push(alt);
                }
            }
            RepairObjective::MaximizeRobustness => {
                if let Some(alt) = self.robustness_alternative(risks, adjacency) {
                    solver_calls += 1;
                    alternatives.push(alt);
                }
            }
            RepairObjective::Balanced => {
                if let Some(a1) = self.deviation_alternative(risks, adjacency) {
                    solver_calls += 1;
                    alternatives.push(a1);
                }
                if let Some(a2) = self.robustness_alternative(risks, adjacency) {
                    solver_calls += 1;
                    alternatives.push(a2);
                }
            }
        }

        let elapsed = start.elapsed().as_millis() as u64;
        let repairs_found = if primary.is_some() { 1 } else { 0 } + alternatives.len();
        RepairResult {
            primary_repair: primary,
            alternatives,
            statistics: RepairStatistics {
                synthesis_time_ms: elapsed,
                solver_calls,
                repairs_found,
            },
        }
    }

    /// Synthesize a plan that fixes an amplification violation on `path`.
    ///
    /// Strategy: greedily reduce the retry count on the edge whose reduction
    /// yields the largest amplification decrease per unit of deviation cost.
    pub fn synthesize_amplification_fix(
        &self,
        path: &[String],
        adj: &[(String, String, u32, u64)],
        threshold: f64,
    ) -> Option<RepairPlan> {
        if path.len() < 2 {
            return None;
        }
        let current_amp = Self::path_amplification(path, adj);
        if current_amp <= threshold {
            return None; // already within threshold
        }

        let mut plan = RepairPlan::default();
        plan.feasible = true;

        // Collect mutable copies of retries for edges on this path.
        let mut edge_retries: Vec<(String, String, u32, u64)> = Vec::new();
        for w in path.windows(2) {
            if let Some(e) = Self::find_edge(&w[0], &w[1], adj) {
                edge_retries.push(e.clone());
            }
        }

        // Greedy loop: reduce retries one at a time on the best edge.
        let max_iters = 100;
        for _ in 0..max_iters {
            let amp = Self::path_amplification_from_local(&edge_retries);
            if amp <= threshold {
                break;
            }

            // Pick the edge whose retry reduction gives the biggest drop.
            let mut best_idx: Option<usize> = None;
            let mut best_drop = 0.0_f64;
            for (i, er) in edge_retries.iter().enumerate() {
                if er.2 <= self.bounds.min_retry {
                    continue;
                }
                let new_retry = er.2 - 1;
                let mut trial = edge_retries.clone();
                trial[i].2 = new_retry;
                let new_amp = Self::path_amplification_from_local(&trial);
                let drop = amp - new_amp;
                if drop > best_drop {
                    best_drop = drop;
                    best_idx = Some(i);
                }
            }

            match best_idx {
                Some(idx) => {
                    edge_retries[idx].2 -= 1;
                }
                None => {
                    // Cannot reduce further.
                    plan.feasible = false;
                    break;
                }
            }
        }

        // Build actions from the changes.
        for w in path.windows(2) {
            if let Some(orig) = Self::find_edge(&w[0], &w[1], adj) {
                if let Some(modified) = edge_retries
                    .iter()
                    .find(|e| e.0 == w[0] && e.1 == w[1])
                {
                    if modified.2 != orig.2 {
                        let action =
                            RepairAction::reduce_retries(&w[0], &w[1], orig.2, modified.2);
                        plan.add_action(action);
                    }
                }
            }
        }

        if plan.is_empty() {
            None
        } else {
            Some(plan)
        }
    }

    /// Synthesize a plan that fixes a timeout-chain violation on `path`.
    ///
    /// Strategy: increase timeout on the edge with the smallest current
    /// timeout first, as that provides the best "bang for buck" in reducing
    /// the cumulative timeout while staying within bounds.
    ///
    /// Wait — increasing timeout makes the sum *larger*, not smaller. The
    /// timeout-chain violation means the cumulative timeout *exceeds* the
    /// deadline. To fix it we must *decrease* some edge timeouts.
    pub fn synthesize_timeout_fix(
        &self,
        path: &[String],
        adj: &[(String, String, u32, u64)],
        deadline_ms: u64,
    ) -> Option<RepairPlan> {
        if path.len() < 2 {
            return None;
        }
        let current_sum = Self::path_timeout(path, adj);
        if current_sum <= deadline_ms {
            return None;
        }

        let mut plan = RepairPlan::default();
        plan.feasible = true;

        // Collect mutable copies.
        let mut edge_timeouts: Vec<(String, String, u32, u64)> = Vec::new();
        for w in path.windows(2) {
            if let Some(e) = Self::find_edge(&w[0], &w[1], adj) {
                edge_timeouts.push(e.clone());
            }
        }

        let excess = current_sum - deadline_ms;
        // Distribute the required reduction proportionally.
        let total_timeout: u64 = edge_timeouts.iter().map(|e| e.3).sum();
        if total_timeout == 0 {
            return None;
        }

        let mut remaining = excess;
        // Sort edges by timeout descending so we trim the largest first.
        let mut indices: Vec<usize> = (0..edge_timeouts.len()).collect();
        indices.sort_by(|a, b| edge_timeouts[*b].3.cmp(&edge_timeouts[*a].3));

        for &i in &indices {
            if remaining == 0 {
                break;
            }
            let current_to = edge_timeouts[i].3;
            let min_to = self.bounds.min_timeout_ms;
            let reducible = current_to.saturating_sub(min_to);
            let reduce_by = remaining.min(reducible);
            if reduce_by > 0 {
                edge_timeouts[i].3 = current_to - reduce_by;
                remaining -= reduce_by;
            }
        }

        if remaining > 0 {
            plan.feasible = false;
        }

        // Build actions.
        for w in path.windows(2) {
            if let Some(orig) = Self::find_edge(&w[0], &w[1], adj) {
                if let Some(modified) = edge_timeouts
                    .iter()
                    .find(|e| e.0 == w[0] && e.1 == w[1])
                {
                    if modified.3 != orig.3 {
                        let action =
                            RepairAction::adjust_timeout(&w[0], &w[1], orig.3, modified.3);
                        plan.add_action(action);
                    }
                }
            }
        }

        if plan.is_empty() {
            None
        } else {
            Some(plan)
        }
    }

    /// Verify that applying `plan` would fix all risks.
    pub fn verify_repair(
        &self,
        plan: &RepairPlan,
        risks: &[RiskyPathInfo],
        adj: &[(String, String, u32, u64)],
    ) -> bool {
        let patched = Self::apply_plan_to_adj(plan, adj);
        for risk in risks {
            if risk.has_amplification_violation() {
                let new_amp = Self::path_amplification(&risk.path, &patched);
                if new_amp > risk.threshold {
                    return false;
                }
            }
            if risk.has_timeout_violation() {
                let new_to = Self::path_timeout(&risk.path, &patched);
                if new_to > risk.deadline_ms {
                    return false;
                }
            }
        }
        true
    }

    // -- private helpers --------------------------------------------------

    fn path_amplification_from_local(edges: &[(String, String, u32, u64)]) -> f64 {
        let mut product = 1.0_f64;
        for e in edges {
            product *= 1.0 + e.2 as f64;
        }
        product
    }

    /// Merge multiple plans into one, deduplicating overlapping actions.
    fn merge_plans(plans: &[RepairPlan]) -> Option<RepairPlan> {
        if plans.is_empty() {
            return None;
        }
        let mut merged = RepairPlan::default();
        merged.feasible = true;
        let mut seen_edges: HashSet<(String, String)> = HashSet::new();
        for p in plans {
            if !p.feasible {
                merged.feasible = false;
            }
            for action in &p.actions {
                let key = action.edge.clone().unwrap_or_default();
                if !seen_edges.contains(&key) {
                    seen_edges.insert(key);
                    merged.add_action(action.clone());
                }
            }
        }
        if merged.is_empty() {
            None
        } else {
            Some(merged)
        }
    }

    /// Create a modified adjacency list by applying a repair plan.
    fn apply_plan_to_adj(
        plan: &RepairPlan,
        adj: &[(String, String, u32, u64)],
    ) -> Vec<(String, String, u32, u64)> {
        let mut result: Vec<(String, String, u32, u64)> = adj.to_vec();
        for action in &plan.actions {
            if let Some((ref src, ref tgt)) = action.edge {
                if let Some(e) = result.iter_mut().find(|e| &e.0 == src && &e.1 == tgt) {
                    match &action.action_type {
                        super::RepairActionType::ReduceRetries { to, .. } => {
                            e.2 = *to;
                        }
                        super::RepairActionType::AdjustTimeout { to_ms, .. } => {
                            e.3 = *to_ms;
                        }
                        _ => {}
                    }
                }
            }
        }
        result
    }

    /// Alternative that minimizes deviation (reduce retries as little as
    /// possible, spreading changes across multiple edges).
    fn deviation_alternative(
        &self,
        risks: &[RiskyPathInfo],
        adj: &[(String, String, u32, u64)],
    ) -> Option<RepairPlan> {
        let mut plan = RepairPlan::default();
        plan.feasible = true;

        for risk in risks.iter().filter(|r| r.has_amplification_violation()) {
            // Spread reduction evenly across all path edges.
            let edges_on_path = self.collect_path_edges(&risk.path, adj);
            if edges_on_path.is_empty() {
                continue;
            }

            let target_amp = risk.threshold;
            let n = edges_on_path.len() as f64;
            // Each factor should be at most target_amp^(1/n).
            let per_edge_max = target_amp.powf(1.0 / n);

            for (src, tgt, retries, _to) in &edges_on_path {
                let current_factor = 1.0 + *retries as f64;
                if current_factor > per_edge_max {
                    let new_retries = ((per_edge_max - 1.0).floor() as u32).max(self.bounds.min_retry);
                    if new_retries < *retries {
                        plan.add_action(RepairAction::reduce_retries(src, tgt, *retries, new_retries));
                    }
                }
            }
        }

        if plan.is_empty() { None } else { Some(plan) }
    }

    /// Alternative that minimizes the number of changes.
    fn changes_alternative(
        &self,
        risks: &[RiskyPathInfo],
        adj: &[(String, String, u32, u64)],
    ) -> Option<RepairPlan> {
        let mut plan = RepairPlan::default();
        plan.feasible = true;

        for risk in risks.iter().filter(|r| r.has_amplification_violation()) {
            // Find the single edge with highest retry count and reduce it aggressively.
            let edges_on_path = self.collect_path_edges(&risk.path, adj);
            if let Some(worst) = edges_on_path.iter().max_by_key(|e| e.2) {
                let needed = self.min_retry_for_threshold(
                    &risk.path,
                    adj,
                    &worst.0,
                    &worst.1,
                    risk.threshold,
                );
                let new_retries = needed.max(self.bounds.min_retry);
                if new_retries < worst.2 {
                    plan.add_action(RepairAction::reduce_retries(
                        &worst.0,
                        &worst.1,
                        worst.2,
                        new_retries,
                    ));
                }
            }
        }

        if plan.is_empty() { None } else { Some(plan) }
    }

    /// Alternative that maximizes robustness margin.
    fn robustness_alternative(
        &self,
        risks: &[RiskyPathInfo],
        adj: &[(String, String, u32, u64)],
    ) -> Option<RepairPlan> {
        let mut plan = RepairPlan::default();
        plan.feasible = true;

        // Reduce all retries on risky paths to minimum bounds.
        for risk in risks.iter().filter(|r| r.has_amplification_violation()) {
            let edges_on_path = self.collect_path_edges(&risk.path, adj);
            for (src, tgt, retries, _) in &edges_on_path {
                if *retries > self.bounds.min_retry {
                    plan.add_action(RepairAction::reduce_retries(
                        src,
                        tgt,
                        *retries,
                        self.bounds.min_retry,
                    ));
                }
            }
        }

        // Reduce all timeouts on risky paths to minimum bounds.
        for risk in risks.iter().filter(|r| r.has_timeout_violation()) {
            let edges_on_path = self.collect_path_edges(&risk.path, adj);
            for (src, tgt, _, timeout) in &edges_on_path {
                if *timeout > self.bounds.min_timeout_ms {
                    plan.add_action(RepairAction::adjust_timeout(
                        src,
                        tgt,
                        *timeout,
                        self.bounds.min_timeout_ms,
                    ));
                }
            }
        }

        if plan.is_empty() { None } else { Some(plan) }
    }

    fn collect_path_edges(
        &self,
        path: &[String],
        adj: &[(String, String, u32, u64)],
    ) -> Vec<(String, String, u32, u64)> {
        let mut out = Vec::new();
        for w in path.windows(2) {
            if let Some(e) = Self::find_edge(&w[0], &w[1], adj) {
                out.push(e.clone());
            }
        }
        out
    }

    /// Find the minimum retry count for a single edge that brings path
    /// amplification at or below `threshold`.
    fn min_retry_for_threshold(
        &self,
        path: &[String],
        adj: &[(String, String, u32, u64)],
        src: &str,
        tgt: &str,
        threshold: f64,
    ) -> u32 {
        for r in 0..=self.bounds.max_retry {
            let amp = Self::path_amplification_with_override(path, adj, src, tgt, r);
            if amp <= threshold {
                return r;
            }
        }
        self.bounds.min_retry
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
            amplification: 6.0 * 5.0 * 4.0, // 120
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
    fn test_risky_path_amplification_violation() {
        let r = amp_risk();
        assert!(r.has_amplification_violation());
        assert!(!r.has_timeout_violation());
    }

    #[test]
    fn test_risky_path_timeout_violation() {
        let r = timeout_risk();
        assert!(!r.has_amplification_violation());
        assert!(r.has_timeout_violation());
    }

    #[test]
    fn test_path_amplification_calculation() {
        let adj = sample_adj();
        let path: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
        let amp = RepairSynthesizer::path_amplification(&path, &adj);
        // (1+5) * (1+4) = 30
        assert!((amp - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_path_timeout_calculation() {
        let adj = sample_adj();
        let path: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
        let t = RepairSynthesizer::path_timeout(&path, &adj);
        assert_eq!(t, 7000);
    }

    #[test]
    fn test_synthesize_amplification_fix() {
        let synth = RepairSynthesizer::new(default_bounds());
        let adj = sample_adj();
        let path: Vec<String> = vec!["A".into(), "B".into(), "C".into(), "D".into()];
        let plan = synth.synthesize_amplification_fix(&path, &adj, 30.0);
        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert!(plan.feasible);
        assert!(!plan.actions.is_empty());
        // Verify the fix actually works.
        let patched = RepairSynthesizer::apply_plan_to_adj(&plan, &adj);
        let new_amp = RepairSynthesizer::path_amplification(&path, &patched);
        assert!(new_amp <= 30.0, "new amp {} should be <= 30", new_amp);
    }

    #[test]
    fn test_synthesize_timeout_fix() {
        let synth = RepairSynthesizer::new(default_bounds());
        let adj = sample_adj();
        let path: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
        let plan = synth.synthesize_timeout_fix(&path, &adj, 5000);
        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert!(plan.feasible);
        let patched = RepairSynthesizer::apply_plan_to_adj(&plan, &adj);
        let new_to = RepairSynthesizer::path_timeout(&path, &patched);
        assert!(new_to <= 5000, "new timeout {} should be <= 5000", new_to);
    }

    #[test]
    fn test_synthesize_returns_no_plan_when_within_threshold() {
        let synth = RepairSynthesizer::new(default_bounds());
        let adj = vec![("A".to_string(), "B".to_string(), 1u32, 1000u64)];
        let path: Vec<String> = vec!["A".into(), "B".into()];
        let plan = synth.synthesize_amplification_fix(&path, &adj, 100.0);
        assert!(plan.is_none());
    }

    #[test]
    fn test_verify_repair() {
        let synth = RepairSynthesizer::new(default_bounds());
        let adj = sample_adj();
        let risk = amp_risk();
        let plan = synth
            .synthesize_amplification_fix(&risk.path, &adj, risk.threshold)
            .unwrap();
        assert!(synth.verify_repair(&plan, &[risk], &adj));
    }

    #[test]
    fn test_full_synthesize_balanced() {
        let synth = RepairSynthesizer::new(default_bounds());
        let adj = sample_adj();
        let risks = vec![amp_risk()];
        let result = synth.synthesize(&risks, &adj, &RepairObjective::Balanced);
        assert!(result.primary_repair.is_some());
        assert!(result.statistics.solver_calls > 0);
    }

    #[test]
    fn test_full_synthesize_minimize_changes() {
        let synth = RepairSynthesizer::new(default_bounds());
        let adj = sample_adj();
        let risks = vec![amp_risk()];
        let result = synth.synthesize(&risks, &adj, &RepairObjective::MinimizeChanges);
        assert!(result.primary_repair.is_some());
    }

    #[test]
    fn test_empty_risks() {
        let synth = RepairSynthesizer::new(default_bounds());
        let result = synth.synthesize(&[], &[], &RepairObjective::Balanced);
        assert!(result.primary_repair.is_none());
        assert!(result.alternatives.is_empty());
    }

    #[test]
    fn test_single_edge_path() {
        let synth = RepairSynthesizer::new(default_bounds());
        let adj = vec![("X".to_string(), "Y".to_string(), 8u32, 2000u64)];
        let path: Vec<String> = vec!["X".into(), "Y".into()];
        // Amplification = 1+8 = 9, threshold = 5
        let plan = synth.synthesize_amplification_fix(&path, &adj, 5.0);
        assert!(plan.is_some());
        let plan = plan.unwrap();
        let patched = RepairSynthesizer::apply_plan_to_adj(&plan, &adj);
        let new_amp = RepairSynthesizer::path_amplification(&path, &patched);
        assert!(new_amp <= 5.0);
    }
}
