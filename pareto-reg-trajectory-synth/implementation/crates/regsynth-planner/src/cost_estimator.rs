use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::roadmap::RoadmapTask;

// ─── Three-Point Estimate (PERT) ────────────────────────────────────────────

/// PERT three-point estimate: optimistic (O), most likely (M), pessimistic (P).
/// Expected value = (O + 4M + P) / 6.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreePointEstimate {
    pub optimistic: f64,
    pub most_likely: f64,
    pub pessimistic: f64,
}

impl ThreePointEstimate {
    pub fn new(optimistic: f64, most_likely: f64, pessimistic: f64) -> Self {
        Self { optimistic, most_likely, pessimistic }
    }

    /// PERT-weighted expected value.
    pub fn expected(&self) -> f64 {
        (self.optimistic + 4.0 * self.most_likely + self.pessimistic) / 6.0
    }

    /// Standard deviation estimate σ = (P − O) / 6.
    pub fn std_deviation(&self) -> f64 {
        (self.pessimistic - self.optimistic) / 6.0
    }

    /// Variance σ² = ((P − O) / 6)².
    pub fn variance(&self) -> f64 {
        let sd = self.std_deviation();
        sd * sd
    }

    /// Confidence interval at the given number of standard deviations.
    pub fn confidence_interval(&self, sigma_count: f64) -> (f64, f64) {
        let mean = self.expected();
        let sd = self.std_deviation();
        (mean - sigma_count * sd, mean + sigma_count * sd)
    }

    /// Validate that optimistic ≤ most_likely ≤ pessimistic.
    pub fn is_valid(&self) -> bool {
        self.optimistic <= self.most_likely && self.most_likely <= self.pessimistic
    }
}

// ─── Cost Estimate ──────────────────────────────────────────────────────────

/// Cost estimate for a single task with PERT distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    pub task_id: String,
    pub estimate: ThreePointEstimate,
    pub currency: String,
    /// Optional category tag (e.g. "personnel", "tooling", "audit").
    #[serde(default)]
    pub category: Option<String>,
}

impl CostEstimate {
    pub fn new(task_id: impl Into<String>, estimate: ThreePointEstimate, currency: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            estimate,
            currency: currency.into(),
            category: None,
        }
    }

    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }
}

// ─── Aggregate Cost Summary ─────────────────────────────────────────────────

/// Summary statistics for a collection of cost estimates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostSummary {
    pub total_expected: f64,
    pub total_optimistic: f64,
    pub total_pessimistic: f64,
    pub total_std_deviation: f64,
    pub confidence_90_low: f64,
    pub confidence_90_high: f64,
    pub by_category: HashMap<String, f64>,
    pub currency: String,
}

// ─── Cost Database ──────────────────────────────────────────────────────────

/// Historical cost database for calibrated estimates.
pub struct CostDatabase {
    pub entries: Vec<CostEstimate>,
    /// Historical actual costs used for calibration (task_id → actual cost).
    actuals: HashMap<String, f64>,
}

impl CostDatabase {
    pub fn new() -> Self {
        Self { entries: Vec::new(), actuals: HashMap::new() }
    }

    pub fn add(&mut self, entry: CostEstimate) {
        self.entries.push(entry);
    }

    pub fn lookup(&self, task_id: &str) -> Option<&CostEstimate> {
        self.entries.iter().find(|e| e.task_id == task_id)
    }

    /// Record actual cost for calibration.
    pub fn record_actual(&mut self, task_id: impl Into<String>, actual_cost: f64) {
        self.actuals.insert(task_id.into(), actual_cost);
    }

    /// Compute calibration factor: mean(actual / estimated) across tasks with both.
    /// A factor > 1 means estimates have been consistently low.
    pub fn calibration_factor(&self) -> f64 {
        let mut ratios = Vec::new();
        for entry in &self.entries {
            if let Some(&actual) = self.actuals.get(&entry.task_id) {
                let expected = entry.estimate.expected();
                if expected > 0.0 {
                    ratios.push(actual / expected);
                }
            }
        }
        if ratios.is_empty() {
            1.0
        } else {
            ratios.iter().sum::<f64>() / ratios.len() as f64
        }
    }

    /// Number of entries with historical actuals.
    pub fn actuals_count(&self) -> usize {
        self.actuals.len()
    }
}

impl Default for CostDatabase {
    fn default() -> Self { Self::new() }
}

// ─── Cost Estimator ─────────────────────────────────────────────────────────

/// Produces calibrated three-point cost estimates for tasks.
pub struct CostEstimator {
    pub database: CostDatabase,
    /// Default daily rate used when no explicit estimate exists.
    pub default_daily_rate: f64,
    /// Currency for generated estimates.
    pub currency: String,
}

impl CostEstimator {
    pub fn new(database: CostDatabase) -> Self {
        Self { database, default_daily_rate: 1000.0, currency: "USD".into() }
    }

    pub fn with_daily_rate(mut self, rate: f64) -> Self {
        self.default_daily_rate = rate;
        self
    }

    pub fn with_currency(mut self, currency: impl Into<String>) -> Self {
        self.currency = currency.into();
        self
    }

    /// Expected total cost across all entries, adjusted by calibration factor.
    pub fn estimate_total(&self) -> f64 {
        let raw: f64 = self.database.entries.iter().map(|e| e.estimate.expected()).sum();
        raw * self.database.calibration_factor()
    }

    /// Produce a cost estimate for a single task. Uses explicit data if available,
    /// otherwise generates from effort days and the default daily rate.
    pub fn estimate_task(&self, task: &RoadmapTask) -> CostEstimate {
        if let Some(existing) = self.database.lookup(&task.id) {
            return existing.clone();
        }

        let base = task.effort_days * self.default_daily_rate;
        let estimate = ThreePointEstimate::new(
            base * 0.7,
            base,
            base * 1.6,
        );

        CostEstimate::new(&task.id, estimate, &self.currency)
    }

    /// Produce a summary across a set of tasks.
    pub fn summarize(&self, tasks: &[RoadmapTask]) -> CostSummary {
        let estimates: Vec<CostEstimate> = tasks.iter().map(|t| self.estimate_task(t)).collect();
        let calibration = self.database.calibration_factor();

        let total_expected: f64 = estimates.iter().map(|e| e.estimate.expected()).sum::<f64>() * calibration;
        let total_optimistic: f64 = estimates.iter().map(|e| e.estimate.optimistic).sum::<f64>() * calibration;
        let total_pessimistic: f64 = estimates.iter().map(|e| e.estimate.pessimistic).sum::<f64>() * calibration;

        // Aggregate variance (assuming independence)
        let total_variance: f64 = estimates.iter().map(|e| e.estimate.variance()).sum::<f64>() * calibration * calibration;
        let total_sd = total_variance.sqrt();

        // 90% confidence ≈ ±1.645σ
        let ci90_low = total_expected - 1.645 * total_sd;
        let ci90_high = total_expected + 1.645 * total_sd;

        let mut by_category: HashMap<String, f64> = HashMap::new();
        for est in &estimates {
            let cat = est.category.clone().unwrap_or_else(|| "uncategorized".into());
            *by_category.entry(cat).or_insert(0.0) += est.estimate.expected() * calibration;
        }

        CostSummary {
            total_expected,
            total_optimistic,
            total_pessimistic,
            total_std_deviation: total_sd,
            confidence_90_low: ci90_low.max(0.0),
            confidence_90_high: ci90_high,
            by_category,
            currency: self.currency.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roadmap::RoadmapTask;

    fn make_task(id: &str, effort: f64) -> RoadmapTask {
        RoadmapTask::new(id, id).with_effort(effort)
    }

    #[test]
    fn test_three_point_expected() {
        let tp = ThreePointEstimate::new(1.0, 4.0, 7.0);
        assert!((tp.expected() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_three_point_std_dev() {
        let tp = ThreePointEstimate::new(1.0, 4.0, 7.0);
        assert!((tp.std_deviation() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_three_point_variance() {
        let tp = ThreePointEstimate::new(1.0, 4.0, 7.0);
        assert!((tp.variance() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_three_point_validity() {
        assert!(ThreePointEstimate::new(1.0, 2.0, 3.0).is_valid());
        assert!(!ThreePointEstimate::new(3.0, 2.0, 1.0).is_valid());
    }

    #[test]
    fn test_confidence_interval() {
        let tp = ThreePointEstimate::new(0.0, 6.0, 12.0);
        let (lo, hi) = tp.confidence_interval(1.0);
        assert!((lo - 4.0).abs() < 1e-9);
        assert!((hi - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_cost_database_lookup() {
        let mut db = CostDatabase::new();
        db.add(CostEstimate::new("t1", ThreePointEstimate::new(10.0, 20.0, 30.0), "USD"));
        assert!(db.lookup("t1").is_some());
        assert!(db.lookup("t2").is_none());
    }

    #[test]
    fn test_calibration_factor_no_actuals() {
        let db = CostDatabase::new();
        assert!((db.calibration_factor() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_calibration_factor_with_actuals() {
        let mut db = CostDatabase::new();
        db.add(CostEstimate::new("t1", ThreePointEstimate::new(80.0, 100.0, 120.0), "USD"));
        // Expected = 100, actual = 150 → factor = 1.5
        db.record_actual("t1", 150.0);
        assert!((db.calibration_factor() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_total() {
        let mut db = CostDatabase::new();
        db.add(CostEstimate::new("t1", ThreePointEstimate::new(100.0, 200.0, 300.0), "USD"));
        db.add(CostEstimate::new("t2", ThreePointEstimate::new(50.0, 100.0, 150.0), "USD"));
        let estimator = CostEstimator::new(db);
        let total = estimator.estimate_total();
        // 200 + 100 = 300 (cal factor = 1.0)
        assert!((total - 300.0).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_task_fallback() {
        let db = CostDatabase::new();
        let estimator = CostEstimator::new(db).with_daily_rate(500.0);
        let task = make_task("unknown", 10.0);
        let est = estimator.estimate_task(&task);
        assert_eq!(est.task_id, "unknown");
        // most_likely = 10 * 500 = 5000
        assert!((est.estimate.most_likely - 5000.0).abs() < 1e-9);
    }

    #[test]
    fn test_summarize() {
        let mut db = CostDatabase::new();
        db.add(CostEstimate::new("t1", ThreePointEstimate::new(70.0, 100.0, 160.0), "USD").with_category("personnel"));
        let estimator = CostEstimator::new(db).with_daily_rate(500.0);

        let tasks = vec![make_task("t1", 5.0), make_task("t2", 10.0)];
        let summary = estimator.summarize(&tasks);
        assert!(summary.total_expected > 0.0);
        assert!(summary.confidence_90_low < summary.total_expected);
        assert!(summary.confidence_90_high > summary.total_expected);
        assert!(summary.by_category.contains_key("personnel"));
    }
}
