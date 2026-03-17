//! Trajectory data management, statistics, convergence detection, and analysis.
//!
//! - [`TrajectoryBuilder`]: incremental construction of trajectories
//! - [`TrajectoryAnalyzer`]: summary statistics, windowed stats, convergence
//! - Cycle detection in price sequences
//! - Autocorrelation and stationarity tests
//! - Data export/import and trajectory merging

use crate::types::*;
use crate::{MarketSimError, MarketSimResult};

// ════════════════════════════════════════════════════════════════════════════
// TrajectoryBuilder
// ════════════════════════════════════════════════════════════════════════════

/// Incremental builder for constructing [`PriceTrajectory`] instances.
#[derive(Debug, Clone)]
pub struct TrajectoryBuilder {
    num_players: usize,
    outcomes: Vec<MarketOutcome>,
    #[allow(dead_code)]
    capacity: usize,
}

impl TrajectoryBuilder {
    pub fn new(num_players: usize) -> Self {
        Self {
            num_players,
            outcomes: Vec::new(),
            capacity: 0,
        }
    }

    pub fn with_capacity(num_players: usize, capacity: usize) -> Self {
        Self {
            num_players,
            outcomes: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, outcome: MarketOutcome) -> &mut Self {
        self.outcomes.push(outcome);
        self
    }

    pub fn push_from_values(
        &mut self,
        round: RoundNumber,
        prices: Vec<f64>,
        quantities: Vec<f64>,
        profits: Vec<f64>,
    ) -> &mut Self {
        self.outcomes.push(MarketOutcome::new(round, prices, quantities, profits));
        self
    }

    pub fn len(&self) -> usize {
        self.outcomes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.outcomes.is_empty()
    }

    /// Build the trajectory, consuming the builder.
    pub fn build(self) -> PriceTrajectory {
        PriceTrajectory::with_outcomes(self.num_players, self.outcomes)
    }

    /// Build and keep the builder intact.
    pub fn snapshot(&self) -> PriceTrajectory {
        PriceTrajectory::with_outcomes(self.num_players, self.outcomes.clone())
    }

    /// Truncate to the last `n` outcomes.
    pub fn truncate_to_last(&mut self, n: usize) -> &mut Self {
        if self.outcomes.len() > n {
            let start = self.outcomes.len() - n;
            self.outcomes = self.outcomes[start..].to_vec();
        }
        self
    }

    /// Extend with outcomes from another trajectory.
    pub fn extend(&mut self, other: &PriceTrajectory) -> &mut Self {
        self.outcomes.extend(other.outcomes.iter().cloned());
        self
    }
}

// ════════════════════════════════════════════════════════════════════════════
// TrajectoryAnalyzer
// ════════════════════════════════════════════════════════════════════════════

/// Computes summary and windowed statistics from a [`PriceTrajectory`].
#[derive(Debug, Clone)]
pub struct TrajectoryAnalyzer<'a> {
    trajectory: &'a PriceTrajectory,
}

impl<'a> TrajectoryAnalyzer<'a> {
    pub fn new(trajectory: &'a PriceTrajectory) -> Self {
        Self { trajectory }
    }

    pub fn num_players(&self) -> usize {
        self.trajectory.num_players
    }

    pub fn num_rounds(&self) -> usize {
        self.trajectory.outcomes.len()
    }

    // ── Global statistics ───────────────────────────────────────────────

    /// Mean price for each player over the full trajectory.
    pub fn mean_prices(&self) -> Vec<f64> {
        self.trajectory.mean_prices()
    }

    /// Mean profit for each player over the full trajectory.
    pub fn mean_profits(&self) -> Vec<f64> {
        self.trajectory.mean_profits()
    }

    /// Variance of prices for each player.
    pub fn price_variance(&self) -> Vec<f64> {
        let n = self.trajectory.num_players;
        let t = self.num_rounds() as f64;
        if t < 2.0 {
            return vec![0.0; n];
        }
        let means = self.mean_prices();
        let mut var = vec![0.0; n];
        for o in &self.trajectory.outcomes {
            for i in 0..n {
                let diff = o.prices[i] - means[i];
                var[i] += diff * diff;
            }
        }
        var.iter().map(|v| v / (t - 1.0)).collect()
    }

    /// Standard deviation of prices.
    pub fn price_std(&self) -> Vec<f64> {
        self.price_variance().iter().map(|v| v.sqrt()).collect()
    }

    /// Variance of profits for each player.
    pub fn profit_variance(&self) -> Vec<f64> {
        let n = self.trajectory.num_players;
        let t = self.num_rounds() as f64;
        if t < 2.0 {
            return vec![0.0; n];
        }
        let means = self.mean_profits();
        let mut var = vec![0.0; n];
        for o in &self.trajectory.outcomes {
            for i in 0..n {
                let diff = o.profits[i] - means[i];
                var[i] += diff * diff;
            }
        }
        var.iter().map(|v| v / (t - 1.0)).collect()
    }

    /// Min and max prices for each player.
    pub fn price_range(&self) -> Vec<(f64, f64)> {
        let n = self.trajectory.num_players;
        let mut ranges = vec![(f64::INFINITY, f64::NEG_INFINITY); n];
        for o in &self.trajectory.outcomes {
            for i in 0..n {
                ranges[i].0 = ranges[i].0.min(o.prices[i]);
                ranges[i].1 = ranges[i].1.max(o.prices[i]);
            }
        }
        ranges
    }

    // ── Windowed statistics ─────────────────────────────────────────────

    /// Rolling mean of prices with a given window size.
    pub fn rolling_mean_prices(&self, window: usize) -> Vec<Vec<f64>> {
        let n = self.trajectory.num_players;
        let t = self.num_rounds();
        if window == 0 || t == 0 {
            return vec![];
        }
        let mut result = Vec::with_capacity(t.saturating_sub(window - 1));
        let mut sums = vec![0.0; n];

        for (idx, o) in self.trajectory.outcomes.iter().enumerate() {
            for i in 0..n {
                sums[i] += o.prices[i];
            }
            if idx >= window {
                let old = &self.trajectory.outcomes[idx - window];
                for i in 0..n {
                    sums[i] -= old.prices[i];
                }
            }
            if idx + 1 >= window {
                let w = window as f64;
                result.push(sums.iter().map(|s| s / w).collect());
            }
        }
        result
    }

    /// Rolling variance of prices.
    pub fn rolling_variance_prices(&self, window: usize) -> Vec<Vec<f64>> {
        let means = self.rolling_mean_prices(window);
        let n = self.trajectory.num_players;
        let t = self.num_rounds();
        if window < 2 || t < window {
            return vec![];
        }

        let mut result = Vec::with_capacity(means.len());
        for (start_offset, mean_vec) in means.iter().enumerate() {
            let mut var = vec![0.0; n];
            for k in 0..window {
                let o = &self.trajectory.outcomes[start_offset + k];
                for i in 0..n {
                    let diff = o.prices[i] - mean_vec[i];
                    var[i] += diff * diff;
                }
            }
            result.push(var.iter().map(|v| v / (window as f64 - 1.0)).collect());
        }
        result
    }

    // ── Convergence detection ───────────────────────────────────────────

    /// Detect price convergence: check if rolling variance drops below threshold.
    /// Returns the round at which convergence is first detected, or None.
    pub fn detect_convergence(
        &self,
        window: usize,
        variance_threshold: f64,
    ) -> Option<u64> {
        let rolling_var = self.rolling_variance_prices(window);

        for (idx, var_vec) in rolling_var.iter().enumerate() {
            let all_converged = var_vec.iter().all(|&v| v < variance_threshold);
            if all_converged {
                let round = (idx + window) as u64;
                return Some(round);
            }
        }
        None
    }

    /// Check if prices have converged to a steady state in the last `tail` rounds.
    pub fn is_converged(&self, tail: usize, tolerance: f64) -> bool {
        let t = self.num_rounds();
        if t < tail {
            return false;
        }
        let start = t - tail;
        let n = self.trajectory.num_players;

        for i in 0..n {
            let prices: Vec<f64> = self.trajectory.outcomes[start..]
                .iter()
                .map(|o| o.prices[i])
                .collect();
            let mean = prices.iter().sum::<f64>() / prices.len() as f64;
            let max_dev = prices.iter().map(|p| (p - mean).abs()).fold(0.0_f64, f64::max);
            if max_dev > tolerance {
                return false;
            }
        }
        true
    }

    // ── Cycle detection ─────────────────────────────────────────────────

    /// Detect cycles in price sequences (e.g., edgeworth cycles).
    /// Looks for repeating patterns of length up to `max_period`.
    /// Returns (period, start_round) if a cycle is found.
    pub fn detect_cycle(
        &self,
        player: PlayerId,
        max_period: usize,
        tolerance: f64,
        min_repetitions: usize,
    ) -> Option<(usize, usize)> {
        let prices = self.trajectory.prices_for_player(player);
        let t = prices.len();

        for period in 2..=max_period.min(t / 2) {
            let needed = period * min_repetitions;
            if needed > t {
                continue;
            }
            // Check from the end of the trajectory
            let start = t - needed;
            let mut is_cycle = true;

            'check: for offset in period..needed {
                let expected = prices[start + offset % period];
                let actual = prices[start + offset];
                if (expected - actual).abs() > tolerance {
                    is_cycle = false;
                    break 'check;
                }
            }

            if is_cycle {
                return Some((period, start));
            }
        }
        None
    }

    // ── Autocorrelation ─────────────────────────────────────────────────

    /// Compute autocorrelation of prices at given lag for a player.
    pub fn autocorrelation(&self, player: PlayerId, lag: usize) -> f64 {
        let prices = self.trajectory.prices_for_player(player);
        let t = prices.len();
        if t <= lag + 1 {
            return 0.0;
        }
        let mean = prices.iter().sum::<f64>() / t as f64;
        let var: f64 = prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / t as f64;
        if var.abs() < 1e-15 {
            return 1.0; // constant series
        }
        let cov: f64 = (0..t - lag)
            .map(|i| (prices[i] - mean) * (prices[i + lag] - mean))
            .sum::<f64>()
            / t as f64;
        cov / var
    }

    /// Compute autocorrelation function (ACF) for lags 0..max_lag.
    pub fn acf(&self, player: PlayerId, max_lag: usize) -> Vec<f64> {
        (0..=max_lag).map(|lag| self.autocorrelation(player, lag)).collect()
    }

    // ── Stationarity test (simplified ADF-like) ─────────────────────────

    /// Simple stationarity test based on the augmented Dickey-Fuller principle.
    ///
    /// Regresses Δp_t = α + β*p_{t-1} + ε_t and tests whether β < 0.
    /// Returns the t-statistic for β. A very negative value suggests stationarity.
    /// Critical values (approximate): -2.86 (5%), -3.43 (1%) for ADF with constant.
    pub fn adf_statistic(&self, player: PlayerId) -> f64 {
        let prices = self.trajectory.prices_for_player(player);
        let t = prices.len();
        if t < 4 {
            return 0.0;
        }

        // Δp_t = α + β * p_{t-1}  →  OLS for β
        let n = t - 1;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for i in 1..t {
            let x = prices[i - 1];
            let y = prices[i] - prices[i - 1];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let nf = n as f64;
        let denom = sum_xx - sum_x * sum_x / nf;
        if denom.abs() < 1e-15 {
            return 0.0;
        }

        let beta = (sum_xy - sum_x * sum_y / nf) / denom;
        let alpha = (sum_y - beta * sum_x) / nf;

        // Compute residual variance
        let mut rss = 0.0;
        for i in 1..t {
            let predicted = alpha + beta * prices[i - 1];
            let actual = prices[i] - prices[i - 1];
            rss += (actual - predicted).powi(2);
        }
        let sigma2 = rss / (nf - 2.0).max(1.0);
        let se_beta = (sigma2 / denom).sqrt();

        if se_beta.abs() < 1e-15 {
            return 0.0;
        }
        beta / se_beta
    }

    /// Check if a series is likely stationary at 5% significance level.
    pub fn is_stationary(&self, player: PlayerId) -> bool {
        let stat = self.adf_statistic(player);
        stat < -2.86 // 5% critical value for ADF with constant
    }

    // ── Segment splitting ───────────────────────────────────────────────

    /// Split the trajectory into non-overlapping segments of given length.
    pub fn split_segments(&self, segment_length: usize) -> Vec<PriceTrajectory> {
        let t = self.num_rounds();
        let n = self.trajectory.num_players;
        let mut segments = Vec::new();

        let mut start = 0;
        while start + segment_length <= t {
            let outcomes = self.trajectory.outcomes[start..start + segment_length].to_vec();
            segments.push(PriceTrajectory::with_outcomes(n, outcomes));
            start += segment_length;
        }
        segments
    }

    /// Split into training, validation, holdout segments by fraction.
    pub fn split_train_val_holdout(
        &self,
        train_frac: f64,
        val_frac: f64,
    ) -> (PriceTrajectory, PriceTrajectory, PriceTrajectory) {
        let t = self.num_rounds();
        let n = self.trajectory.num_players;
        let train_end = (t as f64 * train_frac) as usize;
        let val_end = train_end + (t as f64 * val_frac) as usize;

        let train = PriceTrajectory::with_outcomes(
            n,
            self.trajectory.outcomes[..train_end].to_vec(),
        );
        let val = PriceTrajectory::with_outcomes(
            n,
            self.trajectory.outcomes[train_end..val_end].to_vec(),
        );
        let holdout = PriceTrajectory::with_outcomes(
            n,
            self.trajectory.outcomes[val_end..].to_vec(),
        );
        (train, val, holdout)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Trajectory merging and alignment
// ════════════════════════════════════════════════════════════════════════════

/// Merge multiple trajectories into one by concatenation.
pub fn merge_trajectories(trajectories: &[PriceTrajectory]) -> MarketSimResult<PriceTrajectory> {
    if trajectories.is_empty() {
        return Err(MarketSimError::TrajectoryError(
            "Cannot merge empty trajectory list".into(),
        ));
    }
    let n = trajectories[0].num_players;
    for t in &trajectories[1..] {
        if t.num_players != n {
            return Err(MarketSimError::TrajectoryError(
                "Cannot merge trajectories with different player counts".into(),
            ));
        }
    }
    let mut all_outcomes = Vec::new();
    let mut round_offset = 0u64;
    for t in trajectories {
        for o in &t.outcomes {
            let mut adjusted = o.clone();
            adjusted.round = round_offset + o.round;
            all_outcomes.push(adjusted);
        }
        round_offset += t.outcomes.len() as u64;
    }
    Ok(PriceTrajectory::with_outcomes(n, all_outcomes))
}

/// Align two trajectories to the same length by truncating the longer one.
pub fn align_trajectories(
    a: &PriceTrajectory,
    b: &PriceTrajectory,
) -> (PriceTrajectory, PriceTrajectory) {
    let min_len = a.outcomes.len().min(b.outcomes.len());
    let a_aligned = PriceTrajectory::with_outcomes(
        a.num_players,
        a.outcomes[..min_len].to_vec(),
    );
    let b_aligned = PriceTrajectory::with_outcomes(
        b.num_players,
        b.outcomes[..min_len].to_vec(),
    );
    (a_aligned, b_aligned)
}

// ════════════════════════════════════════════════════════════════════════════
// Data export / import
// ════════════════════════════════════════════════════════════════════════════

/// Export trajectory to CSV-like string.
pub fn trajectory_to_csv(trajectory: &PriceTrajectory) -> String {
    let n = trajectory.num_players;
    let mut lines = Vec::new();

    // Header
    let mut header = "round".to_string();
    for i in 0..n {
        header.push_str(&format!(",price_{i},quantity_{i},profit_{i}"));
    }
    lines.push(header);

    // Data rows
    for o in &trajectory.outcomes {
        let mut row = format!("{}", o.round);
        for i in 0..n {
            row.push_str(&format!(
                ",{:.6},{:.6},{:.6}",
                o.prices[i], o.quantities[i], o.profits[i]
            ));
        }
        lines.push(row);
    }

    lines.join("\n")
}

/// Import trajectory from CSV-like string.
pub fn trajectory_from_csv(csv: &str, num_players: usize) -> MarketSimResult<PriceTrajectory> {
    let mut outcomes = Vec::new();
    let lines: Vec<&str> = csv.lines().collect();
    if lines.is_empty() {
        return Ok(PriceTrajectory::new(num_players));
    }

    // Skip header
    for line in &lines[1..] {
        let fields: Vec<&str> = line.split(',').collect();
        let expected = 1 + 3 * num_players; // round + 3 fields per player
        if fields.len() != expected {
            return Err(MarketSimError::TrajectoryError(format!(
                "Expected {expected} fields, got {}",
                fields.len()
            )));
        }
        let round: u64 = fields[0]
            .parse()
            .map_err(|_| MarketSimError::TrajectoryError("Invalid round number".into()))?;
        let mut prices = Vec::with_capacity(num_players);
        let mut quantities = Vec::with_capacity(num_players);
        let mut profits = Vec::with_capacity(num_players);
        for i in 0..num_players {
            let base = 1 + 3 * i;
            let p: f64 = fields[base]
                .parse()
                .map_err(|_| MarketSimError::TrajectoryError("Invalid price".into()))?;
            let q: f64 = fields[base + 1]
                .parse()
                .map_err(|_| MarketSimError::TrajectoryError("Invalid quantity".into()))?;
            let pi: f64 = fields[base + 2]
                .parse()
                .map_err(|_| MarketSimError::TrajectoryError("Invalid profit".into()))?;
            prices.push(p);
            quantities.push(q);
            profits.push(pi);
        }
        outcomes.push(MarketOutcome::new(round, prices, quantities, profits));
    }

    Ok(PriceTrajectory::with_outcomes(num_players, outcomes))
}

/// Export trajectory to JSON.
pub fn trajectory_to_json(trajectory: &PriceTrajectory) -> MarketSimResult<String> {
    serde_json::to_string_pretty(trajectory)
        .map_err(|e| MarketSimError::TrajectoryError(format!("JSON serialization failed: {e}")))
}

/// Import trajectory from JSON.
pub fn trajectory_from_json(json: &str) -> MarketSimResult<PriceTrajectory> {
    serde_json::from_str(json)
        .map_err(|e| MarketSimError::TrajectoryError(format!("JSON deserialization failed: {e}")))
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn make_trajectory(n: usize, rounds: usize, base_price: f64) -> PriceTrajectory {
        let mut builder = TrajectoryBuilder::new(n);
        for r in 0..rounds {
            let p = base_price + 0.01 * r as f64;
            builder.push_from_values(
                r as u64,
                vec![p; n],
                vec![10.0 - p; n],
                vec![(p - 1.0) * (10.0 - p); n],
            );
        }
        builder.build()
    }

    fn make_constant_trajectory(n: usize, rounds: usize, price: f64) -> PriceTrajectory {
        let mut builder = TrajectoryBuilder::new(n);
        for r in 0..rounds {
            builder.push_from_values(
                r as u64,
                vec![price; n],
                vec![10.0 - price; n],
                vec![(price - 1.0) * (10.0 - price); n],
            );
        }
        builder.build()
    }

    #[test]
    fn test_builder_basic() {
        let mut builder = TrajectoryBuilder::new(2);
        builder.push_from_values(0, vec![3.0, 3.0], vec![7.0, 7.0], vec![14.0, 14.0]);
        builder.push_from_values(1, vec![4.0, 4.0], vec![6.0, 6.0], vec![18.0, 18.0]);
        let traj = builder.build();
        assert_eq!(traj.len(), 2);
        assert_eq!(traj.num_players, 2);
    }

    #[test]
    fn test_builder_truncate() {
        let mut builder = TrajectoryBuilder::new(2);
        for r in 0..100 {
            builder.push_from_values(r, vec![3.0, 3.0], vec![7.0, 7.0], vec![14.0, 14.0]);
        }
        builder.truncate_to_last(10);
        assert_eq!(builder.len(), 10);
    }

    #[test]
    fn test_analyzer_mean_prices() {
        let traj = make_constant_trajectory(2, 100, 3.0);
        let analyzer = TrajectoryAnalyzer::new(&traj);
        let means = analyzer.mean_prices();
        assert!((means[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_analyzer_price_variance() {
        let traj = make_constant_trajectory(2, 100, 3.0);
        let analyzer = TrajectoryAnalyzer::new(&traj);
        let var = analyzer.price_variance();
        assert!(var[0] < 1e-10); // constant → zero variance
    }

    #[test]
    fn test_analyzer_price_range() {
        let traj = make_trajectory(2, 100, 3.0);
        let analyzer = TrajectoryAnalyzer::new(&traj);
        let ranges = analyzer.price_range();
        assert!((ranges[0].0 - 3.0).abs() < 1e-10);
        assert!((ranges[0].1 - 3.99).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_mean() {
        let traj = make_trajectory(2, 100, 3.0);
        let analyzer = TrajectoryAnalyzer::new(&traj);
        let rolling = analyzer.rolling_mean_prices(10);
        assert_eq!(rolling.len(), 91); // 100 - 10 + 1
        // First window mean should be mean of prices 3.00..3.09
        let expected = (0..10).map(|i| 3.0 + 0.01 * i as f64).sum::<f64>() / 10.0;
        assert!((rolling[0][0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_variance() {
        let traj = make_constant_trajectory(2, 100, 5.0);
        let analyzer = TrajectoryAnalyzer::new(&traj);
        let rolling_var = analyzer.rolling_variance_prices(10);
        // Constant → zero rolling variance
        for var in &rolling_var {
            assert!(var[0] < 1e-10);
        }
    }

    #[test]
    fn test_convergence_detection_constant() {
        let traj = make_constant_trajectory(2, 200, 3.0);
        let analyzer = TrajectoryAnalyzer::new(&traj);
        let conv = analyzer.detect_convergence(20, 0.01);
        assert!(conv.is_some());
        assert!(conv.unwrap() <= 40); // Should detect early
    }

    #[test]
    fn test_is_converged() {
        let traj = make_constant_trajectory(2, 100, 3.0);
        let analyzer = TrajectoryAnalyzer::new(&traj);
        assert!(analyzer.is_converged(50, 0.001));
    }

    #[test]
    fn test_is_not_converged() {
        let traj = make_trajectory(2, 100, 3.0);
        let analyzer = TrajectoryAnalyzer::new(&traj);
        assert!(!analyzer.is_converged(50, 0.001));
    }

    #[test]
    fn test_cycle_detection() {
        let mut builder = TrajectoryBuilder::new(2);
        // Create a cycle of period 3: prices 1, 2, 3, 1, 2, 3, ...
        for r in 0..60 {
            let p = (r % 3 + 1) as f64;
            builder.push_from_values(r as u64, vec![p, p], vec![1.0, 1.0], vec![1.0, 1.0]);
        }
        let traj = builder.build();
        let analyzer = TrajectoryAnalyzer::new(&traj);
        let cycle = analyzer.detect_cycle(0, 10, 0.01, 3);
        assert!(cycle.is_some());
        assert_eq!(cycle.unwrap().0, 3); // period 3
    }

    #[test]
    fn test_no_cycle() {
        let traj = make_trajectory(2, 100, 3.0);
        let analyzer = TrajectoryAnalyzer::new(&traj);
        let cycle = analyzer.detect_cycle(0, 10, 0.001, 3);
        assert!(cycle.is_none());
    }

    #[test]
    fn test_autocorrelation_constant() {
        let traj = make_constant_trajectory(2, 100, 3.0);
        let analyzer = TrajectoryAnalyzer::new(&traj);
        let ac = analyzer.autocorrelation(0, 1);
        // Constant series: autocorrelation = 1.0
        assert!((ac - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_acf() {
        let traj = make_trajectory(2, 200, 3.0);
        let analyzer = TrajectoryAnalyzer::new(&traj);
        let acf = analyzer.acf(0, 5);
        assert_eq!(acf.len(), 6); // lags 0..5
        assert!((acf[0] - 1.0).abs() < 1e-10); // lag 0 is always 1
    }

    #[test]
    fn test_adf_statistic_stationary() {
        // Mean-reverting process should have negative ADF stat
        let mut builder = TrajectoryBuilder::new(2);
        let mut rng = rand::thread_rng();
        let mut p = 5.0;
        for r in 0..500 {
            p = 0.5 * p + 2.5 + (rng.gen::<f64>() - 0.5) * 0.1;
            builder.push_from_values(r as u64, vec![p, p], vec![1.0, 1.0], vec![1.0, 1.0]);
        }
        let traj = builder.build();
        let analyzer = TrajectoryAnalyzer::new(&traj);
        let stat = analyzer.adf_statistic(0);
        assert!(stat < 0.0); // should be negative
    }

    #[test]
    fn test_split_segments() {
        let traj = make_trajectory(2, 100, 3.0);
        let analyzer = TrajectoryAnalyzer::new(&traj);
        let segments = analyzer.split_segments(25);
        assert_eq!(segments.len(), 4);
        assert_eq!(segments[0].len(), 25);
    }

    #[test]
    fn test_split_train_val_holdout() {
        let traj = make_trajectory(2, 100, 3.0);
        let analyzer = TrajectoryAnalyzer::new(&traj);
        let (train, val, holdout) = analyzer.split_train_val_holdout(0.6, 0.2);
        assert_eq!(train.len(), 60);
        assert_eq!(val.len(), 20);
        assert_eq!(holdout.len(), 20);
    }

    #[test]
    fn test_merge_trajectories() {
        let t1 = make_trajectory(2, 50, 3.0);
        let t2 = make_trajectory(2, 50, 4.0);
        let merged = merge_trajectories(&[t1, t2]).unwrap();
        assert_eq!(merged.len(), 100);
    }

    #[test]
    fn test_merge_mismatched_players_fails() {
        let t1 = make_trajectory(2, 50, 3.0);
        let t2 = make_trajectory(3, 50, 3.0);
        assert!(merge_trajectories(&[t1, t2]).is_err());
    }

    #[test]
    fn test_align_trajectories() {
        let t1 = make_trajectory(2, 100, 3.0);
        let t2 = make_trajectory(2, 50, 3.0);
        let (a, b) = align_trajectories(&t1, &t2);
        assert_eq!(a.len(), 50);
        assert_eq!(b.len(), 50);
    }

    #[test]
    fn test_csv_roundtrip() {
        let traj = make_trajectory(2, 10, 3.0);
        let csv = trajectory_to_csv(&traj);
        let parsed = trajectory_from_csv(&csv, 2).unwrap();
        assert_eq!(parsed.len(), 10);
        assert!((parsed.outcomes[0].prices[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_json_roundtrip() {
        let traj = make_trajectory(2, 5, 3.0);
        let json = trajectory_to_json(&traj).unwrap();
        let parsed = trajectory_from_json(&json).unwrap();
        assert_eq!(parsed.len(), 5);
        assert!((parsed.outcomes[0].prices[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_builder_snapshot() {
        let mut builder = TrajectoryBuilder::new(2);
        for r in 0..10 {
            builder.push_from_values(r, vec![3.0, 3.0], vec![7.0, 7.0], vec![14.0, 14.0]);
        }
        let snap = builder.snapshot();
        assert_eq!(snap.len(), 10);
        // Builder still usable
        builder.push_from_values(10, vec![3.0, 3.0], vec![7.0, 7.0], vec![14.0, 14.0]);
        assert_eq!(builder.len(), 11);
    }

    #[test]
    fn test_profit_variance() {
        let traj = make_constant_trajectory(2, 100, 3.0);
        let analyzer = TrajectoryAnalyzer::new(&traj);
        let var = analyzer.profit_variance();
        assert!(var[0] < 1e-10);
    }

    #[test]
    fn test_stationarity_check() {
        let traj = make_constant_trajectory(2, 200, 3.0);
        let analyzer = TrajectoryAnalyzer::new(&traj);
        // Constant series: ADF test may be inconclusive (zero variance)
        // Just check it doesn't crash
        let _ = analyzer.is_stationary(0);
    }
}
