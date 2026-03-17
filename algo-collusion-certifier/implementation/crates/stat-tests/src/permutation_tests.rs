//! Distribution-free permutation tests for collusion detection.
//!
//! Implements permutation-based hypothesis tests for punishment detection,
//! temporal dependence, and cross-firm response asymmetries.

use rand::prelude::*;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use shared_types::{CollusionError, CollusionResult, PValue, PermutationResult};

// ── Generic permutation test ────────────────────────────────────────────────

/// Generic permutation testing framework.
#[derive(Debug, Clone)]
pub struct PermutationTest {
    pub num_permutations: usize,
    pub seed: Option<u64>,
}

impl PermutationTest {
    pub fn new(num_permutations: usize) -> Self {
        Self { num_permutations, seed: None }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Run a two-sample permutation test.
    /// `statistic` computes the test statistic from (group1, group2).
    pub fn two_sample<F>(
        &self,
        group1: &[f64],
        group2: &[f64],
        statistic: F,
    ) -> CollusionResult<PermutationResult>
    where
        F: Fn(&[f64], &[f64]) -> f64,
    {
        if group1.is_empty() || group2.is_empty() {
            return Err(CollusionError::StatisticalTest(
                "Permutation test: empty group".into(),
            ));
        }
        let observed = statistic(group1, group2);
        let combined: Vec<f64> = group1.iter().chain(group2.iter()).copied().collect();
        let n1 = group1.len();

        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut null_dist = Vec::with_capacity(self.num_permutations);
        for _ in 0..self.num_permutations {
            let mut shuffled = combined.clone();
            shuffled.shuffle(&mut rng);
            let perm_stat = statistic(&shuffled[..n1], &shuffled[n1..]);
            null_dist.push(perm_stat);
        }

        Ok(PermutationResult::new(observed, null_dist))
    }

    /// One-sample permutation test (sign-flip).
    pub fn one_sample<F>(
        &self,
        data: &[f64],
        statistic: F,
    ) -> CollusionResult<PermutationResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        if data.is_empty() {
            return Err(CollusionError::StatisticalTest(
                "Permutation test: empty data".into(),
            ));
        }
        let observed = statistic(data);

        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut null_dist = Vec::with_capacity(self.num_permutations);
        for _ in 0..self.num_permutations {
            let flipped: Vec<f64> = data
                .iter()
                .map(|x| if rng.gen_bool(0.5) { *x } else { -x })
                .collect();
            null_dist.push(statistic(&flipped));
        }

        Ok(PermutationResult::new(observed, null_dist))
    }
}

/// Generate the permutation distribution for a statistic.
pub fn generate_permutations<F>(
    data: &[f64],
    n_perms: usize,
    split: usize,
    statistic: F,
    seed: Option<u64>,
) -> Vec<f64>
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    (0..n_perms)
        .map(|_| {
            let mut shuffled = data.to_vec();
            shuffled.shuffle(&mut rng);
            statistic(&shuffled[..split], &shuffled[split..])
        })
        .collect()
}

/// Exact p-value from a permutation distribution.
pub fn permutation_p_value(perm_distribution: &[f64], observed: f64) -> PValue {
    if perm_distribution.is_empty() {
        return PValue::new_unchecked(1.0);
    }
    let count = perm_distribution.iter().filter(|&&x| x >= observed).count();
    PValue::new_unchecked((count as f64 + 1.0) / (perm_distribution.len() as f64 + 1.0))
}

// ── Punishment permutation test ─────────────────────────────────────────────

/// Test if post-deviation payoffs are lower than on-path payoffs.
/// Permutes deviation/non-deviation labels to build null distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PunishmentPermutationTest {
    pub num_permutations: usize,
    pub seed: Option<u64>,
}

impl PunishmentPermutationTest {
    pub fn new(num_permutations: usize) -> Self {
        Self { num_permutations, seed: None }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Test whether post-deviation profits are significantly lower than on-path.
    /// `on_path_profits`: profits during non-deviation periods.
    /// `post_deviation_profits`: profits in the window after a deviation.
    pub fn test(
        &self,
        on_path_profits: &[f64],
        post_deviation_profits: &[f64],
    ) -> CollusionResult<PunishmentTestResult> {
        if on_path_profits.is_empty() || post_deviation_profits.is_empty() {
            return Err(CollusionError::StatisticalTest(
                "Punishment test: need both on-path and post-deviation data".into(),
            ));
        }

        let mean_on = mean(on_path_profits);
        let mean_post = mean(post_deviation_profits);
        let observed_diff = mean_on - mean_post; // Positive = punishment

        let pt = PermutationTest::new(self.num_permutations)
            .with_seed(self.seed.unwrap_or(42));

        let diff_stat = |g1: &[f64], g2: &[f64]| -> f64 {
            mean(g1) - mean(g2)
        };

        let result = pt.two_sample(on_path_profits, post_deviation_profits, diff_stat)?;

        let punishment_magnitude = if mean_on.abs() < 1e-15 {
            0.0
        } else {
            observed_diff / mean_on
        };

        Ok(PunishmentTestResult {
            on_path_mean: mean_on,
            post_deviation_mean: mean_post,
            punishment_magnitude,
            p_value: result.p_value,
            num_permutations: self.num_permutations,
            is_punishment_detected: result.p_value.is_significant(0.05),
        })
    }
}

/// Result of a punishment permutation test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PunishmentTestResult {
    pub on_path_mean: f64,
    pub post_deviation_mean: f64,
    pub punishment_magnitude: f64,
    pub p_value: PValue,
    pub num_permutations: usize,
    pub is_punishment_detected: bool,
}

// ── Price shuffle test ──────────────────────────────────────────────────────

/// Permute time indices within segments to test for temporal dependence
/// consistent with collusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceShuffleTest {
    pub num_permutations: usize,
    pub seed: Option<u64>,
}

impl PriceShuffleTest {
    pub fn new(num_permutations: usize) -> Self {
        Self { num_permutations, seed: None }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Test for temporal dependence in a price series by shuffling time indices.
    /// Uses first-order autocorrelation as the test statistic.
    pub fn test(&self, prices: &[f64]) -> CollusionResult<PermutationResult> {
        if prices.len() < 4 {
            return Err(CollusionError::StatisticalTest(
                "Price shuffle test: need ≥4 observations".into(),
            ));
        }

        let observed_ac = autocorrelation(prices, 1);

        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut count_extreme = 0usize;
        for _ in 0..self.num_permutations {
            let mut shuffled = prices.to_vec();
            shuffled.shuffle(&mut rng);
            let perm_ac = autocorrelation(&shuffled, 1);
            if perm_ac >= observed_ac {
                count_extreme += 1;
            }
        }

        let p = (count_extreme as f64 + 1.0) / (self.num_permutations as f64 + 1.0);
        let null_distribution = Vec::new();
        Ok(PermutationResult {
            observed_statistic: observed_ac,
            p_value: PValue::new_unchecked(p),
            num_permutations: self.num_permutations,
            null_distribution,
            num_exceedances: count_extreme,
        })
    }

    /// Test for temporal structure by comparing within-segment variance
    /// to cross-segment variance.
    pub fn test_segmented(
        &self,
        prices: &[f64],
        segment_size: usize,
    ) -> CollusionResult<PermutationResult> {
        if prices.is_empty() || segment_size == 0 {
            return Err(CollusionError::StatisticalTest(
                "Segmented test: invalid parameters".into(),
            ));
        }

        let segments: Vec<&[f64]> = prices.chunks(segment_size).collect();
        let observed = within_vs_between_variance(&segments);

        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut count_extreme = 0usize;
        for _ in 0..self.num_permutations {
            let mut shuffled = prices.to_vec();
            shuffled.shuffle(&mut rng);
            let perm_segments: Vec<&[f64]> = shuffled.chunks(segment_size).collect();
            let perm_stat = within_vs_between_variance(&perm_segments);
            if perm_stat >= observed {
                count_extreme += 1;
            }
        }

        let p = (count_extreme as f64 + 1.0) / (self.num_permutations as f64 + 1.0);
        Ok(PermutationResult {
            observed_statistic: observed,
            p_value: PValue::new_unchecked(p),
            num_permutations: self.num_permutations,
            null_distribution: Vec::new(),
            num_exceedances: count_extreme,
        })
    }
}

// ── Cross-firm permutation test ─────────────────────────────────────────────

/// Permute firm labels to test for asymmetric response patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossFirmPermutationTest {
    pub num_permutations: usize,
    pub seed: Option<u64>,
}

impl CrossFirmPermutationTest {
    pub fn new(num_permutations: usize) -> Self {
        Self { num_permutations, seed: None }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Test for asymmetric cross-firm price adjustment.
    /// `firm_prices`: Vec of price series, one per firm.
    pub fn test(
        &self,
        firm_prices: &[Vec<f64>],
    ) -> CollusionResult<PermutationResult> {
        if firm_prices.len() < 2 {
            return Err(CollusionError::StatisticalTest(
                "Cross-firm test: need ≥2 firms".into(),
            ));
        }
        let min_len = firm_prices.iter().map(|v| v.len()).min().unwrap_or(0);
        if min_len < 2 {
            return Err(CollusionError::StatisticalTest(
                "Cross-firm test: need ≥2 observations per firm".into(),
            ));
        }

        let observed = cross_firm_asymmetry(firm_prices, min_len);

        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let n_firms = firm_prices.len();
        let mut count_extreme = 0usize;

        for _ in 0..self.num_permutations {
            // Permute firm labels at each time step
            let mut permuted: Vec<Vec<f64>> = vec![Vec::with_capacity(min_len); n_firms];
            for t in 0..min_len {
                let mut firm_indices: Vec<usize> = (0..n_firms).collect();
                firm_indices.shuffle(&mut rng);
                for (new_idx, &orig_idx) in firm_indices.iter().enumerate() {
                    permuted[new_idx].push(firm_prices[orig_idx][t]);
                }
            }
            let perm_stat = cross_firm_asymmetry(&permuted, min_len);
            if perm_stat >= observed {
                count_extreme += 1;
            }
        }

        let p = (count_extreme as f64 + 1.0) / (self.num_permutations as f64 + 1.0);
        Ok(PermutationResult {
            observed_statistic: observed,
            p_value: PValue::new_unchecked(p),
            num_permutations: self.num_permutations,
            null_distribution: Vec::new(),
            num_exceedances: count_extreme,
        })
    }
}

// ── Monte Carlo permutation ─────────────────────────────────────────────────

/// Approximate permutation test using B random permutations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloPermutation {
    pub num_permutations: usize,
    pub seed: Option<u64>,
}

impl MonteCarloPermutation {
    pub fn new(num_permutations: usize) -> Self {
        Self { num_permutations, seed: None }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Run Monte Carlo permutation test with generic data and statistic.
    pub fn test<T, F>(
        &self,
        data: &[T],
        split: usize,
        statistic: F,
    ) -> CollusionResult<PermutationResult>
    where
        T: Clone,
        F: Fn(&[T], &[T]) -> f64,
    {
        if data.is_empty() || split == 0 || split >= data.len() {
            return Err(CollusionError::StatisticalTest(
                "Monte Carlo permutation: invalid split".into(),
            ));
        }

        let observed = statistic(&data[..split], &data[split..]);

        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut count_extreme = 0usize;
        for _ in 0..self.num_permutations {
            let mut shuffled = data.to_vec();
            shuffled.shuffle(&mut rng);
            let perm_stat = statistic(&shuffled[..split], &shuffled[split..]);
            if perm_stat >= observed {
                count_extreme += 1;
            }
        }

        let p = (count_extreme as f64 + 1.0) / (self.num_permutations as f64 + 1.0);
        Ok(PermutationResult {
            observed_statistic: observed,
            p_value: PValue::new_unchecked(p),
            num_permutations: self.num_permutations,
            null_distribution: Vec::new(),
            num_exceedances: count_extreme,
        })
    }

    /// Standard error of Monte Carlo p-value.
    pub fn mc_se(p: f64, b: usize) -> f64 {
        (p * (1.0 - p) / b as f64).sqrt()
    }
}

// ── Stratified permutation ──────────────────────────────────────────────────

/// Permute within strata to control for confounders.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratifiedPermutation {
    pub num_permutations: usize,
    pub seed: Option<u64>,
}

impl StratifiedPermutation {
    pub fn new(num_permutations: usize) -> Self {
        Self { num_permutations, seed: None }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Run stratified permutation test.
    /// `strata`: stratum label for each observation.
    /// `group`: group label (0 or 1) for each observation.
    /// `values`: the observed values.
    /// Uses the mean difference between groups as statistic.
    pub fn test(
        &self,
        values: &[f64],
        group: &[usize],
        strata: &[usize],
    ) -> CollusionResult<PermutationResult> {
        let n = values.len();
        if n == 0 || group.len() != n || strata.len() != n {
            return Err(CollusionError::StatisticalTest(
                "Stratified permutation: mismatched lengths".into(),
            ));
        }

        let observed = stratified_mean_diff(values, group, strata);

        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        // Get unique strata
        let mut unique_strata: Vec<usize> = strata.to_vec();
        unique_strata.sort_unstable();
        unique_strata.dedup();

        let mut count_extreme = 0usize;
        for _ in 0..self.num_permutations {
            // Permute group labels within each stratum
            let mut perm_group = group.to_vec();
            for &s in &unique_strata {
                let indices: Vec<usize> = (0..n).filter(|&i| strata[i] == s).collect();
                let mut stratum_groups: Vec<usize> = indices.iter().map(|&i| group[i]).collect();
                stratum_groups.shuffle(&mut rng);
                for (k, &idx) in indices.iter().enumerate() {
                    perm_group[idx] = stratum_groups[k];
                }
            }

            let perm_stat = stratified_mean_diff(values, &perm_group, strata);
            if perm_stat >= observed {
                count_extreme += 1;
            }
        }

        let p = (count_extreme as f64 + 1.0) / (self.num_permutations as f64 + 1.0);
        Ok(PermutationResult {
            observed_statistic: observed,
            p_value: PValue::new_unchecked(p),
            num_permutations: self.num_permutations,
            null_distribution: Vec::new(),
            num_exceedances: count_extreme,
        })
    }
}

// ── Helper functions ────────────────────────────────────────────────────────

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { return 0.0; }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn autocorrelation(xs: &[f64], lag: usize) -> f64 {
    let n = xs.len();
    if n <= lag + 1 { return 0.0; }
    let m = mean(xs);
    let var: f64 = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>();
    if var.abs() < 1e-15 { return 0.0; }
    let cov: f64 = (0..n - lag)
        .map(|i| (xs[i] - m) * (xs[i + lag] - m))
        .sum();
    cov / var
}

fn within_vs_between_variance(segments: &[&[f64]]) -> f64 {
    if segments.is_empty() { return 0.0; }
    let means: Vec<f64> = segments.iter().map(|s| mean(s)).collect();
    let grand_mean = mean(&means);

    let between: f64 = means.iter().map(|m| (m - grand_mean).powi(2)).sum::<f64>()
        / means.len() as f64;

    let within: f64 = segments
        .iter()
        .map(|s| {
            let m = mean(s);
            s.iter().map(|x| (x - m).powi(2)).sum::<f64>() / s.len().max(1) as f64
        })
        .sum::<f64>()
        / segments.len() as f64;

    if within.abs() < 1e-15 { 0.0 } else { between / within }
}

fn cross_firm_asymmetry(firm_prices: &[Vec<f64>], min_len: usize) -> f64 {
    // Measure asymmetry as variance of cross-firm price differences
    let n_firms = firm_prices.len();
    if n_firms < 2 || min_len < 2 { return 0.0; }

    let mut total_asymmetry = 0.0;
    let mut pair_count = 0;

    for i in 0..n_firms {
        for j in (i + 1)..n_firms {
            // Cross-correlation of price changes
            let changes_i: Vec<f64> = (1..min_len)
                .map(|t| firm_prices[i][t] - firm_prices[i][t - 1])
                .collect();
            let changes_j: Vec<f64> = (1..min_len)
                .map(|t| firm_prices[j][t] - firm_prices[j][t - 1])
                .collect();

            // Asymmetry = |corr(Δp_i, Δp_j) - corr(Δp_j, Δp_i_lagged)|
            let fwd = pearson_corr(&changes_i[..changes_i.len() - 1], &changes_j[1..]);
            let bwd = pearson_corr(&changes_j[..changes_j.len() - 1], &changes_i[1..]);
            total_asymmetry += (fwd - bwd).abs();
            pair_count += 1;
        }
    }

    if pair_count == 0 { 0.0 } else { total_asymmetry / pair_count as f64 }
}

fn pearson_corr(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 2 { return 0.0; }
    let mx = mean(&xs[..n]);
    let my = mean(&ys[..n]);
    let mut cov = 0.0;
    let mut sx2 = 0.0;
    let mut sy2 = 0.0;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        cov += dx * dy;
        sx2 += dx * dx;
        sy2 += dy * dy;
    }
    let denom = (sx2 * sy2).sqrt();
    if denom < 1e-15 { 0.0 } else { cov / denom }
}

fn stratified_mean_diff(values: &[f64], group: &[usize], strata: &[usize]) -> f64 {
    let n = values.len();
    let mut unique_strata: Vec<usize> = strata.to_vec();
    unique_strata.sort_unstable();
    unique_strata.dedup();

    let mut total_diff = 0.0;
    let mut weight_sum = 0.0;

    for &s in &unique_strata {
        let g0: Vec<f64> = (0..n)
            .filter(|&i| strata[i] == s && group[i] == 0)
            .map(|i| values[i])
            .collect();
        let g1: Vec<f64> = (0..n)
            .filter(|&i| strata[i] == s && group[i] == 1)
            .map(|i| values[i])
            .collect();
        if g0.is_empty() || g1.is_empty() {
            continue;
        }
        let w = (g0.len() + g1.len()) as f64;
        total_diff += w * (mean(&g1) - mean(&g0));
        weight_sum += w;
    }

    if weight_sum < 1e-15 { 0.0 } else { total_diff / weight_sum }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutation_test_two_sample_equal() {
        let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let pt = PermutationTest::new(999).with_seed(42);
        let result = pt
            .two_sample(&g1, &g2, |a, b| mean(a) - mean(b))
            .unwrap();
        assert!(result.p_value.value() > 0.3);
    }

    #[test]
    fn test_permutation_test_two_sample_different() {
        let g1 = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let g2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let pt = PermutationTest::new(999).with_seed(42);
        let result = pt
            .two_sample(&g1, &g2, |a, b| mean(a) - mean(b))
            .unwrap();
        assert!(result.p_value.value() < 0.05);
    }

    #[test]
    fn test_permutation_test_one_sample() {
        let data = vec![5.0, 6.0, 7.0, 8.0, 9.0];
        let pt = PermutationTest::new(999).with_seed(42);
        let result = pt.one_sample(&data, |d| mean(d)).unwrap();
        assert!(result.observed_statistic > 0.0);
    }

    #[test]
    fn test_permutation_test_empty_error() {
        let pt = PermutationTest::new(100);
        assert!(pt.two_sample(&[], &[1.0], |a, b| mean(a) - mean(b)).is_err());
        assert!(pt.one_sample(&[], |d| mean(d)).is_err());
    }

    #[test]
    fn test_generate_permutations() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let perms = generate_permutations(&data, 500, 3, |a, b| mean(a) - mean(b), Some(42));
        assert_eq!(perms.len(), 500);
    }

    #[test]
    fn test_permutation_p_value() {
        let dist = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let p = permutation_p_value(&dist, 0.85);
        assert!(p.value() > 0.0);
        assert!(p.value() < 0.5);
    }

    #[test]
    fn test_punishment_permutation_test() {
        let on_path = vec![10.0, 10.5, 11.0, 10.2, 10.8, 10.3, 10.7, 10.4];
        let post_dev = vec![5.0, 5.5, 6.0, 5.2, 5.8, 5.3, 5.7, 5.4];
        let ppt = PunishmentPermutationTest::new(999).with_seed(42);
        let result = ppt.test(&on_path, &post_dev).unwrap();
        assert!(result.is_punishment_detected);
        assert!(result.punishment_magnitude > 0.0);
    }

    #[test]
    fn test_punishment_no_difference() {
        let on_path = vec![5.0, 5.1, 4.9, 5.0, 5.2];
        let post_dev = vec![5.0, 4.9, 5.1, 5.0, 4.8];
        let ppt = PunishmentPermutationTest::new(999).with_seed(42);
        let result = ppt.test(&on_path, &post_dev).unwrap();
        assert!(!result.is_punishment_detected);
    }

    #[test]
    fn test_punishment_empty_error() {
        let ppt = PunishmentPermutationTest::new(100);
        assert!(ppt.test(&[], &[1.0]).is_err());
    }

    #[test]
    fn test_price_shuffle_test() {
        // Correlated series: prices that trend upward
        let prices: Vec<f64> = (0..50).map(|i| 2.0 + 0.02 * i as f64).collect();
        let pst = PriceShuffleTest::new(499).with_seed(42);
        let result = pst.test(&prices).unwrap();
        assert!(result.observed_statistic > 0.5); // High autocorrelation
    }

    #[test]
    fn test_price_shuffle_random() {
        let mut rng = StdRng::seed_from_u64(42);
        let prices: Vec<f64> = (0..50).map(|_| rng.gen_range(1.0..5.0)).collect();
        let pst = PriceShuffleTest::new(499).with_seed(42);
        let result = pst.test(&prices).unwrap();
        // Random series: p-value should not be significant
        assert!(result.p_value.value() > 0.01);
    }

    #[test]
    fn test_price_shuffle_segmented() {
        let prices: Vec<f64> = (0..100).map(|i| if i < 50 { 2.0 } else { 4.0 }).collect();
        let pst = PriceShuffleTest::new(499).with_seed(42);
        let result = pst.test_segmented(&prices, 25).unwrap();
        assert!(result.observed_statistic > 0.0);
    }

    #[test]
    fn test_cross_firm_permutation() {
        let firm1 = vec![2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9];
        let firm2 = vec![2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9];
        let cfpt = CrossFirmPermutationTest::new(499).with_seed(42);
        let result = cfpt.test(&[firm1, firm2]).unwrap();
        assert!(result.num_permutations == 499);
    }

    #[test]
    fn test_cross_firm_permutation_error() {
        let cfpt = CrossFirmPermutationTest::new(100);
        assert!(cfpt.test(&[vec![1.0]]).is_err());
    }

    #[test]
    fn test_monte_carlo_permutation() {
        let data: Vec<f64> = vec![10.0, 11.0, 12.0, 1.0, 2.0, 3.0];
        let mcp = MonteCarloPermutation::new(999).with_seed(42);
        let result = mcp
            .test(&data, 3, |a, b| {
                let ma: f64 = a.iter().sum::<f64>() / a.len() as f64;
                let mb: f64 = b.iter().sum::<f64>() / b.len() as f64;
                ma - mb
            })
            .unwrap();
        assert!(result.p_value.value() < 0.1);
    }

    #[test]
    fn test_monte_carlo_se() {
        let se = MonteCarloPermutation::mc_se(0.05, 1000);
        assert!(se > 0.0);
        assert!(se < 0.01);
    }

    #[test]
    fn test_stratified_permutation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0];
        let group = vec![0, 0, 1, 1, 0, 0, 1, 1];
        let strata = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let sp = StratifiedPermutation::new(499).with_seed(42);
        let result = sp.test(&values, &group, &strata).unwrap();
        assert!(result.observed_statistic.is_finite());
    }

    #[test]
    fn test_stratified_permutation_error() {
        let sp = StratifiedPermutation::new(100);
        assert!(sp.test(&[], &[], &[]).is_err());
    }

    #[test]
    fn test_autocorrelation_constant() {
        let xs = vec![5.0; 20];
        let ac = autocorrelation(&xs, 1);
        assert!((ac).abs() < 1e-10 || ac.is_nan());
    }

    #[test]
    fn test_autocorrelation_trend() {
        let xs: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let ac = autocorrelation(&xs, 1);
        assert!(ac > 0.9);
    }
}
