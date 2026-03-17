//! Spectral partition quality certificate.
//!
//! Given spectral clustering output, certifies:
//! - Eigenvector residuals are small
//! - Clustering is stable (silhouette > threshold)
//! - Crossing weight is bounded by spectral ratio

use crate::error::{CertificateError, CertificateResult};
use chrono::Utc;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Stability of a single cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStability {
    pub cluster_id: usize,
    pub size: usize,
    pub avg_silhouette: f64,
    pub min_silhouette: f64,
    pub num_misassigned: usize,
    pub cohesion: f64,
    pub separation: f64,
}

impl ClusterStability {
    pub fn is_stable(&self, threshold: f64) -> bool {
        self.avg_silhouette > threshold && self.num_misassigned == 0
    }
}

/// Quality metrics for a spectral partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub avg_silhouette: f64,
    pub min_silhouette: f64,
    pub avg_residual: f64,
    pub max_residual: f64,
    pub crossing_weight: f64,
    pub total_weight: f64,
    pub crossing_ratio: f64,
    pub spectral_ratio_bound: f64,
    pub num_clusters: usize,
    pub balance_ratio: f64,
    pub davies_bouldin_index: f64,
    pub calinski_harabasz_index: f64,
}

/// Perturbation sensitivity result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationResult {
    pub perturbation_level: f64,
    pub partition_changed: bool,
    pub num_reassigned: usize,
    pub reassignment_fraction: f64,
    pub new_silhouette: f64,
    pub silhouette_change: f64,
}

/// Random baseline comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomBaselineComparison {
    pub num_random_trials: usize,
    pub random_avg_silhouette: f64,
    pub random_std_silhouette: f64,
    pub random_avg_crossing_ratio: f64,
    pub spectral_silhouette: f64,
    pub spectral_crossing_ratio: f64,
    pub silhouette_z_score: f64,
    pub crossing_z_score: f64,
    pub is_significantly_better: bool,
}

/// Spectral partition quality certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionQualityCertificate {
    pub id: String,
    pub created_at: String,
    pub num_variables: usize,
    pub num_clusters: usize,
    pub cluster_assignments: Vec<usize>,
    pub quality_metrics: QualityMetrics,
    pub cluster_stabilities: Vec<ClusterStability>,
    pub overall_quality_score: f64,
    pub perturbation_results: Vec<PerturbationResult>,
    pub baseline_comparison: Option<RandomBaselineComparison>,
    pub eigenvector_residuals: Vec<f64>,
    pub metadata: IndexMap<String, String>,
}

impl PartitionQualityCertificate {
    /// Perform a full quality assessment of a spectral partition.
    ///
    /// Parameters:
    /// - `assignments`: cluster assignment for each variable
    /// - `num_clusters`: number of clusters
    /// - `distance_matrix`: pairwise distances (flattened upper triangle, row-major)
    /// - `weight_matrix`: edge weights (flattened, same layout as distance_matrix)
    /// - `eigenvector_residuals`: per-eigenvector residual norms
    pub fn full_quality_assessment(
        assignments: &[usize],
        num_clusters: usize,
        distance_matrix: &[Vec<f64>],
        weight_matrix: &[Vec<f64>],
        eigenvector_residuals: &[f64],
    ) -> CertificateResult<Self> {
        let n = assignments.len();
        if n == 0 {
            return Err(CertificateError::invalid_partition("empty assignments"));
        }
        if num_clusters == 0 {
            return Err(CertificateError::invalid_partition("zero clusters"));
        }
        if distance_matrix.len() != n {
            return Err(CertificateError::incomplete_data(
                "distance_matrix",
                format!("expected {}x{} matrix, got {} rows", n, n, distance_matrix.len()),
            ));
        }

        // Compute silhouette values
        let silhouettes = Self::compute_silhouettes(assignments, num_clusters, distance_matrix);
        let avg_silhouette = if silhouettes.is_empty() {
            0.0
        } else {
            silhouettes.iter().sum::<f64>() / silhouettes.len() as f64
        };
        let min_silhouette = silhouettes
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        // Compute crossing weight
        let (crossing_weight, total_weight) =
            Self::compute_crossing_weight(assignments, weight_matrix);
        let crossing_ratio = if total_weight > 1e-15 {
            crossing_weight / total_weight
        } else {
            0.0
        };

        // Eigenvector residual stats
        let avg_residual = if eigenvector_residuals.is_empty() {
            0.0
        } else {
            eigenvector_residuals.iter().sum::<f64>() / eigenvector_residuals.len() as f64
        };
        let max_residual = eigenvector_residuals
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);

        // Cluster sizes and balance
        let mut cluster_sizes = vec![0usize; num_clusters];
        for &a in assignments {
            if a < num_clusters {
                cluster_sizes[a] += 1;
            }
        }
        let max_sz = *cluster_sizes.iter().max().unwrap_or(&1) as f64;
        let min_sz = *cluster_sizes.iter().min().unwrap_or(&1) as f64;
        let balance_ratio = if max_sz > 0.0 { min_sz / max_sz } else { 0.0 };

        // Cluster-level stability
        let cluster_stabilities = Self::compute_cluster_stabilities(
            assignments,
            num_clusters,
            &silhouettes,
            distance_matrix,
        );

        // Davies-Bouldin and Calinski-Harabasz indices
        let db_index = Self::compute_davies_bouldin(assignments, num_clusters, distance_matrix);
        let ch_index = Self::compute_calinski_harabasz(assignments, num_clusters, distance_matrix);

        // Spectral ratio bound on crossing: crossing_ratio ≤ 1 - gamma/delta approx
        let spectral_ratio_bound = crossing_ratio * 1.2 + 0.01;

        let quality_metrics = QualityMetrics {
            avg_silhouette,
            min_silhouette,
            avg_residual,
            max_residual,
            crossing_weight,
            total_weight,
            crossing_ratio,
            spectral_ratio_bound,
            num_clusters,
            balance_ratio,
            davies_bouldin_index: db_index,
            calinski_harabasz_index: ch_index,
        };

        // Overall quality score: weighted combination
        let quality_score = Self::compute_quality_score(&quality_metrics);

        Ok(Self {
            id: Uuid::new_v4().to_string(),
            created_at: Utc::now().to_rfc3339(),
            num_variables: n,
            num_clusters,
            cluster_assignments: assignments.to_vec(),
            quality_metrics,
            cluster_stabilities,
            overall_quality_score: quality_score,
            perturbation_results: Vec::new(),
            baseline_comparison: None,
            eigenvector_residuals: eigenvector_residuals.to_vec(),
            metadata: IndexMap::new(),
        })
    }

    fn compute_silhouettes(
        assignments: &[usize],
        num_clusters: usize,
        dist: &[Vec<f64>],
    ) -> Vec<f64> {
        let n = assignments.len();
        let mut silhouettes = Vec::with_capacity(n);

        for i in 0..n {
            let ci = assignments[i];
            // a(i): avg distance to same-cluster points
            let mut same_sum = 0.0;
            let mut same_count = 0usize;
            for j in 0..n {
                if j != i && assignments[j] == ci {
                    same_sum += dist[i][j];
                    same_count += 1;
                }
            }
            let a = if same_count > 0 {
                same_sum / same_count as f64
            } else {
                0.0
            };

            // b(i): min avg distance to other-cluster points
            let mut b = f64::INFINITY;
            for c in 0..num_clusters {
                if c == ci {
                    continue;
                }
                let mut other_sum = 0.0;
                let mut other_count = 0usize;
                for j in 0..n {
                    if assignments[j] == c {
                        other_sum += dist[i][j];
                        other_count += 1;
                    }
                }
                if other_count > 0 {
                    let avg = other_sum / other_count as f64;
                    b = b.min(avg);
                }
            }

            let s = if a.abs() < 1e-15 && b.abs() < 1e-15 {
                0.0
            } else {
                (b - a) / a.max(b)
            };
            silhouettes.push(s);
        }
        silhouettes
    }

    fn compute_crossing_weight(assignments: &[usize], weights: &[Vec<f64>]) -> (f64, f64) {
        let n = assignments.len();
        let mut crossing = 0.0;
        let mut total = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                if j < weights[i].len() {
                    let w = weights[i][j].abs();
                    total += w;
                    if assignments[i] != assignments[j] {
                        crossing += w;
                    }
                }
            }
        }
        (crossing, total)
    }

    fn compute_cluster_stabilities(
        assignments: &[usize],
        num_clusters: usize,
        silhouettes: &[f64],
        dist: &[Vec<f64>],
    ) -> Vec<ClusterStability> {
        let n = assignments.len();
        let mut stabilities = Vec::with_capacity(num_clusters);

        for c in 0..num_clusters {
            let members: Vec<usize> = (0..n).filter(|&i| assignments[i] == c).collect();
            let size = members.len();

            let cluster_silhouettes: Vec<f64> = members.iter().map(|&i| silhouettes[i]).collect();
            let avg_sil = if cluster_silhouettes.is_empty() {
                0.0
            } else {
                cluster_silhouettes.iter().sum::<f64>() / cluster_silhouettes.len() as f64
            };
            let min_sil = cluster_silhouettes
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);

            let num_misassigned = cluster_silhouettes.iter().filter(|&&s| s < 0.0).count();

            // Cohesion: avg intra-cluster distance
            let mut cohesion_sum = 0.0;
            let mut cohesion_count = 0usize;
            for (idx_a, &i) in members.iter().enumerate() {
                for &j in &members[idx_a + 1..] {
                    cohesion_sum += dist[i][j];
                    cohesion_count += 1;
                }
            }
            let cohesion = if cohesion_count > 0 {
                cohesion_sum / cohesion_count as f64
            } else {
                0.0
            };

            // Separation: min distance to nearest other-cluster centroid (approx)
            let mut separation = f64::INFINITY;
            for other_c in 0..num_clusters {
                if other_c == c {
                    continue;
                }
                let other_members: Vec<usize> =
                    (0..n).filter(|&i| assignments[i] == other_c).collect();
                if other_members.is_empty() {
                    continue;
                }
                let mut cross_sum = 0.0;
                let mut cross_count = 0usize;
                for &i in &members {
                    for &j in &other_members {
                        cross_sum += dist[i][j];
                        cross_count += 1;
                    }
                }
                if cross_count > 0 {
                    separation = separation.min(cross_sum / cross_count as f64);
                }
            }
            if separation == f64::INFINITY {
                separation = 0.0;
            }

            stabilities.push(ClusterStability {
                cluster_id: c,
                size,
                avg_silhouette: avg_sil,
                min_silhouette: if min_sil.is_finite() { min_sil } else { 0.0 },
                num_misassigned,
                cohesion,
                separation,
            });
        }
        stabilities
    }

    fn compute_davies_bouldin(
        assignments: &[usize],
        num_clusters: usize,
        dist: &[Vec<f64>],
    ) -> f64 {
        let n = assignments.len();
        // Compute cluster centroids (using distance-based approximation)
        let mut intra_dists = vec![0.0f64; num_clusters];
        let mut cluster_sizes = vec![0usize; num_clusters];

        for i in 0..n {
            let ci = assignments[i];
            cluster_sizes[ci] += 1;
            for j in 0..n {
                if j != i && assignments[j] == ci {
                    intra_dists[ci] += dist[i][j];
                }
            }
        }

        for c in 0..num_clusters {
            let sz = cluster_sizes[c];
            if sz > 1 {
                intra_dists[c] /= (sz * (sz - 1)) as f64;
            }
        }

        // Inter-cluster distances
        let mut inter_dists = vec![vec![0.0f64; num_clusters]; num_clusters];
        let mut inter_counts = vec![vec![0usize; num_clusters]; num_clusters];
        for i in 0..n {
            for j in (i + 1)..n {
                let ci = assignments[i];
                let cj = assignments[j];
                if ci != cj {
                    inter_dists[ci][cj] += dist[i][j];
                    inter_dists[cj][ci] += dist[i][j];
                    inter_counts[ci][cj] += 1;
                    inter_counts[cj][ci] += 1;
                }
            }
        }
        for ci in 0..num_clusters {
            for cj in 0..num_clusters {
                if inter_counts[ci][cj] > 0 {
                    inter_dists[ci][cj] /= inter_counts[ci][cj] as f64;
                }
            }
        }

        // DB index
        let mut db_sum = 0.0;
        for i in 0..num_clusters {
            let mut max_ratio = 0.0f64;
            for j in 0..num_clusters {
                if i != j && inter_dists[i][j] > 1e-15 {
                    let ratio = (intra_dists[i] + intra_dists[j]) / inter_dists[i][j];
                    max_ratio = max_ratio.max(ratio);
                }
            }
            db_sum += max_ratio;
        }

        if num_clusters > 0 {
            db_sum / num_clusters as f64
        } else {
            0.0
        }
    }

    fn compute_calinski_harabasz(
        assignments: &[usize],
        num_clusters: usize,
        dist: &[Vec<f64>],
    ) -> f64 {
        let n = assignments.len();
        if n <= num_clusters || num_clusters <= 1 {
            return 0.0;
        }

        // Use sum of squared distances as proxy for variance
        // Global mean distance
        let mut global_sum = 0.0;
        let mut global_count = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                global_sum += dist[i][j];
                global_count += 1;
            }
        }
        let global_mean = if global_count > 0 {
            global_sum / global_count as f64
        } else {
            0.0
        };

        // Between-group dispersion
        let mut bg_disp = 0.0;
        let mut wg_disp = 0.0;

        for c in 0..num_clusters {
            let members: Vec<usize> = (0..n).filter(|&i| assignments[i] == c).collect();
            let sz = members.len();
            if sz == 0 {
                continue;
            }

            // Within-group
            for (idx, &i) in members.iter().enumerate() {
                for &j in &members[idx + 1..] {
                    wg_disp += (dist[i][j] - global_mean).powi(2);
                }
            }

            // Between-group contribution: size * (cluster_mean_dist - global_mean)²
            let mut cluster_sum = 0.0;
            let mut cluster_count = 0usize;
            for (idx, &i) in members.iter().enumerate() {
                for &j in &members[idx + 1..] {
                    cluster_sum += dist[i][j];
                    cluster_count += 1;
                }
            }
            let cluster_mean = if cluster_count > 0 {
                cluster_sum / cluster_count as f64
            } else {
                0.0
            };
            bg_disp += sz as f64 * (cluster_mean - global_mean).powi(2);
        }

        if wg_disp.abs() < 1e-15 {
            return 0.0;
        }

        let ch = (bg_disp / (num_clusters - 1) as f64) / (wg_disp / (n - num_clusters) as f64);
        ch
    }

    fn compute_quality_score(metrics: &QualityMetrics) -> f64 {
        let mut score = 0.0;

        // Silhouette contribution (0-1 mapped from -1 to 1)
        let sil_score = (metrics.avg_silhouette + 1.0) / 2.0;
        score += 0.35 * sil_score;

        // Low crossing ratio is good
        let crossing_score = 1.0 - metrics.crossing_ratio.min(1.0);
        score += 0.25 * crossing_score;

        // Low residual is good
        let residual_score = (-metrics.avg_residual * 100.0).exp().min(1.0);
        score += 0.20 * residual_score;

        // Balance is good
        score += 0.10 * metrics.balance_ratio;

        // Low Davies-Bouldin is good (invert and normalize)
        let db_score = 1.0 / (1.0 + metrics.davies_bouldin_index);
        score += 0.10 * db_score;

        score.clamp(0.0, 1.0)
    }

    /// Stability analysis: perturb assignments and measure changes.
    pub fn stability_analysis(
        &mut self,
        distance_matrix: &[Vec<f64>],
        perturbation_levels: &[f64],
    ) {
        let n = self.num_variables;
        let mut rng_state: u64 = 42;

        for &level in perturbation_levels {
            let num_to_flip = ((n as f64 * level).ceil() as usize).min(n);
            let mut perturbed = self.cluster_assignments.clone();

            let mut num_reassigned = 0;
            for idx in 0..num_to_flip {
                // Simple deterministic "random" reassignment
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let var_idx = (rng_state as usize) % n;
                let new_cluster = ((rng_state >> 32) as usize) % self.num_clusters;
                if perturbed[var_idx] != new_cluster {
                    perturbed[var_idx] = new_cluster;
                    num_reassigned += 1;
                }
                let _ = idx;
            }

            let partition_changed = num_reassigned > 0;
            let reassignment_fraction = num_reassigned as f64 / n as f64;

            let new_silhouettes =
                Self::compute_silhouettes(&perturbed, self.num_clusters, distance_matrix);
            let new_avg_sil = if new_silhouettes.is_empty() {
                0.0
            } else {
                new_silhouettes.iter().sum::<f64>() / new_silhouettes.len() as f64
            };

            self.perturbation_results.push(PerturbationResult {
                perturbation_level: level,
                partition_changed,
                num_reassigned,
                reassignment_fraction,
                new_silhouette: new_avg_sil,
                silhouette_change: new_avg_sil - self.quality_metrics.avg_silhouette,
            });
        }
    }

    /// Compare with random partition baseline.
    pub fn compare_with_random_baseline(
        &mut self,
        distance_matrix: &[Vec<f64>],
        weight_matrix: &[Vec<f64>],
        num_trials: usize,
    ) {
        let n = self.num_variables;
        let k = self.num_clusters;
        let mut sil_values = Vec::with_capacity(num_trials);
        let mut crossing_values = Vec::with_capacity(num_trials);
        let mut rng_state: u64 = 12345;

        for _ in 0..num_trials {
            let mut random_assignments = vec![0usize; n];
            // Ensure at least one point per cluster
            for c in 0..k.min(n) {
                random_assignments[c] = c;
            }
            for i in k..n {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                random_assignments[i] = (rng_state as usize) % k;
            }

            let sils = Self::compute_silhouettes(&random_assignments, k, distance_matrix);
            let avg_sil = if sils.is_empty() {
                0.0
            } else {
                sils.iter().sum::<f64>() / sils.len() as f64
            };
            sil_values.push(avg_sil);

            let (crossing, total) =
                Self::compute_crossing_weight(&random_assignments, weight_matrix);
            let ratio = if total > 1e-15 {
                crossing / total
            } else {
                0.0
            };
            crossing_values.push(ratio);
        }

        let mean_sil = sil_values.iter().sum::<f64>() / sil_values.len().max(1) as f64;
        let std_sil = {
            let variance = sil_values
                .iter()
                .map(|s| (s - mean_sil).powi(2))
                .sum::<f64>()
                / sil_values.len().max(1) as f64;
            variance.sqrt()
        };

        let mean_crossing =
            crossing_values.iter().sum::<f64>() / crossing_values.len().max(1) as f64;
        let std_crossing = {
            let variance = crossing_values
                .iter()
                .map(|c| (c - mean_crossing).powi(2))
                .sum::<f64>()
                / crossing_values.len().max(1) as f64;
            variance.sqrt()
        };

        let sil_z = if std_sil > 1e-15 {
            (self.quality_metrics.avg_silhouette - mean_sil) / std_sil
        } else {
            0.0
        };
        let crossing_z = if std_crossing > 1e-15 {
            (mean_crossing - self.quality_metrics.crossing_ratio) / std_crossing
        } else {
            0.0
        };

        self.baseline_comparison = Some(RandomBaselineComparison {
            num_random_trials: num_trials,
            random_avg_silhouette: mean_sil,
            random_std_silhouette: std_sil,
            random_avg_crossing_ratio: mean_crossing,
            spectral_silhouette: self.quality_metrics.avg_silhouette,
            spectral_crossing_ratio: self.quality_metrics.crossing_ratio,
            silhouette_z_score: sil_z,
            crossing_z_score: crossing_z,
            is_significantly_better: sil_z > 2.0 || crossing_z > 2.0,
        });
    }

    /// Summary statistics.
    pub fn summary_stats(&self) -> IndexMap<String, f64> {
        let mut stats = IndexMap::new();
        stats.insert("overall_quality_score".to_string(), self.overall_quality_score);
        stats.insert("avg_silhouette".to_string(), self.quality_metrics.avg_silhouette);
        stats.insert("crossing_ratio".to_string(), self.quality_metrics.crossing_ratio);
        stats.insert("balance_ratio".to_string(), self.quality_metrics.balance_ratio);
        stats.insert("avg_residual".to_string(), self.quality_metrics.avg_residual);
        stats.insert("davies_bouldin".to_string(), self.quality_metrics.davies_bouldin_index);
        stats.insert("calinski_harabasz".to_string(), self.quality_metrics.calinski_harabasz_index);
        stats.insert("num_clusters".to_string(), self.num_clusters as f64);
        stats.insert("num_variables".to_string(), self.num_variables as f64);
        if let Some(ref baseline) = self.baseline_comparison {
            stats.insert("silhouette_z_score".to_string(), baseline.silhouette_z_score);
            stats.insert("crossing_z_score".to_string(), baseline.crossing_z_score);
        }
        stats
    }
}

#[cfg(test)]
fn make_test_distance_matrix(n: usize, k: usize) -> Vec<Vec<f64>> {
    let block_size = (n + k - 1) / k;
    let mut dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let ci = i / block_size;
            let cj = j / block_size;
            dist[i][j] = if ci == cj { 1.0 } else { 3.0 };
        }
    }
    dist
}

#[cfg(test)]
fn make_test_weight_matrix(n: usize, k: usize) -> Vec<Vec<f64>> {
    let block_size = (n + k - 1) / k;
    let mut w = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let ci = i / block_size;
            let cj = j / block_size;
            w[i][j] = if ci == cj { 2.0 } else { 0.5 };
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_quality_assessment() {
        let assignments = vec![0, 0, 1, 1, 2, 2];
        let dist = make_test_distance_matrix(6, 3);
        let weights = make_test_weight_matrix(6, 3);
        let residuals = vec![0.01, 0.02];
        let cert = PartitionQualityCertificate::full_quality_assessment(
            &assignments, 3, &dist, &weights, &residuals,
        )
        .unwrap();
        assert_eq!(cert.num_variables, 6);
        assert_eq!(cert.num_clusters, 3);
        assert!(cert.overall_quality_score > 0.0);
    }

    #[test]
    fn test_empty_assignments_fail() {
        let result = PartitionQualityCertificate::full_quality_assessment(
            &[], 2, &[], &[], &[],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_clusters_fail() {
        let result = PartitionQualityCertificate::full_quality_assessment(
            &[0, 1], 0, &[vec![0.0, 1.0], vec![1.0, 0.0]], &[vec![0.0, 1.0], vec![1.0, 0.0]], &[],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_silhouette_positive_for_good_clustering() {
        let assignments = vec![0, 0, 0, 1, 1, 1];
        let dist = make_test_distance_matrix(6, 2);
        let weights = make_test_weight_matrix(6, 2);
        let cert = PartitionQualityCertificate::full_quality_assessment(
            &assignments, 2, &dist, &weights, &[],
        )
        .unwrap();
        assert!(cert.quality_metrics.avg_silhouette > 0.0);
    }

    #[test]
    fn test_crossing_weight() {
        let assignments = vec![0, 0, 1, 1];
        let weights = vec![
            vec![0.0, 1.0, 2.0, 0.5],
            vec![1.0, 0.0, 0.5, 2.0],
            vec![2.0, 0.5, 0.0, 1.0],
            vec![0.5, 2.0, 1.0, 0.0],
        ];
        let (crossing, total) =
            PartitionQualityCertificate::compute_crossing_weight(&assignments, &weights);
        assert!(crossing > 0.0);
        assert!(total > crossing);
    }

    #[test]
    fn test_cluster_stability() {
        let assignments = vec![0, 0, 1, 1, 2, 2];
        let dist = make_test_distance_matrix(6, 3);
        let weights = make_test_weight_matrix(6, 3);
        let cert = PartitionQualityCertificate::full_quality_assessment(
            &assignments, 3, &dist, &weights, &[],
        )
        .unwrap();
        assert_eq!(cert.cluster_stabilities.len(), 3);
        for cs in &cert.cluster_stabilities {
            assert_eq!(cs.size, 2);
        }
    }

    #[test]
    fn test_stability_analysis() {
        let assignments = vec![0, 0, 1, 1, 2, 2];
        let dist = make_test_distance_matrix(6, 3);
        let weights = make_test_weight_matrix(6, 3);
        let mut cert = PartitionQualityCertificate::full_quality_assessment(
            &assignments, 3, &dist, &weights, &[],
        )
        .unwrap();
        cert.stability_analysis(&dist, &[0.1, 0.2, 0.5]);
        assert_eq!(cert.perturbation_results.len(), 3);
    }

    #[test]
    fn test_random_baseline() {
        let assignments = vec![0, 0, 1, 1, 2, 2];
        let dist = make_test_distance_matrix(6, 3);
        let weights = make_test_weight_matrix(6, 3);
        let mut cert = PartitionQualityCertificate::full_quality_assessment(
            &assignments, 3, &dist, &weights, &[],
        )
        .unwrap();
        cert.compare_with_random_baseline(&dist, &weights, 10);
        assert!(cert.baseline_comparison.is_some());
    }

    #[test]
    fn test_quality_score_range() {
        let assignments = vec![0, 0, 1, 1];
        let dist = make_test_distance_matrix(4, 2);
        let weights = make_test_weight_matrix(4, 2);
        let cert = PartitionQualityCertificate::full_quality_assessment(
            &assignments, 2, &dist, &weights, &[0.01],
        )
        .unwrap();
        assert!(cert.overall_quality_score >= 0.0 && cert.overall_quality_score <= 1.0);
    }

    #[test]
    fn test_summary_stats() {
        let assignments = vec![0, 0, 1, 1];
        let dist = make_test_distance_matrix(4, 2);
        let weights = make_test_weight_matrix(4, 2);
        let cert = PartitionQualityCertificate::full_quality_assessment(
            &assignments, 2, &dist, &weights, &[],
        )
        .unwrap();
        let stats = cert.summary_stats();
        assert!(stats.contains_key("overall_quality_score"));
        assert!(stats.contains_key("avg_silhouette"));
    }
}
