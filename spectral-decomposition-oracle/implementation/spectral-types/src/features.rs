//! Feature vector types and spectral feature definitions.
//!
//! Defines the 8 spectral features, 25 syntactic features, 10 graph features,
//! combined feature vectors, normalization, and selection.

use serde::{Deserialize, Serialize};
use crate::scalar::{NAN_SENTINEL, is_nan_sentinel, nan_to_sentinel, sentinel_to_nan};

/// The 8 spectral features derived from constraint hypergraph Laplacian analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralFeatures {
    pub spectral_gap: f64,
    pub spectral_gap_ratio: f64,
    pub eigenvalue_decay_rate: f64,
    pub fiedler_localization_entropy: f64,
    pub algebraic_connectivity_ratio: f64,
    pub coupling_energy: f64,
    pub block_separability_index: f64,
    pub effective_spectral_dimension: f64,
}

impl SpectralFeatures {
    pub fn nan() -> Self {
        Self {
            spectral_gap: f64::NAN,
            spectral_gap_ratio: f64::NAN,
            eigenvalue_decay_rate: f64::NAN,
            fiedler_localization_entropy: f64::NAN,
            algebraic_connectivity_ratio: f64::NAN,
            coupling_energy: f64::NAN,
            block_separability_index: f64::NAN,
            effective_spectral_dimension: f64::NAN,
        }
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.spectral_gap,
            self.spectral_gap_ratio,
            self.eigenvalue_decay_rate,
            self.fiedler_localization_entropy,
            self.algebraic_connectivity_ratio,
            self.coupling_energy,
            self.block_separability_index,
            self.effective_spectral_dimension,
        ]
    }

    pub fn from_vec(v: &[f64]) -> Option<Self> {
        if v.len() < 8 {
            return None;
        }
        Some(Self {
            spectral_gap: v[0],
            spectral_gap_ratio: v[1],
            eigenvalue_decay_rate: v[2],
            fiedler_localization_entropy: v[3],
            algebraic_connectivity_ratio: v[4],
            coupling_energy: v[5],
            block_separability_index: v[6],
            effective_spectral_dimension: v[7],
        })
    }

    pub fn names() -> Vec<&'static str> {
        vec![
            "spectral_gap",
            "spectral_gap_ratio",
            "eigenvalue_decay_rate",
            "fiedler_localization_entropy",
            "algebraic_connectivity_ratio",
            "coupling_energy",
            "block_separability_index",
            "effective_spectral_dimension",
        ]
    }

    pub fn count() -> usize {
        8
    }

    pub fn has_nan(&self) -> bool {
        self.to_vec().iter().any(|v| v.is_nan())
    }

    pub fn sanitize(&self) -> Self {
        Self::from_vec(&self.to_vec().iter().map(|&v| nan_to_sentinel(v)).collect::<Vec<_>>())
            .unwrap()
    }

    pub fn desanitize(&self) -> Self {
        Self::from_vec(
            &self
                .to_vec()
                .iter()
                .map(|&v| sentinel_to_nan(v))
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    /// Compute from eigenvalues (sorted ascending). The core computation.
    pub fn from_eigenvalues(eigenvalues: &[f64], n: usize) -> Self {
        if eigenvalues.is_empty() || n == 0 {
            return Self::nan();
        }

        let k = eigenvalues.len();

        // spectral_gap: lambda_2 - lambda_1 (Fiedler gap)
        let spectral_gap = if k >= 2 {
            eigenvalues[1] - eigenvalues[0]
        } else {
            0.0
        };

        // spectral_gap_ratio: (lambda_2 - lambda_1) / lambda_2
        let spectral_gap_ratio = if k >= 2 && eigenvalues[1].abs() > 1e-15 {
            (eigenvalues[1] - eigenvalues[0]) / eigenvalues[1]
        } else {
            0.0
        };

        // eigenvalue_decay_rate: exponential decay fit -slope of log(lambda_i) vs i
        let eigenvalue_decay_rate = {
            let positive: Vec<(f64, f64)> = eigenvalues
                .iter()
                .enumerate()
                .filter(|(_, &v)| v > 1e-15)
                .map(|(i, &v)| (i as f64, v.ln()))
                .collect();
            if positive.len() >= 2 {
                let n_p = positive.len() as f64;
                let sx: f64 = positive.iter().map(|&(x, _)| x).sum();
                let sy: f64 = positive.iter().map(|&(_, y)| y).sum();
                let sxy: f64 = positive.iter().map(|&(x, y)| x * y).sum();
                let sxx: f64 = positive.iter().map(|&(x, _)| x * x).sum();
                let denom = n_p * sxx - sx * sx;
                if denom.abs() > 1e-15 {
                    -(n_p * sxy - sx * sy) / denom
                } else {
                    0.0
                }
            } else {
                0.0
            }
        };

        // fiedler_localization_entropy: Shannon entropy of |fiedler_vector|^2
        // We approximate using eigenvalue distribution as proxy
        let fiedler_localization_entropy = {
            let total: f64 = eigenvalues.iter().filter(|&&v| v > 1e-15).sum();
            if total > 1e-15 {
                let probs: Vec<f64> = eigenvalues
                    .iter()
                    .filter(|&&v| v > 1e-15)
                    .map(|&v| v / total)
                    .collect();
                -probs
                    .iter()
                    .filter(|&&p| p > 1e-15)
                    .map(|&p| p * p.ln())
                    .sum::<f64>()
            } else {
                0.0
            }
        };

        // algebraic_connectivity_ratio: lambda_2 / lambda_n
        let algebraic_connectivity_ratio = if k >= 2 {
            let lambda_max = eigenvalues[k - 1];
            if lambda_max.abs() > 1e-15 {
                eigenvalues.get(1).copied().unwrap_or(0.0) / lambda_max
            } else {
                0.0
            }
        } else {
            0.0
        };

        // coupling_energy: sum of smallest 10% of nonzero eigenvalues / total
        let coupling_energy = {
            let nonzero: Vec<f64> = eigenvalues.iter().copied().filter(|&v| v > 1e-15).collect();
            if nonzero.is_empty() {
                0.0
            } else {
                let total: f64 = nonzero.iter().sum();
                let count = (nonzero.len() as f64 * 0.1).ceil() as usize;
                let small_sum: f64 = nonzero.iter().take(count.max(1)).sum();
                if total > 1e-15 {
                    small_sum / total
                } else {
                    0.0
                }
            }
        };

        // block_separability_index: fraction of eigenvalues below median
        let block_separability_index = {
            let nonzero: Vec<f64> = eigenvalues.iter().copied().filter(|&v| v > 1e-15).collect();
            if nonzero.is_empty() {
                0.0
            } else {
                let median = {
                    let mid = nonzero.len() / 2;
                    if nonzero.len() % 2 == 0 {
                        (nonzero[mid - 1] + nonzero[mid]) / 2.0
                    } else {
                        nonzero[mid]
                    }
                };
                let below = nonzero.iter().filter(|&&v| v < median * 0.5).count();
                below as f64 / nonzero.len() as f64
            }
        };

        // effective_spectral_dimension: inverse participation ratio of eigenvalues
        let effective_spectral_dimension = {
            let total: f64 = eigenvalues.iter().filter(|&&v| v > 1e-15).sum();
            if total > 1e-15 {
                let sum_sq: f64 = eigenvalues
                    .iter()
                    .filter(|&&v| v > 1e-15)
                    .map(|&v| (v / total).powi(2))
                    .sum();
                if sum_sq > 1e-15 {
                    1.0 / sum_sq
                } else {
                    n as f64
                }
            } else {
                0.0
            }
        };

        Self {
            spectral_gap,
            spectral_gap_ratio,
            eigenvalue_decay_rate,
            fiedler_localization_entropy,
            algebraic_connectivity_ratio,
            coupling_energy,
            block_separability_index,
            effective_spectral_dimension,
        }
    }
}

impl Default for SpectralFeatures {
    fn default() -> Self {
        Self::nan()
    }
}

/// Syntactic features (25) derived from the MIP formulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntacticFeatures {
    pub num_variables: f64,
    pub num_constraints: f64,
    pub num_nonzeros: f64,
    pub density: f64,
    pub constraint_matrix_rank_estimate: f64,
    pub avg_row_nnz: f64,
    pub max_row_nnz: f64,
    pub min_row_nnz: f64,
    pub std_row_nnz: f64,
    pub avg_col_nnz: f64,
    pub max_col_nnz: f64,
    pub min_col_nnz: f64,
    pub std_col_nnz: f64,
    pub coeff_range: f64,
    pub coeff_mean: f64,
    pub coeff_std: f64,
    pub rhs_range: f64,
    pub obj_range: f64,
    pub frac_binary: f64,
    pub frac_integer: f64,
    pub frac_continuous: f64,
    pub frac_equality: f64,
    pub frac_inequality: f64,
    pub variable_bound_tightness: f64,
    pub constraint_redundancy_estimate: f64,
}

impl SyntacticFeatures {
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.num_variables,
            self.num_constraints,
            self.num_nonzeros,
            self.density,
            self.constraint_matrix_rank_estimate,
            self.avg_row_nnz,
            self.max_row_nnz,
            self.min_row_nnz,
            self.std_row_nnz,
            self.avg_col_nnz,
            self.max_col_nnz,
            self.min_col_nnz,
            self.std_col_nnz,
            self.coeff_range,
            self.coeff_mean,
            self.coeff_std,
            self.rhs_range,
            self.obj_range,
            self.frac_binary,
            self.frac_integer,
            self.frac_continuous,
            self.frac_equality,
            self.frac_inequality,
            self.variable_bound_tightness,
            self.constraint_redundancy_estimate,
        ]
    }

    pub fn names() -> Vec<&'static str> {
        vec![
            "num_variables",
            "num_constraints",
            "num_nonzeros",
            "density",
            "constraint_matrix_rank_estimate",
            "avg_row_nnz",
            "max_row_nnz",
            "min_row_nnz",
            "std_row_nnz",
            "avg_col_nnz",
            "max_col_nnz",
            "min_col_nnz",
            "std_col_nnz",
            "coeff_range",
            "coeff_mean",
            "coeff_std",
            "rhs_range",
            "obj_range",
            "frac_binary",
            "frac_integer",
            "frac_continuous",
            "frac_equality",
            "frac_inequality",
            "variable_bound_tightness",
            "constraint_redundancy_estimate",
        ]
    }

    pub fn count() -> usize {
        25
    }

    pub fn from_vec(v: &[f64]) -> Option<Self> {
        if v.len() < 25 {
            return None;
        }
        Some(Self {
            num_variables: v[0],
            num_constraints: v[1],
            num_nonzeros: v[2],
            density: v[3],
            constraint_matrix_rank_estimate: v[4],
            avg_row_nnz: v[5],
            max_row_nnz: v[6],
            min_row_nnz: v[7],
            std_row_nnz: v[8],
            avg_col_nnz: v[9],
            max_col_nnz: v[10],
            min_col_nnz: v[11],
            std_col_nnz: v[12],
            coeff_range: v[13],
            coeff_mean: v[14],
            coeff_std: v[15],
            rhs_range: v[16],
            obj_range: v[17],
            frac_binary: v[18],
            frac_integer: v[19],
            frac_continuous: v[20],
            frac_equality: v[21],
            frac_inequality: v[22],
            variable_bound_tightness: v[23],
            constraint_redundancy_estimate: v[24],
        })
    }
}

/// Graph features (10) derived from the constraint graph structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphFeatures {
    pub num_vertices: f64,
    pub num_edges: f64,
    pub avg_degree: f64,
    pub max_degree: f64,
    pub degree_variance: f64,
    pub num_connected_components: f64,
    pub largest_component_fraction: f64,
    pub edge_density: f64,
    pub avg_clustering_coefficient: f64,
    pub degree_assortativity: f64,
}

impl GraphFeatures {
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.num_vertices,
            self.num_edges,
            self.avg_degree,
            self.max_degree,
            self.degree_variance,
            self.num_connected_components,
            self.largest_component_fraction,
            self.edge_density,
            self.avg_clustering_coefficient,
            self.degree_assortativity,
        ]
    }

    pub fn names() -> Vec<&'static str> {
        vec![
            "num_vertices",
            "num_edges",
            "avg_degree",
            "max_degree",
            "degree_variance",
            "num_connected_components",
            "largest_component_fraction",
            "edge_density",
            "avg_clustering_coefficient",
            "degree_assortativity",
        ]
    }

    pub fn count() -> usize {
        10
    }

    pub fn from_vec(v: &[f64]) -> Option<Self> {
        if v.len() < 10 {
            return None;
        }
        Some(Self {
            num_vertices: v[0],
            num_edges: v[1],
            avg_degree: v[2],
            max_degree: v[3],
            degree_variance: v[4],
            num_connected_components: v[5],
            largest_component_fraction: v[6],
            edge_density: v[7],
            avg_clustering_coefficient: v[8],
            degree_assortativity: v[9],
        })
    }
}

/// Combined feature vector from all sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedFeatureVector {
    pub spectral: SpectralFeatures,
    pub syntactic: SyntacticFeatures,
    pub graph: GraphFeatures,
}

impl CombinedFeatureVector {
    pub fn to_vec(&self) -> Vec<f64> {
        let mut v = self.spectral.to_vec();
        v.extend(self.syntactic.to_vec());
        v.extend(self.graph.to_vec());
        v
    }

    pub fn feature_count() -> usize {
        SpectralFeatures::count() + SyntacticFeatures::count() + GraphFeatures::count()
    }

    pub fn all_names() -> Vec<String> {
        let mut names: Vec<String> = SpectralFeatures::names()
            .into_iter()
            .map(|s| format!("spectral_{}", s))
            .collect();
        names.extend(
            SyntacticFeatures::names()
                .into_iter()
                .map(|s| format!("syntactic_{}", s)),
        );
        names.extend(
            GraphFeatures::names()
                .into_iter()
                .map(|s| format!("graph_{}", s)),
        );
        names
    }

    pub fn sanitize_nans(&self) -> Vec<f64> {
        self.to_vec().iter().map(|&v| nan_to_sentinel(v)).collect()
    }
}

/// Feature normalizer supporting z-score and min-max normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureNormalizer {
    pub method: NormalizationMethod,
    pub params: Vec<NormParam>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationMethod {
    ZScore,
    MinMax,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormParam {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

impl FeatureNormalizer {
    pub fn fit(data: &[Vec<f64>], method: NormalizationMethod) -> Self {
        if data.is_empty() {
            return Self {
                method,
                params: Vec::new(),
            };
        }
        let dim = data[0].len();
        let _n = data.len() as f64;

        let mut params = Vec::with_capacity(dim);
        for j in 0..dim {
            let values: Vec<f64> = data
                .iter()
                .map(|row| row.get(j).copied().unwrap_or(0.0))
                .filter(|v| !v.is_nan() && !is_nan_sentinel(*v))
                .collect();

            if values.is_empty() {
                params.push(NormParam {
                    mean: 0.0,
                    std: 1.0,
                    min: 0.0,
                    max: 1.0,
                });
                continue;
            }

            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let var = if values.len() > 1 {
                values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (values.len() as f64 - 1.0)
            } else {
                0.0
            };
            let std = var.sqrt().max(1e-15);
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            params.push(NormParam {
                mean,
                std,
                min,
                max,
            });
        }

        Self { method, params }
    }

    pub fn transform(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, &v)| {
                if v.is_nan() || is_nan_sentinel(v) {
                    return NAN_SENTINEL;
                }
                let p = self.params.get(i).cloned().unwrap_or(NormParam {
                    mean: 0.0,
                    std: 1.0,
                    min: 0.0,
                    max: 1.0,
                });
                match self.method {
                    NormalizationMethod::ZScore => (v - p.mean) / p.std,
                    NormalizationMethod::MinMax => {
                        let range = p.max - p.min;
                        if range.abs() < 1e-15 {
                            0.0
                        } else {
                            (v - p.min) / range
                        }
                    }
                }
            })
            .collect()
    }

    pub fn inverse_transform(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, &v)| {
                if is_nan_sentinel(v) {
                    return f64::NAN;
                }
                let p = self.params.get(i).cloned().unwrap_or(NormParam {
                    mean: 0.0,
                    std: 1.0,
                    min: 0.0,
                    max: 1.0,
                });
                match self.method {
                    NormalizationMethod::ZScore => v * p.std + p.mean,
                    NormalizationMethod::MinMax => v * (p.max - p.min) + p.min,
                }
            })
            .collect()
    }
}

/// Feature selector using mutual-information-based mRMR (minimum Redundancy Maximum Relevance).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSelector {
    pub selected_indices: Vec<usize>,
    pub selected_names: Vec<String>,
    pub scores: Vec<f64>,
}

impl FeatureSelector {
    /// Select top-k features using mRMR approximation.
    /// `data`: feature matrix (rows = samples), `labels`: class labels.
    pub fn mrmr(data: &[Vec<f64>], labels: &[usize], k: usize, feature_names: &[String]) -> Self {
        let n_samples = data.len().min(labels.len());
        if n_samples == 0 || data[0].is_empty() {
            return Self {
                selected_indices: Vec::new(),
                selected_names: Vec::new(),
                scores: Vec::new(),
            };
        }
        let n_features = data[0].len();
        let k = k.min(n_features);

        // Compute relevance: absolute correlation with (numeric) label
        let label_f: Vec<f64> = labels[..n_samples].iter().map(|&l| l as f64).collect();
        let mut relevance = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let col: Vec<f64> = data[..n_samples]
                .iter()
                .map(|row| row.get(j).copied().unwrap_or(0.0))
                .collect();
            relevance.push(crate::stats::pearson_correlation(&col, &label_f).abs());
        }

        let mut selected = Vec::with_capacity(k);
        let mut selected_set = vec![false; n_features];
        let mut scores = Vec::with_capacity(k);

        // Select first feature: highest relevance
        let first = relevance
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        selected.push(first);
        selected_set[first] = true;
        scores.push(relevance[first]);

        // Greedy mRMR
        for _ in 1..k {
            let mut best_idx = 0;
            let mut best_score = f64::NEG_INFINITY;

            for j in 0..n_features {
                if selected_set[j] {
                    continue;
                }
                let rel = relevance[j];

                // Compute average redundancy with selected features
                let mut redundancy = 0.0;
                let col_j: Vec<f64> = data[..n_samples]
                    .iter()
                    .map(|row| row.get(j).copied().unwrap_or(0.0))
                    .collect();
                for &s in &selected {
                    let col_s: Vec<f64> = data[..n_samples]
                        .iter()
                        .map(|row| row.get(s).copied().unwrap_or(0.0))
                        .collect();
                    redundancy += crate::stats::pearson_correlation(&col_j, &col_s).abs();
                }
                redundancy /= selected.len() as f64;

                let score = rel - redundancy;
                if score > best_score {
                    best_score = score;
                    best_idx = j;
                }
            }

            selected.push(best_idx);
            selected_set[best_idx] = true;
            scores.push(best_score);
        }

        let selected_names = selected
            .iter()
            .map(|&i| {
                feature_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("feature_{}", i))
            })
            .collect();

        Self {
            selected_indices: selected,
            selected_names,
            scores,
        }
    }

    /// Apply selection to a feature vector, keeping only selected indices.
    pub fn apply(&self, features: &[f64]) -> Vec<f64> {
        self.selected_indices
            .iter()
            .map(|&i| features.get(i).copied().unwrap_or(NAN_SENTINEL))
            .collect()
    }
}

/// Feature importance from a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    pub name: String,
    pub importance: f64,
    pub rank: usize,
}

/// Compute feature importances from a list of (name, score) pairs.
pub fn rank_importances(raw: &[(String, f64)]) -> Vec<FeatureImportance> {
    let mut indexed: Vec<(usize, &str, f64)> = raw
        .iter()
        .enumerate()
        .map(|(i, (name, score))| (i, name.as_str(), *score))
        .collect();
    indexed.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    indexed
        .into_iter()
        .enumerate()
        .map(|(rank, (_, name, importance))| FeatureImportance {
            name: name.to_string(),
            importance,
            rank: rank + 1,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_features_to_from_vec() {
        let sf = SpectralFeatures {
            spectral_gap: 0.1,
            spectral_gap_ratio: 0.2,
            eigenvalue_decay_rate: 0.3,
            fiedler_localization_entropy: 0.4,
            algebraic_connectivity_ratio: 0.5,
            coupling_energy: 0.6,
            block_separability_index: 0.7,
            effective_spectral_dimension: 0.8,
        };
        let v = sf.to_vec();
        assert_eq!(v.len(), 8);
        let sf2 = SpectralFeatures::from_vec(&v).unwrap();
        assert!((sf2.spectral_gap - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_features_nan() {
        let sf = SpectralFeatures::nan();
        assert!(sf.has_nan());
    }

    #[test]
    fn test_spectral_features_sanitize() {
        let sf = SpectralFeatures::nan();
        let sanitized = sf.sanitize();
        assert!(!sanitized.to_vec().iter().any(|v| v.is_nan()));
    }

    #[test]
    fn test_spectral_from_eigenvalues() {
        let eigs = vec![0.0, 0.5, 1.0, 2.0, 3.0, 5.0];
        let sf = SpectralFeatures::from_eigenvalues(&eigs, 6);
        assert!((sf.spectral_gap - 0.5).abs() < 1e-10);
        assert!(sf.effective_spectral_dimension > 0.0);
    }

    #[test]
    fn test_syntactic_features_count() {
        assert_eq!(SyntacticFeatures::count(), 25);
        assert_eq!(SyntacticFeatures::names().len(), 25);
    }

    #[test]
    fn test_graph_features_count() {
        assert_eq!(GraphFeatures::count(), 10);
        assert_eq!(GraphFeatures::names().len(), 10);
    }

    #[test]
    fn test_combined_feature_count() {
        assert_eq!(CombinedFeatureVector::feature_count(), 43);
    }

    #[test]
    fn test_feature_normalizer_zscore() {
        let data = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        let norm = FeatureNormalizer::fit(&data, NormalizationMethod::ZScore);
        let t = norm.transform(&[2.0, 20.0]);
        assert!(t[0].abs() < 1e-10);
        assert!(t[1].abs() < 1e-10);
    }

    #[test]
    fn test_feature_normalizer_minmax() {
        let data = vec![vec![1.0, 10.0], vec![3.0, 30.0]];
        let norm = FeatureNormalizer::fit(&data, NormalizationMethod::MinMax);
        let t = norm.transform(&[2.0, 20.0]);
        assert!((t[0] - 0.5).abs() < 1e-10);
        assert!((t[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_feature_normalizer_inverse() {
        let data = vec![vec![1.0], vec![2.0], vec![3.0]];
        let norm = FeatureNormalizer::fit(&data, NormalizationMethod::ZScore);
        let t = norm.transform(&[2.5]);
        let back = norm.inverse_transform(&t);
        assert!((back[0] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_feature_selector_mrmr() {
        let data = vec![
            vec![1.0, 0.0, 0.5],
            vec![2.0, 0.0, 1.0],
            vec![3.0, 0.0, 1.5],
            vec![4.0, 0.0, 2.0],
        ];
        let labels = vec![0, 0, 1, 1];
        let names: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let sel = FeatureSelector::mrmr(&data, &labels, 2, &names);
        assert_eq!(sel.selected_indices.len(), 2);
    }

    #[test]
    fn test_feature_selector_apply() {
        let sel = FeatureSelector {
            selected_indices: vec![0, 2],
            selected_names: vec!["a".into(), "c".into()],
            scores: vec![1.0, 0.5],
        };
        let result = sel.apply(&[10.0, 20.0, 30.0]);
        assert_eq!(result, vec![10.0, 30.0]);
    }

    #[test]
    fn test_rank_importances() {
        let raw = vec![
            ("a".to_string(), 0.3),
            ("b".to_string(), 0.7),
            ("c".to_string(), 0.1),
        ];
        let ranked = rank_importances(&raw);
        assert_eq!(ranked[0].name, "b");
        assert_eq!(ranked[0].rank, 1);
    }

    #[test]
    fn test_spectral_names() {
        let names = SpectralFeatures::names();
        assert_eq!(names.len(), 8);
        assert_eq!(names[0], "spectral_gap");
    }

    #[test]
    fn test_combined_all_names() {
        let names = CombinedFeatureVector::all_names();
        assert_eq!(names.len(), 43);
        assert!(names[0].starts_with("spectral_"));
    }
}
