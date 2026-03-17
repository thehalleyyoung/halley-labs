//! Bilevel intersection cut generation.
//!
//! Given an LP relaxation vertex (x_hat, y_hat) in the bilevel-infeasible set B_bar,
//! compute valid cuts. Steps: (1) identify critical region, (2) trace rays along
//! simplex directions to bilevel-feasible boundary, (3) compute ray-boundary
//! intersection lengths, (4) apply Balas formula.

use crate::balas::{BalasCoefficients, BalasConfig, BalasFormula, RayLength};
use crate::ray_tracing::{
    build_simplex_directions, RayResult, RayTracer, RayTracerConfig, SimplexRayDirection,
};
use crate::{BilevelCut, CutError, CutResult, MIN_EFFICACY, TOLERANCE};
use bicut_types::*;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for intersection cut generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntersectionCutConfig {
    /// Minimum efficacy for a generated cut.
    pub min_efficacy: f64,
    /// Maximum number of cuts to generate per call.
    pub max_cuts: usize,
    /// Whether to attempt cut strengthening after generation.
    pub strengthen: bool,
    /// Whether to normalize cuts.
    pub normalize: bool,
    /// Tolerance for duplicate detection.
    pub duplicate_tolerance: f64,
    /// Ray tracer configuration.
    pub ray_config: RayTracerConfig,
    /// Balas formula configuration.
    pub balas_config: BalasConfig,
    /// Whether to use deep cuts (trace further into feasible region).
    pub deep_cuts: bool,
    /// Maximum rank of generated cuts.
    pub max_rank: u32,
}

impl Default for IntersectionCutConfig {
    fn default() -> Self {
        Self {
            min_efficacy: MIN_EFFICACY,
            max_cuts: 50,
            strengthen: true,
            normalize: true,
            duplicate_tolerance: 1e-6,
            ray_config: RayTracerConfig::default(),
            balas_config: BalasConfig::default(),
            deep_cuts: false,
            max_rank: 3,
        }
    }
}

/// A candidate cut before final filtering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutCandidate {
    pub cut: BilevelCut,
    pub efficacy: f64,
    pub num_finite_rays: usize,
    pub num_infinite_rays: usize,
    pub condition_estimate: f64,
    pub source_vertex: Vec<f64>,
}

/// A fully generated and validated cut.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedCut {
    pub cut: BilevelCut,
    pub efficacy: f64,
    pub rank: u32,
    pub ray_results: Vec<RayLengthSummary>,
    pub strengthened: bool,
}

/// Summary of ray length data for diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RayLengthSummary {
    pub variable_index: usize,
    pub alpha: f64,
    pub intersects: bool,
}

/// The main intersection cut generator.
#[derive(Debug)]
pub struct IntersectionCutGenerator {
    pub config: IntersectionCutConfig,
    balas: BalasFormula,
    ray_tracer: RayTracer,
    cut_counter: u64,
    generated_hashes: HashMap<Vec<OrderedFloat<f64>>, u64>,
    n_leader: usize,
    n_follower: usize,
    follower_obj: Vec<f64>,
    stats: IntersectionStats,
}

/// Statistics for intersection cut generation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IntersectionStats {
    pub total_attempts: usize,
    pub total_generated: usize,
    pub total_duplicates: usize,
    pub total_low_efficacy: usize,
    pub total_numerical_failures: usize,
    pub avg_efficacy: f64,
    pub max_efficacy: f64,
    pub avg_finite_rays: f64,
}

impl IntersectionCutGenerator {
    pub fn new(
        config: IntersectionCutConfig,
        n_leader: usize,
        n_follower: usize,
        follower_obj: Vec<f64>,
    ) -> Self {
        let balas = BalasFormula::new(config.balas_config.clone());
        let ray_tracer = RayTracer::new(
            config.ray_config.clone(),
            n_leader,
            n_follower,
            follower_obj.clone(),
        );
        Self {
            config,
            balas,
            ray_tracer,
            cut_counter: 0,
            generated_hashes: HashMap::new(),
            n_leader,
            n_follower,
            follower_obj,
            stats: IntersectionStats::default(),
        }
    }

    /// Generate intersection cuts from a bilevel-infeasible LP vertex.
    ///
    /// The vertex must satisfy c^T y > phi(x) (bilevel-infeasible).
    /// We trace rays along simplex directions, compute intersection lengths,
    /// and apply the Balas formula.
    pub fn generate(
        &mut self,
        vertex: &[f64],
        basis_status: &[BasisStatus],
        tableau_rows: &[(usize, Vec<f64>)],
        phi_evaluator: &dyn Fn(&[f64]) -> Option<f64>,
        lower_bounds: &[f64],
        upper_bounds: &[f64],
    ) -> CutResult<Vec<GeneratedCut>> {
        self.stats.total_attempts += 1;

        // Verify vertex is bilevel-infeasible.
        let n = vertex.len();
        let x = &vertex[..self.n_leader.min(n)];
        let phi_x = phi_evaluator(x)
            .ok_or_else(|| CutError::ValueFunctionError("Cannot evaluate phi at vertex".into()))?;
        let cy: f64 = self
            .follower_obj
            .iter()
            .enumerate()
            .map(|(i, &c)| c * vertex.get(self.n_leader + i).copied().unwrap_or(0.0))
            .sum();
        let gap = cy - phi_x;

        if gap <= self.config.min_efficacy {
            return Err(CutError::AlreadyFeasible);
        }

        // Build simplex ray directions from basis.
        let ray_directions = build_simplex_directions(basis_status, tableau_rows);
        if ray_directions.is_empty() {
            return Err(CutError::DegenerateBasis(
                "No nonbasic variables found".into(),
            ));
        }

        // Trace rays along each simplex direction.
        let ray_results = self.ray_tracer.trace_all_rays(
            vertex,
            &ray_directions,
            phi_evaluator,
            lower_bounds,
            upper_bounds,
        )?;

        // Convert to ray lengths for Balas formula.
        let ray_lengths: Vec<RayLength> = ray_results
            .iter()
            .map(|r| {
                let mut rl = r.to_ray_length();
                // Determine bound status from the original direction.
                if let Some(dir) = ray_directions
                    .iter()
                    .find(|d| d.nonbasic_var == r.variable_index)
                {
                    rl.at_lower_bound = dir.at_lower_bound;
                }
                rl
            })
            .collect();

        // Apply Balas formula.
        let balas_result = self
            .balas
            .compute(&ray_lengths, vertex, lower_bounds, upper_bounds)?;

        let mut generated = Vec::new();

        if let Some(cut) = balas_result.cut {
            let efficacy = cut.compute_efficacy(vertex);
            if efficacy >= self.config.min_efficacy {
                if !self.is_duplicate(&cut) {
                    let ray_summaries: Vec<RayLengthSummary> = ray_results
                        .iter()
                        .map(|r| RayLengthSummary {
                            variable_index: r.variable_index,
                            alpha: r.alpha,
                            intersects: r.intersects,
                        })
                        .collect();

                    self.register_cut(&cut);
                    self.stats.total_generated += 1;
                    self.update_efficacy_stats(efficacy);

                    generated.push(GeneratedCut {
                        cut,
                        efficacy,
                        rank: 1,
                        ray_results: ray_summaries,
                        strengthened: false,
                    });
                } else {
                    self.stats.total_duplicates += 1;
                }
            } else {
                self.stats.total_low_efficacy += 1;
            }
        }

        // Try to generate additional cuts from subsets of rays (deep cuts).
        if self.config.deep_cuts && generated.len() < self.config.max_cuts {
            let extra = self.generate_deep_cuts(
                &ray_lengths,
                vertex,
                lower_bounds,
                upper_bounds,
                self.config.max_cuts - generated.len(),
            );
            generated.extend(extra);
        }

        Ok(generated)
    }

    /// Generate deep cuts from subsets of rays.
    fn generate_deep_cuts(
        &mut self,
        ray_lengths: &[RayLength],
        vertex: &[f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        max_extra: usize,
    ) -> Vec<GeneratedCut> {
        let mut extra_cuts = Vec::new();

        // Try dropping one ray at a time to get alternative cuts.
        for skip in 0..ray_lengths.len().min(max_extra + 1) {
            let subset: Vec<RayLength> = ray_lengths
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != skip)
                .map(|(_, r)| r.clone())
                .collect();

            if subset.is_empty() {
                continue;
            }

            if let Ok(result) = self
                .balas
                .compute(&subset, vertex, lower_bounds, upper_bounds)
            {
                if let Some(cut) = result.cut {
                    let efficacy = cut.compute_efficacy(vertex);
                    if efficacy >= self.config.min_efficacy && !self.is_duplicate(&cut) {
                        self.register_cut(&cut);
                        extra_cuts.push(GeneratedCut {
                            cut,
                            efficacy,
                            rank: 2,
                            ray_results: Vec::new(),
                            strengthened: false,
                        });
                        if extra_cuts.len() >= max_extra {
                            break;
                        }
                    }
                }
            }
        }

        extra_cuts
    }

    /// Check if a cut is a duplicate of a previously generated one.
    fn is_duplicate(&self, cut: &BilevelCut) -> bool {
        let hash = self.cut_hash(cut);
        self.generated_hashes.contains_key(&hash)
    }

    /// Register a cut in the duplicate detection map.
    fn register_cut(&mut self, cut: &BilevelCut) {
        let hash = self.cut_hash(cut);
        self.generated_hashes.insert(hash, self.cut_counter);
        self.cut_counter += 1;
    }

    /// Compute a hash key for duplicate detection.
    fn cut_hash(&self, cut: &BilevelCut) -> Vec<OrderedFloat<f64>> {
        let precision = 1.0 / self.config.duplicate_tolerance;
        cut.coeffs
            .iter()
            .map(|&(j, a)| {
                let rounded = (a * precision).round() / precision;
                OrderedFloat(j as f64 * 1e6 + rounded)
            })
            .collect()
    }

    /// Update running efficacy statistics.
    fn update_efficacy_stats(&mut self, efficacy: f64) {
        let n = self.stats.total_generated as f64;
        self.stats.avg_efficacy = (self.stats.avg_efficacy * (n - 1.0) + efficacy) / n;
        self.stats.max_efficacy = self.stats.max_efficacy.max(efficacy);
    }

    /// Generate a single intersection cut without full statistics tracking.
    pub fn generate_single(
        &mut self,
        vertex: &[f64],
        ray_lengths: &[RayLength],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
    ) -> CutResult<BilevelCut> {
        let result = self
            .balas
            .compute(ray_lengths, vertex, lower_bounds, upper_bounds)?;
        result
            .cut
            .ok_or_else(|| CutError::NumericalIssue("Balas formula produced no cut".into()))
    }

    /// Get statistics.
    pub fn stats(&self) -> &IntersectionStats {
        &self.stats
    }
    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = IntersectionStats::default();
    }
    /// Clear duplicate detection cache.
    pub fn clear_duplicate_cache(&mut self) {
        self.generated_hashes.clear();
    }
    /// Number of cuts generated.
    pub fn cuts_generated(&self) -> u64 {
        self.cut_counter
    }
}

/// Sort cut candidates by efficacy (descending) and return top k.
pub fn select_top_cuts(mut candidates: Vec<CutCandidate>, k: usize) -> Vec<CutCandidate> {
    candidates.sort_by(|a, b| {
        b.efficacy
            .partial_cmp(&a.efficacy)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(k);
    candidates
}

/// Filter cuts by minimum efficacy threshold.
pub fn filter_by_efficacy(candidates: Vec<CutCandidate>, min_eff: f64) -> Vec<CutCandidate> {
    candidates
        .into_iter()
        .filter(|c| c.efficacy >= min_eff)
        .collect()
}

/// Remove near-parallel cuts, keeping the most efficacious from each group.
pub fn remove_parallel_cuts(
    candidates: Vec<CutCandidate>,
    max_cosine: f64,
    dim: usize,
) -> Vec<CutCandidate> {
    let mut kept: Vec<CutCandidate> = Vec::new();
    for cand in candidates {
        let dense = cand.cut.to_dense(dim);
        let is_parallel = kept.iter().any(|k| {
            let k_dense = k.cut.to_dense(dim);
            let cos = cosine_similarity(&dense, &k_dense);
            cos.abs() > max_cosine
        });
        if !is_parallel {
            kept.push(cand);
        }
    }
    kept
}

/// Cosine similarity between two dense vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < TOLERANCE || nb < TOLERANCE {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Normalize a set of generated cuts.
pub fn normalize_cuts(cuts: &mut [GeneratedCut]) {
    for gc in cuts.iter_mut() {
        gc.cut.normalize();
        gc.efficacy = gc.cut.l2_norm(); // after normalization, recompute
    }
}

/// Compute the bilevel gap c^T y - phi(x) at a point.
pub fn bilevel_gap(point: &[f64], n_leader: usize, follower_obj: &[f64], phi_x: f64) -> f64 {
    let cy: f64 = follower_obj
        .iter()
        .enumerate()
        .map(|(i, &c)| c * point.get(n_leader + i).copied().unwrap_or(0.0))
        .sum();
    cy - phi_x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_generator() -> IntersectionCutGenerator {
        IntersectionCutGenerator::new(IntersectionCutConfig::default(), 2, 2, vec![1.0, 1.0])
    }

    #[test]
    fn test_config_default() {
        let cfg = IntersectionCutConfig::default();
        assert!(cfg.min_efficacy > 0.0);
        assert!(cfg.max_cuts > 0);
    }

    #[test]
    fn test_generator_creation() {
        let gen = make_test_generator();
        assert_eq!(gen.cuts_generated(), 0);
    }

    #[test]
    fn test_generate_single() {
        let mut gen = make_test_generator();
        let rays = vec![
            RayLength::finite(0, 2.0, true),
            RayLength::finite(1, 3.0, true),
        ];
        let result = gen.generate_single(&[0.5, 0.5], &rays, &[0.0; 4], &[1.0; 4]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_bilevel_gap_fn() {
        let gap = bilevel_gap(&[0.5, 0.8], 1, &[1.0], 0.3);
        assert!((gap - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_select_top_cuts() {
        let candidates = vec![
            CutCandidate {
                cut: BilevelCut::new(vec![(0, 1.0)], 1.0, ConstraintSense::Ge),
                efficacy: 0.5,
                num_finite_rays: 1,
                num_infinite_rays: 0,
                condition_estimate: 1.0,
                source_vertex: vec![],
            },
            CutCandidate {
                cut: BilevelCut::new(vec![(1, 1.0)], 1.0, ConstraintSense::Ge),
                efficacy: 0.8,
                num_finite_rays: 1,
                num_infinite_rays: 0,
                condition_estimate: 1.0,
                source_vertex: vec![],
            },
        ];
        let top = select_top_cuts(candidates, 1);
        assert_eq!(top.len(), 1);
        assert!((top[0].efficacy - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_filter_by_efficacy() {
        let candidates = vec![CutCandidate {
            cut: BilevelCut::new(vec![(0, 1.0)], 1.0, ConstraintSense::Ge),
            efficacy: 0.001,
            num_finite_rays: 1,
            num_infinite_rays: 0,
            condition_estimate: 1.0,
            source_vertex: vec![],
        }];
        let filtered = filter_by_efficacy(candidates, 0.01);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let sim = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_parallel() {
        let sim = cosine_similarity(&[1.0, 0.0], &[2.0, 0.0]);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_remove_parallel_cuts() {
        let candidates = vec![
            CutCandidate {
                cut: BilevelCut::new(vec![(0, 1.0)], 1.0, ConstraintSense::Ge),
                efficacy: 0.5,
                num_finite_rays: 1,
                num_infinite_rays: 0,
                condition_estimate: 1.0,
                source_vertex: vec![],
            },
            CutCandidate {
                cut: BilevelCut::new(vec![(0, 2.0)], 2.0, ConstraintSense::Ge),
                efficacy: 0.3,
                num_finite_rays: 1,
                num_infinite_rays: 0,
                condition_estimate: 1.0,
                source_vertex: vec![],
            },
        ];
        let result = remove_parallel_cuts(candidates, 0.99, 2);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_stats_default() {
        let stats = IntersectionStats::default();
        assert_eq!(stats.total_attempts, 0);
        assert_eq!(stats.total_generated, 0);
    }

    #[test]
    fn test_duplicate_detection() {
        let mut gen = make_test_generator();
        let cut = BilevelCut::new(vec![(0, 1.0), (1, 2.0)], 3.0, ConstraintSense::Ge);
        assert!(!gen.is_duplicate(&cut));
        gen.register_cut(&cut);
        assert!(gen.is_duplicate(&cut));
    }
}
