//! Search strategies for sonification parameter optimization.
//!
//! Provides greedy, simulated annealing, beam search, random restart,
//! and hybrid search strategies.

use std::collections::HashMap;
use std::time::Instant;

use crate::{
    MappingConfig, OptimizerError, OptimizerResult, OptimizationSolution,
    ParameterId, StreamId, StreamMapping, BarkBand,
};
use crate::config::OptimizerConfig;
use crate::constraints::ConstraintSet;
use crate::objective::ObjectiveFn;
use crate::propagation::{Domain, DomainStore};

// ─────────────────────────────────────────────────────────────────────────────
// GreedySearch
// ─────────────────────────────────────────────────────────────────────────────

/// Greedy parameter assignment: for each stream, greedily assign to the
/// best available spectral region. O(k * B) where k = streams, B = Bark bands.
#[derive(Debug, Clone)]
pub struct GreedySearch {
    /// Number of Bark bands to consider.
    pub num_bands: usize,
    /// Frequency range for each band.
    pub band_centers: Vec<f64>,
}

impl Default for GreedySearch {
    fn default() -> Self {
        let band_centers: Vec<f64> = (0..BarkBand::NUM_BANDS)
            .map(|i| BarkBand(i as u8).center_frequency())
            .collect();
        GreedySearch {
            num_bands: BarkBand::NUM_BANDS,
            band_centers,
        }
    }
}

impl GreedySearch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run greedy search.
    pub fn search(
        &self,
        stream_ids: &[StreamId],
        objective: &dyn ObjectiveFn,
        constraints: &ConstraintSet,
        base_amplitude_db: f64,
    ) -> OptimizerResult<OptimizationSolution> {
        let start = Instant::now();
        let mut config = MappingConfig::new();
        let mut used_bands: Vec<bool> = vec![false; self.num_bands];
        let mut best_value = f64::NEG_INFINITY;

        for &stream_id in stream_ids {
            let mut best_band = 0;
            let mut best_obj = f64::NEG_INFINITY;

            for band in 0..self.num_bands {
                if used_bands[band] {
                    continue;
                }

                // Try placing this stream in this band
                let freq = self.band_centers[band];
                let mut trial_config = config.clone();
                trial_config.stream_params.insert(
                    stream_id,
                    StreamMapping::new(stream_id, freq, base_amplitude_db),
                );

                let report = constraints.check_all(&trial_config);
                if !report.all_satisfied {
                    continue;
                }

                match objective.evaluate(&trial_config) {
                    Ok(val) => {
                        if val > best_obj {
                            best_obj = val;
                            best_band = band;
                        }
                    }
                    Err(_) => continue,
                }
            }

            // Assign stream to best band
            let freq = self.band_centers[best_band];
            config.stream_params.insert(
                stream_id,
                StreamMapping::new(stream_id, freq, base_amplitude_db),
            );
            used_bands[best_band] = true;
            best_value = best_obj;
        }

        if config.stream_params.is_empty() {
            return Err(OptimizerError::Infeasible(
                "No feasible assignment found by greedy search".into(),
            ));
        }

        // Final evaluation
        let final_value = objective.evaluate(&config).unwrap_or(best_value);

        Ok(OptimizationSolution {
            config,
            objective_value: final_value,
            objective_values: HashMap::new(),
            constraint_satisfaction: 1.0,
            solve_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            nodes_explored: stream_ids.len() * self.num_bands,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SimulatedAnnealing
// ─────────────────────────────────────────────────────────────────────────────

/// Simulated annealing with Metropolis acceptance criterion.
#[derive(Debug, Clone)]
pub struct SimulatedAnnealing {
    pub initial_temperature: f64,
    pub cooling_rate: f64,
    pub min_temperature: f64,
    pub steps_per_temperature: usize,
    pub max_total_steps: usize,
    /// Perturbation scale for neighbor generation.
    pub perturbation_scale: f64,
}

impl Default for SimulatedAnnealing {
    fn default() -> Self {
        SimulatedAnnealing {
            initial_temperature: 100.0,
            cooling_rate: 0.995,
            min_temperature: 0.01,
            steps_per_temperature: 50,
            max_total_steps: 100_000,
            perturbation_scale: 0.1,
        }
    }
}

impl SimulatedAnnealing {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_config(config: &OptimizerConfig) -> Self {
        SimulatedAnnealing {
            initial_temperature: config.strategy.sa_initial_temperature,
            cooling_rate: config.strategy.sa_cooling_rate,
            min_temperature: config.strategy.sa_min_temperature,
            steps_per_temperature: config.strategy.sa_steps_per_temperature,
            max_total_steps: config.solver.max_iterations,
            perturbation_scale: 0.1,
        }
    }

    /// Run simulated annealing.
    pub fn search(
        &self,
        initial: MappingConfig,
        objective: &dyn ObjectiveFn,
        constraints: &ConstraintSet,
        seed: u64,
    ) -> OptimizerResult<OptimizationSolution> {
        let start = Instant::now();
        let mut current = initial.clone();
        let mut current_value = objective.evaluate(&current)?;
        let mut best = current.clone();
        let mut best_value = current_value;
        let mut temperature = self.initial_temperature;
        let mut total_steps = 0;
        let mut rng_state = seed;

        while temperature > self.min_temperature && total_steps < self.max_total_steps {
            for _ in 0..self.steps_per_temperature {
                total_steps += 1;

                // Generate neighbor by perturbing one stream
                let neighbor = self.generate_neighbor(&current, &mut rng_state);
                let report = constraints.check_all(&neighbor);
                if !report.all_satisfied {
                    continue;
                }

                match objective.evaluate(&neighbor) {
                    Ok(neighbor_value) => {
                        let delta = neighbor_value - current_value;

                        // Metropolis acceptance
                        if delta > 0.0 || self.accept(delta, temperature, &mut rng_state) {
                            current = neighbor;
                            current_value = neighbor_value;

                            if current_value > best_value {
                                best = current.clone();
                                best_value = current_value;
                            }
                        }
                    }
                    Err(_) => continue,
                }
            }

            // Cool down
            temperature *= self.cooling_rate;
        }

        Ok(OptimizationSolution {
            config: best,
            objective_value: best_value,
            objective_values: HashMap::new(),
            constraint_satisfaction: 1.0,
            solve_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            nodes_explored: total_steps,
        })
    }

    fn generate_neighbor(&self, config: &MappingConfig, rng_state: &mut u64) -> MappingConfig {
        let mut neighbor = config.clone();
        let stream_ids: Vec<StreamId> = neighbor.stream_params.keys().cloned().collect();

        if stream_ids.is_empty() {
            return neighbor;
        }

        // Pick a random stream to perturb
        let idx = pseudo_random(rng_state) % stream_ids.len();
        let sid = stream_ids[idx];

        if let Some(mapping) = neighbor.stream_params.get_mut(&sid) {
            let r1 = pseudo_random_f64(rng_state);
            let r2 = pseudo_random_f64(rng_state);

            // Perturb frequency (log-scale)
            let freq_factor = 1.0 + (r1 - 0.5) * 2.0 * self.perturbation_scale;
            mapping.frequency_hz *= freq_factor;
            mapping.frequency_hz = mapping.frequency_hz.clamp(20.0, 20000.0);

            // Perturb amplitude
            mapping.amplitude_db += (r2 - 0.5) * 2.0 * self.perturbation_scale * 10.0;
            mapping.amplitude_db = mapping.amplitude_db.clamp(-80.0, 100.0);
        }

        neighbor
    }

    fn accept(&self, delta: f64, temperature: f64, rng_state: &mut u64) -> bool {
        if temperature <= 0.0 {
            return false;
        }
        let prob = (delta / temperature).exp();
        let r = pseudo_random_f64(rng_state);
        r < prob
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BeamSearch
// ─────────────────────────────────────────────────────────────────────────────

/// Beam search: maintain top-k partial solutions.
#[derive(Debug, Clone)]
pub struct BeamSearch {
    pub beam_width: usize,
    pub max_depth: usize,
}

impl Default for BeamSearch {
    fn default() -> Self {
        BeamSearch {
            beam_width: 10,
            max_depth: 50,
        }
    }
}

impl BeamSearch {
    pub fn new(beam_width: usize) -> Self {
        BeamSearch {
            beam_width,
            max_depth: 50,
        }
    }

    /// Run beam search over parameter assignments.
    pub fn search(
        &self,
        stream_ids: &[StreamId],
        candidate_freqs: &[f64],
        base_amplitude_db: f64,
        objective: &dyn ObjectiveFn,
        constraints: &ConstraintSet,
    ) -> OptimizerResult<OptimizationSolution> {
        let start = Instant::now();

        // Initialize beam with empty configs
        let mut beam: Vec<(MappingConfig, f64)> = vec![(MappingConfig::new(), 0.0)];
        let mut total_nodes = 0;

        // Assign each stream one at a time
        for &stream_id in stream_ids {
            let mut candidates: Vec<(MappingConfig, f64)> = Vec::new();

            for (config, _) in &beam {
                for &freq in candidate_freqs {
                    let mut trial = config.clone();
                    trial.stream_params.insert(
                        stream_id,
                        StreamMapping::new(stream_id, freq, base_amplitude_db),
                    );
                    total_nodes += 1;

                    let report = constraints.check_all(&trial);
                    if !report.all_satisfied {
                        continue;
                    }

                    match objective.evaluate(&trial) {
                        Ok(val) => candidates.push((trial, val)),
                        Err(_) => continue,
                    }
                }
            }

            // Keep top beam_width candidates
            candidates.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(self.beam_width);

            if candidates.is_empty() {
                // No feasible candidates; keep current beam
                continue;
            }

            beam = candidates;
        }

        if beam.is_empty() {
            return Err(OptimizerError::Infeasible("Beam search found no feasible solution".into()));
        }

        let (best_config, best_value) = beam.into_iter().next().unwrap();

        Ok(OptimizationSolution {
            config: best_config,
            objective_value: best_value,
            objective_values: HashMap::new(),
            constraint_satisfaction: 1.0,
            solve_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            nodes_explored: total_nodes,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RandomRestart
// ─────────────────────────────────────────────────────────────────────────────

/// Random restart local search: run SA from multiple random starting points.
#[derive(Debug, Clone)]
pub struct RandomRestart {
    pub num_restarts: usize,
    pub sa: SimulatedAnnealing,
}

impl Default for RandomRestart {
    fn default() -> Self {
        RandomRestart {
            num_restarts: 5,
            sa: SimulatedAnnealing::default(),
        }
    }
}

impl RandomRestart {
    pub fn new(num_restarts: usize) -> Self {
        RandomRestart {
            num_restarts,
            sa: SimulatedAnnealing::default(),
        }
    }

    pub fn with_sa(mut self, sa: SimulatedAnnealing) -> Self {
        self.sa = sa;
        self
    }

    /// Run random restart search.
    pub fn search(
        &self,
        stream_ids: &[StreamId],
        objective: &dyn ObjectiveFn,
        constraints: &ConstraintSet,
        base_seed: u64,
    ) -> OptimizerResult<OptimizationSolution> {
        let start = Instant::now();
        let mut best_solution: Option<OptimizationSolution> = None;
        let mut total_nodes = 0;

        for restart in 0..self.num_restarts {
            let seed = base_seed.wrapping_add(restart as u64 * 12345);
            let initial = generate_random_config(stream_ids, seed);

            match self.sa.search(initial, objective, constraints, seed) {
                Ok(solution) => {
                    total_nodes += solution.nodes_explored;

                    if best_solution
                        .as_ref()
                        .map_or(true, |best| solution.objective_value > best.objective_value)
                    {
                        best_solution = Some(solution);
                    }
                }
                Err(_) => continue,
            }
        }

        match best_solution {
            Some(mut sol) => {
                sol.solve_time_ms = start.elapsed().as_secs_f64() * 1000.0;
                sol.nodes_explored = total_nodes;
                Ok(sol)
            }
            None => Err(OptimizerError::Infeasible(
                "All random restarts failed".into(),
            )),
        }
    }
}

/// Generate a random initial configuration.
fn generate_random_config(stream_ids: &[StreamId], seed: u64) -> MappingConfig {
    let mut config = MappingConfig::new();
    let mut rng_state = seed;

    for &sid in stream_ids {
        let freq = 100.0 + pseudo_random_f64(&mut rng_state) * 7900.0; // 100-8000 Hz
        let amp = 30.0 + pseudo_random_f64(&mut rng_state) * 60.0; // 30-90 dB
        config
            .stream_params
            .insert(sid, StreamMapping::new(sid, freq, amp));
    }

    config
}

// ─────────────────────────────────────────────────────────────────────────────
// HybridSearch
// ─────────────────────────────────────────────────────────────────────────────

/// Hybrid search: combine global (B&B) with local (SA) refinement.
#[derive(Debug, Clone)]
pub struct HybridSearch {
    pub sa: SimulatedAnnealing,
    pub greedy: GreedySearch,
    pub num_sa_refinements: usize,
}

impl Default for HybridSearch {
    fn default() -> Self {
        HybridSearch {
            sa: SimulatedAnnealing {
                initial_temperature: 10.0,
                cooling_rate: 0.99,
                steps_per_temperature: 20,
                max_total_steps: 5000,
                ..Default::default()
            },
            greedy: GreedySearch::default(),
            num_sa_refinements: 3,
        }
    }
}

impl HybridSearch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run hybrid search: greedy initialization + SA refinement.
    pub fn search(
        &self,
        stream_ids: &[StreamId],
        objective: &dyn ObjectiveFn,
        constraints: &ConstraintSet,
        seed: u64,
    ) -> OptimizerResult<OptimizationSolution> {
        let start = Instant::now();

        // Phase 1: Greedy initialization
        let greedy_result = self.greedy.search(
            stream_ids,
            objective,
            constraints,
            60.0, // base amplitude
        )?;

        let mut best = greedy_result;
        let mut total_nodes = best.nodes_explored;

        // Phase 2: SA refinement from greedy solution
        for i in 0..self.num_sa_refinements {
            let refinement_seed = seed.wrapping_add(i as u64 * 9999);
            match self.sa.search(
                best.config.clone(),
                objective,
                constraints,
                refinement_seed,
            ) {
                Ok(refined) => {
                    total_nodes += refined.nodes_explored;
                    if refined.objective_value > best.objective_value {
                        best = refined;
                    }
                }
                Err(_) => continue,
            }
        }

        best.solve_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        best.nodes_explored = total_nodes;
        Ok(best)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pseudo-random number generation (deterministic, no external dependency)
// ─────────────────────────────────────────────────────────────────────────────

fn pseudo_random(state: &mut u64) -> usize {
    // xorshift64
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state) as usize
}

fn pseudo_random_f64(state: &mut u64) -> f64 {
    pseudo_random(state);
    (*state as f64) / (u64::MAX as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::Constraint;
    use crate::objective::{DiscriminabilityObjective, LatencyObjective};

    fn basic_constraints() -> ConstraintSet {
        let mut cs = ConstraintSet::new();
        cs.add(Constraint::FrequencyRange { min_hz: 100.0, max_hz: 8000.0 });
        cs.add(Constraint::AmplitudeRange { min_db: 20.0, max_db: 90.0 });
        cs
    }

    #[test]
    fn test_greedy_search_basic() {
        let greedy = GreedySearch::new();
        let streams = vec![StreamId(0), StreamId(1)];
        let objective = DiscriminabilityObjective::new();
        let constraints = basic_constraints();

        let result = greedy.search(&streams, &objective, &constraints, 60.0);
        assert!(result.is_ok());
        let sol = result.unwrap();
        assert_eq!(sol.config.stream_params.len(), 2);
    }

    #[test]
    fn test_greedy_search_single_stream() {
        let greedy = GreedySearch::new();
        let streams = vec![StreamId(0)];
        let objective = LatencyObjective::default();
        let constraints = basic_constraints();

        let result = greedy.search(&streams, &objective, &constraints, 60.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_simulated_annealing_basic() {
        let sa = SimulatedAnnealing {
            initial_temperature: 10.0,
            cooling_rate: 0.9,
            steps_per_temperature: 5,
            max_total_steps: 100,
            ..Default::default()
        };

        let streams = vec![StreamId(0), StreamId(1)];
        let initial = generate_random_config(&streams, 42);
        let objective = DiscriminabilityObjective::new();
        let constraints = basic_constraints();

        let result = sa.search(initial, &objective, &constraints, 42);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sa_improves_or_maintains() {
        let sa = SimulatedAnnealing {
            initial_temperature: 50.0,
            cooling_rate: 0.95,
            steps_per_temperature: 10,
            max_total_steps: 500,
            perturbation_scale: 0.05,
            ..Default::default()
        };

        let streams = vec![StreamId(0), StreamId(1)];
        let initial = generate_random_config(&streams, 123);
        let objective = DiscriminabilityObjective::new();
        let constraints = basic_constraints();

        let initial_value = objective.evaluate(&initial).unwrap_or(0.0);
        let result = sa.search(initial, &objective, &constraints, 123).unwrap();

        // SA's best should be at least as good as initial
        assert!(
            result.objective_value >= initial_value - 0.01,
            "SA should maintain or improve: {} vs {}",
            result.objective_value,
            initial_value
        );
    }

    #[test]
    fn test_beam_search_basic() {
        let beam = BeamSearch::new(5);
        let streams = vec![StreamId(0), StreamId(1)];
        let freqs: Vec<f64> = (0..BarkBand::NUM_BANDS)
            .map(|i| BarkBand(i as u8).center_frequency())
            .collect();
        let objective = DiscriminabilityObjective::new();
        let constraints = basic_constraints();

        let result = beam.search(&streams, &freqs, 60.0, &objective, &constraints);
        assert!(result.is_ok());
        let sol = result.unwrap();
        assert_eq!(sol.config.stream_params.len(), 2);
    }

    #[test]
    fn test_beam_width_affects_quality() {
        let objective = DiscriminabilityObjective::new();
        let constraints = basic_constraints();
        let streams = vec![StreamId(0), StreamId(1), StreamId(2)];
        let freqs: Vec<f64> = (0..BarkBand::NUM_BANDS)
            .map(|i| BarkBand(i as u8).center_frequency())
            .collect();

        let narrow = BeamSearch::new(2);
        let wide = BeamSearch::new(20);

        let r_narrow = narrow.search(&streams, &freqs, 60.0, &objective, &constraints).unwrap();
        let r_wide = wide.search(&streams, &freqs, 60.0, &objective, &constraints).unwrap();

        // Wider beam explores more => at least as good
        assert!(r_wide.objective_value >= r_narrow.objective_value - 0.01);
    }

    #[test]
    fn test_random_restart_basic() {
        let rr = RandomRestart::new(3).with_sa(SimulatedAnnealing {
            initial_temperature: 10.0,
            cooling_rate: 0.9,
            steps_per_temperature: 5,
            max_total_steps: 50,
            ..Default::default()
        });

        let streams = vec![StreamId(0), StreamId(1)];
        let objective = DiscriminabilityObjective::new();
        let constraints = basic_constraints();

        let result = rr.search(&streams, &objective, &constraints, 42);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hybrid_search_basic() {
        let hybrid = HybridSearch {
            sa: SimulatedAnnealing {
                initial_temperature: 5.0,
                cooling_rate: 0.9,
                steps_per_temperature: 5,
                max_total_steps: 50,
                ..Default::default()
            },
            num_sa_refinements: 2,
            ..Default::default()
        };

        let streams = vec![StreamId(0), StreamId(1)];
        let objective = DiscriminabilityObjective::new();
        let constraints = basic_constraints();

        let result = hybrid.search(&streams, &objective, &constraints, 42);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hybrid_better_than_greedy() {
        let greedy = GreedySearch::new();
        let hybrid = HybridSearch {
            sa: SimulatedAnnealing {
                initial_temperature: 20.0,
                cooling_rate: 0.95,
                steps_per_temperature: 10,
                max_total_steps: 200,
                perturbation_scale: 0.05,
                ..Default::default()
            },
            num_sa_refinements: 2,
            ..Default::default()
        };

        let streams = vec![StreamId(0), StreamId(1), StreamId(2)];
        let objective = DiscriminabilityObjective::new();
        let constraints = basic_constraints();

        let g = greedy.search(&streams, &objective, &constraints, 60.0).unwrap();
        let h = hybrid.search(&streams, &objective, &constraints, 42).unwrap();

        // Hybrid should be at least as good as greedy alone
        assert!(
            h.objective_value >= g.objective_value - 0.1,
            "Hybrid {} should be >= greedy {}",
            h.objective_value,
            g.objective_value
        );
    }

    #[test]
    fn test_generate_random_config() {
        let streams = vec![StreamId(0), StreamId(1), StreamId(2)];
        let config = generate_random_config(&streams, 42);
        assert_eq!(config.stream_params.len(), 3);
        for (_, mapping) in &config.stream_params {
            assert!(mapping.frequency_hz >= 20.0);
            assert!(mapping.frequency_hz <= 20000.0);
        }
    }

    #[test]
    fn test_pseudo_random_deterministic() {
        let mut s1 = 42u64;
        let mut s2 = 42u64;
        let r1 = pseudo_random(&mut s1);
        let r2 = pseudo_random(&mut s2);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_pseudo_random_f64_range() {
        let mut state = 12345u64;
        for _ in 0..100 {
            let v = pseudo_random_f64(&mut state);
            assert!(v >= 0.0 && v <= 1.0, "Random f64 out of range: {}", v);
        }
    }
}
