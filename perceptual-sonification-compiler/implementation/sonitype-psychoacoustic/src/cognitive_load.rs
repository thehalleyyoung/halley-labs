//! Cognitive Load Budget Algebra for Psychoacoustic Sonification
//!
//! Implements a resource algebra (L, ⊕, ≤ B) where L is the set of cognitive loads,
//! ⊕ is a composition operator that accounts for inter-stream interference, and B is
//! a budget derived from working-memory research (Cowan's 4±1 rule).
//!
//! The key insight is that monitoring multiple auditory streams is NOT additive in
//! cognitive cost: interference between streams grows super-linearly with stream count,
//! and familiarity / temporal regularity can reduce load through chunking.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// 1. Local Types
// ---------------------------------------------------------------------------

/// A descriptor for an auditory stream's cognitive properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamDescriptor {
    /// Unique identifier for this stream.
    pub id: usize,
    /// Human-readable name (e.g. "pitch contour", "rhythm layer").
    pub name: String,
    /// Shannon information rate carried by this stream (bits/s).
    pub information_rate_bits_per_sec: f64,
    /// Normalised complexity of the stream (0 = trivial, 1 = maximally complex).
    pub stream_complexity: f64,
    /// How often the stream's state changes (Hz).
    pub update_rate_hz: f64,
    /// Number of distinct spectral features, normalised to 0-1.
    pub spectral_complexity: f64,
    /// Temporal regularity: 0 = irregular, 1 = perfectly periodic.
    pub temporal_regularity: f64,
    /// Listener familiarity: 0 = novel, 1 = highly familiar.
    pub familiarity: f64,
    /// Priority weight (0 = unimportant, 1 = critical).
    pub priority: f64,
}

impl StreamDescriptor {
    /// Convenience constructor with all fields explicit.
    pub fn new(
        id: usize,
        name: impl Into<String>,
        information_rate_bits_per_sec: f64,
        stream_complexity: f64,
        update_rate_hz: f64,
        spectral_complexity: f64,
        temporal_regularity: f64,
        familiarity: f64,
        priority: f64,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            information_rate_bits_per_sec,
            stream_complexity: stream_complexity.clamp(0.0, 1.0),
            update_rate_hz: update_rate_hz.max(0.0),
            spectral_complexity: spectral_complexity.clamp(0.0, 1.0),
            temporal_regularity: temporal_regularity.clamp(0.0, 1.0),
            familiarity: familiarity.clamp(0.0, 1.0),
            priority: priority.clamp(0.0, 1.0),
        }
    }

    /// Build a simple stream for quick experimentation.
    pub fn simple(id: usize, name: impl Into<String>, complexity: f64, priority: f64) -> Self {
        Self::new(id, name, 8.0, complexity, 2.0, complexity, 0.5, 0.5, priority)
    }
}

/// Budget constraints derived from psychoacoustic / working-memory literature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveLoadBudget {
    /// Maximum number of concurrent auditory streams.
    pub max_streams: usize,
    /// Maximum normalised cognitive load (1.0 = full capacity).
    pub max_load: f64,
    /// Maximum aggregate information rate the listener can track (bits/s).
    pub max_information_rate: f64,
}

impl CognitiveLoadBudget {
    pub fn new(max_streams: usize, max_load: f64, max_information_rate: f64) -> Self {
        Self {
            max_streams,
            max_load,
            max_information_rate,
        }
    }
}

// ---------------------------------------------------------------------------
// 2. CognitiveLoadModel – the resource algebra (L, ⊕, ≤ B)
// ---------------------------------------------------------------------------

/// Core model that computes per-stream loads and composes them with
/// an interference-aware operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveLoadModel {
    pub budget: CognitiveLoadBudget,
    /// Pairwise interference coefficient used in `compose_loads`.
    pub interference_alpha: f64,
    /// Maximum expected information rate used for normalisation (bits/s).
    pub reference_info_rate: f64,
}

impl CognitiveLoadModel {
    // -- constructors -------------------------------------------------------

    pub fn new(budget: CognitiveLoadBudget) -> Self {
        Self {
            budget,
            interference_alpha: 0.05,
            reference_info_rate: 50.0,
        }
    }

    /// Cowan's 4±1 rule encoded as the default budget.
    pub fn default_budget() -> CognitiveLoadBudget {
        CognitiveLoadBudget {
            max_streams: 4,
            max_load: 1.0,
            max_information_rate: 40.0, // ~40 bits/s is a reasonable tracking ceiling
        }
    }

    pub fn with_default_budget() -> Self {
        Self::new(Self::default_budget())
    }

    // -- per-stream cost ----------------------------------------------------

    /// Compute the cognitive cost of monitoring a single stream.
    ///
    /// Formula (before discounts):
    ///   base = 0.15
    ///        + 0.30 × complexity
    ///        + 0.20 × min(update_rate / 10, 1)
    ///        + 0.15 × (info_rate / reference_info_rate).min(1)
    ///        + 0.10 × spectral_complexity
    ///
    /// Familiarity discount:  ×(1 − 0.3 × familiarity)
    /// Regularity discount:   ×(1 − 0.2 × regularity)
    pub fn stream_load(&self, stream: &StreamDescriptor) -> f64 {
        let info_norm = (stream.information_rate_bits_per_sec / self.reference_info_rate).min(1.0);
        let update_norm = (stream.update_rate_hz / 10.0).min(1.0);

        let base = 0.15
            + 0.30 * stream.stream_complexity
            + 0.20 * update_norm
            + 0.15 * info_norm
            + 0.10 * stream.spectral_complexity;

        let after_familiarity = base * (1.0 - 0.3 * stream.familiarity);
        let after_regularity = after_familiarity * (1.0 - 0.2 * stream.temporal_regularity);

        after_regularity.clamp(0.0, 1.0)
    }

    /// Compute loads for every stream in a slice.
    pub fn stream_loads(&self, streams: &[StreamDescriptor]) -> Vec<f64> {
        streams.iter().map(|s| self.stream_load(s)).collect()
    }

    // -- composition (the ⊕ operator) ---------------------------------------

    /// Compose a set of individual loads into an aggregate load.
    ///
    /// This is **not** a simple sum. Pairwise interference between n streams
    /// adds a penalty of `α × n(n−1)/2` where α = `interference_alpha`.
    pub fn compose_loads(&self, loads: &[f64]) -> f64 {
        if loads.is_empty() {
            return 0.0;
        }
        let sum: f64 = loads.iter().copied().sum();
        let n = loads.len() as f64;
        let pairwise = n * (n - 1.0) / 2.0;
        let interference = self.interference_alpha * pairwise;
        (sum + interference).min(2.0) // soft-cap at 2× budget for legibility
    }

    // -- budget checks ------------------------------------------------------

    /// Returns `true` when `total_load ≤ budget.max_load`.
    pub fn check_budget(&self, total_load: f64) -> bool {
        total_load <= self.budget.max_load
    }

    /// How much budget headroom remains.
    pub fn remaining_budget(&self, current_load: f64) -> f64 {
        (self.budget.max_load - current_load).max(0.0)
    }

    /// Estimate how many *additional* identical average-load streams could be
    /// added before the budget is exhausted.
    pub fn max_additional_streams(&self, current_loads: &[f64]) -> usize {
        let current_total = self.compose_loads(current_loads);
        if current_total >= self.budget.max_load {
            return 0;
        }

        let avg_load = if current_loads.is_empty() {
            0.20 // reasonable default for an "average" stream
        } else {
            current_loads.iter().copied().sum::<f64>() / current_loads.len() as f64
        };

        let mut count = 0usize;
        let mut trial_loads: Vec<f64> = current_loads.to_vec();
        loop {
            trial_loads.push(avg_load);
            let new_total = self.compose_loads(&trial_loads);
            if new_total > self.budget.max_load {
                break;
            }
            count += 1;
            if count > self.budget.max_streams * 2 {
                break; // safety valve
            }
        }
        count
    }

    /// Full diagnostic: compute per-stream loads, compose, and check budget.
    pub fn evaluate(&self, streams: &[StreamDescriptor]) -> LoadReport {
        let per_stream: Vec<(usize, f64)> = streams
            .iter()
            .map(|s| (s.id, self.stream_load(s)))
            .collect();
        let loads: Vec<f64> = per_stream.iter().map(|(_, l)| *l).collect();
        let composed = self.compose_loads(&loads);
        let within = self.check_budget(composed);
        let utilization = if self.budget.max_load > 0.0 {
            composed / self.budget.max_load
        } else {
            f64::INFINITY
        };
        LoadReport {
            stream_loads: per_stream,
            composed_load: composed,
            budget: self.budget.max_load,
            within_budget: within,
            utilization,
            headroom: self.remaining_budget(composed),
        }
    }
}

// ---------------------------------------------------------------------------
// 3. WorkingMemoryModel
// ---------------------------------------------------------------------------

/// Model of auditory working-memory constraints, chunking, and attention
/// switching overhead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemoryModel {
    /// Cowan-style base capacity (default 4).
    pub base_capacity: usize,
    /// Time cost of a single attention switch (seconds).
    pub switch_cost_s: f64,
}

impl WorkingMemoryModel {
    pub fn new() -> Self {
        Self {
            base_capacity: 4,
            switch_cost_s: 0.05,
        }
    }

    /// Estimate how many streams of a given complexity can be held
    /// simultaneously.  Based on Cowan's 4±1, scaled inversely with
    /// complexity.
    ///
    /// capacity = floor(4 / (0.5 + 0.5 × complexity)),  clamped to 1..=7
    pub fn capacity_estimate(&self, stream_complexity: f64) -> usize {
        let c = stream_complexity.clamp(0.0, 1.0);
        let raw = (self.base_capacity as f64) / (0.5 + 0.5 * c);
        (raw.floor() as usize).clamp(1, 7)
    }

    /// Cost of switching attention among `n_streams` active streams.
    ///
    /// Linear term: 0.05 × (n−1)  (one switch per additional stream)
    /// Quadratic term: 0.01 × (n−1)(n−2)/2  (increased confusion)
    pub fn attention_switching_cost(&self, n_streams: usize) -> f64 {
        if n_streams <= 1 {
            return 0.0;
        }
        let n = n_streams as f64;
        let linear = self.switch_cost_s * (n - 1.0);
        let quadratic = 0.01 * (n - 1.0) * (n - 2.0) / 2.0;
        linear + quadratic
    }

    /// Effective information density of a stream (bits/s normalised by
    /// update rate to give bits per update, then scaled back to a rate).
    pub fn information_density(&self, stream: &StreamDescriptor) -> f64 {
        if stream.update_rate_hz <= 0.0 {
            return stream.information_rate_bits_per_sec;
        }
        let bits_per_update = stream.information_rate_bits_per_sec / stream.update_rate_hz;
        let temporal_density = stream.update_rate_hz * (1.0 - stream.temporal_regularity * 0.3);
        bits_per_update * temporal_density
    }

    /// Benefit from temporal chunking: regular, familiar patterns can be
    /// grouped, reducing the per-stream load.
    ///
    /// chunking_factor = regularity × (1 − e^{−duration/5}) × 0.3
    pub fn temporal_chunking_benefit(&self, regularity: f64, duration_s: f64) -> f64 {
        let r = regularity.clamp(0.0, 1.0);
        let exposure = 1.0 - (-duration_s / 5.0).exp();
        r * exposure * 0.3
    }

    /// Effective load of a stream after accounting for chunking due to
    /// exposure over `exposure_duration_s` seconds.
    pub fn effective_load(
        &self,
        stream: &StreamDescriptor,
        exposure_duration_s: f64,
    ) -> f64 {
        let model = CognitiveLoadModel::with_default_budget();
        let base = model.stream_load(stream);
        let chunking = self.temporal_chunking_benefit(
            stream.temporal_regularity,
            exposure_duration_s,
        );
        (base - chunking).max(0.01) // never fully free
    }

    /// How load changes over an extended listening period: returns a vector
    /// of (time_s, effective_load) samples.
    pub fn load_over_time(
        &self,
        stream: &StreamDescriptor,
        total_duration_s: f64,
        sample_count: usize,
    ) -> Vec<(f64, f64)> {
        let dt = if sample_count > 1 {
            total_duration_s / (sample_count - 1) as f64
        } else {
            total_duration_s
        };
        (0..sample_count)
            .map(|i| {
                let t = dt * i as f64;
                (t, self.effective_load(stream, t))
            })
            .collect()
    }

    /// Aggregate effective load of multiple streams at a given exposure time,
    /// including attention-switching overhead.
    pub fn aggregate_effective_load(
        &self,
        streams: &[StreamDescriptor],
        exposure_duration_s: f64,
    ) -> f64 {
        let per_stream: Vec<f64> = streams
            .iter()
            .map(|s| self.effective_load(s, exposure_duration_s))
            .collect();
        let model = CognitiveLoadModel::with_default_budget();
        let composed = model.compose_loads(&per_stream);
        let switching = self.attention_switching_cost(streams.len());
        composed + switching
    }
}

impl Default for WorkingMemoryModel {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// 4. LoadComposition – pure algebra helpers
// ---------------------------------------------------------------------------

/// Stateless helpers exposing the resource-algebra operations.
pub struct LoadComposition;

impl LoadComposition {
    /// Identity element of ⊕: adding zero load changes nothing.
    pub fn identity() -> f64 {
        0.0
    }

    /// Binary composition: a ⊕ b = a + b + α·a·b   (α ≈ 0.1).
    ///
    /// The cross-term models mutual interference: two simultaneous streams
    /// whose loads are both nonzero interfere proportionally to the product
    /// of their individual loads.
    pub fn compose(a: f64, b: f64) -> f64 {
        let alpha = 0.1;
        a + b + alpha * a * b
    }

    /// Left-fold `compose` over an arbitrary number of loads.
    pub fn compose_many(loads: &[f64]) -> f64 {
        loads
            .iter()
            .copied()
            .fold(Self::identity(), |acc, l| Self::compose(acc, l))
    }

    /// Is the aggregated load within the given budget?
    pub fn is_within_budget(load: f64, budget: f64) -> bool {
        load <= budget
    }

    /// Ratio of current load to budget (0.0 = idle, 1.0 = saturated).
    pub fn budget_utilization(load: f64, budget: f64) -> f64 {
        if budget <= 0.0 {
            return f64::INFINITY;
        }
        load / budget
    }

    /// Inverse of `compose` for a single split: given composed value c and
    /// one component a, recover b such that compose(a, b) ≈ c.
    pub fn decompose(composed: f64, a: f64) -> f64 {
        let alpha = 0.1;
        let denom = 1.0 + alpha * a;
        if denom.abs() < 1e-12 {
            return composed - a;
        }
        (composed - a) / denom
    }

    /// Verify algebraic properties hold for given inputs (useful in tests).
    pub fn verify_identity(value: f64) -> bool {
        let result = Self::compose(value, Self::identity());
        (result - value).abs() < 1e-10
    }

    /// Check approximate commutativity: compose(a,b) ≈ compose(b,a).
    pub fn verify_commutativity(a: f64, b: f64) -> bool {
        let ab = Self::compose(a, b);
        let ba = Self::compose(b, a);
        (ab - ba).abs() < 1e-10
    }
}

// ---------------------------------------------------------------------------
// 5. CognitiveLoadOptimizer
// ---------------------------------------------------------------------------

/// Plan produced by the optimizer when current streams exceed the budget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamReductionPlan {
    /// Stream IDs that should be kept.
    pub keep: Vec<usize>,
    /// Stream IDs that should be removed.
    pub remove: Vec<usize>,
    /// Pairs of stream IDs that could be merged.
    pub merge: Vec<(usize, usize)>,
    /// Estimated load after applying the plan.
    pub resulting_load: f64,
    /// Whether the resulting load satisfies the budget.
    pub budget_satisfied: bool,
}

/// Comprehensive report produced by `CognitiveLoadModel::evaluate` and the
/// analyzer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadReport {
    /// Per-stream (id, load) pairs.
    pub stream_loads: Vec<(usize, f64)>,
    /// Composed aggregate load.
    pub composed_load: f64,
    /// Budget ceiling.
    pub budget: f64,
    /// Whether aggregate ≤ budget.
    pub within_budget: bool,
    /// load / budget ratio.
    pub utilization: f64,
    /// budget − load (clamped ≥ 0).
    pub headroom: f64,
}

/// Optimizer that suggests how to reduce or re-allocate streams.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveLoadOptimizer {
    pub budget: CognitiveLoadBudget,
    model: CognitiveLoadModel,
}

impl CognitiveLoadOptimizer {
    pub fn new(budget: CognitiveLoadBudget) -> Self {
        let model = CognitiveLoadModel::new(budget.clone());
        Self { budget, model }
    }

    /// When total load exceeds budget, produce a plan that removes the
    /// lowest-priority streams first, then considers merges.
    pub fn suggest_stream_reduction(
        &self,
        streams: &[StreamDescriptor],
    ) -> StreamReductionPlan {
        let mut indexed: Vec<(usize, f64, f64)> = streams
            .iter()
            .map(|s| (s.id, self.model.stream_load(s), s.priority))
            .collect();

        // Sort by priority ascending (lowest priority removed first).
        indexed.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        let mut keep: Vec<usize> = Vec::new();
        let mut remove: Vec<usize> = Vec::new();
        let mut kept_loads: Vec<f64> = Vec::new();

        // Greedily add streams from highest priority down until budget reached.
        let high_first: Vec<_> = indexed.iter().rev().cloned().collect();
        for (id, load, _pri) in &high_first {
            let mut trial = kept_loads.clone();
            trial.push(*load);
            let composed = self.model.compose_loads(&trial);
            if composed <= self.budget.max_load && keep.len() < self.budget.max_streams {
                keep.push(*id);
                kept_loads.push(*load);
            } else {
                remove.push(*id);
            }
        }

        // Identify merge candidates among kept streams.
        let merge = self.merge_candidates_from_indexed(streams, &keep);

        let resulting_load = self.model.compose_loads(&kept_loads);
        let budget_satisfied = resulting_load <= self.budget.max_load;

        StreamReductionPlan {
            keep,
            remove,
            merge,
            resulting_load,
            budget_satisfied,
        }
    }

    /// Rank streams by an efficiency metric: priority × information_content / load.
    pub fn prioritize_streams(
        &self,
        streams: &[StreamDescriptor],
    ) -> Vec<(usize, f64)> {
        let mut ranked: Vec<(usize, f64)> = streams
            .iter()
            .map(|s| {
                let load = self.model.stream_load(s).max(0.01);
                let info = s.information_rate_bits_per_sec.max(0.01);
                let efficiency = s.priority * info / load;
                (s.id, efficiency)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Allocate an attention budget proportionally to each stream's priority.
    /// The returned weights sum to 1.0.
    pub fn optimal_attention_allocation(
        &self,
        streams: &[StreamDescriptor],
    ) -> Vec<f64> {
        if streams.is_empty() {
            return Vec::new();
        }
        let total_priority: f64 = streams.iter().map(|s| s.priority.max(0.01)).sum();
        if total_priority <= 0.0 {
            let uniform = 1.0 / streams.len() as f64;
            return vec![uniform; streams.len()];
        }
        streams
            .iter()
            .map(|s| s.priority.max(0.01) / total_priority)
            .collect()
    }

    /// Identify pairs of streams that are candidates for merging.
    ///
    /// Two streams are merge candidates when:
    ///   - both have priority < 0.5, AND
    ///   - their spectral_complexity difference < 0.3, AND
    ///   - their update_rate ratio is < 2×.
    pub fn merge_candidates(
        &self,
        streams: &[StreamDescriptor],
    ) -> Vec<(usize, usize)> {
        let mut candidates = Vec::new();
        for i in 0..streams.len() {
            for j in (i + 1)..streams.len() {
                let a = &streams[i];
                let b = &streams[j];
                let both_low = a.priority < 0.5 && b.priority < 0.5;
                let spectral_close =
                    (a.spectral_complexity - b.spectral_complexity).abs() < 0.3;
                let rate_a = a.update_rate_hz.max(0.01);
                let rate_b = b.update_rate_hz.max(0.01);
                let rate_ratio = (rate_a / rate_b).max(rate_b / rate_a);
                let rate_close = rate_ratio < 2.0;
                if both_low && spectral_close && rate_close {
                    candidates.push((a.id, b.id));
                }
            }
        }
        candidates
    }

    /// Estimate the load that would result from merging two streams.
    pub fn merged_stream_load(
        &self,
        a: &StreamDescriptor,
        b: &StreamDescriptor,
    ) -> f64 {
        let merged_complexity = ((a.stream_complexity + b.stream_complexity) / 2.0 + 0.05)
            .clamp(0.0, 1.0);
        let merged_info = a.information_rate_bits_per_sec + b.information_rate_bits_per_sec;
        let merged = StreamDescriptor::new(
            0,
            format!("{}+{}", a.name, b.name),
            merged_info,
            merged_complexity,
            a.update_rate_hz.max(b.update_rate_hz),
            ((a.spectral_complexity + b.spectral_complexity) / 2.0).clamp(0.0, 1.0),
            (a.temporal_regularity + b.temporal_regularity) / 2.0,
            (a.familiarity + b.familiarity) / 2.0,
            a.priority.max(b.priority),
        );
        self.model.stream_load(&merged)
    }

    // -- internal helpers ---------------------------------------------------

    fn merge_candidates_from_indexed(
        &self,
        streams: &[StreamDescriptor],
        keep_ids: &[usize],
    ) -> Vec<(usize, usize)> {
        let kept_streams: Vec<&StreamDescriptor> = streams
            .iter()
            .filter(|s| keep_ids.contains(&s.id))
            .collect();
        let mut out = Vec::new();
        for i in 0..kept_streams.len() {
            for j in (i + 1)..kept_streams.len() {
                let a = kept_streams[i];
                let b = kept_streams[j];
                let both_low = a.priority < 0.5 && b.priority < 0.5;
                let spectral_close =
                    (a.spectral_complexity - b.spectral_complexity).abs() < 0.3;
                if both_low && spectral_close {
                    out.push((a.id, b.id));
                }
            }
        }
        out
    }

    /// Summary statistics about a set of streams.
    pub fn stream_statistics(&self, streams: &[StreamDescriptor]) -> StreamStatistics {
        if streams.is_empty() {
            return StreamStatistics {
                count: 0,
                mean_load: 0.0,
                max_load: 0.0,
                min_load: 0.0,
                total_info_rate: 0.0,
                mean_priority: 0.0,
            };
        }
        let loads: Vec<f64> = streams.iter().map(|s| self.model.stream_load(s)).collect();
        let n = loads.len() as f64;
        StreamStatistics {
            count: streams.len(),
            mean_load: loads.iter().sum::<f64>() / n,
            max_load: loads.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            min_load: loads.iter().cloned().fold(f64::INFINITY, f64::min),
            total_info_rate: streams.iter().map(|s| s.information_rate_bits_per_sec).sum(),
            mean_priority: streams.iter().map(|s| s.priority).sum::<f64>() / n,
        }
    }
}

/// Descriptive statistics for a collection of streams.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamStatistics {
    pub count: usize,
    pub mean_load: f64,
    pub max_load: f64,
    pub min_load: f64,
    pub total_info_rate: f64,
    pub mean_priority: f64,
}

// ---------------------------------------------------------------------------
// 6. CognitiveLoadAnalyzer – high-level façade
// ---------------------------------------------------------------------------

/// Combines the load model and optimizer into a single analysis entry point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveLoadAnalyzer {
    model: CognitiveLoadModel,
    optimizer: CognitiveLoadOptimizer,
    working_memory: WorkingMemoryModel,
}

impl CognitiveLoadAnalyzer {
    pub fn new(budget: CognitiveLoadBudget) -> Self {
        let model = CognitiveLoadModel::new(budget.clone());
        let optimizer = CognitiveLoadOptimizer::new(budget);
        let working_memory = WorkingMemoryModel::new();
        Self {
            model,
            optimizer,
            working_memory,
        }
    }

    /// Produce a full load report for the given streams.
    pub fn analyze(&self, streams: &[StreamDescriptor]) -> LoadReport {
        self.model.evaluate(streams)
    }

    /// Generate human-readable improvement suggestions.
    pub fn suggest_improvements(&self, streams: &[StreamDescriptor]) -> Vec<String> {
        let mut suggestions: Vec<String> = Vec::new();
        let report = self.analyze(streams);

        // 1. Budget check
        if !report.within_budget {
            suggestions.push(format!(
                "Total cognitive load ({:.2}) exceeds budget ({:.2}). \
                 Consider removing or merging streams.",
                report.composed_load, report.budget
            ));

            let plan = self.optimizer.suggest_stream_reduction(streams);
            if !plan.remove.is_empty() {
                let names: Vec<String> = plan
                    .remove
                    .iter()
                    .filter_map(|id| streams.iter().find(|s| s.id == *id))
                    .map(|s| format!("\"{}\" (id={})", s.name, s.id))
                    .collect();
                suggestions.push(format!(
                    "Suggested removals (lowest priority first): {}",
                    names.join(", ")
                ));
            }
            if !plan.merge.is_empty() {
                for (a, b) in &plan.merge {
                    let na = streams
                        .iter()
                        .find(|s| s.id == *a)
                        .map(|s| s.name.as_str())
                        .unwrap_or("?");
                    let nb = streams
                        .iter()
                        .find(|s| s.id == *b)
                        .map(|s| s.name.as_str())
                        .unwrap_or("?");
                    suggestions.push(format!(
                        "Consider merging streams \"{}\" and \"{}\".",
                        na, nb
                    ));
                }
            }
        }

        // 2. Stream count vs working-memory capacity
        let avg_complexity: f64 = if streams.is_empty() {
            0.0
        } else {
            streams.iter().map(|s| s.stream_complexity).sum::<f64>() / streams.len() as f64
        };
        let wm_cap = self.working_memory.capacity_estimate(avg_complexity);
        if streams.len() > wm_cap {
            suggestions.push(format!(
                "Stream count ({}) exceeds estimated working-memory capacity ({}) \
                 at average complexity {:.2}.",
                streams.len(),
                wm_cap,
                avg_complexity
            ));
        }

        // 3. Attention-switching overhead
        let switch_cost = self.working_memory.attention_switching_cost(streams.len());
        if switch_cost > 0.15 {
            suggestions.push(format!(
                "Attention-switching overhead is high ({:.3}s). \
                 Reducing stream count will help.",
                switch_cost
            ));
        }

        // 4. Per-stream notes
        for s in streams {
            let load = self.model.stream_load(s);
            if load > 0.4 {
                suggestions.push(format!(
                    "Stream \"{}\" (id={}) has a high individual load ({:.2}). \
                     Increasing its regularity or familiarity may help.",
                    s.name, s.id, load
                ));
            }
            if s.information_rate_bits_per_sec > self.model.reference_info_rate * 0.8 {
                suggestions.push(format!(
                    "Stream \"{}\" (id={}) has a very high information rate ({:.1} bits/s). \
                     Consider down-sampling or summarising.",
                    s.name, s.id, s.information_rate_bits_per_sec
                ));
            }
        }

        // 5. Information-rate budget
        let total_info: f64 = streams
            .iter()
            .map(|s| s.information_rate_bits_per_sec)
            .sum();
        if total_info > self.model.budget.max_information_rate {
            suggestions.push(format!(
                "Total information rate ({:.1} bits/s) exceeds tracking limit ({:.1} bits/s). \
                 Reduce update rates or aggregate information.",
                total_info, self.model.budget.max_information_rate
            ));
        }

        // 6. Utilisation note if load is very low
        if report.within_budget && report.utilization < 0.3 && !streams.is_empty() {
            suggestions.push(format!(
                "Budget utilisation is only {:.0}%. You could convey more information.",
                report.utilization * 100.0
            ));
        }

        suggestions
    }

    /// Return the underlying model for direct access.
    pub fn model(&self) -> &CognitiveLoadModel {
        &self.model
    }

    /// Return the underlying optimizer for direct access.
    pub fn optimizer(&self) -> &CognitiveLoadOptimizer {
        &self.optimizer
    }

    /// Return the working-memory model.
    pub fn working_memory(&self) -> &WorkingMemoryModel {
        &self.working_memory
    }

    /// Quick check: can we safely add one more stream?
    pub fn can_add_stream(
        &self,
        current_streams: &[StreamDescriptor],
        candidate: &StreamDescriptor,
    ) -> bool {
        let mut all = current_streams.to_vec();
        all.push(candidate.clone());
        let report = self.analyze(&all);
        report.within_budget && all.len() <= self.model.budget.max_streams
    }

    /// Compute a "sonification quality score" (0-1) that balances information
    /// conveyance against cognitive overload.
    pub fn quality_score(&self, streams: &[StreamDescriptor]) -> f64 {
        if streams.is_empty() {
            return 0.0;
        }
        let report = self.analyze(streams);
        let info_coverage = {
            let total_info: f64 = streams
                .iter()
                .map(|s| s.information_rate_bits_per_sec)
                .sum();
            (total_info / self.model.budget.max_information_rate).min(1.0)
        };
        let load_penalty = if report.within_budget {
            1.0 - 0.3 * report.utilization
        } else {
            0.2 / report.utilization.max(0.01)
        };
        let priority_coverage: f64 = {
            let total_pri: f64 = streams.iter().map(|s| s.priority).sum();
            let max_possible = streams.len() as f64;
            if max_possible > 0.0 {
                total_pri / max_possible
            } else {
                0.0
            }
        };
        (0.4 * info_coverage + 0.4 * load_penalty + 0.2 * priority_coverage).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// 7. Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a minimal stream.
    fn make_stream(
        id: usize,
        complexity: f64,
        priority: f64,
        info_rate: f64,
        familiarity: f64,
        regularity: f64,
    ) -> StreamDescriptor {
        StreamDescriptor::new(
            id,
            format!("stream_{}", id),
            info_rate,
            complexity,
            2.0,
            complexity,
            regularity,
            familiarity,
            priority,
        )
    }

    fn default_model() -> CognitiveLoadModel {
        CognitiveLoadModel::with_default_budget()
    }

    // -----------------------------------------------------------------------
    // Test 1: Single simple stream load is reasonable (0 < load < 1)
    // -----------------------------------------------------------------------
    #[test]
    fn test_single_stream_load_reasonable() {
        let model = default_model();
        let s = make_stream(0, 0.5, 0.5, 10.0, 0.5, 0.5);
        let load = model.stream_load(&s);
        assert!(load > 0.0, "load should be positive, got {}", load);
        assert!(load < 1.0, "load should be < 1, got {}", load);
    }

    // -----------------------------------------------------------------------
    // Test 2: Composed loads exceed the simple sum (interference)
    // -----------------------------------------------------------------------
    #[test]
    fn test_compose_loads_exceeds_sum() {
        let model = default_model();
        let loads = vec![0.2, 0.2, 0.2];
        let simple_sum: f64 = loads.iter().sum();
        let composed = model.compose_loads(&loads);
        assert!(
            composed > simple_sum,
            "composed ({}) should exceed sum ({})",
            composed,
            simple_sum
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: Identity element of LoadComposition::compose
    // -----------------------------------------------------------------------
    #[test]
    fn test_compose_identity() {
        let id = LoadComposition::identity();
        assert_eq!(id, 0.0);
        let val = 0.42;
        let result = LoadComposition::compose(val, id);
        assert!(
            (result - val).abs() < 1e-10,
            "compose(x, 0) should equal x, got {}",
            result
        );
        assert!(LoadComposition::verify_identity(val));
    }

    // -----------------------------------------------------------------------
    // Test 4: Budget checking — under budget
    // -----------------------------------------------------------------------
    #[test]
    fn test_budget_under() {
        let model = default_model();
        assert!(model.check_budget(0.5));
        assert!(model.check_budget(1.0));
    }

    // -----------------------------------------------------------------------
    // Test 5: Budget checking — over budget
    // -----------------------------------------------------------------------
    #[test]
    fn test_budget_over() {
        let model = default_model();
        assert!(!model.check_budget(1.01));
        assert!(!model.check_budget(2.0));
    }

    // -----------------------------------------------------------------------
    // Test 6: Working-memory capacity decreases with complexity
    // -----------------------------------------------------------------------
    #[test]
    fn test_capacity_decreases_with_complexity() {
        let wm = WorkingMemoryModel::new();
        let cap_low = wm.capacity_estimate(0.1);
        let cap_high = wm.capacity_estimate(0.9);
        assert!(
            cap_low > cap_high,
            "low complexity cap ({}) should > high complexity cap ({})",
            cap_low,
            cap_high
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: Attention switching cost increases with stream count
    // -----------------------------------------------------------------------
    #[test]
    fn test_switching_cost_increases() {
        let wm = WorkingMemoryModel::new();
        let cost_2 = wm.attention_switching_cost(2);
        let cost_5 = wm.attention_switching_cost(5);
        assert!(
            cost_5 > cost_2,
            "cost_5 ({}) should > cost_2 ({})",
            cost_5,
            cost_2
        );
        assert_eq!(wm.attention_switching_cost(1), 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 8: Temporal chunking reduces effective load
    // -----------------------------------------------------------------------
    #[test]
    fn test_chunking_reduces_load() {
        let wm = WorkingMemoryModel::new();
        let s = make_stream(0, 0.5, 0.5, 10.0, 0.5, 0.8); // high regularity
        let load_0 = wm.effective_load(&s, 0.0);
        let load_30 = wm.effective_load(&s, 30.0);
        assert!(
            load_30 < load_0,
            "30s exposure ({}) should reduce load vs 0s ({})",
            load_30,
            load_0
        );
    }

    // -----------------------------------------------------------------------
    // Test 9: Optimizer reduces streams when over budget
    // -----------------------------------------------------------------------
    #[test]
    fn test_optimizer_reduces_streams() {
        let budget = CognitiveLoadBudget::new(4, 1.0, 40.0);
        let opt = CognitiveLoadOptimizer::new(budget);
        let streams: Vec<StreamDescriptor> = (0..8)
            .map(|i| make_stream(i, 0.6, (i as f64) / 8.0, 15.0, 0.2, 0.3))
            .collect();
        let plan = opt.suggest_stream_reduction(&streams);
        assert!(
            plan.keep.len() <= 4,
            "should keep ≤4 streams, kept {}",
            plan.keep.len()
        );
        assert!(
            !plan.remove.is_empty(),
            "should have removed at least one stream"
        );
        assert!(
            plan.budget_satisfied,
            "resulting plan should satisfy budget"
        );
    }

    // -----------------------------------------------------------------------
    // Test 10: Attention allocation sums to 1.0
    // -----------------------------------------------------------------------
    #[test]
    fn test_attention_allocation_sums_to_one() {
        let budget = CognitiveLoadBudget::new(4, 1.0, 40.0);
        let opt = CognitiveLoadOptimizer::new(budget);
        let streams: Vec<StreamDescriptor> = (0..4)
            .map(|i| make_stream(i, 0.4, 0.3 + 0.2 * i as f64, 10.0, 0.5, 0.5))
            .collect();
        let alloc = opt.optimal_attention_allocation(&streams);
        let sum: f64 = alloc.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "allocations should sum to 1.0, got {}",
            sum
        );
    }

    // -----------------------------------------------------------------------
    // Test 11: Prioritisation respects priority weights
    // -----------------------------------------------------------------------
    #[test]
    fn test_prioritisation_respects_weights() {
        let budget = CognitiveLoadBudget::new(4, 1.0, 40.0);
        let opt = CognitiveLoadOptimizer::new(budget);
        let s_high = make_stream(0, 0.3, 0.9, 10.0, 0.5, 0.5);
        let s_low = make_stream(1, 0.3, 0.1, 10.0, 0.5, 0.5);
        let ranked = opt.prioritize_streams(&[s_high, s_low]);
        assert_eq!(
            ranked[0].0, 0,
            "high-priority stream should rank first"
        );
    }

    // -----------------------------------------------------------------------
    // Test 12: Full analysis report is internally consistent
    // -----------------------------------------------------------------------
    #[test]
    fn test_full_analysis_consistent() {
        let budget = CognitiveLoadBudget::new(4, 1.0, 40.0);
        let analyzer = CognitiveLoadAnalyzer::new(budget);
        let streams: Vec<StreamDescriptor> = (0..3)
            .map(|i| make_stream(i, 0.3, 0.5, 8.0, 0.5, 0.5))
            .collect();
        let report = analyzer.analyze(&streams);

        // Utilisation should be composed_load / budget
        let expected_util = report.composed_load / report.budget;
        assert!(
            (report.utilization - expected_util).abs() < 1e-10,
            "utilization mismatch: {} vs {}",
            report.utilization,
            expected_util
        );

        // Headroom should be budget - composed_load
        let expected_headroom = (report.budget - report.composed_load).max(0.0);
        assert!(
            (report.headroom - expected_headroom).abs() < 1e-10,
            "headroom mismatch"
        );

        // within_budget flag
        assert_eq!(
            report.within_budget,
            report.composed_load <= report.budget
        );

        // Per-stream loads should all be positive
        for (_, load) in &report.stream_loads {
            assert!(*load > 0.0, "all stream loads should be positive");
        }
    }

    // -----------------------------------------------------------------------
    // Test 13: Commutativity of LoadComposition::compose
    // -----------------------------------------------------------------------
    #[test]
    fn test_compose_commutativity() {
        assert!(LoadComposition::verify_commutativity(0.3, 0.5));
        assert!(LoadComposition::verify_commutativity(0.0, 0.7));
    }

    // -----------------------------------------------------------------------
    // Test 14: compose_many matches sequential fold
    // -----------------------------------------------------------------------
    #[test]
    fn test_compose_many_consistency() {
        let loads = vec![0.1, 0.2, 0.15, 0.25];
        let result = LoadComposition::compose_many(&loads);
        let manual = loads
            .iter()
            .copied()
            .fold(0.0, |acc, l| LoadComposition::compose(acc, l));
        assert!(
            (result - manual).abs() < 1e-10,
            "compose_many ({}) should match fold ({})",
            result,
            manual
        );
    }

    // -----------------------------------------------------------------------
    // Test 15: remaining_budget and max_additional_streams agreement
    // -----------------------------------------------------------------------
    #[test]
    fn test_remaining_budget_positive_when_under() {
        let model = default_model();
        let loads = vec![0.1, 0.15];
        let total = model.compose_loads(&loads);
        let remaining = model.remaining_budget(total);
        assert!(remaining > 0.0);
        let max_add = model.max_additional_streams(&loads);
        assert!(max_add >= 1, "should allow at least one more stream");
    }

    // -----------------------------------------------------------------------
    // Test 16: Analyzer suggestions are non-empty when overloaded
    // -----------------------------------------------------------------------
    #[test]
    fn test_suggestions_when_overloaded() {
        let budget = CognitiveLoadBudget::new(4, 1.0, 40.0);
        let analyzer = CognitiveLoadAnalyzer::new(budget);
        let streams: Vec<StreamDescriptor> = (0..8)
            .map(|i| make_stream(i, 0.7, 0.5, 20.0, 0.1, 0.1))
            .collect();
        let suggestions = analyzer.suggest_improvements(&streams);
        assert!(
            !suggestions.is_empty(),
            "should have suggestions for 8 complex streams"
        );
    }

    // -----------------------------------------------------------------------
    // Test 17: Quality score in valid range
    // -----------------------------------------------------------------------
    #[test]
    fn test_quality_score_range() {
        let budget = CognitiveLoadBudget::new(4, 1.0, 40.0);
        let analyzer = CognitiveLoadAnalyzer::new(budget);
        let streams: Vec<StreamDescriptor> = (0..3)
            .map(|i| make_stream(i, 0.4, 0.6, 10.0, 0.5, 0.5))
            .collect();
        let score = analyzer.quality_score(&streams);
        assert!(score >= 0.0 && score <= 1.0, "score {} out of range", score);
    }

    // -----------------------------------------------------------------------
    // Test 18: Empty streams edge case
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_streams() {
        let model = default_model();
        assert_eq!(model.compose_loads(&[]), 0.0);
        let report = model.evaluate(&[]);
        assert!(report.within_budget);
        assert_eq!(report.composed_load, 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 19: Familiarity reduces load
    // -----------------------------------------------------------------------
    #[test]
    fn test_familiarity_reduces_load() {
        let model = default_model();
        let novel = make_stream(0, 0.5, 0.5, 10.0, 0.0, 0.5);
        let familiar = make_stream(1, 0.5, 0.5, 10.0, 1.0, 0.5);
        assert!(
            model.stream_load(&familiar) < model.stream_load(&novel),
            "familiar stream should have lower load"
        );
    }

    // -----------------------------------------------------------------------
    // Test 20: can_add_stream respects budget
    // -----------------------------------------------------------------------
    #[test]
    fn test_can_add_stream() {
        let budget = CognitiveLoadBudget::new(2, 1.0, 40.0);
        let analyzer = CognitiveLoadAnalyzer::new(budget);
        let s1 = make_stream(0, 0.3, 0.5, 8.0, 0.5, 0.5);
        let s2 = make_stream(1, 0.3, 0.5, 8.0, 0.5, 0.5);
        let s3 = make_stream(2, 0.3, 0.5, 8.0, 0.5, 0.5);
        assert!(analyzer.can_add_stream(&[s1.clone()], &s2));
        // With max_streams=2, adding a third should fail
        assert!(!analyzer.can_add_stream(&[s1, s2], &s3));
    }
}
