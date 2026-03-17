//! Counterexample-Guided Abstraction Refinement (CEGAR).
//!
//! When BMC at a bounded depth is inconclusive, CEGAR abstracts the PTA,
//! checks the abstract model, and refines the abstraction when spurious
//! counterexamples are found.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::{
    AtomicPredicate, Edge, LocationId, LocationKind, ModelCheckerError,
    PTA, Predicate, Result, SafetyProperty, SafetyPropertyKind, Update,
    Variable, VariableId, VariableKind, Verdict,
};
use crate::bounded_checker::{BmcConfig, BoundedModelChecker, CheckResult};
use crate::counterexample::{CounterExample, CounterexampleValidator, TraceStep};

// ---------------------------------------------------------------------------
// CegarConfig
// ---------------------------------------------------------------------------

/// Configuration for the CEGAR loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CegarConfig {
    /// Maximum refinement iterations.
    pub max_refinements: usize,
    /// Abstraction strategy to use.
    pub abstraction_strategy: AbstractionStrategyKind,
    /// Overall timeout in seconds.
    pub timeout_secs: f64,
    /// BMC bound for abstract model checking.
    pub abstract_bmc_bound: usize,
    /// Whether to attempt minimisation of abstract counterexamples.
    pub minimize_counterexamples: bool,
    /// Maximum number of predicates in the abstract domain.
    pub max_predicates: usize,
}

impl Default for CegarConfig {
    fn default() -> Self {
        Self {
            max_refinements: 20,
            abstraction_strategy: AbstractionStrategyKind::LocationSplit,
            timeout_secs: 120.0,
            abstract_bmc_bound: 30,
            minimize_counterexamples: true,
            max_predicates: 50,
        }
    }
}

/// Which abstraction strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AbstractionStrategyKind {
    LocationSplit,
    PredicateRefinement,
}

// ---------------------------------------------------------------------------
// CegarResult
// ---------------------------------------------------------------------------

/// Result of a CEGAR run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CegarResult {
    pub verdict: Verdict,
    pub iterations: usize,
    pub final_abstraction: AbstractionInfo,
    pub counterexample: Option<CounterExample>,
    pub stats: CegarStatistics,
}

/// Information about the final abstraction state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractionInfo {
    pub num_abstract_locations: usize,
    pub num_abstract_edges: usize,
    pub num_predicates: usize,
    pub refinement_history: Vec<RefinementRecord>,
}

/// Record of a single refinement step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementRecord {
    pub iteration: usize,
    pub strategy: String,
    pub spurious_length: usize,
    pub predicates_added: usize,
    pub locations_split: usize,
}

// ---------------------------------------------------------------------------
// CegarStatistics
// ---------------------------------------------------------------------------

/// Statistics from the CEGAR loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CegarStatistics {
    pub total_time_secs: f64,
    pub abstract_check_time_secs: f64,
    pub concretisation_time_secs: f64,
    pub refinement_time_secs: f64,
    pub num_spurious: usize,
    pub num_genuine: usize,
    pub peak_abstract_states: usize,
}

impl CegarStatistics {
    pub fn new() -> Self {
        Self {
            total_time_secs: 0.0,
            abstract_check_time_secs: 0.0,
            concretisation_time_secs: 0.0,
            refinement_time_secs: 0.0,
            num_spurious: 0,
            num_genuine: 0,
            peak_abstract_states: 0,
        }
    }
}

impl Default for CegarStatistics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AbstractionState
// ---------------------------------------------------------------------------

/// Tracks the current abstraction of the PTA.
#[derive(Debug, Clone)]
pub struct AbstractionState {
    /// Abstract locations: each maps to a set of concrete locations.
    pub location_partition: Vec<HashSet<LocationId>>,
    /// Predicates used for predicate abstraction.
    pub predicates: Vec<AbstractionPredicate>,
    /// Number of refinement iterations applied.
    pub refinement_count: usize,
    /// Variable precision: how many decimal digits are tracked.
    pub variable_precision: HashMap<VariableId, usize>,
}

/// A predicate used in the abstraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractionPredicate {
    pub id: usize,
    pub description: String,
    pub predicate: Predicate,
}

impl AbstractionState {
    /// Create the initial (coarsest) abstraction from a PTA.
    pub fn initial(pta: &PTA) -> Self {
        // Start with each concrete location as its own abstract location.
        let location_partition: Vec<HashSet<LocationId>> = pta
            .locations
            .iter()
            .map(|l| {
                let mut set = HashSet::new();
                set.insert(l.id);
                set
            })
            .collect();

        let variable_precision: HashMap<VariableId, usize> = pta
            .variables
            .iter()
            .map(|v| (v.id, 1)) // start with coarse precision
            .collect();

        Self {
            location_partition,
            predicates: Vec::new(),
            refinement_count: 0,
            variable_precision,
        }
    }

    /// Number of abstract locations.
    pub fn num_abstract_locations(&self) -> usize {
        self.location_partition.len()
    }

    /// Find which abstract location contains a concrete location.
    pub fn abstract_location(&self, concrete: LocationId) -> Option<usize> {
        self.location_partition
            .iter()
            .position(|part| part.contains(&concrete))
    }

    /// Apply the abstraction to a PTA, returning an abstract PTA.
    pub fn abstract_pta(&self, pta: &PTA) -> PTA {
        let mut abstract_pta = pta.clone();

        // Merge locations in the same partition.
        if self.location_partition.len() < pta.locations.len() {
            // Simplified: just use the original PTA with refined variable bounds.
            for var in &mut abstract_pta.variables {
                if let Some(&precision) = self.variable_precision.get(&var.id) {
                    let scale = 10.0_f64.powi(precision as i32);
                    var.lower_bound = (var.lower_bound * scale).floor() / scale;
                    var.upper_bound = (var.upper_bound * scale).ceil() / scale;
                }
            }
        }

        abstract_pta
    }

    /// Get abstraction info for reporting.
    pub fn info(&self, history: Vec<RefinementRecord>) -> AbstractionInfo {
        AbstractionInfo {
            num_abstract_locations: self.location_partition.len(),
            num_abstract_edges: 0, // computed lazily
            num_predicates: self.predicates.len(),
            refinement_history: history,
        }
    }
}

// ---------------------------------------------------------------------------
// RefinementStrategy trait
// ---------------------------------------------------------------------------

/// A strategy for refining the abstraction when a spurious counterexample is
/// found.
pub trait RefinementStrategy: fmt::Debug {
    /// Refine the abstraction to eliminate the spurious counterexample.
    fn refine(
        &self,
        state: &mut AbstractionState,
        pta: &PTA,
        spurious_cx: &CounterExample,
        property: &SafetyProperty,
    ) -> RefinementRecord;

    /// Name of this strategy.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// LocationSplitStrategy
// ---------------------------------------------------------------------------

/// Refines by splitting abstract locations that are too coarse.
#[derive(Debug, Clone)]
pub struct LocationSplitStrategy;

impl LocationSplitStrategy {
    pub fn new() -> Self {
        Self
    }

    /// Identify which abstract location to split based on the spurious trace.
    fn find_split_point(
        &self,
        state: &AbstractionState,
        pta: &PTA,
        cx: &CounterExample,
    ) -> Option<(usize, LocationId)> {
        // Walk the trace backwards from the violation step.
        for step in cx.steps.iter().rev() {
            let abstract_loc = state.abstract_location(step.location);
            if let Some(abs_loc) = abstract_loc {
                if state.location_partition[abs_loc].len() > 1 {
                    return Some((abs_loc, step.location));
                }
            }
        }

        // If no multi-location partition found, try splitting based on
        // the divergence point.
        if cx.steps.len() >= 2 {
            let mid = cx.violation_step / 2;
            let loc = cx.steps.get(mid).map(|s| s.location).unwrap_or(0);
            let abs_loc = state.abstract_location(loc).unwrap_or(0);
            return Some((abs_loc, loc));
        }

        None
    }
}

impl Default for LocationSplitStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl RefinementStrategy for LocationSplitStrategy {
    fn refine(
        &self,
        state: &mut AbstractionState,
        pta: &PTA,
        spurious_cx: &CounterExample,
        _property: &SafetyProperty,
    ) -> RefinementRecord {
        let mut locations_split = 0;

        if let Some((abs_loc, concrete_loc)) = self.find_split_point(state, pta, spurious_cx) {
            // Split the abstract location: keep concrete_loc in its own
            // partition, move the rest.
            if abs_loc < state.location_partition.len() {
                let original = state.location_partition[abs_loc].clone();
                if original.len() > 1 {
                    let mut remaining = original;
                    remaining.remove(&concrete_loc);

                    state.location_partition[abs_loc] = {
                        let mut s = HashSet::new();
                        s.insert(concrete_loc);
                        s
                    };
                    state.location_partition.push(remaining);
                    locations_split = 1;
                }
            }
        }

        // Also increase variable precision.
        for var_id in state.variable_precision.keys().cloned().collect::<Vec<_>>() {
            let prec = state.variable_precision.entry(var_id).or_insert(1);
            *prec = (*prec + 1).min(6);
        }

        state.refinement_count += 1;

        RefinementRecord {
            iteration: state.refinement_count,
            strategy: "LocationSplit".into(),
            spurious_length: spurious_cx.len(),
            predicates_added: 0,
            locations_split,
        }
    }

    fn name(&self) -> &str {
        "LocationSplit"
    }
}

// ---------------------------------------------------------------------------
// PredicateRefinementStrategy
// ---------------------------------------------------------------------------

/// Refines by adding predicates that distinguish spurious from genuine
/// counterexamples.
#[derive(Debug, Clone)]
pub struct PredicateRefinementStrategy {
    max_predicates: usize,
}

impl PredicateRefinementStrategy {
    pub fn new(max_predicates: usize) -> Self {
        Self { max_predicates }
    }

    /// Extract a distinguishing predicate from the spurious trace.
    fn extract_predicate(
        &self,
        pta: &PTA,
        cx: &CounterExample,
        property: &SafetyProperty,
    ) -> Option<AbstractionPredicate> {
        // Find the first step where the abstract trace diverges from
        // concrete feasibility.
        for step in &cx.steps {
            // Check if any variable bound is violated.
            for (idx, var) in pta.variables.iter().enumerate() {
                let val = step.variable_values.get(idx).copied().unwrap_or(0.0);
                if val < var.lower_bound || val > var.upper_bound {
                    // This variable value is infeasible → create predicate.
                    let mid = (var.lower_bound + var.upper_bound) / 2.0;
                    return Some(AbstractionPredicate {
                        id: 0, // assigned later
                        description: format!("{} ≤ {:.2}", var.name, mid),
                        predicate: Predicate::from_conjuncts(vec![
                            AtomicPredicate::VarLeq { var: var.id, bound: mid },
                        ]),
                    });
                }
            }

            // Check concentration bounds from the property.
            match &property.kind {
                SafetyPropertyKind::ConcentrationBound { drug, max_concentration } => {
                    for (idx, var) in pta.variables.iter().enumerate() {
                        if var.kind == VariableKind::Concentration {
                            let val = step.variable_values.get(idx).copied().unwrap_or(0.0);
                            if val > *max_concentration * 0.8 {
                                let threshold = max_concentration * 0.9;
                                return Some(AbstractionPredicate {
                                    id: 0,
                                    description: format!("{} ≤ {:.2}", var.name, threshold),
                                    predicate: Predicate::from_conjuncts(vec![
                                        AtomicPredicate::VarLeq {
                                            var: var.id,
                                            bound: threshold,
                                        },
                                    ]),
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        None
    }
}

impl Default for PredicateRefinementStrategy {
    fn default() -> Self {
        Self::new(50)
    }
}

impl RefinementStrategy for PredicateRefinementStrategy {
    fn refine(
        &self,
        state: &mut AbstractionState,
        pta: &PTA,
        spurious_cx: &CounterExample,
        property: &SafetyProperty,
    ) -> RefinementRecord {
        let mut predicates_added = 0;

        if state.predicates.len() < self.max_predicates {
            if let Some(mut pred) = self.extract_predicate(pta, spurious_cx, property) {
                pred.id = state.predicates.len();
                state.predicates.push(pred);
                predicates_added = 1;
            }
        }

        state.refinement_count += 1;

        RefinementRecord {
            iteration: state.refinement_count,
            strategy: "PredicateRefinement".into(),
            spurious_length: spurious_cx.len(),
            predicates_added,
            locations_split: 0,
        }
    }

    fn name(&self) -> &str {
        "PredicateRefinement"
    }
}

// ---------------------------------------------------------------------------
// Spuriousness check
// ---------------------------------------------------------------------------

/// Check whether an abstract counterexample is spurious by trying to
/// simulate it on the concrete PTA.
fn is_spurious(cx: &CounterExample, pta: &PTA) -> bool {
    let validator = CounterexampleValidator::new(1e-4);

    // Quick path check.
    if !validator.validate_path(cx, pta) {
        return true;
    }

    // Full validation.
    let result = validator.validate(cx, pta);
    !result.valid
}

// ---------------------------------------------------------------------------
// CegarEngine
// ---------------------------------------------------------------------------

/// The main CEGAR engine.
#[derive(Debug)]
pub struct CegarEngine {
    config: CegarConfig,
    bmc: BoundedModelChecker,
}

impl CegarEngine {
    pub fn new(config: CegarConfig) -> Self {
        let bmc_config = BmcConfig {
            max_bound: config.abstract_bmc_bound,
            global_timeout_secs: config.timeout_secs / 2.0,
            ..BmcConfig::default()
        };
        Self {
            config,
            bmc: BoundedModelChecker::new(bmc_config),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(CegarConfig::default())
    }

    /// Run the full CEGAR loop.
    pub fn run_cegar(
        &self,
        pta: &PTA,
        property: &SafetyProperty,
        config: &CegarConfig,
    ) -> Result<CegarResult> {
        let start = Instant::now();
        let mut stats = CegarStatistics::new();
        let mut abstraction = AbstractionState::initial(pta);
        let mut refinement_history: Vec<RefinementRecord> = Vec::new();

        let strategy: Box<dyn RefinementStrategy> = match config.abstraction_strategy {
            AbstractionStrategyKind::LocationSplit => Box::new(LocationSplitStrategy::new()),
            AbstractionStrategyKind::PredicateRefinement => {
                Box::new(PredicateRefinementStrategy::new(config.max_predicates))
            }
        };

        for iteration in 0..config.max_refinements {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > config.timeout_secs {
                stats.total_time_secs = elapsed;
                return Ok(CegarResult {
                    verdict: Verdict::Unknown,
                    iterations: iteration,
                    final_abstraction: abstraction.info(refinement_history),
                    counterexample: None,
                    stats,
                });
            }

            // Step 1: Build abstract PTA.
            let abstract_pta = abstraction.abstract_pta(pta);
            stats.peak_abstract_states = stats
                .peak_abstract_states
                .max(abstraction.num_abstract_locations());

            // Step 2: Check the abstract PTA.
            let check_start = Instant::now();
            let check_result = self.bmc.check(
                &abstract_pta,
                property,
                config.abstract_bmc_bound,
            )?;
            stats.abstract_check_time_secs += check_start.elapsed().as_secs_f64();

            match check_result.verdict {
                Verdict::Safe => {
                    // Abstract model is safe → concrete model is safe.
                    stats.total_time_secs = start.elapsed().as_secs_f64();
                    return Ok(CegarResult {
                        verdict: Verdict::Safe,
                        iterations: iteration + 1,
                        final_abstraction: abstraction.info(refinement_history),
                        counterexample: None,
                        stats,
                    });
                }
                Verdict::Unsafe => {
                    // Got a counterexample from the abstract model.
                    let abstract_cx = check_result.counterexample.unwrap_or_else(|| {
                        CounterExample::empty(property.id.clone())
                    });

                    // Step 3: Check if counterexample is genuine.
                    let conc_start = Instant::now();
                    let spurious = is_spurious(&abstract_cx, pta);
                    stats.concretisation_time_secs += conc_start.elapsed().as_secs_f64();

                    if !spurious {
                        // Genuine counterexample found.
                        stats.num_genuine += 1;
                        stats.total_time_secs = start.elapsed().as_secs_f64();
                        return Ok(CegarResult {
                            verdict: Verdict::Unsafe,
                            iterations: iteration + 1,
                            final_abstraction: abstraction.info(refinement_history),
                            counterexample: Some(abstract_cx),
                            stats,
                        });
                    }

                    // Step 4: Spurious → refine.
                    stats.num_spurious += 1;
                    let refine_start = Instant::now();
                    let record = strategy.refine(
                        &mut abstraction,
                        pta,
                        &abstract_cx,
                        property,
                    );
                    stats.refinement_time_secs += refine_start.elapsed().as_secs_f64();
                    refinement_history.push(record);
                }
                Verdict::Unknown => {
                    // BMC inconclusive at this abstraction level.
                    stats.total_time_secs = start.elapsed().as_secs_f64();
                    return Ok(CegarResult {
                        verdict: Verdict::Unknown,
                        iterations: iteration + 1,
                        final_abstraction: abstraction.info(refinement_history),
                        counterexample: None,
                        stats,
                    });
                }
            }
        }

        // Exhausted refinement budget.
        stats.total_time_secs = start.elapsed().as_secs_f64();
        Ok(CegarResult {
            verdict: Verdict::Unknown,
            iterations: config.max_refinements,
            final_abstraction: abstraction.info(refinement_history),
            counterexample: None,
            stats,
        })
    }

    /// Get the config.
    pub fn config(&self) -> &CegarConfig {
        &self.config
    }
}

impl Default for CegarEngine {
    fn default() -> Self {
        Self::with_default_config()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{make_test_pta, DrugId, SafetyProperty};

    #[test]
    fn test_cegar_config_default() {
        let config = CegarConfig::default();
        assert_eq!(config.max_refinements, 20);
        assert!(config.timeout_secs > 0.0);
    }

    #[test]
    fn test_abstraction_state_initial() {
        let pta = make_test_pta("drug", 500.0, false);
        let state = AbstractionState::initial(&pta);
        assert_eq!(state.num_abstract_locations(), pta.num_locations());
        assert_eq!(state.refinement_count, 0);
    }

    #[test]
    fn test_abstract_location_lookup() {
        let pta = make_test_pta("drug", 500.0, false);
        let state = AbstractionState::initial(&pta);
        assert_eq!(state.abstract_location(0), Some(0));
        assert_eq!(state.abstract_location(1), Some(1));
    }

    #[test]
    fn test_abstract_pta() {
        let pta = make_test_pta("drug", 500.0, false);
        let state = AbstractionState::initial(&pta);
        let abstract_pta = state.abstract_pta(&pta);
        assert_eq!(abstract_pta.num_locations(), pta.num_locations());
    }

    #[test]
    fn test_location_split_strategy() {
        let pta = make_test_pta("drug", 500.0, true);
        let mut state = AbstractionState::initial(&pta);
        let strategy = LocationSplitStrategy::new();
        let cx = CounterExample::empty("test".into());
        let prop = SafetyProperty::no_error();
        let record = strategy.refine(&mut state, &pta, &cx, &prop);
        assert_eq!(record.iteration, 1);
        assert_eq!(record.strategy, "LocationSplit");
    }

    #[test]
    fn test_predicate_refinement_strategy() {
        let pta = make_test_pta("drug", 500.0, false);
        let mut state = AbstractionState::initial(&pta);
        let strategy = PredicateRefinementStrategy::new(10);
        let cx = CounterExample::empty("test".into());
        let prop = SafetyProperty::concentration_bound(DrugId::new("drug"), 5.0);
        let record = strategy.refine(&mut state, &pta, &cx, &prop);
        assert_eq!(record.strategy, "PredicateRefinement");
    }

    #[test]
    fn test_cegar_safe_pta() {
        let pta = make_test_pta("aspirin", 100.0, false);
        let prop = SafetyProperty::no_error();
        let config = CegarConfig {
            max_refinements: 5,
            abstract_bmc_bound: 10,
            timeout_secs: 10.0,
            ..CegarConfig::default()
        };
        let engine = CegarEngine::new(config.clone());
        let result = engine.run_cegar(&pta, &prop, &config).unwrap();
        assert_eq!(result.verdict, Verdict::Safe);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_cegar_statistics() {
        let stats = CegarStatistics::new();
        assert_eq!(stats.num_spurious, 0);
        assert_eq!(stats.num_genuine, 0);
    }

    #[test]
    fn test_cegar_result_fields() {
        let pta = make_test_pta("drug", 100.0, false);
        let prop = SafetyProperty::no_error();
        let config = CegarConfig {
            max_refinements: 3,
            abstract_bmc_bound: 5,
            timeout_secs: 5.0,
            ..CegarConfig::default()
        };
        let engine = CegarEngine::new(config.clone());
        let result = engine.run_cegar(&pta, &prop, &config).unwrap();
        assert!(result.stats.total_time_secs >= 0.0);
        assert!(result.final_abstraction.num_abstract_locations > 0);
    }

    #[test]
    fn test_abstraction_info() {
        let pta = make_test_pta("drug", 100.0, false);
        let state = AbstractionState::initial(&pta);
        let info = state.info(vec![]);
        assert!(info.num_abstract_locations > 0);
        assert!(info.refinement_history.is_empty());
    }

    #[test]
    fn test_refinement_record_fields() {
        let record = RefinementRecord {
            iteration: 1,
            strategy: "LocationSplit".into(),
            spurious_length: 5,
            predicates_added: 0,
            locations_split: 1,
        };
        assert_eq!(record.iteration, 1);
        assert_eq!(record.locations_split, 1);
    }

    #[test]
    fn test_abstraction_predicate() {
        let pred = AbstractionPredicate {
            id: 0,
            description: "x ≤ 5.0".into(),
            predicate: Predicate::from_conjuncts(vec![
                AtomicPredicate::VarLeq { var: 0, bound: 5.0 },
            ]),
        };
        assert_eq!(pred.id, 0);
        assert!(pred.predicate.evaluate(&[], &[3.0]));
        assert!(!pred.predicate.evaluate(&[], &[7.0]));
    }

    #[test]
    fn test_is_spurious_empty_cx() {
        let pta = make_test_pta("drug", 100.0, false);
        let cx = CounterExample::empty("test".into());
        assert!(is_spurious(&cx, &pta));
    }
}
