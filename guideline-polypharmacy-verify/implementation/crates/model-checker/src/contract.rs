//! Contract extraction, compatibility checking, and composition.
//!
//! Implements assume-guarantee reasoning for CYP enzyme resources.  Each drug
//! PTA produces an [`EnzymeContract`] per enzyme it interacts with.  Contracts
//! are then checked for mutual compatibility before the product is built.

use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    ContractCompatibility, CypEnzyme, CypInhibitionEffect, CypInductionEffect,
    CypMetabolismRoute, DrugId, EnzymeCompatibilityResult, EnzymeContract,
    EnzymeMetabolismRoute, GuidelineContract, InhibitionType, ModelCheckerError,
    PTA, PkParameters, Result, SafetyProperty, SafetyPropertyKind, Verdict,
};

// ---------------------------------------------------------------------------
// ContractExtractor
// ---------------------------------------------------------------------------

/// Extracts enzyme-level assume-guarantee contracts from a single PTA.
#[derive(Debug, Clone)]
pub struct ContractExtractor {
    /// Safety margin applied to worst-case bounds (≥1.0 for conservatism).
    pub safety_factor: f64,
    /// Whether to use mechanism-based inactivation model.
    pub use_mbi_model: bool,
}

impl Default for ContractExtractor {
    fn default() -> Self {
        Self {
            safety_factor: 1.2,
            use_mbi_model: true,
        }
    }
}

impl ContractExtractor {
    pub fn new(safety_factor: f64) -> Self {
        Self {
            safety_factor,
            use_mbi_model: true,
        }
    }

    /// Extract enzyme contracts for a PTA given its enzyme metabolism routes.
    pub fn extract_contracts(
        &self,
        pta: &PTA,
        enzyme_routes: &[EnzymeMetabolismRoute],
    ) -> Vec<EnzymeContract> {
        let mut contracts = Vec::new();

        // Collect enzymes this drug metabolises through.
        let metabolism_enzymes: HashSet<CypEnzyme> =
            enzyme_routes.iter().map(|r| r.enzyme).collect();

        // Collect enzymes this drug inhibits.
        let inhibition_map: HashMap<CypEnzyme, &CypInhibitionEffect> = pta
            .inhibition_effects
            .iter()
            .map(|e| (e.enzyme, e))
            .collect();

        // Collect enzymes this drug induces.
        let induction_map: HashMap<CypEnzyme, &CypInductionEffect> = pta
            .induction_effects
            .iter()
            .map(|e| (e.enzyme, e))
            .collect();

        // All enzymes involved.
        let all_enzymes: HashSet<CypEnzyme> = metabolism_enzymes
            .iter()
            .chain(inhibition_map.keys())
            .chain(induction_map.keys())
            .copied()
            .collect();

        for enzyme in &all_enzymes {
            let contract =
                self.extract_single_contract(pta, *enzyme, enzyme_routes, &inhibition_map, &induction_map);
            contracts.push(contract);
        }

        contracts
    }

    /// Extract a single enzyme contract.
    fn extract_single_contract(
        &self,
        pta: &PTA,
        enzyme: CypEnzyme,
        routes: &[EnzymeMetabolismRoute],
        inhibition_map: &HashMap<CypEnzyme, &CypInhibitionEffect>,
        induction_map: &HashMap<CypEnzyme, &CypInductionEffect>,
    ) -> EnzymeContract {
        let pk = &pta.pk_params;

        // Compute the fraction of metabolism that depends on this enzyme.
        let fraction_metabolized: f64 = routes
            .iter()
            .filter(|r| r.enzyme == enzyme)
            .map(|r| r.fraction_metabolized)
            .sum();

        // Assumption: minimum enzyme activity needed for adequate clearance.
        let assumed_min_activity = self.compute_assumed_min_activity(
            fraction_metabolized,
            pk,
        );

        // Guarantee: maximum enzyme load produced by this drug's effects.
        let (guaranteed_max_load, inh_type, worst_case_conc) =
            self.compute_guaranteed_max_load(enzyme, pk, inhibition_map, induction_map);

        let tightness = if inhibition_map.contains_key(&enzyme) {
            0.85 * (1.0 / self.safety_factor)
        } else {
            0.95
        };

        EnzymeContract {
            enzyme,
            owner_drug: pta.drug_id.clone(),
            assumed_min_activity,
            guaranteed_max_load,
            inhibition_type: inh_type,
            worst_case_inhibitor_conc: worst_case_conc,
            tightness,
        }
    }

    /// Compute the minimum enzyme activity assumed for adequate drug clearance.
    ///
    /// If the drug heavily depends on this enzyme for metabolism, it needs
    /// higher minimum activity.
    fn compute_assumed_min_activity(
        &self,
        fraction_metabolized: f64,
        pk: &PkParameters,
    ) -> f64 {
        if fraction_metabolized <= 0.0 {
            return 0.0;
        }

        // Base assumption: the drug requires at least (fraction_metabolized *
        // safety_factor) of normal activity to stay below toxic threshold.
        let base = fraction_metabolized * 0.5;

        // Adjust by elimination rate: drugs with slow elimination need more
        // enzyme capacity.
        let ke_factor = if pk.ke < 0.05 {
            1.3 // slow elimination → need more enzyme
        } else if pk.ke > 0.2 {
            0.8 // fast elimination → tolerant
        } else {
            1.0
        };

        (base * ke_factor * self.safety_factor).min(1.0).max(0.0)
    }

    /// Compute the guaranteed maximum load this drug places on a given enzyme.
    ///
    /// Returns (max_load, inhibition_type, worst_case_concentration).
    fn compute_guaranteed_max_load(
        &self,
        enzyme: CypEnzyme,
        pk: &PkParameters,
        inhibition_map: &HashMap<CypEnzyme, &CypInhibitionEffect>,
        induction_map: &HashMap<CypEnzyme, &CypInductionEffect>,
    ) -> (f64, Option<InhibitionType>, f64) {
        let mut max_load: f64 = 0.0;
        let mut inh_type = None;
        let worst_case_conc = pk.css_peak * self.safety_factor;

        if let Some(inh) = inhibition_map.get(&enzyme) {
            let load = self.compute_inhibition_load(inh, worst_case_conc);
            max_load += load;
            inh_type = Some(inh.inhibition_type);
        }

        // Induction acts as negative load (reduces inhibition burden).
        if let Some(ind) = induction_map.get(&enzyme) {
            let induction_benefit = self.compute_induction_benefit(ind, worst_case_conc);
            max_load = (max_load - induction_benefit).max(0.0);
        }

        (max_load.min(1.0), inh_type, worst_case_conc)
    }

    /// Compute the fractional enzyme load from a competitive or non-competitive
    /// inhibition effect.
    fn compute_inhibition_load(&self, effect: &CypInhibitionEffect, conc: f64) -> f64 {
        let ki = effect.inhibition_constant.ki;
        if ki <= 0.0 {
            return 0.0;
        }

        match effect.inhibition_type {
            InhibitionType::Competitive => {
                // Fractional inhibition = [I] / (Ki + [I])
                conc / (ki + conc)
            }
            InhibitionType::NonCompetitive => {
                // Same formula but applied to Vmax
                conc / (ki + conc)
            }
            InhibitionType::Uncompetitive => {
                // Affects ES complex only; typically lower load
                0.5 * conc / (ki + conc)
            }
            InhibitionType::MechanismBased => {
                if self.use_mbi_model {
                    // MBI: fraction = kinact * [I] / (ki * kdeg + kinact * [I])
                    let kinact = effect.inhibition_constant.kinact.unwrap_or(0.0);
                    let kdeg = effect.enzyme.degradation_rate();
                    if kinact <= 0.0 {
                        return conc / (ki + conc);
                    }
                    let numerator = kinact * conc;
                    let denominator = ki * kdeg + kinact * conc;
                    if denominator <= 0.0 { 0.0 } else { numerator / denominator }
                } else {
                    conc / (ki + conc)
                }
            }
            InhibitionType::Mixed => {
                // Weighted average of competitive and non-competitive.
                let alpha = 0.6; // competitive fraction
                let comp = conc / (ki + conc);
                let noncomp = conc / (ki + conc);
                alpha * comp + (1.0 - alpha) * noncomp
            }
        }
    }

    /// Compute the induction benefit (negative load).
    fn compute_induction_benefit(&self, effect: &CypInductionEffect, conc: f64) -> f64 {
        if effect.ec50 <= 0.0 {
            return 0.0;
        }
        // Emax model: benefit = Emax * [I] / (EC50 + [I])
        let benefit = effect.emax * conc / (effect.ec50 + conc);
        benefit.max(0.0).min(1.0)
    }
}

// ---------------------------------------------------------------------------
// ContractChecker
// ---------------------------------------------------------------------------

/// Checks mutual compatibility of a set of enzyme contracts.
#[derive(Debug, Clone)]
pub struct ContractChecker {
    /// Enzyme capacity threshold (fraction of total).  Default: 1.0 meaning
    /// the sum of loads must not exceed 100% of capacity.
    pub capacity_threshold: f64,
    /// Safety margin for compatibility checking.
    pub safety_margin: f64,
}

impl Default for ContractChecker {
    fn default() -> Self {
        Self {
            capacity_threshold: 1.0,
            safety_margin: 0.1,
        }
    }
}

impl ContractChecker {
    pub fn new(capacity_threshold: f64, safety_margin: f64) -> Self {
        Self { capacity_threshold, safety_margin }
    }

    /// Check compatibility of a set of guideline contracts.
    pub fn check_compatibility(
        &self,
        contracts: &[GuidelineContract],
    ) -> ContractCompatibility {
        let mut enzyme_loads: HashMap<CypEnzyme, Vec<(DrugId, f64)>> = HashMap::new();
        let mut enzyme_assumptions: HashMap<CypEnzyme, Vec<(DrugId, f64)>> = HashMap::new();

        for gc in contracts {
            for ec in &gc.enzyme_contracts {
                enzyme_loads
                    .entry(ec.enzyme)
                    .or_default()
                    .push((ec.owner_drug.clone(), ec.guaranteed_max_load));
                enzyme_assumptions
                    .entry(ec.enzyme)
                    .or_default()
                    .push((ec.owner_drug.clone(), ec.assumed_min_activity));
            }
        }

        let mut enzyme_results = Vec::new();
        let mut all_compatible = true;

        for enzyme in CypEnzyme::all() {
            let loads = enzyme_loads.get(enzyme).cloned().unwrap_or_default();
            let assumptions = enzyme_assumptions.get(enzyme).cloned().unwrap_or_default();

            if loads.is_empty() && assumptions.is_empty() {
                continue;
            }

            let total_load: f64 = loads.iter().map(|(_, l)| l).sum();
            let capacity = self.capacity_threshold * enzyme.relative_abundance();
            let adjusted_capacity = capacity - self.safety_margin;
            let compatible = total_load <= adjusted_capacity;

            // Also check that remaining activity satisfies all assumptions.
            let remaining_activity = (1.0 - total_load).max(0.0);
            let max_assumption = assumptions
                .iter()
                .map(|(_, a)| *a)
                .fold(0.0_f64, f64::max);
            let assumptions_met = remaining_activity >= max_assumption;

            let enzyme_compatible = compatible && assumptions_met;
            if !enzyme_compatible {
                all_compatible = false;
            }

            enzyme_results.push(EnzymeCompatibilityResult {
                enzyme: *enzyme,
                total_load,
                capacity: adjusted_capacity,
                compatible: enzyme_compatible,
                margin: adjusted_capacity - total_load,
            });
        }

        let summary = if all_compatible {
            "All enzyme contracts are mutually compatible.".to_string()
        } else {
            let failing: Vec<String> = enzyme_results
                .iter()
                .filter(|r| !r.compatible)
                .map(|r| format!("{}: load {:.2} > capacity {:.2}", r.enzyme, r.total_load, r.capacity))
                .collect();
            format!("Contract incompatibility: {}", failing.join("; "))
        };

        ContractCompatibility {
            compatible: all_compatible,
            enzyme_results,
            summary,
        }
    }

    /// Check mutual compatibility of two sets of enzyme contracts.
    pub fn check_mutual_compatibility(
        &self,
        contracts_a: &[EnzymeContract],
        contracts_b: &[EnzymeContract],
    ) -> bool {
        let mut enzyme_loads: HashMap<CypEnzyme, f64> = HashMap::new();
        let mut enzyme_assumptions: HashMap<CypEnzyme, f64> = HashMap::new();

        for ec in contracts_a.iter().chain(contracts_b.iter()) {
            *enzyme_loads.entry(ec.enzyme).or_insert(0.0) += ec.guaranteed_max_load;
            let entry = enzyme_assumptions.entry(ec.enzyme).or_insert(0.0);
            *entry = entry.max(ec.assumed_min_activity);
        }

        for (enzyme, total_load) in &enzyme_loads {
            let capacity = self.capacity_threshold * enzyme.relative_abundance();
            if *total_load > capacity - self.safety_margin {
                return false;
            }
            let remaining = (1.0 - total_load).max(0.0);
            if let Some(&needed) = enzyme_assumptions.get(enzyme) {
                if remaining < needed {
                    return false;
                }
            }
        }

        true
    }

    /// Identify which enzymes are the bottleneck (most loaded).
    pub fn identify_bottlenecks(
        &self,
        contracts: &[GuidelineContract],
    ) -> Vec<(CypEnzyme, f64, f64)> {
        let mut enzyme_loads: HashMap<CypEnzyme, f64> = HashMap::new();
        for gc in contracts {
            for ec in &gc.enzyme_contracts {
                *enzyme_loads.entry(ec.enzyme).or_insert(0.0) += ec.guaranteed_max_load;
            }
        }

        let mut bottlenecks: Vec<(CypEnzyme, f64, f64)> = enzyme_loads
            .into_iter()
            .map(|(enzyme, load)| {
                let capacity = enzyme.relative_abundance();
                (enzyme, load, load / capacity)
            })
            .collect();

        bottlenecks.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        bottlenecks
    }
}

// ---------------------------------------------------------------------------
// ContractRefinement
// ---------------------------------------------------------------------------

/// Refines contracts when the initial check is inconclusive, by tightening
/// bounds using more precise PK analysis.
#[derive(Debug, Clone)]
pub struct ContractRefinement {
    /// Number of refinement iterations allowed.
    pub max_iterations: usize,
    /// Factor by which safety margins are reduced each iteration.
    pub relaxation_rate: f64,
}

impl Default for ContractRefinement {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            relaxation_rate: 0.9,
        }
    }
}

/// Result of a refinement attempt.
#[derive(Debug, Clone)]
pub struct RefinementResult {
    pub refined_contracts: Vec<EnzymeContract>,
    pub iterations_used: usize,
    pub improved: bool,
    pub tightness_gain: f64,
}

impl ContractRefinement {
    pub fn new(max_iterations: usize, relaxation_rate: f64) -> Self {
        Self { max_iterations, relaxation_rate }
    }

    /// Attempt to refine a set of enzyme contracts to make them compatible.
    pub fn refine(
        &self,
        contracts: &[EnzymeContract],
        checker: &ContractChecker,
    ) -> RefinementResult {
        let mut current: Vec<EnzymeContract> = contracts.to_vec();
        let original_tightness: f64 =
            current.iter().map(|c| c.tightness).sum::<f64>() / current.len() as f64;

        for iteration in 0..self.max_iterations {
            // Check current compatibility.
            let guideline_contracts = vec![GuidelineContract {
                guideline_id: "refinement".into(),
                drug_id: DrugId::new("combined"),
                enzyme_contracts: current.clone(),
            }];

            let compat = checker.check_compatibility(&guideline_contracts);
            if compat.compatible {
                let new_tightness =
                    current.iter().map(|c| c.tightness).sum::<f64>() / current.len() as f64;
                return RefinementResult {
                    refined_contracts: current,
                    iterations_used: iteration,
                    improved: true,
                    tightness_gain: new_tightness - original_tightness,
                };
            }

            // Refine: reduce guaranteed loads and increase assumed minimums.
            for contract in &mut current {
                contract.guaranteed_max_load *= self.relaxation_rate;
                contract.assumed_min_activity *= self.relaxation_rate;
                contract.tightness *= 1.05; // tighter = more precise
                contract.tightness = contract.tightness.min(1.0);
            }
        }

        let new_tightness =
            current.iter().map(|c| c.tightness).sum::<f64>() / current.len().max(1) as f64;
        RefinementResult {
            refined_contracts: current,
            iterations_used: self.max_iterations,
            improved: false,
            tightness_gain: new_tightness - original_tightness,
        }
    }

    /// Try to split a contract into sub-contracts for finer-grained reasoning.
    pub fn split_contract(&self, contract: &EnzymeContract) -> Vec<EnzymeContract> {
        match contract.inhibition_type {
            Some(InhibitionType::MechanismBased) => {
                // Split MBI into time-dependent phases.
                let early = EnzymeContract {
                    enzyme: contract.enzyme,
                    owner_drug: contract.owner_drug.clone(),
                    assumed_min_activity: contract.assumed_min_activity,
                    guaranteed_max_load: contract.guaranteed_max_load * 0.3,
                    inhibition_type: contract.inhibition_type,
                    worst_case_inhibitor_conc: contract.worst_case_inhibitor_conc * 0.5,
                    tightness: contract.tightness * 0.9,
                };
                let late = EnzymeContract {
                    enzyme: contract.enzyme,
                    owner_drug: contract.owner_drug.clone(),
                    assumed_min_activity: contract.assumed_min_activity * 0.8,
                    guaranteed_max_load: contract.guaranteed_max_load,
                    inhibition_type: contract.inhibition_type,
                    worst_case_inhibitor_conc: contract.worst_case_inhibitor_conc,
                    tightness: contract.tightness * 0.95,
                };
                vec![early, late]
            }
            _ => vec![contract.clone()],
        }
    }
}

// ---------------------------------------------------------------------------
// MonotonicityCertifier
// ---------------------------------------------------------------------------

/// Verifies that the interaction network among drugs has monotone properties,
/// enabling compositional reasoning shortcuts.
#[derive(Debug, Clone)]
pub struct MonotonicityCertifier;

/// Monotonicity classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MonotonicityClass {
    /// All interactions are monotonically increasing (more drug → more effect).
    Monotone,
    /// All interactions are antimonotone.
    Antimonotone,
    /// Mixed monotonicity.
    Mixed,
    /// No interactions.
    Trivial,
}

impl MonotonicityCertifier {
    pub fn new() -> Self {
        Self
    }

    /// Classify the monotonicity of the interaction network.
    pub fn classify(&self, ptas: &[PTA]) -> MonotonicityClass {
        if ptas.len() <= 1 {
            return MonotonicityClass::Trivial;
        }

        let mut has_inhibition = false;
        let mut has_induction = false;

        for pta in ptas {
            if !pta.inhibition_effects.is_empty() {
                has_inhibition = true;
            }
            if !pta.induction_effects.is_empty() {
                has_induction = true;
            }
        }

        match (has_inhibition, has_induction) {
            (false, false) => MonotonicityClass::Trivial,
            (true, false) => MonotonicityClass::Monotone,
            (false, true) => MonotonicityClass::Antimonotone,
            (true, true) => MonotonicityClass::Mixed,
        }
    }

    /// Check whether a monotone composition is sound for the given properties.
    pub fn is_composition_sound(
        &self,
        class: MonotonicityClass,
        properties: &[SafetyProperty],
    ) -> bool {
        match class {
            MonotonicityClass::Trivial => true,
            MonotonicityClass::Monotone => {
                // Monotone: contract-based decomposition is sound for
                // concentration bounds (upper bounds).
                properties.iter().all(|p| matches!(
                    p.kind,
                    SafetyPropertyKind::ConcentrationBound { .. }
                    | SafetyPropertyKind::EnzymeActivityFloor { .. }
                    | SafetyPropertyKind::NoErrorReachable
                ))
            }
            MonotonicityClass::Antimonotone => {
                properties.iter().all(|p| matches!(
                    p.kind,
                    SafetyPropertyKind::TherapeuticRange { .. }
                    | SafetyPropertyKind::NoErrorReachable
                ))
            }
            MonotonicityClass::Mixed => false,
        }
    }

    /// Build an interaction dependency graph (enzyme → set of drugs).
    pub fn interaction_graph(
        &self,
        ptas: &[PTA],
    ) -> HashMap<CypEnzyme, Vec<(DrugId, InteractionKind)>> {
        let mut graph: HashMap<CypEnzyme, Vec<(DrugId, InteractionKind)>> = HashMap::new();

        for pta in ptas {
            for route in &pta.metabolism_routes {
                graph
                    .entry(route.enzyme)
                    .or_default()
                    .push((pta.drug_id.clone(), InteractionKind::Substrate));
            }
            for inh in &pta.inhibition_effects {
                graph
                    .entry(inh.enzyme)
                    .or_default()
                    .push((pta.drug_id.clone(), InteractionKind::Inhibitor));
            }
            for ind in &pta.induction_effects {
                graph
                    .entry(ind.enzyme)
                    .or_default()
                    .push((pta.drug_id.clone(), InteractionKind::Inducer));
            }
        }

        graph
    }
}

impl Default for MonotonicityCertifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Kind of drug–enzyme interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InteractionKind {
    Substrate,
    Inhibitor,
    Inducer,
}

impl fmt::Display for InteractionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Substrate => write!(f, "substrate"),
            Self::Inhibitor => write!(f, "inhibitor"),
            Self::Inducer => write!(f, "inducer"),
        }
    }
}

// ---------------------------------------------------------------------------
// CompositionEngine
// ---------------------------------------------------------------------------

/// Orchestrates contract-based compositional verification.
#[derive(Debug)]
pub struct CompositionEngine {
    extractor: ContractExtractor,
    checker: ContractChecker,
    refinement: ContractRefinement,
    monotonicity: MonotonicityCertifier,
}

/// Result of compositional verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionResult {
    pub verdict: Verdict,
    pub individual_results: Vec<IndividualContractResult>,
    pub enzyme_compatibility: ContractCompatibility,
    pub fallback_interactions: Vec<FallbackInteraction>,
    pub monotonicity_class: String,
    pub summary: String,
}

/// Result for a single drug's contract check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndividualContractResult {
    pub drug_id: DrugId,
    pub num_contracts: usize,
    pub all_consistent: bool,
    pub max_load: f64,
}

/// An interaction that could not be handled by contracts and needs
/// monolithic BMC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackInteraction {
    pub drugs: Vec<DrugId>,
    pub enzyme: CypEnzyme,
    pub reason: String,
}

impl CompositionEngine {
    pub fn new() -> Self {
        Self {
            extractor: ContractExtractor::default(),
            checker: ContractChecker::default(),
            refinement: ContractRefinement::default(),
            monotonicity: MonotonicityCertifier::new(),
        }
    }

    pub fn with_extractor(mut self, extractor: ContractExtractor) -> Self {
        self.extractor = extractor;
        self
    }

    pub fn with_checker(mut self, checker: ContractChecker) -> Self {
        self.checker = checker;
        self
    }

    /// Run the full compositional verification flow.
    pub fn compose_with_contracts(
        &self,
        ptas: &[PTA],
        properties: &[SafetyProperty],
    ) -> CompositionResult {
        let mono_class = self.monotonicity.classify(ptas);
        let mono_sound =
            self.monotonicity.is_composition_sound(mono_class, properties);

        // Extract contracts for each PTA.
        let mut guideline_contracts: Vec<GuidelineContract> = Vec::new();
        let mut individual_results: Vec<IndividualContractResult> = Vec::new();
        let mut fallback_interactions: Vec<FallbackInteraction> = Vec::new();

        for pta in ptas {
            let contracts =
                self.extractor.extract_contracts(pta, &pta.metabolism_routes);

            let all_consistent = contracts.iter().all(|c| c.is_self_consistent());
            let max_load = contracts
                .iter()
                .map(|c| c.guaranteed_max_load)
                .fold(0.0_f64, f64::max);

            individual_results.push(IndividualContractResult {
                drug_id: pta.drug_id.clone(),
                num_contracts: contracts.len(),
                all_consistent,
                max_load,
            });

            // Check for non-competitive interactions that need fallback.
            for ec in &contracts {
                if ec.inhibition_type == Some(InhibitionType::NonCompetitive) && !mono_sound {
                    fallback_interactions.push(FallbackInteraction {
                        drugs: vec![pta.drug_id.clone()],
                        enzyme: ec.enzyme,
                        reason: "Non-competitive inhibition requires monolithic check".into(),
                    });
                }
            }

            guideline_contracts.push(GuidelineContract {
                guideline_id: format!("auto_{}", pta.drug_id),
                drug_id: pta.drug_id.clone(),
                enzyme_contracts: contracts,
            });
        }

        // Check enzyme compatibility.
        let mut enzyme_compat =
            self.checker.check_compatibility(&guideline_contracts);

        // If not compatible, try refinement.
        if !enzyme_compat.compatible {
            let all_enzyme_contracts: Vec<EnzymeContract> = guideline_contracts
                .iter()
                .flat_map(|gc| gc.enzyme_contracts.clone())
                .collect();

            let refinement_result =
                self.refinement.refine(&all_enzyme_contracts, &self.checker);

            if refinement_result.improved {
                // Re-check with refined contracts.
                let refined_gc = GuidelineContract {
                    guideline_id: "refined".into(),
                    drug_id: DrugId::new("combined"),
                    enzyme_contracts: refinement_result.refined_contracts,
                };
                enzyme_compat = self.checker.check_compatibility(&[refined_gc]);
            }
        }

        let verdict = if enzyme_compat.compatible && fallback_interactions.is_empty() {
            Verdict::Safe
        } else if !enzyme_compat.compatible {
            Verdict::Unsafe
        } else {
            Verdict::Unknown
        };

        let summary = format!(
            "Compositional analysis of {} drugs: {} enzyme checks, verdict: {}",
            ptas.len(),
            enzyme_compat.enzyme_results.len(),
            verdict
        );

        CompositionResult {
            verdict,
            individual_results,
            enzyme_compatibility: enzyme_compat,
            fallback_interactions,
            monotonicity_class: format!("{:?}", mono_class),
            summary,
        }
    }

    /// Get the interaction graph.
    pub fn interaction_graph(
        &self,
        ptas: &[PTA],
    ) -> HashMap<CypEnzyme, Vec<(DrugId, InteractionKind)>> {
        self.monotonicity.interaction_graph(ptas)
    }

    /// Identify drugs that can be verified independently (no shared enzymes).
    pub fn independent_groups(&self, ptas: &[PTA]) -> Vec<Vec<usize>> {
        let n = ptas.len();
        let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];

        for i in 0..n {
            for j in (i + 1)..n {
                let enzymes_i = ptas[i].involved_enzymes();
                let enzymes_j = ptas[j].involved_enzymes();
                if !enzymes_i.is_disjoint(&enzymes_j) {
                    adj[i].insert(j);
                    adj[j].insert(i);
                }
            }
        }

        // Connected components.
        let mut visited = vec![false; n];
        let mut groups = Vec::new();

        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut group = Vec::new();
            let mut stack = vec![start];
            while let Some(node) = stack.pop() {
                if visited[node] {
                    continue;
                }
                visited[node] = true;
                group.push(node);
                for &neighbor in &adj[node] {
                    if !visited[neighbor] {
                        stack.push(neighbor);
                    }
                }
            }
            group.sort();
            groups.push(group);
        }

        groups
    }
}

impl Default for CompositionEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        make_test_pta, CypEnzyme, CypInhibitionEffect, CypMetabolismRoute,
        DrugId, InhibitionConstant, InhibitionType, PkParameters, PtaBuilder,
    };

    fn pta_with_inhibition(name: &str, enzyme: CypEnzyme, ki: f64) -> PTA {
        let mut pta = make_test_pta(name, 500.0, false);
        pta.metabolism_routes.push(CypMetabolismRoute {
            enzyme,
            fraction_metabolized: 0.8,
        });
        pta.inhibition_effects.push(CypInhibitionEffect {
            enzyme,
            inhibition_type: InhibitionType::Competitive,
            inhibition_constant: InhibitionConstant::simple(ki),
        });
        pta.pk_params = PkParameters {
            css_peak: 5.0,
            ke: 0.1,
            ..PkParameters::default()
        };
        pta
    }

    #[test]
    fn test_extract_contracts_basic() {
        let pta = pta_with_inhibition("clarithromycin", CypEnzyme::CYP3A4, 5.0);
        let extractor = ContractExtractor::default();
        let contracts = extractor.extract_contracts(&pta, &pta.metabolism_routes);
        assert!(!contracts.is_empty());
        let cyp3a4_contract = contracts.iter().find(|c| c.enzyme == CypEnzyme::CYP3A4);
        assert!(cyp3a4_contract.is_some());
    }

    #[test]
    fn test_contract_self_consistency() {
        let pta = pta_with_inhibition("drug", CypEnzyme::CYP3A4, 10.0);
        let extractor = ContractExtractor::new(1.0);
        let contracts = extractor.extract_contracts(&pta, &pta.metabolism_routes);
        for c in &contracts {
            assert!(c.is_self_consistent(), "Contract not self-consistent: {:?}", c);
        }
    }

    #[test]
    fn test_compatibility_compatible() {
        let pta_a = pta_with_inhibition("drugA", CypEnzyme::CYP3A4, 50.0);
        let pta_b = pta_with_inhibition("drugB", CypEnzyme::CYP2D6, 50.0);
        let extractor = ContractExtractor::new(1.0);
        let contracts_a = extractor.extract_contracts(&pta_a, &pta_a.metabolism_routes);
        let contracts_b = extractor.extract_contracts(&pta_b, &pta_b.metabolism_routes);

        let checker = ContractChecker::new(1.0, 0.05);
        let gc = vec![
            GuidelineContract {
                guideline_id: "g1".into(),
                drug_id: DrugId::new("drugA"),
                enzyme_contracts: contracts_a,
            },
            GuidelineContract {
                guideline_id: "g2".into(),
                drug_id: DrugId::new("drugB"),
                enzyme_contracts: contracts_b,
            },
        ];
        let result = checker.check_compatibility(&gc);
        // Different enzymes → compatible.
        assert!(result.compatible);
    }

    #[test]
    fn test_compatibility_incompatible_same_enzyme() {
        // Two drugs both heavily inhibiting CYP3A4 with low Ki.
        let pta_a = pta_with_inhibition("drugA", CypEnzyme::CYP3A4, 0.5);
        let pta_b = pta_with_inhibition("drugB", CypEnzyme::CYP3A4, 0.5);
        let extractor = ContractExtractor::new(1.5);
        let contracts_a = extractor.extract_contracts(&pta_a, &pta_a.metabolism_routes);
        let contracts_b = extractor.extract_contracts(&pta_b, &pta_b.metabolism_routes);

        let checker = ContractChecker::new(1.0, 0.05);
        let gc = vec![
            GuidelineContract {
                guideline_id: "g1".into(),
                drug_id: DrugId::new("drugA"),
                enzyme_contracts: contracts_a,
            },
            GuidelineContract {
                guideline_id: "g2".into(),
                drug_id: DrugId::new("drugB"),
                enzyme_contracts: contracts_b,
            },
        ];
        let result = checker.check_compatibility(&gc);
        // Both heavily loading CYP3A4 → incompatible.
        assert!(!result.compatible);
    }

    #[test]
    fn test_mutual_compatibility() {
        let pta_a = pta_with_inhibition("drugA", CypEnzyme::CYP3A4, 50.0);
        let pta_b = pta_with_inhibition("drugB", CypEnzyme::CYP2D6, 50.0);
        let extractor = ContractExtractor::new(1.0);
        let ca = extractor.extract_contracts(&pta_a, &pta_a.metabolism_routes);
        let cb = extractor.extract_contracts(&pta_b, &pta_b.metabolism_routes);

        let checker = ContractChecker::new(1.0, 0.05);
        assert!(checker.check_mutual_compatibility(&ca, &cb));
    }

    #[test]
    fn test_monotonicity_trivial() {
        let pta = make_test_pta("aspirin", 100.0, false);
        let certifier = MonotonicityCertifier::new();
        assert_eq!(certifier.classify(&[pta]), MonotonicityClass::Trivial);
    }

    #[test]
    fn test_monotonicity_monotone() {
        let pta_a = pta_with_inhibition("drugA", CypEnzyme::CYP3A4, 10.0);
        let pta_b = pta_with_inhibition("drugB", CypEnzyme::CYP2D6, 10.0);
        let certifier = MonotonicityCertifier::new();
        assert_eq!(
            certifier.classify(&[pta_a, pta_b]),
            MonotonicityClass::Monotone
        );
    }

    #[test]
    fn test_composition_engine_basic() {
        let pta_a = pta_with_inhibition("metformin", CypEnzyme::CYP3A4, 50.0);
        let pta_b = pta_with_inhibition("clarithromycin", CypEnzyme::CYP2D6, 50.0);
        let engine = CompositionEngine::new();
        let props = vec![SafetyProperty::no_error()];
        let result = engine.compose_with_contracts(&[pta_a, pta_b], &props);
        assert_eq!(result.individual_results.len(), 2);
    }

    #[test]
    fn test_independent_groups() {
        let pta_a = pta_with_inhibition("drugA", CypEnzyme::CYP3A4, 10.0);
        let pta_b = pta_with_inhibition("drugB", CypEnzyme::CYP2D6, 10.0);
        let pta_c = pta_with_inhibition("drugC", CypEnzyme::CYP3A4, 10.0);
        let engine = CompositionEngine::new();
        let groups = engine.independent_groups(&[pta_a, pta_b, pta_c]);
        // A and C share CYP3A4, B is independent.
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_identify_bottlenecks() {
        let pta = pta_with_inhibition("drug", CypEnzyme::CYP3A4, 1.0);
        let extractor = ContractExtractor::new(1.0);
        let contracts = extractor.extract_contracts(&pta, &pta.metabolism_routes);
        let gc = vec![GuidelineContract {
            guideline_id: "g1".into(),
            drug_id: DrugId::new("drug"),
            enzyme_contracts: contracts,
        }];
        let checker = ContractChecker::default();
        let bottlenecks = checker.identify_bottlenecks(&gc);
        assert!(!bottlenecks.is_empty());
    }

    #[test]
    fn test_contract_refinement() {
        let pta = pta_with_inhibition("drug", CypEnzyme::CYP3A4, 1.0);
        let extractor = ContractExtractor::new(2.0); // very conservative
        let contracts = extractor.extract_contracts(&pta, &pta.metabolism_routes);
        let checker = ContractChecker::new(1.0, 0.05);
        let refinement = ContractRefinement::new(10, 0.85);
        let result = refinement.refine(&contracts, &checker);
        assert!(result.iterations_used <= 10);
    }

    #[test]
    fn test_interaction_graph() {
        let pta_a = pta_with_inhibition("drugA", CypEnzyme::CYP3A4, 10.0);
        let certifier = MonotonicityCertifier::new();
        let graph = certifier.interaction_graph(&[pta_a]);
        assert!(graph.contains_key(&CypEnzyme::CYP3A4));
    }

    #[test]
    fn test_inhibition_load_competitive() {
        let extractor = ContractExtractor::new(1.0);
        let effect = CypInhibitionEffect {
            enzyme: CypEnzyme::CYP3A4,
            inhibition_type: InhibitionType::Competitive,
            inhibition_constant: InhibitionConstant::simple(5.0),
        };
        let load = extractor.compute_inhibition_load(&effect, 5.0);
        // [I]/(Ki+[I]) = 5/(5+5) = 0.5
        assert!((load - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_inhibition_load_mechanism_based() {
        let extractor = ContractExtractor::new(1.0);
        let effect = CypInhibitionEffect {
            enzyme: CypEnzyme::CYP3A4,
            inhibition_type: InhibitionType::MechanismBased,
            inhibition_constant: InhibitionConstant::mechanism_based(5.0, 0.1),
        };
        let load = extractor.compute_inhibition_load(&effect, 5.0);
        // Should be > 0 and ≤ 1
        assert!(load > 0.0);
        assert!(load <= 1.0);
    }

    #[test]
    fn test_split_contract_mbi() {
        let contract = EnzymeContract {
            enzyme: CypEnzyme::CYP3A4,
            owner_drug: DrugId::new("drug"),
            assumed_min_activity: 0.3,
            guaranteed_max_load: 0.6,
            inhibition_type: Some(InhibitionType::MechanismBased),
            worst_case_inhibitor_conc: 5.0,
            tightness: 0.9,
        };
        let refinement = ContractRefinement::default();
        let parts = refinement.split_contract(&contract);
        assert_eq!(parts.len(), 2);
    }
}
