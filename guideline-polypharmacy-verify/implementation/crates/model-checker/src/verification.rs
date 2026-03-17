//! Top-level verification orchestrator.
//!
//! Chains contract-based decomposition, bounded model checking, and CEGAR
//! into a single entry point.  Handles timeouts gracefully and reports
//! partial results.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::{
    CypEnzyme, DrugId, EnzymeContract, GuidelineContract, GuidelineDocument,
    ModelCheckerError, PTA, Result, SafetyProperty, SafetyPropertyKind,
    Verdict, VerificationConfig,
};
use crate::bounded_checker::{BmcConfig, BoundedModelChecker, CheckResult, MonolithicBMC};
use crate::cegar::{CegarConfig, CegarEngine, CegarResult};
use crate::contract::{CompositionEngine, CompositionResult, ContractChecker, ContractExtractor};
use crate::counterexample::CounterExample;
use crate::narrator::{ClinicalNarrative, ClinicalNarrator};
use crate::product::{ProductBuilder, ProductPTA};

// ---------------------------------------------------------------------------
// VerificationResult
// ---------------------------------------------------------------------------

/// Result of verifying a single safety property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// The property that was checked.
    pub property_id: String,
    pub property_name: String,
    /// Verification verdict.
    pub verdict: Verdict,
    /// Method used (contract, BMC, CEGAR, monolithic).
    pub method: VerificationMethod,
    /// Counterexample if property is violated.
    pub counterexample: Option<CounterExample>,
    /// Clinical narrative (if counterexample exists and narration is enabled).
    pub narrative: Option<ClinicalNarrative>,
    /// Time spent on this property (seconds).
    pub time_secs: f64,
    /// Additional details.
    pub details: String,
}

/// Which verification method produced the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationMethod {
    ContractBased,
    BoundedModelChecking,
    Cegar,
    Monolithic,
    Skipped,
}

impl fmt::Display for VerificationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContractBased => write!(f, "Contract-Based"),
            Self::BoundedModelChecking => write!(f, "BMC"),
            Self::Cegar => write!(f, "CEGAR"),
            Self::Monolithic => write!(f, "Monolithic"),
            Self::Skipped => write!(f, "Skipped"),
        }
    }
}

// ---------------------------------------------------------------------------
// VerificationSummary
// ---------------------------------------------------------------------------

/// Summary of verification results across all properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationSummary {
    pub total_properties: usize,
    pub verified_safe: usize,
    pub verified_unsafe: usize,
    pub unknown: usize,
    pub total_time_secs: f64,
    pub methods_used: HashMap<String, usize>,
}

impl VerificationSummary {
    pub fn from_results(results: &[VerificationResult], total_time: f64) -> Self {
        let mut methods_used: HashMap<String, usize> = HashMap::new();
        let mut safe = 0;
        let mut unsafe_ = 0;
        let mut unknown = 0;

        for r in results {
            match r.verdict {
                Verdict::Safe => safe += 1,
                Verdict::Unsafe => unsafe_ += 1,
                Verdict::Unknown => unknown += 1,
            }
            *methods_used
                .entry(format!("{}", r.method))
                .or_insert(0) += 1;
        }

        Self {
            total_properties: results.len(),
            verified_safe: safe,
            verified_unsafe: unsafe_,
            unknown,
            total_time_secs: total_time,
            methods_used,
        }
    }

    /// Whether all properties were verified safe.
    pub fn all_safe(&self) -> bool {
        self.verified_unsafe == 0 && self.unknown == 0
    }

    /// Whether any property was violated.
    pub fn has_violations(&self) -> bool {
        self.verified_unsafe > 0
    }
}

impl fmt::Display for VerificationSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Verified: {} safe, {} unsafe, {} unknown out of {} properties ({:.1}s)",
            self.verified_safe,
            self.verified_unsafe,
            self.unknown,
            self.total_properties,
            self.total_time_secs
        )
    }
}

// ---------------------------------------------------------------------------
// VerificationReport
// ---------------------------------------------------------------------------

/// Complete verification report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    /// Per-property results.
    pub results: Vec<VerificationResult>,
    /// Aggregate summary.
    pub summary: VerificationSummary,
    /// Timing breakdown.
    pub timing: TimingBreakdown,
    /// Contract composition result (if contracts were used).
    pub contract_result: Option<CompositionResult>,
}

impl VerificationReport {
    /// Get all violated properties.
    pub fn violations(&self) -> Vec<&VerificationResult> {
        self.results.iter().filter(|r| r.verdict == Verdict::Unsafe).collect()
    }

    /// Get all counterexamples.
    pub fn counterexamples(&self) -> Vec<&CounterExample> {
        self.results
            .iter()
            .filter_map(|r| r.counterexample.as_ref())
            .collect()
    }

    /// Get all narratives.
    pub fn narratives(&self) -> Vec<&ClinicalNarrative> {
        self.results
            .iter()
            .filter_map(|r| r.narrative.as_ref())
            .collect()
    }

    /// Format as a human-readable report.
    pub fn format_report(&self) -> String {
        let mut lines = Vec::new();
        lines.push("╔══════════════════════════════════════════════════╗".to_string());
        lines.push("║     GuardPharma Verification Report              ║".to_string());
        lines.push("╚══════════════════════════════════════════════════╝".to_string());
        lines.push(String::new());
        lines.push(format!("{}", self.summary));
        lines.push(String::new());

        for result in &self.results {
            let icon = match result.verdict {
                Verdict::Safe => "✓",
                Verdict::Unsafe => "✗",
                Verdict::Unknown => "?",
            };
            lines.push(format!(
                "  {} [{}] {} — {} ({:.2}s)",
                icon, result.verdict, result.property_name, result.method, result.time_secs
            ));
            if !result.details.is_empty() {
                lines.push(format!("    {}", result.details));
            }
        }

        lines.push(String::new());
        lines.push(format!("Timing: {}", self.timing));

        lines.join("\n")
    }
}

impl fmt::Display for VerificationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_report())
    }
}

// ---------------------------------------------------------------------------
// TimingBreakdown
// ---------------------------------------------------------------------------

/// Timing breakdown for the verification process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingBreakdown {
    pub contract_analysis_secs: f64,
    pub bmc_secs: f64,
    pub cegar_secs: f64,
    pub narration_secs: f64,
    pub overhead_secs: f64,
    pub total_secs: f64,
}

impl TimingBreakdown {
    pub fn new() -> Self {
        Self {
            contract_analysis_secs: 0.0,
            bmc_secs: 0.0,
            cegar_secs: 0.0,
            narration_secs: 0.0,
            overhead_secs: 0.0,
            total_secs: 0.0,
        }
    }
}

impl Default for TimingBreakdown {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TimingBreakdown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "contract={:.2}s bmc={:.2}s cegar={:.2}s narration={:.2}s total={:.2}s",
            self.contract_analysis_secs,
            self.bmc_secs,
            self.cegar_secs,
            self.narration_secs,
            self.total_secs,
        )
    }
}

// ---------------------------------------------------------------------------
// VerificationEngine
// ---------------------------------------------------------------------------

/// Top-level verification engine that orchestrates all verification methods.
#[derive(Debug)]
pub struct VerificationEngine {
    config: VerificationConfig,
    bmc: BoundedModelChecker,
    contract_checker: ContractChecker,
    composition: CompositionEngine,
    cegar: CegarEngine,
    narrator: ClinicalNarrator,
}

impl VerificationEngine {
    pub fn new(config: VerificationConfig) -> Self {
        let bmc_config = BmcConfig {
            max_bound: config.max_bmc_bound,
            global_timeout_secs: config.per_property_timeout_secs,
            ..BmcConfig::default()
        };
        let cegar_config = CegarConfig {
            max_refinements: config.max_cegar_iterations,
            timeout_secs: config.per_property_timeout_secs,
            ..CegarConfig::default()
        };

        Self {
            config: config.clone(),
            bmc: BoundedModelChecker::new(bmc_config),
            contract_checker: ContractChecker::new(
                config.enzyme_capacity_threshold,
                0.05,
            ),
            composition: CompositionEngine::new(),
            cegar: CegarEngine::new(cegar_config),
            narrator: ClinicalNarrator::new(),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(VerificationConfig::default())
    }

    /// Run the full verification pipeline.
    pub fn verify(
        &self,
        ptas: &[PTA],
        properties: &[SafetyProperty],
        config: &VerificationConfig,
    ) -> Result<VerificationReport> {
        let global_start = Instant::now();
        let mut timing = TimingBreakdown::new();
        let mut results: Vec<VerificationResult> = Vec::new();
        let mut contract_result: Option<CompositionResult> = None;

        // Phase 1: Contract-based compositional analysis.
        let mut contract_safe_props: HashMap<String, bool> = HashMap::new();
        if config.use_contracts && ptas.len() > 1 {
            let contract_start = Instant::now();
            let comp_result =
                self.composition.compose_with_contracts(ptas, properties);

            if comp_result.verdict == Verdict::Safe {
                // All properties covered by contracts are safe.
                for prop in properties {
                    contract_safe_props.insert(prop.id.clone(), true);
                }
            }

            timing.contract_analysis_secs = contract_start.elapsed().as_secs_f64();
            contract_result = Some(comp_result);
        }

        // Phase 2: Check each property.
        for property in properties {
            let elapsed = global_start.elapsed().as_secs_f64();
            if elapsed > config.timeout_secs {
                // Timeout: report remaining as unknown.
                results.push(VerificationResult {
                    property_id: property.id.clone(),
                    property_name: property.name.clone(),
                    verdict: Verdict::Unknown,
                    method: VerificationMethod::Skipped,
                    counterexample: None,
                    narrative: None,
                    time_secs: 0.0,
                    details: "Global timeout exceeded".into(),
                });
                continue;
            }

            // Check if contract analysis already verified this property.
            if contract_safe_props.get(&property.id) == Some(&true) {
                results.push(VerificationResult {
                    property_id: property.id.clone(),
                    property_name: property.name.clone(),
                    verdict: Verdict::Safe,
                    method: VerificationMethod::ContractBased,
                    counterexample: None,
                    narrative: None,
                    time_secs: 0.0,
                    details: "Verified via contract-based decomposition".into(),
                });
                continue;
            }

            // Phase 2a: Try BMC on individual PTAs.
            let prop_start = Instant::now();
            let bmc_result = self.try_bmc(ptas, property, config);
            let bmc_time = prop_start.elapsed().as_secs_f64();
            timing.bmc_secs += bmc_time;

            match bmc_result {
                Some(result) if result.verdict != Verdict::Unknown => {
                    let mut vr = self.make_result(
                        property,
                        result,
                        VerificationMethod::BoundedModelChecking,
                        bmc_time,
                    );

                    // Generate narrative if counterexample exists.
                    if config.generate_narratives {
                        if let Some(ref cx) = vr.counterexample {
                            let narr_start = Instant::now();
                            vr.narrative = Some(self.narrator.narrate(cx, &[]));
                            timing.narration_secs += narr_start.elapsed().as_secs_f64();
                        }
                    }

                    results.push(vr);
                    continue;
                }
                _ => {}
            }

            // Phase 2b: Try CEGAR.
            if config.use_cegar {
                let cegar_start = Instant::now();
                let cegar_result = self.try_cegar(ptas, property, config);
                let cegar_time = cegar_start.elapsed().as_secs_f64();
                timing.cegar_secs += cegar_time;

                if let Some(result) = cegar_result {
                    let mut vr = self.make_result(
                        property,
                        result,
                        VerificationMethod::Cegar,
                        bmc_time + cegar_time,
                    );

                    if config.generate_narratives {
                        if let Some(ref cx) = vr.counterexample {
                            let narr_start = Instant::now();
                            vr.narrative = Some(self.narrator.narrate(cx, &[]));
                            timing.narration_secs += narr_start.elapsed().as_secs_f64();
                        }
                    }

                    results.push(vr);
                    continue;
                }
            }

            // Phase 2c: Report unknown.
            results.push(VerificationResult {
                property_id: property.id.clone(),
                property_name: property.name.clone(),
                verdict: Verdict::Unknown,
                method: VerificationMethod::Skipped,
                counterexample: None,
                narrative: None,
                time_secs: global_start.elapsed().as_secs_f64() - elapsed,
                details: "All verification methods inconclusive".into(),
            });
        }

        timing.total_secs = global_start.elapsed().as_secs_f64();
        timing.overhead_secs = timing.total_secs
            - timing.contract_analysis_secs
            - timing.bmc_secs
            - timing.cegar_secs
            - timing.narration_secs;

        let summary = VerificationSummary::from_results(&results, timing.total_secs);

        Ok(VerificationReport {
            results,
            summary,
            timing,
            contract_result,
        })
    }

    /// Try BMC on each PTA for the given property.
    fn try_bmc(
        &self,
        ptas: &[PTA],
        property: &SafetyProperty,
        config: &VerificationConfig,
    ) -> Option<CheckResult> {
        // For single PTA, check directly.
        if ptas.len() == 1 {
            return self.bmc.check(&ptas[0], property, config.max_bmc_bound).ok();
        }

        // For multiple PTAs, check each relevant one.
        let relevant_drugs = property.referenced_drugs();
        let relevant_enzymes = property.referenced_enzymes();

        for pta in ptas {
            let is_relevant = relevant_drugs.is_empty()
                || relevant_drugs.contains(&pta.drug_id)
                || !relevant_enzymes.is_disjoint(&pta.involved_enzymes());

            if is_relevant {
                if let Ok(result) = self.bmc.check(pta, property, config.max_bmc_bound) {
                    if result.verdict == Verdict::Unsafe {
                        return Some(result);
                    }
                }
            }
        }

        // If no violation found, try monolithic on combined.
        if ptas.len() <= 3 {
            let mono = MonolithicBMC::new(BmcConfig {
                max_bound: config.max_bmc_bound / 2,
                global_timeout_secs: config.per_property_timeout_secs / 2.0,
                ..BmcConfig::default()
            });
            return mono.check_independent(ptas, property).ok();
        }

        None
    }

    /// Try CEGAR on the first relevant PTA.
    fn try_cegar(
        &self,
        ptas: &[PTA],
        property: &SafetyProperty,
        config: &VerificationConfig,
    ) -> Option<CheckResult> {
        let cegar_config = CegarConfig {
            max_refinements: config.max_cegar_iterations,
            timeout_secs: config.per_property_timeout_secs,
            abstract_bmc_bound: config.max_bmc_bound / 2,
            ..CegarConfig::default()
        };

        for pta in ptas {
            if let Ok(result) = self.cegar.run_cegar(pta, property, &cegar_config) {
                let check_result = CheckResult {
                    verdict: result.verdict,
                    counterexample: result.counterexample,
                    stats: crate::bounded_checker::BmcStatistics {
                        bound_reached: result.iterations,
                        total_clauses: 0,
                        total_variables: 0,
                        solve_time_per_bound: vec![result.stats.total_time_secs],
                        total_time_secs: result.stats.total_time_secs,
                        solver_calls: result.iterations,
                        timed_out: false,
                    },
                };

                if check_result.verdict != Verdict::Unknown {
                    return Some(check_result);
                }
            }
        }

        None
    }

    /// Convert a CheckResult into a VerificationResult.
    fn make_result(
        &self,
        property: &SafetyProperty,
        check: CheckResult,
        method: VerificationMethod,
        time: f64,
    ) -> VerificationResult {
        let details = match check.verdict {
            Verdict::Safe => format!(
                "Safe up to bound {}, {} solver calls",
                check.stats.bound_reached, check.stats.solver_calls
            ),
            Verdict::Unsafe => format!(
                "Violation found at bound {}, trace length: {}",
                check.stats.bound_reached,
                check.counterexample.as_ref().map_or(0, |cx| cx.len())
            ),
            Verdict::Unknown => {
                if check.stats.timed_out {
                    "Timeout".into()
                } else {
                    format!("Inconclusive at bound {}", check.stats.bound_reached)
                }
            }
        };

        VerificationResult {
            property_id: property.id.clone(),
            property_name: property.name.clone(),
            verdict: check.verdict,
            method,
            counterexample: check.counterexample,
            narrative: None,
            time_secs: time,
            details,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &VerificationConfig {
        &self.config
    }
}

impl Default for VerificationEngine {
    fn default() -> Self {
        Self::with_default_config()
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Quick verification of a single PTA against a single property.
pub fn quick_verify(pta: &PTA, property: &SafetyProperty) -> Result<Verdict> {
    let engine = VerificationEngine::with_default_config();
    let config = VerificationConfig {
        max_bmc_bound: 20,
        timeout_secs: 30.0,
        per_property_timeout_secs: 30.0,
        use_cegar: false,
        generate_narratives: false,
        ..VerificationConfig::default()
    };
    let report = engine.verify(&[pta.clone()], &[property.clone()], &config)?;
    Ok(report.results.first().map(|r| r.verdict).unwrap_or(Verdict::Unknown))
}

/// Verify multiple PTAs with default configuration and return a report.
pub fn verify_all(
    ptas: &[PTA],
    properties: &[SafetyProperty],
) -> Result<VerificationReport> {
    let engine = VerificationEngine::with_default_config();
    let config = VerificationConfig::default();
    engine.verify(ptas, properties, &config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{make_test_pta, DrugId, SafetyProperty, VerificationConfig};

    #[test]
    fn test_verification_engine_default() {
        let engine = VerificationEngine::with_default_config();
        assert!(engine.config().max_bmc_bound > 0);
    }

    #[test]
    fn test_verify_single_pta_safe() {
        let pta = make_test_pta("aspirin", 100.0, false);
        let prop = SafetyProperty::no_error();
        let engine = VerificationEngine::with_default_config();
        let config = VerificationConfig {
            max_bmc_bound: 10,
            timeout_secs: 10.0,
            per_property_timeout_secs: 10.0,
            use_cegar: false,
            generate_narratives: false,
            ..VerificationConfig::default()
        };
        let report = engine.verify(&[pta], &[prop], &config).unwrap();
        assert_eq!(report.results.len(), 1);
        assert_eq!(report.summary.total_properties, 1);
    }

    #[test]
    fn test_verify_multiple_properties() {
        let pta = make_test_pta("drug", 100.0, false);
        let props = vec![
            SafetyProperty::no_error(),
            SafetyProperty::concentration_bound(DrugId::new("drug"), 50.0),
        ];
        let engine = VerificationEngine::with_default_config();
        let config = VerificationConfig {
            max_bmc_bound: 5,
            timeout_secs: 10.0,
            per_property_timeout_secs: 5.0,
            use_cegar: false,
            generate_narratives: false,
            ..VerificationConfig::default()
        };
        let report = engine.verify(&[pta], &props, &config).unwrap();
        assert_eq!(report.results.len(), 2);
    }

    #[test]
    fn test_verify_multiple_ptas() {
        let ptas = vec![
            make_test_pta("drugA", 100.0, false),
            make_test_pta("drugB", 200.0, false),
        ];
        let props = vec![SafetyProperty::no_error()];
        let engine = VerificationEngine::with_default_config();
        let config = VerificationConfig {
            max_bmc_bound: 5,
            timeout_secs: 10.0,
            per_property_timeout_secs: 5.0,
            use_contracts: true,
            use_cegar: false,
            generate_narratives: false,
            ..VerificationConfig::default()
        };
        let report = engine.verify(&ptas, &props, &config).unwrap();
        assert!(!report.results.is_empty());
    }

    #[test]
    fn test_verification_summary() {
        let results = vec![
            VerificationResult {
                property_id: "p1".into(),
                property_name: "prop1".into(),
                verdict: Verdict::Safe,
                method: VerificationMethod::BoundedModelChecking,
                counterexample: None,
                narrative: None,
                time_secs: 1.0,
                details: String::new(),
            },
            VerificationResult {
                property_id: "p2".into(),
                property_name: "prop2".into(),
                verdict: Verdict::Unsafe,
                method: VerificationMethod::BoundedModelChecking,
                counterexample: None,
                narrative: None,
                time_secs: 2.0,
                details: String::new(),
            },
        ];
        let summary = VerificationSummary::from_results(&results, 3.0);
        assert_eq!(summary.verified_safe, 1);
        assert_eq!(summary.verified_unsafe, 1);
        assert!(!summary.all_safe());
        assert!(summary.has_violations());
    }

    #[test]
    fn test_verification_report_format() {
        let pta = make_test_pta("drug", 100.0, false);
        let prop = SafetyProperty::no_error();
        let engine = VerificationEngine::with_default_config();
        let config = VerificationConfig {
            max_bmc_bound: 5,
            timeout_secs: 5.0,
            per_property_timeout_secs: 5.0,
            use_cegar: false,
            generate_narratives: false,
            ..VerificationConfig::default()
        };
        let report = engine.verify(&[pta], &[prop], &config).unwrap();
        let formatted = report.format_report();
        assert!(formatted.contains("GuardPharma"));
        assert!(formatted.contains("Verified"));
    }

    #[test]
    fn test_timing_breakdown() {
        let timing = TimingBreakdown::new();
        assert_eq!(timing.total_secs, 0.0);
        let s = format!("{}", timing);
        assert!(s.contains("contract="));
    }

    #[test]
    fn test_quick_verify() {
        let pta = make_test_pta("drug", 100.0, false);
        let prop = SafetyProperty::no_error();
        let verdict = quick_verify(&pta, &prop).unwrap();
        assert_eq!(verdict, Verdict::Safe);
    }

    #[test]
    fn test_verify_all() {
        let ptas = vec![make_test_pta("drug", 100.0, false)];
        let props = vec![SafetyProperty::no_error()];
        let report = verify_all(&ptas, &props).unwrap();
        assert_eq!(report.results.len(), 1);
    }

    #[test]
    fn test_verification_method_display() {
        assert_eq!(format!("{}", VerificationMethod::ContractBased), "Contract-Based");
        assert_eq!(format!("{}", VerificationMethod::BoundedModelChecking), "BMC");
        assert_eq!(format!("{}", VerificationMethod::Cegar), "CEGAR");
    }

    #[test]
    fn test_report_violations() {
        let results = vec![
            VerificationResult {
                property_id: "p1".into(),
                property_name: "prop1".into(),
                verdict: Verdict::Unsafe,
                method: VerificationMethod::BoundedModelChecking,
                counterexample: Some(CounterExample::empty("p1".into())),
                narrative: None,
                time_secs: 1.0,
                details: String::new(),
            },
        ];
        let summary = VerificationSummary::from_results(&results, 1.0);
        let report = VerificationReport {
            results,
            summary,
            timing: TimingBreakdown::new(),
            contract_result: None,
        };
        assert_eq!(report.violations().len(), 1);
        assert_eq!(report.counterexamples().len(), 1);
    }

    #[test]
    fn test_verify_with_narrative() {
        let pta = make_test_pta("drug", 100.0, false);
        let prop = SafetyProperty::no_error();
        let engine = VerificationEngine::with_default_config();
        let config = VerificationConfig {
            max_bmc_bound: 5,
            timeout_secs: 5.0,
            per_property_timeout_secs: 5.0,
            use_cegar: false,
            generate_narratives: true,
            ..VerificationConfig::default()
        };
        let report = engine.verify(&[pta], &[prop], &config).unwrap();
        // Safe property → no narrative.
        assert!(report.narratives().is_empty());
    }

    #[test]
    fn test_summary_display() {
        let summary = VerificationSummary {
            total_properties: 3,
            verified_safe: 2,
            verified_unsafe: 1,
            unknown: 0,
            total_time_secs: 5.5,
            methods_used: HashMap::new(),
        };
        let s = format!("{}", summary);
        assert!(s.contains("2 safe"));
        assert!(s.contains("1 unsafe"));
    }
}
