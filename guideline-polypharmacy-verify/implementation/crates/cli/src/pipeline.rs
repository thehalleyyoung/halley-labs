//! Pipeline orchestration for GuardPharma CLI.
//!
//! Manages the two-tier verification pipeline: loading, Tier 1 screening,
//! Tier 2 model checking, conflict analysis, significance scoring,
//! recommendation synthesis, and report generation.

use anyhow::{bail, Context, Result};
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use std::time::Instant;

use guardpharma_types::{
    Concentration, ConcentrationInterval, CypEnzyme, DrugId, DrugRoute, EnzymeActivity, Severity,
    TherapeuticWindow,
};

use crate::config::AppConfig;
use crate::input::{ActiveMedication, GuidelineDocument, GuidelineRule, PatientProfile};

// ──────────────────────── Pipeline Data Types ────────────────────────────

/// Final output of the full verification pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineOutput {
    pub run_id: String,
    pub timestamp: String,
    pub patient: PatientProfile,
    pub verdict: VerificationVerdict,
    pub guidelines_checked: usize,
    pub drug_pairs_checked: usize,
    pub screening_results: Vec<ScreeningResult>,
    pub conflicts: Vec<ConflictReport>,
    pub recommendations: Vec<Recommendation>,
    pub certificate: Option<SafetyCertificate>,
    pub timings: Vec<PhaseTiming>,
}

/// Output of Tier 1 screening only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreeningOutput {
    pub timestamp: String,
    pub patient: PatientProfile,
    pub results: Vec<ScreeningResult>,
    pub timings: Vec<PhaseTiming>,
}

/// Output of detailed analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisOutput {
    pub timestamp: String,
    pub patient: PatientProfile,
    pub conflicts: Vec<ConflictReport>,
    pub enzyme_details: Vec<EnzymePathwayDetail>,
    pub pk_traces: Vec<PkTrace>,
    pub timings: Vec<PhaseTiming>,
}

/// Verification verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationVerdict {
    /// No conflicts found — medication regimen is safe.
    Safe,
    /// Conflicts were detected.
    ConflictsFound { count: usize },
    /// Verification could not be completed.
    Inconclusive { reason: String },
    /// An error occurred during verification.
    Error { message: String },
}

/// Result of a Tier 1 screening check for one drug pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreeningResult {
    pub drug_a: DrugId,
    pub drug_b: DrugId,
    pub drug_a_name: String,
    pub drug_b_name: String,
    pub severity: Severity,
    pub confidence: f64,
    pub mechanism: String,
    pub affected_enzymes: Vec<CypEnzyme>,
    pub concentration_overlap: bool,
    pub guideline_violations: Vec<String>,
}

/// A confirmed conflict with full analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictReport {
    pub drug_a_name: String,
    pub drug_b_name: String,
    pub severity: Severity,
    pub confidence: f64,
    pub mechanism: String,
    pub description: String,
    pub clinical_consequence: String,
    pub trace: Vec<TraceStep>,
}

/// A single step in a counterexample trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStep {
    pub time_hours: f64,
    pub description: String,
    pub concentration_a: Option<(f64, f64)>,
    pub concentration_b: Option<(f64, f64)>,
    pub violation: Option<String>,
}

/// A safety recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub summary: String,
    pub priority: Severity,
    pub category: String,
    pub rationale: String,
    pub alternative: Option<String>,
    pub monitoring: Option<String>,
    pub affected_drugs: Vec<String>,
}

/// Safety verification certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyCertificate {
    pub certificate_id: String,
    pub run_id: String,
    pub timestamp: String,
    pub patient_id: String,
    pub verdict: VerificationVerdict,
    pub guidelines_checked: usize,
    pub drug_pairs_checked: usize,
    pub conflicts_found: usize,
    pub tier1_completed: bool,
    pub tier2_completed: bool,
    pub warnings: Vec<String>,
}

/// Timing information for a pipeline phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTiming {
    pub phase_name: String,
    pub duration_ms: f64,
}

/// Enzyme pathway analysis detail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnzymePathwayDetail {
    pub enzyme: CypEnzyme,
    pub net_activity: f64,
    pub effects: Vec<(String, String)>,
}

/// Pharmacokinetic concentration trace for a drug.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PkTrace {
    pub drug_name: String,
    pub time_points: Vec<f64>,
    pub concentrations_lower: Vec<f64>,
    pub concentrations_upper: Vec<f64>,
    pub therapeutic_window: Option<(f64, f64)>,
}

// ──────────────────── Pipeline Phase Enum ────────────────────────────────

/// Pipeline execution phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    Loading,
    Tier1Screening,
    Tier2Verification,
    ConflictAnalysis,
    SignificanceScoring,
    RecommendationSynthesis,
    ReportGeneration,
}

impl Phase {
    pub fn name(&self) -> &'static str {
        match self {
            Phase::Loading => "Loading",
            Phase::Tier1Screening => "Tier 1 Screening",
            Phase::Tier2Verification => "Tier 2 Verification",
            Phase::ConflictAnalysis => "Conflict Analysis",
            Phase::SignificanceScoring => "Significance Scoring",
            Phase::RecommendationSynthesis => "Recommendation Synthesis",
            Phase::ReportGeneration => "Report Generation",
        }
    }

    pub fn all() -> &'static [Phase] {
        &[
            Phase::Loading,
            Phase::Tier1Screening,
            Phase::Tier2Verification,
            Phase::ConflictAnalysis,
            Phase::SignificanceScoring,
            Phase::RecommendationSynthesis,
            Phase::ReportGeneration,
        ]
    }
}

impl std::fmt::Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ───────────────────── Pipeline Progress ─────────────────────────────────

/// Callback type for reporting pipeline progress.
pub type PipelineProgressCallback = Box<dyn Fn(Phase, &str)>;

// ─────────────────── Pipeline Timer ──────────────────────────────────────

/// Tracks timing for each pipeline phase.
struct PipelineTimer {
    timings: Vec<PhaseTiming>,
    current_start: Option<Instant>,
    current_phase: Option<String>,
}

impl PipelineTimer {
    fn new() -> Self {
        PipelineTimer {
            timings: Vec::new(),
            current_start: None,
            current_phase: None,
        }
    }

    fn start_phase(&mut self, name: &str) {
        self.finish_current();
        self.current_phase = Some(name.to_string());
        self.current_start = Some(Instant::now());
    }

    fn finish_current(&mut self) {
        if let (Some(start), Some(name)) = (self.current_start.take(), self.current_phase.take()) {
            let duration = start.elapsed();
            self.timings.push(PhaseTiming {
                phase_name: name,
                duration_ms: duration.as_secs_f64() * 1000.0,
            });
        }
    }

    fn into_timings(mut self) -> Vec<PhaseTiming> {
        self.finish_current();
        self.timings
    }

    fn total_ms(&self) -> f64 {
        self.timings.iter().map(|t| t.duration_ms).sum()
    }
}

// ─────────────────── Pipeline Orchestrator ───────────────────────────────

/// Orchestrates the full GuardPharma verification pipeline.
pub struct PipelineOrchestrator<'a> {
    config: &'a AppConfig,
    progress: Option<PipelineProgressCallback>,
}

impl<'a> PipelineOrchestrator<'a> {
    pub fn new(config: &'a AppConfig) -> Self {
        PipelineOrchestrator {
            config,
            progress: None,
        }
    }

    pub fn with_progress(mut self, callback: PipelineProgressCallback) -> Self {
        self.progress = Some(callback);
        self
    }

    fn report_progress(&self, phase: Phase, message: &str) {
        if let Some(ref callback) = self.progress {
            callback(phase, message);
        }
    }

    // ─────────────── Full Pipeline ───────────────────────────────────────

    /// Run the full two-tier verification pipeline.
    pub fn run_full_pipeline(
        &self,
        guidelines: &[GuidelineDocument],
        patient: &PatientProfile,
    ) -> Result<PipelineOutput> {
        let mut timer = PipelineTimer::new();
        let run_id = generate_run_id();
        let timestamp = current_timestamp();

        info!("Starting full pipeline run: {}", run_id);

        // Phase 1: Load and validate
        timer.start_phase(Phase::Loading.name());
        self.report_progress(Phase::Loading, "Validating inputs");
        let total_rules: usize = guidelines.iter().map(|g| g.rule_count()).sum();
        let drug_pairs = patient.drug_pairs();
        info!(
            "Loaded {} guidelines with {} rules, {} medications, {} drug pairs",
            guidelines.len(),
            total_rules,
            patient.medication_count(),
            drug_pairs.len()
        );

        // Phase 2: Tier 1 Screening
        timer.start_phase(Phase::Tier1Screening.name());
        self.report_progress(Phase::Tier1Screening, "Running pharmacokinetic screening");
        let screening_results = if self.config.verification.enable_tier1 {
            self.run_tier1_screening(guidelines, patient)?
        } else {
            info!("Tier 1 screening disabled");
            Vec::new()
        };

        let flagged_pairs: Vec<_> = screening_results
            .iter()
            .filter(|r| r.severity >= Severity::Moderate)
            .collect();
        info!(
            "Tier 1: {} results, {} flagged for Tier 2",
            screening_results.len(),
            flagged_pairs.len()
        );

        // Phase 3: Tier 2 Verification (only flagged pairs)
        timer.start_phase(Phase::Tier2Verification.name());
        self.report_progress(Phase::Tier2Verification, "Running model checking");
        let tier2_conflicts = if self.config.verification.enable_tier2 {
            self.run_tier2_verification(guidelines, patient, &screening_results)?
        } else {
            info!("Tier 2 verification disabled");
            Vec::new()
        };

        // Phase 4: Conflict Analysis
        timer.start_phase(Phase::ConflictAnalysis.name());
        self.report_progress(Phase::ConflictAnalysis, "Analyzing confirmed conflicts");
        let conflicts = self.analyze_conflicts(&tier2_conflicts, guidelines, patient)?;

        // Phase 5: Significance Scoring
        timer.start_phase(Phase::SignificanceScoring.name());
        self.report_progress(Phase::SignificanceScoring, "Scoring clinical significance");
        let scored_conflicts = self.score_significance(&conflicts, patient)?;

        // Phase 6: Recommendation Synthesis
        timer.start_phase(Phase::RecommendationSynthesis.name());
        self.report_progress(
            Phase::RecommendationSynthesis,
            "Synthesizing recommendations",
        );
        let recommendations =
            self.synthesize_recommendations(&scored_conflicts, guidelines, patient)?;

        // Phase 7: Report Generation
        timer.start_phase(Phase::ReportGeneration.name());
        self.report_progress(Phase::ReportGeneration, "Generating report");

        let verdict = if scored_conflicts.is_empty() {
            VerificationVerdict::Safe
        } else {
            VerificationVerdict::ConflictsFound {
                count: scored_conflicts.len(),
            }
        };

        let certificate = SafetyCertificate {
            certificate_id: format!("CERT-{}", &run_id[..8.min(run_id.len())]),
            run_id: run_id.clone(),
            timestamp: timestamp.clone(),
            patient_id: patient.id.clone().unwrap_or_else(|| "unknown".to_string()),
            verdict: verdict.clone(),
            guidelines_checked: guidelines.len(),
            drug_pairs_checked: drug_pairs.len(),
            conflicts_found: scored_conflicts.len(),
            tier1_completed: self.config.verification.enable_tier1,
            tier2_completed: self.config.verification.enable_tier2,
            warnings: self.collect_warnings(&screening_results, &scored_conflicts),
        };

        let timings = timer.into_timings();

        Ok(PipelineOutput {
            run_id,
            timestamp,
            patient: patient.clone(),
            verdict,
            guidelines_checked: guidelines.len(),
            drug_pairs_checked: drug_pairs.len(),
            screening_results,
            conflicts: scored_conflicts,
            recommendations,
            certificate: Some(certificate),
            timings,
        })
    }

    // ─────────────── Screening Only ──────────────────────────────────────

    /// Run Tier 1 screening only (fast check).
    pub fn run_screening_only(
        &self,
        guidelines: &[GuidelineDocument],
        patient: &PatientProfile,
    ) -> Result<ScreeningOutput> {
        let mut timer = PipelineTimer::new();
        let timestamp = current_timestamp();

        timer.start_phase(Phase::Loading.name());
        self.report_progress(Phase::Loading, "Validating inputs");

        timer.start_phase(Phase::Tier1Screening.name());
        self.report_progress(Phase::Tier1Screening, "Running Tier 1 screening");
        let results = self.run_tier1_screening(guidelines, patient)?;

        let timings = timer.into_timings();

        Ok(ScreeningOutput {
            timestamp,
            patient: patient.clone(),
            results,
            timings,
        })
    }

    // ─────────────── Analysis Only ───────────────────────────────────────

    /// Run analysis (conflict detection + enzyme details).
    pub fn run_analysis_only(
        &self,
        guidelines: &[GuidelineDocument],
        patient: &PatientProfile,
        show_enzymes: bool,
        show_pk: bool,
    ) -> Result<AnalysisOutput> {
        let mut timer = PipelineTimer::new();
        let timestamp = current_timestamp();

        timer.start_phase(Phase::Loading.name());
        self.report_progress(Phase::Loading, "Validating inputs");

        timer.start_phase(Phase::Tier1Screening.name());
        let screening = self.run_tier1_screening(guidelines, patient)?;

        timer.start_phase(Phase::Tier2Verification.name());
        let raw_conflicts = self.run_tier2_verification(guidelines, patient, &screening)?;

        timer.start_phase(Phase::ConflictAnalysis.name());
        let conflicts = self.analyze_conflicts(&raw_conflicts, guidelines, patient)?;

        let enzyme_details = if show_enzymes {
            self.compute_enzyme_details(patient)?
        } else {
            Vec::new()
        };

        let pk_traces = if show_pk {
            self.compute_pk_traces(patient)?
        } else {
            Vec::new()
        };

        let timings = timer.into_timings();

        Ok(AnalysisOutput {
            timestamp,
            patient: patient.clone(),
            conflicts,
            enzyme_details,
            pk_traces,
            timings,
        })
    }

    // ──────────────── Internal: Tier 1 ───────────────────────────────────

    fn run_tier1_screening(
        &self,
        guidelines: &[GuidelineDocument],
        patient: &PatientProfile,
    ) -> Result<Vec<ScreeningResult>> {
        let mut results = Vec::new();
        let pairs = patient.drug_pairs();
        let max_pairs = self.config.verification.max_drug_pairs.min(pairs.len());

        for (med_a, med_b) in pairs.iter().take(max_pairs) {
            let result = self.screen_drug_pair(med_a, med_b, guidelines, patient);
            match result {
                Ok(Some(screening)) => results.push(screening),
                Ok(None) => {
                    debug!("No interaction found: {} + {}", med_a.name, med_b.name);
                }
                Err(e) => {
                    if self.config.verification.continue_on_error {
                        warn!("Screening error for {} + {}: {}", med_a.name, med_b.name, e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        results.sort_by(|a, b| b.severity.cmp(&a.severity).then(b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)));
        Ok(results)
    }

    fn screen_drug_pair(
        &self,
        med_a: &ActiveMedication,
        med_b: &ActiveMedication,
        guidelines: &[GuidelineDocument],
        patient: &PatientProfile,
    ) -> Result<Option<ScreeningResult>> {
        // Check guideline rules for this drug pair
        let mut violations = Vec::new();
        let mut max_severity = None;
        let mut mechanism = String::new();
        let mut affected_enzymes = Vec::new();

        for guideline in guidelines {
            for rule in &guideline.rules {
                let a_name = med_a.canonical_name();
                let b_name = med_b.canonical_name();

                let a_affected = rule
                    .affected_drugs
                    .iter()
                    .any(|d| d.to_lowercase() == a_name);
                let b_affected = rule
                    .affected_drugs
                    .iter()
                    .any(|d| d.to_lowercase() == b_name);

                if a_affected && b_affected {
                    violations.push(format!(
                        "{}: {} ({})",
                        guideline.name, rule.description, rule.id
                    ));
                    if max_severity.is_none() || rule.severity > max_severity.unwrap() {
                        max_severity = Some(rule.severity);
                        mechanism = rule.description.clone();
                    }
                }
            }
        }

        // Compute CYP enzyme interaction screening
        let enzyme_interaction = self.compute_enzyme_interaction(med_a, med_b);
        if let Some(ref interaction) = enzyme_interaction {
            affected_enzymes = interaction.enzymes.clone();
            if max_severity.is_none()
                || interaction.severity > max_severity.unwrap_or(Severity::Minor)
            {
                max_severity = Some(interaction.severity);
                if mechanism.is_empty() {
                    mechanism = interaction.mechanism.clone();
                }
            }
        }

        // Compute concentration interval overlap
        let conc_overlap =
            self.check_concentration_overlap(med_a, med_b, patient);

        // Determine confidence
        let confidence = compute_screening_confidence(
            &violations,
            enzyme_interaction.as_ref(),
            conc_overlap,
            patient,
        );

        if let Some(severity) = max_severity {
            if confidence >= self.config.verification.confidence_threshold {
                return Ok(Some(ScreeningResult {
                    drug_a: med_a.drug_id.clone(),
                    drug_b: med_b.drug_id.clone(),
                    drug_a_name: med_a.name.clone(),
                    drug_b_name: med_b.name.clone(),
                    severity,
                    confidence,
                    mechanism,
                    affected_enzymes,
                    concentration_overlap: conc_overlap,
                    guideline_violations: violations,
                }));
            }
        }

        Ok(None)
    }

    fn compute_enzyme_interaction(
        &self,
        med_a: &ActiveMedication,
        med_b: &ActiveMedication,
    ) -> Option<EnzymeInteraction> {
        // Check known CYP interactions from drug class
        let known_interactions = get_known_enzyme_interactions();

        let a_name = med_a.canonical_name();
        let b_name = med_b.canonical_name();

        for interaction in &known_interactions {
            let matches_forward = interaction.drug_a == a_name && interaction.drug_b == b_name;
            let matches_reverse = interaction.drug_a == b_name && interaction.drug_b == a_name;

            if matches_forward || matches_reverse {
                return Some(interaction.clone());
            }
        }

        // Heuristic: drugs of certain classes commonly interact
        let class_interaction = check_class_interaction(&med_a.drug_class, &med_b.drug_class);
        if let Some(severity) = class_interaction {
            return Some(EnzymeInteraction {
                drug_a: a_name,
                drug_b: b_name,
                enzymes: vec![CypEnzyme::CYP3A4],
                severity,
                mechanism: format!(
                    "Class interaction: {} + {}",
                    med_a.drug_class, med_b.drug_class
                ),
            });
        }

        None
    }

    fn check_concentration_overlap(
        &self,
        med_a: &ActiveMedication,
        med_b: &ActiveMedication,
        patient: &PatientProfile,
    ) -> bool {
        // Simplified concentration overlap check using basic PK modeling
        let vd_a = estimate_volume_of_distribution(med_a, patient);
        let vd_b = estimate_volume_of_distribution(med_b, patient);

        if vd_a <= 0.0 || vd_b <= 0.0 {
            return false;
        }

        let cmax_a = med_a.dose_mg / vd_a;
        let cmax_b = med_b.dose_mg / vd_b;

        // Check if both drugs reach potentially significant concentrations
        cmax_a > 0.01 && cmax_b > 0.01
    }

    // ──────────────── Internal: Tier 2 ───────────────────────────────────

    fn run_tier2_verification(
        &self,
        guidelines: &[GuidelineDocument],
        patient: &PatientProfile,
        screening: &[ScreeningResult],
    ) -> Result<Vec<RawConflict>> {
        let mut raw_conflicts = Vec::new();

        let flagged: Vec<_> = screening
            .iter()
            .filter(|r| r.severity >= Severity::Moderate || r.concentration_overlap)
            .collect();

        info!("Tier 2: Verifying {} flagged pairs", flagged.len());

        for result in flagged {
            let verification_result = self.verify_drug_pair_tier2(result, guidelines, patient);
            match verification_result {
                Ok(Some(conflict)) => raw_conflicts.push(conflict),
                Ok(None) => {
                    debug!(
                        "Tier 2 cleared: {} + {}",
                        result.drug_a_name, result.drug_b_name
                    );
                }
                Err(e) => {
                    if self.config.verification.continue_on_error {
                        warn!(
                            "Tier 2 error for {} + {}: {}",
                            result.drug_a_name, result.drug_b_name, e
                        );
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Ok(raw_conflicts)
    }

    fn verify_drug_pair_tier2(
        &self,
        screening: &ScreeningResult,
        guidelines: &[GuidelineDocument],
        patient: &PatientProfile,
    ) -> Result<Option<RawConflict>> {
        debug!(
            "Tier 2 verification: {} + {}",
            screening.drug_a_name, screening.drug_b_name
        );

        // Build formal verification context
        let vd_a = estimate_volume_of_distribution_by_name(&screening.drug_a_name, patient);
        let vd_b = estimate_volume_of_distribution_by_name(&screening.drug_b_name, patient);

        let ke_a = estimate_elimination_rate(&screening.drug_a_name);
        let ke_b = estimate_elimination_rate(&screening.drug_b_name);

        // Simulate concentration intervals over time
        let dose_a = get_typical_dose(&screening.drug_a_name);
        let dose_b = get_typical_dose(&screening.drug_b_name);

        let mut trace = Vec::new();
        let mut violation_found = false;
        let mut violation_description = String::new();

        let sim_hours = self.config.pk_model.max_simulation_hours.min(168.0);
        let step = self.config.pk_model.ode_step_size.max(0.5);

        let therapeutic_a = get_therapeutic_window(&screening.drug_a_name);
        let therapeutic_b = get_therapeutic_window(&screening.drug_b_name);

        let mut t = 0.0_f64;
        while t <= sim_hours {
            // Compute concentrations with enzyme interaction effects
            let inhibition_factor = if screening.affected_enzymes.contains(&CypEnzyme::CYP3A4) {
                0.5 // inhibited CYP3A4 reduces clearance
            } else {
                1.0
            };

            let conc_a = compute_interval_concentration(
                dose_a,
                vd_a,
                ke_a * inhibition_factor,
                t,
                24.0, // assume q24h
            );
            let conc_b = compute_interval_concentration(
                dose_b,
                vd_b,
                ke_b * inhibition_factor,
                t,
                24.0,
            );

            // Check for therapeutic window violations
            let a_violated = therapeutic_a.as_ref()
                .map(|tw| conc_a.1 > tw.max_concentration || conc_a.0 < tw.min_concentration * 0.5)
                .unwrap_or(false);
            let b_violated = therapeutic_b.as_ref()
                .map(|tw| conc_b.1 > tw.max_concentration || conc_b.0 < tw.min_concentration * 0.5)
                .unwrap_or(false);

            if (a_violated || b_violated) && !violation_found {
                violation_found = true;
                violation_description = if a_violated && b_violated {
                    format!(
                        "Both {} and {} outside therapeutic range at t={:.1}h",
                        screening.drug_a_name, screening.drug_b_name, t
                    )
                } else if a_violated {
                    format!(
                        "{} outside therapeutic range at t={:.1}h (conc [{:.4}, {:.4}] mg/L)",
                        screening.drug_a_name, t, conc_a.0, conc_a.1
                    )
                } else {
                    format!(
                        "{} outside therapeutic range at t={:.1}h (conc [{:.4}, {:.4}] mg/L)",
                        screening.drug_b_name, t, conc_b.0, conc_b.1
                    )
                };
            }

            // Record trace at selected time points
            let is_key_point = t == 0.0
                || (t % 4.0).abs() < step
                || (a_violated || b_violated)
                || t >= sim_hours - step;

            if is_key_point {
                let step_violation = if a_violated || b_violated {
                    Some(if a_violated && b_violated {
                        "Both drugs outside therapeutic range".to_string()
                    } else if a_violated {
                        format!("{} outside therapeutic range", screening.drug_a_name)
                    } else {
                        format!("{} outside therapeutic range", screening.drug_b_name)
                    })
                } else {
                    None
                };

                trace.push(TraceStep {
                    time_hours: t,
                    description: format!("t={:.1}h simulation state", t),
                    concentration_a: Some(conc_a),
                    concentration_b: Some(conc_b),
                    violation: step_violation,
                });
            }

            t += step;
        }

        // Check guideline constraint violations
        let mut guideline_violations = Vec::new();
        for guideline in guidelines {
            for rule in &guideline.rules {
                let a_name = screening.drug_a_name.to_lowercase();
                let b_name = screening.drug_b_name.to_lowercase();
                let both_affected = rule
                    .affected_drugs
                    .iter()
                    .any(|d| d.to_lowercase() == a_name)
                    && rule
                        .affected_drugs
                        .iter()
                        .any(|d| d.to_lowercase() == b_name);

                if both_affected {
                    guideline_violations.push(format!("{}: {}", guideline.name, rule.description));
                }
            }
        }

        if violation_found || !guideline_violations.is_empty() {
            let severity = screening.severity;
            let description = if violation_found {
                violation_description
            } else {
                guideline_violations.join("; ")
            };

            Ok(Some(RawConflict {
                drug_a_name: screening.drug_a_name.clone(),
                drug_b_name: screening.drug_b_name.clone(),
                severity,
                mechanism: screening.mechanism.clone(),
                description,
                trace,
                guideline_violations,
            }))
        } else {
            Ok(None)
        }
    }

    // ──────────── Internal: Conflict Analysis ────────────────────────────

    fn analyze_conflicts(
        &self,
        raw_conflicts: &[RawConflict],
        _guidelines: &[GuidelineDocument],
        _patient: &PatientProfile,
    ) -> Result<Vec<ConflictReport>> {
        let mut reports = Vec::new();

        for raw in raw_conflicts {
            let clinical_consequence = derive_clinical_consequence(
                &raw.drug_a_name,
                &raw.drug_b_name,
                &raw.mechanism,
                raw.severity,
            );

            reports.push(ConflictReport {
                drug_a_name: raw.drug_a_name.clone(),
                drug_b_name: raw.drug_b_name.clone(),
                severity: raw.severity,
                confidence: compute_tier2_confidence(raw),
                mechanism: raw.mechanism.clone(),
                description: raw.description.clone(),
                clinical_consequence,
                trace: raw.trace.clone(),
            });
        }

        reports.sort_by(|a, b| {
            b.severity
                .cmp(&a.severity)
                .then(b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal))
        });
        Ok(reports)
    }

    // ──────────── Internal: Significance Scoring ─────────────────────────

    fn score_significance(
        &self,
        conflicts: &[ConflictReport],
        patient: &PatientProfile,
    ) -> Result<Vec<ConflictReport>> {
        let mut scored = Vec::new();

        for conflict in conflicts {
            let mut adjusted = conflict.clone();

            // Adjust severity based on patient factors
            if patient.is_elderly() && conflict.severity == Severity::Moderate {
                adjusted.confidence = (adjusted.confidence * 1.2).min(1.0);
            }

            if patient.renal_function().requires_dose_adjustment() {
                adjusted.confidence = (adjusted.confidence * 1.15).min(1.0);
            }

            // Filter by minimum severity threshold
            let severity_score = match adjusted.severity {
                Severity::None => 0.0,
                Severity::Minor => 0.25,
                Severity::Moderate => 0.5,
                Severity::Major => 0.75,
                Severity::Contraindicated => 1.0,
            };

            if severity_score >= self.config.significance.min_severity_threshold {
                scored.push(adjusted);
            }
        }

        Ok(scored)
    }

    // ──────────── Internal: Recommendation Synthesis ─────────────────────

    fn synthesize_recommendations(
        &self,
        conflicts: &[ConflictReport],
        _guidelines: &[GuidelineDocument],
        patient: &PatientProfile,
    ) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();

        for conflict in conflicts {
            // Primary recommendation
            let (summary, rationale, alternative, monitoring) =
                generate_recommendation_text(conflict, patient);

            recommendations.push(Recommendation {
                summary,
                priority: conflict.severity,
                category: categorize_recommendation(conflict),
                rationale,
                alternative,
                monitoring,
                affected_drugs: vec![
                    conflict.drug_a_name.clone(),
                    conflict.drug_b_name.clone(),
                ],
            });
        }

        // Add general polypharmacy recommendations
        if patient.medication_count() >= 5 {
            recommendations.push(Recommendation {
                summary: "Comprehensive medication review recommended".to_string(),
                priority: Severity::Moderate,
                category: "General".to_string(),
                rationale: format!(
                    "Patient is on {} medications — polypharmacy increases interaction risk",
                    patient.medication_count()
                ),
                alternative: None,
                monitoring: Some(
                    "Schedule regular medication reconciliation".to_string(),
                ),
                affected_drugs: patient
                    .medications
                    .iter()
                    .map(|m| m.name.clone())
                    .collect(),
            });
        }

        if patient.is_elderly() {
            recommendations.push(Recommendation {
                summary: "Review Beers Criteria for potentially inappropriate medications"
                    .to_string(),
                priority: Severity::Moderate,
                category: "Age-related".to_string(),
                rationale: format!(
                    "Patient is {:.0} years old — elderly patients have increased sensitivity",
                    patient.age()
                ),
                alternative: None,
                monitoring: Some("Monitor for adverse drug events".to_string()),
                affected_drugs: vec![],
            });
        }

        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));
        Ok(recommendations)
    }

    // ──────────── Internal: Enzyme Details ───────────────────────────────

    fn compute_enzyme_details(
        &self,
        patient: &PatientProfile,
    ) -> Result<Vec<EnzymePathwayDetail>> {
        let mut details = Vec::new();

        for enzyme in CypEnzyme::all() {
            let mut effects = Vec::new();
            let mut net_activity = 1.0_f64;

            for med in &patient.medications {
                let effect = get_drug_enzyme_effect(&med.canonical_name(), enzyme);
                if let Some((effect_type, factor)) = effect {
                    effects.push((med.name.clone(), effect_type));
                    net_activity *= factor;
                }
            }

            // Age-related decline
            if patient.is_elderly() {
                let age_factor = 1.0 - (patient.age() - 65.0) * 0.005;
                net_activity *= age_factor.max(0.5);
            }

            if !effects.is_empty() || net_activity < 0.95 {
                details.push(EnzymePathwayDetail {
                    enzyme: *enzyme,
                    net_activity: net_activity.max(0.0),
                    effects,
                });
            }
        }

        Ok(details)
    }

    // ──────────── Internal: PK Traces ────────────────────────────────────

    fn compute_pk_traces(&self, patient: &PatientProfile) -> Result<Vec<PkTrace>> {
        let mut traces = Vec::new();

        for med in &patient.medications {
            let vd = estimate_volume_of_distribution(med, patient);
            let ke = estimate_elimination_rate(&med.canonical_name());
            let therapeutic = get_therapeutic_window(&med.canonical_name());

            let mut times = Vec::new();
            let mut lowers = Vec::new();
            let mut uppers = Vec::new();

            let sim_hours = 48.0_f64;
            let step = 0.5_f64;
            let mut t = 0.0_f64;

            while t <= sim_hours {
                let (lo, hi) =
                    compute_interval_concentration(med.dose_mg, vd, ke, t, med.frequency_hours);
                times.push(t);
                lowers.push(lo);
                uppers.push(hi);
                t += step;
            }

            traces.push(PkTrace {
                drug_name: med.name.clone(),
                time_points: times,
                concentrations_lower: lowers,
                concentrations_upper: uppers,
                therapeutic_window: therapeutic.map(|tw| (tw.min_concentration, tw.max_concentration)),
            });
        }

        Ok(traces)
    }

    // ──────────── Internal: Warnings ─────────────────────────────────────

    fn collect_warnings(
        &self,
        screening: &[ScreeningResult],
        conflicts: &[ConflictReport],
    ) -> Vec<String> {
        let mut warnings = Vec::new();

        let low_confidence: Vec<_> = conflicts
            .iter()
            .filter(|c| c.confidence < 0.8)
            .collect();
        if !low_confidence.is_empty() {
            warnings.push(format!(
                "{} conflict(s) have confidence below 80%",
                low_confidence.len()
            ));
        }

        let screening_only: usize = screening
            .iter()
            .filter(|s| {
                s.severity >= Severity::Moderate
                    && !conflicts.iter().any(|c| {
                        c.drug_a_name == s.drug_a_name && c.drug_b_name == s.drug_b_name
                    })
            })
            .count();
        if screening_only > 0 {
            warnings.push(format!(
                "{} pair(s) flagged at Tier 1 but not confirmed at Tier 2",
                screening_only
            ));
        }

        warnings
    }
}

// ──────────────────── Internal Data Types ────────────────────────────────

#[derive(Debug, Clone)]
struct RawConflict {
    drug_a_name: String,
    drug_b_name: String,
    severity: Severity,
    mechanism: String,
    description: String,
    trace: Vec<TraceStep>,
    guideline_violations: Vec<String>,
}

#[derive(Debug, Clone)]
struct EnzymeInteraction {
    drug_a: String,
    drug_b: String,
    enzymes: Vec<CypEnzyme>,
    severity: Severity,
    mechanism: String,
}

// ──────────────── Helper Functions ───────────────────────────────────────

fn generate_run_id() -> String {
    let now = chrono::Utc::now();
    format!("VR-{}", now.format("%Y%m%d-%H%M%S-%3f"))
}

fn current_timestamp() -> String {
    chrono::Utc::now().to_rfc3339()
}

fn estimate_volume_of_distribution(med: &ActiveMedication, patient: &PatientProfile) -> f64 {
    let base_vd = match med.canonical_name().as_str() {
        "warfarin" => 10.0,
        "metformin" => 63.0,
        "atorvastatin" => 381.0,
        "lisinopril" => 25.0,
        "amlodipine" => 21.0,
        "metoprolol" => 275.0,
        "omeprazole" => 21.0,
        "fluconazole" => 50.0,
        "clarithromycin" => 250.0,
        "simvastatin" => 260.0,
        "digoxin" => 475.0,
        "phenytoin" => 45.0,
        "carbamazepine" => 80.0,
        "amiodarone" => 5000.0,
        "cyclosporine" => 250.0,
        "tacrolimus" => 1300.0,
        _ => 50.0, // default
    };

    // Weight-based adjustment
    let weight_factor = patient.info.weight_kg / 70.0;
    base_vd * weight_factor
}

fn estimate_volume_of_distribution_by_name(name: &str, patient: &PatientProfile) -> f64 {
    let med = ActiveMedication::new(name, 1.0);
    estimate_volume_of_distribution(&med, patient)
}

fn estimate_elimination_rate(drug_name: &str) -> f64 {
    let name = drug_name.to_lowercase();
    match name.as_str() {
        "warfarin" => 0.693 / 40.0,       // t½ ≈ 40h
        "metformin" => 0.693 / 6.2,        // t½ ≈ 6.2h
        "atorvastatin" => 0.693 / 14.0,    // t½ ≈ 14h
        "lisinopril" => 0.693 / 12.0,      // t½ ≈ 12h
        "amlodipine" => 0.693 / 35.0,      // t½ ≈ 35h
        "metoprolol" => 0.693 / 3.5,       // t½ ≈ 3.5h
        "omeprazole" => 0.693 / 1.0,       // t½ ≈ 1h
        "fluconazole" => 0.693 / 30.0,     // t½ ≈ 30h
        "clarithromycin" => 0.693 / 5.0,   // t½ ≈ 5h
        "simvastatin" => 0.693 / 3.0,      // t½ ≈ 3h
        "digoxin" => 0.693 / 39.0,         // t½ ≈ 39h
        "phenytoin" => 0.693 / 22.0,       // t½ ≈ 22h
        "carbamazepine" => 0.693 / 16.0,   // t½ ≈ 16h
        "amiodarone" => 0.693 / 800.0,     // t½ ≈ 800h (very long)
        "cyclosporine" => 0.693 / 8.4,     // t½ ≈ 8.4h
        "tacrolimus" => 0.693 / 12.0,      // t½ ≈ 12h
        _ => 0.693 / 12.0,                 // default: t½ = 12h
    }
}

fn get_typical_dose(drug_name: &str) -> f64 {
    let name = drug_name.to_lowercase();
    match name.as_str() {
        "warfarin" => 5.0,
        "metformin" => 500.0,
        "atorvastatin" => 20.0,
        "lisinopril" => 10.0,
        "amlodipine" => 5.0,
        "metoprolol" => 50.0,
        "omeprazole" => 20.0,
        "fluconazole" => 200.0,
        "clarithromycin" => 500.0,
        "simvastatin" => 20.0,
        "digoxin" => 0.25,
        "phenytoin" => 300.0,
        "carbamazepine" => 400.0,
        "amiodarone" => 200.0,
        "cyclosporine" => 150.0,
        "tacrolimus" => 2.0,
        "aspirin" => 81.0,
        _ => 100.0,
    }
}

fn get_therapeutic_window(drug_name: &str) -> Option<TherapeuticWindow> {
    let name = drug_name.to_lowercase();
    match name.as_str() {
        "warfarin" => Some(TherapeuticWindow::new(1.0, 4.0, "mcg/mL")),
        "digoxin" => Some(TherapeuticWindow::new(0.0008, 0.002, "mcg/mL")),
        "phenytoin" => Some(TherapeuticWindow::new(10.0, 20.0, "mcg/mL")),
        "carbamazepine" => Some(TherapeuticWindow::new(4.0, 12.0, "mcg/mL")),
        "cyclosporine" => Some(TherapeuticWindow::new(0.1, 0.4, "mcg/mL")),
        "tacrolimus" => Some(TherapeuticWindow::new(0.005, 0.02, "mcg/mL")),
        "metformin" => Some(TherapeuticWindow::new(1.0, 5.0, "mcg/mL")),
        _ => None,
    }
}

fn compute_interval_concentration(
    dose_mg: f64,
    vd_liters: f64,
    ke_per_hour: f64,
    time_hours: f64,
    dosing_interval: f64,
) -> (f64, f64) {
    if vd_liters <= 0.0 || ke_per_hour <= 0.0 || dosing_interval <= 0.0 {
        return (0.0, 0.0);
    }

    // Multiple dose accumulation (superposition principle)
    let doses_given = (time_hours / dosing_interval).floor() as usize + 1;
    let time_since_last_dose = time_hours % dosing_interval;

    let c_single = (dose_mg / vd_liters) * (-ke_per_hour * time_since_last_dose).exp();
    let accumulation = if ke_per_hour * dosing_interval > 0.0 {
        let r = (-ke_per_hour * dosing_interval).exp();
        let n = doses_given.min(50) as f64;
        if r < 1.0 {
            (1.0 - r.powf(n)) / (1.0 - r)
        } else {
            n
        }
    } else {
        doses_given.min(50) as f64
    };

    let point_estimate = c_single * accumulation;

    // Add ±20% interval for uncertainty
    let lower = (point_estimate * 0.8).max(0.0);
    let upper = point_estimate * 1.2;

    (lower, upper)
}

fn get_known_enzyme_interactions() -> Vec<EnzymeInteraction> {
    vec![
        EnzymeInteraction {
            drug_a: "warfarin".to_string(),
            drug_b: "fluconazole".to_string(),
            enzymes: vec![CypEnzyme::CYP2C9, CypEnzyme::CYP3A4],
            severity: Severity::Major,
            mechanism: "Fluconazole inhibits CYP2C9/3A4, increasing warfarin exposure".to_string(),
        },
        EnzymeInteraction {
            drug_a: "simvastatin".to_string(),
            drug_b: "clarithromycin".to_string(),
            enzymes: vec![CypEnzyme::CYP3A4],
            severity: Severity::Contraindicated,
            mechanism: "Clarithromycin strongly inhibits CYP3A4, risk of rhabdomyolysis"
                .to_string(),
        },
        EnzymeInteraction {
            drug_a: "warfarin".to_string(),
            drug_b: "aspirin".to_string(),
            enzymes: vec![],
            severity: Severity::Major,
            mechanism: "Additive anticoagulant/antiplatelet effect, increased bleeding risk"
                .to_string(),
        },
        EnzymeInteraction {
            drug_a: "metformin".to_string(),
            drug_b: "fluconazole".to_string(),
            enzymes: vec![CypEnzyme::CYP2C9],
            severity: Severity::Moderate,
            mechanism: "Potential for hypoglycemia when combined".to_string(),
        },
        EnzymeInteraction {
            drug_a: "cyclosporine".to_string(),
            drug_b: "clarithromycin".to_string(),
            enzymes: vec![CypEnzyme::CYP3A4],
            severity: Severity::Major,
            mechanism: "CYP3A4 inhibition increases cyclosporine levels, nephrotoxicity risk"
                .to_string(),
        },
        EnzymeInteraction {
            drug_a: "phenytoin".to_string(),
            drug_b: "fluconazole".to_string(),
            enzymes: vec![CypEnzyme::CYP2C9, CypEnzyme::CYP2C19],
            severity: Severity::Major,
            mechanism: "CYP2C9/2C19 inhibition increases phenytoin toxicity risk".to_string(),
        },
        EnzymeInteraction {
            drug_a: "digoxin".to_string(),
            drug_b: "amiodarone".to_string(),
            enzymes: vec![],
            severity: Severity::Major,
            mechanism: "Amiodarone inhibits P-glycoprotein, increasing digoxin levels".to_string(),
        },
        EnzymeInteraction {
            drug_a: "tacrolimus".to_string(),
            drug_b: "fluconazole".to_string(),
            enzymes: vec![CypEnzyme::CYP3A4],
            severity: Severity::Major,
            mechanism: "CYP3A4 inhibition increases tacrolimus levels, nephrotoxicity risk"
                .to_string(),
        },
        EnzymeInteraction {
            drug_a: "carbamazepine".to_string(),
            drug_b: "clarithromycin".to_string(),
            enzymes: vec![CypEnzyme::CYP3A4],
            severity: Severity::Major,
            mechanism: "CYP3A4 inhibition increases carbamazepine toxicity risk".to_string(),
        },
        EnzymeInteraction {
            drug_a: "atorvastatin".to_string(),
            drug_b: "clarithromycin".to_string(),
            enzymes: vec![CypEnzyme::CYP3A4],
            severity: Severity::Major,
            mechanism: "CYP3A4 inhibition increases statin levels, rhabdomyolysis risk".to_string(),
        },
    ]
}

fn check_class_interaction(class_a: &str, class_b: &str) -> Option<Severity> {
    let a = class_a.to_lowercase();
    let b = class_b.to_lowercase();

    let dangerous_pairs: Vec<(&str, &str, Severity)> = vec![
        ("anticoagulant", "nsaid", Severity::Major),
        ("anticoagulant", "antiplatelet", Severity::Major),
        ("statin", "macrolide", Severity::Major),
        ("statin", "azole", Severity::Major),
        ("ace_inhibitor", "potassium_sparing_diuretic", Severity::Major),
        ("ssri", "maoi", Severity::Contraindicated),
        ("opioid", "benzodiazepine", Severity::Major),
        ("fluoroquinolone", "nsaid", Severity::Moderate),
        ("anticoagulant", "ssri", Severity::Moderate),
        ("beta_blocker", "calcium_channel_blocker", Severity::Moderate),
    ];

    for (ca, cb, severity) in &dangerous_pairs {
        if (a.contains(ca) && b.contains(cb)) || (a.contains(cb) && b.contains(ca)) {
            return Some(*severity);
        }
    }

    None
}

fn compute_screening_confidence(
    violations: &[String],
    enzyme_interaction: Option<&EnzymeInteraction>,
    conc_overlap: bool,
    patient: &PatientProfile,
) -> f64 {
    let mut confidence = 0.0_f64;

    // Guideline violation evidence
    if !violations.is_empty() {
        confidence += 0.4 + (violations.len() as f64 - 1.0) * 0.1;
    }

    // Enzyme interaction evidence
    if let Some(interaction) = enzyme_interaction {
        confidence += match interaction.severity {
            Severity::None => 0.0,
            Severity::Minor => 0.15,
            Severity::Moderate => 0.25,
            Severity::Major => 0.35,
            Severity::Contraindicated => 0.45,
        };
    }

    // Concentration overlap evidence
    if conc_overlap {
        confidence += 0.15;
    }

    // Patient risk factors
    if patient.is_elderly() {
        confidence += 0.05;
    }
    if patient.renal_function().requires_dose_adjustment() {
        confidence += 0.05;
    }

    confidence.min(1.0)
}

fn compute_tier2_confidence(raw: &RawConflict) -> f64 {
    let mut confidence = 0.7_f64; // base confidence for Tier 2 confirmed conflict

    // Guideline violations boost confidence
    confidence += raw.guideline_violations.len() as f64 * 0.05;

    // Severity-based adjustment
    confidence += match raw.severity {
        Severity::None => 0.0,
        Severity::Minor => 0.0,
        Severity::Moderate => 0.05,
        Severity::Major => 0.1,
        Severity::Contraindicated => 0.15,
    };

    // Trace evidence
    let violation_steps = raw.trace.iter().filter(|t| t.violation.is_some()).count();
    if violation_steps > 0 {
        confidence += (violation_steps as f64 * 0.02).min(0.1);
    }

    confidence.min(1.0)
}

fn derive_clinical_consequence(
    drug_a: &str,
    drug_b: &str,
    mechanism: &str,
    severity: Severity,
) -> String {
    let mechanism_lower = mechanism.to_lowercase();

    if mechanism_lower.contains("bleeding") || mechanism_lower.contains("anticoag") {
        return format!(
            "Increased risk of bleeding when {} is combined with {}. \
             Monitor for signs of hemorrhage including GI bleeding, \
             bruising, and prolonged bleeding from cuts.",
            drug_a, drug_b
        );
    }

    if mechanism_lower.contains("rhabdomyolysis") || mechanism_lower.contains("statin") {
        return format!(
            "Risk of rhabdomyolysis (muscle breakdown) when {} is combined with {}. \
             Monitor for muscle pain, weakness, and dark urine. \
             Check CK levels if symptoms develop.",
            drug_a, drug_b
        );
    }

    if mechanism_lower.contains("nephrotox") || mechanism_lower.contains("kidney") {
        return format!(
            "Increased risk of nephrotoxicity when {} is combined with {}. \
             Monitor renal function (SCr, BUN) regularly.",
            drug_a, drug_b
        );
    }

    if mechanism_lower.contains("qt") || mechanism_lower.contains("arrhythmia") {
        return format!(
            "Risk of QT prolongation and cardiac arrhythmia when {} is combined with {}. \
             Monitor ECG and electrolytes.",
            drug_a, drug_b
        );
    }

    match severity {
        Severity::Contraindicated => format!(
            "This combination of {} and {} is contraindicated. \
             The clinical risk outweighs any potential benefit.",
            drug_a, drug_b
        ),
        Severity::Major => format!(
            "Significant risk of adverse effects when {} is combined with {}. \
             Close monitoring required if combination is necessary.",
            drug_a, drug_b
        ),
        Severity::Moderate => format!(
            "Possible increased risk of adverse effects when {} and {} are combined. \
             Clinical monitoring recommended.",
            drug_a, drug_b
        ),
        Severity::Minor | Severity::None => format!(
            "Minor interaction between {} and {}. \
             Clinically significant effects are unlikely but awareness is recommended.",
            drug_a, drug_b
        ),
    }
}

fn generate_recommendation_text(
    conflict: &ConflictReport,
    patient: &PatientProfile,
) -> (String, String, Option<String>, Option<String>) {
    let drug_a = &conflict.drug_a_name;
    let drug_b = &conflict.drug_b_name;

    let summary = match conflict.severity {
        Severity::Contraindicated => {
            format!("Discontinue {} or {} — combination is contraindicated", drug_a, drug_b)
        }
        Severity::Major => {
            format!("Review combination of {} and {} — major interaction risk", drug_a, drug_b)
        }
        Severity::Moderate => {
            format!("Monitor {} + {} combination closely", drug_a, drug_b)
        }
        Severity::Minor | Severity::None => {
            format!("Be aware of {} + {} interaction", drug_a, drug_b)
        }
    };

    let rationale = format!("{} (confidence: {:.0}%)", conflict.mechanism, conflict.confidence * 100.0);

    let alternative = match conflict.severity {
        Severity::Contraindicated | Severity::Major => {
            Some(format!("Consider alternative to {} or {}", drug_a, drug_b))
        }
        _ => None,
    };

    let monitoring = match conflict.severity {
        Severity::Contraindicated => Some("Immediate clinical review required".to_string()),
        Severity::Major => Some("Close monitoring required; check labs regularly".to_string()),
        Severity::Moderate => Some("Periodic monitoring recommended".to_string()),
        Severity::Minor | Severity::None => None,
    };

    (summary, rationale, alternative, monitoring)
}

fn categorize_recommendation(conflict: &ConflictReport) -> String {
    let mech = conflict.mechanism.to_lowercase();
    if mech.contains("cyp") || mech.contains("enzyme") {
        "CYP enzyme interaction".to_string()
    } else if mech.contains("anticoag") || mech.contains("bleeding") {
        "Bleeding risk".to_string()
    } else if mech.contains("nephro") || mech.contains("kidney") {
        "Nephrotoxicity risk".to_string()
    } else if mech.contains("rhabdo") || mech.contains("muscle") {
        "Myotoxicity risk".to_string()
    } else if mech.contains("qt") || mech.contains("cardiac") {
        "Cardiac risk".to_string()
    } else {
        "Drug interaction".to_string()
    }
}

fn get_drug_enzyme_effect(drug_name: &str, enzyme: &CypEnzyme) -> Option<(String, f64)> {
    match (drug_name, enzyme) {
        ("fluconazole", CypEnzyme::CYP2C9) => {
            Some(("Strong inhibitor".to_string(), 0.3))
        }
        ("fluconazole", CypEnzyme::CYP3A4) => {
            Some(("Moderate inhibitor".to_string(), 0.5))
        }
        ("clarithromycin", CypEnzyme::CYP3A4) => {
            Some(("Strong inhibitor".to_string(), 0.2))
        }
        ("carbamazepine", CypEnzyme::CYP3A4) => {
            Some(("Strong inducer".to_string(), 2.0))
        }
        ("carbamazepine", CypEnzyme::CYP2C9) => {
            Some(("Moderate inducer".to_string(), 1.5))
        }
        ("omeprazole", CypEnzyme::CYP2C19) => {
            Some(("Moderate inhibitor".to_string(), 0.5))
        }
        ("amiodarone", CypEnzyme::CYP2D6) => {
            Some(("Moderate inhibitor".to_string(), 0.4))
        }
        ("amiodarone", CypEnzyme::CYP3A4) => {
            Some(("Moderate inhibitor".to_string(), 0.5))
        }
        _ => None,
    }
}

// ────────────────────────────── Tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AppConfig;
    use crate::input::{ActiveMedication, ClinicalCondition, PatientProfile};
    use guardpharma_types::{DrugRoute, PatientInfo, Sex};

    fn test_config() -> AppConfig {
        AppConfig::default()
    }

    fn test_patient() -> PatientProfile {
        PatientProfile::default().with_medications(vec![
            ActiveMedication::new("Warfarin", 5.0).with_class("anticoagulant"),
            ActiveMedication::new("Aspirin", 81.0).with_class("antiplatelet"),
        ])
    }

    fn test_guidelines() -> Vec<GuidelineDocument> {
        let mut g = GuidelineDocument::new("Test Guideline");
        let mut rule = GuidelineRule::new("R001", "Avoid warfarin + aspirin", Severity::Major);
        rule.affected_drugs = vec!["warfarin".to_string(), "aspirin".to_string()];
        g.rules.push(rule);
        vec![g]
    }

    #[test]
    fn test_phase_names() {
        assert_eq!(Phase::Loading.name(), "Loading");
        assert_eq!(Phase::Tier1Screening.name(), "Tier 1 Screening");
        assert_eq!(Phase::all().len(), 7);
    }

    #[test]
    fn test_phase_display() {
        let phase = Phase::Tier2Verification;
        assert_eq!(format!("{}", phase), "Tier 2 Verification");
    }

    #[test]
    fn test_pipeline_timer() {
        let mut timer = PipelineTimer::new();
        timer.start_phase("Test Phase");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let timings = timer.into_timings();
        assert_eq!(timings.len(), 1);
        assert!(timings[0].duration_ms >= 5.0);
    }

    #[test]
    fn test_pipeline_timer_multiple() {
        let mut timer = PipelineTimer::new();
        timer.start_phase("Phase 1");
        timer.start_phase("Phase 2");
        timer.start_phase("Phase 3");
        let timings = timer.into_timings();
        assert_eq!(timings.len(), 3);
    }

    #[test]
    fn test_generate_run_id() {
        let id = generate_run_id();
        assert!(id.starts_with("VR-"));
        assert!(id.len() > 10);
    }

    #[test]
    fn test_current_timestamp() {
        let ts = current_timestamp();
        assert!(ts.contains('T'));
    }

    #[test]
    fn test_estimate_volume_of_distribution() {
        let med = ActiveMedication::new("Warfarin", 5.0);
        let patient = PatientProfile::default();
        let vd = estimate_volume_of_distribution(&med, &patient);
        assert!(vd > 0.0);
    }

    #[test]
    fn test_estimate_elimination_rate() {
        let ke = estimate_elimination_rate("warfarin");
        assert!(ke > 0.0);
        assert!(ke < 1.0); // warfarin has long half-life

        let ke_fast = estimate_elimination_rate("omeprazole");
        assert!(ke_fast > ke); // omeprazole has shorter half-life
    }

    #[test]
    fn test_get_typical_dose() {
        assert_eq!(get_typical_dose("warfarin"), 5.0);
        assert_eq!(get_typical_dose("metformin"), 500.0);
        assert_eq!(get_typical_dose("unknown_drug"), 100.0);
    }

    #[test]
    fn test_get_therapeutic_window() {
        let tw = get_therapeutic_window("warfarin");
        assert!(tw.is_some());
        let tw = tw.unwrap();
        assert!(tw.lower < tw.upper);

        assert!(get_therapeutic_window("unknown_drug").is_none());
    }

    #[test]
    fn test_compute_interval_concentration() {
        let (lo, hi) = compute_interval_concentration(100.0, 50.0, 0.1, 0.0, 24.0);
        assert!(lo > 0.0);
        assert!(hi > lo);

        let (lo2, hi2) = compute_interval_concentration(100.0, 50.0, 0.1, 12.0, 24.0);
        assert!(lo2 < lo); // concentration decreases after peak
    }

    #[test]
    fn test_compute_interval_concentration_zero_vd() {
        let (lo, hi) = compute_interval_concentration(100.0, 0.0, 0.1, 0.0, 24.0);
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 0.0);
    }

    #[test]
    fn test_known_enzyme_interactions() {
        let interactions = get_known_enzyme_interactions();
        assert!(!interactions.is_empty());

        let warfarin_fluconazole = interactions
            .iter()
            .find(|i| i.drug_a == "warfarin" && i.drug_b == "fluconazole");
        assert!(warfarin_fluconazole.is_some());
    }

    #[test]
    fn test_check_class_interaction() {
        let result = check_class_interaction("anticoagulant", "nsaid");
        assert!(result.is_some());
        assert_eq!(result.unwrap(), Severity::Major);

        let result = check_class_interaction("ssri", "maoi");
        assert!(result.is_some());
        assert_eq!(result.unwrap(), Severity::Contraindicated);

        let result = check_class_interaction("statin", "statin");
        assert!(result.is_none());
    }

    #[test]
    fn test_compute_screening_confidence() {
        let conf = compute_screening_confidence(
            &["violation1".to_string()],
            None,
            true,
            &PatientProfile::default(),
        );
        assert!(conf > 0.0);
        assert!(conf <= 1.0);

        // More evidence = higher confidence
        let conf2 = compute_screening_confidence(
            &["v1".to_string(), "v2".to_string()],
            None,
            true,
            &PatientProfile::default(),
        );
        assert!(conf2 > conf);
    }

    #[test]
    fn test_derive_clinical_consequence() {
        let consequence =
            derive_clinical_consequence("Warfarin", "Aspirin", "Additive anticoagulation", Severity::Major);
        assert!(consequence.contains("bleeding"));
    }

    #[test]
    fn test_categorize_recommendation() {
        let conflict = ConflictReport {
            drug_a_name: "Warfarin".to_string(),
            drug_b_name: "Fluconazole".to_string(),
            severity: Severity::Major,
            confidence: 0.9,
            mechanism: "CYP2C9 inhibition".to_string(),
            description: String::new(),
            clinical_consequence: String::new(),
            trace: vec![],
        };
        assert_eq!(categorize_recommendation(&conflict), "CYP enzyme interaction");
    }

    #[test]
    fn test_pipeline_orchestrator_screening_only() {
        let config = test_config();
        let orchestrator = PipelineOrchestrator::new(&config);
        let patient = test_patient();
        let guidelines = test_guidelines();

        let result = orchestrator.run_screening_only(&guidelines, &patient);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(!output.results.is_empty());
    }

    #[test]
    fn test_pipeline_orchestrator_full() {
        let config = test_config();
        let orchestrator = PipelineOrchestrator::new(&config);
        let patient = test_patient();
        let guidelines = test_guidelines();

        let result = orchestrator.run_full_pipeline(&guidelines, &patient);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.run_id.starts_with("VR-"));
    }

    #[test]
    fn test_pipeline_orchestrator_no_meds() {
        let config = test_config();
        let orchestrator = PipelineOrchestrator::new(&config);
        let patient = PatientProfile::default();
        let guidelines = test_guidelines();

        let result = orchestrator.run_full_pipeline(&guidelines, &patient);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(matches!(output.verdict, VerificationVerdict::Safe));
    }

    #[test]
    fn test_pipeline_analysis_only() {
        let config = test_config();
        let orchestrator = PipelineOrchestrator::new(&config);
        let patient = test_patient();
        let guidelines = test_guidelines();

        let result = orchestrator.run_analysis_only(&guidelines, &patient, true, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_enzyme_details_computation() {
        let config = test_config();
        let orchestrator = PipelineOrchestrator::new(&config);
        let patient = PatientProfile::default().with_medications(vec![
            ActiveMedication::new("Fluconazole", 200.0),
        ]);

        let details = orchestrator.compute_enzyme_details(&patient).unwrap();
        // Fluconazole inhibits CYP2C9 and CYP3A4
        assert!(!details.is_empty());
    }

    #[test]
    fn test_pk_traces_computation() {
        let config = test_config();
        let orchestrator = PipelineOrchestrator::new(&config);
        let patient = PatientProfile::default().with_medications(vec![
            ActiveMedication::new("Warfarin", 5.0),
        ]);

        let traces = orchestrator.compute_pk_traces(&patient).unwrap();
        assert_eq!(traces.len(), 1);
        assert!(!traces[0].time_points.is_empty());
        assert!(traces[0].therapeutic_window.is_some());
    }

    #[test]
    fn test_verification_verdict_serialization() {
        let v = VerificationVerdict::Safe;
        let json = serde_json::to_string(&v).unwrap();
        assert!(json.contains("Safe"));

        let v2 = VerificationVerdict::ConflictsFound { count: 3 };
        let json2 = serde_json::to_string(&v2).unwrap();
        assert!(json2.contains("3"));
    }

    #[test]
    fn test_trace_step_creation() {
        let step = TraceStep {
            time_hours: 4.0,
            description: "Peak concentration".to_string(),
            concentration_a: Some((1.0, 2.0)),
            concentration_b: None,
            violation: None,
        };
        assert_eq!(step.time_hours, 4.0);
        assert!(step.concentration_a.is_some());
        assert!(step.violation.is_none());
    }

    #[test]
    fn test_recommendation_creation() {
        let rec = Recommendation {
            summary: "Test recommendation".to_string(),
            priority: Severity::Major,
            category: "Testing".to_string(),
            rationale: "For testing".to_string(),
            alternative: None,
            monitoring: None,
            affected_drugs: vec!["drug_a".to_string()],
        };
        let json = serde_json::to_string(&rec).unwrap();
        assert!(json.contains("Test recommendation"));
    }

    #[test]
    fn test_safety_certificate_creation() {
        let cert = SafetyCertificate {
            certificate_id: "CERT-001".to_string(),
            run_id: "VR-001".to_string(),
            timestamp: "2024-01-01".to_string(),
            patient_id: "PT-001".to_string(),
            verdict: VerificationVerdict::Safe,
            guidelines_checked: 5,
            drug_pairs_checked: 10,
            conflicts_found: 0,
            tier1_completed: true,
            tier2_completed: true,
            warnings: vec![],
        };
        assert_eq!(cert.conflicts_found, 0);
        assert!(cert.tier1_completed);
    }

    #[test]
    fn test_get_drug_enzyme_effect() {
        let effect = get_drug_enzyme_effect("fluconazole", &CypEnzyme::CYP2C9);
        assert!(effect.is_some());
        let (desc, factor) = effect.unwrap();
        assert!(desc.contains("inhibitor"));
        assert!(factor < 1.0);

        let effect = get_drug_enzyme_effect("carbamazepine", &CypEnzyme::CYP3A4);
        assert!(effect.is_some());
        let (desc, factor) = effect.unwrap();
        assert!(desc.contains("inducer"));
        assert!(factor > 1.0);

        assert!(get_drug_enzyme_effect("unknown", &CypEnzyme::CYP3A4).is_none());
    }

    #[test]
    fn test_generate_recommendation_text() {
        let conflict = ConflictReport {
            drug_a_name: "Warfarin".to_string(),
            drug_b_name: "Aspirin".to_string(),
            severity: Severity::Major,
            confidence: 0.9,
            mechanism: "Additive anticoagulation".to_string(),
            description: String::new(),
            clinical_consequence: String::new(),
            trace: vec![],
        };
        let patient = PatientProfile::default();
        let (summary, rationale, alt, monitoring) =
            generate_recommendation_text(&conflict, &patient);

        assert!(summary.contains("Warfarin"));
        assert!(!rationale.is_empty());
        assert!(alt.is_some());
        assert!(monitoring.is_some());
    }
}
