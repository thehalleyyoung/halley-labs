//! Ready-made clinical guideline templates.
//!
//! Each template implements [`GuidelineTemplate`] and produces a complete
//! [`GuidelineDocument`] with realistic decision points, branches, actions,
//! safety constraints, and monitoring requirements.

use crate::format::{
    Branch, ComparisonOp, ConstraintSeverity, DecisionPoint, DoseSpec, EvidenceLevel,
    GuidelineAction, GuidelineDocument, GuidelineGuard, GuidelineMetadata, MedicationSpec,
    MonitoringRequirement, SafetyConstraint, TransitionRule, Urgency,
};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A template that can produce a complete [`GuidelineDocument`].
pub trait GuidelineTemplate {
    /// Human-readable name of the template.
    fn name(&self) -> &str;
    /// Clinical condition covered.
    fn condition(&self) -> &str;
    /// Build the full guideline document.
    fn build(&self) -> GuidelineDocument;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn dp(
    id: &str,
    label: &str,
    desc: Option<&str>,
    branches: Vec<Branch>,
    initial: bool,
    terminal: bool,
) -> DecisionPoint {
    DecisionPoint {
        id: id.into(),
        label: label.into(),
        description: desc.map(String::from),
        branches,
        is_initial: initial,
        is_terminal: terminal,
        invariants: vec![],
        urgency: None,
    }
}

fn br(id: &str, guard: GuidelineGuard, actions: Vec<GuidelineAction>, target: &str) -> Branch {
    Branch {
        id: id.into(),
        guard,
        actions,
        target: target.into(),
        priority: 0,
        evidence_level: Some(EvidenceLevel::High),
        notes: None,
    }
}

fn lab_guard(test: &str, op: ComparisonOp, val: f64) -> GuidelineGuard {
    GuidelineGuard::LabThreshold {
        test_name: test.into(),
        operator: op,
        value: val,
        unit: None,
    }
}

fn clin_guard(param: &str, op: ComparisonOp, val: f64) -> GuidelineGuard {
    GuidelineGuard::ClinicalPredicate {
        parameter: param.into(),
        operator: op,
        threshold: val,
        unit: None,
    }
}

fn dx_guard(diagnosis: &str) -> GuidelineGuard {
    GuidelineGuard::DiagnosisPresent {
        diagnosis: diagnosis.into(),
    }
}

fn med_active(med: &str) -> GuidelineGuard {
    GuidelineGuard::MedicationActive {
        medication: med.into(),
    }
}

fn start_med(med: &str, dose: f64, unit: &str, freq: &str, route: &str) -> GuidelineAction {
    GuidelineAction::StartMedication {
        medication: med.into(),
        dose: DoseSpec::new(dose, unit).with_frequency(freq),
        route: route.into(),
        reason: None,
    }
}

fn adjust_dose(med: &str, dose: f64, unit: &str) -> GuidelineAction {
    GuidelineAction::AdjustDose {
        medication: med.into(),
        new_dose: DoseSpec::new(dose, unit),
        reason: None,
    }
}

fn stop_med(med: &str) -> GuidelineAction {
    GuidelineAction::StopMedication {
        medication: med.into(),
        taper: None,
        reason: None,
    }
}

fn reassess(days: u32, criteria: Vec<&str>) -> GuidelineAction {
    GuidelineAction::Reassess {
        interval_days: days,
        criteria: criteria.into_iter().map(String::from).collect(),
    }
}

fn order_lab(test: &str) -> GuidelineAction {
    GuidelineAction::OrderLab {
        test_name: test.into(),
        urgency: None,
        repeat_interval_days: None,
    }
}

fn refer(specialty: &str) -> GuidelineAction {
    GuidelineAction::Refer {
        specialty: specialty.into(),
        urgency: None,
        reason: None,
    }
}

fn monitor_req(id: &str, param: &str, interval: u32) -> MonitoringRequirement {
    MonitoringRequirement {
        id: id.into(),
        parameter: param.into(),
        interval_days: interval,
        duration_days: None,
        target_range: None,
        alert_threshold: None,
        applies_to_states: vec![],
    }
}

fn safety(id: &str, desc: &str, guard: GuidelineGuard) -> SafetyConstraint {
    SafetyConstraint {
        id: id.into(),
        description: desc.into(),
        guard,
        severity: ConstraintSeverity::Critical,
        applies_to: vec![],
    }
}

// ---------------------------------------------------------------------------
// Type 2 Diabetes
// ---------------------------------------------------------------------------

pub struct Type2DiabetesTemplate;

impl GuidelineTemplate for Type2DiabetesTemplate {
    fn name(&self) -> &str {
        "ADA Type 2 Diabetes Management"
    }
    fn condition(&self) -> &str {
        "Type 2 Diabetes Mellitus"
    }
    fn build(&self) -> GuidelineDocument {
        let mut doc = GuidelineDocument::new(self.name());
        doc.metadata.condition = Some(self.condition().into());
        doc.metadata.source_organization = Some("ADA".into());
        doc.metadata.evidence_level = Some(EvidenceLevel::High);
        doc.metadata.tags = vec!["diabetes".into(), "endocrine".into(), "metabolic".into()];

        // 1. Initial assessment
        doc.add_decision_point(dp(
            "t2d_initial",
            "Initial T2D Assessment",
            Some("Assess HbA1c, comorbidities, renal function"),
            vec![
                br("t2d_mild", lab_guard("HbA1c", ComparisonOp::Lt, 7.5),
                    vec![start_med("metformin", 500.0, "mg", "BID", "oral")],
                    "t2d_metformin_mono"),
                br("t2d_moderate",
                    GuidelineGuard::And(vec![
                        lab_guard("HbA1c", ComparisonOp::Ge, 7.5),
                        lab_guard("HbA1c", ComparisonOp::Lt, 9.0),
                    ]),
                    vec![start_med("metformin", 1000.0, "mg", "BID", "oral")],
                    "t2d_dual_eval"),
                br("t2d_severe", lab_guard("HbA1c", ComparisonOp::Ge, 9.0),
                    vec![GuidelineAction::CombinationTherapy {
                        medications: vec![
                            MedicationSpec { name: "metformin".into(), dose: DoseSpec::new(1000.0, "mg").with_frequency("BID"), route: Some("oral".into()) },
                            MedicationSpec { name: "insulin_glargine".into(), dose: DoseSpec::new(10.0, "units").with_frequency("QHS"), route: Some("subcutaneous".into()) },
                        ],
                        reason: Some("Severe hyperglycemia".into()),
                    }],
                    "t2d_insulin_titrate"),
            ],
            true, false,
        ));

        // 2. Metformin mono
        doc.add_decision_point(dp(
            "t2d_metformin_mono", "Metformin Monotherapy", Some("3-month follow-up"),
            vec![
                br("t2d_on_target", lab_guard("HbA1c", ComparisonOp::Lt, 7.0),
                    vec![reassess(90, vec!["HbA1c", "renal"])], "t2d_maintenance"),
                br("t2d_above_target", lab_guard("HbA1c", ComparisonOp::Ge, 7.0),
                    vec![], "t2d_dual_eval"),
                br("t2d_intolerant", dx_guard("metformin_intolerance"),
                    vec![stop_med("metformin"), start_med("sitagliptin", 100.0, "mg", "QD", "oral")],
                    "t2d_alt_mono"),
            ],
            false, false,
        ));

        // 3. Dual therapy evaluation
        doc.add_decision_point(dp(
            "t2d_dual_eval", "Dual Therapy Evaluation", Some("Select second agent"),
            vec![
                br("t2d_add_sglt2",
                    GuidelineGuard::And(vec![clin_guard("eGFR", ComparisonOp::Ge, 30.0), dx_guard("heart_failure")]),
                    vec![start_med("empagliflozin", 10.0, "mg", "QD", "oral")],
                    "t2d_dual_monitor"),
                br("t2d_add_glp1", dx_guard("atherosclerotic_cvd"),
                    vec![start_med("semaglutide", 0.25, "mg", "weekly", "subcutaneous")],
                    "t2d_dual_monitor"),
                br("t2d_add_dpp4", GuidelineGuard::True,
                    vec![start_med("sitagliptin", 100.0, "mg", "QD", "oral")],
                    "t2d_dual_monitor"),
            ],
            false, false,
        ));

        // 4. Dual therapy monitoring
        doc.add_decision_point(dp(
            "t2d_dual_monitor", "Dual Therapy Monitoring", Some("Re-check HbA1c at 3 months"),
            vec![
                br("t2d_dual_ctrl", lab_guard("HbA1c", ComparisonOp::Lt, 7.0),
                    vec![reassess(90, vec!["HbA1c"])], "t2d_maintenance"),
                br("t2d_dual_unctrl", lab_guard("HbA1c", ComparisonOp::Ge, 7.0),
                    vec![], "t2d_triple_eval"),
            ],
            false, false,
        ));

        // 5. Triple therapy evaluation
        doc.add_decision_point(dp(
            "t2d_triple_eval", "Triple Therapy Evaluation", None,
            vec![
                br("t2d_add_insulin",
                    lab_guard("HbA1c", ComparisonOp::Ge, 9.0),
                    vec![start_med("insulin_glargine", 10.0, "units", "QHS", "subcutaneous")],
                    "t2d_insulin_titrate"),
                br("t2d_add_third_oral", GuidelineGuard::True,
                    vec![start_med("pioglitazone", 15.0, "mg", "QD", "oral")],
                    "t2d_triple_monitor"),
            ],
            false, false,
        ));

        // 6. Triple therapy monitoring
        doc.add_decision_point(dp(
            "t2d_triple_monitor", "Triple Therapy Monitoring", None,
            vec![
                br("t2d_triple_ctrl", lab_guard("HbA1c", ComparisonOp::Lt, 7.0),
                    vec![reassess(90, vec!["HbA1c"])], "t2d_maintenance"),
                br("t2d_triple_fail", lab_guard("HbA1c", ComparisonOp::Ge, 7.0),
                    vec![], "t2d_insulin_titrate"),
            ],
            false, false,
        ));

        // 7. Insulin titration
        doc.add_decision_point(dp(
            "t2d_insulin_titrate", "Insulin Titration", Some("Titrate basal insulin"),
            vec![
                br("t2d_fg_high", lab_guard("fasting_glucose", ComparisonOp::Gt, 130.0),
                    vec![adjust_dose("insulin_glargine", 2.0, "units")], "t2d_insulin_titrate"),
                br("t2d_fg_low", lab_guard("fasting_glucose", ComparisonOp::Lt, 70.0),
                    vec![adjust_dose("insulin_glargine", -4.0, "units")], "t2d_insulin_titrate"),
                br("t2d_fg_ok",
                    GuidelineGuard::And(vec![
                        lab_guard("fasting_glucose", ComparisonOp::Ge, 70.0),
                        lab_guard("fasting_glucose", ComparisonOp::Le, 130.0),
                    ]),
                    vec![reassess(90, vec!["HbA1c", "fasting_glucose"])], "t2d_maintenance"),
            ],
            false, false,
        ));

        // 8. Alt monotherapy
        doc.add_decision_point(dp(
            "t2d_alt_mono", "Alternative Monotherapy", None,
            vec![
                br("t2d_alt_ok", lab_guard("HbA1c", ComparisonOp::Lt, 7.0),
                    vec![reassess(90, vec!["HbA1c"])], "t2d_maintenance"),
                br("t2d_alt_fail", lab_guard("HbA1c", ComparisonOp::Ge, 7.0),
                    vec![], "t2d_dual_eval"),
            ],
            false, false,
        ));

        // 9. Renal check
        doc.add_decision_point(dp(
            "t2d_renal_check", "Renal Function Check", None,
            vec![
                br("t2d_renal_ok", clin_guard("eGFR", ComparisonOp::Ge, 30.0),
                    vec![reassess(180, vec!["eGFR"])], "t2d_maintenance"),
                br("t2d_renal_low", clin_guard("eGFR", ComparisonOp::Lt, 30.0),
                    vec![stop_med("metformin"), refer("nephrology")],
                    "t2d_renal_adjust"),
            ],
            false, false,
        ));

        // 10. Renal adjustment
        doc.add_decision_point(dp(
            "t2d_renal_adjust", "Renal Dose Adjustment", None,
            vec![
                br("t2d_renal_switch", GuidelineGuard::True,
                    vec![start_med("linagliptin", 5.0, "mg", "QD", "oral")],
                    "t2d_dual_monitor"),
            ],
            false, false,
        ));

        // 11. Maintenance
        doc.add_decision_point(dp(
            "t2d_maintenance", "Maintenance", Some("Stable glycemic control"),
            vec![br("t2d_continue", GuidelineGuard::True,
                vec![reassess(180, vec!["HbA1c", "lipids", "renal"])],
                "t2d_maintenance")],
            false, true,
        ));

        // Safety
        doc.add_safety_constraint(safety("t2d_no_met_low_egfr",
            "Metformin CI when eGFR < 30",
            GuidelineGuard::And(vec![med_active("metformin"), clin_guard("eGFR", ComparisonOp::Lt, 30.0)])));
        doc.add_safety_constraint(safety("t2d_hypo_monitor",
            "Monitor hypoglycemia with insulin",
            med_active("insulin_glargine")));

        // Monitoring
        doc.add_monitoring(monitor_req("t2d_hba1c", "HbA1c", 90));
        doc.add_monitoring(monitor_req("t2d_egfr", "eGFR", 180));
        doc.add_monitoring(monitor_req("t2d_lipids", "lipids", 365));

        doc
    }
}

// ---------------------------------------------------------------------------
// Hypertension
// ---------------------------------------------------------------------------

pub struct HypertensionTemplate;

impl GuidelineTemplate for HypertensionTemplate {
    fn name(&self) -> &str { "ACC/AHA Hypertension Management" }
    fn condition(&self) -> &str { "Hypertension" }
    fn build(&self) -> GuidelineDocument {
        let mut doc = GuidelineDocument::new(self.name());
        doc.metadata.condition = Some(self.condition().into());
        doc.metadata.source_organization = Some("ACC/AHA".into());
        doc.metadata.tags = vec!["hypertension".into(), "cardiovascular".into()];

        doc.add_decision_point(dp(
            "htn_assess", "BP Assessment", Some("Classify blood pressure stage"),
            vec![
                br("htn_elevated",
                    GuidelineGuard::And(vec![clin_guard("systolic_bp", ComparisonOp::Ge, 120.0), clin_guard("systolic_bp", ComparisonOp::Lt, 130.0)]),
                    vec![GuidelineAction::LifestyleModification { category: "diet".into(), description: "DASH diet, sodium < 2300mg/day".into() }],
                    "htn_lifestyle"),
                br("htn_stage1",
                    GuidelineGuard::And(vec![clin_guard("systolic_bp", ComparisonOp::Ge, 130.0), clin_guard("systolic_bp", ComparisonOp::Lt, 140.0)]),
                    vec![start_med("lisinopril", 10.0, "mg", "QD", "oral")],
                    "htn_mono_monitor"),
                br("htn_stage2",
                    clin_guard("systolic_bp", ComparisonOp::Ge, 140.0),
                    vec![GuidelineAction::CombinationTherapy {
                        medications: vec![
                            MedicationSpec { name: "lisinopril".into(), dose: DoseSpec::new(20.0, "mg").with_frequency("QD"), route: Some("oral".into()) },
                            MedicationSpec { name: "amlodipine".into(), dose: DoseSpec::new(5.0, "mg").with_frequency("QD"), route: Some("oral".into()) },
                        ],
                        reason: Some("Stage 2 HTN".into()),
                    }],
                    "htn_dual_monitor"),
                br("htn_crisis",
                    clin_guard("systolic_bp", ComparisonOp::Ge, 180.0),
                    vec![GuidelineAction::EmergencyIntervention {
                        description: "Hypertensive crisis management".into(),
                        medications: vec![MedicationSpec {
                            name: "nicardipine".into(),
                            dose: DoseSpec::new(5.0, "mg/hr").with_frequency("IV continuous"),
                            route: Some("IV".into()),
                        }],
                    }],
                    "htn_emergency"),
            ],
            true, false,
        ));

        doc.add_decision_point(dp("htn_lifestyle", "Lifestyle Only", None,
            vec![br("htn_recheck", GuidelineGuard::True, vec![reassess(90, vec!["blood_pressure"])], "htn_assess")],
            false, false));

        doc.add_decision_point(dp("htn_mono_monitor", "Monotherapy Monitoring", None,
            vec![
                br("htn_mono_ok", clin_guard("systolic_bp", ComparisonOp::Lt, 130.0),
                    vec![reassess(90, vec!["blood_pressure"])], "htn_stable"),
                br("htn_mono_fail", clin_guard("systolic_bp", ComparisonOp::Ge, 130.0),
                    vec![adjust_dose("lisinopril", 20.0, "mg")], "htn_dual_eval"),
            ],
            false, false));

        doc.add_decision_point(dp("htn_dual_eval", "Dual Therapy Evaluation", None,
            vec![
                br("htn_add_ccb", GuidelineGuard::Not(Box::new(dx_guard("peripheral_edema"))),
                    vec![start_med("amlodipine", 5.0, "mg", "QD", "oral")], "htn_dual_monitor"),
                br("htn_add_thiazide", GuidelineGuard::True,
                    vec![start_med("chlorthalidone", 12.5, "mg", "QD", "oral")], "htn_dual_monitor"),
            ],
            false, false));

        doc.add_decision_point(dp("htn_dual_monitor", "Dual Therapy Monitoring", None,
            vec![
                br("htn_dual_ok", clin_guard("systolic_bp", ComparisonOp::Lt, 130.0),
                    vec![reassess(90, vec!["blood_pressure", "electrolytes"])], "htn_stable"),
                br("htn_dual_fail", clin_guard("systolic_bp", ComparisonOp::Ge, 130.0),
                    vec![], "htn_triple"),
            ],
            false, false));

        doc.add_decision_point(dp("htn_triple", "Triple Therapy", None,
            vec![
                br("htn_add_third",
                    GuidelineGuard::True,
                    vec![start_med("chlorthalidone", 25.0, "mg", "QD", "oral")],
                    "htn_triple_monitor"),
            ],
            false, false));

        doc.add_decision_point(dp("htn_triple_monitor", "Triple Monitoring", None,
            vec![
                br("htn_triple_ok", clin_guard("systolic_bp", ComparisonOp::Lt, 130.0),
                    vec![reassess(90, vec!["blood_pressure"])], "htn_stable"),
                br("htn_triple_fail", clin_guard("systolic_bp", ComparisonOp::Ge, 130.0),
                    vec![refer("hypertension_specialist")], "htn_resistant"),
            ],
            false, false));

        doc.add_decision_point(dp("htn_resistant", "Resistant HTN", None,
            vec![br("htn_spiro", GuidelineGuard::True,
                vec![start_med("spironolactone", 25.0, "mg", "QD", "oral"), order_lab("aldosterone")],
                "htn_resistant_monitor")],
            false, false));

        doc.add_decision_point(dp("htn_resistant_monitor", "Resistant HTN Monitoring", None,
            vec![
                br("htn_res_ok", clin_guard("systolic_bp", ComparisonOp::Lt, 130.0),
                    vec![], "htn_stable"),
                br("htn_res_fail", GuidelineGuard::True,
                    vec![refer("hypertension_specialist")], "htn_specialist"),
            ],
            false, false));

        doc.add_decision_point(dp("htn_emergency", "Hypertensive Emergency", None,
            vec![br("htn_em_resolve", clin_guard("systolic_bp", ComparisonOp::Lt, 180.0),
                vec![reassess(1, vec!["blood_pressure"])], "htn_assess")],
            false, false));

        doc.add_decision_point(dp("htn_specialist", "Specialist Care", None,
            vec![], false, true));

        doc.add_decision_point(dp("htn_stable", "Stable BP", None,
            vec![br("htn_continue", GuidelineGuard::True,
                vec![reassess(180, vec!["blood_pressure", "renal", "electrolytes"])], "htn_stable")],
            false, true));

        doc.add_safety_constraint(safety("htn_no_ace_arb",
            "Do not combine ACE-I and ARB",
            GuidelineGuard::And(vec![med_active("lisinopril"), med_active("losartan")])));
        doc.add_safety_constraint(safety("htn_k_spiro",
            "Monitor potassium with spironolactone",
            med_active("spironolactone")));

        doc.add_monitoring(monitor_req("htn_bp", "blood_pressure", 30));
        doc.add_monitoring(monitor_req("htn_cr", "creatinine", 180));
        doc.add_monitoring(monitor_req("htn_k", "potassium", 90));

        doc
    }
}

// ---------------------------------------------------------------------------
// Atrial Fibrillation
// ---------------------------------------------------------------------------

pub struct AtrialFibrillationTemplate;

impl GuidelineTemplate for AtrialFibrillationTemplate {
    fn name(&self) -> &str { "AHA/ACC Atrial Fibrillation Management" }
    fn condition(&self) -> &str { "Atrial Fibrillation" }
    fn build(&self) -> GuidelineDocument {
        let mut doc = GuidelineDocument::new(self.name());
        doc.metadata.condition = Some(self.condition().into());
        doc.metadata.tags = vec!["cardiology".into(), "arrhythmia".into()];

        doc.add_decision_point(dp("af_initial", "AF Risk Stratification", Some("CHA2DS2-VASc scoring"),
            vec![
                br("af_low_risk", clin_guard("chadsvasc", ComparisonOp::Lt, 2.0),
                    vec![GuidelineAction::NoChange { reason: Some("Low stroke risk".into()) }], "af_rate_control"),
                br("af_high_risk", clin_guard("chadsvasc", ComparisonOp::Ge, 2.0),
                    vec![start_med("apixaban", 5.0, "mg", "BID", "oral")], "af_anticoag_monitor"),
            ],
            true, false));

        doc.add_decision_point(dp("af_anticoag_monitor", "Anticoagulation Monitoring", None,
            vec![
                br("af_bleed", clin_guard("hasbled", ComparisonOp::Ge, 3.0),
                    vec![adjust_dose("apixaban", 2.5, "mg")], "af_reduced_dose"),
                br("af_no_bleed", clin_guard("hasbled", ComparisonOp::Lt, 3.0),
                    vec![reassess(90, vec!["renal", "cbc"])], "af_rate_control"),
            ],
            false, false));

        doc.add_decision_point(dp("af_reduced_dose", "Reduced Dose DOAC", None,
            vec![br("af_reduced_ok", GuidelineGuard::True,
                vec![reassess(90, vec!["renal", "cbc"])], "af_rate_control")],
            false, false));

        doc.add_decision_point(dp("af_rate_control", "Rate Control Strategy", None,
            vec![
                br("af_beta", GuidelineGuard::Not(Box::new(dx_guard("asthma"))),
                    vec![start_med("metoprolol", 25.0, "mg", "BID", "oral")], "af_rate_monitor"),
                br("af_ccb", dx_guard("asthma"),
                    vec![start_med("diltiazem", 120.0, "mg", "QD", "oral")], "af_rate_monitor"),
            ],
            false, false));

        doc.add_decision_point(dp("af_rate_monitor", "Rate Control Monitoring", None,
            vec![
                br("af_rate_ok", clin_guard("heart_rate", ComparisonOp::Lt, 110.0),
                    vec![reassess(90, vec!["heart_rate", "ecg"])], "af_stable"),
                br("af_rate_high", clin_guard("heart_rate", ComparisonOp::Ge, 110.0),
                    vec![adjust_dose("metoprolol", 50.0, "mg")], "af_rate_titrate"),
            ],
            false, false));

        doc.add_decision_point(dp("af_rate_titrate", "Rate Titration", None,
            vec![
                br("af_rate_ctrl2", clin_guard("heart_rate", ComparisonOp::Lt, 110.0),
                    vec![], "af_stable"),
                br("af_rhythm", clin_guard("heart_rate", ComparisonOp::Ge, 110.0),
                    vec![refer("electrophysiology")], "af_rhythm_eval"),
            ],
            false, false));

        doc.add_decision_point(dp("af_rhythm_eval", "Rhythm Control Evaluation", None,
            vec![
                br("af_amio", dx_guard("structural_heart_disease"),
                    vec![start_med("amiodarone", 200.0, "mg", "QD", "oral")], "af_rhythm_monitor"),
                br("af_flec", GuidelineGuard::True,
                    vec![start_med("flecainide", 50.0, "mg", "BID", "oral")], "af_rhythm_monitor"),
            ],
            false, false));

        doc.add_decision_point(dp("af_rhythm_monitor", "Rhythm Monitoring", None,
            vec![
                br("af_sinus", dx_guard("sinus_rhythm"),
                    vec![reassess(90, vec!["ecg"])], "af_stable"),
                br("af_persist", GuidelineGuard::True,
                    vec![refer("electrophysiology")], "af_ablation_eval"),
            ],
            false, false));

        doc.add_decision_point(dp("af_ablation_eval", "Ablation Evaluation", None,
            vec![br("af_abl", GuidelineGuard::True, vec![refer("electrophysiology")], "af_stable")],
            false, false));

        doc.add_decision_point(dp("af_stable", "Stable AF Management", None,
            vec![br("af_cont", GuidelineGuard::True,
                vec![reassess(180, vec!["ecg", "renal", "thyroid"])], "af_stable")],
            false, true));

        doc.add_safety_constraint(safety("af_no_dual_anticoag",
            "Avoid dual anticoagulation",
            GuidelineGuard::And(vec![med_active("apixaban"), med_active("warfarin")])));

        doc.add_monitoring(monitor_req("af_ecg", "ecg", 90));
        doc.add_monitoring(monitor_req("af_renal", "renal_function", 180));

        doc
    }
}

// ---------------------------------------------------------------------------
// Heart Failure
// ---------------------------------------------------------------------------

pub struct HeartFailureTemplate;

impl GuidelineTemplate for HeartFailureTemplate {
    fn name(&self) -> &str { "ACC/AHA Heart Failure Management" }
    fn condition(&self) -> &str { "Heart Failure" }
    fn build(&self) -> GuidelineDocument {
        let mut doc = GuidelineDocument::new(self.name());
        doc.metadata.condition = Some(self.condition().into());
        doc.metadata.tags = vec!["cardiology".into(), "heart_failure".into()];

        doc.add_decision_point(dp("hf_classify", "HF Classification", Some("Determine EF category"),
            vec![
                br("hf_hfref", clin_guard("lvef", ComparisonOp::Le, 40.0),
                    vec![start_med("sacubitril_valsartan", 49.0, "mg", "BID", "oral"),
                         start_med("carvedilol", 3.125, "mg", "BID", "oral")],
                    "hf_gdmt_titrate"),
                br("hf_hfmref",
                    GuidelineGuard::And(vec![clin_guard("lvef", ComparisonOp::Gt, 40.0), clin_guard("lvef", ComparisonOp::Le, 50.0)]),
                    vec![start_med("empagliflozin", 10.0, "mg", "QD", "oral")],
                    "hf_sglt2_monitor"),
                br("hf_hfpef", clin_guard("lvef", ComparisonOp::Gt, 50.0),
                    vec![GuidelineAction::LifestyleModification { category: "fluid".into(), description: "Sodium restriction, diuretics PRN".into() }],
                    "hf_pef_manage"),
            ],
            true, false));

        doc.add_decision_point(dp("hf_gdmt_titrate", "GDMT Titration", None,
            vec![
                br("hf_uptitrate", clin_guard("systolic_bp", ComparisonOp::Ge, 100.0),
                    vec![adjust_dose("sacubitril_valsartan", 97.0, "mg"), adjust_dose("carvedilol", 6.25, "mg")],
                    "hf_gdmt_monitor"),
                br("hf_hypotensive", clin_guard("systolic_bp", ComparisonOp::Lt, 100.0),
                    vec![GuidelineAction::NoChange { reason: Some("Hypotension limits titration".into()) }],
                    "hf_gdmt_monitor"),
            ],
            false, false));

        doc.add_decision_point(dp("hf_gdmt_monitor", "GDMT Monitoring", None,
            vec![
                br("hf_improved", clin_guard("lvef", ComparisonOp::Gt, 40.0),
                    vec![reassess(90, vec!["echo", "bnp"])], "hf_stable"),
                br("hf_worsening", clin_guard("nyha_class", ComparisonOp::Ge, 3.0),
                    vec![start_med("spironolactone", 25.0, "mg", "QD", "oral")],
                    "hf_advanced_eval"),
                br("hf_stable_g", GuidelineGuard::True,
                    vec![reassess(90, vec!["echo", "bnp"])], "hf_gdmt_titrate"),
            ],
            false, false));

        doc.add_decision_point(dp("hf_sglt2_monitor", "SGLT2 Monitoring (HFmrEF)", None,
            vec![
                br("hf_sglt2_ok", clin_guard("nyha_class", ComparisonOp::Le, 2.0),
                    vec![reassess(90, vec!["echo"])], "hf_stable"),
                br("hf_sglt2_fail", clin_guard("nyha_class", ComparisonOp::Gt, 2.0),
                    vec![], "hf_gdmt_titrate"),
            ],
            false, false));

        doc.add_decision_point(dp("hf_pef_manage", "HFpEF Management", None,
            vec![
                br("hf_pef_diuretic", clin_guard("weight_change_kg", ComparisonOp::Gt, 2.0),
                    vec![start_med("furosemide", 20.0, "mg", "QD", "oral")], "hf_pef_monitor"),
                br("hf_pef_stable", GuidelineGuard::True,
                    vec![reassess(90, vec!["weight", "bnp"])], "hf_stable"),
            ],
            false, false));

        doc.add_decision_point(dp("hf_pef_monitor", "HFpEF Diuretic Monitoring", None,
            vec![
                br("hf_pef_dry", clin_guard("weight_change_kg", ComparisonOp::Le, 2.0),
                    vec![], "hf_stable"),
                br("hf_pef_wet", GuidelineGuard::True,
                    vec![adjust_dose("furosemide", 40.0, "mg")], "hf_pef_monitor"),
            ],
            false, false));

        doc.add_decision_point(dp("hf_advanced_eval", "Advanced HF Evaluation", None,
            vec![
                br("hf_device", clin_guard("lvef", ComparisonOp::Le, 35.0),
                    vec![refer("electrophysiology")], "hf_device_eval"),
                br("hf_continue", GuidelineGuard::True,
                    vec![reassess(30, vec!["bnp", "renal"])], "hf_gdmt_monitor"),
            ],
            false, false));

        doc.add_decision_point(dp("hf_device_eval", "Device Therapy Evaluation", None,
            vec![br("hf_icd", GuidelineGuard::True, vec![refer("electrophysiology")], "hf_stable")],
            false, false));

        doc.add_decision_point(dp("hf_stable", "Stable HF", None,
            vec![br("hf_cont", GuidelineGuard::True,
                vec![reassess(90, vec!["echo", "bnp", "renal", "electrolytes"])], "hf_stable")],
            false, true));

        doc.add_safety_constraint(safety("hf_k_spiro",
            "Monitor potassium with spironolactone + ACE-I",
            GuidelineGuard::And(vec![med_active("spironolactone"), med_active("sacubitril_valsartan")])));

        doc.add_monitoring(monitor_req("hf_echo", "echo", 180));
        doc.add_monitoring(monitor_req("hf_bnp", "bnp", 90));
        doc.add_monitoring(monitor_req("hf_k", "potassium", 30));

        doc
    }
}

// ---------------------------------------------------------------------------
// COPD
// ---------------------------------------------------------------------------

pub struct COPDTemplate;

impl GuidelineTemplate for COPDTemplate {
    fn name(&self) -> &str { "GOLD COPD Management" }
    fn condition(&self) -> &str { "Chronic Obstructive Pulmonary Disease" }
    fn build(&self) -> GuidelineDocument {
        let mut doc = GuidelineDocument::new(self.name());
        doc.metadata.condition = Some(self.condition().into());
        doc.metadata.tags = vec!["pulmonary".into(), "copd".into()];

        doc.add_decision_point(dp("copd_assess", "COPD Assessment", Some("GOLD staging"),
            vec![
                br("copd_group_a",
                    GuidelineGuard::And(vec![clin_guard("exacerbations_year", ComparisonOp::Lt, 2.0), clin_guard("cat_score", ComparisonOp::Lt, 10.0)]),
                    vec![start_med("salbutamol", 100.0, "mcg", "PRN", "inhaled")],
                    "copd_prn_monitor"),
                br("copd_group_b",
                    GuidelineGuard::And(vec![clin_guard("exacerbations_year", ComparisonOp::Lt, 2.0), clin_guard("cat_score", ComparisonOp::Ge, 10.0)]),
                    vec![start_med("tiotropium", 18.0, "mcg", "QD", "inhaled")],
                    "copd_lama_monitor"),
                br("copd_group_e", clin_guard("exacerbations_year", ComparisonOp::Ge, 2.0),
                    vec![GuidelineAction::CombinationTherapy {
                        medications: vec![
                            MedicationSpec { name: "tiotropium".into(), dose: DoseSpec::new(18.0, "mcg").with_frequency("QD"), route: Some("inhaled".into()) },
                            MedicationSpec { name: "formoterol_budesonide".into(), dose: DoseSpec::new(160.0, "mcg").with_frequency("BID"), route: Some("inhaled".into()) },
                        ],
                        reason: Some("Frequent exacerbations".into()),
                    }],
                    "copd_triple_eval"),
            ],
            true, false));

        doc.add_decision_point(dp("copd_prn_monitor", "PRN Bronchodilator Monitoring", None,
            vec![
                br("copd_prn_ok", clin_guard("cat_score", ComparisonOp::Lt, 10.0),
                    vec![reassess(180, vec!["spirometry"])], "copd_stable"),
                br("copd_prn_worse", clin_guard("cat_score", ComparisonOp::Ge, 10.0),
                    vec![], "copd_lama_monitor"),
            ],
            false, false));

        doc.add_decision_point(dp("copd_lama_monitor", "LAMA Monitoring", None,
            vec![
                br("copd_lama_ok", clin_guard("cat_score", ComparisonOp::Lt, 10.0),
                    vec![reassess(90, vec!["spirometry", "cat_score"])], "copd_stable"),
                br("copd_lama_fail", clin_guard("cat_score", ComparisonOp::Ge, 10.0),
                    vec![start_med("formoterol", 12.0, "mcg", "BID", "inhaled")],
                    "copd_lama_laba_monitor"),
            ],
            false, false));

        doc.add_decision_point(dp("copd_lama_laba_monitor", "LAMA+LABA Monitoring", None,
            vec![
                br("copd_ll_ok", clin_guard("exacerbations_year", ComparisonOp::Lt, 2.0),
                    vec![reassess(90, vec!["spirometry"])], "copd_stable"),
                br("copd_ll_fail", clin_guard("exacerbations_year", ComparisonOp::Ge, 2.0),
                    vec![], "copd_triple_eval"),
            ],
            false, false));

        doc.add_decision_point(dp("copd_triple_eval", "Triple Therapy Evaluation", None,
            vec![
                br("copd_add_ics", clin_guard("eosinophils", ComparisonOp::Ge, 300.0),
                    vec![start_med("budesonide", 400.0, "mcg", "BID", "inhaled")], "copd_triple_monitor"),
                br("copd_no_ics", clin_guard("eosinophils", ComparisonOp::Lt, 300.0),
                    vec![start_med("roflumilast", 500.0, "mcg", "QD", "oral")], "copd_pde4_monitor"),
            ],
            false, false));

        doc.add_decision_point(dp("copd_triple_monitor", "Triple Therapy Monitoring", None,
            vec![
                br("copd_triple_ok", clin_guard("exacerbations_year", ComparisonOp::Lt, 2.0),
                    vec![reassess(90, vec!["spirometry"])], "copd_stable"),
                br("copd_triple_fail", GuidelineGuard::True,
                    vec![refer("pulmonology")], "copd_specialist"),
            ],
            false, false));

        doc.add_decision_point(dp("copd_pde4_monitor", "PDE4 Inhibitor Monitoring", None,
            vec![
                br("copd_pde4_ok", clin_guard("exacerbations_year", ComparisonOp::Lt, 2.0),
                    vec![reassess(90, vec!["spirometry"])], "copd_stable"),
                br("copd_pde4_fail", GuidelineGuard::True, vec![refer("pulmonology")], "copd_specialist"),
            ],
            false, false));

        doc.add_decision_point(dp("copd_specialist", "Specialist Care", None, vec![], false, true));

        doc.add_decision_point(dp("copd_stable", "Stable COPD", None,
            vec![br("copd_cont", GuidelineGuard::True,
                vec![reassess(180, vec!["spirometry", "cat_score"])], "copd_stable")],
            false, true));

        doc.add_monitoring(monitor_req("copd_spiro", "spirometry", 180));
        doc.add_monitoring(monitor_req("copd_eos", "eosinophils", 365));

        doc
    }
}

// ---------------------------------------------------------------------------
// Depression
// ---------------------------------------------------------------------------

pub struct DepressionTemplate;

impl GuidelineTemplate for DepressionTemplate {
    fn name(&self) -> &str { "APA Major Depressive Disorder Management" }
    fn condition(&self) -> &str { "Major Depressive Disorder" }
    fn build(&self) -> GuidelineDocument {
        let mut doc = GuidelineDocument::new(self.name());
        doc.metadata.condition = Some(self.condition().into());
        doc.metadata.tags = vec!["psychiatry".into(), "depression".into()];

        doc.add_decision_point(dp("mdd_assess", "Depression Severity Assessment", Some("PHQ-9 scoring"),
            vec![
                br("mdd_mild", clin_guard("phq9", ComparisonOp::Lt, 10.0),
                    vec![GuidelineAction::LifestyleModification { category: "exercise".into(), description: "Regular aerobic exercise 150min/week".into() },
                         GuidelineAction::PatientEducation { topic: "depression".into(), materials: vec!["depression_self_help".into()] }],
                    "mdd_watchful_wait"),
                br("mdd_moderate",
                    GuidelineGuard::And(vec![clin_guard("phq9", ComparisonOp::Ge, 10.0), clin_guard("phq9", ComparisonOp::Lt, 20.0)]),
                    vec![start_med("sertraline", 50.0, "mg", "QD", "oral")],
                    "mdd_ssri_monitor"),
                br("mdd_severe", clin_guard("phq9", ComparisonOp::Ge, 20.0),
                    vec![start_med("sertraline", 100.0, "mg", "QD", "oral"), refer("psychiatry")],
                    "mdd_ssri_monitor"),
            ],
            true, false));

        doc.add_decision_point(dp("mdd_watchful_wait", "Watchful Waiting", None,
            vec![
                br("mdd_ww_improve", clin_guard("phq9", ComparisonOp::Lt, 5.0),
                    vec![], "mdd_remission"),
                br("mdd_ww_persist", clin_guard("phq9", ComparisonOp::Ge, 10.0),
                    vec![start_med("sertraline", 50.0, "mg", "QD", "oral")], "mdd_ssri_monitor"),
                br("mdd_ww_stable", GuidelineGuard::True,
                    vec![reassess(30, vec!["phq9"])], "mdd_watchful_wait"),
            ],
            false, false));

        doc.add_decision_point(dp("mdd_ssri_monitor", "SSRI Monitoring", Some("4-6 week assessment"),
            vec![
                br("mdd_response", clin_guard("phq9_reduction_pct", ComparisonOp::Ge, 50.0),
                    vec![reassess(30, vec!["phq9"])], "mdd_continuation"),
                br("mdd_partial", 
                    GuidelineGuard::And(vec![clin_guard("phq9_reduction_pct", ComparisonOp::Ge, 25.0), clin_guard("phq9_reduction_pct", ComparisonOp::Lt, 50.0)]),
                    vec![adjust_dose("sertraline", 100.0, "mg")], "mdd_uptitrate_monitor"),
                br("mdd_no_response", clin_guard("phq9_reduction_pct", ComparisonOp::Lt, 25.0),
                    vec![], "mdd_switch_eval"),
            ],
            false, false));

        doc.add_decision_point(dp("mdd_uptitrate_monitor", "Uptitration Monitoring", None,
            vec![
                br("mdd_up_ok", clin_guard("phq9_reduction_pct", ComparisonOp::Ge, 50.0),
                    vec![], "mdd_continuation"),
                br("mdd_up_fail", GuidelineGuard::True, vec![], "mdd_switch_eval"),
            ],
            false, false));

        doc.add_decision_point(dp("mdd_switch_eval", "Switch/Augment Evaluation", None,
            vec![
                br("mdd_switch", GuidelineGuard::Not(Box::new(dx_guard("treatment_resistant"))),
                    vec![stop_med("sertraline"), start_med("venlafaxine", 75.0, "mg", "QD", "oral")],
                    "mdd_snri_monitor"),
                br("mdd_augment", dx_guard("treatment_resistant"),
                    vec![start_med("aripiprazole", 2.0, "mg", "QD", "oral")],
                    "mdd_augment_monitor"),
            ],
            false, false));

        doc.add_decision_point(dp("mdd_snri_monitor", "SNRI Monitoring", None,
            vec![
                br("mdd_snri_ok", clin_guard("phq9_reduction_pct", ComparisonOp::Ge, 50.0),
                    vec![], "mdd_continuation"),
                br("mdd_snri_fail", GuidelineGuard::True,
                    vec![refer("psychiatry")], "mdd_specialist"),
            ],
            false, false));

        doc.add_decision_point(dp("mdd_augment_monitor", "Augmentation Monitoring", None,
            vec![
                br("mdd_aug_ok", clin_guard("phq9_reduction_pct", ComparisonOp::Ge, 50.0),
                    vec![], "mdd_continuation"),
                br("mdd_aug_fail", GuidelineGuard::True,
                    vec![refer("psychiatry")], "mdd_specialist"),
            ],
            false, false));

        doc.add_decision_point(dp("mdd_continuation", "Continuation Phase", Some("Maintain for 6-12 months"),
            vec![
                br("mdd_remit", clin_guard("phq9", ComparisonOp::Lt, 5.0),
                    vec![reassess(90, vec!["phq9"])], "mdd_remission"),
                br("mdd_relapse", clin_guard("phq9", ComparisonOp::Ge, 10.0),
                    vec![], "mdd_switch_eval"),
                br("mdd_cont_stable", GuidelineGuard::True,
                    vec![reassess(30, vec!["phq9"])], "mdd_continuation"),
            ],
            false, false));

        doc.add_decision_point(dp("mdd_specialist", "Psychiatry Specialist", None, vec![], false, true));

        doc.add_decision_point(dp("mdd_remission", "Remission", None,
            vec![br("mdd_rem_cont", GuidelineGuard::True,
                vec![reassess(90, vec!["phq9"])], "mdd_remission")],
            false, true));

        doc.add_safety_constraint(safety("mdd_no_ssri_maoi",
            "Do not combine SSRI with MAOI",
            GuidelineGuard::And(vec![med_active("sertraline"), med_active("phenelzine")])));

        doc.add_monitoring(monitor_req("mdd_phq9", "phq9", 30));

        doc
    }
}

// ---------------------------------------------------------------------------
// CKD
// ---------------------------------------------------------------------------

pub struct CKDTemplate;

impl GuidelineTemplate for CKDTemplate {
    fn name(&self) -> &str { "KDIGO Chronic Kidney Disease Management" }
    fn condition(&self) -> &str { "Chronic Kidney Disease" }
    fn build(&self) -> GuidelineDocument {
        let mut doc = GuidelineDocument::new(self.name());
        doc.metadata.condition = Some(self.condition().into());
        doc.metadata.tags = vec!["nephrology".into(), "ckd".into()];

        doc.add_decision_point(dp("ckd_stage", "CKD Staging", Some("eGFR-based staging"),
            vec![
                br("ckd_g1_g2", clin_guard("eGFR", ComparisonOp::Ge, 60.0),
                    vec![GuidelineAction::LifestyleModification { category: "diet".into(), description: "Protein restriction, sodium limit".into() }],
                    "ckd_risk_factor"),
                br("ckd_g3",
                    GuidelineGuard::And(vec![clin_guard("eGFR", ComparisonOp::Ge, 30.0), clin_guard("eGFR", ComparisonOp::Lt, 60.0)]),
                    vec![start_med("dapagliflozin", 10.0, "mg", "QD", "oral")],
                    "ckd_sglt2_monitor"),
                br("ckd_g4", 
                    GuidelineGuard::And(vec![clin_guard("eGFR", ComparisonOp::Ge, 15.0), clin_guard("eGFR", ComparisonOp::Lt, 30.0)]),
                    vec![refer("nephrology")], "ckd_nephr_care"),
                br("ckd_g5", clin_guard("eGFR", ComparisonOp::Lt, 15.0),
                    vec![refer("nephrology")], "ckd_dialysis_eval"),
            ],
            true, false));

        doc.add_decision_point(dp("ckd_risk_factor", "Risk Factor Management", None,
            vec![
                br("ckd_htn", clin_guard("systolic_bp", ComparisonOp::Ge, 130.0),
                    vec![start_med("lisinopril", 10.0, "mg", "QD", "oral")], "ckd_ace_monitor"),
                br("ckd_no_htn", clin_guard("systolic_bp", ComparisonOp::Lt, 130.0),
                    vec![reassess(180, vec!["eGFR", "uacr"])], "ckd_monitor_routine"),
            ],
            false, false));

        doc.add_decision_point(dp("ckd_ace_monitor", "ACE-I Monitoring", None,
            vec![
                br("ckd_ace_ok",
                    GuidelineGuard::And(vec![clin_guard("systolic_bp", ComparisonOp::Lt, 130.0), clin_guard("potassium", ComparisonOp::Lt, 5.5)]),
                    vec![reassess(90, vec!["eGFR", "potassium"])], "ckd_monitor_routine"),
                br("ckd_ace_hyperK", clin_guard("potassium", ComparisonOp::Ge, 5.5),
                    vec![stop_med("lisinopril"), start_med("amlodipine", 5.0, "mg", "QD", "oral")],
                    "ckd_monitor_routine"),
            ],
            false, false));

        doc.add_decision_point(dp("ckd_sglt2_monitor", "SGLT2i Monitoring", None,
            vec![
                br("ckd_sglt2_ok", clin_guard("eGFR", ComparisonOp::Ge, 25.0),
                    vec![reassess(90, vec!["eGFR", "uacr"])], "ckd_monitor_routine"),
                br("ckd_sglt2_decline", clin_guard("eGFR", ComparisonOp::Lt, 25.0),
                    vec![stop_med("dapagliflozin")], "ckd_nephr_care"),
            ],
            false, false));

        doc.add_decision_point(dp("ckd_nephr_care", "Nephrology Care", None,
            vec![
                br("ckd_neph_stable", clin_guard("eGFR", ComparisonOp::Ge, 15.0),
                    vec![reassess(90, vec!["eGFR", "calcium", "phosphate", "pth"])],
                    "ckd_monitor_routine"),
                br("ckd_neph_progress", clin_guard("eGFR", ComparisonOp::Lt, 15.0),
                    vec![], "ckd_dialysis_eval"),
            ],
            false, false));

        doc.add_decision_point(dp("ckd_dialysis_eval", "Dialysis Evaluation", None,
            vec![
                br("ckd_start_dialysis", clin_guard("eGFR", ComparisonOp::Lt, 10.0),
                    vec![refer("dialysis_center")], "ckd_dialysis"),
                br("ckd_wait", GuidelineGuard::True,
                    vec![reassess(30, vec!["eGFR", "electrolytes"])], "ckd_dialysis_eval"),
            ],
            false, false));

        doc.add_decision_point(dp("ckd_anemia", "Anemia Management", None,
            vec![
                br("ckd_iron_def", lab_guard("ferritin", ComparisonOp::Lt, 100.0),
                    vec![start_med("iron_sucrose", 200.0, "mg", "IV weekly", "IV")],
                    "ckd_anemia_monitor"),
                br("ckd_esa", lab_guard("hemoglobin", ComparisonOp::Lt, 10.0),
                    vec![start_med("epoetin_alfa", 50.0, "units/kg", "TIW", "subcutaneous")],
                    "ckd_anemia_monitor"),
                br("ckd_no_anemia", GuidelineGuard::True, vec![], "ckd_monitor_routine"),
            ],
            false, false));

        doc.add_decision_point(dp("ckd_anemia_monitor", "Anemia Monitoring", None,
            vec![
                br("ckd_hb_ok",
                    GuidelineGuard::And(vec![lab_guard("hemoglobin", ComparisonOp::Ge, 10.0), lab_guard("hemoglobin", ComparisonOp::Le, 12.0)]),
                    vec![reassess(30, vec!["cbc", "ferritin"])], "ckd_monitor_routine"),
                br("ckd_hb_low", lab_guard("hemoglobin", ComparisonOp::Lt, 10.0),
                    vec![], "ckd_anemia"),
            ],
            false, false));

        doc.add_decision_point(dp("ckd_dialysis", "Dialysis", None, vec![], false, true));

        doc.add_decision_point(dp("ckd_monitor_routine", "Routine CKD Monitoring", None,
            vec![br("ckd_routine_cont", GuidelineGuard::True,
                vec![reassess(90, vec!["eGFR", "uacr", "electrolytes"])], "ckd_monitor_routine")],
            false, true));

        doc.add_safety_constraint(safety("ckd_no_nsaids",
            "Avoid NSAIDs in CKD",
            GuidelineGuard::And(vec![med_active("ibuprofen"), clin_guard("eGFR", ComparisonOp::Lt, 60.0)])));

        doc.add_monitoring(monitor_req("ckd_egfr", "eGFR", 90));
        doc.add_monitoring(monitor_req("ckd_uacr", "uacr", 180));
        doc.add_monitoring(monitor_req("ckd_k", "potassium", 90));

        doc
    }
}

// ---------------------------------------------------------------------------
// Chronic Pain
// ---------------------------------------------------------------------------

pub struct ChronicPainTemplate;

impl GuidelineTemplate for ChronicPainTemplate {
    fn name(&self) -> &str { "CDC Chronic Pain Management" }
    fn condition(&self) -> &str { "Chronic Non-Cancer Pain" }
    fn build(&self) -> GuidelineDocument {
        let mut doc = GuidelineDocument::new(self.name());
        doc.metadata.condition = Some(self.condition().into());
        doc.metadata.tags = vec!["pain".into(), "opioid".into()];

        doc.add_decision_point(dp("pain_assess", "Pain Assessment", Some("Evaluate pain severity and type"),
            vec![
                br("pain_mild", clin_guard("pain_score", ComparisonOp::Lt, 4.0),
                    vec![start_med("acetaminophen", 650.0, "mg", "Q6H", "oral"),
                         GuidelineAction::LifestyleModification { category: "physical_therapy".into(), description: "PT referral".into() }],
                    "pain_nonopioid_monitor"),
                br("pain_moderate",
                    GuidelineGuard::And(vec![clin_guard("pain_score", ComparisonOp::Ge, 4.0), clin_guard("pain_score", ComparisonOp::Lt, 7.0)]),
                    vec![start_med("duloxetine", 30.0, "mg", "QD", "oral")],
                    "pain_nonopioid_monitor"),
                br("pain_severe", clin_guard("pain_score", ComparisonOp::Ge, 7.0),
                    vec![start_med("tramadol", 50.0, "mg", "Q6H", "oral"),
                         order_lab("urine_drug_screen")],
                    "pain_opioid_eval"),
            ],
            true, false));

        doc.add_decision_point(dp("pain_nonopioid_monitor", "Non-Opioid Monitoring", None,
            vec![
                br("pain_non_improve", clin_guard("pain_score", ComparisonOp::Lt, 4.0),
                    vec![reassess(90, vec!["pain_score", "function"])], "pain_stable"),
                br("pain_non_fail", clin_guard("pain_score", ComparisonOp::Ge, 4.0),
                    vec![], "pain_escalate"),
            ],
            false, false));

        doc.add_decision_point(dp("pain_escalate", "Escalation Evaluation", None,
            vec![
                br("pain_add_topical", GuidelineGuard::True,
                    vec![start_med("lidocaine_patch", 1.0, "patch", "QD", "topical")],
                    "pain_topical_monitor"),
            ],
            false, false));

        doc.add_decision_point(dp("pain_topical_monitor", "Topical Monitoring", None,
            vec![
                br("pain_top_ok", clin_guard("pain_score", ComparisonOp::Lt, 4.0),
                    vec![], "pain_stable"),
                br("pain_top_fail", clin_guard("pain_score", ComparisonOp::Ge, 4.0),
                    vec![], "pain_opioid_eval"),
            ],
            false, false));

        doc.add_decision_point(dp("pain_opioid_eval", "Opioid Evaluation", Some("Risk assessment before opioid therapy"),
            vec![
                br("pain_high_risk", clin_guard("opioid_risk_score", ComparisonOp::Ge, 8.0),
                    vec![refer("pain_specialist"), start_med("gabapentin", 300.0, "mg", "TID", "oral")],
                    "pain_specialist_care"),
                br("pain_low_risk", clin_guard("opioid_risk_score", ComparisonOp::Lt, 8.0),
                    vec![start_med("tramadol", 50.0, "mg", "Q6H", "oral"), order_lab("urine_drug_screen")],
                    "pain_opioid_monitor"),
            ],
            false, false));

        doc.add_decision_point(dp("pain_opioid_monitor", "Opioid Monitoring", None,
            vec![
                br("pain_opioid_ok",
                    GuidelineGuard::And(vec![clin_guard("pain_score", ComparisonOp::Lt, 4.0), clin_guard("function_score", ComparisonOp::Ge, 70.0)]),
                    vec![reassess(30, vec!["pain_score", "function", "uds"])], "pain_opioid_continue"),
                br("pain_opioid_no_benefit", clin_guard("function_score", ComparisonOp::Lt, 50.0),
                    vec![], "pain_opioid_taper"),
                br("pain_opioid_titrate", GuidelineGuard::True,
                    vec![reassess(30, vec!["pain_score"])], "pain_opioid_monitor"),
            ],
            false, false));

        doc.add_decision_point(dp("pain_opioid_continue", "Opioid Continuation", None,
            vec![
                br("pain_oc_stable", GuidelineGuard::True,
                    vec![reassess(90, vec!["pain_score", "function", "uds"])], "pain_stable"),
            ],
            false, false));

        doc.add_decision_point(dp("pain_opioid_taper", "Opioid Taper", None,
            vec![
                br("pain_taper", GuidelineGuard::True,
                    vec![adjust_dose("tramadol", 25.0, "mg"), reassess(14, vec!["pain_score", "withdrawal"])],
                    "pain_post_taper"),
            ],
            false, false));

        doc.add_decision_point(dp("pain_post_taper", "Post-Taper Evaluation", None,
            vec![
                br("pain_pt_ok", clin_guard("pain_score", ComparisonOp::Lt, 7.0),
                    vec![], "pain_nonopioid_monitor"),
                br("pain_pt_fail", clin_guard("pain_score", ComparisonOp::Ge, 7.0),
                    vec![refer("pain_specialist")], "pain_specialist_care"),
            ],
            false, false));

        doc.add_decision_point(dp("pain_specialist_care", "Pain Specialist", None, vec![], false, true));
        doc.add_decision_point(dp("pain_stable", "Stable Pain Management", None,
            vec![br("pain_cont", GuidelineGuard::True,
                vec![reassess(90, vec!["pain_score", "function"])], "pain_stable")],
            false, true));

        doc.add_safety_constraint(safety("pain_naloxone",
            "Co-prescribe naloxone with opioids ≥ 50 MME",
            med_active("tramadol")));

        doc.add_monitoring(monitor_req("pain_uds", "urine_drug_screen", 90));
        doc.add_monitoring(monitor_req("pain_function", "function_score", 30));

        doc
    }
}

// ---------------------------------------------------------------------------
// All templates
// ---------------------------------------------------------------------------

/// Return all available templates.
pub fn all_templates() -> Vec<Box<dyn GuidelineTemplate>> {
    vec![
        Box::new(Type2DiabetesTemplate),
        Box::new(HypertensionTemplate),
        Box::new(AtrialFibrillationTemplate),
        Box::new(HeartFailureTemplate),
        Box::new(COPDTemplate),
        Box::new(DepressionTemplate),
        Box::new(CKDTemplate),
        Box::new(ChronicPainTemplate),
    ]
}

/// Build all templates and return them keyed by condition.
pub fn build_all_templates() -> HashMap<String, GuidelineDocument> {
    all_templates()
        .iter()
        .map(|t| (t.condition().to_string(), t.build()))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pta_builder::PtaBuilder;

    fn assert_template_valid(t: &dyn GuidelineTemplate) {
        let doc = t.build();
        assert!(!doc.metadata.title.is_empty(), "Title empty for {}", t.name());
        assert!(doc.num_decision_points() >= 8, "{} has < 8 DPs ({})", t.name(), doc.num_decision_points());
        assert!(!doc.initial_points().is_empty(), "No initial point for {}", t.name());
        assert!(!doc.terminal_points().is_empty(), "No terminal point for {}", t.name());
        assert!(doc.total_branches() > 0, "No branches for {}", t.name());

        // Should be convertible to PTA without panic
        let mut builder = PtaBuilder::new();
        let pta = builder.build(&doc);
        assert!(!pta.locations.is_empty());
        assert!(!pta.edges.is_empty());
    }

    #[test]
    fn test_diabetes_template() {
        assert_template_valid(&Type2DiabetesTemplate);
    }

    #[test]
    fn test_hypertension_template() {
        assert_template_valid(&HypertensionTemplate);
    }

    #[test]
    fn test_atrial_fibrillation_template() {
        assert_template_valid(&AtrialFibrillationTemplate);
    }

    #[test]
    fn test_heart_failure_template() {
        assert_template_valid(&HeartFailureTemplate);
    }

    #[test]
    fn test_copd_template() {
        assert_template_valid(&COPDTemplate);
    }

    #[test]
    fn test_depression_template() {
        assert_template_valid(&DepressionTemplate);
    }

    #[test]
    fn test_ckd_template() {
        assert_template_valid(&CKDTemplate);
    }

    #[test]
    fn test_chronic_pain_template() {
        assert_template_valid(&ChronicPainTemplate);
    }

    #[test]
    fn test_all_templates() {
        let templates = all_templates();
        assert_eq!(templates.len(), 8);
        for t in &templates {
            let doc = t.build();
            assert!(!doc.metadata.title.is_empty());
        }
    }

    #[test]
    fn test_build_all_templates() {
        let all = build_all_templates();
        assert!(all.len() >= 8);
        assert!(all.contains_key("Type 2 Diabetes Mellitus"));
        assert!(all.contains_key("Hypertension"));
    }

    #[test]
    fn test_template_serialisation() {
        for t in all_templates() {
            let doc = t.build();
            let json = serde_json::to_string(&doc).unwrap();
            let parsed: GuidelineDocument = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.metadata.title, doc.metadata.title);
        }
    }

    #[test]
    fn test_template_medications() {
        let doc = Type2DiabetesTemplate.build();
        let meds = doc.all_medications();
        assert!(meds.contains(&"metformin".to_string()));
        assert!(meds.contains(&"insulin_glargine".to_string()));
    }
}
