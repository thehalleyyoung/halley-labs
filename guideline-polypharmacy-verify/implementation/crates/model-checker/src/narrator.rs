//! Clinical narrative generation.
//!
//! Translates a formal counterexample trace into a human-readable clinical
//! narrative that a physician can understand.  Includes severity-aware
//! language, timeline formatting, and template-based generation.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    ActionLabel, CypEnzyme, DrugId, GuidelineDocument, LocationKind,
    PTA, SafetyProperty, SafetyPropertyKind, Severity,
};
use crate::counterexample::{CounterExample, TraceStep};

// ---------------------------------------------------------------------------
// Risk level
// ---------------------------------------------------------------------------

/// Risk level associated with a narrative event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RiskLevel {
    None,
    Low,
    Moderate,
    High,
    Critical,
}

impl fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Low => write!(f, "Low"),
            Self::Moderate => write!(f, "Moderate"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

impl RiskLevel {
    /// Convert from a numeric severity score (0.0 = none, 1.0 = critical).
    pub fn from_score(score: f64) -> Self {
        if score >= 0.9 { Self::Critical }
        else if score >= 0.7 { Self::High }
        else if score >= 0.4 { Self::Moderate }
        else if score > 0.0 { Self::Low }
        else { Self::None }
    }

    /// Severity-appropriate warning prefix.
    pub fn warning_prefix(&self) -> &'static str {
        match self {
            Self::Critical => "⚠️ CRITICAL WARNING",
            Self::High => "⚠️ WARNING",
            Self::Moderate => "⚠ CAUTION",
            Self::Low => "Note",
            Self::None => "",
        }
    }

    /// Severity-appropriate language for concentration.
    pub fn concentration_descriptor(&self) -> &'static str {
        match self {
            Self::Critical => "dangerously elevated",
            Self::High => "significantly elevated",
            Self::Moderate => "moderately elevated",
            Self::Low => "slightly elevated",
            Self::None => "within normal range",
        }
    }
}

// ---------------------------------------------------------------------------
// NarrativeEvent
// ---------------------------------------------------------------------------

/// A single event in the clinical narrative timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeEvent {
    /// Day number (1-based).
    pub day: usize,
    /// Hour within the day.
    pub hour: f64,
    /// Human-readable description.
    pub description: String,
    /// Clinical context (why this matters).
    pub clinical_context: String,
    /// Associated risk level.
    pub risk_level: RiskLevel,
    /// Concentration values at this point (if relevant).
    pub concentrations: HashMap<String, f64>,
    /// Drug(s) involved.
    pub drugs_involved: Vec<String>,
}

impl NarrativeEvent {
    /// Format as a timeline entry.
    pub fn format_timeline_entry(&self) -> String {
        let risk_str = if self.risk_level >= RiskLevel::Moderate {
            format!(" [{}]", self.risk_level)
        } else {
            String::new()
        };

        if self.hour < 0.5 {
            format!("Day {}: {}{}", self.day, self.description, risk_str)
        } else {
            format!(
                "Day {} ({:.0}h): {}{}",
                self.day, self.hour, self.description, risk_str
            )
        }
    }
}

impl fmt::Display for NarrativeEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_timeline_entry())
    }
}

// ---------------------------------------------------------------------------
// NarrativeTemplate
// ---------------------------------------------------------------------------

/// A template for generating narrative text with placeholder substitution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeTemplate {
    pub name: String,
    pub template: String,
    pub required_fields: Vec<String>,
}

impl NarrativeTemplate {
    /// Create a new template.
    pub fn new(name: &str, template: &str, fields: Vec<&str>) -> Self {
        Self {
            name: name.to_string(),
            template: template.to_string(),
            required_fields: fields.into_iter().map(String::from).collect(),
        }
    }

    /// Render the template with the given substitutions.
    pub fn render(&self, values: &HashMap<String, String>) -> String {
        let mut result = self.template.clone();
        for (key, value) in values {
            result = result.replace(&format!("{{{{{key}}}}}"), value);
        }
        result
    }

    /// Check that all required fields are present.
    pub fn validate(&self, values: &HashMap<String, String>) -> bool {
        self.required_fields.iter().all(|f| values.contains_key(f))
    }
}

/// Standard narrative templates.
pub fn standard_templates() -> Vec<NarrativeTemplate> {
    vec![
        NarrativeTemplate::new(
            "drug_start",
            "Patient starts {{drug_name}} {{dose}}mg {{route}} {{frequency}} for {{indication}}.",
            vec!["drug_name", "dose", "route", "frequency", "indication"],
        ),
        NarrativeTemplate::new(
            "drug_add",
            "Physician adds {{drug_name}} {{dose}}mg {{route}} {{frequency}} for {{indication}}.",
            vec!["drug_name", "dose", "route", "frequency", "indication"],
        ),
        NarrativeTemplate::new(
            "concentration_rise",
            "{{drug_name}} concentration reaches {{concentration}} μg/mL (therapeutic range: {{range}}).",
            vec!["drug_name", "concentration", "range"],
        ),
        NarrativeTemplate::new(
            "concentration_toxic",
            "{{drug_name}} concentration reaches {{concentration}} μg/mL, exceeding the toxic threshold of {{threshold}} μg/mL. {{mechanism}}",
            vec!["drug_name", "concentration", "threshold", "mechanism"],
        ),
        NarrativeTemplate::new(
            "enzyme_inhibition",
            "{{inhibitor}} inhibits {{enzyme}}, reducing clearance of {{substrate}} by approximately {{percent}}%.",
            vec!["inhibitor", "enzyme", "substrate", "percent"],
        ),
        NarrativeTemplate::new(
            "recommendation",
            "Consider {{action}} to reduce the risk of {{adverse_event}}.",
            vec!["action", "adverse_event"],
        ),
    ]
}

// ---------------------------------------------------------------------------
// ClinicalNarrative
// ---------------------------------------------------------------------------

/// A complete clinical narrative generated from a counterexample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalNarrative {
    /// One-paragraph executive summary.
    pub summary: String,
    /// Ordered timeline of clinical events.
    pub timeline: Vec<NarrativeEvent>,
    /// Clinical recommendation.
    pub recommendation: String,
    /// Overall risk level.
    pub risk_level: RiskLevel,
    /// Drug(s) involved.
    pub drugs: Vec<String>,
    /// Enzyme(s) involved.
    pub enzymes: Vec<String>,
    /// Guideline references.
    pub guideline_references: Vec<String>,
}

impl ClinicalNarrative {
    /// Format the full narrative as a report.
    pub fn format_report(&self) -> String {
        let mut lines = Vec::new();

        // Header.
        lines.push("═══════════════════════════════════════════════════════".to_string());
        lines.push("  CLINICAL DRUG INTERACTION NARRATIVE".to_string());
        lines.push(format!("  Risk Level: {}", self.risk_level));
        lines.push("═══════════════════════════════════════════════════════".to_string());
        lines.push(String::new());

        // Summary.
        lines.push("SUMMARY".to_string());
        lines.push("───────".to_string());
        lines.push(self.summary.clone());
        lines.push(String::new());

        // Timeline.
        lines.push("TIMELINE".to_string());
        lines.push("────────".to_string());
        for event in &self.timeline {
            lines.push(format!("  {}", event.format_timeline_entry()));
            if !event.clinical_context.is_empty() {
                lines.push(format!("    → {}", event.clinical_context));
            }
        }
        lines.push(String::new());

        // Recommendation.
        lines.push("RECOMMENDATION".to_string());
        lines.push("──────────────".to_string());
        lines.push(self.recommendation.clone());
        lines.push(String::new());

        // Guideline references.
        if !self.guideline_references.is_empty() {
            lines.push("REFERENCES".to_string());
            lines.push("──────────".to_string());
            for r in &self.guideline_references {
                lines.push(format!("  • {r}"));
            }
        }

        lines.join("\n")
    }

    /// Duration of the narrative in days.
    pub fn duration_days(&self) -> usize {
        self.timeline
            .iter()
            .map(|e| e.day)
            .max()
            .unwrap_or(0)
    }
}

impl fmt::Display for ClinicalNarrative {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_report())
    }
}

// ---------------------------------------------------------------------------
// ClinicalNarrator
// ---------------------------------------------------------------------------

/// Generates clinical narratives from counterexamples.
#[derive(Debug, Clone)]
pub struct ClinicalNarrator {
    templates: Vec<NarrativeTemplate>,
    severity_threshold: f64,
}

impl Default for ClinicalNarrator {
    fn default() -> Self {
        Self {
            templates: standard_templates(),
            severity_threshold: 0.5,
        }
    }
}

impl ClinicalNarrator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_templates(mut self, templates: Vec<NarrativeTemplate>) -> Self {
        self.templates = templates;
        self
    }

    /// Generate a clinical narrative from a counterexample.
    pub fn narrate(
        &self,
        cx: &CounterExample,
        guidelines: &[GuidelineDocument],
    ) -> ClinicalNarrative {
        let timeline = self.build_timeline(cx);
        let risk_level = self.assess_overall_risk(cx);
        let summary = self.generate_summary(cx, &timeline, risk_level);
        let recommendation = self.generate_recommendation(cx, risk_level, guidelines);

        let drugs: Vec<String> = cx
            .steps
            .iter()
            .flat_map(|s| s.concentrations.keys().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let enzymes = self.extract_enzyme_names(cx);

        let guideline_refs: Vec<String> = guidelines
            .iter()
            .map(|g| format!("{}: {}", g.name, g.description))
            .collect();

        ClinicalNarrative {
            summary,
            timeline,
            recommendation,
            risk_level,
            drugs,
            enzymes,
            guideline_references: guideline_refs,
        }
    }

    /// Build the narrative timeline from counterexample steps.
    fn build_timeline(&self, cx: &CounterExample) -> Vec<NarrativeEvent> {
        let mut events = Vec::new();

        for step in &cx.steps {
            let day = (step.time / 24.0).floor() as usize + 1;
            let hour = step.time % 24.0;

            // Skip boring time-elapse steps.
            if step.action_taken == "time elapse" && step.step != cx.violation_step {
                // Only include if concentration changed significantly.
                let significant = self.is_significant_change(cx, step);
                if !significant {
                    continue;
                }
            }

            let risk_level = self.assess_step_risk(cx, step);
            let (description, context) =
                self.describe_step(cx, step, day);

            events.push(NarrativeEvent {
                day,
                hour,
                description,
                clinical_context: context,
                risk_level,
                concentrations: step.concentrations.clone(),
                drugs_involved: step
                    .concentrations
                    .keys()
                    .cloned()
                    .collect(),
            });
        }

        events
    }

    /// Describe a single trace step in clinical language.
    fn describe_step(
        &self,
        cx: &CounterExample,
        step: &TraceStep,
        day: usize,
    ) -> (String, String) {
        if step.step == cx.violation_step {
            return self.describe_violation(cx, step, day);
        }

        if step.action_taken == "initial state" {
            let drug_name = self.format_drug_name(&cx.drug_id);
            let concs: Vec<String> = step
                .concentrations
                .iter()
                .map(|(k, v)| format!("{}: {:.1} μg/mL", self.format_drug_name_from_var(k), v))
                .collect();
            return (
                format!("Patient baseline established. Drug: {}.", drug_name),
                format!("Initial concentrations: {}", concs.join(", ")),
            );
        }

        if step.action_taken.starts_with("administer") {
            let drug_name = self.extract_drug_from_action(&step.action_taken);
            return (
                format!("Patient receives dose of {}.", drug_name),
                "Plasma concentration expected to rise.".to_string(),
            );
        }

        if step.action_taken.starts_with("absorb") {
            let drug_name = self.extract_drug_from_action(&step.action_taken);
            let conc = step
                .concentrations
                .values()
                .next()
                .copied()
                .unwrap_or(0.0);
            return (
                format!("{} reaches absorption phase. Current concentration: {:.2} μg/mL.", drug_name, conc),
                "Drug is being absorbed from the GI tract.".to_string(),
            );
        }

        if step.action_taken.starts_with("inhibit") {
            let parts = self.parse_enzyme_action(&step.action_taken);
            return (
                format!("Enzyme inhibition detected: {} by {}.", parts.0, parts.1),
                "This may reduce clearance of co-administered drugs.".to_string(),
            );
        }

        if step.action_taken.starts_with("eliminate") {
            let drug_name = self.extract_drug_from_action(&step.action_taken);
            return (
                format!("{} undergoes hepatic elimination.", drug_name),
                "Drug concentration decreasing.".to_string(),
            );
        }

        // Generic step.
        let conc_summary: Vec<String> = step
            .concentrations
            .iter()
            .map(|(k, v)| format!("{}: {:.2}", self.format_drug_name_from_var(k), v))
            .collect();

        (
            step.action_taken.clone(),
            format!("Concentrations: {} μg/mL", conc_summary.join(", ")),
        )
    }

    /// Describe the violation step with urgent clinical language.
    fn describe_violation(
        &self,
        cx: &CounterExample,
        step: &TraceStep,
        day: usize,
    ) -> (String, String) {
        let risk = self.assess_step_risk(cx, step);

        let conc_details: Vec<String> = step
            .concentrations
            .iter()
            .map(|(drug, &val)| {
                format!(
                    "{} concentration reaches {:.1} μg/mL ({})",
                    self.format_drug_name_from_var(drug),
                    val,
                    risk.concentration_descriptor()
                )
            })
            .collect();

        let description = format!(
            "{}: {}",
            risk.warning_prefix(),
            conc_details.join("; ")
        );

        let context = self.generate_violation_context(cx, step);

        (description, context)
    }

    /// Generate clinical context for the violation.
    fn generate_violation_context(&self, cx: &CounterExample, step: &TraceStep) -> String {
        let mut contexts = Vec::new();

        // Check for enzyme interaction.
        let enzyme_mentioned = cx
            .steps
            .iter()
            .any(|s| s.action_taken.contains("inhibit") || s.action_taken.contains("induce"));

        if enzyme_mentioned {
            contexts.push(
                "Drug interaction via CYP enzyme inhibition is the likely cause of elevated concentrations.".to_string()
            );
        }

        // Check concentration trajectory.
        if cx.steps.len() >= 2 {
            if let Some(prev) = cx.pre_violation_step() {
                for (drug, &curr_val) in &step.concentrations {
                    if let Some(&prev_val) = prev.concentrations.get(drug) {
                        if prev_val > 0.0 {
                            let pct_change = ((curr_val - prev_val) / prev_val) * 100.0;
                            if pct_change > 20.0 {
                                contexts.push(format!(
                                    "Rapid concentration increase ({:.0}% since previous measurement) suggests impaired clearance.",
                                    pct_change
                                ));
                            }
                        }
                    }
                }
            }
        }

        if contexts.is_empty() {
            contexts.push("Concentration exceeds safe therapeutic range.".to_string());
        }

        contexts.join(" ")
    }

    /// Generate the executive summary.
    fn generate_summary(
        &self,
        cx: &CounterExample,
        timeline: &[NarrativeEvent],
        risk: RiskLevel,
    ) -> String {
        let drug_name = self.format_drug_name(&cx.drug_id);
        let duration_days = cx.duration_days();

        let violation_desc = if let Some(v) = cx.violation() {
            let peak: f64 = v
                .concentrations
                .values()
                .copied()
                .fold(0.0_f64, f64::max);
            format!(
                "At {:.1} hours (day {}), {} concentration reaches {:.1} μg/mL, which is {}.",
                v.time,
                (v.time / 24.0).floor() as usize + 1,
                drug_name,
                peak,
                risk.concentration_descriptor()
            )
        } else {
            "A safety property violation was detected.".to_string()
        };

        format!(
            "A potential drug interaction has been identified involving {}. \
             {} \
             Risk level: {}. \
             Formal verification over {} simulation steps ({:.1} days) detected this interaction.",
            drug_name,
            violation_desc,
            risk,
            cx.len(),
            duration_days as f64
        )
    }

    /// Generate a clinical recommendation.
    fn generate_recommendation(
        &self,
        cx: &CounterExample,
        risk: RiskLevel,
        guidelines: &[GuidelineDocument],
    ) -> String {
        let mut recommendations = Vec::new();

        match risk {
            RiskLevel::Critical => {
                recommendations.push(
                    "IMMEDIATE ACTION REQUIRED: This drug combination poses a critical safety risk.".to_string()
                );
                recommendations.push(
                    "Consider discontinuing one of the interacting drugs or substituting with an alternative that does not share the same metabolic pathway.".to_string()
                );
                recommendations.push(
                    "If combination is clinically necessary, initiate therapeutic drug monitoring (TDM) with reduced dosing.".to_string()
                );
            }
            RiskLevel::High => {
                recommendations.push(
                    "HIGH RISK: Close monitoring is recommended for this drug combination.".to_string()
                );
                recommendations.push(
                    "Consider dose reduction of the affected drug and implement therapeutic drug monitoring.".to_string()
                );
            }
            RiskLevel::Moderate => {
                recommendations.push(
                    "MODERATE RISK: Monitor patient for signs of drug interaction.".to_string()
                );
                recommendations.push(
                    "Review drug doses and consider alternatives if symptoms develop.".to_string()
                );
            }
            RiskLevel::Low => {
                recommendations.push(
                    "LOW RISK: Standard monitoring is sufficient.".to_string()
                );
            }
            RiskLevel::None => {
                recommendations.push(
                    "No specific action required based on this analysis.".to_string()
                );
            }
        }

        // Add guideline-specific recommendations.
        for gl in guidelines {
            for rec in &gl.recommendations {
                recommendations.push(format!("Per {}: {}", gl.name, rec));
            }
        }

        recommendations.join("\n")
    }

    /// Assess the overall risk level of a counterexample.
    fn assess_overall_risk(&self, cx: &CounterExample) -> RiskLevel {
        if cx.is_empty() {
            return RiskLevel::None;
        }

        let max_risk = cx
            .steps
            .iter()
            .map(|s| self.assess_step_risk(cx, s))
            .max()
            .unwrap_or(RiskLevel::None);

        max_risk
    }

    /// Assess the risk level at a single step.
    fn assess_step_risk(&self, cx: &CounterExample, step: &TraceStep) -> RiskLevel {
        if step.step == cx.violation_step {
            return RiskLevel::High;
        }

        // Check concentrations against typical thresholds.
        let max_conc = step
            .concentrations
            .values()
            .copied()
            .fold(0.0_f64, f64::max);

        if max_conc > 20.0 {
            RiskLevel::Critical
        } else if max_conc > 10.0 {
            RiskLevel::High
        } else if max_conc > 5.0 {
            RiskLevel::Moderate
        } else if max_conc > 0.0 {
            RiskLevel::Low
        } else {
            RiskLevel::None
        }
    }

    /// Check if a step represents a significant concentration change.
    fn is_significant_change(&self, cx: &CounterExample, step: &TraceStep) -> bool {
        if step.step == 0 {
            return true;
        }

        let prev = cx.steps.get(step.step.saturating_sub(1));
        if let Some(prev_step) = prev {
            for (drug, &val) in &step.concentrations {
                if let Some(&prev_val) = prev_step.concentrations.get(drug) {
                    let change = (val - prev_val).abs();
                    if change > self.severity_threshold {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Format a DrugId into a human-readable name.
    fn format_drug_name(&self, drug_id: &DrugId) -> String {
        let name = drug_id.as_str();
        let mut chars = name.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => {
                let mut formatted = first.to_uppercase().to_string();
                formatted.extend(chars);
                formatted.replace('_', " ")
            }
        }
    }

    /// Format a variable name like "conc_metformin" into "Metformin".
    fn format_drug_name_from_var(&self, var_name: &str) -> String {
        let name = var_name
            .strip_prefix("conc_")
            .or_else(|| var_name.strip_prefix("concentration_"))
            .unwrap_or(var_name);
        let mut chars = name.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => {
                let mut formatted = first.to_uppercase().to_string();
                formatted.extend(chars);
                formatted.replace('_', " ")
            }
        }
    }

    /// Extract drug name from an action string like "administer(metformin)".
    fn extract_drug_from_action(&self, action: &str) -> String {
        if let Some(start) = action.find('(') {
            if let Some(end) = action.find(')') {
                let name = &action[start + 1..end];
                return self.format_drug_name(&DrugId::new(name));
            }
        }
        action.to_string()
    }

    /// Parse enzyme action like "inhibit(CYP3A4, drug)" → (enzyme, drug).
    fn parse_enzyme_action(&self, action: &str) -> (String, String) {
        if let Some(start) = action.find('(') {
            if let Some(end) = action.find(')') {
                let inner = &action[start + 1..end];
                let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 2 {
                    return (parts[0].to_string(), parts[1].to_string());
                } else if !parts.is_empty() {
                    return (parts[0].to_string(), "unknown".to_string());
                }
            }
        }
        ("unknown enzyme".to_string(), "unknown drug".to_string())
    }

    /// Extract enzyme names mentioned in the counterexample.
    fn extract_enzyme_names(&self, cx: &CounterExample) -> Vec<String> {
        let mut enzymes = Vec::new();
        for step in &cx.steps {
            if step.action_taken.contains("CYP") {
                let (enzyme, _) = self.parse_enzyme_action(&step.action_taken);
                if !enzymes.contains(&enzyme) {
                    enzymes.push(enzyme);
                }
            }
        }
        enzymes
    }
}

// ---------------------------------------------------------------------------
// Concentration formatting
// ---------------------------------------------------------------------------

/// Format a concentration trajectory for a drug as a table.
pub fn format_concentration_table(
    cx: &CounterExample,
    drug: &str,
    threshold: Option<f64>,
) -> String {
    let mut lines = Vec::new();
    lines.push(format!("Concentration trajectory for {}:", drug));
    lines.push("──────────────────────────────────────".to_string());
    lines.push(format!("{:>8} {:>12} {:>8}", "Time(h)", "Conc(μg/mL)", "Status"));
    lines.push("──────────────────────────────────────".to_string());

    for step in &cx.steps {
        if let Some(&conc) = step.concentrations.get(drug) {
            let status = match threshold {
                Some(t) if conc > t => "⚠️ TOXIC",
                Some(t) if conc > t * 0.8 => "⚡ HIGH",
                _ => "✓ OK",
            };
            lines.push(format!("{:>8.1} {:>12.2} {:>8}", step.time, conc, status));
        }
    }

    lines.join("\n")
}

/// Format a timeline of the counterexample showing only key events.
pub fn format_key_events(cx: &CounterExample) -> String {
    let narrator = ClinicalNarrator::new();
    let timeline = narrator.build_timeline(cx);

    let mut lines = Vec::new();
    lines.push("Key Events:".to_string());

    for event in &timeline {
        if event.risk_level >= RiskLevel::Low {
            lines.push(format!("  {}", event.format_timeline_entry()));
        }
    }

    if lines.len() == 1 {
        lines.push("  No significant events detected.".to_string());
    }

    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counterexample::CounterExample;
    use crate::DrugId;

    fn sample_cx() -> CounterExample {
        let steps = vec![
            TraceStep {
                step: 0,
                time: 0.0,
                location: 0,
                location_name: "idle".into(),
                clock_values: vec![0.0],
                variable_values: vec![0.0],
                concentrations: [("conc_metformin".into(), 0.0)].into(),
                clinical_state: HashMap::new(),
                action_taken: "initial state".into(),
            },
            TraceStep {
                step: 1,
                time: 24.0,
                location: 1,
                location_name: "absorbing".into(),
                clock_values: vec![24.0],
                variable_values: vec![4.0],
                concentrations: [("conc_metformin".into(), 4.0)].into(),
                clinical_state: HashMap::new(),
                action_taken: "administer(metformin)".into(),
            },
            TraceStep {
                step: 2,
                time: 336.0, // day 14
                location: 2,
                location_name: "steady_state".into(),
                clock_values: vec![336.0],
                variable_values: vec![5.5],
                concentrations: [("conc_metformin".into(), 5.5)].into(),
                clinical_state: HashMap::new(),
                action_taken: "inhibit(CYP3A4, clarithromycin)".into(),
            },
            TraceStep {
                step: 3,
                time: 432.0, // day 18
                location: 3,
                location_name: "toxic".into(),
                clock_values: vec![432.0],
                variable_values: vec![6.2],
                concentrations: [("conc_metformin".into(), 6.2)].into(),
                clinical_state: HashMap::new(),
                action_taken: "concentration exceeds threshold".into(),
            },
        ];

        CounterExample {
            steps,
            violation_step: 3,
            violation_property: "conc_bound".into(),
            property_description: "Metformin ≤ 5.0 μg/mL".into(),
            total_time: 432.0,
            drug_id: DrugId::new("metformin"),
        }
    }

    #[test]
    fn test_risk_level_from_score() {
        assert_eq!(RiskLevel::from_score(0.0), RiskLevel::None);
        assert_eq!(RiskLevel::from_score(0.5), RiskLevel::Moderate);
        assert_eq!(RiskLevel::from_score(0.8), RiskLevel::High);
        assert_eq!(RiskLevel::from_score(0.95), RiskLevel::Critical);
    }

    #[test]
    fn test_risk_level_ordering() {
        assert!(RiskLevel::None < RiskLevel::Low);
        assert!(RiskLevel::Low < RiskLevel::Moderate);
        assert!(RiskLevel::Moderate < RiskLevel::High);
        assert!(RiskLevel::High < RiskLevel::Critical);
    }

    #[test]
    fn test_narrative_template_render() {
        let template = NarrativeTemplate::new(
            "test",
            "Patient starts {{drug}} at {{dose}}mg.",
            vec!["drug", "dose"],
        );
        let mut values = HashMap::new();
        values.insert("drug".into(), "Metformin".into());
        values.insert("dose".into(), "500".into());
        let rendered = template.render(&values);
        assert_eq!(rendered, "Patient starts Metformin at 500mg.");
    }

    #[test]
    fn test_narrative_template_validate() {
        let template = NarrativeTemplate::new(
            "test",
            "{{a}} and {{b}}",
            vec!["a", "b"],
        );
        let mut values = HashMap::new();
        values.insert("a".into(), "x".into());
        assert!(!template.validate(&values));
        values.insert("b".into(), "y".into());
        assert!(template.validate(&values));
    }

    #[test]
    fn test_narrate_basic() {
        let cx = sample_cx();
        let narrator = ClinicalNarrator::new();
        let narrative = narrator.narrate(&cx, &[]);
        assert!(!narrative.summary.is_empty());
        assert!(!narrative.timeline.is_empty());
        assert!(!narrative.recommendation.is_empty());
    }

    #[test]
    fn test_narrate_with_guidelines() {
        let cx = sample_cx();
        let guidelines = vec![GuidelineDocument {
            id: "beers".into(),
            name: "Beers Criteria".into(),
            drugs: vec![DrugId::new("metformin")],
            description: "Inappropriate medications for older adults".into(),
            recommendations: vec!["Monitor renal function when using metformin".into()],
        }];
        let narrator = ClinicalNarrator::new();
        let narrative = narrator.narrate(&cx, &guidelines);
        assert!(narrative.guideline_references.iter().any(|r| r.contains("Beers")));
    }

    #[test]
    fn test_narrative_report_format() {
        let cx = sample_cx();
        let narrator = ClinicalNarrator::new();
        let narrative = narrator.narrate(&cx, &[]);
        let report = narrative.format_report();
        assert!(report.contains("CLINICAL DRUG INTERACTION NARRATIVE"));
        assert!(report.contains("SUMMARY"));
        assert!(report.contains("TIMELINE"));
        assert!(report.contains("RECOMMENDATION"));
    }

    #[test]
    fn test_risk_assessment() {
        let cx = sample_cx();
        let narrator = ClinicalNarrator::new();
        let risk = narrator.assess_overall_risk(&cx);
        assert!(risk >= RiskLevel::Moderate);
    }

    #[test]
    fn test_format_drug_name() {
        let narrator = ClinicalNarrator::new();
        assert_eq!(narrator.format_drug_name(&DrugId::new("metformin")), "Metformin");
        assert_eq!(narrator.format_drug_name_from_var("conc_metformin"), "Metformin");
    }

    #[test]
    fn test_extract_drug_from_action() {
        let narrator = ClinicalNarrator::new();
        let name = narrator.extract_drug_from_action("administer(warfarin)");
        assert_eq!(name, "Warfarin");
    }

    #[test]
    fn test_concentration_table() {
        let cx = sample_cx();
        let table = format_concentration_table(&cx, "conc_metformin", Some(5.0));
        assert!(table.contains("Concentration trajectory"));
        assert!(table.contains("TOXIC") || table.contains("HIGH"));
    }

    #[test]
    fn test_format_key_events() {
        let cx = sample_cx();
        let events = format_key_events(&cx);
        assert!(events.contains("Key Events"));
    }

    #[test]
    fn test_standard_templates() {
        let templates = standard_templates();
        assert!(!templates.is_empty());
        assert!(templates.iter().any(|t| t.name == "drug_start"));
        assert!(templates.iter().any(|t| t.name == "concentration_toxic"));
    }

    #[test]
    fn test_narrative_duration() {
        let cx = sample_cx();
        let narrator = ClinicalNarrator::new();
        let narrative = narrator.narrate(&cx, &[]);
        assert!(narrative.duration_days() > 0);
    }

    #[test]
    fn test_parse_enzyme_action() {
        let narrator = ClinicalNarrator::new();
        let (enzyme, drug) = narrator.parse_enzyme_action("inhibit(CYP3A4, clarithromycin)");
        assert_eq!(enzyme, "CYP3A4");
        assert_eq!(drug, "clarithromycin");
    }
}
