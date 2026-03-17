use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use crate::obligation::{ObligationId, ArticleReference, RegulatoryDomain};
use crate::jurisdiction::JurisdictionId;
use crate::constraint::ConstraintId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictSeverity {
    Critical,
    High,
    Medium,
    Low,
    Informational,
}

impl ConflictSeverity {
    pub fn score(&self) -> f64 {
        match self {
            ConflictSeverity::Critical => 1.0,
            ConflictSeverity::High => 0.8,
            ConflictSeverity::Medium => 0.5,
            ConflictSeverity::Low => 0.3,
            ConflictSeverity::Informational => 0.1,
        }
    }

    pub fn from_score(s: f64) -> Self {
        if s >= 0.9 { ConflictSeverity::Critical }
        else if s >= 0.7 { ConflictSeverity::High }
        else if s >= 0.4 { ConflictSeverity::Medium }
        else if s >= 0.2 { ConflictSeverity::Low }
        else { ConflictSeverity::Informational }
    }
}

impl fmt::Display for ConflictSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConflictSeverity::Critical => write!(f, "CRITICAL"),
            ConflictSeverity::High => write!(f, "HIGH"),
            ConflictSeverity::Medium => write!(f, "MEDIUM"),
            ConflictSeverity::Low => write!(f, "LOW"),
            ConflictSeverity::Informational => write!(f, "INFO"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictCore {
    pub constraint_ids: Vec<ConstraintId>,
    pub obligation_ids: Vec<ObligationId>,
    pub article_refs: Vec<ArticleReference>,
    pub jurisdictions: Vec<JurisdictionId>,
    pub domains: Vec<RegulatoryDomain>,
    pub is_minimal: bool,
}

impl ConflictCore {
    pub fn new(constraint_ids: Vec<ConstraintId>) -> Self {
        ConflictCore {
            constraint_ids, obligation_ids: Vec::new(), article_refs: Vec::new(),
            jurisdictions: Vec::new(), domains: Vec::new(), is_minimal: false,
        }
    }

    pub fn with_obligations(mut self, obls: Vec<ObligationId>) -> Self {
        self.obligation_ids = obls;
        self
    }

    pub fn with_articles(mut self, refs: Vec<ArticleReference>) -> Self {
        self.article_refs = refs;
        self
    }

    pub fn with_jurisdictions(mut self, jurs: Vec<JurisdictionId>) -> Self {
        self.jurisdictions = jurs;
        self
    }

    pub fn mark_minimal(mut self) -> Self {
        self.is_minimal = true;
        self
    }

    pub fn size(&self) -> usize { self.constraint_ids.len() }

    pub fn involves_jurisdiction(&self, jid: &JurisdictionId) -> bool {
        self.jurisdictions.contains(jid)
    }

    pub fn cross_jurisdictional(&self) -> bool { self.jurisdictions.len() > 1 }

    pub fn human_readable_explanation(&self) -> String {
        let mut explanation = String::new();
        if self.is_minimal {
            explanation.push_str("MINIMAL UNSATISFIABLE SUBSET (MUS):\n");
        } else {
            explanation.push_str("UNSATISFIABLE CORE:\n");
        }
        explanation.push_str(&format!("  {} conflicting constraints across {} jurisdiction(s)\n",
            self.size(), self.jurisdictions.len()));

        if !self.article_refs.is_empty() {
            explanation.push_str("\n  Conflicting articles:\n");
            for (i, art) in self.article_refs.iter().enumerate() {
                explanation.push_str(&format!("    {}. {} - {}\n", i + 1, art.full_reference(), art.title));
            }
        }

        if self.cross_jurisdictional() {
            explanation.push_str("\n  Cross-jurisdictional conflict between:\n");
            for j in &self.jurisdictions {
                explanation.push_str(&format!("    - {}\n", j));
            }
        }

        if self.is_minimal {
            explanation.push_str("\n  Removing ANY ONE of these constraints resolves the conflict.\n");
        }

        explanation
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationSuggestion {
    pub id: String,
    pub description: String,
    pub constraint_to_relax: ConstraintId,
    pub obligation_to_waive: Option<ObligationId>,
    pub estimated_cost_saving: f64,
    pub residual_risk_increase: f64,
    pub affected_jurisdictions: Vec<JurisdictionId>,
    pub precedence: usize,
}

impl RemediationSuggestion {
    pub fn new(id: &str, description: &str, constraint: ConstraintId) -> Self {
        RemediationSuggestion {
            id: id.to_string(), description: description.to_string(),
            constraint_to_relax: constraint, obligation_to_waive: None,
            estimated_cost_saving: 0.0, residual_risk_increase: 0.0,
            affected_jurisdictions: Vec::new(), precedence: 0,
        }
    }

    pub fn with_cost_saving(mut self, saving: f64) -> Self {
        self.estimated_cost_saving = saving;
        self
    }

    pub fn with_risk_increase(mut self, risk: f64) -> Self {
        self.residual_risk_increase = risk;
        self
    }

    pub fn net_benefit(&self) -> f64 {
        self.estimated_cost_saving - self.residual_risk_increase * 1_000_000.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryDiagnosis {
    pub diagnosis_id: String,
    pub severity: ConflictSeverity,
    pub conflict_cores: Vec<ConflictCore>,
    pub remediation_suggestions: Vec<RemediationSuggestion>,
    pub summary: String,
    pub affected_obligations: usize,
    pub affected_jurisdictions: usize,
    pub metadata: HashMap<String, String>,
}

impl RegulatoryDiagnosis {
    pub fn new(severity: ConflictSeverity, summary: &str) -> Self {
        RegulatoryDiagnosis {
            diagnosis_id: uuid::Uuid::new_v4().to_string(),
            severity, conflict_cores: Vec::new(),
            remediation_suggestions: Vec::new(),
            summary: summary.to_string(),
            affected_obligations: 0, affected_jurisdictions: 0,
            metadata: HashMap::new(),
        }
    }

    pub fn add_conflict_core(&mut self, core: ConflictCore) {
        self.affected_obligations += core.obligation_ids.len();
        let new_jurs: Vec<_> = core.jurisdictions.iter()
            .filter(|j| !self.conflict_cores.iter().any(|c| c.jurisdictions.contains(j)))
            .cloned().collect();
        self.affected_jurisdictions += new_jurs.len();
        self.conflict_cores.push(core);
    }

    pub fn add_remediation(&mut self, suggestion: RemediationSuggestion) {
        self.remediation_suggestions.push(suggestion);
    }

    pub fn sorted_remediations(&self) -> Vec<&RemediationSuggestion> {
        let mut sorted: Vec<_> = self.remediation_suggestions.iter().collect();
        sorted.sort_by(|a, b| b.net_benefit().partial_cmp(&a.net_benefit()).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    pub fn is_cross_jurisdictional(&self) -> bool {
        self.conflict_cores.iter().any(|c| c.cross_jurisdictional())
    }

    pub fn total_conflicts(&self) -> usize { self.conflict_cores.len() }
}

impl fmt::Display for RegulatoryDiagnosis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} ({} conflicts, {} jurisdictions)",
            self.severity, self.summary, self.total_conflicts(), self.affected_jurisdictions)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosisReport {
    pub title: String,
    pub diagnoses: Vec<RegulatoryDiagnosis>,
    pub total_obligations_analyzed: usize,
    pub total_conflicts_found: usize,
    pub cross_jurisdictional_conflicts: usize,
    pub generated_at: String,
}

impl DiagnosisReport {
    pub fn new(title: &str) -> Self {
        DiagnosisReport {
            title: title.to_string(), diagnoses: Vec::new(),
            total_obligations_analyzed: 0, total_conflicts_found: 0,
            cross_jurisdictional_conflicts: 0,
            generated_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    pub fn add_diagnosis(&mut self, diagnosis: RegulatoryDiagnosis) {
        self.total_conflicts_found += diagnosis.total_conflicts();
        if diagnosis.is_cross_jurisdictional() { self.cross_jurisdictional_conflicts += 1; }
        self.diagnoses.push(diagnosis);
    }

    pub fn by_severity(&self) -> HashMap<String, Vec<&RegulatoryDiagnosis>> {
        let mut grouped: HashMap<String, Vec<&RegulatoryDiagnosis>> = HashMap::new();
        for d in &self.diagnoses {
            grouped.entry(d.severity.to_string()).or_default().push(d);
        }
        grouped
    }

    pub fn has_critical(&self) -> bool {
        self.diagnoses.iter().any(|d| d.severity == ConflictSeverity::Critical)
    }

    pub fn executive_summary(&self) -> String {
        let mut summary = format!("Regulatory Diagnosis Report: {}\n", self.title);
        summary.push_str(&format!("Generated: {}\n", self.generated_at));
        summary.push_str(&format!("Total obligations analyzed: {}\n", self.total_obligations_analyzed));
        summary.push_str(&format!("Total conflicts found: {}\n", self.total_conflicts_found));
        summary.push_str(&format!("Cross-jurisdictional conflicts: {}\n", self.cross_jurisdictional_conflicts));
        let severity_counts: HashMap<String, usize> = self.diagnoses.iter().fold(HashMap::new(), |mut acc, d| {
            *acc.entry(d.severity.to_string()).or_insert(0) += 1;
            acc
        });
        for (sev, count) in &severity_counts {
            summary.push_str(&format!("  {}: {} diagnosis(es)\n", sev, count));
        }
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conflict_core() {
        let core = ConflictCore::new(vec![ConstraintId::new("c1"), ConstraintId::new("c2")])
            .with_jurisdictions(vec![JurisdictionId::new("EU"), JurisdictionId::new("US")])
            .mark_minimal();
        assert!(core.cross_jurisdictional());
        assert!(core.is_minimal);
        assert_eq!(core.size(), 2);
    }

    #[test]
    fn test_diagnosis_report() {
        let mut report = DiagnosisReport::new("Test Report");
        let mut diag = RegulatoryDiagnosis::new(ConflictSeverity::High, "Test conflict");
        diag.add_conflict_core(ConflictCore::new(vec![ConstraintId::new("c1")]));
        report.add_diagnosis(diag);
        assert_eq!(report.total_conflicts_found, 1);
    }

    #[test]
    fn test_remediation_benefit() {
        let rem = RemediationSuggestion::new("r1", "Relax constraint", ConstraintId::new("c1"))
            .with_cost_saving(100_000.0)
            .with_risk_increase(0.01);
        assert!(rem.net_benefit() > 0.0);
    }
}
