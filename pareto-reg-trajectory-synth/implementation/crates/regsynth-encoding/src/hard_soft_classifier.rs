use serde::{Deserialize, Serialize};
use crate::obligation_encoder::{RawObligation, ObligationKind};

/// Classifies obligations as hard or soft constraints based on jurisdiction type,
/// enforcement mechanism, and penalty severity.
#[derive(Debug, Clone)]
pub struct HardSoftClassifier {
    binding_threshold: f64,
    penalty_threshold: f64,
    jurisdiction_overrides: std::collections::HashMap<String, bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintClassification {
    Hard,
    Soft,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifiedObligation {
    pub obligation_id: String,
    pub classification: ConstraintClassification,
    pub weight: f64,
    pub reason: String,
}

impl HardSoftClassifier {
    pub fn new() -> Self {
        HardSoftClassifier {
            binding_threshold: 0.7,
            penalty_threshold: 1_000_000.0,
            jurisdiction_overrides: std::collections::HashMap::new(),
        }
    }

    pub fn with_binding_threshold(mut self, threshold: f64) -> Self {
        self.binding_threshold = threshold;
        self
    }

    pub fn with_penalty_threshold(mut self, threshold: f64) -> Self {
        self.penalty_threshold = threshold;
        self
    }

    pub fn override_jurisdiction(&mut self, jurisdiction: &str, is_hard: bool) {
        self.jurisdiction_overrides.insert(jurisdiction.to_string(), is_hard);
    }

    pub fn classify(&self, obl: &RawObligation) -> ClassifiedObligation {
        // Check jurisdiction override first
        if let Some(&forced_hard) = self.jurisdiction_overrides.get(&obl.jurisdiction) {
            return ClassifiedObligation {
                obligation_id: obl.id.clone(),
                classification: if forced_hard { ConstraintClassification::Hard } else { ConstraintClassification::Soft },
                weight: if forced_hard { f64::INFINITY } else { obl.risk_weight },
                reason: format!("Jurisdiction override for {}", obl.jurisdiction),
            };
        }

        // Binding regulations with mandatory obligation types are hard
        if obl.is_binding && (obl.kind == ObligationKind::Obligation || obl.kind == ObligationKind::Prohibition) {
            return ClassifiedObligation {
                obligation_id: obl.id.clone(),
                classification: ConstraintClassification::Hard,
                weight: f64::INFINITY,
                reason: format!("Binding {:?} from {}", obl.kind, obl.jurisdiction),
            };
        }

        // Permissions are always soft
        if obl.kind == ObligationKind::Permission {
            return ClassifiedObligation {
                obligation_id: obl.id.clone(),
                classification: ConstraintClassification::Soft,
                weight: obl.risk_weight * 0.5,
                reason: "Permission type is always soft".to_string(),
            };
        }

        // Non-binding obligations are soft with weight based on risk
        let weight = self.compute_soft_weight(obl);
        ClassifiedObligation {
            obligation_id: obl.id.clone(),
            classification: ConstraintClassification::Soft,
            weight,
            reason: format!("Non-binding obligation with risk weight {:.2}", weight),
        }
    }

    pub fn classify_all(&self, obligations: &[RawObligation]) -> Vec<ClassifiedObligation> {
        obligations.iter().map(|o| self.classify(o)).collect()
    }

    fn compute_soft_weight(&self, obl: &RawObligation) -> f64 {
        let base = obl.risk_weight;
        let jurisdiction_factor = self.jurisdiction_weight(&obl.jurisdiction);
        let kind_factor = match obl.kind {
            ObligationKind::Obligation => 1.0,
            ObligationKind::Prohibition => 0.9,
            ObligationKind::Permission => 0.5,
        };
        base * jurisdiction_factor * kind_factor
    }

    fn jurisdiction_weight(&self, jurisdiction: &str) -> f64 {
        match jurisdiction {
            "EU" => 1.0,
            "US_NIST" | "US" => 0.7,
            "CN" | "China" => 0.8,
            "ISO" => 0.6,
            "GDPR" => 0.9,
            "SG" | "Singapore" => 0.5,
            "UK" => 0.6,
            "KR" | "South Korea" => 0.5,
            "CA" | "Canada" => 0.4,
            "BR" | "Brazil" => 0.4,
            _ => 0.5,
        }
    }

    pub fn summary(&self, classifications: &[ClassifiedObligation]) -> ClassificationSummary {
        let hard_count = classifications.iter().filter(|c| c.classification == ConstraintClassification::Hard).count();
        let soft_count = classifications.iter().filter(|c| c.classification == ConstraintClassification::Soft).count();
        let total_soft_weight: f64 = classifications.iter()
            .filter(|c| c.classification == ConstraintClassification::Soft)
            .map(|c| c.weight)
            .sum();

        ClassificationSummary {
            total: classifications.len(),
            hard_count,
            soft_count,
            total_soft_weight,
            hard_ratio: if classifications.is_empty() { 0.0 } else { hard_count as f64 / classifications.len() as f64 },
        }
    }
}

impl Default for HardSoftClassifier {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationSummary {
    pub total: usize,
    pub hard_count: usize,
    pub soft_count: usize,
    pub total_soft_weight: f64,
    pub hard_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obligation_encoder::ConditionSpec;

    fn make_obl(id: &str, kind: ObligationKind, binding: bool) -> RawObligation {
        RawObligation {
            id: id.to_string(), kind, jurisdiction: "EU".to_string(),
            article_ref: format!("Art.{}", id), description: format!("Test {}", id),
            is_binding: binding, risk_weight: 1.0, conditions: Vec::new(),
            exemptions: Vec::new(), cross_refs: Vec::new(),
        }
    }

    #[test]
    fn test_hard_classification() {
        let classifier = HardSoftClassifier::new();
        let obl = make_obl("h1", ObligationKind::Obligation, true);
        let result = classifier.classify(&obl);
        assert_eq!(result.classification, ConstraintClassification::Hard);
    }

    #[test]
    fn test_soft_classification() {
        let classifier = HardSoftClassifier::new();
        let obl = make_obl("s1", ObligationKind::Permission, false);
        let result = classifier.classify(&obl);
        assert_eq!(result.classification, ConstraintClassification::Soft);
    }

    #[test]
    fn test_classify_all() {
        let classifier = HardSoftClassifier::new();
        let obls = vec![
            make_obl("h1", ObligationKind::Obligation, true),
            make_obl("s1", ObligationKind::Permission, false),
            make_obl("h2", ObligationKind::Prohibition, true),
        ];
        let results = classifier.classify_all(&obls);
        let summary = classifier.summary(&results);
        assert_eq!(summary.hard_count, 2);
        assert_eq!(summary.soft_count, 1);
    }
}
