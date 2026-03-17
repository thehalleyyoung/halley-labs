//! Agreement rules and checking for morphological consistency.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A grammatical agreement pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgreementPattern {
    pub name: String,
    pub controller_pos: String,
    pub target_pos: String,
    pub features: Vec<String>,
    pub description: String,
}

/// A detected agreement violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgreementViolation {
    pub pattern_name: String,
    pub controller_text: String,
    pub target_text: String,
    pub feature: String,
    pub controller_value: String,
    pub target_value: String,
    pub expected_value: String,
    pub position: usize,
}

/// Feature values for agreement checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureBundle {
    pub features: HashMap<String, String>,
}

impl FeatureBundle {
    pub fn new() -> Self {
        Self { features: HashMap::new() }
    }

    pub fn with_feature(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.features.insert(key.into(), value.into());
        self
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.features.get(key).map(|s| s.as_str())
    }

    pub fn agrees_with(&self, other: &FeatureBundle, features: &[String]) -> Vec<String> {
        features.iter().filter(|f| {
            match (self.get(f), other.get(f)) {
                (Some(a), Some(b)) => a != b,
                _ => false,
            }
        }).cloned().collect()
    }
}

impl Default for FeatureBundle {
    fn default() -> Self { Self::new() }
}

/// Checker for grammatical agreement.
pub struct AgreementChecker {
    patterns: Vec<AgreementPattern>,
    violations: Vec<AgreementViolation>,
    total_checks: usize,
}

impl AgreementChecker {
    pub fn new() -> Self {
        Self { patterns: Vec::new(), violations: Vec::new(), total_checks: 0 }
    }

    pub fn english_defaults() -> Self {
        let mut c = Self::new();
        c.add_pattern(AgreementPattern {
            name: "subject_verb_number".into(),
            controller_pos: "NOUN".into(),
            target_pos: "VERB".into(),
            features: vec!["number".into(), "person".into()],
            description: "Subject-verb agreement in number and person".into(),
        });
        c.add_pattern(AgreementPattern {
            name: "determiner_noun_number".into(),
            controller_pos: "NOUN".into(),
            target_pos: "DET".into(),
            features: vec!["number".into()],
            description: "Determiner-noun agreement in number".into(),
        });
        c
    }

    pub fn add_pattern(&mut self, pattern: AgreementPattern) {
        self.patterns.push(pattern);
    }

    pub fn check_pair(
        &mut self, pattern_name: &str,
        controller: &FeatureBundle, controller_text: &str,
        target: &FeatureBundle, target_text: &str, position: usize,
    ) -> Vec<AgreementViolation> {
        self.total_checks += 1;
        let pattern = match self.patterns.iter().find(|p| p.name == pattern_name) {
            Some(p) => p, None => return Vec::new(),
        };
        let disagreements = controller.agrees_with(target, &pattern.features);
        let mut violations = Vec::new();
        for feature in disagreements {
            let cv = controller.get(&feature).unwrap_or("?").to_string();
            let tv = target.get(&feature).unwrap_or("?").to_string();
            let v = AgreementViolation {
                pattern_name: pattern_name.to_string(),
                controller_text: controller_text.to_string(),
                target_text: target_text.to_string(),
                feature: feature.clone(),
                controller_value: cv.clone(),
                target_value: tv,
                expected_value: cv,
                position,
            };
            violations.push(v.clone());
            self.violations.push(v);
        }
        violations
    }

    pub fn check_sentence(
        &mut self, tokens: &[(String, String, FeatureBundle)],
    ) -> Vec<AgreementViolation> {
        let mut all = Vec::new();
        let nouns: Vec<(usize, &str, &FeatureBundle)> = tokens.iter().enumerate()
            .filter(|(_, (_, pos, _))| pos == "NOUN" || pos == "PRON")
            .map(|(i, (text, _, feats))| (i, text.as_str(), feats)).collect();
        let verbs: Vec<(usize, &str, &FeatureBundle)> = tokens.iter().enumerate()
            .filter(|(_, (_, pos, _))| pos == "VERB" || pos == "AUX")
            .map(|(i, (text, _, feats))| (i, text.as_str(), feats)).collect();
        for (v_idx, v_text, v_feats) in &verbs {
            if let Some((_, n_text, n_feats)) = nouns.iter().filter(|(i, _, _)| *i < *v_idx).last() {
                all.extend(self.check_pair("subject_verb_number", n_feats, n_text, v_feats, v_text, *v_idx));
            }
        }
        all
    }

    pub fn violation_count(&self) -> usize { self.violations.len() }
    pub fn total_checks(&self) -> usize { self.total_checks }
    pub fn violations(&self) -> &[AgreementViolation] { &self.violations }
}

impl Default for AgreementChecker {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_bundle_agreement() {
        let a = FeatureBundle::new().with_feature("number", "singular");
        let b = FeatureBundle::new().with_feature("number", "singular");
        assert!(a.agrees_with(&b, &["number".into()]).is_empty());
    }

    #[test]
    fn test_feature_bundle_disagreement() {
        let a = FeatureBundle::new().with_feature("number", "singular");
        let b = FeatureBundle::new().with_feature("number", "plural");
        assert_eq!(a.agrees_with(&b, &["number".into()]), vec!["number"]);
    }

    #[test]
    fn test_checker_violation() {
        let mut c = AgreementChecker::english_defaults();
        let s = FeatureBundle::new().with_feature("number", "singular");
        let v = FeatureBundle::new().with_feature("number", "plural");
        let violations = c.check_pair("subject_verb_number", &s, "cat", &v, "run", 1);
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn test_sentence_check() {
        let mut c = AgreementChecker::english_defaults();
        let tokens = vec![
            ("The".into(), "DET".into(), FeatureBundle::new()),
            ("cat".into(), "NOUN".into(), FeatureBundle::new().with_feature("number", "singular")),
            ("run".into(), "VERB".into(), FeatureBundle::new().with_feature("number", "plural")),
        ];
        assert_eq!(c.check_sentence(&tokens).len(), 1);
    }
}
