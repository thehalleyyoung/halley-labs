//! Morphological rule definitions and inflection tables.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A morphological rule defining how a word form changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphologicalRule {
    pub name: String,
    pub input_features: HashMap<String, String>,
    pub output_features: HashMap<String, String>,
    pub applies_to: Vec<String>,
    pub priority: i32,
}

impl MorphologicalRule {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            input_features: HashMap::new(),
            output_features: HashMap::new(),
            applies_to: Vec::new(),
            priority: 0,
        }
    }

    pub fn with_input(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.input_features.insert(key.into(), val.into());
        self
    }

    pub fn with_output(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.output_features.insert(key.into(), val.into());
        self
    }

    pub fn with_applies_to(mut self, pos: impl Into<String>) -> Self {
        self.applies_to.push(pos.into());
        self
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Check if this rule matches the given features.
    pub fn matches(&self, features: &HashMap<String, String>) -> bool {
        self.input_features.iter().all(|(k, v)| {
            features.get(k).map(|fv| fv == v).unwrap_or(false)
        })
    }

    /// Apply the rule by producing the output features.
    pub fn apply(&self, features: &mut HashMap<String, String>) {
        for (k, v) in &self.output_features {
            features.insert(k.clone(), v.clone());
        }
    }
}

/// Type of inflection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InflectionType {
    PastTense,
    PastParticiple,
    PresentParticiple,
    ThirdSingular,
    Plural,
    Comparative,
    Superlative,
    Possessive,
    Negation,
}

/// A table of inflected forms for a word.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InflectionTable {
    pub base_form: String,
    pub pos: String,
    pub forms: HashMap<InflectionType, String>,
}

impl InflectionTable {
    pub fn new(base: impl Into<String>, pos: impl Into<String>) -> Self {
        Self {
            base_form: base.into(),
            pos: pos.into(),
            forms: HashMap::new(),
        }
    }

    pub fn add_form(&mut self, inflection: InflectionType, form: impl Into<String>) {
        self.forms.insert(inflection, form.into());
    }

    pub fn get_form(&self, inflection: InflectionType) -> Option<&str> {
        self.forms.get(&inflection).map(|s| s.as_str())
    }

    pub fn has_form(&self, inflection: InflectionType) -> bool {
        self.forms.contains_key(&inflection)
    }

    pub fn verb(base: &str, third_sg: &str, past: &str, pp: &str, pres_p: &str) -> Self {
        let mut t = Self::new(base, "VERB");
        t.add_form(InflectionType::ThirdSingular, third_sg);
        t.add_form(InflectionType::PastTense, past);
        t.add_form(InflectionType::PastParticiple, pp);
        t.add_form(InflectionType::PresentParticiple, pres_p);
        t
    }

    pub fn noun(singular: &str, plural: &str) -> Self {
        let mut t = Self::new(singular, "NOUN");
        t.add_form(InflectionType::Plural, plural);
        t
    }

    pub fn adjective(base: &str, comparative: &str, superlative: &str) -> Self {
        let mut t = Self::new(base, "ADJ");
        t.add_form(InflectionType::Comparative, comparative);
        t.add_form(InflectionType::Superlative, superlative);
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_matching() {
        let rule = MorphologicalRule::new("past_tense")
            .with_input("tense", "present")
            .with_output("tense", "past");
        let mut f = HashMap::new();
        f.insert("tense".to_string(), "present".to_string());
        assert!(rule.matches(&f));
        f.insert("tense".to_string(), "past".to_string());
        assert!(!rule.matches(&f));
    }

    #[test]
    fn test_inflection_table() {
        let t = InflectionTable::verb("write", "writes", "wrote", "written", "writing");
        assert_eq!(t.get_form(InflectionType::PastTense).unwrap(), "wrote");
        assert_eq!(t.get_form(InflectionType::PastParticiple).unwrap(), "written");
        assert!(t.has_form(InflectionType::ThirdSingular));
        assert!(!t.has_form(InflectionType::Plural));
    }
}
