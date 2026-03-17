//! Locale registry for managing supported locales.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Identifier for a locale.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LocaleId(pub String);

impl LocaleId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for LocaleId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Locale configuration defining language-specific rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Locale {
    pub id: LocaleId,
    pub name: String,
    pub language: String,
    pub script: String,
    pub morphological_complexity: MorphologicalComplexity,
    pub supported_transformations: Vec<String>,
    pub agreement_features: Vec<String>,
    pub word_order: WordOrder,
    pub has_grammatical_gender: bool,
    pub has_case_system: bool,
    pub has_articles: bool,
}

/// Morphological complexity level.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MorphologicalComplexity {
    Analytic,
    Moderate,
    Synthetic,
    Agglutinative,
    Polysynthetic,
}

/// Basic word order typology.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WordOrder {
    SVO,
    SOV,
    VSO,
    VOS,
    OVS,
    OSV,
    Free,
}

/// Registry of supported locales.
pub struct LocaleRegistry {
    locales: HashMap<LocaleId, Locale>,
}

impl LocaleRegistry {
    pub fn new() -> Self {
        Self {
            locales: HashMap::new(),
        }
    }

    /// Create a registry with all built-in locales.
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.register(Self::english());
        registry.register(Self::german());
        registry.register(Self::french());
        registry.register(Self::spanish());
        registry.register(Self::japanese());
        registry.register(Self::chinese());
        registry
    }

    pub fn register(&mut self, locale: Locale) {
        self.locales.insert(locale.id.clone(), locale);
    }

    pub fn get(&self, id: &LocaleId) -> Option<&Locale> {
        self.locales.get(id)
    }

    pub fn available_locales(&self) -> Vec<&LocaleId> {
        self.locales.keys().collect()
    }

    pub fn supports_transformation(&self, id: &LocaleId, transformation: &str) -> bool {
        self.locales
            .get(id)
            .map(|l| l.supported_transformations.contains(&transformation.to_string()))
            .unwrap_or(false)
    }

    fn english() -> Locale {
        Locale {
            id: LocaleId::new("en"),
            name: "English".to_string(),
            language: "English".to_string(),
            script: "Latin".to_string(),
            morphological_complexity: MorphologicalComplexity::Analytic,
            supported_transformations: vec![
                "passivization".into(), "clefting".into(), "topicalization".into(),
                "relative_clause_insertion".into(), "relative_clause_deletion".into(),
                "tense_change".into(), "agreement_perturbation".into(),
                "synonym_substitution".into(), "negation_insertion".into(),
                "coordinated_np_reorder".into(), "pp_attachment".into(),
                "adverb_repositioning".into(), "there_insertion".into(),
                "dative_alternation".into(), "embedding_depth_change".into(),
            ],
            agreement_features: vec![
                "number".into(), "person".into(), "tense".into(),
            ],
            word_order: WordOrder::SVO,
            has_grammatical_gender: false,
            has_case_system: false,
            has_articles: true,
        }
    }

    fn german() -> Locale {
        Locale {
            id: LocaleId::new("de"),
            name: "German".to_string(),
            language: "German".to_string(),
            script: "Latin".to_string(),
            morphological_complexity: MorphologicalComplexity::Moderate,
            supported_transformations: vec![
                "passivization".into(), "topicalization".into(),
                "tense_change".into(), "agreement_perturbation".into(),
                "negation_insertion".into(), "synonym_substitution".into(),
                "relative_clause_insertion".into(), "relative_clause_deletion".into(),
                "coordinated_np_reorder".into(),
            ],
            agreement_features: vec![
                "number".into(), "person".into(), "gender".into(), "case".into(),
            ],
            word_order: WordOrder::SOV,
            has_grammatical_gender: true,
            has_case_system: true,
            has_articles: true,
        }
    }

    fn french() -> Locale {
        Locale {
            id: LocaleId::new("fr"),
            name: "French".to_string(),
            language: "French".to_string(),
            script: "Latin".to_string(),
            morphological_complexity: MorphologicalComplexity::Moderate,
            supported_transformations: vec![
                "passivization".into(), "clefting".into(),
                "tense_change".into(), "agreement_perturbation".into(),
                "negation_insertion".into(), "synonym_substitution".into(),
                "relative_clause_insertion".into(),
            ],
            agreement_features: vec![
                "number".into(), "person".into(), "gender".into(),
            ],
            word_order: WordOrder::SVO,
            has_grammatical_gender: true,
            has_case_system: false,
            has_articles: true,
        }
    }

    fn spanish() -> Locale {
        Locale {
            id: LocaleId::new("es"),
            name: "Spanish".to_string(),
            language: "Spanish".to_string(),
            script: "Latin".to_string(),
            morphological_complexity: MorphologicalComplexity::Moderate,
            supported_transformations: vec![
                "passivization".into(), "clefting".into(),
                "tense_change".into(), "agreement_perturbation".into(),
                "negation_insertion".into(), "synonym_substitution".into(),
                "topicalization".into(),
            ],
            agreement_features: vec![
                "number".into(), "person".into(), "gender".into(),
            ],
            word_order: WordOrder::SVO,
            has_grammatical_gender: true,
            has_case_system: false,
            has_articles: true,
        }
    }

    fn japanese() -> Locale {
        Locale {
            id: LocaleId::new("ja"),
            name: "Japanese".to_string(),
            language: "Japanese".to_string(),
            script: "Mixed".to_string(),
            morphological_complexity: MorphologicalComplexity::Agglutinative,
            supported_transformations: vec![
                "passivization".into(), "tense_change".into(),
                "negation_insertion".into(), "synonym_substitution".into(),
                "topicalization".into(),
            ],
            agreement_features: vec![
                "politeness".into(), "tense".into(),
            ],
            word_order: WordOrder::SOV,
            has_grammatical_gender: false,
            has_case_system: true,
            has_articles: false,
        }
    }

    fn chinese() -> Locale {
        Locale {
            id: LocaleId::new("zh"),
            name: "Chinese (Mandarin)".to_string(),
            language: "Chinese".to_string(),
            script: "Hanzi".to_string(),
            morphological_complexity: MorphologicalComplexity::Analytic,
            supported_transformations: vec![
                "topicalization".into(), "synonym_substitution".into(),
                "negation_insertion".into(),
            ],
            agreement_features: Vec::new(),
            word_order: WordOrder::SVO,
            has_grammatical_gender: false,
            has_case_system: false,
            has_articles: false,
        }
    }
}

impl Default for LocaleRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_defaults() {
        let reg = LocaleRegistry::with_defaults();
        assert!(reg.available_locales().len() >= 6);
        assert!(reg.get(&LocaleId::new("en")).is_some());
        assert!(reg.get(&LocaleId::new("de")).is_some());
    }

    #[test]
    fn test_transformation_support() {
        let reg = LocaleRegistry::with_defaults();
        assert!(reg.supports_transformation(&LocaleId::new("en"), "passivization"));
        assert!(reg.supports_transformation(&LocaleId::new("en"), "there_insertion"));
        assert!(!reg.supports_transformation(&LocaleId::new("zh"), "passivization"));
    }

    #[test]
    fn test_english_features() {
        let reg = LocaleRegistry::with_defaults();
        let en = reg.get(&LocaleId::new("en")).unwrap();
        assert!(!en.has_grammatical_gender);
        assert!(en.has_articles);
        assert_eq!(en.supported_transformations.len(), 15);
    }

    #[test]
    fn test_custom_locale() {
        let mut reg = LocaleRegistry::new();
        reg.register(Locale {
            id: LocaleId::new("custom"),
            name: "Custom".to_string(),
            language: "Custom".to_string(),
            script: "Latin".to_string(),
            morphological_complexity: MorphologicalComplexity::Moderate,
            supported_transformations: vec!["test_transform".into()],
            agreement_features: vec![],
            word_order: WordOrder::SVO,
            has_grammatical_gender: false,
            has_case_system: false,
            has_articles: false,
        });
        assert!(reg.supports_transformation(&LocaleId::new("custom"), "test_transform"));
    }
}
