//! Linguistic feature system for unification-based grammar checking.
//!
//! Provides Feature enums, FeatureBundle (a map of named features that supports
//! unification), and FeatureStructure (a categorised bundle with a head label).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Atomic feature-value enums ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NumberValue {
    Singular,
    Plural,
    Uncountable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PersonValue {
    First,
    Second,
    Third,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TenseValue {
    Past,
    Present,
    Future,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AspectValue {
    Simple,
    Progressive,
    Perfect,
    PerfectProgressive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VoiceValue {
    Active,
    Passive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MoodValue {
    Indicative,
    Subjunctive,
    Imperative,
    Interrogative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CaseValue {
    Nominative,
    Accusative,
    Genitive,
    Dative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GenderValue {
    Masculine,
    Feminine,
    Neuter,
    Common,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DefinitenessValue {
    Definite,
    Indefinite,
    Bare,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnimacyValue {
    Animate,
    Inanimate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransitivityValue {
    Transitive,
    Intransitive,
    Ditransitive,
    Copular,
    Unaccusative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FinitenessValue {
    Finite,
    NonFinite,
    Infinitive,
    Gerund,
    Participle,
}

// ── Umbrella Feature enum ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Feature {
    Number(NumberValue),
    Person(PersonValue),
    Tense(TenseValue),
    Aspect(AspectValue),
    Voice(VoiceValue),
    Mood(MoodValue),
    Case(CaseValue),
    Gender(GenderValue),
    Definiteness(DefinitenessValue),
    Animacy(AnimacyValue),
    Transitivity(TransitivityValue),
    Finiteness(FinitenessValue),
}

impl Feature {
    /// Return the discriminant name (e.g. "Number", "Person").
    pub fn category_name(&self) -> &'static str {
        match self {
            Feature::Number(_) => "Number",
            Feature::Person(_) => "Person",
            Feature::Tense(_) => "Tense",
            Feature::Aspect(_) => "Aspect",
            Feature::Voice(_) => "Voice",
            Feature::Mood(_) => "Mood",
            Feature::Case(_) => "Case",
            Feature::Gender(_) => "Gender",
            Feature::Definiteness(_) => "Definiteness",
            Feature::Animacy(_) => "Animacy",
            Feature::Transitivity(_) => "Transitivity",
            Feature::Finiteness(_) => "Finiteness",
        }
    }

    /// True when two features belong to the same category regardless of value.
    pub fn same_category(&self, other: &Feature) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }

    /// True when two features are of the same category **and** the same value.
    pub fn is_compatible(&self, other: &Feature) -> bool {
        self == other
    }
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Feature::Number(v) => write!(f, "Number={v:?}"),
            Feature::Person(v) => write!(f, "Person={v:?}"),
            Feature::Tense(v) => write!(f, "Tense={v:?}"),
            Feature::Aspect(v) => write!(f, "Aspect={v:?}"),
            Feature::Voice(v) => write!(f, "Voice={v:?}"),
            Feature::Mood(v) => write!(f, "Mood={v:?}"),
            Feature::Case(v) => write!(f, "Case={v:?}"),
            Feature::Gender(v) => write!(f, "Gender={v:?}"),
            Feature::Definiteness(v) => write!(f, "Definiteness={v:?}"),
            Feature::Animacy(v) => write!(f, "Animacy={v:?}"),
            Feature::Transitivity(v) => write!(f, "Transitivity={v:?}"),
            Feature::Finiteness(v) => write!(f, "Finiteness={v:?}"),
        }
    }
}

// ── FeatureConflict ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureConflict {
    pub feature_name: String,
    pub value1: String,
    pub value2: String,
    pub explanation: String,
}

impl FeatureConflict {
    pub fn new(
        feature_name: impl Into<String>,
        value1: impl Into<String>,
        value2: impl Into<String>,
        explanation: impl Into<String>,
    ) -> Self {
        Self {
            feature_name: feature_name.into(),
            value1: value1.into(),
            value2: value2.into(),
            explanation: explanation.into(),
        }
    }
}

impl fmt::Display for FeatureConflict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Conflict on {}: {} vs {} ({})",
            self.feature_name, self.value1, self.value2, self.explanation
        )
    }
}

// ── FeatureBundle ───────────────────────────────────────────────────────────

/// A named collection of features that supports unification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureBundle {
    features: HashMap<String, Feature>,
}

impl FeatureBundle {
    pub fn new() -> Self {
        Self {
            features: HashMap::new(),
        }
    }

    pub fn from_features(features: HashMap<String, Feature>) -> Self {
        Self { features }
    }

    pub fn get(&self, name: &str) -> Option<&Feature> {
        self.features.get(name)
    }

    pub fn set(&mut self, name: impl Into<String>, feature: Feature) {
        self.features.insert(name.into(), feature);
    }

    pub fn remove(&mut self, name: &str) -> Option<Feature> {
        self.features.remove(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.features.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.features.keys()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Feature)> {
        self.features.iter()
    }

    /// Attempt to unify this bundle with `other`.
    /// Succeeds if every feature present in both bundles has compatible values.
    pub fn unify_with(&self, other: &FeatureBundle) -> Result<FeatureBundle, Vec<FeatureConflict>> {
        unify_features(self, other)
    }

    /// Non-destructive compatibility check (no merge).
    pub fn is_compatible_with(&self, other: &FeatureBundle) -> bool {
        for (name, feat) in &self.features {
            if let Some(other_feat) = other.features.get(name) {
                if !feat.is_compatible(other_feat) {
                    return false;
                }
            }
        }
        true
    }

    /// Merge `other` into self, overwriting on conflict.
    pub fn merge(&mut self, other: &FeatureBundle) {
        for (name, feat) in &other.features {
            self.features.insert(name.clone(), *feat);
        }
    }

    /// Return feature names shared between two bundles.
    pub fn shared_keys(&self, other: &FeatureBundle) -> Vec<String> {
        self.features
            .keys()
            .filter(|k| other.features.contains_key(*k))
            .cloned()
            .collect()
    }
}

impl Default for FeatureBundle {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for FeatureBundle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut pairs: Vec<_> = self.features.iter().collect();
        pairs.sort_by_key(|(k, _)| (*k).clone());
        let parts: Vec<String> = pairs.iter().map(|(k, v)| format!("{k}={v}")).collect();
        write!(f, "[{}]", parts.join(", "))
    }
}

// ── FeatureStructure ────────────────────────────────────────────────────────

/// A categorised feature bundle – the basic unit of syntactic representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureStructure {
    /// Syntactic category: NP, VP, S, PP, AP, AdvP, CP, DP, IP, etc.
    pub category: String,
    pub features: FeatureBundle,
    /// Optional head word / lemma.
    pub head_index: Option<String>,
}

impl FeatureStructure {
    pub fn new(category: impl Into<String>) -> Self {
        Self {
            category: category.into(),
            features: FeatureBundle::new(),
            head_index: None,
        }
    }

    pub fn with_head(mut self, head_index: impl Into<String>) -> Self {
        self.head_index = Some(head_index.into());
        self
    }

    pub fn with_feature(mut self, name: impl Into<String>, feature: Feature) -> Self {
        self.features.set(name, feature);
        self
    }

    pub fn set_feature(&mut self, name: impl Into<String>, feature: Feature) {
        self.features.set(name, feature);
    }

    pub fn get_feature(&self, name: &str) -> Option<&Feature> {
        self.features.get(name)
    }

    pub fn is_np(&self) -> bool {
        self.category == "NP" || self.category == "DP"
    }

    pub fn is_vp(&self) -> bool {
        self.category == "VP"
    }

    pub fn is_clause(&self) -> bool {
        matches!(self.category.as_str(), "S" | "CP" | "IP" | "SBAR")
    }

    pub fn is_pp(&self) -> bool {
        self.category == "PP"
    }

    pub fn unify_with(&self, other: &FeatureStructure) -> Result<FeatureStructure, Vec<FeatureConflict>> {
        let merged = self.features.unify_with(&other.features)?;
        let merged_head = self.head_index.clone().or_else(|| other.head_index.clone());
        Ok(FeatureStructure {
            category: self.category.clone(),
            features: merged,
            head_index: merged_head,
        })
    }
}

impl fmt::Display for FeatureStructure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.category, self.features)?;
        if let Some(h) = &self.head_index {
            write!(f, " head={h}")?;
        }
        Ok(())
    }
}

// ── Free function: unify_features ───────────────────────────────────────────

/// Attempt unification of two `FeatureBundle`s.
///
/// For each feature name present in both bundles the values must be identical.
/// The result contains the union of all features when successful.
pub fn unify_features(
    a: &FeatureBundle,
    b: &FeatureBundle,
) -> Result<FeatureBundle, Vec<FeatureConflict>> {
    let mut conflicts: Vec<FeatureConflict> = Vec::new();
    let mut merged = a.features.clone();

    for (name, b_feat) in &b.features {
        if let Some(a_feat) = a.features.get(name) {
            if a_feat != b_feat {
                conflicts.push(FeatureConflict::new(
                    name,
                    format!("{a_feat:?}"),
                    format!("{b_feat:?}"),
                    format!(
                        "Feature '{name}' has incompatible values in the two bundles"
                    ),
                ));
            }
            // compatible – already in merged
        } else {
            merged.insert(name.clone(), *b_feat);
        }
    }

    if conflicts.is_empty() {
        Ok(FeatureBundle::from_features(merged))
    } else {
        Err(conflicts)
    }
}

// ── Helpers for constructing common bundles ──────────────────────────────────

/// Build an NP feature structure with common defaults.
pub fn np_features(number: NumberValue, person: PersonValue) -> FeatureStructure {
    FeatureStructure::new("NP")
        .with_feature("Number", Feature::Number(number))
        .with_feature("Person", Feature::Person(person))
}

/// Build a VP feature structure with common defaults.
pub fn vp_features(
    tense: TenseValue,
    number: NumberValue,
    person: PersonValue,
) -> FeatureStructure {
    FeatureStructure::new("VP")
        .with_feature("Tense", Feature::Tense(tense))
        .with_feature("Number", Feature::Number(number))
        .with_feature("Person", Feature::Person(person))
}

/// Build a finite clause feature structure.
pub fn clause_features(tense: TenseValue, mood: MoodValue) -> FeatureStructure {
    FeatureStructure::new("S")
        .with_feature("Tense", Feature::Tense(tense))
        .with_feature("Mood", Feature::Mood(mood))
        .with_feature("Finiteness", Feature::Finiteness(FinitenessValue::Finite))
}

// ── Convert shared_types enums to local feature values ──────────────────────

impl TenseValue {
    pub fn from_shared(t: &shared_types::Tense) -> Option<Self> {
        match t {
            shared_types::Tense::Past
            | shared_types::Tense::PastPerfect
            | shared_types::Tense::PastProgressive => Some(TenseValue::Past),
            shared_types::Tense::Present
            | shared_types::Tense::PresentPerfect
            | shared_types::Tense::PresentProgressive => Some(TenseValue::Present),
            shared_types::Tense::Future
            | shared_types::Tense::FuturePerfect
            | shared_types::Tense::FutureProgressive => Some(TenseValue::Future),
            shared_types::Tense::Unknown => None,
        }
    }
}

impl VoiceValue {
    pub fn from_shared(v: &shared_types::Voice) -> Option<Self> {
        match v {
            shared_types::Voice::Active => Some(VoiceValue::Active),
            shared_types::Voice::Passive => Some(VoiceValue::Passive),
            _ => None,
        }
    }
}

impl MoodValue {
    pub fn from_shared(m: &shared_types::Mood) -> Option<Self> {
        match m {
            shared_types::Mood::Indicative => Some(MoodValue::Indicative),
            shared_types::Mood::Subjunctive => Some(MoodValue::Subjunctive),
            shared_types::Mood::Imperative => Some(MoodValue::Imperative),
            shared_types::Mood::Interrogative => Some(MoodValue::Interrogative),
            shared_types::Mood::Unknown => None,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_category_name() {
        let f = Feature::Number(NumberValue::Singular);
        assert_eq!(f.category_name(), "Number");
        let f2 = Feature::Person(PersonValue::Third);
        assert_eq!(f2.category_name(), "Person");
    }

    #[test]
    fn test_feature_same_category() {
        let a = Feature::Number(NumberValue::Singular);
        let b = Feature::Number(NumberValue::Plural);
        assert!(a.same_category(&b));
        assert!(!a.same_category(&Feature::Person(PersonValue::First)));
    }

    #[test]
    fn test_feature_bundle_basic() {
        let mut fb = FeatureBundle::new();
        fb.set("Number", Feature::Number(NumberValue::Singular));
        fb.set("Person", Feature::Person(PersonValue::Third));
        assert_eq!(fb.len(), 2);
        assert!(fb.contains("Number"));
        assert!(!fb.contains("Tense"));
    }

    #[test]
    fn test_unify_compatible() {
        let mut a = FeatureBundle::new();
        a.set("Number", Feature::Number(NumberValue::Singular));
        let mut b = FeatureBundle::new();
        b.set("Person", Feature::Person(PersonValue::Third));
        let merged = a.unify_with(&b).unwrap();
        assert_eq!(merged.len(), 2);
        assert_eq!(
            merged.get("Number"),
            Some(&Feature::Number(NumberValue::Singular))
        );
        assert_eq!(
            merged.get("Person"),
            Some(&Feature::Person(PersonValue::Third))
        );
    }

    #[test]
    fn test_unify_same_value() {
        let mut a = FeatureBundle::new();
        a.set("Number", Feature::Number(NumberValue::Plural));
        let mut b = FeatureBundle::new();
        b.set("Number", Feature::Number(NumberValue::Plural));
        let merged = a.unify_with(&b).unwrap();
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn test_unify_conflict() {
        let mut a = FeatureBundle::new();
        a.set("Number", Feature::Number(NumberValue::Singular));
        let mut b = FeatureBundle::new();
        b.set("Number", Feature::Number(NumberValue::Plural));
        let err = a.unify_with(&b).unwrap_err();
        assert_eq!(err.len(), 1);
        assert_eq!(err[0].feature_name, "Number");
    }

    #[test]
    fn test_is_compatible_with() {
        let mut a = FeatureBundle::new();
        a.set("Person", Feature::Person(PersonValue::First));
        let mut b = FeatureBundle::new();
        b.set("Person", Feature::Person(PersonValue::First));
        b.set("Number", Feature::Number(NumberValue::Singular));
        assert!(a.is_compatible_with(&b));
    }

    #[test]
    fn test_feature_structure_unify() {
        let np = np_features(NumberValue::Singular, PersonValue::Third);
        let vp = FeatureStructure::new("NP")
            .with_feature("Number", Feature::Number(NumberValue::Singular))
            .with_feature("Person", Feature::Person(PersonValue::Third))
            .with_feature("Case", Feature::Case(CaseValue::Nominative));
        let merged = np.unify_with(&vp).unwrap();
        assert_eq!(merged.features.len(), 3);
    }

    #[test]
    fn test_feature_structure_conflict() {
        let np = np_features(NumberValue::Singular, PersonValue::Third);
        let vp = FeatureStructure::new("NP")
            .with_feature("Number", Feature::Number(NumberValue::Plural));
        assert!(np.unify_with(&vp).is_err());
    }

    #[test]
    fn test_merge_overwrites() {
        let mut a = FeatureBundle::new();
        a.set("Number", Feature::Number(NumberValue::Singular));
        let mut b = FeatureBundle::new();
        b.set("Number", Feature::Number(NumberValue::Plural));
        a.merge(&b);
        assert_eq!(
            a.get("Number"),
            Some(&Feature::Number(NumberValue::Plural))
        );
    }

    #[test]
    fn test_np_helper() {
        let np = np_features(NumberValue::Plural, PersonValue::First);
        assert!(np.is_np());
        assert!(!np.is_vp());
    }

    #[test]
    fn test_display_feature_bundle() {
        let mut fb = FeatureBundle::new();
        fb.set("Number", Feature::Number(NumberValue::Singular));
        let s = fb.to_string();
        assert!(s.contains("Number"));
    }
}
