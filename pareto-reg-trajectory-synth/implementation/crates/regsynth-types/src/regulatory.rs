use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use crate::obligation::{ArticleReference, RegulatoryDomain};
use crate::jurisdiction::JurisdictionId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryFramework {
    pub id: String,
    pub name: String,
    pub jurisdiction: JurisdictionId,
    pub framework_type: FrameworkType,
    pub version: String,
    pub effective_date: Option<String>,
    pub articles: Vec<FrameworkArticle>,
    pub annexes: Vec<AnnexReference>,
    pub domains: Vec<RegulatoryDomain>,
    pub url: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameworkType {
    BindingRegulation,
    VoluntaryFramework,
    InternationalStandard,
    ProposedLegislation,
    GuidancePrinciples,
}

impl fmt::Display for FrameworkType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrameworkType::BindingRegulation => write!(f, "Binding Regulation"),
            FrameworkType::VoluntaryFramework => write!(f, "Voluntary Framework"),
            FrameworkType::InternationalStandard => write!(f, "International Standard"),
            FrameworkType::ProposedLegislation => write!(f, "Proposed Legislation"),
            FrameworkType::GuidancePrinciples => write!(f, "Guidance/Principles"),
        }
    }
}

impl FrameworkType {
    pub fn is_binding(&self) -> bool { matches!(self, FrameworkType::BindingRegulation) }
    pub fn default_weight(&self) -> f64 {
        match self {
            FrameworkType::BindingRegulation => 1.0,
            FrameworkType::InternationalStandard => 0.8,
            FrameworkType::VoluntaryFramework => 0.5,
            FrameworkType::GuidancePrinciples => 0.3,
            FrameworkType::ProposedLegislation => 0.4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkArticle {
    pub article_ref: ArticleReference,
    pub obligation_count: usize,
    pub domain: RegulatoryDomain,
    pub summary: String,
    pub formalizability_estimate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnexReference {
    pub annex_id: String,
    pub title: String,
    pub description: String,
    pub related_articles: Vec<String>,
}

impl RegulatoryFramework {
    pub fn new(id: &str, name: &str, jurisdiction: JurisdictionId, ftype: FrameworkType) -> Self {
        RegulatoryFramework {
            id: id.to_string(), name: name.to_string(), jurisdiction, framework_type: ftype,
            version: "1.0".to_string(), effective_date: None,
            articles: Vec::new(), annexes: Vec::new(), domains: Vec::new(), url: None,
        }
    }

    pub fn article_count(&self) -> usize { self.articles.len() }
    pub fn total_obligations(&self) -> usize { self.articles.iter().map(|a| a.obligation_count).sum() }
    pub fn avg_formalizability(&self) -> f64 {
        if self.articles.is_empty() { return 0.0; }
        self.articles.iter().map(|a| a.formalizability_estimate).sum::<f64>() / self.articles.len() as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossJurisdictionalMapping {
    pub source_concept: ConceptId,
    pub target_concept: ConceptId,
    pub mapping_type: MappingType,
    pub confidence: f64,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConceptId {
    pub jurisdiction: JurisdictionId,
    pub domain: String,
    pub concept_name: String,
}

impl ConceptId {
    pub fn new(jurisdiction: JurisdictionId, domain: &str, name: &str) -> Self {
        ConceptId { jurisdiction, domain: domain.to_string(), concept_name: name.to_string() }
    }
}

impl fmt::Display for ConceptId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.jurisdiction, self.domain, self.concept_name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MappingType {
    Equivalent,
    Broader,
    Narrower,
    Related,
    Conflicting,
}

impl MappingType {
    pub fn confidence_factor(&self) -> f64 {
        match self {
            MappingType::Equivalent => 1.0,
            MappingType::Broader => 0.7,
            MappingType::Narrower => 0.7,
            MappingType::Related => 0.5,
            MappingType::Conflicting => 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryOntology {
    pub concepts: Vec<ConceptId>,
    pub mappings: Vec<CrossJurisdictionalMapping>,
    concept_index: HashMap<String, usize>,
}

impl RegulatoryOntology {
    pub fn new() -> Self {
        RegulatoryOntology { concepts: Vec::new(), mappings: Vec::new(), concept_index: HashMap::new() }
    }

    pub fn add_concept(&mut self, concept: ConceptId) {
        let key = concept.to_string();
        if !self.concept_index.contains_key(&key) {
            let idx = self.concepts.len();
            self.concept_index.insert(key, idx);
            self.concepts.push(concept);
        }
    }

    pub fn add_mapping(&mut self, mapping: CrossJurisdictionalMapping) {
        self.add_concept(mapping.source_concept.clone());
        self.add_concept(mapping.target_concept.clone());
        self.mappings.push(mapping);
    }

    pub fn find_equivalent(&self, concept: &ConceptId) -> Vec<&ConceptId> {
        self.mappings.iter()
            .filter(|m| m.source_concept == *concept && m.mapping_type == MappingType::Equivalent)
            .map(|m| &m.target_concept)
            .collect()
    }

    pub fn find_conflicts(&self, concept: &ConceptId) -> Vec<&CrossJurisdictionalMapping> {
        self.mappings.iter()
            .filter(|m| (m.source_concept == *concept || m.target_concept == *concept)
                && m.mapping_type == MappingType::Conflicting)
            .collect()
    }

    pub fn concepts_for_jurisdiction(&self, jid: &JurisdictionId) -> Vec<&ConceptId> {
        self.concepts.iter().filter(|c| c.jurisdiction == *jid).collect()
    }

    pub fn all_conflicts(&self) -> Vec<&CrossJurisdictionalMapping> {
        self.mappings.iter().filter(|m| m.mapping_type == MappingType::Conflicting).collect()
    }
}

impl Default for RegulatoryOntology {
    fn default() -> Self { Self::new() }
}

pub fn build_default_ontology() -> RegulatoryOntology {
    let mut ontology = RegulatoryOntology::new();
    let eu = JurisdictionId::new("EU");
    let us = JurisdictionId::new("US_NIST");
    let cn = JurisdictionId::new("CN");
    let iso_j = JurisdictionId::new("ISO");

    let risk_concepts = vec![
        (eu.clone(), "risk_classification", "high_risk"),
        (us.clone(), "risk_classification", "high"),
        (cn.clone(), "risk_classification", "critical"),
        (iso_j.clone(), "risk_classification", "significant_risk"),
    ];

    for (j, d, c) in &risk_concepts {
        ontology.add_concept(ConceptId::new(j.clone(), d, c));
    }

    ontology.add_mapping(CrossJurisdictionalMapping {
        source_concept: ConceptId::new(eu.clone(), "risk_classification", "high_risk"),
        target_concept: ConceptId::new(us.clone(), "risk_classification", "high"),
        mapping_type: MappingType::Related,
        confidence: 0.75,
        notes: "EU high-risk is broader than NIST high".to_string(),
    });

    ontology.add_mapping(CrossJurisdictionalMapping {
        source_concept: ConceptId::new(eu.clone(), "transparency", "logging_requirement"),
        target_concept: ConceptId::new(eu.clone(), "data_governance", "data_minimization"),
        mapping_type: MappingType::Conflicting,
        confidence: 0.8,
        notes: "EU AI Act logging vs GDPR data minimization".to_string(),
    });

    ontology.add_mapping(CrossJurisdictionalMapping {
        source_concept: ConceptId::new(cn.clone(), "transparency", "algorithm_disclosure"),
        target_concept: ConceptId::new(us.clone(), "ip_protection", "trade_secret"),
        mapping_type: MappingType::Conflicting,
        confidence: 0.9,
        notes: "China disclosure requirements vs US trade secret protection".to_string(),
    });

    ontology
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_framework_creation() {
        let fw = RegulatoryFramework::new("eu_ai_act", "EU AI Act",
            JurisdictionId::new("EU"), FrameworkType::BindingRegulation);
        assert!(fw.framework_type.is_binding());
        assert_eq!(fw.article_count(), 0);
    }

    #[test]
    fn test_ontology() {
        let ontology = build_default_ontology();
        let conflicts = ontology.all_conflicts();
        assert!(conflicts.len() >= 2);
    }

    #[test]
    fn test_mapping_type() {
        assert_eq!(MappingType::Equivalent.confidence_factor(), 1.0);
        assert_eq!(MappingType::Conflicting.confidence_factor(), 0.0);
    }
}
