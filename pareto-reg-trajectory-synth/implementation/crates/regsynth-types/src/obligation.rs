use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use crate::jurisdiction::JurisdictionId;
use crate::temporal::TemporalInterval;
use crate::formalizability::FormalGrade;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObligationType {
    Obligation,
    Permission,
    Prohibition,
}

impl fmt::Display for ObligationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObligationType::Obligation => write!(f, "OBL"),
            ObligationType::Permission => write!(f, "PERM"),
            ObligationType::Prohibition => write!(f, "PROH"),
        }
    }
}

impl ObligationType {
    pub fn from_str_tag(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "OBL" | "OBLIGATION" => Some(ObligationType::Obligation),
            "PERM" | "PERMISSION" => Some(ObligationType::Permission),
            "PROH" | "PROHIBITION" => Some(ObligationType::Prohibition),
            _ => None,
        }
    }

    pub fn is_mandatory(&self) -> bool {
        matches!(self, ObligationType::Obligation | ObligationType::Prohibition)
    }

    pub fn is_permissive(&self) -> bool {
        matches!(self, ObligationType::Permission)
    }

    pub fn negation(&self) -> Self {
        match self {
            ObligationType::Obligation => ObligationType::Prohibition,
            ObligationType::Prohibition => ObligationType::Obligation,
            ObligationType::Permission => ObligationType::Permission,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObligationId(pub String);

impl ObligationId {
    pub fn new(id: impl Into<String>) -> Self {
        ObligationId(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn with_jurisdiction_prefix(jurisdiction: &JurisdictionId, local_id: &str) -> Self {
        ObligationId(format!("{}::{}", jurisdiction.as_str(), local_id))
    }
}

impl fmt::Display for ObligationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskLevel {
    Unacceptable,
    High,
    Limited,
    Minimal,
    Unknown,
}

impl RiskLevel {
    pub fn severity_score(&self) -> f64 {
        match self {
            RiskLevel::Unacceptable => 1.0,
            RiskLevel::High => 0.75,
            RiskLevel::Limited => 0.5,
            RiskLevel::Minimal => 0.25,
            RiskLevel::Unknown => 0.5,
        }
    }

    pub fn from_score(score: f64) -> Self {
        if score >= 0.9 { RiskLevel::Unacceptable }
        else if score >= 0.65 { RiskLevel::High }
        else if score >= 0.4 { RiskLevel::Limited }
        else { RiskLevel::Minimal }
    }

    pub fn max(self, other: Self) -> Self {
        if self.severity_score() >= other.severity_score() { self } else { other }
    }
}

impl fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RiskLevel::Unacceptable => write!(f, "Unacceptable"),
            RiskLevel::High => write!(f, "High"),
            RiskLevel::Limited => write!(f, "Limited"),
            RiskLevel::Minimal => write!(f, "Minimal"),
            RiskLevel::Unknown => write!(f, "Unknown"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegulatoryDomain {
    DataGovernance,
    Transparency,
    RiskClassification,
    HumanOversight,
    Documentation,
    PostMarketSurveillance,
    CrossBorderDataTransfer,
    AlgorithmicAccountability,
    BiasAndFairness,
    SecurityAndRobustness,
    IntellectualProperty,
    ConsentAndNotice,
    General,
}

impl fmt::Display for RegulatoryDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegulatoryDomain::DataGovernance => write!(f, "Data Governance"),
            RegulatoryDomain::Transparency => write!(f, "Transparency"),
            RegulatoryDomain::RiskClassification => write!(f, "Risk Classification"),
            RegulatoryDomain::HumanOversight => write!(f, "Human Oversight"),
            RegulatoryDomain::Documentation => write!(f, "Documentation"),
            RegulatoryDomain::PostMarketSurveillance => write!(f, "Post-Market Surveillance"),
            RegulatoryDomain::CrossBorderDataTransfer => write!(f, "Cross-Border Data Transfer"),
            RegulatoryDomain::AlgorithmicAccountability => write!(f, "Algorithmic Accountability"),
            RegulatoryDomain::BiasAndFairness => write!(f, "Bias & Fairness"),
            RegulatoryDomain::SecurityAndRobustness => write!(f, "Security & Robustness"),
            RegulatoryDomain::IntellectualProperty => write!(f, "Intellectual Property"),
            RegulatoryDomain::ConsentAndNotice => write!(f, "Consent & Notice"),
            RegulatoryDomain::General => write!(f, "General"),
        }
    }
}

impl RegulatoryDomain {
    pub fn all() -> Vec<Self> {
        vec![
            Self::DataGovernance, Self::Transparency, Self::RiskClassification,
            Self::HumanOversight, Self::Documentation, Self::PostMarketSurveillance,
            Self::CrossBorderDataTransfer, Self::AlgorithmicAccountability,
            Self::BiasAndFairness, Self::SecurityAndRobustness,
            Self::IntellectualProperty, Self::ConsentAndNotice, Self::General,
        ]
    }

    pub fn is_technical(&self) -> bool {
        matches!(self, Self::DataGovernance | Self::SecurityAndRobustness | Self::BiasAndFairness)
    }

    pub fn is_procedural(&self) -> bool {
        matches!(self, Self::Documentation | Self::HumanOversight | Self::PostMarketSurveillance)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArticleReference {
    pub framework: String,
    pub article_number: String,
    pub paragraph: Option<String>,
    pub subparagraph: Option<String>,
    pub annex: Option<String>,
    pub title: String,
    pub url: Option<String>,
}

impl ArticleReference {
    pub fn new(framework: &str, article: &str, title: &str) -> Self {
        ArticleReference {
            framework: framework.to_string(),
            article_number: article.to_string(),
            paragraph: None,
            subparagraph: None,
            annex: None,
            title: title.to_string(),
            url: None,
        }
    }

    pub fn with_paragraph(mut self, para: &str) -> Self {
        self.paragraph = Some(para.to_string());
        self
    }

    pub fn with_subparagraph(mut self, sub: &str) -> Self {
        self.subparagraph = Some(sub.to_string());
        self
    }

    pub fn with_annex(mut self, annex: &str) -> Self {
        self.annex = Some(annex.to_string());
        self
    }

    pub fn full_reference(&self) -> String {
        let mut ref_str = format!("{} Art. {}", self.framework, self.article_number);
        if let Some(ref para) = self.paragraph {
            ref_str.push_str(&format!("({})", para));
        }
        if let Some(ref sub) = self.subparagraph {
            ref_str.push_str(&format!(".{}", sub));
        }
        if let Some(ref annex) = self.annex {
            ref_str.push_str(&format!(", Annex {}", annex));
        }
        ref_str
    }
}

impl fmt::Display for ArticleReference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.full_reference())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObligationCondition {
    pub condition_type: ConditionType,
    pub description: String,
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConditionType {
    RiskLevelIs,
    SystemTypeIs,
    DeploymentContextIs,
    DataVolumeExceeds,
    UserCountExceeds,
    CrossBorderTransfer,
    ProcessesSensitiveData,
    AutomatedDecisionMaking,
    Custom,
}

impl ObligationCondition {
    pub fn risk_level(level: RiskLevel) -> Self {
        let mut params = HashMap::new();
        params.insert("level".to_string(), level.to_string());
        ObligationCondition {
            condition_type: ConditionType::RiskLevelIs,
            description: format!("System is classified as {} risk", level),
            parameters: params,
        }
    }

    pub fn cross_border() -> Self {
        ObligationCondition {
            condition_type: ConditionType::CrossBorderTransfer,
            description: "System involves cross-border data transfer".to_string(),
            parameters: HashMap::new(),
        }
    }

    pub fn evaluate(&self, context: &HashMap<String, String>) -> bool {
        match self.condition_type {
            ConditionType::RiskLevelIs => {
                self.parameters.get("level")
                    .and_then(|req| context.get("risk_level").map(|act| req == act))
                    .unwrap_or(false)
            }
            ConditionType::CrossBorderTransfer => {
                context.get("cross_border").map(|v| v == "true").unwrap_or(false)
            }
            ConditionType::DataVolumeExceeds => {
                let threshold = self.parameters.get("threshold").and_then(|t| t.parse::<u64>().ok());
                let actual = context.get("data_volume").and_then(|v| v.parse::<u64>().ok());
                match (threshold, actual) {
                    (Some(t), Some(a)) => a > t,
                    _ => false,
                }
            }
            ConditionType::UserCountExceeds => {
                let threshold = self.parameters.get("threshold").and_then(|t| t.parse::<u64>().ok());
                let actual = context.get("user_count").and_then(|v| v.parse::<u64>().ok());
                match (threshold, actual) {
                    (Some(t), Some(a)) => a > t,
                    _ => false,
                }
            }
            _ => true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Exemption {
    pub exemption_id: String,
    pub description: String,
    pub conditions: Vec<ObligationCondition>,
    pub scope: ExemptionScope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExemptionScope {
    Full,
    Partial,
    Temporal,
    Conditional,
}

impl Exemption {
    pub fn new(id: &str, description: &str, scope: ExemptionScope) -> Self {
        Exemption {
            exemption_id: id.to_string(),
            description: description.to_string(),
            conditions: Vec::new(),
            scope,
        }
    }

    pub fn applies(&self, context: &HashMap<String, String>) -> bool {
        self.conditions.iter().all(|c| c.evaluate(context))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Obligation {
    pub id: ObligationId,
    pub obligation_type: ObligationType,
    pub jurisdiction: JurisdictionId,
    pub article_ref: ArticleReference,
    pub temporal_interval: TemporalInterval,
    pub risk_level: RiskLevel,
    pub domain: RegulatoryDomain,
    pub formalizability: FormalGrade,
    pub description: String,
    pub conditions: Vec<ObligationCondition>,
    pub exemptions: Vec<Exemption>,
    pub penalty_amount: Option<f64>,
    pub penalty_description: Option<String>,
    pub cross_references: Vec<ObligationId>,
    pub tags: HashSet<String>,
}

impl Obligation {
    pub fn new(
        id: ObligationId,
        obligation_type: ObligationType,
        jurisdiction: JurisdictionId,
        article_ref: ArticleReference,
        description: &str,
    ) -> Self {
        Obligation {
            id, obligation_type, jurisdiction, article_ref,
            temporal_interval: TemporalInterval::always(),
            risk_level: RiskLevel::Unknown,
            domain: RegulatoryDomain::General,
            formalizability: FormalGrade::Full,
            description: description.to_string(),
            conditions: Vec::new(),
            exemptions: Vec::new(),
            penalty_amount: None,
            penalty_description: None,
            cross_references: Vec::new(),
            tags: HashSet::new(),
        }
    }

    pub fn with_risk_level(mut self, level: RiskLevel) -> Self {
        self.risk_level = level;
        self
    }

    pub fn with_domain(mut self, domain: RegulatoryDomain) -> Self {
        self.domain = domain;
        self
    }

    pub fn with_formalizability(mut self, grade: FormalGrade) -> Self {
        self.formalizability = grade;
        self
    }

    pub fn with_temporal(mut self, interval: TemporalInterval) -> Self {
        self.temporal_interval = interval;
        self
    }

    pub fn with_penalty(mut self, amount: f64, description: &str) -> Self {
        self.penalty_amount = Some(amount);
        self.penalty_description = Some(description.to_string());
        self
    }

    pub fn add_condition(&mut self, condition: ObligationCondition) {
        self.conditions.push(condition);
    }

    pub fn add_exemption(&mut self, exemption: Exemption) {
        self.exemptions.push(exemption);
    }

    pub fn add_cross_reference(&mut self, ref_id: ObligationId) {
        self.cross_references.push(ref_id);
    }

    pub fn is_active_at(&self, timestamp: i64) -> bool {
        self.temporal_interval.contains(timestamp)
    }

    pub fn is_hard_constraint(&self) -> bool {
        self.obligation_type.is_mandatory() && self.formalizability.is_formalizable()
    }

    pub fn effective_weight(&self) -> f64 {
        let base = if self.obligation_type.is_mandatory() { 1.0 } else { 0.5 };
        let risk = self.risk_level.severity_score();
        let formal = self.formalizability.confidence();
        let penalty = self.penalty_amount.map(|p| (p / 1_000_000.0).min(10.0)).unwrap_or(1.0);
        base * risk * formal * penalty
    }

    pub fn has_exemption_for(&self, context: &HashMap<String, String>) -> bool {
        self.exemptions.iter().any(|e| e.applies(context))
    }

    pub fn conditions_met(&self, context: &HashMap<String, String>) -> bool {
        self.conditions.iter().all(|c| c.evaluate(context))
    }
}

impl fmt::Display for Obligation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} ({}) - {} [{}]",
            self.obligation_type, self.id, self.jurisdiction,
            self.description, self.formalizability)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositionOp {
    Conjunction,
    Disjunction,
    JurisdictionalOverride,
    Exception,
}

impl fmt::Display for CompositionOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompositionOp::Conjunction => write!(f, "⊗"),
            CompositionOp::Disjunction => write!(f, "⊕"),
            CompositionOp::JurisdictionalOverride => write!(f, "▷"),
            CompositionOp::Exception => write!(f, "⊘"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComposedObligation {
    Leaf(Obligation),
    Compose {
        op: CompositionOp,
        left: Box<ComposedObligation>,
        right: Box<ComposedObligation>,
    },
}

impl ComposedObligation {
    pub fn leaf(obl: Obligation) -> Self { ComposedObligation::Leaf(obl) }

    pub fn conjunction(l: ComposedObligation, r: ComposedObligation) -> Self {
        ComposedObligation::Compose { op: CompositionOp::Conjunction, left: Box::new(l), right: Box::new(r) }
    }

    pub fn disjunction(l: ComposedObligation, r: ComposedObligation) -> Self {
        ComposedObligation::Compose { op: CompositionOp::Disjunction, left: Box::new(l), right: Box::new(r) }
    }

    pub fn jurisdictional_override(priority: ComposedObligation, fallback: ComposedObligation) -> Self {
        ComposedObligation::Compose { op: CompositionOp::JurisdictionalOverride, left: Box::new(priority), right: Box::new(fallback) }
    }

    pub fn exception(base: ComposedObligation, exempt: ComposedObligation) -> Self {
        ComposedObligation::Compose { op: CompositionOp::Exception, left: Box::new(base), right: Box::new(exempt) }
    }

    pub fn leaf_obligations(&self) -> Vec<&Obligation> {
        match self {
            ComposedObligation::Leaf(o) => vec![o],
            ComposedObligation::Compose { left, right, .. } => {
                let mut r = left.leaf_obligations();
                r.extend(right.leaf_obligations());
                r
            }
        }
    }

    pub fn jurisdictions(&self) -> HashSet<JurisdictionId> {
        self.leaf_obligations().iter().map(|o| o.jurisdiction.clone()).collect()
    }

    pub fn combined_formalizability(&self) -> FormalGrade {
        match self {
            ComposedObligation::Leaf(o) => o.formalizability.clone(),
            ComposedObligation::Compose { op, left, right } => {
                let lg = left.combined_formalizability();
                let rg = right.combined_formalizability();
                match op {
                    CompositionOp::Conjunction | CompositionOp::JurisdictionalOverride | CompositionOp::Exception
                        => lg.compose_conjunction(&rg),
                    CompositionOp::Disjunction => lg.compose_disjunction(&rg),
                }
            }
        }
    }

    pub fn depth(&self) -> usize {
        match self {
            ComposedObligation::Leaf(_) => 0,
            ComposedObligation::Compose { left, right, .. } => 1 + left.depth().max(right.depth()),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            ComposedObligation::Leaf(_) => 1,
            ComposedObligation::Compose { left, right, .. } => 1 + left.size() + right.size(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObligationSet {
    obligations: Vec<Obligation>,
    index_by_id: HashMap<String, usize>,
    index_by_jurisdiction: HashMap<String, Vec<usize>>,
    index_by_domain: HashMap<String, Vec<usize>>,
}

impl ObligationSet {
    pub fn new() -> Self {
        ObligationSet {
            obligations: Vec::new(),
            index_by_id: HashMap::new(),
            index_by_jurisdiction: HashMap::new(),
            index_by_domain: HashMap::new(),
        }
    }

    pub fn add(&mut self, obligation: Obligation) {
        let idx = self.obligations.len();
        self.index_by_id.insert(obligation.id.0.clone(), idx);
        self.index_by_jurisdiction.entry(obligation.jurisdiction.as_str().to_string()).or_default().push(idx);
        self.index_by_domain.entry(obligation.domain.to_string()).or_default().push(idx);
        self.obligations.push(obligation);
    }

    pub fn get(&self, id: &ObligationId) -> Option<&Obligation> {
        self.index_by_id.get(&id.0).map(|&i| &self.obligations[i])
    }

    pub fn by_jurisdiction(&self, jid: &JurisdictionId) -> Vec<&Obligation> {
        self.index_by_jurisdiction.get(jid.as_str())
            .map(|is| is.iter().map(|&i| &self.obligations[i]).collect())
            .unwrap_or_default()
    }

    pub fn by_domain(&self, domain: &RegulatoryDomain) -> Vec<&Obligation> {
        self.index_by_domain.get(&domain.to_string())
            .map(|is| is.iter().map(|&i| &self.obligations[i]).collect())
            .unwrap_or_default()
    }

    pub fn active_at(&self, ts: i64) -> Vec<&Obligation> {
        self.obligations.iter().filter(|o| o.is_active_at(ts)).collect()
    }

    pub fn hard_constraints(&self) -> Vec<&Obligation> {
        self.obligations.iter().filter(|o| o.is_hard_constraint()).collect()
    }

    pub fn len(&self) -> usize { self.obligations.len() }
    pub fn is_empty(&self) -> bool { self.obligations.is_empty() }
    pub fn iter(&self) -> impl Iterator<Item = &Obligation> { self.obligations.iter() }

    pub fn jurisdictions(&self) -> HashSet<JurisdictionId> {
        self.obligations.iter().map(|o| o.jurisdiction.clone()).collect()
    }

    pub fn domains(&self) -> HashSet<String> {
        self.obligations.iter().map(|o| o.domain.to_string()).collect()
    }

    pub fn statistics(&self) -> ObligationSetStats {
        let total = self.obligations.len();
        let hard = self.hard_constraints().len();
        let mut by_type: HashMap<String, usize> = HashMap::new();
        for obl in &self.obligations {
            *by_type.entry(obl.obligation_type.to_string()).or_insert(0) += 1;
        }
        let avg_f = if total > 0 {
            self.obligations.iter().map(|o| o.formalizability.confidence()).sum::<f64>() / total as f64
        } else { 0.0 };
        ObligationSetStats {
            total, hard_constraints: hard, soft_constraints: total - hard,
            by_type, jurisdiction_count: self.jurisdictions().len(),
            domain_count: self.domains().len(), avg_formalizability: avg_f,
        }
    }
}

impl Default for ObligationSet {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObligationSetStats {
    pub total: usize,
    pub hard_constraints: usize,
    pub soft_constraints: usize,
    pub by_type: HashMap<String, usize>,
    pub jurisdiction_count: usize,
    pub domain_count: usize,
    pub avg_formalizability: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_obl(id: &str, t: ObligationType) -> Obligation {
        Obligation::new(ObligationId::new(id), t, JurisdictionId::new("EU"),
            ArticleReference::new("EU AI Act", "6", "Risk Classification"), "Test")
    }

    #[test]
    fn test_obligation_type() {
        assert_eq!(ObligationType::Obligation.to_string(), "OBL");
        assert!(ObligationType::Obligation.is_mandatory());
        assert!(ObligationType::Permission.is_permissive());
    }

    #[test]
    fn test_obligation_set() {
        let mut set = ObligationSet::new();
        set.add(make_obl("o1", ObligationType::Obligation));
        set.add(make_obl("o2", ObligationType::Permission));
        assert_eq!(set.len(), 2);
        assert!(set.get(&ObligationId::new("o1")).is_some());
    }

    #[test]
    fn test_composition() {
        let c = ComposedObligation::conjunction(
            ComposedObligation::leaf(make_obl("a", ObligationType::Obligation)),
            ComposedObligation::leaf(make_obl("b", ObligationType::Prohibition)),
        );
        assert_eq!(c.size(), 3);
        assert_eq!(c.depth(), 1);
    }

    #[test]
    fn test_article_ref() {
        let r = ArticleReference::new("EU AI Act", "6", "Risk").with_paragraph("1a");
        assert_eq!(r.full_reference(), "EU AI Act Art. 6(1a)");
    }
}
