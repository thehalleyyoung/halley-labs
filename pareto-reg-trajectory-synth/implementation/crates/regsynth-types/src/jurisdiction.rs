use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use serde::{Deserialize, Serialize};

/// Unique identifier for a jurisdiction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct JurisdictionId(pub String);

impl JurisdictionId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for JurisdictionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for JurisdictionId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// The binding nature of a jurisdiction's regulations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JurisdictionType {
    /// Legally binding regulation (e.g., EU AI Act)
    Binding,
    /// Voluntary framework (e.g., NIST AI RMF)
    Voluntary,
    /// Industry standard (e.g., ISO 42001)
    Standard,
    /// Proposed/draft regulation not yet in force
    Proposed,
}

impl JurisdictionType {
    /// Binding strength on [0,1]. Binding=1.0, Voluntary=0.3, etc.
    pub fn binding_strength(&self) -> f64 {
        match self {
            Self::Binding => 1.0,
            Self::Voluntary => 0.3,
            Self::Standard => 0.6,
            Self::Proposed => 0.1,
        }
    }

    /// Whether non-compliance carries legal penalties.
    pub fn has_legal_penalty(&self) -> bool {
        matches!(self, Self::Binding)
    }
}

impl fmt::Display for JurisdictionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binding => write!(f, "Binding"),
            Self::Voluntary => write!(f, "Voluntary"),
            Self::Standard => write!(f, "Standard"),
            Self::Proposed => write!(f, "Proposed"),
        }
    }
}

/// Geographic region categorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum GeographicRegion {
    EU,
    NorthAmerica,
    UnitedKingdom,
    AsiaPacific,
    LatinAmerica,
    Africa,
    MiddleEast,
    Global,
}

impl GeographicRegion {
    pub fn all() -> &'static [GeographicRegion] {
        &[
            Self::EU,
            Self::NorthAmerica,
            Self::UnitedKingdom,
            Self::AsiaPacific,
            Self::LatinAmerica,
            Self::Africa,
            Self::MiddleEast,
            Self::Global,
        ]
    }

    /// Whether this region's regulations may have extra-territorial reach.
    pub fn has_extraterritorial_reach(&self) -> bool {
        matches!(self, Self::EU | Self::Global)
    }
}

impl fmt::Display for GeographicRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::EU => "EU",
            Self::NorthAmerica => "North America",
            Self::UnitedKingdom => "United Kingdom",
            Self::AsiaPacific => "Asia-Pacific",
            Self::LatinAmerica => "Latin America",
            Self::Africa => "Africa",
            Self::MiddleEast => "Middle East",
            Self::Global => "Global",
        };
        write!(f, "{}", s)
    }
}

/// How a jurisdiction enforces its regulations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnforcementModel {
    /// Pre-market conformity assessment
    PreMarket,
    /// Post-market surveillance
    PostMarket,
    /// Sector-specific regulator
    SectorSpecific,
    /// Self-certification / voluntary compliance
    SelfCertification,
    /// Hybrid: combination of pre- and post-market
    Hybrid,
}

impl fmt::Display for EnforcementModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::PreMarket => "Pre-Market Assessment",
            Self::PostMarket => "Post-Market Surveillance",
            Self::SectorSpecific => "Sector-Specific",
            Self::SelfCertification => "Self-Certification",
            Self::Hybrid => "Hybrid",
        };
        write!(f, "{}", s)
    }
}

/// Priority weight for a jurisdiction in multi-jurisdictional reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PriorityWeight(f64);

impl PriorityWeight {
    /// Create a weight in [0, 1]. Clamps out-of-range values.
    pub fn new(w: f64) -> Self {
        Self(w.clamp(0.0, 1.0))
    }

    pub fn value(&self) -> f64 {
        self.0
    }

    /// Combine two weights by taking the max (join in the priority lattice).
    pub fn join(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    /// Combine two weights by taking the min (meet in the priority lattice).
    pub fn meet(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }
}

impl Default for PriorityWeight {
    fn default() -> Self {
        Self(0.5)
    }
}

impl fmt::Display for PriorityWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}", self.0)
    }
}

/// Full information about a single jurisdiction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JurisdictionInfo {
    pub id: JurisdictionId,
    pub name: String,
    pub short_name: String,
    pub jurisdiction_type: JurisdictionType,
    pub region: GeographicRegion,
    pub enforcement: EnforcementModel,
    pub priority: PriorityWeight,
    /// Parent jurisdiction (e.g., EU member state -> EU)
    pub parent: Option<JurisdictionId>,
    /// Known mutual-recognition agreements
    pub mutual_recognition: Vec<JurisdictionId>,
    /// Free-text description
    pub description: String,
    /// Maximum fine as fraction of global turnover (0.0 if none)
    pub max_fine_fraction: f64,
}

impl JurisdictionInfo {
    pub fn effective_weight(&self) -> f64 {
        self.priority.value() * self.jurisdiction_type.binding_strength()
    }

    pub fn is_binding(&self) -> bool {
        self.jurisdiction_type == JurisdictionType::Binding
    }

    pub fn recognizes(&self, other: &JurisdictionId) -> bool {
        self.mutual_recognition.contains(other)
    }
}

impl fmt::Display for JurisdictionInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}] ({}, {}, priority={})",
            self.name, self.id, self.jurisdiction_type, self.region, self.priority
        )
    }
}

/// Builder for JurisdictionInfo.
#[derive(Debug, Default)]
pub struct JurisdictionInfoBuilder {
    id: Option<JurisdictionId>,
    name: Option<String>,
    short_name: Option<String>,
    jurisdiction_type: Option<JurisdictionType>,
    region: Option<GeographicRegion>,
    enforcement: Option<EnforcementModel>,
    priority: PriorityWeight,
    parent: Option<JurisdictionId>,
    mutual_recognition: Vec<JurisdictionId>,
    description: String,
    max_fine_fraction: f64,
}

impl JurisdictionInfoBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(JurisdictionId::new(id));
        self
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn short_name(mut self, s: impl Into<String>) -> Self {
        self.short_name = Some(s.into());
        self
    }

    pub fn jurisdiction_type(mut self, t: JurisdictionType) -> Self {
        self.jurisdiction_type = Some(t);
        self
    }

    pub fn region(mut self, r: GeographicRegion) -> Self {
        self.region = Some(r);
        self
    }

    pub fn enforcement(mut self, e: EnforcementModel) -> Self {
        self.enforcement = Some(e);
        self
    }

    pub fn priority(mut self, w: f64) -> Self {
        self.priority = PriorityWeight::new(w);
        self
    }

    pub fn parent(mut self, p: impl Into<String>) -> Self {
        self.parent = Some(JurisdictionId::new(p));
        self
    }

    pub fn mutual_recognition(mut self, ids: Vec<JurisdictionId>) -> Self {
        self.mutual_recognition = ids;
        self
    }

    pub fn description(mut self, d: impl Into<String>) -> Self {
        self.description = d.into();
        self
    }

    pub fn max_fine_fraction(mut self, f: f64) -> Self {
        self.max_fine_fraction = f;
        self
    }

    pub fn build(self) -> Result<JurisdictionInfo, String> {
        let id = self.id.ok_or("id is required")?;
        let name = self.name.ok_or("name is required")?;
        let short_name = self.short_name.unwrap_or_else(|| name.clone());
        let jurisdiction_type = self.jurisdiction_type.ok_or("jurisdiction_type is required")?;
        let region = self.region.ok_or("region is required")?;
        let enforcement = self.enforcement.ok_or("enforcement is required")?;
        Ok(JurisdictionInfo {
            id,
            name,
            short_name,
            jurisdiction_type,
            region,
            enforcement,
            priority: self.priority,
            parent: self.parent,
            mutual_recognition: self.mutual_recognition,
            description: self.description,
            max_fine_fraction: self.max_fine_fraction,
        })
    }
}

/// Registry holding all known jurisdictions with lookup capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JurisdictionRegistry {
    jurisdictions: BTreeMap<JurisdictionId, JurisdictionInfo>,
}

impl JurisdictionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, info: JurisdictionInfo) {
        self.jurisdictions.insert(info.id.clone(), info);
    }

    pub fn get(&self, id: &JurisdictionId) -> Option<&JurisdictionInfo> {
        self.jurisdictions.get(id)
    }

    pub fn contains(&self, id: &JurisdictionId) -> bool {
        self.jurisdictions.contains_key(id)
    }

    pub fn all_ids(&self) -> Vec<&JurisdictionId> {
        self.jurisdictions.keys().collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&JurisdictionId, &JurisdictionInfo)> {
        self.jurisdictions.iter()
    }

    pub fn len(&self) -> usize {
        self.jurisdictions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.jurisdictions.is_empty()
    }

    pub fn by_region(&self, region: GeographicRegion) -> Vec<&JurisdictionInfo> {
        self.jurisdictions
            .values()
            .filter(|j| j.region == region)
            .collect()
    }

    pub fn by_type(&self, jtype: JurisdictionType) -> Vec<&JurisdictionInfo> {
        self.jurisdictions
            .values()
            .filter(|j| j.jurisdiction_type == jtype)
            .collect()
    }

    pub fn binding_jurisdictions(&self) -> Vec<&JurisdictionInfo> {
        self.by_type(JurisdictionType::Binding)
    }

    /// Find children of a parent jurisdiction.
    pub fn children_of(&self, parent_id: &JurisdictionId) -> Vec<&JurisdictionInfo> {
        self.jurisdictions
            .values()
            .filter(|j| j.parent.as_ref() == Some(parent_id))
            .collect()
    }

    /// Get the ancestor chain from a jurisdiction to its root.
    pub fn ancestors(&self, id: &JurisdictionId) -> Vec<JurisdictionId> {
        let mut result = Vec::new();
        let mut current = id.clone();
        let mut visited = BTreeSet::new();
        while let Some(info) = self.jurisdictions.get(&current) {
            if let Some(ref parent) = info.parent {
                if visited.contains(parent) {
                    break; // cycle guard
                }
                visited.insert(parent.clone());
                result.push(parent.clone());
                current = parent.clone();
            } else {
                break;
            }
        }
        result
    }

    /// Validate internal consistency: all parent refs resolve, no cycles.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        for (id, info) in &self.jurisdictions {
            if let Some(ref parent) = info.parent {
                if !self.jurisdictions.contains_key(parent) {
                    errors.push(format!(
                        "Jurisdiction {} references unknown parent {}",
                        id, parent
                    ));
                }
            }
            for mr in &info.mutual_recognition {
                if !self.jurisdictions.contains_key(mr) {
                    errors.push(format!(
                        "Jurisdiction {} references unknown mutual-recognition partner {}",
                        id, mr
                    ));
                }
            }
            // Check for self-parent cycle
            if info.parent.as_ref() == Some(id) {
                errors.push(format!("Jurisdiction {} is its own parent", id));
            }
        }
        errors
    }
}

/// Operations on the jurisdiction lattice.
///
/// The lattice is ordered by specificity: a child jurisdiction is "below"
/// its parent. join = least upper bound, meet = greatest lower bound.
#[derive(Debug)]
pub struct JurisdictionLattice<'a> {
    registry: &'a JurisdictionRegistry,
}

impl<'a> JurisdictionLattice<'a> {
    pub fn new(registry: &'a JurisdictionRegistry) -> Self {
        Self { registry }
    }

    /// Join: least upper bound of two jurisdictions.
    /// Returns the closest common ancestor, or None if they share no ancestor.
    pub fn join(
        &self,
        a: &JurisdictionId,
        b: &JurisdictionId,
    ) -> Option<JurisdictionId> {
        if a == b {
            return Some(a.clone());
        }
        let ancestors_a = self.ancestor_set(a);
        let ancestors_b = self.ancestor_set(b);
        // find the deepest common ancestor
        let common: Vec<&JurisdictionId> = ancestors_a.intersection(&ancestors_b).collect();
        if common.is_empty() {
            return None;
        }
        // pick the one with the longest ancestor chain (deepest)
        common
            .into_iter()
            .max_by_key(|id| self.depth(id))
            .cloned()
    }

    /// Meet: greatest lower bound. For jurisdictions, this is only defined
    /// when one is an ancestor of the other; returns the more specific one.
    pub fn meet(
        &self,
        a: &JurisdictionId,
        b: &JurisdictionId,
    ) -> Option<JurisdictionId> {
        if a == b {
            return Some(a.clone());
        }
        let ancestors_a = self.ancestor_set(a);
        let ancestors_b = self.ancestor_set(b);
        if ancestors_a.contains(b) {
            Some(a.clone())
        } else if ancestors_b.contains(a) {
            Some(b.clone())
        } else {
            None
        }
    }

    /// Compare two jurisdictions: returns ordering if one is ancestor of the other.
    pub fn compare(
        &self,
        a: &JurisdictionId,
        b: &JurisdictionId,
    ) -> Option<std::cmp::Ordering> {
        if a == b {
            return Some(std::cmp::Ordering::Equal);
        }
        let ancestors_a = self.ancestor_set(a);
        let ancestors_b = self.ancestor_set(b);
        if ancestors_a.contains(b) {
            Some(std::cmp::Ordering::Less) // a is more specific
        } else if ancestors_b.contains(a) {
            Some(std::cmp::Ordering::Greater) // b is more specific
        } else {
            None // incomparable
        }
    }

    fn ancestor_set(&self, id: &JurisdictionId) -> BTreeSet<JurisdictionId> {
        let mut set = BTreeSet::new();
        set.insert(id.clone());
        for ancestor in self.registry.ancestors(id) {
            set.insert(ancestor);
        }
        set
    }

    fn depth(&self, id: &JurisdictionId) -> usize {
        self.registry.ancestors(id).len()
    }
}

/// Build a well-known registry with EU AI Act, NIST, etc.
pub fn well_known_registry() -> JurisdictionRegistry {
    let mut reg = JurisdictionRegistry::new();

    reg.register(
        JurisdictionInfoBuilder::new()
            .id("eu-ai-act")
            .name("EU Artificial Intelligence Act")
            .short_name("EU AI Act")
            .jurisdiction_type(JurisdictionType::Binding)
            .region(GeographicRegion::EU)
            .enforcement(EnforcementModel::Hybrid)
            .priority(1.0)
            .description("Comprehensive EU regulation on AI systems")
            .max_fine_fraction(0.07)
            .build()
            .unwrap(),
    );

    reg.register(
        JurisdictionInfoBuilder::new()
            .id("nist-ai-rmf")
            .name("NIST AI Risk Management Framework")
            .short_name("NIST AI RMF")
            .jurisdiction_type(JurisdictionType::Voluntary)
            .region(GeographicRegion::NorthAmerica)
            .enforcement(EnforcementModel::SelfCertification)
            .priority(0.6)
            .description("US voluntary AI risk management framework")
            .build()
            .unwrap(),
    );

    reg.register(
        JurisdictionInfoBuilder::new()
            .id("iso-42001")
            .name("ISO/IEC 42001 AI Management System")
            .short_name("ISO 42001")
            .jurisdiction_type(JurisdictionType::Standard)
            .region(GeographicRegion::Global)
            .enforcement(EnforcementModel::SelfCertification)
            .priority(0.5)
            .description("International standard for AI management systems")
            .build()
            .unwrap(),
    );

    reg.register(
        JurisdictionInfoBuilder::new()
            .id("uk-ai-regulation")
            .name("UK Pro-Innovation AI Regulation")
            .short_name("UK AI Reg")
            .jurisdiction_type(JurisdictionType::Binding)
            .region(GeographicRegion::UnitedKingdom)
            .enforcement(EnforcementModel::SectorSpecific)
            .priority(0.8)
            .description("UK sector-specific AI regulation approach")
            .max_fine_fraction(0.04)
            .build()
            .unwrap(),
    );

    reg
}

/// Compute the set of applicable jurisdictions given a target set of regions,
/// accounting for mutual recognition and extraterritorial reach.
pub fn applicable_jurisdictions(
    registry: &JurisdictionRegistry,
    target_regions: &[GeographicRegion],
) -> Vec<JurisdictionId> {
    let target_set: BTreeSet<GeographicRegion> = target_regions.iter().copied().collect();
    let mut applicable = BTreeSet::new();

    for (_id, info) in registry.iter() {
        let region_match = target_set.contains(&info.region) || info.region == GeographicRegion::Global;
        let extraterritorial = info.region.has_extraterritorial_reach()
            && info.jurisdiction_type == JurisdictionType::Binding;
        if region_match || extraterritorial {
            applicable.insert(info.id.clone());
        }
    }

    applicable.into_iter().collect()
}

/// Resolve priority ordering: returns jurisdictions sorted by effective weight descending.
pub fn priority_ordering(
    registry: &JurisdictionRegistry,
    ids: &[JurisdictionId],
) -> Vec<(JurisdictionId, f64)> {
    let mut weighted: Vec<(JurisdictionId, f64)> = ids
        .iter()
        .filter_map(|id| {
            registry.get(id).map(|info| (id.clone(), info.effective_weight()))
        })
        .collect();
    weighted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    weighted
}

/// Map from jurisdiction to its relative influence weight (normalized to sum=1).
pub fn normalized_weights(
    registry: &JurisdictionRegistry,
    ids: &[JurisdictionId],
) -> HashMap<JurisdictionId, f64> {
    let weights: Vec<(JurisdictionId, f64)> = ids
        .iter()
        .filter_map(|id| {
            registry.get(id).map(|info| (id.clone(), info.effective_weight()))
        })
        .collect();
    let total: f64 = weights.iter().map(|(_, w)| w).sum();
    if total <= 0.0 {
        return weights.into_iter().map(|(id, _)| (id, 0.0)).collect();
    }
    weights.into_iter().map(|(id, w)| (id, w / total)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jurisdiction_id() {
        let id = JurisdictionId::new("eu-ai-act");
        assert_eq!(id.as_str(), "eu-ai-act");
        assert_eq!(format!("{}", id), "eu-ai-act");
    }

    #[test]
    fn test_jurisdiction_type_strength() {
        assert_eq!(JurisdictionType::Binding.binding_strength(), 1.0);
        assert!(JurisdictionType::Voluntary.binding_strength() < JurisdictionType::Standard.binding_strength());
    }

    #[test]
    fn test_priority_weight_clamping() {
        assert_eq!(PriorityWeight::new(1.5).value(), 1.0);
        assert_eq!(PriorityWeight::new(-0.5).value(), 0.0);
    }

    #[test]
    fn test_priority_lattice() {
        let a = PriorityWeight::new(0.7);
        let b = PriorityWeight::new(0.3);
        assert_eq!(a.join(b).value(), 0.7);
        assert_eq!(a.meet(b).value(), 0.3);
    }

    #[test]
    fn test_builder() {
        let info = JurisdictionInfoBuilder::new()
            .id("test")
            .name("Test Jurisdiction")
            .jurisdiction_type(JurisdictionType::Voluntary)
            .region(GeographicRegion::Global)
            .enforcement(EnforcementModel::SelfCertification)
            .priority(0.4)
            .build()
            .unwrap();
        assert_eq!(info.id.as_str(), "test");
        assert!(!info.is_binding());
    }

    #[test]
    fn test_builder_missing_field() {
        let result = JurisdictionInfoBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_registry() {
        let reg = well_known_registry();
        assert!(reg.len() >= 4);
        assert!(reg.contains(&JurisdictionId::new("eu-ai-act")));

        let binding = reg.binding_jurisdictions();
        assert!(binding.iter().any(|j| j.id.as_str() == "eu-ai-act"));
    }

    #[test]
    fn test_registry_validation() {
        let reg = well_known_registry();
        let errors = reg.validate();
        assert!(errors.is_empty(), "well-known registry should be valid: {:?}", errors);
    }

    #[test]
    fn test_registry_by_region() {
        let reg = well_known_registry();
        let eu = reg.by_region(GeographicRegion::EU);
        assert!(!eu.is_empty());
    }

    #[test]
    fn test_applicable_jurisdictions() {
        let reg = well_known_registry();
        let applicable = applicable_jurisdictions(&reg, &[GeographicRegion::NorthAmerica]);
        // Should include NIST (NA), ISO (Global), and EU (extraterritorial)
        assert!(applicable.contains(&JurisdictionId::new("nist-ai-rmf")));
        assert!(applicable.contains(&JurisdictionId::new("iso-42001")));
    }

    #[test]
    fn test_priority_ordering() {
        let reg = well_known_registry();
        let ids: Vec<JurisdictionId> = reg.all_ids().into_iter().cloned().collect();
        let ordered = priority_ordering(&reg, &ids);
        assert!(!ordered.is_empty());
        // EU AI Act should be first (weight 1.0 * 1.0 = 1.0)
        assert_eq!(ordered[0].0.as_str(), "eu-ai-act");
    }

    #[test]
    fn test_normalized_weights() {
        let reg = well_known_registry();
        let ids: Vec<JurisdictionId> = reg.all_ids().into_iter().cloned().collect();
        let weights = normalized_weights(&reg, &ids);
        let sum: f64 = weights.values().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_jurisdiction_lattice_join_meet() {
        let mut reg = JurisdictionRegistry::new();
        reg.register(
            JurisdictionInfoBuilder::new()
                .id("eu")
                .name("EU")
                .jurisdiction_type(JurisdictionType::Binding)
                .region(GeographicRegion::EU)
                .enforcement(EnforcementModel::Hybrid)
                .build()
                .unwrap(),
        );
        reg.register(
            JurisdictionInfoBuilder::new()
                .id("eu-de")
                .name("Germany")
                .jurisdiction_type(JurisdictionType::Binding)
                .region(GeographicRegion::EU)
                .enforcement(EnforcementModel::Hybrid)
                .parent("eu")
                .build()
                .unwrap(),
        );
        reg.register(
            JurisdictionInfoBuilder::new()
                .id("eu-fr")
                .name("France")
                .jurisdiction_type(JurisdictionType::Binding)
                .region(GeographicRegion::EU)
                .enforcement(EnforcementModel::Hybrid)
                .parent("eu")
                .build()
                .unwrap(),
        );

        let lattice = JurisdictionLattice::new(&reg);

        // join of two siblings = their common parent
        let join = lattice.join(
            &JurisdictionId::new("eu-de"),
            &JurisdictionId::new("eu-fr"),
        );
        assert_eq!(join, Some(JurisdictionId::new("eu")));

        // meet of parent and child = child (more specific)
        let meet = lattice.meet(
            &JurisdictionId::new("eu"),
            &JurisdictionId::new("eu-de"),
        );
        assert_eq!(meet, Some(JurisdictionId::new("eu-de")));

        // compare: child < parent
        let cmp = lattice.compare(
            &JurisdictionId::new("eu-de"),
            &JurisdictionId::new("eu"),
        );
        assert_eq!(cmp, Some(std::cmp::Ordering::Less));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let reg = well_known_registry();
        let json = serde_json::to_string(&reg).unwrap();
        let deser: JurisdictionRegistry = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.len(), reg.len());
    }
}
