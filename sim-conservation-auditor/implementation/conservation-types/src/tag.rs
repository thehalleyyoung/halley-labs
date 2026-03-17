//! Provenance tagging for tracking conservation law violations back to source.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// A unique identifier for a provenance tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Tag(pub u64);

impl Tag {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "τ{}", self.0)
    }
}

/// A tag set representing the combined provenance of a computation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TagSet {
    pub tags: HashSet<Tag>,
    pub operation_chain: Vec<TagOperation>,
}

/// Operations that combine tags.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TagOperation {
    /// Tags combined via addition.
    Sum(Tag, Tag),
    /// Tags combined via multiplication.
    Product(Tag, Tag),
    /// Tag from a composition step.
    Compose(Tag, Tag),
    /// Tag from a BCH expansion term.
    BchTerm { order: u32, source: Tag },
    /// Tag from differentiation.
    Differentiate(Tag),
    /// Tag from truncation error.
    Truncation { order: u32, source: Tag },
}

impl TagSet {
    pub fn empty() -> Self {
        Self {
            tags: HashSet::new(),
            operation_chain: Vec::new(),
        }
    }

    pub fn singleton(tag: Tag) -> Self {
        let mut tags = HashSet::new();
        tags.insert(tag);
        Self {
            tags,
            operation_chain: Vec::new(),
        }
    }

    pub fn union(&self, other: &TagSet) -> Self {
        let mut tags = self.tags.clone();
        tags.extend(other.tags.iter());
        let mut ops = self.operation_chain.clone();
        ops.extend(other.operation_chain.iter().cloned());
        Self {
            tags,
            operation_chain: ops,
        }
    }

    pub fn with_operation(mut self, op: TagOperation) -> Self {
        self.operation_chain.push(op);
        self
    }

    pub fn contains(&self, tag: &Tag) -> bool {
        self.tags.contains(tag)
    }

    pub fn len(&self) -> usize {
        self.tags.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }
}

/// Registry mapping tags to source locations and descriptions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagRegistry {
    next_id: u64,
    tag_info: HashMap<Tag, TagInfo>,
}

/// Information associated with a provenance tag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagInfo {
    pub tag: Tag,
    pub description: String,
    pub source_file: Option<String>,
    pub source_line: Option<u32>,
    pub source_col: Option<u32>,
    pub category: TagCategory,
    pub created_by: String,
}

/// Categories of provenance tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TagCategory {
    /// Tag from source code extraction.
    SourceCode,
    /// Tag from a force computation.
    ForceComputation,
    /// Tag from a position update.
    PositionUpdate,
    /// Tag from a velocity update.
    VelocityUpdate,
    /// Tag from a splitting step.
    SplittingStep,
    /// Tag from a coupling term.
    CouplingTerm,
    /// Tag from a thermostat.
    Thermostat,
    /// Tag from a constraint.
    Constraint,
    /// Tag from a boundary condition.
    BoundaryCondition,
    /// Tag from numerical truncation.
    Truncation,
    /// Tag from round-off error.
    RoundOff,
    /// User-defined.
    Custom,
}

impl TagRegistry {
    pub fn new() -> Self {
        Self {
            next_id: 0,
            tag_info: HashMap::new(),
        }
    }

    /// Create a new tag and register it.
    pub fn create_tag(&mut self, description: impl Into<String>, category: TagCategory) -> Tag {
        let tag = Tag::new(self.next_id);
        self.next_id += 1;
        self.tag_info.insert(
            tag,
            TagInfo {
                tag,
                description: description.into(),
                source_file: None,
                source_line: None,
                source_col: None,
                category,
                created_by: String::new(),
            },
        );
        tag
    }

    /// Create a tag with source location.
    pub fn create_source_tag(
        &mut self,
        description: impl Into<String>,
        file: impl Into<String>,
        line: u32,
        col: u32,
    ) -> Tag {
        let tag = Tag::new(self.next_id);
        self.next_id += 1;
        self.tag_info.insert(
            tag,
            TagInfo {
                tag,
                description: description.into(),
                source_file: Some(file.into()),
                source_line: Some(line),
                source_col: Some(col),
                category: TagCategory::SourceCode,
                created_by: String::new(),
            },
        );
        tag
    }

    /// Look up information about a tag.
    pub fn lookup(&self, tag: &Tag) -> Option<&TagInfo> {
        self.tag_info.get(tag)
    }

    /// Get all tags of a specific category.
    pub fn tags_by_category(&self, category: TagCategory) -> Vec<Tag> {
        self.tag_info
            .iter()
            .filter(|(_, info)| info.category == category)
            .map(|(&tag, _)| tag)
            .collect()
    }

    /// Number of registered tags.
    pub fn len(&self) -> usize {
        self.tag_info.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tag_info.is_empty()
    }
}

impl Default for TagRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Propagation rules for tags through mathematical operations.
#[derive(Debug, Clone)]
pub struct TagPropagator {
    rules: Vec<PropagationRule>,
}

/// A rule for how tags propagate through operations.
#[derive(Debug, Clone)]
pub struct PropagationRule {
    pub name: String,
    pub operation: String,
    pub merge_strategy: MergeStrategy,
}

/// How to merge tag sets when combining expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    Union,
    Intersection,
    LeftOnly,
    RightOnly,
    Weighted,
}

impl TagPropagator {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule(&mut self, rule: PropagationRule) {
        self.rules.push(rule);
    }

    /// Propagate tags through an addition operation.
    pub fn propagate_add(&self, left: &TagSet, right: &TagSet) -> TagSet {
        left.union(right)
    }

    /// Propagate tags through a multiplication operation.
    pub fn propagate_mul(&self, left: &TagSet, right: &TagSet) -> TagSet {
        let result = left.union(right);
        if let (Some(&l), Some(&r)) = (left.tags.iter().next(), right.tags.iter().next()) {
            result.with_operation(TagOperation::Product(l, r))
        } else {
            result
        }
    }

    /// Propagate tags through BCH expansion.
    pub fn propagate_bch(&self, source: &TagSet, order: u32) -> TagSet {
        let mut result = source.clone();
        for &tag in &source.tags {
            result.operation_chain.push(TagOperation::BchTerm {
                order,
                source: tag,
            });
        }
        result
    }
}

impl Default for TagPropagator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tag_creation() {
        let mut reg = TagRegistry::new();
        let t1 = reg.create_tag("force computation", TagCategory::ForceComputation);
        let t2 = reg.create_tag("position update", TagCategory::PositionUpdate);
        assert_ne!(t1, t2);
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn test_tag_set_union() {
        let s1 = TagSet::singleton(Tag::new(0));
        let s2 = TagSet::singleton(Tag::new(1));
        let u = s1.union(&s2);
        assert_eq!(u.len(), 2);
    }

    #[test]
    fn test_tag_propagation() {
        let prop = TagPropagator::new();
        let s1 = TagSet::singleton(Tag::new(0));
        let s2 = TagSet::singleton(Tag::new(1));
        let result = prop.propagate_add(&s1, &s2);
        assert!(result.contains(&Tag::new(0)));
        assert!(result.contains(&Tag::new(1)));
    }

    #[test]
    fn test_tag_registry_lookup() {
        let mut reg = TagRegistry::new();
        let t = reg.create_source_tag("leapfrog kick", "sim.py", 42, 8);
        let info = reg.lookup(&t).unwrap();
        assert_eq!(info.source_line, Some(42));
    }

    #[test]
    fn test_tags_by_category() {
        let mut reg = TagRegistry::new();
        reg.create_tag("f1", TagCategory::ForceComputation);
        reg.create_tag("f2", TagCategory::ForceComputation);
        reg.create_tag("p1", TagCategory::PositionUpdate);
        let force_tags = reg.tags_by_category(TagCategory::ForceComputation);
        assert_eq!(force_tags.len(), 2);
    }
}
