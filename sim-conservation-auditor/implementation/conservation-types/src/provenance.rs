//! Provenance tag algebra for tracking conservation-relevant transformations.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fmt;

/// Unique identifier for a provenance tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TagId(pub u64);

impl TagId {
    pub fn new(id: u64) -> Self { Self(id) }
}

impl fmt::Display for TagId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T{}", self.0)
    }
}

/// A provenance tag tracking conservation-relevant operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProvenanceTag {
    pub id: TagId,
    pub source_ids: BTreeSet<u64>,
    pub operation_kind: TagOperationKind,
    pub order: usize,
    pub metadata: TagMetadata,
}

impl ProvenanceTag {
    pub fn new(id: TagId, kind: TagOperationKind) -> Self {
        let mut source_ids = BTreeSet::new();
        source_ids.insert(id.0);
        Self {
            id,
            source_ids,
            operation_kind: kind,
            order: 0,
            metadata: TagMetadata::default(),
        }
    }

    pub fn identity() -> Self {
        Self {
            id: TagId(0),
            source_ids: BTreeSet::new(),
            operation_kind: TagOperationKind::Identity,
            order: 0,
            metadata: TagMetadata::default(),
        }
    }

    pub fn combine(&self, other: &ProvenanceTag) -> ProvenanceTag {
        let mut source_ids = self.source_ids.clone();
        source_ids.extend(&other.source_ids);
        let new_id = TagId(self.id.0.wrapping_mul(31).wrapping_add(other.id.0));
        ProvenanceTag {
            id: new_id,
            source_ids,
            operation_kind: TagOperationKind::Composition(
                Box::new(self.operation_kind.clone()),
                Box::new(other.operation_kind.clone()),
            ),
            order: self.order.max(other.order) + 1,
            metadata: self.metadata.merge(&other.metadata),
        }
    }

    pub fn bracket(&self, other: &ProvenanceTag) -> ProvenanceTag {
        let mut source_ids = self.source_ids.clone();
        source_ids.extend(&other.source_ids);
        let new_id = TagId(self.id.0.wrapping_mul(37).wrapping_add(other.id.0));
        ProvenanceTag {
            id: new_id,
            source_ids,
            operation_kind: TagOperationKind::Bracket(
                Box::new(self.operation_kind.clone()),
                Box::new(other.operation_kind.clone()),
            ),
            order: self.order + other.order + 1,
            metadata: self.metadata.merge(&other.metadata),
        }
    }

    pub fn contributes_from(&self, source: u64) -> bool {
        self.source_ids.contains(&source)
    }

    pub fn source_count(&self) -> usize {
        self.source_ids.len()
    }

    pub fn depth(&self) -> usize {
        self.operation_kind.depth()
    }
}

impl fmt::Display for ProvenanceTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tag({}, sources={:?}, order={})", self.id, self.source_ids, self.order)
    }
}

/// The kind of operation a tag represents.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TagOperationKind {
    Identity,
    Arithmetic(ArithmeticOp),
    Differentiation { variable: String },
    Integration { variable: String },
    Composition(Box<TagOperationKind>, Box<TagOperationKind>),
    Bracket(Box<TagOperationKind>, Box<TagOperationKind>),
    Projection { component: usize },
    ForceComputation { force_type: String },
    TimeStep,
    Custom(String),
}

impl TagOperationKind {
    pub fn depth(&self) -> usize {
        match self {
            TagOperationKind::Composition(a, b) | TagOperationKind::Bracket(a, b) => {
                1 + a.depth().max(b.depth())
            }
            _ => 0,
        }
    }
}

/// Arithmetic operation types.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ArithmeticOp {
    Add, Sub, Mul, Div, Neg, Pow, Sqrt, Sin, Cos, Exp, Log,
}

/// Metadata associated with a provenance tag.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TagMetadata {
    pub source_file: Option<String>,
    pub source_line: Option<usize>,
    pub integrator_name: Option<String>,
    pub substep_index: Option<usize>,
}

impl TagMetadata {
    pub fn merge(&self, other: &TagMetadata) -> TagMetadata {
        TagMetadata {
            source_file: self.source_file.clone().or_else(|| other.source_file.clone()),
            source_line: self.source_line.or(other.source_line),
            integrator_name: self.integrator_name.clone().or_else(|| other.integrator_name.clone()),
            substep_index: self.substep_index.or(other.substep_index),
        }
    }
}

/// The tag algebra structure.
#[derive(Debug, Clone)]
pub struct TagAlgebra {
    tags: HashMap<TagId, ProvenanceTag>,
    next_id: u64,
    max_order: usize,
}

impl TagAlgebra {
    pub fn new(max_order: usize) -> Self {
        Self { tags: HashMap::new(), next_id: 1, max_order }
    }

    pub fn create_tag(&mut self, kind: TagOperationKind) -> ProvenanceTag {
        let id = TagId(self.next_id);
        self.next_id += 1;
        let tag = ProvenanceTag::new(id, kind);
        self.tags.insert(id, tag.clone());
        tag
    }

    pub fn compose(&mut self, a: &ProvenanceTag, b: &ProvenanceTag) -> Option<ProvenanceTag> {
        let result = a.combine(b);
        if result.order > self.max_order { return None; }
        self.tags.insert(result.id, result.clone());
        Some(result)
    }

    pub fn bracket(&mut self, a: &ProvenanceTag, b: &ProvenanceTag) -> Option<ProvenanceTag> {
        let result = a.bracket(b);
        if result.order > self.max_order { return None; }
        self.tags.insert(result.id, result.clone());
        Some(result)
    }

    pub fn get(&self, id: &TagId) -> Option<&ProvenanceTag> {
        self.tags.get(id)
    }

    pub fn tags_from_source(&self, source: u64) -> Vec<&ProvenanceTag> {
        self.tags.values().filter(|t| t.contributes_from(source)).collect()
    }

    pub fn tag_count(&self) -> usize { self.tags.len() }

    pub fn max_depth(&self) -> usize {
        self.tags.values().map(|t| t.depth()).max().unwrap_or(0)
    }

    pub fn prune(&mut self, max_depth: usize) {
        self.tags.retain(|_, t| t.depth() <= max_depth);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tag_creation() {
        let mut algebra = TagAlgebra::new(10);
        let t1 = algebra.create_tag(TagOperationKind::TimeStep);
        let t2 = algebra.create_tag(TagOperationKind::ForceComputation { force_type: "gravity".to_string() });
        assert_ne!(t1.id, t2.id);
    }

    #[test]
    fn test_tag_composition() {
        let mut algebra = TagAlgebra::new(10);
        let t1 = algebra.create_tag(TagOperationKind::TimeStep);
        let t2 = algebra.create_tag(TagOperationKind::TimeStep);
        let composed = algebra.compose(&t1, &t2).unwrap();
        assert!(composed.contributes_from(t1.id.0));
        assert!(composed.contributes_from(t2.id.0));
    }

    #[test]
    fn test_tag_bracket() {
        let mut algebra = TagAlgebra::new(10);
        let t1 = algebra.create_tag(TagOperationKind::TimeStep);
        let t2 = algebra.create_tag(TagOperationKind::ForceComputation { force_type: "spring".to_string() });
        let bracket = algebra.bracket(&t1, &t2).unwrap();
        assert_eq!(bracket.source_count(), 2);
    }

    #[test]
    fn test_tag_order_limit() {
        let mut algebra = TagAlgebra::new(2);
        let t1 = algebra.create_tag(TagOperationKind::TimeStep);
        let t2 = algebra.create_tag(TagOperationKind::TimeStep);
        let c1 = algebra.compose(&t1, &t2).unwrap();
        let c2 = algebra.compose(&c1, &t1).unwrap();
        let c3 = algebra.compose(&c2, &t1);
        assert!(c3.is_none());
    }

    #[test]
    fn test_tag_identity() {
        let t = ProvenanceTag::identity();
        assert_eq!(t.id, TagId(0));
        assert_eq!(t.order, 0);
    }
}
