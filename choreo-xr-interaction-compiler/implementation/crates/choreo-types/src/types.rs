//! Type system types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpatialType {
    Point,
    Region(RegionType),
    Entity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegionType {
    Sphere,
    Box,
    ConvexHull,
    Composite,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalType {
    Instant,
    Interval,
    Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    Gesture(String),
    Spatial(String),
    Composite(Vec<InteractionType>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChoreographyType {
    Sequential(Vec<ChoreographyType>),
    Parallel(Vec<ChoreographyType>),
    Choice(Vec<ChoreographyType>),
    Interaction(InteractionType),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeEnv {
    pub bindings: HashMap<String, TypeId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeConstraint {
    Equal(TypeId, TypeId),
    Subtype(TypeId, TypeId),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subtyping {
    pub pairs: Vec<(TypeId, TypeId)>,
}
