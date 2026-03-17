use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposedTransformation {
    pub name: String,
    pub components: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionConfidence {
    pub score: f64,
    pub compatible: bool,
}

pub struct CompositionOptimizer;
impl CompositionOptimizer { pub fn new() -> Self { Self } }
impl Default for CompositionOptimizer { fn default() -> Self { Self } }

pub struct CompositionValidator;
impl CompositionValidator { pub fn new() -> Self { Self } }
impl Default for CompositionValidator { fn default() -> Self { Self } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageMatrix {
    pub data: Vec<Vec<bool>>,
    pub row_labels: Vec<String>,
    pub col_labels: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionOverlap { None, Partial, Full }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyntacticPosition { Subject, Verb, Object, Modifier, Clause, Root }
