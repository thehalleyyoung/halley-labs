use serde::{Deserialize, Serialize};

pub trait MetamorphicProperty: Send + Sync {
    fn name(&self) -> &str;
    fn holds(&self, original: &str, transformed: &str) -> bool;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyResult {
    pub property_name: String,
    pub holds: bool,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropertyScope { Global, StageLocal, PairWise }

pub struct PropertyChecker;
impl PropertyChecker { pub fn new() -> Self { Self } }
impl Default for PropertyChecker { fn default() -> Self { Self } }

pub struct PropertyGenerator;
impl PropertyGenerator { pub fn new() -> Self { Self } }
impl Default for PropertyGenerator { fn default() -> Self { Self } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionRule {
    pub name: String,
    pub compatible_pairs: Vec<(String, String)>,
}

pub struct DisjointPositionChecker;
impl DisjointPositionChecker { pub fn new() -> Self { Self } }
impl Default for DisjointPositionChecker { fn default() -> Self { Self } }
