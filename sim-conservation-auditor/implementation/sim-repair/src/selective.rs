//! Selective repair (only fix specific conservation laws).
use serde::{Serialize, Deserialize};

/// Selectively repairs only specified conservation laws.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SelectiveRepair { pub enabled_laws: Vec<String> }
impl SelectiveRepair {
    pub fn new() -> Self { Self::default() }
    pub fn enable(&mut self, law: impl Into<String>) { self.enabled_laws.push(law.into()); }
    pub fn is_enabled(&self, law: &str) -> bool { self.enabled_laws.is_empty() || self.enabled_laws.iter().any(|l| l == law) }
}
