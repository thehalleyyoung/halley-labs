use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SecurityLevel { Public, Secret }

impl SecurityLevel {
    pub fn join(self, other: Self) -> Self { match (self, other) { (Self::Secret, _) | (_, Self::Secret) => Self::Secret, _ => Self::Public } }
    pub fn meet(self, other: Self) -> Self { match (self, other) { (Self::Public, _) | (_, Self::Public) => Self::Public, _ => Self::Secret } }
    pub fn is_secret(self) -> bool { self == Self::Secret }
    pub fn is_public(self) -> bool { self == Self::Public }
}

impl Default for SecurityLevel { fn default() -> Self { Self::Public } }
impl fmt::Display for SecurityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Public => write!(f, "public"), Self::Secret => write!(f, "secret") }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SecurityLattice { levels: Vec<(String, u32)> }
impl SecurityLattice {
    pub fn two_point() -> Self { Self { levels: vec![("public".into(), 0), ("secret".into(), 1)] } }
    pub fn level_count(&self) -> usize { self.levels.len() }
    pub fn join_levels(&self, a: u32, b: u32) -> u32 { a.max(b) }
    pub fn meet_levels(&self, a: u32, b: u32) -> u32 { a.min(b) }
    pub fn flows_to(&self, from: u32, to: u32) -> bool { from <= to }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_sec() { assert_eq!(SecurityLevel::Public.join(SecurityLevel::Secret), SecurityLevel::Secret); }
    #[test] fn test_lat() { let l = SecurityLattice::two_point(); assert!(l.flows_to(0, 1)); assert!(!l.flows_to(1, 0)); }
}
