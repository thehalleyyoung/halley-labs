//! Conservation law registry for managing collections of checkers.

use std::collections::HashMap;

/// Registry for conservation law checkers.
#[derive(Default)]
pub struct LawRegistry {
    laws: HashMap<String, Box<dyn crate::ConservationChecker>>,
}

impl LawRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self { Self::default() }

    /// Register a conservation law checker.
    pub fn register(&mut self, checker: Box<dyn crate::ConservationChecker>) {
        let name = checker.name().to_string();
        self.laws.insert(name, checker);
    }

    /// Get a checker by name.
    pub fn get(&self, name: &str) -> Option<&dyn crate::ConservationChecker> {
        self.laws.get(name).map(|b| b.as_ref())
    }

    /// List all registered law names.
    pub fn names(&self) -> Vec<&str> {
        self.laws.keys().map(|s| s.as_str()).collect()
    }

    /// Number of registered laws.
    pub fn len(&self) -> usize { self.laws.len() }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool { self.laws.is_empty() }
}

impl std::fmt::Debug for LawRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LawRegistry")
            .field("count", &self.laws.len())
            .field("laws", &self.names())
            .finish()
    }
}
