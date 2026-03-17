//! Choreo CLI — library interface.
//!
//! Re-exports key types from the Choreo workspace crates and provides
//! shared utilities consumed by the `choreo` binary and downstream
//! tooling.

// ── Re-exports: foundation types ────────────────────────────────────
pub use choreo_types;
pub use choreo_dsl;
pub use choreo_spatial;
pub use choreo_ec;
pub use choreo_automata;
pub use choreo_cegar;
pub use choreo_codegen;
pub use choreo_gesture;
pub use choreo_conflict;
pub use choreo_runtime;
pub use choreo_simulator;

/// Return the crate version baked in at compile time.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        assert!(!version().is_empty());
    }
}
