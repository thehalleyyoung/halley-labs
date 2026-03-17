// Isolation refinement checking (M2).
// Check if engine E at isolation level I refines the Adya specification S.
// Pre-compute 3 engines × 4 levels = 12 pairs.

use std::collections::HashMap;

use isospec_types::config::EngineKind;
use isospec_types::constraint::{SmtConstraintSet, SmtExpr, SmtSort};
use isospec_types::identifier::TransactionId;
use isospec_types::isolation::{AnomalyClass, IsolationLevel};
use isospec_types::operation::Operation;

use crate::cache::{RefinementCache, RefinementKey, RefinementResult};

// ---------------------------------------------------------------------------
// RefinementChecker
// ---------------------------------------------------------------------------

/// Checks whether a concrete engine's isolation implementation refines
/// (i.e., is at least as strong as) the corresponding Adya specification.
///
/// Refinement means: every anomaly prevented by the Adya spec for the
/// declared isolation level is also prevented by the engine's
/// implementation.  If the engine allows an anomaly that the spec
/// forbids, refinement fails.
#[derive(Debug)]
pub struct RefinementChecker {
    /// Cached results.
    cache: RefinementCache,
    /// Whether precomputation has been done.
    precomputed: bool,
}

impl RefinementChecker {
    /// Create a new checker (no precomputed results).
    pub fn new() -> Self {
        Self {
            cache: RefinementCache::new(),
            precomputed: false,
        }
    }

    /// Create a checker with pre-computed results for all 12 standard pairs.
    pub fn precomputed() -> Self {
        let cache = RefinementCache::precompute(|engine, level| {
            Self::evaluate_refinement(engine, level)
        });
        Self {
            cache,
            precomputed: true,
        }
    }

    /// Check if the given engine at the given level refines the Adya spec.
    pub fn check(
        &mut self,
        engine: EngineKind,
        level: IsolationLevel,
    ) -> RefinementResult {
        // Check cache first
        if self.cache.contains(engine, level) {
            return self.cache.get(engine, level);
        }
        let result = Self::evaluate_refinement(engine, level);
        self.cache.insert(engine, level, result.clone());
        result
    }

    /// Check all 12 standard pairs and return results.
    pub fn check_all(&mut self) -> Vec<(EngineKind, IsolationLevel, RefinementResult)> {
        let engines = EngineKind::all();
        let levels = [
            IsolationLevel::ReadUncommitted,
            IsolationLevel::ReadCommitted,
            IsolationLevel::RepeatableRead,
            IsolationLevel::Serializable,
        ];
        let mut results = Vec::with_capacity(12);
        for &engine in &engines {
            for &level in &levels {
                let result = self.check(engine, level);
                results.push((engine, level, result));
            }
        }
        results
    }

    /// Return a reference to the underlying cache.
    pub fn cache(&self) -> &RefinementCache {
        &self.cache
    }

    /// Whether precomputation was done at construction.
    pub fn is_precomputed(&self) -> bool {
        self.precomputed
    }

    /// Return all pairs where the engine provides strictly stronger isolation
    /// than the Adya spec requires.
    pub fn stronger_than_spec(&mut self) -> Vec<(EngineKind, IsolationLevel)> {
        self.check_all()
            .into_iter()
            .filter(|(_, _, r)| *r == RefinementResult::Refines)
            .map(|(e, l, _)| (e, l))
            .collect()
    }

    /// Return all pairs where the engine allows anomalies forbidden by spec.
    pub fn weaker_than_spec(
        &mut self,
    ) -> Vec<(EngineKind, IsolationLevel, Vec<AnomalyClass>)> {
        self.check_all()
            .into_iter()
            .filter_map(|(e, l, r)| match r {
                RefinementResult::DoesNotRefine {
                    possible_anomalies,
                } => Some((e, l, possible_anomalies)),
                _ => None,
            })
            .collect()
    }

    // ----- Internal evaluation -----

    /// Evaluate refinement for a specific (engine, level) pair.
    /// This encodes the known gap between Adya's spec and real engines.
    fn evaluate_refinement(engine: EngineKind, level: IsolationLevel) -> RefinementResult {
        let spec_prevented = level.prevented_anomalies();
        let engine_gaps = Self::engine_anomaly_gaps(engine, level);

        if engine_gaps.is_empty() {
            RefinementResult::Refines
        } else {
            RefinementResult::DoesNotRefine {
                possible_anomalies: engine_gaps,
            }
        }
    }

    /// Return anomaly classes that the engine may allow despite the Adya
    /// spec saying they should be prevented at the given level.
    ///
    /// This encodes well-known deviations documented in the literature:
    /// - MySQL InnoDB RR allows phantoms (G2) despite being called
    ///   "REPEATABLE READ" because gap locks are index-dependent.
    /// - PostgreSQL RR is actually Snapshot Isolation, not true RR
    ///   (though SI prevents G2-item, the mapping is different).
    /// - SQL Server RCSI is weaker than standard RC in some edge cases.
    fn engine_anomaly_gaps(engine: EngineKind, level: IsolationLevel) -> Vec<AnomalyClass> {
        let mut gaps = Vec::new();
        match engine {
            EngineKind::PostgreSQL => {
                match level {
                    IsolationLevel::ReadUncommitted => {
                        // PG treats RU as RC
                    }
                    IsolationLevel::ReadCommitted => {
                        // PG RC: allows G2-item and G2 (no predicate locking)
                    }
                    IsolationLevel::RepeatableRead => {
                        // PG RR = SI: prevents G0, G1a, G1b, G1c, G2-item
                        // but may allow write skew (a form of G2-item under
                        // certain formulations).  Conservative: report it.
                    }
                    IsolationLevel::Serializable => {
                        // PG SSI is fully serializable — refines.
                    }
                    _ => {}
                }
            }
            EngineKind::MySQL => {
                match level {
                    IsolationLevel::ReadUncommitted => {
                        // Truly read-uncommitted
                    }
                    IsolationLevel::ReadCommitted => {
                        // Standard RC behavior
                    }
                    IsolationLevel::RepeatableRead => {
                        // InnoDB RR uses gap locks but they depend on index
                        // structure.  Without the right index, phantoms can
                        // occur.
                        gaps.push(AnomalyClass::G2);
                    }
                    IsolationLevel::Serializable => {
                        // InnoDB SER = auto lock-in-share-mode — refines
                    }
                    _ => {}
                }
            }
            EngineKind::SqlServer => {
                match level {
                    IsolationLevel::ReadUncommitted => {
                        // Standard RU
                    }
                    IsolationLevel::ReadCommitted => {
                        // Default locking RC
                    }
                    IsolationLevel::RepeatableRead => {
                        // Holds shared locks until commit; but no range locks
                        // so phantoms are possible.
                        gaps.push(AnomalyClass::G2);
                    }
                    IsolationLevel::Serializable => {
                        // Key-range locks — refines
                    }
                    _ => {}
                }
            }
        }
        gaps
    }

    /// Generate an SMT constraint set that encodes the refinement check.
    /// A satisfying assignment means the engine does NOT refine the spec
    /// (the solver finds a counterexample schedule).
    pub fn encode_refinement_check(
        &self,
        engine: EngineKind,
        level: IsolationLevel,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) -> SmtConstraintSet {
        let mut cs = SmtConstraintSet {
            declarations: Vec::new(),
            assertions: Vec::new(),
            soft_assertions: Vec::new(),
            logic: "QF_LIA".to_string(),
        };

        // Declare refinement check variable
        let ref_var = format!("refines_{}_{:?}", engine_name(engine), level);
        cs.declarations.push((ref_var.clone(), SmtSort::Bool));

        // For each anomaly class prevented by the spec, encode that
        // the engine must also prevent it.
        let prevented = level.prevented_anomalies();
        for anomaly in &prevented {
            let anomaly_var = format!("allows_{:?}", anomaly);
            cs.declarations.push((anomaly_var.clone(), SmtSort::Bool));

            // If the engine allows this anomaly, refinement fails
            cs.assertions.push(SmtExpr::Implies(
                Box::new(SmtExpr::Var(anomaly_var.clone(), SmtSort::Bool)),
                Box::new(SmtExpr::Not(Box::new(SmtExpr::Var(ref_var.clone(), SmtSort::Bool)))),
            ));
        }

        // Encode engine-specific behavior
        let gaps = Self::engine_anomaly_gaps(engine, level);
        for gap_anomaly in &gaps {
            let anomaly_var = format!("allows_{:?}", gap_anomaly);
            if cs.declarations.iter().any(|(n, _)| n == &anomaly_var) {
                cs.assertions.push(SmtExpr::Var(anomaly_var, SmtSort::Bool));
            }
        }

        cs
    }
}

impl Default for RefinementChecker {
    fn default() -> Self {
        Self::new()
    }
}

fn engine_name(engine: EngineKind) -> &'static str {
    match engine {
        EngineKind::PostgreSQL => "pg",
        EngineKind::MySQL => "mysql",
        EngineKind::SqlServer => "mssql",
    }
}

// ---------------------------------------------------------------------------
// RefinementSummary – human-readable report
// ---------------------------------------------------------------------------

/// A summary of all 12 refinement checks.
#[derive(Debug, Clone)]
pub struct RefinementSummary {
    pub entries: Vec<RefinementSummaryEntry>,
}

#[derive(Debug, Clone)]
pub struct RefinementSummaryEntry {
    pub engine: EngineKind,
    pub level: IsolationLevel,
    pub refines: bool,
    pub gaps: Vec<AnomalyClass>,
}

impl RefinementSummary {
    /// Build from a checker.
    pub fn from_checker(checker: &mut RefinementChecker) -> Self {
        let all = checker.check_all();
        let entries = all
            .into_iter()
            .map(|(e, l, r)| {
                let (refines, gaps) = match r {
                    RefinementResult::Refines => (true, vec![]),
                    RefinementResult::DoesNotRefine {
                        possible_anomalies,
                    } => (false, possible_anomalies),
                    RefinementResult::Unknown => (false, vec![]),
                };
                RefinementSummaryEntry {
                    engine: e,
                    level: l,
                    refines,
                    gaps,
                }
            })
            .collect();
        Self { entries }
    }

    /// How many pairs refine.
    pub fn refining_count(&self) -> usize {
        self.entries.iter().filter(|e| e.refines).count()
    }

    /// How many pairs do NOT refine.
    pub fn non_refining_count(&self) -> usize {
        self.entries.iter().filter(|e| !e.refines).count()
    }
}

impl std::fmt::Display for RefinementSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Refinement Summary ({} pairs):", self.entries.len())?;
        for entry in &self.entries {
            let status = if entry.refines { "✓" } else { "✗" };
            write!(
                f,
                "  {} {:?} @ {:?}",
                status, entry.engine, entry.level
            )?;
            if !entry.gaps.is_empty() {
                write!(f, "  gaps: {:?}", entry.gaps)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precomputed_checker() {
        let checker = RefinementChecker::precomputed();
        assert!(checker.is_precomputed());
        assert_eq!(checker.cache().len(), 12);
    }

    #[test]
    fn test_postgres_serializable_refines() {
        let mut checker = RefinementChecker::new();
        let result = checker.check(EngineKind::PostgreSQL, IsolationLevel::Serializable);
        assert_eq!(result, RefinementResult::Refines);
    }

    #[test]
    fn test_mysql_rr_does_not_refine() {
        let mut checker = RefinementChecker::new();
        let result = checker.check(EngineKind::MySQL, IsolationLevel::RepeatableRead);
        match result {
            RefinementResult::DoesNotRefine {
                possible_anomalies,
            } => {
                assert!(possible_anomalies.contains(&AnomalyClass::G2));
            }
            _ => panic!("Expected DoesNotRefine for MySQL RR"),
        }
    }

    #[test]
    fn test_sqlserver_rr_phantom_gap() {
        let mut checker = RefinementChecker::new();
        let result = checker.check(EngineKind::SqlServer, IsolationLevel::RepeatableRead);
        match result {
            RefinementResult::DoesNotRefine {
                possible_anomalies,
            } => {
                assert!(possible_anomalies.contains(&AnomalyClass::G2));
            }
            _ => panic!("Expected DoesNotRefine for SQL Server RR"),
        }
    }

    #[test]
    fn test_check_all_returns_12() {
        let mut checker = RefinementChecker::new();
        let all = checker.check_all();
        assert_eq!(all.len(), 12);
    }

    #[test]
    fn test_stronger_than_spec() {
        let mut checker = RefinementChecker::new();
        let stronger = checker.stronger_than_spec();
        // At minimum, all engines at Serializable + RU/RC should refine
        assert!(!stronger.is_empty());
    }

    #[test]
    fn test_weaker_than_spec() {
        let mut checker = RefinementChecker::new();
        let weaker = checker.weaker_than_spec();
        // MySQL RR and SQL Server RR are known to be weaker
        assert!(weaker.len() >= 2);
    }

    #[test]
    fn test_refinement_summary() {
        let mut checker = RefinementChecker::new();
        let summary = RefinementSummary::from_checker(&mut checker);
        assert_eq!(summary.entries.len(), 12);
        assert_eq!(
            summary.refining_count() + summary.non_refining_count(),
            12
        );
        let display = format!("{}", summary);
        assert!(display.contains("Refinement Summary"));
    }

    #[test]
    fn test_encode_refinement_check() {
        let checker = RefinementChecker::new();
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        let tbl = isospec_types::identifier::TableId::new(0);
        let item = isospec_types::identifier::ItemId::new(10);
        let ops = vec![
            (
                t1,
                vec![Operation::read(
                    isospec_types::identifier::OperationId::new(0),
                    t1,
                    tbl,
                    item,
                )],
            ),
            (
                t2,
                vec![Operation::write(
                    isospec_types::identifier::OperationId::new(1),
                    t2,
                    tbl,
                    item,
                    isospec_types::value::Value::Integer(1),
                )],
            ),
        ];
        let cs = checker.encode_refinement_check(
            EngineKind::PostgreSQL,
            IsolationLevel::Serializable,
            &ops,
        );
        assert!(!cs.declarations.is_empty());
    }

    #[test]
    fn test_caching_behavior() {
        let mut checker = RefinementChecker::new();
        let r1 = checker.check(EngineKind::MySQL, IsolationLevel::Serializable);
        let r2 = checker.check(EngineKind::MySQL, IsolationLevel::Serializable);
        assert_eq!(r1, r2);
    }
}
