//! Anomaly classification engine based on the Adya formalization.
//!
//! Classifies dependency cycles found in a Direct Serialization Graph (DSG)
//! into the standard anomaly hierarchy: G0, G1a, G1b, G1c, G2-item, G2.

use std::collections::{HashMap, HashSet};

use isospec_types::dependency::{Dependency, DependencyType};
use isospec_types::identifier::{ItemId, TransactionId};
use isospec_types::isolation::{AnomalyClass, AnomalySeverity, IsolationLevel};

// ---------------------------------------------------------------------------
// CycleInfo
// ---------------------------------------------------------------------------

/// A dependency cycle extracted from the DSG.
#[derive(Debug, Clone)]
pub struct CycleInfo {
    /// Ordered transaction ids forming the cycle (first == last is *not* repeated).
    pub nodes: Vec<TransactionId>,
    /// Edges along the cycle in traversal order.
    pub edges: Vec<Dependency>,
}

impl CycleInfo {
    pub fn new(nodes: Vec<TransactionId>, edges: Vec<Dependency>) -> Self {
        Self { nodes, edges }
    }

    /// Number of distinct transactions in the cycle.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the cycle is empty (degenerate).
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Collect the dependency types present along the cycle edges.
    pub fn edge_types(&self) -> Vec<DependencyType> {
        self.edges.iter().map(|e| e.dep_type).collect()
    }

    /// Returns `true` when at least one edge carries the given type.
    pub fn has_edge_type(&self, dt: DependencyType) -> bool {
        self.edges.iter().any(|e| e.dep_type == dt)
    }

    /// Returns `true` if the given transaction participates in the cycle.
    pub fn contains_txn(&self, txn: TransactionId) -> bool {
        self.nodes.contains(&txn)
    }

    /// Every edge is item-level (no predicate-level dependencies).
    pub fn all_item_level(&self) -> bool {
        self.edges.iter().all(|e| !e.dep_type.is_predicate_level())
    }

    /// At least one edge is predicate-level.
    pub fn has_predicate_edge(&self) -> bool {
        self.edges.iter().any(|e| e.dep_type.is_predicate_level())
    }

    /// At least one edge is an anti-dependency (rw / prw).
    pub fn has_anti_dependency_edge(&self) -> bool {
        self.edges.iter().any(|e| e.dep_type.is_anti_dependency())
    }

    /// True when any node in the cycle is in the supplied `aborted` set.
    pub fn involves_aborted(&self, aborted: &HashSet<TransactionId>) -> bool {
        self.nodes.iter().any(|n| aborted.contains(n))
    }

    /// All distinct item-ids referenced by edges.
    pub fn referenced_items(&self) -> HashSet<ItemId> {
        self.edges.iter().filter_map(|e| e.item_id).collect()
    }
}

// ---------------------------------------------------------------------------
// Confidence
// ---------------------------------------------------------------------------

/// How confident the classifier is in its determination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Confidence {
    Definite,
    Likely,
    Possible,
    Unknown,
}

impl Confidence {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Definite => "definite",
            Self::Likely => "likely",
            Self::Possible => "possible",
            Self::Unknown => "unknown",
        }
    }

    pub fn is_definite(self) -> bool {
        matches!(self, Self::Definite)
    }
}

// ---------------------------------------------------------------------------
// ClassificationResult
// ---------------------------------------------------------------------------

/// The outcome of classifying a single cycle.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub anomaly_class: AnomalyClass,
    pub cycle: CycleInfo,
    pub confidence: Confidence,
    pub explanation: String,
}

impl ClassificationResult {
    pub fn new(
        anomaly_class: AnomalyClass,
        cycle: CycleInfo,
        confidence: Confidence,
        explanation: String,
    ) -> Self {
        Self {
            anomaly_class,
            cycle,
            confidence,
            explanation,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper predicates (free functions)
// ---------------------------------------------------------------------------

/// G0 — Dirty Write: a cycle consisting exclusively of WriteWrite edges where
/// at least one participant has not committed (still active or aborted).
pub fn is_g0_cycle(
    cycle: &CycleInfo,
    committed: &HashSet<TransactionId>,
    _aborted: &HashSet<TransactionId>,
) -> bool {
    if cycle.edges.is_empty() {
        return false;
    }
    let all_ww = cycle
        .edges
        .iter()
        .all(|e| e.dep_type == DependencyType::WriteWrite);
    if !all_ww {
        return false;
    }
    cycle.nodes.iter().any(|t| !committed.contains(t))
}

/// G1a — Aborted Read: cycle contains a WriteRead edge whose *source*
/// transaction later aborted, meaning another txn read a value that was
/// rolled back.
pub fn is_g1a_cycle(
    cycle: &CycleInfo,
    _committed: &HashSet<TransactionId>,
    aborted: &HashSet<TransactionId>,
) -> bool {
    cycle.edges.iter().any(|e| {
        e.dep_type == DependencyType::WriteRead && aborted.contains(&e.from_txn)
    })
}

/// G1b — Intermediate Read: cycle contains a WriteRead dependency on an item
/// where the writing transaction later overwrote that value (the read saw a
/// non-final version). We approximate this as a WR edge on the *same item*
/// co-occurring with a WW edge from the same source transaction, meaning the
/// writer produced more than one version of that item.
pub fn is_g1b_cycle(
    cycle: &CycleInfo,
    committed: &HashSet<TransactionId>,
    aborted: &HashSet<TransactionId>,
) -> bool {
    let wr_items: Vec<(TransactionId, Option<ItemId>)> = cycle
        .edges
        .iter()
        .filter(|e| e.dep_type == DependencyType::WriteRead)
        .map(|e| (e.from_txn, e.item_id))
        .collect();

    if wr_items.is_empty() {
        return false;
    }

    for (from_txn, maybe_item) in &wr_items {
        let has_ww_same_source = cycle.edges.iter().any(|e| {
            e.dep_type == DependencyType::WriteWrite
                && e.from_txn == *from_txn
                && (maybe_item.is_none() || e.item_id == *maybe_item)
        });
        if has_ww_same_source {
            // The txn that wrote the intermediate value should itself be committed
            // (otherwise it is G1a instead).
            if committed.contains(from_txn) && !aborted.contains(from_txn) {
                return true;
            }
        }
    }
    false
}

/// G1c — Circular Information Flow: a cycle of length ≥ 2 among *committed*
/// transactions where every edge is either WriteRead or WriteWrite (no
/// anti-dependencies).
pub fn is_g1c_cycle(
    cycle: &CycleInfo,
    committed: &HashSet<TransactionId>,
    _aborted: &HashSet<TransactionId>,
) -> bool {
    if cycle.len() < 2 || cycle.edges.is_empty() {
        return false;
    }
    let all_committed = cycle.nodes.iter().all(|t| committed.contains(t));
    if !all_committed {
        return false;
    }
    cycle.edges.iter().all(|e| {
        matches!(
            e.dep_type,
            DependencyType::WriteRead | DependencyType::WriteWrite
        )
    })
}

/// G2-item: cycle with at least one item-level anti-dependency (ReadWrite)
/// and *no* predicate-level edges.
pub fn is_g2_item_cycle(cycle: &CycleInfo) -> bool {
    if cycle.edges.is_empty() {
        return false;
    }
    let has_rw = cycle
        .edges
        .iter()
        .any(|e| e.dep_type == DependencyType::ReadWrite);
    has_rw && cycle.all_item_level()
}

/// G2: cycle containing at least one predicate-level anti-dependency edge.
pub fn is_g2_cycle(cycle: &CycleInfo) -> bool {
    if cycle.edges.is_empty() {
        return false;
    }
    cycle.edges.iter().any(|e| {
        e.dep_type.is_predicate_level() && e.dep_type.is_anti_dependency()
    })
}

/// Recommend the minimum standard isolation level that prevents the given
/// anomaly class.
pub fn recommend_isolation_for(class: AnomalyClass) -> IsolationLevel {
    match class {
        AnomalyClass::G0 => IsolationLevel::ReadCommitted,
        AnomalyClass::G1a | AnomalyClass::G1b => IsolationLevel::ReadCommitted,
        AnomalyClass::G1c => IsolationLevel::RepeatableRead,
        AnomalyClass::G2Item => IsolationLevel::RepeatableRead,
        AnomalyClass::G2 => IsolationLevel::Serializable,
    }
}

// ---------------------------------------------------------------------------
// AnomalyClassifier
// ---------------------------------------------------------------------------

/// Stateless classifier that maps DSG cycles to Adya anomaly classes.
#[derive(Debug, Clone)]
pub struct AnomalyClassifier {
    /// Enables strict mode: only report Definite classifications.
    strict: bool,
}

impl Default for AnomalyClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl AnomalyClassifier {
    pub fn new() -> Self {
        Self { strict: false }
    }

    pub fn strict(mut self, yes: bool) -> Self {
        self.strict = yes;
        self
    }

    /// Classify a single cycle.  Returns `None` when the cycle does not match
    /// any known anomaly pattern.
    pub fn classify_cycle(
        &self,
        cycle: &CycleInfo,
        committed: &HashSet<TransactionId>,
        aborted: &HashSet<TransactionId>,
    ) -> Option<ClassificationResult> {
        if cycle.is_empty() {
            return None;
        }

        // Check anomalies in severity order (most severe first) so that the
        // *strongest* applicable classification wins.

        // --- G0: dirty write ---
        if is_g0_cycle(cycle, committed, aborted) {
            return Some(ClassificationResult::new(
                AnomalyClass::G0,
                cycle.clone(),
                Confidence::Definite,
                "Cycle consists entirely of write-write dependencies with an \
                 uncommitted participant — dirty write (G0)."
                    .into(),
            ));
        }

        // --- G1a: aborted read ---
        if is_g1a_cycle(cycle, committed, aborted) {
            return Some(ClassificationResult::new(
                AnomalyClass::G1a,
                cycle.clone(),
                Confidence::Definite,
                "A transaction read data written by a subsequently-aborted \
                 transaction — aborted read (G1a)."
                    .into(),
            ));
        }

        // --- G1b: intermediate read ---
        if is_g1b_cycle(cycle, committed, aborted) {
            let conf = if cycle.edges.iter().all(|e| e.item_id.is_some()) {
                Confidence::Definite
            } else {
                Confidence::Likely
            };
            return Some(ClassificationResult::new(
                AnomalyClass::G1b,
                cycle.clone(),
                conf,
                "A transaction read an intermediate (non-final) version of an \
                 item — intermediate read (G1b)."
                    .into(),
            ));
        }

        // --- G1c: circular information flow ---
        if is_g1c_cycle(cycle, committed, aborted) {
            return Some(ClassificationResult::new(
                AnomalyClass::G1c,
                cycle.clone(),
                Confidence::Definite,
                "Committed transactions form a write-dependency cycle with no \
                 anti-dependencies — circular information flow (G1c)."
                    .into(),
            ));
        }

        // --- G2-item ---
        if is_g2_item_cycle(cycle) {
            let conf = if cycle.nodes.iter().all(|t| committed.contains(t)) {
                Confidence::Definite
            } else {
                Confidence::Likely
            };
            if self.strict && !conf.is_definite() {
                return None;
            }
            return Some(ClassificationResult::new(
                AnomalyClass::G2Item,
                cycle.clone(),
                conf,
                "Cycle contains an item-level anti-dependency (read-write) — \
                 item anti-dependency cycle (G2-item)."
                    .into(),
            ));
        }

        // --- G2: predicate-level ---
        if is_g2_cycle(cycle) {
            let conf = if cycle.nodes.iter().all(|t| committed.contains(t)) {
                Confidence::Definite
            } else {
                Confidence::Likely
            };
            if self.strict && !conf.is_definite() {
                return None;
            }
            return Some(ClassificationResult::new(
                AnomalyClass::G2,
                cycle.clone(),
                conf,
                "Cycle contains a predicate-level anti-dependency — \
                 phantom anomaly (G2)."
                    .into(),
            ));
        }

        // Fall-through: unrecognised pattern.
        None
    }

    /// Classify every cycle in the slice, discarding those that match nothing.
    pub fn classify_all(
        &self,
        cycles: &[CycleInfo],
        committed: &HashSet<TransactionId>,
        aborted: &HashSet<TransactionId>,
    ) -> Vec<ClassificationResult> {
        cycles
            .iter()
            .filter_map(|c| self.classify_cycle(c, committed, aborted))
            .collect()
    }

    /// Convenience: severity of an anomaly class (delegates to the type).
    pub fn severity_for(&self, class: AnomalyClass) -> AnomalySeverity {
        class.severity()
    }
}

// ---------------------------------------------------------------------------
// SeverityAssessment
// ---------------------------------------------------------------------------

/// Aggregated severity report over a collection of classification results.
#[derive(Debug, Clone)]
pub struct SeverityAssessment {
    pub overall_severity: AnomalySeverity,
    pub anomaly_counts: HashMap<AnomalyClass, usize>,
    pub worst_anomaly: Option<AnomalyClass>,
    pub recommendations: Vec<String>,
}

impl SeverityAssessment {
    /// Build an assessment from a set of classification results.
    pub fn assess(results: &[ClassificationResult]) -> Self {
        let mut counts: HashMap<AnomalyClass, usize> = HashMap::new();
        for r in results {
            *counts.entry(r.anomaly_class).or_insert(0) += 1;
        }

        let worst_anomaly = counts
            .keys()
            .copied()
            .max_by_key(|c| severity_rank(c.severity()));

        let overall_severity = worst_anomaly
            .map(|a| a.severity())
            .unwrap_or(AnomalySeverity::Low);

        let mut recommendations: Vec<String> = Vec::new();
        if counts.is_empty() {
            recommendations.push(
                "No anomalies detected; the current isolation level appears sufficient."
                    .into(),
            );
        } else {
            let mut seen_levels: HashSet<String> = HashSet::new();
            for class in counts.keys() {
                let level = recommend_isolation_for(*class);
                let msg = format!(
                    "To prevent {} anomalies, use at least {} isolation.",
                    class.name(),
                    level
                );
                if seen_levels.insert(msg.clone()) {
                    recommendations.push(msg);
                }
            }
            recommendations.sort();
        }

        Self {
            overall_severity,
            anomaly_counts: counts,
            worst_anomaly,
            recommendations,
        }
    }
}

/// Map severity to a numeric rank for comparison.
fn severity_rank(s: AnomalySeverity) -> u8 {
    match s {
        AnomalySeverity::Low => 0,
        AnomalySeverity::Medium => 1,
        AnomalySeverity::High => 2,
        AnomalySeverity::Critical => 3,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use isospec_types::identifier::TransactionId;

    fn txn(id: u64) -> TransactionId {
        TransactionId::new(id)
    }

    fn committed_set(ids: &[u64]) -> HashSet<TransactionId> {
        ids.iter().copied().map(txn).collect()
    }

    fn aborted_set(ids: &[u64]) -> HashSet<TransactionId> {
        ids.iter().copied().map(txn).collect()
    }

    fn dep(from: u64, to: u64, dt: DependencyType) -> Dependency {
        Dependency::new(txn(from), txn(to), dt)
    }

    // ---- G0 ----

    #[test]
    fn classify_pure_ww_cycle_as_g0() {
        let cycle = CycleInfo::new(
            vec![txn(1), txn(2)],
            vec![
                dep(1, 2, DependencyType::WriteWrite),
                dep(2, 1, DependencyType::WriteWrite),
            ],
        );
        let committed = committed_set(&[1]); // txn 2 not committed
        let aborted = aborted_set(&[]);

        let classifier = AnomalyClassifier::new();
        let result = classifier.classify_cycle(&cycle, &committed, &aborted);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.anomaly_class, AnomalyClass::G0);
        assert!(r.confidence.is_definite());
    }

    // ---- G1a ----

    #[test]
    fn classify_wr_from_aborted_as_g1a() {
        let cycle = CycleInfo::new(
            vec![txn(1), txn(2)],
            vec![
                dep(1, 2, DependencyType::WriteRead),
                dep(2, 1, DependencyType::ReadWrite),
            ],
        );
        let committed = committed_set(&[2]);
        let aborted = aborted_set(&[1]);

        let classifier = AnomalyClassifier::new();
        let result = classifier
            .classify_cycle(&cycle, &committed, &aborted)
            .unwrap();
        assert_eq!(result.anomaly_class, AnomalyClass::G1a);
        assert!(result.confidence.is_definite());
    }

    // ---- G1c ----

    #[test]
    fn classify_circular_wr_as_g1c() {
        let cycle = CycleInfo::new(
            vec![txn(1), txn(2)],
            vec![
                dep(1, 2, DependencyType::WriteRead),
                dep(2, 1, DependencyType::WriteRead),
            ],
        );
        let committed = committed_set(&[1, 2]);
        let aborted = aborted_set(&[]);

        let result = AnomalyClassifier::new()
            .classify_cycle(&cycle, &committed, &aborted)
            .unwrap();
        assert_eq!(result.anomaly_class, AnomalyClass::G1c);
    }

    // ---- G2-item ----

    #[test]
    fn classify_rw_cycle_as_g2_item() {
        let cycle = CycleInfo::new(
            vec![txn(1), txn(2)],
            vec![
                dep(1, 2, DependencyType::ReadWrite),
                dep(2, 1, DependencyType::WriteRead),
            ],
        );
        let committed = committed_set(&[1, 2]);
        let aborted = aborted_set(&[]);

        let result = AnomalyClassifier::new()
            .classify_cycle(&cycle, &committed, &aborted)
            .unwrap();
        assert_eq!(result.anomaly_class, AnomalyClass::G2Item);
        assert!(result.confidence.is_definite());
    }

    // ---- G2 (predicate) ----

    #[test]
    fn classify_predicate_rw_as_g2() {
        let cycle = CycleInfo::new(
            vec![txn(1), txn(2)],
            vec![
                dep(1, 2, DependencyType::PredicateReadWrite),
                dep(2, 1, DependencyType::WriteRead),
            ],
        );
        let committed = committed_set(&[1, 2]);
        let aborted = aborted_set(&[]);

        let result = AnomalyClassifier::new()
            .classify_cycle(&cycle, &committed, &aborted)
            .unwrap();
        assert_eq!(result.anomaly_class, AnomalyClass::G2);
    }

    // ---- Severity assessment ----

    #[test]
    fn severity_assessment_aggregates_correctly() {
        let c1 = ClassificationResult::new(
            AnomalyClass::G0,
            CycleInfo::new(vec![txn(1), txn(2)], vec![]),
            Confidence::Definite,
            "g0".into(),
        );
        let c2 = ClassificationResult::new(
            AnomalyClass::G2Item,
            CycleInfo::new(vec![txn(3), txn(4)], vec![]),
            Confidence::Likely,
            "g2-item".into(),
        );
        let assessment = SeverityAssessment::assess(&[c1, c2]);
        assert_eq!(assessment.overall_severity, AnomalySeverity::Critical);
        assert_eq!(assessment.worst_anomaly, Some(AnomalyClass::G0));
        assert_eq!(*assessment.anomaly_counts.get(&AnomalyClass::G0).unwrap(), 1);
        assert_eq!(
            *assessment.anomaly_counts.get(&AnomalyClass::G2Item).unwrap(),
            1
        );
        assert!(!assessment.recommendations.is_empty());
    }

    // ---- Empty cycle ----

    #[test]
    fn empty_cycle_returns_none() {
        let cycle = CycleInfo::new(vec![], vec![]);
        let committed = committed_set(&[]);
        let aborted = aborted_set(&[]);

        let result = AnomalyClassifier::new().classify_cycle(&cycle, &committed, &aborted);
        assert!(result.is_none());
    }

    #[test]
    fn empty_results_give_low_severity() {
        let assessment = SeverityAssessment::assess(&[]);
        assert_eq!(assessment.overall_severity, AnomalySeverity::Low);
        assert!(assessment.worst_anomaly.is_none());
        assert_eq!(assessment.anomaly_counts.len(), 0);
        assert_eq!(assessment.recommendations.len(), 1);
    }

    // ---- Confidence levels ----

    #[test]
    fn confidence_accessors() {
        assert!(Confidence::Definite.is_definite());
        assert!(!Confidence::Likely.is_definite());
        assert!(!Confidence::Possible.is_definite());
        assert!(!Confidence::Unknown.is_definite());

        assert_eq!(Confidence::Definite.as_str(), "definite");
        assert_eq!(Confidence::Likely.as_str(), "likely");
        assert_eq!(Confidence::Possible.as_str(), "possible");
        assert_eq!(Confidence::Unknown.as_str(), "unknown");
    }

    // ---- classify_all ----

    #[test]
    fn classify_all_filters_unrecognised() {
        let good = CycleInfo::new(
            vec![txn(1), txn(2)],
            vec![
                dep(1, 2, DependencyType::WriteRead),
                dep(2, 1, DependencyType::WriteRead),
            ],
        );
        let bad = CycleInfo::new(vec![], vec![]);

        let committed = committed_set(&[1, 2]);
        let aborted = aborted_set(&[]);

        let results =
            AnomalyClassifier::new().classify_all(&[good, bad], &committed, &aborted);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].anomaly_class, AnomalyClass::G1c);
    }

    // ---- CycleInfo helpers ----

    #[test]
    fn cycle_info_edge_queries() {
        let cycle = CycleInfo::new(
            vec![txn(1), txn(2), txn(3)],
            vec![
                dep(1, 2, DependencyType::WriteRead),
                dep(2, 3, DependencyType::ReadWrite),
                dep(3, 1, DependencyType::WriteWrite),
            ],
        );

        assert_eq!(cycle.len(), 3);
        assert!(!cycle.is_empty());
        assert!(cycle.has_edge_type(DependencyType::WriteRead));
        assert!(cycle.has_edge_type(DependencyType::ReadWrite));
        assert!(!cycle.has_edge_type(DependencyType::PredicateReadWrite));
        assert!(cycle.contains_txn(txn(2)));
        assert!(!cycle.contains_txn(txn(99)));
        assert!(cycle.has_anti_dependency_edge());
        assert!(cycle.all_item_level());
        assert!(!cycle.has_predicate_edge());
    }

    // ---- recommend_isolation_for ----

    #[test]
    fn recommend_isolation_mapping() {
        assert_eq!(recommend_isolation_for(AnomalyClass::G0), IsolationLevel::ReadCommitted);
        assert_eq!(recommend_isolation_for(AnomalyClass::G1a), IsolationLevel::ReadCommitted);
        assert_eq!(recommend_isolation_for(AnomalyClass::G1b), IsolationLevel::ReadCommitted);
        assert_eq!(recommend_isolation_for(AnomalyClass::G1c), IsolationLevel::RepeatableRead);
        assert_eq!(recommend_isolation_for(AnomalyClass::G2Item), IsolationLevel::RepeatableRead);
        assert_eq!(recommend_isolation_for(AnomalyClass::G2), IsolationLevel::Serializable);
    }

    // ---- strict mode ----

    #[test]
    fn strict_mode_filters_non_definite() {
        let cycle = CycleInfo::new(
            vec![txn(1), txn(2)],
            vec![
                dep(1, 2, DependencyType::ReadWrite),
                dep(2, 1, DependencyType::WriteRead),
            ],
        );
        let committed = committed_set(&[1]); // txn 2 NOT committed
        let aborted = aborted_set(&[]);

        let strict_classifier = AnomalyClassifier::new().strict(true);
        let result = strict_classifier.classify_cycle(&cycle, &committed, &aborted);
        assert!(result.is_none(), "strict mode should discard non-Definite");

        let lenient = AnomalyClassifier::new();
        let result = lenient.classify_cycle(&cycle, &committed, &aborted);
        assert!(result.is_some());
        assert_eq!(result.unwrap().confidence, Confidence::Likely);
    }
}
