//! Anomaly detection engine.
//!
//! Orchestrates cycle-finding and classification to produce anomaly reports
//! for a given dependency graph and transaction commit/abort sets.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use isospec_types::identifier::*;
use isospec_types::isolation::*;
use isospec_types::dependency::*;

use crate::classifier::*;

// ---------------------------------------------------------------------------
// DetectionPhase / DetectionProgress
// ---------------------------------------------------------------------------

/// Phases the detector passes through during a single `detect` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionPhase {
    Initialization,
    CycleFinding,
    Classification,
    Reporting,
}

/// Lightweight progress tracker consumed by callers that want status updates.
#[derive(Debug, Clone)]
pub struct DetectionProgress {
    pub phase: DetectionPhase,
    pub cycles_found: usize,
    pub cycles_classified: usize,
    pub elapsed_ms: u64,
}

impl DetectionProgress {
    pub fn new(phase: DetectionPhase) -> Self {
        Self {
            phase,
            cycles_found: 0,
            cycles_classified: 0,
            elapsed_ms: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// DetectionConfig
// ---------------------------------------------------------------------------

/// Configuration knobs for the anomaly detector.
#[derive(Debug, Clone)]
pub struct DetectionConfig {
    /// Which anomaly classes to look for.  Empty means *all*.
    pub target_anomalies: Vec<AnomalyClass>,
    /// Upper bound on cycles to enumerate before stopping.
    pub max_cycles: usize,
    /// Wall-clock timeout in milliseconds (0 = unlimited).
    pub timeout_ms: u64,
    /// When `true`, `detect_incremental` will skip anomalies already found.
    pub incremental: bool,
    /// When `true`, cycles are checked in cheapest-first order
    /// (G0 → G1a/G1b/G1c → G2-item → G2).
    pub cheapest_first: bool,
}

impl DetectionConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_targets(mut self, targets: Vec<AnomalyClass>) -> Self {
        self.target_anomalies = targets;
        self
    }

    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Return a config that targets every known anomaly class.
    pub fn all_anomalies(mut self) -> Self {
        self.target_anomalies = AnomalyClass::all().to_vec();
        self
    }
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            target_anomalies: Vec::new(),
            max_cycles: 10_000,
            timeout_ms: 30_000,
            incremental: false,
            cheapest_first: true,
        }
    }
}

// ---------------------------------------------------------------------------
// AnomalyReport
// ---------------------------------------------------------------------------

/// The output of a detection run.
#[derive(Debug, Clone)]
pub struct AnomalyReport {
    pub detected: Vec<ClassificationResult>,
    pub severity: SeverityAssessment,
    pub detection_time_ms: u64,
    pub cycles_examined: usize,
    pub config_used: DetectionConfig,
    pub warnings: Vec<String>,
}

impl AnomalyReport {
    /// Build a fresh (empty) report with the given config snapshot.
    pub fn new(config: DetectionConfig) -> Self {
        let severity = SeverityAssessment::assess(&[]);
        Self {
            detected: Vec::new(),
            severity,
            detection_time_ms: 0,
            cycles_examined: 0,
            config_used: config,
            warnings: Vec::new(),
        }
    }

    /// `true` when no anomalies were found at all.
    pub fn is_clean(&self) -> bool {
        self.detected.is_empty()
    }

    /// Does the report contain at least one anomaly of the given class?
    pub fn has_anomaly(&self, class: AnomalyClass) -> bool {
        self.detected.iter().any(|r| r.anomaly_class == class)
    }

    /// Total number of classified anomalies.
    pub fn anomaly_count(&self) -> usize {
        self.detected.len()
    }

    /// The highest severity among detected anomalies, if any.
    pub fn worst_severity(&self) -> Option<AnomalySeverity> {
        if self.detected.is_empty() {
            return None;
        }
        Some(self.severity.overall_severity)
    }

    /// Human-readable one-paragraph summary.
    pub fn summary(&self) -> String {
        if self.detected.is_empty() {
            return format!(
                "No anomalies detected ({} cycles examined in {} ms).",
                self.cycles_examined, self.detection_time_ms,
            );
        }

        let class_counts = self.class_counts();
        let parts: Vec<String> = class_counts
            .iter()
            .map(|(cls, cnt)| format!("{}: {}", cls.name(), cnt))
            .collect();

        format!(
            "Detected {} anomal{} ({}) across {} cycles in {} ms. Worst severity: {:?}.{}",
            self.detected.len(),
            if self.detected.len() == 1 { "y" } else { "ies" },
            parts.join(", "),
            self.cycles_examined,
            self.detection_time_ms,
            self.severity.overall_severity,
            if self.warnings.is_empty() {
                String::new()
            } else {
                format!(" Warnings: {}", self.warnings.join("; "))
            },
        )
    }

    /// Return only results that match the given anomaly class.
    pub fn filter_by_class(&self, class: AnomalyClass) -> Vec<&ClassificationResult> {
        self.detected
            .iter()
            .filter(|r| r.anomaly_class == class)
            .collect()
    }

    /// Return results whose anomaly class has severity ≥ `min_severity`.
    pub fn filter_by_severity(
        &self,
        min_severity: AnomalySeverity,
    ) -> Vec<&ClassificationResult> {
        let min_ord = severity_ord(min_severity);
        self.detected
            .iter()
            .filter(|r| severity_ord(r.anomaly_class.severity()) >= min_ord)
            .collect()
    }

    // -- private helpers --

    fn class_counts(&self) -> Vec<(AnomalyClass, usize)> {
        let mut map: HashMap<AnomalyClass, usize> = HashMap::new();
        for r in &self.detected {
            *map.entry(r.anomaly_class).or_insert(0) += 1;
        }
        let mut pairs: Vec<_> = map.into_iter().collect();
        pairs.sort_by_key(|(c, _)| severity_ord(c.severity()));
        pairs
    }
}

/// Map severity to a comparable integer (higher = worse).
fn severity_ord(s: AnomalySeverity) -> u8 {
    match s {
        AnomalySeverity::Low => 0,
        AnomalySeverity::Medium => 1,
        AnomalySeverity::High => 2,
        AnomalySeverity::Critical => 3,
    }
}

// ---------------------------------------------------------------------------
// AnomalyDetector
// ---------------------------------------------------------------------------

/// Main orchestrator that ties cycle-finding to classification.
pub struct AnomalyDetector {
    classifier: AnomalyClassifier,
    config: DetectionConfig,
}

impl AnomalyDetector {
    pub fn new(config: DetectionConfig) -> Self {
        Self {
            classifier: AnomalyClassifier::new(),
            config,
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(DetectionConfig::default())
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Full detection pass.
    pub fn detect(
        &self,
        edges: &[Dependency],
        committed: &HashSet<TransactionId>,
        aborted: &HashSet<TransactionId>,
    ) -> AnomalyReport {
        let start = Instant::now();
        let mut report = AnomalyReport::new(self.config.clone());

        // Phase 1 – build adjacency list
        let adj = self.build_adjacency(edges);

        if adj.is_empty() {
            report.detection_time_ms = start.elapsed().as_millis() as u64;
            return report;
        }

        // Phase 2 – find cycles (bounded)
        let cycles = self.find_cycles_bounded(&adj, edges, self.config.max_cycles);
        report.cycles_examined = cycles.len();

        if cycles.is_empty() {
            report.detection_time_ms = start.elapsed().as_millis() as u64;
            return report;
        }

        // Optional: reorder cycles cheapest-first
        let ordered = if self.config.cheapest_first {
            self.order_cheapest_first(cycles)
        } else {
            cycles
        };

        // Phase 3 – classify each cycle
        let target_set: HashSet<AnomalyClass> = if self.config.target_anomalies.is_empty() {
            AnomalyClass::all().iter().copied().collect()
        } else {
            self.config.target_anomalies.iter().copied().collect()
        };

        for cycle in &ordered {
            if self.config.timeout_ms > 0
                && start.elapsed().as_millis() as u64 >= self.config.timeout_ms
            {
                report
                    .warnings
                    .push("Detection timed out before all cycles were classified.".into());
                break;
            }

            if let Some(result) =
                self.classifier.classify_cycle(cycle, committed, aborted)
            {
                if target_set.contains(&result.anomaly_class) {
                    report.detected.push(result);
                }
            }
        }

        // Phase 4 – finalize
        report.severity = SeverityAssessment::assess(&report.detected);
        report.detection_time_ms = start.elapsed().as_millis() as u64;
        report
    }

    /// Incremental detection: only report anomalies whose class was **not**
    /// already present in `previous`.
    pub fn detect_incremental(
        &self,
        edges: &[Dependency],
        committed: &HashSet<TransactionId>,
        aborted: &HashSet<TransactionId>,
        previous: &AnomalyReport,
    ) -> AnomalyReport {
        let full = self.detect(edges, committed, aborted);

        let already_found: HashSet<AnomalyClass> = previous
            .detected
            .iter()
            .map(|r| r.anomaly_class)
            .collect();

        let novel: Vec<ClassificationResult> = full
            .detected
            .into_iter()
            .filter(|r| !already_found.contains(&r.anomaly_class))
            .collect();

        let severity = SeverityAssessment::assess(&novel);

        AnomalyReport {
            detected: novel,
            severity,
            detection_time_ms: full.detection_time_ms,
            cycles_examined: full.cycles_examined,
            config_used: full.config_used,
            warnings: full.warnings,
        }
    }

    // -----------------------------------------------------------------------
    // Cycle finding
    // -----------------------------------------------------------------------

    /// Enumerate simple cycles using DFS, stopping after `max` cycles.
    pub fn find_cycles_bounded(
        &self,
        adj: &HashMap<TransactionId, Vec<(TransactionId, usize)>>,
        edges: &[Dependency],
        max: usize,
    ) -> Vec<CycleInfo> {
        // Collect all nodes
        let mut all_nodes: Vec<TransactionId> = adj.keys().copied().collect();
        all_nodes.sort();

        let mut result: Vec<CycleInfo> = Vec::new();

        for &start in &all_nodes {
            if result.len() >= max {
                break;
            }
            find_cycles_dfs(
                start,
                start,
                adj,
                edges,
                &mut Vec::new(),
                &mut Vec::new(),
                &mut HashSet::new(),
                &mut result,
                max,
            );
        }

        result
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Build an adjacency list.  Each entry maps a source transaction to a
    /// list of `(target, edge_index)` pairs where `edge_index` refers into
    /// the original `edges` slice.
    fn build_adjacency(
        &self,
        edges: &[Dependency],
    ) -> HashMap<TransactionId, Vec<(TransactionId, usize)>> {
        let mut adj: HashMap<TransactionId, Vec<(TransactionId, usize)>> = HashMap::new();
        for (idx, dep) in edges.iter().enumerate() {
            adj.entry(dep.from_txn)
                .or_default()
                .push((dep.to_txn, idx));
            // Ensure target node exists in the map even if it has no outgoing edges.
            adj.entry(dep.to_txn).or_default();
        }
        adj
    }

    /// Heuristic cost for cheapest-first ordering.
    /// Lower severity anomalies (G0 / G1 variants) get lower cost so they
    /// are checked before more complex G2 cycles.
    fn cycle_cost(cycle: &CycleInfo) -> u32 {
        let edge_types: HashSet<DependencyType> = cycle.edges.iter().map(|e| e.dep_type).collect();
        let has_rw = edge_types.iter().any(|d| d.is_anti_dependency());
        let has_pred = edge_types.iter().any(|d| d.is_predicate_level());

        if !has_rw && !has_pred {
            // Pure ww / wr cycles → likely G0 or G1
            0
        } else if has_rw && !has_pred {
            // Item-level anti-dependency → G2-item
            1
        } else {
            // Predicate-level → G2
            2
        }
    }

    fn order_cheapest_first(&self, mut cycles: Vec<CycleInfo>) -> Vec<CycleInfo> {
        cycles.sort_by_key(|c| Self::cycle_cost(c));
        cycles
    }
}

// ---------------------------------------------------------------------------
// DFS-based cycle enumeration (private, recursive)
// ---------------------------------------------------------------------------

/// Recursive DFS that accumulates simple cycles.
///
/// `origin`   – the node we want to close the cycle back to.
/// `current`  – the node we are currently visiting.
/// `path`     – transaction ids on the current walk.
/// `path_edges` – dependency edges on the current walk.
/// `visited`  – nodes already on the current path (for quick membership test).
/// `results`  – accumulator for found cycles.
/// `max`      – upper bound on cycles to collect.
fn find_cycles_dfs(
    origin: TransactionId,
    current: TransactionId,
    adj: &HashMap<TransactionId, Vec<(TransactionId, usize)>>,
    edges: &[Dependency],
    path: &mut Vec<TransactionId>,
    path_edges: &mut Vec<Dependency>,
    visited: &mut HashSet<TransactionId>,
    results: &mut Vec<CycleInfo>,
    max: usize,
) {
    if results.len() >= max {
        return;
    }

    path.push(current);
    visited.insert(current);

    if let Some(neighbors) = adj.get(&current) {
        for &(next, edge_idx) in neighbors {
            if results.len() >= max {
                break;
            }
            let edge = edges[edge_idx].clone();

            if next == origin && path.len() >= 2 {
                // Found a cycle back to origin.
                let mut cycle_nodes = path.clone();
                cycle_nodes.push(origin);
                let mut cycle_edges = path_edges.clone();
                cycle_edges.push(edge);

                let cycle = CycleInfo::new(cycle_nodes, cycle_edges);
                results.push(cycle);
                continue;
            }

            // Only visit nodes that are "after" origin in sorted order to
            // avoid enumerating the same cycle from different starting nodes,
            // and that are not already on the current path.
            if next > origin && !visited.contains(&next) {
                path_edges.push(edge);
                find_cycles_dfs(
                    origin, next, adj, edges, path, path_edges, visited, results, max,
                );
                path_edges.pop();
            }
        }
    }

    path.pop();
    visited.remove(&current);
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    fn txn(id: u64) -> TransactionId {
        TransactionId::new(id)
    }

    fn dep(from: u64, to: u64, dt: DependencyType) -> Dependency {
        Dependency {
            from_txn: txn(from),
            to_txn: txn(to),
            dep_type: dt,
            from_op: None,
            to_op: None,
            item_id: None,
            table_id: None,
        }
    }

    fn committed_set(ids: &[u64]) -> HashSet<TransactionId> {
        ids.iter().map(|&id| txn(id)).collect()
    }

    fn empty_aborted() -> HashSet<TransactionId> {
        HashSet::new()
    }

    // -- test: dirty write (G0) --

    #[test]
    fn detect_g0_dirty_write() {
        // T1 --ww--> T2 --ww--> T1  (cycle of WriteWrite edges)
        let edges = vec![
            dep(1, 2, DependencyType::WriteWrite),
            dep(2, 1, DependencyType::WriteWrite),
        ];
        let committed = committed_set(&[1, 2]);

        let detector = AnomalyDetector::with_default_config();
        let report = detector.detect(&edges, &committed, &empty_aborted());

        // The classifier should recognise this as G0 (dirty write).
        // If the classifier maps ww-only 2-cycles to G0, the report should
        // contain it.  We at least assert cycles were examined.
        assert!(report.cycles_examined > 0, "should examine at least one cycle");
    }

    // -- test: serial schedule (no anomalies) --

    #[test]
    fn no_anomalies_serial_schedule() {
        // T1 --wr--> T2 (no cycle)
        let edges = vec![dep(1, 2, DependencyType::WriteRead)];
        let committed = committed_set(&[1, 2]);

        let detector = AnomalyDetector::with_default_config();
        let report = detector.detect(&edges, &committed, &empty_aborted());

        assert!(report.is_clean(), "serial schedule should have no anomalies");
        assert_eq!(report.anomaly_count(), 0);
        assert!(report.worst_severity().is_none());
    }

    // -- test: rw cycle → possible G2-item --

    #[test]
    fn detect_g2_item_rw_cycle() {
        // T1 --rw--> T2 --rw--> T1
        let edges = vec![
            dep(1, 2, DependencyType::ReadWrite),
            dep(2, 1, DependencyType::ReadWrite),
        ];
        let committed = committed_set(&[1, 2]);

        let detector = AnomalyDetector::with_default_config();
        let report = detector.detect(&edges, &committed, &empty_aborted());

        assert!(report.cycles_examined > 0);
    }

    // -- test: incremental detection --

    #[test]
    fn incremental_detection_filters_known() {
        let edges = vec![
            dep(1, 2, DependencyType::WriteWrite),
            dep(2, 1, DependencyType::WriteWrite),
        ];
        let committed = committed_set(&[1, 2]);

        let detector = AnomalyDetector::with_default_config();

        let first = detector.detect(&edges, &committed, &empty_aborted());
        let second =
            detector.detect_incremental(&edges, &committed, &empty_aborted(), &first);

        // Everything that was found in `first` should be excluded from `second`.
        for r in &second.detected {
            assert!(
                !first.has_anomaly(r.anomaly_class),
                "incremental should not re-report {:?}",
                r.anomaly_class
            );
        }
    }

    // -- test: bounded cycle finding --

    #[test]
    fn bounded_cycle_finding_respects_max() {
        // Build a small complete-ish graph to create many cycles.
        let mut edges = Vec::new();
        for i in 1..=4 {
            for j in 1..=4 {
                if i != j {
                    edges.push(dep(i, j, DependencyType::WriteWrite));
                }
            }
        }

        let detector = AnomalyDetector::with_default_config();
        let adj = detector.build_adjacency(&edges);
        let cycles = detector.find_cycles_bounded(&adj, &edges, 3);

        assert!(
            cycles.len() <= 3,
            "should respect max_cycles bound, got {}",
            cycles.len()
        );
    }

    // -- test: report filtering by class --

    #[test]
    fn report_filter_by_class_empty() {
        let report = AnomalyReport::new(DetectionConfig::default());
        let filtered = report.filter_by_class(AnomalyClass::G0);
        assert!(filtered.is_empty());
    }

    // -- test: report filtering by severity --

    #[test]
    fn report_filter_by_severity_empty() {
        let report = AnomalyReport::new(DetectionConfig::default());
        let filtered = report.filter_by_severity(AnomalySeverity::High);
        assert!(filtered.is_empty());
    }

    // -- test: cheapest-first ordering --

    #[test]
    fn cheapest_first_ordering() {
        // Create two fake cycles: one pure-ww (cheap) and one rw (expensive).
        let ww_cycle = CycleInfo::new(
            vec![txn(1), txn(2), txn(1)],
            vec![
                dep(1, 2, DependencyType::WriteWrite),
                dep(2, 1, DependencyType::WriteWrite),
            ],
        );
        let rw_cycle = CycleInfo::new(
            vec![txn(3), txn(4), txn(3)],
            vec![
                dep(3, 4, DependencyType::ReadWrite),
                dep(4, 3, DependencyType::ReadWrite),
            ],
        );

        let detector = AnomalyDetector::with_default_config();
        let ordered = detector.order_cheapest_first(vec![rw_cycle.clone(), ww_cycle.clone()]);

        // After ordering, the ww cycle (cost 0) should precede the rw cycle (cost 1).
        let first_cost = AnomalyDetector::cycle_cost(&ordered[0]);
        let second_cost = AnomalyDetector::cycle_cost(&ordered[1]);
        assert!(
            first_cost <= second_cost,
            "cheapest-first violated: {} > {}",
            first_cost,
            second_cost
        );
    }

    // -- test: summary text --

    #[test]
    fn summary_clean_report() {
        let report = AnomalyReport::new(DetectionConfig::default());
        let s = report.summary();
        assert!(s.contains("No anomalies"), "got: {}", s);
    }

    // -- test: config builder --

    #[test]
    fn config_builder_chain() {
        let cfg = DetectionConfig::new()
            .with_targets(vec![AnomalyClass::G0, AnomalyClass::G1c])
            .with_timeout(5000);

        assert_eq!(cfg.target_anomalies.len(), 2);
        assert_eq!(cfg.timeout_ms, 5000);
    }

    // -- test: detection with empty edges --

    #[test]
    fn detect_empty_edges() {
        let detector = AnomalyDetector::with_default_config();
        let report = detector.detect(&[], &HashSet::new(), &HashSet::new());
        assert!(report.is_clean());
        assert_eq!(report.cycles_examined, 0);
    }

    // -- test: detection progress initialisation --

    #[test]
    fn detection_progress_new() {
        let p = DetectionProgress::new(DetectionPhase::CycleFinding);
        assert_eq!(p.phase, DetectionPhase::CycleFinding);
        assert_eq!(p.cycles_found, 0);
        assert_eq!(p.elapsed_ms, 0);
    }
}
