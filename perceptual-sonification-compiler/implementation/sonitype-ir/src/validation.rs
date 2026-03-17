//! IR validation.
//!
//! - [`IrValidator`] – comprehensive validation (type checks, connectivity,
//!   parameter ranges, sample-rate consistency, WCET bounds).
//! - [`ValidationReport`] – structured list of warnings and errors.
//! - [`WcetValidator`] – verify the total WCET fits within the buffer period.

use crate::graph::{AudioGraph, NodeId, NodeType, PortDirection};

// ---------------------------------------------------------------------------
// ValidationReport
// ---------------------------------------------------------------------------

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValidationSeverity {
    Info,
    Warning,
    Error,
}

/// A single validation finding.
#[derive(Debug, Clone)]
pub struct ValidationEntry {
    pub severity: ValidationSeverity,
    pub node: Option<NodeId>,
    pub message: String,
}

/// Aggregated validation report.
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    pub entries: Vec<ValidationEntry>,
}

impl ValidationReport {
    pub fn new() -> Self { Self::default() }

    pub fn push(&mut self, severity: ValidationSeverity, node: Option<NodeId>, message: impl Into<String>) {
        self.entries.push(ValidationEntry { severity, node, message: message.into() });
    }

    pub fn error_count(&self) -> usize {
        self.entries.iter().filter(|e| e.severity == ValidationSeverity::Error).count()
    }

    pub fn warning_count(&self) -> usize {
        self.entries.iter().filter(|e| e.severity == ValidationSeverity::Warning).count()
    }

    pub fn is_valid(&self) -> bool { self.error_count() == 0 }

    pub fn errors(&self) -> Vec<&ValidationEntry> {
        self.entries.iter().filter(|e| e.severity == ValidationSeverity::Error).collect()
    }

    pub fn warnings(&self) -> Vec<&ValidationEntry> {
        self.entries.iter().filter(|e| e.severity == ValidationSeverity::Warning).collect()
    }

    pub fn merge(&mut self, other: &ValidationReport) {
        self.entries.extend(other.entries.iter().cloned());
    }

    pub fn summary(&self) -> String {
        format!(
            "validation: {} error(s), {} warning(s), {} info(s)",
            self.error_count(),
            self.warning_count(),
            self.entries.iter().filter(|e| e.severity == ValidationSeverity::Info).count(),
        )
    }
}

// ---------------------------------------------------------------------------
// IrValidator
// ---------------------------------------------------------------------------

/// Comprehensive IR validator.
pub struct IrValidator;

impl IrValidator {
    /// Run all validation checks.
    pub fn validate(graph: &mut AudioGraph) -> ValidationReport {
        let mut report = ValidationReport::new();
        Self::check_empty(graph, &mut report);
        Self::check_cycles(graph, &mut report);
        Self::check_edge_types(graph, &mut report);
        Self::check_topological_order(graph, &mut report);
        Self::check_unconnected_inputs(graph, &mut report);
        Self::check_parameter_ranges(graph, &mut report);
        Self::check_sample_rate_consistency(graph, &mut report);
        Self::check_duplicate_edges(graph, &mut report);
        Self::check_port_directions(graph, &mut report);
        report
    }

    /// Check if graph is empty.
    fn check_empty(graph: &AudioGraph, report: &mut ValidationReport) {
        if graph.is_empty() {
            report.push(ValidationSeverity::Warning, None, "graph is empty");
        }
    }

    /// Check for cycles.
    fn check_cycles(graph: &AudioGraph, report: &mut ValidationReport) {
        if graph.has_cycle() {
            report.push(ValidationSeverity::Error, None, "graph contains a cycle");
        }
    }

    /// Type-check all edges: source and dest port data types must match.
    fn check_edge_types(graph: &AudioGraph, report: &mut ValidationReport) {
        for edge in &graph.edges {
            let src_type = graph.node(edge.source_node)
                .and_then(|n| n.port_by_id(edge.source_port))
                .map(|p| p.data_type);
            let dst_type = graph.node(edge.dest_node)
                .and_then(|n| n.port_by_id(edge.dest_port))
                .map(|p| p.data_type);

            match (src_type, dst_type) {
                (Some(s), Some(d)) if s != d => {
                    report.push(
                        ValidationSeverity::Error,
                        Some(edge.source_node),
                        format!(
                            "edge {}: port type mismatch {:?} → {:?} (nodes {} → {})",
                            edge.id.0, s, d, edge.source_node.0, edge.dest_node.0,
                        ),
                    );
                }
                (None, _) => {
                    report.push(
                        ValidationSeverity::Error,
                        Some(edge.source_node),
                        format!("edge {}: source port {} not found on node {}", edge.id.0, edge.source_port.0, edge.source_node.0),
                    );
                }
                (_, None) => {
                    report.push(
                        ValidationSeverity::Error,
                        Some(edge.dest_node),
                        format!("edge {}: dest port {} not found on node {}", edge.id.0, edge.dest_port.0, edge.dest_node.0),
                    );
                }
                _ => {}
            }
        }
    }

    /// Verify the topological order, if computed, is valid.
    fn check_topological_order(graph: &mut AudioGraph, report: &mut ValidationReport) {
        if graph.topological_order.is_empty() { return; }
        // Every edge must go from an earlier position to a later position.
        let mut pos: std::collections::HashMap<NodeId, usize> = std::collections::HashMap::new();
        for (i, &nid) in graph.topological_order.iter().enumerate() {
            pos.insert(nid, i);
        }
        for edge in &graph.edges {
            let sp = pos.get(&edge.source_node);
            let dp = pos.get(&edge.dest_node);
            match (sp, dp) {
                (Some(&s), Some(&d)) if s >= d => {
                    report.push(
                        ValidationSeverity::Error,
                        Some(edge.source_node),
                        format!("topological order violated: {} (pos {}) → {} (pos {})", edge.source_node.0, s, edge.dest_node.0, d),
                    );
                }
                (None, _) => {
                    report.push(ValidationSeverity::Warning, Some(edge.source_node),
                        format!("node {} in edge but not in topological order", edge.source_node.0));
                }
                (_, None) => {
                    report.push(ValidationSeverity::Warning, Some(edge.dest_node),
                        format!("node {} in edge but not in topological order", edge.dest_node.0));
                }
                _ => {}
            }
        }
    }

    /// Check for unconnected required inputs.
    fn check_unconnected_inputs(graph: &AudioGraph, report: &mut ValidationReport) {
        for node in &graph.nodes {
            for port in &node.inputs {
                if port.required {
                    let connected = graph.edges.iter().any(|e| e.dest_node == node.id && e.dest_port == port.id);
                    if !connected {
                        report.push(
                            ValidationSeverity::Error,
                            Some(node.id),
                            format!("required input '{}' on node '{}' ({}) is not connected", port.name, node.name, node.id.0),
                        );
                    }
                }
            }
        }
    }

    /// Validate parameter ranges for all nodes.
    fn check_parameter_ranges(graph: &AudioGraph, report: &mut ValidationReport) {
        for node in &graph.nodes {
            let errs = node.parameters.validate();
            for e in errs {
                report.push(ValidationSeverity::Error, Some(node.id), format!("parameter: {}", e));
            }
            // Also check NodeType-level implicit params.
            match &node.node_type {
                NodeType::Oscillator { frequency, .. } if *frequency < 0.0 || *frequency > 22050.0 => {
                    report.push(ValidationSeverity::Error, Some(node.id),
                        format!("oscillator frequency {} out of range [0, 22050]", frequency));
                }
                NodeType::Filter { cutoff, q, .. } => {
                    if *cutoff < 20.0 || *cutoff > 22050.0 {
                        report.push(ValidationSeverity::Error, Some(node.id),
                            format!("filter cutoff {} out of range [20, 22050]", cutoff));
                    }
                    if *q < 0.1 || *q > 30.0 {
                        report.push(ValidationSeverity::Warning, Some(node.id),
                            format!("filter Q {} outside recommended range [0.1, 30]", q));
                    }
                }
                NodeType::Gain { level } if *level < -100.0 || *level > 100.0 => {
                    report.push(ValidationSeverity::Warning, Some(node.id),
                        format!("gain level {} outside recommended range [-100, 100]", level));
                }
                NodeType::Pan { position } if *position < -1.0 || *position > 1.0 => {
                    report.push(ValidationSeverity::Error, Some(node.id),
                        format!("pan position {} out of range [-1, 1]", position));
                }
                NodeType::Envelope { attack, decay, sustain, release } => {
                    if *attack < 0.0 { report.push(ValidationSeverity::Error, Some(node.id), "negative attack time".to_string()); }
                    if *decay < 0.0 { report.push(ValidationSeverity::Error, Some(node.id), "negative decay time".to_string()); }
                    if *sustain < 0.0 || *sustain > 1.0 { report.push(ValidationSeverity::Error, Some(node.id), format!("sustain {} out of [0,1]", sustain)); }
                    if *release < 0.0 { report.push(ValidationSeverity::Error, Some(node.id), "negative release time".to_string()); }
                }
                _ => {}
            }
        }
    }

    /// Check sample rate consistency across all nodes.
    fn check_sample_rate_consistency(graph: &AudioGraph, report: &mut ValidationReport) {
        let expected = graph.sample_rate;
        for node in &graph.nodes {
            if (node.sample_rate - expected).abs() > 0.1 {
                report.push(
                    ValidationSeverity::Error,
                    Some(node.id),
                    format!("sample rate mismatch: node '{}' has {} Hz, graph expects {} Hz",
                        node.name, node.sample_rate, expected),
                );
            }
        }
    }

    /// Check for duplicate edges (same source port → same dest port).
    fn check_duplicate_edges(graph: &AudioGraph, report: &mut ValidationReport) {
        let mut seen = std::collections::HashSet::new();
        for edge in &graph.edges {
            let key = (edge.source_node, edge.source_port, edge.dest_node, edge.dest_port);
            if !seen.insert(key) {
                report.push(
                    ValidationSeverity::Warning,
                    Some(edge.source_node),
                    format!("duplicate edge from {}:{} → {}:{}", edge.source_node.0, edge.source_port.0, edge.dest_node.0, edge.dest_port.0),
                );
            }
        }
    }

    /// Ensure edge port directions are correct (source = Output, dest = Input).
    fn check_port_directions(graph: &AudioGraph, report: &mut ValidationReport) {
        for edge in &graph.edges {
            if let Some(src_node) = graph.node(edge.source_node) {
                if let Some(port) = src_node.port_by_id(edge.source_port) {
                    if port.direction != PortDirection::Output {
                        report.push(ValidationSeverity::Error, Some(edge.source_node),
                            format!("edge {} source port '{}' is not an Output", edge.id.0, port.name));
                    }
                }
            }
            if let Some(dst_node) = graph.node(edge.dest_node) {
                if let Some(port) = dst_node.port_by_id(edge.dest_port) {
                    if port.direction != PortDirection::Input {
                        report.push(ValidationSeverity::Error, Some(edge.dest_node),
                            format!("edge {} dest port '{}' is not an Input", edge.id.0, port.name));
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// WcetValidator
// ---------------------------------------------------------------------------

/// Verify total WCET fits in the audio callback budget.
pub struct WcetValidator {
    /// Safety margin factor (1.0 = exact budget, 0.8 = 80% of budget).
    pub margin: f64,
}

impl Default for WcetValidator {
    fn default() -> Self { Self { margin: 0.8 } }
}

impl WcetValidator {
    pub fn new(margin: f64) -> Self { Self { margin } }

    /// Compute the budget (microseconds) for one audio callback.
    pub fn budget_us(sample_rate: f64, block_size: usize) -> f64 {
        (block_size as f64 / sample_rate) * 1_000_000.0
    }

    /// Validate that the graph's critical-path WCET fits in the budget.
    pub fn validate(&self, graph: &mut AudioGraph) -> ValidationReport {
        let mut report = ValidationReport::new();
        let _ = graph.ensure_sorted();

        let budget = Self::budget_us(graph.sample_rate, graph.block_size);
        let allowed = budget * self.margin;
        let critical = graph.critical_path_wcet();
        let total = graph.total_wcet();

        report.push(
            ValidationSeverity::Info,
            None,
            format!("budget: {:.1}us (margin {:.0}%), critical path WCET: {:.1}us, total WCET: {:.1}us",
                budget, self.margin * 100.0, critical, total),
        );

        if critical > allowed {
            report.push(
                ValidationSeverity::Error,
                None,
                format!("critical path WCET ({:.1}us) exceeds budget ({:.1}us, margin {:.0}%)",
                    critical, allowed, self.margin * 100.0),
            );
        }

        // Per-node WCET warnings.
        for node in &graph.nodes {
            if node.wcet_estimate_us > allowed * 0.5 {
                report.push(
                    ValidationSeverity::Warning,
                    Some(node.id),
                    format!("node '{}' WCET ({:.1}us) is > 50% of budget ({:.1}us)",
                        node.name, node.wcet_estimate_us, allowed),
                );
            }
        }

        report
    }

    /// Quick boolean check.
    pub fn is_within_budget(&self, graph: &mut AudioGraph) -> bool {
        self.validate(graph).is_valid()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{AudioGraph, NodeType, GraphBuilder};
    use crate::node::{Waveform, FilterType};

    fn valid_graph() -> AudioGraph {
        let (b, osc) = GraphBuilder::with_defaults()
            .add_oscillator("osc", Waveform::Sine, 440.0);
        let (b, g) = b.add_gain("gain", 0.5);
        let (b, out) = b.add_output("out", "stereo");
        b.connect(osc, "out", g, "in")
         .connect(g, "out", out, "in")
         .build().unwrap()
    }

    #[test]
    fn test_valid_graph_passes() {
        let mut g = valid_graph();
        let report = IrValidator::validate(&mut g);
        assert!(report.is_valid(), "errors: {:?}", report.errors());
    }

    #[test]
    fn test_empty_graph_warning() {
        let mut g = AudioGraph::default();
        let report = IrValidator::validate(&mut g);
        assert!(report.warning_count() > 0);
    }

    #[test]
    fn test_unconnected_required_input() {
        let mut g = AudioGraph::default();
        g.add_node("filt", NodeType::Filter { filter_type: FilterType::LowPass, cutoff: 1000.0, q: 1.0 });
        g.add_node("out", NodeType::Output { format: "mono".into() });
        let report = IrValidator::validate(&mut g);
        assert!(report.error_count() > 0);
    }

    #[test]
    fn test_parameter_range_invalid_frequency() {
        let mut g = AudioGraph::default();
        g.add_node("osc", NodeType::Oscillator { waveform: Waveform::Sine, frequency: 99999.0 });
        let report = IrValidator::validate(&mut g);
        assert!(report.error_count() > 0);
    }

    #[test]
    fn test_parameter_range_invalid_pan() {
        let mut g = AudioGraph::default();
        g.add_node("pan", NodeType::Pan { position: 2.0 });
        let report = IrValidator::validate(&mut g);
        assert!(report.error_count() > 0);
    }

    #[test]
    fn test_sample_rate_mismatch() {
        let mut g = AudioGraph::new(48000.0, 256);
        let id = g.add_node("osc", NodeType::Oscillator { waveform: Waveform::Sine, frequency: 440.0 });
        g.node_mut(id).unwrap().sample_rate = 44100.0;
        let report = IrValidator::validate(&mut g);
        assert!(report.error_count() > 0);
    }

    #[test]
    fn test_wcet_validator_within_budget() {
        let mut g = valid_graph();
        let wv = WcetValidator::default();
        assert!(wv.is_within_budget(&mut g));
    }

    #[test]
    fn test_wcet_validator_budget_computation() {
        let budget = WcetValidator::budget_us(48000.0, 256);
        let expected = (256.0 / 48000.0) * 1e6;
        assert!((budget - expected).abs() < 0.1);
    }

    #[test]
    fn test_wcet_validator_over_budget() {
        let mut g = AudioGraph::new(48000.0, 256);
        let id = g.add_node("heavy", NodeType::TimeStretch);
        // Manually set a huge WCET.
        g.node_mut(id).unwrap().wcet_estimate_us = 100_000.0;
        let out = g.add_node("out", NodeType::Output { format: "mono".into() });
        g.add_edge_by_name(id, "out", out, "in").unwrap();
        let wv = WcetValidator::new(0.8);
        assert!(!wv.is_within_budget(&mut g));
    }

    #[test]
    fn test_validation_report_merge() {
        let mut a = ValidationReport::new();
        a.push(ValidationSeverity::Error, None, "err1");
        let mut b = ValidationReport::new();
        b.push(ValidationSeverity::Warning, None, "warn1");
        a.merge(&b);
        assert_eq!(a.entries.len(), 2);
    }

    #[test]
    fn test_validation_report_summary() {
        let mut r = ValidationReport::new();
        r.push(ValidationSeverity::Error, None, "e");
        r.push(ValidationSeverity::Warning, None, "w");
        r.push(ValidationSeverity::Info, None, "i");
        let s = r.summary();
        assert!(s.contains("1 error(s)"));
        assert!(s.contains("1 warning(s)"));
    }

    #[test]
    fn test_envelope_negative_attack() {
        let mut g = AudioGraph::default();
        g.add_node("env", NodeType::Envelope { attack: -1.0, decay: 0.1, sustain: 0.7, release: 0.3 });
        let report = IrValidator::validate(&mut g);
        assert!(report.error_count() > 0);
    }

    #[test]
    fn test_port_direction_check() {
        let mut g = valid_graph();
        // On a valid graph built by the builder, all directions should be correct.
        let report = IrValidator::validate(&mut g);
        let dir_errors: Vec<_> = report.entries.iter()
            .filter(|e| e.message.contains("not an"))
            .collect();
        assert!(dir_errors.is_empty());
    }

    #[test]
    fn test_wcet_validator_report_has_info() {
        let mut g = valid_graph();
        let wv = WcetValidator::default();
        let report = wv.validate(&mut g);
        let infos: Vec<_> = report.entries.iter()
            .filter(|e| e.severity == ValidationSeverity::Info)
            .collect();
        assert!(!infos.is_empty());
    }
}
