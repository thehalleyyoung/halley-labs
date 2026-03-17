//! Post-generation verification — checks that generated code satisfies WCET
//! bounds, preserves perceptual soundness (Theorem 4), and computes
//! quantization error bounds.

use crate::{
    codegen::GeneratedRenderer,
    scheduler::Schedule,
    wcet::{WcetAnalyzer, WcetBudget},
    CgGraph, CodegenConfig, CodegenError, CodegenResult, NodeKind,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// RendererVerifier
// ---------------------------------------------------------------------------

/// Verifies that a generated renderer satisfies WCET bounds.
#[derive(Debug, Clone)]
pub struct RendererVerifier {
    pub config: CodegenConfig,
}

/// Verification result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether all checks passed.
    pub passed: bool,
    /// Individual check results.
    pub checks: Vec<CheckResult>,
    /// Summary message.
    pub summary: String,
}

/// A single verification check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub name: String,
    pub passed: bool,
    pub message: String,
    pub severity: CheckSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckSeverity {
    Info,
    Warning,
    Error,
}

impl RendererVerifier {
    pub fn new(config: &CodegenConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Run all verification checks.
    pub fn verify(
        &self,
        graph: &CgGraph,
        renderer: &GeneratedRenderer,
    ) -> VerificationResult {
        let mut checks = Vec::new();

        checks.push(self.check_wcet_bounds(graph));
        checks.push(self.check_buffer_allocation(renderer));
        checks.push(self.check_processing_order(graph, renderer));
        checks.push(self.check_all_nodes_covered(graph, renderer));
        checks.push(self.check_utilization(graph));

        let passed = checks.iter().all(|c| c.passed || c.severity != CheckSeverity::Error);
        let failures: Vec<&str> = checks
            .iter()
            .filter(|c| !c.passed && c.severity == CheckSeverity::Error)
            .map(|c| c.name.as_str())
            .collect();
        let summary = if passed {
            "All verification checks passed.".into()
        } else {
            format!("Verification failed: {}", failures.join(", "))
        };

        VerificationResult {
            passed,
            checks,
            summary,
        }
    }

    fn check_wcet_bounds(&self, graph: &CgGraph) -> CheckResult {
        let analyzer = WcetAnalyzer::new(&self.config);
        let budget = analyzer.compute_budget(graph);

        if budget.passes {
            CheckResult {
                name: "WCET bounds".into(),
                passed: true,
                message: format!(
                    "WCET {:.0} cycles within budget {:.0} cycles (margin {:.1}x)",
                    budget.critical_path_cycles,
                    budget.total_budget_cycles,
                    budget.safety_margin
                ),
                severity: CheckSeverity::Info,
            }
        } else {
            CheckResult {
                name: "WCET bounds".into(),
                passed: false,
                message: format!(
                    "WCET {:.0} cycles exceeds safe budget (margin {:.1}x < required {:.1}x)",
                    budget.critical_path_cycles,
                    budget.safety_margin,
                    self.config.target_safety_margin
                ),
                severity: CheckSeverity::Error,
            }
        }
    }

    fn check_buffer_allocation(&self, renderer: &GeneratedRenderer) -> CheckResult {
        let buf_count = renderer.buffer_plan.buffer_count;
        let mem = renderer.buffer_plan.total_memory_bytes;

        // Warn if memory exceeds 1MB
        let max_mem = 1_048_576;
        if mem > max_mem {
            CheckResult {
                name: "Buffer allocation".into(),
                passed: false,
                message: format!(
                    "{} buffers using {} bytes (exceeds {} byte limit)",
                    buf_count, mem, max_mem
                ),
                severity: CheckSeverity::Warning,
            }
        } else {
            CheckResult {
                name: "Buffer allocation".into(),
                passed: true,
                message: format!("{} buffers, {} bytes total", buf_count, mem),
                severity: CheckSeverity::Info,
            }
        }
    }

    fn check_processing_order(&self, graph: &CgGraph, renderer: &GeneratedRenderer) -> CheckResult {
        // Verify that for each edge (src, dst), src appears before dst.
        let order_map: HashMap<u64, usize> = renderer
            .processing_order
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        for edge in &graph.edges {
            if let (Some(&src_pos), Some(&dst_pos)) = (
                order_map.get(&edge.source_node),
                order_map.get(&edge.dest_node),
            ) {
                if src_pos >= dst_pos {
                    return CheckResult {
                        name: "Processing order".into(),
                        passed: false,
                        message: format!(
                            "Node {} (pos {}) must precede node {} (pos {})",
                            edge.source_node, src_pos, edge.dest_node, dst_pos
                        ),
                        severity: CheckSeverity::Error,
                    };
                }
            }
        }

        CheckResult {
            name: "Processing order".into(),
            passed: true,
            message: "Topological order valid".into(),
            severity: CheckSeverity::Info,
        }
    }

    fn check_all_nodes_covered(&self, graph: &CgGraph, renderer: &GeneratedRenderer) -> CheckResult {
        let rendered_ids: std::collections::HashSet<u64> =
            renderer.processing_order.iter().copied().collect();
        let graph_ids: std::collections::HashSet<u64> =
            graph.nodes.iter().map(|n| n.id).collect();

        let missing: Vec<u64> = graph_ids.difference(&rendered_ids).copied().collect();
        if missing.is_empty() {
            CheckResult {
                name: "Node coverage".into(),
                passed: true,
                message: format!("All {} nodes covered", graph.nodes.len()),
                severity: CheckSeverity::Info,
            }
        } else {
            CheckResult {
                name: "Node coverage".into(),
                passed: false,
                message: format!("Missing nodes: {:?}", missing),
                severity: CheckSeverity::Error,
            }
        }
    }

    fn check_utilization(&self, graph: &CgGraph) -> CheckResult {
        let analyzer = WcetAnalyzer::new(&self.config);
        let budget = analyzer.compute_budget(graph);

        if budget.utilization <= self.config.max_utilization {
            CheckResult {
                name: "CPU utilization".into(),
                passed: true,
                message: format!(
                    "Utilization {:.1}% <= max {:.1}%",
                    budget.utilization * 100.0,
                    self.config.max_utilization * 100.0
                ),
                severity: CheckSeverity::Info,
            }
        } else {
            CheckResult {
                name: "CPU utilization".into(),
                passed: false,
                message: format!(
                    "Utilization {:.1}% > max {:.1}%",
                    budget.utilization * 100.0,
                    self.config.max_utilization * 100.0
                ),
                severity: CheckSeverity::Error,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SoundnessChecker — Theorem 4
// ---------------------------------------------------------------------------

/// Verifies Theorem 4 (Perceptual Soundness): the codegen-introduced
/// quantization errors stay below perceptual thresholds.
#[derive(Debug, Clone)]
pub struct SoundnessChecker {
    pub config: CodegenConfig,
}

/// Quantization error analysis for a single parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationError {
    pub parameter_name: String,
    pub node_id: u64,
    /// Maximum absolute error δ introduced by quantization.
    pub max_delta: f64,
    /// Perceptual threshold (JND) for this parameter.
    pub perceptual_threshold: f64,
    /// Whether δ < threshold (soundness preserved).
    pub within_threshold: bool,
}

/// Result of soundness checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoundnessResult {
    pub passed: bool,
    pub errors: Vec<QuantizationError>,
    /// Worst-case δ/threshold ratio across all parameters.
    pub worst_ratio: f64,
    pub summary: String,
}

impl SoundnessChecker {
    pub fn new(config: &CodegenConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Check perceptual soundness of the generated renderer against the
    /// original graph constraints.
    pub fn check(&self, graph: &CgGraph) -> SoundnessResult {
        let mut errors = Vec::new();

        for node in &graph.nodes {
            let node_errors = self.analyze_node_quantization(node);
            errors.extend(node_errors);
        }

        let worst_ratio = errors
            .iter()
            .map(|e| {
                if e.perceptual_threshold > 0.0 {
                    e.max_delta / e.perceptual_threshold
                } else {
                    0.0
                }
            })
            .fold(0.0_f64, f64::max);

        let passed = errors.iter().all(|e| e.within_threshold);
        let summary = if passed {
            format!(
                "Soundness preserved: worst δ/JND ratio = {:.4}",
                worst_ratio
            )
        } else {
            let violations: Vec<String> = errors
                .iter()
                .filter(|e| !e.within_threshold)
                .map(|e| {
                    format!(
                        "{}@node{}: δ={:.6} > JND={:.6}",
                        e.parameter_name, e.node_id, e.max_delta, e.perceptual_threshold
                    )
                })
                .collect();
            format!("Soundness violated: {}", violations.join("; "))
        };

        SoundnessResult {
            passed,
            errors,
            worst_ratio,
            summary,
        }
    }

    /// Analyze quantization errors for a single node's parameters.
    fn analyze_node_quantization(&self, node: &crate::NodeInfo) -> Vec<QuantizationError> {
        let mut errors = Vec::new();

        for (param_name, &value) in &node.parameters {
            let (max_delta, threshold) = self.compute_delta_and_threshold(
                node.kind,
                param_name,
                value,
                node.sample_rate,
            );

            errors.push(QuantizationError {
                parameter_name: param_name.clone(),
                node_id: node.id,
                max_delta,
                perceptual_threshold: threshold,
                within_threshold: max_delta <= threshold,
            });
        }

        errors
    }

    /// Compute the maximum quantization error δ and the perceptual JND
    /// threshold for a given parameter.
    fn compute_delta_and_threshold(
        &self,
        kind: NodeKind,
        param_name: &str,
        value: f64,
        sample_rate: f64,
    ) -> (f64, f64) {
        // δ: worst-case error from f64→f32 quantization and block processing.
        // For f32, relative precision is ~1e-7.
        let f32_relative_error = 1.2e-7_f64;
        let max_delta = value.abs() * f32_relative_error;

        // JND thresholds based on psychoacoustic literature.
        let threshold = match (kind, param_name) {
            (NodeKind::Oscillator, "frequency") => {
                // Frequency JND: ~0.3% for pure tones (Sek & Moore)
                value.abs() * 0.003
            }
            (NodeKind::Filter, "cutoff") => {
                // Cutoff JND: ~1-2% depending on bandwidth
                value.abs() * 0.01
            }
            (NodeKind::Filter, "q") => {
                // Q JND: ~10% of value
                value.abs() * 0.1
            }
            (NodeKind::Gain, "level") | (NodeKind::Envelope, "sustain") => {
                // Amplitude JND: ~1 dB ≈ 0.122 linear
                0.122
            }
            (NodeKind::Pan, "position") => {
                // Minimum audible angle: ~1° ≈ 0.011 in normalized [-1,1]
                0.011
            }
            (NodeKind::Delay, "samples") => {
                // Temporal JND: ~2-5ms
                sample_rate * 0.002
            }
            (NodeKind::Envelope, "attack") | (NodeKind::Envelope, "decay") | (NodeKind::Envelope, "release") => {
                // Envelope timing JND: ~10ms
                0.01
            }
            (NodeKind::Compressor, "threshold") => {
                // Compression threshold JND: ~1dB
                1.0
            }
            (NodeKind::Compressor, "ratio") => {
                // Ratio JND: ~0.5
                0.5
            }
            _ => {
                // Default: 1% of value or 0.001 absolute
                (value.abs() * 0.01).max(0.001)
            }
        };

        (max_delta, threshold)
    }

    /// Compute the aggregate soundness margin: the minimum (JND - δ) / JND
    /// across all parameters.
    pub fn soundness_margin(&self, graph: &CgGraph) -> f64 {
        let result = self.check(graph);
        if result.errors.is_empty() {
            return 1.0;
        }
        result
            .errors
            .iter()
            .map(|e| {
                if e.perceptual_threshold > 0.0 {
                    1.0 - (e.max_delta / e.perceptual_threshold)
                } else {
                    1.0
                }
            })
            .fold(f64::INFINITY, f64::min)
    }
}

// ---------------------------------------------------------------------------
// BenchmarkHarness
// ---------------------------------------------------------------------------

/// Benchmark harness for measuring actual execution times and comparing
/// against WCET estimates.
#[derive(Debug, Clone)]
pub struct BenchmarkHarness {
    pub config: CodegenConfig,
}

/// Benchmark measurement for a single node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMeasurement {
    pub node_id: u64,
    pub node_name: String,
    pub kind: NodeKind,
    pub wcet_estimate_cycles: f64,
    pub measured_cycles: f64,
    pub ratio: f64,
}

/// Benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub measurements: Vec<BenchmarkMeasurement>,
    pub total_wcet_estimate: f64,
    pub total_measured: f64,
    pub overall_ratio: f64,
    pub worst_node_ratio: f64,
}

impl BenchmarkHarness {
    pub fn new(config: &CodegenConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Simulate benchmark measurements using estimated costs.
    /// In a real implementation this would run actual timed iterations.
    pub fn benchmark(&self, graph: &CgGraph) -> BenchmarkReport {
        let analyzer = WcetAnalyzer::new(&self.config);
        let node_wcets = analyzer.analyze_nodes(graph);

        let measurements: Vec<BenchmarkMeasurement> = node_wcets
            .iter()
            .map(|w| {
                // Simulated: actual measurements would be 30-80% of WCET estimate.
                let simulated_ratio = match w.kind {
                    NodeKind::Oscillator => 0.6,
                    NodeKind::Filter => 0.5,
                    NodeKind::Envelope => 0.4,
                    NodeKind::Mixer => 0.7,
                    NodeKind::Delay => 0.65,
                    NodeKind::Compressor => 0.55,
                    _ => 0.5,
                };
                let measured = w.per_block_cycles * simulated_ratio;
                BenchmarkMeasurement {
                    node_id: w.node_id,
                    node_name: w.node_name.clone(),
                    kind: w.kind,
                    wcet_estimate_cycles: w.per_block_cycles,
                    measured_cycles: measured,
                    ratio: simulated_ratio,
                }
            })
            .collect();

        let total_wcet: f64 = measurements.iter().map(|m| m.wcet_estimate_cycles).sum();
        let total_measured: f64 = measurements.iter().map(|m| m.measured_cycles).sum();
        let overall_ratio = if total_wcet > 0.0 {
            total_measured / total_wcet
        } else {
            0.0
        };
        let worst_node_ratio = measurements
            .iter()
            .map(|m| m.ratio)
            .fold(0.0_f64, f64::max);

        BenchmarkReport {
            measurements,
            total_wcet_estimate: total_wcet,
            total_measured,
            overall_ratio,
            worst_node_ratio,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BufferKind, CgGraphBuilder, CodegenConfig, NodeKind};
    use std::collections::HashMap;

    fn test_config() -> CodegenConfig {
        CodegenConfig::default()
    }

    fn simple_chain() -> CgGraph {
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let osc = b.add_node("osc", NodeKind::Oscillator);
        let filt = b.add_node("filt", NodeKind::Filter);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc, filt, BufferKind::Audio);
        b.connect(filt, out, BufferKind::Audio);
        b.build()
    }

    fn simple_renderer() -> GeneratedRenderer {
        let cfg = test_config();
        let gen = crate::codegen::CodeGenerator::new(&cfg);
        let graph = simple_chain();
        gen.generate_from_graph(&graph).unwrap()
    }

    #[test]
    fn test_renderer_verifier_passes() {
        let cfg = test_config();
        let verifier = RendererVerifier::new(&cfg);
        let graph = simple_chain();
        let renderer = simple_renderer();
        let result = verifier.verify(&graph, &renderer);
        assert!(result.passed);
    }

    #[test]
    fn test_renderer_verifier_checks_count() {
        let cfg = test_config();
        let verifier = RendererVerifier::new(&cfg);
        let graph = simple_chain();
        let renderer = simple_renderer();
        let result = verifier.verify(&graph, &renderer);
        assert!(result.checks.len() >= 4);
    }

    #[test]
    fn test_wcet_check_fails_slow_cpu() {
        let cfg = CodegenConfig {
            cpu_frequency_hz: 100.0,
            target_safety_margin: 1000.0,
            ..test_config()
        };
        let verifier = RendererVerifier::new(&cfg);
        let graph = simple_chain();
        let renderer = simple_renderer();
        let result = verifier.verify(&graph, &renderer);
        let wcet_check = result.checks.iter().find(|c| c.name == "WCET bounds").unwrap();
        assert!(!wcet_check.passed);
    }

    #[test]
    fn test_soundness_checker_passes() {
        let cfg = test_config();
        let checker = SoundnessChecker::new(&cfg);
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let mut params = HashMap::new();
        params.insert("frequency".into(), 440.0);
        let osc = b.add_node_with_params("osc", NodeKind::Oscillator, params);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc, out, BufferKind::Audio);
        let graph = b.build();
        let result = checker.check(&graph);
        assert!(result.passed);
    }

    #[test]
    fn test_soundness_checker_quantization_errors() {
        let cfg = test_config();
        let checker = SoundnessChecker::new(&cfg);
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let mut params = HashMap::new();
        params.insert("frequency".into(), 440.0);
        params.insert("amplitude".into(), 0.8);
        let osc = b.add_node_with_params("osc", NodeKind::Oscillator, params.clone());
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc, out, BufferKind::Audio);
        let graph = b.build();
        let result = checker.check(&graph);
        assert!(!result.errors.is_empty());
        // All should be within threshold for normal values
        for e in &result.errors {
            assert!(
                e.within_threshold,
                "{} delta {} > threshold {}",
                e.parameter_name, e.max_delta, e.perceptual_threshold
            );
        }
    }

    #[test]
    fn test_soundness_margin() {
        let cfg = test_config();
        let checker = SoundnessChecker::new(&cfg);
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let mut params = HashMap::new();
        params.insert("frequency".into(), 440.0);
        let osc = b.add_node_with_params("osc", NodeKind::Oscillator, params);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc, out, BufferKind::Audio);
        let graph = b.build();
        let margin = checker.soundness_margin(&graph);
        assert!(margin > 0.9); // f64 precision should give >99.9% margin
    }

    #[test]
    fn test_benchmark_harness() {
        let cfg = test_config();
        let harness = BenchmarkHarness::new(&cfg);
        let graph = simple_chain();
        let report = harness.benchmark(&graph);
        assert_eq!(report.measurements.len(), 3);
        assert!(report.total_measured > 0.0);
        assert!(report.overall_ratio > 0.0 && report.overall_ratio < 1.0);
    }

    #[test]
    fn test_benchmark_worst_ratio() {
        let cfg = test_config();
        let harness = BenchmarkHarness::new(&cfg);
        let graph = simple_chain();
        let report = harness.benchmark(&graph);
        assert!(report.worst_node_ratio <= 1.0);
    }

    #[test]
    fn test_soundness_multiple_node_types() {
        let cfg = test_config();
        let checker = SoundnessChecker::new(&cfg);
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let mut osc_p = HashMap::new();
        osc_p.insert("frequency".into(), 440.0);
        let osc = b.add_node_with_params("osc", NodeKind::Oscillator, osc_p);
        let mut filt_p = HashMap::new();
        filt_p.insert("cutoff".into(), 1000.0);
        filt_p.insert("q".into(), 0.707);
        let filt = b.add_node_with_params("filt", NodeKind::Filter, filt_p);
        let mut gain_p = HashMap::new();
        gain_p.insert("level".into(), 0.5);
        let gain = b.add_node_with_params("gain", NodeKind::Gain, gain_p);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc, filt, BufferKind::Audio);
        b.connect(filt, gain, BufferKind::Audio);
        b.connect(gain, out, BufferKind::Audio);
        let graph = b.build();
        let result = checker.check(&graph);
        assert!(result.passed);
        assert!(result.errors.len() >= 3);
    }

    #[test]
    fn test_verification_summary() {
        let cfg = test_config();
        let verifier = RendererVerifier::new(&cfg);
        let graph = simple_chain();
        let renderer = simple_renderer();
        let result = verifier.verify(&graph, &renderer);
        assert!(!result.summary.is_empty());
    }
}
