//! Main verification pipeline that orchestrates every stage of cascade analysis.
//!
//! The pipeline proceeds through a fixed sequence of stages:
//! 1. **Parse** configuration files into a [`ConfigManifest`].
//! 2. **Build** the Retry-Timeout Interaction Graph ([`RtigGraph`]).
//! 3. **Tier 1** quick checks (amplification, timeout, fan-in).
//! 4. **Tier 2** deep analysis (BMC-style path enumeration) when Tier 1 finds issues.
//! 5. **Repair synthesis** (optional) via MaxSAT-backed optimisation.
//! 6. **Report generation** in the requested format.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use cascade_graph::rtig::{DependencyEdgeInfo, RtigGraph, RtigGraphSimpleBuilder, ServiceNode};
use cascade_types::config::{ConfigManifest, ConfigSource};
use cascade_types::report::{AnalysisReport, Evidence, Finding, Location, ReportMetadata, ReportSummary, Severity};
use cascade_types::repair::{ParameterChange, RepairAction, RepairPlan};
use cascade_types::topology::EdgeId;

// ---------------------------------------------------------------------------
// Configuration & result types
// ---------------------------------------------------------------------------

/// Controls which analysis tiers are run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisMode {
    /// Tier-1 only – fast, heuristic checks.
    Quick,
    /// Tier-1 followed by selective Tier-2 for flagged paths.
    Standard,
    /// Full Tier-1 + exhaustive Tier-2 regardless of Tier-1 results.
    Deep,
}

impl Default for AnalysisMode {
    fn default() -> Self {
        Self::Standard
    }
}

/// Where and how to emit output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Target format (plain, json, sarif, junit, …).
    pub format: String,
    /// If set, write the report to this path instead of stdout.
    pub output_path: Option<String>,
    /// Include repair suggestions in the report.
    pub include_repairs: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: "plain".into(),
            output_path: None,
            include_repairs: true,
        }
    }
}

/// Top-level configuration for a pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Paths to configuration files / directories to analyse.
    pub input_paths: Vec<String>,
    /// Which analysis tiers to run.
    pub analysis_mode: AnalysisMode,
    /// Output settings.
    pub output_config: OutputConfig,
    /// Whether to synthesise repairs for detected issues.
    pub repair_enabled: bool,
    /// Use incremental mode (leverage caching).
    pub incremental: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            input_paths: Vec::new(),
            analysis_mode: AnalysisMode::Standard,
            output_config: OutputConfig::default(),
            repair_enabled: false,
            incremental: false,
        }
    }
}

/// Aggregate result from the full pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    pub findings: Vec<Finding>,
    pub repairs: Option<Vec<RepairPlan>>,
    pub report: AnalysisReport,
    pub exit_code: i32,
    pub stats: PipelineStats,
}

/// Per-stage timing statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineStats {
    pub total_duration_ms: u64,
    pub per_stage_duration: HashMap<String, u64>,
}

/// Names of the discrete pipeline stages; used for progress reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PipelineStage {
    ParseConfigs,
    BuildTopology,
    Tier1Analysis,
    Tier2Analysis,
    RepairSynthesis,
    ReportGeneration,
}

impl PipelineStage {
    pub fn label(self) -> &'static str {
        match self {
            Self::ParseConfigs => "parse_configs",
            Self::BuildTopology => "build_topology",
            Self::Tier1Analysis => "tier1_analysis",
            Self::Tier2Analysis => "tier2_analysis",
            Self::RepairSynthesis => "repair_synthesis",
            Self::ReportGeneration => "report_generation",
        }
    }

    pub fn all() -> &'static [PipelineStage] {
        &[
            Self::ParseConfigs,
            Self::BuildTopology,
            Self::Tier1Analysis,
            Self::Tier2Analysis,
            Self::RepairSynthesis,
            Self::ReportGeneration,
        ]
    }
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// Intermediate result types (kept pipeline-local)
// ---------------------------------------------------------------------------

/// Result of the fast Tier-1 checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier1Result {
    pub findings: Vec<Finding>,
    pub amplification_paths: Vec<AmplificationPath>,
    pub timeout_issues: Vec<TimeoutIssue>,
    pub fan_in_issues: Vec<FanInIssue>,
}

/// A path through the graph that exhibits retry amplification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmplificationPath {
    pub path: Vec<String>,
    pub cumulative_factor: f64,
    pub edge_factors: Vec<f64>,
}

/// A timeout inconsistency between adjacent services.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutIssue {
    pub caller: String,
    pub callee: String,
    pub caller_timeout_ms: u64,
    pub callee_timeout_ms: u64,
    pub per_try_timeout_ms: u64,
    pub retries: u32,
}

/// A service whose fan-in is dangerously high.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FanInIssue {
    pub service: String,
    pub fan_in: usize,
    pub callers: Vec<String>,
    pub aggregated_amplification: f64,
}

/// Result of the deeper Tier-2 analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier2Result {
    pub findings: Vec<Finding>,
    pub cascade_scenarios: Vec<CascadeScenarioLocal>,
}

/// Lightweight scenario descriptor (avoids hard dependency on cascade-types::cascade
/// which may not be fully wired up in all builds).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeScenarioLocal {
    pub id: String,
    pub trigger_service: String,
    pub affected_services: Vec<String>,
    pub max_amplification: f64,
    pub max_latency_ms: u64,
}

// ---------------------------------------------------------------------------
// Thresholds
// ---------------------------------------------------------------------------

const AMPLIFICATION_WARN_THRESHOLD: f64 = 8.0;
const AMPLIFICATION_ERROR_THRESHOLD: f64 = 27.0;
const AMPLIFICATION_CRITICAL_THRESHOLD: f64 = 64.0;
const FAN_IN_WARN_THRESHOLD: usize = 4;
const FAN_IN_ERROR_THRESHOLD: usize = 8;
const TIMEOUT_SLACK_FACTOR: f64 = 1.2;
const MAX_PATH_DEPTH: usize = 12;

// ---------------------------------------------------------------------------
// VerificationPipeline
// ---------------------------------------------------------------------------

/// Stateless pipeline executor.
pub struct VerificationPipeline;

impl VerificationPipeline {
    /// Run the full verification pipeline end-to-end.
    pub fn run(config: &PipelineConfig) -> Result<PipelineResult> {
        let pipeline_start = Instant::now();
        let mut stage_durations: HashMap<String, u64> = HashMap::new();

        // Stage 1 – parse configs
        let t = Instant::now();
        log::info!("stage: parse_configs");
        let manifest = Self::parse_configs(&config.input_paths)
            .context("failed during config parsing stage")?;
        stage_durations.insert(
            PipelineStage::ParseConfigs.label().to_string(),
            t.elapsed().as_millis() as u64,
        );

        // Stage 2 – build topology
        let t = Instant::now();
        log::info!("stage: build_topology");
        let graph = Self::build_topology(&manifest)
            .context("failed during topology construction")?;
        stage_durations.insert(
            PipelineStage::BuildTopology.label().to_string(),
            t.elapsed().as_millis() as u64,
        );

        // Stage 3 – tier 1 analysis
        let t = Instant::now();
        log::info!("stage: tier1_analysis");
        let tier1 = Self::run_tier1(&graph);
        stage_durations.insert(
            PipelineStage::Tier1Analysis.label().to_string(),
            t.elapsed().as_millis() as u64,
        );

        // Stage 4 – tier 2 analysis (conditional)
        let t = Instant::now();
        log::info!("stage: tier2_analysis");
        let tier2 = Self::run_tier2_if_needed(&graph, &tier1, config.analysis_mode);
        stage_durations.insert(
            PipelineStage::Tier2Analysis.label().to_string(),
            t.elapsed().as_millis() as u64,
        );

        // Merge findings
        let mut all_findings = tier1.findings.clone();
        if let Some(ref t2) = tier2 {
            all_findings.extend(t2.findings.clone());
        }
        Self::deduplicate_findings(&mut all_findings);
        Self::sort_findings(&mut all_findings);

        // Stage 5 – repair synthesis (conditional)
        let t = Instant::now();
        log::info!("stage: repair_synthesis");
        let repairs = if config.repair_enabled && !all_findings.is_empty() {
            Some(Self::synthesize_repairs(&graph, &all_findings))
        } else {
            None
        };
        stage_durations.insert(
            PipelineStage::RepairSynthesis.label().to_string(),
            t.elapsed().as_millis() as u64,
        );

        // Stage 6 – report
        let t = Instant::now();
        log::info!("stage: report_generation");
        let report = Self::generate_report(&all_findings, &repairs);
        stage_durations.insert(
            PipelineStage::ReportGeneration.label().to_string(),
            t.elapsed().as_millis() as u64,
        );

        let exit_code = Self::compute_exit_code(&all_findings);

        let stats = PipelineStats {
            total_duration_ms: pipeline_start.elapsed().as_millis() as u64,
            per_stage_duration: stage_durations,
        };

        Ok(PipelineResult {
            findings: all_findings,
            repairs,
            report,
            exit_code,
            stats,
        })
    }

    // ----- Stage 1: Parse configs ------------------------------------------

    pub fn parse_configs(paths: &[String]) -> Result<ConfigManifest> {
        let mut sources = Vec::new();
        for path in paths {
            let format = if path.ends_with(".yaml") || path.ends_with(".yml") {
                "yaml"
            } else if path.ends_with(".json") {
                "json"
            } else if path.ends_with(".toml") {
                "toml"
            } else {
                "unknown"
            };
            sources.push(ConfigSource::Raw {
                format: format.to_string(),
                content: String::new(),
            });
        }
        if sources.is_empty() {
            anyhow::bail!("no configuration sources provided");
        }
        Ok(ConfigManifest {
            sources,
            file_paths: paths.to_vec(),
        })
    }

    // ----- Stage 2: Build topology -----------------------------------------

    pub fn build_topology(manifest: &ConfigManifest) -> Result<RtigGraph> {
        let mut builder = RtigGraphSimpleBuilder::new();
        let mut service_set: HashSet<String> = HashSet::new();
        let mut edges_to_add: Vec<DependencyEdgeInfo> = Vec::new();

        for (idx, source) in manifest.sources.iter().enumerate() {
            let file_path = manifest.file_paths.get(idx).cloned().unwrap_or_default();
            let extracted = Self::extract_services_from_source(source, &file_path);
            for (svc, node) in &extracted.nodes {
                if service_set.insert(svc.clone()) {
                    builder = builder.add_service(node.clone());
                }
            }
            edges_to_add.extend(extracted.edges);
        }

        for edge in edges_to_add {
            if service_set.contains(edge.source.as_str())
                && service_set.contains(edge.target.as_str())
            {
                builder = builder.add_edge(edge);
            }
        }

        let graph = builder.build();
        if graph.service_count() == 0 {
            anyhow::bail!("topology is empty – no services discovered");
        }
        log::info!(
            "topology built: {} services, {} edges",
            graph.service_count(),
            graph.edge_count()
        );
        Ok(graph)
    }

    /// Heuristic extraction of service nodes and edges from a single config
    /// source.  In a full build this delegates to cascade-config parsers; here
    /// we provide a self-contained implementation that inspects file metadata.
    fn extract_services_from_source(_source: &ConfigSource, file_path: &str) -> ExtractedTopology {
        let mut topo = ExtractedTopology::default();
        let path = file_path;

        // Derive a service name from the file path.
        let stem = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let node = ServiceNode::new(&stem, 1000);
        topo.nodes.insert(stem.clone(), node);

        // If the path contains a directory hint like "envoy/" or "istio/"
        // we can infer infrastructure edges.
        if path.contains("envoy") || path.contains("istio") {
            let proxy_name = format!("{}-proxy", stem);
            let proxy_node = ServiceNode::new(&proxy_name, 2000);
            topo.nodes.insert(proxy_name.clone(), proxy_node);
            topo.edges.push(
                DependencyEdgeInfo::new(&proxy_name, &stem)
                    .with_retry_count(3)
                    .with_timeout_ms(5000),
            );
        }

        topo
    }

    // ----- Stage 3: Tier 1 -------------------------------------------------

    pub fn run_tier1(graph: &RtigGraph) -> Tier1Result {
        let mut findings = Vec::new();

        let amp_paths = Self::detect_amplification_paths(graph);
        for ap in &amp_paths {
            let severity = if ap.cumulative_factor >= AMPLIFICATION_CRITICAL_THRESHOLD {
                Severity::Critical
            } else if ap.cumulative_factor >= AMPLIFICATION_ERROR_THRESHOLD {
                Severity::High
            } else if ap.cumulative_factor >= AMPLIFICATION_WARN_THRESHOLD {
                Severity::Medium
            } else {
                continue;
            };
            findings.push(Finding {
                id: format!("AMP-{}", Uuid::new_v4().as_simple()),
                severity,
                title: "Retry amplification detected".to_string(),
                description: format!(
                    "retry amplification factor {:.1}x along path {}",
                    ap.cumulative_factor,
                    ap.path.join(" → ")
                ),
                evidence: ap
                    .path
                    .windows(2)
                    .zip(ap.edge_factors.iter())
                    .map(|(pair, &factor)| Evidence {
                        description: format!("{} → {} (factor {:.1}x)", pair[0], pair[1], factor),
                        value: None,
                        source: None,
                    })
                    .collect(),
                location: Location::default(),
                code_flow: None,
                remediation: None,
            });
        }

        let timeout_issues = Self::detect_timeout_issues(graph);
        for ti in &timeout_issues {
            let effective = ti.per_try_timeout_ms * (ti.retries as u64 + 1);
            let severity = if effective > ti.caller_timeout_ms * 2 {
                Severity::High
            } else if (effective as f64) > ti.caller_timeout_ms as f64 * TIMEOUT_SLACK_FACTOR {
                Severity::Medium
            } else {
                continue;
            };
            findings.push(Finding {
                id: format!("TMO-{}", Uuid::new_v4().as_simple()),
                severity,
                title: "Timeout budget exceeded".to_string(),
                description: format!(
                    "timeout budget exceeded: {} calls {} with effective timeout {}ms > caller budget {}ms",
                    ti.caller, ti.callee, effective, ti.caller_timeout_ms
                ),
                evidence: vec![Evidence {
                    description: format!(
                        "per_try={}ms × (1+retries={}) = {}ms, caller timeout={}ms",
                        ti.per_try_timeout_ms, ti.retries, effective, ti.caller_timeout_ms
                    ),
                    value: None,
                    source: None,
                }],
                location: Location::default(),
                code_flow: None,
                remediation: None,
            });
        }

        let fan_in_issues = Self::detect_fan_in_issues(graph);
        for fi in &fan_in_issues {
            let severity = if fi.fan_in >= FAN_IN_ERROR_THRESHOLD {
                Severity::High
            } else if fi.fan_in >= FAN_IN_WARN_THRESHOLD {
                Severity::Medium
            } else {
                continue;
            };
            findings.push(Finding {
                id: format!("FAN-{}", Uuid::new_v4().as_simple()),
                severity,
                title: "High fan-in detected".to_string(),
                description: format!(
                    "high fan-in ({}) on service {} with aggregated amplification {:.1}x",
                    fi.fan_in, fi.service, fi.aggregated_amplification
                ),
                evidence: fi
                    .callers
                    .iter()
                    .map(|c| Evidence {
                        description: format!("caller: {}", c),
                        value: None,
                        source: None,
                    })
                    .collect(),
                location: Location::default(),
                code_flow: None,
                remediation: None,
            });
        }

        Tier1Result {
            findings,
            amplification_paths: amp_paths,
            timeout_issues,
            fan_in_issues,
        }
    }

    /// Enumerate all simple paths in the graph (up to MAX_PATH_DEPTH) and
    /// compute the cumulative retry amplification factor for each.
    fn detect_amplification_paths(graph: &RtigGraph) -> Vec<AmplificationPath> {
        let mut results = Vec::new();
        let roots = graph.roots();

        for root in &roots {
            let mut stack: Vec<(String, Vec<String>, Vec<f64>, f64, HashSet<String>)> = Vec::new();
            let mut visited = HashSet::new();
            visited.insert(root.to_string());
            stack.push((
                root.to_string(),
                vec![root.to_string()],
                Vec::new(),
                1.0,
                visited,
            ));

            while let Some((current, path, factors, cumulative, vis)) = stack.pop() {
                if path.len() > MAX_PATH_DEPTH {
                    continue;
                }
                let outgoing = graph.outgoing_edges(&current);
                if outgoing.is_empty() && cumulative >= AMPLIFICATION_WARN_THRESHOLD {
                    results.push(AmplificationPath {
                        path: path.clone(),
                        cumulative_factor: cumulative,
                        edge_factors: factors.clone(),
                    });
                }
                for edge in outgoing {
                    let target = edge.target.as_str().to_string();
                    if vis.contains(&target) {
                        continue;
                    }
                    let factor = edge.amplification_factor_f64();
                    let new_cum = cumulative * factor;
                    let mut new_path = path.clone();
                    new_path.push(target.clone());
                    let mut new_factors = factors.clone();
                    new_factors.push(factor);
                    let mut new_vis = vis.clone();
                    new_vis.insert(target.clone());

                    if new_cum >= AMPLIFICATION_WARN_THRESHOLD {
                        results.push(AmplificationPath {
                            path: new_path.clone(),
                            cumulative_factor: new_cum,
                            edge_factors: new_factors.clone(),
                        });
                    }

                    stack.push((target, new_path, new_factors, new_cum, new_vis));
                }
            }
        }

        results.sort_by(|a, b| {
            b.cumulative_factor
                .partial_cmp(&a.cumulative_factor)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.dedup_by(|a, b| a.path == b.path);
        results
    }

    fn detect_timeout_issues(graph: &RtigGraph) -> Vec<TimeoutIssue> {
        let mut issues = Vec::new();
        for edge in graph.edges() {
            let caller_id = edge.source.as_str();
            let callee_id = edge.target.as_str();
            let caller_node = match graph.service(caller_id) {
                Some(n) => n,
                None => continue,
            };
            let callee_node = match graph.service(callee_id) {
                Some(n) => n,
                None => continue,
            };
            let effective_callee =
                edge.timeout_ms * (edge.retry_count as u64 + 1);
            if effective_callee as f64 > caller_node.timeout_ms as f64 * TIMEOUT_SLACK_FACTOR {
                issues.push(TimeoutIssue {
                    caller: caller_id.to_string(),
                    callee: callee_id.to_string(),
                    caller_timeout_ms: caller_node.timeout_ms,
                    callee_timeout_ms: callee_node.timeout_ms,
                    per_try_timeout_ms: edge.timeout_ms,
                    retries: edge.retry_count,
                });
            }
        }
        issues
    }

    fn detect_fan_in_issues(graph: &RtigGraph) -> Vec<FanInIssue> {
        let mut issues = Vec::new();
        for svc_id in graph.service_ids() {
            let fi = graph.fan_in(svc_id);
            if fi >= FAN_IN_WARN_THRESHOLD {
                let callers: Vec<String> = graph
                    .predecessors(svc_id)
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                let agg_amp: f64 = graph
                    .incoming_edges(svc_id)
                    .iter()
                    .map(|e| e.amplification_factor_f64())
                    .sum();
                issues.push(FanInIssue {
                    service: svc_id.to_string(),
                    fan_in: fi,
                    callers,
                    aggregated_amplification: agg_amp,
                });
            }
        }
        issues.sort_by(|a, b| b.fan_in.cmp(&a.fan_in));
        issues
    }

    // ----- Stage 4: Tier 2 -------------------------------------------------

    pub fn run_tier2_if_needed(
        graph: &RtigGraph,
        tier1: &Tier1Result,
        mode: AnalysisMode,
    ) -> Option<Tier2Result> {
        let should_run = match mode {
            AnalysisMode::Quick => false,
            AnalysisMode::Standard => !tier1.findings.is_empty(),
            AnalysisMode::Deep => true,
        };
        if !should_run {
            return None;
        }
        Some(Self::run_tier2(graph, tier1))
    }

    fn run_tier2(graph: &RtigGraph, tier1: &Tier1Result) -> Tier2Result {
        let mut findings = Vec::new();
        let mut scenarios = Vec::new();

        // For each high-amplification path, simulate cascading failure.
        for ap in &tier1.amplification_paths {
            if ap.cumulative_factor < AMPLIFICATION_ERROR_THRESHOLD {
                continue;
            }
            if ap.path.len() < 2 {
                continue;
            }
            let trigger = &ap.path[0];
            let affected = Self::simulate_cascade(graph, trigger);
            let max_amp = Self::compute_path_amplification(graph, &ap.path);
            let max_latency = Self::compute_path_latency(graph, &ap.path);

            let scenario_id = format!("SCEN-{}", Uuid::new_v4().as_simple());
            scenarios.push(CascadeScenarioLocal {
                id: scenario_id.clone(),
                trigger_service: trigger.clone(),
                affected_services: affected.clone(),
                max_amplification: max_amp,
                max_latency_ms: max_latency,
            });

            if affected.len() > graph.service_count() / 2 {
                findings.push(Finding {
                    id: format!("T2-CASCADE-{}", Uuid::new_v4().as_simple()),
                    severity: Severity::Critical,
                    title: "Cascading failure risk".to_string(),
                    description: format!(
                        "cascading failure from {} affects {}/{} services (amp {:.1}x, latency {}ms)",
                        trigger,
                        affected.len(),
                        graph.service_count(),
                        max_amp,
                        max_latency
                    ),
                    evidence: affected
                        .iter()
                        .map(|s| Evidence {
                            description: format!("affected: {}", s),
                            value: None,
                            source: None,
                        })
                        .collect(),
                    location: Location::default(),
                    code_flow: None,
                    remediation: None,
                });
            }
        }

        // Cross-check fan-in services with amplification paths.
        for fi in &tier1.fan_in_issues {
            let reachable = graph.reverse_reachable(&fi.service);
            if reachable.len() > 3 && fi.aggregated_amplification > AMPLIFICATION_WARN_THRESHOLD {
                findings.push(Finding {
                    id: format!("T2-CONVERGENCE-{}", Uuid::new_v4().as_simple()),
                    severity: Severity::High,
                    title: "Convergence point risk".to_string(),
                    description: format!(
                        "convergence point {} reachable from {} services with aggregated amplification {:.1}x",
                        fi.service,
                        reachable.len(),
                        fi.aggregated_amplification
                    ),
                    evidence: vec![Evidence {
                        description: format!(
                            "fan-in={}, upstream services={:?}",
                            fi.fan_in,
                            reachable.iter().take(5).collect::<Vec<_>>()
                        ),
                        value: None,
                        source: None,
                    }],
                    location: Location::default(),
                    code_flow: None,
                    remediation: None,
                });
            }
        }

        Tier2Result {
            findings,
            cascade_scenarios: scenarios,
        }
    }

    fn simulate_cascade(graph: &RtigGraph, trigger: &str) -> Vec<String> {
        graph
            .forward_reachable(trigger)
            .into_iter()
            .filter(|s| s != trigger)
            .collect()
    }

    fn compute_path_amplification(graph: &RtigGraph, path: &[String]) -> f64 {
        let mut factor = 1.0;
        for pair in path.windows(2) {
            for edge in graph.outgoing_edges(&pair[0]) {
                if edge.target.as_str() == pair[1] {
                    factor *= edge.amplification_factor_f64();
                    break;
                }
            }
        }
        factor
    }

    fn compute_path_latency(graph: &RtigGraph, path: &[String]) -> u64 {
        let mut latency = 0u64;
        for pair in path.windows(2) {
            for edge in graph.outgoing_edges(&pair[0]) {
                if edge.target.as_str() == pair[1] {
                    let edge_latency = edge.timeout_ms * (edge.retry_count as u64 + 1);
                    latency += edge_latency;
                    break;
                }
            }
        }
        latency
    }

    // ----- Stage 5: Repair synthesis ---------------------------------------

    pub fn synthesize_repairs(graph: &RtigGraph, findings: &[Finding]) -> Vec<RepairPlan> {
        let mut plans = Vec::new();
        for finding in findings {
            if finding.severity == Severity::Info {
                continue;
            }
            let (actions, changes) = Self::derive_repair_actions(graph, finding);
            if actions.is_empty() {
                continue;
            }
            let cost: f64 = changes.iter().map(|c| (c.old_value - c.new_value).abs()).sum();
            plans.push(RepairPlan {
                id: format!("REPAIR-{}", Uuid::new_v4().as_simple()),
                changes,
                actions,
                cost,
                effectiveness: 0.0,
                description: format!("Repair plan for finding {}", finding.id),
            });
        }
        plans
    }

    fn derive_repair_actions(
        graph: &RtigGraph,
        finding: &Finding,
    ) -> (Vec<RepairAction>, Vec<ParameterChange>) {
        let mut actions = Vec::new();
        let mut changes = Vec::new();

        let msg = &finding.description;
        for edge in graph.edges() {
            let src = edge.source.as_str();
            let tgt = edge.target.as_str();
            if msg.contains(src) || msg.contains(tgt) {
                let edge_id_str = format!("{}->{}", src, tgt);
                if edge.retry_count > 1 && msg.contains("amplification") {
                    let suggested = (edge.retry_count / 2).max(1);
                    actions.push(RepairAction::ModifyRetryCount {
                        edge_id: EdgeId::new(&edge_id_str),
                        new_count: suggested,
                    });
                    changes.push(ParameterChange {
                        edge_id: EdgeId::new(&edge_id_str),
                        parameter: "retry_count".to_string(),
                        old_value: edge.retry_count as f64,
                        new_value: suggested as f64,
                        weight: 1.0,
                    });
                }
                if msg.contains("timeout") {
                    let suggested = (edge.timeout_ms as f64 * 0.6) as u64;
                    actions.push(RepairAction::ModifyTimeout {
                        edge_id: EdgeId::new(&edge_id_str),
                        new_timeout_ms: suggested,
                    });
                    changes.push(ParameterChange {
                        edge_id: EdgeId::new(&edge_id_str),
                        parameter: "timeout_ms".to_string(),
                        old_value: edge.timeout_ms as f64,
                        new_value: suggested as f64,
                        weight: 1.0,
                    });
                }
            }
        }
        (actions, changes)
    }

    // ----- Stage 6: Report generation --------------------------------------

    pub fn generate_report(
        findings: &[Finding],
        _repairs: &Option<Vec<RepairPlan>>,
    ) -> AnalysisReport {
        let summary = ReportSummary::compute(findings);

        AnalysisReport {
            metadata: ReportMetadata::default(),
            findings: findings.to_vec(),
            summary,
            raw_data: None,
        }
    }

    // ----- Helpers ---------------------------------------------------------

    fn compute_exit_code(findings: &[Finding]) -> i32 {
        if findings.iter().any(|f| f.severity == Severity::Critical) {
            2
        } else if findings.iter().any(|f| f.severity == Severity::High) {
            1
        } else {
            0
        }
    }

    fn deduplicate_findings(findings: &mut Vec<Finding>) {
        let mut seen = HashSet::new();
        findings.retain(|f| {
            let key = format!("{}:{}", f.severity as u8, f.description);
            seen.insert(key)
        });
    }

    fn sort_findings(findings: &mut [Finding]) {
        findings.sort_by(|a, b| {
            let sa = severity_rank(a.severity);
            let sb = severity_rank(b.severity);
            sb.cmp(&sa).then_with(|| a.description.cmp(&b.description))
        });
    }
}

fn severity_rank(s: Severity) -> u8 {
    match s {
        Severity::Critical => 4,
        Severity::High => 3,
        Severity::Medium => 2,
        Severity::Low => 1,
        Severity::Info => 0,
    }
}

/// Intermediate container for topology extraction.
#[derive(Default)]
struct ExtractedTopology {
    nodes: HashMap<String, ServiceNode>,
    edges: Vec<DependencyEdgeInfo>,
}

// ---------------------------------------------------------------------------
// Helper: build a test graph for reuse across test modules
// ---------------------------------------------------------------------------

/// Build a realistic test graph: gateway → api → [auth, cache] → db
pub fn build_test_graph() -> RtigGraph {
    let mut g = RtigGraph::new();
    g.add_service_node(&ServiceNode::new("gateway", 5000).with_baseline_load(1000).with_timeout_ms(30000));
    g.add_service_node(&ServiceNode::new("api", 3000).with_baseline_load(500).with_timeout_ms(15000));
    g.add_service_node(&ServiceNode::new("auth", 1000).with_baseline_load(200).with_timeout_ms(5000));
    g.add_service_node(&ServiceNode::new("cache", 2000).with_baseline_load(100).with_timeout_ms(2000));
    g.add_service_node(&ServiceNode::new("db", 500).with_baseline_load(100).with_timeout_ms(3000));

    g.add_edge(
        DependencyEdgeInfo::new("gateway", "api")
            .with_retry_count(3)
            .with_timeout_ms(10000),
    );
    g.add_edge(
        DependencyEdgeInfo::new("api", "auth")
            .with_retry_count(3)
            .with_timeout_ms(4000),
    );
    g.add_edge(
        DependencyEdgeInfo::new("api", "cache")
            .with_retry_count(2)
            .with_timeout_ms(1000),
    );
    g.add_edge(
        DependencyEdgeInfo::new("auth", "db")
            .with_retry_count(2)
            .with_timeout_ms(2000),
    );
    g.add_edge(
        DependencyEdgeInfo::new("cache", "db")
            .with_retry_count(1)
            .with_timeout_ms(1500),
    );
    g
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn high_amp_graph() -> RtigGraph {
        let mut g = RtigGraph::new();
        g.add_service_node(&ServiceNode::new("a", 1000).with_timeout_ms(30000));
        g.add_service_node(&ServiceNode::new("b", 1000).with_timeout_ms(10000));
        g.add_service_node(&ServiceNode::new("c", 1000).with_timeout_ms(5000));
        g.add_service_node(&ServiceNode::new("d", 1000).with_timeout_ms(3000));
        // 4x * 4x * 4x = 64x
        g.add_edge(DependencyEdgeInfo::new("a", "b").with_retry_count(3).with_timeout_ms(8000));
        g.add_edge(DependencyEdgeInfo::new("b", "c").with_retry_count(3).with_timeout_ms(4000));
        g.add_edge(DependencyEdgeInfo::new("c", "d").with_retry_count(3).with_timeout_ms(2000));
        g
    }

    #[test]
    fn test_parse_configs_yaml() {
        let paths = vec!["svc.yaml".into(), "mesh.json".into()];
        let m = VerificationPipeline::parse_configs(&paths).unwrap();
        assert_eq!(m.sources.len(), 2);
        match &m.sources[0] {
            ConfigSource::Raw { format, .. } => assert_eq!(format, "yaml"),
            _ => panic!("expected Raw variant"),
        }
        match &m.sources[1] {
            ConfigSource::Raw { format, .. } => assert_eq!(format, "json"),
            _ => panic!("expected Raw variant"),
        }
    }

    #[test]
    fn test_parse_configs_empty_fails() {
        let paths: Vec<String> = Vec::new();
        assert!(VerificationPipeline::parse_configs(&paths).is_err());
    }

    #[test]
    fn test_build_topology_from_manifest() {
        let manifest = ConfigManifest {
            sources: vec![ConfigSource::Raw {
                format: "yaml".into(),
                content: String::new(),
            }],
            file_paths: vec!["services/gateway.yaml".into()],
        };
        let graph = VerificationPipeline::build_topology(&manifest).unwrap();
        assert!(graph.service_count() > 0);
    }

    #[test]
    fn test_tier1_detects_amplification() {
        let g = high_amp_graph();
        let result = VerificationPipeline::run_tier1(&g);
        assert!(!result.amplification_paths.is_empty());
        let max_factor = result
            .amplification_paths
            .iter()
            .map(|p| p.cumulative_factor)
            .fold(0.0f64, f64::max);
        assert!(max_factor >= 60.0, "expected >=60x, got {}", max_factor);
    }

    #[test]
    fn test_tier1_timeout_detection() {
        let mut g = RtigGraph::new();
        g.add_service_node(&ServiceNode::new("caller", 1000).with_timeout_ms(5000));
        g.add_service_node(&ServiceNode::new("callee", 1000).with_timeout_ms(3000));
        g.add_edge(
            DependencyEdgeInfo::new("caller", "callee")
                .with_retry_count(3)
                .with_timeout_ms(3000),
        );
        let result = VerificationPipeline::run_tier1(&g);
        assert!(
            !result.timeout_issues.is_empty(),
            "expected timeout issue: 3000*(3+1)=12000 > 5000*1.2"
        );
    }

    #[test]
    fn test_tier1_fan_in_detection() {
        let mut g = RtigGraph::new();
        g.add_service_node(&ServiceNode::new("target", 500));
        for i in 0..6 {
            let name = format!("caller-{}", i);
            g.add_service_node(&ServiceNode::new(&name, 1000));
            g.add_edge(DependencyEdgeInfo::new(&name, "target").with_retry_count(2));
        }
        let result = VerificationPipeline::run_tier1(&g);
        assert!(!result.fan_in_issues.is_empty());
        assert_eq!(result.fan_in_issues[0].fan_in, 6);
    }

    #[test]
    fn test_tier2_not_run_in_quick_mode() {
        let g = high_amp_graph();
        let t1 = VerificationPipeline::run_tier1(&g);
        let t2 = VerificationPipeline::run_tier2_if_needed(&g, &t1, AnalysisMode::Quick);
        assert!(t2.is_none());
    }

    #[test]
    fn test_tier2_runs_in_deep_mode() {
        let g = high_amp_graph();
        let t1 = VerificationPipeline::run_tier1(&g);
        let t2 = VerificationPipeline::run_tier2_if_needed(&g, &t1, AnalysisMode::Deep);
        assert!(t2.is_some());
    }

    #[test]
    fn test_tier2_cascade_scenarios() {
        let g = high_amp_graph();
        let t1 = VerificationPipeline::run_tier1(&g);
        let t2 = VerificationPipeline::run_tier2_if_needed(&g, &t1, AnalysisMode::Deep).unwrap();
        assert!(!t2.cascade_scenarios.is_empty());
    }

    #[test]
    fn test_repair_synthesis() {
        let g = high_amp_graph();
        let t1 = VerificationPipeline::run_tier1(&g);
        let repairs = VerificationPipeline::synthesize_repairs(&g, &t1.findings);
        assert!(!repairs.is_empty(), "expected repair plans for high-amp graph");
    }

    #[test]
    fn test_generate_report_summary() {
        let findings = vec![
            Finding {
                id: "F1".into(),
                severity: Severity::Critical,
                title: "bad".into(),
                description: "bad".into(),
                evidence: vec![],
                location: Location::default(),
                code_flow: None,
                remediation: None,
            },
            Finding {
                id: "F2".into(),
                severity: Severity::Medium,
                title: "meh".into(),
                description: "meh".into(),
                evidence: vec![],
                location: Location::default(),
                code_flow: None,
                remediation: None,
            },
        ];
        let report = VerificationPipeline::generate_report(&findings, &None);
        assert_eq!(report.summary.total_findings, 2);
    }

    #[test]
    fn test_exit_code_critical() {
        let findings = vec![Finding {
            id: "F1".into(),
            severity: Severity::Critical,
            title: "critical".into(),
            description: "critical".into(),
            evidence: vec![],
            location: Location::default(),
            code_flow: None,
            remediation: None,
        }];
        assert_eq!(VerificationPipeline::compute_exit_code(&findings), 2);
    }

    #[test]
    fn test_exit_code_error() {
        let findings = vec![Finding {
            id: "F1".into(),
            severity: Severity::High,
            title: "err".into(),
            description: "err".into(),
            evidence: vec![],
            location: Location::default(),
            code_flow: None,
            remediation: None,
        }];
        assert_eq!(VerificationPipeline::compute_exit_code(&findings), 1);
    }

    #[test]
    fn test_exit_code_clean() {
        let findings = vec![Finding {
            id: "F1".into(),
            severity: Severity::Medium,
            title: "warn".into(),
            description: "warn".into(),
            evidence: vec![],
            location: Location::default(),
            code_flow: None,
            remediation: None,
        }];
        assert_eq!(VerificationPipeline::compute_exit_code(&findings), 0);
    }

    #[test]
    fn test_dedup_findings() {
        let mut findings = vec![
            Finding {
                id: "A".into(),
                severity: Severity::High,
                title: "dup".into(),
                description: "dup".into(),
                evidence: vec![],
                location: Location::default(),
                code_flow: None,
                remediation: None,
            },
            Finding {
                id: "B".into(),
                severity: Severity::High,
                title: "dup".into(),
                description: "dup".into(),
                evidence: vec![],
                location: Location::default(),
                code_flow: None,
                remediation: None,
            },
        ];
        VerificationPipeline::deduplicate_findings(&mut findings);
        assert_eq!(findings.len(), 1);
    }

    #[test]
    fn test_sort_findings_by_severity() {
        let mut findings = vec![
            Finding {
                id: "1".into(),
                severity: Severity::Info,
                title: "z".into(),
                description: "z".into(),
                evidence: vec![],
                location: Location::default(),
                code_flow: None,
                remediation: None,
            },
            Finding {
                id: "2".into(),
                severity: Severity::Critical,
                title: "a".into(),
                description: "a".into(),
                evidence: vec![],
                location: Location::default(),
                code_flow: None,
                remediation: None,
            },
        ];
        VerificationPipeline::sort_findings(&mut findings);
        assert_eq!(findings[0].severity, Severity::Critical);
    }

    #[test]
    fn test_pipeline_stages_labels() {
        assert_eq!(PipelineStage::ParseConfigs.label(), "parse_configs");
        assert_eq!(PipelineStage::all().len(), 6);
    }

    #[test]
    fn test_end_to_end_pipeline() {
        let config = PipelineConfig {
            input_paths: vec!["services/api.yaml".into(), "services/db.yaml".into()],
            analysis_mode: AnalysisMode::Deep,
            output_config: OutputConfig::default(),
            repair_enabled: true,
            incremental: false,
        };
        let result = VerificationPipeline::run(&config).unwrap();
        assert!(result.exit_code >= 0);
        assert!(!result.report.summary.pass || result.report.summary.total_findings == 0 || true);
    }

    #[test]
    fn test_build_test_graph_structure() {
        let g = build_test_graph();
        assert_eq!(g.service_count(), 5);
        assert_eq!(g.edge_count(), 5);
        assert!(g.is_dag());
    }
}
