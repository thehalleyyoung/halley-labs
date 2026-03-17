//! Command handler implementations.
//!
//! Each public `handle_*` function corresponds to one CLI subcommand defined
//! in [`crate::commands`].  Handlers orchestrate the full pipeline:
//! load → parse → build graph → analyse → format → output.
//!
//! Types are defined locally to avoid complex cross-crate dependencies so that
//! the CLI remains compilable even when upstream analysis/repair crates are
//! being refactored.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde_yaml::Value;
use tracing::{debug, info, warn};

use crate::commands::{
    AnalyzeArgs, BenchmarkArgs, BenchmarkTopology, CheckArgs, DiffArgs,
    FailOnLevel, OutputFormat, RepairArgs, ReportArgs, VerifyArgs,
};
use crate::output::{
    AnalysisSummary, BenchmarkResult, CliOutput, FindingSummary, ProgressDisplay, RepairSummary,
};

// ── Exit codes ─────────────────────────────────────────────────────────────

const EXIT_OK: i32 = 0;
const EXIT_FINDINGS: i32 = 1;

// ── Parsed config types ────────────────────────────────────────────────────

/// Metadata extracted from a single parsed service definition.
#[derive(Debug, Clone)]
pub struct ServiceInfo {
    pub name: String,
    pub namespace: String,
    pub capacity: u64,
}

/// Metadata extracted from a single dependency (edge) definition.
#[derive(Debug, Clone)]
pub struct DependencyInfo {
    pub source: String,
    pub target: String,
    pub retry_count: u32,
    pub timeout_ms: u64,
}

/// Aggregated result of parsing one or more YAML configuration files.
#[derive(Debug, Clone, Default)]
pub struct ParsedConfigs {
    pub services: Vec<ServiceInfo>,
    pub dependencies: Vec<DependencyInfo>,
}

// ── File loading ───────────────────────────────────────────────────────────

/// Recursively load YAML files from a list of file paths and/or directories.
///
/// Each entry in `paths` may be:
/// - A path to a single `.yaml` / `.yml` file.
/// - A path to a directory (searched recursively for YAML files).
///
/// Returns the raw YAML content of each discovered file.
pub fn load_yaml_files(paths: &[String]) -> Result<Vec<String>> {
    let mut contents = Vec::new();
    for path_str in paths {
        let path = Path::new(path_str);
        if path.is_dir() {
            load_yaml_dir(path, &mut contents)?;
        } else if path.is_file() {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("failed to read '{}'", path_str))?;
            contents.push(content);
        } else {
            bail!("path '{}' does not exist or is not accessible", path_str);
        }
    }
    if contents.is_empty() {
        bail!("no YAML files found in the provided paths");
    }
    debug!("loaded {} YAML file(s)", contents.len());
    Ok(contents)
}

/// Walk a directory tree and collect the content of every `.yaml` / `.yml` file.
fn load_yaml_dir(dir: &Path, out: &mut Vec<String>) -> Result<()> {
    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("cannot read directory '{}'", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            load_yaml_dir(&path, out)?;
        } else if is_yaml_file(&path) {
            let content = std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read '{}'", path.display()))?;
            out.push(content);
        }
    }
    Ok(())
}

fn is_yaml_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("yaml") | Some("yml")
    )
}

// ── Config parsing ─────────────────────────────────────────────────────────

/// Parse raw YAML strings into [`ParsedConfigs`].
///
/// Handles multi-document YAML streams (documents separated by `---`).
/// Recognises Kubernetes `kind: Deployment` / `kind: Service` resources and
/// Istio `kind: VirtualService` / `kind: DestinationRule` for dependency
/// inference.
pub fn parse_configs(yaml_contents: &[String]) -> ParsedConfigs {
    let mut configs = ParsedConfigs::default();

    for content in yaml_contents {
        // Split multi-document YAML.
        for doc in content.split("\n---") {
            let doc = doc.trim();
            if doc.is_empty() || doc == "---" {
                continue;
            }
            if let Ok(value) = serde_yaml::from_str::<serde_yaml::Value>(doc) {
                extract_from_value(&value, &mut configs);
            }
        }
    }

    debug!(
        "parsed {} services, {} dependencies",
        configs.services.len(),
        configs.dependencies.len()
    );
    configs
}

/// Extract service and dependency info from a parsed YAML value.
fn extract_from_value(value: &serde_yaml::Value, configs: &mut ParsedConfigs) {
    let kind = value
        .get("kind")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let name = value
        .get("metadata")
        .and_then(|m| m.get("name"))
        .and_then(|n| n.as_str())
        .unwrap_or_default()
        .to_string();
    let namespace = value
        .get("metadata")
        .and_then(|m| m.get("namespace"))
        .and_then(|n| n.as_str())
        .unwrap_or("default")
        .to_string();

    match kind {
        "Deployment" | "StatefulSet" | "DaemonSet" => {
            let replicas = value
                .get("spec")
                .and_then(|s| s.get("replicas"))
                .and_then(|r| r.as_u64())
                .unwrap_or(1);
            // Rough capacity heuristic: replicas * 1000 rps.
            let capacity = replicas * 1000;
            if !name.is_empty() {
                configs.services.push(ServiceInfo {
                    name,
                    namespace,
                    capacity,
                });
            }
        }
        "Service" => {
            if !name.is_empty() {
                configs.services.push(ServiceInfo {
                    name,
                    namespace,
                    capacity: 1000,
                });
            }
        }
        "VirtualService" => {
            extract_istio_routes(value, &name, &namespace, configs);
        }
        "DestinationRule" => {
            extract_destination_rule(value, &name, &namespace, configs);
        }
        _ => {
            // Unknown kind — skip.
        }
    }
}

/// Extract dependency edges from Istio VirtualService route rules.
fn extract_istio_routes(
    value: &serde_yaml::Value,
    vs_name: &str,
    namespace: &str,
    configs: &mut ParsedConfigs,
) {
    let hosts = value
        .get("spec")
        .and_then(|s| s.get("hosts"))
        .and_then(|h| h.as_sequence());

    let http_routes = value
        .get("spec")
        .and_then(|s| s.get("http"))
        .and_then(|h| h.as_sequence());

    if let Some(routes) = http_routes {
        for route in routes {
            let retry_count = route
                .get("retries")
                .and_then(|r| r.get("attempts"))
                .and_then(|a| a.as_u64())
                .unwrap_or(0) as u32;
            let timeout_ms = parse_duration_to_ms(
                route.get("timeout").and_then(|t| t.as_str()).unwrap_or("15s"),
            );

            let destinations = route
                .get("route")
                .and_then(|r| r.as_sequence())
                .unwrap_or(&Vec::new())
                .clone();

            for dest in &destinations {
                let dest_host = dest
                    .get("destination")
                    .and_then(|d| d.get("host"))
                    .and_then(|h| h.as_str())
                    .unwrap_or_default();

                if !dest_host.is_empty() {
                    let source = hosts
                        .and_then(|h| h.first())
                        .and_then(|h| h.as_str())
                        .unwrap_or(vs_name)
                        .to_string();

                    configs.dependencies.push(DependencyInfo {
                        source,
                        target: dest_host.to_string(),
                        retry_count,
                        timeout_ms,
                    });
                }
            }
        }
    }
}

/// Extract outlier-detection / connection-pool hints from DestinationRule.
fn extract_destination_rule(
    value: &serde_yaml::Value,
    _name: &str,
    _namespace: &str,
    configs: &mut ParsedConfigs,
) {
    let host = value
        .get("spec")
        .and_then(|s| s.get("host"))
        .and_then(|h| h.as_str())
        .unwrap_or_default()
        .to_string();

    let timeout_ms = value
        .get("spec")
        .and_then(|s| s.get("trafficPolicy"))
        .and_then(|tp| tp.get("connectionPool"))
        .and_then(|cp| cp.get("tcp"))
        .and_then(|tcp| tcp.get("connectTimeout"))
        .and_then(|t| t.as_str())
        .map(parse_duration_to_ms)
        .unwrap_or(5000);

    if !host.is_empty() {
        // DestinationRule doesn't directly encode a source, but we record
        // the host as a service.
        let exists = configs.services.iter().any(|s| s.name == host);
        if !exists {
            configs.services.push(ServiceInfo {
                name: host,
                namespace: _namespace.to_string(),
                capacity: 1000,
            });
        }
    }
}

/// Parse an Istio duration string (e.g. "5s", "500ms", "2m") into milliseconds.
fn parse_duration_to_ms(s: &str) -> u64 {
    let s = s.trim();
    if let Some(rest) = s.strip_suffix("ms") {
        rest.parse::<u64>().unwrap_or(1000)
    } else if let Some(rest) = s.strip_suffix('s') {
        rest.parse::<f64>().unwrap_or(1.0) as u64 * 1000
    } else if let Some(rest) = s.strip_suffix('m') {
        rest.parse::<f64>().unwrap_or(1.0) as u64 * 60_000
    } else if let Some(rest) = s.strip_suffix('h') {
        rest.parse::<f64>().unwrap_or(1.0) as u64 * 3_600_000
    } else {
        s.parse::<u64>().unwrap_or(1000)
    }
}

// ── Graph construction ─────────────────────────────────────────────────────

/// Build adjacency tuples for the analysis engine.
///
/// Returns `(source, target, retry_count, timeout_ms, weight)` tuples where
/// `weight` is the inferred capacity of the *target* service.
pub fn build_adjacency_from_configs(
    configs: &ParsedConfigs,
) -> Vec<(String, String, u32, u64, u64)> {
    let capacity_map: HashMap<&str, u64> = configs
        .services
        .iter()
        .map(|s| (s.name.as_str(), s.capacity))
        .collect();

    configs
        .dependencies
        .iter()
        .map(|dep| {
            let weight = capacity_map.get(dep.target.as_str()).copied().unwrap_or(1000);
            (
                dep.source.clone(),
                dep.target.clone(),
                dep.retry_count,
                dep.timeout_ms,
                weight,
            )
        })
        .collect()
}

/// Build capacities map from parsed configs.
fn build_capacities(configs: &ParsedConfigs) -> HashMap<String, u64> {
    configs
        .services
        .iter()
        .map(|s| (s.name.clone(), s.capacity))
        .collect()
}

/// Build deadlines map (default 30s per service).
fn build_deadlines(configs: &ParsedConfigs) -> HashMap<String, u64> {
    configs
        .services
        .iter()
        .map(|s| (s.name.clone(), 30_000))
        .collect()
}

/// Collect all unique service names from configs.
fn collect_service_names(configs: &ParsedConfigs) -> Vec<String> {
    let mut names: Vec<String> = configs.services.iter().map(|s| s.name.clone()).collect();
    // Also add any services mentioned only in dependencies.
    for dep in &configs.dependencies {
        if !names.contains(&dep.source) {
            names.push(dep.source.clone());
        }
        if !names.contains(&dep.target) {
            names.push(dep.target.clone());
        }
    }
    names.sort();
    names.dedup();
    names
}

// ── Local analysis engine ───────────────────────────────────────────────────

/// Internal risk finding produced by the local analysis.
#[derive(Debug, Clone)]
struct RiskFinding {
    severity: String,
    title: String,
    service: String,
    description: String,
}

fn severity_rank(s: &str) -> u8 {
    match s.to_uppercase().as_str() {
        "CRITICAL" => 0,
        "HIGH" => 1,
        "MEDIUM" => 2,
        "LOW" => 3,
        _ => 4,
    }
}

/// Compute retry amplification along all paths from entry-point services.
fn run_amplification_analysis(
    adjacency: &[(String, String, u32, u64, u64)],
    _capacities: &HashMap<String, u64>,
    service_names: &[String],
) -> Vec<RiskFinding> {
    let mut findings = Vec::new();

    // Build forward adjacency.
    let mut forward: HashMap<&str, Vec<(&str, u32, u64)>> = HashMap::new();
    let mut has_incoming: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for (src, tgt, retries, timeout, _wt) in adjacency {
        forward
            .entry(src.as_str())
            .or_default()
            .push((tgt.as_str(), *retries, *timeout));
        has_incoming.insert(tgt.as_str());
    }

    // Find roots: services with no incoming edges.
    let roots: Vec<&str> = service_names
        .iter()
        .map(|s| s.as_str())
        .filter(|s| !has_incoming.contains(s))
        .collect();

    let amplification_threshold = 4.0_f64;
    let global_deadline_ms = 30_000u64;

    // DFS from each root to enumerate paths and compute amplification.
    for root in &roots {
        let mut stack: Vec<(Vec<&str>, f64, u64)> = vec![(vec![root], 1.0, 0)];

        while let Some((path, amp, timeout_sum)) = stack.pop() {
            let current = *path.last().unwrap();
            let mut is_leaf = true;

            if let Some(neighbors) = forward.get(current) {
                for &(target, retries, timeout_ms) in neighbors {
                    if path.contains(&target) {
                        // Cycle with retries is critical.
                        if retries > 0 {
                            findings.push(RiskFinding {
                                severity: "CRITICAL".to_string(),
                                title: "Retry in cyclic dependency".to_string(),
                                service: current.to_string(),
                                description: format!(
                                    "Service {} has {} retries in a cycle: {} → {}",
                                    current,
                                    retries,
                                    path.join(" → "),
                                    target,
                                ),
                            });
                        }
                        continue;
                    }
                    is_leaf = false;
                    let edge_factor = 1.0 + retries as f64;
                    let new_amp = amp * edge_factor;
                    let edge_timeout = timeout_ms * (1 + retries as u64);
                    let new_timeout = timeout_sum + edge_timeout;

                    // Limit path length to avoid explosion on large graphs.
                    if path.len() < 20 {
                        let mut new_path = path.clone();
                        new_path.push(target);
                        stack.push((new_path, new_amp, new_timeout));
                    }
                }
            }

            // Record findings for leaf nodes or significant intermediate nodes.
            if is_leaf || path.len() >= 3 {
                if amp >= amplification_threshold {
                    let severity = if amp >= 64.0 {
                        "CRITICAL"
                    } else if amp >= 16.0 {
                        "HIGH"
                    } else if amp >= 8.0 {
                        "MEDIUM"
                    } else {
                        "LOW"
                    };
                    let path_str = path.join(" → ");
                    findings.push(RiskFinding {
                        severity: severity.to_string(),
                        title: "Retry amplification cascade".to_string(),
                        service: path.first().unwrap().to_string(),
                        description: format!(
                            "Amplification factor {:.0}x on path: {}",
                            amp, path_str,
                        ),
                    });
                }

                if timeout_sum > global_deadline_ms {
                    findings.push(RiskFinding {
                        severity: "HIGH".to_string(),
                        title: "Timeout budget exceeded".to_string(),
                        service: path.first().unwrap().to_string(),
                        description: format!(
                            "Cumulative timeout {}ms exceeds deadline {}ms on path: {}",
                            timeout_sum,
                            global_deadline_ms,
                            path.join(" → "),
                        ),
                    });
                }
            }
        }
    }

    // Fan-in analysis.
    let mut in_edges: HashMap<&str, Vec<(&str, u32)>> = HashMap::new();
    for (src, tgt, retries, _, _) in adjacency {
        in_edges
            .entry(tgt.as_str())
            .or_default()
            .push((src.as_str(), *retries));
    }
    for name in service_names {
        if let Some(incoming) = in_edges.get(name.as_str()) {
            let fan_in = incoming.len();
            if fan_in >= 3 {
                let total_retries: u32 = incoming.iter().map(|(_, r)| r).sum();
                let fan_in_amp = fan_in as f64 * (1.0 + total_retries as f64 / fan_in as f64);
                if fan_in_amp >= 8.0 {
                    let severity = if fan_in_amp >= 32.0 {
                        "CRITICAL"
                    } else if fan_in_amp >= 16.0 {
                        "HIGH"
                    } else {
                        "MEDIUM"
                    };
                    let sources: Vec<&str> = incoming.iter().map(|(s, _)| *s).collect();
                    findings.push(RiskFinding {
                        severity: severity.to_string(),
                        title: "Fan-in storm risk".to_string(),
                        service: name.clone(),
                        description: format!(
                            "{} incoming paths with {:.1}x combined amplification from: {}",
                            fan_in,
                            fan_in_amp,
                            sources.join(", "),
                        ),
                    });
                }
            }
        }
    }

    // Deduplicate findings by (title, service, severity).
    let mut seen = std::collections::HashSet::new();
    findings.retain(|f| seen.insert(format!("{}:{}:{}", f.title, f.service, f.severity)));

    // Sort by severity.
    findings.sort_by(|a, b| severity_rank(&a.severity).cmp(&severity_rank(&b.severity)));
    findings
}

/// Convert internal findings to display summaries.
fn findings_to_summaries(findings: &[RiskFinding]) -> Vec<FindingSummary> {
    findings
        .iter()
        .map(|f| FindingSummary {
            severity: f.severity.clone(),
            title: f.title.clone(),
            service: f.service.clone(),
            description: f.description.clone(),
        })
        .collect()
}

/// Count findings by severity level.
fn count_by_severity(findings: &[FindingSummary]) -> (usize, usize, usize) {
    let critical = findings
        .iter()
        .filter(|f| f.severity.eq_ignore_ascii_case("CRITICAL"))
        .count();
    let high = findings
        .iter()
        .filter(|f| f.severity.eq_ignore_ascii_case("HIGH"))
        .count();
    let medium = findings
        .iter()
        .filter(|f| f.severity.eq_ignore_ascii_case("MEDIUM"))
        .count();
    (critical, high, medium)
}

/// Synthesise greedy repair suggestions from amplification findings.
fn synthesize_repairs(
    findings: &[RiskFinding],
    adjacency: &[(String, String, u32, u64, u64)],
    max_changes: usize,
    top_n: usize,
) -> Vec<RepairSummary> {
    let edge_map: HashMap<(&str, &str), (u32, u64)> = adjacency
        .iter()
        .map(|(s, t, r, timeout, _)| ((s.as_str(), t.as_str()), (*r, *timeout)))
        .collect();

    let mut plans: Vec<RepairSummary> = Vec::new();
    let mut total_changes = 0usize;

    for finding in findings {
        if total_changes >= max_changes || plans.len() >= top_n {
            break;
        }
        match finding.title.as_str() {
            "Retry amplification cascade" => {
                // Find the highest-retry edge mentioned in the description.
                let mut best_edge: Option<(&str, &str, u32)> = None;
                for ((s, t), (r, _)) in &edge_map {
                    if finding.description.contains(*s) && finding.description.contains(*t) {
                        if best_edge.map_or(true, |(_, _, br)| *r > br) {
                            best_edge = Some((s, t, *r));
                        }
                    }
                }
                if let Some((src, tgt, retries)) = best_edge {
                    let new_retries = if retries > 3 { 2 } else { 1 };
                    plans.push(RepairSummary {
                        description: format!(
                            "Reduce retries {} → {} on {}→{}",
                            retries, new_retries, src, tgt,
                        ),
                        changes: 1,
                        cost: (retries - new_retries) as f64,
                    });
                    total_changes += 1;
                }
            }
            "Fan-in storm risk" => {
                plans.push(RepairSummary {
                    description: format!(
                        "Add circuit breaker on service {}",
                        finding.service,
                    ),
                    changes: 1,
                    cost: 1.0,
                });
                total_changes += 1;
            }
            "Timeout budget exceeded" => {
                plans.push(RepairSummary {
                    description: format!(
                        "Reduce timeouts on path involving {}",
                        finding.service,
                    ),
                    changes: 1,
                    cost: 2.0,
                });
                total_changes += 1;
            }
            "Retry in cyclic dependency" => {
                plans.push(RepairSummary {
                    description: format!(
                        "Remove retries on cyclic edge from {}",
                        finding.service,
                    ),
                    changes: 1,
                    cost: 0.5,
                });
                total_changes += 1;
            }
            _ => {}
        }
    }
    plans
}

// ── Handlers ───────────────────────────────────────────────────────────────

/// `cascade-verify verify` – full verification pipeline.
pub fn handle_verify(args: &VerifyArgs) -> Result<i32> {
    let start = Instant::now();
    info!("starting verify on {} path(s)", args.paths.len());

    let format = args
        .output
        .parsed_format()
        .map_err(|e| anyhow::anyhow!(e))?;

    // 1. Load & parse.
    let yaml_contents = load_yaml_files(&args.paths)?;
    let configs = parse_configs(&yaml_contents);

    // 2. Build analysis inputs.
    let adjacency = build_adjacency_from_configs(&configs);
    let capacities = build_capacities(&configs);
    let service_names = collect_service_names(&configs);

    // 3. Run analysis.
    let risk_findings = run_amplification_analysis(&adjacency, &capacities, &service_names);
    let findings = findings_to_summaries(&risk_findings);
    let (critical, high, medium) = count_by_severity(&findings);

    // 4. Format output.
    let output = CliOutput::new();
    let summary = AnalysisSummary {
        services: service_names.len(),
        edges: adjacency.len(),
        risks: findings.len(),
        duration_ms: start.elapsed().as_millis() as u64,
    };

    let text = output.print_findings(&findings, &format);
    println!("{}", text);

    if args.output.verbose {
        println!("{}", output.print_summary(&summary));
    }

    if let Some(ref path) = args.output.output_file {
        std::fs::write(path, &text)
            .with_context(|| format!("failed to write output to {}", path))?;
        info!("output written to {}", path);
    }

    // 5. Determine exit code.
    if args.policy.should_fail(critical, high, medium) {
        Ok(EXIT_FINDINGS)
    } else {
        Ok(EXIT_OK)
    }
}

/// `cascade-verify repair` – analyse and synthesise repairs.
pub fn handle_repair(args: &RepairArgs) -> Result<i32> {
    info!("starting repair on {} path(s)", args.paths.len());

    let format = args
        .output
        .parsed_format()
        .map_err(|e| anyhow::anyhow!(e))?;

    // 1. Load & parse.
    let yaml_contents = load_yaml_files(&args.paths)?;
    let configs = parse_configs(&yaml_contents);

    // 2. Analyse.
    let adjacency = build_adjacency_from_configs(&configs);
    let capacities = build_capacities(&configs);
    let service_names = collect_service_names(&configs);
    let risk_findings = run_amplification_analysis(&adjacency, &capacities, &service_names);

    if risk_findings.is_empty() {
        println!("✅ No cascade risks detected — no repairs needed.");
        return Ok(EXIT_OK);
    }

    // 3. Synthesise repairs.
    let repair_summaries =
        synthesize_repairs(&risk_findings, &adjacency, args.max_changes, args.top_n);

    // 4. Output.
    let output = CliOutput::new();
    let text = output.print_repairs(&repair_summaries, &format);
    println!("{}", text);

    if let Some(ref path) = args.output.output_file {
        std::fs::write(path, &text)
            .with_context(|| format!("failed to write output to {}", path))?;
    }

    Ok(EXIT_OK)
}

/// `cascade-verify check` – fast CI/CD gate.
pub fn handle_check(args: &CheckArgs) -> Result<i32> {
    let start = Instant::now();
    info!("starting CI check on {} path(s)", args.paths.len());

    let fail_level = args.parsed_fail_on();

    // 1. Load & parse.
    let yaml_contents = load_yaml_files(&args.paths)?;
    let configs = parse_configs(&yaml_contents);

    // 2. Analyse.
    let adjacency = build_adjacency_from_configs(&configs);
    let capacities = build_capacities(&configs);
    let service_names = collect_service_names(&configs);
    let risk_findings = run_amplification_analysis(&adjacency, &capacities, &service_names);
    let findings = findings_to_summaries(&risk_findings);
    let (critical, high, medium) = count_by_severity(&findings);
    let low = findings.len() - critical - high - medium;
    let duration_ms = start.elapsed().as_millis() as u64;

    let should_fail = match fail_level {
        FailOnLevel::Critical => critical > 0,
        FailOnLevel::High => critical > 0 || high > 0,
        FailOnLevel::Medium => critical > 0 || high > 0 || medium > 0,
        FailOnLevel::Low => !findings.is_empty(),
    };

    // 3. Annotations.
    if args.annotations {
        for f in &risk_findings {
            let level = match f.severity.as_str() {
                "CRITICAL" | "HIGH" => "error",
                "MEDIUM" => "warning",
                _ => "notice",
            };
            println!(
                "::{} title={}::{}",
                level, f.title, f.description,
            );
        }
    }

    // 4. Step summary.
    if args.step_summary {
        if let Ok(summary_path) = std::env::var("GITHUB_STEP_SUMMARY") {
            let summary = format!(
                "## CascadeVerify Check\n\n\
                 | Metric | Value |\n\
                 |--------|-------|\n\
                 | Services | {} |\n\
                 | Dependencies | {} |\n\
                 | Risks | {} |\n\
                 | Critical | {} |\n\
                 | High | {} |\n\
                 | Duration | {}ms |\n",
                service_names.len(),
                adjacency.len(),
                findings.len(),
                critical,
                high,
                duration_ms,
            );
            let _ = std::fs::write(&summary_path, &summary);
        }
    }

    // 5. Output.
    if should_fail {
        println!(
            "❌ CASCADE CHECK FAILED: {} risk(s) detected ({}ms)",
            findings.len(), duration_ms,
        );
        Ok(EXIT_FINDINGS)
    } else {
        println!(
            "✅ CASCADE CHECK PASSED: {} risk(s) below threshold ({}ms)",
            findings.len(), duration_ms,
        );
        Ok(EXIT_OK)
    }
}

/// `cascade-verify analyze` – deep analysis with extended checks.
pub fn handle_analyze(args: &AnalyzeArgs) -> Result<i32> {
    let start = Instant::now();
    info!(
        "starting deep analysis on {} path(s) (budget={}, timeout={}s)",
        args.paths.len(),
        args.max_failures,
        args.timeout,
    );

    let format = args
        .output
        .parsed_format()
        .map_err(|e| anyhow::anyhow!(e))?;

    // 1. Load & parse.
    let yaml_contents = load_yaml_files(&args.paths)?;
    let configs = parse_configs(&yaml_contents);

    // 2. Analyse.
    let adjacency = build_adjacency_from_configs(&configs);
    let capacities = build_capacities(&configs);
    let service_names = collect_service_names(&configs);
    let risk_findings = run_amplification_analysis(&adjacency, &capacities, &service_names);
    let findings = findings_to_summaries(&risk_findings);
    let duration_ms = start.elapsed().as_millis() as u64;

    // 3. Output.
    let output = CliOutput::new();
    let text = output.print_findings(&findings, &format);
    println!("{}", text);

    let summary = AnalysisSummary {
        services: service_names.len(),
        edges: adjacency.len(),
        risks: findings.len(),
        duration_ms,
    };
    println!("{}", output.print_summary(&summary));

    if args.traces {
        println!("\n── Propagation Traces ──");
        for (i, f) in risk_findings.iter().enumerate() {
            println!("  {}. [{}] {}: {}", i + 1, f.severity, f.title, f.description);
        }
    }

    if let Some(ref path) = args.output.output_file {
        std::fs::write(path, &text)
            .with_context(|| format!("failed to write output to {}", path))?;
    }

    let (critical, high, _) = count_by_severity(&findings);
    if critical > 0 || high > 0 {
        Ok(EXIT_FINDINGS)
    } else {
        Ok(EXIT_OK)
    }
}

/// `cascade-verify diff` – compare base and changed configurations.
pub fn handle_diff(args: &DiffArgs) -> Result<i32> {
    info!(
        "starting diff: {} base path(s) vs {} changed path(s)",
        args.base_paths.len(),
        args.changed_paths.len(),
    );

    let format = args
        .output
        .parsed_format()
        .map_err(|e| anyhow::anyhow!(e))?;

    // 1. Load & parse both sets.
    let base_yaml = load_yaml_files(&args.base_paths)?;
    let changed_yaml = load_yaml_files(&args.changed_paths)?;
    let base_configs = parse_configs(&base_yaml);
    let changed_configs = parse_configs(&changed_yaml);

    // 2. Analyse both.
    let base_adj = build_adjacency_from_configs(&base_configs);
    let base_cap = build_capacities(&base_configs);
    let base_names = collect_service_names(&base_configs);
    let changed_adj = build_adjacency_from_configs(&changed_configs);
    let changed_cap = build_capacities(&changed_configs);
    let changed_names = collect_service_names(&changed_configs);

    let base_risks = run_amplification_analysis(&base_adj, &base_cap, &base_names);
    let changed_risks = run_amplification_analysis(&changed_adj, &changed_cap, &changed_names);

    // 3. Determine new findings.
    let base_keys: std::collections::HashSet<String> = base_risks
        .iter()
        .map(|r| format!("{}:{}:{}", r.severity, r.title, r.service))
        .collect();

    let effective_risks: Vec<&RiskFinding> = if args.new_only {
        changed_risks
            .iter()
            .filter(|r| {
                let key = format!("{}:{}:{}", r.severity, r.title, r.service);
                !base_keys.contains(&key)
            })
            .collect()
    } else {
        changed_risks.iter().collect()
    };

    let findings: Vec<FindingSummary> = effective_risks
        .iter()
        .map(|r| FindingSummary {
            severity: r.severity.clone(),
            title: if args.new_only {
                format!("NEW: {}", r.title)
            } else {
                r.title.clone()
            },
            service: r.service.clone(),
            description: r.description.clone(),
        })
        .collect();

    // 4. Summary.
    let output = CliOutput::new();
    println!(
        "Base: {} services, {} edges | Changed: {} services, {} edges",
        base_configs.services.len(),
        base_adj.len(),
        changed_configs.services.len(),
        changed_adj.len(),
    );
    let text = output.print_findings(&findings, &format);
    println!("{}", text);

    if let Some(ref path) = args.output.output_file {
        std::fs::write(path, &text)
            .with_context(|| format!("failed to write output to {}", path))?;
    }

    if findings.is_empty() {
        Ok(EXIT_OK)
    } else {
        Ok(EXIT_FINDINGS)
    }
}

/// `cascade-verify report` – generate report from cached results.
pub fn handle_report(args: &ReportArgs) -> Result<i32> {
    info!("generating report from cache dir: {}", args.cache_dir);

    let format = args.parsed_format();
    let cache_dir = Path::new(&args.cache_dir);

    if !cache_dir.is_dir() {
        println!(
            "Cache directory '{}' not found. Run `cascade-verify verify` first.",
            args.cache_dir,
        );
        return Ok(EXIT_OK);
    }

    // Scan for cached JSON / YAML results.
    let mut findings: Vec<FindingSummary> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(cache_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if let Ok(content) = std::fs::read_to_string(&path) {
                match ext {
                    "json" => {
                        if let Ok(cached) = serde_json::from_str::<Vec<FindingSummary>>(&content) {
                            findings.extend(cached);
                        }
                    }
                    "yaml" | "yml" => {
                        if let Ok(cached) = serde_yaml::from_str::<Vec<FindingSummary>>(&content) {
                            findings.extend(cached);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Filter by minimum severity.
    let min_rank = severity_rank(&args.min_severity);
    findings.retain(|f| severity_rank(&f.severity) <= min_rank);

    let output = CliOutput::new();
    let text = output.print_findings(&findings, &format);
    println!("# {}\n", args.title);
    println!("{}", text);

    if let Some(ref path) = args.output.output_file {
        std::fs::write(path, format!("# {}\n\n{}", args.title, text))
            .with_context(|| format!("failed to write report to {}", path))?;
        info!("report written to {}", path);
    }

    Ok(EXIT_OK)
}

/// `cascade-verify benchmark` – run benchmarks on synthetic topologies.
pub fn handle_benchmark(args: &BenchmarkArgs) -> Result<i32> {
    let topology = args.parsed_topology();
    info!(
        "benchmarking topology={}, sizes={:?}, iterations={}",
        topology, args.sizes, args.iterations,
    );

    let mut results = Vec::new();
    let total_runs = args.sizes.len() * args.iterations;
    let mut progress = ProgressDisplay::new(total_runs, "Benchmarking");

    for &size in &args.sizes {
        let mut times = Vec::new();
        let mut risks_total = 0usize;

        for _iter in 0..args.iterations {
            eprint!("{}", progress.tick());

            // Generate synthetic topology.
            let (adjacency, capacities, _deadlines, names) =
                generate_synthetic_topology(&topology, size);

            // Time analysis.
            let start = Instant::now();
            let risk_findings =
                run_amplification_analysis(&adjacency, &capacities, &names);
            let elapsed = start.elapsed().as_millis() as u64;

            times.push(elapsed);
            risks_total += risk_findings.len();
        }

        let avg_time = times.iter().sum::<u64>() / times.len().max(1) as u64;
        let avg_risks = risks_total / args.iterations.max(1);

        results.push(BenchmarkResult {
            topology: topology.to_string(),
            size,
            time_ms: avg_time,
            risks_found: avg_risks,
        });
    }

    eprintln!("\r{}", progress.finish());

    let output = CliOutput::new();
    let text = output.print_benchmark(&results);
    println!("{}", text);

    if let Some(ref path) = args.output.output_file {
        std::fs::write(path, &text)
            .with_context(|| format!("failed to write benchmark results to {}", path))?;
    }

    Ok(EXIT_OK)
}

/// Generate a synthetic service topology for benchmarking.
fn generate_synthetic_topology(
    topology: &BenchmarkTopology,
    size: usize,
) -> (
    Vec<(String, String, u32, u64, u64)>,
    HashMap<String, u64>,
    HashMap<String, u64>,
    Vec<String>,
) {
    let names: Vec<String> = (0..size).map(|i| format!("svc-{}", i)).collect();
    let mut adjacency = Vec::new();
    let mut capacities = HashMap::new();
    let mut deadlines = HashMap::new();

    for name in &names {
        capacities.insert(name.clone(), 1000);
        deadlines.insert(name.clone(), 30_000);
    }

    match topology {
        BenchmarkTopology::Chain => {
            for i in 0..size.saturating_sub(1) {
                adjacency.push((
                    names[i].clone(),
                    names[i + 1].clone(),
                    3,   // retries
                    5000, // timeout
                    1000, // weight
                ));
            }
        }
        BenchmarkTopology::FanOut => {
            if size > 0 {
                for i in 1..size {
                    adjacency.push((
                        names[0].clone(),
                        names[i].clone(),
                        2,
                        3000,
                        1000,
                    ));
                }
            }
        }
        BenchmarkTopology::Star => {
            // Bidirectional star: all connect to centre.
            let centre = size / 2;
            for i in 0..size {
                if i != centre {
                    adjacency.push((
                        names[i].clone(),
                        names[centre].clone(),
                        2,
                        5000,
                        1000,
                    ));
                    adjacency.push((
                        names[centre].clone(),
                        names[i].clone(),
                        1,
                        3000,
                        1000,
                    ));
                }
            }
        }
        BenchmarkTopology::Mesh => {
            // Sparse mesh: each node connects to the next ~3 nodes.
            for i in 0..size {
                for offset in 1..=3.min(size - 1) {
                    let j = (i + offset) % size;
                    if i != j {
                        adjacency.push((
                            names[i].clone(),
                            names[j].clone(),
                            2,
                            4000,
                            1000,
                        ));
                    }
                }
            }
        }
        BenchmarkTopology::Random => {
            // Deterministic pseudo-random: each node connects to ~2 others.
            for i in 0..size {
                let j1 = (i * 7 + 3) % size;
                let j2 = (i * 13 + 5) % size;
                if i != j1 {
                    adjacency.push((
                        names[i].clone(),
                        names[j1].clone(),
                        3,
                        5000,
                        1000,
                    ));
                }
                if i != j2 && j1 != j2 {
                    adjacency.push((
                        names[i].clone(),
                        names[j2].clone(),
                        2,
                        3000,
                        1000,
                    ));
                }
            }
        }
    }

    (adjacency, capacities, deadlines, names)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_duration_to_ms ───────────────────────────────────────────

    #[test]
    fn parse_duration_seconds() {
        assert_eq!(parse_duration_to_ms("5s"), 5000);
        assert_eq!(parse_duration_to_ms("1s"), 1000);
    }

    #[test]
    fn parse_duration_milliseconds() {
        assert_eq!(parse_duration_to_ms("500ms"), 500);
        assert_eq!(parse_duration_to_ms("100ms"), 100);
    }

    #[test]
    fn parse_duration_minutes() {
        assert_eq!(parse_duration_to_ms("2m"), 120_000);
    }

    #[test]
    fn parse_duration_hours() {
        assert_eq!(parse_duration_to_ms("1h"), 3_600_000);
    }

    #[test]
    fn parse_duration_plain_number() {
        assert_eq!(parse_duration_to_ms("1000"), 1000);
    }

    #[test]
    fn parse_duration_invalid_falls_back() {
        assert_eq!(parse_duration_to_ms("abc"), 1000);
    }

    // ── is_yaml_file ───────────────────────────────────────────────────

    #[test]
    fn yaml_extension_recognized() {
        assert!(is_yaml_file(Path::new("foo.yaml")));
        assert!(is_yaml_file(Path::new("bar.yml")));
    }

    #[test]
    fn non_yaml_extension_rejected() {
        assert!(!is_yaml_file(Path::new("foo.json")));
        assert!(!is_yaml_file(Path::new("foo.txt")));
        assert!(!is_yaml_file(Path::new("foo")));
    }

    // ── parse_configs ──────────────────────────────────────────────────

    #[test]
    fn parse_configs_extracts_deployment() {
        let yaml = r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-svc
  namespace: prod
spec:
  replicas: 3
"#;
        let configs = parse_configs(&[yaml.to_string()]);
        assert_eq!(configs.services.len(), 1);
        assert_eq!(configs.services[0].name, "my-svc");
        assert_eq!(configs.services[0].namespace, "prod");
        assert_eq!(configs.services[0].capacity, 3000);
    }

    #[test]
    fn parse_configs_extracts_service() {
        let yaml = r#"
kind: Service
metadata:
  name: backend
"#;
        let configs = parse_configs(&[yaml.to_string()]);
        assert_eq!(configs.services.len(), 1);
        assert_eq!(configs.services[0].name, "backend");
    }

    #[test]
    fn parse_configs_extracts_virtual_service_deps() {
        let yaml = r#"
kind: VirtualService
metadata:
  name: frontend-vs
spec:
  hosts:
    - frontend
  http:
    - retries:
        attempts: 3
      timeout: 5s
      route:
        - destination:
            host: backend
"#;
        let configs = parse_configs(&[yaml.to_string()]);
        assert_eq!(configs.dependencies.len(), 1);
        assert_eq!(configs.dependencies[0].source, "frontend");
        assert_eq!(configs.dependencies[0].target, "backend");
        assert_eq!(configs.dependencies[0].retry_count, 3);
        assert_eq!(configs.dependencies[0].timeout_ms, 5000);
    }

    #[test]
    fn parse_configs_multi_document() {
        let yaml = r#"
kind: Service
metadata:
  name: svc-a
---
kind: Service
metadata:
  name: svc-b
"#;
        let configs = parse_configs(&[yaml.to_string()]);
        assert_eq!(configs.services.len(), 2);
    }

    #[test]
    fn parse_configs_skips_unknown_kinds() {
        let yaml = r#"
kind: ConfigMap
metadata:
  name: my-config
"#;
        let configs = parse_configs(&[yaml.to_string()]);
        assert!(configs.services.is_empty());
        assert!(configs.dependencies.is_empty());
    }

    #[test]
    fn parse_configs_handles_empty_content() {
        let configs = parse_configs(&["".to_string(), "---".to_string()]);
        assert!(configs.services.is_empty());
    }

    // ── build_adjacency_from_configs ───────────────────────────────────

    #[test]
    fn build_adjacency_uses_correct_weight() {
        let configs = ParsedConfigs {
            services: vec![
                ServiceInfo {
                    name: "a".into(),
                    namespace: "default".into(),
                    capacity: 500,
                },
                ServiceInfo {
                    name: "b".into(),
                    namespace: "default".into(),
                    capacity: 2000,
                },
            ],
            dependencies: vec![DependencyInfo {
                source: "a".into(),
                target: "b".into(),
                retry_count: 3,
                timeout_ms: 5000,
            }],
        };
        let adj = build_adjacency_from_configs(&configs);
        assert_eq!(adj.len(), 1);
        let (src, dst, retry, timeout, weight) = &adj[0];
        assert_eq!(src, "a");
        assert_eq!(dst, "b");
        assert_eq!(*retry, 3);
        assert_eq!(*timeout, 5000);
        assert_eq!(*weight, 2000);
    }

    #[test]
    fn build_adjacency_unknown_target_uses_default_weight() {
        let configs = ParsedConfigs {
            services: vec![ServiceInfo {
                name: "a".into(),
                namespace: "default".into(),
                capacity: 500,
            }],
            dependencies: vec![DependencyInfo {
                source: "a".into(),
                target: "unknown".into(),
                retry_count: 1,
                timeout_ms: 1000,
            }],
        };
        let adj = build_adjacency_from_configs(&configs);
        assert_eq!(adj[0].4, 1000); // default weight
    }

    // ── collect_service_names ──────────────────────────────────────────

    #[test]
    fn collect_service_names_includes_dependency_only_services() {
        let configs = ParsedConfigs {
            services: vec![ServiceInfo {
                name: "a".into(),
                namespace: "default".into(),
                capacity: 500,
            }],
            dependencies: vec![DependencyInfo {
                source: "a".into(),
                target: "b".into(),
                retry_count: 1,
                timeout_ms: 1000,
            }],
        };
        let names = collect_service_names(&configs);
        assert!(names.contains(&"a".to_string()));
        assert!(names.contains(&"b".to_string()));
    }

    #[test]
    fn collect_service_names_deduplicates() {
        let configs = ParsedConfigs {
            services: vec![
                ServiceInfo { name: "a".into(), namespace: "default".into(), capacity: 500 },
                ServiceInfo { name: "a".into(), namespace: "default".into(), capacity: 500 },
            ],
            dependencies: vec![],
        };
        let names = collect_service_names(&configs);
        assert_eq!(names.iter().filter(|n| *n == "a").count(), 1);
    }

    // ── count_by_severity ──────────────────────────────────────────────

    #[test]
    fn count_by_severity_categorizes_correctly() {
        let findings = vec![
            FindingSummary { severity: "CRITICAL".into(), title: "a".into(), service: "s".into(), description: "d".into() },
            FindingSummary { severity: "HIGH".into(), title: "b".into(), service: "s".into(), description: "d".into() },
            FindingSummary { severity: "HIGH".into(), title: "c".into(), service: "s".into(), description: "d".into() },
            FindingSummary { severity: "MEDIUM".into(), title: "d".into(), service: "s".into(), description: "d".into() },
        ];
        let (c, h, m) = count_by_severity(&findings);
        assert_eq!(c, 1);
        assert_eq!(h, 2);
        assert_eq!(m, 1);
    }

    #[test]
    fn count_by_severity_empty() {
        let (c, h, m) = count_by_severity(&[]);
        assert_eq!(c, 0);
        assert_eq!(h, 0);
        assert_eq!(m, 0);
    }

    // ── generate_synthetic_topology ────────────────────────────────────

    #[test]
    fn synthetic_chain_topology() {
        let (adj, cap, dl, names) = generate_synthetic_topology(&BenchmarkTopology::Chain, 5);
        assert_eq!(names.len(), 5);
        assert_eq!(adj.len(), 4); // 5 nodes → 4 edges in a chain
        assert_eq!(cap.len(), 5);
        assert_eq!(dl.len(), 5);
    }

    #[test]
    fn synthetic_fanout_topology() {
        let (adj, _, _, names) = generate_synthetic_topology(&BenchmarkTopology::FanOut, 5);
        assert_eq!(names.len(), 5);
        assert_eq!(adj.len(), 4); // root → 4 others
    }

    #[test]
    fn synthetic_star_topology() {
        let (adj, _, _, names) = generate_synthetic_topology(&BenchmarkTopology::Star, 5);
        assert_eq!(names.len(), 5);
        // 4 spokes × 2 (bidirectional) = 8
        assert_eq!(adj.len(), 8);
    }

    #[test]
    fn synthetic_mesh_topology() {
        let (adj, _, _, names) = generate_synthetic_topology(&BenchmarkTopology::Mesh, 5);
        assert_eq!(names.len(), 5);
        assert!(!adj.is_empty());
    }

    #[test]
    fn synthetic_random_topology() {
        let (adj, _, _, names) = generate_synthetic_topology(&BenchmarkTopology::Random, 10);
        assert_eq!(names.len(), 10);
        assert!(!adj.is_empty());
    }

    #[test]
    fn synthetic_empty_topology() {
        let (adj, _, _, names) = generate_synthetic_topology(&BenchmarkTopology::Chain, 1);
        assert_eq!(names.len(), 1);
        assert!(adj.is_empty());
    }

    // ── load_yaml_files ────────────────────────────────────────────────

    #[test]
    fn load_yaml_files_reads_single_file() {
        let dir = std::env::temp_dir().join("cascade_handler_test_load");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.yaml");
        std::fs::write(&path, "kind: Service\nmetadata:\n  name: test-svc\n").unwrap();
        let result = load_yaml_files(&[path.to_string_lossy().into_owned()]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_yaml_files_reads_directory() {
        let dir = std::env::temp_dir().join("cascade_handler_test_dir");
        let _ = std::fs::create_dir_all(&dir);
        let f1 = dir.join("a.yaml");
        let f2 = dir.join("b.yml");
        let f3 = dir.join("c.txt"); // should be ignored
        std::fs::write(&f1, "kind: Service\nmetadata:\n  name: a\n").unwrap();
        std::fs::write(&f2, "kind: Service\nmetadata:\n  name: b\n").unwrap();
        std::fs::write(&f3, "not yaml").unwrap();
        let result = load_yaml_files(&[dir.to_string_lossy().into_owned()]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 2);
        let _ = std::fs::remove_file(&f1);
        let _ = std::fs::remove_file(&f2);
        let _ = std::fs::remove_file(&f3);
    }

    #[test]
    fn load_yaml_files_error_on_missing_path() {
        let result = load_yaml_files(&["/nonexistent/path.yaml".into()]);
        assert!(result.is_err());
    }

    // ── tier analysis / repair (self-contained) ──────────────────────

    #[test]
    fn amplification_analysis_detects_chain_risk() {
        let adjacency = vec![
            ("a".into(), "b".into(), 3u32, 5000u64, 1000u64),
            ("b".into(), "c".into(), 3, 5000, 1000),
        ];
        let capacities: HashMap<String, u64> =
            [("a", 1000), ("b", 1000), ("c", 1000)]
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect();
        let names = vec!["a".into(), "b".into(), "c".into()];
        let findings = run_amplification_analysis(&adjacency, &capacities, &names);
        assert!(!findings.is_empty());
        assert!(findings.iter().any(|f| f.title.contains("amplification")));
    }

    #[test]
    fn amplification_analysis_safe_config_no_findings() {
        let adjacency = vec![
            ("a".into(), "b".into(), 1u32, 1000u64, 1000u64),
        ];
        let capacities: HashMap<String, u64> =
            [("a", 1000), ("b", 1000)]
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect();
        let names = vec!["a".into(), "b".into()];
        let findings = run_amplification_analysis(&adjacency, &capacities, &names);
        // (1+1)=2x is below 4x threshold
        let amp_findings: Vec<_> = findings
            .iter()
            .filter(|f| f.title.contains("amplification cascade"))
            .collect();
        assert!(amp_findings.is_empty());
    }

    #[test]
    fn synthesize_repairs_produces_plans() {
        let findings = vec![RiskFinding {
            severity: "CRITICAL".into(),
            title: "Retry amplification cascade".into(),
            service: "a".into(),
            description: "Amplification factor 16x on path: a → b → c".into(),
        }];
        let adjacency = vec![
            ("a".into(), "b".into(), 3u32, 5000u64, 1000u64),
            ("b".into(), "c".into(), 4, 5000, 1000),
        ];
        let plans = synthesize_repairs(&findings, &adjacency, 10, 5);
        assert!(!plans.is_empty());
        assert!(plans[0].description.contains("Reduce retries"));
    }

    #[test]
    fn synthesize_repairs_empty_when_no_risks() {
        let plans = synthesize_repairs(&[], &[], 10, 5);
        assert!(plans.is_empty());
    }

    // ── findings_to_summaries ──────────────────────────────────────────

    #[test]
    fn findings_to_summaries_preserves_data() {
        let findings = vec![RiskFinding {
            severity: "HIGH".into(),
            title: "Test risk".into(),
            service: "svc-x".into(),
            description: "A test description".into(),
        }];
        let summaries = findings_to_summaries(&findings);
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].severity, "HIGH");
        assert_eq!(summaries[0].title, "Test risk");
    }

    // ── severity_rank ──────────────────────────────────────────────────

    #[test]
    fn severity_ranking() {
        assert!(severity_rank("CRITICAL") < severity_rank("HIGH"));
        assert!(severity_rank("HIGH") < severity_rank("MEDIUM"));
        assert!(severity_rank("MEDIUM") < severity_rank("LOW"));
    }
}
