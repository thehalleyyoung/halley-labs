//! Resilience policy analysis – retry amplification, timeout chain
//! feasibility, effective budgets, best-practice checking, and
//! overall resilience scoring.

use std::collections::HashMap;

use cascade_graph::rtig::RtigGraph;
use cascade_types::topology::DependencyType;
use serde::{Deserialize, Serialize};

use crate::mesh::ServiceMesh;

// ── RetryAnalysis ───────────────────────────────────────────────────

/// Per-service retry assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryAssessment {
    pub service: String,
    pub configured_retries: u32,
    pub effective_retries: u32,
    pub amplification_contribution: f64,
    pub recommendation: String,
}

/// Aggregate retry analysis across the mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryAnalysis {
    pub per_service: HashMap<String, RetryAssessment>,
    pub overall_score: f64,
}

// ── TimeoutAnalysis ─────────────────────────────────────────────────

/// Per-service timeout assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutAssessment {
    pub service: String,
    pub local_timeout_ms: u64,
    pub effective_timeout_ms: u64,
    pub chain_contribution_ms: u64,
    pub recommendation: String,
}

/// A timeout chain from a root to a leaf with feasibility check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutChain {
    pub path: Vec<String>,
    pub total_ms: u64,
    pub deadline_ms: u64,
    pub feasible: bool,
}

/// Aggregate timeout analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutAnalysis {
    pub per_service: HashMap<String, TimeoutAssessment>,
    pub chains: Vec<TimeoutChain>,
}

// ── EffectiveRetryBudget / EffectiveTimeout ─────────────────────────

/// Effective retry budget accounting for amplification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveRetryBudget {
    pub local_retries: u32,
    pub amplified_retries: f64,
    pub total_load_factor: f64,
}

/// Effective timeout including chained downstream timeouts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveTimeout {
    pub local_timeout_ms: u64,
    pub chain_timeout_ms: u64,
    pub deadline_ms: u64,
    pub feasible: bool,
}

// ── ResilienceScore ─────────────────────────────────────────────────

/// Overall resilience assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResilienceScore {
    pub retry_score: f64,
    pub timeout_score: f64,
    pub circuit_breaker_score: f64,
    pub overall: f64,
    pub summary: String,
}

// ── ResilienceAnalyzer ──────────────────────────────────────────────

/// Main entry point for resilience analysis.
pub struct ResilienceAnalyzer;

impl ResilienceAnalyzer {
    /// Analyse retry policies across the mesh.
    pub fn analyze_retry_policies(mesh: &ServiceMesh) -> RetryAnalysis {
        let graph = mesh.graph();
        let ids: Vec<String> = graph.service_ids().iter().map(|s| s.to_string()).collect();

        let mut per_service = HashMap::new();
        let mut total_score = 0.0_f64;

        for id in &ids {
            let outgoing = graph.outgoing_edges(id);
            let configured: u32 = outgoing.iter().map(|e| e.retry_count).max().unwrap_or(0);
            let effective = Self::compute_effective_retries(&graph, id);
            let amp = Self::amplification_at(&graph, id);

            let recommendation = if amp > 10.0 {
                format!("CRITICAL: amplification {:.1}x – reduce retries or add circuit breaker", amp)
            } else if amp > 5.0 {
                format!("WARNING: amplification {:.1}x – consider reducing retry count", amp)
            } else if configured == 0 {
                "INFO: no retries configured – consider adding for transient failures".into()
            } else {
                "OK".into()
            };

            let svc_score = if amp <= 3.0 { 1.0 } else { 3.0 / amp };

            per_service.insert(
                id.clone(),
                RetryAssessment {
                    service: id.clone(),
                    configured_retries: configured,
                    effective_retries: effective,
                    amplification_contribution: amp,
                    recommendation,
                },
            );
            total_score += svc_score;
        }

        let count = ids.len().max(1) as f64;
        RetryAnalysis {
            per_service,
            overall_score: total_score / count,
        }
    }

    fn compute_effective_retries(graph: &RtigGraph, service: &str) -> u32 {
        let outgoing = graph.outgoing_edges(service);
        let local_max: u32 = outgoing.iter().map(|e| e.retry_count).max().unwrap_or(0);

        let child_max: u32 = graph
            .successors(service)
            .iter()
            .map(|s| Self::compute_effective_retries(graph, s))
            .max()
            .unwrap_or(0);

        local_max.saturating_add(child_max)
    }

    fn amplification_at(graph: &RtigGraph, service: &str) -> f64 {
        let outgoing = graph.outgoing_edges(service);
        if outgoing.is_empty() {
            return 1.0;
        }
        let mut total = 0.0_f64;
        for edge in &outgoing {
            let child_amp = Self::amplification_at(graph, edge.target.as_str());
            total += edge.amplification_factor_f64() * child_amp;
        }
        total.max(1.0)
    }

    /// Analyse timeout policies across the mesh.
    pub fn analyze_timeout_policies(mesh: &ServiceMesh) -> TimeoutAnalysis {
        let graph = mesh.graph();
        let ids: Vec<String> = graph.service_ids().iter().map(|s| s.to_string()).collect();

        let mut per_service = HashMap::new();
        for id in &ids {
            let node = graph.service(id);
            let local_timeout = node.map(|n| n.timeout_ms).unwrap_or(0);
            let chain_timeout = Self::compute_chain_timeout(&graph, id);
            let effective = chain_timeout.max(local_timeout);

            let recommendation = if chain_timeout > local_timeout && local_timeout > 0 {
                format!(
                    "WARNING: chain timeout ({} ms) exceeds local timeout ({} ms)",
                    chain_timeout, local_timeout
                )
            } else {
                "OK".into()
            };

            per_service.insert(
                id.clone(),
                TimeoutAssessment {
                    service: id.clone(),
                    local_timeout_ms: local_timeout,
                    effective_timeout_ms: effective,
                    chain_contribution_ms: chain_timeout,
                    recommendation,
                },
            );
        }

        let chains = Self::enumerate_timeout_chains(&graph);

        TimeoutAnalysis {
            per_service,
            chains,
        }
    }

    fn compute_chain_timeout(graph: &RtigGraph, service: &str) -> u64 {
        let outgoing = graph.outgoing_edges(service);
        if outgoing.is_empty() {
            return 0;
        }
        outgoing
            .iter()
            .map(|e| {
                let child_chain = Self::compute_chain_timeout(graph, e.target.as_str());
                let retries = e.retry_count.max(1) as u64;
                e.timeout_ms * retries + child_chain
            })
            .max()
            .unwrap_or(0)
    }

    fn enumerate_timeout_chains(graph: &RtigGraph) -> Vec<TimeoutChain> {
        let roots: Vec<String> = graph.roots().iter().map(|s| s.to_string()).collect();
        let leaves: Vec<String> = graph.leaves().iter().map(|s| s.to_string()).collect();
        let mut chains = Vec::new();

        for root in &roots {
            let root_deadline = graph.service(root).map(|n| n.timeout_ms).unwrap_or(30_000);
            for leaf in &leaves {
                let paths = enumerate_simple_paths(graph, root, leaf);
                for path in paths {
                    let total = Self::chain_total_timeout(graph, &path);
                    chains.push(TimeoutChain {
                        path,
                        total_ms: total,
                        deadline_ms: root_deadline,
                        feasible: total <= root_deadline,
                    });
                }
            }
        }
        chains
    }

    fn chain_total_timeout(graph: &RtigGraph, path: &[String]) -> u64 {
        let mut total = 0u64;
        for w in path.windows(2) {
            let edges = graph.outgoing_edges(&w[0]);
            let hop = edges
                .iter()
                .filter(|e| e.target.as_str() == w[1])
                .map(|e| e.timeout_ms * e.retry_count.max(1) as u64)
                .max()
                .unwrap_or(0);
            total += hop;
        }
        total
    }

    /// Compute effective retry budget for a specific service.
    pub fn compute_effective_retry_budget(
        mesh: &ServiceMesh,
        service: &str,
    ) -> EffectiveRetryBudget {
        let graph = mesh.graph();
        let local = graph
            .outgoing_edges(service)
            .iter()
            .map(|e| e.retry_count)
            .max()
            .unwrap_or(0);
        let amp = Self::amplification_at(&graph, service);
        let total_factor = (local as f64 + 1.0) * amp;

        EffectiveRetryBudget {
            local_retries: local,
            amplified_retries: amp,
            total_load_factor: total_factor,
        }
    }

    /// Compute effective timeout for a specific service.
    pub fn compute_effective_timeout(
        mesh: &ServiceMesh,
        service: &str,
    ) -> EffectiveTimeout {
        let graph = mesh.graph();
        let node = graph.service(service);
        let local = node.map(|n| n.timeout_ms).unwrap_or(0);
        let chain = Self::compute_chain_timeout(&graph, service);
        let deadline = local.max(chain);

        EffectiveTimeout {
            local_timeout_ms: local,
            chain_timeout_ms: chain,
            deadline_ms: deadline,
            feasible: chain <= local || local == 0,
        }
    }

    /// Compute overall resilience score for the mesh.
    pub fn compute_resilience_score(mesh: &ServiceMesh) -> ResilienceScore {
        let retry = Self::analyze_retry_policies(mesh);
        let timeout = Self::analyze_timeout_policies(mesh);
        let cb_score = Self::circuit_breaker_coverage(mesh);

        let retry_s = retry.overall_score;
        let timeout_s = Self::timeout_score(&timeout);
        let overall = (retry_s * 0.4 + timeout_s * 0.35 + cb_score * 0.25).min(1.0);

        let summary = if overall > 0.8 {
            "Good resilience posture".into()
        } else if overall > 0.5 {
            "Moderate resilience – review retry amplification and timeout chains".into()
        } else {
            "Poor resilience – significant risk of cascade failures".into()
        };

        ResilienceScore {
            retry_score: retry_s,
            timeout_score: timeout_s,
            circuit_breaker_score: cb_score,
            overall,
            summary,
        }
    }

    fn timeout_score(analysis: &TimeoutAnalysis) -> f64 {
        if analysis.chains.is_empty() {
            return 1.0;
        }
        let feasible = analysis.chains.iter().filter(|c| c.feasible).count();
        feasible as f64 / analysis.chains.len() as f64
    }

    fn circuit_breaker_coverage(mesh: &ServiceMesh) -> f64 {
        let graph = mesh.graph();
        let ids: Vec<String> = graph.service_ids().iter().map(|s| s.to_string()).collect();
        if ids.is_empty() {
            return 1.0;
        }

        // Without edge-level policies, count services that have no outgoing deps
        // (leaf services don't need circuit breakers) as "covered".
        let mut covered = 0usize;
        for id in &ids {
            if graph.fan_out(id) == 0 {
                covered += 1;
            }
        }
        covered as f64 / ids.len() as f64
    }
}

// ── BestPracticeChecker ─────────────────────────────────────────────

/// A best-practice recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestPractice {
    pub rule: String,
    pub severity: BpSeverity,
    pub message: String,
    pub service: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BpSeverity {
    Critical,
    Warning,
    Info,
}

/// Checks mesh policies against industry best practices.
pub struct BestPracticeChecker;

impl BestPracticeChecker {
    pub fn check(mesh: &ServiceMesh) -> Vec<BestPractice> {
        let mut findings = Vec::new();
        Self::check_retry_limits(mesh, &mut findings);
        Self::check_timeout_hierarchy(mesh, &mut findings);
        Self::check_circuit_breaker_presence(mesh, &mut findings);
        Self::check_backoff_strategy(mesh, &mut findings);
        Self::check_retry_idempotency(mesh, &mut findings);
        Self::check_total_amplification(mesh, &mut findings);
        findings
    }

    fn check_retry_limits(mesh: &ServiceMesh, findings: &mut Vec<BestPractice>) {
        let graph = mesh.graph();
        for edge in graph.edges() {
            if edge.retry_count > 5 {
                findings.push(BestPractice {
                    rule: "MAX_RETRIES".into(),
                    severity: BpSeverity::Warning,
                    message: format!(
                        "Edge {} -> {} has {} retries (recommended max: 5)",
                        edge.source.as_str(),
                        edge.target.as_str(),
                        edge.retry_count
                    ),
                    service: Some(edge.source.as_str().to_string()),
                });
            }
        }
    }

    fn check_timeout_hierarchy(mesh: &ServiceMesh, findings: &mut Vec<BestPractice>) {
        let graph = mesh.graph();
        for id in graph.service_ids() {
            let node = match graph.service(id) {
                Some(n) => n,
                None => continue,
            };
            let parent_timeout = node.timeout_ms;
            for edge in graph.outgoing_edges(id) {
                if edge.timeout_ms > parent_timeout && parent_timeout > 0 {
                    findings.push(BestPractice {
                        rule: "TIMEOUT_HIERARCHY".into(),
                        severity: BpSeverity::Critical,
                        message: format!(
                            "Child timeout ({} ms for {}->{}) exceeds parent timeout ({} ms)",
                            edge.timeout_ms,
                            edge.source.as_str(),
                            edge.target.as_str(),
                            parent_timeout
                        ),
                        service: Some(id.to_string()),
                    });
                }
            }
        }
    }

    fn check_circuit_breaker_presence(mesh: &ServiceMesh, findings: &mut Vec<BestPractice>) {
        let graph = mesh.graph();
        for id in graph.service_ids() {
            if graph.fan_out(id) > 0 {
                // Without edge-level policy data, always flag missing CB
                findings.push(BestPractice {
                    rule: "CIRCUIT_BREAKER_MISSING".into(),
                    severity: BpSeverity::Info,
                    message: format!(
                        "Service {} has outgoing dependencies but no circuit breaker",
                        id
                    ),
                    service: Some(id.to_string()),
                });
            }
        }
    }

    fn check_backoff_strategy(_mesh: &ServiceMesh, _findings: &mut Vec<BestPractice>) {
        // Edge-level backoff policies are not available in the current mesh model
    }

    fn check_retry_idempotency(mesh: &ServiceMesh, findings: &mut Vec<BestPractice>) {
        let graph = mesh.graph();
        for edge in graph.edges() {
            if edge.retry_count > 0 && edge.dep_type == DependencyType::Asynchronous {
                findings.push(BestPractice {
                    rule: "ASYNC_RETRY".into(),
                    severity: BpSeverity::Info,
                    message: format!(
                        "Async dependency {}->{} has retries – ensure idempotency",
                        edge.source.as_str(),
                        edge.target.as_str()
                    ),
                    service: Some(edge.source.as_str().to_string()),
                });
            }
        }
    }

    fn check_total_amplification(mesh: &ServiceMesh, findings: &mut Vec<BestPractice>) {
        let graph = mesh.graph();
        for root in graph.roots() {
            let amp = ResilienceAnalyzer::amplification_at(&graph, root);
            if amp > 100.0 {
                findings.push(BestPractice {
                    rule: "AMPLIFICATION_EXTREME".into(),
                    severity: BpSeverity::Critical,
                    message: format!(
                        "Root {} has total amplification factor {:.0}x",
                        root, amp
                    ),
                    service: Some(root.to_string()),
                });
            } else if amp > 20.0 {
                findings.push(BestPractice {
                    rule: "AMPLIFICATION_HIGH".into(),
                    severity: BpSeverity::Warning,
                    message: format!(
                        "Root {} has total amplification factor {:.0}x",
                        root, amp
                    ),
                    service: Some(root.to_string()),
                });
            }
        }
    }
}

// ── helpers ─────────────────────────────────────────────────────────

fn enumerate_simple_paths(graph: &RtigGraph, from: &str, to: &str) -> Vec<Vec<String>> {
    let mut results = Vec::new();
    let mut path = vec![from.to_string()];
    let mut visited = std::collections::HashSet::new();
    visited.insert(from.to_string());
    dfs_simple(graph, from, to, &mut visited, &mut path, &mut results);
    results
}

fn dfs_simple(
    graph: &RtigGraph,
    cur: &str,
    end: &str,
    visited: &mut std::collections::HashSet<String>,
    path: &mut Vec<String>,
    results: &mut Vec<Vec<String>>,
) {
    if cur == end && path.len() > 1 {
        results.push(path.clone());
        return;
    }
    for next in graph.successors(cur) {
        if !visited.contains(next) {
            visited.insert(next.to_string());
            path.push(next.to_string());
            dfs_simple(graph, next, end, visited, path, results);
            path.pop();
            visited.remove(next);
        }
    }
}

// ── ResilienceAnalyzer internal (pub(crate) for BestPracticeChecker)

impl ResilienceAnalyzer {
    pub(crate) fn amplification_at_pub(graph: &RtigGraph, service: &str) -> f64 {
        Self::amplification_at(graph, service)
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::MeshBuilder;
    use cascade_graph::rtig::ServiceNode;
    use cascade_types::topology::DependencyType;

    fn simple_mesh() -> ServiceMesh {
        MeshBuilder::new()
            .add_node(ServiceNode::new("gw", 1000).with_timeout_ms(5000))
            .unwrap()
            .add_node(ServiceNode::new("api", 500).with_timeout_ms(3000))
            .unwrap()
            .add_node(ServiceNode::new("db", 200).with_timeout_ms(1000))
            .unwrap()
            .add_dependency_with_retries("gw", "api", DependencyType::Synchronous, 3, 2000)
            .unwrap()
            .add_dependency_with_retries("api", "db", DependencyType::Synchronous, 2, 1000)
            .unwrap()
            .build()
            .unwrap()
    }

    fn high_retry_mesh() -> ServiceMesh {
        MeshBuilder::new()
            .add_node(ServiceNode::new("entry", 1000).with_timeout_ms(30000))
            .unwrap()
            .add_node(ServiceNode::new("mid", 500).with_timeout_ms(10000))
            .unwrap()
            .add_node(ServiceNode::new("leaf", 100).with_timeout_ms(2000))
            .unwrap()
            .add_dependency_with_retries("entry", "mid", DependencyType::Synchronous, 5, 5000)
            .unwrap()
            .add_dependency_with_retries("mid", "leaf", DependencyType::Synchronous, 5, 2000)
            .unwrap()
            .build()
            .unwrap()
    }

    // ── RetryAnalysis ──────

    #[test]
    fn retry_analysis_basic() {
        let mesh = simple_mesh();
        let analysis = ResilienceAnalyzer::analyze_retry_policies(&mesh);
        assert_eq!(analysis.per_service.len(), 3);
        assert!(analysis.overall_score > 0.0);
    }

    #[test]
    fn retry_analysis_effective_retries() {
        let mesh = simple_mesh();
        let analysis = ResilienceAnalyzer::analyze_retry_policies(&mesh);
        let gw = &analysis.per_service["gw"];
        assert!(gw.effective_retries >= gw.configured_retries);
    }

    #[test]
    fn retry_analysis_high_retries_warning() {
        let mesh = high_retry_mesh();
        let analysis = ResilienceAnalyzer::analyze_retry_policies(&mesh);
        let entry = &analysis.per_service["entry"];
        assert!(entry.amplification_contribution > 5.0);
        assert!(entry.recommendation.contains("WARNING") || entry.recommendation.contains("CRITICAL"));
    }

    // ── TimeoutAnalysis ────

    #[test]
    fn timeout_analysis_basic() {
        let mesh = simple_mesh();
        let analysis = ResilienceAnalyzer::analyze_timeout_policies(&mesh);
        assert_eq!(analysis.per_service.len(), 3);
    }

    #[test]
    fn timeout_chain_feasibility() {
        let mesh = simple_mesh();
        let analysis = ResilienceAnalyzer::analyze_timeout_policies(&mesh);
        // With retries, the chain gw->api->db should have total = 3*2000 + 2*1000 = 8000 ms
        // gw timeout is 5000, so chain may be infeasible
        assert!(!analysis.chains.is_empty());
    }

    #[test]
    fn timeout_chain_count() {
        let mesh = simple_mesh();
        let analysis = ResilienceAnalyzer::analyze_timeout_policies(&mesh);
        // One root (gw), one leaf (db), so one chain
        assert!(analysis.chains.len() >= 1);
    }

    // ── EffectiveRetryBudget ────

    #[test]
    fn effective_retry_budget() {
        let mesh = simple_mesh();
        let budget = ResilienceAnalyzer::compute_effective_retry_budget(&mesh, "gw");
        assert_eq!(budget.local_retries, 3);
        assert!(budget.amplified_retries >= 1.0);
        assert!(budget.total_load_factor >= 1.0);
    }

    // ── EffectiveTimeout ───

    #[test]
    fn effective_timeout() {
        let mesh = simple_mesh();
        let et = ResilienceAnalyzer::compute_effective_timeout(&mesh, "gw");
        assert_eq!(et.local_timeout_ms, 5000);
        assert!(et.chain_timeout_ms > 0);
    }

    #[test]
    fn effective_timeout_leaf() {
        let mesh = simple_mesh();
        let et = ResilienceAnalyzer::compute_effective_timeout(&mesh, "db");
        assert_eq!(et.chain_timeout_ms, 0);
        assert!(et.feasible);
    }

    // ── ResilienceScore ────

    #[test]
    fn resilience_score_range() {
        let mesh = simple_mesh();
        let score = ResilienceAnalyzer::compute_resilience_score(&mesh);
        assert!(score.overall >= 0.0 && score.overall <= 1.0);
        assert!(!score.summary.is_empty());
    }

    // ── BestPracticeChecker ────

    #[test]
    fn best_practices_simple() {
        let mesh = simple_mesh();
        let findings = BestPracticeChecker::check(&mesh);
        // Should have at least the circuit-breaker-missing info
        assert!(findings.iter().any(|f| f.rule == "CIRCUIT_BREAKER_MISSING"));
    }

    #[test]
    fn best_practices_high_retry() {
        let mesh = high_retry_mesh();
        let findings = BestPracticeChecker::check(&mesh);
        assert!(findings.iter().any(|f| f.rule == "MAX_RETRIES"));
    }

    #[test]
    fn best_practices_amplification() {
        let mesh = high_retry_mesh();
        let findings = BestPracticeChecker::check(&mesh);
        let amp_findings: Vec<_> = findings
            .iter()
            .filter(|f| f.rule.starts_with("AMPLIFICATION"))
            .collect();
        assert!(!amp_findings.is_empty());
    }

    #[test]
    fn best_practices_no_critical_on_clean() {
        // A mesh with no retries should have no critical findings
        let mesh = MeshBuilder::new()
            .add_node(ServiceNode::new("a", 100).with_timeout_ms(5000))
            .unwrap()
            .add_node(ServiceNode::new("b", 100).with_timeout_ms(3000))
            .unwrap()
            .add_dep("a", "b", DependencyType::Synchronous)
            .unwrap()
            .build()
            .unwrap();
        let findings = BestPracticeChecker::check(&mesh);
        let critical: Vec<_> = findings
            .iter()
            .filter(|f| f.severity == BpSeverity::Critical)
            .collect();
        assert!(critical.is_empty());
    }
}
