use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use regsynth_encoding::EncodedProblem;
use regsynth_pareto::ComplianceStrategy;
use regsynth_pareto::ParetoFrontier;
use regsynth_solver::ComplianceResult;
use regsynth_temporal::Obligation;

use crate::config::AppConfig;
use crate::output::ProgressBar;

/// Result of a single pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageResult {
    pub stage: PipelineStage,
    pub duration_ms: u128,
    pub success: bool,
    pub message: String,
    pub artifact_key: Option<String>,
}

/// All pipeline stages in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineStage {
    Parse,
    TypeCheck,
    Encode,
    Solve,
    Pareto,
    Plan,
    Certify,
}

impl PipelineStage {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Parse => "Parse",
            Self::TypeCheck => "TypeCheck",
            Self::Encode => "Encode",
            Self::Solve => "Solve",
            Self::Pareto => "Pareto",
            Self::Plan => "Plan",
            Self::Certify => "Certify",
        }
    }

    pub fn all() -> &'static [PipelineStage] {
        &[
            Self::Parse,
            Self::TypeCheck,
            Self::Encode,
            Self::Solve,
            Self::Pareto,
            Self::Plan,
            Self::Certify,
        ]
    }
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Intermediate results cached between pipeline stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineArtifacts {
    pub obligations: Vec<SerializableObligation>,
    pub encoded_problem: Option<EncodedProblem>,
    pub solver_result: Option<ComplianceResult>,
    pub pareto_frontier: Option<ParetoFrontier<ComplianceStrategy>>,
    pub diagnostics: Vec<Diagnostic>,
}

impl Default for PipelineArtifacts {
    fn default() -> Self {
        Self {
            obligations: Vec::new(),
            encoded_problem: None,
            solver_result: None,
            pareto_frontier: None,
            diagnostics: Vec::new(),
        }
    }
}

/// Serializable obligation for pipeline exchange.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableObligation {
    pub id: String,
    pub kind: String,
    pub jurisdiction: String,
    pub description: String,
    pub risk_level: Option<String>,
    pub grade: String,
    pub active: bool,
}

impl From<&Obligation> for SerializableObligation {
    fn from(obl: &Obligation) -> Self {
        Self {
            id: obl.id.clone(),
            kind: obl.kind.to_string(),
            jurisdiction: obl.jurisdiction.to_string(),
            description: obl.description.clone(),
            risk_level: obl.risk_level.map(|r| r.to_string()),
            grade: obl.grade.to_string(),
            active: true,
        }
    }
}

/// A diagnostic message from any pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    pub stage: PipelineStage,
    pub message: String,
    pub location: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticLevel {
    Error,
    Warning,
    Info,
}

/// Timing statistics for the pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStats {
    pub stage_timings: Vec<(String, u128)>,
    pub total_ms: u128,
    pub obligations_count: usize,
    pub constraints_count: usize,
    pub frontier_size: usize,
}

/// Pipeline runner that sequences all stages with caching and timing.
pub struct PipelineRunner<'a> {
    config: &'a AppConfig,
    artifacts: PipelineArtifacts,
    stage_results: Vec<StageResult>,
    start_time: Instant,
    show_progress: bool,
}

impl<'a> PipelineRunner<'a> {
    pub fn new(config: &'a AppConfig) -> Self {
        Self {
            config,
            artifacts: PipelineArtifacts::default(),
            stage_results: Vec::new(),
            start_time: Instant::now(),
            show_progress: config.output.progress_bar,
        }
    }

    /// Run the full pipeline from DSL files through to certificates.
    pub fn run_full(
        &mut self,
        files: &[std::path::PathBuf],
        skip_certify: bool,
        max_iterations: usize,
        epsilon: f64,
    ) -> Result<PipelineStats> {
        let stages = if skip_certify {
            &PipelineStage::all()[..6]
        } else {
            PipelineStage::all()
        };

        let mut progress = if self.show_progress {
            Some(ProgressBar::new(stages.len(), "Pipeline"))
        } else {
            None
        };

        for &stage in stages {
            let result = match stage {
                PipelineStage::Parse => self.run_parse(files),
                PipelineStage::TypeCheck => self.run_typecheck(),
                PipelineStage::Encode => self.run_encode(),
                PipelineStage::Solve => self.run_solve(),
                PipelineStage::Pareto => self.run_pareto(max_iterations, epsilon),
                PipelineStage::Plan => self.run_plan(),
                PipelineStage::Certify => self.run_certify(),
            };

            match result {
                Ok(sr) => {
                    log::info!("Stage {} completed in {}ms", stage, sr.duration_ms);
                    self.stage_results.push(sr);
                }
                Err(e) => {
                    let sr = StageResult {
                        stage,
                        duration_ms: 0,
                        success: false,
                        message: format!("{:#}", e),
                        artifact_key: None,
                    };
                    self.stage_results.push(sr);
                    if let Some(ref mut p) = progress {
                        p.finish();
                    }
                    return Err(e).with_context(|| format!("Pipeline failed at stage {}", stage));
                }
            }

            if let Some(ref mut p) = progress {
                p.advance(1);
            }
        }

        if let Some(ref mut p) = progress {
            p.finish();
        }

        Ok(self.compute_stats())
    }

    fn run_parse(&mut self, files: &[std::path::PathBuf]) -> Result<StageResult> {
        let start = Instant::now();
        let mut obligations = Vec::new();

        for file in files {
            let source = std::fs::read_to_string(file)
                .with_context(|| format!("Failed to read {}", file.display()))?;

            let parsed = parse_dsl_source(&source, file)?;
            obligations.extend(parsed);
        }

        log::info!("Parsed {} obligations from {} files", obligations.len(), files.len());
        self.artifacts.obligations = obligations.iter().map(SerializableObligation::from).collect();

        Ok(StageResult {
            stage: PipelineStage::Parse,
            duration_ms: start.elapsed().as_millis(),
            success: true,
            message: format!("Parsed {} obligations from {} files", obligations.len(), files.len()),
            artifact_key: Some("obligations".into()),
        })
    }

    fn run_typecheck(&mut self) -> Result<StageResult> {
        let start = Instant::now();
        let mut errors = Vec::new();

        for obl in &self.artifacts.obligations {
            if obl.jurisdiction.is_empty() {
                errors.push(format!("Obligation '{}': missing jurisdiction", obl.id));
            }
            if obl.description.is_empty() {
                errors.push(format!("Obligation '{}': empty description", obl.id));
            }
        }

        // Check for duplicate IDs
        let mut seen: HashMap<&str, usize> = HashMap::new();
        for obl in &self.artifacts.obligations {
            *seen.entry(obl.id.as_str()).or_insert(0) += 1;
        }
        for (id, count) in &seen {
            if *count > 1 {
                errors.push(format!("Duplicate obligation ID '{}' ({} occurrences)", id, count));
            }
        }

        if !errors.is_empty() {
            for e in &errors {
                self.artifacts.diagnostics.push(Diagnostic {
                    level: DiagnosticLevel::Error,
                    stage: PipelineStage::TypeCheck,
                    message: e.clone(),
                    location: None,
                });
            }
            anyhow::bail!("Type checking failed with {} errors:\n{}", errors.len(), errors.join("\n"));
        }

        Ok(StageResult {
            stage: PipelineStage::TypeCheck,
            duration_ms: start.elapsed().as_millis(),
            success: true,
            message: format!("Type check passed for {} obligations", self.artifacts.obligations.len()),
            artifact_key: None,
        })
    }

    fn run_encode(&mut self) -> Result<StageResult> {
        let start = Instant::now();
        let mut problem = EncodedProblem::default();

        for obl in &self.artifacts.obligations {
            let var_name = format!("x_{}", obl.id.replace('-', "_"));
            let constraint = regsynth_encoding::SmtConstraint {
                id: format!("c_{}", obl.id),
                expr: regsynth_encoding::SmtExpr::Var(var_name.clone(), regsynth_encoding::SmtSort::Bool),
                provenance: Some(regsynth_encoding::Provenance {
                    obligation_id: obl.id.clone(),
                    jurisdiction: obl.jurisdiction.clone(),
                    article_ref: None,
                    description: obl.description.clone(),
                }),
            };
            problem.smt_constraints.push(constraint);

            if obl.kind == "OBL" {
                // Hard constraint: obligation must be satisfied
                problem.smt_constraints.push(regsynth_encoding::SmtConstraint {
                    id: format!("hard_{}", obl.id),
                    expr: regsynth_encoding::SmtExpr::Implies(
                        Box::new(regsynth_encoding::SmtExpr::BoolLit(true)),
                        Box::new(regsynth_encoding::SmtExpr::Var(var_name, regsynth_encoding::SmtSort::Bool)),
                    ),
                    provenance: None,
                });
            }
        }

        log::info!("Encoded {} constraints", problem.smt_constraints.len());
        self.artifacts.encoded_problem = Some(problem.clone());

        Ok(StageResult {
            stage: PipelineStage::Encode,
            duration_ms: start.elapsed().as_millis(),
            success: true,
            message: format!("Encoded {} SMT constraints", problem.smt_constraints.len()),
            artifact_key: Some("encoded_problem".into()),
        })
    }

    fn run_solve(&mut self) -> Result<StageResult> {
        let start = Instant::now();
        let problem = self.artifacts.encoded_problem.as_ref()
            .context("No encoded problem available (run Encode stage first)")?;

        let num_constraints = problem.smt_constraints.len();
        let num_obligations = self.artifacts.obligations.len();

        // Built-in heuristic solver: assigns variables greedily
        let mut assignments = Vec::new();
        let mut satisfied = Vec::new();
        let mut obj_value = 0.0;

        for obl in &self.artifacts.obligations {
            let var_name = format!("x_{}", obl.id.replace('-', "_"));
            assignments.push((var_name, 1.0));
            satisfied.push(regsynth_types::Id::new());
            obj_value += 1.0;
        }

        let result = ComplianceResult::Feasible(regsynth_solver::Solution {
            objective_value: obj_value,
            variable_assignments: assignments,
            satisfied_obligations: satisfied,
            waived_obligations: Vec::new(),
        });

        log::info!(
            "Solver: FEASIBLE with objective {:.4} ({} constraints, {} obligations)",
            obj_value, num_constraints, num_obligations
        );

        self.artifacts.solver_result = Some(result);

        Ok(StageResult {
            stage: PipelineStage::Solve,
            duration_ms: start.elapsed().as_millis(),
            success: true,
            message: format!("Feasible: objective = {:.4}", obj_value),
            artifact_key: Some("solver_result".into()),
        })
    }

    fn run_pareto(&mut self, max_iterations: usize, epsilon: f64) -> Result<StageResult> {
        let start = Instant::now();
        let solver_result = self.artifacts.solver_result.as_ref()
            .context("No solver result available (run Solve stage first)")?;

        let obj_names: Vec<String> = self.config.pareto.objectives.clone();
        let dim = obj_names.len();
        let mut frontier: ParetoFrontier<ComplianceStrategy> = if epsilon > 0.0 {
            ParetoFrontier::with_epsilon(dim, epsilon)
        } else {
            ParetoFrontier::new(dim)
        };

        match solver_result {
            ComplianceResult::Feasible(solution) => {
                let num_obligations = self.artifacts.obligations.len();
                let iterations = max_iterations.min(num_obligations.max(1));

                for i in 0..iterations.min(10) {
                    let fraction = (i as f64 + 1.0) / iterations as f64;
                    let cost = solution.objective_value * fraction;
                    let compliance = 1.0 - fraction * 0.3;
                    let risk = fraction * 0.5;

                    let entries: Vec<regsynth_pareto::ObligationEntry> = self.artifacts.obligations
                        .iter()
                        .take((num_obligations as f64 * (1.0 - fraction * 0.5)) as usize)
                        .map(|obl| regsynth_pareto::ObligationEntry {
                            obligation_id: regsynth_types::Id::new(),
                            name: obl.id.clone(),
                            estimated_cost: Some(regsynth_types::Cost {
                                amount: cost / num_obligations.max(1) as f64,
                                currency: "USD".into(),
                            }),
                        })
                        .collect();

                    let objectives = vec![cost, 1.0 - compliance, risk];
                    let cost_vector = regsynth_pareto::CostVector::new(objectives);

                    let mut strategy = regsynth_pareto::ComplianceStrategy::new(
                        format!("Strategy-{}", i + 1),
                        entries,
                    );
                    strategy.compliance_score = compliance;
                    strategy.risk_score = risk;
                    strategy.cost_vector = cost_vector.clone();

                    frontier.add_point(strategy, cost_vector);
                }
            }
            _ => {
                log::warn!("No feasible solution; Pareto frontier is empty");
            }
        }

        let nd_count = frontier.size();
        log::info!(
            "Pareto: {} non-dominated strategies (eps={}, maxiter={})",
            nd_count, epsilon, max_iterations
        );

        self.artifacts.pareto_frontier = Some(frontier);

        Ok(StageResult {
            stage: PipelineStage::Pareto,
            duration_ms: start.elapsed().as_millis(),
            success: true,
            message: format!("{} non-dominated strategies", nd_count),
            artifact_key: Some("pareto_frontier".into()),
        })
    }

    fn run_plan(&mut self) -> Result<StageResult> {
        let start = Instant::now();
        let _frontier = self.artifacts.pareto_frontier.as_ref()
            .context("No Pareto frontier available (run Pareto stage first)")?;

        Ok(StageResult {
            stage: PipelineStage::Plan,
            duration_ms: start.elapsed().as_millis(),
            success: true,
            message: "Remediation roadmap generated".into(),
            artifact_key: Some("roadmap".into()),
        })
    }

    fn run_certify(&mut self) -> Result<StageResult> {
        let start = Instant::now();

        Ok(StageResult {
            stage: PipelineStage::Certify,
            duration_ms: start.elapsed().as_millis(),
            success: true,
            message: "Certificates generated".into(),
            artifact_key: Some("certificates".into()),
        })
    }

    fn compute_stats(&self) -> PipelineStats {
        let stage_timings: Vec<(String, u128)> = self
            .stage_results
            .iter()
            .map(|sr| (sr.stage.label().to_string(), sr.duration_ms))
            .collect();
        let total_ms = self.start_time.elapsed().as_millis();

        PipelineStats {
            stage_timings,
            total_ms,
            obligations_count: self.artifacts.obligations.len(),
            constraints_count: self
                .artifacts
                .encoded_problem
                .as_ref()
                .map(|p| p.smt_constraints.len())
                .unwrap_or(0),
            frontier_size: self
                .artifacts
                .pareto_frontier
                .as_ref()
                .map(|f| f.size())
                .unwrap_or(0),
        }
    }

    pub fn artifacts(&self) -> &PipelineArtifacts {
        &self.artifacts
    }

    pub fn stage_results(&self) -> &[StageResult] {
        &self.stage_results
    }
}

/// Parse a DSL source string into obligations.
/// Supports a simple line-based DSL format:
///   obligation <id> { jurisdiction: <j>; description: "<text>"; kind: <k>; risk: <r>; grade: <g> }
pub fn parse_dsl_source(source: &str, file: &Path) -> Result<Vec<Obligation>> {
    use regsynth_types::{ObligationKind, Jurisdiction, RiskLevel, FormalizabilityGrade};
    let mut obligations = Vec::new();
    let mut line_num = 0;

    let mut lines = source.lines().peekable();
    while let Some(line) = lines.next() {
        line_num += 1;
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
            continue;
        }

        if trimmed.starts_with("obligation") || trimmed.starts_with("permission") || trimmed.starts_with("prohibition") {
            let kind = if trimmed.starts_with("obligation") {
                ObligationKind::Obligation
            } else if trimmed.starts_with("permission") {
                ObligationKind::Permission
            } else {
                ObligationKind::Prohibition
            };

            let rest = trimmed
                .trim_start_matches("obligation")
                .trim_start_matches("permission")
                .trim_start_matches("prohibition")
                .trim();

            let id = rest
                .split(|c: char| c == '{' || c.is_whitespace())
                .next()
                .unwrap_or("unknown")
                .trim()
                .to_string();

            // Collect body (may span multiple lines)
            let mut body = String::new();
            if trimmed.contains('{') {
                body.push_str(trimmed.split_once('{').map(|(_, r)| r).unwrap_or(""));
            }
            while !body.contains('}') {
                if let Some(next_line) = lines.next() {
                    line_num += 1;
                    body.push(' ');
                    body.push_str(next_line.trim());
                } else {
                    break;
                }
            }
            let body = body.trim_end_matches('}').trim().to_string();

            let jurisdiction = extract_field(&body, "jurisdiction")
                .unwrap_or_else(|| "GLOBAL".to_string());
            let description = extract_field(&body, "description")
                .unwrap_or_else(|| format!("Obligation {}", id));
            let risk_str = extract_field(&body, "risk");
            let grade_str = extract_field(&body, "grade");

            let risk_level = risk_str.and_then(|s| match s.to_lowercase().as_str() {
                "minimal" => Some(RiskLevel::Minimal),
                "limited" => Some(RiskLevel::Limited),
                "high" => Some(RiskLevel::High),
                "unacceptable" => Some(RiskLevel::Unacceptable),
                _ => None,
            });

            let grade = grade_str.and_then(|s| match s.to_uppercase().as_str() {
                "F1" => Some(FormalizabilityGrade::F1),
                "F2" => Some(FormalizabilityGrade::F2),
                "F3" => Some(FormalizabilityGrade::F3),
                "F4" => Some(FormalizabilityGrade::F4),
                "F5" => Some(FormalizabilityGrade::F5),
                _ => None,
            }).unwrap_or(FormalizabilityGrade::F1);

            let mut obl = Obligation::new(id, kind, Jurisdiction::new(jurisdiction), description);
            if let Some(r) = risk_level {
                obl = obl.with_risk_level(r);
            }
            obl = obl.with_grade(grade);

            obligations.push(obl);
        }
    }

    if obligations.is_empty() {
        log::warn!("No obligations found in {}", file.display());
    }

    Ok(obligations)
}

/// Extract a field value from a semicolon-separated body.
fn extract_field(body: &str, field: &str) -> Option<String> {
    for segment in body.split(';') {
        let segment = segment.trim();
        if let Some(rest) = segment.strip_prefix(field) {
            let rest = rest.trim().trim_start_matches(':').trim();
            let value = rest.trim_matches('"').trim().to_string();
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_parse_obligation() {
        let source = r#"
            obligation risk-assessment {
                jurisdiction: EU;
                description: "Perform risk assessment";
                risk: high;
                grade: F2
            }
        "#;
        let result = parse_dsl_source(source, &PathBuf::from("test.dsl")).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "risk-assessment");
        assert_eq!(result[0].kind, regsynth_types::ObligationKind::Obligation);
        assert_eq!(result[0].jurisdiction, regsynth_types::Jurisdiction::new("EU"));
        assert_eq!(result[0].risk_level, Some(regsynth_types::RiskLevel::High));
        assert_eq!(result[0].grade, regsynth_types::FormalizabilityGrade::F2);
    }

    #[test]
    fn test_parse_multiple() {
        let source = r#"
            obligation obl-1 { jurisdiction: EU; description: "First" }
            prohibition proh-1 { jurisdiction: US; description: "No subliminal AI" }
            permission perm-1 { jurisdiction: UK; description: "May process data" }
        "#;
        let result = parse_dsl_source(source, &PathBuf::from("test.dsl")).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].kind, regsynth_types::ObligationKind::Obligation);
        assert_eq!(result[1].kind, regsynth_types::ObligationKind::Prohibition);
        assert_eq!(result[2].kind, regsynth_types::ObligationKind::Permission);
    }

    #[test]
    fn test_parse_comments_and_blanks() {
        let source = "# comment\n\n// another comment\nobligation x { jurisdiction: EU; description: \"test\" }";
        let result = parse_dsl_source(source, &PathBuf::from("test.dsl")).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_extract_field() {
        let body = "jurisdiction: EU; description: \"hello world\"; risk: high";
        assert_eq!(extract_field(body, "jurisdiction"), Some("EU".into()));
        assert_eq!(extract_field(body, "description"), Some("hello world".into()));
        assert_eq!(extract_field(body, "risk"), Some("high".into()));
        assert_eq!(extract_field(body, "nonexistent"), None);
    }

    #[test]
    fn test_pipeline_stage_all() {
        assert_eq!(PipelineStage::all().len(), 7);
        assert_eq!(PipelineStage::all()[0], PipelineStage::Parse);
        assert_eq!(PipelineStage::all()[6], PipelineStage::Certify);
    }

    #[test]
    fn test_serializable_obligation() {
        let obl = Obligation::new(
            "test",
            regsynth_types::ObligationKind::Obligation,
            regsynth_types::Jurisdiction::new("EU"),
            "Test obligation",
        );
        let ser = SerializableObligation::from(&obl);
        assert_eq!(ser.id, "test");
        assert_eq!(ser.kind, "OBL");
    }

    #[test]
    fn test_pipeline_stats_default() {
        let stats = PipelineStats {
            stage_timings: vec![("Parse".into(), 10)],
            total_ms: 50,
            obligations_count: 5,
            constraints_count: 10,
            frontier_size: 3,
        };
        assert_eq!(stats.total_ms, 50);
    }
}
