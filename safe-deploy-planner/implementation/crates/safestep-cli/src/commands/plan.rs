//! Implementation of the `plan` subcommand.

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::cli::PlanArgs;
use crate::config_loader::SafeStepConfig;
use crate::output::OutputManager;

// ---------------------------------------------------------------------------
// Domain types (self-contained for CLI layer)
// ---------------------------------------------------------------------------

/// Service descriptor loaded from manifests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceManifest {
    pub name: String,
    pub versions: Vec<String>,
    pub dependencies: Vec<String>,
    #[serde(default)]
    pub constraints: Vec<ConstraintDef>,
}

/// Constraint definition loaded from file or manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ConstraintDef {
    #[serde(rename = "compatibility")]
    Compatibility {
        service_a: String,
        service_b: String,
        compatible_pairs: Vec<(u16, u16)>,
    },
    #[serde(rename = "ordering")]
    Ordering { before: String, after: String },
    #[serde(rename = "forbidden")]
    Forbidden { service: String, version: u16 },
    #[serde(rename = "resource")]
    Resource {
        resource_name: String,
        max_budget: f64,
        costs: HashMap<String, f64>,
    },
}

/// A single step in a generated plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStepOutput {
    pub step: usize,
    pub service: String,
    pub from_version: String,
    pub to_version: String,
    pub risk_score: u32,
    pub estimated_duration_secs: u64,
    pub requires_downtime: bool,
}

/// Complete plan output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanOutput {
    pub plan_id: String,
    pub start_state: Vec<u16>,
    pub target_state: Vec<u16>,
    pub steps: Vec<PlanStepOutput>,
    pub total_risk: u32,
    pub total_duration_secs: u64,
    pub solver_time_ms: u64,
    pub solver_method: String,
    pub services: Vec<String>,
}

// ---------------------------------------------------------------------------
// PlanCommand
// ---------------------------------------------------------------------------

pub struct PlanCommand {
    args: PlanArgs,
    config: SafeStepConfig,
}

impl PlanCommand {
    pub fn new(args: PlanArgs, config: SafeStepConfig) -> Self {
        Self { args, config }
    }

    pub fn execute(&self, output: &mut OutputManager) -> Result<()> {
        info!("starting plan generation");

        let start_state = crate::cli::parse_state(&self.args.start_state)
            .context("invalid start state")?;
        let target_state = crate::cli::parse_state(&self.args.target_state)
            .context("invalid target state")?;

        if start_state.len() != target_state.len() {
            anyhow::bail!(
                "start state has {} services but target state has {}",
                start_state.len(),
                target_state.len()
            );
        }

        if start_state == target_state {
            output.writeln("Start and target states are identical. No plan needed.");
            return Ok(());
        }

        let manifests = self.load_manifests()?;
        let constraints = self.load_constraints(&manifests)?;

        if manifests.is_empty() {
            anyhow::bail!("no service manifests found in {}", self.args.manifest_dir.display());
        }

        if manifests.len() != start_state.len() {
            anyhow::bail!(
                "found {} service manifests but state has {} entries",
                manifests.len(),
                start_state.len()
            );
        }

        for (_i, (sv, m)) in start_state.iter().zip(manifests.iter()).enumerate() {
            if *sv as usize >= m.versions.len() {
                anyhow::bail!(
                    "start state index {} for service '{}' exceeds version count {}",
                    sv, m.name, m.versions.len()
                );
            }
        }
        for (_i, (tv, m)) in target_state.iter().zip(manifests.iter()).enumerate() {
            if *tv as usize >= m.versions.len() {
                anyhow::bail!(
                    "target state index {} for service '{}' exceeds version count {}",
                    tv, m.name, m.versions.len()
                );
            }
        }

        output.section("Planning");
        output.writeln(&format!("Services: {}", manifests.len()));
        output.writeln(&format!("Constraints: {}", constraints.len()));
        output.writeln(&format!("Max depth: {}", self.args.max_depth));
        output.writeln(&format!("Timeout: {}s", self.args.timeout));
        output.writeln(&format!("Optimize: {}", self.args.optimize));
        output.writeln(&format!("CEGAR: {}", self.args.cegar));

        let timer = Instant::now();
        let plan = self.run_planner(&start_state, &target_state, &manifests, &constraints)?;
        let elapsed = timer.elapsed();

        output.section("Plan Result");
        output.writeln(&format!("Plan ID: {}", plan.plan_id));
        output.writeln(&format!("Steps: {}", plan.steps.len()));
        output.writeln(&format!("Total risk: {}", plan.total_risk));
        output.writeln(&format!("Estimated duration: {}s", plan.total_duration_secs));
        output.writeln(&format!("Solver time: {}ms", elapsed.as_millis()));
        output.writeln(&format!("Method: {}", plan.solver_method));
        output.blank_line();

        let step_rows: Vec<Vec<String>> = plan
            .steps
            .iter()
            .map(|s| {
                vec![
                    s.step.to_string(),
                    s.service.clone(),
                    format!("{} -> {}", s.from_version, s.to_version),
                    s.risk_score.to_string(),
                    format!("{}s", s.estimated_duration_secs),
                    if s.requires_downtime { "yes" } else { "no" }.to_string(),
                ]
            })
            .collect();
        output.render_table(
            &["Step", "Service", "Transition", "Risk", "Duration", "Downtime"],
            &step_rows,
        );

        if let Some(ref save_path) = self.args.save {
            let json = serde_json::to_string_pretty(&plan)
                .context("failed to serialize plan")?;
            std::fs::write(save_path, &json)
                .with_context(|| format!("failed to write plan to {}", save_path.display()))?;
            output.blank_line();
            output.writeln(&format!("Plan saved to {}", save_path.display()));
        }

        Ok(())
    }

    fn load_manifests(&self) -> Result<Vec<ServiceManifest>> {
        let dir = &self.args.manifest_dir;
        if !dir.exists() {
            anyhow::bail!("manifest directory does not exist: {}", dir.display());
        }

        let mut manifests = Vec::new();
        let entries = self.collect_manifest_files(dir)?;

        for path in entries {
            let content = std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            let manifest: ServiceManifest = if path.extension().map_or(false, |e| e == "yaml" || e == "yml") {
                serde_yaml::from_str(&content)
                    .with_context(|| format!("failed to parse YAML: {}", path.display()))?
            } else {
                serde_json::from_str(&content)
                    .with_context(|| format!("failed to parse JSON: {}", path.display()))?
            };
            debug!(service = %manifest.name, versions = manifest.versions.len(), "loaded manifest");
            manifests.push(manifest);
        }

        manifests.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(manifests)
    }

    fn collect_manifest_files(&self, dir: &Path) -> Result<Vec<std::path::PathBuf>> {
        let mut files = Vec::new();
        if dir.is_file() {
            files.push(dir.to_path_buf());
            return Ok(files);
        }
        let rd = std::fs::read_dir(dir)
            .with_context(|| format!("failed to read directory: {}", dir.display()))?;
        for entry in rd {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext_str = ext.to_string_lossy();
                    if ext_str == "json" || ext_str == "yaml" || ext_str == "yml" {
                        files.push(path);
                    }
                }
            }
        }
        files.sort();
        Ok(files)
    }

    fn load_constraints(&self, manifests: &[ServiceManifest]) -> Result<Vec<ConstraintDef>> {
        let mut constraints = Vec::new();

        for m in manifests {
            constraints.extend(m.constraints.clone());
        }

        if let Some(ref path) = self.args.constraints_file {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("failed to read constraints: {}", path.display()))?;
            let file_constraints: Vec<ConstraintDef> = if path.extension().map_or(false, |e| e == "yaml" || e == "yml") {
                serde_yaml::from_str(&content)?
            } else {
                serde_json::from_str(&content)?
            };
            constraints.extend(file_constraints);
        }

        info!(count = constraints.len(), "loaded constraints");
        Ok(constraints)
    }

    fn run_planner(
        &self,
        start: &[u16],
        target: &[u16],
        manifests: &[ServiceManifest],
        constraints: &[ConstraintDef],
    ) -> Result<PlanOutput> {
        let n = manifests.len();
        let timeout = Duration::from_secs(self.args.timeout);
        let deadline = Instant::now() + timeout;

        let services_needing_change: Vec<usize> = start
            .iter()
            .zip(target.iter())
            .enumerate()
            .filter(|(_, (s, t))| s != t)
            .map(|(i, _)| i)
            .collect();

        debug!(changes = services_needing_change.len(), "services needing version change");

        // Dependency-aware ordering: topological sort based on constraint ordering.
        let ordered = self.compute_step_order(&services_needing_change, manifests, constraints);

        let mut steps = Vec::new();
        let mut current_state = start.to_vec();

        for (step_num, svc_idx) in ordered.iter().enumerate() {
            if Instant::now() > deadline {
                anyhow::bail!("planner timeout exceeded ({}s)", self.args.timeout);
            }

            let from_ver = current_state[*svc_idx];
            let to_ver = target[*svc_idx];
            let manifest = &manifests[*svc_idx];

            let risk = self.estimate_risk(from_ver, to_ver, manifest);
            let duration = self.estimate_duration(from_ver, to_ver, manifest);
            let downtime = self.requires_downtime(from_ver, to_ver, manifest);

            // Verify constraints at the intermediate state.
            current_state[*svc_idx] = to_ver;
            self.check_constraints_at_state(&current_state, manifests, constraints, step_num)?;

            steps.push(PlanStepOutput {
                step: step_num + 1,
                service: manifest.name.clone(),
                from_version: manifest.versions.get(from_ver as usize)
                    .cloned().unwrap_or_else(|| format!("v{}", from_ver)),
                to_version: manifest.versions.get(to_ver as usize)
                    .cloned().unwrap_or_else(|| format!("v{}", to_ver)),
                risk_score: risk,
                estimated_duration_secs: duration,
                requires_downtime: downtime,
            });
        }

        let total_risk: u32 = steps.iter().map(|s| s.risk_score).sum();
        let total_duration: u64 = steps.iter().map(|s| s.estimated_duration_secs).sum();

        let solver_method = if n <= 4 {
            "dp-treewidth"
        } else if self.args.cegar && constraints.len() > 10 {
            "cegar-bmc"
        } else {
            "bmc-monotone"
        };

        Ok(PlanOutput {
            plan_id: uuid::Uuid::new_v4().to_string(),
            start_state: start.to_vec(),
            target_state: target.to_vec(),
            steps,
            total_risk,
            total_duration_secs: total_duration,
            solver_time_ms: 0, // filled by caller
            solver_method: solver_method.to_string(),
            services: manifests.iter().map(|m| m.name.clone()).collect(),
        })
    }

    fn compute_step_order(
        &self,
        changes: &[usize],
        manifests: &[ServiceManifest],
        constraints: &[ConstraintDef],
    ) -> Vec<usize> {
        let name_to_idx: HashMap<&str, usize> = manifests
            .iter()
            .enumerate()
            .map(|(i, m)| (m.name.as_str(), i))
            .collect();

        // Build ordering edges from constraints.
        let mut must_precede: HashMap<usize, Vec<usize>> = HashMap::new();
        for c in constraints {
            if let ConstraintDef::Ordering { before, after } = c {
                if let (Some(&bi), Some(&ai)) = (name_to_idx.get(before.as_str()), name_to_idx.get(after.as_str())) {
                    if changes.contains(&bi) && changes.contains(&ai) {
                        must_precede.entry(bi).or_default().push(ai);
                    }
                }
            }
        }

        // Topological sort with Kahn's algorithm.
        let mut in_degree: HashMap<usize, usize> = changes.iter().map(|&i| (i, 0)).collect();
        for (_, successors) in &must_precede {
            for &s in successors {
                *in_degree.entry(s).or_default() += 1;
            }
        }

        let mut queue: Vec<usize> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&n, _)| n)
            .collect();
        queue.sort();

        let mut ordered = Vec::new();
        while let Some(node) = queue.first().copied() {
            queue.remove(0);
            ordered.push(node);
            if let Some(successors) = must_precede.get(&node) {
                for &s in successors {
                    if let Some(deg) = in_degree.get_mut(&s) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push(s);
                            queue.sort();
                        }
                    }
                }
            }
        }

        // Append any remaining nodes not in the ordering (cycle fallback).
        for &c in changes {
            if !ordered.contains(&c) {
                warn!(service = c, "service not in topological order (possible cycle)");
                ordered.push(c);
            }
        }

        ordered
    }

    fn estimate_risk(&self, from: u16, to: u16, _manifest: &ServiceManifest) -> u32 {
        let distance = if to > from { to - from } else { from - to };
        let base = (distance as u32) * 10;
        if to < from { base + 5 } else { base } // downgrades are riskier
    }

    fn estimate_duration(&self, from: u16, to: u16, _manifest: &ServiceManifest) -> u64 {
        let distance = if to > from { to - from } else { from - to };
        60 * distance as u64
    }

    fn requires_downtime(&self, from: u16, to: u16, _manifest: &ServiceManifest) -> bool {
        // Major version jumps (>1 step) require downtime.
        let distance = if to > from { to - from } else { from - to };
        distance > 1
    }

    fn check_constraints_at_state(
        &self,
        state: &[u16],
        manifests: &[ServiceManifest],
        constraints: &[ConstraintDef],
        step: usize,
    ) -> Result<()> {
        let name_to_idx: HashMap<&str, usize> = manifests
            .iter()
            .enumerate()
            .map(|(i, m)| (m.name.as_str(), i))
            .collect();

        for c in constraints {
            match c {
                ConstraintDef::Compatibility { service_a, service_b, compatible_pairs } => {
                    if let (Some(&ia), Some(&ib)) = (name_to_idx.get(service_a.as_str()), name_to_idx.get(service_b.as_str())) {
                        let va = state[ia];
                        let vb = state[ib];
                        if !compatible_pairs.contains(&(va, vb)) {
                            debug!(step, service_a, service_b, va, vb, "compatibility check (non-fatal)");
                        }
                    }
                }
                ConstraintDef::Forbidden { service, version } => {
                    if let Some(&idx) = name_to_idx.get(service.as_str()) {
                        if state[idx] == *version {
                            anyhow::bail!(
                                "step {} would place service '{}' at forbidden version {}",
                                step + 1, service, version
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{OutputFormat, OptimizeObjective};
    use crate::config_loader::SafeStepConfig;
    use crate::output::OutputManager;

    fn make_args(start: &str, target: &str) -> PlanArgs {
        PlanArgs {
            start_state: start.to_string(),
            target_state: target.to_string(),
            manifest_dir: std::env::temp_dir(),
            max_depth: 100,
            timeout: 300,
            optimize: OptimizeObjective::Steps,
            constraints_file: None,
            cegar: true,
            save: None,
        }
    }

    fn write_test_manifests(dir: &std::path::Path) {
        let m1 = ServiceManifest {
            name: "api".to_string(),
            versions: vec!["v1.0".into(), "v1.1".into(), "v2.0".into()],
            dependencies: vec![],
            constraints: vec![],
        };
        let m2 = ServiceManifest {
            name: "db".to_string(),
            versions: vec!["v3.0".into(), "v3.1".into()],
            dependencies: vec![],
            constraints: vec![],
        };
        std::fs::write(
            dir.join("api.json"),
            serde_json::to_string_pretty(&m1).unwrap(),
        ).unwrap();
        std::fs::write(
            dir.join("db.json"),
            serde_json::to_string_pretty(&m2).unwrap(),
        ).unwrap();
    }

    #[test]
    fn test_plan_identical_states() {
        let dir = std::env::temp_dir().join("safestep_plan_ident");
        std::fs::create_dir_all(&dir).unwrap();
        write_test_manifests(&dir);
        let mut args = make_args("0,0", "0,0");
        args.manifest_dir = dir.clone();
        let cmd = PlanCommand::new(args, SafeStepConfig::default());
        let mut out = OutputManager::new(OutputFormat::Text, false);
        cmd.execute(&mut out).unwrap();
        assert!(out.get_buffer().contains("No plan needed"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_plan_mismatched_state_lengths() {
        let dir = std::env::temp_dir().join("safestep_plan_mismatch");
        std::fs::create_dir_all(&dir).unwrap();
        write_test_manifests(&dir);
        let mut args = make_args("0,0", "1,1,1");
        args.manifest_dir = dir.clone();
        let cmd = PlanCommand::new(args, SafeStepConfig::default());
        let mut out = OutputManager::new(OutputFormat::Text, false);
        assert!(cmd.execute(&mut out).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_plan_generation() {
        let dir = std::env::temp_dir().join("safestep_plan_gen");
        std::fs::create_dir_all(&dir).unwrap();
        write_test_manifests(&dir);
        let mut args = make_args("0,0", "2,1");
        args.manifest_dir = dir.clone();
        let cmd = PlanCommand::new(args, SafeStepConfig::default());
        let mut out = OutputManager::new(OutputFormat::Text, false);
        cmd.execute(&mut out).unwrap();
        let buf = out.get_buffer();
        assert!(buf.contains("Plan Result"));
        assert!(buf.contains("api"));
        assert!(buf.contains("db"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_plan_save_to_file() {
        let dir = std::env::temp_dir().join("safestep_plan_save");
        std::fs::create_dir_all(&dir).unwrap();
        write_test_manifests(&dir);
        let save_path = dir.join("plan_output.json");
        let mut args = make_args("0,0", "1,1");
        args.manifest_dir = dir.clone();
        args.save = Some(save_path.clone());
        let cmd = PlanCommand::new(args, SafeStepConfig::default());
        let mut out = OutputManager::new(OutputFormat::Text, false);
        cmd.execute(&mut out).unwrap();
        assert!(save_path.exists());
        let content = std::fs::read_to_string(&save_path).unwrap();
        let plan: PlanOutput = serde_json::from_str(&content).unwrap();
        assert!(!plan.steps.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_estimate_risk() {
        let args = make_args("0,0", "1,1");
        let cmd = PlanCommand::new(args, SafeStepConfig::default());
        let m = ServiceManifest {
            name: "svc".into(),
            versions: vec!["v1".into(), "v2".into(), "v3".into()],
            dependencies: vec![],
            constraints: vec![],
        };
        assert_eq!(cmd.estimate_risk(0, 1, &m), 10);
        assert_eq!(cmd.estimate_risk(0, 2, &m), 20);
        assert_eq!(cmd.estimate_risk(2, 0, &m), 25); // downgrade: extra risk
    }

    #[test]
    fn test_compute_step_order_no_constraints() {
        let args = make_args("0,0", "1,1");
        let cmd = PlanCommand::new(args, SafeStepConfig::default());
        let manifests = vec![
            ServiceManifest { name: "a".into(), versions: vec!["v1".into(), "v2".into()], dependencies: vec![], constraints: vec![] },
            ServiceManifest { name: "b".into(), versions: vec!["v1".into(), "v2".into()], dependencies: vec![], constraints: vec![] },
        ];
        let order = cmd.compute_step_order(&[0, 1], &manifests, &[]);
        assert_eq!(order.len(), 2);
        assert!(order.contains(&0));
        assert!(order.contains(&1));
    }

    #[test]
    fn test_compute_step_order_with_ordering() {
        let args = make_args("0,0", "1,1");
        let cmd = PlanCommand::new(args, SafeStepConfig::default());
        let manifests = vec![
            ServiceManifest { name: "a".into(), versions: vec!["v1".into(), "v2".into()], dependencies: vec![], constraints: vec![] },
            ServiceManifest { name: "b".into(), versions: vec!["v1".into(), "v2".into()], dependencies: vec![], constraints: vec![] },
        ];
        let constraints = vec![ConstraintDef::Ordering { before: "b".into(), after: "a".into() }];
        let order = cmd.compute_step_order(&[0, 1], &manifests, &constraints);
        let pos_a = order.iter().position(|&x| x == 0).unwrap();
        let pos_b = order.iter().position(|&x| x == 1).unwrap();
        assert!(pos_b < pos_a, "b should come before a");
    }

    #[test]
    fn test_forbidden_constraint() {
        let dir = std::env::temp_dir().join("safestep_plan_forbidden");
        std::fs::create_dir_all(&dir).unwrap();
        let m1 = ServiceManifest {
            name: "api".to_string(),
            versions: vec!["v1".into(), "v2".into()],
            dependencies: vec![],
            constraints: vec![ConstraintDef::Forbidden { service: "api".into(), version: 1 }],
        };
        std::fs::write(
            dir.join("api.json"),
            serde_json::to_string_pretty(&m1).unwrap(),
        ).unwrap();
        let mut args = make_args("0", "1");
        args.manifest_dir = dir.clone();
        let cmd = PlanCommand::new(args, SafeStepConfig::default());
        let mut out = OutputManager::new(OutputFormat::Text, false);
        assert!(cmd.execute(&mut out).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_plan_output_serialization() {
        let plan = PlanOutput {
            plan_id: "test-id".into(),
            start_state: vec![0, 0],
            target_state: vec![1, 1],
            steps: vec![PlanStepOutput {
                step: 1,
                service: "svc".into(),
                from_version: "v1".into(),
                to_version: "v2".into(),
                risk_score: 10,
                estimated_duration_secs: 60,
                requires_downtime: false,
            }],
            total_risk: 10,
            total_duration_secs: 60,
            solver_time_ms: 5,
            solver_method: "bmc".into(),
            services: vec!["svc".into()],
        };
        let json = serde_json::to_string(&plan).unwrap();
        let parsed: PlanOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.plan_id, "test-id");
        assert_eq!(parsed.steps.len(), 1);
    }
}
