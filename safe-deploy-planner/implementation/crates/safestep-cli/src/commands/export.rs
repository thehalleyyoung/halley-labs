//! Implementation of the `export` subcommand.

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::cli::{ExportArgs, GitOpsFormat};
use crate::output::OutputManager;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Plan loaded for export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportPlan {
    pub plan_id: String,
    #[serde(default)]
    pub services: Vec<String>,
    #[serde(default)]
    pub steps: Vec<ExportStep>,
    #[serde(default)]
    pub start_state: Vec<u16>,
    #[serde(default)]
    pub target_state: Vec<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportStep {
    pub step: usize,
    pub service: String,
    #[serde(default)]
    pub from_version: String,
    #[serde(default)]
    pub to_version: String,
    #[serde(default)]
    pub requires_downtime: bool,
}

/// ArgoCD Application resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArgoCdApp {
    pub api_version: String,
    pub kind: String,
    pub metadata: ResourceMetadata,
    pub spec: ArgoCdSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetadata {
    pub name: String,
    pub namespace: String,
    #[serde(default)]
    pub labels: HashMap<String, String>,
    #[serde(default)]
    pub annotations: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArgoCdSpec {
    pub project: String,
    pub source: ArgoCdSource,
    pub destination: ArgoCdDestination,
    pub sync_policy: ArgoCdSyncPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArgoCdSource {
    pub repo_url: String,
    pub path: String,
    pub target_revision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgoCdDestination {
    pub server: String,
    pub namespace: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgoCdSyncPolicy {
    pub automated: ArgoCdAutomated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArgoCdAutomated {
    pub prune: bool,
    pub self_heal: bool,
}

/// Flux HelmRelease resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FluxHelmRelease {
    pub api_version: String,
    pub kind: String,
    pub metadata: ResourceMetadata,
    pub spec: FluxHelmSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FluxHelmSpec {
    pub interval: String,
    pub chart: FluxChartRef,
    pub target_namespace: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxChartRef {
    pub spec: FluxChartSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxChartSpec {
    pub chart: String,
    pub version: String,
    #[serde(rename = "sourceRef")]
    pub source_ref: FluxSourceRef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxSourceRef {
    pub kind: String,
    pub name: String,
}

/// Validation issue for generated resources.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub file: String,
    pub message: String,
    pub is_error: bool,
}

// ---------------------------------------------------------------------------
// ExportCommand
// ---------------------------------------------------------------------------

pub struct ExportCommand {
    args: ExportArgs,
}

impl ExportCommand {
    pub fn new(args: ExportArgs) -> Self {
        Self { args }
    }

    pub fn execute(&self, output: &mut OutputManager) -> Result<()> {
        info!(plan = %self.args.plan_file.display(), format = %self.args.format, "exporting plan");

        let plan = self.load_plan()?;
        std::fs::create_dir_all(&self.args.output_dir)
            .with_context(|| format!("failed to create output dir: {}", self.args.output_dir.display()))?;

        let files_written = match self.args.format {
            GitOpsFormat::Argocd => self.export_argocd(&plan)?,
            GitOpsFormat::Flux => self.export_flux(&plan)?,
        };

        let mut issues = Vec::new();
        if self.args.validate_output {
            issues = self.validate_output(&files_written)?;
        }

        self.render_output(output, &plan, &files_written, &issues)?;
        Ok(())
    }

    fn load_plan(&self) -> Result<ExportPlan> {
        let content = std::fs::read_to_string(&self.args.plan_file)
            .with_context(|| format!("failed to read plan: {}", self.args.plan_file.display()))?;
        serde_json::from_str(&content)
            .with_context(|| format!("failed to parse plan: {}", self.args.plan_file.display()))
    }

    fn export_argocd(&self, plan: &ExportPlan) -> Result<Vec<String>> {
        let mut files = Vec::new();

        for (i, step) in plan.steps.iter().enumerate() {
            let app = ArgoCdApp {
                api_version: "argoproj.io/v1alpha1".into(),
                kind: "Application".into(),
                metadata: ResourceMetadata {
                    name: format!("{}-step-{}", step.service, step.step),
                    namespace: self.args.namespace.clone(),
                    labels: {
                        let mut l = HashMap::new();
                        l.insert("app.kubernetes.io/managed-by".into(), "safestep".into());
                        l.insert("safestep/plan-id".into(), plan.plan_id.clone());
                        l.insert("safestep/step".into(), step.step.to_string());
                        l
                    },
                    annotations: {
                        let mut a = HashMap::new();
                        a.insert("argocd.argoproj.io/sync-wave".into(), i.to_string());
                        if step.requires_downtime {
                            a.insert("safestep/requires-downtime".into(), "true".into());
                        }
                        a
                    },
                },
                spec: ArgoCdSpec {
                    project: "default".into(),
                    source: ArgoCdSource {
                        repo_url: "https://github.com/org/deploy-configs".into(),
                        path: format!("services/{}", step.service),
                        target_revision: step.to_version.clone(),
                    },
                    destination: ArgoCdDestination {
                        server: "https://kubernetes.default.svc".into(),
                        namespace: self.args.namespace.clone(),
                    },
                    sync_policy: ArgoCdSyncPolicy {
                        automated: ArgoCdAutomated {
                            prune: true,
                            self_heal: true,
                        },
                    },
                },
            };

            let filename = format!("{}-step-{}.yaml", step.service, step.step);
            let filepath = self.args.output_dir.join(&filename);
            let yaml = serde_yaml::to_string(&app).context("serialize ArgoCD app")?;
            std::fs::write(&filepath, &yaml)
                .with_context(|| format!("write {}", filepath.display()))?;
            debug!(file = %filepath.display(), "wrote ArgoCD Application");
            files.push(filename);
        }

        // Write a kustomization.yaml that references all resources.
        let kustomization = serde_yaml::to_string(&serde_json::json!({
            "apiVersion": "kustomize.config.k8s.io/v1beta1",
            "kind": "Kustomization",
            "metadata": {
                "name": format!("{}-bundle", plan.plan_id),
                "namespace": self.args.namespace.clone(),
            },
            "resources": files,
        })).context("serialize kustomization")?;
        let kust_path = self.args.output_dir.join("kustomization.yaml");
        std::fs::write(&kust_path, &kustomization).context("write kustomization")?;
        files.push("kustomization.yaml".into());

        Ok(files)
    }

    fn export_flux(&self, plan: &ExportPlan) -> Result<Vec<String>> {
        let mut files = Vec::new();

        for step in &plan.steps {
            let release = FluxHelmRelease {
                api_version: "helm.toolkit.fluxcd.io/v2beta1".into(),
                kind: "HelmRelease".into(),
                metadata: ResourceMetadata {
                    name: format!("{}-step-{}", step.service, step.step),
                    namespace: self.args.namespace.clone(),
                    labels: {
                        let mut l = HashMap::new();
                        l.insert("app.kubernetes.io/managed-by".into(), "safestep".into());
                        l.insert("safestep/plan-id".into(), plan.plan_id.clone());
                        l
                    },
                    annotations: HashMap::new(),
                },
                spec: FluxHelmSpec {
                    interval: "5m".into(),
                    chart: FluxChartRef {
                        spec: FluxChartSpec {
                            chart: step.service.clone(),
                            version: step.to_version.clone(),
                            source_ref: FluxSourceRef {
                                kind: "HelmRepository".into(),
                                name: "main".into(),
                            },
                        },
                    },
                    target_namespace: self.args.namespace.clone(),
                },
            };

            let filename = format!("{}-step-{}.yaml", step.service, step.step);
            let filepath = self.args.output_dir.join(&filename);
            let yaml = serde_yaml::to_string(&release).context("serialize Flux release")?;
            std::fs::write(&filepath, &yaml)
                .with_context(|| format!("write {}", filepath.display()))?;
            debug!(file = %filepath.display(), "wrote Flux HelmRelease");
            files.push(filename);
        }

        Ok(files)
    }

    fn validate_output(&self, files: &[String]) -> Result<Vec<ValidationIssue>> {
        let mut issues = Vec::new();

        for filename in files {
            let path = self.args.output_dir.join(filename);
            if !path.exists() {
                issues.push(ValidationIssue {
                    file: filename.clone(),
                    message: "file does not exist after write".into(),
                    is_error: true,
                });
                continue;
            }

            let content = std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?;

            // Validate it's parseable YAML.
            let parsed: Result<serde_json::Value, _> = serde_yaml::from_str(&content);
            if parsed.is_err() {
                issues.push(ValidationIssue {
                    file: filename.clone(),
                    message: "generated file is not valid YAML".into(),
                    is_error: true,
                });
                continue;
            }

            let doc = parsed.unwrap();

            // Check required fields.
            if doc.get("apiVersion").is_none() {
                issues.push(ValidationIssue {
                    file: filename.clone(),
                    message: "missing apiVersion field".into(),
                    is_error: true,
                });
            }
            if doc.get("kind").is_none() {
                issues.push(ValidationIssue {
                    file: filename.clone(),
                    message: "missing kind field".into(),
                    is_error: true,
                });
            }
            if doc.get("metadata").is_none() {
                issues.push(ValidationIssue {
                    file: filename.clone(),
                    message: "missing metadata field".into(),
                    is_error: true,
                });
            }

            if content.len() > 1_000_000 {
                issues.push(ValidationIssue {
                    file: filename.clone(),
                    message: "file exceeds 1MB".into(),
                    is_error: false,
                });
            }
        }

        Ok(issues)
    }

    fn render_output(
        &self,
        output: &mut OutputManager,
        plan: &ExportPlan,
        files: &[String],
        issues: &[ValidationIssue],
    ) -> Result<()> {
        let colors = output.colors().clone();

        output.section("Export Summary");
        output.writeln(&format!("Plan: {}", plan.plan_id));
        output.writeln(&format!("Format: {}", self.args.format));
        output.writeln(&format!("Output: {}", self.args.output_dir.display()));
        output.writeln(&format!("Files written: {}", files.len()));

        output.blank_line();
        for f in files {
            output.writeln(&format!("  {}", colors.safe(f)));
        }

        if !issues.is_empty() {
            output.blank_line();
            output.section("Validation Issues");
            for issue in issues {
                if issue.is_error {
                    output.writeln(&format!("  {} {}: {}",
                        colors.error("ERROR"), issue.file, issue.message));
                } else {
                    output.writeln(&format!("  {} {}: {}",
                        colors.warning("WARN"), issue.file, issue.message));
                }
            }
            let errors = issues.iter().filter(|i| i.is_error).count();
            if errors > 0 {
                anyhow::bail!("{} validation error(s) in generated resources", errors);
            }
        } else if self.args.validate_output {
            output.blank_line();
            output.writeln(&colors.safe("All generated resources validated successfully."));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::OutputFormat;
    use crate::output::OutputManager;

    fn sample_plan() -> ExportPlan {
        ExportPlan {
            plan_id: "export-test".into(),
            services: vec!["api".into(), "db".into()],
            steps: vec![
                ExportStep { step: 1, service: "api".into(), from_version: "v1".into(), to_version: "v2".into(), requires_downtime: false },
                ExportStep { step: 2, service: "db".into(), from_version: "v3".into(), to_version: "v4".into(), requires_downtime: true },
            ],
            start_state: vec![0, 0],
            target_state: vec![1, 1],
        }
    }

    #[test]
    fn test_export_argocd() {
        let dir = std::env::temp_dir().join("safestep_export_argocd");
        std::fs::create_dir_all(&dir).unwrap();
        let cmd = ExportCommand::new(ExportArgs {
            plan_file: "/tmp/dummy".into(),
            format: GitOpsFormat::Argocd,
            output_dir: dir.clone(),
            namespace: "production".into(),
            validate_output: false,
        });
        let plan = sample_plan();
        let files = cmd.export_argocd(&plan).unwrap();
        assert!(files.len() >= 2);
        assert!(files.iter().any(|f| f.contains("api")));
        assert!(dir.join("kustomization.yaml").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_export_flux() {
        let dir = std::env::temp_dir().join("safestep_export_flux");
        std::fs::create_dir_all(&dir).unwrap();
        let cmd = ExportCommand::new(ExportArgs {
            plan_file: "/tmp/dummy".into(),
            format: GitOpsFormat::Flux,
            output_dir: dir.clone(),
            namespace: "staging".into(),
            validate_output: false,
        });
        let plan = sample_plan();
        let files = cmd.export_flux(&plan).unwrap();
        assert_eq!(files.len(), 2);
        for f in &files {
            let content = std::fs::read_to_string(dir.join(f)).unwrap();
            assert!(content.contains("HelmRelease"));
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_validate_output() {
        let dir = std::env::temp_dir().join("safestep_export_validate");
        std::fs::create_dir_all(&dir).unwrap();
        let cmd = ExportCommand::new(ExportArgs {
            plan_file: "/tmp/dummy".into(),
            format: GitOpsFormat::Argocd,
            output_dir: dir.clone(),
            namespace: "default".into(),
            validate_output: true,
        });
        let plan = sample_plan();
        let files = cmd.export_argocd(&plan).unwrap();
        let issues = cmd.validate_output(&files).unwrap();
        assert!(issues.is_empty(), "expected no validation issues but got: {:?}", issues.iter().map(|i| &i.message).collect::<Vec<_>>());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_validate_missing_file() {
        let dir = std::env::temp_dir().join("safestep_export_missing");
        std::fs::create_dir_all(&dir).unwrap();
        let cmd = ExportCommand::new(ExportArgs {
            plan_file: "/tmp/dummy".into(),
            format: GitOpsFormat::Argocd,
            output_dir: dir.clone(),
            namespace: "default".into(),
            validate_output: true,
        });
        let issues = cmd.validate_output(&["nonexistent.yaml".into()]).unwrap();
        assert!(issues.iter().any(|i| i.is_error));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_argocd_app_serialization() {
        let app = ArgoCdApp {
            api_version: "argoproj.io/v1alpha1".into(),
            kind: "Application".into(),
            metadata: ResourceMetadata {
                name: "test".into(), namespace: "default".into(),
                labels: HashMap::new(), annotations: HashMap::new(),
            },
            spec: ArgoCdSpec {
                project: "default".into(),
                source: ArgoCdSource { repo_url: "https://example.com".into(), path: ".".into(), target_revision: "main".into() },
                destination: ArgoCdDestination { server: "https://kubernetes.default.svc".into(), namespace: "default".into() },
                sync_policy: ArgoCdSyncPolicy { automated: ArgoCdAutomated { prune: true, self_heal: true } },
            },
        };
        let yaml = serde_yaml::to_string(&app).unwrap();
        assert!(yaml.contains("Application"));
        assert!(yaml.contains("argoproj.io"));
    }

    #[test]
    fn test_flux_release_serialization() {
        let release = FluxHelmRelease {
            api_version: "helm.toolkit.fluxcd.io/v2beta1".into(),
            kind: "HelmRelease".into(),
            metadata: ResourceMetadata {
                name: "test".into(), namespace: "default".into(),
                labels: HashMap::new(), annotations: HashMap::new(),
            },
            spec: FluxHelmSpec {
                interval: "5m".into(),
                chart: FluxChartRef {
                    spec: FluxChartSpec { chart: "svc".into(), version: "1.0".into(), source_ref: FluxSourceRef { kind: "HelmRepository".into(), name: "main".into() } },
                },
                target_namespace: "default".into(),
            },
        };
        let yaml = serde_yaml::to_string(&release).unwrap();
        assert!(yaml.contains("HelmRelease"));
    }
}
