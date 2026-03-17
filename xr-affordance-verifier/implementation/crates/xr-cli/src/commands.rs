//! Command implementations for the XR Affordance Verifier CLI.

use std::path::PathBuf;
use std::time::Instant;

use xr_certificate::CertificateGenerator;
use xr_lint::{LintReport, SceneLinter};
use xr_types::certificate::CoverageCertificate;
use xr_types::config::VerifierConfig;
use xr_types::error::{VerifierError, VerifierResult};
use xr_types::scene::SceneModel;

use crate::config::CliConfig;
use crate::demo::DemoSceneGenerator;
use crate::output::OutputFormatter;
use crate::pipeline::VerificationPipeline;
use crate::scene_loader::SceneLoader;
use crate::showcase::{ShowcaseBundleGenerator, ShowcaseInputs};
use crate::webapp::WebAppGenerator;
use crate::{ConfigAction, DemoScene, ShowcaseScenario};

struct SceneAnalysisArtifacts {
    lint_report: LintReport,
    pipeline_result: crate::pipeline::PipelineResult,
    certificate: CoverageCertificate,
}

fn analyze_scene(
    scene: &SceneModel,
    num_samples: usize,
    confidence: f64,
) -> VerifierResult<SceneAnalysisArtifacts> {
    let linter = SceneLinter::new();
    let lint_report = linter.lint(scene);

    let config = VerifierConfig::builder()
        .num_samples(num_samples)
        .confidence_delta(1.0 - confidence)
        .build()?;

    let pipeline = VerificationPipeline::new(config);
    let pipeline_result = pipeline.run_full(scene)?;

    let cert_gen = CertificateGenerator::new();
    let certificate = cert_gen.generate(
        scene,
        pipeline_result.sample_verdicts.clone(),
        pipeline_result.verified_regions.clone(),
        pipeline_result.violations.clone(),
        pipeline_result.epsilon_analytical,
        pipeline_result.total_time().as_secs_f64(),
    )?;

    Ok(SceneAnalysisArtifacts {
        lint_report,
        pipeline_result,
        certificate,
    })
}

// ─── Lint ──────────────────────────────────────────────────────────────────

pub struct LintCommand<'a> {
    pub scene_path: PathBuf,
    pub min_height: f64,
    pub max_height: f64,
    pub disabled_rules: Vec<String>,
    pub output_path: Option<PathBuf>,
    pub cli_config: &'a CliConfig,
}

impl<'a> LintCommand<'a> {
    pub fn execute(&self) -> VerifierResult<()> {
        let start = Instant::now();
        tracing::info!("Linting scene: {}", self.scene_path.display());

        let loader = SceneLoader::new();
        let scene = loader.load(&self.scene_path)?;

        tracing::info!(
            "Loaded scene '{}' with {} elements",
            scene.name,
            scene.elements.len()
        );

        let mut lint_config = xr_lint::LintConfig::default();
        lint_config.min_element_height = self.min_height;
        lint_config.max_element_height = self.max_height;

        let linter = SceneLinter::with_config(lint_config);
        let report = linter.lint(&scene);

        let elapsed = start.elapsed();

        let formatter = OutputFormatter::new(self.cli_config.format, !self.cli_config.no_color);

        let output = formatter.format_lint_report(&report, elapsed);

        if let Some(ref path) = self.output_path {
            std::fs::write(path, &output)?;
            tracing::info!("Report written to {}", path.display());
        } else {
            print!("{output}");
        }

        if report.has_errors() {
            Err(VerifierError::Configuration(format!(
                "Linting found {} error(s)",
                report.errors().len()
            )))
        } else {
            Ok(())
        }
    }
}

// ─── Verify ────────────────────────────────────────────────────────────────

pub struct VerifyCommand<'a> {
    pub scene_path: PathBuf,
    pub num_samples: usize,
    pub smt_timeout: f64,
    pub skip_tier2: bool,
    pub fail_fast: bool,
    pub target_kappa: f64,
    pub output_path: Option<PathBuf>,
    pub cli_config: &'a CliConfig,
}

impl<'a> VerifyCommand<'a> {
    pub fn execute(&self) -> VerifierResult<()> {
        let start = Instant::now();
        tracing::info!("Verifying scene: {}", self.scene_path.display());

        let loader = SceneLoader::new();
        let scene = loader.load(&self.scene_path)?;

        tracing::info!(
            "Loaded scene '{}' with {} elements",
            scene.name,
            scene.elements.len()
        );

        let config = self.build_verifier_config()?;
        let pipeline = VerificationPipeline::new(config);

        let result = if self.skip_tier2 {
            pipeline.run_lint_only(&scene)?
        } else {
            pipeline.run_full(&scene)?
        };

        let elapsed = start.elapsed();

        let formatter = OutputFormatter::new(self.cli_config.format, !self.cli_config.no_color);
        let output = formatter.format_pipeline_result(&result, elapsed);

        if let Some(ref path) = self.output_path {
            std::fs::write(path, &output)?;
            tracing::info!("Results written to {}", path.display());
        } else {
            print!("{output}");
        }

        if result.has_errors() {
            Err(VerifierError::Configuration(format!(
                "Verification found {} issue(s)",
                result.error_count()
            )))
        } else {
            Ok(())
        }
    }

    fn build_verifier_config(&self) -> VerifierResult<VerifierConfig> {
        let config = VerifierConfig::builder()
            .num_samples(self.num_samples)
            .smt_timeout(self.smt_timeout)
            .enable_tier2(!self.skip_tier2)
            .build()?;
        Ok(config)
    }
}

// ─── Certify ───────────────────────────────────────────────────────────────

pub struct CertifyCommand<'a> {
    pub scene_path: PathBuf,
    pub num_samples: usize,
    pub confidence: f64,
    pub output_path: Option<PathBuf>,
    pub generate_svg: bool,
    pub cli_config: &'a CliConfig,
}

impl<'a> CertifyCommand<'a> {
    pub fn execute(&self) -> VerifierResult<()> {
        let start = Instant::now();
        tracing::info!("Generating certificate for: {}", self.scene_path.display());

        let loader = SceneLoader::new();
        let scene = loader.load(&self.scene_path)?;

        tracing::info!(
            "Loaded scene '{}' with {} elements",
            scene.name,
            scene.elements.len()
        );

        let config = VerifierConfig::builder()
            .num_samples(self.num_samples)
            .confidence_delta(1.0 - self.confidence)
            .build()?;

        let pipeline = VerificationPipeline::new(config);
        let pipeline_result = pipeline.run_full(&scene)?;

        let cert_gen = CertificateGenerator::new();
        let certificate = cert_gen.generate(
            &scene,
            pipeline_result.sample_verdicts.clone(),
            pipeline_result.verified_regions.clone(),
            pipeline_result.violations.clone(),
            pipeline_result.epsilon_analytical,
            start.elapsed().as_secs_f64(),
        )?;

        let elapsed = start.elapsed();

        let formatter = OutputFormatter::new(self.cli_config.format, !self.cli_config.no_color);

        let cert_json = certificate.to_json()?;
        let output_path = self
            .output_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("certificate.json"));

        std::fs::write(&output_path, &cert_json)?;
        tracing::info!("Certificate written to {}", output_path.display());

        let summary = formatter.format_certificate_summary(&certificate, elapsed);
        print!("{summary}");

        if self.generate_svg {
            let svg = formatter.generate_svg_report(&certificate);
            let svg_path = output_path.with_extension("svg");
            std::fs::write(&svg_path, &svg)?;
            tracing::info!("SVG diagram written to {}", svg_path.display());
        }

        Ok(())
    }
}

// ─── Inspect ───────────────────────────────────────────────────────────────

pub struct InspectCommand<'a> {
    pub scene_path: PathBuf,
    pub show_elements: bool,
    pub show_deps: bool,
    pub show_devices: bool,
    pub cli_config: &'a CliConfig,
}

impl<'a> InspectCommand<'a> {
    pub fn execute(&self) -> VerifierResult<()> {
        let start = Instant::now();
        tracing::info!("Inspecting scene: {}", self.scene_path.display());

        let loader = SceneLoader::new();
        let (scene, diagnostics) = loader.load_with_validation(&self.scene_path)?;

        let elapsed = start.elapsed();

        let formatter = OutputFormatter::new(self.cli_config.format, !self.cli_config.no_color);
        let output = formatter.format_inspection(
            &scene,
            &diagnostics,
            self.show_elements,
            self.show_deps,
            self.show_devices,
            elapsed,
        );

        print!("{output}");
        Ok(())
    }
}

// ─── Report ────────────────────────────────────────────────────────────────

pub struct ReportCommand<'a> {
    pub certificate_path: PathBuf,
    pub report_format: String,
    pub output_path: Option<PathBuf>,
    pub cli_config: &'a CliConfig,
}

impl<'a> ReportCommand<'a> {
    pub fn execute(&self) -> VerifierResult<()> {
        let start = Instant::now();
        tracing::info!(
            "Generating report from: {}",
            self.certificate_path.display()
        );

        let cert_json = std::fs::read_to_string(&self.certificate_path)?;
        let certificate = CoverageCertificate::from_json(&cert_json)?;

        let elapsed = start.elapsed();

        let formatter = OutputFormatter::new(self.cli_config.format, !self.cli_config.no_color);

        let output = match self.report_format.as_str() {
            "json" => certificate.to_json()?,
            "svg" => formatter.generate_svg_report(&certificate),
            "html" => formatter.generate_html_report(&certificate),
            _ => formatter.format_full_report(&certificate, elapsed),
        };

        if let Some(ref path) = self.output_path {
            std::fs::write(path, &output)?;
            tracing::info!("Report written to {}", path.display());
        } else {
            print!("{output}");
        }

        Ok(())
    }
}

// ─── Webapp ───────────────────────────────────────────────────────────────

pub struct WebAppCommand<'a> {
    pub scene_path: PathBuf,
    pub certificate_path: Option<PathBuf>,
    pub num_samples: usize,
    pub confidence: f64,
    pub output_path: Option<PathBuf>,
    pub title: Option<String>,
    pub cli_config: &'a CliConfig,
}

impl<'a> WebAppCommand<'a> {
    pub fn execute(&self) -> VerifierResult<()> {
        let loader = SceneLoader::new();
        let scene = loader.load(&self.scene_path)?;

        let (certificate, pipeline_result, generated_certificate_path) =
            if let Some(path) = &self.certificate_path {
                let cert_json = std::fs::read_to_string(path)?;
                let cert = CoverageCertificate::from_json(&cert_json)?;
                (Some(cert), None, None)
            } else {
                let config = VerifierConfig::builder()
                    .num_samples(self.num_samples)
                    .confidence_delta(1.0 - self.confidence)
                    .build()?;
                let pipeline = VerificationPipeline::new(config);
                let pipeline_result = pipeline.run_full(&scene)?;
                let cert_gen = CertificateGenerator::new();
                let certificate = cert_gen.generate(
                    &scene,
                    pipeline_result.sample_verdicts.clone(),
                    pipeline_result.verified_regions.clone(),
                    pipeline_result.violations.clone(),
                    pipeline_result.epsilon_analytical,
                    pipeline_result.total_time().as_secs_f64(),
                )?;

                let path = self
                    .output_path
                    .clone()
                    .unwrap_or_else(|| default_webapp_path(&self.scene_path))
                    .with_extension("certificate.json");
                std::fs::write(&path, certificate.to_json()?)?;
                tracing::info!("Generated certificate written to {}", path.display());

                (Some(certificate), Some(pipeline_result), Some(path))
            };

        let html = WebAppGenerator::new().generate(
            &scene,
            certificate.as_ref(),
            pipeline_result.as_ref(),
            self.title.as_deref(),
        );

        let output_path = self
            .output_path
            .clone()
            .unwrap_or_else(|| default_webapp_path(&self.scene_path));
        std::fs::write(&output_path, html)?;

        println!("Web dashboard written to {}", output_path.display());
        if let Some(path) = generated_certificate_path {
            println!("Generated certificate written to {}", path.display());
        }
        if let Some(cert) = &certificate {
            let formatter = OutputFormatter::new(self.cli_config.format, !self.cli_config.no_color);
            print!("{}", formatter.format_certificate_summary(cert, std::time::Duration::from_secs_f64(cert.total_time_s)));
        }

        Ok(())
    }
}

fn default_webapp_path(scene_path: &std::path::Path) -> PathBuf {
    let parent = scene_path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let stem = scene_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("xr_demo");
    parent.join(format!("{stem}.dashboard.html"))
}

// ─── Showcase ─────────────────────────────────────────────────────────────

pub struct ShowcaseCommand<'a> {
    pub scenario: ShowcaseScenario,
    pub output_dir: Option<PathBuf>,
    pub num_samples: usize,
    pub confidence: f64,
    pub title: Option<String>,
    pub cli_config: &'a CliConfig,
}

impl<'a> ShowcaseCommand<'a> {
    pub fn execute(&self) -> VerifierResult<()> {
        let generator = DemoSceneGenerator::new();
        let output_dir = self.output_dir.clone().unwrap_or_else(|| match self.scenario {
            ShowcaseScenario::AccessibilityRemediation => {
                PathBuf::from("xr_showcase_accessibility_bundle")
            }
        });

        let (before_scene, after_scene, remediation_notes, scenario_name, title) =
            match self.scenario {
                ShowcaseScenario::AccessibilityRemediation => {
                    let (before_scene, after_scene) = generator.accessibility_showcase_pair();
                    let title = self
                        .title
                        .clone()
                        .unwrap_or_else(|| "XR Accessibility Remediation Showcase".into());
                    (
                        before_scene,
                        after_scene,
                        generator.accessibility_showcase_repair_notes(),
                        "accessibility-remediation",
                        title,
                    )
                }
            };

        let before = analyze_scene(&before_scene, self.num_samples, self.confidence)?;
        let after = analyze_scene(&after_scene, self.num_samples, self.confidence)?;

        let manifest = ShowcaseBundleGenerator::new().generate_bundle(
            &output_dir,
            ShowcaseInputs {
                title: &title,
                scenario: scenario_name,
                before_scene: &before_scene,
                after_scene: &after_scene,
                before_lint: &before.lint_report,
                after_lint: &after.lint_report,
                before_pipeline: &before.pipeline_result,
                after_pipeline: &after.pipeline_result,
                before_certificate: &before.certificate,
                after_certificate: &after.certificate,
                remediation_notes: &remediation_notes,
            },
        )?;

        println!(
            "✓ Showcase bundle written to {}",
            output_dir.display()
        );
        println!(
            "  Landing page: {}",
            output_dir.join(&manifest.artifacts.landing_page).display()
        );
        println!(
            "  κ delta: {:+.1} pts | violations: {} → {} | lint errors: {} → {}",
            manifest.delta.kappa_delta * 100.0,
            manifest.before.violations,
            manifest.after.violations,
            manifest.before.lint_errors,
            manifest.after.lint_errors,
        );

        Ok(())
    }
}

// ─── Config ────────────────────────────────────────────────────────────────

pub struct ConfigCommand<'a> {
    pub action: ConfigAction,
    pub cli_config: &'a CliConfig,
}

impl<'a> ConfigCommand<'a> {
    pub fn execute(&self) -> VerifierResult<()> {
        match &self.action {
            ConfigAction::Show => self.show_config(),
            ConfigAction::Init { output } => self.init_config(output.as_deref()),
            ConfigAction::Validate { path } => self.validate_config(path),
            ConfigAction::Path => self.show_paths(),
        }
    }

    fn show_config(&self) -> VerifierResult<()> {
        let formatter = OutputFormatter::new(self.cli_config.format, !self.cli_config.no_color);
        let config = &self.cli_config.verifier;
        let output = formatter.format_config(config);
        print!("{output}");
        Ok(())
    }

    fn init_config(&self, output: Option<&std::path::Path>) -> VerifierResult<()> {
        let path = output
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("xr-verify.json"));

        let config = VerifierConfig::default();
        config.save(&path)?;
        tracing::info!("Configuration template written to {}", path.display());
        println!("Created configuration template: {}", path.display());
        Ok(())
    }

    fn validate_config(&self, path: &std::path::Path) -> VerifierResult<()> {
        let config = VerifierConfig::load(path)?;
        let problems = config.validation_problems();

        if problems.is_empty() {
            println!("Configuration is valid.");
        } else {
            println!("Configuration has {} issue(s):", problems.len());
            for (i, problem) in problems.iter().enumerate() {
                println!("  {}. {}", i + 1, problem);
            }
            return Err(VerifierError::Configuration(
                "Configuration validation failed".into(),
            ));
        }
        Ok(())
    }

    fn show_paths(&self) -> VerifierResult<()> {
        let paths = crate::config::CliConfig::config_search_paths();
        println!("Configuration file search order:");
        for (i, path) in paths.iter().enumerate() {
            let exists = path.exists();
            let marker = if exists { "✓" } else { "·" };
            println!("  {} {}. {}", marker, i + 1, path.display());
        }
        Ok(())
    }
}

// ─── Demo ──────────────────────────────────────────────────────────────────

pub struct DemoCommand<'a> {
    pub scene_type: DemoScene,
    pub output_path: Option<PathBuf>,
    pub cli_config: &'a CliConfig,
}

impl<'a> DemoCommand<'a> {
    pub fn execute(&self) -> VerifierResult<()> {
        let generator = DemoSceneGenerator::new();

        let scene = match self.scene_type {
            DemoScene::ButtonPanel => generator.simple_button_panel(),
            DemoScene::ControlRoom => generator.vr_control_room(),
            DemoScene::Manufacturing => generator.manufacturing_training(),
            DemoScene::Accessibility => generator.accessibility_showcase(),
        };

        let json = xr_scene::parser::scene_to_json(&scene)?;

        let output_path = self.output_path.clone().unwrap_or_else(|| {
            let name = match self.scene_type {
                DemoScene::ButtonPanel => "button_panel",
                DemoScene::ControlRoom => "control_room",
                DemoScene::Manufacturing => "manufacturing",
                DemoScene::Accessibility => "accessibility",
            };
            PathBuf::from(format!("{name}_demo.json"))
        });

        std::fs::write(&output_path, &json)?;
        println!("Demo scene written to {}", output_path.display());
        println!(
            "  Elements: {}, Dependencies: {}",
            scene.elements.len(),
            scene.dependencies.len()
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lint_command_builds_config() {
        let cli_config = CliConfig::default();
        let cmd = LintCommand {
            scene_path: PathBuf::from("test.json"),
            min_height: 0.4,
            max_height: 2.2,
            disabled_rules: vec![],
            output_path: None,
            cli_config: &cli_config,
        };
        assert_eq!(cmd.min_height, 0.4);
        assert_eq!(cmd.max_height, 2.2);
    }

    #[test]
    fn test_verify_command_builds_config() {
        let cli_config = CliConfig::default();
        let cmd = VerifyCommand {
            scene_path: PathBuf::from("test.json"),
            num_samples: 100,
            smt_timeout: 30.0,
            skip_tier2: false,
            fail_fast: false,
            target_kappa: 0.95,
            output_path: None,
            cli_config: &cli_config,
        };
        let config = cmd.build_verifier_config().unwrap();
        assert_eq!(config.sampling.num_samples, 100);
    }

    #[test]
    fn test_certify_command_structure() {
        let cli_config = CliConfig::default();
        let cmd = CertifyCommand {
            scene_path: PathBuf::from("test.json"),
            num_samples: 500,
            confidence: 0.95,
            output_path: None,
            generate_svg: false,
            cli_config: &cli_config,
        };
        assert_eq!(cmd.num_samples, 500);
    }

    #[test]
    fn test_inspect_command_flags() {
        let cli_config = CliConfig::default();
        let cmd = InspectCommand {
            scene_path: PathBuf::from("test.json"),
            show_elements: true,
            show_deps: false,
            show_devices: true,
            cli_config: &cli_config,
        };
        assert!(cmd.show_elements);
        assert!(!cmd.show_deps);
        assert!(cmd.show_devices);
    }

    #[test]
    fn test_demo_command_generates_scenes() {
        let gen = DemoSceneGenerator::new();
        let scene = gen.simple_button_panel();
        assert_eq!(scene.elements.len(), 5);

        let scene = gen.vr_control_room();
        assert!(scene.elements.len() >= 20);

        let scene = gen.manufacturing_training();
        assert!(!scene.elements.is_empty());
        assert!(!scene.dependencies.is_empty());
    }
}
