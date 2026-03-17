//! CLI command implementations.
//!
//! Each command struct encapsulates its arguments and provides an `execute`
//! method that drives the appropriate compilation/rendering/linting pipeline.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

use crate::config::CliConfig;
use crate::diagnostics::{
    Diagnostic, DiagnosticEngine, ErrorExplainer, Severity, SourceCache, SourceLocation,
};
use crate::output::{
    CompilationSummary, ConstraintDetail, InfoSummary, JsonOutput, LintResult, OutputFormatter,
    ReportFormat, ReportGenerator, TypeCheckReport, WavMetadata, WavOutput,
};
use crate::pipeline::{CompilationPipeline, PipelineOptions, PipelineResult};
use crate::progress::{CompilationPhase, ProgressReporter, RenderProgress};

// ═══════════════════════════════════════════════════════════════════════════
//  CompileCommand
// ═══════════════════════════════════════════════════════════════════════════

/// Compile a `.soni` DSL file through the full pipeline.
#[derive(Debug)]
pub struct CompileCommand {
    pub input: PathBuf,
    pub output: Option<PathBuf>,
    pub emit_rust: bool,
    pub skip_wcet: bool,
}

impl CompileCommand {
    pub fn new(
        input: PathBuf,
        output: Option<PathBuf>,
        emit_rust: bool,
        skip_wcet: bool,
    ) -> Self {
        Self {
            input,
            output,
            emit_rust,
            skip_wcet,
        }
    }

    pub fn execute(&self, config: &CliConfig, diagnostics: &DiagnosticEngine) -> Result<()> {
        let mut diag = DiagnosticEngine::new(crate::diagnostics::DiagnosticFormat::Plain);

        let mut opts = PipelineOptions::from_config(config);
        opts.emit_rust = self.emit_rust;
        opts.verify_wcet = !self.skip_wcet && !config.skip_wcet;
        opts.output_path = self.output.clone().or_else(|| {
            let stem = self.input.file_stem().unwrap_or_default();
            let ext = if self.emit_rust { "rs" } else { "sonibin" };
            Some(self.input.with_file_name(format!("{}.{}", stem.to_string_lossy(), ext)))
        });

        let mut pipeline =
            CompilationPipeline::new(opts, config.verbose, config.quiet);

        let result = pipeline.run(&self.input, &mut diag)?;

        // Print diagnostics.
        let sources = load_source_cache(&self.input);
        if !config.quiet {
            let rendered = diag.render_all(&sources);
            if !rendered.is_empty() {
                eprint!("{}", rendered);
            }
            eprintln!("{}", OutputFormatter::compilation_summary(&result.summary));
        }

        if !result.success {
            bail!(
                "Compilation failed with {} error(s)",
                result.summary.errors
            );
        }

        // Write output.
        if let Some(ref path) = result.output_path {
            if self.emit_rust {
                let rust_src = generate_rust_source(&result);
                std::fs::write(path, &rust_src)
                    .with_context(|| format!("writing {}", path))?;
            } else {
                let json = JsonOutput::format(&result)?;
                std::fs::write(path, &json)
                    .with_context(|| format!("writing {}", path))?;
            }
            if !config.quiet {
                eprintln!("  Output: {}", path);
            }
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  RenderCommand
// ═══════════════════════════════════════════════════════════════════════════

/// Render a compiled audio graph to a WAV file.
#[derive(Debug)]
pub struct RenderCommand {
    pub graph: PathBuf,
    pub data: PathBuf,
    pub output: PathBuf,
    pub max_duration: Option<f64>,
}

impl RenderCommand {
    pub fn new(
        graph: PathBuf,
        data: PathBuf,
        output: PathBuf,
        max_duration: Option<f64>,
    ) -> Self {
        Self {
            graph,
            data,
            output,
            max_duration,
        }
    }

    pub fn execute(&self, config: &CliConfig, _diagnostics: &DiagnosticEngine) -> Result<()> {
        // Load compiled graph.
        let graph_json = std::fs::read_to_string(&self.graph)
            .with_context(|| format!("reading graph {}", self.graph.display()))?;
        let _graph: serde_json::Value = serde_json::from_str(&graph_json)
            .with_context(|| "parsing compiled graph JSON")?;

        // Load data source.
        let data_str = std::fs::read_to_string(&self.data)
            .with_context(|| format!("reading data {}", self.data.display()))?;
        let data_records = parse_data_source(&self.data, &data_str)?;
        let record_count = data_records.len();

        // Compute render parameters.
        let sample_rate = config.sample_rate;
        let channels = config.channels;
        let duration_secs = self
            .max_duration
            .or(config.max_render_duration)
            .unwrap_or_else(|| (record_count as f64 / 10.0).max(1.0).min(60.0));
        let total_samples = (duration_secs * sample_rate as f64) as u64 * channels as u64;

        let mut progress = RenderProgress::new(total_samples, sample_rate);
        let reporter = ProgressReporter::new(config.verbose, config.quiet);

        // Render (synthesise a simple tone per data record for now).
        let mut samples = Vec::with_capacity(total_samples as usize);
        let samples_per_record = if record_count > 0 {
            total_samples as usize / record_count
        } else {
            total_samples as usize
        };

        for (i, record) in data_records.iter().enumerate() {
            let freq = 200.0 + (record * 600.0);
            for s in 0..samples_per_record {
                let t = s as f32 / sample_rate as f32;
                let sample = (t * freq as f32 * std::f32::consts::TAU).sin() * 0.3;
                for _ch in 0..channels {
                    samples.push(sample);
                }
            }
            progress.advance(samples_per_record as u64 * channels as u64);
            if !config.quiet && i % 100 == 0 {
                reporter.render_tick(&progress);
            }
        }

        // Pad or truncate to exact length.
        samples.resize(total_samples as usize, 0.0);
        progress.advance(total_samples);
        reporter.render_done(&progress);

        // Normalise if configured.
        if config.normalize_output {
            let peak = WavOutput::peak_level(&samples);
            if peak > 0.0 && peak > 1.0 {
                let scale = 0.99 / peak;
                for s in &mut samples {
                    *s *= scale;
                }
            }
        }

        // Write WAV.
        let metadata = WavMetadata {
            title: Some(
                self.graph
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into(),
            ),
            software: Some("sonitype-cli".into()),
            ..Default::default()
        };
        WavOutput::write_with_metadata(
            &self.output,
            &samples,
            channels,
            sample_rate,
            config.bit_depth,
            &metadata,
        )?;

        if !config.quiet {
            let peak_db = 20.0 * WavOutput::peak_level(&samples).max(1e-10).log10();
            let rms_db = 20.0 * WavOutput::rms_level(&samples).max(1e-10).log10();
            eprintln!("Render complete:");
            eprintln!(
                "  Duration:   {:.2}s",
                total_samples as f64 / (sample_rate as f64 * channels as f64)
            );
            eprintln!("  Peak level: {:.1} dB", peak_db);
            eprintln!("  RMS level:  {:.1} dB", rms_db);
            eprintln!("  Output:     {}", self.output.display());
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  CheckCommand
// ═══════════════════════════════════════════════════════════════════════════

/// Type-check a DSL file without full compilation.
#[derive(Debug)]
pub struct CheckCommand {
    pub input: PathBuf,
}

impl CheckCommand {
    pub fn new(input: PathBuf) -> Self {
        Self { input }
    }

    pub fn execute(&self, config: &CliConfig, _diagnostics: &DiagnosticEngine) -> Result<()> {
        let mut diag = DiagnosticEngine::new(crate::diagnostics::DiagnosticFormat::Plain);

        let mut opts = PipelineOptions::from_config(config);
        // Only run up to type checking.
        opts.phases = vec![
            CompilationPhase::Parsing,
            CompilationPhase::SemanticAnalysis,
            CompilationPhase::TypeChecking,
        ];
        opts.verify_wcet = false;

        let mut pipeline = CompilationPipeline::new(opts, config.verbose, config.quiet);
        let result = pipeline.run(&self.input, &mut diag)?;

        let sources = load_source_cache(&self.input);
        if !config.quiet {
            let rendered = diag.render_all(&sources);
            if !rendered.is_empty() {
                eprint!("{}", rendered);
            }
            eprintln!("{}", OutputFormatter::type_check_report(&result.type_report));
            eprintln!("{}", diag.summary());
        }

        if diag.has_errors() {
            bail!("Type check failed");
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  PreviewCommand
// ═══════════════════════════════════════════════════════════════════════════

/// Quick low-quality preview rendering.
#[derive(Debug)]
pub struct PreviewCommand {
    pub input: PathBuf,
    pub duration: f64,
}

impl PreviewCommand {
    pub fn new(input: PathBuf, duration: f64) -> Self {
        Self { input, duration }
    }

    pub fn execute(&self, config: &CliConfig, _diagnostics: &DiagnosticEngine) -> Result<()> {
        let mut diag = DiagnosticEngine::new(crate::diagnostics::DiagnosticFormat::Plain);

        // Run a reduced pipeline.
        let mut opts = PipelineOptions::from_config(config);
        opts.optimization_level = 0; // skip optimisation for speed
        opts.verify_wcet = false;
        // Use a lower sample rate for preview.
        opts.sample_rate = 22050;

        let mut pipeline = CompilationPipeline::new(opts, false, config.quiet);
        let result = pipeline.run(&self.input, &mut diag)?;

        if diag.has_errors() {
            let sources = load_source_cache(&self.input);
            eprint!("{}", diag.render_all(&sources));
            bail!("Cannot preview: compilation errors");
        }

        // Generate a short preview WAV.
        let sr = 22050u32;
        let channels = 1u16;
        let total_samples = (self.duration * sr as f64) as usize;
        let stream_count = result.summary.stream_count.max(1);

        let mut samples = Vec::with_capacity(total_samples);
        for i in 0..total_samples {
            let t = i as f32 / sr as f32;
            let mut val = 0.0f32;
            for s in 0..stream_count {
                let freq = 220.0 * (s + 1) as f32;
                val += (t * freq * std::f32::consts::TAU).sin() * (0.3 / stream_count as f32);
            }
            samples.push(val);
        }

        let preview_path = std::env::temp_dir().join("sonitype_preview.wav");
        WavOutput::write(&preview_path, &samples, channels, sr, 16)?;

        if !config.quiet {
            eprintln!(
                "Preview rendered: {:.1}s, {} streams, {}",
                self.duration,
                stream_count,
                preview_path.display()
            );
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  LintCommand
// ═══════════════════════════════════════════════════════════════════════════

/// Perceptual lint mode: check for common sonification anti-patterns.
#[derive(Debug)]
pub struct LintCommand {
    pub input: PathBuf,
}

impl LintCommand {
    pub fn new(input: PathBuf) -> Self {
        Self { input }
    }

    pub fn execute(&self, config: &CliConfig, _diagnostics: &DiagnosticEngine) -> Result<()> {
        let source = std::fs::read_to_string(&self.input)
            .with_context(|| format!("reading {}", self.input.display()))?;

        let mut results = Vec::new();
        let lines: Vec<&str> = source.lines().collect();

        // ── Lint: masking warnings ──────────────────────────────
        let stream_freqs = extract_stream_frequencies(&source);
        for i in 0..stream_freqs.len() {
            for j in (i + 1)..stream_freqs.len() {
                let (name_a, lo_a, hi_a) = &stream_freqs[i];
                let (name_b, lo_b, hi_b) = &stream_freqs[j];
                if ranges_overlap(*lo_a, *hi_a, *lo_b, *hi_b) {
                    results.push(LintResult {
                        severity: "warning".into(),
                        code: "L100".into(),
                        message: format!(
                            "Potential masking: '{}' ({:.0}–{:.0} Hz) overlaps '{}' ({:.0}–{:.0} Hz)",
                            name_a, lo_a, hi_a, name_b, lo_b, hi_b
                        ),
                        location: format!("{}", self.input.display()),
                    });
                }
            }
        }

        // ── Lint: JND violations ────────────────────────────────
        for (name, lo, hi) in &stream_freqs {
            // Pitch JND ≈ 1 Hz at low frequencies, or ~0.3% at higher.
            let range = hi - lo;
            let jnd = lo * 0.003; // 0.3% of lower bound
            if range < jnd * 5.0 {
                results.push(LintResult {
                    severity: "warning".into(),
                    code: "L101".into(),
                    message: format!(
                        "JND risk for '{}': pitch range {:.0} Hz may be too narrow \
                         (< 5 JNDs at {:.0} Hz)",
                        name, range, lo
                    ),
                    location: format!("{}", self.input.display()),
                });
            }
        }

        // ── Lint: cognitive load ────────────────────────────────
        let stream_count = count_streams(&source);
        if stream_count > config.cognitive_load_budget as usize {
            results.push(LintResult {
                severity: "warning".into(),
                code: "L102".into(),
                message: format!(
                    "Cognitive load: {} simultaneous streams exceeds budget of {}",
                    stream_count, config.cognitive_load_budget
                ),
                location: "global".into(),
            });
        }

        // ── Lint: unused mappings ───────────────────────────────
        for (line_idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("mapping ") {
                if let Some(name) = trimmed
                    .strip_prefix("mapping ")
                    .and_then(|r| r.split_whitespace().next())
                {
                    let used = source.matches(name).count();
                    if used <= 1 {
                        results.push(LintResult {
                            severity: "info".into(),
                            code: "L103".into(),
                            message: format!("Mapping '{}' is defined but never referenced", name),
                            location: format!("{}:{}", self.input.display(), line_idx + 1),
                        });
                    }
                }
            }
        }

        // ── Lint: magic numbers ─────────────────────────────────
        for (line_idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("//") || trimmed.starts_with('#') {
                continue;
            }
            // Check for bare numeric literals that aren't part of a range.
            for word in trimmed.split_whitespace() {
                if let Ok(n) = word.parse::<f64>() {
                    if n.abs() > 1.0 && !trimmed.contains("..") {
                        results.push(LintResult {
                            severity: "info".into(),
                            code: "L104".into(),
                            message: format!(
                                "Magic number {} — consider naming this constant",
                                n
                            ),
                            location: format!("{}:{}", self.input.display(), line_idx + 1),
                        });
                    }
                }
            }
        }

        // ── Lint: suggest improvements ──────────────────────────
        if stream_count >= 3 && !source.contains("spatial") && !source.contains("pan") {
            results.push(LintResult {
                severity: "hint".into(),
                code: "L105".into(),
                message: "With 3+ streams, consider spatial panning for better segregation"
                    .into(),
                location: "global".into(),
            });
        }

        // Output.
        if !config.quiet {
            eprintln!("{}", OutputFormatter::lint_report(&results));
        }

        if results.iter().any(|r| r.severity == "error") {
            bail!("Lint found errors");
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  InfoCommand
// ═══════════════════════════════════════════════════════════════════════════

/// Display information about a DSL file.
#[derive(Debug)]
pub struct InfoCommand {
    pub input: PathBuf,
}

impl InfoCommand {
    pub fn new(input: PathBuf) -> Self {
        Self { input }
    }

    pub fn execute(&self, config: &CliConfig, _diagnostics: &DiagnosticEngine) -> Result<()> {
        let source = std::fs::read_to_string(&self.input)
            .with_context(|| format!("reading {}", self.input.display()))?;

        let stream_count = count_streams(&source);
        let mapping_count = source
            .lines()
            .filter(|l| l.trim().starts_with("mapping "))
            .count();
        let stream_freqs = extract_stream_frequencies(&source);

        // Estimate cognitive load (simple: 1 per stream).
        let cognitive_load = stream_count as f64;

        // Estimate WCET: ~0.05 ms per stream node × 4 nodes/stream + overhead.
        let wcet_estimate = (stream_count as f64 * 4.0 * 0.05) + 0.5;

        // Compute spectral layout in Bark bands.
        let spectral_layout: Vec<(String, u8)> = stream_freqs
            .iter()
            .map(|(name, lo, hi)| {
                let center = (lo + hi) / 2.0;
                let bark = hz_to_bark(center);
                (name.clone(), bark as u8)
            })
            .collect();

        let info = InfoSummary {
            file: self.input.display().to_string(),
            stream_count,
            mapping_count,
            estimated_cognitive_load: cognitive_load,
            cognitive_budget: config.cognitive_load_budget,
            wcet_estimate_ms: wcet_estimate,
            spectral_layout,
        };

        if !config.quiet {
            eprintln!("{}", OutputFormatter::info_summary(&info));
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  InitCommand
// ═══════════════════════════════════════════════════════════════════════════

/// Create a new sonification project from a template.
#[derive(Debug)]
pub struct InitCommand {
    pub path: PathBuf,
    pub template: String,
}

impl InitCommand {
    pub fn new(path: PathBuf, template: String) -> Self {
        Self { path, template }
    }

    pub fn execute(&self, config: &CliConfig, _diagnostics: &DiagnosticEngine) -> Result<()> {
        // Create project directory.
        std::fs::create_dir_all(&self.path)
            .with_context(|| format!("creating directory {}", self.path.display()))?;

        // Write template .soni file.
        let soni_content = match self.template.as_str() {
            "basic" => TEMPLATE_BASIC,
            "multi-stream" | "multi_stream" => TEMPLATE_MULTI_STREAM,
            "spatial" => TEMPLATE_SPATIAL,
            other => bail!("Unknown template: '{}'. Use basic, multi-stream, or spatial.", other),
        };
        let soni_path = self.path.join("main.soni");
        std::fs::write(&soni_path, soni_content)
            .with_context(|| format!("writing {}", soni_path.display()))?;

        // Write sample data CSV.
        let data_path = self.path.join("data.csv");
        std::fs::write(&data_path, SAMPLE_DATA_CSV)
            .with_context(|| format!("writing {}", data_path.display()))?;

        // Write config file.
        let config_content = CliConfig::default().to_toml_string()?;
        let config_path = self.path.join("sonitype.toml");
        std::fs::write(&config_path, &config_content)
            .with_context(|| format!("writing {}", config_path.display()))?;

        if !config.quiet {
            eprintln!("Created new SoniType project:");
            eprintln!("  {}/", self.path.display());
            eprintln!("    main.soni        — sonification definition");
            eprintln!("    data.csv         — sample data");
            eprintln!("    sonitype.toml    — configuration");
            eprintln!();
            eprintln!("Get started:");
            eprintln!("  cd {}", self.path.display());
            eprintln!("  sonitype check main.soni");
            eprintln!("  sonitype compile main.soni");
        }

        Ok(())
    }
}

// ── Templates ───────────────────────────────────────────────────────────────

const TEMPLATE_BASIC: &str = r#"// SoniType — Basic sonification template
//
// Maps a single data column to pitch.

data source = "data.csv"

stream temperature {
  pitch: source.temperature -> 200..800 Hz
  amplitude: -6 dB
}

compose output {
  temperature
}
"#;

const TEMPLATE_MULTI_STREAM: &str = r#"// SoniType — Multi-stream sonification template
//
// Maps two data columns to separate auditory streams with
// spectral separation enforced by the type system.

data source = "data.csv"

stream temperature {
  pitch: source.temperature -> 200..600 Hz  // Bark bands 2–5
  amplitude: source.importance -> -12..-3 dB
}

stream pressure {
  pitch: source.pressure -> 1000..3000 Hz  // Bark bands 10–15
  timbre: "triangle"
  amplitude: -6 dB
}

compose output {
  temperature
  pressure
}
"#;

const TEMPLATE_SPATIAL: &str = r#"// SoniType — Spatial sonification template
//
// Three streams separated in both spectral and spatial dimensions.

data source = "data.csv"

stream temperature {
  pitch: source.temperature -> 200..500 Hz
  pan: -0.8
  amplitude: -6 dB
}

stream pressure {
  pitch: source.pressure -> 800..2000 Hz
  pan: 0.0
  amplitude: -6 dB
}

stream humidity {
  pitch: source.humidity -> 3000..5000 Hz
  pan: 0.8
  amplitude: -9 dB
}

compose output {
  temperature
  pressure
  humidity
}
"#;

const SAMPLE_DATA_CSV: &str = r#"time,temperature,pressure,humidity,importance
0.0,22.5,1013.2,45.0,0.5
0.1,22.7,1013.1,45.2,0.6
0.2,23.0,1012.9,46.0,0.7
0.3,23.4,1012.8,46.5,0.8
0.4,23.8,1012.5,47.0,0.9
0.5,24.1,1012.3,47.5,0.7
0.6,24.0,1012.4,47.2,0.6
0.7,23.8,1012.6,46.8,0.5
0.8,23.5,1012.8,46.3,0.4
0.9,23.2,1013.0,45.8,0.3
"#;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn load_source_cache(path: &Path) -> SourceCache {
    let mut cache = SourceCache::new();
    let _ = cache.load(path);
    cache
}

fn generate_rust_source(result: &PipelineResult) -> String {
    let mut src = String::new();
    src.push_str("//! Auto-generated by sonitype-cli.\n");
    src.push_str("//! Do not edit.\n\n");
    src.push_str(&format!(
        "// Streams: {}, Nodes: {}, WCET: {:.2} ms\n\n",
        result.summary.stream_count, result.summary.node_count, result.summary.wcet_ms,
    ));
    src.push_str("pub fn render(data: &[f64], sample_rate: u32, buffer: &mut [f32]) {\n");
    src.push_str("    let _streams = ");
    src.push_str(&format!("{};\n", result.summary.stream_count));
    src.push_str("    for (i, sample) in buffer.iter_mut().enumerate() {\n");
    src.push_str("        let t = i as f32 / sample_rate as f32;\n");
    src.push_str("        *sample = (t * 440.0 * std::f32::consts::TAU).sin() * 0.3;\n");
    src.push_str("    }\n");
    src.push_str("}\n");
    src
}

/// Parse a data source file (CSV or JSON) into a vector of normalised [0,1] values.
fn parse_data_source(path: &Path, content: &str) -> Result<Vec<f64>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("csv");

    match ext {
        "csv" => {
            let mut values = Vec::new();
            let mut lines = content.lines();
            let _header = lines.next(); // skip header
            for line in lines {
                if line.trim().is_empty() {
                    continue;
                }
                // Take the second column (first data column after time).
                if let Some(val_str) = line.split(',').nth(1) {
                    if let Ok(v) = val_str.trim().parse::<f64>() {
                        values.push(v);
                    }
                }
            }
            // Normalise to [0, 1].
            let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = (max - min).max(1e-10);
            Ok(values.iter().map(|v| (v - min) / range).collect())
        }
        "json" => {
            let parsed: serde_json::Value =
                serde_json::from_str(content).context("parsing JSON data")?;
            if let Some(arr) = parsed.as_array() {
                let values: Vec<f64> = arr
                    .iter()
                    .filter_map(|v| v.as_f64().or_else(|| v.get("value").and_then(|v| v.as_f64())))
                    .collect();
                let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let range = (max - min).max(1e-10);
                Ok(values.iter().map(|v| (v - min) / range).collect())
            } else {
                bail!("JSON data must be an array");
            }
        }
        other => bail!("Unsupported data format: .{}", other),
    }
}

/// Count the number of `stream` declarations in a source string.
fn count_streams(source: &str) -> usize {
    source
        .lines()
        .filter(|l| l.trim().starts_with("stream "))
        .count()
}

/// Extract stream names and their frequency ranges from the source.
fn extract_stream_frequencies(source: &str) -> Vec<(String, f64, f64)> {
    let mut result = Vec::new();
    let mut current_stream: Option<String> = None;

    for line in source.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("stream ") {
            current_stream = trimmed
                .strip_prefix("stream ")
                .and_then(|r| r.split_whitespace().next())
                .map(String::from);
        }

        if let Some(ref name) = current_stream {
            if trimmed.starts_with("pitch:") || trimmed.contains("-> ") {
                if let Some((lo, hi)) = parse_frequency_range(trimmed) {
                    result.push((name.clone(), lo, hi));
                    current_stream = None;
                }
            }
        }

        if trimmed == "}" {
            current_stream = None;
        }
    }

    result
}

/// Parse a frequency range like "200..800 Hz" from a line.
fn parse_frequency_range(line: &str) -> Option<(f64, f64)> {
    // Look for pattern: NUM..NUM
    let stripped = line
        .replace("Hz", "")
        .replace("hz", "")
        .replace("pitch:", "")
        .replace("->", " ");

    for part in stripped.split_whitespace() {
        if let Some((lo_s, hi_s)) = part.split_once("..") {
            if let (Ok(lo), Ok(hi)) = (lo_s.trim().parse::<f64>(), hi_s.trim().parse::<f64>()) {
                return Some((lo, hi));
            }
        }
    }
    None
}

/// Check if two frequency ranges overlap.
fn ranges_overlap(lo_a: f64, hi_a: f64, lo_b: f64, hi_b: f64) -> bool {
    lo_a < hi_b && lo_b < hi_a
}

/// Convert Hz to Bark scale (Traunmüller 1990 approximation).
fn hz_to_bark(freq: f64) -> f64 {
    let z = (26.81 * freq / (1960.0 + freq)) - 0.53;
    z.max(0.0)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_streams_basic() {
        let src = "stream a {\n}\nstream b {\n}\n";
        assert_eq!(count_streams(src), 2);
    }

    #[test]
    fn count_streams_empty() {
        assert_eq!(count_streams(""), 0);
    }

    #[test]
    fn parse_frequency_range_basic() {
        let line = "pitch: source.temp -> 200..800 Hz";
        let (lo, hi) = parse_frequency_range(line).unwrap();
        assert!((lo - 200.0).abs() < 0.1);
        assert!((hi - 800.0).abs() < 0.1);
    }

    #[test]
    fn parse_frequency_range_no_range() {
        assert!(parse_frequency_range("amplitude: -6 dB").is_none());
    }

    #[test]
    fn extract_streams_from_template() {
        let freqs = extract_stream_frequencies(TEMPLATE_MULTI_STREAM);
        assert_eq!(freqs.len(), 2);
        assert_eq!(freqs[0].0, "temperature");
        assert_eq!(freqs[1].0, "pressure");
    }

    #[test]
    fn ranges_overlap_yes() {
        assert!(ranges_overlap(200.0, 600.0, 500.0, 1000.0));
    }

    #[test]
    fn ranges_overlap_no() {
        assert!(!ranges_overlap(200.0, 400.0, 500.0, 1000.0));
    }

    #[test]
    fn hz_to_bark_low() {
        let b = hz_to_bark(100.0);
        assert!(b > 0.0 && b < 2.0);
    }

    #[test]
    fn hz_to_bark_high() {
        let b = hz_to_bark(8000.0);
        assert!(b > 15.0 && b < 25.0);
    }

    #[test]
    fn parse_csv_data() {
        let csv = "time,val\n0,10\n1,20\n2,30\n";
        let vals = parse_data_source(Path::new("d.csv"), csv).unwrap();
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 0.0).abs() < 0.01);
        assert!((vals[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn parse_json_data() {
        let json = "[1.0, 2.0, 3.0, 4.0]";
        let vals = parse_data_source(Path::new("d.json"), json).unwrap();
        assert_eq!(vals.len(), 4);
    }

    #[test]
    fn parse_json_data_objects() {
        let json = r#"[{"value":10},{"value":20}]"#;
        let vals = parse_data_source(Path::new("d.json"), json).unwrap();
        assert_eq!(vals.len(), 2);
    }

    #[test]
    fn parse_unsupported_format() {
        assert!(parse_data_source(Path::new("d.xml"), "").is_err());
    }

    #[test]
    fn generate_rust_source_contains_fn() {
        let result = PipelineResult {
            success: true,
            summary: CompilationSummary {
                success: true,
                stream_count: 2,
                node_count: 10,
                wcet_ms: 1.0,
                total_time_ms: 50.0,
                errors: 0,
                warnings: 0,
            },
            type_report: TypeCheckReport {
                errors: 0,
                warnings: 0,
                constraints_satisfied: 3,
                constraints_total: 3,
                constraint_details: vec![],
            },
            lint_results: vec![],
            wcet_report: None,
            optimization_stats: crate::pipeline::OptimizationStats::default(),
            timing: vec![],
            output_path: None,
        };
        let src = generate_rust_source(&result);
        assert!(src.contains("pub fn render"));
        assert!(src.contains("Streams: 2"));
    }

    #[test]
    fn init_command_creates_files() {
        let dir = std::env::temp_dir().join(format!("sonitype_init_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cmd = InitCommand::new(dir.clone(), "basic".into());
        let config = CliConfig { quiet: true, ..CliConfig::default() };
        let diag = DiagnosticEngine::new(crate::diagnostics::DiagnosticFormat::Plain);
        cmd.execute(&config, &diag).unwrap();

        assert!(dir.join("main.soni").exists());
        assert!(dir.join("data.csv").exists());
        assert!(dir.join("sonitype.toml").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn init_command_invalid_template() {
        let dir = std::env::temp_dir().join(format!("sonitype_init_bad_{}", std::process::id()));
        let cmd = InitCommand::new(dir.clone(), "nonexistent".into());
        let config = CliConfig { quiet: true, ..CliConfig::default() };
        let diag = DiagnosticEngine::new(crate::diagnostics::DiagnosticFormat::Plain);
        let result = cmd.execute(&config, &diag);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compile_command_struct() {
        let cmd = CompileCommand::new(
            PathBuf::from("test.soni"),
            Some(PathBuf::from("out.sonibin")),
            false,
            true,
        );
        assert_eq!(cmd.input, PathBuf::from("test.soni"));
        assert!(cmd.skip_wcet);
    }

    #[test]
    fn render_command_struct() {
        let cmd = RenderCommand::new(
            PathBuf::from("graph.json"),
            PathBuf::from("data.csv"),
            PathBuf::from("out.wav"),
            Some(5.0),
        );
        assert_eq!(cmd.max_duration, Some(5.0));
    }

    #[test]
    fn check_command_struct() {
        let cmd = CheckCommand::new(PathBuf::from("test.soni"));
        assert_eq!(cmd.input, PathBuf::from("test.soni"));
    }

    #[test]
    fn preview_command_struct() {
        let cmd = PreviewCommand::new(PathBuf::from("test.soni"), 2.0);
        assert_eq!(cmd.duration, 2.0);
    }

    #[test]
    fn lint_command_on_template() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("sonitype_lint_test_{}.soni", std::process::id()));
        std::fs::write(&path, TEMPLATE_MULTI_STREAM).unwrap();

        let cmd = LintCommand::new(path.clone());
        let config = CliConfig { quiet: true, ..CliConfig::default() };
        let diag = DiagnosticEngine::new(crate::diagnostics::DiagnosticFormat::Plain);
        // Should not error (warnings are OK).
        cmd.execute(&config, &diag).unwrap();

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn info_command_on_template() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("sonitype_info_test_{}.soni", std::process::id()));
        std::fs::write(&path, TEMPLATE_SPATIAL).unwrap();

        let cmd = InfoCommand::new(path.clone());
        let config = CliConfig { quiet: true, ..CliConfig::default() };
        let diag = DiagnosticEngine::new(crate::diagnostics::DiagnosticFormat::Plain);
        cmd.execute(&config, &diag).unwrap();

        let _ = std::fs::remove_file(&path);
    }
}
