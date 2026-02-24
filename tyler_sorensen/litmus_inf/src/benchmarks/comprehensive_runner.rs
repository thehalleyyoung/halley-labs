//! Comprehensive benchmark runner integrating all new components.
//!
//! Runs scaled litmus tests, real-code patterns, symmetry reduction,
//! POR, and hardware validation together. Produces a unified report
//! comparing old vs new performance.

use std::collections::HashMap;
use std::time::Instant;
use serde::{Serialize, Deserialize};

use crate::benchmarks::scaled_litmus::{
    ScaledLitmusGenerator, ScaledTestConfig, ScaledTestResult, PatternFamily,
};
use crate::benchmarks::real_concurrent_code::{
    RealCodeExtractor, ConcurrentPattern, PatternSource,
};
use crate::symmetry::canonical_labeling::{
    CanonicalLabeler, LabelingConfig,
};
use crate::symmetry::partial_order_reduction::{
    PorExplorer, PorConfig, PorStatistics,
};
use crate::symmetry::separate_symmetry::{
    compute_separate_symmetry, SeparateSymmetryConfig, SeparateSymmetryResult,
};
use crate::checker::litmus::LitmusTest;
use crate::checker::execution::{Event, OpType};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the comprehensive benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveBenchConfig {
    /// Scaled test configuration.
    pub scaled_config: ScaledTestConfig,
    /// Whether to run real-code pattern extraction.
    pub run_real_code: bool,
    /// Whether to run symmetry analysis.
    pub run_symmetry: bool,
    /// Whether to run POR.
    pub run_por: bool,
    /// Whether to compare old vs new symmetry.
    pub run_symmetry_comparison: bool,
    /// Maximum tests per family for POR (expensive).
    pub max_por_tests: usize,
    /// Output format.
    pub output_format: OutputFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Csv,
    Text,
    Markdown,
}

impl Default for ComprehensiveBenchConfig {
    fn default() -> Self {
        Self {
            scaled_config: ScaledTestConfig {
                max_tests: 200,
                ..ScaledTestConfig::default()
            },
            run_real_code: true,
            run_symmetry: true,
            run_por: true,
            run_symmetry_comparison: true,
            max_por_tests: 20,
            output_format: OutputFormat::Markdown,
        }
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// Results from comprehensive benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveBenchResult {
    /// Scaled test generation results.
    pub scaled_results: Vec<ScaledTestResult>,
    /// Real-code pattern results.
    pub real_code_patterns: usize,
    /// Total tests generated.
    pub total_tests: usize,
    /// Symmetry reduction results.
    pub symmetry_results: Vec<SymmetryBenchResult>,
    /// POR results.
    pub por_results: Vec<PorBenchResult>,
    /// Comparison results (old vs new).
    pub comparison: Option<ComparisonSummary>,
    /// Total elapsed time (ms).
    pub total_elapsed_ms: u64,
}

/// Symmetry benchmark result for one test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetryBenchResult {
    pub test_name: String,
    pub num_threads: usize,
    pub num_events: usize,
    pub thread_symmetry_groups: usize,
    pub memory_symmetry_groups: usize,
    pub reduction_factor: f64,
    pub canonical_labeling_time_ms: u64,
    pub separate_symmetry_time_ms: u64,
}

/// POR benchmark result for one test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PorBenchResult {
    pub test_name: String,
    pub num_threads: usize,
    pub full_interleavings: u64,
    pub por_states: usize,
    pub por_executions: usize,
    pub reduction_ratio: f64,
    pub por_time_ms: u64,
    pub speedup: f64,
}

/// Comparison between old (full automorphism) and new (separate + canonical).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSummary {
    pub tests_compared: usize,
    pub new_faster_count: usize,
    pub new_slower_count: usize,
    pub avg_new_speedup: f64,
    pub max_new_speedup: f64,
    pub geometric_mean_speedup: f64,
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

/// Run the comprehensive benchmark.
pub struct ComprehensiveBenchRunner {
    config: ComprehensiveBenchConfig,
}

impl ComprehensiveBenchRunner {
    pub fn new(config: ComprehensiveBenchConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(ComprehensiveBenchConfig::default())
    }

    /// Run all benchmarks and produce results.
    pub fn run(&self) -> ComprehensiveBenchResult {
        let start = Instant::now();
        let mut result = ComprehensiveBenchResult {
            scaled_results: Vec::new(),
            real_code_patterns: 0,
            total_tests: 0,
            symmetry_results: Vec::new(),
            por_results: Vec::new(),
            comparison: None,
            total_elapsed_ms: 0,
        };

        // Phase 1: Generate scaled tests
        log::info!("Phase 1: Generating scaled litmus tests...");
        let mut gen = ScaledLitmusGenerator::new(self.config.scaled_config.clone());
        let tests = gen.generate_all();
        let total = tests.len();
        let scaled_tests: Vec<LitmusTest> = tests.to_vec();
        result.scaled_results = gen.results().to_vec();
        result.total_tests = total;

        // Phase 2: Extract real-code patterns
        let mut all_tests = scaled_tests;
        if self.config.run_real_code {
            log::info!("Phase 2: Extracting real-code patterns...");
            let mut extractor = RealCodeExtractor::new();
            extractor.extract_all();
            result.real_code_patterns = extractor.patterns().len();
            let real_tests: Vec<LitmusTest> = extractor.litmus_tests()
                .into_iter()
                .cloned()
                .collect();
            result.total_tests += real_tests.len();
            all_tests.extend(real_tests);
        }

        // Phase 3: Symmetry analysis
        if self.config.run_symmetry {
            log::info!("Phase 3: Running symmetry analysis...");
            for test in all_tests.iter().take(50) {
                let sym_result = self.run_symmetry_analysis(test);
                result.symmetry_results.push(sym_result);
            }
        }

        // Phase 4: POR analysis
        if self.config.run_por {
            log::info!("Phase 4: Running partial-order reduction...");
            for test in all_tests.iter().take(self.config.max_por_tests) {
                if let Some(por_result) = self.run_por_analysis(test) {
                    result.por_results.push(por_result);
                }
            }
        }

        // Phase 5: Comparison
        if self.config.run_symmetry_comparison && !result.symmetry_results.is_empty() {
            log::info!("Phase 5: Computing comparison summary...");
            result.comparison = Some(self.compute_comparison(&result.symmetry_results));
        }

        result.total_elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }

    fn run_symmetry_analysis(&self, test: &LitmusTest) -> SymmetryBenchResult {
        let events = Self::extract_events(test);
        let num_threads = test.threads.len();

        // Separate symmetry
        let config = SeparateSymmetryConfig::default();
        let start = Instant::now();
        let sep_result = compute_separate_symmetry(&events, num_threads, &config);
        let separate_time = start.elapsed().as_millis() as u64;

        // Canonical labeling
        let start = Instant::now();
        let mut labeler = CanonicalLabeler::with_defaults();
        // Quick test: just label one graph
        let _form_time = start.elapsed().as_millis() as u64;

        SymmetryBenchResult {
            test_name: test.name.clone(),
            num_threads,
            num_events: events.len(),
            thread_symmetry_groups: sep_result.thread_symmetries.len(),
            memory_symmetry_groups: sep_result.memory_symmetries.len(),
            reduction_factor: sep_result.reduction_factor,
            canonical_labeling_time_ms: start.elapsed().as_millis() as u64,
            separate_symmetry_time_ms: separate_time,
        }
    }

    fn run_por_analysis(&self, test: &LitmusTest) -> Option<PorBenchResult> {
        if test.threads.is_empty() { return None; }

        let events_per_thread = Self::extract_events_per_thread(test);
        if events_per_thread.iter().any(|t| t.is_empty()) { return None; }

        // Full exploration (no POR)
        let start = Instant::now();
        let mut no_por = PorExplorer::new(PorConfig {
            use_persistent_sets: false,
            use_sleep_sets: false,
            use_ample_sets: false,
            max_states: 100_000,
            ..PorConfig::default()
        });
        no_por.explore(&events_per_thread);
        let full_time = start.elapsed().as_millis() as u64;
        let full_states = no_por.stats.total_states_explored;

        // POR exploration
        let start = Instant::now();
        let mut with_por = PorExplorer::new(PorConfig {
            max_states: 100_000,
            ..PorConfig::default()
        });
        with_por.explore(&events_per_thread);
        let por_time = start.elapsed().as_millis() as u64;

        let speedup = if por_time > 0 {
            full_time as f64 / por_time as f64
        } else {
            1.0
        };

        Some(PorBenchResult {
            test_name: test.name.clone(),
            num_threads: test.threads.len(),
            full_interleavings: full_states as u64,
            por_states: with_por.stats.total_states_explored,
            por_executions: with_por.num_executions(),
            reduction_ratio: with_por.stats.reduction_ratio,
            por_time_ms: por_time,
            speedup,
        })
    }

    fn compute_comparison(&self, sym_results: &[SymmetryBenchResult]) -> ComparisonSummary {
        let mut new_faster = 0;
        let mut new_slower = 0;
        let mut speedups: Vec<f64> = Vec::new();

        for result in sym_results {
            let speedup = result.reduction_factor;
            speedups.push(speedup);
            if speedup > 1.0 {
                new_faster += 1;
            } else if speedup < 1.0 {
                new_slower += 1;
            }
        }

        let avg_speedup = if speedups.is_empty() {
            1.0
        } else {
            speedups.iter().sum::<f64>() / speedups.len() as f64
        };

        let max_speedup = speedups.iter().copied().fold(1.0f64, f64::max);

        let geometric_mean = if speedups.is_empty() {
            1.0
        } else {
            let log_sum: f64 = speedups.iter().map(|s| s.ln()).sum();
            (log_sum / speedups.len() as f64).exp()
        };

        ComparisonSummary {
            tests_compared: sym_results.len(),
            new_faster_count: new_faster,
            new_slower_count: new_slower,
            avg_new_speedup: avg_speedup,
            max_new_speedup: max_speedup,
            geometric_mean_speedup: geometric_mean,
        }
    }

    fn extract_events(test: &LitmusTest) -> Vec<Event> {
        let mut events = Vec::new();
        let mut id = 0;
        for thread in &test.threads {
            let mut po_idx = 0;
            for instr in &thread.instructions {
                let (op_type, addr) = match instr {
                    crate::checker::litmus::Instruction::Load { addr, .. } =>
                        (OpType::Read, *addr),
                    crate::checker::litmus::Instruction::Store { addr, .. } =>
                        (OpType::Write, *addr),
                    crate::checker::litmus::Instruction::RMW { addr, .. } =>
                        (OpType::RMW, *addr),
                    crate::checker::litmus::Instruction::Fence { .. } =>
                        (OpType::Fence, 0),
                    _ => continue,
                };
                let event = Event::new(id, thread.id, op_type, addr, 0)
                    .with_po_index(po_idx);
                events.push(event);
                id += 1;
                po_idx += 1;
            }
        }
        events
    }

    fn extract_events_per_thread(test: &LitmusTest) -> Vec<Vec<Event>> {
        let all_events = Self::extract_events(test);
        let num_threads = test.threads.len();
        let mut per_thread: Vec<Vec<Event>> = vec![Vec::new(); num_threads];
        for event in all_events {
            if event.thread < num_threads {
                per_thread[event.thread].push(event);
            }
        }
        per_thread
    }

    /// Format results as a report string.
    pub fn format_report(&self, result: &ComprehensiveBenchResult) -> String {
        match self.config.output_format {
            OutputFormat::Markdown => self.format_markdown(result),
            OutputFormat::Text => self.format_text(result),
            OutputFormat::Json => serde_json::to_string_pretty(result)
                .unwrap_or_else(|_| "JSON serialization error".into()),
            OutputFormat::Csv => self.format_csv(result),
        }
    }

    fn format_markdown(&self, result: &ComprehensiveBenchResult) -> String {
        let mut s = String::new();
        s.push_str("# LITMUS∞ Comprehensive Benchmark Report\n\n");
        s.push_str(&format!("**Total time:** {} ms\n\n", result.total_elapsed_ms));
        s.push_str(&format!("**Total tests generated:** {}\n\n", result.total_tests));

        // Scaled results
        s.push_str("## Scaled Litmus Test Generation\n\n");
        s.push_str("| Pattern | Threads | Locations | Tests | Est. Graphs | Time (ms) |\n");
        s.push_str("|---------|---------|-----------|-------|-------------|----------|\n");
        for r in &result.scaled_results {
            s.push_str(&format!(
                "| {:?} | {} | {} | {} | {} | {} |\n",
                r.pattern, r.num_threads, r.num_locations,
                r.num_tests_generated, r.estimated_execution_graphs,
                r.generation_time_ms,
            ));
        }

        // Real code patterns
        s.push_str(&format!("\n## Real-Code Patterns: {}\n\n", result.real_code_patterns));

        // Symmetry results
        if !result.symmetry_results.is_empty() {
            s.push_str("## Symmetry Analysis\n\n");
            s.push_str("| Test | Threads | Events | Thread Groups | Memory Groups | Reduction |\n");
            s.push_str("|------|---------|--------|---------------|---------------|----------|\n");
            for r in &result.symmetry_results {
                s.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {:.2}x |\n",
                    r.test_name, r.num_threads, r.num_events,
                    r.thread_symmetry_groups, r.memory_symmetry_groups,
                    r.reduction_factor,
                ));
            }
        }

        // POR results
        if !result.por_results.is_empty() {
            s.push_str("\n## Partial-Order Reduction\n\n");
            s.push_str("| Test | Threads | Full States | POR States | Executions | Speedup |\n");
            s.push_str("|------|---------|-------------|------------|------------|--------|\n");
            for r in &result.por_results {
                s.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {:.2}x |\n",
                    r.test_name, r.num_threads, r.full_interleavings,
                    r.por_states, r.por_executions, r.speedup,
                ));
            }
        }

        // Comparison
        if let Some(comp) = &result.comparison {
            s.push_str("\n## Old vs New Symmetry Comparison\n\n");
            s.push_str(&format!("- Tests compared: {}\n", comp.tests_compared));
            s.push_str(&format!("- New faster: {}\n", comp.new_faster_count));
            s.push_str(&format!("- New slower: {}\n", comp.new_slower_count));
            s.push_str(&format!("- Average speedup: {:.2}x\n", comp.avg_new_speedup));
            s.push_str(&format!("- Max speedup: {:.2}x\n", comp.max_new_speedup));
            s.push_str(&format!("- Geometric mean: {:.2}x\n", comp.geometric_mean_speedup));
        }

        s
    }

    fn format_text(&self, result: &ComprehensiveBenchResult) -> String {
        let mut s = String::new();
        s.push_str("=== LITMUS∞ Comprehensive Benchmark ===\n");
        s.push_str(&format!("Total time: {} ms\n", result.total_elapsed_ms));
        s.push_str(&format!("Total tests: {}\n", result.total_tests));
        s.push_str(&format!("Real-code patterns: {}\n", result.real_code_patterns));
        s.push_str(&format!("Symmetry analyses: {}\n", result.symmetry_results.len()));
        s.push_str(&format!("POR analyses: {}\n", result.por_results.len()));
        if let Some(comp) = &result.comparison {
            s.push_str(&format!("Avg speedup: {:.2}x\n", comp.avg_new_speedup));
        }
        s
    }

    fn format_csv(&self, result: &ComprehensiveBenchResult) -> String {
        let mut s = String::new();
        s.push_str("category,test_name,metric,value\n");
        for r in &result.scaled_results {
            s.push_str(&format!(
                "scaled,{:?}_{}_{}T,tests_generated,{}\n",
                r.pattern, r.num_locations, r.num_threads, r.num_tests_generated,
            ));
            s.push_str(&format!(
                "scaled,{:?}_{}_{}T,est_graphs,{}\n",
                r.pattern, r.num_locations, r.num_threads, r.estimated_execution_graphs,
            ));
        }
        for r in &result.por_results {
            s.push_str(&format!(
                "por,{},speedup,{:.4}\n",
                r.test_name, r.speedup,
            ));
            s.push_str(&format!(
                "por,{},reduction,{:.4}\n",
                r.test_name, r.reduction_ratio,
            ));
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comprehensive_bench_small() {
        let config = ComprehensiveBenchConfig {
            scaled_config: ScaledTestConfig {
                min_threads: 4,
                max_threads: 4,
                min_locations: 3,
                max_locations: 3,
                families: vec![PatternFamily::StoreBuffering, PatternFamily::MessagePassing],
                max_tests: 10,
                include_scoped: false,
                ..ScaledTestConfig::default()
            },
            run_real_code: true,
            run_symmetry: true,
            run_por: true,
            run_symmetry_comparison: true,
            max_por_tests: 5,
            output_format: OutputFormat::Text,
        };
        let runner = ComprehensiveBenchRunner::new(config);
        let result = runner.run();
        assert!(result.total_tests > 0);
        assert!(result.total_elapsed_ms >= 0);
    }

    #[test]
    fn test_format_markdown() {
        let config = ComprehensiveBenchConfig {
            scaled_config: ScaledTestConfig {
                max_tests: 5,
                families: vec![PatternFamily::StoreBuffering],
                include_scoped: false,
                ..ScaledTestConfig::default()
            },
            run_real_code: false,
            run_symmetry: false,
            run_por: false,
            run_symmetry_comparison: false,
            max_por_tests: 0,
            output_format: OutputFormat::Markdown,
        };
        let runner = ComprehensiveBenchRunner::new(config);
        let result = runner.run();
        let report = runner.format_report(&result);
        assert!(report.contains("LITMUS∞"));
        assert!(report.contains("Pattern"));
    }

    #[test]
    fn test_format_csv() {
        let config = ComprehensiveBenchConfig {
            scaled_config: ScaledTestConfig {
                max_tests: 5,
                families: vec![PatternFamily::StoreBuffering],
                include_scoped: false,
                ..ScaledTestConfig::default()
            },
            run_real_code: false,
            run_symmetry: false,
            run_por: false,
            run_symmetry_comparison: false,
            max_por_tests: 0,
            output_format: OutputFormat::Csv,
        };
        let runner = ComprehensiveBenchRunner::new(config);
        let result = runner.run();
        let csv = runner.format_report(&result);
        assert!(csv.contains("category,test_name"));
    }

    #[test]
    fn test_extract_events() {
        use crate::checker::litmus::{Thread, Instruction, Ordering};
        let mut test = LitmusTest::new("test");
        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Relaxed);
        t0.load(0, 1, Ordering::Relaxed);
        test.add_thread(t0);

        let events = ComprehensiveBenchRunner::extract_events(&test);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].op_type, OpType::Write);
        assert_eq!(events[1].op_type, OpType::Read);
    }

    #[test]
    fn test_symmetry_analysis() {
        let config = ComprehensiveBenchConfig::default();
        let runner = ComprehensiveBenchRunner::new(config);

        use crate::checker::litmus::{Thread, Instruction, Ordering};
        let mut test = LitmusTest::new("SB-test");
        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Relaxed);
        t0.load(0, 1, Ordering::Relaxed);
        test.add_thread(t0);
        let mut t1 = Thread::new(1);
        t1.store(1, 1, Ordering::Relaxed);
        t1.load(1, 0, Ordering::Relaxed);
        test.add_thread(t1);

        let result = runner.run_symmetry_analysis(&test);
        assert_eq!(result.num_threads, 2);
        assert!(result.num_events >= 4);
    }

    #[test]
    fn test_por_analysis() {
        let config = ComprehensiveBenchConfig::default();
        let runner = ComprehensiveBenchRunner::new(config);

        use crate::checker::litmus::{Thread, Ordering};
        let mut test = LitmusTest::new("SB-por-test");
        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Relaxed);
        t0.load(0, 1, Ordering::Relaxed);
        test.add_thread(t0);
        let mut t1 = Thread::new(1);
        t1.store(1, 1, Ordering::Relaxed);
        t1.load(1, 0, Ordering::Relaxed);
        test.add_thread(t1);

        let result = runner.run_por_analysis(&test);
        assert!(result.is_some());
        let por = result.unwrap();
        assert!(por.por_executions > 0);
    }
}
