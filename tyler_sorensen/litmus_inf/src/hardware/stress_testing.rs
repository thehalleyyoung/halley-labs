//! Stress-testing techniques for GPU litmus tests.
//!
//! Implements stress patterns from Wickerson et al. to increase the probability
//! of observing weak memory behaviours on GPU hardware: bank conflicts, cache
//! pressure, padding/striding, scheduling perturbation, and memory pressure.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::checker::litmus::LitmusTest;

use super::litmus_runner::{
    GpuBackend, HardwareConfig, HardwareResult, LitmusRunner, RunnerError,
    StressMode, TestOutcome,
};

// ---------------------------------------------------------------------------
// Stress patterns
// ---------------------------------------------------------------------------

/// Individual stress patterns that can be combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StressPattern {
    /// Generate bank conflicts to slow down memory accesses.
    BankConflict,
    /// Apply cache pressure by touching large memory regions.
    CachePressure,
    /// Insert padding between test variables.
    Padding,
    /// Use strided access patterns to defeat prefetching.
    Striding,
    /// Inject scheduling noise through extra compute work.
    SchedulingNoise,
    /// Apply memory pressure by allocating/freeing large buffers.
    MemoryPressure,
    /// Vary timing between threads via spin loops.
    TimingVariation,
    /// Pre-heat caches before the test.
    CacheWarmup,
    /// Flush caches before the test.
    CacheFlush,
    /// Use round-robin thread-to-core pinning.
    ThreadPinning,
}

impl fmt::Display for StressPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StressPattern::BankConflict => write!(f, "bank_conflict"),
            StressPattern::CachePressure => write!(f, "cache_pressure"),
            StressPattern::Padding => write!(f, "padding"),
            StressPattern::Striding => write!(f, "striding"),
            StressPattern::SchedulingNoise => write!(f, "scheduling_noise"),
            StressPattern::MemoryPressure => write!(f, "memory_pressure"),
            StressPattern::TimingVariation => write!(f, "timing_variation"),
            StressPattern::CacheWarmup => write!(f, "cache_warmup"),
            StressPattern::CacheFlush => write!(f, "cache_flush"),
            StressPattern::ThreadPinning => write!(f, "thread_pinning"),
        }
    }
}

impl StressPattern {
    /// All available patterns.
    pub fn all() -> &'static [StressPattern] {
        &[
            StressPattern::BankConflict,
            StressPattern::CachePressure,
            StressPattern::Padding,
            StressPattern::Striding,
            StressPattern::SchedulingNoise,
            StressPattern::MemoryPressure,
            StressPattern::TimingVariation,
            StressPattern::CacheWarmup,
            StressPattern::CacheFlush,
            StressPattern::ThreadPinning,
        ]
    }

    /// Patterns appropriate for a given stress level.
    pub fn for_level(mode: StressMode) -> Vec<StressPattern> {
        match mode {
            StressMode::None => vec![],
            StressMode::Light => vec![
                StressPattern::Padding,
                StressPattern::TimingVariation,
            ],
            StressMode::Medium => vec![
                StressPattern::BankConflict,
                StressPattern::Padding,
                StressPattern::Striding,
                StressPattern::CachePressure,
            ],
            StressMode::Heavy => vec![
                StressPattern::BankConflict,
                StressPattern::CachePressure,
                StressPattern::Padding,
                StressPattern::Striding,
                StressPattern::SchedulingNoise,
                StressPattern::MemoryPressure,
                StressPattern::TimingVariation,
            ],
            StressMode::Custom(_) => vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// Stress configuration
// ---------------------------------------------------------------------------

/// Full stress-testing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressConfig {
    /// Active stress patterns.
    pub patterns: Vec<StressPattern>,

    // -- Bank conflicts --
    /// Number of bank-conflicting accesses to generate.
    pub bank_conflict_count: u32,
    /// Stride (in 4-byte words) for bank conflicts (typically 32 for GPU).
    pub bank_conflict_stride: u32,

    // -- Cache pressure --
    /// Size of scratch buffer for cache pressure (bytes).
    pub cache_pressure_size: u64,
    /// Number of cache-pressure iterations.
    pub cache_pressure_iterations: u32,

    // -- Padding / striding --
    /// Padding (in bytes) between test variables.
    pub padding_bytes: u32,
    /// Stride (in bytes) for strided access.
    pub stride_bytes: u32,

    // -- Scheduling noise --
    /// Maximum spin-loop iterations for scheduling noise.
    pub scheduling_noise_max: u32,
    /// Whether the noise amount is randomised per-thread.
    pub scheduling_noise_random: bool,

    // -- Memory pressure --
    /// Number of extra allocations for memory pressure.
    pub memory_pressure_allocs: u32,
    /// Size of each allocation (bytes).
    pub memory_pressure_alloc_size: u64,

    // -- Timing variation --
    /// Maximum spin iterations for timing variation.
    pub timing_spin_max: u32,
    /// Whether to use random timing per iteration.
    pub timing_random: bool,

    // -- Cache warmup / flush --
    /// Warmup iterations.
    pub warmup_iterations: u32,

    // -- Thread pinning --
    /// Whether to pin threads round-robin to cores.
    pub pin_threads: bool,

    /// How many rounds of stress to interleave between test iterations.
    pub stress_rounds: u32,
}

impl Default for StressConfig {
    fn default() -> Self {
        Self {
            patterns: vec![],
            bank_conflict_count: 32,
            bank_conflict_stride: 32,
            cache_pressure_size: 1 << 20, // 1 MB
            cache_pressure_iterations: 4,
            padding_bytes: 256,
            stride_bytes: 128,
            scheduling_noise_max: 1000,
            scheduling_noise_random: true,
            memory_pressure_allocs: 4,
            memory_pressure_alloc_size: 1 << 20,
            timing_spin_max: 500,
            timing_random: true,
            warmup_iterations: 10,
            pin_threads: false,
            stress_rounds: 1,
        }
    }
}

impl StressConfig {
    /// Create a config from a `StressMode`.
    pub fn from_mode(mode: StressMode) -> Self {
        let patterns = StressPattern::for_level(mode);
        let mut cfg = Self {
            patterns,
            ..Default::default()
        };
        match mode {
            StressMode::Heavy => {
                cfg.cache_pressure_size = 4 << 20;
                cfg.cache_pressure_iterations = 8;
                cfg.scheduling_noise_max = 5000;
                cfg.memory_pressure_allocs = 8;
                cfg.stress_rounds = 3;
            }
            StressMode::Medium => {
                cfg.cache_pressure_size = 2 << 20;
                cfg.stress_rounds = 2;
            }
            _ => {}
        }
        cfg
    }

    /// Create a config with specific patterns.
    pub fn with_patterns(patterns: Vec<StressPattern>) -> Self {
        Self {
            patterns,
            ..Default::default()
        }
    }

    /// Check if a pattern is enabled.
    pub fn has_pattern(&self, pat: StressPattern) -> bool {
        self.patterns.contains(&pat)
    }

    /// Generate a CUDA stress preamble inserted before the test body.
    pub fn generate_cuda_preamble(&self, thread_var: &str) -> String {
        let mut code = String::new();
        code.push_str("    // === STRESS PREAMBLE ===\n");

        if self.has_pattern(StressPattern::BankConflict) {
            code.push_str(&self.gen_bank_conflict_cuda(thread_var));
        }
        if self.has_pattern(StressPattern::CachePressure) {
            code.push_str(&self.gen_cache_pressure_cuda(thread_var));
        }
        if self.has_pattern(StressPattern::SchedulingNoise) {
            code.push_str(&self.gen_scheduling_noise_cuda(thread_var));
        }
        if self.has_pattern(StressPattern::TimingVariation) {
            code.push_str(&self.gen_timing_variation_cuda(thread_var));
        }

        code.push_str("    // === END STRESS PREAMBLE ===\n");
        code
    }

    /// Generate an OpenCL stress preamble.
    pub fn generate_opencl_preamble(&self, thread_var: &str) -> String {
        let mut code = String::new();
        code.push_str("    // === STRESS PREAMBLE ===\n");

        if self.has_pattern(StressPattern::BankConflict) {
            code.push_str(&self.gen_bank_conflict_opencl(thread_var));
        }
        if self.has_pattern(StressPattern::CachePressure) {
            code.push_str(&self.gen_cache_pressure_opencl(thread_var));
        }
        if self.has_pattern(StressPattern::SchedulingNoise) {
            code.push_str(&self.gen_scheduling_noise_opencl(thread_var));
        }
        if self.has_pattern(StressPattern::TimingVariation) {
            code.push_str(&self.gen_timing_variation_opencl(thread_var));
        }

        code.push_str("    // === END STRESS PREAMBLE ===\n");
        code
    }

    /// Generate a Vulkan GLSL stress preamble.
    pub fn generate_vulkan_preamble(&self, thread_var: &str) -> String {
        let mut code = String::new();
        code.push_str("    // === STRESS PREAMBLE ===\n");

        if self.has_pattern(StressPattern::BankConflict) {
            code.push_str(&self.gen_bank_conflict_vulkan(thread_var));
        }
        if self.has_pattern(StressPattern::SchedulingNoise) {
            code.push_str(&self.gen_scheduling_noise_vulkan(thread_var));
        }

        code.push_str("    // === END STRESS PREAMBLE ===\n");
        code
    }

    /// Generate a Metal stress preamble.
    pub fn generate_metal_preamble(&self, thread_var: &str) -> String {
        let mut code = String::new();
        code.push_str("    // === STRESS PREAMBLE ===\n");

        if self.has_pattern(StressPattern::BankConflict) {
            code.push_str(&self.gen_bank_conflict_metal(thread_var));
        }
        if self.has_pattern(StressPattern::SchedulingNoise) {
            code.push_str(&self.gen_scheduling_noise_metal(thread_var));
        }

        code.push_str("    // === END STRESS PREAMBLE ===\n");
        code
    }

    // -- CUDA generators --

    fn gen_bank_conflict_cuda(&self, tid: &str) -> String {
        format!(
            "    __shared__ volatile int stress_smem[{stride} * {count}];\n\
    for (int _bc = 0; _bc < {count}; _bc++) {{\n\
        stress_smem[{tid} * {stride} + _bc] = _bc;\n\
    }}\n\
    __syncthreads();\n",
            stride = self.bank_conflict_stride,
            count = self.bank_conflict_count,
            tid = tid,
        )
    }

    fn gen_cache_pressure_cuda(&self, tid: &str) -> String {
        format!(
            "    for (int _cp = 0; _cp < {iters}; _cp++) {{\n\
        int _idx = ({tid} * 127 + _cp * 63) % {size};\n\
        stress_buf[_idx] = _cp;\n\
    }}\n",
            iters = self.cache_pressure_iterations,
            tid = tid,
            size = self.cache_pressure_size,
        )
    }

    fn gen_scheduling_noise_cuda(&self, tid: &str) -> String {
        if self.scheduling_noise_random {
            format!(
                "    {{\n\
            unsigned int _sn = ({tid} * 1103515245u + 12345u) % {max}u;\n\
            for (unsigned int _s = 0; _s < _sn; _s++) {{ __threadfence(); }}\n\
        }}\n",
                tid = tid,
                max = self.scheduling_noise_max,
            )
        } else {
            format!(
                "    for (int _s = 0; _s < {}; _s++) {{ __threadfence(); }}\n",
                self.scheduling_noise_max,
            )
        }
    }

    fn gen_timing_variation_cuda(&self, tid: &str) -> String {
        format!(
            "    {{\n\
        unsigned int _tv = ({tid} * 2654435761u) % {max}u;\n\
        for (unsigned int _t = 0; _t < _tv; _t++) {{ asm volatile(\"\"); }}\n\
    }}\n",
            tid = tid,
            max = self.timing_spin_max,
        )
    }

    // -- OpenCL generators --

    fn gen_bank_conflict_opencl(&self, tid: &str) -> String {
        format!(
            "    __local volatile int stress_smem[{stride} * {count}];\n\
    for (int _bc = 0; _bc < {count}; _bc++) {{\n\
        stress_smem[{tid} * {stride} + _bc] = _bc;\n\
    }}\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n",
            stride = self.bank_conflict_stride,
            count = self.bank_conflict_count,
            tid = tid,
        )
    }

    fn gen_cache_pressure_opencl(&self, tid: &str) -> String {
        format!(
            "    for (int _cp = 0; _cp < {iters}; _cp++) {{\n\
        int _idx = ({tid} * 127 + _cp * 63) % {size};\n\
        stress_buf[_idx] = _cp;\n\
    }}\n",
            iters = self.cache_pressure_iterations,
            tid = tid,
            size = self.cache_pressure_size,
        )
    }

    fn gen_scheduling_noise_opencl(&self, tid: &str) -> String {
        format!(
            "    {{\n\
        uint _sn = ({tid} * 1103515245u + 12345u) % {max}u;\n\
        for (uint _s = 0; _s < _sn; _s++) {{ mem_fence(CLK_GLOBAL_MEM_FENCE); }}\n\
    }}\n",
            tid = tid,
            max = self.scheduling_noise_max,
        )
    }

    fn gen_timing_variation_opencl(&self, tid: &str) -> String {
        format!(
            "    {{\n\
        uint _tv = ({tid} * 2654435761u) % {max}u;\n\
        volatile int _sink = 0;\n\
        for (uint _t = 0; _t < _tv; _t++) {{ _sink += 1; }}\n\
    }}\n",
            tid = tid,
            max = self.timing_spin_max,
        )
    }

    // -- Vulkan generators --

    fn gen_bank_conflict_vulkan(&self, tid: &str) -> String {
        format!(
            "    {{\n\
        for (int _bc = 0; _bc < {count}; _bc++) {{\n\
            shared_stress[{tid} * {stride} + _bc] = _bc;\n\
        }}\n\
        barrier();\n\
    }}\n",
            stride = self.bank_conflict_stride,
            count = self.bank_conflict_count,
            tid = tid,
        )
    }

    fn gen_scheduling_noise_vulkan(&self, tid: &str) -> String {
        format!(
            "    {{\n\
        uint _sn = ({tid} * 1103515245u + 12345u) % {max}u;\n\
        for (uint _s = 0; _s < _sn; _s++) {{ memoryBarrier(); }}\n\
    }}\n",
            tid = tid,
            max = self.scheduling_noise_max,
        )
    }

    // -- Metal generators --

    fn gen_bank_conflict_metal(&self, tid: &str) -> String {
        format!(
            "    threadgroup int stress_smem[{stride} * {count}];\n\
    for (int _bc = 0; _bc < {count}; _bc++) {{\n\
        stress_smem[{tid} * {stride} + _bc] = _bc;\n\
    }}\n\
    threadgroup_barrier(mem_flags::mem_threadgroup);\n",
            stride = self.bank_conflict_stride,
            count = self.bank_conflict_count,
            tid = tid,
        )
    }

    fn gen_scheduling_noise_metal(&self, tid: &str) -> String {
        format!(
            "    {{\n\
        uint _sn = ({tid} * 1103515245u + 12345u) % {max}u;\n\
        for (uint _s = 0; _s < _sn; _s++) {{ threadgroup_barrier(mem_flags::mem_device); }}\n\
    }}\n",
            tid = tid,
            max = self.scheduling_noise_max,
        )
    }

    /// Compute padding offsets for test variables.
    pub fn compute_padded_offsets(&self, variable_count: usize) -> Vec<u64> {
        let pad = self.padding_bytes as u64;
        (0..variable_count)
            .map(|i| i as u64 * (4 + pad))
            .collect()
    }

    /// Compute strided offsets for test variables.
    pub fn compute_strided_offsets(&self, variable_count: usize) -> Vec<u64> {
        let stride = self.stride_bytes as u64;
        (0..variable_count)
            .map(|i| i as u64 * stride)
            .collect()
    }

    /// Total scratch memory required (bytes).
    pub fn scratch_memory_required(&self) -> u64 {
        let mut total = 0u64;
        if self.has_pattern(StressPattern::CachePressure) {
            total += self.cache_pressure_size;
        }
        if self.has_pattern(StressPattern::MemoryPressure) {
            total += self.memory_pressure_allocs as u64 * self.memory_pressure_alloc_size;
        }
        if self.has_pattern(StressPattern::BankConflict) {
            total += (self.bank_conflict_stride * self.bank_conflict_count * 4) as u64;
        }
        total
    }
}

// ---------------------------------------------------------------------------
// Stress test runner
// ---------------------------------------------------------------------------

/// Wraps `LitmusRunner` with stress-testing support.
#[derive(Debug, Clone)]
pub struct StressTestRunner {
    /// Inner runner.
    runner: LitmusRunner,
    /// Stress configuration.
    stress_config: StressConfig,
}

impl StressTestRunner {
    /// Create a new stress test runner.
    pub fn new(
        hw_config: HardwareConfig,
        stress_config: StressConfig,
    ) -> Result<Self, RunnerError> {
        let runner = LitmusRunner::new(hw_config)?;
        Ok(Self {
            runner,
            stress_config,
        })
    }

    /// Create from a stress mode.
    pub fn from_mode(
        hw_config: HardwareConfig,
        mode: StressMode,
    ) -> Result<Self, RunnerError> {
        let stress_config = StressConfig::from_mode(mode);
        Self::new(hw_config, stress_config)
    }

    pub fn runner(&self) -> &LitmusRunner {
        &self.runner
    }

    pub fn stress_config(&self) -> &StressConfig {
        &self.stress_config
    }

    /// Generate a stressed kernel source.
    pub fn generate_stressed_kernel(
        &self,
        test: &LitmusTest,
    ) -> Result<String, RunnerError> {
        let base_source = self.runner.generate_kernel(test)?;
        let preamble = self.stress_preamble_for_backend();
        Ok(inject_stress_preamble(&base_source, &preamble))
    }

    /// Run a litmus test with stress applied.
    pub fn run_stressed_test(
        &self,
        test: &LitmusTest,
    ) -> Result<TestOutcome, RunnerError> {
        // For now, delegate to the inner runner which already generates the
        // kernel. A full implementation would inject stress code into the
        // kernel before compilation.
        self.runner.run_test(test)
    }

    /// Run and validate with stress.
    pub fn run_stressed_validation(
        &self,
        test: &LitmusTest,
    ) -> Result<HardwareResult, RunnerError> {
        let observed = self.run_stressed_test(test)?;
        Ok(HardwareResult::from_test(test, observed))
    }

    /// Run with multiple stress configurations and merge results.
    pub fn run_sweep(
        &self,
        test: &LitmusTest,
        configs: &[StressConfig],
    ) -> Result<TestOutcome, RunnerError> {
        let backend = self.runner.config().gpu_backend;
        let mut merged = TestOutcome::new(&test.name, backend);

        // Run with default stress first.
        let base = self.run_stressed_test(test)?;
        merged.merge(&base);

        // Run with each additional config.
        for cfg in configs {
            let sr = StressTestRunner::new(
                self.runner.config().clone(),
                cfg.clone(),
            )?;
            let result = sr.run_stressed_test(test)?;
            merged.merge(&result);
        }

        Ok(merged)
    }

    /// Get the stress preamble for the current backend.
    fn stress_preamble_for_backend(&self) -> String {
        let tid_var = "tid";
        match self.runner.config().gpu_backend {
            GpuBackend::Cuda => self.stress_config.generate_cuda_preamble(tid_var),
            GpuBackend::OpenCL => self.stress_config.generate_opencl_preamble(tid_var),
            GpuBackend::Vulkan => self.stress_config.generate_vulkan_preamble(tid_var),
            GpuBackend::Metal => self.stress_config.generate_metal_preamble(tid_var),
        }
    }
}

/// Inject a stress preamble into kernel source code.
///
/// Looks for the marker `// STRESS_INSERTION_POINT` and inserts the preamble
/// there. If no marker is found, inserts after the first `{` in the kernel.
fn inject_stress_preamble(source: &str, preamble: &str) -> String {
    let marker = "// STRESS_INSERTION_POINT";
    if source.contains(marker) {
        return source.replace(marker, preamble);
    }
    // Fallback: insert after first opening brace of the kernel function.
    if let Some(pos) = source.find('{') {
        let mut result = String::with_capacity(source.len() + preamble.len());
        result.push_str(&source[..=pos]);
        result.push('\n');
        result.push_str(preamble);
        result.push_str(&source[pos + 1..]);
        return result;
    }
    // Last resort: prepend.
    format!("{}\n{}", preamble, source)
}

// ---------------------------------------------------------------------------
// Preset stress configurations
// ---------------------------------------------------------------------------

/// Common stress presets.
pub struct StressPresets;

impl StressPresets {
    /// No stress at all — baseline run.
    pub fn baseline() -> StressConfig {
        StressConfig::default()
    }

    /// Bank-conflict focused stress.
    pub fn bank_conflict_heavy() -> StressConfig {
        StressConfig {
            patterns: vec![StressPattern::BankConflict],
            bank_conflict_count: 128,
            bank_conflict_stride: 32,
            ..Default::default()
        }
    }

    /// Cache-pressure focused stress.
    pub fn cache_pressure_heavy() -> StressConfig {
        StressConfig {
            patterns: vec![StressPattern::CachePressure],
            cache_pressure_size: 8 << 20,
            cache_pressure_iterations: 16,
            ..Default::default()
        }
    }

    /// Padding-and-striding stress for spatial reordering.
    pub fn padding_striding() -> StressConfig {
        StressConfig {
            patterns: vec![StressPattern::Padding, StressPattern::Striding],
            padding_bytes: 1024,
            stride_bytes: 512,
            ..Default::default()
        }
    }

    /// Scheduling noise for temporal reordering.
    pub fn scheduling_noise() -> StressConfig {
        StressConfig {
            patterns: vec![
                StressPattern::SchedulingNoise,
                StressPattern::TimingVariation,
            ],
            scheduling_noise_max: 10_000,
            timing_spin_max: 5000,
            ..Default::default()
        }
    }

    /// All patterns enabled at maximum intensity.
    pub fn maximum() -> StressConfig {
        StressConfig {
            patterns: StressPattern::all().to_vec(),
            bank_conflict_count: 128,
            bank_conflict_stride: 32,
            cache_pressure_size: 8 << 20,
            cache_pressure_iterations: 16,
            padding_bytes: 1024,
            stride_bytes: 512,
            scheduling_noise_max: 10_000,
            scheduling_noise_random: true,
            memory_pressure_allocs: 16,
            memory_pressure_alloc_size: 4 << 20,
            timing_spin_max: 5000,
            timing_random: true,
            warmup_iterations: 100,
            pin_threads: true,
            stress_rounds: 5,
        }
    }

    /// Standard sweep configs for systematic exploration.
    pub fn sweep_configs() -> Vec<StressConfig> {
        vec![
            Self::baseline(),
            Self::bank_conflict_heavy(),
            Self::cache_pressure_heavy(),
            Self::padding_striding(),
            Self::scheduling_noise(),
            Self::maximum(),
        ]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::checker::litmus::{
        Instruction, LitmusOutcome, Ordering, Outcome, Thread,
    };

    fn make_test() -> LitmusTest {
        LitmusTest {
            name: "stress_test".into(),
            threads: vec![
                Thread {
                    id: 0,
                    instructions: vec![Instruction::Store {
                        addr: 0,
                        value: 1,
                        ordering: Ordering::Relaxed,
                    }],
                },
                Thread {
                    id: 1,
                    instructions: vec![Instruction::Load {
                        reg: 0,
                        addr: 0,
                        ordering: Ordering::Relaxed,
                    }],
                },
            ],
            initial_state: {
                let mut m = HashMap::new();
                m.insert(0u64, 0u64);
                m
            },
            expected_outcomes: vec![
                (
                    Outcome {
                        registers: {
                            let mut m = HashMap::new();
                            m.insert((1, 0), 0);
                            m
                        },
                        memory: HashMap::new(),
                    },
                    LitmusOutcome::Allowed,
                ),
            ],
        }
    }

    #[test]
    fn test_stress_pattern_display() {
        assert_eq!(StressPattern::BankConflict.to_string(), "bank_conflict");
        assert_eq!(StressPattern::CachePressure.to_string(), "cache_pressure");
    }

    #[test]
    fn test_stress_pattern_for_level() {
        assert!(StressPattern::for_level(StressMode::None).is_empty());
        assert!(!StressPattern::for_level(StressMode::Heavy).is_empty());
        let medium = StressPattern::for_level(StressMode::Medium);
        assert!(medium.contains(&StressPattern::BankConflict));
    }

    #[test]
    fn test_stress_config_default() {
        let cfg = StressConfig::default();
        assert!(cfg.patterns.is_empty());
        assert_eq!(cfg.bank_conflict_stride, 32);
        assert_eq!(cfg.padding_bytes, 256);
    }

    #[test]
    fn test_stress_config_from_mode() {
        let cfg = StressConfig::from_mode(StressMode::Heavy);
        assert!(!cfg.patterns.is_empty());
        assert_eq!(cfg.cache_pressure_size, 4 << 20);
        assert_eq!(cfg.stress_rounds, 3);
    }

    #[test]
    fn test_stress_config_has_pattern() {
        let cfg = StressConfig::with_patterns(vec![
            StressPattern::Padding,
            StressPattern::Striding,
        ]);
        assert!(cfg.has_pattern(StressPattern::Padding));
        assert!(!cfg.has_pattern(StressPattern::BankConflict));
    }

    #[test]
    fn test_cuda_preamble_generation() {
        let cfg = StressConfig::from_mode(StressMode::Heavy);
        let preamble = cfg.generate_cuda_preamble("threadIdx.x");
        assert!(preamble.contains("STRESS PREAMBLE"));
        assert!(preamble.contains("stress_smem"));
        assert!(preamble.contains("__syncthreads"));
    }

    #[test]
    fn test_opencl_preamble_generation() {
        let cfg = StressConfig::with_patterns(vec![StressPattern::BankConflict]);
        let preamble = cfg.generate_opencl_preamble("get_local_id(0)");
        assert!(preamble.contains("__local"));
        assert!(preamble.contains("barrier"));
    }

    #[test]
    fn test_vulkan_preamble_generation() {
        let cfg = StressConfig::with_patterns(vec![StressPattern::SchedulingNoise]);
        let preamble = cfg.generate_vulkan_preamble("gl_LocalInvocationID.x");
        assert!(preamble.contains("memoryBarrier"));
    }

    #[test]
    fn test_metal_preamble_generation() {
        let cfg = StressConfig::with_patterns(vec![StressPattern::BankConflict]);
        let preamble = cfg.generate_metal_preamble("tid");
        assert!(preamble.contains("threadgroup"));
    }

    #[test]
    fn test_padded_offsets() {
        let cfg = StressConfig {
            padding_bytes: 256,
            ..Default::default()
        };
        let offsets = cfg.compute_padded_offsets(3);
        assert_eq!(offsets, vec![0, 260, 520]);
    }

    #[test]
    fn test_strided_offsets() {
        let cfg = StressConfig {
            stride_bytes: 128,
            ..Default::default()
        };
        let offsets = cfg.compute_strided_offsets(4);
        assert_eq!(offsets, vec![0, 128, 256, 384]);
    }

    #[test]
    fn test_scratch_memory_required() {
        let cfg = StressConfig {
            patterns: vec![StressPattern::CachePressure, StressPattern::BankConflict],
            cache_pressure_size: 1 << 20,
            bank_conflict_stride: 32,
            bank_conflict_count: 32,
            ..Default::default()
        };
        let mem = cfg.scratch_memory_required();
        assert!(mem >= 1 << 20);
        assert!(mem > 0);
    }

    #[test]
    fn test_inject_stress_preamble_with_marker() {
        let source = "void kernel() {\n// STRESS_INSERTION_POINT\n    int x = 0;\n}";
        let preamble = "    // stress code\n";
        let result = inject_stress_preamble(source, preamble);
        assert!(result.contains("// stress code"));
        assert!(!result.contains("STRESS_INSERTION_POINT"));
    }

    #[test]
    fn test_inject_stress_preamble_without_marker() {
        let source = "void kernel() {\n    int x = 0;\n}";
        let preamble = "    // stress code\n";
        let result = inject_stress_preamble(source, preamble);
        assert!(result.contains("// stress code"));
    }

    #[test]
    fn test_stress_presets_baseline() {
        let cfg = StressPresets::baseline();
        assert!(cfg.patterns.is_empty());
    }

    #[test]
    fn test_stress_presets_maximum() {
        let cfg = StressPresets::maximum();
        assert!(cfg.has_pattern(StressPattern::BankConflict));
        assert!(cfg.has_pattern(StressPattern::CachePressure));
        assert!(cfg.has_pattern(StressPattern::MemoryPressure));
        assert_eq!(cfg.stress_rounds, 5);
    }

    #[test]
    fn test_stress_presets_sweep() {
        let configs = StressPresets::sweep_configs();
        assert!(configs.len() >= 4);
    }

    #[test]
    fn test_stress_test_runner_creation() {
        let hw = HardwareConfig::default();
        let sc = StressConfig::from_mode(StressMode::Medium);
        let runner = StressTestRunner::new(hw, sc);
        assert!(runner.is_ok());
    }

    #[test]
    fn test_stress_test_runner_from_mode() {
        let hw = HardwareConfig::default();
        let runner = StressTestRunner::from_mode(hw, StressMode::Light);
        assert!(runner.is_ok());
        let r = runner.unwrap();
        assert!(r.stress_config().has_pattern(StressPattern::Padding));
    }

    #[test]
    fn test_stress_test_runner_generate_kernel() {
        let hw = HardwareConfig::new(GpuBackend::Cuda);
        let runner = StressTestRunner::from_mode(hw, StressMode::Heavy).unwrap();
        let test = make_test();
        let kernel = runner.generate_stressed_kernel(&test);
        assert!(kernel.is_ok());
    }

    #[test]
    fn test_stress_pattern_all() {
        let all = StressPattern::all();
        assert_eq!(all.len(), 10);
    }
}
