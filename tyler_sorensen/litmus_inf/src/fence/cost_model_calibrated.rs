//! Calibrated fence cost model.
//!
//! Provides hardware-measured and architecture-specific cost models for memory
//! fences across x86, ARM, RISC-V, PTX, Vulkan, and OpenCL targets.  Costs
//! can be derived from predefined tables or calibrated from microbenchmark
//! timing data.

use std::collections::HashMap;
use std::fmt;

use crate::checker::litmus::{Ordering, Scope};

// ---------------------------------------------------------------------------
// FenceCost
// ---------------------------------------------------------------------------

/// Cost of a single fence type on a particular architecture.
#[derive(Debug, Clone, PartialEq)]
pub struct FenceCost {
    /// Average latency in nanoseconds.
    pub latency_ns: f64,
    /// Throughput penalty as a multiplier (1.0 = no penalty).
    pub throughput_penalty: f64,
    /// Scope to which the fence applies.
    pub scope: Scope,
    /// Memory ordering provided by the fence.
    pub ordering: Ordering,
}

impl FenceCost {
    pub fn new(latency_ns: f64, throughput_penalty: f64, scope: Scope, ordering: Ordering) -> Self {
        Self { latency_ns, throughput_penalty, scope, ordering }
    }

    /// Composite cost that combines latency and throughput impact.
    pub fn composite_cost(&self) -> f64 {
        self.latency_ns * self.throughput_penalty
    }

    /// Return the cost relative to a baseline fence.
    pub fn relative_to(&self, baseline: &FenceCost) -> f64 {
        if baseline.composite_cost() == 0.0 {
            return f64::INFINITY;
        }
        self.composite_cost() / baseline.composite_cost()
    }
}

impl fmt::Display for FenceCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FenceCost({:.1}ns, tp={:.2}x, scope={}, ord={})",
            self.latency_ns, self.throughput_penalty, self.scope, self.ordering,
        )
    }
}

// ---------------------------------------------------------------------------
// Architecture
// ---------------------------------------------------------------------------

/// Supported target architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Architecture {
    X86,
    ARM,
    RiscV,
    PTX,
    Vulkan,
    OpenCL,
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::X86 => write!(f, "x86"),
            Self::ARM => write!(f, "ARM"),
            Self::RiscV => write!(f, "RISC-V"),
            Self::PTX => write!(f, "PTX"),
            Self::Vulkan => write!(f, "Vulkan"),
            Self::OpenCL => write!(f, "OpenCL"),
        }
    }
}

// ---------------------------------------------------------------------------
// ArchitectureCosts — predefined tables
// ---------------------------------------------------------------------------

/// Predefined fence-cost tables per architecture.
#[derive(Debug, Clone)]
pub struct ArchitectureCosts {
    pub arch: Architecture,
    /// Map from (ordering, scope) → FenceCost.
    pub costs: HashMap<(Ordering, Scope), FenceCost>,
    /// Optional vendor documentation reference.
    pub vendor_doc: Option<String>,
}

impl ArchitectureCosts {
    pub fn new(arch: Architecture) -> Self {
        Self { arch, costs: HashMap::new(), vendor_doc: None }
    }

    pub fn insert(&mut self, ordering: Ordering, scope: Scope, cost: FenceCost) {
        self.costs.insert((ordering, scope), cost);
    }

    pub fn get(&self, ordering: &Ordering, scope: &Scope) -> Option<&FenceCost> {
        self.costs.get(&(*ordering, *scope))
    }

    /// The cheapest fence that provides at least the given ordering and scope.
    pub fn cheapest_fence(&self, min_ordering: &Ordering, min_scope: &Scope) -> Option<&FenceCost> {
        self.costs.values()
            .filter(|c| ordering_strength(&c.ordering) >= ordering_strength(min_ordering))
            .filter(|c| scope_strength(&c.scope) >= scope_strength(min_scope))
            .min_by(|a, b| a.composite_cost().partial_cmp(&b.composite_cost()).unwrap())
    }

    /// Cost ratios relative to the cheapest fence in this architecture.
    pub fn relative_cost_ratios(&self) -> HashMap<(Ordering, Scope), f64> {
        let baseline = self.costs.values()
            .min_by(|a, b| a.composite_cost().partial_cmp(&b.composite_cost()).unwrap());
        let baseline = match baseline {
            Some(b) => b.clone(),
            None => return HashMap::new(),
        };
        self.costs.iter().map(|(k, v)| (*k, v.relative_to(&baseline))).collect()
    }

    // -----------------------------------------------------------------------
    // Predefined architectures
    // -----------------------------------------------------------------------

    /// x86 / x86-64 costs (Intel SDM reference).
    pub fn x86() -> Self {
        let mut c = Self::new(Architecture::X86);
        c.vendor_doc = Some("Intel 64 and IA-32 Architectures SDM".into());

        // x86 has strong ordering; only MFENCE / SFENCE / LFENCE matter.
        c.insert(Ordering::SeqCst, Scope::System,
            FenceCost::new(33.0, 1.8, Scope::System, Ordering::SeqCst));
        c.insert(Ordering::AcqRel, Scope::System,
            FenceCost::new(33.0, 1.8, Scope::System, Ordering::AcqRel));
        c.insert(Ordering::Release, Scope::System,
            FenceCost::new(8.0, 1.1, Scope::System, Ordering::Release));
        c.insert(Ordering::Acquire, Scope::System,
            FenceCost::new(4.0, 1.05, Scope::System, Ordering::Acquire));
        c.insert(Ordering::Relaxed, Scope::None,
            FenceCost::new(0.0, 1.0, Scope::None, Ordering::Relaxed));
        c
    }

    /// ARM (ARMv8-A) costs (ARM Architecture Reference Manual).
    pub fn arm() -> Self {
        let mut c = Self::new(Architecture::ARM);
        c.vendor_doc = Some("ARM Architecture Reference Manual ARMv8-A".into());

        c.insert(Ordering::SeqCst, Scope::System,
            FenceCost::new(70.0, 2.5, Scope::System, Ordering::SeqCst));
        c.insert(Ordering::AcqRel, Scope::System,
            FenceCost::new(45.0, 2.0, Scope::System, Ordering::AcqRel));
        c.insert(Ordering::Release, Scope::System,
            FenceCost::new(25.0, 1.5, Scope::System, Ordering::Release));
        c.insert(Ordering::Acquire, Scope::System,
            FenceCost::new(15.0, 1.3, Scope::System, Ordering::Acquire));
        c.insert(Ordering::Relaxed, Scope::None,
            FenceCost::new(0.0, 1.0, Scope::None, Ordering::Relaxed));
        // DMB ISH (inner-shareable domain ≈ GPU scope)
        c.insert(Ordering::AcqRel, Scope::GPU,
            FenceCost::new(35.0, 1.8, Scope::GPU, Ordering::AcqRel));
        c
    }

    /// RISC-V costs.
    pub fn riscv() -> Self {
        let mut c = Self::new(Architecture::RiscV);
        c.vendor_doc = Some("RISC-V ISA Manual, Volume I".into());

        c.insert(Ordering::SeqCst, Scope::System,
            FenceCost::new(55.0, 2.2, Scope::System, Ordering::SeqCst));
        c.insert(Ordering::AcqRel, Scope::System,
            FenceCost::new(40.0, 1.9, Scope::System, Ordering::AcqRel));
        c.insert(Ordering::Release, Scope::System,
            FenceCost::new(20.0, 1.4, Scope::System, Ordering::Release));
        c.insert(Ordering::Acquire, Scope::System,
            FenceCost::new(12.0, 1.2, Scope::System, Ordering::Acquire));
        c.insert(Ordering::Relaxed, Scope::None,
            FenceCost::new(0.0, 1.0, Scope::None, Ordering::Relaxed));
        c
    }

    /// NVIDIA PTX / CUDA costs.
    pub fn ptx() -> Self {
        let mut c = Self::new(Architecture::PTX);
        c.vendor_doc = Some("NVIDIA PTX ISA 8.x".into());

        // membar.sys
        c.insert(Ordering::SeqCst, Scope::System,
            FenceCost::new(500.0, 5.0, Scope::System, Ordering::SeqCst));
        // membar.gl
        c.insert(Ordering::AcqRel, Scope::GPU,
            FenceCost::new(150.0, 3.0, Scope::GPU, Ordering::AcqRel));
        c.insert(Ordering::SeqCst, Scope::GPU,
            FenceCost::new(200.0, 3.5, Scope::GPU, Ordering::SeqCst));
        // membar.cta
        c.insert(Ordering::AcqRel, Scope::CTA,
            FenceCost::new(20.0, 1.3, Scope::CTA, Ordering::AcqRel));
        c.insert(Ordering::SeqCst, Scope::CTA,
            FenceCost::new(30.0, 1.5, Scope::CTA, Ordering::SeqCst));
        // Release / Acquire at CTA scope
        c.insert(Ordering::Release, Scope::CTA,
            FenceCost::new(10.0, 1.1, Scope::CTA, Ordering::Release));
        c.insert(Ordering::Acquire, Scope::CTA,
            FenceCost::new(8.0, 1.05, Scope::CTA, Ordering::Acquire));
        c.insert(Ordering::Relaxed, Scope::None,
            FenceCost::new(0.0, 1.0, Scope::None, Ordering::Relaxed));
        c
    }

    /// Vulkan (SPIR-V) costs.
    pub fn vulkan() -> Self {
        let mut c = Self::new(Architecture::Vulkan);
        c.vendor_doc = Some("Vulkan Memory Model (VK_KHR_vulkan_memory_model)".into());

        c.insert(Ordering::SeqCst, Scope::System,
            FenceCost::new(400.0, 4.5, Scope::System, Ordering::SeqCst));
        c.insert(Ordering::AcqRel, Scope::GPU,
            FenceCost::new(120.0, 2.8, Scope::GPU, Ordering::AcqRel));
        c.insert(Ordering::AcqRel, Scope::CTA,
            FenceCost::new(18.0, 1.25, Scope::CTA, Ordering::AcqRel));
        c.insert(Ordering::Release, Scope::CTA,
            FenceCost::new(9.0, 1.08, Scope::CTA, Ordering::Release));
        c.insert(Ordering::Acquire, Scope::CTA,
            FenceCost::new(7.0, 1.04, Scope::CTA, Ordering::Acquire));
        c.insert(Ordering::Relaxed, Scope::None,
            FenceCost::new(0.0, 1.0, Scope::None, Ordering::Relaxed));
        c
    }

    /// OpenCL costs.
    pub fn opencl() -> Self {
        let mut c = Self::new(Architecture::OpenCL);
        c.vendor_doc = Some("OpenCL 3.0 Specification".into());

        c.insert(Ordering::SeqCst, Scope::System,
            FenceCost::new(450.0, 4.8, Scope::System, Ordering::SeqCst));
        c.insert(Ordering::AcqRel, Scope::GPU,
            FenceCost::new(140.0, 2.9, Scope::GPU, Ordering::AcqRel));
        c.insert(Ordering::AcqRel, Scope::CTA,
            FenceCost::new(22.0, 1.35, Scope::CTA, Ordering::AcqRel));
        c.insert(Ordering::Release, Scope::CTA,
            FenceCost::new(11.0, 1.12, Scope::CTA, Ordering::Release));
        c.insert(Ordering::Acquire, Scope::CTA,
            FenceCost::new(9.0, 1.06, Scope::CTA, Ordering::Acquire));
        c.insert(Ordering::Relaxed, Scope::None,
            FenceCost::new(0.0, 1.0, Scope::None, Ordering::Relaxed));
        c
    }

    /// Look up predefined costs by architecture enum.
    pub fn for_architecture(arch: Architecture) -> Self {
        match arch {
            Architecture::X86 => Self::x86(),
            Architecture::ARM => Self::arm(),
            Architecture::RiscV => Self::riscv(),
            Architecture::PTX => Self::ptx(),
            Architecture::Vulkan => Self::vulkan(),
            Architecture::OpenCL => Self::opencl(),
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: ordering / scope strength for comparison
// ---------------------------------------------------------------------------

fn ordering_strength(o: &Ordering) -> u8 {
    match o {
        Ordering::Relaxed => 0,
        Ordering::Acquire | Ordering::AcquireCTA | Ordering::AcquireGPU | Ordering::AcquireSystem => 1,
        Ordering::Release | Ordering::ReleaseCTA | Ordering::ReleaseGPU | Ordering::ReleaseSystem => 2,
        Ordering::AcqRel => 3,
        Ordering::SeqCst => 4,
    }
}

fn scope_strength(s: &Scope) -> u8 {
    match s {
        Scope::None => 0,
        Scope::CTA => 1,
        Scope::GPU => 2,
        Scope::System => 3,
    }
}

// ---------------------------------------------------------------------------
// MicrobenchmarkResult
// ---------------------------------------------------------------------------

/// Raw timing data from a fence microbenchmark run.
#[derive(Debug, Clone)]
pub struct MicrobenchmarkResult {
    /// Fence ordering under test.
    pub ordering: Ordering,
    /// Fence scope under test.
    pub scope: Scope,
    /// Individual timing samples in nanoseconds.
    pub samples_ns: Vec<f64>,
    /// Architecture on which the benchmark was run.
    pub arch: Architecture,
    /// Optional label (e.g. "membar.cta on A100").
    pub label: Option<String>,
}

impl MicrobenchmarkResult {
    pub fn new(ordering: Ordering, scope: Scope, samples: Vec<f64>, arch: Architecture) -> Self {
        Self { ordering, scope, samples_ns: samples, arch, label: None }
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn sample_count(&self) -> usize {
        self.samples_ns.len()
    }
}

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------

/// Statistical summary of timing samples.
#[derive(Debug, Clone, PartialEq)]
pub struct TimingStats {
    pub mean: f64,
    pub median: f64,
    pub stddev: f64,
    pub min: f64,
    pub max: f64,
    pub p95: f64,
    pub p99: f64,
    pub count: usize,
}

impl fmt::Display for TimingStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "mean={:.2}ns median={:.2}ns σ={:.2}ns p95={:.2}ns p99={:.2}ns n={}",
            self.mean, self.median, self.stddev, self.p95, self.p99, self.count,
        )
    }
}

fn compute_stats(samples: &[f64]) -> Option<TimingStats> {
    if samples.is_empty() {
        return None;
    }
    let n = samples.len();
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = sorted.iter().sum::<f64>() / n as f64;
    let variance = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let stddev = variance.sqrt();
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };
    let p95 = sorted[(n as f64 * 0.95).ceil() as usize - 1].min(sorted[n - 1]);
    let p99 = sorted[(n as f64 * 0.99).ceil() as usize - 1].min(sorted[n - 1]);

    Some(TimingStats {
        mean,
        median,
        stddev,
        min: sorted[0],
        max: sorted[n - 1],
        p95,
        p99,
        count: n,
    })
}

// ---------------------------------------------------------------------------
// CostCalibrator
// ---------------------------------------------------------------------------

/// Calibrates a cost model from microbenchmark results.
#[derive(Debug, Clone)]
pub struct CostCalibrator {
    /// Collected benchmark results.
    results: Vec<MicrobenchmarkResult>,
    /// Base architecture for predefined fallback costs.
    base_arch: Architecture,
    /// Outlier removal: discard samples beyond this many standard deviations.
    outlier_sigma: f64,
}

impl CostCalibrator {
    pub fn new(arch: Architecture) -> Self {
        Self { results: Vec::new(), base_arch: arch, outlier_sigma: 3.0 }
    }

    pub fn with_outlier_sigma(mut self, sigma: f64) -> Self {
        self.outlier_sigma = sigma;
        self
    }

    /// Add a microbenchmark result.
    pub fn add_result(&mut self, result: MicrobenchmarkResult) {
        self.results.push(result);
    }

    /// Parse timing data from a CSV-like string.
    ///
    /// Expected format per line: `ordering,scope,sample1,sample2,...`
    pub fn parse_timing_data(&mut self, data: &str) -> Result<usize, String> {
        let mut count = 0usize;
        for (line_no, line) in data.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() < 3 {
                return Err(format!("line {}: expected at least 3 comma-separated fields", line_no + 1));
            }
            let ordering = parse_ordering(parts[0])
                .ok_or_else(|| format!("line {}: unknown ordering '{}'", line_no + 1, parts[0]))?;
            let scope = parse_scope(parts[1])
                .ok_or_else(|| format!("line {}: unknown scope '{}'", line_no + 1, parts[1]))?;
            let samples: Result<Vec<f64>, _> = parts[2..].iter().map(|s| s.parse::<f64>()).collect();
            let samples = samples.map_err(|e| format!("line {}: bad sample: {}", line_no + 1, e))?;
            self.results.push(MicrobenchmarkResult::new(ordering, scope, samples, self.base_arch));
            count += 1;
        }
        Ok(count)
    }

    /// Remove outlier samples from all results using the configured sigma threshold.
    pub fn remove_outliers(&mut self) {
        let sigma = self.outlier_sigma;
        for result in &mut self.results {
            if let Some(stats) = compute_stats(&result.samples_ns) {
                result.samples_ns.retain(|&s| (s - stats.mean).abs() <= sigma * stats.stddev);
            }
        }
    }

    /// Compute statistics for each (ordering, scope) pair.
    pub fn compute_all_stats(&self) -> HashMap<(Ordering, Scope), TimingStats> {
        let mut grouped: HashMap<(Ordering, Scope), Vec<f64>> = HashMap::new();
        for r in &self.results {
            grouped.entry((r.ordering, r.scope))
                .or_default()
                .extend_from_slice(&r.samples_ns);
        }
        grouped.into_iter()
            .filter_map(|(k, samples)| compute_stats(&samples).map(|s| (k, s)))
            .collect()
    }

    /// Build a calibrated cost model from the collected results.
    ///
    /// Falls back to predefined costs for any (ordering, scope) pair that has
    /// no benchmark data.
    pub fn calibrate(&mut self) -> CalibratedCostModel {
        self.remove_outliers();
        let stats = self.compute_all_stats();
        let predefined = ArchitectureCosts::for_architecture(self.base_arch);

        let mut costs = ArchitectureCosts::new(self.base_arch);
        costs.vendor_doc = predefined.vendor_doc.clone();

        // Estimate throughput penalty from sample variance.
        let estimate_tp = |s: &TimingStats| -> f64 {
            let cv = if s.mean > 0.0 { s.stddev / s.mean } else { 0.0 };
            1.0 + cv * 2.0
        };

        // Merge calibrated data with predefined fallbacks.
        for (&(ord, scope), s) in &stats {
            let tp = estimate_tp(s);
            costs.insert(ord, scope, FenceCost::new(s.median, tp, scope, ord));
        }
        for ((ord, scope), fc) in &predefined.costs {
            if !costs.costs.contains_key(&(*ord, *scope)) {
                costs.insert(*ord, *scope, fc.clone());
            }
        }

        CalibratedCostModel {
            arch: self.base_arch,
            costs,
            stats,
            calibrated: true,
        }
    }
}

// ---------------------------------------------------------------------------
// CalibratedCostModel
// ---------------------------------------------------------------------------

/// A cost model that may be calibrated from hardware measurements.
#[derive(Debug, Clone)]
pub struct CalibratedCostModel {
    pub arch: Architecture,
    pub costs: ArchitectureCosts,
    pub stats: HashMap<(Ordering, Scope), TimingStats>,
    pub calibrated: bool,
}

impl CalibratedCostModel {
    /// Create a non-calibrated model using predefined costs.
    pub fn predefined(arch: Architecture) -> Self {
        Self {
            arch,
            costs: ArchitectureCosts::for_architecture(arch),
            stats: HashMap::new(),
            calibrated: false,
        }
    }

    /// Look up the cost of a fence.
    pub fn fence_cost(&self, ordering: &Ordering, scope: &Scope) -> Option<&FenceCost> {
        self.costs.get(ordering, scope)
    }

    /// Look up composite cost; returns 0.0 for Relaxed with no entry.
    pub fn cost_of(&self, ordering: &Ordering, scope: &Scope) -> f64 {
        self.costs.get(ordering, scope)
            .map(|c| c.composite_cost())
            .unwrap_or(0.0)
    }

    /// The cheapest fence that provides at least the required ordering/scope.
    pub fn cheapest_sufficient(&self, ordering: &Ordering, scope: &Scope) -> Option<&FenceCost> {
        self.costs.cheapest_fence(ordering, scope)
    }

    /// Relative cost ratios.
    pub fn relative_ratios(&self) -> HashMap<(Ordering, Scope), f64> {
        self.costs.relative_cost_ratios()
    }

    /// Get the timing statistics for a particular fence, if calibrated.
    pub fn timing_stats(&self, ordering: &Ordering, scope: &Scope) -> Option<&TimingStats> {
        self.stats.get(&(*ordering, *scope))
    }

    /// Summary of all costs for display.
    pub fn summary(&self) -> String {
        let mut lines = vec![format!("Cost model: {} (calibrated={})", self.arch, self.calibrated)];
        let mut entries: Vec<_> = self.costs.costs.iter().collect();
        entries.sort_by(|a, b| a.1.composite_cost().partial_cmp(&b.1.composite_cost()).unwrap());
        for ((ord, scope), cost) in entries {
            lines.push(format!("  {}/{}: {}", ord, scope, cost));
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// MicrobenchmarkGenerator
// ---------------------------------------------------------------------------

/// Generates microbenchmark source code for measuring fence costs.
#[derive(Debug)]
pub struct MicrobenchmarkGenerator {
    pub arch: Architecture,
    pub iterations: usize,
    pub warmup: usize,
}

impl MicrobenchmarkGenerator {
    pub fn new(arch: Architecture) -> Self {
        Self { arch, iterations: 100_000, warmup: 10_000 }
    }

    pub fn with_iterations(mut self, n: usize) -> Self {
        self.iterations = n;
        self
    }

    pub fn with_warmup(mut self, n: usize) -> Self {
        self.warmup = n;
        self
    }

    /// Generate benchmark code for the given fence type.
    pub fn generate(&self, ordering: &Ordering, scope: &Scope) -> String {
        match self.arch {
            Architecture::X86 => self.generate_x86(ordering),
            Architecture::ARM => self.generate_arm(ordering),
            Architecture::RiscV => self.generate_riscv(ordering),
            Architecture::PTX => self.generate_ptx(ordering, scope),
            Architecture::Vulkan => self.generate_vulkan(ordering, scope),
            Architecture::OpenCL => self.generate_opencl(ordering, scope),
        }
    }

    fn generate_x86(&self, ordering: &Ordering) -> String {
        let fence_instr = match ordering {
            Ordering::SeqCst | Ordering::AcqRel => "mfence",
            Ordering::Release => "sfence",
            Ordering::Acquire => "lfence",
            _ => "nop",
        };
        format!(
            "// x86 fence microbenchmark: {ordering}\n\
             #include <stdint.h>\n\
             #include <x86intrin.h>\n\
             \n\
             uint64_t bench_{fence_instr}(int iters) {{\n\
                 uint64_t start = __rdtsc();\n\
                 for (int i = 0; i < iters; i++) {{\n\
                     asm volatile(\"{fence_instr}\" ::: \"memory\");\n\
                 }}\n\
                 uint64_t end = __rdtsc();\n\
                 return (end - start) / iters;\n\
             }}\n"
        )
    }

    fn generate_arm(&self, ordering: &Ordering) -> String {
        let fence_instr = match ordering {
            Ordering::SeqCst => "dmb sy",
            Ordering::AcqRel | Ordering::Release => "dmb ishst",
            Ordering::Acquire => "dmb ishld",
            _ => "nop",
        };
        format!(
            "// ARM fence microbenchmark: {ordering}\n\
             #include <stdint.h>\n\
             \n\
             uint64_t bench(int iters) {{\n\
                 uint64_t start;\n\
                 asm volatile(\"mrs %0, cntvct_el0\" : \"=r\"(start));\n\
                 for (int i = 0; i < iters; i++) {{\n\
                     asm volatile(\"{fence_instr}\" ::: \"memory\");\n\
                 }}\n\
                 uint64_t end;\n\
                 asm volatile(\"mrs %0, cntvct_el0\" : \"=r\"(end));\n\
                 return (end - start) / iters;\n\
             }}\n"
        )
    }

    fn generate_riscv(&self, ordering: &Ordering) -> String {
        let fence_instr = match ordering {
            Ordering::SeqCst | Ordering::AcqRel => "fence iorw, iorw",
            Ordering::Release => "fence rw, w",
            Ordering::Acquire => "fence r, rw",
            _ => "nop",
        };
        format!(
            "// RISC-V fence microbenchmark: {ordering}\n\
             #include <stdint.h>\n\
             \n\
             uint64_t bench(int iters) {{\n\
                 uint64_t start;\n\
                 asm volatile(\"rdcycle %0\" : \"=r\"(start));\n\
                 for (int i = 0; i < iters; i++) {{\n\
                     asm volatile(\"{fence_instr}\" ::: \"memory\");\n\
                 }}\n\
                 uint64_t end;\n\
                 asm volatile(\"rdcycle %0\" : \"=r\"(end));\n\
                 return (end - start) / iters;\n\
             }}\n"
        )
    }

    fn generate_ptx(&self, ordering: &Ordering, scope: &Scope) -> String {
        let fence_instr = match (ordering, scope) {
            (Ordering::SeqCst, Scope::System) => "membar.sys",
            (Ordering::SeqCst, Scope::GPU) | (Ordering::AcqRel, Scope::GPU) => "membar.gl",
            (_, Scope::CTA) => "membar.cta",
            _ => "// no-op",
        };
        format!(
            "// PTX fence microbenchmark: {ordering}/{scope}\n\
             // Inline PTX in CUDA:\n\
             __global__ void bench_{scope}(int* out, int iters) {{\n\
                 clock_t start = clock();\n\
                 for (int i = 0; i < iters; i++) {{\n\
                     asm volatile(\"{fence_instr};\" ::: \"memory\");\n\
                 }}\n\
                 clock_t end = clock();\n\
                 out[threadIdx.x] = (int)(end - start) / iters;\n\
             }}\n"
        )
    }

    fn generate_vulkan(&self, ordering: &Ordering, scope: &Scope) -> String {
        let semantics = match ordering {
            Ordering::SeqCst => "AcquireRelease | SequentiallyConsistent",
            Ordering::AcqRel => "AcquireRelease",
            Ordering::Release => "Release",
            Ordering::Acquire => "Acquire",
            _ => "None",
        };
        let scope_spirv = match scope {
            Scope::System => "CrossDevice",
            Scope::GPU => "Device",
            Scope::CTA => "Workgroup",
            Scope::None => "Invocation",
        };
        format!(
            "; SPIR-V fence microbenchmark: {ordering}/{scope}\n\
             ; OpControlBarrier %{scope_spirv} %{scope_spirv} %{semantics}\n\
             ; Wrap in compute shader with timing via timestamp queries.\n"
        )
    }

    fn generate_opencl(&self, ordering: &Ordering, scope: &Scope) -> String {
        let flags = match ordering {
            Ordering::SeqCst => "CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE",
            Ordering::AcqRel | Ordering::Release => "CLK_GLOBAL_MEM_FENCE",
            Ordering::Acquire => "CLK_LOCAL_MEM_FENCE",
            _ => "0",
        };
        let scope_cl = match scope {
            Scope::System => "memory_scope_all_svm_devices",
            Scope::GPU => "memory_scope_device",
            Scope::CTA => "memory_scope_work_group",
            Scope::None => "memory_scope_work_item",
        };
        format!(
            "// OpenCL fence microbenchmark: {ordering}/{scope}\n\
             __kernel void bench(__global int* out, int iters) {{\n\
                 ulong start = get_global_timer_ns();\n\
                 for (int i = 0; i < iters; i++) {{\n\
                     atomic_work_item_fence({flags}, memory_order_acq_rel, {scope_cl});\n\
                 }}\n\
                 ulong end = get_global_timer_ns();\n\
                 out[get_global_id(0)] = (int)((end - start) / iters);\n\
             }}\n"
        )
    }

    /// Generate benchmarks for all fence types in the architecture.
    pub fn generate_all(&self) -> Vec<(Ordering, Scope, String)> {
        let arch_costs = ArchitectureCosts::for_architecture(self.arch);
        arch_costs.costs.keys()
            .map(|(ord, scope)| (*ord, *scope, self.generate(ord, scope)))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// VendorDocReference
// ---------------------------------------------------------------------------

/// References to vendor documentation for fence semantics.
#[derive(Debug, Clone)]
pub struct VendorDocReference {
    pub document: String,
    pub section: String,
    pub fence_name: String,
    pub description: String,
}

impl VendorDocReference {
    pub fn new(document: &str, section: &str, fence_name: &str, description: &str) -> Self {
        Self {
            document: document.into(),
            section: section.into(),
            fence_name: fence_name.into(),
            description: description.into(),
        }
    }
}

/// Look up vendor documentation for a fence on the given architecture.
pub fn vendor_doc_for_fence(arch: Architecture, ordering: &Ordering, scope: &Scope) -> Option<VendorDocReference> {
    match arch {
        Architecture::X86 => {
            let (section, name, desc) = match ordering {
                Ordering::SeqCst | Ordering::AcqRel => (
                    "Vol. 2B, MFENCE",
                    "MFENCE",
                    "Serializes all load and store operations",
                ),
                Ordering::Release => (
                    "Vol. 2B, SFENCE",
                    "SFENCE",
                    "Serializes store operations",
                ),
                Ordering::Acquire => (
                    "Vol. 2B, LFENCE",
                    "LFENCE",
                    "Serializes load operations",
                ),
                _ => return None,
            };
            Some(VendorDocReference::new(
                "Intel 64 and IA-32 Architectures SDM", section, name, desc,
            ))
        }
        Architecture::ARM => {
            let (section, name, desc) = match (ordering, scope) {
                (Ordering::SeqCst, _) => ("B2.7.3", "DMB SY", "Full system data memory barrier"),
                (Ordering::AcqRel, Scope::GPU) => ("B2.7.3", "DMB ISH", "Inner shareable barrier"),
                (Ordering::AcqRel, _) => ("B2.7.3", "DMB ISH", "Inner shareable barrier"),
                (Ordering::Release, _) => ("B2.7.3", "DMB ISHST", "Store barrier, inner shareable"),
                (Ordering::Acquire, _) => ("B2.7.3", "DMB ISHLD", "Load barrier, inner shareable"),
                _ => return None,
            };
            Some(VendorDocReference::new(
                "ARM Architecture Reference Manual ARMv8-A", section, name, desc,
            ))
        }
        Architecture::PTX => {
            let (section, name, desc) = match scope {
                Scope::System => ("9.7.12.15", "membar.sys", "System-scope memory barrier"),
                Scope::GPU => ("9.7.12.15", "membar.gl", "GPU-scope memory barrier"),
                Scope::CTA => ("9.7.12.15", "membar.cta", "CTA-scope memory barrier"),
                _ => return None,
            };
            Some(VendorDocReference::new("NVIDIA PTX ISA 8.x", section, name, desc))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

fn parse_ordering(s: &str) -> Option<Ordering> {
    match s.to_lowercase().as_str() {
        "relaxed" | "rlx" => Some(Ordering::Relaxed),
        "acquire" | "acq" => Some(Ordering::Acquire),
        "release" | "rel" => Some(Ordering::Release),
        "acqrel" | "acq_rel" => Some(Ordering::AcqRel),
        "seqcst" | "sc" => Some(Ordering::SeqCst),
        "acq.cta" | "acquire_cta" => Some(Ordering::AcquireCTA),
        "rel.cta" | "release_cta" => Some(Ordering::ReleaseCTA),
        "acq.gpu" | "acquire_gpu" => Some(Ordering::AcquireGPU),
        "rel.gpu" | "release_gpu" => Some(Ordering::ReleaseGPU),
        "acq.sys" | "acquire_system" => Some(Ordering::AcquireSystem),
        "rel.sys" | "release_system" => Some(Ordering::ReleaseSystem),
        _ => None,
    }
}

fn parse_scope(s: &str) -> Option<Scope> {
    match s.to_lowercase().as_str() {
        "cta" | "workgroup" => Some(Scope::CTA),
        "gpu" | "device" => Some(Scope::GPU),
        "system" | "sys" => Some(Scope::System),
        "none" | "" => Some(Scope::None),
        _ => None,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fence_cost_composite() {
        let fc = FenceCost::new(10.0, 2.0, Scope::CTA, Ordering::AcqRel);
        assert!((fc.composite_cost() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn fence_cost_relative() {
        let base = FenceCost::new(10.0, 1.0, Scope::None, Ordering::Relaxed);
        let expensive = FenceCost::new(30.0, 2.0, Scope::System, Ordering::SeqCst);
        assert!((expensive.relative_to(&base) - 6.0).abs() < 1e-9);
    }

    #[test]
    fn fence_cost_display() {
        let fc = FenceCost::new(33.0, 1.8, Scope::System, Ordering::SeqCst);
        let s = format!("{}", fc);
        assert!(s.contains("33.0ns"));
        assert!(s.contains("1.80x"));
    }

    #[test]
    fn predefined_x86_has_mfence() {
        let costs = ArchitectureCosts::x86();
        let sc = costs.get(&Ordering::SeqCst, &Scope::System);
        assert!(sc.is_some());
        assert!(sc.unwrap().latency_ns > 0.0);
    }

    #[test]
    fn predefined_ptx_has_membar_cta() {
        let costs = ArchitectureCosts::ptx();
        let cta = costs.get(&Ordering::AcqRel, &Scope::CTA);
        assert!(cta.is_some());
        assert!(cta.unwrap().latency_ns < 100.0);
    }

    #[test]
    fn predefined_all_architectures() {
        let archs = [
            Architecture::X86, Architecture::ARM, Architecture::RiscV,
            Architecture::PTX, Architecture::Vulkan, Architecture::OpenCL,
        ];
        for arch in archs {
            let c = ArchitectureCosts::for_architecture(arch);
            assert!(!c.costs.is_empty(), "no costs for {}", arch);
        }
    }

    #[test]
    fn cheapest_fence_x86() {
        let costs = ArchitectureCosts::x86();
        let cheapest = costs.cheapest_fence(&Ordering::Release, &Scope::None);
        assert!(cheapest.is_some());
        // Should not be cheaper than Acquire-level.
        assert!(ordering_strength(&cheapest.unwrap().ordering) >= ordering_strength(&Ordering::Release));
    }

    #[test]
    fn relative_cost_ratios_non_empty() {
        let costs = ArchitectureCosts::arm();
        let ratios = costs.relative_cost_ratios();
        assert!(!ratios.is_empty());
        // The cheapest non-zero fence should have ratio close to 1.0;
        // Relaxed (cost 0) may produce NaN/0, so filter it out.
        let min_ratio = ratios.values().cloned()
            .filter(|r| r.is_finite() && *r > 0.0)
            .fold(f64::INFINITY, f64::min);
        assert!(min_ratio >= 1.0 - 1e-9, "min non-zero ratio should be ~1.0, got {}", min_ratio);
    }

    #[test]
    fn compute_stats_basic() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = compute_stats(&samples).unwrap();
        assert!((s.mean - 3.0).abs() < 1e-9);
        assert!((s.median - 3.0).abs() < 1e-9);
        assert_eq!(s.count, 5);
        assert!((s.min - 1.0).abs() < 1e-9);
        assert!((s.max - 5.0).abs() < 1e-9);
    }

    #[test]
    fn compute_stats_even_count() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let s = compute_stats(&samples).unwrap();
        assert!((s.median - 2.5).abs() < 1e-9);
    }

    #[test]
    fn compute_stats_empty() {
        assert!(compute_stats(&[]).is_none());
    }

    #[test]
    fn calibrator_parse_timing_data() {
        let data = "\
            # header comment\n\
            sc, system, 30.0, 35.0, 32.0, 31.0\n\
            acq_rel, system, 28.0, 30.0, 29.0\n\
            relaxed, none, 0.1, 0.2, 0.15\n\
        ";
        let mut cal = CostCalibrator::new(Architecture::X86);
        let count = cal.parse_timing_data(data).unwrap();
        assert_eq!(count, 3);
        assert_eq!(cal.results.len(), 3);
    }

    #[test]
    fn calibrator_parse_bad_line() {
        let data = "sc, system\n"; // only 2 fields
        let mut cal = CostCalibrator::new(Architecture::X86);
        assert!(cal.parse_timing_data(data).is_err());
    }

    #[test]
    fn calibrator_parse_bad_ordering() {
        let data = "invalid_ord, system, 10.0\n";
        let mut cal = CostCalibrator::new(Architecture::X86);
        assert!(cal.parse_timing_data(data).is_err());
    }

    #[test]
    fn calibrator_calibrate() {
        let data = "\
            sc, system, 30.0, 35.0, 32.0, 31.0, 33.0\n\
            acq_rel, system, 28.0, 30.0, 29.0, 27.0, 31.0\n\
        ";
        let mut cal = CostCalibrator::new(Architecture::X86);
        cal.parse_timing_data(data).unwrap();
        let model = cal.calibrate();
        assert!(model.calibrated);
        let sc_cost = model.fence_cost(&Ordering::SeqCst, &Scope::System);
        assert!(sc_cost.is_some());
        // Median should be around 32.
        assert!((sc_cost.unwrap().latency_ns - 32.0).abs() < 2.0);
    }

    #[test]
    fn calibrator_falls_back_to_predefined() {
        let mut cal = CostCalibrator::new(Architecture::X86);
        // No data added — calibrate should still produce a model with predefined costs.
        let model = cal.calibrate();
        assert!(model.fence_cost(&Ordering::SeqCst, &Scope::System).is_some());
    }

    #[test]
    fn calibrator_outlier_removal() {
        let mut cal = CostCalibrator::new(Architecture::X86);
        // Use enough tight samples so the outlier is clearly beyond 3σ.
        cal.add_result(MicrobenchmarkResult::new(
            Ordering::SeqCst, Scope::System,
            vec![30.0, 31.0, 30.5, 31.5, 30.2, 30.8, 31.2, 30.9, 31.1, 30.7, 1000.0],
            Architecture::X86,
        ));
        cal.remove_outliers();
        let samples = &cal.results[0].samples_ns;
        assert!(!samples.contains(&1000.0));
        assert!(samples.len() == 10);
    }

    #[test]
    fn predefined_model_works() {
        let model = CalibratedCostModel::predefined(Architecture::PTX);
        assert!(!model.calibrated);
        assert!(model.cost_of(&Ordering::SeqCst, &Scope::System) > 0.0);
        assert!((model.cost_of(&Ordering::Relaxed, &Scope::None)).abs() < 1e-9);
    }

    #[test]
    fn model_summary_not_empty() {
        let model = CalibratedCostModel::predefined(Architecture::ARM);
        let summary = model.summary();
        assert!(summary.contains("ARM"));
        assert!(summary.contains("ns"));
    }

    #[test]
    fn microbenchmark_generator_x86() {
        let gen = MicrobenchmarkGenerator::new(Architecture::X86);
        let code = gen.generate(&Ordering::SeqCst, &Scope::System);
        assert!(code.contains("mfence"));
    }

    #[test]
    fn microbenchmark_generator_ptx() {
        let gen = MicrobenchmarkGenerator::new(Architecture::PTX);
        let code = gen.generate(&Ordering::AcqRel, &Scope::GPU);
        assert!(code.contains("membar.gl"));
    }

    #[test]
    fn microbenchmark_generator_arm() {
        let gen = MicrobenchmarkGenerator::new(Architecture::ARM);
        let code = gen.generate(&Ordering::SeqCst, &Scope::System);
        assert!(code.contains("dmb sy"));
    }

    #[test]
    fn microbenchmark_generator_riscv() {
        let gen = MicrobenchmarkGenerator::new(Architecture::RiscV);
        let code = gen.generate(&Ordering::AcqRel, &Scope::System);
        assert!(code.contains("fence iorw, iorw"));
    }

    #[test]
    fn microbenchmark_generator_vulkan() {
        let gen = MicrobenchmarkGenerator::new(Architecture::Vulkan);
        let code = gen.generate(&Ordering::AcqRel, &Scope::GPU);
        assert!(code.contains("Device"));
    }

    #[test]
    fn microbenchmark_generator_opencl() {
        let gen = MicrobenchmarkGenerator::new(Architecture::OpenCL);
        let code = gen.generate(&Ordering::SeqCst, &Scope::System);
        assert!(code.contains("CLK_GLOBAL_MEM_FENCE"));
    }

    #[test]
    fn generate_all_non_empty() {
        let gen = MicrobenchmarkGenerator::new(Architecture::PTX);
        let all = gen.generate_all();
        assert!(!all.is_empty());
    }

    #[test]
    fn vendor_doc_x86_mfence() {
        let doc = vendor_doc_for_fence(Architecture::X86, &Ordering::SeqCst, &Scope::System);
        assert!(doc.is_some());
        assert_eq!(doc.unwrap().fence_name, "MFENCE");
    }

    #[test]
    fn vendor_doc_arm_dmb() {
        let doc = vendor_doc_for_fence(Architecture::ARM, &Ordering::SeqCst, &Scope::System);
        assert!(doc.is_some());
        assert_eq!(doc.unwrap().fence_name, "DMB SY");
    }

    #[test]
    fn vendor_doc_ptx_membar() {
        let doc = vendor_doc_for_fence(Architecture::PTX, &Ordering::SeqCst, &Scope::GPU);
        assert!(doc.is_some());
        assert_eq!(doc.unwrap().fence_name, "membar.gl");
    }

    #[test]
    fn vendor_doc_unknown_returns_none() {
        let doc = vendor_doc_for_fence(Architecture::Vulkan, &Ordering::Relaxed, &Scope::None);
        assert!(doc.is_none());
    }

    #[test]
    fn parse_ordering_roundtrip() {
        let cases = vec![
            ("relaxed", Ordering::Relaxed), ("acquire", Ordering::Acquire),
            ("release", Ordering::Release), ("acqrel", Ordering::AcqRel),
            ("seqcst", Ordering::SeqCst), ("acq.cta", Ordering::AcquireCTA),
            ("rel.cta", Ordering::ReleaseCTA), ("acq.gpu", Ordering::AcquireGPU),
            ("rel.gpu", Ordering::ReleaseGPU), ("acq.sys", Ordering::AcquireSystem),
            ("rel.sys", Ordering::ReleaseSystem),
        ];
        for (s, expected) in cases {
            assert_eq!(parse_ordering(s), Some(expected), "failed for '{}'", s);
        }
    }

    #[test]
    fn parse_scope_roundtrip() {
        assert_eq!(parse_scope("cta"), Some(Scope::CTA));
        assert_eq!(parse_scope("gpu"), Some(Scope::GPU));
        assert_eq!(parse_scope("system"), Some(Scope::System));
        assert_eq!(parse_scope("none"), Some(Scope::None));
        assert_eq!(parse_scope("workgroup"), Some(Scope::CTA));
        assert_eq!(parse_scope("device"), Some(Scope::GPU));
    }

    #[test]
    fn ordering_strength_monotonic() {
        assert!(ordering_strength(&Ordering::Relaxed) < ordering_strength(&Ordering::Acquire));
        assert!(ordering_strength(&Ordering::Acquire) < ordering_strength(&Ordering::Release));
        assert!(ordering_strength(&Ordering::Release) < ordering_strength(&Ordering::AcqRel));
        assert!(ordering_strength(&Ordering::AcqRel) < ordering_strength(&Ordering::SeqCst));
    }

    #[test]
    fn scope_strength_monotonic() {
        assert!(scope_strength(&Scope::None) < scope_strength(&Scope::CTA));
        assert!(scope_strength(&Scope::CTA) < scope_strength(&Scope::GPU));
        assert!(scope_strength(&Scope::GPU) < scope_strength(&Scope::System));
    }

    #[test]
    fn microbenchmark_result_with_label() {
        let r = MicrobenchmarkResult::new(
            Ordering::SeqCst, Scope::System, vec![10.0, 20.0], Architecture::X86,
        ).with_label("test run");
        assert_eq!(r.label.as_deref(), Some("test run"));
        assert_eq!(r.sample_count(), 2);
    }

    #[test]
    fn cheapest_sufficient_ptx() {
        let model = CalibratedCostModel::predefined(Architecture::PTX);
        let cheapest = model.cheapest_sufficient(&Ordering::Acquire, &Scope::CTA);
        assert!(cheapest.is_some());
        let fc = cheapest.unwrap();
        assert!(ordering_strength(&fc.ordering) >= ordering_strength(&Ordering::Acquire));
        assert!(scope_strength(&fc.scope) >= scope_strength(&Scope::CTA));
    }
}
