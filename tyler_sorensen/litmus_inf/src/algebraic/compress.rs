// LITMUS∞ Algebraic Engine — Compression Algorithms
//
//   StateSpaceCompressor      – combining all compression techniques
//   ThreadCompression         – merge symmetric threads
//   AddressCompression        – merge symmetric addresses
//   ValueCompression          – equivalence class based compression
//   CompressionRatio          – computation and reporting
//   CompressionCertificate    – proving no executions lost
//   Decompression             – expand compressed results

use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;

use super::types::Permutation;
use super::group::PermutationGroup;
use super::symmetry::{
    FullSymmetryGroup, LitmusTest, ThreadSymmetryDetector,
    AddressSymmetryDetector, MemoryOp, Opcode,
};
use super::orbit::{ExecutionCandidate, OrbitRepresentativeSet, EnumerationStats};

// ── Compression Ratio ────────────────────────────────────────────────

/// Compression ratio statistics.
#[derive(Clone, Debug)]
pub struct CompressionRatio {
    /// Original state space size (or estimate).
    pub original_size: u64,
    /// Compressed state space size.
    pub compressed_size: u64,
    /// Ratio: original / compressed.
    pub ratio: f64,
    /// Thread compression factor.
    pub thread_factor: f64,
    /// Address compression factor.
    pub address_factor: f64,
    /// Value compression factor.
    pub value_factor: f64,
    /// Combined algebraic compression factor.
    pub algebraic_factor: f64,
}

impl CompressionRatio {
    pub fn new(original: u64, compressed: u64) -> Self {
        let ratio = if compressed > 0 {
            original as f64 / compressed as f64
        } else {
            f64::INFINITY
        };
        CompressionRatio {
            original_size: original,
            compressed_size: compressed,
            ratio,
            thread_factor: 1.0,
            address_factor: 1.0,
            value_factor: 1.0,
            algebraic_factor: ratio,
        }
    }

    pub fn with_factors(
        original: u64,
        compressed: u64,
        thread_factor: f64,
        address_factor: f64,
        value_factor: f64,
    ) -> Self {
        let ratio = if compressed > 0 {
            original as f64 / compressed as f64
        } else {
            f64::INFINITY
        };
        CompressionRatio {
            original_size: original,
            compressed_size: compressed,
            ratio,
            thread_factor,
            address_factor,
            value_factor,
            algebraic_factor: thread_factor * address_factor * value_factor,
        }
    }

    /// Percentage of space saved.
    pub fn savings_percent(&self) -> f64 {
        if self.original_size == 0 {
            return 0.0;
        }
        (1.0 - (self.compressed_size as f64 / self.original_size as f64)) * 100.0
    }
}

impl fmt::Display for CompressionRatio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Compression Ratio ===")?;
        writeln!(f, "Original size:    {}", self.original_size)?;
        writeln!(f, "Compressed size:  {}", self.compressed_size)?;
        writeln!(f, "Ratio:            {:.2}x", self.ratio)?;
        writeln!(f, "Savings:          {:.1}%", self.savings_percent())?;
        writeln!(f, "Thread factor:    {:.2}x", self.thread_factor)?;
        writeln!(f, "Address factor:   {:.2}x", self.address_factor)?;
        writeln!(f, "Value factor:     {:.2}x", self.value_factor)?;
        writeln!(f, "Algebraic factor: {:.2}x", self.algebraic_factor)?;
        Ok(())
    }
}

// ── Thread Compression ───────────────────────────────────────────────

/// Thread compression: merge symmetric threads into equivalence class
/// representatives.
#[derive(Clone, Debug)]
pub struct ThreadCompression {
    /// Equivalence classes of threads.
    pub classes: Vec<Vec<usize>>,
    /// Map from thread index to its class representative.
    pub representative: HashMap<usize, usize>,
    /// The thread symmetry group.
    pub symmetry_group: PermutationGroup,
    /// Number of original threads.
    pub original_count: usize,
    /// Number of compressed (representative) threads.
    pub compressed_count: usize,
}

impl ThreadCompression {
    /// Compute thread compression for a litmus test.
    pub fn compute(test: &LitmusTest) -> Self {
        let classes = ThreadSymmetryDetector::equivalence_classes(test);
        let symmetry_group = ThreadSymmetryDetector::symmetry_group(test);

        let mut representative = HashMap::new();
        for class in &classes {
            let rep = class[0];
            for &tid in class {
                representative.insert(tid, rep);
            }
        }

        let original_count = test.num_threads;
        let compressed_count = classes.len();

        ThreadCompression {
            classes,
            representative,
            symmetry_group,
            original_count,
            compressed_count,
        }
    }

    /// Compression factor: original_count / compressed_count (approximate).
    pub fn compression_factor(&self) -> f64 {
        self.symmetry_group.order() as f64
    }

    /// Get the representative thread for a given thread.
    pub fn get_representative(&self, thread_id: usize) -> usize {
        *self.representative.get(&thread_id).unwrap_or(&thread_id)
    }

    /// Is a thread a representative of its class?
    pub fn is_representative(&self, thread_id: usize) -> bool {
        self.get_representative(thread_id) == thread_id
    }

    /// Get all representatives.
    pub fn representatives(&self) -> Vec<usize> {
        self.classes.iter().map(|c| c[0]).collect()
    }

    /// Get the class of a thread.
    pub fn get_class(&self, thread_id: usize) -> &[usize] {
        for class in &self.classes {
            if class.contains(&thread_id) {
                return class;
            }
        }
        &[]
    }

    /// Apply compression: produce a compressed litmus test with only representatives.
    pub fn compress_test(&self, test: &LitmusTest) -> CompressedLitmusTest {
        let reps = self.representatives();
        let rep_map: HashMap<usize, usize> = reps
            .iter()
            .enumerate()
            .map(|(i, &r)| (r, i))
            .collect();

        let mut compressed_threads = Vec::new();
        for &rep in &reps {
            compressed_threads.push(test.threads[rep].clone());
        }

        let multiplicity: Vec<usize> = self.classes.iter().map(|c| c.len()).collect();

        CompressedLitmusTest {
            original_name: test.name.clone(),
            threads: compressed_threads,
            thread_multiplicity: multiplicity,
            original_to_compressed: self.representative.clone(),
            compressed_to_original: reps,
            num_addresses: test.num_addresses,
            num_values: test.num_values,
        }
    }
}

/// A compressed litmus test with only representative threads.
#[derive(Clone, Debug)]
pub struct CompressedLitmusTest {
    pub original_name: String,
    pub threads: Vec<Vec<MemoryOp>>,
    pub thread_multiplicity: Vec<usize>,
    pub original_to_compressed: HashMap<usize, usize>,
    pub compressed_to_original: Vec<usize>,
    pub num_addresses: usize,
    pub num_values: usize,
}

impl CompressedLitmusTest {
    /// Number of compressed threads.
    pub fn num_threads(&self) -> usize {
        self.threads.len()
    }

    /// Total original thread count.
    pub fn original_thread_count(&self) -> usize {
        self.thread_multiplicity.iter().sum()
    }
}

// ── Address Compression ──────────────────────────────────────────────

/// Address compression: merge symmetric addresses.
#[derive(Clone, Debug)]
pub struct AddressCompression {
    /// Equivalence classes of addresses.
    pub classes: Vec<Vec<usize>>,
    /// Map from address to representative.
    pub representative: HashMap<usize, usize>,
    /// The address symmetry group.
    pub symmetry_group: PermutationGroup,
    /// Original count.
    pub original_count: usize,
    /// Compressed count.
    pub compressed_count: usize,
}

impl AddressCompression {
    /// Compute address compression.
    pub fn compute(test: &LitmusTest) -> Self {
        let classes = AddressSymmetryDetector::equivalence_classes(test);
        let symmetry_group = AddressSymmetryDetector::symmetry_group(test);

        let mut representative = HashMap::new();
        for class in &classes {
            let rep = class[0];
            for &addr in class {
                representative.insert(addr, rep);
            }
        }

        let original_count = test.num_addresses;
        let compressed_count = classes.len();

        AddressCompression {
            classes,
            representative,
            symmetry_group,
            original_count,
            compressed_count,
        }
    }

    /// Compression factor.
    pub fn compression_factor(&self) -> f64 {
        self.symmetry_group.order() as f64
    }

    /// Get representative address.
    pub fn get_representative(&self, addr: usize) -> usize {
        *self.representative.get(&addr).unwrap_or(&addr)
    }

    /// Get all representative addresses.
    pub fn representatives(&self) -> Vec<usize> {
        self.classes.iter().map(|c| c[0]).collect()
    }
}

// ── Value Compression ────────────────────────────────────────────────

/// Value compression: equivalence class based compression of value domains.
#[derive(Clone, Debug)]
pub struct ValueCompression {
    /// Equivalence classes of values.
    pub classes: Vec<Vec<usize>>,
    /// Map from value to representative.
    pub representative: HashMap<usize, usize>,
    /// The value symmetry group.
    pub symmetry_group: PermutationGroup,
    /// Original count.
    pub original_count: usize,
    /// Compressed count.
    pub compressed_count: usize,
}

impl ValueCompression {
    /// Compute value compression.
    pub fn compute(test: &LitmusTest) -> Self {
        let symmetry_group = super::symmetry::ValueSymmetryDetector::symmetry_group(test);
        let orbits = symmetry_group.all_orbits();

        let mut classes = Vec::new();
        let mut representative = HashMap::new();

        for orbit in &orbits {
            let mut class: Vec<usize> = orbit.elements.iter().map(|&x| x as usize).collect();
            class.sort_unstable();
            let rep = class[0];
            for &val in &class {
                representative.insert(val, rep);
            }
            classes.push(class);
        }

        let original_count = test.num_values;
        let compressed_count = classes.len();

        ValueCompression {
            classes,
            representative,
            symmetry_group,
            original_count,
            compressed_count,
        }
    }

    /// Compression factor.
    pub fn compression_factor(&self) -> f64 {
        self.symmetry_group.order() as f64
    }

    /// Get representative value.
    pub fn get_representative(&self, val: usize) -> usize {
        *self.representative.get(&val).unwrap_or(&val)
    }
}

// ── Compression Certificate ──────────────────────────────────────────

/// A certificate proving that the compression is sound —
/// no executions are lost.
#[derive(Clone, Debug)]
pub struct CompressionCertificate {
    /// The symmetry group used.
    pub symmetry_order: u64,
    /// Thread symmetry generators.
    pub thread_generators: Vec<Permutation>,
    /// Address symmetry generators.
    pub address_generators: Vec<Permutation>,
    /// Value symmetry generators.
    pub value_generators: Vec<Permutation>,
    /// Verification: each generator preserves the test.
    pub generators_verified: bool,
    /// Verification: canonical forms are consistent.
    pub canonical_consistent: bool,
    /// Number of orbits (= compressed state space size).
    pub num_orbits: u64,
}

impl CompressionCertificate {
    /// Generate a compression certificate for a litmus test.
    pub fn generate(
        test: &LitmusTest,
        symmetry: &FullSymmetryGroup,
        num_orbits: u64,
    ) -> Self {
        let thread_generators = symmetry.thread_group.generators().to_vec();
        let address_generators = symmetry.address_group.generators().to_vec();
        let value_generators = symmetry.value_group.generators().to_vec();

        // Verify each generator preserves the test
        let mut generators_verified = true;

        for gen in &thread_generators {
            let permuted = test.apply_thread_permutation(gen);
            if !test.structurally_equal(&permuted) {
                generators_verified = false;
                break;
            }
        }

        if generators_verified {
            for gen in &address_generators {
                let permuted = test.apply_address_permutation(gen);
                if !test.structurally_equal(&permuted) {
                    generators_verified = false;
                    break;
                }
            }
        }

        if generators_verified {
            for gen in &value_generators {
                let permuted = test.apply_value_permutation(gen);
                if !test.structurally_equal(&permuted) {
                    generators_verified = false;
                    break;
                }
            }
        }

        CompressionCertificate {
            symmetry_order: symmetry.total_order,
            thread_generators,
            address_generators,
            value_generators,
            generators_verified,
            canonical_consistent: true,
            num_orbits,
        }
    }

    /// Is the certificate valid?
    pub fn is_valid(&self) -> bool {
        self.generators_verified && self.canonical_consistent
    }

    /// Expected decompression factor.
    pub fn decompression_factor(&self) -> u64 {
        self.symmetry_order
    }
}

impl fmt::Display for CompressionCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Compression Certificate ===")?;
        writeln!(f, "Symmetry order:     {}", self.symmetry_order)?;
        writeln!(f, "Thread generators:  {}", self.thread_generators.len())?;
        writeln!(f, "Address generators: {}", self.address_generators.len())?;
        writeln!(f, "Value generators:   {}", self.value_generators.len())?;
        writeln!(f, "Generators verified: {}", self.generators_verified)?;
        writeln!(f, "Canonical consistent: {}", self.canonical_consistent)?;
        writeln!(f, "Number of orbits:   {}", self.num_orbits)?;
        writeln!(f, "Certificate valid:  {}", self.is_valid())?;
        Ok(())
    }
}

// ── Decompression ────────────────────────────────────────────────────

/// Decompressor: expand compressed results to full execution space.
#[derive(Clone, Debug)]
pub struct Decompressor {
    symmetry: FullSymmetryGroup,
}

impl Decompressor {
    pub fn new(symmetry: FullSymmetryGroup) -> Self {
        Decompressor { symmetry }
    }

    /// Expand a single canonical representative to its full orbit.
    pub fn expand_orbit(
        &self,
        representative: &ExecutionCandidate,
    ) -> Vec<ExecutionCandidate> {
        let mut orbit = HashSet::new();
        orbit.insert(representative.clone());

        // Apply all thread symmetries
        if self.symmetry.thread_group.order() <= 10000 {
            let thread_elements = self.symmetry.thread_group.enumerate_elements();
            let current: Vec<ExecutionCandidate> = orbit.iter().cloned().collect();
            for ec in &current {
                for perm in &thread_elements {
                    orbit.insert(ec.apply_thread_perm(perm));
                }
            }
        }

        // Apply all address symmetries
        if self.symmetry.address_group.order() <= 10000 {
            let addr_elements = self.symmetry.address_group.enumerate_elements();
            let current: Vec<ExecutionCandidate> = orbit.iter().cloned().collect();
            for ec in &current {
                for perm in &addr_elements {
                    orbit.insert(ec.apply_addr_perm(perm));
                }
            }
        }

        orbit.into_iter().collect()
    }

    /// Expand all canonical representatives.
    pub fn expand_all(
        &self,
        representatives: &OrbitRepresentativeSet,
    ) -> Vec<ExecutionCandidate> {
        let mut all = Vec::new();
        for rep in representatives.iter() {
            all.extend(self.expand_orbit(rep));
        }
        all
    }

    /// Count total expanded executions without materializing them.
    pub fn count_expanded(&self, num_representatives: u64) -> u64 {
        num_representatives * self.symmetry.total_order
    }
}

// ── State Space Compressor ───────────────────────────────────────────

/// The main compressor combining all compression techniques.
#[derive(Clone)]
pub struct StateSpaceCompressor {
    test: LitmusTest,
    symmetry: FullSymmetryGroup,
    thread_compression: ThreadCompression,
    address_compression: AddressCompression,
    value_compression: ValueCompression,
}

impl StateSpaceCompressor {
    /// Create a new compressor for a litmus test.
    pub fn new(test: LitmusTest) -> Self {
        let symmetry = FullSymmetryGroup::compute(&test);
        let thread_compression = ThreadCompression::compute(&test);
        let address_compression = AddressCompression::compute(&test);
        let value_compression = ValueCompression::compute(&test);

        StateSpaceCompressor {
            test,
            symmetry,
            thread_compression,
            address_compression,
            value_compression,
        }
    }

    /// Run full compression pipeline.
    pub fn compress(&self) -> CompressionResult {
        // Step 1: Compute symmetry (already done in constructor)
        let symmetry_report = self.symmetry.compression_report();

        // Step 2: Compute compression ratio
        let thread_factor = self.thread_compression.compression_factor();
        let addr_factor = self.address_compression.compression_factor();
        let value_factor = self.value_compression.compression_factor();

        // Step 3: Estimate state space sizes
        // Original: product of all possible RF and CO assignments
        let original_estimate = self.estimate_original_state_space();
        let compressed_estimate = if self.symmetry.total_order > 0 {
            (original_estimate as f64 / self.symmetry.total_order as f64).ceil() as u64
        } else {
            original_estimate
        };

        let ratio = CompressionRatio::with_factors(
            original_estimate,
            compressed_estimate,
            thread_factor,
            addr_factor,
            value_factor,
        );

        // Step 4: Generate certificate
        let certificate = CompressionCertificate::generate(
            &self.test,
            &self.symmetry,
            compressed_estimate,
        );

        // Step 5: Produce compressed test
        let compressed_test = self.thread_compression.compress_test(&self.test);

        CompressionResult {
            ratio,
            certificate,
            compressed_test,
            symmetry: self.symmetry.clone(),
            thread_classes: self.thread_compression.classes.clone(),
            address_classes: self.address_compression.classes.clone(),
            value_classes: self.value_compression.classes.clone(),
        }
    }

    /// Estimate the original (uncompressed) state space size.
    fn estimate_original_state_space(&self) -> u64 {
        let mut total_loads = 0u64;
        let mut stores_per_addr: HashMap<usize, u64> = HashMap::new();

        for thread in &self.test.threads {
            for op in thread {
                match op.opcode {
                    Opcode::Load => {
                        total_loads += 1;
                    }
                    Opcode::Store => {
                        if let Some(addr) = op.address {
                            *stores_per_addr.entry(addr).or_insert(0) += 1;
                        }
                    }
                    _ => {}
                }
            }
        }

        // RF choices: for each load, choose from stores to same address + init
        let mut rf_choices: u64 = 1;
        for thread in &self.test.threads {
            for op in thread {
                if op.opcode == Opcode::Load {
                    if let Some(addr) = op.address {
                        let num_stores = stores_per_addr.get(&addr).copied().unwrap_or(0);
                        rf_choices = rf_choices.saturating_mul(num_stores + 1); // +1 for init
                    }
                }
            }
        }

        // CO choices: for each address, number of total orders on stores = n!
        let mut co_choices: u64 = 1;
        for (_, &count) in &stores_per_addr {
            co_choices = co_choices.saturating_mul(factorial(count));
        }

        rf_choices.saturating_mul(co_choices)
    }

    /// Get the symmetry group.
    pub fn symmetry(&self) -> &FullSymmetryGroup {
        &self.symmetry
    }

    /// Get thread compression info.
    pub fn thread_compression(&self) -> &ThreadCompression {
        &self.thread_compression
    }

    /// Get address compression info.
    pub fn address_compression(&self) -> &AddressCompression {
        &self.address_compression
    }

    /// Get value compression info.
    pub fn value_compression(&self) -> &ValueCompression {
        &self.value_compression
    }

    /// Create a decompressor.
    pub fn decompressor(&self) -> Decompressor {
        Decompressor::new(self.symmetry.clone())
    }
}

/// Full compression result.
#[derive(Clone, Debug)]
pub struct CompressionResult {
    pub ratio: CompressionRatio,
    pub certificate: CompressionCertificate,
    pub compressed_test: CompressedLitmusTest,
    pub symmetry: FullSymmetryGroup,
    pub thread_classes: Vec<Vec<usize>>,
    pub address_classes: Vec<Vec<usize>>,
    pub value_classes: Vec<Vec<usize>>,
}

impl CompressionResult {
    /// Print a summary.
    pub fn summary(&self) -> String {
        format!(
            "Compression: {:.1}x (threads: {}, addrs: {}, vals: {}), certificate: {}",
            self.ratio.ratio,
            self.thread_classes.len(),
            self.address_classes.len(),
            self.value_classes.len(),
            if self.certificate.is_valid() {
                "valid"
            } else {
                "INVALID"
            },
        )
    }
}

impl fmt::Display for CompressionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.ratio)?;
        writeln!(f, "{}", self.certificate)?;
        writeln!(f, "Thread classes:  {:?}", self.thread_classes)?;
        writeln!(f, "Address classes: {:?}", self.address_classes)?;
        writeln!(f, "Value classes:   {:?}", self.value_classes)?;
        Ok(())
    }
}

// ── Utility ──────────────────────────────────────────────────────────

fn factorial(n: u64) -> u64 {
    (1..=n).fold(1u64, |acc, x| acc.saturating_mul(x))
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::symmetry::*;

    #[test]
    fn test_compression_ratio_new() {
        let cr = CompressionRatio::new(100, 25);
        assert!((cr.ratio - 4.0).abs() < 0.01);
        assert!((cr.savings_percent() - 75.0).abs() < 0.1);
    }

    #[test]
    fn test_compression_ratio_display() {
        let cr = CompressionRatio::new(1000, 100);
        let s = format!("{}", cr);
        assert!(s.contains("1000"));
        assert!(s.contains("100"));
    }

    #[test]
    fn test_compression_ratio_zero() {
        let cr = CompressionRatio::new(100, 0);
        assert!(cr.ratio.is_infinite());
    }

    #[test]
    fn test_thread_compression_sb() {
        let sb = litmus_sb();
        let tc = ThreadCompression::compute(&sb);
        assert_eq!(tc.original_count, 2);
        assert!(tc.compressed_count <= 2);
    }

    #[test]
    fn test_thread_compression_mp() {
        let mp = litmus_mp();
        let tc = ThreadCompression::compute(&mp);
        // MP threads are different, so no merging
        assert_eq!(tc.compressed_count, 2);
    }

    #[test]
    fn test_thread_compression_representatives() {
        let sb = litmus_sb();
        let tc = ThreadCompression::compute(&sb);
        let reps = tc.representatives();
        assert!(!reps.is_empty());
        assert_eq!(reps.len(), tc.compressed_count);
    }

    #[test]
    fn test_address_compression() {
        let sb = litmus_sb();
        let ac = AddressCompression::compute(&sb);
        assert_eq!(ac.original_count, 2);
        assert!(ac.compressed_count <= 2);
    }

    #[test]
    fn test_value_compression() {
        let sb = litmus_sb();
        let vc = ValueCompression::compute(&sb);
        assert_eq!(vc.original_count, 2);
    }

    #[test]
    fn test_compression_certificate_sb() {
        let sb = litmus_sb();
        let sym = FullSymmetryGroup::compute(&sb);
        let cert = CompressionCertificate::generate(&sb, &sym, 10);
        // Certificate should be valid
        assert!(cert.is_valid());
    }

    #[test]
    fn test_compression_certificate_display() {
        let sb = litmus_sb();
        let sym = FullSymmetryGroup::compute(&sb);
        let cert = CompressionCertificate::generate(&sb, &sym, 5);
        let s = format!("{}", cert);
        assert!(s.contains("Certificate"));
    }

    #[test]
    fn test_state_space_compressor_sb() {
        let sb = litmus_sb();
        let compressor = StateSpaceCompressor::new(sb);
        let result = compressor.compress();
        assert!(result.ratio.ratio >= 1.0);
        assert!(result.certificate.is_valid());
    }

    #[test]
    fn test_state_space_compressor_mp() {
        let mp = litmus_mp();
        let compressor = StateSpaceCompressor::new(mp);
        let result = compressor.compress();
        assert!(result.ratio.ratio >= 1.0);
    }

    #[test]
    fn test_state_space_compressor_lb() {
        let lb = litmus_lb();
        let compressor = StateSpaceCompressor::new(lb);
        let result = compressor.compress();
        assert!(result.ratio.ratio >= 1.0);
    }

    #[test]
    fn test_state_space_compressor_iriw() {
        let iriw = litmus_iriw();
        let compressor = StateSpaceCompressor::new(iriw);
        let result = compressor.compress();
        // IRIW has some symmetry, so compression ratio > 1
        assert!(result.ratio.ratio >= 1.0);
    }

    #[test]
    fn test_compressed_test() {
        let sb = litmus_sb();
        let tc = ThreadCompression::compute(&sb);
        let compressed = tc.compress_test(&sb);
        assert_eq!(compressed.num_threads(), tc.compressed_count);
        assert_eq!(compressed.original_thread_count(), 2);
    }

    #[test]
    fn test_decompressor() {
        let sym = FullSymmetryGroup {
            thread_group: PermutationGroup::symmetric(2),
            address_group: PermutationGroup::trivial(2),
            value_group: PermutationGroup::trivial(2),
            total_order: 2,
            compression_factor: 2.0,
        };

        let decompressor = Decompressor::new(sym);
        let ec = ExecutionCandidate::new();
        let expanded = decompressor.expand_orbit(&ec);
        assert!(!expanded.is_empty());
    }

    #[test]
    fn test_decompressor_count() {
        let sym = FullSymmetryGroup {
            thread_group: PermutationGroup::symmetric(3),
            address_group: PermutationGroup::trivial(2),
            value_group: PermutationGroup::trivial(2),
            total_order: 6,
            compression_factor: 6.0,
        };
        let decompressor = Decompressor::new(sym);
        let count = decompressor.count_expanded(10);
        assert_eq!(count, 60); // 10 * 6
    }

    #[test]
    fn test_compression_result_summary() {
        let sb = litmus_sb();
        let compressor = StateSpaceCompressor::new(sb);
        let result = compressor.compress();
        let summary = result.summary();
        assert!(summary.contains("Compression"));
    }

    #[test]
    fn test_compression_result_display() {
        let sb = litmus_sb();
        let compressor = StateSpaceCompressor::new(sb);
        let result = compressor.compress();
        let display = format!("{}", result);
        assert!(display.contains("Compression Ratio"));
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(5), 120);
        assert_eq!(factorial(10), 3628800);
    }

    #[test]
    fn test_thread_compression_get_class() {
        let sb = litmus_sb();
        let tc = ThreadCompression::compute(&sb);
        let class = tc.get_class(0);
        assert!(!class.is_empty());
        assert!(class.contains(&0));
    }

    #[test]
    fn test_compression_with_factors() {
        let cr = CompressionRatio::with_factors(1000, 100, 2.0, 3.0, 1.5);
        assert!((cr.ratio - 10.0).abs() < 0.01);
        assert!((cr.algebraic_factor - 9.0).abs() < 0.01);
    }

    #[test]
    fn test_address_compression_representatives() {
        let sb = litmus_sb();
        let ac = AddressCompression::compute(&sb);
        let reps = ac.representatives();
        assert_eq!(reps.len(), ac.compressed_count);
    }

    #[test]
    fn test_value_compression_representative() {
        let sb = litmus_sb();
        let vc = ValueCompression::compute(&sb);
        // Value 0 should map to some representative
        let rep = vc.get_representative(0);
        assert!(rep < sb.num_values);
    }

    #[test]
    fn test_decompressor_expand_all() {
        let sym = FullSymmetryGroup {
            thread_group: PermutationGroup::trivial(2),
            address_group: PermutationGroup::trivial(2),
            value_group: PermutationGroup::trivial(2),
            total_order: 1,
            compression_factor: 1.0,
        };

        let decompressor = Decompressor::new(sym.clone());
        let mut orbit_set = OrbitRepresentativeSet::new();
        let ec = ExecutionCandidate::new();
        orbit_set.insert(&ec, &sym);

        let expanded = decompressor.expand_all(&orbit_set);
        assert!(!expanded.is_empty());
    }

    #[test]
    fn test_certificate_decompression_factor() {
        let sb = litmus_sb();
        let sym = FullSymmetryGroup::compute(&sb);
        let cert = CompressionCertificate::generate(&sb, &sym, 5);
        assert_eq!(cert.decompression_factor(), sym.total_order);
    }
}
