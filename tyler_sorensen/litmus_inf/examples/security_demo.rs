//! Security Analysis Demo for LITMUS∞
//!
//! Demonstrates quantitative information flow analysis, timing side
//! channel detection, vulnerability assessment, residue analysis,
//! abstract interpretation, and formal security verification.

use litmus_infinity::security::qif::*;
use litmus_infinity::security::timing::*;
use litmus_infinity::security::vulnerability::*;
use litmus_infinity::security::residue::*;
use litmus_infinity::security::formal::*;
use std::collections::HashMap;

/// Demonstrate Quantitative Information Flow analysis.
fn demo_qif() {
    println!("=== Quantitative Information Flow Analysis ===\n");

    // 1. Channel matrix analysis
    println!("  --- Channel Matrices ---");

    let identity = ChannelMatrix::identity(4);
    let uniform = ChannelMatrix::uniform(4, 4);
    let bsc = ChannelMatrix::binary_symmetric(0.1);
    let deterministic = ChannelMatrix::deterministic(&[0, 0, 1, 1], 2);

    let channels: Vec<(&str, &ChannelMatrix)> = vec![
        ("Identity (full leak)", &identity),
        ("Uniform (no leak)", &uniform),
        ("BSC(0.1)", &bsc),
        ("Deterministic", &deterministic),
    ];

    for (name, channel) in &channels {
        let leak = Leakage::compute(channel);
        let ga = GuessingAdvantage::compute(channel);
        let cap = ChannelCapacity::compute(channel);
        println!("  {} | capacity={:.3} bits | severity={} | advantage={}",
            name, cap.capacity, leak.severity(),
            if ga.has_advantage() { "YES" } else { "no" });
    }

    // 2. Shannon entropy
    println!("\n  --- Entropy Analysis ---");
    let distributions = vec![
        ("Uniform(4)", vec![0.25, 0.25, 0.25, 0.25]),
        ("Skewed", vec![0.7, 0.1, 0.1, 0.1]),
        ("Deterministic", vec![1.0, 0.0, 0.0, 0.0]),
        ("Binary", vec![0.5, 0.5]),
    ];

    for (name, dist) in &distributions {
        let shannon = ShannonEntropy::compute(dist);
        let min_ent = MinEntropy::compute(dist);
        let vuln = MinEntropy::vulnerability(dist);
        println!("  {:15} | H={:.3} | H_min={:.3} | V={:.3}",
            name, shannon, min_ent, vuln);
    }

    // 3. Mutual information
    println!("\n  --- Mutual Information ---");
    let joint_independent = vec![
        vec![0.25, 0.25],
        vec![0.25, 0.25],
    ];
    let joint_correlated = vec![
        vec![0.5, 0.0],
        vec![0.0, 0.5],
    ];
    println!("  Independent: MI = {:.3} bits", MutualInformation::from_joint(&joint_independent));
    println!("  Correlated:  MI = {:.3} bits", MutualInformation::from_joint(&joint_correlated));
}

/// Demonstrate the full information flow analyzer.
fn demo_flow_analyzer() {
    println!("\n=== Information Flow Analyzer ===\n");

    let mut analyzer = InformationFlowAnalyzer::new();

    // Add different types of channels
    analyzer.add_channel("timing",
        ChannelMatrix::binary_symmetric(0.15),
        "Timing side channel (branch-dependent)");

    analyzer.add_channel("cache",
        ChannelMatrix::deterministic(&[0, 0, 1, 1, 2, 2, 3, 3], 4),
        "Cache line side channel");

    let residue = InformationFlowAnalyzer::build_residue_channel(
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[vec![0, 1], vec![1, 0], vec![0, 1], vec![1, 1], vec![0, 0], vec![1, 1], vec![0, 1], vec![1, 0]],
    );
    analyzer.add_channel("residue", residue, "GPU memory residue channel");

    let timing = InformationFlowAnalyzer::build_timing_channel(
        &[0, 1, 2, 3],
        &[100, 200, 150, 250],
        4,
    );
    analyzer.add_channel("timing_model", timing, "Execution time variation");

    let traces: Vec<(usize, usize)> = (0..16).map(|i| (i % 4, i % 3)).collect();
    let empirical = InformationFlowAnalyzer::build_channel_from_traces(&traces, 4, 3);
    analyzer.add_channel("empirical", empirical, "Observed execution traces");

    // Analyze all channels
    let results = analyzer.analyze_all();
    for result in &results {
        println!("  Channel analyzed");
    }

    // Print summary
    let report = analyzer.summary_report();
    println!("{}", report);
}

/// Demonstrate timing side channel analysis.
fn demo_timing_analysis() {
    println!("\n=== Timing Side Channel Analysis ===\n");

    // 1. Execution time model
    let model = ExecutionTimeModel::new("aes_encrypt_kernel");
    println!("  Model: aes_encrypt_kernel");
    println!("  Estimated time (2 branches, 3 misses): {:.1}ns", model.estimate(2, 3));
    println!("  Max variation (5 branches, 10 misses): {:.1}ns", model.max_variation(5, 10));

    // 2. Timing variation analysis
    let secrets = vec![0, 1, 2, 3];
    let measurements = vec![
        vec![100.0, 101.0, 99.5, 100.2, 100.1],  // secret=0
        vec![150.0, 149.0, 151.0, 150.5, 149.5],  // secret=1
        vec![100.5, 100.0, 100.2, 99.8, 100.3],   // secret=2
        vec![200.0, 199.0, 201.0, 200.5, 199.5],  // secret=3
    ];
    let tv = TimingVariation::from_measurements(secrets, measurements);
    println!("\n  Timing variation analysis:");
    println!("    Global mean: {:.1}ns", tv.global_mean());
    println!("    Max mean difference: {:.1}ns", tv.max_mean_difference());
    println!("    Cross-secret CV: {:.4}", tv.cross_secret_cv());
    println!("    Significant (threshold=5.0): {}", tv.is_significant(5.0));
    println!("    Timing classes (ε=10.0): {}", tv.num_timing_classes(10.0));

    // 3. Differential timing
    let dt = DifferentialTiming::analyze(&tv);
    println!("\n  Differential timing:");
    println!("    Total pairs: {}", dt.total_pairs());
    println!("    Distinguishable: {}", dt.distinguishable_pairs());
    println!("    Max t-statistic: {:.2}", dt.max_t_statistic());
    println!("    Distinguishability ratio: {:.2}%", dt.distinguishability_ratio() * 100.0);

    // 4. Covert channel bandwidth
    let variation = TimingVariation::from_measurements(
        vec![0, 1, 2],
        vec![vec![100.0, 101.0, 99.5], vec![200.0, 199.0, 201.0], vec![150.0, 151.0, 149.5]],
    );
    let bw = CovertChannelBandwidth::estimate(&variation, 100.0);
    println!("\n  Covert channel bandwidth:");
    println!("    Viable: {}", bw.is_viable());

    // 5. Full timing report
    let mut detector = TimingChannelDetector::new();
    detector.add_model(ExecutionTimeModel::new("kernel_1"));
    detector.add_model(ExecutionTimeModel::new("kernel_2"));

    let secrets2 = vec![0, 1];
    let meas2 = vec![
        vec![100.0, 100.0, 100.0],
        vec![200.0, 200.0, 200.0],
    ];
    detector.add_variation(TimingVariation::from_measurements(secrets2, meas2));

    let report = detector.generate_report();
    println!("\n  Timing Report:");
    println!("    Severity: {}", report.severity());
    println!("{}", report.to_text());
}

/// Demonstrate vulnerability detection and reporting.
fn demo_vulnerability() {
    println!("\n=== Vulnerability Detection ===\n");

    let mut db = VulnerabilityDatabase::new();

    // Add various vulnerabilities
    let mut v1 = Vulnerability::new("LITMUS-2024-001", Severity::Critical,
        "Cross-CTA data race in shared memory barrier")
        .with_model("PTX")
        .with_category(VulnCategory::WeakMemoryOrder)
        .with_cvss(9.1);
    v1.add_evidence(Evidence::new("Weak outcome observed in SB test")
        .with_trace(vec![
            "Thread 0: st.cta [x], 1; ld.cta r0, [y]".into(),
            "Thread 1: st.cta [y], 1; ld.cta r0, [x]".into(),
            "Observed: r0=0, r1=0 (forbidden under SC)".into(),
        ])
        .with_data("test", "SB")
        .with_data("model", "PTX"));
    v1.add_mitigation("Insert membar.cta between store and load");
    v1.add_mitigation("Use release/acquire orderings");
    db.add(v1);

    let mut v2 = Vulnerability::new("LITMUS-2024-002", Severity::High,
        "Timing side channel in AES kernel")
        .with_category(VulnCategory::TimingSideChannel)
        .with_cvss(7.5);
    v2.add_evidence(Evidence::new("Secret-dependent timing variation of 50ns"));
    v2.add_mitigation("Use constant-time AES implementation");
    db.add(v2);

    let mut v3 = Vulnerability::new("LITMUS-2024-003", Severity::Medium,
        "Memory residue in local memory between kernels")
        .with_category(VulnCategory::InformationFlow)
        .with_cvss(5.0);
    v3.add_evidence(Evidence::new("AES key bytes readable in subsequent kernel"));
    v3.add_mitigation("Zero local memory before kernel exit");
    db.add(v3);

    let v4 = Vulnerability::new("LITMUS-2024-004", Severity::Low,
        "Non-deterministic memory initialization")
        .with_category(VulnCategory::InformationFlow);
    db.add(v4);

    // Reports
    println!("  Total vulnerabilities: {}", db.len());
    let counts = db.count_by_severity();
    for (sev, count) in &counts {
        println!("    {:10}: {}", sev.as_str(), count);
    }

    println!("\n  --- Text Report ---");
    println!("{}", db.to_text_report());

    // CVSS scoring
    let score = SeverityScore::gpu_side_channel();
    println!("  GPU side channel CVSS: {:.1}", score.compute());
}

/// Demonstrate GPU memory residue analysis.
fn demo_residue() {
    println!("\n=== GPU Memory Residue Analysis ===\n");

    // 1. Residue patterns
    let mut detector = ResidueChannelDetector::new();
    detector.add_pattern(
        ResiduePattern::new("R-001", MemoryRegion::SharedMemory, ResidueDataType::Keys)
            .with_size(256)
            .with_persistence(Persistence::AcrossKernels)
            .with_source("aes_encrypt")
            .with_description("AES-256 key in shared memory")
    );
    detector.add_pattern(
        ResiduePattern::new("R-002", MemoryRegion::LocalMemory, ResidueDataType::UserData)
            .with_size(64)
            .with_persistence(Persistence::UntilReallocation)
            .with_source("hash_password")
    );
    detector.add_pattern(
        ResiduePattern::new("R-003", MemoryRegion::GlobalMemory, ResidueDataType::RawBytes)
            .with_size(4096)
            .with_source("matrix_multiply")
    );

    println!("  Residue patterns: {}", detector.count());
    let report = detector.generate_report();
    println!("  Severity: {}", report.severity());
    println!("{}", report.to_text());

    // 2. Allocation patterns
    let mut alloc = AllocationPattern::new();
    alloc.add_allocation(0x10000, 1024, MemoryRegion::GlobalMemory, "kernel_a", 0, true);
    alloc.add_allocation(0x20000, 2048, MemoryRegion::GlobalMemory, "kernel_b", 1, false);
    alloc.add_allocation(0x30000, 512, MemoryRegion::GlobalMemory, "kernel_c", 2, true);
    alloc.add_deallocation(0, 10, true);    // wiped
    alloc.add_deallocation(1, 20, false);   // NOT wiped
    alloc.add_deallocation(2, 30, false);   // NOT wiped

    println!("  Allocation analysis:");
    println!("    Total allocated: {} bytes", alloc.total_allocated());
    println!("    Zeroed count: {}", alloc.zeroed_count());
    println!("    Deallocation rate: {:.0}%", alloc.deallocation_rate() * 100.0);
    println!("    Unwiped: {}", alloc.unwiped_deallocations().len());

    // 3. Leftover locals
    let mut d = LeftoverLocalsDetector::new();
    let mut k1 = KernelProfile::new("crypto_kernel");
    k1.add_local_write(0, 0, 32, true);   // write sensitive data
    k1.add_local_write(0, 32, 16, true);
    let mut k2 = KernelProfile::new("next_kernel");
    k2.add_local_read(0, 0, 32, false);   // read previous data!
    d.analyze(&k1, &k2);
    println!("\n  Leftover locals: {}", d.count());
    println!("  Max severity: {:.2}", d.max_severity());

    // 4. Cache residue
    let cache = CacheResidueAnalyzer::default_gpu_l1();
    let victim_accesses = vec![0x1000, 0x1040, 0x1080, 0x2000, 0x3000];
    let attacker_probes = vec![0x1000, 0x1040, 0x4000, 0x5000];
    let result = cache.analyze_accesses(&victim_accesses, &attacker_probes);
    println!("\n  Cache residue analysis:");
    println!("    Bandwidth estimate: {:.1} bits/s", cache.estimate_bandwidth(10.0, 15.0));
}

/// Demonstrate formal security verification.
fn demo_formal_security() {
    println!("\n=== Formal Security Verification ===\n");

    // 1. Security lattice
    let lattice = SecurityLattice::four_level();
    println!("  Security lattice: {} levels", lattice.size());

    // 2. Information flow checking
    let mut checker = InformationFlowChecker::new();
    checker.set_type("secret_key", InformationFlowType::new(SecurityLevel::High)
        .with_dependency("crypto_module"));
    checker.set_type("public_input", InformationFlowType::new(SecurityLevel::Low));
    checker.set_type("output", InformationFlowType::new(SecurityLevel::Low));
    checker.set_type("temp", InformationFlowType::new(SecurityLevel::Low));

    // Check various assignments
    println!("  Information flow checks:");
    println!("    public_input -> output: {}", checker.check_assignment("public_input", "output"));
    println!("    secret_key -> output:   {}", checker.check_assignment("secret_key", "output"));
    println!("    public_input -> temp:   {}", checker.check_assignment("public_input", "temp"));
    println!("    secret_key -> temp:     {}", checker.check_assignment("secret_key", "temp"));

    if checker.has_violations() {
        println!("\n  Violations detected: {}", checker.get_violations().len());
        for v in checker.get_violations() {
            println!("    - {:?}", v);
        }
    }

    // 3. Security automaton
    println!("\n  Security automaton:");
    let mut automaton = SecurityAutomaton::no_unauthorized_declassification();
    let events = vec![
        SecurityEvent::new("read", SecurityLevel::Low),
        SecurityEvent::new("write", SecurityLevel::Low),
        SecurityEvent::new("read", SecurityLevel::High),
    ];
    for event in &events {
        let result = automaton.process(event);
        println!("    Process {:?} -> {:?}", event, result);
    }
    println!("    In error state: {}", automaton.is_in_error());

    // 4. Declassification policy
    let policy = DeclassificationPolicy::password_hash_policy();
    println!("\n  Declassification policy:");
    println!("    Declassifiable: {:?}", policy.declassifiable_variables());
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║      LITMUS∞ — Security Analysis Demo                   ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    demo_qif();
    demo_flow_analyzer();
    demo_timing_analysis();
    demo_vulnerability();
    demo_residue();
    demo_formal_security();

    println!("\n✅ Security analysis demo completed successfully.");
}
