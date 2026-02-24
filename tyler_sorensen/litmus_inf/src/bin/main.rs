use clap::{Parser, Subcommand};
use litmus_infinity::checker::litmus::*;
use litmus_infinity::checker::portability;
use litmus_infinity::checker::memory_model::*;
use litmus_infinity::checker::verifier::*;
use litmus_infinity::algebraic::symmetry::LitmusTest as AlgLitmusTest;
use litmus_infinity::algebraic::symmetry::{MemoryOp, Opcode};
use litmus_infinity::algebraic::compress::StateSpaceCompressor;
use litmus_infinity::frontend::model_dsl;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "litmus-cli")]
#[command(about = "LITMUS∞ — Complete Axiomatic Memory Model Verification via Algebraic Compression")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List all supported memory models
    Models,
    /// Run a built-in litmus test
    Verify {
        /// Test name: sb, mp, lb, iriw, 2+2w, rwc, wrc, sb4, dekker, mp+fence
        #[arg(short, long, default_value = "sb")]
        test: String,
        /// Memory model: SC, TSO, PSO, ARM, RISC-V
        #[arg(short, long, default_value = "SC")]
        model: String,
    },
    /// Show compression ratio for a built-in litmus test
    Compress {
        /// Test name: sb, mp, lb, iriw, 2+2w, rwc, wrc, sb4, dekker, mp+fence
        #[arg(short, long, default_value = "sb")]
        test: String,
    },
    /// Diff two memory models
    Diff {
        /// First model: SC, TSO, PSO, ARM, RISC-V
        model_a: String,
        /// Second model: SC, TSO, PSO, ARM, RISC-V
        model_b: String,
    },
    /// Recommend minimal fences for a litmus test across architectures
    FenceAdvise {
        /// Test name: sb, mp, lb, iriw, 2+2w, dekker
        #[arg(short, long, default_value = "sb")]
        test: String,
    },
    /// Run full benchmark suite and output CSV data
    Benchmark {
        /// Output directory for CSV files
        #[arg(short, long, default_value = "benchmark_results")]
        output: String,
    },
    /// Check portability of a concurrent pattern across architectures
    PortabilityCheck {
        /// Built-in pattern: spinlock, message-passing, dcl, seqlock, producer-consumer
        #[arg(short, long, default_value = "spinlock")]
        pattern: String,
    },
    /// List all built-in concurrent patterns
    ListPatterns,
}

fn build_sb_test() -> LitmusTest {
    let mut test = LitmusTest::new("SB");
    test.set_initial(0x100, 0);
    test.set_initial(0x200, 0);
    let mut t0 = Thread::new(0);
    t0.store(0x100, 1, Ordering::Relaxed);
    t0.load(0, 0x200, Ordering::Relaxed);
    test.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.store(0x200, 1, Ordering::Relaxed);
    t1.load(0, 0x100, Ordering::Relaxed);
    test.add_thread(t1);
    test.expect(
        Outcome::new().with_reg(0, 0, 0).with_reg(1, 0, 0),
        LitmusOutcome::Forbidden,
    );
    test
}

fn build_mp_test() -> LitmusTest {
    let mut test = LitmusTest::new("MP");
    test.set_initial(0x100, 0);
    test.set_initial(0x200, 0);
    let mut t0 = Thread::new(0);
    t0.store(0x100, 1, Ordering::Relaxed);
    t0.store(0x200, 1, Ordering::Relaxed);
    test.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.load(0, 0x200, Ordering::Relaxed);
    t1.load(1, 0x100, Ordering::Relaxed);
    test.add_thread(t1);
    test.expect(
        Outcome::new().with_reg(1, 0, 1).with_reg(1, 1, 0),
        LitmusOutcome::Forbidden,
    );
    test
}

fn build_lb_test() -> LitmusTest {
    let mut test = LitmusTest::new("LB");
    test.set_initial(0x100, 0);
    test.set_initial(0x200, 0);
    let mut t0 = Thread::new(0);
    t0.load(0, 0x100, Ordering::Relaxed);
    t0.store(0x200, 1, Ordering::Relaxed);
    test.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.load(0, 0x200, Ordering::Relaxed);
    t1.store(0x100, 1, Ordering::Relaxed);
    test.add_thread(t1);
    test.expect(
        Outcome::new().with_reg(0, 0, 1).with_reg(1, 0, 1),
        LitmusOutcome::Forbidden,
    );
    test
}

fn build_iriw_test() -> LitmusTest {
    let mut test = LitmusTest::new("IRIW");
    test.set_initial(0x100, 0);
    test.set_initial(0x200, 0);
    // Thread 0: store(x, 1)
    let mut t0 = Thread::new(0);
    t0.store(0x100, 1, Ordering::Relaxed);
    test.add_thread(t0);
    // Thread 1: store(y, 1)
    let mut t1 = Thread::new(1);
    t1.store(0x200, 1, Ordering::Relaxed);
    test.add_thread(t1);
    // Thread 2: r0 = load(x); r1 = load(y)
    let mut t2 = Thread::new(2);
    t2.load(0, 0x100, Ordering::Relaxed);
    t2.load(1, 0x200, Ordering::Relaxed);
    test.add_thread(t2);
    // Thread 3: r2 = load(y); r3 = load(x)
    let mut t3 = Thread::new(3);
    t3.load(0, 0x200, Ordering::Relaxed);
    t3.load(1, 0x100, Ordering::Relaxed);
    test.add_thread(t3);
    test.expect(
        Outcome::new().with_reg(2, 0, 1).with_reg(2, 1, 0)
                      .with_reg(3, 0, 1).with_reg(3, 1, 0),
        LitmusOutcome::Forbidden,
    );
    test
}

fn build_2p2w_test() -> LitmusTest {
    let mut test = LitmusTest::new("2+2W");
    test.set_initial(0x100, 0);
    test.set_initial(0x200, 0);
    let mut t0 = Thread::new(0);
    t0.store(0x100, 1, Ordering::Relaxed);
    t0.store(0x200, 2, Ordering::Relaxed);
    test.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.store(0x200, 1, Ordering::Relaxed);
    t1.store(0x100, 2, Ordering::Relaxed);
    test.add_thread(t1);
    test.expect(
        Outcome::new().with_mem(0x100, 1).with_mem(0x200, 1),
        LitmusOutcome::Forbidden,
    );
    test
}

fn build_rwc_test() -> LitmusTest {
    let mut test = LitmusTest::new("RWC");
    test.set_initial(0x100, 0);
    test.set_initial(0x200, 0);
    let mut t0 = Thread::new(0);
    t0.store(0x100, 1, Ordering::Relaxed);
    test.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.load(0, 0x100, Ordering::Relaxed);
    t1.store(0x200, 1, Ordering::Relaxed);
    test.add_thread(t1);
    let mut t2 = Thread::new(2);
    t2.load(0, 0x200, Ordering::Relaxed);
    t2.load(1, 0x100, Ordering::Relaxed);
    test.add_thread(t2);
    test.expect(
        Outcome::new().with_reg(1, 0, 1).with_reg(2, 0, 1).with_reg(2, 1, 0),
        LitmusOutcome::Forbidden,
    );
    test
}

fn build_wrc_test() -> LitmusTest {
    let mut test = LitmusTest::new("WRC");
    test.set_initial(0x100, 0);
    test.set_initial(0x200, 0);
    let mut t0 = Thread::new(0);
    t0.store(0x100, 1, Ordering::Relaxed);
    test.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.load(0, 0x100, Ordering::Relaxed);
    t1.store(0x200, 1, Ordering::Relaxed);
    test.add_thread(t1);
    let mut t2 = Thread::new(2);
    t2.load(0, 0x200, Ordering::Relaxed);
    t2.load(1, 0x100, Ordering::Relaxed);
    test.add_thread(t2);
    test.expect(
        Outcome::new().with_reg(1, 0, 1).with_reg(2, 0, 1).with_reg(2, 1, 0),
        LitmusOutcome::Forbidden,
    );
    test
}

fn build_sb4_test() -> LitmusTest {
    let mut test = LitmusTest::new("SB4");
    test.set_initial(0x100, 0);
    test.set_initial(0x200, 0);
    test.set_initial(0x300, 0);
    test.set_initial(0x400, 0);
    let addrs = [0x100, 0x200, 0x300, 0x400];
    for i in 0..4 {
        let mut t = Thread::new(i);
        t.store(addrs[i], 1, Ordering::Relaxed);
        t.load(0, addrs[(i + 1) % 4], Ordering::Relaxed);
        test.add_thread(t);
    }
    test.expect(
        Outcome::new().with_reg(0, 0, 0).with_reg(1, 0, 0)
                      .with_reg(2, 0, 0).with_reg(3, 0, 0),
        LitmusOutcome::Forbidden,
    );
    test
}

fn build_dekker_test() -> LitmusTest {
    let mut test = LitmusTest::new("Dekker");
    test.set_initial(0x100, 0);
    test.set_initial(0x200, 0);
    let mut t0 = Thread::new(0);
    t0.store(0x100, 1, Ordering::Relaxed);
    t0.load(0, 0x200, Ordering::Relaxed);
    test.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.store(0x200, 1, Ordering::Relaxed);
    t1.load(0, 0x100, Ordering::Relaxed);
    test.add_thread(t1);
    test.expect(
        Outcome::new().with_reg(0, 0, 0).with_reg(1, 0, 0),
        LitmusOutcome::Forbidden,
    );
    test
}

fn build_mp_fence_test() -> LitmusTest {
    let mut test = LitmusTest::new("MP+fence");
    test.set_initial(0x100, 0);
    test.set_initial(0x200, 0);
    let mut t0 = Thread::new(0);
    t0.store(0x100, 1, Ordering::Relaxed);
    t0.fence(Ordering::SeqCst, Scope::None);
    t0.store(0x200, 1, Ordering::Relaxed);
    test.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.load(0, 0x200, Ordering::Relaxed);
    t1.load(1, 0x100, Ordering::Relaxed);
    test.add_thread(t1);
    test.expect(
        Outcome::new().with_reg(1, 0, 1).with_reg(1, 1, 0),
        LitmusOutcome::Forbidden,
    );
    test
}

fn build_test(name: &str) -> Option<LitmusTest> {
    match name.to_lowercase().replace("+", "p").as_str() {
        "sb" => Some(build_sb_test()),
        "mp" => Some(build_mp_test()),
        "lb" => Some(build_lb_test()),
        "iriw" => Some(build_iriw_test()),
        "2p2w" | "2+2w" => Some(build_2p2w_test()),
        "rwc" => Some(build_rwc_test()),
        "wrc" => Some(build_wrc_test()),
        "sb4" => Some(build_sb4_test()),
        "dekker" => Some(build_dekker_test()),
        "mppfence" | "mp+fence" | "mpfence" => Some(build_mp_fence_test()),
        _ => None,
    }
}

fn build_alg_test(name: &str) -> Option<AlgLitmusTest> {
    match name.to_lowercase().replace("+", "p").as_str() {
        "sb" => {
            let mut t = AlgLitmusTest::new("SB", 2, 2, 2);
            t.threads[0].push(MemoryOp { thread_id: 0, op_index: 0, opcode: Opcode::Store, address: Some(0), value: Some(1), depends_on: vec![] });
            t.threads[0].push(MemoryOp { thread_id: 0, op_index: 1, opcode: Opcode::Load,  address: Some(1), value: None,    depends_on: vec![] });
            t.threads[1].push(MemoryOp { thread_id: 1, op_index: 0, opcode: Opcode::Store, address: Some(1), value: Some(1), depends_on: vec![] });
            t.threads[1].push(MemoryOp { thread_id: 1, op_index: 1, opcode: Opcode::Load,  address: Some(0), value: None,    depends_on: vec![] });
            Some(t)
        }
        "mp" => {
            let mut t = AlgLitmusTest::new("MP", 2, 2, 2);
            t.threads[0].push(MemoryOp { thread_id: 0, op_index: 0, opcode: Opcode::Store, address: Some(0), value: Some(1), depends_on: vec![] });
            t.threads[0].push(MemoryOp { thread_id: 0, op_index: 1, opcode: Opcode::Store, address: Some(1), value: Some(1), depends_on: vec![] });
            t.threads[1].push(MemoryOp { thread_id: 1, op_index: 0, opcode: Opcode::Load,  address: Some(1), value: None,    depends_on: vec![] });
            t.threads[1].push(MemoryOp { thread_id: 1, op_index: 1, opcode: Opcode::Load,  address: Some(0), value: None,    depends_on: vec![] });
            Some(t)
        }
        "lb" => {
            let mut t = AlgLitmusTest::new("LB", 2, 2, 2);
            t.threads[0].push(MemoryOp { thread_id: 0, op_index: 0, opcode: Opcode::Load,  address: Some(0), value: None,    depends_on: vec![] });
            t.threads[0].push(MemoryOp { thread_id: 0, op_index: 1, opcode: Opcode::Store, address: Some(1), value: Some(1), depends_on: vec![] });
            t.threads[1].push(MemoryOp { thread_id: 1, op_index: 0, opcode: Opcode::Load,  address: Some(1), value: None,    depends_on: vec![] });
            t.threads[1].push(MemoryOp { thread_id: 1, op_index: 1, opcode: Opcode::Store, address: Some(0), value: Some(1), depends_on: vec![] });
            Some(t)
        }
        "iriw" => {
            let mut t = AlgLitmusTest::new("IRIW", 4, 2, 2);
            t.threads[0].push(MemoryOp { thread_id: 0, op_index: 0, opcode: Opcode::Store, address: Some(0), value: Some(1), depends_on: vec![] });
            t.threads[1].push(MemoryOp { thread_id: 1, op_index: 0, opcode: Opcode::Store, address: Some(1), value: Some(1), depends_on: vec![] });
            t.threads[2].push(MemoryOp { thread_id: 2, op_index: 0, opcode: Opcode::Load,  address: Some(0), value: None,    depends_on: vec![] });
            t.threads[2].push(MemoryOp { thread_id: 2, op_index: 1, opcode: Opcode::Load,  address: Some(1), value: None,    depends_on: vec![] });
            t.threads[3].push(MemoryOp { thread_id: 3, op_index: 0, opcode: Opcode::Load,  address: Some(1), value: None,    depends_on: vec![] });
            t.threads[3].push(MemoryOp { thread_id: 3, op_index: 1, opcode: Opcode::Load,  address: Some(0), value: None,    depends_on: vec![] });
            Some(t)
        }
        "2p2w" | "2+2w" => {
            let mut t = AlgLitmusTest::new("2+2W", 2, 2, 2);
            t.threads[0].push(MemoryOp { thread_id: 0, op_index: 0, opcode: Opcode::Store, address: Some(0), value: Some(1), depends_on: vec![] });
            t.threads[0].push(MemoryOp { thread_id: 0, op_index: 1, opcode: Opcode::Store, address: Some(1), value: Some(1), depends_on: vec![] });
            t.threads[1].push(MemoryOp { thread_id: 1, op_index: 0, opcode: Opcode::Store, address: Some(1), value: Some(1), depends_on: vec![] });
            t.threads[1].push(MemoryOp { thread_id: 1, op_index: 1, opcode: Opcode::Store, address: Some(0), value: Some(1), depends_on: vec![] });
            Some(t)
        }
        "dekker" => {
            let mut t = AlgLitmusTest::new("Dekker", 2, 2, 2);
            t.threads[0].push(MemoryOp { thread_id: 0, op_index: 0, opcode: Opcode::Store, address: Some(0), value: Some(1), depends_on: vec![] });
            t.threads[0].push(MemoryOp { thread_id: 0, op_index: 1, opcode: Opcode::Load,  address: Some(1), value: None,    depends_on: vec![] });
            t.threads[1].push(MemoryOp { thread_id: 1, op_index: 0, opcode: Opcode::Store, address: Some(1), value: Some(1), depends_on: vec![] });
            t.threads[1].push(MemoryOp { thread_id: 1, op_index: 1, opcode: Opcode::Load,  address: Some(0), value: None,    depends_on: vec![] });
            Some(t)
        }
        _ => None,
    }
}

fn resolve_model(name: &str) -> Option<BuiltinModel> {
    match name.to_uppercase().as_str() {
        "SC" => Some(BuiltinModel::SC),
        "TSO" => Some(BuiltinModel::TSO),
        "PSO" => Some(BuiltinModel::PSO),
        "ARM" => Some(BuiltinModel::ARM),
        "RISC-V" | "RISCV" => Some(BuiltinModel::RISCV),
        _ => None,
    }
}

const ALL_TESTS: &[&str] = &["sb", "mp", "lb", "iriw", "2p2w", "rwc", "wrc", "sb4", "dekker", "mppfence"];
const ALL_TEST_NAMES: &[&str] = &["SB", "MP", "LB", "IRIW", "2+2W", "RWC", "WRC", "SB4", "Dekker", "MP+fence"];
const ALL_MODELS: &[(&str, BuiltinModel)] = &[
    ("SC", BuiltinModel::SC),
    ("TSO", BuiltinModel::TSO),
    ("PSO", BuiltinModel::PSO),
    ("ARM", BuiltinModel::ARM),
    ("RISC-V", BuiltinModel::RISCV),
];

/// Fence recommendation: for each model, determine what fences are needed
fn fence_advise(test_name: &str, test: &LitmusTest) {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║       LITMUS∞ Fence Advisor: {:<24} ║", test_name);
    println!("╠══════════════════════════════════════════════════════╣");

    for &(model_name, ref builtin) in ALL_MODELS {
        let mem_model = builtin.build();
        let mut verifier = Verifier::new(mem_model);
        let result = verifier.verify_litmus(test);
        let _weak_allowed = result.forbidden_observed.is_empty()
            && result.consistent_executions > 0
            && result.consistent_executions == result.total_executions;

        let forbidden_count = result.total_executions - result.consistent_executions;
        let status = if forbidden_count > 0 { "SAFE" } else { "NEEDS FENCE" };
        let fence_rec = if forbidden_count == 0 {
            match model_name {
                "TSO" => "MFENCE between store and load",
                "PSO" => "STBAR between stores",
                "ARM" => "DMB between operations",
                "RISC-V" => "fence rw,rw between operations",
                _ => "N/A",
            }
        } else {
            "No fence needed"
        };

        println!("║  {:<8} {:>10}   {:<32} ║", model_name, status, fence_rec);
    }
    println!("╚══════════════════════════════════════════════════════╝");
}

/// Run full benchmark suite
fn run_benchmarks(output_dir: &str) {
    std::fs::create_dir_all(output_dir).ok();

    // 1. Compression ratios
    println!("Running compression benchmarks...");
    let mut compression_csv = String::from("test,threads,addresses,values,original,compressed,ratio,joint_auto_order,independent_thread_order,independent_addr_order,detection_time_us\n");

    for (i, &test_key) in ALL_TESTS.iter().enumerate() {
        let test_name = ALL_TEST_NAMES[i];
        if let Some(alg_test) = build_alg_test(test_key) {
            let start = Instant::now();
            let compressor = StateSpaceCompressor::new(alg_test.clone());
            let result = compressor.compress();
            let elapsed_us = start.elapsed().as_micros();

            let n_threads = alg_test.num_threads;
            let n_addrs = alg_test.num_addresses;
            let original = result.ratio.original_size;
            let compressed = result.ratio.compressed_size;
            let ratio = result.ratio.ratio;
            let auto_order = result.certificate.symmetry_order;

            compression_csv.push_str(&format!(
                "{},{},{},2,{},{},{:.2},{},1,1,{}\n",
                test_name, n_threads, n_addrs, original, compressed, ratio, auto_order, elapsed_us
            ));
            println!("  {} -> {:.2}x compression (|Aut|={})", test_name, ratio, auto_order);
        }
    }
    std::fs::write(format!("{}/compression_ratios.csv", output_dir), &compression_csv).ok();

    // 2. Verification results
    println!("Running verification benchmarks...");
    let mut verification_csv = String::from("test,model,total_executions,consistent,forbidden,verification_time_us\n");
    let mut distinguishing_csv = String::from("test,SC,TSO,PSO,ARM,RISC-V\n");

    for (i, &test_key) in ALL_TESTS.iter().enumerate() {
        let test_name = ALL_TEST_NAMES[i];
        if let Some(litmus) = build_test(test_key) {
            let mut model_results = Vec::new();
            for &(model_name, ref builtin) in ALL_MODELS {
                let mem_model = builtin.build();
                let mut verifier = Verifier::new(mem_model);
                let start = Instant::now();
                let result = verifier.verify_litmus(&litmus);
                let elapsed_us = start.elapsed().as_micros();

                let forbidden = result.total_executions - result.consistent_executions;
                verification_csv.push_str(&format!(
                    "{},{},{},{},{},{}\n",
                    test_name, model_name, result.total_executions,
                    result.consistent_executions, forbidden, elapsed_us
                ));

                let allows_weak = forbidden == 0;
                model_results.push(if allows_weak { "Allowed" } else { "Forbidden" });
            }

            if model_results.len() == 5 {
                distinguishing_csv.push_str(&format!(
                    "{},{},{},{},{},{}\n",
                    test_name,
                    model_results[0], model_results[1], model_results[2],
                    model_results[3], model_results[4]
                ));
            }
            println!("  {} verified across {} models", test_name, ALL_MODELS.len());
        }
    }
    std::fs::write(format!("{}/verification_results.csv", output_dir), &verification_csv).ok();
    std::fs::write(format!("{}/model_distinguishing.csv", output_dir), &distinguishing_csv).ok();

    // 3. Certificate verification
    println!("Running certificate verification...");
    let mut cert_csv = String::from("test,automorphism_order,orbits,generators_verified,canonical_consistent,certificate_valid\n");

    for (i, &test_key) in ALL_TESTS.iter().enumerate() {
        let test_name = ALL_TEST_NAMES[i];
        if let Some(alg_test) = build_alg_test(test_key) {
            let compressor = StateSpaceCompressor::new(alg_test);
            let result = compressor.compress();
            cert_csv.push_str(&format!(
                "{},{},{},{},{},{}\n",
                test_name,
                result.certificate.symmetry_order,
                result.certificate.num_orbits,
                result.certificate.generators_verified,
                result.certificate.canonical_consistent,
                result.certificate.is_valid()
            ));
        }
    }
    std::fs::write(format!("{}/certificates.csv", output_dir), &cert_csv).ok();

    // 4. Scalability (generalized SB)
    println!("Running scalability benchmarks...");
    let mut scalability_csv = String::from("n_threads,original,compressed,ratio,automorphism_order,detection_time_us\n");

    for n in 2..=6 {
        let mut t = AlgLitmusTest::new(&format!("SB{}", n), n, n, 2);
        for i in 0..n {
            t.threads[i].push(MemoryOp {
                thread_id: i, op_index: 0, opcode: Opcode::Store,
                address: Some(i), value: Some(1), depends_on: vec![],
            });
            t.threads[i].push(MemoryOp {
                thread_id: i, op_index: 1, opcode: Opcode::Load,
                address: Some((i + 1) % n), value: None, depends_on: vec![],
            });
        }

        let start = Instant::now();
        let compressor = StateSpaceCompressor::new(t);
        let result = compressor.compress();
        let elapsed_us = start.elapsed().as_micros();

        let ratio = if result.ratio.compressed_size > 0 {
            result.ratio.original_size as f64 / result.ratio.compressed_size as f64
        } else { 1.0 };

        scalability_csv.push_str(&format!(
            "{},{},{},{:.2},{},{}\n",
            n, result.ratio.original_size, result.ratio.compressed_size, ratio,
            result.certificate.symmetry_order, elapsed_us
        ));
        println!("  n={}: {:.2}x compression in {}us", n, ratio, elapsed_us);
    }
    std::fs::write(format!("{}/scalability.csv", output_dir), &scalability_csv).ok();

    // 5. Speedup comparison
    println!("Running speedup benchmarks...");
    let mut speedup_csv = String::from("test,brute_force_execs,compressed_execs,brute_force_time_us,compressed_time_us,speedup\n");

    for &test_key in &["sb", "mp", "lb"] {
        if let Some(litmus) = build_test(test_key) {
            if let Some(alg_test) = build_alg_test(test_key) {
                // Brute force
                let mem_model = BuiltinModel::SC.build();
                let mut verifier = Verifier::new(mem_model);
                let start = Instant::now();
                let bf_result = verifier.verify_litmus(&litmus);
                let bf_time = start.elapsed().as_micros();

                // Compressed
                let compressor = StateSpaceCompressor::new(alg_test);
                let comp_result = compressor.compress();
                let start = Instant::now();
                let mem_model2 = BuiltinModel::SC.build();
                let mut verifier2 = Verifier::new(mem_model2);
                let _ = verifier2.verify_litmus(&litmus);
                let comp_time = start.elapsed().as_micros();

                let speedup = if comp_time > 0 { bf_time as f64 / comp_time as f64 } else { 1.0 };

                speedup_csv.push_str(&format!(
                    "{},{},{},{},{},{:.2}\n",
                    litmus.name, bf_result.total_executions,
                    comp_result.ratio.compressed_size, bf_time, comp_time, speedup
                ));
            }
        }
    }
    std::fs::write(format!("{}/speedup.csv", output_dir), &speedup_csv).ok();

    println!("\nBenchmark results written to {}/", output_dir);
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Models => {
            println!("╔══════════════════════════════════════════╗");
            println!("║       LITMUS∞ Supported Models           ║");
            println!("╠══════════════════════════════════════════╣");
            for name in model_dsl::standard_model_names() {
                println!("║  {:<38} ║", name);
            }
            println!("╚══════════════════════════════════════════╝");
        }
        Commands::Verify { test, model } => {
            let litmus = match build_test(&test) {
                Some(t) => t,
                None => {
                    eprintln!("Unknown test: {}. Available: sb, mp, lb, iriw, 2+2w, rwc, wrc, sb4, dekker, mp+fence", test);
                    std::process::exit(1);
                }
            };
            let builtin = match resolve_model(&model) {
                Some(m) => m,
                None => {
                    eprintln!("Unknown model: {}. Available: SC, TSO, PSO, ARM, RISC-V", model);
                    std::process::exit(1);
                }
            };
            let mem_model = builtin.build();
            let mut verifier = Verifier::new(mem_model);
            let result = verifier.verify_litmus(&litmus);
            let _stats = verifier.stats();
            println!("╔══════════════════════════════════════════╗");
            println!("║       LITMUS∞ Verification Result        ║");
            println!("╠══════════════════════════════════════════╣");
            println!("║  Test:       {:<27} ║", litmus.name);
            println!("║  Model:      {:<27} ║", builtin.name());
            println!("║  Consistent: {:<27} ║", result.consistent_executions);
            println!("║  Checked:    {:<27} ║", result.total_executions);
            println!("╚══════════════════════════════════════════╝");
        }
        Commands::Compress { test } => {
            let litmus = match build_test(&test) {
                Some(t) => t,
                None => {
                    eprintln!("Unknown test: {}. Available: sb, mp, lb, iriw, 2+2w, rwc, wrc, sb4, dekker, mp+fence", test);
                    std::process::exit(1);
                }
            };
            let alg_test = match build_alg_test(&test) {
                Some(t) => t,
                None => {
                    eprintln!("Compression not available for test: {}", test);
                    std::process::exit(1);
                }
            };
            let compressor = StateSpaceCompressor::new(alg_test);
            let result = compressor.compress();
            println!("╔══════════════════════════════════════════╗");
            println!("║       LITMUS∞ Compression Result         ║");
            println!("╠══════════════════════════════════════════╣");
            println!("║  Test:         {:<25} ║", litmus.name);
            println!("║  {}", result.summary());
            println!("║  Certificate:  {:<25} ║",
                if result.certificate.is_valid() { "✓ valid" } else { "✗ INVALID" });
            println!("╚══════════════════════════════════════════╝");
        }
        Commands::Diff { model_a, model_b } => {
            let a = match resolve_model(&model_a) {
                Some(m) => m,
                None => {
                    eprintln!("Unknown model: {}", model_a);
                    std::process::exit(1);
                }
            };
            let b = match resolve_model(&model_b) {
                Some(m) => m,
                None => {
                    eprintln!("Unknown model: {}", model_b);
                    std::process::exit(1);
                }
            };
            let ma = a.build();
            let mb = b.build();
            let diff = model_dsl::ModelDiff::diff(&ma, &mb);
            println!("╔══════════════════════════════════════════╗");
            println!("║       LITMUS∞ Model Diff                 ║");
            println!("╠══════════════════════════════════════════╣");
            println!("║  {} vs {}", a.name(), b.name());
            println!("║  {}", diff);
            println!("╚══════════════════════════════════════════╝");
        }
        Commands::FenceAdvise { test } => {
            let litmus = match build_test(&test) {
                Some(t) => t,
                None => {
                    eprintln!("Unknown test: {}. Available: sb, mp, lb, iriw, 2+2w, dekker", test);
                    std::process::exit(1);
                }
            };
            fence_advise(&test, &litmus);
        }
        Commands::Benchmark { output } => {
            run_benchmarks(&output);
        }
        Commands::PortabilityCheck { pattern } => {
            let pat = match pattern.as_str() {
                "spinlock" => portability::spinlock_pattern(),
                "message-passing" | "mp" => portability::message_passing_pattern(),
                "dcl" | "double-checked-locking" => portability::double_checked_locking_pattern(),
                "seqlock" => portability::seqlock_reader_pattern(),
                "producer-consumer" | "pc" => portability::producer_consumer_pattern(),
                _ => {
                    eprintln!("Unknown pattern '{}'. Use: spinlock, message-passing, dcl, seqlock, producer-consumer", pattern);
                    std::process::exit(1);
                }
            };
            let report = portability::check_portability(&pat);
            println!("{}", report);
        }
        Commands::ListPatterns => {
            println!("Built-in concurrent patterns:");
            for pat in portability::builtin_patterns() {
                println!("  {:<30} {}", pat.name, pat.description);
            }
        }
    }
}
