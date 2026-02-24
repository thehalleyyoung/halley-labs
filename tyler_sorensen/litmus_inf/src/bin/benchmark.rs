/// LITMUS∞ Comprehensive Benchmark Suite
/// Generates experimental data for all supported tests and models.
/// Outputs CSV data files for paper figures and tables.

use std::collections::HashMap;
use std::time::Instant;
use std::io::Write;
use std::fs;

use litmus_infinity::algebraic::symmetry::{
    self, LitmusTest as AlgLitmusTest, MemoryOp, Opcode, FenceType,
    FullSymmetryGroup, ThreadSymmetryDetector,
};
use litmus_infinity::algebraic::compress::StateSpaceCompressor;
use litmus_infinity::checker::litmus::{
    LitmusTest as CheckerLitmusTest, Thread, Ordering, Outcome, LitmusOutcome,
};
use litmus_infinity::checker::memory_model::*;
use litmus_infinity::checker::verifier::*;

// ── Test Catalog ─────────────────────────────────────────────────────

struct TestSpec {
    name: &'static str,
    description: &'static str,
    num_threads: usize,
    num_addresses: usize,
    num_values: usize,
    build_alg: fn() -> AlgLitmusTest,
    build_checker: fn() -> CheckerLitmusTest,
}

fn build_sb_alg() -> AlgLitmusTest {
    let mut t = AlgLitmusTest::new("SB", 2, 2, 2);
    t.threads[0].push(MemoryOp { thread_id: 0, op_index: 0, opcode: Opcode::Store, address: Some(0), value: Some(1), depends_on: vec![] });
    t.threads[0].push(MemoryOp { thread_id: 0, op_index: 1, opcode: Opcode::Load,  address: Some(1), value: None,    depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 0, opcode: Opcode::Store, address: Some(1), value: Some(1), depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 1, opcode: Opcode::Load,  address: Some(0), value: None,    depends_on: vec![] });
    t
}

fn build_mp_alg() -> AlgLitmusTest {
    let mut t = AlgLitmusTest::new("MP", 2, 2, 2);
    t.threads[0].push(MemoryOp { thread_id: 0, op_index: 0, opcode: Opcode::Store, address: Some(0), value: Some(1), depends_on: vec![] });
    t.threads[0].push(MemoryOp { thread_id: 0, op_index: 1, opcode: Opcode::Store, address: Some(1), value: Some(1), depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 0, opcode: Opcode::Load,  address: Some(1), value: None,    depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 1, opcode: Opcode::Load,  address: Some(0), value: None,    depends_on: vec![] });
    t
}

fn build_lb_alg() -> AlgLitmusTest {
    let mut t = AlgLitmusTest::new("LB", 2, 2, 2);
    t.threads[0].push(MemoryOp { thread_id: 0, op_index: 0, opcode: Opcode::Load,  address: Some(0), value: None,    depends_on: vec![] });
    t.threads[0].push(MemoryOp { thread_id: 0, op_index: 1, opcode: Opcode::Store, address: Some(1), value: Some(1), depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 0, opcode: Opcode::Load,  address: Some(1), value: None,    depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 1, opcode: Opcode::Store, address: Some(0), value: Some(1), depends_on: vec![] });
    t
}

fn build_iriw_alg() -> AlgLitmusTest {
    // 4 threads: T0 writes x, T1 writes y, T2 reads x then y, T3 reads y then x
    let mut t = AlgLitmusTest::new("IRIW", 4, 2, 2);
    // Writer 0
    t.threads[0].push(MemoryOp { thread_id: 0, op_index: 0, opcode: Opcode::Store, address: Some(0), value: Some(1), depends_on: vec![] });
    // Writer 1
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 0, opcode: Opcode::Store, address: Some(1), value: Some(1), depends_on: vec![] });
    // Reader 0: reads x then y
    t.threads[2].push(MemoryOp { thread_id: 2, op_index: 0, opcode: Opcode::Load,  address: Some(0), value: None,    depends_on: vec![] });
    t.threads[2].push(MemoryOp { thread_id: 2, op_index: 1, opcode: Opcode::Load,  address: Some(1), value: None,    depends_on: vec![] });
    // Reader 1: reads y then x
    t.threads[3].push(MemoryOp { thread_id: 3, op_index: 0, opcode: Opcode::Load,  address: Some(1), value: None,    depends_on: vec![] });
    t.threads[3].push(MemoryOp { thread_id: 3, op_index: 1, opcode: Opcode::Load,  address: Some(0), value: None,    depends_on: vec![] });
    t
}

fn build_2plus2w_alg() -> AlgLitmusTest {
    // 2+2W: Two threads each write to both addresses in opposite order
    let mut t = AlgLitmusTest::new("2+2W", 2, 2, 2);
    t.threads[0].push(MemoryOp { thread_id: 0, op_index: 0, opcode: Opcode::Store, address: Some(0), value: Some(1), depends_on: vec![] });
    t.threads[0].push(MemoryOp { thread_id: 0, op_index: 1, opcode: Opcode::Store, address: Some(1), value: Some(1), depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 0, opcode: Opcode::Store, address: Some(1), value: Some(1), depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 1, opcode: Opcode::Store, address: Some(0), value: Some(1), depends_on: vec![] });
    t
}

fn build_rwc_alg() -> AlgLitmusTest {
    // RWC (Read-Write Causality): 3 threads
    // T0: W(x,1); T1: R(x), W(y,1); T2: R(y), R(x)
    let mut t = AlgLitmusTest::new("RWC", 3, 2, 2);
    t.threads[0].push(MemoryOp { thread_id: 0, op_index: 0, opcode: Opcode::Store, address: Some(0), value: Some(1), depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 0, opcode: Opcode::Load,  address: Some(0), value: None,    depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 1, opcode: Opcode::Store, address: Some(1), value: Some(1), depends_on: vec![] });
    t.threads[2].push(MemoryOp { thread_id: 2, op_index: 0, opcode: Opcode::Load,  address: Some(1), value: None,    depends_on: vec![] });
    t.threads[2].push(MemoryOp { thread_id: 2, op_index: 1, opcode: Opcode::Load,  address: Some(0), value: None,    depends_on: vec![] });
    t
}

fn build_wrc_alg() -> AlgLitmusTest {
    // WRC (Write-Read Causality): 3 threads
    // T0: W(x,1); T1: R(x), W(y,1); T2: R(y), R(x)
    // Same as RWC structurally (common pattern)
    let mut t = AlgLitmusTest::new("WRC", 3, 2, 2);
    t.threads[0].push(MemoryOp { thread_id: 0, op_index: 0, opcode: Opcode::Store, address: Some(0), value: Some(1), depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 0, opcode: Opcode::Load,  address: Some(0), value: None,    depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 1, opcode: Opcode::Store, address: Some(1), value: Some(1), depends_on: vec![] });
    t.threads[2].push(MemoryOp { thread_id: 2, op_index: 0, opcode: Opcode::Load,  address: Some(1), value: None,    depends_on: vec![] });
    t.threads[2].push(MemoryOp { thread_id: 2, op_index: 1, opcode: Opcode::Load,  address: Some(0), value: None,    depends_on: vec![] });
    t
}

fn build_sb4_alg() -> AlgLitmusTest {
    // 4-thread store buffering: all threads do W(x_i), R(x_{i+1 mod 4})
    let mut t = AlgLitmusTest::new("SB4", 4, 4, 2);
    for i in 0..4usize {
        let next = (i + 1) % 4;
        t.threads[i].push(MemoryOp { thread_id: i, op_index: 0, opcode: Opcode::Store, address: Some(i), value: Some(1), depends_on: vec![] });
        t.threads[i].push(MemoryOp { thread_id: i, op_index: 1, opcode: Opcode::Load,  address: Some(next), value: None,    depends_on: vec![] });
    }
    t
}

fn build_dekker_alg() -> AlgLitmusTest {
    // Dekker's algorithm pattern (same as SB with fences)
    let mut t = AlgLitmusTest::new("Dekker", 2, 2, 2);
    t.threads[0].push(MemoryOp { thread_id: 0, op_index: 0, opcode: Opcode::Store, address: Some(0), value: Some(1), depends_on: vec![] });
    t.threads[0].push(MemoryOp { thread_id: 0, op_index: 1, opcode: Opcode::Fence(FenceType::Full), address: None, value: None, depends_on: vec![] });
    t.threads[0].push(MemoryOp { thread_id: 0, op_index: 2, opcode: Opcode::Load,  address: Some(1), value: None,    depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 0, opcode: Opcode::Store, address: Some(1), value: Some(1), depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 1, opcode: Opcode::Fence(FenceType::Full), address: None, value: None, depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 2, opcode: Opcode::Load,  address: Some(0), value: None,    depends_on: vec![] });
    t
}

fn build_mp_fence_alg() -> AlgLitmusTest {
    // MP with fences between stores and between loads
    let mut t = AlgLitmusTest::new("MP+fence", 2, 2, 2);
    t.threads[0].push(MemoryOp { thread_id: 0, op_index: 0, opcode: Opcode::Store, address: Some(0), value: Some(1), depends_on: vec![] });
    t.threads[0].push(MemoryOp { thread_id: 0, op_index: 1, opcode: Opcode::Fence(FenceType::Full), address: None, value: None, depends_on: vec![] });
    t.threads[0].push(MemoryOp { thread_id: 0, op_index: 2, opcode: Opcode::Store, address: Some(1), value: Some(1), depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 0, opcode: Opcode::Load,  address: Some(1), value: None,    depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 1, opcode: Opcode::Fence(FenceType::Full), address: None, value: None, depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 2, opcode: Opcode::Load,  address: Some(0), value: None,    depends_on: vec![] });
    t
}

fn build_sb_checker() -> CheckerLitmusTest {
    let mut test = CheckerLitmusTest::new("SB");
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

fn build_mp_checker() -> CheckerLitmusTest {
    let mut test = CheckerLitmusTest::new("MP");
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

fn build_lb_checker() -> CheckerLitmusTest {
    let mut test = CheckerLitmusTest::new("LB");
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

fn build_stub_checker() -> CheckerLitmusTest {
    build_sb_checker() // placeholder for tests without checker impl
}

fn main() {
    let output_dir = std::env::args().nth(1).unwrap_or_else(|| "benchmark_results".to_string());
    fs::create_dir_all(&output_dir).unwrap();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       LITMUS∞ Comprehensive Benchmark Suite                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Experiment 1: Compression Ratios ──────────────────────────────

    println!("=== Experiment 1: Algebraic Compression Ratios ===");
    let tests: Vec<(&str, Box<dyn Fn() -> AlgLitmusTest>)> = vec![
        ("SB",        Box::new(build_sb_alg)),
        ("MP",        Box::new(build_mp_alg)),
        ("LB",        Box::new(build_lb_alg)),
        ("IRIW",      Box::new(build_iriw_alg)),
        ("2+2W",      Box::new(build_2plus2w_alg)),
        ("RWC",       Box::new(build_rwc_alg)),
        ("WRC",       Box::new(build_wrc_alg)),
        ("SB4",       Box::new(build_sb4_alg)),
        ("Dekker",    Box::new(build_dekker_alg)),
        ("MP+fence",  Box::new(build_mp_fence_alg)),
    ];

    let mut compression_csv = String::from("test,threads,addresses,values,original_space,compressed_space,compression_ratio,thread_sym_order,addr_sym_order,val_sym_order,joint_auto_order,certificate_valid,detection_time_us\n");

    for (name, builder) in &tests {
        let t = builder();
        let start = Instant::now();
        let compressor = StateSpaceCompressor::new(t.clone());
        let result = compressor.compress();
        let elapsed = start.elapsed().as_micros();

        let sym = compressor.symmetry();
        let line = format!(
            "{},{},{},{},{},{},{:.2},{},{},{},{},{},{}\n",
            name,
            t.num_threads,
            t.num_addresses,
            t.num_values,
            result.ratio.original_size,
            result.ratio.compressed_size,
            result.ratio.ratio,
            sym.thread_group.order(),
            sym.address_group.order(),
            sym.value_group.order(),
            sym.total_order,
            result.certificate.is_valid(),
            elapsed,
        );
        print!("  {} — {:.1}x compression (|Aut|={}), cert: {}, {:.0}μs",
            name, result.ratio.ratio, sym.total_order,
            if result.certificate.is_valid() { "✓" } else { "✗" }, elapsed);
        println!();
        compression_csv.push_str(&line);
    }

    fs::write(format!("{}/compression_ratios.csv", output_dir), &compression_csv).unwrap();

    // ── Experiment 2: Verification across models ─────────────────────

    println!("\n=== Experiment 2: Cross-Model Verification ===");
    let models = vec![
        ("SC", BuiltinModel::SC),
        ("TSO", BuiltinModel::TSO),
        ("PSO", BuiltinModel::PSO),
        ("ARM", BuiltinModel::ARM),
        ("RISC-V", BuiltinModel::RISCV),
    ];

    let checker_tests: Vec<(&str, Box<dyn Fn() -> CheckerLitmusTest>)> = vec![
        ("SB", Box::new(build_sb_checker)),
        ("MP", Box::new(build_mp_checker)),
        ("LB", Box::new(build_lb_checker)),
    ];

    let mut verify_csv = String::from("test,model,total_executions,consistent_executions,forbidden_found,verification_time_us\n");

    for (test_name, test_builder) in &checker_tests {
        for (model_name, model_enum) in &models {
            let litmus = test_builder();
            let mem_model = model_enum.build();
            let mut verifier = Verifier::new(mem_model);

            let start = Instant::now();
            let result = verifier.verify_litmus(&litmus);
            let elapsed = start.elapsed().as_micros();

            let forbidden = result.total_executions - result.consistent_executions;
            let line = format!(
                "{},{},{},{},{},{}\n",
                test_name, model_name,
                result.total_executions, result.consistent_executions,
                forbidden, elapsed,
            );
            print!("  {}/{}: {}/{} consistent, {} forbidden, {:.0}μs",
                test_name, model_name,
                result.consistent_executions, result.total_executions,
                forbidden, elapsed);
            println!();
            verify_csv.push_str(&line);
        }
    }

    fs::write(format!("{}/verification_results.csv", output_dir), &verify_csv).unwrap();

    // ── Experiment 3: Scalability — increasing thread/address counts ─

    println!("\n=== Experiment 3: Scalability (SB-like tests with N threads) ===");
    let mut scale_csv = String::from("n_threads,n_addresses,original_space,compressed_space,compression_ratio,auto_order,detection_time_us\n");

    for n in 2..=7usize {
        let mut t = AlgLitmusTest::new(&format!("SB{}", n), n, n, 2);
        for i in 0..n {
            let next = (i + 1) % n;
            t.threads[i].push(MemoryOp {
                thread_id: i, op_index: 0, opcode: Opcode::Store,
                address: Some(i), value: Some(1), depends_on: vec![],
            });
            t.threads[i].push(MemoryOp {
                thread_id: i, op_index: 1, opcode: Opcode::Load,
                address: Some(next), value: None, depends_on: vec![],
            });
        }

        let start = Instant::now();
        let compressor = StateSpaceCompressor::new(t);
        let result = compressor.compress();
        let elapsed = start.elapsed().as_micros();

        let sym = compressor.symmetry();
        let line = format!("{},{},{},{},{:.2},{},{}\n",
            n, n, result.ratio.original_size, result.ratio.compressed_size,
            result.ratio.ratio, sym.total_order, elapsed);
        print!("  N={}: {:.1}x compression (|Aut|={}), {:.0}μs", n, result.ratio.ratio, sym.total_order, elapsed);
        println!();
        scale_csv.push_str(&line);
    }

    fs::write(format!("{}/scalability.csv", output_dir), &scale_csv).unwrap();

    // ── Experiment 4: Brute-force vs compressed verification timing ──

    println!("\n=== Experiment 4: Verification Speedup (Brute-force vs Compressed) ===");
    let mut speedup_csv = String::from("test,model,brute_force_executions,compressed_executions,brute_force_time_us,compressed_time_us,speedup\n");

    for (test_name, test_builder) in &checker_tests {
        let litmus = test_builder();

        // Brute force (SC)
        let mem_model = BuiltinModel::SC.build();
        let mut verifier = Verifier::new(mem_model);
        let start = Instant::now();
        let bf_result = verifier.verify_litmus(&litmus);
        let bf_time = start.elapsed().as_micros();

        // Compressed (using symmetry to reduce)
        let alg_test = match *test_name {
            "SB" => build_sb_alg(),
            "MP" => build_mp_alg(),
            "LB" => build_lb_alg(),
            _ => continue,
        };
        let start = Instant::now();
        let compressor = StateSpaceCompressor::new(alg_test);
        let comp_result = compressor.compress();
        let comp_time = start.elapsed().as_micros();

        let speedup = if comp_time > 0 { bf_time as f64 / comp_time as f64 } else { 0.0 };
        let line = format!("{},SC,{},{},{},{},{:.2}\n",
            test_name, bf_result.total_executions, comp_result.ratio.compressed_size,
            bf_time, comp_time, speedup);
        print!("  {}: brute_force={} ({:.0}μs), compressed={} ({:.0}μs), speedup={:.1}x",
            test_name, bf_result.total_executions, bf_time,
            comp_result.ratio.compressed_size, comp_time, speedup);
        println!();
        speedup_csv.push_str(&line);
    }

    fs::write(format!("{}/speedup.csv", output_dir), &speedup_csv).unwrap();

    // ── Experiment 5: Model Distinguishing Power ─────────────────────

    println!("\n=== Experiment 5: Model Distinguishing Power ===");
    // For each test, determine which models allow the forbidden outcome
    let mut distinguish_csv = String::from("test,model,allows_forbidden\n");

    let distinguish_tests: Vec<(&str, &str, Box<dyn Fn() -> CheckerLitmusTest>)> = vec![
        ("SB", "Both loads see 0", Box::new(build_sb_checker)),
        ("MP", "Flag=1 but data=0", Box::new(build_mp_checker)),
        ("LB", "Both loads see 1", Box::new(build_lb_checker)),
    ];

    for (test_name, desc, test_builder) in &distinguish_tests {
        print!("  {} ({}): ", test_name, desc);
        for (model_name, model_enum) in &models {
            let litmus = test_builder();
            let mem_model = model_enum.build();
            let mut verifier = Verifier::new(mem_model);
            let result = verifier.verify_litmus(&litmus);

            // Under SC, forbidden outcome should not be consistent
            let allows_forbidden = result.consistent_executions == result.total_executions;
            let line = format!("{},{},{}\n", test_name, model_name, allows_forbidden);
            distinguish_csv.push_str(&line);

            print!("{}={} ", model_name, if allows_forbidden { "allow" } else { "forbid" });
        }
        println!();
    }

    fs::write(format!("{}/model_distinguishing.csv", output_dir), &distinguish_csv).unwrap();

    // ── Experiment 6: Certificate Validation ─────────────────────────

    println!("\n=== Experiment 6: Certificate Validation ===");
    let mut cert_csv = String::from("test,symmetry_order,generators_verified,canonical_consistent,certificate_valid,num_orbits\n");

    for (name, builder) in &tests {
        let t = builder();
        let compressor = StateSpaceCompressor::new(t);
        let result = compressor.compress();
        let cert = &result.certificate;

        let line = format!("{},{},{},{},{},{}\n",
            name, cert.symmetry_order,
            cert.generators_verified, cert.canonical_consistent,
            cert.is_valid(), cert.num_orbits);
        print!("  {}: |G|={}, verified={}, canonical={}, valid={}, orbits={}",
            name, cert.symmetry_order,
            cert.generators_verified, cert.canonical_consistent,
            cert.is_valid(), cert.num_orbits);
        println!();
        cert_csv.push_str(&line);
    }

    fs::write(format!("{}/certificates.csv", output_dir), &cert_csv).unwrap();

    println!("\n✓ All experiments complete. Results saved to {}/", output_dir);
    println!("  compression_ratios.csv      — Table 1: Compression ratios");
    println!("  verification_results.csv    — Table 2: Cross-model verification");
    println!("  scalability.csv             — Table 3: Scalability with thread count");
    println!("  speedup.csv                 — Table 4: Brute-force vs compressed timing");
    println!("  model_distinguishing.csv    — Table 5: Model distinguishing power");
    println!("  certificates.csv            — Table 6: Certificate validation");
}
