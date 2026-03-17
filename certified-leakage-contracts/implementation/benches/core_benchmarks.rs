//! Benchmarks for the core abstract domains and composition operations.
//!
//! Run with:
//! ```sh
//! cargo bench --bench core_benchmarks
//! ```

use std::collections::BTreeMap;
use std::time::Instant;

use shared_types::{CacheGeometry, FunctionId};
use leak_contract::{
    CacheSetState, CacheLineState,
    LeakageContract, CacheTransformer, LeakageBound,
    ContractMetadata, ContractStrength,
    compose_sequential, compose_conditional,
};
use leak_quantify::{
    Distribution, ShannonEntropy, MinEntropy, GuessingEntropy,
    CountingDomain, TaintRestrictedCounting,
};
use leak_analysis::{
    SpecState, MisspecKind, SpecWindow, SpecDomain,
};

// ---------------------------------------------------------------------------
// Lightweight benchmark harness (no external dependencies)
// ---------------------------------------------------------------------------

struct BenchResult {
    name: String,
    iterations: u64,
    total_ns: u128,
    per_iter_ns: f64,
}

impl BenchResult {
    fn report(&self) {
        let (value, unit) = if self.per_iter_ns > 1_000_000.0 {
            (self.per_iter_ns / 1_000_000.0, "ms")
        } else if self.per_iter_ns > 1_000.0 {
            (self.per_iter_ns / 1_000.0, "μs")
        } else {
            (self.per_iter_ns, "ns")
        };
        println!(
            "  {:<50} {:>10.2} {}/iter  ({} iters)",
            self.name, value, unit, self.iterations,
        );
    }
}

fn bench<F: FnMut()>(name: &str, mut f: F) -> BenchResult {
    // Warm up
    for _ in 0..10 { f(); }

    // Calibrate: fill ~100ms
    let cal_start = Instant::now();
    let mut cal_iters = 0u64;
    while cal_start.elapsed().as_millis() < 100 {
        f();
        cal_iters += 1;
    }
    let target_iters = (cal_iters * 10).max(100);

    let start = Instant::now();
    for _ in 0..target_iters { f(); }
    let elapsed = start.elapsed().as_nanos();

    BenchResult {
        name: name.to_string(),
        iterations: target_iters,
        total_ns: elapsed,
        per_iter_ns: elapsed as f64 / target_iters as f64,
    }
}

// ---------------------------------------------------------------------------
// Benchmark: Cache domain operations
// ---------------------------------------------------------------------------

fn bench_cache_line_join() -> BenchResult {
    let a = CacheLineState::Known { tag: 42 };
    let b = CacheLineState::Known { tag: 99 };
    bench("CacheLineState::join (Known × Known)", || {
        let _ = a.join(&b);
    })
}

fn bench_cache_line_meet() -> BenchResult {
    let a = CacheLineState::OneOf(vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let b = CacheLineState::OneOf(vec![3, 4, 5, 6, 7, 8, 9, 10]);
    bench("CacheLineState::meet (OneOf × OneOf, 8 tags)", || {
        let _ = a.meet(&b);
    })
}

fn bench_cache_set_join() -> BenchResult {
    let a = CacheSetState::new_empty(8);
    let b = CacheSetState::new_unknown(8);
    bench("CacheSetState::join (empty × unknown, 8-way)", || {
        let _ = a.join(&b);
    })
}

// ---------------------------------------------------------------------------
// Benchmark: Entropy computations
// ---------------------------------------------------------------------------

fn bench_shannon_entropy_256() -> BenchResult {
    let dist = Distribution::from_pairs(
        (0u32..256).map(|i| (i, 1.0 / 256.0))
    ).unwrap();
    bench("ShannonEntropy::compute (uniform, n=256)", || {
        let _ = ShannonEntropy::compute(&dist);
    })
}

fn bench_min_entropy_256() -> BenchResult {
    let dist = Distribution::from_pairs(
        (0u32..256).map(|i| (i, 1.0 / 256.0))
    ).unwrap();
    bench("MinEntropy::compute (uniform, n=256)", || {
        let _ = MinEntropy::compute(&dist);
    })
}

fn bench_shannon_skewed() -> BenchResult {
    let mut probs: Vec<(u32, f64)> = (0u32..256).map(|i| (i, 0.01 / 255.0)).collect();
    probs[0].1 = 0.99;
    let sum: f64 = probs.iter().map(|p| p.1).sum();
    for p in &mut probs { p.1 /= sum; }
    let dist = Distribution::from_pairs(probs).unwrap();
    bench("ShannonEntropy::compute (skewed, p_max=0.99)", || {
        let _ = ShannonEntropy::compute(&dist);
    })
}

fn bench_guessing_entropy() -> BenchResult {
    let dist = Distribution::from_pairs(
        (0u32..1024).map(|i| (i, 1.0 / 1024.0))
    ).unwrap();
    bench("GuessingEntropy::compute (uniform, n=1024)", || {
        let _ = GuessingEntropy::compute(&dist);
    })
}

// ---------------------------------------------------------------------------
// Benchmark: Contract composition
// ---------------------------------------------------------------------------

fn make_contract(name: &str, id: u32, bits: f64) -> LeakageContract {
    LeakageContract::new(
        FunctionId::new(id),
        name,
        CacheTransformer::identity(),
        LeakageBound::constant(bits),
    )
}

fn bench_sequential_composition() -> BenchResult {
    let a = make_contract("f", 0, 4.0);
    let b = make_contract("g", 1, 2.0);
    bench("compose_sequential (2 contracts)", || {
        let _ = compose_sequential(&a, &b);
    })
}

fn bench_conditional_composition() -> BenchResult {
    let a = make_contract("then_branch", 0, 4.0);
    let b = make_contract("else_branch", 1, 2.0);
    bench("compose_conditional (2 branches)", || {
        let _ = compose_conditional(&a, &b);
    })
}

fn bench_chain_composition_10() -> BenchResult {
    let contracts: Vec<_> = (0..10)
        .map(|i| make_contract(&format!("round_{i}"), i, 1.0))
        .collect();
    bench("compose_sequential chain (10 rounds)", || {
        let mut composed = contracts[0].clone();
        for c in &contracts[1..] {
            composed = compose_sequential(&composed, c).unwrap();
        }
        std::hint::black_box(&composed);
    })
}

fn bench_chain_composition_255() -> BenchResult {
    let contracts: Vec<_> = (0..255u32)
        .map(|i| make_contract(&format!("step_{i}"), i, 0.01))
        .collect();
    bench("compose_sequential chain (255 steps)", || {
        let mut composed = contracts[0].clone();
        for c in &contracts[1..] {
            composed = compose_sequential(&composed, c).unwrap();
        }
        std::hint::black_box(&composed);
    })
}

// ---------------------------------------------------------------------------
// Benchmark: Counting domain
// ---------------------------------------------------------------------------

fn bench_counting_domain_create() -> BenchResult {
    bench("CountingDomain::new (line_size=64, assoc=8)", || {
        let _ = CountingDomain::new(64, 8);
    })
}

fn bench_taint_restricted_counting() -> BenchResult {
    let domain = CountingDomain::new(64, 8);
    bench("TaintRestrictedCounting (16 tainted sets)", || {
        let mut trc = TaintRestrictedCounting::new(domain.clone());
        for i in 0..16u32 {
            trc.taint_set(i);
        }
        let _ = trc.distinguishable_states();
    })
}

// ---------------------------------------------------------------------------
// Benchmark: Speculative domain
// ---------------------------------------------------------------------------

fn bench_spec_window() -> BenchResult {
    let window = SpecWindow::new(50);
    bench("SpecWindow::new + basic ops", || {
        let w = SpecWindow::new(50);
        std::hint::black_box(&w);
    })
}

// ---------------------------------------------------------------------------
// Comparative: CacheAudit-style vs LeakCert-style
// ---------------------------------------------------------------------------

fn bench_cacheaudit_style() -> BenchResult {
    let geom = CacheGeometry::l1_default();
    bench("CacheAudit-style monolithic (64 sets)", || {
        let mut total = 0.0_f64;
        for _set in 0..geom.num_sets {
            total += (geom.num_ways as f64).log2();
        }
        std::hint::black_box(total);
    })
}

fn bench_leakcert_style() -> BenchResult {
    let geom = CacheGeometry::l1_default();
    let tainted: Vec<u32> = (0..16).collect();
    bench("LeakCert ρ-reduced (16 tainted sets)", || {
        let mut total = 0.0_f64;
        for &_set in &tainted {
            let raw = (geom.num_ways as f64).log2();
            total += raw * 0.4; // ρ reduction
        }
        std::hint::black_box(total);
    })
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║           LeakCert — Core Domain Benchmarks            ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    println!("─── Cache Domain ───");
    bench_cache_line_join().report();
    bench_cache_line_meet().report();
    bench_cache_set_join().report();
    println!();

    println!("─── Entropy Computations ───");
    bench_shannon_entropy_256().report();
    bench_min_entropy_256().report();
    bench_shannon_skewed().report();
    bench_guessing_entropy().report();
    println!();

    println!("─── Contract Composition ───");
    bench_sequential_composition().report();
    bench_conditional_composition().report();
    bench_chain_composition_10().report();
    bench_chain_composition_255().report();
    println!();

    println!("─── Counting Domain ───");
    bench_counting_domain_create().report();
    bench_taint_restricted_counting().report();
    println!();

    println!("─── Speculative Domain ───");
    bench_spec_window().report();
    println!();

    println!("─── SOTA Comparison (simulated) ───");
    bench_cacheaudit_style().report();
    bench_leakcert_style().report();
    println!();

    println!("═══════════════════════════════════════════════════════════");
    println!("  Benchmarks complete. For full binary analysis, provide");
    println!("  a compiled crypto library binary (see examples/).");
    println!("═══════════════════════════════════════════════════════════");
}
