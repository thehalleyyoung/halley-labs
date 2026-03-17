//! Demonstrates FPBench format parsing and SMT-LIB emission.
//!
//! Run with: `cargo run --example fpbench_demo`

use fpdiag_types::fpbench::{emit_fpcore, emit_smtlib, parse_fpbench_file, parse_fpcore};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   Penumbra: FPBench Format Support Demo                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // --- Parse a single FPCore ---
    let fpcore_src = r#"(FPCore (x)
  :name "expm1"
  :description "exp(x) - 1, numerically unstable for small x"
  (- (exp x) 1.0))"#;

    println!("1. Parsing FPCore expression:");
    println!("   Input: {}", fpcore_src.lines().next().unwrap());
    let core = parse_fpcore(fpcore_src).expect("parse failed");
    println!("   Name: {:?}", core.name);
    println!("   Inputs: {:?}", core.inputs);
    println!("   Nodes in expression tree: {}", core.body.len());
    println!();

    // --- Emit back as FPCore ---
    println!("2. Round-trip: Emit as FPCore:");
    let emitted = emit_fpcore(&core);
    println!("{}", emitted);
    println!();

    // --- Emit as SMT-LIB ---
    println!("3. Emit as SMT-LIB2 (QF_FP logic):");
    let smt = emit_smtlib(&core.body, &core.inputs);
    println!("{}", smt);

    // --- Parse standard FPBench benchmarks ---
    println!("4. Parse multiple FPBench expressions:");
    let benchmarks = r#"
(FPCore (x)
  :name "NMSE-example-3.1"
  :cite (hamming-1987)
  (- (sqrt (+ x 1.0)) (sqrt x)))

(FPCore (x y)
  :name "NMSE-example-3.3"
  :cite (hamming-1987)
  (- (/ 1.0 (+ x 1.0)) (/ 1.0 x)))

(FPCore (a b c)
  :name "quadratic-formula-pos"
  :cite (goldberg-1991)
  (/ (+ (neg b) (sqrt (- (* b b) (* (* 4.0 a) c)))) (* 2.0 a)))

(FPCore (x)
  :name "log1p-unstable"
  (log (+ 1.0 x)))

(FPCore (a b)
  :name "hypot-naive"
  (sqrt (+ (* a a) (* b b))))
"#;

    let results = parse_fpbench_file(benchmarks);
    println!("   Parsed {} benchmarks:", results.len());
    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(core) => {
                println!(
                    "   [{}] {} — {} inputs, {} nodes",
                    i + 1,
                    core.name.as_deref().unwrap_or("unnamed"),
                    core.inputs.len(),
                    core.body.len()
                );
            }
            Err(e) => println!("   [{}] PARSE ERROR: {}", i + 1, e),
        }
    }
    println!();

    // --- Show known numerically unstable patterns ---
    println!("5. Known unstable patterns from FPBench:");
    let patterns = [
        ("sqrt(x+1) - sqrt(x)", "Cancellation for large x"),
        ("exp(x) - 1", "Cancellation near x=0, use expm1"),
        ("log(1 + x)", "Cancellation near x=0, use log1p"),
        ("a² + b²", "Overflow risk, use hypot"),
        (
            "(-b + sqrt(b²-4ac)) / 2a",
            "Cancellation, use stable quadratic",
        ),
    ];

    for (expr, issue) in &patterns {
        println!("   {:<35} → {}", expr, issue);
    }

    println!();
    println!("FPBench integration enables Penumbra to consume the standard");
    println!("benchmark suite and compare diagnosis quality against Herbie.");
}
