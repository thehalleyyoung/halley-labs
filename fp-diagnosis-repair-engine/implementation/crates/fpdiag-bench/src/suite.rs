//! FPBench test suite — standard expressions from fpbench.org
//! and Herbie's benchmark collection.

use fpdiag_types::{
    expression::{ExprBuilder, FpOp},
    precision::Precision,
    trace::{ExecutionTrace, TraceEvent},
};

/// A benchmark case with known instability.
pub struct BenchCase {
    pub name: &'static str,
    pub category: &'static str,
    pub description: &'static str,
    pub fpcore: &'static str,
    pub test_inputs: Vec<Vec<f64>>,
    pub known_bits_lost: f64,
}

/// Standard FPBench expressions known to exhibit numerical issues.
pub fn fpbench_standard_suite() -> Vec<BenchCase> {
    vec![
        BenchCase {
            name: "NMSE-3.1",
            category: "cancellation",
            description: "sqrt(x+1) - sqrt(x) for large x",
            fpcore: "(FPCore (x) :name \"NMSE-3.1\" (- (sqrt (+ x 1.0)) (sqrt x)))",
            test_inputs: vec![
                vec![1e8], vec![1e12], vec![1e15], vec![1e16],
            ],
            known_bits_lost: 26.0,
        },
        BenchCase {
            name: "NMSE-3.3",
            category: "cancellation",
            description: "1/(x+1) - 1/x for large x",
            fpcore: "(FPCore (x) :name \"NMSE-3.3\" (- (/ 1.0 (+ x 1.0)) (/ 1.0 x)))",
            test_inputs: vec![
                vec![1e8], vec![1e12], vec![1e15],
            ],
            known_bits_lost: 32.0,
        },
        BenchCase {
            name: "expm1",
            category: "cancellation",
            description: "exp(x) - 1 for small x",
            fpcore: "(FPCore (x) :name \"expm1\" (- (exp x) 1.0))",
            test_inputs: vec![
                vec![1e-10], vec![1e-14], vec![1e-16],
            ],
            known_bits_lost: 48.0,
        },
        BenchCase {
            name: "log1p",
            category: "cancellation",
            description: "log(1 + x) for small x",
            fpcore: "(FPCore (x) :name \"log1p\" (log (+ 1.0 x)))",
            test_inputs: vec![
                vec![1e-10], vec![1e-14], vec![1e-16],
            ],
            known_bits_lost: 48.0,
        },
        BenchCase {
            name: "quadratic-pos",
            category: "cancellation",
            description: "(-b + sqrt(b²-4ac)) / 2a",
            fpcore: "(FPCore (a b c) :name \"quadratic-pos\" (/ (+ (neg b) (sqrt (- (* b b) (* (* 4.0 a) c)))) (* 2.0 a)))",
            test_inputs: vec![
                vec![1.0, 1e8, 1.0],
                vec![1.0, -1e8, 1.0],
            ],
            known_bits_lost: 30.0,
        },
        BenchCase {
            name: "hypot-naive",
            category: "overflow",
            description: "sqrt(a² + b²) — overflow for large a,b",
            fpcore: "(FPCore (a b) :name \"hypot-naive\" (sqrt (+ (* a a) (* b b))))",
            test_inputs: vec![
                vec![1e154, 1e154],
                vec![3e-200, 4e-200],
            ],
            known_bits_lost: 53.0, // inf or 0
        },
        BenchCase {
            name: "kahan-sum-absorption",
            category: "absorption",
            description: "Naive summation of 1e16 + many 1.0s",
            fpcore: "(FPCore (x) :name \"sum-absorption\" (+ 1e16 x))",
            test_inputs: vec![
                vec![1.0], vec![0.5], vec![0.1],
            ],
            known_bits_lost: 53.0,
        },
        BenchCase {
            name: "polynomial-horner",
            category: "amplified_rounding",
            description: "Wilkinson polynomial near root",
            fpcore: "(FPCore (x) :name \"wilkinson\" (* (- x 1.0) (- x 2.0)))",
            test_inputs: vec![
                vec![1.0 + 1e-15], vec![2.0 - 1e-15],
            ],
            known_bits_lost: 45.0,
        },
    ]
}

/// Build an execution trace from a single FPBench case.
pub fn build_trace_for_case(case: &BenchCase) -> ExecutionTrace {
    let mut trace = ExecutionTrace::new();
    let inputs = &case.test_inputs[0];

    match case.name {
        "expm1" => {
            let x = inputs[0];
            let exp_x = x.exp();
            trace.push(TraceEvent::Operation {
                seq: 0,
                op: FpOp::Exp,
                inputs: vec![x],
                output: exp_x,
                shadow_output: exp_x,
                precision: Precision::Double,
                source: None,
                expr_node: None,
            });
            trace.push(TraceEvent::Operation {
                seq: 1,
                op: FpOp::Sub,
                inputs: vec![exp_x, 1.0],
                output: exp_x - 1.0,
                shadow_output: x, // true value for small x
                precision: Precision::Double,
                source: None,
                expr_node: None,
            });
        }
        "log1p" => {
            let x = inputs[0];
            let one_plus_x = 1.0 + x;
            trace.push(TraceEvent::Operation {
                seq: 0,
                op: FpOp::Add,
                inputs: vec![1.0, x],
                output: one_plus_x,
                shadow_output: 1.0 + x,
                precision: Precision::Double,
                source: None,
                expr_node: None,
            });
            trace.push(TraceEvent::Operation {
                seq: 1,
                op: FpOp::Log,
                inputs: vec![one_plus_x],
                output: one_plus_x.ln(),
                shadow_output: x, // true for small x
                precision: Precision::Double,
                source: None,
                expr_node: None,
            });
        }
        _ => {
            // Generic: single operation trace
            if inputs.len() >= 2 {
                trace.push(TraceEvent::Operation {
                    seq: 0,
                    op: FpOp::Sub,
                    inputs: inputs.clone(),
                    output: inputs[0] - inputs[1],
                    shadow_output: inputs[0] - inputs[1],
                    precision: Precision::Double,
                    source: None,
                    expr_node: None,
                });
            } else {
                trace.push(TraceEvent::Operation {
                    seq: 0,
                    op: FpOp::Add,
                    inputs: vec![inputs[0], 0.0],
                    output: inputs[0],
                    shadow_output: inputs[0],
                    precision: Precision::Double,
                    source: None,
                    expr_node: None,
                });
            }
        }
    }

    trace.finalize();
    trace
}

/// Herbie reference results for comparison.
pub struct HerbieReference {
    pub benchmark: &'static str,
    pub original_bits_correct: f64,
    pub herbie_bits_correct: f64,
    pub herbie_rewrite: &'static str,
}

/// Known Herbie results on standard benchmarks.
pub fn herbie_reference_results() -> Vec<HerbieReference> {
    vec![
        HerbieReference {
            benchmark: "expm1",
            original_bits_correct: 0.0,
            herbie_bits_correct: 52.0,
            herbie_rewrite: "expm1(x)",
        },
        HerbieReference {
            benchmark: "log1p",
            original_bits_correct: 0.0,
            herbie_bits_correct: 52.0,
            herbie_rewrite: "log1p(x)",
        },
        HerbieReference {
            benchmark: "NMSE-3.1",
            original_bits_correct: 26.0,
            herbie_bits_correct: 51.0,
            herbie_rewrite: "1 / (sqrt(x+1) + sqrt(x))",
        },
        HerbieReference {
            benchmark: "NMSE-3.3",
            original_bits_correct: 20.0,
            herbie_bits_correct: 50.0,
            herbie_rewrite: "-1 / (x * (x + 1))",
        },
        HerbieReference {
            benchmark: "quadratic-pos",
            original_bits_correct: 22.0,
            herbie_bits_correct: 50.0,
            herbie_rewrite: "(-b + sqrt(b²-4ac)) / 2a  [citardauq form]",
        },
        HerbieReference {
            benchmark: "hypot-naive",
            original_bits_correct: 0.0,
            herbie_bits_correct: 53.0,
            herbie_rewrite: "hypot(a, b)",
        },
    ]
}
