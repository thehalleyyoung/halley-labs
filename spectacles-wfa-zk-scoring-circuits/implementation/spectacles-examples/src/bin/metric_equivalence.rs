//! Example: Check if two BLEU implementations compute the same function.
//!
//! Runs differential testing across multiple smoothing methods and configurations.

use spectacles_core::scoring::{
    ScoringPair, TripleMetric,
    bleu::{BleuScorer, BleuConfig, SmoothingMethod},
    token_f1::TokenF1Scorer,
    rouge::{RougeNScorer, RougeLScorer, RougeConfig},
    differential::{DifferentialTester, standard_test_suite, random_test_pairs},
};

fn main() {
    env_logger::init();
    
    println!("=== Metric Equivalence Checker ===");
    println!();
    
    // Generate test pairs
    let mut pairs = standard_test_suite();
    pairs.extend(random_test_pairs(200, 42));
    println!("Generated {} test pairs", pairs.len());
    println!();
    
    // 1. Compare BLEU with different smoothing methods
    println!("--- BLEU Smoothing Method Comparison ---");
    let smoothing_methods = vec![
        ("None", SmoothingMethod::None),
        ("Add1", SmoothingMethod::Add1),
        ("AddK", SmoothingMethod::AddK),
        ("Floor", SmoothingMethod::Floor),
        ("ChenCherry", SmoothingMethod::ChenCherry),
    ];
    
    for (name, method) in &smoothing_methods {
        let scorer = BleuScorer::with_smoothing(*method);
        let tester = DifferentialTester::new();
        let report = tester.test_bleu(&pairs);
        
        println!("[{}] BLEU-{}: {}/{} agree ({:.1}%)",
            if report.is_perfect() { "PASS" } else { "FAIL" },
            name,
            report.agreements,
            report.total_tests,
            report.agreement_rate * 100.0);
    }
    
    // 2. Compare BLEU with different max_n values
    println!();
    println!("--- BLEU N-gram Order Comparison ---");
    for n in 1..=4 {
        let scorer = BleuScorer::new(BleuConfig::default().with_max_n(n).with_smoothing(SmoothingMethod::Add1));
        let tester = DifferentialTester::new();
        let report = tester.test_bleu(&pairs);
        
        println!("[{}] BLEU-{}: {}/{} agree ({:.1}%)",
            if report.is_perfect() { "PASS" } else { "FAIL" },
            n,
            report.agreements,
            report.total_tests,
            report.agreement_rate * 100.0);
    }
    
    // 3. Compare ROUGE variants
    println!();
    println!("--- ROUGE Variant Comparison ---");
    let tester = DifferentialTester::new();
    
    let r1_report = tester.test_rouge1(&pairs);
    println!("[{}] ROUGE-1: {}/{} agree ({:.1}%)",
        if r1_report.is_perfect() { "PASS" } else { "FAIL" },
        r1_report.agreements, r1_report.total_tests,
        r1_report.agreement_rate * 100.0);
    
    // ROUGE-2 triple test
    let scorer_r2 = RougeNScorer::rouge2();
    let mut r2_agree = 0;
    for pair in &pairs {
        let result = scorer_r2.score_and_verify(pair);
        if result.agreement { r2_agree += 1; }
    }
    println!("[{}] ROUGE-2: {}/{} agree ({:.1}%)",
        if r2_agree == pairs.len() { "PASS" } else { "FAIL" },
        r2_agree, pairs.len(),
        r2_agree as f64 / pairs.len() as f64 * 100.0);
    
    let rl_report = tester.test_rouge_l(&pairs);
    println!("[{}] ROUGE-L: {}/{} agree ({:.1}%)",
        if rl_report.is_perfect() { "PASS" } else { "FAIL" },
        rl_report.agreements, rl_report.total_tests,
        rl_report.agreement_rate * 100.0);
    
    // 4. Token F1 comparison
    println!();
    println!("--- Token F1 Comparison ---");
    let f1_report = tester.test_token_f1(&pairs);
    println!("[{}] Token-F1: {}/{} agree ({:.1}%)",
        if f1_report.is_perfect() { "PASS" } else { "FAIL" },
        f1_report.agreements, f1_report.total_tests,
        f1_report.agreement_rate * 100.0);
    
    // 5. Exact match comparison
    println!();
    println!("--- Exact Match Comparison ---");
    let em_report = tester.test_exact_match(&pairs);
    println!("[{}] Exact-Match: {}/{} agree ({:.1}%)",
        if em_report.is_perfect() { "PASS" } else { "FAIL" },
        em_report.agreements, em_report.total_tests,
        em_report.agreement_rate * 100.0);
    
    // 6. Coverage report
    println!();
    println!("--- Coverage Report ---");
    let all_reports = tester.test_all_metrics(&pairs);
    if let Some(report) = all_reports.values().next() {
        println!("  Empty inputs:     {}", report.coverage.empty_inputs);
        println!("  Single token:     {}", report.coverage.single_token);
        println!("  Multi token:      {}", report.coverage.multi_token);
        println!("  Exact match:      {}", report.coverage.exact_match_true);
        println!("  No match:         {}", report.coverage.exact_match_false);
        println!("  Partial overlap:  {}", report.coverage.partial_overlap);
        println!("  Length mismatch:  {}", report.coverage.length_mismatch);
        println!("  Repeated tokens:  {}", report.coverage.repeated_tokens);
    }
    
    // Summary
    println!();
    let total_pass = all_reports.values().filter(|r| r.is_perfect()).count();
    let total_metrics = all_reports.len();
    println!("=== Summary: {}/{} metrics have perfect triple agreement ===", total_pass, total_metrics);
}
