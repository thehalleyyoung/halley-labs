//! End-to-end metric certification benchmark.
//!
//! Demonstrates the full Spectacles pipeline: for each metric, build a
//! metric-specific WFA, compile to AIR circuit, generate STARK proof,
//! and verify it.  Outputs JSON results.

use spectacles_core::pipeline::{certify_metric, certify_batch, MetricCertificate};
use spectacles_core::scoring::ScoringPair;
use serde::Serialize;
use std::time::Instant;

#[derive(Serialize)]
struct BenchmarkResults {
    timestamp: String,
    description: String,
    metrics_tested: Vec<String>,
    total_pairs: usize,
    total_certificates: usize,
    certificates_verified: usize,
    triple_agreements: usize,
    results: Vec<MetricResult>,
    summary: BenchmarkSummary,
}

#[derive(Serialize)]
struct MetricResult {
    metric: String,
    candidate: String,
    reference: String,
    score: f64,
    proof_generated: bool,
    proof_verified: bool,
    triple_agreement: bool,
    prove_time_ms: f64,
    verify_time_ms: f64,
    proof_size_bytes: usize,
    num_wfa_states: usize,
    num_constraints: usize,
}

#[derive(Serialize)]
struct BenchmarkSummary {
    all_proofs_verified: bool,
    all_triple_agreements: bool,
    total_prove_time_ms: f64,
    total_verify_time_ms: f64,
    avg_prove_time_ms: f64,
    avg_verify_time_ms: f64,
    avg_proof_size_bytes: f64,
}

fn main() {
    println!("=== Spectacles End-to-End Certification Benchmark ===\n");

    let metrics = vec!["exact_match", "token_f1", "bleu", "rouge_1"];

    let test_pairs = vec![
        ScoringPair {
            candidate: "the cat sat on the mat".to_string(),
            reference: "the cat sat on the mat".to_string(),
        },
        ScoringPair {
            candidate: "the cat sat on the mat".to_string(),
            reference: "the cat is sitting on the mat".to_string(),
        },
        ScoringPair {
            candidate: "hello world foo bar".to_string(),
            reference: "hello world baz qux".to_string(),
        },
        ScoringPair {
            candidate: "a b c d e".to_string(),
            reference: "a b c d e".to_string(),
        },
        ScoringPair {
            candidate: "completely different text here".to_string(),
            reference: "nothing in common at all".to_string(),
        },
        ScoringPair {
            candidate: "the quick brown fox jumps over the lazy dog".to_string(),
            reference: "the quick brown fox jumped over the lazy dog".to_string(),
        },
    ];

    let mut results = Vec::new();
    let mut total_certs = 0;
    let mut certs_verified = 0;
    let mut triple_agr = 0;

    let start = Instant::now();

    for metric in &metrics {
        println!("--- Metric: {} ---", metric);
        for (i, pair) in test_pairs.iter().enumerate() {
            match certify_metric(metric, &pair.candidate, &pair.reference) {
                Ok(cert) => {
                    total_certs += 1;
                    if cert.proof_verified { certs_verified += 1; }
                    if cert.triple_agreement { triple_agr += 1; }

                    println!("  [{}] score={:.4} proof={} verified={} triple={} prove={:.1}ms verify={:.1}ms",
                        i, cert.score,
                        if cert.proof_generated { "✓" } else { "✗" },
                        if cert.proof_verified { "✓" } else { "✗" },
                        if cert.triple_agreement { "✓" } else { "✗" },
                        cert.prove_time_ms, cert.verify_time_ms);

                    results.push(MetricResult {
                        metric: metric.to_string(),
                        candidate: pair.candidate.clone(),
                        reference: pair.reference.clone(),
                        score: cert.score,
                        proof_generated: cert.proof_generated,
                        proof_verified: cert.proof_verified,
                        triple_agreement: cert.triple_agreement,
                        prove_time_ms: cert.prove_time_ms,
                        verify_time_ms: cert.verify_time_ms,
                        proof_size_bytes: cert.proof_size_bytes,
                        num_wfa_states: cert.num_wfa_states,
                        num_constraints: cert.num_constraints,
                    });
                }
                Err(e) => {
                    println!("  [{}] ERROR: {}", i, e);
                }
            }
        }
        println!();
    }

    let total_time = start.elapsed().as_secs_f64() * 1000.0;
    let n = results.len().max(1) as f64;

    let summary = BenchmarkSummary {
        all_proofs_verified: certs_verified == total_certs,
        all_triple_agreements: triple_agr == total_certs,
        total_prove_time_ms: results.iter().map(|r| r.prove_time_ms).sum(),
        total_verify_time_ms: results.iter().map(|r| r.verify_time_ms).sum(),
        avg_prove_time_ms: results.iter().map(|r| r.prove_time_ms).sum::<f64>() / n,
        avg_verify_time_ms: results.iter().map(|r| r.verify_time_ms).sum::<f64>() / n,
        avg_proof_size_bytes: results.iter().map(|r| r.proof_size_bytes as f64).sum::<f64>() / n,
    };

    println!("=== Summary ===");
    println!("Total certificates: {}", total_certs);
    println!("Proofs verified:    {}/{}", certs_verified, total_certs);
    println!("Triple agreements:  {}/{}", triple_agr, total_certs);
    println!("Total time:         {:.1}ms", total_time);
    println!("All verified:       {}", summary.all_proofs_verified);

    let benchmark = BenchmarkResults {
        timestamp: chrono::Utc::now().to_rfc3339(),
        description: "End-to-end metric certification: WFA → circuit → STARK proof → verification".to_string(),
        metrics_tested: metrics.iter().map(|s| s.to_string()).collect(),
        total_pairs: test_pairs.len(),
        total_certificates: total_certs,
        certificates_verified: certs_verified,
        triple_agreements: triple_agr,
        results,
        summary,
    };

    let json = serde_json::to_string_pretty(&benchmark).unwrap();
    std::fs::write("e2e_certification_results.json", &json).unwrap();
    println!("\nResults written to e2e_certification_results.json");

    // Batch certification demo
    println!("\n=== Batch Certification Demo ===");
    let batch_items: Vec<(&str, &str, &str)> = test_pairs.iter()
        .map(|p| ("exact_match", p.candidate.as_str(), p.reference.as_str()))
        .collect();
    let batch_certs = certify_batch(&batch_items);
    let batch_ok = batch_certs.iter().filter(|r| r.as_ref().map_or(false, |c| c.proof_verified)).count();
    println!("Batch exact_match: {}/{} verified", batch_ok, batch_certs.len());
}
