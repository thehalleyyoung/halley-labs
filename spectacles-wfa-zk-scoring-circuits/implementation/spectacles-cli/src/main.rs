//! Spectacles CLI - command-line interface for the Spectacles framework.
//!
//! Subcommands: score, differential-test, compile-circuit, estimate-size, batch-score

use std::path::PathBuf;
use std::fs;
use spectacles_core::scoring::{
    self, ScoringPair, TripleMetric,
    exact_match::ExactMatchScorer,
    token_f1::TokenF1Scorer,
    bleu::{BleuScorer, BleuConfig, SmoothingMethod},
    rouge::{RougeNScorer, RougeLScorer},
    differential::{DifferentialTester, standard_test_suite, random_test_pairs},
};
use spectacles_core::utils::{
    hash::SpectaclesHasher,
    serialization::{ProofSerializer, ProofFormat, CompactProof, estimate_proof_size},
    math,
};

/// Available metrics
#[derive(Debug, Clone, Copy)]
enum MetricType {
    ExactMatch,
    TokenF1,
    Bleu,
    Rouge1,
    Rouge2,
    RougeL,
}

impl MetricType {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "exact_match" | "exact-match" | "em" => Some(Self::ExactMatch),
            "token_f1" | "token-f1" | "f1" => Some(Self::TokenF1),
            "bleu" => Some(Self::Bleu),
            "rouge1" | "rouge-1" => Some(Self::Rouge1),
            "rouge2" | "rouge-2" => Some(Self::Rouge2),
            "rougel" | "rouge-l" => Some(Self::RougeL),
            _ => None,
        }
    }
    
    fn all() -> Vec<&'static str> {
        vec!["exact_match", "token_f1", "bleu", "rouge1", "rouge2", "rougel"]
    }
}

/// CLI entry point
fn main() {
    env_logger::init();
    
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }
    
    let result = match args[1].as_str() {
        "score" => cmd_score(&args[2..]),
        "differential-test" => cmd_differential_test(&args[2..]),
        "compile-circuit" => cmd_compile_circuit(&args[2..]),
        "estimate-size" => cmd_estimate_size(&args[2..]),
        "batch-score" => cmd_batch_score(&args[2..]),
        "hash" => cmd_hash(&args[2..]),
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        "version" | "--version" | "-V" => {
            println!("spectacles-cli 0.1.0");
            Ok(())
        }
        other => {
            eprintln!("Unknown command: {}", other);
            print_usage();
            Err(format!("Unknown command: {}", other))
        }
    };
    
    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn print_usage() {
    println!("spectacles-cli - Spectacles ZK scoring framework");
    println!();
    println!("USAGE:");
    println!("  spectacles-cli <COMMAND> [OPTIONS]");
    println!();
    println!("COMMANDS:");
    println!("  score              Score a candidate against a reference");
    println!("  differential-test  Run differential tests across implementations");
    println!("  compile-circuit    Compile a scoring metric to an arithmetic circuit");
    println!("  estimate-size      Estimate proof size for a configuration");
    println!("  batch-score        Score multiple pairs from a JSON file");
    println!("  hash               Hash input data with BLAKE3");
    println!("  help               Print this help message");
    println!("  version            Print version");
    println!();
    println!("EXAMPLES:");
    println!("  spectacles-cli score --metric bleu --candidate \"the cat sat\" --reference \"the cat sat on the mat\"");
    println!("  spectacles-cli differential-test --metric all --count 100");
    println!("  spectacles-cli estimate-size --constraints 1000 --wires 500");
}

fn cmd_score(args: &[String]) -> Result<(), String> {
    let mut metric_str = "bleu".to_string();
    let mut candidate = String::new();
    let mut reference = String::new();
    let mut triple = false;
    
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--metric" | "-m" => {
                i += 1;
                if i < args.len() { metric_str = args[i].clone(); }
            }
            "--candidate" | "-c" => {
                i += 1;
                if i < args.len() { candidate = args[i].to_string(); }
            }
            "--reference" | "-r" => {
                i += 1;
                if i < args.len() { reference = args[i].to_string(); }
            }
            "--triple" | "-t" => {
                triple = true;
            }
            _ => {}
        }
        i += 1;
    }
    
    if candidate.is_empty() || reference.is_empty() {
        return Err("Both --candidate and --reference are required".to_string());
    }
    
    let metric = MetricType::from_str(&metric_str)
        .ok_or_else(|| format!("Unknown metric: {}. Available: {:?}", metric_str, MetricType::all()))?;
    
    let pair = ScoringPair {
        candidate: candidate.clone(),
        reference: reference.clone(),
    };
    
    println!("Metric: {:?}", metric);
    println!("Candidate: {:?}", candidate);
    println!("Reference: {:?}", reference);
    println!();
    
    match metric {
        MetricType::ExactMatch => {
            let scorer = ExactMatchScorer::case_insensitive();
            if triple {
                let result = scorer.score_and_verify(&pair);
                println!("Reference impl:  {}", result.reference);
                println!("Automaton impl:  {}", result.automaton);
                println!("Circuit impl:    {}", result.circuit);
                println!("Agreement:       {}", result.agreement);
            } else {
                println!("Score: {}", scorer.score_reference(&pair));
            }
        }
        MetricType::TokenF1 => {
            let scorer = TokenF1Scorer::default_scorer();
            if triple {
                let result = scorer.score_and_verify(&pair);
                println!("Reference impl:  P={:.4} R={:.4} F1={:.4}", result.reference.precision, result.reference.recall, result.reference.f1);
                println!("Automaton impl:  P={:.4} R={:.4} F1={:.4}", result.automaton.precision, result.automaton.recall, result.automaton.f1);
                println!("Circuit impl:    P={:.4} R={:.4} F1={:.4}", result.circuit.precision, result.circuit.recall, result.circuit.f1);
                println!("Agreement:       {}", result.agreement);
            } else {
                let score = scorer.score_reference(&pair);
                println!("Precision: {:.4}", score.precision);
                println!("Recall:    {:.4}", score.recall);
                println!("F1:        {:.4}", score.f1);
            }
        }
        MetricType::Bleu => {
            let scorer = BleuScorer::with_smoothing(SmoothingMethod::Add1);
            if triple {
                let result = scorer.score_and_verify(&pair);
                println!("Reference impl:  {:.4}", result.reference.score);
                println!("Automaton impl:  {:.4}", result.automaton.score);
                println!("Circuit impl:    {:.4}", result.circuit.score);
                println!("Agreement:       {}", result.agreement);
            } else {
                let score = scorer.score_reference(&pair);
                println!("BLEU:            {:.4}", score.score);
                println!("Brevity Penalty: {:.4}", score.brevity_penalty);
                for (i, p) in score.precisions.iter().enumerate() {
                    println!("  {}-gram prec:   {:.4}", i + 1, p);
                }
            }
        }
        MetricType::Rouge1 => {
            let scorer = RougeNScorer::rouge1();
            if triple {
                let result = scorer.score_and_verify(&pair);
                println!("Reference impl:  F1={:.4}", result.reference.f1);
                println!("Automaton impl:  F1={:.4}", result.automaton.f1);
                println!("Circuit impl:    F1={:.4}", result.circuit.f1);
                println!("Agreement:       {}", result.agreement);
            } else {
                let score = scorer.score_reference(&pair);
                println!("ROUGE-1 P: {:.4}", score.precision);
                println!("ROUGE-1 R: {:.4}", score.recall);
                println!("ROUGE-1 F: {:.4}", score.f1);
            }
        }
        MetricType::Rouge2 => {
            let scorer = RougeNScorer::rouge2();
            let score = scorer.score_reference(&pair);
            println!("ROUGE-2 P: {:.4}", score.precision);
            println!("ROUGE-2 R: {:.4}", score.recall);
            println!("ROUGE-2 F: {:.4}", score.f1);
        }
        MetricType::RougeL => {
            let scorer = RougeLScorer::default_scorer();
            if triple {
                let result = scorer.score_and_verify(&pair);
                println!("Reference impl:  F1={:.4}", result.reference.f1);
                println!("Automaton impl:  F1={:.4}", result.automaton.f1);
                println!("Circuit impl:    F1={:.4}", result.circuit.f1);
                println!("Agreement:       {}", result.agreement);
            } else {
                let score = scorer.score_reference(&pair);
                println!("ROUGE-L P: {:.4}", score.precision);
                println!("ROUGE-L R: {:.4}", score.recall);
                println!("ROUGE-L F: {:.4}", score.f1);
            }
        }
    }
    
    Ok(())
}

fn cmd_differential_test(args: &[String]) -> Result<(), String> {
    let mut count = 50usize;
    let mut seed = 42u64;
    
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--count" | "-n" => {
                i += 1;
                if i < args.len() {
                    count = args[i].parse().map_err(|_| "Invalid count")?;
                }
            }
            "--seed" | "-s" => {
                i += 1;
                if i < args.len() {
                    seed = args[i].parse().map_err(|_| "Invalid seed")?;
                }
            }
            _ => {}
        }
        i += 1;
    }
    
    println!("Running differential tests with {} pairs (seed={})", count, seed);
    println!();
    
    let mut pairs = standard_test_suite();
    pairs.extend(random_test_pairs(count, seed));
    
    let tester = DifferentialTester::new();
    let reports = tester.test_all_metrics(&pairs);
    
    let mut all_pass = true;
    for (metric, report) in &reports {
        let status = if report.is_perfect() { "PASS" } else { "FAIL" };
        if !report.is_perfect() { all_pass = false; }
        println!("[{}] {}: {}/{} agree ({:.1}%)",
            status, metric, report.agreements, report.total_tests,
            report.agreement_rate * 100.0);
        
        for detail in &report.disagreement_details {
            println!("  Disagreement #{}: ({:?}, {:?}): {}",
                detail.test_index, detail.candidate, detail.reference, detail.description);
        }
    }
    
    println!();
    if all_pass {
        println!("All differential tests PASSED.");
    } else {
        println!("Some differential tests FAILED.");
    }
    
    Ok(())
}

fn cmd_compile_circuit(args: &[String]) -> Result<(), String> {
    let mut metric_str = "exact_match".to_string();
    let mut max_len = 100usize;
    
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--metric" | "-m" => {
                i += 1;
                if i < args.len() { metric_str = args[i].clone(); }
            }
            "--max-length" | "-l" => {
                i += 1;
                if i < args.len() {
                    max_len = args[i].parse().map_err(|_| "Invalid max-length")?;
                }
            }
            _ => {}
        }
        i += 1;
    }
    
    println!("Compiling circuit for metric: {}", metric_str);
    println!("Max input length: {}", max_len);
    
    let circuit = scoring::ScoringCircuit::new();
    // In a real implementation, this would compile the metric to a circuit
    
    let size = estimate_proof_size(max_len * 10, max_len * 5, 128);
    println!("Estimated circuit size: {} constraints", max_len * 10);
    println!("Estimated proof size: {} bytes ({:.1} KB)", size, size as f64 / 1024.0);
    
    Ok(())
}

fn cmd_estimate_size(args: &[String]) -> Result<(), String> {
    let mut constraints = 1000usize;
    let mut wires = 500usize;
    let mut security = 128usize;
    
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--constraints" => {
                i += 1;
                if i < args.len() {
                    constraints = args[i].parse().map_err(|_| "Invalid constraints")?;
                }
            }
            "--wires" => {
                i += 1;
                if i < args.len() {
                    wires = args[i].parse().map_err(|_| "Invalid wires")?;
                }
            }
            "--security" => {
                i += 1;
                if i < args.len() {
                    security = args[i].parse().map_err(|_| "Invalid security")?;
                }
            }
            _ => {}
        }
        i += 1;
    }
    
    let size = estimate_proof_size(constraints, wires, security);
    println!("Proof size estimate:");
    println!("  Constraints:    {}", constraints);
    println!("  Wires:          {}", wires);
    println!("  Security bits:  {}", security);
    println!("  Estimated size: {} bytes ({:.1} KB)", size, size as f64 / 1024.0);
    
    Ok(())
}

fn cmd_batch_score(args: &[String]) -> Result<(), String> {
    let mut metric_str = "bleu".to_string();
    let mut input_file = String::new();
    
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--metric" | "-m" => {
                i += 1;
                if i < args.len() { metric_str = args[i].clone(); }
            }
            "--input" | "-i" => {
                i += 1;
                if i < args.len() { input_file = args[i].to_string(); }
            }
            _ => {}
        }
        i += 1;
    }
    
    if input_file.is_empty() {
        return Err("--input is required for batch scoring".to_string());
    }
    
    let content = fs::read_to_string(&input_file)
        .map_err(|e| format!("Failed to read {}: {}", input_file, e))?;
    
    let pairs: Vec<ScoringPair> = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse JSON: {}", e))?;
    
    println!("Scoring {} pairs with metric: {}", pairs.len(), metric_str);
    
    let metric = MetricType::from_str(&metric_str)
        .ok_or_else(|| format!("Unknown metric: {}", metric_str))?;
    
    match metric {
        MetricType::Bleu => {
            let scorer = BleuScorer::with_smoothing(SmoothingMethod::Add1);
            let corpus_result = scorer.reference_score_corpus(&pairs);
            println!("Corpus BLEU: {:.4}", corpus_result.score);
        }
        MetricType::TokenF1 => {
            let scorer = TokenF1Scorer::default_scorer();
            for (i, pair) in pairs.iter().enumerate() {
                let score = scorer.score_reference(pair);
                println!("[{}] F1={:.4}", i, score.f1);
            }
        }
        _ => {
            println!("Batch scoring for {:?} not yet implemented", metric);
        }
    }
    
    Ok(())
}

fn cmd_hash(args: &[String]) -> Result<(), String> {
    let mut domain = String::new();
    let mut data = String::new();
    
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--domain" | "-d" => {
                i += 1;
                if i < args.len() { domain = args[i].to_string(); }
            }
            "--data" => {
                i += 1;
                if i < args.len() { data = args[i].to_string(); }
            }
            _ => {
                if data.is_empty() {
                    data = args[i].to_string();
                }
            }
        }
        i += 1;
    }
    
    let hasher = if domain.is_empty() {
        SpectaclesHasher::new()
    } else {
        SpectaclesHasher::with_domain(&domain)
    };
    
    let hash = hasher.hash_hex(data.as_bytes());
    println!("{}", hash);
    
    Ok(())
}
