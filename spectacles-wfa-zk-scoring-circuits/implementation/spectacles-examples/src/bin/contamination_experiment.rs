//! End-to-end contamination detection experiment.
//!
//! Creates "known contaminated" and "clean" scenarios, runs PSI-based
//! detection, and demonstrates that the pipeline distinguishes them.
//! Produces ROC-style analysis across multiple thresholds τ.

use spectacles_core::psi::{
    NGramExtractor, NGramSet, NGramConfig,
    PSIProtocol, PSIResult,
    protocol::PSIConfig,
};
use spectacles_core::scoring::{
    ScoringPair, TripleMetric,
    exact_match::ExactMatchScorer,
    token_f1::TokenF1Scorer,
    bleu::{BleuScorer, SmoothingMethod},
};
use spectacles_core::utils::hash::SpectaclesHasher;
use serde::Serialize;
use std::time::Instant;

#[derive(Debug, Serialize)]
struct ContaminationExperiment {
    timestamp: String,
    description: String,
    scenarios: Vec<ScenarioResult>,
    threshold_analysis: ThresholdAnalysis,
    scoring_impact: ScoringImpact,
    summary: ExperimentSummary,
}

#[derive(Debug, Serialize)]
struct ScenarioResult {
    name: String,
    description: String,
    contamination_fraction: f64,
    num_test_items: usize,
    num_training_items: usize,
    num_contaminated_items: usize,
    psi_result: PSIMetrics,
    detection_correct: bool,
}

#[derive(Debug, Serialize)]
struct PSIMetrics {
    intersection_cardinality: usize,
    set_a_cardinality: usize,
    set_b_cardinality: usize,
    jaccard: f64,
    containment_a: f64,
    contamination_score: f64,
    runtime_ms: u64,
}

#[derive(Debug, Serialize)]
struct ThresholdAnalysis {
    thresholds: Vec<f64>,
    true_positive_rates: Vec<f64>,
    false_positive_rates: Vec<f64>,
    best_threshold: f64,
    best_f1: f64,
    auc_estimate: f64,
}

#[derive(Debug, Serialize)]
struct ScoringImpact {
    clean_scores: MetricScores,
    contaminated_scores: MetricScores,
    score_inflation: MetricScores,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct MetricScores {
    exact_match: f64,
    token_f1: f64,
    bleu: f64,
}

#[derive(Debug, Serialize)]
struct ExperimentSummary {
    total_scenarios: usize,
    detection_accuracy: f64,
    separation_achieved: bool,
    clean_max_contamination_score: f64,
    contaminated_min_contamination_score: f64,
    separation_gap: f64,
}

// Simulated QA benchmark data
fn make_test_benchmark() -> Vec<(String, String)> {
    vec![
        ("What is the capital of France?".into(), "Paris".into()),
        ("What is the largest planet in our solar system?".into(), "Jupiter".into()),
        ("Who wrote Romeo and Juliet?".into(), "William Shakespeare".into()),
        ("What is the speed of light?".into(), "299792458 meters per second".into()),
        ("What is photosynthesis?".into(), "The process by which plants convert sunlight to energy".into()),
        ("What year did World War II end?".into(), "1945".into()),
        ("What is the chemical formula for water?".into(), "H2O".into()),
        ("Who painted the Mona Lisa?".into(), "Leonardo da Vinci".into()),
        ("What is the tallest mountain?".into(), "Mount Everest".into()),
        ("What is DNA?".into(), "Deoxyribonucleic acid, the molecule carrying genetic instructions".into()),
        ("What is the boiling point of water?".into(), "100 degrees Celsius".into()),
        ("Who discovered penicillin?".into(), "Alexander Fleming".into()),
        ("What is the Pythagorean theorem?".into(), "a squared plus b squared equals c squared".into()),
        ("What is the currency of Japan?".into(), "Yen".into()),
        ("What is the smallest prime number?".into(), "2".into()),
        ("Who invented the telephone?".into(), "Alexander Graham Bell".into()),
        ("What is an atom?".into(), "The basic unit of matter".into()),
        ("What causes tides?".into(), "Gravitational pull of the moon and sun".into()),
        ("What is the human body temperature?".into(), "37 degrees Celsius or 98.6 degrees Fahrenheit".into()),
        ("What is the square root of 144?".into(), "12".into()),
    ]
}

fn make_clean_training_data() -> Vec<String> {
    // Training data that does NOT overlap with test questions
    vec![
        "Machine learning is a branch of artificial intelligence".into(),
        "The Eiffel Tower is located in Paris France".into(),
        "Quantum computing uses qubits instead of classical bits".into(),
        "The Amazon rainforest is the largest tropical forest".into(),
        "Neural networks are inspired by biological neurons".into(),
        "The Great Wall of China spans thousands of kilometers".into(),
        "Python is a versatile programming language".into(),
        "Climate change is driven by greenhouse gas emissions".into(),
        "The periodic table organizes chemical elements".into(),
        "Gravity is a fundamental force of nature".into(),
        "Photons are particles of light".into(),
        "The Renaissance was a cultural movement in Europe".into(),
        "Algorithms are step by step procedures for computation".into(),
        "Vaccines work by stimulating the immune system".into(),
        "The Internet connects billions of devices worldwide".into(),
        "Calculus studies rates of change and accumulation".into(),
        "Democracy is a system of government by the people".into(),
        "Enzymes catalyze biochemical reactions".into(),
        "Satellites orbit Earth for communication and observation".into(),
        "The human genome contains about 20000 genes".into(),
    ]
}

fn make_contaminated_training_data(test_data: &[(String, String)], contamination_fraction: f64) -> Vec<String> {
    let mut training = make_clean_training_data();
    let num_to_contaminate = (test_data.len() as f64 * contamination_fraction).ceil() as usize;

    // Insert verbatim copies of test questions into training data
    for i in 0..num_to_contaminate.min(test_data.len()) {
        let (q, a) = &test_data[i];
        training.push(format!("{} {}", q, a));
    }
    training
}

fn run_psi_detection(test_data: &[(String, String)], training_data: &[String], ngram_n: usize) -> PSIResult {
    let config = NGramConfig::word_ngrams(ngram_n);

    // Build n-gram sets from test questions+answers and training data
    let mut test_set = NGramSet::new(config.clone());
    for (q, a) in test_data {
        let combined = format!("{} {}", q, a);
        let extractor = NGramExtractor::new(config.clone());
        for ngram in extractor.extract(&combined) {
            test_set.insert(&ngram);
        }
    }

    let mut train_set = NGramSet::new(config.clone());
    for item in training_data {
        let extractor = NGramExtractor::new(config.clone());
        for ngram in extractor.extract(item) {
            train_set.insert(&ngram);
        }
    }

    // Run PSI
    let psi_config = PSIConfig::default();
    let psi = PSIProtocol::new(psi_config);
    psi.run_local(&test_set, &train_set)
}

fn compute_scores(test_data: &[(String, String)], training_data: &[String]) -> MetricScores {
    let em_scorer = ExactMatchScorer::case_sensitive();
    let f1_scorer = TokenF1Scorer::default_scorer();
    let bleu_scorer = BleuScorer::with_smoothing(SmoothingMethod::Add1);

    // Simulate: for contaminated items, model "memorizes" correct answer;
    // for clean items, model gives a generic response
    let mut em_sum = 0.0;
    let mut f1_sum = 0.0;
    let mut bleu_sum = 0.0;
    let n = test_data.len() as f64;

    let train_text = training_data.join(" ").to_lowercase();

    for (q, a) in test_data {
        let combined = format!("{} {}", q, a).to_lowercase();
        let is_memorized = train_text.contains(&combined);

        // If the model has "seen" this Q+A pair, it produces exact answer
        let model_output = if is_memorized {
            a.clone()
        } else {
            // A plausible but imperfect answer
            let words: Vec<&str> = a.split_whitespace().collect();
            if words.len() > 2 {
                words[..words.len() / 2].join(" ")
            } else {
                "I don't know".to_string()
            }
        };

        let pair = ScoringPair {
            candidate: model_output,
            reference: a.clone(),
        };

        let em_result = em_scorer.score_and_verify(&pair);
        em_sum += if em_result.reference { 1.0 } else { 0.0 };

        let f1_result = f1_scorer.score_and_verify(&pair);
        f1_sum += f1_result.reference.f1;

        let bleu_result = bleu_scorer.score_and_verify(&pair);
        bleu_sum += bleu_result.reference.score;
    }

    MetricScores {
        exact_match: em_sum / n,
        token_f1: f1_sum / n,
        bleu: bleu_sum / n,
    }
}

fn main() {
    env_logger::init();
    let start = Instant::now();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  End-to-End Contamination Detection Experiment");
    println!("═══════════════════════════════════════════════════════════════");

    let test_data = make_test_benchmark();
    println!("Test benchmark: {} QA pairs", test_data.len());

    // Define scenarios with varying contamination levels
    let contamination_levels = vec![
        (0.0,  "clean",            "No test data in training set"),
        (0.05, "minimal_5pct",     "5% of test data leaked into training"),
        (0.10, "low_10pct",        "10% of test data leaked"),
        (0.25, "moderate_25pct",   "25% of test data leaked"),
        (0.50, "high_50pct",       "50% of test data leaked"),
        (0.75, "severe_75pct",     "75% of test data leaked"),
        (1.00, "total_100pct",     "All test data leaked"),
    ];

    let mut scenarios = Vec::new();
    let mut clean_contamination_scores = Vec::new();
    let mut contaminated_contamination_scores = Vec::new();

    for (frac, name, desc) in &contamination_levels {
        let training = make_contaminated_training_data(&test_data, *frac);
        let num_contaminated = (test_data.len() as f64 * frac).ceil() as usize;

        println!("\n▸ Scenario: {} (contamination={:.0}%)", name, frac * 100.0);
        let psi_result = run_psi_detection(&test_data, &training, 5);

        let detected_contaminated = psi_result.contamination_score > 0.05;
        let actually_contaminated = *frac > 0.0;
        let detection_correct = detected_contaminated == actually_contaminated;

        println!("  PSI contamination score: {:.4}", psi_result.contamination_score);
        println!("  Jaccard similarity:      {:.4}", psi_result.jaccard());
        println!("  Detection correct:       {}", detection_correct);

        if *frac == 0.0 {
            clean_contamination_scores.push(psi_result.contamination_score);
        } else {
            contaminated_contamination_scores.push(psi_result.contamination_score);
        }

        scenarios.push(ScenarioResult {
            name: name.to_string(),
            description: desc.to_string(),
            contamination_fraction: *frac,
            num_test_items: test_data.len(),
            num_training_items: training.len(),
            num_contaminated_items: num_contaminated.min(test_data.len()),
            psi_result: PSIMetrics {
                intersection_cardinality: psi_result.intersection_cardinality,
                set_a_cardinality: psi_result.set_a_cardinality,
                set_b_cardinality: psi_result.set_b_cardinality,
                jaccard: psi_result.jaccard(),
                containment_a: psi_result.containment_a(),
                contamination_score: psi_result.contamination_score,
                runtime_ms: psi_result.runtime_ms,
            },
            detection_correct,
        });
    }

    // Threshold (ROC-style) analysis
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Threshold Analysis (ROC)");
    println!("═══════════════════════════════════════════════════════════════");

    let thresholds: Vec<f64> = (0..=100).map(|i| i as f64 * 0.01).collect();
    let mut tpr_vec = Vec::new();
    let mut fpr_vec = Vec::new();
    let mut best_f1 = 0.0;
    let mut best_tau = 0.0;

    let all_scores: Vec<(f64, bool)> = scenarios.iter()
        .map(|s| (s.psi_result.contamination_score, s.contamination_fraction > 0.0))
        .collect();

    let num_positive = all_scores.iter().filter(|(_, is_pos)| *is_pos).count() as f64;
    let num_negative = all_scores.iter().filter(|(_, is_pos)| !*is_pos).count() as f64;

    for &tau in &thresholds {
        let tp = all_scores.iter().filter(|(score, is_pos)| *is_pos && *score > tau).count() as f64;
        let fp = all_scores.iter().filter(|(score, is_pos)| !*is_pos && *score > tau).count() as f64;
        let fn_ = all_scores.iter().filter(|(score, is_pos)| *is_pos && *score <= tau).count() as f64;

        let tpr = if num_positive > 0.0 { tp / num_positive } else { 0.0 };
        let fpr = if num_negative > 0.0 { fp / num_negative } else { 0.0 };

        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = tpr;
        let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };

        if f1 > best_f1 {
            best_f1 = f1;
            best_tau = tau;
        }

        tpr_vec.push(tpr);
        fpr_vec.push(fpr);
    }

    // Estimate AUC using trapezoidal rule (integrate TPR over FPR from right to left)
    // Sort by FPR ascending for proper integration
    let mut roc_points: Vec<(f64, f64)> = fpr_vec.iter().cloned().zip(tpr_vec.iter().cloned()).collect();
    roc_points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    roc_points.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-12);
    let mut auc = 0.0;
    for i in 1..roc_points.len() {
        let dx = roc_points[i].0 - roc_points[i-1].0;
        let avg_y = (roc_points[i].1 + roc_points[i-1].1) / 2.0;
        auc += dx * avg_y;
    }

    println!("  Best threshold τ: {:.2}", best_tau);
    println!("  Best F1 score:    {:.4}", best_f1);
    println!("  AUC estimate:     {:.4}", auc);

    // Scoring impact
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Scoring Impact Analysis");
    println!("═══════════════════════════════════════════════════════════════");

    let clean_training = make_clean_training_data();
    let contaminated_training = make_contaminated_training_data(&test_data, 1.0);

    let clean_scores = compute_scores(&test_data, &clean_training);
    let contaminated_scores = compute_scores(&test_data, &contaminated_training);

    println!("  Clean scenario:        EM={:.3}, F1={:.3}, BLEU={:.3}",
        clean_scores.exact_match, clean_scores.token_f1, clean_scores.bleu);
    println!("  Contaminated scenario: EM={:.3}, F1={:.3}, BLEU={:.3}",
        contaminated_scores.exact_match, contaminated_scores.token_f1, contaminated_scores.bleu);
    println!("  Score inflation:       EM={:+.3}, F1={:+.3}, BLEU={:+.3}",
        contaminated_scores.exact_match - clean_scores.exact_match,
        contaminated_scores.token_f1 - clean_scores.token_f1,
        contaminated_scores.bleu - clean_scores.bleu);

    let scoring_impact = ScoringImpact {
        clean_scores: clean_scores,
        contaminated_scores: contaminated_scores,
        score_inflation: MetricScores {
            exact_match: contaminated_scores.exact_match - clean_scores.exact_match,
            token_f1: contaminated_scores.token_f1 - clean_scores.token_f1,
            bleu: contaminated_scores.bleu - clean_scores.bleu,
        },
    };

    // Summary
    let clean_max = clean_contamination_scores.iter().cloned().fold(0.0f64, f64::max);
    let contam_min = contaminated_contamination_scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let separation_gap = contam_min - clean_max;
    let detection_accuracy = scenarios.iter().filter(|s| s.detection_correct).count() as f64
        / scenarios.len() as f64;

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Experiment Summary");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Scenarios tested:    {}", scenarios.len());
    println!("  Detection accuracy:  {:.1}%", detection_accuracy * 100.0);
    println!("  Clean max score:     {:.4}", clean_max);
    println!("  Contaminated min:    {:.4}", contam_min);
    println!("  Separation gap:      {:.4}", separation_gap);
    println!("  Best τ:              {:.2}", best_tau);

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!("  Wall clock:          {:.1} ms", elapsed);

    let experiment = ContaminationExperiment {
        timestamp: chrono::Utc::now().to_rfc3339(),
        description: "End-to-end contamination detection experiment with PSI-based n-gram overlap detection. Tests whether the pipeline can distinguish clean training data from contaminated training data at varying contamination levels (0-100%).".into(),
        scenarios,
        threshold_analysis: ThresholdAnalysis {
            thresholds: thresholds.iter().step_by(10).cloned().collect(),
            true_positive_rates: tpr_vec.iter().step_by(10).cloned().collect(),
            false_positive_rates: fpr_vec.iter().step_by(10).cloned().collect(),
            best_threshold: best_tau,
            best_f1,
            auc_estimate: auc,
        },
        scoring_impact,
        summary: ExperimentSummary {
            total_scenarios: 7,
            detection_accuracy,
            separation_achieved: separation_gap > 0.0,
            clean_max_contamination_score: clean_max,
            contaminated_min_contamination_score: contam_min,
            separation_gap,
        },
    };

    let json = serde_json::to_string_pretty(&experiment).unwrap();
    std::fs::write("contamination_experiment.json", &json).unwrap();
    println!("\n  Results saved to: contamination_experiment.json");
}
