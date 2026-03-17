//! Sentiment analysis metamorphic testing example.
//!
//! Demonstrates:
//! - Building a multi-stage NLP pipeline for sentiment (tokenizer → embedder → classifier)
//! - Defining metamorphic relations (negation should flip, synonym should preserve)
//! - Applying transformations and checking MR violations
//! - Running fault localization on detected violations
//! - Printing formatted output showing which stage caused the issue
//!
//! Run with: cargo run --example sentiment_analysis

use std::collections::HashMap;

use nlp_models::pipeline::{HuggingFaceLikeAdapter, PipelineAdapter};
use metamorphic_core::relation::{
    MRDefinition, MRRegistry, MRType, NegationFlipMR, SentimentPreservationMR,
};
use localization::engine::{LocalizationConfig, LocalizationEngine, SBFLMetric, TestObservation};
use shared_types::{MetamorphicRelation, StageId};
use shrinking::GCHDDShrinker;

/// Test sentences with expected sentiment polarity.
const TEST_SENTENCES: &[(&str, &str)] = &[
    ("The movie was absolutely wonderful and heartwarming.", "positive"),
    ("This restaurant serves terrible food with awful service.", "negative"),
    ("The weather today is perfectly fine.", "neutral"),
    ("I love the brilliant performances in this amazing show.", "positive"),
    ("The product broke immediately and customer support was unhelpful.", "negative"),
];

/// Manual negation transformation for demonstration.
fn negate(sentence: &str) -> String {
    sentence
        .replace("was absolutely", "was not absolutely")
        .replace("serves terrible", "does not serve terrible")
        .replace("is perfectly", "is not perfectly")
        .replace("love the brilliant", "do not love the brilliant")
        .replace("broke immediately", "did not break immediately")
}

/// Manual synonym substitution for demonstration.
fn synonym_substitute(sentence: &str) -> String {
    sentence
        .replace("wonderful", "fantastic")
        .replace("terrible", "dreadful")
        .replace("brilliant", "outstanding")
        .replace("broke", "malfunctioned")
        .replace("amazing", "remarkable")
}

fn main() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║   Sentiment Analysis — Metamorphic Testing      ║");
    println!("╚══════════════════════════════════════════════════╝\n");

    // ── 1. Build the pipeline ───────────────────────────────────────────────
    let pipeline = HuggingFaceLikeAdapter::new();
    let stage_ids = pipeline.stages();
    let stage_names: Vec<String> = vec![
        "tokenizer".into(),
        "embedder".into(),
        "sentiment_classifier".into(),
    ];
    println!(
        "Pipeline: {} stages — {:?}\n",
        stage_ids.len(),
        stage_names
    );

    // ── 2. Register metamorphic relations ───────────────────────────────────
    let mut mr_registry = MRRegistry::with_defaults();
    mr_registry.register(MRDefinition::new(
        "negation-flip",
        "Negation Flip",
        MRType::NegationFlip,
        0.0,
        "Negation insertion should flip the sentiment label",
    ));

    let negation_mr = NegationFlipMR::new();
    let synonym_mr = SentimentPreservationMR::new(0.15);

    println!("Registered {} metamorphic relations.\n", mr_registry.len());

    // ── 3. Set up the fault localization engine ─────────────────────────────
    let config = LocalizationConfig {
        sbfl_metric: SBFLMetric::Ochiai,
        suspiciousness_threshold: 0.3,
        max_peeling_rounds: 5,
        ..LocalizationConfig::default()
    };
    let mut engine = LocalizationEngine::with_config(config);

    let stages_for_engine: Vec<(StageId, String)> = stage_ids
        .iter()
        .zip(stage_names.iter())
        .map(|(id, name)| (*id, name.clone()))
        .collect();
    engine.register_stages(stages_for_engine);

    // ── 4. Run metamorphic tests ────────────────────────────────────────────
    println!("─── Running Metamorphic Tests ───\n");

    let mut violations: Vec<(String, String, String, f64)> = Vec::new();

    for (idx, &(sentence, _expected_sentiment)) in TEST_SENTENCES.iter().enumerate() {
        let original_trace = pipeline.execute(sentence).unwrap();
        let original_final_ir = &original_trace.per_stage_irs.last().unwrap().ir;

        // Test 1: Negation should flip sentiment.
        let negated = negate(sentence);
        let negated_trace = pipeline.execute(&negated).unwrap();
        let negated_final_ir = &negated_trace.per_stage_irs.last().unwrap().ir;

        let neg_detail = negation_mr
            .check_with_detail(original_final_ir, negated_final_ir)
            .unwrap();

        // Compute per-stage differentials for the localization engine.
        let mut per_stage_diffs: HashMap<String, f64> = HashMap::new();
        for (k, name) in stage_names.iter().enumerate() {
            let orig_ir = &original_trace.per_stage_irs[k].ir;
            let trans_ir = &negated_trace.per_stage_irs[k].ir;
            let diff = if orig_ir.sentence.tokens.len() != trans_ir.sentence.tokens.len() {
                0.5
            } else {
                let matching = orig_ir
                    .sentence
                    .tokens
                    .iter()
                    .zip(trans_ir.sentence.tokens.iter())
                    .filter(|(a, b)| a.text == b.text)
                    .count();
                1.0 - (matching as f64 / orig_ir.sentence.tokens.len().max(1) as f64)
            };
            per_stage_diffs.insert(name.clone(), diff);
        }

        engine.record_observation(TestObservation {
            test_id: format!("neg-{}", idx),
            transformation_name: "NegationInsertion".into(),
            input_text: sentence.to_string(),
            transformed_text: negated.clone(),
            violation_detected: !neg_detail.passed,
            violation_magnitude: neg_detail.violation_magnitude,
            per_stage_differentials: per_stage_diffs,
            execution_time_ms: (original_trace.total_time_ms + negated_trace.total_time_ms) as f64,
        });

        println!(
            "  [neg-{}] Negation MR: {} | magnitude={:.4}",
            idx,
            if neg_detail.passed { "PASS" } else { "FAIL" },
            neg_detail.violation_magnitude,
        );

        if !neg_detail.passed {
            violations.push((
                sentence.to_string(),
                "NegationInsertion".into(),
                "NegationFlip".into(),
                neg_detail.violation_magnitude,
            ));
        }

        // Test 2: Synonym substitution should preserve sentiment.
        let synonym = synonym_substitute(sentence);
        let synonym_trace = pipeline.execute(&synonym).unwrap();
        let synonym_final_ir = &synonym_trace.per_stage_irs.last().unwrap().ir;

        let syn_detail = synonym_mr
            .check_with_detail(original_final_ir, synonym_final_ir)
            .unwrap();

        let mut per_stage_diffs2: HashMap<String, f64> = HashMap::new();
        for (k, name) in stage_names.iter().enumerate() {
            let orig_ir = &original_trace.per_stage_irs[k].ir;
            let trans_ir = &synonym_trace.per_stage_irs[k].ir;
            let diff = if orig_ir.sentence.tokens.len() != trans_ir.sentence.tokens.len() {
                0.5
            } else {
                let matching = orig_ir
                    .sentence
                    .tokens
                    .iter()
                    .zip(trans_ir.sentence.tokens.iter())
                    .filter(|(a, b)| a.text == b.text)
                    .count();
                1.0 - (matching as f64 / orig_ir.sentence.tokens.len().max(1) as f64)
            };
            per_stage_diffs2.insert(name.clone(), diff);
        }

        engine.record_observation(TestObservation {
            test_id: format!("syn-{}", idx),
            transformation_name: "SynonymSubstitution".into(),
            input_text: sentence.to_string(),
            transformed_text: synonym.clone(),
            violation_detected: !syn_detail.passed,
            violation_magnitude: syn_detail.violation_magnitude,
            per_stage_differentials: per_stage_diffs2,
            execution_time_ms: (original_trace.total_time_ms + synonym_trace.total_time_ms) as f64,
        });

        println!(
            "  [syn-{}] Synonym MR:  {} | magnitude={:.4}",
            idx,
            if syn_detail.passed { "PASS" } else { "FAIL" },
            syn_detail.violation_magnitude,
        );

        if !syn_detail.passed {
            violations.push((
                sentence.to_string(),
                "SynonymSubstitution".into(),
                "SentimentPreservation".into(),
                syn_detail.violation_magnitude,
            ));
        }
    }

    // ── 5. Compute suspiciousness rankings ──────────────────────────────────
    println!("\n─── Fault Localization Results ───\n");

    let ranking = engine.compute_suspiciousness();
    println!("  Suspiciousness ranking (Ochiai):");
    for entry in &ranking.rankings {
        let bar_len = (entry.score * 30.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!(
            "    #{} {:<24} {:.4}  {}",
            entry.rank, entry.stage_name, entry.score, bar
        );
    }

    // ── 6. Shrink violations to minimal counterexamples ─────────────────────
    if !violations.is_empty() {
        println!("\n─── Counterexample Shrinking ───\n");
        let shrinker = GCHDDShrinker::new(10_000);

        for (text, transform, mr_name, magnitude) in &violations {
            let result = shrinker.shrink(text, transform, mr_name).unwrap();
            println!("  Violation: {} on {:?}", mr_name, text);
            println!("    Transform:    {}", transform);
            println!("    Magnitude:    {:.4}", magnitude);
            println!("    Shrunk to:    {:?}", result.shrunk_text);
            println!("    Steps:        {}", result.shrink_steps);
            println!();
        }
    }

    // ── 7. Summary ──────────────────────────────────────────────────────────
    println!("─── Summary ───\n");
    println!(
        "  Total tests:    {}",
        TEST_SENTENCES.len() * 2
    );
    println!("  Violations:     {}", violations.len());
    if let Some(top) = ranking.rankings.first() {
        println!(
            "  Most suspicious stage: {} (score={:.4})",
            top.stage_name, top.score
        );
    }
    println!("\n✓ Sentiment analysis example complete.");
}
