//! NER (Named Entity Recognition) metamorphic testing example.
//!
//! Demonstrates:
//! - Building an NER pipeline (tokenizer → POS tagger → dependency parser → NER)
//! - Testing entity preservation under passivization, clefting, topicalization
//! - Running the full localize-shrink cycle
//! - Printing minimal counterexamples
//!
//! Run with: cargo run --example ner_testing

use std::collections::HashMap;

use nlp_models::pipeline::{PipelineAdapter, SpacyLikeAdapter};
use metamorphic_core::relation::{EntityPreservationMR, MRRegistry, SemanticEquivalenceMR};
use localization::engine::{LocalizationConfig, LocalizationEngine, SBFLMetric, TestObservation};
use transformations::registry::TransformationRegistry;
use shared_types::{
    DependencyEdge, DependencyRelation, MetamorphicRelation, PosTag, Sentence, StageId, Token,
};
use shrinking::GCHDDShrinker;

/// Build a fully annotated sentence from raw text, tokens with POS tags, and
/// dependency edges so transformations can operate on it.
fn annotated_sentence(
    raw: &str,
    words_pos: &[(&str, PosTag)],
    edges: &[(usize, usize, DependencyRelation)],
) -> Sentence {
    let tokens: Vec<Token> = words_pos
        .iter()
        .enumerate()
        .map(|(i, (w, pos))| Token::new(*w, i).with_pos(*pos))
        .collect();
    let dep_edges: Vec<DependencyEdge> = edges
        .iter()
        .map(|(h, d, r)| DependencyEdge::new(*h, *d, *r))
        .collect();
    let mut s = Sentence::from_tokens(tokens, raw);
    s.dependency_edges = dep_edges;
    s
}

/// Test corpus of entity-bearing sentences with annotations.
fn test_corpus() -> Vec<(Sentence, &'static str)> {
    vec![
        (
            annotated_sentence(
                "John gave Mary a birthday present.",
                &[
                    ("John", PosTag::Noun),
                    ("gave", PosTag::Verb),
                    ("Mary", PosTag::Noun),
                    ("a", PosTag::Det),
                    ("birthday", PosTag::Noun),
                    ("present", PosTag::Noun),
                    (".", PosTag::Punct),
                ],
                &[
                    (1, 1, DependencyRelation::Root),
                    (1, 0, DependencyRelation::Nsubj),
                    (1, 2, DependencyRelation::Iobj),
                    (1, 5, DependencyRelation::Dobj),
                    (5, 3, DependencyRelation::Det),
                    (5, 4, DependencyRelation::Compound),
                    (1, 6, DependencyRelation::Punct),
                ],
            ),
            "dative with named entities",
        ),
        (
            annotated_sentence(
                "The scientist discovered the new element.",
                &[
                    ("The", PosTag::Det),
                    ("scientist", PosTag::Noun),
                    ("discovered", PosTag::Verb),
                    ("the", PosTag::Det),
                    ("new", PosTag::Adj),
                    ("element", PosTag::Noun),
                    (".", PosTag::Punct),
                ],
                &[
                    (2, 2, DependencyRelation::Root),
                    (2, 1, DependencyRelation::Nsubj),
                    (1, 0, DependencyRelation::Det),
                    (2, 5, DependencyRelation::Dobj),
                    (5, 3, DependencyRelation::Det),
                    (5, 4, DependencyRelation::Amod),
                    (2, 6, DependencyRelation::Punct),
                ],
            ),
            "active transitive",
        ),
        (
            annotated_sentence(
                "The cat chased the mouse quickly.",
                &[
                    ("The", PosTag::Det),
                    ("cat", PosTag::Noun),
                    ("chased", PosTag::Verb),
                    ("the", PosTag::Det),
                    ("mouse", PosTag::Noun),
                    ("quickly", PosTag::Adv),
                    (".", PosTag::Punct),
                ],
                &[
                    (2, 2, DependencyRelation::Root),
                    (2, 1, DependencyRelation::Nsubj),
                    (1, 0, DependencyRelation::Det),
                    (2, 4, DependencyRelation::Dobj),
                    (4, 3, DependencyRelation::Det),
                    (2, 5, DependencyRelation::Advmod),
                    (2, 6, DependencyRelation::Punct),
                ],
            ),
            "active transitive with adverb",
        ),
    ]
}

fn main() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║   NER Pipeline — Metamorphic Testing            ║");
    println!("╚══════════════════════════════════════════════════╝\n");

    // ── 1. Build the spaCy-like NER pipeline ────────────────────────────────
    let pipeline = SpacyLikeAdapter::new();
    let stage_ids = pipeline.stages();
    let stage_names: Vec<String> = vec![
        "tokenizer".into(),
        "pos_tagger".into(),
        "dep_parser".into(),
        "ner".into(),
    ];
    println!("Pipeline: {} (4 stages)", pipeline.name());
    println!("  Stages: {}\n", stage_names.join(" → "));

    // ── 2. Set up transformations ───────────────────────────────────────────
    let transform_registry = TransformationRegistry::default();
    println!(
        "Transformation registry: {} transformations loaded.",
        transform_registry.len()
    );

    // ── 3. Set up MRs ──────────────────────────────────────────────────────
    let entity_mr = EntityPreservationMR::new(0.1);
    let semantic_mr = SemanticEquivalenceMR::new(0.2, 0.3, 0.2);
    let mr_registry = MRRegistry::with_defaults();
    println!("MR registry: {} relations loaded.\n", mr_registry.len());

    // ── 4. Set up localization engine ───────────────────────────────────────
    let config = LocalizationConfig {
        sbfl_metric: SBFLMetric::Ensemble,
        suspiciousness_threshold: 0.25,
        ..LocalizationConfig::default()
    };
    let mut engine = LocalizationEngine::with_config(config);
    engine.register_stages(
        stage_ids
            .iter()
            .zip(stage_names.iter())
            .map(|(id, name)| (*id, name.clone()))
            .collect(),
    );

    // ── 5. Run metamorphic tests with tree transformations ──────────────────
    println!("─── Running NER Metamorphic Tests ───\n");

    let corpus = test_corpus();
    let mut violations: Vec<(String, String, String, f64)> = Vec::new();
    let mut test_count = 0usize;

    for (sentence, description) in &corpus {
        let applicable = transform_registry.get_applicable(sentence);
        println!(
            "  Sentence: {:?} ({})",
            sentence.raw_text, description
        );
        println!("    Applicable transformations: {}", applicable.len());

        let results = transform_registry.apply_all_applicable(sentence);

        for (kind, result) in &results {
            test_count += 1;
            let transform_name = kind.name();

            match result {
                Ok(tr) if tr.success => {
                    let transformed_text = &tr.transformed.raw_text;

                    // Run both original and transformed through pipeline.
                    let orig_trace = pipeline.execute(&sentence.raw_text).unwrap();
                    let trans_trace = pipeline.execute(transformed_text).unwrap();

                    let orig_final = &orig_trace.per_stage_irs.last().unwrap().ir;
                    let trans_final = &trans_trace.per_stage_irs.last().unwrap().ir;

                    // Check entity preservation.
                    let entity_check = entity_mr
                        .check_with_detail(orig_final, trans_final)
                        .unwrap();

                    // Check semantic equivalence.
                    let semantic_check = semantic_mr
                        .check_with_detail(orig_final, trans_final)
                        .unwrap();

                    let violated = !entity_check.passed || !semantic_check.passed;
                    let magnitude = entity_check
                        .violation_magnitude
                        .max(semantic_check.violation_magnitude);

                    // Compute per-stage differentials.
                    let mut diffs: HashMap<String, f64> = HashMap::new();
                    for (k, name) in stage_names.iter().enumerate() {
                        if k < orig_trace.per_stage_irs.len()
                            && k < trans_trace.per_stage_irs.len()
                        {
                            let o = &orig_trace.per_stage_irs[k].ir;
                            let t = &trans_trace.per_stage_irs[k].ir;
                            let d = compute_token_diff(o, t);
                            diffs.insert(name.clone(), d);
                        }
                    }

                    engine.record_observation(TestObservation {
                        test_id: format!("ner-{}", test_count),
                        transformation_name: transform_name.to_string(),
                        input_text: sentence.raw_text.clone(),
                        transformed_text: transformed_text.clone(),
                        violation_detected: violated,
                        violation_magnitude: magnitude,
                        per_stage_differentials: diffs,
                        execution_time_ms: (orig_trace.total_time_ms
                            + trans_trace.total_time_ms) as f64,
                    });

                    let status = if violated { "FAIL" } else { "PASS" };
                    println!(
                        "    [{:>3}] {:<28} {} (entity={:.3}, sem={:.3})",
                        test_count,
                        transform_name,
                        status,
                        entity_check.violation_magnitude,
                        semantic_check.violation_magnitude,
                    );

                    if violated {
                        let mr_name = if !entity_check.passed {
                            "EntityPreservation"
                        } else {
                            "SemanticEquivalence"
                        };
                        violations.push((
                            sentence.raw_text.clone(),
                            transform_name.to_string(),
                            mr_name.to_string(),
                            magnitude,
                        ));
                    }
                }
                Ok(_) => {
                    println!("    [{:>3}] {:<28} SKIP (precondition)", test_count, transform_name);
                }
                Err(e) => {
                    println!("    [{:>3}] {:<28} ERR  ({})", test_count, transform_name, e);
                }
            }
        }
        println!();
    }

    // ── 6. Fault localization ───────────────────────────────────────────────
    println!("─── Fault Localization (Ensemble SBFL) ───\n");
    let ranking = engine.compute_suspiciousness();

    for entry in &ranking.rankings {
        let bar_len = (entry.score * 40.0) as usize;
        let bar: String = "▓".repeat(bar_len);
        let indicator = if entry.rank == 1 { " ← MOST SUSPICIOUS" } else { "" };
        println!(
            "  #{} {:<16} score={:.4}  {}{}",
            entry.rank, entry.stage_name, entry.score, bar, indicator
        );
    }

    // ── 7. Shrink violations to minimal counterexamples ─────────────────────
    if !violations.is_empty() {
        println!("\n─── Minimal Counterexamples (GCHDD Shrinking) ───\n");
        let shrinker = GCHDDShrinker::new(10_000);

        for (i, (text, transform, mr, mag)) in violations.iter().enumerate() {
            let result = shrinker.shrink(text, transform, mr).unwrap();
            println!("  Counterexample #{}:", i + 1);
            println!("    Original:     {:?}", text);
            println!("    Transform:    {}", transform);
            println!("    Violated MR:  {}", mr);
            println!("    Magnitude:    {:.4}", mag);
            println!("    Shrunk input: {:?}", result.shrunk_text);
            println!("    Shrink steps: {}", result.shrink_steps);
            println!();
        }
    }

    // ── 8. Summary ──────────────────────────────────────────────────────────
    println!("─── Summary ───\n");
    println!("  Sentences tested:  {}", corpus.len());
    println!("  Total test cases:  {}", test_count);
    println!("  Violations found:  {}", violations.len());
    if let Some(top) = ranking.rankings.first() {
        println!(
            "  Most suspicious:   {} (score={:.4})",
            top.stage_name, top.score
        );
    }
    println!("\n✓ NER testing example complete.");
}

/// Simple token-level differential: 1 - (matching tokens / max tokens).
fn compute_token_diff(
    a: &shared_types::IntermediateRepresentation,
    b: &shared_types::IntermediateRepresentation,
) -> f64 {
    let max_len = a.sentence.tokens.len().max(b.sentence.tokens.len());
    if max_len == 0 {
        return 0.0;
    }
    let matching = a
        .sentence
        .tokens
        .iter()
        .zip(b.sentence.tokens.iter())
        .filter(|(ta, tb)| ta.text == tb.text && ta.pos_tag == tb.pos_tag)
        .count();
    1.0 - (matching as f64 / max_len as f64)
}
