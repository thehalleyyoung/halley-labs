//! Translation-quality metamorphic testing example.
//!
//! Demonstrates:
//! - Testing a translation-like pipeline (tokenizer → encoder → decoder → postprocessor)
//!   modelled here with the built-in stages as a proxy
//! - Using embedding depth and tense change transformations
//! - Computing Behavioral Fragility Index (BFI) per stage
//! - Generating a behavioral atlas
//!
//! Run with: cargo run --example translation_quality

use std::collections::HashMap;

use nlp_models::pipeline::{PipelineAdapter, PipelineBuilder};
use nlp_models::stage::{
    DependencyParserStage, EmbedderStage, PosTaggerStage, TokenizerStage,
};
use metamorphic_core::relation::{MRDefinition, MRType, SemanticEquivalenceMR};
use localization::engine::{LocalizationConfig, LocalizationEngine, SBFLMetric, TestObservation};
use report_gen::bfi::{BFIComputer, BFIInterpretation};
use report_gen::atlas::BehavioralAtlas;
use shared_types::{
    DependencyEdge, DependencyRelation, MetamorphicRelation, PosTag, Sentence, StageId, Token,
};
use transformations::base::TransformationKind;
use transformations::registry::TransformationRegistry;

/// Build a "translation-like" pipeline using available stages as a proxy:
/// tokenizer → POS tagger (encoder) → dep parser (decoder) → embedder (postprocessor).
fn build_translation_pipeline() -> nlp_models::pipeline::Pipeline {
    PipelineBuilder::new("translation-proxy")
        .id("trans-pipeline")
        .add_stage(Box::new(TokenizerStage::new().with_id("tok")))
        .add_stage(Box::new(PosTaggerStage::new().with_id("encoder")))
        .add_stage(Box::new(DependencyParserStage::new().with_id("decoder")))
        .add_stage(Box::new(EmbedderStage::new(64).with_id("postproc")))
        .metadata("task", "translation-quality")
        .build()
}

/// Test sentences covering varied syntactic constructions.
fn test_sentences() -> Vec<(Sentence, &'static str)> {
    vec![
        (
            make_sentence(
                "The engineer designed a robust bridge.",
                &[
                    ("The", PosTag::Det),
                    ("engineer", PosTag::Noun),
                    ("designed", PosTag::Verb),
                    ("a", PosTag::Det),
                    ("robust", PosTag::Adj),
                    ("bridge", PosTag::Noun),
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
            "simple active transitive",
        ),
        (
            make_sentence(
                "The student quickly solved the difficult problem.",
                &[
                    ("The", PosTag::Det),
                    ("student", PosTag::Noun),
                    ("quickly", PosTag::Adv),
                    ("solved", PosTag::Verb),
                    ("the", PosTag::Det),
                    ("difficult", PosTag::Adj),
                    ("problem", PosTag::Noun),
                    (".", PosTag::Punct),
                ],
                &[
                    (3, 3, DependencyRelation::Root),
                    (3, 1, DependencyRelation::Nsubj),
                    (1, 0, DependencyRelation::Det),
                    (3, 2, DependencyRelation::Advmod),
                    (3, 6, DependencyRelation::Dobj),
                    (6, 4, DependencyRelation::Det),
                    (6, 5, DependencyRelation::Amod),
                    (3, 7, DependencyRelation::Punct),
                ],
            ),
            "transitive with adverb",
        ),
        (
            make_sentence(
                "The committee approved the proposal unanimously.",
                &[
                    ("The", PosTag::Det),
                    ("committee", PosTag::Noun),
                    ("approved", PosTag::Verb),
                    ("the", PosTag::Det),
                    ("proposal", PosTag::Noun),
                    ("unanimously", PosTag::Adv),
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

fn make_sentence(
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

fn main() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║   Translation Quality — Metamorphic Testing     ║");
    println!("╚══════════════════════════════════════════════════╝\n");

    // ── 1. Build pipeline ───────────────────────────────────────────────────
    let pipeline = build_translation_pipeline();
    let stage_names: Vec<String> = pipeline.stage_names();
    let stage_ids: Vec<StageId> = pipeline.stage_ids();
    println!(
        "Pipeline: {} stages — {}\n",
        pipeline.stage_count(),
        stage_names.join(" → ")
    );

    // ── 2. Set up transformations ───────────────────────────────────────────
    let transform_registry = TransformationRegistry::default();
    let target_transforms = [
        TransformationKind::Passivization,
        TransformationKind::Topicalization,
        TransformationKind::Clefting,
        TransformationKind::TenseChange,
    ];
    println!(
        "Using {} target transformations: {:?}\n",
        target_transforms.len(),
        target_transforms.iter().map(|t| t.name()).collect::<Vec<_>>()
    );

    // ── 3. Set up MR and engine ─────────────────────────────────────────────
    let semantic_mr = SemanticEquivalenceMR::new(0.2, 0.3, 0.2);

    let config = LocalizationConfig {
        sbfl_metric: SBFLMetric::Ochiai,
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

    // Storage for BFI / atlas computation.
    let num_stages = stage_names.len();
    let mut all_stage_diffs: Vec<Vec<f64>> = vec![Vec::new(); num_stages];
    let mut all_violations: Vec<Vec<bool>> = vec![Vec::new(); num_stages];
    let mut per_transform_diffs: HashMap<String, Vec<Vec<f64>>> = HashMap::new();

    // ── 4. Run metamorphic tests ────────────────────────────────────────────
    println!("─── Running Translation Quality Tests ───\n");

    let corpus = test_sentences();
    let mut test_count = 0usize;
    let mut violation_count = 0usize;

    for (sentence, description) in &corpus {
        println!("  Sentence: {:?} ({})", sentence.raw_text, description);

        for &kind in &target_transforms {
            if let Some(transform) = transform_registry.get(&kind) {
                if !transform.is_applicable(sentence) {
                    continue;
                }
                match transform.apply(sentence) {
                    Ok(tr) if tr.success => {
                        test_count += 1;
                        let transform_name = kind.name();
                        let transformed_text = &tr.transformed.raw_text;

                        let orig_trace = pipeline.execute(&sentence.raw_text).unwrap();
                        let trans_trace = pipeline.execute(transformed_text).unwrap();

                        let orig_final = &orig_trace.per_stage_irs.last().unwrap().ir;
                        let trans_final = &trans_trace.per_stage_irs.last().unwrap().ir;

                        let check = semantic_mr
                            .check_with_detail(orig_final, trans_final)
                            .unwrap();

                        let violated = !check.passed;
                        if violated {
                            violation_count += 1;
                        }

                        // Compute per-stage differentials.
                        let mut diffs: HashMap<String, f64> = HashMap::new();
                        let t_diffs_entry = per_transform_diffs
                            .entry(transform_name.to_string())
                            .or_insert_with(|| vec![Vec::new(); num_stages]);

                        for (k, name) in stage_names.iter().enumerate() {
                            if k < orig_trace.per_stage_irs.len()
                                && k < trans_trace.per_stage_irs.len()
                            {
                                let d = token_diff(
                                    &orig_trace.per_stage_irs[k].ir,
                                    &trans_trace.per_stage_irs[k].ir,
                                );
                                diffs.insert(name.clone(), d);
                                all_stage_diffs[k].push(d);
                                all_violations[k].push(violated);
                                t_diffs_entry[k].push(d);
                            }
                        }

                        engine.record_observation(TestObservation {
                            test_id: format!("trans-{}", test_count),
                            transformation_name: transform_name.to_string(),
                            input_text: sentence.raw_text.clone(),
                            transformed_text: transformed_text.clone(),
                            violation_detected: violated,
                            violation_magnitude: check.violation_magnitude,
                            per_stage_differentials: diffs,
                            execution_time_ms: (orig_trace.total_time_ms
                                + trans_trace.total_time_ms)
                                as f64,
                        });

                        println!(
                            "    {:<28} {} (mag={:.4})",
                            transform_name,
                            if violated { "FAIL" } else { "PASS" },
                            check.violation_magnitude,
                        );
                    }
                    _ => {}
                }
            }
        }
        println!();
    }

    // ── 5. Compute BFI per stage ────────────────────────────────────────────
    println!("─── Behavioral Fragility Index (BFI) ───\n");

    let bfi_computer = BFIComputer::new(1e-6);
    let bfi_results = bfi_computer.compute_all_bfi(&stage_names, &all_stage_diffs);

    for bfi in &bfi_results {
        let interp_icon = match bfi.interpretation {
            BFIInterpretation::Amplifying => "⚠ AMPLIFYING",
            BFIInterpretation::Propagating => "→ Propagating",
            BFIInterpretation::Absorbing => "✓ Absorbing",
            BFIInterpretation::Undefined => "? Undefined",
        };
        println!(
            "  {:<16} BFI={:>8.4}  CI=({:.3}, {:.3})  n={:<4} {}",
            bfi.stage_name,
            bfi.bfi_value,
            bfi.confidence_interval.0,
            bfi.confidence_interval.1,
            bfi.sample_count,
            interp_icon,
        );
    }

    // ── 6. Generate behavioral atlas ────────────────────────────────────────
    println!("\n─── Behavioral Atlas ───\n");

    let suspiciousness: Vec<f64> = {
        let ranking = engine.compute_suspiciousness();
        stage_names
            .iter()
            .map(|name| {
                ranking
                    .rankings
                    .iter()
                    .find(|e| e.stage_name == *name)
                    .map(|e| e.score)
                    .unwrap_or(0.0)
            })
            .collect()
    };

    let atlas = BehavioralAtlas::build(
        &stage_names,
        &all_stage_diffs,
        &per_transform_diffs,
        &all_violations,
        &suspiciousness,
    );

    println!("  Stage Atlas:");
    println!(
        "  {:<16} {:>8} {:>14} {:>12} {:>6}",
        "Stage", "BFI", "Interpretation", "Suspicious", "Rank"
    );
    println!("  {}", "─".repeat(62));
    for entry in atlas.stage_entries() {
        println!(
            "  {:<16} {:>8.4} {:>14} {:>12.4} {:>6}",
            entry.stage_name,
            entry.bfi_value,
            format!("{}", entry.bfi_interpretation),
            entry.suspiciousness_score,
            entry.rank,
        );
    }

    println!("\n  Transformation Atlas:");
    println!(
        "  {:<28} {:>6} {:>10} {:>12}",
        "Transformation", "Tests", "Violations", "Mean Diff"
    );
    println!("  {}", "─".repeat(62));
    for entry in atlas.transformation_entries() {
        println!(
            "  {:<28} {:>6} {:>10} {:>12.4}",
            entry.transformation_name, entry.test_count, entry.violation_count, entry.mean_differential,
        );
    }

    if !atlas.interactions.is_empty() {
        println!("\n  Interaction Matrix (stage × transformation):");
        println!(
            "  {:<16} {:<28} {:>8} {:>10}",
            "Stage", "Transformation", "BFI", "Mean Diff"
        );
        println!("  {}", "─".repeat(68));
        for entry in &atlas.interactions {
            println!(
                "  {:<16} {:<28} {:>8.4} {:>10.4}",
                entry.stage_name,
                entry.transformation_name,
                entry.bfi_value,
                entry.mean_differential,
            );
        }
    }

    // ── 7. Summary ──────────────────────────────────────────────────────────
    println!("\n─── Summary ───\n");
    println!("  Sentences tested:   {}", corpus.len());
    println!("  Total test cases:   {}", test_count);
    println!("  Violations found:   {}", violation_count);
    println!("  Pipeline stages:    {}", num_stages);
    println!("  Transformations:    {}", per_transform_diffs.len());
    println!("\n✓ Translation quality example complete.");
}

fn token_diff(
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
        .filter(|(ta, tb)| ta.text == tb.text)
        .count();
    1.0 - (matching as f64 / max_len as f64)
}
