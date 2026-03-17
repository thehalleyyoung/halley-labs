//! Quick-start example: minimal metamorphic test on an NLP pipeline.
//!
//! Run with: cargo run --example quick_start

use nlp_models::pipeline::{HuggingFaceLikeAdapter, PipelineAdapter};
use metamorphic_core::relation::{MRDefinition, MRType, NegationFlipMR};
use shared_types::MetamorphicRelation;

fn main() {
    println!("=== NLP Metamorphic Localizer — Quick Start ===\n");

    // 1. Build a HuggingFace-like sentiment pipeline (tokenizer → embedder → classifier).
    let pipeline = HuggingFaceLikeAdapter::new();
    println!("Pipeline stages: {:?}", pipeline.stages());

    // 2. Run the original input through the pipeline.
    let original_text = "The movie was absolutely wonderful and heartwarming.";
    let original_trace = pipeline.execute(original_text).unwrap();
    println!(
        "Original: {:?} → {} stages, {}ms",
        original_text,
        original_trace.per_stage_irs.len(),
        original_trace.total_time_ms
    );

    // 3. Apply a negation transformation (manually here for simplicity).
    let transformed_text = "The movie was not absolutely wonderful and heartwarming.";
    let transformed_trace = pipeline.execute(transformed_text).unwrap();
    println!(
        "Transformed: {:?} → {} stages, {}ms",
        transformed_text,
        transformed_trace.per_stage_irs.len(),
        transformed_trace.total_time_ms
    );

    // 4. Check a metamorphic relation — negation should flip sentiment.
    let mr = NegationFlipMR::new();
    let original_ir = &original_trace.per_stage_irs.last().unwrap().ir;
    let transformed_ir = &transformed_trace.per_stage_irs.last().unwrap().ir;

    let detail = mr.check_with_detail(original_ir, transformed_ir).unwrap();
    println!("\n--- MR Check: {} ---", mr.name());
    println!("  Passed:    {}", detail.passed);
    println!("  Expected:  {}", detail.expected);
    println!("  Actual:    {}", detail.actual);
    println!("  Magnitude: {:.4}", detail.violation_magnitude);
    println!("  Reason:    {}", detail.explanation);

    // 5. Also check with an MRDefinition for a sentiment-preservation relation.
    let sentiment_mr = MRDefinition::new(
        "sentiment-pres",
        "Sentiment Preservation",
        MRType::SentimentPreservation,
        0.1,
        "Synonym substitution should preserve sentiment",
    );
    let synonym_text = "The film was absolutely wonderful and heartwarming.";
    let synonym_trace = pipeline.execute(synonym_text).unwrap();
    let synonym_ir = &synonym_trace.per_stage_irs.last().unwrap().ir;

    let detail2 = sentiment_mr
        .check_with_detail(original_ir, synonym_ir)
        .unwrap();
    println!("\n--- MR Check: {} ---", sentiment_mr.name);
    println!("  Passed:    {}", detail2.passed);
    println!("  Magnitude: {:.4}", detail2.violation_magnitude);
    println!("  Reason:    {}", detail2.explanation);

    println!("\n✓ Quick start complete.");
}
