use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};

use shared_types::{
    DependencyEdge, DependencyRelation, DistanceMetric, IRType, IntermediateRepresentation,
    PosTag, Sentence, StageId, Token,
};
use nlp_models::pipeline::{
    HuggingFaceLikeAdapter, PipelineAdapter, SpacyLikeAdapter,
    StanzaLikeAdapter,
};
use differential::{DifferentialComputer, DifferentialTimeSeries, StageDifferential};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
fn make_sentence(text: &str) -> Sentence {
    let words: Vec<&str> = text.split_whitespace().collect();
    let tokens: Vec<Token> = words
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let pos = match i % 5 {
                0 => PosTag::Det,
                1 => PosTag::Noun,
                2 => PosTag::Verb,
                3 => PosTag::Adj,
                _ => PosTag::Adv,
            };
            Token::new(*w, i).with_pos(pos).with_lemma(w.to_lowercase())
        })
        .collect();
    let mut edges = Vec::new();
    if tokens.len() > 1 {
        edges.push(DependencyEdge::new(0, 0, DependencyRelation::Root));
        for i in 1..tokens.len() {
            edges.push(DependencyEdge::new(0, i, DependencyRelation::Dep));
        }
    }
    let mut s = Sentence::from_tokens(tokens, text);
    s.dependency_edges = edges;
    s
}

fn sample_inputs() -> Vec<&'static str> {
    vec![
        "The cat sat on the mat.",
        "A clever scientist discovered a new formula quickly in the laboratory.",
        "The quick brown fox jumps over the lazy sleeping dog near a tall old tree in the dark green forest.",
        "She gave the student a book about the history of the ancient world and its many civilizations.",
    ]
}

fn make_ir(text: &str) -> IntermediateRepresentation {
    let sentence = make_sentence(text);
    IntermediateRepresentation::new(IRType::Tokenized, sentence)
        .with_confidence(0.95)
}

// =========================================================================
// Benchmark group 1: Pipeline execution throughput
// =========================================================================
fn bench_pipeline_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/execution");
    group.sample_size(30);

    let inputs = sample_inputs();
    let spacy = SpacyLikeAdapter::new();
    let hf = HuggingFaceLikeAdapter::new();
    let stanza = StanzaLikeAdapter::new();

    for input in &inputs {
        let n_words = input.split_whitespace().count();

        group.throughput(Throughput::Elements(n_words as u64));

        group.bench_with_input(
            BenchmarkId::new("spacy", n_words),
            input,
            |b, text| {
                b.iter(|| black_box(spacy.execute(text)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("huggingface", n_words),
            input,
            |b, text| {
                b.iter(|| black_box(hf.execute(text)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("stanza", n_words),
            input,
            |b, text| {
                b.iter(|| black_box(stanza.execute(text)))
            },
        );
    }

    group.finish();
}

// =========================================================================
// Benchmark group 2: Pipeline prefix execution (partial pipeline)
// =========================================================================
fn bench_pipeline_prefix(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/prefix");
    group.sample_size(30);

    let spacy = SpacyLikeAdapter::new();
    let stages = spacy.stages();
    let input = "The clever scientist discovered a new formula in the laboratory.";

    // Execute up to each stage
    for (i, stage_id) in stages.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("spacy_prefix", i + 1),
            &(input, stage_id),
            |b, (text, stage_id)| {
                b.iter(|| black_box(spacy.execute_prefix(text, stage_id)))
            },
        );
    }

    group.finish();
}

// =========================================================================
// Benchmark group 3: IR capture overhead
// =========================================================================
fn bench_ir_capture_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/ir_capture");
    group.sample_size(30);

    let spacy = SpacyLikeAdapter::new();
    let hf = HuggingFaceLikeAdapter::new();
    let input = "The quick brown fox jumps over the lazy dog near a tall old tree.";

    // Measure get_ir_at_stage for each stage (captures IR snapshot)
    let spacy_stages = spacy.stages();
    for (i, stage_id) in spacy_stages.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("spacy_ir_at_stage", i),
            &(input, stage_id),
            |b, (text, sid)| {
                b.iter(|| black_box(spacy.get_ir_at_stage(text, sid)))
            },
        );
    }

    let hf_stages = hf.stages();
    for (i, stage_id) in hf_stages.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("hf_ir_at_stage", i),
            &(input, stage_id),
            |b, (text, sid)| {
                b.iter(|| black_box(hf.get_ir_at_stage(text, sid)))
            },
        );
    }

    group.finish();
}

// =========================================================================
// Benchmark group 4: Distance computation between IR pairs
// =========================================================================
fn bench_distance_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/distance");
    group.sample_size(50);

    let computer = DifferentialComputer::new();

    let originals = [
        "The cat sat on the mat.",
        "A scientist discovered a formula.",
        "The quick brown fox jumps over the lazy dog near a tall tree in the forest.",
    ];
    let transformed = [
        "The mat was sat on by the cat.",
        "A formula was discovered by a scientist.",
        "Over the lazy dog near a tall tree in the forest the quick brown fox jumps.",
    ];

    let stage_id = StageId::new("tokenizer");
    for (_i, (orig, trans)) in originals.iter().zip(transformed.iter()).enumerate() {
        let ir_orig = make_ir(orig);
        let ir_trans = make_ir(trans);
        let n_words = orig.split_whitespace().count();

        group.throughput(Throughput::Elements(n_words as u64));
        group.bench_with_input(
            BenchmarkId::new("stage_differential", n_words),
            &(ir_orig, ir_trans),
            |b, (orig, trans)| {
                b.iter(|| {
                    black_box(
                        computer.compute_stage_differential(&stage_id, 0, orig, trans),
                    )
                })
            },
        );
    }

    // Batch computation
    let pairs: Vec<(IntermediateRepresentation, IntermediateRepresentation)> = originals
        .iter()
        .zip(transformed.iter())
        .map(|(o, t)| (make_ir(o), make_ir(t)))
        .collect();

    group.bench_function("batch_compute_3pairs", |b| {
        b.iter(|| black_box(computer.batch_compute(&stage_id, 0, &pairs)))
    });

    group.finish();
}

// =========================================================================
// Benchmark group 5: Behavioral Fragility Index computation
// =========================================================================
fn bench_fragility_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/fragility_index");
    group.sample_size(30);

    let computer = DifferentialComputer::new();

    // Build a differential time series with varying number of stages
    for &n_stages in &[3usize, 5, 7] {
        let stage_ids: Vec<StageId> = (0..n_stages)
            .map(|i| StageId::new(&format!("stage_{i}")))
            .collect();

        // Simulate computing diffs for an original/transformed pair across stages
        let ir_orig = make_ir("The cat sat on the mat.");
        let ir_trans = make_ir("The mat was sat on by the cat.");

        group.bench_with_input(
            BenchmarkId::new("bfi_stages", n_stages),
            &(stage_ids.clone(), ir_orig.clone(), ir_trans.clone()),
            |b, (sids, orig, trans)| {
                b.iter(|| {
                    let mut ts = DifferentialTimeSeries::new("bench_test");
                    for (k, sid) in sids.iter().enumerate() {
                        if let Ok(diff) = computer.compute_stage_differential(sid, k, orig, trans) {
                            ts.push(diff);
                        }
                    }
                    let bfi = ts.fragility_indices();
                    let max_jump = ts.max_jump_stage();
                    black_box((bfi, max_jump))
                })
            },
        );
    }

    group.finish();
}

// =========================================================================
// Benchmark group 6: Cumulative delta computation
// =========================================================================
fn bench_cumulative_deltas(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/cumulative_deltas");
    group.sample_size(50);

    for &n_stages in &[3usize, 5, 7, 10] {
        // Pre-build a time series with synthetic differentials
        let mut ts = DifferentialTimeSeries::new("bench");
        for k in 0..n_stages {
            let diff = StageDifferential::new(
                StageId::new(&format!("stage_{k}")),
                k,
                0.1 * (k as f64 + 1.0),
                DistanceMetric::Jaccard,
            );
            ts.push(diff);
        }

        group.bench_with_input(
            BenchmarkId::new("cumulative", n_stages),
            &ts,
            |b, ts| {
                b.iter(|| black_box(ts.cumulative_deltas()))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fragility_indices", n_stages),
            &ts,
            |b, ts| {
                b.iter(|| black_box(ts.fragility_indices()))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("max_jump", n_stages),
            &ts,
            |b, ts| {
                b.iter(|| black_box(ts.max_jump_stage()))
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_pipeline_execution,
    bench_pipeline_prefix,
    bench_ir_capture_overhead,
    bench_distance_computation,
    bench_fragility_index,
    bench_cumulative_deltas,
);
criterion_main!(benches);
