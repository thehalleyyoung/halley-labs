use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};

use shared_types::{DependencyEdge, DependencyRelation, PosTag, Sentence, Token};
use shrinking::{
    AlwaysAcceptOracle, GCHDDConfig, GCHDDEngine, ShrinkableTree, ShrinkingOracle,
};
use grammar_checker::{GrammarChecker, UnificationEngine};

// ---------------------------------------------------------------------------
// Helper: build a sentence with N words
// ---------------------------------------------------------------------------
fn sentence_of_length(n: usize) -> Sentence {
    let base_words = [
        ("The", PosTag::Det),
        ("quick", PosTag::Adj),
        ("brown", PosTag::Adj),
        ("fox", PosTag::Noun),
        ("jumps", PosTag::Verb),
        ("over", PosTag::Prep),
        ("the", PosTag::Det),
        ("lazy", PosTag::Adj),
        ("sleeping", PosTag::Adj),
        ("dog", PosTag::Noun),
        ("near", PosTag::Prep),
        ("a", PosTag::Det),
        ("tall", PosTag::Adj),
        ("old", PosTag::Adj),
        ("tree", PosTag::Noun),
        ("in", PosTag::Prep),
        ("the", PosTag::Det),
        ("dark", PosTag::Adj),
        ("green", PosTag::Adj),
        ("forest", PosTag::Noun),
    ];

    let mut tokens = Vec::with_capacity(n);
    for i in 0..n {
        let (word, pos) = base_words[i % base_words.len()];
        let token_text = if i < base_words.len() {
            word.to_string()
        } else {
            format!("{word}{}", i / base_words.len())
        };
        tokens.push(Token::new(token_text, i).with_pos(pos));
    }

    let raw_text: String = tokens.iter().map(|t| t.text.as_str()).collect::<Vec<_>>().join(" ");

    let mut edges = Vec::new();
    if n > 0 {
        // Simple star dependency tree rooted at the first verb (index 4 or 0)
        let root_idx = if n > 4 { 4 } else { 0 };
        edges.push(DependencyEdge::new(root_idx, root_idx, DependencyRelation::Root));
        for i in 0..n {
            if i != root_idx {
                let rel = match tokens[i].pos_tag {
                    Some(PosTag::Noun) => DependencyRelation::Nsubj,
                    Some(PosTag::Det) => DependencyRelation::Det,
                    Some(PosTag::Adj) => DependencyRelation::Amod,
                    Some(PosTag::Prep) => DependencyRelation::Prep,
                    Some(PosTag::Adv) => DependencyRelation::Advmod,
                    _ => DependencyRelation::Dep,
                };
                edges.push(DependencyEdge::new(root_idx, i, rel));
            }
        }
    }

    let mut s = Sentence::from_tokens(tokens, raw_text);
    s.dependency_edges = edges;
    s
}

/// Oracle that accepts candidates longer than a minimum and randomly preserves violations.
struct BenchOracle {
    min_words: usize,
}

impl ShrinkingOracle for BenchOracle {
    fn check_validity(&self, sentence: &str) -> bool {
        sentence.split_whitespace().count() >= self.min_words
    }
    fn check_applicability(&self, _sentence: &str) -> bool {
        true
    }
    fn check_violation_preserved(&self, sentence: &str) -> bool {
        // Simulate: violation preserved if sentence still contains "fox" or is long enough
        sentence.contains("fox") || sentence.split_whitespace().count() > 3
    }
}

// =========================================================================
// Benchmark group 1: GCHDD shrinking on varying sentence lengths
// =========================================================================
fn bench_gchdd_shrinking(c: &mut Criterion) {
    let mut group = c.benchmark_group("shrinking/gchdd");
    group.sample_size(10);

    for &n_words in &[10usize, 20, 40, 80] {
        let sentence = sentence_of_length(n_words);
        let tree = ShrinkableTree::from_sentence(&sentence);
        let oracle = BenchOracle { min_words: 3 };

        group.throughput(Throughput::Elements(n_words as u64));
        group.bench_with_input(
            BenchmarkId::new("sentence_words", n_words),
            &(tree, &oracle as &dyn ShrinkingOracle),
            |b, (tree, oracle)| {
                b.iter(|| {
                    let mut engine = GCHDDEngine::new(GCHDDConfig {
                        max_iterations: 200,
                        timeout_seconds: 5,
                        enable_binary_search: false,
                        min_tree_size: 2,
                        max_attempts_per_node: 10,
                    });
                    black_box(engine.shrink(tree, *oracle))
                })
            },
        );
    }
    group.finish();
}

// =========================================================================
// Benchmark group 2: GCHDD with binary search enabled
// =========================================================================
fn bench_gchdd_binary_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("shrinking/gchdd_binary");
    group.sample_size(10);

    for &n_words in &[20usize, 40, 80] {
        let sentence = sentence_of_length(n_words);
        let tree = ShrinkableTree::from_sentence(&sentence);
        let oracle = BenchOracle { min_words: 3 };

        group.bench_with_input(
            BenchmarkId::new("binary_search", n_words),
            &(tree, &oracle as &dyn ShrinkingOracle),
            |b, (tree, oracle)| {
                b.iter(|| {
                    let mut engine = GCHDDEngine::new(GCHDDConfig {
                        max_iterations: 200,
                        timeout_seconds: 5,
                        enable_binary_search: true,
                        min_tree_size: 2,
                        max_attempts_per_node: 10,
                    });
                    black_box(engine.shrink(tree, *oracle))
                })
            },
        );
    }
    group.finish();
}

// =========================================================================
// Benchmark group 3: Always-accept oracle baseline
// =========================================================================
fn bench_gchdd_always_accept(c: &mut Criterion) {
    let mut group = c.benchmark_group("shrinking/always_accept");
    group.sample_size(10);

    for &n_words in &[10usize, 20, 40] {
        let sentence = sentence_of_length(n_words);
        let tree = ShrinkableTree::from_sentence(&sentence);
        let oracle = AlwaysAcceptOracle;

        group.bench_with_input(
            BenchmarkId::new("words", n_words),
            &(tree,),
            |b, (tree,)| {
                b.iter(|| {
                    let mut engine = GCHDDEngine::with_default_config();
                    black_box(engine.shrink(tree, &oracle))
                })
            },
        );
    }
    group.finish();
}

// =========================================================================
// Benchmark group 4: Grammar validity checking throughput
// =========================================================================
fn bench_grammar_validity(c: &mut Criterion) {
    let mut group = c.benchmark_group("shrinking/grammar_check");
    group.sample_size(30);

    let checker = GrammarChecker::new();

    let sentences = [
        "The cat sits on the mat.",
        "A clever scientist discovered a new formula quickly in the laboratory.",
        "The quick brown fox jumps over the lazy sleeping dog near a tall old tree.",
        "She gave the student a book about the history of the ancient world.",
    ];

    for (_i, sent) in sentences.iter().enumerate() {
        let n_words = sent.split_whitespace().count();
        group.throughput(Throughput::Elements(n_words as u64));
        group.bench_with_input(
            BenchmarkId::new("sentence", n_words),
            sent,
            |b, s| {
                b.iter(|| black_box(checker.check(s)))
            },
        );
    }

    group.finish();
}

// =========================================================================
// Benchmark group 5: Feature unification performance
// =========================================================================
fn bench_unification(c: &mut Criterion) {
    let mut group = c.benchmark_group("shrinking/unification");
    group.sample_size(30);

    let engine = UnificationEngine::with_defaults();

    // Build sentences of various complexity for constraint checking
    for &n_words in &[5usize, 10, 20] {
        let sentence = sentence_of_length(n_words);
        if let Some(ref tree) = sentence.parse_tree {
            group.bench_with_input(
                BenchmarkId::new("propagate_features", n_words),
                tree,
                |b, tree| {
                    b.iter(|| black_box(engine.propagate_features(tree)))
                },
            );

            group.bench_with_input(
                BenchmarkId::new("check_all_constraints", n_words),
                tree,
                |b, tree| {
                    b.iter(|| black_box(engine.check_all_constraints(tree)))
                },
            );
        }
    }

    group.finish();
}

// =========================================================================
// Benchmark group 6: ShrinkableTree construction
// =========================================================================
fn bench_tree_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("shrinking/tree_construction");
    group.sample_size(30);

    for &n_words in &[10usize, 20, 40, 80] {
        let sentence = sentence_of_length(n_words);
        group.throughput(Throughput::Elements(n_words as u64));
        group.bench_with_input(
            BenchmarkId::new("from_sentence", n_words),
            &sentence,
            |b, sent| {
                b.iter(|| black_box(ShrinkableTree::from_sentence(sent)))
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_gchdd_shrinking,
    bench_gchdd_binary_search,
    bench_gchdd_always_accept,
    bench_grammar_validity,
    bench_unification,
    bench_tree_construction,
);
criterion_main!(benches);
