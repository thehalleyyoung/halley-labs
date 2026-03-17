use criterion::{black_box, criterion_group, criterion_main, Criterion};

use shared_types::{
    DependencyRelation, PosTag, Sentence,
};
use transformations::{
    make_test_sentence, BaseTransformation, TransformationKind, TransformationRegistry,
    PassivizationTransform, CleftTransform, TopicalizationTransform,
    RelativeClauseInsertTransform, RelativeClauseDeletionTransform,
    TenseChangeTransform, AgreementPerturbationTransform, SynonymSubstitutionTransform,
    NegationInsertionTransform, CoordinatedNpReorderTransform, PpAttachmentTransform,
    AdverbRepositionTransform, ThereInsertionTransform, DativeAlternationTransform,
    EmbeddingDepthTransform,
};
use metamorphic_core::relation::{
    SemanticEquivalenceMR, EntityPreservationMR,
    SentimentPreservationMR, SyntacticConsistencyMR,
};
use shared_types::{IRType, IntermediateRepresentation, MetamorphicRelation};

// ---------------------------------------------------------------------------
// Helper: build a rich test sentence suitable for many transformations
// ---------------------------------------------------------------------------
fn rich_active_sentence() -> Sentence {
    make_test_sentence(
        &[
            ("The", PosTag::Det),
            ("clever", PosTag::Adj),
            ("scientist", PosTag::Noun),
            ("discovered", PosTag::Verb),
            ("a", PosTag::Det),
            ("new", PosTag::Adj),
            ("formula", PosTag::Noun),
            ("quickly", PosTag::Adv),
            ("in", PosTag::Prep),
            ("the", PosTag::Det),
            ("laboratory", PosTag::Noun),
            (".", PosTag::Punct),
        ],
        &[
            (3, 0, DependencyRelation::Det),
            (3, 1, DependencyRelation::Amod),
            (3, 2, DependencyRelation::Nsubj),
            (0, 3, DependencyRelation::Root),
            (6, 4, DependencyRelation::Det),
            (6, 5, DependencyRelation::Amod),
            (3, 6, DependencyRelation::Dobj),
            (3, 7, DependencyRelation::Advmod),
            (3, 8, DependencyRelation::Prep),
            (10, 9, DependencyRelation::Det),
            (8, 10, DependencyRelation::Pobj),
            (3, 11, DependencyRelation::Punct),
        ],
    )
}

fn coordinated_sentence() -> Sentence {
    make_test_sentence(
        &[
            ("The", PosTag::Det),
            ("cat", PosTag::Noun),
            ("and", PosTag::Conj),
            ("the", PosTag::Det),
            ("dog", PosTag::Noun),
            ("chased", PosTag::Verb),
            ("mice", PosTag::Noun),
            (".", PosTag::Punct),
        ],
        &[
            (1, 0, DependencyRelation::Det),
            (5, 1, DependencyRelation::Nsubj),
            (1, 2, DependencyRelation::Cc),
            (4, 3, DependencyRelation::Det),
            (1, 4, DependencyRelation::Conj),
            (0, 5, DependencyRelation::Root),
            (5, 6, DependencyRelation::Dobj),
            (5, 7, DependencyRelation::Punct),
        ],
    )
}

fn dative_sentence() -> Sentence {
    make_test_sentence(
        &[
            ("She", PosTag::Pron),
            ("gave", PosTag::Verb),
            ("the", PosTag::Det),
            ("student", PosTag::Noun),
            ("a", PosTag::Det),
            ("book", PosTag::Noun),
            (".", PosTag::Punct),
        ],
        &[
            (1, 0, DependencyRelation::Nsubj),
            (0, 1, DependencyRelation::Root),
            (3, 2, DependencyRelation::Det),
            (1, 3, DependencyRelation::Iobj),
            (5, 4, DependencyRelation::Det),
            (1, 5, DependencyRelation::Dobj),
            (1, 6, DependencyRelation::Punct),
        ],
    )
}

// ---------------------------------------------------------------------------
// Helper: make IRs from a Sentence for MR checking
// ---------------------------------------------------------------------------
fn ir_from_sentence(s: &Sentence) -> IntermediateRepresentation {
    IntermediateRepresentation::new(IRType::PosTagged, s.clone())
        .with_confidence(0.95)
}

// =========================================================================
// Benchmark group 1: Individual transformation throughput
// =========================================================================
fn bench_individual_transformations(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformations/individual");
    group.sample_size(50);

    let sentence = rich_active_sentence();
    let coord_sentence = coordinated_sentence();
    let dative_sent = dative_sentence();

    let transforms: Vec<(&str, Box<dyn BaseTransformation>, Sentence)> = vec![
        ("passivization", Box::new(PassivizationTransform::new()), sentence.clone()),
        ("clefting", Box::new(CleftTransform::new()), sentence.clone()),
        ("topicalization", Box::new(TopicalizationTransform::new()), sentence.clone()),
        ("rel_clause_insert", Box::new(RelativeClauseInsertTransform::new()), sentence.clone()),
        ("rel_clause_delete", Box::new(RelativeClauseDeletionTransform::new()), sentence.clone()),
        ("tense_change", Box::new(TenseChangeTransform::new()), sentence.clone()),
        ("agreement_perturb", Box::new(AgreementPerturbationTransform::new()), sentence.clone()),
        ("synonym_subst", Box::new(SynonymSubstitutionTransform::new()), sentence.clone()),
        ("negation_insert", Box::new(NegationInsertionTransform::new()), sentence.clone()),
        ("coord_np_reorder", Box::new(CoordinatedNpReorderTransform::new()), coord_sentence.clone()),
        ("pp_attachment", Box::new(PpAttachmentTransform::new()), sentence.clone()),
        ("adverb_reposition", Box::new(AdverbRepositionTransform::new()), sentence.clone()),
        ("there_insertion", Box::new(ThereInsertionTransform::new()), sentence.clone()),
        ("dative_alternation", Box::new(DativeAlternationTransform::new()), dative_sent.clone()),
        ("embedding_depth", Box::new(EmbeddingDepthTransform::new()), sentence.clone()),
    ];

    for (name, transform, sent) in &transforms {
        if transform.is_applicable(sent) {
            group.bench_function(*name, |b| {
                b.iter(|| black_box(transform.apply(sent)))
            });
        }
    }

    group.finish();
}

// =========================================================================
// Benchmark group 2: Transformation composition (pairs and triples)
// =========================================================================
fn bench_transformation_composition(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformations/composition");
    group.sample_size(30);

    let registry = TransformationRegistry::default();
    let sentence = rich_active_sentence();

    // Pair compositions: apply t1, then feed result to t2
    let pair_combos: Vec<(TransformationKind, TransformationKind)> = vec![
        (TransformationKind::NegationInsertion, TransformationKind::Passivization),
        (TransformationKind::TenseChange, TransformationKind::Clefting),
        (TransformationKind::AdverbRepositioning, TransformationKind::Topicalization),
    ];

    for (t1_kind, t2_kind) in &pair_combos {
        let t1 = registry.get(t1_kind);
        let t2 = registry.get(t2_kind);
        if let (Some(t1), Some(t2)) = (t1, t2) {
            let name = format!("pair/{}/{}", t1_kind.name(), t2_kind.name());
            group.bench_function(&name, |b| {
                b.iter(|| {
                    if let Ok(r1) = t1.apply(&sentence) {
                        if t2.is_applicable(&r1.transformed) {
                            black_box(t2.apply(&r1.transformed).ok())
                        } else {
                            black_box(Some(r1))
                        }
                    } else {
                        black_box(None)
                    }
                })
            });
        }
    }

    // Triple composition
    let triple = [
        TransformationKind::NegationInsertion,
        TransformationKind::AdverbRepositioning,
        TransformationKind::TenseChange,
    ];
    let t_triple: Vec<&dyn BaseTransformation> = triple.iter().filter_map(|k| registry.get(k)).collect();
    if t_triple.len() == 3 {
        group.bench_function("triple/neg_adv_tense", |b| {
            b.iter(|| {
                let mut current = sentence.clone();
                for t in &t_triple {
                    if t.is_applicable(&current) {
                        if let Ok(result) = t.apply(&current) {
                            current = result.transformed;
                        }
                    }
                }
                black_box(current)
            })
        });
    }

    group.finish();
}

// =========================================================================
// Benchmark group 3: Registry apply_all_applicable
// =========================================================================
fn bench_registry_coverage(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformations/registry");
    group.sample_size(30);

    let registry = TransformationRegistry::default();
    let sentence = rich_active_sentence();

    group.bench_function("get_applicable", |b| {
        b.iter(|| black_box(registry.get_applicable(&sentence)))
    });

    group.bench_function("apply_all_applicable", |b| {
        b.iter(|| black_box(registry.apply_all_applicable(&sentence)))
    });

    let corpus: Vec<Sentence> = vec![
        rich_active_sentence(),
        coordinated_sentence(),
        dative_sentence(),
    ];
    group.bench_function("coverage_analysis", |b| {
        b.iter(|| black_box(registry.coverage_analysis(&corpus)))
    });

    group.finish();
}

// =========================================================================
// Benchmark group 4: MR checking for different relation types
// =========================================================================
fn bench_mr_checking(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformations/mr_check");
    group.sample_size(50);

    let original = rich_active_sentence();
    // Create a slightly modified version to simulate transformed output
    let mut transformed = original.clone();
    transformed.raw_text = "A new formula was discovered by the clever scientist quickly in the laboratory .".to_string();

    let ir_orig = ir_from_sentence(&original);
    let ir_trans = ir_from_sentence(&transformed);

    group.bench_function("semantic_equivalence", |b| {
        let mr = SemanticEquivalenceMR::new(0.1, 0.5, 0.5);
        b.iter(|| black_box(mr.check_with_detail(&ir_orig, &ir_trans)))
    });

    group.bench_function("entity_preservation", |b| {
        let mr = EntityPreservationMR::new(0.05);
        b.iter(|| black_box(mr.check_with_detail(&ir_orig, &ir_trans)))
    });

    group.bench_function("sentiment_preservation", |b| {
        let mr = SentimentPreservationMR::new(0.1);
        b.iter(|| black_box(mr.check_with_detail(&ir_orig, &ir_trans)))
    });

    group.bench_function("syntactic_consistency", |b| {
        let mr = SyntacticConsistencyMR::new(0.15);
        b.iter(|| black_box(mr.check_with_detail(&ir_orig, &ir_trans)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_individual_transformations,
    bench_transformation_composition,
    bench_registry_coverage,
    bench_mr_checking,
);
criterion_main!(benches);
