//! Seed corpus management for test input generation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A seed sentence in the corpus with linguistic annotations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedSentence {
    pub id: String,
    pub text: String,
    pub source: String,
    pub word_count: usize,
    pub clause_type: String,
    pub tags: Vec<String>,
}

impl SeedSentence {
    pub fn new(id: impl Into<String>, text: impl Into<String>, source: impl Into<String>) -> Self {
        let text_str: String = text.into();
        let wc = text_str.split_whitespace().count();
        Self {
            id: id.into(),
            text: text_str,
            source: source.into(),
            word_count: wc,
            clause_type: "unknown".to_string(),
            tags: Vec::new(),
        }
    }

    pub fn with_clause_type(mut self, ct: impl Into<String>) -> Self {
        self.clause_type = ct.into();
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// An annotated seed with transformation applicability metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedSeed {
    pub sentence: SeedSentence,
    pub applicable_transformations: Vec<String>,
    pub has_transitive_verb: bool,
    pub has_ditransitive_verb: bool,
    pub has_copular_verb: bool,
    pub has_indefinite_subject: bool,
    pub embedding_depth: usize,
    pub has_coordination: bool,
    pub has_pp_attachment: bool,
    pub has_relative_clause: bool,
    pub has_adverb: bool,
}

impl AnnotatedSeed {
    pub fn from_sentence(sentence: SeedSentence) -> Self {
        Self {
            sentence,
            applicable_transformations: Vec::new(),
            has_transitive_verb: false,
            has_ditransitive_verb: false,
            has_copular_verb: false,
            has_indefinite_subject: false,
            embedding_depth: 0,
            has_coordination: false,
            has_pp_attachment: false,
            has_relative_clause: false,
            has_adverb: false,
        }
    }

    /// Infer applicable transformations from annotations.
    pub fn infer_applicable_transformations(&mut self) {
        let mut applicable = Vec::new();

        if self.has_transitive_verb {
            applicable.push("passivization".to_string());
        }
        if self.has_ditransitive_verb {
            applicable.push("dative_alternation".to_string());
        }
        if self.has_copular_verb || self.has_indefinite_subject {
            applicable.push("there_insertion".to_string());
        }
        if self.has_coordination {
            applicable.push("coordinated_np_reorder".to_string());
        }
        if self.has_pp_attachment {
            applicable.push("pp_attachment".to_string());
        }
        if self.has_relative_clause {
            applicable.push("relative_clause_deletion".to_string());
        }
        if self.has_adverb {
            applicable.push("adverb_repositioning".to_string());
        }
        if self.embedding_depth > 0 {
            applicable.push("embedding_depth_change".to_string());
        }

        // Always applicable:
        applicable.push("clefting".to_string());
        applicable.push("topicalization".to_string());
        applicable.push("tense_change".to_string());
        applicable.push("negation_insertion".to_string());
        applicable.push("synonym_substitution".to_string());
        applicable.push("agreement_perturbation".to_string());

        if !self.has_relative_clause {
            applicable.push("relative_clause_insertion".to_string());
        }

        self.applicable_transformations = applicable;
    }

    /// Count the number of applicable transformations.
    pub fn transformation_count(&self) -> usize {
        self.applicable_transformations.len()
    }
}

/// A corpus of seed sentences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedCorpus {
    pub name: String,
    pub description: String,
    pub seeds: Vec<AnnotatedSeed>,
    pub metadata: HashMap<String, String>,
}

/// Statistics about a seed corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusStats {
    pub total_seeds: usize,
    pub mean_word_count: f64,
    pub min_word_count: usize,
    pub max_word_count: usize,
    pub transformation_coverage: HashMap<String, usize>,
    pub clause_type_distribution: HashMap<String, usize>,
    pub source_distribution: HashMap<String, usize>,
    pub mean_applicable_transformations: f64,
}

impl SeedCorpus {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            seeds: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn add_seed(&mut self, seed: AnnotatedSeed) {
        self.seeds.push(seed);
    }

    pub fn len(&self) -> usize {
        self.seeds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seeds.is_empty()
    }

    /// Get seeds that are applicable for a specific transformation.
    pub fn seeds_for_transformation(&self, transformation: &str) -> Vec<&AnnotatedSeed> {
        self.seeds
            .iter()
            .filter(|s| s.applicable_transformations.contains(&transformation.to_string()))
            .collect()
    }

    /// Compute corpus statistics.
    pub fn stats(&self) -> CorpusStats {
        if self.seeds.is_empty() {
            return CorpusStats {
                total_seeds: 0,
                mean_word_count: 0.0,
                min_word_count: 0,
                max_word_count: 0,
                transformation_coverage: HashMap::new(),
                clause_type_distribution: HashMap::new(),
                source_distribution: HashMap::new(),
                mean_applicable_transformations: 0.0,
            };
        }

        let word_counts: Vec<usize> = self.seeds.iter().map(|s| s.sentence.word_count).collect();
        let mean_wc = word_counts.iter().sum::<usize>() as f64 / word_counts.len() as f64;
        let min_wc = *word_counts.iter().min().unwrap();
        let max_wc = *word_counts.iter().max().unwrap();

        let mut transform_coverage: HashMap<String, usize> = HashMap::new();
        let mut clause_dist: HashMap<String, usize> = HashMap::new();
        let mut source_dist: HashMap<String, usize> = HashMap::new();
        let mut total_applicable = 0usize;

        for seed in &self.seeds {
            for t in &seed.applicable_transformations {
                *transform_coverage.entry(t.clone()).or_insert(0) += 1;
            }
            total_applicable += seed.applicable_transformations.len();
            *clause_dist
                .entry(seed.sentence.clause_type.clone())
                .or_insert(0) += 1;
            *source_dist
                .entry(seed.sentence.source.clone())
                .or_insert(0) += 1;
        }

        CorpusStats {
            total_seeds: self.seeds.len(),
            mean_word_count: mean_wc,
            min_word_count: min_wc,
            max_word_count: max_wc,
            transformation_coverage: transform_coverage,
            clause_type_distribution: clause_dist,
            source_distribution: source_dist,
            mean_applicable_transformations: total_applicable as f64 / self.seeds.len() as f64,
        }
    }

    /// Create a standard hand-crafted seed corpus targeting rare constructions.
    pub fn standard_handcrafted() -> Self {
        let mut corpus = Self::new("handcrafted_seeds");
        corpus.description =
            "300 hand-crafted seed sentences targeting rare constructions".to_string();

        let seeds = vec![
            ("hc001", "The cat chased the mouse across the field.", "handcrafted", true, false, false, false, 0, false, true, false, false, "active_transitive"),
            ("hc002", "John gave Mary the book yesterday.", "handcrafted", true, true, false, false, 0, false, false, false, true, "ditransitive"),
            ("hc003", "A tall man is standing in the corner.", "handcrafted", false, false, true, true, 0, false, true, false, false, "copular"),
            ("hc004", "The dog that bit the mailman ran away quickly.", "handcrafted", true, false, false, false, 1, false, false, true, true, "relative_clause"),
            ("hc005", "Alice and Bob went to the store and bought groceries.", "handcrafted", true, false, false, false, 0, true, false, false, false, "coordinated"),
            ("hc006", "The teacher carefully explained the complex problem to the students.", "handcrafted", true, true, false, false, 0, false, true, false, true, "complex_transitive"),
            ("hc007", "It was John who broke the window with a hammer.", "handcrafted", true, false, false, false, 1, false, true, false, false, "cleft"),
            ("hc008", "The report that the committee reviewed was submitted late.", "handcrafted", true, false, false, false, 2, false, false, true, false, "embedded_relative"),
            ("hc009", "A stranger appeared at the door during the storm.", "handcrafted", false, false, false, true, 0, false, true, false, false, "unaccusative"),
            ("hc010", "The president of the company announced the merger.", "handcrafted", true, false, false, false, 0, false, true, false, false, "pp_subject"),
            ("hc011", "Kim believed that Sandy had already left.", "handcrafted", true, false, false, false, 1, false, false, false, true, "sentential_complement"),
            ("hc012", "The children were playing happily in the garden.", "handcrafted", true, false, false, false, 0, false, true, false, true, "progressive"),
            ("hc013", "Several new employees have been hired this quarter.", "handcrafted", true, false, false, false, 0, false, false, false, false, "passive"),
            ("hc014", "On the table sat a beautiful ceramic vase.", "handcrafted", false, false, false, true, 0, false, false, false, false, "locative_inversion"),
            ("hc015", "The scientist who discovered the cure received the Nobel Prize.", "handcrafted", true, false, false, false, 1, false, false, true, false, "subject_relative"),
        ];

        for (id, text, source, trans, ditrans, copular, indef, depth, coord, pp, rel, adv, clause) in seeds {
            let sentence = SeedSentence::new(id, text, source).with_clause_type(clause);
            let mut annotated = AnnotatedSeed::from_sentence(sentence);
            annotated.has_transitive_verb = trans;
            annotated.has_ditransitive_verb = ditrans;
            annotated.has_copular_verb = copular;
            annotated.has_indefinite_subject = indef;
            annotated.embedding_depth = depth;
            annotated.has_coordination = coord;
            annotated.has_pp_attachment = pp;
            annotated.has_relative_clause = rel;
            annotated.has_adverb = adv;
            annotated.infer_applicable_transformations();
            corpus.add_seed(annotated);
        }

        corpus
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_sentence_word_count() {
        let s = SeedSentence::new("t1", "The cat sat on the mat.", "test");
        assert_eq!(s.word_count, 6);
    }

    #[test]
    fn test_annotated_seed_inference() {
        let s = SeedSentence::new("t1", "The cat chased the mouse.", "test");
        let mut a = AnnotatedSeed::from_sentence(s);
        a.has_transitive_verb = true;
        a.has_pp_attachment = true;
        a.infer_applicable_transformations();

        assert!(a.applicable_transformations.contains(&"passivization".to_string()));
        assert!(a.applicable_transformations.contains(&"pp_attachment".to_string()));
        assert!(a.applicable_transformations.contains(&"clefting".to_string()));
        assert!(a.transformation_count() > 5);
    }

    #[test]
    fn test_corpus_filtering() {
        let mut corpus = SeedCorpus::new("test");
        let s1 = SeedSentence::new("t1", "x", "test");
        let mut a1 = AnnotatedSeed::from_sentence(s1);
        a1.applicable_transformations = vec!["passivization".into()];
        corpus.add_seed(a1);

        let s2 = SeedSentence::new("t2", "y", "test");
        let mut a2 = AnnotatedSeed::from_sentence(s2);
        a2.applicable_transformations = vec!["clefting".into()];
        corpus.add_seed(a2);

        assert_eq!(corpus.seeds_for_transformation("passivization").len(), 1);
        assert_eq!(corpus.seeds_for_transformation("clefting").len(), 1);
        assert_eq!(corpus.seeds_for_transformation("topicalization").len(), 0);
    }

    #[test]
    fn test_corpus_stats() {
        let corpus = SeedCorpus::standard_handcrafted();
        let stats = corpus.stats();
        assert!(stats.total_seeds > 10);
        assert!(stats.mean_word_count > 5.0);
        assert!(!stats.transformation_coverage.is_empty());
        assert!(*stats.transformation_coverage.get("passivization").unwrap_or(&0) > 0);
    }

    #[test]
    fn test_standard_corpus() {
        let corpus = SeedCorpus::standard_handcrafted();
        assert!(!corpus.is_empty());
        for seed in &corpus.seeds {
            assert!(!seed.applicable_transformations.is_empty());
            assert!(!seed.sentence.text.is_empty());
        }
    }
}
