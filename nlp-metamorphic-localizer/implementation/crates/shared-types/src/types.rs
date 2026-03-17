//! Core domain types for the NLP metamorphic fault localizer.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

// ── Newtype ID wrappers ─────────────────────────────────────────────────────

macro_rules! id_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub struct $name(pub Uuid);

        impl $name {
            /// Create a deterministic id from a human-readable name.
            pub fn new(name: &str) -> Self {
                Self(Uuid::new_v5(&Uuid::NAMESPACE_DNS, name.as_bytes()))
            }

            /// Create a random id.
            pub fn random() -> Self {
                Self(Uuid::new_v4())
            }

            pub fn from_uuid(id: Uuid) -> Self {
                Self(id)
            }

            pub fn as_uuid(&self) -> &Uuid {
                &self.0
            }

            /// Alias for [`Self::new`].
            pub fn from_name(name: &str) -> Self {
                Self::new(name)
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::random()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl std::str::FromStr for $name {
            type Err = uuid::Error;
            fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
                Ok(Self(Uuid::parse_str(s)?))
            }
        }
    };
}

id_newtype!(
    /// Identifies a single pipeline stage (e.g. tokenizer, POS tagger).
    StageId
);
id_newtype!(
    /// Identifies a full pipeline configuration.
    PipelineId
);
id_newtype!(
    /// Identifies a metamorphic transformation.
    TransformationId
);
id_newtype!(
    /// Identifies a single test case.
    TestCaseId
);

// ── POS tags ────────────────────────────────────────────────────────────────

/// Universal-Dependencies-inspired part-of-speech tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum PosTag {
    Noun,
    Verb,
    Adj,
    Adv,
    Det,
    Prep,
    Conj,
    Pron,
    Aux,
    Punct,
    Num,
    Part,
    Intj,
    Other,
}

impl PosTag {
    pub fn is_content_word(&self) -> bool {
        matches!(self, Self::Noun | Self::Verb | Self::Adj | Self::Adv)
    }

    pub fn is_function_word(&self) -> bool {
        matches!(
            self,
            Self::Det | Self::Prep | Self::Conj | Self::Pron | Self::Aux | Self::Part
        )
    }

    pub fn is_open_class(&self) -> bool {
        matches!(self, Self::Noun | Self::Verb | Self::Adj | Self::Adv | Self::Intj)
    }

    /// Best-effort mapping from Penn Treebank tag strings.
    pub fn from_penn(tag: &str) -> Self {
        match tag {
            "NN" | "NNS" | "NNP" | "NNPS" => Self::Noun,
            "VB" | "VBD" | "VBG" | "VBN" | "VBP" | "VBZ" => Self::Verb,
            "JJ" | "JJR" | "JJS" => Self::Adj,
            "RB" | "RBR" | "RBS" => Self::Adv,
            "DT" | "PDT" | "WDT" => Self::Det,
            "IN" | "TO" => Self::Prep,
            "CC" => Self::Conj,
            "PRP" | "PRP$" | "WP" | "WP$" => Self::Pron,
            "MD" => Self::Aux,
            "." | "," | ":" | "-LRB-" | "-RRB-" | "``" | "''" => Self::Punct,
            "CD" => Self::Num,
            "RP" | "POS" => Self::Part,
            "UH" => Self::Intj,
            _ => Self::Other,
        }
    }
}

impl fmt::Display for PosTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Noun => "NOUN",
            Self::Verb => "VERB",
            Self::Adj => "ADJ",
            Self::Adv => "ADV",
            Self::Det => "DET",
            Self::Prep => "ADP",
            Self::Conj => "CCONJ",
            Self::Pron => "PRON",
            Self::Aux => "AUX",
            Self::Punct => "PUNCT",
            Self::Num => "NUM",
            Self::Part => "PART",
            Self::Intj => "INTJ",
            Self::Other => "X",
        };
        f.write_str(s)
    }
}

// ── Token ───────────────────────────────────────────────────────────────────

/// A single token with its linguistic annotations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Token {
    pub text: String,
    pub lemma: Option<String>,
    pub pos_tag: Option<PosTag>,
    pub index: usize,
    #[serde(default)]
    pub features: HashMap<String, String>,
}

impl Token {
    pub fn new(text: impl Into<String>, index: usize) -> Self {
        Self {
            text: text.into(),
            lemma: None,
            pos_tag: None,
            index,
            features: HashMap::new(),
        }
    }

    pub fn with_pos(mut self, tag: PosTag) -> Self {
        self.pos_tag = Some(tag);
        self
    }

    pub fn with_lemma(mut self, lemma: impl Into<String>) -> Self {
        self.lemma = Some(lemma.into());
        self
    }

    pub fn with_feature(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.features.insert(key.into(), value.into());
        self
    }

    /// The lemma if available, otherwise the lowercased surface form.
    pub fn normalized_form(&self) -> String {
        self.lemma
            .as_deref()
            .unwrap_or(&self.text)
            .to_lowercase()
    }

    pub fn is_punctuation(&self) -> bool {
        self.pos_tag == Some(PosTag::Punct)
            || self.text.chars().all(|c| c.is_ascii_punctuation())
    }

    pub fn char_len(&self) -> usize {
        self.text.chars().count()
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.pos_tag {
            Some(tag) => write!(f, "{}/{}", self.text, tag),
            None => f.write_str(&self.text),
        }
    }
}

// ── Dependency relation ─────────────────────────────────────────────────────

/// Universal-Dependencies-style dependency relation labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DependencyRelation {
    Nsubj,
    Dobj,
    Iobj,
    Nmod,
    Amod,
    Advmod,
    Prep,
    Pobj,
    Det,
    Aux,
    Cop,
    Cc,
    Conj,
    Mark,
    Relcl,
    Advcl,
    Xcomp,
    Ccomp,
    Root,
    Punct,
    Case,
    Compound,
    Flat,
    Appos,
    Parataxis,
    Vocative,
    Discourse,
    Expl,
    Fixed,
    List,
    Orphan,
    Reparandum,
    Goeswith,
    Dep,
    Other,
}

impl DependencyRelation {
    /// True for core argument relations (subject, direct object, etc.).
    pub fn is_core_argument(&self) -> bool {
        matches!(self, Self::Nsubj | Self::Dobj | Self::Iobj | Self::Ccomp | Self::Xcomp)
    }

    /// True for modifier relations.
    pub fn is_modifier(&self) -> bool {
        matches!(
            self,
            Self::Amod | Self::Advmod | Self::Nmod | Self::Relcl | Self::Advcl
        )
    }

    /// True for function-word relations.
    pub fn is_function(&self) -> bool {
        matches!(
            self,
            Self::Det | Self::Aux | Self::Cop | Self::Case | Self::Mark | Self::Cc
        )
    }

    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "nsubj" | "nsubjpass" => Self::Nsubj,
            "dobj" | "obj" => Self::Dobj,
            "iobj" => Self::Iobj,
            "nmod" | "obl" => Self::Nmod,
            "amod" => Self::Amod,
            "advmod" => Self::Advmod,
            "prep" | "case" if s == "prep" => Self::Prep,
            "pobj" => Self::Pobj,
            "det" => Self::Det,
            "aux" | "auxpass" => Self::Aux,
            "cop" => Self::Cop,
            "cc" => Self::Cc,
            "conj" => Self::Conj,
            "mark" => Self::Mark,
            "relcl" | "acl:relcl" | "acl" => Self::Relcl,
            "advcl" => Self::Advcl,
            "xcomp" => Self::Xcomp,
            "ccomp" => Self::Ccomp,
            "root" => Self::Root,
            "punct" => Self::Punct,
            "case" => Self::Case,
            "compound" => Self::Compound,
            "flat" | "flat:name" => Self::Flat,
            "appos" => Self::Appos,
            "parataxis" => Self::Parataxis,
            "vocative" => Self::Vocative,
            "discourse" => Self::Discourse,
            "expl" => Self::Expl,
            "fixed" | "mwe" => Self::Fixed,
            "list" => Self::List,
            "orphan" => Self::Orphan,
            "reparandum" => Self::Reparandum,
            "goeswith" => Self::Goeswith,
            "dep" => Self::Dep,
            _ => Self::Other,
        }
    }
}

impl fmt::Display for DependencyRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ── Dependency edge ─────────────────────────────────────────────────────────

/// An edge in a dependency tree.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DependencyEdge {
    pub head_index: usize,
    pub dependent_index: usize,
    pub relation: DependencyRelation,
}

impl DependencyEdge {
    pub fn new(head: usize, dep: usize, rel: DependencyRelation) -> Self {
        Self {
            head_index: head,
            dependent_index: dep,
            relation: rel,
        }
    }

    /// True when the head is the virtual root (index 0 in CoNLL-U convention).
    pub fn is_root_edge(&self) -> bool {
        self.relation == DependencyRelation::Root
    }
}

impl fmt::Display for DependencyEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}--{}-->{}",
            self.head_index, self.relation, self.dependent_index
        )
    }
}

// ── Entity types ────────────────────────────────────────────────────────────

/// Named-entity labels broadly following OntoNotes categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum EntityLabel {
    Person,
    Organization,
    Location,
    Date,
    Time,
    Money,
    Percent,
    Facility,
    Gpe,
    Product,
    Event,
    WorkOfArt,
    Law,
    Language,
    Norp,
    Quantity,
    Ordinal,
    Cardinal,
    Misc,
}

impl EntityLabel {
    /// True for entity types that refer to named individuals / orgs / places.
    pub fn is_named(&self) -> bool {
        matches!(
            self,
            Self::Person
                | Self::Organization
                | Self::Location
                | Self::Facility
                | Self::Gpe
                | Self::Product
                | Self::Event
                | Self::WorkOfArt
                | Self::Law
                | Self::Language
                | Self::Norp
        )
    }

    /// True for numeric entity types.
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            Self::Date
                | Self::Time
                | Self::Money
                | Self::Percent
                | Self::Quantity
                | Self::Ordinal
                | Self::Cardinal
        )
    }
}

impl fmt::Display for EntityLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// A contiguous span of tokens labeled with a named-entity type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntitySpan {
    /// Token start index (inclusive).
    pub start: usize,
    /// Token end index (exclusive).
    pub end: usize,
    pub label: EntityLabel,
    pub text: String,
    #[serde(default)]
    pub confidence: f64,
}

impl EntitySpan {
    pub fn new(start: usize, end: usize, label: EntityLabel, text: impl Into<String>) -> Self {
        Self {
            start,
            end,
            label,
            text: text.into(),
            confidence: 0.0,
        }
    }

    pub fn token_len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// True when two spans overlap.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// True when this span fully contains `other`.
    pub fn contains(&self, other: &Self) -> bool {
        self.start <= other.start && other.end <= self.end
    }
}

impl fmt::Display for EntitySpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]({}:{}-{})", self.text, self.label, self.start, self.end)
    }
}

// ── Sentence ────────────────────────────────────────────────────────────────

/// A fully annotated sentence carrying tokens, dependency edges, and entities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sentence {
    pub tokens: Vec<Token>,
    pub dependency_edges: Vec<DependencyEdge>,
    pub entities: Vec<EntitySpan>,
    pub raw_text: String,
    /// Cached sentence features (populated by NLP pipeline).
    #[serde(default)]
    pub features: Option<SentenceFeatures>,
    /// Optional constituency parse tree.
    #[serde(default)]
    pub parse_tree: Option<ParseTree>,
}

impl Sentence {
    pub fn new(raw_text: impl Into<String>) -> Self {
        Self {
            tokens: Vec::new(),
            dependency_edges: Vec::new(),
            entities: Vec::new(),
            raw_text: raw_text.into(),
            features: None,
            parse_tree: None,
        }
    }

    /// Alias for [`Self::new`].
    pub fn from_text(raw_text: impl Into<String>) -> Self {
        Self::new(raw_text)
    }

    pub fn from_tokens(tokens: Vec<Token>, raw_text: impl Into<String>) -> Self {
        Self {
            tokens,
            dependency_edges: Vec::new(),
            entities: Vec::new(),
            raw_text: raw_text.into(),
            features: None,
            parse_tree: None,
        }
    }

    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// Return the root token index (the head of the Root edge).
    pub fn root_index(&self) -> Option<usize> {
        self.dependency_edges
            .iter()
            .find(|e| e.relation == DependencyRelation::Root)
            .map(|e| e.dependent_index)
    }

    /// Collect tokens whose POS tag matches `tag`.
    pub fn tokens_with_pos(&self, tag: PosTag) -> Vec<&Token> {
        self.tokens
            .iter()
            .filter(|t| t.pos_tag == Some(tag))
            .collect()
    }

    /// Return entity spans that contain the token at `index`.
    pub fn entities_at(&self, index: usize) -> Vec<&EntitySpan> {
        self.entities
            .iter()
            .filter(|e| index >= e.start && index < e.end)
            .collect()
    }

    /// Dependents of the token at `head_idx`.
    pub fn dependents_of(&self, head_idx: usize) -> Vec<usize> {
        self.dependency_edges
            .iter()
            .filter(|e| e.head_index == head_idx)
            .map(|e| e.dependent_index)
            .collect()
    }

    /// The head edge for the token at `dep_idx`.
    pub fn head_of(&self, dep_idx: usize) -> Option<&DependencyEdge> {
        self.dependency_edges
            .iter()
            .find(|e| e.dependent_index == dep_idx)
    }

    /// Reconstruct whitespace-separated surface text from tokens.
    pub fn surface_text(&self) -> String {
        self.tokens
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Compute shallow syntactic features.
    pub fn compute_features(&self) -> SentenceFeatures {
        let has_passive = self.tokens.iter().any(|t| {
            t.features.get("Voice").map_or(false, |v| v == "Pass")
        }) || self.dependency_edges.iter().any(|e| {
            // nsubjpass relation implies passive
            e.relation == DependencyRelation::Nsubj
                && self
                    .tokens
                    .get(e.dependent_index)
                    .and_then(|t| t.features.get("Voice"))
                    .map_or(false, |v| v == "Pass")
        });

        let has_relative_clause = self
            .dependency_edges
            .iter()
            .any(|e| e.relation == DependencyRelation::Relcl);

        let has_coordination = self
            .dependency_edges
            .iter()
            .any(|e| e.relation == DependencyRelation::Conj);

        let has_negation = self
            .tokens
            .iter()
            .any(|t| {
                let low = t.text.to_lowercase();
                low == "not" || low == "n't" || low == "never" || low == "no" || low == "neither"
            });

        let clause_depth = self.compute_clause_depth();

        let voice = if has_passive { Voice::Passive } else { Voice::Active };

        SentenceFeatures {
            has_passive,
            has_relative_clause,
            clause_depth,
            has_coordination,
            has_negation,
            is_negated: has_negation,
            tense: Some(self.detect_tense()),
            voice: Some(voice),
            mood: Some(self.detect_mood()),
            sentiment_label: None,
            sentiment_score: None,
            extra: HashMap::new(),
        }
    }

    fn compute_clause_depth(&self) -> u32 {
        let clause_rels = [
            DependencyRelation::Relcl,
            DependencyRelation::Advcl,
            DependencyRelation::Ccomp,
            DependencyRelation::Xcomp,
        ];
        if self.dependency_edges.is_empty() {
            return 0;
        }
        // Build children map.
        let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
        for e in &self.dependency_edges {
            children.entry(e.head_index).or_default().push(e.dependent_index);
        }
        let root = self.root_index().unwrap_or(0);
        // BFS computing max clause depth.
        let mut max_depth: u32 = 0;
        let mut stack: Vec<(usize, u32)> = vec![(root, 0)];
        while let Some((node, depth)) = stack.pop() {
            if depth > max_depth {
                max_depth = depth;
            }
            if let Some(kids) = children.get(&node) {
                for &kid in kids {
                    let edge = self
                        .dependency_edges
                        .iter()
                        .find(|e| e.head_index == node && e.dependent_index == kid);
                    let is_clause = edge.map_or(false, |e| clause_rels.contains(&e.relation));
                    stack.push((kid, if is_clause { depth + 1 } else { depth }));
                }
            }
        }
        max_depth
    }

    fn detect_tense(&self) -> Tense {
        for token in &self.tokens {
            if let Some(tense_feat) = token.features.get("Tense") {
                return match tense_feat.as_str() {
                    "Past" => Tense::Past,
                    "Pres" => Tense::Present,
                    "Fut" => Tense::Future,
                    _ => Tense::Unknown,
                };
            }
        }
        // Heuristic: look for past-tense verb forms.
        for token in &self.tokens {
            if token.pos_tag == Some(PosTag::Verb) {
                let low = token.text.to_lowercase();
                if low == "will" || low == "shall" {
                    return Tense::Future;
                }
                if low.ends_with("ed") {
                    return Tense::Past;
                }
            }
        }
        Tense::Present
    }

    fn detect_mood(&self) -> Mood {
        if let Some(last) = self.tokens.last() {
            if last.text == "?" {
                return Mood::Interrogative;
            }
        }
        for token in &self.tokens {
            if let Some(m) = token.features.get("Mood") {
                return match m.as_str() {
                    "Sub" => Mood::Subjunctive,
                    "Imp" => Mood::Imperative,
                    _ => Mood::Indicative,
                };
            }
        }
        Mood::Indicative
    }
}

impl fmt::Display for Sentence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw_text)
    }
}

// ── Parse tree ──────────────────────────────────────────────────────────────

/// A node in a constituency parse tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParseNode {
    pub label: String,
    pub word: Option<String>,
    pub children: Vec<usize>,
    pub parent: Option<usize>,
    #[serde(default)]
    pub features: HashMap<String, String>,
    pub span_start: usize,
    pub span_end: usize,
}

impl ParseNode {
    pub fn new_terminal(label: impl Into<String>, word: impl Into<String>, idx: usize) -> Self {
        Self {
            label: label.into(),
            word: Some(word.into()),
            children: Vec::new(),
            parent: None,
            features: HashMap::new(),
            span_start: idx,
            span_end: idx + 1,
        }
    }

    pub fn new_nonterminal(label: impl Into<String>, span_start: usize, span_end: usize) -> Self {
        Self {
            label: label.into(),
            word: None,
            children: Vec::new(),
            parent: None,
            features: HashMap::new(),
            span_start,
            span_end,
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.children.is_empty()
    }

    pub fn is_preterminal(&self) -> bool {
        self.children.len() == 1 && self.word.is_none()
    }

    pub fn span_len(&self) -> usize {
        self.span_end.saturating_sub(self.span_start)
    }
}

/// A constituency parse tree stored as a flat node array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParseTree {
    pub nodes: Vec<ParseNode>,
    pub root_index: usize,
}

impl ParseTree {
    pub fn new(nodes: Vec<ParseNode>, root_index: usize) -> Self {
        Self { nodes, root_index }
    }

    pub fn root(&self) -> Option<&ParseNode> {
        self.nodes.get(self.root_index)
    }

    pub fn terminals(&self) -> Vec<&ParseNode> {
        self.nodes.iter().filter(|n| n.is_terminal()).collect()
    }

    /// Number of nodes in the tree.
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Max depth from the root.
    pub fn depth(&self) -> usize {
        self.depth_of(self.root_index)
    }

    fn depth_of(&self, idx: usize) -> usize {
        let node = match self.nodes.get(idx) {
            Some(n) => n,
            None => return 0,
        };
        if node.children.is_empty() {
            return 1;
        }
        1 + node
            .children
            .iter()
            .map(|&c| self.depth_of(c))
            .max()
            .unwrap_or(0)
    }

    /// Yield the terminal words in left-to-right order.
    pub fn yield_words(&self) -> Vec<&str> {
        let mut terms: Vec<&ParseNode> = self.terminals();
        terms.sort_by_key(|n| n.span_start);
        terms
            .iter()
            .filter_map(|n| n.word.as_deref())
            .collect()
    }
}

// ── Sentence features / voice / tense / mood ────────────────────────────────

/// Shallow syntactic features of a sentence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SentenceFeatures {
    pub has_passive: bool,
    pub has_relative_clause: bool,
    pub clause_depth: u32,
    pub has_coordination: bool,
    pub has_negation: bool,
    #[serde(default)]
    pub tense: Option<Tense>,
    #[serde(default)]
    pub voice: Option<Voice>,
    #[serde(default)]
    pub mood: Option<Mood>,
    /// Whether the sentence is negated (alias for downstream consumers).
    #[serde(default)]
    pub is_negated: bool,
    /// Optional sentiment label (e.g. "positive", "negative", "neutral").
    #[serde(default)]
    pub sentiment_label: Option<String>,
    /// Optional sentiment score (e.g. on a [-1, 1] scale).
    #[serde(default)]
    pub sentiment_score: Option<f64>,
    /// Arbitrary extra key-value annotations.
    #[serde(default)]
    pub extra: HashMap<String, String>,
}

impl Default for SentenceFeatures {
    fn default() -> Self {
        Self {
            has_passive: false,
            has_relative_clause: false,
            clause_depth: 0,
            has_coordination: false,
            has_negation: false,
            tense: None,
            voice: None,
            mood: None,
            is_negated: false,
            sentiment_label: None,
            sentiment_score: None,
            extra: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Voice {
    Active,
    Passive,
    Middle,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tense {
    Past,
    Present,
    Future,
    PastPerfect,
    PresentPerfect,
    FuturePerfect,
    PastProgressive,
    PresentProgressive,
    FutureProgressive,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Mood {
    Indicative,
    Subjunctive,
    Imperative,
    Interrogative,
    Unknown,
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_id_roundtrip() {
        let id = StageId::new();
        let s = id.to_string();
        let parsed: StageId = s.parse().unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn test_stage_id_from_name_deterministic() {
        let a = StageId::from_name("tokenizer");
        let b = StageId::from_name("tokenizer");
        assert_eq!(a, b);
        assert_ne!(a, StageId::from_name("pos_tagger"));
    }

    #[test]
    fn test_pos_tag_from_penn() {
        assert_eq!(PosTag::from_penn("NN"), PosTag::Noun);
        assert_eq!(PosTag::from_penn("VBZ"), PosTag::Verb);
        assert_eq!(PosTag::from_penn("JJ"), PosTag::Adj);
        assert_eq!(PosTag::from_penn("XYZ"), PosTag::Other);
    }

    #[test]
    fn test_pos_tag_categories() {
        assert!(PosTag::Noun.is_content_word());
        assert!(PosTag::Det.is_function_word());
        assert!(!PosTag::Punct.is_open_class());
    }

    #[test]
    fn test_token_builder() {
        let t = Token::new("running", 0)
            .with_pos(PosTag::Verb)
            .with_lemma("run")
            .with_feature("Tense", "Pres");
        assert_eq!(t.normalized_form(), "run");
        assert!(!t.is_punctuation());
    }

    #[test]
    fn test_dependency_edge_display() {
        let e = DependencyEdge::new(0, 1, DependencyRelation::Nsubj);
        assert!(e.to_string().contains("Nsubj"));
    }

    #[test]
    fn test_dependency_relation_categories() {
        assert!(DependencyRelation::Nsubj.is_core_argument());
        assert!(DependencyRelation::Amod.is_modifier());
        assert!(DependencyRelation::Det.is_function());
    }

    #[test]
    fn test_entity_span_overlap() {
        let a = EntitySpan::new(0, 3, EntityLabel::Person, "John Smith Jr");
        let b = EntitySpan::new(2, 5, EntityLabel::Organization, "Jr Corp LLC");
        assert!(a.overlaps(&b));
        assert!(!a.contains(&b));
    }

    #[test]
    fn test_sentence_root() {
        let mut sent = Sentence::new("The cat sat.");
        sent.tokens = vec![
            Token::new("The", 0).with_pos(PosTag::Det),
            Token::new("cat", 1).with_pos(PosTag::Noun),
            Token::new("sat", 2).with_pos(PosTag::Verb),
            Token::new(".", 3).with_pos(PosTag::Punct),
        ];
        sent.dependency_edges = vec![
            DependencyEdge::new(2, 2, DependencyRelation::Root),
            DependencyEdge::new(2, 1, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 0, DependencyRelation::Det),
            DependencyEdge::new(2, 3, DependencyRelation::Punct),
        ];
        assert_eq!(sent.root_index(), Some(2));
        assert_eq!(sent.tokens_with_pos(PosTag::Noun).len(), 1);
    }

    #[test]
    fn test_sentence_features_negation() {
        let mut sent = Sentence::new("The cat did not sit.");
        sent.tokens = vec![
            Token::new("The", 0),
            Token::new("cat", 1),
            Token::new("did", 2),
            Token::new("not", 3),
            Token::new("sit", 4),
            Token::new(".", 5),
        ];
        let feats = sent.compute_features();
        assert!(feats.has_negation);
    }

    #[test]
    fn test_parse_tree_depth() {
        let nodes = vec![
            ParseNode::new_nonterminal("S", 0, 2),
            ParseNode::new_nonterminal("NP", 0, 1),
            ParseNode::new_terminal("NN", "cat", 0),
            ParseNode::new_nonterminal("VP", 1, 2),
            ParseNode::new_terminal("VBD", "sat", 1),
        ];
        let mut tree = ParseTree::new(nodes, 0);
        tree.nodes[0].children = vec![1, 3];
        tree.nodes[1].children = vec![2];
        tree.nodes[3].children = vec![4];
        assert_eq!(tree.depth(), 3);
        assert_eq!(tree.yield_words(), vec!["cat", "sat"]);
    }

    #[test]
    fn test_token_serde_roundtrip() {
        let t = Token::new("hello", 0).with_pos(PosTag::Noun).with_lemma("hello");
        let json = serde_json::to_string(&t).unwrap();
        let t2: Token = serde_json::from_str(&json).unwrap();
        assert_eq!(t, t2);
    }
}
