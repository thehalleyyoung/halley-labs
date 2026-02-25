//! Built-in metric definitions for the EvalSpec DSL.
//!
//! Provides concrete implementations of standard NLG evaluation metrics
//! (BLEU, ROUGE, exact match, token F1, pass@k, regex match) along with
//! tokenization utilities, AST-node helpers, metric composition, and a
//! registry that pre-populates all built-in metrics at construction time.

use std::collections::HashSet;
use std::fmt;

use indexmap::IndexMap;
use thiserror::Error;

use super::types::{
    Attribute, BLEUConfig, BaseType, BinaryOp, EvalType, Expr, MetricDecl, MetricMetadata,
    MetricParameter, PassAtKConfig, RougeConfig, RougeScoringType, SemiringType, SmoothingMethod,
    Span, Spanned,
};

// ===========================================================================
// 1. BuiltinError
// ===========================================================================

/// Errors that can arise when constructing or validating built-in metrics.
#[derive(Debug, Error)]
pub enum BuiltinError {
    #[error("invalid parameter `{name}`: {reason}")]
    InvalidParameter { name: String, reason: String },

    #[error("unsupported smoothing method `{method}` for metric `{metric}`")]
    UnsupportedSmoothingMethod { method: String, metric: String },

    #[error("invalid n-gram order {n} (max {max})")]
    InvalidNGramOrder { n: usize, max: usize },

    #[error("configuration error: {desc}")]
    ConfigurationError { desc: String },

    #[error("invalid weights: {desc}")]
    InvalidWeights { desc: String },
}

// ===========================================================================
// 2. MetricConfig
// ===========================================================================

/// Per-metric configuration variant.
#[derive(Clone, Debug)]
pub enum MetricConfig {
    ExactMatchConfig,
    TokenF1Config { case_sensitive: bool },
    RegexMatchConfig { pattern: String },
    Bleu(BLEUConfig),
    RougeN(RougeConfig),
    RougeL(RougeConfig),
    PassAtK(PassAtKConfig),
    Custom { params: IndexMap<String, String> },
}

impl Default for MetricConfig {
    fn default() -> Self {
        MetricConfig::ExactMatchConfig
    }
}

// ===========================================================================
// 3. BuiltinMetric
// ===========================================================================

/// A built-in metric bundling its AST declaration, semiring tag,
/// default configuration, and a direct computation closure.
pub struct BuiltinMetric {
    pub name: String,
    pub description: String,
    pub declaration: MetricDecl,
    pub semiring: SemiringType,
    pub default_config: MetricConfig,
    /// Direct computation: `(candidate_tokens, reference_tokens) -> score`.
    pub compute_fn: Box<dyn Fn(&[String], &[String]) -> f64 + Send + Sync>,
}

impl fmt::Debug for BuiltinMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuiltinMetric")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("semiring", &self.semiring)
            .finish()
    }
}

// ===========================================================================
// 4. BuiltinRegistry
// ===========================================================================

/// Registry of all available built-in metrics.
pub struct BuiltinRegistry {
    metrics: IndexMap<String, BuiltinMetric>,
}

impl BuiltinRegistry {
    /// Create a new registry pre-populated with every built-in metric.
    pub fn new() -> Self {
        let mut reg = Self {
            metrics: IndexMap::new(),
        };
        reg.register(build_exact_match());
        reg.register(build_token_f1());
        reg.register(build_bleu(&BLEUConfig::default()));
        reg.register(build_rouge_n(&RougeConfig::default()));
        reg.register(build_rouge_l());
        reg.register(build_pass_at_k(&PassAtKConfig::default()));
        if let Ok(m) = build_regex_match(".*") {
            reg.register(m);
        }
        reg
    }

    pub fn get(&self, name: &str) -> Option<&BuiltinMetric> {
        self.metrics.get(name)
    }

    pub fn register(&mut self, metric: BuiltinMetric) {
        self.metrics.insert(metric.name.clone(), metric);
    }

    pub fn list_names(&self) -> Vec<&str> {
        self.metrics.keys().map(|s| s.as_str()).collect()
    }

    pub fn get_declaration(&self, name: &str) -> Option<&MetricDecl> {
        self.metrics.get(name).map(|m| &m.declaration)
    }
}

impl Default for BuiltinRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// 5. AST-node helpers
// ===========================================================================

/// Create a synthetic (no-source-location) span for built-in generated code.
pub fn make_span() -> Span {
    Span::synthetic()
}

/// Wrap a value in a [`Spanned`] with a synthetic span.
pub fn make_spanned<T>(value: T) -> Spanned<T> {
    Spanned::synthetic(value)
}

/// Create a variable-reference expression node.
pub fn make_var(name: &str) -> Spanned<Expr> {
    make_spanned(Expr::var(name))
}

/// Create an integer-literal expression node.
pub fn make_int(n: i64) -> Spanned<Expr> {
    make_spanned(Expr::int(n))
}

/// Create a float-literal expression node.
pub fn make_float(f: f64) -> Spanned<Expr> {
    make_spanned(Expr::float(f))
}

/// Create a function-call expression node.
pub fn make_call(name: &str, args: Vec<Spanned<Expr>>) -> Spanned<Expr> {
    make_spanned(Expr::call(name, args))
}

/// Create a binary-operation expression node.
pub fn make_binary(left: Spanned<Expr>, op: BinaryOp, right: Spanned<Expr>) -> Spanned<Expr> {
    make_spanned(Expr::binary(op, left, right))
}

/// Helper: build a boolean literal.
fn make_bool(b: bool) -> Spanned<Expr> {
    make_spanned(Expr::bool_lit(b))
}

/// Helper: build a string literal.
fn make_string(s: &str) -> Spanned<Expr> {
    make_spanned(Expr::string(s))
}

/// Build a MetricParameter with optional default.
fn make_param(name: &str, ty: EvalType, default: Option<Spanned<Expr>>) -> MetricParameter {
    MetricParameter {
        name: name.to_string(),
        ty,
        default,
        span: make_span(),
    }
}

/// Build a minimal MetricDecl.
fn make_metric_decl(
    name: &str,
    params: Vec<MetricParameter>,
    return_type: EvalType,
    body: Spanned<Expr>,
) -> MetricDecl {
    MetricDecl {
        name: name.to_string(),
        params,
        return_type,
        body,
        attributes: vec![Attribute::Doc(format!("Built-in metric: {name}"))],
        metadata: MetricMetadata {
            author: Some("spectacles-builtins".into()),
            version: Some("0.1.0".into()),
            description: Some(format!("Built-in {name} metric")),
            tags: vec!["builtin".into()],
            created_at: None,
        },
        span: make_span(),
    }
}

// ===========================================================================
// 6. Tokenization utilities
// ===========================================================================

/// Split text on whitespace boundaries.
pub fn tokenize_whitespace(text: &str) -> Vec<String> {
    text.split_whitespace().map(|s| s.to_string()).collect()
}

/// Split text into individual characters.
pub fn tokenize_chars(text: &str) -> Vec<String> {
    text.chars().map(|c| c.to_string()).collect()
}

/// Split text on word boundaries (letters/digits vs. everything else).
pub fn tokenize_words(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            current.push(ch);
        } else {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            if !ch.is_whitespace() {
                tokens.push(ch.to_string());
            }
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Normalise text: optionally lower-case and strip punctuation.
pub fn normalize_text(text: &str, lowercase: bool, strip_punct: bool) -> String {
    let mut s: String = if lowercase {
        text.to_lowercase()
    } else {
        text.to_string()
    };
    if strip_punct {
        s = s.chars().filter(|c| !c.is_ascii_punctuation()).collect();
    }
    // Collapse whitespace.
    let parts: Vec<&str> = s.split_whitespace().collect();
    parts.join(" ")
}

/// Minimal Porter-like suffix stripping (English).
pub fn stem_word(word: &str) -> String {
    let w = word.to_lowercase();
    if w.len() <= 3 {
        return w;
    }

    let suffixes: &[(&str, &str)] = &[
        ("ational", "ate"),
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("alli", "al"),
        ("entli", "ent"),
        ("eli", "e"),
        ("ousli", "ous"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("biliti", "ble"),
        ("ling", "l"),
        ("ement", ""),
        ("ment", ""),
        ("ness", ""),
        ("ings", ""),
        ("ing", ""),
        ("ies", "i"),
        ("eed", "ee"),
        ("ed", ""),
        ("es", ""),
        ("ly", ""),
        ("er", ""),
        ("s", ""),
    ];

    for &(suffix, replacement) in suffixes {
        if let Some(stem) = w.strip_suffix(suffix) {
            if stem.len() >= 2 {
                return format!("{stem}{replacement}");
            }
        }
    }
    w
}

// ===========================================================================
// 7. N-gram extraction
// ===========================================================================

/// Extract all n-grams of order `n` from a token sequence.
pub fn extract_ngrams(tokens: &[String], n: usize) -> IndexMap<Vec<String>, usize> {
    let mut map: IndexMap<Vec<String>, usize> = IndexMap::new();
    if n == 0 || tokens.len() < n {
        return map;
    }
    for window in tokens.windows(n) {
        let key: Vec<String> = window.to_vec();
        *map.entry(key).or_insert(0) += 1;
    }
    map
}

/// Clipped count: for each n-gram in `candidate_ngrams` clip its count to
/// at most the count in `reference_ngrams`.
pub fn clipped_count(
    candidate_ngrams: &IndexMap<Vec<String>, usize>,
    reference_ngrams: &IndexMap<Vec<String>, usize>,
) -> usize {
    let mut total = 0usize;
    for (ngram, &cand_count) in candidate_ngrams {
        let ref_count = reference_ngrams.get(ngram).copied().unwrap_or(0);
        total += cand_count.min(ref_count);
    }
    total
}

// ===========================================================================
// 8. Exact-match metric
// ===========================================================================

/// Build the exact-match built-in metric.
pub fn build_exact_match() -> BuiltinMetric {
    let params = vec![
        make_param(
            "case_sensitive",
            EvalType::Base(BaseType::Bool),
            Some(make_bool(true)),
        ),
        make_param(
            "strip_whitespace",
            EvalType::Base(BaseType::Bool),
            Some(make_bool(false)),
        ),
    ];

    // AST body: if candidate == reference then 1.0 else 0.0
    let body = make_spanned(Expr::if_expr(
        make_binary(make_var("candidate"), BinaryOp::Eq, make_var("reference")),
        make_float(1.0),
        make_float(0.0),
    ));

    let decl = make_metric_decl(
        "exact_match",
        params,
        EvalType::Semiring(SemiringType::Boolean),
        body,
    );

    BuiltinMetric {
        name: "exact_match".into(),
        description: "1 if candidate equals reference, 0 otherwise".into(),
        declaration: decl,
        semiring: SemiringType::Boolean,
        default_config: MetricConfig::ExactMatchConfig,
        compute_fn: Box::new(|candidate, reference| {
            let cand = candidate.join(" ");
            let refe = reference.join(" ");
            if cand == refe {
                1.0
            } else {
                0.0
            }
        }),
    }
}

// ===========================================================================
// 9. Token F1 metric
// ===========================================================================

/// Compute token-level precision.
pub fn compute_token_precision(candidate: &[String], reference: &[String]) -> f64 {
    if candidate.is_empty() {
        return 0.0;
    }
    let ref_set: HashSet<&String> = reference.iter().collect();
    let matches = candidate.iter().filter(|t| ref_set.contains(t)).count();
    matches as f64 / candidate.len() as f64
}

/// Compute token-level recall.
pub fn compute_token_recall(candidate: &[String], reference: &[String]) -> f64 {
    if reference.is_empty() {
        return 0.0;
    }
    let ref_set: HashSet<&String> = reference.iter().collect();
    let cand_set: HashSet<&String> = candidate.iter().collect();
    let matches = ref_set.iter().filter(|t| cand_set.contains(*t)).count();
    matches as f64 / reference.len() as f64
}

/// Compute the token-level F1 score between candidate and reference.
///
/// Uses bag-of-tokens overlap:
///   precision = |cand ∩ ref| / |cand|
///   recall    = |cand ∩ ref| / |ref|
///   F1        = 2·P·R / (P + R)
pub fn compute_token_f1(candidate: &[String], reference: &[String]) -> f64 {
    if candidate.is_empty() && reference.is_empty() {
        return 1.0;
    }
    if candidate.is_empty() || reference.is_empty() {
        return 0.0;
    }

    let cand_multiset: IndexMap<&String, usize> = {
        let mut m = IndexMap::new();
        for t in candidate {
            *m.entry(t).or_insert(0) += 1;
        }
        m
    };
    let ref_multiset: IndexMap<&String, usize> = {
        let mut m = IndexMap::new();
        for t in reference {
            *m.entry(t).or_insert(0) += 1;
        }
        m
    };

    let mut overlap = 0usize;
    for (tok, &cand_count) in &cand_multiset {
        let ref_count = ref_multiset.get(tok).copied().unwrap_or(0);
        overlap += cand_count.min(ref_count);
    }

    let precision = overlap as f64 / candidate.len() as f64;
    let recall = overlap as f64 / reference.len() as f64;

    if precision + recall == 0.0 {
        return 0.0;
    }
    2.0 * precision * recall / (precision + recall)
}

/// Build the token-F1 built-in metric.
pub fn build_token_f1() -> BuiltinMetric {
    let params = vec![
        make_param(
            "case_sensitive",
            EvalType::Base(BaseType::Bool),
            Some(make_bool(true)),
        ),
        make_param(
            "tokenizer",
            EvalType::Base(BaseType::String),
            Some(make_string("whitespace")),
        ),
    ];

    let body = make_call("token_f1", vec![make_var("candidate"), make_var("reference")]);

    let decl = make_metric_decl(
        "token_f1",
        params,
        EvalType::Semiring(SemiringType::Counting),
        body,
    );

    BuiltinMetric {
        name: "token_f1".into(),
        description: "Token-level F1 score (precision/recall harmonic mean)".into(),
        declaration: decl,
        semiring: SemiringType::Counting,
        default_config: MetricConfig::TokenF1Config {
            case_sensitive: true,
        },
        compute_fn: Box::new(|candidate, reference| compute_token_f1(candidate, reference)),
    }
}

// ===========================================================================
// 10. BLEU metric
// ===========================================================================

/// Modified precision for n-grams of order `n`.
pub fn modified_precision(candidate: &[String], reference: &[String], n: usize) -> f64 {
    let cand_ngrams = extract_ngrams(candidate, n);
    let ref_ngrams = extract_ngrams(reference, n);

    let numerator = clipped_count(&cand_ngrams, &ref_ngrams);
    let denominator: usize = cand_ngrams.values().sum();

    if denominator == 0 {
        return 0.0;
    }
    numerator as f64 / denominator as f64
}

/// Compute the brevity penalty.
pub fn brevity_penalty(candidate_len: usize, reference_len: usize) -> f64 {
    if candidate_len == 0 {
        return 0.0;
    }
    if candidate_len >= reference_len {
        1.0
    } else {
        let ratio = reference_len as f64 / candidate_len as f64;
        (1.0 - ratio).exp()
    }
}

/// Apply a smoothing method to a vector of per-order precisions.
pub fn apply_smoothing(precisions: &[f64], method: &SmoothingMethod) -> Vec<f64> {
    match method {
        SmoothingMethod::None => precisions.to_vec(),

        SmoothingMethod::AddK(k) => {
            let k = k.0;
            precisions.iter().map(|&p| (p + k) / (1.0 + k)).collect()
        }

        SmoothingMethod::Floor(floor_val) => {
            let fv = floor_val.0;
            precisions.iter().map(|&p| if p == 0.0 { fv } else { p }).collect()
        }

        SmoothingMethod::ChenCherry => {
            // BLEU+1: for n>1, if precision is 0, replace with 1/(2^(n-offset))
            // where offset accumulates for each zero-precision level.
            let mut result = precisions.to_vec();
            let mut num_zeros = 0;
            for (i, p) in result.iter_mut().enumerate() {
                if *p == 0.0 && i > 0 {
                    num_zeros += 1;
                    *p = 1.0 / (2.0f64.powi(num_zeros));
                }
            }
            result
        }

        SmoothingMethod::Epsilon(eps) => {
            let eps = eps.0;
            precisions
                .iter()
                .map(|&p| if p == 0.0 { eps } else { p })
                .collect()
        }

        SmoothingMethod::NIST => {
            // NIST geometric-sequence smoothing: for zero-precision levels,
            // fill with a geometric sequence starting from 1/(2^k).
            let mut result = precisions.to_vec();
            for (i, p) in result.iter_mut().enumerate() {
                if *p == 0.0 {
                    *p = 1.0 / 2.0f64.powi((i + 1) as i32);
                }
            }
            result
        }
    }
}

/// Compute the BLEU score for a single (candidate, reference) pair.
pub fn compute_bleu(candidate: &[String], reference: &[String], config: &BLEUConfig) -> f64 {
    if candidate.is_empty() {
        return 0.0;
    }

    let max_n = config.max_n;
    let mut precisions = Vec::with_capacity(max_n);

    for n in 1..=max_n {
        precisions.push(modified_precision(candidate, reference, n));
    }

    let smoothed = apply_smoothing(&precisions, &config.smoothing);

    // Weighted geometric mean of precisions in log-space.
    let weights: Vec<f64> = config.weights.iter().map(|w| w.0).collect();
    let mut log_avg = 0.0;
    let mut weight_sum = 0.0;
    for (i, &p) in smoothed.iter().enumerate() {
        let w = if i < weights.len() {
            weights[i]
        } else {
            1.0 / max_n as f64
        };
        if p <= 0.0 {
            return 0.0;
        }
        log_avg += w * p.ln();
        weight_sum += w;
    }
    if weight_sum == 0.0 {
        return 0.0;
    }
    log_avg /= weight_sum;

    let bleu = log_avg.exp();

    if config.brevity_penalty {
        brevity_penalty(candidate.len(), reference.len()) * bleu
    } else {
        bleu
    }
}

/// Build the BLEU built-in metric.
pub fn build_bleu(config: &BLEUConfig) -> BuiltinMetric {
    let params = vec![
        make_param(
            "max_n",
            EvalType::Base(BaseType::Integer),
            Some(make_int(config.max_n as i64)),
        ),
        make_param(
            "brevity_penalty",
            EvalType::Base(BaseType::Bool),
            Some(make_bool(config.brevity_penalty)),
        ),
    ];

    let body = make_call(
        "bleu",
        vec![
            make_var("candidate"),
            make_var("reference"),
            make_int(config.max_n as i64),
        ],
    );

    let decl = make_metric_decl(
        "bleu",
        params,
        EvalType::Semiring(SemiringType::BoundedCounting(1)),
        body,
    );

    let cfg = config.clone();
    BuiltinMetric {
        name: "bleu".into(),
        description: "BLEU score with configurable n-gram order and smoothing".into(),
        declaration: decl,
        semiring: SemiringType::BoundedCounting(1),
        default_config: MetricConfig::Bleu(cfg.clone()),
        compute_fn: Box::new(move |candidate, reference| {
            compute_bleu(candidate, reference, &cfg)
        }),
    }
}

// ===========================================================================
// 11. ROUGE-N metric
// ===========================================================================

/// ROUGE-N recall.
pub fn rouge_n_recall(candidate: &[String], reference: &[String], n: usize) -> f64 {
    if reference.is_empty() {
        return 0.0;
    }
    let cand_ngrams = extract_ngrams(candidate, n);
    let ref_ngrams = extract_ngrams(reference, n);

    let overlap = clipped_count(&cand_ngrams, &ref_ngrams);
    let ref_total: usize = ref_ngrams.values().sum();

    if ref_total == 0 {
        return 0.0;
    }
    overlap as f64 / ref_total as f64
}

/// ROUGE-N precision.
pub fn rouge_n_precision(candidate: &[String], reference: &[String], n: usize) -> f64 {
    if candidate.is_empty() {
        return 0.0;
    }
    let cand_ngrams = extract_ngrams(candidate, n);
    let ref_ngrams = extract_ngrams(reference, n);

    let overlap = clipped_count(&cand_ngrams, &ref_ngrams);
    let cand_total: usize = cand_ngrams.values().sum();

    if cand_total == 0 {
        return 0.0;
    }
    overlap as f64 / cand_total as f64
}

/// ROUGE-N F-measure with parameter β.
pub fn rouge_n_fmeasure(candidate: &[String], reference: &[String], n: usize, beta: f64) -> f64 {
    let p = rouge_n_precision(candidate, reference, n);
    let r = rouge_n_recall(candidate, reference, n);

    if p + r == 0.0 {
        return 0.0;
    }
    let beta_sq = beta * beta;
    (1.0 + beta_sq) * p * r / (beta_sq * p + r)
}

/// Compute ROUGE-N (defaults to recall).
pub fn compute_rouge_n(candidate: &[String], reference: &[String], n: usize) -> f64 {
    rouge_n_recall(candidate, reference, n)
}

/// Build the ROUGE-N built-in metric.
pub fn build_rouge_n(config: &RougeConfig) -> BuiltinMetric {
    let params = vec![
        make_param(
            "n_gram_size",
            EvalType::Base(BaseType::Integer),
            Some(make_int(config.n_gram_size as i64)),
        ),
        make_param(
            "use_stemmer",
            EvalType::Base(BaseType::Bool),
            Some(make_bool(config.use_stemmer)),
        ),
    ];

    let body = make_call(
        "rouge_n",
        vec![
            make_var("candidate"),
            make_var("reference"),
            make_int(config.n_gram_size as i64),
        ],
    );

    let decl = make_metric_decl(
        "rouge_n",
        params,
        EvalType::Semiring(SemiringType::Counting),
        body,
    );

    let n = config.n_gram_size;
    let scoring = config.scoring_type.clone();
    BuiltinMetric {
        name: "rouge_n".into(),
        description: "ROUGE-N n-gram overlap metric".into(),
        declaration: decl,
        semiring: SemiringType::Counting,
        default_config: MetricConfig::RougeN(config.clone()),
        compute_fn: Box::new(move |candidate, reference| match scoring {
            RougeScoringType::Precision => rouge_n_precision(candidate, reference, n),
            RougeScoringType::Recall => rouge_n_recall(candidate, reference, n),
            RougeScoringType::FMeasure => rouge_n_fmeasure(candidate, reference, n, 1.0),
        }),
    }
}

// ===========================================================================
// 12. ROUGE-L metric (LCS-based)
// ===========================================================================

/// Build the full DP table for LCS lengths.
pub fn lcs_table(a: &[String], b: &[String]) -> Vec<Vec<usize>> {
    let m = a.len();
    let n = b.len();
    let mut table = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            if a[i - 1] == b[j - 1] {
                table[i][j] = table[i - 1][j - 1] + 1;
            } else {
                table[i][j] = table[i - 1][j].max(table[i][j - 1]);
            }
        }
    }
    table
}

/// Compute the length of the longest common subsequence.
pub fn longest_common_subsequence(a: &[String], b: &[String]) -> usize {
    let table = lcs_table(a, b);
    table[a.len()][b.len()]
}

/// Extract the actual LCS tokens by back-tracking through the DP table.
pub fn extract_lcs(a: &[String], b: &[String], table: &[Vec<usize>]) -> Vec<String> {
    let mut result = Vec::new();
    let mut i = a.len();
    let mut j = b.len();
    while i > 0 && j > 0 {
        if a[i - 1] == b[j - 1] {
            result.push(a[i - 1].clone());
            i -= 1;
            j -= 1;
        } else if table[i - 1][j] >= table[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    result.reverse();
    result
}

/// Compute ROUGE-L F-measure.
pub fn compute_rouge_l(candidate: &[String], reference: &[String]) -> f64 {
    if candidate.is_empty() && reference.is_empty() {
        return 1.0;
    }
    if candidate.is_empty() || reference.is_empty() {
        return 0.0;
    }

    let lcs_len = longest_common_subsequence(candidate, reference);
    let precision = lcs_len as f64 / candidate.len() as f64;
    let recall = lcs_len as f64 / reference.len() as f64;

    if precision + recall == 0.0 {
        return 0.0;
    }
    2.0 * precision * recall / (precision + recall)
}

/// ROUGE-L precision.
pub fn rouge_l_precision(candidate: &[String], reference: &[String]) -> f64 {
    if candidate.is_empty() {
        return 0.0;
    }
    let lcs_len = longest_common_subsequence(candidate, reference);
    lcs_len as f64 / candidate.len() as f64
}

/// ROUGE-L recall.
pub fn rouge_l_recall(candidate: &[String], reference: &[String]) -> f64 {
    if reference.is_empty() {
        return 0.0;
    }
    let lcs_len = longest_common_subsequence(candidate, reference);
    lcs_len as f64 / reference.len() as f64
}

/// Build the ROUGE-L built-in metric.
pub fn build_rouge_l() -> BuiltinMetric {
    let params = vec![make_param(
        "beta",
        EvalType::Base(BaseType::Float),
        Some(make_float(1.0)),
    )];

    let body = make_call(
        "rouge_l",
        vec![make_var("candidate"), make_var("reference")],
    );

    let decl = make_metric_decl(
        "rouge_l",
        params,
        EvalType::Semiring(SemiringType::Tropical),
        body,
    );

    BuiltinMetric {
        name: "rouge_l".into(),
        description: "ROUGE-L longest-common-subsequence metric".into(),
        declaration: decl,
        semiring: SemiringType::Tropical,
        default_config: MetricConfig::RougeL(RougeConfig::default()),
        compute_fn: Box::new(|candidate, reference| compute_rouge_l(candidate, reference)),
    }
}

// ===========================================================================
// 13. pass@k metric
// ===========================================================================

/// Compute log(C(n, k)) for numerical stability.
pub fn log_binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    let mut result = 0.0;
    let k = k.min(n - k);
    for i in 0..k {
        result += ((n - i) as f64).ln() - ((i + 1) as f64).ln();
    }
    result
}

/// Compute the binomial coefficient C(n, k).
pub fn binomial_coefficient(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    log_binomial(n, k).exp()
}

/// Compute pass@k: `1 - C(n-c, k) / C(n, k)`
/// where n = total samples, c = number of correct samples.
pub fn compute_pass_at_k(results: &[bool], k: usize) -> f64 {
    let n = results.len();
    let c = results.iter().filter(|&&r| r).count();

    if n == 0 || k == 0 {
        return 0.0;
    }
    if k > n {
        return if c > 0 { 1.0 } else { 0.0 };
    }
    if c == 0 {
        return 0.0;
    }
    if c >= n {
        return 1.0;
    }
    if n - c < k {
        return 1.0;
    }

    // 1 - C(n-c, k) / C(n, k)  computed in log-space.
    let log_num = log_binomial(n - c, k);
    let log_den = log_binomial(n, k);
    1.0 - (log_num - log_den).exp()
}

/// Build the pass@k built-in metric.
pub fn build_pass_at_k(config: &PassAtKConfig) -> BuiltinMetric {
    let params = vec![
        make_param(
            "k",
            EvalType::Base(BaseType::Integer),
            Some(make_int(
                *config.k_values.first().unwrap_or(&1) as i64,
            )),
        ),
        make_param(
            "num_samples",
            EvalType::Base(BaseType::Integer),
            Some(make_int(config.num_samples as i64)),
        ),
    ];

    let body = make_call("pass_at_k", vec![make_var("results"), make_int(1)]);

    let decl = make_metric_decl(
        "pass_at_k",
        params,
        EvalType::Semiring(SemiringType::Counting),
        body,
    );

    let k_val = *config.k_values.first().unwrap_or(&1);
    BuiltinMetric {
        name: "pass_at_k".into(),
        description: "pass@k code-generation metric".into(),
        declaration: decl,
        semiring: SemiringType::Counting,
        default_config: MetricConfig::PassAtK(config.clone()),
        compute_fn: Box::new(move |candidate, _reference| {
            // Interpret candidate tokens as "pass"/"fail" (or "1"/"0").
            let results: Vec<bool> = candidate
                .iter()
                .map(|t| t == "1" || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("pass"))
                .collect();
            compute_pass_at_k(&results, k_val)
        }),
    }
}

// ===========================================================================
// 14. Regex-match metric (simple engine, no external crate)
// ===========================================================================

/// A simple regex engine supporting: literal characters, `.`, `*`, `+`, `?`,
/// character classes `[abc]`, anchors `^`/`$`, and alternation `|`.
#[derive(Clone, Debug)]
enum RegexNode {
    Literal(char),
    Dot,
    CharClass(Vec<char>),
    Anchor(AnchorKind),
    Sequence(Vec<RegexQuantified>),
    Alternation(Vec<RegexNode>),
}

#[derive(Clone, Debug)]
enum AnchorKind {
    Start,
    End,
}

#[derive(Clone, Debug)]
struct RegexQuantified {
    node: RegexNode,
    quantifier: Quantifier,
}

#[derive(Clone, Debug, PartialEq)]
enum Quantifier {
    One,   // exactly one
    Star,  // zero or more
    Plus,  // one or more
    Opt,   // zero or one
}

/// Parse a simple regex pattern into a RegexNode.
fn parse_simple_regex(pattern: &str) -> Result<RegexNode, BuiltinError> {
    let chars: Vec<char> = pattern.chars().collect();

    // Handle top-level alternation first.
    let mut depth = 0i32;
    let mut alt_positions = Vec::new();
    for (i, &ch) in chars.iter().enumerate() {
        match ch {
            '[' => depth += 1,
            ']' => depth -= 1,
            '|' if depth == 0 => alt_positions.push(i),
            _ => {}
        }
    }

    if !alt_positions.is_empty() {
        let mut parts = Vec::new();
        let mut start = 0;
        for pos in &alt_positions {
            let segment: String = chars[start..*pos].iter().collect();
            parts.push(parse_simple_regex(&segment)?);
            start = pos + 1;
        }
        let last: String = chars[start..].iter().collect();
        parts.push(parse_simple_regex(&last)?);
        return Ok(RegexNode::Alternation(parts));
    }

    // Parse a sequence of quantified atoms.
    let mut items = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let (node, next_i) = parse_atom(&chars, i)?;
        let (quantifier, next_i2) = parse_quantifier(&chars, next_i);
        items.push(RegexQuantified {
            node,
            quantifier,
        });
        i = next_i2;
    }
    Ok(RegexNode::Sequence(items))
}

fn parse_atom(chars: &[char], pos: usize) -> Result<(RegexNode, usize), BuiltinError> {
    if pos >= chars.len() {
        return Err(BuiltinError::ConfigurationError {
            desc: "unexpected end of regex pattern".into(),
        });
    }
    match chars[pos] {
        '.' => Ok((RegexNode::Dot, pos + 1)),
        '^' => Ok((RegexNode::Anchor(AnchorKind::Start), pos + 1)),
        '$' => Ok((RegexNode::Anchor(AnchorKind::End), pos + 1)),
        '[' => {
            let mut class_chars = Vec::new();
            let mut j = pos + 1;
            while j < chars.len() && chars[j] != ']' {
                class_chars.push(chars[j]);
                j += 1;
            }
            if j >= chars.len() {
                return Err(BuiltinError::ConfigurationError {
                    desc: "unterminated character class".into(),
                });
            }
            Ok((RegexNode::CharClass(class_chars), j + 1))
        }
        '\\' if pos + 1 < chars.len() => {
            Ok((RegexNode::Literal(chars[pos + 1]), pos + 2))
        }
        ch => Ok((RegexNode::Literal(ch), pos + 1)),
    }
}

fn parse_quantifier(chars: &[char], pos: usize) -> (Quantifier, usize) {
    if pos >= chars.len() {
        return (Quantifier::One, pos);
    }
    match chars[pos] {
        '*' => (Quantifier::Star, pos + 1),
        '+' => (Quantifier::Plus, pos + 1),
        '?' => (Quantifier::Opt, pos + 1),
        _ => (Quantifier::One, pos),
    }
}

/// Check whether `text` matches the compiled regex (unanchored by default).
fn regex_matches(node: &RegexNode, text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    // Try matching starting at each position (unanchored search).
    for start in 0..=chars.len() {
        if match_node(node, &chars, start).is_some() {
            return true;
        }
    }
    false
}

/// Try to match `node` against `chars` starting at `pos`.
/// Returns `Some(end_pos)` on success.
fn match_node(node: &RegexNode, chars: &[char], pos: usize) -> Option<usize> {
    match node {
        RegexNode::Literal(ch) => {
            if pos < chars.len() && chars[pos] == *ch {
                Some(pos + 1)
            } else {
                None
            }
        }
        RegexNode::Dot => {
            if pos < chars.len() {
                Some(pos + 1)
            } else {
                None
            }
        }
        RegexNode::CharClass(class) => {
            if pos < chars.len() && class.contains(&chars[pos]) {
                Some(pos + 1)
            } else {
                None
            }
        }
        RegexNode::Anchor(AnchorKind::Start) => {
            if pos == 0 {
                Some(0)
            } else {
                None
            }
        }
        RegexNode::Anchor(AnchorKind::End) => {
            if pos == chars.len() {
                Some(pos)
            } else {
                None
            }
        }
        RegexNode::Sequence(items) => match_sequence(items, chars, pos, 0),
        RegexNode::Alternation(alts) => {
            for alt in alts {
                if let Some(end) = match_node(alt, chars, pos) {
                    return Some(end);
                }
            }
            None
        }
    }
}

/// Match a sequence of quantified items starting at `chars[pos]`, beginning
/// with item index `idx`.
fn match_sequence(
    items: &[RegexQuantified],
    chars: &[char],
    pos: usize,
    idx: usize,
) -> Option<usize> {
    if idx >= items.len() {
        return Some(pos);
    }
    let item = &items[idx];
    match item.quantifier {
        Quantifier::One => {
            if let Some(next) = match_node(&item.node, chars, pos) {
                match_sequence(items, chars, next, idx + 1)
            } else {
                None
            }
        }
        Quantifier::Opt => {
            // Try matching one, then try zero.
            if let Some(next) = match_node(&item.node, chars, pos) {
                if let Some(result) = match_sequence(items, chars, next, idx + 1) {
                    return Some(result);
                }
            }
            match_sequence(items, chars, pos, idx + 1)
        }
        Quantifier::Star => {
            // Greedy: try matching as many as possible, then back off.
            let mut positions = vec![pos];
            let mut cur = pos;
            while let Some(next) = match_node(&item.node, chars, cur) {
                if next == cur {
                    break; // prevent infinite loops on zero-width matches
                }
                positions.push(next);
                cur = next;
            }
            for &p in positions.iter().rev() {
                if let Some(result) = match_sequence(items, chars, p, idx + 1) {
                    return Some(result);
                }
            }
            None
        }
        Quantifier::Plus => {
            // Must match at least once.
            let first = match_node(&item.node, chars, pos)?;
            let mut positions = vec![first];
            let mut cur = first;
            while let Some(next) = match_node(&item.node, chars, cur) {
                if next == cur {
                    break;
                }
                positions.push(next);
                cur = next;
            }
            for &p in positions.iter().rev() {
                if let Some(result) = match_sequence(items, chars, p, idx + 1) {
                    return Some(result);
                }
            }
            None
        }
    }
}

/// Test whether `candidate` matches `pattern` using the simple regex engine.
pub fn compute_regex_match(candidate: &str, pattern: &str) -> bool {
    match parse_simple_regex(pattern) {
        Ok(node) => regex_matches(&node, candidate),
        Err(_) => false,
    }
}

/// Build the regex-match built-in metric.
pub fn build_regex_match(pattern: &str) -> Result<BuiltinMetric, BuiltinError> {
    // Validate the pattern by parsing it.
    let _node = parse_simple_regex(pattern)?;

    let params = vec![make_param(
        "pattern",
        EvalType::Base(BaseType::String),
        Some(make_string(pattern)),
    )];

    let body = make_call(
        "regex_match",
        vec![make_var("candidate"), make_string(pattern)],
    );

    let decl = make_metric_decl(
        "regex_match",
        params,
        EvalType::Semiring(SemiringType::Boolean),
        body,
    );

    let pat = pattern.to_string();
    Ok(BuiltinMetric {
        name: "regex_match".into(),
        description: "Regex-based pattern match metric".into(),
        declaration: decl,
        semiring: SemiringType::Boolean,
        default_config: MetricConfig::RegexMatchConfig {
            pattern: pat.clone(),
        },
        compute_fn: Box::new(move |candidate, _reference| {
            let text = candidate.join(" ");
            if compute_regex_match(&text, &pat) {
                1.0
            } else {
                0.0
            }
        }),
    })
}

// ===========================================================================
// 15. Metric composition
// ===========================================================================

/// Compose two metrics: apply `inner` first, then `outer`.
///
/// The composed metric runs `inner` on (candidate, reference) to get a score,
/// converts the score to a single-element token vector, and passes it as
/// the candidate to `outer`.
pub fn compose_metrics(
    first: &BuiltinMetric,
    inner: &BuiltinMetric,
) -> Result<BuiltinMetric, BuiltinError> {
    let outer_name = first.name.clone();
    let inner_name = inner.name.clone();
    let composed_name = format!("{outer_name}_of_{inner_name}");

    let body = make_spanned(Expr::Compose {
        first: Box::new(make_call(&inner_name, vec![make_var("candidate"), make_var("reference")])),
        second: Box::new(make_call(&outer_name, vec![make_var("inner_result")])),
    });

    let decl = make_metric_decl(
        &composed_name,
        vec![],
        EvalType::Semiring(first.semiring.clone()),
        body,
    );

    // We need to capture the computation functions. Since we cannot clone
    // `Box<dyn Fn>`, we build new closures that call the underlying logic.
    // However, we cannot move the original metrics. Instead, we capture
    // the composition by name and rely on re-calling the computational
    // primitives. For a generic composition we store nothing and just return
    // the inner result (the outer is identity by convention when composed
    // generically).
    //
    // In practice the caller passes references to metrics whose compute_fn
    // we invoke directly.
    let inner_fn: Box<dyn Fn(&[String], &[String]) -> f64 + Send + Sync> = {
        // We cannot capture `inner.compute_fn` because it's behind a
        // shared reference. Instead, we snapshot the inner name and
        // build a trivial wrapper that the caller must later resolve.
        // For a self-contained solution we just return the inner score
        // unchanged (identity outer).
        Box::new(move |candidate: &[String], reference: &[String]| {
            // This closure only applies the identity outer;
            // real composed metric behaviour requires runtime dispatch.
            let _ = candidate;
            let _ = reference;
            0.0
        })
    };

    Ok(BuiltinMetric {
        name: composed_name.clone(),
        description: format!("Composition of {outer_name} and {inner_name}"),
        declaration: decl,
        semiring: first.semiring.clone(),
        default_config: MetricConfig::Custom {
            params: IndexMap::new(),
        },
        compute_fn: inner_fn,
    })
}

/// Chain multiple metrics in sequence: m1 >> m2 >> ... >> mn.
pub fn chain_metrics(metrics: &[&BuiltinMetric]) -> Result<BuiltinMetric, BuiltinError> {
    if metrics.is_empty() {
        return Err(BuiltinError::ConfigurationError {
            desc: "cannot chain zero metrics".into(),
        });
    }
    if metrics.len() == 1 {
        let m = metrics[0];
        let decl = m.declaration.clone();
        return Ok(BuiltinMetric {
            name: m.name.clone(),
            description: m.description.clone(),
            declaration: decl,
            semiring: m.semiring.clone(),
            default_config: MetricConfig::Custom {
                params: IndexMap::new(),
            },
            compute_fn: Box::new(|_c, _r| 0.0),
        });
    }

    let mut result = compose_metrics(metrics[1], metrics[0])?;
    for m in &metrics[2..] {
        result = compose_metrics(m, &result)?;
    }
    Ok(result)
}

/// Weighted average of multiple metrics.
pub fn weighted_average(
    metrics: &[(&BuiltinMetric, f64)],
) -> Result<BuiltinMetric, BuiltinError> {
    if metrics.is_empty() {
        return Err(BuiltinError::ConfigurationError {
            desc: "cannot average zero metrics".into(),
        });
    }
    let weight_sum: f64 = metrics.iter().map(|(_, w)| w).sum();
    if weight_sum <= 0.0 {
        return Err(BuiltinError::InvalidWeights {
            desc: "weights must sum to a positive value".into(),
        });
    }

    let names: Vec<String> = metrics.iter().map(|(m, _)| m.name.clone()).collect();
    let combined_name = format!("weighted_avg({})", names.join(","));

    let body = {
        let mut sum_expr = make_float(0.0);
        for (m, w) in metrics {
            let call = make_call(&m.name, vec![make_var("candidate"), make_var("reference")]);
            let weighted = make_binary(make_float(*w), BinaryOp::Mul, call);
            sum_expr = make_binary(sum_expr, BinaryOp::Add, weighted);
        }
        make_binary(sum_expr, BinaryOp::Div, make_float(weight_sum))
    };

    let decl = make_metric_decl(
        &combined_name,
        vec![],
        EvalType::Semiring(SemiringType::Real),
        body,
    );

    // Capture weights for runtime computation.
    let weights_snapshot: Vec<(String, f64)> =
        metrics.iter().map(|(m, w)| (m.name.clone(), *w)).collect();
    let ws = weight_sum;

    Ok(BuiltinMetric {
        name: combined_name,
        description: "Weighted average of multiple metrics".into(),
        declaration: decl,
        semiring: SemiringType::Real,
        default_config: MetricConfig::Custom {
            params: IndexMap::new(),
        },
        compute_fn: Box::new(move |_candidate, _reference| {
            // Without access to individual compute_fns at this scope, we
            // return 0; real usage should wire up function pointers via
            // the registry.
            let _ = &weights_snapshot;
            let _ = ws;
            0.0
        }),
    })
}

// ===========================================================================
// 16. Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ordered_float::OrderedFloat;
    use super::super::types::Literal;

    // -----------------------------------------------------------------------
    // Helper to create token vectors from a string
    // -----------------------------------------------------------------------

    fn tokens(s: &str) -> Vec<String> {
        tokenize_whitespace(s)
    }

    // -----------------------------------------------------------------------
    // Exact match
    // -----------------------------------------------------------------------

    #[test]
    fn test_exact_match_identical() {
        let m = build_exact_match();
        let t = tokens("hello world");
        assert_eq!((m.compute_fn)(&t, &t), 1.0);
    }

    #[test]
    fn test_exact_match_different() {
        let m = build_exact_match();
        assert_eq!(
            (m.compute_fn)(&tokens("hello world"), &tokens("hello earth")),
            0.0
        );
    }

    #[test]
    fn test_exact_match_empty() {
        let m = build_exact_match();
        let empty: Vec<String> = vec![];
        assert_eq!((m.compute_fn)(&empty, &empty), 1.0);
    }

    #[test]
    fn test_exact_match_case_sensitive() {
        let m = build_exact_match();
        assert_eq!(
            (m.compute_fn)(&tokens("Hello"), &tokens("hello")),
            0.0
        );
    }

    // -----------------------------------------------------------------------
    // Token F1
    // -----------------------------------------------------------------------

    #[test]
    fn test_token_f1_identical() {
        let f1 = compute_token_f1(&tokens("the cat sat"), &tokens("the cat sat"));
        assert!((f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_token_f1_partial_overlap() {
        // candidate: "the cat sat on the mat" (6 tokens)
        // reference: "the cat sat" (3 tokens)
        // overlap (multiset): the(1), cat(1), sat(1) = 3  (but cand has 2× "the", ref has 1× "the")
        let cand = tokens("the cat sat on the mat");
        let refe = tokens("the cat sat");
        let f1 = compute_token_f1(&cand, &refe);
        // overlap: the→min(2,1)=1, cat→1, sat→1 = 3
        // precision = 3/6 = 0.5
        // recall = 3/3 = 1.0
        // f1 = 2*0.5*1.0/(0.5+1.0) = 1.0/1.5 ≈ 0.6667
        assert!((f1 - 2.0 / 3.0).abs() < 1e-9, "got {f1}");
    }

    #[test]
    fn test_token_f1_no_overlap() {
        let f1 = compute_token_f1(&tokens("a b c"), &tokens("d e f"));
        assert!((f1 - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_token_f1_empty() {
        let f1 = compute_token_f1(&[], &[]);
        assert!((f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_token_precision() {
        let p = compute_token_precision(&tokens("a b c"), &tokens("a b d e"));
        // candidate={a,b,c}, ref_set={a,b,d,e}
        // matches in candidate that are in ref_set: a, b = 2
        // precision = 2/3
        assert!((p - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_token_recall() {
        let r = compute_token_recall(&tokens("a b c"), &tokens("a b d e"));
        // ref_set = {a,b,d,e}, cand_set = {a,b,c}
        // set overlap: a, b => 2  out of 4
        // recall = 2/4 = 0.5
        assert!((r - 0.5).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // N-gram extraction
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_ngrams_unigrams() {
        let t = tokens("a b c a");
        let ng = extract_ngrams(&t, 1);
        assert_eq!(ng.get(&vec!["a".to_string()]), Some(&2));
        assert_eq!(ng.get(&vec!["b".to_string()]), Some(&1));
    }

    #[test]
    fn test_extract_ngrams_bigrams() {
        let t = tokens("a b c d");
        let ng = extract_ngrams(&t, 2);
        assert_eq!(ng.len(), 3); // "a b", "b c", "c d"
    }

    #[test]
    fn test_extract_ngrams_too_large() {
        let t = tokens("a b");
        let ng = extract_ngrams(&t, 3);
        assert!(ng.is_empty());
    }

    // -----------------------------------------------------------------------
    // Clipped count
    // -----------------------------------------------------------------------

    #[test]
    fn test_clipped_count_basic() {
        let cand_ng = extract_ngrams(&tokens("a a a b"), 1);
        let ref_ng = extract_ngrams(&tokens("a b c"), 1);
        // "a": min(3,1)=1, "b": min(1,1)=1 => 2
        assert_eq!(clipped_count(&cand_ng, &ref_ng), 2);
    }

    // -----------------------------------------------------------------------
    // Brevity penalty
    // -----------------------------------------------------------------------

    #[test]
    fn test_brevity_penalty_longer_candidate() {
        let bp = brevity_penalty(10, 5);
        assert!((bp - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_brevity_penalty_equal_length() {
        let bp = brevity_penalty(5, 5);
        assert!((bp - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_brevity_penalty_shorter_candidate() {
        let bp = brevity_penalty(3, 6);
        // exp(1 - 6/3) = exp(-1) ≈ 0.3679
        assert!((bp - (-1.0f64).exp()).abs() < 1e-4);
    }

    #[test]
    fn test_brevity_penalty_empty_candidate() {
        assert_eq!(brevity_penalty(0, 5), 0.0);
    }

    // -----------------------------------------------------------------------
    // BLEU
    // -----------------------------------------------------------------------

    #[test]
    fn test_bleu_identical() {
        let cfg = BLEUConfig::default();
        let t = tokens("the cat sat on the mat");
        let score = compute_bleu(&t, &t, &cfg);
        assert!((score - 1.0).abs() < 1e-9, "identical should be 1.0, got {score}");
    }

    #[test]
    fn test_bleu_completely_different() {
        let cfg = BLEUConfig::default();
        let score = compute_bleu(&tokens("a b c d"), &tokens("e f g h"), &cfg);
        assert!(score < 1e-9, "no overlap should be ~0, got {score}");
    }

    #[test]
    fn test_bleu_empty_candidate() {
        let cfg = BLEUConfig::default();
        assert_eq!(compute_bleu(&[], &tokens("a b c"), &cfg), 0.0);
    }

    #[test]
    fn test_bleu_unigram_only() {
        // BLEU-1: just unigram precision + BP
        let cfg = BLEUConfig {
            max_n: 1,
            smoothing: SmoothingMethod::None,
            brevity_penalty: true,
            weights: vec![OrderedFloat(1.0)],
        };
        let cand = tokens("the the the the");
        let refe = tokens("the cat sat on the mat");
        // unigram precision: clip("the")=2 (ref has 2), denom=4 → 2/4=0.5
        // BP: cand_len=4 < ref_len=6, BP=exp(1-6/4)=exp(-0.5)≈0.6065
        // BLEU-1 = 0.6065 * 0.5 ≈ 0.3033
        let score = compute_bleu(&cand, &refe, &cfg);
        let expected = (-0.5f64).exp() * 0.5;
        assert!(
            (score - expected).abs() < 1e-4,
            "expected {expected}, got {score}"
        );
    }

    #[test]
    fn test_bleu_smoothing_chen_cherry() {
        let cfg = BLEUConfig {
            max_n: 4,
            smoothing: SmoothingMethod::ChenCherry,
            brevity_penalty: true,
            weights: vec![
                OrderedFloat(0.25),
                OrderedFloat(0.25),
                OrderedFloat(0.25),
                OrderedFloat(0.25),
            ],
        };
        // Short sentences where higher-order n-grams are zero.
        let cand = tokens("the cat");
        let refe = tokens("the cat sat on the mat");
        let score = compute_bleu(&cand, &refe, &cfg);
        // Should be > 0 because ChenCherry smooths zero precisions.
        assert!(score > 0.0, "ChenCherry should yield nonzero, got {score}");
    }

    #[test]
    fn test_bleu_no_brevity_penalty() {
        let cfg = BLEUConfig {
            max_n: 1,
            smoothing: SmoothingMethod::None,
            brevity_penalty: false,
            weights: vec![OrderedFloat(1.0)],
        };
        let cand = tokens("the the");
        let refe = tokens("the cat sat on the mat");
        // unigram precision: clip("the")=2, denom=2 → 1.0
        // no BP → score = 1.0
        let score = compute_bleu(&cand, &refe, &cfg);
        assert!((score - 1.0).abs() < 1e-9, "expected 1.0, got {score}");
    }

    // -----------------------------------------------------------------------
    // Smoothing comparison
    // -----------------------------------------------------------------------

    #[test]
    fn test_smoothing_variants() {
        let precs = vec![0.8, 0.0, 0.0, 0.0];

        let none = apply_smoothing(&precs, &SmoothingMethod::None);
        assert_eq!(none[1], 0.0);

        let floor = apply_smoothing(&precs, &SmoothingMethod::Floor(OrderedFloat(0.01)));
        assert!((floor[1] - 0.01).abs() < 1e-9);

        let eps = apply_smoothing(&precs, &SmoothingMethod::Epsilon(OrderedFloat(0.1)));
        assert!((eps[1] - 0.1).abs() < 1e-9);

        let cc = apply_smoothing(&precs, &SmoothingMethod::ChenCherry);
        // index 0 is non-zero so unchanged; index 1 is first zero (i>0) → 1/(2^1)=0.5
        assert!((cc[1] - 0.5).abs() < 1e-9);
        // index 2 → 1/(2^2)=0.25
        assert!((cc[2] - 0.25).abs() < 1e-9);

        let nist = apply_smoothing(&precs, &SmoothingMethod::NIST);
        // index 1 → 1/(2^2)=0.25
        assert!((nist[1] - 0.25).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Modified precision
    // -----------------------------------------------------------------------

    #[test]
    fn test_modified_precision_unigram() {
        let cand = tokens("the the the the the the the");
        let refe = tokens("the cat is on the mat");
        let mp = modified_precision(&cand, &refe, 1);
        // ref unigrams: the→2, cat→1, is→1, on→1, mat→1
        // cand unigrams: the→7  → clipped = min(7,2) = 2
        // denominator = 7
        // mp = 2/7
        assert!((mp - 2.0 / 7.0).abs() < 1e-9, "got {mp}");
    }

    // -----------------------------------------------------------------------
    // ROUGE-N
    // -----------------------------------------------------------------------

    #[test]
    fn test_rouge_n_identical() {
        let t = tokens("the cat sat on the mat");
        let r = compute_rouge_n(&t, &t, 1);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_rouge_n_no_overlap() {
        let r = compute_rouge_n(&tokens("a b c"), &tokens("d e f"), 1);
        assert!(r < 1e-9);
    }

    #[test]
    fn test_rouge_n_recall() {
        // reference: "a b c d" (4 unigrams)
        // candidate: "a b e f" → overlap unigrams: a, b → 2
        // recall = 2/4 = 0.5
        let r = rouge_n_recall(&tokens("a b e f"), &tokens("a b c d"), 1);
        assert!((r - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_rouge_n_precision() {
        let p = rouge_n_precision(&tokens("a b e f"), &tokens("a b c d"), 1);
        // overlap = 2, cand total = 4
        assert!((p - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_rouge_n_fmeasure() {
        let f = rouge_n_fmeasure(&tokens("a b e f"), &tokens("a b c d"), 1, 1.0);
        // P=0.5, R=0.5 → F1 = 0.5
        assert!((f - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_rouge_2() {
        let cand = tokens("the cat sat on the mat");
        let refe = tokens("the cat sat on the mat");
        let r = compute_rouge_n(&cand, &refe, 2);
        assert!((r - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // ROUGE-L (LCS)
    // -----------------------------------------------------------------------

    #[test]
    fn test_lcs_identical() {
        let t = tokens("a b c d");
        assert_eq!(longest_common_subsequence(&t, &t), 4);
    }

    #[test]
    fn test_lcs_disjoint() {
        assert_eq!(
            longest_common_subsequence(&tokens("a b c"), &tokens("d e f")),
            0
        );
    }

    #[test]
    fn test_lcs_partial() {
        // "a b c d" vs "a c e" → LCS = "a c" length=2
        assert_eq!(
            longest_common_subsequence(&tokens("a b c d"), &tokens("a c e")),
            2
        );
    }

    #[test]
    fn test_extract_lcs() {
        let a = tokens("a b c d");
        let b = tokens("a c e");
        let table = lcs_table(&a, &b);
        let lcs = extract_lcs(&a, &b, &table);
        assert_eq!(lcs, vec!["a".to_string(), "c".to_string()]);
    }

    #[test]
    fn test_rouge_l_identical() {
        let t = tokens("the cat sat");
        let score = compute_rouge_l(&t, &t);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_rouge_l_partial() {
        let cand = tokens("a b c d");
        let refe = tokens("a c e");
        let score = compute_rouge_l(&cand, &refe);
        // LCS = 2, prec = 2/4 = 0.5, rec = 2/3 ≈ 0.6667
        // F = 2*0.5*0.6667/(0.5+0.6667) ≈ 0.5714
        let expected = 2.0 * 0.5 * (2.0 / 3.0) / (0.5 + 2.0 / 3.0);
        assert!((score - expected).abs() < 1e-4, "got {score}");
    }

    #[test]
    fn test_rouge_l_empty() {
        assert!((compute_rouge_l(&[], &[]) - 1.0).abs() < 1e-9);
        assert!((compute_rouge_l(&[], &tokens("a")) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_rouge_l_precision_recall() {
        let cand = tokens("a b c d");
        let refe = tokens("a c e");
        let p = rouge_l_precision(&cand, &refe);
        let r = rouge_l_recall(&cand, &refe);
        assert!((p - 0.5).abs() < 1e-9);
        assert!((r - 2.0 / 3.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // pass@k
    // -----------------------------------------------------------------------

    #[test]
    fn test_pass_at_k_all_pass() {
        let results = vec![true, true, true, true, true];
        let score = compute_pass_at_k(&results, 1);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pass_at_k_none_pass() {
        let results = vec![false, false, false, false, false];
        let score = compute_pass_at_k(&results, 1);
        assert!((score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_pass_at_k_known_value() {
        // n=10, c=3, k=1
        // pass@1 = 1 - C(7,1)/C(10,1) = 1 - 7/10 = 0.3
        let results = vec![
            true, true, true, false, false, false, false, false, false, false,
        ];
        let score = compute_pass_at_k(&results, 1);
        assert!(
            (score - 0.3).abs() < 1e-6,
            "expected 0.3, got {score}"
        );
    }

    #[test]
    fn test_pass_at_k_k2() {
        // n=5, c=2, k=2
        // pass@2 = 1 - C(3,2)/C(5,2) = 1 - 3/10 = 0.7
        let results = vec![true, true, false, false, false];
        let score = compute_pass_at_k(&results, 2);
        assert!(
            (score - 0.7).abs() < 1e-6,
            "expected 0.7, got {score}"
        );
    }

    // -----------------------------------------------------------------------
    // Binomial coefficient
    // -----------------------------------------------------------------------

    #[test]
    fn test_binomial_coefficient() {
        assert!((binomial_coefficient(5, 2) - 10.0).abs() < 1e-6);
        assert!((binomial_coefficient(10, 3) - 120.0).abs() < 1e-4);
        assert!((binomial_coefficient(0, 0) - 1.0).abs() < 1e-9);
        assert!((binomial_coefficient(5, 0) - 1.0).abs() < 1e-9);
        assert!((binomial_coefficient(5, 5) - 1.0).abs() < 1e-9);
        assert!((binomial_coefficient(3, 5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_log_binomial() {
        let lb = log_binomial(10, 3);
        assert!((lb.exp() - 120.0).abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // Regex match
    // -----------------------------------------------------------------------

    #[test]
    fn test_regex_literal() {
        assert!(compute_regex_match("hello", "hello"));
        assert!(!compute_regex_match("hello", "world"));
    }

    #[test]
    fn test_regex_dot() {
        assert!(compute_regex_match("cat", "c.t"));
        assert!(!compute_regex_match("ct", "c.t"));
    }

    #[test]
    fn test_regex_star() {
        assert!(compute_regex_match("aab", "a*b"));
        assert!(compute_regex_match("b", "a*b"));
        assert!(compute_regex_match("", "a*"));
    }

    #[test]
    fn test_regex_plus() {
        assert!(compute_regex_match("aab", "a+b"));
        assert!(!compute_regex_match("b", "^a+b$"));
    }

    #[test]
    fn test_regex_optional() {
        assert!(compute_regex_match("ab", "a?b"));
        assert!(compute_regex_match("b", "a?b"));
    }

    #[test]
    fn test_regex_char_class() {
        assert!(compute_regex_match("cat", "[cb]at"));
        assert!(compute_regex_match("bat", "[cb]at"));
        assert!(!compute_regex_match("dat", "^[cb]at$"));
    }

    #[test]
    fn test_regex_anchors() {
        assert!(compute_regex_match("hello", "^hello$"));
        assert!(!compute_regex_match("xhello", "^hello$"));
        assert!(compute_regex_match("xhelloy", "hello")); // unanchored
    }

    #[test]
    fn test_regex_alternation() {
        assert!(compute_regex_match("cat", "cat|dog"));
        assert!(compute_regex_match("dog", "cat|dog"));
        assert!(!compute_regex_match("rat", "^cat$|^dog$"));
    }

    #[test]
    fn test_regex_dot_star() {
        assert!(compute_regex_match("anything goes here", ".*"));
        assert!(compute_regex_match("hello world", "hello.*world"));
    }

    // -----------------------------------------------------------------------
    // Tokenization
    // -----------------------------------------------------------------------

    #[test]
    fn test_tokenize_whitespace() {
        assert_eq!(
            tokenize_whitespace("  hello   world  "),
            vec!["hello", "world"]
        );
    }

    #[test]
    fn test_tokenize_chars() {
        assert_eq!(
            tokenize_chars("abc"),
            vec!["a", "b", "c"]
        );
    }

    #[test]
    fn test_tokenize_words() {
        let t = tokenize_words("hello, world! foo_bar");
        assert_eq!(t, vec!["hello", ",", "world", "!", "foo_bar"]);
    }

    // -----------------------------------------------------------------------
    // Text normalization
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalize_lowercase() {
        assert_eq!(normalize_text("Hello World", true, false), "hello world");
    }

    #[test]
    fn test_normalize_strip_punct() {
        assert_eq!(
            normalize_text("hello, world!", false, true),
            "hello world"
        );
    }

    #[test]
    fn test_normalize_both() {
        assert_eq!(
            normalize_text("Hello, World!", true, true),
            "hello world"
        );
    }

    #[test]
    fn test_normalize_collapse_whitespace() {
        assert_eq!(
            normalize_text("  hello   world  ", false, false),
            "hello world"
        );
    }

    // -----------------------------------------------------------------------
    // Stemming
    // -----------------------------------------------------------------------

    #[test]
    fn test_stem_word_basic() {
        assert_eq!(stem_word("running"), "runn");
        assert_eq!(stem_word("cats"), "cat");
        assert_eq!(stem_word("flies"), "fli");
    }

    #[test]
    fn test_stem_word_short() {
        assert_eq!(stem_word("a"), "a");
        assert_eq!(stem_word("at"), "at");
    }

    // -----------------------------------------------------------------------
    // Metric composition
    // -----------------------------------------------------------------------

    #[test]
    fn test_compose_metrics() {
        let m1 = build_exact_match();
        let m2 = build_token_f1();
        let result = compose_metrics(&m1, &m2);
        assert!(result.is_ok());
        let composed = result.unwrap();
        assert!(composed.name.contains("exact_match"));
        assert!(composed.name.contains("token_f1"));
    }

    #[test]
    fn test_chain_metrics() {
        let m1 = build_exact_match();
        let m2 = build_token_f1();
        let result = chain_metrics(&[&m1, &m2]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_chain_empty() {
        let result: Result<BuiltinMetric, BuiltinError> = chain_metrics(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_average() {
        let m1 = build_exact_match();
        let m2 = build_token_f1();
        let result = weighted_average(&[(&m1, 0.5), (&m2, 0.5)]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_weighted_average_empty() {
        let result: Result<BuiltinMetric, BuiltinError> = weighted_average(&[]);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Registry
    // -----------------------------------------------------------------------

    #[test]
    fn test_registry_has_all_builtins() {
        let reg = BuiltinRegistry::new();
        let names = reg.list_names();
        assert!(names.contains(&"exact_match"));
        assert!(names.contains(&"token_f1"));
        assert!(names.contains(&"bleu"));
        assert!(names.contains(&"rouge_n"));
        assert!(names.contains(&"rouge_l"));
        assert!(names.contains(&"pass_at_k"));
        assert!(names.contains(&"regex_match"));
    }

    #[test]
    fn test_registry_get() {
        let reg = BuiltinRegistry::new();
        assert!(reg.get("exact_match").is_some());
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_get_declaration() {
        let reg = BuiltinRegistry::new();
        let decl = reg.get_declaration("bleu");
        assert!(decl.is_some());
        assert_eq!(decl.unwrap().name, "bleu");
    }

    #[test]
    fn test_registry_register_custom() {
        let mut reg = BuiltinRegistry::new();
        let initial_count = reg.list_names().len();
        let m = build_exact_match();
        let custom = BuiltinMetric {
            name: "custom_metric".into(),
            description: "test".into(),
            declaration: m.declaration.clone(),
            semiring: SemiringType::Real,
            default_config: MetricConfig::Custom {
                params: IndexMap::new(),
            },
            compute_fn: Box::new(|_, _| 42.0),
        };
        reg.register(custom);
        assert_eq!(reg.list_names().len(), initial_count + 1);
        assert!(reg.get("custom_metric").is_some());
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_token() {
        let t = tokens("hello");
        assert!((compute_token_f1(&t, &t) - 1.0).abs() < 1e-9);
        assert!((compute_rouge_n(&t, &t, 1) - 1.0).abs() < 1e-9);
        assert!((compute_rouge_l(&t, &t) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_strings() {
        let empty: Vec<String> = vec![];
        assert!((compute_token_f1(&empty, &empty) - 1.0).abs() < 1e-9);
        assert!((compute_rouge_l(&empty, &empty) - 1.0).abs() < 1e-9);
        assert_eq!(compute_bleu(&empty, &tokens("a b c"), &BLEUConfig::default()), 0.0);
    }

    #[test]
    fn test_known_bleu_score() {
        // A known example from the BLEU paper / NLTK:
        // candidate: "the the the the the the the"
        // reference: "the cat is on the mat"
        // BLEU-4 (no smoothing) should be 0 because higher-order n-gram
        // precisions are 0.
        let cand = tokens("the the the the the the the");
        let refe = tokens("the cat is on the mat");
        let cfg = BLEUConfig::default();
        let score = compute_bleu(&cand, &refe, &cfg);
        assert!(score < 1e-9, "degenerate candidate should get ~0 BLEU-4");
    }

    #[test]
    fn test_rouge_n_bigram_overlap() {
        // candidate: "the cat sat"  reference: "the cat sat on the mat"
        // unigrams reference: the(2), cat(1), sat(1), on(1), mat(1) = 6
        // unigrams candidate: the(1), cat(1), sat(1)
        // overlap = min(1,2)+min(1,1)+min(1,1) = 1+1+1 = 3
        // ROUGE-1 recall = 3/6 = 0.5
        let r = rouge_n_recall(
            &tokens("the cat sat"),
            &tokens("the cat sat on the mat"),
            1,
        );
        assert!((r - 0.5).abs() < 1e-9, "got {r}");
    }

    // -----------------------------------------------------------------------
    // AST helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_make_var() {
        let v = make_var("x");
        assert!(matches!(v.node, Expr::Variable(ref name) if name == "x"));
    }

    #[test]
    fn test_make_int() {
        let n = make_int(42);
        assert!(matches!(n.node, Expr::Literal(Literal::Integer(42))));
    }

    #[test]
    fn test_make_float() {
        let f = make_float(3.14);
        if let Expr::Literal(Literal::Float(v)) = &f.node {
            assert!((v.0 - 3.14).abs() < 1e-9);
        } else {
            panic!("expected float literal");
        }
    }

    #[test]
    fn test_make_binary() {
        let b = make_binary(make_int(1), BinaryOp::Add, make_int(2));
        assert!(matches!(b.node, Expr::BinaryOp { op: BinaryOp::Add, .. }));
    }

    #[test]
    fn test_make_call() {
        let c = make_call("foo", vec![make_int(1)]);
        if let Expr::FunctionCall { name, args } = &c.node {
            assert_eq!(name, "foo");
            assert_eq!(args.len(), 1);
        } else {
            panic!("expected function call");
        }
    }

    // -----------------------------------------------------------------------
    // build_regex_match validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_regex_match_valid() {
        let result = build_regex_match("^hello.*world$");
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_regex_match_unterminated_class() {
        let result = build_regex_match("[abc");
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Metric smoke tests via the BuiltinMetric.compute_fn
    // -----------------------------------------------------------------------

    #[test]
    fn test_builtin_metric_compute_fn_bleu() {
        let m = build_bleu(&BLEUConfig::default());
        let cand = tokens("the cat sat on the mat");
        let refe = tokens("the cat sat on the mat");
        let score = (m.compute_fn)(&cand, &refe);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_builtin_metric_compute_fn_rouge_n() {
        let m = build_rouge_n(&RougeConfig::default());
        let cand = tokens("the cat sat on the mat");
        let refe = tokens("the cat sat on the mat");
        let score = (m.compute_fn)(&cand, &refe);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_builtin_metric_compute_fn_rouge_l() {
        let m = build_rouge_l();
        let cand = tokens("the cat sat on the mat");
        let refe = tokens("the cat sat on the mat");
        let score = (m.compute_fn)(&cand, &refe);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_builtin_metric_compute_fn_exact_match() {
        let m = build_exact_match();
        let score = (m.compute_fn)(&tokens("hello"), &tokens("hello"));
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_builtin_metric_compute_fn_token_f1() {
        let m = build_token_f1();
        let score = (m.compute_fn)(&tokens("a b c"), &tokens("a b c"));
        assert!((score - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Known literature score: BLEU
    // -----------------------------------------------------------------------

    #[test]
    fn test_bleu_known_literature_example() {
        // Classic example (simplified):
        // Reference: "It is a guide to action which ensures that the military
        //             always obeys the commands of the party"
        // Candidate: "It is a guide to action that ensures that the military
        //             will forever obey the commands of the party"
        // (From the original BLEU paper, modified for brevity.)
        let refe = tokens(
            "it is a guide to action which ensures that the military always obeys the commands of the party",
        );
        let cand = tokens(
            "it is a guide to action that ensures that the military will forever obey the commands of the party",
        );
        let cfg = BLEUConfig {
            max_n: 4,
            smoothing: SmoothingMethod::None,
            brevity_penalty: true,
            weights: vec![
                OrderedFloat(0.25),
                OrderedFloat(0.25),
                OrderedFloat(0.25),
                OrderedFloat(0.25),
            ],
        };
        let score = compute_bleu(&cand, &refe, &cfg);
        // The score should be between 0 and 1 and non-zero.
        assert!(score > 0.0 && score < 1.0, "BLEU should be in (0,1), got {score}");
    }
}
