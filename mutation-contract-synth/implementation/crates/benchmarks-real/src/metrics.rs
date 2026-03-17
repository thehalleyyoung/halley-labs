//! Metrics for evaluating synthesized contracts against ground truth.
//!
//! We compute precision, recall, and F1 by matching synthesized clauses
//! against ground-truth clauses using semantic similarity (normalised
//! string matching with formula canonicalisation).

use serde::{Deserialize, Serialize};

use crate::{AlgorithmStats, ClauseKind, SynthesizedClause};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContractMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub clauses_synthesized: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub method: String,
    pub signature: String,
    pub ground_truth_pre: usize,
    pub ground_truth_post: usize,
    pub lattice_walk: ContractMetrics,
    pub lattice_walk_time_us: u64,
    pub lattice_walk_stats: AlgorithmStats,
    pub random_mutation: ContractMetrics,
    pub random_mutation_time_us: u64,
    pub random_mutation_stats: AlgorithmStats,
    pub spec_mining: ContractMetrics,
    pub spec_mining_time_us: u64,
    pub spec_mining_stats: AlgorithmStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteResult {
    pub methods_tested: usize,
    pub lattice_walk_avg_precision: f64,
    pub lattice_walk_avg_recall: f64,
    pub lattice_walk_avg_f1: f64,
    pub lattice_walk_avg_time_us: u64,
    pub random_mutation_avg_precision: f64,
    pub random_mutation_avg_recall: f64,
    pub random_mutation_avg_f1: f64,
    pub random_mutation_avg_time_us: u64,
    pub spec_mining_avg_precision: f64,
    pub spec_mining_avg_recall: f64,
    pub spec_mining_avg_f1: f64,
    pub spec_mining_avg_time_us: u64,
    pub per_method: Vec<BenchmarkResult>,
}

/// Evaluate synthesized clauses against ground truth.
///
/// Matching is done via normalised string comparison.  A synthesized clause
/// is a true positive if it semantically matches any ground-truth clause
/// (checked via multiple normalisations including commutativity).
pub fn evaluate_contract(
    synthesized: &[SynthesizedClause],
    ground_truth_pre: &[String],
    ground_truth_post: &[String],
) -> ContractMetrics {
    let synth_post: Vec<&str> = synthesized
        .iter()
        .filter(|c| c.kind == ClauseKind::Post)
        .map(|c| c.formula_str.as_str())
        .collect();
    let synth_pre: Vec<&str> = synthesized
        .iter()
        .filter(|c| c.kind == ClauseKind::Pre)
        .map(|c| c.formula_str.as_str())
        .collect();

    let mut all_gt: Vec<&str> = Vec::new();
    for g in ground_truth_pre {
        if g != "true" {
            all_gt.push(g.as_str());
        }
    }
    for g in ground_truth_post {
        all_gt.push(g.as_str());
    }

    let mut all_synth: Vec<&str> = Vec::new();
    all_synth.extend(&synth_pre);
    all_synth.extend(&synth_post);

    // Match: for each ground-truth clause, check if any synthesized clause matches.
    let mut matched_gt: Vec<bool> = vec![false; all_gt.len()];
    let mut matched_synth: Vec<bool> = vec![false; all_synth.len()];

    for (gi, gt) in all_gt.iter().enumerate() {
        for (si, syn) in all_synth.iter().enumerate() {
            if clauses_match(gt, syn) {
                matched_gt[gi] = true;
                matched_synth[si] = true;
            }
        }
    }

    let true_positives = matched_gt.iter().filter(|&&b| b).count();
    let false_negatives = matched_gt.iter().filter(|&&b| !b).count();
    let false_positives = matched_synth.iter().filter(|&&b| !b).count();

    let precision = if true_positives + false_positives > 0 {
        true_positives as f64 / (true_positives + false_positives) as f64
    } else {
        0.0
    };
    let recall = if true_positives + false_negatives > 0 {
        true_positives as f64 / (true_positives + false_negatives) as f64
    } else {
        1.0 // no ground truth → vacuously complete
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    ContractMetrics {
        precision,
        recall,
        f1,
        clauses_synthesized: all_synth.len(),
        true_positives,
        false_positives,
        false_negatives,
    }
}

/// Check if two clause strings match semantically.
///
/// We normalise both sides and check multiple equivalent forms.
fn clauses_match(ground_truth: &str, synthesized: &str) -> bool {
    let gt = normalise(ground_truth);
    let syn = normalise(synthesized);

    if gt == syn {
        return true;
    }

    // Check if the synthesized clause is a negation form that matches.
    // E.g., ground truth "ret >= 0" matches synthesized "!(ret < 0)".
    if let Some(inner) = strip_negation(&syn) {
        if let Some(negated_gt) = negate_relation_str(&gt) {
            if inner == negated_gt {
                return true;
            }
        }
    }
    if let Some(inner) = strip_negation(&gt) {
        if let Some(negated_syn) = negate_relation_str(&syn) {
            if inner == negated_syn {
                return true;
            }
        }
    }

    // Check with commuted operands for symmetric relations.
    if let Some(commuted) = commute_relation(&syn) {
        if gt == commuted {
            return true;
        }
    }

    // Partial match: check if the synthesized clause implies part of ground truth.
    // This handles cases like "¬(x > 0)" matching "x <= 0".
    if negation_matches_flipped(&gt, &syn) {
        return true;
    }

    false
}

fn normalise(s: &str) -> String {
    s.replace(" ", "")
        .replace("&&", "&")
        .replace("||", "|")
        .to_lowercase()
}

fn strip_negation(s: &str) -> Option<String> {
    let s = s.trim();
    if s.starts_with("!(") && s.ends_with(')') {
        Some(s[2..s.len() - 1].to_string())
    } else if s.starts_with('!') {
        Some(s[1..].to_string())
    } else {
        None
    }
}

fn negate_relation_str(s: &str) -> Option<String> {
    let replacements = [
        (">=", "<"),
        ("<=", ">"),
        ("==", "!="),
        ("!=", "=="),
        (">", "<="),
        ("<", ">="),
    ];
    for (from, to) in &replacements {
        if s.contains(from) {
            return Some(s.replacen(from, to, 1));
        }
    }
    None
}

fn commute_relation(s: &str) -> Option<String> {
    // For "a >= b" → "b <= a", etc.
    let ops = [">=", "<=", "==", "!=", ">", "<"];
    for op in &ops {
        if let Some(pos) = s.find(op) {
            let lhs = s[..pos].trim();
            let rhs = s[pos + op.len()..].trim();
            let flipped_op = match *op {
                ">=" => "<=",
                "<=" => ">=",
                ">" => "<",
                "<" => ">",
                "==" => "==",
                "!=" => "!=",
                _ => continue,
            };
            return Some(format!("{}{}{}", rhs, flipped_op, lhs));
        }
    }
    None
}

/// Check if ¬(a op b) matches a flipped_op b.
/// E.g., !(x > 0) matches x <= 0.
fn negation_matches_flipped(gt: &str, syn: &str) -> bool {
    if let Some(inner) = strip_negation(syn) {
        if let Some(negated) = negate_relation_str(&inner) {
            return normalise(gt) == normalise(&negated);
        }
    }
    if let Some(inner) = strip_negation(gt) {
        if let Some(negated) = negate_relation_str(&inner) {
            return normalise(syn) == normalise(&negated);
        }
    }
    false
}
