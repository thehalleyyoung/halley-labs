//! Spectacles integration crate - re-exports and integration utilities.

pub use spectacles_core::scoring;
pub use spectacles_core::utils;

/// Run a full pipeline: score → verify agreement → generate proof sketch
pub fn run_scoring_pipeline(
    candidate: &str,
    reference: &str,
    metric: &str,
) -> PipelineResult {
    use spectacles_core::scoring::{ScoringPair, TripleMetric};
    
    let pair = ScoringPair {
        candidate: candidate.to_string(),
        reference: reference.to_string(),
    };
    
    let (score_str, agreement) = match metric {
        "exact_match" => {
            let scorer = scoring::ExactMatchScorer::case_insensitive();
            let result = scorer.score_and_verify(&pair);
            (format!("{}", result.reference), result.agreement)
        }
        "token_f1" => {
            let scorer = scoring::TokenF1Scorer::default_scorer();
            let result = scorer.score_and_verify(&pair);
            (format!("{:.4}", result.reference.f1), result.agreement)
        }
        "bleu" => {
            let scorer = scoring::BleuScorer::with_smoothing(scoring::bleu::SmoothingMethod::Add1);
            let result = scorer.score_and_verify(&pair);
            (format!("{:.4}", result.reference.score), result.agreement)
        }
        "rouge1" => {
            let scorer = scoring::RougeNScorer::rouge1();
            let result = scorer.score_and_verify(&pair);
            (format!("{:.4}", result.reference.f1), result.agreement)
        }
        "rouge_l" => {
            let scorer = scoring::RougeLScorer::default_scorer();
            let result = scorer.score_and_verify(&pair);
            (format!("{:.4}", result.reference.f1), result.agreement)
        }
        _ => ("unsupported".to_string(), false),
    };
    
    // Generate commitment
    let hasher = utils::hash::SpectaclesHasher::with_domain("scoring");
    let score_hash = hasher.hash(score_str.as_bytes());
    
    PipelineResult {
        metric: metric.to_string(),
        score: score_str,
        triple_agreement: agreement,
        commitment: hex::encode(score_hash),
    }
}

/// Result of running the full scoring pipeline
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PipelineResult {
    pub metric: String,
    pub score: String,
    pub triple_agreement: bool,
    pub commitment: String,
}
