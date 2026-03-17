use std::fmt;

use serde::{Deserialize, Serialize};

/// Formalizability grade for an obligation.
///
/// Captures how precisely a regulatory obligation can be translated into
/// a formal constraint (SMT formula, boolean expression, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FormalGrade {
    /// Fully formalizable: a precise logical encoding exists.
    Full,
    /// Partially formalizable with a confidence score in (0, 1).
    Partial(f64),
    /// Opaque: no meaningful formal encoding is possible.
    Opaque,
}

impl FormalGrade {
    /// Numeric score in [0, 1].
    pub fn score(&self) -> f64 {
        match self {
            Self::Full => 1.0,
            Self::Partial(s) => s.clamp(0.0, 1.0),
            Self::Opaque => 0.0,
        }
    }

    /// Create a Partial grade, clamped to [0,1].
    pub fn partial(score: f64) -> Self {
        let s = score.clamp(0.0, 1.0);
        if (s - 1.0).abs() < f64::EPSILON {
            Self::Full
        } else if s < f64::EPSILON {
            Self::Opaque
        } else {
            Self::Partial(s)
        }
    }

    /// Whether this grade is above a threshold.
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.score() >= threshold
    }

    pub fn is_full(&self) -> bool {
        matches!(self, Self::Full)
    }

    pub fn is_opaque(&self) -> bool {
        matches!(self, Self::Opaque)
    }

    pub fn is_partial(&self) -> bool {
        matches!(self, Self::Partial(_))
    }

    /// Whether this grade is at least partially formalizable (not opaque).
    pub fn is_formalizable(&self) -> bool {
        !self.is_opaque()
    }

    /// Alias for `score()` — numeric confidence in [0, 1].
    pub fn confidence(&self) -> f64 {
        self.score()
    }

    /// Compose with another grade under conjunction (minimum).
    pub fn compose_conjunction(&self, other: &FormalGrade) -> FormalGrade {
        FormalGrade::partial(self.score().min(other.score()))
    }

    /// Compose with another grade under disjunction (maximum).
    pub fn compose_disjunction(&self, other: &FormalGrade) -> FormalGrade {
        FormalGrade::partial(self.score().max(other.score()))
    }
}

impl fmt::Display for FormalGrade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => write!(f, "Full"),
            Self::Partial(s) => write!(f, "Partial({:.2})", s),
            Self::Opaque => write!(f, "Opaque"),
        }
    }
}

impl Default for FormalGrade {
    fn default() -> Self {
        Self::Full
    }
}

impl Eq for FormalGrade {}

impl PartialOrd for FormalGrade {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FormalGrade {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score()
            .partial_cmp(&other.score())
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// Confidence algebra
// ---------------------------------------------------------------------------

/// Compose two grades under conjunction (tensor product).
/// When both obligations must hold, the result grade is the minimum.
pub fn compose_conjunction(a: FormalGrade, b: FormalGrade) -> FormalGrade {
    FormalGrade::partial(a.score().min(b.score()))
}

/// Compose two grades under disjunction (par).
/// When either obligation suffices, the result grade is the maximum.
pub fn compose_disjunction(a: FormalGrade, b: FormalGrade) -> FormalGrade {
    FormalGrade::partial(a.score().max(b.score()))
}

/// Compose two grades under sequential dependency.
/// If obligation B depends on the output of A, the joint
/// formalizability is the product of their scores.
pub fn compose_sequential(a: FormalGrade, b: FormalGrade) -> FormalGrade {
    FormalGrade::partial(a.score() * b.score())
}

/// Compose under jurisdictional override: if a higher-priority
/// jurisdiction's obligation overrides a lower one, the resulting
/// grade is that of the overriding obligation.
pub fn compose_override(overriding: FormalGrade, _overridden: FormalGrade) -> FormalGrade {
    overriding
}

/// Weighted combination of multiple grades.
pub fn compose_weighted(grades: &[(FormalGrade, f64)]) -> FormalGrade {
    if grades.is_empty() {
        return FormalGrade::Opaque;
    }
    let total_weight: f64 = grades.iter().map(|(_, w)| w).sum();
    if total_weight <= 0.0 {
        return FormalGrade::Opaque;
    }
    let weighted_sum: f64 = grades.iter().map(|(g, w)| g.score() * w).sum();
    FormalGrade::partial(weighted_sum / total_weight)
}

/// Propagate a grade through a negation (NOT).
/// Negation preserves formalizability.
pub fn propagate_negation(grade: FormalGrade) -> FormalGrade {
    grade
}

/// Propagate through implication: A => B has the grade of compose_disjunction(NOT A, B).
pub fn propagate_implication(antecedent: FormalGrade, consequent: FormalGrade) -> FormalGrade {
    compose_disjunction(propagate_negation(antecedent), consequent)
}

// ---------------------------------------------------------------------------
// Formalizability statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics over a set of formalizability grades.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormalizabilityStats {
    pub total: usize,
    pub full_count: usize,
    pub partial_count: usize,
    pub opaque_count: usize,
    pub mean_score: f64,
    pub min_score: f64,
    pub max_score: f64,
    pub median_score: f64,
}

impl FormalizabilityStats {
    pub fn compute(grades: &[FormalGrade]) -> Self {
        if grades.is_empty() {
            return Self {
                total: 0,
                full_count: 0,
                partial_count: 0,
                opaque_count: 0,
                mean_score: 0.0,
                min_score: 0.0,
                max_score: 0.0,
                median_score: 0.0,
            };
        }

        let total = grades.len();
        let full_count = grades.iter().filter(|g| g.is_full()).count();
        let partial_count = grades.iter().filter(|g| g.is_partial()).count();
        let opaque_count = grades.iter().filter(|g| g.is_opaque()).count();

        let scores: Vec<f64> = grades.iter().map(|g| g.score()).collect();
        let mean_score = scores.iter().sum::<f64>() / total as f64;
        let min_score = scores
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_score = scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_score = if total % 2 == 0 {
            (sorted[total / 2 - 1] + sorted[total / 2]) / 2.0
        } else {
            sorted[total / 2]
        };

        Self {
            total,
            full_count,
            partial_count,
            opaque_count,
            mean_score,
            min_score,
            max_score,
            median_score,
        }
    }

    /// Fraction of obligations that are at least partially formalizable.
    pub fn formalizable_fraction(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.full_count + self.partial_count) as f64 / self.total as f64
    }

    /// Fraction meeting a given score threshold.
    pub fn fraction_above_threshold(&self, _threshold: f64, grades: &[FormalGrade]) -> f64 {
        if grades.is_empty() {
            return 0.0;
        }
        let count = grades
            .iter()
            .filter(|g| g.meets_threshold(_threshold))
            .count();
        count as f64 / grades.len() as f64
    }
}

impl fmt::Display for FormalizabilityStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Formalizability: {}/{} full, {}/{} partial, {}/{} opaque (mean={:.2}, median={:.2})",
            self.full_count,
            self.total,
            self.partial_count,
            self.total,
            self.opaque_count,
            self.total,
            self.mean_score,
            self.median_score
        )
    }
}

// ---------------------------------------------------------------------------
// Threshold operations
// ---------------------------------------------------------------------------

/// Filter grades that meet a threshold, returning indices.
pub fn filter_by_threshold(grades: &[FormalGrade], threshold: f64) -> Vec<usize> {
    grades
        .iter()
        .enumerate()
        .filter(|(_, g)| g.meets_threshold(threshold))
        .map(|(i, _)| i)
        .collect()
}

/// Compute the threshold at which a target fraction of obligations pass.
pub fn threshold_for_fraction(grades: &[FormalGrade], target_fraction: f64) -> f64 {
    if grades.is_empty() {
        return 0.0;
    }
    let mut scores: Vec<f64> = grades.iter().map(|g| g.score()).collect();
    scores.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
    let target_count = (target_fraction * grades.len() as f64).ceil() as usize;
    let target_count = target_count.min(scores.len());
    if target_count == 0 {
        return 1.0;
    }
    scores[target_count - 1]
}

/// Degrade a grade by a factor (e.g., due to translation uncertainty).
pub fn degrade(grade: FormalGrade, factor: f64) -> FormalGrade {
    FormalGrade::partial(grade.score() * factor.clamp(0.0, 1.0))
}

/// Boost a grade (e.g., after human review confirms formalizability).
pub fn boost(grade: FormalGrade, factor: f64) -> FormalGrade {
    FormalGrade::partial((grade.score() * factor.clamp(1.0, f64::MAX)).min(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grade_score() {
        assert_eq!(FormalGrade::Full.score(), 1.0);
        assert_eq!(FormalGrade::Opaque.score(), 0.0);
        assert!((FormalGrade::Partial(0.7).score() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_partial_normalization() {
        assert!(FormalGrade::partial(1.0).is_full());
        assert!(FormalGrade::partial(0.0).is_opaque());
        assert!(FormalGrade::partial(0.5).is_partial());
    }

    #[test]
    fn test_compose_conjunction() {
        let result = compose_conjunction(FormalGrade::Partial(0.8), FormalGrade::Partial(0.5));
        assert!((result.score() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_compose_disjunction() {
        let result = compose_disjunction(FormalGrade::Partial(0.3), FormalGrade::Partial(0.9));
        assert!((result.score() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_compose_sequential() {
        let result = compose_sequential(FormalGrade::Partial(0.8), FormalGrade::Partial(0.5));
        assert!((result.score() - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_compose_override() {
        let result = compose_override(FormalGrade::Full, FormalGrade::Opaque);
        assert!(result.is_full());
    }

    #[test]
    fn test_compose_weighted() {
        let result = compose_weighted(&[
            (FormalGrade::Full, 2.0),
            (FormalGrade::Opaque, 1.0),
        ]);
        // (1.0*2 + 0.0*1) / 3 ≈ 0.667
        assert!((result.score() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats() {
        let grades = vec![
            FormalGrade::Full,
            FormalGrade::Partial(0.7),
            FormalGrade::Partial(0.3),
            FormalGrade::Opaque,
        ];
        let stats = FormalizabilityStats::compute(&grades);
        assert_eq!(stats.total, 4);
        assert_eq!(stats.full_count, 1);
        assert_eq!(stats.partial_count, 2);
        assert_eq!(stats.opaque_count, 1);
        assert!((stats.mean_score - 0.5).abs() < 1e-9);
        assert!((stats.formalizable_fraction() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_filter_by_threshold() {
        let grades = vec![
            FormalGrade::Full,
            FormalGrade::Partial(0.4),
            FormalGrade::Opaque,
        ];
        let indices = filter_by_threshold(&grades, 0.5);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_threshold_for_fraction() {
        let grades = vec![
            FormalGrade::Full,
            FormalGrade::Partial(0.8),
            FormalGrade::Partial(0.5),
            FormalGrade::Opaque,
        ];
        let t = threshold_for_fraction(&grades, 0.5);
        assert!((t - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_degrade_and_boost() {
        let grade = FormalGrade::Partial(0.8);
        let degraded = degrade(grade, 0.5);
        assert!((degraded.score() - 0.4).abs() < 1e-9);

        let boosted = boost(FormalGrade::Partial(0.5), 1.5);
        assert!((boosted.score() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_ordering() {
        assert!(FormalGrade::Opaque < FormalGrade::Partial(0.5));
        assert!(FormalGrade::Partial(0.5) < FormalGrade::Full);
    }

    #[test]
    fn test_serialization() {
        let grade = FormalGrade::Partial(0.73);
        let json = serde_json::to_string(&grade).unwrap();
        let deser: FormalGrade = serde_json::from_str(&json).unwrap();
        assert!((deser.score() - 0.73).abs() < 1e-9);
    }
}
