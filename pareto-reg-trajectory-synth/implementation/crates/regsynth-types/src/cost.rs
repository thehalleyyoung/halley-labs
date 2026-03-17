use serde::{Deserialize, Serialize};
use ordered_float::OrderedFloat;
use std::fmt;
use std::ops::{Add, Mul};

/// 4-dimensional cost vector for regulatory compliance optimization:
/// (implementation_cost, time_to_compliance, residual_risk, operational_burden)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostVector {
    pub implementation_cost: f64,
    pub time_to_compliance: f64,
    pub residual_risk: f64,
    pub operational_burden: f64,
}

impl CostVector {
    pub fn new(impl_cost: f64, time: f64, risk: f64, burden: f64) -> Self {
        CostVector {
            implementation_cost: impl_cost,
            time_to_compliance: time,
            residual_risk: risk,
            operational_burden: burden,
        }
    }

    pub fn zero() -> Self {
        CostVector::new(0.0, 0.0, 0.0, 0.0)
    }

    pub fn max_value() -> Self {
        CostVector::new(f64::MAX, f64::MAX, f64::MAX, f64::MAX)
    }

    pub fn dimensions(&self) -> [f64; 4] {
        [self.implementation_cost, self.time_to_compliance,
         self.residual_risk, self.operational_burden]
    }

    pub fn dimension(&self, idx: CostDimension) -> f64 {
        match idx {
            CostDimension::ImplementationCost => self.implementation_cost,
            CostDimension::TimeToCompliance => self.time_to_compliance,
            CostDimension::ResidualRisk => self.residual_risk,
            CostDimension::OperationalBurden => self.operational_burden,
        }
    }

    pub fn set_dimension(&mut self, idx: CostDimension, value: f64) {
        match idx {
            CostDimension::ImplementationCost => self.implementation_cost = value,
            CostDimension::TimeToCompliance => self.time_to_compliance = value,
            CostDimension::ResidualRisk => self.residual_risk = value,
            CostDimension::OperationalBurden => self.operational_burden = value,
        }
    }

    pub fn weighted_sum(&self, weights: &[f64; 4]) -> f64 {
        let d = self.dimensions();
        d.iter().zip(weights.iter()).map(|(v, w)| v * w).sum()
    }

    pub fn l2_norm(&self) -> f64 {
        let d = self.dimensions();
        d.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    pub fn l1_norm(&self) -> f64 {
        self.dimensions().iter().map(|v| v.abs()).sum()
    }

    pub fn linf_norm(&self) -> f64 {
        self.dimensions().iter().cloned().fold(0.0_f64, f64::max)
    }

    pub fn normalize(&self) -> Self {
        let norm = self.l2_norm();
        if norm < 1e-12 {
            return self.clone();
        }
        CostVector::new(
            self.implementation_cost / norm,
            self.time_to_compliance / norm,
            self.residual_risk / norm,
            self.operational_burden / norm,
        )
    }

    pub fn normalize_to_bounds(&self, lower: &CostVector, upper: &CostVector) -> Self {
        let mut result = CostVector::zero();
        for dim in CostDimension::all() {
            let lo = lower.dimension(dim);
            let hi = upper.dimension(dim);
            let range = hi - lo;
            if range.abs() < 1e-12 {
                result.set_dimension(dim, 0.0);
            } else {
                result.set_dimension(dim, (self.dimension(dim) - lo) / range);
            }
        }
        result
    }

    pub fn dominates(&self, other: &CostVector) -> bool {
        let d_self = self.dimensions();
        let d_other = other.dimensions();
        let mut at_least_one_better = false;
        for (s, o) in d_self.iter().zip(d_other.iter()) {
            if s > o { return false; }
            if s < o { at_least_one_better = true; }
        }
        at_least_one_better
    }

    pub fn epsilon_dominates(&self, other: &CostVector, epsilon: f64) -> bool {
        let d_self = self.dimensions();
        let d_other = other.dimensions();
        let mut at_least_one_better = false;
        for (s, o) in d_self.iter().zip(d_other.iter()) {
            if *s > *o + epsilon { return false; }
            if *s < *o - epsilon { at_least_one_better = true; }
        }
        at_least_one_better
    }

    pub fn pareto_compare(&self, other: &CostVector) -> ParetoOrdering {
        if self == other { return ParetoOrdering::Equal; }
        if self.dominates(other) { return ParetoOrdering::Dominates; }
        if other.dominates(self) { return ParetoOrdering::Dominated; }
        ParetoOrdering::Incomparable
    }

    pub fn component_min(&self, other: &CostVector) -> CostVector {
        CostVector::new(
            self.implementation_cost.min(other.implementation_cost),
            self.time_to_compliance.min(other.time_to_compliance),
            self.residual_risk.min(other.residual_risk),
            self.operational_burden.min(other.operational_burden),
        )
    }

    pub fn component_max(&self, other: &CostVector) -> CostVector {
        CostVector::new(
            self.implementation_cost.max(other.implementation_cost),
            self.time_to_compliance.max(other.time_to_compliance),
            self.residual_risk.max(other.residual_risk),
            self.operational_burden.max(other.operational_burden),
        )
    }

    pub fn distance(&self, other: &CostVector) -> f64 {
        let diff = CostVector::new(
            self.implementation_cost - other.implementation_cost,
            self.time_to_compliance - other.time_to_compliance,
            self.residual_risk - other.residual_risk,
            self.operational_burden - other.operational_burden,
        );
        diff.l2_norm()
    }

    pub fn scale(&self, factor: f64) -> CostVector {
        CostVector::new(
            self.implementation_cost * factor,
            self.time_to_compliance * factor,
            self.residual_risk * factor,
            self.operational_burden * factor,
        )
    }

    pub fn to_ordered(&self) -> OrderedCostVector {
        OrderedCostVector {
            implementation_cost: OrderedFloat(self.implementation_cost),
            time_to_compliance: OrderedFloat(self.time_to_compliance),
            residual_risk: OrderedFloat(self.residual_risk),
            operational_burden: OrderedFloat(self.operational_burden),
        }
    }
}

impl Add for CostVector {
    type Output = CostVector;
    fn add(self, rhs: CostVector) -> CostVector {
        CostVector::new(
            self.implementation_cost + rhs.implementation_cost,
            self.time_to_compliance + rhs.time_to_compliance,
            self.residual_risk + rhs.residual_risk,
            self.operational_burden + rhs.operational_burden,
        )
    }
}

impl Mul<f64> for CostVector {
    type Output = CostVector;
    fn mul(self, scalar: f64) -> CostVector {
        self.scale(scalar)
    }
}

impl fmt::Display for CostVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cost(impl={:.2}, time={:.2}, risk={:.2}, burden={:.2})",
            self.implementation_cost, self.time_to_compliance,
            self.residual_risk, self.operational_burden)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrderedCostVector {
    pub implementation_cost: OrderedFloat<f64>,
    pub time_to_compliance: OrderedFloat<f64>,
    pub residual_risk: OrderedFloat<f64>,
    pub operational_burden: OrderedFloat<f64>,
}

impl OrderedCostVector {
    pub fn to_cost_vector(&self) -> CostVector {
        CostVector::new(
            self.implementation_cost.0,
            self.time_to_compliance.0,
            self.residual_risk.0,
            self.operational_burden.0,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CostDimension {
    ImplementationCost,
    TimeToCompliance,
    ResidualRisk,
    OperationalBurden,
}

impl CostDimension {
    pub fn all() -> [CostDimension; 4] {
        [CostDimension::ImplementationCost, CostDimension::TimeToCompliance,
         CostDimension::ResidualRisk, CostDimension::OperationalBurden]
    }

    pub fn index(&self) -> usize {
        match self {
            CostDimension::ImplementationCost => 0,
            CostDimension::TimeToCompliance => 1,
            CostDimension::ResidualRisk => 2,
            CostDimension::OperationalBurden => 3,
        }
    }
}

impl fmt::Display for CostDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CostDimension::ImplementationCost => write!(f, "Implementation Cost"),
            CostDimension::TimeToCompliance => write!(f, "Time to Compliance"),
            CostDimension::ResidualRisk => write!(f, "Residual Risk"),
            CostDimension::OperationalBurden => write!(f, "Operational Burden"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParetoOrdering {
    Dominates,
    Dominated,
    Incomparable,
    Equal,
}

impl ParetoOrdering {
    pub fn is_dominated(&self) -> bool { matches!(self, ParetoOrdering::Dominated) }
    pub fn is_dominating(&self) -> bool { matches!(self, ParetoOrdering::Dominates) }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBounds {
    pub lower: CostVector,
    pub upper: CostVector,
}

impl CostBounds {
    pub fn new(lower: CostVector, upper: CostVector) -> Self {
        CostBounds { lower, upper }
    }

    pub fn from_points(points: &[CostVector]) -> Option<Self> {
        if points.is_empty() { return None; }
        let mut lower = points[0].clone();
        let mut upper = points[0].clone();
        for p in &points[1..] {
            lower = lower.component_min(p);
            upper = upper.component_max(p);
        }
        Some(CostBounds { lower, upper })
    }

    pub fn contains(&self, point: &CostVector) -> bool {
        let d = point.dimensions();
        let lo = self.lower.dimensions();
        let hi = self.upper.dimensions();
        d.iter().zip(lo.iter().zip(hi.iter())).all(|(v, (l, h))| v >= l && v <= h)
    }

    pub fn volume(&self) -> f64 {
        let lo = self.lower.dimensions();
        let hi = self.upper.dimensions();
        lo.iter().zip(hi.iter()).map(|(l, h)| (h - l).max(0.0)).product()
    }
}

/// Filter dominated points from a set
pub fn filter_dominated(points: &[CostVector]) -> Vec<CostVector> {
    let mut non_dominated = Vec::new();
    for (i, p) in points.iter().enumerate() {
        let is_dominated = points.iter().enumerate().any(|(j, q)| i != j && q.dominates(p));
        if !is_dominated {
            non_dominated.push(p.clone());
        }
    }
    non_dominated
}

/// Compute hypervolume of a point set relative to a reference point (4D)
pub fn hypervolume_4d(points: &[CostVector], reference: &CostVector) -> f64 {
    if points.is_empty() { return 0.0; }
    if points.len() == 1 {
        let d = points[0].dimensions();
        let r = reference.dimensions();
        return d.iter().zip(r.iter()).map(|(p, rr)| (rr - p).max(0.0)).product();
    }
    // Inclusion-exclusion approximation for 4D
    let mut total = 0.0;
    let ref_dims = reference.dimensions();
    for p in points {
        let pd = p.dimensions();
        let vol: f64 = pd.iter().zip(ref_dims.iter()).map(|(pi, ri)| (ri - pi).max(0.0)).product();
        total += vol;
    }
    // Subtract pairwise overlaps
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let pi = points[i].dimensions();
            let pj = points[j].dimensions();
            let overlap: f64 = (0..4).map(|k| {
                let lo = pi[k].max(pj[k]);
                (ref_dims[k] - lo).max(0.0)
            }).product();
            total -= overlap;
        }
    }
    total.max(0.0)
}

/// Generational distance from computed frontier to true frontier
pub fn generational_distance(computed: &[CostVector], true_frontier: &[CostVector]) -> f64 {
    if computed.is_empty() || true_frontier.is_empty() { return f64::MAX; }
    let sum: f64 = computed.iter().map(|c| {
        true_frontier.iter().map(|t| c.distance(t)).fold(f64::MAX, f64::min)
    }).sum();
    sum / computed.len() as f64
}

/// Inverted generational distance
pub fn inverted_generational_distance(computed: &[CostVector], true_frontier: &[CostVector]) -> f64 {
    generational_distance(true_frontier, computed)
}

/// Spread metric measuring uniformity of frontier distribution
pub fn spread_metric(points: &[CostVector]) -> f64 {
    if points.len() < 2 { return 0.0; }
    let mut distances: Vec<f64> = Vec::new();
    for i in 0..points.len() {
        let min_dist = points.iter().enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, p)| points[i].distance(p))
            .fold(f64::MAX, f64::min);
        distances.push(min_dist);
    }
    let mean = distances.iter().sum::<f64>() / distances.len() as f64;
    let variance = distances.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / distances.len() as f64;
    variance.sqrt() / mean.max(1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_vector_basic() {
        let c = CostVector::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(c.dimensions(), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_dominance() {
        let a = CostVector::new(1.0, 1.0, 1.0, 1.0);
        let b = CostVector::new(2.0, 2.0, 2.0, 2.0);
        assert!(a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    #[test]
    fn test_incomparable() {
        let a = CostVector::new(1.0, 3.0, 1.0, 1.0);
        let b = CostVector::new(3.0, 1.0, 1.0, 1.0);
        assert_eq!(a.pareto_compare(&b), ParetoOrdering::Incomparable);
    }

    #[test]
    fn test_filter_dominated() {
        let points = vec![
            CostVector::new(1.0, 1.0, 1.0, 1.0),
            CostVector::new(2.0, 2.0, 2.0, 2.0),
            CostVector::new(1.0, 3.0, 0.5, 1.0),
        ];
        let nd = filter_dominated(&points);
        assert_eq!(nd.len(), 2);
    }

    #[test]
    fn test_weighted_sum() {
        let c = CostVector::new(1.0, 2.0, 3.0, 4.0);
        let w = [0.25, 0.25, 0.25, 0.25];
        assert!((c.weighted_sum(&w) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_hypervolume() {
        let pts = vec![CostVector::new(1.0, 1.0, 1.0, 1.0)];
        let rf = CostVector::new(2.0, 2.0, 2.0, 2.0);
        assert!((hypervolume_4d(&pts, &rf) - 1.0).abs() < 1e-10);
    }
}
