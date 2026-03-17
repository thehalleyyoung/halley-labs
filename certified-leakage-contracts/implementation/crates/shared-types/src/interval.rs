use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntervalBound { NegInf, Finite(i64), PosInf }

impl IntervalBound {
    pub fn is_finite(&self) -> bool { matches!(self, Self::Finite(_)) }
    pub fn value(&self) -> Option<i64> { match self { Self::Finite(v) => Some(*v), _ => None } }
    pub fn min(self, other: Self) -> Self {
        match (self, other) {
            (Self::NegInf, _) | (_, Self::NegInf) => Self::NegInf,
            (Self::PosInf, x) | (x, Self::PosInf) => x,
            (Self::Finite(a), Self::Finite(b)) => Self::Finite(a.min(b)),
        }
    }
    pub fn max(self, other: Self) -> Self {
        match (self, other) {
            (Self::PosInf, _) | (_, Self::PosInf) => Self::PosInf,
            (Self::NegInf, x) | (x, Self::NegInf) => x,
            (Self::Finite(a), Self::Finite(b)) => Self::Finite(a.max(b)),
        }
    }
}

impl PartialOrd for IntervalBound {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for IntervalBound {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Self::NegInf, Self::NegInf) => std::cmp::Ordering::Equal,
            (Self::NegInf, _) => std::cmp::Ordering::Less,
            (_, Self::NegInf) => std::cmp::Ordering::Greater,
            (Self::PosInf, Self::PosInf) => std::cmp::Ordering::Equal,
            (Self::PosInf, _) => std::cmp::Ordering::Greater,
            (_, Self::PosInf) => std::cmp::Ordering::Less,
            (Self::Finite(a), Self::Finite(b)) => a.cmp(b),
        }
    }
}
impl fmt::Display for IntervalBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::NegInf => write!(f, "-inf"), Self::Finite(v) => write!(f, "{}", v), Self::PosInf => write!(f, "+inf") }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Interval { pub lo: IntervalBound, pub hi: IntervalBound }

impl Interval {
    pub const BOTTOM: Self = Self { lo: IntervalBound::PosInf, hi: IntervalBound::NegInf };
    pub const TOP: Self = Self { lo: IntervalBound::NegInf, hi: IntervalBound::PosInf };
    pub fn new(lo: i64, hi: i64) -> Self { Self { lo: IntervalBound::Finite(lo), hi: IntervalBound::Finite(hi) } }
    pub fn singleton(v: i64) -> Self { Self::new(v, v) }
    pub fn is_bottom(&self) -> bool { self.lo > self.hi }
    pub fn is_top(&self) -> bool { self.lo == IntervalBound::NegInf && self.hi == IntervalBound::PosInf }
    pub fn contains(&self, v: i64) -> bool { let fv = IntervalBound::Finite(v); self.lo <= fv && fv <= self.hi }
    pub fn width(&self) -> Option<u64> {
        match (self.lo, self.hi) {
            (IntervalBound::Finite(a), IntervalBound::Finite(b)) if b >= a => Some((b - a) as u64 + 1),
            _ => None,
        }
    }
    pub fn join(&self, other: &Self) -> Self {
        if self.is_bottom() { return *other; }
        if other.is_bottom() { return *self; }
        Self { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }
    pub fn meet(&self, other: &Self) -> Self {
        Self { lo: self.lo.max(other.lo), hi: self.hi.min(other.hi) }
    }
    pub fn widen(&self, other: &Self) -> Self {
        let lo = if other.lo < self.lo { IntervalBound::NegInf } else { self.lo };
        let hi = if other.hi > self.hi { IntervalBound::PosInf } else { self.hi };
        Self { lo, hi }
    }
    pub fn narrow(&self, other: &Self) -> Self {
        let lo = if self.lo == IntervalBound::NegInf { other.lo } else { self.lo };
        let hi = if self.hi == IntervalBound::PosInf { other.hi } else { self.hi };
        Self { lo, hi }
    }
    pub fn includes(&self, other: &Self) -> bool {
        if other.is_bottom() { return true; }
        if self.is_bottom() { return false; }
        self.lo <= other.lo && self.hi >= other.hi
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_bottom() { write!(f, "bot") } else { write!(f, "[{}, {}]", self.lo, self.hi) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_interval_basic() { let i = Interval::new(1, 10); assert!(i.contains(5)); assert!(!i.contains(0)); }
    #[test] fn test_interval_join() { let a = Interval::new(1,5); let b = Interval::new(3,10); assert_eq!(a.join(&b), Interval::new(1,10)); }
    #[test] fn test_interval_meet() { let a = Interval::new(1,5); let b = Interval::new(3,10); assert_eq!(a.meet(&b), Interval::new(3,5)); }
}
