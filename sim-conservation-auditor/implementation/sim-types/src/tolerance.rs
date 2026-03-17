use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToleranceKind {
    Absolute,
    Relative,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Tolerance {
    pub kind: ToleranceKind,
    pub value: f64,
}

impl Tolerance {
    pub fn absolute(value: f64) -> Self {
        Self { kind: ToleranceKind::Absolute, value }
    }

    pub fn relative(value: f64) -> Self {
        Self { kind: ToleranceKind::Relative, value }
    }

    pub fn check(&self, expected: f64, actual: f64) -> bool {
        match self.kind {
            ToleranceKind::Absolute => (actual - expected).abs() <= self.value,
            ToleranceKind::Relative => {
                if expected.abs() < 1e-15 {
                    (actual - expected).abs() <= self.value
                } else {
                    ((actual - expected) / expected).abs() <= self.value
                }
            }
        }
    }
}

impl Default for Tolerance {
    fn default() -> Self {
        Self::absolute(1e-10)
    }
}
