use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalizationHypothesis { pub stage_name: String, pub is_faulty: bool, pub evidence_strength: f64 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisResult { pub hypothesis: LocalizationHypothesis, pub p_value: f64, pub rejected: bool }

pub struct HypothesisFramework { pub significance_level: f64 }
impl HypothesisFramework {
    pub fn new(sl: f64) -> Self { Self { significance_level: sl } }
}
impl Default for HypothesisFramework { fn default() -> Self { Self::new(0.05) } }
