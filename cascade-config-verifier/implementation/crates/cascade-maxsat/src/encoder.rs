//! Repair encoding: translate cascade failure repair problems into MaxSAT.

use serde::{Deserialize, Serialize};

use crate::formula::{HardClause, Literal, SoftClause};
use crate::solver::Model;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ParameterType {
    RetryCount,
    TimeoutMs,
    CircuitBreakerThreshold,
    RateLimit,
    BulkheadSize,
}

impl std::fmt::Display for ParameterType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RetryCount => write!(f, "retry_count"),
            Self::TimeoutMs => write!(f, "timeout_ms"),
            Self::CircuitBreakerThreshold => write!(f, "circuit_breaker_threshold"),
            Self::RateLimit => write!(f, "rate_limit"),
            Self::BulkheadSize => write!(f, "bulkhead_size"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterVariable {
    pub variable_id: u32,
    pub service_id: String,
    pub parameter_type: ParameterType,
    pub min_value: f64,
    pub max_value: f64,
    pub current_value: f64,
    pub step_size: f64,
    pub bit_width: u32,
}

impl ParameterVariable {
    pub fn num_steps(&self) -> u32 {
        let range = self.max_value - self.min_value;
        if self.step_size <= 0.0 || range <= 0.0 {
            return 1;
        }
        (range / self.step_size).ceil() as u32 + 1
    }

    pub fn decode_step(&self, step: u32) -> f64 {
        (self.min_value + step as f64 * self.step_size).min(self.max_value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviationWeight {
    pub parameter_type: ParameterType,
    pub weight_per_unit: u64,
}

pub struct RepairEncoder {
    variables: Vec<ParameterVariable>,
    deviation_weights: Vec<DeviationWeight>,
    next_var: u32,
}

impl RepairEncoder {
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            deviation_weights: Vec::new(),
            next_var: 1,
        }
    }

    pub fn add_parameter(
        &mut self,
        service_id: &str,
        param_type: ParameterType,
        min: f64,
        max: f64,
        current: f64,
        step: f64,
    ) -> ParameterVariable {
        let steps = {
            let range = max - min;
            if step <= 0.0 || range <= 0.0 {
                1u32
            } else {
                (range / step).ceil() as u32 + 1
            }
        };
        let bit_width = (steps as f64).log2().ceil().max(1.0) as u32;
        let variable_id = self.next_var;
        self.next_var += bit_width;

        let pv = ParameterVariable {
            variable_id,
            service_id: service_id.to_string(),
            parameter_type: param_type,
            min_value: min,
            max_value: max,
            current_value: current,
            step_size: step,
            bit_width,
        };
        self.variables.push(pv.clone());
        pv
    }

    pub fn set_deviation_weights(&mut self, weights: Vec<DeviationWeight>) {
        self.deviation_weights = weights;
    }

    pub fn encode_domain_constraints(&self) -> Vec<HardClause> {
        let mut clauses = Vec::new();
        let mut clause_id = 0usize;

        for pv in &self.variables {
            let max_step = pv.num_steps();
            let first_bit = pv.variable_id as Literal;
            clauses.push(HardClause {
                literals: vec![first_bit, -(first_bit)],
                label: format!("domain_{}", pv.service_id),
                id: clause_id,
            });
            clause_id += 1;

            if pv.bit_width > 1 && max_step < (1u32 << pv.bit_width) {
                let mut block_lits: Vec<Literal> = Vec::new();
                for b in 0..pv.bit_width {
                    let bit_val = (max_step >> b) & 1;
                    let var = (pv.variable_id + b) as Literal;
                    block_lits.push(if bit_val == 1 { -var } else { var });
                }
                clauses.push(HardClause {
                    literals: block_lits,
                    label: format!("domain_bound_{}_{}", pv.service_id, pv.parameter_type),
                    id: clause_id,
                });
                clause_id += 1;
            }
        }

        clauses
    }

    pub fn encode_deviation_objectives(&self) -> Vec<SoftClause> {
        let mut clauses = Vec::new();
        let mut clause_id = 0usize;

        for pv in &self.variables {
            let weight = self
                .deviation_weights
                .iter()
                .find(|dw| dw.parameter_type == pv.parameter_type)
                .map(|dw| dw.weight_per_unit)
                .unwrap_or(1);

            let current_step = if pv.step_size > 0.0 {
                ((pv.current_value - pv.min_value) / pv.step_size).round() as u32
            } else {
                0
            };

            for b in 0..pv.bit_width {
                let bit_val = (current_step >> b) & 1;
                let var = (pv.variable_id + b) as Literal;
                let preferred_lit = if bit_val == 1 { var } else { -var };
                clauses.push(SoftClause {
                    literals: vec![preferred_lit],
                    weight,
                    label: format!(
                        "deviation_{}_{}_bit{}",
                        pv.service_id, pv.parameter_type, b
                    ),
                    id: clause_id,
                });
                clause_id += 1;
            }
        }

        clauses
    }

    pub fn decode_model(&self, model: &Model) -> Vec<(String, String, f64)> {
        let mut result = Vec::new();

        for pv in &self.variables {
            let mut step = 0u32;
            for b in 0..pv.bit_width {
                let var = pv.variable_id + b;
                if model.value(var) {
                    step |= 1 << b;
                }
            }
            let value = pv.decode_step(step.min(pv.num_steps().saturating_sub(1)));
            result.push((
                pv.service_id.clone(),
                pv.parameter_type.to_string(),
                value,
            ));
        }

        result
    }

    pub fn variable_count(&self) -> u32 {
        self.next_var - 1
    }
}

impl Default for RepairEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_parameter() {
        let mut enc = RepairEncoder::new();
        let pv = enc.add_parameter("svc-a", ParameterType::RetryCount, 0.0, 5.0, 3.0, 1.0);
        assert_eq!(pv.service_id, "svc-a");
        assert_eq!(pv.num_steps(), 6);
        assert!(pv.bit_width >= 3);
    }

    #[test]
    fn test_encode_domain_constraints() {
        let mut enc = RepairEncoder::new();
        enc.add_parameter("svc-a", ParameterType::TimeoutMs, 100.0, 500.0, 200.0, 100.0);
        let clauses = enc.encode_domain_constraints();
        assert!(!clauses.is_empty());
    }

    #[test]
    fn test_encode_deviation_objectives() {
        let mut enc = RepairEncoder::new();
        enc.add_parameter("svc-a", ParameterType::RateLimit, 10.0, 100.0, 50.0, 10.0);
        enc.set_deviation_weights(vec![DeviationWeight {
            parameter_type: ParameterType::RateLimit,
            weight_per_unit: 3,
        }]);
        let soft = enc.encode_deviation_objectives();
        assert!(!soft.is_empty());
        assert!(soft.iter().all(|s| s.weight == 3));
    }

    #[test]
    fn test_decode_model_roundtrip() {
        let mut enc = RepairEncoder::new();
        enc.add_parameter("svc-a", ParameterType::RetryCount, 0.0, 3.0, 2.0, 1.0);
        let model = Model {
            assignments: vec![false, true],
        };
        let decoded = enc.decode_model(&model);
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].0, "svc-a");
        assert!((decoded[0].2 - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_variable_count() {
        let mut enc = RepairEncoder::new();
        assert_eq!(enc.variable_count(), 0);
        enc.add_parameter("a", ParameterType::BulkheadSize, 1.0, 8.0, 4.0, 1.0);
        assert!(enc.variable_count() > 0);
    }

    #[test]
    fn test_parameter_type_display() {
        assert_eq!(ParameterType::RetryCount.to_string(), "retry_count");
        assert_eq!(ParameterType::TimeoutMs.to_string(), "timeout_ms");
    }

    #[test]
    fn test_parameter_variable_decode_step() {
        let pv = ParameterVariable {
            variable_id: 1,
            service_id: "s".into(),
            parameter_type: ParameterType::RetryCount,
            min_value: 0.0,
            max_value: 5.0,
            current_value: 3.0,
            step_size: 1.0,
            bit_width: 3,
        };
        assert!((pv.decode_step(0) - 0.0).abs() < f64::EPSILON);
        assert!((pv.decode_step(3) - 3.0).abs() < f64::EPSILON);
        assert!((pv.decode_step(100) - 5.0).abs() < f64::EPSILON);
    }
}
