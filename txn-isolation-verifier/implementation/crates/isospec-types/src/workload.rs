//! Workload definitions for analysis.
use serde::{Deserialize, Serialize};
use crate::identifier::*;
use crate::ir::IrProgram;
use crate::schema::Schema;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workload {
    pub id: WorkloadId,
    pub name: String,
    pub program: IrProgram,
    pub schema: Schema,
    pub parameters: WorkloadParameters,
    pub annotations: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadParameters {
    pub transaction_bound: usize,
    pub operation_bound: usize,
    pub data_item_bound: usize,
    pub repetitions: usize,
}

impl Default for WorkloadParameters {
    fn default() -> Self {
        Self { transaction_bound: 3, operation_bound: 10, data_item_bound: 20, repetitions: 1 }
    }
}

impl Workload {
    pub fn new(name: impl Into<String>, program: IrProgram, schema: Schema) -> Self {
        Self {
            id: WorkloadId::new(0),
            name: name.into(),
            program,
            schema,
            parameters: WorkloadParameters::default(),
            annotations: HashMap::new(),
        }
    }
    pub fn with_parameters(mut self, params: WorkloadParameters) -> Self {
        self.parameters = params;
        self
    }
    pub fn transaction_count(&self) -> usize { self.program.transaction_count() }
    pub fn total_operations(&self) -> usize { self.program.total_statements() }
    pub fn tables_accessed(&self) -> Vec<String> { self.program.tables_accessed() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadSuite {
    pub name: String,
    pub workloads: Vec<Workload>,
}

impl WorkloadSuite {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), workloads: Vec::new() }
    }
    pub fn add(&mut self, w: Workload) { self.workloads.push(w); }
    pub fn len(&self) -> usize { self.workloads.len() }
    pub fn is_empty(&self) -> bool { self.workloads.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::IrProgram;
    use crate::schema::Schema;
    #[test]
    fn test_workload_creation() {
        let w = Workload::new("test", IrProgram::new("test"), Schema::new());
        assert_eq!(w.transaction_count(), 0);
    }
}
