use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub id: String,
    pub original_text: String,
    pub transformation_name: String,
    pub transformed_text: String,
    pub violated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub test_case_id: String,
    pub violated: bool,
    pub violation_magnitude: f64,
    pub stage_differentials: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuite {
    pub name: String,
    pub test_cases: Vec<TestCase>,
}

impl TestSuite {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), test_cases: Vec::new() }
    }
    pub fn add(&mut self, tc: TestCase) { self.test_cases.push(tc); }
    pub fn len(&self) -> usize { self.test_cases.len() }
    pub fn is_empty(&self) -> bool { self.test_cases.is_empty() }
}

pub struct TestSuiteBuilder {
    name: String,
    cases: Vec<TestCase>,
}

impl TestSuiteBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), cases: Vec::new() }
    }
    pub fn add_case(mut self, tc: TestCase) -> Self { self.cases.push(tc); self }
    pub fn build(self) -> TestSuite {
        TestSuite { name: self.name, test_cases: self.cases }
    }
}

pub struct TestCaseGenerator;
impl TestCaseGenerator { pub fn new() -> Self { Self } }
impl Default for TestCaseGenerator { fn default() -> Self { Self } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationRecord {
    pub test_case_id: String,
    pub transformation: String,
    pub violated_mr: String,
    pub magnitude: f64,
    pub severity: ViolationSeverity,
    pub faulty_stage: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ViolationSeverity { Low, Medium, High, Critical }

impl std::fmt::Display for ViolationSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Medium => write!(f, "Medium"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoverageTracker {
    pub coverage: HashMap<String, Vec<bool>>,
}

impl CoverageTracker {
    pub fn new() -> Self { Self { coverage: HashMap::new() } }
    pub fn record(&mut self, stage: &str, covered: bool) {
        self.coverage.entry(stage.to_string()).or_default().push(covered);
    }
}

impl Default for CoverageTracker { fn default() -> Self { Self::new() } }
