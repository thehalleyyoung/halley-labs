//! Repair pipeline composing multiple repair strategies.

/// A pipeline of repair steps applied in sequence.
#[derive(Debug, Default)]
pub struct RepairPipeline { steps: Vec<Box<dyn RepairStep>> }

/// A single repair step.
pub trait RepairStep: std::fmt::Debug {
    fn apply(&self, state: &mut [f64]) -> bool;
    fn name(&self) -> &str;
}

impl RepairPipeline {
    pub fn new() -> Self { Self::default() }
    pub fn add_step(&mut self, step: Box<dyn RepairStep>) { self.steps.push(step); }
    /// Run all repair steps in order.
    pub fn run(&self, state: &mut [f64]) -> bool {
        let mut all_ok = true;
        for step in &self.steps {
            if !step.apply(state) { all_ok = false; }
        }
        all_ok
    }
}
