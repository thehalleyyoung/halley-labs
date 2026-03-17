//! Ground truth management for evaluation scenarios.
//!
//! Defines the known-correct fault locations against which the localizer's
//! output is compared to compute accuracy metrics.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A ground truth dataset mapping scenarios to their known faulty stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruth {
    pub name: String,
    pub description: String,
    pub entries: Vec<GroundTruthEntry>,
}

/// A single ground truth entry for one evaluation scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthEntry {
    pub scenario_id: String,
    pub pipeline_name: String,
    pub faulty_stages: Vec<String>,
    pub fault_types: HashMap<String, String>,
    pub expected_ranking: Vec<String>,
    pub severity: f64,
    pub notes: String,
}

impl GroundTruth {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            entries: Vec::new(),
        }
    }

    pub fn add_entry(&mut self, entry: GroundTruthEntry) {
        self.entries.push(entry);
    }

    pub fn get_entry(&self, scenario_id: &str) -> Option<&GroundTruthEntry> {
        self.entries.iter().find(|e| e.scenario_id == scenario_id)
    }

    /// Get all faulty stages across all entries.
    pub fn all_faulty_stages(&self) -> Vec<String> {
        let mut stages: Vec<String> = self
            .entries
            .iter()
            .flat_map(|e| e.faulty_stages.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        stages.sort();
        stages
    }

    /// Get the number of scenarios.
    pub fn scenario_count(&self) -> usize {
        self.entries.len()
    }

    /// Filter entries by pipeline.
    pub fn filter_by_pipeline(&self, pipeline: &str) -> Vec<&GroundTruthEntry> {
        self.entries
            .iter()
            .filter(|e| e.pipeline_name == pipeline)
            .collect()
    }
}

/// Builder for creating ground truth datasets programmatically.
pub struct GroundTruthBuilder {
    ground_truth: GroundTruth,
    counter: usize,
}

impl GroundTruthBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            ground_truth: GroundTruth::new(name),
            counter: 0,
        }
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.ground_truth.description = desc.into();
        self
    }

    /// Add a single-fault scenario.
    pub fn single_fault(
        mut self,
        pipeline: impl Into<String>,
        faulty_stage: impl Into<String>,
        fault_type: impl Into<String>,
        severity: f64,
    ) -> Self {
        self.counter += 1;
        let stage = faulty_stage.into();
        let ft = fault_type.into();
        self.ground_truth.add_entry(GroundTruthEntry {
            scenario_id: format!("scenario_{:03}", self.counter),
            pipeline_name: pipeline.into(),
            faulty_stages: vec![stage.clone()],
            fault_types: {
                let mut m = HashMap::new();
                m.insert(stage.clone(), ft);
                m
            },
            expected_ranking: vec![stage],
            severity,
            notes: String::new(),
        });
        self
    }

    /// Add a multi-fault scenario.
    pub fn multi_fault(
        mut self,
        pipeline: impl Into<String>,
        faulty_stages: Vec<(String, String)>,
        expected_ranking: Vec<String>,
        severity: f64,
    ) -> Self {
        self.counter += 1;
        let stages: Vec<String> = faulty_stages.iter().map(|(s, _)| s.clone()).collect();
        let types: HashMap<String, String> = faulty_stages.into_iter().collect();
        self.ground_truth.add_entry(GroundTruthEntry {
            scenario_id: format!("scenario_{:03}", self.counter),
            pipeline_name: pipeline.into(),
            faulty_stages: stages,
            fault_types: types,
            expected_ranking,
            severity,
            notes: String::new(),
        });
        self
    }

    /// Add a cascading fault scenario.
    pub fn cascading_fault(
        mut self,
        pipeline: impl Into<String>,
        source_stage: impl Into<String>,
        cascading_stages: Vec<String>,
        severity: f64,
    ) -> Self {
        self.counter += 1;
        let source = source_stage.into();
        let mut all_stages = vec![source.clone()];
        all_stages.extend(cascading_stages.iter().cloned());
        let types: HashMap<String, String> = {
            let mut m = HashMap::new();
            m.insert(source.clone(), "source".to_string());
            for s in &cascading_stages {
                m.insert(s.clone(), "cascading".to_string());
            }
            m
        };
        let mut ranking = vec![source];
        ranking.extend(cascading_stages);

        self.ground_truth.add_entry(GroundTruthEntry {
            scenario_id: format!("scenario_{:03}", self.counter),
            pipeline_name: pipeline.into(),
            faulty_stages: all_stages,
            fault_types: types,
            expected_ranking: ranking,
            severity,
            notes: "Cascading fault scenario".to_string(),
        });
        self
    }

    pub fn build(self) -> GroundTruth {
        self.ground_truth
    }
}

/// Create a standard ground truth dataset for a 4-stage spaCy-like pipeline.
pub fn standard_spacy_ground_truth() -> GroundTruth {
    GroundTruthBuilder::new("spacy_standard")
        .description("Standard evaluation ground truth for 4-stage spaCy pipeline")
        .single_fault("spacy", "tagger", "tag_flip", 0.6)
        .single_fault("spacy", "parser", "dep_label_swap", 0.5)
        .single_fault("spacy", "ner", "entity_span_shift", 0.7)
        .single_fault("spacy", "tokenizer", "token_merge", 0.3)
        .multi_fault(
            "spacy",
            vec![
                ("tagger".into(), "tag_flip".into()),
                ("ner".into(), "entity_label_flip".into()),
            ],
            vec!["tagger".into(), "ner".into()],
            0.6,
        )
        .cascading_fault(
            "spacy",
            "tagger",
            vec!["parser".into(), "ner".into()],
            0.7,
        )
        .build()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ground_truth_builder() {
        let gt = GroundTruthBuilder::new("test")
            .description("test gt")
            .single_fault("pipe", "tagger", "flip", 0.5)
            .single_fault("pipe", "parser", "swap", 0.6)
            .build();

        assert_eq!(gt.scenario_count(), 2);
        assert_eq!(gt.all_faulty_stages().len(), 2);
    }

    #[test]
    fn test_ground_truth_lookup() {
        let gt = GroundTruthBuilder::new("test")
            .single_fault("pipe", "tagger", "flip", 0.5)
            .build();

        let entry = gt.get_entry("scenario_001").unwrap();
        assert_eq!(entry.faulty_stages, vec!["tagger"]);
    }

    #[test]
    fn test_standard_ground_truth() {
        let gt = standard_spacy_ground_truth();
        assert!(gt.scenario_count() >= 5);
        assert!(gt.all_faulty_stages().contains(&"tagger".to_string()));
    }

    #[test]
    fn test_cascading_ground_truth() {
        let gt = GroundTruthBuilder::new("cascade_test")
            .cascading_fault("pipe", "tagger", vec!["parser".into()], 0.7)
            .build();

        let entry = gt.get_entry("scenario_001").unwrap();
        assert_eq!(entry.faulty_stages.len(), 2);
        assert_eq!(entry.expected_ranking[0], "tagger");
        assert!(entry.notes.contains("Cascading"));
    }

    #[test]
    fn test_filter_by_pipeline() {
        let gt = GroundTruthBuilder::new("multi_pipe")
            .single_fault("spacy", "tagger", "flip", 0.5)
            .single_fault("hf", "encoder", "noise", 0.6)
            .single_fault("spacy", "parser", "swap", 0.4)
            .build();

        let spacy = gt.filter_by_pipeline("spacy");
        assert_eq!(spacy.len(), 2);
        let hf = gt.filter_by_pipeline("hf");
        assert_eq!(hf.len(), 1);
    }
}
