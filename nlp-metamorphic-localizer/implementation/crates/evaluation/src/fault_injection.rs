//! Fault injection framework for creating ground-truth evaluation scenarios.
//!
//! Implements controlled fault injection into pipeline stages to create
//! known-faulty configurations for evaluating localization accuracy.

use serde::{Deserialize, Serialize};
use shared_types::ir::{IntermediateRepresentation, IRType};
use shared_types::types::StageId;
use std::collections::HashMap;

/// A specific fault that can be injected into a pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectedFault {
    pub id: String,
    pub name: String,
    pub description: String,
    pub target_stage: String,
    pub fault_type: FaultType,
    pub severity: f64,
    pub trigger_condition: TriggerCondition,
}

/// Types of faults that can be injected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FaultType {
    /// Randomly flip POS tags with the given probability.
    TagFlip { flip_probability: f64, target_tags: Vec<String> },
    /// Randomly drop tokens.
    TokenDrop { drop_probability: f64 },
    /// Shift entity spans by the given offset.
    EntitySpanShift { offset: i32 },
    /// Add random noise to embeddings / numeric features.
    GaussianNoise { mean: f64, std_dev: f64 },
    /// Swap dependency arc labels.
    DependencyLabelSwap { swap_pairs: Vec<(String, String)> },
    /// Drop dependency arcs with the given probability.
    DependencyArcDrop { drop_probability: f64 },
    /// Change entity labels randomly.
    EntityLabelFlip { flip_probability: f64, target_labels: Vec<String> },
    /// Merge adjacent tokens.
    TokenMerge { merge_probability: f64 },
    /// Split tokens at random positions.
    TokenSplit { split_probability: f64 },
    /// Truncate output to a fraction of the original.
    OutputTruncation { keep_fraction: f64 },
    /// Custom fault defined by a closure description.
    Custom { mutation_description: String },
}

impl FaultType {
    /// Human-readable name of the fault type.
    pub fn type_name(&self) -> &str {
        match self {
            FaultType::TagFlip { .. } => "tag_flip",
            FaultType::TokenDrop { .. } => "token_drop",
            FaultType::EntitySpanShift { .. } => "entity_span_shift",
            FaultType::GaussianNoise { .. } => "gaussian_noise",
            FaultType::DependencyLabelSwap { .. } => "dep_label_swap",
            FaultType::DependencyArcDrop { .. } => "dep_arc_drop",
            FaultType::EntityLabelFlip { .. } => "entity_label_flip",
            FaultType::TokenMerge { .. } => "token_merge",
            FaultType::TokenSplit { .. } => "token_split",
            FaultType::OutputTruncation { .. } => "output_truncation",
            FaultType::Custom { .. } => "custom",
        }
    }
}

/// Conditions under which a fault is triggered.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerCondition {
    /// Always active.
    Always,
    /// Active only for inputs matching a text pattern.
    InputPattern { pattern: String },
    /// Active only for inputs longer than a threshold.
    InputLengthAbove { min_tokens: usize },
    /// Active with a given probability per input.
    Probabilistic { probability: f64 },
    /// Active only for specific transformation types.
    TransformationType { transformations: Vec<String> },
    /// Active only when the input contains specific POS patterns.
    PosPattern { required_tags: Vec<String> },
}

impl TriggerCondition {
    /// Check if the trigger is satisfied for a given input context.
    pub fn is_triggered(&self, input_text: &str, transformation: &str, rng_val: f64) -> bool {
        match self {
            TriggerCondition::Always => true,
            TriggerCondition::InputPattern { pattern } => input_text.contains(pattern.as_str()),
            TriggerCondition::InputLengthAbove { min_tokens } => {
                input_text.split_whitespace().count() > *min_tokens
            }
            TriggerCondition::Probabilistic { probability } => rng_val < *probability,
            TriggerCondition::TransformationType { transformations } => {
                transformations.iter().any(|t| t == transformation)
            }
            TriggerCondition::PosPattern { required_tags } => {
                // Simplified: check if input text contains any word that looks like a tag.
                required_tags.iter().any(|tag| input_text.contains(tag))
            }
        }
    }
}

/// A fault profile defines a set of faults to inject together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultProfile {
    pub name: String,
    pub description: String,
    pub faults: Vec<InjectedFault>,
    pub expected_localization: Vec<String>,
}

impl FaultProfile {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            faults: Vec::new(),
            expected_localization: Vec::new(),
        }
    }

    pub fn add_fault(&mut self, fault: InjectedFault) {
        self.expected_localization.push(fault.target_stage.clone());
        self.faults.push(fault);
    }

    pub fn faulty_stages(&self) -> Vec<String> {
        self.faults
            .iter()
            .map(|f| f.target_stage.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }
}

/// Injection site in the pipeline where a fault is applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionSite {
    pub stage_name: String,
    pub stage_index: usize,
    pub injection_point: InjectionPoint,
}

/// Where within a stage the fault is injected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InjectionPoint {
    /// Before the stage processes input.
    PreProcessing,
    /// After the stage processes, mutating its output.
    PostProcessing,
    /// Replace the stage entirely.
    Replacement,
}

/// The fault injector manages fault profiles and applies mutations.
pub struct FaultInjector {
    profiles: Vec<FaultProfile>,
    active_profile: Option<usize>,
    injection_log: Vec<InjectionLogEntry>,
}

/// A log entry recording a fault injection event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionLogEntry {
    pub fault_id: String,
    pub stage_name: String,
    pub input_text: String,
    pub was_triggered: bool,
    pub mutation_applied: String,
}

impl FaultInjector {
    pub fn new() -> Self {
        Self {
            profiles: Vec::new(),
            active_profile: None,
            injection_log: Vec::new(),
        }
    }

    /// Register a fault profile.
    pub fn register_profile(&mut self, profile: FaultProfile) -> usize {
        let idx = self.profiles.len();
        self.profiles.push(profile);
        idx
    }

    /// Activate a fault profile by index.
    pub fn activate_profile(&mut self, index: usize) -> bool {
        if index < self.profiles.len() {
            self.active_profile = Some(index);
            true
        } else {
            false
        }
    }

    /// Deactivate all fault profiles.
    pub fn deactivate(&mut self) {
        self.active_profile = None;
    }

    /// Get the currently active profile.
    pub fn active_profile(&self) -> Option<&FaultProfile> {
        self.active_profile
            .and_then(|idx| self.profiles.get(idx))
    }

    /// Check if a fault should be triggered for the given context.
    pub fn should_trigger(
        &self,
        stage_name: &str,
        input_text: &str,
        transformation: &str,
    ) -> Vec<&InjectedFault> {
        let profile = match self.active_profile() {
            Some(p) => p,
            None => return Vec::new(),
        };

        profile
            .faults
            .iter()
            .filter(|f| {
                f.target_stage == stage_name
                    && f.trigger_condition.is_triggered(input_text, transformation, 0.5)
            })
            .collect()
    }

    /// Apply a tag-flip fault to a POS tag sequence.
    pub fn apply_tag_flip(
        tags: &[String],
        flip_probability: f64,
        target_tags: &[String],
        rng_values: &[f64],
    ) -> Vec<String> {
        let replacement_tags = ["NN", "VB", "JJ", "RB", "DT", "IN"];
        tags.iter()
            .enumerate()
            .map(|(i, tag)| {
                let rng = rng_values.get(i).copied().unwrap_or(1.0);
                if rng < flip_probability
                    && (target_tags.is_empty() || target_tags.contains(tag))
                {
                    let idx = (rng * 1000.0) as usize % replacement_tags.len();
                    let replacement = replacement_tags[idx];
                    if replacement != tag {
                        replacement.to_string()
                    } else {
                        replacement_tags[(idx + 1) % replacement_tags.len()].to_string()
                    }
                } else {
                    tag.clone()
                }
            })
            .collect()
    }

    /// Apply entity span shift fault.
    pub fn apply_entity_span_shift(
        spans: &[(usize, usize, String)],
        offset: i32,
        max_len: usize,
    ) -> Vec<(usize, usize, String)> {
        spans
            .iter()
            .map(|(start, end, label)| {
                let new_start = (*start as i32 + offset).max(0) as usize;
                let new_end = (*end as i32 + offset).max(new_start as i32 + 1) as usize;
                (new_start.min(max_len), new_end.min(max_len), label.clone())
            })
            .collect()
    }

    /// Apply token drop fault.
    pub fn apply_token_drop(
        tokens: &[String],
        drop_probability: f64,
        rng_values: &[f64],
    ) -> Vec<String> {
        tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let rng = rng_values.get(*i).copied().unwrap_or(1.0);
                rng >= drop_probability
            })
            .map(|(_, t)| t.clone())
            .collect()
    }

    /// Apply Gaussian noise to numeric features.
    pub fn apply_gaussian_noise(
        values: &[f64],
        mean: f64,
        std_dev: f64,
        noise_values: &[f64],
    ) -> Vec<f64> {
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let noise = noise_values.get(i).copied().unwrap_or(0.0);
                v + mean + std_dev * noise
            })
            .collect()
    }

    /// Apply dependency label swap fault.
    pub fn apply_dep_label_swap(
        labels: &[String],
        swap_pairs: &[(String, String)],
    ) -> Vec<String> {
        labels
            .iter()
            .map(|label| {
                for (from, to) in swap_pairs {
                    if label == from {
                        return to.clone();
                    }
                    if label == to {
                        return from.clone();
                    }
                }
                label.clone()
            })
            .collect()
    }

    /// Get the injection log.
    pub fn get_log(&self) -> &[InjectionLogEntry] {
        &self.injection_log
    }

    /// Clear the injection log.
    pub fn clear_log(&mut self) {
        self.injection_log.clear();
    }

    /// Create a standard set of fault profiles for evaluation.
    pub fn standard_profiles() -> Vec<FaultProfile> {
        let mut profiles = Vec::new();

        // Profile 1: Tagger fault.
        let mut p1 = FaultProfile::new("tagger_fault");
        p1.description = "POS tagger mishandles passive voice constructions".to_string();
        p1.add_fault(InjectedFault {
            id: "tag_flip_passive".to_string(),
            name: "Passive voice tag flip".to_string(),
            description: "Flips VBN to VBD in passive constructions".to_string(),
            target_stage: "tagger".to_string(),
            fault_type: FaultType::TagFlip {
                flip_probability: 0.3,
                target_tags: vec!["VBN".to_string()],
            },
            severity: 0.6,
            trigger_condition: TriggerCondition::Always,
        });
        profiles.push(p1);

        // Profile 2: Parser fault.
        let mut p2 = FaultProfile::new("parser_fault");
        p2.description = "Dependency parser misattaches PPs".to_string();
        p2.add_fault(InjectedFault {
            id: "dep_swap_pp".to_string(),
            name: "PP attachment swap".to_string(),
            description: "Swaps prep/pobj arcs".to_string(),
            target_stage: "parser".to_string(),
            fault_type: FaultType::DependencyLabelSwap {
                swap_pairs: vec![("prep".to_string(), "pobj".to_string())],
            },
            severity: 0.5,
            trigger_condition: TriggerCondition::Always,
        });
        profiles.push(p2);

        // Profile 3: NER fault.
        let mut p3 = FaultProfile::new("ner_fault");
        p3.description = "NER shifts entity spans".to_string();
        p3.add_fault(InjectedFault {
            id: "ner_span_shift".to_string(),
            name: "Entity span shift".to_string(),
            description: "Shifts all entity spans by +1 token".to_string(),
            target_stage: "ner".to_string(),
            fault_type: FaultType::EntitySpanShift { offset: 1 },
            severity: 0.7,
            trigger_condition: TriggerCondition::Always,
        });
        profiles.push(p3);

        // Profile 4: Cascading fault (tagger → parser).
        let mut p4 = FaultProfile::new("cascading_tagger_parser");
        p4.description = "Tagger fault cascading to parser".to_string();
        p4.add_fault(InjectedFault {
            id: "tag_flip_cascade".to_string(),
            name: "Tag flip causing parser cascade".to_string(),
            description: "Flips verb tags causing parser misattachment".to_string(),
            target_stage: "tagger".to_string(),
            fault_type: FaultType::TagFlip {
                flip_probability: 0.2,
                target_tags: vec!["VB".to_string(), "VBZ".to_string(), "VBD".to_string()],
            },
            severity: 0.4,
            trigger_condition: TriggerCondition::Always,
        });
        profiles.push(p4);

        // Profile 5: Multi-fault (tagger + NER).
        let mut p5 = FaultProfile::new("multi_fault_tagger_ner");
        p5.description = "Simultaneous tagger and NER faults".to_string();
        p5.add_fault(InjectedFault {
            id: "multi_tag_flip".to_string(),
            name: "Tag flip".to_string(),
            description: "Random tag flips".to_string(),
            target_stage: "tagger".to_string(),
            fault_type: FaultType::TagFlip {
                flip_probability: 0.15,
                target_tags: vec![],
            },
            severity: 0.3,
            trigger_condition: TriggerCondition::Always,
        });
        p5.add_fault(InjectedFault {
            id: "multi_ner_flip".to_string(),
            name: "Entity label flip".to_string(),
            description: "Random entity label changes".to_string(),
            target_stage: "ner".to_string(),
            fault_type: FaultType::EntityLabelFlip {
                flip_probability: 0.25,
                target_labels: vec![],
            },
            severity: 0.5,
            trigger_condition: TriggerCondition::Always,
        });
        profiles.push(p5);

        // Profile 6: Intermittent tokenizer fault.
        let mut p6 = FaultProfile::new("intermittent_tokenizer");
        p6.description = "Tokenizer occasionally merges tokens on long inputs".to_string();
        p6.add_fault(InjectedFault {
            id: "token_merge_long".to_string(),
            name: "Token merge on long inputs".to_string(),
            description: "Merges adjacent tokens for inputs > 20 tokens".to_string(),
            target_stage: "tokenizer".to_string(),
            fault_type: FaultType::TokenMerge {
                merge_probability: 0.1,
            },
            severity: 0.3,
            trigger_condition: TriggerCondition::InputLengthAbove { min_tokens: 20 },
        });
        profiles.push(p6);

        profiles
    }
}

impl Default for FaultInjector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tag_flip() {
        let tags = vec!["VBN".into(), "NN".into(), "DT".into(), "VBN".into()];
        let rng = vec![0.1, 0.9, 0.9, 0.2];
        let result = FaultInjector::apply_tag_flip(&tags, 0.3, &["VBN".to_string()], &rng);
        assert_ne!(result[0], "VBN"); // should flip
        assert_eq!(result[1], "NN"); // not target tag
        assert_eq!(result[2], "DT"); // rng > threshold
        assert_ne!(result[3], "VBN"); // should flip
    }

    #[test]
    fn test_token_drop() {
        let tokens = vec!["the".into(), "cat".into(), "sat".into(), "on".into()];
        let rng = vec![0.1, 0.9, 0.1, 0.9];
        let result = FaultInjector::apply_token_drop(&tokens, 0.5, &rng);
        assert_eq!(result, vec!["cat", "on"]);
    }

    #[test]
    fn test_entity_span_shift() {
        let spans = vec![
            (2, 4, "PERSON".to_string()),
            (6, 8, "ORG".to_string()),
        ];
        let result = FaultInjector::apply_entity_span_shift(&spans, 1, 20);
        assert_eq!(result[0].0, 3);
        assert_eq!(result[0].1, 5);
        assert_eq!(result[1].0, 7);
    }

    #[test]
    fn test_gaussian_noise() {
        let values = vec![1.0, 2.0, 3.0];
        let noise = vec![0.5, -0.3, 0.1];
        let result = FaultInjector::apply_gaussian_noise(&values, 0.0, 0.1, &noise);
        assert!((result[0] - 1.05).abs() < 0.01);
        assert!((result[1] - 1.97).abs() < 0.01);
    }

    #[test]
    fn test_dep_label_swap() {
        let labels = vec!["nsubj".into(), "prep".into(), "pobj".into(), "det".into()];
        let pairs = vec![("prep".to_string(), "pobj".to_string())];
        let result = FaultInjector::apply_dep_label_swap(&labels, &pairs);
        assert_eq!(result, vec!["nsubj", "pobj", "prep", "det"]);
    }

    #[test]
    fn test_trigger_conditions() {
        assert!(TriggerCondition::Always.is_triggered("any", "any", 0.0));

        let pattern = TriggerCondition::InputPattern {
            pattern: "passive".to_string(),
        };
        assert!(pattern.is_triggered("this is passive voice", "pass", 0.0));
        assert!(!pattern.is_triggered("active voice", "pass", 0.0));

        let length = TriggerCondition::InputLengthAbove { min_tokens: 5 };
        assert!(length.is_triggered("one two three four five six", "t", 0.0));
        assert!(!length.is_triggered("one two", "t", 0.0));

        let prob = TriggerCondition::Probabilistic { probability: 0.5 };
        assert!(prob.is_triggered("x", "t", 0.3));
        assert!(!prob.is_triggered("x", "t", 0.7));

        let trans = TriggerCondition::TransformationType {
            transformations: vec!["passivization".to_string()],
        };
        assert!(trans.is_triggered("x", "passivization", 0.0));
        assert!(!trans.is_triggered("x", "clefting", 0.0));
    }

    #[test]
    fn test_fault_profile() {
        let mut profile = FaultProfile::new("test_profile");
        profile.add_fault(InjectedFault {
            id: "f1".to_string(),
            name: "test".to_string(),
            description: "test fault".to_string(),
            target_stage: "tagger".to_string(),
            fault_type: FaultType::TagFlip {
                flip_probability: 0.5,
                target_tags: vec![],
            },
            severity: 0.5,
            trigger_condition: TriggerCondition::Always,
        });
        profile.add_fault(InjectedFault {
            id: "f2".to_string(),
            name: "test2".to_string(),
            description: "test fault 2".to_string(),
            target_stage: "ner".to_string(),
            fault_type: FaultType::EntityLabelFlip {
                flip_probability: 0.3,
                target_labels: vec![],
            },
            severity: 0.4,
            trigger_condition: TriggerCondition::Always,
        });

        let stages = profile.faulty_stages();
        assert!(stages.contains(&"tagger".to_string()));
        assert!(stages.contains(&"ner".to_string()));
    }

    #[test]
    fn test_standard_profiles() {
        let profiles = FaultInjector::standard_profiles();
        assert!(profiles.len() >= 5);
        for p in &profiles {
            assert!(!p.name.is_empty());
            assert!(!p.faults.is_empty());
        }
    }

    #[test]
    fn test_injector_lifecycle() {
        let mut injector = FaultInjector::new();
        let profiles = FaultInjector::standard_profiles();
        for p in profiles {
            injector.register_profile(p);
        }
        assert!(injector.active_profile().is_none());
        injector.activate_profile(0);
        assert!(injector.active_profile().is_some());
        assert_eq!(injector.active_profile().unwrap().name, "tagger_fault");

        let triggers = injector.should_trigger("tagger", "test input", "passivization");
        assert_eq!(triggers.len(), 1);

        let no_triggers = injector.should_trigger("ner", "test input", "passivization");
        assert!(no_triggers.is_empty());

        injector.deactivate();
        assert!(injector.active_profile().is_none());
    }
}
