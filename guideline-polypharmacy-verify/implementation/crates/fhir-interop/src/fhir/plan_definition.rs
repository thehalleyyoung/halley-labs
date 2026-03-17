//! FHIR R4 PlanDefinition resource → guideline reference.

use serde::{Deserialize, Serialize};

use guardpharma_types::GuidelineId;

use super::types::Meta;
use crate::error::FhirInteropError;

/// FHIR R4 PlanDefinition resource (subset relevant to clinical guidelines).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirPlanDefinition {
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub meta: Option<Meta>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(rename = "useContext", default)]
    pub use_context: Vec<serde_json::Value>,
    #[serde(rename = "relatedArtifact", default)]
    pub related_artifact: Vec<serde_json::Value>,
    #[serde(default)]
    pub action: Vec<PlanDefinitionAction>,
}

/// An action within a PlanDefinition (e.g., "prescribe drug X if condition Y").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanDefinitionAction {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub condition: Vec<PlanDefinitionCondition>,
    #[serde(rename = "definitionCanonical", default)]
    pub definition_canonical: Option<String>,
    #[serde(default)]
    pub action: Vec<PlanDefinitionAction>,
}

/// A condition (applicability trigger) within a PlanDefinition action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanDefinitionCondition {
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub expression: Option<PlanDefinitionExpression>,
}

/// An expression within a PlanDefinition condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanDefinitionExpression {
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub expression: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

/// A simplified guideline reference produced from a PlanDefinition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineReference {
    pub id: GuidelineId,
    pub name: String,
    pub description: Option<String>,
    pub url: Option<String>,
    pub recommended_actions: Vec<String>,
}

impl FhirPlanDefinition {
    /// Parse from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, FhirInteropError> {
        let res: Self = serde_json::from_str(json)?;
        if res.resource_type != "PlanDefinition" {
            return Err(FhirInteropError::ResourceTypeMismatch {
                expected: "PlanDefinition".into(),
                got: res.resource_type,
            });
        }
        Ok(res)
    }

    /// Convert to a GuidelineReference.
    pub fn to_guideline_reference(&self) -> GuidelineReference {
        let name = self
            .title
            .clone()
            .or_else(|| self.name.clone())
            .unwrap_or_else(|| "Unnamed Guideline".to_string());

        let id = {
            let u = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_DNS, name.as_bytes());
            GuidelineId::from_uuid(u)
        };

        let recommended_actions = collect_action_titles(&self.action);

        GuidelineReference {
            id,
            name,
            description: self.description.clone(),
            url: self.url.clone(),
            recommended_actions,
        }
    }
}

/// Recursively collect action titles from a PlanDefinition action tree.
fn collect_action_titles(actions: &[PlanDefinitionAction]) -> Vec<String> {
    let mut titles = Vec::new();
    for action in actions {
        if let Some(t) = &action.title {
            titles.push(t.clone());
        }
        titles.extend(collect_action_titles(&action.action));
    }
    titles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_plan_definition() {
        let json = r#"{
            "resourceType": "PlanDefinition",
            "id": "pd-1",
            "title": "AHA Hypertension Guideline 2023",
            "status": "active",
            "description": "Evidence-based guideline for hypertension management",
            "url": "http://example.org/guidelines/aha-htn-2023",
            "action": [
                {"title": "Start ACE inhibitor or ARB"},
                {"title": "Add thiazide diuretic if BP not at goal",
                 "action": [
                     {"title": "Consider adding CCB"}
                 ]}
            ]
        }"#;

        let pd = FhirPlanDefinition::from_json(json).unwrap();
        let gr = pd.to_guideline_reference();
        assert_eq!(gr.name, "AHA Hypertension Guideline 2023");
        assert_eq!(gr.recommended_actions.len(), 3);
        assert!(gr.recommended_actions.contains(&"Consider adding CCB".to_string()));
    }
}
