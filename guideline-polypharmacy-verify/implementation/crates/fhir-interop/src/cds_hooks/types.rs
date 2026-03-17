//! CDS Hooks type definitions.

use serde::{Deserialize, Serialize};

/// FHIR authorization token for CDS Hooks.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FhirAuthorization {
    #[serde(default)]
    pub access_token: Option<String>,
    #[serde(default)]
    pub token_type: Option<String>,
    #[serde(default)]
    pub expires_in: Option<u64>,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub subject: Option<String>,
}

/// CDS Hooks indicator severity for response cards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Indicator {
    /// Informational – no action required.
    Info,
    /// Warning – review recommended.
    Warning,
    /// Critical – action required.
    Critical,
}

impl Default for Indicator {
    fn default() -> Self {
        Indicator::Info
    }
}

/// CDS Hooks selection behavior for suggestions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SelectionBehavior {
    AtMostOne,
    Any,
}

/// A link to an external resource in a CDS Hook card.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    pub label: String,
    pub url: String,
    #[serde(rename = "type", default)]
    pub link_type: Option<String>,
}

/// A FHIR resource action within a CDS Hook suggestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestionAction {
    #[serde(rename = "type")]
    pub action_type: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub resource: Option<serde_json::Value>,
}

/// A suggestion within a CDS Hook card.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suggestion {
    pub label: String,
    #[serde(default)]
    pub uuid: Option<String>,
    #[serde(rename = "isRecommended", default)]
    pub is_recommended: Option<bool>,
    #[serde(default)]
    pub actions: Vec<SuggestionAction>,
}

/// Source attribution for a CDS Hook card.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Source {
    pub label: String,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub icon: Option<String>,
    #[serde(default)]
    pub topic: Option<serde_json::Value>,
}
