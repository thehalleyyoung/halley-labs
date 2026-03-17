//! CDS Hooks response (card) generation.

use serde::{Deserialize, Serialize};

use super::types::{Indicator, Link, SelectionBehavior, Source, Suggestion};

/// A CDS Hooks response containing decision support cards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdsHookResponse {
    pub cards: Vec<CdsCard>,
}

impl CdsHookResponse {
    /// Create an empty response with no cards.
    pub fn empty() -> Self {
        Self { cards: Vec::new() }
    }

    /// Create a response with a single card.
    pub fn with_card(card: CdsCard) -> Self {
        Self { cards: vec![card] }
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// A single CDS Hooks decision support card.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdsCard {
    /// A brief summary (≤140 characters recommended).
    pub summary: String,
    /// Detailed narrative.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Urgency indicator.
    pub indicator: Indicator,
    /// Where this recommendation came from.
    pub source: Source,
    /// Suggested actions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<Suggestion>,
    /// How suggestions may be selected.
    #[serde(rename = "selectionBehavior", default, skip_serializing_if = "Option::is_none")]
    pub selection_behavior: Option<SelectionBehavior>,
    /// External reference links.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<Link>,
}

/// Builder for constructing CDS Hook cards fluently.
pub struct CdsCardBuilder {
    summary: String,
    detail: Option<String>,
    indicator: Indicator,
    source: Source,
    suggestions: Vec<Suggestion>,
    selection_behavior: Option<SelectionBehavior>,
    links: Vec<Link>,
}

impl CdsCardBuilder {
    /// Start building a card with a summary.
    pub fn new(summary: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            detail: None,
            indicator: Indicator::Info,
            source: Source {
                label: "GuardPharma".to_string(),
                ..Default::default()
            },
            suggestions: Vec::new(),
            selection_behavior: None,
            links: Vec::new(),
        }
    }

    /// Set the detail text.
    pub fn detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    /// Set the indicator level.
    pub fn indicator(mut self, indicator: Indicator) -> Self {
        self.indicator = indicator;
        self
    }

    /// Mark as a warning.
    pub fn warning(self) -> Self {
        self.indicator(Indicator::Warning)
    }

    /// Mark as critical.
    pub fn critical(self) -> Self {
        self.indicator(Indicator::Critical)
    }

    /// Set the source.
    pub fn source(mut self, label: impl Into<String>, url: Option<String>) -> Self {
        self.source = Source {
            label: label.into(),
            url,
            icon: None,
            topic: None,
        };
        self
    }

    /// Add a suggestion.
    pub fn suggestion(mut self, suggestion: Suggestion) -> Self {
        self.suggestions.push(suggestion);
        self
    }

    /// Set the selection behavior.
    pub fn selection_behavior(mut self, behavior: SelectionBehavior) -> Self {
        self.selection_behavior = Some(behavior);
        self
    }

    /// Add a link.
    pub fn link(mut self, label: impl Into<String>, url: impl Into<String>) -> Self {
        self.links.push(Link {
            label: label.into(),
            url: url.into(),
            link_type: Some("absolute".to_string()),
        });
        self
    }

    /// Build the card.
    pub fn build(self) -> CdsCard {
        CdsCard {
            summary: self.summary,
            detail: self.detail,
            indicator: self.indicator,
            source: self.source,
            suggestions: self.suggestions,
            selection_behavior: self.selection_behavior,
            links: self.links,
        }
    }
}

/// Convenience: create a polypharmacy warning card.
pub fn polypharmacy_warning_card(
    medication_count: usize,
    conflicting_drugs: &[(String, String, String)],
) -> CdsCard {
    let summary = format!(
        "Polypharmacy alert: {medication_count} concurrent medications detected"
    );

    let detail = if conflicting_drugs.is_empty() {
        format!(
            "Patient is taking {medication_count} medications. \
             Review for potential deprescribing opportunities."
        )
    } else {
        let conflicts: Vec<String> = conflicting_drugs
            .iter()
            .map(|(a, b, reason)| format!("• {a} + {b}: {reason}"))
            .collect();
        format!(
            "Patient is taking {medication_count} medications with the following interactions:\n{}",
            conflicts.join("\n")
        )
    };

    let indicator = if medication_count >= 10 || !conflicting_drugs.is_empty() {
        Indicator::Critical
    } else if medication_count >= 5 {
        Indicator::Warning
    } else {
        Indicator::Info
    };

    CdsCardBuilder::new(summary)
        .detail(detail)
        .indicator(indicator)
        .source("GuardPharma Polypharmacy Verifier", None)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_card() {
        let card = CdsCardBuilder::new("Test alert")
            .detail("Detailed info")
            .warning()
            .source("GuardPharma", None)
            .link("Learn more", "https://example.org")
            .build();

        assert_eq!(card.summary, "Test alert");
        assert_eq!(card.indicator, Indicator::Warning);
        assert_eq!(card.links.len(), 1);
    }

    #[test]
    fn polypharmacy_card_serializes() {
        let card = polypharmacy_warning_card(
            8,
            &[("Warfarin".into(), "Aspirin".into(), "Increased bleeding risk".into())],
        );
        let response = CdsHookResponse::with_card(card);
        let json = response.to_json().unwrap();
        assert!(json.contains("Polypharmacy alert"));
        assert!(json.contains("Warfarin"));
    }
}
