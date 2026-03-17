//! JSON-LD (JSON for Linked Data) format support.
//!
//! Converts RegSynth obligations and regulatory data into JSON-LD documents
//! using the [schema.org/Legislation](https://schema.org/Legislation) vocabulary
//! and [ELI](http://publications.europa.eu/resource/authority/eli)
//! (European Legislation Identifier) URI patterns.
//!
//! Reference: <https://www.w3.org/TR/json-ld11/>

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};

use regsynth_types::Obligation;
use regsynth_types::regulatory::RegulatoryFramework;

use crate::FormatResult;

// ── Vocabulary constants ──────────────────────────────────────────────

/// Standard JSON-LD context URIs.
pub mod vocab {
    pub const SCHEMA_ORG: &str = "https://schema.org/";
    pub const ELI: &str = "http://data.europa.eu/eli/ontology#";
    pub const DCTERMS: &str = "http://purl.org/dc/terms/";
    pub const SKOS: &str = "http://www.w3.org/2004/02/skos/core#";
    pub const XSD: &str = "http://www.w3.org/2001/XMLSchema#";
    pub const REGSYNTH: &str = "https://regsynth.dev/ontology/";
}

// ── JSON-LD context ───────────────────────────────────────────────────

/// A JSON-LD `@context` block mapping prefixes to IRIs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonLdContext {
    /// Prefix-to-IRI mappings.
    #[serde(flatten)]
    pub prefixes: IndexMap<String, JsonValue>,
}

impl JsonLdContext {
    /// Create the default RegSynth regulatory context.
    pub fn regulatory_default() -> Self {
        let mut prefixes = IndexMap::new();
        prefixes.insert("schema".into(), json!(vocab::SCHEMA_ORG));
        prefixes.insert("eli".into(), json!(vocab::ELI));
        prefixes.insert("dcterms".into(), json!(vocab::DCTERMS));
        prefixes.insert("skos".into(), json!(vocab::SKOS));
        prefixes.insert("xsd".into(), json!(vocab::XSD));
        prefixes.insert("regsynth".into(), json!(vocab::REGSYNTH));

        // Typed property aliases
        prefixes.insert(
            "legislationDate".into(),
            json!({ "@id": "schema:legislationDate", "@type": "xsd:date" }),
        );
        prefixes.insert(
            "legislationIdentifier".into(),
            json!("schema:legislationIdentifier"),
        );
        prefixes.insert(
            "legislationJurisdiction".into(),
            json!("schema:legislationJurisdiction"),
        );
        prefixes.insert("riskLevel".into(), json!("regsynth:riskLevel"));
        prefixes.insert("obligationType".into(), json!("regsynth:obligationType"));
        prefixes.insert(
            "regulatoryDomain".into(),
            json!("regsynth:regulatoryDomain"),
        );
        prefixes.insert(
            "complianceCost".into(),
            json!({ "@id": "regsynth:complianceCost", "@type": "xsd:decimal" }),
        );

        Self { prefixes }
    }

    /// Create a minimal context with only the given prefixes.
    pub fn from_prefixes(prefixes: impl IntoIterator<Item = (String, String)>) -> Self {
        let prefixes = prefixes
            .into_iter()
            .map(|(k, v)| (k, json!(v)))
            .collect();
        Self { prefixes }
    }
}

// ── JSON-LD node ──────────────────────────────────────────────────────

/// A single JSON-LD node with `@id`, `@type`, and properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonLdNode {
    /// Node identifier (IRI or blank node).
    #[serde(rename = "@id")]
    pub id: String,
    /// RDF type(s) for this node.
    #[serde(rename = "@type")]
    pub node_type: Vec<String>,
    /// Additional properties as key → JSON value.
    #[serde(flatten)]
    pub properties: IndexMap<String, JsonValue>,
}

impl JsonLdNode {
    /// Create a node with the given id and types.
    pub fn new(id: impl Into<String>, types: Vec<String>) -> Self {
        Self {
            id: id.into(),
            node_type: types,
            properties: IndexMap::new(),
        }
    }

    /// Set a string property.
    pub fn set_str(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.properties.insert(key.into(), json!(value.into()));
    }

    /// Set a numeric property.
    pub fn set_num(&mut self, key: impl Into<String>, value: f64) {
        self.properties.insert(key.into(), json!(value));
    }

    /// Set a reference (link) to another node by IRI.
    pub fn set_ref(&mut self, key: impl Into<String>, iri: impl Into<String>) {
        self.properties
            .insert(key.into(), json!({ "@id": iri.into() }));
    }

    /// Set a typed literal value.
    pub fn set_typed(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
        datatype: impl Into<String>,
    ) {
        self.properties.insert(
            key.into(),
            json!({ "@value": value.into(), "@type": datatype.into() }),
        );
    }
}

// ── Regulatory linked data document ───────────────────────────────────

/// A complete JSON-LD document for regulatory linked data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryLinkedData {
    /// JSON-LD context.
    #[serde(rename = "@context")]
    pub context: JsonLdContext,
    /// The graph of linked data nodes.
    #[serde(rename = "@graph")]
    pub graph: Vec<JsonLdNode>,
}

impl RegulatoryLinkedData {
    /// Create an empty regulatory linked data document with default context.
    pub fn new() -> Self {
        Self {
            context: JsonLdContext::regulatory_default(),
            graph: Vec::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: JsonLdNode) {
        self.graph.push(node);
    }

    /// Build an ELI-style URI for a regulation.
    ///
    /// Pattern: `http://data.europa.eu/eli/{jurisdiction}/{type}/{year}/{number}`
    pub fn eli_uri(jurisdiction: &str, doc_type: &str, year: u32, number: &str) -> String {
        format!(
            "http://data.europa.eu/eli/{}/{}/{}/{}",
            jurisdiction.to_lowercase(),
            doc_type.to_lowercase(),
            year,
            number.to_lowercase()
        )
    }

    /// Convert a single obligation into a JSON-LD node.
    pub fn obligation_to_node(obligation: &Obligation, base_uri: &str) -> JsonLdNode {
        let node_id = format!("{}/obligation/{}", base_uri, obligation.id.0);
        let mut node = JsonLdNode::new(
            node_id,
            vec![
                "schema:Legislation".into(),
                "regsynth:Obligation".into(),
            ],
        );

        node.set_str("schema:name", &obligation.description);
        node.set_str(
            "legislationIdentifier",
            format!("{} Art. {}", obligation.article_ref.framework, obligation.article_ref.article_number),
        );
        node.set_str(
            "legislationJurisdiction",
            &obligation.jurisdiction.0,
        );
        node.set_str("obligationType", format!("{:?}", obligation.obligation_type));
        node.set_str("regulatoryDomain", format!("{:?}", obligation.domain));
        node.set_str("riskLevel", format!("{:?}", obligation.risk_level));

        node.set_typed(
            "schema:dateCreated",
            obligation.temporal_interval.start.0.format("%Y-%m-%d").to_string(),
            "xsd:date",
        );
        node.set_typed(
            "schema:expires",
            obligation.temporal_interval.end.0.format("%Y-%m-%d").to_string(),
            "xsd:date",
        );

        if let Some(penalty) = obligation.penalty_amount {
            node.set_num("complianceCost", penalty);
        }

        // Cross-references as links
        for xref in &obligation.cross_references {
            node.set_ref("schema:isRelatedTo", format!("{}/obligation/{}", base_uri, xref.0));
        }

        node
    }

    /// Convert a regulatory framework into a JSON-LD node.
    pub fn framework_to_node(framework: &RegulatoryFramework, base_uri: &str) -> JsonLdNode {
        let node_id = format!("{}/framework/{}", base_uri, framework.id);
        let mut node = JsonLdNode::new(
            node_id,
            vec![
                "schema:Legislation".into(),
                "eli:LegalResource".into(),
            ],
        );

        node.set_str("schema:name", &framework.name);
        node.set_str("schema:version", &framework.version);
        node.set_str("legislationJurisdiction", &framework.jurisdiction.0);
        node.set_str("dcterms:type", format!("{:?}", framework.framework_type));

        if let Some(ref url) = framework.url {
            node.set_str("schema:url", url);
        }
        if let Some(ref date) = framework.effective_date {
            node.set_typed("legislationDate", date, "xsd:date");
        }

        node
    }

    /// Build a complete linked-data document from obligations.
    pub fn from_obligations(
        obligations: &[Obligation],
        base_uri: &str,
    ) -> Self {
        let mut doc = Self::new();
        for obl in obligations {
            doc.add_node(Self::obligation_to_node(obl, base_uri));
        }
        doc
    }

    /// Serialize to a JSON-LD string.
    pub fn to_json(&self) -> FormatResult<String> {
        serde_json::to_string_pretty(self).map_err(crate::FormatError::Serialization)
    }

    /// Produce a compact JSON value (for embedding in other documents).
    pub fn to_value(&self) -> FormatResult<JsonValue> {
        serde_json::to_value(self).map_err(crate::FormatError::Serialization)
    }
}

impl Default for RegulatoryLinkedData {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_context_has_schema_org() {
        let ctx = JsonLdContext::regulatory_default();
        assert!(ctx.prefixes.contains_key("schema"));
        assert!(ctx.prefixes.contains_key("eli"));
        assert!(ctx.prefixes.contains_key("regsynth"));
    }

    #[test]
    fn eli_uri_format() {
        let uri = RegulatoryLinkedData::eli_uri("EU", "reg", 2024, "1689");
        assert_eq!(uri, "http://data.europa.eu/eli/eu/reg/2024/1689");
    }

    #[test]
    fn node_properties() {
        let mut node = JsonLdNode::new(
            "https://example.org/test",
            vec!["schema:Legislation".into()],
        );
        node.set_str("schema:name", "Test Regulation");
        node.set_num("complianceCost", 42.0);
        node.set_ref("schema:isRelatedTo", "https://example.org/other");

        let json = serde_json::to_value(&node).unwrap();
        assert_eq!(json["@id"], "https://example.org/test");
        assert_eq!(json["schema:name"], "Test Regulation");
    }

    #[test]
    fn empty_document_json() {
        let doc = RegulatoryLinkedData::new();
        let json_str = doc.to_json().unwrap();
        assert!(json_str.contains("@context"));
        assert!(json_str.contains("@graph"));
        assert!(json_str.contains("schema.org"));
    }

    #[test]
    fn document_with_custom_context() {
        let ctx = JsonLdContext::from_prefixes(vec![
            ("ex".into(), "https://example.org/".into()),
        ]);
        let doc = RegulatoryLinkedData {
            context: ctx,
            graph: vec![],
        };
        let json = doc.to_json().unwrap();
        assert!(json.contains("example.org"));
    }
}
