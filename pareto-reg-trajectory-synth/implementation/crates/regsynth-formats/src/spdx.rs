//! SPDX 2.3 (Software Package Data Exchange) format support.
//!
//! Provides types and conversions for exporting regulatory obligations
//! as SPDX-compatible license compliance data, mapping RegSynth obligation
//! types to SPDX license identifiers and relationship descriptors.
//!
//! Reference: <https://spdx.github.io/spdx-spec/v2.3/>

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fmt;

use regsynth_types::{Obligation, RegulatoryDomain};

use crate::FormatResult;

// ── License expression AST ────────────────────────────────────────────

/// SPDX license expression with AND / OR / WITH operators.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpdxLicenseExpression {
    /// A single SPDX license identifier (e.g. `"MIT"`, `"Apache-2.0"`).
    Identifier(String),
    /// Conjunction: both sub-expressions apply.
    And(Box<SpdxLicenseExpression>, Box<SpdxLicenseExpression>),
    /// Disjunction: either sub-expression may be chosen.
    Or(Box<SpdxLicenseExpression>, Box<SpdxLicenseExpression>),
    /// License with exception (e.g. `GPL-2.0 WITH Classpath-exception-2.0`).
    With(Box<SpdxLicenseExpression>, String),
}

impl SpdxLicenseExpression {
    /// Create a simple identifier expression.
    pub fn id(name: impl Into<String>) -> Self {
        Self::Identifier(name.into())
    }

    /// Combine two expressions with AND.
    pub fn and(self, other: Self) -> Self {
        Self::And(Box::new(self), Box::new(other))
    }

    /// Combine two expressions with OR.
    pub fn or(self, other: Self) -> Self {
        Self::Or(Box::new(self), Box::new(other))
    }

    /// Attach a WITH exception clause.
    pub fn with_exception(self, exception: impl Into<String>) -> Self {
        Self::With(Box::new(self), exception.into())
    }

    /// Parse a simple SPDX expression string.
    ///
    /// Supports `AND`, `OR`, and `WITH` operators with left-to-right precedence.
    pub fn parse(input: &str) -> FormatResult<Self> {
        let tokens: Vec<&str> = input.split_whitespace().collect();
        if tokens.is_empty() {
            return Err(crate::FormatError::Spdx("empty license expression".into()));
        }
        Self::parse_tokens(&tokens)
    }

    fn parse_tokens(tokens: &[&str]) -> FormatResult<Self> {
        if tokens.len() == 1 {
            return Ok(Self::Identifier(tokens[0].to_string()));
        }
        // Scan for OR (lowest precedence), then AND, then WITH
        for (i, &tok) in tokens.iter().enumerate().rev() {
            if tok.eq_ignore_ascii_case("OR") && i > 0 && i < tokens.len() - 1 {
                let left = Self::parse_tokens(&tokens[..i])?;
                let right = Self::parse_tokens(&tokens[i + 1..])?;
                return Ok(Self::Or(Box::new(left), Box::new(right)));
            }
        }
        for (i, &tok) in tokens.iter().enumerate().rev() {
            if tok.eq_ignore_ascii_case("AND") && i > 0 && i < tokens.len() - 1 {
                let left = Self::parse_tokens(&tokens[..i])?;
                let right = Self::parse_tokens(&tokens[i + 1..])?;
                return Ok(Self::And(Box::new(left), Box::new(right)));
            }
        }
        for (i, &tok) in tokens.iter().enumerate() {
            if tok.eq_ignore_ascii_case("WITH") && i > 0 && i < tokens.len() - 1 {
                let base = Self::parse_tokens(&tokens[..i])?;
                let exception = tokens[i + 1..].join(" ");
                return Ok(Self::With(Box::new(base), exception));
            }
        }
        Err(crate::FormatError::Spdx(format!(
            "cannot parse license expression: {}",
            tokens.join(" ")
        )))
    }
}

impl fmt::Display for SpdxLicenseExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identifier(id) => write!(f, "{}", id),
            Self::And(l, r) => write!(f, "({} AND {})", l, r),
            Self::Or(l, r) => write!(f, "({} OR {})", l, r),
            Self::With(base, exc) => write!(f, "({} WITH {})", base, exc),
        }
    }
}

// ── SPDX relationship ─────────────────────────────────────────────────

/// SPDX 2.3 relationship type between elements.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpdxRelationshipType {
    DescribedBy,
    ContainedBy,
    DependsOn,
    GeneratedFrom,
    AncestorOf,
    VariantOf,
    Other(String),
}

/// A relationship between two SPDX elements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpdxRelationship {
    pub spdx_element_id: String,
    pub relationship_type: SpdxRelationshipType,
    pub related_spdx_element: String,
    pub comment: Option<String>,
}

// ── SPDX package ──────────────────────────────────────────────────────

/// An SPDX package representing a regulatory compliance artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpdxPackage {
    /// Unique SPDX identifier for the package (e.g. `SPDXRef-Package-1`).
    pub spdx_id: String,
    /// Human-readable package name.
    pub name: String,
    /// Package version string.
    pub version: String,
    /// The concluded license expression.
    pub license_concluded: SpdxLicenseExpression,
    /// License declared by the originator.
    pub license_declared: SpdxLicenseExpression,
    /// Copyright text.
    pub copyright_text: String,
    /// Package download location URI.
    pub download_location: String,
    /// Package supplier.
    pub supplier: Option<String>,
    /// Package originator.
    pub originator: Option<String>,
    /// Short description.
    pub description: Option<String>,
    /// External references (e.g. CPE, purl).
    pub external_refs: Vec<SpdxExternalRef>,
    /// Checksums for verification.
    pub checksums: Vec<SpdxChecksum>,
    /// Source regulatory domain, if derived from an obligation.
    pub regulatory_domain: Option<String>,
    /// Risk level tag.
    pub risk_level: Option<String>,
}

/// External reference for an SPDX package.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpdxExternalRef {
    pub reference_category: String,
    pub reference_type: String,
    pub reference_locator: String,
}

/// Checksum entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpdxChecksum {
    pub algorithm: String,
    pub value: String,
}

// ── SPDX document ─────────────────────────────────────────────────────

/// Top-level SPDX 2.3 document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpdxDocument {
    /// Always `"SPDX-2.3"`.
    pub spdx_version: String,
    /// Data license (always `"CC0-1.0"` per spec).
    pub data_license: String,
    /// Document SPDX identifier.
    pub spdx_id: String,
    /// Human-readable document name.
    pub document_name: String,
    /// Document namespace URI.
    pub document_namespace: String,
    /// Creation info.
    pub creation_info: SpdxCreationInfo,
    /// Packages listed in this document.
    pub packages: Vec<SpdxPackage>,
    /// Relationships between elements.
    pub relationships: Vec<SpdxRelationship>,
    /// Free-form annotations.
    pub annotations: Vec<SpdxAnnotation>,
}

/// SPDX creation information block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpdxCreationInfo {
    pub created: String,
    pub creators: Vec<String>,
    pub license_list_version: Option<String>,
    pub comment: Option<String>,
}

/// SPDX annotation entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpdxAnnotation {
    pub annotator: String,
    pub annotation_date: String,
    pub annotation_type: String,
    pub comment: String,
}

impl SpdxDocument {
    /// Create a new empty SPDX 2.3 document.
    pub fn new(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            spdx_version: "SPDX-2.3".into(),
            data_license: "CC0-1.0".into(),
            spdx_id: "SPDXRef-DOCUMENT".into(),
            document_name: name.into(),
            document_namespace: namespace.into(),
            creation_info: SpdxCreationInfo {
                created: Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
                creators: vec!["Tool: RegSynth-Formats-0.1.0".into()],
                license_list_version: Some("3.22".into()),
                comment: None,
            },
            packages: Vec::new(),
            relationships: Vec::new(),
            annotations: Vec::new(),
        }
    }

    /// Add a package to the document and create a DESCRIBES relationship.
    pub fn add_package(&mut self, pkg: SpdxPackage) {
        let pkg_id = pkg.spdx_id.clone();
        self.packages.push(pkg);
        self.relationships.push(SpdxRelationship {
            spdx_element_id: self.spdx_id.clone(),
            relationship_type: SpdxRelationshipType::DescribedBy,
            related_spdx_element: pkg_id,
            comment: None,
        });
    }

    /// Serialize the document to JSON.
    pub fn to_json(&self) -> FormatResult<String> {
        serde_json::to_string_pretty(self).map_err(crate::FormatError::Serialization)
    }

    /// Convert a set of RegSynth obligations into an SPDX document.
    ///
    /// Each obligation becomes a package with its domain and risk level
    /// mapped to SPDX-compatible license identifiers.
    pub fn from_obligations(
        obligations: &[Obligation],
        doc_name: &str,
        namespace: &str,
    ) -> Self {
        let mut doc = Self::new(doc_name, namespace);

        for (idx, obl) in obligations.iter().enumerate() {
            let license_id = domain_to_spdx_license(&obl.domain);
            let pkg = SpdxPackage {
                spdx_id: format!("SPDXRef-Obligation-{}", idx),
                name: format!("obligation-{}", obl.id.0),
                version: "1.0.0".into(),
                license_concluded: SpdxLicenseExpression::id(&license_id),
                license_declared: SpdxLicenseExpression::id(&license_id),
                copyright_text: "NOASSERTION".into(),
                download_location: "NOASSERTION".into(),
                supplier: Some(format!("Organization: {}", obl.article_ref.framework)),
                originator: None,
                description: Some(obl.description.clone()),
                external_refs: Vec::new(),
                checksums: Vec::new(),
                regulatory_domain: Some(format!("{:?}", obl.domain)),
                risk_level: Some(format!("{:?}", obl.risk_level)),
            };
            doc.add_package(pkg);
        }

        doc
    }
}

/// Map a regulatory domain to a representative SPDX license identifier.
///
/// These are synthetic identifiers prefixed with `LicenseRef-RegSynth-`
/// since regulatory obligations are not traditional software licenses.
fn domain_to_spdx_license(domain: &RegulatoryDomain) -> String {
    match domain {
        RegulatoryDomain::DataGovernance => "LicenseRef-RegSynth-DataGov".into(),
        RegulatoryDomain::Transparency => "LicenseRef-RegSynth-Transparency".into(),
        RegulatoryDomain::RiskClassification => "LicenseRef-RegSynth-RiskClass".into(),
        RegulatoryDomain::HumanOversight => "LicenseRef-RegSynth-HumanOversight".into(),
        RegulatoryDomain::Documentation => "LicenseRef-RegSynth-Documentation".into(),
        RegulatoryDomain::PostMarketSurveillance => "LicenseRef-RegSynth-PostMarket".into(),
        RegulatoryDomain::CrossBorderDataTransfer => "LicenseRef-RegSynth-CrossBorder".into(),
        RegulatoryDomain::AlgorithmicAccountability => "LicenseRef-RegSynth-AlgoAccount".into(),
        RegulatoryDomain::BiasAndFairness => "LicenseRef-RegSynth-BiasFairness".into(),
        RegulatoryDomain::SecurityAndRobustness => "LicenseRef-RegSynth-SecRobust".into(),
        RegulatoryDomain::IntellectualProperty => "LicenseRef-RegSynth-IP".into(),
        RegulatoryDomain::ConsentAndNotice => "LicenseRef-RegSynth-Consent".into(),
        RegulatoryDomain::General => "LicenseRef-RegSynth-General".into(),
    }
}

/// Map an SPDX `LicenseRef-RegSynth-*` identifier back to a regulatory domain.
pub fn spdx_license_to_domain(license_id: &str) -> Option<RegulatoryDomain> {
    match license_id {
        "LicenseRef-RegSynth-DataGov" => Some(RegulatoryDomain::DataGovernance),
        "LicenseRef-RegSynth-Transparency" => Some(RegulatoryDomain::Transparency),
        "LicenseRef-RegSynth-RiskClass" => Some(RegulatoryDomain::RiskClassification),
        "LicenseRef-RegSynth-HumanOversight" => Some(RegulatoryDomain::HumanOversight),
        "LicenseRef-RegSynth-Documentation" => Some(RegulatoryDomain::Documentation),
        "LicenseRef-RegSynth-PostMarket" => Some(RegulatoryDomain::PostMarketSurveillance),
        "LicenseRef-RegSynth-CrossBorder" => Some(RegulatoryDomain::CrossBorderDataTransfer),
        "LicenseRef-RegSynth-AlgoAccount" => Some(RegulatoryDomain::AlgorithmicAccountability),
        "LicenseRef-RegSynth-BiasFairness" => Some(RegulatoryDomain::BiasAndFairness),
        "LicenseRef-RegSynth-SecRobust" => Some(RegulatoryDomain::SecurityAndRobustness),
        "LicenseRef-RegSynth-IP" => Some(RegulatoryDomain::IntellectualProperty),
        "LicenseRef-RegSynth-Consent" => Some(RegulatoryDomain::ConsentAndNotice),
        "LicenseRef-RegSynth-General" => Some(RegulatoryDomain::General),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_identifier() {
        let expr = SpdxLicenseExpression::parse("MIT").unwrap();
        assert_eq!(expr, SpdxLicenseExpression::Identifier("MIT".into()));
    }

    #[test]
    fn parse_and_expression() {
        let expr = SpdxLicenseExpression::parse("MIT AND Apache-2.0").unwrap();
        assert_eq!(
            expr,
            SpdxLicenseExpression::And(
                Box::new(SpdxLicenseExpression::Identifier("MIT".into())),
                Box::new(SpdxLicenseExpression::Identifier("Apache-2.0".into()))
            )
        );
    }

    #[test]
    fn parse_or_expression() {
        let expr = SpdxLicenseExpression::parse("GPL-2.0 OR MIT").unwrap();
        assert_eq!(
            expr,
            SpdxLicenseExpression::Or(
                Box::new(SpdxLicenseExpression::Identifier("GPL-2.0".into())),
                Box::new(SpdxLicenseExpression::Identifier("MIT".into()))
            )
        );
    }

    #[test]
    fn parse_with_exception() {
        let expr =
            SpdxLicenseExpression::parse("GPL-2.0 WITH Classpath-exception-2.0").unwrap();
        assert_eq!(
            expr,
            SpdxLicenseExpression::With(
                Box::new(SpdxLicenseExpression::Identifier("GPL-2.0".into())),
                "Classpath-exception-2.0".into()
            )
        );
    }

    #[test]
    fn display_roundtrip() {
        let expr = SpdxLicenseExpression::id("MIT")
            .and(SpdxLicenseExpression::id("Apache-2.0"))
            .or(SpdxLicenseExpression::id("BSD-3-Clause"));
        let s = expr.to_string();
        assert!(s.contains("AND"));
        assert!(s.contains("OR"));
    }

    #[test]
    fn spdx_document_json() {
        let mut doc = SpdxDocument::new("test-doc", "https://example.org/test");
        let pkg = SpdxPackage {
            spdx_id: "SPDXRef-Pkg-1".into(),
            name: "test-package".into(),
            version: "1.0.0".into(),
            license_concluded: SpdxLicenseExpression::id("MIT"),
            license_declared: SpdxLicenseExpression::id("MIT"),
            copyright_text: "NOASSERTION".into(),
            download_location: "NOASSERTION".into(),
            supplier: None,
            originator: None,
            description: None,
            external_refs: Vec::new(),
            checksums: Vec::new(),
            regulatory_domain: None,
            risk_level: None,
        };
        doc.add_package(pkg);
        let json = doc.to_json().unwrap();
        assert!(json.contains("SPDX-2.3"));
        assert!(json.contains("CC0-1.0"));
        assert!(json.contains("test-package"));
    }

    #[test]
    fn domain_license_roundtrip() {
        let domain = RegulatoryDomain::Transparency;
        let lic = domain_to_spdx_license(&domain);
        let back = spdx_license_to_domain(&lic);
        assert_eq!(back, Some(RegulatoryDomain::Transparency));
    }
}
