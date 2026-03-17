//! OWL/RDF ontology support for regulatory knowledge graphs.
//!
//! Provides types for building RDF graphs and OWL ontologies that model
//! regulatory concepts such as Regulation, Obligation, Jurisdiction,
//! RiskLevel, and ComplianceStrategy.
//!
//! Supports serialization to Turtle and N-Triples formats using standard
//! vocabularies: `rdf:`, `rdfs:`, `owl:`, `dcterms:`, `xsd:`.
//!
//! Reference: <https://www.w3.org/TR/owl2-overview/>

use serde::{Deserialize, Serialize};
use std::fmt;

use regsynth_types::Obligation;
use regsynth_types::obligation::RiskLevel;
use regsynth_types::regulatory::RegulatoryFramework;

// ── Standard namespace prefixes ───────────────────────────────────────

/// Well-known namespace IRIs.
pub mod ns {
    pub const RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
    pub const RDFS: &str = "http://www.w3.org/2000/01/rdf-schema#";
    pub const OWL: &str = "http://www.w3.org/2002/07/owl#";
    pub const XSD: &str = "http://www.w3.org/2001/XMLSchema#";
    pub const DCTERMS: &str = "http://purl.org/dc/terms/";
    pub const SKOS: &str = "http://www.w3.org/2004/02/skos/core#";
    pub const REGSYNTH: &str = "https://regsynth.dev/ontology/";
}

// ── RDF term ──────────────────────────────────────────────────────────

/// An RDF term: IRI, blank node, or literal.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RdfTerm {
    /// A full IRI reference.
    Iri(String),
    /// A prefixed name (e.g. `"rdfs:Class"`).
    Prefixed(String),
    /// A blank node identifier.
    BlankNode(String),
    /// A plain string literal.
    Literal(String),
    /// A typed literal with datatype IRI.
    TypedLiteral { value: String, datatype: String },
    /// A language-tagged literal.
    LangLiteral { value: String, lang: String },
}

impl RdfTerm {
    /// Convenience constructor for an IRI.
    pub fn iri(s: impl Into<String>) -> Self {
        Self::Iri(s.into())
    }

    /// Convenience constructor for a prefixed name.
    pub fn prefixed(s: impl Into<String>) -> Self {
        Self::Prefixed(s.into())
    }

    /// Convenience constructor for a plain literal.
    pub fn literal(s: impl Into<String>) -> Self {
        Self::Literal(s.into())
    }
}

impl fmt::Display for RdfTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Iri(iri) => write!(f, "<{}>", iri),
            Self::Prefixed(p) => write!(f, "{}", p),
            Self::BlankNode(id) => write!(f, "_:{}", id),
            Self::Literal(s) => write!(f, "\"{}\"", escape_turtle(s)),
            Self::TypedLiteral { value, datatype } => {
                write!(f, "\"{}\"^^<{}>", escape_turtle(value), datatype)
            }
            Self::LangLiteral { value, lang } => {
                write!(f, "\"{}\"@{}", escape_turtle(value), lang)
            }
        }
    }
}

/// Escape special characters for Turtle string literals.
fn escape_turtle(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ── RDF triple ────────────────────────────────────────────────────────

/// A single RDF triple (subject, predicate, object).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RdfTriple {
    pub subject: RdfTerm,
    pub predicate: RdfTerm,
    pub object: RdfTerm,
}

impl RdfTriple {
    pub fn new(subject: RdfTerm, predicate: RdfTerm, object: RdfTerm) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Format as an N-Triples line (terminated by `.`).
    pub fn to_ntriples(&self) -> String {
        format!("{} {} {} .", self.subject, self.predicate, self.object)
    }
}

// ── RDF graph ─────────────────────────────────────────────────────────

/// An in-memory RDF graph (set of triples with prefix declarations).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfGraph {
    /// Prefix declarations for Turtle serialization.
    pub prefixes: Vec<(String, String)>,
    /// The triples in this graph.
    pub triples: Vec<RdfTriple>,
}

impl RdfGraph {
    /// Create a graph with the standard regulatory ontology prefixes.
    pub fn with_standard_prefixes() -> Self {
        Self {
            prefixes: vec![
                ("rdf".into(), ns::RDF.into()),
                ("rdfs".into(), ns::RDFS.into()),
                ("owl".into(), ns::OWL.into()),
                ("xsd".into(), ns::XSD.into()),
                ("dcterms".into(), ns::DCTERMS.into()),
                ("skos".into(), ns::SKOS.into()),
                ("reg".into(), ns::REGSYNTH.into()),
            ],
            triples: Vec::new(),
        }
    }

    /// Add a triple to the graph.
    pub fn add(&mut self, triple: RdfTriple) {
        self.triples.push(triple);
    }

    /// Helper: add a triple from prefixed-name strings.
    pub fn add_prefixed(&mut self, subject: &str, predicate: &str, object: &str) {
        self.add(RdfTriple::new(
            RdfTerm::prefixed(subject),
            RdfTerm::prefixed(predicate),
            RdfTerm::prefixed(object),
        ));
    }

    /// Helper: add a triple with a literal object.
    pub fn add_literal(&mut self, subject: &str, predicate: &str, value: &str) {
        self.add(RdfTriple::new(
            RdfTerm::prefixed(subject),
            RdfTerm::prefixed(predicate),
            RdfTerm::literal(value),
        ));
    }

    /// Number of triples.
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Whether the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Serialize to Turtle format.
    pub fn to_turtle(&self) -> String {
        let mut out = String::new();
        // Prefix declarations
        for (prefix, iri) in &self.prefixes {
            out.push_str(&format!("@prefix {}: <{}> .\n", prefix, iri));
        }
        if !self.prefixes.is_empty() {
            out.push('\n');
        }
        // Group triples by subject for readability
        let mut by_subject: Vec<&RdfTriple> = self.triples.iter().collect();
        by_subject.sort_by(|a, b| format!("{}", a.subject).cmp(&format!("{}", b.subject)));

        let mut current_subject: Option<String> = None;
        for triple in &by_subject {
            let subj_str = format!("{}", triple.subject);
            if current_subject.as_deref() == Some(&subj_str) {
                out.push_str(&format!(
                    "    {} {} ;\n",
                    triple.predicate, triple.object
                ));
            } else {
                if current_subject.is_some() {
                    // Close previous subject block
                    out.push_str(".\n\n");
                }
                current_subject = Some(subj_str.clone());
                out.push_str(&format!(
                    "{}\n    {} {} ;\n",
                    triple.subject, triple.predicate, triple.object
                ));
            }
        }
        if current_subject.is_some() {
            out.push_str(".\n");
        }
        out
    }

    /// Serialize to N-Triples format.
    pub fn to_ntriples(&self) -> String {
        self.triples
            .iter()
            .map(|t| t.to_ntriples())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ── OWL ontology builder ──────────────────────────────────────────────

/// An OWL ontology for regulatory concepts, built on an [`RdfGraph`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwlOntology {
    /// Ontology IRI.
    pub ontology_iri: String,
    /// Version IRI (optional).
    pub version_iri: Option<String>,
    /// Underlying RDF graph.
    pub graph: RdfGraph,
}

impl OwlOntology {
    /// Build the standard RegSynth regulatory ontology.
    ///
    /// Defines OWL classes and properties for regulatory modelling:
    ///
    /// **Classes**: `Regulation`, `Obligation`, `Jurisdiction`, `RiskLevel`,
    /// `ComplianceStrategy`, `RegulatoryDomain`
    ///
    /// **Object properties**: `hasJurisdiction`, `imposesObligation`,
    /// `conflictsWith`, `supersedes`, `hasDomain`, `hasRiskLevel`
    ///
    /// **Data properties**: `penaltyAmount`, `description`,
    /// `effectiveDate`, `expirationDate`
    pub fn regulatory_ontology() -> Self {
        let mut graph = RdfGraph::with_standard_prefixes();
        let ont_iri = ns::REGSYNTH.trim_end_matches('/');

        // Ontology declaration
        graph.add(RdfTriple::new(
            RdfTerm::iri(ont_iri),
            RdfTerm::prefixed("rdf:type"),
            RdfTerm::prefixed("owl:Ontology"),
        ));
        graph.add_literal(
            &format!("<{}>", ont_iri),
            "dcterms:title",
            "RegSynth Regulatory Ontology",
        );

        // ── OWL Classes ───────────────────────────────────────────
        let classes = [
            ("reg:Regulation", "A regulatory framework or legislative act."),
            ("reg:Obligation", "A specific regulatory obligation, permission, or prohibition."),
            ("reg:Jurisdiction", "A legal jurisdiction (country, state, supranational body)."),
            ("reg:RiskLevel", "Risk classification level for AI systems."),
            ("reg:ComplianceStrategy", "A strategy assigning compliance choices to obligations."),
            ("reg:RegulatoryDomain", "A domain of regulatory concern (e.g. transparency, data governance)."),
        ];
        for (class, comment) in &classes {
            graph.add_prefixed(class, "rdf:type", "owl:Class");
            graph.add_literal(class, "rdfs:comment", comment);
        }

        // Risk level individuals
        let risk_levels = [
            ("reg:Minimal", "Minimal risk"),
            ("reg:Limited", "Limited risk"),
            ("reg:High", "High risk"),
            ("reg:Unacceptable", "Unacceptable risk"),
        ];
        for (individual, label) in &risk_levels {
            graph.add_prefixed(individual, "rdf:type", "reg:RiskLevel");
            graph.add_literal(individual, "rdfs:label", label);
        }

        // Regulatory domain individuals
        let domains = [
            "DataGovernance",
            "Transparency",
            "RiskClassification",
            "HumanOversight",
            "Documentation",
            "PostMarketSurveillance",
            "CrossBorderDataTransfer",
            "AlgorithmicAccountability",
            "BiasAndFairness",
            "SecurityAndRobustness",
            "IntellectualProperty",
            "ConsentAndNotice",
        ];
        for d in &domains {
            let indiv = format!("reg:{}", d);
            graph.add_prefixed(&indiv, "rdf:type", "reg:RegulatoryDomain");
            graph.add_literal(&indiv, "rdfs:label", d);
        }

        // ── Object Properties ─────────────────────────────────────
        let object_props: &[(&str, &str, &str, &str)] = &[
            (
                "reg:hasJurisdiction",
                "Relates an obligation or regulation to its jurisdiction.",
                "reg:Obligation",
                "reg:Jurisdiction",
            ),
            (
                "reg:imposesObligation",
                "A regulation imposes an obligation.",
                "reg:Regulation",
                "reg:Obligation",
            ),
            (
                "reg:conflictsWith",
                "Two obligations are in conflict.",
                "reg:Obligation",
                "reg:Obligation",
            ),
            (
                "reg:supersedes",
                "One regulation supersedes another.",
                "reg:Regulation",
                "reg:Regulation",
            ),
            (
                "reg:hasDomain",
                "Obligation belongs to a regulatory domain.",
                "reg:Obligation",
                "reg:RegulatoryDomain",
            ),
            (
                "reg:hasRiskLevel",
                "Obligation is associated with a risk level.",
                "reg:Obligation",
                "reg:RiskLevel",
            ),
        ];
        for (prop, comment, domain, range) in object_props {
            graph.add_prefixed(prop, "rdf:type", "owl:ObjectProperty");
            graph.add_literal(prop, "rdfs:comment", comment);
            graph.add_prefixed(prop, "rdfs:domain", domain);
            graph.add_prefixed(prop, "rdfs:range", range);
        }

        // ── Data Properties ───────────────────────────────────────
        let data_props: &[(&str, &str, &str)] = &[
            ("reg:penaltyAmount", "Monetary penalty amount.", "xsd:decimal"),
            ("reg:description", "Human-readable description.", "xsd:string"),
            ("reg:effectiveDate", "Date when the obligation takes effect.", "xsd:date"),
            ("reg:expirationDate", "Date when the obligation expires.", "xsd:date"),
        ];
        for &(prop, comment, range_dt) in data_props {
            graph.add_prefixed(prop, "rdf:type", "owl:DatatypeProperty");
            graph.add_literal(prop, "rdfs:comment", comment);
            graph.add(RdfTriple::new(
                RdfTerm::prefixed(prop),
                RdfTerm::prefixed("rdfs:range"),
                RdfTerm::prefixed(range_dt),
            ));
        }

        Self {
            ontology_iri: ont_iri.to_string(),
            version_iri: Some(format!("{}/1.0", ont_iri)),
            graph,
        }
    }

    /// Add obligation instances to the ontology graph.
    pub fn add_obligations(&mut self, obligations: &[Obligation], base_uri: &str) {
        for obl in obligations {
            let obl_node = format!("<{}/obligation/{}>", base_uri, obl.id.0);

            self.graph.add_prefixed(&obl_node, "rdf:type", "reg:Obligation");
            self.graph.add_literal(&obl_node, "reg:description", &obl.description);

            // Jurisdiction
            let juris_node = format!("reg:{}", obl.jurisdiction.0);
            self.graph.add_prefixed(&juris_node, "rdf:type", "reg:Jurisdiction");
            self.graph.add_prefixed(&obl_node, "reg:hasJurisdiction", &juris_node);

            // Risk level
            let risk_node = match obl.risk_level {
                RiskLevel::Unacceptable => "reg:Unacceptable",
                RiskLevel::High => "reg:High",
                RiskLevel::Limited => "reg:Limited",
                RiskLevel::Minimal | RiskLevel::Unknown => "reg:Minimal",
            };
            self.graph.add_prefixed(&obl_node, "reg:hasRiskLevel", risk_node);

            // Domain
            let domain_node = format!("reg:{:?}", obl.domain);
            self.graph.add_prefixed(&obl_node, "reg:hasDomain", &domain_node);

            // Temporal data properties
            self.graph.add(RdfTriple::new(
                RdfTerm::prefixed(&obl_node),
                RdfTerm::prefixed("reg:effectiveDate"),
                RdfTerm::TypedLiteral {
                    value: obl.temporal_interval.start.0.format("%Y-%m-%d").to_string(),
                    datatype: format!("{}date", ns::XSD),
                },
            ));

            if let Some(penalty) = obl.penalty_amount {
                self.graph.add(RdfTriple::new(
                    RdfTerm::prefixed(&obl_node),
                    RdfTerm::prefixed("reg:penaltyAmount"),
                    RdfTerm::TypedLiteral {
                        value: penalty.to_string(),
                        datatype: format!("{}decimal", ns::XSD),
                    },
                ));
            }
        }
    }

    /// Add a framework instance to the ontology graph.
    pub fn add_framework(&mut self, framework: &RegulatoryFramework, base_uri: &str) {
        let fw_node = format!("<{}/framework/{}>", base_uri, framework.id);

        self.graph.add_prefixed(&fw_node, "rdf:type", "reg:Regulation");
        self.graph.add_literal(&fw_node, "reg:description", &framework.name);

        let juris_node = format!("reg:{}", framework.jurisdiction.0);
        self.graph.add_prefixed(&juris_node, "rdf:type", "reg:Jurisdiction");
        self.graph.add_prefixed(&fw_node, "reg:hasJurisdiction", &juris_node);
    }

    /// Serialize the ontology in Turtle format.
    pub fn to_turtle(&self) -> String {
        self.graph.to_turtle()
    }

    /// Serialize the ontology in N-Triples format.
    pub fn to_ntriples(&self) -> String {
        self.graph.to_ntriples()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rdf_term_display() {
        assert_eq!(format!("{}", RdfTerm::iri("http://ex.org/a")), "<http://ex.org/a>");
        assert_eq!(format!("{}", RdfTerm::prefixed("rdf:type")), "rdf:type");
        assert_eq!(format!("{}", RdfTerm::literal("hello")), "\"hello\"");
    }

    #[test]
    fn triple_ntriples() {
        let t = RdfTriple::new(
            RdfTerm::iri("http://ex.org/s"),
            RdfTerm::iri("http://ex.org/p"),
            RdfTerm::literal("value"),
        );
        let nt = t.to_ntriples();
        assert!(nt.ends_with('.'));
        assert!(nt.contains("<http://ex.org/s>"));
    }

    #[test]
    fn graph_turtle_output() {
        let mut g = RdfGraph::with_standard_prefixes();
        g.add_prefixed("reg:MyObligation", "rdf:type", "reg:Obligation");
        g.add_literal("reg:MyObligation", "reg:description", "Test obligation");
        let turtle = g.to_turtle();
        assert!(turtle.contains("@prefix rdf:"));
        assert!(turtle.contains("reg:Obligation"));
    }

    #[test]
    fn regulatory_ontology_has_classes() {
        let ont = OwlOntology::regulatory_ontology();
        let turtle = ont.to_turtle();
        assert!(turtle.contains("reg:Regulation"));
        assert!(turtle.contains("reg:Obligation"));
        assert!(turtle.contains("reg:Jurisdiction"));
        assert!(turtle.contains("owl:Class"));
        assert!(turtle.contains("owl:ObjectProperty"));
        assert!(turtle.contains("reg:hasJurisdiction"));
        assert!(turtle.contains("reg:imposesObligation"));
        assert!(turtle.contains("reg:conflictsWith"));
        assert!(turtle.contains("reg:supersedes"));
    }

    #[test]
    fn ntriples_output() {
        let mut g = RdfGraph::with_standard_prefixes();
        g.add_prefixed("reg:X", "rdf:type", "owl:Class");
        let nt = g.to_ntriples();
        assert!(nt.contains("reg:X rdf:type owl:Class ."));
    }

    #[test]
    fn ontology_iri_set() {
        let ont = OwlOntology::regulatory_ontology();
        assert!(ont.ontology_iri.contains("regsynth.dev"));
        assert!(ont.version_iri.is_some());
    }

    #[test]
    fn escape_special_chars() {
        let term = RdfTerm::literal("line1\nline2\twith\"quotes\"");
        let s = format!("{}", term);
        assert!(s.contains("\\n"));
        assert!(s.contains("\\t"));
        assert!(s.contains("\\\""));
    }
}
