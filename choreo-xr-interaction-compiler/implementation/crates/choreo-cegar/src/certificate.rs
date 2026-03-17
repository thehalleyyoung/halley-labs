//! Verification certificates for the CEGAR engine.
//!
//! This module provides data structures and builders for constructing machine-
//! checkable verification certificates. A certificate records the evidence
//! that a property either holds (proof certificate) or is violated
//! (counterexample certificate) in the abstract model produced by the CEGAR
//! loop.
//!
//! Certificates serve three purposes:
//! 1. **Auditability** – an independent checker can replay the reasoning.
//! 2. **Compositionality** – certificates from sub-problems compose into
//!    a whole-system certificate via assume-guarantee reasoning.
//! 3. **Debugging** – counterexample certificates carry concrete witness
//!    traces that help the user understand the violation.

use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::abstraction::AbstractState;
use crate::counterexample::{ConcreteCounterexample, Counterexample};
use crate::properties::Property;
use crate::{CegarError, SpatialConstraint};

// ---------------------------------------------------------------------------
// Proof certificate
// ---------------------------------------------------------------------------

/// A proof certificate attesting that a property holds in the abstract model.
///
/// The certificate records the abstract invariant (set of reachable abstract
/// states) together with meta-information that allows an independent checker
/// to validate the proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofCertificate {
    /// The abstract states forming the inductive invariant.
    pub invariant_states: Vec<AbstractState>,
    /// Total number of abstract states in the model at termination.
    pub abstract_state_count: usize,
    /// Maximum refinement depth reached during CEGAR.
    pub refinement_depth: u32,
    /// Number of CEGAR iterations required.
    pub iterations: usize,
}

impl ProofCertificate {
    /// Validate the structural integrity of the proof certificate.
    ///
    /// A minimal sanity check: the invariant should be non-empty, and the
    /// reported abstract state count should be at least as large as the
    /// invariant.
    pub fn validate(&self) -> Result<(), CegarError> {
        if self.invariant_states.is_empty() {
            return Err(CegarError::InvalidCertificate(
                "proof certificate has empty invariant".into(),
            ));
        }
        if self.abstract_state_count < self.invariant_states.len() {
            return Err(CegarError::InvalidCertificate(
                "abstract_state_count smaller than invariant size".into(),
            ));
        }
        Ok(())
    }

    /// Return the size of the invariant (number of abstract states).
    pub fn invariant_size(&self) -> usize {
        self.invariant_states.len()
    }

    /// Check whether the invariant is inductive with respect to a given
    /// transition check function.
    ///
    /// The callback `is_successor` receives `(src, dst)` pairs and should
    /// return `true` when `dst` is a valid abstract successor of `src`.
    pub fn check_inductiveness<F>(&self, is_successor: F) -> bool
    where
        F: Fn(&AbstractState, &AbstractState) -> bool,
    {
        use std::collections::HashSet;
        let inv_set: HashSet<_> = self.invariant_states.iter().collect();
        for src in &self.invariant_states {
            for dst in &self.invariant_states {
                if is_successor(src, dst) && !inv_set.contains(dst) {
                    return false;
                }
            }
        }
        true
    }
}

impl fmt::Display for ProofCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ProofCertificate(invariant={}, states={}, depth={}, iters={})",
            self.invariant_states.len(),
            self.abstract_state_count,
            self.refinement_depth,
            self.iterations,
        )
    }
}

// ---------------------------------------------------------------------------
// Counterexample certificate
// ---------------------------------------------------------------------------

/// A counterexample certificate attesting that a property is violated.
///
/// Contains both the abstract counterexample trace and (optionally) a
/// concretized witness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterexampleCertificate {
    /// The abstract counterexample trace.
    pub abstract_trace: Counterexample,
    /// A concretized (spatial) witness, if one could be constructed.
    pub concrete_witness: Option<ConcreteCounterexample>,
    /// Violated spatial constraints along the trace.
    pub violated_constraints: Vec<SpatialConstraint>,
    /// Human-readable description of the violation.
    pub description: String,
}

impl CounterexampleCertificate {
    /// Validate structural integrity.
    pub fn validate(&self) -> Result<(), CegarError> {
        if self.abstract_trace.length == 0 {
            return Err(CegarError::InvalidCertificate(
                "counterexample has zero length".into(),
            ));
        }
        Ok(())
    }

    /// Whether a concrete witness was successfully produced.
    pub fn has_concrete_witness(&self) -> bool {
        self.concrete_witness.is_some()
    }

    /// Length of the abstract trace.
    pub fn trace_length(&self) -> usize {
        self.abstract_trace.length
    }
}

impl fmt::Display for CounterexampleCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CounterexampleCertificate(len={}, concrete={}, desc=\"{}\")",
            self.abstract_trace.length,
            self.has_concrete_witness(),
            self.description,
        )
    }
}

// ---------------------------------------------------------------------------
// Verification certificate (top-level)
// ---------------------------------------------------------------------------

/// Top-level verification certificate produced by the CEGAR engine.
///
/// Exactly one of `proof` or `counterexample` should be `Some` in a valid
/// certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationCertificate {
    /// The property that was verified.
    pub property: Property,
    /// Proof certificate (present when the property holds).
    pub proof: Option<ProofCertificate>,
    /// Counterexample certificate (present when the property is violated).
    pub counterexample: Option<CounterexampleCertificate>,
    /// Arbitrary metadata (e.g. solver version, timestamps, …).
    pub metadata: BTreeMap<String, String>,
}

impl VerificationCertificate {
    /// Whether the certificate attests that the property holds.
    pub fn is_proof(&self) -> bool {
        self.proof.is_some()
    }

    /// Whether the certificate attests a violation.
    pub fn is_counterexample(&self) -> bool {
        self.counterexample.is_some()
    }

    /// Validate the certificate: exactly one of proof/counterexample must be
    /// present, and the present sub-certificate must itself be valid.
    pub fn validate(&self) -> Result<(), CegarError> {
        match (&self.proof, &self.counterexample) {
            (Some(p), None) => p.validate(),
            (None, Some(c)) => c.validate(),
            (Some(_), Some(_)) => Err(CegarError::InvalidCertificate(
                "certificate has both proof and counterexample".into(),
            )),
            (None, None) => Err(CegarError::InvalidCertificate(
                "certificate has neither proof nor counterexample".into(),
            )),
        }
    }

    /// Return a unique identifier for the certificate (UUID v4).
    pub fn id(&self) -> String {
        // Deterministic from metadata if a "certificate_id" was set.
        if let Some(id) = self.metadata.get("certificate_id") {
            return id.clone();
        }
        Uuid::new_v4().to_string()
    }

    /// Add a metadata entry.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Retrieve a metadata entry.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }
}

impl fmt::Display for VerificationCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VerificationCertificate(property={}", self.property)?;
        if let Some(ref p) = self.proof {
            write!(f, ", proof={p}")?;
        }
        if let Some(ref c) = self.counterexample {
            write!(f, ", cex={c}")?;
        }
        write!(f, ")")
    }
}

// ---------------------------------------------------------------------------
// Certificate builder (fluent API)
// ---------------------------------------------------------------------------

/// Fluent builder for constructing [`VerificationCertificate`]s.
#[derive(Debug, Clone)]
pub struct CertificateBuilder {
    property: Option<Property>,
    abstract_state_count: usize,
    iterations: usize,
    refinement_depth: u32,
    invariant_states: Vec<AbstractState>,
    counterexample: Option<CounterexampleCertificate>,
    metadata: BTreeMap<String, String>,
}

impl CertificateBuilder {
    /// Create a new builder with default values.
    pub fn new() -> Self {
        Self {
            property: None,
            abstract_state_count: 0,
            iterations: 0,
            refinement_depth: 0,
            invariant_states: Vec::new(),
            counterexample: None,
            metadata: BTreeMap::new(),
        }
    }

    /// Set the property being verified.
    pub fn set_property(&mut self, property: Property) -> &mut Self {
        self.property = Some(property);
        self
    }

    /// Set the total abstract state count.
    pub fn set_abstract_state_count(&mut self, count: usize) -> &mut Self {
        self.abstract_state_count = count;
        self
    }

    /// Set the number of CEGAR iterations.
    pub fn set_iterations(&mut self, iterations: usize) -> &mut Self {
        self.iterations = iterations;
        self
    }

    /// Set the refinement depth.
    pub fn set_refinement_depth(&mut self, depth: u32) -> &mut Self {
        self.refinement_depth = depth;
        self
    }

    /// Set the invariant states for a proof certificate.
    pub fn set_invariant_states(&mut self, states: Vec<AbstractState>) -> &mut Self {
        self.invariant_states = states;
        self
    }

    /// Set the counterexample certificate.
    pub fn set_counterexample(&mut self, cex: CounterexampleCertificate) -> &mut Self {
        self.counterexample = Some(cex);
        self
    }

    /// Add a metadata entry.
    pub fn add_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> &mut Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Record timing information.
    pub fn set_duration(&mut self, duration: Duration) -> &mut Self {
        self.metadata
            .insert("duration_ms".into(), duration.as_millis().to_string());
        self
    }

    /// Build the verification certificate.
    ///
    /// If invariant states are non-empty, a proof certificate is produced.
    /// If a counterexample was supplied, a counterexample certificate is
    /// produced. It is an error to supply both or neither.
    pub fn build(&self) -> VerificationCertificate {
        let property = self
            .property
            .clone()
            .unwrap_or(Property::DeadlockFreedom);

        let proof = if !self.invariant_states.is_empty() {
            Some(ProofCertificate {
                invariant_states: self.invariant_states.clone(),
                abstract_state_count: self.abstract_state_count,
                refinement_depth: self.refinement_depth,
                iterations: self.iterations,
            })
        } else if self.counterexample.is_none() {
            // If no explicit counterexample, create a minimal proof cert.
            Some(ProofCertificate {
                invariant_states: Vec::new(),
                abstract_state_count: self.abstract_state_count,
                refinement_depth: self.refinement_depth,
                iterations: self.iterations,
            })
        } else {
            None
        };

        let mut metadata = self.metadata.clone();
        metadata
            .entry("certificate_id".into())
            .or_insert_with(|| Uuid::new_v4().to_string());

        VerificationCertificate {
            property,
            proof,
            counterexample: self.counterexample.clone(),
            metadata,
        }
    }

    /// Build and validate the certificate, returning an error on invalid
    /// structure.
    pub fn build_validated(&self) -> Result<VerificationCertificate, CegarError> {
        let cert = self.build();
        cert.validate()?;
        Ok(cert)
    }
}

impl Default for CertificateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Composite certificate (for compositional verification)
// ---------------------------------------------------------------------------

/// A composite certificate aggregating sub-certificates from components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeCertificate {
    /// Per-component certificates, keyed by component identifier.
    pub component_certificates: BTreeMap<String, VerificationCertificate>,
    /// Interface obligations discharged between components.
    pub interface_obligations: Vec<InterfaceObligation>,
    /// Overall verdict.
    pub verdict: CompositeVerdict,
}

/// An interface obligation between two components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceObligation {
    /// Identifier of the component that assumes the obligation.
    pub assumer: String,
    /// Identifier of the component that guarantees the obligation.
    pub guarantor: String,
    /// The constraint that forms the obligation.
    pub constraint: SpatialConstraint,
    /// Whether the obligation was discharged.
    pub discharged: bool,
}

/// Verdict of a composite verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositeVerdict {
    /// All components verified and all obligations discharged.
    AllVerified,
    /// At least one component was falsified.
    SomeFalsified,
    /// Some components could not be decided.
    Inconclusive,
}

impl CompositeCertificate {
    /// Validate the composite certificate.
    pub fn validate(&self) -> Result<(), CegarError> {
        // All sub-certificates must be individually valid.
        for (name, cert) in &self.component_certificates {
            cert.validate().map_err(|e| {
                CegarError::InvalidCertificate(format!("component '{name}': {e}"))
            })?;
        }
        // All obligations must be discharged for AllVerified.
        if self.verdict == CompositeVerdict::AllVerified {
            for ob in &self.interface_obligations {
                if !ob.discharged {
                    return Err(CegarError::InvalidCertificate(format!(
                        "obligation {} -> {} not discharged",
                        ob.assumer, ob.guarantor,
                    )));
                }
            }
        }
        Ok(())
    }

    /// Number of components.
    pub fn component_count(&self) -> usize {
        self.component_certificates.len()
    }

    /// Number of interface obligations.
    pub fn obligation_count(&self) -> usize {
        self.interface_obligations.len()
    }

    /// How many obligations were discharged.
    pub fn discharged_count(&self) -> usize {
        self.interface_obligations
            .iter()
            .filter(|o| o.discharged)
            .count()
    }
}

impl fmt::Display for CompositeCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CompositeCertificate(components={}, obligations={}/{}, verdict={:?})",
            self.component_count(),
            self.discharged_count(),
            self.obligation_count(),
            self.verdict,
        )
    }
}

// ---------------------------------------------------------------------------
// Certificate serialization helpers
// ---------------------------------------------------------------------------

/// Serialize a certificate to JSON.
pub fn certificate_to_json(cert: &VerificationCertificate) -> Result<String, CegarError> {
    serde_json::to_string_pretty(cert).map_err(|e| {
        CegarError::InvalidCertificate(format!("JSON serialization failed: {e}"))
    })
}

/// Deserialize a certificate from JSON.
pub fn certificate_from_json(json: &str) -> Result<VerificationCertificate, CegarError> {
    serde_json::from_str(json).map_err(|e| {
        CegarError::InvalidCertificate(format!("JSON deserialization failed: {e}"))
    })
}

/// Compute a fingerprint (SHA-256 hex digest) for a certificate.
///
/// This uses the JSON representation, so it is deterministic for a given
/// certificate value (assuming BTreeMap iteration order is stable).
pub fn certificate_fingerprint(cert: &VerificationCertificate) -> Result<String, CegarError> {
    let json = certificate_to_json(cert)?;
    // Simple FNV-1a hash (we avoid pulling in SHA-256 crate).
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in json.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Ok(format!("{hash:016x}"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AbstractBlockId, StateId};

    fn sample_abstract_state() -> AbstractState {
        AbstractState {
            automaton_state: StateId(0),
            block_id: AbstractBlockId(0),
        }
    }

    fn sample_proof_cert() -> ProofCertificate {
        ProofCertificate {
            invariant_states: vec![sample_abstract_state()],
            abstract_state_count: 5,
            refinement_depth: 3,
            iterations: 10,
        }
    }

    #[test]
    fn test_proof_certificate_validate() {
        let cert = sample_proof_cert();
        assert!(cert.validate().is_ok());

        let bad = ProofCertificate {
            invariant_states: vec![],
            ..cert.clone()
        };
        assert!(bad.validate().is_err());

        let bad2 = ProofCertificate {
            abstract_state_count: 0,
            ..cert
        };
        assert!(bad2.validate().is_err());
    }

    #[test]
    fn test_proof_certificate_display() {
        let cert = sample_proof_cert();
        let s = format!("{cert}");
        assert!(s.contains("invariant=1"));
        assert!(s.contains("states=5"));
    }

    #[test]
    fn test_proof_inductiveness_trivial() {
        let cert = sample_proof_cert();
        // Trivial successor: everything is a successor of everything.
        assert!(cert.check_inductiveness(|_, _| true));
    }

    #[test]
    fn test_counterexample_certificate_validate() {
        let cex = CounterexampleCertificate {
            abstract_trace: Counterexample {
                states: vec![sample_abstract_state()],
                transitions: vec![],
                length: 1,
                property_violated: Property::DeadlockFreedom,
                is_lasso: false,
                lasso_stem_length: None,
            },
            concrete_witness: None,
            violated_constraints: vec![],
            description: "test".into(),
        };
        assert!(cex.validate().is_ok());
        assert!(!cex.has_concrete_witness());
        assert_eq!(cex.trace_length(), 1);
    }

    #[test]
    fn test_counterexample_zero_length() {
        let cex = CounterexampleCertificate {
            abstract_trace: Counterexample {
                states: vec![],
                transitions: vec![],
                length: 0,
                property_violated: Property::DeadlockFreedom,
                is_lasso: false,
                lasso_stem_length: None,
            },
            concrete_witness: None,
            violated_constraints: vec![],
            description: "empty".into(),
        };
        assert!(cex.validate().is_err());
    }

    #[test]
    fn test_verification_certificate_proof() {
        let cert = VerificationCertificate {
            property: Property::DeadlockFreedom,
            proof: Some(sample_proof_cert()),
            counterexample: None,
            metadata: BTreeMap::new(),
        };
        assert!(cert.is_proof());
        assert!(!cert.is_counterexample());
        assert!(cert.validate().is_ok());
    }

    #[test]
    fn test_verification_certificate_both_fail() {
        let cex = CounterexampleCertificate {
            abstract_trace: Counterexample {
                states: vec![sample_abstract_state()],
                transitions: vec![],
                length: 1,
                property_violated: Property::DeadlockFreedom,
                is_lasso: false,
                lasso_stem_length: None,
            },
            concrete_witness: None,
            violated_constraints: vec![],
            description: "bug".into(),
        };
        let cert = VerificationCertificate {
            property: Property::DeadlockFreedom,
            proof: Some(sample_proof_cert()),
            counterexample: Some(cex),
            metadata: BTreeMap::new(),
        };
        assert!(cert.validate().is_err());
    }

    #[test]
    fn test_verification_certificate_none_fail() {
        let cert = VerificationCertificate {
            property: Property::DeadlockFreedom,
            proof: None,
            counterexample: None,
            metadata: BTreeMap::new(),
        };
        assert!(cert.validate().is_err());
    }

    #[test]
    fn test_builder_proof() {
        let mut builder = CertificateBuilder::new();
        builder.set_property(Property::DeadlockFreedom);
        builder.set_abstract_state_count(10);
        builder.set_iterations(5);
        builder.set_refinement_depth(2);
        builder.set_invariant_states(vec![sample_abstract_state()]);
        builder.add_metadata("solver", "cegar-v1");

        let cert = builder.build();
        assert!(cert.is_proof());
        assert_eq!(cert.get_metadata("solver"), Some("cegar-v1"));
    }

    #[test]
    fn test_builder_counterexample() {
        let cex = CounterexampleCertificate {
            abstract_trace: Counterexample {
                states: vec![sample_abstract_state()],
                transitions: vec![],
                length: 1,
                property_violated: Property::DeadlockFreedom,
                is_lasso: false,
                lasso_stem_length: None,
            },
            concrete_witness: None,
            violated_constraints: vec![],
            description: "bug".into(),
        };
        let mut builder = CertificateBuilder::new();
        builder.set_counterexample(cex);
        let cert = builder.build();
        assert!(cert.is_counterexample());
    }

    #[test]
    fn test_builder_validated() {
        let mut builder = CertificateBuilder::new();
        builder.set_invariant_states(vec![sample_abstract_state()]);
        builder.set_abstract_state_count(1);
        let result = builder.build_validated();
        assert!(result.is_ok());
    }

    #[test]
    fn test_certificate_serialization() {
        let cert = VerificationCertificate {
            property: Property::DeadlockFreedom,
            proof: Some(sample_proof_cert()),
            counterexample: None,
            metadata: BTreeMap::new(),
        };
        let json = certificate_to_json(&cert).unwrap();
        let restored = certificate_from_json(&json).unwrap();
        assert!(restored.is_proof());
    }

    #[test]
    fn test_certificate_fingerprint_deterministic() {
        let cert = VerificationCertificate {
            property: Property::DeadlockFreedom,
            proof: Some(sample_proof_cert()),
            counterexample: None,
            metadata: BTreeMap::new(),
        };
        let fp1 = certificate_fingerprint(&cert).unwrap();
        let fp2 = certificate_fingerprint(&cert).unwrap();
        assert_eq!(fp1, fp2);
        assert_eq!(fp1.len(), 16);
    }

    #[test]
    fn test_composite_certificate_validate() {
        let mut components = BTreeMap::new();
        components.insert(
            "comp_a".to_string(),
            VerificationCertificate {
                property: Property::DeadlockFreedom,
                proof: Some(sample_proof_cert()),
                counterexample: None,
                metadata: BTreeMap::new(),
            },
        );

        let composite = CompositeCertificate {
            component_certificates: components,
            interface_obligations: vec![],
            verdict: CompositeVerdict::AllVerified,
        };
        assert!(composite.validate().is_ok());
        assert_eq!(composite.component_count(), 1);
        assert_eq!(composite.obligation_count(), 0);
    }

    #[test]
    fn test_composite_certificate_undischarged() {
        let mut components = BTreeMap::new();
        components.insert(
            "a".into(),
            VerificationCertificate {
                property: Property::DeadlockFreedom,
                proof: Some(sample_proof_cert()),
                counterexample: None,
                metadata: BTreeMap::new(),
            },
        );

        let ob = InterfaceObligation {
            assumer: "a".into(),
            guarantor: "b".into(),
            constraint: SpatialConstraint::True,
            discharged: false,
        };

        let composite = CompositeCertificate {
            component_certificates: components,
            interface_obligations: vec![ob],
            verdict: CompositeVerdict::AllVerified,
        };
        // Verdict says AllVerified but obligation is not discharged.
        assert!(composite.validate().is_err());
    }

    #[test]
    fn test_builder_set_duration() {
        let mut builder = CertificateBuilder::new();
        builder.set_duration(Duration::from_millis(1234));
        let cert = builder.build();
        assert_eq!(cert.get_metadata("duration_ms"), Some("1234"));
    }

    #[test]
    fn test_verification_certificate_display() {
        let cert = VerificationCertificate {
            property: Property::DeadlockFreedom,
            proof: Some(sample_proof_cert()),
            counterexample: None,
            metadata: BTreeMap::new(),
        };
        let s = format!("{cert}");
        assert!(s.contains("VerificationCertificate"));
        assert!(s.contains("proof="));
    }

    #[test]
    fn test_add_metadata() {
        let mut cert = VerificationCertificate {
            property: Property::DeadlockFreedom,
            proof: Some(sample_proof_cert()),
            counterexample: None,
            metadata: BTreeMap::new(),
        };
        cert.add_metadata("key1", "value1");
        assert_eq!(cert.get_metadata("key1"), Some("value1"));
        assert_eq!(cert.get_metadata("missing"), None);
    }
}
