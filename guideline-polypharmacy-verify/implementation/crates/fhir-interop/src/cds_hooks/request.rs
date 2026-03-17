//! CDS Hooks request parsing (order-sign, medication-prescribe).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::FhirAuthorization;
use crate::error::FhirInteropError;
use crate::fhir::bundle::FhirBundle;

/// A CDS Hooks request envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdsHookRequest {
    /// The hook that triggered this request (e.g., "order-sign").
    pub hook: String,
    /// A unique ID for this hook invocation.
    #[serde(rename = "hookInstance")]
    pub hook_instance: String,
    /// The FHIR server base URL.
    #[serde(rename = "fhirServer", default)]
    pub fhir_server: Option<String>,
    /// OAuth2 authorization for FHIR server calls.
    #[serde(rename = "fhirAuthorization", default)]
    pub fhir_authorization: Option<FhirAuthorization>,
    /// Hook-specific context data.
    pub context: CdsHookContext,
    /// Prefetched FHIR resources keyed by template key.
    #[serde(default)]
    pub prefetch: HashMap<String, serde_json::Value>,
}

/// Context data supplied with a CDS Hooks invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdsHookContext {
    /// The patient ID the hook was invoked for.
    #[serde(rename = "patientId", default)]
    pub patient_id: Option<String>,
    /// The user (practitioner) ID.
    #[serde(rename = "userId", default)]
    pub user_id: Option<String>,
    /// For order-sign: the in-progress orders (Bundle of draft MedicationRequests).
    #[serde(rename = "draftOrders", default)]
    pub draft_orders: Option<serde_json::Value>,
    /// For medication-prescribe: the medications being prescribed.
    #[serde(default)]
    pub medications: Option<serde_json::Value>,
    /// Generic extension fields.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl CdsHookRequest {
    /// Parse from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, FhirInteropError> {
        Ok(serde_json::from_str(json)?)
    }

    /// Whether this is an `order-sign` hook.
    pub fn is_order_sign(&self) -> bool {
        self.hook == "order-sign"
    }

    /// Whether this is a `medication-prescribe` hook.
    pub fn is_medication_prescribe(&self) -> bool {
        self.hook == "medication-prescribe"
    }

    /// Extract draft orders as a FHIR Bundle, if present.
    pub fn draft_orders_bundle(&self) -> Option<Result<FhirBundle, FhirInteropError>> {
        let val = self.context.draft_orders.as_ref()?;
        let json = serde_json::to_string(val).ok()?;
        Some(FhirBundle::from_json(&json))
    }

    /// Extract a prefetched resource by key.
    pub fn prefetch_value(&self, key: &str) -> Option<&serde_json::Value> {
        self.prefetch.get(key)
    }

    /// Extract a prefetched resource and deserialize it as a given type.
    pub fn prefetch_as<T: serde::de::DeserializeOwned>(
        &self,
        key: &str,
    ) -> Option<Result<T, FhirInteropError>> {
        let val = self.prefetch.get(key)?;
        Some(serde_json::from_value(val.clone()).map_err(FhirInteropError::from))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_order_sign_request() {
        let json = r#"{
            "hook": "order-sign",
            "hookInstance": "d1577c69-dfbe-44ad-ba6d-3e05e953b2ea",
            "fhirServer": "https://fhir.example.org/r4",
            "context": {
                "patientId": "pt-123",
                "userId": "Practitioner/dr-456",
                "draftOrders": {
                    "resourceType": "Bundle",
                    "type": "collection",
                    "entry": [{
                        "resource": {
                            "resourceType": "MedicationRequest",
                            "id": "mr-draft-1",
                            "status": "draft",
                            "intent": "order",
                            "medicationCodeableConcept": {
                                "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "855332", "display": "Warfarin 5 MG"}]
                            },
                            "subject": {"reference": "Patient/pt-123"}
                        }
                    }]
                }
            },
            "prefetch": {
                "patient": {
                    "resourceType": "Patient",
                    "id": "pt-123",
                    "gender": "male",
                    "birthDate": "1955-03-20"
                }
            }
        }"#;

        let req = CdsHookRequest::from_json(json).unwrap();
        assert!(req.is_order_sign());
        assert_eq!(req.context.patient_id.as_deref(), Some("pt-123"));

        let bundle = req.draft_orders_bundle().unwrap().unwrap();
        assert_eq!(bundle.entry.len(), 1);
    }
}
