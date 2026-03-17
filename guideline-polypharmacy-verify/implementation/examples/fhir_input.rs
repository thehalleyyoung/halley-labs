//! Example: Parsing a FHIR Bundle JSON into GuardPharma types.
//!
//! Demonstrates how a FHIR R4 Patient + MedicationStatement bundle would be
//! converted into GuardPharma's internal PatientProfile representation.
//!
//! Note: The full `guardpharma-fhir-interop` crate is not yet available, so this
//! example performs a manual/illustrative parse using serde_json.
//!
//! Run with: `cargo run --example fhir_input`

use guardpharma_types::{DrugId, PatientInfo, Sex};

/// Minimal FHIR-to-internal conversion (illustrative).
fn parse_fhir_patient(bundle_json: &str) -> Option<(PatientInfo, Vec<(String, f64)>)> {
    let bundle: serde_json::Value = serde_json::from_str(bundle_json).ok()?;

    let entries = bundle.get("entry")?.as_array()?;

    let mut patient_info = PatientInfo::default();
    let mut medications: Vec<(String, f64)> = Vec::new();

    for entry in entries {
        let resource = entry.get("resource")?;
        let resource_type = resource.get("resourceType")?.as_str()?;

        match resource_type {
            "Patient" => {
                // Extract age from birthDate (simplified)
                if let Some(birth) = resource.get("birthDate").and_then(|v| v.as_str()) {
                    let year: i32 = birth[..4].parse().unwrap_or(1970);
                    patient_info.age_years = (2024 - year) as f64;
                }

                // Extract gender
                if let Some(gender) = resource.get("gender").and_then(|v| v.as_str()) {
                    patient_info.sex = match gender {
                        "male" => Sex::Male,
                        "female" => Sex::Female,
                        _ => Sex::Male,
                    };
                }
            }
            "MedicationStatement" => {
                let med_ref = resource
                    .get("medicationCodeableConcept")
                    .and_then(|m| m.get("coding"))
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|c| c.get("display"))
                    .and_then(|d| d.as_str())
                    .unwrap_or("Unknown");

                let dose = resource
                    .get("dosage")
                    .and_then(|d| d.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|d| d.get("doseAndRate"))
                    .and_then(|dr| dr.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|dr| dr.get("doseQuantity"))
                    .and_then(|dq| dq.get("value"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);

                medications.push((med_ref.to_string(), dose));
            }
            _ => {}
        }
    }

    Some((patient_info, medications))
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  GuardPharma — FHIR Bundle Input Example                        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // A simplified FHIR R4 Bundle with Patient + MedicationStatements
    let fhir_bundle = r#"{
  "resourceType": "Bundle",
  "type": "collection",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "id": "example-patient-001",
        "birthDate": "1952-03-15",
        "gender": "male",
        "name": [{"family": "Smith", "given": ["John"]}]
      }
    },
    {
      "resource": {
        "resourceType": "MedicationStatement",
        "status": "active",
        "medicationCodeableConcept": {
          "coding": [{
            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
            "code": "860975",
            "display": "Warfarin"
          }]
        },
        "dosage": [{
          "doseAndRate": [{
            "doseQuantity": {"value": 5.0, "unit": "mg"}
          }],
          "timing": {"repeat": {"frequency": 1, "period": 1, "periodUnit": "d"}}
        }]
      }
    },
    {
      "resource": {
        "resourceType": "MedicationStatement",
        "status": "active",
        "medicationCodeableConcept": {
          "coding": [{
            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
            "code": "6809",
            "display": "Metformin"
          }]
        },
        "dosage": [{
          "doseAndRate": [{
            "doseQuantity": {"value": 1000.0, "unit": "mg"}
          }],
          "timing": {"repeat": {"frequency": 2, "period": 1, "periodUnit": "d"}}
        }]
      }
    },
    {
      "resource": {
        "resourceType": "MedicationStatement",
        "status": "active",
        "medicationCodeableConcept": {
          "coding": [{
            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
            "code": "29046",
            "display": "Lisinopril"
          }]
        },
        "dosage": [{
          "doseAndRate": [{
            "doseQuantity": {"value": 20.0, "unit": "mg"}
          }],
          "timing": {"repeat": {"frequency": 1, "period": 1, "periodUnit": "d"}}
        }]
      }
    }
  ]
}"#;

    println!("Input FHIR Bundle (truncated):");
    println!("  resourceType: Bundle");
    println!("  entries: 4 (1 Patient + 3 MedicationStatements)\n");

    // Parse the FHIR bundle
    match parse_fhir_patient(fhir_bundle) {
        Some((patient_info, medications)) => {
            println!("Parsed Patient Info:");
            println!("  Age: {:.0} years", patient_info.age_years);
            println!("  Sex: {:?}", patient_info.sex);
            println!("  Weight: {:.1} kg (default — not in FHIR)", patient_info.weight_kg);
            println!("  Creatinine: {:.2} mg/dL (default — not in FHIR)", patient_info.serum_creatinine);
            println!();

            println!("Parsed Medications:");
            for (name, dose) in &medications {
                let drug_id = DrugId::new(name);
                println!("  • {} (ID: {}) — {:.0} mg", name, drug_id.as_str(), dose);
            }
            println!();

            println!("Conversion to GuardPharma Internal Format:");
            println!("  PatientInfo {{ age_years: {:.1}, sex: {:?}, weight_kg: {:.1}, ... }}",
                     patient_info.age_years, patient_info.sex, patient_info.weight_kg);
            println!("  Medications: {} active", medications.len());
            println!();

            println!("Notes:");
            println!("  • FHIR → GuardPharma mapping handles RxNorm codes to DrugId");
            println!("  • Weight/height may come from FHIR Observation resources");
            println!("  • Lab values (creatinine, eGFR) from Observation resources");
            println!("  • Full interop via guardpharma-fhir-interop crate (planned)");
        }
        None => {
            println!("  ✗ Failed to parse FHIR bundle");
        }
    }
}
