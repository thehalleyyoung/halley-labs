//! NDC → RxCUI mapping and drug class lookups.

use super::concept::{RxCui, RxClassEntry, RxNormConcept, RxNormTermType};
use super::database::EMBEDDED_DB;

/// Lookup an RxCUI from an NDC (National Drug Code).
///
/// Uses the embedded lookup table for common drugs. Returns `None` if
/// the NDC is not found in the embedded database.
pub fn ndc_to_rxcui(ndc: &str) -> Option<RxCui> {
    let normalized = normalize_ndc(ndc);
    EMBEDDED_DB
        .ndc_to_rxcui
        .iter()
        .find(|(n, _)| *n == normalized)
        .map(|(_, cui)| RxCui::new(*cui))
}

/// Lookup an RxNorm concept by its RxCUI.
pub fn lookup_rxcui(rxcui: &str) -> Option<&'static RxNormConceptEntry> {
    EMBEDDED_DB
        .concepts
        .iter()
        .find(|c| c.rxcui == rxcui)
}

/// Get the drug class(es) for a given RxCUI.
pub fn drug_classes_for_rxcui(rxcui: &str) -> Vec<RxClassEntry> {
    EMBEDDED_DB
        .rxcui_to_class
        .iter()
        .filter(|(cui, _, _, _)| *cui == rxcui)
        .map(|(_, class_id, class_name, class_type)| RxClassEntry {
            class_id: class_id.to_string(),
            class_name: class_name.to_string(),
            class_type: class_type.to_string(),
        })
        .collect()
}

/// Map an RxNorm drug class name to a GuardPharma DrugClass.
pub fn rxclass_to_drug_class(class_name: &str) -> guardpharma_types::DrugClass {
    let lower = class_name.to_lowercase();
    if lower.contains("nsaid") || lower.contains("anti-inflammatory") {
        guardpharma_types::DrugClass::NSAID
    } else if lower.contains("anticoagulant") {
        guardpharma_types::DrugClass::Anticoagulant
    } else if lower.contains("antihypertensive") {
        guardpharma_types::DrugClass::Antihypertensive
    } else if lower.contains("antidiabetic") || lower.contains("hypoglycemic") {
        guardpharma_types::DrugClass::Antidiabetic
    } else if lower.contains("statin") || lower.contains("hmg-coa") {
        guardpharma_types::DrugClass::Statin
    } else if lower.contains("antidepressant") || lower.contains("ssri") || lower.contains("snri") {
        guardpharma_types::DrugClass::Antidepressant
    } else if lower.contains("antibiotic") || lower.contains("antimicrobial") {
        guardpharma_types::DrugClass::Antibiotic
    } else if lower.contains("antiarrhythmic") {
        guardpharma_types::DrugClass::Antiarrhythmic
    } else if lower.contains("opioid") {
        guardpharma_types::DrugClass::Opioid
    } else if lower.contains("benzodiazepine") {
        guardpharma_types::DrugClass::Benzodiazepine
    } else if lower.contains("proton pump") || lower.contains("ppi") {
        guardpharma_types::DrugClass::PPI
    } else if lower.contains("anticonvulsant") || lower.contains("antiepileptic") {
        guardpharma_types::DrugClass::Anticonvulsant
    } else if lower.contains("corticosteroid") {
        guardpharma_types::DrugClass::Corticosteroid
    } else if lower.contains("immunosuppressant") {
        guardpharma_types::DrugClass::Immunosuppressant
    } else if lower.contains("bronchodilator") || lower.contains("beta-2 agonist") {
        guardpharma_types::DrugClass::Bronchodilator
    } else if lower.contains("diuretic") {
        guardpharma_types::DrugClass::Diuretic
    } else if lower.contains("ace inhibitor") || lower.contains("angiotensin-converting") {
        guardpharma_types::DrugClass::ACEInhibitor
    } else if lower.contains("angiotensin receptor blocker") || lower.contains("arb") {
        guardpharma_types::DrugClass::ARB
    } else if lower.contains("beta blocker") || lower.contains("beta-adrenergic") {
        guardpharma_types::DrugClass::BetaBlocker
    } else if lower.contains("calcium channel") {
        guardpharma_types::DrugClass::CalciumChannelBlocker
    } else if lower.contains("antiplatelet") {
        guardpharma_types::DrugClass::Antiplatelet
    } else if lower.contains("antipsychotic") {
        guardpharma_types::DrugClass::Antipsychotic
    } else if lower.contains("sedative") || lower.contains("hypnotic") {
        guardpharma_types::DrugClass::Sedative
    } else if lower.contains("antifungal") {
        guardpharma_types::DrugClass::Antifungal
    } else if lower.contains("antiviral") {
        guardpharma_types::DrugClass::Antiviral
    } else {
        guardpharma_types::DrugClass::Other(class_name.to_string())
    }
}

/// Resolve a full RxNorm concept from an RxCUI, including its drug classes.
pub fn resolve_concept(rxcui: &str) -> Option<RxNormConcept> {
    let entry = lookup_rxcui(rxcui)?;
    Some(RxNormConcept {
        rxcui: RxCui::new(entry.rxcui),
        name: entry.name.to_string(),
        tty: RxNormTermType::from_tty(entry.tty),
        active: true,
    })
}

/// Static entry for the embedded concept table.
pub struct RxNormConceptEntry {
    pub rxcui: &'static str,
    pub name: &'static str,
    pub tty: &'static str,
}

/// Normalize an NDC code to the standard 11-digit format (no dashes).
fn normalize_ndc(ndc: &str) -> String {
    ndc.replace('-', "")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndc_lookup() {
        // Metformin 500mg – NDC 00378-0234-01
        let cui = ndc_to_rxcui("00378-0234-01");
        assert!(cui.is_some());
        assert_eq!(cui.unwrap().as_str(), "860975");
    }

    #[test]
    fn rxcui_lookup() {
        let concept = resolve_concept("860975");
        assert!(concept.is_some());
        let c = concept.unwrap();
        assert!(c.name.to_lowercase().contains("metformin"));
        assert_eq!(c.tty, RxNormTermType::SCD);
    }

    #[test]
    fn drug_class_lookup() {
        let classes = drug_classes_for_rxcui("860975");
        assert!(!classes.is_empty());
    }

    #[test]
    fn rxclass_mapping() {
        assert_eq!(
            rxclass_to_drug_class("HMG-CoA Reductase Inhibitors (Statins)"),
            guardpharma_types::DrugClass::Statin
        );
        assert_eq!(
            rxclass_to_drug_class("Anticoagulants"),
            guardpharma_types::DrugClass::Anticoagulant
        );
    }
}
