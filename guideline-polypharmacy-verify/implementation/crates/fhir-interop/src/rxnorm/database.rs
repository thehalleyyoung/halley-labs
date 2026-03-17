//! Embedded lookup tables for common drugs (RxCUI, NDC, drug classes).
//!
//! These tables provide offline resolution for the most commonly
//! prescribed medications without requiring an external API call.

use super::mapping::RxNormConceptEntry;

/// Embedded database of common drug concepts, NDC mappings, and class assignments.
pub struct EmbeddedRxNormDb {
    /// RxNorm concept entries: (rxcui, name, tty).
    pub concepts: &'static [RxNormConceptEntry],
    /// NDC to RxCUI mapping: (normalized_ndc, rxcui).
    pub ndc_to_rxcui: &'static [(&'static str, &'static str)],
    /// RxCUI to drug class mapping: (rxcui, class_id, class_name, class_type).
    pub rxcui_to_class: &'static [(&'static str, &'static str, &'static str, &'static str)],
}

pub static EMBEDDED_DB: EmbeddedRxNormDb = EmbeddedRxNormDb {
    concepts: &[
        // ── Antidiabetics ──────────────────────────────────────────────
        RxNormConceptEntry { rxcui: "860975", name: "Metformin 500 MG Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "860981", name: "Metformin 850 MG Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "860999", name: "Metformin 1000 MG Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "6809", name: "Metformin", tty: "IN" },
        // ── Cardiovascular ─────────────────────────────────────────────
        RxNormConceptEntry { rxcui: "314076", name: "Lisinopril 10 MG Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "314077", name: "Lisinopril 20 MG Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "29046", name: "Lisinopril", tty: "IN" },
        RxNormConceptEntry { rxcui: "197361", name: "Amlodipine 5 MG Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "197362", name: "Amlodipine 10 MG Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "1202", name: "Amlodipine", tty: "IN" },
        RxNormConceptEntry { rxcui: "200031", name: "Losartan 50 MG Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "200032", name: "Losartan 100 MG Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "52175", name: "Losartan", tty: "IN" },
        RxNormConceptEntry { rxcui: "866924", name: "Metoprolol Succinate 50 MG Extended Release Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "6918", name: "Metoprolol", tty: "IN" },
        RxNormConceptEntry { rxcui: "310798", name: "Hydrochlorothiazide 25 MG Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "5487", name: "Hydrochlorothiazide", tty: "IN" },
        // ── Statins ────────────────────────────────────────────────────
        RxNormConceptEntry { rxcui: "259255", name: "Atorvastatin 20 MG Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "259256", name: "Atorvastatin 40 MG Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "83367", name: "Atorvastatin", tty: "IN" },
        RxNormConceptEntry { rxcui: "314231", name: "Simvastatin 20 MG Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "36567", name: "Simvastatin", tty: "IN" },
        RxNormConceptEntry { rxcui: "861634", name: "Rosuvastatin 10 MG Oral Tablet", tty: "SCD" },
        // ── Anticoagulants / Antiplatelets ─────────────────────────────
        RxNormConceptEntry { rxcui: "855332", name: "Warfarin 5 MG Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "11289", name: "Warfarin", tty: "IN" },
        RxNormConceptEntry { rxcui: "318272", name: "Aspirin 81 MG Delayed Release Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "1191", name: "Aspirin", tty: "IN" },
        RxNormConceptEntry { rxcui: "32968", name: "Clopidogrel", tty: "IN" },
        RxNormConceptEntry { rxcui: "1364430", name: "Apixaban 5 MG Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "1232082", name: "Rivaroxaban 20 MG Oral Tablet", tty: "SCD" },
        // ── PPIs ───────────────────────────────────────────────────────
        RxNormConceptEntry { rxcui: "311566", name: "Omeprazole 20 MG Delayed Release Oral Capsule", tty: "SCD" },
        RxNormConceptEntry { rxcui: "7646", name: "Omeprazole", tty: "IN" },
        RxNormConceptEntry { rxcui: "261106", name: "Pantoprazole 40 MG Delayed Release Oral Tablet", tty: "SCD" },
        // ── Antidepressants ────────────────────────────────────────────
        RxNormConceptEntry { rxcui: "312938", name: "Sertraline 50 MG Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "36437", name: "Sertraline", tty: "IN" },
        RxNormConceptEntry { rxcui: "310384", name: "Fluoxetine 20 MG Oral Capsule", tty: "SCD" },
        RxNormConceptEntry { rxcui: "4493", name: "Fluoxetine", tty: "IN" },
        // ── Pain / Opioids ─────────────────────────────────────────────
        RxNormConceptEntry { rxcui: "313782", name: "Acetaminophen 500 MG Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "161", name: "Acetaminophen", tty: "IN" },
        RxNormConceptEntry { rxcui: "197696", name: "Ibuprofen 200 MG Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "5640", name: "Ibuprofen", tty: "IN" },
        // ── Respiratory ────────────────────────────────────────────────
        RxNormConceptEntry { rxcui: "745679", name: "Albuterol 0.09 MG/ACTUAT Metered Dose Inhaler", tty: "SCD" },
        RxNormConceptEntry { rxcui: "435", name: "Albuterol", tty: "IN" },
        // ── Thyroid ────────────────────────────────────────────────────
        RxNormConceptEntry { rxcui: "966247", name: "Levothyroxine 50 MCG Oral Tablet", tty: "SCD" },
        RxNormConceptEntry { rxcui: "10582", name: "Levothyroxine", tty: "IN" },
    ],

    ndc_to_rxcui: &[
        // Metformin
        ("00378023401", "860975"),
        ("00378023501", "860981"),
        ("00378023601", "860999"),
        // Lisinopril
        ("00378018001", "314076"),
        ("00378018101", "314077"),
        // Amlodipine
        ("00378003201", "197361"),
        ("00378003301", "197362"),
        // Atorvastatin
        ("00378215501", "259255"),
        ("00378215601", "259256"),
        // Warfarin
        ("00056017270", "855332"),
        // Aspirin 81mg
        ("41250027301", "318272"),
        // Omeprazole
        ("62175013643", "311566"),
        // Sertraline
        ("00378426001", "312938"),
        // Ibuprofen
        ("00113046062", "197696"),
        // Levothyroxine
        ("00378180601", "966247"),
        // Hydrochlorothiazide
        ("00378202001", "310798"),
        // Losartan
        ("00378001501", "200031"),
    ],

    rxcui_to_class: &[
        // Metformin
        ("860975", "N0000175440", "Antidiabetic Agents", "EPC"),
        ("860981", "N0000175440", "Antidiabetic Agents", "EPC"),
        ("860999", "N0000175440", "Antidiabetic Agents", "EPC"),
        // Lisinopril
        ("314076", "N0000175557", "ACE Inhibitors", "EPC"),
        ("314077", "N0000175557", "ACE Inhibitors", "EPC"),
        // Amlodipine
        ("197361", "N0000175558", "Calcium Channel Blockers", "EPC"),
        ("197362", "N0000175558", "Calcium Channel Blockers", "EPC"),
        // Losartan
        ("200031", "N0000175559", "Angiotensin Receptor Blockers (ARBs)", "EPC"),
        ("200032", "N0000175559", "Angiotensin Receptor Blockers (ARBs)", "EPC"),
        // Metoprolol
        ("866924", "N0000175560", "Beta-Adrenergic Blocking Agents", "EPC"),
        // HCTZ
        ("310798", "N0000175561", "Diuretic Agents", "EPC"),
        // Atorvastatin
        ("259255", "N0000175562", "HMG-CoA Reductase Inhibitors (Statins)", "EPC"),
        ("259256", "N0000175562", "HMG-CoA Reductase Inhibitors (Statins)", "EPC"),
        // Simvastatin
        ("314231", "N0000175562", "HMG-CoA Reductase Inhibitors (Statins)", "EPC"),
        // Warfarin
        ("855332", "N0000175563", "Anticoagulants", "EPC"),
        // Aspirin
        ("318272", "N0000175564", "Antiplatelet Agents", "EPC"),
        // Omeprazole
        ("311566", "N0000175565", "Proton Pump Inhibitors (PPIs)", "EPC"),
        // Sertraline
        ("312938", "N0000175566", "Antidepressant Agents (SSRIs)", "EPC"),
        // Ibuprofen
        ("197696", "N0000175567", "NSAIDs", "EPC"),
        // Albuterol
        ("745679", "N0000175568", "Bronchodilator Agents (Beta-2 Agonists)", "EPC"),
        // Levothyroxine
        ("966247", "N0000175569", "Thyroid Hormones", "EPC"),
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_db_has_entries() {
        assert!(!EMBEDDED_DB.concepts.is_empty());
        assert!(!EMBEDDED_DB.ndc_to_rxcui.is_empty());
        assert!(!EMBEDDED_DB.rxcui_to_class.is_empty());
    }

    #[test]
    fn can_find_metformin() {
        let found = EMBEDDED_DB.concepts.iter().any(|c| c.rxcui == "860975");
        assert!(found);
    }
}
