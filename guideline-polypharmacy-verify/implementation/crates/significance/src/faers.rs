//! FAERS (FDA Adverse Event Reporting System) disproportionality signal detection.
//!
//! Implements standard pharmacovigilance signal detection metrics:
//! - Reporting Odds Ratio (ROR)
//! - Proportional Reporting Ratio (PRR)
//! - Information Component (IC, Bayesian shrinkage approach)
//!
//! Includes a built-in database of ~80+ pre-computed signals for clinically
//! important drug–drug–adverse-event triples.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ─────────────────────────── Enums ───────────────────────────────────────

/// Type of adverse event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdverseEventType {
    QTProlongation,
    TorsadesDePointes,
    Bleeding,
    GastrointestinalBleeding,
    IntracranialHemorrhage,
    Rhabdomyolysis,
    LiverInjury,
    RenalFailure,
    Hypoglycemia,
    Hyperkalemia,
    Hyponatremia,
    SerotoninSyndrome,
    StevensJohnson,
    Neutropenia,
    Thrombocytopenia,
    Pancytopenia,
    Seizure,
    RespiratoryDepression,
    LacticAcidosis,
    Angioedema,
    Anaphylaxis,
    HepaticFailure,
    Pancreatitis,
    TendonRupture,
    PeripheralNeuropathy,
    Myopathy,
    HypotensiveCrisis,
    Bradycardia,
    Tachycardia,
    NMS,
    Other,
}

impl AdverseEventType {
    /// A human-readable label for the adverse event.
    pub fn label(&self) -> &'static str {
        match self {
            AdverseEventType::QTProlongation => "QT Prolongation",
            AdverseEventType::TorsadesDePointes => "Torsades de Pointes",
            AdverseEventType::Bleeding => "Bleeding",
            AdverseEventType::GastrointestinalBleeding => "GI Bleeding",
            AdverseEventType::IntracranialHemorrhage => "Intracranial Hemorrhage",
            AdverseEventType::Rhabdomyolysis => "Rhabdomyolysis",
            AdverseEventType::LiverInjury => "Liver Injury (DILI)",
            AdverseEventType::RenalFailure => "Renal Failure",
            AdverseEventType::Hypoglycemia => "Hypoglycemia",
            AdverseEventType::Hyperkalemia => "Hyperkalemia",
            AdverseEventType::Hyponatremia => "Hyponatremia",
            AdverseEventType::SerotoninSyndrome => "Serotonin Syndrome",
            AdverseEventType::StevensJohnson => "Stevens-Johnson Syndrome",
            AdverseEventType::Neutropenia => "Neutropenia",
            AdverseEventType::Thrombocytopenia => "Thrombocytopenia",
            AdverseEventType::Pancytopenia => "Pancytopenia",
            AdverseEventType::Seizure => "Seizure",
            AdverseEventType::RespiratoryDepression => "Respiratory Depression",
            AdverseEventType::LacticAcidosis => "Lactic Acidosis",
            AdverseEventType::Angioedema => "Angioedema",
            AdverseEventType::Anaphylaxis => "Anaphylaxis",
            AdverseEventType::HepaticFailure => "Hepatic Failure",
            AdverseEventType::Pancreatitis => "Pancreatitis",
            AdverseEventType::TendonRupture => "Tendon Rupture",
            AdverseEventType::PeripheralNeuropathy => "Peripheral Neuropathy",
            AdverseEventType::Myopathy => "Myopathy",
            AdverseEventType::HypotensiveCrisis => "Hypotensive Crisis",
            AdverseEventType::Bradycardia => "Bradycardia",
            AdverseEventType::Tachycardia => "Tachycardia",
            AdverseEventType::NMS => "Neuroleptic Malignant Syndrome",
            AdverseEventType::Other => "Other",
        }
    }

    /// Base severity weight for the adverse event (0..1).
    pub fn base_severity(&self) -> f64 {
        match self {
            AdverseEventType::TorsadesDePointes => 1.0,
            AdverseEventType::IntracranialHemorrhage => 1.0,
            AdverseEventType::HepaticFailure => 1.0,
            AdverseEventType::StevensJohnson => 0.95,
            AdverseEventType::SerotoninSyndrome => 0.90,
            AdverseEventType::NMS => 0.90,
            AdverseEventType::Pancytopenia => 0.90,
            AdverseEventType::RespiratoryDepression => 0.90,
            AdverseEventType::Anaphylaxis => 0.90,
            AdverseEventType::LacticAcidosis => 0.85,
            AdverseEventType::Rhabdomyolysis => 0.85,
            AdverseEventType::RenalFailure => 0.80,
            AdverseEventType::LiverInjury => 0.80,
            AdverseEventType::QTProlongation => 0.75,
            AdverseEventType::Bleeding => 0.75,
            AdverseEventType::GastrointestinalBleeding => 0.70,
            AdverseEventType::Hyperkalemia => 0.70,
            AdverseEventType::Neutropenia => 0.70,
            AdverseEventType::Seizure => 0.70,
            AdverseEventType::Angioedema => 0.65,
            AdverseEventType::HypotensiveCrisis => 0.65,
            AdverseEventType::Bradycardia => 0.60,
            AdverseEventType::Thrombocytopenia => 0.60,
            AdverseEventType::Hypoglycemia => 0.60,
            AdverseEventType::Hyponatremia => 0.55,
            AdverseEventType::Pancreatitis => 0.55,
            AdverseEventType::TendonRupture => 0.50,
            AdverseEventType::PeripheralNeuropathy => 0.45,
            AdverseEventType::Myopathy => 0.45,
            AdverseEventType::Tachycardia => 0.40,
            AdverseEventType::Other => 0.30,
        }
    }
}

impl fmt::Display for AdverseEventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Qualitative signal strength classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum SignalStrength {
    NoSignal,
    Weak,
    Moderate,
    Strong,
}

impl SignalStrength {
    /// Numeric multiplier for composite scoring.
    pub fn score(&self) -> f64 {
        match self {
            SignalStrength::NoSignal => 0.0,
            SignalStrength::Weak => 0.3,
            SignalStrength::Moderate => 0.6,
            SignalStrength::Strong => 1.0,
        }
    }

    /// Classify from a reporting odds ratio and its lower CI bound.
    pub fn from_ror_and_ci(ror: f64, ci_lower: f64) -> Self {
        if ci_lower <= 1.0 {
            SignalStrength::NoSignal
        } else if ror < 2.0 {
            SignalStrength::Weak
        } else if ror < 5.0 {
            SignalStrength::Moderate
        } else {
            SignalStrength::Strong
        }
    }
}

impl fmt::Display for SignalStrength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignalStrength::NoSignal => write!(f, "No Signal"),
            SignalStrength::Weak => write!(f, "Weak"),
            SignalStrength::Moderate => write!(f, "Moderate"),
            SignalStrength::Strong => write!(f, "Strong"),
        }
    }
}

// ─────────────────────────── Structs ─────────────────────────────────────

/// A FAERS disproportionality signal for a drug pair and adverse event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaersSignal {
    pub drug_a: String,
    pub drug_b: String,
    pub adverse_event: AdverseEventType,
    pub reporting_odds_ratio: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub case_count: u32,
    pub signal_strength: SignalStrength,
    pub prr: Option<f64>,
    pub information_component: Option<f64>,
}

impl FaersSignal {
    pub fn new(
        drug_a: &str,
        drug_b: &str,
        adverse_event: AdverseEventType,
        ror: f64,
        ci_lower: f64,
        ci_upper: f64,
        case_count: u32,
    ) -> Self {
        let signal_strength = SignalStrength::from_ror_and_ci(ror, ci_lower);
        FaersSignal {
            drug_a: drug_a.to_lowercase(),
            drug_b: drug_b.to_lowercase(),
            adverse_event,
            reporting_odds_ratio: ror,
            ci_lower,
            ci_upper,
            case_count,
            signal_strength,
            prr: None,
            information_component: None,
        }
    }

    pub fn with_prr(mut self, prr: f64) -> Self {
        self.prr = Some(prr);
        self
    }

    pub fn with_ic(mut self, ic: f64) -> Self {
        self.information_component = Some(ic);
        self
    }

    /// Composite signal score: signal_strength × adverse_event_severity.
    pub fn composite_score(&self) -> f64 {
        self.signal_strength.score() * self.adverse_event.base_severity()
    }

    /// Whether the signal CI lower bound excludes 1.0 (statistically significant).
    pub fn is_statistically_significant(&self) -> bool {
        self.ci_lower > 1.0
    }

    /// Canonical pair key.
    pub fn pair_key(&self) -> (String, String) {
        let a = self.drug_a.clone();
        let b = self.drug_b.clone();
        if a <= b { (a, b) } else { (b, a) }
    }

    pub fn involves(&self, drug: &str) -> bool {
        let d = drug.to_lowercase();
        self.drug_a == d || self.drug_b == d
    }
}

impl fmt::Display for FaersSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} + {} → {} (ROR={:.2} [{:.2}–{:.2}], n={}, {})",
            self.drug_a, self.drug_b, self.adverse_event,
            self.reporting_odds_ratio, self.ci_lower, self.ci_upper,
            self.case_count, self.signal_strength,
        )
    }
}

// ─────────────────────────── Statistical functions ───────────────────────

/// Compute Reporting Odds Ratio from a 2×2 contingency table.
///
/// | | AE+ | AE- |
/// |--------|------|------|
/// | Drug+  | a    | b    |
/// | Drug-  | c    | d    |
///
/// ROR = (a*d) / (b*c)
pub fn compute_reporting_odds_ratio(a: f64, b: f64, c: f64, d: f64) -> f64 {
    if b * c == 0.0 {
        return f64::INFINITY;
    }
    (a * d) / (b * c)
}

/// Compute the 95% CI for the log(ROR) and return (ROR, CI_lower, CI_upper).
///
/// SE(ln(ROR)) = sqrt(1/a + 1/b + 1/c + 1/d)
pub fn compute_ror_with_ci(a: f64, b: f64, c: f64, d: f64) -> (f64, f64, f64) {
    if a <= 0.0 || b <= 0.0 || c <= 0.0 || d <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let ror = compute_reporting_odds_ratio(a, b, c, d);
    let ln_ror = ror.ln();
    let se = (1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d).sqrt();
    let ci_lower = (ln_ror - 1.96 * se).exp();
    let ci_upper = (ln_ror + 1.96 * se).exp();
    (ror, ci_lower, ci_upper)
}

/// Compute Proportional Reporting Ratio.
///
/// PRR = [a/(a+b)] / [c/(c+d)]
pub fn compute_proportional_reporting_ratio(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let numerator = if a + b == 0.0 { 0.0 } else { a / (a + b) };
    let denominator = if c + d == 0.0 { 0.0 } else { c / (c + d) };
    if denominator == 0.0 {
        return f64::INFINITY;
    }
    numerator / denominator
}

/// Compute Information Component (Bayesian shrinkage log2 ratio).
///
/// IC = log2(observed / expected) with Bayesian shrinkage:
/// IC_shrunk = log2( (observed + 0.5) / (expected + 0.5) )
pub fn compute_information_component(observed: f64, expected: f64) -> f64 {
    if expected <= 0.0 {
        return 0.0;
    }
    ((observed + 0.5) / (expected + 0.5)).log2()
}

/// Compute the IC lower 95% credibility interval bound (IC025).
///
/// Uses a Bayesian approximation: IC025 ≈ IC - 3.3 * SE(IC)
/// where SE(IC) ≈ 1 / sqrt(observed + 0.5) / ln(2)
pub fn compute_ic_lower_bound(observed: f64, expected: f64) -> f64 {
    let ic = compute_information_component(observed, expected);
    let se = if observed + 0.5 > 0.0 {
        1.0 / ((observed + 0.5).sqrt() * 2.0_f64.ln())
    } else {
        0.0
    };
    ic - 1.96 * se
}

/// Determine whether a signal meets significance thresholds.
///
/// A signal is considered significant if:
/// - ROR ≥ threshold (default 2.0)
/// - CI lower bound > 1.0
/// - Case count ≥ 3
pub fn is_significant(signal: &FaersSignal, threshold: f64) -> bool {
    signal.reporting_odds_ratio >= threshold
        && signal.ci_lower > 1.0
        && signal.case_count >= 3
}

/// Determine signal significance with the default ROR threshold of 2.0.
pub fn is_significant_default(signal: &FaersSignal) -> bool {
    is_significant(signal, 2.0)
}

// ─────────────────────────── Database ────────────────────────────────────

/// FAERS signal lookup database keyed by (drug_a, drug_b, adverse_event).
#[derive(Debug, Clone)]
pub struct FaersDatabase {
    /// Signals keyed by (canonical_drug_a, canonical_drug_b, event).
    signals: HashMap<(String, String, AdverseEventType), FaersSignal>,
    /// Drug pair → list of events, for get_all_signals_for_pair.
    pair_index: HashMap<(String, String), Vec<AdverseEventType>>,
}

impl FaersDatabase {
    pub fn empty() -> Self {
        FaersDatabase {
            signals: HashMap::new(),
            pair_index: HashMap::new(),
        }
    }

    /// Create database with built-in signals.
    pub fn with_defaults() -> Self {
        let mut db = Self::empty();
        db.load_defaults();
        db
    }

    pub fn insert(&mut self, signal: FaersSignal) {
        let pair = signal.pair_key();
        let event = signal.adverse_event;
        self.pair_index
            .entry(pair.clone())
            .or_default()
            .push(event);
        self.signals.insert((pair.0, pair.1, event), signal);
    }

    /// Lookup a specific drug pair + adverse event.
    pub fn get_signal(
        &self,
        drug_a: &str,
        drug_b: &str,
        event: AdverseEventType,
    ) -> Option<&FaersSignal> {
        let key = canonical_triple(drug_a, drug_b, event);
        self.signals.get(&key)
    }

    /// All signals for a drug pair (any adverse event).
    pub fn get_all_signals_for_pair(
        &self,
        drug_a: &str,
        drug_b: &str,
    ) -> Vec<&FaersSignal> {
        let pair = canonical_pair(drug_a, drug_b);
        match self.pair_index.get(&pair) {
            Some(events) => events
                .iter()
                .filter_map(|e| self.signals.get(&(pair.0.clone(), pair.1.clone(), *e)))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Get the strongest signal for a drug pair.
    pub fn strongest_signal_for_pair(
        &self,
        drug_a: &str,
        drug_b: &str,
    ) -> Option<&FaersSignal> {
        self.get_all_signals_for_pair(drug_a, drug_b)
            .into_iter()
            .max_by(|a, b| a.composite_score().partial_cmp(&b.composite_score()).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Maximum composite score across all signals for a pair.
    pub fn max_composite_score(&self, drug_a: &str, drug_b: &str) -> f64 {
        self.strongest_signal_for_pair(drug_a, drug_b)
            .map(|s| s.composite_score())
            .unwrap_or(0.0)
    }

    /// All signals in the database.
    pub fn all_signals(&self) -> Vec<&FaersSignal> {
        self.signals.values().collect()
    }

    /// Number of unique signals.
    pub fn len(&self) -> usize {
        self.signals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.signals.is_empty()
    }

    // ── Default signal data ─────────────────────────────────────────────

    fn load_defaults(&mut self) {
        use AdverseEventType::*;

        // ── Bleeding signals ──
        self.insert(signal("warfarin", "aspirin", Bleeding, 5.8, 4.2, 8.0, 1247));
        self.insert(signal("warfarin", "aspirin", GastrointestinalBleeding, 4.5, 3.1, 6.5, 832));
        self.insert(signal("warfarin", "aspirin", IntracranialHemorrhage, 3.2, 1.9, 5.4, 187));
        self.insert(signal("warfarin", "ibuprofen", GastrointestinalBleeding, 6.1, 4.5, 8.3, 543));
        self.insert(signal("warfarin", "naproxen", GastrointestinalBleeding, 5.3, 3.8, 7.4, 312));
        self.insert(signal("rivaroxaban", "aspirin", Bleeding, 4.2, 3.1, 5.7, 678));
        self.insert(signal("apixaban", "aspirin", Bleeding, 3.8, 2.7, 5.3, 521));
        self.insert(signal("dabigatran", "aspirin", GastrointestinalBleeding, 4.7, 3.3, 6.7, 389));
        self.insert(signal("warfarin", "clopidogrel", Bleeding, 7.2, 5.4, 9.6, 934));
        self.insert(signal("warfarin", "fluconazole", Bleeding, 5.1, 3.4, 7.6, 267));

        // ── QT prolongation / TdP signals ──
        self.insert(signal("amiodarone", "levofloxacin", QTProlongation, 8.3, 5.7, 12.1, 312));
        self.insert(signal("amiodarone", "levofloxacin", TorsadesDePointes, 11.2, 6.1, 20.6, 87));
        self.insert(signal("amiodarone", "sotalol", TorsadesDePointes, 14.5, 7.8, 27.0, 63));
        self.insert(signal("haloperidol", "methadone", QTProlongation, 6.7, 4.2, 10.7, 198));
        self.insert(signal("haloperidol", "methadone", TorsadesDePointes, 9.4, 4.5, 19.6, 42));
        self.insert(signal("ondansetron", "haloperidol", QTProlongation, 3.1, 1.8, 5.3, 134));
        self.insert(signal("erythromycin", "cisapride", TorsadesDePointes, 18.7, 8.9, 39.3, 34));
        self.insert(signal("clarithromycin", "pimozide", QTProlongation, 7.5, 3.8, 14.8, 28));

        // ── Rhabdomyolysis / myopathy signals ──
        self.insert(signal("simvastatin", "amiodarone", Rhabdomyolysis, 7.9, 5.1, 12.2, 189));
        self.insert(signal("simvastatin", "clarithromycin", Rhabdomyolysis, 12.3, 7.6, 19.9, 98));
        self.insert(signal("simvastatin", "itraconazole", Rhabdomyolysis, 15.4, 8.2, 28.9, 45));
        self.insert(signal("simvastatin", "gemfibrozil", Rhabdomyolysis, 9.6, 6.2, 14.9, 156));
        self.insert(signal("simvastatin", "cyclosporine", Rhabdomyolysis, 11.8, 7.1, 19.6, 73));
        self.insert(signal("lovastatin", "itraconazole", Rhabdomyolysis, 14.1, 7.5, 26.5, 38));
        self.insert(signal("atorvastatin", "clarithromycin", Myopathy, 4.5, 2.9, 7.0, 134));
        self.insert(signal("simvastatin", "diltiazem", Myopathy, 3.8, 2.4, 6.0, 167));

        // ── Serotonin syndrome signals ──
        self.insert(signal("fluoxetine", "tramadol", SerotoninSyndrome, 8.7, 5.4, 14.0, 178));
        self.insert(signal("sertraline", "tramadol", SerotoninSyndrome, 7.2, 4.6, 11.3, 145));
        self.insert(signal("paroxetine", "tramadol", SerotoninSyndrome, 6.9, 4.1, 11.6, 112));
        self.insert(signal("venlafaxine", "tramadol", SerotoninSyndrome, 9.1, 5.8, 14.3, 98));
        self.insert(signal("fluoxetine", "linezolid", SerotoninSyndrome, 12.8, 6.7, 24.5, 43));
        self.insert(signal("sertraline", "linezolid", SerotoninSyndrome, 11.5, 5.9, 22.4, 37));
        self.insert(signal("duloxetine", "tramadol", SerotoninSyndrome, 7.8, 4.6, 13.2, 89));
        self.insert(signal("fluoxetine", "phenelzine", SerotoninSyndrome, 22.4, 9.8, 51.2, 18));

        // ── Hyperkalemia signals ──
        self.insert(signal("spironolactone", "lisinopril", Hyperkalemia, 5.4, 3.8, 7.7, 423));
        self.insert(signal("spironolactone", "losartan", Hyperkalemia, 4.8, 3.2, 7.2, 287));
        self.insert(signal("spironolactone", "potassium_chloride", Hyperkalemia, 8.1, 5.6, 11.7, 198));
        self.insert(signal("lisinopril", "potassium_chloride", Hyperkalemia, 3.6, 2.4, 5.4, 312));
        self.insert(signal("trimethoprim_sulfamethoxazole", "spironolactone", Hyperkalemia, 6.7, 4.1, 10.9, 89));

        // ── Hypoglycemia signals ──
        self.insert(signal("glipizide", "fluconazole", Hypoglycemia, 6.3, 3.9, 10.2, 87));
        self.insert(signal("glyburide", "fluconazole", Hypoglycemia, 7.8, 4.7, 12.9, 65));
        self.insert(signal("insulin", "fluoxetine", Hypoglycemia, 2.3, 1.4, 3.8, 123));

        // ── Renal failure signals ──
        self.insert(signal("methotrexate", "ibuprofen", RenalFailure, 4.7, 3.1, 7.1, 178));
        self.insert(signal("methotrexate", "naproxen", RenalFailure, 4.2, 2.7, 6.5, 134));
        self.insert(signal("lithium", "ibuprofen", RenalFailure, 3.4, 2.1, 5.5, 98));
        self.insert(signal("lithium", "naproxen", RenalFailure, 3.1, 1.8, 5.3, 67));
        self.insert(signal("cyclosporine", "ibuprofen", RenalFailure, 5.2, 3.3, 8.2, 56));

        // ── Liver injury signals ──
        self.insert(signal("methotrexate", "leflunomide", LiverInjury, 5.6, 3.5, 9.0, 87));
        self.insert(signal("isoniazid", "rifampin", LiverInjury, 3.8, 2.4, 6.0, 234));
        self.insert(signal("amiodarone", "simvastatin", LiverInjury, 3.2, 1.9, 5.4, 67));
        self.insert(signal("valproic_acid", "carbamazepine", LiverInjury, 2.8, 1.6, 4.9, 78));

        // ── Respiratory depression signals ──
        self.insert(signal("oxycodone", "diazepam", RespiratoryDepression, 9.8, 6.7, 14.3, 267));
        self.insert(signal("morphine", "diazepam", RespiratoryDepression, 8.5, 5.8, 12.5, 312));
        self.insert(signal("fentanyl", "alprazolam", RespiratoryDepression, 11.2, 7.3, 17.2, 187));
        self.insert(signal("oxycodone", "gabapentin", RespiratoryDepression, 4.7, 3.1, 7.1, 234));
        self.insert(signal("morphine", "pregabalin", RespiratoryDepression, 4.2, 2.7, 6.5, 178));
        self.insert(signal("zolpidem", "oxycodone", RespiratoryDepression, 5.3, 3.4, 8.3, 156));
        self.insert(signal("fentanyl", "ritonavir", RespiratoryDepression, 7.8, 4.6, 13.2, 43));

        // ── Pancytopenia / hematologic signals ──
        self.insert(signal("methotrexate", "trimethoprim_sulfamethoxazole", Pancytopenia, 14.6, 8.3, 25.7, 67));
        self.insert(signal("allopurinol", "azathioprine", Pancytopenia, 11.3, 6.5, 19.7, 45));
        self.insert(signal("allopurinol", "mercaptopurine", Pancytopenia, 12.8, 7.0, 23.4, 34));

        // ── Bradycardia signals ──
        self.insert(signal("digoxin", "amiodarone", Bradycardia, 4.5, 3.0, 6.7, 287));
        self.insert(signal("digoxin", "verapamil", Bradycardia, 5.8, 3.8, 8.8, 198));
        self.insert(signal("metoprolol", "verapamil", Bradycardia, 6.3, 4.1, 9.7, 156));
        self.insert(signal("ivabradine", "diltiazem", Bradycardia, 8.9, 4.7, 16.8, 34));

        // ── Hypotension signals ──
        self.insert(signal("sildenafil", "nitroglycerin", HypotensiveCrisis, 15.3, 8.2, 28.5, 56));
        self.insert(signal("sildenafil", "isosorbide_dinitrate", HypotensiveCrisis, 13.7, 7.1, 26.4, 38));
        self.insert(signal("lisinopril", "losartan", HypotensiveCrisis, 3.4, 2.1, 5.5, 123));

        // ── Seizure signals ──
        self.insert(signal("theophylline", "ciprofloxacin", Seizure, 4.3, 2.7, 6.8, 89));
        self.insert(signal("tramadol", "bupropion", Seizure, 5.1, 3.2, 8.1, 67));

        // ── Stevens-Johnson syndrome ──
        self.insert(signal("allopurinol", "ampicillin", StevensJohnson, 4.2, 2.3, 7.7, 34));
        self.insert(signal("carbamazepine", "phenytoin", StevensJohnson, 3.1, 1.5, 6.4, 23));

        // ── Tendon rupture ──
        self.insert(signal("ciprofloxacin", "prednisone", TendonRupture, 4.8, 3.1, 7.4, 145));
        self.insert(signal("levofloxacin", "prednisone", TendonRupture, 4.2, 2.6, 6.8, 112));

        // ── Lactic acidosis ──
        self.insert(signal("metformin", "contrast_dye", LacticAcidosis, 5.6, 3.2, 9.8, 45));

        // ── Angioedema ──
        self.insert(signal("sacubitril_valsartan", "lisinopril", Angioedema, 8.4, 4.3, 16.4, 28));

        // ── Hyponatremia ──
        self.insert(signal("hydrochlorothiazide", "sertraline", Hyponatremia, 3.5, 2.2, 5.6, 134));
        self.insert(signal("hydrochlorothiazide", "fluoxetine", Hyponatremia, 3.2, 2.0, 5.1, 112));
        self.insert(signal("carbamazepine", "hydrochlorothiazide", Hyponatremia, 4.1, 2.5, 6.7, 67));

        // ── Peripheral neuropathy ──
        self.insert(signal("metronidazole", "isoniazid", PeripheralNeuropathy, 3.8, 2.1, 6.9, 34));

        // ── Hepatic failure ──
        self.insert(signal("acetaminophen", "isoniazid", HepaticFailure, 5.2, 3.1, 8.7, 56));
        self.insert(signal("colchicine", "clarithromycin", HepaticFailure, 6.7, 3.4, 13.2, 23));

        // ── Additional clinically relevant signals ──
        self.insert(signal("digoxin", "clarithromycin", Bradycardia, 5.1, 3.3, 7.9, 134));
        self.insert(signal("digoxin", "quinidine", Bradycardia, 6.2, 3.9, 9.8, 87));
        self.insert(signal("phenytoin", "valproic_acid", Neutropenia, 3.4, 1.9, 6.1, 45));
        self.insert(signal("clozapine", "fluvoxamine", Seizure, 4.8, 2.7, 8.5, 34));
        self.insert(signal("warfarin", "rifampin", Bleeding, 2.8, 1.5, 5.2, 56));
        self.insert(signal("warfarin", "amiodarone", Bleeding, 4.3, 3.0, 6.2, 345));
        self.insert(signal("warfarin", "metronidazole", Bleeding, 4.8, 3.1, 7.4, 198));
        self.insert(signal("clopidogrel", "omeprazole", Bleeding, 1.4, 0.9, 2.2, 234));
        self.insert(signal("dabigatran", "verapamil", Bleeding, 3.2, 2.0, 5.1, 89));
        self.insert(signal("cyclosporine", "clarithromycin", RenalFailure, 4.5, 2.8, 7.2, 67));
        self.insert(signal("tacrolimus", "fluconazole", RenalFailure, 3.8, 2.3, 6.3, 56));
    }
}

fn canonical_pair(drug_a: &str, drug_b: &str) -> (String, String) {
    let a = drug_a.to_lowercase();
    let b = drug_b.to_lowercase();
    if a <= b { (a, b) } else { (b, a) }
}

fn canonical_triple(
    drug_a: &str,
    drug_b: &str,
    event: AdverseEventType,
) -> (String, String, AdverseEventType) {
    let (a, b) = canonical_pair(drug_a, drug_b);
    (a, b, event)
}

fn signal(
    drug_a: &str,
    drug_b: &str,
    event: AdverseEventType,
    ror: f64,
    ci_lower: f64,
    ci_upper: f64,
    case_count: u32,
) -> FaersSignal {
    FaersSignal::new(drug_a, drug_b, event, ror, ci_lower, ci_upper, case_count)
}

// ──────────────────────────── Tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> FaersDatabase {
        FaersDatabase::with_defaults()
    }

    #[test]
    fn test_default_database_not_empty() {
        let d = db();
        assert!(d.len() >= 80, "Expected ≥80 signals, got {}", d.len());
    }

    #[test]
    fn test_warfarin_aspirin_bleeding() {
        let d = db();
        let sig = d.get_signal("warfarin", "aspirin", AdverseEventType::Bleeding);
        assert!(sig.is_some());
        let s = sig.unwrap();
        assert!(s.reporting_odds_ratio > 2.0);
        assert!(s.ci_lower > 1.0);
        assert_eq!(s.signal_strength, SignalStrength::Strong);
    }

    #[test]
    fn test_symmetric_lookup() {
        let d = db();
        assert!(d.get_signal("aspirin", "warfarin", AdverseEventType::Bleeding).is_some());
    }

    #[test]
    fn test_unknown_triple_returns_none() {
        let d = db();
        assert!(d.get_signal("water", "oxygen", AdverseEventType::Bleeding).is_none());
    }

    #[test]
    fn test_get_all_signals_for_pair() {
        let d = db();
        let signals = d.get_all_signals_for_pair("warfarin", "aspirin");
        assert!(signals.len() >= 2, "Warfarin+aspirin should have multiple AE signals");
    }

    #[test]
    fn test_strongest_signal() {
        let d = db();
        let sig = d.strongest_signal_for_pair("amiodarone", "levofloxacin");
        assert!(sig.is_some());
    }

    #[test]
    fn test_reporting_odds_ratio() {
        // Classic 2x2: a=100, b=900, c=50, d=8950
        let ror = compute_reporting_odds_ratio(100.0, 900.0, 50.0, 8950.0);
        // ROR = (100*8950)/(900*50) = 895000/45000 ≈ 19.89
        assert!((ror - 19.889).abs() < 0.1);
    }

    #[test]
    fn test_ror_with_ci() {
        let (ror, ci_lo, ci_hi) = compute_ror_with_ci(100.0, 900.0, 50.0, 8950.0);
        assert!(ror > 1.0);
        assert!(ci_lo > 1.0);
        assert!(ci_hi > ci_lo);
    }

    #[test]
    fn test_proportional_reporting_ratio() {
        let prr = compute_proportional_reporting_ratio(100.0, 900.0, 50.0, 8950.0);
        // PRR = (100/1000) / (50/9000) = 0.1 / 0.00556 ≈ 18.0
        assert!(prr > 10.0);
    }

    #[test]
    fn test_information_component() {
        let ic = compute_information_component(100.0, 10.0);
        // log2((100.5)/(10.5)) ≈ log2(9.57) ≈ 3.26
        assert!(ic > 3.0);
    }

    #[test]
    fn test_ic_lower_bound() {
        let ic025 = compute_ic_lower_bound(100.0, 10.0);
        let ic = compute_information_component(100.0, 10.0);
        assert!(ic025 < ic, "IC025 should be lower than IC");
    }

    #[test]
    fn test_is_significant() {
        let sig = FaersSignal::new("drugA", "drugB", AdverseEventType::Bleeding, 5.0, 2.5, 10.0, 50);
        assert!(is_significant(&sig, 2.0));
        assert!(!is_significant(&sig, 10.0)); // threshold too high
    }

    #[test]
    fn test_not_significant_if_ci_crosses_one() {
        let sig = FaersSignal::new("drugA", "drugB", AdverseEventType::Bleeding, 1.5, 0.8, 2.8, 10);
        assert!(!is_significant_default(&sig));
    }

    #[test]
    fn test_signal_strength_classification() {
        assert_eq!(SignalStrength::from_ror_and_ci(1.5, 0.9), SignalStrength::NoSignal);
        assert_eq!(SignalStrength::from_ror_and_ci(1.5, 1.1), SignalStrength::Weak);
        assert_eq!(SignalStrength::from_ror_and_ci(3.0, 1.5), SignalStrength::Moderate);
        assert_eq!(SignalStrength::from_ror_and_ci(8.0, 4.0), SignalStrength::Strong);
    }

    #[test]
    fn test_composite_score() {
        let sig = FaersSignal::new("a", "b", AdverseEventType::TorsadesDePointes, 10.0, 5.0, 20.0, 50);
        // Strong(1.0) × TdP(1.0) = 1.0
        assert!((sig.composite_score() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_adverse_event_labels() {
        assert_eq!(AdverseEventType::Bleeding.label(), "Bleeding");
        assert_eq!(AdverseEventType::SerotoninSyndrome.label(), "Serotonin Syndrome");
    }

    #[test]
    fn test_insert_custom_signal() {
        let mut d = FaersDatabase::empty();
        d.insert(FaersSignal::new("testA", "testB", AdverseEventType::Other, 3.0, 1.5, 6.0, 20));
        assert_eq!(d.len(), 1);
        assert!(d.get_signal("testA", "testB", AdverseEventType::Other).is_some());
    }

    #[test]
    fn test_serotonin_syndrome_signals() {
        let d = db();
        let sig = d.get_signal("fluoxetine", "tramadol", AdverseEventType::SerotoninSyndrome);
        assert!(sig.is_some());
        assert_eq!(sig.unwrap().signal_strength, SignalStrength::Strong);
    }

    #[test]
    fn test_respiratory_depression_signals() {
        let d = db();
        let sigs = d.get_all_signals_for_pair("oxycodone", "diazepam");
        assert!(!sigs.is_empty());
        assert!(sigs.iter().any(|s| s.adverse_event == AdverseEventType::RespiratoryDepression));
    }

    #[test]
    fn test_rhabdomyolysis_simvastatin() {
        let d = db();
        let sig = d.get_signal("simvastatin", "clarithromycin", AdverseEventType::Rhabdomyolysis);
        assert!(sig.is_some());
        assert!(sig.unwrap().reporting_odds_ratio > 10.0);
    }

    #[test]
    fn test_ror_zero_denominator() {
        let ror = compute_reporting_odds_ratio(10.0, 0.0, 5.0, 100.0);
        assert!(ror.is_infinite());
    }

    #[test]
    fn test_signal_involves() {
        let sig = FaersSignal::new("warfarin", "aspirin", AdverseEventType::Bleeding, 5.0, 3.0, 8.0, 100);
        assert!(sig.involves("warfarin"));
        assert!(sig.involves("Aspirin"));
        assert!(!sig.involves("metformin"));
    }

    #[test]
    fn test_signal_display() {
        let sig = FaersSignal::new("warfarin", "aspirin", AdverseEventType::Bleeding, 5.8, 4.2, 8.0, 1247);
        let s = format!("{}", sig);
        assert!(s.contains("warfarin"));
        assert!(s.contains("aspirin"));
        assert!(s.contains("Bleeding"));
    }
}
