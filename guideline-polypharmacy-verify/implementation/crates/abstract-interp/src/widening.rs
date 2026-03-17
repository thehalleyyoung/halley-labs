//! PK-aware widening and narrowing operators for pharmacokinetic abstract
//! interpretation.
//!
//! Standard interval widening pushes bounds to ±∞, which loses all
//! pharmacokinetic meaning.  The PK-aware operators in this module instead
//! widen to *physically meaningful* bounds derived from worst-case PK
//! parameters, therapeutic windows, and enzyme activity envelopes.
//!
//! # Operator hierarchy
//!
//! ```text
//! WideningOperator (trait)
//! ├── StandardWidening          — classic interval widening (→ ±∞)
//! ├── PkAwareWidening           — widen to [0, Css_max(worst-case)]
//! ├── ThresholdWidening         — snap to configurable threshold set
//! └── DelayedWidening           — join for k steps, then delegate
//!
//! NarrowingOperator (trait)
//! └── PkAwareNarrowing          — tighten toward therapeutic windows
//!
//! ConvergenceChecker            — tolerance-based fixed-point test
//! WideningStrategy (enum)       — factory for boxed operators
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::domain::{
    AbstractValue, ClinicalAbstractDomain, ConcentrationAbstractDomain,
    ConcentrationInterval, CypEnzyme, DrugId, EnzymeAbstractDomain,
    EnzymeActivityAbstractInterval, ProductDomain, TherapeuticWindow,
};

// ============================================================================
// Constants
// ============================================================================

/// Default maximum enzyme activity fold for widening (matches domain module).
const DEFAULT_ENZYME_MAX_FOLD: f64 = 10.0;

/// Default maximum induction fold used when no specific data is available.
const DEFAULT_MAX_INDUCTION_FOLD: f64 = 10.0;

/// Minimum enzyme activity (full inhibition).
const ENZYME_FULL_INHIBITION: f64 = 0.0;

// ============================================================================
// WorstCaseParameters — population variability factors
// ============================================================================

/// Population variability factors for computing worst-case PK envelopes.
///
/// Each factor is a multiplier applied to the base PK parameter to obtain
/// the extreme value in the population.  For example, a `clearance_lower_factor`
/// of 0.5 means the slowest metaboliser has 50 % of the reference clearance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorstCaseParameters {
    /// Lower bound multiplier for clearance (slow metabolisers).
    pub clearance_lower_factor: f64,
    /// Upper bound multiplier for clearance (fast metabolisers).
    pub clearance_upper_factor: f64,
    /// Lower bound multiplier for volume of distribution.
    pub vd_lower_factor: f64,
    /// Upper bound multiplier for volume of distribution.
    pub vd_upper_factor: f64,
    /// Lower bound multiplier for absorption rate.
    pub ka_lower_factor: f64,
    /// Upper bound multiplier for absorption rate.
    pub ka_upper_factor: f64,
    /// Lower bound multiplier for bioavailability.
    pub f_lower_factor: f64,
    /// Upper bound multiplier for bioavailability.
    pub f_upper_factor: f64,
}

impl Default for WorstCaseParameters {
    fn default() -> Self {
        WorstCaseParameters {
            clearance_lower_factor: 0.5,
            clearance_upper_factor: 2.0,
            vd_lower_factor: 0.7,
            vd_upper_factor: 1.5,
            ka_lower_factor: 0.5,
            ka_upper_factor: 2.0,
            f_lower_factor: 0.7,
            f_upper_factor: 1.3,
        }
    }
}

impl WorstCaseParameters {
    /// Compute the worst-case (maximum) steady-state concentration.
    ///
    /// Uses the combination of parameters that maximises C_ss:
    ///   C_ss_max = (F_max · Dose) / (CL_min · τ)
    ///
    /// This produces the highest possible average steady-state concentration
    /// for a patient in the worst-case population.
    pub fn worst_case_css_max(
        &self,
        base_cl: f64,
        base_vd: f64,
        _base_ka: f64,
        base_f: f64,
        dose: f64,
        interval: f64,
    ) -> f64 {
        let cl_min = (base_cl * self.clearance_lower_factor).max(f64::EPSILON);
        let _vd_max = base_vd * self.vd_upper_factor;
        let f_max = (base_f * self.f_upper_factor).min(1.0);
        let effective_dose = f_max * dose;

        if interval <= 0.0 || cl_min <= 0.0 {
            return 0.0;
        }

        // Average Css = F·D / (CL·τ)
        let css_avg = effective_dose / (cl_min * interval);

        // Peak approximation: account for accumulation with worst-case ke
        let ke = cl_min / (base_vd * self.vd_lower_factor).max(f64::EPSILON);
        let accumulation = if ke * interval > 0.0 {
            1.0 / (1.0 - (-ke * interval).exp())
        } else {
            1.0
        };
        let cmax = (effective_dose / (base_vd * self.vd_lower_factor).max(f64::EPSILON))
            * accumulation;

        // Return the greater of the two estimates as a conservative bound.
        css_avg.max(cmax)
    }

    /// Compute the worst-case (minimum) steady-state concentration.
    ///
    /// Uses parameters that minimise C_ss:
    ///   C_ss_min = (F_min · Dose) / (CL_max · τ)
    pub fn worst_case_css_min(
        &self,
        base_cl: f64,
        _base_vd: f64,
        _base_ka: f64,
        base_f: f64,
        dose: f64,
        interval: f64,
    ) -> f64 {
        let cl_max = base_cl * self.clearance_upper_factor;
        let f_min = (base_f * self.f_lower_factor).max(0.0);
        let effective_dose = f_min * dose;

        if interval <= 0.0 || cl_max <= 0.0 {
            return 0.0;
        }

        effective_dose / (cl_max * interval)
    }
}

// ============================================================================
// DrugPkEntry — per-drug PK data for PK-aware operators
// ============================================================================

/// Pharmacokinetic entry for a single drug, used by PK-aware widening.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugPkEntry {
    pub drug_id: DrugId,
    pub clearance_l_h: f64,
    pub volume_distribution_l: f64,
    pub absorption_rate_h: f64,
    pub bioavailability: f64,
    pub half_life_h: f64,
    pub therapeutic_window: Option<TherapeuticWindow>,
    pub metabolism_enzymes: Vec<(CypEnzyme, f64)>,
}

impl DrugPkEntry {
    pub fn new(drug_id: DrugId, clearance: f64, vd: f64) -> Self {
        let half_life = if clearance > 0.0 {
            (0.693 * vd) / clearance
        } else {
            f64::INFINITY
        };
        DrugPkEntry {
            drug_id,
            clearance_l_h: clearance,
            volume_distribution_l: vd,
            absorption_rate_h: 1.0,
            bioavailability: 1.0,
            half_life_h: half_life,
            therapeutic_window: None,
            metabolism_enzymes: Vec::new(),
        }
    }

    pub fn with_therapeutic_window(mut self, tw: TherapeuticWindow) -> Self {
        self.therapeutic_window = Some(tw);
        self
    }

    pub fn with_bioavailability(mut self, f: f64) -> Self {
        self.bioavailability = f.clamp(0.0, 1.0);
        self
    }

    pub fn with_absorption_rate(mut self, ka: f64) -> Self {
        self.absorption_rate_h = ka.max(0.0);
        self
    }

    pub fn with_enzyme(mut self, enzyme: CypEnzyme, fraction: f64) -> Self {
        self.metabolism_enzymes.push((enzyme, fraction.clamp(0.0, 1.0)));
        self
    }
}

// ============================================================================
// DrugDatabase — in-memory lookup for PK entries
// ============================================================================

/// In-memory database of per-drug PK parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugDatabase {
    entries: IndexMap<DrugId, DrugPkEntry>,
}

impl DrugDatabase {
    /// Create an empty database.
    pub fn new() -> Self {
        DrugDatabase {
            entries: IndexMap::new(),
        }
    }

    /// Insert a PK entry.
    pub fn insert(&mut self, entry: DrugPkEntry) {
        self.entries.insert(entry.drug_id.clone(), entry);
    }

    /// Look up a drug by identifier.
    pub fn lookup(&self, drug_id: &DrugId) -> Option<&DrugPkEntry> {
        self.entries.get(drug_id)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the database is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for DrugDatabase {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// WideningOperator trait
// ============================================================================

/// A widening operator for the `ProductDomain`.
///
/// Widening guarantees termination of ascending iteration chains by
/// extrapolating bounds.  Different implementations provide varying degrees
/// of precision.
pub trait WideningOperator: fmt::Debug + Send + Sync {
    /// Apply widening: given the *old* iterate and the *new* (post-transfer)
    /// iterate, return a sound over-approximation that ensures the ascending
    /// chain eventually stabilises.
    fn widen(&self, old: &ProductDomain, new: &ProductDomain) -> ProductDomain;
}

// ============================================================================
// StandardWidening
// ============================================================================

/// Classic interval widening that pushes unstable bounds toward ±∞.
///
/// For concentrations, uses `ConcentrationInterval::widen` with an empty
/// threshold set (pure jump to ±∞).  For enzyme activity, pushes lo → 0
/// and hi → `ENZYME_WIDEN_MAX`.  For the clinical domain, just joins.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardWidening;

impl StandardWidening {
    pub fn new() -> Self {
        StandardWidening
    }
}

impl Default for StandardWidening {
    fn default() -> Self {
        Self::new()
    }
}

impl WideningOperator for StandardWidening {
    fn widen(&self, old: &ProductDomain, new: &ProductDomain) -> ProductDomain {
        // Concentrations: standard widening with empty threshold set (→ ±∞)
        let conc = widen_concentration_standard(&old.concentrations, &new.concentrations);

        // Enzymes: standard AbstractValue widening (lo→0, hi→ENZYME_WIDEN_MAX)
        let enz = old.enzymes.widen(&new.enzymes);

        // Clinical: finite lattice, join suffices
        let clin = old.clinical.join(&new.clinical);

        ProductDomain {
            concentrations: conc,
            enzymes: enz,
            clinical: clin,
        }
    }
}

/// Standard concentration widening with empty thresholds (bounds jump to ±∞).
fn widen_concentration_standard(
    old: &ConcentrationAbstractDomain,
    new: &ConcentrationAbstractDomain,
) -> ConcentrationAbstractDomain {
    let empty_thresholds: Vec<f64> = Vec::new();
    let keys = all_drug_keys(old, new);
    let mut result = IndexMap::new();

    for k in keys {
        let a = old.get(&k);
        let b = new.get(&k);
        let widened = if a.is_bottom() {
            b
        } else if b.is_bottom() {
            a
        } else {
            a.widen(&b, &empty_thresholds)
        };
        if !widened.is_bottom() {
            result.insert(k, widened);
        }
    }

    ConcentrationAbstractDomain::new(result)
}

// ============================================================================
// PkAwareWidening
// ============================================================================

/// PK-aware widening that bounds intervals using worst-case pharmacokinetic
/// parameters instead of diverging to ±∞.
///
/// For each drug, the widening upper bound is capped at
/// `worst_case_css_max(...)` computed from population variability factors.
/// This yields tighter (more precise) invariants while remaining sound.
///
/// For enzyme activity, the widening envelope is `[0, max_induction_fold]`
/// (typically `[0, 10]`), representing the range from full inhibition to
/// maximum induction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PkAwareWidening {
    /// Per-drug pharmacokinetic parameters.
    pub drug_database: DrugDatabase,
    /// Population variability factors.
    pub worst_case: WorstCaseParameters,
    /// Per-drug dosing: `(dose_mg, interval_hours)`.
    pub dose_map: IndexMap<DrugId, (f64, f64)>,
    /// Maximum enzyme induction fold for enzyme widening envelope.
    pub max_induction_fold: f64,
}

impl PkAwareWidening {
    /// Create a new PK-aware widening operator.
    pub fn new(drug_database: DrugDatabase, worst_case: WorstCaseParameters) -> Self {
        PkAwareWidening {
            drug_database,
            worst_case,
            dose_map: IndexMap::new(),
            max_induction_fold: DEFAULT_MAX_INDUCTION_FOLD,
        }
    }

    /// Register a drug's dosing schedule for Css computation.
    pub fn with_dose(mut self, drug_id: DrugId, dose_mg: f64, interval_hours: f64) -> Self {
        self.dose_map.insert(drug_id, (dose_mg, interval_hours));
        self
    }

    /// Set the maximum induction fold for enzyme widening.
    pub fn with_max_induction_fold(mut self, fold: f64) -> Self {
        self.max_induction_fold = fold.max(1.0);
        self
    }

    /// Compute the worst-case Css_max for a drug, returning `None` if the drug
    /// or its dosing is not registered.
    fn css_max_for_drug(&self, drug_id: &DrugId) -> Option<f64> {
        let entry = self.drug_database.lookup(drug_id)?;
        let &(dose, interval) = self.dose_map.get(drug_id)?;
        let css_max = self.worst_case.worst_case_css_max(
            entry.clearance_l_h,
            entry.volume_distribution_l,
            entry.absorption_rate_h,
            entry.bioavailability,
            dose,
            interval,
        );
        Some(css_max)
    }

    /// Widen a single concentration interval using PK bounds.
    ///
    /// If a worst-case Css_max is available, the interval widens to
    /// `[0, css_max]` instead of `[−∞, +∞]`.  If PK data is missing,
    /// falls back to standard widening.
    fn widen_concentration_interval(
        &self,
        drug_id: &DrugId,
        old: ConcentrationInterval,
        new: ConcentrationInterval,
    ) -> ConcentrationInterval {
        if old.is_bottom() {
            return new;
        }
        if new.is_bottom() {
            return old;
        }

        match self.css_max_for_drug(drug_id) {
            Some(css_max) => {
                // PK-aware: widen toward [0, css_max]
                let lo = if new.lo < old.lo {
                    0.0_f64
                } else {
                    old.lo
                };
                let hi = if new.hi > old.hi {
                    css_max
                } else {
                    old.hi
                };
                // Ensure the result is at least as large as the join
                let lo = lo.min(old.lo.min(new.lo));
                let hi = hi.max(old.hi.max(new.hi)).min(css_max);
                // If new.hi already exceeds css_max, use css_max
                let hi = if new.hi > css_max { css_max } else { hi };
                ConcentrationInterval::new(lo.max(0.0), hi.max(lo.max(0.0)))
            }
            None => {
                // No PK data: fall back to standard widening (→ ±∞)
                let empty: Vec<f64> = Vec::new();
                old.widen(&new, &empty)
            }
        }
    }

    /// Widen a single enzyme activity interval using PK-aware bounds.
    fn widen_enzyme_interval(
        &self,
        old: &EnzymeActivityAbstractInterval,
        new: &EnzymeActivityAbstractInterval,
    ) -> EnzymeActivityAbstractInterval {
        if old.is_bottom() {
            return new.clone();
        }
        if new.is_bottom() {
            return old.clone();
        }

        let lo = if new.lo < old.lo {
            ENZYME_FULL_INHIBITION
        } else {
            old.lo
        };
        let hi = if new.hi > old.hi {
            self.max_induction_fold
        } else {
            old.hi
        };
        EnzymeActivityAbstractInterval::new(lo, hi)
    }
}

impl WideningOperator for PkAwareWidening {
    fn widen(&self, old: &ProductDomain, new: &ProductDomain) -> ProductDomain {
        // Concentrations: PK-aware widening
        let keys = all_drug_keys(&old.concentrations, &new.concentrations);
        let mut conc_result = IndexMap::new();
        for k in keys {
            let a = old.concentrations.get(&k);
            let b = new.concentrations.get(&k);
            let widened = self.widen_concentration_interval(&k, a, b);
            if !widened.is_bottom() {
                conc_result.insert(k, widened);
            }
        }
        let concentrations = ConcentrationAbstractDomain::new(conc_result);

        // Enzymes: PK-aware widening to [0, max_induction_fold]
        let enz_keys = all_enzyme_keys(&old.enzymes, &new.enzymes);
        let mut enz_result = IndexMap::new();
        for k in enz_keys {
            let a = old.enzymes.get(&k);
            let b = new.enzymes.get(&k);
            let widened = self.widen_enzyme_interval(&a, &b);
            enz_result.insert(k, widened);
        }
        let enzymes = EnzymeAbstractDomain::new(enz_result);

        // Clinical: join (finite lattice)
        let clinical = old.clinical.join(&new.clinical);

        ProductDomain {
            concentrations,
            enzymes,
            clinical,
        }
    }
}

// ============================================================================
// ThresholdWidening
// ============================================================================

/// Threshold-based widening that snaps unstable bounds to the nearest
/// configurable threshold rather than jumping directly to ±∞.
///
/// The threshold set should be sorted in ascending order and typically
/// includes pharmacokinetically meaningful concentrations (e.g. therapeutic
/// window boundaries, toxic thresholds).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdWidening {
    /// Sorted thresholds for concentration widening.
    pub thresholds: Vec<f64>,
    /// Sorted thresholds for enzyme activity widening.
    pub enzyme_thresholds: Vec<f64>,
}

impl ThresholdWidening {
    /// Create a new threshold widening with the given concentration thresholds.
    pub fn new(mut thresholds: Vec<f64>) -> Self {
        thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        thresholds.dedup();
        ThresholdWidening {
            thresholds,
            enzyme_thresholds: vec![0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 10.0],
        }
    }

    /// Create with both concentration and enzyme thresholds.
    pub fn with_enzyme_thresholds(mut self, mut thresholds: Vec<f64>) -> Self {
        thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        thresholds.dedup();
        self.enzyme_thresholds = thresholds;
        self
    }

    /// Default pharmacokinetic thresholds: common concentration breakpoints.
    pub fn default_pk_thresholds() -> Self {
        Self::new(vec![
            0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0,
            10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 5000.0,
        ])
    }

    /// Widen a single enzyme activity interval using thresholds.
    fn widen_enzyme_with_thresholds(
        &self,
        old: &EnzymeActivityAbstractInterval,
        new: &EnzymeActivityAbstractInterval,
    ) -> EnzymeActivityAbstractInterval {
        if old.is_bottom() {
            return new.clone();
        }
        if new.is_bottom() {
            return old.clone();
        }

        let lo = if new.lo < old.lo {
            self.enzyme_thresholds
                .iter()
                .rev()
                .find(|&&t| t <= new.lo)
                .copied()
                .unwrap_or(0.0)
        } else {
            old.lo
        };
        let hi = if new.hi > old.hi {
            self.enzyme_thresholds
                .iter()
                .find(|&&t| t >= new.hi)
                .copied()
                .unwrap_or(DEFAULT_ENZYME_MAX_FOLD)
        } else {
            old.hi
        };
        EnzymeActivityAbstractInterval::new(lo, hi)
    }
}

impl WideningOperator for ThresholdWidening {
    fn widen(&self, old: &ProductDomain, new: &ProductDomain) -> ProductDomain {
        // Concentrations: threshold widening
        let keys = all_drug_keys(&old.concentrations, &new.concentrations);
        let mut conc_result = IndexMap::new();
        for k in keys {
            let a = old.concentrations.get(&k);
            let b = new.concentrations.get(&k);
            let widened = if a.is_bottom() {
                b
            } else if b.is_bottom() {
                a
            } else {
                a.widen(&b, &self.thresholds)
            };
            if !widened.is_bottom() {
                conc_result.insert(k, widened);
            }
        }
        let concentrations = ConcentrationAbstractDomain::new(conc_result);

        // Enzymes: threshold widening
        let enz_keys = all_enzyme_keys(&old.enzymes, &new.enzymes);
        let mut enz_result = IndexMap::new();
        for k in enz_keys {
            let a = old.enzymes.get(&k);
            let b = new.enzymes.get(&k);
            let widened = self.widen_enzyme_with_thresholds(&a, &b);
            enz_result.insert(k, widened);
        }
        let enzymes = EnzymeAbstractDomain::new(enz_result);

        // Clinical: join
        let clinical = old.clinical.join(&new.clinical);

        ProductDomain {
            concentrations,
            enzymes,
            clinical,
        }
    }
}

// ============================================================================
// DelayedWidening
// ============================================================================

/// Widening that returns the plain join for the first `delay` iterations,
/// then delegates to an inner widening operator.
///
/// This is useful for allowing the iteration to gather information (via
/// joins) before applying the more aggressive widening extrapolation.
/// The iteration counter is stored in an `AtomicUsize` for thread-safe interior mutability.
#[derive(Debug)]
pub struct DelayedWidening {
    /// Number of initial iterations that use join instead of widening.
    pub delay: usize,
    /// The underlying widening operator applied after the delay.
    pub inner: Box<dyn WideningOperator + Send + Sync>,
    /// Current iteration count (interior mutability, thread-safe).
    iteration_count: AtomicUsize,
}

impl DelayedWidening {
    /// Create a delayed widening.
    ///
    /// The first `delay` applications return the join of old and new;
    /// subsequent applications delegate to `inner`.
    pub fn new(delay: usize, inner: Box<dyn WideningOperator + Send + Sync>) -> Self {
        DelayedWidening {
            delay,
            inner,
            iteration_count: AtomicUsize::new(0),
        }
    }

    /// Create with a standard inner widening.
    pub fn with_standard(delay: usize) -> Self {
        Self::new(delay, Box::new(StandardWidening::new()))
    }

    /// Reset the iteration counter to zero.
    pub fn reset(&self) {
        self.iteration_count.store(0, Ordering::Relaxed);
    }

    /// Current iteration count.
    pub fn current_iteration(&self) -> usize {
        self.iteration_count.load(Ordering::Relaxed)
    }
}

impl WideningOperator for DelayedWidening {
    fn widen(&self, old: &ProductDomain, new: &ProductDomain) -> ProductDomain {
        let count = self.iteration_count.fetch_add(1, Ordering::Relaxed);

        if count < self.delay {
            // Before the delay threshold: use join (no extrapolation)
            old.join(new)
        } else {
            // After the delay: delegate to inner widening operator
            self.inner.widen(old, new)
        }
    }
}

// ============================================================================
// NarrowingOperator trait
// ============================================================================

/// A narrowing operator for the `ProductDomain`.
///
/// Narrowing refines an over-approximated fixed point (obtained via widening)
/// by intersecting with transfer-function results, without losing soundness.
pub trait NarrowingOperator: fmt::Debug + Send + Sync {
    /// Apply narrowing: given the *old* (widened) iterate and the *new*
    /// (post-transfer) iterate, return a tighter approximation.
    fn narrow(&self, old: &ProductDomain, new: &ProductDomain) -> ProductDomain;
}

// ============================================================================
// PkAwareNarrowing
// ============================================================================

/// PK-aware narrowing that tightens concentration intervals toward known
/// therapeutic windows and enzyme activity toward physiological bounds.
///
/// If a drug has a known therapeutic window, the narrowing operator uses it
/// as an additional constraint: the concentration interval is intersected
/// with the window when the current iterate extends beyond it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PkAwareNarrowing {
    /// Per-drug therapeutic windows.
    pub therapeutic_windows: IndexMap<DrugId, TherapeuticWindow>,
    /// Maximum enzyme activity bound for narrowing.
    pub max_enzyme_activity: f64,
}

impl PkAwareNarrowing {
    /// Create a new PK-aware narrowing with no therapeutic windows.
    pub fn new() -> Self {
        PkAwareNarrowing {
            therapeutic_windows: IndexMap::new(),
            max_enzyme_activity: DEFAULT_ENZYME_MAX_FOLD,
        }
    }

    /// Register a drug's therapeutic window for narrowing guidance.
    pub fn with_therapeutic_window(mut self, drug_id: DrugId, tw: TherapeuticWindow) -> Self {
        self.therapeutic_windows.insert(drug_id, tw);
        self
    }

    /// Set the maximum enzyme activity for narrowing.
    pub fn with_max_enzyme_activity(mut self, max: f64) -> Self {
        self.max_enzyme_activity = max.max(1.0);
        self
    }

    /// Narrow a single concentration interval, guided by the therapeutic window
    /// if available.
    fn narrow_concentration_interval(
        &self,
        drug_id: &DrugId,
        old: ConcentrationInterval,
        new: ConcentrationInterval,
    ) -> ConcentrationInterval {
        if old.is_bottom() || new.is_bottom() {
            return ConcentrationInterval::bottom();
        }

        // Standard narrowing: replace ±∞ bounds with finite ones from `new`
        let mut narrowed = old.narrow(&new);

        // If a therapeutic window is known and the narrowed interval extends
        // well beyond it, further constrain towards a PK-meaningful range.
        // We use a factor of 3× the window width as a safety margin.
        if let Some(tw) = self.therapeutic_windows.get(drug_id) {
            let margin = tw.width() * 3.0;
            let pk_lo = (tw.min_concentration - margin).max(0.0);
            let pk_hi = tw.max_concentration + margin;
            let pk_bound = ConcentrationInterval::new(pk_lo, pk_hi);

            // Only narrow if the current interval is larger; never widen
            if !narrowed.is_bottom() && narrowed.width() > pk_bound.width() {
                let tightened = narrowed.meet(&pk_bound);
                if !tightened.is_bottom() {
                    narrowed = tightened;
                }
            }
        }

        narrowed
    }

    /// Narrow a single enzyme activity interval.
    fn narrow_enzyme_interval(
        &self,
        old: &EnzymeActivityAbstractInterval,
        new: &EnzymeActivityAbstractInterval,
    ) -> EnzymeActivityAbstractInterval {
        if old.is_bottom() || new.is_bottom() {
            return EnzymeActivityAbstractInterval::bottom();
        }

        // Standard narrowing: replace extreme bounds
        let lo = if old.lo == 0.0 && new.lo > 0.0 {
            new.lo
        } else {
            old.lo
        };
        let hi = if (old.hi - self.max_enzyme_activity).abs() < f64::EPSILON
            && new.hi < self.max_enzyme_activity
        {
            new.hi
        } else {
            old.hi
        };

        if lo > hi {
            EnzymeActivityAbstractInterval::bottom()
        } else {
            EnzymeActivityAbstractInterval::new(lo, hi)
        }
    }
}

impl Default for PkAwareNarrowing {
    fn default() -> Self {
        Self::new()
    }
}

impl NarrowingOperator for PkAwareNarrowing {
    fn narrow(&self, old: &ProductDomain, new: &ProductDomain) -> ProductDomain {
        // Concentrations: PK-aware narrowing
        let keys = all_drug_keys(&old.concentrations, &new.concentrations);
        let mut conc_result = IndexMap::new();
        for k in keys {
            let a = old.concentrations.get(&k);
            let b = new.concentrations.get(&k);
            let narrowed = self.narrow_concentration_interval(&k, a, b);
            if !narrowed.is_bottom() {
                conc_result.insert(k, narrowed);
            }
        }
        let concentrations = ConcentrationAbstractDomain::new(conc_result);

        // Enzymes: PK-aware narrowing
        let enz_keys = all_enzyme_keys(&old.enzymes, &new.enzymes);
        let mut enz_result = IndexMap::new();
        for k in enz_keys {
            let a = old.enzymes.get(&k);
            let b = new.enzymes.get(&k);
            let narrowed = self.narrow_enzyme_interval(&a, &b);
            enz_result.insert(k, narrowed);
        }
        let enzymes = EnzymeAbstractDomain::new(enz_result);

        // Clinical: standard narrowing (meet)
        let clinical = old.clinical.narrow(&new.clinical);

        ProductDomain {
            concentrations,
            enzymes,
            clinical,
        }
    }
}

// ============================================================================
// ConvergenceChecker
// ============================================================================

/// Checks whether the abstract iteration has converged by comparing
/// successive iterates within a tolerance.
///
/// Convergence is determined by checking that the maximum absolute change
/// across all concentration and enzyme activity interval bounds is less
/// than the configured tolerance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceChecker {
    /// Maximum allowed change in any interval bound for convergence.
    pub tolerance: f64,
}

impl ConvergenceChecker {
    /// Create a convergence checker with the given tolerance.
    pub fn new(tolerance: f64) -> Self {
        ConvergenceChecker {
            tolerance: tolerance.max(0.0),
        }
    }

    /// Default checker with tolerance 1e-6.
    pub fn default_checker() -> Self {
        Self::new(1e-6)
    }

    /// Check whether the iteration has converged.
    ///
    /// Returns `true` if the maximum change across all components is below
    /// the tolerance, or if both states are structurally equal.
    pub fn has_converged(&self, old: &ProductDomain, new: &ProductDomain) -> bool {
        if old.structurally_equal(new) {
            return true;
        }
        self.max_change(old, new) < self.tolerance
    }

    /// Compute the maximum absolute change across all concentration and
    /// enzyme activity interval bounds between two product domain states.
    pub fn max_change(&self, old: &ProductDomain, new: &ProductDomain) -> f64 {
        let mut max_delta = 0.0_f64;

        // Check concentration changes
        let conc_keys = all_drug_keys(&old.concentrations, &new.concentrations);
        for k in &conc_keys {
            let a = old.concentrations.get(k);
            let b = new.concentrations.get(k);
            let delta = interval_change(a.lo, a.hi, a.is_bottom(), b.lo, b.hi, b.is_bottom());
            max_delta = max_delta.max(delta);
        }

        // Check enzyme changes
        let enz_keys = all_enzyme_keys(&old.enzymes, &new.enzymes);
        for k in &enz_keys {
            let a = old.enzymes.get(k);
            let b = new.enzymes.get(k);
            let delta = interval_change(a.lo, a.hi, a.is_bottom(), b.lo, b.hi, b.is_bottom());
            max_delta = max_delta.max(delta);
        }

        // Clinical domain: discrete — any change is a big delta
        if old.clinical != new.clinical {
            max_delta = max_delta.max(f64::INFINITY);
        }

        max_delta
    }

    /// Compute per-drug concentration changes, useful for diagnostics.
    pub fn concentration_changes(
        &self,
        old: &ProductDomain,
        new: &ProductDomain,
    ) -> IndexMap<DrugId, f64> {
        let keys = all_drug_keys(&old.concentrations, &new.concentrations);
        let mut changes = IndexMap::new();
        for k in keys {
            let a = old.concentrations.get(&k);
            let b = new.concentrations.get(&k);
            let delta = interval_change(a.lo, a.hi, a.is_bottom(), b.lo, b.hi, b.is_bottom());
            changes.insert(k, delta);
        }
        changes
    }

    /// Compute per-enzyme activity changes, useful for diagnostics.
    pub fn enzyme_changes(
        &self,
        old: &ProductDomain,
        new: &ProductDomain,
    ) -> IndexMap<CypEnzyme, f64> {
        let keys = all_enzyme_keys(&old.enzymes, &new.enzymes);
        let mut changes = IndexMap::new();
        for k in keys {
            let a = old.enzymes.get(&k);
            let b = new.enzymes.get(&k);
            let delta = interval_change(a.lo, a.hi, a.is_bottom(), b.lo, b.hi, b.is_bottom());
            changes.insert(k, delta);
        }
        changes
    }
}

impl Default for ConvergenceChecker {
    fn default() -> Self {
        Self::default_checker()
    }
}

/// Compute the maximum absolute bound change between two intervals.
fn interval_change(
    lo1: f64, hi1: f64, is_bottom1: bool,
    lo2: f64, hi2: f64, is_bottom2: bool,
) -> f64 {
    match (is_bottom1, is_bottom2) {
        (true, true) => 0.0,
        (true, false) => {
            // Went from bottom to something: infinite change conceptually,
            // but report the width for practical comparison
            if hi2.is_finite() && lo2.is_finite() {
                (hi2 - lo2).abs().max(lo2.abs()).max(hi2.abs())
            } else {
                f64::INFINITY
            }
        }
        (false, true) => {
            if hi1.is_finite() && lo1.is_finite() {
                (hi1 - lo1).abs().max(lo1.abs()).max(hi1.abs())
            } else {
                f64::INFINITY
            }
        }
        (false, false) => {
            let lo_delta = if lo1.is_finite() && lo2.is_finite() {
                (lo1 - lo2).abs()
            } else if lo1 == lo2 {
                0.0
            } else {
                f64::INFINITY
            };
            let hi_delta = if hi1.is_finite() && hi2.is_finite() {
                (hi1 - hi2).abs()
            } else if hi1 == hi2 {
                0.0
            } else {
                f64::INFINITY
            };
            lo_delta.max(hi_delta)
        }
    }
}

// ============================================================================
// WideningStrategy enum
// ============================================================================

/// Strategy selector for creating boxed widening operators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WideningStrategy {
    /// Classic interval widening (bounds → ±∞).
    Standard,
    /// PK-aware widening using population PK envelopes.
    PkAware {
        drug_database: DrugDatabase,
        worst_case: WorstCaseParameters,
        dose_map: IndexMap<DrugId, (f64, f64)>,
        max_induction_fold: f64,
    },
    /// Threshold-based widening with configurable snap points.
    Threshold {
        thresholds: Vec<f64>,
        enzyme_thresholds: Vec<f64>,
    },
    /// Delayed widening: join for `delay` iterations, then apply `inner`.
    Delayed {
        delay: usize,
        inner: Box<WideningStrategy>,
    },
}

impl WideningStrategy {
    /// Instantiate a boxed `WideningOperator` from this strategy.
    pub fn create_operator(&self) -> Box<dyn WideningOperator + Send + Sync> {
        match self {
            WideningStrategy::Standard => Box::new(StandardWidening::new()),

            WideningStrategy::PkAware {
                drug_database,
                worst_case,
                dose_map,
                max_induction_fold,
            } => {
                let mut op = PkAwareWidening::new(drug_database.clone(), worst_case.clone());
                op.dose_map = dose_map.clone();
                op.max_induction_fold = *max_induction_fold;
                Box::new(op)
            }

            WideningStrategy::Threshold {
                thresholds,
                enzyme_thresholds,
            } => {
                let op = ThresholdWidening::new(thresholds.clone())
                    .with_enzyme_thresholds(enzyme_thresholds.clone());
                Box::new(op)
            }

            WideningStrategy::Delayed { delay, inner } => {
                let inner_op = inner.create_operator();
                Box::new(DelayedWidening::new(*delay, inner_op))
            }
        }
    }

    /// Shorthand for a PK-aware strategy with default worst-case parameters.
    pub fn pk_aware(db: DrugDatabase) -> Self {
        WideningStrategy::PkAware {
            drug_database: db,
            worst_case: WorstCaseParameters::default(),
            dose_map: IndexMap::new(),
            max_induction_fold: DEFAULT_MAX_INDUCTION_FOLD,
        }
    }

    /// Shorthand for the default threshold strategy.
    pub fn default_threshold() -> Self {
        let tw = ThresholdWidening::default_pk_thresholds();
        WideningStrategy::Threshold {
            thresholds: tw.thresholds,
            enzyme_thresholds: tw.enzyme_thresholds,
        }
    }

    /// Shorthand for a delayed strategy wrapping a standard widening.
    pub fn delayed_standard(delay: usize) -> Self {
        WideningStrategy::Delayed {
            delay,
            inner: Box::new(WideningStrategy::Standard),
        }
    }
}

// ============================================================================
// Utility helpers
// ============================================================================

/// Collect all drug keys present in either concentration domain.
fn all_drug_keys(
    a: &ConcentrationAbstractDomain,
    b: &ConcentrationAbstractDomain,
) -> Vec<DrugId> {
    let mut keys: Vec<DrugId> = a.drugs();
    for k in b.drugs() {
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    keys
}

/// Collect all enzyme keys present in either enzyme domain.
fn all_enzyme_keys(
    a: &EnzymeAbstractDomain,
    b: &EnzymeAbstractDomain,
) -> Vec<CypEnzyme> {
    let mut keys: Vec<CypEnzyme> = a.activities.keys().copied().collect();
    for &k in b.activities.keys() {
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    keys
}

// ============================================================================
// Display impls
// ============================================================================

impl fmt::Display for WorstCaseParameters {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WorstCase(CL:[{:.2}×,{:.2}×], Vd:[{:.2}×,{:.2}×])",
            self.clearance_lower_factor,
            self.clearance_upper_factor,
            self.vd_lower_factor,
            self.vd_upper_factor,
        )
    }
}

impl fmt::Display for StandardWidening {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StandardWidening")
    }
}

impl fmt::Display for PkAwareWidening {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PkAwareWidening({} drugs, max_fold={:.1})",
            self.dose_map.len(),
            self.max_induction_fold,
        )
    }
}

impl fmt::Display for ThresholdWidening {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ThresholdWidening({} thresholds)",
            self.thresholds.len(),
        )
    }
}

impl fmt::Display for DelayedWidening {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DelayedWidening(delay={}, iter={}, inner={:?})",
            self.delay,
            self.iteration_count.load(Ordering::Relaxed),
            self.inner,
        )
    }
}

impl fmt::Display for PkAwareNarrowing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PkAwareNarrowing({} windows)",
            self.therapeutic_windows.len(),
        )
    }
}

impl fmt::Display for ConvergenceChecker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ConvergenceChecker(tol={:.1e})", self.tolerance)
    }
}

impl fmt::Display for WideningStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WideningStrategy::Standard => write!(f, "Standard"),
            WideningStrategy::PkAware { .. } => write!(f, "PkAware"),
            WideningStrategy::Threshold { thresholds, .. } => {
                write!(f, "Threshold({})", thresholds.len())
            }
            WideningStrategy::Delayed { delay, inner } => {
                write!(f, "Delayed({}, {})", delay, inner)
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Test helpers
    // ------------------------------------------------------------------

    fn drug(name: &str) -> DrugId {
        DrugId::new(name)
    }

    /// Build a minimal ProductDomain with one drug concentration.
    fn single_drug_state(name: &str, lo: f64, hi: f64) -> ProductDomain {
        let mut pd = ProductDomain::initial();
        pd.set_concentration(drug(name), ConcentrationInterval::new(lo, hi));
        pd
    }

    /// Build a ProductDomain with one drug and one enzyme.
    fn drug_enzyme_state(
        name: &str,
        conc_lo: f64,
        conc_hi: f64,
        enzyme: CypEnzyme,
        enz_lo: f64,
        enz_hi: f64,
    ) -> ProductDomain {
        let mut pd = single_drug_state(name, conc_lo, conc_hi);
        pd.set_enzyme_activity(enzyme, EnzymeActivityAbstractInterval::new(enz_lo, enz_hi));
        pd
    }

    /// Build a PK-aware widening operator for a test drug.
    fn test_pk_widening(name: &str, cl: f64, vd: f64, dose: f64, interval: f64) -> PkAwareWidening {
        let mut db = DrugDatabase::new();
        db.insert(DrugPkEntry::new(drug(name), cl, vd));
        PkAwareWidening::new(db, WorstCaseParameters::default())
            .with_dose(drug(name), dose, interval)
    }

    // ------------------------------------------------------------------
    // StandardWidening tests
    // ------------------------------------------------------------------

    #[test]
    fn test_standard_widening_stable_bounds() {
        let op = StandardWidening::new();
        let old = single_drug_state("aspirin", 1.0, 5.0);
        let new = single_drug_state("aspirin", 1.0, 5.0);
        let result = op.widen(&old, &new);
        let iv = result.concentration(&drug("aspirin"));
        assert_eq!(iv.lo, 1.0);
        assert_eq!(iv.hi, 5.0);
    }

    #[test]
    fn test_standard_widening_growing_upper_bound() {
        let op = StandardWidening::new();
        let old = single_drug_state("warfarin", 2.0, 8.0);
        let new = single_drug_state("warfarin", 2.0, 10.0);
        let result = op.widen(&old, &new);
        let iv = result.concentration(&drug("warfarin"));
        // Upper bound grew → standard widening pushes to +∞
        assert_eq!(iv.lo, 2.0);
        assert!(iv.hi.is_infinite() && iv.hi > 0.0);
    }

    #[test]
    fn test_standard_widening_shrinking_lower_bound() {
        let op = StandardWidening::new();
        let old = single_drug_state("digoxin", 3.0, 10.0);
        let new = single_drug_state("digoxin", 1.0, 10.0);
        let result = op.widen(&old, &new);
        let iv = result.concentration(&drug("digoxin"));
        // Lower bound shrank → standard widening pushes to -∞
        assert!(iv.lo.is_infinite() && iv.lo < 0.0);
        assert_eq!(iv.hi, 10.0);
    }

    #[test]
    fn test_standard_widening_enzyme_grows() {
        let op = StandardWidening::new();
        let old = drug_enzyme_state("a", 1.0, 5.0, CypEnzyme::CYP3A4, 0.5, 1.0);
        let new = drug_enzyme_state("a", 1.0, 5.0, CypEnzyme::CYP3A4, 0.5, 1.5);
        let result = op.widen(&old, &new);
        let enz = result.enzyme_activity(&CypEnzyme::CYP3A4);
        // hi grew → widen to ENZYME_WIDEN_MAX
        assert_eq!(enz.lo, 0.5);
        assert!((enz.hi - 10.0).abs() < f64::EPSILON);
    }

    // ------------------------------------------------------------------
    // PkAwareWidening tests
    // ------------------------------------------------------------------

    #[test]
    fn test_pk_aware_widening_caps_at_css_max() {
        let op = test_pk_widening("metoprolol", 10.0, 200.0, 100.0, 12.0);
        let css_max = op.css_max_for_drug(&drug("metoprolol")).unwrap();
        assert!(css_max > 0.0, "Css_max should be positive");

        let old = single_drug_state("metoprolol", 5.0, 20.0);
        let new = single_drug_state("metoprolol", 5.0, 25.0);
        let result = op.widen(&old, &new);
        let iv = result.concentration(&drug("metoprolol"));

        // Upper bound grew → PK-aware caps at css_max, not +∞
        assert!(iv.hi.is_finite(), "PK-aware should not go to +∞");
        assert!(
            iv.hi <= css_max + f64::EPSILON,
            "Upper bound {} should be <= Css_max {}",
            iv.hi,
            css_max,
        );
    }

    #[test]
    fn test_pk_aware_widening_unknown_drug_falls_back() {
        let op = test_pk_widening("known", 10.0, 200.0, 100.0, 12.0);
        let old = single_drug_state("unknown", 2.0, 8.0);
        let new = single_drug_state("unknown", 2.0, 12.0);
        let result = op.widen(&old, &new);
        let iv = result.concentration(&drug("unknown"));
        // Unknown drug → falls back to standard widening → +∞
        assert!(iv.hi.is_infinite());
    }

    #[test]
    fn test_pk_aware_enzyme_widening() {
        let mut op = test_pk_widening("drug", 10.0, 200.0, 100.0, 12.0);
        op.max_induction_fold = 8.0;

        let old = drug_enzyme_state("drug", 5.0, 10.0, CypEnzyme::CYP2D6, 0.3, 1.0);
        let new = drug_enzyme_state("drug", 5.0, 10.0, CypEnzyme::CYP2D6, 0.3, 1.5);
        let result = op.widen(&old, &new);
        let enz = result.enzyme_activity(&CypEnzyme::CYP2D6);
        // hi grew → caps at max_induction_fold (8.0)
        assert!((enz.hi - 8.0).abs() < f64::EPSILON);
    }

    // ------------------------------------------------------------------
    // ThresholdWidening tests
    // ------------------------------------------------------------------

    #[test]
    fn test_threshold_widening_snaps_to_threshold() {
        let op = ThresholdWidening::new(vec![0.0, 5.0, 10.0, 20.0, 50.0, 100.0]);
        let old = single_drug_state("phenytoin", 3.0, 15.0);
        let new = single_drug_state("phenytoin", 3.0, 18.0);
        let result = op.widen(&old, &new);
        let iv = result.concentration(&drug("phenytoin"));
        // hi grew from 15 → 18, next threshold >= 18 is 20
        assert_eq!(iv.lo, 3.0);
        assert!((iv.hi - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_threshold_widening_no_matching_threshold() {
        let op = ThresholdWidening::new(vec![0.0, 5.0, 10.0]);
        let old = single_drug_state("drug", 1.0, 8.0);
        let new = single_drug_state("drug", 1.0, 12.0);
        let result = op.widen(&old, &new);
        let iv = result.concentration(&drug("drug"));
        // hi grew past all thresholds → jumps to +∞
        assert!(iv.hi.is_infinite());
    }

    // ------------------------------------------------------------------
    // DelayedWidening tests
    // ------------------------------------------------------------------

    #[test]
    fn test_delayed_widening_joins_before_delay() {
        let op = DelayedWidening::with_standard(3);
        let old = single_drug_state("a", 2.0, 6.0);
        let new = single_drug_state("a", 2.0, 8.0);

        // Iterations 0, 1, 2 should join (not widen)
        for i in 0..3 {
            let result = op.widen(&old, &new);
            let iv = result.concentration(&drug("a"));
            assert!(
                iv.hi.is_finite(),
                "iteration {} should join, not widen",
                i,
            );
            assert!(
                (iv.hi - 8.0).abs() < f64::EPSILON,
                "join should give hi=8.0, got {}",
                iv.hi,
            );
        }

        // Iteration 3 should widen
        let result = op.widen(&old, &new);
        let iv = result.concentration(&drug("a"));
        assert!(
            iv.hi.is_infinite(),
            "iteration 3 should apply standard widening",
        );
    }

    #[test]
    fn test_delayed_widening_reset() {
        let op = DelayedWidening::with_standard(2);
        let old = single_drug_state("b", 1.0, 5.0);
        let new = single_drug_state("b", 1.0, 7.0);

        // Use up delay
        let _ = op.widen(&old, &new);
        let _ = op.widen(&old, &new);
        assert_eq!(op.current_iteration(), 2);

        // Reset
        op.reset();
        assert_eq!(op.current_iteration(), 0);

        // Should join again
        let result = op.widen(&old, &new);
        let iv = result.concentration(&drug("b"));
        assert!(iv.hi.is_finite());
    }

    // ------------------------------------------------------------------
    // PkAwareNarrowing tests
    // ------------------------------------------------------------------

    #[test]
    fn test_pk_aware_narrowing_tightens_with_window() {
        let op = PkAwareNarrowing::new()
            .with_therapeutic_window(drug("warfarin"), TherapeuticWindow::new(1.0, 4.0));

        // Suppose widening gave us a very wide interval
        let old = single_drug_state("warfarin", 0.0, 100.0);
        let new = single_drug_state("warfarin", 0.5, 10.0);
        let result = op.narrow(&old, &new);
        let iv = result.concentration(&drug("warfarin"));

        // Should be tighter than [0, 100] via standard narrowing and PK guidance
        assert!(iv.hi < 100.0, "narrowing should reduce upper bound");
        assert!(iv.lo >= 0.0);
    }

    #[test]
    fn test_pk_aware_narrowing_enzyme_tightens() {
        let op = PkAwareNarrowing::new().with_max_enzyme_activity(10.0);
        let old = drug_enzyme_state("d", 1.0, 5.0, CypEnzyme::CYP3A4, 0.0, 10.0);
        let new = drug_enzyme_state("d", 1.0, 5.0, CypEnzyme::CYP3A4, 0.2, 3.5);
        let result = op.narrow(&old, &new);
        let enz = result.enzyme_activity(&CypEnzyme::CYP3A4);
        // lo was 0 (widened extreme) and new.lo is 0.2 → narrow picks up 0.2
        assert!((enz.lo - 0.2).abs() < f64::EPSILON);
        // hi was 10.0 (widened extreme) and new.hi is 3.5 → narrow picks up 3.5
        assert!((enz.hi - 3.5).abs() < f64::EPSILON);
    }

    // ------------------------------------------------------------------
    // ConvergenceChecker tests
    // ------------------------------------------------------------------

    #[test]
    fn test_convergence_identical_states() {
        let checker = ConvergenceChecker::new(1e-6);
        let state = single_drug_state("x", 2.0, 8.0);
        assert!(checker.has_converged(&state, &state));
        assert!((checker.max_change(&state, &state)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_convergence_small_change() {
        let checker = ConvergenceChecker::new(0.01);
        let old = single_drug_state("x", 2.0, 8.0);
        let new = single_drug_state("x", 2.0, 8.005);
        assert!(checker.has_converged(&old, &new));
    }

    #[test]
    fn test_convergence_large_change() {
        let checker = ConvergenceChecker::new(0.01);
        let old = single_drug_state("x", 2.0, 8.0);
        let new = single_drug_state("x", 2.0, 9.0);
        assert!(!checker.has_converged(&old, &new));
        let delta = checker.max_change(&old, &new);
        assert!((delta - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_convergence_per_drug_changes() {
        let checker = ConvergenceChecker::new(0.1);
        let mut old = single_drug_state("a", 1.0, 5.0);
        old.set_concentration(drug("b"), ConcentrationInterval::new(10.0, 20.0));
        let mut new = single_drug_state("a", 1.0, 5.05);
        new.set_concentration(drug("b"), ConcentrationInterval::new(10.0, 22.0));

        let changes = checker.concentration_changes(&old, &new);
        assert!(changes[&drug("a")] < 0.1);
        assert!((changes[&drug("b")] - 2.0).abs() < f64::EPSILON);
    }

    // ------------------------------------------------------------------
    // WideningStrategy tests
    // ------------------------------------------------------------------

    #[test]
    fn test_strategy_creates_standard() {
        let op = WideningStrategy::Standard.create_operator();
        let old = single_drug_state("x", 1.0, 5.0);
        let new = single_drug_state("x", 1.0, 7.0);
        let result = op.widen(&old, &new);
        let iv = result.concentration(&drug("x"));
        assert!(iv.hi.is_infinite());
    }

    #[test]
    fn test_strategy_creates_delayed() {
        let strategy = WideningStrategy::delayed_standard(2);
        let op = strategy.create_operator();
        let old = single_drug_state("y", 1.0, 5.0);
        let new = single_drug_state("y", 1.0, 7.0);

        // First two: join
        let r1 = op.widen(&old, &new);
        assert!(r1.concentration(&drug("y")).hi.is_finite());
        let r2 = op.widen(&old, &new);
        assert!(r2.concentration(&drug("y")).hi.is_finite());

        // Third: widen
        let r3 = op.widen(&old, &new);
        assert!(r3.concentration(&drug("y")).hi.is_infinite());
    }

    #[test]
    fn test_strategy_creates_threshold() {
        let strategy = WideningStrategy::default_threshold();
        let op = strategy.create_operator();
        let old = single_drug_state("z", 1.0, 8.0);
        let new = single_drug_state("z", 1.0, 12.0);
        let result = op.widen(&old, &new);
        let iv = result.concentration(&drug("z"));
        // Should snap to a threshold, not +∞
        assert!(iv.hi.is_finite(), "threshold widening should snap, not go to ∞");
        assert!(iv.hi >= 12.0);
    }

    // ------------------------------------------------------------------
    // Edge-case tests
    // ------------------------------------------------------------------

    #[test]
    fn test_widening_with_bottom_old() {
        let op = StandardWidening::new();
        let old = ProductDomain::initial();
        let new = single_drug_state("d", 3.0, 7.0);
        let result = op.widen(&old, &new);
        let iv = result.concentration(&drug("d"));
        // old is bottom for this drug → result = new
        assert!((iv.lo - 3.0).abs() < f64::EPSILON);
        assert!((iv.hi - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_widening_with_bottom_new() {
        let op = StandardWidening::new();
        let old = single_drug_state("d", 3.0, 7.0);
        let new = ProductDomain::initial();
        let result = op.widen(&old, &new);
        let iv = result.concentration(&drug("d"));
        // new is bottom for this drug → result = old
        assert!((iv.lo - 3.0).abs() < f64::EPSILON);
        assert!((iv.hi - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_worst_case_parameters_css_max() {
        let wc = WorstCaseParameters::default();
        let css = wc.worst_case_css_max(10.0, 200.0, 1.0, 0.8, 100.0, 12.0);
        assert!(css > 0.0, "Css_max must be positive");
        assert!(css.is_finite(), "Css_max must be finite");

        // Css_min should be less than Css_max
        let css_min = wc.worst_case_css_min(10.0, 200.0, 1.0, 0.8, 100.0, 12.0);
        assert!(css_min < css, "Css_min ({}) should be < Css_max ({})", css_min, css);
    }

    #[test]
    fn test_multiple_drugs_widening() {
        let mut db = DrugDatabase::new();
        db.insert(DrugPkEntry::new(drug("a"), 5.0, 100.0));
        db.insert(DrugPkEntry::new(drug("b"), 20.0, 300.0));
        let op = PkAwareWidening::new(db, WorstCaseParameters::default())
            .with_dose(drug("a"), 50.0, 8.0)
            .with_dose(drug("b"), 200.0, 12.0);

        let mut old = ProductDomain::initial();
        old.set_concentration(drug("a"), ConcentrationInterval::new(1.0, 5.0));
        old.set_concentration(drug("b"), ConcentrationInterval::new(3.0, 10.0));

        let mut new = ProductDomain::initial();
        new.set_concentration(drug("a"), ConcentrationInterval::new(1.0, 7.0));
        new.set_concentration(drug("b"), ConcentrationInterval::new(3.0, 15.0));

        let result = op.widen(&old, &new);
        let iv_a = result.concentration(&drug("a"));
        let iv_b = result.concentration(&drug("b"));

        assert!(iv_a.hi.is_finite(), "drug a should be PK-bounded");
        assert!(iv_b.hi.is_finite(), "drug b should be PK-bounded");
        assert!(iv_a.hi >= 7.0, "widened must contain new");
        assert!(iv_b.hi >= 15.0, "widened must contain new");
    }

    #[test]
    fn test_narrowing_without_therapeutic_window() {
        let op = PkAwareNarrowing::new();
        let old = single_drug_state("generic", 0.0, 50.0);
        let new = single_drug_state("generic", 2.0, 30.0);
        let result = op.narrow(&old, &new);
        let iv = result.concentration(&drug("generic"));
        // Standard narrowing: old.lo=0 is not -∞ so stays; old.hi=50 is not +∞ so stays
        // Effectively just gets the old narrowed with new
        assert!(iv.lo >= 0.0);
        assert!(iv.hi <= 50.0);
    }

    #[test]
    fn test_convergence_enzyme_change() {
        let checker = ConvergenceChecker::new(0.001);
        let old = drug_enzyme_state("x", 1.0, 5.0, CypEnzyme::CYP2C9, 0.5, 1.0);
        let new = drug_enzyme_state("x", 1.0, 5.0, CypEnzyme::CYP2C9, 0.5, 1.5);
        assert!(!checker.has_converged(&old, &new));

        let enz_changes = checker.enzyme_changes(&old, &new);
        assert!((enz_changes[&CypEnzyme::CYP2C9] - 0.5).abs() < f64::EPSILON);
    }
}
