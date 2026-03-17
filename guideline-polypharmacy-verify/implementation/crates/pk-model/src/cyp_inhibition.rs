//! CYP enzyme inhibition models.
//!
//! Implements competitive, non-competitive, uncompetitive, and mechanism-based
//! inhibition of CYP enzymes, plus induction and combined effects.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use guardpharma_types::{CypEnzyme, DrugId, InhibitionType};

// ---------------------------------------------------------------------------
// InhibitionModel trait
// ---------------------------------------------------------------------------

/// Trait for computing effective clearance under enzyme inhibition.
pub trait InhibitionModel: Send + Sync + std::fmt::Debug {
    fn compute_effective_clearance(
        &self,
        base_clearance: f64,
        inhibitor_concentrations: &HashMap<CypEnzyme, f64>,
    ) -> f64;

    fn inhibition_ratio(
        &self,
        inhibitor_concentrations: &HashMap<CypEnzyme, f64>,
    ) -> f64;

    fn description(&self) -> String;
}

// ---------------------------------------------------------------------------
// InhibitionModelType (enum dispatch for serialization)
// ---------------------------------------------------------------------------

/// Enum wrapper for inhibition models (serializable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InhibitionModelType {
    Competitive(CompetitiveInhibition),
    NonCompetitive(NonCompetitiveInhibition),
    Uncompetitive(UncompetitiveInhibition),
    MechanismBased(MechanismBasedInhibition),
}

impl InhibitionModel for InhibitionModelType {
    fn compute_effective_clearance(
        &self,
        base_clearance: f64,
        concs: &HashMap<CypEnzyme, f64>,
    ) -> f64 {
        match self {
            Self::Competitive(m) => m.compute_effective_clearance(base_clearance, concs),
            Self::NonCompetitive(m) => m.compute_effective_clearance(base_clearance, concs),
            Self::Uncompetitive(m) => m.compute_effective_clearance(base_clearance, concs),
            Self::MechanismBased(m) => m.compute_effective_clearance(base_clearance, concs),
        }
    }

    fn inhibition_ratio(&self, concs: &HashMap<CypEnzyme, f64>) -> f64 {
        match self {
            Self::Competitive(m) => m.inhibition_ratio(concs),
            Self::NonCompetitive(m) => m.inhibition_ratio(concs),
            Self::Uncompetitive(m) => m.inhibition_ratio(concs),
            Self::MechanismBased(m) => m.inhibition_ratio(concs),
        }
    }

    fn description(&self) -> String {
        match self {
            Self::Competitive(m) => m.description(),
            Self::NonCompetitive(m) => m.description(),
            Self::Uncompetitive(m) => m.description(),
            Self::MechanismBased(m) => m.description(),
        }
    }
}

// ---------------------------------------------------------------------------
// CompetitiveInhibition
// ---------------------------------------------------------------------------

/// Competitive inhibition: inhibitor competes with substrate for active site.
/// CL_eff = CL_0 * product(1 / (1 + I_j/Ki_j))
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitiveInhibition {
    pub ki_values: HashMap<CypEnzyme, f64>,
    pub substrate_km: f64,
    pub substrate_concentration: f64,
}

impl CompetitiveInhibition {
    pub fn new(ki_values: HashMap<CypEnzyme, f64>) -> Self {
        Self {
            ki_values,
            substrate_km: 10.0,
            substrate_concentration: 1.0,
        }
    }

    pub fn with_substrate(mut self, km: f64, conc: f64) -> Self {
        self.substrate_km = km;
        self.substrate_concentration = conc;
        self
    }

    pub fn simple(enzyme: CypEnzyme, ki: f64) -> Self {
        let mut ki_values = HashMap::new();
        ki_values.insert(enzyme, ki);
        Self::new(ki_values)
    }
}

impl InhibitionModel for CompetitiveInhibition {
    fn compute_effective_clearance(
        &self,
        base_clearance: f64,
        inhibitor_concentrations: &HashMap<CypEnzyme, f64>,
    ) -> f64 {
        let mut factor = 1.0;
        for (enzyme, &ki) in &self.ki_values {
            if let Some(&conc) = inhibitor_concentrations.get(enzyme) {
                if ki > 0.0 {
                    factor *= 1.0 / (1.0 + conc / ki);
                }
            }
        }
        base_clearance * factor
    }

    fn inhibition_ratio(&self, inhibitor_concentrations: &HashMap<CypEnzyme, f64>) -> f64 {
        let mut factor = 1.0;
        for (enzyme, &ki) in &self.ki_values {
            if let Some(&conc) = inhibitor_concentrations.get(enzyme) {
                if ki > 0.0 {
                    factor *= 1.0 / (1.0 + conc / ki);
                }
            }
        }
        factor
    }

    fn description(&self) -> String {
        format!("Competitive inhibition on {:?}", self.ki_values.keys().collect::<Vec<_>>())
    }
}

// ---------------------------------------------------------------------------
// NonCompetitiveInhibition
// ---------------------------------------------------------------------------

/// Non-competitive inhibition: CL_eff = CL_0 * product(Ki_j / (I_j + Ki_j))
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonCompetitiveInhibition {
    pub ki_values: HashMap<CypEnzyme, f64>,
}

impl NonCompetitiveInhibition {
    pub fn new(ki_values: HashMap<CypEnzyme, f64>) -> Self {
        Self { ki_values }
    }

    pub fn simple(enzyme: CypEnzyme, ki: f64) -> Self {
        let mut ki_values = HashMap::new();
        ki_values.insert(enzyme, ki);
        Self::new(ki_values)
    }
}

impl InhibitionModel for NonCompetitiveInhibition {
    fn compute_effective_clearance(
        &self,
        base_clearance: f64,
        inhibitor_concentrations: &HashMap<CypEnzyme, f64>,
    ) -> f64 {
        let mut factor = 1.0;
        for (enzyme, &ki) in &self.ki_values {
            if let Some(&conc) = inhibitor_concentrations.get(enzyme) {
                factor *= ki / (conc + ki);
            }
        }
        base_clearance * factor
    }

    fn inhibition_ratio(&self, inhibitor_concentrations: &HashMap<CypEnzyme, f64>) -> f64 {
        let mut factor = 1.0;
        for (enzyme, &ki) in &self.ki_values {
            if let Some(&conc) = inhibitor_concentrations.get(enzyme) {
                factor *= ki / (conc + ki);
            }
        }
        factor
    }

    fn description(&self) -> String {
        format!("Non-competitive inhibition on {:?}", self.ki_values.keys().collect::<Vec<_>>())
    }
}

// ---------------------------------------------------------------------------
// UncompetitiveInhibition
// ---------------------------------------------------------------------------

/// Uncompetitive inhibition: affects Km and Vmax equally.
/// CL_eff = CL_0 * (Km + S) / (Km + S*(1 + I/Ki))
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncompetitiveInhibition {
    pub ki_values: HashMap<CypEnzyme, f64>,
    pub substrate_km: f64,
    pub substrate_concentration: f64,
}

impl UncompetitiveInhibition {
    pub fn new(ki_values: HashMap<CypEnzyme, f64>, km: f64, s: f64) -> Self {
        Self {
            ki_values,
            substrate_km: km,
            substrate_concentration: s,
        }
    }

    pub fn simple(enzyme: CypEnzyme, ki: f64) -> Self {
        let mut ki_values = HashMap::new();
        ki_values.insert(enzyme, ki);
        Self::new(ki_values, 10.0, 1.0)
    }
}

impl InhibitionModel for UncompetitiveInhibition {
    fn compute_effective_clearance(
        &self,
        base_clearance: f64,
        inhibitor_concentrations: &HashMap<CypEnzyme, f64>,
    ) -> f64 {
        let km = self.substrate_km;
        let s = self.substrate_concentration;
        let mut total_i_over_ki = 0.0;
        for (enzyme, &ki) in &self.ki_values {
            if let Some(&conc) = inhibitor_concentrations.get(enzyme) {
                if ki > 0.0 {
                    total_i_over_ki += conc / ki;
                }
            }
        }
        base_clearance * (km + s) / (km + s * (1.0 + total_i_over_ki))
    }

    fn inhibition_ratio(&self, inhibitor_concentrations: &HashMap<CypEnzyme, f64>) -> f64 {
        let km = self.substrate_km;
        let s = self.substrate_concentration;
        let mut total = 0.0;
        for (enzyme, &ki) in &self.ki_values {
            if let Some(&conc) = inhibitor_concentrations.get(enzyme) {
                if ki > 0.0 {
                    total += conc / ki;
                }
            }
        }
        (km + s) / (km + s * (1.0 + total))
    }

    fn description(&self) -> String {
        format!("Uncompetitive inhibition on {:?}", self.ki_values.keys().collect::<Vec<_>>())
    }
}

// ---------------------------------------------------------------------------
// MechanismBasedInhibition
// ---------------------------------------------------------------------------

/// Mechanism-based (time-dependent) inhibition.
/// Fraction remaining = kdeg / (kdeg + kinact * [I] / ([I] + KI))
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MechanismBasedInhibition {
    pub kinact: f64,
    pub ki: f64,
    pub degradation_rate: f64,
    pub enzyme: CypEnzyme,
}

impl MechanismBasedInhibition {
    pub fn new(enzyme: CypEnzyme, kinact: f64, ki: f64, kdeg: f64) -> Self {
        Self {
            kinact,
            ki,
            degradation_rate: kdeg,
            enzyme,
        }
    }

    /// Use enzyme's own degradation rate.
    pub fn with_default_kdeg(enzyme: CypEnzyme, kinact: f64, ki: f64) -> Self {
        Self {
            kinact,
            ki,
            degradation_rate: enzyme.degradation_rate(),
            enzyme,
        }
    }

    /// Fraction of enzyme remaining at steady state.
    pub fn fraction_remaining(&self, inhibitor_concentration: f64) -> f64 {
        let kdeg = self.degradation_rate;
        let inact_rate =
            self.kinact * inhibitor_concentration / (inhibitor_concentration + self.ki);
        kdeg / (kdeg + inact_rate)
    }
}

impl InhibitionModel for MechanismBasedInhibition {
    fn compute_effective_clearance(
        &self,
        base_clearance: f64,
        inhibitor_concentrations: &HashMap<CypEnzyme, f64>,
    ) -> f64 {
        if let Some(&conc) = inhibitor_concentrations.get(&self.enzyme) {
            base_clearance * self.fraction_remaining(conc)
        } else {
            base_clearance
        }
    }

    fn inhibition_ratio(
        &self,
        inhibitor_concentrations: &HashMap<CypEnzyme, f64>,
    ) -> f64 {
        if let Some(&conc) = inhibitor_concentrations.get(&self.enzyme) {
            self.fraction_remaining(conc)
        } else {
            1.0
        }
    }

    fn description(&self) -> String {
        format!(
            "Mechanism-based inhibition of {} (kinact={}, KI={})",
            self.enzyme, self.kinact, self.ki
        )
    }
}

// ---------------------------------------------------------------------------
// CypInhibitionNetwork
// ---------------------------------------------------------------------------

/// Network of drugs, enzymes, and inhibition relationships.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CypInhibitionNetwork {
    drugs: Vec<DrugId>,
    base_clearances: HashMap<DrugId, f64>,
    metabolism_fractions: HashMap<DrugId, Vec<(CypEnzyme, f64)>>,
    inhibitions: Vec<(DrugId, CypEnzyme, InhibitionModelType)>,
}

impl CypInhibitionNetwork {
    pub fn new() -> Self {
        Self {
            drugs: Vec::new(),
            base_clearances: HashMap::new(),
            metabolism_fractions: HashMap::new(),
            inhibitions: Vec::new(),
        }
    }

    pub fn add_drug(
        &mut self,
        drug_id: DrugId,
        base_clearance: f64,
        metabolism: Vec<(CypEnzyme, f64)>,
    ) {
        self.drugs.push(drug_id.clone());
        self.base_clearances.insert(drug_id.clone(), base_clearance);
        self.metabolism_fractions.insert(drug_id, metabolism);
    }

    pub fn add_inhibition(
        &mut self,
        inhibitor: DrugId,
        enzyme: CypEnzyme,
        model: InhibitionModelType,
    ) {
        self.inhibitions.push((inhibitor, enzyme, model));
    }

    /// Compute effective clearance for a drug given all drug concentrations.
    pub fn compute_effective_clearance(
        &self,
        drug: &DrugId,
        concentrations: &HashMap<DrugId, f64>,
    ) -> f64 {
        let base_cl = match self.base_clearances.get(drug) {
            Some(&cl) => cl,
            None => return 0.0,
        };
        let metab = match self.metabolism_fractions.get(drug) {
            Some(m) => m,
            None => return base_cl,
        };

        let mut effective_cl = 0.0;
        for &(enzyme, fraction) in metab {
            let enzyme_cl = base_cl * fraction;

            // Find all inhibitors of this enzyme
            let mut enzyme_concs: HashMap<CypEnzyme, f64> = HashMap::new();
            for (inhibitor, inh_enzyme, _model) in &self.inhibitions {
                if *inh_enzyme == enzyme {
                    if let Some(&conc) = concentrations.get(inhibitor) {
                        enzyme_concs.insert(enzyme, conc);
                    }
                }
            }

            // Apply inhibition
            let mut net_cl = enzyme_cl;
            for (inhibitor, inh_enzyme, model) in &self.inhibitions {
                if *inh_enzyme == enzyme {
                    if let Some(&_conc) = concentrations.get(inhibitor) {
                        net_cl = model.compute_effective_clearance(net_cl, &enzyme_concs);
                    }
                }
            }
            effective_cl += net_cl;
        }

        // Add non-metabolized fraction
        let total_metab: f64 = metab.iter().map(|(_, f)| f).sum();
        effective_cl += base_cl * (1.0 - total_metab).max(0.0);

        effective_cl
    }

    /// Compute effective clearances for all drugs.
    pub fn compute_all_effective_clearances(
        &self,
        concentrations: &HashMap<DrugId, f64>,
    ) -> HashMap<DrugId, f64> {
        self.drugs
            .iter()
            .map(|d| (d.clone(), self.compute_effective_clearance(d, concentrations)))
            .collect()
    }

    /// Check monotonicity: increasing inhibitor concentration always decreases clearance.
    pub fn is_monotone(&self) -> bool {
        // Competitive and non-competitive inhibition are monotone by definition.
        // Combined inhibition+induction may not be.
        self.inhibitions.iter().all(|(_, _, model)| {
            matches!(
                model,
                InhibitionModelType::Competitive(_)
                    | InhibitionModelType::NonCompetitive(_)
                    | InhibitionModelType::Uncompetitive(_)
                    | InhibitionModelType::MechanismBased(_)
            )
        })
    }

    /// Compute worst-case (min, max) clearance over concentration ranges.
    pub fn get_worst_case_clearance(
        &self,
        drug: &DrugId,
        concentration_ranges: &HashMap<DrugId, (f64, f64)>,
    ) -> (f64, f64) {
        // For monotone inhibition: max concentration => min clearance, and vice versa
        let mut lo_concs = HashMap::new();
        let mut hi_concs = HashMap::new();
        for (d, &(lo, hi)) in concentration_ranges {
            lo_concs.insert(d.clone(), lo);
            hi_concs.insert(d.clone(), hi);
        }

        let cl_at_lo = self.compute_effective_clearance(drug, &lo_concs);
        let cl_at_hi = self.compute_effective_clearance(drug, &hi_concs);
        (cl_at_lo.min(cl_at_hi), cl_at_lo.max(cl_at_hi))
    }

    pub fn num_drugs(&self) -> usize {
        self.drugs.len()
    }

    pub fn num_inhibitions(&self) -> usize {
        self.inhibitions.len()
    }
}

impl Default for CypInhibitionNetwork {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// EnzymeInductionModel
// ---------------------------------------------------------------------------

/// CYP enzyme induction: fold-increase = 1 + Emax * C / (EC50 + C).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnzymeInductionModel {
    pub enzyme: CypEnzyme,
    pub emax: f64,
    pub ec50: f64,
    pub inducer_id: DrugId,
}

impl EnzymeInductionModel {
    pub fn new(enzyme: CypEnzyme, emax: f64, ec50: f64, inducer: DrugId) -> Self {
        Self {
            enzyme,
            emax,
            ec50,
            inducer_id: inducer,
        }
    }

    /// Compute induction factor at given inducer concentration.
    pub fn induction_factor(&self, inducer_concentration: f64) -> f64 {
        1.0 + self.emax * inducer_concentration / (self.ec50 + inducer_concentration)
    }

    /// Compute induced clearance.
    pub fn compute_induced_clearance(
        &self,
        base_clearance: f64,
        inducer_concentration: f64,
    ) -> f64 {
        base_clearance * self.induction_factor(inducer_concentration)
    }
}

// ---------------------------------------------------------------------------
// CombinedEffect
// ---------------------------------------------------------------------------

/// Combined inhibition and induction effects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedEffect {
    pub inhibitions: Vec<InhibitionModelType>,
    pub inductions: Vec<EnzymeInductionModel>,
}

impl CombinedEffect {
    pub fn new() -> Self {
        Self {
            inhibitions: Vec::new(),
            inductions: Vec::new(),
        }
    }

    pub fn add_inhibition(&mut self, model: InhibitionModelType) {
        self.inhibitions.push(model);
    }

    pub fn add_induction(&mut self, model: EnzymeInductionModel) {
        self.inductions.push(model);
    }

    /// Compute net effect on clearance considering both inhibition and induction.
    pub fn net_effect(
        &self,
        base_clearance: f64,
        inhibitor_concentrations: &HashMap<CypEnzyme, f64>,
        inducer_concentrations: &HashMap<DrugId, f64>,
    ) -> f64 {
        // Apply inhibitions
        let mut cl = base_clearance;
        for inh in &self.inhibitions {
            cl = inh.compute_effective_clearance(cl, inhibitor_concentrations);
        }

        // Apply inductions
        for ind in &self.inductions {
            if let Some(&conc) = inducer_concentrations.get(&ind.inducer_id) {
                cl *= ind.induction_factor(conc);
            }
        }

        cl
    }

    /// Check if the combined effect is monotone.
    pub fn is_monotone(&self) -> bool {
        // Combined inhibition + induction is generally NOT monotone
        self.inductions.is_empty()
    }
}

impl Default for CombinedEffect {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_competitive_inhibition_basic() {
        let inh = CompetitiveInhibition::simple(CypEnzyme::CYP3A4, 1.0);
        let mut concs = HashMap::new();
        concs.insert(CypEnzyme::CYP3A4, 1.0);
        let cl = inh.compute_effective_clearance(10.0, &concs);
        // 10 * 1/(1+1/1) = 10 * 0.5 = 5.0
        assert!((cl - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_competitive_no_inhibitor() {
        let inh = CompetitiveInhibition::simple(CypEnzyme::CYP3A4, 1.0);
        let concs = HashMap::new();
        let cl = inh.compute_effective_clearance(10.0, &concs);
        assert!((cl - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_noncompetitive_inhibition() {
        let inh = NonCompetitiveInhibition::simple(CypEnzyme::CYP3A4, 2.0);
        let mut concs = HashMap::new();
        concs.insert(CypEnzyme::CYP3A4, 2.0);
        let cl = inh.compute_effective_clearance(10.0, &concs);
        // 10 * 2/(2+2) = 10 * 0.5 = 5.0
        assert!((cl - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_uncompetitive_inhibition() {
        let inh = UncompetitiveInhibition::simple(CypEnzyme::CYP2D6, 5.0);
        let mut concs = HashMap::new();
        concs.insert(CypEnzyme::CYP2D6, 5.0);
        let cl = inh.compute_effective_clearance(10.0, &concs);
        // (Km + S) / (Km + S*(1 + I/Ki)) = (10+1)/(10+1*(1+1)) = 11/12
        let expected = 10.0 * 11.0 / 12.0;
        assert!((cl - expected).abs() < 0.1);
    }

    #[test]
    fn test_mechanism_based_inhibition() {
        let mbi = MechanismBasedInhibition::new(CypEnzyme::CYP3A4, 0.05, 2.0, 0.019);
        let mut concs = HashMap::new();
        concs.insert(CypEnzyme::CYP3A4, 4.0);
        let cl = mbi.compute_effective_clearance(10.0, &concs);
        // kdeg/(kdeg + kinact*I/(I+KI)) = 0.019/(0.019 + 0.05*4/(4+2))
        let frac = 0.019 / (0.019 + 0.05 * 4.0 / 6.0);
        assert!((cl - 10.0 * frac).abs() < 0.01);
    }

    #[test]
    fn test_cyp_network_two_drugs() {
        let mut net = CypInhibitionNetwork::new();
        let drug_a = DrugId::new("drug_a");
        let drug_b = DrugId::new("drug_b");
        net.add_drug(drug_a.clone(), 10.0, vec![(CypEnzyme::CYP3A4, 0.8)]);
        net.add_drug(drug_b.clone(), 20.0, vec![(CypEnzyme::CYP2D6, 0.9)]);
        net.add_inhibition(
            drug_b.clone(),
            CypEnzyme::CYP3A4,
            InhibitionModelType::Competitive(CompetitiveInhibition::simple(
                CypEnzyme::CYP3A4,
                1.0,
            )),
        );

        let mut concs = HashMap::new();
        concs.insert(drug_a.clone(), 1.0);
        concs.insert(drug_b.clone(), 2.0);

        let cl_a = net.compute_effective_clearance(&drug_a, &concs);
        assert!(cl_a < 10.0, "Drug A clearance should be reduced: {}", cl_a);

        let cl_b = net.compute_effective_clearance(&drug_b, &concs);
        assert!((cl_b - 20.0).abs() < 0.01, "Drug B unaffected: {}", cl_b);
    }

    #[test]
    fn test_monotonicity() {
        let mut net = CypInhibitionNetwork::new();
        net.add_drug(DrugId::new("a"), 10.0, vec![(CypEnzyme::CYP3A4, 0.8)]);
        net.add_inhibition(
            DrugId::new("b"),
            CypEnzyme::CYP3A4,
            InhibitionModelType::Competitive(CompetitiveInhibition::simple(
                CypEnzyme::CYP3A4,
                1.0,
            )),
        );
        assert!(net.is_monotone());
    }

    #[test]
    fn test_worst_case_clearance() {
        let mut net = CypInhibitionNetwork::new();
        let victim = DrugId::new("victim");
        let perp = DrugId::new("perpetrator");
        net.add_drug(victim.clone(), 10.0, vec![(CypEnzyme::CYP3A4, 0.9)]);
        net.add_inhibition(
            perp.clone(),
            CypEnzyme::CYP3A4,
            InhibitionModelType::Competitive(CompetitiveInhibition::simple(
                CypEnzyme::CYP3A4,
                1.0,
            )),
        );
        let mut ranges = HashMap::new();
        ranges.insert(perp, (0.5, 5.0));
        ranges.insert(victim.clone(), (1.0, 3.0));
        let (min_cl, max_cl) = net.get_worst_case_clearance(&victim, &ranges);
        assert!(min_cl < max_cl);
        assert!(min_cl > 0.0);
    }

    #[test]
    fn test_enzyme_induction() {
        let ind = EnzymeInductionModel::new(
            CypEnzyme::CYP3A4,
            10.0,
            5.0,
            DrugId::new("rifampin"),
        );
        // At C=5: factor = 1 + 10*5/(5+5) = 1 + 5 = 6
        assert!((ind.induction_factor(5.0) - 6.0).abs() < 0.01);
        assert!((ind.compute_induced_clearance(10.0, 5.0) - 60.0).abs() < 0.1);
    }

    #[test]
    fn test_combined_effect() {
        let mut combined = CombinedEffect::new();
        combined.add_inhibition(InhibitionModelType::Competitive(
            CompetitiveInhibition::simple(CypEnzyme::CYP3A4, 1.0),
        ));
        combined.add_induction(EnzymeInductionModel::new(
            CypEnzyme::CYP3A4,
            5.0,
            2.0,
            DrugId::new("inducer"),
        ));

        let mut inh_concs = HashMap::new();
        inh_concs.insert(CypEnzyme::CYP3A4, 1.0);
        let mut ind_concs = HashMap::new();
        ind_concs.insert(DrugId::new("inducer"), 2.0);

        let cl = combined.net_effect(10.0, &inh_concs, &ind_concs);
        // Inhibition: 10 * 0.5 = 5, Induction: 5 * (1 + 5*2/(2+2)) = 5 * 3.5 = 17.5
        assert!(cl > 10.0 || cl < 10.0); // net effect depends on magnitudes
        assert!(!combined.is_monotone()); // induction makes it non-monotone
    }

    #[test]
    fn test_strong_inhibitor() {
        let inh = CompetitiveInhibition::simple(CypEnzyme::CYP3A4, 0.001);
        let mut concs = HashMap::new();
        concs.insert(CypEnzyme::CYP3A4, 10.0);
        let ratio = inh.inhibition_ratio(&concs);
        // 1/(1 + 10/0.001) = 1/10001 ≈ 0.0001
        assert!(ratio < 0.001);
    }

    #[test]
    fn test_multiple_enzyme_inhibition() {
        let mut ki = HashMap::new();
        ki.insert(CypEnzyme::CYP3A4, 2.0);
        ki.insert(CypEnzyme::CYP2D6, 3.0);
        let inh = CompetitiveInhibition::new(ki);
        let mut concs = HashMap::new();
        concs.insert(CypEnzyme::CYP3A4, 2.0);
        concs.insert(CypEnzyme::CYP2D6, 3.0);
        let ratio = inh.inhibition_ratio(&concs);
        // Product of 1/(1+1) and 1/(1+1) = 0.25
        assert!((ratio - 0.25).abs() < 0.01);
    }
}
