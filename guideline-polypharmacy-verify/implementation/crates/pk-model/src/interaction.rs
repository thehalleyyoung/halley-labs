//! Drug interaction network and analysis.
//!
//! Models pharmacokinetic and pharmacodynamic interactions,
//! cascade detection, monotonicity checking, and severity classification.

use std::collections::{HashMap, HashSet, VecDeque};
use serde::{Deserialize, Serialize};
use guardpharma_types::{DrugId, CypEnzyme, InhibitionType, Severity};
use guardpharma_types::concentration::ConcentrationInterval;

// ---------------------------------------------------------------------------
// InteractionMechanism
// ---------------------------------------------------------------------------

/// Mechanism by which two drugs interact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InteractionMechanism {
    /// CYP-mediated metabolic inhibition.
    CypInhibition {
        enzyme: CypEnzyme,
        inhibition_type: InhibitionType,
        ki: f64,
    },
    /// CYP enzyme induction.
    CypInduction {
        enzyme: CypEnzyme,
        emax: f64,
        ec50: f64,
    },
    /// Competitive plasma-protein binding displacement.
    ProteinBindingDisplacement {
        binding_protein: String,
        displacement_fraction: f64,
    },
    /// P-glycoprotein inhibition affecting absorption/efflux.
    PgpInhibition { ki: f64 },
    /// P-glycoprotein induction.
    PgpInduction { emax: f64, ec50: f64 },
    /// Renal transporter inhibition (OAT, OCT, etc.).
    RenalTransporterInhibition {
        transporter: String,
        ki: f64,
    },
    /// pH-dependent absorption interaction.
    GastrointestinalPhChange { delta_ph: f64 },
    /// Pharmacodynamic synergy.
    PdSynergy { effect_multiplier: f64 },
    /// Pharmacodynamic antagonism.
    PdAntagonism { effect_reduction: f64 },
    /// QT prolongation additive risk.
    QtProlongation { delta_qtc_ms: f64 },
    /// Serotonin syndrome risk (additive serotonergic effects).
    SerotoninSyndrome { risk_score: f64 },
}

impl InteractionMechanism {
    /// Whether this is a PK (pharmacokinetic) interaction.
    pub fn is_pharmacokinetic(&self) -> bool {
        matches!(self,
            InteractionMechanism::CypInhibition { .. } |
            InteractionMechanism::CypInduction { .. } |
            InteractionMechanism::ProteinBindingDisplacement { .. } |
            InteractionMechanism::PgpInhibition { .. } |
            InteractionMechanism::PgpInduction { .. } |
            InteractionMechanism::RenalTransporterInhibition { .. } |
            InteractionMechanism::GastrointestinalPhChange { .. }
        )
    }

    /// Whether this is a PD (pharmacodynamic) interaction.
    pub fn is_pharmacodynamic(&self) -> bool {
        !self.is_pharmacokinetic()
    }

    /// Compute AUC ratio for CYP inhibition interactions.
    pub fn auc_ratio(&self, substrate_fm: f64, inhibitor_conc: f64) -> Option<f64> {
        match self {
            InteractionMechanism::CypInhibition { ki, .. } => {
                if *ki <= 0.0 { return None; }
                let ratio = 1.0 / (1.0 - substrate_fm + substrate_fm / (1.0 + inhibitor_conc / ki));
                Some(ratio)
            }
            InteractionMechanism::CypInduction { emax, ec50, .. } => {
                if *ec50 <= 0.0 { return None; }
                let induction_fold = 1.0 + emax * inhibitor_conc / (ec50 + inhibitor_conc);
                let ratio = 1.0 / (1.0 - substrate_fm + substrate_fm * induction_fold);
                Some(ratio)
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// InteractionEdge
// ---------------------------------------------------------------------------

/// An edge in the interaction network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEdge {
    pub perpetrator: DrugId,
    pub victim: DrugId,
    pub mechanism: InteractionMechanism,
    pub severity: Severity,
    pub clinical_significance: String,
    pub auc_fold_change: Option<f64>,
    pub bidirectional: bool,
}

impl InteractionEdge {
    pub fn new(
        perpetrator: DrugId,
        victim: DrugId,
        mechanism: InteractionMechanism,
        severity: Severity,
    ) -> Self {
        Self {
            perpetrator, victim, mechanism, severity,
            clinical_significance: String::new(),
            auc_fold_change: None,
            bidirectional: false,
        }
    }

    pub fn with_auc_change(mut self, fold: f64) -> Self {
        self.auc_fold_change = Some(fold);
        self
    }

    pub fn with_clinical_note(mut self, note: &str) -> Self {
        self.clinical_significance = note.to_string();
        self
    }

    pub fn with_bidirectional(mut self) -> Self {
        self.bidirectional = true;
        self
    }
}

// ---------------------------------------------------------------------------
// InteractionNetwork
// ---------------------------------------------------------------------------

/// Drug interaction network represented as an adjacency list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionNetwork {
    pub edges: Vec<InteractionEdge>,
    adjacency: HashMap<DrugId, Vec<usize>>,
}

impl InteractionNetwork {
    pub fn new() -> Self {
        Self { edges: Vec::new(), adjacency: HashMap::new() }
    }

    pub fn add_edge(&mut self, edge: InteractionEdge) {
        let idx = self.edges.len();
        self.adjacency
            .entry(edge.perpetrator.clone())
            .or_default()
            .push(idx);
        if edge.bidirectional {
            self.adjacency
                .entry(edge.victim.clone())
                .or_default()
                .push(idx);
        }
        self.edges.push(edge);
    }

    pub fn edge_count(&self) -> usize { self.edges.len() }
    pub fn drug_count(&self) -> usize { self.all_drugs().len() }

    pub fn all_drugs(&self) -> HashSet<DrugId> {
        let mut drugs = HashSet::new();
        for e in &self.edges {
            drugs.insert(e.perpetrator.clone());
            drugs.insert(e.victim.clone());
        }
        drugs
    }

    /// All edges where `drug` is the perpetrator.
    pub fn interactions_from(&self, drug: &DrugId) -> Vec<&InteractionEdge> {
        self.adjacency
            .get(drug)
            .map(|idxs| idxs.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    /// All edges where `drug` is the victim.
    pub fn interactions_affecting(&self, drug: &DrugId) -> Vec<&InteractionEdge> {
        self.edges.iter().filter(|e| e.victim == *drug).collect()
    }

    /// All pairwise interactions for a set of co-prescribed drugs.
    pub fn interactions_among(&self, drugs: &[DrugId]) -> Vec<&InteractionEdge> {
        let set: HashSet<_> = drugs.iter().cloned().collect();
        self.edges.iter()
            .filter(|e| set.contains(&e.perpetrator) && set.contains(&e.victim))
            .collect()
    }

    /// Highest severity among interactions in the set.
    pub fn max_severity(&self, drugs: &[DrugId]) -> Option<Severity> {
        self.interactions_among(drugs)
            .iter()
            .map(|e| e.severity)
            .max()
    }

    /// Find interaction chains (cascades) up to max_depth.
    pub fn find_chains(&self, start: &DrugId, max_depth: usize) -> Vec<InteractionChain> {
        let mut chains = Vec::new();
        let mut queue: VecDeque<(Vec<DrugId>, Vec<usize>)> = VecDeque::new();
        queue.push_back((vec![start.clone()], vec![]));

        while let Some((path, edge_idxs)) = queue.pop_front() {
            if path.len() > 1 {
                chains.push(InteractionChain {
                    drugs: path.clone(),
                    edge_indices: edge_idxs.clone(),
                });
            }
            if path.len() > max_depth { continue; }

            let current = path.last().unwrap();
            if let Some(idxs) = self.adjacency.get(current) {
                for &idx in idxs {
                    let edge = &self.edges[idx];
                    let next = if edge.perpetrator == *current {
                        &edge.victim
                    } else {
                        &edge.perpetrator
                    };
                    if !path.contains(next) {
                        let mut new_path = path.clone();
                        new_path.push(next.clone());
                        let mut new_edges = edge_idxs.clone();
                        new_edges.push(idx);
                        queue.push_back((new_path, new_edges));
                    }
                }
            }
        }
        chains
    }

    /// Detect cycles in the interaction network (BFS-based).
    pub fn detect_cycles(&self) -> Vec<Vec<DrugId>> {
        let mut cycles = Vec::new();
        let drugs: Vec<DrugId> = self.all_drugs().into_iter().collect();

        for drug in &drugs {
            let mut visited = HashSet::new();
            let mut stack: Vec<(DrugId, Vec<DrugId>)> = vec![(drug.clone(), vec![drug.clone()])];

            while let Some((current, path)) = stack.pop() {
                if visited.contains(&current) { continue; }
                visited.insert(current.clone());

                if let Some(idxs) = self.adjacency.get(&current) {
                    for &idx in idxs {
                        let edge = &self.edges[idx];
                        let next = if edge.perpetrator == current {
                            &edge.victim
                        } else {
                            &edge.perpetrator
                        };
                        if *next == *drug && path.len() > 2 {
                            let mut cycle = path.clone();
                            cycle.push(drug.clone());
                            cycles.push(cycle);
                        } else if !visited.contains(next) {
                            let mut new_path = path.clone();
                            new_path.push(next.clone());
                            stack.push((next.clone(), new_path));
                        }
                    }
                }
            }
        }
        cycles
    }
}

impl Default for InteractionNetwork {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// InteractionChain
// ---------------------------------------------------------------------------

/// A chain of drug interactions (cascade).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionChain {
    pub drugs: Vec<DrugId>,
    pub edge_indices: Vec<usize>,
}

impl InteractionChain {
    pub fn len(&self) -> usize { self.drugs.len() }
    pub fn is_empty(&self) -> bool { self.drugs.is_empty() }
    pub fn depth(&self) -> usize { self.edge_indices.len() }

    pub fn combined_auc_ratio(&self, network: &InteractionNetwork) -> Option<f64> {
        let mut ratio = 1.0;
        for &idx in &self.edge_indices {
            if let Some(fold) = network.edges[idx].auc_fold_change {
                ratio *= fold;
            } else {
                return None;
            }
        }
        Some(ratio)
    }

    pub fn max_severity(&self, network: &InteractionNetwork) -> Option<Severity> {
        self.edge_indices.iter()
            .map(|&i| network.edges[i].severity)
            .max()
    }
}

// ---------------------------------------------------------------------------
// InteractionResult
// ---------------------------------------------------------------------------

/// Result of analysing drug interactions for a regimen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionResult {
    pub drug_pair: (DrugId, DrugId),
    pub mechanisms: Vec<InteractionMechanism>,
    pub overall_severity: Severity,
    pub predicted_auc_ratio: Option<f64>,
    pub concentration_change: Option<ConcentrationInterval>,
    pub clinical_recommendation: String,
}

impl InteractionResult {
    pub fn new(perp: DrugId, victim: DrugId, severity: Severity) -> Self {
        Self {
            drug_pair: (perp, victim),
            mechanisms: Vec::new(),
            overall_severity: severity,
            predicted_auc_ratio: None,
            concentration_change: None,
            clinical_recommendation: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// MonotonicityChecker
// ---------------------------------------------------------------------------

/// Checks monotonicity of concentration response to interaction parameters.
///
/// For contract decomposition, we need the concentration mapping to be monotone
/// with respect to inhibition strength: stronger inhibition → higher victim Css.
#[derive(Debug, Clone)]
pub struct MonotonicityChecker {
    pub tolerance: f64,
}

impl MonotonicityChecker {
    pub fn new(tol: f64) -> Self { Self { tolerance: tol } }

    /// Check that f(x1) ≤ f(x2) when x1 ≤ x2 for all sample points.
    pub fn check_monotone_increasing(&self, xs: &[f64], ys: &[f64]) -> bool {
        if xs.len() != ys.len() || xs.len() < 2 { return false; }
        let mut pairs: Vec<(f64, f64)> = xs.iter().copied().zip(ys.iter().copied()).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for w in pairs.windows(2) {
            if w[1].1 < w[0].1 - self.tolerance {
                return false;
            }
        }
        true
    }

    /// Check that f(x1) ≥ f(x2) when x1 ≤ x2 for all sample points.
    pub fn check_monotone_decreasing(&self, xs: &[f64], ys: &[f64]) -> bool {
        if xs.len() != ys.len() || xs.len() < 2 { return false; }
        let mut pairs: Vec<(f64, f64)> = xs.iter().copied().zip(ys.iter().copied()).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for w in pairs.windows(2) {
            if w[1].1 > w[0].1 + self.tolerance {
                return false;
            }
        }
        true
    }

    /// Check inhibition-concentration monotonicity:
    /// As inhibitor concentration ↑, victim AUC ratio should ↑.
    pub fn check_inhibition_monotonicity(
        &self,
        inhibitor_concs: &[f64],
        auc_ratios: &[f64],
    ) -> bool {
        self.check_monotone_increasing(inhibitor_concs, auc_ratios)
    }

    /// Check induction monotonicity: inducer ↑ → victim AUC ↓.
    pub fn check_induction_monotonicity(
        &self,
        inducer_concs: &[f64],
        auc_ratios: &[f64],
    ) -> bool {
        self.check_monotone_decreasing(inducer_concs, auc_ratios)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Classify interaction severity based on AUC fold change.
pub fn classify_interaction_severity(auc_ratio: f64) -> Severity {
    if auc_ratio >= 5.0 { Severity::Contraindicated }
    else if auc_ratio >= 3.0 { Severity::Major }
    else if auc_ratio >= 2.0 { Severity::Moderate }
    else if auc_ratio >= 1.25 { Severity::Minor }
    else { Severity::None }
}

/// Compute AUC ratio from competitive inhibition parameters.
/// AUC_ratio = 1 / (1 - fm + fm/(1 + [I]/Ki))
pub fn compute_auc_ratio_from_inhibition(
    substrate_fm: f64,
    inhibitor_concentration: f64,
    ki: f64,
) -> f64 {
    if ki <= 0.0 || substrate_fm <= 0.0 || substrate_fm > 1.0 {
        return 1.0;
    }
    1.0 / (1.0 - substrate_fm + substrate_fm / (1.0 + inhibitor_concentration / ki))
}

/// Compute concentration interval after interaction.
pub fn interaction_adjusted_concentration(
    base: &ConcentrationInterval,
    auc_ratio_range: &ConcentrationInterval,
) -> ConcentrationInterval {
    ConcentrationInterval::new(
        base.lo * auc_ratio_range.lo,
        base.hi * auc_ratio_range.hi,
    )
}

/// Compute combined AUC ratio from multiple independent inhibitors on the same enzyme.
pub fn combined_inhibition_auc_ratio(
    substrate_fm: f64,
    inhibitors: &[(f64, f64)], // (concentration, Ki) pairs
) -> f64 {
    if substrate_fm <= 0.0 || substrate_fm > 1.0 { return 1.0; }
    let sum_ratio: f64 = inhibitors.iter()
        .map(|(conc, ki)| if *ki > 0.0 { conc / ki } else { 0.0 })
        .sum();
    1.0 / (1.0 - substrate_fm + substrate_fm / (1.0 + sum_ratio))
}

/// Build a default interaction network with common known interactions.
pub fn build_default_interaction_network() -> InteractionNetwork {
    let mut net = InteractionNetwork::new();

    // Fluconazole → CYP2C9 inhibition → Warfarin
    net.add_edge(
        InteractionEdge::new(
            DrugId::from_name("fluconazole"),
            DrugId::from_name("warfarin"),
            InteractionMechanism::CypInhibition {
                enzyme: CypEnzyme::CYP2C9,
                inhibition_type: InhibitionType::Competitive,
                ki: 7.0,
            },
            Severity::Major,
        ).with_auc_change(2.0)
         .with_clinical_note("Monitor INR closely; warfarin dose reduction likely needed"),
    );

    // Ketoconazole → CYP3A4 inhibition → Simvastatin
    net.add_edge(
        InteractionEdge::new(
            DrugId::from_name("ketoconazole"),
            DrugId::from_name("simvastatin"),
            InteractionMechanism::CypInhibition {
                enzyme: CypEnzyme::CYP3A4,
                inhibition_type: InhibitionType::Competitive,
                ki: 0.015,
            },
            Severity::Contraindicated,
        ).with_auc_change(12.0)
         .with_clinical_note("Contraindicated: rhabdomyolysis risk"),
    );

    // Rifampin → CYP3A4 induction → Cyclosporine
    net.add_edge(
        InteractionEdge::new(
            DrugId::from_name("rifampin"),
            DrugId::from_name("cyclosporine"),
            InteractionMechanism::CypInduction {
                enzyme: CypEnzyme::CYP3A4,
                emax: 10.0,
                ec50: 0.5,
            },
            Severity::Major,
        ).with_auc_change(0.1)
         .with_clinical_note("Cyclosporine levels reduced ~90%; transplant rejection risk"),
    );

    // Paroxetine → CYP2D6 inhibition → Codeine
    net.add_edge(
        InteractionEdge::new(
            DrugId::from_name("paroxetine"),
            DrugId::from_name("codeine"),
            InteractionMechanism::CypInhibition {
                enzyme: CypEnzyme::CYP2D6,
                inhibition_type: InhibitionType::Competitive,
                ki: 0.15,
            },
            Severity::Major,
        ).with_auc_change(0.3)
         .with_clinical_note("Reduced morphine conversion; therapeutic failure"),
    );

    // Omeprazole → CYP2C19 inhibition → Clopidogrel
    net.add_edge(
        InteractionEdge::new(
            DrugId::from_name("omeprazole"),
            DrugId::from_name("clopidogrel"),
            InteractionMechanism::CypInhibition {
                enzyme: CypEnzyme::CYP2C19,
                inhibition_type: InhibitionType::Competitive,
                ki: 2.0,
            },
            Severity::Major,
        ).with_auc_change(0.5)
         .with_clinical_note("Reduced antiplatelet effect; consider pantoprazole"),
    );

    // Amiodarone → QT prolongation + CYP3A4 inhibition → Multiple
    net.add_edge(
        InteractionEdge::new(
            DrugId::from_name("amiodarone"),
            DrugId::from_name("simvastatin"),
            InteractionMechanism::CypInhibition {
                enzyme: CypEnzyme::CYP3A4,
                inhibition_type: InhibitionType::NonCompetitive,
                ki: 1.5,
            },
            Severity::Major,
        ).with_auc_change(2.5)
         .with_clinical_note("Limit simvastatin to 20mg/day with amiodarone"),
    );

    // Ciprofloxacin → CYP1A2 inhibition → Theophylline
    net.add_edge(
        InteractionEdge::new(
            DrugId::from_name("ciprofloxacin"),
            DrugId::from_name("theophylline"),
            InteractionMechanism::CypInhibition {
                enzyme: CypEnzyme::CYP1A2,
                inhibition_type: InhibitionType::Competitive,
                ki: 4.0,
            },
            Severity::Moderate,
        ).with_auc_change(1.8)
         .with_clinical_note("Monitor theophylline levels; reduce dose if needed"),
    );

    // Fluoxetine → CYP2D6 inhibition → Metoprolol
    net.add_edge(
        InteractionEdge::new(
            DrugId::from_name("fluoxetine"),
            DrugId::from_name("metoprolol"),
            InteractionMechanism::CypInhibition {
                enzyme: CypEnzyme::CYP2D6,
                inhibition_type: InhibitionType::Competitive,
                ki: 0.5,
            },
            Severity::Moderate,
        ).with_auc_change(2.0)
         .with_clinical_note("Monitor heart rate and blood pressure"),
    );

    net
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_severity() {
        assert_eq!(classify_interaction_severity(1.0), Severity::None);
        assert_eq!(classify_interaction_severity(1.5), Severity::Minor);
        assert_eq!(classify_interaction_severity(2.5), Severity::Moderate);
        assert_eq!(classify_interaction_severity(4.0), Severity::Major);
        assert_eq!(classify_interaction_severity(6.0), Severity::Contraindicated);
    }

    #[test]
    fn test_auc_ratio_from_inhibition() {
        // No inhibitor → ratio = 1
        let r = compute_auc_ratio_from_inhibition(0.8, 0.0, 1.0);
        assert!((r - 1.0).abs() < 1e-6);

        // High inhibitor → high ratio
        let r = compute_auc_ratio_from_inhibition(0.9, 100.0, 1.0);
        assert!(r > 5.0);
    }

    #[test]
    fn test_combined_inhibition() {
        let r = combined_inhibition_auc_ratio(0.8, &[(1.0, 1.0), (2.0, 1.0)]);
        assert!(r > 1.0);
        // Two inhibitors should give greater ratio than one
        let r_single = combined_inhibition_auc_ratio(0.8, &[(1.0, 1.0)]);
        assert!(r > r_single);
    }

    #[test]
    fn test_interaction_network_basic() {
        let mut net = InteractionNetwork::new();
        net.add_edge(InteractionEdge::new(
            DrugId::from_name("A"),
            DrugId::from_name("B"),
            InteractionMechanism::PdSynergy { effect_multiplier: 1.5 },
            Severity::Moderate,
        ));
        assert_eq!(net.edge_count(), 1);
        assert_eq!(net.drug_count(), 2);
    }

    #[test]
    fn test_interactions_among() {
        let mut net = InteractionNetwork::new();
        net.add_edge(InteractionEdge::new(
            DrugId::from_name("A"), DrugId::from_name("B"),
            InteractionMechanism::PdSynergy { effect_multiplier: 1.5 },
            Severity::Moderate,
        ));
        net.add_edge(InteractionEdge::new(
            DrugId::from_name("C"), DrugId::from_name("D"),
            InteractionMechanism::PdSynergy { effect_multiplier: 1.5 },
            Severity::Minor,
        ));
        let drugs = vec![DrugId::from_name("A"), DrugId::from_name("B")];
        let ints = net.interactions_among(&drugs);
        assert_eq!(ints.len(), 1);
    }

    #[test]
    fn test_interaction_chains() {
        let mut net = InteractionNetwork::new();
        net.add_edge(InteractionEdge::new(
            DrugId::from_name("A"), DrugId::from_name("B"),
            InteractionMechanism::CypInhibition {
                enzyme: CypEnzyme::CYP3A4,
                inhibition_type: InhibitionType::Competitive,
                ki: 1.0,
            },
            Severity::Moderate,
        ).with_auc_change(2.0));
        net.add_edge(InteractionEdge::new(
            DrugId::from_name("B"), DrugId::from_name("C"),
            InteractionMechanism::CypInhibition {
                enzyme: CypEnzyme::CYP2D6,
                inhibition_type: InhibitionType::Competitive,
                ki: 0.5,
            },
            Severity::Major,
        ).with_auc_change(3.0));

        let chains = net.find_chains(&DrugId::from_name("A"), 3);
        assert!(chains.len() >= 2); // A→B and A→B→C
        let long_chain = chains.iter().find(|c| c.depth() == 2).unwrap();
        let combined = long_chain.combined_auc_ratio(&net).unwrap();
        assert!((combined - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_monotonicity_checker() {
        let mc = MonotonicityChecker::new(0.001);
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys_inc: Vec<f64> = xs.iter().map(|x| x * 2.0).collect();
        assert!(mc.check_monotone_increasing(&xs, &ys_inc));
        assert!(!mc.check_monotone_decreasing(&xs, &ys_inc));

        let ys_dec: Vec<f64> = xs.iter().map(|x| 10.0 - x * 2.0).collect();
        assert!(mc.check_monotone_decreasing(&xs, &ys_dec));
    }

    #[test]
    fn test_inhibition_monotonicity() {
        let mc = MonotonicityChecker::new(0.001);
        let concs = vec![0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        let ratios: Vec<f64> = concs.iter()
            .map(|c| compute_auc_ratio_from_inhibition(0.8, *c, 1.0))
            .collect();
        assert!(mc.check_inhibition_monotonicity(&concs, &ratios));
    }

    #[test]
    fn test_mechanism_classification() {
        let m = InteractionMechanism::CypInhibition {
            enzyme: CypEnzyme::CYP3A4,
            inhibition_type: InhibitionType::Competitive,
            ki: 1.0,
        };
        assert!(m.is_pharmacokinetic());
        assert!(!m.is_pharmacodynamic());

        let m = InteractionMechanism::QtProlongation { delta_qtc_ms: 10.0 };
        assert!(m.is_pharmacodynamic());
    }

    #[test]
    fn test_mechanism_auc_ratio() {
        let m = InteractionMechanism::CypInhibition {
            enzyme: CypEnzyme::CYP3A4,
            inhibition_type: InhibitionType::Competitive,
            ki: 1.0,
        };
        let ratio = m.auc_ratio(0.8, 5.0).unwrap();
        assert!(ratio > 1.0);
    }

    #[test]
    fn test_default_network() {
        let net = build_default_interaction_network();
        assert!(net.edge_count() >= 8);
        let warfarin_ints = net.interactions_affecting(&DrugId::from_name("warfarin"));
        assert!(!warfarin_ints.is_empty());
    }

    #[test]
    fn test_interaction_adjusted_concentration() {
        let base = ConcentrationInterval::new(5.0, 10.0);
        let ratio = ConcentrationInterval::new(1.5, 2.5);
        let adj = interaction_adjusted_concentration(&base, &ratio);
        assert!((adj.lo - 7.5).abs() < 0.01);
        assert!((adj.hi - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_max_severity() {
        let mut net = InteractionNetwork::new();
        net.add_edge(InteractionEdge::new(
            DrugId::from_name("A"), DrugId::from_name("B"),
            InteractionMechanism::PdSynergy { effect_multiplier: 1.5 },
            Severity::Minor,
        ));
        net.add_edge(InteractionEdge::new(
            DrugId::from_name("A"), DrugId::from_name("C"),
            InteractionMechanism::PdSynergy { effect_multiplier: 2.0 },
            Severity::Major,
        ));
        let drugs = vec![
            DrugId::from_name("A"),
            DrugId::from_name("B"),
            DrugId::from_name("C"),
        ];
        assert_eq!(net.max_severity(&drugs), Some(Severity::Major));
    }
}
