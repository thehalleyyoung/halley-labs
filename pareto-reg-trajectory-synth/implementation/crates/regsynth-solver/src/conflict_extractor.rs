// regsynth-solver: MUS (Minimal Unsatisfiable Subset) extraction
// Deletion-based MUS extraction + MARCO algorithm for all MUSes.
// Maps MUS back to regulatory obligations via provenance.

use crate::result::{
    Clause, Literal, MinimalUnsatisfiableSubset, SatResult, SolverStatistics,
    lit_neg, lit_var, make_lit,
};
use crate::sat_solver::{solve_cnf, DpllSolver};
use crate::solver_config::SolverConfig;
use regsynth_encoding::Provenance;
use std::collections::HashSet;
use std::time::Instant;

// ─── MUS Extractor ──────────────────────────────────────────────────────────

/// Extractor for Minimal Unsatisfiable Subsets.
pub struct MusExtractor {
    config: SolverConfig,
    pub stats: SolverStatistics,
}

impl MusExtractor {
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            stats: SolverStatistics::new(),
        }
    }

    /// Extract a single MUS from an unsatisfiable set of clauses using deletion-based approach.
    ///
    /// Algorithm:
    /// 1. Start with the full clause set (known UNSAT).
    /// 2. For each clause c_i:
    ///    a. Remove c_i temporarily.
    ///    b. If the remaining set is still UNSAT, c_i is redundant → keep it removed.
    ///    c. If the remaining set is SAT, c_i is necessary → restore it.
    /// 3. The remaining clauses form a MUS.
    pub fn extract_mus(
        &mut self,
        clauses: &[Clause],
        constraint_ids: &[String],
        provenances: &[Option<Provenance>],
    ) -> Option<MinimalUnsatisfiableSubset> {
        let start = Instant::now();

        if clauses.is_empty() {
            return None;
        }

        // First, verify the clause set is UNSAT
        let num_vars = count_vars(clauses);
        let result = solve_cnf(num_vars, clauses);
        if result.is_sat() {
            return None; // Not UNSAT, no MUS exists
        }

        // Get initial UNSAT core to narrow down
        let initial_core: Vec<usize> = match &result {
            SatResult::Unsat(core) => {
                // Try to map core clauses back to original indices
                let core_set: HashSet<Vec<Literal>> = core.iter().cloned().collect();
                clauses
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| core_set.contains(*c))
                    .map(|(i, _)| i)
                    .collect()
            }
            _ => (0..clauses.len()).collect(),
        };

        // Use the core as starting point (or all clauses if core mapping is unclear)
        let working_set: Vec<usize> = if initial_core.is_empty() {
            (0..clauses.len()).collect()
        } else {
            initial_core
        };

        // Deletion-based MUS extraction
        let mut mus_indices: Vec<usize> = working_set.clone();
        let mut i = 0;

        while i < mus_indices.len() {
            if start.elapsed() > self.config.timeout {
                break;
            }

            self.stats.decisions += 1;

            // Try removing clause at index i
            let removed_idx = mus_indices[i];
            let remaining: Vec<Clause> = mus_indices
                .iter()
                .filter(|&&idx| idx != removed_idx)
                .map(|&idx| clauses[idx].clone())
                .collect();

            if remaining.is_empty() {
                // Can't remove the last clause
                i += 1;
                continue;
            }

            let nv = count_vars(&remaining);
            let check = solve_cnf(nv, &remaining);
            self.stats.conflicts += 1;

            match check {
                SatResult::Unsat(_) => {
                    // Still UNSAT without this clause → redundant, remove it
                    mus_indices.remove(i);
                    // Don't increment i since we removed an element
                }
                SatResult::Sat(_) | SatResult::Unknown(_) => {
                    // SAT without this clause → this clause is necessary
                    i += 1;
                }
            }
        }

        self.stats.time_ms = start.elapsed().as_millis() as u64;

        // Build the MUS result
        let mus_constraint_ids: Vec<String> = mus_indices
            .iter()
            .filter_map(|&idx| constraint_ids.get(idx).cloned())
            .collect();

        let mus_obligations: Vec<String> = mus_indices
            .iter()
            .filter_map(|&idx| {
                provenances.get(idx).and_then(|p| {
                    p.as_ref().map(|prov| {
                        format!(
                            "[{}] {} ({})",
                            prov.jurisdiction,
                            prov.description,
                            prov.article_ref.as_deref().unwrap_or("N/A")
                        )
                    })
                })
            })
            .collect();

        Some(MinimalUnsatisfiableSubset {
            constraint_indices: mus_indices,
            constraint_ids: mus_constraint_ids,
            obligations: mus_obligations,
        })
    }

    /// Extract all MUSes using the MARCO algorithm.
    ///
    /// MARCO (Mapping Regions of Constraint sets to determine all minimal
    /// Unsatisfiable Subsets) algorithm:
    /// 1. Maintain a "map solver" that tracks explored subsets.
    /// 2. Repeatedly ask the map solver for an unexplored subset.
    /// 3. If the subset is SAT → it's a Maximal Satisfiable Subset (MSS).
    ///    Block all subsets of it (add clause: at least one clause NOT in MSS must be in).
    /// 4. If the subset is UNSAT → extract a MUS from it.
    ///    Block all supersets of the MUS (add clause: at least one clause in MUS must be out).
    /// 5. Repeat until no unexplored subsets remain.
    pub fn extract_all_mus(
        &mut self,
        clauses: &[Clause],
        constraint_ids: &[String],
        provenances: &[Option<Provenance>],
        max_mus: usize,
    ) -> Vec<MinimalUnsatisfiableSubset> {
        let start = Instant::now();
        let n = clauses.len();

        if n == 0 {
            return Vec::new();
        }

        let mut all_muses = Vec::new();

        // Map solver: one variable per clause (variable i+1 means clause i is "in")
        let map_num_vars = n as u32;
        let mut map_clauses: Vec<Clause> = Vec::new();

        // Start with an initial seed: all clauses in
        // We use the map solver to generate subsets to explore

        let max_iterations = max_mus * 10; // Bound iterations
        for _iter in 0..max_iterations {
            if start.elapsed() > self.config.timeout {
                break;
            }
            if all_muses.len() >= max_mus {
                break;
            }

            // Ask map solver for an unexplored seed
            let seed = self.get_unexplored_seed(map_num_vars, &map_clauses);
            let seed = match seed {
                Some(s) => s,
                None => break, // All subsets explored
            };

            // Extract the subset of clauses indicated by the seed
            let subset_indices: Vec<usize> = seed.iter().copied().collect();
            let subset_clauses: Vec<Clause> = subset_indices
                .iter()
                .map(|&i| clauses[i].clone())
                .collect();

            if subset_clauses.is_empty() {
                // Block this empty seed
                let block: Clause = (1..=map_num_vars).map(|v| make_lit(v, true)).collect();
                map_clauses.push(block);
                continue;
            }

            let nv = count_vars(&subset_clauses);
            let result = solve_cnf(nv, &subset_clauses);

            match result {
                SatResult::Sat(_) => {
                    // This subset is SAT → it's contained in an MSS.
                    // Block all subsets: at least one clause NOT in this subset must be present.
                    let complement: Vec<usize> = (0..n)
                        .filter(|i| !subset_indices.contains(i))
                        .collect();
                    if !complement.is_empty() {
                        let block: Clause = complement
                            .iter()
                            .map(|&i| make_lit((i + 1) as u32, true))
                            .collect();
                        map_clauses.push(block);
                    } else {
                        // All clauses are SAT → no MUS exists
                        break;
                    }
                }
                SatResult::Unsat(_) => {
                    // This subset is UNSAT → extract a MUS from it
                    let subset_ids: Vec<String> = subset_indices
                        .iter()
                        .filter_map(|&i| constraint_ids.get(i).cloned())
                        .collect();
                    let subset_provs: Vec<Option<Provenance>> = subset_indices
                        .iter()
                        .map(|&i| provenances.get(i).cloned().flatten())
                        .collect();

                    if let Some(mus) =
                        self.extract_mus(&subset_clauses, &subset_ids, &subset_provs)
                    {
                        // Map MUS indices back to original clause indices
                        let original_mus_indices: Vec<usize> = mus
                            .constraint_indices
                            .iter()
                            .filter_map(|&local_idx| subset_indices.get(local_idx).copied())
                            .collect();

                        let original_ids: Vec<String> = original_mus_indices
                            .iter()
                            .filter_map(|&i| constraint_ids.get(i).cloned())
                            .collect();

                        let original_obligations: Vec<String> = original_mus_indices
                            .iter()
                            .filter_map(|&idx| {
                                provenances.get(idx).and_then(|p| {
                                    p.as_ref().map(|prov| {
                                        format!(
                                            "[{}] {} ({})",
                                            prov.jurisdiction,
                                            prov.description,
                                            prov.article_ref.as_deref().unwrap_or("N/A")
                                        )
                                    })
                                })
                            })
                            .collect();

                        // Block all supersets of this MUS
                        let block: Clause = original_mus_indices
                            .iter()
                            .map(|&i| make_lit((i + 1) as u32, false))
                            .collect();
                        map_clauses.push(block);

                        all_muses.push(MinimalUnsatisfiableSubset {
                            constraint_indices: original_mus_indices,
                            constraint_ids: original_ids,
                            obligations: original_obligations,
                        });
                    } else {
                        // Couldn't extract MUS (shouldn't happen if subset is UNSAT)
                        break;
                    }
                }
                SatResult::Unknown(_) => {
                    continue;
                }
            }
        }

        self.stats.time_ms = start.elapsed().as_millis() as u64;
        all_muses
    }

    /// Get an unexplored subset seed from the map solver.
    fn get_unexplored_seed(
        &self,
        num_vars: u32,
        map_clauses: &[Clause],
    ) -> Option<Vec<usize>> {
        if map_clauses.is_empty() {
            // First call: return the full set
            return Some((0..num_vars as usize).collect());
        }

        let result = solve_cnf(num_vars, map_clauses);
        match result {
            SatResult::Sat(assignment) => {
                let mut seed = Vec::new();
                for v in 1..=num_vars {
                    if assignment.get(v) == Some(true) {
                        seed.push((v - 1) as usize);
                    }
                }
                Some(seed)
            }
            _ => None, // No more unexplored subsets
        }
    }

    /// Convenience: extract MUS from SmtConstraints.
    pub fn extract_mus_from_smt(
        &mut self,
        constraints: &[regsynth_encoding::SmtConstraint],
        clauses: &[Clause],
    ) -> Option<MinimalUnsatisfiableSubset> {
        let ids: Vec<String> = constraints.iter().map(|c| c.id.clone()).collect();
        let provs: Vec<Option<Provenance>> =
            constraints.iter().map(|c| c.provenance.clone()).collect();
        self.extract_mus(clauses, &ids, &provs)
    }
}

/// Count the maximum variable index in a set of clauses.
fn count_vars(clauses: &[Clause]) -> u32 {
    clauses
        .iter()
        .flat_map(|c| c.iter())
        .map(|l| lit_var(*l))
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> SolverConfig {
        SolverConfig::default()
    }

    #[test]
    fn test_mus_simple() {
        let mut extractor = MusExtractor::new(default_config());
        // Three clauses: {x1}, {NOT x1, x2}, {NOT x2}
        // All three are needed for UNSAT (it's already a MUS)
        let clauses = vec![vec![1], vec![-1, 2], vec![-2]];
        let ids = vec!["c1".to_string(), "c2".to_string(), "c3".to_string()];
        let provs = vec![None, None, None];

        let mus = extractor.extract_mus(&clauses, &ids, &provs);
        assert!(mus.is_some());
        let mus = mus.unwrap();
        // This is already minimal: removing any single clause makes it SAT
        assert!(mus.constraint_indices.len() <= 3);
        assert!(mus.constraint_indices.len() >= 2);
    }

    #[test]
    fn test_mus_redundant_clause() {
        let mut extractor = MusExtractor::new(default_config());
        // Clauses: {x1}, {NOT x1}, {x2}, {NOT x2}
        // MUS should be either {x1, NOT x1} or {x2, NOT x2}
        let clauses = vec![vec![1], vec![-1], vec![2], vec![-2]];
        let ids = vec![
            "c1".to_string(),
            "c2".to_string(),
            "c3".to_string(),
            "c4".to_string(),
        ];
        let provs = vec![None, None, None, None];

        let mus = extractor.extract_mus(&clauses, &ids, &provs);
        assert!(mus.is_some());
        let mus = mus.unwrap();
        assert_eq!(mus.constraint_indices.len(), 2);
    }

    #[test]
    fn test_mus_satisfiable() {
        let mut extractor = MusExtractor::new(default_config());
        // SAT formula: no MUS exists
        let clauses = vec![vec![1, 2], vec![-1, 2]];
        let ids = vec!["c1".to_string(), "c2".to_string()];
        let provs = vec![None, None];

        let mus = extractor.extract_mus(&clauses, &ids, &provs);
        assert!(mus.is_none());
    }

    #[test]
    fn test_extract_all_mus_single() {
        let mut extractor = MusExtractor::new(default_config());
        // Single MUS: {x1}, {NOT x1}
        let clauses = vec![vec![1], vec![-1]];
        let ids = vec!["c1".to_string(), "c2".to_string()];
        let provs = vec![None, None];

        let muses = extractor.extract_all_mus(&clauses, &ids, &provs, 10);
        assert!(muses.len() >= 1);
        assert_eq!(muses[0].constraint_indices.len(), 2);
    }

    #[test]
    fn test_extract_all_mus_multiple() {
        let mut extractor = MusExtractor::new(default_config());
        // {x1}, {NOT x1}, {x2}, {NOT x2}
        // Two MUSes: {x1, NOT x1} and {x2, NOT x2}
        let clauses = vec![vec![1], vec![-1], vec![2], vec![-2]];
        let ids = vec![
            "c1".to_string(),
            "c2".to_string(),
            "c3".to_string(),
            "c4".to_string(),
        ];
        let provs = vec![None, None, None, None];

        let muses = extractor.extract_all_mus(&clauses, &ids, &provs, 10);
        assert!(muses.len() >= 1);
        // Should find at least one MUS of size 2
        assert!(muses.iter().any(|m| m.constraint_indices.len() == 2));
    }

    #[test]
    fn test_mus_with_provenance() {
        let mut extractor = MusExtractor::new(default_config());
        let clauses = vec![vec![1], vec![-1]];
        let ids = vec!["eu_art6".to_string(), "us_sec5".to_string()];
        let provs = vec![
            Some(Provenance {
                obligation_id: "ob1".to_string(),
                jurisdiction: "EU".to_string(),
                article_ref: Some("Art. 6".to_string()),
                description: "Risk assessment required".to_string(),
            }),
            Some(Provenance {
                obligation_id: "ob2".to_string(),
                jurisdiction: "US".to_string(),
                article_ref: Some("Sec. 5".to_string()),
                description: "No risk assessment for category X".to_string(),
            }),
        ];

        let mus = extractor.extract_mus(&clauses, &ids, &provs);
        assert!(mus.is_some());
        let mus = mus.unwrap();
        assert_eq!(mus.obligations.len(), 2);
        assert!(mus.obligations[0].contains("EU"));
        assert!(mus.obligations[1].contains("US"));
    }

    #[test]
    fn test_empty_clauses() {
        let mut extractor = MusExtractor::new(default_config());
        let mus = extractor.extract_mus(&[], &[], &[]);
        assert!(mus.is_none());
    }
}
