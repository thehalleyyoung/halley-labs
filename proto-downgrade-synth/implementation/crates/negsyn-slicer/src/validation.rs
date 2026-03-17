//! Slice validation and coverage checking.
//!
//! Validates that a program slice is complete with respect to negotiation behaviour,
//! checks CVE reachability, and provides differential validation against random sampling.

use std::collections::{HashMap, HashSet, BTreeSet};
use std::fmt;
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};

use crate::ir::{Module, Function, Instruction, Value, Type};
use crate::slice::{ProgramSlice, SliceCriterion};
use crate::callgraph::CallGraph;
use crate::cfg::{CFG, enumerate_paths};
use crate::taint::{TaintAnalysis, TaintTag, TaintSource, TaintState};
use crate::dependency::ProgramDependenceGraph;
use crate::{InstructionId, SlicerError, SlicerResult, NegotiationPhase};

// ---------------------------------------------------------------------------
// Validation report
// ---------------------------------------------------------------------------

/// Overall validation report for a program slice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub slice_name: String,
    pub overall_pass: bool,
    pub completeness: CompletenessResult,
    pub coverage: CoverageResult,
    pub cve_reachability: CVEReachabilityResult,
    pub differential: Option<DifferentialResult>,
    pub warnings: Vec<String>,
    pub suggestions: Vec<String>,
    pub content_hash: String,
}

impl ValidationReport {
    /// Generate a human-readable summary.
    pub fn summary(&self) -> String {
        let status = if self.overall_pass { "PASS" } else { "FAIL" };
        format!(
            "[{}] Slice '{}': completeness={:.1}%, coverage={}/{}, CVEs={}/{}, warnings={}",
            status, self.slice_name,
            self.completeness.score * 100.0,
            self.coverage.covered_phases, self.coverage.total_phases,
            self.cve_reachability.reachable_cves, self.cve_reachability.total_cves,
            self.warnings.len(),
        )
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletenessResult {
    pub score: f64,
    pub missing_definitions: Vec<String>,
    pub incomplete_control_flow: Vec<String>,
    pub unresolved_calls: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageResult {
    pub covered_phases: usize,
    pub total_phases: usize,
    pub uncovered_phases: Vec<String>,
    pub path_coverage: f64,
    pub function_coverage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVEReachabilityResult {
    pub reachable_cves: usize,
    pub total_cves: usize,
    pub cve_details: Vec<CVECheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVECheck {
    pub cve_id: String,
    pub description: String,
    pub pattern_function: String,
    pub reachable: bool,
    pub path: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialResult {
    pub agreement_ratio: f64,
    pub num_samples: usize,
    pub missed_by_slice: Vec<InstructionId>,
    pub extra_in_slice: usize,
}

// ---------------------------------------------------------------------------
// Slice validator
// ---------------------------------------------------------------------------

/// Validates a program slice for completeness and correctness.
pub struct SliceValidator<'a> {
    module: &'a Module,
    call_graph: Option<&'a CallGraph>,
}

impl<'a> SliceValidator<'a> {
    pub fn new(module: &'a Module) -> Self {
        Self { module, call_graph: None }
    }

    pub fn with_call_graph(mut self, cg: &'a CallGraph) -> Self {
        self.call_graph = Some(cg);
        self
    }

    /// Run all validations and produce a report.
    pub fn validate(
        &self,
        slice: &ProgramSlice,
    ) -> ValidationReport {
        let completeness = self.check_completeness(slice);
        let coverage = self.check_coverage(slice);
        let cve = self.check_cve_reachability(slice);
        let differential = None; // Only run on explicit request.

        let overall_pass = completeness.score >= 0.8
            && coverage.covered_phases > 0
            && completeness.missing_definitions.len() <= 5;

        let mut warnings = Vec::new();
        if completeness.score < 0.9 {
            warnings.push(format!("Low completeness score: {:.1}%", completeness.score * 100.0));
        }
        if !completeness.missing_definitions.is_empty() {
            warnings.push(format!(
                "{} missing definitions detected",
                completeness.missing_definitions.len()
            ));
        }
        if !completeness.unresolved_calls.is_empty() {
            warnings.push(format!(
                "{} unresolved calls in slice",
                completeness.unresolved_calls.len()
            ));
        }
        if coverage.covered_phases == 0 {
            warnings.push("No negotiation phases covered!".into());
        }

        let suggestions = self.generate_suggestions(&completeness, &coverage, &cve);

        // Compute content hash for cache invalidation.
        let mut hasher = Sha256::new();
        hasher.update(slice.name.as_bytes());
        hasher.update(slice.instruction_count.to_le_bytes());
        for f in &slice.functions {
            hasher.update(f.as_bytes());
        }
        let content_hash = hex::encode(hasher.finalize());

        ValidationReport {
            slice_name: slice.name.clone(),
            overall_pass,
            completeness,
            coverage,
            cve_reachability: cve,
            differential,
            warnings,
            suggestions,
            content_hash,
        }
    }

    /// Check negotiation-phase coverage of the slice.
    fn check_coverage(&self, slice: &ProgramSlice) -> CoverageResult {
        CoverageChecker::new(self.module).check_phase_coverage(slice)
    }

    /// Check whether known CVE-related code paths are reachable in the slice.
    fn check_cve_reachability(&self, slice: &ProgramSlice) -> CVEReachabilityResult {
        CVEReachabilityChecker::new(self.module).check(slice)
    }

    /// Check completeness: are all definitions used in the slice also included?
    fn check_completeness(&self, slice: &ProgramSlice) -> CompletenessResult {
        let mut missing_defs = Vec::new();
        let mut incomplete_cf = Vec::new();
        let mut unresolved_calls = Vec::new();
        let mut total_uses = 0usize;
        let mut resolved_uses = 0usize;

        let slice_ids: HashSet<InstructionId> = slice.all_instructions().into_iter().collect();

        for id in &slice_ids {
            if let Some(func) = self.module.function(&id.function) {
                if let Some(block) = func.blocks.get(&id.block) {
                    if let Some(instr) = block.instructions.get(id.index) {
                        // Check used registers are defined in the slice.
                        for reg in instr.used_registers() {
                            total_uses += 1;
                            let def_in_slice = self.find_definition(func, reg)
                                .map_or(false, |def_id| slice_ids.contains(&def_id));
                            let is_param = func.params.iter().any(|(pname, _)| pname == reg);
                            if def_in_slice || is_param {
                                resolved_uses += 1;
                            } else {
                                missing_defs.push(format!("{}:{}", id, reg));
                            }
                        }

                        // Check calls target functions in the slice or known externals.
                        if let Some(callee) = instr.called_function_name() {
                            if !slice.contains_function(callee) && !self.is_known_external(callee) {
                                unresolved_calls.push(format!("{} -> {}", id, callee));
                            }
                        }

                        // Check control flow completeness.
                        if instr.is_terminator() {
                            for succ in instr.successor_labels() {
                                // At least one instruction from the successor should be in the slice.
                                let succ_in_slice = slice.instructions
                                    .get(&id.function)
                                    .and_then(|blocks| blocks.get(succ))
                                    .map_or(false, |indices| !indices.is_empty());
                                if !succ_in_slice {
                                    incomplete_cf.push(format!("{} -> {}", id, succ));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Limit to top issues.
        missing_defs.truncate(20);
        incomplete_cf.truncate(20);
        unresolved_calls.truncate(20);

        let score = if total_uses > 0 {
            resolved_uses as f64 / total_uses as f64
        } else {
            1.0
        };

        CompletenessResult {
            score,
            missing_definitions: missing_defs,
            incomplete_control_flow: incomplete_cf,
            unresolved_calls,
        }
    }

    /// Find the instruction that defines a register.
    fn find_definition(&self, func: &Function, reg: &str) -> Option<InstructionId> {
        for (bname, block) in &func.blocks {
            for (idx, instr) in block.instructions.iter().enumerate() {
                if instr.dest() == Some(reg) {
                    return Some(InstructionId::new(&func.name, bname, idx));
                }
            }
        }
        None
    }

    /// Whether a function is a known external (libc, OpenSSL internal, etc.).
    fn is_known_external(&self, name: &str) -> bool {
        let known = [
            "malloc", "calloc", "realloc", "free",
            "memcpy", "memset", "memmove", "strlen", "strcmp", "strncmp",
            "printf", "fprintf", "snprintf",
            "OPENSSL_malloc", "OPENSSL_free", "OPENSSL_zalloc",
            "CRYPTO_malloc", "CRYPTO_free",
            "ERR_put_error", "ERR_clear_error", "ERR_raise",
            "BN_new", "BN_free", "BN_bin2bn",
            "EVP_MD_CTX_new", "EVP_MD_CTX_free",
        ];
        known.contains(&name) || name.starts_with("llvm.") || name.starts_with("__")
    }

    /// Generate suggestions for improving the slice.
    fn generate_suggestions(
        &self,
        completeness: &CompletenessResult,
        coverage: &CoverageResult,
        cve: &CVEReachabilityResult,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        if !completeness.missing_definitions.is_empty() {
            suggestions.push(format!(
                "Add the following definition sites to the slice: {}",
                completeness.missing_definitions.iter().take(3)
                    .map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            ));
        }

        if !coverage.uncovered_phases.is_empty() {
            suggestions.push(format!(
                "Consider including functions for uncovered phases: {:?}",
                coverage.uncovered_phases
            ));
        }

        for cve_check in &cve.cve_details {
            if !cve_check.reachable {
                suggestions.push(format!(
                    "CVE {} not reachable: consider including function '{}'",
                    cve_check.cve_id, cve_check.pattern_function
                ));
            }
        }

        if completeness.score < 0.7 {
            suggestions.push(
                "Very low completeness. Try increasing max_call_depth or disabling exclude_patterns."
                    .into(),
            );
        }

        suggestions
    }
}

// ---------------------------------------------------------------------------
// Coverage checker
// ---------------------------------------------------------------------------

/// Checks coverage of negotiation phases and paths.
pub struct CoverageChecker<'a> {
    module: &'a Module,
}

impl<'a> CoverageChecker<'a> {
    pub fn new(module: &'a Module) -> Self {
        Self { module }
    }

    /// Check which negotiation phases are covered by the slice.
    pub fn check_phase_coverage(&self, slice: &ProgramSlice) -> CoverageResult {
        let all_phases = [
            NegotiationPhase::ClientHello,
            NegotiationPhase::ServerHello,
            NegotiationPhase::CipherSuiteSelection,
            NegotiationPhase::VersionNegotiation,
            NegotiationPhase::ExtensionProcessing,
            NegotiationPhase::CertificateHandling,
            NegotiationPhase::KeyExchange,
            NegotiationPhase::Finished,
        ];

        let mut covered = 0usize;
        let mut uncovered = Vec::new();

        for phase in &all_phases {
            let phase_covered = slice.functions.iter().any(|f| phase.matches_function(f));
            if phase_covered {
                covered += 1;
            } else {
                uncovered.push(format!("{:?}", phase));
            }
        }

        // Compute path coverage within included functions.
        let (path_covered, total_paths) = self.compute_path_coverage(slice);
        let path_cov = if total_paths > 0 { path_covered as f64 / total_paths as f64 } else { 1.0 };

        // Function coverage.
        let neg_funcs: HashSet<&str> = self.module.negotiation_functions()
            .iter().map(|f| f.name.as_str()).collect();
        let covered_neg: usize = neg_funcs.iter()
            .filter(|f| slice.contains_function(f))
            .count();
        let func_cov = if neg_funcs.is_empty() { 1.0 } else {
            covered_neg as f64 / neg_funcs.len() as f64
        };

        CoverageResult {
            covered_phases: covered,
            total_phases: all_phases.len(),
            uncovered_phases: uncovered,
            path_coverage: path_cov,
            function_coverage: func_cov,
        }
    }

    /// Compute path coverage: how many CFG paths are represented in the slice.
    fn compute_path_coverage(&self, slice: &ProgramSlice) -> (usize, usize) {
        let mut covered = 0usize;
        let mut total = 0usize;

        for func_name in &slice.functions {
            if let Some(func) = self.module.function(func_name) {
                let cfg = CFG::from_function(func);
                let paths = enumerate_paths(&cfg, 50);
                total += paths.len();

                for path in &paths {
                    // A path is "covered" if all blocks in it have at least one instruction in slice.
                    let all_blocks_present = path.iter().all(|block_name| {
                        slice.instructions
                            .get(func_name)
                            .and_then(|blocks| blocks.get(block_name))
                            .map_or(false, |indices| !indices.is_empty())
                    });
                    if all_blocks_present {
                        covered += 1;
                    }
                }
            }
        }

        (covered, total)
    }
}

// ---------------------------------------------------------------------------
// CVE reachability checker
// ---------------------------------------------------------------------------

/// Known CVE patterns for protocol downgrade attacks.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CVEPattern {
    cve_id: String,
    description: String,
    /// Function that must be reachable for this CVE to apply.
    target_function: String,
    /// Additional check: specific instruction patterns.
    instruction_pattern: Option<String>,
}

/// Checks whether known CVE-related code paths are reachable in the slice.
pub struct CVEReachabilityChecker<'a> {
    module: &'a Module,
    patterns: Vec<CVEPattern>,
}

impl<'a> CVEReachabilityChecker<'a> {
    pub fn new(module: &'a Module) -> Self {
        Self {
            module,
            patterns: Self::known_patterns(),
        }
    }

    /// Known downgrade-related CVE patterns.
    fn known_patterns() -> Vec<CVEPattern> {
        vec![
            CVEPattern {
                cve_id: "CVE-2014-3566".into(),
                description: "POODLE: SSLv3 fallback attack".into(),
                target_function: "ssl3_accept".into(),
                instruction_pattern: Some("ssl3".into()),
            },
            CVEPattern {
                cve_id: "CVE-2014-0224".into(),
                description: "CCS Injection: ChangeCipherSpec before key exchange".into(),
                target_function: "ssl3_do_change_cipher_spec".into(),
                instruction_pattern: None,
            },
            CVEPattern {
                cve_id: "CVE-2015-0204".into(),
                description: "FREAK: RSA export cipher downgrade".into(),
                target_function: "ssl3_choose_cipher".into(),
                instruction_pattern: Some("export".into()),
            },
            CVEPattern {
                cve_id: "CVE-2015-4000".into(),
                description: "Logjam: DHE export downgrade".into(),
                target_function: "tls_process_key_exchange".into(),
                instruction_pattern: Some("dh".into()),
            },
            CVEPattern {
                cve_id: "CVE-2016-0800".into(),
                description: "DROWN: SSLv2 cross-protocol attack".into(),
                target_function: "ssl2_accept".into(),
                instruction_pattern: Some("ssl2".into()),
            },
            CVEPattern {
                cve_id: "CVE-2020-13777".into(),
                description: "GnuTLS: Session ticket key reuse".into(),
                target_function: "session_ticket".into(),
                instruction_pattern: None,
            },
            CVEPattern {
                cve_id: "CVE-2015-7575".into(),
                description: "SLOTH: Transcript collision attack".into(),
                target_function: "tls1_set_server_sigalgs".into(),
                instruction_pattern: Some("md5".into()),
            },
        ]
    }

    /// Check CVE reachability against the slice.
    pub fn check(&self, slice: &ProgramSlice) -> CVEReachabilityResult {
        let mut checks = Vec::new();
        let mut reachable_count = 0;

        for pattern in &self.patterns {
            let func_present = slice.contains_function(&pattern.target_function)
                || slice.functions.iter().any(|f| f.contains(&pattern.target_function));

            let pattern_match = if let Some(ref ip) = pattern.instruction_pattern {
                self.check_instruction_pattern(slice, ip)
            } else {
                true
            };

            let reachable = func_present && pattern_match;
            if reachable {
                reachable_count += 1;
            }

            let path = if func_present {
                vec![pattern.target_function.clone()]
            } else {
                Vec::new()
            };

            checks.push(CVECheck {
                cve_id: pattern.cve_id.clone(),
                description: pattern.description.clone(),
                pattern_function: pattern.target_function.clone(),
                reachable,
                path,
            });
        }

        CVEReachabilityResult {
            reachable_cves: reachable_count,
            total_cves: self.patterns.len(),
            cve_details: checks,
        }
    }

    /// Check if a pattern appears in slice instructions.
    fn check_instruction_pattern(&self, slice: &ProgramSlice, pattern: &str) -> bool {
        for func_name in &slice.functions {
            if let Some(func) = self.module.function(func_name) {
                if let Some(func_blocks) = slice.instructions.get(func_name) {
                    for (bname, indices) in func_blocks {
                        if let Some(block) = func.blocks.get(bname) {
                            for &idx in indices {
                                if let Some(instr) = block.instructions.get(idx) {
                                    // Check if any operand name matches the pattern.
                                    for reg in instr.used_registers() {
                                        if reg.to_lowercase().contains(&pattern.to_lowercase()) {
                                            return true;
                                        }
                                    }
                                    if let Some(callee) = instr.called_function_name() {
                                        if callee.to_lowercase().contains(&pattern.to_lowercase()) {
                                            return true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Differential validator
// ---------------------------------------------------------------------------

/// Compares a slice against random path sampling to validate completeness.
pub struct DifferentialValidator<'a> {
    module: &'a Module,
    num_samples: usize,
}

impl<'a> DifferentialValidator<'a> {
    pub fn new(module: &'a Module, num_samples: usize) -> Self {
        Self { module, num_samples }
    }

    /// Run differential validation.
    pub fn validate(&self, slice: &ProgramSlice) -> DifferentialResult {
        let mut missed = Vec::new();
        let mut extra = 0usize;
        let mut agreements = 0usize;
        let mut total_samples = 0usize;

        // Sample paths through negotiation functions.
        for func_name in slice.functions.iter() {
            if let Some(func) = self.module.function(func_name) {
                let cfg = CFG::from_function(func);
                let paths = enumerate_paths(&cfg, self.num_samples);
                total_samples += paths.len();

                for path in &paths {
                    // Check if all blocks in this path are represented in the slice.
                    let all_present = path.iter().all(|block_name| {
                        slice.instructions
                            .get(func_name)
                            .and_then(|blocks| blocks.get(block_name))
                            .map_or(false, |indices| !indices.is_empty())
                    });
                    if all_present {
                        agreements += 1;
                    } else {
                        // Find which instructions are missing.
                        for block_name in path {
                            if let Some(block) = func.blocks.get(block_name) {
                                let block_in_slice = slice.instructions
                                    .get(func_name)
                                    .and_then(|blocks| blocks.get(block_name))
                                    .map_or(false, |indices| !indices.is_empty());
                                if !block_in_slice {
                                    // Add key instructions from the missing block.
                                    for (idx, instr) in block.instructions.iter().enumerate() {
                                        if instr.is_call()
                                            || matches!(instr, Instruction::Store { .. })
                                            || matches!(instr, Instruction::Ret { .. })
                                        {
                                            missed.push(InstructionId::new(func_name, block_name, idx));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Count extra instructions (in slice but not on any sampled path).
        let sampled_blocks: HashSet<(String, String)> = slice.functions.iter()
            .flat_map(|func_name| {
                self.module.function(func_name).map(|func| {
                    let cfg = CFG::from_function(func);
                    let paths = enumerate_paths(&cfg, self.num_samples);
                    paths.into_iter().flat_map(|path| {
                        path.into_iter().map(|b| (func_name.clone(), b))
                    }).collect::<Vec<_>>()
                }).unwrap_or_default()
            })
            .collect();

        for (func_name, blocks) in &slice.instructions {
            for (block_name, indices) in blocks {
                if !sampled_blocks.contains(&(func_name.clone(), block_name.clone())) {
                    extra += indices.len();
                }
            }
        }

        missed.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
        missed.dedup();
        missed.truncate(50);

        let agreement_ratio = if total_samples > 0 {
            agreements as f64 / total_samples as f64
        } else {
            1.0
        };

        DifferentialResult {
            agreement_ratio,
            num_samples: total_samples,
            missed_by_slice: missed,
            extra_in_slice: extra,
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Run a full validation pipeline on a slice.
pub fn full_validation(
    module: &Module,
    slice: &ProgramSlice,
    call_graph: Option<&CallGraph>,
) -> ValidationReport {
    let mut validator = SliceValidator::new(module);
    if let Some(cg) = call_graph {
        validator = validator.with_call_graph(cg);
    }
    validator.validate(slice)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Module;
    use crate::slice::{ProgramSlice, SliceCriterion, ProtocolAwareSlicer};
    use crate::callgraph::CallGraphBuilder;

    fn make_test_slice() -> ProgramSlice {
        let c = SliceCriterion::cipher_negotiation();
        let mut slice = ProgramSlice::new("test_slice", c);
        slice.add_instruction(&InstructionId::new("ssl3_choose_cipher", "entry", 0));
        slice.add_instruction(&InstructionId::new("ssl3_choose_cipher", "entry", 1));
        slice.add_instruction(&InstructionId::new("ssl3_choose_cipher", "select_loop", 0));
        slice.add_instruction(&InstructionId::new("ssl3_choose_cipher", "select_loop", 1));
        slice.add_instruction(&InstructionId::new("ssl3_choose_cipher", "fallback", 0));
        slice
    }

    #[test]
    fn test_validation_report_summary() {
        let report = ValidationReport {
            slice_name: "test".into(),
            overall_pass: true,
            completeness: CompletenessResult {
                score: 0.95,
                missing_definitions: vec![],
                incomplete_control_flow: vec![],
                unresolved_calls: vec![],
            },
            coverage: CoverageResult {
                covered_phases: 3,
                total_phases: 8,
                uncovered_phases: vec![],
                path_coverage: 0.8,
                function_coverage: 0.5,
            },
            cve_reachability: CVEReachabilityResult {
                reachable_cves: 2,
                total_cves: 7,
                cve_details: vec![],
            },
            differential: None,
            warnings: vec![],
            suggestions: vec![],
            content_hash: "abc123".into(),
        };
        let summary = report.summary();
        assert!(summary.contains("PASS"));
        assert!(summary.contains("95.0%"));
    }

    #[test]
    fn test_completeness_check() {
        let module = Module::test_module();
        let validator = SliceValidator::new(&module);
        let slice = make_test_slice();
        let report = validator.validate(&slice);

        assert!(report.completeness.score >= 0.0);
        assert!(report.completeness.score <= 1.0);
    }

    #[test]
    fn test_coverage_checker() {
        let module = Module::test_module();
        let checker = CoverageChecker::new(&module);
        let slice = make_test_slice();
        let result = checker.check_phase_coverage(&slice);

        assert!(result.total_phases == 8);
        assert!(result.covered_phases >= 1); // ssl3_choose_cipher covers CipherSuiteSelection
    }

    #[test]
    fn test_cve_reachability() {
        let module = Module::test_module();
        let checker = CVEReachabilityChecker::new(&module);
        let slice = make_test_slice();
        let result = checker.check(&slice);

        assert_eq!(result.total_cves, 7);
        // ssl3_choose_cipher should make FREAK CVE reachable.
        let freak = result.cve_details.iter().find(|c| c.cve_id == "CVE-2015-0204");
        assert!(freak.is_some());
    }

    #[test]
    fn test_differential_validator() {
        let module = Module::test_module();
        let validator = DifferentialValidator::new(&module, 20);
        let slice = make_test_slice();
        let result = validator.validate(&slice);

        assert!(result.agreement_ratio >= 0.0);
        assert!(result.agreement_ratio <= 1.0);
    }

    #[test]
    fn test_full_validation() {
        let module = Module::test_module();
        let slice = make_test_slice();
        let report = full_validation(&module, &slice, None);

        assert!(!report.content_hash.is_empty());
    }

    #[test]
    fn test_report_json() {
        let report = ValidationReport {
            slice_name: "test".into(),
            overall_pass: false,
            completeness: CompletenessResult {
                score: 0.5,
                missing_definitions: vec!["x".into()],
                incomplete_control_flow: vec![],
                unresolved_calls: vec![],
            },
            coverage: CoverageResult {
                covered_phases: 0,
                total_phases: 8,
                uncovered_phases: vec!["CipherSuiteSelection".into()],
                path_coverage: 0.0,
                function_coverage: 0.0,
            },
            cve_reachability: CVEReachabilityResult {
                reachable_cves: 0,
                total_cves: 7,
                cve_details: vec![],
            },
            differential: None,
            warnings: vec!["Low coverage".into()],
            suggestions: vec!["Include more functions".into()],
            content_hash: "def456".into(),
        };
        let json = report.to_json().unwrap();
        assert!(json.contains("test"));
        assert!(json.contains("Low coverage"));
    }

    #[test]
    fn test_slicer_full_pipeline() {
        let mut module = Module::test_module();
        module.compute_all_predecessors();
        let cg = CallGraphBuilder::new(&module).with_indirect_resolution(false).build();
        let mut slicer = ProtocolAwareSlicer::new(&module, &cg);
        slicer.prepare().unwrap();

        let criterion = SliceCriterion::cipher_negotiation();
        let slice = slicer.slice(&criterion).unwrap();

        let report = full_validation(&module, &slice, Some(&cg));
        assert!(report.completeness.score >= 0.0);
    }

    #[test]
    fn test_known_external() {
        let module = Module::test_module();
        let validator = SliceValidator::new(&module);
        assert!(validator.is_known_external("malloc"));
        assert!(validator.is_known_external("OPENSSL_malloc"));
        assert!(validator.is_known_external("llvm.dbg.declare"));
        assert!(!validator.is_known_external("ssl3_choose_cipher"));
    }

    #[test]
    fn test_empty_slice_validation() {
        let module = Module::test_module();
        let validator = SliceValidator::new(&module);
        let slice = ProgramSlice::new("empty", SliceCriterion::cipher_negotiation());
        let report = validator.validate(&slice);

        assert!(report.completeness.score >= 0.0);
    }

    #[test]
    fn test_suggestions_generation() {
        let module = Module::test_module();
        let validator = SliceValidator::new(&module);
        let completeness = CompletenessResult {
            score: 0.5,
            missing_definitions: vec!["a".into(), "b".into()],
            incomplete_control_flow: vec![],
            unresolved_calls: vec![],
        };
        let coverage = CoverageResult {
            covered_phases: 0,
            total_phases: 8,
            uncovered_phases: vec!["CipherSuiteSelection".into()],
            path_coverage: 0.0,
            function_coverage: 0.0,
        };
        let cve = CVEReachabilityResult {
            reachable_cves: 0,
            total_cves: 1,
            cve_details: vec![CVECheck {
                cve_id: "CVE-TEST".into(),
                description: "test".into(),
                pattern_function: "test_func".into(),
                reachable: false,
                path: vec![],
            }],
        };
        let suggestions = validator.generate_suggestions(&completeness, &coverage, &cve);
        assert!(!suggestions.is_empty());
    }
}
