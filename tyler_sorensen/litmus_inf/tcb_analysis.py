#!/usr/bin/env python3
"""
Trusted Computing Base (TCB) Analysis for LITMUS∞.

Addresses critical weakness: TCB is significantly larger than acknowledged.
The Z3 solver is not the only trusted component — the entire pipeline from
source code to verdict depends on:

  1. Z3 solver (external, ~500K LoC)
  2. Python Z3 bindings (~10K LoC)  
  3. portcheck.py: pattern database + enumeration engine
  4. smt_validation.py: SMT encoding logic
  5. model_dsl.py: DSL parser and interpreter
  6. ast_analyzer.py: AST-based code analysis
  7. memory_model.py: model definitions
  8. compositional_reasoning.py: composition theorems (paper proofs)
  9. severity_classification.py: CWE-calibrated taxonomy

This module provides:
  - Formal TCB enumeration with per-component analysis
  - Trust classification: VERIFIED / VALIDATED / TRUSTED
  - Guarantee dependency graph: which claims depend on which components
  - Mitigation analysis: what reduces risk for each trusted component
  - Comparison with other formal verification tools' TCBs
"""

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from enum import Enum

sys.path.insert(0, os.path.dirname(__file__))


class TrustLevel(Enum):
    """Classification of trust in a component."""
    VERIFIED = "verified"          # Machine-checked (Z3 proofs, model checking)
    VALIDATED = "validated"        # Tested/cross-validated but not proven
    TRUSTED = "trusted"            # Assumed correct (in TCB)
    MITIGATED = "mitigated"        # Trusted but with risk mitigation


@dataclass
class TCBComponent:
    """A single component in the trusted computing base."""
    name: str
    description: str
    trust_level: TrustLevel
    loc: int  # lines of code
    file_paths: List[str]
    guarantees_affected: List[str]
    mitigations: List[str]
    external: bool = False
    estimated_external_loc: int = 0
    
    @property
    def total_loc(self):
        return self.loc + self.estimated_external_loc


@dataclass  
class GuaranteeDependency:
    """Maps a system guarantee to its trust dependencies."""
    guarantee: str
    description: str
    depends_on: List[str]  # component names
    weakest_link: str
    weakest_trust_level: TrustLevel
    mitigation_summary: str


class TCBAnalyzer:
    """Formal TCB analysis for LITMUS∞."""
    
    def __init__(self, project_root=None):
        if project_root is None:
            project_root = os.path.dirname(__file__)
        self.project_root = project_root
        self.components: List[TCBComponent] = []
        self.guarantees: List[GuaranteeDependency] = []
        self._build_tcb()
        self._build_guarantees()
    
    def _count_lines(self, filename: str) -> int:
        """Count non-blank, non-comment lines in a Python file."""
        filepath = os.path.join(self.project_root, filename)
        if not os.path.exists(filepath):
            return 0
        count = 0
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                        count += 1
        except Exception:
            return 0
        return count
    
    def _build_tcb(self):
        """Enumerate all TCB components with trust classifications."""
        
        # 1. Z3 SMT Solver (external)
        self.components.append(TCBComponent(
            name="z3_solver",
            description="Z3 SMT solver: proof search, SAT/UNSAT determination, proof production",
            trust_level=TrustLevel.TRUSTED,
            loc=0,
            file_paths=[],
            guarantees_affected=[
                "1370/1370 certificate coverage",
                "735 UNSAT safety proofs",
                "635 SAT witnesses",
                "228/228 SMT internal consistency",
                "108/108 GPU SMT consistency",
                "Fence sufficiency proofs",
            ],
            mitigations=[
                "SMT-LIB2 export enables cross-solver replay (CVC5, Yices)",
                "QF_LIA fragment is decidable and well-tested",
                "SAT models verified by direct substitution (self-certifying)",
                "Alethe proof extraction provides independently checkable proofs",
                "Z3 is mature (~15 years), widely used in industry and academia",
            ],
            external=True,
            estimated_external_loc=500000,
        ))
        
        # 2. Python Z3 bindings
        self.components.append(TCBComponent(
            name="z3_python_bindings",
            description="Z3 Python API: solver interface, model extraction, proof access",
            trust_level=TrustLevel.TRUSTED,
            loc=0,
            file_paths=[],
            guarantees_affected=["All Z3-based results"],
            mitigations=[
                "Official Z3 bindings, maintained by Microsoft Research",
                "Thin wrapper around C API",
            ],
            external=True,
            estimated_external_loc=10000,
        ))
        
        # 3. Pattern database
        self.components.append(TCBComponent(
            name="pattern_database",
            description="137 litmus test patterns with operations, forbidden outcomes, and dependencies",
            trust_level=TrustLevel.VALIDATED,
            loc=self._count_lines('portcheck.py'),
            file_paths=['portcheck.py'],
            guarantees_affected=[
                "Pattern correctness (forbidden outcomes match specifications)",
                "Pattern coverage (137 patterns cover important idioms)",
                "Code recognition accuracy",
            ],
            mitigations=[
                "228/228 herd7 agreement validates CPU pattern outcomes",
                "170/171 DSL-to-.cat correspondence validates model encoding",
                "94/94 GPU external cross-validation",
                "Derived from published litmus test specifications",
            ],
        ))
        
        # 4. Enumeration engine
        self.components.append(TCBComponent(
            name="enumeration_engine",
            description="RF×CO exhaustive enumeration with ghb acyclicity checking",
            trust_level=TrustLevel.VALIDATED,
            loc=self._count_lines('portcheck.py'),
            file_paths=['portcheck.py'],
            guarantees_affected=[
                "Enumeration completeness (no candidate execution missed)",
                "ghb acyclicity check correctness",
            ],
            mitigations=[
                "228/228 agreement with independent SMT encoding",
                "RF×CO decomposition lemma (Lemma 1) proves completeness",
                "Cross-validated against herd7 specifications",
            ],
        ))
        
        # 5. SMT encoding
        self.components.append(TCBComponent(
            name="smt_encoding",
            description="Translation of litmus tests + memory models to QF_LIA SMT formulas",
            trust_level=TrustLevel.VALIDATED,
            loc=self._count_lines('smt_validation.py'),
            file_paths=['smt_validation.py'],
            guarantees_affected=[
                "SMT formula faithfully represents the memory model",
                "UNSAT proofs correspond to actual safety",
                "SAT witnesses correspond to actual violations",
            ],
            mitigations=[
                "228/228 agreement with enumeration engine",
                "Full SMT-LIB2 specification documented in paper",
                "Encoding follows standard axiomatic framework (Alglave et al.)",
            ],
        ))
        
        # 6. DSL interpreter
        self.components.append(TCBComponent(
            name="dsl_interpreter",
            description="Custom memory model DSL: parser, interpreter, fence selection",
            trust_level=TrustLevel.VALIDATED,
            loc=self._count_lines('model_dsl.py'),
            file_paths=['model_dsl.py'],
            guarantees_affected=[
                "Custom model results match user intent",
                "170/171 DSL-to-.cat correspondence",
            ],
            mitigations=[
                "170/171 empirical correspondence with herd7 .cat specifications",
                "Formal denotational semantics (dsl_denotational_semantics.py)",
                "1 known mismatch (mp_fence_wr on RISC-V) documented",
            ],
        ))
        
        # 7. AST analyzer
        self.components.append(TCBComponent(
            name="ast_analyzer",
            description="tree-sitter AST parsing, pattern matching, coverage confidence",
            trust_level=TrustLevel.TRUSTED,
            loc=self._count_lines('ast_analyzer.py'),
            file_paths=['ast_analyzer.py'],
            guarantees_affected=[
                "Code recognition accuracy (86.1% exact, 93.3% top-3)",
                "Pattern identification from source code",
                "Coverage confidence metric",
            ],
            mitigations=[
                "238-snippet benchmark with 35 independently sourced",
                "UnrecognizedPatternWarning when coverage < 50%",
                "Does not affect formal verification (back-end is verified)",
                "Users can manually specify patterns to bypass AST analysis",
            ],
        ))
        
        # 8. Memory model definitions
        self.components.append(TCBComponent(
            name="model_definitions",
            description="Built-in x86, SPARC, ARM, RISC-V, GPU model specifications",
            trust_level=TrustLevel.VALIDATED,
            loc=self._count_lines('memory_model.py'),
            file_paths=['memory_model.py'],
            guarantees_affected=[
                "Model fidelity to hardware specifications",
                "Relaxation and fence characterizations",
            ],
            mitigations=[
                "Derived from published specifications (Owens et al., Pulte et al., RISC-V ISA)",
                "228/228 herd7 agreement for CPU models",
                "25/25 literature-based hardware consistency",
                "94/94 GPU external cross-validation",
            ],
        ))
        
        # 9. Compositional reasoning
        self.components.append(TCBComponent(
            name="compositional_reasoning",
            description="Multi-pattern composition: disjoint, SC-shared, Owicki-Gries, RG",
            trust_level=TrustLevel.TRUSTED,
            loc=self._count_lines('compositional_reasoning.py'),
            file_paths=['compositional_reasoning.py'],
            guarantees_affected=[
                "Theorems 4-7 (composition soundness)",
                "Multi-pattern program analysis correctness",
            ],
            mitigations=[
                "Paper proofs for all theorems",
                "15 multi-pattern experimental evaluations",
                "Conservative fallback for unhandled cases",
                "Not mechanized in proof assistant",
            ],
        ))
        
        # 10. Severity classification
        self.components.append(TCBComponent(
            name="severity_classification",
            description="CWE-calibrated severity taxonomy: data_race, security, benign",
            trust_level=TrustLevel.TRUSTED,
            loc=self._count_lines('severity_classification.py') if os.path.exists(
                os.path.join(self.project_root, 'severity_classification.py')) else 200,
            file_paths=['severity_classification.py'],
            guarantees_affected=[
                "Severity classification accuracy",
                "CWE mapping correctness",
            ],
            mitigations=[
                "Rule-based (deterministic, auditable)",
                "CWE-calibrated against MITRE taxonomy",
                "Not validated against CVE instances",
            ],
        ))
        
        # 11. SMT-LIB certificate extractor
        self.components.append(TCBComponent(
            name="certificate_extractor",
            description="SMT-LIB2 file generation, Alethe proof extraction",
            trust_level=TrustLevel.VALIDATED,
            loc=self._count_lines('smtlib_certificate_extractor.py'),
            file_paths=['smtlib_certificate_extractor.py', 'alethe_proof_extractor.py'],
            guarantees_affected=[
                "Certificate file correctness",
                "Cross-solver reproducibility",
                "Alethe proof validity",
            ],
            mitigations=[
                "Certificates are standalone verifiable files",
                "SAT models verified by direct substitution",
                "Alethe proofs checkable by external validators",
            ],
        ))
    
    def _build_guarantees(self):
        """Map system guarantees to their trust dependencies."""
        
        self.guarantees = [
            GuaranteeDependency(
                guarantee="UNSAT safety proofs (735 pairs)",
                description="If LITMUS∞ reports SAFE, the forbidden outcome is truly unreachable under the model",
                depends_on=["z3_solver", "smt_encoding", "pattern_database", "model_definitions"],
                weakest_link="z3_solver",
                weakest_trust_level=TrustLevel.TRUSTED,
                mitigation_summary="SMT-LIB2 cross-solver replay + Alethe proof extraction + QF_LIA decidability",
            ),
            GuaranteeDependency(
                guarantee="SAT unsafety witnesses (635 pairs)",
                description="If LITMUS∞ reports UNSAFE, a concrete violating execution exists",
                depends_on=["z3_solver", "smt_encoding", "pattern_database", "model_definitions"],
                weakest_link="smt_encoding",
                weakest_trust_level=TrustLevel.VALIDATED,
                mitigation_summary="SAT models self-certifying by direct substitution; 228/228 cross-validation",
            ),
            GuaranteeDependency(
                guarantee="Code recognition (86.1% exact-match)",
                description="Source code correctly identified as matching a litmus test pattern",
                depends_on=["ast_analyzer", "pattern_database"],
                weakest_link="ast_analyzer",
                weakest_trust_level=TrustLevel.TRUSTED,
                mitigation_summary="238-snippet benchmark; unverified front-end does not affect back-end guarantees",
            ),
            GuaranteeDependency(
                guarantee="Compositional safety (Theorems 4-7)",
                description="Multi-pattern programs composed correctly",
                depends_on=["compositional_reasoning", "pattern_database", "z3_solver"],
                weakest_link="compositional_reasoning",
                weakest_trust_level=TrustLevel.TRUSTED,
                mitigation_summary="Paper proofs only; conservative fallback ensures no false negatives",
            ),
            GuaranteeDependency(
                guarantee="Severity classification (423/82/130)",
                description="Unsafe pairs correctly classified by severity",
                depends_on=["severity_classification", "z3_solver", "pattern_database"],
                weakest_link="severity_classification",
                weakest_trust_level=TrustLevel.TRUSTED,
                mitigation_summary="Rule-based and CWE-calibrated; deterministic and auditable",
            ),
            GuaranteeDependency(
                guarantee="Custom model results (DSL)",
                description="User-defined model correctly interpreted",
                depends_on=["dsl_interpreter", "enumeration_engine"],
                weakest_link="dsl_interpreter",
                weakest_trust_level=TrustLevel.VALIDATED,
                mitigation_summary="170/171 correspondence + denotational semantics validation",
            ),
            GuaranteeDependency(
                guarantee="Fence recommendations",
                description="Recommended fences are minimal and sufficient",
                depends_on=["enumeration_engine", "model_definitions", "pattern_database"],
                weakest_link="model_definitions",
                weakest_trust_level=TrustLevel.VALIDATED,
                mitigation_summary="99 UNSAT fence proofs; argmin construction trivially correct",
            ),
            GuaranteeDependency(
                guarantee="herd7 agreement (228/228)",
                description="Tool results match herd7 .cat specifications",
                depends_on=["enumeration_engine", "pattern_database", "model_definitions"],
                weakest_link="enumeration_engine",
                weakest_trust_level=TrustLevel.VALIDATED,
                mitigation_summary="Independent validation pathway against established tool",
            ),
        ]
    
    def generate_report(self, output_dir=None):
        """Generate comprehensive TCB analysis report."""
        if output_dir is None:
            output_dir = os.path.join(self.project_root, 'paper_results_v8', 'tcb_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute metrics
        internal_loc = sum(c.loc for c in self.components if not c.external)
        external_loc = sum(c.estimated_external_loc for c in self.components if c.external)
        total_loc = internal_loc + external_loc
        
        trusted_loc = sum(c.total_loc for c in self.components if c.trust_level == TrustLevel.TRUSTED)
        validated_loc = sum(c.total_loc for c in self.components if c.trust_level == TrustLevel.VALIDATED)
        verified_loc = sum(c.total_loc for c in self.components if c.trust_level == TrustLevel.VERIFIED)
        
        # Component summary
        component_data = []
        for c in self.components:
            component_data.append({
                'name': c.name,
                'description': c.description,
                'trust_level': c.trust_level.value,
                'internal_loc': c.loc,
                'external_loc': c.estimated_external_loc,
                'total_loc': c.total_loc,
                'files': c.file_paths,
                'guarantees_affected': c.guarantees_affected,
                'mitigations': c.mitigations,
            })
        
        # Guarantee dependency analysis
        guarantee_data = []
        for g in self.guarantees:
            guarantee_data.append({
                'guarantee': g.guarantee,
                'description': g.description,
                'depends_on': g.depends_on,
                'weakest_link': g.weakest_link,
                'weakest_trust_level': g.weakest_trust_level.value,
                'mitigation': g.mitigation_summary,
            })
        
        # Comparison with other tools
        comparison = {
            'litmus_infinity': {
                'internal_tcb_loc': internal_loc,
                'external_tcb_loc': external_loc,
                'solver': 'Z3 (~500K LoC)',
                'proof_extraction': 'Alethe format',
                'cross_validation': 'SMT-LIB2 replay, herd7, direct substitution',
                'scope': '137 patterns (not full programs)',
            },
            'dartagnan': {
                'internal_tcb_loc': '~15K (estimated)',
                'external_tcb_loc': '~500K (Z3/MathSAT)',
                'solver': 'Z3 + MathSAT',
                'proof_extraction': 'None',
                'cross_validation': 'None documented',
                'scope': 'Full C programs (bounded)',
            },
            'cbmc': {
                'internal_tcb_loc': '~100K (estimated)',
                'external_tcb_loc': '~50K (MiniSAT/Glucose)',
                'solver': 'MiniSAT/Glucose',
                'proof_extraction': 'DIMACS (SAT only)',
                'cross_validation': 'None documented',
                'scope': 'Full C programs (bounded)',
            },
            'genmc': {
                'internal_tcb_loc': '~20K (estimated)',
                'external_tcb_loc': '~5K (LLVM interpreter)',
                'solver': 'Custom DPOR',
                'proof_extraction': 'Execution traces',
                'cross_validation': 'None documented',
                'scope': 'Full C11 programs',
            },
        }
        
        report = {
            'summary': {
                'internal_loc': internal_loc,
                'external_loc': external_loc,
                'total_loc': total_loc,
                'trusted_loc': trusted_loc,
                'validated_loc': validated_loc,
                'verified_loc': verified_loc,
                'trust_ratio': f"{validated_loc + verified_loc}/{total_loc}",
                'num_components': len(self.components),
                'num_trusted': sum(1 for c in self.components if c.trust_level == TrustLevel.TRUSTED),
                'num_validated': sum(1 for c in self.components if c.trust_level == TrustLevel.VALIDATED),
                'num_verified': sum(1 for c in self.components if c.trust_level == TrustLevel.VERIFIED),
            },
            'components': component_data,
            'guarantees': guarantee_data,
            'comparison': comparison,
            'honest_assessment': {
                'acknowledged_tcb': [
                    "Z3 solver (external, ~500K LoC): all formal results depend on Z3 correctness",
                    "Python encoding logic (~1500 LoC): SMT formula construction",
                    "Pattern database (~2900 LoC): forbidden outcomes and model definitions",
                    "AST analyzer (~2100 LoC): code recognition (does not affect formal guarantees)",
                    "DSL interpreter (~470 LoC): custom model parsing",
                    "Compositional reasoning (~1760 LoC): paper proofs only",
                ],
                'mitigations_in_place': [
                    "SMT-LIB2 export: enables cross-solver validation (Z3, CVC5, Yices)",
                    "Alethe proof extraction: independently checkable UNSAT proofs",
                    "SAT model verification: direct substitution (self-certifying)",
                    "herd7 cross-validation: 228/228 agreement for CPU models",
                    "GPU external validation: 94/94 against published literature",
                    "Denotational semantics: formal meaning for DSL constructs",
                    "QF_LIA decidability: fragment is well-tested across all solvers",
                ],
                'remaining_gaps': [
                    "Compositional theorems (Thms 4-7) are paper proofs, not mechanized",
                    "AST analyzer has no formal correctness guarantee",
                    "No adversarial testing of pattern database correctness",
                    "Severity taxonomy not validated against real CVEs",
                    "Z3 proof production mode may not be exercised in all code paths",
                ],
            },
        }
        
        # Write report
        with open(os.path.join(output_dir, 'tcb_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_report(self):
        """Print a human-readable TCB report."""
        report = self.generate_report()
        
        print("=" * 70)
        print("LITMUS∞ Trusted Computing Base Analysis")
        print("=" * 70)
        
        s = report['summary']
        print(f"\nTotal TCB: {s['total_loc']:,} LoC")
        print(f"  Internal: {s['internal_loc']:,} LoC")
        print(f"  External: {s['external_loc']:,} LoC (Z3 + bindings)")
        print(f"\nTrust classification:")
        print(f"  Verified (machine-checked): {s['num_verified']} components")
        print(f"  Validated (tested): {s['num_validated']} components")
        print(f"  Trusted (assumed correct): {s['num_trusted']} components")
        
        print(f"\n{'Component':<30} {'Trust':<12} {'LoC':>8} {'Mitigations':>12}")
        print("-" * 62)
        for c in report['components']:
            print(f"  {c['name']:<28} {c['trust_level']:<12} {c['total_loc']:>8,} {len(c['mitigations']):>12}")
        
        print(f"\n{'Guarantee':<45} {'Weakest Link':<25} {'Trust':<12}")
        print("-" * 82)
        for g in report['guarantees']:
            print(f"  {g['guarantee']:<43} {g['weakest_link']:<25} {g['weakest_trust_level']}")
        
        return report


def run_tcb_analysis():
    """Entry point for TCB analysis."""
    analyzer = TCBAnalyzer()
    return analyzer.print_report()


if __name__ == '__main__':
    run_tcb_analysis()
