"""
CoaCert-TLA Specification Library.

Built-in TLA-lite specifications for benchmarking and testing the
CoaCert-TLA coalgebraic certification pipeline.  Each specification
is a programmatic AST builder that produces a ``Module`` from
``coacert.parser.ast_nodes``.

Included specifications:
  - TwoPhaseCommit  — Two-Phase Commit protocol
  - LeaderElection  — Ring-based leader election (Chang-Roberts)
  - Peterson        — Peterson's mutual exclusion / filter lock
  - Paxos           — Single-decree Paxos (Synod)
"""

from .two_phase_commit import TwoPhaseCommitSpec
from .leader_election import LeaderElectionSpec
from .peterson import PetersonSpec
from .paxos import PaxosSpec
from .spec_registry import SpecRegistry, SpecMetadata, BenchmarkPreset
from .spec_utils import (
    ModuleBuilder,
    ident,
    primed,
    int_lit,
    bool_lit,
    str_lit,
    make_variable_decl,
    make_constant_decl,
    make_operator_def,
    make_set_enum,
    make_string_set,
    make_int_set,
    make_range,
    make_int_range,
    make_function_construction,
    make_func_apply,
    make_record,
    make_record_access,
    make_tuple,
    make_conjunction,
    make_disjunction,
    make_forall,
    make_exists,
    make_primed,
    make_unchanged,
    make_wf,
    make_sf,
    make_fairness,
    make_always,
    make_eventually,
    make_leads_to,
    make_invariant_property,
    make_safety_property,
    make_liveness_property,
    make_temporal_property,
)

__all__ = [
    # Spec builders
    "TwoPhaseCommitSpec",
    "LeaderElectionSpec",
    "PetersonSpec",
    "PaxosSpec",
    # Registry
    "SpecRegistry",
    "SpecMetadata",
    "BenchmarkPreset",
    # Module builder
    "ModuleBuilder",
    # AST construction utilities
    "ident",
    "primed",
    "int_lit",
    "bool_lit",
    "str_lit",
    "make_variable_decl",
    "make_constant_decl",
    "make_operator_def",
    "make_set_enum",
    "make_string_set",
    "make_int_set",
    "make_range",
    "make_int_range",
    "make_function_construction",
    "make_func_apply",
    "make_record",
    "make_record_access",
    "make_tuple",
    "make_conjunction",
    "make_disjunction",
    "make_forall",
    "make_exists",
    "make_primed",
    "make_unchanged",
    "make_wf",
    "make_sf",
    "make_fairness",
    "make_always",
    "make_eventually",
    "make_leads_to",
    "make_invariant_property",
    "make_safety_property",
    "make_liveness_property",
    "make_temporal_property",
]
