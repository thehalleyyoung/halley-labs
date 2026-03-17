#!/usr/bin/env python3
"""Generate approach.json for NegSynth theory stage."""
import json, os

data = {
  "title": "Negotiation Under Fire: Bounded-Complete Synthesis of Protocol Downgrade Attacks from Library Source Code",
  "slug": "proto-downgrade-synth",
  "theory_stage_verdict": "CONTINUE",
  "theory_health_score": 7.5,
  "target_venues": ["IEEE S&P", "USENIX Security", "ACM CCS"],
  "best_paper_probability": "5-10%",
  "acceptance_probability": "45-55%",

  "definitions": [
    {
      "id": "D1",
      "name": "Negotiation Protocol LTS",
      "formal_statement": "A Negotiation Protocol Labeled Transition System is a tuple N = (S, S_0, Lambda, delta, O, obs) where S is a finite set of negotiation states s = (pi, C_offered, c_sel, v, E) with pi in {init, ch_sent, sh_recv, negotiated, done, abort}, C_offered subset of C (IANA cipher universe), c_sel in C union {bot}, v in V (version set), E subset of E (extensions). Lambda = Lambda_C + Lambda_S + Lambda_A partitions labels into client, server, adversary. delta: S x Lambda -> S is a partial deterministic transition function. O is the observation domain of negotiation outcomes. obs: S -> O union {bot} extracts observable from terminal states.",
      "axioms": {
        "A1_finite_outcome": "The sets C, V, E are finite. |C| <= 350 (IANA), |V| <= 6, |E| <= 30.",
        "A2_lattice_preferences": "There exists a partial order <=_pref on C forming a bounded lattice. The selection function select: P(C) x P(C) -> C is monotone.",
        "A3_monotonic_progression": "Handshake phases form an acyclic DAG with strict total order <_pi. For honest transitions delta(s, l) = s', s'.pi >=_pi s.pi.",
        "A4_deterministic_selection": "Given fixed offered cipher sets C_C, C_S, version pair (v_C, v_S), and extensions E_C, E_S, |{c | c = select(C_C cap C_S, <=_pref)}| = 1."
      },
      "serves_module": "State Machine Extractor, Protocol Modules",
      "load_bearing_justification": "Foundation for all downstream formal objects. Without D1, the merge operator, bisimulation, and SMT encoding have no domain to operate on."
    },
    {
      "id": "D2",
      "name": "Protocol-Aware Merge Operator",
      "formal_statement": "The merge operator bowtie: Sigma x Sigma -> Sigma is a partial function on symbolic negotiation states hat_s = (s, phi). Mergeability predicate: mergeable(hat_s1, hat_s2) iff (1) s1.pi = s2.pi, (2) s1.C_offered = s2.C_offered, (3) s1.v = s2.v, (4) phi1 OR phi2 is satisfiable. Merge construction: hat_s1 bowtie hat_s2 = (s_merged, phi1 OR phi2) where s_merged.c_sel = ITE(phi1 AND NOT phi2, s1.c_sel, ITE(phi2 AND NOT phi1, s2.c_sel, s1.c_sel)).",
      "serves_module": "Merge Operator (~7K LoC), KLEE Integration Layer",
      "load_bearing_justification": "Crown algorithmic contribution. Enables tractable analysis of production libraries by reducing O(2^n) paths to O(n*m). Without D2, symbolic execution is infeasible on real code."
    },
    {
      "id": "D3",
      "name": "Protocol Bisimulation",
      "formal_statement": "A protocol bisimulation on N is a symmetric relation R subset S x S such that: (1) Observation agreement: if obs(s1) != bot then obs(s2) != bot and obs(s1) = obs(s2). (2) Transfer property: for all l in Lambda, if delta(s1,l) = s1' then exists s2' with delta(s2,l) = s2' and (s1',s2') in R. States s1, s2 are protocol-bisimilar (s1 ~_P s2) if such R exists.",
      "serves_module": "State Machine Extractor (bisimulation quotient), Theorem T3",
      "load_bearing_justification": "Defines the correctness criterion for the merge operator. Without D3, we cannot state that merging preserves behavior."
    },
    {
      "id": "D4",
      "name": "Bounded Dolev-Yao Adversary",
      "formal_statement": "Message algebra T over sorted signature Sigma_DY with atoms (nonces, keys, cipher IDs, versions, bytes), constructors (enc_s, enc_a, mac, hash, pair, record, packet), destructors (dec_s, dec_a, fst, snd, verify). Knowledge set K is closure of observed messages under destructors and public constructors. A (k,n)-bounded DY adversary is a sequence of <= n actions (intercept, inject, drop, modify) applied to execution of depth <= k.",
      "serves_module": "DY+SMT Encoder (~10K LoC), Concretizer",
      "load_bearing_justification": "Defines the attack model. Without D4, the SMT encoding has no adversary to reason about."
    },
    {
      "id": "D5",
      "name": "Downgrade Freedom Property",
      "formal_statement": "Define <=_sec on C as a total preorder reflecting cryptographic strength. Honest outcome o_H = (c_H, v_H, E_H) = obs(s_done) without adversary. A (k,n)-bounded adversary trace a is a downgrade attack if obs(exec(N, a)) = (c', v', E') with c' <_sec c_H or v' < v_H. N is (k,n)-downgrade-free if no such trace exists.",
      "serves_module": "DY+SMT Encoder (property encoding), Certificate generation",
      "load_bearing_justification": "Defines what NegSynth searches for. Without D5, 'attack' and 'certificate' are undefined."
    },
    {
      "id": "D6",
      "name": "Bounded-Completeness Certificate",
      "formal_statement": "A certificate is a triple Cert = (l, (k,n), Pi) where l is a library identifier with version/commit, (k,n) are bounds, Pi is an UNSAT proof from the SMT solver. Cert is valid under Assumption A0 if: T1 (extraction sound), T3 (merge correct), T5 (encoding correct), and Pi is a valid UNSAT witness.",
      "serves_module": "Certificate generation, CLI/Reporting module",
      "load_bearing_justification": "Primary empirical contribution -- the first formal downgrade-freedom guarantee for production TLS/SSH libraries."
    },
    {
      "id": "D7",
      "name": "Observation Function",
      "formal_statement": "obs(s) = (s.c_sel, s.v, s.E) if s.pi in {done, abort}; bot otherwise. Explicitly excludes: alert codes, timing, message byte-content beyond negotiation outcome, internal error states.",
      "attack_class_scope": {
        "in_scope": ["Cipher downgrade (FREAK, Logjam)", "Version downgrade (POODLE version component)", "Extension stripping (Terrapin)", "Message-ordering causing downgrade (CCS Injection)"],
        "out_of_scope": ["Alert-code oracles (POODLE padding)", "Timing side-channels (Lucky13)", "Implementation-level crypto oracles (DROWN padding)", "Denial-of-service"]
      },
      "serves_module": "All formal objects, scopes the certificate"
    },
    {
      "id": "D8",
      "name": "Adversary Budget Semantics",
      "formal_statement": "Each atomic adversary operation costs 1 unit of budget: intercept(i) = 1, inject(t) = 1, drop(i) = 1, modify(i,f) = 1. modify is a single atomic operation (intercept + replace), not 2.",
      "per_cve_bounds": {
        "CVE-2015-0204_FREAK": {"min_n": 2, "min_k": 12},
        "CVE-2015-4000_Logjam": {"min_n": 2, "min_k": 12},
        "CVE-2014-3566_POODLE": {"min_n": 1, "min_k": 10},
        "CVE-2023-48795_Terrapin": {"min_n": 3, "min_k": 10},
        "CVE-2016-0703_DROWN": {"min_n": 2, "min_k": 15},
        "CVE-2015-3197_SSLv2_override": {"min_n": 1, "min_k": 10},
        "CVE-2014-0224_CCS_Injection": {"min_n": 2, "min_k": 15},
        "CVE-2015-0291_sigalgs_DoS": {"min_n": 1, "min_k": 8}
      },
      "headroom": "Max observed: n=3 (Terrapin), k=15 (DROWN/CCS). Default bounds n=5,k=20 provide 67% headroom.",
      "serves_module": "DY+SMT Encoder, Certificate interpretation"
    }
  ],

  "assumptions": [
    {
      "id": "A0", "name": "Slicer Soundness",
      "statement": "The protocol-aware slicer computes a superset of all code paths that can influence the negotiation outcome as defined by obs (D7).",
      "validation_method": "CVE reachability (8/8), random path sampling (10K traces, miss rate < 0.1%), differential testing of assembly stubs",
      "risk_level": "MEDIUM", "failure_consequence": "Silent false certificates"
    },
    {"id": "A-KLEE", "name": "KLEE Correctness", "statement": "KLEE correctly executes LLVM IR of sliced code.", "risk_level": "LOW"},
    {"id": "A-Z3", "name": "Z3 Correctness", "statement": "Z3 correctly decides queries it returns results for.", "risk_level": "LOW"},
    {"id": "A-STUB", "name": "Assembly Stub Faithfulness", "statement": "C stubs faithfully model replaced assembly routines.", "validation_method": "Differential testing: 10K random inputs", "risk_level": "LOW-MEDIUM"},
    {"id": "A-PROTO", "name": "Protocol Module Correctness", "statement": "Protocol modules (~20K LoC) correctly encode RFC semantics.", "validation_method": "RFC test vectors, community review", "risk_level": "LOW-MEDIUM"}
  ],

  "theorems": [
    {
      "id": "T1", "name": "Extraction Soundness",
      "formal_statement": "Let P be the source program and N_P the extracted Negotiation LTS. There exists a simulation relation R such that: (1) every trace of N_P corresponds to a feasible path in P, (2) every reachable negotiation state in P maps to a reachable state in N_P.",
      "proof_strategy": "Forward simulation by induction on execution steps. Stuttering for internal steps. Merge abstractions via L1.",
      "key_lemmas": ["L1 (Merge Congruence)"], "depends_on_assumptions": ["A0"],
      "proof_difficulty": 3, "novel_vs_known": "Adapted from standard simulation relations (Milner 1989)",
      "serves_module": "negsyn-slicer + negsyn-extract", "risk_pct": 5, "effort_months": 2
    },
    {
      "id": "T2", "name": "Attack Trace Concretizability",
      "formal_statement": "SAT models concretize to valid byte-level traces with rate >= 1 - epsilon (epsilon < 0.01). CEGAR terminates in <= |C|^k iterations.",
      "proof_strategy": "CEGAR soundness + convergence over finite domain. Framing correctness via L5.",
      "key_lemmas": ["L5 (Framing Correctness)", "L6 (CEGAR Termination)"],
      "proof_difficulty": 3.5, "novel_vs_known": "Adapted CEGAR with novel DY-domain refinement predicates",
      "serves_module": "negsyn-concrete", "risk_pct": 15, "effort_months": 3
    },
    {
      "id": "T3", "name": "Protocol-Aware Merge Correctness", "is_crown": True,
      "formal_statement": "(Correctness) hat_s1 bowtie hat_s2 ~_P hat_s1 || hat_s2. (Complexity) O(n*m) states for n ciphers, m phases satisfying A1-A4 vs O(2^n * m) for generic veritesting.",
      "proof_strategy": "Part 1: Bisimulation up-to congruence using A1-A4. Part 2: Counting argument from finite outcome space + deterministic selection.",
      "graceful_fallback": "Per-region property checker; when A1-A4 violated, no merge, fall back to generic exploration. ~10-15% of real code needs fallback.",
      "key_lemmas": ["L1 (Merge Congruence)", "L2 (Bounded Branching)"],
      "proof_difficulty": 5, "novel_vs_known": "Genuinely new: domain identification + polynomial bound for structured merge",
      "serves_module": "negsyn-merge", "risk_pct": 10, "effort_months": 2
    },
    {
      "id": "T4", "name": "Bounded Completeness", "is_headline": True,
      "formal_statement": "Within (k,n), NegSynth finds every downgrade attack (prob >= 1-epsilon) or certifies absence, under Assumption A0.",
      "proof_strategy": "Three-level composition: T1 -> T3 -> T5, with T2 for ATTACK path. CompCert-style transitivity.",
      "epsilon_definition": "epsilon = fraction of SAT models where CEGAR fails within 3 iterations. Applies ONLY to ATTACK path.",
      "bounds_validation": {"structural": "TLS <= 10 round trips, k=20 = 2x", "empirical": "All CVEs need k<=15, n<=5", "coverage": ">=99% states at k=20,n=5"},
      "key_lemmas": ["All of T1, T3, T5"],
      "proof_difficulty": 4, "novel_vs_known": "New composition, no prior end-to-end guarantee for protocol analysis",
      "serves_module": "Pipeline orchestrator", "risk_pct": 15, "effort_months": 3
    },
    {
      "id": "T5", "name": "SMT Encoding Correctness",
      "formal_statement": "Phi_{l,(k,n)} is equisatisfiable with existence of (k,n)-downgrade attack on N_l.",
      "proof_strategy": "Encoding-decoding bijection. SAT -> Attack extraction. Attack -> SAT construction.",
      "key_lemmas": ["L4 (DY Knowledge Monotonicity)"],
      "proof_difficulty": 3, "novel_vs_known": "Adapted from AVISPA DY encoding with novel negotiation LTS encoding",
      "serves_module": "negsyn-encode", "risk_pct": 10, "effort_months": 2
    },
    {
      "id": "C1", "name": "Covering-Design Differential Completeness", "extension_only": True,
      "formal_statement": "B(n,k,t) test configs guarantee detection of all t-way behavioral deviations between N >= 3 libraries.",
      "proof_strategy": "Stein-Lovasz-Johnson bound + detection argument",
      "proof_difficulty": 6, "novel_vs_known": "Non-obvious connection between combinatorial designs and protocol testing",
      "serves_module": "negsyn-differential (Phase 2)", "risk_pct": 15, "effort_months": 3
    }
  ],

  "lemmas": [
    {"id": "L1", "name": "Merge Congruence", "statement": "bowtie is a congruence w.r.t. ~_P", "difficulty": 2, "serves": "T3"},
    {"id": "L2", "name": "Bounded Branching", "statement": "LTS branching factor <= |C| + n", "difficulty": 1, "serves": "T3, T4"},
    {"id": "L4", "name": "DY Knowledge Monotonicity", "statement": "K_i subset K_j for i < j", "difficulty": 1, "serves": "T5"},
    {"id": "L5", "name": "Framing Correctness", "statement": "parse(frame(sigma)) = extract_trace(sigma)", "difficulty": 2, "serves": "T2"},
    {"id": "L6", "name": "CEGAR Termination", "statement": "CEGAR terminates in <= |C|^k * |V|^k iterations", "difficulty": 2, "serves": "T2"}
  ],

  "algorithms": [
    {
      "id": "ALG1", "name": "PROTOSLICE", "description": "Protocol-aware slicer",
      "input": "LLVM IR bitcode, protocol entry points",
      "output": "Negotiation-relevant LLVM IR slice (target <= 2% of source)",
      "time_complexity": "O(V*E)", "space_complexity": "O(V^2)",
      "practical_estimate": "~10 min on OpenSSL",
      "module": "negsyn-slicer (~11K LoC)"
    },
    {
      "id": "ALG2", "name": "PROTOMERGE", "description": "Protocol-aware merge operator (CROWN)",
      "input": "Two symbolic states during KLEE exploration",
      "output": "Merged state or FAIL",
      "time_complexity": "O(n*m) total states", "space_complexity": "O(n*m)",
      "practical_estimate": "10-100x path reduction, 1-3h symex",
      "module": "negsyn-merge (~7K LoC) + negsyn-klee (~8K LoC)"
    },
    {
      "id": "ALG3", "name": "SMEXTRACT", "description": "Bisimulation-quotient state machine extractor",
      "input": "Symbolic execution traces", "output": "Finite state machine",
      "time_complexity": "O(S*logS)", "space_complexity": "O(S^2)",
      "practical_estimate": "~10 min", "module": "negsyn-extract (~8K LoC)"
    },
    {
      "id": "ALG4", "name": "DYENCODE", "description": "DY+SMT constraint encoder",
      "input": "State machine, message grammar, downgrade property",
      "output": "SMT formula in BV+Arrays+UF+LIA",
      "time_complexity": "O(S*Lambda*k)", "space_complexity": "O(S*k*n)",
      "practical_estimate": "30-60 min encode + 1-4h solve",
      "module": "negsyn-encode (~10K LoC)"
    },
    {
      "id": "ALG5", "name": "CONCRETIZE", "description": "CEGAR concretization loop",
      "input": "Satisfying SMT assignment", "output": "Byte-level attack trace or refinement",
      "time_complexity": "O(k*n) per attempt", "space_complexity": "O(k)",
      "practical_estimate": "~15 min", "module": "negsyn-concrete (~6K LoC)"
    }
  ],

  "complexity_analysis": {
    "end_to_end": {
      "total_time": "4-8 hours per library (8-core laptop, 32GB RAM)",
      "breakdown": {"slicing": "~10 min", "symbolic_execution": "1-3 hours", "extraction": "~10 min", "encoding": "30-60 min", "solving": "1-4 hours", "concretization": "~15 min"},
      "full_benchmark": "20 configs, 80-160h sequential, 10-20h with 8-way parallelism",
      "peak_memory": "16-24 GB (Z3 dominated)"
    }
  },

  "scope_exclusions": [
    {"id": "SE-1", "what": "Multi-renegotiation attacks (TLS 1.2, >1 cycle)", "why": "Unbounded protocol depth"},
    {"id": "SE-2", "what": "Timing side-channels", "why": "Invisible to symbolic execution"},
    {"id": "SE-3", "what": "Implementation-level crypto oracles", "why": "DY assumes perfect crypto"},
    {"id": "SE-4", "what": "Cross-session attacks", "why": "Single-session adversary model"},
    {"id": "SE-5", "what": "Dynamically-loaded providers (OpenSSL 3.x)", "why": "KLEE limitation"},
    {"id": "SE-6", "what": "Attacks with n>5 or k>20", "why": "Explicit in certificate bounds"},
    {"id": "SE-7", "what": "Denial-of-service", "why": "obs captures outcome, not availability"}
  ],

  "experimental_plan": {
    "hypotheses": [
      {"id": "H1", "claim": "CVE recall >= 7/8", "falsification": "< 7/8", "experiment": "E1"},
      {"id": "H2", "claim": "Merge speedup >= 10x", "falsification": "< 10x on >= 2 libs", "experiment": "E2"},
      {"id": "H3", "claim": "State coverage >= 99%", "falsification": "< 95%", "experiment": "E3"},
      {"id": "H4", "claim": "Per-lib time < 8h", "falsification": "> 12h", "experiment": "E4"},
      {"id": "H5", "claim": "FP rate < 10%", "falsification": ">= 15%", "experiment": "E5,E6"},
      {"id": "H6", "claim": "epsilon <= 0.01", "falsification": "> 0.05", "experiment": "E6"}
    ],
    "experiments": [
      {"id": "E1", "name": "Known CVE Recovery", "subjects": "8 CVEs x vulnerable versions", "metrics": "Recall, time-to-synthesis, min (k,n)"},
      {"id": "E2", "name": "Merge Money Plot", "subjects": "4 libs x cipher sweep 5-50", "metrics": "Path count with/without merge"},
      {"id": "E3", "name": "Certificates", "subjects": "4 current HEAD libs", "metrics": "Certificate, coverage%, findings"},
      {"id": "E4", "name": "Scalability", "subjects": "All 4 libs", "metrics": "Per-stage time, peak memory"},
      {"id": "E5", "name": "Tool Comparison", "subjects": "vs tlspuffin, KLEE, TLS-Attacker", "metrics": "Bugs, time, FP rate"},
      {"id": "E6", "name": "Concretization", "subjects": "All SAT queries", "metrics": "Success rate, CEGAR iterations, epsilon"}
    ],
    "baselines": ["tlspuffin (S&P 2024, DY-fuzzer)", "KLEE vanilla (ablation)", "TLS-Attacker (black-box)"]
  },

  "implementation_mapping": {
    "modules": [
      {"name": "negsyn-slicer", "lang": "Rust", "loc": 11000, "theorems": ["T1"]},
      {"name": "negsyn-merge", "lang": "Rust/C++", "loc": 7000, "theorems": ["T3"]},
      {"name": "negsyn-klee", "lang": "C++/Rust", "loc": 8000, "theorems": ["T3"]},
      {"name": "negsyn-extract", "lang": "Rust", "loc": 8000, "theorems": ["T1"]},
      {"name": "negsyn-encode", "lang": "Rust", "loc": 10000, "theorems": ["T5"]},
      {"name": "negsyn-concrete", "lang": "Rust+Python", "loc": 6000, "theorems": ["T2"]},
      {"name": "negsyn-proto-tls", "lang": "Rust", "loc": 12000, "theorems": []},
      {"name": "negsyn-proto-ssh", "lang": "Rust", "loc": 8000, "theorems": []},
      {"name": "negsyn-eval", "lang": "Rust+Python", "loc": 12000, "theorems": []},
      {"name": "negsyn-cli", "lang": "Rust", "loc": 7000, "theorems": []},
      {"name": "negsyn-test", "lang": "Rust+Shell", "loc": 11000, "theorems": ["T1","T3","T5"]}
    ],
    "total_novel_loc": 50000, "total_integration_loc": 40000,
    "reused": ["KLEE (~95K)", "Z3", "tlspuffin DY model", "TLS-Attacker"]
  },

  "risk_matrix": [
    {"id": "R1", "risk": "KLEE + OpenSSL bitcode fails", "prob": 0.30, "impact": "Critical", "gate": "G0 (Week 4)"},
    {"id": "R2", "risk": "Slicer > 5% of source", "prob": 0.25, "impact": "Critical", "gate": "G1 (Week 6)"},
    {"id": "R3", "risk": "Z3 timeout on DY+SMT", "prob": 0.40, "impact": "High", "gate": "G2 (Week 10)"},
    {"id": "R4", "risk": "Merge < 10x speedup", "prob": 0.20, "impact": "High", "gate": "G2 (Week 10)"},
    {"id": "R5", "risk": "Rust/C++ FFI soundness", "prob": 0.25, "impact": "Medium", "gate": "Ongoing"},
    {"id": "R6", "risk": "No new vulnerability", "prob": 0.50, "impact": "Low", "gate": "None"},
    {"id": "R7", "risk": "epsilon > 0.01", "prob": 0.20, "impact": "Medium", "gate": "epsilon > 0.05 = KILL"}
  ],

  "critique_resolutions": {
    "critical_resolved": 4, "serious_resolved": 7,
    "key_decisions": [
      "Graceful fallback for algebraic property violations (per-region checker)",
      "Slicer soundness as Assumption A0 with empirical validation",
      "Narrow observation function with explicit scope table",
      "Epsilon covers concretization failure ONLY",
      "Atomic adversary budget counting with per-CVE table"
    ]
  },

  "literature_connections": {
    "builds_on": [
      "KLEE (Cadar et al., OSDI 2008) -- symbolic execution foundation",
      "Veritesting (Avgerinos et al., ICSE 2014) -- generic state merging",
      "State merging (Kuznetsov et al., PLDI 2012) -- cost-based merging",
      "Milner bisimulation (1989) -- behavioral equivalence",
      "Dolev-Yao model (1983) -- adversary model",
      "tlspuffin (S&P 2024) -- DY term algebra",
      "CEGAR (Clarke et al., CAV 2000) -- refinement loop",
      "CompCert (Leroy, POPL 2006) -- composition template"
    ],
    "novel_contributions": [
      "First bounded-complete downgrade synthesis from implementation source code",
      "First polynomial path bound for merge operator on structured program domain",
      "First bounded-completeness certificates for production TLS/SSH libraries"
    ]
  }
}

path = '/Users/halleyyoung/Documents/div/mathdivergence/pipeline_100/area-088-security-privacy-and-cryptography/proto-downgrade-synth/theory/approach.json'
with open(path, 'w') as f:
    json.dump(data, f, indent=2)
print(f'Written {os.path.getsize(path)} bytes to approach.json')
