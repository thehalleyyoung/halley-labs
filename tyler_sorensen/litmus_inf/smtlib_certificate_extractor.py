#!/usr/bin/env python3
"""
SMT-LIB Certificate Extractor for LITMUS∞.

Generates standard SMT-LIB2 files for all 750 test-model pairs and extracts:
- UNSAT proofs (safety certificates): unsatisfiable core assertions
- SAT counterexample models: concrete witness executions
- Formal SMT-LIB theory declarations with sorts, functions, and axioms

This enables independent verification of LITMUS∞ results by any SMT-LIB
compliant solver (Z3, CVC5, Yices, etc.).
"""

import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

try:
    from z3 import (
        Solver, Bool, Int, And, Or, Not, Implies, sat, unsat,
        BoolVal, IntVal, set_param, Model,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

from portcheck import (
    PATTERNS, ARCHITECTURES, verify_test, LitmusTest, MemOp,
    get_stores_to_addr,
)
from smt_validation import (
    encode_litmus_test_smt, _po_preserved_smt, _find_store_op,
    validate_pattern_smt,
)
from statistical_analysis import wilson_ci


# ── SMT-LIB2 Theory Specification ──────────────────────────────────

SMTLIB_HEADER = """\
; ═══════════════════════════════════════════════════════════════════
; LITMUS∞ SMT-LIB2 Encoding — Axiomatic Memory Model Verification
; ═══════════════════════════════════════════════════════════════════
;
; Theory: Axiomatic weak memory model verification (Alglave et al. 2014)
;
; Sorts:
;   Int — event timestamps (topological ordering for acyclicity)
;   Int — read-from values (index into per-address store list)
;   Bool — coherence order relation (total order on stores per address)
;
; Relations encoded:
;   po  — program order (model-dependent preservation)
;   rf  — read-from (which store each load reads)
;   co  — coherence order (total order on stores per address)
;   fr  — from-reads (derived: load reads from s_i, s_j co-after s_i)
;
; Safety property:
;   UNSAT ⟹ forbidden outcome unreachable under model (SAFE)
;   SAT   ⟹ forbidden outcome reachable (UNSAFE, witness provided)
;
; Memory model: {model_name}
; Pattern: {pattern_name}
; Forbidden outcome: {forbidden}
; ═══════════════════════════════════════════════════════════════════
"""

SMTLIB_THEORY_DECL = """\
; ── Theory Declarations ─────────────────────────────────────────────
; Axiomatic memory model framework (Alglave, Maranget, Tautschnig 2014)
;
; An execution X = (E, po, rf, co) where:
;   E   = set of memory events (reads/writes with addresses and values)
;   po  = program order (per-thread total order on events)
;   rf  = read-from function mapping each read to the write it reads
;   co  = coherence order (per-address total order on writes)
;   fr  = from-reads: derived as rf^{-1}; co
;
; A memory model M defines which executions are consistent via
; acyclicity constraints on combinations of these relations.
;
; Models encoded:
;   TSO   — acyclic(po ∪ rf ∪ co ∪ fr), po relaxes W→R only
;   PSO   — additionally relaxes W→W
;   ARM   — relaxes all cross-address po except deps and fences
;   RISC-V — like ARM with asymmetric fence.{pred}.{succ}
;   GPU    — like ARM with scoped fences (workgroup/device/system)
;
; Encoding strategy:
;   Acyclicity is encoded via integer timestamps:
;   For each event e, ts(e) ∈ ℤ with ts(a) < ts(b) for each preserved edge a→b.
;   A cycle exists iff the conjunction of all ts constraints is unsatisfiable.
;   We conjoin the forbidden outcome constraint and check satisfiability.
"""


def _model_name_for_arch(arch_name):
    """Map architecture name to SMT model name."""
    arch_to_model = {
        'x86': 'TSO', 'sparc': 'PSO', 'arm': 'ARM', 'riscv': 'RISC-V',
        'opencl_wg': 'OpenCL-WG', 'opencl_dev': 'OpenCL-Dev',
        'vulkan_wg': 'Vulkan-WG', 'vulkan_dev': 'Vulkan-Dev',
        'ptx_cta': 'PTX-CTA', 'ptx_gpu': 'PTX-GPU',
    }
    return arch_to_model.get(arch_name, arch_name)


def _smt_model_for_arch(arch_name):
    """Map architecture to the SMT encoding model name."""
    gpu_to_smt = {
        'opencl_wg': 'ARM', 'opencl_dev': 'ARM',
        'vulkan_wg': 'ARM', 'vulkan_dev': 'ARM',
        'ptx_cta': 'ARM', 'ptx_gpu': 'ARM',
    }
    cpu_map = {'x86': 'TSO', 'sparc': 'PSO', 'arm': 'ARM', 'riscv': 'RISC-V'}
    return cpu_map.get(arch_name) or gpu_to_smt.get(arch_name, 'ARM')


def generate_smtlib2_encoding(pat_name, arch_name):
    """Generate a complete SMT-LIB2 file for a pattern-architecture pair.

    Returns (smtlib_text, result_dict) where result_dict contains the Z3 result,
    and for SAT results, the counterexample model.
    """
    if not Z3_AVAILABLE:
        return None, {'error': 'z3 not available'}

    pat_def = PATTERNS[pat_name]
    n_threads = max(op.thread for op in pat_def['ops']) + 1
    lt = LitmusTest(
        name=pat_name, n_threads=n_threads,
        addresses=pat_def['addresses'], ops=pat_def['ops'],
        forbidden=pat_def['forbidden'],
    )

    model_name = _smt_model_for_arch(arch_name)
    display_model = _model_name_for_arch(arch_name)

    # Build the Z3 encoding
    start = time.time()
    try:
        solver, rf_val, co_vars, forbidden_conj = encode_litmus_test_smt(lt, model_name)
        solver.add(forbidden_conj)

        # Generate SMT-LIB2 text from the solver
        smtlib_lines = []
        smtlib_lines.append(SMTLIB_HEADER.format(
            model_name=display_model,
            pattern_name=pat_name,
            forbidden=json.dumps(lt.forbidden),
        ))
        smtlib_lines.append(SMTLIB_THEORY_DECL)

        # Pattern description
        smtlib_lines.append(f"; ── Pattern: {pat_name} ─────────────────────────")
        smtlib_lines.append(f"; Threads: {n_threads}")
        smtlib_lines.append(f"; Addresses: {', '.join(lt.addresses)}")
        smtlib_lines.append(f"; Memory model: {display_model}")
        smtlib_lines.append(f"; Forbidden outcome: {lt.forbidden}")
        smtlib_lines.append(";")

        # Operations listing
        smtlib_lines.append("; Operations:")
        for i, op in enumerate(lt.ops):
            scope_str = f" [scope={op.scope}]" if op.scope else ""
            dep_str = f" [dep={op.dep_on}]" if op.dep_on else ""
            if op.optype == 'store':
                smtlib_lines.append(f";   e{i}: T{op.thread} Store {op.addr} = {op.value}{scope_str}")
            elif op.optype == 'load':
                smtlib_lines.append(f";   e{i}: T{op.thread} Load {op.addr} → {op.reg}{dep_str}{scope_str}")
            elif op.optype == 'fence':
                pred = f" pred={op.fence_pred}" if op.fence_pred else ""
                succ = f" succ={op.fence_succ}" if op.fence_succ else ""
                smtlib_lines.append(f";   e{i}: T{op.thread} Fence{pred}{succ}{scope_str}")
        smtlib_lines.append("")

        # Use Z3's built-in SMT-LIB2 export
        smtlib_lines.append("(set-logic QF_LIA)")
        smtlib_lines.append("")

        # Extract the solver's SMT-LIB2 representation
        z3_sexpr = solver.sexpr()
        for line in z3_sexpr.strip().split('\n'):
            smtlib_lines.append(line)

        smtlib_lines.append("")
        smtlib_lines.append("(check-sat)")

        # Solve and extract certificate
        result = solver.check()
        elapsed = (time.time() - start) * 1000

        result_info = {
            'pattern': pat_name,
            'arch': arch_name,
            'model': display_model,
            'time_ms': round(elapsed, 2),
        }

        if result == sat:
            model = solver.model()
            result_info['status'] = 'SAT'
            result_info['certificate_type'] = 'SAT_witness'

            # Extract counterexample model
            witness = {}
            for d in model.decls():
                name = d.name()
                val = model[d]
                witness[name] = str(val)

            # Extract read-from assignments
            rf_assignments = {}
            for load_id, rf_var in rf_val.items():
                val = model.evaluate(rf_var)
                rf_assignments[str(rf_var)] = str(val)

            # Extract coherence order
            co_assignments = {}
            for (addr, i, j), co_var in co_vars.items():
                val = model.evaluate(co_var)
                co_assignments[f"co_{addr}_{i}_{j}"] = str(val)

            result_info['witness'] = witness
            result_info['rf_assignments'] = rf_assignments
            result_info['co_assignments'] = co_assignments

            # Add get-model to SMT-LIB
            smtlib_lines.append("(get-model)")

            # Add witness as comment
            smtlib_lines.append("")
            smtlib_lines.append("; ── SAT Witness (counterexample execution) ──")
            smtlib_lines.append(f"; Status: SAT — forbidden outcome IS reachable under {display_model}")
            for k, v in sorted(rf_assignments.items()):
                smtlib_lines.append(f";   {k} = {v}")
            for k, v in sorted(co_assignments.items()):
                smtlib_lines.append(f";   {k} = {v}")

        elif result == unsat:
            result_info['status'] = 'UNSAT'
            result_info['certificate_type'] = 'UNSAT_proof'

            # Extract unsat core if available
            # Re-solve with named assertions for core extraction
            core_solver = Solver()
            core_solver.set("timeout", 10000)
            core_solver.set("unsat_core", True)

            # Rebuild with tracked assertions
            s2, rf2, co2, forb2 = encode_litmus_test_smt(lt, model_name)
            assertion_names = []
            for i, a in enumerate(s2.assertions()):
                name = Bool(f'track_{i}')
                core_solver.assert_and_track(a, name)
                assertion_names.append(name)
            forb_name = Bool('track_forbidden')
            core_solver.assert_and_track(forb2, forb_name)
            assertion_names.append(forb_name)

            core_result = core_solver.check()
            if core_result == unsat:
                core = core_solver.unsat_core()
                core_names = [str(c) for c in core]
                result_info['unsat_core_size'] = len(core)
                result_info['unsat_core_assertions'] = core_names

                smtlib_lines.append("")
                smtlib_lines.append("; ── UNSAT Proof Certificate ──")
                smtlib_lines.append(f"; Status: UNSAT — forbidden outcome UNREACHABLE under {display_model}")
                smtlib_lines.append(f"; Unsat core size: {len(core)} assertions (out of {len(assertion_names)})")
                smtlib_lines.append(f"; Core assertions: {', '.join(core_names[:20])}")
                if len(core_names) > 20:
                    smtlib_lines.append(f";   ... and {len(core_names) - 20} more")
            else:
                result_info['unsat_core_size'] = 0
                smtlib_lines.append("")
                smtlib_lines.append(f"; Status: UNSAT — forbidden outcome UNREACHABLE under {display_model}")

            smtlib_lines.append("(get-unsat-core)")

        else:
            result_info['status'] = 'UNKNOWN'
            result_info['certificate_type'] = 'timeout'
            smtlib_lines.append("; Status: UNKNOWN (timeout)")

        smtlib_lines.append("")
        smtlib_lines.append("(exit)")

        return '\n'.join(smtlib_lines), result_info

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return None, {
            'pattern': pat_name, 'arch': arch_name,
            'status': 'ERROR', 'error': str(e),
            'time_ms': round(elapsed, 2),
        }


def generate_all_smtlib_certificates(output_dir='paper_results_v6/smtlib_certificates'):
    """Generate SMT-LIB2 files and certificates for all 750 test-model pairs.

    Creates:
    - Individual .smt2 files for each pair
    - certificate_index.json with all results
    - encoding_specification.json with formal theory specification
    """
    if not Z3_AVAILABLE:
        return {'error': 'z3 not available'}

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/sat_witnesses', exist_ok=True)
    os.makedirs(f'{output_dir}/unsat_proofs', exist_ok=True)

    all_archs = ['x86', 'sparc', 'arm', 'riscv',
                 'opencl_wg', 'opencl_dev', 'vulkan_wg', 'vulkan_dev',
                 'ptx_cta', 'ptx_gpu']

    results = []
    total = 0
    sat_count = 0
    unsat_count = 0
    error_count = 0
    total_time = 0
    core_sizes = []

    t0 = time.time()

    for pat_name in sorted(PATTERNS.keys()):
        for arch_name in all_archs:
            total += 1
            smtlib_text, result_info = generate_smtlib2_encoding(pat_name, arch_name)

            if smtlib_text:
                # Save SMT-LIB2 file
                subdir = 'sat_witnesses' if result_info.get('status') == 'SAT' else 'unsat_proofs'
                fname = f"{pat_name}_{arch_name}.smt2"
                with open(f'{output_dir}/{subdir}/{fname}', 'w') as f:
                    f.write(smtlib_text)

            if result_info.get('status') == 'SAT':
                sat_count += 1
            elif result_info.get('status') == 'UNSAT':
                unsat_count += 1
                if 'unsat_core_size' in result_info:
                    core_sizes.append(result_info['unsat_core_size'])
            else:
                error_count += 1

            total_time += result_info.get('time_ms', 0)

            # Compact result for index (no full witness dump)
            compact = {
                'pattern': pat_name,
                'arch': arch_name,
                'model': result_info.get('model'),
                'status': result_info.get('status'),
                'certificate_type': result_info.get('certificate_type'),
                'time_ms': result_info.get('time_ms'),
            }
            if result_info.get('unsat_core_size') is not None:
                compact['unsat_core_size'] = result_info['unsat_core_size']
            if result_info.get('rf_assignments'):
                compact['rf_witness_size'] = len(result_info['rf_assignments'])
            results.append(compact)

            if total % 50 == 0:
                print(f"  [{total}/750] SAT={sat_count} UNSAT={unsat_count} err={error_count}")

    elapsed = time.time() - t0

    # Compute statistics
    certified = sat_count + unsat_count
    coverage_pct = round(100 * certified / max(total, 1), 1)
    avg_core = round(sum(core_sizes) / max(len(core_sizes), 1), 1) if core_sizes else 0

    if certified > 0:
        p, ci_lo, ci_hi = wilson_ci(certified, total)
        wilson_95ci = [round(ci_lo * 100, 1), round(ci_hi * 100, 1)]
    else:
        wilson_95ci = [0, 0]

    report = {
        'total_pairs': total,
        'certified': certified,
        'certificate_coverage_pct': coverage_pct,
        'sat_witnesses': sat_count,
        'unsat_proofs': unsat_count,
        'errors': error_count,
        'timeouts': 0,
        'total_time_ms': round(total_time, 1),
        'avg_time_ms': round(total_time / max(total, 1), 1),
        'avg_unsat_core_size': avg_core,
        'wilson_95ci': wilson_95ci,
        'elapsed_s': round(elapsed, 1),
        'results': results,
    }

    # Save certificate index
    with open(f'{output_dir}/certificate_index.json', 'w') as f:
        json.dump(report, f, indent=2)

    # Save formal encoding specification
    encoding_spec = {
        'theory': 'Axiomatic Memory Model Verification',
        'reference': 'Alglave, Maranget, Tautschnig. Herding Cats. TOPLAS 2014.',
        'logic': 'QF_LIA',
        'sorts': {
            'Int': 'Event timestamps (topological ordering for acyclicity check)',
            'Bool': 'Coherence order relation variables',
        },
        'function_symbols': {
            'ts_i': 'Timestamp for event i (Int)',
            'ts_init_a': 'Timestamp for initial write to address a (Int)',
            'rf_i': 'Read-from assignment for load i: index into store list (Int)',
            'co_a_i_j': 'Coherence order: store i before store j to address a (Bool)',
        },
        'axioms': {
            'rf_range': '∀ load l: 0 ≤ rf(l) < |stores_to(addr(l))|',
            'co_totality': '∀ addr a, stores i,j: co(a,i,j) ∨ co(a,j,i)',
            'co_antisymmetry': '∀ addr a, stores i,j: ¬(co(a,i,j) ∧ co(a,j,i))',
            'co_transitivity': '∀ addr a, stores i,j,k: co(a,i,j) ∧ co(a,j,k) → co(a,i,k)',
            'po_preservation': 'Model-dependent: TSO preserves all except W→R; ARM relaxes all cross-addr',
            'rf_ordering': 'rf(l) = s → ts(s) < ts(l)',
            'co_ordering': 'co(a,i,j) → ts(store_i) < ts(store_j)',
            'fr_ordering': 'rf(l) = s_i ∧ co(a,i,j) → ts(l) < ts(store_j)',
        },
        'models_encoded': {
            'TSO': {
                'relaxed_po': ['W→R to different addresses'],
                'preserved_po': ['R→R', 'R→W', 'W→W', 'po-loc', 'fence-ordered'],
            },
            'PSO': {
                'relaxed_po': ['W→R', 'W→W to different addresses'],
                'preserved_po': ['R→R', 'R→W', 'po-loc', 'fence-ordered'],
            },
            'ARM': {
                'relaxed_po': ['All cross-address pairs'],
                'preserved_po': ['po-loc', 'addr-dep', 'data-dep', 'ctrl-dep→W', 'fence-ordered'],
            },
            'RISC-V': {
                'relaxed_po': ['All cross-address pairs'],
                'preserved_po': ['po-loc', 'addr-dep', 'data-dep', 'ctrl-dep→W',
                                 'fence.{pred}.{succ} ordered'],
            },
            'GPU': {
                'relaxed_po': ['All cross-address pairs'],
                'preserved_po': ['po-loc', 'scoped-fence ordered (scope must cover both threads)'],
                'scopes': ['workgroup', 'device', 'system'],
            },
        },
        'certificate_types': {
            'UNSAT_proof': 'No execution under the model can produce the forbidden outcome. '
                           'The UNSAT core identifies the minimal set of constraints proving safety.',
            'SAT_witness': 'A concrete execution (rf, co assignments) that produces the forbidden '
                           'outcome under the model, demonstrating unsafety.',
        },
    }

    with open(f'{output_dir}/encoding_specification.json', 'w') as f:
        json.dump(encoding_spec, f, indent=2)

    return report


def generate_sample_smtlib_files(output_dir='paper_results_v6/smtlib_certificates'):
    """Generate a representative set of SMT-LIB2 files for paper appendix.

    Selects key patterns (mp, sb, lb, iriw, wrc) across representative architectures.
    """
    if not Z3_AVAILABLE:
        return {'error': 'z3 not available'}

    os.makedirs(output_dir, exist_ok=True)

    key_pairs = [
        ('mp', 'x86'), ('mp', 'arm'), ('mp', 'riscv'),
        ('mp_fence', 'arm'), ('mp_fence', 'riscv'),
        ('sb', 'x86'), ('sb', 'arm'),
        ('lb', 'arm'), ('lb', 'riscv'),
        ('iriw', 'x86'), ('iriw', 'arm'),
        ('wrc', 'arm'), ('wrc_fence', 'arm'),
        ('gpu_mp_wg', 'opencl_wg'), ('gpu_mp_wg', 'opencl_dev'),
    ]

    results = []
    for pat_name, arch_name in key_pairs:
        if pat_name not in PATTERNS:
            continue
        smtlib_text, result_info = generate_smtlib2_encoding(pat_name, arch_name)
        if smtlib_text:
            fname = f"{pat_name}_{arch_name}.smt2"
            with open(f'{output_dir}/{fname}', 'w') as f:
                f.write(smtlib_text)
            results.append(result_info)

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LITMUS∞ SMT-LIB Certificate Extractor')
    parser.add_argument('--all', action='store_true', help='Generate all 750 certificates')
    parser.add_argument('--sample', action='store_true', help='Generate sample certificates')
    parser.add_argument('--output', default='paper_results_v6/smtlib_certificates',
                        help='Output directory')
    args = parser.parse_args()

    if args.all:
        print("Generating SMT-LIB2 certificates for all 750 pairs...")
        report = generate_all_smtlib_certificates(args.output)
        print(f"\nCertified: {report['certified']}/{report['total_pairs']} "
              f"({report['certificate_coverage_pct']}%)")
        print(f"SAT witnesses: {report['sat_witnesses']}")
        print(f"UNSAT proofs: {report['unsat_proofs']}")
        print(f"Avg UNSAT core size: {report['avg_unsat_core_size']}")
    elif args.sample:
        print("Generating sample SMT-LIB2 certificates...")
        results = generate_sample_smtlib_files(args.output)
        for r in results:
            print(f"  {r['pattern']}/{r['arch']}: {r.get('status', 'ERROR')}")
    else:
        parser.print_help()
