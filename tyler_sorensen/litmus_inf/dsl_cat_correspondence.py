#!/usr/bin/env python3
"""
DSL-to-.cat Formal Correspondence Validation for LITMUS∞.

Extends the formal DSL-to-.cat correspondence from x86-TSO to ARM and RISC-V.

The .cat memory model specification language (used by herd7) is the de facto
standard for formally defining memory models. This module validates that the
LITMUS∞ DSL model definitions (in model_dsl.py) produce identical portability
predictions as the .cat reference specifications.

Validation methodology:
  1. For each memory model (TSO, ARM, RISC-V), define the .cat axioms
  2. Run every litmus pattern through both the DSL checker and the .cat
     reference checker (portcheck.py, which was validated against herd7)
  3. Report agreement rate with Wilson confidence intervals
  4. Identify and classify any mismatches

This provides formal correspondence evidence for ALL models, not just TSO.

References:
  - herd7 .cat specifications: x86tso.cat, aarch64.cat, riscv.cat
  - Alglave et al., "Herding Cats", TOPLAS 2014
  - Pulte et al., "Simplifying ARM Concurrency", POPL 2018
  - RISC-V Memory Consistency Model spec, Chapter 14
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from portcheck import (
    PATTERNS, ARCHITECTURES, verify_test, LitmusTest, MemOp,
    HERD7_EXPECTED, get_stores_to_addr,
)
from model_dsl import ModelRegistry, CustomModel, FenceSpec, get_registry
from statistical_analysis import wilson_ci

# .cat axiom definitions for each model
# These mirror the herd7 .cat files' semantics
CAT_AXIOMS = {
    'x86tso': {
        'description': 'x86-TSO: Total Store Order (x86tso.cat)',
        'relaxed': {('store', 'load')},
        'preserves_deps': True,
        'multi_copy_atomic': True,
        'reference': 'x86tso.cat (Owens, Sarkar, Sewell 2009)',
    },
    'aarch64': {
        'description': 'AArch64: ARMv8 (aarch64.cat, Pulte et al. 2018)',
        'relaxed': {('store', 'load'), ('store', 'store'),
                    ('load', 'load'), ('load', 'store')},
        'preserves_deps': True,
        'multi_copy_atomic': False,
        'reference': 'aarch64.cat (Pulte et al. POPL 2018)',
    },
    'riscv': {
        'description': 'RVWMO: RISC-V Weak Memory Ordering (riscv.cat)',
        'relaxed': {('store', 'load'), ('store', 'store'),
                    ('load', 'load'), ('load', 'store')},
        'preserves_deps': True,
        'multi_copy_atomic': False,
        'reference': 'riscv.cat (RISC-V ISA spec, Chapter 14)',
    },
}

# DSL definitions that should correspond to the .cat axioms
DSL_DEFINITIONS = {
    'x86tso': """
model x86_TSO {
    description "Total Store Order - x86tso.cat correspondence"
    relaxes W->R
    preserves deps
    fence mfence (cost=4) { orders W->R, W->W, R->R, R->W }
}
""",
    'aarch64': """
model AArch64 {
    description "ARMv8 relaxed - aarch64.cat correspondence"
    relaxes W->R, W->W, R->R, R->W
    preserves deps
    not multi-copy-atomic
    fence dmb_ishst (cost=1) { orders W->W }
    fence dmb_ishld (cost=1) { orders R->R, R->W }
    fence dmb_ish (cost=4) { orders W->R, W->W, R->R, R->W }
}
""",
    'riscv': """
model RVWMO {
    description "RISC-V Weak Memory Ordering - riscv.cat correspondence"
    relaxes W->R, W->W, R->R, R->W
    preserves deps
    not multi-copy-atomic
    fence fence_rr (cost=1) { orders R->R }
    fence fence_ww (cost=1) { orders W->W }
    fence fence_rw (cost=1) { orders R->W }
    fence fence_wr (cost=2) { orders W->R }
    fence fence_rwrw (cost=4) { orders W->R, W->W, R->R, R->W }
}
""",
}

# Mapping from .cat model names to portcheck.py architecture names
CAT_TO_ARCH = {
    'x86tso': 'x86',
    'aarch64': 'arm',
    'riscv': 'riscv',
}


def validate_dsl_cat_correspondence(cat_model_name):
    """Validate DSL model against .cat reference for a single model.

    Compares the DSL definition's predictions against the portcheck.py
    enumeration checker (which was validated against herd7 .cat specs).

    Returns validation report with agreement metrics.
    """
    if cat_model_name not in CAT_AXIOMS:
        raise ValueError(f"Unknown .cat model: {cat_model_name}")

    arch_name = CAT_TO_ARCH[cat_model_name]
    cat_axioms = CAT_AXIOMS[cat_model_name]

    # Parse the DSL definition
    reg = ModelRegistry()
    dsl_text = DSL_DEFINITIONS[cat_model_name]
    dsl_model = reg.register_dsl(dsl_text)

    # Validate structural correspondence
    structural_match = {
        'relaxed_pairs_match': dsl_model.relaxed_pairs == cat_axioms['relaxed'],
        'preserves_deps_match': dsl_model.preserves_deps == cat_axioms['preserves_deps'],
        'multi_copy_atomic_match': dsl_model.multi_copy_atomic == cat_axioms['multi_copy_atomic'],
    }

    # For each .cat relaxed pair, check DSL has it
    dsl_relaxed = dsl_model.relaxed_pairs
    cat_relaxed = cat_axioms['relaxed']
    missing_in_dsl = cat_relaxed - dsl_relaxed
    extra_in_dsl = dsl_relaxed - cat_relaxed

    # Run all CPU patterns through both checkers
    cpu_patterns = [p for p in sorted(PATTERNS.keys()) if not p.startswith('gpu_')]
    results = []
    agree = 0
    disagree = 0

    for pat_name in cpu_patterns:
        pat_def = PATTERNS[pat_name]
        ops = pat_def['ops']
        n_threads = max(op.thread for op in ops) + 1
        lt = LitmusTest(
            name=pat_name, n_threads=n_threads,
            addresses=pat_def['addresses'], ops=ops,
            forbidden=pat_def['forbidden'],
        )

        # Reference result: portcheck.py enumeration (validated against herd7)
        ref_allowed, ref_checked = verify_test(lt, ARCHITECTURES[arch_name])

        # DSL result
        dsl_result = reg.check_pattern_custom(pat_name, dsl_model)
        dsl_allowed = not dsl_result['safe']

        # herd7 reference if available
        herd7_key = (pat_name, arch_name)
        herd7_allowed = HERD7_EXPECTED.get(herd7_key)

        agrees = ref_allowed == dsl_allowed
        if agrees:
            agree += 1
        else:
            disagree += 1

        entry = {
            'pattern': pat_name,
            'reference_allowed': ref_allowed,
            'dsl_allowed': dsl_allowed,
            'agrees': agrees,
        }
        if herd7_allowed is not None:
            entry['herd7_allowed'] = herd7_allowed
            entry['herd7_agrees_ref'] = herd7_allowed == ref_allowed
            entry['herd7_agrees_dsl'] = herd7_allowed == dsl_allowed

        results.append(entry)

    total = agree + disagree
    if total > 0:
        p_hat, ci_lo, ci_hi = wilson_ci(agree, total)
    else:
        p_hat, ci_lo, ci_hi = 0, 0, 0

    mismatches = [r for r in results if not r['agrees']]

    # herd7 triple agreement
    herd7_triple = [r for r in results if 'herd7_allowed' in r]
    herd7_triple_agree = sum(1 for r in herd7_triple
                             if r.get('herd7_agrees_ref') and r.get('herd7_agrees_dsl'))

    return {
        'model': cat_model_name,
        'cat_reference': cat_axioms['reference'],
        'dsl_model_name': dsl_model.name,
        'structural_correspondence': structural_match,
        'relaxed_pairs': {
            'cat': sorted([f"{a}->{b}" for a, b in cat_relaxed]),
            'dsl': sorted([f"{a}->{b}" for a, b in dsl_relaxed]),
            'missing_in_dsl': sorted([f"{a}->{b}" for a, b in missing_in_dsl]),
            'extra_in_dsl': sorted([f"{a}->{b}" for a, b in extra_in_dsl]),
        },
        'total_patterns': total,
        'agree': agree,
        'disagree': disagree,
        'agreement_rate': round(p_hat * 100, 1),
        'wilson_95ci': [round(ci_lo * 100, 1), round(ci_hi * 100, 1)],
        'mismatches': mismatches,
        'herd7_triple_validation': {
            'total': len(herd7_triple),
            'all_three_agree': herd7_triple_agree,
        },
        'results': results,
    }


def validate_all_models():
    """Validate DSL-.cat correspondence for all three models.

    Returns comprehensive report with per-model and aggregate statistics.
    """
    model_reports = {}
    total_agree = 0
    total_disagree = 0

    for cat_model in ['x86tso', 'aarch64', 'riscv']:
        report = validate_dsl_cat_correspondence(cat_model)
        model_reports[cat_model] = report
        total_agree += report['agree']
        total_disagree += report['disagree']

    total = total_agree + total_disagree
    if total > 0:
        overall_p, overall_lo, overall_hi = wilson_ci(total_agree, total)
    else:
        overall_p, overall_lo, overall_hi = 0, 0, 0

    all_mismatches = []
    for model_name, report in model_reports.items():
        for m in report['mismatches']:
            m['model'] = model_name
            all_mismatches.append(m)

    return {
        'validation_type': 'DSL-to-.cat formal correspondence',
        'models_validated': list(model_reports.keys()),
        'total_checks': total,
        'total_agree': total_agree,
        'total_disagree': total_disagree,
        'overall_agreement_rate': round(overall_p * 100, 1),
        'overall_wilson_95ci': [round(overall_lo * 100, 1), round(overall_hi * 100, 1)],
        'per_model': {k: {
            'model': v['model'],
            'cat_reference': v['cat_reference'],
            'dsl_model': v['dsl_model_name'],
            'structural_match': v['structural_correspondence'],
            'total': v['total_patterns'],
            'agree': v['agree'],
            'disagree': v['disagree'],
            'agreement_rate': v['agreement_rate'],
            'wilson_95ci': v['wilson_95ci'],
            'mismatches': v['mismatches'],
            'herd7_triple': v['herd7_triple_validation'],
        } for k, v in model_reports.items()},
        'all_mismatches': all_mismatches,
    }


def analyze_mismatch_root_causes(mismatches):
    """Analyze root causes of DSL-.cat mismatches.

    Common causes:
    - Fence scope: DSL fence doesn't cover the same pairs as .cat
    - Dependency handling: Different treatment of addr/data/ctrl deps
    - Multi-copy atomicity: DSL and .cat treat differently
    """
    if not mismatches:
        return {'count': 0, 'analysis': 'No mismatches — perfect correspondence'}

    causes = []
    for m in mismatches:
        pat_name = m['pattern']
        ref = m['reference_allowed']
        dsl = m['dsl_allowed']

        cause = {
            'pattern': pat_name,
            'reference_says': 'allowed' if ref else 'forbidden',
            'dsl_says': 'allowed' if dsl else 'forbidden',
        }

        # Analyze based on pattern structure
        pat_def = PATTERNS[pat_name]
        has_fence = any(op.optype == 'fence' for op in pat_def['ops'])
        has_dep = any(op.dep_on is not None for op in pat_def['ops'])
        n_threads = max(op.thread for op in pat_def['ops']) + 1

        if has_fence and dsl and not ref:
            cause['likely_cause'] = 'DSL fence model too weak — .cat fence provides stronger ordering'
        elif has_fence and not dsl and ref:
            cause['likely_cause'] = 'DSL fence model too strong — .cat fence is weaker than DSL assumes'
        elif has_dep and dsl and not ref:
            cause['likely_cause'] = 'DSL does not properly model dependency preservation'
        elif n_threads > 2 and not ref and dsl:
            cause['likely_cause'] = 'Multi-thread ordering difference — possible multi-copy atomicity issue'
        else:
            cause['likely_cause'] = 'Relaxation model difference — investigate axiom mapping'

        causes.append(cause)

    return {
        'count': len(causes),
        'causes': causes,
    }


def run_dsl_cat_validation():
    """Run full DSL-.cat correspondence validation and save results."""
    print("=" * 70)
    print("LITMUS∞ DSL-to-.cat Formal Correspondence Validation")
    print("=" * 70)

    start = time.time()
    report = validate_all_models()
    elapsed = time.time() - start

    print(f"\nModels validated: {', '.join(report['models_validated'])}")
    print(f"Total checks: {report['total_checks']}")
    print(f"Agreement: {report['total_agree']}/{report['total_checks']} "
          f"({report['overall_agreement_rate']}%)")
    print(f"Wilson 95% CI: [{report['overall_wilson_95ci'][0]}%, "
          f"{report['overall_wilson_95ci'][1]}%]")

    print(f"\nPer-model breakdown:")
    for model_name, model_data in report['per_model'].items():
        print(f"  {model_name} ({model_data['cat_reference']}):")
        print(f"    DSL model: {model_data['dsl_model']}")
        sm = model_data['structural_match']
        structural_ok = all(sm.values())
        print(f"    Structural match: {'✓' if structural_ok else '✗'} "
              f"(relaxed={sm['relaxed_pairs_match']}, deps={sm['preserves_deps_match']}, "
              f"mca={sm['multi_copy_atomic_match']})")
        print(f"    Agreement: {model_data['agree']}/{model_data['total']} "
              f"({model_data['agreement_rate']}%, CI {model_data['wilson_95ci']})")
        if model_data['mismatches']:
            print(f"    Mismatches ({len(model_data['mismatches'])}):")
            for m in model_data['mismatches'][:5]:
                print(f"      {m['pattern']}: ref={m['reference_allowed']}, dsl={m['dsl_allowed']}")
        h7 = model_data['herd7_triple']
        print(f"    herd7 triple agreement: {h7['all_three_agree']}/{h7['total']}")

    if report['all_mismatches']:
        analysis = analyze_mismatch_root_causes(report['all_mismatches'])
        print(f"\nMismatch root cause analysis ({analysis['count']} mismatches):")
        for c in analysis.get('causes', [])[:10]:
            print(f"  {c['pattern']}: {c['likely_cause']}")
    else:
        print(f"\n✓ Perfect correspondence — zero mismatches across all models")

    print(f"\nCompleted in {elapsed:.1f}s")

    os.makedirs('paper_results_v5', exist_ok=True)
    with open('paper_results_v5/dsl_cat_correspondence.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved to paper_results_v5/dsl_cat_correspondence.json")

    return report


if __name__ == '__main__':
    run_dsl_cat_validation()
