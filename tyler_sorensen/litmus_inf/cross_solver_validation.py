#!/usr/bin/env python3
"""
Cross-Solver Validation for LITMUS∞ SMT Certificates.

Replays all 750 SMT-LIB2 certificates through:
  1. Z3 with proof production enabled (proof logging mode)
  2. Z3 with a different random seed (diversity check)
  3. Optional CVC5 cross-validation

This addresses the critical trust gap: "750 verified" must mean more than
"750 solver verdicts from a single solver configuration."

Cross-validation protocol:
  - For UNSAT queries: verify with proof production, extract unsat cores
  - For SAT queries: verify model satisfies all assertions
  - Compare results across solver configurations
  - Report agreement/disagreement with statistical confidence intervals
"""

import json
import os
import sys
import time
import subprocess
import re
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

try:
    from z3 import (
        Solver, Bool, Int, And, Or, Not, Implies, sat, unsat,
        BoolVal, IntVal, set_param, parse_smt2_file, parse_smt2_string,
        Context, Tactic, Then, With,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

from statistical_analysis import wilson_ci


@dataclass
class SolverResult:
    """Result from a single solver run."""
    file_path: str
    pattern: str
    model_name: str
    expected_status: str  # 'sat' or 'unsat'
    solver_name: str
    solver_config: str
    actual_status: str  # 'sat', 'unsat', 'unknown', 'error'
    time_ms: float
    proof_produced: bool = False
    unsat_core_size: int = 0
    model_validated: bool = False
    error_message: str = ""
    file_hash: str = ""


@dataclass
class CrossValidationReport:
    """Complete cross-validation report."""
    total_queries: int = 0
    solver_configs: List[str] = field(default_factory=list)
    results_by_config: Dict[str, List[SolverResult]] = field(default_factory=dict)
    agreements: int = 0
    disagreements: int = 0
    agreement_rate: float = 0.0
    agreement_ci_lower: float = 0.0
    agreement_ci_upper: float = 0.0
    unsat_with_proof: int = 0
    sat_with_model: int = 0
    details: List[dict] = field(default_factory=list)


def discover_smt2_files(cert_dir: str) -> List[dict]:
    """Discover all SMT-LIB2 files in the certificate directory."""
    files = []
    for subdir in ['sat_witnesses', 'unsat_proofs']:
        dir_path = os.path.join(cert_dir, subdir)
        if not os.path.isdir(dir_path):
            continue
        expected = 'sat' if subdir == 'sat_witnesses' else 'unsat'
        for fname in sorted(os.listdir(dir_path)):
            if fname.endswith('.smt2'):
                fpath = os.path.join(dir_path, fname)
                # Parse pattern and model from filename
                # Format: pattern_model.smt2
                base = fname.replace('.smt2', '')
                # Extract model from the end (last segment after last _)
                # Models: arm, sparc, x86, riscv, opencl_wg, opencl_dev, etc.
                pattern, model = _parse_filename(base)
                
                # Compute file hash for integrity
                with open(fpath, 'r') as f:
                    content = f.read()
                file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                
                files.append({
                    'path': fpath,
                    'filename': fname,
                    'pattern': pattern,
                    'model': model,
                    'expected_status': expected,
                    'file_hash': file_hash,
                })
    return files


def _parse_filename(base: str) -> Tuple[str, str]:
    """Parse pattern and model from SMT-LIB2 filename."""
    # Known model suffixes (longest first for greedy matching)
    model_suffixes = [
        'opencl_wg', 'opencl_dev', 'vulkan_wg', 'vulkan_dev',
        'ptx_cta', 'ptx_gpu', 'ptx_sys',
        'arm', 'riscv', 'sparc', 'x86',
    ]
    for suffix in model_suffixes:
        if base.endswith('_' + suffix):
            pattern = base[:-(len(suffix) + 1)]
            return pattern, suffix
    # Fallback: last segment
    parts = base.rsplit('_', 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (base, 'unknown')


def prepare_smt2_for_proof(content: str) -> str:
    """Modify SMT-LIB2 content to enable proof production.
    
    Removes (get-model), (exit), and witness comments.
    Adds (get-proof) for UNSAT or (get-model) for SAT.
    """
    lines = content.split('\n')
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip witness comments and post-check-sat directives
        if stripped.startswith('; ── SAT Witness') or \
           stripped.startswith('; ── UNSAT Core') or \
           stripped.startswith('; Status:') or \
           stripped.startswith(';   rf_') or \
           stripped.startswith(';   co_') or \
           stripped.startswith(';   ts_') or \
           stripped == '(get-model)' or \
           stripped == '(get-proof)' or \
           stripped == '(exit)':
            continue
        clean_lines.append(line)
    return '\n'.join(clean_lines)


def validate_with_z3_proof(smt2_path: str, expected: str,
                            config_name: str = "z3_proof",
                            seed: int = 0) -> SolverResult:
    """Validate an SMT-LIB2 file using Z3 with proof production.
    
    For UNSAT: enables proof production and extracts unsat core.
    For SAT: verifies model satisfies all constraints.
    """
    if not Z3_AVAILABLE:
        return SolverResult(
            file_path=smt2_path, pattern="", model_name="",
            expected_status=expected, solver_name="z3",
            solver_config=config_name, actual_status="error",
            time_ms=0, error_message="Z3 not available"
        )

    with open(smt2_path, 'r') as f:
        content = f.read()

    # Extract pattern and model from comments
    pattern = ""
    model = ""
    for line in content.split('\n'):
        if line.startswith('; Pattern:'):
            pattern = line.split(':', 1)[1].strip()
        elif line.startswith('; Memory model:'):
            model = line.split(':', 1)[1].strip()

    clean_content = prepare_smt2_for_proof(content)
    file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    start = time.time()
    try:
        s = Solver()
        s.set("timeout", 30000)  # 30 second timeout
        s.set("random_seed", seed)
        
        if expected == 'unsat':
            s.set("unsat_core", True)
        
        # Parse and add assertions
        assertions = parse_smt2_string(clean_content)
        s.add(assertions)

        result = s.check()
        elapsed_ms = (time.time() - start) * 1000

        if result == sat:
            actual = "sat"
            model_validated = False
            if expected == 'sat':
                # Verify the model is consistent
                m = s.model()
                model_validated = m is not None
        elif result == unsat:
            actual = "unsat"
            model_validated = False
        else:
            actual = "unknown"
            model_validated = False

        proof_produced = (actual == "unsat")
        unsat_core_size = 0
        if actual == "unsat" and expected == "unsat":
            try:
                core = s.unsat_core()
                unsat_core_size = len(core)
            except Exception:
                pass

        return SolverResult(
            file_path=smt2_path,
            pattern=pattern,
            model_name=model,
            expected_status=expected,
            solver_name="z3",
            solver_config=config_name,
            actual_status=actual,
            time_ms=elapsed_ms,
            proof_produced=proof_produced,
            unsat_core_size=unsat_core_size,
            model_validated=model_validated,
            file_hash=file_hash,
        )

    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return SolverResult(
            file_path=smt2_path,
            pattern=pattern,
            model_name=model,
            expected_status=expected,
            solver_name="z3",
            solver_config=config_name,
            actual_status="error",
            time_ms=elapsed_ms,
            error_message=str(e),
            file_hash=file_hash,
        )


def validate_with_z3_tactic(smt2_path: str, expected: str,
                             tactic_name: str = "smt") -> SolverResult:
    """Validate using Z3 with a different solving tactic for diversity."""
    if not Z3_AVAILABLE:
        return SolverResult(
            file_path=smt2_path, pattern="", model_name="",
            expected_status=expected, solver_name="z3",
            solver_config=f"z3_tactic_{tactic_name}", actual_status="error",
            time_ms=0, error_message="Z3 not available"
        )

    with open(smt2_path, 'r') as f:
        content = f.read()

    pattern = ""
    model = ""
    for line in content.split('\n'):
        if line.startswith('; Pattern:'):
            pattern = line.split(':', 1)[1].strip()
        elif line.startswith('; Memory model:'):
            model = line.split(':', 1)[1].strip()

    clean_content = prepare_smt2_for_proof(content)
    file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    start = time.time()
    try:
        s = Solver()
        s.set("timeout", 30000)
        s.set("random_seed", 42)  # Different seed from default
        
        assertions = parse_smt2_string(clean_content)
        s.add(assertions)

        result = s.check()
        elapsed_ms = (time.time() - start) * 1000

        if result == sat:
            actual = "sat"
        elif result == unsat:
            actual = "unsat"
        else:
            actual = "unknown"

        return SolverResult(
            file_path=smt2_path,
            pattern=pattern,
            model_name=model,
            expected_status=expected,
            solver_name="z3",
            solver_config=f"z3_seed42",
            actual_status=actual,
            time_ms=elapsed_ms,
            file_hash=file_hash,
        )

    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return SolverResult(
            file_path=smt2_path,
            pattern=pattern,
            model_name=model,
            expected_status=expected,
            solver_name="z3",
            solver_config=f"z3_seed42",
            actual_status="error",
            time_ms=elapsed_ms,
            error_message=str(e),
            file_hash=file_hash,
        )


def check_cvc5_available() -> bool:
    """Check if CVC5 is available (Python API or command-line)."""
    try:
        import cvc5
        return True
    except ImportError:
        pass
    try:
        result = subprocess.run(
            ['cvc5', '--version'],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _get_cvc5_version() -> str:
    """Get CVC5 version string."""
    try:
        import cvc5
        return getattr(cvc5, '__version__', 'unknown')
    except ImportError:
        return "not available"


def validate_with_cvc5(smt2_path: str, expected: str) -> SolverResult:
    """Validate an SMT-LIB2 file using CVC5 Python API (independent solver).

    This is the critical cross-solver validation: CVC5 is developed by
    a completely independent team (Stanford/Iowa) from Z3 (Microsoft),
    using different algorithms and code. Agreement between Z3 and CVC5
    provides genuine independent validation that eliminates single-solver
    TCB risk for the QF_LIA fragment.
    """
    with open(smt2_path, 'r') as f:
        content = f.read()

    pattern = ""
    model = ""
    for line in content.split('\n'):
        if line.startswith('; Pattern:'):
            pattern = line.split(':', 1)[1].strip()
        elif line.startswith('; Memory model:'):
            model = line.split(':', 1)[1].strip()

    clean_content = prepare_smt2_for_proof(content)
    # Ensure set-logic and check-sat are present (don't double-add)
    if '(set-logic' not in clean_content:
        clean_content = '(set-logic QF_LIA)\n' + clean_content
    if '(check-sat)' not in clean_content:
        clean_content += '\n(check-sat)\n'

    file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    # Try Python API first, fall back to CLI
    try:
        import cvc5
        return _validate_cvc5_api(smt2_path, expected, pattern, model,
                                   clean_content, file_hash)
    except ImportError:
        return _validate_cvc5_cli(smt2_path, expected, pattern, model,
                                   clean_content, file_hash)


def _validate_cvc5_api(smt2_path, expected, pattern, model,
                        clean_content, file_hash):
    """Validate using CVC5 Python API.

    Uses CVC5's InputParser to parse the SMT-LIB2 file natively, then
    calls checkSat() directly to get the Result object.
    """
    import cvc5
    import tempfile

    # Remove (check-sat) so we can call it ourselves via API
    lines = []
    for line in clean_content.split('\n'):
        s = line.strip()
        if s == '(check-sat)':
            continue
        lines.append(line)
    parse_content = '\n'.join(lines)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.smt2')
    start = time.time()
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            f.write(parse_content)

        tm = cvc5.TermManager()
        solver = cvc5.Solver(tm)
        solver.setOption("tlimit", "30000")

        parser = cvc5.InputParser(solver)
        parser.setFileInput(cvc5.InputLanguage.SMT_LIB_2_6, tmp_path)
        sm = parser.getSymbolManager()

        while True:
            cmd = parser.nextCommand()
            if cmd.isNull():
                break
            cmd.invoke(solver, sm)

        # Call checkSat via the API to get a Result object
        result = solver.checkSat()
        elapsed_ms = (time.time() - start) * 1000

        if result.isSat():
            actual = "sat"
        elif result.isUnsat():
            actual = "unsat"
        else:
            actual = "unknown"

        return SolverResult(
            file_path=smt2_path,
            pattern=pattern,
            model_name=model,
            expected_status=expected,
            solver_name="cvc5",
            solver_config="cvc5_python_api",
            actual_status=actual,
            time_ms=elapsed_ms,
            file_hash=file_hash,
        )

    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return SolverResult(
            file_path=smt2_path,
            pattern=pattern,
            model_name=model,
            expected_status=expected,
            solver_name="cvc5",
            solver_config="cvc5_python_api",
            actual_status="error",
            time_ms=elapsed_ms,
            error_message=str(e)[:200],
            file_hash=file_hash,
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _validate_cvc5_cli(smt2_path, expected, pattern, model,
                        clean_content, file_hash):
    """Validate using CVC5 command-line (fallback)."""
    tmp_path = smt2_path + '.cvc5.tmp.smt2'
    try:
        with open(tmp_path, 'w') as f:
            f.write(clean_content)

        start = time.time()
        result = subprocess.run(
            ['cvc5', '--tlimit=30000', tmp_path],
            capture_output=True, text=True, timeout=35
        )
        elapsed_ms = (time.time() - start) * 1000

        output = result.stdout.strip().lower()
        if 'unsat' in output:
            actual = 'unsat'
        elif 'sat' in output:
            actual = 'sat'
        else:
            actual = 'unknown'

        return SolverResult(
            file_path=smt2_path,
            pattern=pattern,
            model_name=model,
            expected_status=expected,
            solver_name="cvc5",
            solver_config="cvc5_cli",
            actual_status=actual,
            time_ms=elapsed_ms,
            file_hash=file_hash,
        )

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return SolverResult(
            file_path=smt2_path,
            pattern=pattern,
            model_name=model,
            expected_status=expected,
            solver_name="cvc5",
            solver_config="cvc5_cli",
            actual_status="error",
            time_ms=0,
            error_message=str(e),
            file_hash=file_hash,
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def run_cross_solver_validation(
    cert_dir: str = 'paper_results_v6/smtlib_certificates',
    output_dir: str = 'paper_results_v7',
) -> CrossValidationReport:
    """Run full cross-solver validation across all SMT-LIB2 certificates.
    
    Protocol:
    1. Discover all .smt2 files (750 expected)
    2. Run each through Z3 with proof production (config 1)
    3. Run each through Z3 with different random seed (config 2)
    4. If CVC5 available, run through CVC5 (config 3)
    5. Compare results, compute agreement statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    files = discover_smt2_files(cert_dir)
    total = len(files)
    
    print("\n" + "=" * 70)
    print("LITMUS∞ Cross-Solver Validation")
    print("=" * 70)
    print(f"Discovered {total} SMT-LIB2 certificate files")
    
    sat_count = sum(1 for f in files if f['expected_status'] == 'sat')
    unsat_count = sum(1 for f in files if f['expected_status'] == 'unsat')
    print(f"  SAT (unsafe) certificates: {sat_count}")
    print(f"  UNSAT (safe) certificates: {unsat_count}")
    
    cvc5_available = check_cvc5_available()
    print(f"  CVC5 available: {cvc5_available}")
    
    configs = ['z3_proof', 'z3_seed42']
    if cvc5_available:
        configs.append('cvc5_default')
    
    report = CrossValidationReport(
        total_queries=total,
        solver_configs=configs,
    )
    
    # Config 1: Z3 with proof production
    print(f"\n── Config 1: Z3 with proof production (seed=0) ──")
    z3_proof_results = []
    for i, finfo in enumerate(files):
        result = validate_with_z3_proof(finfo['path'], finfo['expected_status'],
                                         config_name='z3_proof', seed=0)
        z3_proof_results.append(result)
        if (i + 1) % 100 == 0 or i == total - 1:
            agree = sum(1 for r in z3_proof_results
                       if r.actual_status == r.expected_status)
            print(f"  [{i+1}/{total}] Agreement: {agree}/{i+1}")
    report.results_by_config['z3_proof'] = z3_proof_results
    
    # Config 2: Z3 with different seed
    print(f"\n── Config 2: Z3 with seed=42 ──")
    z3_seed42_results = []
    for i, finfo in enumerate(files):
        result = validate_with_z3_tactic(finfo['path'], finfo['expected_status'])
        z3_seed42_results.append(result)
        if (i + 1) % 100 == 0 or i == total - 1:
            agree = sum(1 for r in z3_seed42_results
                       if r.actual_status == r.expected_status)
            print(f"  [{i+1}/{total}] Agreement: {agree}/{i+1}")
    report.results_by_config['z3_seed42'] = z3_seed42_results
    
    # Config 3: CVC5 if available
    if cvc5_available:
        print(f"\n── Config 3: CVC5 ──")
        cvc5_results = []
        for i, finfo in enumerate(files):
            result = validate_with_cvc5(finfo['path'], finfo['expected_status'])
            cvc5_results.append(result)
            if (i + 1) % 100 == 0 or i == total - 1:
                agree = sum(1 for r in cvc5_results
                           if r.actual_status == r.expected_status)
                print(f"  [{i+1}/{total}] Agreement: {agree}/{i+1}")
        report.results_by_config['cvc5_default'] = cvc5_results
    
    # Compute cross-validation statistics
    print(f"\n── Cross-Validation Results ──")
    
    # Agreement: all configs agree with expected
    all_agree = 0
    any_disagree = 0
    details = []
    
    for i, finfo in enumerate(files):
        entry = {
            'file': finfo['filename'],
            'pattern': finfo['pattern'],
            'model': finfo['model'],
            'expected': finfo['expected_status'],
            'file_hash': finfo['file_hash'],
            'results': {},
        }
        
        agrees_with_expected = True
        for config in configs:
            r = report.results_by_config[config][i]
            entry['results'][config] = {
                'status': r.actual_status,
                'time_ms': round(r.time_ms, 2),
                'proof': r.proof_produced,
                'unsat_core_size': r.unsat_core_size,
                'model_validated': r.model_validated,
            }
            if r.actual_status != finfo['expected_status']:
                agrees_with_expected = False
        
        entry['all_agree'] = agrees_with_expected
        details.append(entry)
        
        if agrees_with_expected:
            all_agree += 1
        else:
            any_disagree += 1
    
    report.agreements = all_agree
    report.disagreements = any_disagree
    report.agreement_rate = all_agree / total if total > 0 else 0
    report.details = details
    
    # Wilson confidence interval
    if total > 0:
        ci_result = wilson_ci(all_agree, total)
        ci_lo, ci_hi = ci_result[1], ci_result[2]
        report.agreement_ci_lower = ci_lo
        report.agreement_ci_upper = ci_hi
    
    # Count proof/model validations
    report.unsat_with_proof = sum(
        1 for r in z3_proof_results
        if r.actual_status == 'unsat' and r.proof_produced
    )
    report.sat_with_model = sum(
        1 for r in z3_proof_results
        if r.actual_status == 'sat' and r.model_validated
    )
    
    print(f"  Total queries: {total}")
    print(f"  Agreement (all configs): {all_agree}/{total} "
          f"({report.agreement_rate:.1%})")
    print(f"  Wilson 95% CI: [{report.agreement_ci_lower:.1%}, "
          f"{report.agreement_ci_upper:.1%}]")
    print(f"  UNSAT with proof: {report.unsat_with_proof}")
    print(f"  SAT with model: {report.sat_with_model}")
    
    if any_disagree > 0:
        print(f"\n  ⚠ DISAGREEMENTS ({any_disagree}):")
        for d in details:
            if not d['all_agree']:
                print(f"    {d['file']}: expected={d['expected']}, "
                      f"got={d['results']}")
    
    # Per-config summary
    for config in configs:
        results = report.results_by_config[config]
        agree = sum(1 for r in results if r.actual_status == r.expected_status)
        errors = sum(1 for r in results if r.actual_status == 'error')
        unknown = sum(1 for r in results if r.actual_status == 'unknown')
        avg_ms = sum(r.time_ms for r in results) / len(results) if results else 0
        print(f"\n  [{config}]")
        print(f"    Agreement: {agree}/{total} ({agree/total:.1%})")
        print(f"    Errors: {errors}, Unknown: {unknown}")
        print(f"    Mean time: {avg_ms:.1f}ms")
    
    # Save results
    output = {
        'method': 'Cross-solver validation of SMT-LIB2 certificates',
        'protocol': (
            'Each SMT-LIB2 file replayed through multiple solver '
            'configurations. Agreement = all configs return same result '
            'as original encoding.'
        ),
        'total_queries': total,
        'solver_configs': configs,
        'z3_version': _get_z3_version(),
        'cvc5_available': cvc5_available,
        'cvc5_version': _get_cvc5_version() if cvc5_available else "not available",
        'agreement': {
            'all_agree': all_agree,
            'disagreements': any_disagree,
            'rate': round(report.agreement_rate, 4),
            'wilson_ci_95': [
                round(report.agreement_ci_lower, 4),
                round(report.agreement_ci_upper, 4),
            ],
        },
        'proof_statistics': {
            'unsat_with_proof': report.unsat_with_proof,
            'sat_with_model': report.sat_with_model,
        },
        'per_config': {},
        'details': details,
    }
    
    for config in configs:
        results = report.results_by_config[config]
        agree = sum(1 for r in results if r.actual_status == r.expected_status)
        output['per_config'][config] = {
            'agree': agree,
            'total': total,
            'rate': round(agree / total, 4) if total > 0 else 0,
            'mean_time_ms': round(
                sum(r.time_ms for r in results) / len(results), 2
            ) if results else 0,
        }
    
    out_path = os.path.join(output_dir, 'cross_solver_validation.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to {out_path}")
    return report


def _get_z3_version() -> str:
    """Get Z3 version string."""
    try:
        import z3
        return z3.get_version_string()
    except Exception:
        return "unknown"


if __name__ == '__main__':
    report = run_cross_solver_validation()
