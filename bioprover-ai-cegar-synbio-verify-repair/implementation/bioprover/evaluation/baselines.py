"""Baseline comparison interfaces for BioProver evaluation.

Provides adapters for external verification tools (PRISM, Storm, dReal, Breach)
and ablation baselines, enabling systematic performance comparison.
"""
from __future__ import annotations
import logging, math, os, shutil, subprocess, tempfile, time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from bioprover.cegar import (CEGARConfig, CEGAREngine, CEGARStatistics,
                             VerificationResult, VerificationStatus)
from bioprover.models import BioModel, Parameter, ParameterSet, Reaction, Species
from bioprover.temporal import (Always, Eventually, Interval, Predicate,
                                STLAnd, STLFormula, STLNot, STLOr, Until)

logger = logging.getLogger(__name__)

@dataclass
class BaselineResult:
    """Outcome of running a single baseline verification tool."""
    status: VerificationStatus
    wall_time: float
    memory_mb: float
    tool_name: str
    raw_output: str = ""
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.status != VerificationStatus.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool_name, "status": self.status.name,
                "wall_time": self.wall_time, "memory_mb": self.memory_mb,
                "error": self.error}

class BaselineTool(ABC):
    """Interface every external baseline tool must implement."""
    @property
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def convert_model(self, model: BioModel) -> str: ...
    @abstractmethod
    def convert_property(self, spec: STLFormula) -> str: ...
    @abstractmethod
    def run(self, model: BioModel, spec: STLFormula,
            timeout: float = 3600.0) -> BaselineResult: ...

    def is_available(self) -> bool:
        return False

    @staticmethod
    def _pred_str(pred: Predicate) -> str:
        return f"{pred.expr.variable or 'const'} {pred.op.value} {pred.threshold}"

    def _stl_str(self, s: STLFormula) -> str:
        if isinstance(s, Predicate): return self._pred_str(s)
        if isinstance(s, STLNot): return f"!({self._stl_str(s.child)})"
        if isinstance(s, STLAnd): return f"({self._stl_str(s.left)}) & ({self._stl_str(s.right)})"
        if isinstance(s, STLOr): return f"({self._stl_str(s.left)}) | ({self._stl_str(s.right)})"
        if isinstance(s, Always): return f"G[{s.interval.lo},{s.interval.hi}]({self._stl_str(s.child)})"
        if isinstance(s, Eventually): return f"F[{s.interval.lo},{s.interval.hi}]({self._stl_str(s.child)})"
        if isinstance(s, Until):
            return f"({self._stl_str(s.left)}) U[{s.interval.lo},{s.interval.hi}] ({self._stl_str(s.right)})"
        return str(s)

    def _err(self, what: str, timeout: float = 0.0, msg: str = "") -> BaselineResult:
        return BaselineResult(VerificationStatus.UNKNOWN, timeout, 0.0, self.name, "", msg or what)


class PRISMBaseline(BaselineTool):
    """Adapter for the PRISM probabilistic model checker."""
    def __init__(self, prism_path: str = "prism") -> None:
        self._path = prism_path

    @property
    def name(self) -> str: return "PRISM"

    def is_available(self) -> bool: return shutil.which(self._path) is not None

    def convert_model(self, model: BioModel) -> str:
        lines: List[str] = ["ctmc", ""]
        for p in model.parameters:
            lines.append(f"const double {p.name} = {p.value};")
        lines.append("")
        for sp in model.species:
            init = int(sp.initial_concentration)
            lines.append(f"module {sp.name}_mod")
            lines.append(f"  {sp.name} : [0..100] init {init};")
            for rxn in model.reactions:
                delta = int(sum(e.coefficient for e in rxn.products if e.species == sp.name)
                            - sum(e.coefficient for e in rxn.reactants if e.species == sp.name))
                if delta == 0: continue
                guard = f"{sp.name} > 0" if delta < 0 else "true"
                rate = rxn.kinetic_law.rate_expression() if hasattr(rxn.kinetic_law, "rate_expression") else rxn.name
                lines.append(f"  [] {guard} -> {rate} : ({sp.name}' = {sp.name} + {delta});")
            lines.append("endmodule\n")
        return "\n".join(lines)

    def convert_property(self, spec: STLFormula) -> str:
        if isinstance(spec, (Always, Eventually)):
            inner = self._csl(spec.child)
            bnd = f"<={spec.interval.hi}" if spec.interval.hi < float("inf") else ""
            op = "G" if isinstance(spec, Always) else "F"
            return f"P>=1 [ {op}{bnd} {inner} ]"
        return f"P=? [ G {self._stl_str(spec)} ]"

    def _csl(self, s: STLFormula) -> str:
        if isinstance(s, Predicate): return self._pred_str(s)
        if isinstance(s, STLAnd): return f"({self._csl(s.left)} & {self._csl(s.right)})"
        if isinstance(s, STLOr): return f"({self._csl(s.left)} | {self._csl(s.right)})"
        if isinstance(s, STLNot): return f"!({self._csl(s.child)})"
        return self._stl_str(s)

    def run(self, model: BioModel, spec: STLFormula, timeout: float = 3600.0) -> BaselineResult:
        with tempfile.TemporaryDirectory(prefix="bp_prism_") as td:
            mp, pp = os.path.join(td, "model.pm"), os.path.join(td, "prop.pctl")
            with open(mp, "w") as f: f.write(self.convert_model(model))
            with open(pp, "w") as f: f.write(self.convert_property(spec))
            t0 = time.monotonic()
            try:
                proc = subprocess.run([self._path, mp, pp], capture_output=True, text=True, timeout=timeout)
                return BaselineResult(self._parse_prism_output(proc.stdout),
                                      time.monotonic() - t0, 0.0, self.name, proc.stdout)
            except subprocess.TimeoutExpired: return self._err("timeout", timeout)
            except FileNotFoundError: return self._err("PRISM binary not found")

    @staticmethod
    def _parse_prism_output(stdout: str) -> VerificationStatus:
        for line in stdout.splitlines():
            if line.strip().startswith("Result:"):
                tok = line.split(":", 1)[1].strip().lower()
                if tok in ("true", "1.0"): return VerificationStatus.VERIFIED
                if tok in ("false", "0.0"): return VerificationStatus.FALSIFIED
        return VerificationStatus.UNKNOWN


class StormBaseline(BaselineTool):
    """Adapter for the Storm probabilistic model checker."""
    def __init__(self, storm_path: str = "storm") -> None:
        self._path = storm_path

    @property
    def name(self) -> str: return "Storm"
    def is_available(self) -> bool: return shutil.which(self._path) is not None
    def convert_model(self, model: BioModel) -> str: return PRISMBaseline().convert_model(model)
    def convert_property(self, spec: STLFormula) -> str: return PRISMBaseline().convert_property(spec)

    def run(self, model: BioModel, spec: STLFormula, timeout: float = 3600.0) -> BaselineResult:
        with tempfile.TemporaryDirectory(prefix="bp_storm_") as td:
            mp = os.path.join(td, "model.pm")
            with open(mp, "w") as f: f.write(self.convert_model(model))
            t0 = time.monotonic()
            try:
                proc = subprocess.run([self._path, "--prism", mp, "--prop", self.convert_property(spec)],
                                      capture_output=True, text=True, timeout=timeout)
                return BaselineResult(self._parse_storm_output(proc.stdout),
                                      time.monotonic() - t0, 0.0, self.name, proc.stdout)
            except subprocess.TimeoutExpired: return self._err("timeout", timeout)
            except FileNotFoundError: return self._err("Storm binary not found")

    @staticmethod
    def _parse_storm_output(stdout: str) -> VerificationStatus:
        for line in stdout.splitlines():
            low = line.strip().lower()
            if "result" in low:
                if "true" in low: return VerificationStatus.VERIFIED
                if "false" in low: return VerificationStatus.FALSIFIED
        return VerificationStatus.UNKNOWN


class DRealBaseline(BaselineTool):
    """Adapter for the dReal SMT solver — flat Euler-discretised ODE encoding."""
    def __init__(self, dreal_path: str = "dreal", dt: float = 0.01, horizon: int = 100) -> None:
        self._path, self._dt, self._N = dreal_path, dt, horizon

    @property
    def name(self) -> str: return "dReal"
    def is_available(self) -> bool: return shutil.which(self._path) is not None
    def convert_model(self, model: BioModel) -> str: return ""
    def convert_property(self, spec: STLFormula) -> str: return ""

    def run(self, model: BioModel, spec: STLFormula, timeout: float = 3600.0) -> BaselineResult:
        smt2 = self._build_smt2(model, spec)
        with tempfile.TemporaryDirectory(prefix="bp_dreal_") as td:
            sp = os.path.join(td, "enc.smt2")
            with open(sp, "w") as f: f.write(smt2)
            t0 = time.monotonic()
            try:
                proc = subprocess.run([self._path, sp], capture_output=True, text=True, timeout=timeout)
                out = proc.stdout.strip().lower()
                st = (VerificationStatus.VERIFIED if "unsat" in out
                      else VerificationStatus.FALSIFIED if "sat" in out
                      else VerificationStatus.UNKNOWN)
                return BaselineResult(st, time.monotonic() - t0, 0.0, self.name, proc.stdout)
            except subprocess.TimeoutExpired: return self._err("timeout", timeout)
            except FileNotFoundError: return self._err("dReal binary not found")

    def _build_smt2(self, model: BioModel, spec: STLFormula) -> str:
        L: List[str] = ["(set-logic QF_NRA)"]
        names = [s.name for s in model.species]
        for k in range(self._N + 1):
            for s in names:
                L.append(f"(declare-fun {s}_{k} () Real)")
        for sp in model.species:
            L.append(f"(assert (= {sp.name}_0 {sp.initial_concentration}))")
        for k in range(self._N):
            for sp in model.species:
                terms: List[str] = []
                for rxn in model.reactions:
                    delta = (sum(e.coefficient for e in rxn.products if e.species == sp.name)
                             - sum(e.coefficient for e in rxn.reactants if e.species == sp.name))
                    if delta == 0: continue
                    rvars = [f"{e.species}_{k}" for e in rxn.reactants]
                    prop = ("(* " + " ".join(rvars) + ")") if len(rvars) > 1 else (rvars[0] if rvars else "1.0")
                    rn = rxn.name + "_rate"
                    rv = model.parameters.get(rn).value if rn in model.parameters else 1.0
                    terms.append(f"(* {rv} (* {delta} {prop}))")
                rhs = terms[0] if len(terms) == 1 else f"(+ {' '.join(terms)})" if terms else "0.0"
                L.append(f"(assert (= {sp.name}_{k+1} (+ {sp.name}_{k} (* {self._dt} {rhs}))))")
        L.append(f"(assert (not {self._enc_stl(spec, names)}))")
        L += ["(check-sat)", "(exit)"]
        return "\n".join(L)

    def _enc_stl(self, s: STLFormula, sp: List[str]) -> str:
        if isinstance(s, Predicate):
            v = s.expr.variable or sp[0]
            ps = [f"({s.op.value} {v}_{k} {s.threshold})" for k in range(self._N + 1)]
            return f"(and {' '.join(ps)})" if len(ps) > 1 else ps[0]
        if isinstance(s, Always):
            lo, hi = max(0, int(s.interval.lo / self._dt)), min(self._N, int(s.interval.hi / self._dt))
            ps = [self._enc_at(s.child, sp, k) for k in range(lo, hi + 1)]
            return f"(and {' '.join(ps)})" if len(ps) > 1 else (ps[0] if ps else "true")
        if isinstance(s, Eventually):
            lo, hi = max(0, int(s.interval.lo / self._dt)), min(self._N, int(s.interval.hi / self._dt))
            ps = [self._enc_at(s.child, sp, k) for k in range(lo, hi + 1)]
            return f"(or {' '.join(ps)})" if len(ps) > 1 else (ps[0] if ps else "false")
        if isinstance(s, STLAnd): return f"(and {self._enc_stl(s.left, sp)} {self._enc_stl(s.right, sp)})"
        if isinstance(s, STLOr): return f"(or {self._enc_stl(s.left, sp)} {self._enc_stl(s.right, sp)})"
        if isinstance(s, STLNot): return f"(not {self._enc_stl(s.child, sp)})"
        return "true"

    def _enc_at(self, s: STLFormula, sp: List[str], k: int) -> str:
        if isinstance(s, Predicate):
            return f"({s.op.value} {s.expr.variable or sp[0]}_{k} {s.threshold})"
        if isinstance(s, STLAnd): return f"(and {self._enc_at(s.left, sp, k)} {self._enc_at(s.right, sp, k)})"
        if isinstance(s, STLOr): return f"(or {self._enc_at(s.left, sp, k)} {self._enc_at(s.right, sp, k)})"
        if isinstance(s, STLNot): return f"(not {self._enc_at(s.child, sp, k)})"
        return "true"


class BreachBaseline(BaselineTool):
    """Adapter for the Breach STL falsification toolbox (MATLAB)."""
    def __init__(self, matlab_path: str = "matlab") -> None:
        self._path = matlab_path

    @property
    def name(self) -> str: return "Breach"
    def is_available(self) -> bool: return shutil.which(self._path) is not None

    def convert_model(self, model: BioModel) -> str:
        sp_names = [s.name for s in model.species]
        lines = [f"function dydt = ode_model(t, y)", f"  dydt = zeros({len(sp_names)}, 1);"]
        for i, sp in enumerate(model.species, 1):
            terms: List[str] = []
            for rxn in model.reactions:
                delta = (sum(e.coefficient for e in rxn.products if e.species == sp.name)
                         - sum(e.coefficient for e in rxn.reactants if e.species == sp.name))
                if delta == 0: continue
                ridx = [f"y({sp_names.index(e.species)+1})" for e in rxn.reactants
                        for _ in range(int(e.coefficient))]
                terms.append(f"{delta} * {' * '.join(ridx) if ridx else '1'}")
            lines.append(f"  dydt({i}) = {' + '.join(terms) if terms else '0'};")
        lines.append("end")
        return "\n".join(lines)

    def convert_property(self, spec: STLFormula) -> str:
        if isinstance(spec, Predicate):
            return f"{spec.expr.variable or 'x'}[t] {spec.op.value} {spec.threshold}"
        if isinstance(spec, Always):
            return f"alw_[{spec.interval.lo},{spec.interval.hi}]({self.convert_property(spec.child)})"
        if isinstance(spec, Eventually):
            return f"ev_[{spec.interval.lo},{spec.interval.hi}]({self.convert_property(spec.child)})"
        if isinstance(spec, Until):
            return (f"({self.convert_property(spec.left)}) until_[{spec.interval.lo},{spec.interval.hi}]"
                    f" ({self.convert_property(spec.right)})")
        if isinstance(spec, STLAnd):
            return f"({self.convert_property(spec.left)}) and ({self.convert_property(spec.right)})"
        if isinstance(spec, STLOr):
            return f"({self.convert_property(spec.left)}) or ({self.convert_property(spec.right)})"
        if isinstance(spec, STLNot):
            return f"not ({self.convert_property(spec.child)})"
        return str(spec)

    def run(self, model: BioModel, spec: STLFormula, timeout: float = 3600.0) -> BaselineResult:
        with tempfile.TemporaryDirectory(prefix="bp_breach_") as td:
            with open(os.path.join(td, "ode_model.m"), "w") as f:
                f.write(self.convert_model(model))
            stl = self.convert_property(spec)
            script = "\n".join([
                f"addpath('{td}');", "InitBreach;",
                "mdl = BreachSimulinkSystem('ode_model');",
                f"phi = STL_Formula('prop', '{stl}');",
                "falsif = FalsificationProblem(mdl, phi);",
                f"falsif.max_time = {timeout};", "falsif.solve();",
                "if falsif.obj_best < 0, disp('FALSIFIED');",
                "else, disp('NOT_FALSIFIED'); end", "exit;"])
            sp = os.path.join(td, "run_breach.m")
            with open(sp, "w") as f: f.write(script)
            t0 = time.monotonic()
            try:
                proc = subprocess.run([self._path, "-batch", f"run('{sp}')"],
                                      capture_output=True, text=True, timeout=timeout)
                st = (VerificationStatus.FALSIFIED if "FALSIFIED" in proc.stdout
                      else VerificationStatus.BOUNDED_GUARANTEE if "NOT_FALSIFIED" in proc.stdout
                      else VerificationStatus.UNKNOWN)
                return BaselineResult(st, time.monotonic() - t0, 0.0, self.name, proc.stdout)
            except subprocess.TimeoutExpired: return self._err("timeout", timeout)
            except FileNotFoundError: return self._err("MATLAB binary not found")


class BioProverNoAIBaseline(BaselineTool):
    """BioProver with ML-guided refinement disabled (ablation baseline)."""
    def __init__(self, base_config: Optional[CEGARConfig] = None) -> None:
        cfg = base_config or CEGARConfig()
        self._config = CEGARConfig(
            max_iterations=cfg.max_iterations, timeout=cfg.timeout,
            initial_grid_resolution=cfg.initial_grid_resolution,
            step_size=cfg.step_size, delta=cfg.delta,
            stagnation_window=cfg.stagnation_window,
            coverage_target=cfg.coverage_target, strategy_name=cfg.strategy_name,
            enable_ai_heuristic=False,
            enable_bounded_guarantee=cfg.enable_bounded_guarantee,
            max_workers=cfg.max_workers)

    @property
    def name(self) -> str: return "BioProver-NoAI"
    def is_available(self) -> bool: return True
    def convert_model(self, model: BioModel) -> str: return ""
    def convert_property(self, spec: STLFormula) -> str: return ""

    def run(self, model: BioModel, spec: STLFormula, timeout: float = 3600.0) -> BaselineResult:
        cfg = CEGARConfig(
            max_iterations=self._config.max_iterations,
            timeout=min(self._config.timeout, timeout),
            initial_grid_resolution=self._config.initial_grid_resolution,
            step_size=self._config.step_size, delta=self._config.delta,
            stagnation_window=self._config.stagnation_window,
            coverage_target=self._config.coverage_target,
            strategy_name=self._config.strategy_name,
            enable_ai_heuristic=False,
            enable_bounded_guarantee=self._config.enable_bounded_guarantee,
            max_workers=self._config.max_workers)
        engine = CEGAREngine(config=cfg)
        t0 = time.monotonic()
        try:
            result: VerificationResult = engine.verify()
            return BaselineResult(result.status, time.monotonic() - t0, 0.0,
                                  self.name, str(result.to_dict()))
        except Exception as exc:
            logger.exception("BioProver-NoAI failed")
            return BaselineResult(VerificationStatus.UNKNOWN, time.monotonic() - t0,
                                  0.0, self.name, "", str(exc))


class BaselineComparison:
    """Run multiple baseline tools and aggregate results for evaluation."""
    def __init__(self, tools: List[BaselineTool],
                 bioprover_config: Optional[CEGARConfig] = None) -> None:
        self._tools = list(tools)
        self._config = bioprover_config or CEGARConfig()

    def compare(self, model: BioModel, spec: STLFormula,
                timeout: float = 3600.0) -> Dict[str, BaselineResult]:
        results: Dict[str, BaselineResult] = {}
        for tool in self._tools:
            logger.info("Running baseline: %s", tool.name)
            try:
                results[tool.name] = tool.run(model, spec, timeout=timeout)
            except Exception as exc:
                logger.error("Tool %s crashed: %s", tool.name, exc)
                results[tool.name] = BaselineResult(
                    VerificationStatus.UNKNOWN, 0.0, 0.0, tool.name, "", str(exc))
        return results

    def compare_batch(self, benchmarks: List[Tuple[BioModel, STLFormula]],
                      timeout: float = 3600.0) -> List[Dict[str, BaselineResult]]:
        """Evaluate all tools across a list of benchmarks."""
        return [self.compare(m, s, timeout) for m, s in benchmarks]

    def generate_table(self, results: List[Dict[str, BaselineResult]]) -> str:
        """Produce a human-readable ASCII table summarising results."""
        tools = sorted({t for r in results for t in r})
        cw = max(18, max((len(n) for n in tools), default=0) + 2)
        hdr = f"{'Benchmark':<12}" + "".join(f"{'| ' + n:<{cw}}" for n in tools)
        sep = "-" * len(hdr)
        rows = [sep, hdr, sep]
        for i, br in enumerate(results):
            cells = [f"B{i:<11d}"]
            for tn in tools:
                r = br.get(tn)
                if r is None: cells.append(f"{'| -':<{cw}}")
                elif r.error: cells.append(f"{'| ERR':<{cw}}")
                else: cells.append(f"{'| ' + r.status.name[:4] + f' {r.wall_time:6.1f}s':<{cw}}")
            rows.append("".join(cells))
        rows.append(sep)
        return "\n".join(rows)

    def generate_latex_table(self, results: List[Dict[str, BaselineResult]]) -> str:
        """Produce a LaTeX tabular environment summarising results."""
        tools = sorted({t for r in results for t in r})
        lines = [r"\begin{table}[t]", r"\centering",
                 r"\caption{Baseline comparison results}", r"\label{tab:baselines}",
                 r"\begin{tabular}{l" + "c" * len(tools) + "}",
                 r"\toprule", "Benchmark & " + " & ".join(tools) + r" \\", r"\midrule"]
        for i, br in enumerate(results):
            cells = [f"B{i}"]
            for tn in tools:
                r = br.get(tn)
                if r is None: cells.append("--")
                elif r.error: cells.append(r"\textsc{err}")
                else: cells.append(f"{r.status.name} ({r.wall_time:.1f}s)")
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    @staticmethod
    def statistical_significance(results_a: List[BaselineResult],
                                 results_b: List[BaselineResult]) -> Dict[str, Any]:
        """Paired t-test and Wilcoxon signed-rank test on wall-clock times.

        Returns p-values and Cohen's d effect size.  scipy.stats is imported
        lazily so the rest of the module works without SciPy installed.
        """
        ta, tb = [r.wall_time for r in results_a], [r.wall_time for r in results_b]
        n = min(len(ta), len(tb))
        if n < 2:
            return {"error": "need at least 2 paired observations"}
        ta, tb = ta[:n], tb[:n]
        try:
            from scipy import stats as sp_stats
        except ImportError:
            return {"error": "scipy not installed"}
        t_stat, t_pval = sp_stats.ttest_rel(ta, tb)
        try:
            w_stat, w_pval = sp_stats.wilcoxon(ta, tb)
        except ValueError:
            w_stat, w_pval = 0.0, 1.0
        diffs = [a - b for a, b in zip(ta, tb)]
        mean_d = sum(diffs) / n
        std_d = math.sqrt(sum((d - mean_d) ** 2 for d in diffs) / max(n - 1, 1))
        return {"n": n, "mean_a": sum(ta) / n, "mean_b": sum(tb) / n,
                "t_statistic": t_stat, "t_pvalue": t_pval,
                "wilcoxon_statistic": w_stat, "wilcoxon_pvalue": w_pval,
                "cohens_d": mean_d / std_d if std_d > 0 else 0.0}
