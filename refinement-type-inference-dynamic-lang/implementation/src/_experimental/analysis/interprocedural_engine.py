"""
Interprocedural analysis engine with refinement type function summaries.

Computes function summaries as refinement types:
    f: {x:int|P} -> {r:int|Q}

Supports:
- Context-sensitive analysis using calling context
- Recursive functions via fixpoint iteration
- Call graph construction
- Summary-based compositional analysis
"""

from __future__ import annotations

import ast
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import z3

from ..refinement_lattice import (
    ANY_TYPE,
    BaseTypeKind,
    BaseTypeR,
    BOOL_TYPE,
    DepFuncType,
    FLOAT_TYPE,
    INT_TYPE,
    NEVER_TYPE,
    NONE_TYPE,
    Pred,
    PredOp,
    PredicateAbstractionDomain,
    RefEnvironment,
    RefType,
    RefinementLattice,
    STR_TYPE,
    Z3Encoder,
)

from ..cegar.cegar_loop import (
    AbstractInterpreter,
    Alarm,
    CEGARConfig,
    harvest_guards,
)


# ---------------------------------------------------------------------------
# Call graph
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CallSite:
    """A call site in the program."""
    caller: str      # function name
    callee: str      # function name
    line: int
    args: Tuple[str, ...]  # argument variable names


@dataclass
class CallGraph:
    """Call graph for interprocedural analysis."""
    edges: List[CallSite] = field(default_factory=list)
    _callers: Dict[str, List[CallSite]] = field(default_factory=lambda: defaultdict(list))
    _callees: Dict[str, List[CallSite]] = field(default_factory=lambda: defaultdict(list))

    def add_edge(self, site: CallSite) -> None:
        self.edges.append(site)
        self._callers[site.callee].append(site)
        self._callees[site.caller].append(site)

    def callers_of(self, func: str) -> List[CallSite]:
        return self._callers.get(func, [])

    def callees_of(self, func: str) -> List[CallSite]:
        return self._callees.get(func, [])

    def all_functions(self) -> Set[str]:
        funcs: Set[str] = set()
        for e in self.edges:
            funcs.add(e.caller)
            funcs.add(e.callee)
        return funcs

    def topological_order(self) -> List[str]:
        """Return functions in reverse topological order (callees first).

        Handles cycles by breaking them arbitrarily.
        """
        funcs = self.all_functions()
        visited: Set[str] = set()
        order: List[str] = []
        in_stack: Set[str] = set()

        def dfs(f: str):
            if f in visited:
                return
            if f in in_stack:
                return  # cycle: break it
            in_stack.add(f)
            for site in self.callees_of(f):
                dfs(site.callee)
            in_stack.discard(f)
            visited.add(f)
            order.append(f)

        for f in funcs:
            dfs(f)

        return order  # callees come before callers

    def is_recursive(self, func: str) -> bool:
        """Check if a function is (mutually) recursive."""
        visited: Set[str] = set()

        def reaches(start: str, target: str) -> bool:
            if start in visited:
                return False
            visited.add(start)
            for site in self.callees_of(start):
                if site.callee == target:
                    return True
                if reaches(site.callee, target):
                    return True
            return False

        return reaches(func, func)

    def sccs(self) -> List[List[str]]:
        """Compute strongly connected components (Tarjan's algorithm)."""
        index_counter = [0]
        stack: List[str] = []
        lowlink: Dict[str, int] = {}
        index: Dict[str, int] = {}
        on_stack: Set[str] = set()
        result: List[List[str]] = []
        funcs = self.all_functions()

        def strongconnect(v: str):
            index[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)

            for site in self.callees_of(v):
                w = site.callee
                if w not in index:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])

            if lowlink[v] == index[v]:
                scc: List[str] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.append(w)
                    if w == v:
                        break
                result.append(scc)

        for v in funcs:
            if v not in index:
                strongconnect(v)

        return result


# ---------------------------------------------------------------------------
# Call graph builder from AST
# ---------------------------------------------------------------------------

class CallGraphBuilder(ast.NodeVisitor):
    """Build a call graph from Python AST."""

    def __init__(self):
        self.call_graph = CallGraph()
        self._current_func: str = "<module>"
        self._functions: Dict[str, ast.FunctionDef] = {}

    def build(self, tree: ast.Module) -> Tuple[CallGraph, Dict[str, ast.FunctionDef]]:
        # First pass: collect function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._functions[node.name] = node

        # Second pass: collect call sites
        self.visit(tree)
        return self.call_graph, self._functions

    def visit_FunctionDef(self, node: ast.FunctionDef):
        old = self._current_func
        self._current_func = node.name
        self.generic_visit(node)
        self._current_func = old

    def visit_Call(self, node: ast.Call):
        callee_name = self._resolve_callee(node.func)
        if callee_name and callee_name in self._functions:
            arg_names = tuple(
                arg.id if isinstance(arg, ast.Name) else f"<expr:{i}>"
                for i, arg in enumerate(node.args)
            )
            self.call_graph.add_edge(CallSite(
                caller=self._current_func,
                callee=callee_name,
                line=getattr(node, 'lineno', 0),
                args=arg_names,
            ))
        self.generic_visit(node)

    def _resolve_callee(self, func: ast.expr) -> Optional[str]:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None


# ---------------------------------------------------------------------------
# Function summary computation
# ---------------------------------------------------------------------------

@dataclass
class FunctionSummary:
    """Refinement type summary for a function.

    Maps (param refinements) → (return refinement, alarms).
    """
    name: str
    dep_type: DepFuncType
    alarms: List[Alarm] = field(default_factory=list)
    contexts_analyzed: int = 0
    converged: bool = False
    fixpoint_iterations: int = 0

    def pretty(self) -> str:
        return f"{self.name}: {self.dep_type.pretty()}"


@dataclass
class InterproceduralResult:
    """Result of interprocedural analysis."""
    summaries: Dict[str, FunctionSummary]
    call_graph: CallGraph
    total_alarms: List[Alarm]
    analysis_time_ms: float
    functions_analyzed: int
    sccs: List[List[str]]

    def summary_text(self) -> str:
        lines = [
            f"Interprocedural Analysis Result:",
            f"  Functions analyzed: {self.functions_analyzed}",
            f"  Total alarms: {len(self.total_alarms)}",
            f"  SCCs: {len(self.sccs)}",
            f"  Time: {self.analysis_time_ms:.1f}ms",
            "",
            "Function summaries:",
        ]
        for name, s in self.summaries.items():
            lines.append(f"  {s.pretty()}")
            for a in s.alarms:
                lines.append(f"    ⚠ {a.pretty()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interprocedural analyzer
# ---------------------------------------------------------------------------

class InterproceduralAnalyzer:
    """Summary-based interprocedural analysis with refinement types.

    Algorithm:
    1. Build call graph
    2. Compute SCCs for fixpoint ordering
    3. For each SCC (bottom-up):
       a. Initialize summaries to ⊥
       b. Iterate until fixpoint:
          - Analyze each function using current summaries
          - Update summaries with new results
          - Widen after delay iterations
    4. Use summaries at call sites for context-sensitive analysis
    """

    def __init__(self, config: Optional[CEGARConfig] = None):
        self.config = config or CEGARConfig()
        self.lattice = RefinementLattice(timeout_ms=self.config.timeout_ms)
        self.encoder = Z3Encoder()
        self.summaries: Dict[str, FunctionSummary] = {}

    def analyze(self, source: str) -> InterproceduralResult:
        """Run full interprocedural analysis on Python source."""
        start = time.monotonic()

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return InterproceduralResult(
                summaries={}, call_graph=CallGraph(),
                total_alarms=[], analysis_time_ms=0.0,
                functions_analyzed=0, sccs=[],
            )

        # Build call graph
        builder = CallGraphBuilder()
        call_graph, functions = builder.build(tree)

        # Harvest predicates
        all_preds = harvest_guards(source) + [
            Pred.var_neq("ν", 0),
            Pred.var_ge("ν", 0),
            Pred.var_gt("ν", 0),
            Pred.is_none("ν"),
            Pred.is_not_none("ν"),
        ]

        # Compute SCCs
        sccs = call_graph.sccs()

        # Process SCCs bottom-up
        all_alarms: List[Alarm] = []

        for scc in sccs:
            scc_funcs = [f for f in scc if f in functions]
            if not scc_funcs:
                continue

            is_recursive = len(scc_funcs) > 1 or call_graph.is_recursive(scc_funcs[0])

            if is_recursive:
                self._analyze_recursive_scc(
                    scc_funcs, functions, all_preds, all_alarms)
            else:
                for func_name in scc_funcs:
                    self._analyze_single_function(
                        func_name, functions[func_name],
                        all_preds, all_alarms)

        elapsed = (time.monotonic() - start) * 1000

        return InterproceduralResult(
            summaries=dict(self.summaries),
            call_graph=call_graph,
            total_alarms=all_alarms,
            analysis_time_ms=elapsed,
            functions_analyzed=len(functions),
            sccs=sccs,
        )

    def _analyze_single_function(
            self, name: str, func: ast.FunctionDef,
            predicates: List[Pred],
            all_alarms: List[Alarm]) -> None:
        """Analyze a non-recursive function and compute its summary."""
        domain = PredicateAbstractionDomain(predicates, self.lattice)
        interp = SummaryAwareInterpreter(domain, self.lattice, self.summaries)

        entry_env = RefEnvironment()
        for arg in func.args.args:
            entry_env = entry_env.set(arg.arg, RefType.trivial(ANY_TYPE))

        result_env, alarms = interp.analyze_function(func, entry_env)
        all_alarms.extend(alarms)

        # Build summary
        param_types = []
        for arg in func.args.args:
            ty = result_env.get(arg.arg) or RefType.trivial(ANY_TYPE)
            param_types.append((arg.arg, ty))

        ret_type = self._infer_return_type(func, result_env, interp)

        dep_type = DepFuncType(tuple(param_types), ret_type)
        self.summaries[name] = FunctionSummary(
            name=name, dep_type=dep_type, alarms=alarms,
            contexts_analyzed=1, converged=True,
        )

    def _analyze_recursive_scc(
            self, scc: List[str],
            functions: Dict[str, ast.FunctionDef],
            predicates: List[Pred],
            all_alarms: List[Alarm]) -> None:
        """Analyze a recursive SCC with fixpoint iteration.

        Start with ⊤ summaries (unconstrained), then iterate:
        1. Analyze each function using current summaries
        2. Update summaries
        3. Widen after delay iterations
        4. Stop when summaries stabilize
        """
        # Initialize summaries to ⊤
        for name in scc:
            func = functions[name]
            param_types = [
                (arg.arg, RefType.trivial(ANY_TYPE))
                for arg in func.args.args
            ]
            self.summaries[name] = FunctionSummary(
                name=name,
                dep_type=DepFuncType(tuple(param_types),
                                     RefType.trivial(ANY_TYPE)),
            )

        max_fp_iters = 10
        for fp_iter in range(max_fp_iters):
            changed = False

            for name in scc:
                func = functions[name]
                domain = PredicateAbstractionDomain(predicates, self.lattice)
                interp = SummaryAwareInterpreter(
                    domain, self.lattice, self.summaries)

                entry_env = RefEnvironment()
                for arg in func.args.args:
                    entry_env = entry_env.set(
                        arg.arg, RefType.trivial(ANY_TYPE))

                result_env, alarms = interp.analyze_function(func, entry_env)

                # Compute new summary
                param_types = []
                for arg in func.args.args:
                    ty = result_env.get(arg.arg) or RefType.trivial(ANY_TYPE)
                    param_types.append((arg.arg, ty))

                ret_type = self._infer_return_type(func, result_env, interp)
                new_dep_type = DepFuncType(tuple(param_types), ret_type)

                old_summary = self.summaries[name]

                # Widening after delay
                if fp_iter >= self.config.widening_delay:
                    old_ret = old_summary.dep_type.ret
                    new_ret = self.lattice.widen(
                        old_ret, ret_type, predicates)
                    new_dep_type = DepFuncType(
                        tuple(param_types), new_ret)

                # Check if summary changed
                if not self._summary_equal(old_summary.dep_type, new_dep_type):
                    changed = True

                self.summaries[name] = FunctionSummary(
                    name=name,
                    dep_type=new_dep_type,
                    alarms=alarms,
                    contexts_analyzed=fp_iter + 1,
                    converged=not changed,
                    fixpoint_iterations=fp_iter + 1,
                )

            if not changed:
                break

        # Collect final alarms
        for name in scc:
            all_alarms.extend(self.summaries[name].alarms)

    def _summary_equal(self, a: DepFuncType, b: DepFuncType) -> bool:
        """Check if two function summaries are equivalent."""
        if len(a.params) != len(b.params):
            return False
        for (na, ta), (nb, tb) in zip(a.params, b.params):
            if na != nb:
                return False
            if not self.lattice.equiv(ta, tb):
                return False
        return self.lattice.equiv(a.ret, b.ret)

    def _infer_return_type(self, func: ast.FunctionDef,
                           env: RefEnvironment,
                           interp: AbstractInterpreter) -> RefType:
        """Infer the return type of a function from its body."""
        returns: List[RefType] = []

        class ReturnCollector(ast.NodeVisitor):
            def visit_Return(self, node: ast.Return):
                if node.value:
                    ty = interp._infer_expr_type(node.value, env)
                    returns.append(ty)
                else:
                    returns.append(RefType.trivial(NONE_TYPE))

        ReturnCollector().visit(func)

        if not returns:
            return RefType.trivial(NONE_TYPE)

        result = returns[0]
        for r in returns[1:]:
            result = self.lattice.join(result, r)
        return result


# ---------------------------------------------------------------------------
# Summary-aware interpreter
# ---------------------------------------------------------------------------

class SummaryAwareInterpreter(AbstractInterpreter):
    """Abstract interpreter that uses function summaries at call sites.

    When analyzing a call f(x₁, ..., xₙ), if a summary for f is available,
    we:
    1. Check argument types against parameter refinements
    2. Substitute argument types into the return refinement
    3. Use the refined return type in the caller
    """

    def __init__(self, domain: PredicateAbstractionDomain,
                 lattice: RefinementLattice,
                 summaries: Dict[str, FunctionSummary]):
        super().__init__(domain, lattice)
        self.func_summaries = summaries

    def _infer_expr_type(self, expr: ast.expr,
                         env: RefEnvironment) -> RefType:
        """Override: use function summaries for call expressions."""
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            func_name = expr.func.id
            if func_name in self.func_summaries:
                summary = self.func_summaries[func_name]
                dep_type = summary.dep_type

                # Build argument map
                arg_types: Dict[str, RefType] = {}
                for i, (param_name, param_type) in enumerate(dep_type.params):
                    if i < len(expr.args):
                        arg_type = super()._infer_expr_type(
                            expr.args[i], env)
                        arg_types[param_name] = arg_type

                        # Check argument against parameter
                        errors = dep_type.check_args(arg_types, self.lattice)
                        for err in errors:
                            from ..cegar.cegar_loop import AlarmKind
                            self.alarms.append(Alarm(
                                AlarmKind.TYPE_ERROR,
                                getattr(expr, 'lineno', 0),
                                getattr(expr, 'col_offset', 0),
                                err,
                                Pred.true_(),
                            ))

                # Apply summary
                return dep_type.apply(arg_types, self.lattice)

        return super()._infer_expr_type(expr, env)


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def analyze_interprocedural(source: str,
                            verbose: bool = False) -> InterproceduralResult:
    """Run interprocedural analysis on Python source code."""
    config = CEGARConfig(verbose=verbose)
    analyzer = InterproceduralAnalyzer(config)
    return analyzer.analyze(source)


def analyze_file_interprocedural(filepath: str,
                                 verbose: bool = False) -> InterproceduralResult:
    """Run interprocedural analysis on a Python file."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        source = f.read()
    return analyze_interprocedural(source, verbose)
