"""
Optimal memory fence placement via delay set analysis, ILP (Simplex),
heuristic placement, verification, and fence strength reduction.
"""

import numpy as np
from collections import defaultdict
from itertools import combinations
import copy


# ---------------------------------------------------------------------------
# Program representation
# ---------------------------------------------------------------------------

class MemoryAccess:
    """Single memory access instruction."""

    def __init__(self, thread, index, op, address, value=None):
        self.thread = thread
        self.index = index      # position within thread
        self.op = op            # 'load' or 'store'
        self.address = address
        self.value = value

    def is_load(self):
        return self.op == 'load'

    def is_store(self):
        return self.op == 'store'

    def __repr__(self):
        return f"T{self.thread}[{self.index}]:{self.op}({self.address})"

    def __eq__(self, other):
        return (isinstance(other, MemoryAccess)
                and self.thread == other.thread
                and self.index == other.index)

    def __hash__(self):
        return hash((self.thread, self.index))


class FenceInsertion:
    """Represents a fence inserted between two instructions."""

    def __init__(self, thread, after_index, fence_type='mfence'):
        self.thread = thread
        self.after_index = after_index
        self.fence_type = fence_type

    def __repr__(self):
        return f"Fence({self.fence_type}, T{self.thread}, after={self.after_index})"

    def __eq__(self, other):
        return (isinstance(other, FenceInsertion)
                and self.thread == other.thread
                and self.after_index == other.after_index
                and self.fence_type == other.fence_type)

    def __hash__(self):
        return hash((self.thread, self.after_index, self.fence_type))


class ConcurrentProgram:
    """A concurrent program with multiple threads."""

    def __init__(self):
        self.threads = defaultdict(list)  # tid -> list of MemoryAccess
        self.n_threads = 0

    def add_access(self, thread, op, address, value=None):
        idx = len(self.threads[thread])
        acc = MemoryAccess(thread, idx, op, address, value)
        self.threads[thread].append(acc)
        self.n_threads = max(self.n_threads, thread + 1)
        return acc

    def add_store(self, thread, address, value=None):
        return self.add_access(thread, 'store', address, value)

    def add_load(self, thread, address):
        return self.add_access(thread, 'load', address)

    def all_accesses(self):
        result = []
        for tid in sorted(self.threads.keys()):
            result.extend(self.threads[tid])
        return result

    def get_pairs_in_thread(self, tid):
        """Return all consecutive pairs of accesses in a thread."""
        accs = self.threads[tid]
        pairs = []
        for i in range(len(accs) - 1):
            pairs.append((accs[i], accs[i + 1]))
        return pairs

    def copy(self):
        return copy.deepcopy(self)


class FencedProgram:
    """Program with fence insertions."""

    def __init__(self, program, fences=None):
        self.program = program
        self.fences = fences or []

    def total_cost(self, cost_model=None):
        if cost_model is None:
            cost_model = FenceCostModel()
        return sum(cost_model.cost(f.fence_type) for f in self.fences)

    def __repr__(self):
        lines = []
        for tid in sorted(self.program.threads.keys()):
            accs = self.program.threads[tid]
            thread_fences = sorted(
                [f for f in self.fences if f.thread == tid],
                key=lambda f: f.after_index
            )
            parts = []
            fence_after = {f.after_index: f for f in thread_fences}
            for acc in accs:
                parts.append(str(acc))
                if acc.index in fence_after:
                    parts.append(f"  [{fence_after[acc.index].fence_type}]")
            lines.append(f"T{tid}: " + " ; ".join(parts))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fence cost model
# ---------------------------------------------------------------------------

class FenceCostModel:
    """Cost of different fence types on x86."""

    COSTS = {
        'mfence': 100,     # full fence, most expensive
        'sfence': 20,      # store fence
        'lfence': 20,      # load fence
        'lock_prefix': 50, # lock prefix on RMW
        'dmb_ish': 80,     # ARM full barrier
        'dmb_ishst': 40,   # ARM store barrier
        'dmb_ishld': 30,   # ARM load barrier
        'isb': 60,         # ARM instruction barrier
        'sync': 90,        # POWER sync
        'lwsync': 45,      # POWER lwsync
    }

    def cost(self, fence_type):
        return self.COSTS.get(fence_type, 50)

    def all_types(self, target='x86'):
        if target == 'x86':
            return ['mfence', 'sfence', 'lfence']
        elif target == 'arm':
            return ['dmb_ish', 'dmb_ishst', 'dmb_ishld']
        elif target == 'power':
            return ['sync', 'lwsync']
        return ['mfence']


# ---------------------------------------------------------------------------
# Delay set analysis
# ---------------------------------------------------------------------------

class DelaySetAnalyzer:
    """Compute which instruction pairs need ordering to ensure SC."""

    def __init__(self, program):
        self.program = program

    def compute_delay_set(self, target_model='tso'):
        """Compute the set of instruction pairs that may be reordered
        under target_model but must be ordered for SC.
        Returns list of (acc_a, acc_b) pairs within same thread.
        """
        delays = []
        for tid in self.program.threads:
            accs = self.program.threads[tid]
            for i in range(len(accs)):
                for j in range(i + 1, len(accs)):
                    a, b = accs[i], accs[j]
                    if self._reorderable(a, b, target_model):
                        delays.append((a, b))
        return delays

    def _reorderable(self, a, b, model):
        """Check if (a, b) can be reordered under model."""
        if model == 'tso':
            # TSO allows store->load reordering (to different addresses)
            return a.is_store() and b.is_load() and a.address != b.address
        elif model == 'pso':
            # PSO also allows store->store to different addresses
            if a.is_store() and b.is_load() and a.address != b.address:
                return True
            if a.is_store() and b.is_store() and a.address != b.address:
                return True
            return False
        elif model == 'arm' or model == 'relaxed':
            # ARM allows all reorderings except same-address
            if a.address == b.address:
                return False
            return True
        return False

    def compute_critical_pairs(self, target_model='tso'):
        """Find the minimal set of pairs whose ordering prevents all SC violations.
        Uses the cycle-based analysis: find all critical cycles.
        """
        delays = self.compute_delay_set(target_model)
        if not delays:
            return []

        # For simplicity, all delay pairs are critical in the minimal case
        # A more precise analysis would check which delays participate in cycles
        critical = []
        for a, b in delays:
            # Check if this pair participates in a cross-thread communication pattern
            other_thread_accesses = []
            for tid in self.program.threads:
                if tid != a.thread:
                    for acc in self.program.threads[tid]:
                        if acc.address == a.address or acc.address == b.address:
                            other_thread_accesses.append(acc)
            if other_thread_accesses:
                critical.append((a, b))

        return critical if critical else delays


# ---------------------------------------------------------------------------
# Simplex solver for ILP (minimal implementation)
# ---------------------------------------------------------------------------

class SimplexSolver:
    """Minimal Simplex implementation for fence placement ILP.
    Solves: minimize c^T x subject to Ax <= b, x >= 0.
    """

    def __init__(self, c, A, b):
        self.c = np.array(c, dtype=np.float64)
        self.A = np.array(A, dtype=np.float64)
        self.b = np.array(b, dtype=np.float64)
        self.m, self.n = self.A.shape

    def solve(self, max_iterations=1000):
        """Solve LP relaxation. Returns (optimal_value, solution) or None."""
        m, n = self.m, self.n
        # Add slack variables
        tableau = np.zeros((m + 1, n + m + 1))
        tableau[:m, :n] = self.A
        tableau[:m, n:n+m] = np.eye(m)
        tableau[:m, -1] = self.b
        tableau[-1, :n] = -self.c

        basis = list(range(n, n + m))

        for _ in range(max_iterations):
            # Find pivot column (most negative in objective row)
            obj_row = tableau[-1, :-1]
            if np.all(obj_row >= -1e-10):
                break  # Optimal
            pivot_col = int(np.argmin(obj_row))

            # Find pivot row (minimum ratio test)
            col_vals = tableau[:m, pivot_col]
            rhs = tableau[:m, -1]
            ratios = np.full(m, np.inf)
            for i in range(m):
                if col_vals[i] > 1e-10:
                    ratios[i] = rhs[i] / col_vals[i]

            if np.all(np.isinf(ratios)):
                return None  # Unbounded

            pivot_row = int(np.argmin(ratios))
            basis[pivot_row] = pivot_col

            # Pivot operation
            pivot_val = tableau[pivot_row, pivot_col]
            tableau[pivot_row, :] /= pivot_val
            for i in range(m + 1):
                if i != pivot_row:
                    factor = tableau[i, pivot_col]
                    tableau[i, :] -= factor * tableau[pivot_row, :]

        # Extract solution
        solution = np.zeros(n)
        for i, var in enumerate(basis):
            if var < n:
                solution[var] = tableau[i, -1]

        optimal_value = float(tableau[-1, -1])
        return optimal_value, solution

    def solve_integer(self):
        """Solve integer program by LP relaxation + rounding."""
        result = self.solve()
        if result is None:
            return None
        opt_val, solution = result
        # Round to nearest integer (branch-and-bound would be more precise)
        int_solution = np.round(solution).astype(int)
        int_solution = np.maximum(int_solution, 0)
        # Verify feasibility
        if np.all(self.A @ int_solution <= self.b + 1e-10):
            int_val = float(self.c @ int_solution)
            return int_val, int_solution
        # If rounding fails, try ceiling
        ceil_solution = np.ceil(solution).astype(int)
        ceil_solution = np.maximum(ceil_solution, 0)
        if np.all(self.A @ ceil_solution <= self.b + 1e-10):
            ceil_val = float(self.c @ ceil_solution)
            return ceil_val, ceil_solution
        return float(self.c @ int_solution), int_solution


# ---------------------------------------------------------------------------
# Fence Optimizer
# ---------------------------------------------------------------------------

class FenceOptimizer:
    """Optimize fence placement for a concurrent program."""

    def __init__(self, cost_model=None):
        self.cost_model = cost_model or FenceCostModel()

    def optimize(self, program, target_model='tso'):
        """Find minimum-cost fence placement to ensure SC on target model."""
        analyzer = DelaySetAnalyzer(program)
        delay_pairs = analyzer.compute_delay_set(target_model)

        if not delay_pairs:
            return FencedProgram(program, [])

        # Try ILP first
        fences = self._ilp_placement(program, delay_pairs, target_model)
        if fences is not None:
            return FencedProgram(program, fences)

        # Fallback to heuristic
        fences = self._heuristic_placement(program, delay_pairs, target_model)
        return FencedProgram(program, fences)

    def _ilp_placement(self, program, delay_pairs, target_model):
        """Optimal fence placement via ILP.
        Variables: x_{t,i} = 1 if fence after instruction i in thread t.
        Objective: minimize sum of costs.
        Constraints: for each delay pair (a,b), at least one fence between them.
        """
        # Enumerate all possible fence positions
        positions = []
        pos_index = {}
        for tid in sorted(program.threads.keys()):
            accs = program.threads[tid]
            for i in range(len(accs) - 1):
                pos = (tid, i)
                pos_index[pos] = len(positions)
                positions.append(pos)

        n_vars = len(positions)
        if n_vars == 0:
            return []

        # Determine fence type needed for each position
        fence_types = {}
        for pos in positions:
            tid, idx = pos
            a = program.threads[tid][idx]
            b = program.threads[tid][idx + 1]
            fence_types[pos] = self._needed_fence_type(a, b, target_model)

        # Objective: minimize cost
        c = np.array([self.cost_model.cost(fence_types[p]) for p in positions],
                      dtype=np.float64)

        # Constraints: for each delay pair, at least one fence between a and b
        constraints_A = []
        constraints_b = []
        for a, b in delay_pairs:
            if a.thread != b.thread:
                continue
            tid = a.thread
            row = np.zeros(n_vars)
            for idx in range(a.index, b.index):
                pos = (tid, idx)
                if pos in pos_index:
                    row[pos_index[pos]] = -1.0  # -x >= -1 => x >= 1
            constraints_A.append(row)
            constraints_b.append(-1.0)  # At least one fence

        if not constraints_A:
            return []

        A = np.array(constraints_A)
        b_vec = np.array(constraints_b)

        solver = SimplexSolver(c, A, b_vec)
        result = solver.solve_integer()
        if result is None:
            return None

        _, solution = result
        fences = []
        for i, pos in enumerate(positions):
            if solution[i] > 0.5:
                tid, idx = pos
                fences.append(FenceInsertion(tid, idx, fence_types[pos]))
        return fences

    def _needed_fence_type(self, a, b, target_model):
        """Determine the minimum fence type needed between a and b."""
        if target_model == 'tso':
            if a.is_store() and b.is_load():
                return 'mfence'
            return 'mfence'
        elif target_model == 'pso':
            if a.is_store() and b.is_store():
                return 'sfence'
            if a.is_store() and b.is_load():
                return 'mfence'
            return 'mfence'
        elif target_model in ('arm', 'relaxed'):
            if a.is_store() and b.is_load():
                return 'dmb_ish'
            if a.is_store() and b.is_store():
                return 'dmb_ishst'
            if a.is_load() and b.is_load():
                return 'dmb_ishld'
            return 'dmb_ish'
        return 'mfence'

    def _heuristic_placement(self, program, delay_pairs, target_model):
        """Heuristic: insert fence after each instruction that starts a delay pair."""
        fences = set()
        for a, b in delay_pairs:
            if a.thread != b.thread:
                continue
            ft = self._needed_fence_type(a, b, target_model)
            fences.add(FenceInsertion(a.thread, a.index, ft))
        return list(fences)


# ---------------------------------------------------------------------------
# Fence verification
# ---------------------------------------------------------------------------

class FenceVerifier:
    """Verify that a fenced program achieves SC on the target model."""

    def verify(self, fenced_program, target_model='tso'):
        """Check that all delay pairs are covered by fences."""
        program = fenced_program.program
        analyzer = DelaySetAnalyzer(program)
        delay_pairs = analyzer.compute_delay_set(target_model)

        fences_by_thread = defaultdict(list)
        for f in fenced_program.fences:
            fences_by_thread[f.thread].append(f)

        uncovered = []
        for a, b in delay_pairs:
            if a.thread != b.thread:
                continue
            tid = a.thread
            covered = False
            for f in fences_by_thread[tid]:
                if a.index <= f.after_index < b.index:
                    if self._fence_covers(f, a, b, target_model):
                        covered = True
                        break
            if not covered:
                uncovered.append((a, b))

        return {
            'verified': len(uncovered) == 0,
            'uncovered_pairs': uncovered,
            'total_delay_pairs': len(delay_pairs),
            'covered_pairs': len(delay_pairs) - len(uncovered),
        }

    def _fence_covers(self, fence, a, b, target_model):
        """Check if fence type is sufficient to order (a, b)."""
        ft = fence.fence_type
        if ft in ('mfence', 'dmb_ish', 'sync'):
            return True
        if ft in ('sfence', 'dmb_ishst') and a.is_store() and b.is_store():
            return True
        if ft in ('lfence', 'dmb_ishld') and a.is_load() and b.is_load():
            return True
        if ft == 'lwsync':
            if not (a.is_store() and b.is_load()):
                return True
        return False


# ---------------------------------------------------------------------------
# Fence strength reduction
# ---------------------------------------------------------------------------

class FenceStrengthReducer:
    """Replace strong fences with weaker ones where safe."""

    def __init__(self, cost_model=None):
        self.cost_model = cost_model or FenceCostModel()

    def reduce(self, fenced_program, target_model='tso'):
        """Try to replace each fence with a weaker one."""
        verifier = FenceVerifier()
        program = fenced_program.program

        # Sort fences by cost (try to reduce most expensive first)
        fences = list(fenced_program.fences)
        fences.sort(key=lambda f: -self.cost_model.cost(f.fence_type))

        strength_order = self._strength_order(target_model)
        reduced_fences = list(fences)

        for i, fence in enumerate(fences):
            current_type = fence.fence_type
            if current_type not in strength_order:
                continue
            current_idx = strength_order.index(current_type)

            # Try weaker fences
            for weaker_idx in range(current_idx + 1, len(strength_order)):
                trial_fences = list(reduced_fences)
                trial_fences[i] = FenceInsertion(
                    fence.thread, fence.after_index,
                    strength_order[weaker_idx]
                )
                trial = FencedProgram(program, trial_fences)
                result = verifier.verify(trial, target_model)
                if result['verified']:
                    reduced_fences[i] = trial_fences[i]
                    break

        return FencedProgram(program, reduced_fences)

    def _strength_order(self, target_model):
        """Fence types from strongest to weakest."""
        if target_model == 'tso':
            return ['mfence', 'sfence', 'lfence']
        elif target_model in ('arm', 'relaxed'):
            return ['dmb_ish', 'dmb_ishst', 'dmb_ishld']
        elif target_model == 'power':
            return ['sync', 'lwsync']
        return ['mfence']


# ---------------------------------------------------------------------------
# Heuristic: fence at control-flow merge points
# ---------------------------------------------------------------------------

class MergePointFencer:
    """Insert fences at control-flow merge points in the program.
    For straight-line code, merge points are after conditional branches.
    Here we model merge points as positions where multiple paths converge.
    """

    def __init__(self, cost_model=None):
        self.cost_model = cost_model or FenceCostModel()

    def place_fences(self, program, target_model='tso', merge_points=None):
        """Place fences at specified merge points, or infer them."""
        if merge_points is None:
            merge_points = self._infer_merge_points(program)

        fences = []
        for tid, idx in merge_points:
            ft = 'mfence' if target_model == 'tso' else 'dmb_ish'
            fences.append(FenceInsertion(tid, idx, ft))
        return FencedProgram(program, fences)

    def _infer_merge_points(self, program):
        """Infer merge points as positions between stores and loads
        (where reordering is most dangerous).
        """
        merge_points = []
        for tid in program.threads:
            accs = program.threads[tid]
            for i in range(len(accs) - 1):
                a, b = accs[i], accs[i + 1]
                if a.is_store() and b.is_load() and a.address != b.address:
                    merge_points.append((tid, i))
        return merge_points


# ---------------------------------------------------------------------------
# Complete optimization pipeline
# ---------------------------------------------------------------------------

class FenceOptimizationPipeline:
    """Full pipeline: analyze -> optimize -> reduce -> verify."""

    def __init__(self, cost_model=None):
        self.cost_model = cost_model or FenceCostModel()
        self.optimizer = FenceOptimizer(self.cost_model)
        self.reducer = FenceStrengthReducer(self.cost_model)
        self.verifier = FenceVerifier()

    def run(self, program, target_model='tso'):
        """Run full optimization pipeline."""
        # Step 1: initial optimization
        fenced = self.optimizer.optimize(program, target_model)
        initial_cost = fenced.total_cost(self.cost_model)

        # Step 2: strength reduction
        reduced = self.reducer.reduce(fenced, target_model)
        reduced_cost = reduced.total_cost(self.cost_model)

        # Step 3: verify
        verification = self.verifier.verify(reduced, target_model)

        return {
            'fenced_program': reduced,
            'initial_cost': initial_cost,
            'reduced_cost': reduced_cost,
            'savings': initial_cost - reduced_cost,
            'n_fences': len(reduced.fences),
            'verified': verification['verified'],
            'uncovered': verification['uncovered_pairs'],
        }


# ---------------------------------------------------------------------------
# Example programs
# ---------------------------------------------------------------------------

def make_store_buffering_program():
    """SB: T0: store x; load y   T1: store y; load x"""
    prog = ConcurrentProgram()
    prog.add_store(0, 'x', 1)
    prog.add_load(0, 'y')
    prog.add_store(1, 'y', 1)
    prog.add_load(1, 'x')
    return prog


def make_message_passing_program():
    """MP: T0: store x; store y   T1: load y; load x"""
    prog = ConcurrentProgram()
    prog.add_store(0, 'x', 1)
    prog.add_store(0, 'y', 1)
    prog.add_load(1, 'y')
    prog.add_load(1, 'x')
    return prog


def make_dekker_program():
    """Dekker's algorithm (simplified):
    T0: store flag0; load flag1; [critical section]
    T1: store flag1; load flag0; [critical section]
    """
    prog = ConcurrentProgram()
    prog.add_store(0, 'flag0', 1)
    prog.add_load(0, 'flag1')
    prog.add_store(0, 'cs', 1)
    prog.add_store(1, 'flag1', 1)
    prog.add_load(1, 'flag0')
    prog.add_store(1, 'cs', 2)
    return prog


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Fence Optimizer: Store Buffering ===")
    sb = make_store_buffering_program()
    pipeline = FenceOptimizationPipeline()
    result = pipeline.run(sb, 'tso')
    print(f"Fences: {result['n_fences']}")
    print(f"Cost: {result['reduced_cost']}")
    print(f"Verified: {result['verified']}")
    print(result['fenced_program'])

    print("\n=== Fence Optimizer: Message Passing (ARM) ===")
    mp = make_message_passing_program()
    result_arm = pipeline.run(mp, 'arm')
    print(f"Fences: {result_arm['n_fences']}")
    print(f"Cost: {result_arm['reduced_cost']}")
    print(f"Verified: {result_arm['verified']}")
    print(result_arm['fenced_program'])

    print("\n=== Simplex Solver Test ===")
    # min 2x + 3y s.t. x + y >= 1 (i.e. -x - y <= -1), x,y >= 0
    c = np.array([2.0, 3.0])
    A = np.array([[-1.0, -1.0]])
    b = np.array([-1.0])
    solver = SimplexSolver(c, A, b)
    res = solver.solve()
    assert res is not None
    val, sol = res
    print(f"Simplex: optimal={val:.2f}, solution={sol}")
    assert abs(val - 2.0) < 0.1, f"Expected ~2.0, got {val}"

    print("\nfence_optimizer.py self-test passed")
