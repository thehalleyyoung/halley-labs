"""
Litmus test generation, evaluation, and classification engine.

Supports standard litmus tests (SB, MP, LB, IRIW, WRC, CoRR/CoWR/CoRW/CoWW),
outcome enumeration, test minimization, classification, and random generation.
"""

import numpy as np
from itertools import permutations, product
from collections import defaultdict
import copy
import re

from memory_model import (
    MemoryEvent, EventType, FenceType, Scope,
    Read, Write, Fence, RMW,
    Execution, Relation, ExecutionBuilder,
    SequentialConsistency, TotalStoreOrder, PartialStoreOrder,
    RelaxedMemoryModel, RISCVModel, PTXModel,
    get_model, MemoryModel,
)


# ---------------------------------------------------------------------------
# Instruction representation for litmus tests
# ---------------------------------------------------------------------------

class Instruction:
    """Single instruction in a litmus test thread."""

    def __init__(self, op, address, value=None, register=None, fence_type=None):
        self.op = op            # 'store', 'load', 'fence', 'rmw', 'cas'
        self.address = address  # memory address (string or int)
        self.value = value      # value for stores
        self.register = register  # register for loads
        self.fence_type = fence_type

    def __repr__(self):
        if self.op == 'store':
            return f"W({self.address})={self.value}"
        elif self.op == 'load':
            return f"{self.register}=R({self.address})"
        elif self.op == 'fence':
            return f"FENCE({self.fence_type})"
        elif self.op == 'rmw':
            return f"RMW({self.address},{self.value})"
        elif self.op == 'cas':
            return f"CAS({self.address},{self.value})"
        return f"?({self.op})"

    def __eq__(self, other):
        return (isinstance(other, Instruction) and self.op == other.op
                and self.address == other.address and self.value == other.value
                and self.register == other.register)

    def __hash__(self):
        return hash((self.op, self.address, self.value, self.register))


def Store(addr, val):
    return Instruction('store', addr, value=val)


def Load(addr, reg):
    return Instruction('load', addr, register=reg)


def FenceInst(ftype='full'):
    return Instruction('fence', None, fence_type=ftype)


def RMWInst(addr, val):
    return Instruction('rmw', addr, value=val)


# ---------------------------------------------------------------------------
# Outcome representation
# ---------------------------------------------------------------------------

class Outcome:
    """An outcome maps registers to values."""

    def __init__(self, register_values=None):
        self.values = dict(register_values) if register_values else {}

    def set(self, reg, val):
        self.values[reg] = val

    def get(self, reg):
        return self.values.get(reg)

    def __eq__(self, other):
        return isinstance(other, Outcome) and self.values == other.values

    def __hash__(self):
        return hash(tuple(sorted(self.values.items())))

    def __repr__(self):
        parts = [f"{k}={v}" for k, v in sorted(self.values.items())]
        return "{" + ", ".join(parts) + "}"

    def matches(self, other):
        """Check if this outcome matches (is a superset of) other."""
        for k, v in other.values.items():
            if k in self.values and self.values[k] != v:
                return False
        return True


# ---------------------------------------------------------------------------
# LitmusTest
# ---------------------------------------------------------------------------

class LitmusTest:
    """A litmus test with threads, initial state, and expected outcomes."""

    def __init__(self, name, threads, initial_state=None,
                 forbidden_outcomes=None, expected_outcomes=None):
        self.name = name
        self.threads = threads              # list of list of Instruction
        self.initial_state = initial_state or {}  # addr -> value
        self.forbidden_outcomes = forbidden_outcomes or []
        self.expected_outcomes = expected_outcomes or []

    def num_threads(self):
        return len(self.threads)

    def all_addresses(self):
        addrs = set()
        for thread in self.threads:
            for instr in thread:
                if instr.address is not None:
                    addrs.add(instr.address)
        return addrs

    def all_registers(self):
        regs = set()
        for thread in self.threads:
            for instr in thread:
                if instr.register is not None:
                    regs.add(instr.register)
        return regs

    def __repr__(self):
        lines = [f"Litmus Test: {self.name}"]
        for i, thread in enumerate(self.threads):
            instrs = " ; ".join(str(ins) for ins in thread)
            lines.append(f"  T{i}: {instrs}")
        if self.forbidden_outcomes:
            lines.append(f"  Forbidden: {self.forbidden_outcomes}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in litmus tests
# ---------------------------------------------------------------------------

def make_store_buffering():
    """SB: Store Buffering test.
    T0: W(x)=1; r0=R(y)
    T1: W(y)=1; r1=R(x)
    Forbidden on SC: r0=0, r1=0
    """
    t0 = [Store('x', 1), Load('y', 'r0')]
    t1 = [Store('y', 1), Load('x', 'r1')]
    forbidden = [Outcome({'r0': 0, 'r1': 0})]
    return LitmusTest("SB", [t0, t1], {'x': 0, 'y': 0}, forbidden)


def make_message_passing():
    """MP: Message Passing test.
    T0: W(x)=1; W(y)=1
    T1: r0=R(y); r1=R(x)
    Forbidden on SC: r0=1, r1=0
    """
    t0 = [Store('x', 1), Store('y', 1)]
    t1 = [Load('y', 'r0'), Load('x', 'r1')]
    forbidden = [Outcome({'r0': 1, 'r1': 0})]
    return LitmusTest("MP", [t0, t1], {'x': 0, 'y': 0}, forbidden)


def make_load_buffering():
    """LB: Load Buffering test.
    T0: r0=R(x); W(y)=1
    T1: r1=R(y); W(x)=1
    Forbidden on SC: r0=1, r1=1
    """
    t0 = [Load('x', 'r0'), Store('y', 1)]
    t1 = [Load('y', 'r1'), Store('x', 1)]
    forbidden = [Outcome({'r0': 1, 'r1': 1})]
    return LitmusTest("LB", [t0, t1], {'x': 0, 'y': 0}, forbidden)


def make_iriw():
    """IRIW: Independent Reads of Independent Writes.
    T0: W(x)=1
    T1: W(y)=1
    T2: r0=R(x); r1=R(y)
    T3: r2=R(y); r3=R(x)
    Forbidden on SC: r0=1,r1=0,r2=1,r3=0
    """
    t0 = [Store('x', 1)]
    t1 = [Store('y', 1)]
    t2 = [Load('x', 'r0'), Load('y', 'r1')]
    t3 = [Load('y', 'r2'), Load('x', 'r3')]
    forbidden = [Outcome({'r0': 1, 'r1': 0, 'r2': 1, 'r3': 0})]
    return LitmusTest("IRIW", [t0, t1, t2, t3], {'x': 0, 'y': 0}, forbidden)


def make_wrc():
    """WRC: Write-Read Causality.
    T0: W(x)=1
    T1: r0=R(x); W(y)=1
    T2: r1=R(y); r2=R(x)
    Forbidden on SC: r0=1, r1=1, r2=0
    """
    t0 = [Store('x', 1)]
    t1 = [Load('x', 'r0'), Store('y', 1)]
    t2 = [Load('y', 'r1'), Load('x', 'r2')]
    forbidden = [Outcome({'r0': 1, 'r1': 1, 'r2': 0})]
    return LitmusTest("WRC", [t0, t1, t2], {'x': 0, 'y': 0}, forbidden)


def make_corr():
    """CoRR: Coherence Read-Read.
    T0: W(x)=1
    T1: W(x)=2
    T2: r0=R(x); r1=R(x)
    Forbidden on all models: r0=2, r1=1 (violates coherence order)
    """
    t0 = [Store('x', 1)]
    t1 = [Store('x', 2)]
    t2 = [Load('x', 'r0'), Load('x', 'r1')]
    forbidden = [Outcome({'r0': 2, 'r1': 1})]
    return LitmusTest("CoRR", [t0, t1, t2], {'x': 0}, forbidden)


def make_cowr():
    """CoWR: Coherence Write-Read.
    T0: W(x)=1; r0=R(x)
    T1: W(x)=2
    Forbidden: W(x)=2 co-before W(x)=1, yet r0=1 (seeing own stale write)
    """
    t0 = [Store('x', 1), Load('x', 'r0')]
    t1 = [Store('x', 2)]
    forbidden = [Outcome({'r0': 2})]
    return LitmusTest("CoWR", [t0, t1], {'x': 0}, forbidden)


def make_corw():
    """CoRW: Coherence Read-Write.
    T0: r0=R(x); W(x)=1
    T1: W(x)=2
    """
    t0 = [Load('x', 'r0'), Store('x', 1)]
    t1 = [Store('x', 2)]
    forbidden = []
    return LitmusTest("CoRW", [t0, t1], {'x': 0}, forbidden)


def make_coww():
    """CoWW: Coherence Write-Write.
    T0: W(x)=1; W(x)=2
    Coherence requires T0's writes in program order.
    """
    t0 = [Store('x', 1), Store('x', 2)]
    t1 = [Load('x', 'r0')]
    forbidden = []
    return LitmusTest("CoWW", [t0, t1], {'x': 0}, forbidden)


BUILTIN_TESTS = {
    'SB': make_store_buffering,
    'MP': make_message_passing,
    'LB': make_load_buffering,
    'IRIW': make_iriw,
    'WRC': make_wrc,
    'CoRR': make_corr,
    'CoWR': make_cowr,
    'CoRW': make_corw,
    'CoWW': make_coww,
}


def get_all_builtin_tests():
    return {name: fn() for name, fn in BUILTIN_TESTS.items()}


# ---------------------------------------------------------------------------
# Outcome enumeration
# ---------------------------------------------------------------------------

def _addr_to_int(addr):
    """Convert address to int for memory model events."""
    if isinstance(addr, int):
        return addr
    return hash(addr) % 10000


class OutcomeEnumerator:
    """Enumerate all possible outcomes of a litmus test under a memory model."""

    def __init__(self, test, model=None):
        self.test = test
        self.model = model

    def generate_all_outcomes(self):
        """Enumerate all interleavings and collect distinct outcomes."""
        threads = self.test.threads
        n_threads = len(threads)
        # Generate all possible interleavings via interleaving semantics
        outcomes = set()
        initial_mem = dict(self.test.initial_state)

        thread_lengths = [len(t) for t in threads]
        self._enumerate_recursive(threads, thread_lengths,
                                  [0] * n_threads, initial_mem, {},
                                  outcomes)
        return outcomes

    def _enumerate_recursive(self, threads, lengths, pcs, memory, registers, outcomes):
        """Recursively enumerate all interleavings."""
        # Check if all threads are done
        if all(pcs[i] >= lengths[i] for i in range(len(threads))):
            outcomes.add(Outcome(registers))
            return

        for tid in range(len(threads)):
            if pcs[tid] >= lengths[tid]:
                continue
            instr = threads[tid][pcs[tid]]
            new_pcs = list(pcs)
            new_pcs[tid] += 1
            new_mem = dict(memory)
            new_regs = dict(registers)

            if instr.op == 'store':
                new_mem[instr.address] = instr.value
                self._enumerate_recursive(threads, lengths, new_pcs,
                                          new_mem, new_regs, outcomes)
            elif instr.op == 'load':
                val = new_mem.get(instr.address, 0)
                new_regs[instr.register] = val
                self._enumerate_recursive(threads, lengths, new_pcs,
                                          new_mem, new_regs, outcomes)
            elif instr.op == 'fence':
                self._enumerate_recursive(threads, lengths, new_pcs,
                                          new_mem, new_regs, outcomes)
            elif instr.op == 'rmw':
                old_val = new_mem.get(instr.address, 0)
                new_mem[instr.address] = instr.value
                self._enumerate_recursive(threads, lengths, new_pcs,
                                          new_mem, new_regs, outcomes)

    def generate_all_outcomes_for_model(self, model):
        """Generate outcomes and classify as allowed/forbidden under model."""
        all_outcomes = self.generate_all_outcomes()
        allowed = set()
        forbidden = set()

        for outcome in all_outcomes:
            exe = self._outcome_to_execution(outcome)
            if exe is not None:
                result, _ = model.check(exe)
                if result == "allowed":
                    allowed.add(outcome)
                else:
                    forbidden.add(outcome)
            else:
                allowed.add(outcome)

        return allowed, forbidden

    def _outcome_to_execution(self, outcome):
        """Convert an outcome back to an execution for model checking."""
        eb = ExecutionBuilder()
        for addr in self.test.all_addresses():
            val = self.test.initial_state.get(addr, 0)
            eb.init(_addr_to_int(addr), val)

        events_map = {}
        for tid, thread in enumerate(self.test.threads):
            for idx, instr in enumerate(thread):
                addr_int = _addr_to_int(instr.address) if instr.address else None
                if instr.op == 'store':
                    e = eb.write(tid, addr_int, instr.value)
                    events_map[(tid, idx)] = e
                elif instr.op == 'load':
                    val = outcome.get(instr.register)
                    if val is None:
                        val = 0
                    e = eb.read(tid, addr_int, val)
                    events_map[(tid, idx)] = e
                elif instr.op == 'fence':
                    ft = FenceType.FULL
                    e = eb.fence(tid, ft)
                    events_map[(tid, idx)] = e

        # Build rf: each read reads from the write that produced its value
        exe = eb.build()
        for e in exe.events:
            if e.is_read():
                writes = exe.writes_to(e.address)
                init_w = exe.init_writes.get(e.address)
                # Find a write with matching value
                matched = False
                for w in writes:
                    if w.value == e.value and w.thread != e.thread:
                        exe.rf.add(w, e)
                        matched = True
                        break
                if not matched:
                    for w in writes:
                        if w.value == e.value:
                            exe.rf.add(w, e)
                            matched = True
                            break
                if not matched and init_w and init_w.value == e.value:
                    exe.rf.add(init_w, e)

        return exe


def generate_all_outcomes(test):
    """Convenience: enumerate all possible outcomes."""
    enumerator = OutcomeEnumerator(test)
    return enumerator.generate_all_outcomes()


def check_outcome(test, outcome, model):
    """Check if a specific outcome is allowed under a model."""
    enumerator = OutcomeEnumerator(test, model)
    exe = enumerator._outcome_to_execution(outcome)
    if exe is None:
        return "allowed"
    result, reason = model.check(exe)
    return result


# ---------------------------------------------------------------------------
# Litmus test parser
# ---------------------------------------------------------------------------

def litmus_test_from_string(text):
    """Parse a litmus test from a simple text format.

    Format:
        name: TestName
        init: x=0; y=0
        T0: W(x)=1; r0=R(y)
        T1: W(y)=1; r1=R(x)
        forbidden: r0=0, r1=0
    """
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    name = "unnamed"
    init_state = {}
    threads = []
    forbidden = []

    for line in lines:
        if line.startswith('name:'):
            name = line.split(':', 1)[1].strip()
        elif line.startswith('init:'):
            init_str = line.split(':', 1)[1].strip()
            for part in init_str.split(';'):
                part = part.strip()
                if '=' in part:
                    addr, val = part.split('=')
                    init_state[addr.strip()] = int(val.strip())
        elif line.startswith('T'):
            match = re.match(r'T(\d+)\s*:\s*(.*)', line)
            if match:
                tid = int(match.group(1))
                while len(threads) <= tid:
                    threads.append([])
                instrs_str = match.group(2)
                instrs = _parse_instructions(instrs_str)
                threads[tid] = instrs
        elif line.startswith('forbidden:'):
            forb_str = line.split(':', 1)[1].strip()
            outcome = _parse_outcome(forb_str)
            if outcome:
                forbidden.append(outcome)

    return LitmusTest(name, threads, init_state, forbidden)


def _parse_instructions(text):
    """Parse instruction list from string like 'W(x)=1; r0=R(y)'."""
    instrs = []
    parts = [p.strip() for p in text.split(';')]
    for part in parts:
        if not part:
            continue
        # Store: W(x)=1
        m = re.match(r'W\((\w+)\)\s*=\s*(\d+)', part)
        if m:
            instrs.append(Store(m.group(1), int(m.group(2))))
            continue
        # Load: r0=R(x)
        m = re.match(r'(\w+)\s*=\s*R\((\w+)\)', part)
        if m:
            instrs.append(Load(m.group(2), m.group(1)))
            continue
        # Fence
        m = re.match(r'FENCE(?:\((\w+)\))?', part, re.IGNORECASE)
        if m:
            ftype = m.group(1) if m.group(1) else 'full'
            instrs.append(FenceInst(ftype))
            continue
    return instrs


def _parse_outcome(text):
    """Parse outcome from string like 'r0=0, r1=0'."""
    vals = {}
    parts = [p.strip() for p in text.split(',')]
    for part in parts:
        if '=' in part:
            reg, val = part.split('=')
            vals[reg.strip()] = int(val.strip())
    return Outcome(vals) if vals else None


# ---------------------------------------------------------------------------
# Litmus test minimization
# ---------------------------------------------------------------------------

class TestMinimizer:
    """Remove unnecessary instructions from litmus tests."""

    def __init__(self, test, model=None):
        self.test = test
        self.model = model or SequentialConsistency()

    def minimize(self):
        """Remove instructions that don't affect the distinguishing outcome."""
        if not self.test.forbidden_outcomes:
            return self.test

        current = copy.deepcopy(self.test)
        changed = True
        while changed:
            changed = False
            for tid in range(len(current.threads)):
                for idx in range(len(current.threads[tid]) - 1, -1, -1):
                    instr = current.threads[tid][idx]
                    # Don't remove loads that appear in forbidden outcome registers
                    if instr.register:
                        in_forbidden = any(
                            instr.register in fo.values
                            for fo in current.forbidden_outcomes
                        )
                        if in_forbidden:
                            continue

                    trial = copy.deepcopy(current)
                    trial.threads[tid].pop(idx)

                    if len(trial.threads[tid]) == 0:
                        continue

                    if self._still_distinguishes(trial):
                        current = trial
                        changed = True
                        break
                if changed:
                    break

        # Remove empty threads
        current.threads = [t for t in current.threads if len(t) > 0]
        return current

    def _still_distinguishes(self, test):
        """Check if the test still has the forbidden outcome as SC-forbidden."""
        enumerator = OutcomeEnumerator(test)
        all_out = enumerator.generate_all_outcomes()
        for fo in test.forbidden_outcomes:
            found_match = False
            for out in all_out:
                if all(out.get(k) == v for k, v in fo.values.items()):
                    found_match = True
                    break
            if not found_match:
                return False
        return True


def minimize_test(test, model=None):
    """Convenience: minimize a litmus test."""
    return TestMinimizer(test, model).minimize()


# ---------------------------------------------------------------------------
# Litmus test classification
# ---------------------------------------------------------------------------

class TestClassifier:
    """Classify which memory model relaxations a litmus test exposes."""

    MODELS = {
        'SC': SequentialConsistency,
        'TSO': TotalStoreOrder,
        'PSO': PartialStoreOrder,
        'Relaxed': RelaxedMemoryModel,
    }

    def classify(self, test):
        """Return dict of model_name -> 'allowed'/'forbidden' for the
        test's forbidden outcome under each model."""
        if not test.forbidden_outcomes:
            return {}

        results = {}
        fo = test.forbidden_outcomes[0]

        for name, model_cls in self.MODELS.items():
            model = model_cls()
            result = check_outcome(test, fo, model)
            results[name] = result

        return results

    def relaxations_exposed(self, test):
        """Return list of relaxation names this test exposes."""
        classification = self.classify(test)
        relaxations = []

        if classification.get('SC') == 'forbidden' and classification.get('TSO') == 'allowed':
            relaxations.append('store-load reordering')
        if classification.get('TSO') == 'forbidden' and classification.get('PSO') == 'allowed':
            relaxations.append('store-store reordering (diff addr)')
        if classification.get('SC') == 'forbidden' and classification.get('Relaxed') == 'allowed':
            relaxations.append('load-load reordering')
            relaxations.append('load-store reordering')
        if not relaxations and classification.get('SC') == 'forbidden':
            relaxations.append('general relaxation')

        return relaxations


def classify_test(test):
    """Convenience: classify a litmus test."""
    return TestClassifier().classify(test)


# ---------------------------------------------------------------------------
# Random litmus test generation
# ---------------------------------------------------------------------------

class RandomTestGenerator:
    """Generate random litmus tests with configurable parameters."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def generate(self, n_threads=2, n_instructions=3, n_addresses=2,
                 max_value=2, include_fences=False):
        """Generate a random litmus test."""
        addrs = [chr(ord('x') + i) for i in range(n_addresses)]
        reg_counter = 0
        threads = []
        all_regs = []

        for tid in range(n_threads):
            thread = []
            for _ in range(n_instructions):
                addr = addrs[self.rng.randint(0, n_addresses)]
                op = self.rng.choice(['store', 'load', 'fence'] if include_fences
                                     else ['store', 'load'])
                if op == 'store':
                    val = self.rng.randint(1, max_value + 1)
                    thread.append(Store(addr, int(val)))
                elif op == 'load':
                    reg = f"r{reg_counter}"
                    reg_counter += 1
                    thread.append(Load(addr, reg))
                    all_regs.append(reg)
                elif op == 'fence':
                    thread.append(FenceInst())
            threads.append(thread)

        init_state = {a: 0 for a in addrs}
        name = f"rand_{n_threads}t_{n_instructions}i_{self.rng.randint(0, 10000)}"
        test = LitmusTest(name, threads, init_state)
        return test

    def generate_targeted(self, relaxation='store-load', n_threads=2):
        """Generate a test targeting a specific relaxation."""
        if relaxation == 'store-load':
            return make_store_buffering()
        elif relaxation == 'message-passing':
            return make_message_passing()
        elif relaxation == 'load-load':
            return make_load_buffering()
        elif relaxation == 'iriw':
            return make_iriw()
        else:
            return self.generate(n_threads)

    def generate_batch(self, count=10, **kwargs):
        """Generate a batch of random litmus tests."""
        tests = []
        for _ in range(count):
            tests.append(self.generate(**kwargs))
        return tests


def generate_random_tests(count=10, seed=42, **kwargs):
    """Convenience: generate random litmus tests."""
    gen = RandomTestGenerator(seed)
    return gen.generate_batch(count, **kwargs)


# ---------------------------------------------------------------------------
# Litmus test comparison
# ---------------------------------------------------------------------------

class TestComparator:
    """Compare litmus tests for equivalence and subsumption."""

    def structurally_equivalent(self, t1, t2):
        """Check if two tests have the same structure up to renaming."""
        if len(t1.threads) != len(t2.threads):
            return False
        for i in range(len(t1.threads)):
            if len(t1.threads[i]) != len(t2.threads[i]):
                return False
            for j in range(len(t1.threads[i])):
                if t1.threads[i][j].op != t2.threads[i][j].op:
                    return False
        return True

    def outcome_equivalent(self, t1, t2, model):
        """Check if two tests have the same set of allowed outcomes."""
        e1 = OutcomeEnumerator(t1)
        e2 = OutcomeEnumerator(t2)
        o1 = e1.generate_all_outcomes()
        o2 = e2.generate_all_outcomes()
        return o1 == o2

    def subsumes(self, t1, t2, model):
        """Check if t1's outcomes are a superset of t2's outcomes."""
        e1 = OutcomeEnumerator(t1)
        e2 = OutcomeEnumerator(t2)
        o1 = e1.generate_all_outcomes()
        o2 = e2.generate_all_outcomes()
        return o2.issubset(o1)


# ---------------------------------------------------------------------------
# Litmus test suite
# ---------------------------------------------------------------------------

class LitmusTestSuite:
    """Collection of litmus tests with batch operations."""

    def __init__(self, name="suite"):
        self.name = name
        self.tests = []

    def add(self, test):
        self.tests.append(test)

    def add_builtins(self):
        for name, fn in BUILTIN_TESTS.items():
            self.tests.append(fn())

    def run_all(self, model):
        """Run all tests against a model. Returns dict of name -> results."""
        results = {}
        for test in self.tests:
            enumerator = OutcomeEnumerator(test)
            all_outcomes = enumerator.generate_all_outcomes()
            test_result = {
                'name': test.name,
                'total_outcomes': len(all_outcomes),
                'forbidden_found': [],
                'outcomes': list(all_outcomes),
            }
            for fo in test.forbidden_outcomes:
                for out in all_outcomes:
                    if all(out.get(k) == v for k, v in fo.values.items()):
                        test_result['forbidden_found'].append(fo)
                        break
            results[test.name] = test_result
        return results

    def summary(self, model):
        """Print summary of test results."""
        results = self.run_all(model)
        lines = [f"Suite: {self.name}, Model: {model.name}"]
        for name, r in results.items():
            n_forb = len(r['forbidden_found'])
            lines.append(f"  {name}: {r['total_outcomes']} outcomes, "
                         f"{n_forb} forbidden found")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Litmus test mutation
# ---------------------------------------------------------------------------

class TestMutator:
    """Mutate litmus tests for fuzzing."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def swap_instructions(self, test):
        """Swap two adjacent instructions in a random thread."""
        test = copy.deepcopy(test)
        tid = self.rng.randint(0, len(test.threads))
        thread = test.threads[tid]
        if len(thread) < 2:
            return test
        idx = self.rng.randint(0, len(thread) - 1)
        thread[idx], thread[idx + 1] = thread[idx + 1], thread[idx]
        test.name = test.name + "_swapped"
        return test

    def add_fence(self, test):
        """Add a fence at a random position."""
        test = copy.deepcopy(test)
        tid = self.rng.randint(0, len(test.threads))
        idx = self.rng.randint(0, len(test.threads[tid]) + 1)
        test.threads[tid].insert(idx, FenceInst())
        test.name = test.name + "_fenced"
        return test

    def remove_instruction(self, test):
        """Remove a random instruction."""
        test = copy.deepcopy(test)
        tid = self.rng.randint(0, len(test.threads))
        if len(test.threads[tid]) > 1:
            idx = self.rng.randint(0, len(test.threads[tid]))
            test.threads[tid].pop(idx)
            test.name = test.name + "_reduced"
        return test

    def change_address(self, test, n_addresses=2):
        """Change the address of a random instruction."""
        test = copy.deepcopy(test)
        addrs = list(test.all_addresses())
        if not addrs:
            return test
        tid = self.rng.randint(0, len(test.threads))
        instrs_with_addr = [(i, ins) for i, ins in enumerate(test.threads[tid])
                            if ins.address is not None]
        if instrs_with_addr:
            idx, instr = instrs_with_addr[self.rng.randint(0, len(instrs_with_addr))]
            new_addr = addrs[self.rng.randint(0, len(addrs))]
            test.threads[tid][idx].address = new_addr
            test.name = test.name + "_readdr"
        return test


# ---------------------------------------------------------------------------
# Exports & main
# ---------------------------------------------------------------------------

def get_standard_litmus_tests():
    """Return list of all standard litmus tests."""
    return [fn() for fn in BUILTIN_TESTS.values()]


if __name__ == "__main__":
    # Self-test
    sb = make_store_buffering()
    mp = make_message_passing()

    print(sb)
    print()
    print(mp)

    outcomes = generate_all_outcomes(sb)
    print(f"\nSB outcomes ({len(outcomes)}):")
    for o in sorted(outcomes, key=lambda x: str(x)):
        print(f"  {o}")

    # Verify forbidden outcome exists in enumeration
    fo = sb.forbidden_outcomes[0]
    found = any(all(o.get(k) == v for k, v in fo.values.items()) for o in outcomes)
    print(f"\nForbidden outcome {fo} found in enumeration: {found}")
    assert found, "SB forbidden outcome should be reachable via interleaving"

    # Parse test from string
    text = """
    name: TestParse
    init: x=0; y=0
    T0: W(x)=1; r0=R(y)
    T1: W(y)=1; r1=R(x)
    forbidden: r0=0, r1=0
    """
    parsed = litmus_test_from_string(text)
    print(f"\nParsed: {parsed}")
    assert parsed.name == "TestParse"
    assert len(parsed.threads) == 2

    print("\nlitmus_test_engine.py self-test passed")
