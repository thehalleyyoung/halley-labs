"""
Systematic weak memory testing.

Provides exhaustive outcome enumeration, random testing, stress testing,
litmus test extraction, outcome classification, model fitness, and
hardware comparison for concurrent programs under weak memory models.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from enum import Enum
import itertools
import copy
import time


class MemoryModelType(Enum):
    SC = "SC"
    TSO = "TSO"
    PSO = "PSO"
    ARM = "ARM"
    POWER = "POWER"
    RISCV = "RISCV"


class InstructionType(Enum):
    STORE = "store"
    LOAD = "load"
    FENCE = "fence"
    CAS = "cas"
    FETCH_ADD = "fetch_add"
    NOP = "nop"


@dataclass
class MemInstruction:
    thread_id: int
    index: int
    inst_type: InstructionType
    variable: str = ""
    value: Any = None
    register: str = ""
    fence_type: str = ""

    def __repr__(self):
        if self.inst_type == InstructionType.STORE:
            return f"T{self.thread_id}: {self.variable} = {self.value}"
        elif self.inst_type == InstructionType.LOAD:
            return f"T{self.thread_id}: {self.register} = {self.variable}"
        elif self.inst_type == InstructionType.FENCE:
            return f"T{self.thread_id}: fence({self.fence_type})"
        return f"T{self.thread_id}: {self.inst_type.value}"


@dataclass
class LitmusTest:
    name: str
    threads: Dict[int, List[MemInstruction]]
    initial_state: Dict[str, int]
    expected_outcomes: Optional[Dict[str, Set[Tuple]]] = None
    description: str = ""

    @property
    def n_threads(self):
        return len(self.threads)

    @property
    def variables(self) -> Set[str]:
        vars_set = set(self.initial_state.keys())
        for tid, insts in self.threads.items():
            for inst in insts:
                if inst.variable:
                    vars_set.add(inst.variable)
        return vars_set


@dataclass
class Outcome:
    registers: Dict[str, int]
    final_memory: Dict[str, int]

    def __hash__(self):
        return hash((
            tuple(sorted(self.registers.items())),
            tuple(sorted(self.final_memory.items()))
        ))

    def __eq__(self, other):
        return (isinstance(other, Outcome) and
                self.registers == other.registers and
                self.final_memory == other.final_memory)

    def __repr__(self):
        reg_str = ", ".join(f"{k}={v}" for k, v in sorted(self.registers.items()))
        mem_str = ", ".join(f"{k}={v}" for k, v in sorted(self.final_memory.items()))
        return f"Outcome(regs=[{reg_str}], mem=[{mem_str}])"

    def as_tuple(self) -> Tuple:
        return tuple(v for _, v in sorted(self.registers.items()))


@dataclass
class TestReport:
    test_name: str
    outcomes_observed: Dict[Outcome, int]
    frequency_distribution: Dict[str, float]
    model_consistency: Dict[str, bool]
    anomalies: List[str]
    total_runs: int = 0
    unique_outcomes: int = 0
    elapsed_time: float = 0.0
    classification: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "test": self.test_name,
            "total_runs": self.total_runs,
            "unique_outcomes": self.unique_outcomes,
            "anomalies": len(self.anomalies),
            "model_consistency": self.model_consistency,
            "elapsed_time": f"{self.elapsed_time:.3f}s",
        }


class SequentialConsistencyChecker:
    """Check if an outcome is allowed under sequential consistency."""

    def check(self, test: LitmusTest, outcome: Outcome) -> bool:
        all_instructions = []
        for tid in sorted(test.threads.keys()):
            for inst in test.threads[tid]:
                all_instructions.append(inst)

        n = len(all_instructions)
        if n > 12:
            return self._approximate_check(test, outcome)

        # Try all interleavings that preserve per-thread program order
        thread_indices = {tid: 0 for tid in test.threads}
        return self._try_interleavings(test, outcome, all_instructions, thread_indices)

    def _try_interleavings(self, test: LitmusTest, outcome: Outcome,
                           all_instructions: List[MemInstruction],
                           thread_indices: Dict[int, int]) -> bool:
        thread_orders = []
        for tid in sorted(test.threads.keys()):
            thread_orders.append(list(range(len(test.threads[tid]))))

        # Generate all valid interleavings using DFS
        return self._dfs_interleave(test, outcome, {tid: 0 for tid in test.threads},
                                    dict(test.initial_state), {})

    def _dfs_interleave(self, test: LitmusTest, outcome: Outcome,
                        pcs: Dict[int, int], memory: Dict[str, int],
                        registers: Dict[str, int]) -> bool:
        all_done = all(pcs[tid] >= len(test.threads[tid]) for tid in test.threads)
        if all_done:
            # Check if this interleaving produces the target outcome
            return (registers == outcome.registers and
                    {k: memory.get(k, 0) for k in outcome.final_memory} == outcome.final_memory)

        for tid in sorted(test.threads.keys()):
            if pcs[tid] >= len(test.threads[tid]):
                continue

            inst = test.threads[tid][pcs[tid]]
            saved_mem = dict(memory)
            saved_regs = dict(registers)

            if inst.inst_type == InstructionType.STORE:
                memory[inst.variable] = inst.value
            elif inst.inst_type == InstructionType.LOAD:
                registers[inst.register] = memory.get(inst.variable, 0)
            elif inst.inst_type == InstructionType.FENCE:
                pass  # fences don't change state under SC

            pcs[tid] += 1
            if self._dfs_interleave(test, outcome, pcs, memory, registers):
                return True
            pcs[tid] -= 1

            memory.clear()
            memory.update(saved_mem)
            registers.clear()
            registers.update(saved_regs)

        return False

    def _approximate_check(self, test: LitmusTest, outcome: Outcome) -> bool:
        # For large programs, use sampling
        rng = np.random.RandomState(42)
        for _ in range(1000):
            memory = dict(test.initial_state)
            registers = {}
            pcs = {tid: 0 for tid in test.threads}

            while not all(pcs[tid] >= len(test.threads[tid]) for tid in test.threads):
                eligible = [tid for tid in test.threads if pcs[tid] < len(test.threads[tid])]
                if not eligible:
                    break
                tid = eligible[int(rng.randint(0, len(eligible)))]
                inst = test.threads[tid][pcs[tid]]

                if inst.inst_type == InstructionType.STORE:
                    memory[inst.variable] = inst.value
                elif inst.inst_type == InstructionType.LOAD:
                    registers[inst.register] = memory.get(inst.variable, 0)

                pcs[tid] += 1

            if (registers == outcome.registers and
                    {k: memory.get(k, 0) for k in outcome.final_memory} == outcome.final_memory):
                return True
        return False


class TSOChecker:
    """Check if an outcome is allowed under TSO (x86)."""

    def check(self, test: LitmusTest, outcome: Outcome) -> bool:
        return self._explore_tso(test, outcome)

    def _explore_tso(self, test: LitmusTest, outcome: Outcome) -> bool:
        initial_buffers = {tid: [] for tid in test.threads}
        initial_pcs = {tid: 0 for tid in test.threads}
        initial_mem = dict(test.initial_state)

        visited = set()
        stack = [(initial_pcs, initial_mem, {}, initial_buffers)]

        while stack:
            pcs, memory, registers, buffers = stack.pop()

            state_key = (
                tuple(sorted(pcs.items())),
                tuple(sorted(memory.items())),
                tuple(sorted(registers.items())),
                tuple(tuple(b) for b in [buffers[t] for t in sorted(buffers.keys())])
            )
            if state_key in visited:
                continue
            visited.add(state_key)

            if len(visited) > 50000:
                break

            all_done = all(pcs[tid] >= len(test.threads[tid]) for tid in test.threads)
            all_flushed = all(len(buffers[tid]) == 0 for tid in test.threads)

            if all_done and all_flushed:
                if (registers == outcome.registers and
                        {k: memory.get(k, 0) for k in outcome.final_memory} == outcome.final_memory):
                    return True
                continue

            for tid in sorted(test.threads.keys()):
                # Flush store buffer entry
                if buffers[tid]:
                    new_buffers = {t: list(b) for t, b in buffers.items()}
                    var, val = new_buffers[tid].pop(0)
                    new_mem = dict(memory)
                    new_mem[var] = val
                    stack.append((dict(pcs), new_mem, dict(registers), new_buffers))

                if pcs[tid] >= len(test.threads[tid]):
                    continue

                inst = test.threads[tid][pcs[tid]]
                new_pcs = dict(pcs)
                new_mem = dict(memory)
                new_regs = dict(registers)
                new_buffers = {t: list(b) for t, b in buffers.items()}

                if inst.inst_type == InstructionType.STORE:
                    new_buffers[tid].append((inst.variable, inst.value))
                    new_pcs[tid] += 1
                    stack.append((new_pcs, new_mem, new_regs, new_buffers))

                elif inst.inst_type == InstructionType.LOAD:
                    # Check store buffer first (store forwarding)
                    val = None
                    for var, v in reversed(new_buffers[tid]):
                        if var == inst.variable:
                            val = v
                            break
                    if val is None:
                        val = new_mem.get(inst.variable, 0)
                    new_regs[inst.register] = val
                    new_pcs[tid] += 1
                    stack.append((new_pcs, new_mem, new_regs, new_buffers))

                elif inst.inst_type == InstructionType.FENCE:
                    # Fence flushes store buffer
                    for var, val in new_buffers[tid]:
                        new_mem[var] = val
                    new_buffers[tid] = []
                    new_pcs[tid] += 1
                    stack.append((new_pcs, new_mem, new_regs, new_buffers))

        return False


class ExhaustiveEnumerator:
    """Enumerate all possible outcomes for a litmus test."""

    def __init__(self, model: str = "SC"):
        self.model = model
        self.sc_checker = SequentialConsistencyChecker()
        self.tso_checker = TSOChecker()

    def enumerate(self, test: LitmusTest) -> Set[Outcome]:
        if self.model == "TSO":
            return self._enumerate_tso(test)
        return self._enumerate_sc(test)

    def _enumerate_sc(self, test: LitmusTest) -> Set[Outcome]:
        outcomes = set()
        pcs = {tid: 0 for tid in test.threads}
        self._sc_dfs(test, pcs, dict(test.initial_state), {}, outcomes)
        return outcomes

    def _sc_dfs(self, test: LitmusTest, pcs: Dict[int, int],
                memory: Dict[str, int], registers: Dict[str, int],
                outcomes: Set[Outcome]):
        all_done = all(pcs[tid] >= len(test.threads[tid]) for tid in test.threads)
        if all_done:
            outcome = Outcome(
                registers=dict(registers),
                final_memory={k: memory.get(k, 0) for k in test.initial_state}
            )
            outcomes.add(outcome)
            return

        if len(outcomes) > 10000:
            return

        for tid in sorted(test.threads.keys()):
            if pcs[tid] >= len(test.threads[tid]):
                continue
            inst = test.threads[tid][pcs[tid]]

            saved_mem = dict(memory)
            saved_regs = dict(registers)

            if inst.inst_type == InstructionType.STORE:
                memory[inst.variable] = inst.value
            elif inst.inst_type == InstructionType.LOAD:
                registers[inst.register] = memory.get(inst.variable, 0)

            pcs[tid] += 1
            self._sc_dfs(test, pcs, memory, registers, outcomes)
            pcs[tid] -= 1

            memory.clear()
            memory.update(saved_mem)
            registers.clear()
            registers.update(saved_regs)

    def _enumerate_tso(self, test: LitmusTest) -> Set[Outcome]:
        outcomes = set()
        initial_buffers = {tid: [] for tid in test.threads}
        pcs = {tid: 0 for tid in test.threads}

        visited = set()
        stack = [(dict(pcs), dict(test.initial_state), {}, {t: [] for t in test.threads})]

        while stack:
            pcs_s, mem, regs, bufs = stack.pop()
            state_key = (
                tuple(sorted(pcs_s.items())),
                tuple(sorted(mem.items())),
                tuple(sorted(regs.items())),
                tuple(tuple(b) for b in [bufs[t] for t in sorted(bufs.keys())])
            )
            if state_key in visited:
                continue
            visited.add(state_key)

            if len(visited) > 100000:
                break

            all_done = all(pcs_s[t] >= len(test.threads[t]) for t in test.threads)
            all_flushed = all(len(bufs[t]) == 0 for t in test.threads)

            if all_done and all_flushed:
                outcomes.add(Outcome(
                    registers=dict(regs),
                    final_memory={k: mem.get(k, 0) for k in test.initial_state}
                ))
                continue

            for tid in sorted(test.threads.keys()):
                if bufs[tid]:
                    nb = {t: list(b) for t, b in bufs.items()}
                    var, val = nb[tid].pop(0)
                    nm = dict(mem)
                    nm[var] = val
                    stack.append((dict(pcs_s), nm, dict(regs), nb))

                if pcs_s[tid] >= len(test.threads[tid]):
                    continue
                inst = test.threads[tid][pcs_s[tid]]
                np_ = dict(pcs_s)
                nm = dict(mem)
                nr = dict(regs)
                nb = {t: list(b) for t, b in bufs.items()}

                if inst.inst_type == InstructionType.STORE:
                    nb[tid].append((inst.variable, inst.value))
                    np_[tid] += 1
                    stack.append((np_, nm, nr, nb))
                elif inst.inst_type == InstructionType.LOAD:
                    val = None
                    for var, v in reversed(nb[tid]):
                        if var == inst.variable:
                            val = v
                            break
                    if val is None:
                        val = nm.get(inst.variable, 0)
                    nr[inst.register] = val
                    np_[tid] += 1
                    stack.append((np_, nm, nr, nb))
                elif inst.inst_type == InstructionType.FENCE:
                    for var, val in nb[tid]:
                        nm[var] = val
                    nb[tid] = []
                    np_[tid] += 1
                    stack.append((np_, nm, nr, nb))

        return outcomes


class RandomTester:
    """Random testing with random thread interleavings."""

    def __init__(self, rng_seed: int = 42):
        self.rng = np.random.RandomState(rng_seed)

    def test(self, test_case: LitmusTest, n_runs: int = 10000,
             model: str = "SC") -> Dict[Outcome, int]:
        outcome_counts: Dict[Outcome, int] = {}

        for _ in range(n_runs):
            outcome = self._single_run(test_case, model)
            if outcome in outcome_counts:
                outcome_counts[outcome] += 1
            else:
                outcome_counts[outcome] = 1

        return outcome_counts

    def _single_run(self, test: LitmusTest, model: str) -> Outcome:
        memory = dict(test.initial_state)
        registers = {}
        pcs = {tid: 0 for tid in test.threads}
        buffers = {tid: [] for tid in test.threads} if model == "TSO" else None

        max_steps = sum(len(insts) for insts in test.threads.values()) * 3
        steps = 0

        while steps < max_steps:
            eligible = [tid for tid in test.threads if pcs[tid] < len(test.threads[tid])]

            if model == "TSO":
                for tid in test.threads:
                    if buffers[tid]:
                        eligible.append((-1, tid))  # flush option

            if not eligible and (model != "TSO" or all(
                    len(buffers[t]) == 0 for t in test.threads)):
                break

            flat_eligible = []
            for item in eligible:
                if isinstance(item, tuple):
                    flat_eligible.append(item)
                else:
                    flat_eligible.append(("exec", item))

            if not flat_eligible:
                break

            choice = flat_eligible[int(self.rng.randint(0, len(flat_eligible)))]

            if isinstance(choice, tuple) and choice[0] == -1:
                tid = choice[1]
                if buffers[tid]:
                    var, val = buffers[tid].pop(0)
                    memory[var] = val
            elif isinstance(choice, tuple):
                tid = choice[1]
                self._execute_inst(test.threads[tid][pcs[tid]], memory, registers, pcs, buffers, model)
            else:
                tid = choice
                self._execute_inst(test.threads[tid][pcs[tid]], memory, registers, pcs, buffers, model)

            steps += 1

        # Flush remaining buffers
        if model == "TSO" and buffers:
            for tid in test.threads:
                for var, val in buffers[tid]:
                    memory[var] = val

        return Outcome(
            registers=dict(registers),
            final_memory={k: memory.get(k, 0) for k in test.initial_state}
        )

    def _execute_inst(self, inst: MemInstruction, memory: Dict, registers: Dict,
                      pcs: Dict, buffers: Optional[Dict], model: str):
        tid = inst.thread_id
        if inst.inst_type == InstructionType.STORE:
            if model == "TSO" and buffers is not None:
                buffers[tid].append((inst.variable, inst.value))
            else:
                memory[inst.variable] = inst.value
        elif inst.inst_type == InstructionType.LOAD:
            if model == "TSO" and buffers is not None:
                val = None
                for var, v in reversed(buffers[tid]):
                    if var == inst.variable:
                        val = v
                        break
                if val is None:
                    val = memory.get(inst.variable, 0)
                registers[inst.register] = val
            else:
                registers[inst.register] = memory.get(inst.variable, 0)
        elif inst.inst_type == InstructionType.FENCE:
            if model == "TSO" and buffers is not None:
                for var, val in buffers[tid]:
                    memory[var] = val
                buffers[tid] = []
        pcs[tid] += 1


class StressTester:
    """High-contention stress testing to expose weak memory behaviors."""

    def __init__(self, rng_seed: int = 42):
        self.rng = np.random.RandomState(rng_seed)

    def stress_test(self, test_case: LitmusTest, n_runs: int = 50000,
                    model: str = "TSO") -> Dict[Outcome, int]:
        random_tester = RandomTester(rng_seed=int(self.rng.randint(0, 2**31)))
        return random_tester.test(test_case, n_runs, model)


class OutcomeClassifier:
    """Classify outcomes by which memory models allow them."""

    def __init__(self):
        self.sc_enumerator = ExhaustiveEnumerator("SC")
        self.tso_enumerator = ExhaustiveEnumerator("TSO")

    def classify(self, test: LitmusTest,
                 outcomes: Set[Outcome]) -> Dict[Outcome, Dict[str, bool]]:
        sc_outcomes = self.sc_enumerator.enumerate(test)
        tso_outcomes = self.tso_enumerator.enumerate(test)

        classification = {}
        for outcome in outcomes:
            classification[outcome] = {
                "SC": outcome in sc_outcomes,
                "TSO": outcome in tso_outcomes,
                "ARM": True,  # ARM allows all SC/TSO outcomes and more
                "POWER": True,
            }
        return classification


class ModelFitness:
    """Evaluate how well observed outcomes match model predictions."""

    def evaluate(self, test: LitmusTest, observed: Dict[Outcome, int],
                 model_name: str) -> Dict[str, float]:
        if model_name == "SC":
            enumerator = ExhaustiveEnumerator("SC")
        else:
            enumerator = ExhaustiveEnumerator("TSO")

        predicted = enumerator.enumerate(test)
        observed_set = set(observed.keys())

        true_positives = len(observed_set & predicted)
        false_positives = len(observed_set - predicted)
        false_negatives = len(predicted - observed_set)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        total_runs = sum(observed.values())
        forbidden_fraction = sum(
            count for outcome, count in observed.items()
            if outcome not in predicted
        ) / total_runs if total_runs > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "forbidden_fraction": forbidden_fraction,
            "predicted_outcomes": len(predicted),
            "observed_outcomes": len(observed_set),
            "consistent": false_positives == 0,
        }


class HardwareComparator:
    """Compare observed behaviors across simulated architectures."""

    def compare(self, test: LitmusTest, models: List[str],
                n_runs: int = 10000) -> Dict[str, Dict[str, Any]]:
        results = {}
        for model in models:
            random_tester = RandomTester(rng_seed=42)
            outcomes = random_tester.test(test, n_runs, model)

            total = sum(outcomes.values())
            freq = {}
            for outcome, count in outcomes.items():
                freq[str(outcome.as_tuple())] = count / total

            results[model] = {
                "unique_outcomes": len(outcomes),
                "total_runs": total,
                "frequency": freq,
                "outcomes": outcomes,
            }

        return results


class LitmusTestExtractor:
    """Extract relevant litmus patterns from a program."""

    def extract(self, threads: Dict[int, List[MemInstruction]],
                variables: Set[str]) -> List[LitmusTest]:
        tests = []

        # Extract pairwise patterns
        thread_ids = sorted(threads.keys())
        for i, t1 in enumerate(thread_ids):
            for t2 in thread_ids[i + 1:]:
                shared = self._shared_variables(threads[t1], threads[t2])
                if shared:
                    test = self._extract_pair(t1, threads[t1], t2, threads[t2], shared)
                    if test:
                        tests.append(test)

        return tests

    def _shared_variables(self, thread1: List[MemInstruction],
                          thread2: List[MemInstruction]) -> Set[str]:
        vars1 = {inst.variable for inst in thread1 if inst.variable}
        vars2 = {inst.variable for inst in thread2 if inst.variable}
        return vars1 & vars2

    def _extract_pair(self, t1: int, insts1: List[MemInstruction],
                      t2: int, insts2: List[MemInstruction],
                      shared: Set[str]) -> Optional[LitmusTest]:
        relevant1 = [i for i in insts1 if i.variable in shared]
        relevant2 = [i for i in insts2 if i.variable in shared]

        if not relevant1 or not relevant2:
            return None

        initial = {var: 0 for var in shared}
        return LitmusTest(
            name=f"extracted_T{t1}_T{t2}",
            threads={t1: relevant1, t2: relevant2},
            initial_state=initial,
        )


class WeakMemoryTester:
    """Main tester: orchestrates all testing strategies."""

    def __init__(self, rng_seed: int = 42):
        self.rng_seed = rng_seed
        self.random_tester = RandomTester(rng_seed)
        self.stress_tester = StressTester(rng_seed)
        self.enumerator_sc = ExhaustiveEnumerator("SC")
        self.enumerator_tso = ExhaustiveEnumerator("TSO")
        self.classifier = OutcomeClassifier()
        self.fitness = ModelFitness()
        self.hw_comparator = HardwareComparator()
        self.extractor = LitmusTestExtractor()

    def test(self, test_case: LitmusTest, model: str = "SC",
             n_runs: int = 10000) -> TestReport:
        start_time = time.time()

        # Random testing
        observed = self.random_tester.test(test_case, n_runs, model)

        # Frequency distribution
        total = sum(observed.values())
        freq = {}
        for outcome, count in observed.items():
            key = str(outcome.as_tuple())
            freq[key] = count / total

        # Model consistency
        sc_outcomes = self.enumerator_sc.enumerate(test_case)
        tso_outcomes = self.enumerator_tso.enumerate(test_case)
        observed_set = set(observed.keys())

        model_consistency = {
            "SC": observed_set.issubset(sc_outcomes),
            "TSO": observed_set.issubset(tso_outcomes),
        }

        # Anomalies
        anomalies = []
        non_sc = observed_set - sc_outcomes
        if non_sc:
            anomalies.append(f"{len(non_sc)} non-SC outcomes observed")
        if model == "TSO":
            non_tso = observed_set - tso_outcomes
            if non_tso:
                anomalies.append(f"{len(non_tso)} non-TSO outcomes observed")

        # Classification
        classification = {}
        for outcome in observed_set:
            classification[str(outcome.as_tuple())] = (
                "SC" if outcome in sc_outcomes else
                "TSO-only" if outcome in tso_outcomes else
                "relaxed"
            )

        elapsed = time.time() - start_time

        return TestReport(
            test_name=test_case.name,
            outcomes_observed=observed,
            frequency_distribution=freq,
            model_consistency=model_consistency,
            anomalies=anomalies,
            total_runs=total,
            unique_outcomes=len(observed),
            elapsed_time=elapsed,
            classification=classification,
        )

    def exhaustive_test(self, test_case: LitmusTest,
                        model: str = "SC") -> TestReport:
        start_time = time.time()
        if model == "TSO":
            outcomes = self.enumerator_tso.enumerate(test_case)
        else:
            outcomes = self.enumerator_sc.enumerate(test_case)

        observed = {o: 1 for o in outcomes}
        total = len(outcomes)
        freq = {str(o.as_tuple()): 1.0 / total for o in outcomes} if total > 0 else {}

        elapsed = time.time() - start_time
        return TestReport(
            test_name=test_case.name,
            outcomes_observed=observed,
            frequency_distribution=freq,
            model_consistency={model: True},
            anomalies=[],
            total_runs=total,
            unique_outcomes=len(outcomes),
            elapsed_time=elapsed,
        )


def build_sb_test() -> LitmusTest:
    """Store Buffer (SB) litmus test: x=y=0; T0: x=1; r0=y; T1: y=1; r1=x."""
    return LitmusTest(
        name="SB",
        threads={
            0: [
                MemInstruction(0, 0, InstructionType.STORE, "x", 1),
                MemInstruction(0, 1, InstructionType.LOAD, "y", register="r0"),
            ],
            1: [
                MemInstruction(1, 0, InstructionType.STORE, "y", 1),
                MemInstruction(1, 1, InstructionType.LOAD, "x", register="r1"),
            ],
        },
        initial_state={"x": 0, "y": 0},
        description="Store Buffer test: can r0=r1=0?",
    )


def build_mp_test() -> LitmusTest:
    """Message Passing (MP) litmus test."""
    return LitmusTest(
        name="MP",
        threads={
            0: [
                MemInstruction(0, 0, InstructionType.STORE, "data", 1),
                MemInstruction(0, 1, InstructionType.STORE, "flag", 1),
            ],
            1: [
                MemInstruction(1, 0, InstructionType.LOAD, "flag", register="r0"),
                MemInstruction(1, 1, InstructionType.LOAD, "data", register="r1"),
            ],
        },
        initial_state={"data": 0, "flag": 0},
        description="Message Passing: if r0=1, must r1=1?",
    )


def build_iriw_test() -> LitmusTest:
    """Independent Reads of Independent Writes (IRIW) litmus test."""
    return LitmusTest(
        name="IRIW",
        threads={
            0: [MemInstruction(0, 0, InstructionType.STORE, "x", 1)],
            1: [MemInstruction(1, 0, InstructionType.STORE, "y", 1)],
            2: [
                MemInstruction(2, 0, InstructionType.LOAD, "x", register="r0"),
                MemInstruction(2, 1, InstructionType.LOAD, "y", register="r1"),
            ],
            3: [
                MemInstruction(3, 0, InstructionType.LOAD, "y", register="r2"),
                MemInstruction(3, 1, InstructionType.LOAD, "x", register="r3"),
            ],
        },
        initial_state={"x": 0, "y": 0},
        description="IRIW: can T2 see x=1,y=0 and T3 see y=1,x=0?",
    )


def build_dekker_test() -> LitmusTest:
    """Dekker's algorithm litmus test."""
    return LitmusTest(
        name="Dekker",
        threads={
            0: [
                MemInstruction(0, 0, InstructionType.STORE, "flag0", 1),
                MemInstruction(0, 1, InstructionType.LOAD, "flag1", register="r0"),
            ],
            1: [
                MemInstruction(1, 0, InstructionType.STORE, "flag1", 1),
                MemInstruction(1, 1, InstructionType.LOAD, "flag0", register="r1"),
            ],
        },
        initial_state={"flag0": 0, "flag1": 0},
        description="Dekker: can both threads enter critical section?",
    )
