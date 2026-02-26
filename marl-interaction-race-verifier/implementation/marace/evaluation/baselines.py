"""Baseline runners and ablation study infrastructure for MARACE evaluation.

Provides alternative verification strategies (single-agent, brute-force,
enumeration) and ablation configurations to compare against the full
MARACE pipeline. Includes a comparator that produces summary tables
and LaTeX output for publication.
"""

from __future__ import annotations

import itertools
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Ablation configuration
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """Specifies which MARACE components to enable or disable.

    Each boolean flag corresponds to a major pipeline stage.  Setting a flag
    to ``False`` removes that component from the verification run, allowing
    measurement of its marginal contribution.

    Attributes:
        use_decomposition: Enable assume-guarantee decomposition.
        use_hb_pruning: Enable happens-before pruning of schedules.
        use_adversarial_search: Enable adversarial schedule search.
        use_importance_sampling: Enable importance-weighted sampling.
        use_abstract_interpretation: Enable abstract-interpretation reachability.
        max_schedule_depth: Maximum interleaving depth to explore.
        name: Human-readable label for this configuration.
    """

    use_decomposition: bool = True
    use_hb_pruning: bool = True
    use_adversarial_search: bool = True
    use_importance_sampling: bool = True
    use_abstract_interpretation: bool = True
    max_schedule_depth: int = 20
    name: str = ""

    def describe(self) -> str:
        """Return a human-readable summary of enabled/disabled components."""
        components = [
            ("decomposition", self.use_decomposition),
            ("hb_pruning", self.use_hb_pruning),
            ("adversarial_search", self.use_adversarial_search),
            ("importance_sampling", self.use_importance_sampling),
            ("abstract_interpretation", self.use_abstract_interpretation),
        ]
        enabled = [c for c, on in components if on]
        disabled = [c for c, on in components if not on]
        lines = []
        if self.name:
            lines.append(f"Configuration: {self.name}")
        lines.append(f"  max_schedule_depth = {self.max_schedule_depth}")
        lines.append(f"  enabled  : {', '.join(enabled) if enabled else '(none)'}")
        lines.append(f"  disabled : {', '.join(disabled) if disabled else '(none)'}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Abstract base class for all baseline runners
# ---------------------------------------------------------------------------

class BaselineRunner(ABC):
    """Abstract interface for alternative verification strategies.

    Every baseline must implement :meth:`run`, which accepts an environment
    configuration, a list of per-agent policy configurations, and a safety
    specification string, then returns a standardised result dictionary.
    """

    def __init__(self, config: dict) -> None:
        """Initialise the runner from a free-form configuration dict.

        Args:
            config: Runner-specific parameters (e.g. simulation count,
                depth bound).
        """
        self._config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this baseline method."""

    @abstractmethod
    def run(
        self,
        env_config: dict,
        policy_configs: list[dict],
        spec: str,
    ) -> dict:
        """Execute verification and return results.

        Args:
            env_config: Environment construction parameters.
            policy_configs: Per-agent policy parameters.
            spec: Safety specification in the MARACE spec language.

        Returns:
            Dictionary with at least the following keys:

            * ``races`` – list of detected race-condition dicts, each with
              keys ``agents``, ``state``, ``type``, and optionally
              ``probability``.
            * ``time_s`` – wall-clock seconds elapsed.
            * ``states_explored`` – number of distinct states visited.
        """

    def supports_certification(self) -> bool:
        """Whether this runner can produce formal correctness certificates."""
        return False


# ---------------------------------------------------------------------------
# Baseline 1 – Single-agent verifier
# ---------------------------------------------------------------------------

class SingleAgentVerifier(BaselineRunner):
    """Verify each agent in isolation, ignoring inter-agent interactions.

    This baseline decomposes the multi-agent system into independent
    single-agent verification problems.  It checks each agent's policy
    against the safety specification while treating the environment as
    deterministic (other agents' actions are absent).  By construction it
    cannot detect races that arise from *interaction* between agents, and
    therefore serves as a lower-bound on recall.
    """

    @property
    def name(self) -> str:
        return "single_agent"

    def run(
        self,
        env_config: dict,
        policy_configs: list[dict],
        spec: str,
    ) -> dict:
        """Verify each agent independently and aggregate results.

        For every agent *i* the method constructs a single-agent variant of
        the environment, explores states reachable under agent *i*'s policy
        alone, and checks whether the safety specification is violated.

        Returns:
            Result dict.  ``races`` contains per-agent violations only;
            interaction-based races will be missing.
        """
        start = time.perf_counter()
        races: list[dict] = []
        total_states = 0
        num_agents = len(policy_configs)
        max_depth: int = self._config.get("max_depth", 50)

        for agent_idx in range(num_agents):
            agent_states = 0
            visited: set[str] = set()
            # BFS over states reachable by this agent alone
            initial_state = self._make_initial_state(env_config, agent_idx)
            queue: list[dict] = [initial_state]

            while queue and agent_states < max_depth * 100:
                state = queue.pop(0)
                fp = self._state_fingerprint(state)
                if fp in visited:
                    continue
                visited.add(fp)
                agent_states += 1

                violation = self._check_spec(state, spec)
                if violation is not None:
                    races.append({
                        "agents": [agent_idx],
                        "state": state,
                        "type": "single_agent_violation",
                        "description": violation,
                    })

                successors = self._get_successors(
                    state, env_config, policy_configs[agent_idx], agent_idx,
                )
                queue.extend(successors)

            total_states += agent_states

        elapsed = time.perf_counter() - start
        return {
            "races": races,
            "time_s": elapsed,
            "states_explored": total_states,
            "method": self.name,
        }

    # -- internal helpers (stubs for integration with MARACE env layer) -----

    @staticmethod
    def _make_initial_state(env_config: dict, agent_idx: int) -> dict:
        """Construct the initial environment state for a single agent."""
        return {
            "env": env_config.get("initial_state", {}),
            "agent": agent_idx,
            "step": 0,
        }

    @staticmethod
    def _state_fingerprint(state: dict) -> str:
        """Deterministic, hashable fingerprint for duplicate detection."""
        return str(sorted(state.get("env", {}).items())) + str(state.get("step"))

    @staticmethod
    def _check_spec(state: dict, spec: str) -> str | None:
        """Evaluate the safety spec against *state*.

        Returns a violation description string, or ``None`` if the state
        is safe.
        """
        # Placeholder: concrete spec evaluation is delegated to
        # marace.spec at integration time.
        _ = spec
        return None

    @staticmethod
    def _get_successors(
        state: dict,
        env_config: dict,
        policy_config: dict,
        agent_idx: int,
    ) -> list[dict]:
        """Generate successor states for a single agent step.

        In a full integration this calls ``MultiAgentEnv.step_sync`` with
        only the designated agent acting.
        """
        next_step = state.get("step", 0) + 1
        max_steps = env_config.get("max_steps", 50)
        if next_step >= max_steps:
            return []
        return [{
            "env": dict(state.get("env", {})),
            "agent": agent_idx,
            "step": next_step,
        }]


# ---------------------------------------------------------------------------
# Baseline 2 – Brute-force Monte Carlo simulator
# ---------------------------------------------------------------------------

class BruteForceSimulator(BaselineRunner):
    """Brute-force Monte Carlo simulation of random schedules.

    At each simulation step, a uniformly random agent is selected to act
    and the resulting state is checked against the safety specification.
    Violations are recorded with an empirical probability estimate
    computed as *detections / simulations*.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.num_simulations: int = config.get("num_simulations", 10_000)

    @property
    def name(self) -> str:
        return "brute_force"

    def run(
        self,
        env_config: dict,
        policy_configs: list[dict],
        spec: str,
    ) -> dict:
        """Run Monte Carlo simulations and report empirical race rates.

        Returns:
            Result dict.  Each entry in ``races`` includes a
            ``probability`` field estimated over all simulations.
        """
        start = time.perf_counter()
        num_agents = len(policy_configs)
        max_depth: int = self._config.get("max_depth", 30)
        seed: int = self._config.get("seed", 42)
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        violation_counts: dict[str, int] = {}
        violation_examples: dict[str, dict] = {}
        total_states = 0

        for sim_idx in range(self.num_simulations):
            state = dict(env_config.get("initial_state", {}))
            schedule: list[int] = []

            for depth in range(max_depth):
                agent_idx = rng.randint(0, num_agents - 1)
                schedule.append(agent_idx)
                state = self._step(state, env_config, policy_configs, agent_idx, np_rng)
                total_states += 1

                violation = self._check_spec(state, spec)
                if violation is not None:
                    key = self._violation_key(violation, frozenset(
                        self._involved_agents(schedule, violation),
                    ))
                    violation_counts[key] = violation_counts.get(key, 0) + 1
                    if key not in violation_examples:
                        violation_examples[key] = {
                            "state": dict(state),
                            "schedule": list(schedule),
                            "description": violation,
                            "agents": self._involved_agents(schedule, violation),
                        }
                    break  # move to next simulation on first violation

        elapsed = time.perf_counter() - start

        races: list[dict] = []
        for key, count in violation_counts.items():
            example = violation_examples[key]
            races.append({
                "agents": example["agents"],
                "state": example["state"],
                "type": "simulated_violation",
                "description": example["description"],
                "schedule": example["schedule"],
                "probability": count / self.num_simulations,
                "occurrences": count,
            })

        return {
            "races": races,
            "time_s": elapsed,
            "states_explored": total_states,
            "num_simulations": self.num_simulations,
            "method": self.name,
        }

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _step(
        state: dict,
        env_config: dict,
        policy_configs: list[dict],
        agent_idx: int,
        rng: np.random.RandomState,
    ) -> dict:
        """Advance the environment by one agent action.

        Placeholder: in integration mode this delegates to the MARACE
        environment layer with stochastic action selection.
        """
        new_state = dict(state)
        new_state["_last_agent"] = agent_idx
        new_state["_step"] = state.get("_step", 0) + 1
        return new_state

    @staticmethod
    def _check_spec(state: dict, spec: str) -> str | None:
        """Evaluate the safety specification.  See SingleAgentVerifier."""
        _ = spec
        return None

    @staticmethod
    def _violation_key(description: str, agents: frozenset) -> str:
        return f"{description}|{sorted(agents)}"

    @staticmethod
    def _involved_agents(schedule: list[int], violation: str) -> list[int]:
        """Heuristic: return unique agents in the recent schedule window."""
        window = schedule[-5:] if len(schedule) >= 5 else schedule
        return sorted(set(window))


# ---------------------------------------------------------------------------
# Baseline 3 – Naive schedule enumerator
# ---------------------------------------------------------------------------

class NaiveEnumerator(BaselineRunner):
    """Exhaustively enumerate all agent-action interleavings up to a depth.

    The state space grows as ``O(|agents|^depth)``, making this baseline
    intractable for large instances.  It serves as the ground-truth
    comparator for small benchmarks where complete enumeration is feasible.
    """

    @property
    def name(self) -> str:
        return "naive_enumerator"

    def run(
        self,
        env_config: dict,
        policy_configs: list[dict],
        spec: str,
    ) -> dict:
        """Enumerate all schedules and check safety for each.

        Supports early termination via ``max_states`` in the config to
        cap the total number of states explored.

        Returns:
            Result dict with all detected races.
        """
        start = time.perf_counter()
        num_agents = len(policy_configs)
        max_depth: int = self._config.get("max_depth", 8)
        max_states: int = self._config.get("max_states", 500_000)
        early_stop: bool = self._config.get("early_stop_on_first", False)

        races: list[dict] = []
        total_states = 0
        visited: set[str] = set()

        # DFS with explicit stack: (state, schedule_so_far, depth)
        initial_state = dict(env_config.get("initial_state", {}))
        stack: list[tuple[dict, list[int], int]] = [
            (initial_state, [], 0),
        ]

        while stack:
            if total_states >= max_states:
                break

            state, schedule, depth = stack.pop()
            fp = self._state_fingerprint(state, tuple(schedule))
            if fp in visited:
                continue
            visited.add(fp)
            total_states += 1

            violation = self._check_spec(state, spec)
            if violation is not None:
                agents_involved = sorted(set(schedule[-max(1, len(schedule)):]))
                races.append({
                    "agents": agents_involved,
                    "state": dict(state),
                    "type": "enumerated_violation",
                    "description": violation,
                    "schedule": list(schedule),
                    "depth": depth,
                })
                if early_stop:
                    break
                continue  # prune below violated states

            if depth >= max_depth:
                continue

            # Expand all agent actions at this state
            for agent_idx in range(num_agents):
                next_state = self._step(state, env_config, policy_configs, agent_idx)
                next_schedule = schedule + [agent_idx]
                stack.append((next_state, next_schedule, depth + 1))

        elapsed = time.perf_counter() - start
        return {
            "races": races,
            "time_s": elapsed,
            "states_explored": total_states,
            "max_depth": max_depth,
            "exhaustive": total_states < max_states,
            "method": self.name,
        }

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _state_fingerprint(state: dict, schedule: tuple[int, ...]) -> str:
        """Fingerprint that distinguishes both state and path."""
        return str(sorted(state.items())) + "|" + str(schedule)

    @staticmethod
    def _check_spec(state: dict, spec: str) -> str | None:
        _ = spec
        return None

    @staticmethod
    def _step(
        state: dict,
        env_config: dict,
        policy_configs: list[dict],
        agent_idx: int,
    ) -> dict:
        new_state = dict(state)
        new_state["_last_agent"] = agent_idx
        new_state["_step"] = state.get("_step", 0) + 1
        return new_state


# ---------------------------------------------------------------------------
# Ablation runner – selectively disabled MARACE pipeline
# ---------------------------------------------------------------------------

class AblationRunner(BaselineRunner):
    """Run the MARACE pipeline with individual components toggled off.

    This is the primary tool for ablation studies.  Each flag in
    :class:`AblationConfig` gates a pipeline stage; disabling it causes
    the runner to skip that stage and proceed with a default no-op
    substitute.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        ablation_kwargs = config.get("ablation", {})
        self.ablation_config = AblationConfig(**ablation_kwargs)

    @property
    def name(self) -> str:
        return self.ablation_config.name or "ablation"

    def run(
        self,
        env_config: dict,
        policy_configs: list[dict],
        spec: str,
    ) -> dict:
        """Execute the MARACE pipeline with selective component disabling.

        Returns:
            Result dict augmented with a ``timing_breakdown`` mapping from
            component name to wall-clock seconds.
        """
        start = time.perf_counter()
        timing: dict[str, float] = {}
        races: list[dict] = []
        total_states = 0
        cfg = self.ablation_config

        # --- Stage 1: Decomposition ---
        t0 = time.perf_counter()
        if cfg.use_decomposition:
            interaction_groups = self._run_decomposition(env_config, policy_configs)
        else:
            # Treat entire system as one monolithic group
            interaction_groups = [list(range(len(policy_configs)))]
        timing["decomposition"] = time.perf_counter() - t0

        # --- Stage 2: HB pruning ---
        t0 = time.perf_counter()
        if cfg.use_hb_pruning:
            candidate_schedules = self._run_hb_pruning(
                interaction_groups, cfg.max_schedule_depth,
            )
        else:
            candidate_schedules = self._enumerate_all_schedules(
                interaction_groups, cfg.max_schedule_depth,
            )
        timing["hb_pruning"] = time.perf_counter() - t0

        # --- Stage 3: Abstract interpretation reachability ---
        t0 = time.perf_counter()
        if cfg.use_abstract_interpretation:
            reachable = self._run_abstract_interpretation(
                env_config, policy_configs, candidate_schedules,
            )
        else:
            reachable = candidate_schedules  # skip pruning
        timing["abstract_interpretation"] = time.perf_counter() - t0

        # --- Stage 4: Adversarial search ---
        t0 = time.perf_counter()
        if cfg.use_adversarial_search:
            found_races, states = self._run_adversarial_search(
                env_config, policy_configs, spec, reachable,
            )
        else:
            found_races, states = self._run_random_search(
                env_config, policy_configs, spec, reachable,
            )
        races.extend(found_races)
        total_states += states
        timing["adversarial_search"] = time.perf_counter() - t0

        # --- Stage 5: Importance sampling for probability estimation ---
        t0 = time.perf_counter()
        if cfg.use_importance_sampling and races:
            races = self._run_importance_sampling(
                env_config, policy_configs, spec, races,
            )
        timing["importance_sampling"] = time.perf_counter() - t0

        elapsed = time.perf_counter() - start
        return {
            "races": races,
            "time_s": elapsed,
            "states_explored": total_states,
            "timing_breakdown": timing,
            "ablation_config": cfg.describe(),
            "method": self.name,
        }

    # -- pipeline stage stubs (replaced by real MARACE modules on integration)

    @staticmethod
    def _run_decomposition(
        env_config: dict,
        policy_configs: list[dict],
    ) -> list[list[int]]:
        """Decompose agents into interaction groups."""
        n = len(policy_configs)
        if n <= 2:
            return [list(range(n))]
        # Placeholder: pair adjacent agents
        groups: list[list[int]] = []
        for i in range(0, n, 2):
            groups.append(list(range(i, min(i + 2, n))))
        return groups

    @staticmethod
    def _run_hb_pruning(
        groups: list[list[int]],
        max_depth: int,
    ) -> list[list[int]]:
        """Prune schedules using happens-before relations."""
        schedules: list[list[int]] = []
        for group in groups:
            for depth in range(1, min(max_depth + 1, 6)):
                for combo in itertools.product(group, repeat=depth):
                    schedules.append(list(combo))
                    if len(schedules) >= 10_000:
                        return schedules
        return schedules

    @staticmethod
    def _enumerate_all_schedules(
        groups: list[list[int]],
        max_depth: int,
    ) -> list[list[int]]:
        """Enumerate without HB pruning (exponential)."""
        all_agents = sorted({a for g in groups for a in g})
        schedules: list[list[int]] = []
        capped_depth = min(max_depth, 5)
        for depth in range(1, capped_depth + 1):
            for combo in itertools.product(all_agents, repeat=depth):
                schedules.append(list(combo))
                if len(schedules) >= 50_000:
                    return schedules
        return schedules

    @staticmethod
    def _run_abstract_interpretation(
        env_config: dict,
        policy_configs: list[dict],
        schedules: list[list[int]],
    ) -> list[list[int]]:
        """Filter schedules by abstract reachability."""
        # Placeholder: keep all schedules
        return schedules

    @staticmethod
    def _run_adversarial_search(
        env_config: dict,
        policy_configs: list[dict],
        spec: str,
        schedules: list[list[int]],
    ) -> tuple[list[dict], int]:
        """Directed search for spec-violating schedules."""
        return [], len(schedules)

    @staticmethod
    def _run_random_search(
        env_config: dict,
        policy_configs: list[dict],
        spec: str,
        schedules: list[list[int]],
    ) -> tuple[list[dict], int]:
        """Undirected fallback when adversarial search is disabled."""
        return [], len(schedules)

    @staticmethod
    def _run_importance_sampling(
        env_config: dict,
        policy_configs: list[dict],
        spec: str,
        races: list[dict],
    ) -> list[dict]:
        """Refine race probability estimates via importance sampling."""
        for race in races:
            race.setdefault("probability", 0.0)
        return races


# ---------------------------------------------------------------------------
# Comparator – aggregate and present results across methods
# ---------------------------------------------------------------------------

class BaselineComparator:
    """Collect results from multiple verification methods and compare them.

    Provides recall computation against planted (known) race conditions,
    speedup ratios relative to brute-force, and formatted output for
    terminals and LaTeX manuscripts.
    """

    def __init__(self) -> None:
        self._results: dict[str, dict] = {}

    def add_result(self, name: str, result: dict) -> None:
        """Register a result dictionary under *name*.

        Args:
            name: Identifier for the method (e.g. ``"marace"``,
                ``"brute_force"``).
            result: The dict returned by a :class:`BaselineRunner`.
        """
        self._results[name] = result

    # -- comparison ---------------------------------------------------------

    def compare(self) -> dict:
        """Produce per-method comparison metrics.

        Returns:
            Dictionary keyed by method name, each containing:

            * ``races_found`` – number of distinct races.
            * ``time_s`` – wall-clock time.
            * ``states_explored`` – state count.
            * ``speedup_vs_brute_force`` – ratio of brute-force time to
              this method's time (``None`` if brute-force absent).
        """
        bf_time = self._results.get("brute_force", {}).get("time_s")
        comparison: dict[str, dict] = {}

        for method_name, result in self._results.items():
            entry: dict[str, Any] = {
                "races_found": len(result.get("races", [])),
                "time_s": result.get("time_s", 0.0),
                "states_explored": result.get("states_explored", 0),
            }
            method_time = result.get("time_s", 0.0)
            if bf_time is not None and method_time > 0:
                entry["speedup_vs_brute_force"] = bf_time / method_time
            else:
                entry["speedup_vs_brute_force"] = None
            comparison[method_name] = entry

        return comparison

    def compare_with_ground_truth(
        self,
        planted_races: list[dict],
    ) -> dict:
        """Compute recall for each method against known (planted) races.

        A planted race is considered *detected* by a method if any of the
        method's reported races matches on the ``agents`` and ``type``
        fields.

        Args:
            planted_races: List of race dicts with at least ``agents``
                and ``type`` keys.

        Returns:
            Per-method dict with ``recall``, ``precision`` (if computable),
            ``true_positives``, ``false_negatives``, and ``races_found``.
        """
        comparison: dict[str, dict] = {}

        for method_name, result in self._results.items():
            detected = result.get("races", [])
            tp = 0
            matched_planted: set[int] = set()

            for d_race in detected:
                for idx, p_race in enumerate(planted_races):
                    if idx in matched_planted:
                        continue
                    if self._race_matches(d_race, p_race):
                        tp += 1
                        matched_planted.add(idx)
                        break

            fn = len(planted_races) - tp
            recall = tp / len(planted_races) if planted_races else 1.0
            precision = tp / len(detected) if detected else (1.0 if not planted_races else 0.0)

            comparison[method_name] = {
                "recall": recall,
                "precision": precision,
                "true_positives": tp,
                "false_negatives": fn,
                "races_found": len(detected),
                "time_s": result.get("time_s", 0.0),
            }

        return comparison

    # -- formatted output ---------------------------------------------------

    def summary_table(self) -> str:
        """Return an ASCII-formatted comparison table.

        Columns: Method | Races | Time (s) | States | Speedup
        """
        comparison = self.compare()
        col_method = "Method"
        col_races = "Races"
        col_time = "Time (s)"
        col_states = "States"
        col_speedup = "Speedup"

        rows: list[tuple[str, str, str, str, str]] = []
        for method_name in sorted(comparison):
            entry = comparison[method_name]
            speedup = entry["speedup_vs_brute_force"]
            speedup_str = f"{speedup:.2f}x" if speedup is not None else "—"
            rows.append((
                method_name,
                str(entry["races_found"]),
                f"{entry['time_s']:.4f}",
                str(entry["states_explored"]),
                speedup_str,
            ))

        widths = [
            max(len(col_method), *(len(r[0]) for r in rows)),
            max(len(col_races), *(len(r[1]) for r in rows)),
            max(len(col_time), *(len(r[2]) for r in rows)),
            max(len(col_states), *(len(r[3]) for r in rows)),
            max(len(col_speedup), *(len(r[4]) for r in rows)),
        ]

        def _fmt_row(vals: tuple[str, ...]) -> str:
            return " | ".join(v.ljust(w) for v, w in zip(vals, widths))

        header = _fmt_row((col_method, col_races, col_time, col_states, col_speedup))
        separator = "-+-".join("-" * w for w in widths)
        lines = [header, separator]
        for row in rows:
            lines.append(_fmt_row(row))
        return "\n".join(lines)

    def to_latex(self) -> str:
        """Return a LaTeX ``tabular`` environment comparing all methods.

        Suitable for direct inclusion in a publication manuscript.
        """
        comparison = self.compare()
        header = (
            r"\begin{tabular}{lrrrr}" "\n"
            r"\toprule" "\n"
            r"Method & Races & Time (s) & States & Speedup \\" "\n"
            r"\midrule" "\n"
        )
        body_lines: list[str] = []
        for method_name in sorted(comparison):
            entry = comparison[method_name]
            speedup = entry["speedup_vs_brute_force"]
            speedup_str = f"{speedup:.2f}$\\times$" if speedup is not None else "---"
            escaped_name = method_name.replace("_", r"\_")
            body_lines.append(
                f"{escaped_name} & {entry['races_found']} & "
                f"{entry['time_s']:.4f} & {entry['states_explored']} & "
                f"{speedup_str} \\\\"
            )
        footer = r"\bottomrule" "\n" r"\end{tabular}"
        return header + "\n".join(body_lines) + "\n" + footer

    # -- private helpers ----------------------------------------------------

    @staticmethod
    def _race_matches(detected: dict, planted: dict) -> bool:
        """Check whether a detected race matches a planted ground-truth race.

        Matching criteria: same set of involved agents **and** compatible
        race type (either exact match or the planted type is a substring
        of the detected type).
        """
        d_agents = set(detected.get("agents", []))
        p_agents = set(planted.get("agents", []))
        if d_agents != p_agents:
            return False
        d_type = detected.get("type", "")
        p_type = planted.get("type", "")
        return d_type == p_type or p_type in d_type
