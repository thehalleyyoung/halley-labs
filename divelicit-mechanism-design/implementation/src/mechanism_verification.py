"""
Mechanism Verification Module
=============================
Formal verification of game-theoretic mechanism properties: incentive
compatibility, individual rationality, budget balance, efficiency,
envy-freeness, revenue optimality, collusion resistance, and false-name-proofness.
"""

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist


@dataclass
class VerificationResult:
    """Outcome of a single verification check.

    Attributes:
        passed: Whether the mechanism satisfies the property.
        property_name: Name of the verified property.
        details: Human-readable explanation.
        counterexample: Concrete violation witness, or None.
        metric_value: Scalar summarising violation magnitude.
        confidence: Statistical confidence for sampling-based checks.
    """
    passed: bool
    property_name: str = ""
    details: str = ""
    counterexample: Optional[Dict[str, Any]] = None
    metric_value: Optional[float] = None
    confidence: Optional[float] = None


@dataclass
class Mechanism:
    """Abstract mechanism with allocation and payment rules.

    Attributes:
        num_agents: Number of participating agents.
        type_space: Per-agent finite type arrays (None → continuous [0,1]).
        allocation_rule: types → allocation vector.
        payment_rule: types → payment vector.
        valuation: (agent, type, alloc) → value; defaults to type*alloc.
        reserve_utility: Per-agent outside option; defaults to zeros.
    """
    num_agents: int
    type_space: Optional[List[Optional[np.ndarray]]] = None
    allocation_rule: Optional[Callable[..., np.ndarray]] = None
    payment_rule: Optional[Callable[..., np.ndarray]] = None
    valuation: Optional[Callable[..., float]] = None
    reserve_utility: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.reserve_utility is None:
            self.reserve_utility = np.zeros(self.num_agents)
        if self.valuation is None:
            self.valuation = lambda i, t, a: t * a

    def allocate(self, types: np.ndarray) -> np.ndarray:
        """Return allocation vector for the given type profile."""
        if self.allocation_rule is None:
            raise ValueError("allocation_rule not set")
        return np.asarray(self.allocation_rule(types), dtype=np.float64)

    def pay(self, types: np.ndarray) -> np.ndarray:
        """Return payment vector for the given type profile."""
        if self.payment_rule is None:
            raise ValueError("payment_rule not set")
        return np.asarray(self.payment_rule(types), dtype=np.float64)

    def utility(self, agent: int, own_type: float, types: np.ndarray) -> float:
        """Quasi-linear utility: v_i(θ_i, x_i) − p_i."""
        allocs = self.allocate(types)
        payments = self.pay(types)
        return float(self.valuation(agent, own_type, allocs[agent]) - payments[agent])

    def is_finite(self) -> bool:
        """True when every agent has an explicit finite type space."""
        if self.type_space is None:
            return False
        return all(ts is not None for ts in self.type_space)


class MechanismVerifier:
    """Verifies game-theoretic properties of a Mechanism.

    Args:
        num_samples: Monte-Carlo samples for continuous checks.
        tolerance: Numerical tolerance.
        seed: RNG seed for reproducibility.
    """

    def __init__(self, num_samples: int = 5000, tolerance: float = 1e-8,
                 seed: int = 42) -> None:
        self.num_samples = num_samples
        self.tolerance = tolerance
        self.rng = np.random.default_rng(seed)

    # ---- dispatch ----

    def verify(self, mechanism: Mechanism, property_name: str) -> VerificationResult:
        """Verify a named property of *mechanism*."""
        dispatch = {
            "incentive_compatibility": self.check_incentive_compatibility,
            "individual_rationality": self.check_individual_rationality,
            "weak_budget_balance": self.check_weak_budget_balance,
            "strong_budget_balance": self.check_strong_budget_balance,
            "efficiency": self.check_efficiency,
            "envy_freeness": self.check_envy_freeness,
            "revenue_optimality": self.check_revenue_optimality,
            "collusion_resistance": self.check_collusion_resistance,
            "false_name_proofness": self.check_false_name_proofness,
        }
        checker = dispatch.get(property_name)
        if checker is None:
            return VerificationResult(
                passed=False, property_name=property_name,
                details=f"Unknown property '{property_name}'. "
                        f"Available: {list(dispatch.keys())}",
            )
        result = checker(mechanism)
        result.property_name = property_name
        return result

    # ================================================================
    # Incentive Compatibility
    # ================================================================

    def check_incentive_compatibility(self, mech: Mechanism) -> VerificationResult:
        """DSIC check: no agent benefits from unilateral misreport.

        Exhaustive over finite type spaces; sampling + gradient search
        for continuous type spaces.
        """
        if mech.is_finite():
            return self._ic_finite(mech)
        return self._ic_sampled(mech)

    def _ic_finite(self, mech: Mechanism) -> VerificationResult:
        """Exhaustive IC: enumerate all profiles × all deviations."""
        spaces = mech.type_space
        n = mech.num_agents
        max_viol = 0.0
        worst: Optional[Dict[str, Any]] = None
        for profile in itertools.product(*spaces):
            types = np.array(profile, dtype=np.float64)
            for i in range(n):
                truth_u = mech.utility(i, types[i], types)
                for alt in spaces[i]:
                    if np.abs(alt - types[i]) < self.tolerance:
                        continue
                    dev = types.copy(); dev[i] = alt
                    dev_u = mech.utility(i, types[i], dev)
                    gain = dev_u - truth_u
                    if gain > max_viol:
                        max_viol = gain
                        worst = dict(agent=i, true_type=float(types[i]),
                                     deviation=float(alt), others=types.tolist(),
                                     truth_utility=truth_u, deviation_utility=dev_u,
                                     gain=float(gain))
        passed = max_viol <= self.tolerance
        return VerificationResult(
            passed=passed,
            details="IC satisfied (exhaustive)." if passed
                    else f"IC violated: max gain={max_viol:.6g}",
            counterexample=worst if not passed else None,
            metric_value=max_viol, confidence=1.0)

    def _ic_sampled(self, mech: Mechanism) -> VerificationResult:
        """Sampling IC: for each profile use L-BFGS-B to find best deviation."""
        n = mech.num_agents
        max_viol = 0.0
        worst: Optional[Dict[str, Any]] = None
        violations = 0
        for _ in range(self.num_samples):
            types = self.rng.uniform(0, 1, size=n)
            for i in range(n):
                truth_u = mech.utility(i, types[i], types)
                def neg_gain(alt, _i=i, _types=types.copy(), _tu=truth_u):
                    d = _types.copy(); d[_i] = np.clip(alt[0], 0, 1)
                    return -(mech.utility(_i, _types[_i], d) - _tu)
                res = minimize(neg_gain, x0=[self.rng.uniform()],
                               bounds=[(0, 1)], method="L-BFGS-B")
                gain = -res.fun
                if gain > self.tolerance:
                    violations += 1
                if gain > max_viol:
                    max_viol = gain
                    worst = dict(agent=i, true_type=float(types[i]),
                                 deviation=float(np.clip(res.x[0], 0, 1)),
                                 others=types.tolist(), gain=float(gain))
        passed = violations == 0
        conf = self._clopper_pearson(violations, self.num_samples * n)
        return VerificationResult(
            passed=passed,
            details=f"IC {'satisfied' if passed else 'violated'} "
                    f"({self.num_samples} samples, {violations} violations).",
            counterexample=worst if not passed else None,
            metric_value=max_viol, confidence=conf)

    # ================================================================
    # Individual Rationality
    # ================================================================

    def check_individual_rationality(self, mech: Mechanism) -> VerificationResult:
        """IR: every agent gets at least reserve utility under truth-telling."""
        if mech.is_finite():
            return self._ir_finite(mech)
        return self._ir_sampled(mech)

    def _ir_finite(self, mech: Mechanism) -> VerificationResult:
        n = mech.num_agents
        min_surplus = np.inf
        worst: Optional[Dict[str, Any]] = None
        for profile in itertools.product(*mech.type_space):
            types = np.array(profile, dtype=np.float64)
            for i in range(n):
                u = mech.utility(i, types[i], types)
                surplus = u - mech.reserve_utility[i]
                if surplus < min_surplus:
                    min_surplus = surplus
                    if surplus < -self.tolerance:
                        worst = dict(agent=i, types=types.tolist(), utility=u,
                                     reserve=float(mech.reserve_utility[i]),
                                     deficit=float(-surplus))
        passed = min_surplus >= -self.tolerance
        return VerificationResult(
            passed=passed,
            details="IR satisfied (exhaustive)." if passed
                    else f"IR violated: worst deficit={-min_surplus:.6g}",
            counterexample=worst if not passed else None,
            metric_value=float(min_surplus), confidence=1.0)

    def _ir_sampled(self, mech: Mechanism) -> VerificationResult:
        n = mech.num_agents
        min_surplus = np.inf
        worst: Optional[Dict[str, Any]] = None
        violations = 0
        for _ in range(self.num_samples):
            types = self.rng.uniform(0, 1, size=n)
            for i in range(n):
                u = mech.utility(i, types[i], types)
                surplus = u - mech.reserve_utility[i]
                if surplus < -self.tolerance:
                    violations += 1
                if surplus < min_surplus:
                    min_surplus = surplus
                    if surplus < -self.tolerance:
                        worst = dict(agent=i, types=types.tolist(), utility=u,
                                     reserve=float(mech.reserve_utility[i]),
                                     deficit=float(-surplus))
        passed = violations == 0
        conf = self._clopper_pearson(violations, self.num_samples * n)
        return VerificationResult(
            passed=passed,
            details=f"IR {'satisfied' if passed else 'violated'} "
                    f"({self.num_samples} samples).",
            counterexample=worst if not passed else None,
            metric_value=float(min_surplus), confidence=conf)

    # ================================================================
    # Budget Balance
    # ================================================================

    def check_weak_budget_balance(self, mech: Mechanism) -> VerificationResult:
        """Weak BB: Σ payments ≥ 0 for every profile (no deficit)."""
        return self._budget_balance(mech, strong=False)

    def check_strong_budget_balance(self, mech: Mechanism) -> VerificationResult:
        """Strong BB: Σ payments = 0 for every profile (balanced transfers)."""
        return self._budget_balance(mech, strong=True)

    def _budget_balance(self, mech: Mechanism, strong: bool) -> VerificationResult:
        label = "Strong" if strong else "Weak"
        profiles = self._get_profiles(mech)
        worst_val = 0.0 if strong else np.inf
        worst: Optional[Dict[str, Any]] = None
        violations = 0
        for profile in profiles:
            types = np.array(profile, dtype=np.float64)
            payments = mech.pay(types)
            total = float(np.sum(payments))
            if strong:
                if np.abs(total) > self.tolerance:
                    violations += 1
                    if np.abs(total) > np.abs(worst_val):
                        worst_val = total
                        worst = dict(types=types.tolist(),
                                     payments=payments.tolist(), total=total)
            else:
                if total < -self.tolerance:
                    violations += 1
                if total < worst_val:
                    worst_val = total
                    if total < -self.tolerance:
                        worst = dict(types=types.tolist(),
                                     payments=payments.tolist(), total=total)
        passed = violations == 0
        conf = 1.0 if mech.is_finite() else self._clopper_pearson(violations, len(profiles))
        return VerificationResult(
            passed=passed,
            details=f"{label} BB {'satisfied' if passed else 'violated'}.",
            counterexample=worst if not passed else None,
            metric_value=float(worst_val), confidence=conf)

    # ================================================================
    # Efficiency
    # ================================================================

    def check_efficiency(self, mech: Mechanism) -> VerificationResult:
        """Efficiency: allocation maximises total surplus Σ v_i(θ_i, x_i)."""
        if mech.is_finite():
            return self._eff_finite(mech)
        return self._eff_sampled(mech)

    def _total_surplus(self, mech: Mechanism, types: np.ndarray,
                       allocs: np.ndarray) -> float:
        return float(sum(mech.valuation(i, types[i], allocs[i])
                         for i in range(mech.num_agents)))

    def _eff_finite(self, mech: Mechanism) -> VerificationResult:
        n = mech.num_agents
        max_gap = 0.0
        worst: Optional[Dict[str, Any]] = None
        for profile in itertools.product(*mech.type_space):
            types = np.array(profile, dtype=np.float64)
            chosen = mech.allocate(types)
            chosen_s = self._total_surplus(mech, types, chosen)
            best_s = -np.inf
            best_a = chosen.copy()
            for bits in itertools.product([0.0, 1.0], repeat=n):
                alt = np.array(bits, dtype=np.float64)
                s = self._total_surplus(mech, types, alt)
                if s > best_s:
                    best_s = s; best_a = alt.copy()
            gap = best_s - chosen_s
            if gap > max_gap:
                max_gap = gap
                if gap > self.tolerance:
                    worst = dict(types=types.tolist(),
                                 chosen_alloc=chosen.tolist(),
                                 optimal_alloc=best_a.tolist(),
                                 chosen_surplus=chosen_s,
                                 optimal_surplus=best_s, gap=gap)
        passed = max_gap <= self.tolerance
        return VerificationResult(
            passed=passed,
            details="Efficiency satisfied (exhaustive)." if passed
                    else f"Efficiency violated: gap={max_gap:.6g}",
            counterexample=worst if not passed else None,
            metric_value=max_gap, confidence=1.0)

    def _eff_sampled(self, mech: Mechanism) -> VerificationResult:
        n = mech.num_agents
        max_gap = 0.0
        worst: Optional[Dict[str, Any]] = None
        violations = 0
        for _ in range(self.num_samples):
            types = self.rng.uniform(0, 1, size=n)
            chosen = mech.allocate(types)
            chosen_s = self._total_surplus(mech, types, chosen)
            def neg_s(x, _t=types):
                return -self._total_surplus(mech, _t, np.clip(x, 0, 1))
            res = minimize(neg_s, x0=self.rng.uniform(0, 1, size=n),
                           bounds=[(0, 1)] * n, method="L-BFGS-B")
            best_s = -res.fun
            gap = best_s - chosen_s
            if gap > self.tolerance:
                violations += 1
            if gap > max_gap:
                max_gap = gap
                if gap > self.tolerance:
                    worst = dict(types=types.tolist(),
                                 chosen_alloc=chosen.tolist(),
                                 optimal_alloc=np.clip(res.x, 0, 1).tolist(),
                                 gap=gap)
        passed = violations == 0
        conf = self._clopper_pearson(violations, self.num_samples)
        return VerificationResult(
            passed=passed,
            details=f"Efficiency {'satisfied' if passed else 'violated'} "
                    f"({self.num_samples} samples).",
            counterexample=worst if not passed else None,
            metric_value=max_gap, confidence=conf)

    # ================================================================
    # Envy-Freeness
    # ================================================================

    def check_envy_freeness(self, mech: Mechanism) -> VerificationResult:
        """EF: no agent prefers another agent's (allocation, payment) bundle."""
        if mech.is_finite():
            return self._ef_finite(mech)
        return self._ef_sampled(mech)

    def _ef_finite(self, mech: Mechanism) -> VerificationResult:
        n = mech.num_agents
        max_envy = 0.0
        worst: Optional[Dict[str, Any]] = None
        for profile in itertools.product(*mech.type_space):
            types = np.array(profile, dtype=np.float64)
            allocs = mech.allocate(types)
            pays = mech.pay(types)
            for i in range(n):
                ui = mech.valuation(i, types[i], allocs[i]) - pays[i]
                for j in range(n):
                    if i == j:
                        continue
                    uj = mech.valuation(i, types[i], allocs[j]) - pays[j]
                    envy = uj - ui
                    if envy > max_envy:
                        max_envy = envy
                        if envy > self.tolerance:
                            worst = dict(types=types.tolist(), envious=i,
                                         envied=j, own_util=ui,
                                         envied_util=uj, envy=envy)
        passed = max_envy <= self.tolerance
        return VerificationResult(
            passed=passed,
            details="EF satisfied (exhaustive)." if passed
                    else f"EF violated: max envy={max_envy:.6g}",
            counterexample=worst if not passed else None,
            metric_value=max_envy, confidence=1.0)

    def _ef_sampled(self, mech: Mechanism) -> VerificationResult:
        n = mech.num_agents
        max_envy = 0.0
        worst: Optional[Dict[str, Any]] = None
        violations = 0
        for _ in range(self.num_samples):
            types = self.rng.uniform(0, 1, size=n)
            allocs = mech.allocate(types)
            pays = mech.pay(types)
            for i in range(n):
                ui = mech.valuation(i, types[i], allocs[i]) - pays[i]
                for j in range(n):
                    if i == j:
                        continue
                    uj = mech.valuation(i, types[i], allocs[j]) - pays[j]
                    envy = uj - ui
                    if envy > self.tolerance:
                        violations += 1
                    if envy > max_envy:
                        max_envy = envy
                        if envy > self.tolerance:
                            worst = dict(types=types.tolist(), envious=i,
                                         envied=j, own_util=ui,
                                         envied_util=uj, envy=envy)
        passed = violations == 0
        total = self.num_samples * n * max(n - 1, 1)
        conf = self._clopper_pearson(violations, total)
        return VerificationResult(
            passed=passed,
            details=f"EF {'satisfied' if passed else 'violated'} "
                    f"({self.num_samples} samples).",
            counterexample=worst if not passed else None,
            metric_value=max_envy, confidence=conf)

    # ================================================================
    # Revenue Optimality (Myerson benchmark)
    # ================================================================

    def check_revenue_optimality(self, mech: Mechanism) -> VerificationResult:
        """Compare mechanism revenue to Myerson optimal auction (U[0,1] iid).

        Myerson optimal = second-price with reserve 0.5.  We compute the
        ratio mechanism_rev / myerson_rev via Monte-Carlo and flag if the
        mechanism exceeds the theoretical bound (impossible for DSIC+IR).
        """
        n = mech.num_agents
        mech_revs = np.empty(self.num_samples)
        myer_revs = np.empty(self.num_samples)
        for s in range(self.num_samples):
            types = self.rng.uniform(0, 1, size=n)
            mech_revs[s] = float(np.sum(mech.pay(types)))
            myer_revs[s] = self._myerson_rev(types, reserve=0.5)
        mech_mean = float(np.mean(mech_revs))
        myer_mean = float(np.mean(myer_revs))
        ratio = mech_mean / myer_mean if myer_mean > self.tolerance else (
            np.inf if mech_mean > self.tolerance else 1.0)
        exceeds = ratio > 1.0 + self.tolerance
        conf = self._bootstrap_conf(mech_revs, myer_revs)
        return VerificationResult(
            passed=not exceeds,
            details=f"Mechanism rev={mech_mean:.6g}, Myerson rev={myer_mean:.6g}, "
                    f"ratio={ratio:.4f}",
            counterexample=dict(ratio=ratio, mech=mech_mean, myerson=myer_mean)
                          if exceeds else None,
            metric_value=ratio, confidence=conf)

    @staticmethod
    def _myerson_rev(types: np.ndarray, reserve: float) -> float:
        """Myerson (second-price + reserve) revenue for one profile."""
        eligible = types[types >= reserve]
        if len(eligible) == 0:
            return 0.0
        if len(eligible) == 1:
            return reserve
        top2 = np.partition(eligible, -2)[-2:]
        return float(max(np.min(top2), reserve))

    def _bootstrap_conf(self, mech_r: np.ndarray, myer_r: np.ndarray,
                         B: int = 1000) -> float:
        """Bootstrap confidence that mechanism rev ≤ Myerson rev."""
        n = len(mech_r)
        ok = 0
        for _ in range(B):
            idx = self.rng.integers(0, n, size=n)
            if np.mean(mech_r[idx]) <= np.mean(myer_r[idx]) + self.tolerance:
                ok += 1
        return ok / B

    # ================================================================
    # Collusion Resistance
    # ================================================================

    def check_collusion_resistance(self, mech: Mechanism) -> VerificationResult:
        """Check that no coalition can jointly deviate for mutual benefit.

        For each coalition S ⊆ N (|S| ≤ 4) and each type profile, use
        scipy.optimize to search for a joint misreport where every member
        weakly gains and at least one strictly gains.
        """
        n = mech.num_agents
        max_cs = min(n, 4)
        max_gain = 0.0
        worst: Optional[Dict[str, Any]] = None
        violations = 0
        profiles = self._get_profiles(mech, cap=500)
        for profile in profiles:
            types = np.array(profile, dtype=np.float64)
            truth_u = np.array([mech.utility(i, types[i], types) for i in range(n)])
            for sz in range(2, max_cs + 1):
                for coal in itertools.combinations(range(n), sz):
                    g = self._coal_deviation(mech, types, truth_u, list(coal))
                    if g > self.tolerance:
                        violations += 1
                    if g > max_gain:
                        max_gain = g
                        if g > self.tolerance:
                            worst = dict(types=types.tolist(),
                                         coalition=list(coal), gain=g)
        passed = violations == 0
        return VerificationResult(
            passed=passed,
            details="Collusion resistant." if passed
                    else f"Collusion found: gain={max_gain:.6g}",
            counterexample=worst if not passed else None,
            metric_value=max_gain,
            confidence=1.0 if mech.is_finite() else 0.95)

    def _coal_deviation(self, mech: Mechanism, types: np.ndarray,
                        truth_u: np.ndarray, coal: List[int]) -> float:
        """Find best joint deviation for coalition via L-BFGS-B.

        Objective: maximise total coalition gain subject to every member
        gaining weakly.  Penalty term enforces individual rationality
        within the coalition.
        """
        k = len(coal)
        tol = self.tolerance

        def objective(flat):
            devs = np.clip(flat, 0, 1)
            d = types.copy()
            for idx, ag in enumerate(coal):
                d[ag] = devs[idx]
            gains = np.array([mech.utility(ag, types[ag], d) - truth_u[ag]
                              for ag in coal])
            mg = float(np.min(gains))
            if mg < -tol:
                return -mg * 10.0  # penalise hurting a member
            return -float(np.sum(gains))

        res = minimize(objective, x0=self.rng.uniform(0, 1, size=k),
                       bounds=[(0, 1)] * k, method="L-BFGS-B")
        d = types.copy()
        for idx, ag in enumerate(coal):
            d[ag] = np.clip(res.x[idx], 0, 1)
        gains = np.array([mech.utility(ag, types[ag], d) - truth_u[ag]
                          for ag in coal])
        if np.all(gains >= -tol) and np.any(gains > tol):
            return float(np.sum(gains))
        return 0.0

    # ================================================================
    # False-Name-Proofness (Sybil resistance)
    # ================================================================

    def check_false_name_proofness(self, mech: Mechanism) -> VerificationResult:
        """Check that no agent profits from splitting into two identities.

        Agent i takes over slot j and reports (t_a, t_b), collecting both
        bundles.  We search for a split that increases combined utility
        above the truthful single-identity utility.
        """
        n = mech.num_agents
        max_gain = 0.0
        worst: Optional[Dict[str, Any]] = None
        violations = 0
        profiles = self._get_profiles(mech, cap=500)
        for profile in profiles:
            types = np.array(profile, dtype=np.float64)
            for i in range(n):
                g, info = self._sybil_deviation(mech, types, i)
                if g > self.tolerance:
                    violations += 1
                if g > max_gain:
                    max_gain = g
                    if g > self.tolerance:
                        worst = dict(agent=i, types=types.tolist(),
                                     gain=g, **info)
        passed = violations == 0
        return VerificationResult(
            passed=passed,
            details="False-name-proof." if passed
                    else f"Sybil attack found: gain={max_gain:.6g}",
            counterexample=worst if not passed else None,
            metric_value=max_gain,
            confidence=1.0 if mech.is_finite() else 0.95)

    def _sybil_deviation(self, mech: Mechanism, types: np.ndarray,
                         agent: int) -> Tuple[float, Dict[str, Any]]:
        """Search for a profitable Sybil split for *agent*.

        Agent controls both slot *agent* and slot *j*, reporting (t_a, t_b).
        Combined utility = v(θ_i, x_agent) + v(θ_i, x_j) − p_agent − p_j.
        """
        n = mech.num_agents
        true_u = mech.utility(agent, types[agent], types)
        best_gain = 0.0
        best_info: Dict[str, Any] = {}
        for j in range(n):
            if j == agent:
                continue
            def neg_gain(params, _j=j):
                ta, tb = np.clip(params, 0, 1)
                ft = types.copy(); ft[agent] = ta; ft[_j] = tb
                allocs = mech.allocate(ft)
                pays = mech.pay(ft)
                va = mech.valuation(agent, types[agent], allocs[agent])
                vb = mech.valuation(agent, types[agent], allocs[_j])
                return -(va + vb - pays[agent] - pays[_j] - true_u)
            res = minimize(neg_gain, x0=self.rng.uniform(0, 1, size=2),
                           bounds=[(0, 1)] * 2, method="L-BFGS-B")
            g = -res.fun
            if g > best_gain:
                best_gain = g
                ta, tb = np.clip(res.x, 0, 1)
                best_info = dict(sybil_slot=j, report_a=float(ta),
                                 report_b=float(tb))
        return best_gain, best_info

    # ================================================================
    # Helpers
    # ================================================================

    def _get_profiles(self, mech: Mechanism,
                      cap: Optional[int] = None) -> List[Tuple]:
        """Enumerate (finite) or sample (continuous) type profiles."""
        if mech.is_finite():
            ps = list(itertools.product(*mech.type_space))
            if cap and len(ps) > cap:
                idx = self.rng.choice(len(ps), size=cap, replace=False)
                ps = [ps[i] for i in idx]
            return ps
        n = mech.num_agents
        cnt = cap if cap else self.num_samples
        return [tuple(self.rng.uniform(0, 1, size=n)) for _ in range(cnt)]

    @staticmethod
    def _clopper_pearson(k: int, n: int, alpha: float = 0.05) -> float:
        """Clopper-Pearson confidence that true violation rate is low.

        When k=0 violations in n trials, uses the rule-of-three bound.
        Otherwise computes exact Clopper-Pearson upper CI endpoint.
        """
        if n == 0:
            return 0.0
        if k == 0:
            upper = 1.0 - alpha ** (1.0 / max(n, 1))
            return 1.0 - upper
        upper = beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
        return 1.0 - upper

    # ---- batch helpers ----

    def verify_all(self, mech: Mechanism,
                   properties: Optional[List[str]] = None
                   ) -> Dict[str, VerificationResult]:
        """Verify multiple properties; defaults to all nine checks."""
        if properties is None:
            properties = [
                "incentive_compatibility", "individual_rationality",
                "weak_budget_balance", "strong_budget_balance",
                "efficiency", "envy_freeness", "revenue_optimality",
                "collusion_resistance", "false_name_proofness",
            ]
        return {p: self.verify(mech, p) for p in properties}

    def summary_report(self, results: Dict[str, VerificationResult]) -> str:
        """Human-readable summary of verification results."""
        lines = ["=" * 60, "Mechanism Verification Report", "=" * 60]
        for name, r in results.items():
            tag = "PASS" if r.passed else "FAIL"
            ln = f"[{tag}] {name}"
            if r.metric_value is not None:
                ln += f"  (metric={r.metric_value:.6g})"
            if r.confidence is not None:
                ln += f"  [conf={r.confidence:.3f}]"
            lines.append(ln)
            if r.details:
                lines.append(f"       {r.details}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ====================================================================
# Factory helpers for common mechanisms
# ====================================================================

def second_price_auction(num_agents: int, reserve: float = 0.0) -> Mechanism:
    """Vickrey second-price single-item auction with optional reserve."""
    def alloc(types):
        a = np.zeros(len(types))
        elig = np.where(types >= reserve)[0]
        if len(elig) > 0:
            a[elig[np.argmax(types[elig])]] = 1.0
        return a
    def pay(types):
        p = np.zeros(len(types))
        elig = np.where(types >= reserve)[0]
        if len(elig) == 0:
            return p
        w = elig[np.argmax(types[elig])]
        s = np.sort(types[elig])[::-1]
        p[w] = max(s[1], reserve) if len(s) >= 2 else reserve
        return p
    return Mechanism(num_agents=num_agents, allocation_rule=alloc,
                     payment_rule=pay)


def first_price_auction(num_agents: int) -> Mechanism:
    """First-price auction (NOT incentive-compatible)."""
    def alloc(types):
        a = np.zeros(len(types)); a[np.argmax(types)] = 1.0; return a
    def pay(types):
        p = np.zeros(len(types)); p[np.argmax(types)] = types[np.argmax(types)]
        return p
    return Mechanism(num_agents=num_agents, allocation_rule=alloc,
                     payment_rule=pay)


def all_pay_auction(num_agents: int) -> Mechanism:
    """All-pay auction (not IC, not IR)."""
    def alloc(types):
        a = np.zeros(len(types)); a[np.argmax(types)] = 1.0; return a
    def pay(types):
        return types.copy()
    return Mechanism(num_agents=num_agents, allocation_rule=alloc,
                     payment_rule=pay)


def vcg_mechanism(num_agents: int) -> Mechanism:
    """VCG mechanism for a single item (IC + IR + efficient)."""
    def alloc(types):
        a = np.zeros(len(types)); a[np.argmax(types)] = 1.0; return a
    def pay(types):
        p = np.zeros(len(types))
        w = int(np.argmax(types))
        rem = np.delete(types, w)
        p[w] = float(np.max(rem)) if len(rem) > 0 else 0.0
        return p
    return Mechanism(num_agents=num_agents, allocation_rule=alloc,
                     payment_rule=pay)
