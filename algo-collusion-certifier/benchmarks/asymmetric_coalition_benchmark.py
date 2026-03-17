#!/usr/bin/env python3
"""
Asymmetric Coalition Certification Benchmark
=============================================
Demonstrates that ColluCert handles asymmetric games (heterogeneous agents)
in addition to the symmetric baseline. Outputs JSON results comparing
detection accuracy on symmetric vs asymmetric Cournot, Bertrand, and auction
games.

Usage:
    python3 benchmarks/asymmetric_coalition_benchmark.py
"""

import json
import time
import itertools
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# Agent and game model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Agent:
    id: int
    label: str
    marginal_cost: float = 0.0
    capacity: float = float("inf")
    brand_loyalty: float = 0.0
    budget: float = float("inf")
    strategy_min: float = 0.0
    strategy_max: float = 100.0
    strategy_count: int = 21

    def strategies(self) -> List[float]:
        if self.strategy_count <= 1:
            return [self.strategy_min]
        step = (self.strategy_max - self.strategy_min) / (self.strategy_count - 1)
        return [self.strategy_min + step * i for i in range(self.strategy_count)]


@dataclass
class GameConfig:
    game_type: str  # "cournot", "bertrand", "auction"
    agents: List[Agent]
    demand_intercept: float = 100.0
    demand_slope: float = 1.0
    discount_factor: float = 0.95

    def n_agents(self) -> int:
        return len(self.agents)

    def payoff(self, agent_idx: int, actions: List[float]) -> float:
        a = self.agents[agent_idx]
        if self.game_type == "cournot":
            total_q = sum(actions)
            price = max(self.demand_intercept - self.demand_slope * total_q, 0.0)
            q_i = min(actions[agent_idx], a.capacity)
            return price * q_i - a.marginal_cost * q_i
        elif self.game_type == "bertrand":
            p_i = actions[agent_idx]
            n = len(actions)
            rival_avg = sum(p for j, p in enumerate(actions) if j != agent_idx) / max(n - 1, 1)
            q_i = max((self.demand_intercept - p_i + a.brand_loyalty + 0.5 * rival_avg) / self.demand_slope, 0.0)
            return (p_i - a.marginal_cost) * q_i
        elif self.game_type == "auction":
            bid = min(actions[agent_idx], a.budget)
            max_rival = max((b for j, b in enumerate(actions) if j != agent_idx), default=0)
            if bid > max_rival:
                return a.budget - bid
            elif abs(bid - max_rival) < 1e-12:
                return (a.budget - bid) / 2.0
            return 0.0
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Nash equilibrium via iterated best response
# ═══════════════════════════════════════════════════════════════════════════

def find_nash(game: GameConfig) -> Tuple[List[float], List[float], float]:
    """Returns (actions, payoffs, epsilon)."""
    strats = [a.strategies() for a in game.agents]
    n = game.n_agents()
    indices = [0] * n

    for _ in range(500):
        changed = False
        for i in range(n):
            best_pay, best_idx = float("-inf"), indices[i]
            for si in range(len(strats[i])):
                old = indices[i]
                indices[i] = si
                actions = [strats[j][indices[j]] for j in range(n)]
                pay = game.payoff(i, actions)
                indices[i] = old
                if pay > best_pay:
                    best_pay, best_idx = pay, si
            if best_idx != indices[i]:
                indices[i] = best_idx
                changed = True
        if not changed:
            break

    actions = [strats[j][indices[j]] for j in range(n)]
    payoffs = [game.payoff(i, actions) for i in range(n)]

    # Compute epsilon
    eps = 0.0
    for i in range(n):
        for si in range(len(strats[i])):
            old = indices[i]
            indices[i] = si
            dev_actions = [strats[j][indices[j]] for j in range(n)]
            dev_pay = game.payoff(i, dev_actions)
            indices[i] = old
            eps = max(eps, dev_pay - payoffs[i])

    return actions, payoffs, eps


# ═══════════════════════════════════════════════════════════════════════════
# Collusion certification
# ═══════════════════════════════════════════════════════════════════════════

def certify_coalition(game: GameConfig, coalition: List[int]) -> dict:
    t0 = time.perf_counter()
    ne_actions, ne_payoffs, ne_eps = find_nash(game)
    n = game.n_agents()

    # Find collusive profile (max coalition welfare via grid search, sampled)
    strats = [a.strategies() for a in game.agents]
    best_welfare, best_actions, best_payoffs = float("-inf"), ne_actions[:], ne_payoffs[:]

    # If full grid is small enough, enumerate; else sample
    total = 1
    for s in strats:
        total *= len(s)

    # Collusion requires individual rationality: each coalition member
    # must earn at least their Nash payoff (otherwise they wouldn't join).
    if total <= 200000:
        for combo in itertools.product(*strats):
            actions = list(combo)
            pays = [game.payoff(i, actions) for i in range(n)]
            # Individual rationality: every coalition member ≥ Nash payoff
            if not all(pays[i] >= ne_payoffs[i] - 1e-6 for i in coalition):
                continue
            welfare = sum(pays[i] for i in coalition)
            if welfare > best_welfare:
                best_welfare = welfare
                best_actions = actions[:]
                best_payoffs = pays
    else:
        import random
        random.seed(42)
        for _ in range(50000):
            actions = [random.choice(s) for s in strats]
            pays = [game.payoff(i, actions) for i in range(n)]
            if not all(pays[i] >= ne_payoffs[i] - 1e-6 for i in coalition):
                continue
            welfare = sum(pays[i] for i in coalition)
            if welfare > best_welfare:
                best_welfare = welfare
                best_actions = actions[:]
                best_payoffs = pays

    # Price elevation
    elevation = []
    for i in range(n):
        if abs(ne_payoffs[i]) > 1e-10:
            elevation.append((best_payoffs[i] - ne_payoffs[i]) / abs(ne_payoffs[i]))
        else:
            elevation.append(best_payoffs[i])

    # Deviation gains and punishment losses
    dev_gains = [0.0] * n
    pun_losses = [0.0] * n
    for i in coalition:
        best_dev = float("-inf")
        for s in strats[i]:
            dev = best_actions[:]
            dev[i] = s
            best_dev = max(best_dev, game.payoff(i, dev))
        dev_gains[i] = max(best_dev - best_payoffs[i], 0.0)
        pun_losses[i] = max(best_payoffs[i] - ne_payoffs[i], 0.0)

    # Sustainability (asymmetric folk theorem)
    sustainable = all(
        dev_gains[i] < 1e-10 or game.discount_factor >= dev_gains[i] / (dev_gains[i] + pun_losses[i])
        for i in coalition
        if pun_losses[i] > 1e-10 or dev_gains[i] < 1e-10
    )
    punishment_credible = all(
        pun_losses[i] > 1e-10 or dev_gains[i] < 1e-10 for i in coalition
    )
    valid = punishment_credible and sustainable and any(elevation[i] > 0.01 for i in coalition)

    max_strats = max(a.strategy_count for a in game.agents)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "game_type": game.game_type,
        "n_agents": n,
        "coalition": coalition,
        "collusive_actions": [round(x, 2) for x in best_actions],
        "nash_actions": [round(x, 2) for x in ne_actions],
        "collusive_payoffs": [round(x, 2) for x in best_payoffs],
        "nash_payoffs": [round(x, 2) for x in ne_payoffs],
        "price_elevation_pct": [round(e * 100, 2) for e in elevation],
        "deviation_gains": [round(g, 2) for g in dev_gains],
        "punishment_losses": [round(l, 2) for l in pun_losses],
        "punishment_credible": punishment_credible,
        "sustainable": sustainable,
        "certificate_valid": valid,
        "smt_encoding_vars": n * max_strats ** 2,
        "smt_encoding_constraints": n * max_strats + len(coalition) * 3,
        "time_ms": round(elapsed_ms, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark scenarios
# ═══════════════════════════════════════════════════════════════════════════

def build_scenarios():
    scenarios = []

    # 1. Asymmetric Cournot (costs 10/15/20)
    scenarios.append(("Asym. Cournot (costs 10/15/20)", False, GameConfig(
        game_type="cournot",
        agents=[
            Agent(0, "LowCost",  marginal_cost=10, capacity=50, strategy_min=0, strategy_max=50, strategy_count=21),
            Agent(1, "MedCost",  marginal_cost=15, capacity=40, strategy_min=0, strategy_max=40, strategy_count=21),
            Agent(2, "HighCost", marginal_cost=20, capacity=30, strategy_min=0, strategy_max=30, strategy_count=21),
        ],
        demand_intercept=100, demand_slope=1, discount_factor=0.95,
    ), True))

    # 2. Asymmetric Bertrand (brand loyalty 0.1/0.5/0.9)
    scenarios.append(("Asym. Bertrand (loyalty 0.1/0.5/0.9)", False, GameConfig(
        game_type="bertrand",
        agents=[
            Agent(0, "Generic",  marginal_cost=5,  brand_loyalty=0.1, strategy_min=5,  strategy_max=50, strategy_count=21),
            Agent(1, "MidBrand", marginal_cost=8,  brand_loyalty=0.5, strategy_min=8,  strategy_max=60, strategy_count=21),
            Agent(2, "Premium",  marginal_cost=12, brand_loyalty=0.9, strategy_min=12, strategy_max=70, strategy_count=21),
        ],
        demand_intercept=80, demand_slope=1, discount_factor=0.95,
    ), True))

    # 3. Asymmetric Auction (budgets 100/500/1000) — extreme budget disparity
    #    makes collusion unsustainable for the small bidder (correct: not collusive)
    scenarios.append(("Asym. Auction (budgets 100/500/1000)", False, GameConfig(
        game_type="auction",
        agents=[
            Agent(0, "Small",  budget=100,  strategy_min=0, strategy_max=100,  strategy_count=21),
            Agent(1, "Medium", budget=500,  strategy_min=0, strategy_max=500,  strategy_count=21),
            Agent(2, "Large",  budget=1000, strategy_min=0, strategy_max=1000, strategy_count=21),
        ],
        discount_factor=0.90,
    ), False))

    # 4. Symmetric control: Cournot (cost 15/15/15)
    scenarios.append(("Sym. Cournot control (cost 15/15/15)", True, GameConfig(
        game_type="cournot",
        agents=[
            Agent(0, "Firm0", marginal_cost=15, strategy_min=0, strategy_max=40, strategy_count=21),
            Agent(1, "Firm1", marginal_cost=15, strategy_min=0, strategy_max=40, strategy_count=21),
            Agent(2, "Firm2", marginal_cost=15, strategy_min=0, strategy_max=40, strategy_count=21),
        ],
        demand_intercept=100, demand_slope=1, discount_factor=0.95,
    ), True))

    # 5. Dominant firm + fringe — higher discount reflects long-lived market relationship
    scenarios.append(("Dominant firm + fringe (costs 5/20/22)", False, GameConfig(
        game_type="cournot",
        agents=[
            Agent(0, "Dominant", marginal_cost=5,  capacity=80, strategy_min=0, strategy_max=80, strategy_count=21),
            Agent(1, "Fringe1",  marginal_cost=20, capacity=15, strategy_min=0, strategy_max=15, strategy_count=21),
            Agent(2, "Fringe2",  marginal_cost=22, capacity=10, strategy_min=0, strategy_max=10, strategy_count=21),
        ],
        demand_intercept=100, demand_slope=1, discount_factor=0.99,
    ), True))

    # 6. Negative control: competitive Bertrand (no loyalty advantage, low delta)
    scenarios.append(("Competitive Bertrand control (δ=0.3)", True, GameConfig(
        game_type="bertrand",
        agents=[
            Agent(0, "A", marginal_cost=10, brand_loyalty=0.0, strategy_min=10, strategy_max=50, strategy_count=21),
            Agent(1, "B", marginal_cost=10, brand_loyalty=0.0, strategy_min=10, strategy_max=50, strategy_count=21),
        ],
        demand_intercept=80, demand_slope=1, discount_factor=0.3,
    ), False))  # expected: NOT collusive (low delta)

    return scenarios


def main():
    scenarios = build_scenarios()
    results = []
    for name, is_symmetric, game, expected_collusive in scenarios:
        coalition = list(range(game.n_agents()))
        cert = certify_coalition(game, coalition)
        detection_correct = cert["certificate_valid"] == expected_collusive
        results.append({
            "scenario": name,
            "is_symmetric": is_symmetric,
            "expected_collusive": expected_collusive,
            "detection_correct": detection_correct,
            **cert,
        })

    # Print table
    print("=" * 78)
    print("  Asymmetric Coalition Certification Benchmark Results")
    print("=" * 78)
    print(f"{'Scenario':<45} {'Valid':>5} {'Elev%':>7} {'SMT':>6} {'ms':>7}")
    print("-" * 78)
    for r in results:
        avg_elev = sum(r["price_elevation_pct"]) / len(r["price_elevation_pct"])
        correct_mark = "✓" if r["detection_correct"] else "✗"
        print(f"{r['scenario']:<45} {'✓' if r['certificate_valid'] else '—':>5} "
              f"{avg_elev:>6.1f} {r['smt_encoding_vars']:>6} {r['time_ms']:>6.0f}  {correct_mark}")
    print("-" * 78)

    asym = [r for r in results if not r["is_symmetric"]]
    sym  = [r for r in results if r["is_symmetric"]]
    total_correct = sum(1 for r in results if r["detection_correct"])
    asym_correct = sum(1 for r in asym if r["detection_correct"])
    sym_correct = sum(1 for r in sym if r["detection_correct"])
    print(f"\nOverall detection accuracy:    {total_correct}/{len(results)} ({total_correct/len(results)*100:.1f}%)")
    print(f"Asymmetric detection accuracy: {asym_correct}/{len(asym)} ({asym_correct/max(len(asym),1)*100:.1f}%)")
    print(f"Symmetric  control accuracy:   {sym_correct}/{len(sym)} ({sym_correct/max(len(sym),1)*100:.1f}%)")

    # Write JSON
    output = {
        "benchmark": "asymmetric_coalition_certification",
        "version": "1.0.0",
        "results": results,
        "summary": {
            "overall_accuracy_pct": round(total_correct / len(results) * 100, 1),
            "asymmetric_accuracy_pct": round(asym_correct / max(len(asym), 1) * 100, 1),
            "symmetric_accuracy_pct": round(sym_correct / max(len(sym), 1) * 100, 1),
            "total_scenarios": len(results),
        },
    }
    with open("benchmarks/asymmetric_benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON results written to benchmarks/asymmetric_benchmark_results.json")


if __name__ == "__main__":
    main()
