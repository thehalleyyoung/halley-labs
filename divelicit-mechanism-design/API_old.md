# DivFlow API Reference

Complete reference for all public functions and data classes.

---

## Table of Contents

- [Core API](#core-api)
- [Transport (Sinkhorn)](#transport-srctransportpy)
- [Coverage Certificates](#coverage-certificates-srccoveragepy)
- [Composition Theorem & ε-IC](#composition-theorem-srccomposition_theorempy)
- [Mechanism Design](#mechanism-design-srcmechanismpy)
- [Auction Mechanisms](#auction-mechanisms-srcauction_mechanismspy)
- [Information Aggregation](#information-aggregation-srcinformation_aggregationpy)
- [Strategic Diversity](#strategic-diversity-srcstrategic_diversitypy)
- [Deliberation](#deliberation-srcdeliberationpy)
- [Portfolio Optimization](#portfolio-optimization-srcportfolio_optimizationpy)
- [Experiment Design](#experiment-design-srcexperiment_designpy)
- [Collective Intelligence](#collective-intelligence-srccollective_intelligencepy)
- [Ethical Elicitation](#ethical-elicitation-srcethical_elicitationpy)
- [Data Classes](#data-classes)
- [Mechanism Core](#mechanism-core-srcmechanism_corepy)
- [Voting Systems](#voting-systems-srcvoting_systemspy)
- [Auction Engine](#auction-engine-srcauction_enginepy)
- [Fair Division](#fair-division-srcfair_divisionpy)
- [Matching Markets](#matching-markets-srcmatching_marketspy)
- [Information Elicitation](#information-elicitation-srcinformation_elicitationpy)
- [Social Choice Analysis](#social-choice-analysis-srcsocial_choice_analysispy)
- [Population Games](#population-games-srcpopulation_gamespy)

---

## Transport (`src/transport.py`)

### `sinkhorn_divergence(X, Y, reg=0.1, n_iter=50) -> float`
Debiased Sinkhorn divergence: S_ε(μ, ν) = OT_ε(μ,ν) - ½OT_ε(μ,μ) - ½OT_ε(ν,ν). Non-negative, zero iff μ=ν.

### `sinkhorn_potentials(X, Y, reg=0.1, n_iter=100) -> (f, g)`
Dual potentials (f, g) where g(y_j) measures how underserved location y_j is.

### `sinkhorn_candidate_scores(candidates, history, reference, reg=None) -> ndarray`
Score each candidate by marginal Sinkhorn divergence reduction when added to history.

### `cost_matrix(X, Y, metric="euclidean") -> ndarray`
Pairwise cost matrix. Metrics: "euclidean", "sqeuclidean", "cosine".

---

## Coverage Certificates (`src/coverage.py`)

### `estimate_coverage(points, epsilon, dim=None) -> CoverageCertificate`
Coverage estimate using metric entropy bounds with explicit constants (C_cov=3).
Returns certificate with `ci_lower`, `ci_upper` (Clopper-Pearson), and `explicit_constants` dict.

### `epsilon_net_certificate(points, reference, epsilon, delta=0.05) -> CoverageCertificate`
ε-net certificate with Clopper-Pearson exact CI. Projects to effective subspace via PCA.

### `clopper_pearson_ci(k, n, alpha=0.05) -> (lower, upper)`
Exact binomial confidence interval on proportion k/n.

### `bootstrap_ci(values, n_bootstrap=2000, alpha=0.05) -> (mean, ci_lo, ci_hi)`
Bootstrap confidence interval for the mean.

### `CoverageCertificate` (dataclass)
Fields: `coverage_fraction`, `confidence`, `n_samples`, `epsilon_radius`, `method`, `ci_lower`, `ci_upper`, `explicit_constants`.

---

## Composition Theorem (`src/composition_theorem.py`)

### `verify_ic_with_ci(embs, quals, selected, payments, select_fn, k, n_trials=1000) -> EpsilonICResult`
IC verification with Clopper-Pearson CIs. Returns violation rate, CI, max/mean utility gain, ε-IC bound.

### `composition_theorem_check(embs, quals, k, reg=0.1) -> dict`
Empirically verify composition conditions: diminishing returns, quasi-linearity, payment independence.

### `ICViolationMonitor(threshold=0.20, window_size=100)`
Runtime monitor for IC violations with graceful degradation.
- `.record(is_violation, utility_gain)` — record check result
- `.violation_rate` — current windowed rate
- `.should_degrade` — whether rate exceeds threshold
- `.get_status()` — dict with total, rate, CI, max gain

### `EpsilonICResult` (dataclass)
Fields: `violation_rate`, `ci_lower`, `ci_upper`, `n_violations`, `n_trials`, `max_utility_gain`, `mean_utility_gain`, `epsilon_ic_bound`, `violation_characterization`.

---

## Core API

### `elicit_diverse(prompt, agents, k, **kwargs) -> ElicitationResult`

Elicit diverse responses from multiple agents, then select the most diverse subset.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | required | Prompt to send to each agent |
| `agents` | `List[Callable]` | required | Callables `agent(prompt) -> str` |
| `k` | `int` | required | Number of diverse responses to select |
| `n_per_agent` | `int` | `5` | Queries per agent |
| `quality_weight` | `float` | `0.3` | Quality vs. diversity trade-off `[0, 1]` |
| `sinkhorn_epsilon` | `float` | `0.1` | Sinkhorn regularization |
| `embed_fn` | `Callable` | `None` | Custom `text -> np.ndarray` embedder |
| `embed_dim` | `int` | `64` | Dimension for default embedder |

```python
from divelicit.src.api import elicit_diverse

result = elicit_diverse(
    "What are creative uses for paperclips?",
    agents=[agent_gpt, agent_claude, agent_gemini],
    k=5,
    quality_weight=0.3,
)
print(result.responses)           # 5 diverse responses
print(result.diversity_score)     # Sinkhorn diversity
print(result.agent_contributions) # {0: 2, 1: 2, 2: 1}
```

---

### `select_diverse_subset(items, k, **kwargs) -> List`

Select k diverse items from any list.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `items` | `List[Any]` | required | Candidate pool |
| `k` | `int` | required | Selection budget |
| `embed_fn` | `Callable` | `None` | Item → embedding vector |
| `quality_fn` | `Callable` | `None` | Item → quality score `[0, 1]` |
| `quality_weight` | `float` | `0.3` | Quality weight |
| `method` | `str` | `"flow"` | `"flow"`, `"mmr"`, `"dpp"`, `"kmedoids"` |

```python
from divelicit.src.api import select_diverse_subset

ideas = ["idea A ...", "idea B ...", "idea C ...", ...]
best_5 = select_diverse_subset(ideas, k=5, method="flow")
```

---

### `compute_coverage_certificate(selected, universe, **kwargs) -> CoverageResult`

Compute how well a selected subset covers the universe.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `selected` | `List` | required | Chosen subset |
| `universe` | `List` | required | Full candidate pool |
| `embed_fn` | `Callable` | `None` | Embedding function |
| `epsilon` | `float` | `0.3` | Coverage ball radius |
| `confidence` | `float` | `0.95` | Certificate confidence |

```python
from divelicit.src.api import compute_coverage_certificate

result = compute_coverage_certificate(selected_ideas, all_ideas)
print(f"Coverage: {result.certificate.coverage_fraction:.1%}")
print(f"Fill distance: {result.fill_distance:.3f}")
```

---

### `mechanism_compare(items, k, methods, **kwargs) -> ComparisonResult`

Compare selection mechanisms side by side.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `items` | `List` | required | Candidate pool |
| `k` | `int` | required | Selection budget |
| `methods` | `List[str]` | `["flow","mmr","dpp","kmedoids"]` | Methods to compare |

```python
from divelicit.src.api import mechanism_compare

comp = mechanism_compare(ideas, k=5)
print(comp.best_method)          # "flow"
print(comp.metrics["flow"])      # {"cosine_diversity": 0.82, ...}
print(comp.rankings)             # per-metric rankings
```

---

## Auction Mechanisms (`src/auction_mechanisms.py`)

### `diversity_auction(items, bidders, diversity_bonus) -> AuctionResult`

Run a sealed-bid second-price auction with diversity bonus. Items are allocated to bidders; bids from bidders whose embeddings are far from already-winning bidders receive a diversity bonus, encouraging heterogeneous winners.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `items` | `List[Item]` | required | Items available for auction |
| `bidders` | `List[Bidder]` | required | Participating bidders with embeddings and budgets |
| `diversity_bonus` | `float` | `0.1` | Multiplicative bonus for diverse bidders `[0, 1]` |

**Returns:** `AuctionResult` — winners, payments, items won, diversity bonuses, total welfare, and total revenue.

```python
from divelicit.src.auction_mechanisms import diversity_auction, Item, Bidder
import numpy as np

items = [
    Item(id="item1", embedding=np.array([1.0, 0.0]), reserve_price=0.5, quality=0.8),
    Item(id="item2", embedding=np.array([0.0, 1.0]), reserve_price=0.3, quality=0.9),
]
bidders = [
    Bidder(id="bidder_a", embedding=np.array([1.0, 0.2]), budget=10.0),
    Bidder(id="bidder_b", embedding=np.array([0.0, 0.8]), budget=10.0),
]
result = diversity_auction(items, bidders, diversity_bonus=0.15)
print(result.winners)            # ["bidder_a", "bidder_b"]
print(result.payments)           # {"bidder_a": 0.5, "bidder_b": 0.3}
print(result.total_welfare)      # combined welfare score
```

---

### `vcg_with_diversity(items, valuations, diversity_fn) -> VCGResult`

Vickrey-Clarke-Groves mechanism with diversity in the welfare function. Computes the welfare-maximizing allocation where welfare includes a diversity contribution measured by `diversity_fn`, then charges each bidder a VCG payment.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `items` | `List[Item]` | required | Items to allocate |
| `valuations` | `Dict[str, Dict[str, float]]` | required | Bidder → item → valuation mapping |
| `diversity_fn` | `Callable[[np.ndarray, List[np.ndarray]], float]` | required | Function measuring diversity contribution of a new embedding given existing ones |

**Returns:** `VCGResult` — allocation, VCG payments, social welfare, diversity welfare, and individual welfare.

```python
from divelicit.src.auction_mechanisms import vcg_with_diversity, Item
import numpy as np

items = [Item(id="x", embedding=np.array([1, 0]), reserve_price=0.0, quality=1.0)]
valuations = {"alice": {"x": 5.0}, "bob": {"x": 3.0}}

def div_fn(new_emb, existing):
    if not existing:
        return 0.0
    return float(min(np.linalg.norm(new_emb - e) for e in existing))

result = vcg_with_diversity(items, valuations, diversity_fn=div_fn)
print(result.allocation)         # {"alice": ["x"]}
print(result.payments)           # VCG payments
print(result.social_welfare)     # welfare including diversity
```

---

### `combinatorial_auction_diverse(lots, bidders) -> CAResult`

Combinatorial auction with diversity bonus. Lots are bundles of items; the mechanism selects non-overlapping lots that maximize welfare plus a coverage bonus reflecting how well the winning lots span the embedding space.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lots` | `List[Lot]` | required | Bundles of items to auction |
| `bidders` | `List[Bidder]` | required | Bidders with embeddings and budgets |

**Returns:** `CAResult` — winning lots, lot-winner mapping, payments, total welfare, and coverage score.

```python
from divelicit.src.auction_mechanisms import combinatorial_auction_diverse, Lot, Item, Bidder
import numpy as np

item_a = Item(id="a", embedding=np.array([1, 0]), reserve_price=0.0, quality=1.0)
item_b = Item(id="b", embedding=np.array([0, 1]), reserve_price=0.0, quality=1.0)
lots = [
    Lot(id="lot1", items=[item_a]),
    Lot(id="lot2", items=[item_b]),
]
bidders = [Bidder(id="b1", embedding=np.array([0.5, 0.5]), budget=10.0)]
result = combinatorial_auction_diverse(lots, bidders)
print(result.winning_lots)       # ["lot1", "lot2"]
print(result.coverage_score)     # spatial coverage metric
```

---

### `ascending_clock_auction(items, bidders, max_rounds, base_increment, diversity_slowdown) -> AscendingResult`

Ascending clock auction with diversity-aware price increments. Prices rise each round; items with few diverse bidders see slower price increases (controlled by `diversity_slowdown`) to preserve heterogeneity.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `items` | `List[Item]` | required | Items for sale |
| `bidders` | `List[Bidder]` | required | Bidders with budgets |
| `max_rounds` | `int` | `100` | Maximum number of clock rounds |
| `base_increment` | `float` | `0.05` | Base price increment per round |
| `diversity_slowdown` | `float` | `0.5` | Factor to slow price rise when bidders are diverse `[0, 1]` |

**Returns:** `AscendingResult` — winners per item, final prices, rounds completed, dropout history, and total revenue.

```python
from divelicit.src.auction_mechanisms import ascending_clock_auction, Item, Bidder
import numpy as np

items = [Item(id="i1", embedding=np.array([1, 0]), reserve_price=1.0, quality=0.9)]
bidders = [
    Bidder(id="b1", embedding=np.array([1, 0.1]), budget=5.0),
    Bidder(id="b2", embedding=np.array([0, 1.0]), budget=5.0),
]
result = ascending_clock_auction(items, bidders, max_rounds=50)
print(result.winners)            # {"i1": "b1"}
print(result.final_prices)       # {"i1": 2.35}
print(result.rounds_completed)   # number of clock rounds
```

---

### `all_pay_contest(entries, judge_fn, diversity_weight, n_winners) -> ContestResult`

All-pay contest with diversity-weighted scoring. Every entrant pays their bid regardless of outcome; winners are selected by a composite score of quality (from `judge_fn`) and diversity contribution.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entries` | `List[Entry]` | required | Contest entries with embeddings and bids |
| `judge_fn` | `Callable[[Entry], float]` | required | Quality scoring function |
| `diversity_weight` | `float` | `0.3` | Weight of diversity in final score `[0, 1]` |
| `n_winners` | `int` | `3` | Number of winners to select |

**Returns:** `ContestResult` — rankings, composite scores, quality scores, diversity scores, prizes, and total effort.

```python
from divelicit.src.auction_mechanisms import all_pay_contest, Entry
import numpy as np

entries = [
    Entry(id="e1", embedding=np.array([1, 0]), bid=2.0, quality_score=0.8),
    Entry(id="e2", embedding=np.array([0, 1]), bid=1.5, quality_score=0.9),
    Entry(id="e3", embedding=np.array([1, 1]), bid=3.0, quality_score=0.7),
]
result = all_pay_contest(entries, judge_fn=lambda e: e.quality_score, n_winners=2)
print(result.rankings)           # ["e2", "e1", "e3"]
print(result.prizes)             # {"e2": 4.33, "e1": 2.17}
print(result.total_effort)       # 6.5
```

---

### `budget_balanced_mechanism(agents, reports, diversity_bonus, quality_scores) -> BBResult`

Budget-balanced (AGV-style) mechanism for diverse reporting. Agents submit vector reports; the mechanism selects a diverse subset and computes transfers such that the sum of transfers is approximately zero.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `List[str]` | required | Agent identifiers |
| `reports` | `Dict[str, np.ndarray]` | required | Agent → report vector mapping |
| `diversity_bonus` | `float` | `0.1` | Bonus for diverse reports |
| `quality_scores` | `Optional[Dict[str, float]]` | `None` | Optional per-agent quality scores |

**Returns:** `BBResult` — selected allocation, transfers, budget surplus, reports used, and diversity score.

```python
from divelicit.src.auction_mechanisms import budget_balanced_mechanism
import numpy as np

agents = ["alice", "bob", "carol"]
reports = {
    "alice": np.array([1.0, 0.0, 0.0]),
    "bob":   np.array([0.0, 1.0, 0.0]),
    "carol": np.array([0.9, 0.1, 0.0]),
}
result = budget_balanced_mechanism(agents, reports, diversity_bonus=0.2)
print(result.allocation)         # ["alice", "bob"]
print(result.transfers)          # {"alice": -0.05, "bob": 0.05}
print(result.budget_surplus)     # ≈ 0.0
```

---

## Information Aggregation (`src/information_aggregation.py`)

### `bayesian_truth_serum(responses, prior) -> BTSResult`

Bayesian Truth Serum (Prelec 2004). Scores respondents by comparing their answers to their predictions about others' answers. Respondents who give "surprisingly common" answers — more frequent than predicted — receive higher information scores.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `responses` | `List[Response]` | required | Respondent answers with predictions of others' answers |
| `prior` | `Optional[Dict[str, float]]` | `None` | Prior distribution over answers; estimated from data if `None` |

**Returns:** `BTSResult` — information scores, estimated truth, confidence, high-information respondents, and answer frequencies.

```python
from divelicit.src.information_aggregation import bayesian_truth_serum, Response

responses = [
    Response(respondent_id="r1", answer="A", prediction={"A": 0.6, "B": 0.4}),
    Response(respondent_id="r2", answer="B", prediction={"A": 0.5, "B": 0.5}),
    Response(respondent_id="r3", answer="A", prediction={"A": 0.7, "B": 0.3}),
]
result = bayesian_truth_serum(responses)
print(result.estimated_truth)    # "A"
print(result.confidence)         # 0.85
print(result.scores)             # {"r1": 0.42, "r2": 0.15, "r3": 0.38}
```

---

### `peer_prediction(responses, reference_fn) -> PeerPredResult`

Peer-prediction mechanism (Miller, Resnick, Zeckhauser 2005). Scores each respondent based on how well their answer predicts a randomly chosen peer's answer. Incentivizes truthful reporting without requiring ground truth.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `responses` | `List[Response]` | required | Respondent answers |
| `reference_fn` | `Optional[Callable[[Response, Response], float]]` | `None` | Custom reference scoring function; uses answer agreement by default |

**Returns:** `PeerPredResult` — scores, adjusted payments, mechanism properties, agreement matrix, and reference scores.

```python
from divelicit.src.information_aggregation import peer_prediction, Response

responses = [
    Response(respondent_id="r1", answer="good", prediction={"good": 0.8, "bad": 0.2}),
    Response(respondent_id="r2", answer="good", prediction={"good": 0.7, "bad": 0.3}),
    Response(respondent_id="r3", answer="bad",  prediction={"good": 0.4, "bad": 0.6}),
]
result = peer_prediction(responses)
print(result.scores)             # {"r1": 0.72, "r2": 0.68, "r3": 0.41}
print(result.mechanism_properties)  # {"incentive_compatible": True, ...}
```

---

### `surprisingly_popular(responses) -> SPResult`

Surprisingly-popular algorithm (Prelec, Seung, McCoy 2017). Identifies the answer whose actual frequency most exceeds its predicted frequency — the answer that is "surprisingly popular" relative to respondents' expectations.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `responses` | `List[Response]` | required | Respondent answers with predictions of the population distribution |

**Returns:** `SPResult` — the surprisingly popular answer, vote tallies, SP scores, confidence, actual vs. predicted frequencies, and frequency gaps.

```python
from divelicit.src.information_aggregation import surprisingly_popular, Response

responses = [
    Response(respondent_id="r1", answer="Philly", prediction={"Philly": 0.4, "NYC": 0.6}),
    Response(respondent_id="r2", answer="NYC",    prediction={"Philly": 0.3, "NYC": 0.7}),
    Response(respondent_id="r3", answer="Philly", prediction={"Philly": 0.3, "NYC": 0.7}),
]
result = surprisingly_popular(responses)
print(result.surprisingly_popular_answer)  # "Philly"
print(result.frequency_gaps)     # {"Philly": 0.33, "NYC": -0.33}
print(result.confidence)         # strength of the SP signal
```

---

### `wisdom_of_experts(expert_opinions, expertise_weights) -> AggregateResult`

Weighted aggregation of expert point estimates. Combines estimates via weighted mean, trimmed mean, median, and log-linear pooling. Detects outliers and computes confidence intervals.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expert_opinions` | `List[ExpertOpinion]` | required | Expert estimates with optional confidence intervals |
| `expertise_weights` | `Optional[Dict[str, float]]` | `None` | Custom expert weights; uses uniform or `expertise_score` if `None` |

**Returns:** `AggregateResult` — weighted mean, trimmed mean, median, linear pool estimate, confidence interval, disagreement, outlier IDs, and weights used.

```python
from divelicit.src.information_aggregation import wisdom_of_experts, ExpertOpinion

opinions = [
    ExpertOpinion(expert_id="e1", estimate=42.0, expertise_score=0.9),
    ExpertOpinion(expert_id="e2", estimate=38.0, expertise_score=0.7),
    ExpertOpinion(expert_id="e3", estimate=55.0, expertise_score=0.4),
]
result = wisdom_of_experts(opinions)
print(result.weighted_mean)      # 42.6
print(result.outlier_ids)        # ["e3"]
print(result.confidence_interval)  # (37.5, 47.7)
```

---

### `calibrated_aggregation(forecasts, calibration_data) -> CalibratedForecast`

Aggregate probability forecasts with calibration adjustment. Recalibrates individual forecasters using isotonic regression (when historical data is available), then pools calibrated probabilities.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecasts` | `List[Forecast]` | required | Forecaster probability estimates for a binary event |
| `calibration_data` | `Optional[Dict[str, Tuple[List[float], List[int]]]]` | `None` | Per-forecaster `(past_forecasts, past_outcomes)` for recalibration |

**Returns:** `CalibratedForecast` — aggregated probability, per-forecaster calibrated values, calibration scores (Brier, log-loss), aggregate Brier score, and whether recalibration was applied.

```python
from divelicit.src.information_aggregation import calibrated_aggregation, Forecast

forecasts = [
    Forecast(forecaster_id="f1", probability=0.7, past_forecasts=[0.6, 0.8], past_outcomes=[1, 1]),
    Forecast(forecaster_id="f2", probability=0.5, past_forecasts=[0.5, 0.5], past_outcomes=[0, 1]),
]
result = calibrated_aggregation(forecasts)
print(result.aggregated_probability)  # 0.62
print(result.calibration_scores)      # per-forecaster Brier and log-loss
print(result.recalibrated)            # True if historical data was used
```

---

### `extremize_forecasts(forecasts, factor) -> List[float]`

Super-forecasting extremization (Tetlock & Gardner 2015). Pushes aggregated forecasts away from 0.5 toward 0 or 1, compensating for the well-documented tendency of forecast averaging to be insufficiently extreme.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecasts` | `Sequence[float]` | required | Probability forecasts in `[0, 1]` |
| `factor` | `float` | `2.5` | Extremization factor; higher values push further from 0.5 |

**Returns:** `List[float]` — extremized probability forecasts.

```python
from divelicit.src.information_aggregation import extremize_forecasts

probs = [0.6, 0.55, 0.7, 0.45]
extremized = extremize_forecasts(probs, factor=2.0)
print(extremized)  # [0.69, 0.60, 0.82, 0.36]
```

---

## Strategic Diversity (`src/strategic_diversity.py`)

### `nash_diverse_equilibrium(agents, strategies) -> NashResult`

Find Nash equilibria of the diversity game. Constructs a normal-form game where agents choose strategies with features; payoffs combine private value and pairwise diversity. Solves via support enumeration and fictitious play.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `List[Agent]` | required | Agents with type vectors and available strategies |
| `strategies` | `List[Strategy]` | required | Strategies with feature vectors |

**Returns:** `NashResult` — equilibrium mixed strategies, expected payoffs, diversity score, whether the solution is approximate, and support sizes.

```python
from divelicit.src.strategic_diversity import nash_diverse_equilibrium, Agent, Strategy
import numpy as np

strategies = [
    Strategy(name="safe", index=0, features=np.array([1.0, 0.0])),
    Strategy(name="risky", index=1, features=np.array([0.0, 1.0])),
]
agents = [
    Agent(id=0, type_vector=np.array([0.8, 0.2]), available_strategies=strategies),
    Agent(id=1, type_vector=np.array([0.3, 0.7]), available_strategies=strategies),
]
result = nash_diverse_equilibrium(agents, strategies)
print(result.equilibrium_strategies)  # mixed strategy profiles
print(result.diversity_score)         # diversity at equilibrium
print(result.is_approximate)          # True if found via fictitious play
```

---

### `correlated_equilibrium_diverse(game, correlation_device) -> CorrelatedResult`

Compute a correlated equilibrium that maximizes diversity. Solves a linear program over joint strategy distributions subject to no-unilateral-deviation constraints, maximizing expected diversity welfare.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `game` | `GamePayoff` | required | Payoff structure with player count, strategy counts, and payoff matrices |
| `correlation_device` | `Optional[NDArray[np.float64]]` | `None` | Initial joint distribution; uses uniform if `None` |

**Returns:** `CorrelatedResult` — joint distribution, expected welfare, deviation incentives, feasibility flag, and max constraint violation.

```python
from divelicit.src.strategic_diversity import correlated_equilibrium_diverse, GamePayoff
import numpy as np

game = GamePayoff(
    n_players=2,
    n_strategies=[2, 2],
    matrices=[
        np.array([[3, 0], [5, 1]]),  # player 0 payoffs
        np.array([[3, 5], [0, 1]]),  # player 1 payoffs
    ],
)
result = correlated_equilibrium_diverse(game)
print(result.joint_distribution)  # 2x2 probability matrix
print(result.expected_welfare)    # expected social welfare
print(result.is_feasible)         # True
```

---

### `mechanism_design_for_diversity(agent_types, social_welfare_fn) -> Mechanism`

Design a direct-revelation mechanism for diversity maximization. Constructs allocation and payment rules (Groves transfers) for a mechanism where agents report their types and the allocation maximizes a diversity-aware social welfare function.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_types` | `List[NDArray[np.float64]]` | required | Type vectors for each agent |
| `social_welfare_fn` | `Callable[[Dict[int, int], List[NDArray[np.float64]]], float]` | required | Welfare function mapping (allocation, types) → welfare |

**Returns:** `Mechanism` — a direct-revelation mechanism with allocation rule, payment rule, and metadata.

```python
from divelicit.src.strategic_diversity import mechanism_design_for_diversity
import numpy as np

types = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

def welfare(alloc, reported_types):
    used = [reported_types[i] for i in alloc.values()]
    return sum(np.linalg.norm(a - b) for a in used for b in used)

mech = mechanism_design_for_diversity(types, welfare)
print(mech.description)          # mechanism description
print(mech.n_agents)             # 3
```

---

### `revelation_principle_check(mechanism) -> bool`

Verify that a mechanism satisfies the revelation principle. Tests whether truth-telling is a weakly dominant strategy by checking that no agent can improve their payoff by misreporting.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mechanism` | `Mechanism` | required | Mechanism to verify |

**Returns:** `bool` — `True` if the mechanism satisfies the revelation principle.

```python
from divelicit.src.strategic_diversity import revelation_principle_check

is_valid = revelation_principle_check(mech)
print(is_valid)  # True
```

---

### `incentive_compatibility_test(mechanism, agent_strategies) -> bool`

Test if a mechanism is dominant-strategy incentive compatible (DSIC). For each agent, checks all alternative strategies (misreports) and verifies that truthful reporting always yields at least as high a payoff.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mechanism` | `Mechanism` | required | Mechanism to test |
| `agent_strategies` | `List[List[NDArray[np.float64]]]` | required | Per-agent list of possible type reports (first element is truthful) |

**Returns:** `bool` — `True` if truth-telling is dominant for every agent.

```python
from divelicit.src.strategic_diversity import incentive_compatibility_test
import numpy as np

strategies = [
    [np.array([1, 0]), np.array([0, 1])],  # agent 0: true type, alternative
    [np.array([0, 1]), np.array([1, 0])],  # agent 1: true type, alternative
]
is_dsic = incentive_compatibility_test(mech, strategies)
print(is_dsic)  # True
```

---

### `implement_diverse_allocation(mechanism, reports) -> Allocation`

Execute a mechanism given agent reports. Runs the allocation rule and payment rule to produce an assignment of agents to outcomes along with monetary transfers.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mechanism` | `Mechanism` | required | Mechanism to execute |
| `reports` | `List[NDArray[np.float64]]` | required | Agent type reports |

**Returns:** `Allocation` — assignments, transfers, total welfare, and budget balance flag.

```python
from divelicit.src.strategic_diversity import implement_diverse_allocation
import numpy as np

reports = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
alloc = implement_diverse_allocation(mech, reports)
print(alloc.assignments)         # {0: 0, 1: 1, 2: 2}
print(alloc.transfers)           # {0: -0.1, 1: -0.1, 2: -0.1}
print(alloc.is_budget_balanced)  # True
```

---

## Deliberation (`src/deliberation.py`)

### `structured_debate(topic, participants, rounds, dim, rng_seed) -> DebateResult`

Multi-round structured debate among participants. Each round, participants generate arguments (with Toulmin components), update positions based on novel arguments, and build an argument graph capturing supports and attacks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topic` | `str` | required | Debate topic |
| `participants` | `List[Participant]` | required | Debate participants with initial positions |
| `rounds` | `int` | `5` | Number of debate rounds |
| `dim` | `int` | `16` | Embedding dimension for arguments |
| `rng_seed` | `int` | `42` | Random seed for reproducibility |

**Returns:** `DebateResult` — final positions, argument graph, consensus measure, per-round data, position trajectories, and argument count.

```python
from divelicit.src.deliberation import structured_debate, Participant, Position
import numpy as np

participants = [
    Participant(id="p1", name="Alice", position=Position(vector=np.random.randn(16), conviction=0.8, label="pro")),
    Participant(id="p2", name="Bob",   position=Position(vector=np.random.randn(16), conviction=0.6, label="con")),
    Participant(id="p3", name="Carol", position=Position(vector=np.random.randn(16), conviction=0.5, label="neutral")),
]
result = structured_debate("Should AI be regulated?", participants, rounds=3)
print(result.consensus_measure)          # 0.72
print(result.argument_graph.n_arguments) # total arguments generated
print(result.total_arguments)            # 9
```

---

### `socratic_dialogue(question, participants, max_iterations, challenge_threshold, convergence_threshold) -> DialogueResult`

Simulated Socratic dialogue among participants. Iteratively selects a questioner who challenges the participant with the most different position. Positions update as participants respond to challenges; converges when position drift falls below threshold.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | `str` | required | Central question for dialogue |
| `participants` | `List[Participant]` | required | Dialogue participants |
| `max_iterations` | `int` | `50` | Maximum dialogue iterations |
| `challenge_threshold` | `float` | `0.3` | Minimum position distance to trigger a challenge |
| `convergence_threshold` | `float` | `0.01` | Position change below which dialogue converges |

**Returns:** `DialogueResult` — final positions, position evolution over iterations, questioner sequence, convergence iteration, final spread, and total challenges.

```python
from divelicit.src.deliberation import socratic_dialogue, Participant, Position
import numpy as np

participants = [
    Participant(id="s1", name="Socrates", position=Position(vector=np.array([1.0, 0.0, 0.5]), conviction=0.9, label="inquiry")),
    Participant(id="s2", name="Thrasymachus", position=Position(vector=np.array([0.0, 1.0, -0.5]), conviction=0.7, label="power")),
]
result = socratic_dialogue("What is justice?", participants, max_iterations=30)
print(result.convergence_iteration)  # 18
print(result.challenges_made)        # 12
print(result.final_spread)           # position spread at convergence
```

---

### `argument_mapping(statements, similarity_threshold, attack_threshold, embedding_dim) -> ArgumentMap`

Build a Toulmin argument map from a list of statements. Computes pairwise similarity between statement embeddings; creates support edges above `similarity_threshold` and attack edges below `attack_threshold`. Identifies strongest, most central, and key rebuttal arguments.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `statements` | `List[Statement]` | required | Statements with Toulmin components (claim, grounds, warrant, etc.) |
| `similarity_threshold` | `float` | `0.3` | Cosine similarity above which a support edge is created |
| `attack_threshold` | `float` | `-0.2` | Cosine similarity below which an attack edge is created |
| `embedding_dim` | `int` | `32` | Embedding dimension for statement vectors |

**Returns:** `ArgumentMap` — arguments dict, support/attack edges with weights, strongest arguments, central claims, and key rebuttals.

```python
from divelicit.src.deliberation import argument_mapping, Statement

statements = [
    Statement(claim="AI improves healthcare", grounds="Diagnostic accuracy", warrant="Data-driven decisions are better", backing="Studies show 95% accuracy", qualifier=0.8, rebuttal="Bias in training data"),
    Statement(claim="AI risks job loss", grounds="Automation replaces workers", warrant="Efficiency drives adoption", backing="Historical precedent", qualifier=0.7, rebuttal="New jobs are created"),
]
arg_map = argument_mapping(statements)
print(arg_map.n_arguments)       # 2
print(arg_map.strongest_arguments)  # top arguments by strength
print(arg_map.support_edges)     # [(src_id, tgt_id, weight), ...]
```

---

### `consensus_building(positions, mediator_fn, max_rounds, convergence_threshold, holdout_threshold) -> ConsensusResult`

Find consensus among diverse positions via iterative compromise. Each round, positions move toward the centroid (or a mediator-proposed target); positions that remain far from consensus after convergence are flagged as holdouts.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `positions` | `List[Position]` | required | Initial participant positions |
| `mediator_fn` | `Optional[Callable[[List[Position], np.ndarray], np.ndarray]]` | `None` | Custom mediator function; uses centroid if `None` |
| `max_rounds` | `int` | `100` | Maximum negotiation rounds |
| `convergence_threshold` | `float` | `0.01` | Position change threshold for convergence |
| `holdout_threshold` | `float` | `0.1` | Distance from consensus above which a position is a holdout |

**Returns:** `ConsensusResult` — final positions, agreement level, convergence round, holdout positions, position history, and centroid history.

```python
from divelicit.src.deliberation import consensus_building, Position
import numpy as np

positions = [
    Position(vector=np.array([1.0, 0.0]), conviction=0.8, label="left"),
    Position(vector=np.array([0.0, 1.0]), conviction=0.6, label="right"),
    Position(vector=np.array([0.5, 0.5]), conviction=0.5, label="center"),
]
result = consensus_building(positions, max_rounds=50)
print(result.agreement_level)    # 0.93
print(result.convergence_round)  # 22
print(len(result.holdout_positions))  # 0
```

---

### `deliberative_polling(topic, participants, n_groups, information_shift, deliberation_rounds, rng_seed) -> PollResult`

Simulated deliberative polling following the Fishkin method. Participants are assigned to diverse small groups, exposed to balanced information (modeled as a position shift), then deliberate in groups over multiple rounds.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topic` | `str` | required | Polling topic |
| `participants` | `List[Participant]` | required | Poll participants with initial positions |
| `n_groups` | `int` | `4` | Number of deliberation groups |
| `information_shift` | `float` | `0.15` | Magnitude of information-induced position shift |
| `deliberation_rounds` | `int` | `5` | Rounds of within-group deliberation |
| `rng_seed` | `int` | `42` | Random seed |

**Returns:** `PollResult` — pre/post positions, opinion change per participant, polarization before/after, information gain, group assignments, and phase positions.

```python
from divelicit.src.deliberation import deliberative_polling, Participant, Position
import numpy as np

participants = [
    Participant(id=f"p{i}", name=f"Person {i}",
                position=Position(vector=np.random.randn(8), conviction=0.5, label="voter"))
    for i in range(20)
]
result = deliberative_polling("Universal basic income", participants, n_groups=4)
print(result.polarization_before)  # 0.65
print(result.polarization_after)   # 0.42
print(result.information_gain)     # 0.18
```

---

### `citizens_assembly(issues, participants, assembly_size, n_groups, expert_rounds, deliberation_rounds, voting_threshold, dim, rng_seed) -> AssemblyResult`

Simulated citizens' assembly with sortition and structured deliberation. Randomly selects participants (sortition), assigns them to diverse groups, runs expert information rounds followed by deliberation rounds, then votes on each issue.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `issues` | `List[str]` | required | Issues to deliberate |
| `participants` | `List[Participant]` | required | Candidate pool for sortition |
| `assembly_size` | `int` | `0` | Number to select (0 = all participants) |
| `n_groups` | `int` | `4` | Number of deliberation groups |
| `expert_rounds` | `int` | `2` | Number of expert briefing rounds |
| `deliberation_rounds` | `int` | `5` | Number of group deliberation rounds |
| `voting_threshold` | `float` | `0.6` | Supermajority threshold for recommendations |
| `dim` | `int` | `8` | Position vector dimension |
| `rng_seed` | `int` | `42` | Random seed |

**Returns:** `AssemblyResult` — recommendations per issue, confidence levels, minority reports, selected participants, voting results, group assignments, and position trajectories.

```python
from divelicit.src.deliberation import citizens_assembly, Participant, Position
import numpy as np

participants = [
    Participant(id=f"c{i}", name=f"Citizen {i}",
                position=Position(vector=np.random.randn(8), conviction=0.5, label="citizen"))
    for i in range(50)
]
result = citizens_assembly(
    issues=["Climate policy", "Housing reform"],
    participants=participants,
    assembly_size=20,
    voting_threshold=0.6,
)
print(result.recommendations)       # per-issue recommendation vectors
print(result.confidence_levels)     # {"Climate policy": 0.78, ...}
print(result.voting_results)        # detailed per-issue vote breakdown
```

---

## Portfolio Optimization (`src/portfolio_optimization.py`)

### `diverse_portfolio(assets, returns, risk, diversity_target) -> Portfolio`

Construct a portfolio maximizing return subject to risk and diversity constraints. Uses projected gradient descent to find weights that satisfy both a risk ceiling and a minimum diversity level (measured by effective number of bets).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `assets` | `Sequence[Asset]` | required | Investable assets with expected returns and sectors |
| `returns` | `NDArray[np.float64]` | required | Historical return matrix (time × assets) |
| `risk` | `float` | required | Maximum portfolio volatility |
| `diversity_target` | `float` | `0.5` | Minimum diversity score (0 = concentrated, 1 = equal weight) |

**Returns:** `Portfolio` — weights, expected return, risk, diversity score, and asset names.

```python
from divelicit.src.portfolio_optimization import diverse_portfolio, Asset
import numpy as np

assets = [
    Asset(name="AAPL", expected_return=0.12, market_cap_weight=0.3, sector="tech"),
    Asset(name="JPM",  expected_return=0.08, market_cap_weight=0.2, sector="finance"),
    Asset(name="XOM",  expected_return=0.06, market_cap_weight=0.15, sector="energy"),
]
returns = np.random.randn(252, 3) * 0.02  # daily returns
port = diverse_portfolio(assets, returns, risk=0.15, diversity_target=0.6)
print(port.weights)              # [0.45, 0.30, 0.25]
print(port.diversity_score)      # 0.62
print(port.expected_return)      # 0.094
```

---

### `markowitz_with_diversity(assets, cov_matrix, expected_returns, risk_aversion, diversity_penalty) -> Portfolio`

Mean-variance optimization with a diversity regularizer. Adds a penalty term proportional to the HHI (Herfindahl-Hirschman Index) of the portfolio weights, discouraging concentration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `assets` | `Sequence[Asset]` | required | Investable assets |
| `cov_matrix` | `NDArray[np.float64]` | required | Asset covariance matrix |
| `expected_returns` | `Optional[NDArray[np.float64]]` | `None` | Expected returns; uses `Asset.expected_return` if `None` |
| `risk_aversion` | `float` | `2.5` | Risk aversion parameter |
| `diversity_penalty` | `float` | `0.1` | Penalty weight on portfolio concentration |

**Returns:** `Portfolio` — optimized weights with diversity regularization.

```python
from divelicit.src.portfolio_optimization import markowitz_with_diversity, Asset
import numpy as np

assets = [Asset(name=n, expected_return=r, market_cap_weight=0.25, sector=s)
          for n, r, s in [("A", 0.10, "tech"), ("B", 0.08, "health"),
                          ("C", 0.06, "energy"), ("D", 0.09, "finance")]]
cov = np.eye(4) * 0.04
port = markowitz_with_diversity(assets, cov, diversity_penalty=0.2)
print(port.weights)              # near-equal weights due to penalty
print(port.diversity_score)      # high diversity score
```

---

### `risk_parity_diverse(assets, cov_matrix) -> Portfolio`

Risk-parity portfolio: each asset contributes equally to total risk. Solves for weights where the marginal risk contribution of every asset is identical, naturally producing diversified portfolios.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `assets` | `Sequence[Asset]` | required | Investable assets |
| `cov_matrix` | `NDArray[np.float64]` | required | Asset covariance matrix |

**Returns:** `Portfolio` — risk-parity weights with per-asset risk contributions.

```python
from divelicit.src.portfolio_optimization import risk_parity_diverse, Asset
import numpy as np

assets = [Asset(name=n, expected_return=0.08, market_cap_weight=0.25, sector="mixed")
          for n in ["A", "B", "C", "D"]]
cov = np.array([[0.04, 0.01, 0.0, 0.0],
                [0.01, 0.09, 0.02, 0.0],
                [0.0,  0.02, 0.16, 0.0],
                [0.0,  0.0,  0.0,  0.01]])
port = risk_parity_diverse(assets, cov)
print(port.weights)              # more weight to low-vol assets
print(port.risk_contributions)   # approximately equal across assets
```

---

### `black_litterman_diverse(market_data, views, diversity_bonus) -> Portfolio`

Black-Litterman model with diversity bonus. Combines market-implied equilibrium returns with investor views via Bayesian updating, then adds a diversity bonus to underweighted asset classes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `market_data` | `MarketData` | required | Market-wide data (assets, covariance, cap weights, risk-free rate) |
| `views` | `Sequence[View]` | required | Investor views with pick vectors and confidence |
| `diversity_bonus` | `float` | `0.1` | Bonus for underweighted assets to encourage diversification |

**Returns:** `Portfolio` — posterior weights, blended returns, and posterior covariance.

```python
from divelicit.src.portfolio_optimization import black_litterman_diverse, build_market_data, View
import numpy as np

md = build_market_data(
    names=["US Equity", "Intl Equity", "Bonds"],
    expected_returns=np.array([0.08, 0.06, 0.03]),
    cov_matrix=np.eye(3) * 0.04,
    market_cap_weights=np.array([0.5, 0.3, 0.2]),
)
views = [View(pick_vector=np.array([1, -1, 0]), expected_return=0.02, confidence=0.5)]
port = black_litterman_diverse(md, views, diversity_bonus=0.15)
print(port.weights)              # tilted toward views + diversity
print(port.blended_returns)      # posterior expected returns
```

---

### `rebalance_for_diversity(current_portfolio, target_diversity, transaction_cost) -> Trades`

Compute minimum-cost trades to reach a target diversity level. Iteratively adjusts portfolio weights to increase diversity (measured by effective number of bets) while minimizing transaction costs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `current_portfolio` | `Portfolio` | required | Current portfolio to rebalance |
| `target_diversity` | `float` | required | Target diversity score to achieve |
| `transaction_cost` | `float` | `0.001` | Cost per unit traded (e.g., 10 bps) |

**Returns:** `Trades` — list of trades, total cost, new diversity score, and old diversity score.

```python
from divelicit.src.portfolio_optimization import rebalance_for_diversity, Portfolio
import numpy as np

current = Portfolio(
    weights=np.array([0.8, 0.1, 0.1]),
    expected_return=0.10,
    risk=0.15,
    diversity_score=0.35,
    asset_names=["AAPL", "GOOG", "MSFT"],
)
trades = rebalance_for_diversity(current, target_diversity=0.7, transaction_cost=0.002)
print(trades.summary())         # human-readable trade list
print(trades.total_cost)        # transaction cost
print(trades.new_diversity)     # achieved diversity
```

---

## Experiment Design (`src/experiment_design.py`)

### `diverse_experiment_design(factors, levels, n_runs) -> DesignMatrix`

Generate an experimental design that maximizes space-filling diversity. Uses coordinate exchange to optimize D-efficiency and space-filling properties simultaneously, supporting factorial, fractional-factorial, and custom designs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `factors` | `List[Factor]` | required | Experimental factors with levels and bounds |
| `levels` | `Optional[List[List[Level]]]` | `None` | Override factor levels; uses `factor.levels` if `None` |
| `n_runs` | `Optional[int]` | `None` | Number of runs; computed from factors if `None` |

**Returns:** `DesignMatrix` — design matrix, factor names, D-efficiency, space-filling metric, and design type.

```python
from divelicit.src.experiment_design import diverse_experiment_design, Factor, Level

factors = [
    Factor(name="Temperature", levels=[Level("low", 20), Level("high", 80)], lower=20, upper=80),
    Factor(name="Pressure",    levels=[Level("low", 1),  Level("high", 5)],  lower=1,  upper=5),
    Factor(name="Catalyst",    levels=[Level("A", 0),    Level("B", 1)],     lower=0,  upper=1),
]
design = diverse_experiment_design(factors, n_runs=12)
print(design.matrix.shape)       # (12, 3)
print(design.d_efficiency)       # D-efficiency score
print(design.space_filling_metric)  # maximin distance
```

---

### `latin_hypercube_diverse(n_dims, n_samples, seed) -> np.ndarray`

Generate a maximin Latin Hypercube Design via simulated annealing. Starts from a random LHD and iteratively swaps elements within columns to maximize the minimum pairwise distance.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_dims` | `int` | required | Number of dimensions (factors) |
| `n_samples` | `int` | required | Number of sample points |
| `seed` | `int` | `42` | Random seed |

**Returns:** `np.ndarray` — design matrix of shape `(n_samples, n_dims)` with values in `[0, 1]`.

```python
from divelicit.src.experiment_design import latin_hypercube_diverse

design = latin_hypercube_diverse(n_dims=5, n_samples=20, seed=0)
print(design.shape)  # (20, 5)
print(design.min(), design.max())  # values in [0, 1]
```

---

### `space_filling_design(bounds, n_points, method) -> np.ndarray`

Generate a space-filling design within rectangular bounds. Supports maximin (maximizes minimum pairwise distance), minimax (minimizes maximum distance to nearest design point), and uniform random methods.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bounds` | `np.ndarray` | required | `(n_dims, 2)` array of `[lower, upper]` per dimension |
| `n_points` | `int` | required | Number of design points |
| `method` | `Literal['maximin', 'minimax', 'uniform']` | `'maximin'` | Space-filling criterion |

**Returns:** `np.ndarray` — design matrix of shape `(n_points, n_dims)`.

```python
from divelicit.src.experiment_design import space_filling_design
import numpy as np

bounds = np.array([[0, 10], [0, 5], [-1, 1]])
design = space_filling_design(bounds, n_points=30, method="maximin")
print(design.shape)  # (30, 3)
```

---

### `sequential_design(model, acquisition_fn, diversity_penalty, bounds, n_random, n_local) -> NextPoint`

Recommend the next evaluation point using acquisition function + diversity penalty. Searches the design space for the point that maximizes acquisition value minus a diversity penalty (distance to nearest existing observation).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `SurrogateModel` | required | Fitted GP surrogate model |
| `acquisition_fn` | `Literal['ei', 'ucb', 'pi']` | `'ei'` | Acquisition function: expected improvement, upper confidence bound, or probability of improvement |
| `diversity_penalty` | `float` | `0.1` | Weight of diversity in acquisition score |
| `bounds` | `Optional[np.ndarray]` | `None` | Search bounds; inferred from training data if `None` |
| `n_random` | `int` | `5000` | Number of random candidate points |
| `n_local` | `int` | `50` | Number of local optimization restarts |

**Returns:** `NextPoint` — recommended point, acquisition value, and diversity contribution.

```python
from divelicit.src.experiment_design import sequential_design, SurrogateModel, rbf_kernel
import numpy as np

X = np.random.randn(10, 2)
y = np.sum(X**2, axis=1)
kernel_fn = lambda X1, X2: rbf_kernel(X1, X2, lengthscale=1.0)
# (Construct SurrogateModel from fitted GP state)
# next_pt = sequential_design(model, acquisition_fn="ei", diversity_penalty=0.2)
# print(next_pt.point)            # recommended next experiment
# print(next_pt.acquisition_value)
```

---

### `bayesian_optimization_diverse(objective, bounds, n_calls, n_initial, diversity_weight, lengthscale, kernel_variance, noise, seed) -> OptResult`

Bayesian optimization loop with diversity-aware acquisition. Runs a full BO loop: initial random evaluations, then iteratively fits a GP surrogate and selects the next point using a diversity-penalized acquisition function.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `objective` | `Callable[[np.ndarray], float]` | required | Black-box objective function to minimize |
| `bounds` | `np.ndarray` | required | `(n_dims, 2)` search bounds |
| `n_calls` | `int` | `50` | Total function evaluations |
| `n_initial` | `int` | `5` | Initial random evaluations before BO |
| `diversity_weight` | `float` | `0.1` | Weight of diversity in acquisition |
| `lengthscale` | `float` | `1.0` | GP kernel lengthscale |
| `kernel_variance` | `float` | `1.0` | GP kernel variance |
| `noise` | `float` | `1e-6` | Observation noise |
| `seed` | `int` | `42` | Random seed |

**Returns:** `OptResult` — best point, best value, all evaluations, fitted surrogate model, and convergence history.

```python
from divelicit.src.experiment_design import bayesian_optimization_diverse
import numpy as np

def sphere(x):
    return float(np.sum(x**2))

bounds = np.array([[-5, 5], [-5, 5]])
result = bayesian_optimization_diverse(sphere, bounds, n_calls=30, diversity_weight=0.2)
print(result.best_point)         # near [0, 0]
print(result.best_value)         # near 0.0
print(len(result.convergence))   # 30 best-so-far values
```

---

## Collective Intelligence (`src/collective_intelligence.py`)

### `collective_problem_solving(problem, agents, method, n_iterations, rng) -> Solution`

Solve an optimization problem collectively using diverse agents. Each agent explores from its own perspective; solutions are shared and recombined. Supports `"diverse"` (diversity-maintained swarm) and `"independent"` (parallel restarts) methods.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `problem` | `Problem` | required | Optimization problem with objective and bounds |
| `agents` | `List[SwarmAgent]` | required | Agents with positions, velocities, and perspectives |
| `method` | `str` | `"diverse"` | Solving method: `"diverse"` or `"independent"` |
| `n_iterations` | `int` | `50` | Number of iterations |
| `rng` | `np.random.Generator \| None` | `None` | Random number generator |

**Returns:** `Solution` — best position, best value, diversity metric, convergence history, and all final solutions.

```python
from divelicit.src.collective_intelligence import collective_problem_solving, Problem, SwarmAgent
import numpy as np

problem = Problem(
    objective=lambda x: float(np.sum(x**2)),
    bounds=[(-5, 5), (-5, 5)],
    description="Minimize sphere function",
)
agents = [SwarmAgent(position=np.random.randn(2), velocity=np.zeros(2),
                     perspective=np.random.randn(2), agent_type="explorer")
          for _ in range(10)]
sol = collective_problem_solving(problem, agents, method="diverse", n_iterations=100)
print(sol.best_value)            # near 0.0
print(sol.diversity)             # swarm diversity at termination
```

---

### `idea_marketplace(ideas, evaluators, n_rounds, decay, rng) -> MarketplaceResult`

Simulate a double-auction idea marketplace. Ideas have initial quality scores and embeddings; evaluators (represented as preference vectors) buy/sell ideas. Market prices converge to reflect collective quality assessment, and undervalued diverse ideas are identified.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ideas` | `List[Idea]` | required | Ideas with embeddings and initial quality |
| `evaluators` | `List[np.ndarray]` | required | Evaluator preference vectors |
| `n_rounds` | `int` | `50` | Number of trading rounds |
| `decay` | `float` | `0.1` | Price update decay rate |
| `rng` | `np.random.Generator \| None` | `None` | Random number generator |

**Returns:** `MarketplaceResult` — rankings, final prices, undervalued idea IDs, market efficiency, and price history.

```python
from divelicit.src.collective_intelligence import idea_marketplace, Idea
import numpy as np

ideas = [
    Idea(idea_id=0, embedding=np.array([1, 0, 0]), initial_quality=0.7, description="Solar roads"),
    Idea(idea_id=1, embedding=np.array([0, 1, 0]), initial_quality=0.5, description="Vertical farms"),
    Idea(idea_id=2, embedding=np.array([0, 0, 1]), initial_quality=0.9, description="Carbon capture"),
]
evaluators = [np.array([0.5, 0.3, 0.2]), np.array([0.1, 0.6, 0.3])]
result = idea_marketplace(ideas, evaluators, n_rounds=30)
print(result.rankings)           # [2, 0, 1]
print(result.undervalued)        # ideas priced below their quality
print(result.market_efficiency)  # correlation between price and quality
```

---

### `innovation_tournament(challenges, participants, n_stages, n_iterations_per_stage, rng) -> TournamentResult`

Run a multi-stage innovation tournament. Participants compete across challenges in stages; each stage filters the top performers while preserving diversity. Scores combine solution quality and novelty.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `challenges` | `List[Challenge]` | required | Tournament challenges with objectives and bounds |
| `participants` | `List[SwarmAgent]` | required | Tournament participants |
| `n_stages` | `int` | `3` | Number of elimination stages |
| `n_iterations_per_stage` | `int` | `20` | Optimization iterations per stage |
| `rng` | `np.random.Generator \| None` | `None` | Random number generator |

**Returns:** `TournamentResult` — winners with scores, per-stage scores, innovation quality, diversity of winners, and winning solutions.

```python
from divelicit.src.collective_intelligence import innovation_tournament, Challenge, SwarmAgent
import numpy as np

challenges = [
    Challenge(challenge_id=0, objective=lambda x: -float(np.sum(x**2)),
              bounds=[(-5, 5), (-5, 5)], description="Minimize quadratic"),
]
participants = [SwarmAgent(position=np.random.randn(2), velocity=np.zeros(2),
                           perspective=np.random.randn(2), agent_type="innovator")
                for _ in range(20)]
result = innovation_tournament(challenges, participants, n_stages=3)
print(result.winners)            # [(agent_idx, score), ...]
print(result.diversity_of_winners)  # diversity among top performers
```

---

### `crowd_labeling_diverse(items, labelers, n_classes, diversity_req, labeler_types, confidence_threshold) -> Labels`

Diverse crowd labeling using Dawid-Skene EM. Aggregates noisy labels from multiple annotators using EM to estimate true labels and annotator quality. Ensures labeler diversity by requiring minimum type representation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `items` | `List[LabelItem]` | required | Items to label |
| `labelers` | `dict[int, dict[int, int]]` | required | Labeler → item → label mapping |
| `n_classes` | `int` | `2` | Number of label classes |
| `diversity_req` | `float` | `0.3` | Minimum fraction of each labeler type |
| `labeler_types` | `dict[int, str] \| None` | `None` | Labeler type assignments (e.g., "expert", "crowd") |
| `confidence_threshold` | `float` | `0.8` | Minimum posterior probability for confident labels |

**Returns:** `Labels` — estimated labels, labeler quality scores, inter-annotator agreement, confident items, and label probabilities.

```python
from divelicit.src.collective_intelligence import crowd_labeling_diverse, LabelItem

items = [LabelItem(item_id=i, true_label=i % 2) for i in range(10)]
labelers = {
    0: {0: 0, 1: 1, 2: 0, 3: 1},  # labeler 0's labels
    1: {0: 0, 1: 1, 2: 1, 3: 1},  # labeler 1's labels
    2: {0: 0, 1: 0, 2: 0, 3: 1},  # labeler 2's labels
}
result = crowd_labeling_diverse(items, labelers, n_classes=2)
print(result.estimated_labels)         # {0: 0, 1: 1, 2: 0, 3: 1}
print(result.labeler_quality)          # {0: 0.95, 1: 0.80, 2: 0.85}
print(result.inter_annotator_agreement)  # Cohen's kappa
```

---

### `swarm_intelligence(objective, agents, n_iterations, w, c1, c2, diversity_threshold, repulsion_strength, bounds, rng) -> SwarmResult`

Particle swarm optimization with diversity maintenance. Standard PSO with an added repulsion mechanism: when swarm diversity drops below `diversity_threshold`, particles are repelled from each other to maintain exploration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `objective` | `Callable[[np.ndarray], float]` | required | Objective function to minimize |
| `agents` | `List[SwarmAgent]` | required | Swarm particles with positions and velocities |
| `n_iterations` | `int` | `100` | Number of PSO iterations |
| `w` | `float` | `0.7` | Inertia weight |
| `c1` | `float` | `1.5` | Cognitive (personal best) coefficient |
| `c2` | `float` | `1.5` | Social (global best) coefficient |
| `diversity_threshold` | `float \| None` | `None` | Diversity below which repulsion activates |
| `repulsion_strength` | `float` | `0.5` | Strength of diversity-maintaining repulsion |
| `bounds` | `Sequence[Tuple[float, float]] \| None` | `None` | Per-dimension bounds |
| `rng` | `np.random.Generator \| None` | `None` | Random number generator |

**Returns:** `SwarmResult` — best position, best value, convergence history, diversity history, and final particle positions.

```python
from divelicit.src.collective_intelligence import swarm_intelligence, SwarmAgent
import numpy as np

agents = [SwarmAgent(position=np.random.uniform(-5, 5, 3), velocity=np.zeros(3),
                     perspective=np.random.randn(3), agent_type="particle")
          for _ in range(20)]
result = swarm_intelligence(
    objective=lambda x: float(np.sum(x**2)),
    agents=agents,
    n_iterations=200,
    diversity_threshold=0.1,
    bounds=[(-5, 5), (-5, 5), (-5, 5)],
)
print(result.best_value)         # near 0.0
print(result.diversity_history[-1])  # final swarm diversity
```

---

## Ethical Elicitation (`src/ethical_elicitation.py`)

### `detect_manipulation(responses) -> List[ManipulationAttempt]`

Detect manipulation attempts in elicitation responses. Checks for copy-paste duplication, outlier embeddings (z-score), suspiciously identical timing patterns, and coordinated response fingerprints.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `responses` | `List[Dict[str, Any]]` | required | Response dicts with keys like `"agent_id"`, `"text"`, `"embedding"`, `"timestamp"` |

**Returns:** `List[ManipulationAttempt]` — detected attempts with agent ID, manipulation type, confidence, evidence, and description.

```python
from divelicit.src.ethical_elicitation import detect_manipulation

responses = [
    {"agent_id": "a1", "text": "Answer A", "embedding": [0.1, 0.2], "timestamp": 100},
    {"agent_id": "a2", "text": "Answer A", "embedding": [0.1, 0.2], "timestamp": 100},
    {"agent_id": "a3", "text": "Answer B", "embedding": [0.9, 0.8], "timestamp": 105},
]
attempts = detect_manipulation(responses)
for a in attempts:
    print(f"{a.agent_id}: {a.manipulation_type} (confidence={a.confidence:.2f})")
    # a1: duplication (confidence=0.95)
    # a2: duplication (confidence=0.95)
```

---

### `fairness_audit(mechanism, demographic_groups) -> FairnessReport`

Audit a selection mechanism for fairness across demographic groups. Computes representation parity, quality parity, diversity parity, disparate impact ratios, equalized odds proxies, and individual fairness scores. Checks the 80% rule.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mechanism` | `Dict[str, Any]` | required | Mechanism description with keys like `"selected"`, `"candidates"`, `"embeddings"` |
| `demographic_groups` | `List[DemographicGroup]` | required | Groups with member IDs, selection rates, and quality scores |

**Returns:** `FairnessReport` — representation/quality/diversity parity, individual fairness score, disparate impact ratios, equalized odds, overall score, recommendations, and 80% rule pass/fail.

```python
from divelicit.src.ethical_elicitation import fairness_audit, DemographicGroup

groups = [
    DemographicGroup(name="Group A", member_ids=["1", "2", "3"],
                     selection_rate=0.67, average_quality=0.8, quality_scores=[0.7, 0.8, 0.9]),
    DemographicGroup(name="Group B", member_ids=["4", "5", "6"],
                     selection_rate=0.33, average_quality=0.75, quality_scores=[0.7, 0.8, 0.75]),
]
mechanism = {"selected": ["1", "2", "4"], "candidates": ["1", "2", "3", "4", "5", "6"]}
report = fairness_audit(mechanism, groups)
print(report.overall_fairness_score)  # 0.78
print(report.passes_80_percent_rule)  # True/False
print(report.recommendations)         # actionable suggestions
```

---

### `privacy_preserving_elicitation(query, agents, epsilon, total_budget, sensitivity) -> PrivateResult`

Differentially private elicitation via the Laplace mechanism. Adds calibrated Laplace noise to the aggregate of agent responses, providing (ε, 0)-differential privacy. Tracks cumulative privacy budget.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `np.ndarray` | required | Query vector to aggregate |
| `agents` | `List[Dict[str, Any]]` | required | Agent dicts with `"response"` vectors |
| `epsilon` | `float` | `1.0` | Privacy parameter (lower = more private) |
| `total_budget` | `float` | `10.0` | Total privacy budget across queries |
| `sensitivity` | `float` | `1.0` | Query sensitivity (L1 norm of max change from one agent) |

**Returns:** `PrivateResult` — noisy aggregate, true aggregate (for evaluation), epsilon spent, delta, accuracy estimate, noise scale, agent count, and remaining budget.

```python
from divelicit.src.ethical_elicitation import privacy_preserving_elicitation
import numpy as np

query = np.array([1.0, 0.0, 0.0])
agents = [
    {"response": np.array([0.8, 0.1, 0.1])},
    {"response": np.array([0.7, 0.2, 0.1])},
    {"response": np.array([0.9, 0.0, 0.1])},
]
result = privacy_preserving_elicitation(query, agents, epsilon=1.0)
print(result.noisy_aggregate)    # approximately [0.8, 0.1, 0.1] + noise
print(result.epsilon_spent)      # 1.0
print(result.accuracy_estimate)  # expected error magnitude
print(result.composition_budget_remaining)  # 9.0
```

---

### `informed_consent_check(mechanism) -> ConsentReport`

Check whether a mechanism satisfies informed consent requirements. Evaluates criteria including purpose disclosure, data usage transparency, opt-out availability, risk disclosure, and benefit explanation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mechanism` | `Dict[str, Any]` | required | Mechanism description with keys like `"purpose"`, `"data_usage"`, `"opt_out"`, `"risks"`, `"benefits"` |

**Returns:** `ConsentReport` — list of consent criteria with scores and satisfaction flags, overall score, consent validity, and recommendations.

```python
from divelicit.src.ethical_elicitation import informed_consent_check

mechanism = {
    "purpose": "Elicit diverse perspectives on climate policy",
    "data_usage": "Aggregated and anonymized for research",
    "opt_out": True,
    "risks": "Minimal; no personally identifiable information collected",
    "benefits": "Contributes to better policy recommendations",
}
report = informed_consent_check(mechanism)
print(report.consent_valid)      # True
print(report.overall_score)      # 0.92
for c in report.criteria:
    print(f"  {c.name}: {'✓' if c.satisfied else '✗'} ({c.score:.2f})")
```

---

### `stakeholder_analysis(decision, stakeholders) -> StakeholderReport`

Analyze the impact of a decision on different stakeholders. Computes per-stakeholder impact scores, builds a power-interest grid, identifies underrepresented stakeholders, and recommends engagement strategies.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `decision` | `Dict[str, Any]` | required | Decision description with impact vectors |
| `stakeholders` | `List[Stakeholder]` | required | Stakeholders with concerns, power, interest, and impact scores |

**Returns:** `StakeholderReport` — per-stakeholder impacts, power-interest grid, underrepresented groups, engagement strategies, and total positive/negative impact.

```python
from divelicit.src.ethical_elicitation import stakeholder_analysis, Stakeholder

stakeholders = [
    Stakeholder(id="s1", name="Local residents", concerns=[0.9, 0.1],
                power=0.3, interest=0.9, impact=0.8, group="community"),
    Stakeholder(id="s2", name="Industry leaders", concerns=[0.1, 0.8],
                power=0.9, interest=0.7, impact=0.5, group="business"),
    Stakeholder(id="s3", name="Environmental groups", concerns=[0.7, 0.3],
                power=0.4, interest=0.8, impact=0.6, group="advocacy"),
]
decision = {"impact_vector": [0.5, 0.5], "description": "New factory construction"}
report = stakeholder_analysis(decision, stakeholders)
print(report.power_interest_grid)     # {"high_power_high_interest": ["s2"], ...}
print(report.underrepresented)        # stakeholders with low power but high impact
print(report.engagement_strategy)     # {"s1": "keep_informed", "s2": "manage_closely", ...}
```

---

### `moral_uncertainty_aggregation(ethical_views, credences, parliament_seats) -> AggregatedView`

Aggregate recommendations under moral uncertainty. Implements both expected choiceworthiness (weighted average across moral frameworks) and moral parliament (seat-proportional voting) approaches to reach a decision under normative disagreement.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ethical_views` | `List[EthicalView]` | required | Views from different ethical frameworks |
| `credences` | `Optional[Dict[str, float]]` | `None` | Credence (belief weight) per framework; uniform if `None` |
| `parliament_seats` | `int` | `100` | Total seats in the moral parliament |

**Returns:** `AggregatedView` — aggregated recommendation, method used, per-framework contributions, regret bound, parliament seat allocation, and overall confidence.

```python
from divelicit.src.ethical_elicitation import moral_uncertainty_aggregation, EthicalView
import numpy as np

views = [
    EthicalView(framework="utilitarian",   recommendation=np.array([0.8, 0.2]), confidence=0.7, choiceworthiness=0.9),
    EthicalView(framework="deontological", recommendation=np.array([0.3, 0.7]), confidence=0.8, choiceworthiness=0.6),
    EthicalView(framework="virtue_ethics", recommendation=np.array([0.5, 0.5]), confidence=0.5, choiceworthiness=0.7),
]
result = moral_uncertainty_aggregation(views, credences={"utilitarian": 0.5, "deontological": 0.3, "virtue_ethics": 0.2})
print(result.recommendation)     # weighted recommendation vector
print(result.parliament_seats)   # {"utilitarian": 50, "deontological": 30, "virtue_ethics": 20}
print(result.regret_bound)       # minimax regret bound
```

---

## Data Classes

### Core Data Classes

#### `ElicitationResult`
| Field | Type | Description |
|-------|------|-------------|
| `responses` | `List[str]` | Selected diverse responses |
| `embeddings` | `np.ndarray` | Embedding matrix for selected |
| `diversity_score` | `float` | Sinkhorn diversity |
| `quality_scores` | `List[float]` | Per-response quality |
| `agent_contributions` | `Dict[int, int]` | Agent → count selected |
| `all_responses` | `List[str]` | Full candidate pool |
| `selected_indices` | `List[int]` | Indices into `all_responses` |
| `coverage` | `CoverageCertificate` | Coverage certificate |

#### `CoverageResult`
| Field | Type | Description |
|-------|------|-------------|
| `certificate` | `CoverageCertificate` | Hoeffding-based coverage bound |
| `fill_distance` | `float` | Max NN distance (universe → selected) |
| `dispersion` | `float` | Mean NN distance |
| `effective_dim` | `int` | PCA effective dimension |
| `n_selected` | `int` | Size of selected set |
| `n_universe` | `int` | Size of universe |

#### `ComparisonResult`
| Field | Type | Description |
|-------|------|-------------|
| `method_results` | `Dict[str, List[int]]` | Selected indices per method |
| `rankings` | `Dict[str, List[str]]` | Per-metric ranking of methods |
| `best_method` | `str` | Overall best method |
| `metrics` | `Dict[str, Dict[str, float]]` | Raw metrics per method |

---

### Auction Mechanisms Data Classes

#### `Bidder`
| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Bidder identifier |
| `embedding` | `np.ndarray` | Bidder embedding for diversity computation |
| `budget` | `float` | Maximum total spend |
| `active` | `bool` | Whether bidder is still active |
| `metadata` | `Dict` | Additional bidder metadata |

#### `Item`
| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Item identifier |
| `embedding` | `np.ndarray` | Item embedding vector |
| `reserve_price` | `float` | Minimum acceptable price |
| `quality` | `float` | Item quality score |
| `metadata` | `Dict` | Additional item metadata |

#### `Lot`
| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Lot identifier |
| `items` | `List[Item]` | Items in the bundle |
| `region_label` | `Optional[str]` | Optional region label |
| `metadata` | `Dict` | Additional lot metadata |

#### `Entry`
| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Entry identifier |
| `embedding` | `np.ndarray` | Entry embedding |
| `bid` | `float` | Amount bid (paid regardless in all-pay) |
| `content` | `Optional[str]` | Optional entry content |
| `quality_score` | `float` | Quality of the entry |
| `metadata` | `Dict` | Additional entry metadata |

#### `AuctionResult`
| Field | Type | Description |
|-------|------|-------------|
| `winners` | `List[str]` | Winning bidder IDs |
| `payments` | `Dict[str, float]` | Bidder → payment amount |
| `items_won` | `Dict[str, List[str]]` | Bidder → list of item IDs won |
| `diversity_bonuses` | `Dict[str, float]` | Diversity bonus per bidder |
| `total_welfare` | `float` | Total social welfare |
| `total_revenue` | `float` | Total auction revenue |
| `metadata` | `Dict` | Additional result metadata |

#### `VCGResult`
| Field | Type | Description |
|-------|------|-------------|
| `allocation` | `Dict[str, List[str]]` | Bidder → items allocated |
| `payments` | `Dict[str, float]` | VCG payments per bidder |
| `social_welfare` | `float` | Total social welfare |
| `diversity_welfare` | `float` | Diversity component of welfare |
| `individual_welfare` | `Dict[str, float]` | Per-bidder welfare |
| `metadata` | `Dict` | Additional result metadata |

#### `CAResult`
| Field | Type | Description |
|-------|------|-------------|
| `winning_lots` | `List[str]` | IDs of winning lots |
| `lot_winners` | `Dict[str, str]` | Lot → winning bidder |
| `payments` | `Dict[str, float]` | Payments per bidder |
| `total_welfare` | `float` | Total social welfare |
| `coverage_score` | `float` | Spatial coverage of winning lots |
| `metadata` | `Dict` | Additional result metadata |

#### `AscendingResult`
| Field | Type | Description |
|-------|------|-------------|
| `winners` | `Dict[str, str]` | Item → winning bidder |
| `final_prices` | `Dict[str, float]` | Final clock prices per item |
| `rounds_completed` | `int` | Number of rounds before termination |
| `dropout_history` | `List[Tuple[int, str, str]]` | (round, bidder, item) dropout events |
| `total_revenue` | `float` | Total auction revenue |
| `metadata` | `Dict` | Additional result metadata |

#### `ContestResult`
| Field | Type | Description |
|-------|------|-------------|
| `rankings` | `List[str]` | Entry IDs ranked by composite score |
| `scores` | `Dict[str, float]` | Composite scores per entry |
| `quality_scores` | `Dict[str, float]` | Quality component per entry |
| `diversity_scores` | `Dict[str, float]` | Diversity component per entry |
| `prizes` | `Dict[str, float]` | Prize amounts for winners |
| `total_effort` | `float` | Sum of all bids |
| `metadata` | `Dict` | Additional result metadata |

#### `BBResult`
| Field | Type | Description |
|-------|------|-------------|
| `allocation` | `List[str]` | Selected agent IDs |
| `transfers` | `Dict[str, float]` | Monetary transfers (sum ≈ 0) |
| `budget_surplus` | `float` | Net surplus (ideally 0) |
| `reports_used` | `Dict[str, np.ndarray]` | Agent reports used in allocation |
| `diversity_score` | `float` | Diversity of selected agents |
| `metadata` | `Dict` | Additional result metadata |

---

### Information Aggregation Data Classes

#### `Response`
| Field | Type | Description |
|-------|------|-------------|
| `respondent_id` | `str` | Respondent identifier |
| `answer` | `str` | Respondent's answer |
| `prediction` | `Dict[str, float]` | Predicted population distribution over answers |
| `metadata` | `Dict[str, Any]` | Additional metadata |

#### `ExpertOpinion`
| Field | Type | Description |
|-------|------|-------------|
| `expert_id` | `str` | Expert identifier |
| `estimate` | `float` | Point estimate |
| `confidence_interval` | `Optional[Tuple[float, float]]` | Optional (low, high) CI |
| `expertise_score` | `Optional[float]` | Quality weight for this expert |
| `metadata` | `Dict[str, Any]` | Additional metadata |

#### `Forecast`
| Field | Type | Description |
|-------|------|-------------|
| `forecaster_id` | `str` | Forecaster identifier |
| `probability` | `float` | Probability estimate for a binary event |
| `past_forecasts` | `Optional[List[float]]` | Historical forecasts for calibration |
| `past_outcomes` | `Optional[List[int]]` | Historical outcomes (0/1) for calibration |
| `metadata` | `Dict[str, Any]` | Additional metadata |

#### `BTSResult`
| Field | Type | Description |
|-------|------|-------------|
| `scores` | `Dict[str, float]` | Per-respondent BTS information scores |
| `estimated_truth` | `str` | Most likely true answer |
| `confidence` | `float` | Confidence in the estimated truth |
| `high_information_respondents` | `List[str]` | Top-scoring respondent IDs |
| `answer_frequencies` | `Dict[str, float]` | Observed answer frequencies |
| `information_scores` | `Dict[str, float]` | Information component of scores |
| `prediction_scores` | `Dict[str, float]` | Prediction component of scores |
| `raw_responses` | `List[Response]` | Original response objects |

#### `PeerPredResult`
| Field | Type | Description |
|-------|------|-------------|
| `scores` | `Dict[str, float]` | Per-respondent peer-prediction scores |
| `adjusted_payments` | `Dict[str, float]` | Incentive-compatible payments |
| `mechanism_properties` | `Dict[str, Any]` | Properties like incentive compatibility |
| `agreement_matrix` | `Dict[str, Dict[str, float]]` | Pairwise agreement scores |
| `reference_scores` | `Dict[str, float]` | Reference answer scores |

#### `SPResult`
| Field | Type | Description |
|-------|------|-------------|
| `surprisingly_popular_answer` | `str` | The answer most exceeding predictions |
| `votes` | `Dict[str, float]` | Vote tallies |
| `sp_scores` | `Dict[str, float]` | SP scores per answer |
| `confidence` | `float` | Confidence in the SP answer |
| `actual_frequencies` | `Dict[str, float]` | Observed answer frequencies |
| `mean_predicted_frequencies` | `Dict[str, float]` | Mean predicted frequencies |
| `frequency_gaps` | `Dict[str, float]` | Actual − predicted per answer |
| `confidence_intervals` | `Dict[str, Tuple[float, float]]` | CIs on frequency gaps |

#### `AggregateResult`
| Field | Type | Description |
|-------|------|-------------|
| `weighted_mean` | `float` | Weighted mean of expert estimates |
| `trimmed_mean` | `float` | Trimmed mean (outlier-robust) |
| `median` | `float` | Median estimate |
| `linear_pool` | `float` | Log-linear pool estimate |
| `confidence_interval` | `Tuple[float, float]` | Aggregate CI |
| `disagreement` | `float` | Expert disagreement level |
| `outlier_ids` | `List[str]` | IDs of outlier experts |
| `weights_used` | `Dict[str, float]` | Final weights per expert |

#### `CalibratedForecast`
| Field | Type | Description |
|-------|------|-------------|
| `aggregated_probability` | `float` | Final aggregated probability |
| `individual_calibrated` | `Dict[str, float]` | Per-forecaster calibrated values |
| `calibration_scores` | `Dict[str, Dict[str, float]]` | Per-forecaster Brier/log-loss |
| `aggregate_brier` | `float` | Aggregate Brier score |
| `aggregate_log_loss` | `float` | Aggregate log-loss |
| `weights_used` | `Dict[str, float]` | Final weights per forecaster |
| `recalibrated` | `bool` | Whether isotonic recalibration was applied |

---

### Strategic Diversity Data Classes

#### `Strategy`
| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Strategy name |
| `index` | `int` | Strategy index in payoff matrices |
| `features` | `NDArray[np.float64]` | Feature vector for diversity computation |

#### `Agent`
| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` | Agent identifier |
| `type_vector` | `NDArray[np.float64]` | Agent type (private information) |
| `available_strategies` | `List[Strategy]` | Strategies available to this agent |
| `quality` | `float` | Agent quality score |

#### `GamePayoff`
| Field | Type | Description |
|-------|------|-------------|
| `n_players` | `int` | Number of players |
| `n_strategies` | `List[int]` | Number of strategies per player |
| `matrices` | `List[NDArray[np.float64]]` | Payoff matrix per player |

#### `NashResult`
| Field | Type | Description |
|-------|------|-------------|
| `equilibrium_strategies` | `List[NDArray[np.float64]]` | Mixed strategy profile per player |
| `expected_payoffs` | `NDArray[np.float64]` | Expected payoff per player |
| `diversity_score` | `float` | Diversity at equilibrium |
| `is_approximate` | `bool` | Whether the solution is approximate |
| `support_sizes` | `List[int]` | Support size per player |

#### `CorrelatedResult`
| Field | Type | Description |
|-------|------|-------------|
| `joint_distribution` | `NDArray[np.float64]` | Joint probability distribution |
| `expected_welfare` | `float` | Expected social welfare |
| `deviation_incentives` | `NDArray[np.float64]` | Deviation incentives per player |
| `is_feasible` | `bool` | Whether the equilibrium is feasible |
| `max_violation` | `float` | Maximum constraint violation |

#### `Allocation`
| Field | Type | Description |
|-------|------|-------------|
| `assignments` | `Dict[int, int]` | Agent → outcome assignment |
| `transfers` | `Dict[int, float]` | Monetary transfers per agent |
| `total_welfare` | `float` | Total social welfare |
| `is_budget_balanced` | `bool` | Whether transfers sum to ≤ 0 |

#### `Mechanism`
| Field | Type | Description |
|-------|------|-------------|
| `allocation_rule` | `Callable` | Maps reports → allocation |
| `payment_rule` | `Callable` | Maps (reports, allocation) → payments |
| `n_agents` | `int` | Number of agents |
| `n_outcomes` | `int` | Number of possible outcomes |
| `description` | `str` | Human-readable description |
| `valuations` | `Optional[NDArray[np.float64]]` | Agent valuation matrix |
| `diversity_fn` | `Optional[Callable]` | Diversity measurement function |

---

### Deliberation Data Classes

#### `Position`
| Field | Type | Description |
|-------|------|-------------|
| `vector` | `np.ndarray` | Position in opinion space |
| `conviction` | `float` | Strength of conviction `[0, 1]` |
| `label` | `str` | Human-readable position label |

#### `Statement`
| Field | Type | Description |
|-------|------|-------------|
| `claim` | `str` | Central claim |
| `grounds` | `str` | Evidence/data supporting the claim |
| `warrant` | `str` | Reasoning connecting grounds to claim |
| `backing` | `str` | Support for the warrant |
| `qualifier` | `float` | Confidence qualifier `[0, 1]` |
| `rebuttal` | `str` | Potential counter-argument |
| `embedding` | `Optional[np.ndarray]` | Statement embedding vector |
| `id` | `str` | Unique identifier |

#### `Participant`
| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Participant identifier |
| `name` | `str` | Display name |
| `position` | `Position` | Current position |
| `expertise_weights` | `Optional[np.ndarray]` | Topic expertise weights |
| `open_mindedness` | `float` | Willingness to change position `[0, 1]` |
| `position_history` | `List[np.ndarray]` | Historical positions |

#### `Argument`
| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Argument identifier |
| `statement` | `Statement` | Underlying statement |
| `supports` | `List[str]` | IDs of supported arguments |
| `attacks` | `List[str]` | IDs of attacked arguments |
| `strength` | `float` | Argument strength score |
| `centrality` | `float` | Graph centrality score |

#### `ArgumentMap`
| Field | Type | Description |
|-------|------|-------------|
| `arguments` | `Dict[str, Argument]` | All arguments by ID |
| `support_edges` | `List[Tuple[str, str, float]]` | (source, target, weight) support edges |
| `attack_edges` | `List[Tuple[str, str, float]]` | (source, target, weight) attack edges |
| `strongest_arguments` | `List[str]` | IDs of strongest arguments |
| `central_claims` | `List[str]` | IDs of most central arguments |
| `key_rebuttals` | `List[str]` | IDs of key rebuttal arguments |

#### `DebateResult`
| Field | Type | Description |
|-------|------|-------------|
| `final_positions` | `List[Position]` | Participant positions after debate |
| `argument_graph` | `ArgumentMap` | Full argument map |
| `consensus_measure` | `float` | Degree of consensus `[0, 1]` |
| `rounds_data` | `List[Dict[str, Any]]` | Per-round statistics |
| `position_trajectories` | `Dict[str, List[np.ndarray]]` | Position evolution per participant |
| `total_arguments` | `int` | Total arguments generated |

#### `DialogueResult`
| Field | Type | Description |
|-------|------|-------------|
| `final_positions` | `List[Position]` | Positions after dialogue |
| `position_evolution` | `Dict[str, List[np.ndarray]]` | Position history per participant |
| `questioner_sequence` | `List[str]` | Who asked questions each iteration |
| `convergence_iteration` | `int` | Iteration at which dialogue converged |
| `final_spread` | `float` | Position spread at termination |
| `challenges_made` | `int` | Total challenges issued |

#### `ConsensusResult`
| Field | Type | Description |
|-------|------|-------------|
| `final_positions` | `List[Position]` | Positions after consensus process |
| `agreement_level` | `float` | Degree of agreement `[0, 1]` |
| `convergence_round` | `int` | Round at which consensus was reached |
| `holdout_positions` | `List[Position]` | Positions that did not converge |
| `position_history` | `List[List[np.ndarray]]` | All positions per round |
| `centroid_history` | `List[np.ndarray]` | Centroid evolution |

#### `PollResult`
| Field | Type | Description |
|-------|------|-------------|
| `pre_poll_positions` | `List[np.ndarray]` | Positions before deliberation |
| `post_poll_positions` | `List[np.ndarray]` | Positions after deliberation |
| `opinion_change` | `List[float]` | Per-participant opinion shift magnitude |
| `polarization_before` | `float` | Polarization before deliberation |
| `polarization_after` | `float` | Polarization after deliberation |
| `information_gain` | `float` | Aggregate information gain |
| `group_assignments` | `Dict[str, int]` | Participant → group assignment |
| `phase_positions` | `Dict[str, List[np.ndarray]]` | Positions at each phase |

#### `AssemblyResult`
| Field | Type | Description |
|-------|------|-------------|
| `recommendations` | `Dict[str, np.ndarray]` | Per-issue recommendation vectors |
| `confidence_levels` | `Dict[str, float]` | Confidence per issue |
| `minority_reports` | `Dict[str, List[Dict[str, Any]]]` | Dissenting views per issue |
| `selected_participants` | `List[str]` | Sortition-selected participant IDs |
| `voting_results` | `Dict[str, Dict[str, Any]]` | Per-issue voting breakdown |
| `group_assignments` | `Dict[str, Dict[str, int]]` | Per-issue group assignments |
| `position_trajectories` | `Dict[str, Dict[str, List[np.ndarray]]]` | Position evolution per issue per participant |

---

### Portfolio Optimization Data Classes

#### `Asset`
| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Asset name/ticker |
| `expected_return` | `float` | Expected annual return |
| `market_cap_weight` | `float` | Market capitalization weight |
| `sector` | `str` | Asset sector classification |

#### `Portfolio`
| Field | Type | Description |
|-------|------|-------------|
| `weights` | `NDArray[np.float64]` | Portfolio weights (sum to 1) |
| `expected_return` | `float` | Expected portfolio return |
| `risk` | `float` | Portfolio volatility |
| `diversity_score` | `float` | Diversity metric (effective # of bets) |
| `asset_names` | `List[str]` | Asset names corresponding to weights |
| `risk_contributions` | `Optional[NDArray[np.float64]]` | Per-asset risk contribution |
| `blended_returns` | `Optional[NDArray[np.float64]]` | Posterior expected returns (Black-Litterman) |
| `posterior_cov` | `Optional[NDArray[np.float64]]` | Posterior covariance (Black-Litterman) |

#### `Trade`
| Field | Type | Description |
|-------|------|-------------|
| `asset_name` | `str` | Asset to trade |
| `direction` | `str` | `"buy"` or `"sell"` |
| `quantity` | `float` | Trade size (weight change) |
| `estimated_cost` | `float` | Estimated transaction cost |

#### `Trades`
| Field | Type | Description |
|-------|------|-------------|
| `trades` | `List[Trade]` | Individual trade instructions |
| `total_cost` | `float` | Total transaction cost |
| `new_diversity` | `float` | Diversity score after rebalancing |
| `old_diversity` | `float` | Diversity score before rebalancing |

#### `MarketData`
| Field | Type | Description |
|-------|------|-------------|
| `assets` | `List[Asset]` | Investable assets |
| `cov_matrix` | `NDArray[np.float64]` | Asset covariance matrix |
| `market_cap_weights` | `NDArray[np.float64]` | Market cap weights |
| `risk_free_rate` | `float` | Risk-free rate |
| `risk_aversion` | `float` | Market risk aversion |
| `tau` | `float` | Black-Litterman scaling parameter |

#### `View`
| Field | Type | Description |
|-------|------|-------------|
| `pick_vector` | `NDArray[np.float64]` | Portfolio pick vector (long/short weights) |
| `expected_return` | `float` | Expected return of the view |
| `confidence` | `float` | View confidence `(0, 1]` |

---

### Experiment Design Data Classes

#### `Factor`
| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Factor name |
| `levels` | `List[Level]` | Discrete levels |
| `lower` | `float` | Lower bound for continuous range |
| `upper` | `float` | Upper bound for continuous range |

#### `Level`
| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Level name |
| `value` | `float` | Numeric value |

#### `DesignMatrix`
| Field | Type | Description |
|-------|------|-------------|
| `matrix` | `np.ndarray` | Design matrix (runs × factors) |
| `factor_names` | `List[str]` | Factor names |
| `d_efficiency` | `float` | D-efficiency score |
| `space_filling_metric` | `float` | Maximin distance metric |
| `design_type` | `str` | Design type description |

#### `NextPoint`
| Field | Type | Description |
|-------|------|-------------|
| `point` | `np.ndarray` | Recommended next evaluation point |
| `acquisition_value` | `float` | Acquisition function value |
| `diversity_contribution` | `float` | Diversity contribution score |

#### `SurrogateModel`
| Field | Type | Description |
|-------|------|-------------|
| `X_train` | `np.ndarray` | Training inputs |
| `y_train` | `np.ndarray` | Training outputs |
| `K_inv` | `np.ndarray` | Inverse kernel matrix |
| `alpha` | `np.ndarray` | GP weights |
| `kernel_fn` | `Callable` | Kernel function |
| `noise` | `float` | Observation noise |
| `lengthscale` | `float` | Kernel lengthscale |
| `variance` | `float` | Kernel variance |

#### `OptResult`
| Field | Type | Description |
|-------|------|-------------|
| `best_point` | `np.ndarray` | Best point found |
| `best_value` | `float` | Best objective value |
| `X_evals` | `np.ndarray` | All evaluated points |
| `y_evals` | `np.ndarray` | All objective values |
| `surrogate` | `SurrogateModel` | Final fitted surrogate model |
| `convergence` | `List[float]` | Best-so-far value per iteration |

---

### Collective Intelligence Data Classes

#### `Problem`
| Field | Type | Description |
|-------|------|-------------|
| `objective` | `Callable[[np.ndarray], float]` | Objective function |
| `bounds` | `Sequence[Tuple[float, float]]` | Per-dimension search bounds |
| `description` | `str` | Problem description |

#### `SwarmAgent`
| Field | Type | Description |
|-------|------|-------------|
| `position` | `np.ndarray` | Current position |
| `velocity` | `np.ndarray` | Current velocity |
| `perspective` | `np.ndarray` | Agent perspective vector |
| `agent_type` | `str` | Agent type label |

#### `Solution`
| Field | Type | Description |
|-------|------|-------------|
| `best_position` | `np.ndarray` | Best solution found |
| `best_value` | `float` | Best objective value |
| `diversity` | `float` | Final swarm diversity |
| `convergence_history` | `List[float]` | Best value per iteration |
| `all_solutions` | `List[np.ndarray]` | All final agent positions |

#### `Idea`
| Field | Type | Description |
|-------|------|-------------|
| `idea_id` | `int` | Idea identifier |
| `embedding` | `np.ndarray` | Idea embedding vector |
| `initial_quality` | `float` | Initial quality estimate |
| `description` | `str` | Idea description |

#### `MarketplaceResult`
| Field | Type | Description |
|-------|------|-------------|
| `rankings` | `List[int]` | Idea IDs ranked by final price |
| `final_prices` | `dict[int, float]` | Final market price per idea |
| `undervalued` | `List[int]` | Ideas priced below quality |
| `market_efficiency` | `float` | Price-quality correlation |
| `price_history` | `dict[int, List[float]]` | Price evolution per idea |

#### `Challenge`
| Field | Type | Description |
|-------|------|-------------|
| `challenge_id` | `int` | Challenge identifier |
| `objective` | `Callable[[np.ndarray], float]` | Challenge objective function |
| `bounds` | `Sequence[Tuple[float, float]]` | Search bounds |
| `description` | `str` | Challenge description |

#### `TournamentResult`
| Field | Type | Description |
|-------|------|-------------|
| `winners` | `List[Tuple[int, float]]` | (agent index, score) tuples |
| `stage_scores` | `List[List[Tuple[int, float]]]` | Per-stage scores |
| `innovation_quality` | `float` | Overall innovation quality |
| `diversity_of_winners` | `float` | Diversity among winners |
| `winning_solutions` | `List[np.ndarray]` | Winning solution vectors |

#### `LabelItem`
| Field | Type | Description |
|-------|------|-------------|
| `item_id` | `int` | Item identifier |
| `true_label` | `int` | Ground-truth label |
| `features` | `Optional[np.ndarray]` | Optional feature vector |

#### `Labels`
| Field | Type | Description |
|-------|------|-------------|
| `estimated_labels` | `dict[int, int]` | Item → estimated label |
| `labeler_quality` | `dict[int, float]` | Per-labeler quality score |
| `inter_annotator_agreement` | `float` | Cohen's kappa agreement |
| `confident_items` | `List[int]` | Items exceeding confidence threshold |
| `label_probabilities` | `dict[int, np.ndarray]` | Per-item class probabilities |

#### `SwarmResult`
| Field | Type | Description |
|-------|------|-------------|
| `best_position` | `np.ndarray` | Best position found |
| `best_value` | `float` | Best objective value |
| `convergence_history` | `List[float]` | Best value per iteration |
| `diversity_history` | `List[float]` | Swarm diversity per iteration |
| `final_positions` | `List[np.ndarray]` | Final particle positions |

---

### Ethical Elicitation Data Classes

#### `ManipulationAttempt`
| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Agent suspected of manipulation |
| `manipulation_type` | `str` | Type: `"duplication"`, `"outlier"`, `"coordination"` |
| `confidence` | `float` | Detection confidence `[0, 1]` |
| `evidence` | `Dict[str, Any]` | Supporting evidence |
| `description` | `str` | Human-readable description |

#### `DemographicGroup`
| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Group name |
| `member_ids` | `List[str]` | IDs of group members |
| `selection_rate` | `float` | Fraction of group selected |
| `average_quality` | `float` | Mean quality score |
| `quality_scores` | `List[float]` | Per-member quality scores |

#### `FairnessReport`
| Field | Type | Description |
|-------|------|-------------|
| `representation_parity` | `Dict[str, float]` | Representation ratios per group |
| `quality_parity` | `Dict[str, float]` | Quality ratios per group |
| `diversity_parity` | `Dict[str, float]` | Diversity ratios per group |
| `individual_fairness_score` | `float` | Individual fairness metric |
| `disparate_impact_ratios` | `Dict[str, float]` | Disparate impact per group pair |
| `equalized_odds` | `Dict[str, Dict[str, float]]` | Equalized odds proxies |
| `overall_fairness_score` | `float` | Aggregate fairness score `[0, 1]` |
| `recommendations` | `List[str]` | Actionable fairness recommendations |
| `passes_80_percent_rule` | `bool` | Whether 80% rule is satisfied |

#### `PrivateResult`
| Field | Type | Description |
|-------|------|-------------|
| `noisy_aggregate` | `np.ndarray` | Differentially private aggregate |
| `true_aggregate` | `Optional[np.ndarray]` | True aggregate (for evaluation) |
| `epsilon_spent` | `float` | Privacy budget consumed |
| `delta` | `float` | Delta parameter (0 for pure DP) |
| `accuracy_estimate` | `float` | Expected accuracy given noise |
| `noise_scale` | `float` | Laplace noise scale used |
| `num_agents` | `int` | Number of contributing agents |
| `composition_budget_remaining` | `float` | Remaining privacy budget |

#### `ConsentCriterion`
| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Criterion name |
| `description` | `str` | What the criterion checks |
| `score` | `float` | Score `[0, 1]` |
| `satisfied` | `bool` | Whether the criterion is met |
| `evidence` | `str` | Evidence for the score |

#### `ConsentReport`
| Field | Type | Description |
|-------|------|-------------|
| `criteria` | `List[ConsentCriterion]` | All evaluated criteria |
| `overall_score` | `float` | Aggregate consent score |
| `consent_valid` | `bool` | Whether informed consent is valid |
| `recommendations` | `List[str]` | Recommendations for improvement |

#### `Stakeholder`
| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Stakeholder identifier |
| `name` | `str` | Stakeholder name |
| `concerns` | `List[float]` | Concern vector |
| `power` | `float` | Power level `[0, 1]` |
| `interest` | `float` | Interest level `[0, 1]` |
| `impact` | `float` | Expected impact `[0, 1]` |
| `group` | `str` | Stakeholder group classification |

#### `StakeholderReport`
| Field | Type | Description |
|-------|------|-------------|
| `stakeholder_impacts` | `Dict[str, float]` | Per-stakeholder impact scores |
| `power_interest_grid` | `Dict[str, List[str]]` | Quadrant → stakeholder IDs |
| `underrepresented` | `List[str]` | Low-power high-impact stakeholders |
| `engagement_strategy` | `Dict[str, str]` | Recommended engagement per stakeholder |
| `total_positive_impact` | `float` | Sum of positive impacts |
| `total_negative_impact` | `float` | Sum of negative impacts |

#### `EthicalView`
| Field | Type | Description |
|-------|------|-------------|
| `framework` | `str` | Ethical framework name |
| `recommendation` | `np.ndarray` | Recommendation vector |
| `confidence` | `float` | Framework confidence `[0, 1]` |
| `choiceworthiness` | `float` | Choiceworthiness score |

#### `AggregatedView`
| Field | Type | Description |
|-------|------|-------------|
| `recommendation` | `np.ndarray` | Aggregated recommendation vector |
| `method` | `str` | Aggregation method used |
| `framework_contributions` | `Dict[str, float]` | Per-framework weight in result |
| `regret_bound` | `float` | Minimax regret bound |
| `parliament_seats` | `Dict[str, int]` | Seat allocation in moral parliament |
| `confidence` | `float` | Overall confidence |

---

## Multi-Agent Module (`src/multi_agent.py`)

### `MultiAgentElicitor`
Run the same prompt through multiple LLM providers with diversity-aware ensemble.

```python
from divelicit.src.multi_agent import MultiAgentElicitor

elicitor = MultiAgentElicitor()
elicitor.add_provider("openai", model="gpt-4.1-nano", temperatures=[0.3, 0.7, 1.0])
elicitor.add_provider("anthropic", model="claude-sonnet-4-20250514")

result = elicitor.elicit("Propose solutions to urban traffic congestion", k=8)
print(result.provider_stats)  # which provider contributed most unique ideas
```

---

## Debate Module (`src/debate.py`)

### `DiverseDebate`
Structured multi-round debate with diversity constraints.

```python
from divelicit.src.debate import DiverseDebate

debate = DiverseDebate(n_rounds=3, n_agents=4)
result = debate.run("Should cities ban private cars?")
print(result.argument_graph)     # directed graph of arguments
print(result.consensus_points)   # areas of agreement
print(result.diversity_score)    # how diverse the debate was
```

---

## Crowdsourcing Module (`src/crowdsourcing.py`)

### `IdeaCrowdsourcer`
API for human+AI mixed idea elicitation with deduplication.

```python
from divelicit.src.crowdsourcing import IdeaCrowdsourcer

cs = IdeaCrowdsourcer(embed_dim=64)
cs.add_idea("Use solar panels on rooftops", source="human")
cs.add_idea("Solar rooftop installations for energy", source="ai")  # deduplicated
cs.add_idea("Wind turbines in urban areas", source="human")

print(cs.unique_count)         # 2 (deduplication detected similarity)
print(cs.coverage_report())    # coverage analysis
```

---

## Evaluation Suite (`src/evaluation_suite.py`)

### `DiversityEvaluator`
Evaluate the quality of diverse elicitation.

```python
from divelicit.src.evaluation_suite import DiversityEvaluator

evaluator = DiversityEvaluator()
report = evaluator.evaluate(selected_responses, all_candidates)
print(report.coverage)          # fraction of space covered
print(report.novelty_scores)    # per-item novelty
print(report.redundancy)        # pairwise redundancy
```

---

## Mechanism Core (`src/mechanism_core.py`)

Core mechanism design primitives: VCG, Groves, Myerson, and automated mechanism design.

### Classes

#### `Allocation(assignment, social_welfare=0.0)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `assignment` | `Dict[int, List[int]]` | Maps agent ID → list of item IDs |
| `social_welfare` | `float` | Total social welfare of allocation |

#### `PaymentResult(payments)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `payments` | `Dict[int, float]` | Maps agent ID → payment amount |

#### `VCGMechanism(n_items)`

Implements the Vickrey–Clarke–Groves mechanism for welfare-maximizing allocation with externality payments.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_items` | `int` | Number of items to allocate |

**Methods:**
- `allocate(valuations) -> Allocation` — Compute welfare-maximizing allocation
- `compute_payments(valuations, allocation) -> PaymentResult` — Compute VCG externality payments
- `run(valuations) -> Tuple[Allocation, PaymentResult]` — Run full VCG mechanism

#### `GrovesScheme(n_items, h_functions=None)`

Generalized Groves mechanism with customizable h-functions for budget-balance tradeoffs.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_items` | `int` | Number of items |
| `h_functions` | `Optional[Dict[int, Callable]]` | Per-agent rebate functions |

#### `MyersonAuction(distributions)`

Myerson's optimal single-item auction with virtual value computation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `distributions` | `Dict[int, Tuple[str, dict]]` | Per-agent value distributions |

#### `AutomatedMechanismDesign(n_agents, n_outcomes, type_space_sizes)`

LP-based automated mechanism design.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_agents` | `int` | Number of agents |
| `n_outcomes` | `int` | Number of possible outcomes |
| `type_space_sizes` | `List[int]` | Size of each agent's type space |

#### `OnlineMechanism(n_items, n_rounds)`

Online mechanism for sequential allocation.

#### `MechanismComparator(mechanisms)`

Compare multiple mechanisms on welfare, revenue, and fairness metrics.

### Functions

#### `verify_strategy_proofness(mechanism, type_space, agents) -> Tuple[bool, Optional[Dict]]`

Verify strategy-proofness by checking all deviations in the type space.

#### `check_individual_rationality(mechanism, valuations, outside_option=0.0) -> Dict[int, bool]`

Check whether each agent receives non-negative utility.

#### `generate_random_valuations(n_agents, n_items, rng=None, additive=True) -> Dict[int, Dict[tuple, float]]`

Generate random valuations for testing.

```python
from implementation.src.mechanism_core import VCGMechanism, verify_strategy_proofness

vcg = VCGMechanism(n_items=5)
alloc, payments = vcg.run(valuations)
print(f"Welfare: {alloc.social_welfare}, Payments: {payments.payments}")

sp, violation = verify_strategy_proofness(vcg, type_space, agents)
print(f"Strategy-proof: {sp}")
```

---

## Voting Systems (`src/voting_systems.py`)

Single-winner and proportional voting methods with impossibility demonstrations.

### Classes

#### `ElectionResult(winner, ranking, scores, explanation)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `winner` | `int` | Winning candidate ID |
| `ranking` | `List[int]` | Full candidate ranking |
| `scores` | `Dict[int, float]` | Per-candidate scores |
| `explanation` | `str` | Human-readable explanation |

#### `PluralityVoting()`

First-past-the-post voting.

**Methods:**
- `compute_winner(ballots: List[List[int]]) -> ElectionResult`
- `compute_ranking(ballots: List[List[int]]) -> List[int]`

#### `BordaCount()`

Borda count positional voting. Each candidate receives points based on rank position.

#### `InstantRunoffVoting()`

Instant-runoff (ranked-choice) voting with iterative elimination.

#### `CopelandMethod()`

Pairwise comparison method. Winner beats the most other candidates head-to-head.

#### `SchulzeMethod()`

Schulze (beatpath) method. Finds strongest paths in the pairwise defeat graph. Condorcet-consistent.

#### `KemenyYoung()`

Kemeny–Young optimal ranking method. Finds the ranking minimizing Kendall tau distance to all ballots.

#### `DHondt()`

D'Hondt proportional representation method.

#### `SainteLague()`

Sainte-Laguë proportional representation method.

#### `STV()`

Single Transferable Vote for multi-winner elections.

#### `VotingSimulator(n_candidates, n_voters, seed=42)`

Generate random elections and compare voting methods.

### Functions

#### `find_condorcet_winner(ballots) -> Optional[int]`

Find the Condorcet winner (if one exists) from ranked ballots.

#### `find_condorcet_cycle(ballots) -> Optional[List[int]]`

Detect a Condorcet cycle in the pairwise majority graph.

#### `demonstrate_arrow_impossibility(n_candidates=3, n_voters=3) -> Dict`

Construct preference profiles witnessing Arrow's impossibility theorem.

#### `find_gibbard_satterthwaite_manipulation(method, ballots, candidates) -> Optional[Dict]`

Search for a profitable misreport under the given voting method.

#### `condorcet_efficiency(method, n_candidates=4, n_voters=100, n_trials=100, seed=42) -> float`

Estimate Condorcet efficiency of a voting method via simulation.

#### `compute_all_condorcet_efficiencies(n_candidates=4, n_voters=50, n_trials=100, seed=42) -> Dict[str, float]`

Compute Condorcet efficiency for all implemented voting methods.

```python
from implementation.src.voting_systems import SchulzeMethod, find_condorcet_winner

ballots = [[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1]]
schulze = SchulzeMethod()
result = schulze.compute_winner(ballots)
print(f"Winner: {result.winner}, Ranking: {result.ranking}")

cw = find_condorcet_winner(ballots)
print(f"Condorcet winner: {cw}")
```

---

## Auction Engine (`src/auction_engine.py`)

Combinatorial, double, sequential, and spectrum auctions with market equilibrium computation.

### Classes

#### `Bid(bidder_id, bundle, value)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `bidder_id` | `int` | Bidder identifier |
| `bundle` | `Tuple[int, ...]` | Tuple of item IDs in the bid bundle |
| `value` | `float` | Bid value for the bundle |

#### `AuctionAllocation(assignment, payments, social_welfare, revenue)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `assignment` | `Dict[int, List[Tuple[int, ...]]]` | Bidder → list of won bundles |
| `payments` | `Dict[int, float]` | Bidder → payment |
| `social_welfare` | `float` | Total social welfare |
| `revenue` | `float` | Auctioneer revenue |

#### `CombinatorialAuction(items)`

VCG-based combinatorial auction supporting bundle bids.

**Methods:**
- `add_bid(bid: Bid)` — Add a bid
- `solve() -> AuctionAllocation` — Solve winner determination and compute VCG payments

#### `DoubleAuction()`

Continuous double auction for two-sided markets.

**Methods:**
- `submit_order(order: Order)` — Submit a buy/sell order
- `clear() -> List[Tuple[Order, Order]]` — Match orders and return trades

#### `SequentialAuction(items, n_rounds=20)`

Sequential auction selling items one at a time.

#### `SpectrumAuction(items, reserve_prices=None)`

Spectrum-style auction with reserve prices and regulatory constraints.

#### `AuctionAnalyzer`

Analyze auction outcomes for efficiency, revenue, and fairness.

### Functions

#### `find_walrasian_equilibrium(n_goods, n_agents, utility_functions, endowments, tol=1e-6, max_iter=1000) -> Tuple[np.ndarray, np.ndarray]`

Compute Walrasian (competitive) equilibrium prices and allocations.

#### `compute_ceei(n_agents, n_goods, utility_functions, budget=1.0, n_iter=500) -> Tuple[np.ndarray, np.ndarray]`

Compute Competitive Equilibrium from Equal Incomes.

#### `generate_random_combinatorial_valuations(n_bidders, n_items, synergy=0.2, rng=None) -> Dict[int, Dict[Tuple[int, ...], float]]`

Generate random combinatorial valuations with synergy effects.

```python
from implementation.src.auction_engine import CombinatorialAuction, Bid

auction = CombinatorialAuction(items=[0, 1, 2])
auction.add_bid(Bid(bidder_id=0, bundle=(0, 1), value=10.0))
auction.add_bid(Bid(bidder_id=1, bundle=(1, 2), value=8.0))
auction.add_bid(Bid(bidder_id=2, bundle=(0,), value=5.0))
result = auction.solve()
print(f"Welfare: {result.social_welfare}, Revenue: {result.revenue}")
```

---

## Fair Division (`src/fair_division.py`)

Algorithms for fair allocation of indivisible goods.

### Classes

#### `FairAllocation(assignment, utilities, method)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `assignment` | `Dict[int, List[int]]` | Agent → list of assigned items |
| `utilities` | `Dict[int, float]` | Agent → utility from assignment |
| `method` | `str` | Name of the allocation method used |

#### `EnvyFreeUpToOneItem()`

EF1 allocation via envy-cycle elimination.

**Methods:**
- `allocate(valuations: np.ndarray) -> FairAllocation`

#### `MaximinShareGuarantee()`

MMS-approximate allocation.

**Methods:**
- `allocate(valuations: np.ndarray) -> FairAllocation`

#### `RoundRobin()`

Round-robin picking protocol. Always EF1 for additive valuations.

#### `MaxNashWelfare()`

Maximize product of utilities. EF1 + Pareto-optimal for additive valuations.

#### `CutAndChoose()`

Classical 2-agent fair division.

#### `AdjustedWinner()`

Brams–Taylor adjusted winner procedure for 2 agents.

#### `RentDivision(n_rooms, total_rent)`

Envy-free room assignment with rent payments.

#### `FairDivisionAnalyzer(valuations)`

Post-hoc fairness analysis of any allocation.

### Functions

#### `check_proportionality(valuations, allocation) -> Dict[int, bool]`

Check if each agent receives at least 1/n of their total value.

#### `check_envy_freeness(valuations, allocation) -> Dict[Tuple[int, int], bool]`

Check pairwise envy-freeness.

#### `check_ef1(valuations, allocation) -> Dict[Tuple[int, int], bool]`

Check EF1: for every envious pair, removing one item from the envied bundle eliminates envy.

#### `check_equitability(valuations, allocation, tolerance=0.1) -> bool`

Check if all agents receive approximately equal utility.

#### `generate_random_valuations(n_agents, n_items, distribution='uniform', rng=None) -> np.ndarray`

Generate random valuation matrix.

```python
from implementation.src.fair_division import EnvyFreeUpToOneItem, check_ef1
import numpy as np

valuations = np.random.rand(4, 8)
ef1 = EnvyFreeUpToOneItem()
alloc = ef1.allocate(valuations)
print(f"Assignment: {alloc.assignment}")
print(f"EF1 satisfied: {all(check_ef1(valuations, alloc).values())}")
```

---

## Matching Markets (`src/matching_markets.py`)

Stable matching algorithms for two-sided and one-sided markets.

### Classes

#### `Matching(pairs, stable, method)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `pairs` | `Dict[int, int]` | Matched pairs (proposer → acceptor) |
| `stable` | `bool` | Whether the matching is stable |
| `method` | `str` | Algorithm used |

#### `GaleShapley()`

Deferred acceptance algorithm producing proposer-optimal stable matchings.

**Methods:**
- `match(proposer_prefs: Dict[int, List[int]], acceptor_prefs: Dict[int, List[int]]) -> Matching`

#### `TopTradingCycles()`

TTC algorithm for one-sided matching with initial endowments. Strategy-proof.

**Methods:**
- `allocate(preferences: Dict[int, List[int]], endowments: Dict[int, int]) -> Dict[int, int]`

#### `StableRoommates()`

Irving's algorithm for one-sided stable matching (roommate problem).

**Methods:**
- `match(preferences: Dict[int, List[int]]) -> Optional[Matching]`

#### `SchoolChoice()`

Student-optimal stable mechanism for school assignment.

**Methods:**
- `assign(student_prefs: Dict[int, List[int]], school_prefs: Dict[int, List[int]], capacities: Dict[int, int]) -> Dict[int, int]`

#### `MatchingMarketSimulator(n_proposers, n_acceptors, seed=42)`

Generate random matching markets and run experiments.

### Functions

#### `find_blocking_pairs(matching, prefs_a, prefs_b) -> List[Tuple[int, int]]`

Find all blocking pairs in a matching (pairs who would both prefer to deviate).

#### `check_matching_strategy_proofness(mechanism_func, agent_id, true_prefs, other_side_prefs, all_alternatives) -> Tuple[bool, Optional[Dict]]`

Check if an agent can profitably misreport preferences.

```python
from implementation.src.matching_markets import GaleShapley, find_blocking_pairs

gs = GaleShapley()
proposer_prefs = {0: [1, 0, 2], 1: [0, 2, 1], 2: [1, 2, 0]}
acceptor_prefs = {0: [2, 0, 1], 1: [0, 1, 2], 2: [1, 2, 0]}
matching = gs.match(proposer_prefs, acceptor_prefs)
print(f"Pairs: {matching.pairs}, Stable: {matching.stable}")
blocks = find_blocking_pairs(matching, proposer_prefs, acceptor_prefs)
print(f"Blocking pairs: {blocks}")  # []
```

---

## Information Elicitation (`src/information_elicitation.py`)

Proper scoring rules, prediction markets (LMSR), and forecast aggregation.

### Classes

#### `ProperScoringRule`

Base class for proper scoring rules (Brier, Log, Spherical).

**Methods:**
- `score(forecast: np.ndarray, outcome: int) -> float`
- `check_properness(n_outcomes: int, n_samples: int) -> Tuple[bool, float]`

#### `PeerPrediction`

Peer prediction mechanism for eliciting truthful reports without ground truth.

#### `MarketState(probabilities, cost, shares_outstanding)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `probabilities` | `np.ndarray` | Current market prices/probabilities |
| `cost` | `float` | Current cost function value |
| `shares_outstanding` | `np.ndarray` | Outstanding shares per outcome |

#### `PredictionMarket(n_outcomes, liquidity=100.0)`

LMSR-based prediction market.

**Methods:**
- `get_prices() -> np.ndarray` — Current market prices
- `buy(outcome: int, quantity: float) -> float` — Buy shares, returns cost
- `sell(outcome: int, quantity: float) -> float` — Sell shares, returns revenue
- `get_state() -> MarketState`

#### `DelphinMethod(n_experts, n_rounds=10)`

Delphi method for iterative expert consensus.

**Methods:**
- `run(initial_estimates: np.ndarray) -> Tuple[float, bool, int]` — Returns (aggregate, converged, rounds)

### Functions

#### `linear_opinion_pool(forecasts, weights=None) -> np.ndarray`

Weighted average of probability forecasts.

#### `logarithmic_opinion_pool(forecasts, weights=None) -> np.ndarray`

Geometric (log-linear) average of probability forecasts.

#### `extremized_aggregation(forecasts, alpha=2.5, weights=None) -> np.ndarray`

Extremized aggregation that sharpens pooled forecasts.

#### `calibration_score(forecasts, outcomes, n_bins=10) -> Tuple[float, List[Dict]]`

Compute calibration score with binned reliability analysis.

#### `brier_skill_score(forecasts, outcomes, baseline=None) -> float`

Brier skill score relative to baseline (climatological) forecast.

#### `brier_decomposition(forecasts, outcomes, n_bins=10) -> Dict[str, float]`

Decompose Brier score into reliability, resolution, and uncertainty.

#### `simulate_prediction_market(n_traders, n_outcomes, true_probs, n_rounds=100, liquidity=50.0, seed=42) -> Dict`

Run a simulated prediction market and return convergence results.

```python
from implementation.src.information_elicitation import PredictionMarket, linear_opinion_pool
import numpy as np

market = PredictionMarket(n_outcomes=3, liquidity=50.0)
cost = market.buy(outcome=0, quantity=10.0)
print(f"Prices after trade: {market.get_prices()}")

forecasts = [np.array([0.6, 0.3, 0.1]), np.array([0.5, 0.3, 0.2])]
pooled = linear_opinion_pool(forecasts)
print(f"Pooled forecast: {pooled}")
```

---

## Social Choice Analysis (`src/social_choice_analysis.py`)

Cooperative game theory: Shapley values, core, nucleolus, and bargaining solutions.

### Classes

#### `CooperativeGame(n_players, value_function)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_players` | `int` | Number of players |
| `value_function` | `Dict[frozenset, float]` | Coalition → value mapping |

#### `SimpleGame(n_players, winning_coalitions)`

Simple (0/1-valued) cooperative game defined by winning coalitions.

#### `WeightedVotingGame(quota, weights)`

Weighted voting game [q; w₁, ..., wₙ].

| Parameter | Type | Description |
|-----------|------|-------------|
| `quota` | `float` | Voting quota |
| `weights` | `List[float]` | Per-player weights |

### Functions

#### `compute_shapley_values(game) -> Dict[int, float]`

Compute exact Shapley values for a cooperative game.

#### `compute_banzhaf_index(game) -> Dict[int, float]`

Compute normalized Banzhaf power indices.

#### `find_core(game) -> Optional[np.ndarray]`

Find a point in the core of a cooperative game using linear programming.

#### `nucleolus(game) -> np.ndarray`

Compute the nucleolus via iterated LP.

#### `verify_shapley_axioms(game) -> Dict[str, bool]`

Verify efficiency, symmetry, dummy, and additivity axioms.

#### `nash_bargaining_solution(feasible_set, disagreement_point) -> np.ndarray`

Compute the Nash bargaining solution (maximize product of surplus).

#### `kalai_smorodinsky_solution(feasible_set, disagreement_point) -> np.ndarray`

Compute the Kalai–Smorodinsky proportional solution.

#### `compute_social_welfare(allocation, utilities, method='utilitarian') -> float`

Compute social welfare (utilitarian, egalitarian, or Nash).

#### `check_pareto_optimality(allocation, valuations) -> bool`

Check if an allocation is Pareto-optimal.

#### `check_strategyproofness(mechanism, type_space, agents) -> Tuple[bool, Optional[Dict]]`

Check strategy-proofness of a mechanism.

#### `is_convex_game(game) -> bool`

Check if a cooperative game is convex (supermodular).

```python
from implementation.src.social_choice_analysis import (
    CooperativeGame, compute_shapley_values, verify_shapley_axioms, find_core
)

# 3-player majority game
value_fn = {frozenset(): 0}
for i in range(3):
    value_fn[frozenset([i])] = 0
    for j in range(i+1, 3):
        value_fn[frozenset([i, j])] = 1
value_fn[frozenset([0, 1, 2])] = 1

game = CooperativeGame(n_players=3, value_function=value_fn)
shapley = compute_shapley_values(game)
print(f"Shapley values: {shapley}")  # {0: 1/3, 1: 1/3, 2: 1/3}
print(f"Axioms: {verify_shapley_axioms(game)}")
```

---

## Population Games (`src/population_games.py`)

Evolutionary dynamics, ESS analysis, and Nash equilibrium computation.

### Classes

#### `GameResult(trajectory, final_state, converged, equilibrium_type)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `trajectory` | `np.ndarray` | Population state over time (T × n_strategies) |
| `final_state` | `np.ndarray` | Final population state |
| `converged` | `bool` | Whether dynamics converged |
| `equilibrium_type` | `str` | Type of equilibrium reached |

#### `ReplicatorDynamics(payoff_matrix)`

Continuous-time replicator equation: ẋᵢ = xᵢ[(Ax)ᵢ − x⊤Ax].

| Parameter | Type | Description |
|-----------|------|-------------|
| `payoff_matrix` | `np.ndarray` | n×n symmetric payoff matrix |

**Methods:**
- `run(initial_pop: np.ndarray, timesteps: int = 1000, dt: float = 0.01) -> GameResult`

#### `BestResponseDynamics(payoff_matrix_row, payoff_matrix_col)`

Deterministic best-response dynamics for two-player games.

#### `FictitiousPlay(payoff_matrix_row, payoff_matrix_col)`

Brown's fictitious play: agents best-respond to empirical opponent frequencies.

#### `EvolutionaryStableStrategy(payoff_matrix)`

Find all evolutionary stable strategies of a symmetric game.

**Methods:**
- `find_all_ess() -> List[np.ndarray]`

#### `NashEquilibriumFinder(payoff_row, payoff_col)`

Enumerate all Nash equilibria of a two-player game via support enumeration.

**Methods:**
- `find_all() -> List[Tuple[np.ndarray, np.ndarray]]`

### Functions

#### `logit_dynamics(payoff_matrix, initial_pop, beta=1.0, timesteps=1000, dt=0.01) -> GameResult`

Logit (noisy best-response) dynamics with temperature parameter β.

#### `imitation_dynamics(payoff_matrix, initial_pop, timesteps=1000, dt=0.01) -> GameResult`

Pairwise imitation dynamics.

#### `smith_dynamics(payoff_matrix, initial_pop, timesteps=1000, dt=0.01) -> GameResult`

Smith dynamics (excess payoff dynamics).

#### `prisoners_dilemma() -> Tuple[np.ndarray, np.ndarray]`

Return standard Prisoner's Dilemma payoff matrices.

#### `hawk_dove(v=4.0, c=6.0) -> np.ndarray`

Return Hawk–Dove payoff matrix with parameters V and C.

#### `rock_paper_scissors() -> np.ndarray`

Return Rock–Paper–Scissors payoff matrix.

#### `coordination_game(a=2.0, b=1.0) -> Tuple[np.ndarray, np.ndarray]`

Return coordination game payoff matrices.

#### `battle_of_sexes(a=3.0, b=2.0) -> Tuple[np.ndarray, np.ndarray]`

Return Battle of the Sexes payoff matrices.

```python
from implementation.src.population_games import (
    ReplicatorDynamics, hawk_dove, EvolutionaryStableStrategy, NashEquilibriumFinder,
    coordination_game
)
import numpy as np

# Replicator dynamics on Hawk-Dove
A = hawk_dove(v=4.0, c=6.0)
rd = ReplicatorDynamics(A)
result = rd.run(np.array([0.5, 0.5]))
print(f"Final state: {result.final_state}")  # ≈ [0.667, 0.333]

# Find ESS
ess_finder = EvolutionaryStableStrategy(A)
ess_list = ess_finder.find_all_ess()
print(f"ESS: {ess_list}")

# Nash equilibria of coordination game
R, C = coordination_game()
ne = NashEquilibriumFinder(R, C)
equilibria = ne.find_all()
print(f"Found {len(equilibria)} Nash equilibria")
```
