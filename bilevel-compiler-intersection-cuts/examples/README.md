# BiCut Examples

Example bilevel optimization problems in TOML format for use with the `bicut` CLI.

## Examples

### `simple_bilevel.toml`

A textbook bilevel LP with one leader variable and one follower variable. Good for verifying basic solver functionality.

- **Leader**: minimizes −x − 7y subject to −2x + y ≤ 4
- **Follower**: minimizes −y subject to −x + y ≤ 1, x + y ≤ 5, y ≥ 0
- **Optimal**: x = 2, y = 3, leader objective = −23

```bash
bicut solve --input examples/simple_bilevel.toml
bicut compile --input examples/simple_bilevel.toml --reformulation kkt --output simple_kkt.mps
```

### `interdiction.toml`

A shortest-path network interdiction problem on a small graph. The leader removes up to K edges to maximize the attacker's shortest-path cost.

- **Leader**: selects ≤ 1 edge to interdict (binary variables)
- **Follower**: finds the minimum-cost s–t flow in the residual network
- **Optimal objective**: 5

```bash
bicut solve --input examples/interdiction.toml
bicut compile --input examples/interdiction.toml --reformulation vf --output interdiction_vf.mps
```

### `strategic_bidding.toml`

A strategic bidding problem in a simplified electricity market. A producer chooses bid prices to maximize profit, anticipating that the market operator clears the market at minimum cost.

- **Leader**: sets bid prices for two generators to maximize profit
- **Follower**: dispatches generators to meet 150 MW demand at minimum cost
- **Note**: contains bilinear terms in the leader objective (bid × dispatch)

```bash
bicut solve --input examples/strategic_bidding.toml
bicut compile --input examples/strategic_bidding.toml --reformulation sd --output bidding_sd.mps
```

## Running Examples

### Solve directly

```bash
bicut solve --input <file.toml> [--backend gurobi|scip|highs|cplex] [--time-limit 60]
```

### Compile to single-level MILP

```bash
bicut compile --input <file.toml> --reformulation <kkt|sd|vf|auto> --output <output.mps>
```

Reformulation options:
- `kkt` — KKT conditions (requires constraint qualifications at follower level)
- `sd` — Strong duality (exact for LP lower levels)
- `vf` — Value function (always valid, may require big-M bounds)
- `auto` — Automatically select the best reformulation

### Inspect problem structure

```bash
bicut info --input <file.toml>
```

### Verify constraint qualifications

```bash
bicut check-cq --input <file.toml>
```
