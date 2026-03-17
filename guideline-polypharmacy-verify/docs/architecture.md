# GuardPharma Architecture

## System Overview

GuardPharma implements a two-tier formal verification engine for polypharmacy safety. This document describes the architectural design decisions and data flow.

## Data Flow

```
Clinical Guidelines (JSON/TOML)    Patient Profile (TOML/JSON)
            │                                │
            ▼                                ▼
    ┌───────────────┐               ┌───────────────┐
    │  Guideline    │               │   Clinical    │
    │  Parser       │               │   State Space │
    │  + PTA Build  │               │   Model       │
    └───────┬───────┘               └───────┬───────┘
            │                                │
            ▼                                ▼
    ┌─────────────────────────────────────────────┐
    │         PTA Composition Engine               │
    │  (Product automaton + CYP interface          │
    │   contracts + PK state modeling)             │
    └─────────────────┬───────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   ┌──────────────┐      ┌──────────────────┐
   │   TIER 1     │      │     TIER 2       │
   │   Abstract   │─────▶│  Model Checker   │
   │   Screening  │ flag │  (Contract-based  │
   │              │      │   + BMC + CEGAR)  │
   └──────┬───────┘      └────────┬─────────┘
          │ safe                  │ unsafe/safe
          ▼                       ▼
   ┌──────────────────────────────────────────┐
   │        Clinical Significance Filter       │
   │  (Beers + DrugBank + FAERS + Medicare)    │
   └──────────────────┬───────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   ┌──────────────┐      ┌──────────────────┐
   │   Safety     │      │  Conflict Report │
   │   Certificate│      │  + Counterexample│
   │              │      │  + Narrative     │
   └──────────────┘      └──────────────────┘
```

## Crate Dependency Graph

```
types ──┬──▶ pk-model ──┬──▶ abstract-interp
        │               │
        ├──▶ clinical   ├──▶ smt-encoder
        │               │
        │               └──▶ model-checker
        │
        ├──▶ guideline-parser
        │
        ├──▶ conflict-detect ──▶ significance
        │                   └──▶ recommendation
        │
        └──▶ evaluation ──▶ cli
```

## Key Design Decisions

### 1. Two-Tier Verification

Abstract interpretation (Tier 1) handles ~75% of drug pairs in <1s each by computing PK concentration interval over-approximations. Only pairs flagged "possibly unsafe" proceed to Tier 2 (compositional model checking), which produces concrete counterexample trajectories.

### 2. Contract-Based Composition

Instead of building monolithic product automata (exponential), we decompose multi-guideline verification into per-guideline + per-enzyme-interface checks. Each guideline carries an (assume, guarantee) contract over shared CYP enzymes.

### 3. Metzler Matrix Structure

All linear compartmental PK models produce Metzler system matrices (non-negative off-diagonal). This structure guarantees monotone solution trajectories, enabling efficient zonotopic reachability and the widening convergence bound.

### 4. Separation of Clinical Significance

Not every formal conflict is clinically actionable. The significance filter integrates four independent severity signals to reduce false-positive burden on clinicians.
