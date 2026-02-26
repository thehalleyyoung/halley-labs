# VMEE API Reference

Public classes and methods grouped by module.

---

## `vmee.config`

```python
from vmee.config import VMEEConfig, load_config, default_config, save_config
config = default_config()           # Returns VMEEConfig with all defaults
config = load_config("config.toml") # Load from TOML
```

Sub-configs: `LOBConfig`, `CausalConfig`, `BayesianConfig`, `TemporalConfig`, `ProofConfig`, `AdversarialConfig`, `EvidenceConfig`, `CalibrationConfig`.

---

## `vmee.lob`

| Class | Key Methods |
|-------|-------------|
| `LOBSimulator(config)` | `generate_trading_day(seed)` → LOB data with 44 microstructure features |
| `ManipulationPlanter(config)` | `plant_spoofing(data)`, `plant_layering(data)`, `plant_wash_trading(data)`, `plant_sec_scenario(data, name)` |

---

## `vmee.causal`

| Class | Key Methods |
|-------|-------------|
| `CausalDiscoveryEngine(config)` | `discover(data)`, `run_prior_sensitivity(data)`, `compute_misspecification_bound(dag, data)`, `detect_structural_changes(data, window)` |
| `HSICTest(kernel, bandwidth)` | `test(x, y, z)` → p-value for X ⊥ Y \| Z |
| `DoCalculusEngine(dag)` | `identify_effect(treatment, outcome)` → backdoor/frontdoor identification |
| `MultipleTestingCorrection` | `apply_holm_bonferroni(pvalues, alpha)` (static), `compute(pvalues, alpha)` |
| `FCIEngine(alpha)` | `discover(data)` → PAG with edge marks (handles latent confounders) |
| `GESEngine()` | `discover(data)` → DAG via greedy equivalence search |
| `PriorSensitivityAnalysis()` | `run(data, dag)` → posteriors under informative/skeptical/uniform priors + robustness flag |
| `DAGMisspecificationBound(dag, data)` | `degradation_curve()` → TV distance vs. edge perturbation |
| `StructuralBreakTest(window_size)` | `detect(data)` → changepoint indices |

```python
from vmee.causal.discovery import CausalDiscoveryEngine, FCIEngine, GESEngine

dag = CausalDiscoveryEngine(config.causal).discover(data)
pag = FCIEngine(alpha=0.05).discover(data)
dag = GESEngine().discover(data)
```

---

## `vmee.bayesian`

| Class | Key Methods |
|-------|-------------|
| `BayesianInferenceEngine(config)` | `infer(dag, data)`, `multi_prior_inference(dag, data)`, `verify_circuit_brute_force(dag, data)` |
| `ArithmeticCircuit(variables)` | `add_gate(type, inputs)`, `evaluate(evidence)`, `check_decomposability()`, `check_determinism()`, `get_trace()` |
| `TreeDecomposition(graph)` | `decompose()` → tree decomposition for treewidth bound |
| `ManipulationHMM(n_states)` | `set_emission_params(means, stds)`, `forward_backward(obs)`, `viterbi(obs)` |

```python
from vmee.bayesian.engine import BayesianInferenceEngine, ManipulationHMM

result = BayesianInferenceEngine(config.bayesian).infer(dag, data)
hmm = ManipulationHMM(n_states=3)
hmm.set_emission_params(means, stds)
path = hmm.viterbi(observations)
```

---

## `vmee.temporal`

| Class | Key Methods |
|-------|-------------|
| `Formula` | `is_in_decidable_fragment()` → BMTL_safe check, `max_future_bound()` |
| `TemporalMonitor(config)` | `monitor(events)` → violations, `reduce_to_qf_lra(violations)` → QF_LRA claims |

```python
from vmee.temporal.monitor import TemporalMonitor
violations = TemporalMonitor(config.temporal).monitor(events)
```

---

## `vmee.proof`

| Class | Key Methods |
|-------|-------------|
| `ProofBridge(config)` | `encode_evidence_claim(claim)`, `encode_circuit_trace(trace)`, `generate_proof(claim)`, `generate_proofs(claims)`, `validate_translation(claim, proof)` |
| `SoundnessTheorem` | `construct(bridge, claim)` (static) → Theorem 1 instance |
| `QFLRAEncodingSpec` | `create(variables, constraints)` (static), `describe_encoding()` |
| `TCBAnalysis()` | `verified_components()`, `unverified_components()`, `verified_fraction()` → 5/9, `summary()` |
| `CircuitEncodingCertificate()` | `add_step(gate_id, error)`, `within_budget()`, `summary()` |
| `EncodingErrorBudget(precision_bits)` | `summary()` |

Standalone functions: `mccormick_lower(x, y, x_lo, x_hi, y_lo, y_hi)`, `mccormick_upper(...)`, `extract_z3_proof_object(solver)`

```python
from vmee.proof.bridge import ProofBridge, TCBAnalysis, extract_z3_proof_object

bridge = ProofBridge(config.proof)
proof = bridge.generate_proof(claim)
bridge.validate_translation(claim, proof)

tcb = TCBAnalysis()
print(tcb.verified_fraction())  # 0.556 (5/9)
```

---

## `vmee.composition`

| Class | Key Methods |
|-------|-------------|
| `CompositionFramework()` | `check_soundness(causal, bayesian, temporal, proofs)` → cross-module compatibility |

---

## `vmee.adversarial`

| Class | Key Methods |
|-------|-------------|
| `AdversarialTrainer(config)` | `train(env, episodes)` → REINFORCE training with coverage tracking (coverage=1.0) |

```python
from vmee.adversarial.trainer import AdversarialTrainer
results = AdversarialTrainer(config.adversarial).train(env, episodes=1000)
```

---

## `vmee.evidence`

| Class | Key Methods |
|-------|-------------|
| `EvidenceAssembler(config)` | `assemble(causal, bayesian, temporal, proofs)`, `save_bundle(bundle, path)` |
| `BundleVerifier()` | `verify(path)`, `verify_bundle(bundle_dict)` |

```python
from vmee.evidence.assembler import EvidenceAssembler, BundleVerifier
bundle = EvidenceAssembler(config.evidence).assemble(causal=dag, bayesian=result,
                                                      temporal=violations, proofs=proofs)
BundleVerifier().verify("evidence_bundle.json")
```

---

## `vmee.calibration`

| Class | Key Methods |
|-------|-------------|
| `SimToRealCalibrator(config)` | `calibrate(sim_data)` → KS-statistic comparison against published LOB stats |

---

## `vmee.evaluation`

| Class | Key Methods |
|-------|-------------|
| `BenchmarkRunner(config)` | `run_full_evaluation()`, `run_scenario(name)`, `save_results(results, path)` |
| `BaselineDetector(method)` | `detect(features)` — threshold-based baseline |
| `StatisticalBaselineDetector(method)` | `detect(features)` — z-score / statistical test baseline |

```python
from vmee.evaluation.benchmark import BenchmarkRunner
results = BenchmarkRunner(config).run_full_evaluation()
```

---

## `vmee.cli`

Entry point: `vmee`. Commands:

```bash
vmee run --output results/               # Full pipeline
vmee verify results/evidence_bundle.json  # Verify bundle
vmee evaluate --scenarios sarao_2010      # Run benchmarks
vmee generate-data --output data/         # Generate synthetic data
vmee adversarial --output adv/            # RL stress-testing
vmee calibrate --output cal/              # Sim-to-real calibration
vmee init-config --output config.toml     # Generate default config
```
