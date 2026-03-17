# CascadeVerify Benchmark Suite

Realistic Kubernetes/Istio topology configurations for benchmarking
CascadeVerify's static analysis at varying scales.

## Topologies

| File | Services | Shape | Purpose |
|------|----------|-------|---------|
| `topologies/chain-10.yaml` | 10 | Linear chain | Basic retry amplification scaling |
| `topologies/chain-50.yaml` | 50 | Linear chain | Scalability stress-test for path enumeration |
| `topologies/star-100.yaml` | 100 | Star (1 → 99) | Fan-out amplification and blast-radius analysis |
| `topologies/mesh-500.yaml` | 500 | Tiered mesh | Large-scale realistic microservice architecture |

## Comparison Baselines

| File | Tool Compared | What It Shows |
|------|---------------|---------------|
| `comparison/kubelinter-baseline.yaml` | KubeLinter | Per-resource issues (missing limits, no probes) vs cross-service cascade risks |
| `comparison/istio-analyze-baseline.yaml` | `istioctl analyze` | Config hygiene (orphaned VS, missing DR) vs retry amplification / timeout violations |

## Running Benchmarks

### Static Analysis (CascadeVerify)

```bash
# Analyze a single topology
cascade-verify check benchmarks/topologies/chain-10.yaml

# Analyze all topologies
for f in benchmarks/topologies/*.yaml; do
  echo "=== $f ==="
  cascade-verify check "$f"
done
```

### Criterion Micro-benchmarks

```bash
cd implementation
cargo bench --bench topology_scaling
```

### Comparison

```bash
# KubeLinter (per-resource linting — misses cross-service cascades)
kube-linter lint benchmarks/comparison/kubelinter-baseline.yaml

# istioctl analyze (config hygiene — misses retry amplification)
istioctl analyze benchmarks/comparison/istio-analyze-baseline.yaml

# CascadeVerify (finds the cascade risks others miss)
cascade-verify check benchmarks/comparison/kubelinter-baseline.yaml
cascade-verify check benchmarks/comparison/istio-analyze-baseline.yaml
```

## Expected Results

- **chain-10**: Amplification factor 4^9 = 262,144× at the tail service
- **chain-50**: Amplification factor 4^49 ≈ 3.2×10^29× (exponential blowup)
- **star-100**: 99 parallel fan-out paths, each with 4× amplification
- **mesh-500**: Multiple cross-tier amplification paths; worst-case depends on
  longest retry-stacking path through the tier hierarchy
