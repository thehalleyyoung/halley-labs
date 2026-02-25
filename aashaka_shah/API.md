# TOPOS — API Reference

See [topos/API.md](topos/API.md) for the full API reference.

## Quick Overview

```python
from api import recommend_algorithm_with_confidence

r = recommend_algorithm_with_confidence("dgx-h100-4node", "25MB")
# Returns: algorithm, confidence, is_ood, costs, cost_ratios, decomposition

from smt_analysis import run_verification_experiment
result = run_verification_experiment()
# Returns: initial/final verification rates, counterexamples, Z3/RF timings

from expanded_topology_dataset import generate_expanded_dataset, dataset_summary
entries = generate_expanded_dataset()
# Returns: 1,842 entries across 175 topologies (up to 128 nodes)
```

All imports assume `cd topos/`.
