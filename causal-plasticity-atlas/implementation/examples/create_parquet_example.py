#!/usr/bin/env python3
"""Generate a sample Parquet file for CPA.

Creates a synthetic multi-context dataset and saves it as Parquet,
demonstrating how to prepare data for ``cpa run --data data.parquet``.

Usage
-----
::

    python examples/create_parquet_example.py
    cpa run --data example_contexts.parquet --context-column context --profile fast
"""

from __future__ import annotations

import numpy as np

try:
    import pandas as pd
except ImportError:
    raise SystemExit("pandas is required: pip install pandas")


def main() -> None:
    rng = np.random.default_rng(42)

    K, n, p = 3, 200, 5
    var_names = [f"X{i}" for i in range(p)]
    rows = []

    for k in range(K):
        data = rng.standard_normal((n, p))
        # Add a simple causal shift per context
        data[:, 1] += 0.5 * data[:, 0] * (1 + 0.3 * k)
        data[:, 2] += 0.4 * data[:, 1]

        df_k = pd.DataFrame(data, columns=var_names)
        df_k["context"] = f"ctx_{k}"
        rows.append(df_k)

    df = pd.concat(rows, ignore_index=True)
    out = "example_contexts.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} rows × {len(var_names)} variables to {out}")
    print(f"Contexts: {df['context'].unique().tolist()}")
    print()
    print("Load with CPA:")
    print(f"  cpa run --data {out} --context-column context --profile fast")
    print()
    print("Or from Python:")
    print("  from cpa.io.readers import ParquetReader")
    print(f'  dataset = ParquetReader("{out}", context_column="context").read()')


if __name__ == "__main__":
    main()
