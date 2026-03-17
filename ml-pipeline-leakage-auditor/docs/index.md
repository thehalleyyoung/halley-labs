# TaintFlow Documentation

**TaintFlow** is a static/abstract-interpretation tool that detects data leakage
in machine-learning pipelines. It models information flow through
fit/transform/predict operations using a partition-taint lattice and bounds the
severity of leakage via channel capacity theory.

## Contents

### Theory

- [Theory Overview](theory.md) — Partition-taint lattice, channel capacity
  bounds, fit-transform decomposition, and the soundness theorem that guarantees
  TaintFlow never misses true leakage.

### API Reference

- [API Reference](api.md) — Detailed documentation for all public classes and
  functions: `TaintElement`, `PartitionTaintLattice`, `ColumnTaintMap`,
  `DataFrameAbstractState`, `ChannelCapacityBound`, and `TaintFlowConfig`.

### Examples

- [StandardScaler Leakage](../examples/standardscaler_leakage.py) — Detecting
  leakage when `StandardScaler.fit_transform` is called before train/test split.
- [Target Encoding Leakage](../examples/target_encoding_leakage.py) — Detecting
  target variable information leaking into features via target encoding.
- [Correct Pipeline](../examples/correct_pipeline.py) — A properly constructed
  sklearn `Pipeline` with no leakage, verified clean by TaintFlow.

### Contributing

- [Contributing Guide](../CONTRIBUTING.md) — Development setup, code style,
  testing, and how to add new detectors and capacity models.

### Project

- [Changelog](../CHANGELOG.md) — Release history and upcoming changes.
- [License](../LICENSE) — MIT License.
