# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- **Core lattice module** (`taintflow/lattice.py`): `TaintElement` with four-level
  partition-taint lattice (⊥ → Train → Test → ⊤), join/meet operations, and
  `PartitionTaintLattice` with verified monotonicity.
- **Column taint tracking** (`taintflow/taint_map.py`): `ColumnTaintMap` for
  per-column taint propagation and `DataFrameAbstractState` for tracking taint
  through dataframe operations.
- **Channel capacity models** (`taintflow/capacity/`):
  - `MutualInformationBound` — upper bound via mutual information between
    train and test partitions through a shared transformer.
  - `CountingBound` — discrete counting bound for categorical encoders.
  - `GaussianChannelBound` — closed-form bound for Gaussian-parameterised
    transformers (e.g., StandardScaler).
- **PI-DAG extraction** (`taintflow/dag.py`): Parse sklearn `Pipeline` and
  `ColumnTransformer` objects into a directed acyclic graph of
  fit/transform/predict operations with taint annotations.
- **Pattern detectors** (`taintflow/detectors/`):
  - `FitBeforeSplitDetector` — detects `fit_transform` called on the full
    dataset before `train_test_split`.
  - `TargetLeakageDetector` — detects target variable information flowing
    into feature columns.
- **CLI scaffolding** (`taintflow/cli.py`): `taintflow audit <script.py>` command
  with JSON and human-readable output formats.
- **Documentation**: theory overview, API reference, and examples.
- **Test suite**: 95%+ coverage on core lattice and capacity modules.
