# Changelog

All notable changes to IsoSpec will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-01

### Added

- **isospec-types**: Core type definitions for transactions, operations,
  predicates, isolation levels, dependencies, schedules, SMT constraints,
  schema, and configuration.
- **isospec-core**: Analysis engine with DSG construction, cycle detection,
  SMT encoding, refinement checking, portability analysis, conflict resolution,
  predicate analysis, and mixed-isolation optimization.
- **isospec-engines**: Labeled transition system models for PostgreSQL 16.x
  SSI, MySQL 8.0 InnoDB gap locking, and SQL Server 2022 dual-mode concurrency.
- **isospec-history**: Transaction history parsing, builder API, replay engine,
  and trace analysis.
- **isospec-anomaly**: Anomaly detection (G0–G2), isolation level classification,
  Hermitage test suite integration, and structured reporting.
- **isospec-smt**: SMT solver interface with SMT-LIB2 encoding, incremental
  solving, model extraction, and MaxSMT optimization support.
- **isospec-witness**: Witness schedule synthesis, SQL script generation,
  MUS-based minimization, and timing control.
- **isospec-adapter**: Database adapter layer with connection management,
  SQL parsing, query execution, and Docker-based validation.
- **isospec-bench**: Benchmarking harness with TPC-C and TPC-E workload
  generators, metrics collection, and report generation.
- **isospec-cli**: Command-line interface with analyze, portability, classify,
  optimize, validate, and bench commands.
- **Documentation**: README with architecture, examples, theory overview,
  benchmarks, and API reference.
- **Tool paper**: LaTeX paper with SOTA comparison against Jepsen, Elle,
  DBcop, Cobra, and Hermitage.
- **Groundings**: 25 grounded claims with evidence and citations.

### Known Issues

- MySQL gap lock over-approximation introduces ~2.7% false positive rate.
- Predicate-level decidability requires NOT NULL columns; nullable columns
  use sound over-approximation.
- SMT performance degrades beyond 200 transactions; windowed analysis
  recommended for larger workloads.
