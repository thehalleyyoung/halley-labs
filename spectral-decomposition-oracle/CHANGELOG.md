# Changelog

All notable changes to the Spectral Decomposition Oracle will be documented in
this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2025-03-09

### Added

#### Workspace & Build
- Initial 7-crate Rust workspace structure
- Workspace-level `Cargo.toml` with shared dependency management
- CI configuration for formatting, linting, testing, and documentation

#### `spectral-types` (v0.1.0)
- Compressed Sparse Row (`CsrMatrix`) and Compressed Sparse Column (`CscMatrix`)
  matrix types with generic element support
- Coordinate format (`CooMatrix`) for matrix construction and conversion
- `Hypergraph` and `HypergraphBuilder` types for constraint hypergraph
  representation
- `SpectralFeatures` struct for extracted eigenvalue-based feature vectors
- `OracleConfig` for TOML-based configuration parsing
- `SpectralError` unified error type with variants for all crate errors
- Common traits: `MatrixOps`, `Decomposable`, `Certifiable`

#### `spectral-core` (v0.1.0)
- Hypergraph Laplacian construction (unnormalized, normalized, signless variants)
- Lanczos eigensolver with full reorthogonalization
- LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient) eigensolver
- Spectral clustering via eigenvector embedding and k-means
- Eigengap heuristic for automatic cluster count detection
- Spectral feature extraction (algebraic connectivity, spectral gap, spectral
  radius, Cheeger constant bound, eigenvalue distribution statistics)
- Support for computing top-k and bottom-k eigenvalues

#### `matrix-decomp` (v0.1.0)
- LU factorization with partial pivoting
- QR factorization via Householder reflections
- Singular Value Decomposition (SVD) via bidiagonalization
- Cholesky factorization for symmetric positive definite matrices
- Lanczos tridiagonalization for symmetric matrices
- LOBPCG iterative eigensolver implementation
- Householder reflection computation and application
- Givens rotation computation and application
- Forward and backward substitution solvers

#### `oracle` (v0.1.0)
- Random forest classifier for decomposition strategy prediction
- Gradient boosting classifier with configurable learning rate and depth
- Logistic regression (multinomial) with L2 regularization
- Matrix structure detection: block-angular, staircase, bordered
  block-diagonal, arrowhead patterns
- Feature importance analysis for interpretable predictions
- Cross-validation support for model evaluation
- Strategy prediction with confidence scores

#### `optimization` (v0.1.0)
- Benders decomposition with single and multi-cut variants
- Magnanti-Wong cut strengthening for Benders
- Dantzig-Wolfe column generation with reduced cost and Farkas pricing
- Du Merle stabilization for Dantzig-Wolfe
- Lagrangian relaxation with subgradient, bundle, and cutting plane methods
- Proximal bundle method with serious/null step management
- Camerini, Polyak, diminishing, and fixed step size rules for subgradient
- Revised simplex method for LP subproblems
- Interior point solver (Mehrotra predictor-corrector) for LP subproblems
- Warm-start support for iterative decomposition

#### `certificate` (v0.1.0)
- Davis-Kahan sin(Θ) theorem certificates for subspace stability
- Weyl inequality certificates for eigenvalue perturbation bounds
- Residual bound certificates for per-eigenvalue accuracy verification
- Lemma L3 partition-to-bound bridge certificates connecting spectral partition
  quality to optimization bound quality
- Proposition T2 spectral scaling law estimation for quality extrapolation
- Futility predictor to detect when decomposition is unlikely to help
- Certificate output in JSON, LaTeX, and human-readable formats

#### `spectral-cli` (v0.1.0)
- `analyze` subcommand for spectral feature extraction
- `certify` subcommand for certificate generation
- `decompose` subcommand for executing decomposition strategies
- `benchmark` subcommand for performance evaluation
- TOML configuration file support (`spectral-oracle.toml`)
- MPS, LP, and MTX input format parsing
- JSON, CSV, and binary output formats
- Verbose and quiet output modes
- Multi-threaded execution with configurable thread count

#### Documentation
- Comprehensive README with architecture diagrams, examples, and theory overview
- API documentation for all public items
- CONTRIBUTING guide with development setup and code style guidelines
- CHANGELOG following Keep a Changelog format

#### Examples
- PCA verification example demonstrating eigenvalue certification
- Graph Laplacian analysis example with spectral clustering
- Quantum Hamiltonian verification example with perturbation bounds

[0.1.0]: https://github.com/spectral-decomp/spectral-decomposition-oracle/releases/tag/v0.1.0
