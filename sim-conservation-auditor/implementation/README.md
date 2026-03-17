# ConservationLint

**Automatic conservation-law auditing for physics simulations.**

[![Crates.io](https://img.shields.io/crates/v/conservation-lint.svg)](https://crates.io/crates/conservation-lint)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/conservation-lint/conservation-lint/ci.yml)](https://github.com/conservation-lint/conservation-lint/actions)

ConservationLint monitors physics simulations for violations of fundamental
conservation laws — energy, momentum, angular momentum, charge, mass, symplectic
structure, and vorticity. It bridges **Noether's theorem** and **program
analysis** to detect, classify, localize, and repair conservation-law violations
in numerical simulations.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
- [Conservation Laws](#conservation-laws)
- [Integrators](#integrators)
- [Benchmarks](#benchmarks)
- [Library Usage](#library-usage)
- [Format Support](#format-support)
- [Examples](#examples)
- [Comparison with Existing Tools](#comparison-with-existing-tools)
- [Theory](#theory)
- [Crate Structure](#crate-structure)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Overview

Conservation violations are the **silent killers** of computational science.
Unlike crashes or type errors, a conservation violation produces
*plausible-looking but quantitatively wrong* results. Energy created from
nothing doesn't trigger an exception — it just makes your climate 0.3 °C warmer
over a century. Angular momentum leaking from a molecular system doesn't
segfault — it just shifts your free energy estimate by 2 kJ/mol.

Existing monitors (GROMACS `gmx energy`, LAMMPS `thermo_style`) detect
*that* conservation is violated but not *why* or *where*. ConservationLint
provides:

1. **Detection** — statistical and symbolic violation detection with
   configurable tolerance
2. **Classification** — categorize violations as secular drift, oscillatory,
   stochastic, or catastrophic
3. **Localization** — trace violations to specific time intervals and
   subsystem components
4. **Repair** — projection-based corrections to restore conservation
   properties
5. **Obstruction analysis** — determine whether a violation is locally
   repairable or architecturally unfixable

---

## Key Features

- **8 conservation laws** — energy, linear momentum, angular momentum, mass,
  charge, symplectic form, vorticity, center-of-mass
- **14+ integrators** — from forward Euler to 8th-order Yoshida, with
  symplecticity verification
- **Statistical detection** — χ², Kolmogorov–Smirnov, Grubbs, CUSUM,
  Page–Hinkley, ADWIN
- **Anomaly detection** — Z-score, IQR, isolation forest, local outlier
  factor
- **Change-point localization** — PELT, binary search, cost-function methods
- **Ensemble detection** — majority voting, weighted voting, unanimity
- **Backward error analysis** — modified equation computation, modified
  Hamiltonians, shadow orbits
- **Spectral analysis** — DFT, power spectral density, windowing functions
- **Phase-space analysis** — Poincaré sections, return maps, KAM torus
  detection
- **Lyapunov exponents** — finite-time Lyapunov exponents and spectra
- **Repair strategies** — orthogonal projection, SHAKE/RATTLE, velocity
  scaling, Newton–Raphson, BFGS
- **Trace recording** — delta compression, checkpointing, replay, filtering
- **Multiple output formats** — text, JSON, SARIF, CSV, Markdown
- **VTK and HDF5 support** — read/write scientific data formats
- **nalgebra and ndarray integration** — interoperate with Rust's numerical
  ecosystem
- **CLI tool** — `conservation-lint audit`, `conservation-lint bench`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    conservation-lint CLI                      │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│sim-monitor│sim-detect│sim-analysis│sim-repair│  sim-trace     │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│                      sim-laws                                │
├──────────────────────────────────────────────────────────────┤
│               sim-integrator    sim-eval                     │
├──────────────────────────────────────────────────────────────┤
│                      sim-types                               │
├──────────────────────────────────────────────────────────────┤
│                  conservation-types                           │
└──────────────────────────────────────────────────────────────┘
```

### Crate Dependency Graph

```
conservation-types          (core shared types, symmetry, provenance)
       │
  sim-types                 (vectors, particles, fields, trajectories)
       │
  ┌────┼────┬───────┬──────────┐
  │    │    │       │          │
sim-laws  sim-integrator  sim-detect  sim-trace
  │    │    │       │          │
  └────┼────┴───────┴──────────┘
       │
  sim-analysis              (spectral, Lyapunov, phase portrait, backward error)
  sim-monitor               (real-time conservation monitoring)
  sim-repair                (projection, SHAKE/RATTLE, velocity scaling)
  sim-eval                  (benchmark scenarios: Kepler, N-body, fluid, …)
       │
  sim-cli                   (conservation-lint binary)
```

---

## Installation

### From Source

```bash
git clone https://github.com/conservation-lint/conservation-lint.git
cd conservation-lint
cargo install --path sim-cli
```

### From crates.io (when published)

```bash
cargo install conservation-lint
```

### As a Library Dependency

Add individual crates to your `Cargo.toml`:

```toml
[dependencies]
sim-types = { version = "0.1", path = "sim-types" }
sim-laws = { version = "0.1", path = "sim-laws" }
sim-integrator = { version = "0.1", path = "sim-integrator" }
sim-detect = { version = "0.1", path = "sim-detect" }
sim-analysis = { version = "0.1", path = "sim-analysis" }
sim-monitor = { version = "0.1", path = "sim-monitor" }
sim-repair = { version = "0.1", path = "sim-repair" }
```

---

## Quick Start

### Audit a Simulation Trace

```bash
# Run a built-in Kepler orbit benchmark and audit conservation
conservation-lint bench --suite kepler --steps 50000

# Audit a trace file
conservation-lint audit trajectory.json --format json

# List available conservation laws
conservation-lint list laws
```

### Library Usage

```rust
use sim_types::{Particle, Vec3, SimulationState, Tolerance, ToleranceKind};
use sim_laws::{TotalMechanicalEnergy, TotalLinearMomentum, TotalAngularMomentum};
use sim_integrator::{VelocityVerlet, Leapfrog};

// Define a two-body gravitational system
let particles = vec![
    Particle::new(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.5, 0.0)),
    Particle::new(1.0, Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, -0.5, 0.0)),
];
let state = SimulationState::from_particles(particles, 0.0);

// Compute conserved quantities
let energy = TotalMechanicalEnergy::gravitational(&state, 1.0);
let momentum = TotalLinearMomentum::compute(&state);
let ang_momentum = TotalAngularMomentum::compute(&state);

println!("E = {energy:.6}, p = {momentum:?}, L = {ang_momentum:?}");
```

---

## CLI Usage

### `conservation-lint audit`

Audit a simulation trace file for conservation law violations.

```
conservation-lint audit <TRACE> [OPTIONS]

Arguments:
  <TRACE>    Path to the simulation trace file (JSON)

Options:
  -c, --config <FILE>    Configuration file
  -f, --format <FMT>     Output format: text, json, sarif, csv [default: text]
  -o, --output <FILE>    Output file (stdout if omitted)
  -v, --verbose          Increase verbosity (-v, -vv, -vvv)
```

### `conservation-lint bench`

Run built-in benchmark suites with conservation auditing.

```
conservation-lint bench [OPTIONS]

Options:
  -s, --suite <NAME>     Suite: kepler, nbody, spring, fluid, all [default: all]
  -n, --steps <N>        Time steps per benchmark [default: 10000]
```

### `conservation-lint list`

List available conservation laws, integrators, or benchmarks.

```
conservation-lint list <WHAT>

Arguments:
  <WHAT>    What to list: laws, integrators, benchmarks
```

### `conservation-lint report`

Generate a diagnostic report from previous audit results.

```
conservation-lint report <RESULTS> [OPTIONS]

Options:
  -f, --format <FMT>     Format: text, json, sarif, markdown, html [default: text]
```

---

## Conservation Laws

| Law | Description | Noether Symmetry | Implementation |
|-----|-------------|------------------|----------------|
| **Energy** | Total kinetic + potential energy | Time translation | `sim_laws::TotalMechanicalEnergy` |
| **Linear Momentum** | Total momentum Σ mᵢvᵢ | Spatial translation | `sim_laws::TotalLinearMomentum` |
| **Angular Momentum** | Total L = Σ rᵢ × pᵢ | Rotational symmetry | `sim_laws::TotalAngularMomentum` |
| **Mass** | Total mass (particle/continuum) | Phase symmetry | `sim_laws::TotalMass` |
| **Charge** | Total electric charge | U(1) gauge symmetry | `sim_laws::TotalCharge` |
| **Symplectic Form** | Phase-space volume (Liouville) | Hamiltonian flow | `sim_laws::SymplecticFormComputation` |
| **Vorticity** | Circulation / enstrophy | Kelvin's theorem | `sim_laws::Circulation` |
| **Center of Mass** | COM velocity (isolated system) | Galilean invariance | `sim_laws::CenterOfMassVelocity` |

Each law includes:
- Exact computation from simulation state
- Tolerance-based violation detection (relative and absolute)
- Severity classification (info, warning, error, critical)
- Provenance tracking for diagnostic reports

---

## Integrators

ConservationLint includes a comprehensive library of numerical integrators for
benchmarking and comparison:

### Non-Symplectic Methods
| Method | Order | Properties |
|--------|-------|------------|
| Forward Euler | 1 | Explicit, unstable for oscillatory systems |
| Backward Euler | 1 | Implicit, A-stable, heavy damping |
| Improved Euler (Heun) | 2 | Explicit, better stability |
| RK2 (Midpoint) | 2 | Explicit |
| RK4 (Classical) | 4 | Explicit, widely used |
| RK3/8 | 4 | Explicit, alternative 4th-order |
| RKF45 | 4(5) | Embedded pair, adaptive |
| DOPRI5 | 4(5) | Dormand–Prince, adaptive |

### Symplectic Methods
| Method | Order | Properties |
|--------|-------|------------|
| Symplectic Euler A/B | 1 | Simplest symplectic |
| Störmer–Verlet | 2 | Symplectic, time-reversible |
| Velocity Verlet | 2 | Symplectic, most popular in MD |
| Leapfrog (DKD/KDK) | 2 | Symplectic, equivalent to Verlet |
| Ruth 3rd order | 3 | Symplectic |
| Ruth 4th order | 4 | Symplectic, early higher-order |
| Forest–Ruth | 4 | Symplectic, optimized |
| PEFRL | 4 | Position-extended Forest–Ruth-like |
| Yoshida 4th order | 4 | Symplectic, triple-jump |
| Yoshida 6th order | 6 | Symplectic, 7 stages |
| Yoshida 8th order | 8 | Symplectic, 15 stages |

### Implicit Methods
| Method | Order | Properties |
|--------|-------|------------|
| Implicit Midpoint | 2 | Symplectic, symmetric |
| Gauss–Legendre 2 | 2 | Symplectic, A-stable |
| Gauss–Legendre 4 | 4 | Symplectic, A-stable |
| Gauss–Legendre 6 | 6 | Symplectic, A-stable |

### Composition & Splitting
| Method | Properties |
|--------|------------|
| ABA Composition | General symmetric composition |
| Suzuki Composition | Fractal composition |
| Triple Jump | Triple-jump technique |
| Lie–Trotter Splitting | First-order operator splitting |
| Strang Splitting | Second-order operator splitting |

---

## Benchmarks

ConservationLint ships with an extensive suite of physics benchmarks, each with
known analytical solutions for validation:

### Orbital Mechanics
- **Circular Kepler orbit** — constant energy, eccentricity 0
- **Elliptical Kepler orbit** — constant energy, Runge–Lenz vector
- **Figure-eight three-body** — Chenciner–Montgomery choreography
- **Pythagorean three-body** — chaotic scattering, high sensitivity
- **Inner solar system** — four-body Sun/Mercury/Venus/Earth

### Oscillatory Systems
- **Simple harmonic oscillator** — exact sinusoidal solution
- **Anharmonic oscillator** — Duffing equation, energy surfaces
- **Coupled oscillators** — normal mode decomposition
- **Damped/driven oscillator** — non-conservative reference
- **Simple pendulum** — nonlinear but integrable
- **Double pendulum** — chaotic, sensitive to initial conditions
- **Spherical pendulum** — 3D, angular momentum conservation

### Rigid Body Dynamics
- **Free rigid body** — Euler equations, exact Jacobi elliptic solution
- **Symmetric top** — precession and nutation
- **Asymmetric top** — intermediate axis theorem

### Electromagnetic Systems
- **Cyclotron motion** — circular orbit in uniform B-field
- **E×B drift** — crossed electric and magnetic fields
- **Magnetic bottle** — adiabatic invariant (magnetic moment)
- **Coulomb scattering** — Rutherford scattering cross-section

### Fluid Dynamics
- **Linear advection** — exact translation, mass conservation
- **Burgers equation** — shock formation, energy dissipation
- **Shallow water 1D** — mass + momentum conservation
- **Sod shock tube** — Riemann problem with exact solution

### Wave Phenomena
- **Standing wave** — energy oscillation between kinetic/potential
- **Traveling wave** — constant profile propagation

Run benchmarks with:

```bash
# All benchmarks
conservation-lint bench --suite all --steps 50000

# Specific suite
conservation-lint bench --suite kepler --steps 100000
conservation-lint bench --suite nbody --steps 10000
conservation-lint bench --suite fluid --steps 20000
```

---

## Library Usage

### Computing Conservation Quantities

```rust
use sim_types::{Particle, Vec3, SimulationState};
use sim_laws::{KineticEnergy, GravitationalPotentialEnergy, TotalLinearMomentum};

let state = SimulationState::from_particles(vec![
    Particle::new(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
    Particle::new(1.0, Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0)),
], 0.0);

let ke = KineticEnergy::compute(&state);
let pe = GravitationalPotentialEnergy::compute(&state, 1.0);
let momentum = TotalLinearMomentum::compute(&state);
```

### Running an Integrator

```rust
use sim_integrator::{VelocityVerlet, Yoshida4, DOPRI5};

// Velocity Verlet for N-body
let dt = 0.001;
let mut q = vec![1.0, 0.0];  // position
let mut p = vec![0.0, 1.0];  // momentum
let force = |q: &[f64]| -> Vec<f64> {
    let r = (q[0]*q[0] + q[1]*q[1]).sqrt();
    vec![-q[0]/(r*r*r), -q[1]/(r*r*r)]
};

VelocityVerlet::step(&mut q, &mut p, &force, dt);
```

### Using the Monitor

```rust
use sim_monitor::{Monitor, MonitorConfig};

let config = MonitorConfig {
    check_interval: 10,
    relative_tolerance: 1e-8,
    absolute_tolerance: 1e-14,
    max_events: 4096,
};
let mut monitor = Monitor::new(config);

// In your simulation loop:
// monitor.observe(&state);
// for event in monitor.drain_events() {
//     log::warn!("Conservation violation: {:?}", event);
// }
```

### Detecting Violations

```rust
use sim_detect::{Detector, DetectionConfig, Cusum, PageHinkley};

// CUSUM change-point detection
let cusum = Cusum::new(5.0, 0.5);  // threshold, drift

// Page-Hinkley drift detection
let ph = PageHinkley::new(50.0, 0.01);  // threshold, delta
```

### Backward Error Analysis

```rust
use sim_analysis::{BackwardErrorAnalyzer, ModifiedEquation, ModifiedHamiltonian};

// Compute the modified Hamiltonian for a symplectic integrator
// The modified Hamiltonian H~ = H + dt^2 * H_2 + dt^4 * H_4 + ...
// is exactly conserved by the numerical method (to truncation order)
```

---

## Format Support

### VTK (Visualization Toolkit)

ConservationLint can read and write VTK files for visualization of simulation
states and conservation violations:

```rust
use sim_trace::format::{VtkWriter, VtkReader};

// Write particle data to VTK for ParaView visualization
let writer = VtkWriter::new("output.vtk");
writer.write_particles(&particles, &violations)?;

// Read VTK unstructured grid data
let reader = VtkReader::new("simulation.vtk");
let state = reader.read_state()?;
```

### HDF5 (Hierarchical Data Format)

Support for HDF5, the standard format for large-scale scientific simulation
data:

```rust
use sim_trace::format::{Hdf5Writer, Hdf5Reader};

// Write simulation trajectory to HDF5
let writer = Hdf5Writer::new("trajectory.h5")?;
writer.write_trajectory(&trajectory)?;

// Read time-series data from HDF5
let reader = Hdf5Reader::new("simulation.h5")?;
let trajectory = reader.read_trajectory("particles/positions")?;
```

### nalgebra Integration

Seamless conversion between `sim-types` vectors and `nalgebra` types:

```rust
use sim_types::Vec3;
use nalgebra::Vector3;

let v: Vec3 = Vec3::new(1.0, 2.0, 3.0);
let na_v: Vector3<f64> = v.into();
let back: Vec3 = na_v.into();
```

### ndarray Integration

Convert between simulation data and N-dimensional arrays:

```rust
use sim_types::SimulationState;
use ndarray::Array2;

// Export particle positions as an ndarray matrix (N × 3)
let positions: Array2<f64> = state.positions_as_array();
```

---

## Examples

The `examples/` directory contains complete, runnable examples:

| Example | Description |
|---------|-------------|
| `nbody_audit.rs` | N-body gravitational simulation with conservation auditing |
| `fluid_audit.rs` | 1D fluid dynamics (advection + Burgers) with mass/energy tracking |
| `electromagnetic_audit.rs` | Charged particle in EM fields with energy/momentum monitoring |
| `benchmark_integrators.rs` | Compare integrator conservation properties on Kepler orbit |

Run examples:

```bash
cargo run --example nbody_audit
cargo run --example fluid_audit
cargo run --example electromagnetic_audit
cargo run --example benchmark_integrators
```

---

## Comparison with Existing Tools

| Feature | ConservationLint | GROMACS `gmx energy` | LAMMPS `thermo_style` | MATLAB ODE Suite |
|---------|-----------------|---------------------|----------------------|-----------------|
| Energy monitoring | ✅ | ✅ | ✅ | ✅ |
| Momentum monitoring | ✅ | Partial | ✅ | ❌ |
| Angular momentum | ✅ | ❌ | Partial | ❌ |
| Charge conservation | ✅ | ❌ | ❌ | ❌ |
| Symplectic form | ✅ | ❌ | ❌ | ❌ |
| Vorticity/circulation | ✅ | ❌ | ❌ | ❌ |
| Statistical detection | ✅ (χ², KS, CUSUM) | ❌ | ❌ | ❌ |
| Change-point localization | ✅ (PELT, binary search) | ❌ | ❌ | ❌ |
| Anomaly detection | ✅ (Z-score, IQR, LOF) | ❌ | ❌ | ❌ |
| Backward error analysis | ✅ | ❌ | ❌ | ❌ |
| Repair/correction | ✅ (projection, SHAKE) | Thermostat only | Thermostat only | ❌ |
| Multi-law simultaneous | ✅ | ❌ | Partial | ❌ |
| Obstruction detection | ✅ | ❌ | ❌ | ❌ |
| SARIF output | ✅ | ❌ | ❌ | ❌ |
| VTK/HDF5 support | ✅ | ✅ | ✅ | ❌ |
| Programmable API | ✅ (Rust) | C/C++ | C++ | MATLAB |

---

## Theory

ConservationLint is grounded in three pillars of mathematical physics:

### Noether's Theorem (1918)

Every differentiable symmetry of a physical system's action has a
corresponding conservation law. ConservationLint maps:

| Symmetry | Conserved Quantity |
|----------|-------------------|
| Time translation | Energy |
| Spatial translation | Linear momentum |
| Rotation | Angular momentum |
| Phase (U(1)) | Charge |
| Galilean boost | Center-of-mass velocity |

### Backward Error Analysis (Hairer, Lubich & Wanner, 2006)

A numerical integrator applied to a Hamiltonian system H exactly solves a
*modified* Hamiltonian H̃ = H + h²H₂ + h⁴H₄ + ⋯. Symplectic integrators
preserve H̃ exponentially long; non-symplectic methods may exhibit secular
energy drift. ConservationLint computes modified Hamiltonians and tracks which
terms break which conservation laws.

### Symplectic Geometry (Arnold, 1989)

Hamiltonian systems preserve the symplectic 2-form ω = Σ dpᵢ ∧ dqᵢ. Symplectic
integrators preserve a nearby symplectic form; non-symplectic methods contract
or expand phase-space volume. ConservationLint monitors the symplectic form
and Poincaré invariants to verify structural preservation.

---

## Crate Structure

| Crate | Purpose | Lines |
|-------|---------|-------|
| `conservation-types` | Core types, symmetry groups, provenance tags | ~2000 |
| `sim-types` | Vectors, particles, fields, trajectories, tolerances | ~3000 |
| `sim-laws` | Conservation law implementations (8 laws + registry) | ~2000 |
| `sim-integrator` | 14+ numerical integrators with symplecticity checks | ~2500 |
| `sim-detect` | Statistical/symbolic violation detection | ~2000 |
| `sim-analysis` | Spectral, Lyapunov, phase portrait, backward error | ~2000 |
| `sim-monitor` | Real-time conservation monitoring | ~500 |
| `sim-repair` | Projection, SHAKE/RATTLE, velocity scaling, BFGS | ~2500 |
| `sim-trace` | Trace recording, compression, replay, filtering | ~2000 |
| `sim-eval` | Benchmark scenarios with analytical solutions | ~2500 |
| `sim-cli` | CLI binary (`conservation-lint`) | ~200 |
| **Total** | | **~20,000** |

---

## Contributing

Contributions are welcome! Areas of particular interest:

1. **New conservation laws** — implement additional conserved quantities
   (e.g., Casimir invariants, helicity, enstrophy in 2D)
2. **New integrators** — add methods (Boris integrator for plasmas,
   exponential integrators, variational integrators)
3. **Format support** — additional simulation file formats (LAMMPS dump,
   GROMACS XTC/TRR, NetCDF)
4. **Language bindings** — Python bindings via PyO3
5. **Benchmarks** — additional physics scenarios with known conservation
   properties

### Development

```bash
# Build everything
cargo build --workspace

# Run all tests
cargo test --workspace

# Run benchmarks
cargo bench --workspace

# Run the CLI
cargo run -p sim-cli -- list laws
```

---

## Citation

If you use ConservationLint in your research, please cite:

```bibtex
@software{conservationlint2025,
  title     = {ConservationLint: Automatic Conservation-Law Auditing
               for Physics Simulations},
  author    = {ConservationLint Team},
  year      = {2025},
  url       = {https://github.com/conservation-lint/conservation-lint},
  note      = {Bridges Noether's theorem and program analysis to detect,
               classify, localize, and repair conservation-law violations
               in numerical simulations}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file
for details.

---

## References

1. E. Noether, "Invariante Variationsprobleme," *Nachrichten von der
   Gesellschaft der Wissenschaften zu Göttingen*, 1918.
2. E. Hairer, C. Lubich, G. Wanner, *Geometric Numerical Integration*,
   2nd ed., Springer, 2006.
3. R. I. McLachlan, G. R. W. Quispel, "Splitting Methods," *Acta Numerica*,
   11:341–434, 2002.
4. V. I. Arnold, *Mathematical Methods of Classical Mechanics*, 2nd ed.,
   Springer, 1989.
5. B. Leimkuhler, S. Reich, *Simulating Hamiltonian Dynamics*, Cambridge
   University Press, 2004.
6. J. E. Marsden, M. West, "Discrete Mechanics and Variational Integrators,"
   *Acta Numerica*, 10:357–514, 2001.
7. S. Wan et al., "Identifying and correcting a systematic energy conservation
   error in an earth system model," *JAMES*, 2019.
8. H. Yoshida, "Construction of higher order symplectic integrators,"
   *Physics Letters A*, 150(5-7):262–268, 1990.
9. E. Forest, R. D. Ruth, "Fourth-order symplectic integration,"
   *Physica D*, 43(1):105–117, 1990.
