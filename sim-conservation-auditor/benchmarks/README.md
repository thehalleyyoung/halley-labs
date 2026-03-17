# ConservationLint Benchmark Suite

Evaluation of ConservationLint on 25 simulation kernels with known ground-truth
conservation properties, compared against GROMACS `gmx energy`, LAMMPS `thermo_style`,
and the MATLAB ODE suite.

## Benchmark Kernels

The 25 kernels span five categories of numerical integrators:

### Symplectic Integrators (8 kernels)
| Kernel | Integrator | Conservation Laws | Expected |
|--------|-----------|-------------------|----------|
| `verlet_harmonic` | Velocity Verlet | E, p, L | All preserved |
| `leapfrog_kepler` | Leapfrog | E, p, L | All preserved |
| `stormer_verlet_spring` | Stormer-Verlet | E | Preserved |
| `ruth3_henon_heiles` | Ruth 3rd order | E | Preserved |
| `yoshida4_solar` | Yoshida 4th order | E, p, L | All preserved |
| `forest_ruth_duffing` | Forest-Ruth | E | Preserved |
| `composition_yoshida6` | Yoshida 6th order | E | Preserved |
| `symplectic_euler_pendulum` | Symplectic Euler | E | Bounded oscillation |

### Splitting Methods (3 kernels)
| Kernel | Method | Expected Violation |
|--------|--------|-------------------|
| `strang_split_wave` | Strang splitting | None (symmetric) |
| `lie_trotter_schrodinger` | Lie-Trotter | Energy (1st order) |
| `leapfrog_ewald_split` | Leapfrog + Ewald | Angular momentum (architectural) |

### Thermostat / Stochastic (3 kernels)
| Kernel | Method | Expected |
|--------|--------|----------|
| `verlet_thermostat_nvt` | Verlet + Berendsen | Energy violated (intentional) |
| `nose_hoover_chain` | Nose-Hoover chain | System E violated; extended E preserved |
| `langevin_brownian` | Langevin dynamics | Energy violated (stochastic) |

### Constrained / Multi-rate (4 kernels)
| Kernel | Method | Expected |
|--------|--------|----------|
| `rattle_constrained` | RATTLE | Energy preserved + constraints |
| `shake_water` | SHAKE | Energy preserved + constraints |
| `respa_multi_timestep` | r-RESPA | Energy violated (resonance) |
| `multi_rate_orbital` | Multi-rate leapfrog | E, L violated (coupling) |

### Advanced Methods (7 kernels)
| Kernel | Method | Notes |
|--------|--------|-------|
| `velocity_verlet_lj` | Velocity Verlet (LJ) | Standard MD baseline |
| `implicit_midpoint_stiff` | Implicit midpoint | Outside liftable fragment |
| `gauss_legendre_rk4` | Gauss-Legendre RK4 | Outside liftable fragment |
| `boris_push_emag` | Boris push | Electromagnetic; magnetic moment preserved |
| `pic_vlasov_poisson` | PIC Vlasov-Poisson | Charge preserved; energy violated |
| `sph_fluid_navier_stokes` | SPH | Mass/momentum preserved; energy violated |
| `dg_euler_equations` | DG Euler | Mass/momentum preserved; energy violated |

## Running Benchmarks

### Quick mode (~2 minutes)
```bash
./benchmarks/run_benchmarks.sh --quick
```

### Full mode (~30 minutes)
```bash
./benchmarks/run_benchmarks.sh --full
```

## Results Summary

| Metric | ConservationLint | GROMACS `gmx energy` | LAMMPS `thermo` |
|--------|:---:|:---:|:---:|
| Detection rate | **92.0%** (23/25) | 68.0% (17/25) | 72.0% (18/25) |
| Localization accuracy | **91.2%** (21/23) | 0% | 0% |
| Obstruction accuracy | **95.7%** (22/23) | 0% | 0% |
| False positive rate | 0% | 12% | 8% |
| Median analysis time | 11.3s | N/A (runtime) | N/A (runtime) |

### Key observations

1. **ConservationLint misses 2/25 benchmarks** — both are implicit methods
   (`implicit_midpoint_stiff`, `gauss_legendre_rk4`) that fall outside the
   liftable fragment. This is a known limitation of the static analysis approach.

2. **GROMACS and LAMMPS detect only energy-related violations** and cannot
   detect angular momentum, charge, or symplecticity violations without
   explicit configuration. Neither provides source-code localization.

3. **Localization accuracy** (91.2%) is measured as: given a correctly detected
   violation, did ConservationLint identify the correct source region? The 2
   failures occur in complex grid-particle coupling (PIC) and SPH kernels.

4. **Obstruction classification** is correct 95.7% of the time. The single
   failure misclassifies artificial viscosity dissipation as locally repairable.

## Reproducing Results

```bash
cargo build --release
python3 -m pip install numpy scipy
./benchmarks/run_benchmarks.sh --full --output results_reproduced.json
python3 scripts/compare_results.py results.json results_reproduced.json
```

## Known Limitations

- Implicit methods (midpoint, Gauss-Legendre) are outside the liftable fragment
- PIC methods with grid-particle coupling have noisy localization
- Analysis time scales superlinearly with code complexity for kernels >300 LoC
- Stochastic integrators (Langevin) are detected but localization is approximate
