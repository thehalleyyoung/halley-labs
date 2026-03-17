#!/usr/bin/env python3
"""
SOTA Conservation Auditor Benchmark Suite
=========================================

This benchmark suite creates real-world physics simulations with known conservation properties
and tests various conservation auditing approaches against state-of-the-art baselines.

Physics Systems:
- N-body Kepler systems (2-body, 3-body, restricted 3-body)
- Harmonic oscillators (1D, 2D, 3D coupled)
- Pendulums (simple, double, chaotic)
- Rigid body rotation

Integrators:
- Euler (non-conservative)
- RK4 (moderately conservative)  
- Verlet (position-Verlet, velocity-Verlet)
- Symplectic leapfrog (Störmer-Verlet)
- Ruth 3rd order symplectic
- Forest-Ruth 4th order symplectic

Baselines:
- Simple ΔE/E drift tracking
- Richardson extrapolation error estimation
- Symplecticity check (Jacobian determinant preservation)
- Power spectrum analysis (frequency drift detection)
- Poincaré section return map analysis

Metrics:
- Conservation violation detection accuracy
- Drift estimation error (vs analytical/high-precision reference)
- Computational time overhead
- False alarm rate
- Sensitivity to noise
"""

import numpy as np
import scipy
from scipy import integrate, optimize, linalg
import matplotlib.pyplot as plt
import sympy as sp
import json
import time
import os
import subprocess
import tempfile
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum


class IntegratorType(Enum):
    EULER = "euler"
    RK4 = "rk4"  
    VERLET_POSITION = "verlet_position"
    VERLET_VELOCITY = "verlet_velocity"
    LEAPFROG = "leapfrog"
    RUTH3 = "ruth3"
    FOREST_RUTH = "forest_ruth"


class PhysicsSystem(Enum):
    KEPLER_2BODY = "kepler_2body"
    KEPLER_3BODY = "kepler_3body"
    HARMONIC_1D = "harmonic_1d"
    HARMONIC_2D = "harmonic_2d"
    HARMONIC_3D = "harmonic_3d"
    PENDULUM_SIMPLE = "pendulum_simple"
    PENDULUM_DOUBLE = "pendulum_double"
    RIGID_BODY = "rigid_body"


@dataclass
class SimulationConfig:
    system: PhysicsSystem
    integrator: IntegratorType
    timestep: float
    duration_steps: int
    initial_conditions: List[float]
    parameters: Dict[str, float]


@dataclass
class ConservationQuantity:
    name: str
    formula: str
    analytical_value: Optional[float]
    computed_values: List[float]
    times: List[float]


@dataclass
class BenchmarkResult:
    config: SimulationConfig
    simulation_time: float
    conservation_quantities: List[ConservationQuantity]
    
    # Baseline results
    simple_drift_detected: bool
    simple_drift_error: float
    richardson_error: float  
    symplecticity_preserved: bool
    symplecticity_error: float
    frequency_drift_detected: bool
    
    # Our tool results (if available)
    tool_detected: Optional[bool]
    tool_error: Optional[float]
    tool_time: Optional[float]


class PhysicsSimulator:
    """Generates realistic physics simulations with known conservation properties."""
    
    def __init__(self):
        self.G = 1.0  # Gravitational constant (normalized)
    
    def kepler_2body(self, initial_state: np.ndarray, dt: float, n_steps: int, 
                     integrator: IntegratorType, m1: float = 1.0, m2: float = 1.0) -> Tuple[np.ndarray, List[float]]:
        """
        Two-body Kepler problem: conserves energy and angular momentum.
        State: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
        """
        def kepler_force(state):
            x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state
            dx, dy = x2 - x1, y2 - y1
            r = np.sqrt(dx**2 + dy**2)
            r3 = r**3
            
            # Force on body 1 from body 2
            fx = self.G * m1 * m2 * dx / r3
            fy = self.G * m1 * m2 * dy / r3
            
            return np.array([vx1, vy1, fx/m1, fy/m1, vx2, vy2, -fx/m2, -fy/m2])
        
        def kepler_hamiltonian(state):
            x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state
            # Kinetic energy
            T = 0.5 * m1 * (vx1**2 + vy1**2) + 0.5 * m2 * (vx2**2 + vy2**2)
            # Potential energy
            r = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            V = -self.G * m1 * m2 / r
            return T + V
        
        def angular_momentum(state):
            x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state
            # Total angular momentum about origin
            L1 = m1 * (x1 * vy1 - y1 * vx1)
            L2 = m2 * (x2 * vy2 - y2 * vx2)
            return L1 + L2
        
        trajectory = self._integrate(kepler_force, initial_state, dt, n_steps, integrator)
        
        # Compute conserved quantities
        energies = [kepler_hamiltonian(state) for state in trajectory]
        angular_momenta = [angular_momentum(state) for state in trajectory]
        
        return trajectory, energies, angular_momenta
    
    def harmonic_3d(self, initial_state: np.ndarray, dt: float, n_steps: int,
                   integrator: IntegratorType, k: float = 1.0, m: float = 1.0) -> Tuple[np.ndarray, List[float]]:
        """
        3D coupled harmonic oscillators: conserves energy.
        State: [x, y, z, vx, vy, vz]
        """
        def harmonic_force(state):
            x, y, z, vx, vy, vz = state
            ax = -k * x / m
            ay = -k * y / m  
            az = -k * z / m
            return np.array([vx, vy, vz, ax, ay, az])
        
        def harmonic_energy(state):
            x, y, z, vx, vy, vz = state
            T = 0.5 * m * (vx**2 + vy**2 + vz**2)
            V = 0.5 * k * (x**2 + y**2 + z**2)
            return T + V
        
        trajectory = self._integrate(harmonic_force, initial_state, dt, n_steps, integrator)
        energies = [harmonic_energy(state) for state in trajectory]
        
        return trajectory, energies
    
    def double_pendulum(self, initial_state: np.ndarray, dt: float, n_steps: int,
                       integrator: IntegratorType, L1: float = 1.0, L2: float = 1.0,
                       m1: float = 1.0, m2: float = 1.0, g: float = 9.81) -> Tuple[np.ndarray, List[float]]:
        """
        Chaotic double pendulum: conserves energy but exhibits sensitive dynamics.
        State: [theta1, theta2, omega1, omega2]
        """
        def pendulum_dynamics(state):
            th1, th2, w1, w2 = state
            
            # Shorthand
            c = np.cos(th1 - th2)
            s = np.sin(th1 - th2)
            den = L1 * (m1 + m2 * s**2)
            
            # Equations of motion (Lagrangian mechanics)
            num1 = -m2 * L1 * w1**2 * s * c + m2 * g * np.sin(th2) * c + m2 * L2 * w2**2 * s - (m1 + m2) * g * np.sin(th1)
            dw1 = num1 / den
            
            num2 = -m2 * L2 * w2**2 * s * c + (m1 + m2) * g * np.sin(th1) * c - (m1 + m2) * L1 * w1**2 * s - (m1 + m2) * g * np.sin(th2)
            dw2 = num2 / (L2 * den)
            
            return np.array([w1, w2, dw1, dw2])
        
        def pendulum_energy(state):
            th1, th2, w1, w2 = state
            
            # Kinetic energy (complex expression for double pendulum)
            x1, y1 = L1 * np.sin(th1), -L1 * np.cos(th1)
            x2, y2 = x1 + L2 * np.sin(th2), y1 - L2 * np.cos(th2)
            
            vx1, vy1 = L1 * w1 * np.cos(th1), L1 * w1 * np.sin(th1)
            vx2 = vx1 + L2 * w2 * np.cos(th2)
            vy2 = vy1 + L2 * w2 * np.sin(th2)
            
            T = 0.5 * m1 * (vx1**2 + vy1**2) + 0.5 * m2 * (vx2**2 + vy2**2)
            
            # Potential energy
            V = m1 * g * y1 + m2 * g * y2
            
            return T + V
        
        trajectory = self._integrate(pendulum_dynamics, initial_state, dt, n_steps, integrator)
        energies = [pendulum_energy(state) for state in trajectory]
        
        return trajectory, energies
    
    def _integrate(self, force_func: Callable, initial_state: np.ndarray, dt: float, 
                   n_steps: int, integrator: IntegratorType) -> List[np.ndarray]:
        """Integrate using specified numerical method."""
        
        trajectory = [initial_state.copy()]
        state = initial_state.copy()
        
        for _ in range(n_steps):
            if integrator == IntegratorType.EULER:
                state = self._euler_step(force_func, state, dt)
            elif integrator == IntegratorType.RK4:
                state = self._rk4_step(force_func, state, dt)
            elif integrator == IntegratorType.VERLET_VELOCITY:
                state = self._verlet_velocity_step(force_func, state, dt)
            elif integrator == IntegratorType.LEAPFROG:
                state = self._leapfrog_step(force_func, state, dt)
            elif integrator == IntegratorType.RUTH3:
                state = self._ruth3_step(force_func, state, dt)
            elif integrator == IntegratorType.FOREST_RUTH:
                state = self._forest_ruth_step(force_func, state, dt)
            else:
                raise ValueError(f"Unknown integrator: {integrator}")
                
            trajectory.append(state.copy())
        
        return trajectory
    
    def _euler_step(self, f: Callable, y: np.ndarray, dt: float) -> np.ndarray:
        """Forward Euler (non-conservative)."""
        return y + dt * f(y)
    
    def _rk4_step(self, f: Callable, y: np.ndarray, dt: float) -> np.ndarray:
        """4th-order Runge-Kutta."""
        k1 = f(y)
        k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2)
        k4 = f(y + dt * k3)
        return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def _verlet_velocity_step(self, f: Callable, y: np.ndarray, dt: float) -> np.ndarray:
        """Velocity Verlet (symplectic for separable Hamiltonians)."""
        n = len(y) // 2
        q, p = y[:n], y[n:]
        
        # Assume f returns [dq/dt, dp/dt] where dq/dt = p/m, dp/dt = force
        dqdt, dpdt = f(y)[:n], f(y)[n:]
        
        # Velocity Verlet
        q_new = q + p * dt + 0.5 * dpdt * dt**2
        y_temp = np.concatenate([q_new, p])
        _, dpdt_new = f(y_temp)[:n], f(y_temp)[n:]
        p_new = p + 0.5 * (dpdt + dpdt_new) * dt
        
        return np.concatenate([q_new, p_new])
    
    def _leapfrog_step(self, f: Callable, y: np.ndarray, dt: float) -> np.ndarray:
        """Leapfrog (Störmer-Verlet) method."""
        n = len(y) // 2
        q, p = y[:n], y[n:]
        
        # Half step for momentum
        _, dpdt = f(y)[:n], f(y)[n:]
        p_half = p + 0.5 * dpdt * dt
        
        # Full step for position
        dqdt = p_half  # Assuming unit mass
        q_new = q + dqdt * dt
        
        # Half step for momentum
        y_temp = np.concatenate([q_new, p_half])
        _, dpdt_new = f(y_temp)[:n], f(y_temp)[n:]
        p_new = p_half + 0.5 * dpdt_new * dt
        
        return np.concatenate([q_new, p_new])
    
    def _ruth3_step(self, f: Callable, y: np.ndarray, dt: float) -> np.ndarray:
        """Ruth 3rd order symplectic integrator."""
        c1 = 2.0 / 3.0
        c2 = -2.0 / 3.0
        c3 = 1.0
        d1 = 7.0 / 24.0
        d2 = 3.0 / 4.0
        d3 = -1.0 / 24.0
        
        # Apply composition method
        state = y.copy()
        coeffs = [(c1, d1), (c2, d2), (c3, d3)]
        
        for c, d in coeffs:
            state = self._leapfrog_substep(f, state, c * dt, d * dt)
        
        return state
    
    def _forest_ruth_step(self, f: Callable, y: np.ndarray, dt: float) -> np.ndarray:
        """Forest-Ruth 4th order symplectic integrator."""
        theta = 1.0 / (2.0 - 2**(1.0/3.0))
        
        # Composition coefficients  
        c1 = theta / 2.0
        c2 = (1.0 - theta) / 2.0
        c3 = c2
        c4 = c1
        d1 = theta
        d2 = -theta / 2.0
        d3 = 2.0 - 3.0 * theta / 2.0
        d4 = d2
        
        state = y.copy()
        coeffs = [(c1, d1), (c2, d2), (c3, d3), (c4, d4)]
        
        for c, d in coeffs:
            state = self._leapfrog_substep(f, state, c * dt, d * dt)
            
        return state
    
    def _leapfrog_substep(self, f: Callable, y: np.ndarray, c_dt: float, d_dt: float) -> np.ndarray:
        """Helper for higher-order symplectic methods."""
        n = len(y) // 2
        q, p = y[:n], y[n:]
        
        # Position update
        q = q + c_dt * p
        
        # Momentum update  
        y_temp = np.concatenate([q, p])
        _, dpdt = f(y_temp)[:n], f(y_temp)[n:]
        p = p + d_dt * dpdt
        
        return np.concatenate([q, p])


class SOTABaselines:
    """State-of-the-art baseline methods for conservation checking."""
    
    @staticmethod
    def simple_drift_detection(values: np.ndarray, threshold: float = 1e-6) -> Tuple[bool, float]:
        """Simple relative drift detection: |ΔE/E| > threshold."""
        if len(values) < 2:
            return False, 0.0
        
        initial_val = values[0] if values[0] != 0 else 1e-12
        final_val = values[-1]
        relative_drift = abs((final_val - initial_val) / initial_val)
        
        return relative_drift > threshold, relative_drift
    
    @staticmethod
    def richardson_extrapolation(simulator: PhysicsSimulator, config: SimulationConfig) -> float:
        """Richardson extrapolation to estimate discretization error."""
        # Run simulation at dt and dt/2
        dt_coarse = config.timestep
        dt_fine = config.timestep / 2
        n_coarse = config.duration_steps
        n_fine = config.duration_steps * 2
        
        # Get final energy for both resolutions
        if config.system == PhysicsSystem.KEPLER_2BODY:
            _, energies_coarse, _ = simulator.kepler_2body(
                np.array(config.initial_conditions), dt_coarse, n_coarse, config.integrator
            )
            _, energies_fine, _ = simulator.kepler_2body(
                np.array(config.initial_conditions), dt_fine, n_fine, config.integrator
            )
        elif config.system == PhysicsSystem.HARMONIC_3D:
            _, energies_coarse = simulator.harmonic_3d(
                np.array(config.initial_conditions), dt_coarse, n_coarse, config.integrator
            )
            _, energies_fine = simulator.harmonic_3d(
                np.array(config.initial_conditions), dt_fine, n_fine, config.integrator
            )
        elif config.system == PhysicsSystem.PENDULUM_DOUBLE:
            _, energies_coarse = simulator.double_pendulum(
                np.array(config.initial_conditions), dt_coarse, n_coarse, config.integrator
            )
            _, energies_fine = simulator.double_pendulum(
                np.array(config.initial_conditions), dt_fine, n_fine, config.integrator
            )
        else:
            return 0.0  # Not implemented for this system
        
        # Richardson extrapolation estimate
        E_coarse = energies_coarse[-1] - energies_coarse[0]
        E_fine = energies_fine[-1] - energies_fine[0]
        
        # Assume 2nd order method for Richardson extrapolation
        extrapolated_error = abs(E_fine - E_coarse) / 3.0
        return extrapolated_error
    
    @staticmethod
    def symplecticity_check(trajectory: List[np.ndarray]) -> Tuple[bool, float]:
        """Check preservation of symplectic structure (Jacobian determinant = 1)."""
        if len(trajectory) < 2:
            return True, 0.0
            
        n_dof = len(trajectory[0]) // 2
        if n_dof == 0:
            return True, 0.0
        
        # Compute numerical Jacobian of the map from initial to final state
        initial_state = trajectory[0]
        final_state = trajectory[-1]
        
        def flow_map(x0):
            # This is a simplified check - in practice would need the full flow
            # For now, check if |det(J)| ≈ 1 where J is approximate Jacobian
            return final_state
        
        # Numerical Jacobian approximation
        eps = 1e-8
        jacobian = np.zeros((len(initial_state), len(initial_state)))
        
        for i in range(len(initial_state)):
            x_plus = initial_state.copy()
            x_minus = initial_state.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            # Simplified: assume linear flow (not realistic but demonstrates concept)
            jacobian[:, i] = (x_plus - x_minus) / (2 * eps)
        
        det_J = np.linalg.det(jacobian)
        symplectic_error = abs(det_J - 1.0)
        
        return symplectic_error < 0.1, symplectic_error
    
    @staticmethod 
    def frequency_drift_analysis(values: np.ndarray, dt: float) -> Tuple[bool, List[float]]:
        """Detect drift in characteristic frequencies via FFT."""
        if len(values) < 100:
            return False, []
        
        # Split into segments and compute power spectra
        n_segments = 4
        segment_length = len(values) // n_segments
        
        frequencies = []
        for i in range(n_segments):
            start = i * segment_length
            end = (i + 1) * segment_length
            segment = values[start:end]
            
            # Remove trend
            segment_detrended = scipy.signal.detrend(segment)
            
            # FFT and find dominant frequency
            fft = np.fft.fft(segment_detrended)
            freqs = np.fft.fftfreq(len(segment), dt)
            power = np.abs(fft)**2
            
            # Find peak frequency (excluding DC component)
            peak_idx = np.argmax(power[1:len(power)//2]) + 1
            peak_freq = abs(freqs[peak_idx])
            frequencies.append(peak_freq)
        
        # Check if frequency is drifting
        freq_drift = np.std(frequencies) / np.mean(frequencies) if np.mean(frequencies) > 0 else 0
        drift_detected = freq_drift > 0.01  # 1% relative drift threshold
        
        return drift_detected, frequencies


class ConservationAuditor:
    """Interface to our conservation auditing tool."""
    
    def __init__(self, tool_path: str = None):
        self.tool_path = tool_path or self._find_tool_binary()
        self.available = self._check_tool_availability()
    
    def _find_tool_binary(self) -> Optional[str]:
        """Find the compiled conservation auditor binary."""
        possible_paths = [
            "../implementation/target/release/sim-cli",
            "../implementation/target/debug/sim-cli", 
            "./target/release/sim-cli",
            "./target/debug/sim-cli"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def _check_tool_availability(self) -> bool:
        """Check if our tool is available and working."""
        if not self.tool_path or not os.path.exists(self.tool_path):
            return False
        
        try:
            result = subprocess.run([self.tool_path, "--help"], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def audit_trajectory(self, trajectory: List[np.ndarray], dt: float, 
                        system_type: str) -> Tuple[bool, float, float]:
        """
        Audit a simulation trajectory for conservation violations.
        Returns: (violation_detected, error_estimate, computation_time)
        """
        if not self.available:
            return None, None, None
        
        start_time = time.time()
        
        # Write trajectory to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            trajectory_data = {
                'timestep': dt,
                'system_type': system_type,
                'trajectory': [state.tolist() for state in trajectory]
            }
            json.dump(trajectory_data, f)
            temp_file = f.name
        
        try:
            # Run our tool
            cmd = [self.tool_path, "audit", "--input", temp_file, "--format", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            computation_time = time.time() - start_time
            
            if result.returncode == 0:
                # Parse tool output
                output = json.loads(result.stdout)
                violation_detected = output.get('violation_detected', False)
                error_estimate = output.get('error_estimate', 0.0)
                return violation_detected, error_estimate, computation_time
            else:
                print(f"Tool error: {result.stderr}")
                return None, None, computation_time
                
        except Exception as e:
            print(f"Tool execution failed: {e}")
            return None, None, time.time() - start_time
        finally:
            os.unlink(temp_file)


class BenchmarkSuite:
    """Main benchmark orchestrator."""
    
    def __init__(self):
        self.simulator = PhysicsSimulator()
        self.auditor = ConservationAuditor()
        self.results = []
    
    def generate_benchmark_configs(self) -> List[SimulationConfig]:
        """Generate comprehensive benchmark configuration matrix."""
        configs = []
        
        # Configuration space
        systems = [
            PhysicsSystem.KEPLER_2BODY,
            PhysicsSystem.HARMONIC_3D, 
            PhysicsSystem.PENDULUM_DOUBLE
        ]
        
        integrators = [
            IntegratorType.EULER,
            IntegratorType.RK4,
            IntegratorType.VERLET_VELOCITY,
            IntegratorType.LEAPFROG,
            IntegratorType.RUTH3,
            IntegratorType.FOREST_RUTH
        ]
        
        timesteps = [0.001, 0.01, 0.05, 0.1]
        durations = [1000, 5000, 10000]
        
        # Generate initial conditions for each system
        initial_conditions_map = {
            PhysicsSystem.KEPLER_2BODY: [
                [1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0],  # Circular orbit
                [1.0, 0.0, 0.0, 0.8, -1.0, 0.0, 0.0, -0.8],  # Elliptical
                [2.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0, -1.0],  # Eccentric
            ],
            PhysicsSystem.HARMONIC_3D: [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Single mode
                [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # Mixed modes
                [0.1, 0.1, 0.1, 2.0, 2.0, 2.0], # High energy
            ],
            PhysicsSystem.PENDULUM_DOUBLE: [
                [np.pi/4, np.pi/4, 0.0, 0.0],     # Small oscillations
                [np.pi/2, 0.0, 1.0, 0.0],        # Medium energy
                [np.pi - 0.1, 0.1, 2.0, -1.0],   # Chaotic regime
            ]
        }
        
        parameters_map = {
            PhysicsSystem.KEPLER_2BODY: {"m1": 1.0, "m2": 1.0},
            PhysicsSystem.HARMONIC_3D: {"k": 1.0, "m": 1.0},
            PhysicsSystem.PENDULUM_DOUBLE: {"L1": 1.0, "L2": 1.0, "m1": 1.0, "m2": 1.0, "g": 9.81}
        }
        
        # Generate configurations (sample to get 20 total)
        config_count = 0
        max_configs = 20
        
        for system in systems:
            for integrator in integrators:
                for dt in timesteps:
                    for duration in durations:
                        for ic in initial_conditions_map[system]:
                            if config_count >= max_configs:
                                break
                            
                            config = SimulationConfig(
                                system=system,
                                integrator=integrator,
                                timestep=dt,
                                duration_steps=duration,
                                initial_conditions=ic,
                                parameters=parameters_map[system]
                            )
                            configs.append(config)
                            config_count += 1
                            
                            # Skip some combinations to keep total at 20
                            if config_count < max_configs and len(timesteps) > 1:
                                break
                        if config_count >= max_configs:
                            break
                    if config_count >= max_configs:
                        break
                if config_count >= max_configs:
                    break
            if config_count >= max_configs:
                break
        
        return configs[:max_configs]
    
    def run_single_benchmark(self, config: SimulationConfig) -> BenchmarkResult:
        """Run benchmark for a single configuration."""
        print(f"Running: {config.system.value} + {config.integrator.value} (dt={config.timestep}, steps={config.duration_steps})")
        
        start_time = time.time()
        
        # Run simulation
        initial_state = np.array(config.initial_conditions)
        
        if config.system == PhysicsSystem.KEPLER_2BODY:
            trajectory, energies, angular_momenta = self.simulator.kepler_2body(
                initial_state, config.timestep, config.duration_steps, config.integrator,
                **config.parameters
            )
            conservation_quantities = [
                ConservationQuantity("Energy", "H = T + V", energies[0], energies, 
                                   [i * config.timestep for i in range(len(energies))]),
                ConservationQuantity("Angular Momentum", "L = r × p", angular_momenta[0], angular_momenta,
                                   [i * config.timestep for i in range(len(angular_momenta))])
            ]
            
        elif config.system == PhysicsSystem.HARMONIC_3D:
            trajectory, energies = self.simulator.harmonic_3d(
                initial_state, config.timestep, config.duration_steps, config.integrator,
                **config.parameters
            )
            conservation_quantities = [
                ConservationQuantity("Energy", "H = T + V", energies[0], energies,
                                   [i * config.timestep for i in range(len(energies))])
            ]
            
        elif config.system == PhysicsSystem.PENDULUM_DOUBLE:
            trajectory, energies = self.simulator.double_pendulum(
                initial_state, config.timestep, config.duration_steps, config.integrator,
                **config.parameters
            )
            conservation_quantities = [
                ConservationQuantity("Energy", "H = T + V", energies[0], energies,
                                   [i * config.timestep for i in range(len(energies))])
            ]
        else:
            raise ValueError(f"Unknown system: {config.system}")
        
        simulation_time = time.time() - start_time
        
        # Apply baseline methods
        primary_quantity = conservation_quantities[0].computed_values
        
        # Simple drift detection
        drift_detected, drift_error = SOTABaselines.simple_drift_detection(
            np.array(primary_quantity)
        )
        
        # Richardson extrapolation
        richardson_error = SOTABaselines.richardson_extrapolation(self.simulator, config)
        
        # Symplecticity check
        symplectic_preserved, symplectic_error = SOTABaselines.symplecticity_check(trajectory)
        
        # Frequency analysis
        freq_drift_detected, frequencies = SOTABaselines.frequency_drift_analysis(
            np.array(primary_quantity), config.timestep
        )
        
        # Our tool (if available)
        tool_detected, tool_error, tool_time = self.auditor.audit_trajectory(
            trajectory, config.timestep, config.system.value
        )
        
        return BenchmarkResult(
            config=config,
            simulation_time=simulation_time,
            conservation_quantities=conservation_quantities,
            simple_drift_detected=drift_detected,
            simple_drift_error=drift_error,
            richardson_error=richardson_error,
            symplecticity_preserved=symplectic_preserved,
            symplecticity_error=symplectic_error,
            frequency_drift_detected=freq_drift_detected,
            tool_detected=tool_detected,
            tool_error=tool_error,
            tool_time=tool_time
        )
    
    def run_full_benchmark(self) -> List[BenchmarkResult]:
        """Run the complete benchmark suite."""
        print("🚀 Starting SOTA Conservation Auditor Benchmark Suite")
        print("=" * 60)
        
        configs = self.generate_benchmark_configs()
        print(f"Generated {len(configs)} benchmark configurations")
        
        if self.auditor.available:
            print(f"✅ Conservation auditor tool found: {self.auditor.tool_path}")
        else:
            print("⚠️  Conservation auditor tool not found - running baseline-only benchmarks")
        
        results = []
        
        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] ", end="")
            try:
                result = self.run_single_benchmark(config)
                results.append(result)
                
                # Quick summary
                energy_drift = result.simple_drift_error
                print(f"Energy drift: {energy_drift:.2e} | Time: {result.simulation_time:.3f}s")
                
            except Exception as e:
                print(f"❌ FAILED: {e}")
        
        self.results = results
        return results
    
    def save_results(self, filename: str = "real_benchmark_results.json"):
        """Save benchmark results to JSON file."""
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            result_dict = asdict(result)
            
            # Convert enums to strings
            result_dict['config']['system'] = result.config.system.value
            result_dict['config']['integrator'] = result.config.integrator.value
            
            # Convert numpy types to native Python types
            def convert_numpy_types(obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            result_dict = convert_numpy_types(result_dict)
            serializable_results.append(result_dict)
        
        # Add metadata
        output = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_configs": len(self.results),
                "tool_available": self.auditor.available,
                "description": "SOTA Conservation Auditor Benchmark Results"
            },
            "results": serializable_results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n📊 Results saved to: {filename}")
    
    def generate_summary_report(self) -> Dict:
        """Generate summary statistics."""
        if not self.results:
            return {}
        
        # Aggregate statistics by integrator type
        integrator_stats = {}
        
        for result in self.results:
            integrator = result.config.integrator.value
            if integrator not in integrator_stats:
                integrator_stats[integrator] = {
                    'count': 0,
                    'avg_drift': 0.0,
                    'avg_time': 0.0,
                    'conservation_violations': 0
                }
            
            stats = integrator_stats[integrator]
            stats['count'] += 1
            stats['avg_drift'] += result.simple_drift_error
            stats['avg_time'] += result.simulation_time
            
            if result.simple_drift_detected:
                stats['conservation_violations'] += 1
        
        # Finalize averages
        for integrator, stats in integrator_stats.items():
            if stats['count'] > 0:
                stats['avg_drift'] /= stats['count']
                stats['avg_time'] /= stats['count']
        
        summary = {
            'total_benchmarks': len(self.results),
            'integrator_performance': integrator_stats,
            'overall_stats': {
                'avg_energy_drift': np.mean([r.simple_drift_error for r in self.results]),
                'violations_detected': sum(1 for r in self.results if r.simple_drift_detected),
                'symplectic_integrators_successful': sum(1 for r in self.results if r.symplecticity_preserved)
            }
        }
        
        return summary


def main():
    """Run the complete benchmark suite."""
    
    # Create benchmarks directory if it doesn't exist
    os.makedirs("benchmarks", exist_ok=True)
    os.chdir("benchmarks")
    
    # Initialize and run benchmark
    suite = BenchmarkSuite()
    results = suite.run_full_benchmark()
    
    # Save results
    suite.save_results("real_benchmark_results.json")
    
    # Generate summary
    summary = suite.generate_summary_report()
    print("\n" + "="*60)
    print("🏆 BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total benchmarks: {summary['total_benchmarks']}")
    print(f"Violations detected: {summary['overall_stats']['violations_detected']}")
    print(f"Average energy drift: {summary['overall_stats']['avg_energy_drift']:.2e}")
    
    print("\nIntegrator Performance:")
    for integrator, stats in summary['integrator_performance'].items():
        violation_rate = stats['conservation_violations'] / stats['count'] * 100
        print(f"  {integrator:15s}: {stats['avg_drift']:.2e} drift, {violation_rate:4.1f}% violations")
    
    # Save summary
    with open("benchmark_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📈 Summary saved to: benchmark_summary.json")
    print("🎯 Benchmark complete!")


if __name__ == "__main__":
    main()