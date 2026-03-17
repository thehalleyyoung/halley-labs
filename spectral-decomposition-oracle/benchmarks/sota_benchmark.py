#!/usr/bin/env python3
"""
SOTA Benchmark for Spectral Decomposition Oracle.

Comprehensive benchmarking suite with real-world optimization problems,
comparing our spectral oracle against state-of-the-art baselines.

Tests 20 real optimization instances:
- 5 small (n=10-50): rapid development & debugging  
- 10 medium (n=100-500): realistic mid-scale problems
- 5 large (n=1000-5000): scalability evaluation

Matrix types: sparse random, graph Laplacians, covariance, Toeplitz, tridiagonal
Baselines: scipy.linalg.eigh (LAPACK), numpy SVD, scipy ARPACK, power iteration
Metrics: accuracy, condition number estimation, timing, memory

Usage:
    python3 benchmarks/sota_benchmark.py --output benchmarks/real_benchmark_results.json
"""

import argparse
import json
import math
import os
import psutil
import random
import statistics
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import scipy.sparse.linalg as spla
from scipy.sparse import csc_matrix, diags
import matplotlib.pyplot as plt

# =============================================================================
# Data Structures  
# =============================================================================

@dataclass
class ProblemInstance:
    """Real-world optimization problem instance."""
    name: str
    size: int
    matrix_type: str
    description: str
    matrix: np.ndarray
    ground_truth_eigenvals: Optional[np.ndarray] = None
    ground_truth_eigenvecs: Optional[np.ndarray] = None
    condition_number: Optional[float] = None

@dataclass
class BenchmarkResult:
    """Single method benchmark result."""
    method_name: str
    instance_name: str
    success: bool
    runtime_sec: float
    memory_mb: float
    eigenval_error: Optional[float] = None
    eigenvec_error: Optional[float] = None
    condition_estimate: Optional[float] = None
    condition_error: Optional[float] = None
    residual_norm: Optional[float] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass  
class SotaBenchmarkSuite:
    """Complete SOTA benchmark results."""
    instances: List[ProblemInstance] = field(default_factory=list)
    results: List[BenchmarkResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Problem Instance Generators
# =============================================================================

def generate_sparse_random_matrix(n: int, density: float = 0.1, seed: int = 42) -> np.ndarray:
    """Generate symmetric sparse random matrix."""
    np.random.seed(seed)
    A = sp.random(n, n, density=density, format='csr', dtype=np.float64)
    A = (A + A.T) / 2  # Make symmetric
    return A.toarray()

def generate_graph_laplacian(n: int, connectivity: float = 0.15, seed: int = 42) -> np.ndarray:
    """Generate Laplacian matrix of random graph."""
    np.random.seed(seed)
    # Generate adjacency matrix for random graph
    A = np.random.rand(n, n) < connectivity
    A = A.astype(float)
    A = (A + A.T) / 2  # Make symmetric  
    np.fill_diagonal(A, 0)  # No self-loops
    
    # Build Laplacian: L = D - A
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    return L

def generate_covariance_matrix(n: int, rank_ratio: float = 0.8, seed: int = 42) -> np.ndarray:
    """Generate covariance matrix from random data."""
    np.random.seed(seed)
    rank = max(1, int(n * rank_ratio))
    samples = max(n, int(1.5 * n))  # Ensure positive definite
    
    # Generate random data matrix  
    X = np.random.randn(samples, rank)
    # Add noise to avoid rank deficiency
    X += 0.01 * np.random.randn(samples, rank)
    
    # Embed in higher dimension if needed
    if rank < n:
        embedding = np.random.randn(rank, n) 
        X = X @ embedding
    else:
        X = X[:, :n]
    
    # Compute covariance matrix
    C = (X.T @ X) / (samples - 1)
    return C

def generate_toeplitz_matrix(n: int, decay: float = 0.8, seed: int = 42) -> np.ndarray:
    """Generate symmetric Toeplitz matrix."""
    np.random.seed(seed)
    # Generate first row with exponential decay
    first_row = np.array([decay**i for i in range(n)])
    # Add random perturbations
    first_row += 0.1 * np.random.randn(n)
    first_row[0] = abs(first_row[0])  # Ensure positive diagonal
    
    # Build Toeplitz matrix
    from scipy.linalg import toeplitz
    T = toeplitz(first_row)
    return T

def generate_tridiagonal_matrix(n: int, seed: int = 42) -> np.ndarray:
    """Generate symmetric tridiagonal matrix.""" 
    np.random.seed(seed)
    
    # Main diagonal (positive for positive definiteness)
    main = 2 + np.random.rand(n)
    # Off-diagonals
    off = np.random.randn(n-1)
    
    # Build tridiagonal matrix
    T = np.zeros((n, n))
    np.fill_diagonal(T, main)
    np.fill_diagonal(T[1:, :-1], off)
    np.fill_diagonal(T[:-1, 1:], off)  # Symmetric
    
    return T

# =============================================================================
# Problem Instance Suite
# =============================================================================

def create_problem_instances() -> List[ProblemInstance]:
    """Create 20 real-world optimization problem instances."""
    instances = []
    
    # SMALL PROBLEMS (n=10-50): rapid prototyping
    sizes_small = [10, 20, 30, 40, 50]
    
    for i, n in enumerate(sizes_small):
        if i == 0:  # Sparse random
            matrix = generate_sparse_random_matrix(n, density=0.2, seed=100+i)
            instances.append(ProblemInstance(
                name=f"sparse_random_small_{n}",
                size=n,
                matrix_type="sparse_random",
                description=f"Small sparse random matrix ({n}x{n}, 20% density)",
                matrix=matrix
            ))
        elif i == 1:  # Graph Laplacian
            matrix = generate_graph_laplacian(n, connectivity=0.3, seed=100+i)
            instances.append(ProblemInstance(
                name=f"graph_laplacian_small_{n}",
                size=n,  
                matrix_type="graph_laplacian",
                description=f"Small graph Laplacian ({n}x{n}, 30% connectivity)",
                matrix=matrix
            ))
        elif i == 2:  # Covariance
            matrix = generate_covariance_matrix(n, rank_ratio=0.7, seed=100+i)
            instances.append(ProblemInstance(
                name=f"covariance_small_{n}",
                size=n,
                matrix_type="covariance", 
                description=f"Small covariance matrix ({n}x{n}, 70% rank ratio)",
                matrix=matrix
            ))
        elif i == 3:  # Toeplitz
            matrix = generate_toeplitz_matrix(n, decay=0.7, seed=100+i)
            instances.append(ProblemInstance(
                name=f"toeplitz_small_{n}",
                size=n,
                matrix_type="toeplitz",
                description=f"Small Toeplitz matrix ({n}x{n}, 0.7 decay)",
                matrix=matrix
            ))
        else:  # Tridiagonal  
            matrix = generate_tridiagonal_matrix(n, seed=100+i)
            instances.append(ProblemInstance(
                name=f"tridiagonal_small_{n}",
                size=n,
                matrix_type="tridiagonal", 
                description=f"Small tridiagonal matrix ({n}x{n})",
                matrix=matrix
            ))

    # MEDIUM PROBLEMS (n=100-500): realistic scale
    sizes_medium = [100, 150, 200, 250, 300, 350, 400, 450, 500, 320]
    matrix_types = ["sparse_random", "graph_laplacian", "covariance", "toeplitz", "tridiagonal"] * 2
    
    for i, (n, mat_type) in enumerate(zip(sizes_medium, matrix_types)):
        seed = 200 + i
        if mat_type == "sparse_random":
            matrix = generate_sparse_random_matrix(n, density=0.05, seed=seed)
            desc = f"Medium sparse random matrix ({n}x{n}, 5% density)"
        elif mat_type == "graph_laplacian": 
            matrix = generate_graph_laplacian(n, connectivity=0.1, seed=seed)
            desc = f"Medium graph Laplacian ({n}x{n}, 10% connectivity)"
        elif mat_type == "covariance":
            matrix = generate_covariance_matrix(n, rank_ratio=0.6, seed=seed) 
            desc = f"Medium covariance matrix ({n}x{n}, 60% rank ratio)"
        elif mat_type == "toeplitz":
            matrix = generate_toeplitz_matrix(n, decay=0.8, seed=seed)
            desc = f"Medium Toeplitz matrix ({n}x{n}, 0.8 decay)"
        else:  # tridiagonal
            matrix = generate_tridiagonal_matrix(n, seed=seed)
            desc = f"Medium tridiagonal matrix ({n}x{n})"
            
        instances.append(ProblemInstance(
            name=f"{mat_type}_medium_{n}",
            size=n,
            matrix_type=mat_type,
            description=desc,
            matrix=matrix
        ))

    # LARGE PROBLEMS (n=1000-5000): scalability test
    sizes_large = [1000, 2000, 3000, 4000, 5000]
    
    for i, n in enumerate(sizes_large):
        seed = 300 + i
        if i == 0:  # Sparse random
            matrix = generate_sparse_random_matrix(n, density=0.01, seed=seed)
            instances.append(ProblemInstance(
                name=f"sparse_random_large_{n}",
                size=n,
                matrix_type="sparse_random",
                description=f"Large sparse random matrix ({n}x{n}, 1% density)",
                matrix=matrix
            ))
        elif i == 1:  # Graph Laplacian
            matrix = generate_graph_laplacian(n, connectivity=0.005, seed=seed) 
            instances.append(ProblemInstance(
                name=f"graph_laplacian_large_{n}",
                size=n,
                matrix_type="graph_laplacian",
                description=f"Large graph Laplacian ({n}x{n}, 0.5% connectivity)", 
                matrix=matrix
            ))
        elif i == 2:  # Covariance
            matrix = generate_covariance_matrix(n, rank_ratio=0.1, seed=seed)
            instances.append(ProblemInstance(
                name=f"covariance_large_{n}",
                size=n,
                matrix_type="covariance",
                description=f"Large covariance matrix ({n}x{n}, 10% rank ratio)",
                matrix=matrix
            ))
        elif i == 3:  # Toeplitz
            matrix = generate_toeplitz_matrix(n, decay=0.9, seed=seed)
            instances.append(ProblemInstance(
                name=f"toeplitz_large_{n}",  
                size=n,
                matrix_type="toeplitz",
                description=f"Large Toeplitz matrix ({n}x{n}, 0.9 decay)",
                matrix=matrix
            ))
        else:  # Tridiagonal
            matrix = generate_tridiagonal_matrix(n, seed=seed)
            instances.append(ProblemInstance(
                name=f"tridiagonal_large_{n}",
                size=n,
                matrix_type="tridiagonal", 
                description=f"Large tridiagonal matrix ({n}x{n})",
                matrix=matrix
            ))
    
    return instances

# =============================================================================
# Spectral Methods (Our Oracle Implementation)
# =============================================================================

def spectral_oracle_eigendecomposition(matrix: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Our spectral oracle eigendecomposition method.
    
    Implements adaptive algorithm selection based on matrix properties:
    - Uses condition-aware routing for numerical stability
    - Applies Ruiz equilibration as preconditioner for ill-conditioned dense solves
    - Back-transforms eigenvectors and applies Rayleigh quotient refinement
    - Selects appropriate solver (direct vs iterative)
    """
    n = matrix.shape[0]
    
    # Estimate condition number (fast approximation)
    if n <= 100:
        try:
            eigvals_est, _ = la.eigh(matrix)
            condition_est = abs(eigvals_est.max()) / max(abs(eigvals_est.min()), 1e-15)
        except:
            condition_est = 1e12
    else:
        condition_est = estimate_condition_number_fast(matrix)
    
    # Algorithm selection based on size and structure
    if n <= 500:
        # Dense path: equilibrate if ill-conditioned, refine afterward
        if condition_est > 1e8:
            matrix_processed, D_left, D_right = ruiz_equilibration(matrix, return_scaling=True)
            eigvals, eigvecs = la.eigh(matrix_processed)
            # Back-transform and refine against original matrix
            eigvecs = _back_transform_and_refine(matrix, eigvecs, eigvals, D_right)
            eigvals = np.array([
                float(v @ matrix @ v) / max(float(v @ v), 1e-30)
                for v in eigvecs.T
            ])
        else:
            eigvals, eigvecs = la.eigh(matrix)
    else:
        # Large: use ARPACK on original matrix (handles conditioning natively)
        if k is None:
            k = min(50, n // 10)
        try:
            matrix_sparse = csc_matrix(matrix) if not sp.issparse(matrix) else matrix
            eigvals, eigvecs = spla.eigsh(matrix_sparse, k=k, which='LA')
        except:
            eigvals, eigvecs = la.eigh(matrix)
    
    # Sort eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    return eigvals, eigvecs


def _back_transform_and_refine(
    A: np.ndarray, vecs: np.ndarray, vals: np.ndarray,
    D: np.ndarray, rqi_iters: int = 3
) -> np.ndarray:
    """Back-transform equilibrated eigenvectors and refine via Rayleigh quotient iteration."""
    n = A.shape[0]
    k = vecs.shape[1]
    refined = np.empty_like(vecs)
    
    # For large matrices, skip expensive RQI solves — just back-transform
    use_rqi = n <= 1000
    
    for i in range(k):
        # Back-transform: v_orig ≈ D * v_equil
        v = D * vecs[:, i]
        v = v / np.linalg.norm(v)
        
        if use_rqi:
            # Rayleigh quotient iteration: converges cubically
            for _ in range(rqi_iters):
                rq = float(v @ A @ v)
                try:
                    shift = A - rq * np.eye(n)
                    v_new = la.solve(shift, v, assume_a='sym')
                    v_new = v_new / np.linalg.norm(v_new)
                    if v_new @ v < 0:
                        v_new = -v_new
                    v = v_new
                except np.linalg.LinAlgError:
                    break
        else:
            # Cheap refinement: a few matrix-vector power-method steps
            for _ in range(5):
                rq = float(v @ A @ v)
                v_new = A @ v
                v_new = v_new / np.linalg.norm(v_new)
                if v_new @ v < 0:
                    v_new = -v_new
                v = v_new
        
        refined[:, i] = v
    
    return refined

def estimate_condition_number_fast(matrix: np.ndarray, max_iter: int = 10) -> float:
    """Fast condition number estimation via power iteration."""
    n = matrix.shape[0]
    
    # Estimate largest eigenvalue  
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(max_iter):
        v_new = matrix @ v
        lambda_max = np.dot(v, v_new)
        v = v_new / np.linalg.norm(v_new)
    
    # Estimate smallest eigenvalue (via shifted inverse)
    shift = lambda_max * 0.1
    try:
        shifted = matrix + shift * np.eye(n)
        v = np.random.randn(n) 
        v = v / np.linalg.norm(v)
        
        for _ in range(max_iter):
            v_new = la.solve(shifted, v)
            lambda_inv = np.dot(v, v_new)
            v = v_new / np.linalg.norm(v_new)
        
        lambda_min = 1.0 / lambda_inv - shift
        condition = lambda_max / max(lambda_min, 1e-15)
    except:
        condition = 1e12  # Fallback for singular matrices
        
    return condition

def ruiz_equilibration(matrix: np.ndarray, max_iter: int = 5,
                       return_scaling: bool = False):
    """
    Ruiz diagonal equilibration for better conditioning.
    
    For symmetric matrices uses a single diagonal D so that A' = D A D
    preserves symmetry. When return_scaling=True, returns (A', D, D).
    """
    A = matrix.copy()
    n = A.shape[0]
    D_accum = np.ones(n)
    
    for _ in range(max_iter):
        # For symmetric matrices, use symmetric scaling: same D for rows and cols
        row_norms = np.sqrt(np.sum(A**2, axis=1))
        row_norms = np.where(row_norms > 1e-15, row_norms, 1.0)
        
        D = 1.0 / np.sqrt(row_norms)
        D_accum *= D
        
        # Apply symmetric scaling: A' = D A D
        A = D[:, None] * A * D[None, :]
        
        # Check convergence
        if np.max(row_norms) / np.min(row_norms) < 2.0:
            break
    
    if return_scaling:
        return A, D_accum, D_accum
    return A

# =============================================================================
# SOTA Baseline Methods
# =============================================================================

def scipy_lapack_eigh(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline: scipy.linalg.eigh (LAPACK dsyev)."""
    eigvals, eigvecs = la.eigh(matrix)
    # Sort in descending order
    idx = np.argsort(eigvals)[::-1] 
    return eigvals[idx], eigvecs[:, idx]

def numpy_svd_method(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline: numpy SVD-based eigendecomposition.""" 
    U, s, Vt = np.linalg.svd(matrix)
    # For symmetric matrices, eigenvalues = singular values
    eigvals = s
    eigvecs = U
    return eigvals, eigvecs

def scipy_arpack_eigsh(matrix: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline: scipy ARPACK iterative solver."""
    n = matrix.shape[0]
    if k is None:
        k = min(50, n//2 - 1)
    
    # Convert to sparse if needed
    if not sp.issparse(matrix):
        matrix_sparse = csc_matrix(matrix)
    else:
        matrix_sparse = matrix
        
    eigvals, eigvecs = spla.eigsh(matrix_sparse, k=k, which='LA') 
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx], eigvecs[:, idx]

def power_iteration_method(matrix: np.ndarray, k: int = 10, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline: Power iteration for top-k eigenvalues."""
    n = matrix.shape[0]
    k = min(k, n)
    
    eigvals = []
    eigvecs = []
    A = matrix.copy()
    
    for i in range(k):
        # Power iteration for dominant eigenvalue
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        for _ in range(max_iter):
            v_new = A @ v
            eigenval = np.dot(v, v_new)
            v_new_norm = np.linalg.norm(v_new)
            
            if v_new_norm < 1e-10:
                break
                
            v = v_new / v_new_norm
            
        eigvals.append(eigenval)
        eigvecs.append(v.copy())
        
        # Deflation: A = A - λvvᵀ
        A = A - eigenval * np.outer(v, v)
        
    return np.array(eigvals), np.column_stack(eigvecs)

# =============================================================================
# Evaluation Metrics  
# =============================================================================

def compute_eigenvalue_error(computed_vals: np.ndarray, reference_vals: np.ndarray) -> float:
    """Compute relative eigenvalue error."""
    k = min(len(computed_vals), len(reference_vals))
    if k == 0:
        return float('inf')
    
    computed_k = computed_vals[:k]
    reference_k = reference_vals[:k]
    
    # Relative error with regularization
    denom = np.maximum(np.abs(reference_k), 1e-10)
    rel_errors = np.abs(computed_k - reference_k) / denom
    
    return np.mean(rel_errors)

def compute_eigenvector_error(computed_vecs: np.ndarray, reference_vecs: np.ndarray) -> float:
    """Compute eigenvector error (subspace angle).""" 
    k = min(computed_vecs.shape[1], reference_vecs.shape[1])
    if k == 0:
        return float('inf')
        
    V1 = computed_vecs[:, :k]
    V2 = reference_vecs[:, :k]
    
    # Handle sign ambiguity 
    for i in range(k):
        if np.dot(V1[:, i], V2[:, i]) < 0:
            V1[:, i] *= -1
            
    # Subspace angle via SVD
    try:
        _, s, _ = np.linalg.svd(V1.T @ V2)
        s = np.clip(s, 0, 1)  # Numerical stability
        angles = np.arccos(s)
        return np.mean(angles)
    except:
        return float('inf')

def compute_residual_norm(matrix: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray) -> float:
    """Compute ||Av - λv|| residual norm."""
    if len(eigvals) == 0 or eigvecs.shape[1] == 0:
        return float('inf')
        
    k = min(len(eigvals), eigvecs.shape[1], 5)  # Check top 5 pairs
    residuals = []
    
    for i in range(k):
        v = eigvecs[:, i]
        lam = eigvals[i]
        residual = matrix @ v - lam * v
        residuals.append(np.linalg.norm(residual))
        
    return np.mean(residuals)

def estimate_memory_usage() -> float:
    """Estimate current memory usage in MB."""
    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except:
        return 0.0

# =============================================================================
# Benchmarking Engine
# =============================================================================

def benchmark_method(method_func, method_name: str, instance: ProblemInstance, 
                    timeout_sec: float = 300.0) -> BenchmarkResult:
    """Benchmark a single method on a problem instance."""
    
    print(f"  Running {method_name} on {instance.name}...")
    
    # Memory tracking
    tracemalloc.start()
    mem_start = estimate_memory_usage()
    
    result = BenchmarkResult(
        method_name=method_name,
        instance_name=instance.name, 
        success=False,
        runtime_sec=0.0,
        memory_mb=0.0
    )
    
    try:
        # Time the method
        start_time = time.perf_counter()
        
        # Call the method (with timeout handling)
        if method_name == "power_iteration":
            k = min(10, instance.size//10)
            eigvals, eigvecs = method_func(instance.matrix, k=k)
        elif method_name == "scipy_arpack" and instance.size > 100:
            k = min(20, instance.size//10) 
            eigvals, eigvecs = method_func(instance.matrix, k=k)
        else:
            eigvals, eigvecs = method_func(instance.matrix)
            
        end_time = time.perf_counter()
        runtime = end_time - start_time
        
        # Memory usage
        mem_end = estimate_memory_usage()
        memory_used = max(0, mem_end - mem_start)
        
        result.success = True
        result.runtime_sec = runtime
        result.memory_mb = memory_used
        
        # Compute ground truth if needed
        if instance.ground_truth_eigenvals is None:
            # Use scipy LAPACK as reference (most reliable)
            try:
                ref_vals, ref_vecs = scipy_lapack_eigh(instance.matrix)
                instance.ground_truth_eigenvals = ref_vals
                instance.ground_truth_eigenvecs = ref_vecs
                instance.condition_number = ref_vals.max() / max(ref_vals.min(), 1e-15)
            except:
                pass
                
        # Compute accuracy metrics
        if instance.ground_truth_eigenvals is not None:
            result.eigenval_error = compute_eigenvalue_error(
                eigvals, instance.ground_truth_eigenvals
            )
            result.eigenvec_error = compute_eigenvector_error(
                eigvecs, instance.ground_truth_eigenvecs  
            )
            
        # Compute residual
        result.residual_norm = compute_residual_norm(
            instance.matrix, eigvals, eigvecs
        )
        
        # Condition number estimation 
        if len(eigvals) > 1:
            result.condition_estimate = eigvals.max() / max(eigvals.min(), 1e-15)
            if instance.condition_number is not None:
                result.condition_error = abs(
                    result.condition_estimate - instance.condition_number
                ) / instance.condition_number
                
        # Extra metrics
        result.extra_metrics = {
            "num_eigenvals": len(eigvals),
            "spectral_gap": eigvals[0] - eigvals[1] if len(eigvals) > 1 else 0.0,
            "trace": np.sum(eigvals) if len(eigvals) == instance.size else np.trace(instance.matrix)
        }
        
    except Exception as e:
        print(f"    ERROR in {method_name}: {str(e)}")
        result.success = False
        result.extra_metrics = {"error": str(e)}
        
    finally:
        tracemalloc.stop()
        
    return result

def run_sota_benchmark(output_path: str = "benchmarks/real_benchmark_results.json") -> SotaBenchmarkSuite:
    """Run the complete SOTA benchmark suite."""
    
    print("=" * 70)
    print("SPECTRAL DECOMPOSITION ORACLE - SOTA BENCHMARK SUITE")  
    print("=" * 70)
    print()
    
    # Create problem instances
    print("Creating 20 real-world problem instances...")
    instances = create_problem_instances()
    print(f"Created {len(instances)} instances")
    print()
    
    # Define methods to benchmark
    methods = [
        (spectral_oracle_eigendecomposition, "spectral_oracle"),
        (scipy_lapack_eigh, "scipy_lapack"), 
        (numpy_svd_method, "numpy_svd"),
        (scipy_arpack_eigsh, "scipy_arpack"),
        (power_iteration_method, "power_iteration")
    ]
    
    # Run benchmarks
    results = []
    total_runs = len(instances) * len(methods)
    current_run = 0
    
    for instance in instances:
        print(f"Benchmarking instance: {instance.name} ({instance.size}x{instance.size})")
        
        for method_func, method_name in methods:
            current_run += 1
            print(f"  [{current_run}/{total_runs}] {method_name}")
            
            # Skip certain method combinations for efficiency
            skip = False
            if instance.size > 2000 and method_name in ["numpy_svd", "scipy_lapack"]:
                print(f"    SKIPPED (too large for dense method)")
                skip = True
            elif instance.size > 1000 and method_name == "power_iteration":
                print(f"    SKIPPED (power iteration too slow)")  
                skip = True
                
            if skip:
                result = BenchmarkResult(
                    method_name=method_name,
                    instance_name=instance.name,
                    success=False,
                    runtime_sec=0.0,
                    memory_mb=0.0,
                    extra_metrics={"skipped": True}
                )
            else:
                result = benchmark_method(method_func, method_name, instance)
                
            results.append(result)
            
        print()
    
    # Compute summary statistics
    print("Computing benchmark summary...")
    summary = compute_benchmark_summary(instances, results)
    
    # Create benchmark suite
    suite = SotaBenchmarkSuite(
        instances=instances,
        results=results, 
        summary=summary
    )
    
    # Save results
    print(f"Saving results to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to serializable format
    suite_dict = {
        "instances": [
            {
                "name": inst.name,
                "size": inst.size, 
                "matrix_type": inst.matrix_type,
                "description": inst.description,
                "condition_number": inst.condition_number
            } for inst in suite.instances
        ],
        "results": [asdict(result) for result in suite.results],
        "summary": suite.summary
    }
    
    with open(output_path, 'w') as f:
        json.dump(suite_dict, f, indent=2, default=str)
        
    print("Benchmark complete!")
    return suite

def compute_benchmark_summary(instances: List[ProblemInstance], 
                            results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Compute summary statistics from benchmark results."""
    
    summary = {
        "total_instances": len(instances),
        "total_methods": len(set(r.method_name for r in results)),
        "methods": {},
        "size_categories": {"small": [], "medium": [], "large": []},
        "matrix_types": {}
    }
    
    # Group results by method
    by_method = {}
    for result in results:
        if result.method_name not in by_method:
            by_method[result.method_name] = []
        by_method[result.method_name].append(result)
    
    # Compute method-wise statistics
    for method_name, method_results in by_method.items():
        successful = [r for r in method_results if r.success]
        
        if successful:
            runtimes = [r.runtime_sec for r in successful]
            memories = [r.memory_mb for r in successful]
            eigenval_errors = [r.eigenval_error for r in successful if r.eigenval_error is not None]
            residuals = [r.residual_norm for r in successful if r.residual_norm is not None]
            
            summary["methods"][method_name] = {
                "success_rate": len(successful) / len(method_results),
                "avg_runtime_sec": statistics.mean(runtimes) if runtimes else 0,
                "std_runtime_sec": statistics.stdev(runtimes) if len(runtimes) > 1 else 0,
                "avg_memory_mb": statistics.mean(memories) if memories else 0,
                "avg_eigenval_error": statistics.mean(eigenval_errors) if eigenval_errors else None,
                "avg_residual_norm": statistics.mean(residuals) if residuals else None,
                "total_runs": len(method_results),
                "successful_runs": len(successful)
            }
        else:
            summary["methods"][method_name] = {
                "success_rate": 0.0,
                "total_runs": len(method_results),
                "successful_runs": 0
            }
    
    # Size category analysis
    for instance in instances:
        if instance.size <= 50:
            category = "small"
        elif instance.size <= 500:
            category = "medium" 
        else:
            category = "large"
            
        summary["size_categories"][category].append({
            "name": instance.name,
            "size": instance.size,
            "condition_number": instance.condition_number
        })
    
    # Matrix type analysis  
    for instance in instances:
        if instance.matrix_type not in summary["matrix_types"]:
            summary["matrix_types"][instance.matrix_type] = []
        summary["matrix_types"][instance.matrix_type].append(instance.name)
        
    return summary

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SOTA Benchmark for Spectral Oracle")
    parser.add_argument("--output", default="benchmarks/real_benchmark_results.json",
                       help="Output JSON file path")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Run benchmark
    suite = run_sota_benchmark(args.output)
    
    # Print summary 
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal instances: {suite.summary['total_instances']}")
    print(f"Total methods: {suite.summary['total_methods']}")
    
    print("\nMethod Performance:")
    for method_name, stats in suite.summary["methods"].items():
        print(f"  {method_name}:")
        print(f"    Success rate: {stats['success_rate']:.1%}")
        if stats.get('avg_runtime_sec'):
            print(f"    Avg runtime: {stats['avg_runtime_sec']:.3f}s")
        if stats.get('avg_eigenval_error'):
            print(f"    Avg eigenval error: {stats['avg_eigenval_error']:.2e}")
    
    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()