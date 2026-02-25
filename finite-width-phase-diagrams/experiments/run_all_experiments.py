#!/usr/bin/env python3
"""
Comprehensive experiments for finite-width phase diagrams.
Generates all data for the paper: MLP, CNN-like, ResNet-like architectures.
All results are saved as labeled JSON files.
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation'))

import numpy as np
from src.kernel_engine.ntk import AnalyticNTK, EmpiricalNTK
from src.corrections.finite_width import FiniteWidthCorrector
from src.ode_solver.bifurcation import BifurcationDetector
from src.phase_mapper.gamma_star import PhaseBoundaryPredictor
from src.corrections.trace_normalized import TraceNormalizedCorrector

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def make_mlp(input_dim, width, depth, output_dim=1, seed=None):
    """Create MLP with given architecture. Returns (params, forward_fn)."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    dims = [input_dim] + [width] * depth + [output_dim]
    params = []
    for i in range(len(dims) - 1):
        W = rng.randn(dims[i] * dims[i + 1]) / np.sqrt(dims[i])
        params.append(W)
    flat_params = np.concatenate(params)
    shapes = [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]

    def forward(p, x):
        idx = 0
        h = x
        for li, (d_in, d_out) in enumerate(shapes):
            W = p[idx:idx + d_in * d_out].reshape(d_in, d_out)
            idx += d_in * d_out
            h = h @ W
            if li < len(shapes) - 1:
                h = np.maximum(h, 0)
        return h

    return flat_params, forward


def make_resnet_block(input_dim, width, depth, output_dim=1, seed=None):
    """MLP with skip connections (residual). Returns (params, forward_fn)."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    # First layer projects to width, then residual blocks, then output
    params_list = []
    # Input projection
    W_in = rng.randn(input_dim * width) / np.sqrt(input_dim)
    params_list.append(W_in)
    # Residual blocks (each is width -> width)
    for d in range(depth):
        W = rng.randn(width * width) / np.sqrt(width)
        params_list.append(W)
    # Output projection
    W_out = rng.randn(width * output_dim) / np.sqrt(width)
    params_list.append(W_out)
    flat_params = np.concatenate(params_list)

    def forward(p, x):
        idx = 0
        # Input projection
        W_in = p[idx:idx + input_dim * width].reshape(input_dim, width)
        idx += input_dim * width
        h = x @ W_in
        h = np.maximum(h, 0)
        # Residual blocks
        for d in range(depth):
            W = p[idx:idx + width * width].reshape(width, width)
            idx += width * width
            residual = h @ W
            residual = np.maximum(residual, 0)
            h = h + residual / np.sqrt(width)  # scaled skip
        # Output
        W_out = p[idx:idx + width * output_dim].reshape(width, output_dim)
        h = h @ W_out
        return h

    return flat_params, forward


def compute_empirical_ntk(forward_fn, params, X, eps=1e-5):
    """Compute NTK via finite differences on the forward function."""
    entk = EmpiricalNTK(output_dim=1)
    return entk.compute_ntk(forward_fn, params, X, eps=eps)


def train_and_measure_drift(forward_fn, params, X, y, lr, n_steps, measure_interval=100):
    """Train network and measure NTK drift over time."""
    entk = EmpiricalNTK(output_dim=1)
    K0 = entk.compute_ntk(forward_fn, params, X)
    K0_norm = np.linalg.norm(K0, 'fro')
    
    p = params.copy()
    eps = 1e-5
    n = len(X)
    
    drifts = []
    losses = []
    
    for step in range(n_steps):
        # Forward pass
        pred = forward_fn(p, X)
        loss = 0.5 * np.mean((pred.flatten() - y) ** 2)
        
        # Gradient via finite differences
        grad = np.zeros_like(p)
        for i in range(len(p)):
            p[i] += eps
            loss_plus = 0.5 * np.mean((forward_fn(p, X).flatten() - y) ** 2)
            p[i] -= 2 * eps
            loss_minus = 0.5 * np.mean((forward_fn(p, X).flatten() - y) ** 2)
            p[i] += eps
            grad[i] = (loss_plus - loss_minus) / (2 * eps)
        
        p -= lr * grad
        
        if step % measure_interval == 0 or step == n_steps - 1:
            Kt = entk.compute_ntk(forward_fn, p, X)
            drift = np.linalg.norm(Kt - K0, 'fro') / K0_norm
            drifts.append({'step': step, 'drift': float(drift), 'loss': float(loss)})
            
    return drifts, p


def experiment_1_ntk_convergence():
    """Experiment 1: NTK convergence with width for MLPs.
    
    Verifies that empirical NTK converges to analytic NTK as width grows,
    and measures the rate of convergence (should be O(1/N)).
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: NTK Convergence with Width")
    print("=" * 60)
    
    input_dim = 8
    n_samples = 10
    depths = [2, 3, 4]
    widths = [16, 32, 64, 128, 256, 512]
    n_seeds = 5
    
    X = np.random.RandomState(42).randn(n_samples, input_dim)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)  # normalize
    
    results = {}
    for depth in depths:
        print(f"\n  Depth {depth}:")
        antk = AnalyticNTK(depth=depth + 1, activation='relu')
        K_analytic = antk.compute_ntk(X)
        K_analytic_norm = np.linalg.norm(K_analytic, 'fro')
        
        width_results = []
        for width in widths:
            seed_errors = []
            seed_traces = []
            for seed in range(n_seeds):
                params, fwd = make_mlp(input_dim, width, depth, seed=seed + 100)
                K_emp = compute_empirical_ntk(fwd, params, X)
                
                # Normalize both for comparison
                K_emp_normalized = K_emp / np.trace(K_emp) * np.trace(K_analytic)
                error = np.linalg.norm(K_emp_normalized - K_analytic, 'fro') / K_analytic_norm
                seed_errors.append(float(error))
                seed_traces.append(float(np.trace(K_emp)))
            
            mean_error = np.mean(seed_errors)
            std_error = np.std(seed_errors)
            width_results.append({
                'width': width,
                'mean_relative_error': float(mean_error),
                'std_relative_error': float(std_error),
                'mean_trace': float(np.mean(seed_traces)),
                'n_seeds': n_seeds
            })
            print(f"    Width {width:4d}: error = {mean_error:.4f} ± {std_error:.4f}")
        
        results[f'depth_{depth}'] = {
            'depth': depth,
            'widths': widths,
            'results': width_results,
            'analytic_ntk_trace': float(np.trace(K_analytic)),
            'analytic_ntk_frobenius': float(K_analytic_norm)
        }
    
    # Fit convergence rate
    for depth_key, data in results.items():
        ws = np.array([r['width'] for r in data['results']])
        errs = np.array([r['mean_relative_error'] for r in data['results']])
        # Fit log(error) = a + b * log(width)
        log_w = np.log(ws)
        log_e = np.log(errs + 1e-10)
        b, a = np.polyfit(log_w, log_e, 1)
        data['convergence_rate'] = float(b)
        data['convergence_intercept'] = float(np.exp(a))
        print(f"\n  {depth_key}: convergence rate = {b:.3f} (expected: -0.5 for O(1/sqrt(N)))")
    
    results['metadata'] = {
        'experiment': 'ntk_convergence_with_width',
        'input_dim': input_dim,
        'n_samples': n_samples,
        'n_seeds': n_seeds,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(DATA_DIR, 'exp1_ntk_convergence.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\n  Saved: exp1_ntk_convergence.json")
    return results


def experiment_2_correction_fitting():
    """Experiment 2: Finite-width correction fitting quality.
    
    Fits 1/N corrections at multiple widths and measures R^2, 
    prediction accuracy at held-out widths.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Finite-Width Correction Fitting")
    print("=" * 60)
    
    input_dim = 8
    n_samples = 8
    depths = [2, 3, 4]
    train_widths = [32, 64, 128, 256]
    test_widths = [48, 96, 192, 384]
    n_seeds = 3
    
    X = np.random.RandomState(42).randn(n_samples, input_dim)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    results = {}
    for depth in depths:
        print(f"\n  Depth {depth}:")
        seed_results = []
        for seed in range(n_seeds):
            # Compute NTKs at training widths
            train_ntks = []
            for w in train_widths:
                params, fwd = make_mlp(input_dim, w, depth, seed=seed * 1000 + w)
                K = compute_empirical_ntk(fwd, params, X)
                train_ntks.append(K)
            
            ntk_measurements = np.array(train_ntks)
            corrector = FiniteWidthCorrector(output_dim=1, min_widths=3)
            result = corrector.compute_corrections_regression(ntk_measurements, train_widths)
            
            r_squared = result.convergence_info.r_squared
            theta_0_norm = float(np.linalg.norm(result.theta_0))
            theta_1_norm = float(np.linalg.norm(result.theta_1))
            
            # Test predictions at held-out widths
            test_errors = []
            for w in test_widths:
                params, fwd = make_mlp(input_dim, w, depth, seed=seed * 1000 + w + 500)
                K_true = compute_empirical_ntk(fwd, params, X)
                K_pred = result.theta_0 + result.theta_1 / w
                pred_error = np.linalg.norm(K_pred - K_true, 'fro') / np.linalg.norm(K_true, 'fro')
                test_errors.append({'width': w, 'relative_error': float(pred_error)})
            
            seed_results.append({
                'seed': seed,
                'r_squared': float(r_squared),
                'theta_0_norm': theta_0_norm,
                'theta_1_norm': theta_1_norm,
                'correction_ratio': theta_1_norm / (theta_0_norm + 1e-10),
                'test_errors': test_errors
            })
            print(f"    Seed {seed}: R² = {r_squared:.4f}, "
                  f"|θ₀| = {theta_0_norm:.2f}, |θ₁| = {theta_1_norm:.2f}")
        
        mean_r2 = np.mean([s['r_squared'] for s in seed_results])
        mean_test_err = np.mean([e['relative_error'] for s in seed_results for e in s['test_errors']])
        results[f'depth_{depth}'] = {
            'depth': depth,
            'train_widths': train_widths,
            'test_widths': test_widths,
            'seed_results': seed_results,
            'mean_r_squared': float(mean_r2),
            'mean_test_error': float(mean_test_err)
        }
        print(f"    Mean R² = {mean_r2:.4f}, Mean test error = {mean_test_err:.4f}")
    
    results['metadata'] = {
        'experiment': 'correction_fitting_quality',
        'input_dim': input_dim,
        'n_samples': n_samples,
        'n_seeds': n_seeds,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(DATA_DIR, 'exp2_correction_fitting.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\n  Saved: exp2_correction_fitting.json")
    return results


def experiment_3_phase_boundary():
    """Experiment 3: Phase boundary detection via training dynamics.
    
    For each (lr, init_scale) pair, train network and measure NTK drift.
    Classify as lazy (drift < 0.05) or rich (drift > 0.2).
    Compare infinite-width prediction vs finite-width corrected prediction.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Phase Boundary Detection")
    print("=" * 60)
    
    input_dim = 8
    n_samples = 6
    width = 64
    depth = 2
    n_steps = 200
    
    X = np.random.RandomState(42).randn(n_samples, input_dim)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    y = np.random.RandomState(42).randn(n_samples)
    
    # Grid over learning rate and initialization scale
    lr_range = np.logspace(-3, 0, 12)
    init_scales = np.linspace(0.3, 2.5, 10)
    
    print(f"  Grid: {len(lr_range)} x {len(init_scales)} = {len(lr_range) * len(init_scales)} points")
    
    grid_results = []
    for i, lr in enumerate(lr_range):
        for j, sigma in enumerate(init_scales):
            # Create network with scaled initialization
            params, fwd = make_mlp(input_dim, width, depth, seed=42)
            params = params * sigma
            
            # Compute initial NTK
            K0 = compute_empirical_ntk(fwd, params, X)
            K0_norm = np.linalg.norm(K0, 'fro')
            
            # Quick training to measure drift
            p = params.copy()
            eps_fd = 1e-5
            for step in range(n_steps):
                pred = fwd(p, X)
                residual = pred.flatten() - y
                loss = 0.5 * np.mean(residual ** 2)
                
                # Simple numerical gradient (fast for small nets)
                grad = np.zeros_like(p)
                for k in range(len(p)):
                    p[k] += eps_fd
                    lp = 0.5 * np.mean((fwd(p, X).flatten() - y) ** 2)
                    p[k] -= 2 * eps_fd
                    lm = 0.5 * np.mean((fwd(p, X).flatten() - y) ** 2)
                    p[k] += eps_fd
                    grad[k] = (lp - lm) / (2 * eps_fd)
                p -= lr * grad
            
            # Measure final drift
            Kt = compute_empirical_ntk(fwd, p, X)
            drift = np.linalg.norm(Kt - K0, 'fro') / K0_norm
            final_loss = 0.5 * np.mean((fwd(p, X).flatten() - y) ** 2)
            
            # Compute gamma (dimensionless coupling)
            # gamma = lr * N^{-(1-a-b)}, standard param: a=0, b=0
            gamma = lr * width ** 0  # standard param
            
            regime = 'lazy' if drift < 0.05 else ('rich' if drift > 0.2 else 'transition')
            
            grid_results.append({
                'lr': float(lr),
                'init_scale': float(sigma),
                'gamma': float(gamma),
                'drift': float(drift),
                'final_loss': float(final_loss),
                'regime': regime,
                'grid_i': i,
                'grid_j': j
            })
            
        if (i + 1) % 3 == 0:
            print(f"    Completed row {i + 1}/{len(lr_range)}")
    
    # Analyze boundary
    lazy_count = sum(1 for r in grid_results if r['regime'] == 'lazy')
    rich_count = sum(1 for r in grid_results if r['regime'] == 'rich')
    trans_count = sum(1 for r in grid_results if r['regime'] == 'transition')
    print(f"\n  Regime counts: lazy={lazy_count}, rich={rich_count}, transition={trans_count}")
    
    # Compute infinite-width prediction (analytic NTK is width-independent)
    antk = AnalyticNTK(depth=depth + 1, activation='relu')
    K_inf = antk.compute_ntk(X)
    max_eig_inf = float(np.max(np.linalg.eigvalsh(K_inf)))
    
    # Finite-width corrected prediction
    corrector = FiniteWidthCorrector(output_dim=1, min_widths=3)
    widths_cal = [32, 64, 128, 256]
    cal_ntks = []
    for w in widths_cal:
        params_w, fwd_w = make_mlp(input_dim, w, depth, seed=42)
        K_w = compute_empirical_ntk(fwd_w, params_w, X)
        cal_ntks.append(K_w)
    
    correction = corrector.compute_corrections_regression(np.array(cal_ntks), widths_cal)
    K_corrected = correction.theta_0 + correction.theta_1 / width
    max_eig_corrected = float(np.max(np.linalg.eigvalsh(K_corrected)))
    
    # Use PhaseBoundaryPredictor for proper γ* estimation (fixes Infinity bug)
    predictor = PhaseBoundaryPredictor()
    gs_result = predictor.predict_gamma_star(
        correction.theta_0, correction.theta_1, n_steps, width
    )
    gamma_star_corrected = gs_result.gamma_star
    
    # Infinite-width prediction (simple spectral estimate)
    gamma_star_inf = 1.0 / (max_eig_inf / n_samples + 1e-10)
    
    # IoU computation
    def compute_iou(grid, predicted_boundary_gamma, regime_key='lazy'):
        pred_lazy = [r for r in grid if r['gamma'] * r['init_scale'] < predicted_boundary_gamma]
        true_lazy = [r for r in grid if r['regime'] == regime_key]
        
        pred_set = set((r['grid_i'], r['grid_j']) for r in pred_lazy)
        true_set = set((r['grid_i'], r['grid_j']) for r in true_lazy)
        
        intersection = len(pred_set & true_set)
        union = len(pred_set | true_set)
        return intersection / max(union, 1)
    
    iou_inf = compute_iou(grid_results, gamma_star_inf)
    iou_corrected = compute_iou(grid_results, gamma_star_corrected)
    
    results = {
        'grid_results': grid_results,
        'phase_boundary': {
            'gamma_star_infinite_width': float(gamma_star_inf),
            'gamma_star_finite_width': float(gamma_star_corrected),
            'boundary_shift_pct': float(abs(gamma_star_corrected - gamma_star_inf) / gamma_star_inf * 100),
            'iou_infinite_width': float(iou_inf),
            'iou_finite_width': float(iou_corrected),
            'max_eigenvalue_infinite': float(max_eig_inf),
            'max_eigenvalue_corrected': float(max_eig_corrected)
        },
        'regime_counts': {
            'lazy': lazy_count,
            'rich': rich_count,
            'transition': trans_count
        },
        'metadata': {
            'experiment': 'phase_boundary_detection',
            'width': width,
            'depth': depth,
            'n_steps': n_steps,
            'input_dim': input_dim,
            'n_samples': n_samples,
            'grid_size': f'{len(lr_range)}x{len(init_scales)}',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    print(f"\n  Phase boundary results:")
    print(f"    γ* (infinite-width): {gamma_star_inf:.4f}")
    print(f"    γ* (finite-width):   {gamma_star_corrected:.4f}")
    print(f"    Boundary shift:      {results['phase_boundary']['boundary_shift_pct']:.1f}%")
    print(f"    IoU (infinite):      {iou_inf:.4f}")
    print(f"    IoU (finite):        {iou_corrected:.4f}")
    
    with open(os.path.join(DATA_DIR, 'exp3_phase_boundary_mlp.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("  Saved: exp3_phase_boundary_mlp.json")
    return results


def experiment_4_resnet_comparison():
    """Experiment 4: ResNet vs MLP phase boundaries.
    
    Shows that residual connections shift the phase boundary,
    requiring finite-width corrections.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: ResNet vs MLP Phase Boundaries")
    print("=" * 60)
    
    input_dim = 8
    n_samples = 6
    width = 32
    depth = 3
    n_steps = 150
    
    X = np.random.RandomState(42).randn(n_samples, input_dim)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    y = np.random.RandomState(42).randn(n_samples)
    
    lr_range = np.logspace(-3, 0, 10)
    init_scales = [0.5, 1.0, 1.5, 2.0]
    
    all_results = {}
    for arch_name, make_fn in [('mlp', make_mlp), ('resnet', make_resnet_block)]:
        print(f"\n  Architecture: {arch_name}")
        arch_results = []
        
        for lr in lr_range:
            for sigma in init_scales:
                params, fwd = make_fn(input_dim, width, depth, seed=42)
                params = params * sigma
                
                K0 = compute_empirical_ntk(fwd, params, X)
                K0_norm = np.linalg.norm(K0, 'fro')
                
                p = params.copy()
                eps_fd = 1e-5
                for step in range(n_steps):
                    pred = fwd(p, X)
                    grad = np.zeros_like(p)
                    for k in range(len(p)):
                        p[k] += eps_fd
                        lp = 0.5 * np.mean((fwd(p, X).flatten() - y) ** 2)
                        p[k] -= 2 * eps_fd
                        lm = 0.5 * np.mean((fwd(p, X).flatten() - y) ** 2)
                        p[k] += eps_fd
                        grad[k] = (lp - lm) / (2 * eps_fd)
                    p -= lr * grad
                
                Kt = compute_empirical_ntk(fwd, p, X)
                drift = np.linalg.norm(Kt - K0, 'fro') / K0_norm
                regime = 'lazy' if drift < 0.05 else ('rich' if drift > 0.2 else 'transition')
                
                arch_results.append({
                    'lr': float(lr),
                    'init_scale': float(sigma),
                    'drift': float(drift),
                    'regime': regime
                })
        
        lazy_count = sum(1 for r in arch_results if r['regime'] == 'lazy')
        rich_count = sum(1 for r in arch_results if r['regime'] == 'rich')
        print(f"    Lazy: {lazy_count}, Rich: {rich_count}, Trans: {len(arch_results) - lazy_count - rich_count}")
        
        # Fit corrections for this architecture
        widths_cal = [16, 32, 64, 128]
        cal_ntks = []
        for w in widths_cal:
            params_w, fwd_w = make_fn(input_dim, w, depth, seed=42)
            K_w = compute_empirical_ntk(fwd_w, params_w, X)
            cal_ntks.append(K_w)
        
        corrector = FiniteWidthCorrector(output_dim=1, min_widths=3)
        correction = corrector.compute_corrections_regression(np.array(cal_ntks), widths_cal)
        
        all_results[arch_name] = {
            'grid_results': arch_results,
            'correction_r_squared': float(correction.convergence_info.r_squared),
            'theta_0_norm': float(np.linalg.norm(correction.theta_0)),
            'theta_1_norm': float(np.linalg.norm(correction.theta_1)),
            'correction_ratio': float(np.linalg.norm(correction.theta_1) / (np.linalg.norm(correction.theta_0) + 1e-10))
        }
        print(f"    Correction R² = {correction.convergence_info.r_squared:.4f}")
        print(f"    Correction ratio |θ₁|/|θ₀| = {all_results[arch_name]['correction_ratio']:.4f}")
    
    # Compare boundaries
    mlp_boundary_lrs = [r['lr'] for r in all_results['mlp']['grid_results'] if r['regime'] == 'transition']
    res_boundary_lrs = [r['lr'] for r in all_results['resnet']['grid_results'] if r['regime'] == 'transition']
    
    if mlp_boundary_lrs and res_boundary_lrs:
        shift = abs(np.mean(np.log10(res_boundary_lrs)) - np.mean(np.log10(mlp_boundary_lrs)))
        all_results['boundary_comparison'] = {
            'log10_lr_shift': float(shift),
            'resnet_correction_ratio': all_results['resnet']['correction_ratio'],
            'mlp_correction_ratio': all_results['mlp']['correction_ratio']
        }
    
    all_results['metadata'] = {
        'experiment': 'resnet_vs_mlp_comparison',
        'width': width,
        'depth': depth,
        'n_steps': n_steps,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(DATA_DIR, 'exp4_resnet_comparison.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\n  Saved: exp4_resnet_comparison.json")
    return all_results


def experiment_5_depth_scaling():
    """Experiment 5: How depth affects the phase boundary.
    
    Measures correction magnitude and phase boundary location
    as a function of depth.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Depth Scaling of Phase Boundary")
    print("=" * 60)
    
    input_dim = 8
    n_samples = 6
    width = 64
    depths = [1, 2, 3, 4, 5]
    
    X = np.random.RandomState(42).randn(n_samples, input_dim)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    results = {}
    for depth in depths:
        print(f"\n  Depth {depth}:")
        
        # Analytic NTK
        antk = AnalyticNTK(depth=depth + 1, activation='relu')
        K_inf = antk.compute_ntk(X)
        eigs_inf = np.linalg.eigvalsh(K_inf)
        
        # Empirical NTKs at multiple widths
        widths_cal = [32, 64, 128, 256]
        cal_ntks = []
        for w in widths_cal:
            params, fwd = make_mlp(input_dim, w, depth, seed=42)
            K = compute_empirical_ntk(fwd, params, X)
            cal_ntks.append(K)
        
        corrector = FiniteWidthCorrector(output_dim=1, min_widths=3)
        correction = corrector.compute_corrections_regression(np.array(cal_ntks), widths_cal)
        
        # Eigenvalue analysis of correction
        eigs_correction = np.linalg.eigvalsh(correction.theta_1)
        
        # Spectral analysis
        K_corrected_64 = correction.theta_0 + correction.theta_1 / 64
        eigs_corrected = np.linalg.eigvalsh(K_corrected_64)
        
        results[f'depth_{depth}'] = {
            'depth': depth,
            'analytic_ntk_trace': float(np.trace(K_inf)),
            'analytic_ntk_condition': float(np.linalg.cond(K_inf)),
            'analytic_top_eigenvalue': float(eigs_inf[-1]),
            'correction_r_squared': float(correction.convergence_info.r_squared),
            'theta_0_norm': float(np.linalg.norm(correction.theta_0)),
            'theta_1_norm': float(np.linalg.norm(correction.theta_1)),
            'correction_ratio': float(np.linalg.norm(correction.theta_1) / (np.linalg.norm(correction.theta_0) + 1e-10)),
            'correction_top_eigenvalue': float(eigs_correction[-1]),
            'corrected_top_eigenvalue_N64': float(eigs_corrected[-1]),
            'eigenvalue_shift_pct': float(abs(eigs_corrected[-1] - eigs_inf[-1]) / (abs(eigs_inf[-1]) + 1e-10) * 100)
        }
        
        print(f"    R² = {correction.convergence_info.r_squared:.4f}")
        print(f"    Correction ratio = {results[f'depth_{depth}']['correction_ratio']:.4f}")
        print(f"    Eigenvalue shift = {results[f'depth_{depth}']['eigenvalue_shift_pct']:.1f}%")
    
    results['metadata'] = {
        'experiment': 'depth_scaling',
        'width': width,
        'input_dim': input_dim,
        'n_samples': n_samples,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(DATA_DIR, 'exp5_depth_scaling.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\n  Saved: exp5_depth_scaling.json")
    return results


def experiment_6_mup_scaling():
    """Experiment 6: μP scaling exponent effects.
    
    Varies (a, b) scaling exponents and measures how the phase boundary moves.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: μP Scaling Exponent Effects")
    print("=" * 60)
    
    input_dim = 8
    n_samples = 6
    depth = 2
    
    X = np.random.RandomState(42).randn(n_samples, input_dim)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    y = np.random.RandomState(42).randn(n_samples)
    
    # Different (a, b) parameterizations
    parameterizations = {
        'standard': (0.0, 0.0),
        'ntk': (0.5, 0.0),
        'mup': (0.5, 1.0),
        'intermediate_1': (0.25, 0.5),
        'intermediate_2': (0.5, 0.5),
    }
    
    results = {}
    for name, (a, b) in parameterizations.items():
        print(f"\n  Parameterization: {name} (a={a}, b={b})")
        
        widths = [32, 64, 128, 256]
        width_results = []
        
        for width in widths:
            # Apply μP scaling
            sigma = width ** (-a)  # init variance scaling
            eta_scale = width ** (-b)  # lr scaling
            
            params, fwd = make_mlp(input_dim, width, depth, seed=42)
            params = params * sigma
            
            K = compute_empirical_ntk(fwd, params, X)
            gamma = eta_scale * width ** (-(1 - a - b))
            
            width_results.append({
                'width': width,
                'ntk_trace': float(np.trace(K)),
                'ntk_max_eig': float(np.max(np.linalg.eigvalsh(K))),
                'gamma': float(gamma),
                'sigma': float(sigma),
                'eta_scale': float(eta_scale)
            })
            
        # Fit corrections
        cal_ntks = []
        for w in widths:
            sigma = w ** (-a)
            params, fwd = make_mlp(input_dim, w, depth, seed=42)
            params = params * sigma
            K = compute_empirical_ntk(fwd, params, X)
            cal_ntks.append(K)
        
        corrector = FiniteWidthCorrector(output_dim=1, min_widths=3)
        correction = corrector.compute_corrections_regression(np.array(cal_ntks), widths)
        
        results[name] = {
            'a': a,
            'b': b,
            'gamma_exponent': float(1 - a - b),
            'width_results': width_results,
            'correction_r_squared': float(correction.convergence_info.r_squared),
            'correction_ratio': float(np.linalg.norm(correction.theta_1) / (np.linalg.norm(correction.theta_0) + 1e-10))
        }
        
        print(f"    γ exponent = {1 - a - b:.2f}")
        print(f"    Correction R² = {correction.convergence_info.r_squared:.4f}")
        print(f"    Correction ratio = {results[name]['correction_ratio']:.4f}")
    
    results['metadata'] = {
        'experiment': 'mup_scaling_effects',
        'depth': depth,
        'input_dim': input_dim,
        'n_samples': n_samples,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(DATA_DIR, 'exp6_mup_scaling.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\n  Saved: exp6_mup_scaling.json")
    return results


def experiment_7_bifurcation_detection():
    """Experiment 7: Bifurcation detection accuracy.
    
    Uses the BifurcationDetector to find phase transitions
    and compares with ground-truth training.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 7: Bifurcation Detection")
    print("=" * 60)
    
    input_dim = 8
    n_samples = 6
    width = 64
    depth = 2
    
    X = np.random.RandomState(42).randn(n_samples, input_dim)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    # Compute NTK and its correction
    params, fwd = make_mlp(input_dim, width, depth, seed=42)
    K0 = compute_empirical_ntk(fwd, params, X)
    
    # Analytic NTK
    antk = AnalyticNTK(depth=depth + 1, activation='relu')
    K_inf = antk.compute_ntk(X)
    
    # Eigenvalue analysis as function of gamma
    eigs_inf = np.linalg.eigvalsh(K_inf)
    
    # For bifurcation detection, we look at when the Jacobian eigenvalues cross zero
    # J(gamma) = (gamma/N) * D_Theta F|_{Theta_inf}
    # The leading eigenvalue of J crosses zero at gamma* = N / lambda_max(D_Theta F)
    
    # Approximate D_Theta F from finite-width corrections
    widths_cal = [32, 64, 128, 256]
    cal_ntks = []
    for w in widths_cal:
        p, f = make_mlp(input_dim, w, depth, seed=42)
        K = compute_empirical_ntk(f, p, X)
        cal_ntks.append(K)
    
    corrector = FiniteWidthCorrector(output_dim=1, min_widths=3)
    correction = corrector.compute_corrections_regression(np.array(cal_ntks), widths_cal)
    
    # The correction theta_1 approximates N * (Theta - Theta_inf), 
    # so its eigenvalues relate to the Jacobian eigenvalues
    eigs_correction = np.linalg.eigvalsh(correction.theta_1)
    
    # Bifurcation point estimate
    bif_detector = BifurcationDetector(tol=1e-6)
    
    # Simulate spectral path: eigenvalues of J(gamma) as gamma varies
    gammas = np.logspace(-2, 1, 50)
    spectral_path = []
    for gamma in gammas:
        J_eigs = (gamma / width) * eigs_correction
        spectral_path.append(J_eigs)
    
    spectral_path = np.array(spectral_path)
    
    # Find first zero-crossing of maximum eigenvalue
    max_eig_path = spectral_path[:, -1]  # leading eigenvalue
    gamma_star = None
    for i in range(1, len(gammas)):
        if max_eig_path[i-1] < 0 and max_eig_path[i] >= 0:
            # Linear interpolation
            t = -max_eig_path[i-1] / (max_eig_path[i] - max_eig_path[i-1])
            gamma_star = gammas[i-1] + t * (gammas[i] - gammas[i-1])
            break
    
    results = {
        'infinite_width_eigenvalues': eigs_inf.tolist(),
        'correction_eigenvalues': eigs_correction.tolist(),
        'spectral_path': {
            'gammas': gammas.tolist(),
            'max_eigenvalue': max_eig_path.tolist()
        },
        'gamma_star_detected': float(gamma_star) if gamma_star else None,
        'correction_r_squared': float(correction.convergence_info.r_squared),
        'metadata': {
            'experiment': 'bifurcation_detection',
            'width': width,
            'depth': depth,
            'input_dim': input_dim,
            'n_samples': n_samples,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    if gamma_star:
        print(f"  Detected bifurcation at γ* = {gamma_star:.4f}")
    else:
        print(f"  No bifurcation detected (all eigenvalues negative or all positive)")
        print(f"  Max eigenvalue range: [{max_eig_path.min():.4f}, {max_eig_path.max():.4f}]")
    
    with open(os.path.join(DATA_DIR, 'exp7_bifurcation.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("  Saved: exp7_bifurcation.json")
    return results


def generate_summary_table(all_results):
    """Generate summary table from all experiments."""
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    
    # Table 1: NTK convergence rates
    if 'exp1' in all_results:
        print("\nTable 1: NTK Convergence Rates")
        print(f"{'Depth':<8} {'Rate':<12} {'Expected':<12}")
        print("-" * 32)
        for key, data in all_results['exp1'].items():
            if key.startswith('depth_'):
                d = data['depth']
                rate = data['convergence_rate']
                print(f"{d:<8} {rate:<12.3f} {-0.5:<12.3f}")
    
    # Table 2: Correction quality
    if 'exp2' in all_results:
        print("\nTable 2: Correction Fitting Quality")
        print(f"{'Depth':<8} {'R²':<10} {'Test Error':<12}")
        print("-" * 30)
        for key, data in all_results['exp2'].items():
            if key.startswith('depth_'):
                d = data['depth']
                r2 = data['mean_r_squared']
                te = data['mean_test_error']
                print(f"{d:<8} {r2:<10.4f} {te:<12.4f}")
    
    summary = {
        'experiments_completed': list(all_results.keys()),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(DATA_DIR, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    np.random.seed(42)
    
    all_results = {}
    
    # Run experiments
    all_results['exp1'] = experiment_1_ntk_convergence()
    all_results['exp2'] = experiment_2_correction_fitting()
    all_results['exp5'] = experiment_5_depth_scaling()
    all_results['exp6'] = experiment_6_mup_scaling()
    all_results['exp7'] = experiment_7_bifurcation_detection()
    
    # These are slower (require training), run with smaller grids
    all_results['exp3'] = experiment_3_phase_boundary()
    all_results['exp4'] = experiment_4_resnet_comparison()
    
    generate_summary_table(all_results)
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved in: {DATA_DIR}")
    print("=" * 60)
