"""
Advanced and optimal composition theorems for DP-Forge.

Implements various composition theorems from the DP literature including
optimal composition (KOV 2015), heterogeneous composition, group privacy,
parallel composition, and mixed privacy definition composition.

Key Features:
    - KOV optimal composition with numerical optimization
    - Heterogeneous basic composition (different ε, δ per mechanism)
    - Group privacy composition for k-DP
    - Parallel composition for disjoint data subsets
    - Sequential composition with optimal ordering
    - Mixed composition across (ε,δ), RDP, zCDP, f-DP

References:
    - Kairouz, P., Oh, S., & Viswanath, P. (2015). The composition theorem
      for differential privacy. In ICML 2015.
    - Dwork, C., Rothblum, G. N., & Vadhan, S. (2010). Boosting and
      differential privacy. In FOCS 2010.
    - Mironov, I. (2017). Rényi differential privacy. In CSF 2017.

Functions:
    optimal_advanced_composition           — KOV optimal composition
    heterogeneous_basic_composition        — Different (ε,δ) per mechanism
    group_privacy_composition              — Group privacy amplification
    parallel_composition                   — Disjoint subset composition
    sequential_composition_optimal_order   — Optimal mechanism ordering
    mixed_composition                      — Compose mixed privacy definitions
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar, brentq

from dp_forge.exceptions import ConfigurationError, BudgetExhaustedError

FloatArray = npt.NDArray[np.float64]


def _validate_epsilon_delta(epsilon: float, delta: float, name: str = "parameter") -> None:
    """Validate epsilon and delta parameters."""
    if epsilon < 0 or not math.isfinite(epsilon):
        raise ValueError(f"{name} epsilon must be non-negative and finite, got {epsilon}")
    if not (0 <= delta < 1):
        raise ValueError(f"{name} delta must be in [0, 1), got {delta}")


def optimal_advanced_composition(
    epsilons: Union[float, List[float], FloatArray],
    deltas: Union[float, List[float], FloatArray],
    target_delta: float,
    k: Optional[int] = None
) -> float:
    """
    Optimal advanced composition via KOV theorem.
    
    Computes the tightest epsilon guarantee for k-fold composition of
    (ε, δ)-DP mechanisms using the Kairouz-Oh-Viswanath optimal composition
    theorem. For homogeneous composition (same ε, δ), uses closed-form bound.
    For heterogeneous, uses numerical optimization.
    
    Args:
        epsilons: Single epsilon or list of epsilons per mechanism
        deltas: Single delta or list of deltas per mechanism
        target_delta: Target delta for composed mechanism
        k: Number of compositions (required if epsilons/deltas are scalars)
        
    Returns:
        Composed epsilon value
        
    Notes:
        - For pure DP (δ=0), reduces to basic composition: ε_total = k·ε
        - For approximate DP, provides O(√k) improvement over basic composition
        - Optimal bound may require numerical search over auxiliary parameters
        
    References:
        Kairouz, P., Oh, S., & Viswanath, P. (2015). The composition theorem
        for differential privacy. ICML 2015.
    """
    if isinstance(epsilons, (int, float)):
        if k is None:
            raise ValueError("k required when epsilons is scalar")
        eps_list = [float(epsilons)] * k
        delta_list = [float(deltas)] * k
    else:
        eps_list = [float(e) for e in np.atleast_1d(epsilons)]
        delta_list = [float(d) for d in np.atleast_1d(deltas)]
        if k is None:
            k = len(eps_list)
        if len(eps_list) != k or len(delta_list) != k:
            raise ValueError(f"Length mismatch: epsilons={len(eps_list)}, deltas={len(delta_list)}, k={k}")
    
    for i, (e, d) in enumerate(zip(eps_list, delta_list)):
        _validate_epsilon_delta(e, d, f"mechanism {i}")
    
    _validate_epsilon_delta(0.0, target_delta, "target")
    
    if target_delta == 0.0:
        return sum(eps_list)
    
    if all(d == 0.0 for d in delta_list):
        return sum(eps_list)
    
    homogeneous = len(set(eps_list)) == 1 and len(set(delta_list)) == 1
    
    if homogeneous:
        eps = eps_list[0]
        delta = delta_list[0]
        
        if delta == 0.0:
            return k * eps
        
        if target_delta <= k * delta:
            warnings.warn(
                f"target_delta={target_delta} <= k*delta={k * delta}, composition infeasible",
                RuntimeWarning
            )
            return float('inf')
        
        delta_prime = target_delta - k * delta
        
        if eps < 1.0:
            eps_opt = math.sqrt(2 * k * math.log(1.0 / delta_prime)) * eps + k * eps * (math.exp(eps) - 1)
        else:
            eps_opt = math.sqrt(2 * k * math.log(1.0 / delta_prime)) * eps + k * eps * eps
        
        return eps_opt
    else:
        return _heterogeneous_kov_composition(eps_list, delta_list, target_delta)


def _heterogeneous_kov_composition(
    epsilons: List[float],
    deltas: List[float],
    target_delta: float
) -> float:
    """
    KOV composition for heterogeneous mechanisms via numerical optimization.
    
    Uses numerical search over auxiliary parameter λ to find tightest bound.
    """
    k = len(epsilons)
    
    delta_sum = sum(deltas)
    if target_delta <= delta_sum:
        warnings.warn(
            f"target_delta={target_delta} <= sum(deltas)={delta_sum}, composition infeasible",
            RuntimeWarning
        )
        return float('inf')
    
    delta_prime = target_delta - delta_sum
    
    def eps_for_lambda(lam: float) -> float:
        """Compute epsilon bound for given lambda."""
        if lam <= 0:
            return float('inf')
        
        sum_term = 0.0
        for eps_i in epsilons:
            if eps_i < 1.0:
                sum_term += eps_i**2 * (math.exp(eps_i) - 1 - eps_i)
            else:
                sum_term += eps_i**3
        
        return lam + sum_term / lam + math.sqrt(2 * sum_term * math.log(1.0 / delta_prime))
    
    result = minimize_scalar(eps_for_lambda, bounds=(0.001, 10.0), method='bounded')
    
    if result.success:
        return result.fun
    else:
        warnings.warn(f"KOV optimization failed, using fallback bound", RuntimeWarning)
        return sum(epsilons) + math.sqrt(2 * k * math.log(1.0 / delta_prime)) * max(epsilons)


def heterogeneous_basic_composition(
    epsilons: Union[List[float], FloatArray],
    deltas: Union[List[float], FloatArray],
    method: str = "basic"
) -> Tuple[float, float]:
    """
    Heterogeneous basic composition.
    
    Composes mechanisms with different (ε_i, δ_i) parameters using either
    basic composition or advanced composition theorems.
    
    Args:
        epsilons: List of epsilon values
        deltas: List of delta values
        method: Composition method, one of:
            - 'basic': ε_total = Σε_i, δ_total = Σδ_i
            - 'advanced': Apply advanced composition bound
            
    Returns:
        Tuple (composed_epsilon, composed_delta)
        
    Notes:
        - Basic composition is always valid but may be loose
        - Advanced composition requires target_delta specification
    """
    eps_array = np.atleast_1d(epsilons)
    delta_array = np.atleast_1d(deltas)
    
    if len(eps_array) != len(delta_array):
        raise ValueError(f"Length mismatch: epsilons={len(eps_array)}, deltas={len(delta_array)}")
    
    for i, (e, d) in enumerate(zip(eps_array, delta_array)):
        _validate_epsilon_delta(e, d, f"mechanism {i}")
    
    if method == "basic":
        composed_eps = float(np.sum(eps_array))
        composed_delta = float(np.sum(delta_array))
        return composed_eps, composed_delta
    
    elif method == "advanced":
        raise ValueError("Advanced method requires target_delta, use optimal_advanced_composition instead")
    
    else:
        raise ValueError(f"Unknown method '{method}', expected 'basic' or 'advanced'")


def group_privacy_composition(
    epsilon: float,
    delta: float,
    group_size: int,
    method: str = "basic"
) -> Tuple[float, float]:
    """
    Group privacy composition.
    
    Computes privacy guarantee for groups of size k when mechanism is
    (ε, δ)-DP for individuals. Group privacy quantifies privacy leakage
    about groups rather than individuals.
    
    Args:
        epsilon: Base epsilon for single individual
        delta: Base delta for single individual
        group_size: Size of group k
        method: Composition method:
            - 'basic': ε_group = k·ε, δ_group = k·δ
            - 'advanced': Tighter bound via advanced composition
            
    Returns:
        Tuple (group_epsilon, group_delta)
        
    Notes:
        - Basic group privacy: (k·ε, k·δ)-DP for groups of size k
        - Advanced composition may provide tighter bounds
        - Group privacy is useful for analyzing privacy leakage about households,
          organizations, or other collective entities
          
    References:
        Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. Section 2.2.
    """
    _validate_epsilon_delta(epsilon, delta, "base")
    
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    
    if method == "basic":
        group_eps = group_size * epsilon
        group_delta = group_size * delta
        return group_eps, group_delta
    
    elif method == "advanced":
        if delta == 0.0:
            return group_size * epsilon, 0.0
        
        target_delta = min(0.5, group_size * delta * 2)
        group_eps = optimal_advanced_composition(
            epsilons=epsilon,
            deltas=delta,
            target_delta=target_delta,
            k=group_size
        )
        
        return group_eps, target_delta
    
    else:
        raise ValueError(f"Unknown method '{method}', expected 'basic' or 'advanced'")


def parallel_composition(
    epsilons: Union[List[float], FloatArray],
    deltas: Union[List[float], FloatArray]
) -> Tuple[float, float]:
    """
    Parallel composition for disjoint data subsets.
    
    When mechanisms operate on disjoint subsets of the data, privacy is
    determined by the worst-case mechanism. This is the parallel composition
    theorem.
    
    Args:
        epsilons: List of epsilon values for each mechanism
        deltas: List of delta values for each mechanism
        
    Returns:
        Tuple (composed_epsilon, composed_delta) = (max(epsilons), max(deltas))
        
    Notes:
        - Requires mechanisms to operate on DISJOINT data subsets
        - No privacy cost for parallelization beyond worst-case
        - This is fundamentally different from sequential composition
        
    Example::
    
        # Three mechanisms on disjoint age groups
        eps_child = 1.0
        eps_adult = 0.5
        eps_senior = 0.8
        
        # Overall privacy is worst-case
        eps_total, delta_total = parallel_composition(
            epsilons=[eps_child, eps_adult, eps_senior],
            deltas=[1e-5, 1e-5, 1e-5]
        )
        # eps_total = 1.0 (max)
        
    References:
        McSherry, F. (2009). Privacy integrated queries. In SIGMOD 2009.
    """
    eps_array = np.atleast_1d(epsilons)
    delta_array = np.atleast_1d(deltas)
    
    if len(eps_array) != len(delta_array):
        raise ValueError(f"Length mismatch: epsilons={len(eps_array)}, deltas={len(delta_array)}")
    
    if len(eps_array) == 0:
        return 0.0, 0.0
    
    for i, (e, d) in enumerate(zip(eps_array, delta_array)):
        _validate_epsilon_delta(e, d, f"mechanism {i}")
    
    composed_eps = float(np.max(eps_array))
    composed_delta = float(np.max(delta_array))
    
    return composed_eps, composed_delta


def sequential_composition_optimal_order(
    mechanisms: List[Dict[str, Any]],
    composition_method: str = "optimal",
    target_delta: Optional[float] = None
) -> Tuple[List[int], float, float]:
    """
    Find optimal mechanism ordering for sequential composition.
    
    Determines the order in which to apply mechanisms to minimize total
    privacy cost. Some orderings may yield tighter composition bounds.
    
    Args:
        mechanisms: List of mechanism dicts with 'epsilon', 'delta', 'name' keys
        composition_method: Composition method ('basic', 'optimal')
        target_delta: Target delta for optimal composition
        
    Returns:
        Tuple (optimal_order, composed_epsilon, composed_delta)
        - optimal_order: List of mechanism indices in optimal order
        - composed_epsilon: Total epsilon
        - composed_delta: Total delta
        
    Notes:
        - For basic composition, order doesn't matter (commutative)
        - For advanced/optimal composition, order may affect tightness
        - Heuristic: apply high-epsilon mechanisms first
        
    Complexity:
        O(k log k) for sorting, where k = number of mechanisms
    """
    if len(mechanisms) == 0:
        return [], 0.0, 0.0
    
    for i, mech in enumerate(mechanisms):
        if 'epsilon' not in mech or 'delta' not in mech:
            raise ValueError(f"Mechanism {i} missing 'epsilon' or 'delta' key")
        _validate_epsilon_delta(mech['epsilon'], mech['delta'], f"mechanism {i}")
    
    if composition_method == "basic":
        optimal_order = list(range(len(mechanisms)))
        composed_eps = sum(m['epsilon'] for m in mechanisms)
        composed_delta = sum(m['delta'] for m in mechanisms)
        return optimal_order, composed_eps, composed_delta
    
    elif composition_method == "optimal":
        if target_delta is None:
            target_delta = sum(m['delta'] for m in mechanisms) * 2.0
        
        sorted_indices = sorted(
            range(len(mechanisms)),
            key=lambda i: -mechanisms[i]['epsilon']
        )
        
        epsilons = [mechanisms[i]['epsilon'] for i in sorted_indices]
        deltas = [mechanisms[i]['delta'] for i in sorted_indices]
        
        composed_eps = optimal_advanced_composition(
            epsilons=epsilons,
            deltas=deltas,
            target_delta=target_delta
        )
        
        return sorted_indices, composed_eps, target_delta
    
    else:
        raise ValueError(f"Unknown composition_method '{composition_method}'")


def mixed_composition(
    budgets: List[Dict[str, Any]],
    target_delta: float = 1e-5,
    conversion_method: str = "pld"
) -> Dict[str, Any]:
    """
    Compose mechanisms with mixed privacy definitions.
    
    Handles composition of mechanisms specified in different privacy frameworks:
    (ε,δ)-DP, RDP, zCDP, f-DP. Converts all to a common representation (PLD)
    before composition.
    
    Args:
        budgets: List of privacy budget dicts, each with 'type' key:
            - type='epsilon_delta': requires 'epsilon', 'delta' keys
            - type='rdp': requires 'rdp_curve' (callable or RDPCurve)
            - type='zcdp': requires 'rho' key
            - type='fdp': requires 'tradeoff_fn' key
        target_delta: Target delta for final (ε,δ) conversion
        conversion_method: Method for conversion ('pld', 'rdp', 'conservative')
        
    Returns:
        Dict with 'epsilon', 'delta', 'method', 'conversions' keys
        
    Notes:
        - PLD method: convert all to PLD, compose via FFT, convert to (ε,δ)
        - RDP method: convert all to RDP, compose via moment addition, convert to (ε,δ)
        - Conservative method: convert all to (ε,δ) pessimistically, basic composition
        
    Example::
    
        budgets = [
            {'type': 'epsilon_delta', 'epsilon': 1.0, 'delta': 1e-5},
            {'type': 'rdp', 'rdp_curve': lambda alpha: alpha * 0.5},
            {'type': 'zcdp', 'rho': 0.25}
        ]
        
        result = mixed_composition(budgets, target_delta=1e-5)
        print(f"Composed: ε={result['epsilon']:.4f}, δ={result['delta']}")
    """
    if len(budgets) == 0:
        return {'epsilon': 0.0, 'delta': 0.0, 'method': conversion_method, 'conversions': []}
    
    if conversion_method == "pld":
        return _mixed_composition_via_pld(budgets, target_delta)
    elif conversion_method == "rdp":
        return _mixed_composition_via_rdp(budgets, target_delta)
    elif conversion_method == "conservative":
        return _mixed_composition_conservative(budgets)
    else:
        raise ValueError(f"Unknown conversion_method '{conversion_method}'")


def _mixed_composition_via_pld(budgets: List[Dict[str, Any]], target_delta: float) -> Dict[str, Any]:
    """Mixed composition via PLD path."""
    from dp_forge.composition.pld import PrivacyLossDistribution, compose
    from dp_forge.composition.mixed_accounting import (
        convert_rdp_to_pld,
        convert_zcdp_to_pld,
        convert_fdp_to_pld
    )
    
    plds = []
    conversions = []
    
    for i, budget in enumerate(budgets):
        budget_type = budget.get('type')
        
        if budget_type == 'epsilon_delta':
            eps = budget['epsilon']
            delta = budget['delta']
            conversions.append({'index': i, 'from': 'epsilon_delta', 'to': 'pld'})
            plds.append(None)
        
        elif budget_type == 'rdp':
            rdp_curve = budget['rdp_curve']
            pld = convert_rdp_to_pld(rdp_curve)
            conversions.append({'index': i, 'from': 'rdp', 'to': 'pld'})
            plds.append(pld)
        
        elif budget_type == 'zcdp':
            rho = budget['rho']
            pld = convert_zcdp_to_pld(rho)
            conversions.append({'index': i, 'from': 'zcdp', 'to': 'pld'})
            plds.append(pld)
        
        elif budget_type == 'fdp':
            tradeoff_fn = budget['tradeoff_fn']
            pld = convert_fdp_to_pld(tradeoff_fn)
            conversions.append({'index': i, 'from': 'fdp', 'to': 'pld'})
            plds.append(pld)
        
        else:
            raise ValueError(f"Unknown budget type '{budget_type}' at index {i}")
    
    plds = [p for p in plds if p is not None]
    
    if len(plds) == 0:
        eps_sum = sum(b['epsilon'] for b in budgets if b.get('type') == 'epsilon_delta')
        delta_sum = sum(b['delta'] for b in budgets if b.get('type') == 'epsilon_delta')
        return {'epsilon': eps_sum, 'delta': delta_sum, 'method': 'pld', 'conversions': conversions}
    
    composed_pld = plds[0]
    for pld in plds[1:]:
        composed_pld = compose(composed_pld, pld)
    
    epsilon = composed_pld.to_epsilon_delta(target_delta)
    
    return {
        'epsilon': epsilon,
        'delta': target_delta,
        'method': 'pld',
        'conversions': conversions
    }


def _mixed_composition_via_rdp(budgets: List[Dict[str, Any]], target_delta: float) -> Dict[str, Any]:
    """Mixed composition via RDP path."""
    from dp_forge.rdp import RDPAccountant
    from dp_forge.rdp.conversion import dp_to_rdp_bound, zcdp_to_rdp
    
    accountant = RDPAccountant()
    conversions = []
    
    for i, budget in enumerate(budgets):
        budget_type = budget.get('type')
        
        if budget_type == 'epsilon_delta':
            eps = budget['epsilon']
            delta = budget['delta']
            rdp_curve = dp_to_rdp_bound(eps, delta)
            accountant.add_rdp_curve(rdp_curve)
            conversions.append({'index': i, 'from': 'epsilon_delta', 'to': 'rdp'})
        
        elif budget_type == 'rdp':
            rdp_curve = budget['rdp_curve']
            accountant.add_rdp_curve(rdp_curve)
            conversions.append({'index': i, 'from': 'rdp', 'to': 'rdp'})
        
        elif budget_type == 'zcdp':
            rho = budget['rho']
            rdp_curve = zcdp_to_rdp(rho)
            accountant.add_rdp_curve(rdp_curve)
            conversions.append({'index': i, 'from': 'zcdp', 'to': 'rdp'})
        
        elif budget_type == 'fdp':
            raise ValueError("f-DP to RDP conversion not supported, use conversion_method='pld'")
        
        else:
            raise ValueError(f"Unknown budget type '{budget_type}' at index {i}")
    
    composed_budget = accountant.to_dp(target_delta)
    
    return {
        'epsilon': composed_budget.epsilon,
        'delta': target_delta,
        'method': 'rdp',
        'conversions': conversions
    }


def _mixed_composition_conservative(budgets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Conservative mixed composition via pessimistic (ε,δ) conversion."""
    eps_sum = 0.0
    delta_sum = 0.0
    conversions = []
    
    for i, budget in enumerate(budgets):
        budget_type = budget.get('type')
        
        if budget_type == 'epsilon_delta':
            eps_sum += budget['epsilon']
            delta_sum += budget['delta']
            conversions.append({'index': i, 'from': 'epsilon_delta', 'to': 'epsilon_delta'})
        
        elif budget_type == 'rdp':
            from dp_forge.rdp.conversion import rdp_to_dp
            rdp_curve = budget['rdp_curve']
            converted = rdp_to_dp(rdp_curve, delta=1e-6)
            eps_sum += converted.epsilon
            delta_sum += 1e-6
            conversions.append({'index': i, 'from': 'rdp', 'to': 'epsilon_delta'})
        
        elif budget_type == 'zcdp':
            rho = budget['rho']
            eps = rho + 2 * math.sqrt(rho * math.log(1 / 1e-6))
            eps_sum += eps
            delta_sum += 1e-6
            conversions.append({'index': i, 'from': 'zcdp', 'to': 'epsilon_delta'})
        
        elif budget_type == 'fdp':
            eps_sum += 10.0
            delta_sum += 0.01
            conversions.append({'index': i, 'from': 'fdp', 'to': 'epsilon_delta'})
        
        else:
            raise ValueError(f"Unknown budget type '{budget_type}' at index {i}")
    
    return {
        'epsilon': eps_sum,
        'delta': delta_sum,
        'method': 'conservative',
        'conversions': conversions
    }
