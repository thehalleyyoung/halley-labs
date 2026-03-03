"""Input validation utilities for the CPA engine.

Provides decorators, standalone validators, and composable checking
functions used to guard public APIs with informative error messages.
"""

from __future__ import annotations

import functools
import inspect
from collections import deque
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])


# ===================================================================
# Low-level validators (raise on failure)
# ===================================================================


def validate_adjacency_matrix(
    adj: np.ndarray,
    *,
    name: str = "adjacency_matrix",
    allow_weighted: bool = True,
) -> None:
    """Validate that *adj* is a well-formed adjacency matrix.

    Parameters
    ----------
    adj : np.ndarray
        2-D array to validate.
    name : str
        Name used in error messages.
    allow_weighted : bool
        If ``False``, entries must be 0 or 1.

    Raises
    ------
    TypeError
        If *adj* is not an ndarray.
    ValueError
        If shape, dtype, or value constraints are violated.
    """
    if not isinstance(adj, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(adj).__name__}")
    if adj.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got {adj.ndim}-D array")
    if adj.shape[0] != adj.shape[1]:
        raise ValueError(
            f"{name} must be square, got shape {adj.shape}"
        )
    if not np.issubdtype(adj.dtype, np.number):
        raise ValueError(f"{name} must have numeric dtype, got {adj.dtype}")
    if np.any(np.isnan(adj)):
        raise ValueError(f"{name} contains NaN values")
    if np.any(np.isinf(adj)):
        raise ValueError(f"{name} contains infinite values")
    if not allow_weighted:
        unique = np.unique(adj)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError(
                f"{name} must be binary (0/1) when allow_weighted=False, "
                f"found values {unique}"
            )


def validate_square_matrix(
    mat: np.ndarray,
    *,
    name: str = "matrix",
    min_size: int = 0,
) -> None:
    """Validate that *mat* is a 2-D square numeric array.

    Parameters
    ----------
    mat : np.ndarray
        Array to check.
    name : str
        Label for error messages.
    min_size : int
        Minimum number of rows/columns.

    Raises
    ------
    TypeError, ValueError
    """
    if not isinstance(mat, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(mat).__name__}")
    if mat.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got {mat.ndim}-D")
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} must be square, got shape {mat.shape}")
    if mat.shape[0] < min_size:
        raise ValueError(
            f"{name} must be at least {min_size}×{min_size}, got {mat.shape}"
        )


def validate_array_shape(
    arr: np.ndarray,
    *,
    expected_shape: Optional[Tuple[Optional[int], ...]] = None,
    expected_ndim: Optional[int] = None,
    name: str = "array",
) -> None:
    """Validate array shape and dimensionality.

    Parameters
    ----------
    arr : np.ndarray
        Array to validate.
    expected_shape : tuple of int or None, optional
        Expected shape.  ``None`` entries act as wildcards.
    expected_ndim : int, optional
        Expected number of dimensions.
    name : str
        Label for error messages.

    Raises
    ------
    TypeError, ValueError
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(arr).__name__}")
    if expected_ndim is not None and arr.ndim != expected_ndim:
        raise ValueError(
            f"{name} must be {expected_ndim}-D, got {arr.ndim}-D (shape {arr.shape})"
        )
    if expected_shape is not None:
        if len(expected_shape) != arr.ndim:
            raise ValueError(
                f"{name} expected {len(expected_shape)}-D, got {arr.ndim}-D"
            )
        for i, (exp, act) in enumerate(zip(expected_shape, arr.shape)):
            if exp is not None and exp != act:
                raise ValueError(
                    f"{name} dimension {i}: expected {exp}, got {act} "
                    f"(full shape {arr.shape})"
                )


def validate_dtype(
    arr: np.ndarray,
    *,
    allowed: Sequence[np.dtype] | None = None,
    must_be_numeric: bool = False,
    name: str = "array",
) -> None:
    """Validate array dtype.

    Parameters
    ----------
    arr : np.ndarray
        Array to check.
    allowed : sequence of dtypes, optional
        Whitelist of allowed dtypes.
    must_be_numeric : bool
        If ``True``, dtype must be a numeric sub-dtype.
    name : str
        Label for error messages.

    Raises
    ------
    TypeError, ValueError
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(arr).__name__}")
    if must_be_numeric and not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must have numeric dtype, got {arr.dtype}")
    if allowed is not None:
        if arr.dtype not in allowed:
            raise ValueError(
                f"{name} dtype {arr.dtype} not in allowed set "
                f"{[str(d) for d in allowed]}"
            )


def validate_probability(
    value: float,
    *,
    name: str = "probability",
    strict: bool = False,
) -> None:
    """Validate that *value* is a probability in [0, 1] (or (0,1) if strict).

    Parameters
    ----------
    value : float
        Value to check.
    name : str
        Label for error messages.
    strict : bool
        If ``True``, boundary values 0 and 1 are disallowed.

    Raises
    ------
    TypeError, ValueError
    """
    if not isinstance(value, (int, float, np.floating, np.integer)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    v = float(value)
    if np.isnan(v) or np.isinf(v):
        raise ValueError(f"{name} must be finite, got {v}")
    if strict:
        if v <= 0.0 or v >= 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {v}")
    else:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {v}")


def validate_positive(
    value: float,
    *,
    name: str = "value",
    allow_zero: bool = False,
) -> None:
    """Validate that *value* is positive (or non-negative).

    Parameters
    ----------
    value : float
        Value to check.
    name : str
        Label for error messages.
    allow_zero : bool
        If ``True``, zero is accepted.

    Raises
    ------
    TypeError, ValueError
    """
    if not isinstance(value, (int, float, np.floating, np.integer)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    v = float(value)
    if np.isnan(v) or np.isinf(v):
        raise ValueError(f"{name} must be finite, got {v}")
    if allow_zero:
        if v < 0.0:
            raise ValueError(f"{name} must be >= 0, got {v}")
    else:
        if v <= 0.0:
            raise ValueError(f"{name} must be > 0, got {v}")


def validate_sample_size(
    n: int,
    *,
    min_size: int = 1,
    name: str = "sample_size",
) -> None:
    """Validate that *n* is a positive integer at least *min_size*.

    Parameters
    ----------
    n : int
        Sample size to validate.
    min_size : int
        Minimum acceptable value.
    name : str
        Label for error messages.

    Raises
    ------
    TypeError, ValueError
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(n).__name__}")
    if int(n) < min_size:
        raise ValueError(f"{name} must be >= {min_size}, got {n}")


def validate_dag(
    adj: np.ndarray,
    *,
    name: str = "adjacency_matrix",
) -> None:
    """Validate that *adj* encodes a directed acyclic graph.

    Uses Kahn's algorithm to check for cycles.

    Parameters
    ----------
    adj : np.ndarray
        Binary adjacency matrix (adj[i,j]=1 means i -> j).
    name : str
        Label for error messages.

    Raises
    ------
    ValueError
        If the graph contains a cycle.
    """
    validate_adjacency_matrix(adj, name=name)
    n = adj.shape[0]
    binary = (adj != 0).astype(int)
    in_degree = binary.sum(axis=0).astype(int)
    queue: deque[int] = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for child in range(n):
            if binary[node, child]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
    if visited != n:
        raise ValueError(
            f"{name} contains a cycle — only {visited}/{n} nodes "
            "could be topologically sorted"
        )


def validate_variable_names(
    names: Sequence[str],
    *,
    expected_count: Optional[int] = None,
    label: str = "variable_names",
) -> None:
    """Validate a sequence of variable name strings.

    Parameters
    ----------
    names : sequence of str
        Variable names.
    expected_count : int, optional
        If given, the length must match.
    label : str
        Label for error messages.

    Raises
    ------
    TypeError, ValueError
    """
    if not isinstance(names, (list, tuple)):
        raise TypeError(f"{label} must be a list or tuple, got {type(names).__name__}")
    if expected_count is not None and len(names) != expected_count:
        raise ValueError(
            f"{label} length {len(names)} != expected {expected_count}"
        )
    seen: Set[str] = set()
    for i, nm in enumerate(names):
        if not isinstance(nm, str):
            raise TypeError(
                f"{label}[{i}] must be a string, got {type(nm).__name__}"
            )
        if not nm:
            raise ValueError(f"{label}[{i}] is empty")
        if nm in seen:
            raise ValueError(f"{label} contains duplicate '{nm}'")
        seen.add(nm)


def validate_context_id(cid: str, *, label: str = "context_id") -> None:
    """Validate a context identifier string.

    Parameters
    ----------
    cid : str
        Context id.
    label : str
        Label for error messages.

    Raises
    ------
    TypeError, ValueError
    """
    if not isinstance(cid, str):
        raise TypeError(f"{label} must be a string, got {type(cid).__name__}")
    if not cid.strip():
        raise ValueError(f"{label} must be non-empty")


def validate_dict_keys(
    d: Dict[str, Any],
    *,
    required: Optional[Iterable[str]] = None,
    optional: Optional[Iterable[str]] = None,
    label: str = "dict",
) -> None:
    """Check that *d* has the required keys and no unexpected extras.

    Parameters
    ----------
    d : dict
        Dictionary to check.
    required : iterable of str, optional
        Keys that must be present.
    optional : iterable of str, optional
        Keys that may be present.  If both *required* and *optional* are
        given, any key not in either set raises.
    label : str
        Label for error messages.

    Raises
    ------
    TypeError, ValueError
    """
    if not isinstance(d, dict):
        raise TypeError(f"{label} must be a dict, got {type(d).__name__}")
    req = set(required or [])
    opt = set(optional or [])
    missing = req - set(d)
    if missing:
        raise ValueError(f"{label} missing required keys: {sorted(missing)}")
    if req or opt:
        allowed = req | opt
        extra = set(d) - allowed
        if extra:
            raise ValueError(f"{label} has unexpected keys: {sorted(extra)}")


# ===================================================================
# Decorator-based validation
# ===================================================================


class _ParamSpec:
    """Specification for validating a single parameter."""

    __slots__ = ("type_", "check")

    def __init__(
        self,
        type_: Optional[Union[Type, Tuple[Type, ...]]] = None,
        check: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self.type_ = type_
        self.check = check


def validated(**specs: _ParamSpec) -> Callable[[F], F]:
    """Decorator that validates function parameters.

    Parameters
    ----------
    **specs : _ParamSpec
        Mapping of parameter name → ``_ParamSpec``.

    Returns
    -------
    Callable
        Decorated function with validation.

    Examples
    --------
    >>> @validated(alpha=_ParamSpec(type_=float, check=lambda v: validate_probability(v, name="alpha")))
    ... def fit(alpha: float) -> None:
    ...     ...
    """

    def decorator(fn: F) -> F:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            for param_name, spec in specs.items():
                if param_name not in bound.arguments:
                    continue
                val = bound.arguments[param_name]
                if spec.type_ is not None and not isinstance(val, spec.type_):
                    raise TypeError(
                        f"Parameter '{param_name}' of {fn.__qualname__} must be "
                        f"{spec.type_}, got {type(val).__name__}"
                    )
                if spec.check is not None:
                    spec.check(val)
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# ===================================================================
# Composite validators
# ===================================================================


def validate_data_matrix(
    X: np.ndarray,
    *,
    min_samples: int = 1,
    min_variables: int = 1,
    name: str = "X",
) -> None:
    """Validate an n×p data matrix.

    Parameters
    ----------
    X : np.ndarray
        2-D data matrix (samples × variables).
    min_samples : int
        Minimum number of rows.
    min_variables : int
        Minimum number of columns.
    name : str
        Label for error messages.

    Raises
    ------
    TypeError, ValueError
    """
    if not isinstance(X, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(X).__name__}")
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got {X.ndim}-D")
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got dtype {X.dtype}")
    if X.shape[0] < min_samples:
        raise ValueError(
            f"{name} requires at least {min_samples} samples, got {X.shape[0]}"
        )
    if X.shape[1] < min_variables:
        raise ValueError(
            f"{name} requires at least {min_variables} variables, got {X.shape[1]}"
        )
    if np.any(np.isnan(X)):
        raise ValueError(f"{name} contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError(f"{name} contains infinite values")


def validate_covariance_matrix(
    cov: np.ndarray,
    *,
    name: str = "covariance",
    check_pd: bool = True,
) -> None:
    """Validate a covariance matrix (symmetric, positive-(semi)definite).

    Parameters
    ----------
    cov : np.ndarray
        Square matrix.
    name : str
        Label for error messages.
    check_pd : bool
        If ``True``, check positive definiteness via eigenvalues.

    Raises
    ------
    TypeError, ValueError
    """
    validate_square_matrix(cov, name=name, min_size=1)
    if not np.allclose(cov, cov.T, atol=1e-10):
        max_asym = np.max(np.abs(cov - cov.T))
        raise ValueError(
            f"{name} is not symmetric (max asymmetry {max_asym:.2e})"
        )
    if check_pd:
        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals < -1e-10):
            raise ValueError(
                f"{name} is not positive semi-definite "
                f"(min eigenvalue {eigvals.min():.2e})"
            )


def validate_permutation(
    perm: Sequence[int],
    *,
    n: int,
    name: str = "permutation",
) -> None:
    """Validate that *perm* is a valid permutation of ``range(n)``.

    Parameters
    ----------
    perm : sequence of int
        Candidate permutation.
    n : int
        Expected length.
    name : str
        Label for error messages.

    Raises
    ------
    ValueError
    """
    if len(perm) != n:
        raise ValueError(f"{name} length {len(perm)} != expected {n}")
    if set(perm) != set(range(n)):
        raise ValueError(f"{name} is not a valid permutation of range({n})")


def validate_mapping(
    mapping: Dict[str, str],
    *,
    domain: Set[str],
    codomain: Set[str],
    name: str = "mapping",
    injective: bool = False,
) -> None:
    """Validate a string-to-string mapping between two variable sets.

    Parameters
    ----------
    mapping : dict
        Mapping from domain to codomain.
    domain : set of str
        Expected keys.
    codomain : set of str
        Expected values must be a subset of this.
    name : str
        Label for error messages.
    injective : bool
        If ``True``, no two keys may map to the same value.

    Raises
    ------
    ValueError
    """
    if set(mapping.keys()) != domain:
        missing = domain - set(mapping.keys())
        extra = set(mapping.keys()) - domain
        parts: List[str] = []
        if missing:
            parts.append(f"missing keys {sorted(missing)}")
        if extra:
            parts.append(f"extra keys {sorted(extra)}")
        raise ValueError(f"{name}: {'; '.join(parts)}")
    bad_vals = set(mapping.values()) - codomain
    if bad_vals:
        raise ValueError(
            f"{name} maps to values not in codomain: {sorted(bad_vals)}"
        )
    if injective:
        vals = list(mapping.values())
        if len(vals) != len(set(vals)):
            raise ValueError(f"{name} is not injective — duplicate values found")
