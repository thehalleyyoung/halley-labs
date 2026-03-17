"""Statistical utilities for Pareto analysis.

Implements hypervolume indicators, generational distance, spread metrics,
Pareto front computation, crowding distance, non-dominated sorting,
summary statistics, and non-parametric testing — all using only the
standard library math module.
"""

import math


def compute_hypervolume(points: list, reference: tuple) -> float:
    """Compute the hypervolume indicator dominated by a point set.

    For 2D, uses an exact sweep-line algorithm. For 3D+, uses a
    simple inclusion-exclusion approximation.

    All objectives are assumed to be *minimized*; the reference point
    should dominate no member of the set (i.e., be worse than all).

    Args:
        points: List of objective-value tuples.
        reference: Reference (anti-ideal) point.

    Returns:
        Hypervolume value (area in 2D, volume in 3D, etc.).
    """
    if not points:
        return 0.0

    ndim = len(reference)

    if ndim == 2:
        return _hypervolume_2d(points, reference)

    return _hypervolume_nd_approx(points, reference)


def _hypervolume_2d(points: list, reference: tuple) -> float:
    """Exact 2D hypervolume via sweep line."""
    # Filter points that are dominated by the reference
    valid = [p for p in points if p[0] < reference[0] and p[1] < reference[1]]
    if not valid:
        return 0.0

    # Sort by first objective ascending
    valid.sort(key=lambda p: p[0])

    hv = 0.0
    prev_y = reference[1]
    for p in valid:
        if p[1] < prev_y:
            hv += (reference[0] - p[0]) * (prev_y - p[1])
            prev_y = p[1]

    return hv


def _hypervolume_nd_approx(points: list, reference: tuple) -> float:
    """Approximate hypervolume for n >= 3 dimensions via inclusion-exclusion.

    Uses single-point contributions only (ignores overlap corrections
    beyond pairwise) for tractability on small fronts.
    """
    ndim = len(reference)

    # Individual box volumes
    total = 0.0
    for p in points:
        vol = 1.0
        for d in range(ndim):
            extent = reference[d] - p[d]
            if extent <= 0:
                vol = 0.0
                break
            vol *= extent
        total += vol

    # Subtract pairwise overlaps
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            overlap = 1.0
            for d in range(ndim):
                extent = reference[d] - max(points[i][d], points[j][d])
                if extent <= 0:
                    overlap = 0.0
                    break
                overlap *= extent
            total -= overlap

    return max(total, 0.0)


def generational_distance(front_a: list, front_b: list) -> float:
    """Average Euclidean distance from each point in A to nearest in B.

    GD = (1/|A|) * sum of min_distance(a, B) for a in A.
    """
    if not front_a or not front_b:
        return 0.0

    total = 0.0
    for a in front_a:
        min_dist = min(euclidean_distance(a, b) for b in front_b)
        total += min_dist
    return total / len(front_a)


def inverted_generational_distance(front_a: list, front_b: list) -> float:
    """Average distance from each point in B to nearest in A.

    IGD = (1/|B|) * sum of min_distance(b, A) for b in B.
    """
    return generational_distance(front_b, front_a)


def spread_metric(front: list) -> float:
    """Measure spread/diversity of a Pareto front.

    Based on the range of consecutive distances: computes the standard
    deviation of consecutive point distances along the sorted front.
    Lower values indicate more uniform spread.
    """
    if len(front) < 2:
        return 0.0

    # Sort by first objective
    sorted_front = sorted(front, key=lambda p: p[0])

    distances = []
    for i in range(len(sorted_front) - 1):
        d = euclidean_distance(sorted_front[i], sorted_front[i + 1])
        distances.append(d)

    if not distances:
        return 0.0

    mean_d = sum(distances) / len(distances)
    if mean_d == 0:
        return 0.0

    # Spread = (d_first + d_last + sum|d_i - mean|) / (d_first + d_last + (N-1)*mean)
    d_first = distances[0]
    d_last = distances[-1]
    deviation_sum = sum(abs(d - mean_d) for d in distances)
    numerator = d_first + d_last + deviation_sum
    denominator = d_first + d_last + (len(distances)) * mean_d

    if denominator == 0:
        return 0.0

    return numerator / denominator


def spacing_metric(front: list) -> float:
    """Measure uniformity of distribution along the front.

    SP = sqrt( (1/(N-1)) * sum (d_i - d_mean)^2 )
    where d_i is the distance from point i to its nearest neighbor.
    """
    n = len(front)
    if n < 2:
        return 0.0

    # Min distance from each point to any other
    min_dists = []
    for i in range(n):
        md = float("inf")
        for j in range(n):
            if i != j:
                d = euclidean_distance(front[i], front[j])
                if d < md:
                    md = d
        min_dists.append(md)

    mean_d = sum(min_dists) / n
    variance = sum((d - mean_d) ** 2 for d in min_dists) / (n - 1)
    return math.sqrt(variance)


def epsilon_indicator(front_a: list, front_b: list) -> float:
    """Compute additive epsilon indicator.

    Minimum epsilon such that for every point b in B, there exists
    a point a in A where a[k] - epsilon <= b[k] for all objectives k.

    Assumes minimization of all objectives.
    """
    if not front_a or not front_b:
        return 0.0

    ndim = len(front_b[0])
    eps = float("-inf")

    for b in front_b:
        min_eps_for_b = float("inf")
        for a in front_a:
            max_diff = float("-inf")
            for k in range(ndim):
                diff = a[k] - b[k]
                if diff > max_diff:
                    max_diff = diff
            if max_diff < min_eps_for_b:
                min_eps_for_b = max_diff
        if min_eps_for_b > eps:
            eps = min_eps_for_b

    return eps


def dominates(a: tuple, b: tuple, minimize: list = None) -> bool:
    """Check if point a dominates point b.

    Args:
        minimize: List of bools per objective. True = minimize (default),
            False = maximize.

    Returns:
        True if a dominates b (at least as good in all objectives,
        strictly better in at least one).
    """
    ndim = len(a)
    if minimize is None:
        minimize = [True] * ndim

    at_least_as_good = True
    strictly_better = False

    for i in range(ndim):
        if minimize[i]:
            if a[i] > b[i]:
                at_least_as_good = False
                break
            if a[i] < b[i]:
                strictly_better = True
        else:
            if a[i] < b[i]:
                at_least_as_good = False
                break
            if a[i] > b[i]:
                strictly_better = True

    return at_least_as_good and strictly_better


def compute_pareto_front(points: list, minimize: list = None) -> list:
    """Return indices of non-dominated points.

    Args:
        points: List of objective-value tuples.
        minimize: Per-objective minimization flags (default all True).

    Returns:
        List of indices into `points` that form the Pareto front.
    """
    n = len(points)
    if n == 0:
        return []

    is_dominated = [False] * n
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            if dominates(points[j], points[i], minimize):
                is_dominated[i] = True
                break

    return [i for i in range(n) if not is_dominated[i]]


def crowding_distance(front: list) -> list:
    """Compute NSGA-II crowding distance for each point in the front.

    Points at the extremes of each objective get infinite distance.

    Returns:
        List of crowding distances (same length as front).
    """
    n = len(front)
    if n == 0:
        return []
    if n <= 2:
        return [float("inf")] * n

    ndim = len(front[0])
    distances = [0.0] * n

    for m in range(ndim):
        # Sort indices by objective m
        sorted_idx = sorted(range(n), key=lambda i: front[i][m])

        obj_min = front[sorted_idx[0]][m]
        obj_max = front[sorted_idx[-1]][m]
        obj_range = obj_max - obj_min

        # Boundary points get infinite distance
        distances[sorted_idx[0]] = float("inf")
        distances[sorted_idx[-1]] = float("inf")

        if obj_range == 0:
            continue

        for k in range(1, n - 1):
            idx = sorted_idx[k]
            prev_val = front[sorted_idx[k - 1]][m]
            next_val = front[sorted_idx[k + 1]][m]
            distances[idx] += (next_val - prev_val) / obj_range

    return distances


def contribution(point_index: int, front: list, reference: tuple) -> float:
    """Compute hypervolume contribution of a single point.

    Contribution = HV(front) - HV(front without point).
    """
    if not front or point_index < 0 or point_index >= len(front):
        return 0.0

    hv_full = compute_hypervolume(front, reference)
    reduced = [p for i, p in enumerate(front) if i != point_index]
    hv_reduced = compute_hypervolume(reduced, reference)
    return hv_full - hv_reduced


def euclidean_distance(a: tuple, b: tuple) -> float:
    """Compute Euclidean distance between two points."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def normalize_front(front: list) -> list:
    """Normalize each dimension of the front to [0, 1].

    If a dimension has zero range, all values map to 0.0.
    """
    if not front:
        return []

    ndim = len(front[0])
    mins = [min(p[d] for p in front) for d in range(ndim)]
    maxs = [max(p[d] for p in front) for d in range(ndim)]
    ranges = [maxs[d] - mins[d] for d in range(ndim)]

    normalized = []
    for p in front:
        norm_p = tuple(
            (p[d] - mins[d]) / ranges[d] if ranges[d] > 0 else 0.0
            for d in range(ndim)
        )
        normalized.append(norm_p)

    return normalized


def rank_fronts(points: list, minimize: list = None) -> list:
    """Non-dominated sorting: return list of fronts (lists of indices).

    First list is the Pareto front (rank 0), second is the front of
    remaining points after removing rank 0, and so on.
    """
    n = len(points)
    if n == 0:
        return []

    remaining = set(range(n))
    fronts = []

    while remaining:
        remaining_list = list(remaining)
        remaining_points = [points[i] for i in remaining_list]

        front_local = compute_pareto_front(remaining_points, minimize)
        front_global = [remaining_list[i] for i in front_local]

        if not front_global:
            # All remaining are in one front (fallback for edge cases)
            fronts.append(list(remaining))
            break

        fronts.append(front_global)
        remaining -= set(front_global)

    return fronts


def summary_statistics(values: list) -> dict:
    """Compute summary statistics for a list of numeric values.

    Returns:
        {mean, median, std, min, max, q25, q75}.
    """
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q25": 0.0,
            "q75": 0.0,
        }

    n = len(values)
    sorted_v = sorted(values)

    mean = sum(sorted_v) / n
    variance = sum((x - mean) ** 2 for x in sorted_v) / n
    std = math.sqrt(variance)

    def _percentile(data, p):
        """Linear interpolation percentile."""
        k = (len(data) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return data[int(k)]
        return data[int(f)] * (c - k) + data[int(c)] * (k - f)

    median = _percentile(sorted_v, 0.5)
    q25 = _percentile(sorted_v, 0.25)
    q75 = _percentile(sorted_v, 0.75)

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "min": sorted_v[0],
        "max": sorted_v[-1],
        "q25": q25,
        "q75": q75,
    }


def mann_whitney_u(sample_a: list, sample_b: list) -> dict:
    """Non-parametric Mann-Whitney U test.

    Computes the U statistic, z-score (normal approximation), and
    approximate two-sided p-value.

    Returns:
        {u_statistic, z_score, p_value, significant} where significant
        is True if p_value < 0.05.
    """
    n1 = len(sample_a)
    n2 = len(sample_b)

    if n1 == 0 or n2 == 0:
        return {
            "u_statistic": 0.0,
            "z_score": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    # Rank all values together
    combined = [(v, 0) for v in sample_a] + [(v, 1) for v in sample_b]
    combined.sort(key=lambda x: x[0])

    # Assign ranks with tie handling (average rank)
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # Sum of ranks for sample A
    r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)

    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u_stat = min(u1, u2)

    # Normal approximation for large samples
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)

    if sigma == 0:
        return {
            "u_statistic": u_stat,
            "z_score": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    z = (u_stat - mu) / sigma

    # Two-sided p-value from standard normal approximation
    p_value = 2.0 * _normal_cdf(-abs(z))

    return {
        "u_statistic": u_stat,
        "z_score": z,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def _normal_cdf(x: float) -> float:
    """Approximate CDF of the standard normal distribution.

    Uses the Abramowitz and Stegun approximation (formula 7.1.26).
    """
    # erfc-based: Phi(x) = 0.5 * erfc(-x / sqrt(2))
    return 0.5 * math.erfc(-x / math.sqrt(2.0))
