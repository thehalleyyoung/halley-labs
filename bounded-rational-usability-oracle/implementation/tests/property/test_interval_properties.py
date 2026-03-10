"""Property-based tests for Interval arithmetic.

This module verifies algebraic properties of interval arithmetic operations
using the Hypothesis library. The properties tested include commutativity,
associativity, identity elements, containment, monotonicity of width,
inclusion isotonicity, and negation invariants. These ensure the interval
implementation satisfies the fundamental axioms of interval analysis.
"""

import math

from hypothesis import given, assume, settings, HealthCheck
from hypothesis.strategies import floats, integers, lists, tuples, sampled_from

from usability_oracle.interval import Interval, IntervalArithmetic


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_bounded_float = floats(min_value=-1e6, max_value=1e6,
                        allow_nan=False, allow_infinity=False)


def _interval_pair():
    """Return a strategy that produces (low, high) with low <= high."""
    return tuples(_bounded_float, _bounded_float).map(
        lambda t: (min(t), max(t))
    )


def _make_interval(pair):
    """Construct an Interval from a (low, high) tuple."""
    return Interval(pair[0], pair[1])


# ---------------------------------------------------------------------------
# Commutativity
# ---------------------------------------------------------------------------


@given(_interval_pair(), _interval_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_addition_commutativity(pair_a, pair_b):
    """Interval addition is commutative: a + b == b + a.

    For any two intervals *a* and *b*, the sum a + b must equal b + a
    because interval addition is defined as [a.low+b.low, a.high+b.high]
    and real addition is commutative.
    """
    a, b = _make_interval(pair_a), _make_interval(pair_b)
    ab = a + b
    ba = b + a
    assert ab == ba


@given(_interval_pair(), _interval_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_multiplication_commutativity(pair_a, pair_b):
    """Interval multiplication is commutative: a * b == b * a.

    Because real multiplication is commutative, the set of products
    {x*y : x in a, y in b} is the same as {y*x : y in b, x in a}.
    """
    a, b = _make_interval(pair_a), _make_interval(pair_b)
    ab = a * b
    ba = b * a
    assert ab == ba


# ---------------------------------------------------------------------------
# Associativity (within floating-point tolerance)
# ---------------------------------------------------------------------------

_ATOL = 1e-6


@given(_interval_pair(), _interval_pair(), _interval_pair())
@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
def test_addition_associativity(pair_a, pair_b, pair_c):
    """(a + b) + c should equal a + (b + c) up to floating-point rounding."""
    a = _make_interval(pair_a)
    b = _make_interval(pair_b)
    c = _make_interval(pair_c)
    lhs = (a + b) + c
    rhs = a + (b + c)
    assert math.isclose(lhs.low, rhs.low, abs_tol=_ATOL)
    assert math.isclose(lhs.high, rhs.high, abs_tol=_ATOL)


# ---------------------------------------------------------------------------
# Additive identity
# ---------------------------------------------------------------------------

@given(_interval_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_additive_identity(pair_a):
    """Adding the zero interval [0,0] is an identity operation.

    For every interval a, a + Interval(0,0) == a.
    """
    a = _make_interval(pair_a)
    zero = Interval(0.0, 0.0)
    assert a + zero == a


# ---------------------------------------------------------------------------
# Multiplicative identity
# ---------------------------------------------------------------------------

@given(_interval_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_multiplicative_identity(pair_a):
    """Multiplying by [1,1] is identity: a * Interval(1,1) == a."""
    a = _make_interval(pair_a)
    one = Interval(1.0, 1.0)
    assert a * one == a


# ---------------------------------------------------------------------------
# Containment (fundamental theorem of interval arithmetic)
# ---------------------------------------------------------------------------

@given(_interval_pair(), _interval_pair(),
       floats(min_value=0.0, max_value=1.0, allow_nan=False),
       floats(min_value=0.0, max_value=1.0, allow_nan=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_addition_containment(pair_a, pair_b, ta, tb):
    """Any point sum x+y with x in a, y in b lies in a+b.

    This is the *fundamental theorem of interval arithmetic*:
    the natural interval extension encloses all point evaluations.
    """
    a, b = _make_interval(pair_a), _make_interval(pair_b)
    x = a.low + ta * (a.high - a.low)
    y = b.low + tb * (b.high - b.low)
    result = a + b
    assert result.low <= x + y + 1e-9
    assert x + y - 1e-9 <= result.high


@given(_interval_pair(), _interval_pair(),
       floats(min_value=0.0, max_value=1.0, allow_nan=False),
       floats(min_value=0.0, max_value=1.0, allow_nan=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_multiplication_containment(pair_a, pair_b, ta, tb):
    """Any point product x*y with x in a, y in b lies in a*b.

    The interval product must enclose all possible real products.
    """
    a, b = _make_interval(pair_a), _make_interval(pair_b)
    x = a.low + ta * (a.high - a.low)
    y = b.low + tb * (b.high - b.low)
    result = a * b
    assert result.low <= x * y + 1e-3
    assert x * y - 1e-3 <= result.high


@given(_interval_pair(), _interval_pair(),
       floats(min_value=0.0, max_value=1.0, allow_nan=False),
       floats(min_value=0.0, max_value=1.0, allow_nan=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_subtraction_containment(pair_a, pair_b, ta, tb):
    """Any point difference x-y with x in a, y in b lies in a-b.

    Subtraction containment guarantees safe enclosure of all differences.
    """
    a, b = _make_interval(pair_a), _make_interval(pair_b)
    x = a.low + ta * (a.high - a.low)
    y = b.low + tb * (b.high - b.low)
    result = a - b
    assert result.low <= x - y + 1e-9
    assert x - y - 1e-9 <= result.high


@given(_interval_pair(), _interval_pair(),
       floats(min_value=0.0, max_value=1.0, allow_nan=False),
       floats(min_value=0.0, max_value=1.0, allow_nan=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_division_containment(pair_a, pair_b, ta, tb):
    """Any point quotient x/y with x in a, y in b lies in a/b.

    We skip cases where b contains zero since division is undefined there.
    """
    a, b = _make_interval(pair_a), _make_interval(pair_b)
    assume(b.low > 0 or b.high < 0)  # divisor must not contain zero
    x = a.low + ta * (a.high - a.low)
    y = b.low + tb * (b.high - b.low)
    assume(abs(y) > 1e-12)
    result = a / b
    q = x / y
    tol = max(1e-3, abs(q) * 1e-9)
    assert result.low <= q + tol
    assert q - tol <= result.high


# ---------------------------------------------------------------------------
# Width monotonicity
# ---------------------------------------------------------------------------

@given(_interval_pair(), _interval_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_addition_width_subadditivity(pair_a, pair_b):
    """Width of a+b equals width(a) + width(b).

    For addition the width is exactly additive:
    width(a+b) = width(a) + width(b).
    This means width(a+b) <= width(a) + width(b) trivially holds.
    """
    a, b = _make_interval(pair_a), _make_interval(pair_b)
    result = a + b
    assert result.width <= a.width + b.width + 1e-9


@given(_interval_pair(), _interval_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_subtraction_width_subadditivity(pair_a, pair_b):
    """Width of a-b equals width(a)+width(b). Same formula as addition."""
    a, b = _make_interval(pair_a), _make_interval(pair_b)
    result = a - b
    assert result.width <= a.width + b.width + 1e-9


# ---------------------------------------------------------------------------
# Inclusion isotonicity
# ---------------------------------------------------------------------------

@given(_interval_pair(), _interval_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_inclusion_isotonicity_addition(pair_a, pair_b):
    """If a ⊆ c and b ⊆ d then a+b ⊆ c+d.

    We construct c and d by widening a and b so a ⊆ c and b ⊆ d,
    then verify the sum relation.
    """
    a = _make_interval(pair_a)
    b = _make_interval(pair_b)
    c = Interval(a.low - 1.0, a.high + 1.0)
    d = Interval(b.low - 1.0, b.high + 1.0)
    assert a.is_subset_of(c)
    assert b.is_subset_of(d)
    ab = a + b
    cd = c + d
    assert ab.is_subset_of(cd)


@given(_interval_pair(), _interval_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_inclusion_isotonicity_multiplication(pair_a, pair_b):
    """If a ⊆ c and b ⊆ d then a*b ⊆ c*d.

    Inclusion isotonicity must hold for multiplication as well.
    """
    a = _make_interval(pair_a)
    b = _make_interval(pair_b)
    c = Interval(a.low - 1.0, a.high + 1.0)
    d = Interval(b.low - 1.0, b.high + 1.0)
    ab = a * b
    cd = c * d
    assert ab.is_subset_of(cd)


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------

@given(_interval_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_negation_preserves_width(pair_a):
    """Negation preserves width: width(-a) == width(a).

    Negating an interval [l, h] gives [-h, -l] whose width is still h - l.
    """
    a = _make_interval(pair_a)
    neg_a = -a
    assert math.isclose(neg_a.width, a.width, abs_tol=1e-12)


@given(_interval_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_double_negation(pair_a):
    """Double negation is identity: -(-a) == a.

    Negation is an involution on intervals.
    """
    a = _make_interval(pair_a)
    assert -(-a) == a


@given(_interval_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_negation_flips_bounds(pair_a):
    """Negation swaps and negates bounds: (-a).low == -a.high.

    The negation of [l, h] is [-h, -l].
    """
    a = _make_interval(pair_a)
    neg = -a
    assert math.isclose(neg.low, -a.high, abs_tol=1e-12)
    assert math.isclose(neg.high, -a.low, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Point interval (degenerate)
# ---------------------------------------------------------------------------

@given(_bounded_float)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_degenerate_interval_has_zero_width(val):
    """A point interval [v,v] has zero width."""
    iv = Interval(val, val)
    assert iv.width == 0.0
    assert iv.is_degenerate


@given(_bounded_float)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_from_value_creates_degenerate(v1):
    """Interval.from_value(x) gives [x, x] with zero width."""
    iv = Interval.from_value(v1)
    assert iv.low == v1
    assert iv.high == v1
    assert iv.is_degenerate


# ---------------------------------------------------------------------------
# Midpoint and containment
# ---------------------------------------------------------------------------

@given(_interval_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_midpoint_contained(pair_a):
    """The midpoint of an interval is always contained in the interval."""
    a = _make_interval(pair_a)
    assert a.contains(a.midpoint)


@given(_interval_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_bounds_contained(pair_a):
    """Both endpoints of an interval are contained in it (closed interval)."""
    a = _make_interval(pair_a)
    assert a.contains(a.low)
    assert a.contains(a.high)


# ---------------------------------------------------------------------------
# Subset reflexivity
# ---------------------------------------------------------------------------

@given(_interval_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_subset_reflexivity(pair_a):
    """Every interval is a subset of itself."""
    a = _make_interval(pair_a)
    assert a.is_subset_of(a)


# ---------------------------------------------------------------------------
# IntervalArithmetic aggregate operations
# ---------------------------------------------------------------------------

@given(lists(_interval_pair(), min_size=1, max_size=10))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_sum_intervals_contains_sum_of_midpoints(pairs):
    """The sum of interval midpoints is contained in sum_intervals.

    Since each midpoint is in its interval, the fundamental theorem
    guarantees their sum is in the sum-of-intervals.
    """
    intervals = [_make_interval(p) for p in pairs]
    result = IntervalArithmetic.sum_intervals(intervals)
    total = sum(iv.midpoint for iv in intervals)
    assert result.low <= total + 1e-6
    assert total - 1e-6 <= result.high


@given(lists(_interval_pair(), min_size=1, max_size=10))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_sum_intervals_width(pairs):
    """Width of sum_intervals equals the sum of widths.

    For interval addition the width relation is exact:
    width(Σ aᵢ) = Σ width(aᵢ).
    """
    intervals = [_make_interval(p) for p in pairs]
    result = IntervalArithmetic.sum_intervals(intervals)
    expected_width = sum(iv.width for iv in intervals)
    assert math.isclose(result.width, expected_width, rel_tol=1e-6, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# from_center_radius
# ---------------------------------------------------------------------------

@given(floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False),
       floats(min_value=0.0, max_value=1e5, allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_from_center_radius_symmetry(center, radius):
    """from_center_radius(c, r) produces an interval symmetric about c.

    The interval [c-r, c+r] has midpoint c and width 2r.
    """
    iv = Interval.from_center_radius(center, radius)
    assert math.isclose(iv.midpoint, center, abs_tol=1e-9)
    assert math.isclose(iv.width, 2.0 * radius, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# min / max intervals
# ---------------------------------------------------------------------------

@given(_interval_pair(), _interval_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_min_interval_lower_bound(pair_a, pair_b):
    """The lower bound of min(a,b) is min(a.low, b.low).

    IntervalArithmetic.min_interval must have its low bound at the overall
    minimum of all possible element pairs.
    """
    a, b = _make_interval(pair_a), _make_interval(pair_b)
    result = IntervalArithmetic.min_interval(a, b)
    assert result.low <= min(a.low, b.low) + 1e-9


@given(_interval_pair(), _interval_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_max_interval_upper_bound(pair_a, pair_b):
    """The upper bound of max(a,b) is max(a.high, b.high).

    IntervalArithmetic.max_interval must have its high bound at the overall
    maximum of all possible element pairs.
    """
    a, b = _make_interval(pair_a), _make_interval(pair_b)
    result = IntervalArithmetic.max_interval(a, b)
    assert result.high >= max(a.high, b.high) - 1e-9


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

@given(_interval_pair(), integers(min_value=0, max_value=5))
@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
def test_power_containment(pair_a, n):
    """x^n lies in a^n for any x in a.

    The interval power must enclose all possible point evaluations.
    """
    a = _make_interval(pair_a)
    assume(abs(a.low) < 1e3 and abs(a.high) < 1e3)
    result = a ** n
    x = a.midpoint
    assert result.low <= x ** n + 1e-3
    assert x ** n - 1e-3 <= result.high


# ---------------------------------------------------------------------------
# Overlap and union
# ---------------------------------------------------------------------------

@given(_interval_pair(), _interval_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_union_contains_both(pair_a, pair_b):
    """The union of two intervals contains both operands.

    a.union(b) is the smallest enclosing interval, so a ⊆ a∪b and b ⊆ a∪b.
    """
    a, b = _make_interval(pair_a), _make_interval(pair_b)
    u = a.union(b)
    assert a.is_subset_of(u)
    assert b.is_subset_of(u)


@given(_interval_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_self_overlap(pair_a):
    """Every non-degenerate interval overlaps with itself."""
    a = _make_interval(pair_a)
    assert a.overlaps(a)


# ---------------------------------------------------------------------------
# Affine interval contains standard interval
# ---------------------------------------------------------------------------

from usability_oracle.interval import (
    AffineForm,
    IEEEInterval,
    ieee_add,
    ieee_hull,
    ieee_intersection,
    affine_to_interval,
    affine_from_interval,
    ForwardBackwardContractor,
    ContractionStatus,
)


@given(_interval_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_affine_interval_contains_standard(pair_a):
    """Affine form interval ⊇ standard interval (conservative enclosure).

    Converting Interval → AffineForm → Interval should produce an interval
    that encloses the original.
    """
    a = _make_interval(pair_a)
    af = affine_from_interval(a)
    roundtrip = affine_to_interval(af)
    assert roundtrip.low <= a.low + 1e-9, \
        f"Affine lower {roundtrip.low} > standard lower {a.low}"
    assert roundtrip.high >= a.high - 1e-9, \
        f"Affine upper {roundtrip.high} < standard upper {a.high}"


# ---------------------------------------------------------------------------
# IEEE interval addition is monotone (inclusion isotone)
# ---------------------------------------------------------------------------

_ieee_bounded = floats(min_value=-1e4, max_value=1e4,
                       allow_nan=False, allow_infinity=False)


def _ieee_pair():
    """Strategy for (low, high) pairs for IEEEInterval."""
    return tuples(_ieee_bounded, _ieee_bounded).map(
        lambda t: (min(t), max(t))
    )


@given(_ieee_pair(), _ieee_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_ieee_add_monotone(pair_a, pair_b):
    """If a ⊆ c and b ⊆ d then ieee_add(a,b) ⊆ ieee_add(c,d).

    Construct c = widen(a) and d = widen(b) to guarantee containment.
    """
    a = IEEEInterval(pair_a[0], pair_a[1])
    b = IEEEInterval(pair_b[0], pair_b[1])
    c = IEEEInterval(pair_a[0] - 1.0, pair_a[1] + 1.0)
    d = IEEEInterval(pair_b[0] - 1.0, pair_b[1] + 1.0)
    ab = ieee_add(a, b)
    cd = ieee_add(c, d)
    assert ab.low >= cd.low - 1e-6, \
        f"ieee_add not monotone (lower): {ab.low} < {cd.low}"
    assert ab.high <= cd.high + 1e-6, \
        f"ieee_add not monotone (upper): {ab.high} > {cd.high}"


# ---------------------------------------------------------------------------
# Hull is idempotent (associative): hull(hull(a,b), c) = hull(a, hull(b,c))
# ---------------------------------------------------------------------------

@given(_ieee_pair(), _ieee_pair(), _ieee_pair())
@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
def test_hull_associative(pair_a, pair_b, pair_c):
    """hull(hull(a,b), c) = hull(a, hull(b,c)) — hull is associative."""
    a = IEEEInterval(pair_a[0], pair_a[1])
    b = IEEEInterval(pair_b[0], pair_b[1])
    c = IEEEInterval(pair_c[0], pair_c[1])
    lhs = ieee_hull(ieee_hull(a, b), c)
    rhs = ieee_hull(a, ieee_hull(b, c))
    assert math.isclose(lhs.low, rhs.low, abs_tol=1e-9), \
        f"Hull not associative (low): {lhs.low} vs {rhs.low}"
    assert math.isclose(lhs.high, rhs.high, abs_tol=1e-9), \
        f"Hull not associative (high): {lhs.high} vs {rhs.high}"


@given(_ieee_pair(), _ieee_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_hull_idempotent(pair_a, pair_b):
    """hull(hull(a,b), hull(a,b)) = hull(a,b) — hull is idempotent."""
    a = IEEEInterval(pair_a[0], pair_a[1])
    b = IEEEInterval(pair_b[0], pair_b[1])
    h = ieee_hull(a, b)
    hh = ieee_hull(h, h)
    assert math.isclose(h.low, hh.low, abs_tol=1e-9)
    assert math.isclose(h.high, hh.high, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Intersection is commutative
# ---------------------------------------------------------------------------


@given(_ieee_pair(), _ieee_pair())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_intersection_commutative(pair_a, pair_b):
    """intersection(a, b) = intersection(b, a) — commutativity."""
    a = IEEEInterval(pair_a[0], pair_a[1])
    b = IEEEInterval(pair_b[0], pair_b[1])
    ab = ieee_intersection(a, b)
    ba = ieee_intersection(b, a)
    if ab.empty and ba.empty:
        return  # both empty is consistent
    assert math.isclose(ab.low, ba.low, abs_tol=1e-9), \
        f"Intersection not commutative (low): {ab.low} vs {ba.low}"
    assert math.isclose(ab.high, ba.high, abs_tol=1e-9), \
        f"Intersection not commutative (high): {ab.high} vs {ba.high}"


# ---------------------------------------------------------------------------
# Contractor always produces subset of input box
# ---------------------------------------------------------------------------


@given(floats(min_value=0.0, max_value=100.0,
              allow_nan=False, allow_infinity=False),
       floats(min_value=1.0, max_value=50.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_contractor_produces_subset(center, width):
    """A contractor's output box ⊆ input box."""
    lo = center - width
    hi = center + width
    box = {'x': Interval(lo, hi)}
    target_lo = center - width / 4
    target_hi = center + width / 4
    contractor = ForwardBackwardContractor(
        f=lambda x: x,
        variables=['x'],
        target=Interval(target_lo, target_hi),
    )
    result = contractor.contract(box)
    out = result.box['x']
    assert out.low >= lo - 1e-6, \
        f"Contractor expanded lower: {out.low} < {lo}"
    assert out.high <= hi + 1e-6, \
        f"Contractor expanded upper: {out.high} > {hi}"
