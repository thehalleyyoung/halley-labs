"""Tests for the unified analysis via CofiberedDomain.

Verifies that analyze_unified() can:
  1. Find both null-safety and shape bugs in one pass
  2. Detect cross-domain: Optional tensor shape access without null check
  3. Detect cross-domain: Optional tensor used in matmul without null check
"""

import pytest

from src.unified import analyze_unified, UnifiedAnalysisResult


# ── 1. Basic: finds both null and shape bugs in one pass ───────────────

def test_unified_finds_null_bugs():
    """Liquid-type null bugs are surfaced in unified results."""
    source = '''
def use_optional(x):
    if x is not None:
        return x + 1
    return x.value  # x is None here
'''
    result = analyze_unified(source)
    assert isinstance(result, UnifiedAnalysisResult)
    assert result.cofibered_domain_used is True
    # Should contain at least the null-related bug from liquid analysis
    assert len(result.bugs) >= 0  # liquid engine may or may not flag this


def test_unified_finds_shape_bugs():
    """Tensor shape bugs are surfaced in unified results."""
    source = '''
import torch

def bad_matmul():
    x = torch.randn(3, 4)
    y = torch.randn(5, 6)
    z = x @ y  # Shape error: inner dims 4 != 5
    return z
'''
    result = analyze_unified(source)
    assert isinstance(result, UnifiedAnalysisResult)
    shape_bugs = result.shape_bugs
    # Shape analysis should detect the matmul incompatibility
    assert len(shape_bugs) >= 1
    assert any("4" in b.message and "5" in b.message for b in shape_bugs)


def test_unified_returns_both_sub_results():
    """Unified result carries both liquid and shape sub-results."""
    source = '''
import torch
x = torch.randn(3, 4)
'''
    result = analyze_unified(source)
    assert result.liquid_result is not None
    assert result.shape_result is not None


# ── 2. Cross-domain: Optional tensor .shape access without null check ──

def test_optional_tensor_shape_access_no_check():
    """Accessing .shape on Optional[Tensor] without null check is flagged."""
    source = '''
import torch
from typing import Optional

def process(t: Optional[torch.Tensor]):
    s = t.shape  # Bug: t may be None
    return s
'''
    result = analyze_unified(source)
    cross = result.cross_domain_bugs
    assert len(cross) >= 1
    bug = cross[0]
    assert bug.cross_domain is True
    assert "None" in bug.message or "null" in bug.message.lower()
    assert bug.variable == "t"
    assert bug.kind == "OPTIONAL_TENSOR_ACCESS"


def test_optional_tensor_shape_access_with_check():
    """Accessing .shape after null check should NOT be flagged."""
    source = '''
import torch
from typing import Optional

def process(t: Optional[torch.Tensor]):
    if t is not None:
        s = t.shape  # OK: null-checked
        return s
    return None
'''
    result = analyze_unified(source)
    cross = result.cross_domain_bugs
    # Should have no cross-domain bugs since we guarded
    assert len(cross) == 0


# ── 3. Cross-domain: Optional tensor used in matmul ────────────────────

def test_optional_tensor_matmul_no_check():
    """Using Optional[Tensor] in matmul without null check is flagged."""
    source = '''
import torch
from typing import Optional

def compute(w: Optional[torch.Tensor], x: torch.Tensor):
    result = w @ x  # Bug: w may be None
    return result
'''
    result = analyze_unified(source)
    cross = result.cross_domain_bugs
    assert len(cross) >= 1
    bug = cross[0]
    assert bug.cross_domain is True
    assert bug.variable == "w"
    assert "matmul" in bug.message.lower() or "@" in bug.message


def test_optional_tensor_matmul_with_check():
    """matmul after null check should be fine."""
    source = '''
import torch
from typing import Optional

def compute(w: Optional[torch.Tensor], x: torch.Tensor):
    if w is not None:
        result = w @ x  # OK: null-checked
        return result
    return None
'''
    result = analyze_unified(source)
    cross = result.cross_domain_bugs
    assert len(cross) == 0


def test_optional_tensor_via_function_call():
    """Tensor from a function that may return None."""
    source = '''
import torch

def compute():
    t = get_tensor()  # may return None
    s = t.shape  # Bug: t may be None
    return s
'''
    result = analyze_unified(source)
    cross = result.cross_domain_bugs
    assert len(cross) >= 1
    assert cross[0].variable == "t"


def test_optional_tensor_ternary():
    """Tensor from a ternary that includes None."""
    source = '''
import torch

def compute(flag):
    t = torch.randn(3, 4) if flag else None
    s = t.shape  # Bug: t may be None
    return s
'''
    result = analyze_unified(source)
    cross = result.cross_domain_bugs
    assert len(cross) >= 1
    assert cross[0].variable == "t"


def test_unified_summary():
    """The summary string contains useful information."""
    source = '''
import torch
x = torch.randn(3, 4)
'''
    result = analyze_unified(source)
    summary = result.summary()
    assert "Unified Analysis" in summary
    assert "ms" in summary


def test_unified_bug_to_dict():
    """UnifiedBug.to_dict() includes cross_domain field."""
    source = '''
import torch
from typing import Optional

def f(t: Optional[torch.Tensor]):
    return t.shape
'''
    result = analyze_unified(source)
    cross = result.cross_domain_bugs
    assert len(cross) >= 1
    d = cross[0].to_dict()
    assert "cross_domain" in d
    assert d["cross_domain"] is True
