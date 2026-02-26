"""
Smoke tests for the benchmark runner.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase_cartographer.benchmarks.runner import (
    run_benchmark, run_ablation, run_scalability_study,
)
from phase_cartographer.models.benchmark_models import list_benchmarks, get_benchmark


class TestBenchmarks:
    def test_list_benchmarks(self):
        bms = list_benchmarks()
        assert "toggle_switch" in bms
        assert "brusselator" in bms
        assert "selkov" in bms
        assert "repressilator" in bms
        assert "goodwin" in bms

    def test_toggle_smoke(self):
        """Quick smoke test on toggle switch (small grid)."""
        result = run_benchmark("toggle_switch", max_depth=2, max_cells=10)
        assert result.model_name == "toggle_switch"
        assert result.certified_cells >= 1
        assert result.tier1_pass_rate >= 0.0
        assert result.total_time_s > 0

    def test_selkov_smoke(self):
        result = run_benchmark("selkov", max_depth=2, max_cells=10)
        assert result.model_name == "selkov"
        assert result.certified_cells >= 1

    def test_repressilator_model_loads(self):
        bm = get_benchmark("repressilator")
        assert bm.n_states == 3
        assert bm.n_params == 4
        # Widened alpha range
        assert bm.parameter_domain[0] == (3.5, 5.5)

    def test_goodwin_model_loads(self):
        bm = get_benchmark("goodwin")
        assert bm.n_states == 3
        assert bm.n_params == 2
        assert bm.parameter_domain == [(3.0, 8.0), (0.8, 1.2)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
