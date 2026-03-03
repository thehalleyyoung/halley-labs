"""Tests for caching utilities.

Covers LRUCache, DiskCache, memoize decorator, and cache key helpers.
"""

from __future__ import annotations

import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

from cpa.utils.caching import (
    LRUCache,
    DiskCache,
    memoize,
    array_cache_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache():
    return LRUCache(maxsize=10)


@pytest.fixture
def disk_cache_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def disk_cache(disk_cache_dir):
    return DiskCache(directory=disk_cache_dir, max_entries=50)


# ---------------------------------------------------------------------------
# Test LRUCache
# ---------------------------------------------------------------------------

class TestLRUCache:

    def test_put_and_get(self, cache):
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing(self, cache):
        assert cache.get("missing") is None

    def test_get_default(self, cache):
        assert cache.get("missing", default="default") == "default"

    def test_len(self, cache):
        cache.put("a", 1)
        cache.put("b", 2)
        assert len(cache) == 2

    def test_contains(self, cache):
        cache.put("key", "val")
        assert "key" in cache
        assert "other" not in cache

    def test_eviction(self):
        small_cache = LRUCache(maxsize=3)
        small_cache.put("a", 1)
        small_cache.put("b", 2)
        small_cache.put("c", 3)
        small_cache.put("d", 4)  # Should evict "a"
        assert "a" not in small_cache
        assert "d" in small_cache
        assert len(small_cache) == 3

    def test_lru_order(self):
        """Most recently used should survive eviction."""
        c = LRUCache(maxsize=3)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        c.get("a")  # Access "a" — now most recent
        c.put("d", 4)  # Should evict "b" (least recently used)
        assert "a" in c
        assert "b" not in c

    def test_invalidate(self, cache):
        cache.put("key", "val")
        assert cache.invalidate("key")
        assert "key" not in cache

    def test_invalidate_missing(self, cache):
        assert not cache.invalidate("missing")

    def test_clear(self, cache):
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert len(cache) == 0

    def test_keys(self, cache):
        cache.put("a", 1)
        cache.put("b", 2)
        keys = cache.keys()
        assert set(keys) == {"a", "b"}

    def test_maxsize_property(self, cache):
        assert cache.maxsize == 10

    def test_hit_rate(self, cache):
        cache.put("key", "val")
        cache.get("key")  # hit
        cache.get("missing")  # miss
        rate = cache.hit_rate
        assert 0.0 <= rate <= 1.0

    def test_hits_misses(self, cache):
        cache.put("key", "val")
        cache.get("key")
        cache.get("missing")
        assert cache.hits >= 1
        assert cache.misses >= 1

    def test_overwrite(self, cache):
        cache.put("key", "old")
        cache.put("key", "new")
        assert cache.get("key") == "new"
        assert len(cache) == 1

    def test_various_types(self, cache):
        cache.put("int", 42)
        cache.put("float", 3.14)
        cache.put("list", [1, 2, 3])
        cache.put("dict", {"a": 1})
        cache.put("array", np.array([1, 2, 3]))
        assert cache.get("int") == 42
        assert cache.get("float") == 3.14


# ---------------------------------------------------------------------------
# Test DiskCache
# ---------------------------------------------------------------------------

class TestDiskCache:

    def test_put_and_get(self, disk_cache):
        disk_cache.put("key1", {"data": [1, 2, 3]})
        result = disk_cache.get("key1")
        assert result == {"data": [1, 2, 3]}

    def test_get_missing(self, disk_cache):
        assert disk_cache.get("missing") is None

    def test_get_default(self, disk_cache):
        assert disk_cache.get("missing", default="def") == "def"

    def test_invalidate(self, disk_cache):
        disk_cache.put("key", "val")
        assert disk_cache.invalidate("key")
        assert disk_cache.get("key") is None

    def test_clear(self, disk_cache):
        disk_cache.put("a", 1)
        disk_cache.put("b", 2)
        disk_cache.clear()
        assert disk_cache.size == 0

    def test_size(self, disk_cache):
        disk_cache.put("a", 1)
        disk_cache.put("b", 2)
        assert disk_cache.size == 2

    def test_total_bytes(self, disk_cache):
        disk_cache.put("key", "some data")
        assert disk_cache.total_bytes > 0

    def test_numpy_array_cache(self, disk_cache):
        arr = np.array([1.0, 2.0, 3.0])
        disk_cache.put("arr", arr.tolist())
        result = disk_cache.get("arr")
        assert result == [1.0, 2.0, 3.0]

    def test_persistence(self, disk_cache_dir):
        cache1 = DiskCache(directory=disk_cache_dir, max_entries=50)
        cache1.put("persistent", "value")
        cache2 = DiskCache(directory=disk_cache_dir, max_entries=50)
        result = cache2.get("persistent")
        assert result == "value"


# ---------------------------------------------------------------------------
# Test memoize decorator
# ---------------------------------------------------------------------------

class TestMemoize:

    def test_basic_memoize(self):
        call_count = [0]

        @memoize()
        def expensive(x, y):
            call_count[0] += 1
            return x + y

        assert expensive(1, 2) == 3
        assert expensive(1, 2) == 3  # cached
        assert call_count[0] == 1

    def test_different_args(self):
        @memoize()
        def add(x, y):
            return x + y

        assert add(1, 2) == 3
        assert add(3, 4) == 7

    def test_custom_cache(self):
        c = LRUCache(maxsize=5)

        @memoize(cache=c)
        def mul(x, y):
            return x * y

        assert mul(2, 3) == 6
        assert len(c) == 1

    def test_key_prefix(self):
        @memoize(key_prefix="my_func")
        def f(x):
            return x * 2

        assert f(5) == 10


# ---------------------------------------------------------------------------
# Test array_cache_key
# ---------------------------------------------------------------------------

class TestArrayCacheKey:

    def test_deterministic(self):
        arr = np.array([1.0, 2.0, 3.0])
        k1 = array_cache_key(arr)
        k2 = array_cache_key(arr)
        assert k1 == k2

    def test_different_arrays(self):
        k1 = array_cache_key(np.array([1.0, 2.0]))
        k2 = array_cache_key(np.array([3.0, 4.0]))
        assert k1 != k2

    def test_same_values_same_key(self):
        a1 = np.array([1.0, 2.0, 3.0])
        a2 = np.array([1.0, 2.0, 3.0])
        assert array_cache_key(a1) == array_cache_key(a2)

    def test_returns_string(self):
        k = array_cache_key(np.zeros(10))
        assert isinstance(k, str)
