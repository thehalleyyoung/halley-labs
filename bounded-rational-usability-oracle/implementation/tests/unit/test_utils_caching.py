"""
Unit tests for usability_oracle.utils.caching.

Tests cover LRU eviction, TTL expiration, content-addressable deduplication,
memoised function caching, cache statistics, thread safety, and memory-limit
enforcement.
"""

from __future__ import annotations

import hashlib
import threading
import time

import numpy as np
import pytest

from usability_oracle.utils.caching import (
    CacheStats,
    ContentAddressableStore,
    CostCache,
    LRUCache,
    MemoizedFunction,
    PolicyCache,
    TreeCache,
    cache_stats,
    clear_all_caches,
)


# ===================================================================
# LRU eviction policy
# ===================================================================

class TestLRUEviction:
    """LRU eviction correctness."""

    def test_evicts_oldest_when_full(self) -> None:
        cache: LRUCache[str, int] = LRUCache(max_size=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        # "a" is LRU — should be evicted when "d" is inserted
        cache.put("d", 4)
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_access_refreshes_position(self) -> None:
        cache: LRUCache[str, int] = LRUCache(max_size=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        # Access "a" to refresh it
        cache.get("a")
        cache.put("d", 4)
        # "b" is now LRU and should be evicted
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_update_refreshes_position(self) -> None:
        cache: LRUCache[str, int] = LRUCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        # Update "a" — it should become MRU
        cache.put("a", 10)
        cache.put("c", 3)
        assert cache.get("a") == 10
        assert cache.get("b") is None

    def test_single_element_cache(self) -> None:
        cache: LRUCache[str, int] = LRUCache(max_size=1)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_len_and_contains(self) -> None:
        cache: LRUCache[str, int] = LRUCache(max_size=5)
        cache.put("x", 42)
        assert len(cache) == 1
        assert "x" in cache
        assert "y" not in cache

    def test_eviction_counter(self) -> None:
        cache: LRUCache[str, int] = LRUCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # evicts "a"
        cache.put("d", 4)  # evicts "b"
        stats = cache.stats()
        assert stats.evictions == 2


# ===================================================================
# TTL expiration
# ===================================================================

class TestTTLExpiration:
    """TTL-based cache expiry."""

    def test_entry_expires_after_ttl(self) -> None:
        cache: LRUCache[str, int] = LRUCache(max_size=10, ttl=0.05)
        cache.put("k", 99)
        assert cache.get("k") == 99
        time.sleep(0.08)
        assert cache.get("k") is None

    def test_non_expired_entry_survives(self) -> None:
        cache: LRUCache[str, int] = LRUCache(max_size=10, ttl=5.0)
        cache.put("k", 42)
        assert cache.get("k") == 42

    def test_contains_respects_ttl(self) -> None:
        cache: LRUCache[str, int] = LRUCache(max_size=10, ttl=0.05)
        cache.put("k", 1)
        assert "k" in cache
        time.sleep(0.08)
        assert "k" not in cache

    def test_ttl_eviction_counted_in_stats(self) -> None:
        cache: LRUCache[str, int] = LRUCache(max_size=10, ttl=0.05)
        cache.put("k", 1)
        time.sleep(0.08)
        cache.get("k")  # triggers eviction
        stats = cache.stats()
        assert stats.evictions >= 1
        assert stats.misses >= 1


# ===================================================================
# ContentAddressableStore deduplication
# ===================================================================

class TestContentAddressableStore:
    """Content-addressable store deduplication."""

    def test_store_and_fetch(self) -> None:
        cas = ContentAddressableStore()
        data = b"hello world"
        key = cas.store(data)
        assert cas.fetch(key) == data

    def test_deduplication(self) -> None:
        cas = ContentAddressableStore()
        data = b"duplicate content"
        key1 = cas.store(data)
        key2 = cas.store(data)
        assert key1 == key2

    def test_different_content_different_keys(self) -> None:
        cas = ContentAddressableStore()
        k1 = cas.store(b"aaa")
        k2 = cas.store(b"bbb")
        assert k1 != k2

    def test_missing_key_returns_none(self) -> None:
        cas = ContentAddressableStore()
        assert cas.fetch("nonexistent_hash") is None

    def test_contains(self) -> None:
        cas = ContentAddressableStore()
        key = cas.store(b"data")
        assert key in cas
        assert "missing" not in cas

    def test_key_is_sha256(self) -> None:
        cas = ContentAddressableStore()
        data = b"test data for hash"
        key = cas.store(data)
        expected = hashlib.sha256(data).hexdigest()
        assert key == expected


# ===================================================================
# MemoizedFunction
# ===================================================================

class TestMemoizedFunction:
    """Memoised function decorator."""

    def test_caches_result(self) -> None:
        call_count = 0

        @MemoizedFunction
        def add(x: int, y: int) -> int:
            nonlocal call_count
            call_count += 1
            return x + y

        assert add(1, 2) == 3
        assert add(1, 2) == 3
        assert call_count == 1  # second call served from cache

    def test_different_args_different_results(self) -> None:
        @MemoizedFunction
        def square(x: int) -> int:
            return x * x

        assert square(3) == 9
        assert square(4) == 16

    def test_with_numpy_args(self) -> None:
        call_count = 0

        @MemoizedFunction
        def norm(arr: np.ndarray) -> float:
            nonlocal call_count
            call_count += 1
            return float(np.linalg.norm(arr))

        a = np.array([3.0, 4.0])
        assert norm(a) == pytest.approx(5.0)
        assert norm(a) == pytest.approx(5.0)
        assert call_count == 1

    def test_cache_clear(self) -> None:
        call_count = 0

        @MemoizedFunction
        def identity(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        identity(1)
        identity.cache_clear()
        identity(1)
        assert call_count == 2

    def test_with_kwargs(self) -> None:
        @MemoizedFunction(max_size=64)
        def greet(name: str, greeting: str = "hello") -> str:
            return f"{greeting} {name}"

        assert greet("alice") == "hello alice"
        assert greet("alice", greeting="hi") == "hi alice"

    def test_cache_stats(self) -> None:
        @MemoizedFunction
        def inc(x: int) -> int:
            return x + 1

        inc(1)
        inc(1)  # hit
        inc(2)
        stats = inc.cache_stats()
        assert stats.hits >= 1
        assert stats.misses >= 2


# ===================================================================
# Domain caches (TreeCache, PolicyCache, CostCache)
# ===================================================================

class TestDomainCaches:
    """Domain-specific cache wrappers."""

    def test_tree_cache_roundtrip(self) -> None:
        tc = TreeCache(max_size=8)
        data = b"<html><body>test</body></html>"
        parsed = {"tag": "html"}
        key = tc.put(data, parsed)
        assert tc.get(data) == parsed
        assert key == hashlib.sha256(data).hexdigest()

    def test_tree_cache_miss(self) -> None:
        tc = TreeCache(max_size=8)
        assert tc.get(b"unknown") is None

    def test_policy_cache_roundtrip(self) -> None:
        pc = PolicyCache(max_size=8)
        mdp = b"mdp_data_bytes"
        beta = 1.5
        policy = np.array([0.3, 0.7])
        pc.put(mdp, beta, policy)
        result = pc.get(mdp, beta)
        np.testing.assert_array_equal(result, policy)

    def test_policy_cache_miss(self) -> None:
        pc = PolicyCache(max_size=8)
        assert pc.get(b"unknown", 1.0) is None

    def test_cost_cache_roundtrip(self) -> None:
        cc = CostCache(max_size=8)
        tree = b"tree_bytes"
        config = b"config_bytes"
        result = {"total": 1.23}
        cc.put(tree, config, result)
        assert cc.get(tree, config) == result

    def test_cost_cache_miss(self) -> None:
        cc = CostCache(max_size=8)
        assert cc.get(b"a", b"b") is None


# ===================================================================
# Cache statistics
# ===================================================================

class TestCacheStatistics:
    """Cache hit/miss/eviction statistics."""

    def test_stats_counters(self) -> None:
        cache: LRUCache[str, int] = LRUCache(max_size=2)
        cache.put("a", 1)
        cache.get("a")    # hit
        cache.get("b")    # miss
        stats = cache.stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.current_size == 1
        assert stats.max_size == 2

    def test_hit_rate(self) -> None:
        stats = CacheStats(hits=7, misses=3)
        assert stats.hit_rate == pytest.approx(0.7)

    def test_hit_rate_no_accesses(self) -> None:
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_clear_resets_stats(self) -> None:
        cache: LRUCache[str, int] = LRUCache(max_size=10)
        cache.put("a", 1)
        cache.get("a")
        cache.clear()
        stats = cache.stats()
        assert stats.hits == 0
        assert stats.current_size == 0

    def test_global_cache_stats_returns_dict(self) -> None:
        result = cache_stats()
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, CacheStats)


# ===================================================================
# Thread safety
# ===================================================================

class TestThreadSafety:
    """Thread-safe concurrent access."""

    def test_concurrent_puts(self) -> None:
        cache: LRUCache[int, int] = LRUCache(max_size=1000)
        errors: list[Exception] = []

        def worker(start: int, count: int) -> None:
            try:
                for i in range(start, start + count):
                    cache.put(i, i * 2)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i * 100, 100))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(cache) <= 1000

    def test_concurrent_get_put(self) -> None:
        cache: LRUCache[int, int] = LRUCache(max_size=100)
        for i in range(100):
            cache.put(i, i)

        errors: list[Exception] = []

        def reader() -> None:
            try:
                for i in range(100):
                    cache.get(i)
            except Exception as e:
                errors.append(e)

        def writer() -> None:
            try:
                for i in range(100, 200):
                    cache.put(i, i)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader) for _ in range(5)
        ] + [
            threading.Thread(target=writer) for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ===================================================================
# Memory limit enforcement
# ===================================================================

class TestMemoryLimit:
    """Memory-based eviction."""

    def test_memory_limit_evicts_entries(self) -> None:
        # Each numpy array of 100 float64 ≈ 800 bytes
        cache: LRUCache[str, np.ndarray] = LRUCache(
            max_size=100, max_memory_bytes=2000
        )
        cache.put("a", np.zeros(100))  # ~800 bytes
        cache.put("b", np.zeros(100))  # ~800 bytes
        cache.put("c", np.zeros(100))  # ~800 bytes; should trigger eviction
        # At least one entry should have been evicted
        stats = cache.stats()
        assert stats.evictions >= 1

    def test_no_eviction_within_budget(self) -> None:
        cache: LRUCache[str, int] = LRUCache(
            max_size=100, max_memory_bytes=100_000
        )
        for i in range(10):
            cache.put(str(i), i)
        assert len(cache) == 10
        assert cache.stats().evictions == 0


# ===================================================================
# clear_all_caches
# ===================================================================

class TestClearAllCaches:
    """Global cache reset."""

    def test_clear_all(self) -> None:
        c1: LRUCache[str, int] = LRUCache(max_size=10)
        c2: LRUCache[str, int] = LRUCache(max_size=10)
        c1.put("a", 1)
        c2.put("b", 2)
        clear_all_caches()
        assert len(c1) == 0
        assert len(c2) == 0
