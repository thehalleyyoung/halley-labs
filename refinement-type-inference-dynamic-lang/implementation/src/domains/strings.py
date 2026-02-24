from __future__ import annotations

"""
String Abstract Domain Implementation
======================================
Full implementation of the string abstract domain for refinement type inference
in dynamically-typed languages. Tracks finite sets of possible string values
for variables used as dictionary keys or attribute names. Enables reasoning
about hasattr/key-in-dict patterns.

Uses counterexample-guided contract discovery (CEGAR-style) compatible interfaces.
"""

import copy
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# StringConstant
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StringConstant:
    """Represents a known string constant."""
    value: str

    def __str__(self) -> str:
        return repr(self.value)

    def __len__(self) -> int:
        return len(self.value)

    def chars(self) -> FrozenSet[str]:
        return frozenset(self.value)

    def prefix(self, n: int) -> str:
        return self.value[:n]

    def suffix(self, n: int) -> str:
        return self.value[-n:] if n > 0 else ""

    def startswith(self, prefix: str) -> bool:
        return self.value.startswith(prefix)

    def endswith(self, suffix: str) -> bool:
        return self.value.endswith(suffix)

    def contains(self, sub: str) -> bool:
        return sub in self.value


# ---------------------------------------------------------------------------
# StringSet
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StringSet:
    """Finite set of possible string values with a top element.
    When is_top is True, the set represents all possible strings.
    When strings is empty and is_top is False, it represents bottom (empty set)."""
    strings: FrozenSet[str]
    is_top: bool = False

    @staticmethod
    def bottom() -> StringSet:
        return StringSet(frozenset(), False)

    @staticmethod
    def top_value() -> StringSet:
        return StringSet(frozenset(), True)

    @staticmethod
    def singleton(s: str) -> StringSet:
        return StringSet(frozenset({s}), False)

    @staticmethod
    def from_set(s: Set[str]) -> StringSet:
        return StringSet(frozenset(s), False)

    @staticmethod
    def from_list(lst: List[str]) -> StringSet:
        return StringSet(frozenset(lst), False)

    def is_bottom(self) -> bool:
        return not self.is_top and len(self.strings) == 0

    def contains(self, s: str) -> bool:
        if self.is_top:
            return True
        return s in self.strings

    def size(self) -> int:
        if self.is_top:
            return -1  # infinite
        return len(self.strings)

    def join(self, other: StringSet) -> StringSet:
        if self.is_top or other.is_top:
            return StringSet.top_value()
        if self.is_bottom():
            return other
        if other.is_bottom():
            return self
        return StringSet(self.strings | other.strings)

    def meet(self, other: StringSet) -> StringSet:
        if self.is_top:
            return other
        if other.is_top:
            return self
        return StringSet(self.strings & other.strings)

    def widen(self, other: StringSet, threshold: int = 32) -> StringSet:
        """Widen to top when set exceeds threshold."""
        joined = self.join(other)
        if joined.is_top:
            return joined
        if len(joined.strings) > threshold:
            return StringSet.top_value()
        return joined

    def narrow(self, other: StringSet) -> StringSet:
        if self.is_top:
            return other
        return self

    def leq(self, other: StringSet) -> bool:
        if other.is_top:
            return True
        if self.is_top:
            return False
        return self.strings <= other.strings

    def subtract(self, value: str) -> StringSet:
        if self.is_top:
            return self  # Cannot remove from top
        return StringSet(self.strings - {value})

    def add(self, value: str) -> StringSet:
        if self.is_top:
            return self
        return StringSet(self.strings | {value})

    def map(self, fn: Callable[[str], str]) -> StringSet:
        if self.is_top:
            return StringSet.top_value()
        if self.is_bottom():
            return self
        return StringSet(frozenset(fn(s) for s in self.strings))

    def filter(self, pred: Callable[[str], bool]) -> StringSet:
        if self.is_top:
            return StringSet.top_value()
        return StringSet(frozenset(s for s in self.strings if pred(s)))

    def __iter__(self) -> Iterator[str]:
        return iter(self.strings)

    def __str__(self) -> str:
        if self.is_top:
            return "⊤"
        if self.is_bottom():
            return "⊥"
        sorted_strs = sorted(self.strings)
        if len(sorted_strs) <= 5:
            return "{" + ", ".join(repr(s) for s in sorted_strs) + "}"
        return "{" + ", ".join(repr(s) for s in sorted_strs[:5]) + ", ...}"


# ---------------------------------------------------------------------------
# StringPrefix
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StringPrefix:
    """Prefix abstraction for strings.
    prefix: the known prefix, or None for no information (top).
    is_bottom: True if this represents the empty set."""
    prefix: Optional[str]
    _is_bottom: bool = False

    @staticmethod
    def bottom() -> StringPrefix:
        return StringPrefix(None, True)

    @staticmethod
    def top() -> StringPrefix:
        return StringPrefix(None, False)

    @staticmethod
    def from_string(s: str) -> StringPrefix:
        return StringPrefix(s)

    def is_bottom(self) -> bool:
        return self._is_bottom

    def is_top(self) -> bool:
        return self.prefix is None and not self._is_bottom

    def length(self) -> int:
        return len(self.prefix) if self.prefix is not None else 0

    def matches(self, s: str) -> bool:
        if self._is_bottom:
            return False
        if self.prefix is None:
            return True
        return s.startswith(self.prefix)

    def join(self, other: StringPrefix) -> StringPrefix:
        if self._is_bottom:
            return other
        if other._is_bottom:
            return self
        if self.prefix is None or other.prefix is None:
            return StringPrefix.top()
        # Find common prefix
        common = _common_prefix(self.prefix, other.prefix)
        if not common:
            return StringPrefix.top()
        return StringPrefix(common)

    def meet(self, other: StringPrefix) -> StringPrefix:
        if self._is_bottom or other._is_bottom:
            return StringPrefix.bottom()
        if self.prefix is None:
            return other
        if other.prefix is None:
            return self
        # One must be prefix of the other, or bottom
        if self.prefix.startswith(other.prefix):
            return self
        if other.prefix.startswith(self.prefix):
            return other
        return StringPrefix.bottom()

    def widen(self, other: StringPrefix) -> StringPrefix:
        joined = self.join(other)
        # Widen: if prefix gets shorter, go to top
        if joined.is_top():
            return joined
        if self.prefix is not None and joined.prefix is not None:
            if len(joined.prefix) < len(self.prefix):
                return StringPrefix.top()
        return joined

    def leq(self, other: StringPrefix) -> bool:
        if self._is_bottom:
            return True
        if other.prefix is None:
            return True
        if self.prefix is None:
            return False
        return self.prefix.startswith(other.prefix)

    def __str__(self) -> str:
        if self._is_bottom:
            return "⊥"
        if self.prefix is None:
            return "⊤"
        return f"prefix={self.prefix!r}"


def _common_prefix(a: str, b: str) -> str:
    """Find the longest common prefix of two strings."""
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return a[:i]


def _common_suffix(a: str, b: str) -> str:
    """Find the longest common suffix of two strings."""
    i = 0
    while i < len(a) and i < len(b) and a[-(i + 1)] == b[-(i + 1)]:
        i += 1
    return a[len(a) - i:] if i > 0 else ""


# ---------------------------------------------------------------------------
# StringSuffix
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StringSuffix:
    """Suffix abstraction for strings."""
    suffix: Optional[str]
    _is_bottom: bool = False

    @staticmethod
    def bottom() -> StringSuffix:
        return StringSuffix(None, True)

    @staticmethod
    def top() -> StringSuffix:
        return StringSuffix(None, False)

    @staticmethod
    def from_string(s: str) -> StringSuffix:
        return StringSuffix(s)

    def is_bottom(self) -> bool:
        return self._is_bottom

    def is_top(self) -> bool:
        return self.suffix is None and not self._is_bottom

    def matches(self, s: str) -> bool:
        if self._is_bottom:
            return False
        if self.suffix is None:
            return True
        return s.endswith(self.suffix)

    def join(self, other: StringSuffix) -> StringSuffix:
        if self._is_bottom:
            return other
        if other._is_bottom:
            return self
        if self.suffix is None or other.suffix is None:
            return StringSuffix.top()
        common = _common_suffix(self.suffix, other.suffix)
        if not common:
            return StringSuffix.top()
        return StringSuffix(common)

    def meet(self, other: StringSuffix) -> StringSuffix:
        if self._is_bottom or other._is_bottom:
            return StringSuffix.bottom()
        if self.suffix is None:
            return other
        if other.suffix is None:
            return self
        if self.suffix.endswith(other.suffix):
            return self
        if other.suffix.endswith(self.suffix):
            return other
        return StringSuffix.bottom()

    def widen(self, other: StringSuffix) -> StringSuffix:
        joined = self.join(other)
        if joined.is_top():
            return joined
        if self.suffix is not None and joined.suffix is not None:
            if len(joined.suffix) < len(self.suffix):
                return StringSuffix.top()
        return joined

    def leq(self, other: StringSuffix) -> bool:
        if self._is_bottom:
            return True
        if other.suffix is None:
            return True
        if self.suffix is None:
            return False
        return self.suffix.endswith(other.suffix)

    def __str__(self) -> str:
        if self._is_bottom:
            return "⊥"
        if self.suffix is None:
            return "⊤"
        return f"suffix={self.suffix!r}"


# ---------------------------------------------------------------------------
# StringLength
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StringLength:
    """Length abstraction: interval of possible lengths [min_len, max_len]."""
    min_len: int  # -1 for bottom
    max_len: int  # -1 for infinity (top), -2 for bottom

    @staticmethod
    def bottom() -> StringLength:
        return StringLength(-1, -2)

    @staticmethod
    def top() -> StringLength:
        return StringLength(0, -1)

    @staticmethod
    def exact(n: int) -> StringLength:
        return StringLength(n, n)

    @staticmethod
    def at_least(n: int) -> StringLength:
        return StringLength(n, -1)

    @staticmethod
    def range(lo: int, hi: int) -> StringLength:
        return StringLength(lo, hi)

    def is_bottom(self) -> bool:
        return self.max_len == -2

    def is_top(self) -> bool:
        return self.min_len == 0 and self.max_len == -1

    def is_exact(self) -> bool:
        return self.min_len == self.max_len and self.min_len >= 0

    def contains(self, n: int) -> bool:
        if self.is_bottom():
            return False
        if n < self.min_len:
            return False
        if self.max_len >= 0 and n > self.max_len:
            return False
        return True

    def join(self, other: StringLength) -> StringLength:
        if self.is_bottom():
            return other
        if other.is_bottom():
            return self
        lo = min(self.min_len, other.min_len)
        if self.max_len == -1 or other.max_len == -1:
            hi = -1
        else:
            hi = max(self.max_len, other.max_len)
        return StringLength(lo, hi)

    def meet(self, other: StringLength) -> StringLength:
        if self.is_bottom() or other.is_bottom():
            return StringLength.bottom()
        lo = max(self.min_len, other.min_len)
        if self.max_len == -1:
            hi = other.max_len
        elif other.max_len == -1:
            hi = self.max_len
        else:
            hi = min(self.max_len, other.max_len)
        if hi >= 0 and lo > hi:
            return StringLength.bottom()
        return StringLength(lo, hi)

    def widen(self, other: StringLength) -> StringLength:
        if self.is_bottom():
            return other
        if other.is_bottom():
            return self
        lo = self.min_len if other.min_len >= self.min_len else 0
        if self.max_len == -1:
            hi = -1
        elif other.max_len == -1 or other.max_len > self.max_len:
            hi = -1
        else:
            hi = self.max_len
        return StringLength(lo, hi)

    def narrow(self, other: StringLength) -> StringLength:
        if self.is_bottom():
            return StringLength.bottom()
        if other.is_bottom():
            return StringLength.bottom()
        lo = other.min_len if self.min_len == 0 else self.min_len
        if self.max_len == -1:
            hi = other.max_len
        else:
            hi = self.max_len
        if hi >= 0 and lo > hi:
            return StringLength.bottom()
        return StringLength(lo, hi)

    def add(self, other: StringLength) -> StringLength:
        """Length of concatenation."""
        if self.is_bottom() or other.is_bottom():
            return StringLength.bottom()
        lo = self.min_len + other.min_len
        if self.max_len == -1 or other.max_len == -1:
            hi = -1
        else:
            hi = self.max_len + other.max_len
        return StringLength(lo, hi)

    def leq(self, other: StringLength) -> bool:
        if self.is_bottom():
            return True
        if other.is_bottom():
            return False
        if self.min_len < other.min_len:
            return False
        if other.max_len == -1:
            return True
        if self.max_len == -1:
            return False
        return self.max_len <= other.max_len

    def __str__(self) -> str:
        if self.is_bottom():
            return "⊥"
        hi_str = "∞" if self.max_len == -1 else str(self.max_len)
        return f"len∈[{self.min_len}, {hi_str}]"


# ---------------------------------------------------------------------------
# StringCharSet
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StringCharSet:
    """Character set abstraction: possible characters at each position.
    positions maps position index -> set of possible chars.
    all_chars is the union of all possible characters."""
    all_chars: FrozenSet[str]
    positions: Tuple[FrozenSet[str], ...]
    _is_top: bool = False
    _is_bottom: bool = False

    @staticmethod
    def bottom() -> StringCharSet:
        return StringCharSet(frozenset(), (), False, True)

    @staticmethod
    def top() -> StringCharSet:
        return StringCharSet(frozenset(), (), True, False)

    @staticmethod
    def from_string(s: str) -> StringCharSet:
        chars = frozenset(s)
        positions = tuple(frozenset({c}) for c in s)
        return StringCharSet(chars, positions)

    @staticmethod
    def from_strings(strings: Set[str]) -> StringCharSet:
        if not strings:
            return StringCharSet.bottom()
        all_chars: Set[str] = set()
        max_len = max(len(s) for s in strings)
        positions: List[Set[str]] = [set() for _ in range(max_len)]
        for s in strings:
            all_chars.update(s)
            for i, c in enumerate(s):
                positions[i].add(c)
        return StringCharSet(
            frozenset(all_chars),
            tuple(frozenset(p) for p in positions),
        )

    def is_bottom(self) -> bool:
        return self._is_bottom

    def is_top(self) -> bool:
        return self._is_top

    def join(self, other: StringCharSet) -> StringCharSet:
        if self._is_bottom:
            return other
        if other._is_bottom:
            return self
        if self._is_top or other._is_top:
            return StringCharSet.top()
        chars = self.all_chars | other.all_chars
        max_len = max(len(self.positions), len(other.positions))
        positions: List[FrozenSet[str]] = []
        for i in range(max_len):
            s1 = self.positions[i] if i < len(self.positions) else frozenset()
            s2 = other.positions[i] if i < len(other.positions) else frozenset()
            positions.append(s1 | s2)
        return StringCharSet(chars, tuple(positions))

    def meet(self, other: StringCharSet) -> StringCharSet:
        if self._is_bottom or other._is_bottom:
            return StringCharSet.bottom()
        if self._is_top:
            return other
        if other._is_top:
            return self
        chars = self.all_chars & other.all_chars
        min_len = min(len(self.positions), len(other.positions))
        positions: List[FrozenSet[str]] = []
        for i in range(min_len):
            p = self.positions[i] & other.positions[i]
            if not p:
                return StringCharSet.bottom()
            positions.append(p)
        return StringCharSet(chars, tuple(positions))

    def leq(self, other: StringCharSet) -> bool:
        if self._is_bottom:
            return True
        if other._is_top:
            return True
        if self._is_top:
            return False
        if not self.all_chars <= other.all_chars:
            return False
        return True

    def __str__(self) -> str:
        if self._is_bottom:
            return "⊥"
        if self._is_top:
            return "⊤"
        return f"chars={''.join(sorted(self.all_chars))}"


# ---------------------------------------------------------------------------
# StringPattern
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StringPattern:
    """Regex pattern abstraction. Tracks whether strings match a known pattern."""
    pattern: Optional[str]  # regex pattern string, None for top
    _is_bottom: bool = False

    @staticmethod
    def bottom() -> StringPattern:
        return StringPattern(None, True)

    @staticmethod
    def top() -> StringPattern:
        return StringPattern(None, False)

    @staticmethod
    def from_pattern(pat: str) -> StringPattern:
        return StringPattern(pat)

    @staticmethod
    def literal(s: str) -> StringPattern:
        return StringPattern(re.escape(s))

    def is_bottom(self) -> bool:
        return self._is_bottom

    def is_top(self) -> bool:
        return self.pattern is None and not self._is_bottom

    def matches(self, s: str) -> bool:
        if self._is_bottom:
            return False
        if self.pattern is None:
            return True
        try:
            return bool(re.fullmatch(self.pattern, s))
        except re.error:
            return True  # conservative

    def join(self, other: StringPattern) -> StringPattern:
        if self._is_bottom:
            return other
        if other._is_bottom:
            return self
        if self.pattern is None or other.pattern is None:
            return StringPattern.top()
        if self.pattern == other.pattern:
            return self
        # Cannot precisely join two different patterns, go to top
        return StringPattern.top()

    def meet(self, other: StringPattern) -> StringPattern:
        if self._is_bottom or other._is_bottom:
            return StringPattern.bottom()
        if self.pattern is None:
            return other
        if other.pattern is None:
            return self
        if self.pattern == other.pattern:
            return self
        # Intersection of patterns is hard; return self conservatively
        return self

    def leq(self, other: StringPattern) -> bool:
        if self._is_bottom:
            return True
        if other.is_top():
            return True
        if self.is_top():
            return False
        return self.pattern == other.pattern

    def __str__(self) -> str:
        if self._is_bottom:
            return "⊥"
        if self.pattern is None:
            return "⊤"
        return f"pattern=/{self.pattern}/"


# ---------------------------------------------------------------------------
# StringValue (reduced product)
# ---------------------------------------------------------------------------

@dataclass
class StringValue:
    """Combined string abstract value (reduced product of all sub-domains)."""
    string_set: StringSet
    prefix: StringPrefix
    suffix: StringSuffix
    length: StringLength
    charset: StringCharSet
    pattern: StringPattern

    @staticmethod
    def bottom() -> StringValue:
        return StringValue(
            StringSet.bottom(),
            StringPrefix.bottom(),
            StringSuffix.bottom(),
            StringLength.bottom(),
            StringCharSet.bottom(),
            StringPattern.bottom(),
        )

    @staticmethod
    def top() -> StringValue:
        return StringValue(
            StringSet.top_value(),
            StringPrefix.top(),
            StringSuffix.top(),
            StringLength.top(),
            StringCharSet.top(),
            StringPattern.top(),
        )

    @staticmethod
    def from_constant(s: str) -> StringValue:
        return StringValue(
            StringSet.singleton(s),
            StringPrefix.from_string(s),
            StringSuffix.from_string(s),
            StringLength.exact(len(s)),
            StringCharSet.from_string(s),
            StringPattern.literal(s),
        )

    @staticmethod
    def from_set(strings: Set[str]) -> StringValue:
        if not strings:
            return StringValue.bottom()
        ss = StringSet.from_set(strings)
        lengths = [len(s) for s in strings]
        slist = list(strings)
        prefix_val = slist[0]
        for s in slist[1:]:
            prefix_val = _common_prefix(prefix_val, s)
        suffix_val = slist[0]
        for s in slist[1:]:
            suffix_val = _common_suffix(suffix_val, s)
        return StringValue(
            ss,
            StringPrefix.from_string(prefix_val) if prefix_val else StringPrefix.top(),
            StringSuffix.from_string(suffix_val) if suffix_val else StringSuffix.top(),
            StringLength.range(min(lengths), max(lengths)),
            StringCharSet.from_strings(strings),
            StringPattern.top(),
        )

    def is_bottom(self) -> bool:
        return (self.string_set.is_bottom() or self.prefix.is_bottom() or
                self.suffix.is_bottom() or self.length.is_bottom() or
                self.charset.is_bottom() or self.pattern.is_bottom())

    def is_top(self) -> bool:
        return (self.string_set.is_top and self.prefix.is_top() and
                self.suffix.is_top() and self.length.is_top() and
                self.charset.is_top() and self.pattern.is_top())

    def copy(self) -> StringValue:
        return StringValue(
            self.string_set, self.prefix, self.suffix,
            self.length, self.charset, self.pattern,
        )

    def join(self, other: StringValue) -> StringValue:
        if self.is_bottom():
            return other.copy()
        if other.is_bottom():
            return self.copy()
        return StringValue(
            self.string_set.join(other.string_set),
            self.prefix.join(other.prefix),
            self.suffix.join(other.suffix),
            self.length.join(other.length),
            self.charset.join(other.charset),
            self.pattern.join(other.pattern),
        )

    def meet(self, other: StringValue) -> StringValue:
        if self.is_bottom() or other.is_bottom():
            return StringValue.bottom()
        result = StringValue(
            self.string_set.meet(other.string_set),
            self.prefix.meet(other.prefix),
            self.suffix.meet(other.suffix),
            self.length.meet(other.length),
            self.charset.meet(other.charset),
            self.pattern.meet(other.pattern),
        )
        return result.reduce()

    def widen(self, other: StringValue) -> StringValue:
        return StringValue(
            self.string_set.widen(other.string_set),
            self.prefix.widen(other.prefix),
            self.suffix.widen(other.suffix),
            self.length.widen(other.length),
            self.charset.join(other.charset),
            self.pattern.join(other.pattern),
        )

    def narrow(self, other: StringValue) -> StringValue:
        return StringValue(
            self.string_set.narrow(other.string_set),
            self.prefix if not self.prefix.is_top() else other.prefix,
            self.suffix if not self.suffix.is_top() else other.suffix,
            self.length.narrow(other.length),
            self.charset.meet(other.charset) if self.charset.is_top() else self.charset,
            self.pattern.meet(other.pattern) if self.pattern.is_top() else self.pattern,
        )

    def leq(self, other: StringValue) -> bool:
        return (self.string_set.leq(other.string_set) and
                self.prefix.leq(other.prefix) and
                self.suffix.leq(other.suffix) and
                self.length.leq(other.length) and
                self.charset.leq(other.charset) and
                self.pattern.leq(other.pattern))

    def reduce(self) -> StringValue:
        """Apply reduction rules between sub-domains."""
        if self.is_bottom():
            return StringValue.bottom()
        ss = self.string_set
        prefix = self.prefix
        suffix = self.suffix
        length = self.length
        charset = self.charset
        pattern = self.pattern

        # Reduce string set by prefix/suffix/length/charset/pattern
        if not ss.is_top and not ss.is_bottom():
            filtered = set()
            for s in ss.strings:
                if prefix.matches(s) and suffix.matches(s) and length.contains(len(s)):
                    if pattern.matches(s):
                        filtered.add(s)
            ss = StringSet.from_set(filtered)
            if ss.is_bottom():
                return StringValue.bottom()

        # If string set is a singleton, tighten everything
        if not ss.is_top and len(ss.strings) == 1:
            s = next(iter(ss.strings))
            return StringValue.from_constant(s)

        # If string set is small and finite, tighten other components
        if not ss.is_top and not ss.is_bottom() and len(ss.strings) <= 50:
            strs = ss.strings
            lengths = [len(s) for s in strs]
            length = length.meet(StringLength.range(min(lengths), max(lengths)))
            if length.is_bottom():
                return StringValue.bottom()

        return StringValue(ss, prefix, suffix, length, charset, pattern)

    def __str__(self) -> str:
        if self.is_bottom():
            return "⊥"
        if self.is_top():
            return "⊤"
        parts = []
        if not self.string_set.is_top:
            parts.append(f"set={self.string_set}")
        if not self.prefix.is_top():
            parts.append(str(self.prefix))
        if not self.suffix.is_top():
            parts.append(str(self.suffix))
        if not self.length.is_top():
            parts.append(str(self.length))
        return " ∧ ".join(parts) if parts else "⊤"


# ---------------------------------------------------------------------------
# StringDomain
# ---------------------------------------------------------------------------

class StringDomain:
    """Full string abstract domain with all lattice operations and transfer functions."""

    def bottom(self) -> StringValue:
        return StringValue.bottom()

    def top(self) -> StringValue:
        return StringValue.top()

    def is_bottom(self, v: StringValue) -> bool:
        return v.is_bottom()

    def is_top(self, v: StringValue) -> bool:
        return v.is_top()

    def leq(self, a: StringValue, b: StringValue) -> bool:
        return a.leq(b)

    def join(self, a: StringValue, b: StringValue) -> StringValue:
        return a.join(b)

    def meet(self, a: StringValue, b: StringValue) -> StringValue:
        return a.meet(b)

    def widen(self, a: StringValue, b: StringValue) -> StringValue:
        return a.widen(b)

    def narrow(self, a: StringValue, b: StringValue) -> StringValue:
        return a.narrow(b)

    # --- String operations ---

    def assume_eq(self, v: StringValue, literal: str) -> StringValue:
        """Assume v == literal."""
        c = StringValue.from_constant(literal)
        return v.meet(c)

    def assume_ne(self, v: StringValue, literal: str) -> StringValue:
        """Assume v != literal."""
        if v.is_bottom():
            return v
        result = v.copy()
        result.string_set = result.string_set.subtract(literal)
        return result.reduce()

    def concatenate(self, a: StringValue, b: StringValue) -> StringValue:
        """Compute a + b (string concatenation)."""
        if a.is_bottom() or b.is_bottom():
            return StringValue.bottom()
        # String set
        ss: StringSet
        if not a.string_set.is_top and not b.string_set.is_top:
            if a.string_set.size() * b.string_set.size() <= 100:
                result_strs: Set[str] = set()
                for s1 in a.string_set:
                    for s2 in b.string_set:
                        result_strs.add(s1 + s2)
                ss = StringSet.from_set(result_strs)
            else:
                ss = StringSet.top_value()
        else:
            ss = StringSet.top_value()
        # Prefix: a's prefix (if a is not top)
        pfx = a.prefix if not a.prefix.is_top() else StringPrefix.top()
        # Suffix: b's suffix
        sfx = b.suffix if not b.suffix.is_top() else StringSuffix.top()
        # Length: sum
        length = a.length.add(b.length)
        # Charset: union
        cs = a.charset.join(b.charset)
        return StringValue(ss, pfx, sfx, length, cs, StringPattern.top())

    def substring(self, v: StringValue, start: int, end: Optional[int] = None) -> StringValue:
        """Compute v[start:end]."""
        if v.is_bottom():
            return StringValue.bottom()
        if not v.string_set.is_top and not v.string_set.is_bottom():
            result_strs = set()
            for s in v.string_set:
                result_strs.add(s[start:end])
            return StringValue.from_set(result_strs)
        return StringValue.top()

    def startswith(self, v: StringValue, prefix: str) -> Tuple[StringValue, StringValue]:
        """Split v into (v where v.startswith(prefix), v where not v.startswith(prefix))."""
        if v.is_bottom():
            return StringValue.bottom(), StringValue.bottom()
        true_branch = v.copy()
        false_branch = v.copy()
        # True branch: add prefix constraint
        true_branch.prefix = true_branch.prefix.meet(StringPrefix.from_string(prefix))
        if not true_branch.string_set.is_top:
            true_branch.string_set = true_branch.string_set.filter(lambda s: s.startswith(prefix))
        # False branch: filter out matching strings
        if not false_branch.string_set.is_top:
            false_branch.string_set = false_branch.string_set.filter(lambda s: not s.startswith(prefix))
        return true_branch.reduce(), false_branch.reduce()

    def endswith(self, v: StringValue, suffix: str) -> Tuple[StringValue, StringValue]:
        """Split v into (v where v.endswith(suffix), v where not)."""
        if v.is_bottom():
            return StringValue.bottom(), StringValue.bottom()
        true_branch = v.copy()
        false_branch = v.copy()
        true_branch.suffix = true_branch.suffix.meet(StringSuffix.from_string(suffix))
        if not true_branch.string_set.is_top:
            true_branch.string_set = true_branch.string_set.filter(lambda s: s.endswith(suffix))
        if not false_branch.string_set.is_top:
            false_branch.string_set = false_branch.string_set.filter(lambda s: not s.endswith(suffix))
        return true_branch.reduce(), false_branch.reduce()

    def contains_sub(self, v: StringValue, sub: str) -> Tuple[StringValue, StringValue]:
        """Split v by 'sub' in v."""
        if v.is_bottom():
            return StringValue.bottom(), StringValue.bottom()
        true_branch = v.copy()
        false_branch = v.copy()
        if not true_branch.string_set.is_top:
            true_branch.string_set = true_branch.string_set.filter(lambda s: sub in s)
        if not false_branch.string_set.is_top:
            false_branch.string_set = false_branch.string_set.filter(lambda s: sub not in s)
        return true_branch.reduce(), false_branch.reduce()

    def split(self, v: StringValue, sep: str) -> Tuple[StringValue, StringLength]:
        """Model v.split(sep).
        Returns (element_domain, result_length) where result_length is the
        interval of the number of parts."""
        if v.is_bottom():
            return StringValue.bottom(), StringLength.bottom()
        if not v.string_set.is_top and not v.string_set.is_bottom():
            all_parts: Set[str] = set()
            lengths: List[int] = []
            for s in v.string_set:
                parts = s.split(sep)
                all_parts.update(parts)
                lengths.append(len(parts))
            elem = StringValue.from_set(all_parts)
            count = StringLength.range(min(lengths), max(lengths))
            return elem, count
        # Conservative: at least 1 part
        return StringValue.top(), StringLength.at_least(1)

    def join_strings(self, sep: StringValue, parts: StringValue,
                     part_count: StringLength) -> StringValue:
        """Model sep.join(parts)."""
        if sep.is_bottom() or parts.is_bottom():
            return StringValue.bottom()
        if (not sep.string_set.is_top and not parts.string_set.is_top and
                sep.string_set.size() <= 5 and parts.string_set.size() <= 10 and
                part_count.is_exact() and part_count.min_len <= 5):
            result_strs: Set[str] = set()
            n = part_count.min_len
            for s in sep.string_set:
                # Generate all n-tuples of parts (limited)
                combos = [[]]
                for _ in range(n):
                    new_combos = []
                    for combo in combos:
                        for p in parts.string_set:
                            new_combos.append(combo + [p])
                    combos = new_combos
                    if len(combos) > 100:
                        return StringValue.top()
                for combo in combos:
                    result_strs.add(s.join(combo))
            return StringValue.from_set(result_strs) if result_strs else StringValue.bottom()
        return StringValue.top()

    def format_string(self, template: str, args: Dict[str, StringValue]) -> StringValue:
        """Track f-string / format string results.
        template is the format string with {key} placeholders."""
        if not args:
            return StringValue.from_constant(template)
        # Try to enumerate if all args are small finite sets
        all_finite = all(
            not v.string_set.is_top and v.string_set.size() <= 10
            for v in args.values()
        )
        if all_finite:
            result_strs: Set[str] = set()
            combos: List[Dict[str, str]] = [{}]
            for key, val in args.items():
                new_combos = []
                for combo in combos:
                    for s in val.string_set:
                        c = dict(combo)
                        c[key] = s
                        new_combos.append(c)
                combos = new_combos
                if len(combos) > 200:
                    return StringValue.top()
            for combo in combos:
                try:
                    result_strs.add(template.format(**combo))
                except (KeyError, IndexError, ValueError):
                    pass
            return StringValue.from_set(result_strs) if result_strs else StringValue.top()
        return StringValue.top()

    def upper(self, v: StringValue) -> StringValue:
        """Model v.upper()."""
        if v.is_bottom():
            return v
        if not v.string_set.is_top:
            return StringValue.from_set({s.upper() for s in v.string_set})
        result = v.copy()
        result.string_set = StringSet.top_value()
        return result

    def lower(self, v: StringValue) -> StringValue:
        """Model v.lower()."""
        if v.is_bottom():
            return v
        if not v.string_set.is_top:
            return StringValue.from_set({s.lower() for s in v.string_set})
        result = v.copy()
        result.string_set = StringSet.top_value()
        return result

    def strip(self, v: StringValue, chars: Optional[str] = None) -> StringValue:
        """Model v.strip()."""
        if v.is_bottom():
            return v
        if not v.string_set.is_top:
            return StringValue.from_set({s.strip(chars) for s in v.string_set})
        return StringValue.top()

    def lstrip(self, v: StringValue, chars: Optional[str] = None) -> StringValue:
        if v.is_bottom():
            return v
        if not v.string_set.is_top:
            return StringValue.from_set({s.lstrip(chars) for s in v.string_set})
        return StringValue.top()

    def rstrip(self, v: StringValue, chars: Optional[str] = None) -> StringValue:
        if v.is_bottom():
            return v
        if not v.string_set.is_top:
            return StringValue.from_set({s.rstrip(chars) for s in v.string_set})
        return StringValue.top()

    def replace(self, v: StringValue, old: str, new: str,
                count: int = -1) -> StringValue:
        """Model v.replace(old, new, count)."""
        if v.is_bottom():
            return v
        if not v.string_set.is_top:
            if count < 0:
                return StringValue.from_set({s.replace(old, new) for s in v.string_set})
            return StringValue.from_set({s.replace(old, new, count) for s in v.string_set})
        return StringValue.top()

    def find(self, v: StringValue, sub: str) -> Tuple[bool, bool]:
        """Model v.find(sub).  Returns (may_find, must_find)."""
        if v.is_bottom():
            return False, False
        if not v.string_set.is_top:
            found_some = any(sub in s for s in v.string_set)
            found_all = all(sub in s for s in v.string_set)
            return found_some, found_all
        return True, False

    def count_sub(self, v: StringValue, sub: str) -> StringLength:
        """Model v.count(sub) as a length interval."""
        if v.is_bottom():
            return StringLength.bottom()
        if not v.string_set.is_top:
            counts = [s.count(sub) for s in v.string_set]
            return StringLength.range(min(counts), max(counts))
        return StringLength.at_least(0)

    def str_from_int(self) -> StringValue:
        """Model str(x) where x is int. Very imprecise."""
        return StringValue.top()

    def str_from_float(self) -> StringValue:
        """Model str(x) where x is float. Very imprecise."""
        return StringValue.top()

    def int_from_str(self, v: StringValue) -> bool:
        """Check if int(v) would succeed.  Returns True if it might."""
        if v.is_bottom():
            return False
        if not v.string_set.is_top:
            return any(_is_int_string(s) for s in v.string_set)
        return True

    def float_from_str(self, v: StringValue) -> bool:
        """Check if float(v) would succeed."""
        if v.is_bottom():
            return False
        if not v.string_set.is_top:
            return any(_is_float_string(s) for s in v.string_set)
        return True

    def encode(self, v: StringValue, encoding: str = "utf-8") -> StringValue:
        """Model v.encode()."""
        # In the abstract, encoding doesn't change the string set
        return v.copy()

    def decode(self, v: StringValue, encoding: str = "utf-8") -> StringValue:
        """Model bytes.decode()."""
        return v.copy()


def _is_int_string(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def _is_float_string(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# StringEqualityDomain
# ---------------------------------------------------------------------------

class StringEqualityDomain:
    """Core finite-set equality domain for strings.
    Tracks which string constants a variable may equal.
    This is the primary domain used for hasattr/dict key reasoning."""

    def __init__(self, widen_threshold: int = 32) -> None:
        self.widen_threshold = widen_threshold
        self._env: Dict[str, StringSet] = {}

    def copy(self) -> StringEqualityDomain:
        d = StringEqualityDomain(self.widen_threshold)
        d._env = dict(self._env)
        return d

    def get(self, var: str) -> StringSet:
        return self._env.get(var, StringSet.top_value())

    def set(self, var: str, value: StringSet) -> None:
        self._env[var] = value

    def assign_const(self, var: str, value: str) -> None:
        self._env[var] = StringSet.singleton(value)

    def assign_var(self, dst: str, src: str) -> None:
        self._env[dst] = self.get(src)

    def assign_top(self, var: str) -> None:
        self._env[var] = StringSet.top_value()

    def forget(self, var: str) -> None:
        self._env.pop(var, None)

    def assume_eq(self, var: str, value: str) -> bool:
        """Assume var == value.  Returns False if this is impossible."""
        current = self.get(var)
        if not current.contains(value):
            return False
        self._env[var] = StringSet.singleton(value)
        return True

    def assume_ne(self, var: str, value: str) -> None:
        """Assume var != value."""
        current = self.get(var)
        self._env[var] = current.subtract(value)

    def assume_in_set(self, var: str, values: Set[str]) -> bool:
        """Assume var ∈ values."""
        current = self.get(var)
        result = current.meet(StringSet.from_set(values))
        if result.is_bottom():
            return False
        self._env[var] = result
        return True

    def join(self, other: StringEqualityDomain) -> StringEqualityDomain:
        result = StringEqualityDomain(self.widen_threshold)
        all_vars = set(self._env.keys()) | set(other._env.keys())
        for v in all_vars:
            a = self.get(v)
            b = other.get(v)
            result._env[v] = a.join(b)
        return result

    def meet(self, other: StringEqualityDomain) -> StringEqualityDomain:
        result = StringEqualityDomain(self.widen_threshold)
        all_vars = set(self._env.keys()) | set(other._env.keys())
        for v in all_vars:
            a = self.get(v)
            b = other.get(v)
            result._env[v] = a.meet(b)
        return result

    def widen(self, other: StringEqualityDomain) -> StringEqualityDomain:
        result = StringEqualityDomain(self.widen_threshold)
        all_vars = set(self._env.keys()) | set(other._env.keys())
        for v in all_vars:
            a = self.get(v)
            b = other.get(v)
            result._env[v] = a.widen(b, self.widen_threshold)
        return result

    def leq(self, other: StringEqualityDomain) -> bool:
        for v, val in self._env.items():
            if not val.leq(other.get(v)):
                return False
        return True

    def is_bottom(self) -> bool:
        return any(v.is_bottom() for v in self._env.values())

    def variables(self) -> Set[str]:
        return set(self._env.keys())

    def transfer_hasattr(self, obj_var: str, attr: str,
                         true_branch: bool) -> StringEqualityDomain:
        """Transfer function for hasattr(obj, attr) or 'attr' in obj.__dict__."""
        result = self.copy()
        attr_set_var = f"#attrs({obj_var})"
        if true_branch:
            current = result.get(attr_set_var)
            if current.is_top:
                result.set(attr_set_var, StringSet.singleton(attr))
            else:
                result.set(attr_set_var, current.add(attr))
        return result

    def transfer_dict_key_check(self, dict_var: str, key: str,
                                 true_branch: bool) -> StringEqualityDomain:
        """Transfer for 'key' in dict_var."""
        result = self.copy()
        key_set_var = f"#keys({dict_var})"
        if true_branch:
            current = result.get(key_set_var)
            if current.is_top:
                result.set(key_set_var, StringSet.singleton(key))
            else:
                result.set(key_set_var, current.add(key))
        return result

    def __str__(self) -> str:
        if not self._env:
            return "⊤"
        parts = []
        for v in sorted(self._env.keys()):
            parts.append(f"{v} ∈ {self._env[v]}")
        return " ∧ ".join(parts)


# ---------------------------------------------------------------------------
# DictKeyDomain
# ---------------------------------------------------------------------------

@dataclass
class DictKeyInfo:
    """Information about dictionary keys."""
    definite_keys: FrozenSet[str]  # keys known to be present
    possible_keys: FrozenSet[str]  # keys that might be present (superset of definite)
    is_top: bool = False  # unknown keys possible

    @staticmethod
    def bottom() -> DictKeyInfo:
        return DictKeyInfo(frozenset(), frozenset())

    @staticmethod
    def top_value() -> DictKeyInfo:
        return DictKeyInfo(frozenset(), frozenset(), True)

    @staticmethod
    def from_keys(keys: Set[str]) -> DictKeyInfo:
        fk = frozenset(keys)
        return DictKeyInfo(fk, fk)

    def has_key(self, key: str) -> Optional[bool]:
        """Returns True if key is definitely present, False if definitely absent,
        None if unknown."""
        if key in self.definite_keys:
            return True
        if not self.is_top and key not in self.possible_keys:
            return False
        return None

    def add_key(self, key: str) -> DictKeyInfo:
        return DictKeyInfo(
            self.definite_keys | {key},
            self.possible_keys | {key},
            self.is_top,
        )

    def remove_key(self, key: str) -> DictKeyInfo:
        return DictKeyInfo(
            self.definite_keys - {key},
            self.possible_keys,  # might still be present after conditional removal
            self.is_top,
        )

    def definitely_remove_key(self, key: str) -> DictKeyInfo:
        return DictKeyInfo(
            self.definite_keys - {key},
            self.possible_keys - {key},
            self.is_top,
        )

    def join(self, other: DictKeyInfo) -> DictKeyInfo:
        return DictKeyInfo(
            self.definite_keys & other.definite_keys,
            self.possible_keys | other.possible_keys,
            self.is_top or other.is_top,
        )

    def meet(self, other: DictKeyInfo) -> DictKeyInfo:
        return DictKeyInfo(
            self.definite_keys | other.definite_keys,
            self.possible_keys & other.possible_keys if not self.is_top and not other.is_top else
            self.possible_keys if other.is_top else other.possible_keys,
            self.is_top and other.is_top,
        )

    def widen(self, other: DictKeyInfo, threshold: int = 50) -> DictKeyInfo:
        joined = self.join(other)
        if len(joined.possible_keys) > threshold:
            return DictKeyInfo(joined.definite_keys, frozenset(), True)
        return joined

    def leq(self, other: DictKeyInfo) -> bool:
        if other.is_top:
            return True
        if self.is_top:
            return False
        if not self.definite_keys >= other.definite_keys:
            return False
        return self.possible_keys <= other.possible_keys

    def __str__(self) -> str:
        if self.is_top and not self.definite_keys:
            return "⊤"
        parts = []
        if self.definite_keys:
            parts.append(f"definite={{{', '.join(repr(k) for k in sorted(self.definite_keys))}}}")
        if self.is_top:
            parts.append("possible=⊤")
        elif self.possible_keys - self.definite_keys:
            maybe = self.possible_keys - self.definite_keys
            parts.append(f"maybe={{{', '.join(repr(k) for k in sorted(maybe))}}}")
        return " ∧ ".join(parts) if parts else "∅"


class DictKeyDomain:
    """Tracks which keys a dictionary definitely/possibly has."""

    def __init__(self) -> None:
        self._dicts: Dict[str, DictKeyInfo] = {}

    def copy(self) -> DictKeyDomain:
        d = DictKeyDomain()
        d._dicts = dict(self._dicts)
        return d

    def get(self, dict_var: str) -> DictKeyInfo:
        return self._dicts.get(dict_var, DictKeyInfo.top_value())

    def set_info(self, dict_var: str, info: DictKeyInfo) -> None:
        self._dicts[dict_var] = info

    def create_dict(self, dict_var: str, keys: Set[str]) -> None:
        self._dicts[dict_var] = DictKeyInfo.from_keys(keys)

    def add_key(self, dict_var: str, key: str) -> None:
        info = self.get(dict_var)
        self._dicts[dict_var] = info.add_key(key)

    def remove_key(self, dict_var: str, key: str) -> None:
        info = self.get(dict_var)
        self._dicts[dict_var] = info.remove_key(key)

    def definitely_remove_key(self, dict_var: str, key: str) -> None:
        info = self.get(dict_var)
        self._dicts[dict_var] = info.definitely_remove_key(key)

    def has_key(self, dict_var: str, key: str) -> Optional[bool]:
        return self.get(dict_var).has_key(key)

    def assume_has_key(self, dict_var: str, key: str) -> DictKeyDomain:
        result = self.copy()
        info = result.get(dict_var)
        result._dicts[dict_var] = info.add_key(key)
        return result

    def assume_no_key(self, dict_var: str, key: str) -> DictKeyDomain:
        result = self.copy()
        info = result.get(dict_var)
        result._dicts[dict_var] = info.definitely_remove_key(key)
        return result

    def join(self, other: DictKeyDomain) -> DictKeyDomain:
        result = DictKeyDomain()
        all_dicts = set(self._dicts.keys()) | set(other._dicts.keys())
        for d in all_dicts:
            a = self.get(d)
            b = other.get(d)
            result._dicts[d] = a.join(b)
        return result

    def meet(self, other: DictKeyDomain) -> DictKeyDomain:
        result = DictKeyDomain()
        all_dicts = set(self._dicts.keys()) | set(other._dicts.keys())
        for d in all_dicts:
            a = self.get(d)
            b = other.get(d)
            result._dicts[d] = a.meet(b)
        return result

    def widen(self, other: DictKeyDomain) -> DictKeyDomain:
        result = DictKeyDomain()
        all_dicts = set(self._dicts.keys()) | set(other._dicts.keys())
        for d in all_dicts:
            a = self.get(d)
            b = other.get(d)
            result._dicts[d] = a.widen(b)
        return result

    def leq(self, other: DictKeyDomain) -> bool:
        for d, info in self._dicts.items():
            if not info.leq(other.get(d)):
                return False
        return True

    def intersect_with_string_domain(self, dict_var: str,
                                      key_var: str,
                                      str_domain: StringEqualityDomain) -> DictKeyDomain:
        """Intersect dict key info with string domain for hasattr reasoning."""
        result = self.copy()
        key_values = str_domain.get(key_var)
        if not key_values.is_top and not key_values.is_bottom():
            info = result.get(dict_var)
            if not info.is_top:
                # Only keys that are both possible in dict and in string domain
                new_possible = info.possible_keys & key_values.strings
                new_definite = info.definite_keys & key_values.strings
                result._dicts[dict_var] = DictKeyInfo(new_definite, new_possible)
        return result

    def forget(self, dict_var: str) -> None:
        self._dicts.pop(dict_var, None)

    def __str__(self) -> str:
        if not self._dicts:
            return "⊤"
        parts = []
        for d in sorted(self._dicts.keys()):
            parts.append(f"{d}: {self._dicts[d]}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# AttributePresenceDomain
# ---------------------------------------------------------------------------

@dataclass
class AttributeInfo:
    """Information about object attributes."""
    definite_attrs: FrozenSet[str]
    possible_attrs: FrozenSet[str]
    type_tag: Optional[str] = None  # optional type tag for method resolution
    is_top: bool = False

    @staticmethod
    def bottom() -> AttributeInfo:
        return AttributeInfo(frozenset(), frozenset())

    @staticmethod
    def top_value() -> AttributeInfo:
        return AttributeInfo(frozenset(), frozenset(), None, True)

    @staticmethod
    def from_attrs(attrs: Set[str], type_tag: Optional[str] = None) -> AttributeInfo:
        fa = frozenset(attrs)
        return AttributeInfo(fa, fa, type_tag)

    def has_attr(self, attr: str) -> Optional[bool]:
        if attr in self.definite_attrs:
            return True
        if not self.is_top and attr not in self.possible_attrs:
            return False
        return None

    def add_attr(self, attr: str) -> AttributeInfo:
        return AttributeInfo(
            self.definite_attrs | {attr},
            self.possible_attrs | {attr},
            self.type_tag,
            self.is_top,
        )

    def remove_attr(self, attr: str) -> AttributeInfo:
        return AttributeInfo(
            self.definite_attrs - {attr},
            self.possible_attrs,
            self.type_tag,
            self.is_top,
        )

    def join(self, other: AttributeInfo) -> AttributeInfo:
        tag = self.type_tag if self.type_tag == other.type_tag else None
        return AttributeInfo(
            self.definite_attrs & other.definite_attrs,
            self.possible_attrs | other.possible_attrs,
            tag,
            self.is_top or other.is_top,
        )

    def meet(self, other: AttributeInfo) -> AttributeInfo:
        tag = self.type_tag or other.type_tag
        return AttributeInfo(
            self.definite_attrs | other.definite_attrs,
            self.possible_attrs & other.possible_attrs if not self.is_top and not other.is_top
            else self.possible_attrs if other.is_top else other.possible_attrs,
            tag,
            self.is_top and other.is_top,
        )

    def widen(self, other: AttributeInfo, threshold: int = 50) -> AttributeInfo:
        joined = self.join(other)
        if len(joined.possible_attrs) > threshold:
            return AttributeInfo(joined.definite_attrs, frozenset(), joined.type_tag, True)
        return joined

    def leq(self, other: AttributeInfo) -> bool:
        if other.is_top:
            return True
        if self.is_top:
            return False
        if not self.definite_attrs >= other.definite_attrs:
            return False
        return self.possible_attrs <= other.possible_attrs

    def __str__(self) -> str:
        if self.is_top and not self.definite_attrs:
            return "⊤"
        parts = []
        if self.type_tag:
            parts.append(f"type={self.type_tag}")
        if self.definite_attrs:
            parts.append(f"definite={{{', '.join(repr(a) for a in sorted(self.definite_attrs))}}}")
        if self.is_top:
            parts.append("possible=⊤")
        elif self.possible_attrs - self.definite_attrs:
            maybe = self.possible_attrs - self.definite_attrs
            parts.append(f"maybe={{{', '.join(repr(a) for a in sorted(maybe))}}}")
        return " ∧ ".join(parts) if parts else "∅"


class AttributePresenceDomain:
    """Tracks which attributes an object has.  Similar to DictKeyDomain."""

    def __init__(self) -> None:
        self._objects: Dict[str, AttributeInfo] = {}

    def copy(self) -> AttributePresenceDomain:
        d = AttributePresenceDomain()
        d._objects = dict(self._objects)
        return d

    def get(self, obj_var: str) -> AttributeInfo:
        return self._objects.get(obj_var, AttributeInfo.top_value())

    def set_info(self, obj_var: str, info: AttributeInfo) -> None:
        self._objects[obj_var] = info

    def create_object(self, obj_var: str, attrs: Set[str],
                      type_tag: Optional[str] = None) -> None:
        self._objects[obj_var] = AttributeInfo.from_attrs(attrs, type_tag)

    def add_attr(self, obj_var: str, attr: str) -> None:
        info = self.get(obj_var)
        self._objects[obj_var] = info.add_attr(attr)

    def remove_attr(self, obj_var: str, attr: str) -> None:
        info = self.get(obj_var)
        self._objects[obj_var] = info.remove_attr(attr)

    def has_attr(self, obj_var: str, attr: str) -> Optional[bool]:
        return self.get(obj_var).has_attr(attr)

    def assume_has_attr(self, obj_var: str, attr: str) -> AttributePresenceDomain:
        result = self.copy()
        info = result.get(obj_var)
        result._objects[obj_var] = info.add_attr(attr)
        return result

    def assume_no_attr(self, obj_var: str, attr: str) -> AttributePresenceDomain:
        result = self.copy()
        info = result.get(obj_var)
        result._objects[obj_var] = AttributeInfo(
            info.definite_attrs - {attr},
            info.possible_attrs - {attr},
            info.type_tag,
            info.is_top,
        )
        return result

    def get_type_tag(self, obj_var: str) -> Optional[str]:
        return self.get(obj_var).type_tag

    def set_type_tag(self, obj_var: str, tag: str) -> None:
        info = self.get(obj_var)
        self._objects[obj_var] = AttributeInfo(
            info.definite_attrs, info.possible_attrs, tag, info.is_top
        )

    def resolve_method(self, obj_var: str, method: str) -> Optional[bool]:
        """Check if obj has a method. Uses type_tag if available."""
        return self.has_attr(obj_var, method)

    def join(self, other: AttributePresenceDomain) -> AttributePresenceDomain:
        result = AttributePresenceDomain()
        all_objs = set(self._objects.keys()) | set(other._objects.keys())
        for o in all_objs:
            a = self.get(o)
            b = other.get(o)
            result._objects[o] = a.join(b)
        return result

    def meet(self, other: AttributePresenceDomain) -> AttributePresenceDomain:
        result = AttributePresenceDomain()
        all_objs = set(self._objects.keys()) | set(other._objects.keys())
        for o in all_objs:
            a = self.get(o)
            b = other.get(o)
            result._objects[o] = a.meet(b)
        return result

    def widen(self, other: AttributePresenceDomain) -> AttributePresenceDomain:
        result = AttributePresenceDomain()
        all_objs = set(self._objects.keys()) | set(other._objects.keys())
        for o in all_objs:
            a = self.get(o)
            b = other.get(o)
            result._objects[o] = a.widen(b)
        return result

    def leq(self, other: AttributePresenceDomain) -> bool:
        for o, info in self._objects.items():
            if not info.leq(other.get(o)):
                return False
        return True

    def forget(self, obj_var: str) -> None:
        self._objects.pop(obj_var, None)

    def __str__(self) -> str:
        if not self._objects:
            return "⊤"
        parts = []
        for o in sorted(self._objects.keys()):
            parts.append(f"{o}: {self._objects[o]}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# StringTransferFunctions
# ---------------------------------------------------------------------------

class StringOpKind(Enum):
    SPLIT = auto()
    JOIN = auto()
    REPLACE = auto()
    FIND = auto()
    COUNT = auto()
    UPPER = auto()
    LOWER = auto()
    STRIP = auto()
    LSTRIP = auto()
    RSTRIP = auto()
    STARTSWITH = auto()
    ENDSWITH = auto()
    FORMAT = auto()
    ENCODE = auto()
    DECODE = auto()
    CONCAT = auto()
    SUBSTRING = auto()
    CONTAINS = auto()


@dataclass
class StringOp:
    kind: StringOpKind
    target: Optional[str] = None
    operand: Optional[str] = None
    args: Optional[List[str]] = None
    const_arg: Optional[str] = None
    const_args: Optional[List[str]] = None
    int_args: Optional[List[int]] = None
    format_template: Optional[str] = None
    format_keys: Optional[Dict[str, str]] = None


class StringTransferFunctions:
    """Transfer functions for string operations in the abstract domain."""

    def __init__(self) -> None:
        self.domain = StringDomain()
        self.eq_domain = StringEqualityDomain()

    def transfer(self, env: Dict[str, StringValue], op: StringOp) -> Dict[str, StringValue]:
        """Apply a string operation to the environment."""
        env = dict(env)
        if op.kind == StringOpKind.CONCAT:
            return self._transfer_concat(env, op)
        elif op.kind == StringOpKind.SPLIT:
            return self._transfer_split(env, op)
        elif op.kind == StringOpKind.JOIN:
            return self._transfer_join(env, op)
        elif op.kind == StringOpKind.REPLACE:
            return self._transfer_replace(env, op)
        elif op.kind == StringOpKind.UPPER:
            return self._transfer_case(env, op, str.upper)
        elif op.kind == StringOpKind.LOWER:
            return self._transfer_case(env, op, str.lower)
        elif op.kind == StringOpKind.STRIP:
            return self._transfer_strip(env, op)
        elif op.kind == StringOpKind.LSTRIP:
            return self._transfer_lstrip(env, op)
        elif op.kind == StringOpKind.RSTRIP:
            return self._transfer_rstrip(env, op)
        elif op.kind == StringOpKind.STARTSWITH:
            return self._transfer_startswith(env, op)
        elif op.kind == StringOpKind.ENDSWITH:
            return self._transfer_endswith(env, op)
        elif op.kind == StringOpKind.FORMAT:
            return self._transfer_format(env, op)
        elif op.kind == StringOpKind.SUBSTRING:
            return self._transfer_substring(env, op)
        elif op.kind == StringOpKind.CONTAINS:
            return self._transfer_contains(env, op)
        elif op.kind == StringOpKind.FIND:
            return self._transfer_find(env, op)
        elif op.kind == StringOpKind.COUNT:
            return self._transfer_count(env, op)
        return env

    def _get_val(self, env: Dict[str, StringValue], var: Optional[str]) -> StringValue:
        if var is None:
            return StringValue.top()
        return env.get(var, StringValue.top())

    def _transfer_concat(self, env: Dict[str, StringValue], op: StringOp) -> Dict[str, StringValue]:
        assert op.target is not None
        assert op.args is not None and len(op.args) == 2
        a = self._get_val(env, op.args[0])
        b = self._get_val(env, op.args[1])
        env[op.target] = self.domain.concatenate(a, b)
        return env

    def _transfer_split(self, env: Dict[str, StringValue], op: StringOp) -> Dict[str, StringValue]:
        assert op.target is not None
        assert op.operand is not None
        sep = op.const_arg or " "
        v = self._get_val(env, op.operand)
        elem, _ = self.domain.split(v, sep)
        env[op.target] = elem
        return env

    def _transfer_join(self, env: Dict[str, StringValue], op: StringOp) -> Dict[str, StringValue]:
        assert op.target is not None
        assert op.operand is not None
        sep = self._get_val(env, op.operand)
        parts = self._get_val(env, op.args[0] if op.args else None)
        env[op.target] = self.domain.join_strings(sep, parts, StringLength.at_least(1))
        return env

    def _transfer_replace(self, env: Dict[str, StringValue], op: StringOp) -> Dict[str, StringValue]:
        assert op.target is not None
        assert op.operand is not None
        assert op.const_args is not None and len(op.const_args) >= 2
        v = self._get_val(env, op.operand)
        old = op.const_args[0]
        new = op.const_args[1]
        env[op.target] = self.domain.replace(v, old, new)
        return env

    def _transfer_case(self, env: Dict[str, StringValue], op: StringOp,
                       fn: Callable[[str], str]) -> Dict[str, StringValue]:
        assert op.target is not None
        assert op.operand is not None
        v = self._get_val(env, op.operand)
        if not v.string_set.is_top and not v.string_set.is_bottom():
            env[op.target] = StringValue.from_set({fn(s) for s in v.string_set})
        else:
            env[op.target] = StringValue.top()
        return env

    def _transfer_strip(self, env: Dict[str, StringValue], op: StringOp) -> Dict[str, StringValue]:
        assert op.target is not None
        assert op.operand is not None
        v = self._get_val(env, op.operand)
        env[op.target] = self.domain.strip(v, op.const_arg)
        return env

    def _transfer_lstrip(self, env: Dict[str, StringValue], op: StringOp) -> Dict[str, StringValue]:
        assert op.target is not None
        assert op.operand is not None
        v = self._get_val(env, op.operand)
        env[op.target] = self.domain.lstrip(v, op.const_arg)
        return env

    def _transfer_rstrip(self, env: Dict[str, StringValue], op: StringOp) -> Dict[str, StringValue]:
        assert op.target is not None
        assert op.operand is not None
        v = self._get_val(env, op.operand)
        env[op.target] = self.domain.rstrip(v, op.const_arg)
        return env

    def _transfer_startswith(self, env: Dict[str, StringValue], op: StringOp) -> Dict[str, StringValue]:
        assert op.operand is not None
        assert op.const_arg is not None
        v = self._get_val(env, op.operand)
        true_val, _ = self.domain.startswith(v, op.const_arg)
        if op.target:
            env[op.target] = true_val
        return env

    def _transfer_endswith(self, env: Dict[str, StringValue], op: StringOp) -> Dict[str, StringValue]:
        assert op.operand is not None
        assert op.const_arg is not None
        v = self._get_val(env, op.operand)
        true_val, _ = self.domain.endswith(v, op.const_arg)
        if op.target:
            env[op.target] = true_val
        return env

    def _transfer_format(self, env: Dict[str, StringValue], op: StringOp) -> Dict[str, StringValue]:
        assert op.target is not None
        assert op.format_template is not None
        args: Dict[str, StringValue] = {}
        if op.format_keys:
            for k, var in op.format_keys.items():
                args[k] = self._get_val(env, var)
        env[op.target] = self.domain.format_string(op.format_template, args)
        return env

    def _transfer_substring(self, env: Dict[str, StringValue], op: StringOp) -> Dict[str, StringValue]:
        assert op.target is not None
        assert op.operand is not None
        assert op.int_args is not None
        v = self._get_val(env, op.operand)
        start = op.int_args[0] if len(op.int_args) > 0 else 0
        end = op.int_args[1] if len(op.int_args) > 1 else None
        env[op.target] = self.domain.substring(v, start, end)
        return env

    def _transfer_contains(self, env: Dict[str, StringValue], op: StringOp) -> Dict[str, StringValue]:
        assert op.operand is not None
        assert op.const_arg is not None
        v = self._get_val(env, op.operand)
        true_val, _ = self.domain.contains_sub(v, op.const_arg)
        if op.target:
            env[op.target] = true_val
        return env

    def _transfer_find(self, env: Dict[str, StringValue], op: StringOp) -> Dict[str, StringValue]:
        assert op.operand is not None
        assert op.const_arg is not None
        v = self._get_val(env, op.operand)
        may_find, must_find = self.domain.find(v, op.const_arg)
        # find() returns -1 if not found, otherwise the index
        # We just track whether it's possible
        return env

    def _transfer_count(self, env: Dict[str, StringValue], op: StringOp) -> Dict[str, StringValue]:
        assert op.operand is not None
        assert op.const_arg is not None
        v = self._get_val(env, op.operand)
        count = self.domain.count_sub(v, op.const_arg)
        # Store count result as metadata
        return env

    def transfer_regex_match(self, env: Dict[str, StringValue],
                              target: str, operand: str,
                              pattern: str) -> Dict[str, StringValue]:
        """Transfer for re.match(pattern, operand)."""
        env = dict(env)
        v = self._get_val(env, operand)
        if not v.string_set.is_top and not v.string_set.is_bottom():
            matching = set()
            for s in v.string_set:
                if re.match(pattern, s):
                    matching.add(s)
            if matching:
                env[target] = StringValue.from_set(matching)
            else:
                env[target] = StringValue.bottom()
        else:
            env[target] = StringValue.top()
        return env

    def transfer_regex_search(self, env: Dict[str, StringValue],
                               target: str, operand: str,
                               pattern: str) -> Dict[str, StringValue]:
        """Transfer for re.search(pattern, operand)."""
        env = dict(env)
        v = self._get_val(env, operand)
        if not v.string_set.is_top and not v.string_set.is_bottom():
            matching = set()
            for s in v.string_set:
                if re.search(pattern, s):
                    matching.add(s)
            if matching:
                env[target] = StringValue.from_set(matching)
            else:
                env[target] = StringValue.bottom()
        else:
            env[target] = StringValue.top()
        return env

    def transfer_regex_findall(self, env: Dict[str, StringValue],
                                target: str, operand: str,
                                pattern: str) -> Dict[str, StringValue]:
        """Transfer for re.findall(pattern, operand)."""
        env = dict(env)
        v = self._get_val(env, operand)
        if not v.string_set.is_top and not v.string_set.is_bottom():
            all_matches: Set[str] = set()
            for s in v.string_set:
                matches = re.findall(pattern, s)
                for m in matches:
                    if isinstance(m, str):
                        all_matches.add(m)
            env[target] = StringValue.from_set(all_matches) if all_matches else StringValue.bottom()
        else:
            env[target] = StringValue.top()
        return env

    def transfer_comparison(self, env: Dict[str, StringValue],
                             var: str, op: str,
                             const: str) -> Tuple[Dict[str, StringValue], Dict[str, StringValue]]:
        """Transfer for string comparison: var op const.
        Returns (true_env, false_env)."""
        true_env = dict(env)
        false_env = dict(env)
        v = self._get_val(env, var)
        if op == "==":
            true_env[var] = self.domain.assume_eq(v, const)
            false_env[var] = self.domain.assume_ne(v, const)
        elif op == "!=":
            true_env[var] = self.domain.assume_ne(v, const)
            false_env[var] = self.domain.assume_eq(v, const)
        elif op == "in":
            true_v, false_v = self.domain.contains_sub(v, const)
            true_env[var] = true_v
            false_env[var] = false_v
        return true_env, false_env


# ---------------------------------------------------------------------------
# HasAttrAnalysis
# ---------------------------------------------------------------------------

class HasAttrAnalysis:
    """Specialized analysis for hasattr patterns.
    Pattern: if hasattr(obj, 'field'): obj.field
    Tracks field presence through control flow."""

    def __init__(self) -> None:
        self.attr_domain = AttributePresenceDomain()
        self.str_domain = StringEqualityDomain()

    def copy(self) -> HasAttrAnalysis:
        result = HasAttrAnalysis()
        result.attr_domain = self.attr_domain.copy()
        result.str_domain = self.str_domain.copy()
        return result

    def analyze_hasattr_check(self, obj_var: str, attr: str) -> Tuple[HasAttrAnalysis, HasAttrAnalysis]:
        """Analyze: if hasattr(obj, attr)
        Returns (true_state, false_state)."""
        true_state = self.copy()
        false_state = self.copy()
        true_state.attr_domain = true_state.attr_domain.assume_has_attr(obj_var, attr)
        false_state.attr_domain = false_state.attr_domain.assume_no_attr(obj_var, attr)
        # Also update string domain
        true_state.str_domain = true_state.str_domain.transfer_hasattr(obj_var, attr, True)
        false_state.str_domain = false_state.str_domain.transfer_hasattr(obj_var, attr, False)
        return true_state, false_state

    def analyze_getattr_safe(self, obj_var: str, attr: str,
                              default: Optional[str] = None) -> Tuple[Optional[bool], HasAttrAnalysis]:
        """Analyze: getattr(obj, attr, default)
        Returns (has_attr, updated_state)."""
        has = self.attr_domain.has_attr(obj_var, attr)
        state = self.copy()
        return has, state

    def analyze_dynamic_attr_access(self, obj_var: str,
                                     attr_var: str) -> HasAttrAnalysis:
        """Analyze: getattr(obj, attr_var) where attr_var is a variable."""
        result = self.copy()
        possible_attrs = self.str_domain.get(attr_var)
        if not possible_attrs.is_top and not possible_attrs.is_bottom():
            for attr in possible_attrs:
                result.attr_domain.add_attr(obj_var, attr)
        return result

    def analyze_setattr(self, obj_var: str, attr: str) -> HasAttrAnalysis:
        """Analyze: setattr(obj, attr, value) or obj.attr = value."""
        result = self.copy()
        result.attr_domain.add_attr(obj_var, attr)
        return result

    def analyze_delattr(self, obj_var: str, attr: str) -> HasAttrAnalysis:
        """Analyze: delattr(obj, attr)."""
        result = self.copy()
        result.attr_domain.remove_attr(obj_var, attr)
        return result

    def is_safe_access(self, obj_var: str, attr: str) -> bool:
        """Check if obj.attr is safe (attr definitely exists)."""
        return self.attr_domain.has_attr(obj_var, attr) is True

    def may_fail_access(self, obj_var: str, attr: str) -> bool:
        """Check if obj.attr might fail (attr might not exist)."""
        result = self.attr_domain.has_attr(obj_var, attr)
        return result is None or result is False

    def join(self, other: HasAttrAnalysis) -> HasAttrAnalysis:
        result = HasAttrAnalysis()
        result.attr_domain = self.attr_domain.join(other.attr_domain)
        result.str_domain = self.str_domain.join(other.str_domain)
        return result

    def meet(self, other: HasAttrAnalysis) -> HasAttrAnalysis:
        result = HasAttrAnalysis()
        result.attr_domain = self.attr_domain.meet(other.attr_domain)
        result.str_domain = self.str_domain.meet(other.str_domain)
        return result

    def widen(self, other: HasAttrAnalysis) -> HasAttrAnalysis:
        result = HasAttrAnalysis()
        result.attr_domain = self.attr_domain.widen(other.attr_domain)
        result.str_domain = self.str_domain.widen(other.str_domain)
        return result

    def __str__(self) -> str:
        return f"HasAttr:\n  attrs: {self.attr_domain}\n  strings: {self.str_domain}"


# ---------------------------------------------------------------------------
# DictKeyAnalysis
# ---------------------------------------------------------------------------

class DictKeyAnalysis:
    """Specialized analysis for dict key patterns.
    Pattern: if 'key' in d: d['key']
    Pattern: d.get('key', default)
    Tracks key presence through control flow."""

    def __init__(self) -> None:
        self.key_domain = DictKeyDomain()
        self.str_domain = StringEqualityDomain()

    def copy(self) -> DictKeyAnalysis:
        result = DictKeyAnalysis()
        result.key_domain = self.key_domain.copy()
        result.str_domain = self.str_domain.copy()
        return result

    def analyze_key_check(self, dict_var: str, key: str) -> Tuple[DictKeyAnalysis, DictKeyAnalysis]:
        """Analyze: if 'key' in dict_var
        Returns (true_state, false_state)."""
        true_state = self.copy()
        false_state = self.copy()
        true_state.key_domain = true_state.key_domain.assume_has_key(dict_var, key)
        false_state.key_domain = false_state.key_domain.assume_no_key(dict_var, key)
        true_state.str_domain = true_state.str_domain.transfer_dict_key_check(dict_var, key, True)
        false_state.str_domain = false_state.str_domain.transfer_dict_key_check(dict_var, key, False)
        return true_state, false_state

    def analyze_dynamic_key_check(self, dict_var: str,
                                   key_var: str) -> Tuple[DictKeyAnalysis, DictKeyAnalysis]:
        """Analyze: if key_var in dict_var (key is a variable)."""
        true_state = self.copy()
        false_state = self.copy()
        possible_keys = self.str_domain.get(key_var)
        if not possible_keys.is_top and not possible_keys.is_bottom():
            for key in possible_keys:
                true_state.key_domain.add_key(dict_var, key)
        return true_state, false_state

    def analyze_dict_get(self, dict_var: str, key: str,
                          has_default: bool) -> Tuple[Optional[bool], DictKeyAnalysis]:
        """Analyze: d.get(key, default)
        Returns (has_key, updated_state)."""
        has = self.key_domain.has_key(dict_var, key)
        state = self.copy()
        return has, state

    def analyze_dict_setitem(self, dict_var: str, key: str) -> DictKeyAnalysis:
        """Analyze: d[key] = value."""
        result = self.copy()
        result.key_domain.add_key(dict_var, key)
        return result

    def analyze_dict_delitem(self, dict_var: str, key: str) -> DictKeyAnalysis:
        """Analyze: del d[key]."""
        result = self.copy()
        result.key_domain.definitely_remove_key(dict_var, key)
        return result

    def analyze_dict_pop(self, dict_var: str, key: str,
                          has_default: bool) -> DictKeyAnalysis:
        """Analyze: d.pop(key, default)."""
        result = self.copy()
        result.key_domain.definitely_remove_key(dict_var, key)
        return result

    def analyze_dict_update(self, dict_var: str, other_dict_var: str) -> DictKeyAnalysis:
        """Analyze: d.update(other)."""
        result = self.copy()
        other_info = result.key_domain.get(other_dict_var)
        my_info = result.key_domain.get(dict_var)
        # After update, dict_var has all keys from both
        new_definite = my_info.definite_keys | other_info.definite_keys
        new_possible = my_info.possible_keys | other_info.possible_keys
        result.key_domain.set_info(dict_var, DictKeyInfo(
            new_definite, new_possible,
            my_info.is_top or other_info.is_top,
        ))
        return result

    def analyze_dict_literal(self, dict_var: str, keys: Set[str]) -> DictKeyAnalysis:
        """Analyze: d = {'key1': v1, 'key2': v2, ...}."""
        result = self.copy()
        result.key_domain.create_dict(dict_var, keys)
        return result

    def is_safe_access(self, dict_var: str, key: str) -> bool:
        """Check if d[key] is safe."""
        return self.key_domain.has_key(dict_var, key) is True

    def may_fail_access(self, dict_var: str, key: str) -> bool:
        """Check if d[key] might raise KeyError."""
        result = self.key_domain.has_key(dict_var, key)
        return result is None or result is False

    def join(self, other: DictKeyAnalysis) -> DictKeyAnalysis:
        result = DictKeyAnalysis()
        result.key_domain = self.key_domain.join(other.key_domain)
        result.str_domain = self.str_domain.join(other.str_domain)
        return result

    def meet(self, other: DictKeyAnalysis) -> DictKeyAnalysis:
        result = DictKeyAnalysis()
        result.key_domain = self.key_domain.meet(other.key_domain)
        result.str_domain = self.str_domain.meet(other.str_domain)
        return result

    def widen(self, other: DictKeyAnalysis) -> DictKeyAnalysis:
        result = DictKeyAnalysis()
        result.key_domain = self.key_domain.widen(other.key_domain)
        result.str_domain = self.str_domain.widen(other.str_domain)
        return result

    def __str__(self) -> str:
        return f"DictKeys:\n  keys: {self.key_domain}\n  strings: {self.str_domain}"


# ---------------------------------------------------------------------------
# StringFormatAnalysis
# ---------------------------------------------------------------------------

class FormatSpecKind(Enum):
    STRING = auto()
    INT = auto()
    FLOAT = auto()
    REPR = auto()
    DICT_KEY = auto()
    ATTR = auto()
    INDEX = auto()


@dataclass
class FormatSpec:
    kind: FormatSpecKind
    key: Optional[str] = None  # for {key} or {0[key]}
    format_spec: Optional[str] = None  # for :.2f etc.


class StringFormatAnalysis:
    """Analyzes format strings for type information.
    Examples:
      f"Value: {x:.2f}" implies x is numeric
      "{0[key]}" implies arg is dict-like with 'key'
    """

    def __init__(self) -> None:
        self._format_cache: Dict[str, List[FormatSpec]] = {}

    def analyze_format_string(self, template: str) -> List[FormatSpec]:
        """Parse a format string and extract type information for each placeholder."""
        if template in self._format_cache:
            return self._format_cache[template]
        specs: List[FormatSpec] = []
        # Simple format string parser
        i = 0
        while i < len(template):
            if template[i] == '{':
                if i + 1 < len(template) and template[i + 1] == '{':
                    i += 2
                    continue
                # Find matching }
                j = template.find('}', i)
                if j < 0:
                    break
                field = template[i + 1:j]
                spec = self._parse_field(field)
                specs.append(spec)
                i = j + 1
            else:
                i += 1
        self._format_cache[template] = specs
        return specs

    def _parse_field(self, field: str) -> FormatSpec:
        """Parse a single format field like 'x:.2f' or '0[key]'."""
        # Split on ':' to separate field name from format spec
        parts = field.split(':', 1)
        field_name = parts[0].strip()
        fmt = parts[1].strip() if len(parts) > 1 else None

        # Check for dict-like access: {0[key]}
        bracket_match = re.match(r'(\w*)\[(\w+)\]', field_name)
        if bracket_match:
            key = bracket_match.group(2)
            return FormatSpec(FormatSpecKind.DICT_KEY, key=key, format_spec=fmt)

        # Check for attribute access: {0.attr}
        dot_match = re.match(r'(\w+)\.(\w+)', field_name)
        if dot_match:
            attr = dot_match.group(2)
            return FormatSpec(FormatSpecKind.ATTR, key=attr, format_spec=fmt)

        # Determine type from format spec
        if fmt:
            kind = self._infer_type_from_format(fmt)
            return FormatSpec(kind, format_spec=fmt)

        return FormatSpec(FormatSpecKind.STRING, format_spec=fmt)

    @staticmethod
    def _infer_type_from_format(fmt: str) -> FormatSpecKind:
        """Infer the type from a format specifier."""
        if not fmt:
            return FormatSpecKind.STRING
        # Last character is the type code
        type_char = fmt[-1] if fmt else ''
        if type_char in ('d', 'n', 'o', 'x', 'X', 'b'):
            return FormatSpecKind.INT
        if type_char in ('f', 'F', 'e', 'E', 'g', 'G', '%'):
            return FormatSpecKind.FLOAT
        if type_char in ('r',):
            return FormatSpecKind.REPR
        if type_char in ('s',):
            return FormatSpecKind.STRING
        return FormatSpecKind.STRING

    def get_type_constraints(self, template: str) -> Dict[int, FormatSpecKind]:
        """Get type constraints for positional arguments."""
        specs = self.analyze_format_string(template)
        constraints: Dict[int, FormatSpecKind] = {}
        for i, spec in enumerate(specs):
            constraints[i] = spec.kind
        return constraints

    def get_key_accesses(self, template: str) -> List[str]:
        """Get dict key accesses from format string."""
        specs = self.analyze_format_string(template)
        return [s.key for s in specs if s.kind == FormatSpecKind.DICT_KEY and s.key]

    def get_attr_accesses(self, template: str) -> List[str]:
        """Get attribute accesses from format string."""
        specs = self.analyze_format_string(template)
        return [s.key for s in specs if s.kind == FormatSpecKind.ATTR and s.key]

    def implies_numeric(self, template: str, position: int) -> bool:
        """Check if position in format string implies numeric type."""
        specs = self.analyze_format_string(template)
        if position < len(specs):
            return specs[position].kind in (FormatSpecKind.INT, FormatSpecKind.FLOAT)
        return False


# ---------------------------------------------------------------------------
# StringConcatenationAnalysis
# ---------------------------------------------------------------------------

class ConcatPart:
    """A part of a string concatenation."""
    pass

@dataclass
class LiteralPart(ConcatPart):
    value: str

@dataclass
class VarPart(ConcatPart):
    var_name: str

@dataclass
class ConcatChain:
    """Tracks a chain of string concatenations."""
    parts: List[ConcatPart]

    @staticmethod
    def from_literal(s: str) -> ConcatChain:
        return ConcatChain([LiteralPart(s)])

    @staticmethod
    def from_var(var: str) -> ConcatChain:
        return ConcatChain([VarPart(var)])

    def concat(self, other: ConcatChain) -> ConcatChain:
        # Merge adjacent literals
        parts = list(self.parts)
        for p in other.parts:
            if parts and isinstance(parts[-1], LiteralPart) and isinstance(p, LiteralPart):
                parts[-1] = LiteralPart(parts[-1].value + p.value)
            else:
                parts.append(p)
        return ConcatChain(parts)

    def variables(self) -> Set[str]:
        return {p.var_name for p in self.parts if isinstance(p, VarPart)}

    def is_all_literal(self) -> bool:
        return all(isinstance(p, LiteralPart) for p in self.parts)

    def evaluate(self) -> Optional[str]:
        if self.is_all_literal():
            return "".join(p.value for p in self.parts if isinstance(p, LiteralPart))
        return None

    def __str__(self) -> str:
        parts_str = []
        for p in self.parts:
            if isinstance(p, LiteralPart):
                parts_str.append(repr(p.value))
            else:
                parts_str.append(p.var_name)
        return " + ".join(parts_str)


class StringConcatenationAnalysis:
    """Tracks string building patterns (concatenation chains)."""

    def __init__(self) -> None:
        self._chains: Dict[str, ConcatChain] = {}

    def copy(self) -> StringConcatenationAnalysis:
        result = StringConcatenationAnalysis()
        result._chains = dict(self._chains)
        return result

    def assign_literal(self, var: str, value: str) -> None:
        self._chains[var] = ConcatChain.from_literal(value)

    def assign_var(self, dst: str, src: str) -> None:
        if src in self._chains:
            self._chains[dst] = self._chains[src]
        else:
            self._chains[dst] = ConcatChain.from_var(src)

    def concat(self, target: str, left: str, right: str) -> None:
        l_chain = self._chains.get(left, ConcatChain.from_var(left))
        r_chain = self._chains.get(right, ConcatChain.from_var(right))
        self._chains[target] = l_chain.concat(r_chain)

    def concat_literal(self, target: str, var: str, literal: str) -> None:
        v_chain = self._chains.get(var, ConcatChain.from_var(var))
        self._chains[target] = v_chain.concat(ConcatChain.from_literal(literal))

    def get_chain(self, var: str) -> Optional[ConcatChain]:
        return self._chains.get(var)

    def evaluate(self, var: str) -> Optional[str]:
        chain = self._chains.get(var)
        if chain is None:
            return None
        return chain.evaluate()

    def get_variables_in_chain(self, var: str) -> Set[str]:
        chain = self._chains.get(var)
        if chain is None:
            return set()
        return chain.variables()

    def forget(self, var: str) -> None:
        self._chains.pop(var, None)

    def join(self, other: StringConcatenationAnalysis) -> StringConcatenationAnalysis:
        """Join: keep chains that are identical in both."""
        result = StringConcatenationAnalysis()
        for var in set(self._chains.keys()) & set(other._chains.keys()):
            if str(self._chains[var]) == str(other._chains[var]):
                result._chains[var] = self._chains[var]
        return result

    def __str__(self) -> str:
        parts = []
        for v in sorted(self._chains.keys()):
            parts.append(f"{v} = {self._chains[v]}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# StringInterning
# ---------------------------------------------------------------------------

class StringInterning:
    """Optimization for commonly used strings.
    Caches string abstract values to avoid redundant computation."""

    def __init__(self, max_cache_size: int = 1000) -> None:
        self._cache: Dict[str, StringValue] = {}
        self._access_count: Dict[str, int] = {}
        self.max_cache_size = max_cache_size

    def intern(self, s: str) -> StringValue:
        """Get or create the abstract value for a string constant."""
        if s in self._cache:
            self._access_count[s] = self._access_count.get(s, 0) + 1
            return self._cache[s]
        val = StringValue.from_constant(s)
        self._cache[s] = val
        self._access_count[s] = 1
        if len(self._cache) > self.max_cache_size:
            self._evict()
        return val

    def intern_set(self, strings: Set[str]) -> StringValue:
        """Get or create the abstract value for a set of strings."""
        key = "|".join(sorted(strings))
        if key in self._cache:
            self._access_count[key] = self._access_count.get(key, 0) + 1
            return self._cache[key]
        val = StringValue.from_set(strings)
        self._cache[key] = val
        self._access_count[key] = 1
        if len(self._cache) > self.max_cache_size:
            self._evict()
        return val

    def _evict(self) -> None:
        """Evict least-accessed entries."""
        if not self._access_count:
            return
        entries = sorted(self._access_count.items(), key=lambda x: x[1])
        to_remove = len(self._cache) - self.max_cache_size // 2
        for key, _ in entries[:to_remove]:
            self._cache.pop(key, None)
            del self._access_count[key]

    def cache_size(self) -> int:
        return len(self._cache)

    def hit_rate(self) -> float:
        total = sum(self._access_count.values())
        hits = total - len(self._access_count)
        return hits / total if total > 0 else 0.0

    def clear(self) -> None:
        self._cache.clear()
        self._access_count.clear()

    def most_common(self, n: int = 10) -> List[Tuple[str, int]]:
        return sorted(self._access_count.items(), key=lambda x: -x[1])[:n]


# ---------------------------------------------------------------------------
# StringStatistics
# ---------------------------------------------------------------------------

@dataclass
class StringStatistics:
    """Domain operation statistics for the string domain."""
    join_count: int = 0
    meet_count: int = 0
    widen_count: int = 0
    narrow_count: int = 0
    concat_count: int = 0
    split_count: int = 0
    format_count: int = 0
    hasattr_checks: int = 0
    dict_key_checks: int = 0
    regex_ops: int = 0
    interning_hits: int = 0
    interning_misses: int = 0
    max_set_size: int = 0
    total_strings_tracked: int = 0
    widen_to_top_count: int = 0

    def record_join(self) -> None:
        self.join_count += 1

    def record_meet(self) -> None:
        self.meet_count += 1

    def record_widen(self, went_to_top: bool = False) -> None:
        self.widen_count += 1
        if went_to_top:
            self.widen_to_top_count += 1

    def record_narrow(self) -> None:
        self.narrow_count += 1

    def record_concat(self) -> None:
        self.concat_count += 1

    def record_split(self) -> None:
        self.split_count += 1

    def record_format(self) -> None:
        self.format_count += 1

    def record_hasattr_check(self) -> None:
        self.hasattr_checks += 1

    def record_dict_key_check(self) -> None:
        self.dict_key_checks += 1

    def record_regex_op(self) -> None:
        self.regex_ops += 1

    def record_set_size(self, size: int) -> None:
        self.max_set_size = max(self.max_set_size, size)
        self.total_strings_tracked += size

    def record_interning(self, hit: bool) -> None:
        if hit:
            self.interning_hits += 1
        else:
            self.interning_misses += 1

    def summary(self) -> str:
        return (
            f"String Domain Statistics:\n"
            f"  joins={self.join_count}, meets={self.meet_count}, "
            f"widens={self.widen_count}, narrows={self.narrow_count}\n"
            f"  concats={self.concat_count}, splits={self.split_count}, "
            f"formats={self.format_count}\n"
            f"  hasattr_checks={self.hasattr_checks}, "
            f"dict_key_checks={self.dict_key_checks}, "
            f"regex_ops={self.regex_ops}\n"
            f"  max_set_size={self.max_set_size}, "
            f"widen_to_top={self.widen_to_top_count}\n"
            f"  interning: hits={self.interning_hits}, "
            f"misses={self.interning_misses}"
        )

    def reset(self) -> None:
        self.join_count = 0
        self.meet_count = 0
        self.widen_count = 0
        self.narrow_count = 0
        self.concat_count = 0
        self.split_count = 0
        self.format_count = 0
        self.hasattr_checks = 0
        self.dict_key_checks = 0
        self.regex_ops = 0
        self.interning_hits = 0
        self.interning_misses = 0
        self.max_set_size = 0
        self.total_strings_tracked = 0
        self.widen_to_top_count = 0


class InstrumentedStringDomain:
    """String domain wrapper that records statistics."""

    def __init__(self) -> None:
        self._inner = StringDomain()
        self.stats = StringStatistics()

    def bottom(self) -> StringValue:
        return self._inner.bottom()

    def top(self) -> StringValue:
        return self._inner.top()

    def join(self, a: StringValue, b: StringValue) -> StringValue:
        self.stats.record_join()
        return self._inner.join(a, b)

    def meet(self, a: StringValue, b: StringValue) -> StringValue:
        self.stats.record_meet()
        return self._inner.meet(a, b)

    def widen(self, a: StringValue, b: StringValue) -> StringValue:
        self.stats.record_widen()
        result = self._inner.widen(a, b)
        if result.is_top():
            self.stats.record_widen(True)
        return result

    def narrow(self, a: StringValue, b: StringValue) -> StringValue:
        self.stats.record_narrow()
        return self._inner.narrow(a, b)

    def concatenate(self, a: StringValue, b: StringValue) -> StringValue:
        self.stats.record_concat()
        return self._inner.concatenate(a, b)

    def split(self, v: StringValue, sep: str) -> Tuple[StringValue, StringLength]:
        self.stats.record_split()
        return self._inner.split(v, sep)

    def format_string(self, template: str, args: Dict[str, StringValue]) -> StringValue:
        self.stats.record_format()
        return self._inner.format_string(template, args)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_string_constant() -> None:
    """Test StringConstant."""
    c = StringConstant("hello")
    assert len(c) == 5
    assert c.startswith("hel")
    assert c.endswith("llo")
    assert c.contains("ell")
    assert not c.contains("xyz")
    assert c.prefix(3) == "hel"
    assert c.suffix(3) == "llo"
    assert c.chars() == frozenset("helo")
    print("  test_string_constant: PASSED")


def test_string_set() -> None:
    """Test StringSet operations."""
    a = StringSet.from_list(["foo", "bar"])
    b = StringSet.from_list(["bar", "baz"])
    assert a.contains("foo")
    assert not a.contains("baz")
    j = a.join(b)
    assert j.contains("foo") and j.contains("baz")
    m = a.meet(b)
    assert m.contains("bar")
    assert not m.contains("foo")
    assert StringSet.bottom().is_bottom()
    assert StringSet.top_value().is_top
    assert a.leq(j)
    assert not j.leq(a)
    s = a.subtract("foo")
    assert not s.contains("foo")
    assert s.contains("bar")
    w = a.widen(b, threshold=2)
    assert w.is_top  # exceeded threshold
    print("  test_string_set: PASSED")


def test_string_prefix() -> None:
    """Test StringPrefix."""
    a = StringPrefix.from_string("hello")
    b = StringPrefix.from_string("help")
    j = a.join(b)
    assert j.prefix == "hel"
    assert j.matches("helmet")
    m = a.meet(b)
    assert m.is_bottom()  # neither is prefix of the other
    assert StringPrefix.top().is_top()
    assert StringPrefix.bottom().is_bottom()
    c = StringPrefix.from_string("hel")
    m2 = a.meet(c)
    assert m2.prefix == "hello"  # "hello" starts with "hel"
    print("  test_string_prefix: PASSED")


def test_string_suffix() -> None:
    """Test StringSuffix."""
    a = StringSuffix.from_string("ing")
    b = StringSuffix.from_string("ring")
    j = a.join(b)
    assert j.suffix == "ing"
    assert j.matches("testing")
    c = StringSuffix.from_string("ed")
    j2 = a.join(c)
    assert j2.is_top()
    print("  test_string_suffix: PASSED")


def test_string_length() -> None:
    """Test StringLength."""
    a = StringLength.exact(5)
    b = StringLength.range(3, 8)
    j = a.join(b)
    assert j.min_len == 3 and j.max_len == 8
    m = a.meet(b)
    assert m.min_len == 5 and m.max_len == 5
    assert a.contains(5) and not a.contains(4)
    c = StringLength.at_least(3)
    assert c.contains(100)
    add_result = a.add(b)
    assert add_result.min_len == 8 and add_result.max_len == 13
    w = a.widen(StringLength.range(5, 10))
    assert w.max_len == -1  # widened to infinity
    print("  test_string_length: PASSED")


def test_string_charset() -> None:
    """Test StringCharSet."""
    a = StringCharSet.from_string("hello")
    assert 'h' in a.all_chars
    assert 'z' not in a.all_chars
    b = StringCharSet.from_string("world")
    j = a.join(b)
    assert 'h' in j.all_chars and 'w' in j.all_chars
    m = a.meet(b)
    # Position-wise meet: 'hello' and 'world' share no chars at same positions
    # so meet is bottom
    assert m.is_bottom()
    print("  test_string_charset: PASSED")


def test_string_pattern() -> None:
    """Test StringPattern."""
    a = StringPattern.from_pattern(r"\d+")
    assert a.matches("123")
    assert not a.matches("abc")
    b = StringPattern.literal("hello")
    assert b.matches("hello")
    assert not b.matches("world")
    j = a.join(b)
    assert j.is_top()
    print("  test_string_pattern: PASSED")


def test_string_value() -> None:
    """Test StringValue."""
    v = StringValue.from_constant("hello")
    assert not v.is_bottom()
    assert not v.is_top()
    assert v.string_set.contains("hello")
    assert v.length == StringLength.exact(5)
    v2 = StringValue.from_constant("help")
    j = v.join(v2)
    assert j.string_set.contains("hello") and j.string_set.contains("help")
    m = v.meet(v2)
    assert m.is_bottom()
    s = StringValue.from_set({"foo", "bar"})
    assert s.string_set.size() == 2
    print("  test_string_value: PASSED")


def test_string_domain_basic() -> None:
    """Test basic StringDomain operations."""
    d = StringDomain()
    v = StringValue.from_constant("hello")
    eq = d.assume_eq(v, "hello")
    assert eq.string_set.contains("hello")
    ne = d.assume_ne(v, "hello")
    assert ne.is_bottom()
    print("  test_string_domain_basic: PASSED")


def test_string_concatenation() -> None:
    """Test string concatenation."""
    d = StringDomain()
    a = StringValue.from_constant("hello")
    b = StringValue.from_constant(" world")
    c = d.concatenate(a, b)
    assert c.string_set.contains("hello world")
    a2 = StringValue.from_set({"hi", "hey"})
    b2 = StringValue.from_constant("!")
    c2 = d.concatenate(a2, b2)
    assert c2.string_set.contains("hi!")
    assert c2.string_set.contains("hey!")
    print("  test_string_concatenation: PASSED")


def test_string_split() -> None:
    """Test string split."""
    d = StringDomain()
    v = StringValue.from_constant("a,b,c")
    elem, count = d.split(v, ",")
    assert count.min_len == 3 and count.max_len == 3
    assert elem.string_set.contains("a")
    assert elem.string_set.contains("b")
    print("  test_string_split: PASSED")


def test_string_startswith_endswith() -> None:
    """Test startswith/endswith."""
    d = StringDomain()
    v = StringValue.from_set({"hello", "help", "world"})
    true_v, false_v = d.startswith(v, "hel")
    assert true_v.string_set.contains("hello")
    assert true_v.string_set.contains("help")
    assert not true_v.string_set.contains("world")
    assert false_v.string_set.contains("world")
    true_e, false_e = d.endswith(v, "ld")
    assert true_e.string_set.contains("world")
    assert not true_e.string_set.contains("hello")
    print("  test_string_startswith_endswith: PASSED")


def test_string_contains() -> None:
    """Test contains."""
    d = StringDomain()
    v = StringValue.from_set({"hello world", "goodbye", "help"})
    true_v, false_v = d.contains_sub(v, "ell")
    assert true_v.string_set.contains("hello world")
    assert not true_v.string_set.contains("goodbye")
    print("  test_string_contains: PASSED")


def test_string_case_ops() -> None:
    """Test case operations."""
    d = StringDomain()
    v = StringValue.from_constant("Hello")
    assert d.upper(v).string_set.contains("HELLO")
    assert d.lower(v).string_set.contains("hello")
    v2 = StringValue.from_constant("  hi  ")
    assert d.strip(v2).string_set.contains("hi")
    print("  test_string_case_ops: PASSED")


def test_string_replace() -> None:
    """Test replace."""
    d = StringDomain()
    v = StringValue.from_constant("hello world")
    r = d.replace(v, "world", "there")
    assert r.string_set.contains("hello there")
    print("  test_string_replace: PASSED")


def test_string_find() -> None:
    """Test find."""
    d = StringDomain()
    v = StringValue.from_set({"hello", "world"})
    may, must = d.find(v, "llo")
    assert may  # "hello" contains "llo"
    assert not must  # "world" doesn't
    print("  test_string_find: PASSED")


def test_string_count() -> None:
    """Test count."""
    d = StringDomain()
    v = StringValue.from_set({"aabaa", "ab"})
    c = d.count_sub(v, "a")
    assert c.min_len == 1  # "ab" has 1 'a'
    assert c.max_len == 4  # "aabaa" has 4 'a's
    print("  test_string_count: PASSED")


def test_string_format() -> None:
    """Test format string."""
    d = StringDomain()
    args = {"name": StringValue.from_set({"Alice", "Bob"})}
    result = d.format_string("Hello, {name}!", args)
    assert result.string_set.contains("Hello, Alice!")
    assert result.string_set.contains("Hello, Bob!")
    print("  test_string_format: PASSED")


def test_string_join() -> None:
    """Test string join."""
    d = StringDomain()
    sep = StringValue.from_constant(",")
    parts = StringValue.from_set({"a", "b"})
    result = d.join_strings(sep, parts, StringLength.exact(2))
    assert result.string_set.contains("a,b")
    assert result.string_set.contains("b,a")
    print("  test_string_join: PASSED")


def test_string_equality_domain() -> None:
    """Test StringEqualityDomain."""
    d = StringEqualityDomain()
    d.assign_const("x", "hello")
    assert d.get("x").contains("hello")
    assert not d.get("x").contains("world")
    ok = d.assume_eq("x", "hello")
    assert ok
    fail = d.assume_eq("x", "world")
    assert not fail
    d.assign_top("y")
    assert d.get("y").is_top
    d.assume_ne("x", "hello")
    assert d.get("x").is_bottom()
    print("  test_string_equality_domain: PASSED")


def test_string_equality_join_meet() -> None:
    """Test join and meet of equality domains."""
    a = StringEqualityDomain()
    a.assign_const("x", "hello")
    b = StringEqualityDomain()
    b.assign_const("x", "world")
    j = a.join(b)
    assert j.get("x").contains("hello")
    assert j.get("x").contains("world")
    m = a.meet(b)
    assert m.get("x").is_bottom()
    print("  test_string_equality_join_meet: PASSED")


def test_string_equality_widen() -> None:
    """Test widening of equality domain."""
    a = StringEqualityDomain(widen_threshold=3)
    a.set("x", StringSet.from_list(["a", "b"]))
    b = StringEqualityDomain(widen_threshold=3)
    b.set("x", StringSet.from_list(["a", "b", "c", "d"]))
    w = a.widen(b)
    assert w.get("x").is_top
    print("  test_string_equality_widen: PASSED")


def test_dict_key_domain() -> None:
    """Test DictKeyDomain."""
    d = DictKeyDomain()
    d.create_dict("d", {"a", "b", "c"})
    assert d.has_key("d", "a") is True
    assert d.has_key("d", "z") is False
    d.add_key("d", "z")
    assert d.has_key("d", "z") is True
    d.definitely_remove_key("d", "a")
    assert d.has_key("d", "a") is False
    print("  test_dict_key_domain: PASSED")


def test_dict_key_join_meet() -> None:
    """Test DictKeyDomain join and meet."""
    a = DictKeyDomain()
    a.create_dict("d", {"a", "b"})
    b = DictKeyDomain()
    b.create_dict("d", {"b", "c"})
    j = a.join(b)
    assert j.has_key("d", "b") is True  # definite in both
    assert j.has_key("d", "a") is None  # possible but not definite
    m = a.meet(b)
    assert m.has_key("d", "a") is True  # definite in at least one
    assert m.has_key("d", "c") is True
    print("  test_dict_key_join_meet: PASSED")


def test_dict_key_assume() -> None:
    """Test DictKeyDomain assume."""
    d = DictKeyDomain()
    d.create_dict("d", {"a", "b"})
    d_true = d.assume_has_key("d", "c")
    assert d_true.has_key("d", "c") is True
    d_false = d.assume_no_key("d", "a")
    assert d_false.has_key("d", "a") is False
    print("  test_dict_key_assume: PASSED")


def test_attribute_presence_domain() -> None:
    """Test AttributePresenceDomain."""
    d = AttributePresenceDomain()
    d.create_object("obj", {"x", "y"}, type_tag="MyClass")
    assert d.has_attr("obj", "x") is True
    assert d.has_attr("obj", "z") is False
    assert d.get_type_tag("obj") == "MyClass"
    d.add_attr("obj", "z")
    assert d.has_attr("obj", "z") is True
    d.remove_attr("obj", "x")
    assert d.has_attr("obj", "x") is None  # now only possible, not definite
    print("  test_attribute_presence_domain: PASSED")


def test_attribute_presence_join() -> None:
    """Test AttributePresenceDomain join."""
    a = AttributePresenceDomain()
    a.create_object("obj", {"x", "y"}, "MyClass")
    b = AttributePresenceDomain()
    b.create_object("obj", {"y", "z"}, "MyClass")
    j = a.join(b)
    assert j.has_attr("obj", "y") is True  # definite in both
    assert j.has_attr("obj", "x") is None  # possible, not definite
    assert j.get_type_tag("obj") == "MyClass"
    print("  test_attribute_presence_join: PASSED")


def test_hasattr_analysis() -> None:
    """Test HasAttrAnalysis."""
    ha = HasAttrAnalysis()
    ha.attr_domain.create_object("obj", {"x"})
    true_s, false_s = ha.analyze_hasattr_check("obj", "y")
    assert true_s.attr_domain.has_attr("obj", "y") is True
    assert false_s.attr_domain.has_attr("obj", "y") is False
    assert true_s.is_safe_access("obj", "y")
    assert not false_s.is_safe_access("obj", "y")
    print("  test_hasattr_analysis: PASSED")


def test_hasattr_setattr_delattr() -> None:
    """Test setattr/delattr in HasAttrAnalysis."""
    ha = HasAttrAnalysis()
    ha.attr_domain.create_object("obj", {"x"})
    ha = ha.analyze_setattr("obj", "y")
    assert ha.is_safe_access("obj", "y")
    ha = ha.analyze_delattr("obj", "y")
    assert ha.may_fail_access("obj", "y")
    print("  test_hasattr_setattr_delattr: PASSED")


def test_dict_key_analysis() -> None:
    """Test DictKeyAnalysis."""
    dka = DictKeyAnalysis()
    dka = dka.analyze_dict_literal("d", {"a", "b"})
    true_s, false_s = dka.analyze_key_check("d", "a")
    assert true_s.is_safe_access("d", "a")
    dka2 = dka.analyze_dict_setitem("d", "c")
    assert dka2.is_safe_access("d", "c")
    dka3 = dka2.analyze_dict_delitem("d", "a")
    assert dka3.may_fail_access("d", "a")
    print("  test_dict_key_analysis: PASSED")


def test_dict_key_analysis_get() -> None:
    """Test dict.get() analysis."""
    dka = DictKeyAnalysis()
    dka = dka.analyze_dict_literal("d", {"a"})
    has, _ = dka.analyze_dict_get("d", "a", has_default=False)
    assert has is True
    has2, _ = dka.analyze_dict_get("d", "z", has_default=True)
    assert has2 is False
    print("  test_dict_key_analysis_get: PASSED")


def test_dict_key_analysis_update() -> None:
    """Test dict.update()."""
    dka = DictKeyAnalysis()
    dka = dka.analyze_dict_literal("d1", {"a"})
    dka = dka.analyze_dict_literal("d2", {"b"})
    dka = dka.analyze_dict_update("d1", "d2")
    assert dka.is_safe_access("d1", "a")
    assert dka.is_safe_access("d1", "b")
    print("  test_dict_key_analysis_update: PASSED")


def test_dict_key_analysis_pop() -> None:
    """Test dict.pop()."""
    dka = DictKeyAnalysis()
    dka = dka.analyze_dict_literal("d", {"a", "b"})
    dka = dka.analyze_dict_pop("d", "a", has_default=True)
    assert dka.may_fail_access("d", "a")
    assert dka.is_safe_access("d", "b")
    print("  test_dict_key_analysis_pop: PASSED")


def test_string_format_analysis() -> None:
    """Test StringFormatAnalysis."""
    sfa = StringFormatAnalysis()
    specs = sfa.analyze_format_string("Value: {x:.2f}, Count: {y:d}")
    assert len(specs) == 2
    assert specs[0].kind == FormatSpecKind.FLOAT
    assert specs[1].kind == FormatSpecKind.INT
    # Test dict key access
    specs2 = sfa.analyze_format_string("{0[key]}")
    assert len(specs2) == 1
    assert specs2[0].kind == FormatSpecKind.DICT_KEY
    assert specs2[0].key == "key"
    keys = sfa.get_key_accesses("{0[key]}")
    assert keys == ["key"]
    assert sfa.implies_numeric("Value: {x:.2f}", 0)
    print("  test_string_format_analysis: PASSED")


def test_string_concatenation_analysis() -> None:
    """Test StringConcatenationAnalysis."""
    sca = StringConcatenationAnalysis()
    sca.assign_literal("a", "hello")
    sca.assign_literal("b", " world")
    sca.concat("c", "a", "b")
    chain = sca.get_chain("c")
    assert chain is not None
    result = chain.evaluate()
    assert result == "hello world"
    sca.concat_literal("d", "c", "!")
    assert sca.evaluate("d") == "hello world!"
    print("  test_string_concatenation_analysis: PASSED")


def test_string_concatenation_with_vars() -> None:
    """Test concatenation analysis with variables."""
    sca = StringConcatenationAnalysis()
    sca.assign_literal("prefix", "key_")
    sca.assign_var("suffix", "x")  # x is unknown
    sca.concat("result", "prefix", "suffix")
    chain = sca.get_chain("result")
    assert chain is not None
    assert chain.evaluate() is None  # has variable part
    assert "x" in chain.variables()
    print("  test_string_concatenation_with_vars: PASSED")


def test_string_interning() -> None:
    """Test StringInterning."""
    si = StringInterning(max_cache_size=5)
    v1 = si.intern("hello")
    v2 = si.intern("hello")
    assert v1.string_set.contains("hello")
    assert v2.string_set.contains("hello")
    v3 = si.intern_set({"a", "b"})
    assert v3.string_set.contains("a")
    assert si.cache_size() == 2
    for i in range(10):
        si.intern(f"str_{i}")
    assert si.cache_size() <= 5
    common = si.most_common(3)
    assert len(common) <= 3
    print("  test_string_interning: PASSED")


def test_string_statistics() -> None:
    """Test StringStatistics."""
    stats = StringStatistics()
    stats.record_join()
    stats.record_meet()
    stats.record_concat()
    stats.record_hasattr_check()
    stats.record_dict_key_check()
    stats.record_set_size(10)
    assert stats.join_count == 1
    assert stats.meet_count == 1
    assert stats.concat_count == 1
    assert stats.max_set_size == 10
    summary = stats.summary()
    assert "joins=1" in summary
    stats.reset()
    assert stats.join_count == 0
    print("  test_string_statistics: PASSED")


def test_instrumented_string_domain() -> None:
    """Test InstrumentedStringDomain."""
    d = InstrumentedStringDomain()
    a = StringValue.from_constant("hello")
    b = StringValue.from_constant("world")
    c = d.join(a, b)
    d.meet(a, b)
    d.concatenate(a, b)
    assert d.stats.join_count == 1
    assert d.stats.meet_count == 1
    assert d.stats.concat_count == 1
    print("  test_instrumented_string_domain: PASSED")


def test_string_transfer_functions() -> None:
    """Test StringTransferFunctions."""
    tf = StringTransferFunctions()
    env: Dict[str, StringValue] = {
        "a": StringValue.from_constant("hello"),
        "b": StringValue.from_constant(" world"),
    }
    # Concat
    op = StringOp(kind=StringOpKind.CONCAT, target="c", args=["a", "b"])
    env = tf.transfer(env, op)
    assert env["c"].string_set.contains("hello world")
    # Upper
    op2 = StringOp(kind=StringOpKind.UPPER, target="d", operand="a")
    env = tf.transfer(env, op2)
    assert env["d"].string_set.contains("HELLO")
    # Split
    env["csv"] = StringValue.from_constant("x,y,z")
    op3 = StringOp(kind=StringOpKind.SPLIT, target="parts", operand="csv", const_arg=",")
    env = tf.transfer(env, op3)
    assert env["parts"].string_set.contains("x")
    # Replace
    op4 = StringOp(kind=StringOpKind.REPLACE, target="r", operand="a",
                   const_args=["hello", "hi"])
    env = tf.transfer(env, op4)
    assert env["r"].string_set.contains("hi")
    print("  test_string_transfer_functions: PASSED")


def test_string_transfer_regex() -> None:
    """Test regex transfer functions."""
    tf = StringTransferFunctions()
    env: Dict[str, StringValue] = {
        "s": StringValue.from_set({"123", "abc", "456"}),
    }
    env = tf.transfer_regex_match(env, "m", "s", r"\d+")
    assert env["m"].string_set.contains("123")
    assert env["m"].string_set.contains("456")
    assert not env["m"].string_set.contains("abc")
    env = tf.transfer_regex_findall(env, "f", "s", r"\d")
    print("  test_string_transfer_regex: PASSED")


def test_string_transfer_comparison() -> None:
    """Test string comparison transfer."""
    tf = StringTransferFunctions()
    env: Dict[str, StringValue] = {
        "x": StringValue.from_set({"hello", "world", "hi"}),
    }
    true_env, false_env = tf.transfer_comparison(env, "x", "==", "hello")
    assert true_env["x"].string_set.contains("hello")
    assert true_env["x"].string_set.size() == 1
    assert false_env["x"].string_set.contains("world")
    assert not false_env["x"].string_set.contains("hello")
    print("  test_string_transfer_comparison: PASSED")


def test_string_value_reduce() -> None:
    """Test StringValue reduction."""
    v = StringValue(
        StringSet.from_set({"hello", "world", "hi"}),
        StringPrefix.from_string("h"),
        StringSuffix.top(),
        StringLength.range(2, 5),
        StringCharSet.top(),
        StringPattern.top(),
    )
    reduced = v.reduce()
    # "world" should be filtered out (doesn't start with "h")
    assert not reduced.string_set.contains("world")
    assert reduced.string_set.contains("hello")
    assert reduced.string_set.contains("hi")
    print("  test_string_value_reduce: PASSED")


def test_string_value_singleton_reduce() -> None:
    """Test that singleton sets tighten all components."""
    v = StringValue(
        StringSet.singleton("test"),
        StringPrefix.top(),
        StringSuffix.top(),
        StringLength.top(),
        StringCharSet.top(),
        StringPattern.top(),
    )
    reduced = v.reduce()
    assert reduced.length == StringLength.exact(4)
    assert reduced.prefix.prefix == "test"
    print("  test_string_value_singleton_reduce: PASSED")


def test_string_domain_substring() -> None:
    """Test substring operation."""
    d = StringDomain()
    v = StringValue.from_set({"hello", "world"})
    sub = d.substring(v, 0, 3)
    assert sub.string_set.contains("hel")
    assert sub.string_set.contains("wor")
    print("  test_string_domain_substring: PASSED")


def test_string_int_conversion() -> None:
    """Test int/float conversion checks."""
    d = StringDomain()
    v1 = StringValue.from_set({"123", "456"})
    assert d.int_from_str(v1)
    v2 = StringValue.from_set({"abc"})
    assert not d.int_from_str(v2)
    v3 = StringValue.from_set({"3.14"})
    assert d.float_from_str(v3)
    print("  test_string_int_conversion: PASSED")


def test_hasattr_join_widen() -> None:
    """Test HasAttrAnalysis join and widen."""
    ha1 = HasAttrAnalysis()
    ha1.attr_domain.create_object("obj", {"x", "y"})
    ha2 = HasAttrAnalysis()
    ha2.attr_domain.create_object("obj", {"y", "z"})
    j = ha1.join(ha2)
    assert j.attr_domain.has_attr("obj", "y") is True
    assert j.attr_domain.has_attr("obj", "x") is None
    w = ha1.widen(ha2)
    assert w.attr_domain.has_attr("obj", "y") is True
    print("  test_hasattr_join_widen: PASSED")


def test_dict_key_analysis_join_widen() -> None:
    """Test DictKeyAnalysis join and widen."""
    a = DictKeyAnalysis()
    a = a.analyze_dict_literal("d", {"x"})
    b = DictKeyAnalysis()
    b = b.analyze_dict_literal("d", {"y"})
    j = a.join(b)
    assert j.key_domain.has_key("d", "x") is None
    assert j.key_domain.has_key("d", "y") is None
    w = a.widen(b)
    print("  test_dict_key_analysis_join_widen: PASSED")


def test_dict_key_intersect_string() -> None:
    """Test intersecting dict keys with string domain."""
    dkd = DictKeyDomain()
    dkd.create_dict("d", {"a", "b", "c"})
    sd = StringEqualityDomain()
    sd.set("key", StringSet.from_list(["a", "b"]))
    result = dkd.intersect_with_string_domain("d", "key", sd)
    info = result.get("d")
    assert "c" not in info.possible_keys
    print("  test_dict_key_intersect_string: PASSED")


def test_string_format_attr_access() -> None:
    """Test format string attribute access detection."""
    sfa = StringFormatAnalysis()
    attrs = sfa.get_attr_accesses("{obj.name}")
    assert "name" in attrs
    specs = sfa.analyze_format_string("{obj.name}")
    assert len(specs) == 1
    assert specs[0].kind == FormatSpecKind.ATTR
    print("  test_string_format_attr_access: PASSED")


def test_concat_chain_join() -> None:
    """Test ConcatChain join."""
    sca1 = StringConcatenationAnalysis()
    sca1.assign_literal("x", "hello")
    sca2 = StringConcatenationAnalysis()
    sca2.assign_literal("x", "hello")
    j = sca1.join(sca2)
    assert j.evaluate("x") == "hello"
    sca3 = StringConcatenationAnalysis()
    sca3.assign_literal("x", "world")
    j2 = sca1.join(sca3)
    assert j2.get_chain("x") is None
    print("  test_concat_chain_join: PASSED")


def test_string_domain_encode_decode() -> None:
    """Test encode/decode."""
    d = StringDomain()
    v = StringValue.from_constant("hello")
    e = d.encode(v)
    assert e.string_set.contains("hello")
    dec = d.decode(e)
    assert dec.string_set.contains("hello")
    print("  test_string_domain_encode_decode: PASSED")


def test_string_set_map_filter() -> None:
    """Test StringSet map and filter."""
    ss = StringSet.from_list(["hello", "world", "hi"])
    upper = ss.map(str.upper)
    assert upper.contains("HELLO")
    short = ss.filter(lambda s: len(s) <= 3)
    assert short.contains("hi")
    assert not short.contains("hello")
    print("  test_string_set_map_filter: PASSED")


def test_string_length_narrow() -> None:
    """Test StringLength narrowing."""
    a = StringLength.top()
    b = StringLength.range(3, 7)
    n = a.narrow(b)
    assert n.min_len == 3
    assert n.max_len == 7
    print("  test_string_length_narrow: PASSED")


def test_string_value_widen_narrow() -> None:
    """Test StringValue widen and narrow."""
    a = StringValue.from_set({"x", "y"})
    b = StringValue.from_set({"y", "z"})
    w = a.widen(b)
    n = w.narrow(b)
    print("  test_string_value_widen_narrow: PASSED")


def test_string_value_leq() -> None:
    """Test StringValue leq."""
    a = StringValue.from_constant("hello")
    b = StringValue.from_set({"hello", "world"})
    assert a.leq(b)
    assert not b.leq(a)
    assert a.leq(StringValue.top())
    assert StringValue.bottom().leq(a)
    print("  test_string_value_leq: PASSED")


def test_dict_key_info_widen() -> None:
    """Test DictKeyInfo widening."""
    a = DictKeyInfo.from_keys({"a"})
    keys = {f"key_{i}" for i in range(100)}
    b = DictKeyInfo.from_keys(keys)
    w = a.widen(b, threshold=10)
    assert w.is_top
    print("  test_dict_key_info_widen: PASSED")


def test_attribute_info_widen() -> None:
    """Test AttributeInfo widening."""
    a = AttributeInfo.from_attrs({"x"})
    attrs = {f"attr_{i}" for i in range(100)}
    b = AttributeInfo.from_attrs(attrs)
    w = a.widen(b, threshold=10)
    assert w.is_top
    print("  test_attribute_info_widen: PASSED")


def run_all_tests() -> None:
    """Run all unit tests."""
    print("Running string domain tests...")
    test_string_constant()
    test_string_set()
    test_string_prefix()
    test_string_suffix()
    test_string_length()
    test_string_charset()
    test_string_pattern()
    test_string_value()
    test_string_domain_basic()
    test_string_concatenation()
    test_string_split()
    test_string_startswith_endswith()
    test_string_contains()
    test_string_case_ops()
    test_string_replace()
    test_string_find()
    test_string_count()
    test_string_format()
    test_string_join()
    test_string_equality_domain()
    test_string_equality_join_meet()
    test_string_equality_widen()
    test_dict_key_domain()
    test_dict_key_join_meet()
    test_dict_key_assume()
    test_attribute_presence_domain()
    test_attribute_presence_join()
    test_hasattr_analysis()
    test_hasattr_setattr_delattr()
    test_dict_key_analysis()
    test_dict_key_analysis_get()
    test_dict_key_analysis_update()
    test_dict_key_analysis_pop()
    test_string_format_analysis()
    test_string_concatenation_analysis()
    test_string_concatenation_with_vars()
    test_string_interning()
    test_string_statistics()
    test_instrumented_string_domain()
    test_string_transfer_functions()
    test_string_transfer_regex()
    test_string_transfer_comparison()
    test_string_value_reduce()
    test_string_value_singleton_reduce()
    test_string_domain_substring()
    test_string_int_conversion()
    test_hasattr_join_widen()
    test_dict_key_analysis_join_widen()
    test_dict_key_intersect_string()
    test_string_format_attr_access()
    test_concat_chain_join()
    test_string_domain_encode_decode()
    test_string_set_map_filter()
    test_string_length_narrow()
    test_string_value_widen_narrow()
    test_string_value_leq()
    test_dict_key_info_widen()
    test_attribute_info_widen()
    print("\nAll string domain tests passed!")


if __name__ == "__main__":
    run_all_tests()
