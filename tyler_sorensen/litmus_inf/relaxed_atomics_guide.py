"""Practical relaxed atomics guide: patterns, verification, and tutorials."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple, Set
import re


class Language(Enum):
    C = auto()
    CPP = auto()
    RUST = auto()
    JAVA = auto()
    GO = auto()
    CSHARP = auto()

    def __str__(self) -> str:
        return self.name


class OrderingLevel(Enum):
    RELAXED = 0
    CONSUME = 1
    ACQUIRE = 2
    RELEASE = 3
    ACQ_REL = 4
    SEQ_CST = 5

    def __str__(self) -> str:
        return self.name.lower().replace("_", "/")


class PatternCategory(Enum):
    COUNTER = auto()
    FLAG = auto()
    LOCK = auto()
    QUEUE = auto()
    STACK = auto()
    REFERENCE_COUNT = auto()
    SEQUENCE_LOCK = auto()
    RCU = auto()
    HAZARD_POINTER = auto()
    EPOCH_BASED = auto()

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()


@dataclass
class AtomicPattern:
    name: str
    category: PatternCategory
    description: str
    code: Dict[str, str]
    correct_orderings: Dict[str, OrderingLevel]
    common_mistakes: List[str]
    explanation: str

    def __str__(self) -> str:
        langs = ", ".join(self.code.keys())
        mistakes = "; ".join(self.common_mistakes[:2])
        return f"[{self.category}] {self.name}: {self.description} (langs: {langs}) mistakes: {mistakes}"


@dataclass
class OrderingRecommendation:
    use_case: str
    recommended: OrderingLevel
    reason: str
    alternatives: List[Tuple[OrderingLevel, str]]
    code_example: str

    def __str__(self) -> str:
        alts = ", ".join(f"{o}({r})" for o, r in self.alternatives)
        return f"Use case: {self.use_case} -> {self.recommended} ({self.reason}). Alternatives: {alts}"


@dataclass
class VerificationIssue:
    line: int
    description: str
    severity: str
    fix: str
    pattern_violated: str

    def __str__(self) -> str:
        return f"L{self.line} [{self.severity}] {self.description} (fix: {self.fix})"


@dataclass
class VerificationResult:
    source: str
    language: Language
    issues: List[VerificationIssue]
    patterns_detected: List[str]
    correct: bool
    suggestions: List[str]

    def __str__(self) -> str:
        status = "CORRECT" if self.correct else f"ISSUES({len(self.issues)})"
        pats = ", ".join(self.patterns_detected) if self.patterns_detected else "none"
        return f"Verification [{self.language}]: {status}, patterns: {pats}"


@dataclass
class TutorialSection:
    title: str
    content: str
    code_examples: List[str]
    exercises: List[str]

    def __str__(self) -> str:
        return f"## {self.title}\n{self.content}\nExamples: {len(self.code_examples)}, Exercises: {len(self.exercises)}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ordering_strength(ordering: OrderingLevel) -> int:
    """Return numeric strength for comparison."""
    return ordering.value


def _minimal_ordering(pattern: str, role: str) -> OrderingLevel:
    """Return minimal correct ordering for a pattern/role combination."""
    table: Dict[str, Dict[str, OrderingLevel]] = {
        "counter": {"increment": OrderingLevel.RELAXED, "read": OrderingLevel.RELAXED},
        "flag": {"writer": OrderingLevel.RELEASE, "reader": OrderingLevel.ACQUIRE},
        "message_passing": {"sender": OrderingLevel.RELEASE, "receiver": OrderingLevel.ACQUIRE},
        "refcount": {"increment": OrderingLevel.RELAXED, "decrement": OrderingLevel.RELEASE,
                      "dealloc_fence": OrderingLevel.ACQUIRE},
        "spinlock": {"lock": OrderingLevel.ACQUIRE, "unlock": OrderingLevel.RELEASE,
                     "trylock": OrderingLevel.ACQ_REL},
        "seqlock": {"writer_begin": OrderingLevel.RELEASE, "writer_end": OrderingLevel.RELEASE,
                    "reader_begin": OrderingLevel.ACQUIRE, "reader_end": OrderingLevel.ACQUIRE},
        "dcl": {"check": OrderingLevel.ACQUIRE, "publish": OrderingLevel.RELEASE},
        "spsc": {"producer_index": OrderingLevel.RELEASE, "consumer_index": OrderingLevel.RELEASE,
                 "read_index": OrderingLevel.ACQUIRE},
        "mpsc": {"enqueue": OrderingLevel.ACQ_REL, "dequeue": OrderingLevel.ACQ_REL},
        "ticket_lock": {"ticket": OrderingLevel.RELAXED, "serving_read": OrderingLevel.ACQUIRE,
                        "serving_write": OrderingLevel.RELEASE},
    }
    entry = table.get(pattern, {})
    return entry.get(role, OrderingLevel.SEQ_CST)


def _language_atomic_patterns(language: str) -> Dict[str, str]:
    """Return regex patterns to detect atomic operations in source for the language."""
    lang = language.upper()
    if lang in ("C",):
        return {
            "atomic_load": r"atomic_load_explicit\s*\(",
            "atomic_store": r"atomic_store_explicit\s*\(",
            "atomic_fetch_add": r"atomic_fetch_add_explicit\s*\(",
            "atomic_fetch_sub": r"atomic_fetch_sub_explicit\s*\(",
            "atomic_cas": r"atomic_compare_exchange_(weak|strong)_explicit\s*\(",
            "ordering": r"memory_order_(relaxed|consume|acquire|release|acq_rel|seq_cst)",
            "atomic_decl": r"_Atomic\s+",
            "fence": r"atomic_thread_fence\s*\(",
        }
    if lang in ("CPP", "C++"):
        return {
            "atomic_load": r"\.load\s*\(",
            "atomic_store": r"\.store\s*\(",
            "atomic_fetch_add": r"\.fetch_add\s*\(",
            "atomic_fetch_sub": r"\.fetch_sub\s*\(",
            "atomic_cas": r"\.compare_exchange_(weak|strong)\s*\(",
            "ordering": r"std::memory_order_(relaxed|consume|acquire|release|acq_rel|seq_cst)"
                        r"|memory_order_(relaxed|consume|acquire|release|acq_rel|seq_cst)",
            "atomic_decl": r"std::atomic\s*<",
            "fence": r"std::atomic_thread_fence\s*\(",
        }
    if lang == "RUST":
        return {
            "atomic_load": r"\.load\s*\(",
            "atomic_store": r"\.store\s*\(",
            "atomic_fetch_add": r"\.fetch_add\s*\(",
            "atomic_fetch_sub": r"\.fetch_sub\s*\(",
            "atomic_cas": r"\.compare_exchange(_weak)?\s*\(",
            "ordering": r"Ordering::(Relaxed|Acquire|Release|AcqRel|SeqCst)",
            "atomic_decl": r"Atomic(Usize|I32|U32|I64|U64|Bool|Ptr)\s*",
            "fence": r"(std::sync::atomic::)?fence\s*\(",
        }
    if lang == "JAVA":
        return {
            "atomic_load": r"\.(get|getAcquire|getOpaque|getPlain)\s*\(",
            "atomic_store": r"\.(set|setRelease|setOpaque|setPlain)\s*\(",
            "atomic_fetch_add": r"\.(getAndAdd|addAndGet|incrementAndGet|getAndIncrement)\s*\(",
            "atomic_fetch_sub": r"\.(getAndAdd|addAndGet|decrementAndGet|getAndDecrement)\s*\(",
            "atomic_cas": r"\.(compareAndSet|compareAndExchange|weakCompareAndSet)\s*\(",
            "ordering": r"(getAcquire|setRelease|getOpaque|setOpaque|volatile)",
            "atomic_decl": r"(AtomicInteger|AtomicLong|AtomicReference|AtomicBoolean|VarHandle)",
            "fence": r"VarHandle\.(fullFence|acquireFence|releaseFence)\s*\(",
        }
    if lang == "GO":
        return {
            "atomic_load": r"atomic\.(Load|LoadInt32|LoadInt64|LoadUint32|LoadUint64|LoadPointer)\s*\(",
            "atomic_store": r"atomic\.(Store|StoreInt32|StoreInt64|StoreUint32|StoreUint64|StorePointer)\s*\(",
            "atomic_fetch_add": r"atomic\.(AddInt32|AddInt64|AddUint32|AddUint64)\s*\(",
            "atomic_cas": r"atomic\.CompareAndSwap(Int32|Int64|Uint32|Uint64|Pointer)\s*\(",
            "ordering": r"atomic\.",
            "atomic_decl": r"(int32|int64|uint32|uint64|unsafe\.Pointer)",
            "fence": r"atomic\.(Load|Store)",
        }
    # C# fallback
    return {
        "atomic_load": r"Volatile\.Read\s*\(|Interlocked\.",
        "atomic_store": r"Volatile\.Write\s*\(",
        "atomic_fetch_add": r"Interlocked\.(Add|Increment)\s*\(",
        "atomic_fetch_sub": r"Interlocked\.(Add|Decrement)\s*\(",
        "atomic_cas": r"Interlocked\.CompareExchange\s*\(",
        "ordering": r"(Volatile|Interlocked|MemoryBarrier)",
        "atomic_decl": r"(volatile\s+int|Interlocked)",
        "fence": r"Thread\.MemoryBarrier\s*\(",
    }


def _detect_pattern(source: str, language: str) -> List[str]:
    """Identify concurrency patterns present in source code."""
    detected: List[str] = []
    lower = source.lower()
    pats = _language_atomic_patterns(language)

    has_load = bool(re.search(pats["atomic_load"], source))
    has_store = bool(re.search(pats["atomic_store"], source))
    has_add = bool(re.search(pats["atomic_fetch_add"], source))
    has_sub = bool(re.search(pats["atomic_fetch_sub"], source))
    has_cas = bool(re.search(pats["atomic_cas"], source))
    has_fence = bool(re.search(pats["fence"], source))

    # Counter: fetch_add without CAS
    if has_add and not has_cas:
        detected.append("counter")
    # Reference counting: fetch_sub present
    if has_sub:
        detected.append("refcount")
    # Flag / signal: store + load without arithmetic
    if has_store and has_load and not has_add and not has_sub:
        detected.append("flag")
    # Spinlock: CAS + store pattern
    if has_cas and has_store:
        if "lock" in lower or "unlock" in lower or "spin" in lower:
            detected.append("spinlock")
        else:
            detected.append("cas_loop")
    # Seqlock: counter increment by 2 or seq in names
    if ("seq" in lower and "lock" in lower) or re.search(r"fetch_add.*2", source):
        detected.append("seqlock")
    # Message passing: store data then store flag
    if has_store and has_load:
        lines = source.split("\n")
        store_count = sum(1 for l in lines if re.search(pats["atomic_store"], l))
        if store_count >= 2:
            detected.append("message_passing")
    # Queue patterns
    if ("queue" in lower or "enqueue" in lower or "dequeue" in lower) and has_cas:
        detected.append("mpsc")
    if "ring" in lower or "spsc" in lower:
        detected.append("spsc")
    # Double-checked locking
    if "once" in lower or "singleton" in lower or ("check" in lower and has_cas):
        detected.append("dcl")
    if not detected:
        if has_load or has_store:
            detected.append("unknown_atomic_usage")
    return detected


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _tutorial_sections(language: str) -> List[TutorialSection]:
    """Build tutorial sections for a language."""
    lang = language.upper()
    sections: List[TutorialSection] = []

    # Language-specific syntax helpers
    if lang in ("C",):
        decl, load, store = "_Atomic int counter;", "atomic_load_explicit(&counter, memory_order_relaxed)", "atomic_store_explicit(&counter, 1, memory_order_release)"
        inc = "atomic_fetch_add_explicit(&counter, 1, memory_order_relaxed);"
        acq, rel, rlx, sc = "memory_order_acquire", "memory_order_release", "memory_order_relaxed", "memory_order_seq_cst"
        cas_ex = "atomic_compare_exchange_strong_explicit(&val, &expected, desired, memory_order_acq_rel, memory_order_acquire);"
    elif lang in ("CPP", "C++"):
        decl, load, store = "std::atomic<int> counter{0};", "counter.load(std::memory_order_relaxed)", "counter.store(1, std::memory_order_release)"
        inc = "counter.fetch_add(1, std::memory_order_relaxed);"
        acq, rel, rlx, sc = "std::memory_order_acquire", "std::memory_order_release", "std::memory_order_relaxed", "std::memory_order_seq_cst"
        cas_ex = "val.compare_exchange_strong(expected, desired, std::memory_order_acq_rel, std::memory_order_acquire);"
    elif lang == "RUST":
        decl, load, store = "let counter = AtomicUsize::new(0);", "counter.load(Ordering::Relaxed)", "counter.store(1, Ordering::Release)"
        inc = "counter.fetch_add(1, Ordering::Relaxed);"
        acq, rel, rlx, sc = "Ordering::Acquire", "Ordering::Release", "Ordering::Relaxed", "Ordering::SeqCst"
        cas_ex = "val.compare_exchange(expected, desired, Ordering::AcqRel, Ordering::Acquire);"
    elif lang == "JAVA":
        decl, load, store = "AtomicInteger counter = new AtomicInteger(0);", "counter.get()", "counter.set(1) // or VarHandle setRelease"
        inc = "counter.incrementAndGet();"
        acq, rel, rlx, sc = "getAcquire()", "setRelease()", "getOpaque()", "get()/set() (volatile)"
        cas_ex = "counter.compareAndSet(expected, desired);"
    elif lang == "GO":
        decl, load, store = "var counter int64", "atomic.LoadInt64(&counter)", "atomic.StoreInt64(&counter, 1)"
        inc = "atomic.AddInt64(&counter, 1)"
        acq, rel, rlx, sc = "atomic.LoadInt64 (implicit acquire)", "atomic.StoreInt64 (implicit release)", "N/A (Go atomics are SC)", "atomic.LoadInt64/StoreInt64 (default)"
        cas_ex = "atomic.CompareAndSwapInt64(&val, expected, desired)"
    else:
        decl, load, store = "volatile int counter;", "Volatile.Read(ref counter)", "Volatile.Write(ref counter, 1)"
        inc = "Interlocked.Increment(ref counter);"
        acq, rel, rlx, sc = "Volatile.Read", "Volatile.Write", "N/A", "Interlocked.*"
        cas_ex = "Interlocked.CompareExchange(ref val, desired, expected);"

    sections.append(TutorialSection(
        title="1. What Are Atomics and Why",
        content=(
            "Atomic operations are indivisible memory operations that complete without interruption. "
            "On modern multi-core CPUs, ordinary reads and writes can be reordered by the compiler and "
            "hardware, leading to data races. Atomics provide two guarantees: (1) the operation itself "
            "is atomic (no torn reads/writes), and (2) you can specify a memory ordering that controls "
            "how other memory operations are ordered relative to this one."
        ),
        code_examples=[f"// Declaration:\n{decl}\n\n// Atomic load:\nint val = {load};\n\n// Atomic store:\n{store};"],
        exercises=["Declare an atomic variable and perform a load and store in your language.",
                   "Explain why a plain int shared between threads is unsafe even on x86."],
    ))

    sections.append(TutorialSection(
        title="2. Memory Orderings Explained",
        content=(
            "Memory orderings control visibility of writes across threads. From weakest to strongest:\n"
            f"  Relaxed ({rlx}): No ordering guarantees, only atomicity.\n"
            f"  Acquire ({acq}): All subsequent reads/writes stay after this load.\n"
            f"  Release ({rel}): All preceding reads/writes stay before this store.\n"
            f"  AcqRel: Combines acquire and release (for read-modify-write).\n"
            f"  SeqCst ({sc}): Total global order; strongest but slowest.\n"
            "Acquire pairs with Release to form a happens-before edge."
        ),
        code_examples=[
            f"// Acquire load:\nint v = {load.replace('relaxed', 'acquire').replace('Relaxed', 'Acquire')};\n"
            f"// Release store:\n{store};",
        ],
        exercises=["Draw a happens-before diagram for a producer-consumer with acquire/release.",
                   "Explain what 'reordering' means at both the compiler and hardware level."],
    ))

    sections.append(TutorialSection(
        title="3. Relaxed Ordering — When Safe",
        content=(
            "Relaxed ordering provides atomicity but no ordering. Safe uses:\n"
            "  - Statistics counters where exact ordering doesn't matter.\n"
            "  - Progress indicators read occasionally by another thread.\n"
            "  - Sequence numbers where only uniqueness is needed.\n"
            "NEVER use relaxed for synchronization (flags, locks, message passing)."
        ),
        code_examples=[f"// Safe: statistics counter\n{inc}"],
        exercises=["Identify which of these are safe with Relaxed: (a) page-view counter, "
                   "(b) ready flag, (c) unique ID generator."],
    ))

    sections.append(TutorialSection(
        title="4. Acquire/Release — The Workhorse",
        content=(
            "Most concurrent patterns need acquire/release. The rule is simple:\n"
            "  - The thread that PUBLISHES data uses a Release store.\n"
            "  - The thread that CONSUMES data uses an Acquire load.\n"
            "This ensures all writes before the Release are visible after the Acquire.\n"
            "Read-modify-write operations (like CAS) often use AcqRel."
        ),
        code_examples=[
            f"// Producer:\ndata = 42;  // non-atomic write\n{store}  // release\n\n"
            f"// Consumer:\nwhile ({load.replace('relaxed', 'acquire').replace('Relaxed', 'Acquire')} == 0) {{}}\n"
            f"assert(data == 42);  // guaranteed by acquire/release",
            f"// CAS with AcqRel:\n{cas_ex}",
        ],
        exercises=["Implement a simple flag-based notification between two threads.",
                   "Explain why acquire/release is sufficient for a spinlock."],
    ))

    sections.append(TutorialSection(
        title="5. SeqCst — The Safe Default",
        content=(
            "Sequential consistency (SeqCst) provides a single total order visible to all threads. "
            "It is the strongest (and most expensive) ordering. Use it when:\n"
            "  - You need all threads to agree on the order of operations.\n"
            "  - You are unsure which ordering is correct (correctness first, optimize later).\n"
            "  - Implementing Dekker's algorithm or similar patterns that need total order.\n"
            "On x86, SeqCst loads are free but stores require an MFENCE."
        ),
        code_examples=[
            f"// SeqCst is often the default:\n{load.replace('relaxed', 'seq_cst').replace('Relaxed', 'SeqCst')};",
        ],
        exercises=["Write a Dekker-style mutual exclusion and explain why acq/rel is insufficient.",
                   "Measure the performance difference between Relaxed and SeqCst fetch_add on your machine."],
    ))

    sections.append(TutorialSection(
        title="6. Common Patterns",
        content=(
            "Key patterns and their orderings:\n"
            "  Counter (stats):      fetch_add Relaxed\n"
            "  Flag/signal:          store Release, load Acquire\n"
            "  Message passing:      store Release, load Acquire\n"
            "  Reference counting:   fetch_add Relaxed, fetch_sub Release, fence Acquire before dealloc\n"
            "  Spinlock:             CAS Acquire to lock, store Release to unlock\n"
            "  Sequence lock:        store Release for writer, load Acquire for reader\n"
            "  SPSC ring buffer:     Release on index update, Acquire on index read\n"
            "  Ticket lock:          fetch_add Relaxed for ticket, load Acquire for serving"
        ),
        code_examples=[],
        exercises=["Implement a spinlock and a reference-counted pointer in your language."],
    ))

    sections.append(TutorialSection(
        title="7. Pitfalls",
        content=(
            "Common mistakes:\n"
            "  1. Using Relaxed for flags or message passing — data races.\n"
            "  2. Using SeqCst everywhere — wasteful, hides design intent.\n"
            "  3. Missing fence before deallocation in reference counting.\n"
            "  4. Forgetting that CAS failure ordering can be weaker.\n"
            "  5. Assuming x86 behavior on ARM/RISC-V (x86 is strongly ordered).\n"
            "  6. Not testing on weak memory model hardware (ARM, POWER).\n"
            "  7. Confusing volatile (language-level) with atomic."
        ),
        code_examples=[],
        exercises=["Review your own concurrent code for these pitfalls.",
                   "Run your code under ThreadSanitizer or a model checker."],
    ))
    return sections


def atomics_tutorial(language: str) -> str:
    """Return a comprehensive multi-section tutorial string for the given language."""
    sections = _tutorial_sections(language)
    parts: List[str] = [f"=== Relaxed Atomics Tutorial ({language.upper()}) ===\n"]
    for sec in sections:
        parts.append(str(sec))
        for ex in sec.code_examples:
            parts.append(ex)
        if sec.exercises:
            parts.append("Exercises:")
            for i, e in enumerate(sec.exercises, 1):
                parts.append(f"  {i}. {e}")
        parts.append("")
    return "\n".join(parts)


def common_patterns(language: str) -> List[AtomicPattern]:
    """Return 10 verified atomic patterns with code for the given language."""
    lang = language.upper()
    patterns: List[AtomicPattern] = []

    # Helpers for language-specific code
    def _counter_code() -> str:
        if lang in ("C",):
            return "_Atomic int counter = 0;\nvoid inc(void) { atomic_fetch_add_explicit(&counter, 1, memory_order_relaxed); }\nint read(void) { return atomic_load_explicit(&counter, memory_order_relaxed); }"
        if lang in ("CPP", "C++"):
            return "std::atomic<int> counter{0};\nvoid inc() { counter.fetch_add(1, std::memory_order_relaxed); }\nint read() { return counter.load(std::memory_order_relaxed); }"
        if lang == "RUST":
            return "let counter = AtomicUsize::new(0);\ncounter.fetch_add(1, Ordering::Relaxed);\nlet val = counter.load(Ordering::Relaxed);"
        if lang == "JAVA":
            return "AtomicInteger counter = new AtomicInteger(0);\ncounter.incrementAndGet(); // inherently SeqCst in Java\nint val = counter.get();"
        if lang == "GO":
            return "var counter int64\natomic.AddInt64(&counter, 1)\nval := atomic.LoadInt64(&counter)"
        return "volatile int counter = 0;\nInterlocked.Increment(ref counter);\nint val = Volatile.Read(ref counter);"

    def _flag_code() -> str:
        if lang in ("C",):
            return "_Atomic int ready = 0;\nint data = 0;\n// Producer:\ndata = 42;\natomic_store_explicit(&ready, 1, memory_order_release);\n// Consumer:\nwhile (!atomic_load_explicit(&ready, memory_order_acquire)) {}\nassert(data == 42);"
        if lang in ("CPP", "C++"):
            return "std::atomic<bool> ready{false};\nint data = 0;\n// Producer:\ndata = 42;\nready.store(true, std::memory_order_release);\n// Consumer:\nwhile (!ready.load(std::memory_order_acquire)) {}\nassert(data == 42);"
        if lang == "RUST":
            return "let ready = AtomicBool::new(false);\n// Producer:\nunsafe { DATA = 42; }\nready.store(true, Ordering::Release);\n// Consumer:\nwhile !ready.load(Ordering::Acquire) {}\nassert_eq!(unsafe { DATA }, 42);"
        if lang == "JAVA":
            return "volatile boolean ready = false;\nint data = 0;\n// Producer:\ndata = 42;\nready = true; // volatile write = release\n// Consumer:\nwhile (!ready) {} // volatile read = acquire\nassert data == 42;"
        if lang == "GO":
            return "var ready int32\nvar data int\n// Producer:\ndata = 42\natomic.StoreInt32(&ready, 1)\n// Consumer:\nfor atomic.LoadInt32(&ready) == 0 {}\n// data is now visible"
        return "volatile bool ready = false;\nint data = 0;\n// Producer:\ndata = 42;\nVolatile.Write(ref ready, true);\n// Consumer:\nwhile (!Volatile.Read(ref ready)) {}\nDebug.Assert(data == 42);"

    def _refcount_code() -> str:
        if lang in ("CPP", "C++"):
            return ("struct RefCounted {\n  std::atomic<int> refcount{1};\n  void acquire() { refcount.fetch_add(1, std::memory_order_relaxed); }\n"
                    "  void release() {\n    if (refcount.fetch_sub(1, std::memory_order_release) == 1) {\n"
                    "      std::atomic_thread_fence(std::memory_order_acquire);\n      delete this;\n    }\n  }\n};")
        if lang in ("C",):
            return ("struct RefCounted { _Atomic int refcount; };\nvoid acquire(struct RefCounted *r) {\n"
                    "  atomic_fetch_add_explicit(&r->refcount, 1, memory_order_relaxed);\n}\n"
                    "void release(struct RefCounted *r) {\n"
                    "  if (atomic_fetch_sub_explicit(&r->refcount, 1, memory_order_release) == 1) {\n"
                    "    atomic_thread_fence(memory_order_acquire);\n    free(r);\n  }\n}")
        if lang == "RUST":
            return ("// Arc pattern (simplified):\nstruct MyArc<T> { inner: *mut ArcInner<T> }\nstruct ArcInner<T> { refcount: AtomicUsize, data: T }\n"
                    "impl<T> Clone for MyArc<T> {\n  fn clone(&self) -> Self {\n"
                    "    unsafe { (*self.inner).refcount.fetch_add(1, Ordering::Relaxed); }\n    MyArc { inner: self.inner }\n  }\n}\n"
                    "impl<T> Drop for MyArc<T> {\n  fn drop(&mut self) {\n"
                    "    if unsafe { (*self.inner).refcount.fetch_sub(1, Ordering::Release) } == 1 {\n"
                    "      fence(Ordering::Acquire);\n      unsafe { drop(Box::from_raw(self.inner)); }\n    }\n  }\n}")
        if lang == "JAVA":
            return "// Java uses garbage collection; manual refcount rarely needed.\nAtomicInteger refcount = new AtomicInteger(1);\nvoid acquire() { refcount.incrementAndGet(); }\nvoid release() {\n  if (refcount.decrementAndGet() == 0) {\n    cleanup(); // GC handles memory\n  }\n}"
        if lang == "GO":
            return "type RefCounted struct {\n  refcount int64\n}\nfunc (r *RefCounted) Acquire() { atomic.AddInt64(&r.refcount, 1) }\nfunc (r *RefCounted) Release() {\n  if atomic.AddInt64(&r.refcount, -1) == 0 {\n    r.cleanup()\n  }\n}"
        return "class RefCounted {\n  volatile int refcount = 1;\n  void Acquire() { Interlocked.Increment(ref refcount); }\n  void Release() {\n    if (Interlocked.Decrement(ref refcount) == 0) {\n      Thread.MemoryBarrier();\n      Dispose();\n    }\n  }\n}"

    def _spinlock_code() -> str:
        if lang in ("CPP", "C++"):
            return "class Spinlock {\n  std::atomic<bool> locked{false};\npublic:\n  void lock() {\n    while (locked.exchange(true, std::memory_order_acquire)) {\n      while (locked.load(std::memory_order_relaxed)) {} // spin\n    }\n  }\n  void unlock() { locked.store(false, std::memory_order_release); }\n};"
        if lang in ("C",):
            return "_Atomic int locked = 0;\nvoid lock(void) {\n  while (atomic_exchange_explicit(&locked, 1, memory_order_acquire)) {\n    while (atomic_load_explicit(&locked, memory_order_relaxed)) {}\n  }\n}\nvoid unlock(void) { atomic_store_explicit(&locked, 0, memory_order_release); }"
        if lang == "RUST":
            return "struct Spinlock { locked: AtomicBool }\nimpl Spinlock {\n  fn lock(&self) {\n    while self.locked.swap(true, Ordering::Acquire) {\n      while self.locked.load(Ordering::Relaxed) { std::hint::spin_loop(); }\n    }\n  }\n  fn unlock(&self) { self.locked.store(false, Ordering::Release); }\n}"
        if lang == "JAVA":
            return "class Spinlock {\n  AtomicBoolean locked = new AtomicBoolean(false);\n  void lock() {\n    while (locked.getAndSet(true)) {\n      while (locked.get()) { Thread.onSpinWait(); }\n    }\n  }\n  void unlock() { locked.set(false); }\n}"
        if lang == "GO":
            return "type Spinlock struct { locked int32 }\nfunc (s *Spinlock) Lock() {\n  for !atomic.CompareAndSwapInt32(&s.locked, 0, 1) {\n    runtime.Gosched()\n  }\n}\nfunc (s *Spinlock) Unlock() { atomic.StoreInt32(&s.locked, 0) }"
        return "class Spinlock {\n  int locked = 0;\n  void Lock() {\n    while (Interlocked.Exchange(ref locked, 1) != 0) {\n      while (Volatile.Read(ref locked) != 0) { Thread.SpinWait(1); }\n    }\n  }\n  void Unlock() { Volatile.Write(ref locked, 0); }\n}"

    def _seqlock_code() -> str:
        if lang in ("CPP", "C++"):
            return ("class SeqLock {\n  std::atomic<unsigned> seq{0};\n  int data1, data2;\npublic:\n"
                    "  void write(int d1, int d2) {\n    unsigned s = seq.load(std::memory_order_relaxed);\n"
                    "    seq.store(s + 1, std::memory_order_release); // odd = writing\n    data1 = d1; data2 = d2;\n"
                    "    seq.store(s + 2, std::memory_order_release); // even = done\n  }\n"
                    "  std::pair<int,int> read() {\n    unsigned s1, s2; int d1, d2;\n    do {\n"
                    "      s1 = seq.load(std::memory_order_acquire);\n      d1 = data1; d2 = data2;\n"
                    "      s2 = seq.load(std::memory_order_acquire);\n    } while (s1 != s2 || (s1 & 1));\n"
                    "    return {d1, d2};\n  }\n};")
        if lang == "RUST":
            return ("struct SeqLock { seq: AtomicUsize, data: UnsafeCell<(i32,i32)> }\nimpl SeqLock {\n"
                    "  fn write(&self, d1: i32, d2: i32) {\n    let s = self.seq.load(Ordering::Relaxed);\n"
                    "    self.seq.store(s+1, Ordering::Release);\n    unsafe { *self.data.get() = (d1,d2); }\n"
                    "    self.seq.store(s+2, Ordering::Release);\n  }\n"
                    "  fn read(&self) -> (i32,i32) {\n    loop {\n      let s1 = self.seq.load(Ordering::Acquire);\n"
                    "      let d = unsafe { *self.data.get() };\n      let s2 = self.seq.load(Ordering::Acquire);\n"
                    "      if s1 == s2 && s1 & 1 == 0 { return d; }\n    }\n  }\n}")
        return ("// SeqLock pseudocode for " + language + ":\n"
                "// Writer: seq++; write data; seq++;\n// Reader: do { s1=seq; read data; s2=seq; } while (s1!=s2 || s1 is odd);")

    def _dcl_code() -> str:
        if lang in ("CPP", "C++"):
            return ("std::atomic<Singleton*> instance{nullptr};\nstd::mutex mtx;\nSingleton* get() {\n"
                    "  auto* p = instance.load(std::memory_order_acquire);\n  if (!p) {\n"
                    "    std::lock_guard<std::mutex> lock(mtx);\n    p = instance.load(std::memory_order_relaxed);\n"
                    "    if (!p) { p = new Singleton(); instance.store(p, std::memory_order_release); }\n  }\n  return p;\n}")
        if lang == "RUST":
            return "// Rust uses std::sync::OnceLock or lazy_static for this pattern.\nuse std::sync::OnceLock;\nstatic INSTANCE: OnceLock<Singleton> = OnceLock::new();\nfn get() -> &'static Singleton { INSTANCE.get_or_init(|| Singleton::new()) }"
        if lang == "JAVA":
            return "class Singleton {\n  private static volatile Singleton instance;\n  static Singleton get() {\n    if (instance == null) {\n      synchronized (Singleton.class) {\n        if (instance == null) instance = new Singleton();\n      }\n    }\n    return instance;\n  }\n}"
        return "// DCL pattern for " + language + ":\n// Check, lock, check again, initialize, publish with release."

    def _spsc_code() -> str:
        if lang in ("CPP", "C++"):
            return ("template<typename T, size_t N>\nclass SPSCQueue {\n  T buf[N];\n  std::atomic<size_t> head{0}, tail{0};\npublic:\n"
                    "  bool push(const T& val) {\n    size_t h = head.load(std::memory_order_relaxed);\n"
                    "    size_t next = (h + 1) % N;\n    if (next == tail.load(std::memory_order_acquire)) return false;\n"
                    "    buf[h] = val;\n    head.store(next, std::memory_order_release);\n    return true;\n  }\n"
                    "  bool pop(T& val) {\n    size_t t = tail.load(std::memory_order_relaxed);\n"
                    "    if (t == head.load(std::memory_order_acquire)) return false;\n"
                    "    val = buf[t];\n    tail.store((t + 1) % N, std::memory_order_release);\n    return true;\n  }\n};")
        if lang == "RUST":
            return ("struct SPSCQueue<T, const N: usize> {\n  buf: UnsafeCell<[MaybeUninit<T>; N]>,\n"
                    "  head: AtomicUsize, tail: AtomicUsize,\n}\nimpl<T, const N: usize> SPSCQueue<T, N> {\n"
                    "  fn push(&self, val: T) -> bool {\n    let h = self.head.load(Ordering::Relaxed);\n"
                    "    let next = (h + 1) % N;\n    if next == self.tail.load(Ordering::Acquire) { return false; }\n"
                    "    unsafe { (*self.buf.get())[h] = MaybeUninit::new(val); }\n"
                    "    self.head.store(next, Ordering::Release);\n    true\n  }\n}")
        return "// SPSC ring buffer for " + language + ":\n// head (Release write), tail (Release write), read indices with Acquire."

    def _mpsc_code() -> str:
        if lang in ("CPP", "C++"):
            return ("template<typename T>\nclass MPSCQueue {\n  struct Node { T data; std::atomic<Node*> next{nullptr}; };\n"
                    "  std::atomic<Node*> head, tail;\npublic:\n"
                    "  MPSCQueue() { auto* s = new Node{}; head.store(s); tail.store(s); }\n"
                    "  void push(T val) {\n    auto* n = new Node{std::move(val)};\n"
                    "    auto* prev = head.exchange(n, std::memory_order_acq_rel);\n"
                    "    prev->next.store(n, std::memory_order_release);\n  }\n"
                    "  bool pop(T& val) {\n    auto* t = tail.load(std::memory_order_relaxed);\n"
                    "    auto* next = t->next.load(std::memory_order_acquire);\n"
                    "    if (!next) return false;\n    val = std::move(next->data);\n"
                    "    tail.store(next, std::memory_order_release);\n    delete t;\n    return true;\n  }\n};")
        return "// MPSC queue for " + language + ": CAS-based push with AcqRel, pop with Acquire."

    def _ticket_lock_code() -> str:
        if lang in ("CPP", "C++"):
            return ("class TicketLock {\n  std::atomic<unsigned> ticket{0}, serving{0};\npublic:\n"
                    "  void lock() {\n    unsigned my = ticket.fetch_add(1, std::memory_order_relaxed);\n"
                    "    while (serving.load(std::memory_order_acquire) != my) {}\n  }\n"
                    "  void unlock() { serving.store(serving.load(std::memory_order_relaxed) + 1, std::memory_order_release); }\n};")
        if lang == "RUST":
            return ("struct TicketLock { ticket: AtomicUsize, serving: AtomicUsize }\nimpl TicketLock {\n"
                    "  fn lock(&self) {\n    let my = self.ticket.fetch_add(1, Ordering::Relaxed);\n"
                    "    while self.serving.load(Ordering::Acquire) != my { std::hint::spin_loop(); }\n  }\n"
                    "  fn unlock(&self) {\n    let s = self.serving.load(Ordering::Relaxed);\n"
                    "    self.serving.store(s + 1, Ordering::Release);\n  }\n}")
        return "// Ticket lock for " + language + ": fetch_add Relaxed for ticket, load Acquire for serving, store Release."

    default_lang = lang if lang not in ("CPP",) else "C++"

    patterns.append(AtomicPattern(
        name="Simple Counter", category=PatternCategory.COUNTER,
        description="Statistics counter where exact inter-thread ordering is irrelevant",
        code={default_lang: _counter_code()},
        correct_orderings={"increment": OrderingLevel.RELAXED, "read": OrderingLevel.RELAXED},
        common_mistakes=["Using SeqCst for a stats counter (wasteful)", "Using non-atomic increment (data race)"],
        explanation="Counters only need atomicity, not ordering. Relaxed is optimal.",
    ))
    patterns.append(AtomicPattern(
        name="Flag / Signal", category=PatternCategory.FLAG,
        description="One thread signals another via a boolean flag",
        code={default_lang: _flag_code()},
        correct_orderings={"writer": OrderingLevel.RELEASE, "reader": OrderingLevel.ACQUIRE},
        common_mistakes=["Using Relaxed for the flag — consumer may not see producer's data writes",
                         "Using volatile instead of atomic (C/C++)"],
        explanation="Release on write + Acquire on read ensures all prior writes are visible to the consumer.",
    ))
    patterns.append(AtomicPattern(
        name="Message Passing", category=PatternCategory.FLAG,
        description="Passing non-atomic data between threads via an atomic flag",
        code={default_lang: _flag_code()},
        correct_orderings={"sender_store": OrderingLevel.RELEASE, "receiver_load": OrderingLevel.ACQUIRE},
        common_mistakes=["Using Relaxed — the non-atomic data may not be visible",
                         "Missing the acquire on the consumer side"],
        explanation="Identical to flag pattern; the release/acquire pair creates a happens-before edge over the data.",
    ))
    patterns.append(AtomicPattern(
        name="Reference Counting", category=PatternCategory.REFERENCE_COUNT,
        description="Shared ownership with atomic reference count (like shared_ptr/Arc)",
        code={default_lang: _refcount_code()},
        correct_orderings={"increment": OrderingLevel.RELAXED, "decrement": OrderingLevel.RELEASE,
                           "dealloc_fence": OrderingLevel.ACQUIRE},
        common_mistakes=["Using Relaxed for decrement — other thread's writes may not be visible before dealloc",
                         "Missing acquire fence before deallocation",
                         "Using SeqCst for everything (unnecessarily expensive)"],
        explanation="Increment is Relaxed (just need atomicity). Decrement is Release so writes are published. "
                    "An Acquire fence before dealloc ensures we see all writes from other threads that decremented.",
    ))
    patterns.append(AtomicPattern(
        name="Spinlock", category=PatternCategory.LOCK,
        description="Test-and-set spinlock with backoff",
        code={default_lang: _spinlock_code()},
        correct_orderings={"lock_exchange": OrderingLevel.ACQUIRE, "unlock_store": OrderingLevel.RELEASE,
                           "spin_read": OrderingLevel.RELAXED},
        common_mistakes=["Missing Acquire on lock — critical section may see stale data",
                         "Missing Release on unlock — writes in critical section may not be visible",
                         "No backoff spin — hammering cache line causes contention"],
        explanation="Acquire on lock ensures all reads in the critical section see prior writes. "
                    "Release on unlock publishes all writes to the next lock holder.",
    ))
    patterns.append(AtomicPattern(
        name="Sequence Lock", category=PatternCategory.SEQUENCE_LOCK,
        description="Low-overhead reader-writer lock; readers never block",
        code={default_lang: _seqlock_code()},
        correct_orderings={"writer_inc": OrderingLevel.RELEASE, "reader_seq": OrderingLevel.ACQUIRE},
        common_mistakes=["Using Relaxed for the reader's sequence read — may read torn data",
                         "Not checking for odd sequence (writer in progress)"],
        explanation="Writer increments seq with Release before and after writing. Reader loads seq with Acquire "
                    "and retries if seq is odd or changed.",
    ))
    patterns.append(AtomicPattern(
        name="Double-Checked Locking", category=PatternCategory.FLAG,
        description="Lazy initialization with minimal locking",
        code={default_lang: _dcl_code()},
        correct_orderings={"first_check": OrderingLevel.ACQUIRE, "publish": OrderingLevel.RELEASE},
        common_mistakes=["Using Relaxed for the first check — may see partially constructed object",
                         "Not using atomic at all (pre-C++11 broken DCL)"],
        explanation="Acquire on first check + Release on publish ensures the fully constructed object is visible.",
    ))
    patterns.append(AtomicPattern(
        name="SPSC Ring Buffer", category=PatternCategory.QUEUE,
        description="Single-producer single-consumer lock-free queue",
        code={default_lang: _spsc_code()},
        correct_orderings={"head_write": OrderingLevel.RELEASE, "tail_write": OrderingLevel.RELEASE,
                           "head_read": OrderingLevel.ACQUIRE, "tail_read": OrderingLevel.ACQUIRE},
        common_mistakes=["Using Relaxed for index reads — consumer may read stale data from buffer",
                         "Using SeqCst unnecessarily — SPSC only needs acq/rel"],
        explanation="Producer writes data then updates head with Release. Consumer reads head with Acquire "
                    "to see the data, then updates tail with Release.",
    ))
    patterns.append(AtomicPattern(
        name="MPSC Queue", category=PatternCategory.QUEUE,
        description="Multi-producer single-consumer lock-free queue",
        code={default_lang: _mpsc_code()},
        correct_orderings={"push_exchange": OrderingLevel.ACQ_REL, "push_next": OrderingLevel.RELEASE,
                           "pop_next_read": OrderingLevel.ACQUIRE},
        common_mistakes=["Using Relaxed for head exchange — producers may corrupt the list",
                         "Not using AcqRel on the exchange — missing synchronization between producers"],
        explanation="Producers use exchange(AcqRel) to atomically insert. Consumer reads next with Acquire.",
    ))
    patterns.append(AtomicPattern(
        name="Ticket Lock", category=PatternCategory.LOCK,
        description="Fair FIFO spinlock using ticket/serving counters",
        code={default_lang: _ticket_lock_code()},
        correct_orderings={"ticket_fetch_add": OrderingLevel.RELAXED, "serving_load": OrderingLevel.ACQUIRE,
                           "serving_store": OrderingLevel.RELEASE},
        common_mistakes=["Using Acquire for ticket fetch_add — unnecessary overhead",
                         "Using Relaxed for serving load — may miss unlock"],
        explanation="Ticket is just a unique counter (Relaxed). Serving load needs Acquire to see the unlock's "
                    "Release and all writes in the prior critical section.",
    ))
    return patterns


def ordering_chooser(use_case: str) -> str:
    """Given a natural language use case, recommend an ordering level with explanation."""
    lower = use_case.lower()
    keyword_map: List[Tuple[List[str], OrderingLevel, str]] = [
        (["counter", "statistic", "metric", "tally", "count"],
         OrderingLevel.RELAXED, "Simple counters only need atomicity, not ordering."),
        (["progress", "indicator", "percentage"],
         OrderingLevel.RELAXED, "Progress indicators are approximate; Relaxed suffices."),
        (["flag", "signal", "notify", "ready", "done"],
         OrderingLevel.ACQ_REL, "Flags need Acquire on read and Release on write to synchronize data."),
        (["message", "passing", "publish", "send", "channel"],
         OrderingLevel.ACQ_REL, "Message passing requires Release on send, Acquire on receive."),
        (["lock", "mutex", "spin", "critical section"],
         OrderingLevel.ACQ_REL, "Locks need Acquire on entry (see prior writes) and Release on exit (publish writes)."),
        (["reference count", "refcount", "shared_ptr", "arc"],
         OrderingLevel.RELEASE, "Refcount: Relaxed increment, Release decrement, Acquire fence before dealloc."),
        (["queue", "buffer", "ring", "fifo"],
         OrderingLevel.ACQ_REL, "Queues need Acquire/Release on index updates to synchronize buffer contents."),
        (["stack", "treiber", "lifo"],
         OrderingLevel.ACQ_REL, "Lock-free stacks use CAS with AcqRel to synchronize head pointer."),
        (["singleton", "once", "lazy", "double.check"],
         OrderingLevel.ACQ_REL, "DCL pattern: Acquire on check, Release on publish."),
        (["sequence", "seqlock", "version"],
         OrderingLevel.ACQ_REL, "Seqlocks: Release on writer update, Acquire on reader check."),
    ]

    for keywords, ordering, reason in keyword_map:
        if any(kw in lower for kw in keywords):
            rec = OrderingRecommendation(
                use_case=use_case, recommended=ordering, reason=reason,
                alternatives=[
                    (OrderingLevel.SEQ_CST, "Always correct but may be slower"),
                    (OrderingLevel.RELAXED, "Only if no data synchronization is needed"),
                ],
                code_example=f"// Recommended: {ordering} — {reason}",
            )
            return str(rec)

    rec = OrderingRecommendation(
        use_case=use_case, recommended=OrderingLevel.SEQ_CST,
        reason="When unsure, SeqCst is the safe default. Optimize only after verifying correctness.",
        alternatives=[
            (OrderingLevel.ACQ_REL, "If you can identify a clear acquire/release pair"),
            (OrderingLevel.RELAXED, "Only for truly independent counters with no data dependency"),
        ],
        code_example="// Default: SeqCst until you can prove a weaker ordering is correct.",
    )
    return str(rec)


def reference_counter_pattern(language: str) -> str:
    """Return the correct reference counting pattern with fence-before-dealloc explanation."""
    lang = language.upper()
    explanation = (
        "\n// WHY fence(Acquire) before deallocation?\n"
        "// When the last thread decrements the refcount to zero, it must see ALL\n"
        "// writes from ALL other threads that previously held a reference.\n"
        "// The Release on fetch_sub ensures each thread's writes are published.\n"
        "// The Acquire fence on the deallocating thread synchronizes-with all\n"
        "// those Releases, creating happens-before edges from every prior holder.\n"
        "// Without this fence, the deallocating thread might free memory while\n"
        "// another thread's writes to the object are still in flight.\n"
    )
    if lang in ("CPP", "C++"):
        code = (
            "// C++ Reference Counting (shared_ptr semantics)\n"
            "struct RefCounted {\n"
            "  std::atomic<int> refcount{1};\n\n"
            "  void acquire() {\n"
            "    // Relaxed: we just need a unique count, no ordering needed.\n"
            "    // We already hold a reference, so the object won't be freed.\n"
            "    refcount.fetch_add(1, std::memory_order_relaxed);\n"
            "  }\n\n"
            "  void release() {\n"
            "    // Release: publishes all of this thread's writes to the object.\n"
            "    if (refcount.fetch_sub(1, std::memory_order_release) == 1) {\n"
            "      // Acquire fence: synchronizes-with all prior Release decrements.\n"
            "      std::atomic_thread_fence(std::memory_order_acquire);\n"
            "      delete this;\n"
            "    }\n"
            "  }\n"
            "};\n"
        )
    elif lang in ("C",):
        code = (
            "// C Reference Counting\n"
            "struct RefCounted { _Atomic int refcount; void *data; };\n\n"
            "void rc_acquire(struct RefCounted *rc) {\n"
            "  atomic_fetch_add_explicit(&rc->refcount, 1, memory_order_relaxed);\n"
            "}\n\n"
            "void rc_release(struct RefCounted *rc) {\n"
            "  if (atomic_fetch_sub_explicit(&rc->refcount, 1, memory_order_release) == 1) {\n"
            "    atomic_thread_fence(memory_order_acquire);\n"
            "    free(rc->data);\n"
            "    free(rc);\n"
            "  }\n"
            "}\n"
        )
    elif lang == "RUST":
        code = (
            "// Rust Arc pattern (simplified)\n"
            "use std::sync::atomic::{AtomicUsize, Ordering, fence};\n\n"
            "struct ArcInner<T> { refcount: AtomicUsize, data: T }\n\n"
            "struct MyArc<T> { ptr: *mut ArcInner<T> }\n\n"
            "impl<T> Clone for MyArc<T> {\n"
            "    fn clone(&self) -> Self {\n"
            "        let inner = unsafe { &*self.ptr };\n"
            "        // Relaxed: we hold a ref, object can't be freed.\n"
            "        let old = inner.refcount.fetch_add(1, Ordering::Relaxed);\n"
            "        if old > isize::MAX as usize { std::process::abort(); }\n"
            "        MyArc { ptr: self.ptr }\n"
            "    }\n"
            "}\n\n"
            "impl<T> Drop for MyArc<T> {\n"
            "    fn drop(&mut self) {\n"
            "        let inner = unsafe { &*self.ptr };\n"
            "        if inner.refcount.fetch_sub(1, Ordering::Release) == 1 {\n"
            "            fence(Ordering::Acquire);  // See all writes from other holders\n"
            "            unsafe { drop(Box::from_raw(self.ptr)); }\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
    elif lang == "JAVA":
        code = (
            "// Java: GC handles deallocation, but the pattern applies to cleanup logic.\n"
            "class RefCounted {\n"
            "    private final AtomicInteger refcount = new AtomicInteger(1);\n\n"
            "    void acquire() {\n"
            "        refcount.incrementAndGet(); // Java AtomicInteger is SeqCst\n"
            "    }\n\n"
            "    void release() {\n"
            "        if (refcount.decrementAndGet() == 0) {\n"
            "            // Java's decrementAndGet has full fence semantics\n"
            "            cleanup();\n"
            "        }\n"
            "    }\n\n"
            "    private void cleanup() { /* release resources */ }\n"
            "}\n"
        )
    elif lang == "GO":
        code = (
            "// Go: all atomics have sequential consistency\n"
            "type RefCounted struct {\n"
            "    refcount int64\n"
            "    // data fields\n"
            "}\n\n"
            "func (rc *RefCounted) Acquire() {\n"
            "    atomic.AddInt64(&rc.refcount, 1)\n"
            "}\n\n"
            "func (rc *RefCounted) Release() {\n"
            "    if atomic.AddInt64(&rc.refcount, -1) == 0 {\n"
            "        // Go atomics provide SC; no extra fence needed\n"
            "        rc.cleanup()\n"
            "    }\n"
            "}\n"
        )
    else:
        code = (
            "// C# Reference Counting\n"
            "class RefCounted : IDisposable {\n"
            "    private volatile int _refcount = 1;\n\n"
            "    public void Acquire() {\n"
            "        Interlocked.Increment(ref _refcount);\n"
            "    }\n\n"
            "    public void Release() {\n"
            "        if (Interlocked.Decrement(ref _refcount) == 0) {\n"
            "            Thread.MemoryBarrier(); // Acquire fence\n"
            "            Dispose();\n"
            "        }\n"
            "    }\n\n"
            "    public void Dispose() { /* release resources */ }\n"
            "}\n"
        )
    return code + explanation


def lock_free_patterns(language: str) -> Dict[str, str]:
    """Return verified lock-free implementations for the given language."""
    lang = language.upper()
    result: Dict[str, str] = {}

    if lang in ("CPP", "C++"):
        result["treiber_stack"] = (
            "template<typename T>\n"
            "class TreiberStack {\n"
            "  struct Node { T data; Node* next; };\n"
            "  std::atomic<Node*> head{nullptr};\n"
            "public:\n"
            "  void push(T val) {\n"
            "    auto* n = new Node{std::move(val), nullptr};\n"
            "    n->next = head.load(std::memory_order_relaxed);\n"
            "    while (!head.compare_exchange_weak(\n"
            "        n->next, n,\n"
            "        std::memory_order_release,\n"
            "        std::memory_order_relaxed)) {}\n"
            "  }\n"
            "  bool pop(T& val) {\n"
            "    auto* old = head.load(std::memory_order_acquire);\n"
            "    while (old && !head.compare_exchange_weak(\n"
            "        old, old->next,\n"
            "        std::memory_order_acq_rel,\n"
            "        std::memory_order_acquire)) {}\n"
            "    if (!old) return false;\n"
            "    val = std::move(old->data);\n"
            "    delete old;  // ABA caution: use hazard pointers in production\n"
            "    return true;\n"
            "  }\n"
            "};"
        )
        result["michael_scott_queue"] = (
            "template<typename T>\n"
            "class MSQueue {\n"
            "  struct Node { T data; std::atomic<Node*> next{nullptr}; };\n"
            "  std::atomic<Node*> head, tail;\n"
            "public:\n"
            "  MSQueue() { auto* s = new Node{}; head.store(s); tail.store(s); }\n"
            "  void enqueue(T val) {\n"
            "    auto* n = new Node{std::move(val)};\n"
            "    while (true) {\n"
            "      auto* t = tail.load(std::memory_order_acquire);\n"
            "      auto* next = t->next.load(std::memory_order_acquire);\n"
            "      if (t == tail.load(std::memory_order_acquire)) {\n"
            "        if (!next) {\n"
            "          if (t->next.compare_exchange_weak(\n"
            "              next, n, std::memory_order_release)) {\n"
            "            tail.compare_exchange_strong(t, n, std::memory_order_release);\n"
            "            return;\n"
            "          }\n"
            "        } else {\n"
            "          tail.compare_exchange_weak(t, next, std::memory_order_release);\n"
            "        }\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "  bool dequeue(T& val) {\n"
            "    while (true) {\n"
            "      auto* h = head.load(std::memory_order_acquire);\n"
            "      auto* t = tail.load(std::memory_order_acquire);\n"
            "      auto* next = h->next.load(std::memory_order_acquire);\n"
            "      if (h == head.load(std::memory_order_acquire)) {\n"
            "        if (h == t) {\n"
            "          if (!next) return false;\n"
            "          tail.compare_exchange_weak(t, next, std::memory_order_release);\n"
            "        } else {\n"
            "          val = next->data;\n"
            "          if (head.compare_exchange_weak(\n"
            "              h, next, std::memory_order_acq_rel)) {\n"
            "            delete h;\n"
            "            return true;\n"
            "          }\n"
            "        }\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "};"
        )
        result["spsc_ring_buffer"] = (
            "template<typename T, size_t N>\n"
            "class SPSCRing {\n"
            "  T buffer[N];\n"
            "  std::atomic<size_t> write_idx{0};\n"
            "  std::atomic<size_t> read_idx{0};\n"
            "public:\n"
            "  bool push(const T& val) {\n"
            "    size_t w = write_idx.load(std::memory_order_relaxed);\n"
            "    size_t next = (w + 1) % N;\n"
            "    if (next == read_idx.load(std::memory_order_acquire)) return false;\n"
            "    buffer[w] = val;\n"
            "    write_idx.store(next, std::memory_order_release);\n"
            "    return true;\n"
            "  }\n"
            "  bool pop(T& val) {\n"
            "    size_t r = read_idx.load(std::memory_order_relaxed);\n"
            "    if (r == write_idx.load(std::memory_order_acquire)) return false;\n"
            "    val = buffer[r];\n"
            "    read_idx.store((r + 1) % N, std::memory_order_release);\n"
            "    return true;\n"
            "  }\n"
            "};"
        )
        result["seqlock"] = (
            "class SeqLock {\n"
            "  std::atomic<unsigned> seq{0};\n"
            "  int data1{0}, data2{0};\n"
            "public:\n"
            "  void write(int d1, int d2) {\n"
            "    unsigned s = seq.load(std::memory_order_relaxed);\n"
            "    seq.store(s + 1, std::memory_order_release);\n"
            "    data1 = d1;\n"
            "    data2 = d2;\n"
            "    seq.store(s + 2, std::memory_order_release);\n"
            "  }\n"
            "  bool read(int& d1, int& d2) {\n"
            "    unsigned s1 = seq.load(std::memory_order_acquire);\n"
            "    if (s1 & 1) return false;\n"
            "    d1 = data1;\n"
            "    d2 = data2;\n"
            "    std::atomic_thread_fence(std::memory_order_acquire);\n"
            "    unsigned s2 = seq.load(std::memory_order_relaxed);\n"
            "    return s1 == s2;\n"
            "  }\n"
            "};"
        )
        result["rcu_like"] = (
            "// Simplified RCU-like pattern (not full RCU)\n"
            "template<typename T>\n"
            "class RCUPtr {\n"
            "  std::atomic<T*> ptr{nullptr};\n"
            "  std::atomic<int> readers{0};\n"
            "public:\n"
            "  // Reader: acquire the current pointer\n"
            "  T* read_lock() {\n"
            "    readers.fetch_add(1, std::memory_order_acquire);\n"
            "    return ptr.load(std::memory_order_acquire);\n"
            "  }\n"
            "  void read_unlock() {\n"
            "    readers.fetch_sub(1, std::memory_order_release);\n"
            "  }\n"
            "  // Writer: publish new version, wait for readers, free old\n"
            "  void update(T* new_val) {\n"
            "    T* old = ptr.exchange(new_val, std::memory_order_acq_rel);\n"
            "    // Grace period: wait until no readers hold old pointer\n"
            "    while (readers.load(std::memory_order_acquire) > 0) {\n"
            "      std::this_thread::yield();\n"
            "    }\n"
            "    std::atomic_thread_fence(std::memory_order_acquire);\n"
            "    delete old;\n"
            "  }\n"
            "};"
        )
        result["hazard_pointer"] = (
            "// Simplified hazard pointer scheme\n"
            "constexpr int MAX_THREADS = 64;\n"
            "template<typename T>\n"
            "class HazardPointer {\n"
            "  std::atomic<T*> hp[MAX_THREADS]{};\n"
            "  std::atomic<T*> protected_ptr{nullptr};\n"
            "public:\n"
            "  // Protect a pointer for the current thread\n"
            "  T* protect(int tid, std::atomic<T*>& src) {\n"
            "    T* p;\n"
            "    do {\n"
            "      p = src.load(std::memory_order_acquire);\n"
            "      hp[tid].store(p, std::memory_order_release);\n"
            "    } while (p != src.load(std::memory_order_acquire));\n"
            "    return p;\n"
            "  }\n"
            "  // Clear hazard pointer\n"
            "  void clear(int tid) {\n"
            "    hp[tid].store(nullptr, std::memory_order_release);\n"
            "  }\n"
            "  // Check if any thread protects this pointer\n"
            "  bool is_protected(T* p) {\n"
            "    for (int i = 0; i < MAX_THREADS; ++i) {\n"
            "      if (hp[i].load(std::memory_order_acquire) == p) return true;\n"
            "    }\n"
            "    return false;\n"
            "  }\n"
            "  // Retire: defer deletion until no thread protects p\n"
            "  void retire(T* p) {\n"
            "    while (is_protected(p)) { std::this_thread::yield(); }\n"
            "    delete p;\n"
            "  }\n"
            "};"
        )
    elif lang == "RUST":
        result["treiber_stack"] = (
            "use std::sync::atomic::{AtomicPtr, Ordering};\n"
            "use std::ptr;\n\n"
            "struct Node<T> { data: T, next: *mut Node<T> }\n\n"
            "struct TreiberStack<T> { head: AtomicPtr<Node<T>> }\n\n"
            "impl<T> TreiberStack<T> {\n"
            "    fn new() -> Self { TreiberStack { head: AtomicPtr::new(ptr::null_mut()) } }\n"
            "    fn push(&self, val: T) {\n"
            "        let n = Box::into_raw(Box::new(Node { data: val, next: ptr::null_mut() }));\n"
            "        loop {\n"
            "            let old = self.head.load(Ordering::Relaxed);\n"
            "            unsafe { (*n).next = old; }\n"
            "            if self.head.compare_exchange_weak(\n"
            "                old, n, Ordering::Release, Ordering::Relaxed).is_ok() { break; }\n"
            "        }\n"
            "    }\n"
            "    fn pop(&self) -> Option<T> {\n"
            "        loop {\n"
            "            let old = self.head.load(Ordering::Acquire);\n"
            "            if old.is_null() { return None; }\n"
            "            let next = unsafe { (*old).next };\n"
            "            if self.head.compare_exchange_weak(\n"
            "                old, next, Ordering::AcqRel, Ordering::Acquire).is_ok() {\n"
            "                let node = unsafe { Box::from_raw(old) };\n"
            "                return Some(node.data);\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}"
        )
        result["spsc_ring_buffer"] = (
            "use std::sync::atomic::{AtomicUsize, Ordering};\n"
            "use std::cell::UnsafeCell;\n"
            "use std::mem::MaybeUninit;\n\n"
            "struct SPSCRing<T, const N: usize> {\n"
            "    buf: UnsafeCell<[MaybeUninit<T>; N]>,\n"
            "    head: AtomicUsize,\n"
            "    tail: AtomicUsize,\n"
            "}\n\n"
            "impl<T, const N: usize> SPSCRing<T, N> {\n"
            "    fn push(&self, val: T) -> bool {\n"
            "        let h = self.head.load(Ordering::Relaxed);\n"
            "        let next = (h + 1) % N;\n"
            "        if next == self.tail.load(Ordering::Acquire) { return false; }\n"
            "        unsafe { (*self.buf.get())[h] = MaybeUninit::new(val); }\n"
            "        self.head.store(next, Ordering::Release);\n"
            "        true\n"
            "    }\n"
            "    fn pop(&self) -> Option<T> {\n"
            "        let t = self.tail.load(Ordering::Relaxed);\n"
            "        if t == self.head.load(Ordering::Acquire) { return None; }\n"
            "        let val = unsafe { (*self.buf.get())[t].assume_init_read() };\n"
            "        self.tail.store((t + 1) % N, Ordering::Release);\n"
            "        Some(val)\n"
            "    }\n"
            "}"
        )
        result["seqlock"] = "// SeqLock in Rust: see common_patterns() for implementation."
        result["michael_scott_queue"] = "// MS Queue in Rust requires unsafe + hazard pointers; see hazard_pointer."
        result["rcu_like"] = (
            "use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering, fence};\n"
            "use std::ptr;\n\n"
            "struct RcuPtr<T> { ptr: AtomicPtr<T>, readers: AtomicUsize }\n\n"
            "impl<T> RcuPtr<T> {\n"
            "    fn read_lock(&self) -> *mut T {\n"
            "        self.readers.fetch_add(1, Ordering::Acquire);\n"
            "        self.ptr.load(Ordering::Acquire)\n"
            "    }\n"
            "    fn read_unlock(&self) {\n"
            "        self.readers.fetch_sub(1, Ordering::Release);\n"
            "    }\n"
            "    fn update(&self, new_val: Box<T>) {\n"
            "        let new_ptr = Box::into_raw(new_val);\n"
            "        let old = self.ptr.swap(new_ptr, Ordering::AcqRel);\n"
            "        while self.readers.load(Ordering::Acquire) > 0 {\n"
            "            std::thread::yield_now();\n"
            "        }\n"
            "        fence(Ordering::Acquire);\n"
            "        if !old.is_null() { unsafe { drop(Box::from_raw(old)); } }\n"
            "    }\n"
            "}"
        )
        result["hazard_pointer"] = (
            "use std::sync::atomic::{AtomicPtr, Ordering};\n"
            "use std::ptr;\n\n"
            "const MAX_THREADS: usize = 64;\n\n"
            "struct HazardPointer<T> {\n"
            "    hp: [AtomicPtr<T>; MAX_THREADS],\n"
            "}\n\n"
            "impl<T> HazardPointer<T> {\n"
            "    fn protect(&self, tid: usize, src: &AtomicPtr<T>) -> *mut T {\n"
            "        loop {\n"
            "            let p = src.load(Ordering::Acquire);\n"
            "            self.hp[tid].store(p, Ordering::Release);\n"
            "            if p == src.load(Ordering::Acquire) { return p; }\n"
            "        }\n"
            "    }\n"
            "    fn clear(&self, tid: usize) {\n"
            "        self.hp[tid].store(ptr::null_mut(), Ordering::Release);\n"
            "    }\n"
            "    fn is_protected(&self, p: *mut T) -> bool {\n"
            "        self.hp.iter().any(|h| h.load(Ordering::Acquire) == p)\n"
            "    }\n"
            "}"
        )
    else:
        # Generic implementations for other languages
        result["treiber_stack"] = f"// Treiber stack for {language}: CAS loop on head pointer.\n// push: create node, CAS head; pop: read head, CAS to head.next."
        result["michael_scott_queue"] = f"// Michael-Scott queue for {language}: two CAS pointers (head, tail) with sentinel."
        result["spsc_ring_buffer"] = f"// SPSC ring buffer for {language}: atomic head/tail indices, Release on write, Acquire on read."
        result["seqlock"] = f"// SeqLock for {language}: atomic sequence counter, Release on write, Acquire on read."
        result["rcu_like"] = f"// RCU-like for {language}: atomic pointer swap with grace period."
        result["hazard_pointer"] = f"// Hazard pointer for {language}: per-thread atomic pointer array for safe reclamation."
    return result


def verify_user_code(source: str, language: str) -> VerificationResult:
    """Parse source for atomic operations, detect patterns, check orderings, flag mistakes."""
    lang_enum = Language[language.upper()] if language.upper() in Language.__members__ else Language.C
    pats = _language_atomic_patterns(language)
    detected = _detect_pattern(source, language)
    issues: List[VerificationIssue] = []
    suggestions: List[str] = []
    lines = source.split("\n")

    # Extract all orderings used
    ordering_matches: List[Tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        for m in re.finditer(pats["ordering"], line):
            ordering_matches.append((i, m.group(0)))

    # Normalize ordering string to OrderingLevel
    def _parse_ordering(s: str) -> Optional[OrderingLevel]:
        s_lower = s.lower().replace("std::", "").replace("memory_order_", "").replace("ordering::", "")
        mapping = {"relaxed": OrderingLevel.RELAXED, "consume": OrderingLevel.CONSUME,
                   "acquire": OrderingLevel.ACQUIRE, "release": OrderingLevel.RELEASE,
                   "acq_rel": OrderingLevel.ACQ_REL, "acqrel": OrderingLevel.ACQ_REL,
                   "seq_cst": OrderingLevel.SEQ_CST, "seqcst": OrderingLevel.SEQ_CST}
        return mapping.get(s_lower)

    # Check for Relaxed used in synchronization patterns
    for line_no, ord_str in ordering_matches:
        parsed = _parse_ordering(ord_str)
        if parsed == OrderingLevel.RELAXED:
            line_text = lines[line_no - 1] if line_no <= len(lines) else ""
            # Check if this relaxed is in a store that looks like a flag/signal
            if re.search(pats["atomic_store"], line_text):
                if "flag" in detected or "message_passing" in detected:
                    issues.append(VerificationIssue(
                        line=line_no,
                        description="Relaxed store used for flag/message passing",
                        severity="ERROR",
                        fix=f"Use Release ordering: change {ord_str} to Release/memory_order_release",
                        pattern_violated="flag/message_passing requires Release store",
                    ))
            if re.search(pats["atomic_load"], line_text):
                if "flag" in detected or "message_passing" in detected:
                    issues.append(VerificationIssue(
                        line=line_no,
                        description="Relaxed load used for flag/message passing",
                        severity="ERROR",
                        fix=f"Use Acquire ordering: change {ord_str} to Acquire/memory_order_acquire",
                        pattern_violated="flag/message_passing requires Acquire load",
                    ))

    # Check for spinlock pattern issues
    if "spinlock" in detected:
        has_acquire_on_lock = False
        has_release_on_unlock = False
        for line_no, ord_str in ordering_matches:
            parsed = _parse_ordering(ord_str)
            line_text = lines[line_no - 1] if line_no <= len(lines) else ""
            if parsed in (OrderingLevel.ACQUIRE, OrderingLevel.ACQ_REL) and \
               (re.search(pats["atomic_cas"], line_text) or "exchange" in line_text.lower()):
                has_acquire_on_lock = True
            if parsed == OrderingLevel.RELEASE and re.search(pats["atomic_store"], line_text):
                has_release_on_unlock = True
        if not has_acquire_on_lock:
            issues.append(VerificationIssue(
                line=0, description="Spinlock missing Acquire on lock operation",
                severity="ERROR", fix="Use Acquire or AcqRel ordering on the lock CAS/exchange",
                pattern_violated="spinlock requires Acquire on lock",
            ))
        if not has_release_on_unlock:
            issues.append(VerificationIssue(
                line=0, description="Spinlock missing Release on unlock operation",
                severity="ERROR", fix="Use Release ordering on the unlock store",
                pattern_violated="spinlock requires Release on unlock",
            ))

    # Check for refcount issues
    if "refcount" in detected:
        has_release_on_sub = False
        has_fence = bool(re.search(pats["fence"], source))
        for line_no, ord_str in ordering_matches:
            parsed = _parse_ordering(ord_str)
            line_text = lines[line_no - 1] if line_no <= len(lines) else ""
            if parsed == OrderingLevel.RELEASE and re.search(pats["atomic_fetch_sub"], line_text):
                has_release_on_sub = True
        if not has_release_on_sub:
            issues.append(VerificationIssue(
                line=0, description="Reference count decrement not using Release ordering",
                severity="ERROR", fix="Use Release ordering on fetch_sub for refcount decrement",
                pattern_violated="refcount decrement requires Release",
            ))
        if not has_fence:
            issues.append(VerificationIssue(
                line=0, description="Missing Acquire fence before deallocation in refcount pattern",
                severity="ERROR",
                fix="Add atomic_thread_fence(Acquire) / fence(Ordering::Acquire) before deallocation",
                pattern_violated="refcount requires Acquire fence before dealloc",
            ))

    # Check for excessive SeqCst usage
    seqcst_count = sum(1 for _, o in ordering_matches if _parse_ordering(o) == OrderingLevel.SEQ_CST)
    total_orderings = len(ordering_matches)
    if total_orderings > 2 and seqcst_count == total_orderings:
        issues.append(VerificationIssue(
            line=0, description="All atomic operations use SeqCst — likely over-synchronized",
            severity="WARNING",
            fix="Consider using Relaxed for counters, Acquire/Release for synchronization pairs",
            pattern_violated="performance: unnecessary SeqCst",
        ))
        suggestions.append("Review each atomic operation to determine if a weaker ordering suffices.")

    # General suggestions
    if "counter" in detected:
        suggestions.append("Counter pattern detected — Relaxed ordering is typically sufficient.")
    if "flag" in detected or "message_passing" in detected:
        suggestions.append("Flag/message passing detected — ensure Release on store, Acquire on load.")
    if "refcount" in detected:
        suggestions.append("Refcount detected — use Relaxed increment, Release decrement, Acquire fence before dealloc.")
    if not detected or detected == ["unknown_atomic_usage"]:
        suggestions.append("Could not identify a specific pattern. Review orderings manually.")

    correct = len([i for i in issues if i.severity == "ERROR"]) == 0
    return VerificationResult(
        source=source[:80] + "..." if len(source) > 80 else source,
        language=lang_enum,
        issues=issues,
        patterns_detected=detected,
        correct=correct,
        suggestions=suggestions,
    )
