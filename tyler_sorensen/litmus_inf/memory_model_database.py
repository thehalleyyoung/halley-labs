"""
Memory model database for comparing hardware and language memory models.

Encodes reordering rules, axioms, fences, and litmus test behaviors for
x86-TSO, SPARC, ARM, RISC-V, PTX, Vulkan, OpenCL, Java, C++11, Go, and Rust.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set
import re


class ModelStrength(Enum):
    SC = "sc"
    TSO = "tso"
    PSO = "pso"
    RMO = "rmo"
    RELAXED = "relaxed"


class FenceType(Enum):
    FULL = "full"
    STORE_STORE = "store_store"
    LOAD_LOAD = "load_load"
    STORE_LOAD = "store_load"
    LOAD_STORE = "load_store"
    ACQUIRE = "acquire"
    RELEASE = "release"
    ACQ_REL = "acq_rel"
    SEQ_CST = "seq_cst"
    DMB_ISH = "dmb_ish"
    DMB_ISHST = "dmb_ishst"
    DMB_ISHLD = "dmb_ishld"
    FENCE_RW = "fence_rw"
    FENCE_WW = "fence_ww"
    FENCE_RR = "fence_rr"
    MEMBAR_GL = "membar_gl"
    MEMBAR_CTA = "membar_cta"


class AxiomKind(Enum):
    COHERENCE = "coherence"
    CAUSALITY = "causality"
    MULTI_COPY_ATOMICITY = "multi_copy_atomicity"
    STORE_ORDERING = "store_ordering"
    LOAD_ORDERING = "load_ordering"
    DEPENDENCY = "dependency"
    BARRIER = "barrier"
    SCOPE = "scope"


_STRENGTH_ORDER = [
    ModelStrength.SC,
    ModelStrength.TSO,
    ModelStrength.PSO,
    ModelStrength.RMO,
    ModelStrength.RELAXED,
]


@dataclass
class Axiom:
    kind: AxiomKind
    name: str
    description: str
    formal_def: str

    def __str__(self) -> str:
        return f"{self.name} ({self.kind.value}): {self.description}"


@dataclass
class Fence:
    fence_type: FenceType
    cost_cycles: int
    architectures: List[str] = field(default_factory=list)
    description: str = ""

    def __str__(self) -> str:
        archs = ", ".join(self.architectures)
        return f"{self.fence_type.value} (~{self.cost_cycles} cyc) [{archs}]: {self.description}"


@dataclass
class Reordering:
    store_store: bool = False
    store_load: bool = False
    load_load: bool = False
    load_store: bool = False
    dependent_loads: bool = False

    def __str__(self) -> str:
        pairs = [
            ("St->St", self.store_store),
            ("St->Ld", self.store_load),
            ("Ld->Ld", self.load_load),
            ("Ld->St", self.load_store),
            ("DepLd", self.dependent_loads),
        ]
        allowed = [n for n, v in pairs if v]
        preserved = [n for n, v in pairs if not v]
        return f"reorder_allowed=[{', '.join(allowed)}] preserved=[{', '.join(preserved)}]"

    def allows_any(self) -> bool:
        return any([
            self.store_store, self.store_load,
            self.load_load, self.load_store,
            self.dependent_loads,
        ])

    def reorder_count(self) -> int:
        return sum([
            self.store_store, self.store_load,
            self.load_load, self.load_store,
            self.dependent_loads,
        ])


@dataclass
class MemoryModel:
    name: str
    arch: str
    strength: ModelStrength
    reorderings: Reordering
    axioms: List[Axiom] = field(default_factory=list)
    fences: List[Fence] = field(default_factory=list)
    description: str = ""
    multi_copy_atomic: bool = True
    has_dependency_ordering: bool = False

    def __str__(self) -> str:
        lines = [
            f"MemoryModel({self.name}, arch={self.arch}, "
            f"strength={self.strength.value})",
            f"  {self.reorderings}",
            f"  multi_copy_atomic={self.multi_copy_atomic}",
            f"  axioms={len(self.axioms)}, fences={len(self.fences)}",
        ]
        return "\n".join(lines)


@dataclass
class ModelComparison:
    model_a: str
    model_b: str
    a_allows_not_b: List[str] = field(default_factory=list)
    b_allows_not_a: List[str] = field(default_factory=list)
    common_behaviors: List[str] = field(default_factory=list)
    distinguishing_tests: List[str] = field(default_factory=list)
    fence_mapping: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Comparison: {self.model_a} vs {self.model_b}",
            f"  {self.model_a} allows but not {self.model_b}: "
            f"{self.a_allows_not_b}",
            f"  {self.model_b} allows but not {self.model_a}: "
            f"{self.b_allows_not_a}",
            f"  common behaviors: {self.common_behaviors}",
            f"  distinguishing tests: {self.distinguishing_tests}",
        ]
        return "\n".join(lines)


@dataclass
class Behavior:
    test_name: str
    outcome: Dict[str, int]
    allowed: bool
    explanation: str

    def __str__(self) -> str:
        vals = ", ".join(f"{k}={v}" for k, v in self.outcome.items())
        tag = "allowed" if self.allowed else "forbidden"
        return f"{self.test_name} [{tag}] ({vals}): {self.explanation}"


# ── Litmus test encodings ──────────────────────────────────────────────

_LITMUS_TESTS: Dict[str, Dict] = {
    "mp": {
        "desc": "Message passing: W x=1; W y=1 || R y; R x",
        "relaxed_outcome": {"r1": 1, "r2": 0},
        "requires_store_load": False,
        "requires_store_store": True,
        "requires_load_load": True,
        "requires_load_store": False,
        "requires_dep": False,
    },
    "sb": {
        "desc": "Store buffering: W x=1; R y || W y=1; R x",
        "relaxed_outcome": {"r1": 0, "r2": 0},
        "requires_store_load": True,
        "requires_store_store": False,
        "requires_load_load": False,
        "requires_load_store": False,
        "requires_dep": False,
    },
    "lb": {
        "desc": "Load buffering: R x; W y=1 || R y; W x=1",
        "relaxed_outcome": {"r1": 1, "r2": 1},
        "requires_store_load": False,
        "requires_store_store": False,
        "requires_load_load": False,
        "requires_load_store": True,
        "requires_dep": False,
    },
    "iriw": {
        "desc": "Independent reads of independent writes",
        "relaxed_outcome": {"r1": 1, "r2": 0, "r3": 1, "r4": 0},
        "requires_store_load": False,
        "requires_store_store": False,
        "requires_load_load": True,
        "requires_load_store": False,
        "requires_dep": False,
        "requires_mca": True,
    },
    "2+2w": {
        "desc": "Two writes per thread: W x=1; W y=2 || W y=1; W x=2",
        "relaxed_outcome": {"x": 1, "y": 1},
        "requires_store_load": False,
        "requires_store_store": True,
        "requires_load_load": False,
        "requires_load_store": False,
        "requires_dep": False,
    },
    "rwc": {
        "desc": "Read-write causality: W x=1 || R x; W y=1 || R y; R x",
        "relaxed_outcome": {"r1": 1, "r2": 1, "r3": 0},
        "requires_store_load": True,
        "requires_store_store": False,
        "requires_load_load": True,
        "requires_load_store": False,
        "requires_dep": False,
        "requires_mca": True,
    },
    "wrc": {
        "desc": "Write-read causality: W x=1 || R x; W y=1 || R y; R x",
        "relaxed_outcome": {"r1": 1, "r2": 1, "r3": 0},
        "requires_store_load": False,
        "requires_store_store": False,
        "requires_load_load": True,
        "requires_load_store": False,
        "requires_dep": False,
        "requires_mca": True,
    },
    "dekker": {
        "desc": "Dekker's algorithm: W flag0=1; R flag1 || W flag1=1; R flag0",
        "relaxed_outcome": {"r1": 0, "r2": 0},
        "requires_store_load": True,
        "requires_store_store": False,
        "requires_load_load": False,
        "requires_load_store": False,
        "requires_dep": False,
    },
    "peterson": {
        "desc": "Peterson's lock: W flag[i]=1; W turn=j; R turn; R flag[j]",
        "relaxed_outcome": {"r1": 0, "r2": 0},
        "requires_store_load": True,
        "requires_store_store": True,
        "requires_load_load": True,
        "requires_load_store": False,
        "requires_dep": False,
    },
}


# ── Main database class ───────────────────────────────────────────────

class MemoryModelDB:
    """Database of hardware and language memory models with comparison tools."""

    def __init__(self) -> None:
        self._models: Dict[str, MemoryModel] = {}
        self._build_x86_tso()
        self._build_sparc_tso()
        self._build_sparc_pso()
        self._build_sparc_rmo()
        self._build_arm()
        self._build_riscv()
        self._build_ptx_cta()
        self._build_ptx_gpu()
        self._build_vulkan_wg()
        self._build_vulkan_dev()
        self._build_opencl_wg()
        self._build_opencl_dev()
        self._build_java_jmm()
        self._build_cpp11()
        self._build_go()
        self._build_rust()

    def __str__(self) -> str:
        lines = [f"MemoryModelDB ({len(self._models)} models):"]
        for name, m in sorted(self._models.items()):
            lines.append(f"  {m.name} [{m.strength.value}]")
        return "\n".join(lines)

    # ── Model builders ─────────────────────────────────────────────

    def _build_x86_tso(self) -> None:
        self._models["x86-tso"] = MemoryModel(
            name="x86-TSO",
            arch="x86-tso",
            strength=ModelStrength.TSO,
            reorderings=Reordering(
                store_store=False, store_load=True,
                load_load=False, load_store=False,
                dependent_loads=False,
            ),
            axioms=[
                Axiom(AxiomKind.STORE_ORDERING, "TotalStoreOrder",
                      "All stores are observed in a single total order by all processors",
                      "∀ p q: p ≠ q → StoreOrder(p) = StoreOrder(q)"),
                Axiom(AxiomKind.COHERENCE, "StoreBufferForwarding",
                      "A processor may read its own store before it is globally visible",
                      "∀ p: Read(p,x) may return WriteBuffer(p,x) before flush"),
                Axiom(AxiomKind.LOAD_ORDERING, "LoadLoadOrder",
                      "Loads are not reordered with respect to other loads",
                      "∀ a b: po(a,b) ∧ IsLoad(a) ∧ IsLoad(b) → a <_m b"),
                Axiom(AxiomKind.STORE_ORDERING, "StoreStoreOrder",
                      "Stores are not reordered with respect to other stores",
                      "∀ a b: po(a,b) ∧ IsStore(a) ∧ IsStore(b) → a <_m b"),
                Axiom(AxiomKind.MULTI_COPY_ATOMICITY, "MultiCopyAtomic",
                      "Stores become visible to all processors at the same time",
                      "∀ w: ∀ p q: vis(w,p) ∧ vis(w,q) → same_instant"),
            ],
            fences=[
                Fence(FenceType.FULL, 33, ["x86"],
                      "MFENCE: full serializing barrier"),
                Fence(FenceType.STORE_LOAD, 33, ["x86"],
                      "MFENCE used to prevent store-load reordering"),
                Fence(FenceType.STORE_STORE, 0, ["x86"],
                      "SFENCE: store fence (nop on x86-TSO, only for NT stores)"),
                Fence(FenceType.LOAD_LOAD, 0, ["x86"],
                      "LFENCE: load fence (nop on x86-TSO for normal loads)"),
            ],
            description="x86 Total Store Order: only store-load reordering, "
                        "write buffer forwarding, multi-copy atomic",
            multi_copy_atomic=True,
            has_dependency_ordering=False,
        )

    def _build_sparc_tso(self) -> None:
        self._models["sparc-tso"] = MemoryModel(
            name="SPARC-TSO",
            arch="sparc-tso",
            strength=ModelStrength.TSO,
            reorderings=Reordering(
                store_store=False, store_load=True,
                load_load=False, load_store=False,
                dependent_loads=False,
            ),
            axioms=[
                Axiom(AxiomKind.STORE_ORDERING, "TotalStoreOrder",
                      "Total order on all stores",
                      "∀ w1 w2: w1 <_store w2 ∨ w2 <_store w1"),
                Axiom(AxiomKind.COHERENCE, "StoreBufferFwd",
                      "Processor can read own buffered store",
                      "Read(p,x) may see WriteBuffer(p,x)"),
                Axiom(AxiomKind.LOAD_ORDERING, "LoadOrder",
                      "Loads respect program order",
                      "po(Ld_a, Ld_b) → Ld_a <_m Ld_b"),
                Axiom(AxiomKind.MULTI_COPY_ATOMICITY, "MCA",
                      "Stores are multi-copy atomic",
                      "vis(w,p) simultaneous for all p"),
            ],
            fences=[
                Fence(FenceType.FULL, 35, ["sparc"],
                      "MEMBAR #StoreLoad|#StoreStore|#LoadLoad|#LoadStore"),
                Fence(FenceType.STORE_LOAD, 35, ["sparc"],
                      "MEMBAR #StoreLoad"),
            ],
            description="SPARC TSO mode: equivalent to x86-TSO",
            multi_copy_atomic=True,
            has_dependency_ordering=False,
        )

    def _build_sparc_pso(self) -> None:
        self._models["sparc-pso"] = MemoryModel(
            name="SPARC-PSO",
            arch="sparc-pso",
            strength=ModelStrength.PSO,
            reorderings=Reordering(
                store_store=True, store_load=True,
                load_load=False, load_store=False,
                dependent_loads=False,
            ),
            axioms=[
                Axiom(AxiomKind.STORE_ORDERING, "PartialStoreOrder",
                      "Stores to different addresses may be reordered",
                      "po(St_x, St_y) ∧ x≠y → ¬(St_x <_m St_y) possible"),
                Axiom(AxiomKind.COHERENCE, "PerLocCoherence",
                      "Stores to the same address are coherent",
                      "co(w1,w2) ∧ loc(w1)=loc(w2) → w1 <_co w2 total"),
                Axiom(AxiomKind.LOAD_ORDERING, "LoadOrder",
                      "Loads are ordered in program order",
                      "po(Ld_a, Ld_b) → Ld_a <_m Ld_b"),
            ],
            fences=[
                Fence(FenceType.FULL, 35, ["sparc"],
                      "MEMBAR full barrier"),
                Fence(FenceType.STORE_STORE, 20, ["sparc"],
                      "MEMBAR #StoreStore"),
                Fence(FenceType.STORE_LOAD, 35, ["sparc"],
                      "MEMBAR #StoreLoad"),
            ],
            description="SPARC PSO: stores may be reordered, loads preserved",
            multi_copy_atomic=True,
            has_dependency_ordering=False,
        )

    def _build_sparc_rmo(self) -> None:
        self._models["sparc-rmo"] = MemoryModel(
            name="SPARC-RMO",
            arch="sparc-rmo",
            strength=ModelStrength.RMO,
            reorderings=Reordering(
                store_store=True, store_load=True,
                load_load=True, load_store=True,
                dependent_loads=False,
            ),
            axioms=[
                Axiom(AxiomKind.COHERENCE, "PerLocCoherence",
                      "Per-location coherence maintained",
                      "co is total per location"),
                Axiom(AxiomKind.DEPENDENCY, "DataDep",
                      "Address dependencies respected",
                      "addr(a,b) → a <_m b"),
                Axiom(AxiomKind.BARRIER, "MembarOrdering",
                      "MEMBAR instructions enforce ordering",
                      "fence(a,b) → a <_m b"),
            ],
            fences=[
                Fence(FenceType.FULL, 35, ["sparc"],
                      "MEMBAR full barrier"),
                Fence(FenceType.STORE_STORE, 20, ["sparc"],
                      "MEMBAR #StoreStore"),
                Fence(FenceType.LOAD_LOAD, 15, ["sparc"],
                      "MEMBAR #LoadLoad"),
                Fence(FenceType.STORE_LOAD, 35, ["sparc"],
                      "MEMBAR #StoreLoad"),
                Fence(FenceType.LOAD_STORE, 15, ["sparc"],
                      "MEMBAR #LoadStore"),
            ],
            description="SPARC RMO: all reorderings except dependent loads",
            multi_copy_atomic=True,
            has_dependency_ordering=True,
        )

    def _build_arm(self) -> None:
        self._models["arm"] = MemoryModel(
            name="ARM",
            arch="arm",
            strength=ModelStrength.RELAXED,
            reorderings=Reordering(
                store_store=True, store_load=True,
                load_load=True, load_store=True,
                dependent_loads=False,
            ),
            axioms=[
                Axiom(AxiomKind.COHERENCE, "CoherenceOrder",
                      "Per-location coherence is maintained",
                      "co is a total order per location"),
                Axiom(AxiomKind.DEPENDENCY, "AddrDependency",
                      "Address dependencies create ordering",
                      "addr(a,b) → a <_ob b"),
                Axiom(AxiomKind.DEPENDENCY, "DataDependency",
                      "Data dependencies create ordering",
                      "data(a,b) → a <_ob b"),
                Axiom(AxiomKind.DEPENDENCY, "CtrlIsb",
                      "Control dependency + ISB creates ordering",
                      "ctrl(a,b) ∧ isb → a <_ob b"),
                Axiom(AxiomKind.BARRIER, "DMB_Ordering",
                      "DMB enforces ordering between accesses",
                      "dmb(a,b) → a <_ob b"),
                Axiom(AxiomKind.MULTI_COPY_ATOMICITY, "NonMCA",
                      "ARM is NOT multi-copy atomic",
                      "∃ w p q: vis(w,p) before vis(w,q)"),
            ],
            fences=[
                Fence(FenceType.DMB_ISH, 40, ["arm"],
                      "DMB ISH: full barrier inner-shareable domain"),
                Fence(FenceType.DMB_ISHST, 25, ["arm"],
                      "DMB ISHST: store-store barrier inner-shareable"),
                Fence(FenceType.DMB_ISHLD, 20, ["arm"],
                      "DMB ISHLD: load-load/load-store barrier inner-shareable"),
                Fence(FenceType.FULL, 40, ["arm"],
                      "DSB: data synchronization barrier"),
                Fence(FenceType.ACQUIRE, 10, ["arm"],
                      "LDAR: load-acquire"),
                Fence(FenceType.RELEASE, 10, ["arm"],
                      "STLR: store-release"),
            ],
            description="ARM relaxed model: all reorderings except data/addr "
                        "dependencies, non-multi-copy-atomic",
            multi_copy_atomic=False,
            has_dependency_ordering=True,
        )

    def _build_riscv(self) -> None:
        self._models["riscv"] = MemoryModel(
            name="RISC-V",
            arch="riscv",
            strength=ModelStrength.RELAXED,
            reorderings=Reordering(
                store_store=True, store_load=True,
                load_load=True, load_store=True,
                dependent_loads=False,
            ),
            axioms=[
                Axiom(AxiomKind.COHERENCE, "CoherenceOrder",
                      "Per-location coherence maintained (RVWMO)",
                      "co total per location"),
                Axiom(AxiomKind.DEPENDENCY, "AddrDep",
                      "Address dependencies preserved",
                      "addr(a,b) → a <_ppo b"),
                Axiom(AxiomKind.DEPENDENCY, "DataDep",
                      "Data dependencies preserved",
                      "data(a,b) → a <_ppo b"),
                Axiom(AxiomKind.DEPENDENCY, "CtrlFenceDep",
                      "Control dependency to store is preserved",
                      "ctrl(a,b) ∧ IsStore(b) → a <_ppo b"),
                Axiom(AxiomKind.BARRIER, "FenceOrdering",
                      "FENCE instruction orders specified access types",
                      "fence(pred,succ) → ∀ a∈pred, b∈succ: a <_m b"),
                Axiom(AxiomKind.MULTI_COPY_ATOMICITY, "RVWMO_MCA",
                      "RISC-V RVWMO is multi-copy atomic",
                      "stores become visible to all harts simultaneously"),
            ],
            fences=[
                Fence(FenceType.FULL, 30, ["riscv"],
                      "FENCE rw,rw: full read-write barrier"),
                Fence(FenceType.FENCE_RW, 30, ["riscv"],
                      "FENCE rw,rw: orders all prior r/w before subsequent r/w"),
                Fence(FenceType.FENCE_WW, 15, ["riscv"],
                      "FENCE w,w: store-store fence"),
                Fence(FenceType.FENCE_RR, 12, ["riscv"],
                      "FENCE r,r: load-load fence"),
                Fence(FenceType.ACQUIRE, 8, ["riscv"],
                      "LR.aq / AMOSWAP.aq: acquire semantics"),
                Fence(FenceType.RELEASE, 8, ["riscv"],
                      "SC.rl / AMOSWAP.rl: release semantics"),
                Fence(FenceType.SEQ_CST, 30, ["riscv"],
                      "FENCE rw,rw + LR.aqrl/SC.aqrl: seq-cst ordering"),
            ],
            description="RISC-V RVWMO: relaxed model with FENCE instruction, "
                        "multi-copy atomic, dependency ordering",
            multi_copy_atomic=True,
            has_dependency_ordering=True,
        )

    def _build_ptx_cta(self) -> None:
        self._models["ptx-cta"] = MemoryModel(
            name="PTX-CTA",
            arch="ptx-cta",
            strength=ModelStrength.RELAXED,
            reorderings=Reordering(
                store_store=True, store_load=True,
                load_load=True, load_store=True,
                dependent_loads=False,
            ),
            axioms=[
                Axiom(AxiomKind.SCOPE, "CTAScope",
                      "Ordering scoped to cooperative thread array",
                      "vis_cta(w, t) for t ∈ same CTA"),
                Axiom(AxiomKind.COHERENCE, "CTACoherence",
                      "Per-location coherence within CTA",
                      "co_cta total per location within CTA"),
                Axiom(AxiomKind.BARRIER, "MembarCTA",
                      "membar.cta orders within CTA scope",
                      "membar.cta(a,b) → a <_cta b"),
            ],
            fences=[
                Fence(FenceType.MEMBAR_CTA, 20, ["ptx"],
                      "membar.cta: barrier scoped to CTA (threadblock)"),
                Fence(FenceType.MEMBAR_GL, 50, ["ptx"],
                      "membar.gl: barrier scoped to GPU device"),
                Fence(FenceType.FULL, 50, ["ptx"],
                      "membar.sys: system-wide barrier"),
            ],
            description="NVIDIA PTX CTA scope: relaxed within threadblock, "
                        "membar.cta for intra-block ordering",
            multi_copy_atomic=False,
            has_dependency_ordering=True,
        )

    def _build_ptx_gpu(self) -> None:
        self._models["ptx-gpu"] = MemoryModel(
            name="PTX-GPU",
            arch="ptx-gpu",
            strength=ModelStrength.RELAXED,
            reorderings=Reordering(
                store_store=True, store_load=True,
                load_load=True, load_store=True,
                dependent_loads=False,
            ),
            axioms=[
                Axiom(AxiomKind.SCOPE, "GPUScope",
                      "Ordering scoped to entire GPU device",
                      "vis_gpu(w, t) for all t on same GPU"),
                Axiom(AxiomKind.COHERENCE, "GPUCoherence",
                      "Per-location coherence at GPU scope",
                      "co_gpu total per location across CTAs"),
                Axiom(AxiomKind.BARRIER, "MembarGL",
                      "membar.gl orders at GPU scope",
                      "membar.gl(a,b) → a <_gpu b"),
                Axiom(AxiomKind.MULTI_COPY_ATOMICITY, "ScopedMCA",
                      "Multi-copy atomicity scoped to GPU",
                      "vis_gpu(w) is atomic within GPU"),
            ],
            fences=[
                Fence(FenceType.MEMBAR_GL, 50, ["ptx"],
                      "membar.gl: barrier at GPU device scope"),
                Fence(FenceType.FULL, 80, ["ptx"],
                      "membar.sys: system-wide full barrier"),
            ],
            description="NVIDIA PTX GPU scope: relaxed model, membar.gl for "
                        "cross-CTA ordering within GPU",
            multi_copy_atomic=False,
            has_dependency_ordering=True,
        )

    def _build_vulkan_wg(self) -> None:
        self._models["vulkan-wg"] = MemoryModel(
            name="Vulkan-WorkGroup",
            arch="vulkan-wg",
            strength=ModelStrength.RELAXED,
            reorderings=Reordering(
                store_store=True, store_load=True,
                load_load=True, load_store=True,
                dependent_loads=False,
            ),
            axioms=[
                Axiom(AxiomKind.SCOPE, "WorkGroupScope",
                      "Memory ordering scoped to workgroup",
                      "vis_wg(w, inv) for inv ∈ same workgroup"),
                Axiom(AxiomKind.BARRIER, "ControlBarrier",
                      "OpControlBarrier synchronizes within workgroup",
                      "barrier_wg(a,b) → a <_wg b"),
                Axiom(AxiomKind.COHERENCE, "WGCoherence",
                      "Coherence within workgroup scope",
                      "co_wg total per location in workgroup"),
            ],
            fences=[
                Fence(FenceType.FULL, 30, ["vulkan"],
                      "OpControlBarrier Workgroup: full workgroup barrier"),
                Fence(FenceType.ACQUIRE, 10, ["vulkan"],
                      "OpMemoryBarrier Acquire workgroup scope"),
                Fence(FenceType.RELEASE, 10, ["vulkan"],
                      "OpMemoryBarrier Release workgroup scope"),
            ],
            description="Vulkan workgroup scope: relaxed with barriers and "
                        "acquire/release semantics via SPIR-V",
            multi_copy_atomic=False,
            has_dependency_ordering=False,
        )

    def _build_vulkan_dev(self) -> None:
        self._models["vulkan-dev"] = MemoryModel(
            name="Vulkan-Device",
            arch="vulkan-dev",
            strength=ModelStrength.RELAXED,
            reorderings=Reordering(
                store_store=True, store_load=True,
                load_load=True, load_store=True,
                dependent_loads=False,
            ),
            axioms=[
                Axiom(AxiomKind.SCOPE, "DeviceScope",
                      "Memory ordering scoped to device",
                      "vis_dev(w, inv) for all inv on device"),
                Axiom(AxiomKind.BARRIER, "DeviceBarrier",
                      "OpControlBarrier at device scope",
                      "barrier_dev(a,b) → a <_dev b"),
                Axiom(AxiomKind.COHERENCE, "DevCoherence",
                      "Device-scoped coherence",
                      "co_dev total per location on device"),
                Axiom(AxiomKind.CAUSALITY, "DevCausality",
                      "Causal ordering at device scope via release/acquire",
                      "rel(a) → acq(b) → a <_dev b"),
            ],
            fences=[
                Fence(FenceType.FULL, 60, ["vulkan"],
                      "OpControlBarrier Device: device-wide barrier"),
                Fence(FenceType.ACQUIRE, 15, ["vulkan"],
                      "OpMemoryBarrier Acquire device scope"),
                Fence(FenceType.RELEASE, 15, ["vulkan"],
                      "OpMemoryBarrier Release device scope"),
                Fence(FenceType.ACQ_REL, 25, ["vulkan"],
                      "OpMemoryBarrier AcquireRelease device scope"),
            ],
            description="Vulkan device scope: relaxed, requires explicit "
                        "barriers for cross-workgroup ordering",
            multi_copy_atomic=False,
            has_dependency_ordering=False,
        )

    def _build_opencl_wg(self) -> None:
        self._models["opencl-wg"] = MemoryModel(
            name="OpenCL-WorkGroup",
            arch="opencl-wg",
            strength=ModelStrength.RELAXED,
            reorderings=Reordering(
                store_store=True, store_load=True,
                load_load=True, load_store=True,
                dependent_loads=False,
            ),
            axioms=[
                Axiom(AxiomKind.SCOPE, "WGScope",
                      "Ordering within work-group scope",
                      "vis_wg(w, wi) for wi ∈ same work-group"),
                Axiom(AxiomKind.BARRIER, "WGBarrier",
                      "work_group_barrier orders within work-group",
                      "wg_barrier(a,b) → a <_wg b"),
                Axiom(AxiomKind.COHERENCE, "LocalCoherence",
                      "Per-location coherence in local memory",
                      "co_local total per location"),
            ],
            fences=[
                Fence(FenceType.FULL, 25, ["opencl"],
                      "work_group_barrier(CLK_GLOBAL_MEM_FENCE)"),
                Fence(FenceType.ACQUIRE, 8, ["opencl"],
                      "atomic_work_item_fence acquire work-group"),
                Fence(FenceType.RELEASE, 8, ["opencl"],
                      "atomic_work_item_fence release work-group"),
            ],
            description="OpenCL work-group scope: relaxed with work_group_barrier "
                        "and atomic fences for local ordering",
            multi_copy_atomic=False,
            has_dependency_ordering=False,
        )

    def _build_opencl_dev(self) -> None:
        self._models["opencl-dev"] = MemoryModel(
            name="OpenCL-Device",
            arch="opencl-dev",
            strength=ModelStrength.RELAXED,
            reorderings=Reordering(
                store_store=True, store_load=True,
                load_load=True, load_store=True,
                dependent_loads=False,
            ),
            axioms=[
                Axiom(AxiomKind.SCOPE, "DevScope",
                      "Ordering within device scope",
                      "vis_dev(w, wi) for all wi on device"),
                Axiom(AxiomKind.BARRIER, "DevBarrier",
                      "atomic_work_item_fence at device scope",
                      "dev_fence(a,b) → a <_dev b"),
                Axiom(AxiomKind.COHERENCE, "GlobalCoherence",
                      "Per-location coherence in global memory",
                      "co_global total per location"),
                Axiom(AxiomKind.CAUSALITY, "AcqRelCausality",
                      "Release-acquire chains establish causality",
                      "rel(a); acq(b) → a happens-before b"),
            ],
            fences=[
                Fence(FenceType.FULL, 55, ["opencl"],
                      "atomic_work_item_fence seq_cst device scope"),
                Fence(FenceType.ACQUIRE, 12, ["opencl"],
                      "atomic_work_item_fence acquire device scope"),
                Fence(FenceType.RELEASE, 12, ["opencl"],
                      "atomic_work_item_fence release device scope"),
                Fence(FenceType.ACQ_REL, 20, ["opencl"],
                      "atomic_work_item_fence acq_rel device scope"),
            ],
            description="OpenCL device scope: relaxed, explicit fences needed "
                        "for cross-work-group ordering",
            multi_copy_atomic=False,
            has_dependency_ordering=False,
        )

    def _build_java_jmm(self) -> None:
        self._models["java-jmm"] = MemoryModel(
            name="Java-JMM",
            arch="java-jmm",
            strength=ModelStrength.RELAXED,
            reorderings=Reordering(
                store_store=True, store_load=True,
                load_load=True, load_store=True,
                dependent_loads=False,
            ),
            axioms=[
                Axiom(AxiomKind.CAUSALITY, "HappensBefore",
                      "Happens-before partial order defines visibility",
                      "hb(a,b) = po(a,b) ∪ sw(a,b) ∪ transitivity"),
                Axiom(AxiomKind.STORE_ORDERING, "SynchronizationOrder",
                      "Total order on synchronization actions",
                      "so is total on all sync actions, consistent with hb"),
                Axiom(AxiomKind.COHERENCE, "ProgramOrder",
                      "Intra-thread semantics as-if-serial",
                      "within thread, result = sequential execution"),
                Axiom(AxiomKind.CAUSALITY, "CausalityRequirement",
                      "Committed writes must have causal justification",
                      "each committed write has hb-justification chain"),
                Axiom(AxiomKind.BARRIER, "VolatileOrdering",
                      "Volatile reads/writes act as acquire/release",
                      "volatile_write sw volatile_read → hb edge"),
            ],
            fences=[
                Fence(FenceType.SEQ_CST, 30, ["java"],
                      "volatile: sequential consistency for volatiles"),
                Fence(FenceType.FULL, 40, ["java"],
                      "synchronized block: monitor enter/exit acts as full fence"),
                Fence(FenceType.ACQUIRE, 15, ["java"],
                      "monitor enter / volatile read: acquire semantics"),
                Fence(FenceType.RELEASE, 15, ["java"],
                      "monitor exit / volatile write: release semantics"),
            ],
            description="Java Memory Model: happens-before based, volatile for "
                        "ordering, synchronized blocks for mutual exclusion",
            multi_copy_atomic=True,
            has_dependency_ordering=False,
        )

    def _build_cpp11(self) -> None:
        self._models["c++11"] = MemoryModel(
            name="C++11",
            arch="c++11",
            strength=ModelStrength.RELAXED,
            reorderings=Reordering(
                store_store=True, store_load=True,
                load_load=True, load_store=True,
                dependent_loads=False,
            ),
            axioms=[
                Axiom(AxiomKind.CAUSALITY, "HappensBefore",
                      "Happens-before ordering from sequenced-before and "
                      "synchronizes-with",
                      "hb = sb ∪ sw ∪ (sb;sw) ∪ (sw;sb)"),
                Axiom(AxiomKind.COHERENCE, "CoherenceOrder",
                      "Modification order total per atomic object",
                      "mo is total per atomic variable"),
                Axiom(AxiomKind.STORE_ORDERING, "SeqCstOrder",
                      "Total order S on all seq_cst operations",
                      "S is consistent with hb and mo"),
                Axiom(AxiomKind.CAUSALITY, "RelAcqSync",
                      "Release store synchronizes-with acquire load",
                      "rel(w) reads-from acq(r) → w sw r"),
                Axiom(AxiomKind.DEPENDENCY, "Consume",
                      "Release to consume via dependency chain",
                      "rel(w) dep-ordered-before consume(r) → w dob r"),
                Axiom(AxiomKind.MULTI_COPY_ATOMICITY, "SeqCstMCA",
                      "seq_cst operations are multi-copy atomic",
                      "seq_cst total order observed consistently"),
            ],
            fences=[
                Fence(FenceType.SEQ_CST, 30, ["c++"],
                      "atomic_thread_fence(memory_order_seq_cst)"),
                Fence(FenceType.ACQ_REL, 20, ["c++"],
                      "atomic_thread_fence(memory_order_acq_rel)"),
                Fence(FenceType.ACQUIRE, 10, ["c++"],
                      "atomic_thread_fence(memory_order_acquire)"),
                Fence(FenceType.RELEASE, 10, ["c++"],
                      "atomic_thread_fence(memory_order_release)"),
            ],
            description="C++11 memory model: relaxed by default, explicit "
                        "memory_order on atomics, seq_cst/acq/rel/relaxed",
            multi_copy_atomic=True,
            has_dependency_ordering=True,
        )

    def _build_go(self) -> None:
        self._models["go"] = MemoryModel(
            name="Go",
            arch="go",
            strength=ModelStrength.RELAXED,
            reorderings=Reordering(
                store_store=True, store_load=True,
                load_load=True, load_store=True,
                dependent_loads=False,
            ),
            axioms=[
                Axiom(AxiomKind.CAUSALITY, "HappensBefore",
                      "Go happens-before from goroutine creation, channels, sync",
                      "hb = goroutine_create ∪ chan_send_recv ∪ sync_primitives"),
                Axiom(AxiomKind.COHERENCE, "SingleGoroutine",
                      "Within a goroutine, reads/writes as-if sequentially ordered",
                      "intra-goroutine order matches program order"),
                Axiom(AxiomKind.BARRIER, "ChannelSync",
                      "Channel send happens-before corresponding receive",
                      "ch <- v hb <-ch for buffered; reverse for unbuffered close"),
                Axiom(AxiomKind.CAUSALITY, "SyncOnce",
                      "sync.Once f() happens-before any Once.Do returns",
                      "once.Do(f) → f() hb all once.Do returns"),
            ],
            fences=[
                Fence(FenceType.FULL, 25, ["go"],
                      "channel operation: implicit full fence"),
                Fence(FenceType.ACQ_REL, 15, ["go"],
                      "sync.Mutex Lock/Unlock: acquire/release"),
                Fence(FenceType.SEQ_CST, 20, ["go"],
                      "sync/atomic operations: sequentially consistent"),
            ],
            description="Go memory model: happens-before via channels, sync "
                        "package, and sync/atomic; data races are undefined",
            multi_copy_atomic=True,
            has_dependency_ordering=False,
        )

    def _build_rust(self) -> None:
        self._models["rust"] = MemoryModel(
            name="Rust",
            arch="rust",
            strength=ModelStrength.RELAXED,
            reorderings=Reordering(
                store_store=True, store_load=True,
                load_load=True, load_store=True,
                dependent_loads=False,
            ),
            axioms=[
                Axiom(AxiomKind.CAUSALITY, "HappensBefore",
                      "Rust inherits C++11 happens-before via atomics",
                      "hb = sb ∪ sw ∪ transitivity"),
                Axiom(AxiomKind.COHERENCE, "ModificationOrder",
                      "Modification order per atomic variable",
                      "mo total per AtomicUsize/AtomicBool etc."),
                Axiom(AxiomKind.STORE_ORDERING, "SeqCstTotal",
                      "SeqCst operations in a single total order",
                      "S total on all SeqCst ops, consistent with hb"),
                Axiom(AxiomKind.CAUSALITY, "AcqRelSync",
                      "Release store synchronizes-with acquire load of same var",
                      "store(Release) rf load(Acquire) → sw edge"),
                Axiom(AxiomKind.BARRIER, "FenceOrdering",
                      "std::sync::atomic::fence provides ordering",
                      "fence(Ordering) acts as C++11 atomic_thread_fence"),
            ],
            fences=[
                Fence(FenceType.SEQ_CST, 30, ["rust"],
                      "fence(Ordering::SeqCst)"),
                Fence(FenceType.ACQ_REL, 20, ["rust"],
                      "fence(Ordering::AcqRel)"),
                Fence(FenceType.ACQUIRE, 10, ["rust"],
                      "fence(Ordering::Acquire)"),
                Fence(FenceType.RELEASE, 10, ["rust"],
                      "fence(Ordering::Release)"),
            ],
            description="Rust memory model: inherits C++11 model, data races "
                        "prevented by ownership/borrowing, atomics for shared state",
            multi_copy_atomic=True,
            has_dependency_ordering=True,
        )

    # ── Public API ─────────────────────────────────────────────────

    def get_model(self, arch: str) -> Optional[MemoryModel]:
        """Return the MemoryModel for *arch* (case-insensitive)."""
        key = arch.lower().strip()
        if key in self._models:
            return self._models[key]
        for k, m in self._models.items():
            if m.name.lower() == key or m.arch.lower() == key:
                return m
        return None

    def list_models(self) -> List[str]:
        """Return all registered architecture keys."""
        return sorted(self._models.keys())

    def compare_models(self, arch_a: str, arch_b: str) -> Optional[ModelComparison]:
        """Compute behavioral comparison between two models."""
        ma = self.get_model(arch_a)
        mb = self.get_model(arch_b)
        if ma is None or mb is None:
            return None

        a_allows: List[str] = []
        b_allows: List[str] = []
        common: List[str] = []
        distinguishing: List[str] = []

        for test_name in _LITMUS_TESTS:
            beh_a = self.allowed_behaviors(test_name, ma.arch)
            beh_b = self.allowed_behaviors(test_name, mb.arch)

            relaxed_a = any(b.allowed and not _is_sc_outcome(b) for b in beh_a)
            relaxed_b = any(b.allowed and not _is_sc_outcome(b) for b in beh_b)

            if relaxed_a and relaxed_b:
                common.append(test_name)
            elif relaxed_a and not relaxed_b:
                a_allows.append(test_name)
                distinguishing.append(test_name)
            elif relaxed_b and not relaxed_a:
                b_allows.append(test_name)
                distinguishing.append(test_name)
            else:
                common.append(test_name)

        fence_map = self._map_fences(ma, mb)

        return ModelComparison(
            model_a=ma.name,
            model_b=mb.name,
            a_allows_not_b=a_allows,
            b_allows_not_a=b_allows,
            common_behaviors=common,
            distinguishing_tests=distinguishing,
            fence_mapping=fence_map,
        )

    def allowed_behaviors(self, test: str, model_arch: str) -> List[Behavior]:
        """Compute which outcomes a model allows for a litmus test."""
        m = self.get_model(model_arch)
        if m is None:
            return []
        test_key = test.lower().strip()
        if test_key not in _LITMUS_TESTS:
            return []

        spec = _LITMUS_TESTS[test_key]
        behaviors: List[Behavior] = []

        sc_outcome = _sc_outcome_for(test_key)
        behaviors.append(Behavior(
            test_name=test_key,
            outcome=sc_outcome,
            allowed=True,
            explanation="SC outcome: always allowed under any model",
        ))

        relaxed_outcome = spec["relaxed_outcome"]
        relaxed_allowed = self._is_relaxed_allowed(m, spec)

        if relaxed_allowed:
            explanation = self._explain_relaxed(m, spec, test_key)
        else:
            explanation = self._explain_forbidden(m, spec, test_key)

        behaviors.append(Behavior(
            test_name=test_key,
            outcome=relaxed_outcome,
            allowed=relaxed_allowed,
            explanation=explanation,
        ))

        return behaviors

    def required_fences(self, test: str, model_arch: str) -> List[Fence]:
        """Compute fences needed to forbid relaxed behaviors for a test."""
        m = self.get_model(model_arch)
        if m is None:
            return []
        test_key = test.lower().strip()
        if test_key not in _LITMUS_TESTS:
            return []

        spec = _LITMUS_TESTS[test_key]
        if not self._is_relaxed_allowed(m, spec):
            return []

        needed: List[Fence] = []
        r = m.reorderings

        if spec.get("requires_store_load") and r.store_load:
            f = self._find_fence(m, FenceType.STORE_LOAD, FenceType.FULL,
                                 FenceType.DMB_ISH, FenceType.FENCE_RW,
                                 FenceType.SEQ_CST, FenceType.ACQ_REL,
                                 FenceType.MEMBAR_GL, FenceType.MEMBAR_CTA)
            if f:
                needed.append(f)

        if spec.get("requires_store_store") and r.store_store:
            f = self._find_fence(m, FenceType.STORE_STORE, FenceType.FULL,
                                 FenceType.DMB_ISHST, FenceType.FENCE_WW,
                                 FenceType.RELEASE, FenceType.MEMBAR_CTA)
            if f:
                needed.append(f)

        if spec.get("requires_load_load") and r.load_load:
            f = self._find_fence(m, FenceType.LOAD_LOAD, FenceType.FULL,
                                 FenceType.DMB_ISHLD, FenceType.FENCE_RR,
                                 FenceType.ACQUIRE, FenceType.MEMBAR_CTA)
            if f:
                needed.append(f)

        if spec.get("requires_load_store") and r.load_store:
            f = self._find_fence(m, FenceType.LOAD_STORE, FenceType.FULL,
                                 FenceType.DMB_ISH, FenceType.FENCE_RW,
                                 FenceType.ACQUIRE, FenceType.ACQ_REL)
            if f:
                needed.append(f)

        if spec.get("requires_mca") and not m.multi_copy_atomic:
            f = self._find_fence(m, FenceType.FULL, FenceType.DMB_ISH,
                                 FenceType.MEMBAR_GL, FenceType.SEQ_CST)
            if f and f not in needed:
                needed.append(f)

        if not needed and self._is_relaxed_allowed(m, spec):
            f = self._find_fence(m, FenceType.FULL, FenceType.SEQ_CST,
                                 FenceType.DMB_ISH, FenceType.FENCE_RW,
                                 FenceType.MEMBAR_GL, FenceType.ACQ_REL)
            if f:
                needed.append(f)

        return needed

    def weakest_to_strongest(self) -> List[str]:
        """Return architecture keys ordered from weakest to strongest model."""
        models = list(self._models.values())
        strength_rank = {s: i for i, s in enumerate(_STRENGTH_ORDER)}

        def sort_key(m: MemoryModel) -> Tuple[int, int, str]:
            rank = strength_rank.get(m.strength, len(_STRENGTH_ORDER))
            reorder_cnt = m.reorderings.reorder_count()
            mca_penalty = 0 if m.multi_copy_atomic else 1
            return (-rank, -reorder_cnt - mca_penalty, m.name)

        models.sort(key=sort_key)
        return [m.arch for m in models]

    def fence_cost_comparison(
        self, test: str, architectures: List[str]
    ) -> Dict[str, List[Tuple[str, int]]]:
        """Compare fence costs for a test across architectures.

        Returns {arch: [(fence_desc, cost_cycles), ...]} for each arch.
        """
        result: Dict[str, List[Tuple[str, int]]] = {}
        for arch in architectures:
            fences = self.required_fences(test, arch)
            if fences:
                result[arch] = [(f.description, f.cost_cycles) for f in fences]
            else:
                m = self.get_model(arch)
                if m is not None:
                    result[arch] = [("no fence needed", 0)]
                else:
                    result[arch] = [("unknown architecture", -1)]
        return result

    def generate_distinguishing_test(
        self, model_a: str, model_b: str
    ) -> Optional[str]:
        """Find a litmus test that distinguishes two models.

        Returns the test name where one model allows the relaxed outcome
        and the other forbids it, or None if no such test exists.
        """
        ma = self.get_model(model_a)
        mb = self.get_model(model_b)
        if ma is None or mb is None:
            return None

        preference = ["sb", "mp", "lb", "iriw", "2+2w", "rwc", "wrc",
                       "dekker", "peterson"]

        for test_name in preference:
            spec = _LITMUS_TESTS[test_name]
            a_relaxed = self._is_relaxed_allowed(ma, spec)
            b_relaxed = self._is_relaxed_allowed(mb, spec)
            if a_relaxed != b_relaxed:
                return test_name

        return None

    # ── Internal helpers ───────────────────────────────────────────

    def _compute_test_outcomes(
        self, test_name: str
    ) -> List[Dict[str, int]]:
        """Return all possible outcomes (SC + relaxed) for a test."""
        test_key = test_name.lower().strip()
        if test_key not in _LITMUS_TESTS:
            return []
        spec = _LITMUS_TESTS[test_key]
        sc = _sc_outcome_for(test_key)
        relaxed = spec["relaxed_outcome"]
        outcomes = [sc]
        if relaxed != sc:
            outcomes.append(relaxed)
        return outcomes

    def _is_relaxed_allowed(
        self, model: MemoryModel, spec: Dict
    ) -> bool:
        """Determine whether the relaxed outcome is allowed under *model*."""
        r = model.reorderings
        needs_mca = spec.get("requires_mca", False)

        if needs_mca and not model.multi_copy_atomic:
            return True

        if spec.get("requires_store_load") and r.store_load:
            return True
        if spec.get("requires_store_store") and r.store_store:
            return True
        if spec.get("requires_load_load") and r.load_load:
            return True
        if spec.get("requires_load_store") and r.load_store:
            return True
        if spec.get("requires_dep") and r.dependent_loads:
            return True

        return False

    def _explain_relaxed(
        self, model: MemoryModel, spec: Dict, test: str
    ) -> str:
        """Generate explanation of why relaxed outcome is allowed."""
        reasons: List[str] = []
        r = model.reorderings
        if spec.get("requires_store_load") and r.store_load:
            reasons.append("store-load reordering (store buffer)")
        if spec.get("requires_store_store") and r.store_store:
            reasons.append("store-store reordering")
        if spec.get("requires_load_load") and r.load_load:
            reasons.append("load-load reordering")
        if spec.get("requires_load_store") and r.load_store:
            reasons.append("load-store reordering")
        if spec.get("requires_mca") and not model.multi_copy_atomic:
            reasons.append("non-multi-copy-atomic stores")
        if not reasons:
            reasons.append("relaxed ordering allows reordering")
        return (f"{model.name} allows relaxed outcome for {test}: "
                + ", ".join(reasons))

    def _explain_forbidden(
        self, model: MemoryModel, spec: Dict, test: str
    ) -> str:
        """Generate explanation of why relaxed outcome is forbidden."""
        preserved: List[str] = []
        r = model.reorderings
        if spec.get("requires_store_load") and not r.store_load:
            preserved.append("store-load order preserved")
        if spec.get("requires_store_store") and not r.store_store:
            preserved.append("store-store order preserved")
        if spec.get("requires_load_load") and not r.load_load:
            preserved.append("load-load order preserved")
        if spec.get("requires_load_store") and not r.load_store:
            preserved.append("load-store order preserved")
        if spec.get("requires_mca") and model.multi_copy_atomic:
            preserved.append("multi-copy atomic")
        if not preserved:
            preserved.append("model preserves required orderings")
        return (f"{model.name} forbids relaxed outcome for {test}: "
                + ", ".join(preserved))

    def _find_fence(
        self, model: MemoryModel, *fence_types: FenceType
    ) -> Optional[Fence]:
        """Find cheapest fence from model matching any of the given types."""
        candidates: List[Fence] = []
        for ft in fence_types:
            for f in model.fences:
                if f.fence_type == ft:
                    candidates.append(f)
        if not candidates:
            return None
        candidates.sort(key=lambda f: f.cost_cycles)
        return candidates[0]

    def _map_fences(
        self, model_a: MemoryModel, model_b: MemoryModel
    ) -> Dict[str, str]:
        """Map fences from model_a to closest equivalents in model_b."""
        mapping: Dict[str, str] = {}
        for fa in model_a.fences:
            best: Optional[Fence] = None
            best_score = -1
            for fb in model_b.fences:
                score = _fence_similarity(fa.fence_type, fb.fence_type)
                if score > best_score:
                    best_score = score
                    best = fb
            if best is not None:
                mapping[fa.description] = best.description
            else:
                mapping[fa.description] = "(no equivalent)"
        return mapping


# ── Module-level helpers ──────────────────────────────────────────────

def _sc_outcome_for(test_name: str) -> Dict[str, int]:
    """Return the sequentially consistent outcome for a standard test."""
    sc_outcomes: Dict[str, Dict[str, int]] = {
        "mp": {"r1": 1, "r2": 1},
        "sb": {"r1": 1, "r2": 1},
        "lb": {"r1": 0, "r2": 0},
        "iriw": {"r1": 1, "r2": 1, "r3": 1, "r4": 1},
        "2+2w": {"x": 2, "y": 2},
        "rwc": {"r1": 1, "r2": 1, "r3": 1},
        "wrc": {"r1": 1, "r2": 1, "r3": 1},
        "dekker": {"r1": 1, "r2": 1},
        "peterson": {"r1": 1, "r2": 1},
    }
    return sc_outcomes.get(test_name, {})


def _is_sc_outcome(behavior: Behavior) -> bool:
    """Check if behavior matches the SC outcome for its test."""
    sc = _sc_outcome_for(behavior.test_name)
    return behavior.outcome == sc


_FENCE_GROUPS: Dict[FenceType, int] = {
    FenceType.FULL: 0,
    FenceType.SEQ_CST: 0,
    FenceType.DMB_ISH: 0,
    FenceType.FENCE_RW: 0,
    FenceType.MEMBAR_GL: 0,
    FenceType.ACQ_REL: 1,
    FenceType.ACQUIRE: 2,
    FenceType.RELEASE: 3,
    FenceType.STORE_STORE: 4,
    FenceType.DMB_ISHST: 4,
    FenceType.FENCE_WW: 4,
    FenceType.MEMBAR_CTA: 4,
    FenceType.LOAD_LOAD: 5,
    FenceType.DMB_ISHLD: 5,
    FenceType.FENCE_RR: 5,
    FenceType.STORE_LOAD: 6,
    FenceType.LOAD_STORE: 7,
}


def _fence_similarity(a: FenceType, b: FenceType) -> int:
    """Score similarity between two fence types (higher = more similar)."""
    ga = _FENCE_GROUPS.get(a, -1)
    gb = _FENCE_GROUPS.get(b, -2)
    if a == b:
        return 10
    if ga == gb:
        return 7
    if ga == 0 or gb == 0:
        return 3
    diff = abs(ga - gb)
    return max(0, 5 - diff)
