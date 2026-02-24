#!/usr/bin/env python3
"""
Custom Memory Model DSL for LITMUS∞.

Allows users to define new memory models via a simple text-based DSL,
inspired by herd7's .cat files but simplified for practical use.

Example DSL:
    model MyWeakModel {
        relaxes W->R    # store-to-load reordering allowed
        relaxes W->W    # store-to-store reordering allowed
        preserves deps  # dependencies preserved
        fence full { orders R->R, R->W, W->R, W->W }
        fence store { orders W->W, W->R }
    }
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from portcheck import (
    MemOp, LitmusTest, PATTERNS, ARCHITECTURES,
    verify_test, verify_test_generic, recommend_fence,
)


@dataclass
class FenceSpec:
    """A fence type in a custom model."""
    name: str
    orders: Set[Tuple[str, str]]  # set of (before_type, after_type) pairs
    cost: float = 1.0
    scope: Optional[str] = None  # None = CPU, 'workgroup', 'device'

@dataclass
class CustomModel:
    """A user-defined memory model."""
    name: str
    description: str = ""
    # Which orderings are relaxed (not preserved by hardware)
    relaxed_pairs: Set[Tuple[str, str]] = field(default_factory=set)
    # Whether dependencies are preserved
    preserves_deps: bool = True
    # Whether the model is multi-copy atomic
    multi_copy_atomic: bool = True
    # Available fence types
    fences: List[FenceSpec] = field(default_factory=list)
    # GPU scope level (None for CPU models)
    scope: Optional[str] = None
    # Parent model (inherits relaxed pairs)
    parent: Optional[str] = None

    def is_relaxed(self, before: str, after: str) -> bool:
        """Check if the ordering (before->after) is relaxed."""
        return (before, after) in self.relaxed_pairs

    def cheapest_fence(self, needed_pairs: Set[Tuple[str, str]]) -> Optional[FenceSpec]:
        """Find cheapest fence covering all needed pairs."""
        candidates = []
        for fence in self.fences:
            if needed_pairs <= fence.orders:
                candidates.append(fence)
        if not candidates:
            # Return most comprehensive fence
            for fence in sorted(self.fences, key=lambda f: -len(f.orders)):
                return fence
            return None
        return min(candidates, key=lambda f: f.cost)


# ── DSL Parser ─────────────────────────────────────────────────────

class ModelDSLParser:
    """Parse the custom model DSL into CustomModel objects."""

    PAIR_RE = re.compile(r'([RW])\s*->\s*([RW])', re.IGNORECASE)
    PAIR_MAP = {'R': 'load', 'W': 'store', 'r': 'load', 'w': 'store'}

    def parse(self, dsl_text: str) -> CustomModel:
        """Parse a DSL string into a CustomModel."""
        lines = dsl_text.strip().split('\n')
        model = CustomModel(name="unnamed")

        in_fence = False
        current_fence_name = None
        current_fence_orders = set()
        current_fence_cost = 1.0
        current_fence_scope = None

        for raw_line in lines:
            line = raw_line.split('#')[0].strip()
            if not line:
                continue

            # Model header
            m = re.match(r'model\s+(\w+)\s*(?:extends\s+(\w+))?\s*\{?', line, re.IGNORECASE)
            if m:
                model.name = m.group(1)
                if m.group(2):
                    model.parent = m.group(2)
                continue

            # Description
            m = re.match(r'description\s+"(.+)"', line, re.IGNORECASE)
            if m:
                model.description = m.group(1)
                continue

            # Relaxed ordering
            m = re.match(r'relaxes\s+(.+)', line, re.IGNORECASE)
            if m:
                pairs_str = m.group(1)
                for pair_m in self.PAIR_RE.finditer(pairs_str):
                    before = self.PAIR_MAP[pair_m.group(1)]
                    after = self.PAIR_MAP[pair_m.group(2)]
                    model.relaxed_pairs.add((before, after))
                continue

            # Preserves
            m = re.match(r'preserves\s+(.+)', line, re.IGNORECASE)
            if m:
                what = m.group(1).strip().lower()
                if 'deps' in what or 'dependencies' in what:
                    model.preserves_deps = True
                if 'mca' in what or 'multi-copy' in what:
                    model.multi_copy_atomic = True
                continue

            # Multi-copy atomicity
            m = re.match(r'not\s+multi[_-]?copy[_-]?atomic', line, re.IGNORECASE)
            if m:
                model.multi_copy_atomic = False
                continue

            # No dependency preservation
            m = re.match(r'no[_-]?deps|does\s+not\s+preserve\s+deps', line, re.IGNORECASE)
            if m:
                model.preserves_deps = False
                continue

            # Scope
            m = re.match(r'scope\s+(\w+)', line, re.IGNORECASE)
            if m:
                model.scope = m.group(1)
                continue

            # Fence definition start
            m = re.match(r'fence\s+(\w+)\s*(?:\(cost\s*=?\s*([\d.]+)\))?\s*\{?\s*(?:orders\s+([^}]+))?\s*\}?', line, re.IGNORECASE)
            if m:
                fname = m.group(1)
                fcost = float(m.group(2)) if m.group(2) else 1.0
                orders_str = m.group(3) or ""

                fence_orders = set()
                for pair_m in self.PAIR_RE.finditer(orders_str):
                    before = self.PAIR_MAP[pair_m.group(1)]
                    after = self.PAIR_MAP[pair_m.group(2)]
                    fence_orders.add((before, after))

                if '{' in line and '}' not in line:
                    in_fence = True
                    current_fence_name = fname
                    current_fence_cost = fcost
                    current_fence_orders = fence_orders
                else:
                    model.fences.append(FenceSpec(
                        name=fname, orders=fence_orders, cost=fcost
                    ))
                continue

            # Inside fence block
            if in_fence:
                if '}' in line:
                    model.fences.append(FenceSpec(
                        name=current_fence_name,
                        orders=current_fence_orders,
                        cost=current_fence_cost,
                    ))
                    in_fence = False
                    continue

                # orders line
                m_orders = re.match(r'orders\s+(.+)', line, re.IGNORECASE)
                if m_orders:
                    for pair_m in self.PAIR_RE.finditer(m_orders.group(1)):
                        before = self.PAIR_MAP[pair_m.group(1)]
                        after = self.PAIR_MAP[pair_m.group(2)]
                        current_fence_orders.add((before, after))

                # cost line
                m_cost = re.match(r'cost\s+([\d.]+)', line, re.IGNORECASE)
                if m_cost:
                    current_fence_cost = float(m_cost.group(1))

            # Closing brace for model
            if line == '}':
                continue

        return model

    def parse_file(self, filepath: str) -> CustomModel:
        """Parse a .model file."""
        with open(filepath) as f:
            return self.parse(f.read())


# ── Model registry ─────────────────────────────────────────────────

class ModelRegistry:
    """Registry for both built-in and custom memory models."""

    # Built-in models (matching portcheck.py ARCHITECTURES)
    BUILTIN = {
        'TSO': CustomModel(
            name='TSO', description='Total Store Order (x86)',
            relaxed_pairs={('store', 'load')},
            preserves_deps=True, multi_copy_atomic=True,
            fences=[FenceSpec('mfence', {('store','load'),('store','store'),('load','store'),('load','load')}, 4.0)],
        ),
        'PSO': CustomModel(
            name='PSO', description='Partial Store Order (SPARC)',
            relaxed_pairs={('store', 'load'), ('store', 'store')},
            preserves_deps=True, multi_copy_atomic=True,
            fences=[
                FenceSpec('membar_storestore', {('store','store')}, 1.0),
                FenceSpec('membar_full', {('store','load'),('store','store'),('load','store'),('load','load')}, 4.0),
            ],
        ),
        'ARM': CustomModel(
            name='ARM', description='ARMv8 (all reorderings, preserves deps)',
            relaxed_pairs={('store','load'),('store','store'),('load','load'),('load','store')},
            preserves_deps=True, multi_copy_atomic=False,
            fences=[
                FenceSpec('dmb_ishst', {('store','store')}, 1.0),
                FenceSpec('dmb_ishld', {('load','load'),('load','store')}, 1.0),
                FenceSpec('dmb_ish', {('store','load'),('store','store'),('load','store'),('load','load')}, 4.0),
            ],
        ),
        'RISC-V': CustomModel(
            name='RISC-V', description='RVWMO (asymmetric fences)',
            relaxed_pairs={('store','load'),('store','store'),('load','load'),('load','store')},
            preserves_deps=True, multi_copy_atomic=False,
            fences=[
                FenceSpec('fence_rr', {('load','load')}, 1.0),
                FenceSpec('fence_ww', {('store','store')}, 1.0),
                FenceSpec('fence_rw', {('load','store')}, 1.0),
                FenceSpec('fence_wr', {('store','load')}, 2.0),
                FenceSpec('fence_rwrw', {('store','load'),('store','store'),('load','store'),('load','load')}, 4.0),
            ],
        ),
    }

    def __init__(self):
        self._custom_models: Dict[str, CustomModel] = {}
        self._parser = ModelDSLParser()

    def register(self, model: CustomModel):
        """Register a custom model."""
        # Handle inheritance
        if model.parent:
            parent = self.get(model.parent)
            if parent:
                model.relaxed_pairs = parent.relaxed_pairs | model.relaxed_pairs
                if not model.fences:
                    model.fences = list(parent.fences)
                if not model.description:
                    model.description = f"extends {parent.name}"
        self._custom_models[model.name] = model

    def register_dsl(self, dsl_text: str) -> CustomModel:
        """Parse and register a model from DSL text."""
        model = self._parser.parse(dsl_text)
        self.register(model)
        return model

    def register_file(self, filepath: str) -> CustomModel:
        """Parse and register a model from a .model file."""
        model = self._parser.parse_file(filepath)
        self.register(model)
        return model

    def get(self, name: str) -> Optional[CustomModel]:
        """Get a model by name (checks custom first, then built-in)."""
        if name in self._custom_models:
            return self._custom_models[name]
        if name in self.BUILTIN:
            return self.BUILTIN[name]
        return None

    def list_models(self) -> List[str]:
        """List all available model names."""
        return sorted(set(list(self.BUILTIN.keys()) + list(self._custom_models.keys())))

    def check_pattern_custom(self, pattern_name: str, model: CustomModel) -> Dict:
        """Check a litmus pattern against a custom model using full model checking."""
        if pattern_name not in PATTERNS:
            raise ValueError(f"Unknown pattern: {pattern_name}")

        pat = PATTERNS[pattern_name]
        ops = pat['ops']
        n_threads = max(op.thread for op in ops) + 1

        lt = LitmusTest(
            name=pattern_name, n_threads=n_threads,
            addresses=pat['addresses'], ops=ops,
            forbidden=pat['forbidden'],
        )

        # Convert CustomModel to model_info dict for verify_test_generic
        model_info = {
            'relaxed_pairs': model.relaxed_pairs,
            'preserves_deps': model.preserves_deps,
            'multi_copy_atomic': model.multi_copy_atomic,
            'fences': [{'name': f.name, 'orders': f.orders, 'cost': f.cost}
                       for f in model.fences],
        }

        forbidden_allowed, _ = verify_test_generic(lt, model_info)
        safe = not forbidden_allowed

        # Get fence recommendation if unsafe
        fence_rec = None
        if not safe:
            # Determine which pairs are violated
            non_fence = [op for op in ops if op.optype != 'fence']
            violated_pairs = set()
            for t in range(n_threads):
                t_ops = [op for op in non_fence if op.thread == t]
                for i in range(len(t_ops)):
                    for j in range(i + 1, len(t_ops)):
                        before, after = t_ops[i], t_ops[j]
                        if before.addr != after.addr:
                            pair_type = (before.optype, after.optype)
                            if model.is_relaxed(before.optype, after.optype):
                                violated_pairs.add(pair_type)
            if violated_pairs and model.fences:
                best = model.cheapest_fence(violated_pairs)
                if best:
                    fence_rec = best.name

        return {
            'pattern': pattern_name,
            'model': model.name,
            'safe': safe,
            'violated_pairs': list(violated_pairs) if not safe else [],
            'fence_recommendation': fence_rec,
        }

    def compare_models(self, model_a: str, model_b: str) -> List[Dict]:
        """Compare two models across all patterns."""
        ma = self.get(model_a)
        mb = self.get(model_b)
        if not ma or not mb:
            raise ValueError(f"Model not found: {model_a if not ma else model_b}")

        results = []
        for pat_name in sorted(PATTERNS.keys()):
            ra = self.check_pattern_custom(pat_name, ma)
            rb = self.check_pattern_custom(pat_name, mb)
            if ra['safe'] != rb['safe']:
                results.append({
                    'pattern': pat_name,
                    'model_a': model_a,
                    'safe_a': ra['safe'],
                    'model_b': model_b,
                    'safe_b': rb['safe'],
                    'discriminator': True,
                })
        return results


# ── Module-level convenience ────────────────────────────────────────

_registry = None

def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry

def register_model(dsl_text: str) -> CustomModel:
    """Register a custom model from DSL text."""
    return get_registry().register_dsl(dsl_text)

def register_model_file(filepath: str) -> CustomModel:
    """Register a custom model from a .model file."""
    return get_registry().register_file(filepath)

def check_custom(pattern: str, model_name: str) -> Dict:
    """Check a pattern against a registered model."""
    reg = get_registry()
    model = reg.get(model_name)
    if not model:
        raise ValueError(f"Unknown model: {model_name}")
    return reg.check_pattern_custom(pattern, model)

def list_models() -> List[str]:
    """List all available models."""
    return get_registry().list_models()


# ── Built-in model definitions (DSL format) ────────────────────────

EXAMPLE_MODELS = {
    "c11_relaxed": """
model C11_Relaxed {
    description "C11 memory model with relaxed atomics only"
    relaxes W->R, W->W, R->R, R->W
    preserves deps
    not multi-copy-atomic
    fence seq_cst (cost=8) { orders W->R, W->W, R->R, R->W }
    fence release (cost=4) { orders W->W, W->R }
    fence acquire (cost=4) { orders R->R, R->W }
    fence acq_rel (cost=6) { orders W->W, W->R, R->R, R->W }
}
""",
    "power": """
model POWER {
    description "IBM POWER memory model"
    relaxes W->R, W->W, R->R, R->W
    preserves deps
    not multi-copy-atomic
    fence hwsync (cost=8) { orders W->R, W->W, R->R, R->W }
    fence lwsync (cost=4) { orders W->W, R->R, R->W }
    fence isync (cost=2) { orders R->R }
}
""",
    "alpha": """
model Alpha {
    description "DEC Alpha (weakest practical CPU model)"
    relaxes W->R, W->W, R->R, R->W
    no-deps
    not multi-copy-atomic
    fence mb (cost=8) { orders W->R, W->W, R->R, R->W }
    fence wmb (cost=4) { orders W->W }
    fence rmb (cost=4) { orders R->R }
}
""",
    "sc": """
model SC {
    description "Sequential Consistency"
    fence full (cost=0) { orders W->R, W->W, R->R, R->W }
}
""",
}


if __name__ == '__main__':
    import json

    print("=" * 60)
    print("LITMUS∞ Custom Model DSL — Demo")
    print("=" * 60)

    reg = get_registry()

    # Register example models
    for name, dsl in EXAMPLE_MODELS.items():
        m = reg.register_dsl(dsl)
        print(f"Registered: {m.name} - {m.description}")
        print(f"  Relaxed: {m.relaxed_pairs}")
        print(f"  Fences: {[f.name for f in m.fences]}")
        print()

    # Compare POWER vs ARM
    print("POWER vs ARM discriminators:")
    diffs = reg.compare_models('POWER', 'ARM')
    for d in diffs:
        print(f"  {d['pattern']}: POWER={'safe' if d['safe_a'] else 'unsafe'}, "
              f"ARM={'safe' if d['safe_b'] else 'unsafe'}")

    print(f"\nAll models: {reg.list_models()}")
