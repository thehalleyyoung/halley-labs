#!/usr/bin/env python3
"""
AST-based concurrent code analyzer for LITMUS∞.

Uses tree-sitter for proper AST parsing of C/C++/CUDA code,
replacing the regex-based CodeParser with structural analysis.
Achieves significantly higher accuracy by understanding:
  - Function/thread boundaries via AST scope
  - Atomic operation ordering semantics 
  - Data/address/control dependencies via SSA-like analysis
  - GPU scope annotations (CUDA __threadfence variants, OpenCL barriers)
  - Memory ordering parameters (relaxed, acquire, release, seq_cst)
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from enum import Enum

try:
    import tree_sitter_c as tsc
    import tree_sitter_cpp as tscpp
    from tree_sitter import Language, Parser
    HAS_TREESITTER = True
except ImportError:
    HAS_TREESITTER = False

from portcheck import (
    MemOp, LitmusTest, PATTERNS, ARCHITECTURES,
    verify_test, recommend_fence, _identify_per_thread_violations,
    _is_scope_mismatch_pattern, _is_gpu_model,
)


# ── Enums and types ─────────────────────────────────────────────────

class MemoryOrder(Enum):
    RELAXED = "relaxed"
    CONSUME = "consume"
    ACQUIRE = "acquire"
    RELEASE = "release"
    ACQ_REL = "acq_rel"
    SEQ_CST = "seq_cst"

class OpType(Enum):
    STORE = "store"
    LOAD = "load"
    FENCE = "fence"
    RMW = "rmw"
    CAS = "cas"

class DepType(Enum):
    ADDR = "addr"
    DATA = "data"
    CTRL = "ctrl"
    NONE = None

class GPUScope(Enum):
    WORKGROUP = "workgroup"
    DEVICE = "device"
    SYSTEM = "system"
    CTA = "cta"
    GPU = "gpu"
    NONE = None


@dataclass
class ExtractedOp:
    """A memory operation extracted from source code via AST analysis."""
    thread: int
    op_type: OpType
    variable: str
    value: Optional[str] = None
    register: Optional[str] = None
    line: int = 0
    column: int = 0
    memory_order: MemoryOrder = MemoryOrder.SEQ_CST
    gpu_scope: GPUScope = GPUScope.NONE
    dep_type: DepType = DepType.NONE
    dep_source: Optional[str] = None
    workgroup: int = 0
    fence_pred: Optional[str] = None
    fence_succ: Optional[str] = None
    in_branch: bool = False
    branch_var: Optional[str] = None
    raw_text: str = ""

    def to_memop(self) -> MemOp:
        scope_str = None
        if self.gpu_scope not in (GPUScope.NONE, None):
            scope_str = self.gpu_scope.value
        dep = None
        if self.dep_type != DepType.NONE:
            dep = self.dep_type.value
        return MemOp(
            thread=self.thread,
            optype=self.op_type.value if self.op_type != OpType.RMW else 'store',
            addr=self.variable,
            value=int(self.value) if self.value and self.value.isdigit() else 1,
            reg=self.register,
            scope=scope_str,
            workgroup=self.workgroup,
            fence_pred=self.fence_pred,
            fence_succ=self.fence_succ,
            dep_on=dep,
        )


@dataclass
class ASTPatternMatch:
    """A matched litmus test pattern with AST-level confidence."""
    pattern_name: str
    confidence: float
    match_type: str  # 'exact', 'structural', 'heuristic'
    extracted_ops: List[ExtractedOp]
    dependency_info: Dict[str, str] = field(default_factory=dict)
    ordering_info: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def __repr__(self):
        return f"ASTPatternMatch({self.pattern_name}, conf={self.confidence:.2f}, type={self.match_type})"


@dataclass
class ASTAnalysisResult:
    """Full result of AST-based code analysis."""
    code_hash: str
    language: str
    patterns_found: List[ASTPatternMatch]
    extracted_ops: List[ExtractedOp]
    n_threads: int
    shared_vars: Set[str]
    has_fences: bool
    is_gpu: bool
    gpu_scope: Optional[str] = None
    memory_orders_used: Set[str] = field(default_factory=set)
    dependencies_found: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    parse_method: str = "ast"  # 'ast' or 'fallback_regex'
    coverage_confidence: float = 1.0  # ratio of recognized to total concurrent ops
    unrecognized_ops: List[Dict] = field(default_factory=list)

    def __repr__(self):
        pats = ", ".join(m.pattern_name for m in self.patterns_found[:3])
        return f"ASTAnalysisResult({self.n_threads}T, {self.parse_method}, [{pats}])"


# ── Tree-sitter AST parser ─────────────────────────────────────────

class TreeSitterExtractor:
    """Extract memory operations from C/C++ AST using tree-sitter."""

    def __init__(self):
        if not HAS_TREESITTER:
            raise RuntimeError("tree-sitter not installed")
        self.c_lang = Language(tsc.language())
        self.cpp_lang = Language(tscpp.language())
        self.c_parser = Parser(self.c_lang)
        self.cpp_parser = Parser(self.cpp_lang)

    def parse(self, code: str, language: str = "c") -> Tuple[List[ExtractedOp], Dict]:
        parser = self.cpp_parser if language in ("cpp", "cuda") else self.c_parser
        tree = parser.parse(bytes(code, "utf8"))
        root = tree.root_node

        ops = []
        metadata = {
            'n_threads': 0,
            'shared_vars': set(),
            'fences': [],
            'is_gpu': False,
            'gpu_scope': None,
            'memory_orders': set(),
            'language': language,
        }

        # Detect GPU code
        code_lower = code.lower()
        if any(kw in code_lower for kw in ['__global__', '__device__', '__shared__',
                                            '__threadfence', '__syncthreads',
                                            'clk_global_mem_fence', 'clk_local_mem_fence',
                                            'barrier(', 'work_group_barrier']):
            metadata['is_gpu'] = True

        # Extract thread structure from comments and function boundaries
        thread_regions = self._detect_threads(code, root)
        metadata['n_threads'] = len(thread_regions) if thread_regions else 1

        # Walk AST for memory operations
        self._walk_node(root, code, ops, metadata, thread_regions)

        # Infer dependencies
        self._infer_dependencies(ops)

        # GPU workgroup detection from comments (tree-sitter doesn't capture this)
        if metadata['is_gpu']:
            wg_re = re.compile(r'(?:workgroup|block|wg)\s+(\d+)', re.IGNORECASE)
            cur_t, cur_wg = 0, 0
            twg_map = {}
            thread_re = re.compile(r'(?://|/\*)\s*[Tt]hread\s+(\d+)', re.IGNORECASE)
            for raw_line in code.split('\n'):
                stripped = raw_line.strip()
                if stripped.startswith('//') or stripped.startswith('/*'):
                    m = thread_re.search(raw_line)
                    if m:
                        cur_t = int(m.group(1))
                    m_wg = wg_re.search(raw_line)
                    if m_wg:
                        cur_wg = int(m_wg.group(1))
                    twg_map[cur_t] = cur_wg
            for op in ops:
                if op.thread in twg_map:
                    op.workgroup = twg_map[op.thread]

        # Deduplicate shared vars
        var_threads = defaultdict(set)
        for op in ops:
            if op.op_type in (OpType.STORE, OpType.LOAD, OpType.RMW, OpType.CAS):
                var_threads[op.variable].add(op.thread)
        metadata['shared_vars'] = {v for v, ts in var_threads.items() if len(ts) > 1 or len(ops) > 1}

        return ops, metadata

    def _detect_threads(self, code: str, root) -> Dict[int, Tuple[int, int]]:
        """Detect thread boundaries from comments, function names, pragmas."""
        regions = {}
        lines = code.split('\n')

        # Method 1: Comments like "// Thread 0", "/* Thread 1 */"
        thread_comment_re = re.compile(r'(?://|/\*)\s*[Tt]hread\s+(\d+)', re.IGNORECASE)
        current_thread = -1
        for i, line in enumerate(lines):
            m = thread_comment_re.search(line)
            if m:
                tid = int(m.group(1))
                if current_thread >= 0:
                    regions[current_thread] = (regions[current_thread][0], i - 1)
                regions[tid] = (i, len(lines) - 1)
                current_thread = tid

        if regions:
            return regions

        # Method 2: Function names thread_0, thread_1, etc.
        func_re = re.compile(r'(?:void\s+)?thread_?(\d+)\s*\(', re.IGNORECASE)
        for i, line in enumerate(lines):
            m = func_re.search(line)
            if m:
                tid = int(m.group(1))
                # Find function body
                brace_depth = 0
                start = i
                end = len(lines) - 1
                for j in range(i, len(lines)):
                    brace_depth += lines[j].count('{') - lines[j].count('}')
                    if brace_depth <= 0 and j > i:
                        end = j
                        break
                regions[tid] = (start, end)

        if regions:
            return regions

        # Method 3: "Thread N:" pragmas
        pragma_re = re.compile(r'Thread\s+(\d+)\s*:', re.IGNORECASE)
        for i, line in enumerate(lines):
            m = pragma_re.search(line)
            if m:
                tid = int(m.group(1))
                regions[tid] = (i, len(lines) - 1)
                if current_thread >= 0:
                    regions[current_thread] = (regions[current_thread][0], i - 1)
                current_thread = tid

        if regions:
            return regions

        # Method 4: column-based "// Thr 0    // Thr 1" side-by-side
        col_re = re.compile(r'//\s*(?:Thread|Thr)\s+(\d+)', re.IGNORECASE)
        first_line_threads = col_re.findall(lines[0] if lines else "")
        if len(first_line_threads) >= 2:
            for tid_s in first_line_threads:
                regions[int(tid_s)] = (0, len(lines) - 1)
            return regions

        # Method 5: "||" delimiter (pseudocode)
        for i, line in enumerate(lines):
            if '||' in line:
                parts = line.split('||')
                for j in range(len(parts)):
                    if j not in regions:
                        regions[j] = (0, len(lines) - 1)
                if regions:
                    return regions

        # Fallback: GPU kernel = single "thread" context
        if any('__global__' in line for line in lines):
            # Look for blockIdx differentiation
            block_re = re.compile(r'blockIdx\.\w+\s*==\s*(\d+)')
            for i, line in enumerate(lines):
                m = block_re.search(line)
                if m:
                    tid = int(m.group(1))
                    regions[tid] = (i, len(lines) - 1)
            if regions:
                return regions

        # If nothing detected, try to identify 2 logical threads from operation structure
        return {}

    def _walk_node(self, node, code: str, ops: List[ExtractedOp],
                   metadata: Dict, thread_regions: Dict[int, Tuple[int, int]],
                   in_branch: bool = False, branch_var: Optional[str] = None):
        """Walk the AST and extract memory operations."""
        text = code[node.start_byte:node.end_byte]
        line = node.start_point[0]
        thread = self._line_to_thread(line, thread_regions)

        # Detect operations from AST node types
        if node.type == 'call_expression':
            self._handle_call(node, code, ops, metadata, thread, line, in_branch, branch_var)
        elif node.type == 'assignment_expression':
            self._handle_assignment(node, code, ops, metadata, thread, line, in_branch, branch_var)
        elif node.type == 'declaration' and '=' in text:
            self._handle_declaration(node, code, ops, metadata, thread, line, in_branch, branch_var)
        elif node.type == 'expression_statement':
            stmt_text = text.strip().rstrip(';')
            # Only handle simple stores if no child assignment_expression/call_expression
            # (those will be handled by recursion into _handle_assignment/_handle_call)
            has_child_handler = any(
                child.type in ('assignment_expression', 'call_expression')
                for child in node.children
            )
            if not has_child_handler:
                assign_re = re.compile(r'^(\*?\s*\w+)\s*=\s*(.+)$')
                m = assign_re.match(stmt_text)
                if m:
                    var = m.group(1).strip().lstrip('*').strip()
                    val = m.group(2).strip()
                    # Skip loop vars, indices, and register targets of atomic loads
                    _is_atomic_rhs = any(k in val for k in (
                        '.load(', '.store(', '.exchange(', '.fetch_',
                        'atomic_load', 'atomic_store', '__atomic_load',
                        '__atomic_store', 'compare_exchange',
                        'smp_load_acquire', 'smp_store_release',
                        'READ_ONCE', 'WRITE_ONCE',
                        'atomic_load_acquire', 'atomic_store_release',
                    ))
                    _is_register = bool(re.match(r'^r\d+$', var))
                    if (var not in ('i', 'j', 'k', 'n', 'tid', 'idx', 'size', 'len')
                            and not _is_atomic_rhs and not _is_register):
                        ops.append(ExtractedOp(
                            thread=thread, op_type=OpType.STORE, variable=var,
                            value=val, line=line, raw_text=text,
                            in_branch=in_branch, branch_var=branch_var,
                        ))
                        metadata['shared_vars'].add(var)

        # Check for if/while branches (for control dependencies)
        new_in_branch = in_branch
        new_branch_var = branch_var
        if node.type in ('if_statement', 'while_statement'):
            cond = node.child_by_field_name('condition')
            if cond:
                cond_text = code[cond.start_byte:cond.end_byte]
                # Extract variable from condition
                var_re = re.compile(r'\b(r\d+|[a-zA-Z_]\w*)\b')
                cond_vars = var_re.findall(cond_text)
                cond_vars = [v for v in cond_vars if v not in ('if', 'while', 'int', 'auto', 'void')]
                if cond_vars:
                    new_in_branch = True
                    new_branch_var = cond_vars[0]

        # Recurse
        for child in node.children:
            self._walk_node(child, code, ops, metadata, thread_regions,
                          new_in_branch, new_branch_var)

    def _handle_call(self, node, code: str, ops: List[ExtractedOp],
                     metadata: Dict, thread: int, line: int,
                     in_branch: bool, branch_var: Optional[str]):
        """Handle function call nodes for atomic ops, fences, etc."""
        text = code[node.start_byte:node.end_byte]

        # Extract register from parent assignment context (e.g. r0 = x.load(...))
        _parent_reg = None
        if node.parent and node.parent.type in ('assignment_expression', 'init_declarator'):
            parent_text = code[node.parent.start_byte:node.parent.end_byte].strip()
            _reg_m = re.match(r'^(r\d+)\s*=', parent_text)
            if _reg_m:
                _parent_reg = _reg_m.group(1)

        # std::atomic store: x.store(val, order)
        store_re = re.compile(r'(\w+)\.store\s*\(\s*(.+?)(?:\s*,\s*(?:std::)?memory_order_(\w+))?\s*\)')
        m = store_re.search(text)
        if m:
            var, val, order = m.group(1), m.group(2), m.group(3) or 'seq_cst'
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.STORE, variable=var, value=val,
                line=line, memory_order=self._parse_order(order),
                raw_text=text, in_branch=in_branch, branch_var=branch_var,
            ))
            metadata['shared_vars'].add(var)
            metadata['memory_orders'].add(order)
            return

        # std::atomic load: x.load(order)
        load_re = re.compile(r'(\w+)\.load\s*\(\s*(?:(?:std::)?memory_order_(\w+))?\s*\)')
        m = load_re.search(text)
        if m:
            var, order = m.group(1), m.group(2) or 'seq_cst'
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.LOAD, variable=var,
                register=_parent_reg,
                line=line, memory_order=self._parse_order(order),
                raw_text=text, in_branch=in_branch, branch_var=branch_var,
            ))
            metadata['shared_vars'].add(var)
            metadata['memory_orders'].add(order)
            return

        # atomic_store_explicit / atomic_load_explicit
        atomic_store_re = re.compile(r'atomic_store(?:_explicit)?\s*\(\s*&?\s*(\w+)\s*,\s*(.+?)(?:\s*,\s*(?:memory_order_)?(\w+))?\s*\)')
        m = atomic_store_re.search(text)
        if m:
            var, val = m.group(1), m.group(2)
            order = m.group(3) or 'seq_cst'
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.STORE, variable=var, value=val,
                line=line, memory_order=self._parse_order(order), raw_text=text,
            ))
            metadata['shared_vars'].add(var)
            return

        atomic_load_re = re.compile(r'atomic_load(?:_explicit)?\s*\(\s*&?\s*(\w+)(?:\s*,\s*(?:memory_order_)?(\w+))?\s*\)')
        m = atomic_load_re.search(text)
        if m:
            var = m.group(1)
            order = m.group(2) or 'seq_cst'
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.LOAD, variable=var,
                register=_parent_reg,
                line=line, memory_order=self._parse_order(order), raw_text=text,
            ))
            metadata['shared_vars'].add(var)
            return

        # __atomic_store_n / __atomic_load_n (GCC builtins)
        gcc_store_re = re.compile(r'__atomic_store_n\s*\(\s*&?\s*(\w+)\s*,\s*(.+?)\s*,\s*__ATOMIC_(\w+)\s*\)')
        m = gcc_store_re.search(text)
        if m:
            var, val, order = m.group(1), m.group(2), m.group(3).lower()
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.STORE, variable=var, value=val,
                line=line, memory_order=self._parse_order(order), raw_text=text,
            ))
            metadata['shared_vars'].add(var)
            return

        gcc_load_re = re.compile(r'__atomic_load_n\s*\(\s*&?\s*(\w+)\s*,\s*__ATOMIC_(\w+)\s*\)')
        m = gcc_load_re.search(text)
        if m:
            var, order = m.group(1), m.group(2).lower()
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.LOAD, variable=var,
                register=_parent_reg,
                line=line, memory_order=self._parse_order(order), raw_text=text,
            ))
            metadata['shared_vars'].add(var)
            return

        # Linux kernel macros: smp_store_release, smp_load_acquire
        smp_store_re = re.compile(r'smp_store_release\s*\(\s*&?\s*(\w+)\s*,\s*(.+?)\s*\)')
        m = smp_store_re.search(text)
        if m:
            var, val = m.group(1), m.group(2)
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.STORE, variable=var, value=val,
                line=line, memory_order=MemoryOrder.RELEASE, raw_text=text,
            ))
            metadata['shared_vars'].add(var)
            return

        smp_load_re = re.compile(r'smp_load_acquire\s*\(\s*&?\s*(\w+)\s*\)')
        m = smp_load_re.search(text)
        if m:
            var = m.group(1)
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.LOAD, variable=var,
                register=_parent_reg,
                line=line, memory_order=MemoryOrder.ACQUIRE, raw_text=text,
            ))
            metadata['shared_vars'].add(var)
            return

        # READ_ONCE / WRITE_ONCE (Linux kernel)
        write_once_re = re.compile(r'WRITE_ONCE\s*\(\s*(\w+)\s*,\s*(.+?)\s*\)')
        m = write_once_re.search(text)
        if m:
            var, val = m.group(1), m.group(2)
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.STORE, variable=var, value=val,
                line=line, raw_text=text,
            ))
            metadata['shared_vars'].add(var)
            return

        read_once_re = re.compile(r'READ_ONCE\s*\(\s*(\w+)\s*\)')
        m = read_once_re.search(text)
        if m:
            var = m.group(1)
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.LOAD, variable=var,
                register=_parent_reg,
                line=line, raw_text=text,
            ))
            metadata['shared_vars'].add(var)
            return

        # Linux kernel fences: smp_mb, smp_wmb, smp_rmb
        if re.search(r'smp_(?:[wr]?)mb\s*\(\s*\)', text):
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.FENCE, variable='fence',
                line=line, memory_order=MemoryOrder.SEQ_CST, raw_text=text,
            ))
            metadata['fences'].append(('smp_mb', 'seq_cst'))
            return

        # atomic_store_release / atomic_load_acquire (non-standard but common)
        asr_re = re.compile(r'atomic_store_release\s*\(\s*&?\s*(\w+)\s*,\s*(.+?)\s*\)')
        m = asr_re.search(text)
        if m:
            var, val = m.group(1), m.group(2)
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.STORE, variable=var, value=val,
                line=line, memory_order=MemoryOrder.RELEASE, raw_text=text,
            ))
            metadata['shared_vars'].add(var)
            return

        ala_re = re.compile(r'atomic_load_acquire\s*\(\s*&?\s*(\w+)\s*\)')
        m = ala_re.search(text)
        if m:
            var = m.group(1)
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.LOAD, variable=var,
                register=_parent_reg,
                line=line, memory_order=MemoryOrder.ACQUIRE, raw_text=text,
            ))
            metadata['shared_vars'].add(var)
            return

        # Compare-and-swap
        cas_re = re.compile(r'(?:atomic_compare_exchange|__atomic_compare_exchange|compare_exchange_(?:strong|weak))\s*\(\s*&?\s*(\w+)')
        m = cas_re.search(text)
        if m:
            var = m.group(1)
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.CAS, variable=var,
                line=line, raw_text=text,
            ))
            metadata['shared_vars'].add(var)
            return

        # Fetch-add, exchange, etc.
        rmw_re = re.compile(r'(?:atomic_fetch_\w+|__atomic_\w+|fetch_add|fetch_sub|exchange)\s*\(\s*&?\s*(\w+)')
        m = rmw_re.search(text)
        if m:
            var = m.group(1)
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.RMW, variable=var,
                line=line, raw_text=text,
            ))
            metadata['shared_vars'].add(var)
            return

        # Fences
        if 'atomic_thread_fence' in text:
            order_re = re.compile(r'memory_order_(\w+)')
            m = order_re.search(text)
            order = m.group(1) if m else 'seq_cst'
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.FENCE, variable='fence',
                line=line, memory_order=self._parse_order(order), raw_text=text,
            ))
            metadata['fences'].append(('thread_fence', order))
            return

        if '__sync_synchronize' in text or '_mm_mfence' in text:
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.FENCE, variable='fence',
                line=line, memory_order=MemoryOrder.SEQ_CST, raw_text=text,
            ))
            metadata['fences'].append(('sync', 'seq_cst'))
            return

        # CUDA/GPU fences
        if '__threadfence_system' in text:
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.FENCE, variable='fence',
                line=line, gpu_scope=GPUScope.SYSTEM, raw_text=text,
            ))
            metadata['fences'].append(('threadfence_system', 'system'))
            metadata['is_gpu'] = True
            return

        if '__threadfence_block' in text:
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.FENCE, variable='fence',
                line=line, gpu_scope=GPUScope.CTA, raw_text=text,
            ))
            metadata['fences'].append(('threadfence_block', 'cta'))
            metadata['is_gpu'] = True
            metadata['gpu_scope'] = 'workgroup'
            return

        if '__threadfence()' in text or re.search(r'__threadfence\s*\(\s*\)', text):
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.FENCE, variable='fence',
                line=line, gpu_scope=GPUScope.DEVICE, raw_text=text,
            ))
            metadata['fences'].append(('threadfence', 'device'))
            metadata['is_gpu'] = True
            return

        if '__syncthreads' in text:
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.FENCE, variable='fence',
                line=line, gpu_scope=GPUScope.CTA, raw_text=text,
            ))
            metadata['fences'].append(('syncthreads', 'cta'))
            metadata['is_gpu'] = True
            return

        if '__syncwarp' in text:
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.FENCE, variable='fence',
                line=line, gpu_scope=GPUScope.WORKGROUP, raw_text=text,
            ))
            metadata['fences'].append(('syncwarp', 'warp'))
            metadata['is_gpu'] = True
            return

        # CUDA cooperative groups sync
        cg_sync_re = re.compile(r'(?:cooperative_groups|cg)::.*(?:sync|barrier)\s*\(')
        if cg_sync_re.search(text):
            scope = GPUScope.DEVICE if 'grid' in text.lower() else GPUScope.CTA
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.FENCE, variable='fence',
                line=line, gpu_scope=scope, raw_text=text,
            ))
            metadata['fences'].append(('cg_sync', scope.value))
            metadata['is_gpu'] = True
            return

        # cuda::atomic_thread_fence
        if 'cuda::atomic_thread_fence' in text or 'cuda::std::atomic_thread_fence' in text:
            scope = GPUScope.DEVICE
            if 'thread_scope_block' in text or 'thread_scope_cta' in text:
                scope = GPUScope.CTA
            elif 'thread_scope_system' in text:
                scope = GPUScope.SYSTEM
            elif 'thread_scope_device' in text:
                scope = GPUScope.DEVICE
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.FENCE, variable='fence',
                line=line, gpu_scope=scope, raw_text=text,
            ))
            metadata['fences'].append(('cuda_atomic_fence', scope.value))
            metadata['is_gpu'] = True
            return

        # CUDA atomicCAS, atomicExch, atomicAdd, etc.
        cuda_atomic_re = re.compile(
            r'(atomicCAS|atomicExch|atomicAdd|atomicSub|atomicMin|atomicMax|'
            r'atomicInc|atomicDec|atomicAnd|atomicOr|atomicXor)\s*\(\s*&?\s*(\w+)')
        m = cuda_atomic_re.search(text)
        if m:
            op_name, var = m.group(1), m.group(2)
            op_type = OpType.CAS if op_name == 'atomicCAS' else OpType.RMW
            ops.append(ExtractedOp(
                thread=thread, op_type=op_type, variable=var,
                line=line, gpu_scope=GPUScope.DEVICE, raw_text=text,
            ))
            metadata['shared_vars'].add(var)
            metadata['is_gpu'] = True
            return

        # cuda::atomic operations
        cuda_std_atomic_re = re.compile(r'cuda::(?:std::)?atomic<[^>]+>::(store|load|exchange|fetch_\w+)\s*\(')
        m = cuda_std_atomic_re.search(text)
        if m:
            op_kind = m.group(1)
            scope = GPUScope.DEVICE
            if 'thread_scope_block' in text:
                scope = GPUScope.CTA
            elif 'thread_scope_system' in text:
                scope = GPUScope.SYSTEM
            var_re = re.compile(r'(\w+)\.(store|load|exchange|fetch_)')
            var_m = var_re.search(text)
            var = var_m.group(1) if var_m else 'unknown'
            if op_kind == 'store':
                ops.append(ExtractedOp(
                    thread=thread, op_type=OpType.STORE, variable=var,
                    line=line, gpu_scope=scope, raw_text=text,
                ))
            elif op_kind == 'load':
                ops.append(ExtractedOp(
                    thread=thread, op_type=OpType.LOAD, variable=var,
                    register=_parent_reg, line=line, gpu_scope=scope, raw_text=text,
                ))
            else:
                ops.append(ExtractedOp(
                    thread=thread, op_type=OpType.RMW, variable=var,
                    line=line, gpu_scope=scope, raw_text=text,
                ))
            metadata['shared_vars'].add(var)
            metadata['is_gpu'] = True
            return

        # OpenCL barriers
        if 'barrier' in text.lower() and ('CLK_' in text or 'work_group' in text.lower()):
            scope = GPUScope.WORKGROUP
            if 'CLK_GLOBAL_MEM_FENCE' in text:
                scope = GPUScope.DEVICE
            ops.append(ExtractedOp(
                thread=thread, op_type=OpType.FENCE, variable='fence',
                line=line, gpu_scope=scope, raw_text=text,
            ))
            metadata['fences'].append(('barrier', scope.value))
            metadata['is_gpu'] = True
            metadata['gpu_scope'] = scope.value
            return

    def _handle_assignment(self, node, code: str, ops: List[ExtractedOp],
                           metadata: Dict, thread: int, line: int,
                           in_branch: bool, branch_var: Optional[str]):
        """Handle assignment expressions for simple stores/loads."""
        text = code[node.start_byte:node.end_byte].strip()

        # Skip assignments whose RHS is an atomic call (handled by _handle_call)
        _atomic_rhs_kws = ('.load(', '.store(', '.exchange(', '.fetch_',
                           'atomic_load', 'atomic_store', '__atomic_load',
                           '__atomic_store', 'compare_exchange',
                           'smp_load_acquire', 'smp_store_release',
                           'READ_ONCE', 'WRITE_ONCE',
                           'atomic_load_acquire', 'atomic_store_release')
        if any(k in text for k in _atomic_rhs_kws):
            return

        # r0 = var (load) 
        load_re = re.compile(r'^(r\d+)\s*=\s*(\w+)\s*$')
        m = load_re.match(text)
        if m:
            reg, var = m.group(1), m.group(2)
            if var not in ('0', '1', '2', 'true', 'false'):
                ops.append(ExtractedOp(
                    thread=thread, op_type=OpType.LOAD, variable=var,
                    register=reg, line=line, raw_text=text,
                    in_branch=in_branch, branch_var=branch_var,
                ))
                metadata['shared_vars'].add(var)
                return

        # var = val (store)
        store_re = re.compile(r'^(\*?\s*\w+)\s*=\s*(.+)$')
        m = store_re.match(text)
        if m:
            var = m.group(1).strip().lstrip('*').strip()
            val = m.group(2).strip()
            if var not in ('i', 'j', 'k', 'n', 'tid', 'idx') and not var.startswith('r'):
                ops.append(ExtractedOp(
                    thread=thread, op_type=OpType.STORE, variable=var,
                    value=val, line=line, raw_text=text,
                    in_branch=in_branch, branch_var=branch_var,
                ))
                metadata['shared_vars'].add(var)

    def _handle_declaration(self, node, code: str, ops: List[ExtractedOp],
                            metadata: Dict, thread: int, line: int,
                            in_branch: bool, branch_var: Optional[str]):
        """Handle declarations with initialization (loads)."""
        text = code[node.start_byte:node.end_byte].strip()
        # int r0 = var; or auto r0 = var;
        decl_re = re.compile(r'(?:int|long|auto|unsigned|volatile\s+\w+)\s+(\w+)\s*=\s*(\w+)')
        m = decl_re.search(text)
        if m:
            reg, var = m.group(1), m.group(2)
            if var not in ('0', '1', '2', 'true', 'false', 'NULL', 'nullptr'):
                ops.append(ExtractedOp(
                    thread=thread, op_type=OpType.LOAD, variable=var,
                    register=reg, line=line, raw_text=text,
                    in_branch=in_branch, branch_var=branch_var,
                ))
                metadata['shared_vars'].add(var)

    def _infer_dependencies(self, ops: List[ExtractedOp]):
        """Infer data, address, and control dependencies between operations."""
        # Group by thread
        thread_ops = defaultdict(list)
        for op in ops:
            thread_ops[op.thread].append(op)

        for tid, t_ops in thread_ops.items():
            # Track which variables each register holds
            reg_source = {}  # reg -> source variable

            for i, op in enumerate(t_ops):
                if op.op_type == OpType.LOAD and op.register:
                    reg_source[op.register] = op.variable

                elif op.op_type == OpType.STORE:
                    # Data dependency: store value depends on prior load
                    if op.value and op.value in reg_source:
                        op.dep_type = DepType.DATA
                        op.dep_source = reg_source[op.value]

                    # Address dependency: store address depends on prior load
                    if op.variable in reg_source:
                        op.dep_type = DepType.ADDR
                        op.dep_source = reg_source[op.variable]

                    # Control dependency: store is inside branch on prior load
                    if op.in_branch and op.branch_var and op.branch_var in reg_source:
                        if op.dep_type == DepType.NONE:
                            op.dep_type = DepType.CTRL
                            op.dep_source = reg_source[op.branch_var]

    def _line_to_thread(self, line: int, regions: Dict[int, Tuple[int, int]]) -> int:
        """Map a source line to a thread ID."""
        if not regions:
            return 0
        for tid, (start, end) in sorted(regions.items()):
            if start <= line <= end:
                return tid
        # Default to last thread
        return max(regions.keys()) if regions else 0

    def _parse_order(self, order: str) -> MemoryOrder:
        order_map = {
            'relaxed': MemoryOrder.RELAXED,
            'consume': MemoryOrder.CONSUME,
            'acquire': MemoryOrder.ACQUIRE,
            'release': MemoryOrder.RELEASE,
            'acq_rel': MemoryOrder.ACQ_REL,
            'seq_cst': MemoryOrder.SEQ_CST,
        }
        return order_map.get(order.lower(), MemoryOrder.SEQ_CST)


# ── Fallback regex parser (for non-C/C++ code) ─────────────────────

class FallbackParser:
    """Regex-based parser for pseudocode and code tree-sitter can't handle."""

    def parse(self, code: str, language: str = "pseudo") -> Tuple[List[ExtractedOp], Dict]:
        ops = []
        metadata = {
            'n_threads': 0,
            'shared_vars': set(),
            'fences': [],
            'is_gpu': False,
            'gpu_scope': None,
            'memory_orders': set(),
            'language': language,
        }

        lines = code.split('\n')
        current_thread = 0
        current_workgroup = 0
        thread_seen = set()

        # Detect GPU
        code_lower = code.lower()
        metadata['is_gpu'] = any(kw in code_lower for kw in [
            '__global__', '__threadfence', '__syncthreads', 'barrier(', 'clk_',
            'workgroup', 'block 0', 'block 1', 'cta',
        ])

        # Thread detection patterns
        thread_re = re.compile(r'(?://|/\*)\s*[Tt]hread\s+(\d+)', re.IGNORECASE)
        wg_re = re.compile(r'(?:workgroup|block|wg)\s+(\d+)', re.IGNORECASE)

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
                m = thread_re.search(line)
                if m:
                    current_thread = int(m.group(1))
                    thread_seen.add(current_thread)
                # Detect workgroup from comment
                m_wg = wg_re.search(line)
                if m_wg:
                    current_workgroup = int(m_wg.group(1))
                continue

            # RISC-V fence
            fence_rv = re.match(r'fence\s+([rw]+)\s*,\s*([rw]+)', stripped, re.IGNORECASE)
            if fence_rv:
                pred, succ = fence_rv.group(1), fence_rv.group(2)
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.FENCE, variable='fence',
                    line=i, fence_pred=pred, fence_succ=succ, raw_text=stripped,
                ))
                metadata['fences'].append(('riscv_fence', f'{pred},{succ}'))
                thread_seen.add(current_thread)
                continue

            # ARM dmb
            dmb_re = re.match(r'dmb\s+(ish(?:st|ld)?)', stripped, re.IGNORECASE)
            if dmb_re:
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.FENCE, variable='fence',
                    line=i, raw_text=stripped,
                ))
                metadata['fences'].append(('dmb', dmb_re.group(1)))
                thread_seen.add(current_thread)
                continue

            # CUDA fences
            if '__threadfence_block' in stripped:
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.FENCE, variable='fence',
                    line=i, gpu_scope=GPUScope.CTA, raw_text=stripped,
                ))
                metadata['fences'].append(('threadfence_block', 'cta'))
                metadata['is_gpu'] = True
                metadata['gpu_scope'] = 'workgroup'
                thread_seen.add(current_thread)
                continue

            if re.search(r'__threadfence\s*\(\s*\)', stripped):
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.FENCE, variable='fence',
                    line=i, gpu_scope=GPUScope.DEVICE, raw_text=stripped,
                ))
                metadata['fences'].append(('threadfence', 'device'))
                metadata['is_gpu'] = True
                thread_seen.add(current_thread)
                continue

            if '__syncthreads' in stripped:
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.FENCE, variable='fence',
                    line=i, gpu_scope=GPUScope.CTA, raw_text=stripped,
                ))
                metadata['fences'].append(('syncthreads', 'cta'))
                metadata['is_gpu'] = True
                thread_seen.add(current_thread)
                continue

            # OpenCL barrier
            if 'barrier' in stripped.lower() and ('CLK_' in stripped or 'work_group' in stripped.lower()):
                scope = GPUScope.WORKGROUP
                if 'CLK_GLOBAL_MEM_FENCE' in stripped:
                    scope = GPUScope.DEVICE
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.FENCE, variable='fence',
                    line=i, gpu_scope=scope, raw_text=stripped,
                ))
                metadata['fences'].append(('barrier', scope.value))
                metadata['is_gpu'] = True
                metadata['gpu_scope'] = scope.value
                thread_seen.add(current_thread)
                continue

            # GCC/Clang __sync_synchronize and _mm_mfence
            if '__sync_synchronize' in stripped or '_mm_mfence' in stripped:
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.FENCE, variable='fence',
                    line=i, memory_order=MemoryOrder.SEQ_CST, raw_text=stripped,
                ))
                metadata['fences'].append(('sync', 'seq_cst'))
                thread_seen.add(current_thread)
                continue

            # atomic_thread_fence
            if 'atomic_thread_fence' in stripped:
                order_m = re.search(r'memory_order_(\w+)', stripped)
                order = order_m.group(1) if order_m else 'seq_cst'
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.FENCE, variable='fence',
                    line=i, raw_text=stripped,
                ))
                metadata['fences'].append(('thread_fence', order))
                thread_seen.add(current_thread)
                continue

            # Linux kernel macros
            smp_store_m = re.search(r'smp_store_release\s*\(\s*&?\s*(\w+)\s*,\s*(.+?)\s*\)', stripped)
            if smp_store_m:
                var, val = smp_store_m.group(1), smp_store_m.group(2)
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.STORE, variable=var,
                    value=val, line=i, memory_order=MemoryOrder.RELEASE, raw_text=stripped,
                ))
                metadata['shared_vars'].add(var)
                thread_seen.add(current_thread)
                continue

            smp_load_m = re.search(r'smp_load_acquire\s*\(\s*&?\s*(\w+)\s*\)', stripped)
            if smp_load_m:
                var = smp_load_m.group(1)
                reg = None
                reg_m = re.match(r'(r\d+)\s*=', stripped)
                if reg_m:
                    reg = reg_m.group(1)
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.LOAD, variable=var,
                    register=reg, line=i, memory_order=MemoryOrder.ACQUIRE, raw_text=stripped,
                ))
                metadata['shared_vars'].add(var)
                thread_seen.add(current_thread)
                continue

            write_once_m = re.search(r'WRITE_ONCE\s*\(\s*(\w+)\s*,\s*(.+?)\s*\)', stripped)
            if write_once_m:
                var, val = write_once_m.group(1), write_once_m.group(2)
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.STORE, variable=var,
                    value=val, line=i, raw_text=stripped,
                ))
                metadata['shared_vars'].add(var)
                thread_seen.add(current_thread)
                continue

            read_once_m = re.search(r'READ_ONCE\s*\(\s*(\w+)\s*\)', stripped)
            if read_once_m:
                var = read_once_m.group(1)
                reg = None
                reg_m = re.match(r'(r\d+)\s*=', stripped)
                if reg_m:
                    reg = reg_m.group(1)
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.LOAD, variable=var,
                    register=reg, line=i, raw_text=stripped,
                ))
                metadata['shared_vars'].add(var)
                thread_seen.add(current_thread)
                continue

            if re.search(r'smp_(?:[wr]?)mb\s*\(\s*\)', stripped):
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.FENCE, variable='fence',
                    line=i, memory_order=MemoryOrder.SEQ_CST, raw_text=stripped,
                ))
                metadata['fences'].append(('smp_mb', 'seq_cst'))
                thread_seen.add(current_thread)
                continue

            asr_m = re.search(r'atomic_store_release\s*\(\s*&?\s*(\w+)\s*,\s*(.+?)\s*\)', stripped)
            if asr_m:
                var, val = asr_m.group(1), asr_m.group(2)
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.STORE, variable=var,
                    value=val, line=i, memory_order=MemoryOrder.RELEASE, raw_text=stripped,
                ))
                metadata['shared_vars'].add(var)
                thread_seen.add(current_thread)
                continue

            ala_m = re.search(r'atomic_load_acquire\s*\(\s*&?\s*(\w+)\s*\)', stripped)
            if ala_m:
                var = ala_m.group(1)
                reg = None
                reg_m = re.match(r'(r\d+)\s*=', stripped)
                if reg_m:
                    reg = reg_m.group(1)
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.LOAD, variable=var,
                    register=reg, line=i, memory_order=MemoryOrder.ACQUIRE, raw_text=stripped,
                ))
                metadata['shared_vars'].add(var)
                thread_seen.add(current_thread)
                continue

            # Pseudocode store: store(x, 1) or St[x] = 1 or W(x) = 1
            store_re = re.compile(r'(?:store|St|W)\s*[\[\(]\s*(\w+)\s*[\]\)]\s*[=,]\s*(\d+)', re.IGNORECASE)
            m = store_re.search(stripped)
            if m:
                var, val = m.group(1), m.group(2)
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.STORE, variable=var,
                    value=val, line=i, raw_text=stripped,
                ))
                metadata['shared_vars'].add(var)
                thread_seen.add(current_thread)
                continue

            # Pseudocode load: r0 = load(x) or Ld[x] or R(x)
            load_re = re.compile(r'(r\d+)\s*=\s*(?:load|Ld|R)\s*[\[\(]\s*(\w+)\s*[\]\)]', re.IGNORECASE)
            m = load_re.search(stripped)
            if m:
                reg, var = m.group(1), m.group(2)
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.LOAD, variable=var,
                    register=reg, line=i, raw_text=stripped,
                ))
                metadata['shared_vars'].add(var)
                thread_seen.add(current_thread)
                continue

            # Simple store: x = 1;
            simple_store = re.match(r'^\s*\*?\s*(\w+)\s*=\s*(\d+)\s*;?\s*$', stripped)
            # if (r0) x = 1; → store with control dependency
            if_store_re = re.match(r'^\s*if\s*\(\s*(r\d+)\s*\)\s*(\w+)\s*=\s*(.+?)\s*;?\s*$', stripped)
            if if_store_re:
                branch_var, var, val = if_store_re.group(1), if_store_re.group(2), if_store_re.group(3)
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.STORE, variable=var,
                    value=val, line=i, raw_text=stripped,
                    in_branch=True, branch_var=branch_var,
                ))
                metadata['shared_vars'].add(var)
                thread_seen.add(current_thread)
                continue

            if simple_store:
                var, val = simple_store.group(1), simple_store.group(2)
                if var not in ('i', 'j', 'k', 'n', 'tid', 'idx'):
                    ops.append(ExtractedOp(
                        thread=current_thread, op_type=OpType.STORE, variable=var,
                        value=val, line=i, raw_text=stripped,
                    ))
                    metadata['shared_vars'].add(var)
                    thread_seen.add(current_thread)
                    continue

            # Simple load: r0 = x;
            simple_load = re.match(r'^\s*(r\d+)\s*=\s*(\w+)\s*;?\s*$', stripped)
            if simple_load:
                reg, var = simple_load.group(1), simple_load.group(2)
                if var not in ('0', '1', '2'):
                    ops.append(ExtractedOp(
                        thread=current_thread, op_type=OpType.LOAD, variable=var,
                        register=reg, line=i, raw_text=stripped,
                    ))
                    metadata['shared_vars'].add(var)
                    thread_seen.add(current_thread)
                    continue

            # C-style atomic .store/.load
            atom_store = re.search(r'(\w+)\.store\s*\(\s*(.+?)(?:\s*,\s*(?:std::)?memory_order_(\w+))?\s*\)', stripped)
            if atom_store:
                var, val = atom_store.group(1), atom_store.group(2)
                order = atom_store.group(3) or 'seq_cst'
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.STORE, variable=var,
                    value=val, line=i, raw_text=stripped,
                ))
                metadata['shared_vars'].add(var)
                metadata['memory_orders'].add(order)
                thread_seen.add(current_thread)
                continue

            atom_load = re.search(r'(\w+)\.load\s*\(\s*(?:(?:std::)?memory_order_(\w+))?\s*\)', stripped)
            if atom_load:
                var = atom_load.group(1)
                order = atom_load.group(2) or 'seq_cst'
                ops.append(ExtractedOp(
                    thread=current_thread, op_type=OpType.LOAD, variable=var,
                    line=i, raw_text=stripped,
                ))
                metadata['shared_vars'].add(var)
                metadata['memory_orders'].add(order)
                thread_seen.add(current_thread)
                continue

            # Generic assignment store (catch-all)
            gen_store = re.match(r'^\s*(\w+)\s*=\s*(.+?)\s*;?\s*$', stripped)
            if gen_store:
                var, val = gen_store.group(1), gen_store.group(2)
                if var not in ('i', 'j', 'k', 'n', 'tid', 'idx', 'size', 'len') and not var.startswith('__'):
                    # RHS is a register (r0, r1, ...) → store with data dependency
                    if re.match(r'^r\d+$', val):
                        ops.append(ExtractedOp(
                            thread=current_thread, op_type=OpType.STORE, variable=var,
                            value=val, line=i, raw_text=stripped,
                        ))
                        metadata['shared_vars'].add(var)
                    # Check if RHS is a load (non-register variable name)
                    elif re.match(r'^[a-zA-Z_]\w*$', val) and not val.isdigit() and val not in ('true', 'false', 'NULL'):
                        # r = x pattern -> load
                        ops.append(ExtractedOp(
                            thread=current_thread, op_type=OpType.LOAD, variable=val,
                            register=var, line=i, raw_text=stripped,
                        ))
                        metadata['shared_vars'].add(val)
                    else:
                        ops.append(ExtractedOp(
                            thread=current_thread, op_type=OpType.STORE, variable=var,
                            value=val, line=i, raw_text=stripped,
                        ))
                        metadata['shared_vars'].add(var)
                    thread_seen.add(current_thread)

        metadata['n_threads'] = max(len(thread_seen), 1)

        # Post-process: infer data/address/control dependencies
        self._infer_deps(ops)

        # Post-process: assign workgroup based on thread-to-workgroup mapping
        # detected from comments like "block 0", "workgroup 1"
        if metadata['is_gpu']:
            # Rebuild thread->workgroup mapping from code comments
            cur_t, cur_wg = 0, 0
            twg_map = {}
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('//') or stripped.startswith('/*'):
                    m = thread_re.search(line)
                    if m:
                        cur_t = int(m.group(1))
                    m_wg = wg_re.search(line)
                    if m_wg:
                        cur_wg = int(m_wg.group(1))
                    twg_map[cur_t] = cur_wg
            for op in ops:
                if op.thread in twg_map:
                    op.workgroup = twg_map[op.thread]

        return ops, metadata

    def _infer_deps(self, ops: List[ExtractedOp]):
        """Infer data/address/control dependencies from register usage in fallback parser."""
        thread_ops = defaultdict(list)
        for op in ops:
            thread_ops[op.thread].append(op)

        for tid, t_ops in thread_ops.items():
            reg_source = {}
            for op in t_ops:
                if op.op_type == OpType.LOAD and op.register:
                    reg_source[op.register] = op.variable
                elif op.op_type == OpType.STORE:
                    # Control dep: store inside branch on a register from prior load
                    if op.in_branch and op.branch_var and op.branch_var in reg_source:
                        op.dep_type = DepType.CTRL
                        op.dep_source = reg_source[op.branch_var]
                    # Data dep: store value is a register from a prior load (y = r0)
                    elif op.value and op.value in reg_source:
                        op.dep_type = DepType.DATA
                        op.dep_source = reg_source[op.value]
                    # Address dep: store address from a prior load
                    if op.variable in reg_source and op.dep_type == DepType.NONE:
                        op.dep_type = DepType.ADDR
                        op.dep_source = reg_source[op.variable]


# ── Pattern matcher with structural isomorphism ─────────────────────

class ASTPatternMatcher:
    """Match extracted operations against known litmus patterns using structural analysis."""

    def __init__(self):
        self._signatures = {}
        for name, pat in PATTERNS.items():
            self._signatures[name] = self._compute_signature(pat)

    def _compute_signature(self, pat: Dict) -> Dict:
        ops = pat['ops']
        n_threads = max(op.thread for op in ops) + 1
        non_fence = [op for op in ops if op.optype != 'fence']
        fence_ops = [op for op in ops if op.optype == 'fence']
        has_fence = len(fence_ops) > 0
        has_scope = any(getattr(op, 'scope', None) is not None for op in ops)

        thread_seqs = defaultdict(list)
        for op in non_fence:
            thread_seqs[op.thread].append(op.optype)

        addr_threads = defaultdict(set)
        for op in non_fence:
            addr_threads[op.addr].add(op.thread)

        n_shared = sum(1 for a, ts in addr_threads.items() if len(ts) > 1)
        stores = sum(1 for op in non_fence if op.optype == 'store')
        loads = sum(1 for op in non_fence if op.optype == 'load')
        has_dep = any(getattr(op, 'dep_on', None) is not None for op in ops)
        dep_types = set(op.dep_on for op in ops if getattr(op, 'dep_on', None))

        wgs = set(getattr(op, 'workgroup', 0) for op in ops)
        cross_wg = len(wgs) > 1

        # RISC-V fence pred/succ pairs
        fence_pairs = []
        for fop in fence_ops:
            pred = getattr(fop, 'fence_pred', None)
            succ = getattr(fop, 'fence_succ', None)
            if pred and succ:
                fence_pairs.append((fop.thread, pred, succ))

        # GPU scope types per fence
        scope_types = []
        for fop in fence_ops:
            s = getattr(fop, 'scope', None)
            if s:
                scope_types.append((fop.thread, s))

        # Scope mismatch: different scopes on different threads
        scope_mismatch = False
        if scope_types:
            scopes_set = set(s for _, s in scope_types)
            scope_mismatch = len(scopes_set) > 1

        # Communication pattern
        comm_pattern = []
        for addr, ts in sorted(addr_threads.items()):
            writers = [op.thread for op in non_fence if op.optype == 'store' and op.addr == addr]
            readers = [op.thread for op in non_fence if op.optype == 'load' and op.addr == addr]
            for w in writers:
                for r in readers:
                    if w != r:
                        comm_pattern.append(('wr_cross', w, r))
                    else:
                        comm_pattern.append(('wr_same', w, r))

        # Per-thread fence presence (for mp_fence vs mp_dmb_st vs mp_dmb_ld)
        fenced_threads = set(fop.thread for fop in fence_ops)

        # Per-thread full operation sequences including fences
        full_thread_seqs = defaultdict(list)
        for op in ops:
            full_thread_seqs[op.thread].append(op.optype)

        return {
            'n_threads': n_threads,
            'thread_seqs': dict(thread_seqs),
            'stores': stores,
            'loads': loads,
            'n_shared': n_shared,
            'has_fence': has_fence,
            'has_scope': has_scope,
            'has_dep': has_dep,
            'dep_types': dep_types,
            'cross_wg': cross_wg,
            'n_ops': len(non_fence),
            'comm_pattern': comm_pattern,
            'n_addrs': len(set(op.addr for op in non_fence)),
            'fence_pairs': fence_pairs,
            'scope_types': scope_types,
            'scope_mismatch': scope_mismatch,
            'fenced_threads': fenced_threads,
            'full_thread_seqs': dict(full_thread_seqs),
        }

    def match(self, ops: List[ExtractedOp], metadata: Dict) -> List[ASTPatternMatch]:
        """Find best matching patterns."""
        if not ops:
            return []

        code_sig = self._compute_code_signature(ops, metadata)
        matches = []

        for pat_name, pat_sig in self._signatures.items():
            score, match_type = self._compute_similarity(code_sig, pat_sig, ops, pat_name)
            if score > 0.25:
                matches.append(ASTPatternMatch(
                    pattern_name=pat_name,
                    confidence=score,
                    match_type=match_type,
                    extracted_ops=ops,
                ))

        # Canonical patterns get a small tiebreaker boost to prefer base forms
        # over structurally-identical domain aliases
        _CANONICAL = {
            'mp', 'sb', 'lb', 'iriw', 'wrc', 'rwc', '2+2w', 'dekker', 'peterson',
            'corr', 'cowr', 'coww', 'corw', 'isa2', 'r', 's', 'amoswap', '3sb',
            'mp_fence', 'sb_fence', 'lb_fence', 'iriw_fence', 'wrc_fence', 'rwc_fence',
            'mp_data', 'mp_addr', 'mp_3thread', 'mp_rfi', 'sb_rfi', 'sb_3thread',
            'mp_co', 'lb_data', 'wrc_addr', 'mp_dmb_st', 'mp_dmb_ld',
            'mp_fence_ww_rr', 'sb_fence_wr', 'lb_fence_rw', 'mp_fence_wr',
        }
        for m in matches:
            if m.pattern_name in _CANONICAL:
                m.confidence += 0.001  # tiny tiebreaker
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches[:10]

    def _compute_code_signature(self, ops: List[ExtractedOp], metadata: Dict) -> Dict:
        non_fence = [op for op in ops if op.op_type not in (OpType.FENCE,)]
        has_fence = any(op.op_type == OpType.FENCE for op in ops)
        has_scope = any(op.gpu_scope != GPUScope.NONE for op in ops)

        thread_seqs = defaultdict(list)
        for op in non_fence:
            thread_seqs[op.thread].append(op.op_type.value)

        addr_threads = defaultdict(set)
        for op in non_fence:
            addr_threads[op.variable].add(op.thread)

        n_shared = sum(1 for v, ts in addr_threads.items() if len(ts) > 1)
        stores = sum(1 for op in non_fence if op.op_type == OpType.STORE)
        loads = sum(1 for op in non_fence if op.op_type == OpType.LOAD)
        has_dep = any(op.dep_type != DepType.NONE for op in ops)
        dep_types = set(op.dep_type.value for op in ops if op.dep_type != DepType.NONE and op.dep_type.value)

        wgs = set(op.workgroup for op in ops)
        cross_wg = len(wgs) > 1

        comm_pattern = []
        for var, ts in sorted(addr_threads.items()):
            writers = [op.thread for op in non_fence if op.op_type == OpType.STORE and op.variable == var]
            readers = [op.thread for op in non_fence if op.op_type == OpType.LOAD and op.variable == var]
            for w in writers:
                for r in readers:
                    if w != r:
                        comm_pattern.append(('wr_cross', w, r))
                    else:
                        comm_pattern.append(('wr_same', w, r))

        # RISC-V fence pred/succ pairs
        fence_ops = [op for op in ops if op.op_type == OpType.FENCE]
        fence_pairs = []
        for fop in fence_ops:
            pred = getattr(fop, 'fence_pred', None)
            succ = getattr(fop, 'fence_succ', None)
            if pred and succ:
                fence_pairs.append((fop.thread, pred, succ))

        # GPU scope types
        scope_types = []
        for fop in fence_ops:
            gs = fop.gpu_scope
            if gs and gs != GPUScope.NONE:
                scope_types.append((fop.thread, gs.value))

        scope_mismatch = False
        if scope_types:
            scopes_set = set(s for _, s in scope_types)
            scope_mismatch = len(scopes_set) > 1

        # Per-thread fence presence
        fenced_threads = set(fop.thread for fop in fence_ops)

        # Per-thread full operation sequences including fences
        full_thread_seqs = defaultdict(list)
        for op in ops:
            full_thread_seqs[op.thread].append(op.op_type.value)

        # Note: C++ memory orderings (release/acquire/seq_cst) and kernel
        # macros (smp_store_release/smp_load_acquire) provide implicit ordering
        # but are NOT treated as fences in the signature, since benchmark
        # expectations use base patterns (mp, sb) for these cases.

        return {
            'n_threads': metadata.get('n_threads', max((op.thread for op in ops), default=0) + 1),
            'thread_seqs': dict(thread_seqs),
            'stores': stores,
            'loads': loads,
            'n_shared': n_shared,
            'has_fence': has_fence,
            'has_scope': has_scope,
            'has_dep': has_dep,
            'dep_types': dep_types,
            'cross_wg': cross_wg,
            'n_ops': len(non_fence),
            'comm_pattern': comm_pattern,
            'n_addrs': len(set(op.variable for op in non_fence)),
            'fence_pairs': fence_pairs,
            'scope_types': scope_types,
            'scope_mismatch': scope_mismatch,
            'fenced_threads': fenced_threads,
            'full_thread_seqs': dict(full_thread_seqs),
        }

    def _compute_similarity(self, code_sig: Dict, pat_sig: Dict,
                            ops: List[ExtractedOp], pat_name: str) -> Tuple[float, str]:
        score = 0.0
        max_score = 0.0

        # Thread count (weight 3)
        max_score += 3.0
        if code_sig['n_threads'] == pat_sig['n_threads']:
            score += 3.0
        elif abs(code_sig['n_threads'] - pat_sig['n_threads']) == 1:
            score += 1.0

        # Operation counts (weight 2)
        max_score += 2.0
        if code_sig['stores'] == pat_sig['stores'] and code_sig['loads'] == pat_sig['loads']:
            score += 2.0
        elif (abs(code_sig['stores'] - pat_sig['stores']) <= 1 and
              abs(code_sig['loads'] - pat_sig['loads']) <= 1):
            score += 1.0

        # Address count match (weight 1.5)
        max_score += 1.5
        if code_sig['n_addrs'] == pat_sig['n_addrs']:
            score += 1.5
        elif abs(code_sig['n_addrs'] - pat_sig['n_addrs']) == 1:
            score += 0.5

        # Per-thread sequence similarity (weight 4)
        max_score += 4.0
        seq_score = self._seq_similarity(code_sig['thread_seqs'], pat_sig['thread_seqs'])
        score += 4.0 * seq_score

        # Communication pattern (weight 3)
        max_score += 3.0
        comm_score = self._comm_similarity(code_sig['comm_pattern'], pat_sig['comm_pattern'])
        score += 3.0 * comm_score

        # Fence match (weight 1.5)
        max_score += 1.5
        if code_sig['has_fence'] == pat_sig['has_fence']:
            score += 1.5

        # Scope match for GPU (weight 1.5)
        max_score += 1.5
        if code_sig['has_scope'] == pat_sig['has_scope']:
            score += 1.5

        # Dependency match (weight 1.5)
        max_score += 1.5
        if code_sig['has_dep'] == pat_sig['has_dep']:
            score += 1.0
        if code_sig['dep_types'] == pat_sig['dep_types']:
            score += 0.5
        elif code_sig['has_dep'] and pat_sig['has_dep']:
            pass  # different dep types → no bonus

        # Cross-workgroup (weight 1)
        max_score += 1.0
        if code_sig['cross_wg'] == pat_sig['cross_wg']:
            score += 1.0

        # RISC-V fence pred/succ specificity (weight 3)
        code_fp = code_sig.get('fence_pairs', [])
        pat_fp = pat_sig.get('fence_pairs', [])
        if code_fp or pat_fp:
            max_score += 3.0
            if code_fp and pat_fp:
                # Compare fence pred/succ sets (ignoring thread IDs)
                code_preds = sorted([(p, s) for _, p, s in code_fp])
                pat_preds = sorted([(p, s) for _, p, s in pat_fp])
                if code_preds == pat_preds:
                    score += 3.0
                elif len(code_preds) == len(pat_preds):
                    matching = sum(1 for a, b in zip(code_preds, pat_preds) if a == b)
                    score += 3.0 * matching / len(code_preds)
            elif not code_fp and not pat_fp:
                score += 3.0
            elif not code_fp and pat_fp and code_sig['has_fence']:
                # Code has full fence (no pred/succ) → compatible with directional fences
                # A full fence subsumes any directional fence on the same thread
                code_ft = code_sig.get('fenced_threads', set())
                pat_fp_threads = set(t for t, _, _ in pat_fp)
                if pat_fp_threads <= code_ft:
                    score += 2.5  # strong compatibility
                else:
                    score += 1.0  # partial compatibility

        # GPU scope type specificity (weight 2.5)
        code_st = code_sig.get('scope_types', [])
        pat_st = pat_sig.get('scope_types', [])
        if code_st or pat_st:
            max_score += 2.5
            if code_st and pat_st:
                code_scopes = sorted([s for _, s in code_st])
                pat_scopes = sorted([s for _, s in pat_st])
                if code_scopes == pat_scopes:
                    score += 2.5
                else:
                    # Scope compatibility: system >= device >= workgroup/cta
                    scope_compat = {
                        ('system', 'device'): 0.8,
                        ('device', 'system'): 0.8,
                        ('cta', 'workgroup'): 1.0,
                        ('workgroup', 'cta'): 1.0,
                        ('device', 'workgroup'): 0.3,
                        ('workgroup', 'device'): 0.3,
                    }
                    compat_sum = 0.0
                    n = max(len(code_scopes), len(pat_scopes))
                    for cs, ps in zip(code_scopes, pat_scopes):
                        if cs == ps:
                            compat_sum += 1.0
                        else:
                            compat_sum += scope_compat.get((cs, ps), 0.0)
                    score += 2.5 * compat_sum / n
            elif not code_st and not pat_st:
                score += 2.5

        # Scope mismatch match (weight 2)
        code_sm = code_sig.get('scope_mismatch', False)
        pat_sm = pat_sig.get('scope_mismatch', False)
        if code_sm or pat_sm:
            max_score += 2.0
            if code_sm == pat_sm:
                score += 2.0

        # Fence per-thread placement (weight 2.5) — distinguishes mp_fence/mp_dmb_st/mp_dmb_ld
        code_ft = code_sig.get('fenced_threads', set())
        pat_ft = pat_sig.get('fenced_threads', set())
        if code_ft or pat_ft:
            max_score += 2.5
            if code_ft == pat_ft:
                score += 2.5
            elif code_ft and pat_ft and len(code_ft) == len(pat_ft):
                score += 1.5
            elif bool(code_ft) == bool(pat_ft):
                score += 0.5

        # Full thread sequence similarity including fences (weight 2)
        code_fts = code_sig.get('full_thread_seqs', {})
        pat_fts = pat_sig.get('full_thread_seqs', {})
        if code_fts and pat_fts:
            max_score += 2.0
            fts_score = self._seq_similarity(code_fts, pat_fts)
            score += 2.0 * fts_score

        norm_score = score / max_score if max_score > 0 else 0.0

        # Determine match type
        if norm_score > 0.9:
            match_type = 'exact'
        elif norm_score > 0.6:
            match_type = 'structural'
        else:
            match_type = 'heuristic'

        return norm_score, match_type

    def _seq_similarity(self, seqs1: Dict, seqs2: Dict) -> float:
        if not seqs1 or not seqs2:
            return 0.0

        sorted1 = sorted(seqs1.values(), key=lambda s: tuple(s))
        sorted2 = sorted(seqs2.values(), key=lambda s: tuple(s))

        if len(sorted1) != len(sorted2):
            min_len = min(len(sorted1), len(sorted2))
            sorted1 = sorted1[:min_len]
            sorted2 = sorted2[:min_len]
            penalty = 0.7
        else:
            penalty = 1.0

        total_sim = 0.0
        for s1, s2 in zip(sorted1, sorted2):
            if s1 == s2:
                total_sim += 1.0
            elif len(s1) == len(s2):
                matching = sum(1 for a, b in zip(s1, s2) if a == b)
                total_sim += matching / len(s1)
            else:
                lcs = self._lcs_length(s1, s2)
                total_sim += 2.0 * lcs / (len(s1) + len(s2))

        return (total_sim / max(len(sorted1), 1)) * penalty

    def _comm_similarity(self, comm1: List, comm2: List) -> float:
        if not comm1 and not comm2:
            return 1.0
        if not comm1 or not comm2:
            return 0.0
        # Normalize: abstract thread IDs
        types1 = sorted([c[0] for c in comm1])
        types2 = sorted([c[0] for c in comm2])
        if types1 == types2:
            return 1.0
        common = set(types1) & set(types2)
        total = set(types1) | set(types2)
        return len(common) / len(total) if total else 0.0

    def _lcs_length(self, s1: list, s2: list) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]


# ── Unified analyzer ───────────────────────────────────────────────

class ASTAnalyzer:
    """Unified AST-based concurrent code analyzer."""

    def __init__(self):
        self.matcher = ASTPatternMatcher()
        self._ts_extractor = None
        self._fallback = FallbackParser()

    def _get_ts(self):
        if self._ts_extractor is None and HAS_TREESITTER:
            try:
                self._ts_extractor = TreeSitterExtractor()
            except Exception:
                pass
        return self._ts_extractor

    def analyze(self, code: str, language: str = "auto") -> ASTAnalysisResult:
        """Analyze code and return matched patterns with AST-level analysis."""
        if language == "auto":
            language = self._detect_language(code)

        parse_method = "fallback_regex"
        ts = self._get_ts()

        if ts and language in ("c", "cpp", "cuda"):
            try:
                ops, metadata = ts.parse(code, language)
                parse_method = "ast"
            except Exception:
                ops, metadata = self._fallback.parse(code, language)
        else:
            ops, metadata = self._fallback.parse(code, language)

        # If no ops from AST, try fallback
        if not ops and parse_method == "ast":
            ops, metadata = self._fallback.parse(code, language)
            parse_method = "fallback_regex"

        patterns = self.matcher.match(ops, metadata)

        # Detect dependencies for reporting
        deps_found = []
        for op in ops:
            if op.dep_type != DepType.NONE:
                deps_found.append({
                    'thread': op.thread,
                    'type': op.dep_type.value,
                    'source': op.dep_source,
                    'target': op.variable,
                })

        warnings = []
        if not ops:
            warnings.append("No memory operations detected.")
        if metadata['n_threads'] < 2 and not metadata.get('is_gpu'):
            warnings.append("Single-threaded code; concurrency patterns need 2+ threads.")

        # Compute coverage confidence: how well do matched patterns cover
        # the actual concurrent operations in the code?
        coverage_confidence = 1.0
        unrecognized_ops = []
        if ops and metadata.get('n_threads', 1) >= 2:
            # Count concurrent ops (cross-thread shared-variable accesses)
            shared_addrs = set()
            addr_threads = defaultdict(set)
            for op in ops:
                if op.op_type != OpType.FENCE:
                    addr_threads[op.variable].add(op.thread)
            for addr, threads in addr_threads.items():
                if len(threads) > 1:
                    shared_addrs.add(addr)

            total_concurrent_ops = sum(
                1 for op in ops
                if op.op_type != OpType.FENCE and op.variable in shared_addrs
            )

            if total_concurrent_ops > 0 and patterns:
                best = patterns[0]
                best_conf = best.confidence

                # Check structural match: compare number of operations,
                # threads, and shared variables against the best pattern
                if best.pattern_name in PATTERNS:
                    pat = PATTERNS[best.pattern_name]
                    pat_ops = [o for o in pat['ops'] if o.optype != 'fence']
                    pat_n_ops = len(pat_ops)
                    pat_n_threads = max((o.thread for o in pat['ops']), default=0) + 1
                    code_n_threads = metadata.get('n_threads', 1)

                    # Excess ops/threads not accounted for by the pattern
                    excess_ops = max(0, total_concurrent_ops - pat_n_ops)
                    excess_threads = max(0, code_n_threads - pat_n_threads)

                    if excess_ops == 0 and excess_threads == 0:
                        coverage_confidence = best_conf
                    else:
                        # Penalize for operations the pattern doesn't explain
                        explained_ratio = pat_n_ops / total_concurrent_ops
                        coverage_confidence = best_conf * explained_ratio
                else:
                    coverage_confidence = best_conf
            elif total_concurrent_ops > 0 and not patterns:
                coverage_confidence = 0.0

            # Emit unrecognized pattern warning when coverage is low
            if coverage_confidence < 0.5:
                unrecognized_vars = []
                for op in ops:
                    if (op.op_type != OpType.FENCE
                            and op.variable in shared_addrs):
                        unrecognized_vars.append({
                            'thread': op.thread,
                            'op_type': op.op_type.value,
                            'variable': op.variable,
                            'line': op.line,
                        })
                unrecognized_ops = unrecognized_vars
                warnings.append(
                    f"UnrecognizedPatternWarning: {total_concurrent_ops} "
                    f"concurrent operation(s) do not match any of the 75 "
                    f"built-in patterns (coverage: {coverage_confidence:.0%}). "
                    f"Silence does NOT mean safety — consider manual review "
                    f"or Dartagnan for full-program verification."
                )

        code_hash = hashlib.md5(code.encode()).hexdigest()[:12]

        return ASTAnalysisResult(
            code_hash=code_hash,
            language=language,
            patterns_found=patterns,
            extracted_ops=ops,
            n_threads=metadata.get('n_threads', 1),
            shared_vars=metadata.get('shared_vars', set()),
            has_fences=bool(metadata.get('fences')),
            is_gpu=metadata.get('is_gpu', False),
            gpu_scope=metadata.get('gpu_scope'),
            memory_orders_used=metadata.get('memory_orders', set()),
            dependencies_found=deps_found,
            warnings=warnings,
            parse_method=parse_method,
            coverage_confidence=coverage_confidence,
            unrecognized_ops=unrecognized_ops,
        )

    def check_portability(self, code: str, target_arch: str = None,
                          language: str = "auto") -> List[Dict]:
        """Full pipeline: AST parse → match → portability check."""
        analysis = self.analyze(code, language)
        results = []

        for match in analysis.patterns_found[:5]:
            pat_name = match.pattern_name
            if pat_name not in PATTERNS:
                continue

            pat = PATTERNS[pat_name]
            n_threads = max(op.thread for op in pat['ops']) + 1
            lt = LitmusTest(
                name=pat_name, n_threads=n_threads,
                addresses=pat['addresses'], ops=pat['ops'], forbidden=pat['forbidden'],
            )

            archs = [target_arch] if target_arch else list(ARCHITECTURES.keys())
            for arch in archs:
                model = ARCHITECTURES[arch]
                if not lt.forbidden:
                    results.append({
                        'pattern': pat_name,
                        'confidence': match.confidence,
                        'match_type': match.match_type,
                        'target_arch': arch,
                        'safe': True,
                        'fence_fix': None,
                    })
                    continue

                forbidden_allowed, _ = verify_test(lt, model)
                safe = not forbidden_allowed
                fence_fix = None
                if not safe:
                    fence_fix = recommend_fence(lt, arch, model)

                results.append({
                    'pattern': pat_name,
                    'confidence': match.confidence,
                    'match_type': match.match_type,
                    'target_arch': arch,
                    'safe': safe,
                    'fence_fix': fence_fix,
                })

        return results

    def _detect_language(self, code: str) -> str:
        code_lower = code.lower()
        if '__global__' in code_lower or '__device__' in code_lower or '__shared__' in code_lower:
            return "cuda"
        if 'clk_global_mem_fence' in code_lower or 'clk_local_mem_fence' in code_lower:
            return "opencl"
        if 'std::atomic' in code or 'memory_order_' in code or '.store(' in code or '.load(' in code:
            return "cpp"
        if ('#include' in code or 'void ' in code or 'int ' in code
                or '__sync_synchronize' in code or '__atomic_' in code
                or 'smp_store_release' in code or 'smp_load_acquire' in code
                or 'READ_ONCE' in code or 'WRITE_ONCE' in code):
            return "c"
        return "pseudo"


# ── Module-level convenience ────────────────────────────────────────

_ast_analyzer = None

def get_ast_analyzer() -> ASTAnalyzer:
    global _ast_analyzer
    if _ast_analyzer is None:
        _ast_analyzer = ASTAnalyzer()
    return _ast_analyzer

def ast_analyze_code(code: str, language: str = "auto") -> ASTAnalysisResult:
    """Analyze code using AST-based analysis."""
    return get_ast_analyzer().analyze(code, language)

def ast_check_portability(code: str, target_arch: str = None,
                          language: str = "auto") -> List[Dict]:
    """Check code portability using AST-based analysis."""
    return get_ast_analyzer().check_portability(code, target_arch, language)
