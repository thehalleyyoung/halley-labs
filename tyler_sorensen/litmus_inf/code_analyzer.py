#!/usr/bin/env python3
"""
Code-to-pattern analyzer for LITMUS∞.

Accepts real C/C++/CUDA/pseudocode strings, extracts memory operation
patterns from concurrent code, and maps them to known litmus test patterns
for portability checking. This bridges the gap between arbitrary code
and the 57-pattern portability database.

Supports:
  - C/C++ with pthreads, std::atomic, volatile
  - CUDA kernels with __threadfence_block(), __syncthreads()
  - Pseudocode (store/load notation)
  - Custom user-defined MemOp sequences (arbitrary patterns)
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

from portcheck import (
    MemOp, LitmusTest, PATTERNS, ARCHITECTURES,
    compute_joint_automorphisms, compute_orbits,
    verify_test, recommend_fence, _identify_per_thread_violations,
    _is_scope_mismatch_pattern, _is_gpu_model,
)


# ── Result types ────────────────────────────────────────────────────

@dataclass
class PatternMatch:
    """A matched litmus test pattern from code analysis."""
    pattern_name: str
    confidence: float  # 0.0 to 1.0
    matched_ops: List[Tuple[int, str, str]]  # (thread, optype, var)
    code_lines: List[int]  # source line numbers
    description: str = ""

    def __repr__(self):
        return f"PatternMatch({self.pattern_name}, conf={self.confidence:.2f})"


@dataclass
class CodeAnalysisResult:
    """Full result of analyzing a code snippet."""
    code_hash: str
    patterns_found: List[PatternMatch]
    extracted_ops: List[MemOp]
    n_threads: int
    shared_vars: Set[str]
    has_fences: bool
    is_gpu: bool
    gpu_scope: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def __repr__(self):
        pats = ", ".join(m.pattern_name for m in self.patterns_found[:3])
        return f"CodeAnalysisResult({self.n_threads} threads, patterns=[{pats}])"


# ── Code parser ─────────────────────────────────────────────────────

class CodeParser:
    """Parse concurrent code in C/C++/CUDA/pseudocode into MemOp sequences."""

    # Regex patterns for different code styles
    _PATTERNS = {
        # C/C++ atomic stores
        'atomic_store': re.compile(
            r'(?:atomic_store|__atomic_store_n|\.store)\s*\(\s*&?\s*(\w+)\s*,\s*(\w+)',
            re.IGNORECASE),
        'atomic_store_explicit': re.compile(
            r'(\w+)\.store\s*\(\s*(\w+)\s*,\s*std::memory_order_(\w+)',
            re.IGNORECASE),
        # C/C++ atomic loads
        'atomic_load': re.compile(
            r'(?:atomic_load|__atomic_load_n|\.load)\s*\(\s*&?\s*(\w+)',
            re.IGNORECASE),
        'atomic_load_explicit': re.compile(
            r'(\w+)\.load\s*\(\s*std::memory_order_(\w+)',
            re.IGNORECASE),
        # Simple assignment stores (volatile or shared)
        'simple_store': re.compile(
            r'^\s*\*?\s*(\w+)\s*=\s*(\d+)\s*;', re.MULTILINE),
        # Simple assignment loads
        'simple_load': re.compile(
            r'^\s*(?:int|long|auto|volatile\s+\w+)\s+(\w+)\s*=\s*\*?\s*(\w+)\s*;',
            re.MULTILINE),
        'simple_load_reg': re.compile(
            r'^\s*(r\d+)\s*=\s*\*?\s*(\w+)\s*;', re.MULTILINE),
        # Pseudocode stores/loads
        'pseudo_store': re.compile(
            r'(?:store|St|W)\s*[\[\(]\s*(\w+)\s*[\]\)]\s*[=,]\s*(\d+)',
            re.IGNORECASE),
        'pseudo_load': re.compile(
            r'(?:load|Ld|R)\s*[\[\(]\s*(\w+)\s*[\]\)]',
            re.IGNORECASE),
        # Thread markers
        'thread_comment': re.compile(
            r'(?://|/\*)\s*[Tt]hread\s+(\d+)', re.IGNORECASE),
        'thread_func': re.compile(
            r'(?:void\s+)?thread_?(\d+)\s*\(', re.IGNORECASE),
        'thread_pragma': re.compile(
            r'Thread\s+(\d+)\s*:', re.IGNORECASE),
        # Fences
        'fence_mfence': re.compile(r'(?:_mm_mfence|__sync_synchronize|mfence)', re.IGNORECASE),
        'fence_dmb': re.compile(r'(?:dmb\s+(ish(?:st|ld)?)|__dmb\s*\(\s*\w+\s*\))', re.IGNORECASE),
        'fence_riscv': re.compile(r'fence\s+([rw]+)\s*,\s*([rw]+)', re.IGNORECASE),
        'fence_atomic_thread': re.compile(r'atomic_thread_fence\s*\(\s*std::memory_order_(\w+)', re.IGNORECASE),
        # CUDA specific
        'cuda_threadfence_block': re.compile(r'__threadfence_block\s*\(\s*\)', re.IGNORECASE),
        'cuda_threadfence': re.compile(r'__threadfence\s*\(\s*\)(?!_)', re.IGNORECASE),
        'cuda_syncthreads': re.compile(r'__syncthreads\s*\(\s*\)', re.IGNORECASE),
        'cuda_kernel': re.compile(r'__global__\s+void\s+(\w+)', re.IGNORECASE),
        'cuda_shared': re.compile(r'__shared__\s+\w+\s+(\w+)', re.IGNORECASE),
        'cuda_blockidx': re.compile(r'blockIdx\.(\w+)', re.IGNORECASE),
        'cuda_threadidx': re.compile(r'threadIdx\.(\w+)', re.IGNORECASE),
    }

    def parse(self, code: str, language: str = "auto") -> Tuple[List[MemOp], Dict]:
        """Parse code string into a list of MemOps and metadata."""
        if language == "auto":
            language = self._detect_language(code)

        metadata = {
            'language': language,
            'is_gpu': language in ('cuda', 'opencl', 'vulkan'),
            'gpu_scope': None,
            'n_threads': 0,
            'shared_vars': set(),
            'fences': [],
            'line_map': {},  # op_index -> source line
        }

        ops = []
        lines = code.split('\n')
        current_thread = 0
        current_workgroup = 0
        thread_sections = self._split_threads(code)

        if thread_sections:
            for tid, section_lines in thread_sections.items():
                section_code = '\n'.join(section_lines)
                thread_ops, thread_meta = self._parse_section(
                    section_code, tid, current_workgroup, language)
                ops.extend(thread_ops)
                metadata['shared_vars'].update(thread_meta.get('vars', set()))
                metadata['fences'].extend(thread_meta.get('fences', []))
                if language == 'cuda' and self._has_cross_block_comm(section_code):
                    current_workgroup += 1
        else:
            # Single section — try column-based parsing (// Thread 0 | // Thread 1)
            ops, meta = self._parse_flat(code, language)
            metadata['shared_vars'] = meta.get('vars', set())
            metadata['fences'] = meta.get('fences', [])

        metadata['n_threads'] = max((op.thread for op in ops), default=-1) + 1

        if language == 'cuda':
            metadata['is_gpu'] = True
            metadata['gpu_scope'] = self._detect_cuda_scope(code)

        return ops, metadata

    def _detect_language(self, code: str) -> str:
        if self._PATTERNS['cuda_kernel'].search(code) or '__threadfence' in code:
            return 'cuda'
        if 'CLK_GLOBAL_MEM_FENCE' in code or 'work_group_barrier' in code:
            return 'opencl'
        if 'std::atomic' in code or '.store(' in code or '.load(' in code:
            return 'cpp'
        if 'atomic_store' in code or '__atomic_' in code:
            return 'c'
        if re.search(r'(?:store|load|St|Ld|W\[|R\[)\s*[\[\(]', code, re.IGNORECASE):
            return 'pseudo'
        return 'c'

    def _split_threads(self, code: str) -> Dict[int, List[str]]:
        """Split code into per-thread sections."""
        sections = {}
        current_tid = None
        current_lines = []

        for line in code.split('\n'):
            # Check for thread markers
            m = self._PATTERNS['thread_pragma'].match(line.strip())
            if m:
                if current_tid is not None:
                    sections[current_tid] = current_lines
                current_tid = int(m.group(1))
                current_lines = []
                continue

            m = self._PATTERNS['thread_comment'].search(line)
            if m:
                if current_tid is not None:
                    sections[current_tid] = current_lines
                current_tid = int(m.group(1))
                current_lines = []
                continue

            m = self._PATTERNS['thread_func'].search(line)
            if m:
                if current_tid is not None:
                    sections[current_tid] = current_lines
                current_tid = int(m.group(1))
                current_lines = []
                continue

            if current_tid is not None:
                current_lines.append(line)

        if current_tid is not None:
            sections[current_tid] = current_lines

        # Try column-based splitting (Thread 0 || Thread 1)
        if not sections and '||' in code:
            parts = code.split('||')
            for i, part in enumerate(parts):
                sections[i] = part.strip().split('\n')

        return sections

    def _parse_section(self, code: str, thread_id: int, workgroup: int,
                       language: str) -> Tuple[List[MemOp], Dict]:
        """Parse a single thread section into MemOps."""
        ops = []
        meta = {'vars': set(), 'fences': []}
        reg_counter = 0

        for line_num, line in enumerate(code.split('\n')):
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('#'):
                continue

            # Check for fences first
            fence_op = self._parse_fence(line, thread_id, workgroup, language)
            if fence_op:
                ops.append(fence_op)
                meta['fences'].append(fence_op)
                continue

            # Try to parse memory operations
            parsed = self._parse_memop(line, thread_id, workgroup, language, reg_counter)
            if parsed:
                for op in parsed:
                    ops.append(op)
                    if op.addr:
                        meta['vars'].add(op.addr)
                    if op.optype == 'load':
                        reg_counter += 1

        return ops, meta

    def _parse_memop(self, line: str, tid: int, wg: int,
                     lang: str, reg_idx: int) -> List[MemOp]:
        """Parse a single line into MemOp(s)."""
        ops = []

        # Pseudocode store: store(x, 1) or St[x]=1 or W[x]=1
        m = self._PATTERNS['pseudo_store'].search(line)
        if m:
            ops.append(MemOp(tid, 'store', m.group(1), int(m.group(2)), workgroup=wg))
            return ops

        # Pseudocode load: load(x) or Ld[x] or R[x]
        m = self._PATTERNS['pseudo_load'].search(line)
        if m and not self._PATTERNS['pseudo_store'].search(line):
            ops.append(MemOp(tid, 'load', m.group(1), reg=f'r{reg_idx}', workgroup=wg))
            return ops

        # C++ atomic store: x.store(1, ...)
        m = self._PATTERNS['atomic_store_explicit'].search(line)
        if m:
            ops.append(MemOp(tid, 'store', m.group(1), 1, workgroup=wg))
            return ops
        m = self._PATTERNS['atomic_store'].search(line)
        if m:
            ops.append(MemOp(tid, 'store', m.group(1), 1, workgroup=wg))
            return ops

        # C++ atomic load: x.load(...)
        m = self._PATTERNS['atomic_load_explicit'].search(line)
        if m:
            ops.append(MemOp(tid, 'load', m.group(1), reg=f'r{reg_idx}', workgroup=wg))
            return ops
        m = self._PATTERNS['atomic_load'].search(line)
        if m:
            ops.append(MemOp(tid, 'load', m.group(1), reg=f'r{reg_idx}', workgroup=wg))
            return ops

        # Simple register load: r0 = x;
        m = self._PATTERNS['simple_load_reg'].search(line)
        if m:
            ops.append(MemOp(tid, 'load', m.group(2), reg=m.group(1), workgroup=wg))
            return ops

        # Simple typed load: int r = x;
        m = self._PATTERNS['simple_load'].search(line)
        if m:
            ops.append(MemOp(tid, 'load', m.group(2), reg=m.group(1), workgroup=wg))
            return ops

        # Simple store: x = 1;
        m = self._PATTERNS['simple_store'].search(line)
        if m:
            var = m.group(1)
            # Skip common non-shared vars
            if var not in ('i', 'j', 'k', 'n', 'ret', 'result', 'idx', 'tid'):
                ops.append(MemOp(tid, 'store', var, int(m.group(2)), workgroup=wg))
                return ops

        return ops

    def _parse_fence(self, line: str, tid: int, wg: int, lang: str) -> Optional[MemOp]:
        """Parse fence/barrier from a line."""
        # CUDA fences
        if self._PATTERNS['cuda_threadfence_block'].search(line):
            return MemOp(tid, 'fence', '', scope='workgroup', workgroup=wg)
        if self._PATTERNS['cuda_threadfence'].search(line):
            return MemOp(tid, 'fence', '', scope='device', workgroup=wg)
        if self._PATTERNS['cuda_syncthreads'].search(line):
            return MemOp(tid, 'fence', '', scope='workgroup', workgroup=wg)

        # x86 mfence
        if self._PATTERNS['fence_mfence'].search(line):
            return MemOp(tid, 'fence', '', workgroup=wg)

        # ARM dmb
        m = self._PATTERNS['fence_dmb'].search(line)
        if m:
            dmb_type = m.group(1) or 'ish'
            fence_r = dmb_type != 'ishst'
            fence_w = dmb_type != 'ishld'
            return MemOp(tid, 'fence', '', fence_read=fence_r, fence_write=fence_w, workgroup=wg)

        # RISC-V fence
        m = self._PATTERNS['fence_riscv'].search(line)
        if m:
            return MemOp(tid, 'fence', '', fence_pred=m.group(1), fence_succ=m.group(2), workgroup=wg)

        # C++ atomic_thread_fence
        m = self._PATTERNS['fence_atomic_thread'].search(line)
        if m:
            return MemOp(tid, 'fence', '', workgroup=wg)

        return None

    def _parse_flat(self, code: str, language: str) -> Tuple[List[MemOp], Dict]:
        """Parse code without explicit thread markers — infer threads from structure."""
        ops = []
        meta = {'vars': set(), 'fences': []}

        # Try semicolon-separated or line-separated with || delimiter
        if '||' in code:
            parts = code.split('||')
            reg_idx = 0
            for tid, part in enumerate(parts):
                for line in part.strip().split(';'):
                    line = line.strip()
                    if not line:
                        continue
                    parsed = self._parse_memop(line, tid, 0, language, reg_idx)
                    if parsed:
                        ops.extend(parsed)
                        for op in parsed:
                            if op.addr:
                                meta['vars'].add(op.addr)
                            if op.optype == 'load':
                                reg_idx += 1
                    fence = self._parse_fence(line, tid, 0, language)
                    if fence:
                        ops.append(fence)
                        meta['fences'].append(fence)
        else:
            # Single-thread or pseudocode
            thread_ops, thread_meta = self._parse_section(code, 0, 0, language)
            ops = thread_ops
            meta = thread_meta

        return ops, meta

    def _detect_cuda_scope(self, code: str) -> str:
        if '__threadfence()' in code and '__threadfence_block' not in code:
            return 'device'
        if '__threadfence_block' in code:
            return 'workgroup'
        if '__syncthreads' in code:
            return 'workgroup'
        return 'device'

    def _has_cross_block_comm(self, code: str) -> bool:
        return 'blockIdx' in code or 'gridDim' in code


# ── Pattern matcher ─────────────────────────────────────────────────

class PatternMatcher:
    """Match extracted MemOps against known litmus test patterns."""

    def __init__(self):
        self._pattern_signatures = {}
        for name, pat in PATTERNS.items():
            self._pattern_signatures[name] = self._compute_signature(pat['ops'])

    def _compute_signature(self, ops: List[MemOp]) -> Dict:
        """Compute a structural signature for a pattern."""
        n_threads = max(op.thread for op in ops) + 1
        non_fence = [op for op in ops if op.optype != 'fence']
        has_fence = any(op.optype == 'fence' for op in ops)
        has_scope = any(op.scope is not None for op in ops if op.optype == 'fence')

        # Per-thread operation sequence (abstracting away variable names)
        thread_seqs = defaultdict(list)
        for op in non_fence:
            thread_seqs[op.thread].append(op.optype)

        # Cross-thread address sharing pattern
        addr_threads = defaultdict(set)
        for op in non_fence:
            addr_threads[op.addr].add(op.thread)
        n_shared_addrs = sum(1 for a, ts in addr_threads.items() if len(ts) > 1)

        # Operation type counts
        n_stores = sum(1 for op in non_fence if op.optype == 'store')
        n_loads = sum(1 for op in non_fence if op.optype == 'load')

        # Dependencies
        has_dep = any(op.dep_on is not None for op in ops)
        dep_types = set(op.dep_on for op in ops if op.dep_on)

        # Workgroup spread
        workgroups = set(op.workgroup for op in ops)
        cross_wg = len(workgroups) > 1

        return {
            'n_threads': n_threads,
            'thread_seqs': dict(thread_seqs),
            'n_stores': n_stores,
            'n_loads': n_loads,
            'n_shared_addrs': n_shared_addrs,
            'has_fence': has_fence,
            'has_scope': has_scope,
            'has_dep': has_dep,
            'dep_types': dep_types,
            'cross_wg': cross_wg,
            'n_ops': len(non_fence),
        }

    def match(self, ops: List[MemOp], metadata: Dict) -> List[PatternMatch]:
        """Find best matching patterns for the given ops."""
        if not ops:
            return []

        code_sig = self._compute_signature(ops)
        matches = []

        for pat_name, pat_sig in self._pattern_signatures.items():
            score = self._similarity(code_sig, pat_sig, ops, PATTERNS[pat_name]['ops'])
            if score > 0.3:
                matched_ops = [(op.thread, op.optype, op.addr) for op in ops
                               if op.optype != 'fence']
                matches.append(PatternMatch(
                    pattern_name=pat_name,
                    confidence=score,
                    matched_ops=matched_ops,
                    code_lines=[],
                    description=PATTERNS[pat_name].get('description', ''),
                ))

        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches[:10]  # top 10 matches

    def _similarity(self, code_sig: Dict, pat_sig: Dict,
                    code_ops: List[MemOp], pat_ops: List[MemOp]) -> float:
        """Compute similarity score between code and pattern signatures."""
        score = 0.0
        max_score = 0.0

        # Thread count match (weight: 2)
        max_score += 2.0
        if code_sig['n_threads'] == pat_sig['n_threads']:
            score += 2.0
        elif abs(code_sig['n_threads'] - pat_sig['n_threads']) == 1:
            score += 1.0

        # Operation sequence similarity (weight: 3)
        max_score += 3.0
        seq_score = self._seq_similarity(code_sig['thread_seqs'], pat_sig['thread_seqs'])
        score += 3.0 * seq_score

        # Store/load count match (weight: 1.5)
        max_score += 1.5
        if (code_sig['n_stores'] == pat_sig['n_stores'] and
            code_sig['n_loads'] == pat_sig['n_loads']):
            score += 1.5
        elif abs(code_sig['n_stores'] - pat_sig['n_stores']) <= 1:
            score += 0.75

        # Fence presence (weight: 1)
        max_score += 1.0
        if code_sig['has_fence'] == pat_sig['has_fence']:
            score += 1.0

        # Scope presence for GPU (weight: 1)
        max_score += 1.0
        if code_sig['has_scope'] == pat_sig['has_scope']:
            score += 1.0
        elif not code_sig['has_scope'] and not pat_sig['has_scope']:
            score += 1.0

        # Cross-workgroup (weight: 0.5)
        max_score += 0.5
        if code_sig['cross_wg'] == pat_sig['cross_wg']:
            score += 0.5

        # Structural isomorphism check (weight: 3)
        max_score += 3.0
        iso_score = self._check_isomorphism(code_ops, pat_ops)
        score += 3.0 * iso_score

        return score / max_score if max_score > 0 else 0.0

    def _seq_similarity(self, seqs1: Dict, seqs2: Dict) -> float:
        """Compare per-thread operation sequences."""
        if not seqs1 or not seqs2:
            return 0.0

        # Normalize thread IDs
        sorted1 = sorted(seqs1.values(), key=lambda s: tuple(s))
        sorted2 = sorted(seqs2.values(), key=lambda s: tuple(s))

        if len(sorted1) != len(sorted2):
            # Try to match what we can
            min_len = min(len(sorted1), len(sorted2))
            sorted1 = sorted1[:min_len]
            sorted2 = sorted2[:min_len]

        total_sim = 0.0
        for s1, s2 in zip(sorted1, sorted2):
            if s1 == s2:
                total_sim += 1.0
            elif len(s1) == len(s2):
                matching = sum(1 for a, b in zip(s1, s2) if a == b)
                total_sim += matching / len(s1)
            else:
                # LCS-based similarity
                lcs = self._lcs_length(s1, s2)
                total_sim += 2.0 * lcs / (len(s1) + len(s2))

        return total_sim / max(len(sorted1), 1)

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

    def _check_isomorphism(self, code_ops: List[MemOp], pat_ops: List[MemOp]) -> float:
        """Check structural isomorphism between code ops and pattern ops."""
        code_nf = [op for op in code_ops if op.optype != 'fence']
        pat_nf = [op for op in pat_ops if op.optype != 'fence']

        if not code_nf or not pat_nf:
            return 0.0

        if len(code_nf) != len(pat_nf):
            # Partial match
            min_len = min(len(code_nf), len(pat_nf))
            code_nf = code_nf[:min_len]
            pat_nf = pat_nf[:min_len]

        # Try all thread ID and address mappings
        code_threads = sorted(set(op.thread for op in code_nf))
        pat_threads = sorted(set(op.thread for op in pat_nf))
        code_addrs = sorted(set(op.addr for op in code_nf))
        pat_addrs = sorted(set(op.addr for op in pat_nf))

        if len(code_threads) != len(pat_threads) or len(code_addrs) != len(pat_addrs):
            return 0.0

        # Simple mapping: match in order
        t_map = dict(zip(code_threads, pat_threads))
        a_map = dict(zip(code_addrs, pat_addrs))

        matches = 0
        for c_op, p_op in zip(code_nf, pat_nf):
            if (c_op.optype == p_op.optype and
                t_map.get(c_op.thread) == p_op.thread and
                a_map.get(c_op.addr) == p_op.addr):
                matches += 1

        return matches / len(pat_nf)


# ── Custom pattern support ──────────────────────────────────────────

def build_custom_pattern(ops: List[MemOp], forbidden: Dict[str, int],
                         name: str = "custom") -> LitmusTest:
    """Build a LitmusTest from user-provided MemOps and forbidden outcome."""
    n_threads = max(op.thread for op in ops) + 1
    addrs = sorted(set(op.addr for op in ops if op.addr))
    return LitmusTest(
        name=name,
        n_threads=n_threads,
        addresses=addrs,
        ops=ops,
        forbidden=forbidden,
    )


def check_custom_pattern(ops: List[MemOp], forbidden: Dict[str, int],
                         target_arch: str = None) -> List:
    """Check a custom (user-defined) pattern against architectures."""
    from portcheck import PortabilityResult
    test = build_custom_pattern(ops, forbidden)
    autos = compute_joint_automorphisms(test)
    total, n_orbits = compute_orbits(test, autos)

    targets = [target_arch] if target_arch else list(ARCHITECTURES.keys())
    results = []
    for arch in targets:
        model = ARCHITECTURES[arch]
        if not test.forbidden:
            safe = True
        else:
            forbidden_allowed, _ = verify_test(test, model)
            safe = not forbidden_allowed
        fence = None
        if not safe:
            fence = recommend_fence(test, arch, model)
        results.append(PortabilityResult(
            pattern='custom',
            source_arch='x86',
            target_arch=arch,
            safe=safe,
            forbidden_outcome=forbidden,
            fence_recommendation=fence,
            compression_ratio=total / n_orbits if n_orbits > 0 else 1.0,
            orbits_checked=n_orbits,
            total_outcomes=total,
        ))
    return results


# ── Main analyzer ───────────────────────────────────────────────────

class ConcurrencyAnalyzer:
    """High-level analyzer: code string → pattern matches → portability results."""

    def __init__(self):
        self.parser = CodeParser()
        self.matcher = PatternMatcher()

    def analyze_code(self, code: str, language: str = "auto") -> CodeAnalysisResult:
        """Analyze a code string and return matched patterns."""
        ops, metadata = self.parser.parse(code, language)

        patterns = self.matcher.match(ops, metadata)

        warnings = []
        if not ops:
            warnings.append("No memory operations detected in code.")
        if metadata['n_threads'] < 2 and not metadata['is_gpu']:
            warnings.append("Single-threaded code detected; concurrency patterns require 2+ threads.")
        if metadata['is_gpu'] and metadata.get('gpu_scope') == 'workgroup':
            warnings.append("Workgroup-scoped synchronization detected; check for scope mismatches.")

        code_hash = hex(hash(code) & 0xFFFFFFFF)

        return CodeAnalysisResult(
            code_hash=code_hash,
            patterns_found=patterns,
            extracted_ops=ops,
            n_threads=metadata['n_threads'],
            shared_vars=metadata['shared_vars'],
            has_fences=bool(metadata['fences']),
            is_gpu=metadata['is_gpu'],
            gpu_scope=metadata.get('gpu_scope'),
            warnings=warnings,
        )

    def check_code(self, code: str, target_arch: str = None,
                   language: str = "auto") -> List[Dict]:
        """Full pipeline: parse code → match patterns → check portability."""
        analysis = self.analyze_code(code, language)

        results = []
        for match in analysis.patterns_found[:5]:
            from api import check_portability as api_check
            if target_arch:
                port_result = api_check(match.pattern_name, 'x86', target_arch)
                results.append({
                    'pattern': match.pattern_name,
                    'confidence': match.confidence,
                    'target_arch': target_arch,
                    'safe': port_result.safe,
                    'fence_fix': port_result.fence_fix,
                    'explanation': port_result.explanation,
                })
            else:
                for arch in ARCHITECTURES:
                    port_result = api_check(match.pattern_name, 'x86', arch)
                    results.append({
                        'pattern': match.pattern_name,
                        'confidence': match.confidence,
                        'target_arch': arch,
                        'safe': port_result.safe,
                        'fence_fix': port_result.fence_fix,
                    })

        return results


# ── Convenience functions ───────────────────────────────────────────

_analyzer = None

def get_analyzer() -> ConcurrencyAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ConcurrencyAnalyzer()
    return _analyzer


def analyze_code(code: str, language: str = "auto") -> CodeAnalysisResult:
    """Analyze concurrent code and identify litmus test patterns."""
    return get_analyzer().analyze_code(code, language)


def check_code_portability(code: str, target_arch: str = None,
                           language: str = "auto") -> List[Dict]:
    """Check concurrent code for portability bugs."""
    return get_analyzer().check_code(code, target_arch, language)


# ── Real-world code snippets for case studies ───────────────────────

REAL_CODE_SNIPPETS = {
    "spsc_queue": {
        "description": "Lock-free SPSC queue (producer-consumer)",
        "language": "c",
        "expected_pattern": "mp",
        "port": "x86→arm",
        "code": """\
// Thread 0 (producer)
data = 1;
flag = 1;

// Thread 1 (consumer)
r0 = flag;
r1 = data;
""",
    },
    "dekker_mutex": {
        "description": "Dekker's mutual exclusion algorithm",
        "language": "c",
        "expected_pattern": "sb",
        "port": "x86→arm",
        "code": """\
// Thread 0
flag0 = 1;
r0 = flag1;

// Thread 1
flag1 = 1;
r1 = flag0;
""",
    },
    "seqlock_reader": {
        "description": "Seqlock reader pattern",
        "language": "c",
        "expected_pattern": "mp",
        "port": "x86→riscv",
        "code": """\
// Thread 0 (writer)
data = 1;
seq = 1;

// Thread 1 (reader)
r0 = seq;
r1 = data;
""",
    },
    "iriw_multicore": {
        "description": "Independent Reads of Independent Writes",
        "language": "c",
        "expected_pattern": "iriw",
        "port": "x86→arm",
        "code": """\
// Thread 0
x = 1;

// Thread 1
y = 1;

// Thread 2
r0 = x;
r1 = y;

// Thread 3
r2 = y;
r3 = x;
""",
    },
    "rcu_publish": {
        "description": "RCU publish pattern",
        "language": "c",
        "expected_pattern": "mp",
        "port": "x86→riscv",
        "code": """\
// Thread 0 (publisher)
data = 1;
ptr = 1;

// Thread 1 (reader)
r0 = ptr;
r1 = data;
""",
    },
    "cuda_scope_bug": {
        "description": "CUDA kernel with CTA-scoped fence (cross-CTA comm)",
        "language": "cuda",
        "expected_pattern": "gpu_barrier_scope_mismatch",
        "port": "cpu→gpu",
        "code": """\
__global__ void kernel() {
    // Thread 0 (block 0)
    data = 1;
    __threadfence_block();
    flag = 1;

    // Thread 1 (block 1)
    r0 = flag;
    __threadfence_block();
    r1 = data;
}
""",
    },
    "opencl_barrier_scope": {
        "description": "OpenCL kernel with workgroup barrier (cross-WG comm)",
        "language": "opencl",
        "expected_pattern": "gpu_barrier_scope_mismatch",
        "port": "cpu→gpu",
        "code": """\
// Thread 0 (workgroup 0)
store(x, 1);
barrier(CLK_LOCAL_MEM_FENCE);
store(y, 1);

// Thread 1 (workgroup 1)
r0 = load(y);
barrier(CLK_LOCAL_MEM_FENCE);
r1 = load(x);
""",
    },
    "riscv_asymmetric_fence": {
        "description": "RISC-V asymmetric fence pitfall (w,r instead of w,w)",
        "language": "c",
        "expected_pattern": "mp_fence_wr",
        "port": "arm→riscv",
        "code": """\
// Thread 0
x = 1;
fence w,r;
y = 1;

// Thread 1
r0 = y;
fence r,r;
r1 = x;
""",
    },
    "cpp_atomic_mp": {
        "description": "C++ std::atomic message passing with relaxed ordering",
        "language": "cpp",
        "expected_pattern": "mp",
        "port": "x86→arm",
        "code": """\
// Thread 0
data.store(42, std::memory_order_relaxed);
flag.store(1, std::memory_order_relaxed);

// Thread 1
r0 = flag.load(std::memory_order_relaxed);
r1 = data.load(std::memory_order_relaxed);
""",
    },
    "linux_spinlock": {
        "description": "Linux kernel spinlock pattern (test-and-set)",
        "language": "c",
        "expected_pattern": "sb",
        "port": "x86→arm",
        "code": """\
// Thread 0
lock = 1;
r0 = shared;

// Thread 1
shared = 1;
r1 = lock;
""",
    },
    "folly_mpmc_queue": {
        "description": "Folly MPMC queue turn publication (from Meta)",
        "language": "cpp",
        "expected_pattern": "mp",
        "port": "x86→arm",
        "code": """\
// Thread 0 (enqueue)
data.store(42, std::memory_order_relaxed);
turn.store(1, std::memory_order_relaxed);

// Thread 1 (dequeue)
r0 = turn.load(std::memory_order_relaxed);
r1 = data.load(std::memory_order_relaxed);
""",
    },
    "crossbeam_deque": {
        "description": "Crossbeam work-stealing deque (Rust, modeled as C)",
        "language": "c",
        "expected_pattern": "sb",
        "port": "x86→riscv",
        "code": """\
// Thread 0 (push)
buffer = 1;
r0 = top;

// Thread 1 (steal)
top = 1;
r1 = buffer;
""",
    },
}


def run_case_studies() -> List[Dict]:
    """Run the code analyzer on all real-world case study snippets."""
    analyzer = get_analyzer()
    results = []

    for name, snippet in REAL_CODE_SNIPPETS.items():
        analysis = analyzer.analyze_code(snippet['code'], snippet['language'])
        expected = snippet['expected_pattern']

        # Check if expected pattern was found in top matches
        matched_names = [m.pattern_name for m in analysis.patterns_found]
        found_expected = expected in matched_names
        rank = matched_names.index(expected) + 1 if found_expected else -1
        confidence = next(
            (m.confidence for m in analysis.patterns_found if m.pattern_name == expected),
            0.0)

        # Get portability result for expected pattern
        port_parts = snippet['port'].split('→')
        source = port_parts[0].strip()
        target = port_parts[1].strip() if len(port_parts) > 1 else 'arm'

        # Map source/target names
        arch_map = {'cpu': 'x86', 'gpu': 'ptx_cta'}
        source = arch_map.get(source, source)
        target = arch_map.get(target, target)

        from api import check_portability as api_check
        try:
            port_result = api_check(expected, source, target)
            safe = port_result.safe
            fix = port_result.fence_fix
        except Exception:
            safe = None
            fix = None

        results.append({
            'case_study': name,
            'description': snippet['description'],
            'expected_pattern': expected,
            'found_patterns': matched_names[:5],
            'matched_expected': found_expected,
            'match_rank': rank,
            'match_confidence': confidence,
            'port_from': source,
            'port_to': target,
            'has_bug': not safe if safe is not None else None,
            'fix': fix,
            'n_threads': analysis.n_threads,
            'n_ops': len(analysis.extracted_ops),
            'is_gpu': analysis.is_gpu,
        })

    return results


if __name__ == '__main__':
    import json

    print("=" * 70)
    print("LITMUS∞ Code-to-Pattern Analyzer — Case Study Results")
    print("=" * 70)

    results = run_case_studies()

    n_matched = sum(1 for r in results if r['matched_expected'])
    n_total = len(results)

    for r in results:
        status = "✓" if r['matched_expected'] else "✗"
        bug = "BUG" if r.get('has_bug') else "SAFE"
        print(f"\n{status} {r['case_study']}: {r['description']}")
        print(f"  Expected: {r['expected_pattern']}, Found: {r['found_patterns'][:3]}")
        print(f"  Rank: {r['match_rank']}, Confidence: {r['match_confidence']:.2f}")
        print(f"  Port {r['port_from']}→{r['port_to']}: {bug}")
        if r.get('fix'):
            print(f"  Fix: {r['fix']}")

    print(f"\n{'='*70}")
    print(f"Summary: {n_matched}/{n_total} case studies correctly matched ({100*n_matched/n_total:.0f}%)")

    # Save results
    import os
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_results')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'code_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to paper_results/code_analysis_results.json")
