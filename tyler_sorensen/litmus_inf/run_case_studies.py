#!/usr/bin/env python3
"""
LLM-assisted litmus test generation case study for LITMUS∞.
Uses GPT-4.1-nano to generate litmus test descriptions, then validates them.
"""

import json
import os
import sys
import time

RESULTS_DIR = "paper_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

from portcheck import PATTERNS, check_portability

# Real-world concurrency scenarios to test
CASE_STUDIES = [
    {
        "name": "lock_free_queue_x86_to_arm",
        "description": "Lock-free SPSC queue using store-release/load-acquire, ported from x86 to ARM",
        "pattern": "mp",
        "source_arch": "x86",
        "target_arch": "arm",
        "context": "A producer stores data then sets a flag. Consumer reads flag then data. On x86 TSO, the store ordering is guaranteed. On ARM, without dmb, the consumer may see the flag before the data.",
    },
    {
        "name": "dekker_mutex_arm",
        "description": "Dekker's mutual exclusion algorithm on ARM",
        "pattern": "dekker",
        "source_arch": "x86",
        "target_arch": "arm",
        "context": "Dekker's algorithm relies on store-buffering visibility. On x86, only SB reordering breaks it. On ARM, all reorderings are possible, requiring full barriers.",
    },
    {
        "name": "seqlock_reader_riscv",
        "description": "Seqlock reader pattern ported to RISC-V",
        "pattern": "mp",
        "source_arch": "x86",
        "target_arch": "riscv",
        "context": "Seqlock readers observe a sequence number then read data. The MP pattern captures the key ordering requirement. RISC-V RVWMO may reorder both stores and loads.",
    },
    {
        "name": "cuda_kernel_wg_scope",
        "description": "CUDA kernel with workgroup-scoped synchronization across CTAs",
        "pattern": "gpu_mp_scope_mismatch_dev",
        "source_arch": "ptx_gpu",
        "target_arch": "ptx_cta",
        "context": "A CUDA kernel uses __threadfence_block() (CTA scope) but threads communicate across CTAs. This is a scope mismatch: CTA-scoped fences don't order cross-CTA communication.",
    },
    {
        "name": "iriw_multicore_arm",
        "description": "IRIW pattern in multi-core ARM system",
        "pattern": "iriw",
        "source_arch": "x86",
        "target_arch": "arm",
        "context": "Two independent writers and two readers. On x86 (multi-copy atomic), both readers must agree on write order. On ARM (not multi-copy atomic), readers may disagree. This is observable on real Cortex-A53/A57 hardware.",
    },
    {
        "name": "rcu_publish_riscv",
        "description": "RCU-style publish pattern on RISC-V",
        "pattern": "mp",
        "source_arch": "x86",
        "target_arch": "riscv",
        "context": "Linux RCU uses rcu_assign_pointer (store with release semantics) and rcu_dereference (load with consume/address dependency). The MP pattern captures this, and RISC-V RVWMO requires explicit fences.",
    },
    {
        "name": "opencl_gpu_barrier_scope",
        "description": "OpenCL kernel barrier with wrong scope across workgroups",
        "pattern": "gpu_barrier_scope_mismatch",
        "source_arch": "opencl_dev",
        "target_arch": "opencl_wg",
        "context": "An OpenCL kernel uses barrier(CLK_LOCAL_MEM_FENCE) but accesses global memory across workgroups. The workgroup-scoped barrier does not synchronize global memory across workgroups.",
    },
    {
        "name": "asymmetric_fence_riscv",
        "description": "RISC-V asymmetric fence (fence w,r) insufficient for MP",
        "pattern": "mp_fence_wr",
        "source_arch": "arm",
        "target_arch": "riscv",
        "context": "ARM's dmb covers all orderings symmetrically. RISC-V fence w,r only orders stores-before-reads. For the MP pattern, the reader thread needs fence r,r to order its loads, but fence w,r on the reader only orders writes-before-reads, missing the load-load ordering.",
    },
]


def run_case_studies():
    """Run all case studies and validate with portcheck."""
    print("="*80)
    print("CASE STUDY: Real-World Portability Scenarios")
    print("="*80)

    results = []
    for cs in CASE_STUDIES:
        print(f"\n  Case: {cs['name']}")
        print(f"    Scenario: {cs['description']}")

        # Run portcheck
        port_results = check_portability(cs["pattern"], target_arch=cs["target_arch"])
        if not port_results:
            print(f"    ERROR: Pattern {cs['pattern']} not found")
            continue
        r = port_results[0]

        # Also get all-arch view
        all_results = check_portability(cs["pattern"])
        arch_summary = {ar.target_arch: "Safe" if ar.safe else "Allowed" for ar in all_results}

        result = {
            "name": cs["name"],
            "description": cs["description"],
            "pattern": cs["pattern"],
            "source_arch": cs["source_arch"],
            "target_arch": cs["target_arch"],
            "safe": r.safe,
            "fence_recommendation": r.fence_recommendation,
            "context": cs["context"],
            "all_arch_results": arch_summary,
        }
        results.append(result)

        status = "✓ SAFE" if r.safe else "✗ BUG DETECTED"
        print(f"    Result: {status}")
        if r.fence_recommendation:
            print(f"    Fix: {r.fence_recommendation}")

    # Try LLM-based generation if API key available
    llm_results = try_llm_generation()

    all_data = {
        "case_studies": results,
        "llm_generated": llm_results,
        "summary": {
            "total_cases": len(results),
            "bugs_found": sum(1 for r in results if not r["safe"]),
            "safe": sum(1 for r in results if r["safe"]),
        }
    }

    with open(f"{RESULTS_DIR}/case_study_results.json", "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"\n  Summary: {all_data['summary']['bugs_found']}/{all_data['summary']['total_cases']} "
          f"portability bugs detected")
    return all_data


def try_llm_generation():
    """Use LLM to generate novel litmus test descriptions and validate them."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  Skipping LLM generation (no OPENAI_API_KEY)")
        return []

    try:
        import urllib.request
        import urllib.error
    except ImportError:
        return []

    prompt = """You are a memory model expert. Generate 5 novel concurrency bug scenarios that 
would be caught by a cross-architecture portability checker. For each scenario, provide:
1. A short name
2. The litmus test pattern it maps to (one of: mp, sb, lb, iriw, wrc, 2+2w)
3. Source architecture (x86)
4. Target architecture (arm or riscv)
5. A 2-sentence description of the real-world scenario

Format as JSON array of objects with keys: name, pattern, source, target, description.
Focus on realistic scenarios from systems programming (kernels, databases, lock-free structures).
Return ONLY the JSON array, no markdown."""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = json.dumps({
        "model": "gpt-4.1-nano",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000,
    }).encode()

    try:
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
                                     data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())

        content = result["choices"][0]["message"]["content"]
        # Parse JSON from response
        scenarios = json.loads(content)

        llm_results = []
        for s in scenarios:
            pname = s.get("pattern", "mp").lower()
            target = s.get("target", "arm").lower()
            if pname not in PATTERNS:
                pname = "mp"  # fallback

            port_results = check_portability(pname, target_arch=target)
            if port_results:
                r = port_results[0]
                llm_results.append({
                    "name": s.get("name", "unknown"),
                    "pattern": pname,
                    "target": target,
                    "description": s.get("description", ""),
                    "safe": r.safe,
                    "fence_recommendation": r.fence_recommendation,
                    "llm_generated": True,
                })
                status = "Safe" if r.safe else "BUG"
                print(f"    LLM scenario '{s.get('name', '?')}': {status}")

        return llm_results

    except Exception as e:
        print(f"  LLM generation failed: {e}")
        return []


if __name__ == "__main__":
    run_case_studies()
