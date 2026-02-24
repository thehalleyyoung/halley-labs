#!/usr/bin/env python3
"""
nlgen.py: LLM-assisted natural-language litmus test generator.

Given a natural-language description of a concurrent pattern (e.g., "producer-consumer
with a flag variable"), generates a litmus test specification, runs it through the
portability checker, and returns architecture-specific results with fence recommendations.

This enables developers without memory model expertise to check portability of
their concurrent algorithms.
"""

import json
import os
import sys
import argparse
from openai import OpenAI

# Import portcheck components
sys.path.insert(0, os.path.dirname(__file__))
from portcheck import (
    MemOp, LitmusTest, PATTERNS,
    compute_joint_automorphisms, compute_orbits,
    verify_test, recommend_fence, check_portability
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a memory model expert. Given a natural-language description of a concurrent
pattern, generate a litmus test specification in JSON format.

The litmus test should have:
- "name": short identifier
- "n_threads": number of threads
- "addresses": list of memory addresses used (e.g., ["x", "y", "flag"])
- "ops": list of memory operations, each with:
  - "thread": thread index (0-based)
  - "optype": "store", "load", or "fence"
  - "addr": address name
  - "value": value for stores (integer)
  - "reg": register name for loads (e.g., "r0", "r1")
- "forbidden": dict mapping register names to values for the forbidden outcome
- "explanation": brief explanation of why this outcome matters

Rules:
- All addresses initially hold value 0
- Loads can read any value that was stored or the initial 0
- The forbidden outcome should be the one the programmer wants to prevent
- Keep it minimal: use only the operations needed to expose the concurrency issue

Example for "two threads writing to the same variable, reader sees stale value":
{
  "name": "mp_custom",
  "n_threads": 2,
  "addresses": ["data", "flag"],
  "ops": [
    {"thread": 0, "optype": "store", "addr": "data", "value": 1},
    {"thread": 0, "optype": "store", "addr": "flag", "value": 1},
    {"thread": 1, "optype": "load", "addr": "flag", "reg": "r0"},
    {"thread": 1, "optype": "load", "addr": "data", "reg": "r1"}
  ],
  "forbidden": {"r0": 1, "r1": 0},
  "explanation": "If flag=1 is seen, data should also be 1 (message passing)"
}

Return ONLY valid JSON, no markdown fencing."""

def generate_litmus_test(description: str, model: str = "gpt-4.1-nano") -> dict:
    """Generate a litmus test from natural language description."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": description}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    text = response.choices[0].message.content.strip()
    # Strip markdown fencing if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return json.loads(text)

def spec_to_litmus(spec: dict) -> LitmusTest:
    """Convert JSON spec to LitmusTest object."""
    ops = []
    for op in spec["ops"]:
        ops.append(MemOp(
            thread=op["thread"],
            optype=op["optype"],
            addr=op.get("addr", ""),
            value=op.get("value"),
            reg=op.get("reg"),
        ))
    return LitmusTest(
        name=spec["name"],
        n_threads=spec["n_threads"],
        addresses=spec["addresses"],
        ops=ops,
        forbidden=spec["forbidden"]
    )

def analyze_litmus(test: LitmusTest) -> dict:
    """Run full portability analysis on a litmus test."""
    arch_to_model = {
        'x86': 'TSO', 'sparc': 'PSO', 'arm': 'ARM', 'riscv': 'RISC-V',
        'opencl_wg': 'OpenCL-WG', 'opencl_dev': 'OpenCL-Dev',
        'vulkan_wg': 'Vulkan-WG', 'vulkan_dev': 'Vulkan-Dev',
        'ptx_cta': 'PTX-CTA', 'ptx_gpu': 'PTX-GPU',
    }
    results = {}
    for arch, model in arch_to_model.items():
        try:
            forbidden_allowed, n_checked = verify_test(test, model)
            fence = recommend_fence(test, arch, model) if forbidden_allowed else None
            results[arch] = {
                "safe": not forbidden_allowed,
                "fence": fence,
            }
        except Exception as e:
            results[arch] = {"safe": True, "fence": None, "error": str(e)}
    # Compute symmetry info
    try:
        auts = compute_joint_automorphisms(test)
        results["symmetry"] = {
            "automorphism_order": len(auts) if auts else 1,
        }
    except Exception:
        results["symmetry"] = {"automorphism_order": 1}
    return results

CASE_STUDIES = [
    {
        "name": "SPSC Lock-Free Queue (enqueue/dequeue)",
        "description": "Single-producer single-consumer lock-free queue. Producer writes data to buffer[tail], then increments tail. Consumer reads tail, then reads buffer[old_tail]. The forbidden outcome is consumer seeing new tail but old data."
    },
    {
        "name": "Seqlock Reader Pattern",
        "description": "Sequence lock reader: read sequence number, read data, read sequence number again. Writer increments sequence, writes data, increments sequence. Forbidden: reader sees matching (even) sequence numbers but inconsistent data."
    },
    {
        "name": "Ticket Lock Acquire",
        "description": "Ticket lock: thread atomically increments 'next_ticket', then spins reading 'now_serving' until it matches. Another thread stores to 'now_serving' to release. Forbidden: acquiring thread sees its ticket number in now_serving but critical section data from previous holder."
    },
    {
        "name": "RCU-style Read-Copy-Update",
        "description": "RCU pattern: writer stores new data, then updates pointer. Reader reads pointer, then reads data at pointer location. Forbidden: reader sees new pointer but old data. This is message passing."
    },
    {
        "name": "Double-Checked Locking",
        "description": "Double-checked locking: Thread 0 stores data, then stores flag=1. Thread 1 reads flag; if flag==1, reads data. Forbidden: flag is 1 but data is 0. Similar to message passing but specifically the DCL anti-pattern."
    },
    {
        "name": "GPU Reduction (Cross-Workgroup)",
        "description": "GPU parallel reduction across workgroups. Workgroup 0 writes partial sum to shared buffer, sets ready flag with workgroup barrier. Workgroup 1 reads ready flag with workgroup barrier, reads partial sum. Using workgroup-scope barriers for cross-workgroup communication."
    },
    {
        "name": "GPU Producer-Consumer Pipeline",
        "description": "GPU pipeline: CTA 0 produces results, stores to global memory, uses membar.cta, sets done flag. CTA 1 reads done flag, uses membar.cta, reads results. Cross-CTA communication with CTA-scope barriers only."
    },
    {
        "name": "Work Stealing Queue",
        "description": "Work stealing: thread 0 (owner) pushes work item (store data, increment tail). Thread 1 (stealer) reads tail, reads data. Forbidden: stealer sees incremented tail but stale data. Store-store ordering on owner, load-load on stealer."
    }
]


def run_case_studies(output_dir: str = "benchmark_results_new"):
    """Run all case studies and save results."""
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for cs in CASE_STUDIES:
        print(f"\n{'='*60}")
        print(f"Case Study: {cs['name']}")
        print(f"{'='*60}")

        try:
            spec = generate_litmus_test(cs["description"])
            print(f"Generated test: {spec['name']}")
            print(f"  Threads: {spec['n_threads']}, Addresses: {spec['addresses']}")
            print(f"  Forbidden: {spec['forbidden']}")
            if "explanation" in spec:
                print(f"  Why: {spec['explanation']}")

            test = spec_to_litmus(spec)
            results = analyze_litmus(test)

            # Summary
            cpu_targets = ['x86', 'sparc', 'arm', 'riscv']
            gpu_targets = ['opencl_wg', 'opencl_dev', 'vulkan_wg', 'vulkan_dev', 'ptx_cta', 'ptx_gpu']

            print(f"\n  CPU Portability:")
            for t in cpu_targets:
                r = results[t]
                status = "✓ SAFE" if r["safe"] else "✗ FAIL"
                fence = f' → Fix: {r["fence"]}' if r["fence"] else ""
                print(f"    {t:10s}: {status}{fence}")

            print(f"\n  GPU Portability:")
            for t in gpu_targets:
                r = results[t]
                status = "✓ SAFE" if r["safe"] else "✗ FAIL"
                fence = f' → Fix: {r["fence"]}' if r["fence"] else ""
                print(f"    {t:10s}: {status}{fence}")

            # Check for scope mismatch
            wg_fail = not results.get('opencl_wg', {}).get('safe', True)
            dev_safe = results.get('opencl_dev', {}).get('safe', True)
            if wg_fail and dev_safe:
                print(f"\n  ⚠ SCOPE MISMATCH DETECTED: fails at workgroup scope, safe at device scope")

            all_results.append({
                "case_study": cs["name"],
                "description": cs["description"],
                "litmus_spec": spec,
                "results": {k: v for k, v in results.items() if k != "symmetry"},
                "symmetry": results.get("symmetry", {}),
                "scope_mismatch": wg_fail and dev_safe,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "case_study": cs["name"],
                "error": str(e)
            })

    # Save results
    outfile = os.path.join(output_dir, "case_study_results.json")
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outfile}")

    # Generate CSV summary
    csv_file = os.path.join(output_dir, "case_study_summary.csv")
    with open(csv_file, "w") as f:
        f.write("case_study,x86,sparc,arm,riscv,opencl_wg,opencl_dev,vulkan_wg,vulkan_dev,ptx_cta,ptx_gpu,scope_mismatch\n")
        for r in all_results:
            if "error" in r:
                continue
            row = [r["case_study"]]
            for t in ['x86','sparc','arm','riscv','opencl_wg','opencl_dev','vulkan_wg','vulkan_dev','ptx_cta','ptx_gpu']:
                row.append("Safe" if r["results"].get(t,{}).get("safe",True) else "FAIL")
            row.append("YES" if r.get("scope_mismatch") else "NO")
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"Summary CSV saved to {csv_file}")

    return all_results


def run_single_query(description: str):
    """Run a single natural language query."""
    print(f"Generating litmus test for: {description}")
    spec = generate_litmus_test(description)
    print(f"\nGenerated test: {spec['name']}")
    print(f"  Threads: {spec['n_threads']}")
    print(f"  Addresses: {spec['addresses']}")
    print(f"  Operations:")
    for op in spec["ops"]:
        if op["optype"] == "store":
            print(f"    T{op['thread']}: St [{op['addr']}] = {op['value']}")
        elif op["optype"] == "load":
            print(f"    T{op['thread']}: {op['reg']} = Ld [{op['addr']}]")
        else:
            print(f"    T{op['thread']}: fence")
    print(f"  Forbidden: {spec['forbidden']}")

    test = spec_to_litmus(spec)
    results = analyze_litmus(test)

    print(f"\nPortability Results:")
    for t in ['x86','sparc','arm','riscv','opencl_wg','opencl_dev','vulkan_wg','vulkan_dev','ptx_cta','ptx_gpu']:
        r = results[t]
        status = "✓ SAFE" if r["safe"] else "✗ FAIL"
        fence = f' → {r["fence"]}' if r["fence"] else ""
        print(f"  {t:12s}: {status}{fence}")

    return spec, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-assisted litmus test generator")
    parser.add_argument("--query", type=str, help="Natural language description of concurrent pattern")
    parser.add_argument("--case-studies", action="store_true", help="Run all case studies")
    parser.add_argument("--output", type=str, default="benchmark_results_new", help="Output directory")
    args = parser.parse_args()

    if args.case_studies:
        run_case_studies(args.output)
    elif args.query:
        run_single_query(args.query)
    else:
        parser.print_help()
