#!/usr/bin/env python3
"""Run the bundled SOTA benchmark and save results inside this checkout."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = REPO_ROOT / "benchmarks"
sys.path.append(str(BENCHMARK_DIR))

from sota_benchmark import main

def run_and_save_benchmark():
    """Run benchmark and save results"""
    print("Running SOTA benchmark and saving results...")
    
    # Run the benchmark
    results = main()
    
    # Save to JSON file
    output_path = BENCHMARK_DIR / "real_benchmark_results.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary statistics
    summary = results['summary']
    print("\nBENCHMARK SUMMARY")
    print("=" * 50)
    
    print(f"Total scenarios: {summary['benchmark_metadata']['num_scenarios']}")
    print(f"Total approaches: {summary['benchmark_metadata']['num_approaches']}")
    print(f"Total runs: {summary['benchmark_metadata']['total_runs']}")
    
    print("\nApproach Rankings (by average hypervolume):")
    approach_hv = [(name, perf.get('avg_hypervolume', 0.0)) 
                   for name, perf in summary['approach_performance'].items()
                   if perf.get('avg_hypervolume') is not None]
    approach_hv.sort(key=lambda x: x[1], reverse=True)
    
    for i, (approach, hv) in enumerate(approach_hv, 1):
        perf = summary['approach_performance'][approach]
        print(f"{i}. {approach}: HV={hv:.4f} (Success: {perf['success_rate']:.1%}, "
              f"Time: {perf['avg_synthesis_time_ms']:.1f}ms)")
    
    return results

if __name__ == "__main__":
    run_and_save_benchmark()
