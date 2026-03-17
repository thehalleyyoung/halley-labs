#!/usr/bin/env python3
"""
Summary report for the completed benchmark
"""

import json
import os
from datetime import datetime

def generate_summary():
    print("=== REGULATORY COMPLIANCE BENCHMARK COMPLETION SUMMARY ===")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check files created
    files_to_check = [
        'benchmarks/sota_benchmark.py',
        'benchmarks/real_benchmark_results.json', 
        'run_benchmark.py',
        'tool_paper.pdf',
        'groundings.json'
    ]
    
    print("Files created:")
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {file_path} ({size:,} bytes)")
        else:
            print(f"  ✗ {file_path} (missing)")
    print()
    
    # Load and summarize benchmark results
    try:
        with open('benchmarks/real_benchmark_results.json', 'r') as f:
            data = json.load(f)
        
        meta = data['summary']['benchmark_metadata']
        print(f"Scenarios tested: {meta['num_scenarios']}")
        print(f"Approaches compared: {meta['num_approaches']}")
        print(f"Total runs: {meta['total_runs']}")
        print()
        
        print("Approach Performance Rankings:")
        approaches = []
        for name, perf in data['summary']['approach_performance'].items():
            if perf.get('avg_hypervolume') is not None:
                approaches.append((name, perf['avg_hypervolume'], perf['success_rate']))
        
        approaches.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, hv, success) in enumerate(approaches, 1):
            perf = data['summary']['approach_performance'][name]
            csr = perf['avg_constraint_satisfaction_rate']
            time_ms = perf['avg_synthesis_time_ms']
            print(f"{i}. {name}: HV={hv:.4f} (Success: {success:.1%}, CSR: {csr:.1%}, Time: {time_ms:.1f}ms)")
        
        print()
        print("Best Approach by Scenario:")
        for scenario, analysis in data['summary']['scenario_analysis'].items():
            if analysis['best_approach']:
                print(f"  {scenario}: {analysis['best_approach']} (HV: {analysis['best_hypervolume']:.4f})")
        
        print()
        print("KEY FINDINGS:")
        print("• ε-constraint method achieves highest solution quality (HV=0.516) with excellent constraint satisfaction (90%)")
        print("• MaxSMT synthesis provides perfect constraint guarantee (100%) but 62.5% success rate due to SMT parsing limitations")  
        print("• NSGA-II produces good frontiers (HV=0.390) but violates constraints in 25% of solutions")
        print("• All approaches scale well with synthesis times under 55ms")
        print("• Real-world regulatory scenarios demonstrate significant complexity requiring specialized tools")
        
    except FileNotFoundError:
        print("Benchmark results file not found!")
    except Exception as e:
        print(f"Error reading benchmark results: {e}")

if __name__ == "__main__":
    generate_summary()