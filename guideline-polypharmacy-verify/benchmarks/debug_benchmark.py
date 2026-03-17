#!/usr/bin/env python3
"""Debug benchmark to understand GuardPharma behavior"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from sota_benchmark import *

def debug_scenario(scenario_id: int):
    """Debug a specific scenario to understand the safety assessment"""
    scenarios = create_benchmark_scenarios()
    scenario = scenarios[scenario_id - 1]
    
    print(f"🔍 Debugging Scenario {scenario_id}")
    print(f"Drugs: {[d.name for d in scenario.drugs]}")
    print(f"Doses: {scenario.doses} mg")
    print(f"Ground Truth: {'SAFE' if scenario.is_safe else 'UNSAFE'}")
    print(f"Interaction Type: {scenario.interaction_type}")
    print(f"Severity: {scenario.severity}")
    print()
    
    # Test GuardPharma
    guardpharma = GuardPharmaVerifier()
    is_safe, details = guardpharma.verify_safety(scenario)
    
    print(f"GuardPharma Prediction: {'SAFE' if is_safe else 'UNSAFE'}")
    print(f"Safety Violations: {details['safety_violations']}")
    print()
    
    print("Analysis Details:")
    print(f"  Toxic Violations: {len(details['toxic_violations'])}")
    for tv in details['toxic_violations']:
        print(f"    - {tv['drug']}: {tv['max_concentration']:.3f} > {tv['toxic_threshold']:.3f}")
    
    print(f"  Dangerous Combinations: {len(details['dangerous_combinations'])}")
    for dc in details['dangerous_combinations']:
        print(f"    - {dc['drugs']}: {dc['interaction_type']} ({dc['severity']})")
    
    print(f"  CYP Interactions: {len(details['cyp_interaction_violations'])}")
    for ci in details['cyp_interaction_violations']:
        print(f"    - {ci['inhibitor' if ci['type'] == 'inhibition' else 'inducer']} → {ci['substrate']}: {ci['enzyme']} {ci['type']} ({ci['severity']})")
    
    print(f"  Therapeutic Violations: {len(details['therapeutic_violations'])}")
    for tv in details['therapeutic_violations']:
        print(f"    - {tv['drug']}: {tv['steady_state_concentration']:.3f} outside {tv['therapeutic_range']}")
    
    print(f"  CYP Activities: {details['cyp_activities']}")
    print(f"  Significant CYP Changes: {details.get('significant_cyp_changes', False)}")
    print()

if __name__ == "__main__":
    # Debug first few scenarios
    for i in range(1, 6):
        debug_scenario(i)
        print("="*60)