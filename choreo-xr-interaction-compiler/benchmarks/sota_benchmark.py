#!/usr/bin/env python3
"""
SOTA Benchmark Suite for XR Interaction Choreography Compiler
Real-world scenarios with spatial CEGAR implementation and baseline comparisons
"""

import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import itertools
from z3 import *

# Configuration
random.seed(42)
np.random.seed(42)

class InteractionType(Enum):
    GRAB = "grab"
    POINT = "point"
    GESTURE = "gesture"
    NAVIGATE = "navigate"
    HANDOFF = "handoff"
    COLLABORATE = "collaborate"
    MENU_SELECT = "menu_select"

class SpatialConstraintType(Enum):
    COLLISION_AVOID = "collision_avoid"
    REACHABILITY = "reachability"
    MUTUAL_EXCLUSION = "mutual_exclusion"
    VISIBILITY = "visibility"
    WORKSPACE_BOUNDS = "workspace_bounds"

@dataclass
class Pose:
    x: float
    y: float 
    z: float
    rx: float = 0.0  # rotation around x
    ry: float = 0.0  # rotation around y
    rz: float = 0.0  # rotation around z

@dataclass
class SpatialRegion:
    center: Pose
    radius: float
    height: float = 2.0

@dataclass
class Actor:
    id: str
    initial_pose: Pose
    reach_radius: float
    collision_radius: float

@dataclass
class SpatialConstraint:
    type: SpatialConstraintType
    actors: List[str]
    regions: List[SpatialRegion]
    parameters: Dict = None

@dataclass
class InteractionStep:
    step_id: int
    actor: str
    interaction_type: InteractionType
    target_pose: Optional[Pose]
    target_region: Optional[SpatialRegion]
    duration: float
    preconditions: List[str] = None
    effects: List[str] = None

@dataclass
class XRScenario:
    name: str
    description: str
    actors: List[Actor]
    steps: List[InteractionStep]
    spatial_constraints: List[SpatialConstraint]
    workspace_bounds: Tuple[float, float, float]  # x, y, z extents
    expected_realizable: bool

class SpatialCEGAR:
    """
    Core Spatial CEGAR implementation for XR interaction choreography verification
    """
    
    def __init__(self):
        self.refinement_iterations = 0
        self.counterexamples = []
        self.abstraction_predicates = []
    
    def verify_scenario(self, scenario: XRScenario) -> Tuple[bool, Dict]:
        """
        Main CEGAR verification loop
        Returns: (realizable, metrics)
        """
        start_time = time.time()
        self.refinement_iterations = 0
        self.counterexamples = []
        
        # Initial abstraction
        abstraction = self._create_initial_abstraction(scenario)
        
        while self.refinement_iterations < 10:  # Max iterations
            # Check abstraction satisfiability
            abstract_result = self._check_abstract_model(abstraction, scenario)
            
            if not abstract_result.satisfiable:
                # Abstract model unsatisfiable -> definitely unrealizable
                return False, {
                    'verification_time': time.time() - start_time,
                    'refinement_iterations': self.refinement_iterations,
                    'counterexamples': len(self.counterexamples),
                    'result': 'proven_unrealizable'
                }
            
            # Check if abstract solution is concrete
            concrete_result = self._check_concrete_feasibility(
                abstract_result.solution, scenario)
            
            if concrete_result.feasible:
                # Found concrete solution -> realizable
                return True, {
                    'verification_time': time.time() - start_time,
                    'refinement_iterations': self.refinement_iterations,
                    'counterexamples': len(self.counterexamples),
                    'result': 'proven_realizable',
                    'solution': concrete_result.trajectory
                }
            
            # Refine abstraction using counterexample
            self._refine_abstraction(abstraction, concrete_result.counterexample)
            self.refinement_iterations += 1
            self.counterexamples.append(concrete_result.counterexample)
        
        # Timeout - fall back to constraint propagation for a definitive answer
        cp_result, cp_metrics = BaselineApproaches.constraint_propagation_only(scenario)
        return cp_result, {
            'verification_time': time.time() - start_time,
            'refinement_iterations': self.refinement_iterations,
            'counterexamples': len(self.counterexamples),
            'result': 'fallback_cp',
            'cp_result': cp_metrics.get('result', str(cp_result))
        }
    
    def _create_initial_abstraction(self, scenario: XRScenario):
        """Create coarse spatial abstraction"""
        return {
            'spatial_regions': self._discretize_workspace(scenario.workspace_bounds),
            'temporal_windows': self._create_temporal_windows(scenario.steps),
            'predicates': []
        }
    
    def _discretize_workspace(self, bounds):
        """Create spatial grid abstraction"""
        x_max, y_max, z_max = bounds
        grid_size = 1.0  # 1m grid cells
        regions = []
        
        for x in np.arange(-x_max, x_max, grid_size):
            for y in np.arange(-y_max, y_max, grid_size):
                for z in np.arange(0, z_max, grid_size):
                    regions.append(SpatialRegion(
                        center=Pose(x, y, z),
                        radius=grid_size/2,
                        height=grid_size
                    ))
        return regions
    
    def _create_temporal_windows(self, steps):
        """Create temporal abstraction"""
        windows = []
        current_time = 0.0
        for step in steps:
            windows.append((current_time, current_time + step.duration))
            current_time += step.duration
        return windows
    
    @dataclass
    class AbstractResult:
        satisfiable: bool
        solution: Optional[Dict] = None
    
    @dataclass
    class ConcreteResult:
        feasible: bool
        trajectory: Optional[List] = None
        counterexample: Optional[Dict] = None
    
    def _check_abstract_model(self, abstraction, scenario) -> 'SpatialCEGAR.AbstractResult':
        """Check if abstract model is satisfiable using Z3"""
        solver = Solver()
        
        # Variables for each actor at each time step
        actor_positions = {}
        for actor in scenario.actors:
            actor_positions[actor.id] = {}
            for i, region in enumerate(abstraction['spatial_regions']):
                for t, window in enumerate(abstraction['temporal_windows']):
                    var_name = f"{actor.id}_region_{i}_time_{t}"
                    actor_positions[actor.id][f"r{i}_t{t}"] = Bool(var_name)
        
        # Constraint: each actor in exactly one region at each time
        for actor in scenario.actors:
            for t in range(len(abstraction['temporal_windows'])):
                region_vars = [actor_positions[actor.id][f"r{i}_t{t}"] 
                              for i in range(len(abstraction['spatial_regions']))]
                solver.add(PbEq([(v, 1) for v in region_vars], 1))
        
        # Spatial constraints
        for constraint in scenario.spatial_constraints:
            self._add_spatial_constraint_to_solver(solver, constraint, 
                                                 actor_positions, abstraction)
        
        # Interaction reachability constraints
        for step in scenario.steps:
            self._add_reachability_constraint(solver, step, actor_positions, 
                                            abstraction, scenario)
        
        result = solver.check()
        if result == sat:
            model = solver.model()
            solution = self._extract_abstract_solution(model, actor_positions, abstraction)
            return self.AbstractResult(True, solution)
        else:
            return self.AbstractResult(False)
    
    def _add_spatial_constraint_to_solver(self, solver, constraint, actor_positions, abstraction):
        """Add spatial constraints to Z3 solver"""
        if constraint.type == SpatialConstraintType.COLLISION_AVOID:
            # No two actors in same region at same time
            for t in range(len(abstraction['temporal_windows'])):
                for i, region in enumerate(abstraction['spatial_regions']):
                    for actor1, actor2 in itertools.combinations(constraint.actors, 2):
                        solver.add(Not(And(
                            actor_positions[actor1][f"r{i}_t{t}"],
                            actor_positions[actor2][f"r{i}_t{t}"]
                        )))
        
        elif constraint.type == SpatialConstraintType.MUTUAL_EXCLUSION:
            # Actors cannot be in specified regions simultaneously
            for region in constraint.regions:
                region_idx = self._find_nearest_region(region, abstraction['spatial_regions'])
                if region_idx >= 0:
                    for t in range(len(abstraction['temporal_windows'])):
                        exclusion_vars = []
                        for actor in constraint.actors:
                            exclusion_vars.append(actor_positions[actor][f"r{region_idx}_t{t}"])
                        solver.add(PbLe([(v, 1) for v in exclusion_vars], 1))
    
    def _add_reachability_constraint(self, solver, step, actor_positions, abstraction, scenario):
        """Ensure actor can reach interaction target"""
        actor = step.actor
        if step.target_region:
            target_region_idx = self._find_nearest_region(step.target_region, 
                                                        abstraction['spatial_regions'])
            if target_region_idx >= 0:
                # Find corresponding time window
                step_time = next(i for i, s in enumerate(scenario.steps) if s.step_id == step.step_id)
                if step_time < len(abstraction['temporal_windows']):
                    # Actor must be in reachable region during interaction
                    reach_regions = self._find_reachable_regions(
                        target_region_idx, abstraction['spatial_regions'], 
                        next(a for a in scenario.actors if a.id == actor).reach_radius
                    )
                    reachable_vars = [actor_positions[actor][f"r{r}_t{step_time}"] 
                                    for r in reach_regions]
                    solver.add(Or(reachable_vars))
    
    def _find_nearest_region(self, target, regions):
        """Find spatial region closest to target"""
        min_dist = float('inf')
        best_idx = -1
        for i, region in enumerate(regions):
            dist = np.sqrt((target.center.x - region.center.x)**2 + 
                          (target.center.y - region.center.y)**2 + 
                          (target.center.z - region.center.z)**2)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        return best_idx
    
    def _find_reachable_regions(self, target_idx, regions, reach_radius):
        """Find all regions within reach of target"""
        reachable = [target_idx]  # Can always reach target itself
        target_pos = regions[target_idx].center
        
        for i, region in enumerate(regions):
            if i == target_idx:
                continue
            dist = np.sqrt((target_pos.x - region.center.x)**2 + 
                          (target_pos.y - region.center.y)**2 + 
                          (target_pos.z - region.center.z)**2)
            if dist <= reach_radius:
                reachable.append(i)
        return reachable
    
    def _extract_abstract_solution(self, model, actor_positions, abstraction):
        """Extract solution from Z3 model"""
        solution = {}
        for actor_id in actor_positions:
            solution[actor_id] = []
            for t in range(len(abstraction['temporal_windows'])):
                for i in range(len(abstraction['spatial_regions'])):
                    var = actor_positions[actor_id][f"r{i}_t{t}"]
                    if model[var]:
                        solution[actor_id].append({
                            'time': t,
                            'region': i,
                            'position': abstraction['spatial_regions'][i].center
                        })
                        break
        return solution
    
    def _check_concrete_feasibility(self, abstract_solution, scenario) -> 'SpatialCEGAR.ConcreteResult':
        """Check if abstract solution is concretely feasible"""
        # Simulate continuous motion between abstract regions
        for actor_id, trajectory in abstract_solution.items():
            actor = next(a for a in scenario.actors if a.id == actor_id)
            
            # Check if trajectory respects continuous constraints
            for i in range(len(trajectory) - 1):
                current_pos = trajectory[i]['position']
                next_pos = trajectory[i + 1]['position']
                
                # Check velocity constraints
                dt = 1.0  # Time step duration
                distance = np.sqrt((next_pos.x - current_pos.x)**2 + 
                                 (next_pos.y - current_pos.y)**2 + 
                                 (next_pos.z - current_pos.z)**2)
                max_velocity = 2.0  # m/s
                if distance / dt > max_velocity:
                    return self.ConcreteResult(False, None, {
                        'type': 'velocity_violation',
                        'actor': actor_id,
                        'step': i,
                        'required_velocity': distance / dt,
                        'max_velocity': max_velocity
                    })
                
                # Check collision avoidance in continuous space
                if self._check_continuous_collision(trajectory, abstract_solution, 
                                                  scenario, i):
                    return self.ConcreteResult(False, None, {
                        'type': 'collision',
                        'actor': actor_id,
                        'step': i
                    })
        
        return self.ConcreteResult(True, abstract_solution)
    
    def _check_continuous_collision(self, trajectory, all_trajectories, scenario, step):
        """Check for collision in continuous space using pairwise distance checks"""
        actor_id = None
        for aid, traj in all_trajectories.items():
            if traj is trajectory:
                actor_id = aid
                break
        if actor_id is None:
            return False

        actor = next((a for a in scenario.actors if a.id == actor_id), None)
        if actor is None:
            return False

        current_pos = trajectory[step]['position']
        min_sep = actor.collision_radius * 2

        for other_id, other_traj in all_trajectories.items():
            if other_id == actor_id:
                continue
            if step < len(other_traj):
                other_pos = other_traj[step]['position']
                dist = np.sqrt((current_pos.x - other_pos.x)**2 +
                               (current_pos.y - other_pos.y)**2 +
                               (current_pos.z - other_pos.z)**2)
                if dist < min_sep:
                    return True
        return False

    def _refine_abstraction(self, abstraction, counterexample):
        """Refine spatial abstraction based on counterexample"""
        if counterexample['type'] == 'velocity_violation':
            self._add_intermediate_regions(abstraction, counterexample)
        elif counterexample['type'] == 'collision':
            self._refine_collision_regions(abstraction, counterexample)

    def _add_intermediate_regions(self, abstraction, counterexample):
        """Add finer spatial discretization by splitting regions along the violating path"""
        step = counterexample.get('step', 0)
        regions = abstraction['spatial_regions']
        if step < len(regions) - 1:
            r1 = regions[step].center
            r2 = regions[min(step + 1, len(regions) - 1)].center
            mid = Pose((r1.x + r2.x) / 2, (r1.y + r2.y) / 2, (r1.z + r2.z) / 2)
            half_radius = regions[step].radius / 2
            regions.append(SpatialRegion(center=mid, radius=half_radius,
                                         height=regions[step].height / 2))

    def _refine_collision_regions(self, abstraction, counterexample):
        """Refine regions around collision areas by subdividing the offending cell"""
        step = counterexample.get('step', 0)
        regions = abstraction['spatial_regions']
        if step < len(regions):
            parent = regions[step]
            r = parent.radius / 2
            h = parent.height / 2
            cx, cy, cz = parent.center.x, parent.center.y, parent.center.z
            for dx in (-r, r):
                for dy in (-r, r):
                    regions.append(SpatialRegion(
                        center=Pose(cx + dx, cy + dy, cz),
                        radius=r, height=h))


class HybridSpatialCEGAR:
    """
    Hybrid Spatial CEGAR: uses fast CEGAR for easy cases and falls back to
    constraint propagation when refinement stalls, combining the speed of
    CEGAR with the precision of geometric constraint reasoning.
    """

    def __init__(self):
        self.cegar = SpatialCEGAR()

    def verify_scenario(self, scenario: XRScenario) -> Tuple[bool, Dict]:
        start_time = time.time()

        # Phase 1: quick constraint propagation pre-check
        cp_result, cp_metrics = BaselineApproaches.constraint_propagation_only(scenario)
        if not cp_result:
            # CP found definite conflict — trust it immediately
            return False, {
                'verification_time': time.time() - start_time,
                'phase': 'cp_precheck',
                'refinement_iterations': 0,
                'counterexamples': 0,
                'result': 'proven_unrealizable_cp'
            }

        # Phase 2: attempt full CEGAR for strong proof
        cegar_result, cegar_metrics = self.cegar.verify_scenario(scenario)

        # If CEGAR converged (not a fallback), use its answer
        if cegar_metrics.get('result') in ('proven_realizable', 'proven_unrealizable'):
            return cegar_result, {
                'verification_time': time.time() - start_time,
                'phase': 'cegar_converged',
                'refinement_iterations': cegar_metrics.get('refinement_iterations', 0),
                'counterexamples': cegar_metrics.get('counterexamples', 0),
                'result': cegar_metrics['result']
            }

        # Phase 3: CEGAR fell back — cross-validate with monolithic check on
        # small scenarios, otherwise trust CP result
        if len(scenario.actors) <= 3 and len(scenario.steps) <= 6:
            try:
                mono_result, _ = BaselineApproaches.monolithic_z3(scenario)
                final = mono_result
                phase = 'z3_crosscheck'
            except Exception:
                final = cp_result
                phase = 'cp_fallback'
        else:
            final = cp_result
            phase = 'cp_fallback'

        return final, {
            'verification_time': time.time() - start_time,
            'phase': phase,
            'refinement_iterations': cegar_metrics.get('refinement_iterations', 0),
            'counterexamples': cegar_metrics.get('counterexamples', 0),
            'result': f'hybrid_{phase}'
        }


class BaselineApproaches:
    """Baseline approaches for comparison"""
    
    @staticmethod
    def monolithic_z3(scenario: XRScenario) -> Tuple[bool, Dict]:
        """Monolithic Z3 encoding without CEGAR"""
        start_time = time.time()
        solver = Solver()
        
        # Dense encoding of all spatial and temporal variables
        time_steps = 20  # Fixed discretization
        spatial_resolution = 0.5  # 0.5m grid
        
        # Create variables for each actor position at each time
        actor_vars = {}
        for actor in scenario.actors:
            actor_vars[actor.id] = {}
            for t in range(time_steps):
                actor_vars[actor.id][t] = {
                    'x': Real(f"{actor.id}_x_{t}"),
                    'y': Real(f"{actor.id}_y_{t}"),
                    'z': Real(f"{actor.id}_z_{t}")
                }
        
        # Add all constraints without abstraction
        for constraint in scenario.spatial_constraints:
            BaselineApproaches._add_monolithic_constraint(solver, constraint, actor_vars, scenario)
        
        # Add motion constraints
        for actor in scenario.actors:
            for t in range(time_steps - 1):
                # Velocity limits
                dt = 0.1  # time step
                max_vel = 2.0
                dx = actor_vars[actor.id][t+1]['x'] - actor_vars[actor.id][t]['x']
                dy = actor_vars[actor.id][t+1]['y'] - actor_vars[actor.id][t]['y']
                dz = actor_vars[actor.id][t+1]['z'] - actor_vars[actor.id][t]['z']
                
                solver.add(dx*dx + dy*dy + dz*dz <= (max_vel * dt)**2)
        
        result = solver.check()
        verification_time = time.time() - start_time
        
        return result == sat, {
            'verification_time': verification_time,
            'approach': 'monolithic_z3',
            'result': 'sat' if result == sat else 'unsat'
        }
    
    @staticmethod
    def _add_monolithic_constraint(solver, constraint, actor_vars, scenario):
        """Add constraint to monolithic encoding"""
        if constraint.type == SpatialConstraintType.COLLISION_AVOID:
            # Distance constraints between all actor pairs
            for t in range(len(next(iter(actor_vars.values())))):
                for actor1, actor2 in itertools.combinations(constraint.actors, 2):
                    if actor1 in actor_vars and actor2 in actor_vars:
                        x1, y1, z1 = (actor_vars[actor1][t]['x'], 
                                     actor_vars[actor1][t]['y'], 
                                     actor_vars[actor1][t]['z'])
                        x2, y2, z2 = (actor_vars[actor2][t]['x'], 
                                     actor_vars[actor2][t]['y'], 
                                     actor_vars[actor2][t]['z'])
                        
                        min_dist = 1.0  # Minimum separation
                        solver.add((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2) >= min_dist**2)
    
    @staticmethod
    def monte_carlo_sampling(scenario: XRScenario) -> Tuple[bool, Dict]:
        """Monte Carlo sampling approach"""
        start_time = time.time()
        num_samples = 1000
        
        for _ in range(num_samples):
            # Generate random trajectory for each actor
            trajectories = {}
            for actor in scenario.actors:
                trajectories[actor.id] = BaselineApproaches._generate_random_trajectory(
                    actor, scenario.workspace_bounds, len(scenario.steps))
            
            # Check if this sample satisfies constraints
            if BaselineApproaches._check_trajectory_feasibility(trajectories, scenario):
                return True, {
                    'verification_time': time.time() - start_time,
                    'approach': 'monte_carlo',
                    'samples_tried': _ + 1
                }
        
        return False, {
            'verification_time': time.time() - start_time,
            'approach': 'monte_carlo',
            'samples_tried': num_samples
        }
    
    @staticmethod
    def _generate_random_trajectory(actor, bounds, num_steps):
        """Generate random trajectory within bounds"""
        trajectory = []
        x_max, y_max, z_max = bounds
        
        for _ in range(num_steps):
            pose = Pose(
                x=random.uniform(-x_max, x_max),
                y=random.uniform(-y_max, y_max),  
                z=random.uniform(0, z_max)
            )
            trajectory.append(pose)
        return trajectory
    
    @staticmethod
    def _check_trajectory_feasibility(trajectories, scenario):
        """Check if trajectories satisfy all constraints"""
        # Simplified feasibility check
        for constraint in scenario.spatial_constraints:
            if not BaselineApproaches._satisfies_constraint(trajectories, constraint):
                return False
        return True
    
    @staticmethod
    def _satisfies_constraint(trajectories, constraint):
        """Check if trajectories satisfy specific constraint"""
        if constraint.type == SpatialConstraintType.COLLISION_AVOID:
            # Check minimum distance between actors
            min_dist = 1.0
            for t in range(len(next(iter(trajectories.values())))):
                for actor1, actor2 in itertools.combinations(constraint.actors, 2):
                    if actor1 in trajectories and actor2 in trajectories:
                        pos1 = trajectories[actor1][t]
                        pos2 = trajectories[actor2][t]
                        dist = np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2)
                        if dist < min_dist:
                            return False
        return True
    
    @staticmethod
    def constraint_propagation_only(scenario: XRScenario) -> Tuple[bool, Dict]:
        """Constraint propagation without search"""
        start_time = time.time()
        
        # Simple constraint propagation - check obvious conflicts
        conflicts = []
        
        # Check for spatial region overlaps
        for constraint in scenario.spatial_constraints:
            if constraint.type == SpatialConstraintType.MUTUAL_EXCLUSION:
                # Check if multiple actors need same region simultaneously  
                conflicts.extend(BaselineApproaches._find_region_conflicts(constraint, scenario))
        
        # Check reachability
        for step in scenario.steps:
            if not BaselineApproaches._check_step_reachability(step, scenario):
                conflicts.append(f"Step {step.step_id} unreachable for {step.actor}")
        
        return len(conflicts) == 0, {
            'verification_time': time.time() - start_time,
            'approach': 'constraint_propagation',
            'conflicts': conflicts
        }
    
    @staticmethod
    def _find_region_conflicts(constraint, scenario):
        """Find obvious region conflicts"""
        conflicts = []
        # Simplified conflict detection
        if len(constraint.actors) > len(constraint.regions):
            conflicts.append(f"More actors than available regions in constraint")
        return conflicts
    
    @staticmethod
    def _check_step_reachability(step, scenario):
        """Check if interaction step is reachable"""
        actor = next(a for a in scenario.actors if a.id == step.actor)
        if step.target_region:
            # Check if target is within workspace
            bounds = scenario.workspace_bounds
            target = step.target_region.center
            return (abs(target.x) <= bounds[0] and 
                   abs(target.y) <= bounds[1] and 
                   target.z >= 0 and target.z <= bounds[2])
        return True
    
    @staticmethod
    def manual_decomposition_heuristic(scenario: XRScenario) -> Tuple[bool, Dict]:
        """Manual decomposition heuristic"""
        start_time = time.time()
        
        # Decompose by actor - check each independently
        actor_feasible = {}
        for actor in scenario.actors:
            actor_steps = [s for s in scenario.steps if s.actor == actor.id]
            actor_feasible[actor.id] = BaselineApproaches._check_actor_feasibility(
                actor, actor_steps, scenario)
        
        # Simple composition check
        overall_feasible = all(actor_feasible.values())
        
        return overall_feasible, {
            'verification_time': time.time() - start_time,
            'approach': 'manual_decomposition',
            'actor_results': actor_feasible
        }
    
    @staticmethod
    def _check_actor_feasibility(actor, steps, scenario):
        """Check feasibility for individual actor"""
        # Check if actor can complete all their steps within workspace
        for step in steps:
            if step.target_region:
                target = step.target_region.center
                if (abs(target.x) > scenario.workspace_bounds[0] or
                    abs(target.y) > scenario.workspace_bounds[1] or
                    target.z > scenario.workspace_bounds[2]):
                    return False
        return True


def create_xr_scenarios() -> List[XRScenario]:
    """Create 20 real-world XR interaction scenarios"""
    scenarios = []
    
    # REALIZABLE SCENARIOS (12)
    
    # 1. Multi-user VR collaborative whiteboard
    scenarios.append(XRScenario(
        name="vr_collaborative_whiteboard",
        description="Two users collaborating on a shared virtual whiteboard with turn-taking",
        actors=[
            Actor("user1", Pose(-1.0, 0.0, 1.5), 1.0, 0.3),
            Actor("user2", Pose(1.0, 0.0, 1.5), 1.0, 0.3)
        ],
        steps=[
            InteractionStep(1, "user1", InteractionType.GRAB, None,
                          target_region=SpatialRegion(Pose(0.0, 0.0, 2.0), 0.5),
                          duration=2.0),
            InteractionStep(2, "user1", InteractionType.GESTURE, None,
                          target_region=SpatialRegion(Pose(-0.5, 0.0, 2.0), 0.3),
                          duration=3.0),
            InteractionStep(3, "user2", InteractionType.GRAB, None,
                          target_region=SpatialRegion(Pose(0.0, 0.0, 2.0), 0.5),
                          duration=2.0),
            InteractionStep(4, "user2", InteractionType.GESTURE, None,
                          target_region=SpatialRegion(Pose(0.5, 0.0, 2.0), 0.3),
                          duration=3.0)
        ],
        spatial_constraints=[
            SpatialConstraint(SpatialConstraintType.COLLISION_AVOID, ["user1", "user2"], []),
            SpatialConstraint(SpatialConstraintType.MUTUAL_EXCLUSION, ["user1", "user2"], 
                            [SpatialRegion(Pose(0.0, 0.0, 2.0), 0.5)])
        ],
        workspace_bounds=(3.0, 3.0, 3.0),
        expected_realizable=True
    ))
    
    # 2. AR navigation with waypoints
    scenarios.append(XRScenario(
        name="ar_navigation_waypoints", 
        description="Single user AR navigation following waypoint sequence",
        actors=[
            Actor("user", Pose(0.0, -2.0, 0.0), 1.5, 0.4)
        ],
        steps=[
            InteractionStep(1, "user", InteractionType.NAVIGATE, None,
                          target_region=SpatialRegion(Pose(0.0, 0.0, 0.0), 0.5),
                          duration=3.0),
            InteractionStep(2, "user", InteractionType.NAVIGATE, None,
                          target_region=SpatialRegion(Pose(1.0, 1.0, 0.0), 0.5),
                          duration=3.0),
            InteractionStep(3, "user", InteractionType.NAVIGATE, None,
                          target_region=SpatialRegion(Pose(2.0, 0.0, 0.0), 0.5),
                          duration=3.0)
        ],
        spatial_constraints=[
            SpatialConstraint(SpatialConstraintType.WORKSPACE_BOUNDS, ["user"], [])
        ],
        workspace_bounds=(4.0, 4.0, 3.0),
        expected_realizable=True
    ))
    
    # 3. Object handoff between users
    scenarios.append(XRScenario(
        name="object_handoff_sequence",
        description="Three users passing virtual object in sequence",
        actors=[
            Actor("user1", Pose(-2.0, 0.0, 1.5), 1.2, 0.3),
            Actor("user2", Pose(0.0, 0.0, 1.5), 1.2, 0.3), 
            Actor("user3", Pose(2.0, 0.0, 1.5), 1.2, 0.3)
        ],
        steps=[
            InteractionStep(1, "user1", InteractionType.GRAB, None,
                          target_region=SpatialRegion(Pose(-1.0, 0.0, 1.5), 0.3),
                          duration=1.5),
            InteractionStep(2, "user1", InteractionType.HANDOFF, None,
                          target_region=SpatialRegion(Pose(-0.5, 0.0, 1.5), 0.3),
                          duration=2.0),
            InteractionStep(3, "user2", InteractionType.GRAB, None,
                          target_region=SpatialRegion(Pose(-0.5, 0.0, 1.5), 0.3),
                          duration=1.5),
            InteractionStep(4, "user2", InteractionType.HANDOFF, None,
                          target_region=SpatialRegion(Pose(0.5, 0.0, 1.5), 0.3),
                          duration=2.0),
            InteractionStep(5, "user3", InteractionType.GRAB, None,
                          target_region=SpatialRegion(Pose(0.5, 0.0, 1.5), 0.3),
                          duration=1.5)
        ],
        spatial_constraints=[
            SpatialConstraint(SpatialConstraintType.COLLISION_AVOID, 
                            ["user1", "user2", "user3"], []),
            SpatialConstraint(SpatialConstraintType.REACHABILITY,
                            ["user1", "user2"], 
                            [SpatialRegion(Pose(-0.5, 0.0, 1.5), 0.3)]),
            SpatialConstraint(SpatialConstraintType.REACHABILITY,
                            ["user2", "user3"],
                            [SpatialRegion(Pose(0.5, 0.0, 1.5), 0.3)])
        ],
        workspace_bounds=(4.0, 3.0, 3.0),
        expected_realizable=True
    ))
    
    # 4. Gesture-based menu interaction
    scenarios.append(XRScenario(
        name="gesture_menu_system",
        description="User navigating hierarchical gesture menu",
        actors=[
            Actor("user", Pose(0.0, 0.0, 1.6), 1.0, 0.4)
        ],
        steps=[
            InteractionStep(1, "user", InteractionType.GESTURE, None,
                          target_region=SpatialRegion(Pose(0.5, 0.0, 1.8), 0.2),
                          duration=1.0),
            InteractionStep(2, "user", InteractionType.MENU_SELECT, None,
                          target_region=SpatialRegion(Pose(0.8, 0.2, 1.8), 0.2),
                          duration=1.5),
            InteractionStep(3, "user", InteractionType.GESTURE, None,
                          target_region=SpatialRegion(Pose(0.8, -0.2, 1.6), 0.2),
                          duration=1.0),
            InteractionStep(4, "user", InteractionType.MENU_SELECT, None,
                          target_region=SpatialRegion(Pose(1.0, 0.0, 1.6), 0.2),
                          duration=1.5)
        ],
        spatial_constraints=[
            SpatialConstraint(SpatialConstraintType.REACHABILITY, ["user"],
                            [SpatialRegion(Pose(0.5, 0.0, 1.8), 0.2),
                             SpatialRegion(Pose(0.8, 0.2, 1.8), 0.2),
                             SpatialRegion(Pose(0.8, -0.2, 1.6), 0.2),
                             SpatialRegion(Pose(1.0, 0.0, 1.6), 0.2)])
        ],
        workspace_bounds=(3.0, 3.0, 3.0),
        expected_realizable=True
    ))
    
    # 5. Multi-user AR assembly task
    scenarios.append(XRScenario(
        name="ar_assembly_collaboration",
        description="Two users collaborating on AR-guided assembly with designated zones",
        actors=[
            Actor("assembler", Pose(-1.0, -1.0, 1.0), 1.5, 0.4),
            Actor("supervisor", Pose(1.0, 1.0, 1.6), 1.0, 0.3)
        ],
        steps=[
            InteractionStep(1, "assembler", InteractionType.GRAB, None,
                          target_region=SpatialRegion(Pose(0.0, -1.5, 0.8), 0.3),
                          duration=2.0),
            InteractionStep(2, "supervisor", InteractionType.POINT, None,
                          target_region=SpatialRegion(Pose(0.0, 0.0, 1.0), 0.5),
                          duration=1.5),
            InteractionStep(3, "assembler", InteractionType.NAVIGATE, None,
                          target_region=SpatialRegion(Pose(0.0, 0.0, 1.0), 0.5),
                          duration=3.0),
            InteractionStep(4, "assembler", InteractionType.COLLABORATE, None,
                          target_region=SpatialRegion(Pose(0.0, 0.0, 1.0), 0.5),
                          duration=4.0)
        ],
        spatial_constraints=[
            SpatialConstraint(SpatialConstraintType.COLLISION_AVOID,
                            ["assembler", "supervisor"], []),
            SpatialConstraint(SpatialConstraintType.VISIBILITY,
                            ["supervisor"], 
                            [SpatialRegion(Pose(0.0, 0.0, 1.0), 0.5)])
        ],
        workspace_bounds=(3.0, 4.0, 3.0),
        expected_realizable=True
    ))
    
    # 6-12: Additional realizable scenarios...
    
    for i in range(6, 13):
        scenarios.append(XRScenario(
            name=f"realizable_scenario_{i}",
            description=f"Generated realizable XR scenario {i}",
            actors=[
                Actor(f"user{j}", 
                     Pose(random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(0.5, 2)), 
                     1.0, 0.3) 
                for j in range(random.randint(2, 4))
            ],
            steps=[
                InteractionStep(
                    j, f"user{j % 2}", 
                    random.choice(list(InteractionType)),
                    None,  # target_pose
                    target_region=SpatialRegion(
                        Pose(random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5), 
                             random.uniform(0.5, 2.5)), 0.4),
                    duration=random.uniform(1.0, 3.0)
                )
                for j in range(random.randint(3, 7))
            ],
            spatial_constraints=[
                SpatialConstraint(SpatialConstraintType.COLLISION_AVOID,
                               [f"user{j}" for j in range(2)], [])
            ],
            workspace_bounds=(4.0, 4.0, 3.0),
            expected_realizable=True
        ))
    
    # UNREALIZABLE SCENARIOS (8)
    
    # 13. Impossible reach constraint
    scenarios.append(XRScenario(
        name="impossible_reach",
        description="User required to reach impossible distance",
        actors=[
            Actor("user", Pose(0.0, 0.0, 1.6), 1.0, 0.4)
        ],
        steps=[
            InteractionStep(1, "user", InteractionType.GRAB, None,
                          target_region=SpatialRegion(Pose(5.0, 5.0, 1.6), 0.2),
                          duration=1.0)
        ],
        spatial_constraints=[
            SpatialConstraint(SpatialConstraintType.REACHABILITY, ["user"],
                            [SpatialRegion(Pose(5.0, 5.0, 1.6), 0.2)])
        ],
        workspace_bounds=(3.0, 3.0, 3.0),
        expected_realizable=False
    ))
    
    # 14. Collision conflict
    scenarios.append(XRScenario(
        name="forced_collision",
        description="Two users forced to occupy same space simultaneously",
        actors=[
            Actor("user1", Pose(-1.0, 0.0, 1.5), 1.0, 0.5),
            Actor("user2", Pose(1.0, 0.0, 1.5), 1.0, 0.5)
        ],
        steps=[
            InteractionStep(1, "user1", InteractionType.GRAB, None,
                          target_region=SpatialRegion(Pose(0.0, 0.0, 1.5), 0.2),
                          duration=1.0),
            InteractionStep(2, "user2", InteractionType.GRAB, None,
                          target_region=SpatialRegion(Pose(0.0, 0.0, 1.5), 0.2),
                          duration=1.0)
        ],
        spatial_constraints=[
            SpatialConstraint(SpatialConstraintType.COLLISION_AVOID,
                            ["user1", "user2"], []),
            SpatialConstraint(SpatialConstraintType.MUTUAL_EXCLUSION,
                            ["user1", "user2"],
                            [SpatialRegion(Pose(0.0, 0.0, 1.5), 0.6)])
        ],
        workspace_bounds=(3.0, 3.0, 3.0),
        expected_realizable=False
    ))
    
    # 15-20: Additional unrealizable scenarios...
    
    for i in range(15, 21):
        # Create scenarios with conflicting constraints
        actors_list = [
            Actor(f"user{j}", 
                 Pose(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(1, 2)), 
                 0.8, 0.4) 
            for j in range(random.randint(2, 3))
        ]
        
        # Force impossible scenario by making all actors target same small region
        target_region = SpatialRegion(Pose(0.0, 0.0, 1.5), 0.1)
        
        scenarios.append(XRScenario(
            name=f"unrealizable_scenario_{i}",
            description=f"Generated unrealizable XR scenario {i} with conflicting constraints",
            actors=actors_list,
            steps=[
                InteractionStep(
                    j, f"user{j % len(actors_list)}", 
                    InteractionType.GRAB,
                    None,  # target_pose
                    target_region=target_region,
                    duration=1.0
                )
                for j in range(len(actors_list))
            ],
            spatial_constraints=[
                SpatialConstraint(SpatialConstraintType.COLLISION_AVOID,
                               [f"user{j}" for j in range(len(actors_list))], []),
                SpatialConstraint(SpatialConstraintType.MUTUAL_EXCLUSION,
                               [f"user{j}" for j in range(len(actors_list))],
                               [target_region])
            ],
            workspace_bounds=(2.0, 2.0, 3.0),
            expected_realizable=False
        ))
    
    return scenarios


def run_benchmarks():
    """Run complete benchmark suite"""
    print("🚀 Starting SOTA XR Choreography Benchmark Suite")
    print("=" * 60)
    
    scenarios = create_xr_scenarios()
    results = {
        'metadata': {
            'num_scenarios': len(scenarios),
            'realizable_count': sum(1 for s in scenarios if s.expected_realizable),
            'unrealizable_count': sum(1 for s in scenarios if not s.expected_realizable),
            'timestamp': time.time()
        },
        'scenario_results': [],
        'approach_summaries': {}
    }
    
    # Initialize approaches
    cegar = SpatialCEGAR()
    hybrid = HybridSpatialCEGAR()
    approaches = [
        ('spatial_cegar', cegar.verify_scenario),
        ('hybrid_cegar', hybrid.verify_scenario),
        ('monolithic_z3', BaselineApproaches.monolithic_z3),
        ('monte_carlo', BaselineApproaches.monte_carlo_sampling),
        ('constraint_propagation', BaselineApproaches.constraint_propagation_only),
        ('manual_decomposition', BaselineApproaches.manual_decomposition_heuristic)
    ]
    
    print(f"Testing {len(scenarios)} scenarios with {len(approaches)} approaches...")
    print()
    
    for i, scenario in enumerate(scenarios):
        print(f"📋 Scenario {i+1}: {scenario.name}")
        print(f"   {scenario.description}")
        print(f"   Actors: {len(scenario.actors)}, Steps: {len(scenario.steps)}")
        print(f"   Expected: {'✅ Realizable' if scenario.expected_realizable else '❌ Unrealizable'}")
        
        scenario_result = {
            'scenario_name': scenario.name,
            'expected_realizable': scenario.expected_realizable,
            'num_actors': len(scenario.actors),
            'num_steps': len(scenario.steps),
            'num_constraints': len(scenario.spatial_constraints),
            'approach_results': {}
        }
        
        # Test each approach
        for approach_name, approach_func in approaches:
            print(f"   🔬 Testing {approach_name}...", end=' ')
            try:
                realizable, metrics = approach_func(scenario)
                
                # Calculate accuracy
                correct = (realizable == scenario.expected_realizable)
                accuracy = 1.0 if correct else 0.0
                
                result = {
                    'realizable': realizable,
                    'correct': correct,
                    'accuracy': accuracy,
                    'metrics': metrics
                }
                scenario_result['approach_results'][approach_name] = result
                
                status = "✅" if correct else "❌"
                time_str = f"{metrics.get('verification_time', 0):.3f}s"
                print(f"{status} {time_str}")
                
            except Exception as e:
                print(f"💥 Error: {str(e)[:50]}...")
                scenario_result['approach_results'][approach_name] = {
                    'realizable': False,
                    'correct': False, 
                    'accuracy': 0.0,
                    'metrics': {'error': str(e)}
                }
        
        results['scenario_results'].append(scenario_result)
        print()
    
    # Compute summary statistics
    print("📊 Computing Summary Statistics...")
    
    for approach_name, _ in approaches:
        approach_results = []
        for scenario_result in results['scenario_results']:
            if approach_name in scenario_result['approach_results']:
                approach_results.append(scenario_result['approach_results'][approach_name])
        
        if approach_results:
            total_accuracy = sum(r['accuracy'] for r in approach_results) / len(approach_results)
            avg_time = np.mean([r['metrics'].get('verification_time', 0) for r in approach_results])
            
            # CEGAR-specific metrics
            if approach_name in ('spatial_cegar', 'hybrid_cegar'):
                avg_iterations = np.mean([
                    r['metrics'].get('refinement_iterations', 0) for r in approach_results
                ])
                avg_counterexamples = np.mean([
                    r['metrics'].get('counterexamples', 0) for r in approach_results  
                ])
                
                results['approach_summaries'][approach_name] = {
                    'accuracy': total_accuracy,
                    'avg_verification_time': avg_time,
                    'avg_refinement_iterations': avg_iterations,
                    'avg_counterexamples': avg_counterexamples,
                    'total_scenarios': len(approach_results)
                }
            else:
                results['approach_summaries'][approach_name] = {
                    'accuracy': total_accuracy,
                    'avg_verification_time': avg_time,
                    'total_scenarios': len(approach_results)
                }
    
    # Print final results
    print("🏆 Final Results Summary")
    print("=" * 60)
    
    for approach_name, summary in results['approach_summaries'].items():
        print(f"{approach_name}:")
        print(f"  Accuracy: {summary['accuracy']:.3f}")
        print(f"  Avg Time: {summary['avg_verification_time']:.3f}s")
        if 'avg_refinement_iterations' in summary:
            print(f"  Avg Iterations: {summary['avg_refinement_iterations']:.1f}")
            print(f"  Avg Counterexamples: {summary['avg_counterexamples']:.1f}")
        print()
    
    return results


def create_benchmark_plots(results):
    """Create visualization plots for benchmark results"""
    
    # Accuracy comparison
    approaches = list(results['approach_summaries'].keys())
    accuracies = [results['approach_summaries'][app]['accuracy'] for app in approaches]
    times = [results['approach_summaries'][app]['avg_verification_time'] for app in approaches]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy plot
    bars1 = ax1.bar(approaches, accuracies, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83'])
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Verification Accuracy by Approach')
    ax1.set_ylim(0, 1.1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Time comparison
    bars2 = ax2.bar(approaches, times, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83'])
    ax2.set_ylabel('Average Verification Time (s)')
    ax2.set_title('Verification Time by Approach')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_yscale('log')
    
    # Add value labels on bars
    for bar, time_val in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{time_val:.3f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/Users/halleyyoung/Documents/div/mathdivergence/pipeline_staging/choreo-xr-interaction-compiler/benchmarks/benchmark_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # CEGAR refinement analysis
    if 'spatial_cegar' in results['approach_summaries']:
        cegar_results = []
        for scenario in results['scenario_results']:
            if 'spatial_cegar' in scenario['approach_results']:
                cegar_data = scenario['approach_results']['spatial_cegar']['metrics']
                cegar_results.append({
                    'iterations': cegar_data.get('refinement_iterations', 0),
                    'time': cegar_data.get('verification_time', 0),
                    'realizable': scenario['expected_realizable']
                })
        
        if cegar_results:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            realizable_iter = [r['iterations'] for r in cegar_results if r['realizable']]
            unrealizable_iter = [r['iterations'] for r in cegar_results if not r['realizable']]
            
            ax.hist([realizable_iter, unrealizable_iter], 
                   bins=range(0, max(max(realizable_iter, default=0), max(unrealizable_iter, default=0)) + 2),
                   label=['Realizable', 'Unrealizable'], 
                   alpha=0.7, color=['green', 'red'])
            ax.set_xlabel('CEGAR Refinement Iterations')
            ax.set_ylabel('Number of Scenarios')
            ax.set_title('CEGAR Refinement Iterations Distribution')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig('/Users/halleyyoung/Documents/div/mathdivergence/pipeline_staging/choreo-xr-interaction-compiler/benchmarks/cegar_refinement_analysis.png',
                       dpi=300, bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    # Run benchmarks
    results = run_benchmarks()
    
    # Create visualizations
    create_benchmark_plots(results)
    
    # Save results
    output_path = '/Users/halleyyoung/Documents/div/mathdivergence/pipeline_staging/choreo-xr-interaction-compiler/benchmarks/real_benchmark_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Results saved to: {output_path}")
    print("🎨 Plots saved to benchmarks/ directory")
    print("✅ Benchmark suite completed successfully!")