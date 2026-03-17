#!/usr/bin/env python3
"""
SOTA Benchmark for Safe Deployment Planner

Creates 20 realistic Kubernetes deployment scenarios and evaluates deployment 
planning algorithms against state-of-the-art baselines.

Real-world characteristics:
- Microservice dependency graphs (3-50 services)
- Resource constraints (CPU/memory limits)  
- Health check requirements
- Rollback conditions
- Known unsafe scenarios (circular deps, over-subscription)

Baselines:
- Topological sort deployment (NetworkX)
- Random ordering with constraint checking  
- Greedy resource-first scheduling
- Safe deployment planner (Z3-based)

Metrics:
- Plan safety (constraint violations)
- Optimality (total deployment time)
- Planning time
- Rollback detection accuracy
"""

import json
import time
import random
import math
import statistics
import hashlib
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import networkx as nx
import numpy as np
from z3 import *

# Set random seed for reproducible results
random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

class ServiceType(Enum):
    API_GATEWAY = "api_gateway"
    MICROSERVICE = "microservice" 
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    LOAD_BALANCER = "load_balancer"
    AUTH_SERVICE = "auth_service"

@dataclass
class ResourceRequirements:
    cpu_cores: float
    memory_gb: float
    storage_gb: float
    network_mbps: float

@dataclass
class HealthCheck:
    path: str
    port: int
    initial_delay_seconds: int
    timeout_seconds: int
    period_seconds: int
    failure_threshold: int

@dataclass
class Service:
    name: str
    service_type: ServiceType
    version: str
    replicas: int
    resources: ResourceRequirements
    health_check: HealthCheck
    dependencies: Set[str]
    rollback_conditions: List[str]
    deploy_time_seconds: int
    criticality: str  # "critical", "important", "standard"
    
    def __post_init__(self):
        if isinstance(self.dependencies, list):
            self.dependencies = set(self.dependencies)

@dataclass
class ClusterResources:
    total_cpu_cores: float
    total_memory_gb: float
    total_storage_gb: float
    total_network_mbps: float
    max_nodes: int

@dataclass
class DeploymentConstraints:
    max_simultaneous_down_percent: float
    max_deployment_time_minutes: int
    require_dependency_ordering: bool
    require_health_checks: bool
    allow_rollback: bool
    critical_services_first: bool

@dataclass
class DeploymentScenario:
    name: str
    description: str
    services: List[Service]
    cluster_resources: ClusterResources
    constraints: DeploymentConstraints
    known_unsafe: bool
    unsafe_reason: str = ""
    expected_violations: List[str] = field(default_factory=list)

@dataclass
class DeploymentPlan:
    service_order: List[str]
    parallel_groups: List[List[str]]
    estimated_total_time_minutes: float
    resource_allocation: Dict[str, ResourceRequirements]
    rollback_plan: List[str]

@dataclass
class BenchmarkResult:
    scenario_name: str
    algorithm: str
    plan_valid: bool
    constraint_violations: List[str]
    total_deployment_time_minutes: float
    planning_time_seconds: float
    safety_score: float  # 0-1, higher is better
    optimality_score: float  # 0-1, higher is better
    rollback_detected: bool
    success: bool

# ─────────────────────────────────────────────────────────────────────────────
# Scenario Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_realistic_service(name: str, service_type: ServiceType, dependencies: Set[str] = None) -> Service:
    """Generate a realistic microservice based on type."""
    if dependencies is None:
        dependencies = set()
    
    # Resource profiles based on service type
    resource_profiles = {
        ServiceType.API_GATEWAY: ResourceRequirements(2.0, 4.0, 10.0, 1000.0),
        ServiceType.MICROSERVICE: ResourceRequirements(1.0, 2.0, 5.0, 200.0),
        ServiceType.DATABASE: ResourceRequirements(4.0, 16.0, 100.0, 500.0),
        ServiceType.CACHE: ResourceRequirements(0.5, 8.0, 20.0, 800.0),
        ServiceType.MESSAGE_QUEUE: ResourceRequirements(1.0, 4.0, 50.0, 300.0),
        ServiceType.LOAD_BALANCER: ResourceRequirements(1.0, 1.0, 2.0, 2000.0),
        ServiceType.AUTH_SERVICE: ResourceRequirements(1.5, 3.0, 10.0, 400.0),
    }
    
    base_resources = resource_profiles[service_type]
    # Add some variance
    variance = 0.2
    resources = ResourceRequirements(
        cpu_cores=base_resources.cpu_cores * random.uniform(1-variance, 1+variance),
        memory_gb=base_resources.memory_gb * random.uniform(1-variance, 1+variance),
        storage_gb=base_resources.storage_gb * random.uniform(1-variance, 1+variance),
        network_mbps=base_resources.network_mbps * random.uniform(1-variance, 1+variance)
    )
    
    # Health check based on service type
    if service_type in [ServiceType.API_GATEWAY, ServiceType.MICROSERVICE, ServiceType.AUTH_SERVICE]:
        health_check = HealthCheck("/health", 8080, 30, 10, 10, 3)
    elif service_type == ServiceType.DATABASE:
        health_check = HealthCheck("/", 5432, 60, 30, 30, 5)
    else:
        health_check = HealthCheck("/", 80, 15, 5, 5, 2)
    
    # Deployment time based on complexity
    deploy_times = {
        ServiceType.DATABASE: random.randint(300, 600),
        ServiceType.MESSAGE_QUEUE: random.randint(120, 300),
        ServiceType.API_GATEWAY: random.randint(60, 180),
        ServiceType.MICROSERVICE: random.randint(30, 120),
        ServiceType.CACHE: random.randint(20, 60),
        ServiceType.LOAD_BALANCER: random.randint(30, 90),
        ServiceType.AUTH_SERVICE: random.randint(60, 150),
    }
    
    criticality_weights = {
        ServiceType.DATABASE: ["critical"] * 7 + ["important"] * 3,
        ServiceType.AUTH_SERVICE: ["critical"] * 6 + ["important"] * 4,
        ServiceType.API_GATEWAY: ["critical"] * 5 + ["important"] * 4 + ["standard"] * 1,
        ServiceType.LOAD_BALANCER: ["important"] * 6 + ["critical"] * 3 + ["standard"] * 1,
        ServiceType.MESSAGE_QUEUE: ["important"] * 5 + ["critical"] * 2 + ["standard"] * 3,
        ServiceType.MICROSERVICE: ["standard"] * 6 + ["important"] * 3 + ["critical"] * 1,
        ServiceType.CACHE: ["standard"] * 5 + ["important"] * 4 + ["critical"] * 1,
    }
    
    return Service(
        name=name,
        service_type=service_type,
        version=f"v{random.randint(1, 10)}.{random.randint(0, 20)}.{random.randint(0, 50)}",
        replicas=random.randint(1, 8),
        resources=resources,
        health_check=health_check,
        dependencies=dependencies,
        rollback_conditions=[
            "health_check_failure",
            "high_error_rate",
            "resource_exhaustion",
            "dependency_failure"
        ],
        deploy_time_seconds=deploy_times[service_type],
        criticality=random.choice(criticality_weights[service_type])
    )

def create_ecommerce_scenario() -> DeploymentScenario:
    """E-commerce platform with typical microservices."""
    services = [
        generate_realistic_service("api-gateway", ServiceType.API_GATEWAY),
        generate_realistic_service("user-service", ServiceType.MICROSERVICE),
        generate_realistic_service("product-service", ServiceType.MICROSERVICE),
        generate_realistic_service("order-service", ServiceType.MICROSERVICE),
        generate_realistic_service("payment-service", ServiceType.MICROSERVICE),
        generate_realistic_service("inventory-service", ServiceType.MICROSERVICE),
        generate_realistic_service("notification-service", ServiceType.MICROSERVICE),
        generate_realistic_service("auth-service", ServiceType.AUTH_SERVICE),
        generate_realistic_service("user-db", ServiceType.DATABASE),
        generate_realistic_service("product-db", ServiceType.DATABASE),
        generate_realistic_service("order-db", ServiceType.DATABASE),
        generate_realistic_service("redis-cache", ServiceType.CACHE),
        generate_realistic_service("rabbitmq", ServiceType.MESSAGE_QUEUE),
        generate_realistic_service("load-balancer", ServiceType.LOAD_BALANCER)
    ]
    
    # Set up realistic dependencies
    services[0].dependencies = {"auth-service", "load-balancer"}  # api-gateway
    services[1].dependencies = {"user-db", "auth-service"}  # user-service
    services[2].dependencies = {"product-db", "redis-cache"}  # product-service
    services[3].dependencies = {"order-db", "user-service", "inventory-service"}  # order-service
    services[4].dependencies = {"payment-db", "auth-service"}  # payment-service
    services[5].dependencies = {"product-service", "redis-cache"}  # inventory-service
    services[6].dependencies = {"rabbitmq", "user-service"}  # notification-service
    # Note: load-balancer sits in front of api-gateway but does NOT
    # create a deployment-time dependency cycle; it only needs the
    # gateway's DNS record, which Kubernetes provisions at Service-
    # creation time before the pods are ready.  So we leave load-balancer
    # with *no* deployment dependency on api-gateway.
    
    # Add payment-db
    payment_db = generate_realistic_service("payment-db", ServiceType.DATABASE)
    services.append(payment_db)
    
    return DeploymentScenario(
        name="ecommerce-platform",
        description="E-commerce platform with 15 microservices",
        services=services,
        cluster_resources=ClusterResources(128.0, 512.0, 4000.0, 20000.0, 10),
        constraints=DeploymentConstraints(
            max_simultaneous_down_percent=0.3,
            max_deployment_time_minutes=45,
            require_dependency_ordering=True,
            require_health_checks=True,
            allow_rollback=True,
            critical_services_first=True
        ),
        known_unsafe=False
    )

def create_unsafe_circular_deps_scenario() -> DeploymentScenario:
    """Scenario with circular dependencies - known to be unsafe."""
    services = [
        generate_realistic_service("service-a", ServiceType.MICROSERVICE, {"service-c"}),
        generate_realistic_service("service-b", ServiceType.MICROSERVICE, {"service-a"}),
        generate_realistic_service("service-c", ServiceType.MICROSERVICE, {"service-b"}),
        generate_realistic_service("database", ServiceType.DATABASE)
    ]
    
    return DeploymentScenario(
        name="circular-dependencies",
        description="Services with circular dependencies",
        services=services,
        cluster_resources=ClusterResources(16.0, 64.0, 500.0, 5000.0, 5),
        constraints=DeploymentConstraints(
            max_simultaneous_down_percent=0.5,
            max_deployment_time_minutes=30,
            require_dependency_ordering=True,
            require_health_checks=True,
            allow_rollback=True,
            critical_services_first=False
        ),
        known_unsafe=True,
        unsafe_reason="circular_dependencies",
        expected_violations=["dependency_cycle_detected"]
    )

def create_resource_oversubscription_scenario() -> DeploymentScenario:
    """Scenario that requires more resources than available - known unsafe."""
    services = []
    
    # Create resource-hungry services
    for i in range(5):
        service = generate_realistic_service(f"heavy-service-{i}", ServiceType.DATABASE)
        service.resources = ResourceRequirements(8.0, 32.0, 200.0, 1000.0)
        services.append(service)
    
    return DeploymentScenario(
        name="resource-oversubscription",
        description="Services requiring more resources than cluster capacity",
        services=services,
        cluster_resources=ClusterResources(20.0, 80.0, 500.0, 2000.0, 3),  # Not enough for all services
        constraints=DeploymentConstraints(
            max_simultaneous_down_percent=0.2,
            max_deployment_time_minutes=60,
            require_dependency_ordering=True,
            require_health_checks=True,
            allow_rollback=True,
            critical_services_first=True
        ),
        known_unsafe=True,
        unsafe_reason="insufficient_cluster_resources",
        expected_violations=["resource_constraint_violation"]
    )

def generate_random_microservice_topology(num_services: int) -> DeploymentScenario:
    """Generate a random but realistic microservice topology."""
    service_types = [
        ServiceType.MICROSERVICE,
        ServiceType.DATABASE,
        ServiceType.CACHE,
        ServiceType.MESSAGE_QUEUE,
        ServiceType.AUTH_SERVICE
    ]
    
    services = []
    
    # Always include an API gateway
    api_gateway = generate_realistic_service("api-gateway", ServiceType.API_GATEWAY)
    services.append(api_gateway)
    
    # Generate other services
    for i in range(1, num_services):
        service_type = random.choice(service_types)
        service_name = f"{service_type.value.replace('_', '-')}-{i}"
        service = generate_realistic_service(service_name, service_type)
        services.append(service)
    
    # Create realistic DAG dependency graph (only depend on earlier services)
    for idx, service in enumerate(services[1:], start=1):
        # Each service can only depend on services earlier in the list (DAG guarantee)
        num_deps = random.randint(0, min(3, idx))
        potential_deps = [services[j].name for j in range(idx)]
        
        if num_deps > 0 and potential_deps:
            deps = random.sample(potential_deps, min(num_deps, len(potential_deps)))
            service.dependencies.update(deps)
    
    # API gateway depends on some auth/load-balancer services.
    # We must ensure this doesn't create a cycle: only add deps
    # on services that do NOT (transitively) depend on api-gateway.
    dep_graph = nx.DiGraph()
    for s in services:
        dep_graph.add_node(s.name)
        for d in s.dependencies:
            dep_graph.add_edge(d, s.name)
    
    reachable_from_gw = nx.descendants(dep_graph, "api-gateway")
    potential_deps = [s.name for s in services[1:]
                      if s.service_type in [ServiceType.AUTH_SERVICE, ServiceType.LOAD_BALANCER]
                      and s.name not in reachable_from_gw]
    if potential_deps:
        api_gateway.dependencies.update(random.sample(potential_deps, min(2, len(potential_deps))))
    
    # Calculate appropriate cluster resources
    total_cpu = sum(s.resources.cpu_cores * s.replicas for s in services)
    total_memory = sum(s.resources.memory_gb * s.replicas for s in services)
    total_storage = sum(s.resources.storage_gb * s.replicas for s in services)
    total_network = sum(s.resources.network_mbps * s.replicas for s in services)
    
    # Add 20-50% overhead
    overhead = random.uniform(1.2, 1.5)
    cluster_resources = ClusterResources(
        total_cpu_cores=total_cpu * overhead,
        total_memory_gb=total_memory * overhead,
        total_storage_gb=total_storage * overhead,
        total_network_mbps=total_network * overhead,
        max_nodes=max(3, num_services // 3)
    )
    
    return DeploymentScenario(
        name=f"random-topology-{num_services}",
        description=f"Random microservice topology with {num_services} services",
        services=services,
        cluster_resources=cluster_resources,
        constraints=DeploymentConstraints(
            max_simultaneous_down_percent=random.uniform(0.2, 0.4),
            max_deployment_time_minutes=random.randint(30, 120),
            require_dependency_ordering=True,
            require_health_checks=True,
            allow_rollback=True,
            critical_services_first=random.choice([True, False])
        ),
        known_unsafe=False
    )

def create_all_scenarios() -> List[DeploymentScenario]:
    """Create all 20 benchmark scenarios."""
    scenarios = []
    
    # Hand-crafted realistic scenarios
    scenarios.append(create_ecommerce_scenario())
    scenarios.append(create_unsafe_circular_deps_scenario())
    scenarios.append(create_resource_oversubscription_scenario())
    
    # Media streaming platform
    streaming_services = [
        generate_realistic_service("video-api", ServiceType.API_GATEWAY),
        generate_realistic_service("video-service", ServiceType.MICROSERVICE, {"video-db", "cdn-cache"}),
        generate_realistic_service("user-service", ServiceType.MICROSERVICE, {"user-db"}),
        generate_realistic_service("recommendation-service", ServiceType.MICROSERVICE, {"ml-db", "redis-cache"}),
        generate_realistic_service("auth-service", ServiceType.AUTH_SERVICE, {"auth-db"}),
        generate_realistic_service("billing-service", ServiceType.MICROSERVICE, {"billing-db"}),
        generate_realistic_service("video-db", ServiceType.DATABASE),
        generate_realistic_service("user-db", ServiceType.DATABASE),
        generate_realistic_service("ml-db", ServiceType.DATABASE),
        generate_realistic_service("auth-db", ServiceType.DATABASE),
        generate_realistic_service("billing-db", ServiceType.DATABASE),
        generate_realistic_service("redis-cache", ServiceType.CACHE),
        generate_realistic_service("cdn-cache", ServiceType.CACHE),
        generate_realistic_service("kafka", ServiceType.MESSAGE_QUEUE)
    ]
    
    scenarios.append(DeploymentScenario(
        name="streaming-platform",
        description="Video streaming platform with ML recommendations",
        services=streaming_services,
        cluster_resources=ClusterResources(256.0, 1024.0, 10000.0, 100000.0, 20),
        constraints=DeploymentConstraints(
            max_simultaneous_down_percent=0.25,
            max_deployment_time_minutes=60,
            require_dependency_ordering=True,
            require_health_checks=True,
            allow_rollback=True,
            critical_services_first=True
        ),
        known_unsafe=False
    ))
    
    # Financial trading system  
    trading_services = [
        generate_realistic_service("trading-gateway", ServiceType.API_GATEWAY),
        generate_realistic_service("order-matching", ServiceType.MICROSERVICE, {"market-db"}),
        generate_realistic_service("risk-engine", ServiceType.MICROSERVICE, {"risk-db"}),
        generate_realistic_service("position-tracker", ServiceType.MICROSERVICE, {"position-db"}),
        generate_realistic_service("market-data", ServiceType.MICROSERVICE, {"market-db", "redis-cache"}),
        generate_realistic_service("settlement", ServiceType.MICROSERVICE, {"settlement-db"}),
        generate_realistic_service("auth-service", ServiceType.AUTH_SERVICE, {"auth-db"}),
        generate_realistic_service("market-db", ServiceType.DATABASE),
        generate_realistic_service("risk-db", ServiceType.DATABASE),
        generate_realistic_service("position-db", ServiceType.DATABASE),
        generate_realistic_service("settlement-db", ServiceType.DATABASE),
        generate_realistic_service("auth-db", ServiceType.DATABASE),
        generate_realistic_service("redis-cache", ServiceType.CACHE)
    ]
    
    scenarios.append(DeploymentScenario(
        name="trading-system",
        description="High-frequency trading system with strict latency requirements",
        services=trading_services,
        cluster_resources=ClusterResources(256.0, 1024.0, 10000.0, 100000.0, 30),
        constraints=DeploymentConstraints(
            max_simultaneous_down_percent=0.1,  # Very strict
            max_deployment_time_minutes=30,
            require_dependency_ordering=True,
            require_health_checks=True,
            allow_rollback=True,
            critical_services_first=True
        ),
        known_unsafe=False
    ))
    
    # Generate random scenarios of varying sizes
    sizes = [5, 8, 12, 15, 20, 25, 30, 35, 40, 45, 50, 7, 10, 18, 28]
    for size in sizes:
        scenarios.append(generate_random_microservice_topology(size))
    
    return scenarios

# ─────────────────────────────────────────────────────────────────────────────
# Deployment Planning Algorithms  
# ─────────────────────────────────────────────────────────────────────────────

class TopologicalSortPlanner:
    """Baseline: Topological sort of dependency graph."""
    
    def plan_deployment(self, scenario: DeploymentScenario) -> Tuple[DeploymentPlan, List[str]]:
        start_time = time.time()
        
        # Build dependency graph
        G = nx.DiGraph()
        for service in scenario.services:
            G.add_node(service.name)
            for dep in service.dependencies:
                G.add_edge(dep, service.name)
        
        violations = []
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(G):
            violations.append("dependency_cycle_detected")
            return DeploymentPlan([], [], float('inf'), {}, []), violations
        
        # Topological sort
        try:
            topo_order = list(nx.topological_sort(G))
        except nx.NetworkXError:
            violations.append("topological_sort_failed")
            return DeploymentPlan([], [], float('inf'), {}, []), violations
        
        # Simple sequential deployment
        total_time = sum(next(s.deploy_time_seconds for s in scenario.services if s.name == name) / 60.0 for name in topo_order)
        
        # Check resource constraints
        total_cpu = sum(s.resources.cpu_cores * s.replicas for s in scenario.services)
        total_memory = sum(s.resources.memory_gb * s.replicas for s in scenario.services) 
        
        if total_cpu > scenario.cluster_resources.total_cpu_cores:
            violations.append("cpu_constraint_violation")
        if total_memory > scenario.cluster_resources.total_memory_gb:
            violations.append("memory_constraint_violation")
        
        parallel_groups = [[name] for name in topo_order]
        
        plan = DeploymentPlan(
            service_order=topo_order,
            parallel_groups=parallel_groups,
            estimated_total_time_minutes=total_time,
            resource_allocation={s.name: s.resources for s in scenario.services},
            rollback_plan=list(reversed(topo_order))
        )
        
        return plan, violations

class RandomOrderPlanner:
    """Baseline: Random ordering with basic constraint checking."""
    
    def plan_deployment(self, scenario: DeploymentScenario) -> Tuple[DeploymentPlan, List[str]]:
        violations = []
        
        # Random shuffle
        service_names = [s.name for s in scenario.services]
        random.shuffle(service_names)
        
        # Check dependencies are satisfied
        deployed = set()
        valid_order = []
        
        max_attempts = len(service_names) * 2
        attempts = 0
        
        while service_names and attempts < max_attempts:
            attempts += 1
            made_progress = False
            
            for name in service_names[:]:
                service = next(s for s in scenario.services if s.name == name)
                
                # Check if dependencies are satisfied
                if service.dependencies.issubset(deployed):
                    valid_order.append(name)
                    deployed.add(name)
                    service_names.remove(name)
                    made_progress = True
            
            if not made_progress:
                violations.append("dependency_deadlock")
                break
        
        if service_names:  # Some services couldn't be deployed
            violations.append("unresolvable_dependencies")
        
        # Calculate time
        total_time = sum(next(s.deploy_time_seconds for s in scenario.services if s.name == name) / 60.0 for name in valid_order)
        
        # Check resources
        total_cpu = sum(s.resources.cpu_cores * s.replicas for s in scenario.services)
        if total_cpu > scenario.cluster_resources.total_cpu_cores:
            violations.append("cpu_constraint_violation")
        
        parallel_groups = [[name] for name in valid_order]
        
        plan = DeploymentPlan(
            service_order=valid_order,
            parallel_groups=parallel_groups,
            estimated_total_time_minutes=total_time,
            resource_allocation={s.name: s.resources for s in scenario.services},
            rollback_plan=list(reversed(valid_order))
        )
        
        return plan, violations

class GreedyResourcePlanner:
    """Baseline: Greedy scheduling prioritizing resource availability."""
    
    def plan_deployment(self, scenario: DeploymentScenario) -> Tuple[DeploymentPlan, List[str]]:
        violations = []
        
        # Sort by resource usage (largest first) and criticality
        def resource_priority(service):
            resource_score = (service.resources.cpu_cores * service.replicas + 
                            service.resources.memory_gb * service.replicas * 0.1)
            criticality_bonus = {"critical": 1000, "important": 100, "standard": 0}[service.criticality]
            return resource_score + criticality_bonus
        
        services_by_priority = sorted(scenario.services, key=resource_priority, reverse=True)
        
        # Build dependency graph for validation
        deps = {s.name: s.dependencies for s in scenario.services}
        
        # Greedy placement respecting dependencies
        deployed = set()
        deployment_order = []
        
        while len(deployment_order) < len(scenario.services):
            made_progress = False
            
            for service in services_by_priority:
                if (service.name not in deployed and 
                    service.dependencies.issubset(deployed)):
                    
                    deployment_order.append(service.name)
                    deployed.add(service.name)
                    made_progress = True
                    break
            
            if not made_progress:
                # Find services that can be deployed (no unmet deps)
                remaining = [s for s in services_by_priority if s.name not in deployed]
                for service in remaining:
                    if service.dependencies.issubset(deployed):
                        deployment_order.append(service.name)
                        deployed.add(service.name)
                        made_progress = True
                        break
            
            if not made_progress:
                violations.append("greedy_scheduling_deadlock")
                break
        
        # Resource validation
        total_cpu = sum(s.resources.cpu_cores * s.replicas for s in scenario.services)
        if total_cpu > scenario.cluster_resources.total_cpu_cores:
            violations.append("cpu_constraint_violation")
        
        # Time calculation
        total_time = sum(next(s.deploy_time_seconds for s in scenario.services if s.name == name) / 60.0 
                        for name in deployment_order)
        
        parallel_groups = [[name] for name in deployment_order]
        
        plan = DeploymentPlan(
            service_order=deployment_order,
            parallel_groups=parallel_groups,
            estimated_total_time_minutes=total_time,
            resource_allocation={s.name: s.resources for s in scenario.services},
            rollback_plan=list(reversed(deployment_order))
        )
        
        return plan, violations

class SafeDeploymentPlanner:
    """Our Z3-based safe deployment planner with cycle detection,
    resource validation, and parallel-group optimization."""
    
    def plan_deployment(self, scenario: DeploymentScenario) -> Tuple[DeploymentPlan, List[str]]:
        violations = []
        
        services = {s.name: s for s in scenario.services}
        service_names = list(services.keys())
        n = len(service_names)
        
        # ── Phase 0: structural pre-checks (cycle + resource) ──
        G = nx.DiGraph()
        for s in scenario.services:
            G.add_node(s.name)
            for dep in s.dependencies:
                if dep in services:
                    G.add_edge(dep, s.name)
        
        if not nx.is_directed_acyclic_graph(G):
            violations.append("dependency_cycle_detected")
            return DeploymentPlan([], [], float('inf'), {}, []), violations
        
        total_cpu = sum(s.resources.cpu_cores * s.replicas for s in scenario.services)
        total_memory = sum(s.resources.memory_gb * s.replicas for s in scenario.services)
        
        if total_cpu > scenario.cluster_resources.total_cpu_cores:
            violations.append("cpu_constraint_violation")
        if total_memory > scenario.cluster_resources.total_memory_gb:
            violations.append("memory_constraint_violation")
        
        if "cpu_constraint_violation" in violations or "memory_constraint_violation" in violations:
            violations.append("resource_constraint_violation")
            return DeploymentPlan([], [], float('inf'), {}, []), violations
        
        # ── Phase 1: Z3 Solver for hard dependency constraints ──
        try:
            solver = Solver()
            solver.set("timeout", 10000)  # 10 s timeout
            
            order_vars = {name: Int(f"order_{name}") for name in service_names}
            
            for name in service_names:
                solver.add(order_vars[name] >= 0)
                solver.add(order_vars[name] < n)
            
            solver.add(Distinct([order_vars[name] for name in service_names]))
            
            # Hard dependency ordering
            for service in scenario.services:
                for dep in service.dependencies:
                    if dep in order_vars:
                        solver.add(order_vars[dep] < order_vars[service.name])
            
            result = solver.check()
            
            if result != sat:
                if result == unsat:
                    violations.append("unsatisfiable_constraints")
                else:
                    violations.append("solver_timeout")
                return DeploymentPlan([], [], float('inf'), {}, []), violations
            
            model = solver.model()
            order_assignments = {name: model[order_vars[name]].as_long()
                                 for name in service_names}
            
            deployment_order = sorted(service_names,
                                      key=lambda x: order_assignments[x])
            
        except Exception as e:
            violations.append(f"z3_solver_error: {str(e)}")
            return DeploymentPlan([], [], float('inf'), {}, []), violations
        
        # ── Phase 2: criticality-aware parallel groups ──
        parallel_groups = self._create_parallel_groups(deployment_order, services)
        
        # Within each group, sort critical services first
        if scenario.constraints.critical_services_first:
            crit_rank = {"critical": 0, "important": 1, "standard": 2}
            parallel_groups = [
                sorted(group, key=lambda nm: crit_rank.get(services[nm].criticality, 1))
                for group in parallel_groups
            ]
        
        # Flatten to get the final order
        deployment_order = [s for group in parallel_groups for s in group]
        
        total_time = self._calculate_parallel_time(parallel_groups, services)
        
        plan = DeploymentPlan(
            service_order=deployment_order,
            parallel_groups=parallel_groups,
            estimated_total_time_minutes=total_time,
            resource_allocation={s.name: s.resources for s in scenario.services},
            rollback_plan=list(reversed(deployment_order))
        )
        
        return plan, violations
    
    def _create_parallel_groups(self, order: List[str], services: Dict[str, Service]) -> List[List[str]]:
        """Create parallel deployment groups from ordered services."""
        groups = []
        remaining = order[:]
        
        while remaining:
            # Find services that can be deployed in parallel (no interdependencies)
            parallel_group = []
            deployed_so_far = set(s for group in groups for s in group)
            
            for service_name in remaining[:]:
                service = services[service_name]
                
                # Check if all dependencies are already deployed
                if service.dependencies.issubset(deployed_so_far):
                    # Check if this service conflicts with others in current group
                    can_parallelize = True
                    for other_name in parallel_group:
                        other_service = services[other_name]
                        if (service_name in other_service.dependencies or 
                            other_name in service.dependencies):
                            can_parallelize = False
                            break
                    
                    if can_parallelize:
                        parallel_group.append(service_name)
                        remaining.remove(service_name)
            
            if parallel_group:
                groups.append(parallel_group)
            else:
                # If no progress, take the first remaining service
                groups.append([remaining.pop(0)])
        
        return groups
    
    def _calculate_parallel_time(self, parallel_groups: List[List[str]], services: Dict[str, Service]) -> float:
        """Calculate total deployment time with parallel execution."""
        total_time = 0.0
        
        for group in parallel_groups:
            # Time for this group is the maximum time of any service in the group
            group_time = max(services[name].deploy_time_seconds / 60.0 for name in group)
            total_time += group_time
        
        return total_time

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Execution and Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_plan_safety(plan: DeploymentPlan, scenario: DeploymentScenario, violations: List[str]) -> float:
    """Calculate safety score (0-1, higher is better)."""
    if not plan.service_order:
        return 0.0
    
    safety_score = 1.0
    
    # Penalize constraint violations
    safety_score -= len(violations) * 0.2
    
    # Check deployment ordering respects dependencies
    deployed = set()
    for service_name in plan.service_order:
        service = next(s for s in scenario.services if s.name == service_name)
        
        # Check if dependencies are satisfied
        unmet_deps = service.dependencies - deployed
        if unmet_deps:
            safety_score -= 0.1 * len(unmet_deps)
        
        deployed.add(service_name)
    
    # Check resource constraints
    total_cpu = sum(s.resources.cpu_cores * s.replicas for s in scenario.services)
    total_memory = sum(s.resources.memory_gb * s.replicas for s in scenario.services)
    
    if total_cpu > scenario.cluster_resources.total_cpu_cores:
        safety_score -= 0.3
    if total_memory > scenario.cluster_resources.total_memory_gb:
        safety_score -= 0.3
    
    # Check criticality ordering
    if scenario.constraints.critical_services_first:
        critical_positions = []
        standard_positions = []
        
        for i, service_name in enumerate(plan.service_order):
            service = next(s for s in scenario.services if s.name == service_name)
            if service.criticality == "critical":
                critical_positions.append(i)
            elif service.criticality == "standard":
                standard_positions.append(i)
        
        # Critical services should generally come before standard ones
        for crit_pos in critical_positions:
            for std_pos in standard_positions:
                if crit_pos > std_pos:
                    safety_score -= 0.05
    
    return max(0.0, min(1.0, safety_score))

def evaluate_plan_optimality(plan: DeploymentPlan, scenario: DeploymentScenario) -> float:
    """Calculate optimality score (0-1, higher is better) based on deployment time."""
    if not plan.service_order:
        return 0.0
    
    # Calculate theoretical minimum time (all services in parallel, ignoring dependencies)
    min_possible_time = max(s.deploy_time_seconds / 60.0 for s in scenario.services)
    
    # Calculate maximum time (all services sequential)
    max_possible_time = sum(s.deploy_time_seconds / 60.0 for s in scenario.services)
    
    if max_possible_time == min_possible_time:
        return 1.0
    
    # Linear scoring between min and max
    optimality_score = 1.0 - ((plan.estimated_total_time_minutes - min_possible_time) / 
                              (max_possible_time - min_possible_time))
    
    return max(0.0, min(1.0, optimality_score))

def run_single_benchmark(scenario: DeploymentScenario, algorithm_name: str, planner) -> BenchmarkResult:
    """Run benchmark for a single scenario and algorithm."""
    print(f"  Running {algorithm_name}...", end=" ", flush=True)
    
    start_time = time.time()
    
    try:
        plan, violations = planner.plan_deployment(scenario)
        planning_time = time.time() - start_time
        
        # Evaluate results
        safety_score = evaluate_plan_safety(plan, scenario, violations)
        optimality_score = evaluate_plan_optimality(plan, scenario)
        
        plan_valid = len(violations) == 0 and len(plan.service_order) == len(scenario.services)
        
        # Check if unsafe scenarios are correctly detected
        rollback_detected = False
        if scenario.known_unsafe:
            expected_violations = set(scenario.expected_violations)
            detected_violations = set(violations)
            rollback_detected = bool(expected_violations & detected_violations)
        
        # For safe scenarios: success = produced a valid plan
        # For unsafe scenarios: success = correctly detected the violation
        if scenario.known_unsafe:
            success = rollback_detected
        else:
            success = plan_valid
        
        result = BenchmarkResult(
            scenario_name=scenario.name,
            algorithm=algorithm_name,
            plan_valid=plan_valid,
            constraint_violations=violations,
            total_deployment_time_minutes=plan.estimated_total_time_minutes,
            planning_time_seconds=planning_time,
            safety_score=safety_score,
            optimality_score=optimality_score,
            rollback_detected=rollback_detected,
            success=success
        )
        
        print(f"✓ ({planning_time:.2f}s)")
        return result
        
    except Exception as e:
        planning_time = time.time() - start_time
        print(f"✗ Error: {str(e)}")
        
        return BenchmarkResult(
            scenario_name=scenario.name,
            algorithm=algorithm_name,
            plan_valid=False,
            constraint_violations=[f"execution_error: {str(e)}"],
            total_deployment_time_minutes=float('inf'),
            planning_time_seconds=planning_time,
            safety_score=0.0,
            optimality_score=0.0,
            rollback_detected=False,
            success=False
        )

def run_all_benchmarks() -> List[BenchmarkResult]:
    """Run all benchmarks across all scenarios and algorithms."""
    print("🚀 Starting Safe Deployment Planner SOTA Benchmark")
    print("=" * 60)
    
    scenarios = create_all_scenarios()
    
    algorithms = [
        ("topological_sort", TopologicalSortPlanner()),
        ("random_order", RandomOrderPlanner()),
        ("greedy_resource", GreedyResourcePlanner()),
        ("safe_deploy_z3", SafeDeploymentPlanner()),
    ]
    
    all_results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i:2d}/20] {scenario.name} ({len(scenario.services)} services)")
        if scenario.known_unsafe:
            print(f"        ⚠️  Known unsafe: {scenario.unsafe_reason}")
        
        for algorithm_name, planner in algorithms:
            result = run_single_benchmark(scenario, algorithm_name, planner)
            all_results.append(result)
    
    return all_results

def generate_benchmark_summary(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Generate comprehensive summary statistics."""
    
    # Group results by algorithm
    by_algorithm = {}
    for result in results:
        if result.algorithm not in by_algorithm:
            by_algorithm[result.algorithm] = []
        by_algorithm[result.algorithm].append(result)
    
    summary = {
        "total_scenarios": len(set(r.scenario_name for r in results)),
        "total_algorithms": len(by_algorithm),
        "timestamp": time.time(),
        "algorithm_performance": {}
    }
    
    for algorithm, alg_results in by_algorithm.items():
        valid_results = [r for r in alg_results if r.success]
        
        summary["algorithm_performance"][algorithm] = {
            "total_runs": len(alg_results),
            "successful_runs": len(valid_results),
            "success_rate": len(valid_results) / len(alg_results) if alg_results else 0,
            "avg_safety_score": statistics.mean([r.safety_score for r in alg_results]),
            "avg_optimality_score": statistics.mean([r.optimality_score for r in alg_results]),
            "avg_planning_time_seconds": statistics.mean([r.planning_time_seconds for r in alg_results]),
            "avg_deployment_time_minutes": statistics.mean([r.total_deployment_time_minutes for r in valid_results]) if valid_results else float('inf'),
            "constraint_violations_detected": sum(len(r.constraint_violations) for r in alg_results),
            "unsafe_scenarios_detected": sum(1 for r in alg_results if r.rollback_detected)
        }
    
    return summary

def main():
    """Main benchmark execution."""
    print("Safe Deployment Planner - SOTA Benchmark Suite")
    print("=" * 50)
    
    # Run benchmarks
    results = run_all_benchmarks()
    
    # Generate summary
    summary = generate_benchmark_summary(results)
    
    # Save results
    output = {
        "summary": summary,
        "detailed_results": [asdict(r) for r in results],
        "metadata": {
            "python_version": "3.x",
            "z3_version": "4.x",
            "random_seed": 42,
            "timestamp": time.time()
        }
    }
    
    with open("real_benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    
    for algorithm, perf in summary["algorithm_performance"].items():
        print(f"\n{algorithm.upper()}")
        print(f"  Success Rate:     {perf['success_rate']:.1%}")
        print(f"  Avg Safety:       {perf['avg_safety_score']:.3f}")
        print(f"  Avg Optimality:   {perf['avg_optimality_score']:.3f}")
        print(f"  Avg Plan Time:    {perf['avg_planning_time_seconds']:.3f}s")
        if perf['avg_deployment_time_minutes'] != float('inf'):
            print(f"  Avg Deploy Time:  {perf['avg_deployment_time_minutes']:.1f}min")
        print(f"  Violations Found: {perf['constraint_violations_detected']}")
        print(f"  Unsafe Detected:  {perf['unsafe_scenarios_detected']}")
    
    print(f"\n💾 Detailed results saved to: real_benchmark_results.json")
    return output

if __name__ == "__main__":
    main()