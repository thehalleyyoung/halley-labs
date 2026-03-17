//! # cascade-service
//!
//! Service dependency modelling and mesh topology handling for the
//! CascadeVerify project.  Builds on [`cascade_types`] and
//! [`cascade_graph`] to provide higher-level abstractions for
//! analysing resilience, capacity, and failure-propagation behaviour
//! of micro-service meshes.

pub mod capacity;
pub mod dependency;
pub mod mesh;
pub mod resilience;
pub mod simulation;
pub mod topology_patterns;

pub use capacity::{
    CapacityBottleneck, CapacityEstimate, CapacityModel, CapacityPlanner, CapacitySource,
    LoadModel,
};
pub use dependency::{
    CriticalDependency, DependencyCycle, DependencyAnalyzer, DependencyGraph,
    DependencyImpact, DependencyMetrics, ImpactAssessment,
};
pub use mesh::{
    BlastRadius, CriticalPath, MeshBuilder, MeshHealthModel, MeshPolicies, MeshTopology,
    MeshValidator, ServiceMesh, ServiceTier, TrafficModel,
};
pub use resilience::{
    BestPracticeChecker, EffectiveRetryBudget, EffectiveTimeout, ResilienceAnalyzer,
    ResilienceScore, RetryAnalysis, RetryAssessment, TimeoutAnalysis, TimeoutAssessment,
    TimeoutChain,
};
pub use simulation::{
    FailureEvent, MeshSimulator, MeshState, SimEvent, SimulationConfig, SimulationResult,
    SimulationStep, SimulationTrace,
};
pub use topology_patterns::{
    ChainRisk, PatternAnalyzer, PatternDetector, PatternGenerator, StarRisk, TopologyPattern,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Re-export smoke tests -----------------------------------------------

    #[test]
    fn test_capacity_model_reexport() {
        let _model = CapacityModel;
        let _ = format!("{:?}", "CapacityModel");
    }

    #[test]
    fn test_dependency_analyzer_reexport() {
        let _analyzer = DependencyAnalyzer;
        let _ = format!("{:?}", "DependencyAnalyzer");
    }

    #[test]
    fn test_resilience_score_reexport() {
        let score = ResilienceScore {
            overall: 0.8,
            retry_score: 0.9,
            timeout_score: 0.7,
            circuit_breaker_score: 0.6,
            summary: "Good".into(),
        };
        assert!(score.overall > 0.0 && score.overall <= 1.0);
    }

    #[test]
    fn test_simulation_config_reexport() {
        let config = SimulationConfig::default();
        let _ = format!("{:?}", config);
    }

    #[test]
    fn test_topology_pattern_variants() {
        let chain = TopologyPattern::Chain;
        let _ = format!("{:?}", chain);
    }

    #[test]
    fn test_service_tier_ordering() {
        assert!(ServiceTier::Critical as u32 > ServiceTier::Standard as u32);
    }

    #[test]
    fn test_mesh_builder_reexport() {
        let _builder = MeshBuilder::new();
    }

    #[test]
    fn test_pattern_detector_reexport() {
        let _detector = PatternDetector;
    }

    #[test]
    fn test_effective_timeout_reexport() {
        let et = EffectiveTimeout {
            local_timeout_ms: 5000,
            chain_timeout_ms: 3000,
            deadline_ms: 5000,
            feasible: true,
        };
        assert_eq!(et.local_timeout_ms, 5000);
    }

    #[test]
    fn test_blast_radius_reexport() {
        let br = BlastRadius {
            directly_affected: vec!["svc-b".into()],
            transitively_affected: vec!["svc-c".into()],
            impact_score: 0.5,
        };
        assert_eq!(br.directly_affected.len(), 1);
    }
}
