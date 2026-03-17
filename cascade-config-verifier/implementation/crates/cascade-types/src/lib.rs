//! # cascade-types
//!
//! Foundational shared types for the CascadeVerify project.
//! Provides service definitions, resilience policies, topology structures,
//! configuration sources, cascade analysis types, repair actions,
//! reporting formats, SMT expression types, and common error types.

pub mod cascade;
pub mod config;
pub mod errors;
pub mod policy;
pub mod repair;
pub mod report;
pub mod service;
pub mod smt;
pub mod topology;

pub use cascade::{
    CascadeClassification, CascadeMetrics, CascadeResult, CascadeScenario, CascadeSeverity,
    FailureMode, FailureSet, MinimalFailureSet, PropagationStep, PropagationTrace,
};
pub use config::{
    ConfigManifest, ConfigSource, ConfigValidationError, ConfigWarning, EnvoyConfig,
    HelmValues, IstioConfig, KubernetesConfig, KustomizeOverlay,
};
pub use errors::{
    AnalysisError, CascadeError, ConfigError, ParseError, RepairError, SmtError,
    ValidationError,
};
pub use policy::{
    BackoffStrategy, BulkheadPolicy, CircuitBreakerPolicy, PolicyMerger, PolicyPrecedence,
    PolicySource, RateLimitPolicy, ResiliencePolicy, RetryCondition, RetryPolicy,
    TimeoutPolicy,
};
pub use repair::{
    ConfigDiff, ParameterChange, ParetoFrontier, RepairAction, RepairCandidate,
    RepairConstraint, RepairObjective, RepairPlan, RepairValidation,
};
pub use report::{
    AnalysisReport, Evidence, Finding, HumanReadableReport, JUnitReport, Location,
    ReportFormat, SarifReport, Severity,
};
pub use service::{
    Protocol, Service, ServiceEndpoint, ServiceHealth, ServiceId, ServiceMetadata,
    ServiceName, ServiceNamespace, ServicePort, ServiceSpec, ServiceState, ServiceType,
};
pub use smt::{
    SmtConstraint, SmtExpr, SmtFormula, SmtModel, SmtResult, SmtSort, SmtVariable,
    VariableEncoding,
};
pub use topology::{
    DependencyEdge, DependencyType, EdgeId, EdgeWeight, FanInInfo, FanOutInfo, PathInfo,
    ServiceTopology, TopologyMetadata, TopologyStats,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Re-export smoke tests -----------------------------------------------
    // Verify that the most important re-exported types are accessible and
    // behave as expected.

    #[test]
    fn test_cascade_severity_variants() {
        let low = CascadeSeverity::Low;
        let med = CascadeSeverity::Medium;
        let high = CascadeSeverity::High;
        let crit = CascadeSeverity::Critical;
        // Just ensure they exist and Debug works
        let _ = format!("{:?} {:?} {:?} {:?}", low, med, high, crit);
    }

    #[test]
    fn test_service_id_creation() {
        let id = service::ServiceId::new("my-service");
        assert_eq!(id.as_str(), "my-service");
        assert!(!id.is_empty());
        assert_eq!(format!("{id}"), "my-service");
    }

    #[test]
    fn test_service_id_from_str() {
        let id: service::ServiceId = "test-svc".into();
        assert_eq!(id.as_str(), "test-svc");
    }

    #[test]
    fn test_service_id_from_string() {
        let id: service::ServiceId = String::from("svc").into();
        assert_eq!(id.0, "svc");
    }

    #[test]
    fn test_service_id_empty() {
        let id = service::ServiceId::new("");
        assert!(id.is_empty());
    }

    #[test]
    fn test_service_id_as_ref() {
        let id = service::ServiceId::new("hello");
        let r: &str = id.as_ref();
        assert_eq!(r, "hello");
    }

    #[test]
    fn test_service_id_hash_and_eq() {
        use std::collections::HashSet;
        let a = service::ServiceId::new("abc");
        let b = service::ServiceId::new("abc");
        let c = service::ServiceId::new("xyz");
        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
        assert!(!set.contains(&c));
    }

    #[test]
    fn test_service_id_ordering() {
        let a = service::ServiceId::new("aaa");
        let b = service::ServiceId::new("zzz");
        assert!(a < b);
    }

    // -- Policy types -------------------------------------------------------

    #[test]
    fn test_retry_policy_default() {
        let rp = policy::RetryPolicy::default();
        assert_eq!(rp.max_retries, 3);
        assert_eq!(rp.amplification_factor(), 4);
    }

    #[test]
    fn test_retry_policy_builder() {
        let rp = policy::RetryPolicy::new(5)
            .with_per_try_timeout(3000)
            .with_retry_budget(0.2);
        assert_eq!(rp.max_retries, 5);
        assert_eq!(rp.per_try_timeout_ms, 3000);
        assert_eq!(rp.retry_budget, Some(0.2));
    }

    #[test]
    fn test_retry_policy_noop() {
        let rp = policy::RetryPolicy::new(0);
        assert!(rp.is_noop());
        assert_eq!(rp.amplification_factor(), 1);
    }

    #[test]
    fn test_retry_policy_validation_ok() {
        let rp = policy::RetryPolicy::new(3);
        assert!(rp.validate().is_empty());
    }

    #[test]
    fn test_retry_policy_validation_high_retries() {
        let rp = policy::RetryPolicy::new(30);
        let errors = rp.validate();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_retry_policy_worst_case_latency() {
        let rp = policy::RetryPolicy::new(2).with_per_try_timeout(1000);
        let lat = rp.worst_case_latency_ms();
        // 3 attempts * 1000ms + backoff for 2 retries
        assert!(lat >= 3000);
    }

    #[test]
    fn test_timeout_policy_default() {
        let tp = policy::TimeoutPolicy::default();
        assert_eq!(tp.request_timeout_ms, 30_000);
    }

    #[test]
    fn test_timeout_policy_consistency() {
        let tp = policy::TimeoutPolicy::new(5000).with_per_try_timeout(3000);
        assert!(tp.is_consistent());

        let tp2 = policy::TimeoutPolicy::new(5000).with_per_try_timeout(10000);
        assert!(!tp2.is_consistent());
    }

    #[test]
    fn test_timeout_policy_effective_per_try() {
        let tp = policy::TimeoutPolicy::new(5000);
        assert_eq!(tp.effective_per_try_timeout_ms(), 5000);

        let tp2 = policy::TimeoutPolicy::new(5000).with_per_try_timeout(2000);
        assert_eq!(tp2.effective_per_try_timeout_ms(), 2000);
    }

    #[test]
    fn test_resilience_policy_empty() {
        let rp = policy::ResiliencePolicy::empty();
        assert!(!rp.has_any());
        assert_eq!(rp.amplification_factor(), 1);
    }

    #[test]
    fn test_resilience_policy_with_retry() {
        let rp = policy::ResiliencePolicy::empty()
            .with_retry(policy::RetryPolicy::new(3));
        assert!(rp.has_any());
        assert_eq!(rp.amplification_factor(), 4);
    }

    #[test]
    fn test_resilience_policy_validation() {
        let rp = policy::ResiliencePolicy::empty()
            .with_retry(policy::RetryPolicy::new(3));
        let errors = rp.validate();
        assert!(errors.is_empty());
    }

    // -- Backoff strategy ---------------------------------------------------

    #[test]
    fn test_backoff_fixed() {
        let bs = policy::BackoffStrategy::Fixed { delay_ms: 100 };
        assert_eq!(bs.delay_for_attempt(0), 100);
        assert_eq!(bs.delay_for_attempt(5), 100);
        assert_eq!(bs.total_delay(3), 300);
    }

    #[test]
    fn test_backoff_exponential() {
        let bs = policy::BackoffStrategy::Exponential {
            base_ms: 100,
            max_ms: 10000,
        };
        assert_eq!(bs.delay_for_attempt(0), 100);
        assert_eq!(bs.delay_for_attempt(1), 200);
        assert_eq!(bs.delay_for_attempt(2), 400);
        // Should cap at max_ms
        assert!(bs.delay_for_attempt(20) <= 10000);
    }

    #[test]
    fn test_backoff_linear() {
        let bs = policy::BackoffStrategy::Linear { increment_ms: 50 };
        assert_eq!(bs.delay_for_attempt(0), 50);
        assert_eq!(bs.delay_for_attempt(1), 100);
        assert_eq!(bs.delay_for_attempt(2), 150);
    }

    #[test]
    fn test_backoff_display() {
        let fixed = policy::BackoffStrategy::Fixed { delay_ms: 100 };
        let s = format!("{fixed}");
        assert!(s.contains("100ms"));

        let exp = policy::BackoffStrategy::default();
        let s = format!("{exp}");
        assert!(s.contains("exponential"));
    }

    // -- RetryCondition -----------------------------------------------------

    #[test]
    fn test_retry_condition_safety() {
        assert!(policy::RetryCondition::ConnectFailure.is_always_safe());
        assert!(policy::RetryCondition::Reset.is_always_safe());
        assert!(!policy::RetryCondition::ServerError.is_always_safe());
    }

    #[test]
    fn test_retry_condition_status_based() {
        assert!(policy::RetryCondition::ServerError.is_status_based());
        assert!(policy::RetryCondition::GatewayError.is_status_based());
        assert!(!policy::RetryCondition::Reset.is_status_based());
    }

    #[test]
    fn test_retry_condition_display() {
        assert_eq!(format!("{}", policy::RetryCondition::ServerError), "5xx");
        assert_eq!(
            format!("{}", policy::RetryCondition::ConnectFailure),
            "connect-failure"
        );
    }

    // -- CircuitBreaker & RateLimit -----------------------------------------

    #[test]
    fn test_circuit_breaker_default() {
        let cb = policy::CircuitBreakerPolicy::default();
        assert_eq!(cb.max_connections, 1024);
        assert_eq!(cb.consecutive_errors, 5);
        assert!(!cb.is_strict());
    }

    #[test]
    fn test_circuit_breaker_strict() {
        let cb = policy::CircuitBreakerPolicy::new()
            .with_consecutive_errors(2)
            .with_max_ejection_percent(60.0);
        assert!(cb.is_strict());
    }

    #[test]
    fn test_circuit_breaker_validation() {
        let mut cb = policy::CircuitBreakerPolicy::default();
        assert!(cb.validate().is_empty());
        cb.max_connections = 0;
        assert!(!cb.validate().is_empty());
    }

    // -- Error types --------------------------------------------------------

    #[test]
    fn test_cascade_error_debug() {
        let e = CascadeError::Analysis("test error".into());
        let s = format!("{:?}", e);
        assert!(s.contains("test error"));
    }

    // -- Serde cross-type ---------------------------------------------------

    #[test]
    fn test_smt_sort_variants() {
        let sorts = vec![SmtSort::Bool, SmtSort::Int, SmtSort::Real];
        for s in sorts {
            let json = serde_json::to_string(&s).unwrap();
            let deser: SmtSort = serde_json::from_str(&json).unwrap();
            assert_eq!(deser, s);
        }
    }

    #[test]
    fn test_smt_variable_creation() {
        let v = SmtVariable {
            name: "x".into(),
            sort: SmtSort::Int,
        };
        assert_eq!(v.name, "x");
        assert_eq!(v.sort, SmtSort::Int);
    }
}
