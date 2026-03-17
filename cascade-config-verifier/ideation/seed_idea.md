Community: area-067-distributed-systems-and-cloud-infrastruc

Idea: An SMT-based static analyzer that ingests microservice resilience configurations (retry policies, timeouts, circuit breakers, resource limits) from Kubernetes/Istio/Envoy manifests and automatically discovers minimal component-failure sets that trigger cascading outages via retry-amplification and timeout-chain modeling, then synthesizes provably-safe parameter repairs.
