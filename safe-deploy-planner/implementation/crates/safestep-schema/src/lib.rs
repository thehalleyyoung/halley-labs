//! safestep-schema: API schema parsing and compatibility analysis.
//!
//! Parses OpenAPI, Protobuf, GraphQL, and Avro schemas,
//! computes diffs, and produces compatibility predicates
//! used by the SafeStep planning engine.

pub mod openapi;
pub mod openapi_diff;
pub mod protobuf;
pub mod protobuf_diff;
pub mod graphql;
pub mod avro;
pub mod unified;
pub mod semver_analysis;
pub mod confidence;

pub use openapi::{OpenApiSchema, OpenApiPath, OpenApiOperation, SchemaObject, OpenApiParser};
pub use openapi_diff::{OpenApiDiff, SchemaDiff, BreakingChangeDetector, ChangeClassification, SchemaEvolution};
pub use protobuf::{ProtobufSchema, ProtoMessage, ProtoField, ProtoService, ProtoMethod, ProtobufParser};
pub use protobuf_diff::{ProtobufDiff, ProtoDiffResult, ProtoBreakingChange, WireCompatibility};
pub use graphql::{GraphqlSchema, GraphqlType, GraphqlField, GraphqlParser, GraphqlDiff, GraphqlBreakingChange};
pub use avro::{AvroSchema, AvroRecord, AvroField, AvroCompatibility, AvroDiff};
pub use unified::{
    UnifiedSchema, UnifiedEndpoint, UnifiedType, UnifiedDiff,
    CompatibilityClassifier, CompatibilityPredicate, SchemaRegistry,
};
pub use semver_analysis::{SemverAnalyzer, VersionRangeAnalyzer, DeprecationTracker};
pub use confidence::{ConfidenceScore, ConfidenceModel, RedTagging};
