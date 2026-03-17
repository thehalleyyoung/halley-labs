//! Real-time streaming infrastructure for continuous data sonification.
//!
//! This crate provides lock-free buffers, a streaming pipeline architecture,
//! transport control, data stream management, real-time scheduling, and
//! performance monitoring for the SoniType perceptual sonification compiler.

pub mod buffer;
pub mod data_stream;
pub mod monitor;
pub mod pipeline;
pub mod scheduling;
pub mod transport;
pub mod csv_input;
pub mod hdf5_input;

pub use buffer::{AudioRingBuffer, EventQueue, OverflowPolicy, RingBuffer, TripleBuffer};
pub use data_stream::{DataRateAdapter, DataStream, DataStreamMultiplexer, InterpolationMethod};
pub use monitor::{HealthCheck, HealthStatus, PerformanceLog, StreamMonitor};
pub use pipeline::{
    DataInputStage, MappingStage, OutputStage, PipelineBuilder, PipelineStage, RenderStage,
    StreamingPipeline,
};
pub use scheduling::{CallbackTimer, LoadBalancer, RealtimeScheduler};
pub use transport::{Timeline, TransportController, TransportEvent, TransportState};
pub use csv_input::{CsvConfig, CsvDataSource, CsvColumn, CsvInputError, ColumnType, ColumnStats};
pub use hdf5_input::{Hdf5DataSource, Hdf5FileMeta, Hdf5DatasetMeta, Hdf5InputError};
