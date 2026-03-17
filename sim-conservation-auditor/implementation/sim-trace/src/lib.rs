pub mod recorder;
pub mod serializer;
pub mod replay;
pub mod compress;
pub mod checkpoint;
pub mod query;
pub mod format;
pub mod diff;
pub mod filter;
pub mod events;
pub mod vtk;
pub mod hdf5;

pub use recorder::{TraceRecorder, RecordingPolicy, AdaptiveRecording, TraceMetadata};
pub use serializer::{JsonSerializer, BinarySerializer, CsvSerializer, TraceSerializer};
pub use replay::{TracePlayer, PlaybackMode, PlaybackSpeed};
pub use compress::{
    DeltaCompression, QuantizationCompression, KeyframeCompression,
    RunLengthEncoding, LossyCompression, CompressionStats,
};
pub use checkpoint::{Checkpoint, CheckpointManager};
pub use query::{TraceQuery, TimeRangeQuery, ParticleQuery, ConservationQuery, MaxViolationQuery, AggregateQuery};
pub use format::{TraceFileHeader, TraceFileIndex, TraceFormat, FormatVersion};
pub use diff::{TraceDiff, StateDiff, ParticleDiff, DiffStatistics};
pub use filter::{
    LowPassFilter, MovingAverageFilter, SavitzkyGolayFilter, MedianFilter,
    DownsampleFilter, FilterPipeline, TraceFilter,
};
pub use events::{TraceEvent, EventKind, EventLog, EventQuery};
pub use vtk::{VtkWriter, VtkReader, VtkFormat, VtkParticleData, VtkFieldData};
pub use hdf5::{Hdf5Writer, Hdf5Reader, Hdf5Trajectory, Hdf5Metadata, DatasetMeta};
