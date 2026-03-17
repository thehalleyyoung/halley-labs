//! Trace serialization formats.

/// Trace serializer trait.
pub trait TraceSerializer { fn serialize(&self, times: &[f64], states: &[Vec<f64>]) -> Vec<u8>; fn name(&self) -> &str; }

/// JSON serializer.
#[derive(Debug, Clone, Default)]
pub struct JsonSerializer;
impl TraceSerializer for JsonSerializer {
    fn serialize(&self, times: &[f64], states: &[Vec<f64>]) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({"times": times, "states": states})).unwrap_or_default()
    }
    fn name(&self) -> &str { "JSON" }
}

/// Binary serializer.
#[derive(Debug, Clone, Default)]
pub struct BinarySerializer;
impl TraceSerializer for BinarySerializer {
    fn serialize(&self, times: &[f64], _states: &[Vec<f64>]) -> Vec<u8> {
        let mut buf = Vec::new();
        for t in times { buf.extend_from_slice(&t.to_le_bytes()); }
        buf
    }
    fn name(&self) -> &str { "Binary" }
}

/// CSV serializer.
#[derive(Debug, Clone, Default)]
pub struct CsvSerializer;
impl TraceSerializer for CsvSerializer {
    fn serialize(&self, times: &[f64], states: &[Vec<f64>]) -> Vec<u8> {
        let mut out = String::new();
        for (t, s) in times.iter().zip(states) {
            out.push_str(&format!("{}", t));
            for v in s { out.push_str(&format!(",{}", v)); }
            out.push('\n');
        }
        out.into_bytes()
    }
    fn name(&self) -> &str { "CSV" }
}
