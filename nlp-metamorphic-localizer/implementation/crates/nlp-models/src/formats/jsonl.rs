//! JSONL (line-delimited JSON) reader and writer for NLP datasets.
//!
//! Each line in a JSONL file is a self-contained JSON object.  The canonical
//! schema expected by this module requires at minimum a `"text"` field; the
//! optional fields `"labels"`, `"entities"`, and `"metadata"` are recognised
//! when present.

use serde::{Deserialize, Serialize};
use shared_types::{LocalizerError, Result};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;

// ── Record type ─────────────────────────────────────────────────────────────

/// A single record in a JSONL dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonlRecord {
    /// The raw input text (required).
    pub text: String,

    /// Optional classification / annotation labels.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,

    /// Optional entity annotations (start, end, label).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<JsonlEntity>,

    /// Arbitrary key-value metadata.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// An entity span inside a [`JsonlRecord`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonlEntity {
    pub start: usize,
    pub end: usize,
    pub label: String,
}

impl JsonlRecord {
    /// Create a minimal record with only text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            labels: Vec::new(),
            entities: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Builder: attach labels.
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = labels;
        self
    }

    /// Builder: attach entity annotations.
    pub fn with_entities(mut self, entities: Vec<JsonlEntity>) -> Self {
        self.entities = entities;
        self
    }

    /// Builder: attach a single metadata key-value pair.
    pub fn with_meta(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

// ── Reader ──────────────────────────────────────────────────────────────────

/// Reads JSONL data from files, strings, or any [`BufRead`] source.
///
/// Supports streaming via [`JsonlReader::lines`] for memory-efficient processing
/// of large datasets.
pub struct JsonlReader;

impl JsonlReader {
    /// Read all records from a JSONL file.
    pub fn read_file(path: impl AsRef<Path>) -> Result<Vec<JsonlRecord>> {
        let file = std::fs::File::open(path.as_ref())?;
        let reader = std::io::BufReader::new(file);
        Self::read(reader)
    }

    /// Read all records from a [`BufRead`] source.
    pub fn read<R: BufRead>(reader: R) -> Result<Vec<JsonlRecord>> {
        let mut records = Vec::new();
        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let record: JsonlRecord = serde_json::from_str(trimmed).map_err(|e| {
                LocalizerError::validation(
                    "jsonl",
                    format!("line {}: {}", line_num + 1, e),
                )
            })?;
            records.push(record);
        }
        Ok(records)
    }

    /// Parse all records from a string.
    pub fn read_string(text: &str) -> Result<Vec<JsonlRecord>> {
        Self::read(std::io::Cursor::new(text))
    }

    /// Return a streaming iterator over records from a [`BufRead`] source.
    ///
    /// Each call to `next()` reads and parses exactly one line, keeping memory
    /// usage constant regardless of file size.
    pub fn lines<R: BufRead>(reader: R) -> JsonlLineIter<R> {
        JsonlLineIter {
            reader,
            line_num: 0,
            buf: String::new(),
        }
    }
}

/// Streaming iterator returned by [`JsonlReader::lines`].
pub struct JsonlLineIter<R> {
    reader: R,
    line_num: usize,
    buf: String,
}

impl<R: BufRead> Iterator for JsonlLineIter<R> {
    type Item = Result<JsonlRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.buf.clear();
            match self.reader.read_line(&mut self.buf) {
                Ok(0) => return None,
                Ok(_) => {
                    self.line_num += 1;
                    let trimmed = self.buf.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let result = serde_json::from_str::<JsonlRecord>(trimmed).map_err(|e| {
                        LocalizerError::validation(
                            "jsonl",
                            format!("line {}: {}", self.line_num, e),
                        )
                    });
                    return Some(result);
                }
                Err(e) => return Some(Err(LocalizerError::IoError(e))),
            }
        }
    }
}

// ── Writer ──────────────────────────────────────────────────────────────────

/// Writes [`JsonlRecord`]s to files, strings, or any [`Write`] sink.
pub struct JsonlWriter;

impl JsonlWriter {
    /// Write records to a file on disk.
    pub fn write_file(path: impl AsRef<Path>, records: &[JsonlRecord]) -> Result<()> {
        let file = std::fs::File::create(path.as_ref())?;
        let mut writer = std::io::BufWriter::new(file);
        Self::write(&mut writer, records)
    }

    /// Write records to any [`Write`] sink.
    pub fn write<W: Write>(writer: &mut W, records: &[JsonlRecord]) -> Result<()> {
        for record in records {
            let json = serde_json::to_string(record).map_err(|e| {
                LocalizerError::validation("jsonl", format!("serialization failed: {e}"))
            })?;
            writeln!(writer, "{}", json)?;
        }
        Ok(())
    }

    /// Render records to a [`String`].
    pub fn write_string(records: &[JsonlRecord]) -> Result<String> {
        let mut buf = Vec::new();
        Self::write(&mut buf, records)?;
        String::from_utf8(buf)
            .map_err(|e| LocalizerError::validation("jsonl", format!("non-UTF-8 output: {e}")))
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_JSONL: &str = r#"{"text":"The cat sat on the mat.","labels":["declarative"],"entities":[{"start":4,"end":7,"label":"ANIMAL"}],"metadata":{"source":"test"}}
{"text":"Dogs run fast."}
{"text":"Birds fly.","labels":["declarative","short"]}
"#;

    #[test]
    fn test_read_records() {
        let records = JsonlReader::read_string(SAMPLE_JSONL).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].text, "The cat sat on the mat.");
        assert_eq!(records[0].labels, vec!["declarative"]);
        assert_eq!(records[0].entities.len(), 1);
        assert_eq!(records[0].entities[0].label, "ANIMAL");
        assert!(records[0].metadata.contains_key("source"));

        assert_eq!(records[1].text, "Dogs run fast.");
        assert!(records[1].labels.is_empty());
    }

    #[test]
    fn test_roundtrip() {
        let records = JsonlReader::read_string(SAMPLE_JSONL).unwrap();
        let output = JsonlWriter::write_string(&records).unwrap();
        let reparsed = JsonlReader::read_string(&output).unwrap();

        assert_eq!(records.len(), reparsed.len());
        for (a, b) in records.iter().zip(reparsed.iter()) {
            assert_eq!(a.text, b.text);
            assert_eq!(a.labels, b.labels);
            assert_eq!(a.entities.len(), b.entities.len());
        }
    }

    #[test]
    fn test_streaming_lines() {
        let cursor = std::io::Cursor::new(SAMPLE_JSONL);
        let records: Vec<JsonlRecord> = JsonlReader::lines(cursor)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[2].text, "Birds fly.");
    }

    #[test]
    fn test_builder() {
        let record = JsonlRecord::new("hello world")
            .with_labels(vec!["greeting".into()])
            .with_entities(vec![JsonlEntity {
                start: 0,
                end: 5,
                label: "SALUTATION".into(),
            }])
            .with_meta("lang", serde_json::Value::String("en".into()));

        assert_eq!(record.text, "hello world");
        assert_eq!(record.labels, vec!["greeting"]);
        assert_eq!(record.entities.len(), 1);
        assert!(record.metadata.contains_key("lang"));
    }

    #[test]
    fn test_empty_input() {
        let records = JsonlReader::read_string("").unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn test_blank_lines_skipped() {
        let input = "{\"text\":\"a\"}\n\n{\"text\":\"b\"}\n";
        let records = JsonlReader::read_string(input).unwrap();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_invalid_json_error() {
        let result = JsonlReader::read_string("not json\n");
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_text_field_error() {
        let result = JsonlReader::read_string("{\"labels\":[\"a\"]}\n");
        assert!(result.is_err());
    }
}
