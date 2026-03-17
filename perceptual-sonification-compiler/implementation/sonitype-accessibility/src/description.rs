//! Audio description generation: textual descriptions of sonification
//! mappings, data narration, and legend/key generation.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// StreamMapping description
// ---------------------------------------------------------------------------

/// Describes the mapping of a single sonification stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMappingDesc {
    pub stream_name: String,
    pub data_field: String,
    pub parameter: String,
    pub low_label: String,
    pub high_label: String,
    pub unit: String,
    pub data_min: f64,
    pub data_max: f64,
}

// ---------------------------------------------------------------------------
// AudioDescriptionGenerator
// ---------------------------------------------------------------------------

/// Generates human-readable text descriptions of sonification configurations.
pub struct AudioDescriptionGenerator {
    mappings: Vec<StreamMappingDesc>,
    title: String,
    preamble: String,
}

impl AudioDescriptionGenerator {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            mappings: Vec::new(),
            title: title.into(),
            preamble: String::new(),
        }
    }

    pub fn set_preamble(&mut self, text: impl Into<String>) {
        self.preamble = text.into();
    }

    pub fn add_mapping(&mut self, desc: StreamMappingDesc) {
        self.mappings.push(desc);
    }

    pub fn clear_mappings(&mut self) {
        self.mappings.clear();
    }

    pub fn mapping_count(&self) -> usize {
        self.mappings.len()
    }

    /// Generate a complete description of all active mappings.
    pub fn generate_full_description(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("=== {} ===\n", self.title));
        if !self.preamble.is_empty() {
            out.push_str(&self.preamble);
            out.push('\n');
        }
        out.push('\n');
        if self.mappings.is_empty() {
            out.push_str("No active sonification mappings.\n");
            return out;
        }
        out.push_str(&format!(
            "This sonification has {} active stream(s):\n\n",
            self.mappings.len()
        ));
        for (i, m) in self.mappings.iter().enumerate() {
            out.push_str(&format!("  Stream {}: \"{}\"\n", i + 1, m.stream_name));
            out.push_str(&format!(
                "    {} represents {}: {} = {}, {} = {}\n",
                m.parameter,
                m.data_field,
                m.low_label,
                m.data_min,
                m.high_label,
                m.data_max
            ));
            if !m.unit.is_empty() {
                out.push_str(&format!("    Unit: {}\n", m.unit));
            }
            out.push('\n');
        }
        out
    }

    /// Generate a short one-line summary per mapping.
    pub fn generate_summary(&self) -> String {
        if self.mappings.is_empty() {
            return "No active mappings.".to_string();
        }
        self.mappings
            .iter()
            .map(|m| {
                format!(
                    "{} maps {} to {} ({} – {})",
                    m.stream_name, m.data_field, m.parameter, m.low_label, m.high_label
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// Describe a single mapping as a sentence.
    pub fn describe_mapping(&self, index: usize) -> Option<String> {
        self.mappings.get(index).map(|m| {
            format!(
                "{} represents {}: {} = {} ({:.1}), {} = {} ({:.1}).",
                m.parameter,
                m.data_field,
                m.low_label,
                m.data_min,
                m.data_min,
                m.high_label,
                m.data_max,
                m.data_max
            )
        })
    }
}

// ---------------------------------------------------------------------------
// DataNarrator
// ---------------------------------------------------------------------------

/// Trend direction for narration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Trend {
    Rising,
    Falling,
    Stable,
    Volatile,
}

/// Generates spoken-text-like descriptions of data values, trends, and outliers.
pub struct DataNarrator {
    field_name: String,
    unit: String,
    history: Vec<f64>,
    max_history: usize,
    outlier_threshold_sigma: f64,
}

impl DataNarrator {
    pub fn new(field_name: impl Into<String>, unit: impl Into<String>) -> Self {
        Self {
            field_name: field_name.into(),
            unit: unit.into(),
            history: Vec::new(),
            max_history: 100,
            outlier_threshold_sigma: 2.0,
        }
    }

    pub fn set_outlier_threshold(&mut self, sigma: f64) {
        self.outlier_threshold_sigma = sigma.max(0.5);
    }

    pub fn push_value(&mut self, value: f64) {
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(value);
    }

    /// Narrate the current value.
    pub fn narrate_current(&self) -> String {
        match self.history.last() {
            Some(v) => format!("{} is {:.2} {}.", self.field_name, v, self.unit),
            None => format!("No data for {}.", self.field_name),
        }
    }

    /// Narrate a trend over recent values.
    pub fn narrate_trend(&self) -> String {
        let trend = self.detect_trend();
        let desc = match trend {
            Trend::Rising => "rising",
            Trend::Falling => "falling",
            Trend::Stable => "stable",
            Trend::Volatile => "volatile",
        };
        if self.history.len() < 2 {
            return format!("{} has insufficient data for trend.", self.field_name);
        }
        let first = self.history.first().unwrap();
        let last = self.history.last().unwrap();
        format!(
            "{} is {} from {:.2} to {:.2} {}.",
            self.field_name, desc, first, last, self.unit
        )
    }

    /// Detect the current trend.
    pub fn detect_trend(&self) -> Trend {
        if self.history.len() < 3 {
            return Trend::Stable;
        }
        let n = self.history.len();
        let recent = &self.history[n.saturating_sub(10)..];
        let diffs: Vec<f64> = recent.windows(2).map(|w| w[1] - w[0]).collect();
        if diffs.is_empty() {
            return Trend::Stable;
        }
        let mean_diff: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let variance: f64 =
            diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / diffs.len() as f64;
        let stddev = variance.sqrt();

        let value_range = self.range();
        let threshold = value_range * 0.02;

        if stddev > value_range * 0.1 {
            Trend::Volatile
        } else if mean_diff > threshold {
            Trend::Rising
        } else if mean_diff < -threshold {
            Trend::Falling
        } else {
            Trend::Stable
        }
    }

    /// Check if the latest value is an outlier.
    pub fn is_outlier(&self) -> bool {
        if self.history.len() < 5 {
            return false;
        }
        let mean = self.mean();
        let sd = self.stddev();
        if sd < 1e-12 {
            return false;
        }
        let last = *self.history.last().unwrap();
        ((last - mean) / sd).abs() > self.outlier_threshold_sigma
    }

    /// Announce an outlier if detected.
    pub fn announce_outlier(&self) -> Option<String> {
        if self.is_outlier() {
            let last = self.history.last().unwrap();
            Some(format!(
                "Alert: {} has an unusual value of {:.2} {}.",
                self.field_name, last, self.unit
            ))
        } else {
            None
        }
    }

    fn mean(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history.iter().sum::<f64>() / self.history.len() as f64
    }

    fn stddev(&self) -> f64 {
        let m = self.mean();
        let var = self
            .history
            .iter()
            .map(|v| (v - m).powi(2))
            .sum::<f64>()
            / self.history.len() as f64;
        var.sqrt()
    }

    fn range(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let min = self.history.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self
            .history
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        max - min
    }

    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    pub fn clear(&mut self) {
        self.history.clear();
    }
}

// ---------------------------------------------------------------------------
// LegendGenerator
// ---------------------------------------------------------------------------

/// Output format for the legend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegendFormat {
    PlainText,
    Html,
    Json,
}

/// Scale description within the legend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleEntry {
    pub parameter: String,
    pub data_field: String,
    pub data_range: (f64, f64),
    pub perceptual_range: String,
    pub description: String,
}

/// Generates a human-readable legend/key for a sonification.
pub struct LegendGenerator {
    title: String,
    entries: Vec<ScaleEntry>,
    notes: Vec<String>,
}

impl LegendGenerator {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            entries: Vec::new(),
            notes: Vec::new(),
        }
    }

    pub fn add_entry(&mut self, entry: ScaleEntry) {
        self.entries.push(entry);
    }

    pub fn add_note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Render the legend in the given format.
    pub fn render(&self, format: LegendFormat) -> String {
        match format {
            LegendFormat::PlainText => self.render_text(),
            LegendFormat::Html => self.render_html(),
            LegendFormat::Json => self.render_json(),
        }
    }

    fn render_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("Legend: {}\n", self.title));
        out.push_str(&"─".repeat(40));
        out.push('\n');
        for e in &self.entries {
            out.push_str(&format!(
                "  {} → {} [{:.1} – {:.1}] ({})\n",
                e.data_field, e.parameter, e.data_range.0, e.data_range.1, e.perceptual_range
            ));
            if !e.description.is_empty() {
                out.push_str(&format!("    {}\n", e.description));
            }
        }
        if !self.notes.is_empty() {
            out.push('\n');
            out.push_str("Notes:\n");
            for n in &self.notes {
                out.push_str(&format!("  • {}\n", n));
            }
        }
        out
    }

    fn render_html(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("<h3>{}</h3>\n<table>\n", self.title));
        out.push_str("  <tr><th>Data Field</th><th>Parameter</th><th>Range</th><th>Description</th></tr>\n");
        for e in &self.entries {
            out.push_str(&format!(
                "  <tr><td>{}</td><td>{}</td><td>{:.1} – {:.1} ({})</td><td>{}</td></tr>\n",
                e.data_field,
                e.parameter,
                e.data_range.0,
                e.data_range.1,
                e.perceptual_range,
                e.description
            ));
        }
        out.push_str("</table>\n");
        if !self.notes.is_empty() {
            out.push_str("<ul>\n");
            for n in &self.notes {
                out.push_str(&format!("  <li>{}</li>\n", n));
            }
            out.push_str("</ul>\n");
        }
        out
    }

    fn render_json(&self) -> String {
        let map: BTreeMap<&str, serde_json::Value> = [
            ("title", serde_json::json!(self.title)),
            ("entries", serde_json::json!(self.entries)),
            ("notes", serde_json::json!(self.notes)),
        ]
        .into_iter()
        .collect();
        serde_json::to_string_pretty(&map).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.notes.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_mapping() -> StreamMappingDesc {
        StreamMappingDesc {
            stream_name: "Temperature".into(),
            data_field: "temp_celsius".into(),
            parameter: "Pitch".into(),
            low_label: "low pitch".into(),
            high_label: "high pitch".into(),
            unit: "°C".into(),
            data_min: 0.0,
            data_max: 100.0,
        }
    }

    #[test]
    fn description_generator_full() {
        let mut gen = AudioDescriptionGenerator::new("Weather Sonification");
        gen.add_mapping(sample_mapping());
        let desc = gen.generate_full_description();
        assert!(desc.contains("Temperature"));
        assert!(desc.contains("Pitch"));
    }

    #[test]
    fn description_generator_empty() {
        let gen = AudioDescriptionGenerator::new("Empty");
        let desc = gen.generate_full_description();
        assert!(desc.contains("No active"));
    }

    #[test]
    fn description_generator_summary() {
        let mut gen = AudioDescriptionGenerator::new("T");
        gen.add_mapping(sample_mapping());
        let s = gen.generate_summary();
        assert!(s.contains("Temperature"));
    }

    #[test]
    fn description_single_mapping() {
        let mut gen = AudioDescriptionGenerator::new("T");
        gen.add_mapping(sample_mapping());
        let d = gen.describe_mapping(0).unwrap();
        assert!(d.contains("Pitch"));
    }

    #[test]
    fn narrator_current_value() {
        let mut dn = DataNarrator::new("Temperature", "°C");
        dn.push_value(23.5);
        assert!(dn.narrate_current().contains("23.50"));
    }

    #[test]
    fn narrator_trend_rising() {
        let mut dn = DataNarrator::new("Temp", "°C");
        for i in 0..20 {
            dn.push_value(20.0 + i as f64);
        }
        let trend = dn.detect_trend();
        assert_eq!(trend, Trend::Rising);
        let text = dn.narrate_trend();
        assert!(text.contains("rising"));
    }

    #[test]
    fn narrator_trend_stable() {
        let mut dn = DataNarrator::new("Temp", "°C");
        for _ in 0..20 {
            dn.push_value(20.0);
        }
        assert_eq!(dn.detect_trend(), Trend::Stable);
    }

    #[test]
    fn narrator_outlier_detection() {
        let mut dn = DataNarrator::new("Pressure", "hPa");
        for _ in 0..50 {
            dn.push_value(1013.0);
        }
        dn.push_value(900.0); // outlier
        assert!(dn.is_outlier());
        assert!(dn.announce_outlier().is_some());
    }

    #[test]
    fn narrator_no_outlier() {
        let mut dn = DataNarrator::new("P", "hPa");
        for i in 0..20 {
            dn.push_value(1013.0 + (i as f64 * 0.1));
        }
        assert!(!dn.is_outlier());
        assert!(dn.announce_outlier().is_none());
    }

    #[test]
    fn legend_plain_text() {
        let mut lg = LegendGenerator::new("My Sonification");
        lg.add_entry(ScaleEntry {
            parameter: "Pitch".into(),
            data_field: "temperature".into(),
            data_range: (0.0, 100.0),
            perceptual_range: "200 Hz – 2000 Hz".into(),
            description: "Higher pitch = higher temp".into(),
        });
        lg.add_note("Use headphones for best results.");
        let text = lg.render(LegendFormat::PlainText);
        assert!(text.contains("temperature"));
        assert!(text.contains("headphones"));
    }

    #[test]
    fn legend_html() {
        let mut lg = LegendGenerator::new("HTML Legend");
        lg.add_entry(ScaleEntry {
            parameter: "Volume".into(),
            data_field: "wind_speed".into(),
            data_range: (0.0, 50.0),
            perceptual_range: "quiet – loud".into(),
            description: "".into(),
        });
        let html = lg.render(LegendFormat::Html);
        assert!(html.contains("<table>"));
        assert!(html.contains("Volume"));
    }

    #[test]
    fn legend_json() {
        let mut lg = LegendGenerator::new("JSON Legend");
        lg.add_entry(ScaleEntry {
            parameter: "Pan".into(),
            data_field: "latitude".into(),
            data_range: (-90.0, 90.0),
            perceptual_range: "left – right".into(),
            description: "North = right".into(),
        });
        let json = lg.render(LegendFormat::Json);
        assert!(json.contains("latitude"));
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed["entries"].is_array());
    }

    #[test]
    fn legend_clear() {
        let mut lg = LegendGenerator::new("X");
        lg.add_entry(ScaleEntry {
            parameter: "P".into(),
            data_field: "D".into(),
            data_range: (0.0, 1.0),
            perceptual_range: "".into(),
            description: "".into(),
        });
        lg.clear();
        assert_eq!(lg.entry_count(), 0);
    }

    #[test]
    fn narrator_clear() {
        let mut dn = DataNarrator::new("X", "");
        dn.push_value(1.0);
        dn.clear();
        assert_eq!(dn.history_len(), 0);
    }
}
