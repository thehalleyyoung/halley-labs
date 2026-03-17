//! Color scale definitions for mapping numeric values to colors.
//!
//! Provides utilities for converting scalar values into RGB colors using
//! linear, diverging, and categorical color scales. Includes predefined
//! palettes for suspiciousness scores, differential magnitudes, and fault
//! type categorization.

use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Core color type
// ---------------------------------------------------------------------------

/// An RGB color with 8-bit channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Color {
    /// Red channel (0–255).
    pub r: u8,
    /// Green channel (0–255).
    pub g: u8,
    /// Blue channel (0–255).
    pub b: u8,
}

impl Color {
    /// Create a new [`Color`] from RGB components.
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Pure white `#FFFFFF`.
    pub fn white() -> Self {
        Self::new(255, 255, 255)
    }

    /// Pure black `#000000`.
    pub fn black() -> Self {
        Self::new(0, 0, 0)
    }

    /// Convert to a CSS hex string (e.g. `#FF00AA`).
    pub fn to_hex(&self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }

    /// Convert to a CSS `rgb()` function string.
    pub fn to_rgb_string(&self) -> String {
        format!("rgb({},{},{})", self.r, self.g, self.b)
    }

    /// Linear interpolation between `self` and `other` at `t ∈ [0, 1]`.
    pub fn lerp(&self, other: &Color, t: f64) -> Color {
        let t = t.clamp(0.0, 1.0);
        let r = self.r as f64 + (other.r as f64 - self.r as f64) * t;
        let g = self.g as f64 + (other.g as f64 - self.g as f64) * t;
        let b = self.b as f64 + (other.b as f64 - self.b as f64) * t;
        Color::new(r.round() as u8, g.round() as u8, b.round() as u8)
    }

    /// Perceived luminance (ITU-R BT.709).
    pub fn luminance(&self) -> f64 {
        0.2126 * (self.r as f64 / 255.0)
            + 0.7152 * (self.g as f64 / 255.0)
            + 0.0722 * (self.b as f64 / 255.0)
    }

    /// Return a contrasting text color (black or white) for readability.
    pub fn contrast_text(&self) -> Color {
        if self.luminance() > 0.5 {
            Color::black()
        } else {
            Color::white()
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

// ---------------------------------------------------------------------------
// Color stop for gradient definitions
// ---------------------------------------------------------------------------

/// A position–color pair inside a gradient definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorStop {
    /// Normalized position in `[0, 1]`.
    pub position: f64,
    /// Color at this position.
    pub color: Color,
}

impl ColorStop {
    /// Create a new color stop.
    pub fn new(position: f64, color: Color) -> Self {
        Self {
            position: position.clamp(0.0, 1.0),
            color,
        }
    }
}

// ---------------------------------------------------------------------------
// ColorScale trait
// ---------------------------------------------------------------------------

/// Trait for mapping a scalar value to a color.
pub trait ColorScale: fmt::Debug + Send + Sync {
    /// Map a value in the scale's domain to a [`Color`].
    fn map_value(&self, value: f64) -> Color;

    /// Return the domain minimum.
    fn domain_min(&self) -> f64;

    /// Return the domain maximum.
    fn domain_max(&self) -> f64;

    /// Normalize `value` to `[0, 1]` within the domain.
    fn normalize(&self, value: f64) -> f64 {
        let min = self.domain_min();
        let max = self.domain_max();
        if (max - min).abs() < f64::EPSILON {
            return 0.5;
        }
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    }

    /// Generate `n` evenly-spaced sample colors across the scale.
    fn sample(&self, n: usize) -> Vec<Color> {
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![self.map_value((self.domain_min() + self.domain_max()) / 2.0)];
        }
        let step = (self.domain_max() - self.domain_min()) / (n - 1) as f64;
        (0..n)
            .map(|i| self.map_value(self.domain_min() + step * i as f64))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// LinearScale – multi-stop linear gradient
// ---------------------------------------------------------------------------

/// A multi-stop linear color scale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearScale {
    /// Ordered color stops (positions normalized to domain).
    pub stops: Vec<ColorStop>,
    /// Domain minimum value.
    pub min: f64,
    /// Domain maximum value.
    pub max: f64,
}

impl LinearScale {
    /// Build a new linear scale from stops and a domain.
    pub fn new(stops: Vec<ColorStop>, min: f64, max: f64) -> Self {
        let mut s = Self { stops, min, max };
        s.stops.sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap());
        s
    }

    /// Two-stop linear scale from `low_color` to `high_color`.
    pub fn two_stop(low: Color, high: Color, min: f64, max: f64) -> Self {
        Self::new(
            vec![ColorStop::new(0.0, low), ColorStop::new(1.0, high)],
            min,
            max,
        )
    }

    /// Three-stop linear scale (low → mid → high).
    pub fn three_stop(low: Color, mid: Color, high: Color, min: f64, max: f64) -> Self {
        Self::new(
            vec![
                ColorStop::new(0.0, low),
                ColorStop::new(0.5, mid),
                ColorStop::new(1.0, high),
            ],
            min,
            max,
        )
    }

    /// Predefined green → yellow → red suspiciousness scale.
    pub fn suspiciousness(min: f64, max: f64) -> Self {
        Self::three_stop(
            Color::new(34, 139, 34),   // forest green
            Color::new(255, 215, 0),   // gold
            Color::new(220, 20, 60),   // crimson
            min,
            max,
        )
    }

    /// Predefined cool-blue → white → warm-red differential scale.
    pub fn differential(min: f64, max: f64) -> Self {
        Self::three_stop(
            Color::new(33, 102, 172),  // steel blue
            Color::white(),
            Color::new(178, 24, 43),   // brick red
            min,
            max,
        )
    }
}

impl ColorScale for LinearScale {
    fn map_value(&self, value: f64) -> Color {
        if self.stops.is_empty() {
            return Color::black();
        }
        let t = self.normalize(value);
        if self.stops.len() == 1 {
            return self.stops[0].color;
        }
        // Find the two surrounding stops.
        for window in self.stops.windows(2) {
            let lo = &window[0];
            let hi = &window[1];
            if t >= lo.position && t <= hi.position {
                let seg_t = if (hi.position - lo.position).abs() < f64::EPSILON {
                    0.0
                } else {
                    (t - lo.position) / (hi.position - lo.position)
                };
                return lo.color.lerp(&hi.color, seg_t);
            }
        }
        // Clamp to edge stops.
        if t <= self.stops[0].position {
            self.stops[0].color
        } else {
            self.stops.last().unwrap().color
        }
    }

    fn domain_min(&self) -> f64 {
        self.min
    }

    fn domain_max(&self) -> f64 {
        self.max
    }
}

// ---------------------------------------------------------------------------
// DivergingScale – centered at a midpoint
// ---------------------------------------------------------------------------

/// A diverging color scale centered at a midpoint value.
///
/// Values below the midpoint interpolate between `low_color` and `mid_color`,
/// values above interpolate between `mid_color` and `high_color`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergingScale {
    /// Color for the domain minimum.
    pub low_color: Color,
    /// Color for the midpoint.
    pub mid_color: Color,
    /// Color for the domain maximum.
    pub high_color: Color,
    /// Domain minimum.
    pub min: f64,
    /// Midpoint value.
    pub mid: f64,
    /// Domain maximum.
    pub max: f64,
}

impl DivergingScale {
    /// Create a new diverging scale.
    pub fn new(low: Color, mid: Color, high: Color, min: f64, midpoint: f64, max: f64) -> Self {
        Self {
            low_color: low,
            mid_color: mid,
            high_color: high,
            min,
            mid: midpoint,
            max,
        }
    }

    /// Blue → white → red diverging scale centered at `mid`.
    pub fn blue_white_red(min: f64, mid: f64, max: f64) -> Self {
        Self::new(
            Color::new(33, 102, 172),
            Color::white(),
            Color::new(178, 24, 43),
            min,
            mid,
            max,
        )
    }

    /// Purple → white → orange diverging scale.
    pub fn purple_white_orange(min: f64, mid: f64, max: f64) -> Self {
        Self::new(
            Color::new(128, 0, 128),
            Color::white(),
            Color::new(255, 140, 0),
            min,
            mid,
            max,
        )
    }
}

impl ColorScale for DivergingScale {
    fn map_value(&self, value: f64) -> Color {
        if value <= self.mid {
            let range = self.mid - self.min;
            let t = if range.abs() < f64::EPSILON {
                0.0
            } else {
                ((value - self.min) / range).clamp(0.0, 1.0)
            };
            self.low_color.lerp(&self.mid_color, t)
        } else {
            let range = self.max - self.mid;
            let t = if range.abs() < f64::EPSILON {
                1.0
            } else {
                ((value - self.mid) / range).clamp(0.0, 1.0)
            };
            self.mid_color.lerp(&self.high_color, t)
        }
    }

    fn domain_min(&self) -> f64 {
        self.min
    }

    fn domain_max(&self) -> f64 {
        self.max
    }
}

// ---------------------------------------------------------------------------
// CategoricalPalette
// ---------------------------------------------------------------------------

/// A fixed set of named colors for categorical data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalPalette {
    /// Named color entries.
    pub entries: Vec<(String, Color)>,
}

impl CategoricalPalette {
    /// Create a new categorical palette from name–color pairs.
    pub fn new(entries: Vec<(String, Color)>) -> Self {
        Self { entries }
    }

    /// Look up a color by category name (case-insensitive).
    pub fn get(&self, name: &str) -> Option<Color> {
        let lower = name.to_lowercase();
        self.entries
            .iter()
            .find(|(n, _)| n.to_lowercase() == lower)
            .map(|(_, c)| *c)
    }

    /// Get a color by index, wrapping around if needed.
    pub fn by_index(&self, idx: usize) -> Color {
        if self.entries.is_empty() {
            return Color::black();
        }
        self.entries[idx % self.entries.len()].1
    }

    /// Predefined fault-type palette.
    pub fn fault_types() -> Self {
        Self::new(vec![
            ("direct".into(), Color::new(220, 20, 60)),      // crimson
            ("indirect".into(), Color::new(255, 165, 0)),     // orange
            ("confounded".into(), Color::new(148, 103, 189)), // purple
            ("benign".into(), Color::new(44, 160, 44)),       // green
            ("unknown".into(), Color::new(150, 150, 150)),    // gray
        ])
    }

    /// Predefined pipeline stage palette (up to 10 stages).
    pub fn pipeline_stages() -> Self {
        Self::new(vec![
            ("tokenizer".into(), Color::new(31, 119, 180)),
            ("pos_tagger".into(), Color::new(255, 127, 14)),
            ("parser".into(), Color::new(44, 160, 44)),
            ("ner".into(), Color::new(214, 39, 40)),
            ("lemmatizer".into(), Color::new(148, 103, 189)),
            ("dependency".into(), Color::new(140, 86, 75)),
            ("sentiment".into(), Color::new(227, 119, 194)),
            ("coreference".into(), Color::new(127, 127, 127)),
            ("srl".into(), Color::new(188, 189, 34)),
            ("embedding".into(), Color::new(23, 190, 207)),
        ])
    }

    /// Number of categories in this palette.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the palette has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_hex_roundtrip() {
        let c = Color::new(255, 128, 0);
        assert_eq!(c.to_hex(), "#FF8000");
        assert_eq!(c.to_rgb_string(), "rgb(255,128,0)");
    }

    #[test]
    fn color_lerp_midpoint() {
        let a = Color::black();
        let b = Color::white();
        let mid = a.lerp(&b, 0.5);
        assert!((mid.r as i16 - 128).abs() <= 1);
    }

    #[test]
    fn contrast_text_on_dark() {
        let dark = Color::new(10, 10, 10);
        assert_eq!(dark.contrast_text(), Color::white());
    }

    #[test]
    fn contrast_text_on_light() {
        let light = Color::new(240, 240, 240);
        assert_eq!(light.contrast_text(), Color::black());
    }

    #[test]
    fn linear_scale_suspiciousness() {
        let s = LinearScale::suspiciousness(0.0, 1.0);
        let low = s.map_value(0.0);
        let high = s.map_value(1.0);
        // low should be greenish, high should be reddish
        assert!(low.g > low.r);
        assert!(high.r > high.g);
    }

    #[test]
    fn linear_scale_sample() {
        let s = LinearScale::two_stop(Color::black(), Color::white(), 0.0, 1.0);
        let samples = s.sample(3);
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0], Color::black());
        assert_eq!(samples[2], Color::white());
    }

    #[test]
    fn diverging_scale_midpoint() {
        let s = DivergingScale::blue_white_red(-1.0, 0.0, 1.0);
        let mid = s.map_value(0.0);
        assert_eq!(mid, Color::white());
    }

    #[test]
    fn categorical_palette_lookup() {
        let p = CategoricalPalette::fault_types();
        assert!(p.get("direct").is_some());
        assert!(p.get("Direct").is_some()); // case-insensitive
        assert!(p.get("nonexistent").is_none());
    }

    #[test]
    fn categorical_by_index_wraps() {
        let p = CategoricalPalette::fault_types();
        let c0 = p.by_index(0);
        let cw = p.by_index(p.len());
        assert_eq!(c0, cw);
    }

    #[test]
    fn linear_scale_normalize_flat() {
        let s = LinearScale::two_stop(Color::black(), Color::white(), 5.0, 5.0);
        assert!((s.normalize(5.0) - 0.5).abs() < f64::EPSILON);
    }
}
