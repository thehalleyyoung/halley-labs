//! Audio buffer types and operations for the SoniType renderer.
//!
//! Provides multi-channel audio buffers, frame types, buffer manipulation
//! (mix, gain, pan, fade, normalize, clip, resample), a ring buffer for
//! real-time streaming, WAV header writing, level metering, and windowing
//! functions.

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fmt;
use std::io::{self, Write};

// ===========================================================================
// Window functions
// ===========================================================================

/// Windowing function types for spectral analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WindowFunction {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
    Kaiser { beta: ordered_float::OrderedFloat<f64> },
}

impl WindowFunction {
    /// Generate a window of the given length.
    pub fn generate(&self, length: usize) -> Vec<f64> {
        match self {
            WindowFunction::Rectangular => vec![1.0; length],
            WindowFunction::Hann => (0..length)
                .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (length - 1) as f64).cos()))
                .collect(),
            WindowFunction::Hamming => (0..length)
                .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (length - 1) as f64).cos())
                .collect(),
            WindowFunction::Blackman => (0..length)
                .map(|i| {
                    let n = i as f64 / (length - 1) as f64;
                    0.42 - 0.5 * (2.0 * PI * n).cos() + 0.08 * (4.0 * PI * n).cos()
                })
                .collect(),
            WindowFunction::Kaiser { beta } => {
                let beta = beta.into_inner();
                let m = (length - 1) as f64;
                (0..length)
                    .map(|i| {
                        let x = 2.0 * i as f64 / m - 1.0;
                        bessel_i0(beta * (1.0 - x * x).sqrt()) / bessel_i0(beta)
                    })
                    .collect()
            }
        }
    }
}

/// Zeroth-order modified Bessel function of the first kind (for Kaiser window).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..25 {
        term *= (x / (2.0 * k as f64)).powi(2);
        sum += term;
        if term < 1e-15 { break; }
    }
    sum
}

// ===========================================================================
// AudioFrame
// ===========================================================================

/// A single audio frame: one sample per channel.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioFrame {
    pub samples: Vec<f32>,
}

impl AudioFrame {
    pub fn new(channels: usize) -> Self {
        Self { samples: vec![0.0; channels] }
    }

    pub fn from_samples(samples: Vec<f32>) -> Self {
        Self { samples }
    }

    pub fn channels(&self) -> usize {
        self.samples.len()
    }

    pub fn mono(sample: f32) -> Self {
        Self { samples: vec![sample] }
    }

    pub fn stereo(left: f32, right: f32) -> Self {
        Self { samples: vec![left, right] }
    }
}

impl fmt::Display for AudioFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Frame{:?}", self.samples)
    }
}

// ===========================================================================
// AudioBuffer
// ===========================================================================

/// Multi-channel audio buffer stored as interleaved channel data.
///
/// Internal layout: `data[channel][sample_index]` - each channel is a
/// contiguous Vec<f32>.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioBuffer {
    /// Per-channel sample data.
    channels: Vec<Vec<f32>>,
    pub sample_rate: u32,
}

impl AudioBuffer {
    /// Create a silent buffer with the given number of channels and frames.
    pub fn new(num_channels: usize, num_frames: usize, sample_rate: u32) -> Self {
        Self {
            channels: vec![vec![0.0; num_frames]; num_channels],
            sample_rate,
        }
    }

    /// Create from pre-existing channel data.
    pub fn from_channels(channels: Vec<Vec<f32>>, sample_rate: u32) -> Self {
        Self { channels, sample_rate }
    }

    /// Create a mono buffer from a single channel of samples.
    pub fn from_mono(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self { channels: vec![samples], sample_rate }
    }

    /// Number of channels.
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Number of sample frames.
    pub fn num_frames(&self) -> usize {
        self.channels.first().map(|c| c.len()).unwrap_or(0)
    }

    /// Duration of the buffer in seconds.
    pub fn duration_seconds(&self) -> f64 {
        self.num_frames() as f64 / self.sample_rate as f64
    }

    /// Get a sample at (channel, frame).
    pub fn get(&self, channel: usize, frame: usize) -> f32 {
        self.channels[channel][frame]
    }

    /// Set a sample at (channel, frame).
    pub fn set(&mut self, channel: usize, frame: usize, value: f32) {
        self.channels[channel][frame] = value;
    }

    /// Get a reference to a channel's sample data.
    pub fn channel(&self, ch: usize) -> &[f32] {
        &self.channels[ch]
    }

    /// Get a mutable reference to a channel's sample data.
    pub fn channel_mut(&mut self, ch: usize) -> &mut [f32] {
        &mut self.channels[ch]
    }

    /// Extract a frame at the given index.
    pub fn frame(&self, index: usize) -> AudioFrame {
        AudioFrame {
            samples: self.channels.iter().map(|ch| ch[index]).collect(),
        }
    }

    /// Fill the entire buffer with silence.
    pub fn clear(&mut self) {
        for ch in &mut self.channels {
            for s in ch.iter_mut() { *s = 0.0; }
        }
    }

    // ---- Operations ----

    /// Apply gain (multiply all samples by `gain`).
    pub fn gain(&mut self, gain: f32) {
        for ch in &mut self.channels {
            for s in ch.iter_mut() { *s *= gain; }
        }
    }

    /// Mix another buffer into this one (additive).
    pub fn mix(&mut self, other: &AudioBuffer) {
        let frames = self.num_frames().min(other.num_frames());
        let chans = self.num_channels().min(other.num_channels());
        for c in 0..chans {
            for i in 0..frames {
                self.channels[c][i] += other.channels[c][i];
            }
        }
    }

    /// Apply stereo panning. Only works on 2-channel buffers.
    pub fn pan(&mut self, pan_value: f32) {
        if self.num_channels() != 2 { return; }
        let angle = (pan_value as f64 + 1.0) * std::f64::consts::FRAC_PI_4;
        let gain_l = angle.cos() as f32;
        let gain_r = angle.sin() as f32;
        let frames = self.num_frames();
        for i in 0..frames {
            self.channels[0][i] *= gain_l;
            self.channels[1][i] *= gain_r;
        }
    }

    /// Apply a linear fade-in over `num_samples` frames.
    pub fn fade_in(&mut self, num_samples: usize) {
        let n = num_samples.min(self.num_frames());
        for ch in &mut self.channels {
            for i in 0..n {
                ch[i] *= i as f32 / n as f32;
            }
        }
    }

    /// Apply a linear fade-out over the last `num_samples` frames.
    pub fn fade_out(&mut self, num_samples: usize) {
        let total = self.num_frames();
        let n = num_samples.min(total);
        let start = total - n;
        for ch in &mut self.channels {
            for i in 0..n {
                ch[start + i] *= (n - i) as f32 / n as f32;
            }
        }
    }

    /// Normalize so the peak absolute value is `target` (default 1.0).
    pub fn normalize(&mut self, target: f32) {
        let peak = self.peak_level();
        if peak < 1e-10 { return; }
        let scale = target / peak;
        self.gain(scale);
    }

    /// Hard-clip all samples to [-limit, +limit].
    pub fn clip(&mut self, limit: f32) {
        for ch in &mut self.channels {
            for s in ch.iter_mut() {
                *s = s.clamp(-limit, limit);
            }
        }
    }

    /// Simple linear resampling to a new sample rate.
    pub fn resample_linear(&self, target_rate: u32) -> AudioBuffer {
        if target_rate == self.sample_rate {
            return self.clone();
        }
        let ratio = target_rate as f64 / self.sample_rate as f64;
        let new_len = (self.num_frames() as f64 * ratio).ceil() as usize;
        let mut out = AudioBuffer::new(self.num_channels(), new_len, target_rate);
        for c in 0..self.num_channels() {
            for i in 0..new_len {
                let src_pos = i as f64 / ratio;
                let lo = src_pos.floor() as usize;
                let hi = (lo + 1).min(self.num_frames() - 1);
                let frac = (src_pos - lo as f64) as f32;
                out.channels[c][i] = self.channels[c][lo] * (1.0 - frac) + self.channels[c][hi] * frac;
            }
        }
        out
    }

    // ---- Level metering ----

    /// Peak absolute level across all channels.
    pub fn peak_level(&self) -> f32 {
        self.channels.iter()
            .flat_map(|ch| ch.iter())
            .map(|s| s.abs())
            .fold(0.0f32, f32::max)
    }

    /// RMS level for a specific channel.
    pub fn rms_level(&self, channel: usize) -> f32 {
        let ch = &self.channels[channel];
        if ch.is_empty() { return 0.0; }
        let sum: f32 = ch.iter().map(|s| s * s).sum();
        (sum / ch.len() as f32).sqrt()
    }

    /// RMS level averaged across all channels.
    pub fn rms_level_all(&self) -> f32 {
        if self.num_channels() == 0 { return 0.0; }
        let sum: f32 = (0..self.num_channels()).map(|c| self.rms_level(c)).sum();
        sum / self.num_channels() as f32
    }

    /// Apply a window function to all channels.
    pub fn apply_window(&mut self, window: &WindowFunction) {
        let win = window.generate(self.num_frames());
        for ch in &mut self.channels {
            for (s, &w) in ch.iter_mut().zip(win.iter()) {
                *s *= w as f32;
            }
        }
    }
}

impl fmt::Display for AudioBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AudioBuffer({}ch, {} frames, {} Hz)",
            self.num_channels(), self.num_frames(), self.sample_rate)
    }
}

// ===========================================================================
// WAV file writing
// ===========================================================================

/// Write a WAV file header for 16-bit PCM data.
pub fn write_wav_header<W: Write>(
    writer: &mut W,
    sample_rate: u32,
    num_channels: u16,
    num_samples: u32,
) -> io::Result<()> {
    let bit_depth: u16 = 16;
    let byte_rate = sample_rate * num_channels as u32 * (bit_depth as u32 / 8);
    let block_align = num_channels * (bit_depth / 8);
    let data_size = num_samples * num_channels as u32 * (bit_depth as u32 / 8);
    let file_size = 36 + data_size;

    writer.write_all(b"RIFF")?;
    writer.write_all(&file_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;     // fmt chunk size
    writer.write_all(&1u16.to_le_bytes())?;      // PCM format
    writer.write_all(&num_channels.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&bit_depth.to_le_bytes())?;
    writer.write_all(b"data")?;
    writer.write_all(&data_size.to_le_bytes())?;
    Ok(())
}

/// Write an AudioBuffer as 16-bit PCM WAV data (header + samples).
pub fn write_wav_buffer<W: Write>(writer: &mut W, buffer: &AudioBuffer) -> io::Result<()> {
    write_wav_header(
        writer,
        buffer.sample_rate,
        buffer.num_channels() as u16,
        buffer.num_frames() as u32,
    )?;
    for i in 0..buffer.num_frames() {
        for c in 0..buffer.num_channels() {
            let sample = buffer.get(c, i).clamp(-1.0, 1.0);
            let i16_val = (sample * 32767.0) as i16;
            writer.write_all(&i16_val.to_le_bytes())?;
        }
    }
    Ok(())
}

// ===========================================================================
// Ring buffer
// ===========================================================================

/// Lock-free-style ring buffer for real-time audio streaming.
#[derive(Clone, Debug)]
pub struct RingBuffer {
    data: Vec<f32>,
    capacity: usize,
    write_pos: usize,
    read_pos: usize,
    count: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            capacity,
            write_pos: 0,
            read_pos: 0,
            count: 0,
        }
    }

    pub fn capacity(&self) -> usize { self.capacity }
    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn is_full(&self) -> bool { self.count == self.capacity }
    pub fn available_write(&self) -> usize { self.capacity - self.count }
    pub fn available_read(&self) -> usize { self.count }

    /// Push a single sample. Returns false if full.
    pub fn push(&mut self, sample: f32) -> bool {
        if self.is_full() { return false; }
        self.data[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.count += 1;
        true
    }

    /// Pop a single sample. Returns None if empty.
    pub fn pop(&mut self) -> Option<f32> {
        if self.is_empty() { return None; }
        let val = self.data[self.read_pos];
        self.read_pos = (self.read_pos + 1) % self.capacity;
        self.count -= 1;
        Some(val)
    }

    /// Write a slice of samples. Returns the number actually written.
    pub fn write(&mut self, samples: &[f32]) -> usize {
        let n = samples.len().min(self.available_write());
        for &s in &samples[..n] {
            self.data[self.write_pos] = s;
            self.write_pos = (self.write_pos + 1) % self.capacity;
        }
        self.count += n;
        n
    }

    /// Read into a slice. Returns the number actually read.
    pub fn read(&mut self, output: &mut [f32]) -> usize {
        let n = output.len().min(self.available_read());
        for o in &mut output[..n] {
            *o = self.data[self.read_pos];
            self.read_pos = (self.read_pos + 1) % self.capacity;
        }
        self.count -= n;
        n
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.read_pos = 0;
        self.count = 0;
    }

    /// Peek at the oldest sample without consuming it.
    pub fn peek(&self) -> Option<f32> {
        if self.is_empty() { None } else { Some(self.data[self.read_pos]) }
    }
}

impl fmt::Display for RingBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RingBuffer({}/{})", self.count, self.capacity)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_buffer_creation() {
        let buf = AudioBuffer::new(2, 1024, 44100);
        assert_eq!(buf.num_channels(), 2);
        assert_eq!(buf.num_frames(), 1024);
        assert_eq!(buf.sample_rate, 44100);
    }

    #[test]
    fn audio_buffer_get_set() {
        let mut buf = AudioBuffer::new(2, 10, 44100);
        buf.set(0, 5, 0.5);
        assert_eq!(buf.get(0, 5), 0.5);
        assert_eq!(buf.get(1, 5), 0.0);
    }

    #[test]
    fn audio_buffer_gain() {
        let mut buf = AudioBuffer::from_mono(vec![0.5, -0.5, 1.0], 44100);
        buf.gain(2.0);
        assert_eq!(buf.get(0, 0), 1.0);
        assert_eq!(buf.get(0, 1), -1.0);
    }

    #[test]
    fn audio_buffer_mix() {
        let mut a = AudioBuffer::from_mono(vec![0.5, 0.5], 44100);
        let b = AudioBuffer::from_mono(vec![0.3, 0.3], 44100);
        a.mix(&b);
        assert!((a.get(0, 0) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn audio_buffer_fade() {
        let mut buf = AudioBuffer::from_mono(vec![1.0; 100], 44100);
        buf.fade_in(10);
        assert_eq!(buf.get(0, 0), 0.0);
        assert!(buf.get(0, 5) > 0.0 && buf.get(0, 5) < 1.0);
        assert_eq!(buf.get(0, 50), 1.0);

        buf.fade_out(10);
        // Last sample gets factor 1/n, which is near zero but not exactly zero
        assert!(buf.get(0, 99) < 0.15);
    }

    #[test]
    fn audio_buffer_normalize() {
        let mut buf = AudioBuffer::from_mono(vec![0.25, -0.5, 0.1], 44100);
        buf.normalize(1.0);
        assert!((buf.peak_level() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn audio_buffer_clip() {
        let mut buf = AudioBuffer::from_mono(vec![2.0, -2.0, 0.5], 44100);
        buf.clip(1.0);
        assert_eq!(buf.get(0, 0), 1.0);
        assert_eq!(buf.get(0, 1), -1.0);
        assert_eq!(buf.get(0, 2), 0.5);
    }

    #[test]
    fn audio_buffer_resample() {
        let buf = AudioBuffer::from_mono(vec![0.0, 1.0, 0.0, -1.0], 44100);
        let resampled = buf.resample_linear(88200);
        assert!(resampled.num_frames() > buf.num_frames());
        assert_eq!(resampled.sample_rate, 88200);
    }

    #[test]
    fn audio_buffer_rms() {
        let buf = AudioBuffer::from_mono(vec![1.0, -1.0, 1.0, -1.0], 44100);
        assert!((buf.rms_level(0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn window_hann() {
        let win = WindowFunction::Hann.generate(8);
        assert_eq!(win.len(), 8);
        assert!(win[0].abs() < 1e-10);
        // For N=8, peak is between indices 3 and 4, each ~0.95
        assert!(win[3] > 0.9 && win[4] > 0.9);
    }

    #[test]
    fn window_hamming() {
        let win = WindowFunction::Hamming.generate(16);
        assert_eq!(win.len(), 16);
        assert!(win[0] > 0.0); // Hamming doesn't go to zero
    }

    #[test]
    fn window_blackman() {
        let win = WindowFunction::Blackman.generate(32);
        assert_eq!(win.len(), 32);
        assert!(win[16] > win[0]);
    }

    #[test]
    fn ring_buffer_basic() {
        let mut rb = RingBuffer::new(4);
        assert!(rb.is_empty());
        assert!(rb.push(1.0));
        assert!(rb.push(2.0));
        assert_eq!(rb.len(), 2);
        assert_eq!(rb.pop(), Some(1.0));
        assert_eq!(rb.pop(), Some(2.0));
        assert!(rb.is_empty());
    }

    #[test]
    fn ring_buffer_wrap_around() {
        let mut rb = RingBuffer::new(3);
        rb.push(1.0);
        rb.push(2.0);
        rb.push(3.0);
        assert!(rb.is_full());
        assert!(!rb.push(4.0)); // should fail
        assert_eq!(rb.pop(), Some(1.0));
        assert!(rb.push(4.0)); // now should work
        assert_eq!(rb.pop(), Some(2.0));
        assert_eq!(rb.pop(), Some(3.0));
        assert_eq!(rb.pop(), Some(4.0));
    }

    #[test]
    fn ring_buffer_read_write() {
        let mut rb = RingBuffer::new(8);
        let written = rb.write(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(written, 4);
        let mut out = [0.0f32; 3];
        let read = rb.read(&mut out);
        assert_eq!(read, 3);
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn wav_header_write() {
        let mut buf = Vec::new();
        write_wav_header(&mut buf, 44100, 2, 1000).unwrap();
        assert_eq!(&buf[0..4], b"RIFF");
        assert_eq!(&buf[8..12], b"WAVE");
        assert_eq!(&buf[12..16], b"fmt ");
        assert_eq!(buf.len(), 44);
    }

    #[test]
    fn audio_frame_stereo() {
        let frame = AudioFrame::stereo(0.5, -0.5);
        assert_eq!(frame.channels(), 2);
        assert_eq!(frame.samples[0], 0.5);
        assert_eq!(frame.samples[1], -0.5);
    }

    #[test]
    fn audio_buffer_duration() {
        let buf = AudioBuffer::new(1, 44100, 44100);
        assert!((buf.duration_seconds() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn audio_buffer_apply_window() {
        let mut buf = AudioBuffer::from_mono(vec![1.0; 8], 44100);
        buf.apply_window(&WindowFunction::Hann);
        assert!(buf.get(0, 0).abs() < 1e-5);
        assert!(buf.get(0, 4) > 0.9);
    }

    #[test]
    fn ring_buffer_peek() {
        let mut rb = RingBuffer::new(4);
        assert_eq!(rb.peek(), None);
        rb.push(42.0);
        assert_eq!(rb.peek(), Some(42.0));
        assert_eq!(rb.len(), 1);
    }
}
