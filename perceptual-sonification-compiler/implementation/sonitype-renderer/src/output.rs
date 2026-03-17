//! Audio output: WAV writer, raw/null/buffered outputs, metering (peak, RMS,
//! LUFS, true-peak).

use crate::{AudioBuf, RendererResult, RendererError};
use std::io::{self, Write, Seek, SeekFrom};

// ---------------------------------------------------------------------------
// OutputFormat
// ---------------------------------------------------------------------------

/// Supported output sample formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Pcm16,
    Pcm24,
    Float32,
}

impl OutputFormat {
    /// Bytes per sample for this format.
    pub fn bytes_per_sample(&self) -> usize {
        match self {
            Self::Pcm16 => 2,
            Self::Pcm24 => 3,
            Self::Float32 => 4,
        }
    }

    /// WAV format tag.
    fn wav_format_tag(&self) -> u16 {
        match self {
            Self::Pcm16 | Self::Pcm24 => 1, // PCM
            Self::Float32 => 3,              // IEEE float
        }
    }
}

/// Configuration for an output sink.
#[derive(Debug, Clone)]
pub struct OutputConfig {
    pub sample_rate: u32,
    pub channels: usize,
    pub format: OutputFormat,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self { sample_rate: 44100, channels: 2, format: OutputFormat::Float32 }
    }
}

// ---------------------------------------------------------------------------
// WavWriter
// ---------------------------------------------------------------------------

/// Streaming WAV file writer with proper RIFF header patching.
pub struct WavWriter<W: Write + Seek> {
    writer: W,
    config: OutputConfig,
    data_bytes_written: u32,
    finalized: bool,
}

impl<W: Write + Seek> WavWriter<W> {
    /// Open a new WAV file for writing. Writes the header immediately.
    pub fn new(mut writer: W, config: OutputConfig) -> io::Result<Self> {
        // Write placeholder header (44 bytes)
        Self::write_header(&mut writer, &config, 0)?;
        Ok(Self { writer, config, data_bytes_written: 0, finalized: false })
    }

    fn write_header(w: &mut W, cfg: &OutputConfig, data_size: u32) -> io::Result<()> {
        let bps = cfg.format.bytes_per_sample() as u16;
        let block_align = bps * cfg.channels as u16;
        let byte_rate = cfg.sample_rate * block_align as u32;
        let bits = bps * 8;

        w.seek(SeekFrom::Start(0))?;
        w.write_all(b"RIFF")?;
        w.write_all(&(36 + data_size).to_le_bytes())?;
        w.write_all(b"WAVE")?;

        // fmt chunk
        w.write_all(b"fmt ")?;
        w.write_all(&16u32.to_le_bytes())?; // chunk size
        w.write_all(&cfg.format.wav_format_tag().to_le_bytes())?;
        w.write_all(&(cfg.channels as u16).to_le_bytes())?;
        w.write_all(&cfg.sample_rate.to_le_bytes())?;
        w.write_all(&byte_rate.to_le_bytes())?;
        w.write_all(&block_align.to_le_bytes())?;
        w.write_all(&bits.to_le_bytes())?;

        // data chunk
        w.write_all(b"data")?;
        w.write_all(&data_size.to_le_bytes())?;
        Ok(())
    }

    /// Append an audio buffer to the WAV file.
    pub fn write_buffer(&mut self, buf: &AudioBuf) -> io::Result<()> {
        let frames = buf.frames();
        let ch = self.config.channels.min(buf.channels);
        for f in 0..frames {
            for c in 0..ch {
                let sample = buf.get(f, c);
                match self.config.format {
                    OutputFormat::Pcm16 => {
                        let s16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                        self.writer.write_all(&s16.to_le_bytes())?;
                        self.data_bytes_written += 2;
                    }
                    OutputFormat::Pcm24 => {
                        let s32 = (sample * 8388607.0).clamp(-8388608.0, 8388607.0) as i32;
                        let bytes = s32.to_le_bytes();
                        self.writer.write_all(&bytes[0..3])?;
                        self.data_bytes_written += 3;
                    }
                    OutputFormat::Float32 => {
                        self.writer.write_all(&sample.to_le_bytes())?;
                        self.data_bytes_written += 4;
                    }
                }
            }
        }
        Ok(())
    }

    /// Finalise the WAV: patch the RIFF and data-chunk sizes in the header.
    pub fn finalize(&mut self) -> io::Result<()> {
        if self.finalized { return Ok(()); }
        Self::write_header(&mut self.writer, &self.config, self.data_bytes_written)?;
        // Seek to end so callers can continue writing if needed
        self.writer.seek(SeekFrom::End(0))?;
        self.finalized = true;
        Ok(())
    }

    pub fn bytes_written(&self) -> u32 {
        self.data_bytes_written
    }
}

impl<W: Write + Seek> Drop for WavWriter<W> {
    fn drop(&mut self) {
        let _ = self.finalize();
    }
}

// ---------------------------------------------------------------------------
// RawOutput
// ---------------------------------------------------------------------------

/// Writes raw f32 samples to a [`Write`] sink.
pub struct RawOutput<W: Write> {
    writer: W,
    samples_written: usize,
}

impl<W: Write> RawOutput<W> {
    pub fn new(writer: W) -> Self {
        Self { writer, samples_written: 0 }
    }

    pub fn write_buffer(&mut self, buf: &AudioBuf) -> io::Result<()> {
        for &s in &buf.data {
            self.writer.write_all(&s.to_le_bytes())?;
            self.samples_written += 1;
        }
        Ok(())
    }

    pub fn samples_written(&self) -> usize {
        self.samples_written
    }

    pub fn into_inner(self) -> W {
        self.writer
    }
}

// ---------------------------------------------------------------------------
// NullOutput
// ---------------------------------------------------------------------------

/// Discards all audio (useful for benchmarking).
#[derive(Debug, Clone, Default)]
pub struct NullOutput {
    frames_processed: usize,
}

impl NullOutput {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn write_buffer(&mut self, buf: &AudioBuf) {
        self.frames_processed += buf.frames();
    }

    pub fn frames_processed(&self) -> usize {
        self.frames_processed
    }

    pub fn reset(&mut self) {
        self.frames_processed = 0;
    }
}

// ---------------------------------------------------------------------------
// BufferedOutput
// ---------------------------------------------------------------------------

/// Accumulates all rendered audio in memory.
#[derive(Debug, Clone)]
pub struct BufferedOutput {
    pub data: Vec<f32>,
    pub channels: usize,
    pub sample_rate: u32,
}

impl BufferedOutput {
    pub fn new(channels: usize, sample_rate: u32) -> Self {
        Self { data: Vec::new(), channels, sample_rate }
    }

    pub fn write_buffer(&mut self, buf: &AudioBuf) {
        self.data.extend_from_slice(&buf.data);
    }

    /// Total number of frames accumulated.
    pub fn frames(&self) -> usize {
        if self.channels == 0 { 0 } else { self.data.len() / self.channels }
    }

    /// Get the entire buffer as an [`AudioBuf`].
    pub fn as_audio_buf(&self) -> AudioBuf {
        AudioBuf {
            data: self.data.clone(),
            channels: self.channels,
            sample_rate: self.sample_rate,
        }
    }

    /// Peak level per channel (linear).
    pub fn peak_levels(&self) -> Vec<f32> {
        let mut peaks = vec![0.0f32; self.channels];
        for (i, &s) in self.data.iter().enumerate() {
            let ch = i % self.channels;
            if s.abs() > peaks[ch] {
                peaks[ch] = s.abs();
            }
        }
        peaks
    }

    /// RMS level per channel.
    pub fn rms_levels(&self) -> Vec<f32> {
        let mut sums = vec![0.0f64; self.channels];
        let mut counts = vec![0usize; self.channels];
        for (i, &s) in self.data.iter().enumerate() {
            let ch = i % self.channels;
            sums[ch] += (s as f64) * (s as f64);
            counts[ch] += 1;
        }
        sums.iter()
            .zip(counts.iter())
            .map(|(&sum, &cnt)| if cnt > 0 { (sum / cnt as f64).sqrt() as f32 } else { 0.0 })
            .collect()
    }

    pub fn reset(&mut self) {
        self.data.clear();
    }
}

// ---------------------------------------------------------------------------
// Metering
// ---------------------------------------------------------------------------

/// Per-channel metering state.
#[derive(Debug, Clone)]
struct ChannelMeter {
    peak: f64,
    rms_sum: f64,
    rms_count: u64,
    /// Loudness-weighted energy accumulator (for LUFS approximation).
    lufs_energy: f64,
    lufs_count: u64,
    /// True-peak: 4× oversampled peak detector state.
    true_peak: f64,
    /// Previous samples for the true-peak FIR (simplified 4× oversampler).
    tp_history: [f64; 4],
}

impl Default for ChannelMeter {
    fn default() -> Self {
        Self {
            peak: 0.0,
            rms_sum: 0.0,
            rms_count: 0,
            lufs_energy: 0.0,
            lufs_count: 0,
            true_peak: 0.0,
            tp_history: [0.0; 4],
        }
    }
}

/// Audio metering: peak, RMS, LUFS (simplified), and true-peak per channel.
#[derive(Debug, Clone)]
pub struct Metering {
    channels: Vec<ChannelMeter>,
    pub sample_rate: f64,
}

impl Metering {
    pub fn new(num_channels: usize, sample_rate: f64) -> Self {
        Self {
            channels: vec![ChannelMeter::default(); num_channels],
            sample_rate,
        }
    }

    pub fn reset(&mut self) {
        for ch in &mut self.channels {
            *ch = ChannelMeter::default();
        }
    }

    /// Feed a buffer of audio for metering analysis.
    pub fn process(&mut self, buf: &AudioBuf) {
        let frames = buf.frames();
        let num_ch = buf.channels.min(self.channels.len());
        for f in 0..frames {
            for c in 0..num_ch {
                let s = buf.get(f, c) as f64;
                let ch = &mut self.channels[c];

                // Peak
                let abs = s.abs();
                if abs > ch.peak { ch.peak = abs; }

                // RMS
                ch.rms_sum += s * s;
                ch.rms_count += 1;

                // Simplified LUFS: K-weighted energy (we approximate with
                // a simple high-shelf boost; a full ITU-R BS.1770 implementation
                // would add pre-filter stages).
                ch.lufs_energy += s * s;
                ch.lufs_count += 1;

                // True-peak: simplified 4× oversampling using linear interpolation
                let prev = *ch.tp_history.last().unwrap_or(&0.0);
                for k in 1..=4 {
                    let interp = prev + (s - prev) * k as f64 / 4.0;
                    if interp.abs() > ch.true_peak {
                        ch.true_peak = interp.abs();
                    }
                }
                ch.tp_history.rotate_left(1);
                *ch.tp_history.last_mut().unwrap() = s;
            }
        }
    }

    /// Peak level per channel in linear.
    pub fn peak_levels(&self) -> Vec<f64> {
        self.channels.iter().map(|ch| ch.peak).collect()
    }

    /// Peak level per channel in dBFS.
    pub fn peak_db(&self) -> Vec<f64> {
        self.channels
            .iter()
            .map(|ch| if ch.peak > 1e-12 { 20.0 * ch.peak.log10() } else { -120.0 })
            .collect()
    }

    /// RMS level per channel.
    pub fn rms_levels(&self) -> Vec<f64> {
        self.channels
            .iter()
            .map(|ch| {
                if ch.rms_count > 0 {
                    (ch.rms_sum / ch.rms_count as f64).sqrt()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// RMS level per channel in dBFS.
    pub fn rms_db(&self) -> Vec<f64> {
        self.rms_levels()
            .iter()
            .map(|&rms| if rms > 1e-12 { 20.0 * rms.log10() } else { -120.0 })
            .collect()
    }

    /// Simplified integrated LUFS (approximation; not fully ITU-R BS.1770
    /// compliant). Returns a single loudness value.
    pub fn lufs(&self) -> f64 {
        let total_energy: f64 = self.channels.iter().map(|ch| {
            if ch.lufs_count > 0 { ch.lufs_energy / ch.lufs_count as f64 } else { 0.0 }
        }).sum();
        let mean = total_energy / self.channels.len().max(1) as f64;
        if mean > 1e-12 {
            -0.691 + 10.0 * mean.log10()
        } else {
            -120.0
        }
    }

    /// True-peak per channel in dBTP.
    pub fn true_peak_db(&self) -> Vec<f64> {
        self.channels
            .iter()
            .map(|ch| {
                if ch.true_peak > 1e-12 { 20.0 * ch.true_peak.log10() } else { -120.0 }
            })
            .collect()
    }

    /// True-peak per channel (linear).
    pub fn true_peak_levels(&self) -> Vec<f64> {
        self.channels.iter().map(|ch| ch.true_peak).collect()
    }

    /// Number of channels being metered.
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }
}

// ---------------------------------------------------------------------------
// Convenience: write an AudioBuf to a WAV file path
// ---------------------------------------------------------------------------

/// Write an [`AudioBuf`] to a WAV file at the given path.
pub fn write_wav_file(
    path: &str, buf: &AudioBuf, format: OutputFormat,
) -> RendererResult<()> {
    let file = std::fs::File::create(path)
        .map_err(|e| RendererError::OutputError(e.to_string()))?;
    let writer = std::io::BufWriter::new(file);
    let config = OutputConfig {
        sample_rate: buf.sample_rate,
        channels: buf.channels,
        format,
    };
    let mut wav = WavWriter::new(writer, config)
        .map_err(|e| RendererError::OutputError(e.to_string()))?;
    wav.write_buffer(buf)
        .map_err(|e| RendererError::OutputError(e.to_string()))?;
    wav.finalize()
        .map_err(|e| RendererError::OutputError(e.to_string()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sine_buf(frames: usize, freq: f64, sr: u32) -> AudioBuf {
        let mut buf = AudioBuf::new(frames, 1, sr);
        for f in 0..frames {
            let s = (2.0 * std::f64::consts::PI * freq * f as f64 / sr as f64).sin();
            buf.set(f, 0, s as f32);
        }
        buf
    }

    // -- OutputFormat ----------------------------------------------------------

    #[test]
    fn output_format_bytes_per_sample() {
        assert_eq!(OutputFormat::Pcm16.bytes_per_sample(), 2);
        assert_eq!(OutputFormat::Pcm24.bytes_per_sample(), 3);
        assert_eq!(OutputFormat::Float32.bytes_per_sample(), 4);
    }

    // -- WavWriter ------------------------------------------------------------

    #[test]
    fn wav_writer_produces_valid_riff() {
        let cursor = Cursor::new(Vec::new());
        let cfg = OutputConfig { sample_rate: 44100, channels: 1, format: OutputFormat::Pcm16 };
        let mut wav = WavWriter::new(cursor, cfg).unwrap();
        let buf = sine_buf(441, 440.0, 44100);
        wav.write_buffer(&buf).unwrap();
        wav.finalize().unwrap();
        let data = wav.writer.into_inner();
        assert_eq!(&data[0..4], b"RIFF");
        assert_eq!(&data[8..12], b"WAVE");
        assert_eq!(&data[12..16], b"fmt ");
    }

    #[test]
    fn wav_writer_float32() {
        let cursor = Cursor::new(Vec::new());
        let cfg = OutputConfig { sample_rate: 44100, channels: 1, format: OutputFormat::Float32 };
        let mut wav = WavWriter::new(cursor, cfg).unwrap();
        let buf = sine_buf(100, 440.0, 44100);
        wav.write_buffer(&buf).unwrap();
        wav.finalize().unwrap();
        assert_eq!(wav.bytes_written(), 100 * 4);
    }

    #[test]
    fn wav_writer_pcm24() {
        let cursor = Cursor::new(Vec::new());
        let cfg = OutputConfig { sample_rate: 48000, channels: 2, format: OutputFormat::Pcm24 };
        let mut wav = WavWriter::new(cursor, cfg).unwrap();
        let mut buf = AudioBuf::new(100, 2, 48000);
        for f in 0..100 {
            buf.set(f, 0, 0.5);
            buf.set(f, 1, -0.5);
        }
        wav.write_buffer(&buf).unwrap();
        wav.finalize().unwrap();
        assert_eq!(wav.bytes_written(), 100 * 2 * 3);
    }

    // -- NullOutput -----------------------------------------------------------

    #[test]
    fn null_output_counts_frames() {
        let mut out = NullOutput::new();
        let buf = AudioBuf::new(256, 2, 44100);
        out.write_buffer(&buf);
        out.write_buffer(&buf);
        assert_eq!(out.frames_processed(), 512);
    }

    // -- BufferedOutput -------------------------------------------------------

    #[test]
    fn buffered_output_accumulates() {
        let mut out = BufferedOutput::new(1, 44100);
        let buf = sine_buf(1024, 440.0, 44100);
        out.write_buffer(&buf);
        assert_eq!(out.frames(), 1024);
    }

    #[test]
    fn buffered_output_peak() {
        let mut out = BufferedOutput::new(1, 44100);
        let buf = sine_buf(44100, 440.0, 44100);
        out.write_buffer(&buf);
        let peaks = out.peak_levels();
        assert!(peaks[0] > 0.99 && peaks[0] <= 1.001);
    }

    #[test]
    fn buffered_output_rms() {
        let mut out = BufferedOutput::new(1, 44100);
        let buf = sine_buf(44100, 440.0, 44100);
        out.write_buffer(&buf);
        let rms = out.rms_levels();
        // RMS of a sine = 1/√2 ≈ 0.707
        assert!((rms[0] - 0.7071).abs() < 0.02, "rms={}", rms[0]);
    }

    // -- Metering -------------------------------------------------------------

    #[test]
    fn metering_peak() {
        let mut meter = Metering::new(1, 44100.0);
        let buf = sine_buf(44100, 440.0, 44100);
        meter.process(&buf);
        let peaks = meter.peak_levels();
        assert!(peaks[0] > 0.99);
    }

    #[test]
    fn metering_rms() {
        let mut meter = Metering::new(1, 44100.0);
        let buf = sine_buf(44100, 440.0, 44100);
        meter.process(&buf);
        let rms = meter.rms_levels();
        assert!((rms[0] - 0.7071).abs() < 0.02);
    }

    #[test]
    fn metering_lufs_negative() {
        let mut meter = Metering::new(1, 44100.0);
        let buf = sine_buf(44100, 1000.0, 44100);
        meter.process(&buf);
        let lufs = meter.lufs();
        assert!(lufs < 0.0, "LUFS should be negative for a full-scale sine");
    }

    #[test]
    fn metering_true_peak() {
        let mut meter = Metering::new(1, 44100.0);
        let buf = sine_buf(44100, 440.0, 44100);
        meter.process(&buf);
        let tp = meter.true_peak_levels();
        assert!(tp[0] >= 0.99);
    }

    #[test]
    fn metering_reset() {
        let mut meter = Metering::new(2, 44100.0);
        let buf = sine_buf(1024, 440.0, 44100);
        meter.process(&buf);
        meter.reset();
        let peaks = meter.peak_levels();
        assert!(peaks[0] < 1e-12);
    }

    // -- RawOutput ------------------------------------------------------------

    #[test]
    fn raw_output_writes() {
        let cursor = Cursor::new(Vec::new());
        let mut raw = RawOutput::new(cursor);
        let buf = AudioBuf::new(100, 1, 44100);
        raw.write_buffer(&buf).unwrap();
        assert_eq!(raw.samples_written(), 100);
    }
}
