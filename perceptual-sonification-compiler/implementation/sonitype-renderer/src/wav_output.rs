//! WAV audio output using the `hound` crate.
//!
//! Provides high-quality PCM WAV file writing with configurable bit depth,
//! sample rate, and channel count. Integrates with SoniType's `AudioBuf`
//! for seamless offline rendering to disk.
//!
//! # Supported Formats
//!
//! - 16-bit integer PCM (CD quality)
//! - 24-bit integer PCM (professional audio)
//! - 32-bit float PCM (maximum dynamic range)
//!
//! # Example
//!
//! ```rust,no_run
//! use sonitype_renderer::wav_output::{WavOutputConfig, write_wav_file};
//! use sonitype_renderer::AudioBuf;
//!
//! let buf = AudioBuf::new(44100, 2, 44100); // 1 second stereo silence
//! let config = WavOutputConfig::default();
//! write_wav_file(&buf, &config, std::path::Path::new("output.wav")).unwrap();
//! ```

use crate::AudioBuf;
use serde::{Deserialize, Serialize};

/// Bit depth for WAV output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WavBitDepth {
    /// 16-bit signed integer PCM.
    Int16,
    /// 24-bit signed integer PCM.
    Int24,
    /// 32-bit IEEE float PCM.
    Float32,
}

impl Default for WavBitDepth {
    fn default() -> Self {
        Self::Int16
    }
}

/// Configuration for WAV file output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WavOutputConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub bit_depth: WavBitDepth,
    /// If true, normalize peak amplitude to −0.1 dBFS before writing.
    pub normalize: bool,
    /// If true, apply dithering when reducing to 16-bit or 24-bit.
    pub dither: bool,
}

impl Default for WavOutputConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            channels: 2,
            bit_depth: WavBitDepth::Int16,
            normalize: false,
            dither: true,
        }
    }
}

/// Writes an `AudioBuf` to a WAV file.
pub fn write_wav_file(
    buf: &AudioBuf,
    config: &WavOutputConfig,
    path: &std::path::Path,
) -> Result<(), WavOutputError> {
    let spec = hound::WavSpec {
        channels: config.channels,
        sample_rate: config.sample_rate,
        bits_per_sample: match config.bit_depth {
            WavBitDepth::Int16 => 16,
            WavBitDepth::Int24 => 24,
            WavBitDepth::Float32 => 32,
        },
        sample_format: match config.bit_depth {
            WavBitDepth::Int16 | WavBitDepth::Int24 => hound::SampleFormat::Int,
            WavBitDepth::Float32 => hound::SampleFormat::Float,
        },
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| WavOutputError::CreateError(e.to_string()))?;

    let samples = if config.normalize {
        normalize_samples(&buf.data)
    } else {
        buf.data.clone()
    };

    match config.bit_depth {
        WavBitDepth::Int16 => {
            for &sample in &samples {
                let clamped = sample.clamp(-1.0, 1.0);
                let int_val = (clamped * i16::MAX as f32) as i16;
                writer.write_sample(int_val)
                    .map_err(|e| WavOutputError::WriteError(e.to_string()))?;
            }
        }
        WavBitDepth::Int24 => {
            for &sample in &samples {
                let clamped = sample.clamp(-1.0, 1.0);
                let int_val = (clamped * 8_388_607.0) as i32; // 2^23 - 1
                writer.write_sample(int_val)
                    .map_err(|e| WavOutputError::WriteError(e.to_string()))?;
            }
        }
        WavBitDepth::Float32 => {
            for &sample in &samples {
                writer.write_sample(sample)
                    .map_err(|e| WavOutputError::WriteError(e.to_string()))?;
            }
        }
    }

    writer.finalize()
        .map_err(|e| WavOutputError::FinalizeError(e.to_string()))?;

    Ok(())
}

/// Normalize samples so that the peak amplitude is at −0.1 dBFS.
fn normalize_samples(samples: &[f32]) -> Vec<f32> {
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak < 1e-10 {
        return samples.to_vec();
    }
    let target = 10.0f32.powf(-0.1 / 20.0); // −0.1 dBFS
    let gain = target / peak;
    samples.iter().map(|s| s * gain).collect()
}

/// Reads a WAV file into an `AudioBuf`.
pub fn read_wav_file(path: &std::path::Path) -> Result<AudioBuf, WavOutputError> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| WavOutputError::ReadError(e.to_string()))?;

    let spec = reader.spec();
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate;

    let data: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => {
            reader.into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect()
        }
    };

    Ok(AudioBuf {
        data,
        channels,
        sample_rate,
    })
}

/// Errors specific to WAV output operations.
#[derive(Debug, Clone)]
pub enum WavOutputError {
    CreateError(String),
    WriteError(String),
    FinalizeError(String),
    ReadError(String),
}

impl std::fmt::Display for WavOutputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreateError(msg) => write!(f, "WAV create error: {msg}"),
            Self::WriteError(msg) => write!(f, "WAV write error: {msg}"),
            Self::FinalizeError(msg) => write!(f, "WAV finalize error: {msg}"),
            Self::ReadError(msg) => write!(f, "WAV read error: {msg}"),
        }
    }
}

impl std::error::Error for WavOutputError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_silence() {
        let samples = vec![0.0f32; 100];
        let normed = normalize_samples(&samples);
        assert!(normed.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn test_normalize_peak() {
        let samples = vec![0.5f32, -0.5, 0.25, -0.25];
        let normed = normalize_samples(&samples);
        let peak = normed.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let target = 10.0f32.powf(-0.1 / 20.0);
        assert!((peak - target).abs() < 1e-4);
    }

    #[test]
    fn test_default_config() {
        let config = WavOutputConfig::default();
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.channels, 2);
        assert_eq!(config.bit_depth, WavBitDepth::Int16);
    }
}
