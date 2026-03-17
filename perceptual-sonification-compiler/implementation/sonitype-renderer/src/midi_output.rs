//! MIDI output support for SoniType sonification rendering.
//!
//! Converts optimized audio graph parameters into MIDI events using the `midly`
//! crate. Supports both Standard MIDI File (SMF) export and real-time MIDI
//! streaming for integration with external synthesizers and DAWs.
//!
//! # MIDI Mapping Strategy
//!
//! SoniType maps perceptual parameters to MIDI as follows:
//! - **Pitch** (Hz) → MIDI note number (0–127) via `hz_to_midi()`
//! - **Loudness** (dB SPL) → MIDI velocity (1–127) via perceptual scaling
//! - **Timbre** → MIDI program change (instrument selection)
//! - **Pan** → MIDI CC#10 (pan position)
//! - **Stream identity** → MIDI channel (0–15)
//!
//! The mapping preserves JND-level discriminability guarantees: adjacent MIDI
//! note numbers that fall below the pitch JND threshold are flagged by the
//! perceptual linter.

use serde::{Deserialize, Serialize};

/// Configuration for MIDI output generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidiOutputConfig {
    /// Ticks per quarter note (MIDI resolution). Default: 480.
    pub ticks_per_quarter: u16,
    /// Tempo in BPM. Default: 120.
    pub tempo_bpm: f64,
    /// Whether to quantize note timing to the nearest tick.
    pub quantize: bool,
    /// Minimum velocity to emit (avoids inaudible notes). Default: 1.
    pub min_velocity: u8,
    /// Maximum number of simultaneous MIDI channels. Default: 16.
    pub max_channels: u8,
}

impl Default for MidiOutputConfig {
    fn default() -> Self {
        Self {
            ticks_per_quarter: 480,
            tempo_bpm: 120.0,
            quantize: true,
            min_velocity: 1,
            max_channels: 16,
        }
    }
}

/// A single MIDI event in the SoniType intermediate representation.
#[derive(Debug, Clone)]
pub struct SoniMidiEvent {
    /// Absolute tick position from the start of the sequence.
    pub tick: u32,
    /// MIDI channel (0–15).
    pub channel: u8,
    /// Event payload.
    pub kind: SoniMidiEventKind,
}

/// Kinds of MIDI events that SoniType can generate.
#[derive(Debug, Clone)]
pub enum SoniMidiEventKind {
    NoteOn { note: u8, velocity: u8 },
    NoteOff { note: u8, velocity: u8 },
    ProgramChange { program: u8 },
    ControlChange { controller: u8, value: u8 },
    PitchBend { bend: i16 },
    TempoChange { microseconds_per_beat: u32 },
}

/// Converts a frequency in Hz to the nearest MIDI note number.
///
/// Uses the standard formula: `note = 69 + 12 * log2(freq / 440)`.
/// Returns `None` if the frequency is out of MIDI range (roughly 8.18–12543 Hz).
pub fn hz_to_midi_note(freq_hz: f64) -> Option<u8> {
    if freq_hz <= 0.0 {
        return None;
    }
    let note = 69.0 + 12.0 * (freq_hz / 440.0).log2();
    let rounded = note.round() as i32;
    if (0..=127).contains(&rounded) {
        Some(rounded as u8)
    } else {
        None
    }
}

/// Converts a MIDI note number back to frequency in Hz.
pub fn midi_note_to_hz(note: u8) -> f64 {
    440.0 * 2.0f64.powf((note as f64 - 69.0) / 12.0)
}

/// Maps a loudness value (dB SPL, typically 30–90 range) to MIDI velocity.
pub fn loudness_to_velocity(db_spl: f64, min_db: f64, max_db: f64) -> u8 {
    let normalized = ((db_spl - min_db) / (max_db - min_db)).clamp(0.0, 1.0);
    // Use a power curve for more perceptually uniform mapping.
    let velocity = 1.0 + 126.0 * normalized.powf(0.75);
    (velocity.round() as u8).clamp(1, 127)
}

/// Maps a pan position (−1.0 = full left, +1.0 = full right) to MIDI CC#10.
pub fn pan_to_cc(pan: f64) -> u8 {
    let normalized = (pan + 1.0) / 2.0;
    (normalized * 127.0).round().clamp(0.0, 127.0) as u8
}

/// Writes a sequence of SoniType MIDI events to a Standard MIDI File (SMF)
/// using the `midly` crate.
///
/// # Arguments
/// * `events` – Sorted sequence of MIDI events.
/// * `config` – MIDI output configuration.
/// * `path` – File system path to write the `.mid` file.
pub fn write_midi_file(
    events: &[SoniMidiEvent],
    config: &MidiOutputConfig,
    path: &std::path::Path,
) -> Result<(), MidiOutputError> {
    use midly::{
        Format, Header, MidiMessage, Smf, Timing, Track, TrackEvent, TrackEventKind,
    };

    let timing = Timing::Metrical(midly::num::u15::new(config.ticks_per_quarter));
    let header = Header::new(Format::Parallel, timing);
    let mut smf = Smf::new(header);

    // Group events by channel → one MIDI track per channel.
    let max_ch = events.iter().map(|e| e.channel).max().unwrap_or(0) as usize;
    let mut tracks: Vec<Vec<&SoniMidiEvent>> = vec![vec![]; max_ch + 1];
    for ev in events {
        tracks[ev.channel as usize].push(ev);
    }

    // Add a conductor track with tempo.
    let tempo_us = (60_000_000.0 / config.tempo_bpm).round() as u32;
    let mut conductor: Track = vec![];
    conductor.push(TrackEvent {
        delta: 0.into(),
        kind: TrackEventKind::Meta(midly::MetaMessage::Tempo(
            midly::num::u24::new(tempo_us),
        )),
    });
    conductor.push(TrackEvent {
        delta: 0.into(),
        kind: TrackEventKind::Meta(midly::MetaMessage::EndOfTrack),
    });
    smf.tracks.push(conductor);

    // Convert each channel's events to a MIDI track.
    for (ch_idx, ch_events) in tracks.iter().enumerate() {
        let ch = midly::num::u4::new(ch_idx as u8);
        let mut track: Track = vec![];
        let mut last_tick: u32 = 0;

        for ev in ch_events {
            let delta = ev.tick.saturating_sub(last_tick);
            last_tick = ev.tick;

            let kind = match &ev.kind {
                SoniMidiEventKind::NoteOn { note, velocity } => {
                    TrackEventKind::Midi {
                        channel: ch,
                        message: MidiMessage::NoteOn {
                            key: midly::num::u7::new(*note),
                            vel: midly::num::u7::new(*velocity),
                        },
                    }
                }
                SoniMidiEventKind::NoteOff { note, velocity } => {
                    TrackEventKind::Midi {
                        channel: ch,
                        message: MidiMessage::NoteOff {
                            key: midly::num::u7::new(*note),
                            vel: midly::num::u7::new(*velocity),
                        },
                    }
                }
                SoniMidiEventKind::ProgramChange { program } => {
                    TrackEventKind::Midi {
                        channel: ch,
                        message: MidiMessage::ProgramChange {
                            program: midly::num::u7::new(*program),
                        },
                    }
                }
                SoniMidiEventKind::ControlChange { controller, value } => {
                    TrackEventKind::Midi {
                        channel: ch,
                        message: MidiMessage::Controller {
                            controller: midly::num::u7::new(*controller),
                            value: midly::num::u7::new(*value),
                        },
                    }
                }
                SoniMidiEventKind::PitchBend { bend } => {
                    let raw = (*bend as i32 + 8192).clamp(0, 16383) as u16;
                    TrackEventKind::Midi {
                        channel: ch,
                        message: MidiMessage::PitchBend {
                            bend: midly::PitchBend(midly::num::u14::new(raw)),
                        },
                    }
                }
                SoniMidiEventKind::TempoChange { microseconds_per_beat } => {
                    TrackEventKind::Meta(midly::MetaMessage::Tempo(
                        midly::num::u24::new(*microseconds_per_beat),
                    ))
                }
            };

            track.push(TrackEvent {
                delta: delta.into(),
                kind,
            });
        }

        track.push(TrackEvent {
            delta: 0.into(),
            kind: TrackEventKind::Meta(midly::MetaMessage::EndOfTrack),
        });

        smf.tracks.push(track);
    }

    smf.save(path).map_err(|e| MidiOutputError::WriteError(e.to_string()))?;
    Ok(())
}

/// Errors specific to MIDI output.
#[derive(Debug, Clone)]
pub enum MidiOutputError {
    FrequencyOutOfRange(f64),
    TooManyChannels(usize),
    WriteError(String),
}

impl std::fmt::Display for MidiOutputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FrequencyOutOfRange(hz) => write!(f, "frequency {hz:.1} Hz out of MIDI range"),
            Self::TooManyChannels(n) => write!(f, "{n} streams exceed 16 MIDI channels"),
            Self::WriteError(msg) => write!(f, "MIDI write error: {msg}"),
        }
    }
}

impl std::error::Error for MidiOutputError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hz_to_midi_a440() {
        assert_eq!(hz_to_midi_note(440.0), Some(69));
    }

    #[test]
    fn test_hz_to_midi_c4() {
        assert_eq!(hz_to_midi_note(261.63), Some(60));
    }

    #[test]
    fn test_midi_note_to_hz_roundtrip() {
        for note in 21..=108 {
            let hz = midi_note_to_hz(note);
            assert_eq!(hz_to_midi_note(hz), Some(note));
        }
    }

    #[test]
    fn test_loudness_to_velocity_range() {
        let v_low = loudness_to_velocity(30.0, 30.0, 90.0);
        let v_high = loudness_to_velocity(90.0, 30.0, 90.0);
        assert!(v_low >= 1);
        assert!(v_high <= 127);
        assert!(v_high > v_low);
    }

    #[test]
    fn test_pan_center() {
        assert_eq!(pan_to_cc(0.0), 64);
    }
}
