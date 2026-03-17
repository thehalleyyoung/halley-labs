//! Trace replay.
use serde::{Serialize, Deserialize};

/// Trace playback modes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PlaybackMode { Forward, Reverse, PingPong }

/// Playback speed control.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PlaybackSpeed { pub multiplier: f64 }
impl Default for PlaybackSpeed { fn default() -> Self { Self { multiplier: 1.0 } } }

/// Plays back recorded simulation traces.
#[derive(Debug, Clone)]
pub struct TracePlayer { pub mode: PlaybackMode, pub speed: PlaybackSpeed, current_index: usize }
impl Default for TracePlayer { fn default() -> Self { Self { mode: PlaybackMode::Forward, speed: PlaybackSpeed::default(), current_index: 0 } } }
impl TracePlayer {
    /// Advance to the next frame.
    pub fn advance(&mut self) { self.current_index += 1; }
    /// Get the current frame index.
    pub fn current_frame(&self) -> usize { self.current_index }
    /// Reset to beginning.
    pub fn reset(&mut self) { self.current_index = 0; }
}
