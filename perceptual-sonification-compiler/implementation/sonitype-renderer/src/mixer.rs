//! Mixing, routing, and panning utilities.
//!
//! Provides stereo/multichannel mixers, crossfaders, channel splitters /
//! mergers, and send-return buses.

use crate::AudioBuf;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// StereoMixer
// ---------------------------------------------------------------------------

/// Input strip for the stereo mixer.
#[derive(Debug, Clone)]
pub struct StereoMixerInput {
    /// Linear gain (0..∞).
    pub gain: f64,
    /// Pan position (−1 = left, 0 = centre, +1 = right).
    pub pan: f64,
    /// Mute flag.
    pub mute: bool,
    /// Solo flag (handled by the mixer globally).
    pub solo: bool,
}

impl Default for StereoMixerInput {
    fn default() -> Self {
        Self { gain: 1.0, pan: 0.0, mute: false, solo: false }
    }
}

/// Sums N stereo inputs with per-channel gain and constant-power panning.
#[derive(Debug, Clone)]
pub struct StereoMixer {
    pub inputs: Vec<StereoMixerInput>,
    pub master_gain: f64,
}

impl StereoMixer {
    pub fn new(num_inputs: usize) -> Self {
        Self {
            inputs: vec![StereoMixerInput::default(); num_inputs],
            master_gain: 1.0,
        }
    }

    /// Set gain for an input channel.
    pub fn set_gain(&mut self, index: usize, gain: f64) {
        if let Some(inp) = self.inputs.get_mut(index) {
            inp.gain = gain;
        }
    }

    /// Set pan for an input channel.
    pub fn set_pan(&mut self, index: usize, pan: f64) {
        if let Some(inp) = self.inputs.get_mut(index) {
            inp.pan = pan.clamp(-1.0, 1.0);
        }
    }

    pub fn set_mute(&mut self, index: usize, mute: bool) {
        if let Some(inp) = self.inputs.get_mut(index) {
            inp.mute = mute;
        }
    }

    pub fn set_solo(&mut self, index: usize, solo: bool) {
        if let Some(inp) = self.inputs.get_mut(index) {
            inp.solo = solo;
        }
    }

    /// Constant-power pan: returns (left_gain, right_gain).
    #[inline]
    fn pan_gains(pan: f64) -> (f64, f64) {
        let angle = (pan + 1.0) * 0.25 * PI; // 0..π/2
        (angle.cos(), angle.sin())
    }

    /// Mix `sources` into `output` (must be stereo, i.e. 2 channels).
    ///
    /// Each source is a mono or stereo buffer. Mono sources are panned; stereo
    /// sources have pan applied as a balance control.
    pub fn process(&self, sources: &[&AudioBuf], output: &mut AudioBuf) {
        output.zero();
        let frames = output.frames();
        let any_solo = self.inputs.iter().any(|i| i.solo);

        for (idx, &src) in sources.iter().enumerate() {
            let inp = self.inputs.get(idx).cloned().unwrap_or_default();
            if inp.mute { continue; }
            if any_solo && !inp.solo { continue; }

            let (lg, rg) = Self::pan_gains(inp.pan);
            let gain = inp.gain;

            let src_frames = src.frames().min(frames);
            if src.channels == 1 {
                for f in 0..src_frames {
                    let s = src.get(f, 0) as f64 * gain;
                    let cur_l = output.get(f, 0) as f64;
                    let cur_r = output.get(f, 1) as f64;
                    output.set(f, 0, (cur_l + s * lg) as f32);
                    output.set(f, 1, (cur_r + s * rg) as f32);
                }
            } else {
                // Stereo: pan acts as balance
                for f in 0..src_frames {
                    let l = src.get(f, 0) as f64 * gain;
                    let r = src.get(f, 1.min(src.channels - 1)) as f64 * gain;
                    let cur_l = output.get(f, 0) as f64;
                    let cur_r = output.get(f, 1) as f64;
                    output.set(f, 0, (cur_l + l * lg) as f32);
                    output.set(f, 1, (cur_r + r * rg) as f32);
                }
            }
        }

        // Master gain
        if (self.master_gain - 1.0).abs() > 1e-12 {
            output.apply_gain(self.master_gain as f32);
        }
    }
}

// ---------------------------------------------------------------------------
// MultichannelMixer
// ---------------------------------------------------------------------------

/// N-input, M-output mixing matrix with per-element gain.
#[derive(Debug, Clone)]
pub struct MultichannelMixer {
    /// matrix\[output\]\[input\] = gain.
    pub matrix: Vec<Vec<f64>>,
    pub num_inputs: usize,
    pub num_outputs: usize,
}

impl MultichannelMixer {
    /// Create an N→M mixer. Matrix is initialised to identity where possible.
    pub fn new(num_inputs: usize, num_outputs: usize) -> Self {
        let mut matrix = vec![vec![0.0; num_inputs]; num_outputs];
        for i in 0..num_inputs.min(num_outputs) {
            matrix[i][i] = 1.0;
        }
        Self { matrix, num_inputs, num_outputs }
    }

    /// Set gain for a specific input→output pair.
    pub fn set_gain(&mut self, output_ch: usize, input_ch: usize, gain: f64) {
        if output_ch < self.num_outputs && input_ch < self.num_inputs {
            self.matrix[output_ch][input_ch] = gain;
        }
    }

    /// Get gain for a specific pair.
    pub fn get_gain(&self, output_ch: usize, input_ch: usize) -> f64 {
        if output_ch < self.num_outputs && input_ch < self.num_inputs {
            self.matrix[output_ch][input_ch]
        } else {
            0.0
        }
    }

    /// Process: `inputs` is a slice of mono buffers (one per input channel).
    /// `output` must have `num_outputs` channels.
    pub fn process(&self, inputs: &[&AudioBuf], output: &mut AudioBuf) {
        output.zero();
        let frames = output.frames();
        for out_ch in 0..self.num_outputs.min(output.channels) {
            for (in_ch, &inp) in inputs.iter().enumerate().take(self.num_inputs) {
                let gain = self.matrix[out_ch][in_ch];
                if gain.abs() < 1e-12 { continue; }
                let n = inp.frames().min(frames);
                for f in 0..n {
                    let cur = output.get(f, out_ch) as f64;
                    let s = inp.get(f, 0) as f64 * gain;
                    output.set(f, out_ch, (cur + s) as f32);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Crossfader
// ---------------------------------------------------------------------------

/// Crossfade curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossfadeCurve {
    Linear,
    EqualPower,
    SCurve,
}

/// Smooth transition between two audio sources.
#[derive(Debug, Clone)]
pub struct Crossfader {
    /// 0.0 = fully source A, 1.0 = fully source B.
    pub position: f64,
    pub curve: CrossfadeCurve,
}

impl Crossfader {
    pub fn new(curve: CrossfadeCurve) -> Self {
        Self { position: 0.0, curve }
    }

    pub fn set_position(&mut self, pos: f64) {
        self.position = pos.clamp(0.0, 1.0);
    }

    /// Compute (gain_a, gain_b) for the current position.
    pub fn gains(&self) -> (f64, f64) {
        let p = self.position;
        match self.curve {
            CrossfadeCurve::Linear => (1.0 - p, p),
            CrossfadeCurve::EqualPower => {
                let angle = p * 0.5 * PI;
                (angle.cos(), angle.sin())
            }
            CrossfadeCurve::SCurve => {
                // Hermite S-curve: 3p² − 2p³
                let s = 3.0 * p * p - 2.0 * p * p * p;
                (1.0 - s, s)
            }
        }
    }

    /// Mix two buffers into `output`.
    pub fn process(&self, a: &AudioBuf, b: &AudioBuf, output: &mut AudioBuf) {
        let (ga, gb) = self.gains();
        let frames = output.frames().min(a.frames()).min(b.frames());
        let ch = output.channels.min(a.channels).min(b.channels);
        for f in 0..frames {
            for c in 0..ch {
                let va = a.get(f, c) as f64 * ga;
                let vb = b.get(f, c) as f64 * gb;
                output.set(f, c, (va + vb) as f32);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ChannelSplitter
// ---------------------------------------------------------------------------

/// Splits a mono signal into stereo (or multi-channel) by duplicating or
/// distributing.
#[derive(Debug, Clone)]
pub struct ChannelSplitter;

impl ChannelSplitter {
    /// Mono → Stereo: duplicate mono into both channels.
    pub fn mono_to_stereo(input: &AudioBuf, output: &mut AudioBuf) {
        let frames = input.frames().min(output.frames());
        for f in 0..frames {
            let s = input.get(f, 0);
            output.set(f, 0, s);
            if output.channels > 1 {
                output.set(f, 1, s);
            }
        }
    }

    /// Stereo → multi-channel: copy each stereo channel to corresponding
    /// output channels, zeroing the rest.
    pub fn stereo_to_multi(input: &AudioBuf, output: &mut AudioBuf) {
        output.zero();
        let frames = input.frames().min(output.frames());
        let src_ch = input.channels.min(output.channels);
        for f in 0..frames {
            for c in 0..src_ch {
                output.set(f, c, input.get(f, c));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ChannelMerger
// ---------------------------------------------------------------------------

/// Merges multi-channel audio down to stereo or mono.
#[derive(Debug, Clone)]
pub struct ChannelMerger;

impl ChannelMerger {
    /// Stereo → mono: average L and R.
    pub fn stereo_to_mono(input: &AudioBuf, output: &mut AudioBuf) {
        let frames = input.frames().min(output.frames());
        for f in 0..frames {
            let l = input.get(f, 0) as f64;
            let r = if input.channels > 1 { input.get(f, 1) as f64 } else { l };
            output.set(f, 0, ((l + r) * 0.5) as f32);
        }
    }

    /// Multi-channel → stereo down-mix: evenly distribute odd channels to L,
    /// even channels to R.
    pub fn multi_to_stereo(input: &AudioBuf, output: &mut AudioBuf) {
        output.zero();
        let frames = input.frames().min(output.frames());
        let n = input.channels;
        if n == 0 { return; }
        let gain = 1.0 / (n as f64 / 2.0).max(1.0);
        for f in 0..frames {
            let mut l = 0.0f64;
            let mut r = 0.0f64;
            for c in 0..n {
                let s = input.get(f, c) as f64;
                if c % 2 == 0 { l += s; } else { r += s; }
            }
            output.set(f, 0, (l * gain) as f32);
            if output.channels > 1 {
                output.set(f, 1, (r * gain) as f32);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SendReturn
// ---------------------------------------------------------------------------

/// Auxiliary send / return bus with wet / dry mix.
#[derive(Debug, Clone)]
pub struct SendReturn {
    /// Send level (0..1).
    pub send_level: f64,
    /// Return level (0..1).
    pub return_level: f64,
    /// Wet/dry mix (0 = dry only, 1 = wet only).
    pub mix: f64,
    send_buffer: AudioBuf,
}

impl SendReturn {
    pub fn new(frames: usize, channels: usize, sample_rate: u32) -> Self {
        Self {
            send_level: 1.0,
            return_level: 1.0,
            mix: 0.5,
            send_buffer: AudioBuf::new(frames, channels, sample_rate),
        }
    }

    pub fn set_send_level(&mut self, level: f64) {
        self.send_level = level.clamp(0.0, 1.0);
    }

    pub fn set_return_level(&mut self, level: f64) {
        self.return_level = level.clamp(0.0, 1.0);
    }

    pub fn set_mix(&mut self, mix: f64) {
        self.mix = mix.clamp(0.0, 1.0);
    }

    /// Get the send buffer (the effect processor reads from this).
    pub fn send_buffer(&self) -> &AudioBuf {
        &self.send_buffer
    }

    /// Fill the send buffer by copying from `dry` at the configured send level.
    pub fn fill_send(&mut self, dry: &AudioBuf) {
        let frames = dry.frames().min(self.send_buffer.frames());
        let ch = dry.channels.min(self.send_buffer.channels);
        for f in 0..frames {
            for c in 0..ch {
                self.send_buffer.set(f, c, dry.get(f, c) * self.send_level as f32);
            }
        }
    }

    /// Mix dry + wet (return from effect) into `output`.
    pub fn mix_return(&self, dry: &AudioBuf, wet: &AudioBuf, output: &mut AudioBuf) {
        let frames = output.frames().min(dry.frames()).min(wet.frames());
        let ch = output.channels.min(dry.channels).min(wet.channels);
        let dry_gain = (1.0 - self.mix) as f32;
        let wet_gain = (self.mix * self.return_level) as f32;
        for f in 0..frames {
            for c in 0..ch {
                let d = dry.get(f, c) * dry_gain;
                let w = wet.get(f, c) * wet_gain;
                output.set(f, c, d + w);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mono_buf(frames: usize, value: f32) -> AudioBuf {
        let mut b = AudioBuf::new(frames, 1, 44100);
        for s in b.data.iter_mut() { *s = value; }
        b
    }

    fn stereo_buf(frames: usize, l: f32, r: f32) -> AudioBuf {
        let mut b = AudioBuf::new(frames, 2, 44100);
        for f in 0..frames {
            b.set(f, 0, l);
            b.set(f, 1, r);
        }
        b
    }

    // -- StereoMixer ----------------------------------------------------------

    #[test]
    fn stereo_mixer_single_mono_centre() {
        let mixer = StereoMixer::new(1);
        let src = mono_buf(64, 1.0);
        let mut out = AudioBuf::new(64, 2, 44100);
        mixer.process(&[&src], &mut out);
        let l = out.get(0, 0);
        let r = out.get(0, 1);
        // Centre pan → equal power → both ≈ 0.707
        assert!((l - r).abs() < 0.01, "centre should be equal L/R");
        assert!(l > 0.5);
    }

    #[test]
    fn stereo_mixer_pan_left() {
        let mut mixer = StereoMixer::new(1);
        mixer.set_pan(0, -1.0);
        let src = mono_buf(64, 1.0);
        let mut out = AudioBuf::new(64, 2, 44100);
        mixer.process(&[&src], &mut out);
        assert!(out.get(0, 0) > 0.9);
        assert!(out.get(0, 1) < 0.01);
    }

    #[test]
    fn stereo_mixer_mute() {
        let mut mixer = StereoMixer::new(1);
        mixer.set_mute(0, true);
        let src = mono_buf(64, 1.0);
        let mut out = AudioBuf::new(64, 2, 44100);
        mixer.process(&[&src], &mut out);
        assert!(out.get(0, 0).abs() < 1e-6);
    }

    #[test]
    fn stereo_mixer_solo() {
        let mut mixer = StereoMixer::new(2);
        mixer.set_solo(1, true);
        let src0 = mono_buf(64, 0.5);
        let src1 = mono_buf(64, 1.0);
        let mut out = AudioBuf::new(64, 2, 44100);
        mixer.process(&[&src0, &src1], &mut out);
        // Only source 1 should be audible
        // With centre pan, both L and R should be non-zero
        let l = out.get(0, 0);
        assert!(l > 0.5, "solo channel should be audible");
    }

    #[test]
    fn stereo_mixer_master_gain() {
        let mut mixer = StereoMixer::new(1);
        mixer.master_gain = 0.5;
        let src = mono_buf(64, 1.0);
        let mut out = AudioBuf::new(64, 2, 44100);
        mixer.process(&[&src], &mut out);
        assert!(out.get(0, 0) < 0.55);
    }

    // -- MultichannelMixer ----------------------------------------------------

    #[test]
    fn multichannel_identity() {
        let mixer = MultichannelMixer::new(2, 2);
        let a = mono_buf(64, 0.7);
        let b = mono_buf(64, 0.3);
        let mut out = AudioBuf::new(64, 2, 44100);
        mixer.process(&[&a, &b], &mut out);
        assert!((out.get(0, 0) - 0.7).abs() < 1e-4);
        assert!((out.get(0, 1) - 0.3).abs() < 1e-4);
    }

    #[test]
    fn multichannel_custom_routing() {
        let mut mixer = MultichannelMixer::new(2, 2);
        mixer.set_gain(0, 0, 0.5);
        mixer.set_gain(0, 1, 0.5);
        mixer.set_gain(1, 0, 0.0);
        mixer.set_gain(1, 1, 1.0);
        let a = mono_buf(64, 1.0);
        let b = mono_buf(64, 1.0);
        let mut out = AudioBuf::new(64, 2, 44100);
        mixer.process(&[&a, &b], &mut out);
        assert!((out.get(0, 0) - 1.0).abs() < 1e-4); // 0.5 + 0.5
        assert!((out.get(0, 1) - 1.0).abs() < 1e-4); // 0 + 1
    }

    // -- Crossfader -----------------------------------------------------------

    #[test]
    fn crossfade_linear_midpoint() {
        let cf = Crossfader { position: 0.5, curve: CrossfadeCurve::Linear };
        let (ga, gb) = cf.gains();
        assert!((ga - 0.5).abs() < 1e-6);
        assert!((gb - 0.5).abs() < 1e-6);
    }

    #[test]
    fn crossfade_equal_power_unity() {
        // Equal-power: ga² + gb² should be constant ≈ 1
        let cf = Crossfader { position: 0.5, curve: CrossfadeCurve::EqualPower };
        let (ga, gb) = cf.gains();
        let power = ga * ga + gb * gb;
        assert!((power - 1.0).abs() < 0.01);
    }

    #[test]
    fn crossfade_process() {
        let cf = Crossfader { position: 0.5, curve: CrossfadeCurve::Linear };
        let a = mono_buf(64, 1.0);
        let b = mono_buf(64, 0.0);
        let mut out = AudioBuf::new(64, 1, 44100);
        cf.process(&a, &b, &mut out);
        assert!((out.get(0, 0) - 0.5).abs() < 1e-4);
    }

    // -- ChannelSplitter / Merger ---------------------------------------------

    #[test]
    fn splitter_mono_to_stereo() {
        let mono = mono_buf(64, 0.75);
        let mut stereo = AudioBuf::new(64, 2, 44100);
        ChannelSplitter::mono_to_stereo(&mono, &mut stereo);
        assert!((stereo.get(0, 0) - 0.75).abs() < 1e-6);
        assert!((stereo.get(0, 1) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn merger_stereo_to_mono() {
        let stereo = stereo_buf(64, 0.6, 0.4);
        let mut mono = AudioBuf::new(64, 1, 44100);
        ChannelMerger::stereo_to_mono(&stereo, &mut mono);
        assert!((mono.get(0, 0) - 0.5).abs() < 1e-4);
    }

    // -- SendReturn -----------------------------------------------------------

    #[test]
    fn send_return_dry_only() {
        let sr = SendReturn { send_level: 1.0, return_level: 1.0, mix: 0.0,
            send_buffer: AudioBuf::new(64, 1, 44100) };
        let dry = mono_buf(64, 0.8);
        let wet = mono_buf(64, 0.5);
        let mut out = AudioBuf::new(64, 1, 44100);
        sr.mix_return(&dry, &wet, &mut out);
        assert!((out.get(0, 0) - 0.8).abs() < 1e-4);
    }

    #[test]
    fn send_return_wet_only() {
        let sr = SendReturn { send_level: 1.0, return_level: 1.0, mix: 1.0,
            send_buffer: AudioBuf::new(64, 1, 44100) };
        let dry = mono_buf(64, 0.8);
        let wet = mono_buf(64, 0.5);
        let mut out = AudioBuf::new(64, 1, 44100);
        sr.mix_return(&dry, &wet, &mut out);
        assert!((out.get(0, 0) - 0.5).abs() < 1e-4);
    }
}
