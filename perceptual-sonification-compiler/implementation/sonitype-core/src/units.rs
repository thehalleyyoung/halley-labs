//! Unit conversion utilities for psychoacoustic scales.
//!
//! Provides conversions between frequency (Hz), Bark, Mel, ERB scales,
//! MIDI note numbers, musical intervals, and amplitude/dB SPL.

use std::f64::consts::PI;

// ===========================================================================
// Hz <-> Bark (Zwicker & Terhardt critical-band-rate scale)
// ===========================================================================

/// Convert a frequency in Hz to the Bark scale.
///
/// Uses the Zwicker formula:
///   bark = 13 * arctan(0.00076 * f) + 3.5 * arctan((f / 7500)^2)
pub fn hz_to_bark(freq_hz: f64) -> f64 {
    13.0 * (0.00076 * freq_hz).atan() + 3.5 * ((freq_hz / 7500.0).powi(2)).atan()
}

/// Convert a Bark value back to Hz (iterative Newton-Raphson inversion).
pub fn bark_to_hz(bark: f64) -> f64 {
    // Initial guess via Traunmuller approximation
    let mut f = 1960.0 * (bark + 0.53) / (26.28 - bark);
    if f < 0.0 { f = 1.0; }
    for _ in 0..50 {
        let b = hz_to_bark(f);
        let err = b - bark;
        if err.abs() < 1e-6 { break; }
        let df = 0.01;
        let db = hz_to_bark(f + df) - b;
        let deriv = db / df;
        if deriv.abs() < 1e-15 { break; }
        f -= err / deriv;
        if f < 0.0 { f = 1.0; }
    }
    f
}

/// Bandwidth of a critical band at a given frequency in Hz.
///
/// Approximation: BW ~ 52.548 + 75 * (1 + 1.4 * (f/1000)^2)^0.69
pub fn bark_bandwidth(freq_hz: f64) -> f64 {
    52.548 + 75.0 * (1.0 + 1.4 * (freq_hz / 1000.0).powi(2)).powf(0.69)
}

// ===========================================================================
// Hz <-> Mel
// ===========================================================================

/// Convert Hz to the Mel scale: mel = 2595 * log10(1 + f/700).
pub fn hz_to_mel(freq_hz: f64) -> f64 {
    2595.0 * (1.0 + freq_hz / 700.0).log10()
}

/// Convert Mel back to Hz: f = 700 * (10^(mel/2595) - 1).
pub fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

// ===========================================================================
// Hz <-> ERB (Equivalent Rectangular Bandwidth)
// ===========================================================================

/// ERB at a given frequency (Glasberg & Moore 1990):
///   ERB(f) = 24.7 * (4.37 * f/1000 + 1)
pub fn erb_of_frequency(freq_hz: f64) -> f64 {
    24.7 * (4.37 * freq_hz / 1000.0 + 1.0)
}

/// Convert Hz to ERB-rate (number of ERBs from 0 Hz):
///   ERB-rate = 21.4 * log10(4.37 * f/1000 + 1)
pub fn hz_to_erb_rate(freq_hz: f64) -> f64 {
    21.4 * (4.37 * freq_hz / 1000.0 + 1.0).log10()
}

/// Convert ERB-rate back to Hz.
pub fn erb_rate_to_hz(erb_rate: f64) -> f64 {
    (10.0_f64.powf(erb_rate / 21.4) - 1.0) * 1000.0 / 4.37
}

// ===========================================================================
// Amplitude <-> dB SPL
// ===========================================================================

/// Convert a linear amplitude to dB (relative to unity).
///
/// Returns `f64::NEG_INFINITY` for amplitude <= 0.
pub fn amplitude_to_db_spl(amplitude: f64) -> f64 {
    if amplitude <= 0.0 { return f64::NEG_INFINITY; }
    20.0 * amplitude.log10()
}

/// Convert dB to linear amplitude.
pub fn db_spl_to_amplitude(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

// ===========================================================================
// MIDI <-> Hz
// ===========================================================================

/// Convert a MIDI note number to Hz: f = 440 * 2^((note - 69) / 12).
pub fn midi_to_hz(note: f64) -> f64 {
    440.0 * 2.0_f64.powf((note - 69.0) / 12.0)
}

/// Convert Hz to a (possibly fractional) MIDI note number.
pub fn hz_to_midi(freq_hz: f64) -> f64 {
    69.0 + 12.0 * (freq_hz / 440.0).log2()
}

// ===========================================================================
// Musical interval helpers
// ===========================================================================

/// Convert semitones to a frequency ratio.
pub fn semitones_to_ratio(semitones: f64) -> f64 {
    2.0_f64.powf(semitones / 12.0)
}

/// Convert cents (1/100 of a semitone) to a frequency ratio.
pub fn cents_to_ratio(cents: f64) -> f64 {
    2.0_f64.powf(cents / 1200.0)
}

/// Convert a frequency ratio to cents.
pub fn ratio_to_cents(ratio: f64) -> f64 {
    1200.0 * ratio.log2()
}

/// Convert a frequency ratio to semitones.
pub fn ratio_to_semitones(ratio: f64) -> f64 {
    12.0 * ratio.log2()
}

// ===========================================================================
// Critical band rate (Traunmuller approximation)
// ===========================================================================

/// Critical-band rate by Traunmuller (1990):
///   z = (26.81 * f) / (1960 + f) - 0.53
pub fn critical_band_rate(freq_hz: f64) -> f64 {
    let z = (26.81 * freq_hz) / (1960.0 + freq_hz) - 0.53;
    if z < 2.0 {
        z + 0.15 * (2.0 - z)
    } else if z > 20.1 {
        z + 0.22 * (z - 20.1)
    } else {
        z
    }
}

// ===========================================================================
// Loudness weighting helpers
// ===========================================================================

/// A-weighting approximation for a frequency in Hz (returns linear gain).
pub fn a_weighting(freq_hz: f64) -> f64 {
    let f2 = freq_hz * freq_hz;
    let numerator = 12194.0_f64.powi(2) * f2 * f2;
    let denominator = (f2 + 20.6_f64.powi(2))
        * ((f2 + 107.7_f64.powi(2)) * (f2 + 737.9_f64.powi(2))).sqrt()
        * (f2 + 12194.0_f64.powi(2));
    if denominator.abs() < 1e-30 { return 0.0; }
    let ra = numerator / denominator;
    let ref_val = {
        let rf2 = 1_000_000.0_f64;
        let rn = 12194.0_f64.powi(2) * rf2 * rf2;
        let rd = (rf2 + 20.6_f64.powi(2))
            * ((rf2 + 107.7_f64.powi(2)) * (rf2 + 737.9_f64.powi(2))).sqrt()
            * (rf2 + 12194.0_f64.powi(2));
        rn / rd
    };
    ra / ref_val
}

/// Absolute threshold of hearing in dB SPL (Terhardt approximation).
pub fn absolute_threshold_of_hearing(freq_hz: f64) -> f64 {
    let f_khz = freq_hz / 1000.0;
    3.64 * f_khz.powf(-0.8) - 6.5 * (-0.6 * (f_khz - 3.3).powi(2)).exp()
        + 1e-3 * f_khz.powi(4)
}

/// Wavelength in meters for a given frequency at a given speed of sound.
pub fn wavelength(freq_hz: f64, speed_of_sound: f64) -> f64 {
    if freq_hz.abs() < 1e-15 { return f64::INFINITY; }
    speed_of_sound / freq_hz
}

/// Period in seconds for a given frequency.
pub fn period(freq_hz: f64) -> f64 {
    if freq_hz.abs() < 1e-15 { return f64::INFINITY; }
    1.0 / freq_hz
}

/// Convert angular frequency (radians/sec) to Hz.
pub fn angular_to_hz(omega: f64) -> f64 {
    omega / (2.0 * PI)
}

/// Convert Hz to angular frequency (radians/sec).
pub fn hz_to_angular(freq_hz: f64) -> f64 {
    2.0 * PI * freq_hz
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    const EPS: f64 = 1e-2;

    #[test]
    fn bark_at_known_frequencies() {
        let b = hz_to_bark(100.0);
        assert!((b - 1.0).abs() < 0.15, "100 Hz -> {b} Bark");
        let b = hz_to_bark(1000.0);
        assert!((b - 8.5).abs() < 0.5, "1000 Hz -> {b} Bark");
    }

    #[test]
    fn bark_roundtrip() {
        for &f in &[100.0, 440.0, 1000.0, 4000.0, 8000.0, 15000.0] {
            let b = hz_to_bark(f);
            let f2 = bark_to_hz(b);
            assert!((f - f2).abs() < 1.0, "Roundtrip: {f} -> {b} -> {f2}");
        }
    }

    #[test]
    fn mel_at_1000hz() {
        let m = hz_to_mel(1000.0);
        assert!((m - 1000.0).abs() < 50.0, "1000 Hz -> {m} mel");
    }

    #[test]
    fn mel_roundtrip() {
        for &f in &[200.0, 500.0, 1000.0, 5000.0] {
            let m = hz_to_mel(f);
            let f2 = mel_to_hz(m);
            assert!((f - f2).abs() < EPS, "Mel roundtrip: {f} -> {m} -> {f2}");
        }
    }

    #[test]
    fn erb_at_1000hz() {
        let e = erb_of_frequency(1000.0);
        assert!((e - 132.6).abs() < 1.0, "ERB at 1000 Hz = {e}");
    }

    #[test]
    fn erb_rate_roundtrip() {
        for &f in &[100.0, 1000.0, 8000.0] {
            let r = hz_to_erb_rate(f);
            let f2 = erb_rate_to_hz(r);
            assert!((f - f2).abs() < EPS, "ERB-rate roundtrip: {f} -> {r} -> {f2}");
        }
    }

    #[test]
    fn db_roundtrip() {
        for &a in &[0.001, 0.01, 0.1, 0.5, 1.0] {
            let db = amplitude_to_db_spl(a);
            let a2 = db_spl_to_amplitude(db);
            assert!((a - a2).abs() < 1e-10, "dB roundtrip: {a} -> {db} -> {a2}");
        }
    }

    #[test]
    fn amplitude_zero_gives_neg_inf() {
        assert!(amplitude_to_db_spl(0.0).is_infinite());
    }

    #[test]
    fn midi_a440() {
        assert!((midi_to_hz(69.0) - 440.0).abs() < EPS);
    }

    #[test]
    fn midi_roundtrip() {
        for note in [21.0, 60.0, 69.0, 108.0] {
            let f = midi_to_hz(note);
            let n2 = hz_to_midi(f);
            assert!((note - n2).abs() < 1e-10, "MIDI roundtrip: {note} -> {f} -> {n2}");
        }
    }

    #[test]
    fn semitone_ratio() {
        assert!((semitones_to_ratio(12.0) - 2.0).abs() < EPS);
    }

    #[test]
    fn cents_roundtrip() {
        let c = 700.0;
        let r = cents_to_ratio(c);
        let c2 = ratio_to_cents(r);
        assert!((c - c2).abs() < EPS);
    }

    #[test]
    fn bark_bandwidth_increases() {
        assert!(bark_bandwidth(8000.0) > bark_bandwidth(200.0));
    }

    #[test]
    fn a_weighting_at_1khz() {
        let w = a_weighting(1000.0);
        assert!((w - 1.0).abs() < 0.05, "A-weighting at 1 kHz = {w}");
    }

    #[test]
    fn ath_minimum_near_3khz() {
        assert!(absolute_threshold_of_hearing(3500.0) < absolute_threshold_of_hearing(1000.0));
    }

    #[test]
    fn wavelength_and_period_test() {
        let wl = wavelength(440.0, 343.0);
        assert!((wl - 0.78).abs() < 0.01);
        assert!((period(440.0) - 1.0 / 440.0).abs() < 1e-10);
    }
}
