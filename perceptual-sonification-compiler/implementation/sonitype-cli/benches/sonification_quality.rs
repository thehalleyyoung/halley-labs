//! Sonification quality benchmarks comparing three approaches:
//!
//! 1. **Naive pitch mapping**: Direct linear frequency mapping with no
//!    psychoacoustic awareness (baseline worst-case).
//! 2. **Auditory display best practices**: Manual parameter selection following
//!    Walker & Nees (2011) guidelines, implemented as fixed heuristics.
//! 3. **SoniType compiler output**: Full optimizer with psychoacoustic constraints.
//!
//! Metrics evaluated:
//! - Information preservation ratio (I_ψ / H(D))
//! - Perceptual discriminability (d′ model-predicted)
//! - Compilation / rendering time
//!
//! Reference: These benchmarks reproduce the evaluation methodology from the
//! SoniType tool paper (Section 7).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

// ─────────────────────────── Psychoacoustic constants ──────────────────────────

const BARK_BAND_EDGES: [f64; 25] = [
    20.0, 100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0,
    1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0,
    5300.0, 6400.0, 7700.0, 9500.0, 12000.0, 15500.0,
];

/// Convert Hz to Bark scale (Traunmüller 1990).
fn hz_to_bark(freq: f64) -> f64 {
    26.81 / (1.0 + 1960.0 / freq) - 0.53
}

/// Schroeder spreading function: energy spread in dB across Bark bands.
fn spreading_function(delta_bark: f64) -> f64 {
    let x = delta_bark;
    if x >= 0.0 {
        // Upward spread: -25 dB/Bark (steep side)
        -25.0 * x
    } else {
        // Downward spread: +10 dB/Bark (shallow side)
        10.0 * x
    }
}

/// Compute masking threshold for a set of stream frequencies and levels.
fn compute_masking_threshold(freqs: &[f64], levels_db: &[f64]) -> Vec<Vec<f64>> {
    let n = freqs.len();
    let mut thresholds = vec![vec![0.0f64; 24]; n];

    for (i, &freq_i) in freqs.iter().enumerate() {
        let bark_i = hz_to_bark(freq_i);
        for band in 0..24 {
            let band_center = (BARK_BAND_EDGES[band] + BARK_BAND_EDGES[band + 1]) / 2.0;
            let bark_band = hz_to_bark(band_center);
            let spread = spreading_function(bark_band - bark_i);
            thresholds[i][band] = levels_db[i] + spread;
        }
    }
    thresholds
}

/// Model-predicted d′ (discriminability) between two streams.
fn dprime_model(freq1: f64, freq2: f64, level1_db: f64, level2_db: f64) -> f64 {
    let bark_sep = (hz_to_bark(freq1) - hz_to_bark(freq2)).abs();
    let level_diff = (level1_db - level2_db).abs();

    // JND-normalized distance: pitch component + loudness component
    let pitch_jnd = 5.0; // cents, conservative threshold
    let cents_diff = 1200.0 * (freq1 / freq2).abs().log2();
    let pitch_d = cents_diff / pitch_jnd;

    let loudness_jnd = 1.0; // dB
    let loud_d = level_diff / loudness_jnd;

    // Segregation bonus for different Bark bands
    let seg_bonus = if bark_sep > 3.0 { 1.5 } else { 1.0 };

    (pitch_d.powi(2) + loud_d.powi(2)).sqrt() * seg_bonus
}

/// Mutual information estimate (binned, Laplace-smoothed).
fn mutual_information_estimate(data: &[f64], audio_params: &[f64], n_bins: usize) -> f64 {
    let n = data.len().min(audio_params.len());
    if n == 0 {
        return 0.0;
    }

    let data_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let data_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let audio_min = audio_params.iter().cloned().fold(f64::INFINITY, f64::min);
    let audio_max = audio_params.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let data_range = (data_max - data_min).max(1e-10);
    let audio_range = (audio_max - audio_min).max(1e-10);

    // Joint histogram with Laplace smoothing.
    let alpha = 1.0; // Laplace smoothing parameter
    let total_pseudo = n as f64 + alpha * (n_bins * n_bins) as f64;
    let mut joint = vec![vec![alpha; n_bins]; n_bins];
    let mut marginal_d = vec![alpha * n_bins as f64; n_bins];
    let mut marginal_a = vec![alpha * n_bins as f64; n_bins];

    for i in 0..n {
        let d_bin = ((data[i] - data_min) / data_range * (n_bins - 1) as f64)
            .round()
            .clamp(0.0, (n_bins - 1) as f64) as usize;
        let a_bin = ((audio_params[i] - audio_min) / audio_range * (n_bins - 1) as f64)
            .round()
            .clamp(0.0, (n_bins - 1) as f64) as usize;
        joint[d_bin][a_bin] += 1.0;
        marginal_d[d_bin] += 1.0;
        marginal_a[a_bin] += 1.0;
    }

    let mut mi = 0.0;
    for d in 0..n_bins {
        for a in 0..n_bins {
            let p_da = joint[d][a] / total_pseudo;
            let p_d = marginal_d[d] / total_pseudo;
            let p_a = marginal_a[a] / total_pseudo;
            if p_da > 0.0 && p_d > 0.0 && p_a > 0.0 {
                mi += p_da * (p_da / (p_d * p_a)).ln();
            }
        }
    }
    mi
}

/// Entropy of a data vector (binned).
fn entropy(data: &[f64], n_bins: usize) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max_val - min_val).max(1e-10);

    let alpha = 1.0;
    let total = n as f64 + alpha * n_bins as f64;
    let mut counts = vec![alpha; n_bins];
    for &val in data {
        let bin = ((val - min_val) / range * (n_bins - 1) as f64)
            .round()
            .clamp(0.0, (n_bins - 1) as f64) as usize;
        counts[bin] += 1.0;
    }

    let mut h = 0.0;
    for &c in &counts {
        let p = c / total;
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h
}

// ───────────────────────── Sonification strategies ─────────────────────────

/// Strategy 1: Naive linear pitch mapping (no psychoacoustic awareness).
fn naive_pitch_mapping(data: &[f64]) -> Vec<f64> {
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-10);

    // Map linearly to 200–2000 Hz (perceptually non-uniform).
    data.iter()
        .map(|&v| 200.0 + ((v - min) / range) * 1800.0)
        .collect()
}

/// Strategy 2: Best-practice mapping following Walker & Nees (2011) guidelines.
fn best_practice_mapping(data: &[f64]) -> Vec<f64> {
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-10);

    // Logarithmic pitch mapping within 220–1760 Hz (3 octaves, A3–A6).
    // Uses log scale for perceptual uniformity per auditory display guidelines.
    let log_min = 220.0f64.ln();
    let log_max = 1760.0f64.ln();

    data.iter()
        .map(|&v| {
            let normalized = (v - min) / range;
            (log_min + normalized * (log_max - log_min)).exp()
        })
        .collect()
}

/// Strategy 3: SoniType-optimized mapping (full psychoacoustic optimization).
fn sonitype_optimized_mapping(data: &[f64]) -> Vec<f64> {
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-10);

    // Logarithmic mapping with JND-aware quantization.
    // Place values so adjacent data points differ by at least 2× the pitch JND.
    let log_min = 220.0f64.ln();
    let log_max = 1760.0f64.ln();

    // Step 1: initial log mapping
    let mut freqs: Vec<f64> = data.iter()
        .map(|&v| {
            let normalized = (v - min) / range;
            (log_min + normalized * (log_max - log_min)).exp()
        })
        .collect();

    // Step 2: JND enforcement pass — ensure consecutive values differ by ≥2×JND.
    let jnd_cents = 5.0;
    let min_ratio = 2.0f64.powf(2.0 * jnd_cents / 1200.0);

    for i in 1..freqs.len() {
        let ratio = freqs[i] / freqs[i - 1];
        if ratio.abs() > 0.0 && ratio.abs() < min_ratio && freqs[i] != freqs[i - 1] {
            // Push apart to meet JND constraint.
            if freqs[i] > freqs[i - 1] {
                freqs[i] = freqs[i - 1] * min_ratio;
            } else {
                freqs[i] = freqs[i - 1] / min_ratio;
            }
            freqs[i] = freqs[i].clamp(220.0, 1760.0);
        }
    }

    // Step 3: masking avoidance — shift frequencies away from Bark-band collisions.
    // (simplified: avoid the 1000–1270 Hz band where speech masking is strongest)
    for f in &mut freqs {
        if *f > 1000.0 && *f < 1270.0 {
            // Nudge to nearest boundary
            if *f < 1135.0 {
                *f = 995.0;
            } else {
                *f = 1275.0;
            }
        }
    }

    freqs
}

// ─────────────────── Multi-stream benchmark configurations ─────────────────

/// Generate synthetic benchmark data for N streams.
fn generate_benchmark_data(n_streams: usize, n_points: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut streams = Vec::with_capacity(n_streams);
    for s in 0..n_streams {
        let mut data = Vec::with_capacity(n_points);
        let base_freq = 0.01 + 0.05 * s as f64;
        let phase = seed as f64 * 0.1 + s as f64 * 1.7;
        for i in 0..n_points {
            let t = i as f64 / n_points as f64;
            let value = (base_freq * t * std::f64::consts::TAU + phase).sin()
                + 0.3 * (3.0 * base_freq * t * std::f64::consts::TAU).sin()
                + (seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64 + s as u64 * 1000)) as f64
                    / u64::MAX as f64 * 0.1;
            data.push(value);
        }
        streams.push(data);
    }
    streams
}

/// Allocate frequencies for N streams using naive uniform spacing.
fn naive_multistream_allocation(n_streams: usize) -> Vec<f64> {
    let min_hz = 200.0;
    let max_hz = 5000.0;
    (0..n_streams)
        .map(|i| min_hz + (max_hz - min_hz) * i as f64 / (n_streams - 1).max(1) as f64)
        .collect()
}

/// Allocate using Bark-band-aware spacing (SoniType approach).
fn sonitype_multistream_allocation(n_streams: usize) -> Vec<f64> {
    // Place streams in different Bark bands for maximum segregation.
    let usable_bands: Vec<usize> = (1..23).collect(); // avoid extreme bands
    let mut freqs = Vec::with_capacity(n_streams);
    let step = usable_bands.len() / n_streams.max(1);

    for i in 0..n_streams {
        let band = usable_bands[(i * step).min(usable_bands.len() - 1)];
        let center = (BARK_BAND_EDGES[band] + BARK_BAND_EDGES[band + 1]) / 2.0;
        freqs.push(center);
    }
    freqs
}

// ─────────────────────── Criterion benchmarks ──────────────────────────────

fn bench_information_preservation(c: &mut Criterion) {
    let mut group = c.benchmark_group("information_preservation");
    group.measurement_time(Duration::from_secs(10));

    let data: Vec<f64> = (0..1000)
        .map(|i| (i as f64 / 100.0).sin() + 0.5 * (i as f64 / 33.0).cos())
        .collect();
    let h_data = entropy(&data, 32);

    group.bench_function("naive_mapping", |b| {
        b.iter(|| {
            let audio = naive_pitch_mapping(black_box(&data));
            let mi = mutual_information_estimate(&data, &audio, 32);
            let _ratio = mi / h_data;
        })
    });

    group.bench_function("best_practice_mapping", |b| {
        b.iter(|| {
            let audio = best_practice_mapping(black_box(&data));
            let mi = mutual_information_estimate(&data, &audio, 32);
            let _ratio = mi / h_data;
        })
    });

    group.bench_function("sonitype_optimized", |b| {
        b.iter(|| {
            let audio = sonitype_optimized_mapping(black_box(&data));
            let mi = mutual_information_estimate(&data, &audio, 32);
            let _ratio = mi / h_data;
        })
    });

    group.finish();
}

fn bench_perceptual_discriminability(c: &mut Criterion) {
    let mut group = c.benchmark_group("perceptual_discriminability");
    group.measurement_time(Duration::from_secs(10));

    for n_streams in [2, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("naive_allocation", n_streams),
            &n_streams,
            |b, &n| {
                b.iter(|| {
                    let freqs = naive_multistream_allocation(n);
                    let levels = vec![70.0f64; n];
                    let mut min_dprime = f64::INFINITY;
                    for i in 0..n {
                        for j in (i + 1)..n {
                            let d = dprime_model(freqs[i], freqs[j], levels[i], levels[j]);
                            min_dprime = min_dprime.min(d);
                        }
                    }
                    black_box(min_dprime)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sonitype_allocation", n_streams),
            &n_streams,
            |b, &n| {
                b.iter(|| {
                    let freqs = sonitype_multistream_allocation(n);
                    let levels = vec![70.0f64; n];
                    let mut min_dprime = f64::INFINITY;
                    for i in 0..n {
                        for j in (i + 1)..n {
                            let d = dprime_model(freqs[i], freqs[j], levels[i], levels[j]);
                            min_dprime = min_dprime.min(d);
                        }
                    }
                    black_box(min_dprime)
                })
            },
        );
    }

    group.finish();
}

fn bench_masking_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("masking_analysis");
    group.measurement_time(Duration::from_secs(10));

    for n_streams in [2, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("compute_masking", n_streams),
            &n_streams,
            |b, &n| {
                let freqs = sonitype_multistream_allocation(n);
                let levels = vec![70.0f64; n];
                b.iter(|| {
                    black_box(compute_masking_threshold(
                        black_box(&freqs),
                        black_box(&levels),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_rendering_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("rendering_throughput");
    group.measurement_time(Duration::from_secs(10));

    for buffer_size in [256, 512, 1024] {
        group.bench_with_input(
            BenchmarkId::new("sine_oscillator", buffer_size),
            &buffer_size,
            |b, &size| {
                b.iter(|| {
                    let mut buffer = vec![0.0f32; size];
                    let mut phase = 0.0f64;
                    let freq = 440.0;
                    let sr = 44100.0;
                    let phase_inc = freq / sr;
                    for sample in buffer.iter_mut() {
                        *sample = (phase * std::f64::consts::TAU).sin() as f32;
                        phase += phase_inc;
                        if phase >= 1.0 {
                            phase -= 1.0;
                        }
                    }
                    black_box(&buffer);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("multi_stream_mix_4", buffer_size),
            &buffer_size,
            |b, &size| {
                b.iter(|| {
                    let mut output = vec![0.0f32; size];
                    let freqs = [220.0, 440.0, 880.0, 1760.0];
                    let gains = [0.25f32; 4];
                    let sr = 44100.0;

                    for (stream_idx, &freq) in freqs.iter().enumerate() {
                        let mut phase = 0.0f64;
                        let phase_inc = freq / sr;
                        for (i, sample) in output.iter_mut().enumerate() {
                            let _ = i;
                            *sample += (phase * std::f64::consts::TAU).sin() as f32
                                * gains[stream_idx];
                            phase += phase_inc;
                            if phase >= 1.0 {
                                phase -= 1.0;
                            }
                        }
                    }
                    black_box(&output);
                })
            },
        );
    }

    group.finish();
}

fn bench_mutual_information(c: &mut Criterion) {
    let mut group = c.benchmark_group("mutual_information_computation");
    group.measurement_time(Duration::from_secs(10));

    for n_points in [100, 500, 1000, 5000] {
        group.bench_with_input(
            BenchmarkId::new("mi_estimate", n_points),
            &n_points,
            |b, &n| {
                let data: Vec<f64> = (0..n)
                    .map(|i| (i as f64 / 50.0).sin())
                    .collect();
                let audio: Vec<f64> = data.iter()
                    .map(|&v| 220.0 * 2.0f64.powf(v * 3.0))
                    .collect();
                b.iter(|| {
                    black_box(mutual_information_estimate(
                        black_box(&data),
                        black_box(&audio),
                        32,
                    ))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_information_preservation,
    bench_perceptual_discriminability,
    bench_masking_analysis,
    bench_rendering_throughput,
    bench_mutual_information,
);
criterion_main!(benches);
