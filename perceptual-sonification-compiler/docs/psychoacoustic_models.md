# Psychoacoustic Models in SoniType

## Overview

SoniType integrates five psychoacoustic models as compile-time constraint
checkers and optimization objectives. Each model is implemented in the
`sonitype-psychoacoustic` crate with parameters grounded in published data.

## 1. Critical-Band Masking (masking.rs)

### Model

Simultaneous masking is modeled using the Schroeder spreading function:

```
SF(z) = 15.81 + 7.5(z + 0.474) - 17.5 * sqrt(1 + (z + 0.474)²)
```

where `z` is the frequency distance in Bark.

### Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Bark bands | 24 | Zwicker (1961) |
| Upward spread slope | ~25 dB/Bark | Schroeder et al. (1979) |
| Downward spread slope | ~-10 dB/Bark | Schroeder et al. (1979) |
| Minimum masking clearance | 6 dB (default) | Conservative estimate |

### Validation

Model predictions are validated against Moore (2012) published masking data.
Cross-model evaluation uses Glasberg-Moore as a second reference.

## 2. Just-Noticeable Differences (jnd.rs)

### Pitch JND

Frequency discrimination follows Weber's Law with corrections:

```
Δf/f ≈ 0.003 for f in [500, 2000] Hz  (pure tones)
Δf/f ≈ 0.01  for f < 200 Hz or f > 5000 Hz
```

SoniType converts to cents: default JND = 5 cents (conservative).

### Loudness JND

Intensity discrimination:
```
ΔL ≈ 1.0 dB for broadband noise (Weber's Law)
ΔL ≈ 0.5-1.5 dB for pure tones (near-miss to Weber's Law)
```

### Timbre JND

Based on Grey (1977) MDS dimensions:
- Spectral centroid: JND ≈ 5% relative change
- Attack time: JND ≈ 10-20 ms
- Spectral flux: JND ≈ 10% relative change

### Sources

- Wier, C.C., Jesteadt, W., & Green, D.M. (1977). JASA, 61(1), 178-184.
- Jesteadt, W., Wier, C.C., & Green, D.M. (1977). JASA, 61(1), 169-177.
- Moore, B.C.J. (2012). An Introduction to the Psychology of Hearing.

## 3. Stream Segregation (segregation.rs)

### Model

Based on Bregman's (1990) auditory scene analysis, streams segregate if at
least one criterion is strongly met:

| Criterion | Threshold | Source |
|-----------|-----------|--------|
| Frequency separation | > 3 semitones | Van Noorden (1975) |
| Onset asynchrony | > 30 ms | Bregman (1990) |
| Spatial separation | > 10° azimuth | Blauert (1997) |
| Timbre dissimilarity | > 0.3 (normalized) | McAdams et al. (1995) |
| Harmonicity difference | > 2% inharmonicity | De Cheveigné (2005) |

For k streams, O(k²) pairwise checks are performed.

### Segregation Strength

- **Strong**: 2+ criteria met
- **Moderate**: 1 criterion met strongly
- **Weak**: criteria partially met (warning issued)
- **None**: no criterion met (error: streams will fuse)

## 4. Cognitive Load (cognitive_load.rs)

### Model

Based on Cowan's (2001) working memory capacity limit:

```
Max simultaneous tracked streams ≤ 4 ± 1
```

SoniType defaults to 4 as the hard ceiling. The accessibility module can
reduce this for users with cognitive processing differences.

### Factors

| Factor | Effect on capacity |
|--------|-------------------|
| Stream similarity | Reduces effective capacity |
| Task complexity | Reduces by 1 for complex tasks |
| Hearing loss | May reduce by 1-2 |
| Training/expertise | May increase by 1 |

## 5. Pitch Perception (pitch.rs)

### Model

Combines three pitch estimation methods:

1. **Spectral method**: Peak-picking in the frequency domain
2. **Temporal method**: Autocorrelation of the waveform
3. **Virtual pitch**: Subharmonic summation (Terhardt, 1979)

The models are used to predict perceived pitch from complex tones and verify
that pitch mappings produce the intended perceptual effect.

### Key Parameters

| Parameter | Default | Range |
|-----------|---------|-------|
| min_freq | 50 Hz | 20-200 Hz |
| max_freq | 5000 Hz | 2000-10000 Hz |
| max_subharmonic_order | 5 | 3-8 |
| autocorrelation_threshold | 0.5 | 0.3-0.8 |
| harmonic_tolerance_cents | 50 | 20-100 |

## Model Interaction

SoniType evaluates all five models jointly during type checking:

1. Masking check: Are all stream pairs spectrally clear?
2. JND check: Do parameter differences exceed thresholds?
3. Segregation check: Will listeners hear separate streams?
4. Cognitive load: Can a listener track all streams?
5. Pitch perception: Will mapped pitches be perceived correctly?

Failure in any model produces a compile-time error with diagnostic
information about which streams/parameters violate the constraint.
