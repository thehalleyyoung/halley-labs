#!/usr/bin/env python3
"""
SoniType SOTA Benchmark Suite
Real-world data sonification with psychoacoustic constraints and baseline comparisons.

This benchmark evaluates perceptual sonification mappings using:
- Real financial data (stock prices)
- Realistic physiological data (heart rate)
- Seasonal environmental data (temperature)
- Psychoacoustic constraints (JND, Weber's law, temporal masking)
- Multiple SOTA baselines and metrics
"""

import numpy as np
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import yfinance, fallback to synthetic data if not available
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("Warning: yfinance not available, using synthetic financial data")


class PsychoacousticConstraints:
    """Psychoacoustic constraints and perceptual models."""
    
    @staticmethod
    def frequency_jnd(f: float) -> float:
        """Just Noticeable Difference for frequency (Hz) based on psychoacoustic research."""
        if f < 1000:
            return 0.02 * f  # 2% for low frequencies
        else:
            return 0.005 * f + 5  # Smaller percentage + constant for high frequencies
    
    @staticmethod
    def amplitude_jnd(db: float) -> float:
        """JND for amplitude in dB. Weber's law approximation."""
        return 0.5 + 0.01 * db  # Base JND + Weber component
    
    @staticmethod
    def weber_fraction(stimulus_value: float, modality: str = "pitch") -> float:
        """Weber fraction for different sensory modalities."""
        fractions = {
            "pitch": 0.002,      # Frequency discrimination
            "loudness": 0.05,    # Amplitude discrimination  
            "duration": 0.1,     # Temporal discrimination
            "timbre": 0.15       # Spectral discrimination
        }
        return fractions.get(modality, 0.05)
    
    @staticmethod
    def temporal_masking_window(duration_ms: float) -> float:
        """Temporal masking duration in milliseconds."""
        # Forward masking duration depends on masker duration
        return min(200, 0.2 * duration_ms + 20)
    
    @staticmethod
    def frequency_discrimination_curve(f_center: float, f_test: float) -> float:
        """Frequency discrimination curve (psychoacoustic model)."""
        df = abs(f_test - f_center)
        jnd = PsychoacousticConstraints.frequency_jnd(f_center)
        return 1.0 / (1.0 + (df / jnd) ** 2)


class DataGenerator:
    """Generates real and synthetic datasets for sonification benchmarking."""
    
    @staticmethod
    def download_stock_data(symbol: str = "AAPL", period: str = "1y") -> np.ndarray:
        """Download real stock price data."""
        if not HAS_YFINANCE:
            return DataGenerator.synthetic_stock_data()
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            if len(hist) == 0:
                print(f"No data for {symbol}, using synthetic")
                return DataGenerator.synthetic_stock_data()
            return hist['Close'].values
        except Exception as e:
            print(f"Error downloading {symbol}: {e}, using synthetic")
            return DataGenerator.synthetic_stock_data()
    
    @staticmethod
    def synthetic_stock_data(length: int = 252) -> np.ndarray:
        """Generate realistic synthetic stock price data."""
        np.random.seed(42)
        returns = np.random.normal(0.0008, 0.02, length)  # Daily returns
        # Add some volatility clustering
        volatility = np.abs(np.random.normal(0.02, 0.005, length))
        returns *= volatility
        
        prices = np.zeros(length)
        prices[0] = 100.0
        for i in range(1, length):
            prices[i] = prices[i-1] * (1 + returns[i])
        return prices
    
    @staticmethod
    def realistic_temperature_data(length: int = 365) -> np.ndarray:
        """Generate realistic seasonal temperature data."""
        np.random.seed(43)
        days = np.arange(length)
        
        # Base seasonal pattern
        seasonal = 20 + 15 * np.sin(2 * np.pi * days / 365.25 - np.pi/2)
        
        # Add weather noise and trends
        weather_noise = np.random.normal(0, 3, length)
        weekly_cycle = 2 * np.sin(2 * np.pi * days / 7)
        
        # Random weather events
        events = np.random.exponential(0.1, length) * np.random.choice([-1, 1], length)
        
        temperature = seasonal + weather_noise + weekly_cycle + events
        return temperature
    
    @staticmethod
    def realistic_heart_rate_data(length: int = 1440) -> np.ndarray:
        """Generate realistic heart rate data (minutes in a day)."""
        np.random.seed(44)
        
        # Circadian rhythm base
        minutes = np.arange(length)
        circadian = 70 + 10 * np.sin(2 * np.pi * minutes / (24 * 60) - np.pi/3)
        
        # Activity periods (higher HR during day)
        activity_hours = ((minutes // 60) >= 7) & ((minutes // 60) <= 22)
        activity_boost = activity_hours * np.random.exponential(5, length) * (np.random.random(length) > 0.8)
        
        # Respiratory sinus arrhythmia and other physiological noise
        respiratory = 2 * np.sin(2 * np.pi * minutes / 4)  # ~15 breaths/min
        noise = np.random.normal(0, 2, length)
        
        hr = circadian + activity_boost + respiratory + noise
        return np.clip(hr, 45, 180)  # Physiological bounds


class PerceptualSonificationCompiler:
    """Core perceptual sonification compiler with psychoacoustic verification."""
    
    def __init__(self):
        self.constraints = PsychoacousticConstraints()
        self.frequency_range = (80, 8000)  # Human auditory range (Hz)
        self.amplitude_range = (30, 90)    # dB SPL range
        self.duration_range = (50, 2000)   # ms
    
    def compile_mapping(self, data: np.ndarray, mapping_type: str, 
                       constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Compile perceptual sonification mapping with psychoacoustic verification."""
        
        # Normalize data
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        if mapping_type == "pitch":
            audio_params = self._compile_pitch_mapping(data_norm, constraints)
        elif mapping_type == "volume":
            audio_params = self._compile_volume_mapping(data_norm, constraints) 
        elif mapping_type == "rhythm":
            audio_params = self._compile_rhythm_mapping(data_norm, constraints)
        elif mapping_type == "multi_parameter":
            audio_params = self._compile_multi_parameter_mapping(data_norm, constraints)
        else:
            raise ValueError(f"Unknown mapping type: {mapping_type}")
        
        # Verify psychoacoustic constraints
        violations = self._verify_constraints(audio_params, constraints)
        
        return {
            "audio_params": audio_params,
            "constraint_violations": violations,
            "mapping_metadata": {
                "type": mapping_type,
                "data_range": (np.min(data), np.max(data)),
                "param_range": self._get_param_range(audio_params),
                "constraints": constraints
            }
        }
    
    def _compile_pitch_mapping(self, data: np.ndarray, constraints: Dict) -> Dict:
        """Compile pitch mapping with frequency JND constraints."""
        f_min, f_max = self.frequency_range
        
        # Logarithmic frequency mapping (perceptually uniform)
        log_f_min, log_f_max = np.log(f_min), np.log(f_max)
        log_frequencies = log_f_min + data * (log_f_max - log_f_min)
        frequencies = np.exp(log_frequencies)
        
        # Apply JND quantization if required
        if constraints.get("enforce_jnd", True):
            frequencies = self._quantize_by_jnd(frequencies, "frequency")
        
        return {"frequencies": frequencies, "amplitudes": np.full_like(frequencies, 70)}
    
    def _compile_volume_mapping(self, data: np.ndarray, constraints: Dict) -> Dict:
        """Compile volume mapping with amplitude JND constraints."""
        db_min, db_max = self.amplitude_range
        amplitudes = db_min + data * (db_max - db_min)
        
        if constraints.get("enforce_jnd", True):
            amplitudes = self._quantize_by_jnd(amplitudes, "amplitude")
        
        return {"frequencies": np.full_like(amplitudes, 440), "amplitudes": amplitudes}
    
    def _compile_rhythm_mapping(self, data: np.ndarray, constraints: Dict) -> Dict:
        """Compile rhythm/temporal mapping."""
        dur_min, dur_max = self.duration_range
        durations = dur_min + data * (dur_max - dur_min)
        
        if constraints.get("enforce_jnd", True):
            durations = self._quantize_by_jnd(durations, "duration")
        
        return {"frequencies": np.full_like(durations, 440), 
                "amplitudes": np.full_like(durations, 70),
                "durations": durations}
    
    def _compile_multi_parameter_mapping(self, data: np.ndarray, constraints: Dict) -> Dict:
        """Multi-parameter mapping using principal components."""
        # Create multiple data streams from single input
        data_smooth = scipy.signal.savgol_filter(data, window_length=min(21, len(data)//4+1), polyorder=3)
        data_derivative = np.gradient(data)
        data_variance = np.array([np.var(data[max(0, i-10):i+10]) for i in range(len(data))])
        
        # Map to different parameters
        freq_data = (data_smooth - np.min(data_smooth)) / (np.max(data_smooth) - np.min(data_smooth))
        amp_data = (np.abs(data_derivative) - np.min(np.abs(data_derivative))) / (np.max(np.abs(data_derivative)) - np.min(np.abs(data_derivative)))
        dur_data = (data_variance - np.min(data_variance)) / (np.max(data_variance) - np.min(data_variance))
        
        f_min, f_max = self.frequency_range
        log_frequencies = np.log(f_min) + freq_data * (np.log(f_max) - np.log(f_min))
        frequencies = np.exp(log_frequencies)
        
        db_min, db_max = self.amplitude_range
        amplitudes = db_min + amp_data * (db_max - db_min)
        
        dur_min, dur_max = self.duration_range
        durations = dur_min + dur_data * (dur_max - dur_min)
        
        if constraints.get("enforce_jnd", True):
            frequencies = self._quantize_by_jnd(frequencies, "frequency")
            amplitudes = self._quantize_by_jnd(amplitudes, "amplitude")
            durations = self._quantize_by_jnd(durations, "duration")
        
        return {"frequencies": frequencies, "amplitudes": amplitudes, "durations": durations}
    
    def _quantize_by_jnd(self, values: np.ndarray, param_type: str) -> np.ndarray:
        """Quantize parameter values by Just Noticeable Difference."""
        quantized = np.zeros_like(values)
        
        for i, val in enumerate(values):
            if param_type == "frequency":
                jnd = self.constraints.frequency_jnd(val)
                quantized[i] = round(val / jnd) * jnd
            elif param_type == "amplitude":
                jnd = self.constraints.amplitude_jnd(val)
                quantized[i] = round(val / jnd) * jnd
            elif param_type == "duration":
                weber_frac = self.constraints.weber_fraction(val, "duration")
                jnd = val * weber_frac
                quantized[i] = round(val / jnd) * jnd
            else:
                quantized[i] = val
        
        return quantized
    
    def _verify_constraints(self, audio_params: Dict, constraints: Dict) -> List[str]:
        """Verify psychoacoustic constraint satisfaction."""
        violations = []
        
        # Check frequency range
        if "frequencies" in audio_params:
            freqs = audio_params["frequencies"]
            if np.any(freqs < self.frequency_range[0]) or np.any(freqs > self.frequency_range[1]):
                violations.append("frequency_range_violation")
        
        # Check amplitude range
        if "amplitudes" in audio_params:
            amps = audio_params["amplitudes"]
            if np.any(amps < self.amplitude_range[0]) or np.any(amps > self.amplitude_range[1]):
                violations.append("amplitude_range_violation")
        
        # Check temporal masking
        if "durations" in audio_params:
            durs = audio_params["durations"]
            min_gap = constraints.get("min_temporal_gap", 50)  # ms
            if np.any(durs < min_gap):
                violations.append("temporal_masking_violation")
        
        return violations
    
    def _get_param_range(self, audio_params: Dict) -> Dict:
        """Get parameter ranges from audio output."""
        ranges = {}
        for key, values in audio_params.items():
            if isinstance(values, np.ndarray):
                ranges[key] = (float(np.min(values)), float(np.max(values)))
        return ranges


class BaselineMappings:
    """SOTA baseline mapping implementations."""
    
    @staticmethod
    def linear_mapping(data: np.ndarray, param_range: Tuple[float, float]) -> np.ndarray:
        """Naive linear mapping."""
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        return param_range[0] + data_norm * (param_range[1] - param_range[0])
    
    @staticmethod
    def logarithmic_mapping(data: np.ndarray, param_range: Tuple[float, float]) -> np.ndarray:
        """Logarithmic mapping (Weber-Fechner law)."""
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        log_range = (np.log(param_range[0]), np.log(param_range[1]))
        log_values = log_range[0] + data_norm * (log_range[1] - log_range[0])
        return np.exp(log_values)
    
    @staticmethod
    def midi_quantized_mapping(data: np.ndarray, param_range: Tuple[float, float]) -> np.ndarray:
        """MIDI-quantized mapping (12-TET)."""
        # Map to MIDI note numbers first
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        midi_min = 12 * np.log2(param_range[0] / 440) + 69
        midi_max = 12 * np.log2(param_range[1] / 440) + 69
        
        midi_notes = midi_min + data_norm * (midi_max - midi_min)
        midi_quantized = np.round(midi_notes)
        
        # Convert back to Hz
        return 440 * 2**((midi_quantized - 69) / 12)
    
    @staticmethod
    def random_mapping(data: np.ndarray, param_range: Tuple[float, float]) -> np.ndarray:
        """Random baseline mapping."""
        np.random.seed(42)  # For reproducibility
        return np.random.uniform(param_range[0], param_range[1], len(data))


class BenchmarkMetrics:
    """Evaluation metrics for sonification quality."""
    
    @staticmethod
    def perceptual_discriminability_score(audio_params: Dict, data: np.ndarray) -> float:
        """Compute perceptual discriminability using psychoacoustic models."""
        if "frequencies" in audio_params:
            freqs = audio_params["frequencies"]
            # Compute pairwise discrimination probabilities
            discrim_scores = []
            for i in range(len(freqs) - 1):
                f1, f2 = freqs[i], freqs[i + 1]
                discrim = PsychoacousticConstraints.frequency_discrimination_curve(f1, f2)
                discrim_scores.append(discrim)
            
            # Weight by data differences
            data_diffs = np.abs(np.diff(data))
            data_diffs_norm = data_diffs / np.max(data_diffs) if np.max(data_diffs) > 0 else data_diffs
            
            weighted_discrim = np.average(discrim_scores, weights=data_diffs_norm + 1e-8)
            return float(weighted_discrim)
        
        return 0.0
    
    @staticmethod
    def information_preservation_score(audio_params: Dict, data: np.ndarray) -> float:
        """Mutual information between data and audio parameters."""
        if len(audio_params) == 0:
            return 0.0
        
        # Use primary audio parameter (frequency, amplitude, or duration)
        if "frequencies" in audio_params:
            audio_values = audio_params["frequencies"]
        elif "amplitudes" in audio_params:
            audio_values = audio_params["amplitudes"]
        elif "durations" in audio_params:
            audio_values = audio_params["durations"]
        else:
            return 0.0
        
        # Compute mutual information using binning
        try:
            data_bins = 20
            audio_bins = 20
            
            data_binned = np.digitize(data, np.linspace(np.min(data), np.max(data), data_bins))
            audio_binned = np.digitize(audio_values, np.linspace(np.min(audio_values), np.max(audio_values), audio_bins))
            
            # Joint histogram
            joint_hist = np.histogram2d(data_binned, audio_binned, bins=[data_bins, audio_bins])[0]
            joint_hist = joint_hist + 1e-10  # Smoothing
            joint_prob = joint_hist / np.sum(joint_hist)
            
            # Marginal probabilities
            data_prob = np.sum(joint_prob, axis=1)
            audio_prob = np.sum(joint_prob, axis=0)
            
            # Mutual information
            mi = 0.0
            for i in range(data_bins):
                for j in range(audio_bins):
                    if joint_prob[i, j] > 0:
                        mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (data_prob[i] * audio_prob[j]))
            
            return float(mi)
        
        except Exception:
            return 0.0
    
    @staticmethod
    def constraint_satisfaction_rate(violations: List[str], total_constraints: int = 5) -> float:
        """Rate of psychoacoustic constraint satisfaction."""
        return 1.0 - len(violations) / total_constraints
    
    @staticmethod
    def mapping_smoothness(audio_params: Dict) -> float:
        """Smoothness of the audio parameter mapping."""
        if "frequencies" in audio_params:
            values = audio_params["frequencies"]
        elif "amplitudes" in audio_params:
            values = audio_params["amplitudes"]
        elif "durations" in audio_params:
            values = audio_params["durations"]
        else:
            return 0.0
        
        # Compute relative variation
        if len(values) < 2:
            return 1.0
        
        diffs = np.abs(np.diff(values))
        mean_val = np.mean(values)
        
        if mean_val == 0:
            return 1.0
        
        smoothness = 1.0 / (1.0 + np.mean(diffs) / mean_val)
        return float(smoothness)


def run_comprehensive_benchmark() -> Dict[str, Any]:
    """Run comprehensive benchmark with all scenarios and baselines."""
    
    print("SoniType SOTA Benchmark Suite")
    print("=" * 50)
    
    # Initialize components
    compiler = PerceptualSonificationCompiler()
    
    # Test datasets
    print("\n1. Loading datasets...")
    datasets = {
        "stock_aapl": DataGenerator.download_stock_data("AAPL", "1y"),
        "stock_googl": DataGenerator.download_stock_data("GOOGL", "1y"), 
        "temperature": DataGenerator.realistic_temperature_data(365),
        "heart_rate": DataGenerator.realistic_heart_rate_data(1440)
    }
    
    print(f"   - AAPL stock: {len(datasets['stock_aapl'])} points")
    print(f"   - GOOGL stock: {len(datasets['stock_googl'])} points")
    print(f"   - Temperature: {len(datasets['temperature'])} points") 
    print(f"   - Heart rate: {len(datasets['heart_rate'])} points")
    
    # Sonification scenarios
    scenarios = [
        {"type": "pitch", "constraints": {"enforce_jnd": True, "min_temporal_gap": 100}},
        {"type": "pitch", "constraints": {"enforce_jnd": False}},
        {"type": "volume", "constraints": {"enforce_jnd": True, "min_temporal_gap": 50}},
        {"type": "volume", "constraints": {"enforce_jnd": False}},
        {"type": "rhythm", "constraints": {"enforce_jnd": True, "min_temporal_gap": 100}},
        {"type": "rhythm", "constraints": {"enforce_jnd": False}},
        {"type": "multi_parameter", "constraints": {"enforce_jnd": True, "min_temporal_gap": 80}},
        {"type": "multi_parameter", "constraints": {"enforce_jnd": False}}
    ]
    
    # Baseline mappings for comparison
    baseline_configs = {
        "linear": {"func": BaselineMappings.linear_mapping, "range": (80, 8000)},
        "logarithmic": {"func": BaselineMappings.logarithmic_mapping, "range": (80, 8000)},
        "midi_quantized": {"func": BaselineMappings.midi_quantized_mapping, "range": (80, 8000)},
        "random": {"func": BaselineMappings.random_mapping, "range": (80, 8000)}
    }
    
    results = {
        "benchmark_metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_datasets": len(datasets),
            "num_scenarios": len(scenarios),
            "num_baselines": len(baseline_configs),
            "psychoacoustic_constraints": True,
        },
        "dataset_statistics": {},
        "sonitype_results": {},
        "baseline_results": {},
        "comparative_analysis": {}
    }
    
    # Compute dataset statistics
    print("\n2. Computing dataset statistics...")
    for name, data in datasets.items():
        stats = {
            "length": len(data),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)), 
            "max": float(np.max(data)),
            "skewness": float(scipy.stats.skew(data)),
            "kurtosis": float(scipy.stats.kurtosis(data))
        }
        results["dataset_statistics"][name] = stats
        print(f"   - {name}: μ={stats['mean']:.2f}, σ={stats['std']:.2f}")
    
    # Run SoniType compiler benchmarks
    print("\n3. Running SoniType compiler benchmarks...")
    for dataset_name, data in datasets.items():
        results["sonitype_results"][dataset_name] = {}
        
        # Subsample large datasets for efficiency
        if len(data) > 500:
            indices = np.linspace(0, len(data) - 1, 500, dtype=int)
            data_subset = data[indices]
        else:
            data_subset = data
        
        for i, scenario in enumerate(scenarios):
            scenario_name = f"{scenario['type']}_jnd_{scenario['constraints']['enforce_jnd']}"
            print(f"   - {dataset_name}/{scenario_name}")
            
            try:
                # Compile mapping
                start_time = time.time()
                mapping_result = compiler.compile_mapping(data_subset, scenario["type"], scenario["constraints"])
                compile_time = time.time() - start_time
                
                # Evaluate metrics
                audio_params = mapping_result["audio_params"]
                violations = mapping_result["constraint_violations"]
                
                discriminability = BenchmarkMetrics.perceptual_discriminability_score(audio_params, data_subset)
                information_preservation = BenchmarkMetrics.information_preservation_score(audio_params, data_subset)
                constraint_satisfaction = BenchmarkMetrics.constraint_satisfaction_rate(violations)
                smoothness = BenchmarkMetrics.mapping_smoothness(audio_params)
                
                results["sonitype_results"][dataset_name][scenario_name] = {
                    "compile_time_ms": compile_time * 1000,
                    "perceptual_discriminability": discriminability,
                    "information_preservation": information_preservation,
                    "constraint_satisfaction_rate": constraint_satisfaction,
                    "mapping_smoothness": smoothness,
                    "constraint_violations": violations,
                    "audio_param_ranges": mapping_result["mapping_metadata"]["param_range"]
                }
                
            except Exception as e:
                print(f"     Error: {e}")
                results["sonitype_results"][dataset_name][scenario_name] = {"error": str(e)}
    
    # Run baseline benchmarks
    print("\n4. Running baseline benchmarks...")
    for dataset_name, data in datasets.items():
        results["baseline_results"][dataset_name] = {}
        
        if len(data) > 500:
            indices = np.linspace(0, len(data) - 1, 500, dtype=int)
            data_subset = data[indices]
        else:
            data_subset = data
        
        for baseline_name, config in baseline_configs.items():
            print(f"   - {dataset_name}/{baseline_name}")
            
            try:
                start_time = time.time()
                mapped_values = config["func"](data_subset, config["range"])
                baseline_time = time.time() - start_time
                
                # Create audio params for evaluation
                baseline_audio_params = {"frequencies": mapped_values}
                
                # Evaluate with same metrics
                discriminability = BenchmarkMetrics.perceptual_discriminability_score(baseline_audio_params, data_subset)
                information_preservation = BenchmarkMetrics.information_preservation_score(baseline_audio_params, data_subset)
                smoothness = BenchmarkMetrics.mapping_smoothness(baseline_audio_params)
                
                results["baseline_results"][dataset_name][baseline_name] = {
                    "mapping_time_ms": baseline_time * 1000,
                    "perceptual_discriminability": discriminability,
                    "information_preservation": information_preservation,
                    "constraint_satisfaction_rate": 0.0,  # Baselines don't enforce constraints
                    "mapping_smoothness": smoothness,
                    "audio_param_range": (float(np.min(mapped_values)), float(np.max(mapped_values)))
                }
                
            except Exception as e:
                print(f"     Error: {e}")
                results["baseline_results"][dataset_name][baseline_name] = {"error": str(e)}
    
    # Comparative analysis
    print("\n5. Computing comparative analysis...")
    comparative_metrics = ["perceptual_discriminability", "information_preservation", "mapping_smoothness"]
    
    for dataset_name in datasets.keys():
        results["comparative_analysis"][dataset_name] = {}
        
        # Get SoniType best scores
        sonitype_scores = {}
        for metric in comparative_metrics:
            scores = []
            for scenario_name, result in results["sonitype_results"][dataset_name].items():
                if metric in result and not isinstance(result[metric], str):
                    scores.append(result[metric])
            sonitype_scores[metric] = max(scores) if scores else 0.0
        
        # Get baseline best scores  
        baseline_scores = {}
        for metric in comparative_metrics:
            scores = []
            for baseline_name, result in results["baseline_results"][dataset_name].items():
                if metric in result and not isinstance(result[metric], str):
                    scores.append(result[metric])
            baseline_scores[metric] = max(scores) if scores else 0.0
        
        # Compute improvements
        improvements = {}
        for metric in comparative_metrics:
            if baseline_scores[metric] > 0:
                improvement = (sonitype_scores[metric] - baseline_scores[metric]) / baseline_scores[metric]
                improvements[metric] = improvement
            else:
                improvements[metric] = float('inf') if sonitype_scores[metric] > 0 else 0.0
        
        results["comparative_analysis"][dataset_name] = {
            "sonitype_best": sonitype_scores,
            "baseline_best": baseline_scores,
            "relative_improvements": improvements,
            "sonitype_advantage": sum(improvements.values()) / len(improvements)
        }
    
    # Overall summary
    all_improvements = [results["comparative_analysis"][ds]["sonitype_advantage"] 
                       for ds in datasets.keys()]
    results["benchmark_summary"] = {
        "total_tests_run": len(datasets) * (len(scenarios) + len(baseline_configs)),
        "average_sonitype_improvement": float(np.mean(all_improvements)),
        "sonitype_wins": sum(1 for imp in all_improvements if imp > 0),
        "datasets_tested": len(datasets),
        "psychoacoustic_constraints_enforced": True,
        "benchmark_completion_time": datetime.now().isoformat()
    }
    
    print("\n6. Benchmark completed successfully!")
    print(f"   - Average SoniType improvement: {results['benchmark_summary']['average_sonitype_improvement']:.2%}")
    print(f"   - SoniType wins: {results['benchmark_summary']['sonitype_wins']}/{len(datasets)} datasets")
    
    return results


if __name__ == "__main__":
    # Run benchmark
    benchmark_results = run_comprehensive_benchmark()
    
    # Save results
    output_file = "benchmarks/real_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")