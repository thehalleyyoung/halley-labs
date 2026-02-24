"""
Experiment predictor for neural networks.

Predicts training loss curves, time per epoch, memory usage,
hyperparameter sensitivity, and failure modes without running experiments.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import warnings


@dataclass
class ExperimentPrediction:
    """Prediction for an experiment configuration."""
    predicted_loss_curve: List[float]
    time_per_epoch: float  # seconds
    memory_bytes: int
    risk_factors: List[str] = field(default_factory=list)
    convergence_probability: float = 0.0
    predicted_final_loss: float = 0.0
    hyperparameter_sensitivity: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    layer_widths: List[int]
    activation: str = "relu"
    learning_rate: float = 0.01
    batch_size: int = 32
    n_epochs: int = 100
    optimizer: str = "sgd"  # "sgd", "adam", "sgd_momentum"
    weight_decay: float = 0.0
    momentum: float = 0.0
    sigma_w: float = 1.0
    sigma_b: float = 0.0
    n_train: int = 1000
    n_test: int = 200
    input_dim: int = 10
    output_dim: int = 1
    task_type: str = "regression"
    has_residual: bool = False
    has_batchnorm: bool = False
    dropout: float = 0.0
    data_complexity: float = 1.0  # relative complexity of data
    hardware: str = "cpu"  # "cpu", "gpu", "tpu"
    dtype: str = "float32"  # "float32", "float64", "float16"


class LossCurvePredictor:
    """Predict training loss curves from architecture and optimizer settings."""

    def predict(self, config: ExperimentConfig) -> List[float]:
        """Predict loss curve for the given configuration.

        Uses a combination of:
        1. NTK convergence rate (for early training)
        2. Scaling law extrapolation (for late training)
        3. Optimizer-specific adjustments

        Args:
            config: Experiment configuration.

        Returns:
            List of predicted loss values per epoch.
        """
        n_epochs = config.n_epochs

        # Initial loss estimate
        if config.task_type == "regression":
            initial_loss = config.data_complexity
        elif config.task_type == "classification":
            initial_loss = np.log(max(config.output_dim, 2))
        else:
            initial_loss = 1.0

        # Convergence rate from NTK analysis
        conv_rate = self._estimate_convergence_rate(config)

        # Final achievable loss (from scaling law)
        final_loss = self._estimate_final_loss(config)

        # Generate loss curve
        loss_curve = []
        for epoch in range(n_epochs):
            t = epoch / max(n_epochs - 1, 1)

            # Exponential decay with NTK rate
            ntk_loss = initial_loss * np.exp(-conv_rate * epoch)

            # Power-law component (captures feature learning phase)
            power_loss = initial_loss / (1.0 + conv_rate * epoch) ** 0.5

            # Blend: early training is exponential, late is power-law
            blend = 1.0 / (1.0 + np.exp(-5 * (t - 0.3)))  # sigmoid transition
            loss = (1.0 - blend) * ntk_loss + blend * power_loss

            # Floor at final loss
            loss = max(loss, final_loss)

            # Optimizer-specific modifications
            if config.optimizer == "adam":
                loss *= 0.9  # Adam typically converges faster
            elif config.optimizer == "sgd_momentum":
                loss *= 0.95

            # Add noise from SGD
            if config.batch_size < config.n_train:
                noise_scale = np.sqrt(config.learning_rate / config.batch_size) * 0.01
                noise = np.random.RandomState(epoch).randn() * noise_scale
                loss = max(loss + noise * loss, final_loss * 0.9)

            loss_curve.append(float(loss))

        return loss_curve

    def _estimate_convergence_rate(self, config: ExperimentConfig) -> float:
        """Estimate convergence rate from architecture parameters."""
        depth = len(config.layer_widths) - 1
        min_width = min(config.layer_widths[1:]) if len(config.layer_widths) > 1 else 100

        # NTK convergence: rate ~ lr * lambda_min
        # lambda_min ~ sigma_w^2 for well-conditioned NTK
        lambda_min_est = config.sigma_w ** 2 / max(depth, 1)

        # Learning rate effect
        rate = config.learning_rate * lambda_min_est

        # Width effect: wider = more NTK-like = more predictable convergence
        width_factor = min(1.0, min_width / 100.0)
        rate *= width_factor

        # Depth penalty: deeper networks converge slower
        depth_penalty = 1.0 / (1.0 + 0.05 * depth ** 2)
        rate *= depth_penalty

        # BatchNorm helps
        if config.has_batchnorm:
            rate *= 1.5

        # Residual connections help deep networks
        if config.has_residual and depth > 5:
            rate *= 1.5

        return float(np.clip(rate, 1e-6, 10.0))

    def _estimate_final_loss(self, config: ExperimentConfig) -> float:
        """Estimate the final achievable loss."""
        n_params = self._count_params(config)

        # Approximation error: decreases with params
        approx_error = config.data_complexity / max(n_params, 1) ** 0.5

        # Estimation error: decreases with data
        est_error = n_params / max(config.n_train, 1)

        # Irreducible error
        irr_error = 0.001 * config.data_complexity

        # Regularization effect
        reg_error = config.weight_decay * 0.1

        total = approx_error + est_error + irr_error + reg_error
        return float(max(total, 1e-6))

    def _count_params(self, config: ExperimentConfig) -> int:
        """Count parameters in the network."""
        total = 0
        for i in range(len(config.layer_widths) - 1):
            total += config.layer_widths[i] * config.layer_widths[i + 1]
            total += config.layer_widths[i + 1]
        return total


class TimePredictor:
    """Predict training time per epoch."""

    # Approximate FLOP rates for different hardware (GFLOPS)
    FLOP_RATES = {
        "cpu": 50,
        "gpu": 10000,
        "tpu": 100000,
    }

    def predict(self, config: ExperimentConfig) -> float:
        """Predict time per epoch in seconds.

        Args:
            config: Experiment configuration.

        Returns:
            Predicted time per epoch in seconds.
        """
        # FLOPs per forward pass
        flops_forward = self._compute_forward_flops(config)

        # Backward is ~2x forward
        flops_per_sample = flops_forward * 3

        # Total FLOPs per epoch
        n_batches = max(config.n_train // config.batch_size, 1)
        flops_per_epoch = flops_per_sample * config.n_train

        # Hardware throughput
        gflops = self.FLOP_RATES.get(config.hardware, 50)
        flops_per_second = gflops * 1e9

        # Dtype adjustment
        if config.dtype == "float16":
            flops_per_second *= 2  # half precision is ~2x faster on GPU
        elif config.dtype == "float64":
            flops_per_second *= 0.5

        # Compute time
        time_seconds = flops_per_epoch / flops_per_second

        # Overhead: data loading, optimizer updates, etc.
        overhead_factor = 1.2
        if config.optimizer == "adam":
            overhead_factor = 1.3  # Adam has more state
        if config.has_batchnorm:
            overhead_factor *= 1.1
        if config.dropout > 0:
            overhead_factor *= 1.05

        return float(time_seconds * overhead_factor)

    def _compute_forward_flops(self, config: ExperimentConfig) -> int:
        """Compute FLOPs for one forward pass."""
        total = 0
        for i in range(len(config.layer_widths) - 1):
            # Matrix multiply: 2 * m * n FLOPs
            total += 2 * config.layer_widths[i] * config.layer_widths[i + 1]
            # Bias add: n FLOPs
            total += config.layer_widths[i + 1]
            # Activation: n FLOPs (approximate)
            total += config.layer_widths[i + 1]
        return total


class MemoryPredictor:
    """Predict peak memory usage."""

    DTYPE_SIZES = {
        "float16": 2,
        "float32": 4,
        "float64": 8,
    }

    def predict(self, config: ExperimentConfig) -> int:
        """Predict peak memory usage in bytes.

        Includes:
        - Model parameters
        - Gradients
        - Optimizer state
        - Activations (for backprop)
        - Input batch

        Args:
            config: Experiment configuration.

        Returns:
            Peak memory in bytes.
        """
        bytes_per_float = self.DTYPE_SIZES.get(config.dtype, 4)

        # Model parameters
        n_params = 0
        for i in range(len(config.layer_widths) - 1):
            n_params += config.layer_widths[i] * config.layer_widths[i + 1]
            n_params += config.layer_widths[i + 1]
        param_memory = n_params * bytes_per_float

        # Gradients: same size as parameters
        grad_memory = param_memory

        # Optimizer state
        if config.optimizer == "adam":
            opt_memory = 2 * param_memory  # first and second moments
        elif config.optimizer == "sgd_momentum":
            opt_memory = param_memory  # momentum buffer
        else:
            opt_memory = 0

        # Activations: need to store for backward pass
        activation_memory = 0
        for i in range(len(config.layer_widths)):
            activation_memory += config.batch_size * config.layer_widths[i] * bytes_per_float

        # Input batch
        input_memory = config.batch_size * config.input_dim * bytes_per_float

        # BatchNorm state
        bn_memory = 0
        if config.has_batchnorm:
            for w in config.layer_widths[1:]:
                bn_memory += 4 * w * bytes_per_float  # running mean, var, gamma, beta

        total = param_memory + grad_memory + opt_memory + activation_memory + input_memory + bn_memory

        # Overhead: framework, etc. (~20%)
        total = int(total * 1.2)

        return total


class HyperparameterSensitivity:
    """Analyze which hyperparameters matter most."""

    def analyze(self, config: ExperimentConfig) -> Dict[str, float]:
        """Compute sensitivity of performance to each hyperparameter.

        Uses local sensitivity analysis: how much does predicted loss
        change when each hyperparameter is perturbed by 10%?

        Args:
            config: Base experiment configuration.

        Returns:
            Dictionary mapping hyperparameter name to sensitivity score.
        """
        predictor = LossCurvePredictor()
        base_losses = predictor.predict(config)
        base_final_loss = base_losses[-1] if base_losses else 1.0

        sensitivities = {}

        # Learning rate sensitivity
        config_lr = self._copy_config(config)
        config_lr.learning_rate *= 1.1
        losses_lr = predictor.predict(config_lr)
        sensitivities["learning_rate"] = self._relative_change(
            base_final_loss, losses_lr[-1] if losses_lr else base_final_loss
        )

        # Batch size sensitivity
        config_bs = self._copy_config(config)
        config_bs.batch_size = max(1, int(config_bs.batch_size * 1.1))
        losses_bs = predictor.predict(config_bs)
        sensitivities["batch_size"] = self._relative_change(
            base_final_loss, losses_bs[-1] if losses_bs else base_final_loss
        )

        # Weight init sensitivity
        config_sw = self._copy_config(config)
        config_sw.sigma_w *= 1.1
        losses_sw = predictor.predict(config_sw)
        sensitivities["sigma_w"] = self._relative_change(
            base_final_loss, losses_sw[-1] if losses_sw else base_final_loss
        )

        # Weight decay sensitivity
        config_wd = self._copy_config(config)
        config_wd.weight_decay = max(config.weight_decay + 1e-4, config.weight_decay * 1.1)
        losses_wd = predictor.predict(config_wd)
        sensitivities["weight_decay"] = self._relative_change(
            base_final_loss, losses_wd[-1] if losses_wd else base_final_loss
        )

        # Width sensitivity (change first hidden layer)
        if len(config.layer_widths) > 2:
            config_w = self._copy_config(config)
            config_w.layer_widths = config.layer_widths.copy()
            config_w.layer_widths[1] = int(config_w.layer_widths[1] * 1.1)
            losses_w = predictor.predict(config_w)
            sensitivities["width"] = self._relative_change(
                base_final_loss, losses_w[-1] if losses_w else base_final_loss
            )

        # Depth sensitivity (add a layer)
        config_d = self._copy_config(config)
        if len(config_d.layer_widths) > 2:
            mid_width = config_d.layer_widths[1]
            config_d.layer_widths = (
                config.layer_widths[:2] + [mid_width] + config.layer_widths[2:]
            )
            losses_d = predictor.predict(config_d)
            sensitivities["depth"] = self._relative_change(
                base_final_loss, losses_d[-1] if losses_d else base_final_loss
            )

        return sensitivities

    def _relative_change(self, base: float, perturbed: float) -> float:
        """Compute relative change."""
        return abs(perturbed - base) / max(abs(base), 1e-10)

    def _copy_config(self, config: ExperimentConfig) -> ExperimentConfig:
        """Create a copy of the config."""
        return ExperimentConfig(
            layer_widths=config.layer_widths.copy(),
            activation=config.activation,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            optimizer=config.optimizer,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            sigma_w=config.sigma_w,
            sigma_b=config.sigma_b,
            n_train=config.n_train,
            n_test=config.n_test,
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            task_type=config.task_type,
            has_residual=config.has_residual,
            has_batchnorm=config.has_batchnorm,
            dropout=config.dropout,
            data_complexity=config.data_complexity,
            hardware=config.hardware,
            dtype=config.dtype,
        )


class FailureModePredictor:
    """Predict failure modes for a given configuration."""

    def predict(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Predict potential failure modes.

        Args:
            config: Experiment configuration.

        Returns:
            List of failure mode dictionaries with type, probability, and advice.
        """
        failures = []

        # Check for divergence
        divergence_risk = self._check_divergence_risk(config)
        if divergence_risk > 0.1:
            failures.append({
                "type": "divergence",
                "probability": float(divergence_risk),
                "description": "Training loss may diverge (increase without bound).",
                "advice": "Reduce learning rate or use gradient clipping.",
            })

        # Check for plateau
        plateau_risk = self._check_plateau_risk(config)
        if plateau_risk > 0.1:
            failures.append({
                "type": "plateau",
                "probability": float(plateau_risk),
                "description": "Training may plateau before reaching good performance.",
                "advice": "Use learning rate warmup/scheduling or Adam optimizer.",
            })

        # Check for oscillation
        oscillation_risk = self._check_oscillation_risk(config)
        if oscillation_risk > 0.1:
            failures.append({
                "type": "oscillation",
                "probability": float(oscillation_risk),
                "description": "Loss may oscillate instead of converging.",
                "advice": "Reduce learning rate or increase batch size.",
            })

        # Check for overfitting
        overfit_risk = self._check_overfitting_risk(config)
        if overfit_risk > 0.1:
            failures.append({
                "type": "overfitting",
                "probability": float(overfit_risk),
                "description": "Model may overfit to training data.",
                "advice": "Add regularization (weight decay, dropout) or get more data.",
            })

        # Check for vanishing gradients
        vanishing_risk = self._check_vanishing_gradient_risk(config)
        if vanishing_risk > 0.1:
            failures.append({
                "type": "vanishing_gradients",
                "probability": float(vanishing_risk),
                "description": "Gradients may vanish in deep layers.",
                "advice": "Use residual connections, batch normalization, or better initialization.",
            })

        # Check for OOM
        oom_risk = self._check_oom_risk(config)
        if oom_risk > 0.1:
            failures.append({
                "type": "out_of_memory",
                "probability": float(oom_risk),
                "description": "May run out of memory during training.",
                "advice": "Reduce batch size, model size, or use gradient checkpointing.",
            })

        # Sort by probability
        failures.sort(key=lambda x: x["probability"], reverse=True)
        return failures

    def _check_divergence_risk(self, config: ExperimentConfig) -> float:
        """Check risk of training divergence."""
        risk = 0.0
        depth = len(config.layer_widths) - 1

        # High learning rate
        critical_lr = 2.0 / (config.sigma_w ** 2 * depth)
        if config.learning_rate > critical_lr:
            risk += 0.5
        elif config.learning_rate > 0.5 * critical_lr:
            risk += 0.2

        # Large sigma_w
        if config.sigma_w > 2.0:
            risk += 0.2

        # No batch norm + deep
        if not config.has_batchnorm and depth > 10:
            risk += 0.1

        return min(risk, 1.0)

    def _check_plateau_risk(self, config: ExperimentConfig) -> float:
        """Check risk of training plateau."""
        risk = 0.0
        depth = len(config.layer_widths) - 1

        # Very small learning rate
        if config.learning_rate < 1e-5:
            risk += 0.3

        # Sigmoid activation (saturates)
        if config.activation == "sigmoid":
            risk += 0.3

        # Too few epochs
        min_width = min(config.layer_widths[1:]) if len(config.layer_widths) > 1 else 100
        expected_epochs = max(10, 100 * depth / max(min_width, 1) * 10)
        if config.n_epochs < expected_epochs * 0.5:
            risk += 0.2

        return min(risk, 1.0)

    def _check_oscillation_risk(self, config: ExperimentConfig) -> float:
        """Check risk of loss oscillation."""
        risk = 0.0
        depth = len(config.layer_widths) - 1

        # Learning rate too high for SGD
        if config.optimizer == "sgd" and config.learning_rate > 0.1:
            risk += 0.3

        # Small batch size with high LR
        temperature = config.learning_rate / max(config.batch_size, 1)
        if temperature > 0.01:
            risk += 0.3

        return min(risk, 1.0)

    def _check_overfitting_risk(self, config: ExperimentConfig) -> float:
        """Check risk of overfitting."""
        n_params = sum(
            config.layer_widths[i] * config.layer_widths[i + 1] + config.layer_widths[i + 1]
            for i in range(len(config.layer_widths) - 1)
        )

        ratio = n_params / max(config.n_train, 1)

        risk = 0.0
        if ratio > 10:
            risk += 0.4
        elif ratio > 1:
            risk += 0.2

        # No regularization
        if config.weight_decay == 0 and config.dropout == 0:
            risk += 0.2

        # Few training samples
        if config.n_train < 100:
            risk += 0.2

        return min(risk, 1.0)

    def _check_vanishing_gradient_risk(self, config: ExperimentConfig) -> float:
        """Check risk of vanishing gradients."""
        depth = len(config.layer_widths) - 1
        risk = 0.0

        # Deep network without residual/batchnorm
        if depth > 10 and not config.has_residual and not config.has_batchnorm:
            risk += 0.4

        # Sigmoid/tanh can saturate
        if config.activation in ("sigmoid", "tanh") and depth > 5:
            risk += 0.3

        # Small sigma_w
        if config.sigma_w < 0.5 and depth > 5:
            risk += 0.2

        return min(risk, 1.0)

    def _check_oom_risk(self, config: ExperimentConfig) -> float:
        """Check risk of out-of-memory."""
        mem_predictor = MemoryPredictor()
        predicted_memory = mem_predictor.predict(config)

        # Typical memory limits
        memory_limits = {
            "cpu": 16e9,  # 16 GB
            "gpu": 8e9,   # 8 GB (typical consumer GPU)
            "tpu": 16e9,  # 16 GB per core
        }

        limit = memory_limits.get(config.hardware, 16e9)
        usage_fraction = predicted_memory / limit

        if usage_fraction > 0.9:
            return 0.8
        elif usage_fraction > 0.7:
            return 0.3
        elif usage_fraction > 0.5:
            return 0.1
        return 0.0


class ConfigurationRanker:
    """Rank multiple experiment configurations by predicted performance."""

    def rank(self, configs: List[ExperimentConfig]) -> List[Dict[str, Any]]:
        """Rank configurations by predicted performance.

        Args:
            configs: List of experiment configurations.

        Returns:
            Sorted list of dicts with config index, predicted loss, and scores.
        """
        loss_predictor = LossCurvePredictor()
        time_predictor = TimePredictor()
        memory_predictor = MemoryPredictor()
        failure_predictor = FailureModePredictor()

        rankings = []
        for i, config in enumerate(configs):
            loss_curve = loss_predictor.predict(config)
            final_loss = loss_curve[-1] if loss_curve else float("inf")
            time_per_epoch = time_predictor.predict(config)
            memory = memory_predictor.predict(config)
            failures = failure_predictor.predict(config)

            total_risk = sum(f["probability"] for f in failures)
            convergence_prob = max(0.0, 1.0 - total_risk)

            # Composite score (lower is better)
            score = final_loss * (1.0 + total_risk)

            rankings.append({
                "config_index": i,
                "predicted_final_loss": float(final_loss),
                "time_per_epoch": float(time_per_epoch),
                "memory_bytes": memory,
                "convergence_probability": float(convergence_prob),
                "risk_score": float(total_risk),
                "composite_score": float(score),
                "top_risks": [f["type"] for f in failures[:3]],
            })

        rankings.sort(key=lambda x: x["composite_score"])
        for rank, r in enumerate(rankings):
            r["rank"] = rank + 1

        return rankings


class ExperimentPredictor:
    """Main class for predicting experiment outcomes."""

    def __init__(self):
        self.loss_predictor = LossCurvePredictor()
        self.time_predictor = TimePredictor()
        self.memory_predictor = MemoryPredictor()
        self.sensitivity_analyzer = HyperparameterSensitivity()
        self.failure_predictor = FailureModePredictor()
        self.ranker = ConfigurationRanker()

    def predict(self, config: ExperimentConfig) -> ExperimentPrediction:
        """Predict full experiment outcome.

        Args:
            config: Experiment configuration.

        Returns:
            ExperimentPrediction with all predictions.
        """
        loss_curve = self.loss_predictor.predict(config)
        time_per_epoch = self.time_predictor.predict(config)
        memory = self.memory_predictor.predict(config)
        sensitivity = self.sensitivity_analyzer.analyze(config)
        failures = self.failure_predictor.predict(config)

        risk_factors = [f["description"] for f in failures if f["probability"] > 0.2]
        total_risk = sum(f["probability"] for f in failures)
        convergence_prob = max(0.0, 1.0 - total_risk)

        return ExperimentPrediction(
            predicted_loss_curve=loss_curve,
            time_per_epoch=time_per_epoch,
            memory_bytes=memory,
            risk_factors=risk_factors,
            convergence_probability=convergence_prob,
            predicted_final_loss=loss_curve[-1] if loss_curve else float("inf"),
            hyperparameter_sensitivity=sensitivity,
        )

    def rank_configs(self, configs: List[ExperimentConfig]) -> List[Dict[str, Any]]:
        """Rank multiple configurations."""
        return self.ranker.rank(configs)
