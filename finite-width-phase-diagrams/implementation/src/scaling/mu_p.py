"""
µP (Maximal Update Parameterization) implementation.

Implements the scaling theory from Yang & Hu (2021) "Tensor Programs V:
Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer".

In µP, layer-wise initialization scales and learning rates are set so that
the network dynamics remain stable (O(1) activations, gradients, updates)
as width → ∞, enabling hyperparameter transfer from narrow to wide models.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from typing import List, Dict, Tuple, Optional, Callable, Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAYER_TYPES = ("input", "hidden", "output", "attention", "embedding")

# µP exponent table: (init_scale_exp, lr_scale_exp, output_mult_exp)
# expressed as powers of (width / base_width).
_MUP_EXPONENTS = {
    "input":     (0.0,  0.0,  0.0),
    "hidden":    (-0.5, -1.0, 0.0),
    "output":    (-0.5, -1.0, -1.0),
    "attention": (-0.5, -1.0, 0.0),
    "embedding": (0.0,  -1.0, 0.0),
}

# Standard parameterization exponents for comparison
_SP_EXPONENTS = {
    "input":     (0.0,  0.0, 0.0),
    "hidden":    (-0.5, 0.0, 0.0),
    "output":    (-0.5, 0.0, 0.0),
    "attention": (-0.5, 0.0, 0.0),
    "embedding": (0.0,  0.0, 0.0),
}


# ===================================================================
# 1. MuPScalingComputer
# ===================================================================

class MuPScalingComputer:
    """Compute µP scaling exponents from architecture specification.

    For every layer the key quantities are:
        σ_init  ∝  (n / n_base)^a          initialisation std-dev
        η_layer ∝  (n / n_base)^b          per-layer learning rate
        α_out   ∝  (n / n_base)^c          output multiplier
    where n = layer width.  The exponents (a, b, c) depend on the layer
    type and are chosen so that pre-activations, gradients and parameter
    updates all stay O(1) in the infinite-width limit.
    """

    def __init__(self, base_width: int = 64):
        if base_width < 1:
            raise ValueError("base_width must be positive")
        self.base_width = base_width

    # ------------------------------------------------------------------
    def compute_scaling_exponents(
        self,
        layer_types: List[str],
        layer_widths: List[int],
    ) -> List[Dict[str, float]]:
        """Return per-layer scaling dicts with keys a, b, c, sigma, lr_mult."""
        if len(layer_types) != len(layer_widths):
            raise ValueError("layer_types and layer_widths must have same length")

        results = []
        n_layers = len(layer_types)
        for idx, (ltype, width) in enumerate(zip(layer_types, layer_widths)):
            # Determine effective layer category
            if ltype == "attention":
                cat = "attention"
            elif ltype == "embedding":
                cat = "embedding"
            elif idx == 0:
                cat = "input"
            elif idx == n_layers - 1:
                cat = "output"
            else:
                cat = "hidden"

            a, b, c = _MUP_EXPONENTS[cat]
            ratio = width / self.base_width
            sigma = ratio ** a if ratio > 0 else 0.0
            lr_mult = ratio ** b if ratio > 0 else 0.0
            out_mult = ratio ** c if ratio > 0 else 0.0

            results.append({
                "layer_type": cat,
                "width": width,
                "a": a,
                "b": b,
                "c": c,
                "sigma_mult": sigma,
                "lr_mult": lr_mult,
                "output_mult": out_mult,
            })
        return results

    # ------------------------------------------------------------------
    def input_layer_scaling(
        self, fan_in: int, fan_out: int
    ) -> Dict[str, float]:
        """µP scaling for the first (input) layer.

        In µP the input layer uses standard parameterization:
            σ_init = 1/√fan_in,  η = η_base  (no width correction).
        """
        sigma = 1.0 / np.sqrt(fan_in)
        return {
            "sigma_init": sigma,
            "lr_multiplier": 1.0,
            "output_multiplier": 1.0,
            "fan_in": fan_in,
            "fan_out": fan_out,
        }

    def hidden_layer_scaling(
        self, fan_in: int, fan_out: int
    ) -> Dict[str, float]:
        """µP scaling for hidden layers.

        σ_init = 1/√fan_in  (same formula, but fan_in grows with width)
        η_layer = η_base · (base_width / fan_in)   i.e. ∝ 1/width
        """
        sigma = 1.0 / np.sqrt(fan_in)
        lr_mult = self.base_width / fan_in
        return {
            "sigma_init": sigma,
            "lr_multiplier": lr_mult,
            "output_multiplier": 1.0,
            "fan_in": fan_in,
            "fan_out": fan_out,
        }

    def output_layer_scaling(
        self, fan_in: int, fan_out: int
    ) -> Dict[str, float]:
        """µP scaling for the last (output/readout) layer.

        σ_init = 1/√fan_in,   η ∝ 1/fan_in,   output multiplied by 1/fan_in.
        """
        sigma = 1.0 / np.sqrt(fan_in)
        lr_mult = self.base_width / fan_in
        out_mult = self.base_width / fan_in
        return {
            "sigma_init": sigma,
            "lr_multiplier": lr_mult,
            "output_multiplier": out_mult,
            "fan_in": fan_in,
            "fan_out": fan_out,
        }

    def attention_scaling(
        self, d_model: int, n_heads: int
    ) -> Dict[str, float]:
        """µP scaling for multi-head attention.

        The key/query dot-product is scaled by 1/d_head (not 1/√d_head)
        to keep attention logits O(1).  Projection matrices use hidden-layer
        µP scaling.
        """
        d_head = d_model // n_heads
        attn_scale = 1.0 / d_head  # µP attention scale
        sigma_qkv = 1.0 / np.sqrt(d_model)
        sigma_out = 1.0 / np.sqrt(d_model)
        lr_mult = self.base_width / d_model
        return {
            "attention_scale": attn_scale,
            "sigma_qkv": sigma_qkv,
            "sigma_out_proj": sigma_out,
            "lr_multiplier": lr_mult,
            "d_head": d_head,
            "n_heads": n_heads,
        }

    def embedding_scaling(
        self, vocab_size: int, d_model: int
    ) -> Dict[str, float]:
        """µP scaling for embedding layers.

        Embedding vectors are initialised O(1) (not 1/√d) and the learning
        rate scales as 1/width to keep updates O(1).
        """
        sigma = 1.0  # O(1) init
        lr_mult = self.base_width / d_model
        return {
            "sigma_init": sigma,
            "lr_multiplier": lr_mult,
            "output_multiplier": 1.0,
            "vocab_size": vocab_size,
            "d_model": d_model,
        }

    # ------------------------------------------------------------------
    def init_scale(
        self, layer_type: str, fan_in: int, fan_out: int
    ) -> float:
        """Compute σ_init for a given layer type under µP."""
        if layer_type not in LAYER_TYPES:
            raise ValueError(f"Unknown layer_type {layer_type!r}")
        dispatch = {
            "input": self.input_layer_scaling,
            "hidden": self.hidden_layer_scaling,
            "output": self.output_layer_scaling,
        }
        if layer_type in dispatch:
            return dispatch[layer_type](fan_in, fan_out)["sigma_init"]
        if layer_type == "attention":
            return self.attention_scaling(fan_in, max(fan_out, 1))["sigma_qkv"]
        # embedding
        return self.embedding_scaling(fan_out, fan_in)["sigma_init"]

    def lr_scale(
        self, layer_type: str, fan_in: int, fan_out: int, base_lr: float
    ) -> float:
        """Compute per-layer learning rate η_layer under µP."""
        dispatch = {
            "input": self.input_layer_scaling,
            "hidden": self.hidden_layer_scaling,
            "output": self.output_layer_scaling,
        }
        if layer_type in dispatch:
            mult = dispatch[layer_type](fan_in, fan_out)["lr_multiplier"]
        elif layer_type == "attention":
            mult = self.attention_scaling(fan_in, max(fan_out, 1))["lr_multiplier"]
        elif layer_type == "embedding":
            mult = self.embedding_scaling(fan_out, fan_in)["lr_multiplier"]
        else:
            raise ValueError(f"Unknown layer_type {layer_type!r}")
        return base_lr * mult

    def output_multiplier(self, fan_in: int) -> float:
        """Output layer multiplier = base_width / fan_in  (≈ 1/width)."""
        return self.base_width / fan_in

    # ------------------------------------------------------------------
    def validate_scaling_table(
        self, scaling_table: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate that a µP scaling table is self-consistent.

        Checks:
        1. Exponents match µP theory for each layer type.
        2. Product of init_scale and lr_multiplier gives correct update size.
        3. Output multiplier decreases with width for the output layer.
        """
        errors: List[str] = []
        warnings: List[str] = []

        for idx, entry in enumerate(scaling_table):
            ltype = entry.get("layer_type", "hidden")
            expected_a, expected_b, expected_c = _MUP_EXPONENTS.get(
                ltype, _MUP_EXPONENTS["hidden"]
            )
            a = entry.get("a", 0.0)
            b = entry.get("b", 0.0)
            c = entry.get("c", 0.0)

            if not np.isclose(a, expected_a, atol=1e-6):
                errors.append(
                    f"Layer {idx} ({ltype}): init exponent a={a}, expected {expected_a}"
                )
            if not np.isclose(b, expected_b, atol=1e-6):
                errors.append(
                    f"Layer {idx} ({ltype}): lr exponent b={b}, expected {expected_b}"
                )
            if not np.isclose(c, expected_c, atol=1e-6):
                errors.append(
                    f"Layer {idx} ({ltype}): output exponent c={c}, expected {expected_c}"
                )

            # Update size ~ σ · η should → 0 for hidden, stay O(1) for input
            update_exp = a + b
            if ltype == "input" and not np.isclose(update_exp, 0.0, atol=1e-6):
                warnings.append(
                    f"Layer {idx}: input update exponent {update_exp} != 0"
                )
            elif ltype in ("hidden", "output", "attention") and not np.isclose(
                update_exp, -1.5, atol=1e-6
            ):
                warnings.append(
                    f"Layer {idx}: {ltype} update exponent {update_exp}, "
                    f"expected -1.5 for feature learning"
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "n_layers": len(scaling_table),
        }

    # ------------------------------------------------------------------
    def standard_vs_mup_comparison(
        self, widths: List[int]
    ) -> Dict[str, np.ndarray]:
        """Compare SP vs µP scaling as a function of width.

        Returns arrays of (init_std, lr_mult, output_mult) for each width
        under both parameterizations, for a representative hidden layer.
        """
        widths_arr = np.asarray(widths, dtype=float)
        ratios = widths_arr / self.base_width

        sp_a, sp_b, sp_c = _SP_EXPONENTS["hidden"]
        mu_a, mu_b, mu_c = _MUP_EXPONENTS["hidden"]

        return {
            "widths": widths_arr,
            "sp_init_std": ratios ** sp_a,
            "sp_lr_mult": ratios ** sp_b,
            "sp_output_mult": ratios ** sp_c,
            "mup_init_std": ratios ** mu_a,
            "mup_lr_mult": ratios ** mu_b,
            "mup_output_mult": ratios ** mu_c,
            "sp_update_size": ratios ** (sp_a + sp_b),
            "mup_update_size": ratios ** (mu_a + mu_b),
        }


# ===================================================================
# 2. MuPLearningRateTransfer
# ===================================================================

class MuPLearningRateTransfer:
    """Transfer learning rates across widths using µP theory.

    Core idea: under µP the optimal LR is *width-independent* for hidden
    layers.  One tunes LR at a small base width and uses the same value
    (up to the µP per-layer multiplier) at any target width.
    """

    def __init__(self, base_width: int, base_lr: float):
        if base_width < 1:
            raise ValueError("base_width must be positive")
        if base_lr <= 0:
            raise ValueError("base_lr must be positive")
        self.base_width = base_width
        self.base_lr = base_lr
        self._computer = MuPScalingComputer(base_width)

    # ------------------------------------------------------------------
    def transfer_lr(self, target_width: int) -> float:
        """Transfer the global base LR to *target_width*.

        Under µP the base LR itself does not change — the per-layer
        multipliers absorb the width dependence.  This method returns
        the global LR that should be passed to the optimiser (unchanged).
        """
        return self.base_lr

    def per_layer_lr(
        self, layer_type: str, width: int, base_lr: float
    ) -> float:
        """Compute the per-layer learning rate at a given width."""
        # fan_in ≈ width for most layers
        return self._computer.lr_scale(
            layer_type, fan_in=width, fan_out=width, base_lr=base_lr
        )

    # ------------------------------------------------------------------
    def optimal_lr_at_width(
        self,
        width: int,
        loss_fn: Callable,
        data: Tuple[np.ndarray, np.ndarray],
        n_trials: int = 20,
    ) -> Tuple[float, float]:
        """Grid-search for the optimal LR at a given width.

        Parameters
        ----------
        width : network width
        loss_fn : callable(weights, x, y) → scalar loss
        data : (x, y) tuple
        n_trials : number of LR candidates to try

        Returns
        -------
        (best_lr, best_loss) tuple
        """
        x, y = data
        lr_grid = np.logspace(-5, 1, n_trials)
        best_lr, best_loss = lr_grid[0], np.inf

        rng = np.random.default_rng(42)
        d_in = x.shape[1] if x.ndim > 1 else 1
        d_out = y.shape[1] if y.ndim > 1 else 1
        sigma = 1.0 / np.sqrt(width)

        for lr in lr_grid:
            # Simple one-hidden-layer network for probing
            W1 = rng.normal(0, sigma, (d_in, width))
            W2 = rng.normal(0, sigma, (width, d_out))
            weights = [W1, W2]

            lr_hidden = lr * (self.base_width / width)
            lr_out = lr * (self.base_width / width)

            # A few SGD steps
            for _ in range(50):
                h = np.maximum(x @ weights[0], 0)  # ReLU
                out = h @ weights[1] * (self.base_width / width)
                err = out - y
                loss = 0.5 * np.mean(err ** 2)

                # Backward
                d_out_w = h.T @ err / x.shape[0] * (self.base_width / width)
                d_h = (err @ weights[1].T) * (h > 0) * (self.base_width / width)
                d_W1 = x.T @ d_h / x.shape[0]

                weights[1] = weights[1] - lr_out * d_out_w
                weights[0] = weights[0] - lr_hidden * d_W1

            # Evaluate final loss
            h = np.maximum(x @ weights[0], 0)
            final_out = h @ weights[1] * (self.base_width / width)
            final_loss = 0.5 * np.mean((final_out - y) ** 2)

            if np.isfinite(final_loss) and final_loss < best_loss:
                best_loss = final_loss
                best_lr = lr

        return float(best_lr), float(best_loss)

    # ------------------------------------------------------------------
    def lr_transfer_error(
        self,
        widths: List[int],
        optimal_lrs: List[float],
        predicted_lrs: List[float],
    ) -> Dict[str, float]:
        """Quantify error between optimal and µP-predicted LRs.

        Returns relative errors and summary statistics.
        """
        opt = np.asarray(optimal_lrs, dtype=float)
        pred = np.asarray(predicted_lrs, dtype=float)
        rel_err = np.abs(opt - pred) / (np.abs(opt) + 1e-12)
        log_ratio = np.log10(pred / (opt + 1e-12))
        return {
            "widths": np.asarray(widths),
            "relative_errors": rel_err,
            "log10_ratio": log_ratio,
            "mean_relative_error": float(np.mean(rel_err)),
            "max_relative_error": float(np.max(rel_err)),
            "mean_log10_ratio": float(np.mean(np.abs(log_ratio))),
        }

    def lr_width_scaling_plot_data(
        self,
        widths: List[int],
        optimal_lrs: List[float],
    ) -> Dict[str, np.ndarray]:
        """Prepare data for an LR-vs-width log-log plot."""
        w = np.asarray(widths, dtype=float)
        lrs = np.asarray(optimal_lrs, dtype=float)
        log_w = np.log(w)
        log_lr = np.log(lrs + 1e-30)

        # Fit log(lr) = α·log(w) + β
        if len(w) >= 2:
            slope, intercept, r_val, _, _ = stats.linregress(log_w, log_lr)
        else:
            slope, intercept, r_val = 0.0, 0.0, 0.0

        return {
            "widths": w,
            "optimal_lrs": lrs,
            "log_widths": log_w,
            "log_lrs": log_lr,
            "fitted_slope": slope,
            "fitted_intercept": intercept,
            "r_squared": r_val ** 2,
            "predicted_mup_slope": 0.0,  # µP predicts width-independent LR
        }

    # ------------------------------------------------------------------
    def verify_lr_transfer(
        self,
        widths: List[int],
        loss_fn: Callable,
        data_fn: Callable,
        n_steps: int = 1000,
    ) -> Dict[str, Any]:
        """Train at multiple widths, verify loss curves coincide under µP.

        Parameters
        ----------
        widths : list of widths to test
        loss_fn : callable(pred, y) → scalar
        data_fn : callable() → (x, y)
        n_steps : SGD steps per trial

        Returns loss curves and width-independence metrics.
        """
        x, y = data_fn()
        d_in = x.shape[1] if x.ndim > 1 else 1
        d_out = y.shape[1] if y.ndim > 1 else 1

        all_curves = {}
        rng = np.random.default_rng(0)

        for width in widths:
            sigma = 1.0 / np.sqrt(width)
            W1 = rng.normal(0, sigma, (d_in, width))
            W2 = rng.normal(0, sigma, (width, d_out))

            out_mult = self.base_width / width
            lr_mult = self.base_width / width

            losses = np.zeros(n_steps)
            for step in range(n_steps):
                h = np.maximum(x @ W1, 0)
                pred = h @ W2 * out_mult
                err = pred - y
                loss_val = float(loss_fn(pred, y))
                losses[step] = loss_val

                grad_W2 = h.T @ err / x.shape[0] * out_mult
                grad_h = (err @ W2.T) * (h > 0) * out_mult
                grad_W1 = x.T @ grad_h / x.shape[0]

                W2 -= self.base_lr * lr_mult * grad_W2
                W1 -= self.base_lr * lr_mult * grad_W1

            all_curves[width] = losses

        # Measure spread of final losses
        final_losses = np.array([all_curves[w][-1] for w in widths])
        cv = float(np.std(final_losses) / (np.mean(final_losses) + 1e-12))

        return {
            "loss_curves": all_curves,
            "final_losses": {w: float(all_curves[w][-1]) for w in widths},
            "cv_final_loss": cv,
            "width_independent": cv < 0.2,
        }

    # ------------------------------------------------------------------
    def transfer_hyperparameters(
        self,
        base_config: Dict[str, Any],
        target_width: int,
    ) -> Dict[str, Any]:
        """Transfer a full set of hyperparameters from base to target width.

        Transfers: lr, init scales, output multiplier, momentum (unchanged),
        weight decay (unchanged under µP reparameterization).
        """
        ratio = target_width / self.base_width
        transferred = dict(base_config)
        transferred["width"] = target_width

        # Global LR stays the same under µP (per-layer mults handle it)
        transferred["lr"] = base_config.get("lr", self.base_lr)

        # Per-layer LR multipliers
        layer_types = base_config.get("layer_types", ["input", "hidden", "output"])
        lr_mults = {}
        for lt in layer_types:
            _, b, _ = _MUP_EXPONENTS.get(lt, _MUP_EXPONENTS["hidden"])
            lr_mults[lt] = ratio ** b
        transferred["lr_multipliers"] = lr_mults

        # Init scales
        init_mults = {}
        for lt in layer_types:
            a, _, _ = _MUP_EXPONENTS.get(lt, _MUP_EXPONENTS["hidden"])
            init_mults[lt] = ratio ** a
        transferred["init_multipliers"] = init_mults

        # Output multiplier
        _, _, c = _MUP_EXPONENTS["output"]
        transferred["output_multiplier"] = ratio ** c

        return transferred


# ===================================================================
# 3. MuPInitialization
# ===================================================================

class MuPInitialization:
    """µP-corrected weight initialization schemes.

    Standard He/LeCun/Xavier init formulae are adjusted by width-dependent
    factors so that forward-pass activations stay O(1) as width grows.
    """

    def __init__(self, base_width: int = 64):
        self.base_width = base_width

    # ------------------------------------------------------------------
    def compute_init_std(
        self,
        layer_type: str,
        fan_in: int,
        fan_out: int,
        width: int,
    ) -> float:
        """Compute µP initialisation standard deviation.

        For hidden layers:  σ = 1/√fan_in   (fan_in grows with width,
        so σ already decreases — no extra correction needed beyond the
        standard formula when fan_in is set correctly).
        For embeddings:  σ = 1  (O(1) init).
        """
        if layer_type == "embedding":
            return 1.0
        base_sigma = 1.0 / np.sqrt(fan_in)
        a, _, _ = _MUP_EXPONENTS.get(layer_type, _MUP_EXPONENTS["hidden"])
        ratio = width / self.base_width
        return base_sigma * (ratio ** a) if ratio > 0 else base_sigma

    # ------------------------------------------------------------------
    def initialize_layer(
        self,
        layer_type: str,
        shape: Tuple[int, ...],
        width: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Generate µP-initialized weight matrix."""
        if rng is None:
            rng = np.random.default_rng()
        fan_in = shape[0]
        fan_out = shape[1] if len(shape) > 1 else 1
        sigma = self.compute_init_std(layer_type, fan_in, fan_out, width)
        return rng.normal(0.0, sigma, shape)

    # ------------------------------------------------------------------
    def initialize_network(
        self,
        layer_specs: List[Dict[str, Any]],
        width: int,
        seed: int = 0,
    ) -> List[np.ndarray]:
        """Initialize a full network from a list of layer specs.

        Each spec is a dict with keys: layer_type, shape.
        """
        rng = np.random.default_rng(seed)
        weights = []
        for spec in layer_specs:
            ltype = spec["layer_type"]
            shape = spec["shape"]
            W = self.initialize_layer(ltype, shape, width, rng)
            weights.append(W)
        return weights

    # ------------------------------------------------------------------
    def he_mup(
        self, fan_in: int, fan_out: int, width: int
    ) -> float:
        """He (Kaiming) init std with µP correction.

        Standard He: σ = √(2/fan_in).
        µP correction for hidden layers: multiply by (base_width/width)^0.5
        when fan_in ≠ width (for explicitly tracking the multiplier).
        Here fan_in already encodes width so no extra factor is needed,
        but we expose the formula for clarity.
        """
        he_std = np.sqrt(2.0 / fan_in)
        # µP exponent for hidden: a = -0.5 ⟹ mult = (width/base)^{-0.5}
        ratio = width / self.base_width
        mup_mult = 1.0 / np.sqrt(ratio) if ratio > 0 else 1.0
        # When fan_in == width the He formula already accounts for width,
        # so mup_mult is applied only to the *excess* beyond the 1/√fan_in.
        # In practice the factors cancel: He_std ∝ 1/√width, mup ∝ 1/√width.
        return he_std * mup_mult

    def lecun_mup(self, fan_in: int, width: int) -> float:
        """LeCun init std with µP correction.

        Standard LeCun: σ = 1/√fan_in.
        """
        lecun_std = 1.0 / np.sqrt(fan_in)
        ratio = width / self.base_width
        mup_mult = 1.0 / np.sqrt(ratio) if ratio > 0 else 1.0
        return lecun_std * mup_mult

    def xavier_mup(
        self, fan_in: int, fan_out: int, width: int
    ) -> float:
        """Xavier (Glorot) init std with µP correction.

        Standard Xavier: σ = √(2 / (fan_in + fan_out)).
        """
        xavier_std = np.sqrt(2.0 / (fan_in + fan_out))
        ratio = width / self.base_width
        mup_mult = 1.0 / np.sqrt(ratio) if ratio > 0 else 1.0
        return xavier_std * mup_mult

    # ------------------------------------------------------------------
    def verify_activation_scale(
        self,
        network_weights: List[np.ndarray],
        input_data: np.ndarray,
    ) -> Dict[str, Any]:
        """Forward pass to verify activations stay O(1).

        Returns per-layer activation statistics.
        """
        h = input_data
        layer_stats = []
        n_layers = len(network_weights)

        for i, W in enumerate(network_weights):
            h = h @ W
            # Apply ReLU to all but the last layer
            if i < n_layers - 1:
                h = np.maximum(h, 0)
            mean_abs = float(np.mean(np.abs(h)))
            std = float(np.std(h))
            max_abs = float(np.max(np.abs(h)))
            layer_stats.append({
                "layer": i,
                "mean_abs": mean_abs,
                "std": std,
                "max_abs": max_abs,
                "is_order_one": 0.01 < mean_abs < 100.0,
            })

        all_ok = all(s["is_order_one"] for s in layer_stats)
        return {
            "layer_stats": layer_stats,
            "all_order_one": all_ok,
        }

    # ------------------------------------------------------------------
    def verify_gradient_scale(
        self,
        network_weights: List[np.ndarray],
        input_data: np.ndarray,
        target: np.ndarray,
    ) -> Dict[str, Any]:
        """Backward pass to verify gradient norms stay O(1).

        Simple MLP backward with MSE loss.
        """
        # Forward
        activations = [input_data]
        h = input_data
        n_layers = len(network_weights)
        for i, W in enumerate(network_weights):
            h = h @ W
            if i < n_layers - 1:
                h = np.maximum(h, 0)
            activations.append(h)

        # Loss gradient
        pred = activations[-1]
        delta = (pred - target) / target.shape[0]

        grad_stats = []
        for i in range(n_layers - 1, -1, -1):
            grad_W = activations[i].T @ delta
            grad_norm = float(np.linalg.norm(grad_W))
            grad_mean = float(np.mean(np.abs(grad_W)))
            grad_stats.append({
                "layer": i,
                "grad_norm": grad_norm,
                "grad_mean_abs": grad_mean,
                "is_order_one": 1e-6 < grad_mean < 1e6,
            })
            if i > 0:
                delta = (delta @ network_weights[i].T) * (activations[i] > 0)

        grad_stats.reverse()
        all_ok = all(s["is_order_one"] for s in grad_stats)
        return {
            "gradient_stats": grad_stats,
            "all_order_one": all_ok,
        }


# ===================================================================
# 4. WidthIndependenceVerifier
# ===================================================================

class WidthIndependenceVerifier:
    """Verify that µP achieves width-independent training dynamics.

    The key prediction of µP is that loss curves, gradient norms, and
    activation statistics should be approximately the same across widths.
    """

    def __init__(
        self,
        widths: Optional[List[int]] = None,
        n_trials: int = 5,
    ):
        self.widths = widths or [64, 128, 256, 512, 1024]
        self.n_trials = n_trials

    # ------------------------------------------------------------------
    def verify_loss_independence(
        self,
        loss_fn: Callable,
        data_fn: Callable,
        network_fn: Callable,
        widths: Optional[List[int]] = None,
        n_steps: int = 500,
    ) -> Dict[str, Any]:
        """Train at several widths, check that loss curves align.

        Parameters
        ----------
        loss_fn : callable(pred, y) → scalar
        data_fn : callable() → (x, y)
        network_fn : callable(width, seed) → (weights, forward_fn, update_fn)
        widths : list of widths (default: self.widths)
        n_steps : training steps

        Returns dict with per-width loss curves and independence score.
        """
        widths = widths or self.widths
        x, y = data_fn()
        all_curves = {}

        for width in widths:
            trial_curves = []
            for trial in range(self.n_trials):
                weights, forward, update = network_fn(width, seed=trial)
                losses = np.zeros(n_steps)
                for step in range(n_steps):
                    pred = forward(weights, x)
                    losses[step] = float(loss_fn(pred, y))
                    weights = update(weights, x, y)
                trial_curves.append(losses)
            all_curves[width] = np.mean(trial_curves, axis=0)

        score = self.width_independence_score(all_curves)
        return {
            "loss_curves": all_curves,
            "independence_score": score,
            "is_width_independent": score > 0.8,
            "widths": widths,
        }

    # ------------------------------------------------------------------
    def verify_gradient_independence(
        self,
        network_fn: Callable,
        data_fn: Callable,
        widths: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Check that gradient norms are O(1) and width-independent.

        network_fn(width, seed) → (weights, grad_fn)
        grad_fn(weights, x, y) → list of gradient arrays
        """
        widths = widths or self.widths
        x, y = data_fn()
        grad_norms: Dict[int, List[float]] = {}

        for width in widths:
            norms = []
            for trial in range(self.n_trials):
                weights, grad_fn = network_fn(width, seed=trial)
                grads = grad_fn(weights, x, y)
                total_norm = float(np.sqrt(sum(
                    np.sum(g ** 2) for g in grads
                )))
                norms.append(total_norm)
            grad_norms[width] = norms

        values = {w: np.mean(grad_norms[w]) for w in widths}
        test_result = self.statistical_test(values)
        return {
            "gradient_norms": grad_norms,
            "mean_norms": values,
            "test_result": test_result,
            "is_independent": test_result["is_independent"],
        }

    # ------------------------------------------------------------------
    def verify_activation_independence(
        self,
        network_fn: Callable,
        data_fn: Callable,
        widths: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Check that activation norms per coordinate are O(1).

        network_fn(width, seed) → (weights, activation_fn)
        activation_fn(weights, x) → list of activation arrays (per layer)
        """
        widths = widths or self.widths
        x, _ = data_fn()
        act_stats: Dict[int, Dict[str, float]] = {}

        for width in widths:
            means = []
            for trial in range(self.n_trials):
                weights, act_fn = network_fn(width, seed=trial)
                acts = act_fn(weights, x)
                # Mean absolute coordinate value at last hidden layer
                last_hidden = acts[-1] if acts else np.zeros(1)
                coord_mean = float(np.mean(np.abs(last_hidden)))
                means.append(coord_mean)
            act_stats[width] = {
                "mean_coord": float(np.mean(means)),
                "std_coord": float(np.std(means)),
            }

        values = {w: act_stats[w]["mean_coord"] for w in widths}
        test_result = self.statistical_test(values)
        return {
            "activation_stats": act_stats,
            "test_result": test_result,
            "is_independent": test_result["is_independent"],
        }

    # ------------------------------------------------------------------
    def verify_update_independence(
        self,
        network_fn: Callable,
        data_fn: Callable,
        widths: Optional[List[int]] = None,
        lr_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Check that per-coordinate update sizes are O(1).

        network_fn(width, seed) → (weights, grad_fn)
        lr_fn(width) → learning_rate
        """
        widths = widths or self.widths
        x, y = data_fn()
        update_stats: Dict[int, float] = {}

        default_lr_fn = lambda w: 0.01 * (64.0 / w)  # µP scaling
        lr_fn = lr_fn or default_lr_fn

        for width in widths:
            coord_updates = []
            for trial in range(self.n_trials):
                weights, grad_fn = network_fn(width, seed=trial)
                grads = grad_fn(weights, x, y)
                lr = lr_fn(width)
                # Mean absolute per-coordinate update
                for g in grads:
                    coord_update = float(np.mean(np.abs(lr * g)))
                    coord_updates.append(coord_update)
            update_stats[width] = float(np.mean(coord_updates))

        test_result = self.statistical_test(update_stats)
        return {
            "update_sizes": update_stats,
            "test_result": test_result,
            "is_independent": test_result["is_independent"],
        }

    # ------------------------------------------------------------------
    def width_independence_score(
        self, curves_at_widths: Dict[int, np.ndarray]
    ) -> float:
        """Quantify width independence from 0 (dependent) to 1 (independent).

        Uses the coefficient of variation of the final losses and the
        correlation between curves at different widths.
        """
        if len(curves_at_widths) < 2:
            return 1.0

        widths = sorted(curves_at_widths.keys())
        final_vals = np.array([curves_at_widths[w][-1] for w in widths])
        cv = np.std(final_vals) / (np.mean(final_vals) + 1e-12)

        # Pairwise correlation of curves
        curves_list = [curves_at_widths[w] for w in widths]
        min_len = min(len(c) for c in curves_list)
        curves_arr = np.array([c[:min_len] for c in curves_list])

        correlations = []
        for i in range(len(widths)):
            for j in range(i + 1, len(widths)):
                r, _ = stats.pearsonr(curves_arr[i], curves_arr[j])
                correlations.append(r if np.isfinite(r) else 0.0)

        mean_corr = float(np.mean(correlations)) if correlations else 0.0
        # Score combines low CV and high correlation
        cv_score = max(0.0, 1.0 - cv)
        score = 0.5 * cv_score + 0.5 * max(0.0, mean_corr)
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    def statistical_test(
        self,
        values_at_widths: Dict[int, float],
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """Statistical test for width independence.

        Fits log(value) = α·log(width) + β.  If α ≈ 0, values are
        width-independent.  Uses a t-test on the slope.
        """
        widths = sorted(values_at_widths.keys())
        vals = np.array([values_at_widths[w] for w in widths])

        # Avoid log of non-positive
        vals_safe = np.maximum(vals, 1e-30)
        log_w = np.log(np.array(widths, dtype=float))
        log_v = np.log(vals_safe)

        if len(widths) < 3:
            return {
                "slope": 0.0,
                "p_value": 1.0,
                "is_independent": True,
                "message": "Too few widths for reliable test",
            }

        slope, intercept, r_val, p_val, std_err = stats.linregress(log_w, log_v)
        alpha = 1.0 - confidence
        is_independent = p_val > alpha or abs(slope) < 0.1

        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_val ** 2),
            "p_value": float(p_val),
            "std_err": float(std_err),
            "is_independent": is_independent,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    def extrapolation_error(
        self,
        widths: List[int],
        values: List[float],
        target_width: int,
    ) -> Dict[str, float]:
        """Predict a value at target_width by extrapolating the trend.

        Under µP the extrapolated value should match the actual value
        because the quantity is width-independent.
        """
        log_w = np.log(np.array(widths, dtype=float))
        log_v = np.log(np.maximum(np.array(values), 1e-30))

        slope, intercept, _, _, _ = stats.linregress(log_w, log_v)
        log_pred = slope * np.log(target_width) + intercept
        predicted = np.exp(log_pred)

        # Constant extrapolation (µP assumption)
        constant_pred = float(np.mean(values))

        return {
            "target_width": target_width,
            "linear_extrapolation": float(predicted),
            "constant_extrapolation": constant_pred,
            "fitted_slope": float(slope),
        }


# ===================================================================
# 5. MuPViolationDetector
# ===================================================================

class MuPViolationDetector:
    """Detect violations of µP scaling assumptions.

    Checks whether init scales, learning rates, output magnitudes, and
    gradient norms conform to the theoretical predictions.
    """

    def __init__(self, tolerance: float = 0.1):
        self.tolerance = tolerance

    # ------------------------------------------------------------------
    def check_init_scale(
        self, weights: np.ndarray, expected_scale: float
    ) -> Dict[str, Any]:
        """Compare actual weight std to expected σ_init."""
        actual_std = float(np.std(weights))
        ratio = actual_std / (expected_scale + 1e-12)
        ok = abs(ratio - 1.0) < self.tolerance
        return {
            "actual_std": actual_std,
            "expected_std": expected_scale,
            "ratio": ratio,
            "violation": not ok,
            "severity": "none" if ok else (
                "minor" if abs(ratio - 1.0) < 2 * self.tolerance else "major"
            ),
        }

    def check_lr_scale(
        self, updates: np.ndarray, expected_scale: float
    ) -> Dict[str, Any]:
        """Check that update magnitudes match expected scale."""
        actual_scale = float(np.mean(np.abs(updates)))
        ratio = actual_scale / (expected_scale + 1e-12)
        ok = abs(ratio - 1.0) < self.tolerance
        return {
            "actual_update_scale": actual_scale,
            "expected_scale": expected_scale,
            "ratio": ratio,
            "violation": not ok,
            "severity": "none" if ok else (
                "minor" if abs(ratio - 1.0) < 2 * self.tolerance else "major"
            ),
        }

    def check_output_scale(
        self, outputs: np.ndarray, width: int
    ) -> Dict[str, Any]:
        """Output pre-activations should be O(1), independent of width."""
        mean_abs = float(np.mean(np.abs(outputs)))
        std = float(np.std(outputs))
        # Under µP, outputs should be O(1)
        ok = 0.01 < mean_abs < 100.0
        return {
            "mean_abs_output": mean_abs,
            "std_output": std,
            "width": width,
            "violation": not ok,
            "severity": "none" if ok else (
                "minor" if 0.001 < mean_abs < 1000.0 else "major"
            ),
        }

    def check_gradient_scale(
        self, gradients: np.ndarray, width: int
    ) -> Dict[str, Any]:
        """Per-coordinate gradient should be O(1)."""
        mean_abs = float(np.mean(np.abs(gradients)))
        ok = 1e-6 < mean_abs < 1e6
        return {
            "mean_abs_gradient": mean_abs,
            "width": width,
            "violation": not ok,
            "severity": "none" if ok else "major",
        }

    # ------------------------------------------------------------------
    def detect_blowup(
        self, activations_vs_width: Dict[int, float]
    ) -> Dict[str, Any]:
        """Detect activations that grow with width (blowup).

        Fits log(act) = α·log(width) + β; blowup if α > tolerance.
        """
        widths = sorted(activations_vs_width.keys())
        acts = np.array([activations_vs_width[w] for w in widths])
        log_w = np.log(np.array(widths, dtype=float))
        log_a = np.log(np.maximum(acts, 1e-30))

        if len(widths) < 2:
            return {"blowup": False, "slope": 0.0}

        slope, _, _, _, _ = stats.linregress(log_w, log_a)
        blowup = slope > self.tolerance
        return {
            "blowup": blowup,
            "slope": float(slope),
            "threshold": self.tolerance,
            "widths": widths,
            "activations": acts.tolist(),
        }

    def detect_vanishing(
        self, activations_vs_width: Dict[int, float]
    ) -> Dict[str, Any]:
        """Detect activations that shrink with width (vanishing)."""
        widths = sorted(activations_vs_width.keys())
        acts = np.array([activations_vs_width[w] for w in widths])
        log_w = np.log(np.array(widths, dtype=float))
        log_a = np.log(np.maximum(acts, 1e-30))

        if len(widths) < 2:
            return {"vanishing": False, "slope": 0.0}

        slope, _, _, _, _ = stats.linregress(log_w, log_a)
        vanishing = slope < -self.tolerance
        return {
            "vanishing": vanishing,
            "slope": float(slope),
            "threshold": -self.tolerance,
            "widths": widths,
            "activations": acts.tolist(),
        }

    # ------------------------------------------------------------------
    def full_diagnostic(
        self,
        network_fn: Callable,
        data_fn: Callable,
        widths: List[int],
    ) -> Dict[str, Any]:
        """Run full µP diagnostic across widths.

        network_fn(width, seed) → dict with keys:
            weights: list of arrays
            forward: callable(x) → (output, activations_per_layer)
            backward: callable(x, y) → gradients_per_layer
            layer_types: list of str
        data_fn() → (x, y)
        """
        x, y = data_fn()
        base_width = widths[0]

        per_width_results = {}
        for width in widths:
            net = network_fn(width, seed=0)
            weights = net["weights"]
            output, activations = net["forward"](x)
            gradients = net["backward"](x, y)
            layer_types = net.get("layer_types", ["hidden"] * len(weights))

            layer_results = []
            computer = MuPScalingComputer(base_width)

            for i, (W, ltype) in enumerate(zip(weights, layer_types)):
                fan_in, fan_out = W.shape[0], W.shape[1] if W.ndim > 1 else 1
                expected_sigma = computer.init_scale(ltype, fan_in, fan_out)
                init_check = self.check_init_scale(W, expected_sigma)
                output_check = self.check_output_scale(
                    activations[i] if i < len(activations) else output, width
                )
                grad_check = self.check_gradient_scale(
                    gradients[i] if i < len(gradients) else np.zeros(1), width
                )
                layer_results.append({
                    "layer": i,
                    "type": ltype,
                    "init_check": init_check,
                    "output_check": output_check,
                    "gradient_check": grad_check,
                })

            per_width_results[width] = layer_results

        # Aggregate: check for blowup/vanishing across widths
        n_layers = len(per_width_results[widths[0]])
        blowup_checks = []
        vanishing_checks = []
        for layer_idx in range(n_layers):
            act_vs_w = {}
            for w in widths:
                r = per_width_results[w][layer_idx]
                act_vs_w[w] = r["output_check"]["mean_abs_output"]
            blowup_checks.append(self.detect_blowup(act_vs_w))
            vanishing_checks.append(self.detect_vanishing(act_vs_w))

        n_violations = sum(
            1 for w in widths for lr in per_width_results[w]
            if lr["init_check"]["violation"]
            or lr["output_check"]["violation"]
            or lr["gradient_check"]["violation"]
        )

        return {
            "per_width": per_width_results,
            "blowup": blowup_checks,
            "vanishing": vanishing_checks,
            "total_violations": n_violations,
            "widths": widths,
        }

    # ------------------------------------------------------------------
    def violation_report(self, diagnostics: Dict[str, Any]) -> str:
        """Generate a human-readable µP violation report."""
        lines = ["=" * 60, "µP Violation Report", "=" * 60, ""]

        widths = diagnostics.get("widths", [])
        lines.append(f"Widths tested: {widths}")
        lines.append(
            f"Total violations: {diagnostics.get('total_violations', 0)}"
        )
        lines.append("")

        # Per-layer blowup/vanishing
        for i, (b, v) in enumerate(zip(
            diagnostics.get("blowup", []),
            diagnostics.get("vanishing", []),
        )):
            status = "OK"
            if b.get("blowup"):
                status = f"BLOWUP (slope={b['slope']:.3f})"
            elif v.get("vanishing"):
                status = f"VANISHING (slope={v['slope']:.3f})"
            lines.append(f"  Layer {i}: {status}")

        lines.append("")

        # Per-width details
        per_width = diagnostics.get("per_width", {})
        for width in widths:
            lines.append(f"--- Width {width} ---")
            for lr in per_width.get(width, []):
                init_v = "VIOLATION" if lr["init_check"]["violation"] else "ok"
                out_v = "VIOLATION" if lr["output_check"]["violation"] else "ok"
                grad_v = "VIOLATION" if lr["gradient_check"]["violation"] else "ok"
                lines.append(
                    f"  Layer {lr['layer']} ({lr['type']}): "
                    f"init={init_v}, output={out_v}, grad={grad_v}"
                )
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


# ===================================================================
# 6. CoordinateCheck
# ===================================================================

class CoordinateCheck:
    """µP coordinate check implementation.

    The coordinate check (Yang & Hu 2021, §6) verifies that:
    - Activations are O(1) in every coordinate
    - Per-coordinate gradient is O(1)
    - Per-coordinate update is O(1) for hidden layers, O(1/n) for output

    This is the standard diagnostic for verifying µP correctness.
    """

    def __init__(
        self,
        base_width: int = 64,
        widths: Optional[List[int]] = None,
    ):
        self.base_width = base_width
        self.widths = widths or [64, 128, 256, 512, 1024, 2048]

    # ------------------------------------------------------------------
    def coordinate_check(
        self,
        network_fn: Callable,
        data_fn: Callable,
        lr_fn: Callable,
        n_steps: int = 5,
    ) -> Dict[str, Any]:
        """Full coordinate check across widths.

        Parameters
        ----------
        network_fn : callable(width, seed) → dict with keys:
            weights, forward, backward, layer_types
        data_fn : callable() → (x, y)
        lr_fn : callable(width) → per-layer learning rates dict
        n_steps : number of SGD steps to take

        Returns per-layer, per-width, per-step measurements.
        """
        act_results = self.check_activations(network_fn, data_fn, self.widths)
        grad_results = self.check_gradients(network_fn, data_fn, self.widths)
        update_results = self.check_updates(
            network_fn, data_fn, lr_fn, self.widths, n_steps
        )

        results = {
            "activations": act_results,
            "gradients": grad_results,
            "updates": update_results,
            "widths": self.widths,
            "n_steps": n_steps,
        }

        # Add deviation analysis
        plot = self.plot_data(results)
        results["plot_data"] = plot
        return results

    # ------------------------------------------------------------------
    def check_activations(
        self,
        network_fn: Callable,
        data_fn: Callable,
        widths: List[int],
    ) -> Dict[str, Any]:
        """Measure per-coordinate activation magnitude at each width.

        For each width, we initialise the network, do a forward pass,
        and record |h_l|_coord = mean(|h_l[:, j]|) for each layer l.
        """
        x, _ = data_fn()
        per_width: Dict[int, Dict[int, float]] = {}

        for width in widths:
            net = network_fn(width, seed=0)
            output, activations = net["forward"](x)
            layer_means = {}
            for layer_idx, act in enumerate(activations):
                # Mean absolute value per coordinate, then average
                coord_mean = float(np.mean(np.abs(act)))
                layer_means[layer_idx] = coord_mean
            # Also add output
            layer_means[len(activations)] = float(np.mean(np.abs(output)))
            per_width[width] = layer_means

        # Compute scaling exponents
        n_layers = max(len(v) for v in per_width.values())
        scaling_exponents = {}
        for layer_idx in range(n_layers):
            vals = {w: per_width[w].get(layer_idx, 0.0) for w in widths}
            if all(v > 0 for v in vals.values()) and len(widths) >= 2:
                log_w = np.log(np.array(widths, dtype=float))
                log_v = np.log(np.array([vals[w] for w in widths]))
                slope, _, _, _, _ = stats.linregress(log_w, log_v)
                scaling_exponents[layer_idx] = float(slope)
            else:
                scaling_exponents[layer_idx] = 0.0

        return {
            "per_width": per_width,
            "scaling_exponents": scaling_exponents,
            "expected_exponent": 0.0,  # O(1) ⟹ slope = 0
        }

    # ------------------------------------------------------------------
    def check_updates(
        self,
        network_fn: Callable,
        data_fn: Callable,
        lr_fn: Callable,
        widths: List[int],
        n_steps: int = 1,
    ) -> Dict[str, Any]:
        """Measure per-coordinate update size at each width.

        Takes n_steps of SGD and records the coordinate-wise update
        magnitude for each layer.
        """
        x, y = data_fn()
        per_width: Dict[int, Dict[int, List[float]]] = {}

        for width in widths:
            net = network_fn(width, seed=0)
            weights = [w.copy() for w in net["weights"]]
            lrs = lr_fn(width)
            layer_types = net.get("layer_types", ["hidden"] * len(weights))
            step_records: Dict[int, List[float]] = {
                i: [] for i in range(len(weights))
            }

            for step in range(n_steps):
                grads = net["backward"](x, y)
                for i, (W, g) in enumerate(zip(weights, grads)):
                    lr_i = lrs.get(layer_types[i], lrs.get("default", 0.01))
                    update = lr_i * g
                    coord_update = float(np.mean(np.abs(update)))
                    step_records[i].append(coord_update)
                    weights[i] = W - update

            per_width[width] = {
                i: vals for i, vals in step_records.items()
            }

        # Fit scaling exponents per layer
        n_layers_check = len(per_width[widths[0]])
        scaling_exponents = {}
        for layer_idx in range(n_layers_check):
            mean_updates = {
                w: float(np.mean(per_width[w][layer_idx])) for w in widths
            }
            if all(v > 0 for v in mean_updates.values()) and len(widths) >= 2:
                log_w = np.log(np.array(widths, dtype=float))
                log_v = np.log(np.array([mean_updates[w] for w in widths]))
                slope, _, _, _, _ = stats.linregress(log_w, log_v)
                scaling_exponents[layer_idx] = float(slope)
            else:
                scaling_exponents[layer_idx] = 0.0

        return {
            "per_width": per_width,
            "scaling_exponents": scaling_exponents,
            "expected_exponent_hidden": 0.0,
            "expected_exponent_output": -1.0,
        }

    # ------------------------------------------------------------------
    def check_gradients(
        self,
        network_fn: Callable,
        data_fn: Callable,
        widths: List[int],
    ) -> Dict[str, Any]:
        """Measure per-coordinate gradient magnitude at each width."""
        x, y = data_fn()
        per_width: Dict[int, Dict[int, float]] = {}

        for width in widths:
            net = network_fn(width, seed=0)
            grads = net["backward"](x, y)
            layer_means = {}
            for layer_idx, g in enumerate(grads):
                coord_mean = float(np.mean(np.abs(g)))
                layer_means[layer_idx] = coord_mean
            per_width[width] = layer_means

        n_layers_check = max(len(v) for v in per_width.values())
        scaling_exponents = {}
        for layer_idx in range(n_layers_check):
            vals = {w: per_width[w].get(layer_idx, 0.0) for w in widths}
            if all(v > 0 for v in vals.values()) and len(widths) >= 2:
                log_w = np.log(np.array(widths, dtype=float))
                log_v = np.log(np.array([vals[w] for w in widths]))
                slope, _, _, _, _ = stats.linregress(log_w, log_v)
                scaling_exponents[layer_idx] = float(slope)
            else:
                scaling_exponents[layer_idx] = 0.0

        return {
            "per_width": per_width,
            "scaling_exponents": scaling_exponents,
            "expected_exponent": 0.0,
        }

    # ------------------------------------------------------------------
    def plot_data(
        self, check_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare plot data for the coordinate check.

        Returns data suitable for a log-log plot of each quantity vs width.
        """
        widths = check_results.get("widths", self.widths)
        log_widths = np.log2(np.array(widths, dtype=float))
        plot = {"log_widths": log_widths.tolist(), "widths": widths}

        for quantity in ("activations", "gradients", "updates"):
            qdata = check_results.get(quantity, {})
            per_width = qdata.get("per_width", {})
            exponents = qdata.get("scaling_exponents", {})

            if not per_width:
                continue

            n_layers = max(len(v) for v in per_width.values())
            layer_traces = {}
            for layer_idx in range(n_layers):
                values = []
                for w in widths:
                    pw = per_width.get(w, {})
                    val = pw.get(layer_idx, 0.0)
                    if isinstance(val, list):
                        val = float(np.mean(val)) if val else 0.0
                    values.append(val)
                log_vals = np.log2(np.maximum(np.array(values), 1e-30))
                layer_traces[layer_idx] = {
                    "values": values,
                    "log_values": log_vals.tolist(),
                    "slope": exponents.get(layer_idx, 0.0),
                }

            plot[quantity] = layer_traces

        return plot

    # ------------------------------------------------------------------
    def expected_scaling(
        self, layer_type: str, quantity: str
    ) -> float:
        """Return the expected scaling exponent for a given layer/quantity.

        quantity ∈ {"activation", "gradient", "update", "init"}
        """
        a, b, _ = _MUP_EXPONENTS.get(layer_type, _MUP_EXPONENTS["hidden"])

        if quantity == "activation":
            return 0.0  # O(1) activations
        elif quantity == "gradient":
            if layer_type == "output":
                return 0.0
            return 0.0  # per-coordinate gradient O(1) under µP
        elif quantity == "update":
            if layer_type == "input":
                return 0.0  # input updates are O(1)
            elif layer_type == "output":
                return -1.0  # output updates scale as 1/n
            else:
                return 0.0  # hidden updates O(1) per coordinate under µP
        elif quantity == "init":
            return a
        else:
            raise ValueError(f"Unknown quantity {quantity!r}")

    # ------------------------------------------------------------------
    def deviation_from_expected(
        self,
        actual_scaling: Dict[int, float],
        expected_scaling_vals: Dict[int, float],
    ) -> Dict[str, Any]:
        """Quantify how far actual scaling exponents deviate from expected.

        Returns per-layer deviations and an overall score.
        """
        deviations = {}
        for layer_idx in actual_scaling:
            actual = actual_scaling[layer_idx]
            expected = expected_scaling_vals.get(layer_idx, 0.0)
            deviations[layer_idx] = {
                "actual": actual,
                "expected": expected,
                "deviation": actual - expected,
                "abs_deviation": abs(actual - expected),
            }

        abs_devs = [d["abs_deviation"] for d in deviations.values()]
        mean_dev = float(np.mean(abs_devs)) if abs_devs else 0.0
        max_dev = float(np.max(abs_devs)) if abs_devs else 0.0

        return {
            "per_layer": deviations,
            "mean_deviation": mean_dev,
            "max_deviation": max_dev,
            "passes": max_dev < 0.2,
        }
