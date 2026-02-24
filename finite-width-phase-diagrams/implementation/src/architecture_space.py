"""
Architecture design space exploration for neural networks.

Implements architecture encoding, random sampling with constraints,
mutation, crossover, performance prediction, clustering,
design rules derivation, and evolutionary architecture search.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.special import softmax
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
import warnings


@dataclass
class Architecture:
    """Representation of a neural network architecture."""
    depth: int = 5
    widths: List[int] = field(default_factory=lambda: [256] * 5)
    activations: List[str] = field(default_factory=lambda: ["relu"] * 5)
    skip_connections: List[bool] = field(default_factory=lambda: [False] * 5)
    has_batchnorm: List[bool] = field(default_factory=lambda: [False] * 5)
    sigma_w: float = 1.0
    sigma_b: float = 0.0
    input_dim: int = 784
    output_dim: int = 10
    dropout_rates: List[float] = field(default_factory=lambda: [0.0] * 5)

    @property
    def n_params(self) -> int:
        """Compute total number of parameters."""
        total = 0
        prev_dim = self.input_dim
        for i in range(self.depth):
            w = self.widths[i] if i < len(self.widths) else self.widths[-1]
            total += prev_dim * w + w  # weights + biases
            if i < len(self.has_batchnorm) and self.has_batchnorm[i]:
                total += 2 * w  # gamma and beta
            prev_dim = w
        total += prev_dim * self.output_dim + self.output_dim
        return total

    @property
    def flops(self) -> int:
        """Estimate FLOPs for one forward pass."""
        total = 0
        prev_dim = self.input_dim
        for i in range(self.depth):
            w = self.widths[i] if i < len(self.widths) else self.widths[-1]
            total += 2 * prev_dim * w  # matmul
            total += w  # activation
            prev_dim = w
        total += 2 * prev_dim * self.output_dim
        return total

    def to_vector(self) -> np.ndarray:
        """Encode architecture as a numerical vector."""
        activation_map = {"relu": 0, "tanh": 1, "sigmoid": 2, "gelu": 3,
                          "silu": 4, "elu": 5, "selu": 6, "leaky_relu": 7,
                          "mish": 8, "softplus": 9, "sin": 10}

        max_depth = 20
        vec = np.zeros(max_depth * 4 + 5)
        vec[0] = self.depth
        vec[1] = self.input_dim
        vec[2] = self.output_dim
        vec[3] = self.sigma_w
        vec[4] = self.sigma_b

        for i in range(min(self.depth, max_depth)):
            offset = 5 + i * 4
            vec[offset] = self.widths[i] if i < len(self.widths) else 256
            act_name = self.activations[i] if i < len(self.activations) else "relu"
            vec[offset + 1] = activation_map.get(act_name, 0)
            vec[offset + 2] = float(self.skip_connections[i]) if i < len(self.skip_connections) else 0
            vec[offset + 3] = float(self.has_batchnorm[i]) if i < len(self.has_batchnorm) else 0

        return vec

    @staticmethod
    def from_vector(vec: np.ndarray) -> 'Architecture':
        """Decode architecture from numerical vector."""
        activation_map_rev = {0: "relu", 1: "tanh", 2: "sigmoid", 3: "gelu",
                              4: "silu", 5: "elu", 6: "selu", 7: "leaky_relu",
                              8: "mish", 9: "softplus", 10: "sin"}

        depth = max(1, int(vec[0]))
        input_dim = max(1, int(vec[1]))
        output_dim = max(1, int(vec[2]))
        sigma_w = float(vec[3])
        sigma_b = float(vec[4])

        widths = []
        activations = []
        skip_connections = []
        has_batchnorm = []

        for i in range(depth):
            offset = 5 + i * 4
            if offset + 3 < len(vec):
                widths.append(max(1, int(vec[offset])))
                activations.append(activation_map_rev.get(int(vec[offset + 1]) % 11, "relu"))
                skip_connections.append(bool(vec[offset + 2] > 0.5))
                has_batchnorm.append(bool(vec[offset + 3] > 0.5))
            else:
                widths.append(256)
                activations.append("relu")
                skip_connections.append(False)
                has_batchnorm.append(False)

        return Architecture(
            depth=depth, widths=widths, activations=activations,
            skip_connections=skip_connections, has_batchnorm=has_batchnorm,
            sigma_w=sigma_w, sigma_b=sigma_b,
            input_dim=input_dim, output_dim=output_dim,
        )


@dataclass
class ArchitectureConstraints:
    """Constraints on architecture search space."""
    max_params: int = 10_000_000
    max_flops: int = 100_000_000
    min_depth: int = 2
    max_depth: int = 20
    min_width: int = 16
    max_width: int = 2048
    allowed_activations: List[str] = field(default_factory=lambda: [
        "relu", "gelu", "silu", "tanh", "elu", "selu"
    ])
    allow_skip_connections: bool = True
    allow_batchnorm: bool = True
    max_dropout: float = 0.5


@dataclass
class ArchitectureMetrics:
    """Metrics for evaluating an architecture."""
    n_params: int = 0
    flops: int = 0
    chi_1: float = 0.0
    depth_scale: float = 0.0
    max_trainable_depth: int = 0
    phase: str = "unknown"
    predicted_train_loss: float = 0.0
    predicted_test_loss: float = 0.0
    trainability_score: float = 0.0
    expressivity_score: float = 0.0
    overall_score: float = 0.0


class ArchitectureSampler:
    """Sample random architectures with constraints."""

    def __init__(self, constraints: Optional[ArchitectureConstraints] = None):
        self.constraints = constraints or ArchitectureConstraints()

    def sample(self, n: int = 1) -> List[Architecture]:
        """Sample n random architectures satisfying constraints."""
        architectures = []
        max_attempts = n * 20

        for _ in range(max_attempts):
            if len(architectures) >= n:
                break
            arch = self._sample_one()
            if self._satisfies_constraints(arch):
                architectures.append(arch)

        return architectures

    def _sample_one(self) -> Architecture:
        """Sample one random architecture."""
        c = self.constraints
        depth = np.random.randint(c.min_depth, c.max_depth + 1)

        width_pattern = np.random.choice(["constant", "funnel", "hourglass", "random"])
        if width_pattern == "constant":
            base_width = 2 ** np.random.randint(
                int(np.log2(max(c.min_width, 1))),
                int(np.log2(c.max_width)) + 1
            )
            widths = [base_width] * depth
        elif width_pattern == "funnel":
            start = 2 ** np.random.randint(int(np.log2(max(c.min_width * 2, 2))),
                                            int(np.log2(c.max_width)) + 1)
            end = max(c.min_width, start // (2 ** np.random.randint(1, 4)))
            widths = np.linspace(start, end, depth).astype(int).tolist()
        elif width_pattern == "hourglass":
            edge = 2 ** np.random.randint(int(np.log2(max(c.min_width, 1))),
                                           int(np.log2(c.max_width)))
            middle = max(c.min_width, edge // 2)
            half = depth // 2
            widths = (list(np.linspace(edge, middle, half).astype(int)) +
                      list(np.linspace(middle, edge, depth - half).astype(int)))
        else:
            widths = [2 ** np.random.randint(
                int(np.log2(max(c.min_width, 1))),
                int(np.log2(c.max_width)) + 1
            ) for _ in range(depth)]

        widths = [max(c.min_width, min(c.max_width, w)) for w in widths]
        activations = [np.random.choice(c.allowed_activations) for _ in range(depth)]

        skip_connections = [False] * depth
        if c.allow_skip_connections:
            for i in range(depth):
                skip_connections[i] = np.random.rand() < 0.3

        has_batchnorm = [False] * depth
        if c.allow_batchnorm:
            for i in range(depth):
                has_batchnorm[i] = np.random.rand() < 0.3

        sigma_w = np.random.choice([1.0, np.sqrt(2.0), np.sqrt(2.0 / widths[0])])

        return Architecture(
            depth=depth, widths=widths, activations=activations,
            skip_connections=skip_connections, has_batchnorm=has_batchnorm,
            sigma_w=sigma_w, sigma_b=0.0,
        )

    def _satisfies_constraints(self, arch: Architecture) -> bool:
        """Check if architecture satisfies all constraints."""
        if arch.n_params > self.constraints.max_params:
            return False
        if arch.flops > self.constraints.max_flops:
            return False
        if arch.depth < self.constraints.min_depth or arch.depth > self.constraints.max_depth:
            return False
        for w in arch.widths:
            if w < self.constraints.min_width or w > self.constraints.max_width:
                return False
        return True


class ArchitectureMutator:
    """Mutate architectures."""

    def __init__(self, constraints: Optional[ArchitectureConstraints] = None):
        self.constraints = constraints or ArchitectureConstraints()

    def mutate(self, arch: Architecture) -> Architecture:
        """Apply one random mutation to architecture."""
        mutation_type = np.random.choice([
            "add_layer", "remove_layer", "change_width",
            "change_activation", "toggle_skip", "toggle_batchnorm",
            "change_sigma_w"
        ])

        new_arch = Architecture(
            depth=arch.depth,
            widths=arch.widths.copy(),
            activations=arch.activations.copy(),
            skip_connections=arch.skip_connections.copy(),
            has_batchnorm=arch.has_batchnorm.copy(),
            sigma_w=arch.sigma_w,
            sigma_b=arch.sigma_b,
            input_dim=arch.input_dim,
            output_dim=arch.output_dim,
        )

        if mutation_type == "add_layer" and new_arch.depth < self.constraints.max_depth:
            pos = np.random.randint(0, new_arch.depth + 1)
            width = np.random.choice(new_arch.widths) if new_arch.widths else 256
            new_arch.widths.insert(pos, width)
            new_arch.activations.insert(pos, np.random.choice(self.constraints.allowed_activations))
            new_arch.skip_connections.insert(pos, np.random.rand() < 0.3)
            new_arch.has_batchnorm.insert(pos, np.random.rand() < 0.3)
            new_arch.depth += 1

        elif mutation_type == "remove_layer" and new_arch.depth > self.constraints.min_depth:
            pos = np.random.randint(0, new_arch.depth)
            new_arch.widths.pop(pos)
            new_arch.activations.pop(pos)
            new_arch.skip_connections.pop(pos)
            new_arch.has_batchnorm.pop(pos)
            new_arch.depth -= 1

        elif mutation_type == "change_width" and new_arch.depth > 0:
            pos = np.random.randint(0, new_arch.depth)
            factor = np.random.choice([0.5, 0.75, 1.5, 2.0])
            new_width = max(self.constraints.min_width,
                           min(self.constraints.max_width,
                               int(new_arch.widths[pos] * factor)))
            new_arch.widths[pos] = new_width

        elif mutation_type == "change_activation" and new_arch.depth > 0:
            pos = np.random.randint(0, new_arch.depth)
            new_arch.activations[pos] = np.random.choice(self.constraints.allowed_activations)

        elif mutation_type == "toggle_skip" and new_arch.depth > 0:
            pos = np.random.randint(0, new_arch.depth)
            new_arch.skip_connections[pos] = not new_arch.skip_connections[pos]

        elif mutation_type == "toggle_batchnorm" and new_arch.depth > 0:
            pos = np.random.randint(0, new_arch.depth)
            new_arch.has_batchnorm[pos] = not new_arch.has_batchnorm[pos]

        elif mutation_type == "change_sigma_w":
            new_arch.sigma_w *= np.random.choice([0.8, 0.9, 1.1, 1.2])

        return new_arch

    def crossover(self, parent1: Architecture, parent2: Architecture) -> Architecture:
        """Combine aspects of two architectures."""
        depth = np.random.choice([parent1.depth, parent2.depth])
        widths = []
        activations = []
        skip_connections = []
        has_batchnorm = []

        for i in range(depth):
            if np.random.rand() < 0.5:
                source = parent1
            else:
                source = parent2

            idx = min(i, source.depth - 1)
            widths.append(source.widths[idx] if idx < len(source.widths) else 256)
            activations.append(source.activations[idx] if idx < len(source.activations) else "relu")
            skip_connections.append(source.skip_connections[idx]
                                    if idx < len(source.skip_connections) else False)
            has_batchnorm.append(source.has_batchnorm[idx]
                                 if idx < len(source.has_batchnorm) else False)

        sigma_w = parent1.sigma_w if np.random.rand() < 0.5 else parent2.sigma_w

        return Architecture(
            depth=depth, widths=widths, activations=activations,
            skip_connections=skip_connections, has_batchnorm=has_batchnorm,
            sigma_w=sigma_w, sigma_b=0.0,
            input_dim=parent1.input_dim, output_dim=parent1.output_dim,
        )


class PerformancePredictor:
    """Predict architecture performance from features."""

    def __init__(self, n_samples: int = 30000):
        self.n_samples = n_samples

    def extract_features(self, arch: Architecture) -> np.ndarray:
        """Extract features from architecture for prediction."""
        features = [
            arch.depth,
            np.mean(arch.widths),
            np.std(arch.widths) if len(arch.widths) > 1 else 0,
            np.max(arch.widths),
            np.min(arch.widths),
            arch.n_params,
            arch.flops,
            arch.sigma_w,
            np.sum(arch.skip_connections),
            np.sum(arch.has_batchnorm),
            len(set(arch.activations)),
            arch.n_params / (arch.depth * np.mean(arch.widths) + 1),
        ]
        return np.array(features)

    def predict_trainability(self, arch: Architecture) -> float:
        """Predict trainability score based on mean field theory."""
        z = np.random.randn(self.n_samples)

        activation_chi = {
            "relu": arch.sigma_w ** 2 / 2.0,
            "tanh": arch.sigma_w ** 2 * 0.6,
            "sigmoid": arch.sigma_w ** 2 * 0.25,
            "gelu": arch.sigma_w ** 2 * 0.45,
            "silu": arch.sigma_w ** 2 * 0.4,
            "elu": arch.sigma_w ** 2 * 0.55,
            "selu": arch.sigma_w ** 2 * 0.5,
            "leaky_relu": arch.sigma_w ** 2 * (0.5 + 0.5 * 0.01 ** 2),
        }

        chi_values = []
        for act in arch.activations:
            chi = activation_chi.get(act, arch.sigma_w ** 2 * 0.5)
            chi_values.append(chi)

        mean_chi = np.mean(chi_values)
        criticality = 1.0 / (abs(mean_chi - 1.0) + 0.01)

        if abs(mean_chi - 1.0) < 0.01:
            depth_penalty = 0.0
        elif mean_chi < 1.0:
            depth_scale = -1.0 / np.log(abs(mean_chi) + 1e-12)
            depth_penalty = max(0, 1.0 - depth_scale / arch.depth)
        else:
            depth_scale = 1.0 / np.log(mean_chi + 1e-12)
            depth_penalty = max(0, 1.0 - depth_scale / arch.depth)

        skip_bonus = 0.1 * np.sum(arch.skip_connections) / (arch.depth + 1)
        bn_bonus = 0.1 * np.sum(arch.has_batchnorm) / (arch.depth + 1)

        score = criticality * (1 - depth_penalty) + skip_bonus + bn_bonus
        return float(np.clip(score, 0, 10))

    def predict_expressivity(self, arch: Architecture) -> float:
        """Predict expressivity from architecture features."""
        depth_factor = np.log2(arch.depth + 1)
        width_factor = np.log2(np.mean(arch.widths))
        param_factor = np.log10(arch.n_params + 1)

        n_nonlinear = sum(1 for a in arch.activations if a not in ["linear", "identity"])
        nonlinearity_factor = n_nonlinear / arch.depth

        score = (depth_factor * width_factor * nonlinearity_factor * 0.1 +
                 param_factor * 0.2)
        return float(np.clip(score, 0, 10))

    def predict(self, arch: Architecture) -> ArchitectureMetrics:
        """Predict all metrics for architecture."""
        metrics = ArchitectureMetrics()
        metrics.n_params = arch.n_params
        metrics.flops = arch.flops
        metrics.trainability_score = self.predict_trainability(arch)
        metrics.expressivity_score = self.predict_expressivity(arch)
        metrics.overall_score = 0.5 * metrics.trainability_score + 0.5 * metrics.expressivity_score

        activation_chi = {
            "relu": arch.sigma_w ** 2 / 2.0,
            "tanh": arch.sigma_w ** 2 * 0.6,
            "gelu": arch.sigma_w ** 2 * 0.45,
        }
        act = arch.activations[0] if arch.activations else "relu"
        metrics.chi_1 = activation_chi.get(act, arch.sigma_w ** 2 * 0.5)

        if metrics.chi_1 < 0.95:
            metrics.phase = "ordered"
        elif metrics.chi_1 > 1.05:
            metrics.phase = "chaotic"
        else:
            metrics.phase = "critical"

        if abs(metrics.chi_1) < 1 and metrics.chi_1 > 0:
            metrics.depth_scale = -1.0 / np.log(abs(metrics.chi_1) + 1e-12)
        else:
            metrics.depth_scale = float(arch.depth)
        metrics.max_trainable_depth = int(min(10000, metrics.depth_scale * 10))

        metrics.predicted_train_loss = 1.0 / (metrics.overall_score + 0.1)
        metrics.predicted_test_loss = metrics.predicted_train_loss * (
            1 + arch.n_params / (1000 * arch.depth * np.mean(arch.widths)))

        return metrics


class ArchitectureClusterer:
    """Cluster similar architectures and analyze clusters."""

    def __init__(self):
        pass

    def cluster(self, architectures: List[Architecture],
                n_clusters: int = 5) -> Dict[str, Any]:
        """Cluster architectures based on features."""
        if len(architectures) < 2:
            return {"labels": [0] * len(architectures), "n_clusters": 1}

        vectors = np.array([arch.to_vector() for arch in architectures])
        vectors_norm = (vectors - vectors.mean(axis=0)) / (vectors.std(axis=0) + 1e-12)

        distances = pdist(vectors_norm, metric="euclidean")
        Z = linkage(distances, method="ward")
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")

        cluster_info = {}
        for c in range(1, n_clusters + 1):
            mask = labels == c
            cluster_archs = [arch for arch, m in zip(architectures, mask) if m]
            if cluster_archs:
                cluster_info[f"cluster_{c}"] = {
                    "size": len(cluster_archs),
                    "avg_depth": float(np.mean([a.depth for a in cluster_archs])),
                    "avg_width": float(np.mean([np.mean(a.widths) for a in cluster_archs])),
                    "avg_params": float(np.mean([a.n_params for a in cluster_archs])),
                    "common_activations": self._most_common([
                        a.activations[0] for a in cluster_archs if a.activations
                    ]),
                }

        return {
            "labels": labels.tolist(),
            "n_clusters": n_clusters,
            "cluster_info": cluster_info,
        }

    def _most_common(self, items: List[str]) -> str:
        """Find most common item in list."""
        if not items:
            return "unknown"
        counts = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        return max(counts, key=counts.get)


class DesignRuleDeriver:
    """Derive design rules from analyzed architecture space."""

    def __init__(self):
        self.predictor = PerformancePredictor()

    def derive_rules(self, architectures: List[Architecture],
                     metrics: List[ArchitectureMetrics]) -> List[Dict[str, Any]]:
        """Derive heuristic design rules from architecture-metric pairs."""
        rules = []

        depths = [a.depth for a in architectures]
        scores = [m.overall_score for m in metrics]
        if len(depths) > 5:
            corr = np.corrcoef(depths, scores)[0, 1]
            if abs(corr) > 0.3:
                optimal_depth_idx = np.argmax(scores)
                rules.append({
                    "rule": f"Optimal depth around {depths[optimal_depth_idx]}",
                    "correlation": float(corr),
                    "confidence": float(abs(corr)),
                })

        widths = [np.mean(a.widths) for a in architectures]
        if len(widths) > 5:
            corr = np.corrcoef(widths, scores)[0, 1]
            if abs(corr) > 0.3:
                rules.append({
                    "rule": f"Width {'positively' if corr > 0 else 'negatively'} correlated with performance",
                    "correlation": float(corr),
                    "confidence": float(abs(corr)),
                })

        activation_scores = {}
        for arch, metric in zip(architectures, metrics):
            act = arch.activations[0] if arch.activations else "relu"
            if act not in activation_scores:
                activation_scores[act] = []
            activation_scores[act].append(metric.overall_score)

        if activation_scores:
            best_act = max(activation_scores,
                          key=lambda k: np.mean(activation_scores[k]))
            rules.append({
                "rule": f"Best activation: {best_act}",
                "mean_score": float(np.mean(activation_scores[best_act])),
                "confidence": float(len(activation_scores[best_act]) / len(architectures)),
            })

        skip_archs = [(a, m) for a, m in zip(architectures, metrics)
                      if any(a.skip_connections)]
        no_skip_archs = [(a, m) for a, m in zip(architectures, metrics)
                         if not any(a.skip_connections)]

        if skip_archs and no_skip_archs:
            skip_score = np.mean([m.overall_score for _, m in skip_archs])
            no_skip_score = np.mean([m.overall_score for _, m in no_skip_archs])
            benefit = skip_score - no_skip_score
            rules.append({
                "rule": f"Skip connections {'help' if benefit > 0 else 'hurt'} ({benefit:+.3f})",
                "skip_score": float(skip_score),
                "no_skip_score": float(no_skip_score),
                "confidence": float(min(len(skip_archs), len(no_skip_archs)) / len(architectures)),
            })

        params = [a.n_params for a in architectures]
        if len(params) > 5:
            efficiency = [s / (p + 1) * 1e6 for s, p in zip(scores, params)]
            best_eff_idx = np.argmax(efficiency)
            rules.append({
                "rule": f"Most efficient param count: {params[best_eff_idx]:,}",
                "efficiency": float(efficiency[best_eff_idx]),
                "confidence": 0.5,
            })

        return rules


class EvolutionaryArchitectureSearch:
    """Evolutionary search for optimal architectures."""

    def __init__(self, constraints: Optional[ArchitectureConstraints] = None,
                 population_size: int = 50, n_generations: int = 20,
                 mutation_rate: float = 0.3, crossover_rate: float = 0.3):
        self.constraints = constraints or ArchitectureConstraints()
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.sampler = ArchitectureSampler(self.constraints)
        self.mutator = ArchitectureMutator(self.constraints)
        self.predictor = PerformancePredictor()

    def search(self, fitness_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """Run evolutionary search."""
        if fitness_fn is None:
            fitness_fn = lambda arch: self.predictor.predict(arch).overall_score

        population = self.sampler.sample(self.population_size)
        best_fitness_history = []
        mean_fitness_history = []

        for generation in range(self.n_generations):
            fitness_values = []
            for arch in population:
                try:
                    fitness = fitness_fn(arch)
                except Exception:
                    fitness = 0.0
                fitness_values.append(fitness)

            fitness_values = np.array(fitness_values)
            best_idx = np.argmax(fitness_values)
            best_fitness_history.append(float(fitness_values[best_idx]))
            mean_fitness_history.append(float(np.mean(fitness_values)))

            sorted_indices = np.argsort(fitness_values)[::-1]
            n_elite = max(2, self.population_size // 5)
            elite = [population[i] for i in sorted_indices[:n_elite]]

            new_population = list(elite)

            fitness_probs = np.maximum(fitness_values, 0)
            fitness_probs = fitness_probs / (np.sum(fitness_probs) + 1e-12)

            while len(new_population) < self.population_size:
                if np.random.rand() < self.crossover_rate and len(elite) >= 2:
                    p1_idx, p2_idx = np.random.choice(len(elite), size=2, replace=False)
                    child = self.mutator.crossover(elite[p1_idx], elite[p2_idx])
                else:
                    parent_idx = np.random.choice(len(population), p=fitness_probs)
                    child = population[parent_idx]

                if np.random.rand() < self.mutation_rate:
                    child = self.mutator.mutate(child)

                if self._check_constraints(child):
                    new_population.append(child)

            population = new_population[:self.population_size]

        final_fitness = []
        for arch in population:
            try:
                final_fitness.append(fitness_fn(arch))
            except Exception:
                final_fitness.append(0.0)

        best_idx = np.argmax(final_fitness)

        return {
            "best_architecture": population[best_idx],
            "best_fitness": float(final_fitness[best_idx]),
            "best_fitness_history": best_fitness_history,
            "mean_fitness_history": mean_fitness_history,
            "final_population": population,
            "final_fitness": [float(f) for f in final_fitness],
            "n_generations": self.n_generations,
        }

    def _check_constraints(self, arch: Architecture) -> bool:
        """Check if architecture satisfies constraints."""
        if arch.n_params > self.constraints.max_params:
            return False
        if arch.flops > self.constraints.max_flops:
            return False
        if arch.depth < self.constraints.min_depth or arch.depth > self.constraints.max_depth:
            return False
        return True


class ArchitectureSpace:
    """Main class for architecture space exploration."""

    def __init__(self, constraints: Optional[ArchitectureConstraints] = None):
        self.constraints = constraints or ArchitectureConstraints()
        self.sampler = ArchitectureSampler(self.constraints)
        self.mutator = ArchitectureMutator(self.constraints)
        self.predictor = PerformancePredictor()
        self.clusterer = ArchitectureClusterer()
        self.rule_deriver = DesignRuleDeriver()
        self.searcher = EvolutionaryArchitectureSearch(
            self.constraints, population_size=30, n_generations=10
        )

    def define(self, constraints: ArchitectureConstraints):
        """Define the architecture search space."""
        self.constraints = constraints
        self.sampler = ArchitectureSampler(constraints)
        self.mutator = ArchitectureMutator(constraints)
        self.searcher = EvolutionaryArchitectureSearch(
            constraints, population_size=30, n_generations=10
        )

    def sample(self, n: int = 10) -> List[Architecture]:
        """Sample n random architectures."""
        return self.sampler.sample(n)

    def evaluate(self, arch: Architecture) -> ArchitectureMetrics:
        """Evaluate an architecture."""
        return self.predictor.predict(arch)

    def evaluate_batch(self, architectures: List[Architecture]) -> List[ArchitectureMetrics]:
        """Evaluate a batch of architectures."""
        return [self.evaluate(arch) for arch in architectures]

    def analyze_space(self, n_samples: int = 100) -> Dict[str, Any]:
        """Analyze the architecture space."""
        architectures = self.sample(n_samples)
        metrics = self.evaluate_batch(architectures)

        clustering = self.clusterer.cluster(architectures,
                                             n_clusters=min(5, len(architectures) // 3 + 1))
        rules = self.rule_deriver.derive_rules(architectures, metrics)

        params = [a.n_params for a in architectures]
        depths = [a.depth for a in architectures]
        widths = [np.mean(a.widths) for a in architectures]
        scores = [m.overall_score for m in metrics]

        return {
            "n_sampled": len(architectures),
            "param_range": [int(np.min(params)), int(np.max(params))],
            "depth_range": [int(np.min(depths)), int(np.max(depths))],
            "width_range": [float(np.min(widths)), float(np.max(widths))],
            "score_range": [float(np.min(scores)), float(np.max(scores))],
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "clustering": clustering,
            "rules": rules,
            "best_architecture_idx": int(np.argmax(scores)),
            "constraints_satisfied": all(
                self.sampler._satisfies_constraints(a) for a in architectures
            ),
        }

    def search(self, fitness_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """Search for optimal architecture."""
        return self.searcher.search(fitness_fn)

    def compare(self, architectures: List[Architecture]) -> Dict[str, Any]:
        """Compare multiple architectures."""
        metrics = self.evaluate_batch(architectures)

        comparison = []
        for arch, met in zip(architectures, metrics):
            comparison.append({
                "depth": arch.depth,
                "mean_width": float(np.mean(arch.widths)),
                "n_params": met.n_params,
                "flops": met.flops,
                "trainability": met.trainability_score,
                "expressivity": met.expressivity_score,
                "overall": met.overall_score,
                "phase": met.phase,
            })

        best_idx = int(np.argmax([m.overall_score for m in metrics]))
        return {
            "comparison": comparison,
            "best_index": best_idx,
            "rankings": np.argsort([-m.overall_score for m in metrics]).tolist(),
        }
