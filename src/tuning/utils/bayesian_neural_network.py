from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


class BayesianNeuralNetwork:
    """Fully-connected Bayesian neural network using diagonal Gaussian posteriors.

    The implementation intentionally remains lightweight (NumPy-only) for integration
    into agent components where external deep-learning frameworks are not required.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        learning_rate: float = 0.01,
        prior_mu: float = 0.0,
        prior_logvar: float = 0.0,
        random_state: Optional[int] = None,
        logvar_clip_range: Tuple[float, float] = (-8.0, 4.0),
    ) -> None:
        self.layer_sizes = [int(v) for v in layer_sizes]
        self.learning_rate = float(learning_rate)
        self.prior_mu = float(prior_mu)
        self.prior_logvar = float(prior_logvar)
        self.random_state = random_state
        self.logvar_clip_range = logvar_clip_range

        self._validate_init_args()

        self.num_layers = len(self.layer_sizes) - 1
        self.rng = np.random.default_rng(self.random_state)

        self.weights_mu: List[Array] = []
        self.weights_logvar: List[Array] = []
        self.biases_mu: List[Array] = []
        self.biases_logvar: List[Array] = []
        self._initialize_variational_parameters()

    def _validate_init_args(self) -> None:
        if len(self.layer_sizes) < 2:
            raise ValueError("layer_sizes must include input and output dimensions.")
        if any(size <= 0 for size in self.layer_sizes):
            raise ValueError("All layer sizes must be positive integers.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a positive finite float.")
        if not (isinstance(self.logvar_clip_range, tuple) and len(self.logvar_clip_range) == 2):
            raise ValueError("logvar_clip_range must be a (min, max) tuple.")
        if self.logvar_clip_range[0] >= self.logvar_clip_range[1]:
            raise ValueError("logvar_clip_range lower bound must be < upper bound.")

    def _initialize_variational_parameters(self) -> None:
        initial_logvar = math.log(0.1)
        for i in range(self.num_layers):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            scale = math.sqrt(2.0 / fan_in) * 0.1

            self.weights_mu.append(self.rng.normal(loc=0.0, scale=scale, size=(fan_in, fan_out)))
            self.weights_logvar.append(np.full((fan_in, fan_out), initial_logvar, dtype=float))
            self.biases_mu.append(np.zeros(fan_out, dtype=float))
            self.biases_logvar.append(np.full(fan_out, initial_logvar, dtype=float))

    def sample_parameters(self) -> Tuple[List[Array], List[Array]]:
        """Sample network parameters from the variational posterior."""
        weights: List[Array] = []
        biases: List[Array] = []

        for i in range(self.num_layers):
            eps_w = self.rng.standard_normal(self.weights_mu[i].shape)
            eps_b = self.rng.standard_normal(self.biases_mu[i].shape)

            std_w = np.exp(0.5 * self.weights_logvar[i])
            std_b = np.exp(0.5 * self.biases_logvar[i])

            weights.append(self.weights_mu[i] + eps_w * std_w)
            biases.append(self.biases_mu[i] + eps_b * std_b)

        return weights, biases

    def forward(self, x: Array, weights: List[Array], biases: List[Array]) -> Array:
        """Forward pass through sampled network parameters."""
        self._validate_forward_inputs(x, weights, biases)

        activation = x
        for layer_idx in range(self.num_layers - 1):
            z = np.dot(activation, weights[layer_idx]) + biases[layer_idx]
            activation = np.maximum(0.0, z)

        return np.dot(activation, weights[-1]) + biases[-1]

    def elbo(self, x: Array, y: Array, num_samples: int = 1) -> Tuple[float, float]:
        """Estimate ELBO and KL term with Monte Carlo sampling."""
        x_batch, y_batch = self._validate_training_batch(x, y)
        sample_count = self._validate_sample_count(num_samples)

        kl_divergence = self._total_kl_divergence()
        log_likelihood = 0.0

        for _ in range(sample_count):
            sampled_weights, sampled_biases = self.sample_parameters()
            outputs = self.forward(x_batch, sampled_weights, sampled_biases)
            log_likelihood -= 0.5 * float(np.sum((y_batch - outputs) ** 2))

        log_likelihood /= sample_count
        elbo_value = log_likelihood - kl_divergence / x_batch.shape[0]
        return float(elbo_value), float(kl_divergence)

    def _total_kl_divergence(self) -> float:
        prior_var = math.exp(self.prior_logvar)
        total = 0.0
        for i in range(self.num_layers):
            total += self._kl_divergence(self.weights_mu[i], self.weights_logvar[i], self.prior_mu, prior_var)
            total += self._kl_divergence(self.biases_mu[i], self.biases_logvar[i], self.prior_mu, prior_var)
        return float(total)

    @staticmethod
    def _kl_divergence(mu: Array, logvar: Array, prior_mu: float, prior_var: float) -> float:
        var = np.exp(logvar)
        return float(0.5 * np.sum((var + (mu - prior_mu) ** 2) / prior_var - logvar + math.log(prior_var) - 1.0))

    def train_step(self, x_batch: Array, y_batch: Array, num_samples: int = 1) -> Tuple[float, float]:
        """Perform one stochastic variational update step and return (ELBO, KL)."""
        x_valid, y_valid = self._validate_training_batch(x_batch, y_batch)
        sample_count = self._validate_sample_count(num_samples)

        gradients = self._compute_gradients(x_valid, y_valid, sample_count)

        for i in range(self.num_layers):
            self.weights_mu[i] += self.learning_rate * gradients["weights_mu"][i]
            self.weights_logvar[i] += self.learning_rate * gradients["weights_logvar"][i]
            self.biases_mu[i] += self.learning_rate * gradients["biases_mu"][i]
            self.biases_logvar[i] += self.learning_rate * gradients["biases_logvar"][i]

            self.weights_logvar[i] = np.clip(self.weights_logvar[i], *self.logvar_clip_range)
            self.biases_logvar[i] = np.clip(self.biases_logvar[i], *self.logvar_clip_range)

        return self.elbo(x_valid, y_valid, sample_count)

    def _compute_gradients(self, x: Array, y: Array, num_samples: int = 1) -> Dict[str, List[Array]]:
        """Compute reparameterization gradients for variational parameters."""
        gradients: Dict[str, List[Array]] = {
            "weights_mu": [np.zeros_like(w) for w in self.weights_mu],
            "weights_logvar": [np.zeros_like(w) for w in self.weights_logvar],
            "biases_mu": [np.zeros_like(b) for b in self.biases_mu],
            "biases_logvar": [np.zeros_like(b) for b in self.biases_logvar],
        }

        batch_size = x.shape[0]
        prior_var = math.exp(self.prior_logvar)

        for _ in range(num_samples):
            sampled_weights: List[Array] = []
            sampled_biases: List[Array] = []
            epsilons_w: List[Array] = []
            epsilons_b: List[Array] = []

            for i in range(self.num_layers):
                eps_w = self.rng.standard_normal(self.weights_mu[i].shape)
                eps_b = self.rng.standard_normal(self.biases_mu[i].shape)
                std_w = np.exp(0.5 * self.weights_logvar[i])
                std_b = np.exp(0.5 * self.biases_logvar[i])

                sampled_weights.append(self.weights_mu[i] + eps_w * std_w)
                sampled_biases.append(self.biases_mu[i] + eps_b * std_b)
                epsilons_w.append(eps_w)
                epsilons_b.append(eps_b)

            activations = [x]
            pre_activations = []

            activation = x
            for i in range(self.num_layers - 1):
                z = np.dot(activation, sampled_weights[i]) + sampled_biases[i]
                pre_activations.append(z)
                activation = np.maximum(0.0, z)
                activations.append(activation)

            z_final = np.dot(activation, sampled_weights[-1]) + sampled_biases[-1]
            pre_activations.append(z_final)
            activations.append(z_final)

            delta = (activations[-1] - y) / num_samples

            for i in range(self.num_layers - 1, -1, -1):
                d_w = np.dot(activations[i].T, delta)
                d_b = np.sum(delta, axis=0)

                gradients["weights_mu"][i] += d_w
                gradients["weights_logvar"][i] += d_w * epsilons_w[i] * 0.5 * np.exp(0.5 * self.weights_logvar[i])

                gradients["biases_mu"][i] += d_b
                gradients["biases_logvar"][i] += d_b * epsilons_b[i] * 0.5 * np.exp(0.5 * self.biases_logvar[i])

                if i > 0:
                    delta = np.dot(delta, sampled_weights[i].T)
                    delta = delta * (pre_activations[i - 1] > 0).astype(float)

        batch_scale = 1.0 / batch_size
        for i in range(self.num_layers):
            d_kl_w_mu = (self.weights_mu[i] - self.prior_mu) / prior_var
            d_kl_w_logvar = 0.5 * (np.exp(self.weights_logvar[i]) / prior_var - 1.0)
            d_kl_b_mu = (self.biases_mu[i] - self.prior_mu) / prior_var
            d_kl_b_logvar = 0.5 * (np.exp(self.biases_logvar[i]) / prior_var - 1.0)

            gradients["weights_mu"][i] -= batch_scale * d_kl_w_mu
            gradients["weights_logvar"][i] -= batch_scale * d_kl_w_logvar
            gradients["biases_mu"][i] -= batch_scale * d_kl_b_mu
            gradients["biases_logvar"][i] -= batch_scale * d_kl_b_logvar

        return gradients

    def predict(self, x: Array, num_samples: int = 100) -> Tuple[Array, Array]:
        """Predict mean and epistemic uncertainty with Monte Carlo sampling."""
        x_array = np.asarray(x, dtype=float)
        if x_array.ndim != 2:
            raise ValueError("x must be a 2D array of shape (batch_size, input_dim).")
        if x_array.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"Input feature dimension mismatch: expected {self.layer_sizes[0]}, got {x_array.shape[1]}."
            )
        sample_count = self._validate_sample_count(num_samples)

        predictions = []
        for _ in range(sample_count):
            sampled_weights, sampled_biases = self.sample_parameters()
            predictions.append(self.forward(x_array, sampled_weights, sampled_biases))

        stacked = np.stack(predictions, axis=0)
        return np.mean(stacked, axis=0), np.std(stacked, axis=0)

    def save(self, filename: str) -> None:
        """Persist model parameters and metadata to JSON."""
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "layer_sizes": self.layer_sizes,
            "learning_rate": self.learning_rate,
            "prior_mu": self.prior_mu,
            "prior_logvar": self.prior_logvar,
            "random_state": self.random_state,
            "logvar_clip_range": list(self.logvar_clip_range),
            "weights_mu": [w.tolist() for w in self.weights_mu],
            "weights_logvar": [w.tolist() for w in self.weights_logvar],
            "biases_mu": [b.tolist() for b in self.biases_mu],
            "biases_logvar": [b.tolist() for b in self.biases_logvar],
        }

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    @classmethod
    def load(cls, filename: str) -> "BayesianNeuralNetwork":
        """Load a saved BNN model from JSON file."""
        with Path(filename).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        model = cls(
            layer_sizes=payload["layer_sizes"],
            learning_rate=float(payload.get("learning_rate", 0.01)),
            prior_mu=float(payload.get("prior_mu", 0.0)),
            prior_logvar=float(payload.get("prior_logvar", 0.0)),
            random_state=payload.get("random_state"),
            logvar_clip_range=tuple(payload.get("logvar_clip_range", [-8.0, 4.0])),
        )

        model.weights_mu = [np.asarray(w, dtype=float) for w in payload["weights_mu"]]
        model.weights_logvar = [np.asarray(w, dtype=float) for w in payload["weights_logvar"]]
        model.biases_mu = [np.asarray(b, dtype=float) for b in payload["biases_mu"]]
        model.biases_logvar = [np.asarray(b, dtype=float) for b in payload["biases_logvar"]]
        model._validate_parameter_shapes()
        return model

    def _validate_forward_inputs(self, x: Array, weights: List[Array], biases: List[Array]) -> None:
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (batch_size, input_dim).")
        if x.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"Input feature dimension mismatch: expected {self.layer_sizes[0]}, got {x.shape[1]}."
            )
        if len(weights) != self.num_layers or len(biases) != self.num_layers:
            raise ValueError("weights and biases must match network layer count.")

    def _validate_training_batch(self, x: Array, y: Array) -> Tuple[Array, Array]:
        x_batch = np.asarray(x, dtype=float)
        y_batch = np.asarray(y, dtype=float)

        if x_batch.ndim != 2:
            raise ValueError("x_batch must be a 2D array.")
        if y_batch.ndim == 1:
            y_batch = y_batch.reshape(-1, 1)
        if y_batch.ndim != 2:
            raise ValueError("y_batch must be a 1D or 2D array.")
        if x_batch.shape[0] != y_batch.shape[0]:
            raise ValueError("x_batch and y_batch must have matching sample counts.")
        if x_batch.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"Input feature dimension mismatch: expected {self.layer_sizes[0]}, got {x_batch.shape[1]}."
            )
        if y_batch.shape[1] != self.layer_sizes[-1]:
            raise ValueError(
                f"Target feature dimension mismatch: expected {self.layer_sizes[-1]}, got {y_batch.shape[1]}."
            )

        return x_batch, y_batch

    @staticmethod
    def _validate_sample_count(num_samples: int) -> int:
        sample_count = int(num_samples)
        if sample_count < 1:
            raise ValueError("num_samples must be >= 1.")
        return sample_count

    def _validate_parameter_shapes(self) -> None:
        if not (
            len(self.weights_mu)
            == len(self.weights_logvar)
            == len(self.biases_mu)
            == len(self.biases_logvar)
            == self.num_layers
        ):
            raise ValueError("Loaded parameter lists have inconsistent layer counts.")

        for i in range(self.num_layers):
            expected_weight_shape = (self.layer_sizes[i], self.layer_sizes[i + 1])
            expected_bias_shape = (self.layer_sizes[i + 1],)
            if self.weights_mu[i].shape != expected_weight_shape or self.weights_logvar[i].shape != expected_weight_shape:
                raise ValueError(f"Invalid weight shape at layer {i}: expected {expected_weight_shape}.")
            if self.biases_mu[i].shape != expected_bias_shape or self.biases_logvar[i].shape != expected_bias_shape:
                raise ValueError(f"Invalid bias shape at layer {i}: expected {expected_bias_shape}.")
