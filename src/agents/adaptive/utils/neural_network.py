from __future__ import annotations

import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...base.modules.activation_engine import (
    ELU,
    GELU,
    LeakyReLU,
    Linear,
    Mish,
    ReLU,
    Sigmoid,
    Softmax,
    Swish,
    Tanh,
    he_init,
    lecun_normal,
    xavier_uniform,
)
from .adaptive_errors import *
from .config_loader import load_global_config, get_config_section
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Adaptive Neural Network")
printer = PrettyPrinter

NETWORK_COMPONENT = "neural_network"
ACTOR_CRITIC_COMPONENT = "actor_critic"
BAYESIAN_COMPONENT = "bayesian_dqn"


class ActivationWrapper(nn.Module):
    """Wrap activation engine objects with a PyTorch-compatible module interface."""

    def __init__(self, activation: Any):
        super().__init__()
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation.forward(x)

    def extra_repr(self) -> str:
        return f"activation={type(self.activation).__name__}"


class AdaptiveNetworkBase(nn.Module):
    """
    Shared functionality for adaptive neural-network implementations.

    Responsibilities:
    - retain project config loading semantics via load_global_config/get_config_section
    - validate layer specifications and dimensionality
    - centralize activation, initialization, checkpointing, and tensor conversion
    - provide consistent, structured error handling using adaptive_errors.py
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = load_global_config()
        self.problem_type = str(self.config.get("problem_type", "regression")).lower()
        self.dim = int(self.config.get("final_activation_dim", -1))
        self.device_setting = self.config.get("device", "auto")
        self.seed = self.config.get("seed")
        if self.seed is not None:
            self._set_seed(int(self.seed))
        self.device = self._resolve_device(self.device_setting)
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _resolve_device(device_setting: str) -> torch.device:
        if device_setting == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_setting)

    def _activation_registry(self, alpha: float) -> Dict[str, Any]:
        return {
            "relu": ReLU(),
            "leaky_relu": LeakyReLU(alpha),
            "elu": ELU(alpha),
            "swish": Swish(),
            "mish": Mish(),
            "gelu": GELU(),
            "sigmoid": Sigmoid(),
            "tanh": Tanh(),
            "linear": Linear(),
            "softmax": Softmax(dim=self.dim),
        }

    def _get_activation(self, name: str, config: Optional[Dict[str, Any]] = None, *, component: str = NETWORK_COMPONENT) -> Any:
        config = config or {}
        alpha = float(config.get("alpha", 0.01))
        activation_map = self._activation_registry(alpha)
        normalized_name = str(name).lower()
        if normalized_name not in activation_map:
            raise UnsupportedActivationError(
                f"Unsupported activation '{name}'.",
                component=component,
                details={"requested_activation": name, "available": sorted(activation_map.keys())},
                remediation="Update adaptive_config.yaml to use a supported activation.",
            )
        return activation_map[normalized_name]

    def _init_weights(self, layer: nn.Linear, init_method: str, *, component: str = NETWORK_COMPONENT) -> None:
        shape = (layer.out_features, layer.in_features)
        device = layer.weight.device
        init_name = str(init_method).lower()

        try:
            with torch.no_grad():
                if init_name == "uniform_scaled":
                    limit = 1.0 / math.sqrt(layer.in_features)
                    layer.weight.copy_(torch.empty(shape, device=device).uniform_(-limit, limit))
                elif init_name == "he_normal":
                    layer.weight.copy_(he_init(shape, nonlinearity="relu", device=device))
                elif init_name == "xavier_uniform":
                    layer.weight.copy_(xavier_uniform(shape, device=device))
                elif init_name == "lecun_normal":
                    layer.weight.copy_(lecun_normal(shape, device=device))
                else:
                    raise UnsupportedInitializationMethodError(
                        f"Unsupported initialization method '{init_method}'.",
                        component=component,
                        details={"requested_initialization": init_method},
                        remediation="Use one of: uniform_scaled, he_normal, xavier_uniform, lecun_normal.",
                    )
                nn.init.constant_(layer.bias, 0.0)
        except AdaptiveError:
            raise
        except Exception as exc:
            raise wrap_exception(
                exc,
                UnsupportedInitializationMethodError,
                f"Failed to initialize layer with method '{init_method}'.",
                component=component,
                details={"shape": shape, "init_method": init_method},
            )

    def _normalize_layer_config(
        self,
        layers: Union[Sequence[int], Sequence[Dict[str, Any]]],
        *,
        component: str,
        default_activation: str = "relu",
    ) -> List[Dict[str, Any]]:
        ensure_instance(layers, (list, tuple), "layers", component=component)
        ensure_non_empty(layers, "layers", component=component)

        normalized: List[Dict[str, Any]] = []
        for idx, layer in enumerate(layers):
            if isinstance(layer, int):
                ensure_positive(layer, f"layers[{idx}]", component=component)
                normalized.append(
                    {
                        "neurons": int(layer),
                        "activation": default_activation,
                        "batch_norm": False,
                        "dropout": 0.0,
                    }
                )
                continue

            ensure_instance(layer, dict, f"layers[{idx}]", component=component)
            if "neurons" not in layer:
                raise NetworkConfigurationError(
                    f"Layer configuration at index {idx} must define 'neurons'.",
                    component=component,
                    details={"layer_index": idx, "layer": layer},
                    remediation="Add a positive integer 'neurons' field to the layer config.",
                )

            neurons = int(layer["neurons"])
            ensure_positive(neurons, f"layers[{idx}].neurons", component=component)
            dropout = float(layer.get("dropout", 0.0))
            ensure_in_range(dropout, f"layers[{idx}].dropout", minimum=0.0, maximum=1.0, component=component)

            normalized_layer = {
                "neurons": neurons,
                "activation": str(layer.get("activation", default_activation)).lower(),
                "batch_norm": bool(layer.get("batch_norm", False)),
                "dropout": dropout,
                "init": str(layer.get("init", "he_normal")).lower(),
            }
            if "alpha" in layer:
                normalized_layer["alpha"] = float(layer["alpha"])
            normalized.append(normalized_layer)

        return normalized

    def _force_output_dim(
        self,
        layers: List[Dict[str, Any]],
        output_dim: int,
        *,
        component: str,
        output_activation: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        ensure_positive(output_dim, "output_dim", component=component)
        normalized = deepcopy(layers)
        ensure_non_empty(normalized, "layers", component=component)
        normalized[-1]["neurons"] = int(output_dim)
        if output_activation is not None:
            normalized[-1]["activation"] = str(output_activation).lower()
        return normalized

    def _build_mlp(
        self,
        input_dim: int,
        layer_config: List[Dict[str, Any]],
        *,
        component: str,
        default_init_method: str,
        activate_last_layer: bool,
        default_activation: str = "relu",
        return_metadata: bool = False,
    ) -> Union[nn.Sequential, Tuple[nn.Sequential, List[Dict[str, Any]]]]:
        ensure_positive(input_dim, "input_dim", component=component)
        layers = self._normalize_layer_config(layer_config, component=component, default_activation=default_activation)

        modules: List[nn.Module] = []
        metadata: List[Dict[str, Any]] = []
        current_dim = int(input_dim)

        for idx, layer_conf in enumerate(layers):
            neurons = int(layer_conf["neurons"])
            linear = nn.Linear(current_dim, neurons)
            init_method = str(layer_conf.get("init", default_init_method)).lower()
            self._init_weights(linear, init_method, component=component)
            modules.append(linear)

            is_last = idx == len(layers) - 1
            should_activate = activate_last_layer or not is_last
            batch_norm = bool(layer_conf.get("batch_norm", False)) and (not is_last or activate_last_layer)
            dropout = float(layer_conf.get("dropout", 0.0)) if (not is_last or activate_last_layer) else 0.0

            if batch_norm:
                modules.append(nn.BatchNorm1d(neurons))
            if should_activate:
                modules.append(ActivationWrapper(self._get_activation(layer_conf.get("activation", default_activation), layer_conf, component=component)))
            if dropout > 0.0:
                modules.append(nn.Dropout(dropout))

            metadata.append(
                {
                    "in_features": current_dim,
                    "out_features": neurons,
                    "init": init_method,
                    "activation": str(layer_conf.get("activation", default_activation)).lower() if should_activate else None,
                    "batch_norm": batch_norm,
                    "dropout": dropout,
                }
            )
            current_dim = neurons

        model = nn.Sequential(*modules)
        return (model, metadata) if return_metadata else model

    def _prepare_input_tensor(
        self,
        inputs: Union[np.ndarray, torch.Tensor, Sequence[float]],
        *,
        expected_dim: Optional[int] = None,
        component: str = NETWORK_COMPONENT,
    ) -> torch.Tensor:
        try:
            if isinstance(inputs, torch.Tensor):
                tensor = inputs.detach().clone().float()
            else:
                tensor = torch.as_tensor(inputs, dtype=torch.float32)
        except Exception as exc:
            raise wrap_exception(
                exc,
                InvalidTypeError,
                "Failed to convert inputs to a torch.FloatTensor.",
                component=component,
                details={"input_type": type(inputs).__name__},
            )

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        if expected_dim is not None:
            ensure_dimension(int(tensor.shape[-1]), int(expected_dim), name="input feature dimension", component=component)

        return tensor.to(self.device)

    def _ensure_parent_dir(self, filepath: Union[str, Path]) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _safe_checkpoint_save(self, payload: Dict[str, Any], filepath: Union[str, Path], *, component: str) -> Path:
        path = self._ensure_parent_dir(filepath)
        try:
            torch.save(payload, path)
            return path
        except Exception as exc:
            raise wrap_exception(
                exc,
                CheckpointSaveError,
                f"Failed to save checkpoint to '{path}'.",
                component=component,
                details={"filepath": str(path)},
                remediation="Verify write permissions and checkpoint payload compatibility.",
            )

    @staticmethod
    def _safe_checkpoint_load(filepath: Union[str, Path], *, map_location: str = "cpu", component: str) -> Dict[str, Any]:
        path = Path(filepath)
        if not path.exists():
            raise CheckpointNotFoundError(
                f"Checkpoint file '{path}' was not found.",
                component=component,
                details={"filepath": str(path)},
                remediation="Save the model first or provide a valid checkpoint path.",
            )
        try:
            checkpoint = torch.load(path, map_location=map_location)
        except Exception as exc:
            raise wrap_exception(
                exc,
                CheckpointLoadError,
                f"Failed to load checkpoint from '{path}'.",
                component=component,
                details={"filepath": str(path)},
                remediation="Verify that the checkpoint is readable and compatible with this model version.",
            )

        ensure_instance(checkpoint, dict, "checkpoint", component=component)
        return checkpoint

    def parameter_count(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(param.numel() for param in self.parameters() if param.requires_grad)
        return sum(param.numel() for param in self.parameters())


class NeuralNetwork(AdaptiveNetworkBase):
    """
    Production-ready adaptive neural network with:
    - config-driven architecture from adaptive_config.yaml
    - robust training/evaluation/prediction helpers
    - classification-aware output handling
    - optional scheduler and early stopping
    - structured persistence and error handling
    """

    def __init__(
        self,
        nn_config_override: Optional[Dict[str, Any]] = None,
        global_config_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if global_config_override:
            self.config.update(global_config_override)
            self.problem_type = str(self.config.get("problem_type", "regression")).lower()
            self.dim = int(self.config.get("final_activation_dim", -1))
            self.device_setting = self.config.get("device", self.device_setting)
            self.device = self._resolve_device(self.device_setting)

        self.nn_config = get_config_section("neural_network")
        if nn_config_override:
            self.nn_config.update(nn_config_override)

        self.input_dim = ensure_not_none(self.nn_config.get("input_dim"), "neural_network.input_dim", component=NETWORK_COMPONENT)
        self.layer_config = ensure_not_none(self.nn_config.get("layer_config"), "neural_network.layer_config", component=NETWORK_COMPONENT)
        self.input_dim = int(self.input_dim)
        ensure_positive(self.input_dim, "neural_network.input_dim", component=NETWORK_COMPONENT)

        self.default_init_method = str(self.nn_config.get("initialization_method_default", "he_normal")).lower()
        self.optimizer_name = str(self.nn_config.get("optimizer_name", "adam")).lower()
        self.learning_rate = float(self.nn_config.get("learning_rate", self.config.get("parameter_tuner", {}).get("base_learning_rate", 1e-3)))
        self.weight_decay = float(self.nn_config.get("weight_decay", self.config.get("parameter_tuner", {}).get("weight_decay_lambda", 0.0) or 0.0))
        self.loss_name = str(self.nn_config.get("loss_function_name", "mse")).lower()
        self.clip_grad_norm = float(self.nn_config.get("clip_grad_norm", self.config.get("gradient_clip_value", 1.0)))
        self.batch_size = int(self.nn_config.get("batch_size", 32))
        self.default_epochs = int(self.nn_config.get("epochs", 50))
        self.restore_best_weights = bool(self.nn_config.get("restore_best_weights", True))
        self.track_history = bool(self.nn_config.get("track_history", True))

        ensure_positive(self.learning_rate, "neural_network.learning_rate", component=NETWORK_COMPONENT)
        ensure_in_range(self.weight_decay, "neural_network.weight_decay", minimum=0.0, component=NETWORK_COMPONENT)
        ensure_positive(self.clip_grad_norm, "neural_network.clip_grad_norm", allow_zero=True, component=NETWORK_COMPONENT)
        ensure_positive(self.batch_size, "neural_network.batch_size", component=NETWORK_COMPONENT)
        ensure_positive(self.default_epochs, "neural_network.epochs", component=NETWORK_COMPONENT)

        normalized_layers = self._normalize_layer_config(self.layer_config, component=NETWORK_COMPONENT)
        ensure_non_empty(normalized_layers, "neural_network.layer_config", component=NETWORK_COMPONENT)
        self.output_dim = int(normalized_layers[-1]["neurons"])
        ensure_positive(self.output_dim, "neural_network.output_dim", component=NETWORK_COMPONENT)

        default_final_activation = (
            "sigmoid"
            if self.problem_type == "binary_classification"
            else "softmax"
            if self.problem_type == "multiclass_classification"
            else "linear"
        )
        self.final_activation_name = str(self.nn_config.get("final_activation", default_final_activation)).lower()
        self.apply_final_activation_in_forward = bool(
            self.nn_config.get(
                "apply_final_activation_in_forward",
                self.problem_type == "regression" and self.final_activation_name != "linear",
            )
        )
        self.final_activation = self._get_activation(self.final_activation_name, component=NETWORK_COMPONENT)

        self.network, self.layer_metadata = self._build_mlp(
            self.input_dim,
            normalized_layers,
            component=NETWORK_COMPONENT,
            default_init_method=self.default_init_method,
            activate_last_layer=False,
            return_metadata=True,
        )
        self.layers = self.network
        ensure(
            self.parameter_count(trainable_only=True) > 0,
            "No trainable parameters found in the neural network.",
            exc_type=NetworkConfigurationError,
            component=NETWORK_COMPONENT,
            details={"input_dim": self.input_dim, "layer_config": normalized_layers},
            remediation="Ensure layer_config defines at least one valid linear layer.",
        )

        self.optimizer = self._configure_optimizer()
        self.loss_fn = self._configure_loss_function()
        self.scheduler = self._configure_scheduler()
        self._is_fitted = False
        self.to(self.device)

        logger.info(
            "Neural Network initialized | problem_type=%s | input_dim=%s | output_dim=%s | params=%s | device=%s",
            self.problem_type,
            self.input_dim,
            self.output_dim,
            self.parameter_count(),
            self.device,
        )

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        printer.status("INIT", "Configuring neural network optimizer", "info")
        optimizer_name = self.optimizer_name
        if optimizer_name == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if optimizer_name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if optimizer_name == "sgd":
            momentum = float(self.nn_config.get("momentum", 0.0))
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=momentum, weight_decay=self.weight_decay)
        if optimizer_name == "rmsprop":
            momentum = float(self.nn_config.get("momentum", 0.0))
            alpha = float(self.nn_config.get("alpha", 0.99))
            return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, alpha=alpha, momentum=momentum, weight_decay=self.weight_decay)
        raise UnsupportedOptimizerError(
            f"Unsupported optimizer '{self.optimizer_name}'.",
            component=NETWORK_COMPONENT,
            details={"requested_optimizer": self.optimizer_name},
            remediation="Use one of: adam, adamw, sgd, rmsprop.",
        )

    def _configure_loss_function(self) -> nn.Module:
        printer.status("INIT", "Configuring neural network loss function", "info")
        loss_name = self.loss_name

        if loss_name in {"mse", "mse_loss"}:
            return nn.MSELoss()
        if loss_name in {"mae", "l1", "l1_loss"}:
            return nn.L1Loss()
        if loss_name in {"huber", "smooth_l1"}:
            beta = float(self.nn_config.get("huber_beta", 1.0))
            return nn.SmoothL1Loss(beta=beta)
        if loss_name in {"cross_entropy", "ce"}:
            if self.problem_type == "multiclass_classification":
                return nn.CrossEntropyLoss()
            if self.problem_type == "binary_classification":
                return nn.BCEWithLogitsLoss()
            raise UnsupportedLossError(
                "Cross-entropy loss requires a binary_classification or multiclass_classification problem type.",
                component=NETWORK_COMPONENT,
                details={"problem_type": self.problem_type, "loss_name": self.loss_name},
            )
        if loss_name in {"bce", "bce_with_logits", "binary_cross_entropy"}:
            ensure(
                self.problem_type == "binary_classification",
                "BCEWithLogitsLoss requires problem_type='binary_classification'.",
                exc_type=UnsupportedLossError,
                component=NETWORK_COMPONENT,
                details={"problem_type": self.problem_type, "loss_name": self.loss_name},
            )
            return nn.BCEWithLogitsLoss()

        raise UnsupportedLossError(
            f"Unsupported loss function '{self.loss_name}'.",
            component=NETWORK_COMPONENT,
            details={"requested_loss": self.loss_name},
            remediation="Use one of: mse, mae, huber, cross_entropy, bce_with_logits.",
        )

    def _configure_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        scheduler_cfg = self.nn_config.get("scheduler")
        if not scheduler_cfg:
            return None
        ensure_instance(scheduler_cfg, dict, "neural_network.scheduler", component=NETWORK_COMPONENT)
        scheduler_name = str(scheduler_cfg.get("name", "")).lower().strip()
        if not scheduler_name:
            return None

        if scheduler_name == "step":
            step_size = int(scheduler_cfg.get("step_size", 10))
            gamma = float(scheduler_cfg.get("gamma", 0.1))
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        if scheduler_name == "cosine":
            t_max = int(scheduler_cfg.get("t_max", max(1, self.default_epochs)))
            eta_min = float(scheduler_cfg.get("eta_min", 0.0))
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)
        if scheduler_name == "reduce_on_plateau":
            mode = str(scheduler_cfg.get("mode", "min"))
            factor = float(scheduler_cfg.get("factor", 0.5))
            patience = int(scheduler_cfg.get("patience", 5))
            min_lr = float(scheduler_cfg.get("min_lr", 1e-6))
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            )

        raise InvalidConfigurationValueError(
            f"Unsupported scheduler '{scheduler_name}'.",
            component=NETWORK_COMPONENT,
            details={"scheduler": scheduler_cfg},
            remediation="Use one of: step, cosine, reduce_on_plateau.",
        )

    def _apply_final_activation(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.final_activation_name == "linear":
            return outputs
        return self.final_activation.forward(outputs)

    def _loss_expects_logits(self) -> bool:
        return isinstance(self.loss_fn, (nn.CrossEntropyLoss, nn.BCEWithLogitsLoss))

    def forward(self, x: Union[np.ndarray, torch.Tensor, Sequence[float]]) -> torch.Tensor:
        printer.status("INIT", "Forward pass initialized", "info")
        inputs = self._prepare_input_tensor(x, expected_dim=self.input_dim, component=NETWORK_COMPONENT)
        try:
            outputs = self.network(inputs)
            if self.apply_final_activation_in_forward and not self._loss_expects_logits():
                outputs = self._apply_final_activation(outputs)
            return outputs
        except AdaptiveError:
            raise
        except Exception as exc:
            raise wrap_exception(
                exc,
                ForwardPassError,
                "Forward pass failed for NeuralNetwork.",
                component=NETWORK_COMPONENT,
                details={"input_shape": tuple(inputs.shape)},
            )

    def predict_tensor(self, inputs: Union[np.ndarray, torch.Tensor, Sequence[float]], apply_output_activation: bool = True) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs)
            if apply_output_activation and self._loss_expects_logits():
                outputs = self._apply_final_activation(outputs)
            return outputs

    def predict(self, inputs: Union[np.ndarray, torch.Tensor, Sequence[float]], apply_output_activation: bool = True) -> np.ndarray:
        printer.status("PREDICT", "Generating predictions", "info")
        outputs = self.predict_tensor(inputs, apply_output_activation=apply_output_activation)
        self._is_fitted = True if self._is_fitted else self._is_fitted
        return outputs.detach().cpu().numpy()

    def predict_proba(self, inputs: Union[np.ndarray, torch.Tensor, Sequence[float]]) -> np.ndarray:
        if self.problem_type not in {"binary_classification", "multiclass_classification"}:
            raise UnsupportedLossError(
                "predict_proba is only supported for classification problem types.",
                component=NETWORK_COMPONENT,
                details={"problem_type": self.problem_type},
            )
        return self.predict(inputs, apply_output_activation=True)

    def predict_class(self, inputs: Union[np.ndarray, torch.Tensor, Sequence[float]]) -> np.ndarray:
        outputs = self.predict(inputs, apply_output_activation=True)
        if self.problem_type == "binary_classification":
            if outputs.shape[-1] == 1:
                return (outputs >= 0.5).astype(int).reshape(-1)
            return np.argmax(outputs, axis=-1)
        if self.problem_type == "multiclass_classification":
            return np.argmax(outputs, axis=-1)
        return outputs

    def _prepare_targets(self, targets: Sequence[Any]) -> torch.Tensor:
        printer.status("INIT", "Preparing target tensors", "info")
        ensure_instance(targets, (list, tuple), "targets", component=NETWORK_COMPONENT)
        ensure_non_empty(targets, "targets", component=NETWORK_COMPONENT)

        if self.problem_type == "multiclass_classification":
            class_indices: List[int] = []
            for target in targets:
                arr = np.asarray(target)
                if arr.ndim == 0:
                    class_indices.append(int(arr))
                else:
                    class_indices.append(int(np.argmax(arr)))
            return torch.as_tensor(class_indices, dtype=torch.long, device=self.device)

        target_array = np.asarray(targets, dtype=np.float32)
        if self.problem_type == "binary_classification":
            if target_array.ndim == 1:
                target_array = target_array.reshape(-1, 1)
        return torch.as_tensor(target_array, dtype=torch.float32, device=self.device)

    def _prepare_dataset(
        self,
        dataset: Sequence[Tuple[np.ndarray, Any]],
        *,
        dataset_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ensure_instance(dataset, (list, tuple), dataset_name, component=NETWORK_COMPONENT)
        ensure_non_empty(dataset, dataset_name, component=NETWORK_COMPONENT)

        inputs: List[np.ndarray] = []
        targets: List[Any] = []
        for idx, sample in enumerate(dataset):
            ensure_instance(sample, (list, tuple), f"{dataset_name}[{idx}]", component=NETWORK_COMPONENT)
            ensure(
                len(sample) == 2,
                f"Each item in {dataset_name} must contain (input, target).",
                exc_type=InvalidTypeError,
                component=NETWORK_COMPONENT,
                details={"dataset_name": dataset_name, "index": idx, "value": sample},
            )
            x, y = sample
            x_arr = np.asarray(x, dtype=np.float32)
            if x_arr.ndim != 1:
                x_arr = x_arr.reshape(-1)
            ensure_dimension(int(x_arr.shape[0]), self.input_dim, name=f"{dataset_name}[{idx}] input_dim", component=NETWORK_COMPONENT)
            inputs.append(x_arr)
            targets.append(y)

        inputs_tensor = torch.as_tensor(np.stack(inputs), dtype=torch.float32, device=self.device)
        targets_tensor = self._prepare_targets(targets)
        return inputs_tensor, targets_tensor

    def _slice_batch(self, inputs: torch.Tensor, targets: torch.Tensor, start: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end = min(start + batch_size, inputs.shape[0])
        return inputs[start:end], targets[start:end]

    def _train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        printer.status("INIT", "Training step initialized", "info")
        self.optimizer.zero_grad(set_to_none=True)
        try:
            outputs = self.forward(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            return loss.detach()
        except AdaptiveError:
            raise
        except Exception as exc:
            raise wrap_exception(
                exc,
                TrainingStepError,
                "Training step failed for NeuralNetwork.",
                component=NETWORK_COMPONENT,
                details={
                    "input_shape": tuple(inputs.shape),
                    "target_shape": tuple(targets.shape),
                    "optimizer": self.optimizer_name,
                    "loss_name": self.loss_name,
                },
            )

    def train_network(
        self,
        training_data: List[Tuple[np.ndarray, Any]],
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        validation_data: Optional[List[Tuple[np.ndarray, Any]]] = None,
        early_stopping_patience: Optional[int] = None,
        verbose: bool = True,
        shuffle: bool = True,
    ) -> Dict[str, Any]:
        printer.status("TRAIN", "Neural network training initialized", "info")
        epochs = int(epochs or self.default_epochs)
        batch_size = int(batch_size or self.batch_size)
        ensure_positive(epochs, "epochs", component=NETWORK_COMPONENT)
        ensure_positive(batch_size, "batch_size", component=NETWORK_COMPONENT)

        train_inputs, train_targets = self._prepare_dataset(training_data, dataset_name="training_data")
        val_inputs = val_targets = None
        if validation_data is not None:
            val_inputs, val_targets = self._prepare_dataset(validation_data, dataset_name="validation_data")

        best_val_loss = float("inf")
        best_state_dict = None
        epochs_without_improvement = 0

        self.train()
        for epoch in range(epochs):
            if shuffle:
                permutation = torch.randperm(train_inputs.shape[0], device=self.device)
                epoch_inputs = train_inputs[permutation]
                epoch_targets = train_targets[permutation]
            else:
                epoch_inputs = train_inputs
                epoch_targets = train_targets

            batch_losses: List[float] = []
            for start_idx in range(0, epoch_inputs.shape[0], batch_size):
                batch_x, batch_y = self._slice_batch(epoch_inputs, epoch_targets, start_idx, batch_size)
                loss = self._train_step(batch_x, batch_y)
                batch_losses.append(float(loss.item()))

            train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            val_loss = None

            if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()

            if val_inputs is not None and val_targets is not None:
                val_loss = self.evaluate(validation_data)
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = deepcopy(self.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(train_loss)

            if self.track_history:
                self.training_history["train_loss"].append(train_loss)
                self.training_history["learning_rate"].append(float(self.optimizer.param_groups[0]["lr"]))
                if val_loss is not None:
                    self.training_history["val_loss"].append(float(val_loss))

            if verbose:
                val_msg = f" | val_loss={val_loss:.6f}" if val_loss is not None else ""
                print(
                    f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.6f}{val_msg} | lr={self.optimizer.param_groups[0]['lr']:.6e}"
                )

            if early_stopping_patience is not None and val_loss is not None:
                if epochs_without_improvement >= int(early_stopping_patience):
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        if self.restore_best_weights and best_state_dict is not None:
            self.load_state_dict(best_state_dict)

        self._is_fitted = True
        self.eval()
        return {
            "epochs_ran": len(self.training_history["train_loss"]),
            "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
            "final_train_loss": self.training_history["train_loss"][-1] if self.training_history["train_loss"] else None,
            "final_val_loss": self.training_history["val_loss"][-1] if self.training_history["val_loss"] else None,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "parameter_count": self.parameter_count(),
        }

    def evaluate(self, test_data: List[Tuple[np.ndarray, Any]]) -> float:
        printer.status("EVAL", "Evaluating neural network", "info")
        inputs, targets = self._prepare_dataset(test_data, dataset_name="test_data")
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs)
            loss = self.loss_fn(outputs, targets)
        return float(loss.item())

    def get_weights_biases(self) -> List[Dict[str, np.ndarray]]:
        weights_biases: List[Dict[str, np.ndarray]] = []
        for module in self.network:
            if isinstance(module, nn.Linear):
                weights_biases.append(
                    {
                        "weights": module.weight.detach().cpu().numpy().copy(),
                        "bias": module.bias.detach().cpu().numpy().copy(),
                    }
                )
        return weights_biases

    def set_weights_biases(self, weights_biases: List[Dict[str, Any]]) -> None:
        ensure_instance(weights_biases, list, "weights_biases", component=NETWORK_COMPONENT)
        ensure_non_empty(weights_biases, "weights_biases", component=NETWORK_COMPONENT)

        linear_layers = [module for module in self.network if isinstance(module, nn.Linear)]
        if len(weights_biases) != len(linear_layers):
            raise DimensionMismatchError(
                "weights_biases length does not match the number of linear layers.",
                component=NETWORK_COMPONENT,
                details={"provided": len(weights_biases), "expected": len(linear_layers)},
                remediation="Provide one weight/bias entry per linear layer.",
            )

        for idx, (layer, params) in enumerate(zip(linear_layers, weights_biases)):
            ensure_instance(params, dict, f"weights_biases[{idx}]", component=NETWORK_COMPONENT)
            ensure("weights" in params and "bias" in params, f"weights_biases[{idx}] must contain 'weights' and 'bias'.", exc_type=InvalidTypeError, component=NETWORK_COMPONENT)

            weights = np.asarray(params["weights"], dtype=np.float32)
            bias = np.asarray(params["bias"], dtype=np.float32)
            expected_weight_shape = (layer.out_features, layer.in_features)
            expected_bias_shape = (layer.out_features,)

            if weights.shape != expected_weight_shape:
                raise DimensionMismatchError(
                    f"Layer {idx} weight shape mismatch.",
                    component=NETWORK_COMPONENT,
                    details={"expected": expected_weight_shape, "received": weights.shape},
                )
            if bias.shape != expected_bias_shape:
                raise DimensionMismatchError(
                    f"Layer {idx} bias shape mismatch.",
                    component=NETWORK_COMPONENT,
                    details={"expected": expected_bias_shape, "received": bias.shape},
                )

            with torch.no_grad():
                layer.weight.copy_(torch.as_tensor(weights, dtype=torch.float32, device=layer.weight.device))
                layer.bias.copy_(torch.as_tensor(bias, dtype=torch.float32, device=layer.bias.device))

    def get_model_summary(self) -> Dict[str, Any]:
        return {
            "class_name": self.__class__.__name__,
            "problem_type": self.problem_type,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "optimizer_name": self.optimizer_name,
            "loss_name": self.loss_name,
            "device": str(self.device),
            "parameter_count": self.parameter_count(),
            "final_activation": self.final_activation_name,
            "layer_metadata": self.layer_metadata,
        }

    def save_model(self, filepath: Union[str, Path]) -> Path:
        payload = {
            "class_name": self.__class__.__name__,
            "state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "training_history": self.training_history,
            "problem_type": self.problem_type,
            "global_config": {
                "problem_type": self.problem_type,
                "final_activation_dim": self.dim,
                "device": self.device_setting,
                "seed": self.seed,
            },
            "nn_config": {
                "input_dim": self.input_dim,
                "layer_config": self._normalize_layer_config(self.layer_config, component=NETWORK_COMPONENT),
                "initialization_method_default": self.default_init_method,
                "optimizer_name": self.optimizer_name,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "loss_function_name": self.loss_name,
                "clip_grad_norm": self.clip_grad_norm,
                "final_activation": self.final_activation_name,
                "apply_final_activation_in_forward": self.apply_final_activation_in_forward,
                "batch_size": self.batch_size,
                "epochs": self.default_epochs,
                "scheduler": self.nn_config.get("scheduler"),
                "restore_best_weights": self.restore_best_weights,
                "track_history": self.track_history,
            },
            "is_fitted": self._is_fitted,
        }
        path = self._safe_checkpoint_save(payload, filepath, component=NETWORK_COMPONENT)
        logger.info("NeuralNetwork saved to %s", path)
        return path

    @classmethod
    def load_model(cls, filepath: Union[str, Path], map_location: str = "cpu") -> "NeuralNetwork":
        checkpoint = cls._safe_checkpoint_load(filepath, map_location=map_location, component=NETWORK_COMPONENT)
        nn_config_override = checkpoint.get("nn_config")
        global_config_override = checkpoint.get("global_config")
        model = cls(nn_config_override=nn_config_override, global_config_override=global_config_override)
        try:
            model.load_state_dict(checkpoint["state_dict"])
            optimizer_state = checkpoint.get("optimizer_state_dict")
            if optimizer_state is not None:
                model.optimizer.load_state_dict(optimizer_state)
            scheduler_state = checkpoint.get("scheduler_state_dict")
            if model.scheduler is not None and scheduler_state is not None:
                model.scheduler.load_state_dict(scheduler_state)
            model.training_history = checkpoint.get("training_history", model.training_history)
            model._is_fitted = bool(checkpoint.get("is_fitted", False))
            model.to(model.device)
            model.eval()
            return model
        except Exception as exc:
            raise wrap_exception(
                exc,
                CheckpointLoadError,
                "Failed to restore NeuralNetwork state from checkpoint.",
                component=NETWORK_COMPONENT,
                details={"filepath": str(filepath), "class_name": checkpoint.get("class_name")},
            )


class BayesianDQN(NeuralNetwork):
    """
    Bayesian DQN wrapper with Monte Carlo dropout uncertainty estimation.

    Design choices:
    - keeps deterministic inference unless an uncertainty method is explicitly invoked
    - reuses NeuralNetwork persistence/training utilities
    - config stays in adaptive_config.yaml under bayesian_dqn
    """

    def __init__(
        self,
        nn_config_override: Optional[Dict[str, Any]] = None,
        bayesian_config_override: Optional[Dict[str, Any]] = None,
        global_config_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(nn_config_override=nn_config_override, global_config_override=global_config_override)
        self.dqn_config = get_config_section("bayesian_dqn")
        if bayesian_config_override:
            self.dqn_config.update(bayesian_config_override)

        self.dropout_rate = float(self.dqn_config.get("dropout_rate", self.nn_config.get("default_dropout_rate", 0.1)))
        self.num_uncertainty_samples = int(self.dqn_config.get("num_uncertainty_samples", 20))
        self.uncertainty_threshold = float(self.dqn_config.get("uncertainty_threshold", 0.1))

        ensure_in_range(self.dropout_rate, "bayesian_dqn.dropout_rate", minimum=0.0, maximum=1.0, component=BAYESIAN_COMPONENT)
        ensure_positive(self.num_uncertainty_samples, "bayesian_dqn.num_uncertainty_samples", component=BAYESIAN_COMPONENT)
        ensure_in_range(self.uncertainty_threshold, "bayesian_dqn.uncertainty_threshold", minimum=0.0, component=BAYESIAN_COMPONENT)

        self._configure_dropout_layers(self.dropout_rate)
        logger.info(
            "BayesianDQN initialized | dropout_rate=%s | uncertainty_samples=%s | threshold=%s",
            self.dropout_rate,
            self.num_uncertainty_samples,
            self.uncertainty_threshold,
        )

    def _iter_dropout_modules(self) -> Iterable[nn.Dropout]:
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                yield module

    def _configure_dropout_layers(self, dropout_rate: float) -> None:
        for module in self._iter_dropout_modules():
            module.p = dropout_rate

    def _set_dropout_mode(self, enabled: bool) -> List[Tuple[nn.Dropout, bool]]:
        previous_states: List[Tuple[nn.Dropout, bool]] = []
        for module in self._iter_dropout_modules():
            previous_states.append((module, module.training))
            module.train(enabled)
        return previous_states

    @staticmethod
    def _restore_dropout_mode(previous_states: List[Tuple[nn.Dropout, bool]]) -> None:
        for module, was_training in previous_states:
            module.train(was_training)

    def _mc_predictions(self, state: Union[np.ndarray, torch.Tensor, Sequence[float]], num_samples: Optional[int] = None) -> np.ndarray:
        sample_count = int(num_samples or self.num_uncertainty_samples)
        ensure_positive(sample_count, "num_samples", component=BAYESIAN_COMPONENT)
        inputs = self._prepare_input_tensor(state, expected_dim=self.input_dim, component=BAYESIAN_COMPONENT)

        self.eval()
        previous_states = self._set_dropout_mode(True)
        predictions: List[torch.Tensor] = []
        try:
            with torch.no_grad():
                for _ in range(sample_count):
                    predictions.append(self.predict_tensor(inputs, apply_output_activation=False))
            stacked = torch.stack(predictions, dim=0)
            return stacked.detach().cpu().numpy()
        finally:
            self._restore_dropout_mode(previous_states)

    def estimate_uncertainty(self, state: Union[np.ndarray, torch.Tensor, Sequence[float]], num_samples: Optional[int] = None) -> np.ndarray:
        printer.status("UNCERTAINTY", "Estimating Bayesian uncertainty", "info")
        predictions = self._mc_predictions(state, num_samples=num_samples)
        std_dev = predictions.std(axis=0)
        return np.clip(std_dev, 0.0, self.uncertainty_threshold) if self.uncertainty_threshold > 0 else std_dev

    def get_uncertainty_metrics(self, state: Union[np.ndarray, torch.Tensor, Sequence[float]], num_samples: Optional[int] = None) -> Dict[str, Any]:
        predictions = self._mc_predictions(state, num_samples=num_samples)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        variance = predictions.var(axis=0)
        confidence_radius = 1.96 * std / np.sqrt(predictions.shape[0])
        max_uncertainty = float(np.max(std)) if std.size > 0 else 0.0
        return {
            "mean": mean,
            "std": std,
            "variance": variance,
            "confidence_interval": (mean - confidence_radius, mean + confidence_radius),
            "max_uncertainty": max_uncertainty,
            "uncertainty_flag": bool(max_uncertainty > self.uncertainty_threshold),
            "samples": int(predictions.shape[0]),
        }

    def predict_with_uncertainty(
        self,
        inputs: Union[np.ndarray, torch.Tensor, Sequence[float]],
        num_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        prediction = self.predict(inputs, apply_output_activation=True)
        uncertainty = self.estimate_uncertainty(inputs, num_samples=num_samples)
        return prediction, uncertainty

    def select_action(self, state: Union[np.ndarray, torch.Tensor, Sequence[float]], return_uncertainty: bool = False) -> Union[int, Tuple[int, Dict[str, Any]]]:
        metrics = self.get_uncertainty_metrics(state)
        mean_q = metrics["mean"]
        action = int(np.argmax(mean_q, axis=-1).reshape(-1)[0])
        return (action, metrics) if return_uncertainty else action

    def save_model(self, filepath: Union[str, Path]) -> Path:
        payload = {
            "class_name": self.__class__.__name__,
            "state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "training_history": self.training_history,
            "global_config": {
                "problem_type": self.problem_type,
                "final_activation_dim": self.dim,
                "device": self.device_setting,
                "seed": self.seed,
            },
            "nn_config": {
                "input_dim": self.input_dim,
                "layer_config": self._normalize_layer_config(self.layer_config, component=NETWORK_COMPONENT),
                "initialization_method_default": self.default_init_method,
                "optimizer_name": self.optimizer_name,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "loss_function_name": self.loss_name,
                "clip_grad_norm": self.clip_grad_norm,
                "final_activation": self.final_activation_name,
                "apply_final_activation_in_forward": self.apply_final_activation_in_forward,
                "batch_size": self.batch_size,
                "epochs": self.default_epochs,
                "scheduler": self.nn_config.get("scheduler"),
                "restore_best_weights": self.restore_best_weights,
                "track_history": self.track_history,
            },
            "bayesian_config": {
                "dropout_rate": self.dropout_rate,
                "num_uncertainty_samples": self.num_uncertainty_samples,
                "uncertainty_threshold": self.uncertainty_threshold,
            },
            "is_fitted": self._is_fitted,
        }
        path = self._safe_checkpoint_save(payload, filepath, component=BAYESIAN_COMPONENT)
        logger.info("BayesianDQN saved to %s", path)
        return path

    @classmethod
    def load_model(cls, filepath: Union[str, Path], map_location: str = "cpu") -> "BayesianDQN":
        checkpoint = cls._safe_checkpoint_load(filepath, map_location=map_location, component=BAYESIAN_COMPONENT)
        model = cls(
            nn_config_override=checkpoint.get("nn_config"),
            bayesian_config_override=checkpoint.get("bayesian_config"),
            global_config_override=checkpoint.get("global_config"),
        )
        try:
            model.load_state_dict(checkpoint["state_dict"])
            optimizer_state = checkpoint.get("optimizer_state_dict")
            if optimizer_state is not None:
                model.optimizer.load_state_dict(optimizer_state)
            scheduler_state = checkpoint.get("scheduler_state_dict")
            if model.scheduler is not None and scheduler_state is not None:
                model.scheduler.load_state_dict(scheduler_state)
            model.training_history = checkpoint.get("training_history", model.training_history)
            model._is_fitted = bool(checkpoint.get("is_fitted", False))
            model.to(model.device)
            model.eval()
            return model
        except Exception as exc:
            raise wrap_exception(
                exc,
                CheckpointLoadError,
                "Failed to restore BayesianDQN state from checkpoint.",
                component=BAYESIAN_COMPONENT,
                details={"filepath": str(filepath), "class_name": checkpoint.get("class_name")},
            )


class ActorCriticNetwork(AdaptiveNetworkBase):
    """
    Shared actor-critic network for adaptive skill workers.

    Integration goals:
    - maintains the constructor used by reinforcement_learning.SkillWorker
    - supports discrete and continuous policies
    - supports optional shared bases
    - ensures actor output matches action_dim and critic output matches scalar value prediction
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_layers: Union[List[int], List[Dict[str, Any]]],
        critic_layers: Union[List[int], List[Dict[str, Any]]],
        acn_config_override: Optional[Dict[str, Any]] = None,
        global_config_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if global_config_override:
            self.config.update(global_config_override)
            self.problem_type = str(self.config.get("problem_type", "regression")).lower()
            self.dim = int(self.config.get("final_activation_dim", -1))
            self.device_setting = self.config.get("device", self.device_setting)
            self.device = self._resolve_device(self.device_setting)

        self.acn_config = get_config_section("actor_critic")
        if acn_config_override:
            self.acn_config.update(acn_config_override)

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        ensure_positive(self.state_dim, "actor_critic.state_dim", component=ACTOR_CRITIC_COMPONENT)
        ensure_positive(self.action_dim, "actor_critic.action_dim", component=ACTOR_CRITIC_COMPONENT)

        self.default_init_method = str(self.acn_config.get("initialization_method_default", "he_normal")).lower()
        self.shared_base = bool(self.acn_config.get("shared_base", False))
        self.continuous_action = bool(self.acn_config.get("continuous_action", False))
        self.initial_std = float(self.acn_config.get("initial_std", 0.5))
        self.min_std = float(self.acn_config.get("min_std", 1e-4))
        self.max_std = float(self.acn_config.get("max_std", 10.0))
        self.actor_output_activation = str(self.acn_config.get("actor_output_activation", "linear")).lower()
        self.critic_output_activation = str(self.acn_config.get("critic_output_activation", "linear")).lower()
        self.shared_output_activation = str(self.acn_config.get("shared_output_activation", "relu")).lower()

        ensure_positive(self.initial_std, "actor_critic.initial_std", component=ACTOR_CRITIC_COMPONENT)
        ensure_positive(self.min_std, "actor_critic.min_std", component=ACTOR_CRITIC_COMPONENT)
        ensure_positive(self.max_std, "actor_critic.max_std", component=ACTOR_CRITIC_COMPONENT)
        ensure(self.max_std >= self.min_std, "actor_critic.max_std must be >= actor_critic.min_std.", exc_type=InvalidConfigurationValueError, component=ACTOR_CRITIC_COMPONENT)

        normalized_actor_layers = self._normalize_layer_config(actor_layers, component=ACTOR_CRITIC_COMPONENT)
        normalized_critic_layers = self._normalize_layer_config(critic_layers, component=ACTOR_CRITIC_COMPONENT)
        self.actor_layers = self._force_output_dim(
            normalized_actor_layers,
            self.action_dim,
            component=ACTOR_CRITIC_COMPONENT,
            output_activation=self.actor_output_activation,
        )
        self.critic_layers = self._force_output_dim(
            normalized_critic_layers,
            1,
            component=ACTOR_CRITIC_COMPONENT,
            output_activation=self.critic_output_activation,
        )

        shared_layers_raw = self.acn_config.get("shared_layers", [])
        self.shared_layers = (
            self._normalize_layer_config(shared_layers_raw, component=ACTOR_CRITIC_COMPONENT)
            if self.shared_base and shared_layers_raw
            else []
        )

        if self.shared_base and not self.shared_layers:
            raise NetworkConfigurationError(
                "actor_critic.shared_base=True requires a non-empty shared_layers config.",
                component=ACTOR_CRITIC_COMPONENT,
                remediation="Populate actor_critic.shared_layers in adaptive_config.yaml.",
            )

        if self.shared_base:
            self.base_network, self.base_metadata = self._build_mlp(
                self.state_dim,
                self.shared_layers,
                component=ACTOR_CRITIC_COMPONENT,
                default_init_method=self.default_init_method,
                activate_last_layer=True,
                default_activation=self.shared_output_activation,
                return_metadata=True,
            )
            base_output_dim = self.shared_layers[-1]["neurons"]
            self.actor, self.actor_metadata = self._build_mlp(
                base_output_dim,
                self.actor_layers,
                component=ACTOR_CRITIC_COMPONENT,
                default_init_method=self.default_init_method,
                activate_last_layer=False,
                return_metadata=True,
            )
            self.critic, self.critic_metadata = self._build_mlp(
                base_output_dim,
                self.critic_layers,
                component=ACTOR_CRITIC_COMPONENT,
                default_init_method=self.default_init_method,
                activate_last_layer=False,
                return_metadata=True,
            )
        else:
            self.base_network = None
            self.base_metadata = []
            self.actor, self.actor_metadata = self._build_mlp(
                self.state_dim,
                self.actor_layers,
                component=ACTOR_CRITIC_COMPONENT,
                default_init_method=self.default_init_method,
                activate_last_layer=False,
                return_metadata=True,
            )
            self.critic, self.critic_metadata = self._build_mlp(
                self.state_dim,
                self.critic_layers,
                component=ACTOR_CRITIC_COMPONENT,
                default_init_method=self.default_init_method,
                activate_last_layer=False,
                return_metadata=True,
            )

        self.action_std: Optional[nn.Parameter] = None
        if self.continuous_action:
            self.action_std = nn.Parameter(torch.full((1, self.action_dim), self.initial_std, dtype=torch.float32))

        self.to(self.device)
        logger.info(
            "ActorCriticNetwork initialized | state_dim=%s | action_dim=%s | shared_base=%s | continuous_action=%s | device=%s",
            self.state_dim,
            self.action_dim,
            self.shared_base,
            self.continuous_action,
            self.device,
        )

    def _prepare_state_tensor(self, state: Union[np.ndarray, torch.Tensor, Sequence[float]], *, require_batch: bool = True) -> torch.Tensor:
        tensor = self._prepare_input_tensor(state, expected_dim=self.state_dim, component=ACTOR_CRITIC_COMPONENT)
        if require_batch and tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _forward_base(self, x: Union[np.ndarray, torch.Tensor, Sequence[float]]) -> torch.Tensor:
        tensor = self._prepare_state_tensor(x)
        if self.shared_base and self.base_network is not None:
            return self.base_network(tensor)
        return tensor

    def forward_actor(self, x: Union[np.ndarray, torch.Tensor, Sequence[float]]) -> torch.Tensor:
        features = self._forward_base(x)
        return self.actor(features)

    def forward_critic(self, x: Union[np.ndarray, torch.Tensor, Sequence[float]]) -> torch.Tensor:
        features = self._forward_base(x)
        return self.critic(features)

    def forward(self, x: Union[np.ndarray, torch.Tensor, Sequence[float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self._forward_base(x)
        return self.actor(features), self.critic(features)

    def _continuous_distribution(self, actor_output: torch.Tensor) -> torch.distributions.Normal:
        ensure_not_none(self.action_std, "action_std", component=ACTOR_CRITIC_COMPONENT)
        clamped_std = torch.clamp(self.action_std, min=self.min_std, max=self.max_std)
        action_std = clamped_std.expand_as(actor_output)
        return torch.distributions.Normal(actor_output, action_std)

    def get_action(self, state: Union[np.ndarray, torch.Tensor, Sequence[float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        actor_output = self.forward_actor(state)
        if self.continuous_action:
            dist = self._continuous_distribution(actor_output)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            return action, log_prob

        dist = torch.distributions.Categorical(logits=actor_output)
        action = dist.sample()
        return action, dist.log_prob(action)

    def get_value(self, state: Union[np.ndarray, torch.Tensor, Sequence[float]]) -> torch.Tensor:
        return self.forward_critic(state)

    def evaluate_actions(
        self,
        states: Union[np.ndarray, torch.Tensor, Sequence[float]],
        actions: Union[np.ndarray, torch.Tensor, Sequence[float]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states_tensor = self._prepare_state_tensor(states)
        actor_out, critic_out = self.forward(states_tensor)
        actions_tensor = torch.as_tensor(actions, device=self.device)

        if self.continuous_action:
            actions_tensor = actions_tensor.float()
            if actions_tensor.ndim == 1:
                actions_tensor = actions_tensor.unsqueeze(0)
            dist = self._continuous_distribution(actor_out)
            log_probs = dist.log_prob(actions_tensor).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1).mean()
        else:
            actions_tensor = actions_tensor.long().view(-1)
            dist = torch.distributions.Categorical(logits=actor_out)
            log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()

        return log_probs, critic_out.squeeze(-1), entropy

    def get_actor_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        seen: set[int] = set()

        def add_from(module: Optional[nn.Module]) -> None:
            if module is None:
                return
            for parameter in module.parameters():
                ident = id(parameter)
                if ident not in seen:
                    params.append(parameter)
                    seen.add(ident)

        if self.shared_base and getattr(self, "base_network", None) is not None:
            add_from(self.base_network)
        add_from(self.actor)
        if getattr(self, "action_std", None) is not None:
            ident = id(self.action_std)
            if ident not in seen:
                params.append(self.action_std)
                seen.add(ident)
        return params

    def get_critic_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        seen: set[int] = set()

        def add_from(module: Optional[nn.Module]) -> None:
            if module is None:
                return
            for parameter in module.parameters():
                ident = id(parameter)
                if ident not in seen:
                    params.append(parameter)
                    seen.add(ident)

        if self.shared_base and getattr(self, "base_network", None) is not None:
            add_from(self.base_network)
        add_from(self.critic)
        return params

    def get_model_summary(self) -> Dict[str, Any]:
        return {
            "class_name": self.__class__.__name__,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "shared_base": self.shared_base,
            "continuous_action": self.continuous_action,
            "device": str(self.device),
            "parameter_count": self.parameter_count(),
            "actor_layers": self.actor_layers,
            "critic_layers": self.critic_layers,
            "shared_layers": self.shared_layers,
        }

    def save_model(self, filepath: Union[str, Path]) -> Path:
        payload = {
            "class_name": self.__class__.__name__,
            "state_dict": self.state_dict(),
            "global_config": {
                "problem_type": self.problem_type,
                "final_activation_dim": self.dim,
                "device": self.device_setting,
                "seed": self.seed,
            },
            "actor_critic_config": {
                "shared_base": self.shared_base,
                "shared_layers": self.shared_layers,
                "continuous_action": self.continuous_action,
                "initial_std": self.initial_std,
                "min_std": self.min_std,
                "max_std": self.max_std,
                "initialization_method_default": self.default_init_method,
                "actor_output_activation": self.actor_output_activation,
                "critic_output_activation": self.critic_output_activation,
                "shared_output_activation": self.shared_output_activation,
            },
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "actor_layers": self.actor_layers,
            "critic_layers": self.critic_layers,
        }
        path = self._safe_checkpoint_save(payload, filepath, component=ACTOR_CRITIC_COMPONENT)
        logger.info("ActorCriticNetwork saved to %s", path)
        return path

    @classmethod
    def load_model(cls, filepath: Union[str, Path], map_location: str = "cpu") -> "ActorCriticNetwork":
        checkpoint = cls._safe_checkpoint_load(filepath, map_location=map_location, component=ACTOR_CRITIC_COMPONENT)
        model = cls(
            state_dim=int(checkpoint["state_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            actor_layers=checkpoint["actor_layers"],
            critic_layers=checkpoint["critic_layers"],
            acn_config_override=checkpoint.get("actor_critic_config"),
            global_config_override=checkpoint.get("global_config"),
        )
        try:
            model.load_state_dict(checkpoint["state_dict"])
            model.to(model.device)
            model.eval()
            return model
        except Exception as exc:
            raise wrap_exception(
                exc,
                CheckpointLoadError,
                "Failed to restore ActorCriticNetwork state from checkpoint.",
                component=ACTOR_CRITIC_COMPONENT,
                details={"filepath": str(filepath), "class_name": checkpoint.get("class_name")},
            )


def _build_synthetic_dataset(model: NeuralNetwork, sample_count: int) -> List[Tuple[np.ndarray, Any]]:
    dataset: List[Tuple[np.ndarray, Any]] = []
    for _ in range(sample_count):
        x = np.random.randn(model.input_dim).astype(np.float32)
        if model.problem_type == "regression":
            y = np.random.randn(model.output_dim).astype(np.float32)
        elif model.problem_type == "binary_classification":
            if model.output_dim == 1:
                y = np.array([np.random.randint(0, 2)], dtype=np.float32)
            else:
                y = np.eye(model.output_dim, dtype=np.float32)[np.random.randint(0, model.output_dim)]
        elif model.problem_type == "multiclass_classification":
            y = int(np.random.randint(0, model.output_dim))
        else:
            raise InvalidConfigurationValueError(
                f"Unsupported problem_type '{model.problem_type}' for synthetic test data.",
                component=NETWORK_COMPONENT,
            )
        dataset.append((x, y))
    return dataset


if __name__ == "__main__":
    print("\n=== Running Neural Network ===\n")
    printer.status("TEST", "Neural Network initialized", "info")

    try:
        network = NeuralNetwork()
        printer.status("TEST", f"NeuralNetwork summary: {network.get_model_summary()}", "success")

        test_input = np.random.randn(network.input_dim).astype(np.float32)
        forward_output = network.forward(test_input)
        printer.status("TEST", f"Forward output shape: {tuple(forward_output.shape)}", "success")

        training_data = _build_synthetic_dataset(network, max(16, network.batch_size * 2))
        validation_data = _build_synthetic_dataset(network, max(8, network.batch_size))
        train_result = network.train_network(
            training_data=training_data,
            epochs=min(3, network.default_epochs),
            batch_size=min(network.batch_size, len(training_data)),
            validation_data=validation_data,
            early_stopping_patience=2,
            verbose=True,
        )
        printer.status("TEST", f"Training result: {train_result}", "success")

        eval_loss = network.evaluate(validation_data)
        predictions = network.predict(np.stack([row[0] for row in validation_data[:4]]))
        classes = network.predict_class(np.stack([row[0] for row in validation_data[:4]]))
        weights_snapshot = network.get_weights_biases()
        network.set_weights_biases(weights_snapshot)
        printer.status("TEST", f"Evaluation loss: {eval_loss:.6f}", "success")
        printer.status("TEST", f"Prediction shape: {predictions.shape} | Class output shape: {np.asarray(classes).shape}", "success")

        bayesian_network = BayesianDQN()
        bayesian_forward = bayesian_network.forward(test_input)
        mean_prediction, uncertainty = bayesian_network.predict_with_uncertainty(np.stack([test_input, test_input]))
        uncertainty_metrics = bayesian_network.get_uncertainty_metrics(test_input)
        selected_action = bayesian_network.select_action(test_input)
        printer.status("TEST", f"Bayesian forward shape: {tuple(bayesian_forward.shape)}", "success")
        printer.status(
            "TEST",
            f"Bayesian prediction shape: {mean_prediction.shape} | uncertainty shape: {uncertainty.shape} | selected_action: {selected_action}",
            "success",
        )
        printer.status("TEST", f"Bayesian uncertainty metrics keys: {list(uncertainty_metrics.keys())}", "success")

        actor_layers = [64, network.output_dim if network.problem_type != "multiclass_classification" else max(2, network.output_dim)]
        critic_layers = [64, 1]
        actor_critic = ActorCriticNetwork(
            state_dim=network.input_dim,
            action_dim=max(2, network.output_dim),
            actor_layers=actor_layers,
            critic_layers=critic_layers,
        )
        state_batch = np.random.randn(4, actor_critic.state_dim).astype(np.float32)
        actor_logits = actor_critic.forward_actor(state_batch)
        critic_values = actor_critic.forward_critic(state_batch)
        action, log_prob = actor_critic.get_action(state_batch[0])
        actions = np.random.randint(0, actor_critic.action_dim, size=(4,))
        log_probs, values, entropy = actor_critic.evaluate_actions(state_batch, actions)
        printer.status("TEST", f"Actor logits shape: {tuple(actor_logits.shape)} | Critic values shape: {tuple(critic_values.shape)}", "success")
        printer.status("TEST", f"Sampled action: {action} | log_prob: {log_prob}", "success")
        printer.status(
            "TEST",
            f"Evaluate actions -> log_probs shape: {tuple(log_probs.shape)} | values shape: {tuple(values.shape)} | entropy: {float(entropy):.6f}",
            "success",
        )

        checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        nn_ckpt = checkpoint_dir / "neural_network_test.pt"
        bnn_ckpt = checkpoint_dir / "bayesian_dqn_test.pt"
        ac_ckpt = checkpoint_dir / "actor_critic_test.pt"

        network.save_model(nn_ckpt)
        bayesian_network.save_model(bnn_ckpt)
        actor_critic.save_model(ac_ckpt)

        loaded_network = NeuralNetwork.load_model(nn_ckpt)
        loaded_bayesian = BayesianDQN.load_model(bnn_ckpt)
        loaded_actor_critic = ActorCriticNetwork.load_model(ac_ckpt)
        printer.status("TEST", f"Loaded NeuralNetwork params: {loaded_network.parameter_count()}", "success")
        printer.status("TEST", f"Loaded BayesianDQN params: {loaded_bayesian.parameter_count()}", "success")
        printer.status("TEST", f"Loaded ActorCritic params: {loaded_actor_critic.parameter_count()}", "success")

        print("\n=== Test ran successfully ===\n")
    except Exception as exc:
        logger.error("Neural network test block failed: %s", exc, exc_info=True)
        raise
