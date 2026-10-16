"""
Lazy public exports for Base Agent modules.
Importing this package does not initialize neural backends.
Neural dependencies are resolved only when their exported features are requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_GROUPS = {

    ".activation_engine": (
        "Activation",
        "ReLU",
        "Sigmoid",
        "Tanh",
        "Softmax",
        "LeakyReLU",
        "ELU",
        "Linear",
        "Swish",
        "SiLU",
        "GELU",
        "Mish",
        "get_activation",
        "gelu_tensor",
        "swish_tensor",
        "mish_tensor",
        "sigmoid_tensor",
        "he_init",
        "lecun_normal",
        "xavier_uniform",
        "xavier_normal",
    ),

    ".base_engineering": (
        "MedicalEngineering",
        "CivilEngineering",
        "ComputerEngineering",
        "ElectronicsCommsEngineering",
        "MechanicalEngineering",
        "EngineeringEngine",
        "get_engineering_engine",
    ),

    ".base_tokenizer": (
        "BaseTokenizer",
        "TokenizerStats",
        "TrainingSummary",
    ),

    ".base_transformer": (
        "BaseTransformer",
        "TransformerConfig",
        "TransformerRunStats",
        "GenerationOutput",
        "PositionalEncoding",
    ),

    ".biology_constraints": (
        "BiologyConfig",
        "BiologyStepSummary",
        "BiologyEngine",
        "apply_biological_constants",
        "apply_biological_processes",
        "enforce_biological_constraints",
        "apply_all_biological_constraints",
    ),

    ".chemistry_constraints": (
        "ChemistryConfig",
        "ChemistryStepSummary",
        "ChemistryEngine",
        "apply_chemical_constants",
        "apply_chemical_processes",
        "enforce_chemical_constraints",
        "apply_all_chemical_constraints",
    ),

    ".physics_constraints": (
        "PhysicsStepSummary",
        "PhysicsEngine",
        "apply_constants",
        "apply_environmental_effects",
        "enforce_physics_constraints",
        "apply_all_physics_constraints",
    ),

    ".input_sanitizer": (
        "SanitizationRecord",
        "InputSanitizerStats",
        "InputSanitizer",
    ),

    ".numpy_encoder": (
        "NumpyEncoder",
        "NumpyEncodingRecord",
        "NumpyEncoderStats",
    ),
}


_EXPORTS = {
    name: (module_name, name)
    for module_name, names in _EXPORT_GROUPS.items()
    for name in names
}


__all__ = sorted(_EXPORTS) # type: ignore


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(
            f"module {__name__!r} "
            f"has no attribute {name!r}"
        )

    module_name, attribute_name = target
    try:
        module = import_module(module_name, __name__)

    except ModuleNotFoundError as exc:

        missing = exc.name or ""
        if (missing == "torch" or missing.startswith("torch.")):
            raise ModuleNotFoundError(f"{name} requires PyTorch. Install the neural dependency profile before using this feature.") from exc

        raise

    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
