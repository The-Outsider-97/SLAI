from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Set

from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Config Contract")
printer = PrettyPrinter()


@dataclass(frozen=True)
class ConfigValidationResult:
    valid: bool
    errors: Sequence[str]
    warnings: Sequence[str]


class ConfigContractError(ValueError):
    """Raised when an agent configuration violates the shared contract."""


GLOBAL_REQUIRED_TOP_LEVEL_KEYS: Set[str] = {
    "agent_factory",
    "base_agent",
}

GLOBAL_ALLOWED_EXTENSION_TOP_LEVEL_KEYS: Set[str] = {
    "sensitive_attributes",
    "issue_database",
    "operations_notification",
}


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _normalize_allowed_keys(allowed_keys: Optional[Iterable[str]]) -> Optional[Set[str]]:
    if allowed_keys is None:
        return None
    return {str(key).strip() for key in allowed_keys if str(key).strip()}


def validate_config_contract(
    *,
    global_config: Mapping[str, Any],
    agent_key: str,
    agent_config: Mapping[str, Any],
    agent_allowed_keys: Optional[Iterable[str]] = None,
    required_agent_keys: Optional[Iterable[str]] = None,
    require_global_keys: bool = True,
    require_agent_section: bool = True,
    warn_unknown_global_keys: bool = True,
) -> ConfigValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not _is_mapping(global_config):
        return ConfigValidationResult(False, ["Global config must be a mapping/dict."], warnings)

    if require_global_keys:
        missing_global_keys = sorted(k for k in GLOBAL_REQUIRED_TOP_LEVEL_KEYS if k not in global_config)
        if missing_global_keys:
            errors.append(f"Missing required global config keys: {missing_global_keys}")

    if require_agent_section and agent_key not in global_config:
        errors.append(f"Missing required agent section '{agent_key}' in global config")

    if not _is_mapping(agent_config):
        errors.append(f"Agent config for '{agent_key}' must be a mapping/dict")
        return ConfigValidationResult(False, errors, warnings)

    if warn_unknown_global_keys:
        allowed_global = set(GLOBAL_REQUIRED_TOP_LEVEL_KEYS)
        allowed_global.update(GLOBAL_ALLOWED_EXTENSION_TOP_LEVEL_KEYS)
        allowed_global.add(agent_key)

        unknown_global = sorted(
            k for k in global_config.keys()
            if k not in allowed_global and not str(k).startswith("__")
        )
        if unknown_global:
            warnings.append(
                f"Global config has additional top-level keys outside strict contract for '{agent_key}': {unknown_global}"
            )

    normalized_allowed = _normalize_allowed_keys(agent_allowed_keys)
    if normalized_allowed is not None:
        unknown_agent_keys = sorted(k for k in agent_config.keys() if k not in normalized_allowed)
        if unknown_agent_keys:
            errors.append(f"Unknown keys in '{agent_key}' config: {unknown_agent_keys}")

    if required_agent_keys:
        missing_required_agent_keys = sorted(k for k in required_agent_keys if k not in agent_config)
        if missing_required_agent_keys:
            errors.append(f"Missing required keys in '{agent_key}' config: {missing_required_agent_keys}")

    return ConfigValidationResult(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def assert_valid_config_contract(
    *,
    global_config: Mapping[str, Any],
    agent_key: str,
    agent_config: Mapping[str, Any],
    agent_allowed_keys: Optional[Iterable[str]] = None,
    required_agent_keys: Optional[Iterable[str]] = None,
    require_global_keys: bool = True,
    require_agent_section: bool = True,
    warn_unknown_global_keys: bool = True,
    logger: Any = None,
) -> None:
    result = validate_config_contract(
        global_config=global_config,
        agent_key=agent_key,
        agent_config=agent_config,
        agent_allowed_keys=agent_allowed_keys,
        required_agent_keys=required_agent_keys,
        require_global_keys=require_global_keys,
        require_agent_section=require_agent_section,
        warn_unknown_global_keys=warn_unknown_global_keys,
    )

    if logger:
        for warning in result.warnings:
            logger.warning("[config_contract] %s", warning)

    if not result.valid:
        raise ConfigContractError("; ".join(result.errors))
