from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from src.utils.config_loader import ConfigLoader


def test_config_loader_uses_native_merge_and_preserves_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "safety:\n"
        "  risk_threshold: 0.6\n"
        "telemetry:\n"
        "  custom_label: audit\n",
        encoding="utf-8",
    )
    original_defaults = deepcopy(ConfigLoader._DEFAULT_CONFIG)

    with ConfigLoader(str(config_path)) as config:
        assert config.get("safety.risk_threshold") == 0.6
        assert config.get("safety.max_throughput") == 1000
        assert config.get("telemetry.metrics_collection") is True
        assert config.get("telemetry.custom_label") == "audit"

    assert ConfigLoader._DEFAULT_CONFIG == original_defaults


def test_missing_config_returns_detached_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "missing.yaml"

    with ConfigLoader(str(config_path)) as config:
        config._config["safety"]["risk_threshold"] = 0.9

    assert ConfigLoader._DEFAULT_CONFIG["safety"]["risk_threshold"] == 0.35
