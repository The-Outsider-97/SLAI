"""SPlanner settings, loaded via SLAI's shared configuration repository."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.utils.configuration import ConfigRepository


def default_data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return base / "SLAI" / "SPlanner"
    base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return base / "slai" / "splanner"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    port: int = 8769
    reminder_poll_seconds: int = 20
    max_request_bytes: int = 262144

    @classmethod
    def load(cls, *, config_path: Path | None = None, data_dir: Path | None = None,
             port: int | None = None) -> "Settings":
        source = config_path or Path(__file__).with_name("config.yaml")
        config = ConfigRepository().load(source)
        app = config.get("app", {})
        if not isinstance(app, dict):
            raise ValueError("Configuration section 'app' must be a mapping")
        resolved_port = port if port is not None else app.get("port", 8769)
        interval = app.get("reminder_poll_seconds", 20)
        size = app.get("max_request_bytes", 262144)
        for label, value, low, high in (("port", resolved_port, 1, 65535),
                                        ("reminder_poll_seconds", interval, 5, 3600),
                                        ("max_request_bytes", size, 4096, 1048576)):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid SPlanner configuration: {label}")
        return cls(data_dir=(data_dir or default_data_dir()).expanduser().resolve(),
                   port=resolved_port, reminder_poll_seconds=interval,
                   max_request_bytes=size)
