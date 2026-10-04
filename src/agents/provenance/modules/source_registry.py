from __future__ import annotations

from typing import Any, Optional

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.provenance_errors import *
from ..utils.provenance_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Source Registry")
printer = PrettyPrinter()


class SourceRegistry:
    def __init__(self):
        self.config = load_global_config()
        self.source_registry_config = get_config_section('source_registry')

        logger.info(f"SourceRegistry initialized with config: {self.source_registry_config}")

    def register_source(self, source_id: str, source_info: dict) -> None:
        """
        Register a new source in the source registry.

        Args:
            source_id (str): The unique identifier of the source.
            source_info (dict): A dictionary containing information about the source.
        """
        # Implementation for registering a new source
        raise NotImplementedError("Source registration is not implemented yet.")


__all__ = ["SourceRegistry"]