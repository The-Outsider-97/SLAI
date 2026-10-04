from __future__ import annotations

from typing import Any, Optional

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.provenance_errors import *
from ..utils.provenance_helpers import *
from .base_lineage import BaseLineage
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Model Lineage")
printer = PrettyPrinter()


class ModelLineage(BaseLineage):
    def __init__(self, config: Optional[Any] = None):
        super().__init__(config=config)
        self.config = load_global_config()
        self.model_lineage_config = get_config_section('model_lineage')

        logger.info(f"ModelLineage initialized with config: {self.model_lineage_config}")

    def record_model_lineage(self, model_id: str, parent_model_id: str, transformation: str, timestamp: Optional[str] = None):
        """
        Record the lineage of a model.

        Args:
            model_id (str): The unique identifier of the model.
            parent_model_id (str): The unique identifier of the parent model.
            transformation (str): Description of the transformation applied to the parent model.
            timestamp (Optional[str]): The timestamp of when the lineage was recorded. If None, current time is used.

        Returns:
            None
        """
        # Implementation for recording model lineage
        raise NotImplementedError("Model lineage recording is not implemented yet.")


__all__ = ["ModelLineage"]