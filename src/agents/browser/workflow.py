from __future__ import annotations

from typing import Any, Dict, List


class WorkFlow:
    """Workflow utility for browser orchestration.

    This module intentionally stays thin and browser-driver agnostic. The BrowserAgent
    owns concrete execution while this helper normalizes workflow steps.
    """

    SUPPORTED_ACTIONS = {
        "navigate",
        "search",
        "click",
        "type",
        "scroll",
        "copy",
        "cut",
        "paste",
        "extract",
    }

    def normalize(self, workflow_script: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for index, step in enumerate(workflow_script or []):
            action = (step or {}).get("action", "").lower().strip()
            params = (step or {}).get("params", {})
            if action not in self.SUPPORTED_ACTIONS:
                raise ValueError(f"Unsupported workflow action at index {index}: '{action}'")
            normalized.append({"action": action, "params": params})
        return normalized
