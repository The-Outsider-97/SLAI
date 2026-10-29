"""SLAI v2.4 Animation Agent: cautious SGEO-backed orchestration foundation."""

from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Mapping
from typing import Any

from .base_agent import BaseAgent
from .base.utils.main_config_loader import get_config_section
from .animations.animation_types import AnimationRequest
from .animations.animation_evaluation import evaluate_animation
from .animations.animation_commands import update_animation
from .animations.animation_memory import AnimationMemory, AnimationEditRecord
from .animations.utils.animation_errors import AnimationUnsupportedOperationError
from logs.logger import get_logger, PrettyPrinter # type: ignore

logger = get_logger("Animation Agent")
printer = PrettyPrinter()


class AnimationAgent(BaseAgent):
    AGENT_KEY = "animation_agent"

    def __init__(self, shared_memory: Any = None, agent_factory: Any = None,
                 config: Mapping[str, Any] | None = None, *, checkpoint_manager: Any = None):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, checkpoint_manager=checkpoint_manager)
        self.agent_config = dict(get_config_section(self.AGENT_KEY) or {})
        if config:
            self.agent_config.update(dict(config))
        self._scenes: dict[str, Any] = {}
        self.animation_memory = AnimationMemory(int(self.agent_config.get("max_recent_edits", 256)))

    def bind_scene(self, scene: Any) -> str:
        if not all(hasattr(scene, attr) for attr in ("document_id", "revision", "metadata")):
            raise TypeError("scene must expose document_id, revision and metadata")
        if not callable(getattr(scene, "evaluate", None)) or not callable(getattr(scene, "set_animation", None)):
            raise TypeError("scene does not implement SGEO animation protocol")
        self._scenes[scene.document_id] = scene
        return scene.document_id

    def perform_task(self, task_data: Any) -> dict[str, Any]:
        if not isinstance(task_data, Mapping):
            raise TypeError("animation request must be a mapping")
        req = AnimationRequest(**{k: v for k, v in task_data.items() if k in AnimationRequest.__dataclass_fields__})
        if req.scene_id is None:
            raise AnimationUnsupportedOperationError("scene not bound")
        scene = self._scenes.get(req.scene_id)
        if scene is None:
            raise AnimationUnsupportedOperationError("scene not bound")
        if req.operation == "evaluate":
            state = evaluate_animation(scene, req.payload["time"], expected_revision=req.expected_revision)
            return {"status": "ok", "request_id": req.request_id, "revision": state.revision, "time": state.time, "objects": state.ids()}
        if req.operation in {"set_animation", "clear_animation", "add_keyframe", "update_keyframe", "delete_keyframe"}:
            before = scene.revision
            result = update_animation(scene, req.operation, req.payload, expected_revision=req.expected_revision, dry_run=req.dry_run)
            if not req.dry_run:
                self.animation_memory.record(AnimationEditRecord(req.operation, req.scene_id, before, result["revision"]))
            return {"status": "ok", "request_id": req.request_id, **result}
        raise AnimationUnsupportedOperationError(f"unsupported operation: {req.operation}")

    def predict(self, state: Any, context: Any = None) -> dict[str, Any]:
        return self.perform_task(state)
