"""Regression tests for the optional SGEO animation runtime factory."""

from __future__ import annotations

import subprocess
import sys

import pytest

from src.utils.spatial_formats.adapters.slai_animation import SGEOAnimationAdapter
from src.utils.spatial_formats.adapters.spatial_format_runtime import build_animation_adapter
from src.utils.spatial_formats.sgeo.scene import Scene


def test_factory_binds_the_supplied_authoritative_scene():
    scene = Scene(name="Runtime factory regression")
    adapter = build_animation_adapter(scene)
    assert isinstance(adapter, SGEOAnimationAdapter)
    assert adapter.document_id == scene.document_id
    assert adapter.revision == scene.revision
    assert adapter.metadata is scene.metadata
    assert adapter.animation_edit_lock is scene._mutation_lock


def test_factory_rejects_invalid_scene_types():
    with pytest.raises(TypeError, match="SGEO Scene"):
        build_animation_adapter(None)


def test_factory_import_does_not_initialize_agents():
    statement = (
        "import sys; "
        "from src.utils.spatial_formats.adapters.spatial_format_runtime "
        "import build_animation_adapter; "
        "from src.utils.spatial_formats.sgeo.scene import Scene; "
        "adapter = build_animation_adapter(Scene()); "
        "assert adapter.document_id; "
        "assert 'src.agents.animation_agent' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", statement], check=True, timeout=25)
