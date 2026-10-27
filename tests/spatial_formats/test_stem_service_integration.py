"""Exercise the spatial-format STEM bridge against the real STEM UnitSystem."""

from __future__ import annotations

import subprocess
import sys

import pytest

from src.utils.spatial_formats.adapters.slai_stem import (
    STEMQuantityAdapter,
    convert_quantity,
    quantity,
)
from src.utils.spatial_formats.adapters.spatial_format_runtime import build_stem_adapter
from src.utils.spatial_formats.common.errors import ValidationError


@pytest.fixture
def stem_adapter() -> STEMQuantityAdapter:
    # Importing the actual agent implementation is an explicit test/application
    # decision. Importing spatial_formats itself must never initialize STEM.
    from src.agents.stem.units.unit_system import UnitSystem

    return STEMQuantityAdapter(UnitSystem())


@pytest.mark.parametrize(
    ("kind", "value", "source", "target", "expected"),
    [
        ("length", 10.0, "cm", "mm", 100.0),
        ("area", 1.0, "m", "cm", 10_000.0),
        ("volume", 1.0, "m", "cm", 1_000_000.0),
        ("volume", 1.0, "m", "mm", 1_000_000_000.0),
    ],
)
def test_real_stem_conversion(stem_adapter, kind, value, source, target, expected):
    original = stem_adapter.quantity(value, source, kind)
    result = stem_adapter.convert_quantity(original, target)
    assert result == {
        "value": pytest.approx(expected),
        "unit": target,
        "dimension": kind,
    }
    assert stem_adapter.convert_quantity(result, source)["value"] == pytest.approx(value)


def test_stem_registry_rejects_unsupported_unit_without_local_fallback(stem_adapter):
    from src.agents.stem.utils.stem_errors import STEMUnitError

    # Inches belong to the file schema, but are not part of the default STEM
    # registry; the bridge must not silently substitute its local table.
    with pytest.raises(STEMUnitError):
        stem_adapter.convert_quantity(quantity(1, "in"), "m")


def test_stem_dimension_check_uses_real_registry():
    from src.agents.stem.stem_types import Dimension
    from src.agents.stem.units.unit_system import UnitSystem

    service = UnitSystem()
    service.register_derived("ft", "deliberately non-length unit", Dimension({"T": 1}))
    adapter = STEMQuantityAdapter(service)
    with pytest.raises(ValidationError, match="STEM length dimension"):
        adapter.quantity(1.0, "ft")


def test_invalid_records_fail_before_calling_stem(stem_adapter):
    with pytest.raises(ValidationError):
        stem_adapter.convert_quantity({"dimension": "length", "value": True, "unit": "m"}, "cm")
    with pytest.raises(ValidationError):
        stem_adapter.convert_quantity({"dimension": "temperature", "value": 10, "unit": "m"}, "cm")
    with pytest.raises(ValidationError):
        stem_adapter.quantity(float("nan"), "m")
    with pytest.raises(ValidationError):
        stem_adapter.quantity(1, "seconds")


def test_existing_agent_free_api_is_unchanged():
    assert convert_quantity(quantity(10, "cm"), "mm")["value"] == pytest.approx(100)
    assert convert_quantity(quantity(1, "m", "area"), "cm")["value"] == pytest.approx(10_000)
    assert convert_quantity(quantity(1, "m", "volume"), "cm")["value"] == pytest.approx(1_000_000)


def test_bridge_requires_a_real_service_contract():
    with pytest.raises(TypeError, match="STEM unit service"):
        STEMQuantityAdapter(None)


def test_runtime_wires_the_real_stem_service():
    adapter = build_stem_adapter()
    from src.agents.stem.units.unit_system import UnitSystem

    assert isinstance(adapter.unit_system, UnitSystem)
    assert adapter.convert_quantity(adapter.quantity(1, "m", "area"), "cm")["value"] == pytest.approx(10_000)
    assert build_stem_adapter(adapter.unit_system).unit_system is adapter.unit_system


def test_importing_spatial_bridge_does_not_initialize_stem():
    statement = (
        "import sys; import src.utils.spatial_formats.adapters.slai_stem; "
        "import src.utils.spatial_formats.adapters.spatial_format_runtime; "
        "assert 'src.agents.stem' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", statement], check=True, timeout=20)
