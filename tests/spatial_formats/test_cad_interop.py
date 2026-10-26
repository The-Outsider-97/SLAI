"""Adapters: capability policies, backend absence, malformed inputs and bounded DXF."""
from __future__ import annotations
from pathlib import Path
import pytest
from src.utils.spatial_formats.adapters.contracts import Policy, Capabilities, enforce_policy
from src.utils.spatial_formats.adapters import dxf, dwg, gltf, ifc, step
from src.utils.spatial_formats.common.errors import (
    ValidationError, ResourceLimitError, UnsupportedFeatureError, DependencyUnavailableError,
)
from src.utils.spatial_formats import Drawing

def test_policy_is_explicit():
    assert enforce_policy(Policy.STRICT)==Policy.STRICT
    with pytest.raises(UnsupportedFeatureError):
        enforce_policy(Policy.STRICT,losses=["some metadata"])
    with pytest.raises(UnsupportedFeatureError):
        enforce_policy(Policy.ALLOW_APPROXIMATION,losses=["hierarchy"])
    assert enforce_policy(Policy.BEST_EFFORT,losses=["hierarchy"])==Policy.BEST_EFFORT

def test_all_capability_contracts_return_sensible_data():
    for module in (dxf,gltf,ifc,step):
        cap=module.capabilities()
        assert isinstance(cap,Capabilities)
        assert isinstance(cap.available,bool)
        assert isinstance(cap.entities,tuple)
    assert not dwg.capabilities().available

def test_dwg_requires_configured_converter(tmp_path):
    with pytest.raises(DependencyUnavailableError):
        dwg.import_dwg(tmp_path/"example.dwg",executable=None)
    with pytest.raises(DependencyUnavailableError):
        dwg._executable(None)
    with pytest.raises(DependencyUnavailableError):
        dwg._executable(str(tmp_path/"nope.exe"))
    with pytest.raises(DependencyUnavailableError):
        dwg._convert(tmp_path/"broken",executable=None,version="wrong")


def test_dwg_backend_validation_precedes_source_validation(tmp_path):
    """Missing converter is consistently a dependency error, not file error."""
    absent = tmp_path / "absent.dwg"
    with pytest.raises(DependencyUnavailableError):
        dwg.import_dwg(absent, executable=None)
    # An input with a recognizable header does not waive backend requirements.
    header = tmp_path / "candidate.dwg"
    header.write_bytes(b"AC1032" + bytes(64))
    with pytest.raises(DependencyUnavailableError):
        dwg.import_dwg(header, executable=None)


def test_dwg_valid_backend_still_validates_missing_input(tmp_path):
    """Once configured, missing/corrupt source files are not treated as valid."""
    import sys
    # This only checks input-validation ordering; it never launches Python as
    # a DWG converter or claims end-to-end ODA compatibility.
    with pytest.raises(ValidationError, match="DWG input"):
        dwg.import_dwg(tmp_path / "absent.dwg", executable=str(Path(sys.executable).resolve()))

def test_dwg_header_validation(tmp_path):
    bad=tmp_path/"bad.dwg";bad.write_bytes(b"NOTDWG"*3)
    with pytest.raises(UnsupportedFeatureError):
        dwg.validate_dwg(bad)
    valid=tmp_path/"version.dwg";valid.write_bytes(b"AC1032"+b"\x00"*100)
    assert dwg.validate_dwg(valid)=="AC1032"

def test_step_header_boundary(tmp_path):
    missing=tmp_path/"broken.step";missing.write_bytes(b"NOT-STEP")
    with pytest.raises(ValidationError):
        step.validate_step(missing)
    valid=tmp_path/"part.step";valid.write_bytes(b"ISO-10303-21;\nHEADER;\nENDSEC;\n")
    assert step.validate_step(valid)==valid

def test_glb_bad_header_rejected_without_backend():
    with pytest.raises(ValidationError):
        gltf.import_glb_scene(b"invalid")
    malformed=b"glTF"+(2).to_bytes(4,"little")+(99).to_bytes(4,"little")+b"more bytes"
    with pytest.raises(ValidationError):
        gltf.import_glb_scene(malformed)

def test_ifc_missing_file_or_optional_dependency(tmp_path):
    if ifc.capabilities().available:
        with pytest.raises((OSError,ResourceLimitError,ValueError)):
            ifc.import_ifc(tmp_path/"missing.ifc")
    else:
        with pytest.raises(DependencyUnavailableError):
            ifc.import_ifc(tmp_path/"missing.ifc")

def test_step_unavailable_raises_typed_exception():
    if not step.capabilities().available:
        with pytest.raises(DependencyUnavailableError):
            step.import_step("missing.step")

def test_dxf_import_rejects_missing_file_with_ezdxf(tmp_path):
    if not dxf.capabilities().available:
        pytest.skip("ezdxf optional backend unavailable")
    with pytest.raises(ResourceLimitError):
        dxf.import_dxf(tmp_path/"missing.dxf")

def test_dxf_core_roundtrip(tmp_path):
    if not dxf.capabilities().available:
        pytest.skip("ezdxf optional backend unavailable")
    drawing=Drawing(unit="mm")
    drawing.add_layer("A-WALL")
    drawing.add("line",{"start":[0,0],"end":[10,0]},layer="A-WALL")
    drawing.add("circle",{"center":[5,5],"radius":2},layer="A-WALL")
    path=tmp_path/"plan.dxf"
    dxf.export_dxf(drawing,path)
    result=dxf.import_dxf(path,unit="mm")
    assert [x["kind"] for x in result.entities]==["line","circle"]
    assert {x["name"] for x in result.layers}=={"0","A-WALL"}

def test_glb_single_mesh_roundtrip():
    if not gltf.capabilities().available:
        pytest.skip("trimesh optional backend unavailable")
    shape={"kind":"mesh","vertices":[[0,0,0],[1,0,0],[0,1,0]],"faces":[[0,1,2]]}
    binary=gltf.export_glb(shape)
    result=gltf.import_glb_scene(binary)
    assert len([x for x in result.objects if x.get("geometry")])==1
