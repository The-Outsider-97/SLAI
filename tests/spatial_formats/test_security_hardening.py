"""Adversarial tests for spatial ZIP resource boundaries and numeric contracts."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_STORED, ZIP_BZIP2
import pytest

from src.utils.spatial_formats import Scene, load_package, write_package
from src.utils.spatial_formats.common import (
    ValidationError, IntegrityError, ResourceLimitError, UnsupportedFeatureError,
)
from src.utils.spatial_formats.sgeo.parameters import evaluate_expression, Quantity
from src.utils.spatial_formats.container.buffers import validate_descriptor

def test_zip_duplicate_entries_rejected(tmp_path):
    path=tmp_path/"dup.sgeo"
    with ZipFile(path,"w") as z:
        z.writestr("mimetype","application/x-slai-sgeo+zip")
        z.writestr("mimetype","application/x-slai-sgeo+zip")
        z.writestr("manifest.json","{}")
    with pytest.raises((ValidationError,ResourceLimitError)):
        load_package(path)

def test_unsupported_zip_compression_rejected(tmp_path):
    path=tmp_path/"bzip2.sgeo"
    with ZipFile(path,"w") as z:
        z.writestr("mimetype",b"application/x-slai-sgeo+zip",compress_type=ZIP_STORED)
        z.writestr("manifest.json",b"{}",compress_type=ZIP_BZIP2)
        z.writestr("scene.json",b"{}",compress_type=ZIP_STORED)
    with pytest.raises(UnsupportedFeatureError):
        load_package(path)

def test_traversal_archive_path_rejected(tmp_path):
    path=tmp_path/"unsafe.sgeo"
    with ZipFile(path,"w") as z:
        z.writestr("mimetype",b"application/x-slai-sgeo+zip")
        z.writestr("../manifest.json",b"{}")
        z.writestr("scene.json",b"{}")
    with pytest.raises(ValidationError):
        load_package(path)

def test_externalized_scene_rejects_extra_asset(tmp_path):
    scene=Scene()
    path=tmp_path/"extra.sgeo"
    with pytest.raises(ValidationError):
        write_package(path,scene.manifest(),scene.as_dict(),assets={"geometry/fake.bin":b"illegal"})
    assert not path.exists()
    # Passing an empty explicit override must not bypass codec ownership.
    with pytest.raises(ValidationError):
        write_package(path,scene.manifest(),scene.as_dict(),assets={})
    assert not path.exists()

def test_binary_descriptor_enforces_content_address():
    descriptor={"path":"geometry/sha256-"+"f"*64+".bin","sha256":"0"*64,
                "dtype":"f64","shape":[3,3],"byte_order":"little"}
    with pytest.raises(ValidationError):
        validate_descriptor(descriptor)

def test_expression_rejects_numeric_infinities():
    with pytest.raises(ValidationError):
        evaluate_expression("length(1e12)*length(1e12)",{})
    with pytest.raises(ValidationError):
        evaluate_expression("1/0",{})
