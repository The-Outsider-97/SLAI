"""Opt-in DWG exchange through a verified, explicitly configured ODA converter.

No DWG parsing, shell calls, network access or runtime software installation.
Supports ODAFileConverter CLI: input-dir output-dir version type recurse audit.
Backend-specific geometry is subsequently handled only by the DXF adapter.
"""
from __future__ import annotations
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from ..common.errors import DependencyUnavailableError, IntegrityError, ResourceLimitError, ValidationError, UnsupportedFeatureError
from .contracts import Capabilities
from .dxf import import_dxf, export_dxf

_DWG_MAGIC=(b"AC1012",b"AC1014",b"AC1015",b"AC1018",b"AC1021",b"AC1024",b"AC1027",b"AC1032")
_DXF_MAGIC=b"AutoCAD Binary DXF"
_VERSIONS=frozenset(("ACAD2000","ACAD2004","ACAD2007","ACAD2010","ACAD2013","ACAD2018"))
_MAX_BYTES=64*1024*1024

def _executable(executable):
    if not executable:
        raise DependencyUnavailableError("DWG conversion requires an explicitly configured ODAFileConverter executable")
    candidate=Path(executable).expanduser()
    if not candidate.is_absolute() or not candidate.is_file():
        raise DependencyUnavailableError("DWG converter must be an existing absolute executable path")
    if candidate.is_symlink():
        raise ValidationError("symlinked converter executable is prohibited")
    return str(candidate)

def capabilities(executable=None):
    ready=False
    try: _executable(executable);ready=True
    except (DependencyUnavailableError,ValidationError): pass
    return Capabilities("dwg","ODAFileConverter",ready,tuple(sorted(_VERSIONS)) if ready else (),
                        tuple(sorted(_VERSIONS)) if ready else (),("DXF-supported planar entities",),
                        ("Requires separately licensed ODA executable","Model-space DXF subset only",
                         "Paper-space and DWG-specific entities are not preserved"))

def validate_dwg(path):
    source=Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValidationError("DWG input must be a regular, non-symlink file")
    size=source.stat().st_size
    if not 6<=size<=_MAX_BYTES:
        raise ResourceLimitError("DWG file exceeds allowed size or is empty")
    with source.open("rb") as stream: marker=stream.read(6)
    if marker not in _DWG_MAGIC:
        raise UnsupportedFeatureError("unrecognized DWG file version/header")
    return marker.decode("ascii")

def _convert(source, *, executable, version="ACAD2018", target_type="DXF", timeout=90):
    exe=_executable(executable)
    if version not in _VERSIONS or target_type not in ("DXF","DWG"):
        raise UnsupportedFeatureError("unsupported ODA conversion target/version")
    if type(timeout) not in (int,float) or not 1<=timeout<=600:
        raise ValidationError("converter timeout outside [1,600] seconds")
    with tempfile.TemporaryDirectory(prefix="slai-dwg-") as workspace:
        root=Path(workspace);incoming=root/"in";outgoing=root/"out"
        incoming.mkdir();outgoing.mkdir()
        filename="drawing." + source.suffix.lower().lstrip(".")
        if filename not in ("drawing.dwg","drawing.dxf"):
            raise ValidationError("invalid converter input suffix")
        isolated=incoming/filename
        if source.stat().st_size>_MAX_BYTES:
            raise ResourceLimitError("input too large")
        shutil.copyfile(source,isolated)
        # No shell, no caller-controlled output directory or environment arguments.
        args=[exe,str(incoming),str(outgoing),version,target_type,"0","1"]
        try:
            result=subprocess.run(args,shell=False,stdin=subprocess.DEVNULL,
                                  capture_output=True,timeout=timeout,check=False,cwd=root)
        except subprocess.TimeoutExpired as exc:
            raise ResourceLimitError("DWG backend conversion timed out") from exc
        except OSError as exc:
            raise DependencyUnavailableError("DWG backend could not start") from exc
        if result.returncode!=0:
            raise IntegrityError("DWG backend failed with exit code "+str(result.returncode))
        target=outgoing/("drawing."+target_type.lower())
        if target.is_symlink() or not target.is_file() or not 0<target.stat().st_size<=_MAX_BYTES:
            raise IntegrityError("DWG backend did not produce valid output")
        if target_type=="DWG":
            validate_dwg(target)
        else:
            with target.open("rb") as file: first=file.read(32)
            if b"SECTION" not in first.upper() and not first.startswith(_DXF_MAGIC):
                raise IntegrityError("backend output does not resemble DXF")
        return target.read_bytes()

def import_dwg(path, *, executable, unit="mm", version="ACAD2018", timeout=90):
    """Import through an explicitly configured backend.

    Backend availability is checked first, so callers receive a stable
    DependencyUnavailableError even when the source file is also absent.
    Once configured, untrusted DWG input is validated before conversion.
    """
    _executable(executable)
    validate_dwg(path)
    data=_convert(Path(path),executable=executable,version=version,target_type="DXF",timeout=timeout)
    with tempfile.TemporaryDirectory(prefix="slai-dwg-read-") as root:
        file=Path(root)/"drawing.dxf";file.write_bytes(data)
        return import_dxf(file,unit=unit)

def export_dwg(drawing,path,*,executable,version="ACAD2018",timeout=90,overwrite=False):
    destination=Path(path)
    if destination.suffix.lower()!=".dwg":
        raise ValidationError("DWG output requires .dwg suffix")
    if destination.exists() and not overwrite:
        raise ValidationError("DWG destination already exists")
    with tempfile.TemporaryDirectory(prefix="slai-dwg-export-") as root:
        intermediary=Path(root)/"drawing.dxf"
        export_dxf(drawing,intermediary)
        content=_convert(intermediary,executable=executable,version=version,target_type="DWG",timeout=timeout)
        with tempfile.NamedTemporaryFile(dir=destination.parent,delete=False,suffix=".dwg") as stream:
            stage=Path(stream.name);stream.write(content);stream.flush();os.fsync(stream.fileno())
        try:
            validate_dwg(stage)
            if destination.exists() and not overwrite:
                raise ValidationError("DWG destination already exists")
            os.replace(stage,destination)
        finally:
            stage.unlink(missing_ok=True)
    return destination
