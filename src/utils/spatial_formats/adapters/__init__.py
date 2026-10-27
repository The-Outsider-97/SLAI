"""Optional, loss-aware spatial exchange backends; no heavy import at package load.

Importing this package only imports the dependency-free SVG, STL and GLB adapters.
DXF (ezdxf), IFC (IfcOpenShell), STEP (Open CASCADE) and DWG (an
explicitly configured ODA converter) are imported lazily by the functions that
need them and fail with :class:`DependencyUnavailableError` when absent.
"""
from .svg import export_svg, import_svg
from .stl import export_binary_stl, import_binary_stl
from .contracts import Capabilities, Policy

__all__ = ["export_svg", "import_svg", "export_binary_stl", "import_binary_stl", "Capabilities", "Policy",
           "capabilities"]


def capabilities(*, dwg_executable=None) -> dict:
    """Availability and limits of every exchange format, keyed by format name.

    Pure introspection: nothing is installed, downloaded or executed.
    ``dwg_executable`` is the optional absolute path of an ODA converter to probe.
    """
    from . import dwg, dxf, gltf, ifc, step
    builtin = {
        "svg": Capabilities("svg", "xml.etree", True, ("1.1 subset",), ("1.1 subset",),
                            ("point", "line", "polyline", "polygon", "circle", "arc"),
                            ("Fails closed on styles, text, transforms, use-elements and general path data",
                             "Display coordinates are not y-flipped")),
        "stl": Capabilities("stl", "struct", True, ("binary",), ("binary",), ("triangle mesh",),
                            ("No units, names, hierarchy or semantics", "float32 coordinates")),
    }
    probes = {"dxf": dxf.capabilities, "glb": gltf.capabilities, "ifc": ifc.capabilities,
              "step": step.capabilities, "dwg": lambda: dwg.capabilities(dwg_executable)}
    result = {name: caps.as_dict() for name, caps in builtin.items()}
    for name, probe in probes.items():
        result[name] = probe().as_dict()
    return result
