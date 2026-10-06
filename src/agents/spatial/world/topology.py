"""
sources:
- Egenhofer & Franzosa (1991) — point-set topology and boundary/interior relations.
- Randell, Cui & Cohn (1992) — region connection calculus.
- Clementini, Di Felice & van Oosterom (1993) — useful topological relations for spatial systems.
- ISO 19107:2019 — standardized vector geometry and topology concepts.

clear seperation:
geometry.py:
Where exactly do these boundaries intersect?

topology.py:
What invariant spatial relationship does that imply?

spatial_relations.py:
How should SLAI represent/query/reason over that relationship?
"""
from __future__ import annotations

__version__ = "2.3.0"


from ..utils.spatial_errors import *
from ..utils.spatial_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Topology")
printer = PrettyPrinter()

def connected():
    pass


def disconnected():
    pass


def boundary():
    pass


class Topology:
    def __init__(self):
        pass

    def interior(self): ...

    def contains(self):
        self.boundary = boundary

    def inside(self):
        self.boundary = boundary

    def touches(self): ...

    def overlaps(self): ...

    def crosses(self): ...

    def equal(self): ...

    def closure(self):
        self.disconnect = disconnected
        return self.disconnect


__all__ = [
    "connected",
    "disconnected",
    "boundary",
    "Topology",
]


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Topology ===\n")
    printer.status("TEST", "Topology initialized", "info")