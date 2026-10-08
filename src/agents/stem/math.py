"""Legacy path marker for SLAI v2.3.

The authoritative implementation is the sibling ``stem/math/`` package.
Python's package importer resolves that package for ``src.agents.stem.math``;
this historical file is retained only so repository-relative tooling that
expects the legacy path does not fail. It intentionally contains no competing
implementation.
"""
__version__ = "2.3.0"
__all__: list[str] = []
