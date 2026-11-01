"""
SLAI deployment composition package.

This package owns concrete host-runtime bindings for applications installed
inside SLAI.

Application packages must not import this package. Dependency direction is:

    deployment
        -> applications
        -> domain / application ports

Never the reverse.
"""

from __future__ import annotations


__all__: list[str] = []