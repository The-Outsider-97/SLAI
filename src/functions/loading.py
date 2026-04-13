"""Convenience functions for wiring Loader to UI overlays."""

from __future__ import annotations

from typing import Optional

from src.functions.loader import Loader


def create_loading_controller(total_steps: Optional[int] = None) -> Loader:
    """Create a loader instance used by desktop loading overlays."""
    return Loader(total_steps=total_steps)


def start_loading(loader: Loader, message: str) -> None:
    """Start the loading lifecycle with a message."""
    loader.start(message=message)


def update_loading(
    loader: Loader,
    *,
    steps_done: Optional[int] = None,
    progress: Optional[float] = None,
    message: Optional[str] = None,
) -> None:
    """Update the loading lifecycle."""
    loader.update(steps_done=steps_done, progress=progress, message=message)


def complete_loading(loader: Loader, message: str = "Completed") -> None:
    """Mark a loading lifecycle as complete."""
    loader.complete(message=message)
