"""Loader overlay with ETA estimation for UI loading states.

This module provides a thread-safe loader state manager that can be used
with any UI framework (e.g., PyQt, Tkinter) to display a loading overlay
with estimated time remaining based on progress updates.
"""

from __future__ import annotations

import time
import threading
from typing import Optional, Callable, Any
from dataclasses import dataclass, field

from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Loader")
printer = PrettyPrinter


@dataclass
class LoaderState:
    """Internal state for the loader."""
    start_time: Optional[float] = None
    total_steps: int = 0
    completed_steps: int = 0
    progress: float = 0.0
    message: str = ""
    is_running: bool = False
    eta: Optional[float] = None


class Loader:
    """Thread‑safe loader with ETA estimation.

    Usage:
        loader = Loader(total_steps=100)
        loader.start("Loading data...")
        for i in range(100):
            time.sleep(0.05)
            loader.update(message=f"Step {i+1}/100")
        loader.complete("Done!")

    For indefinite progress, you can pass `total_steps=None` and use
    `update()` with a progress value between 0 and 1. ETA will be
    estimated based on the rate of progress.
    """

    def __init__(
        self,
        total_steps: Optional[int] = None,
        smoothing_factor: float = 0.3,
        on_update: Optional[Callable[[LoaderState], Any]] = None,
        on_complete: Optional[Callable[[], Any]] = None,
    ) -> None:
        """
        Args:
            total_steps: If provided, progress is measured in steps.
                         If `None`, progress is assumed to be a value between 0 and 1.
            smoothing_factor: Exponential moving average factor for ETA
                              when total_steps is not provided (0 < alpha ≤ 1).
            on_update: Callback invoked when state changes (e.g., update UI).
            on_complete: Callback invoked when loading finishes.
        """
        self.total_steps = total_steps
        self.smoothing_factor = min(max(smoothing_factor, 0.05), 1.0)
        self.on_update = on_update
        self.on_complete = on_complete

        self._state = LoaderState()
        self._lock = threading.RLock()
        self._last_eta_update = 0.0
        self._progress_history: list[tuple[float, float]] = []  # (timestamp, progress)
        self._eta_filter: Optional[float] = None

    def start(self, message: str = "") -> None:
        """Start the loader."""
        with self._lock:
            if self._state.is_running:
                # Treat duplicate starts as message refreshes; this keeps UI
                # interactions idempotent and avoids warning spam under rapid
                # user clicks/racey event handlers.
                if message:
                    self._state.message = message
                    self._notify_update()
                logger.debug("Loader already started; refreshed message")
                return
            self._state.start_time = time.time()
            self._state.is_running = True
            self._state.message = message
            self._state.progress = 0.0
            self._state.completed_steps = 0
            self._state.eta = None
            self._progress_history.clear()
            self._eta_filter = None
            self._last_eta_update = 0.0
            logger.debug(f"Loader started: {message}")
            self._notify_update()

    def update(
        self,
        steps_done: Optional[int] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
    ) -> None:
        """Update the loader progress.

        Args:
            steps_done: Number of steps completed (if total_steps given).
            progress: Progress value (0‑1) if total_steps not given.
            message: Optional new message.
        """
        with self._lock:
            if not self._state.is_running:
                logger.debug("Loader not started, ignoring update")
                return

            now = time.time()
            if self.total_steps is not None:
                if steps_done is not None:
                    self._state.completed_steps = steps_done
                elif progress is not None:
                    self._state.completed_steps = int(progress * self.total_steps)
                self._state.progress = self._state.completed_steps / self.total_steps
            else:
                if progress is not None:
                    self._state.progress = max(0.0, min(1.0, progress))
                elif steps_done is not None:
                    self._state.progress = steps_done / max(self.total_steps or 1, 1)

            if message is not None:
                self._state.message = message

            # Compute ETA based on rate of progress
            if self._state.progress > 0:
                elapsed = now - self._state.start_time
                remaining_time = (elapsed / self._state.progress) - elapsed
                self._state.eta = max(0.0, remaining_time)
                # For indefinite, we also update the filter
                if self.total_steps is None:
                    # Exponential moving average to smooth noisy progress
                    self._eta_filter = (
                        self._eta_filter
                        if self._eta_filter is not None
                        else self._state.eta
                    )
                    self._eta_filter = (self.smoothing_factor * self._state.eta +
                                       (1 - self.smoothing_factor) * self._eta_filter)
                    self._state.eta = self._eta_filter
            else:
                self._state.eta = None

            self._last_eta_update = now
            self._notify_update()

    def complete(self, message: str = "") -> None:
        """Mark the loader as complete."""
        with self._lock:
            if not self._state.is_running:
                logger.debug("Loader not started, ignoring complete()")
                return
            self._state.is_running = False
            self._state.progress = 1.0
            self._state.message = message
            self._state.eta = 0.0
            self._notify_update()
            if self.on_complete:
                try:
                    self.on_complete()
                except Exception as e:
                    logger.error(f"on_complete callback error: {e}")
            logger.debug("Loader completed")

    def cancel(self) -> None:
        """Cancel the loader (same as complete but with a different message)."""
        with self._lock:
            if self._state.is_running:
                self.complete("Cancelled")

    def get_state(self) -> LoaderState:
        """Return a copy of the current state."""
        with self._lock:
            return LoaderState(
                start_time=self._state.start_time,
                total_steps=self.total_steps if self.total_steps is not None else 0,
                completed_steps=self._state.completed_steps,
                progress=self._state.progress,
                message=self._state.message,
                is_running=self._state.is_running,
                eta=self._state.eta,
            )

    def _notify_update(self) -> None:
        """Call the on_update callback if provided."""
        if self.on_update:
            try:
                self.on_update(self.get_state())
            except Exception as e:
                logger.error(f"on_update callback error: {e}")


# Convenience function for one‑off loading with context manager
class LoaderContext:
    """Context manager for using a Loader."""
    def __init__(
        self,
        loader: Loader,
        message: str = "",
        total_steps: Optional[int] = None,
    ) -> None:
        self.loader = loader
        self.message = message
        self.total_steps = total_steps

    def __enter__(self) -> Loader:
        self.loader.start(self.message)
        return self.loader

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self.loader.complete()
        else:
            self.loader.cancel()


if __name__ == "__main__":
    print("\n=== Running Loader ===\n")
    printer.status("TEST", "Loader initialized", "info")
    loader = Loader(
        smoothing_factor=0.5
    )
    text= "Taking our sweet time"


    load = LoaderContext(
        loader=loader,
        message=text,
        total_steps=None
    )
    print(load)

    print("\n* * * * * Phase 2 - Plan * * * * *\n")
    import tkinter as tk
    from tkinter import ttk
    import time
    
    def run_gui_test():
        root = tk.Tk()
        root.title("Loader Demo")
        root.geometry("1440x150")
        root.resizable(False, False)
    
        # Widgets
        label = ttk.Label(root, text="Starting...", font=("Arial", 10))
        label.pack(pady=5)
    
        progress = ttk.Progressbar(root, orient="horizontal", length=720, mode="determinate")
        progress.pack(pady=5)
    
        eta_label = ttk.Label(root, text="ETA: --", font=("Arial", 9))
        eta_label.pack(pady=2)
    
        # Loader instance
        loader = Loader(
            total_steps=100,
            on_update=lambda state: root.after(0, lambda: update_ui(state)),
            on_complete=lambda: root.after(0, root.destroy)
        )
    
        def update_ui(state):
            progress["value"] = state.progress * 100
            label.config(text=state.message if state.message else "Loading...")
            if state.eta is not None:
                eta_label.config(text=f"ETA: {state.eta:.1f}s")
            else:
                eta_label.config(text="ETA: --")
    
        # Start loader
        loader.start("Loading...")
    
        # Simulate progress over 10 seconds (100 steps)
        step_count = 0
        total_steps = 100
        interval = 100  # ms per step (10s total)
    
        def step():
            nonlocal step_count
            if step_count < total_steps:
                step_count += 1
                loader.update(steps_done=step_count, message=f"Step {step_count}/{total_steps}")
                root.after(interval, step)
            else:
                loader.complete("Done!")
    
        root.after(0, step)  # start simulation
        root.mainloop()

    run_gui_test()
    print("\n=== Successfully ran the Loader ===\n")