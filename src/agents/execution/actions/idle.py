import time

from typing import Dict, Any, Optional

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.execution_error import SoftInterrupt
from ..actions.base_action import BaseAction, ActionStatus
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Idle Action")
printer = PrettyPrinter

class IdleAction(BaseAction):
    """
    Idle action that recovers energy over time. Can be interrupted.
    Duration is either explicitly given or computed from energy deficit.
    """
    name = "idle"
    priority = 0
    preconditions = []
    postconditions = ["has_rested"]
    _required_context_keys = []  # optional

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        self.config = load_global_config()
        self.idle_config = get_config_section("idle_action") or {}

        self.default_duration = self.idle_config.get("default_duration", 5.0)
        self.min_rest_threshold = self.idle_config.get("min_rest_threshold", 0.3)
        self.max_rest_threshold = self.idle_config.get("max_rest_threshold", 1.0)
        self.energy_recovery_rate = self.idle_config.get("energy_recovery_rate", 0.1)  # per second

        # Internal state
        self.idle_duration: float = 0.0
        self.time_elapsed: float = 0.0
        self.energy_gained: float = 0.0

        # Context defaults
        self.context.setdefault("max_idle_time", self.default_duration)
        self.context.setdefault("energy", 10.0)
        self.context.setdefault("max_energy", 10.0)

        logger.info("IdleAction initialized")

    def _execute(self) -> bool:
        """Perform idle with energy recovery, checking for interruptions."""
        self.idle_duration = self._calculate_duration()
        logger.info(f"Idling for {self.idle_duration:.1f} seconds...")

        start_time = time.time()
        self.time_elapsed = 0.0
        self.energy_gained = 0.0

        # Update status to show we're "accelerating" into idle (warm-up)
        self.status = ActionStatus.ACCELERATING

        while self.time_elapsed < self.idle_duration:
            if self._should_interrupt():
                raise SoftInterrupt("Idle interrupted")

            self.time_elapsed = time.time() - start_time
            self._recover_energy()
            self._update_status(new_status=self.status)
            time.sleep(0.1)

        logger.info(f"Finished idling. Gained {self.energy_gained:.1f} energy")
        return True

    def _calculate_duration(self) -> float:
        """Determine idle duration based on explicit context or energy deficit."""
        if "idle_duration" in self.context:
            return float(self.context["idle_duration"])

        current = self.context.get("energy", 1.0)
        max_energy = self.context.get("max_energy", 10.0)
        deficit = max_energy - current
        if deficit > 0:
            needed = deficit / self.energy_recovery_rate
            max_allowed = self.context.get("max_idle_time", self.default_duration)
            return min(needed, max_allowed)
        return self.default_duration

    def _recover_energy(self) -> None:
        """Increase energy gradually based on elapsed time."""
        # Compute energy gain since last call (simplified: use time step)
        # Since we call this every ~0.1s, we can approximate delta = 0.1
        # But for accuracy, compute actual delta.
        # We'll store last energy update time.
        if not hasattr(self, "_last_energy_time"):
            self._last_energy_time = time.time()
        now = time.time()
        delta = min(0.1, now - self._last_energy_time)
        self._last_energy_time = now

        gain = delta * self.energy_recovery_rate
        if gain > 0 and "energy" in self.context:
            new_energy = min(self.context.get("max_energy", 10.0),
                             self.context["energy"] + gain)
            self.energy_gained += new_energy - self.context["energy"]
            self.context["energy"] = new_energy

    def _update_status(self, new_status: ActionStatus) -> None:
        """Update action status based on progress through idle duration."""
        if self.idle_duration <= 0:
            return
        progress = self.time_elapsed / self.idle_duration
        if progress < 0.3:
            self.status = ActionStatus.ACCELERATING
        elif progress > 0.7:
            self.status = ActionStatus.DECELERATING
        else:
            self.status = ActionStatus.RUNNING

    # ------------------------ Lifecycle Overrides ----------------------
    def _pre_execute(self):
        super()._pre_execute()
        logger.info("Entering idle state")
        self.context.setdefault("energy", 8.0)
        self.context.setdefault("max_energy", 10.0)
        self._last_energy_time = time.time()

    def _post_execute(self, success: bool):
        super()._post_execute(success)
        # Reset interruption flags
        if "interrupt_idle" in self.context:
            self.context["interrupt_idle"] = False
        # Apply rest benefit even if interrupted (as long as some time passed)
        if self.time_elapsed > self.min_rest_threshold:
            self.context["has_rested"] = True
            logger.debug("Applied rest benefits")

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "idle_duration": self.idle_duration,
            "time_elapsed": self.time_elapsed,
            "energy_gained": self.energy_gained,
            "recovery_rate": self.energy_recovery_rate,
        })
        return base

if __name__ == "__main__":
    print("\n=== Running Execution IDLE Action ===\n")
    printer.status("TEST", "Starting Execution IDLE Action tests", "info")
    context={"idle_duration": 8.0, "energy": 5.0}

    idle = IdleAction(context)
    print(f"{idle}")

    print("\n* * * * * Phase 2 * * * * *\n")

    success= True
    status = ActionStatus.RUNNING

    printer.pretty("EXEC", idle._execute(), "success")
    printer.pretty("CALC", idle._calculate_duration(), "success")
    printer.pretty("RECOVER", idle._recover_energy(), "success")
    printer.pretty("INTERRUPT", idle._should_interrupt(), "success")
    printer.pretty("PRE", idle._pre_execute(), "success")
    printer.pretty("POST", idle._post_execute(success=success), "success")
    printer.pretty("DICT", idle.to_dict(), "success")
    printer.pretty("STATUS", idle._update_status(new_status=status), "success")

    print("\n=== All tests completed successfully! ===\n")
