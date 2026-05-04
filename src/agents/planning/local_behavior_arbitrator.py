"""
Local behavior arbitration for short-horizon planning.

This module is intentionally domain-agnostic: robotics can provide obstacle,
clearance, and motion hints, while other domains can still use the same
priority-based override mechanism for immediate safety behaviors.
"""

from __future__ import annotations

import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.agents.base.utils.main_config_loader import load_global_config, get_config_section
from src.agents.planning.planning_memory import PlanningMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Local Behavior Arbitrator")
printer = PrettyPrinter


class BehaviorType(str, Enum):
    CONTINUE = "continue"
    STOP = "stop"
    SLOW = "slow"
    TURN = "turn"
    BACKUP = "backup"


@dataclass
class LocalPlanningContext:
    obstacle_distance_m: Optional[float] = None
    obstacle_bearing_deg: Optional[float] = None
    clearance_left_m: Optional[float] = None
    clearance_right_m: Optional[float] = None
    reverse_clearance_m: Optional[float] = None
    current_speed_mps: float = 0.0
    desired_speed_mps: float = 0.0
    horizon_seconds: float = 2.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArbitrationDecision:
    behavior: BehaviorType
    priority: int
    reason: str
    command: Dict[str, Any]

    @property
    def is_override(self) -> bool:
        return self.behavior != BehaviorType.CONTINUE


class LocalBehaviorArbitrator:
    """Priority arbitration between reactive safety and nominal plan execution."""

    def __init__(self, config: Optional[Dict] = None, memory: Optional[PlanningMemory] = None) -> None:
        self.config = load_global_config()
        self.lba_config = get_config_section("local_planning")

        self.stop_distance_m = float(self.lba_config.get("stop_distance_m", 0.35))
        self.slow_distance_m = float(self.lba_config.get("slow_distance_m", 0.8))
        self.backup_distance_m = float(self.lba_config.get("backup_distance_m", 0.2))
        self.turn_clearance_m = float(self.lba_config.get("turn_clearance_m", 0.3))
        self.slow_speed_factor = float(self.lba_config.get("slow_speed_factor", 0.4))
        self.max_local_horizon_s = float(self.lba_config.get("max_local_horizon_s", 3.0))

        self.memory = memory
        self._last_context: Optional[LocalPlanningContext] = None
        self._last_decision: Optional[ArbitrationDecision] = None

        logger.info(f"Local Behavior Arbitrator successfully initialized")

    # -------------------------------------------------------------------------
    # Decision logic
    # -------------------------------------------------------------------------
    def decide(self, context: LocalPlanningContext) -> ArbitrationDecision:
        """Main entry point: rule‑based decision, optionally refined by memory."""
        self._last_context = context
        base = self._rule_based_decision(context)
        if not self.memory:
            return base
        
        self._last_decision = base   # store the rule‑based decision for later logging

        similar = self._find_similar_contexts(context, k=5)
        if similar:
            stats = {}
            for entry in similar:
                beh = entry.get("behavior")
                outcome = entry.get("outcome")
                if beh and outcome is not None:
                    stats.setdefault(beh, {"success": 0, "total": 0})
                    stats[beh]["total"] += 1
                    if outcome:
                        stats[beh]["success"] += 1
            rates = {b: d["success"] / d["total"] for b, d in stats.items() if d["total"] >= 2}
            if rates:
                best = max(rates.items(), key=lambda x: x[1])
                if best[1] > 0.7 and best[0] != base.behavior.value:
                    logger.info(f"Overriding with memory-based behavior: {best[0]} (success rate {best[1]:.2f})")
                    return self._create_decision_from_behavior(best[0], context)
        return base

    def _rule_based_decision(self, context: LocalPlanningContext) -> ArbitrationDecision:
        """Original deterministic decision logic."""
        obstacle = context.obstacle_distance_m
        if obstacle is None:
            return ArbitrationDecision(
                behavior=BehaviorType.CONTINUE,
                priority=0,
                reason="No local obstacle signal",
                command={"type": "continue"},
            )

        if obstacle <= self.backup_distance_m and (context.reverse_clearance_m or 0.0) > self.backup_distance_m:
            return ArbitrationDecision(
                behavior=BehaviorType.BACKUP,
                priority=100,
                reason="Obstacle critically close; backing up to regain clearance",
                command={"type": "motion", "mode": "backup", "distance_m": 0.2, "speed_scale": 0.25},
            )

        if obstacle <= self.stop_distance_m:
            return ArbitrationDecision(
                behavior=BehaviorType.STOP,
                priority=95,
                reason="Obstacle inside stop distance",
                command={"type": "motion", "mode": "stop"},
            )

        if obstacle <= self.slow_distance_m:
            cmd = {
                "type": "motion",
                "mode": "slow",
                "speed_scale": max(0.05, min(1.0, self.slow_speed_factor)),
            }
            left = context.clearance_left_m or 0.0
            right = context.clearance_right_m or 0.0
            if left > self.turn_clearance_m or right > self.turn_clearance_m:
                cmd.update(
                    {
                        "mode": "turn",
                        "turn_direction": "left" if left >= right else "right",
                        "speed_scale": 0.2,
                    }
                )
                return ArbitrationDecision(
                    behavior=BehaviorType.TURN,
                    priority=85,
                    reason="Obstacle nearby; selecting safer turn corridor",
                    command=cmd,
                )
            return ArbitrationDecision(
                behavior=BehaviorType.SLOW,
                priority=75,
                reason="Obstacle within caution zone",
                command=cmd,
            )

        return ArbitrationDecision(
            behavior=BehaviorType.CONTINUE,
            priority=0,
            reason="Path clearance acceptable",
            command={"type": "continue"},
        )

    def _create_decision_from_behavior(self, behavior: str, context: LocalPlanningContext) -> ArbitrationDecision:
        """
        Re‑create a full decision object for a given behavior string.
        This mirrors the command generation logic from the rule‑based decision.
        """
        if behavior == BehaviorType.STOP.value:
            return ArbitrationDecision(
                behavior=BehaviorType.STOP,
                priority=95,
                reason="Memory-based stop",
                command={"type": "motion", "mode": "stop"},
            )
        elif behavior == BehaviorType.SLOW.value:
            cmd = {
                "type": "motion",
                "mode": "slow",
                "speed_scale": max(0.05, min(1.0, self.slow_speed_factor)),
            }
            return ArbitrationDecision(
                behavior=BehaviorType.SLOW,
                priority=75,
                reason="Memory-based slow",
                command=cmd,
            )
        elif behavior == BehaviorType.TURN.value:
            left = context.clearance_left_m or 0.0
            right = context.clearance_right_m or 0.0
            cmd = {
                "type": "motion",
                "mode": "turn",
                "turn_direction": "left" if left >= right else "right",
                "speed_scale": 0.2,
            }
            return ArbitrationDecision(
                behavior=BehaviorType.TURN,
                priority=85,
                reason="Memory-based turn",
                command=cmd,
            )
        elif behavior == BehaviorType.BACKUP.value:
            return ArbitrationDecision(
                behavior=BehaviorType.BACKUP,
                priority=100,
                reason="Memory-based backup",
                command={"type": "motion", "mode": "backup", "distance_m": 0.2, "speed_scale": 0.25},
            )
        else:
            return ArbitrationDecision(
                behavior=BehaviorType.CONTINUE,
                priority=0,
                reason="Fallback to continue",
                command={"type": "continue"},
            )

    # -------------------------------------------------------------------------
    # Memory integration
    # -------------------------------------------------------------------------
    def _find_similar_contexts(self, context: LocalPlanningContext, k: int = 3) -> List[Dict]:
        """
        Retrieve the k most similar past local behavior events from memory.
        Uses weighted Euclidean distance on numeric features.
        """
        if not self.memory:
            return []
        history = self.memory._state["execution_history"]
        # Filter only local behavior entries
        local_entries = [e for e in history if e.get("type") == "local_behavior"]
        if not local_entries:
            return []

        # Define features and weights (tune as needed)
        weights = {
            "obstacle_distance_m": 1.0,
            "clearance_left_m": 0.8,
            "clearance_right_m": 0.8,
            "current_speed_mps": 0.5,
        }
        features = {}
        for key in weights:
            val = getattr(context, key, None)
            if val is not None:
                features[key] = val
        if not features:
            return []

        scored = []
        for entry in local_entries:
            ctx = entry.get("context", {})
            dist = 0.0
            valid = True
            for key, weight in weights.items():
                a = features.get(key)
                b = ctx.get(key)
                if a is None or b is None:
                    valid = False
                    break
                dist += weight * (a - b) ** 2
            if valid:
                scored.append((dist, entry))
        scored.sort(key=lambda x: x[0])
        return [e for _, e in scored[:k]]

    def record_outcome(self, context: LocalPlanningContext, behavior: BehaviorType, success: bool) -> None:
        if not self.memory:
            return
        entry = {
            "type": "local_behavior",
            "timestamp": time.time(),
            "behavior": behavior.value,
            "context": {
                "obstacle_distance_m": context.obstacle_distance_m,
                "obstacle_bearing_deg": context.obstacle_bearing_deg,
                "clearance_left_m": context.clearance_left_m,
                "clearance_right_m": context.clearance_right_m,
                "reverse_clearance_m": context.reverse_clearance_m,
                "current_speed_mps": context.current_speed_mps,
                "desired_speed_mps": context.desired_speed_mps,
                "horizon_seconds": context.horizon_seconds,
            },
            "outcome": success,
        }
        # Append to execution_history (PlanningMemory will handle maxlen)
        self.memory._state["execution_history"].append(entry)
        # Optionally trigger a checkpoint if you want to persist
        self.memory.save_checkpoint(label="local_behavior_outcome")

    def should_trigger_short_horizon_replan(self, decision: ArbitrationDecision, context: LocalPlanningContext) -> bool:
        if not decision.is_override:
            return False
        return context.horizon_seconds <= self.max_local_horizon_s

    def build_reactive_task_name(self, decision: ArbitrationDecision) -> str:
        return f"local_{decision.behavior.value}"
    

if __name__ == "__main__":
    print("\n=== Running Local Behavior Arbitrator Test ===\n")
    printer.status("Init", "Local Behavior Arbitrator initialized", "success")

    arbitrator = LocalBehaviorArbitrator()
    print(arbitrator)

    # ---------------------------------------------------------------------
    # Test 1: Basic rule-based decisions (no memory)
    # ---------------------------------------------------------------------
    print("\n* * * * * Phase 1 – Rule‑based decisions * * * * *\n")
    arbitrator = LocalBehaviorArbitrator()
    test_cases = [
        (LocalPlanningContext(obstacle_distance_m=0.1, reverse_clearance_m=0.3), BehaviorType.BACKUP, "critical distance with reverse clearance"),
        (LocalPlanningContext(obstacle_distance_m=0.3), BehaviorType.STOP, "stop distance"),
        (LocalPlanningContext(obstacle_distance_m=0.6), BehaviorType.SLOW, "slow distance without clearance"),
        (LocalPlanningContext(obstacle_distance_m=0.6, clearance_left_m=0.5, clearance_right_m=0.2), BehaviorType.TURN, "slow distance with left clearance"),
        (LocalPlanningContext(obstacle_distance_m=1.5), BehaviorType.CONTINUE, "clear path"),
        (LocalPlanningContext(obstacle_distance_m=None), BehaviorType.CONTINUE, "no sensor data"),
    ]
    for ctx, expected, desc in test_cases:
        decision = arbitrator.decide(ctx)
        assert decision.behavior == expected, f"{desc}: expected {expected}, got {decision.behavior}"
        print(f"✓ {desc}: {decision.behavior} (reason: {decision.reason})")

    # ---------------------------------------------------------------------
    # Test 3: Edge cases and thresholds
    # ---------------------------------------------------------------------
    print("\n* * * * * Phase 3 – Edge cases * * * * *\n")
    # Exact boundaries
    ctx = LocalPlanningContext(obstacle_distance_m=0.35)  # stop_distance_m
    decision = arbitrator.decide(ctx)
    assert decision.behavior == BehaviorType.STOP
    print(f"Boundary stop: {decision.behavior}")

    ctx = LocalPlanningContext(obstacle_distance_m=0.8)  # slow_distance_m
    decision = arbitrator.decide(ctx)
    assert decision.behavior == BehaviorType.SLOW
    print(f"Boundary slow: {decision.behavior}")

    # Missing clearance but turn_clearance_m not met
    ctx = LocalPlanningContext(obstacle_distance_m=0.6, clearance_left_m=0.2, clearance_right_m=0.2)
    decision = arbitrator.decide(ctx)
    assert decision.behavior == BehaviorType.SLOW
    print(f"No turn corridor: {decision.behavior}")

    # Reverse clearance enough for backup
    ctx = LocalPlanningContext(obstacle_distance_m=0.15, reverse_clearance_m=0.3)
    decision = arbitrator.decide(ctx)
    assert decision.behavior == BehaviorType.BACKUP
    print(f"Backup triggered: {decision.behavior}")

    # ---------------------------------------------------------------------
    # Test 4: Utility methods
    # ---------------------------------------------------------------------
    print("\n* * * * * Phase 4 – Utilities * * * * *\n")
    decision = arbitrator._rule_based_decision(LocalPlanningContext(obstacle_distance_m=0.1, reverse_clearance_m=0.3))
    name = arbitrator.build_reactive_task_name(decision)
    assert name == "local_backup"
    print(f"Reactive task name: {name}")

    should_replan = arbitrator.should_trigger_short_horizon_replan(
        decision, LocalPlanningContext(horizon_seconds=2.0)
    )
    assert should_replan == True
    print(f"Short horizon replan: {should_replan}")

    # ---------------------------------------------------------------------
    # Test 5: Memory with no prior entries – fallback to rule‑based
    # ---------------------------------------------------------------------
    print("\n* * * * * Phase 5 – Empty memory * * * * *\n")
    fresh_memory = PlanningMemory()
    fresh_memory._base_state["execution_history"].clear()
    arbitrator2 = LocalBehaviorArbitrator(memory=fresh_memory)
    ctx = LocalPlanningContext(obstacle_distance_m=0.6, clearance_left_m=0.5)
    decision = arbitrator2.decide(ctx)
    assert decision.behavior == BehaviorType.TURN
    print(f"Fallback to rule‑based: {decision.behavior}")

    # ---------------------------------------------------------------------
    # Test 2: Memory learning with real PlanningMemory
    # ---------------------------------------------------------------------
    print("\n* * * * * Phase 2 – Memory learning (real PlanningMemory) * * * * *\n")
    memory = PlanningMemory()
    # Clear any existing execution_history (fresh start)
    memory._base_state["execution_history"].clear()
    arbitrator = LocalBehaviorArbitrator(memory=memory)

    # Context that would normally trigger TURN
    ctx_turn = LocalPlanningContext(
        obstacle_distance_m=0.6,
        clearance_left_m=0.5,
        clearance_right_m=0.2,
        current_speed_mps=0.5
    )
    decision = arbitrator.decide(ctx_turn)
    print(f"Initial decision: {decision.behavior} (expected TURN)")
    assert decision.behavior == BehaviorType.TURN

    # Record 5 TURN failures for this exact context
    for _ in range(5):
        arbitrator.record_outcome(ctx_turn, BehaviorType.TURN, success=False)

    # Record 3 CONTINUE successes for a very similar context
    similar_ctx = LocalPlanningContext(
        obstacle_distance_m=0.55,
        clearance_left_m=0.48,
        clearance_right_m=0.22,
        current_speed_mps=0.5
    )
    for _ in range(3):
        arbitrator.record_outcome(similar_ctx, BehaviorType.CONTINUE, success=True)

    # Now decide again – should override to CONTINUE because of the high success rate
    decision2 = arbitrator.decide(ctx_turn)
    print(f"After memory training: {decision2.behavior} (expected CONTINUE if override works)")
    assert decision2.behavior == BehaviorType.CONTINUE, "Memory override did NOT occur"
    print("✓ Memory override successful")

    print("\n=== Successfully Ran Local Behavior Arbitrator ===\n")