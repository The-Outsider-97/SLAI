import time
import math
import heapq

from typing import Dict, Any, Tuple, List, Optional

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.execution_error import ActionFailureError, UnreachableTargetError, SoftInterrupt
from ..actions.base_action import BaseAction, ActionStatus
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Move To Action")
printer = PrettyPrinter

class MoveToAction(BaseAction):
    """
    Generic movement action using A* pathfinding and BaseAction's movement physics.
    Updates current_position over time.
    """
    name = "move_to"
    priority = 3
    preconditions = ["has_destination"]
    postconditions = ["at_destination"]
    _required_context_keys = ["current_position", "destination", "map_data"]

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        cfg = get_config_section("move_to_action") or {}

        self.base_speed = cfg.get("base_speed", 1.0)
        self.energy_cost = cfg.get("energy_cost", 0.05)
        self.replan_threshold = cfg.get("replan_threshold", 0.5)
        self.avoidance_radius = cfg.get("avoidance_radius", 1.0)
        self.path_update_interval = cfg.get("path_update_interval", 2.0)
        self.arrival_tolerance = cfg.get("arrival_tolerance", 0.1)

        self.path: List[Tuple[float, float]] = []
        self.current_target_idx: int = 0
        self.total_distance: float = 0.0
        self.distance_traveled: float = 0.0
        self.last_update_time: float = 0.0
        self.last_replan_time: float = 0.0

        self.context.setdefault("obstacles", [])
        self.context.setdefault("dynamic_objects", {})
        self.context.setdefault("energy", 10.0)

        logger.info("MoveToAction initialized")

    def _execute(self) -> bool:
        logger.info(f"Moving from {self.context['current_position']} to {self.context['destination']}")

        self._replan_path()
        if not self.path:
            raise UnreachableTargetError(self.name, self.context["destination"],
                                         self.context["current_position"])

        start_time = time.time()
        self.last_update_time = start_time
        self.last_replan_time = start_time

        # Set movement target speed using base action's movement system
        self.set_movement_target(1.0)  # full speed

        while not self._has_reached_destination():
            if self._should_interrupt():
                raise SoftInterrupt("Movement interrupted")
            if self.context.get("energy", 0) <= 0:
                raise ActionFailureError(self.name, "Energy depleted")

            elapsed = time.time() - start_time
            straight_dist = self._distance_between(self.context["current_position"],
                                                   self.context["destination"])
            max_duration = max(30.0, (straight_dist / self.base_speed) * 3.0) if self.base_speed > 0 else 300.0
            if elapsed > max_duration:
                raise ActionFailureError(self.name, f"Movement timeout after {elapsed:.1f}s")

            if self._should_replan_path():
                self._replan_path()
                if not self.path:
                    raise UnreachableTargetError(self.name, self.context["destination"],
                                                 self.context["current_position"])

            current_time = time.time()
            delta_time = current_time - self.last_update_time
            self.last_update_time = current_time

            # Update movement physics (speed, acceleration)
            current_speed, new_status = self.update_movement(delta_time)
            # Move along path using current speed
            self._move_along_path(current_speed, delta_time)

            self._avoid_obstacles()
            time.sleep(0.05)

        # Stop movement
        self.set_movement_target(0.0)
        self.update_movement(0.0)
        logger.info(f"Reached destination in {elapsed:.1f}s, traveled {self.distance_traveled:.2f} units")
        return True

    def _move_along_path(self, speed: float, delta_time: float) -> None:
        """Move current_position towards the current waypoint using given speed."""
        if not self.path or self.current_target_idx >= len(self.path):
            return

        current = self.context["current_position"]
        target = self.path[self.current_target_idx]
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        distance_to_target = math.hypot(dx, dy)

        if distance_to_target < self.arrival_tolerance:
            self.current_target_idx += 1
            return

        move_distance = min(distance_to_target, speed * delta_time)
        if distance_to_target > 0:
            ratio = move_distance / distance_to_target
            new_x = current[0] + dx * ratio
            new_y = current[1] + dy * ratio
            self.context["current_position"] = (new_x, new_y)
            self.distance_traveled += move_distance
            # Energy consumption handled by BaseAction's _consume_movement_energy
            self._consume_movement_energy(move_distance)

        # Update status based on distance to destination
        if self._distance_to_destination() < 2.0:
            self.status = ActionStatus.DECELERATING
        elif self.carry_capacity > 0:
            self.status = ActionStatus.JUGGING
        else:
            self.status = ActionStatus.WALKING

    # ------------------------ Pathfinding ------------------------------
    def _replan_path(self) -> None:
        start = self.context["current_position"]
        goal = self.context["destination"]
        grid = self._build_planning_grid()
    
        start_cell = (int(round(start[0])), int(round(start[1])))
        goal_cell = (int(round(goal[0])), int(round(goal[1])))
    
        start_cell = self._nearest_free_cell(start_cell, grid)
        goal_cell = self._nearest_free_cell(goal_cell, grid)
    
        try:
            cell_path = self._a_star_search(start_cell, goal_cell, grid)
            self.path = [(float(x), float(y)) for x, y in cell_path]
    
            # skip the first waypoint if it is effectively the current cell
            self.current_target_idx = 1 if len(self.path) > 1 else 0
            self.total_distance = self._calculate_path_distance()
            self.last_replan_time = time.time()
        except Exception as e:
            logger.error(f"Pathfinding failed: {e}")
            self.path = []

    def _nearest_free_cell(self, cell: Tuple[int, int], grid: List[List[int]]) -> Tuple[int, int]:
        x, y = cell
        rows, cols = len(grid), len(grid[0])
        x = max(0, min(x, rows - 1))
        y = max(0, min(y, cols - 1))
    
        if grid[x][y] == 0:
            return (x, y)
    
        visited = {(x, y)}
        queue = [(x, y)]
        directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
    
        while queue:
            cx, cy = queue.pop(0)
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                    if grid[nx][ny] == 0:
                        return (nx, ny)
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    
        raise RuntimeError("No free cell found")

    def _a_star_search(self, start: Tuple[int, int],
                       goal: Tuple[int, int],
                       grid: List[List[int]]) -> List[Tuple[int, int]]:
        def heuristic(a, b):
            return math.hypot(a[0]-b[0], a[1]-b[1])

        directions = [(dx, dy) for dx in (-1,0,1) for dy in (-1,0,1) if dx != 0 or dy != 0]
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                if not (0 <= nx < len(grid) and 0 <= ny < len(grid[0])):
                    continue
                if grid[nx][ny] == 1:
                    continue
                tentative_g = g_score[current] + math.hypot(dx, dy)
                neighbor = (nx, ny)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        raise RuntimeError("No path found")

    def _calculate_path_distance(self) -> float:
        dist = 0.0
        for i in range(1, len(self.path)):
            dist += math.hypot(self.path[i][0]-self.path[i-1][0],
                               self.path[i][1]-self.path[i-1][1])
        return dist

    def _avoid_obstacles(self) -> None:
        current = self.context["current_position"]
        now = time.time()
    
        if now - self.last_replan_time < 0.5:
            return
    
        for obs in self.context.get("obstacles", []):
            if self._distance_between(current, obs) < self.avoidance_radius:
                logger.warning(f"Near obstacle at {obs}, replanning")
                self._replan_path()
                return
    
        for obj_id, obj_data in self.context.get("dynamic_objects", {}).items():
            obj_pos = obj_data.get("position")
            if obj_pos and self._distance_between(current, obj_pos) < self.avoidance_radius:
                logger.info(f"Avoiding {obj_id} at {obj_pos}")
                self._adjust_path_around_object(obj_pos)
                self.last_replan_time = now
                return

    def _adjust_path_around_object(self, obj_pos: Tuple[float, float]) -> None:
        if self.current_target_idx >= len(self.path) - 1:
            return
        current = self.context["current_position"]
        next_target = self.path[self.current_target_idx]
        dx = next_target[0] - current[0]
        dy = next_target[1] - current[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return
        dx /= length
        dy /= length
        perp = (-dy, dx)
        detour_distance = self.avoidance_radius * 1.5
        detour_point = (obj_pos[0] + perp[0] * detour_distance,
                        obj_pos[1] + perp[1] * detour_distance)
        if self.current_target_idx > 0:
            self.path.insert(self.current_target_idx, detour_point)
        else:
            self.path = [current, detour_point] + self.path[1:]
        logger.debug(f"Added detour at {detour_point}")

    def _should_replan_path(self) -> bool:
        if not self.path:
            return True
        now = time.time()
        if now - self.last_replan_time > self.path_update_interval:
            self.last_replan_time = now
            return True
        if self._distance_to_path() > self.replan_threshold:
            logger.debug("Path deviation exceeded")
            self.last_replan_time = now
            return True
        if self._is_path_blocked():
            logger.debug("Path blocked")
            self.last_replan_time = now
            return True
        return False

    def _distance_to_path(self) -> float:
        if len(self.path) < 2:
            return float('inf')
        current = self.context["current_position"]
        min_dist = float('inf')
        for i in range(len(self.path)-1):
            dist = self._point_to_segment_distance(current, self.path[i], self.path[i+1])
            min_dist = min(min_dist, dist)
        return min_dist

    @staticmethod
    def _point_to_segment_distance(p, a, b):
        ax, ay = a
        bx, by = b
        px, py = p
        abx = bx - ax
        aby = by - ay
        t = ((px - ax) * abx + (py - ay) * aby) / (abx*abx + aby*aby) if (abx*abx+aby*aby) > 0 else 0
        t = max(0.0, min(1.0, t))
        proj_x = ax + t * abx
        proj_y = ay + t * aby
        return math.hypot(px - proj_x, py - proj_y)

    def _is_path_blocked(self) -> bool:
        if self.current_target_idx >= len(self.path) - 1:
            return False
        start = self.context["current_position"]
        end = self.path[self.current_target_idx]
        grid = self.context["map_data"]
        steps = max(2, int(self._distance_between(start, end) * 2))
        for i in range(1, steps+1):
            t = i / steps
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < len(grid) and 0 <= iy < len(grid[0]):
                if grid[ix][iy] == 1:
                    return True
        return False

    def _has_reached_destination(self) -> bool:
        return self._distance_to_destination() < self.arrival_tolerance

    def _distance_to_destination(self) -> float:
        current = self.context["current_position"]
        dest = self.context["destination"]
        return math.hypot(current[0]-dest[0], current[1]-dest[1])

    def _distance_between(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def _pre_execute(self):
        super()._pre_execute()
        # Set initial speed for movement
        self.status = ActionStatus.WALKING
        self.set_movement_target(1.0)
        self.distance_traveled = 0.0
        self.context.setdefault("energy", 10.0)

    def _post_execute(self, success: bool):
        super()._post_execute(success)
        if success:
            self.context["at_destination"] = True
        if "cancel_movement" in self.context:
            self.context["cancel_movement"] = False

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "path_length": len(self.path),
            "current_target_idx": self.current_target_idx,
            "distance_traveled": self.distance_traveled,
            "total_distance": self.total_distance,
        })
        return base
    
    def _build_planning_grid(self) -> List[List[int]]:
        grid = [row[:] for row in self.context["map_data"]]
        rows, cols = len(grid), len(grid[0])
    
        # mark static point obstacles onto grid
        for ox, oy in self.context.get("obstacles", []):
            cx, cy = int(round(ox)), int(round(oy))
            inflate = max(1, int(math.ceil(self.avoidance_radius)))
            for x in range(cx - inflate, cx + inflate + 1):
                for y in range(cy - inflate, cy + inflate + 1):
                    if 0 <= x < rows and 0 <= y < cols:
                        if math.hypot(x - ox, y - oy) <= self.avoidance_radius:
                            grid[x][y] = 1
        return grid


# ------------------------ Self‑Test ----------------------------------
if __name__ == "__main__":
    print("\n=== Running MoveToAction Test (Generic) ===\n")
    context = {
        "current_position": (0.0, 0.0),
        "destination": (4.0, 4.0),
        "map_data": [
            [0,0,0,0,0],
            [0,1,1,0,0],
            [0,0,1,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]
        ],
        "has_destination": True,
        "energy": 100.0,
        "obstacles": [(2.5, 2.5)]
    }

    action = MoveToAction(context)
    success = action.execute()
    printer.pretty("RESULT", "SUCCESS" if success else "FAILURE",
                   "success" if success else "error")
    print(f"Final position: {context['current_position']}")