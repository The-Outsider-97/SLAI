
import time
import math
import heapq

from typing import Dict, Any, Tuple, List, Optional

from src.agents.execution.utils.config_loader import load_global_config, get_config_section
from src.agents.execution.utils.execution_error import ActionInterruptionError
from src.agents.execution.actions.base_action import BaseAction, ActionStatus
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Move To Action")
printer = PrettyPrinter

class MoveToAction(BaseAction):
    name = "move_to"
    priority = 3
    preconditions = ["has_destination"]
    postconditions = ["at_destination"]
    _required_context_keys = ["current_position", "destination", "map_data"]
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        self.config = load_global_config()
        self.move_config = get_config_section("move_to_action")
        
        # Movement parameters
        self.path: List[Tuple[float, float]] = []
        self.current_target_idx = 0
        self.total_distance = 0.0
        self.distance_traveled = 0.0
        self.last_update_time = 0.0
        
        # Configuration
        self.base_speed = self.move_config.get("base_speed")
        self.energy_cost = self.move_config.get("energy_cost")  # per unit distance
        self.replan_threshold = self.move_config.get("replan_threshold")
        self.avoidance_radius = self.move_config.get("avoidance_radius")
        self.path_update_interval = self.move_config.get("path_update_interval")
        
        # Initialize from context
        self.context.setdefault("obstacles", [])
        self.context.setdefault("dynamic_objects", {})

        logger.info(f"Move To Action initialized")

    def _execute(self) -> bool:
        """Execute movement to destination with pathfinding and obstacle avoidance"""
        printer.status("MOVE", "Executing movement...", "info")

        try:
            # Initialize movement parameters
            self._initialize_movement()
            start_time = time.time()
            self.last_update_time = start_time

            # Calculate max duration based on distance and speed
            start_pos = self.context["current_position"]
            goal_pos = self.context["destination"]
            straight_distance = math.sqrt(
                (goal_pos[0]-start_pos[0])**2 + 
                (goal_pos[1]-start_pos[1])**2
            )
            
            # Allow 3x the estimated time plus a minimum buffer
            max_duration = max(30, (straight_distance / self.base_speed) * 3) if self.base_speed > 0 else 300
            
            # Main movement loop
            while not self._has_reached_destination():
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > max_duration:
                    logger.error(f"Movement timed out after {elapsed:.1f}s (max: {max_duration:.1f}s)")
                    return False

                current_time = time.time()
                delta_time = current_time - self.last_update_time
                self.last_update_time = current_time
                
                # Handle interruptions
                if self._should_interrupt():
                    return False
                    
                # Recalculate path periodically or when blocked
                if self._should_replan_path():
                    self._calculate_path()
                    
                # Update movement physics
                self._update_movement(delta_time)
                
                # Check for obstacles and avoid
                self._avoid_obstacles()
                
                # Update context position
                self.context["current_position"] = self._get_current_position()
                
                # Consume energy
                self._consume_energy(delta_time)
                
                # Sleep for simulation (remove in real-time systems)
                time.sleep(0.05)
                
            logger.info(f"Reached destination in {time.time()-start_time:.1f}s")
            return True

        except KeyboardInterrupt:
            logger.info("Movement gracefully interrupted")
            self.context["cancel_movement"] = True
            return False

    def _initialize_movement(self):
        """Prepare for movement execution"""
        printer.status("MOVE", "Initializing movement", "info")

        self.set_carry_capacity(self.context.get("carrying_items", 0))
        self.set_movement_target(self.movement_speed)
        self._calculate_path()
        logger.info(f"Moving from {self.context['current_position']} to {self.context['destination']}")

    def _calculate_path(self):
        """Calculate optimal path using A* algorithm"""
        printer.status("MOVE", "Calculating path...", "info")

        start = self.context["current_position"]
        goal = self.context["destination"]
        grid = self.context["map_data"]
        
        try:
            self.path = self._a_star_search(start, goal, grid)
            self.current_target_idx = 0
            self.total_distance = self._calculate_path_distance()
            self.distance_traveled = 0.0
            logger.debug(f"New path calculated with {len(self.path)} points")
        except Exception as e:
            logger.error(f"Pathfinding failed: {str(e)}")
            self.path = [start, goal]  # Fallback to direct path

    def _a_star_search(self, start: Tuple[float, float],
                       goal: Tuple[float, float], grid: List[List[int]]) -> List[Tuple[float, float]]:
        """A* pathfinding algorithm implementation"""
        printer.status("MOVE", "Pathfinding algorithm...", "info")
    
        # Heuristic function (Euclidean distance)
        def heuristic(a, b):
            return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
            
        # Neighbor directions (8-way movement)
        directions = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]
        
        # Priority queue: (f_score, position)
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            # Reconstruct path if goal reached
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
                
            # Check neighbors
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds and obstacles using modulo indexing
                grid_x = int(neighbor[0]) % len(grid)
                grid_y = int(neighbor[1]) % len(grid[0])
                if grid[grid_x][grid_y] == 1:  # 1 = obstacle
                    continue
                    
                # Calculate tentative g_score
                tentative_g = g_score[current] + math.sqrt(dx**2 + dy**2)
                
                # Update if better path found
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        raise RuntimeError("No path found to destination")

    def _update_movement(self, delta_time: float):
        """Update position based on movement physics"""
        printer.status("MOVE", "Updating movement...", "info")

        # Get current and next target positions
        current_pos = self._get_current_position()
        target_pos = self.path[self.current_target_idx]
        
        # Calculate direction vector
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        # Move toward target
        if distance > 0.01:  # Minimum distance threshold
            # Calculate movement vector
            move_distance = min(distance, self._current_speed * delta_time)
            ratio = move_distance / distance
            new_pos = (
                current_pos[0] + dx * ratio,
                current_pos[1] + dy * ratio
            )
            
            # Update progress
            self.distance_traveled += move_distance
            
            # Update context position
            self.context["current_position"] = new_pos
            
            # Update movement state
            self._update_movement_status()
        else:
            # Move to next target in path
            if self.current_target_idx < len(self.path) - 1:
                self.current_target_idx += 1

    def _update_movement_status(self):
        """Update action status based on movement characteristics"""
        printer.status("MOVE", "Updating movement...", "info")

        # Near destination - decelerate
        if self._distance_to_destination() < 2.0:
            self.status = ActionStatus.DECELERATE
        # Carrying items - jugging movement
        elif self.carry_capacity > 0:
            self.status = ActionStatus.JUGGING
        # Normal movement
        else:
            self.status = ActionStatus.WALKING

    def _avoid_obstacles(self):
        """Dynamic obstacle avoidance"""
        printer.status("MOVE", "Avoiding obstacles...", "info")
    
        current_pos = self.context["current_position"]
        
        # Check static obstacles
        for obstacle in self.context.get("obstacles", []):
            if self._distance_to_point(obstacle) < self.avoidance_radius:
                logger.warning(f"Near obstacle at {obstacle}, recalculating path")
                self._calculate_path()
                return
                
        # Check dynamic objects
        for obj_id, obj_data in self.context.get("dynamic_objects", {}).items():
            obj_pos = obj_data["position"]
            distance = self._distance_to_point(obj_pos)
            if distance < self.avoidance_radius:
                logger.info(f"Avoiding dynamic object {obj_id} at {obj_pos}")
                logger.debug(f"Current position: {current_pos}, Distance: {distance:.2f}")
                self._adjust_path_around_object(obj_pos)

    def _adjust_path_around_object(self, object_pos: Tuple[float, float]):
        """Create local avoidance around an object"""
        printer.status("MOVE", "Adjuting path...", "info")

        if self.current_target_idx >= len(self.path) - 1:
            return
            
        # Create detour point
        current_pos = self.context["current_position"]
        next_target = self.path[self.current_target_idx]
        
        # Calculate perpendicular vector
        dx = next_target[0] - current_pos[0]
        dy = next_target[1] - current_pos[1]
        length = math.sqrt(dx**2 + dy**2)
        
        if length > 0:
            # Normalize
            dx /= length
            dy /= length
            
            # Perpendicular vector
            perp = (-dy, dx)
            
            # Create detour point
            detour_distance = self.avoidance_radius * 1.5
            detour_point = (
                object_pos[0] + perp[0] * detour_distance,
                object_pos[1] + perp[1] * detour_distance
            )
            
            # Insert detour into path
            if self.current_target_idx > 0:
                self.path.insert(self.current_target_idx, detour_point)
            else:
                self.path = [current_pos, detour_point] + self.path[1:]
                
            logger.debug(f"Added detour at {detour_point}")

    def _consume_energy(self, delta_time: float):
        """Update energy based on movement"""
        printer.status("MOVE", "Consuming energy", "info")

        if "energy" in self.context:
            # Base consumption + carrying penalty
            consumption = self.energy_cost * self._current_speed * delta_time
            consumption *= (1 + 0.3 * self.carry_capacity)  # 30% penalty per item
            
            self.context["energy"] = max(0, self.context["energy"] - consumption)
            
            # Check for exhaustion
            if self.context["energy"] <= 0:
                logger.error("Energy depleted during movement!")
                self._handle_failure("Energy depleted")
                raise RuntimeError("Movement halted due to energy depletion")
            logger.debug(f"Energy consumed: {consumption:.2f}, Remaining: {self.context['energy']:.2f}")

    def _should_interrupt(self) -> bool:
        """Check if movement should be interrupted"""
        printer.status("MOVE", "Interrupting...", "info")

        # External interruption signal
        if self.context.get("cancel_movement", False):
            return True
            
        # Critical event requires attention
        if self.context.get("urgent_event", False):
            return True
            
        # Agent cancellation
        if self.status == ActionStatus.CANCELLED:
            return True
            
        return False

    def _should_replan_path(self) -> bool:
        """Determine if path should be recalculated"""
        printer.status("MOVE", "Recalculating path...", "info")

        # First run
        if not self.path:
            return True
            
        # Periodic update
        if time.time() - self.last_update_time > self.path_update_interval:
            return True
            
        # Significant deviation from path
        if self._distance_to_path() > self.replan_threshold:
            return True
            
        # Blocked path segment
        if self._is_path_blocked():
            return True
            
        return False

    def _is_path_blocked(self) -> bool:
        """Check if current path segment is blocked"""
        printer.status("MOVE", "Checking path...", "info")

        if self.current_target_idx >= len(self.path) - 1:
            return False
            
        # Check line of sight to next point
        current_pos = self.context["current_position"]
        next_target = self.path[self.current_target_idx]
        
        # Simple grid-based check (could be enhanced with raycasting)
        grid = self.context["map_data"]
        steps = max(2, int(self._distance_to_point(next_target) * 2))
        
        for i in range(1, steps):
            ratio = i / steps
            check_pos = (
                current_pos[0] * (1-ratio) + next_target[0] * ratio,
                current_pos[1] * (1-ratio) + next_target[1] * ratio
            )
            
            # Convert to grid coordinates
            grid_x, grid_y = int(check_pos[0]), int(check_pos[1])
            
            # Check if valid grid position
            if 0 <= grid_x < len(grid) and 0 <= grid_y < len(grid[0]):
                if grid[grid_x][grid_y] == 1:  # Obstacle
                    return True
                    
        return False

    def _has_reached_destination(self) -> bool:
        """Check if destination has been reached"""
        printer.status("MOVE", "Checking destination", "info")

        return self._distance_to_destination() < 0.1

    def _distance_to_destination(self) -> float:
        """Calculate distance to final destination"""
        printer.status("MOVE", "Calculating destination", "info")

        current = self.context["current_position"]
        dest = self.context["destination"]
        return math.sqrt((current[0]-dest[0])**2 + (current[1]-dest[1])**2)

    def _distance_to_path(self) -> float:
        """Calculate distance to planned path"""
        printer.status("MOVE", "Calculating planned path", "info")

        if not self.path or self.current_target_idx >= len(self.path):
            return float('inf')
            
        current_pos = self.context["current_position"]
        segment_start = self.path[self.current_target_idx]
        
        if self.current_target_idx > 0:
            prev_point = self.path[self.current_target_idx-1]
            # Calculate distance to line segment
            return self._point_to_line_distance(current_pos, prev_point, segment_start)
        else:
            return self._distance_to_point(segment_start)

    def _point_to_line_distance(self, point, line_start, line_end) -> float:
        """Calculate distance from point to line segment"""
        printer.status("MOVE", "Calculating distance from point to line segment", "info")

        # Vector math for line segment distance
        line_vec = (line_end[0]-line_start[0], line_end[1]-line_start[1])
        point_vec = (point[0]-line_start[0], point[1]-line_start[1])
        line_len = math.sqrt(line_vec[0]**2 + line_vec[1]**2)
        
        if line_len == 0:
            return self._distance_to_point(line_start)
            
        line_unit = (line_vec[0]/line_len, line_vec[1]/line_len)
        point_proj = point_vec[0]*line_unit[0] + point_vec[1]*line_unit[1]
        
        if point_proj < 0:
            return self._distance_to_point(line_start)
        elif point_proj > line_len:
            return self._distance_to_point(line_end)
        else:
            proj_point = (
                line_start[0] + line_unit[0] * point_proj,
                line_start[1] + line_unit[1] * point_proj
            )
            return self._distance_to_point(proj_point)

    def _distance_to_point(self, point: Tuple[float, float]) -> float:
        """Calculate distance to a specific point"""
        printer.status("MOVE", "Calculating distance to point", "info")

        current = self.context["current_position"]
        return math.sqrt((current[0]-point[0])**2 + (current[1]-point[1])**2)

    def _get_current_position(self) -> Tuple[float, float]:
        """Get current position with type safety"""
        printer.status("MOVE", "Getting current position", "info")

        pos = self.context["current_position"]
        return (float(pos[0]), float(pos[1]))

    def _calculate_path_distance(self) -> float:
        """Calculate total path distance"""
        printer.status("MOVE", "Calculating total path distance", "info")

        distance = 0.0
        for i in range(1, len(self.path)):
            p1 = self.path[i-1]
            p2 = self.path[i]
            distance += math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        return distance

    def _pre_execute(self):
        """Specialized setup for movement"""
        printer.status("MOVE", "Pre execute", "info")

        super()._pre_execute()
        # Initialize energy tracking
        self.context.setdefault("energy", 10.0)
        self._current_speed = self._target_speed
        logger.info("Beginning movement sequence")

    def _post_execute(self, success: bool):
        """Cleanup after movement completes"""
        printer.status("MOVE", "Post execute", "info")

        super()._post_execute(success)
        # Reset movement flags
        if "cancel_movement" in self.context:
            self.context["cancel_movement"] = False
            
        # Apply destination reached
        if success:
            self.context["at_destination"] = True

    def to_dict(self) -> Dict[str, Any]:
        """Enhanced serialization with movement info"""
        base = super().to_dict()
        base.update({
            "path_length": len(self.path),
            "distance_traveled": self.distance_traveled,
            "total_distance": self.total_distance,
            "path_progress": f"{self.current_target_idx}/{len(self.path)}",
            "current_target": self.path[self.current_target_idx] if self.path else None
        })
        return base

if __name__ == "__main__":
    print("\n=== Running Execution MOVE_TO Action ===\n")
    printer.status("TEST", "Starting Execution MOVE_TO Action tests", "info")
    
    # Test context with simple grid
    context = {
        "current_position": (0, 0),
        "destination": (4, 4),
        "map_data": [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ],
        "energy": 10.0
    }
    
    move_action = MoveToAction(context)
    print(f"{move_action}")
    
    print("\n* * * * * Phase 2 - Pathfinding * * * * *\n")
    
    # Test pathfinding
    move_action._calculate_path()
    printer.pretty("PATH", move_action.path, "success")
    
    print("\n* * * * * Phase 3 - Movement Simulation * * * * *\n")
    
    # Simulate movement
    move_action._initialize_movement()
    for _ in range(10):
        move_action._update_movement(0.1)
        pos = move_action.context["current_position"]
        printer.pretty("POSITION", f"({pos[0]:.2f}, {pos[1]:.2f})", "info")

    print("\n* * * * * Phase 4 - Execute * * * * *\n")

    try:
        printer.pretty("EXECUTION", move_action._execute(), "success")
    except ActionInterruptionError:
        printer.pretty("INTERRUPT", "Action safely interrupted", "warning")

    print("\n=== All tests completed successfully! ===\n")
