
import json

from datetime import datetime
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, OrderedDict, Union

from foodie.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("SLAI Food Delivery App")
printer = PrettyPrinter

class FoodieMemory:
    """
    Specialized local memory management for AI-powered food delivery operations.
    Maintains session-specific data, user context, and food-related knowledge.
    """
    
    def __init__(self):
        self.config = load_global_config()
        self.memory_config = get_config_section('foodie_memory')
        self.max_menu_items = self.memory_config.get('max_menu_items')
        self.max_history = self.memory_config.get('max_session_history')
        self.historical_data = deque(maxlen=self.max_history)

        self.session_data = {
            "current_user": None,
            "active_order": None,
            "user_preferences": {}
        }
        self.knowledge_base = {
            "menu_items": OrderedDict(),
            "cuisine_affinities": defaultdict(float),
            "dietary_restrictions": {}
        }
        self.conversation_context = []

    def set_user_context(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """Set current user context and preferences"""
        self.session_data["current_user"] = user_id
        self.session_data["user_preferences"] = preferences
        logger.info(f"Set context for user: {user_id}")

    def update_order_state(self, order_id: str, state: str, metadata: Dict[str, Any]) -> None:
        """Track active order state with metadata"""
        if not self.session_data["active_order"]:
            self.session_data["active_order"] = {"order_id": order_id}
        
        self.session_data["active_order"].update({
            "state": state,
            "last_updated": datetime.now().isoformat(),
            "metadata": metadata
        })
        logger.debug(f"Order {order_id} updated to state: {state}")

    def cache_menu_data(self, restaurant_id: str, menu_items: List[Dict[str, Any]]) -> None:
        """Cache menu items with intelligent eviction policy"""
        # Evict least accessed items if over limit
        if len(self.knowledge_base["menu_items"]) >= self.max_menu_items:
            oldest = next(iter(self.knowledge_base["menu_items"]))
            del self.knowledge_base["menu_items"][oldest]
        
        # Store new items with access tracking
        for item in menu_items:
            item_id = f"{restaurant_id}_{item['id']}"
            self.knowledge_base["menu_items"][item_id] = {
                "data": item,
                "last_accessed": datetime.now().isoformat(),
                "access_count": 0
            }

    def get_menu_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve menu item with access tracking"""
        if item_id in self.knowledge_base["menu_items"]:
            item = self.knowledge_base["menu_items"][item_id]
            item["access_count"] += 1
            item["last_accessed"] = datetime.now().isoformat()
            return item["data"]
        return None

    def learn_preference(self, food_category: str, affinity: float) -> None:
        """Adapt user preferences based on interactions"""
        self.knowledge_base["cuisine_affinities"][food_category] = max(
            0, min(1, affinity)
        )
        logger.debug(f"Updated affinity for {food_category}: {affinity}")

    def record_conversation(self, role: str, message: str, intent: str) -> None:
        """Maintain conversation context for AI continuity"""
        self.conversation_context.append({
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "message": message,
            "intent": intent
        })
        # Keep only last 10 exchanges
        if len(self.conversation_context) > 10:
            self.conversation_context.pop(0)

    def get_last_intent(self) -> Optional[str]:
        """Get most recent user intent from conversation"""
        return self.conversation_context[-1]["intent"] if self.conversation_context else None

    def get_dietary_restrictions(self, user_id: str) -> Dict[str, Any]:
        """Retrieve dietary restrictions with fallback to default"""
        return self.knowledge_base["dietary_restrictions"].get(
            user_id,
            {"gluten_free": False, "vegan": False, "allergies": []}
        )

    def save_session(self) -> Dict[str, Any]:
        """Export current session state for persistence"""
        return {
            "session": self.session_data,
            "conversation": self.conversation_context.copy(),
            "knowledge_snapshot": {
                "top_cuisines": sorted(
                    self.knowledge_base["cuisine_affinities"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            }
        }

    def load_session(self, session_data: Dict[str, Any]) -> None:
        """Import session state from persisted data"""
        self.session_data = session_data.get("session", {})
        self.conversation_context = session_data.get("conversation", [])
        
        # Merge cuisine affinities
        for cuisine, affinity in session_data.get("knowledge_snapshot", {}).get("top_cuisines", []):
            current = self.knowledge_base["cuisine_affinities"].get(cuisine, 0)
            self.knowledge_base["cuisine_affinities"][cuisine] = (current + affinity) / 2

    def reset_session(self) -> None:
        """Clear temporary session data while preserving knowledge"""
        self.historical_data.append(self.save_session())
        self.session_data = {
            "current_user": None,
            "active_order": None,
            "user_preferences": {}
        }
        self.conversation_context = []
        logger.info("Session reset with historical preservation")

    def get_recommendation_context(self) -> Dict[str, Any]:
        """Prepare context for AI recommendation engine"""
        return {
            "user_preferences": self.session_data["user_preferences"],
            "dietary_restrictions": self.get_dietary_restrictions(
                self.session_data["current_user"]
            ),
            "cuisine_affinities": dict(self.knowledge_base["cuisine_affinities"]),
            "last_order": self.historical_data[-1]["session"]["active_order"] if self.historical_data else None
        }

if __name__ == "__main__":
    print("\n=== Running Foodie Memory ===\n")
    printer.status("Init", "Foodie Memory initialized", "success")
    memory = FoodieMemory()
    
    # Set user context
    memory.set_user_context("user123", {"preferred_cuisines": ["Italian", "Japanese"]})
    
    # Record dietary restrictions
    memory.knowledge_base["dietary_restrictions"]["user123"] = {
        "vegan": True,
        "allergies": ["peanuts"]
    }
    
    # Cache menu items
    memory.cache_menu_data("restaurant456", [
        {"id": "dish1", "name": "Vegan Pasta", "category": "Italian"},
        {"id": "dish2", "name": "Avocado Roll", "category": "Japanese"}
    ])
    
    # Get AI recommendation context
    rec_context = memory.get_recommendation_context()