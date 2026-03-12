
import json
import os
import re

from typing import Optional, OrderedDict, Union, List, Dict, Any

from foodie.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Foodie Cache KeyGen")
printer = PrettyPrinter

class FoodieCache:
    """
    A stand-alone utility to generate standardized, unhashed key strings for
    different domains within the Foodie application.
    The actual hashing is handled by a separate caching system.
    """
    def __init__(self):
        self.config = load_global_config()
        self.cache_config = get_config_section('foodie_cache')

        self.key_templates = self._load_key_templates()
        # Allow version suffix in IDs (@v123)
        self.namespace_validator = re.compile(r'^[a-z][a-z0-9_]{1,30}(@v\d+)?$')
        # Updated base validator to accept both UUIDs and namespaces
        self.base_id_validator = re.compile(
            r'^([a-z][a-z0-9_]{1,30}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$'
        )

        logger.info("FoodieCache KeyGen utility initialized.")
    def _load_key_templates(self) -> Dict[str, str]:
        """Load key templates from config or use defaults"""
        default_templates = {
            "user_profile": "user_profile::{user_id}",
            "seller_profile": "seller_profile::{seller_id}",
            "search_query": "search_query::{search_term}",
            "restaurant_menu": "restaurant_menu::{restaurant_id}",
            "restaurant_menu_versioned": "restaurant_menu::{restaurant_id}@v{version}",
            "user_order_history": "user_orders::{user_id}::{date_range}",
            "composite": "{namespace}::{entity_id}::{sub_entity}",
            "versioned": "{base_key}@v{version}"
        }
        return self.config.get("key_templates", default_templates)

    def create_key_string(self, key_type: str,
        **params: Union[str, int, float]) -> str:
        """
        Generic key generator with validation and sanitization
        
        Args:
            key_type: Predefined key template identifier
            params: Key component values
            
        Returns:
            str: Generated key string
        """
        if key_type not in self.key_templates:
            raise ValueError(f"Invalid key type: {key_type}. "
                             f"Valid types: {list(self.key_templates.keys())}")
        
        template = self.key_templates[key_type]
        sanitized_params = self._sanitize_params(params)
        
        try:
            return template.format(**sanitized_params)
        except KeyError as e:
            missing = str(e).strip("'")
            raise ValueError(f"Missing required parameter: {missing}") from None

    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Sanitize and validate key parameters"""
        sanitized = {}
        for k, v in params.items():
            # Convert non-string values
            if not isinstance(v, str):
                sanitized[k] = str(v)
            else:
                sanitized[k] = v
            
            # Validate base IDs (without version suffix)
            if k.endswith("_id") or k == "namespace":
                # Extract base ID before any version suffix
                base_value = v.split('@')[0] if '@' in v else v
                self._validate_base_id(base_value)
        
        return sanitized
    
    def _validate_base_id(self, base_value: str):
        """
        Validate that a base ID (e.g., user_id, restaurant_id, entity_id) is
        a lowercase alphanumeric string (with underscores) starting with a letter,
        and no longer than 31 characters (1 + up to 30 more).
        
        Args:
            base_value (str): The raw ID to validate (without versioning suffix).
            
        Raises:
            ValueError: If the base ID is malformed or unsafe.
        """
        if not self.base_id_validator.match(base_value):
            raise ValueError(
                f"Invalid ID format: '{base_value}'. Must match pattern: "
                r"^[a-z][a-z0-9_]{1,30}$ (lowercase, alphanumeric, underscores, "
                "starts with letter, max 31 chars)"
            )

    def _validate_namespace(self, value: str):
        """Ensure namespace/ID follows naming conventions"""
        if not self.namespace_validator.match(value):
            raise ValueError(f"Invalid namespace format: {value}. "
                             "Must match ^[a-z][a-z0-9_]{1,30}$")

    # --- Domain-Specific Convenience Methods ---

    def create_user_profile_key_string(self, user_id: str) -> str:
        return self.create_key_string("user_profile", user_id=user_id)

    def create_seller_profile_key_string(self, seller_id: str) -> str:
        return self.create_key_string("seller_profile", seller_id=seller_id)

    def create_search_query_key_string(
        self,
        search_term: str,
        filters: Optional[Dict] = None
    ) -> str:
        """Enhanced with optional search filters"""
        params = {"search_term": search_term}
        if filters:
            # Create stable representation of filters
            sorted_filters = OrderedDict(sorted(filters.items()))
            params["search_term"] += f"|filters:{json.dumps(sorted_filters)}"
        return self.create_key_string("search_query", **params)

    def create_restaurant_menu_key_string(
        self,
        restaurant_id: str,
        version: Optional[int] = None
    ) -> str:
        """Add versioning support for menus"""
        if version:
            return self.create_key_string(
                "restaurant_menu_versioned",
                restaurant_id=restaurant_id,
                version=version
            )
        return self.create_key_string("restaurant_menu", restaurant_id=restaurant_id)

    def create_composite_key(
        self,
        namespace: str,
        entity_id: str,
        sub_entity: str
    ) -> str:
        """Generic composite key generator"""
        return self.create_key_string(
            "composite",
            namespace=namespace,
            entity_id=entity_id,
            sub_entity=sub_entity
        )

    def create_versioned_key(
        self,
        base_key: str,
        version: int
    ) -> str:
        """Add versioning to existing keys"""
        return self.create_key_string(
            "versioned",
            base_key=base_key,
            version=version
        )

if __name__ == "__main__":
    printer.status("MAIN", "Starting Foodie Cache Demo", "info")

    cache = FoodieCache()

    # 1. Basic usage (backward compatible)
    user_key = cache.create_user_profile_key_string("user_123")
    printer.pretty("USER KEY", user_key, "success")
    
    # 2. Search with filters
    filters = {"dietary": ["vegan", "gluten-free"], "rating": 4.5}
    search_key = cache.create_search_query_key_string("tacos", filters=filters)
    printer.pretty("SEARCH KEY", search_key, "success")
    
    # 3. Versioned resource
    menu_key_v1 = cache.create_restaurant_menu_key_string("resto_789", version=1)
    menu_key_v2 = cache.create_restaurant_menu_key_string("resto_789", version=2)
    printer.pretty("MENU KEY v1", menu_key_v1, "success")
    printer.pretty("MENU KEY v2", menu_key_v2, "success")
    
    # 4. Composite key
    composite_key = cache.create_composite_key("order", "order_456", "items")
    printer.pretty("COMPOSITE KEY", composite_key, "success")
    
    # 5. Versioned composite key
    versioned_composite = cache.create_versioned_key(composite_key, 3)
    printer.pretty("VERSIONED COMPOSITE", versioned_composite, "success")