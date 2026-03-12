
import json
import uuid
import re
import copy

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union

from foodie.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Registrator")
printer = PrettyPrinter

class Registrator:
    """
    Handles the business logic for creating and structuring Freelancer/Indie seller profiles. It implements:
    - Comprehensive data validation
    - Menu management capabilities
    - Profile versioning
    - Business hour scheduling
    - Service area definitions
    - Flexible update handling
    - Deactivation/reactivation
    """
    def __init__(self):
        self.config = load_global_config()
        self.required_fields = self.config.get('required_fields', [])

        self.registrator_config = get_config_section('registrator')
        self.allowed_cuisine_types = self.registrator_config.get('allowed_cuisine_types', [])
        self.max_menu_items = self.registrator_config.get('max_menu_items')
        self.restaurant_dir = self.registrator_config.get('restaurant_dir')
        self.seller_dir = self.registrator_config.get('seller_dir')

        self.profile_version = 1

        logger.info("Registrator module initialized.")

    def create_seller_profile_data(self, seller_data: Dict) -> Dict:
        """
        Creates a structured seller profile with validation and default values.
        
        Args:
            seller_data: Dictionary containing seller information
            
        Returns:
            Dict: Complete structured profile
        """
        # Validate input data
        self._validate_seller_data(seller_data)
        
        # Generate unique ID
        seller_id = str(uuid.uuid4())
        
        # Build structured profile
        profile_data = {
            "seller_id": seller_id,
            "profile_type": "indie_seller",
            "is_active": True,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "version": self.profile_version,
            "menu": [],
            "service_areas": [],
            "business_hours": [],
            "ratings": {
                "average": 0.0,
                "count": 0,
                "reviews": []
            },
            "metadata": {
                "profile_source": "self_registration",
                "verification_status": "pending"
            },
            **seller_data
        }
        
        logger.info(f"Created new seller profile: {seller_id}")
        return profile_data

    def _validate_seller_data(self, seller_data: Dict) -> None:
        """Validate seller data against business rules"""
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in seller_data:
                errors.append(f"Missing required field: {field}")
            elif not seller_data[field]:
                errors.append(f"Field cannot be empty: {field}")
        
        # Validate email format
        if "contact_email" in seller_data:
            email = seller_data["contact_email"]
            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                errors.append("Invalid email format")
        
        # Validate cuisine type
        if "cuisine_type" in seller_data:
            cuisine = seller_data["cuisine_type"]
            allowed_types = [t.lower() for t in self.allowed_cuisine_types]
            if cuisine.lower() not in allowed_types:
                errors.append(
                    f"Invalid cuisine type: {cuisine}. "
                    f"Allowed types: {', '.join(self.allowed_cuisine_types)}"
                )
        
        # Validate phone number if present
        if "contact_phone" in seller_data:
            phone = seller_data["contact_phone"]
            if not re.match(r"^\+?[0-9\s\-()]{7,20}$", phone):
                errors.append("Invalid phone number format")
        
        # Validate location coordinates
        if "location" in seller_data:
            loc = seller_data["location"]
            if not (-90 <= loc.get("latitude", 100) <= 90):
                errors.append("Latitude must be between -90 and 90")
            if not (-180 <= loc.get("longitude", 200) <= 180):
                errors.append("Longitude must be between -180 and 180")
        
        if errors:
            raise ValueError("; ".join(errors))

    def add_menu_item(
        self,
        profile: Dict,
        item_name: str,
        price: float,
        description: str = "",
        dietary_info: List[str] = None,
        is_available: bool = True
    ) -> Dict:
        """
        Adds a new menu item to a seller's profile
        
        Args:
            profile: Existing seller profile
            item_name: Name of the menu item
            price: Price of the item
            description: Item description
            dietary_info: Dietary information tags
            is_available: Current availability
            
        Returns:
            Updated profile dictionary
        """
        # Check menu size limit
        if len(profile.get("menu", [])) >= self.max_menu_items:
            raise ValueError(f"Cannot add more than {self.max_menu_items} menu items")
        
        exchange_rate = 1.8  # AFL to USD
        price_usd = round(price / exchange_rate, 2)
        
        # Create menu item structure
        new_item = {
            "item_id": str(uuid.uuid4()),
            "name": item_name,
            "price": round(price, 2),
            "price_usd": price_usd,
            "description": description,
            "dietary_info": dietary_info or [],
            "is_available": is_available,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Add to profile
        updated_profile = copy.deepcopy(profile)
        updated_profile["menu"].append(new_item)
        updated_profile["updated_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Added menu item '{item_name}' to {profile['seller_id']}")
        return updated_profile

    def update_menu_item(
        self,
        profile: Dict,
        item_id: str,
        updates: Dict[str, Union[str, float, bool, List]]
    ) -> Dict:
        """
        Updates an existing menu item
        
        Args:
            profile: Seller profile
            item_id: ID of menu item to update
            updates: Dictionary of fields to update
            
        Returns:
            Updated profile dictionary
        """
        updated_profile = copy.deepcopy(profile)
        
        # Find and update item
        for item in updated_profile["menu"]:
            if item["item_id"] == item_id:
                # Apply updates
                for key, value in updates.items():
                    # Special handling for price rounding
                    if key == "price":
                        item[key] = round(float(value), 2)
                    else:
                        item[key] = value
                item["updated_at"] = datetime.utcnow().isoformat()
                break
        else:
            raise ValueError(f"Menu item not found: {item_id}")
        
        updated_profile["updated_at"] = datetime.utcnow().isoformat()
        logger.info(f"Updated menu item {item_id} in {profile['seller_id']}")
        return updated_profile

    def get_updated_profile_data(
        self,
        current_profile: Dict,
        updates: Dict,
        deep_update: bool = False
    ) -> Dict:
        """
        Applies updates to a profile with flexible update strategies
        
        Args:
            current_profile: Current profile data
            updates: Fields to update
            deep_update: Whether to perform nested dictionary merging
            
        Returns:
            Updated profile data
        """
        # Create deep copy to avoid mutating original
        updated_profile = copy.deepcopy(current_profile)
        
        if deep_update:
            # Recursively merge nested dictionaries
            self._deep_merge(updated_profile, updates)
        else:
            # Standard shallow update
            updated_profile.update(updates)
        
        # Update metadata
        updated_profile["updated_at"] = datetime.utcnow().isoformat()
        updated_profile["version"] = updated_profile.get("version", 0) + 1
        
        logger.info(
            f"Updated profile for seller: {current_profile.get('seller_id')}"
        )
        return updated_profile

    def _deep_merge(self, base: Dict, updates: Dict) -> None:
        """Recursively merge nested dictionaries"""
        for key, value in updates.items():
            if (key in base and 
                isinstance(base[key], dict) and 
                isinstance(value, dict)):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def set_profile_status(
        self,
        profile: Dict,
        is_active: bool
    ) -> Dict:
        """
        Activates or deactivates a seller profile
        
        Args:
            profile: Seller profile
            is_active: New status
            
        Returns:
            Updated profile
        """
        updated_profile = copy.deepcopy(profile)
        updated_profile["is_active"] = is_active
        updated_profile["updated_at"] = datetime.utcnow().isoformat()
        
        status = "activated" if is_active else "deactivated"
        logger.info(
            f"Profile {status} for seller: {profile.get('seller_id')}"
        )
        return updated_profile

    def add_service_area(
        self,
        profile: Dict,
        area_name: str,
        polygon_coords: List[List[float]],
        delivery_fee: float = 0.0,
        min_order: float = 0.0
    ) -> Dict:
        """
        Adds a service area to the seller's profile
        
        Args:
            profile: Seller profile
            area_name: Name of service area
            polygon_coords: List of [lat, lng] coordinates defining the area
            delivery_fee: Fee for this area
            min_order: Minimum order amount
            
        Returns:
            Updated profile
        """
        # Validate coordinates
        for coord in polygon_coords:
            if len(coord) != 2:
                raise ValueError("Coordinates must be [lat, lng] pairs")
            lat, lng = coord
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                raise ValueError("Invalid coordinate values")
        
        # Create area definition
        new_area = {
            "area_id": str(uuid.uuid4()),
            "name": area_name,
            "polygon": polygon_coords,
            "delivery_fee": delivery_fee,
            "min_order": min_order
        }
        
        updated_profile = copy.deepcopy(profile)
        updated_profile["service_areas"].append(new_area)
        updated_profile["updated_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Added service area '{area_name}' to {profile['seller_id']}")
        return updated_profile

    def add_business_hours(
        self,
        profile: Dict,
        day_of_week: int,  # 0=Monday, 6=Sunday
        open_time: str,     # "HH:MM" format
        close_time: str    # "HH:MM" format
    ) -> Dict:
        """
        Adds business hours to the seller's profile
        
        Args:
            profile: Seller profile
            day_of_week: Integer representing day (0-6)
            open_time: Opening time (e.g., "09:00")
            close_time: Closing time (e.g., "17:00")
            
        Returns:
            Updated profile
        """
        # Validate day
        if not 0 <= day_of_week <= 6:
            raise ValueError("Day of week must be 0-6")
        
        # Validate time formats
        time_pattern = re.compile(r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
        if not time_pattern.match(open_time):
            raise ValueError("Invalid open_time format")
        if not time_pattern.match(close_time):
            raise ValueError("Invalid close_time format")
        
        # Create hours entry
        new_hours = {
            "day": day_of_week,
            "open": open_time,
            "close": close_time
        }
        
        updated_profile = copy.deepcopy(profile)
        updated_profile["business_hours"].append(new_hours)
        updated_profile["updated_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Added business hours to {profile['seller_id']}")
        return updated_profile

    def add_review(
        self,
        profile: Dict,
        rating: int,  # 1-5
        comment: str,
        reviewer_id: str
    ) -> Dict:
        """
        Adds a review to the seller's profile and updates average rating
        
        Args:
            profile: Seller profile
            rating: Rating value (1-5)
            comment: Review text
            reviewer_id: ID of the reviewer
            
        Returns:
            Updated profile
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")
        
        # Create review structure
        new_review = {
            "review_id": str(uuid.uuid4()),
            "rating": rating,
            "comment": comment,
            "reviewer_id": reviewer_id,
            "created_at": datetime.utcnow().isoformat()
        }
        
        updated_profile = copy.deepcopy(profile)
        ratings = updated_profile["ratings"]
        
        # Add review
        ratings["reviews"].append(new_review)
        
        # Update rating statistics
        total_reviews = len(ratings["reviews"])
        total_rating = sum(r["rating"] for r in ratings["reviews"])
        ratings["average"] = round(total_rating / total_reviews, 1)
        ratings["count"] = total_reviews
        
        updated_profile["updated_at"] = datetime.utcnow().isoformat()
        logger.info(f"Added review to {profile['seller_id']}")
        return updated_profile

    def save_restaurant_profile(self, profile_data: Dict) -> None:
        """Saves restaurant profile to JSON file in specified directory"""
        try:
            # Create directory if it doesn't exist
            directory = Path(self.restaurant_dir)
            directory.mkdir(parents=True, exist_ok=True)
            
            # Sanitize restaurant name for filename
            name = profile_data.get("name", "unknown")
            sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '', name.replace(' ', '_'))
            
            # Generate filename
            seller_id_short = profile_data['seller_id'][:3]
            filename = f"restaurant_{seller_id_short}_{sanitized_name}.json"
            filepath = directory / filename
            
            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved restaurant profile to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving restaurant profile: {str(e)}")
            raise

if __name__ == "__main__":
    printer.status("MAIN", "Starting Foodie Cache Demo", "info")

    registra = Registrator()
    print(registra)

    # Create seller
    seller_data = {
        "name": "Mini Deli",
        "cuisine_type": registra.allowed_cuisine_types[0] if registra.allowed_cuisine_types else "Local",
        "contact_email": "minideli@example.com"
    }
    profile = registra.create_seller_profile_data(seller_data)

    # Add a menu item
    profile = registra.add_menu_item(profile, "Bento Box", 12.5, "Delicious combo", ["gluten-free"])

    # Update that menu item
    item_id = profile["menu"][0]["item_id"]
    profile = registra.update_menu_item(profile, item_id, {"price": 13.0, "is_available": False})

    # Add service area
    profile = registra.add_service_area(profile, "Downtown", [[12.3, 45.6], [12.4, 45.7]])

    # Add business hours
    profile = registra.add_business_hours(profile, 0, "09:00", "17:00")

    # Add a review
    profile = registra.add_review(profile, 5, "Great food!", "user_abc")

    # Update profile with new field
    profile = registra.get_updated_profile_data(profile, {"new_field": "test"})

    # Deactivate profile
    profile = registra.set_profile_status(profile, False)

    printer.pretty("TESTED PROFILE", profile, "success")