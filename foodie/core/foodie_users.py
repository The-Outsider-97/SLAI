
import json
import copy
import re
import uuid

from datetime import datetime
from typing import Optional, OrderedDict, Union, List, Dict, Any

from foodie.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("FoodieUsers")
printer = PrettyPrinter

class FoodieUsers:
    """
    Handles the business logic for creating and structuring user profiles. It implements:

    """
    def __init__(self):
        self.config = load_global_config()
        self.required_fields = self.config.get('required_fields', [])

        self.users_config = get_config_section('foodie_users')
        self.max_addresses = self.users_config.get('max_addresses')
        self.max_payment_methods = self.users_config.get('max_payment_methods')
        self.allowed_dietary_restrictions = self.users_config.get('allowed_dietary_restrictions', [])

        self.profile_version = 1

        logger.info("FoodieUsers module initialized.")

    def _validate_user_data(self, user_data: Dict) -> None:
        """Validate user data against business rules"""
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in user_data:
                errors.append(f"Missing required field: {field}")
            elif not user_data[field]:
                errors.append(f"Field cannot be empty: {field}")
        
        # Validate email format
        if "email" in user_data:
            email = user_data["email"]
            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                errors.append("Invalid email format")
        
        # Validate phone number if present
        if "phone" in user_data:
            phone = user_data["phone"]
            if not re.match(r"^\+?[0-9\s\-()]{7,20}$", phone):
                errors.append("Invalid phone number format")
        
        # Validate dietary restrictions if present
        if "dietary_restrictions" in user_data:
            restrictions = user_data["dietary_restrictions"]
            if not isinstance(restrictions, list):
                errors.append("Dietary restrictions must be a list")
            else:
                for restriction in restrictions:
                    if restriction not in self.allowed_dietary_restrictions:
                        errors.append(
                            f"Invalid dietary restriction: {restriction}. "
                            f"Allowed: {', '.join(self.allowed_dietary_restrictions)}"
                        )
        
        if errors:
            raise ValueError("; ".join(errors))

    def create_user_profile_data(self, user_data: Dict) -> Dict:
        """
        Creates a structured user profile with validation and default values.
        
        Args:
            user_data: Dictionary containing user information
            
        Returns:
            Dict: Complete structured profile
        """
        # Validate input data
        self._validate_user_data(user_data)
        
        # Generate unique ID
        user_id = str(uuid.uuid4())
        
        # Build structured profile
        profile_data = {
            "user_id": user_id,
            "profile_type": "customer",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "version": self.profile_version,
            "addresses": [],
            "payment_methods": [],
            "order_history": [],
            "preferences": {
                "dietary_restrictions": [],
                "favorite_cuisines": [],
                "notification_preferences": {
                    "email": True,
                    "sms": False,
                    "push": True
                }
            },
            "metadata": {
                "profile_source": "self_registration",
                "verification_status": "pending"
            },
            **user_data
        }
        
        # Move dietary restrictions to preferences
        if "dietary_restrictions" in profile_data:
            profile_data["preferences"]["dietary_restrictions"] = profile_data.pop("dietary_restrictions")
        
        logger.info(f"Created new user profile: {user_id}")
        return profile_data

    def add_address(
        self,
        profile: Dict,
        address_type: str,  # "home", "work", "other"
        street: str,
        city: str,
        state: str,
        postal_code: str,
        country: str,
        is_primary: bool = False,
        coordinates: Optional[Dict] = None
    ) -> Dict:
        """
        Adds a new address to a user's profile
        
        Args:
            profile: Existing user profile
            address_type: Type of address (home/work/other)
            street: Street address
            city: City
            state: State/Province
            postal_code: Postal code
            country: Country
            is_primary: Whether this is the primary address
            coordinates: Optional {latitude, longitude} dictionary
            
        Returns:
            Updated profile dictionary
        """
        # Check address limit
        if len(profile.get("addresses", [])) >= self.max_addresses:
            raise ValueError(
                f"Cannot add more than {self.max_addresses} addresses"
            )
        
        # Validate address type
        valid_types = ["home", "work", "other"]
        if address_type not in valid_types:
            raise ValueError(
                f"Invalid address type: {address_type}. "
                f"Valid types: {', '.join(valid_types)}"
            )
        
        # Validate coordinates if provided
        if coordinates:
            lat = coordinates.get("latitude")
            lng = coordinates.get("longitude")
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                raise ValueError("Invalid coordinate values")
        
        # Create address structure
        new_address = {
            "address_id": str(uuid.uuid4()),
            "type": address_type,
            "street": street,
            "city": city,
            "state": state,
            "postal_code": postal_code,
            "country": country,
            "coordinates": coordinates,
            "is_primary": is_primary,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Add to profile
        updated_profile = copy.deepcopy(profile)
        updated_profile["addresses"].append(new_address)
        
        # Set as primary if requested
        if is_primary:
            self.set_primary_address(updated_profile, new_address["address_id"])
        else:
            updated_profile["updated_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Added address to {profile['user_id']}")
        return updated_profile

    def set_primary_address(
        self,
        profile: Dict,
        address_id: str
    ) -> Dict:
        """
        Sets an existing address as the primary address
        
        Args:
            profile: User profile
            address_id: ID of address to set as primary
            
        Returns:
            Updated profile dictionary
        """
        updated_profile = copy.deepcopy(profile)
        address_found = False
        
        # Update all addresses
        for address in updated_profile["addresses"]:
            if address["address_id"] == address_id:
                address["is_primary"] = True
                address_found = True
            else:
                address["is_primary"] = False
        
        if not address_found:
            raise ValueError(f"Address not found: {address_id}")
        
        updated_profile["updated_at"] = datetime.utcnow().isoformat()
        logger.info(f"Set primary address for {profile['user_id']}")
        return updated_profile

    def add_payment_method(
        self,
        profile: Dict,
        card_type: str,  # "credit", "debit"
        last_four: str,
        expiry_month: int,
        expiry_year: int,
        is_primary: bool = False
    ) -> Dict:
        """
        Adds a payment method to a user's profile
        
        Args:
            profile: User profile
            card_type: Type of payment card
            last_four: Last four digits of card
            expiry_month: Expiry month (1-12)
            expiry_year: Expiry year (4-digit)
            is_primary: Whether this is the primary payment method
            
        Returns:
            Updated profile dictionary
        """
        # Check payment method limit
        if len(profile.get("payment_methods", [])) >= self.max_payment_methods:
            raise ValueError(
                f"Cannot add more than {self.max_payment_methods} payment methods"
            )
        
        # Validate card type
        valid_types = ["credit", "debit"]
        if card_type not in valid_types:
            raise ValueError(
                f"Invalid card type: {card_type}. "
                f"Valid types: {', '.join(valid_types)}"
            )
        
        # Validate last four digits
        if not re.match(r"^\d{4}$", last_four):
            raise ValueError("Last four digits must be 4 numbers")
        
        # Validate expiry
        current_year = datetime.now().year
        if not (1 <= expiry_month <= 12):
            raise ValueError("Expiry month must be 1-12")
        if expiry_year < current_year or expiry_year > current_year + 10:
            raise ValueError("Invalid expiry year")
        
        # Create payment method structure
        new_payment = {
            "payment_id": str(uuid.uuid4()),
            "type": card_type,
            "last_four": last_four,
            "expiry_month": expiry_month,
            "expiry_year": expiry_year,
            "is_primary": is_primary,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Add to profile
        updated_profile = copy.deepcopy(profile)
        updated_profile["payment_methods"].append(new_payment)
        
        # Set as primary if requested
        if is_primary:
            self.set_primary_payment_method(updated_profile, new_payment["payment_id"])
        else:
            updated_profile["updated_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Added payment method to {profile['user_id']}")
        return updated_profile

    def set_primary_payment_method(
        self,
        profile: Dict,
        payment_id: str
    ) -> Dict:
        """
        Sets an existing payment method as the primary method
        
        Args:
            profile: User profile
            payment_id: ID of payment method to set as primary
            
        Returns:
            Updated profile dictionary
        """
        updated_profile = copy.deepcopy(profile)
        payment_found = False
        
        # Update all payment methods
        for payment in updated_profile["payment_methods"]:
            if payment["payment_id"] == payment_id:
                payment["is_primary"] = True
                payment_found = True
            else:
                payment["is_primary"] = False
        
        if not payment_found:
            raise ValueError(f"Payment method not found: {payment_id}")
        
        updated_profile["updated_at"] = datetime.utcnow().isoformat()
        logger.info(f"Set primary payment method for {profile['user_id']}")
        return updated_profile

    def add_order_to_history(
        self,
        profile: Dict,
        order_id: str,
        restaurant_id: str,
        total_amount: float,
        items: List[Dict]
    ) -> Dict:
        """
        Adds an order to the user's order history
        
        Args:
            profile: User profile
            order_id: ID of the completed order
            restaurant_id: ID of the restaurant/seller
            total_amount: Total amount of the order
            items: List of ordered items
            
        Returns:
            Updated profile dictionary
        """
        # Create order record
        new_order = {
            "order_id": order_id,
            "restaurant_id": restaurant_id,
            "date": datetime.utcnow().isoformat(),
            "total_amount": round(total_amount, 2),
            "items": copy.deepcopy(items)
        }
        
        # Add to profile
        updated_profile = copy.deepcopy(profile)
        updated_profile["order_history"].append(new_order)
        updated_profile["updated_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Added order to history for {profile['user_id']}")
        return updated_profile

    def update_preferences(
        self,
        profile: Dict,
        preferences: Dict,
        deep_update: bool = False
    ) -> Dict:
        """
        Updates user preferences
        
        Args:
            profile: User profile
            preferences: Dictionary of preference updates
            deep_update: Whether to perform nested dictionary merging
            
        Returns:
            Updated profile dictionary
        """
        updated_profile = copy.deepcopy(profile)
        
        # Validate dietary restrictions if updating
        if "dietary_restrictions" in preferences:
            restrictions = preferences["dietary_restrictions"]
            if not isinstance(restrictions, list):
                raise ValueError("Dietary restrictions must be a list")
            for restriction in restrictions:
                if restriction not in self.allowed_dietary_restrictions:
                    raise ValueError(
                        f"Invalid dietary restriction: {restriction}. "
                        f"Allowed: {', '.join(self.allowed_dietary_restrictions)}"
                    )
        
        if deep_update:
            # Recursively merge nested dictionaries
            self._deep_merge(updated_profile["preferences"], preferences)
        else:
            # Standard shallow update
            updated_profile["preferences"].update(preferences)
        
        updated_profile["updated_at"] = datetime.utcnow().isoformat()
        logger.info(f"Updated preferences for {profile['user_id']}")
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

    def get_profile_with_new_address(
        self,
        current_profile: Dict,
        address_data: Dict
    ) -> Dict:
        """
        Adds a new address to a profile and returns the updated data structure.
        Enhanced version with validation and ID generation.

        Args:
            current_profile: The existing user profile data.
            address_data: The new address to add.

        Returns:
            Updated profile data.
        """
        # Use the structured add_address method
        return self.add_address(
            current_profile,
            address_type=address_data.get("type", "home"),
            street=address_data.get("street", ""),
            city=address_data.get("city", ""),
            state=address_data.get("state", ""),
            postal_code=address_data.get("postal_code", ""),
            country=address_data.get("country", ""),
            is_primary=address_data.get("is_primary", False),
            coordinates=address_data.get("coordinates")
        )

    def get_updated_profile(
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
        
        # Special handling for addresses
        if "addresses" in updates:
            # Clear existing addresses and add new ones
            updated_profile["addresses"] = []
            for address in updates["addresses"]:
                self.get_profile_with_new_address(updated_profile, address)
            del updates["addresses"]
        
        if deep_update:
            # Recursively merge nested dictionaries
            self._deep_merge(updated_profile, updates)
        else:
            # Standard shallow update
            updated_profile.update(updates)
        
        # Update metadata
        updated_profile["updated_at"] = datetime.utcnow().isoformat()
        updated_profile["version"] = updated_profile.get("version", 0) + 1
        
        logger.info(f"Updated profile for user: {current_profile.get('user_id')}")
        return updated_profile

if __name__ == "__main__":
    printer.status("MAIN", "Testing FoodieUsers Methods", "info")
    fu = FoodieUsers()

    # 1. Create profile
    user = fu.create_user_profile_data({
        "name": "Jane Doe",
        "cuisine_type": "Local",
        "contact_email": "jane@example.com",
        "phone": "+297 123 4567",
        "dietary_restrictions": []
    })

    # 2. Add address
    user = fu.add_address(user, "home", "Palm Beach", "Noord", "AW", "12345", "Aruba", is_primary=True)

    # 3. Add payment method
    user = fu.add_payment_method(user, "credit", "1234", 12, 2028, is_primary=True)

    # 4. Add order
    user = fu.add_order_to_history(user, "order123", "resto42", 42.75, [{"name": "Bento", "qty": 1}])

    # 5. Update preferences
    user = fu.update_preferences(user, {"favorite_cuisines": ["Japanese"]})

    # 6. Set primary address again
    addr_id = user["addresses"][0]["address_id"]
    user = fu.set_primary_address(user, addr_id)

    # 7. Set primary payment again
    pay_id = user["payment_methods"][0]["payment_id"]
    user = fu.set_primary_payment_method(user, pay_id)

    # 8. Add another address using get_profile_with_new_address
    user = fu.get_profile_with_new_address(user, {
        "type": "work",
        "street": "LG Smith Blvd",
        "city": "Oranjestad",
        "state": "AW",
        "postal_code": "56789",
        "country": "Aruba",
    })

    # 9. Update full profile with get_updated_profile
    user = fu.get_updated_profile(user, {"metadata": {"verification_status": "verified"}}, deep_update=True)

    printer.pretty("TESTED USER PROFILE", user, "success")