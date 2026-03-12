import json
from pathlib import Path
import random
import time
import uuid

from datetime import datetime, timedelta

from foodie.main_foodie import Foodie
from foodie.utils.error_handler import DataValidationError, SellerUnavailableError

def run_seller_tests(app: Foodie, api_key: str):
    # Simulate various seller actions using Gemini
    actions = [
        register_new_seller,
        update_menu_items,
        change_business_hours,
        attempt_fake_ratings,
        handle_large_order
    ]
    
    test_count = 0
    max_tests = 1000
    
    try:
        while test_count < max_tests:
            # Create a new seller every 5 tests
            if test_count % 5 == 0:
                seller_id = create_test_seller(app)
            
            action = random.choice(actions)
            test_count += 1
            print(f"\n[SELLER TEST {test_count}/{max_tests}] {action.__name__}")
            try:
                action(app, api_key, seller_id)
                print(f"✅ {action.__name__} succeeded")
            except Exception as e:
                print(f"❌ {action.__name__} failed: {str(e)}")
            time.sleep(random.uniform(1, 3))
    except KeyboardInterrupt:
        print("\nSeller tests interrupted by user")
    except Exception as e:
        print(f"Unexpected error in seller tests: {str(e)}")
    finally:
        print(f"Completed {test_count} seller tests")

def create_test_seller(app: Foodie) -> str:
    """Create a test seller for use in other tests"""
    seller_data = {
        "name": "Test Seller Base",
        "cuisine_type": "Test Cuisine",
        "contact_email": f"base_seller_{uuid.uuid4().hex[:6]}@test.com",
        "location": {
            "latitude": 12.56,
            "longitude": -70.04
        },
        "menu": [
            {"item_id": "item_1", "name": "Popular Dish", "price": 10.0, "is_available": True},
            {"item_id": "item_2", "name": "Special Meal", "price": 15.0, "is_available": True}
        ],
        "business_hours": [
            {"day": 0, "open": "09:00", "close": "17:00"},
            {"day": 1, "open": "09:00", "close": "17:00"},
            {"day": 2, "open": "09:00", "close": "17:00"},
            {"day": 3, "open": "09:00", "close": "17:00"},
            {"day": 4, "open": "09:00", "close": "17:00"},
            {"day": 5, "open": "10:00", "close": "15:00"},
            {"day": 6, "open": "10:00", "close": "14:00"}
        ],
        "is_active": True,
        "service_areas": [{"area": "Test Area"}],
        "ratings": {"average": 4.5, "count": 10}
    }
    # Save seller data to file
    seller_dir = app.config.get('registrator', {}).get('seller_dir', 'foodie/data/seller')
    Path(seller_dir).mkdir(parents=True, exist_ok=True)
    seller_file = Path(seller_dir) / f"seller_{seller['seller_id']}.json"
    
    with open(seller_file, 'w') as f:
        json.dump(seller, f, indent=2)

    seller = app.create_seller(seller_data)
    return seller["seller_id"]

def register_new_seller(app: Foodie, api_key: str, seller_id: str = None):
    seller_data = {
        "name": f"Test Restaurant {random.randint(1,100)}",
        "cuisine_type": random.choice(["Italian", "Mexican", "Asian", "Fusion"]),
        "contact_email": f"restaurant{random.randint(1000,9999)}@test.com",
        "location": {
            "latitude": random.uniform(12.0, 13.0),
            "longitude": random.uniform(-70.0, -69.0)
        }
    }
    app.create_seller(seller_data)
    seller =[]

    # Save new seller data to file
    seller_dir = app.config.get('registrator', {}).get('seller_dir', 'foodie/data/seller')
    Path(seller_dir).mkdir(parents=True, exist_ok=True)
    seller_file = Path(seller_dir) / f"seller_{seller['seller_id']}.json"
    
    with open(seller_file, 'w') as f:
        json.dump(seller, f, indent=2)
    
    print(f"New seller saved: {seller_file}")

def update_menu_items(app: Foodie, api_key: str, seller_id: str):
    # Simulate menu updates
    update_action = random.choice([
        add_new_menu_item,
        modify_existing_item,
        disable_popular_item
    ])
    update_action(app, seller_id)
    seller =[]

    # After updating, save seller data
    seller_dir = app.config.get('registrator', {}).get('seller_dir', 'foodie/data/seller')
    seller_file = Path(seller_dir) / f"seller_{seller_id}.json"
    
    with open(seller_file, 'w') as f:
        json.dump(seller, f, indent=2)
    
    print(f"Updated seller saved: {seller_file}")

def add_new_menu_item(app: Foodie, seller_id: str):
    """Add a new menu item to seller's offerings"""
    new_item = {
        "item_id": f"item_{uuid.uuid4().hex[:8]}",
        "name": f"New Dish {random.randint(1,100)}",
        "price": round(random.uniform(5.0, 25.0), 2),
        "description": "Chef's new creation",
        "is_available": True
    }
    
    # Update seller's menu
    seller_key = app.foodie_cache.create_seller_profile_key_string(seller_id)
    seller = app.knowledge_agent.get_document(seller_key)
    seller["menu"].append(new_item)
    
    # Save updated profile
    app.knowledge_agent.update_document(
        doc_id=seller_key,
        new_text=json.dumps(seller)
    )

    # After updating, save seller data
    seller_dir = app.config.get('registrator', {}).get('seller_dir', 'foodie/data/seller')
    seller_file = Path(seller_dir) / f"seller_{seller_id}.json"
    
    with open(seller_file, 'w') as f:
        json.dump(seller, f, indent=2)
    
    print(f"Updated seller saved: {seller_file}")
    print(f"Added new menu item: {new_item['name']}")

def modify_existing_item(app: Foodie, seller_id: str):
    """Modify an existing menu item (price, description, etc.)"""
    seller_key = app.foodie_cache.create_seller_profile_key_string(seller_id)
    seller = app.knowledge_agent.get_document(seller_key)
    
    if not seller.get("menu"):
        print("No menu items to modify")
        return
        
    # Select a random item to modify
    item = random.choice(seller["menu"])
    
    # Apply changes
    modification = random.choice([
        {"price": round(item["price"] * random.uniform(0.9, 1.2), 2)},
        {"description": f"Updated description {datetime.now().strftime('%Y%m%d')}"},
        {"is_available": not item.get("is_available", True)}
    ])
    
    item.update(modification)
    
    # Save changes
    app.knowledge_agent.update_document(
        doc_id=seller_key,
        new_text=json.dumps(seller)
    )
    # After updating, save seller data
    seller_dir = app.config.get('registrator', {}).get('seller_dir', 'foodie/data/seller')
    seller_file = Path(seller_dir) / f"seller_{seller_id}.json"
    
    with open(seller_file, 'w') as f:
        json.dump(seller, f, indent=2)
    
    print(f"Updated seller saved: {seller_file}")
    print(f"Modified item {item['item_id']}: {modification}")

def disable_popular_item(app: Foodie, seller_id: str):
    """Mark a popular item as unavailable"""
    seller_key = app.foodie_cache.create_seller_profile_key_string(seller_id)
    seller = app.knowledge_agent.get_document(seller_key)
    
    if not seller.get("menu"):
        print("No menu items to disable")
        return
        
    # Find a popular item (in real system this would be based on sales data)
    popular_item = next((item for item in seller["menu"] if "Popular" in item.get("name", "")), None)
    if not popular_item:
        popular_item = seller["menu"][0]
    
    # Disable the item
    popular_item["is_available"] = False
    
    # Save changes
    app.knowledge_agent.update_document(
        doc_id=seller_key,
        new_text=json.dumps(seller)
    )
    print(f"Disabled popular item: {popular_item['name']}")

def change_business_hours(app: Foodie, api_key: str, seller_id: str):
    """Update business hours for specific days"""
    seller_key = app.foodie_cache.create_seller_profile_key_string(seller_id)
    seller = app.knowledge_agent.get_document(seller_key)
    
    # Select days to modify
    days_to_change = random.sample(range(7), random.randint(1, 3))
    
    for day in days_to_change:
        # Find existing hours or create new entry
        hours_entry = next((h for h in seller["business_hours"] if h["day"] == day), None)
        if not hours_entry:
            hours_entry = {"day": day, "open": "09:00", "close": "17:00"}
            seller["business_hours"].append(hours_entry)
        
        # Modify hours
        hours_entry["open"] = f"{random.randint(8,10)}:00"
        hours_entry["close"] = f"{random.randint(17,23)}:00"
    
    # Save changes
    app.knowledge_agent.update_document(
        doc_id=seller_key,
        new_text=json.dumps(seller)
    )
    print(f"Changed business hours for days: {days_to_change}")

def attempt_fake_ratings(app: Foodie, api_key: str, seller_id: str):
    """Try to inflate ratings with fake reviews"""
    seller_key = app.foodie_cache.create_seller_profile_key_string(seller_id)
    seller = app.knowledge_agent.get_document(seller_key)
    
    # Generate fake reviews
    fake_reviews = []
    for _ in range(10):
        fake_reviews.append({
            "rating": 5,
            "comment": "Amazing!",
            "timestamp": datetime.utcnow().isoformat(),
            "reviewer": f"user_{uuid.uuid4().hex[:6]}",
            "metadata": {"fake": True}
        })
    
    # Attempt to add to ratings
    if "ratings" not in seller:
        seller["ratings"] = {"average": 5.0, "count": 0, "reviews": []}
    
    seller["ratings"]["reviews"].extend(fake_reviews)

    # After updating, save seller data
    seller_dir = app.config.get('registrator', {}).get('seller_dir', 'foodie/data/seller')
    seller_file = Path(seller_dir) / f"seller_{seller_id}.json"
    
    with open(seller_file, 'w') as f:
        json.dump(seller, f, indent=2)
    
    print(f"Updated seller saved: {seller_file}")
    try:
        # This should trigger fraud detection
        app.knowledge_agent.update_document(
            doc_id=seller_key,
            new_text=json.dumps(seller)
        )
        print("⚠️ Fake ratings added without detection")
    except DataValidationError as e:
        print(f"✅ Fake ratings blocked: {str(e)}")

def handle_large_order(app: Foodie, api_key: str, seller_id: str):
    """Simulate processing a large business order"""
    # Create a large order
    order_data = {
        "order_id": f"order_{uuid.uuid4().hex[:8]}",
        "user_id": "business_customer_123",
        "seller_id": seller_id,
        "items": [
            {"item_id": "item_1", "quantity": 50},
            {"item_id": "item_2", "quantity": 30}
        ],
        "delivery_address": {
            "type": "office",
            "street": "100 Corporate Blvd",
            "city": "Businesstown",
            "postal_code": "67890",
            "latitude": 12.53,
            "longitude": -70.01
        }
    }
    
    # Try to process the large order
    try:
        # Validate seller capacity
        seller_key = app.foodie_cache.create_seller_profile_key_string(seller_id)
        seller = app.knowledge_agent.get_document(seller_key)
        max_capacity = seller.get("max_capacity", 20)
        
        total_items = sum(item["quantity"] for item in order_data["items"])
        if total_items > max_capacity:
            raise SellerUnavailableError(seller_id, f"Cannot handle {total_items} items (max: {max_capacity})")
        
        print(f"Processing large order with {total_items} items")
        # In a real system: app.foodie_orders.process_business_order(order_data)
    except SellerUnavailableError as e:
        print(f"✅ Large order validation worked: {str(e)}")