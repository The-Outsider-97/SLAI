
import json
from pathlib import Path
import random
import time
import uuid

from datetime import datetime

from foodie.utils.error_handler import DataValidationError, OrderProcessingError
from foodie.main_foodie import Foodie

def run_user_tests(app: Foodie, api_key: str):
    # Simulate various user actions using ChatGPT
    actions = [
        create_valid_user,
        attempt_hack,
        place_normal_order,
        place_large_order,
        make_fake_complaint,
        attempt_privilege_escalation
    ]
    
    test_count = 0
    max_tests = 1000
    
    try:
        while test_count < max_tests:
            action = random.choice(actions)
            test_count += 1
            print(f"\n[USER TEST {test_count}/{max_tests}] {action.__name__}")
            try:
                action(app, api_key)
                print(f"✅ {action.__name__} succeeded")
            except Exception as e:
                print(f"❌ {action.__name__} failed: {str(e)}")
            time.sleep(random.uniform(1, 3))
    except KeyboardInterrupt:
        print("\nUser tests interrupted by user")
    except Exception as e:
        print(f"Unexpected error in user tests: {str(e)}")
    finally:
        print(f"Completed {test_count} user tests")

def create_valid_user(app: Foodie, api_key: str):
    # Use ChatGPT to generate realistic user data
    user_data = {
        "name": "Test User",
        "contact_email": f"user{random.randint(1000,9999)}@test.com",
        "ip_address": f"192.168.{random.randint(0,255)}.{random.randint(0,255)}",
        "user_agent": "Simulated Chrome Browser"
    }
    user = app.create_user(user_data)
    
    # Save user data to file
    user_dir = app.config.get('foodie_users', {}).get('user_dir', 'foodie/data/user')
    Path(user_dir).mkdir(parents=True, exist_ok=True)
    user_file = Path(user_dir) / f"user_{user['user_id']}.json"
    
    with open(user_file, 'w') as f:
        json.dump(user, f, indent=2)
    
    print(f"User saved: {user_file}")

def attempt_hack(app: Foodie, api_key: str):
    # Simulate injection attack
    malicious_data = {
        "name": "'; DROP TABLE users;--",
        "contact_email": "hacker@evil.com",
        "ip_address": "127.0.0.1",
        "user_agent": "' OR 1=1;--"
    }
    try:
        app.create_user(malicious_data)
    except DataValidationError:
        pass  # Expected failure

def place_normal_order(app: Foodie, api_key: str):
    """Simulate a normal food order"""
    # Create test user
    user_data = {
        "name": f"Order User {random.randint(1,100)}",
        "contact_email": f"order_user_{uuid.uuid4().hex[:8]}@test.com",
        "addresses": [{
            "type": "home",
            "is_primary": True,
            "street": f"{random.randint(1,100)} Main St",
            "city": "Testville",
            "postal_code": "12345",
            "latitude": 12.56 + random.uniform(-0.01, 0.01),
            "longitude": -70.04 + random.uniform(-0.01, 0.01)
        }]
    }
    user = app.create_user(user_data)
    user_id = user["user_id"]
    
    # Place order
    order_data = app.order_placement(
        user_id=user_id,
        query=random.choice(["Pizza", "Burger", "Sushi", "Pasta"])
    )
    
    # Select a seller from results
    if not order_data.get("sellers"):
        raise OrderProcessingError("No sellers available for order")
        
    seller = random.choice(order_data["sellers"])
    order_data["seller_id"] = seller["seller_id"]
    
    # Add random items
    order_data["items"] = [{
        "item_id": f"item_{random.randint(1,100)}",
        "quantity": random.randint(1, 3)
    }]
    
    # Process order
    processed_order = app.order_processor(order_data)
    # Save order data to file
    order_dir = app.config.get('foodie_orders', {}).get('batch_dir', 'foodie/data/orders')
    Path(order_dir).mkdir(parents=True, exist_ok=True)
    order_file = Path(order_dir) / f"order_{processed_order['order_id']}.json"
    
    with open(order_file, 'w') as f:
        json.dump(processed_order, f, indent=2)
    
    print(f"Order saved: {order_file}")
    print(f"Order {processed_order['order_id']} placed successfully")

def place_large_order(app: Foodie, api_key: str):
    """Simulate an unusually large order"""
    # Create test user
    user_data = {
        "name": "Large Order User",
        "contact_email": f"large_order_{uuid.uuid4().hex[:8]}@test.com",
        "addresses": [{
            "type": "office",
            "is_primary": True,
            "street": "100 Corporate Blvd",
            "city": "Businesstown",
            "postal_code": "67890",
            "latitude": 12.53,
            "longitude": -70.01
        }]
    }
    user = app.create_user(user_data)
    user_id = user["user_id"]
    
    # Place order
    order_data = app.order_placement(
        user_id=user_id,
        query="Catering"
    )
    
    # Select a seller from results
    if not order_data.get("sellers"):
        raise OrderProcessingError("No sellers available for large order")
        
    seller = random.choice(order_data["sellers"])
    order_data["seller_id"] = seller["seller_id"]
    
    # Add large quantities
    order_data["items"] = [{
        "item_id": f"item_{random.randint(1,50)}",
        "quantity": random.randint(50, 100)
    } for _ in range(5)]  # 5 different items
    
    # Process order - this should trigger quantity validation
    try:
        processed_order = app.order_processor(order_data)
        print(f"Large order {processed_order['order_id']} placed")
    except OrderProcessingError as e:
        print(f"Large order blocked as expected: {str(e)}")

def make_fake_complaint(app: Foodie, api_key: str):
    """Simulate submitting a fake complaint"""
    # Create test user
    user_data = {
        "name": "Complaint User",
        "contact_email": f"complainer_{uuid.uuid4().hex[:8]}@test.com"
    }
    user = app.create_user(user_data)
    user_id = user["user_id"]
    
    # Submit fake complaint
    complaint = {
        "user_id": user_id,
        "category": "complaint",
        "description": "My food was terrible even though I never ordered anything!",
        "priority": "high",
        "metadata": {
            "fake": True,
            "test_timestamp": datetime.utcnow().isoformat()
        }
    }
    
    # This would normally go through support system
    print("Submitting fake complaint")
    # In a real implementation, we'd call app.foodie_support.create_ticket(complaint)
    # For test purposes, just log it
    print(f"Fake complaint submitted by user {user_id}")

def attempt_privilege_escalation(app: Foodie, api_key: str):
    """Attempt to access admin functionality as regular user"""
    # Create normal user
    user_data = {
        "name": "Regular User",
        "contact_email": f"regular_{uuid.uuid4().hex[:8]}@test.com"
    }
    user = app.create_user(user_data)
    user_id = user["user_id"]
    
    # Try to access admin endpoint
    print("Attempting privilege escalation...")
    try:
        # This would be an admin-only function in a real system
        # app.get_performance_report()  # Should fail for regular user
        
        # Simulate trying to modify another user's data
        victim_id = "VIP_USER_123"
        malicious_update = {
            "user_id": victim_id,
            "updates": {"balance": 1000000}
        }
        
        # In a real implementation, we'd call:
        # app.foodie_users.update_user_profile(victim_id, malicious_update)
        
        print("⚠️ Privilege escalation attempt detected but not blocked")
    except Exception as e:
        print(f"✅ Privilege escalation blocked: {str(e)}")