
import json
import os
import uuid

from pathlib import Path
from flask import Flask, request, jsonify, render_template, abort, send_from_directory
from flask_cors import CORS
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

from foodie.utils.error_handler import DataValidationError
from foodie.utils.receipt_generator import generate_order_receipt
from foodie.main_foodie import Foodie
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Flask_App")
printer = PrettyPrinter

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

foodie_app = None
try:
    foodie_app = Foodie()
    logger.info("Foodie application orchestrator initialized successfully.")
except Exception as e:
    logger.error(f"FATAL: Could not initialize the Foodie application. Error: {e}", exc_info=True)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    # Settings
    font_size = db.Column(db.Integer, default=16)
    font_family = db.Column(db.String(50), default='Arial')
    max_chatlog = db.Column(db.Integer, default=200)
    chatlog_lifespan = db.Column(db.Integer, default=30)
    theme_preference = db.Column(db.String(10), default='dark')
    animations_enabled = db.Column(db.Boolean, default=True)
    notification_sound = db.Column(db.String(20), default='default')
    message_density = db.Column(db.String(20), default='cozy')
    log_auto_scroll = db.Column(db.Boolean, default=True)
    time_format = db.Column(db.String(10), default='12h')
    # Profile
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(30))
    phone = db.Column(db.String(30), unique=True, nullable=True)
    address = db.Column(db.String(200))
    card = db.Column(db.String(20)) # Storing last 4 digits is safer
    points = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    chat_messages = db.relationship('ChatMessage', backref='user', lazy=True)

    def to_dict(self):
        # This will be used to send user data to the frontend
        return {
            "userId": self.user_id,
            "username": self.username,
            "email": self.email,
            "phone": self.phone,
            "addresses": [{"type": "Primary", "address": self.address}] if self.address else [],
            "payment_methods": [{"type": "Visa", "last4": self.card}] if self.card else [],
            "orders": [], # Placeholder for order history
            "points": self.points,
            "settings": {
                'theme': self.theme_preference,
                'fontSize': self.font_size,
                'fontFamily': self.font_family,
                'animationsEnabled': self.animations_enabled,
                'notificationSound': self.notification_sound,
                'timeFormat': self.time_format
            }
        }

in_memory_users = {}

# --- Static Frontend Routes ---
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/restaurants.html')
def serve_restaurants():
    return send_from_directory(app.static_folder, 'restaurants.html')

@app.route('/indie.html')
def serve_indie():
    return send_from_directory(app.static_folder, 'indie.html')

@app.route('/business_order.html')
def serve_business_order():
    return send_from_directory(app.static_folder, 'business_order.html')

@app.route('/user_profile.html')
def serve_user_profile():
    return send_from_directory(app.static_folder, 'user_profile.html')

# --- API Endpoints ---
# These endpoints are what your JavaScript will call using fetch().
@app.route('/api/search', methods=['POST'])
def search():
    if not foodie_app:
        return jsonify({"error": "Application backend not initialized"}), 503
    data = request.get_json()
    query = data.get('query')
    filters = data.get('filters', None)
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    # For this demo, user_id is hardcoded. In a real app, this would come from a session.
    user_id = "user_demo_001" 
    
    try:
        search_results = foodie_app.order_placement(user_id=user_id, query=query, filters=filters)
        return jsonify(search_results)
    except Exception as e:
        logger.error(f"API Error in /api/search: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during search"}), 500

@app.route('/api/seller/register', methods=['POST'])
def register_seller():
    if not foodie_app:
        return jsonify({"error": "Application backend not initialized"}), 503
        
    seller_data = request.get_json()
    required_fields = [
        'company_name', 'company_address', 'company_phone',
        'registration_number', 'contact_name', 'contact_email',
        'cuisine_type'
    ]
    
    missing = [field for field in required_fields if field not in seller_data]
    if missing:
        return jsonify({
            "error": f"Missing required fields: {', '.join(missing)}"
        }), 400
        
    try:
        created_profile = foodie_app.create_seller(seller_data)
        return jsonify({
            "message": "Seller registered successfully!",
            "profile": created_profile
        }), 201
    except Exception as e:
        logger.error(f"API Error in /api/seller/register: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/seller/menu/setup', methods=['POST'])
def setup_seller_menu():
    if not foodie_app:
        return jsonify({"error": "Application backend not initialized"}), 503
        
    data = request.get_json()
    seller_id = data.get('sellerId')
    menu_items = data.get('menuItems')
    
    if not seller_id or not menu_items:
        return jsonify({"error": "Missing seller ID or menu items"}), 400
    
    try:
        # Update seller profile with new menu
        seller_key = foodie_app.foodie_cache.create_seller_profile_key_string(seller_id)
        seller_profile = foodie_app._get_document_by_id(seller_key)
        
        if not seller_profile:
            return jsonify({"error": "Seller not found"}), 404
        
        # Update menu items
        seller_profile["menu"] = menu_items
        foodie_app.knowledge_agent.add_document(
            text=json.dumps(seller_profile),
            doc_id=seller_key,
            metadata={"type": "seller_profile"}
        )
        
        # Update cache
        cache_key = foodie_app.knowledge_cache.hash_query(seller_key)
        foodie_app.knowledge_cache.set(cache_key, seller_profile)
        
        return jsonify({
            "message": "Menu updated successfully",
            "item_count": len(menu_items)
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating menu: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to update menu"}), 500

@app.route('/api/order/place', methods=['POST'])
def place_order():
    if not foodie_app:
        return jsonify({"error": "Application backend not initialized"}), 503
    order_data = request.get_json()
    if not order_data or 'user_id' not in order_data or 'items' not in order_data:
        return jsonify({"error": "Invalid order data"}), 400
    try:
        processed_order = foodie_app.order_processor(order_data)
        # In a real app, this would now trigger the execution agent
        # foodie_app.execution_agent.perform_task(...)
        return jsonify({"message": "Order placed successfully!", "order": processed_order}), 201
    except Exception as e:
        logger.error(f"API Error in /api/order/place: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/business_order/submit', methods=['POST'])
def submit_business_order():
    if not foodie_app:
        return jsonify({"error": "Application backend not initialized"}), 503
    order_data = request.get_json()
    
    # Validate required fields
    required = [
        "company_name", "company_address", "contact_name", "email",
        "details", "arrival_time", "registration_number"
    ]
    missing_fields = [field for field in required if not order_data.get(field)]
    if missing_fields:
        return jsonify({"error": f"Missing required field(s): {', '.join(missing_fields)}"}), 400
        
    try:
        order = foodie_app.create_business_order(order_data)
        processed_order = foodie_app.process_business_order(order)
        
        # Generate receipt
        receipt_format = order_data.get("format", "pdf")
        receipt_path = generate_order_receipt(
            processed_order,
            format=receipt_format
        )
        receipt_filename = os.path.basename(receipt_path)
        
        return jsonify({
            "message": "Business order submitted successfully!",
            "order_id": processed_order["order_id"],
            "matched_sellers": processed_order["processing_results"]["matched_sellers"],
            "receipt_url": f"/static/order_receipts/{receipt_filename}"
        }), 201
    except DataValidationError as e:
        logger.error(f"Validation error in business order: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"API Error in /api/business_order/submit: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Failed to process business order",
            "details": str(e)
        }), 500   

@app.route('/api/restaurants', methods=['GET'])
def get_restaurants():
    if not foodie_app:
        return jsonify({"error": "Application backend not initialized"}), 503
    try:
        # Return JSON data instead of HTML
        restaurant_data = foodie_app.get_restaurants_data()
        return jsonify(restaurant_data)
    except Exception as e:
        logger.error(f"Error in /api/restaurants: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/indie', methods=['GET'])
def get_indie():
    if not foodie_app:
        return jsonify({"error": "Application backend not initialized"}), 503
    try:
        # Return JSON data instead of HTML
        indie_data = foodie_app.get_indie_data()
        return jsonify(indie_data)
    except Exception as e:
        logger.error(f"Error in /api/indie: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/user/register', methods=['POST'])
def register_user():
    data = request.get_json()
    
    # --- Server-side validation for required fields ---
    required_fields = ['fullname', 'email', 'password', 'age', 'gender']
    if not all(k in data for k in required_fields):
        return jsonify({"error": "Missing required fields"}), 400
    
    if data['email'] in [u['email'] for u in in_memory_users.values()]:
        return jsonify({"error": "An account with this email already exists"}), 400

    # In a real application, you would use a secure hashing library like Werkzeug
    # from werkzeug.security import generate_password_hash
    # password_hash = generate_password_hash(data['password'])

    user_id = f"user_{len(in_memory_users) + 1}"
    new_user = {
        "userId": user_id,
        "username": data['fullname'],
        "email": data['email'],
        "password": data['password'],  # UNSAFE: For demo only. Use password_hash in production.
        "age": data['age'],
        "gender": data['gender'],
        "phone": data.get('phone'), # .get() for optional fields
        "addresses": [
            {
                "type": "Primary", 
                "address": data.get('address')
            }
        ] if data.get('address') else [],
        "payment_methods": [
            {
                "type": "Visa", # Assuming Visa for demo
                "last4": data.get('card', '----')[-4:]
            }
        ] if data.get('card') else [],
        "photo_url": data.get('photo'), # Placeholder for photo handling
        "orders": [],
        "points": 300  # The promised loyalty points
    }
    
    in_memory_users[user_id] = new_user
    
    # Return user data (excluding the password for security)
    profile_data = new_user.copy()
    del profile_data['password']
    
    return jsonify(profile_data), 201

@app.route('/api/user/login', methods=['POST'])
def login_user():
    data = request.get_json()
    identifier = data.get('identifier')
    password = data.get('password')

    found_user = None
    # --- Login with email, phone, or username ---
    for user in in_memory_users.values():
        is_identifier_match = (
            identifier == user['email'] or
            identifier == user.get('phone') or
            identifier == user['username']
        )
        
        # In a real app, you would use check_password_hash(user['password_hash'], password)
        if is_identifier_match and user['password'] == password:
            found_user = user
            break
    
    if found_user:
        profile_data = found_user.copy()
        del profile_data['password']
        return jsonify(profile_data)
    
    return jsonify({"error": "Invalid credentials or user not found"}), 401

# Modify the existing profile fetcher to use the in-memory store
@app.route('/api/user/profile', methods=['GET'])
def get_user_profile():
    user_id = request.args.get('userId')
    if user_id in in_memory_users:
        profile_data = in_memory_users[user_id].copy()
        if 'password' in profile_data:
            del profile_data['password']
        return jsonify(profile_data)
    
    return jsonify({"error": "User not found"}), 404

@app.route('/api/user/settings', methods=['POST'])
def save_user_settings():
    data = request.get_json()
    user_id = data.get('userId')
    settings = data.get('settings')
    
    if not user_id or not settings:
        return jsonify({"error": "Missing user ID or settings"}), 400
    
    user = User.query.filter_by(user_id=user_id).first()
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    # Update settings
    user.theme_preference = settings.get('theme', 'dark')
    user.font_size = settings.get('fontSize', 16)
    user.font_family = settings.get('fontFamily', 'Arial')
    user.animations_enabled = settings.get('animationsEnabled', True)
    user.notification_sound = settings.get('notificationSound', 'default')
    user.time_format = settings.get('timeFormat', '12h')
    
    try:
        db.session.commit()
        return jsonify({"message": "Settings updated successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500



@app.route('/api/business_order/vendors', methods=['POST'])
def get_business_vendors():
    if not foodie_app:
        return jsonify({"error": "Application backend not initialized"}), 503
        
    data = request.get_json()
    try:
        vendors = foodie_app.find_vendors_for_business(
            employee_count=data.get('employee_count', 0),
            cuisines=data.get('cuisine_preferences', []),
            location=data.get('location', '')
        )
        return jsonify(vendors)
    except Exception as e:
        logger.error(f"Error finding vendors: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/business_order/finalize', methods=['POST'])
def finalize_business_order():
    if not foodie_app:
        return jsonify({"error": "Application backend not initialized"}), 503
        
    order_data = request.get_json()
    try:
        result = foodie_app.finalize_business_order(
            order_data['order_id'],
            order_data['vendor_id']
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error finalizing order: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/seller/<seller_id>', methods=['GET'])
def get_seller(seller_id):
    if not foodie_app:
        return jsonify({"error": "Application backend not initialized"}), 503
        
    try:
        seller_key = foodie_app.foodie_cache.create_seller_profile_key_string(seller_id)
        seller_profile = foodie_app._get_document_by_id(seller_key)
        
        if not seller_profile:
            return jsonify({"error": "Seller not found"}), 404
        
        # Simplify data for frontend
        response = {
            "id": seller_id,
            "name": seller_profile.get("name", "Unknown Seller"),
            "motto": seller_profile.get("company_motto", ""),
            "logo": seller_profile.get("image_url", "img/default_seller.jpg"),
            "rating": seller_profile.get("ratings", {}).get("average", 0),
            "menu": seller_profile.get("menu", []),
            "reviews": seller_profile.get("ratings", {}).get("reviews", [])
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error retrieving seller {seller_id}: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/pages/seller_TEMPLATE.html')
def serve_seller_template():
    return send_from_directory('static/pages', 'seller_TEMPLATE.html')

# Command to create/update database:
# flask db init (only first time)
# flask db migrate -m "Added User model updates."
# flask db upgrade
@app.cli.command("init-db")
def init_db_command():
    """Creates the database tables."""
    with app.app_context():
        db.create_all()
    print("Initialized the database.")

if __name__ == '__main__':
    if foodie_app is None:
        print("\n" + "="*50)
        print("COULD NOT START SERVER: Foodie application failed to initialize.")
        print("Please check the terminal logs for errors in the 'Foodie(...)' class.")
        print("="*50 + "\n")
    else:
        # host='0.0.0.0' makes it accessible from other devices on the same network
        with app.app_context():
            db.create_all()
        app.run(host='0.0.0.0', port=5000, debug=True)