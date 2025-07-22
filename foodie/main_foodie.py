__version__ = "1.0.0"

import hashlib
import json
import time 
import uuid
import numpy as np

from copy import deepcopy
from functools import lru_cache
from collections import defaultdict
from difflib import get_close_matches
from datetime import datetime, timedelta#, time 
from typing import List, Optional, OrderedDict, Dict
from dotenv import load_dotenv
load_dotenv()

from foodie.utils.config_loader import load_global_config, get_config_section
from foodie.utils.foodie_memory import FoodieMemory
from foodie.utils.error_handler import (DataValidationError, GeolocationError,
    CacheRetrievalFailure, InvalidOrderStructureError, SellerUnavailableError,
    ItemOutOfStockError, OrderProcessingError, MissingAddressError
)
from foodie.utils.web_socket import WebSocketManager
from foodie.core.foodie_support import FoodieSupport
from foodie.core.foodie_orders import FoodieOrders
from foodie.core.foodie_cache import FoodieCache # Module that handles cache key generation by working with knowledge_cache in Foodie
from foodie.core.foodie_users import FoodieUsers # Module that handles user's profiles
from foodie.core.registrator import Registrator # Module that handles Freelancer/Indie seller profiles
from foodie.core.foodie_card import FoodieCard
from foodie.core.foodie_map import FoodieMap
from foodie.core.foodie_security import FoodieSecurity
from src.agents.knowledge.knowledge_cache import KnowledgeCache
from src.agents.planning.planning_types import Task, TaskType
from src.agents.collaborative.shared_memory import SharedMemory
from src.agents.agent_factory import AgentFactory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("SLAI Food Delivery App")
printer = PrettyPrinter

class Foodie:
    def __init__(self):
        self.local_memory = FoodieMemory()  # User session management
        self.shared_memory = SharedMemory()  # Inter-agent coordination
        self.agent_factory = AgentFactory()

        self.config = load_global_config()
        self.central_hub = self.config.get('central_hub')
        self.main_config = get_config_section('main')

        self.foodie_users = FoodieUsers()
        self.registrator = Registrator()
        self.foodie_orders = FoodieOrders()
        self.foodie_card = FoodieCard()
        self.foodie_map = FoodieMap()
        self.foodie_security = FoodieSecurity()
        self.foodie_support = FoodieSupport()
        self.foodie_cache = FoodieCache()          # Key generator
        self.knowledge_cache = KnowledgeCache()    # Low-level cache for storage/hashing

        # Initialize other agents
        self.agent_factory = AgentFactory(config={"knowledge_agent": {"cache": self.knowledge_cache}})
        self.planning_agent = self.agent_factory.create("planning", self.shared_memory)     # Vital for logistics.
        self.knowledge_agent = self.agent_factory.create("knowledge", self.shared_memory)   # Crucial for managing all the data.
        self.execution_agent = self.agent_factory.create("execution", self.shared_memory)   # The "doer" for delivery tasks.
        self.adaptive_agent = self.agent_factory.create("adaptive", self.shared_memory)   # Real-time adjustments and responsiveness.
        self.safety_agent = self.agent_factory.create("safety", self.shared_memory)   # Safety and security of the platform and its users.
        self.evaluation_agent = self.agent_factory.create("evaluation", self.shared_memory)   # Monitors the performance of the platform.
        self.reasoning_agent = self.agent_factory.create("reasoning", self.shared_memory)   # Handles complex decision-making and problem-solving.

        self.execution_agent.attach_adaptive(self.adaptive_agent)
        self.foodie_security.validate_action = self.foodie_security.validate_action
        self.foodie_support.set_reasoning_agent(self.reasoning_agent)
        self._setup_adaptive_connections()
        self._setup_safety_connections()
        self._register_support_reasoning_tasks()

        self.cache_key={}
        # Link the actual cache instance to the Knowledge Agent
        self._prime_caches()

        # Initialize session context
        self.shared_memory.put("system:session_context", {
            "active_users": {},
            "ongoing_orders": {}
        }, ttl=timedelta(hours=24))

        self.shared_memory.put("evaluation_agent", self.evaluation_agent)

        # Initialize performance metrics dictionary
        self.shared_memory.put("performance_metrics", {
            "delivery_times": [],
            "order_accuracy": [],
            "courier_efficiency": defaultdict(list),
            "seller_performance": defaultdict(lambda: {"orders": 0, "rating_sum": 0.0}),
            "customer_satisfaction": []
        })

        # Validate critical configuration
        if not hasattr(self.planning_agent, 'register_task'):
            logger.error("PlanningAgent missing task registration capability")
            raise TypeError("Incompatible PlanningAgent implementation")

        self._register_delivery_tasks()

        logger.info("Orchestrator: Initializing agents for SLAI Foodie...")

    @property
    def cache(self):
        if not hasattr(self, '_cache') or self._cache is None:
            logger.critical("KnowledgeAgent cache not initialized!")
            raise SystemError("KnowledgeAgent cache missing")
        return self._cache
    
    @cache.setter
    def cache(self, value):
        self._cache = value

    def _setup_adaptive_connections(self):
        """Connect AdaptiveAgent to other components"""
        # Share FoodieMap with AdaptiveAgent
        self.adaptive_agent.register_utility('map', self.foodie_map)
        
        # Connect to PlanningAgent for real-time adjustments
        self.adaptive_agent.register_callback('route_adjustment', self.handle_route_adjustment)
        self.adaptive_agent.connect_planner(self.planning_agent)
    
    def handle_route_adjustment(self, adjustment_data):
        """Process real-time route adjustments from AdaptiveAgent"""
        order_id = adjustment_data['order_id']
        new_route = adjustment_data['new_route']
        
        # Update shared memory
        self.shared_memory.put(
            key=f"delivery_plan:{order_id}",
            value={
                "plan": new_route,
                "adjusted_at": datetime.utcnow()
            },
            ttl=timedelta(hours=2))
        
        # Notify ExecutionAgent
        self.execution_agent.update_active_task(
            order_id,
            {"route": new_route}
        )
        logger.info(f"Route adjusted for order {order_id}")


    def _setup_safety_connections(self):
        """Connect SafetyAgent to critical components"""
        # Connect to security module
        self.safety_agent.register_utility('security', self.foodie_security)
        
        # Connect to knowledge agent for data validation
        self.safety_agent.register_utility('knowledge', self.knowledge_agent)
        
        # Register safety callbacks
        self.foodie_security.register_safety_callback(self.safety_agent.perform_task)
        self.knowledge_agent.register_safety_check(self.safety_agent.validate_action)


    def _register_support_reasoning_tasks(self):
        """Register specialized reasoning tasks for support operations"""
        # Rule for complaint resolution
        def complaint_resolution_rule(kb):
            unresolved = [t for t in self.foodie_support.get_tickets() 
                         if t['status'] == 'open' and t['category'] == 'complaint']
            resolutions = {}
            for ticket in unresolved:
                # Complex reasoning to determine resolution path
                resolution_path = self.reasoning_agent.reason(
                    problem=ticket['description'],
                    reasoning_type='abduction+cause_effect'
                )
                resolutions[ticket['id']] = resolution_path
            return resolutions
        
        self.reasoning_agent.add_rule(
            complaint_resolution_rule,
            "complaint_resolution",
            weight=0.9
        )
        
        # Rule for dispute mediation
        def dispute_mediation_rule(kb):
            disputes = [t for t in self.foodie_support.get_tickets() 
                       if 'dispute' in t['category']]
            resolutions = {}
            for dispute in disputes:
                # Multi-step reasoning for fair mediation
                mediation_plan = self.reasoning_agent.react_loop(
                    f"Resolve dispute between {dispute['parties']} regarding {dispute['issue']}",
                    max_steps=7
                )
                resolutions[dispute['id']] = mediation_plan
            return resolutions
        
        self.reasoning_agent.add_rule(
            dispute_mediation_rule,
            "dispute_mediation",
            weight=0.85
        )


    def _prime_caches(self):
        # Preload common seller profiles using existing methods
        seller_profiles = self.knowledge_agent.retrieve_documents_by_type("seller_profile")
        
        # Sort by average rating (if available) descending
        seller_profiles.sort(
            key=lambda doc: doc.get('ratings', {}).get('average', 0),
            reverse=True
        )
        top_sellers = seller_profiles[:20]
    
        for seller in top_sellers:
            seller_id = seller.get('id') or seller.get('seller_id')
            if not seller_id:
                continue
                
            key = self.foodie_cache.create_seller_profile_key_string(seller_id)
            self.knowledge_cache.set(
                self.knowledge_cache.hash_query(key),
                seller
            )
    
        logger.info("Primed caches with top 20 sellers")

    def _register_delivery_tasks(self):
        """Register food delivery specific tasks with the planning agent"""
        # Helper functions for preconditions and effects
        def has_vehicle(state):
            return state.get('vehicle_available', True)
        
        def gps_enabled(state):
            return state.get('gps_status', 'active') == 'active'
        
        def update_location(state, new_location):
            state['current_location'] = new_location
        
        def cooling_available(state):
            return state.get('cooling_system', False)
        
        # Create effect functions with fixed parameters
        def create_navigate_effect(location):
            return lambda state: update_location(state, location)
        
        # Primitive Tasks
        navigate_to_seller = Task(
            name="navigate_to_seller",
            task_type=TaskType.PRIMITIVE,
            preconditions=[has_vehicle, gps_enabled],
            effects=[lambda state: self.foodie_map.update_location(state, state['seller_location'])],
            duration=300
        )
        
        navigate_to_customer = Task(
            name="navigate_to_customer",
            task_type=TaskType.PRIMITIVE,
            preconditions=[has_vehicle, gps_enabled],
            effects=[lambda state: self.foodie_map.update_location(state, state['customer_location'])],
            duration=300
        )
        
        pickup_order = Task(
            name="pickup_order",
            task_type=TaskType.PRIMITIVE,
            preconditions=[lambda state: self.foodie_map._locations_equal(
                state.get('current_location'), 
                state.get('seller_location')
            )],
            effects=[lambda state: state.update({'order_picked_up': True})],
            duration=60
        )
        
        deliver_order = Task(
            name="deliver_order",
            task_type=TaskType.PRIMITIVE,
            is_probabilistic=True,
            probabilistic_actions=[
                # 10% chance of traffic delay
                (0.1, lambda state: state.update({"duration": state.get("duration", 0) + 600}))
            ],
            preconditions=[lambda state: self.foodie_map._locations_equal(
                state.get("current_location"),
                state.get("customer_location")
            )],
            effects=[lambda state: state.update({'order_delivered': True})],
            duration=120
        )
        
        prepare_cooling = Task(
            name="prepare_cooling",
            task_type=TaskType.PRIMITIVE,
            preconditions=[cooling_available],
            effects=[lambda state: state.update({'cooling_active': True})],
            duration=180  # 3 minutes
        )
        
        # Abstract Delivery Task
        delivery_task = Task(
            name="delivery_process",
            task_type=TaskType.ABSTRACT,
            methods=[
                # Method 1: Standard delivery
                [
                    navigate_to_seller.copy(),
                    pickup_order.copy(),
                    navigate_to_customer.copy(),
                    deliver_order.copy()
                ],
                # Method 2: Temperature controlled delivery
                [
                    prepare_cooling.copy(),
                    navigate_to_seller.copy(),
                    pickup_order.copy(),
                    navigate_to_customer.copy(),
                    deliver_order.copy()
                ]
            ]
        )
        
        # Register all tasks with the planning agent
        self.planning_agent.register_task(navigate_to_seller)
        self.planning_agent.register_task(navigate_to_customer)
        self.planning_agent.register_task(pickup_order)
        self.planning_agent.register_task(deliver_order)
        self.planning_agent.register_task(prepare_cooling)
        self.planning_agent.register_task(delivery_task)

    def _get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Helper to find a document in the KnowledgeAgent's memory by its ID."""
        for doc in self.knowledge_agent.knowledge_agent:
            if doc.get('doc_id') == doc_id:
                try:
                    return json.loads(doc['text'])
                except (json.JSONDecodeError, TypeError):
                    logger.error(f"Could not parse document text for doc_id: {doc_id}")
                    return None
        return None

    # --- Orchestrated Workflows ---

    def create_user(self, user_data: Dict) -> Dict:
        """
        Orchestrates the creation of a new user with enhanced validation, nuanced safety checks,
        and intelligent caching.

        This workflow:
        1.  Validates the input data structure and business rules.
        2.  Performs a multi-layered safety and security assessment.
        3.  Blocks creation only for critical safety violations (e.g., malicious content).
        4.  For lower-risk issues (e.g., low safety scores), it flags the user profile for
            review but allows creation to proceed.
        5.  Stores the final profile in the knowledge base and primes the cache.

        Args:
            user_data: A dictionary containing the initial user information, such as name and email.

        Returns:
            A dictionary representing the complete, structured user profile.

        Raises:
            DataValidationError: If the input data is invalid or fails a critical safety check.
            RuntimeError: For unexpected and unrecoverable internal errors.
        """
        try:
            logger.info(f"Received request to create new user: {user_data.get('name')}")

            # 1. Generate the initial, validated profile structure
            profile_data = self.foodie_users.create_user_profile_data(user_data)
            user_id = profile_data['user_id']

            # 2. Perform a comprehensive safety and security assessment
            safety_context = {
                "operation": "user_creation",
                "name": user_data.get('name'),
                "email_domain": user_data.get('contact_email', '').split('@')[-1] if user_data.get('contact_email') else "unknown",
                "ip_address": user_data.get('ip_address', 'unknown'),
                "user_agent": user_data.get('user_agent', 'unknown')
            }
            
            safety_report = self.safety_agent.perform_task(
                safety_context,
                context={"operation": "user_creation"}
            )

            # 3. Handle the safety assessment outcome with nuance
            is_safe = safety_report.get("is_safe", False)
            recommendation = safety_report.get("overall_recommendation", "proceed_with_caution")

            if recommendation in ["block", "block_due_to_error"]:
                # --- CRITICAL FAILURE: Block user creation ---
                logger.error(f"Critical safety violation during user creation for '{user_data.get('name')}'. Recommendation: {recommendation}")
                
                # Extract a clear reason for the failure
                reports = safety_report.get('reports', {})
                reason = reports.get('safety_guard', {}).get('details', 'Blocked by a security policy.')
                raise DataValidationError(
                    message=f"User data failed critical safety validation: {reason}",
                    field="security_validation"
                )

            elif not is_safe:
                # --- WARNING: Flag for review but proceed ---
                logger.warning(f"User creation for '{user_data.get('name')}' flagged for review. Reason: {recommendation}")
                
                # Add a 'safety_flags' field to the user's metadata for auditing
                profile_data['metadata']['safety_flags'] = {
                    'status': 'review_required',
                    'recommendation': recommendation,
                    'final_score': safety_report.get('final_safety_score'),
                    'timestamp': datetime.utcnow().isoformat()
                }

            # 4. Store the final profile in the knowledge base and cache
            doc_id_string = self.foodie_cache.create_user_profile_key_string(user_id)
            
            # Use the (potentially modified) profile_data for storage
            self.knowledge_agent.add_document(
                text=json.dumps(profile_data),
                doc_id=doc_id_string,
                metadata={"type": "user_profile"}
            )
    
            cache_key = self.knowledge_cache.hash_query(doc_id_string)
            self.knowledge_cache.set(cache_key, profile_data)

            logger.info(f"User '{user_id}' created and cached successfully.")
            return profile_data

        except DataValidationError as e:
            # Re-raise validation errors to be handled by the caller
            logger.error(f"User creation failed due to validation error: {str(e)}")
            raise
        except ValueError as e:
            # Catch validation errors from foodie_users
            logger.error(f"User data validation failed: {str(e)}")
            raise DataValidationError(f"Invalid user data: {str(e)}")
        except Exception as e:
            # Catch all other unexpected errors
            logger.exception(f"An unexpected internal error occurred during user creation: {str(e)}")
            raise RuntimeError("User creation failed due to a system error.") from e
    
    def find_user_profile(self, user_id: str) -> Optional[Dict]:
        """Finds user profile with enhanced caching and error handling"""
        try:
            doc_id_string = self.foodie_cache.create_user_profile_key_string(user_id)
            cache_key = self.knowledge_cache.hash_query(doc_id_string)
            
            # Check cache first
            if cached_profile := self.knowledge_cache.get(cache_key):
                logger.debug(f"Cache hit for user: {user_id}")
                return cached_profile
            
            logger.info(f"Cache miss for user: {user_id}. Retrieving from KB.")
            profile = self._get_document_by_id(doc_id_string)
            
            if profile:
                self.knowledge_cache.set(cache_key, profile)
                return profile
            
            logger.warning(f"User profile not found: {user_id}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving user profile {user_id}: {str(e)}")
            raise CacheRetrievalFailure(f"Profile retrieval failed: {str(e)}")
    
    def create_seller(self, seller_data: Dict) -> Dict:
        """
        Orchestrates the creation of a new seller with robust validation, nuanced safety checks,
        and intelligent caching.

        This workflow:
        1.  Determines seller type (Restaurant vs. Indie) and validates the input data.
        2.  Performs a multi-layered safety and security assessment on the seller's data.
        3.  Blocks creation only for critical safety violations (e.g., malicious content, prompt injection).
        4.  For lower-risk issues (e.g., low safety scores), it flags the seller's profile
            for administrative review but allows the creation to proceed.
        5.  Stores the final, potentially flagged, profile in the knowledge base and cache.
        6.  Generates and stores a corresponding "seller card" for frontend display.

        Args:
            seller_data: A dictionary containing the initial seller information.

        Returns:
            A dictionary representing the complete, structured seller profile.

        Raises:
            DataValidationError: If the input data is invalid or fails a critical safety check.
            RuntimeError: For unexpected and unrecoverable internal errors.
        """
        try:
            # 1. Validate and structure the initial profile data
            is_indie = 'company_address' not in seller_data or seller_data['company_address'] == 'Home Kitchen'
            seller_type = "indie_seller_profile" if is_indie else "restaurant_profile"
            logger.info(f"Received request to create new {seller_type.replace('_profile', '')}: {seller_data.get('name')}")
            
            profile_data = self.registrator.create_seller_profile_data(seller_data)
            seller_id = profile_data['seller_id']

            # 2. Perform a comprehensive safety and security assessment
            safety_report = self.safety_agent.perform_task(
                seller_data,
                context={"operation": "seller_registration", "seller_type": seller_type}
            )

            # 3. Handle the safety assessment outcome with nuance
            is_safe = safety_report.get("is_safe", False)
            recommendation = safety_report.get("overall_recommendation", "proceed_with_caution")

            if recommendation in ["block", "block_due_to_error"]:
                # --- CRITICAL FAILURE: Block seller creation ---
                logger.error(f"Critical safety violation for seller '{seller_data.get('name')}'. Halting creation. Reason: {recommendation}")
                reports = safety_report.get('reports', {})
                reason = reports.get('safety_guard', {}).get('details', 'Blocked by a critical security policy.')
                raise DataValidationError(
                    message=f"Seller data failed critical safety validation: {reason}",
                    field="security_validation"
                )

            elif not is_safe:
                # --- WARNING: Flag for review but proceed ---
                logger.warning(f"Seller '{seller_data.get('name')}' flagged for review but creation will proceed. Reason: {recommendation}")
                profile_data['metadata']['safety_flags'] = {
                    'status': 'review_required',
                    'recommendation': recommendation,
                    'final_score': safety_report.get('final_safety_score'),
                    'timestamp': datetime.utcnow().isoformat()
                }

            # 4. Store the final profile in the knowledge base and cache
            doc_id_string = self.foodie_cache.create_seller_profile_key_string(seller_id)
            self.knowledge_agent.add_document(
                text=json.dumps(profile_data),
                doc_id=doc_id_string,
                metadata={"type": seller_type}
            )
    
            cache_key = self.knowledge_cache.hash_query(doc_id_string)
            self.knowledge_cache.set(cache_key, profile_data)

            # 5. Generate and store a public-facing seller card
            card_data = {
                'id': seller_id,
                'name': profile_data.get('name', ''),
                'description': profile_data.get('description', ''),
                'rating': 0,
                'review_count': 0,
                'image_url': profile_data.get('image_url', ''),
                'alt_text': profile_data.get('name', 'Seller')
            }
            card_doc_id = f"{'indie_seller' if is_indie else 'restaurant'}_card_{seller_id}"
            self.knowledge_agent.add_document(
                text=json.dumps(card_data),
                doc_id=card_doc_id,
                metadata={"type": f"{'indie_seller' if is_indie else 'restaurant'}_card"}
            )

            logger.info(f"Seller '{seller_id}' created and cached successfully.")
            return profile_data

        except DataValidationError as e:
            # Re-raise validation errors to be handled by the caller
            logger.error(f"Seller creation failed due to validation error: {str(e)}")
            raise
        except ValueError as e:
            # Catch validation errors from the registrator
            logger.error(f"Seller data validation failed: {str(e)}")
            raise DataValidationError(f"Invalid seller data: {str(e)}")
        except Exception as e:
            # Catch all other unexpected errors
            logger.exception(f"An unexpected internal error occurred during seller creation: {str(e)}")
            raise RuntimeError("Seller creation failed due to a system error.") from e

    # --- Business Order Workflow ---
    def create_business_order(self, order_data: Dict) -> Dict:
        """Orchestrates creation of new business order"""
        try:
            logger.info(f"Creating new business order for: {order_data.get('company_name')}")
            order = self.foodie_orders.create_business_order(order_data)
            return order
        except ValueError as e:
            logger.error(f"Business order validation failed: {str(e)}")
            raise DataValidationError(f"Invalid order data: {str(e)}")
        except Exception as e:
            logger.exception(f"Unexpected error creating business order: {str(e)}")
            raise RuntimeError("Business order creation failed") from e
    
    def process_business_order(self, order: Dict) -> Dict:
        """Processes business order using agents"""
        try:
            logger.info(f"Processing business order: {order['order_id']}")
            processed_order = self.foodie_orders.process_order_with_agents(
                order,
                self.knowledge_agent,
                self.planning_agent
            )
            return processed_order
        except Exception as e:
            logger.exception(f"Error processing business order: {str(e)}")
            raise RuntimeError("Business order processing failed") from e
        
    def find_vendors_for_business(self, employee_count, cuisines, location):
        """Find vendors suitable for business orders"""
        min_rating = 4.0 if employee_count > 20 else 3.5
        min_capacity = max(10, employee_count * 1.2)
        
        # Get all vendors
        all_vendors = self.knowledge_agent.retrieve_documents_by_type("seller_profile")
        
        # Filter and rank
        suitable = []
        for vendor in all_vendors:
            # Check capacity
            if vendor.get('max_capacity', 0) < min_capacity:
                continue
                
            # Check rating
            if vendor.get('ratings', {}).get('average', 0) < min_rating:
                continue
                
            # Check cuisine match
            vendor_cuisines = vendor.get('cuisine_types', [])
            cuisine_match = any(c in vendor_cuisines for c in cuisines) if cuisines else True
                
            # Check location
            location_match = True
            if location:
                vendor_areas = vendor.get('service_areas', [])
                location_match = any(area.lower() in location.lower() for area in vendor_areas)
                
            if cuisine_match and location_match:
                suitable.append({
                    'id': vendor['seller_id'],
                    'name': vendor['name'],
                    'description': vendor.get('description', ''),
                    'image_url': vendor.get('image_url', ''),
                    'rating': vendor.get('ratings', {}).get('average', 0),
                    'review_count': vendor.get('ratings', {}).get('count', 0),
                    'business_discount': vendor.get('business_discount', 0),
                    'type': 'indie' if vendor.get('is_indie') else 'restaurant'
                })
        
        # Sort by discount then rating
        suitable.sort(key=lambda v: (-v['business_discount'], -v['rating']))
        return suitable[:10]  # Return top 10
    
    def finalize_business_order(self, order_id, vendor_id):
        """Finalize business order with selected vendor"""
        # Retrieve order
        order = self._get_business_order(order_id)
        
        # Retrieve vendor
        vendor_key = self.foodie_cache.create_seller_profile_key_string(vendor_id)
        vendor = self._get_document_by_id(vendor_key)
        
        if not order or not vendor:
            raise ValueError("Invalid order or vendor")
        
        # Apply business discount
        discount = vendor.get('business_discount', 0)
        if discount:
            order['discount'] = discount
            # Apply discount logic here
        
        # Update order status
        order['status'] = 'confirmed'
        order['vendor_id'] = vendor_id
        order['vendor_name'] = vendor['name']
        
        # Save updated order
        self._save_business_order(order)
        
        return {
            "message": "Order confirmed with vendor",
            "vendor_name": vendor['name'],
            "discount_applied": discount
        }

    # --- Business Cards ---
    def get_restaurants_html(self) -> str:
        """Generates HTML for restaurant cards using knowledge agent"""
        # Retrieve restaurant profiles from knowledge agent
        restaurants = self.knowledge_agent.retrieve_documents_by_type("restaurant_profile")
        
        # Process data for card generation
        restaurant_data = []
        for restaurant in restaurants:
            restaurant_data.append({
                'id': restaurant.get('id', ''),
                'name': restaurant.get('name', ''),
                'description': restaurant.get('description', ''),
                'rating': restaurant.get('rating', {}).get('average', 0),
                'review_count': restaurant.get('rating', {}).get('count', 0),
                'image_url': restaurant.get('image_url', ''),
                'alt_text': restaurant.get('alt_text', 'Restaurant')
            })
        
        # Generate cards
        restaurant_cards = self.foodie_card.generate_dynamic_cards(
            card_type='restaurant',
            items=restaurant_data,
            link_base_url='restaurant_details.html'
        )
        return self.foodie_card.generate_cards_only(restaurant_cards)
    
    def get_indie_html(self) -> str:
        """Generates HTML for indie seller cards using knowledge agent"""
        # Retrieve indie seller profiles from knowledge agent
        indie_sellers = self.knowledge_agent.retrieve_documents_by_type("indie_seller_profile")
        
        # Process data for card generation
        indie_data = []
        for seller in indie_sellers:
            indie_data.append({
                'id': seller.get('id', ''),
                'name': seller.get('name', ''),
                'description': seller.get('description', ''),
                'rating': seller.get('rating', {}).get('average', 0),
                'review_count': seller.get('rating', {}).get('count', 0),
                'image_url': seller.get('image_url', ''),
                'alt_text': seller.get('alt_text', 'Indie Seller')
            })
        
        # Generate cards
        indie_cards = self.foodie_card.generate_dynamic_cards(
            card_type='indie',
            items=indie_data,
            link_base_url='indie_seller.html'
        )
        return self.foodie_card.generate_cards_only(indie_cards)
    
    def get_restaurants_data(self):
        """Returns structured restaurant data for API"""
        restaurants = self.knowledge_agent.retrieve_documents_by_type("restaurant_profile")
        restaurant_data = []
        for restaurant in restaurants:
            restaurant_data.append({
                'id': restaurant.get('id', ''),
                'name': restaurant.get('name', ''),
                'description': restaurant.get('description', ''),
                'rating': restaurant.get('rating', {}).get('average', 0),
                'review_count': restaurant.get('rating', {}).get('count', 0),
                'image_url': restaurant.get('image_url', ''),
                'alt_text': restaurant.get('alt_text', 'Restaurant')
            })
        return [r for r in restaurants if not r.get('is_indie')]
    
    def get_indie_data(self):
        """Returns structured indie seller data for API"""
        indie_sellers = self.knowledge_agent.retrieve_documents_by_type("indie_seller_profile")
        indie_data = []
        for seller in indie_sellers:
            indie_data.append({
                'id': seller.get('id', ''),
                'name': seller.get('name', ''),
                'description': seller.get('description', ''),
                'rating': seller.get('rating', {}).get('average', 0),
                'review_count': seller.get('rating', {}).get('count', 0),
                'image_url': seller.get('image_url', ''),
                'alt_text': seller.get('alt_text', 'Indie Seller')
            })
        return [s for s in indie_sellers if s.get('is_indie')]

# ========== Order Pipeline ==========
    def order_placement(self, user_id: str, query: str, filters: Optional[Dict] = None) -> Dict:
        """
        Handles user interaction flow from search to order placement
        Orchestrates KnowledgeAgent for data retrieval and validation
        
        Args:
            user_id: Authenticated user ID
            query: Search query (dish/restaurant name)
            filters: Dietary/cuisine filters
            
        Returns:
            Structured order placement data
        """
        try:
            # Retrieve user profile with caching
            user_profile = self.find_user_profile(user_id)
            self.local_memory.set_user_context(user_id, user_profile.get('preferences', {}))
            if not user_profile:
                raise ValueError(f"User profile not found: {user_id}")
        
            # Generate cache-aware search key
            search_key = self.foodie_cache.create_search_query_key_string(query, filters)
            
            # Retrieve matching sellers/menus
            seller_results = self.knowledge_agent.retrieve(search_key, k=5)
            valid_sellers = self._validate_seller_availability(seller_results)
        
            # Get primary user address
            primary_address = next((addr for addr in user_profile["addresses"] if addr["is_primary"]), None)
            if not primary_address:
                raise MissingAddressError("user", user_id)
        
            # Structure response
            order_id = str(uuid.uuid4())
            response = {
                "order_id": order_id,
                "user_id": user_id,
                "query": query,
                "filters": filters,
                "sellers": valid_sellers,
                "delivery_address": primary_address,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Cache results in shared memory for agents
            self.shared_memory.put(
                key=f"search:{user_id}:{hashlib.sha256(query.encode()).hexdigest()}",
                value=valid_sellers,
                ttl=timedelta(minutes=30)
            )

            return response

        except DataValidationError as e:
            logger.error(f"Validation failed for user {user_id}: {str(e)}")
            return {"status": "error", "code": "VALIDATION_FAILURE", "details": str(e)}
        except CacheRetrievalFailure:
            logger.warning("Cache failure - falling back to direct query")
            return self._fallback_seller_search(query, filters)
    
    def order_processor(self, order_data: Dict) -> Dict:
        """
        Processes confirmed orders and initiates fulfillment.
        Leverages PlanningAgent for logistics optimization.
        
        Args:
            order_data: Structured order from placement phase
            
        Returns:
            Order confirmation with fulfillment details
        """
        try:
            order_id = order_data.get("order_id", str(uuid.uuid4()))
            logger.info(f"Processing order {order_id}...")

            # 1. Perform a comprehensive safety and security assessment
            action_validation = self.safety_agent.validate_action(order_data,
                action_context={"operation": "order_processing", "user_id": order_data.get("user_id")
                })

            # Create the initial order structure
            structured_order = {
                "order_id": order_id,
                "user_id": order_data.get("user_id"),
                "seller_id": order_data.get("seller_id"),
                "items": order_data.get("items", []),
                "delivery_address": order_data.get("delivery_address"),
                "status": "PROCESSING",
                "created_at": datetime.utcnow().isoformat(),
                "metadata": {}
            }

            # 2. Handle the safety assessment outcome with nuance
            recommendation = action_validation.get("overall_recommendation", "proceed_with_caution")
            if recommendation in ["block", "block_due_to_error"]:
                logger.error(f"Critical safety violation for order {order_id}. Halting processing. Reason: {recommendation}")
                reason = action_validation.get('details', ['Blocked by a critical security policy.'])[0]
                raise OrderProcessingError(f"Order failed critical security validation: {reason}")

            elif not action_validation.get("approved", True):
                logger.warning(f"Order {order_id} flagged for review but processing will continue. Reason: {recommendation}")
                structured_order['metadata']['safety_flags'] = {
                    'status': 'review_required',
                    'recommendation': recommendation,
                    'details': action_validation.get('details', []),
                    'timestamp': datetime.utcnow().isoformat()
                }

            # 3. Validate seller and item availability
            seller_id = order_data["seller_id"]
            seller = self._get_validated_seller(seller_id, order_data["items"])
            structured_order["total"] = self.foodie_orders._calculate_order_total(seller, order_data["items"])
    
            # Track order in local memory
            self.local_memory.update_order_state(
                order_data["order_id"], 
                "PROCESSING", 
                order_data
            )
            
            # Share order context with agents
            self.shared_memory.put(
                key=f"order:{order_data['order_id']}",
                value=order_data,
                tags=["active_order"]
            )
    
            # 4. Set up the planning world state for the PlanningAgent
            self.planning_agent.world_state = {
                "vehicle_available": True,
                "gps_status": "active",
                "cooling_system": any(item.get("requires_cold", False) for item in structured_order["items"]),
                "current_location": seller["location"],
                "seller_location": seller["location"],
                "customer_location": structured_order["delivery_address"],
                "order_picked_up": False,
                "order_delivered": False,
            }

            # Store order in knowledge base
            order_key = self.foodie_cache.create_composite_key("order", order_id, "metadata")
            self.knowledge_agent.add_document(
                text=json.dumps(structured_order),
                doc_id=order_key,
                metadata={"type": "order", "status": "processing"}
            )
    
            # Initiate planning
            delivery_task = Task(
                name="delivery_process",
                task_type=TaskType.ABSTRACT,
                goal_state={
                    "seller_location": seller["location"],
                    "delivery_address": order_data["delivery_address"],
                    "preparation_time": seller["avg_prep_time"],
                    "temperature_control": any(item.get("requires_cold") for item in order_data["items"])
                }
            )
    
            # Get user location safely
            user_profile = self.find_user_profile(order_data["user_id"])
            if not user_profile:
                raise MissingAddressError("user", order_data["user_id"])
                
            primary_address = next((addr for addr in user_profile.get("addresses", []) if addr.get("is_primary")), None)
            if not primary_address:
                raise MissingAddressError("user", order_data["user_id"])
    
            # Validate all locations
            location_checks = [
                (seller.get("location"), "Seller location"),
                (primary_address, "User primary address"),
                (order_data.get("delivery_address"), "Delivery address")
            ]
            
            for loc, name in location_checks:
                if not loc or not isinstance(loc, dict):
                    raise InvalidOrderStructureError(f"{name} missing")
                if 'latitude' not in loc or 'longitude' not in loc:
                    raise GeolocationError(f"{name} incomplete")
    
            # Generate delivery plan
            delivery_plan_steps = []
            schedule = {}
            delivery_plan = self.planning_agent.generate_plan(delivery_task)
            estimated_completion = None
            
            # Handle different PlanningAgent return types
            if isinstance(delivery_plan, Task):
                # PlanningAgent returned a single Task object
                delivery_plan_steps = self._extract_steps_from_task(delivery_plan)
            elif isinstance(delivery_plan, list):
                # PlanningAgent returned a list of steps
                delivery_plan_steps = delivery_plan
            elif isinstance(delivery_plan, dict):
                # PlanningAgent returned a structured response
                delivery_plan_steps = delivery_plan.get('plan_steps', [])
                schedule = delivery_plan.get('schedule', {})
                estimated_completion = delivery_plan.get('estimated_completion') or estimated_completion
            else:
                # Fallback - try to process as iterable
                try:
                    delivery_plan_steps = list(delivery_plan)
                except TypeError:
                    logger.warning("PlanningAgent returned unsupported plan format")
                    delivery_plan_steps = []
    
            # Extract task details safely
            plan_steps = []
            for task in delivery_plan_steps:
                if not isinstance(task, Task):
                    logger.warning(f"Skipping invalid task in plan: {type(task)}")
                    continue
                    
                step_info = {
                    "name": task.name,
                    "type": task.task_type.name,
                    "duration": getattr(task, 'duration', 0)
                }
                
                # Handle probabilistic actions
                if getattr(task, 'is_probabilistic', False):
                    step_info['probabilistic'] = True
                    step_info['actions'] = [
                        {"probability": p, "effect": e.__name__}
                        for p, e in getattr(task, 'probabilistic_actions', [])
                    ]
                
                plan_steps.append(step_info)
    
            # Build result
            result = {
                **structured_order,
                "delivery_plan": plan_steps,
                "estimated_delivery": estimated_completion or datetime.utcnow() + timedelta(minutes=30)
            }
    
            # Update shared memory
            self.shared_memory.put(
                key=f"delivery_plan:{order_id}",
                value={
                    "plan": delivery_plan_steps,
                    "schedule": schedule,
                    "metadata": structured_order
                },
                ttl=timedelta(hours=24)
            )
            
            return result
        
        except (ItemOutOfStockError, SellerUnavailableError) as e:
            logger.error(f"Inventory issue: {str(e)}")
            raise OrderProcessingError(f"Item unavailable: {getattr(e, 'item_id', 'unknown')}")
        except (GeolocationError, MissingAddressError, InvalidOrderStructureError) as e:
            logger.error(f"Location error: {str(e)}")
            raise OrderProcessingError("Invalid location data") from e
        except Exception as e:
            logger.exception("Critical order processing failure")
            raise OrderProcessingError("System error processing order") from e
    
    def order_monitoring(self, order_id: str) -> Dict:
        """
        ExecutionAgent takes the delivery plan and manages the real-world execution (communicating with couriers, tracking progress)
        AdaptiveAgent monitors real-time conditions and suggests adjustments to the PlanningAgent
        EvaluationAgent collects data on the execution (time, success, issues)

        Args:
            order_id: Target order ID
            
        Returns:
            Current order status with tracking details

        """
        status_changed = False

        # Websocket integration for live updates
        ws_key = f"order_ws:{order_id}"
        if ws_key not in self.shared_memory:
            self.shared_memory.put(
                ws_key, 
                WebSocketManager.create_channel(order_id),
                ttl=timedelta(hours=2)
            )

        # Retrieve order metadata
        order_key = self.foodie_cache.create_composite_key("order", order_id, "metadata")
        order = self._get_document_by_id(order_key)
        order_state = self.local_memory.session_data["active_order"]

        # Get real-time updates from shared memory
        delivery_update = self.shared_memory.get(
            key=f"delivery_status:{order_id}",
            default={}
        )
        
        # Get active delivery task
        delivery_task = self.execution_agent.get_active_task(f"delivery_{order_id}")
        
        # Build real-time status
        status = {
            "order_id": order_id,
            "current_status": order["status"],
            "last_updated": datetime.utcnow().isoformat(),
            "milestones": []
        }
        
        if delivery_task:
            status.update({
                "courier_location": delivery_task.state["current_position"],
                "estimated_remaining": delivery_task.estimated_completion,
                "milestones": delivery_task.progress_milestones
            })
        
        # Handle status transitions
        if not delivery_task and order["status"] == "PROCESSING":
            self._update_order_status(order_id, "READY_FOR_PICKUP")
            status_changed = True

        # Update shared memory with new status
        if status_changed:
            self.shared_memory.publish(
                channel=f"order_updates:{order_id}",
                message=status
            )

        if ws_key in self.shared_memory:
            self.shared_memory.get(ws_key).broadcast(status)

        # Adaptive monitoring
        if delivery_task:
            # Check if adjustment is needed
            adjustment_needed = self.adaptive_agent.monitor_delivery(
                order_id=order_id,
                current_location=delivery_task.state["current_position"],
                planned_route=delivery_task.state.get("route"),
                deadline=order.get('estimated_delivery')
            )
            
            if adjustment_needed:
                printer.status("ADAPTIVE", f"Rerouting order {order_id}", "warning")
                self._trigger_adaptive_reroute(order_id, delivery_task)

        # When order is delivered
        if status_changed and order["status"] == "DELIVERED":
            self._record_delivery_metrics(order_id)

        return status
    
    def _extract_steps_from_task(self, task: Task) -> list:
        """Recursively extract steps from a Task hierarchy"""
        steps = []
        current = task
        
        while current:
            steps.append(current)
            # Try different ways to get subtasks
            if hasattr(current, 'subtasks') and current.subtasks:
                current = current.subtasks[0]
            elif hasattr(current, 'children') and current.children:
                current = current.children[0]
            elif hasattr(current, 'get_subtasks') and callable(current.get_subtasks):
                subtasks = current.get_subtasks()
                current = subtasks[0] if subtasks else None
            else:
                current = None
        
        return steps
    
    def _parse_plan_step(self, step) -> Optional[dict]:
        """Safely parse different step representations"""
        try:
            # Handle Task objects
            if isinstance(step, Task):
                return {
                    "name": getattr(step, 'name', 'unknown'),
                    "type": getattr(step.task_type, 'name', 'UNKNOWN'),
                    "duration": getattr(step, 'duration', 0)
                }
            
            # Handle dictionaries
            elif isinstance(step, dict):
                return {
                    "name": step.get('name', 'unknown'),
                    "type": step.get('type', 'UNKNOWN'),
                    "duration": step.get('duration', 0)
                }
            
            # Handle tuples
            elif isinstance(step, tuple):
                return {
                    "name": step[0] if len(step) > 0 else 'unknown',
                    "type": step[1] if len(step) > 1 else 'UNKNOWN',
                    "duration": step[2] if len(step) > 2 else 0
                }
            
            # Handle strings
            elif isinstance(step, str):
                return {"name": step, "type": "PRIMITIVE", "duration": 0}
            
            logger.warning(f"Unsupported step format: {type(step)}")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing plan step: {str(e)}")
            return None

    def _fallback_seller_search(self, query: str, filters: Optional[Dict] = None) -> Dict:
        """
        Executes a fallback strategy to find sellers matching a user's search query
        when cache retrieval fails. This bypasses cache and uses real-time data retrieval,
        applying basic filter logic and availability validation.
    
        Args:
            query: The search string provided by the user (e.g., dish name or seller name).
            filters: Optional filters such as dietary restrictions, cuisine type, rating.
    
        Returns:
            Dict containing sellers, query context, and fallback status.
        """
        logger.warning(f"Fallback triggered: Direct seller search for query '{query}' with filters {filters}")
        
        # Directly retrieve raw seller documents from the knowledge agent
        all_sellers = self.knowledge_agent.retrieve_documents_by_type("seller_profile")
    
        # Step 1: Basic text match on seller name and menu items
        matching_sellers = []
        query_lower = query.lower()
    
        for seller in all_sellers:
            name_match = query_lower in seller.get("name", "").lower()
    
            menu_match = any(
                query_lower in item.get("name", "").lower()
                for item in seller.get("menu", [])
            )
    
            if name_match or menu_match:
                matching_sellers.append(seller)
    
        logger.info(f"{len(matching_sellers)} raw matches found in fallback search.")
    
        # Step 2: Apply filters (e.g., cuisine type, dietary, rating)
        if filters:
            def passes_filters(seller):
                # Cuisine filter
                cuisine_ok = filters.get("cuisine") is None or \
                             seller.get("cuisine_type") == filters["cuisine"]
                
                # Rating filter
                min_rating = filters.get("rating", 0)
                rating_ok = seller.get("ratings", {}).get("average", 0) >= min_rating
    
                # Dietary filter
                dietary_filters = filters.get("dietary", [])
                if dietary_filters:
                    menu_items = seller.get("menu", [])
                    has_diet_friendly_item = any(
                        any(diet in item.get("dietary_info", []) for diet in dietary_filters)
                        for item in menu_items
                    )
                else:
                    has_diet_friendly_item = True
    
                return cuisine_ok and rating_ok and has_diet_friendly_item
            
            matching_sellers = list(filter(passes_filters, matching_sellers))
            logger.info(f"{len(matching_sellers)} sellers matched after applying filters.")
    
        # Step 3: Validate real-time availability
        available_sellers = self._validate_seller_availability(matching_sellers)
    
        logger.info(f"{len(available_sellers)} sellers available after fallback validation.")
    
        return {
            "status": "fallback_success",
            "query": query,
            "filters": filters,
            "sellers": available_sellers,
            "timestamp": datetime.utcnow().isoformat(),
            "fallback": True
        }

# ========== Supporting Infrastructure ==========
    # ========== Seller Validation ==========
    def _validate_seller_availability(self, seller_results: List) -> List:
        """Robust validation of seller availability considering business hours"""
        valid = []
        current_time = datetime.utcnow()
        current_weekday = current_time.weekday()
        current_time_obj = current_time.time()
        #self.knowledge_cache.set(self.cache_key, seller, ttl=300)
        
        logger.info(f"Validating seller availability. Current time (UTC): {current_time_obj}, weekday: {current_weekday}")
        
        for seller in seller_results:
            if not seller.get("is_active", False):
                logger.debug(f"Skipping inactive seller: {seller.get('seller_id')}")
                continue
                
            hours = seller.get("business_hours", [])
            open_now = False
            
            for h in hours:
                if h["day"] != current_weekday:
                    continue
                    
                try:
                    open_time = datetime.strptime(h["open"], "%H:%M").time()
                    close_time = datetime.strptime(h["close"], "%H:%M").time()
                    
                    # Handle overnight hours (e.g., 22:00-02:00)
                    if close_time < open_time:
                        # Current time is either after open_time OR before close_time
                        if current_time_obj >= open_time or current_time_obj <= close_time:
                            open_now = True
                            break
                    else:
                        # Normal daytime hours
                        if open_time <= current_time_obj <= close_time:
                            open_now = True
                            break
                except Exception as e:
                    logger.error(f"Error parsing business hours for seller {seller.get('seller_id')}: {str(e)}")
                    continue
            
            if open_now:
                logger.info(f"Seller available: {seller.get('seller_id')}")
                valid.append({
                    "seller_id": seller["seller_id"],
                    "name": seller["name"],
                    "rating": seller["ratings"]["average"],
                    "delivery_options": seller["service_areas"]
                })
            else:
                logger.info(f"Seller unavailable: {seller.get('seller_id')}. Open hours not matching")
        
        return valid

    def _get_validated_seller(self, seller_id: str, items: List[Dict]) -> Dict:
        """Comprehensive seller validation with detailed error handling"""
        logger.info(f"Validating seller: {seller_id}")
        # Caching layer
        cache_key = f"seller_validation:{seller_id}"
        if cached := self.knowledge_cache.get(cache_key):
            return cached
        
        # Retrieve seller profile
        seller_key = self.foodie_cache.create_seller_profile_key_string(seller_id)
        seller = self._get_document_by_id(seller_key)
        
        if not seller:
            logger.error(f"Seller not found: {seller_id}")
            raise ValueError(f"Seller not found: {seller_id}")
        if not seller.get("is_active", False):
            logger.warning(f"Seller inactive: {seller_id}")
            raise SellerUnavailableError(seller_id)
        
        # Validate business hours
        current_time = datetime.utcnow()
        current_weekday = current_time.weekday()
        current_time_obj = current_time.time()
        open_now = False
        hours_info = []
        
        for hours in seller.get("business_hours", []):
            if hours["day"] != current_weekday:
                continue
                
            try:
                open_time = datetime.strptime(hours["open"], "%H:%M").time()
                close_time = datetime.strptime(hours["close"], "%H:%M").time()
                hours_info.append(f"{open_time.strftime('%H:%M')}-{close_time.strftime('%H:%M')}")
                
                # Handle overnight hours (e.g., 22:00-02:00)
                if close_time < open_time:
                    if current_time_obj >= open_time or current_time_obj <= close_time:
                        open_now = True
                        break
                else:
                    if open_time <= current_time_obj <= close_time:
                        open_now = True
                        break
            except Exception as e:
                logger.error(f"Error parsing business hours for seller {seller_id}: {str(e)}")
                continue
        
        if not open_now:
            hours_str = ", ".join(hours_info) if hours_info else "No hours defined"
            logger.warning(f"Seller closed: {seller_id}. Current time: {current_time_obj}. Hours: {hours_str}")
            raise ValueError("Seller is currently closed")
        
        # Validate items
        menu_items = {item["item_id"]: item for item in seller.get("menu", [])}
        for item in items:
            item_id = item["item_id"]
            quantity = item.get("quantity", 0)
            
            if not menu_items.get(item_id):
                logger.error(f"Item not on menu: {item_id}")
                raise ValueError(f"Item not on menu: {item_id}")
            if not menu_items[item_id].get("is_available", True):
                logger.warning(f"Item out of stock: {item_id}")
                raise ItemOutOfStockError(item_id)
            if quantity <= 0:
                logger.error(f"Invalid quantity for item: {item_id}")
                raise ValueError(f"Invalid quantity for item: {item_id}")
        
        logger.info(f"Seller validation passed: {seller_id}")
        self.knowledge_cache.set(cache_key, seller, ttl=300)
        return seller

    # Order status update handler
    def _update_order_status(self, order_id: str, status: str):
        """Atomic order status update with notification triggers"""
        order_key = self.foodie_cache.create_composite_key("order", order_id, "metadata")
        current = self._get_document_by_id(order_key)
        
        updated = {**current, "status": status, "updated_at": datetime.utcnow().isoformat()}
        self.knowledge_agent.add_document(
            text=json.dumps(updated),
            doc_id=order_key,
            metadata={"type": "order", "status": status.lower()}
        )
        
        # Trigger notifications
        if status == "OUT_FOR_DELIVERY":
            self.execution_agent.dispatch_task(
                task_type="delivery",
                task_data={
                    "order_id": order_id,
                    "pickup_location": current["seller_location"],
                    "dropoff_location": current["delivery_address"]
                }
            )

    def _trigger_adaptive_reroute(self, order_id, delivery_task):
        """Initiate adaptive rerouting process"""
        # Get current location
        current_location = delivery_task.state["current_position"]
        
        # Get original destination from order data
        order_key = self.foodie_cache.create_composite_key("order", order_id, "metadata")
        order_data = self._get_document_by_id(order_key)
        destination = order_data['delivery_address']
        
        # Get new route with live data
        new_route = self.foodie_map.get_live_route(
            origin=current_location,
            destination=destination
        )
        
        # Update through adaptive agent
        self.adaptive_agent.adjust_route(
            order_id=order_id,
            new_route=new_route
        )

    def _record_delivery_metrics(self, order_id: str):
        """Record delivery metrics for evaluation"""
        try:
            # Calculate delivery time
            start_time = self.shared_memory.get(f"order_start:{order_id}")
            if not start_time:
                logger.warning(f"No start time found for order {order_id}")
                return
                
            delivery_time = (datetime.utcnow() - start_time).total_seconds() / 60  # in minutes
            
            # Update delivery times
            metrics = self.shared_memory.get("performance_metrics")
            metrics["delivery_times"].append(delivery_time)
            
            # Record courier efficiency (simplified)
            if delivery_task := self.execution_agent.get_active_task(f"delivery_{order_id}"):
                courier_id = delivery_task.get("courier_id", "unknown")
                metrics["courier_efficiency"][courier_id].append(delivery_time)
            
            # Update seller performance
            order = self._get_document_by_id(
                self.foodie_cache.create_composite_key("order", order_id, "metadata")
            )
            if order:
                seller_id = order.get("seller_id")
                if seller_id:
                    seller_metrics = metrics["seller_performance"][seller_id]
                    seller_metrics["orders"] += 1
                    # Rating will be updated later when feedback is received
            
            self.shared_memory.put("performance_metrics", metrics)
            
            # Trigger periodic evaluation
            if len(metrics["delivery_times"]) % 10 == 0:  # Every 10 orders
                self.evaluation_agent.evaluate_performance()
                
        except Exception as e:
            logger.error(f"Error recording delivery metrics: {str(e)}")

    def record_customer_feedback(self, order_id: str, rating: int, comments: str = ""):
        """Record customer feedback for order evaluation"""
        try:
            # Validate rating
            if not 1 <= rating <= 5:
                raise ValueError("Rating must be between 1 and 5")
                
            # Get order details
            order_key = self.foodie_cache.create_composite_key("order", order_id, "metadata")
            order = self._get_document_by_id(order_key)
            if not order:
                logger.warning(f"Order not found for feedback: {order_id}")
                return
                
            # Update seller performance
            seller_id = order.get("seller_id")
            if seller_id:
                metrics = self.shared_memory.get("performance_metrics")
                seller_metrics = metrics["seller_performance"][seller_id]
                seller_metrics["rating_sum"] += rating
                
            # Update customer satisfaction metrics
            metrics["customer_satisfaction"].append(rating)
            self.shared_memory.put("performance_metrics", metrics)
            
            # Store feedback in knowledge base
            feedback_id = f"feedback_{order_id}"
            self.knowledge_agent.add_document(
                text=json.dumps({
                    "order_id": order_id,
                    "rating": rating,
                    "comments": comments,
                    "timestamp": datetime.utcnow().isoformat()
                }),
                doc_id=feedback_id,
                metadata={"type": "customer_feedback"}
            )
            
            # Trigger immediate evaluation for poor ratings
            if rating < 3:
                self.evaluation_agent.evaluate_performance()
                
        except Exception as e:
            logger.error(f"Error recording customer feedback: {str(e)}")
            raise


    def handle_complex_support_case(self, ticket_id):
        """Resolve complex support cases using reasoning agent"""
        ticket = self.foodie_support.get_ticket(ticket_id)
        
        # Determine reasoning strategy based on ticket type
        if ticket['category'] in ['business_order', 'complex_dispute']:
            reasoning_type = "decompositional+analogical"
        elif 'technical' in ticket['category']:
            reasoning_type = "deductive+cause_effect"
        else:
            reasoning_type = "abduction+induction"
        
        # Apply reasoning
        resolution = self.reasoning_agent.reason(
            problem=ticket['description'],
            reasoning_type=reasoning_type,
            context={
                "user_id": ticket['user_id'],
                "order_id": ticket.get('related_order'),
                "priority": ticket['priority']
            }
        )
        
        # Execute resolution plan
        if resolution and resolution.get('action_plan'):
            self._execute_resolution_plan(ticket_id, resolution['action_plan'])
        
        return resolution

    def optimize_business_delivery(self, order_data):
        """Optimize complex business deliveries using reasoning"""
        # Create multi-hop reasoning query
        reasoning_result = self.reasoning_agent.multi_hop_reasoning(
            query=(
                "Optimize delivery", 
                "for_order", 
                order_data['order_id']
            ),
            max_depth=4
        )
        
        # Apply optimizations
        if reasoning_result.get('optimization_path'):
            for optimization in reasoning_result['optimization_path']:
                self.planning_agent.apply_optimization(
                    order_data['order_id'],
                    optimization
                )
        
        return reasoning_result

    def onboard_indie_seller(self, seller_id):
        """Guide indie sellers using reasoning agent"""
        seller_data = self.registrator.get_seller(seller_id)
        
        # Generate personalized guidance
        guidance = self.reasoning_agent.react_loop(
            problem=f"Onboard new indie seller specializing in {seller_data['cuisine_type']}",
            max_steps=5
        )
        
        # Create onboarding plan
        onboarding_plan = {
            "steps": guidance.get('solution_steps', []),
            "resources": self._match_resources_to_guidance(guidance)
        }
        
        return onboarding_plan

    # These methods will remain a placeholder for now
    # ==================================================
    def adaptation_cycle(self):
        """
        EvaluationAgent processes performance data
        LearningAgent uses this feedback to update its models and strategies,
        improving future PlanningAgent decisions and AdaptiveAgent responses
        """
        return []
    
    def governance_layer(self):
        """
        SafetyAgent and AlignmentAgent continuously monitor data, processes,
        and agent behaviors to ensure compliance, fairness, and security. They can flag issues and intervene
        """
        return []

    # ==================================================

    def get_performance_report(self, period: str = "daily") -> Dict:
        """Get performance evaluation report"""
        return self.evaluation_agent.generate_report(period)

if __name__ == "__main__":
    printer.status("MAIN", "Starting Continuous Foodie Test Suite", "info")
    import threading
    import os
    app = Foodie()

    # Import test scripts
    from foodie.test.foodie_user_test import run_user_tests
    from foodie.test.foodie_seller_test import run_seller_tests

    # Configure API keys from .env
    openai_key = os.getenv("OPENAI_API_KEY2") or os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY2") or os.getenv("GEMINI_API_KEY")
    
    if not openai_key or not gemini_key:
        printer.status("ERROR", "Missing API keys in environment variables", "error")
        exit(1)

    # Run test suites concurrently
    user_thread = threading.Thread(
        target=run_user_tests, 
        args=(app, openai_key),
        name="UserSimulator"
    )
    
    seller_thread = threading.Thread(
        target=run_seller_tests,
        args=(app, gemini_key),
        name="SellerSimulator"
    )
    
    printer.status("TEST", "Starting continuous user simulation tests...", "info")
    user_thread.start()
    
    printer.status("TEST", "Starting continuous seller simulation tests...", "info")
    seller_thread.start()
    
    try:
        # Monitor test progress
        while user_thread.is_alive() or seller_thread.is_alive():
            printer.status(
                "MONITOR", 
                f"Active threads: {threading.active_count() - 1}", 
                "warning"
            )
            time.sleep(5)
    except KeyboardInterrupt:
        printer.status("TEST", "Keyboard interrupt received. Stopping tests...", "warning")
    
    user_thread.join()
    seller_thread.join()
    
    printer.status("TEST", "All test simulations completed", "success")