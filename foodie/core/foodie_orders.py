
import json
import copy
import os
from pathlib import Path
import re
import uuid

from datetime import datetime
from typing import Dict, List, Optional

from foodie.utils.config_loader import load_global_config, get_config_section
from src.agents.planning.planning_types import Task, TaskType
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("FoodieOrders")
printer = PrettyPrinter

class FoodieOrders:
    def __init__(self):
        self.config = load_global_config()
        self.required_fields = self.config.get('required_fields', [])
        self.orders_config = get_config_section('foodie_orders')
        self.order_statuses = self.orders_config.get('order_statuses', [])
        self.business_order_fields = self.orders_config.get('business_order_fields', [])
        self.batch_dir = self.orders_config.get('batch_dir')

        self.current_batch_file = None
        self.current_batch_size = 0
        self._initialize_batch_system()

    def _initialize_batch_system(self):
        """Initialize batch file system"""
        # Create data directory if not exists
        Path(self.batch_dir).mkdir(exist_ok=True)
        
        # Find latest batch file
        batch_files = list(Path(self.batch_dir).glob("business_order*.json"))
        if batch_files:
            latest_file = max(batch_files, key=os.path.getctime)
            self.current_batch_file = latest_file
            self.current_batch_size = os.path.getsize(latest_file)
        else:
            self._create_new_batch_file()

    def _create_new_batch_file(self):
        """Create a new batch file with next sequential number"""
        existing = [int(f.stem[-2:]) for f in Path(self.batch_dir).glob("business_order*.json")]
        next_num = max(existing) + 1 if existing else 1
        filename = f"business_order{next_num:02d}.json"
        self.current_batch_file = Path(self.batch_dir) / filename
        self.current_batch_file.touch()
        self.current_batch_size = 0

    def _save_to_batch(self, order_data: Dict):
        """Save order to batch file, creating new file if needed"""
        batch_entry = {
            "Id number": order_data["order_id"],
            "Order time/date": datetime.utcnow().isoformat(),
            "Order specificity": {
                "company_info": order_data["details"]["company_info"],
                "contact_info": order_data["details"]["contact_info"],
                "requirements": order_data["details"]["requirements"]
            }
        }
        
        entry_json = json.dumps(batch_entry) + "\n"
        entry_size = len(entry_json.encode('utf-8'))
        
        # Create new batch if needed
        if self.current_batch_size + entry_size > 1_000_000:  # 1MB
            self._create_new_batch_file()
        
        # Append to current batch file
        with open(self.current_batch_file, "a") as f:
            f.write(entry_json)
        
        self.current_batch_size += entry_size

    def _validate_business_order(self, order_data: Dict) -> None:
        """Validate business order data against business rules"""
        if int(order_data.get('employee_count', 0)) < 5:
            errors.append("Business orders require at least 5 employees")

        errors = []
        
        # Check required fields
        for field in self.business_order_fields:
            if field not in order_data:
                errors.append(f"Missing required field: {field}")
            elif not order_data[field]:
                errors.append(f"Field cannot be empty: {field}")
        
        # Validate email format
        if "email" in order_data:
            email = order_data["email"]
            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                errors.append("Invalid email format")
        
        # Validate employee count (only if present)
        if "employee_count" in order_data and order_data["employee_count"]:
            try:
                count = int(order_data["employee_count"])
                if count < 1 or count > 1000:
                    errors.append("Employee count must be between 1 and 1000")
            except ValueError:
                errors.append("Employee count must be a valid number")
        
        if errors:
            raise ValueError("; ".join(errors))

    def create_business_order(self, order_data: Dict) -> Dict:
        """
        Creates a structured business order with validation and default values
        
        Args:
            order_data: Dictionary containing business order information
            
        Returns:
            Dict: Complete structured order
        """
        structured_order['min_employee'] = 5

        # Validate input data
        self._validate_business_order(order_data)
        
        # Generate unique ID
        order_id = str(uuid.uuid4())
        
        # Build structured order
        structured_order = {
            "order_id": order_id,
            "order_type": "business",
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": {
                "source": "web_form",
                "priority": "normal",
                "assigned_agent": None
            },
            "details": {
                "company_info": {
                    "name": order_data.get("company_name"),
                    "address": order_data.get("company_address"),
                    "registration_number": order_data.get("registration_number")
                },
                "contact_info": {
                    "name": order_data.get("contact_name"),
                    "email": order_data.get("email"),
                    "phone": order_data.get("phone", "")
                },
                "requirements": {
                    "employee_count": int(order_data.get("employee_count", 0)),
                    "frequency": order_data.get("frequency", "once"),
                    "cuisine_preferences": order_data.get("cuisine_preferences", "").split(","),
                    "special_requests": order_data.get("details", ""),
                    "arrival_time": order_data.get("arrival_time")
                }
            },
            "history": []
        }
        self._save_to_batch(structured_order)
        
        logger.info(f"Created new business order: {order_id}")
        return structured_order

    def process_order_with_agents(
        self, 
        order: Dict, 
        knowledge_agent, 
        planning_agent
    ) -> Dict:
        """
        Process business order using KnowledgeAgent and PlanningAgent
        
        Args:
            order: Business order data
            knowledge_agent: KnowledgeAgent instance
            planning_agent: PlanningAgent instance
            
        Returns:
            Dict: Enriched order with agent processing results
        """
        # Store order in knowledge base
        order_key = f"business_order:{order['order_id']}"
        knowledge_agent.add_document(
            text=json.dumps(order),
            doc_id=order_key,
            metadata={"type": "business_order", "status": "pending"}
        )
        
        # Retrieve matching sellers using KnowledgeAgent
        cuisine_prefs = order["details"]["requirements"]["cuisine_preferences"]
        query = " OR ".join(cuisine_prefs)
        seller_results = knowledge_agent.retrieve(query, k=5)
        
        # Filter and rank sellers
        valid_sellers = []
        for score, doc in seller_results:
            seller = doc.get("text")
            if isinstance(seller, str):
                try:
                    seller = json.loads(seller)
                except json.JSONDecodeError:
                    continue
            
            # Check if seller can handle business orders
            if seller.get("business_orders_enabled", False):
                valid_sellers.append({
                    "seller_id": seller.get("seller_id"),
                    "name": seller.get("name"),
                    "rating": seller.get("ratings", {}).get("average", 0),
                    "business_discount": seller.get("business_discount", 0)
                })
        
        # Plan logistics with PlanningAgent
        delivery_task = Task(
            name="delivery_process",
            task_type=TaskType.ABSTRACT,
            goal_state={
                "order_type": "business",
                "address": order["details"]["company_info"]["address"],
                "employee_count": order["details"]["requirements"]["employee_count"],
            }
        )
        
        # Generate logistics plan using the existing method
        logistics_plan = planning_agent.generate_plan(delivery_task)

        # Update order with processing results
        order["processing_results"] = {
            "matched_sellers": valid_sellers,
            "logistics_plan": logistics_plan,
            "knowledge_agent_query": query,
            "retrieved_docs_count": len(seller_results)
        }
        
        # Update order status
        order["status"] = "processing"
        order["updated_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Processed business order {order['order_id']} with agents")
        return order

    def update_order_status(self, order: Dict, new_status: str) -> Dict:
        """
        Update order status with validation
        
        Args:
            order: Existing order data
            new_status: New status to set
            
        Returns:
            Dict: Updated order
        """
        if new_status not in self.order_statuses:
            raise ValueError(f"Invalid status: {new_status}. Valid statuses: {self.order_statuses}")
        
        updated_order = copy.deepcopy(order)
        updated_order["status"] = new_status
        updated_order["updated_at"] = datetime.utcnow().isoformat()
        
        # Add to history
        updated_order["history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": "status_update",
            "from_status": order["status"],
            "to_status": new_status
        })
        
        logger.info(f"Updated order {order['order_id']} status: {order['status']} → {new_status}")
        return updated_order

    def assign_order_agent(self, order: Dict, agent_id: str) -> Dict:
        """
        Assign an agent to handle the business order
        
        Args:
            order: Existing order data
            agent_id: ID of agent to assign
            
        Returns:
            Dict: Updated order
        """
        updated_order = copy.deepcopy(order)
        updated_order["metadata"]["assigned_agent"] = agent_id
        updated_order["updated_at"] = datetime.utcnow().isoformat()
        
        # Add to history
        updated_order["history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": "agent_assignment",
            "agent_id": agent_id
        })
        
        logger.info(f"Assigned agent {agent_id} to order {order['order_id']}")
        return updated_order

    def add_order_note(self, order: Dict, note: str, author: str) -> Dict:
        """
        Add a note to the business order
        
        Args:
            order: Existing order data
            note: Note content
            author: Author of the note
            
        Returns:
            Dict: Updated order
        """
        updated_order = copy.deepcopy(order)
        
        if "notes" not in updated_order["metadata"]:
            updated_order["metadata"]["notes"] = []
            
        updated_order["metadata"]["notes"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "author": author,
            "content": note
        })
        
        updated_order["updated_at"] = datetime.utcnow().isoformat()
        logger.info(f"Added note to order {order['order_id']}")
        return updated_order

    def get_order_summary(self, order: Dict) -> Dict:
        """
        Create a summary view of the business order for reporting
        
        Args:
            order: Full order data
            
        Returns:
            Dict: Summary view
        """
        return {
            "order_id": order["order_id"],
            "company_name": order["details"]["company_info"]["name"],
            "contact_name": order["details"]["contact_info"]["name"],
            "status": order["status"],
            "created_at": order["created_at"],
            "employee_count": order["details"]["requirements"]["employee_count"],
            "assigned_agent": order["metadata"].get("assigned_agent"),
            "matched_sellers_count": len(order.get("processing_results", {}).get("matched_sellers", []))
        }

    def _calculate_order_total(self, seller: Dict, items: List[Dict]) -> float:
        """
        Calculates order total including item prices and applicable fees.
        
        Args:
            seller: Validated seller profile
            items: List of items with 'item_id' and 'quantity'
            
        Returns:
            Total order amount (float)
        """
        # Map menu items for quick lookup
        menu_items = {item["item_id"]: item for item in seller.get("menu", [])}
        subtotal = 0.0

        # Calculate item costs
        for item in items:
            menu_item = menu_items[item["item_id"]]
            subtotal += menu_item["price"] * item["quantity"]

        # Apply service fee (if any)
        service_fee = seller.get("service_fee", 0.0)
        service_fee_type = seller.get("service_fee_type", "fixed")

        if service_fee_type == "percentage":
            total = subtotal * (1 + service_fee / 100)
        else:  # fixed fee
            total = subtotal + service_fee

        # Apply taxes
        tax_rate = seller.get("tax_rate", 0.0)
        total *= (1 + tax_rate / 100)

        return round(total, 2)

if __name__ == "__main__":
    # Test business order functionality
    printer.status("MAIN", "Testing FoodieOrders Methods", "info")
    orders = FoodieOrders()
    
    # Create sample order data matching the HTML form
    order_data = {
        "company_name": "Tech Innovations Inc.",
        "company_address": "123 Innovation Blvd, Oranjestad",
        "contact_name": "Jane Smith",
        "email": "jane.smith@techinnovations.com",
        "phone": "+297 555-1234",
        "employee_count": "50",
        "frequency": "weekly",
        "cuisine_preferences": "Local,Vegetarian,International",
        "details": "Need lunch options for weekly team meetings, vegetarian options required"
    }
    
    # Create structured order
    order = orders.create_business_order(order_data)
    printer.pretty("CREATED ORDER", order, "success")
    
    # Simulate agent processing
    class MockKnowledgeAgent:
        def retrieve(self, query, k):
            return [(0.9, {"text": json.dumps({
                "seller_id": "seller_123",
                "name": "Gourmet Catering",
                "ratings": {"average": 4.8},
                "business_orders_enabled": True,
                "business_discount": 10
            })})]
            
        def add_document(self, text, doc_id, metadata):
            print(f"Stored document {doc_id} in knowledge base")
    
    class MockPlanningAgent:
        def generate_business_delivery_plan(self, address, employee_count):
            return {
                "delivery_schedule": "weekly",
                "delivery_day": "Wednesday",
                "delivery_time": "11:30 AM",
                "packaging": "eco-friendly boxes",
                "special_requirements": "vegetarian labeling"
            }
    
    # Process with agents
    processed_order = orders.process_order_with_agents(
        order, 
        MockKnowledgeAgent(), 
        MockPlanningAgent()
    )
    printer.pretty("PROCESSED ORDER", processed_order, "success")
    
    # Update status
    updated_order = orders.update_order_status(processed_order, "confirmed")
    printer.pretty("CONFIRMED ORDER", updated_order, "info")
    
    # Create summary
    summary = orders.get_order_summary(updated_order)
    printer.pretty("ORDER SUMMARY", summary, "info")