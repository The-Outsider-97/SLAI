
import json
import os
import uuid

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from foodie.utils.config_loader import load_global_config, get_config_section
from foodie.utils.error_handler import (
    FoodieError, DataValidationError, OrderProcessingError, 
    SellerUnavailableError, AgentCommunicationError
)
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Foodie Support")
printer = PrettyPrinter

class SupportTicket:
    """Represents a support ticket with tracking and resolution capabilities"""
    def __init__(self, ticket_data: Dict):
        self.ticket_id = str(uuid.uuid4())
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at
        self.status = "open"
        self.priority = ticket_data.get("priority", "medium")
        self.category = ticket_data["category"]
        self.user_id = ticket_data["user_id"]
        self.subject = ticket_data["subject"]
        self.description = ticket_data["description"]
        self.related_order = ticket_data.get("related_order")
        self.assigned_agent = None
        self.resolution = None
        self.history = [{
            "timestamp": self.created_at,
            "action": "ticket_created",
            "agent": "system"
        }]

class FoodieSupport:
    """Handles support operations for users and sellers including ticket management"""
    def __init__(self):
        self.config = load_global_config()
        self.support_config = get_config_section('foodie_support')
        
        self.categories = self.support_config.get('categories', [])
        self.priorities = self.support_config.get('priorities', [])
        self.max_open_tickets = self.support_config.get('max_open_tickets')
        self.data_dir = self.support_config.get('data_dir')
        
        # Create data directory if needed
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info("FoodieSupport initialized")

    def set_reasoning_agent(self, reasoning_agent):
        if reasoning_agent is None:
            raise ValueError("Reasoning agent cannot be None")
        self.reasoning_agent = reasoning_agent
        logger.info("Reasoning agent linked to FoodieSupport")

    def _validate_ticket_data(self, ticket_data: Dict) -> None:
        """Validate ticket creation data"""
        errors = []
        required_fields = ['category', 'user_id', 'subject', 'description']
        
        for field in required_fields:
            if field not in ticket_data:
                errors.append(f"Missing required field: {field}")
            elif not ticket_data[field]:
                errors.append(f"Field cannot be empty: {field}")
                
        if 'category' in ticket_data and ticket_data['category'] not in self.categories:
            errors.append(
                f"Invalid category: {ticket_data['category']}. "
                f"Allowed: {', '.join(self.categories)}"
            )
            
        if 'priority' in ticket_data and ticket_data['priority'] not in self.priorities:
            errors.append(
                f"Invalid priority: {ticket_data['priority']}. "
                f"Allowed: {', '.join(self.priorities)}"
            )
            
        if errors:
            raise DataValidationError("; ".join(errors))

    def create_support_ticket(self, ticket_data: Dict) -> SupportTicket:
        """
        Creates a new support ticket with validation
        
        Args:
            ticket_data: Dictionary containing ticket information
            
        Returns:
            SupportTicket: Created ticket instance
        """
        self._validate_ticket_data(ticket_data)
        
        # Check open ticket limit
        if self.get_user_open_ticket_count(ticket_data["user_id"]) >= self.max_open_tickets:
            raise FoodieError(
                f"User has too many open tickets (max: {self.max_open_tickets})"
            )
        
        ticket = SupportTicket(ticket_data)
        self._save_ticket(ticket)
        
        logger.info(f"Created support ticket: {ticket.ticket_id}")
        return ticket

    def _save_ticket(self, ticket: SupportTicket) -> None:
        """Save ticket to file storage"""
        file_path = os.path.join(self.data_dir, f"{ticket.ticket_id}.json")
        with open(file_path, 'w') as f:
            json.dump(ticket.__dict__, f, indent=2)

    def _load_ticket(self, ticket_id: str) -> SupportTicket:
        """Load ticket from file storage"""
        file_path = os.path.join(self.data_dir, f"{ticket_id}.json")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                ticket = SupportTicket({
                    "category": data["category"],
                    "user_id": data["user_id"],
                    "subject": data["subject"],
                    "description": data["description"],
                    "priority": data["priority"],
                    "related_order": data.get("related_order")
                })
                # Restore state
                ticket.ticket_id = data["ticket_id"]
                ticket.created_at = data["created_at"]
                ticket.updated_at = data["updated_at"]
                ticket.status = data["status"]
                ticket.assigned_agent = data["assigned_agent"]
                ticket.resolution = data["resolution"]
                ticket.history = data["history"]
                return ticket
        except FileNotFoundError:
            raise FoodieError(f"Ticket not found: {ticket_id}") from None
        except json.JSONDecodeError:
            raise FoodieError(f"Corrupted ticket data: {ticket_id}") from None

    def update_ticket_status(self, ticket_id: str, new_status: str, agent_id: str) -> SupportTicket:
        """
        Update ticket status with validation
        
        Args:
            ticket_id: ID of ticket to update
            new_status: New status to set
            agent_id: ID of agent performing update
            
        Returns:
            SupportTicket: Updated ticket instance
        """
        valid_statuses = ["open", "in_progress", "on_hold", "resolved", "closed"]
        if new_status not in valid_statuses:
            raise DataValidationError(
                f"Invalid status: {new_status}. "
                f"Valid statuses: {', '.join(valid_statuses)}"
            )
            
        ticket = self._load_ticket(ticket_id)
        ticket.status = new_status
        ticket.updated_at = datetime.utcnow().isoformat()
        ticket.history.append({
            "timestamp": ticket.updated_at,
            "action": "status_update",
            "from_status": ticket.status,
            "to_status": new_status,
            "agent": agent_id
        })
        
        self._save_ticket(ticket)
        logger.info(f"Updated ticket {ticket_id} status to {new_status}")
        return ticket

    def assign_ticket_agent(self, ticket_id: str, agent_id: str) -> SupportTicket:
        """
        Assign an agent to a support ticket
        
        Args:
            ticket_id: ID of ticket to assign
            agent_id: ID of agent to assign
            
        Returns:
            SupportTicket: Updated ticket instance
        """
        ticket = self._load_ticket(ticket_id)
        ticket.assigned_agent = agent_id
        ticket.updated_at = datetime.utcnow().isoformat()
        ticket.history.append({
            "timestamp": ticket.updated_at,
            "action": "agent_assigned",
            "agent": agent_id
        })
        
        self._save_ticket(ticket)
        logger.info(f"Assigned agent {agent_id} to ticket {ticket_id}")
        return ticket

    def add_ticket_comment(self, ticket_id: str, comment: str, author_id: str) -> SupportTicket:
        """
        Add a comment to a support ticket
        
        Args:
            ticket_id: ID of ticket to update
            comment: Comment content
            author_id: ID of comment author
            
        Returns:
            SupportTicket: Updated ticket instance
        """
        if not comment.strip():
            raise DataValidationError("Comment cannot be empty")
            
        ticket = self._load_ticket(ticket_id)
        ticket.updated_at = datetime.utcnow().isoformat()
        ticket.history.append({
            "timestamp": ticket.updated_at,
            "action": "comment_added",
            "author": author_id,
            "comment": comment
        })
        
        self._save_ticket(ticket)
        logger.info(f"Added comment to ticket {ticket_id}")
        return ticket

    def resolve_ticket(self, ticket_id: str, resolution: str, agent_id: str) -> SupportTicket:
        """
        Resolve a support ticket with a resolution note
        
        Args:
            ticket_id: ID of ticket to resolve
            resolution: Resolution description
            agent_id: ID of agent resolving ticket
            
        Returns:
            SupportTicket: Resolved ticket instance
        """
        if not resolution.strip():
            raise DataValidationError("Resolution note cannot be empty")
            
        ticket = self._load_ticket(ticket_id)
        ticket.status = "resolved"
        ticket.resolution = resolution
        ticket.updated_at = datetime.utcnow().isoformat()
        ticket.history.append({
            "timestamp": ticket.updated_at,
            "action": "ticket_resolved",
            "agent": agent_id,
            "resolution": resolution
        })
        
        self._save_ticket(ticket)
        logger.info(f"Resolved ticket {ticket_id}")
        return ticket

    def get_user_tickets(self, user_id: str, status_filter: Optional[str] = None) -> List[SupportTicket]:
        """
        Retrieve tickets for a specific user
        
        Args:
            user_id: User ID to retrieve tickets for
            status_filter: Optional status filter
            
        Returns:
            List of SupportTicket objects
        """
        user_tickets = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.data_dir, filename), 'r') as f:
                        data = json.load(f)
                        if data['user_id'] == user_id:
                            if not status_filter or data['status'] == status_filter:
                                ticket = SupportTicket({
                                    "category": data["category"],
                                    "user_id": data["user_id"],
                                    "subject": data["subject"],
                                    "description": data["description"]
                                })
                                ticket.ticket_id = data["ticket_id"]
                                ticket.created_at = data["created_at"]
                                ticket.updated_at = data["updated_at"]
                                ticket.status = data["status"]
                                ticket.assigned_agent = data["assigned_agent"]
                                ticket.resolution = data["resolution"]
                                ticket.history = data["history"]
                                user_tickets.append(ticket)
                except (json.JSONDecodeError, KeyError):
                    logger.error(f"Error loading ticket file: {filename}")
                    continue
        return user_tickets

    def get_user_open_ticket_count(self, user_id: str) -> int:
        """Get count of open tickets for a user"""
        return len(self.get_user_tickets(user_id, status_filter="open"))

    def escalate_ticket(self, ticket_id: str, reason: str, 
                            agent_id: str) -> SupportTicket:
        """
        Escalate a ticket to higher priority
        
        Args:
            ticket_id: ID of ticket to escalate
            reason: Reason for escalation
            agent_id: ID of agent performing escalation
            
        Returns:
            SupportTicket: Updated ticket instance
        """
        ticket = self._load_ticket(ticket_id)
        
        # Determine new priority based on current
        priority_map = {
            "low": "medium",
            "medium": "high",
            "high": "urgent",
            "urgent": "urgent"  # Already at max
        }
        
        new_priority = priority_map.get(ticket.priority, ticket.priority)
        ticket.priority = new_priority
        ticket.updated_at = datetime.utcnow().isoformat()
        ticket.history.append({
            "timestamp": ticket.updated_at,
            "action": "ticket_escalated",
            "agent": agent_id,
            "reason": reason,
            "new_priority": new_priority
        })
        
        self._save_ticket(ticket)
        logger.warning(f"Escalated ticket {ticket_id} to {new_priority} priority")
        return ticket

    def link_ticket_to_order(self, ticket_id: str, order_id: str) -> SupportTicket:
        """
        Link a support ticket to an order
        
        Args:
            ticket_id: ID of ticket to update
            order_id: Order ID to link
            
        Returns:
            SupportTicket: Updated ticket instance
        """
        ticket = self._load_ticket(ticket_id)
        ticket.related_order = order_id
        ticket.updated_at = datetime.utcnow().isoformat()
        ticket.history.append({
            "timestamp": ticket.updated_at,
            "action": "order_linked",
            "order_id": order_id
        })
        
        self._save_ticket(ticket)
        logger.info(f"Linked ticket {ticket_id} to order {order_id}")
        return ticket

    def generate_ticket_report(self, agent_id: Optional[str] = None, status: Optional[str] = None) -> Dict[str, int]:
        """
        Generate ticket statistics report
        
        Args:
            agent_id: Optional agent filter
            status: Optional status filter
            
        Returns:
            Dictionary of ticket counts by category
        """
        report = {category: 0 for category in self.categories}
        report["total"] = 0
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.data_dir, filename), 'r') as f:
                        data = json.load(f)
                        # Apply filters
                        if agent_id and data.get("assigned_agent") != agent_id:
                            continue
                        if status and data.get("status") != status:
                            continue
                            
                        category = data["category"]
                        report[category] = report.get(category, 0) + 1
                        report["total"] += 1
                except (json.JSONDecodeError, KeyError):
                    logger.error(f"Error processing ticket file: {filename}")
                    continue
                    
        return report

    def handle_order_issue(self, ticket_id: str, order_system, knowledge_agent,
                           planning_agent) -> Tuple[SupportTicket, Dict]:
        """
        Resolve order-related issues using specialized agents
        
        Args:
            ticket_id: ID of ticket with order issue
            order_system: Order management system instance
            knowledge_agent: KnowledgeAgent instance
            planning_agent: PlanningAgent instance
            
        Returns:
            Tuple of updated ticket and resolution details
        """
        ticket = self._load_ticket(ticket_id)
        
        if not ticket.related_order:
            raise OrderProcessingError(
                "Ticket not linked to an order",
                ticket_id=ticket_id
            )
            
        try:
            # Retrieve order details
            order = order_system.get_order(ticket.related_order)
            
            # Process with specialized agents
            resolution = order_system.process_order_with_agents(
                order, 
                knowledge_agent, 
                planning_agent
            )
            
            # Update ticket with resolution
            resolution_note = (
                f"Order {ticket.related_order} reprocessed. "
                f"Status: {resolution['status']}"
            )
            ticket = self.resolve_ticket(
                ticket_id, 
                resolution_note, 
                "system_agent"
            )
            
            return ticket, resolution
            
        except SellerUnavailableError as e:
            raise AgentCommunicationError(
                "SupportSystem",
                "OrderSystem",
                f"Seller unavailable: {e.seller_id}"
            ) from e

if __name__ == "__main__":
    printer.status("MAIN", "Testing FoodieSupport Methods", "info")
    support = FoodieSupport()
    
    # Create sample ticket data
    ticket_data = {
        "category": "order_issue",
        "user_id": "user_123",
        "subject": "Missing items in order #ORD-456",
        "description": "My order was missing 2 vegetarian meals",
        "priority": "high",
        "related_order": "ORD-456"
    }
    
    # Create ticket
    ticket = support.create_support_ticket(ticket_data)
    printer.pretty("CREATED TICKET", ticket.__dict__, "success")
    
    # Add comment
    ticket = support.add_ticket_comment(
        ticket.ticket_id,
        "Customer confirmed missing items via photo",
        "agent_007"
    )
    printer.pretty("TICKET WITH COMMENT", ticket.history[-1], "info")
    
    # Assign agent
    ticket = support.assign_ticket_agent(ticket.ticket_id, "agent_007")
    printer.pretty("ASSIGNED TICKET", ticket.assigned_agent, "info")
    
    # Resolve ticket
    ticket = support.resolve_ticket(
        ticket.ticket_id,
        "Issued refund for missing items and sent discount coupon",
        "agent_007"
    )
    printer.pretty("RESOLVED TICKET", {
        "status": ticket.status,
        "resolution": ticket.resolution
    }, "success")
    
    # Generate report
    report = support.generate_ticket_report()
    printer.pretty("TICKET REPORT", report, "info")