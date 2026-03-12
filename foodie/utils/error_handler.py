
from datetime import datetime

class FoodieError(Exception):
    """Base exception for all custom Foodie errors."""
    def __init__(self, message="Food delivery operation failed"):
        super().__init__(message)
        self.error_code = "FOODIE_GENERIC_ERROR"
        self.timestamp = datetime.utcnow().isoformat()

class DataValidationError(FoodieError):
    """Raised when data fails schema or format validation."""
    def __init__(self, message="Invalid data provided", field=None):
        super().__init__(message)
        self.error_code = "VALIDATION_FAILURE"
        self.field = field  # Optional field that failed validation

class CacheRetrievalFailure(FoodieError):
    """Raised when cache lookup fails or returns corrupted data."""
    def __init__(self, message="Unable to retrieve from cache", key=None):
        super().__init__(message)
        self.error_code = "CACHE_FAILURE"
        self.cache_key = key  # Problematic cache key

class SellerUnavailableError(FoodieError):
    """Raised when a seller is inactive, offline, or otherwise unavailable."""
    def __init__(self, seller_id: str, reason="inactive"):
        super().__init__(f"Seller {seller_id} is {reason}")
        self.error_code = "SELLER_UNAVAILABLE"
        self.seller_id = seller_id
        self.reason = reason  # inactive, offline, closed, suspended

class ItemOutOfStockError(FoodieError):
    """Raised when an item is requested but not available."""
    def __init__(self, item_id: str, alternative=None):
        super().__init__(f"Item out of stock or invalid: {item_id}")
        self.error_code = "ITEM_UNAVAILABLE"
        self.item_id = item_id
        self.alternative = alternative  # Suggested replacement item

class PaymentProcessingError(FoodieError):
    """Raised during payment transaction failures."""
    def __init__(self, message="Payment processing failed", gateway_code=None):
        super().__init__(message)
        self.error_code = "PAYMENT_FAILURE"
        self.gateway_code = gateway_code  # Payment gateway error code

class DeliveryFailureError(FoodieError):
    """Raised when order delivery cannot be completed."""
    def __init__(self, order_id: str, reason="courier_unavailable"):
        super().__init__(f"Delivery failed for order {order_id}")
        self.error_code = "DELIVERY_FAILURE"
        self.order_id = order_id
        self.reason = reason  # courier_unavailable, location_unreachable, etc.

class InventoryConflictError(FoodieError):
    """Raised when real-time inventory doesn't match expectations."""
    def __init__(self, item_id: str, expected: int, actual: int):
        super().__init__(
            f"Inventory conflict for {item_id}. Expected: {expected}, Actual: {actual}"
        )
        self.error_code = "INVENTORY_CONFLICT"
        self.item_id = item_id
        self.expected = expected
        self.actual = actual

class ServiceAreaError(FoodieError):
    """Raised when delivery address is outside service coverage."""
    def __init__(self, postal_code: str, seller_id: str):
        super().__init__(
            f"Address {postal_code} not serviced by seller {seller_id}"
        )
        self.error_code = "SERVICE_AREA_VIOLATION"
        self.postal_code = postal_code
        self.seller_id = seller_id

class ConfigurationError(FoodieError):
    """Raised due to misconfiguration in application settings."""
    def __init__(self, component: str, param: str):
        super().__init__(f"Configuration error in {component} ({param})")
        self.error_code = "CONFIG_ERROR"
        self.component = component
        self.param = param

class AgentCommunicationError(FoodieError):
    """Raised when inter-agent communication fails."""
    def __init__(self, source: str, target: str):
        super().__init__(f"Communication failure between {source} and {target}")
        self.error_code = "AGENT_COMM_FAILURE"
        self.source_agent = source
        self.target_agent = target

class RateLimitExceededError(FoodieError):
    """Raised when API rate limits are violated."""
    def __init__(self, endpoint: str, limit: int):
        super().__init__(f"Rate limit exceeded for {endpoint} (max {limit}/min)")
        self.error_code = "RATE_LIMIT_EXCEEDED"
        self.endpoint = endpoint
        self.limit = limit

class GeolocationError(FoodieError):
    """Raised when geolocation services fail."""
    def __init__(self, operation: str, coordinates: tuple):
        super().__init__(f"Geolocation failed for {operation} at {coordinates}")
        self.error_code = "GEOLOCATION_FAILURE"
        self.operation = operation  # reverse_lookup, distance_calc, etc.
        self.coordinates = coordinates

class OrderProcessingError(FoodieError):
    """Raised when order creation or fulfillment fails due to logical or system issues."""
    def __init__(self, message="Order processing error", order_id=None, step=None):
        super().__init__(message)
        self.error_code = "ORDER_PROCESSING_FAILURE"
        self.order_id = order_id
        self.failed_step = step  # e.g., 'validation', 'planning', 'caching', 'execution'

class MissingAddressError(FoodieError):
    """Raised when a user or seller lacks a valid primary address."""
    def __init__(self, entity_type: str, entity_id: str):
        super().__init__(f"Missing primary address for {entity_type} {entity_id}")
        self.error_code = "ADDRESS_MISSING"
        self.entity_type = entity_type  # user or seller
        self.entity_id = entity_id

class InvalidOrderStructureError(FoodieError):
    """Raised when order data is incomplete or malformed."""
    def __init__(self, field: str):
        super().__init__(f"Missing or invalid field in order: {field}")
        self.error_code = "ORDER_STRUCTURE_INVALID"
        self.field = field

class TaskPlanFailure(FoodieError):
    """Raised when PlanningAgent fails to return a viable delivery plan."""
    def __init__(self, order_id: str, reason: str = "unsolvable"):
        super().__init__(f"Planning failed for order {order_id}: {reason}")
        self.error_code = "PLANNING_FAILURE"
        self.order_id = order_id
        self.reason = reason
class SafetyViolationError(FoodieError):
    """Raised when safety protocols are violated during delivery operations."""
    def __init__(self, violation_type: str, details: str):
        super().__init__(f"Safety violation: {violation_type}")
        self.error_code = "SAFETY_VIOLATION"
        self.violation_type = violation_type  # e.g., 'weather', 'vehicle', 'area'
        self.details = details  # Specific details of the violation

class SecurityBreachError(FoodieError):
    """Raised when a security breach or unauthorized access is detected."""
    def __init__(self, resource: str, access_type: str):
        super().__init__(f"Security breach detected on {resource}")
        self.error_code = "SECURITY_BREACH"
        self.resource = resource  # e.g., 'database', 'api', 'user_data'
        self.access_type = access_type  # e.g., 'unauthorized_access', 'data_leak'

class AuthenticationFailure(FoodieError):
    """Raised when user or system authentication fails."""
    def __init__(self, entity: str, auth_method: str):
        super().__init__(f"Authentication failed for {entity}")
        self.error_code = "AUTH_FAILURE"
        self.entity = entity  # e.g., 'user', 'system', 'api'
        self.auth_method = auth_method  # e.g., 'password', 'token', 'biometric'

class PermissionDeniedError(FoodieError):
    """Raised when an operation is attempted without sufficient permissions."""
    def __init__(self, operation: str, required_role: str):
        super().__init__(f"Permission denied for {operation}")
        self.error_code = "PERMISSION_DENIED"
        self.operation = operation
        self.required_role = required_role

class DataPrivacyError(FoodieError):
    """Raised when data privacy regulations are violated."""
    def __init__(self, data_type: str, regulation: str):
        super().__init__(f"Data privacy violation for {data_type}")
        self.error_code = "PRIVACY_VIOLATION"
        self.data_type = data_type  # e.g., 'PII', 'payment_data', 'health_data'
        self.regulation = regulation  # e.g., 'GDPR', 'CCPA', 'local'

class SuspiciousActivityError(FoodieError):
    """Raised when potentially malicious activity is detected."""
    def __init__(self, activity_type: str, severity: str):
        super().__init__(f"Suspicious activity detected: {activity_type}")
        self.error_code = "SUSPICIOUS_ACTIVITY"
        self.activity_type = activity_type  # e.g., 'brute_force', 'data_scraping'
        self.severity = severity  # 'low', 'medium', 'high'

class ComplianceViolationError(FoodieError):
    """Raised when regulatory or compliance requirements are not met."""
    def __init__(self, regulation: str, requirement: str):
        super().__init__(f"Compliance violation: {regulation}")
        self.error_code = "COMPLIANCE_FAILURE"
        self.regulation = regulation  # e.g., 'food_safety', 'labor_laws'
        self.requirement = requirement  # Specific requirement that was violated

class SanitationError(FoodieError):
    """Raised when food sanitation or hygiene standards are violated."""
    def __init__(self, facility: str, violation: str):
        super().__init__(f"Sanitation issue at {facility}")
        self.error_code = "SANITATION_ISSUE"
        self.facility = facility  # e.g., 'kitchen', 'delivery_vehicle'
        self.violation = violation  # Specific sanitation issue

class EmergencyProtocolError(FoodieError):
    """Raised when emergency protocols fail or are improperly executed."""
    def __init__(self, protocol: str, reason: str):
        super().__init__(f"Emergency protocol failure: {protocol}")
        self.error_code = "EMERGENCY_FAILURE"
        self.protocol = protocol  # e.g., 'evacuation', 'medical'
        self.reason = reason  # Reason for failure