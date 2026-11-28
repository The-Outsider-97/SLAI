"""SLAI native secure communication interfaces; storage is imported explicitly by trusted hosts."""
from .communication_manager import AgentCommunication, DeliveryError

__all__ = ['AgentCommunication', 'DeliveryError']
