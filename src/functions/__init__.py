"""Shared app functions package."""

from .ratelimiter import RateLimiter
from .email import EmailService, SMTPBackend
from .storage import Storage, LocalStorage, S3Storage
from .transport import (
    ChannelState,
    ChannelStatus,
    LoRaAdapter,
    LTEAdapter,
    MeshAdapter,
    SATCOMAdapter,
    SerialAdapter,
    TransportPacket,
    TransportService,
    TransportType,
)
from .auth import AuthService, AuthSession, AuthToken, RefreshToken, UserRecord
from .dropdown import (
    AnimationConfig,
    DropdownMenu,
    DropdownOption,
    EASING_PRESETS,
    INTERPOLATION_STRATEGIES,
)
from .functions_memory import CredentialPolicy, PasswordHasher, PortableStore, TTLCache
from .loader import Loader, LoaderContext
from .loading import create_loading_controller, start_loading, update_loading, complete_loading
from .search import (
    BasicAnalyzer,
    SearchEngine,
    SearchResult,
    StemAnalyzer,
    StopwordAnalyzer,
)
from .sidebar import Sidebar, SidebarAnimation, SidebarSection
from .utils.inverted_index import InvertedIndex, BM25Scorer, SearchAnalyzer

__all__ = [
    # Auth
    "AuthService",
    "AuthSession",
    "AuthToken",
    "RefreshToken",
    "UserRecord",
    # Dropdown
    "AnimationConfig",
    "DropdownMenu",
    "DropdownOption",
    "EASING_PRESETS",
    "INTERPOLATION_STRATEGIES",
    # Memory & Security
    "CredentialPolicy",
    "PasswordHasher",
    "PortableStore",
    "TTLCache",
    # Search
    "BasicAnalyzer",
    "SearchEngine",
    "SearchResult",
    "StemAnalyzer",
    "StopwordAnalyzer",
    "InvertedIndex",
    "BM25Scorer",
    "SearchAnalyzer",
    # Sidebar
    "Sidebar",
    "SidebarAnimation",
    "SidebarSection",
    # Loader
    "Loader",
    "LoaderContext",
    "create_loading_controller",
    "start_loading",
    "update_loading",
    "complete_loading",
    # Limiter
    "RateLimiter",
    # Email
    "EmailService",
    "SMTPBackend",
    # Storage
    "Storage",
    "LocalStorage",
    "S3Storage",
    # Transport
    "TransportType",
    "ChannelStatus",
    "ChannelState",
    "TransportPacket",
    "TransportService",
    "LoRaAdapter",
    "SerialAdapter",
    "MeshAdapter",
    "LTEAdapter",
    "SATCOMAdapter",
]
