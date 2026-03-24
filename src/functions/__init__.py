"""Shared app functions package."""

from .auth import AuthService, AuthSession, AuthToken, RefreshToken, UserRecord
from .dropdown import (
    AnimationConfig,
    DropdownMenu,
    DropdownOption,
    EASING_PRESETS,
    INTERPOLATION_STRATEGIES,
)
from .functions_memory import CredentialPolicy, PasswordHasher, PortableStore, TTLCache
from .search import (
    BasicAnalyzer,
    LanguageAwareAnalyzer,
    SearchAnalyzer,
    SearchEngine,
    SearchResult,
    StemAnalyzer,
)
from .sidebar import Sidebar, SidebarAnimation, SidebarSection

__all__ = [
    "AnimationConfig",
    "AuthService",
    "AuthSession",
    "AuthToken",
    "BasicAnalyzer",
    "CredentialPolicy",
    "DropdownMenu",
    "DropdownOption",
    "EASING_PRESETS",
    "INTERPOLATION_STRATEGIES",
    "LanguageAwareAnalyzer",
    "PasswordHasher",
    "PortableStore",
    "RefreshToken",
    "SearchAnalyzer",
    "SearchEngine",
    "SearchResult",
    "Sidebar",
    "SidebarAnimation",
    "SidebarSection",
    "StemAnalyzer",
    "TTLCache",
    "UserRecord",
]
