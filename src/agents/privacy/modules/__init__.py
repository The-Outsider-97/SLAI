"""Privacy intelligence modules for SLAI.

The package contains evidence-driven privacy reasoning components.  Modules do
not instantiate or own the core privacy domain services; orchestration remains
with ``src.agents.privacy_agent.PrivacyAgent``.
"""

from .privacy_flow_analyzer import *
from .privacy_risk_engine import *
from .residual_exposure import *


from .privacy_flow_analyzer import __all__ as _privacy_flow_analyzer_exports
from .privacy_risk_engine import __all__ as _privacy_risk_engine_exports
from .residual_exposure import __all__ as _residual_exposure_exports


__all__ = [
    *_privacy_flow_analyzer_exports,
    *_privacy_risk_engine_exports,
    *_residual_exposure_exports,
] # type: ignore
