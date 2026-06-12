"""DocMaster package for SLAIHub.

Primary entrypoint:
    from documaster.documaster import DocumasterWindow
"""

from .documaster import DocumasterWindow, create_documaster_flask_app, launch_documaster

__all__ = ["DocumasterWindow", "create_documaster_flask_app", "launch_documaster"]
