"""Internal planning utilities.

Keep this package initializer side-effect free. Several utility modules use
the shared types from :mod:`src.agents.planning.planning_types`, while that
module uses ``utils.config_loader``. Eager re-exports here would therefore
create a circular import.

Import utilities from their defining modules instead of this namespace.
"""