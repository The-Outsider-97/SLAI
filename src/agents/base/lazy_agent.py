
import traceback

from typing import Any

from src.agents.base.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger

logger = get_logger("Lazy Agent")

class LazyAgent:
    """
    Wrapper for deferred initialization of an object (often an agent).
    The initialization function (`init_fn`) is called only when an attribute 
    of the wrapped object is first accessed.
    """
    _instance = None # Class variable to store the actual wrapped object instance
    _initialized = False # Flag to track if _init_fn has been called

    def __init__(self, init_fn: callable):
        self.config = load_global_config()
        self.lazy_config = get_config_section('lazy_agent') 

        if not callable(init_fn):
            raise ValueError("LazyAgent's init_fn must be callable")
        
        self._init_fn = init_fn
        self._instance_local = None 
        self._initialized_local = False
        self.logger = get_logger(
            f"{self.__class__.__name__}["
            f"{init_fn.__name__ if hasattr(init_fn, '__name__') else 'anonymous_fn'}]"
        )
        # Additional initialization using config
        self.max_init_attempts = self.lazy_config.get('max_init_attempts', 1)

        logger.info(f"Lazy Agent succesfully initialized")


    def _ensure_initialized(self):
        """Internal helper to initialize the wrapped object if not already done."""
        if not self._initialized_local:
            self.logger.debug(f"Initializing wrapped object...")
            try:
                self._instance_local = self._init_fn()
                if self._instance_local is None:
                    self.logger.error("Initialization function returned None. Wrapped object cannot be used.")
                    # Consider raising an error here if None is an invalid state
                    # raise RuntimeError("LazyAgent's init_fn returned None.")
                else:
                    self.logger.debug(f"Wrapped object of type '{type(self._instance_local).__name__}' initialized successfully.")
            except Exception as e:
                self.logger.error(f"Error during lazy initialization of wrapped object: {e}")
                self.logger.debug(traceback.format_exc())
                # self._instance_local remains None, getattr will likely fail or an error should be raised
                raise # Re-raise the exception to make the failure visible
            finally:
                self._initialized_local = True # Mark as initialized even if it failed, to prevent re-attempts


    def __getattr__(self, name: str) -> Any:
        """
        Initializes the wrapped object if necessary, then delegates attribute access.
        """
        self._ensure_initialized() # Initialize if not already
        
        if self._instance_local is None: # If initialization failed or returned None
            raise AttributeError(
                f"Wrapped object in LazyAgent is None (likely due to initialization failure). "
                f"Cannot access attribute '{name}'."
            )
            
        try:
            return getattr(self._instance_local, name)
        except AttributeError:
            # This provides a more informative error message if the attribute doesn't exist on the wrapped object
            raise AttributeError(
                f"Attribute '{name}' not found on the lazily initialized object "
                f"of type '{type(self._instance_local).__name__}'."
            )

    def __repr__(self) -> str:
        if self._initialized_local and self._instance_local is not None:
            return f"<LazyAgent wrapping [{self._instance_local.__class__.__name__}] (initialized)>"
        else:
            func_name = self._init_fn.__name__ if hasattr(self._init_fn, '__name__') else 'anonymous_function'
            status = "initialization_failed" if self._initialized_local and self._instance_local is None else "uninitialized"
            return f"<LazyAgent for [{func_name}] ({status})>"


# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Lazy Agent ===\n")
    config = load_global_config()
    init_fn=callable

    agent = LazyAgent(init_fn=init_fn)
    print(f"\n{agent}")
    print("\n=== Successfully Ran Lazy Agent ===\n")
