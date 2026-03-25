import numpy as np

# Global NumPy 2.x compatibility shim for legacy Gym code paths that still check np.bool8.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from src.agents.collaborative.shared_memory import SharedMemory
