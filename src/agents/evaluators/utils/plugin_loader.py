import importlib
import os

from .evaluation_errors import *
from .evaluation_helpers import *

def load_evaluator_plugins(plugin_dir="src/agents/evaluators/custom/") -> dict:
    plugins = {}
    if not os.path.exists(plugin_dir):
        return plugins

    for filename in os.listdir(plugin_dir):
        if filename.endswith(".py") and not filename.startswith("_"):
            mod_name = filename[:-3]
            mod = importlib.import_module(f"src.agents.evaluators.custom.{mod_name}")
            if hasattr(mod, "CustomEvaluator"):
                plugins[mod_name] = mod.CustomEvaluator
    return plugins


__all__ = ["load_evaluator_plugins"]