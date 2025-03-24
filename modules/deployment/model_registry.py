import os
import json

REGISTRY_FILE = "models/registry.json"

def register_model(model_name, path, metadata=None):
    registry = load_registry()
    registry[model_name] = {
        "path": path,
        "metadata": metadata or {}
    }
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Registered model: {model_name} at {path}")

def load_registry():
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}

def get_model_path(model_name):
    registry = load_registry()
    entry = registry.get(model_name)
    if entry:
        return entry["path"]
    else:
        raise ValueError(f"Model '{model_name}' not found in registry.")
