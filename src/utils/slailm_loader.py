import threading
import time

_model_cache = {}
_loading_flag = threading.Event()

def get_slailm(shared_memory, agent_factory=None, key="default"):
    if key not in _model_cache and not _loading_flag.is_set():
        _loading_flag.set()
        thread = threading.Thread(target=_preload_model, args=(shared_memory, agent_factory, key))
        thread.start()

    while key not in _model_cache:
        time.sleep(0.1)
    return _model_cache[key]

def _preload_model(shared_memory, agent_factory=None, key="default"):
    from models.slai_lm_registry import SLAILMManager
    _model_cache[key] = SLAILMManager.get_instance(
        key=key,
        shared_memory=shared_memory,
        agent_factory=agent_factory
    )
