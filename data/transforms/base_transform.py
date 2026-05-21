from __future__ import annotations
 
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
 
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.data_error import *
from ..utils.data_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]
 
logger = get_logger("Base Transform")
printer = PrettyPrinter()
 
 
class Transform(ABC):
    """Base class for all feature-engineering transforms.
 
    Every concrete transform must implement ``__call__``, which receives a
    single record dict and the active modality name, and returns the
    (potentially modified) record dict.
 
    The ``to_config`` / ``_get_params`` pair enables full pipeline
    serialisation: a pipeline built from these objects can be saved to JSON
    or YAML and reconstructed without loss of configuration.
 
    Design notes
    ------------
    * ``__call__`` must be idempotent on records that do not belong to
      the transform's modality — it should return the record unchanged.
    * Subclasses that need config values read them via ``get_config_section``
      in their own ``__init__``; they do **not** re-call ``load_global_config``
      as a function reference (a common copy-paste error in the original code).
    * All errors raised from ``__call__`` must be ``DataTransformError``
      (or a ``DataError`` subclass); bare ``Exception`` must never escape.
    """
 
    def __init__(self, name: Optional[str] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.base_cfg: Dict[str, Any] = get_config_section("transforms")
        self.name: str = name or self.__class__.__name__
        logger.debug({"event": "transform_init", "name": self.name})
 
    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
 
    @abstractmethod
    def __call__(self, record: Dict[str, Any], modality: str) -> Dict[str, Any]:
        """Apply this transform to a single record.
 
        Parameters
        ----------
        record:
            A mutable dict representing one data sample.  The transform may
            modify it in-place *or* return a new dict — callers must always
            use the returned value.
        modality:
            The active modality key, e.g. ``"vision"``, ``"text"``,
            ``"audio"``.  Transforms that target a specific modality should
            return *record* unchanged when this does not match.
 
        Returns
        -------
        dict
            The (possibly modified) record.
 
        Raises
        ------
        DataTransformError
            On any transformation failure.
        """
 
    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
 
    def to_config(self) -> Dict[str, Any]:
        """Return a fully serialisable config dict for this transform.
 
        The dict is compatible with ``registry.get_transform`` so that a
        pipeline can be rebuilt from ``load_pipeline``.
        """
        return {"type": self.name, "params": self._get_params()}
 
    def _get_params(self) -> Dict[str, Any]:
        """Return constructor parameters as a serialisable dict.
 
        Override in every concrete subclass to expose all parameters that
        affect transform behaviour.  The default returns an empty dict,
        which is correct only for stateless transforms.
        """
        return {}
 
    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------
 
    def __repr__(self) -> str:
        params = self._get_params()
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.name}({param_str})"
 

if __name__ == "__main__":
    print("\n=== Running base_transform ===\n")
    printer.status("TEST", "base_transform initialized", "info")
 
    # Abstract enforcement
    try:
        Transform()  # type: ignore[abstract]
        assert False, "Transform must not be instantiable"
    except TypeError:
        printer.status("PASS", "Transform correctly prevents direct instantiation", "success")
 
    # Minimal concrete subclass
    class _Identity(Transform):
        def __call__(self, record, modality):
            return record
 
    t = _Identity()
    assert t.name == "_Identity"
    printer.status("PASS", "name defaults to class name", "success")
 
    rec = {"text": "hello", "score": 1}
    out = t(rec, "text")
    assert out is rec
    printer.status("PASS", "__call__ returns record unchanged", "success")
 
    cfg = t.to_config()
    assert cfg == {"type": "_Identity", "params": {}}
    printer.status("PASS", "to_config serialises correctly", "success")
 
    assert repr(t) == "_Identity()"
    printer.status("PASS", "__repr__ correct", "success")
 
    # Custom name
    t2 = _Identity(name="my_identity")
    assert t2.name == "my_identity"
    printer.status("PASS", "custom name accepted", "success")
 
    print("\n=== Test ran successfully ===\n")