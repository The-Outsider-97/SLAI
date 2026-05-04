"""
Pydantic (or equivalent) schema validation for knowledge_config.yaml at startup.
Benefit: catches malformed config, missing paths, and type mismatches before runtime threads start.
"""

from pathlib import Path
from typing import Dict, Any, List, Union
import os

from ..utils.knowledge_errors import RuntimeHealthError
from ..utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Config Schema")
printer = PrettyPrinter


class ConfigValidationError(RuntimeHealthError):
    """Raised when configuration validation fails."""
    def __init__(self, section: str, field: str, message: str):
        super().__init__(
            component="config_loader",
            check_name=f"{section}.{field}",
            details=message
        )
        self.section = section
        self.field = field


class ConfigSchema:
    """
    Validates knowledge_config.yaml structure, types, and file paths.
    Raises ConfigValidationError on any failure.
    Optionally creates missing directories (auto_create_dirs, default True).
    """

    def __init__(self):
        self.config = load_global_config()
        self.schema_config = get_config_section('config_schema')
        self.auto_create_dirs = self.schema_config.get('auto_create_dirs', True)
        self.project_root = self._get_project_root()
        self._validate_all()

    def _get_project_root(self) -> Path:
        """
        Find the project root directory (where `src` and `knowledge` folders exist).
        Assumes this file is located at `src/agents/knowledge/validation/config_schema.py`.
        """
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "src").exists() and (parent / "knowledge").exists():
                return parent
        return current.parents[4]

    def _validate_all(self) -> None:
        """Run all validation checks."""
        self._validate_root()
        self._validate_relevance_weights()
        self._validate_temporal_decay()
        self._validate_knowledge_memory()
        self._validate_rule_engine()
        self._validate_governor()
        self._validate_perform_action()
        self._validate_knowledge_cache()
        self._validate_knowledge_monitor()
        self._validate_knowledge_sync()
        self._validate_ontology_manager()
        self._validate_runtime_health()
        self._validate_runtime_metrics()
        logger.info("Configuration validation passed")

    def _assert_type(self, value: Any, expected_type: type, section: str, field: str) -> None:
        if not isinstance(value, expected_type):
            raise ConfigValidationError(
                section, field,
                f"Expected {expected_type.__name__}, got {type(value).__name__}"
            )

    def _assert_in(self, value: Any, allowed: List[Any], section: str, field: str) -> None:
        if value not in allowed:
            raise ConfigValidationError(
                section, field,
                f"Value '{value}' not in allowed {allowed}"
            )

    def _assert_path_exists(self, path_str: str, section: str, field: str, is_directory: bool = False) -> None:
        """
        Validate a filesystem path. If auto_create_dirs is True, create missing directories.
        - is_directory=True: ensure directory exists (create if missing).
        - is_directory=False: ensure parent directory exists (create if missing).
        """
        if not path_str:
            return  # optional path
        full_path = self.project_root / path_str

        if is_directory:
            if not full_path.exists():
                if self.auto_create_dirs:
                    full_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created missing directory: {full_path}")
                else:
                    raise ConfigValidationError(
                        section, field,
                        f"Directory does not exist: {full_path}"
                    )
        else:
            parent = full_path.parent
            if not parent.exists():
                if self.auto_create_dirs:
                    parent.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created missing parent directory: {parent}")
                else:
                    raise ConfigValidationError(
                        section, field,
                        f"Parent directory does not exist for file: {full_path} (expected {parent})"
                    )

    # -------------------------------------------------------------------------
    # Section validators (unchanged except for path calls)
    # -------------------------------------------------------------------------
    def _validate_root(self) -> None:
        if "enabled" not in self.config:
            raise ConfigValidationError("root", "enabled", "Missing required field")
        self._assert_type(self.config["enabled"], bool, "root", "enabled")

    def _validate_relevance_weights(self) -> None:
        weights = self.config.get("relevance_weights", {})
        required = ["semantic", "contextual", "temporal", "structural"]
        for key in required:
            if key not in weights:
                raise ConfigValidationError("relevance_weights", key, "Missing required field")
            self._assert_type(weights[key], (int, float), "relevance_weights", key)
        total = sum(weights[k] for k in required)
        if abs(total - 1.0) > 0.01:
            logger.warning(f"relevance_weights sum is {total}, expected ~1.0")

    def _validate_temporal_decay(self) -> None:
        decay = self.config.get("temporal_decay", {})
        if "half_life_days" not in decay:
            raise ConfigValidationError("temporal_decay", "half_life_days", "Missing required field")
        self._assert_type(decay["half_life_days"], (int, float), "temporal_decay", "half_life_days")

    def _validate_knowledge_memory(self) -> None:
        mem = self.config.get("knowledge_memory", {})
        required = [
            "max_entries", "cache_size", "relevance_mode", "similarity_threshold",
            "decay_factor", "context_window", "enable_ontology_expansion",
            "enable_rule_engine", "embedding_model", "persist_file", "knowledge_dir"
        ]
        for key in required:
            if key not in mem:
                raise ConfigValidationError("knowledge_memory", key, "Missing required field")
        self._assert_type(mem["max_entries"], int, "knowledge_memory", "max_entries")
        self._assert_type(mem["cache_size"], int, "knowledge_memory", "cache_size")
        self._assert_in(mem["relevance_mode"], ["tfidf", "embedding", "hybrid"], "knowledge_memory", "relevance_mode")
        self._assert_type(mem["similarity_threshold"], (int, float), "knowledge_memory", "similarity_threshold")
        self._assert_type(mem["decay_factor"], (int, float), "knowledge_memory", "decay_factor")
        self._assert_type(mem["context_window"], int, "knowledge_memory", "context_window")
        self._assert_type(mem["enable_ontology_expansion"], bool, "knowledge_memory", "enable_ontology_expansion")
        self._assert_type(mem["enable_rule_engine"], bool, "knowledge_memory", "enable_rule_engine")
        self._assert_type(mem["embedding_model"], str, "knowledge_memory", "embedding_model")
        # persist_file is a file: ensure parent directory exists
        self._assert_path_exists(mem["persist_file"], "knowledge_memory", "persist_file", is_directory=False)
        # knowledge_dir is a directory: ensure directory exists
        self._assert_path_exists(mem["knowledge_dir"], "knowledge_memory", "knowledge_dir", is_directory=True)

    def _validate_rule_engine(self) -> None:
        re = self.config.get("rule_engine", {})
        required = [
            "rule_timeout_seconds", "max_concurrent_rules", "verbose_logging",
            "auto_discover", "min_rule_confidence", "slow_rule_threshold",
            "rules_dir", "max_facts_per_rule"
        ]
        for key in required:
            if key not in re:
                raise ConfigValidationError("rule_engine", key, "Missing required field")
        self._assert_type(re["rule_timeout_seconds"], (int, float), "rule_engine", "rule_timeout_seconds")
        self._assert_type(re["max_concurrent_rules"], int, "rule_engine", "max_concurrent_rules")
        self._assert_type(re["min_rule_confidence"], (int, float), "rule_engine", "min_rule_confidence")
        # rules_dir is a directory
        self._assert_path_exists(re["rules_dir"], "rule_engine", "rules_dir", is_directory=True)
        if "rule_sources" in re:
            self._assert_type(re["rule_sources"], list, "rule_engine", "rule_sources")
            for src in re["rule_sources"]:
                self._assert_path_exists(src, "rule_engine", "rule_sources", is_directory=False)

    def _validate_governor(self) -> None:
        gov = self.config.get("governor", {})
        required = ["realtime_monitoring", "audit_interval", "enforcement_mode", "guideline_paths"]
        for key in required:
            if key not in gov:
                raise ConfigValidationError("governor", key, "Missing required field")
        self._assert_type(gov["realtime_monitoring"], bool, "governor", "realtime_monitoring")
        self._assert_type(gov["audit_interval"], int, "governor", "audit_interval")
        self._assert_in(gov["enforcement_mode"], ["log", "alert", "restrict"], "governor", "enforcement_mode")
        self._assert_type(gov["guideline_paths"], list, "governor", "guideline_paths")
        for path in gov["guideline_paths"]:
            self._assert_path_exists(path, "governor", "guideline_paths", is_directory=False)

    def _validate_perform_action(self) -> None:
        pa = self.config.get("perform_action", {})
        required = ["allowed_action_types", "retry_attempts", "timeout", "enable_sandbox"]
        for key in required:
            if key not in pa:
                raise ConfigValidationError("perform_action", key, "Missing required field")
        self._assert_type(pa["allowed_action_types"], list, "perform_action", "allowed_action_types")
        self._assert_type(pa["retry_attempts"], int, "perform_action", "retry_attempts")
        self._assert_type(pa["timeout"], (int, float), "perform_action", "timeout")
        self._assert_type(pa["enable_sandbox"], bool, "perform_action", "enable_sandbox")

    def _validate_knowledge_cache(self) -> None:
        kc = self.config.get("knowledge_cache", {})
        required = ["max_size", "enable_encryption", "hashing_method"]
        for key in required:
            if key not in kc:
                raise ConfigValidationError("knowledge_cache", key, "Missing required field")
        self._assert_type(kc["max_size"], int, "knowledge_cache", "max_size")
        self._assert_type(kc["enable_encryption"], bool, "knowledge_cache", "enable_encryption")
        self._assert_in(kc["hashing_method"], ["md5", "simhash"], "knowledge_cache", "hashing_method")
        if "stopwords" in kc:
            self._assert_path_exists(kc["stopwords"], "knowledge_cache", "stopwords", is_directory=False)

    def _validate_knowledge_monitor(self) -> None:
        km = self.config.get("knowledge_monitor", {})
        required = ["enabled", "check_interval", "violation_policy"]
        for key in required:
            if key not in km:
                raise ConfigValidationError("knowledge_monitor", key, "Missing required field")
        self._assert_type(km["enabled"], bool, "knowledge_monitor", "enabled")
        self._assert_type(km["check_interval"], int, "knowledge_monitor", "check_interval")
        self._assert_in(km["violation_policy"], ["log", "quarantine", "alert"], "knowledge_monitor", "violation_policy")

    def _validate_knowledge_sync(self) -> None:
        ks = self.config.get("knowledge_sync", {})
        if not ks:
            return
        if "conflict_resolution" in ks:
            cr = ks["conflict_resolution"]
            if "strategy" in cr:
                self._assert_in(cr["strategy"], ["timestamp", "confidence", "semantic", "governance"], "knowledge_sync", "conflict_resolution.strategy")
        if "versioning" in ks:
            ver = ks["versioning"]
            if "enabled" in ver:
                self._assert_type(ver["enabled"], bool, "knowledge_sync", "versioning.enabled")

    def _validate_ontology_manager(self) -> None:
        om = self.config.get("ontology_manager", {})
        required = ["knowledge_ontology_path"]
        for key in required:
            if key not in om:
                raise ConfigValidationError("ontology_manager", key, "Missing required field")
        self._assert_path_exists(om["knowledge_ontology_path"], "ontology_manager", "knowledge_ontology_path", is_directory=False)

    def _validate_runtime_health(self) -> None:
        rh = self.config.get("runtime_health", {})
        if not rh:
            return
        if "degraded_after_seconds" in rh:
            self._assert_type(rh["degraded_after_seconds"], (int, float), "runtime_health", "degraded_after_seconds")
        if "unhealthy_after_seconds" in rh:
            self._assert_type(rh["unhealthy_after_seconds"], (int, float), "runtime_health", "unhealthy_after_seconds")
        if "enable_periodic_checks" in rh:
            self._assert_type(rh["enable_periodic_checks"], bool, "runtime_health", "enable_periodic_checks")
        if "check_interval_seconds" in rh:
            self._assert_type(rh["check_interval_seconds"], (int, float), "runtime_health", "check_interval_seconds")

    def _validate_runtime_metrics(self) -> None:
        rm = self.config.get("runtime_metrics", {})
        if not rm:
            return
        if "prometheus_export_port" in rm:
            self._assert_type(rm["prometheus_export_port"], int, "runtime_metrics", "prometheus_export_port")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def get_validated_section(self, section: str) -> Dict[str, Any]:
        """Return a validated section (already validated during init)."""
        return self.config.get(section, {})

    def is_enabled(self) -> bool:
        return self.config.get("enabled", False)


# -----------------------------------------------------------------------------
# Test block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Validating Configuration Schema ===\n")
    printer.status("Init", "Config Schema test started", "info")

    try:
        schema = ConfigSchema()
        print("✓ Configuration validation passed")
        print(f"  Enabled: {schema.is_enabled()}")
        print(f"  Rule timeout: {schema.get_validated_section('rule_engine').get('rule_timeout_seconds')}")
    except ConfigValidationError as e:
        print(f"✗ Validation failed: {e}")
        raise

    print("\n=== Config Schema validation complete ===")