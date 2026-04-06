from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from finance.core.utils.config_loader import load_global_config, get_config_section
from finance.core.utils.financial_errors import (InvalidConfigurationError, log_error,
                                                 StateStoreUnavailableError, ErrorContext,
                                                 PersistenceError, ValidationError)
from finance.core.finance_memory import FinanceMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Batch Manager")
printer = PrettyPrinter

DEFAULT_MAX_BATCHES_PER_FILE = 2016
DEFAULT_BATCH_SAVE_INTERVAL = 300
DEFAULT_FLUSH_CHECK_INTERVAL = 60
DEFAULT_BATCHES_DIR = "finance/checkpoints"
DEFAULT_ACTIVE_BATCH_FILE_NAME = "active_batches.json"
DEFAULT_ARCHIVE_PREFIX = "datapack"
DEFAULT_BATCH_PRIORITY = "high"


@dataclass(slots=True)
class DataPoint:
    symbol: str
    price: float
    volume: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BatchRecord:
    batch_id: str
    batch_timestamp: float
    created_at: str
    reason: str
    symbol_count: int
    data: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BatchManager:
    def __init__(self) -> None:
        self.config = load_global_config()
        self.manager_config = get_config_section("batch_manager")
        self.memory = FinanceMemory()

        self.max_batches_per_file = int(self.manager_config.get("max_batches_per_file", DEFAULT_MAX_BATCHES_PER_FILE))
        self.batch_save_interval = int(self.manager_config.get("batch_save_interval", DEFAULT_BATCH_SAVE_INTERVAL))
        self.flush_check_interval = int(self.manager_config.get("flush_check_interval", DEFAULT_FLUSH_CHECK_INTERVAL))
        self.batch_priority = str(self.manager_config.get("batch_priority", DEFAULT_BATCH_PRIORITY)).lower()
        self.archive_prefix = str(self.manager_config.get("archive_prefix", DEFAULT_ARCHIVE_PREFIX))
        self.active_batch_file_name = str(self.manager_config.get("active_batch_file_name", DEFAULT_ACTIVE_BATCH_FILE_NAME))
        self.batches_dir = str(self.manager_config.get("batches_dir", self.memory.checkpoint_dir or DEFAULT_BATCHES_DIR))
        self.auto_start_controller = bool(self.manager_config.get("auto_start_controller", True))
        self.persist_to_memory = bool(self.manager_config.get("persist_to_memory", True))
        self.persist_to_disk = bool(self.manager_config.get("persist_to_disk", True))

        self._validate_configuration()

        self.current_batch_data_points: Dict[str, Dict[str, Any]] = {}
        self.batches_in_active_file: List[Dict[str, Any]] = []
        self.last_save_time = time.time()
        self.archived_file_counter = 0
        self.batch_sequence = 0
        self.active_file_path = os.path.join(self.batches_dir, self.active_batch_file_name)

        self.data_points_lock = threading.RLock()
        self.file_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(
            target=self._periodic_flush_controller,
            daemon=True,
            name="BatchManagerFlushController",
        )

        Path(self.batches_dir).mkdir(parents=True, exist_ok=True)
        self._load_or_initialize_active_file()
        if self.auto_start_controller:
            self._flush_thread.start()

        logger.info(
            "Batch Manager initialized | max_batches_per_file=%s batch_save_interval=%ss flush_check_interval=%ss active_file=%s",
            self.max_batches_per_file,
            self.batch_save_interval,
            self.flush_check_interval,
            os.path.abspath(self.active_file_path),
        )

    # ------------------------------------------------------------------
    # Lifecycle and validation
    # ------------------------------------------------------------------

    def __enter__(self) -> "BatchManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop_controller()

    def _context(self, operation: str, **metadata: Any) -> ErrorContext:
        return ErrorContext(
            component="batch_manager",
            operation=operation,
            metadata=metadata or {},
        )

    def _validate_configuration(self) -> None:
        if self.max_batches_per_file <= 0:
            raise InvalidConfigurationError(
                "batch_manager.max_batches_per_file must be positive.",
                context=self._context("validate_config", max_batches_per_file=self.max_batches_per_file),
            )
        if self.batch_save_interval <= 0:
            raise InvalidConfigurationError(
                "batch_manager.batch_save_interval must be positive.",
                context=self._context("validate_config", batch_save_interval=self.batch_save_interval),
            )
        if self.flush_check_interval <= 0:
            raise InvalidConfigurationError(
                "batch_manager.flush_check_interval must be positive.",
                context=self._context("validate_config", flush_check_interval=self.flush_check_interval),
            )
        if not self.batches_dir or not str(self.batches_dir).strip():
            raise InvalidConfigurationError(
                "batch_manager.batches_dir must not be empty.",
                context=self._context("validate_config"),
            )

    def start_controller(self) -> None:
        if self._flush_thread.is_alive():
            return
        self._stop_event.clear()
        self._flush_thread = threading.Thread(
            target=self._periodic_flush_controller,
            daemon=True,
            name="BatchManagerFlushController",
        )
        self._flush_thread.start()
        logger.info("BatchManager periodic flush controller started.")

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def _atomic_write_json(self, path: str, payload: Any) -> None:
        directory = os.path.dirname(path) or "."
        fd, temp_path = tempfile.mkstemp(prefix=".batch_manager_", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _load_json_file(self, path: str) -> Any:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_or_initialize_active_file(self) -> None:
        with self.file_lock:
            self.batches_in_active_file = []
            if not os.path.exists(self.active_file_path):
                self._write_active_file_locked()
                return

            try:
                content = self._load_json_file(self.active_file_path)
                if isinstance(content, list):
                    self.batches_in_active_file = content
                elif isinstance(content, Mapping) and isinstance(content.get("batches"), list):
                    self.batches_in_active_file = list(content["batches"])
                else:
                    logger.warning("Active batch file %s does not contain a valid batch list. Reinitializing.", self.active_file_path)
                    self.batches_in_active_file = []
                    self._write_active_file_locked()
                    return

                self.archived_file_counter = self._discover_archive_counter()
                self.batch_sequence = max(self.batch_sequence, len(self.batches_in_active_file))
                logger.info("Loaded %s batches from %s", len(self.batches_in_active_file), self.active_file_path)
            except json.JSONDecodeError as exc:
                handled = PersistenceError(
                    "Active batch file is corrupted and could not be decoded.",
                    context=self._context("load_active_file", path=self.active_file_path),
                    cause=exc,
                )
                log_error(handled, logger_=logger)
                self.batches_in_active_file = []
                self._write_active_file_locked()
            except OSError as exc:
                handled = StateStoreUnavailableError(
                    "Failed to read active batch file.",
                    context=self._context("load_active_file", path=self.active_file_path),
                    cause=exc,
                )
                log_error(handled, logger_=logger)
                raise handled from exc

    def _write_active_file(self) -> None:
        with self.file_lock:
            self._write_active_file_locked()

    def _write_active_file_locked(self) -> None:
        try:
            self._atomic_write_json(self.active_file_path, self.batches_in_active_file)
            logger.debug("Wrote %s batches to %s", len(self.batches_in_active_file), self.active_file_path)
        except Exception as exc:
            handled = PersistenceError(
                "Failed to write active batch file.",
                context=self._context("write_active_file", path=self.active_file_path),
                cause=exc,
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def _discover_archive_counter(self) -> int:
        counter = 0
        if not os.path.isdir(self.batches_dir):
            return counter
        for filename in os.listdir(self.batches_dir):
            if not (filename.startswith(f"{self.archive_prefix}_") and filename.endswith(".json")):
                continue
            stem = filename[:-5]
            try:
                sequence = int(stem.rsplit("_", 1)[-1])
            except Exception:
                continue
            counter = max(counter, sequence)
        return counter

    def _archive_active_file(self) -> Optional[str]:
        with self.file_lock:
            return self._archive_active_file_locked()

    def _archive_active_file_locked(self) -> Optional[str]:
        if not self.batches_in_active_file:
            logger.info("Active batch file is empty; nothing to archive.")
            return None

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.archived_file_counter += 1
        archived_file_name = f"{self.archive_prefix}_{timestamp_str}_{self.archived_file_counter:04d}.json"
        archived_file_path = os.path.join(self.batches_dir, archived_file_name)

        try:
            if os.path.exists(self.active_file_path):
                shutil.move(self.active_file_path, archived_file_path)
            else:
                self._atomic_write_json(archived_file_path, self.batches_in_active_file)
            self.batches_in_active_file = []
            self._write_active_file_locked()
            logger.info("Archived active batch file to %s", archived_file_path)
            return archived_file_path
        except Exception as exc:
            handled = PersistenceError(
                "Failed to archive active batch file.",
                context=self._context(
                    "archive_active_file",
                    source_path=self.active_file_path,
                    destination_path=archived_file_path,
                ),
                cause=exc,
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def add_data_point(
        self,
        symbol: str,
        price: float,
        volume: float = 0.0,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        event_time: Optional[float] = None,
    ) -> None:
        normalized_symbol = str(symbol).strip().upper() if isinstance(symbol, str) else ""
        if not normalized_symbol:
            raise ValidationError(
                "symbol must be a non-empty string.",
                context=self._context("add_data_point", symbol=symbol),
            )
        if not isinstance(price, (float, int)):
            raise ValidationError(
                "price must be numeric.",
                context=self._context("add_data_point", symbol=normalized_symbol, price=price),
            )
        if float(price) <= 0:
            raise ValidationError(
                "price must be positive.",
                context=self._context("add_data_point", symbol=normalized_symbol, price=price),
            )
        if not isinstance(volume, (float, int)) or float(volume) < 0:
            raise ValidationError(
                "volume must be a non-negative number.",
                context=self._context("add_data_point", symbol=normalized_symbol, volume=volume),
            )

        point = DataPoint(
            symbol=normalized_symbol,
            price=float(price),
            volume=float(volume),
            timestamp=float(event_time or time.time()),
            metadata=dict(metadata or {}),
        )
        with self.data_points_lock:
            self.current_batch_data_points[normalized_symbol] = point.to_dict()
        logger.debug(
            "Buffered data point | symbol=%s price=%.6f volume=%.6f buffer_size=%s",
            normalized_symbol,
            float(price),
            float(volume),
            len(self.current_batch_data_points),
        )

    def add_data_points(self, data_points: Iterable[Mapping[str, Any]]) -> int:
        count = 0
        for item in data_points:
            self.add_data_point(
                symbol=str(item.get("symbol", "")),
                price=float(item.get("price", 0.0)),
                volume=float(item.get("volume", 0.0)),
                metadata=item.get("metadata") if isinstance(item.get("metadata"), Mapping) else None,
                event_time=item.get("timestamp"),
            )
            count += 1
        return count

    def _build_batch_record(self, data_to_save: Dict[str, Dict[str, Any]], *, reason: str) -> BatchRecord:
        self.batch_sequence += 1
        batch_timestamp = time.time()
        batch_id = f"batch_{int(batch_timestamp)}_{self.batch_sequence:08d}"
        return BatchRecord(
            batch_id=batch_id,
            batch_timestamp=batch_timestamp,
            created_at=datetime.now(timezone.utc).isoformat(),
            reason=reason,
            symbol_count=len(data_to_save),
            data=data_to_save,
        )

    def form_and_save_batch(self, *, reason: str = "manual") -> Optional[Dict[str, Any]]:
        with self.data_points_lock:
            if not self.current_batch_data_points:
                logger.info("form_and_save_batch called with an empty buffer.")
                return None
            data_to_save = dict(self.current_batch_data_points)
            self.current_batch_data_points.clear()

        batch_record = self._build_batch_record(data_to_save, reason=reason)
        batch_payload = batch_record.to_dict()

        try:
            if self.persist_to_memory:
                self.memory.add_financial_data(
                    data=batch_payload,
                    data_type="batch",
                    tags=["batch_data", f"src_batch_{batch_record.batch_id}"],
                    priority=self.batch_priority,
                    metadata={"batch_id": batch_record.batch_id, "symbol_count": batch_record.symbol_count},
                )

            if self.persist_to_disk:
                with self.file_lock:
                    self.batches_in_active_file.append(batch_payload)
                    if len(self.batches_in_active_file) >= self.max_batches_per_file:
                        self._write_active_file_locked()
                        self._archive_active_file_locked()
                    else:
                        self._write_active_file_locked()

            self.last_save_time = time.time()
            logger.info(
                "Saved batch | batch_id=%s symbol_count=%s active_batches=%s reason=%s",
                batch_record.batch_id,
                batch_record.symbol_count,
                len(self.batches_in_active_file),
                reason,
            )
            return batch_payload
        except Exception as exc:
            handled = PersistenceError(
                "Failed to persist formed batch.",
                context=self._context("form_and_save_batch", batch_id=batch_record.batch_id, reason=reason),
                cause=exc,
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def flush(self, *, reason: str = "manual_flush") -> Optional[Dict[str, Any]]:
        return self.form_and_save_batch(reason=reason)

    # ------------------------------------------------------------------
    # Background flush controller
    # ------------------------------------------------------------------

    def _periodic_flush_controller(self) -> None:
        logger.info(
            "BatchManager periodic flush controller started | check_interval=%ss save_interval=%ss",
            self.flush_check_interval,
            self.batch_save_interval,
        )
        try:
            while not self._stop_event.wait(self.flush_check_interval):
                try:
                    self._trigger_timed_flush()
                except Exception as exc:
                    handled = PersistenceError(
                        "Timed batch flush failed.",
                        context=self._context("periodic_flush_controller"),
                        cause=exc,
                    )
                    log_error(handled, logger_=logger)
        finally:
            logger.info("BatchManager periodic flush controller stopped.")

    def _trigger_timed_flush(self) -> None:
        current_time = time.time()
        time_since_last_save = current_time - self.last_save_time
        if time_since_last_save < self.batch_save_interval:
            return

        with self.data_points_lock:
            has_data = bool(self.current_batch_data_points)
            num_points = len(self.current_batch_data_points)

        if has_data:
            logger.info(
                "Timed flush triggered | elapsed=%.2fs buffered_points=%s",
                time_since_last_save,
                num_points,
            )
            self.form_and_save_batch(reason="timed_flush")
        else:
            self.last_save_time = current_time
            logger.info("Timed flush reached interval but no buffered data was available.")

    def stop_controller(self) -> None:
        logger.info("Stopping BatchManager periodic flush controller...")
        self._stop_event.set()
        if self._flush_thread.is_alive():
            self._flush_thread.join(timeout=max(10, self.flush_check_interval + 5))

        with self.data_points_lock:
            has_data = bool(self.current_batch_data_points)
        if has_data:
            self.form_and_save_batch(reason="shutdown_flush")
        logger.info("BatchManager controller stopped and final flush completed.")

    shutdown = stop_controller

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def get_all_batches(self, *, include_disk: bool = False) -> List[Dict[str, Any]]:
        batches: List[Dict[str, Any]] = []
        try:
            batches = list(
                self.memory.query(
                    data_type="batch",
                    tags=["batch_data"],
                    limit=self.max_batches_per_file * 10,
                )
            )
        except Exception as exc:
            handled = PersistenceError(
                "Failed to query batch records from finance memory.",
                context=self._context("get_all_batches"),
                cause=exc,
            )
            log_error(handled, logger_=logger)
            if not include_disk:
                raise handled from exc

        if include_disk:
            disk_batches = self.load_all_batches_from_disk()
            if not batches:
                return disk_batches
            memory_batch_ids = {
                entry.get("data", {}).get("batch_id")
                for entry in batches
                if isinstance(entry, Mapping)
            }
            for disk_batch in disk_batches:
                if disk_batch.get("batch_id") not in memory_batch_ids:
                    batches.append({"data": disk_batch, "metadata": {"source": "disk"}})
        return batches

    def load_all_batches_from_disk(self) -> List[Dict[str, Any]]:
        logger.info("Loading all batch data from disk directory: %s", self.batches_dir)
        all_batches: List[Dict[str, Any]] = []
        file_paths_to_load: List[str] = []
        if os.path.exists(self.active_file_path):
            file_paths_to_load.append(self.active_file_path)

        for filename in os.listdir(self.batches_dir):
            if filename.startswith(f"{self.archive_prefix}_") and filename.endswith(".json"):
                full_path = os.path.join(self.batches_dir, filename)
                if full_path not in file_paths_to_load:
                    file_paths_to_load.append(full_path)

        file_paths_to_load.sort()
        for file_path in file_paths_to_load:
            try:
                content = self._load_json_file(file_path)
                if isinstance(content, list):
                    all_batches.extend(content)
                elif isinstance(content, Mapping) and isinstance(content.get("batches"), list):
                    all_batches.extend(content["batches"])
                else:
                    logger.warning("Batch file %s does not contain a valid batch list. Skipping.", file_path)
            except json.JSONDecodeError as exc:
                handled = PersistenceError(
                    "Failed to decode batch archive JSON.",
                    context=self._context("load_all_batches_from_disk", path=file_path),
                    cause=exc,
                )
                log_error(handled, logger_=logger)
            except OSError as exc:
                handled = StateStoreUnavailableError(
                    "Failed to read batch archive file.",
                    context=self._context("load_all_batches_from_disk", path=file_path),
                    cause=exc,
                )
                log_error(handled, logger_=logger)

        logger.info("Loaded %s batch records from %s files.", len(all_batches), len(file_paths_to_load))
        return all_batches


if __name__ == "__main__":  # pragma: no cover
    printer.status("INIT", "Testing Batch Manager", "info")
    manager = BatchManager()
    try:
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN"]
        prices = [182.1, 411.5, 912.3, 178.4]
        volumes = [1000, 1200, 900, 800]
        for symbol, price, volume in zip(symbols, prices, volumes):
            manager.add_data_point(symbol=symbol, price=price, volume=volume)

        printer.pretty("flush", manager.flush(reason="example"), "success")
        printer.pretty("memory_batches", manager.get_all_batches(), "info")
        printer.pretty("disk_batches", manager.load_all_batches_from_disk(), "info")
    finally:
        manager.stop_controller()
        printer.status("RESULT", "Batch Manager test completed", "success")
