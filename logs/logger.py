from __future__ import annotations

import gzip
import logging
import os, sys
import time
import math
import queue
import hashlib
import zlib
import statistics
import atexit
import shutil
import pprint
import threading
if os.name == 'nt':
    import msvcrt
else:
    msvcrt = None
import uuid
from logging.handlers import RotatingFileHandler
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .standards import LogDomain, default_log_path
if TYPE_CHECKING:
    from monitoring.system_optimizer import SystemOptimizer # pyright: ignore[reportMissingImports]

# ========== Status Tags ==========
INIT       = "[INIT]"
START      = "[START]"
STOP       = "[STOP]"
RESTART    = "[RESTART]"
RUNNING    = "[RUNNING]"
IDLE       = "[IDLE]"
SLEEP      = "[SLEEP]"
PAUSE      = "[PAUSE]"
RESUME     = "[RESUME]"
DONE       = "[DONE]"
COMPLETE   = "[COMPLETE]"

# ========== Information Tags ==========
INFO       = "[INFO]"
STATUS     = "[STATUS]"
REFRESH    = "[REFRESH]"
UPDATE     = "[UPDATE]"
SYNC       = "[SYNC]"
LOAD       = "[LOAD]"
SAVE       = "[SAVE]"
CONFIG     = "[CONFIG]"
CACHE      = "[CACHE]"
DATA       = "[DATA]"
ENV        = "[ENV]"

# ========== Web/Network Tags ==========
WEB        = "[WEB]"
FETCH      = "[FETCH]"
API        = "[API]"
REQUEST    = "[REQ]"
RESPONSE   = "[RESP]"
CONNECT    = "[CONNECT]"
DISCONNECT = "[DISCONNECT]"

# ========== Learning & Agents ==========
LEARN      = "[LEARN]"
TRAIN      = "[TRAIN]"
INFER      = "[INFER]"
AGENT      = "[AGENT]"
MEMORY     = "[MEM]"
TASK       = "[TASK]"
TRIGGER    = "[TRIGGER]"
CYCLE      = "[CYCLE]"
EVAL       = "[EVAL]"

# ========== Debugging & Performance ==========
DEBUG      = "[DEBUG]"
TRACE      = "[TRACE]"
PERF       = "[PERF]"
SPEED      = "[SPEED]"
METRIC     = "[METRIC]"
SCORE      = "[SCORE]"

# ========== Result Tags ==========
SUCCESS    = "[OK]"
FAILURE    = "[FAIL]"
ERROR      = "[ERROR]"
WARN       = "[WARN]"
EXCEPTION  = "[EXCEPTION]"
TIMEOUT    = "[TIMEOUT]"

# ========== Misc ==========
USER       = "[USER]"
AUTH       = "[AUTH]"
SECURE     = "[SECURE]"
RETRY      = "[RETRY]"
SKIP       = "[SKIP]"
EVENT      = "[EVENT]"

COLOR_CODES = {
    'RESET': "\033[0m",
    'BLUE': "\033[94m",
    'GREEN': "\033[92m",
    'YELLOW': "\033[93m",
    'RED': "\033[91m",
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'white': '\033[37m',
    'black': '\033[30m',
}
STYLES = {
    'reset': '\033[0m',
    'bold': '\033[1m',
    'dim': '\033[2m',
    'italic': '\033[3m',
    'underline': '\033[4m',
    'blink': '\033[5m',
    'inverse': '\033[7m',
    'hidden': '\033[8m',
    'strike': '\033[9m',

    'black': '\033[30m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'white': '\033[37m',
    'bg_black': '\033[40m',
    'bg_red': '\033[41m',
    'bg_green': '\033[42m',
    'bg_yellow': '\033[43m',
    'bg_blue': '\033[44m',
    'bg_magenta': '\033[45m',
    'bg_cyan': '\033[46m',
    'bg_white': '\033[47m',
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',

    # see https://www.ditig.com/256-colors-cheat-sheet
    'Grey0': '\033[38;5;16m',   # Darkest
    'NavyBlue': '\033[38;5;17m',
    'DarkBlue': '\033[38;5;18m',
    'Blue3a': '\033[38;5;19m',
    'Blue3b': '\033[38;5;20m',
    'Blue1': '\033[38;5;21m',
    'DarkGreen': '\033[38;5;22m',
    'DeepSkyBlue4a': '\033[38;5;23m',
    'DeepSkyBlue4b': '\033[38;5;24m',
    'DeepSkyBlue4c': '\033[38;5;25m',
    'DodgerBlue3': '\033[38;5;26m',
    'DodgerBlue2': '\033[38;5;27m',
    'Green4': '\033[38;5;28m',
    'SpringGreen4': '\033[38;5;29m',
    'color14': '\033[38;5;30m',
    'DeepSkyBlue3a': '\033[38;5;31m',
    'DeepSkyBlue3b': '\033[38;5;32m',
    'DodgerBlue1': '\033[38;5;33m',
    'Green3': '\033[38;5;34m',
    'SpringGreen3': '\033[38;5;35m',
    'DarkCyan': '\033[38;5;36m',
    'LightSeaGreen': '\033[38;5;37m',
    'DeepSkyBlue2': '\033[38;5;38m',
    'DeepSkyBlue1': '\033[38;5;39m',
    'Green3': '\033[38;5;40m',
    'SpringGreen3': '\033[38;5;41m',
    'SpringGreen2': '\033[38;5;42m',
    'Cyan3': '\033[38;5;43m',
    'DarkTurquoise': '\033[38;5;44m',
    'Turquoise2': '\033[38;5;45m',
    'Green1': '\033[38;5;46m',
    'SpringGreen2': '\033[38;5;47m',
    'SpringGreen1': '\033[38;5;48m',
    'MediumSpringGreen': '\033[38;5;49m',
    'Cyan2': '\033[38;5;50m',
    'Cyan1': '\033[38;5;51m',
    'DarkRed': '\033[38;5;52m',
    'DeepPink4': '\033[38;5;53m',
    'Purple4a': '\033[38;5;54m',
    'Purple4b': '\033[38;5;55m',
    'Purple3': '\033[38;5;56m',
    'BlueViolet': '\033[38;5;57m',
    'Orange4': '\033[38;5;58m',
    'Grey37': '\033[38;5;59m',
    'MediumPurple4': '\033[38;5;60m',
    'SlateBlue3a': '\033[38;5;61m',
    'SlateBlue3b': '\033[38;5;62m',
    'DarkSeaGreen4': '\033[38;5;65m',
    'PaleTurquoise4': '\033[38;5;66m',   # Mid-range
    'SteelBlue': '\033[38;5;67m',
    'SteelBlue3': '\033[38;5;68m',
    'CornflowerBlue': '\033[38;5;69m',
    'Chartreuse3': '\033[38;5;70m',
    'DarkSeaGreen4': '\033[38;5;71m',
    'CadetBlue': '\033[38;5;72m',
    'CadetBlue': '\033[38;5;73m',
    'SkyBlue3': '\033[38;5;74m',
    'SteelBlue1': '\033[38;5;75m',
    'Chartreuse3': '\033[38;5;76m',
    'PaleGreen3': '\033[38;5;77m',
    'SeaGreen3': '\033[38;5;78m',
    'Aquamarine3': '\033[38;5;79m',
    'MediumTurquoise': '\033[38;5;80m',
    'SteelBlue1': '\033[38;5;81m',
    'Chartreuse2': '\033[38;5;82m',
    'SeaGreen2': '\033[38;5;83m',
    'SeaGreen1a': '\033[38;5;84m',
    'SeaGreen1b': '\033[38;5;85m',
    'Aquamarine1': '\033[38;5;86m',
    'DarkSlateGray2': '\033[38;5;87m',
    'DarkRed': '\033[38;5;88m',
    'DeepPink4': '\033[38;5;89m',
    'DarkMagentaA': '\033[38;5;90m',
    'DarkMagentaB': '\033[38;5;91m',
    'DarkViolet': '\033[38;5;92m',
    'Purple': '\033[38;5;93m',
    'Orange4': '\033[38;5;94m',
    'LightPink4': '\033[38;5;95m',
    'Plum4': '\033[38;5;96m',
    'MediumPurple3a': '\033[38;5;97m',
    'MediumPurple3b': '\033[38;5;98m',
    'SlateBlue1': '\033[38;5;99m',
    'Yellow4': '\033[38;5;100m',
    'Wheat4': '\033[38;5;101m',
    'Grey53': '\033[38;5;102m',
    'LightSlateGrey': '\033[38;5;103m',
    'MediumPurple': '\033[38;5;104m',
    'LightSlateBlue': '\033[38;5;105m',
    'Yellow4': '\033[38;5;106m',
    'DarkOliveGreen3': '\033[38;5;107m',
    'DarkSeaGreen': '\033[38;5;108m',
    'LightSkyBlue3a': '\033[38;5;109m',
    'LightSkyBlue3b': '\033[38;5;110m',
    'SkyBlue2': '\033[38;5;111m',
    'Chartreuse2': '\033[38;5;112m',
    'DarkOliveGreen3': '\033[38;5;113m',
    'PaleGreen3': '\033[38;5;114m',
    'DarkSeaGreen3': '\033[38;5;115m',   # Lightest
    'Orange1': '\033[38;5;214m',
    'Gold1': '\033[38;5;220m',
}

# Shared logging queue
log_queue = queue.Queue()

# Logging is configured explicitly by application entry points. Merely importing
# this module or declaring a module logger must not touch files, streams, or the
# process-wide root logger.
_logger_initialized = False
_logging_lock = threading.RLock()
_atexit_registered = False
_HANDLER_MARKER = "_slai_managed_handler"


@dataclass(frozen=True)
class LoggingSettings:
    level: int = logging.INFO
    console: bool = True
    file: bool = True
    queue: bool = True
    log_path: Path = default_log_path(LogDomain.RUNTIME, "app.log")
    max_bytes: int = 1_000_000
    backup_count: int = 5


def _mark_managed(handler: logging.Handler) -> logging.Handler:
    setattr(handler, _HANDLER_MARKER, True)
    return handler


def _managed_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [handler for handler in logger.handlers if getattr(handler, _HANDLER_MARKER, False)]

class ColorFormatter(logging.Formatter):
    """Formatter that adds color to the level name only."""
    _level_colors = {
        logging.WARNING:  STYLES['yellow'],
        logging.ERROR:    STYLES['red'],
        logging.CRITICAL: STYLES['magenta'],
        # All other levels (DEBUG, INFO) use no extra color (default white)
    }

    def format(self, record: logging.LogRecord) -> str:
        # Save original levelname
        original_levelname = record.levelname
        # Determine color
        color = self._level_colors.get(record.levelno, '')
        if color and sys.stdout.isatty():
            record.levelname = f"{color}{original_levelname}{STYLES['reset']}"
        # Let the parent formatter do the rest (timestamp, name, message)
        result = super().format(record)
        # Restore original levelname to avoid side effects
        record.levelname = original_levelname
        return result

class QueueLogHandler(logging.Handler):
    def __init__(self, q: queue.Queue, batch_size: int = 10, flush_interval: int = 5) -> None:
        super().__init__()
        self.queue = q
        self.batch = []
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.last_flush = time.time()
        self.hash_chain = hashlib.sha256(b'initial_seed').hexdigest()

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.batch.append(msg)
        current_time = time.time()

        # Batch processing using Little's Law fundamentals
        if len(self.batch) >= self.batch_size or \
           current_time - self.last_flush >= self.flush_interval:
            self._flush_batch()

    def _flush_batch(self) -> None:
        # Cryptographic chaining for tamper evidence
        chain_hash = self.hash_chain
        for msg in self.batch:
            chain_hash = hashlib.sha256(chain_hash.encode('utf-8') + msg.encode('utf-8')).hexdigest()
            self.queue.put((chain_hash, msg))
        self.hash_chain = chain_hash
        self.batch.clear()
        self.last_flush = time.time()

    def flush(self) -> None:
        """Publish a partial batch during explicit logging shutdown."""

        self.acquire()
        try:
            if self.batch:
                self._flush_batch()
        finally:
            self.release()
        super().flush()

def get_logger(name: str) -> logging.Logger:
    """Return a named logger without configuring global logging."""

    return logging.getLogger(name)


def configure_logging(settings: LoggingSettings | None = None, *, force: bool = False) -> logging.Logger:
    """Configure SLAI-owned root handlers explicitly and idempotently.

    Existing handlers owned by embedding applications, test runners, or other
    libraries are preserved. ``force=True`` replaces only SLAI-managed
    handlers.
    """

    global _logger_initialized, _atexit_registered
    settings = settings or LoggingSettings()

    with _logging_lock:
        root_logger = logging.getLogger()
        if _logger_initialized and not force:
            return root_logger
        if force:
            shutdown_logging()

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        root_logger.setLevel(settings.level)

        if settings.file:
            settings.log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = _mark_managed(
                RotatingHandler(
                    str(settings.log_path),
                    maxBytes=settings.max_bytes,
                    backupCount=settings.backup_count,
                    delay=True,
                    encoding="utf-8",
                    errors="replace",
                )
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            root_logger.addHandler(file_handler)

        if settings.console:
            console_handler = _mark_managed(logging.StreamHandler(sys.stdout))
            console_handler.setFormatter(ColorFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            root_logger.addHandler(console_handler)

        if settings.queue:
            queue_handler = _mark_managed(QueueLogHandler(log_queue, batch_size=10, flush_interval=5))
            queue_handler.setFormatter(formatter)
            root_logger.addHandler(queue_handler)

        _logger_initialized = True
        if not _atexit_registered:
            atexit.register(shutdown_logging)
            _atexit_registered = True
        return root_logger

def get_log_queue():
    return log_queue

def cleanup_logger(name):
    """
    Clean up and close all handlers for a given logger.
    Useful before rollback or app shutdown to release file locks.
    """
    logger = logging.getLogger(name)
    handlers = logger.handlers[:] if name is not None else _managed_handlers(logger)
    for handler in handlers:
        try:
            handler.flush()
        except (OSError, ValueError):
            # Console and test-capture streams may already be closed during
            # interpreter shutdown. Cleanup must remain best effort.
            pass
        try:
            handler.close()
        except (OSError, ValueError):
            pass
        finally:
            logger.removeHandler(handler)

def shutdown_logging() -> None:
    """Flush and close only handlers installed by :func:`configure_logging`."""

    global _logger_initialized
    with _logging_lock:
        cleanup_logger(None)
        _logger_initialized = False


def exit_handler() -> None:
    """Backward-compatible alias for explicit logging shutdown."""

    shutdown_logging()

class RotatingHandler(RotatingFileHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compress_queue = deque(maxlen=5)
        self.compress_threshold = 5
        self.last_rollover_time = 0
        self.rollover_cooldown = 60

    def doRollover(self):
        current_time = time.time()
        if current_time - self.last_rollover_time < self.rollover_cooldown:
            return

        self.last_rollover_time = current_time
        errors = []
        
        # Only close this handler's stream
        self.close()
        
        # Generate backup filename
        timestamp = int(time.time())
        unique_id = uuid.uuid4().hex[:6]
        dfn = self.rotation_filename(f"{self.baseFilename}.{timestamp}_{unique_id}")

        # Remove existing backup if needed
        if os.path.exists(dfn):
            try:
                os.remove(dfn)
            except OSError as e:
                errors.append(f"Failed to remove existing backup {dfn}: {e}")

        # Rotate current log to backup
        if os.path.exists(self.baseFilename):
            for attempt in range(5):
                try:
                    os.rename(self.baseFilename, dfn)
                    break
                except OSError as e:
                    if attempt == 4:
                        errors.append(f"Failed to rename log file: {e}")
                    else:
                        time.sleep(0.5)

        # Reopen current log
        if not self.delay:
            try:
                self.stream = self._open()
            except Exception as e:
                errors.append(f"Failed to reopen log: {e}")

        # Enforce log limits and compression
        try:
            self._enforce_log_limits(max_total_gb=40, max_files=300)
        except Exception as e:
            errors.append(f"Log limits enforcement failed: {e}")
            
        try:
            self._compress_queue.append(dfn)
            self._manage_compression()
        except Exception as e:
            errors.append(f"Compression failed: {e}")

        # Log any errors
        if errors:
            try:
                logger = logging.getLogger("RotatingHandler")
                for error in errors:
                    logger.error(error)
            except Exception:
                pass  # Fallback if logging unavailable

    def _enforce_log_limits(self, max_total_gb=40, max_files=300):
        max_total_bytes = max_total_gb * (1024**3)
        log_dir_path = os.path.dirname(self.baseFilename) or '.'
        
        if not os.path.isdir(log_dir_path):
            return
    
        file_infos = []
        base_name = os.path.basename(self.baseFilename)
        
        for filename in os.listdir(log_dir_path):
            file_path = os.path.join(log_dir_path, filename)
            if not os.path.isfile(file_path):
                continue
                
            # Only match backup files (base_name + extra characters)
            if filename.startswith(base_name + '.'):
                try:
                    mtime = os.path.getmtime(file_path)
                    size = os.path.getsize(file_path)
                    file_infos.append((file_path, mtime, size))
                except OSError:
                    continue
    
        file_infos.sort(key=lambda x: x[1])  # Sort by mtime
        total_size = sum(size for _, _, size in file_infos)
        file_count = len(file_infos)
        files_to_delete = []
    
        for file_path, _, size in file_infos:
            if file_count <= max_files and total_size <= max_total_bytes:
                break
                
            files_to_delete.append(file_path)
            total_size -= size
            file_count -= 1
    
        # Delete files with retries
        for file_path in files_to_delete:
            for attempt in range(3):
                try:
                    os.remove(file_path)
                    break
                except Exception as e:
                    if attempt == 2:
                        logging.error(f"Failed to delete {file_path} after 3 attempts: {e}")
        
    def _manage_compression(self):
        while self._compress_queue:
            path = self._compress_queue.popleft()
            if not os.path.exists(path):
                continue
            gz_path = path + '.gz'
            try:
                # Explicit chunked copy to avoid type confusion and memory spikes
                with open(path, 'rb') as f_in:
                    with gzip.open(gz_path, 'wb') as f_out:
                        while chunk := f_in.read(64 * 1024):   # 64 KiB chunks
                            f_out.write(chunk) # pyright: ignore[reportArgumentType]
                os.remove(path)
            except Exception as e:
                logging.getLogger("RotatingHandler").error(f"Compression error for {path}: {e}")

class ResourceLogger:
    def __init__(self, optimizer: SystemOptimizer):
        import psutil  # type: ignore

        self.optimizer = optimizer
        self._psutil = psutil
        self.cpu_history = deque(maxlen=60)  # 60 samples for 1-min window
        self.mem_history = deque(maxlen=60)
        self._gpu_initialized = False
        self._pynvml = None
        self.throughput_window = deque(maxlen=100)
        
    def _initialize_gpu(self):
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._gpu_initialized = True
        except Exception:
            self._pynvml = None

    def _get_gpu_usage(self):
        if not self._gpu_initialized:
            self._initialize_gpu()
        if not self._gpu_initialized or self._pynvml is None:
            return 0.0
            
        try:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(0)
            util = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except Exception:
            return 0.0
    
    def collect_metrics(self) -> dict:
        self.record_event()

        metrics = {
            'cpu': self._exp_smoothed_cpu(),
            'mem': self._psutil.virtual_memory().percent,
            'gpu': self._get_gpu_usage(),
            'throughput': self._calc_throughput(),
            'entropy': self._log_entropy()
        }
        return metrics

    def _exp_smoothed_cpu(self, alpha=0.7):
        # Exponential smoothing for noise reduction
        current = self._psutil.cpu_percent()
        if not self.cpu_history:
            return current
        return alpha * current + (1-alpha) * self.cpu_history[-1]

    def _calc_throughput(self) -> float:
        """
        Calculate average throughput (events per second) based on recorded (timestamp, count) entries.
        """
        if len(self.throughput_window) < 2:
            return 0.0
    
        timestamps, counts = zip(*self.throughput_window)
        duration = timestamps[-1] - timestamps[0]
    
        if duration == 0:
            return float('inf')
    
        return sum(counts) / duration

    def _log_entropy(self):
        # Calculate information entropy of recent logs
        log_contents = "\n".join(str(item) for item in list(get_log_queue().queue)[-100:])
        if not log_contents:
            return 0.0
        prob = {}
        for c in log_contents:
            prob[c] = prob.get(c, 0) + 1/len(log_contents)
        return -sum(p * math.log2(p) for p in prob.values() if p > 0)
    
    def record_event(self, count: int = 1):
        self.throughput_window.append((time.time(), count))

class AnomalyDetector:
    def __init__(self, window_size=100, sigma=3):
        self.error_counts = deque(maxlen=window_size)
        self.sigma = sigma
        self.mean = 0
        self.std = 0
        
    def analyze(self, record):
        if record.levelno >= logging.ERROR:
            self.error_counts.append(time.time())
            self._update_stats()
            
        return self._check_anomaly()

    def _update_stats(self):
        errors = list(self.error_counts)
        intervals = [t2 - t1 for t1, t2 in zip(errors, errors[1:])]
        if intervals:
            self.mean = statistics.mean(intervals)
            self.std = statistics.stdev(intervals) if len(intervals) > 1 else 0

    def _check_anomaly(self):
        if len(self.error_counts) < 2 or self.std == 0:
            return False
        latest_interval = self.error_counts[-1] - self.error_counts[-2]
        z_score = (latest_interval - self.mean) / self.std
        return abs(z_score) > self.sigma

class PrettyPrinter:
    @classmethod
    def pretty(cls, label: str, obj: Any, status: str = "info"):
        """Pretty-print structured objects (e.g., dicts, lists) in readable form"""
        formatted = pprint.pformat(obj, indent=2, width=100, compact=False)
        cls.status(label, "\n" + formatted, status)

    @classmethod
    def _style(cls, text, *styles):
        if not sys.stdout.isatty():
            return text
        codes = []
        for style in styles:
            if style in STYLES:
                codes.append(STYLES[style])
            elif style in COLOR_CODES:
                codes.append(COLOR_CODES[style])
        return f"{''.join(codes)}{text}{STYLES['reset']}"

    @classmethod
    def table(cls, headers, rows, title=None):
        # Create formatted table with borders
        col_width = [max(len(str(item)) for item in col) for col in zip(headers, *rows)]
        
        if title:
            total_width = sum(col_width) + 3*(len(headers)-1)
            print(cls._style(f"╒{'═'*(total_width)}╕", 'bold', 'blue'))
            print(cls._style(f"│ {title.center(total_width)} │", 'bold', 'blue'))
            print(cls._style(f"╞{'╪'.join('═'*w for w in col_width)}╡", 'bold', 'blue'))
        
        # Header
        header = cls._style("│ ", 'blue') + cls._style(" │ ", 'blue').join(
            cls._style(str(h).ljust(w), 'bold', 'white', 'bg_blue') 
            for h, w in zip(headers, col_width)
        ) + cls._style(" │", 'blue')
        print(header)
        
        # Separator
        print(cls._style(f"├{'┼'.join('─'*w for w in col_width)}┤", 'blue'))
        
        # Rows
        for row in rows:
            cells = []
            for item, w in zip(row, col_width):
                cell = cls._style(str(item).ljust(w), 'cyan')
                cells.append(cell)
            print(cls._style("│ ", 'blue') + cls._style(" │ ", 'blue').join(cells) + cls._style(" │", 'blue'))
        
        # Footer
        print(cls._style(f"╘{'╧'.join('═'*w for w in col_width)}╛", 'bold', 'blue'))

    @classmethod
    def _truncate_text(cls, text, max_length):
        """Truncate text with ellipsis if it exceeds max_length"""
        if len(text) <= max_length:
            return text
        # If we need to truncate, add an ellipsis
        if max_length > 3:
            return text[:max_length-3] + "..."
        return text[:max_length]

    @classmethod
    def section_header(cls, text):
        print("\n" + cls._style("╒═══════════════════════════════", 'bold', 'magenta'))
        print(cls._style(f" {text.upper()}", 'bold', 'magenta', 'italic'))
        print(cls._style("╘═══════════════════════════════", 'bold', 'magenta'))

    @classmethod
    def status(cls, label, message, status="info"):
        status_colors = {
            'info': ('blue', 'ℹ'),
            'success': ('green', '✔'),
            'warning': ('yellow', '⚠'),
            'error': ('red', '✖')
        }
        color, icon = status_colors.get(status, ('white', '○'))
        label_text = cls._style(f"[{label}]", 'bold', color)
        print(f"{cls._style(icon, color)} {label_text} {message}")

    @classmethod
    def code_block(cls, code, language="python"):
        print(cls._style(f"┏ {' ' + language + ' ':-^76} ┓", 'bold', 'white'))
        for line in code.split('\n'):
            print(cls._style("┃ ", 'white') + cls._style(f"{line:76}", 'cyan') + cls._style(" ┃", 'white'))
        print(cls._style(f"┗ {'':-^78} ┛", 'bold', 'white'))

    @classmethod
    def progress_bar(cls, current, total, label="Progress"):
        width = 50
        progress = current / total
        filled = int(width * progress)
        bar = cls._style("█" * filled, 'green') + cls._style("░" * (width - filled), 'dim')
        percent = cls._style(f"{progress:.0%}", 'bold', 'yellow')
        print(f"{label}: [{bar}] {percent} ({current}/{total})")
