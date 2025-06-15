from __future__ import annotations
import logging
import os, sys
import time
import math
import queue
import psutil
import hashlib
import zlib
import statistics
import atexit
import shutil
import pprint
import msvcrt
import uuid

from datetime import datetime
from logging.handlers import RotatingFileHandler
from collections import deque
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from src.utils.system_optimizer import SystemOptimizer

if os.name == 'nt':
    os.system("")

sys.stdout.isatty = lambda: True

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    os.chmod(log_dir, 0o755)  # Read/write for owner, read for others

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

# Global flag to track initialization
_logger_initialized = False

class ColorFormatter(logging.Formatter):

    def format(self, record):
        if not sys.stdout.isatty():
            return super().format(record)
        level = record.levelname
        message = record.getMessage()

        if "initializ" in message.lower():
            color = COLOR_CODES['BLUE']
        elif "load" in message.lower() and record.levelno < logging.WARNING:
            color = COLOR_CODES['GREEN']
        elif record.levelno >= logging.CRITICAL:
            color = COLOR_CODES['RED']
        elif record.levelno >= logging.WARNING:
            color = COLOR_CODES['YELLOW']
        elif record.levelno >= logging.ERROR:
            color = STYLES['Orange1']
        else:
            color = COLOR_CODES['RESET']

        formatted = f"{record.levelname}:{record.name}:{message}"
        return f"{color}{formatted}{COLOR_CODES['RESET']}"

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

def get_logger(name: str) -> logging.Logger:
    global _logger_initialized, log_queue
    logger = logging.getLogger(name)
    
    if not _logger_initialized:
        _logger_initialized = True
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')

        # Initialize root logger first
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Remove any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # File handler
        file_handler = RotatingHandler(
            'logs/app.log', 
            maxBytes=1000000, 
            backupCount=5, 
            delay=True  # Defer file opening until first log
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
        root_logger.addHandler(console_handler)

        # Queue handler
        handler = QueueLogHandler(log_queue, batch_size=10, flush_interval=5)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    return logger

def get_log_queue():
    return log_queue

def cleanup_logger(name):
    """
    Clean up and close all handlers for a given logger.
    Useful before rollback or app shutdown to release file locks.
    """
    logger = logging.getLogger(name)
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

def exit_handler():
    cleanup_logger(None)  # Cleanup root logger

atexit.register(exit_handler)

class RotatingHandler(RotatingFileHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compress_queue = deque(maxlen=5)
        self.compress_threshold = 5  # Number of backups before compression
        self.last_rollover_time = 0
        self.rollover_cooldown = 60  # 60 seconds cooldown between rollover attempts

    def doRollover(self):
        current_time = time.time()
        if current_time - self.last_rollover_time < self.rollover_cooldown:
            return  # Skip if we tried recently
    
        self.last_rollover_time = current_time
    
        # Flush and close all handlers to release file locks
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            try:
                handler.flush()
                handler.close()
            except Exception:
                pass
    
        self.stream = None
    
        #dfn = self.rotation_filename(self.baseFilename + ".1")
        dfn = self.rotation_filename(self.baseFilename + f".{int(time.time())}_{uuid.uuid4().hex[:6]}")
    
        try:
            if os.path.exists(dfn):
                try:
                    with open(dfn, 'a'):
                        pass
                    os.remove(dfn)
                except (PermissionError, IOError):
                    logging.debug(f"Backup file {dfn} is locked, skipping removal")
    
            if os.path.exists(self.baseFilename):
                try:
                    with open(self.baseFilename, 'r+b') as src:
                        msvcrt.locking(src.fileno(), msvcrt.LK_LOCK, 1)
                        with open(dfn, 'wb') as dst:
                            shutil.copyfileobj(src, dst)
                        msvcrt.locking(src.fileno(), msvcrt.LK_UNLCK, 1)
                    os.remove(self.baseFilename)
                except Exception as e:
                    logging.warning(f"Log rotation fallback failed: {str(e)}")
                    return

                else:
                    logging.warning(f"Log file {self.baseFilename} is locked, skipping rotation")
                    return
    
        except Exception as e:
            logging.error(f"Log rotation error: {str(e)}")
            return
    
        if not self.delay:
            try:
                self.stream = self._open()
            except Exception:
                self.stream = None
    
        self._compress_queue.append(dfn)
        self._manage_compression()

    def _manage_compression(self):
        while len(self._compress_queue) > self.compress_threshold:
            old_log = self._compress_queue.popleft()
            try:
                # Read and compress with context managers
                with open(old_log, 'rb') as f:
                    data = zlib.compress(f.read(), level=9)
                with open(old_log + '.z', 'wb') as f:
                    f.write(data)
                os.remove(old_log)
            except PermissionError as e:
                logging.error(f"Failed to compress {old_log}: {e}")

class ResourceLogger:
    def __init__(self, optimizer: SystemOptimizer):
        self.optimizer = optimizer
        self.cpu_history = deque(maxlen=60)  # 60 samples for 1-min window
        self.mem_history = deque(maxlen=60)
        self._gpu_initialized = False
        
    def _initialize_gpu(self):
        try:
            pynvml.nvmlInit()
            self._gpu_initialized = True
        except:
            pass

    def collect_metrics(self) -> dict:
        metrics = {
            'cpu': self._exp_smoothed_cpu(),
            'mem': psutil.virtual_memory().percent,
            'gpu': self._get_gpu_usage(),
            'throughput': self._calc_throughput(),
            'entropy': self._log_entropy()
        }
        return metrics

    def _exp_smoothed_cpu(self, alpha=0.7):
        # Exponential smoothing for noise reduction
        current = psutil.cpu_percent()
        if not self.cpu_history:
            return current
        return alpha * current + (1-alpha) * self.cpu_history[-1]

    def _get_gpu_usage(self):
        if not self._gpu_initialized:
            self._initialize_gpu()
            return 0.0
            
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            return 0.0

    def _log_entropy(self):
        # Calculate information entropy of recent logs
        log_contents = "\n".join(list(get_log_queue().queue)[-100:])
        prob = {}
        for c in log_contents:
            prob[c] = prob.get(c, 0) + 1/len(log_contents)
        return -sum(p * math.log2(p) for p in prob.values() if p > 0)

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

USE_ANSI = sys.stdout.isatty()
class PrettyPrinter:
    @classmethod
    def pretty(cls, label: str, obj: Any, status: str = "info"):
        """Pretty-print structured objects (e.g., dicts, lists) in readable form"""
        formatted = pprint.pformat(obj, indent=2, width=100, compact=False)
        cls.status(label, "\n" + formatted, status)

    @classmethod
    def _style(cls, text, *styles):
        if not USE_ANSI:
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
