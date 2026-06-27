import logging
import pytz
import sys
from datetime import datetime
import os
from pathlib import Path
import contextvars



BASE_DIR = Path(__file__).resolve().parent
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# ── ANSI color codes for terminal output ─────────────────────────────────────
COLORS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[1;41m", # Bold white on red background
}
RESET = "\033[0m"

# Enable ANSI color support on Windows
if sys.platform == "win32":
    os.system("")  # triggers Windows VT100 mode


class ISTFormatter(logging.Formatter):

    def formatTime(self, record, datefmt=None):
        utc_time = datetime.utcnow().replace(tzinfo=pytz.utc)
        ist_time = utc_time.astimezone(pytz.timezone("Asia/Kolkata"))
        return ist_time.strftime("%Y-%m-%d %H:%M:%S")


class ColoredISTFormatter(ISTFormatter):
    """Formatter that adds ANSI colors to log level for terminal output."""

    def format(self, record):
        level = record.levelname
        color = COLORS.get(level, "")
        record.levelname = f"{color}{level:<8}{RESET}"
        result = super().format(record)
        record.levelname = level  # restore for other handlers
        return result


class PrintLogger:

    def __init__(self, logger):
        self.logger = logger

    def write(self, message):
        if message != "\n":
            self.logger.info(message.strip())

    def flush(self):
        pass

    def isatty(self):
        return False


# ── Configure root logger (colors ALL loggers: uvicorn, httpx, boto3, etc.) ──
_console_handler = logging.StreamHandler(sys.stderr)
_console_handler.setLevel(logging.DEBUG)
_console_formatter = ColoredISTFormatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
_console_handler.setFormatter(_console_formatter)

logging.root.setLevel(logging.INFO)
logging.root.addHandler(_console_handler)

# Quiet down noisy third-party loggers
for _noisy in ("httpcore", "httpx", "urllib3", "botocore", "boto3", "s3transfer"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

def set_request_id(request_id: str):
    _request_id.set(request_id) 

# ── App logger ───────────────────────────────────────────────────────────────
logger = logging.getLogger("Adv-RAG")
logger.setLevel(logging.DEBUG)

# File handler (no colors — plain text for log files)
os.makedirs(BASE_DIR / "logs", exist_ok=True)
_file_handler = logging.FileHandler(
    str(BASE_DIR / "logs" / "app.log"),
    encoding="utf-8"
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(ISTFormatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_file_handler)


# Redirect print()
sys.stdout = PrintLogger(logger)
# sys.stderr = PrintLogger(logger)
#ended here

# Prevent duplicate console output (root logger already has the console handler)
#logger.propagate = True
logger.propagate = False
#changed for printing all logs only in log file


def log_user_action(message: str = "No Message given", level: str = "info"):
    req_id = _request_id.get()
    # Log to global app logger

    getattr(logger, level)(message)


def setup_logger(name, level=logging.INFO):
    """
    Set up and return a logger with file handler that includes userid in the format

    Args:
        name: Name of the logger
        level: Logging level (default: INFO)

    Returns:
        A configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)

    # Create log directory if it doesn't exist
    log_dir = BASE_DIR / "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create handler for this logger
    file_path = log_dir / "app.log"
    file_handler = logging.FileHandler(file_path, mode="a")

    # Set formatter with userid
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(userid)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    # Set level
    logger.setLevel(level)

    return logger