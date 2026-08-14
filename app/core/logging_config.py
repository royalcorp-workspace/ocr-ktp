import os
import logging
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar

# ContextVar untuk tracking request_id pada eksekusi async
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIDFilter(logging.Filter):
    """
    Filter logging untuk menyisipkan request_id dari ContextVar secara dinamis
    ke dalam setiap LogRecord.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logger(name: str = "ocr_app", log_filename: str = "app.log") -> logging.Logger:
    """
    Inisialisasi reusable logging configuration.
    Mendukung console output dan UTF-8 rotating file logging di logs/<log_filename>.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_filename)

    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logger_inst = logging.getLogger(name)
    logger_inst.setLevel(log_level)

    if not logger_inst.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [req_id=%(request_id)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        req_filter = RequestIDFilter()

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(req_filter)

        # Rotating File Handler (Max 10 MB, 5 backup files, UTF-8 Encoding)
        file_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(req_filter)

        logger_inst.addHandler(console_handler)
        logger_inst.addHandler(file_handler)

    return logger_inst


# Export logger instance & request context variable
logger = setup_logger()
ktp_logger = setup_logger(name="ktp_app", log_filename="ktp.log")