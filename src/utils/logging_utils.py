import os
import logging
from logging import FileHandler
from typing import Optional

# # Base directory for storing logs (if not specified through environment variable, set it to `logs` dir under project root)
# LOG_DIR = os.getenv("LOG_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs"))
# # LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
# os.makedirs(LOG_DIR, exist_ok=True)
#
# # Logging level project-wide
# LOG_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with a specific name and optional file logging.

    Args:
        name (str): Logger name, typically the module's `__name__`.
        log_file (str): Log file name. If None, defaults to "<name>.log" under the logs directory.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)

    return logger

def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('locomo_eval')
    
    if logger.hasHandlers():
    
        if log_file:
            if not any(isinstance(h, FileHandler) and h.baseFilename == log_file for h in logger.handlers):
                try:
                    file_handler = logging.FileHandler(log_file)
                    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
                except Exception as e:
                    
                    print(f"Error setting up file handler for {log_file}: {e}")
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Error setting up file handler for {log_file}: {e}")

    logger.propagate = False
    return logger

