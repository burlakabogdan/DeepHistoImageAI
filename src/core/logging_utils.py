import datetime
import logging
import os
from pathlib import Path


def setup_logging(log_level=logging.INFO):
    """
    Set up logging configuration to write logs to the 'logs' folder.

    Args:
        log_level: The logging level (default: logging.INFO)

    Returns:
        The configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = (
        Path(
            os.path.abspath(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(__file__))))) /
        "logs")
    logs_dir.mkdir(exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"deep_image_{timestamp}.log"

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def get_logger(name):
    """
    Get a named logger.

    Args:
        name: The name for the logger, typically the module name

    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)
