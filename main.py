import os
import sys
import traceback
from pathlib import Path

from src.core.logging_utils import setup_logging
from src.gui import run_app

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.dirname(__file__))

# Add project root to Python path
sys.path.insert(0, project_root)

# Import after setting up the path


def excepthook(exc_type, exc_value, exc_traceback):
    """
    Global exception handler to log uncaught exceptions
    """
    logger = setup_logging()
    logger.critical(
        "Uncaught exception",
        exc_info=(
            exc_type,
            exc_value,
            exc_traceback))
    # Call the default exception handler
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


if __name__ == "__main__":
    # Set up global exception handler
    sys.excepthook = excepthook

    try:
        sys.exit(run_app())
    except Exception as e:
        logger = setup_logging()
        logger.critical(f"Fatal error: {str(e)}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
