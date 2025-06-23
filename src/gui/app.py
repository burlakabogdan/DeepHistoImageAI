import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from src.core.database import init_database
from src.core.logging_utils import setup_logging


def run_app():
    """Initialize and run the application."""
    # Set up logging
    logger = setup_logging()
    logger.info("Application starting")

    app = QApplication(sys.argv)

    # Set application-wide attributes
    app.setStyle("Fusion")  # Modern cross-platform style
    app.setApplicationName("Deep Histo Image AI")
    app.setApplicationVersion("0.0.1")
    logger.info("Application configured")

    # Initialize database
    db_session = init_database()
    logger.info("Database initialized")

    # Import MainWindow only when needed to avoid circular imports
    from src.gui.main_window import MainWindow

    # Create and show main window
    main_window = MainWindow(db_session)
    main_window.show()
    logger.info("Main window displayed")

    return app.exec()
