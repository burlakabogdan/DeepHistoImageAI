import os
from pathlib import Path
import sqlite3

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.exc import OperationalError

from src.core.models import Base
from src.core import get_logger

# Create session factory class
SessionLocal = sessionmaker(autocommit=False, autoflush=False)
logger = get_logger(__name__)


def init_database(app_data_dir=None):
    """
    Initialize the database and return a session factory.

    Args:
        app_data_dir: Optional directory to store the database file.
                     If None, uses the 'data' directory in the project root.

    Returns:
        A scoped session factory that can be used to create database sessions.

    Raises:
        PermissionError: If database directory is not writable
        OperationalError: If database cannot be initialized
    """
    try:
        if app_data_dir is None:
            # Use a 'data' directory in the project root
            app_data_dir = Path(
                os.path.abspath(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(__file__)))))
            app_data_dir = app_data_dir / "data"

        # Ensure the directory exists and is writable
        app_data_dir = Path(app_data_dir)
        if not app_data_dir.exists():
            try:
                app_data_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created database directory: {app_data_dir}")
            except Exception as e:
                logger.error(f"Failed to create database directory: {str(e)}")
                raise

        # Validate directory permissions
        if not os.access(app_data_dir, os.W_OK):
            error_msg = f"Database directory {app_data_dir} must be writable"
            logger.error(error_msg)
            raise PermissionError(error_msg)

        # Database file path
        db_path = app_data_dir / "deep_image.db"
        
        # Test database connection and creation
        try:
            # Try creating a test connection to validate SQLite access
            test_conn = sqlite3.connect(db_path)
            test_conn.close()
            logger.info("Successfully validated database access")
        except sqlite3.Error as e:
            logger.error(f"SQLite database access error: {str(e)}")
            raise OperationalError(f"Database access error: {str(e)}")

        # Create engine with proper error handling
        try:
            engine = create_engine(f"sqlite:///{db_path}")
            Base.metadata.create_all(bind=engine)
            logger.info("Successfully created database tables")
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise

        # Create session factory
        session_factory = scoped_session(
            sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=engine))

        logger.info("Database initialization completed successfully")
        return session_factory

    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise
