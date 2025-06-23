"""
Example module demonstrating how to use the logging system.
This file is for demonstration purposes only and can be deleted.
"""

from src.core import get_logger

# Get a logger for this module
logger = get_logger(__name__)


def example_function(param):
    """Example function that logs its activity."""
    logger.debug(f"example_function called with param: {param}")

    try:
        # Some operation that might fail
        result = 100 / param
        logger.info(f"Calculation successful, result: {result}")
        return result
    except Exception as e:
        # Log the error
        logger.error(f"Error in calculation: {str(e)}")
        logger.exception("Exception details:")
        raise


def main():
    """Example main function demonstrating different log levels."""
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    try:
        # Call function with valid parameter
        example_function(10)

        # Call function with parameter that will cause an error
        example_function(0)
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}")


if __name__ == "__main__":
    # This will run if this file is executed directly
    main()
