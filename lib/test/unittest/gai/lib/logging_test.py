from gai.lib.logging import getLogger
import logging

if __name__ == "__main__":
    
    logger = getLogger("example_logger")
    logger.setLevel(logging.DEBUG)  # Set the log level to DEBUG
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")