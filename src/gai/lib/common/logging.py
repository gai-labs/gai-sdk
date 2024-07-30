import logging
import os
from colorlog import ColoredFormatter

def getLogger(name):

    # Create a logger object
    logger = logging.getLogger(name)

    # Remove all handlers associated with the logger
    while logger.handlers:
        logger.handlers.pop()

    # Set a new log level
    logger.setLevel(os.environ.get("LOG_LEVEL", "WARNING"))

    # Create a console handler
    console_handler = logging.StreamHandler()

    # Set the format for the console handler using ColoredFormatter
    formatter = ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(message_log_color)s%(message)s%(reset)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'bg_purple,fg_black',
            'INFO':     'bg_green,fg_black',
            'WARNING':  'bg_yellow,fg_black',
            'ERROR':    'bg_red,fg_black',
            'CRITICAL': 'bg_red,fg_white',
        },
        secondary_log_colors={
            'message': {
                'DEBUG':    'fg_purple',
                'INFO':     'fg_green',
                'WARNING':  'fg_yellow',
                'ERROR':    'fg_red',
                'CRITICAL': 'fg_red',
            }
        },
        style='%'
    )

    # Add formatter to the console handler
    console_handler.setFormatter(formatter)

    # Add console handler to the logger
    logger.addHandler(console_handler)

    return logger