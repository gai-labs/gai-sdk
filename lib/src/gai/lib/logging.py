import os
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from rich.style import Style
from rich.text import Text

def getLogger(name):
    # Create a logger object
    logger = logging.getLogger(name)

    # Remove all handlers associated with the logger
    while logger.handlers:
        logger.handlers.pop()

    # Set a new log level
    logger.setLevel(os.environ.get("LOG_LEVEL", "WARNING"))

    # Create a custom theme to match the original color scheme
    custom_theme = Theme({
        "logging.level.debug": Style(color="black", bgcolor="purple"),
        "logging.level.info": Style(color="black", bgcolor="green"),
        "logging.level.warning": Style(color="black", bgcolor="yellow"),
        "logging.level.error": Style(color="#FFFFFF", bgcolor="red"),
        "logging.level.critical": Style(color="#FFFFFF", bgcolor="red", bold=True),
        
        "logging.text.debug":    Style(color="purple"),
        "logging.text.info":     Style(color="green"),
        "logging.text.warning": Style(color="yellow"),
        "logging.text.error": Style(color="bright_red"),
        "logging.text.critical": Style(color="red"),
    })

    # Create console with custom theme
    console = Console(theme=custom_theme, color_system="truecolor")
    
    # subclass RichHandler
    class LevelAwareRichHandler(RichHandler):
        def render_message(self, record: logging.LogRecord, message: str):
            # RichHandler.render_message(...) returns a rich.text.Text
            text = super().render_message(record, message)
            lvl = record.levelname.lower()  # e.g. "info", "error"
            # apply your per‐level message style
            text.stylize(f"logging.text.{lvl}")
            return text   
    
    # Create a custom Rich handler with our styling
    rich_handler = LevelAwareRichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=False,
        show_path=False,
        markup=False,
        log_time_format="",
        enable_link_path=False,
    )

    # Configure the formatter to be minimal since Rich handles formatting
    formatter = logging.Formatter("%(message)s")
    rich_handler.setFormatter(formatter)

    # Add Rich handler to the logger
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(rich_handler)

    return logger 