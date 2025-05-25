import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")


# Configure logging
def setup_logging():
    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create handlers
    # File handler for all logs
    all_handler = logging.FileHandler(
        f'logs/all_{datetime.now().strftime("%Y%m%d")}.log', encoding="utf-8"
    )
    all_handler.setLevel(logging.INFO)
    all_handler.setFormatter(file_formatter)

    # File handler for errors
    error_handler = logging.FileHandler(
        f'logs/error_{datetime.now().strftime("%Y%m%d")}.log', encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Create logger
    logger = logging.getLogger("legal_chatbot")
    logger.setLevel(logging.INFO)
    logger.addHandler(all_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger
