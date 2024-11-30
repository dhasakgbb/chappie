# src/logger.py

"""
logger.py

Configures centralized logging for the application.
Ensures consistent logging across all modules with appropriate log levels and formats.
"""

import logging
import os

def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Creates and returns a configured logger.

    :param name: Name of the logger.
    :param log_file: File path for logging output. If None, logs to console.
    :return: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Define the formatter for the file handler
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Define the formatter for the console handler
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')

    # Check if the logger already has handlers to avoid adding duplicates
    if not logger.handlers:
        # Create and configure the file handler if a log file is specified
        if log_file:
            # Ensure the log file is stored in the logs directory at the same level as data/ and src/
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            log_dir = os.path.join(project_root, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, log_file)
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        # Create and configure the console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger