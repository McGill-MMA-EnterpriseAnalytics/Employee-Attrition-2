"""
Logging utility for the project.
"""
import logging
import sys

def setup_logger(name, level=logging.INFO):
    """Set up a logger with the given name and level.
    
    Args:
        name: Name of the logger
        level: Logging level (default: INFO)
        
    Returns:
        Logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger 