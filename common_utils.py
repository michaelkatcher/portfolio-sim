#!/usr/bin/env python3
"""
Common utilities for the MCA Portfolio Simulator.

This module provides common functionality like logging setup and custom exceptions.
"""

import logging
import os
from datetime import datetime


def setup_logging(name, level=logging.INFO, log_to_file=False, log_dir='logs'):
    """
    Set up logging for a module.
    
    Args:
        name: Name of the logger (typically __name__)
        level: Logging level
        log_to_file: Whether to log to a file in addition to console
        log_dir: Directory for log files if log_to_file is True
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Only add handlers if they don't exist yet
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add console handler to logger
        logger.addHandler(console_handler)
        
        # Add file handler if requested
        if log_to_file:
            # Create logs directory if it doesn't exist
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # Create file handler with timestamp in filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            
            # Add file handler to logger
            logger.addHandler(file_handler)
    
    return logger


class MCASimulatorError(Exception):
    """Base exception for all MCA Simulator errors."""
    pass


class DataError(MCASimulatorError):
    """Exception raised for errors in the data."""
    pass


class SimulationError(MCASimulatorError):
    """Exception raised for errors during simulation."""
    pass


class ExportError(MCASimulatorError):
    """Exception raised for errors during export."""
    pass


class ConfigurationError(MCASimulatorError):
    """Exception raised for configuration errors."""
    pass