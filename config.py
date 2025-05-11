#!/usr/bin/env python3
"""
Configuration settings for the Play Whe prediction system
"""

import os
import logging

# Base directory is the parent directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directory
DATA_DIR = os.path.join(BASE_DIR, "data")

# Models directory
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Analysis directory
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")

# Logs directory
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Predictions directory
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, ANALYSIS_DIR, LOGS_DIR, PREDICTIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

# File paths
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "play_whe_processed.csv")
PRODUCTION_DATA_FILE = os.path.join(DATA_DIR, "play_whe_production.csv")
SAMPLE_DATA_FILE = os.path.join(DATA_DIR, "sample_play_whe_results.csv")
EXTENDED_SAMPLE_DATA_FILE = os.path.join(DATA_DIR, "extended_sample_play_whe_results.csv")
CULTURAL_EVENTS_FILE = os.path.join(DATA_DIR, "cultural_events.csv")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "system.log")),
        logging.StreamHandler()
    ]
)

# Scraper settings
SCRAPER_DELAY_MIN = 1  # Minimum delay between requests in seconds
SCRAPER_DELAY_MAX = 3  # Maximum delay between requests in seconds
SCRAPER_BASE_URL = "https://www.nlcbplaywhelotto.com/nlcb-play-whe-results/"

# Prediction model settings
DEFAULT_TOP_N = 5  # Default number of top predictions to return

def get_logger(name):
    """
    Get a logger with the specified name
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(name)
    # Add a file handler specific to this module
    handler = logging.FileHandler(os.path.join(LOGS_DIR, f"{name}.log"))
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger

def get_data_file_path(filename):
    """
    Get full path for a file in the data directory
    
    Args:
        filename (str): Name of the file
        
    Returns:
        str: Full path to the file
    """
    return os.path.join(DATA_DIR, filename)

def get_output_file_path(directory, filename):
    """
    Get full path for an output file
    
    Args:
        directory (str): Directory name (e.g., "models", "analysis")
        filename (str): Name of the file
        
    Returns:
        str: Full path to the file
    """
    dir_mapping = {
        "data": DATA_DIR,
        "models": MODELS_DIR,
        "analysis": ANALYSIS_DIR,
        "logs": LOGS_DIR,
        "predictions": PREDICTIONS_DIR
    }
    
    if directory in dir_mapping:
        return os.path.join(dir_mapping[directory], filename)
    else:
        # Default to data directory
        return os.path.join(DATA_DIR, filename)