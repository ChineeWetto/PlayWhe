# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a prediction system for the Play Whe lottery from Trinidad and Tobago. The system uses statistical analysis, machine learning, and cultural pattern recognition to predict lottery outcomes.

## Code Architecture

The system is structured around several key components:

1. **Data Collection**
   - Multiple scrapers (`scraper.py`, `nlcb_scraper.py`, `selenium_scraper.py`, `requests_scraper.py`) collect historical lottery data
   - Each implements different approaches for reliability and redundancy

2. **Data Processing**
   - `merge_data.py` cleans and processes raw data
   - Handles schedule changes, normalizes time formats, and adds derived features

3. **Prediction Models**
   - `prediction_models.py` implements multiple prediction algorithms
   - Models include frequency-based, sequential pattern, hot/cold number, and hybrid approaches

4. **Enhanced Prediction System**
   - `enhanced_prediction_system.py` integrates all components
   - Advanced models in specialized files (`adaptive_frequency_model.py`, `advanced_sequential_model.py`)
   - Cultural analysis in `cultural_patterns.py` and `enhanced_cultural_patterns.py`
   - Self-learning mechanism in `self_learning_mechanism.py`

5. **User Interface**
   - Command-line interfaces in `make_prediction.py` and `update_models.py`

## Common Commands

### Running the Prediction System

```bash
# Process data and build models
python merge_data.py
python prediction_models.py

# Run the enhanced prediction system
python enhanced_prediction_system.py

# Make predictions with the command-line interface
python make_prediction.py --period morning --top 5

# Update models with actual results
python update_models.py --number 14 --analyze
```

### Testing and Development

```bash
# Test enhanced models
python test_enhanced_models.py

# Test with sample data
python test_with_sample_data.py

# Run a specific analysis
python cultural_patterns.py
python data_analysis.py
```

### Updating the Cultural Events Calendar

The cultural events calendar in `data/cultural_events.csv` should be periodically updated with new events to keep the cultural pattern analysis current.