# Installation and Setup Guide

This guide provides instructions for setting up the Play Whe Prediction System.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for version control)

## Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/play-whe-prediction.git
   cd play-whe-prediction
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Initial Setup

1. Create necessary directories if they don't exist:
   ```
   mkdir -p data models analysis predictions
   ```

2. Ensure sample data files are present:
   - `data/sample_play_whe_results.csv` - Sample lottery results
   - `data/cultural_events.csv` - Calendar of cultural events for analysis

## Running the System

1. Process and prepare the data:
   ```
   python merge_data.py
   ```

2. Build prediction models:
   ```
   python prediction_models.py
   ```

3. Make predictions:
   ```
   python make_prediction.py --period morning --top 5
   ```

4. Update models with actual results:
   ```
   python update_models.py --number 14 --analyze
   ```

## Troubleshooting

- If you encounter issues with web scraping modules, ensure you have the appropriate web drivers installed for Selenium.
- For data processing errors, check the log files in the `data/` directory.
- Model-related errors are typically logged in the `models/` directory.

## Advanced Configuration

The system can be customized by modifying the configuration parameters in the respective Python files:

- Scraper settings in `scraper.py`
- Data processing parameters in `merge_data.py`
- Model weights in `prediction_models.py`
- Cultural patterns in `cultural_patterns.py`