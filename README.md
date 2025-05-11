# Play Whe Lottery Prediction System

This project implements a comprehensive data collection system and predictive model for the Play Whe lottery from Trinidad and Tobago. It includes web scraping capabilities, data analysis, and multiple prediction algorithms with confidence scores.

## Project Structure

```
play_whe_lottery/
├── data/                  # Raw and processed data files
├── scripts/               # Python scripts for data collection and analysis
│   ├── scraper.py         # Web scraper for historical Play Whe data
│   ├── merge_data.py      # Data cleaning and preparation
│   ├── data_analysis.py   # Statistical analysis of lottery data
│   ├── prediction_models.py # Prediction algorithms implementation
├── analysis/              # Analysis results and visualizations
├── models/                # Prediction models and evaluation results
└── README.md              # Project documentation
```

## Key Features

1. **Data Collection System**
   - Web scraper for Play Whe lottery results from nlcbgames.com
   - Data organized in structured CSV format with draw date, time, winning number, and day of week
   - Handles error recovery and rate limiting

2. **Data Analysis**
   - Frequency analysis of each number (1-36)
   - Time-based analysis (by draw time, day of week)
   - Sequential pattern analysis
   - Hot/cold number identification
   - Statistical significance testing

3. **Prediction Models**
   - Frequency-based model: Predicts based on historical frequency
   - Sequential pattern model: Predicts based on transitions from previous draws
   - Hot/cold number model: Predicts based on recent frequency patterns
   - Hybrid model: Combines multiple approaches for improved accuracy

4. **Confidence Scores**
   - Each prediction includes a confidence score
   - Scores based on statistical deviation from expected probability
   - Helps guide wagering decisions

## Installation and Setup

1. Clone the repository:
```
git clone https://github.com/yourusername/play_whe_lottery.git
cd play_whe_lottery
```

2. Install required dependencies:
```
pip install requests beautifulsoup4 pandas numpy matplotlib seaborn scikit-learn statsmodels
```

## Usage

### Data Collection

To collect historical Play Whe data:
```
python scripts/scraper.py
```

### Data Preparation

To merge and clean the collected data:
```
python scripts/merge_data.py
```

### Data Analysis

To perform comprehensive analysis on the data:
```
python scripts/data_analysis.py
```

### Prediction Models

To build and evaluate prediction models:
```
python scripts/prediction_models.py
```

## Results

The system was evaluated using historical data from September 2016 to March 2025. Key findings:

- The frequency-based model performed best with 14.93% accuracy (7.5% improvement over random guessing)
- Sequential and hot/cold models showed patterns but did not consistently outperform random chance
- The hybrid model provides the most robust approach for long-term use

## Reports and Visualizations

- Detailed analysis report: `analysis/analysis_report.md`
- Prediction models report: `models/prediction_report.md`
- Various visualizations in the `analysis/` and `models/` directories

## Limitations and Future Work

1. **Limitations**
   - Past patterns may not continue in the future
   - The lottery system may have changed over time
   - Some apparent patterns may be due to random chance

2. **Future Improvements**
   - Incorporate additional features such as lunar cycles, holidays, or special events
   - Implement machine learning models with more sophisticated pattern recognition
   - Develop a real-time updating system that adjusts predictions after each draw
   - Explore potential cyclical patterns over longer time periods

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- National Lotteries Control Board (NLCB) of Trinidad and Tobago for the Play Whe lottery data
- Various statistical and data science libraries used in this project
