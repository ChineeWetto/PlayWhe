#!/usr/bin/env python3
"""
Test script for enhanced Play Whe prediction models using sample data
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import shutil
from datetime import datetime
from merge_data import PlayWheDataMerger
from prediction_models import PlayWhePredictionModels
from cultural_patterns import CulturalPatternAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/test_sample_data.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("test_with_sample_data")

class SampleDataProcessor:
    """
    A class to process sample data for testing
    """
    
    def __init__(self, sample_results_file="data/sample_play_whe_results.csv", output_file="data/play_whe_processed.csv"):
        """
        Initialize the processor with configuration parameters
        
        Args:
            sample_results_file (str): Path to the sample results file
            output_file (str): Path to save the processed data
        """
        self.sample_results_file = sample_results_file
        self.output_file = output_file
        
    def process_sample_data(self):
        """
        Process the sample data to create a processed dataset
        
        Returns:
            pandas.DataFrame: Processed DataFrame
        """
        logger.info(f"Processing sample data from {self.sample_results_file}")
        
        try:
            # Read sample data
            df = pd.read_csv(self.sample_results_file)
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Add schedule change flag (transition from 3 to 4 draws per day)
            # For sample data, we'll pretend the change happened on 2024-01-15
            schedule_change_date = pd.to_datetime('2024-01-15')
            df['post_schedule_change'] = df['date'] > schedule_change_date
            
            # Normalize time format and create draw period
            df['time'] = df['time'].apply(self._normalize_time_format)
            df['draw_period'] = df['time'].apply(self._normalize_draw_period)
            
            # Add additional derived columns
            df['month'] = df['date'].dt.month_name()
            df['year'] = df['date'].dt.year
            df['day_of_month'] = df['date'].dt.day
            df['week_of_year'] = df['date'].dt.isocalendar().week
            
            # Add sequential features
            df = self._add_sequential_features(df)
            
            # Sort by date and time
            df = df.sort_values(by=['date', 'time'])
            
            # Reset index
            df = df.reset_index(drop=True)
            
            # Save processed data
            df.to_csv(self.output_file, index=False)
            logger.info(f"Saved processed sample data to {self.output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing sample data: {e}")
            return None
    
    def _normalize_time_format(self, time_str):
        """
        Normalize time formats to a standard format
        
        Args:
            time_str (str): Time string to normalize
            
        Returns:
            str: Normalized time string
        """
        if pd.isna(time_str):
            return ""
            
        time_str = str(time_str).upper().strip()
        
        # For sample data, time is already in a consistent format
        return time_str
        
    def _normalize_draw_period(self, time_str):
        """
        Map draw times to consistent periods
        
        Args:
            time_str (str): Time string
            
        Returns:
            str: Normalized draw period
        """
        if pd.isna(time_str):
            return 'unknown'
            
        time_str = str(time_str).upper()
        
        if '10:30AM' in time_str:
            return 'morning'
        elif '1:00PM' in time_str:
            return 'midday'
        elif '4:00PM' in time_str:
            return 'afternoon'
        elif '7:00PM' in time_str:
            return 'evening'
        else:
            return 'unknown'
    
    def _add_sequential_features(self, df):
        """
        Add sequential features to the data
        
        Args:
            df (pandas.DataFrame): DataFrame to add features to
            
        Returns:
            pandas.DataFrame: DataFrame with added features
        """
        # Sort by date and time to ensure proper sequence
        df = df.sort_values(by=['date', 'time'])
        
        # Add previous draw numbers
        for i in range(1, 6):
            df[f'prev_{i}_number'] = df['number'].shift(i)
            
        # Add next draw number (for training only)
        df['next_number'] = df['number'].shift(-1)
        
        # Calculate days since last occurrence of each number
        number_indices = {}
        days_since_last = np.zeros(len(df))
        
        for idx, row in df.iterrows():
            num = row['number']
            if num in number_indices:
                days_since_last[idx] = (row['date'] - df.loc[number_indices[num], 'date']).days
            number_indices[num] = idx
            
        df['days_since_last'] = days_since_last
        
        # Calculate rolling frequency of each number
        window_sizes = [10, 30, 50, 100]
        for window in window_sizes:
            df[f'freq_last_{window}'] = df.groupby('number')['number'].transform(
                lambda x: x.rolling(window, min_periods=1).count()
            )
            
        return df

def main():
    """
    Main function to test enhanced prediction models with sample data
    """
    # Create output directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("analysis", exist_ok=True)

    # Check if extended sample data exists, if not, generate it
    extended_sample_file = "data/extended_sample_play_whe_results.csv"
    if not os.path.exists(extended_sample_file):
        logger.info("Extended sample data not found. Generating it...")
        try:
            from data.create_sample_data import create_sample_data
            create_sample_data(num_days=90, output_path=extended_sample_file)
        except Exception as e:
            logger.error(f"Failed to generate extended sample data: {e}")
            logger.info("Falling back to default sample data...")

    # Determine which sample file to use
    sample_file = extended_sample_file if os.path.exists(extended_sample_file) else "data/sample_play_whe_results.csv"
    logger.info(f"Using sample data from: {sample_file}")

    # Step 1: Process sample data
    logger.info("Step 1: Processing sample data...")
    sample_processor = SampleDataProcessor(sample_results_file=sample_file)
    processed_df = sample_processor.process_sample_data()
    
    if processed_df is None or processed_df.empty:
        logger.error("Failed to process sample data. Exiting.")
        return
    
    # Step 2: Build enhanced prediction models
    logger.info("Step 2: Building enhanced prediction models...")
    prediction_models = PlayWhePredictionModels(data_file="data/play_whe_processed.csv")
    models = prediction_models.build_all_models()
    
    if models is None:
        logger.error("Failed to build prediction models. Exiting.")
        return
    
    # Step 3: Analyze cultural patterns
    logger.info("Step 3: Analyzing cultural patterns...")
    cultural_analyzer = CulturalPatternAnalyzer(
        data_file="data/play_whe_processed.csv",
        events_file="data/cultural_events.csv"
    )
    cultural_results = cultural_analyzer.run_all_analyses()
    
    if cultural_results is None:
        logger.warning("Cultural pattern analysis failed or returned no results.")
    
    # Step 4: Test predictions
    logger.info("Step 4: Testing predictions...")
    
    # Get the most recent draw
    recent_draw = processed_df.iloc[-1]
    prev_num = recent_draw['number']
    draw_period = recent_draw['draw_period']
    
    logger.info(f"Most recent draw: Number {prev_num} in {draw_period} period")
    
    # Get predictions from each model
    predictions = {}
    
    # Standard frequency model
    freq_predictions = prediction_models._predict_frequency(n=5)
    predictions['Frequency'] = freq_predictions
    
    # Time-sensitive frequency model
    try:
        time_freq_predictions = prediction_models._predict_time_sensitive_frequency(period=draw_period, n=5)
        predictions['Time-Sensitive Frequency'] = time_freq_predictions
    except Exception as e:
        logger.error(f"Error getting time-sensitive frequency predictions: {e}")
    
    # Sequential model
    seq_predictions = prediction_models._predict_sequential(prev_num=prev_num, n=5)
    predictions['Sequential'] = seq_predictions
    
    # Hot/cold model
    hot_predictions = prediction_models._predict_hot_cold(n=5, strategy='hot')
    predictions['Hot'] = hot_predictions
    
    # Original hybrid model
    try:
        hybrid_predictions = prediction_models._predict_hybrid(prev_num=prev_num, n=5)
        predictions['Hybrid'] = hybrid_predictions
    except Exception as e:
        logger.error(f"Error getting hybrid predictions: {e}")
    
    # Optimized hybrid model
    try:
        opt_hybrid_predictions = prediction_models._predict_optimized_hybrid(prev_num=prev_num, period=draw_period, n=5)
        predictions['Optimized Hybrid'] = opt_hybrid_predictions
    except Exception as e:
        logger.error(f"Error getting optimized hybrid predictions: {e}")
    
    # Print predictions
    print("\n=== PLAY WHE PREDICTIONS ===")
    print(f"Previous number: {prev_num}")
    print(f"Draw period: {draw_period}")
    print("===========================")
    
    for model_name, model_predictions in predictions.items():
        print(f"\n{model_name} Model Predictions:")
        print("-----------------------------")
        for i, (num, prob, conf) in enumerate(model_predictions, 1):
            print(f"{i}. Number {num}: Probability {prob:.6f}, Confidence {conf:.2f}%")
    
    # Step 5: Generate comparison visualization
    logger.info("Step 5: Generating comparison visualization...")
    
    # Prepare data for visualization
    model_names = list(predictions.keys())
    top_predictions = {}
    
    for model_name, model_predictions in predictions.items():
        if model_predictions:
            top_predictions[model_name] = model_predictions[0][0]  # Get top number
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    
    # Plot top predictions
    bars = plt.bar(model_names, [top_predictions.get(model, 0) for model in model_names])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{int(height)}', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Top Predicted Number')
    plt.title('Top Predictions by Model')
    plt.ylim(0, 36)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join("models", 'enhanced_model_comparison.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved comparison visualization to {fig_path}")
    
    print("\nEnhanced model testing with sample data complete!")
    print(f"Comparison visualization saved to {fig_path}")
    print("\nKey improvements implemented:")
    print("1. Enhanced data cleaning with better deduplication")
    print("2. Schedule change handling (3 to 4 draws per day)")
    print("3. Time-sensitive frequency model with recency weighting")
    print("4. Cultural pattern integration")
    print("5. Optimized hybrid model with dynamic weighting")

if __name__ == "__main__":
    main()