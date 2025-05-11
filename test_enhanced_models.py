#!/usr/bin/env python3
"""
Test script for enhanced Play Whe prediction models
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from merge_data import PlayWheDataMerger
from prediction_models import PlayWhePredictionModels
from cultural_patterns import CulturalPatternAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/test_enhanced_models.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("test_enhanced_models")

def main():
    """
    Main function to test enhanced prediction models
    """
    # Create output directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("analysis", exist_ok=True)
    
    # Step 1: Process data with enhanced cleaning
    logger.info("Step 1: Processing data with enhanced cleaning...")
    data_merger = PlayWheDataMerger()
    processed_df = data_merger.process_data()
    
    if processed_df is None or processed_df.empty:
        logger.error("Failed to process data. Exiting.")
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
    cultural_analyzer = CulturalPatternAnalyzer(data_file="data/play_whe_processed.csv")
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
    
    print("\nEnhanced model testing complete!")
    print(f"Comparison visualization saved to {fig_path}")

if __name__ == "__main__":
    main()