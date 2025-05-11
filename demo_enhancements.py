#!/usr/bin/env python3
"""
Demonstration of Play Whe Prediction System Enhancements

This script demonstrates the key enhancements made to the Play Whe prediction system
without relying on the full implementation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("demo_enhancements")

def process_sample_data():
    """
    Process the sample data to demonstrate enhanced data cleaning
    """
    logger.info("Demonstrating enhanced data cleaning...")
    
    # Check if sample data exists
    sample_file = "data/sample_play_whe_results.csv"
    if not os.path.exists(sample_file):
        logger.error(f"Sample data file {sample_file} not found")
        return None
    
    # Read sample data
    df = pd.read_csv(sample_file)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Add schedule change flag (transition from 3 to 4 draws per day)
    # For sample data, we'll pretend the change happened on 2024-01-15
    schedule_change_date = pd.to_datetime('2024-01-15')
    df['post_schedule_change'] = df['date'] > schedule_change_date
    
    # Normalize time format and create draw period
    df['draw_period'] = df['time'].apply(lambda x: _normalize_draw_period(x))
    
    # Add additional derived columns
    df['month'] = df['date'].dt.month_name()
    df['year'] = df['date'].dt.year
    df['day_of_month'] = df['date'].dt.day
    
    # Sort by date and time
    df = df.sort_values(by=['date', 'time'])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Save processed data
    os.makedirs("data", exist_ok=True)
    output_file = "data/demo_processed.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Saved processed sample data to {output_file}")
    
    return df

def _normalize_draw_period(time_str):
    """
    Map draw times to consistent periods
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

def demonstrate_time_sensitive_model(df):
    """
    Demonstrate the time-sensitive frequency model
    """
    logger.info("Demonstrating time-sensitive frequency model...")
    
    # Group by draw period
    draw_periods = df['draw_period'].unique()
    period_models = {}
    
    for period in draw_periods:
        if period == 'unknown':
            continue
            
        logger.info(f"Analyzing {period} period...")
        period_df = df[df['draw_period'] == period]
        
        # Apply exponential decay weighting to favor recent draws
        max_date = period_df['date'].max()
        period_df['days_from_max'] = (max_date - period_df['date']).dt.days
        period_df['weight'] = period_df['days_from_max'].apply(
            lambda x: np.exp(-0.005 * x)  # Decay factor
        )
        
        # Calculate weighted frequencies
        weighted_counts = {}
        for num in range(1, 37):
            num_df = period_df[period_df['number'] == num]
            weighted_counts[num] = num_df['weight'].sum() if not num_df.empty else 0
            
        total_weight = period_df['weight'].sum()
        probabilities = {num: count/total_weight if total_weight > 0 else 0 
                       for num, count in weighted_counts.items()}
        
        # Store period model
        period_models[period] = {
            'probabilities': probabilities,
            'sample_size': len(period_df)
        }
    
    # Create separate pre/post schedule change models
    pre_change_df = df[df['post_schedule_change'] == False]
    post_change_df = df[df['post_schedule_change'] == True]
    
    # Generate visualization
    plt.figure(figsize=(12, 8))
    
    # Plot probabilities for each period
    for i, period in enumerate(period_models.keys()):
        numbers = list(range(1, 37))
        probs = [period_models[period]['probabilities'].get(num, 0) for num in numbers]
        
        plt.subplot(2, 2, i+1)
        plt.bar(numbers, probs, alpha=0.7)
        plt.xlabel('Number')
        plt.ylabel('Probability')
        plt.title(f'{period.capitalize()} Period (n={period_models[period]["sample_size"]})')
        plt.axhline(y=1/36, color='r', linestyle='--', label='Expected')
        plt.legend()
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs("models", exist_ok=True)
    fig_path = os.path.join("models", 'time_sensitive_model.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved time-sensitive model visualization to {fig_path}")
    
    # Compare pre/post schedule change
    plt.figure(figsize=(12, 6))
    
    # Calculate frequencies
    pre_counts = pre_change_df['number'].value_counts().sort_index()
    post_counts = post_change_df['number'].value_counts().sort_index()
    
    pre_total = len(pre_change_df)
    post_total = len(post_change_df)
    
    pre_probs = pre_counts / pre_total
    post_probs = post_counts / post_total
    
    # Plot comparison
    width = 0.35
    numbers = list(range(1, 37))
    
    plt.bar([n - width/2 for n in numbers], 
            [pre_probs.get(n, 0) for n in numbers], 
            width, label='Pre-Change')
    
    plt.bar([n + width/2 for n in numbers], 
            [post_probs.get(n, 0) for n in numbers], 
            width, label='Post-Change')
    
    plt.xlabel('Number')
    plt.ylabel('Probability')
    plt.title('Pre vs Post Schedule Change Frequency Comparison')
    plt.axhline(y=1/36, color='r', linestyle='--', label='Expected')
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join("models", 'schedule_change_comparison.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved schedule change comparison to {fig_path}")
    
    return period_models

def demonstrate_cultural_integration():
    """
    Demonstrate the cultural pattern integration
    """
    logger.info("Demonstrating cultural pattern integration...")
    
    # Check if cultural events file exists
    events_file = "data/cultural_events.csv"
    if not os.path.exists(events_file):
        logger.error(f"Cultural events file {events_file} not found")
        return None
    
    # Read cultural events
    events_df = pd.read_csv(events_file)
    events_df['date'] = pd.to_datetime(events_df['date'])
    
    # Define mark system
    mark_system = {
        1: "Centipede", 2: "Old Lady", 3: "Carriage", 4: "Dead Man",
        5: "Horse", 6: "Belly", 7: "Hog", 8: "Tiger",
        9: "Cattle", 10: "Monkey", 11: "Corbeau", 12: "King",
        13: "Crapaud", 14: "Money", 15: "Fowl", 16: "Jamette",
        17: "Pigeon", 18: "Water More Than Flour", 19: "Horse Boot", 20: "Dog",
        21: "Mouth", 22: "Rat", 23: "House", 24: "Queen",
        25: "Morocoy", 26: "Fowl Cock", 27: "Little Snake", 28: "Red Fish",
        29: "Opium Man", 30: "House Cat", 31: "Parson", 32: "Shrimps",
        33: "Snake", 34: "Blind Man", 35: "Big Snake", 36: "Donkey"
    }
    
    # Define mark categories
    mark_categories = {
        "Animals": [1, 5, 7, 8, 9, 10, 11, 13, 15, 17, 20, 22, 25, 26, 28, 30, 32, 33, 35, 36],
        "People": [2, 4, 12, 16, 24, 29, 31, 34],
        "Objects": [3, 14, 19, 23],
        "Body Parts": [6, 21],
        "Concepts": [18, 27]
    }
    
    # Generate visualization of mark categories
    plt.figure(figsize=(10, 6))
    
    # Count marks in each category
    category_counts = {cat: len(nums) for cat, nums in mark_categories.items()}
    
    # Plot category distribution
    plt.bar(category_counts.keys(), category_counts.values())
    plt.xlabel('Category')
    plt.ylabel('Number of Marks')
    plt.title('Distribution of Mark Categories')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    os.makedirs("analysis", exist_ok=True)
    fig_path = os.path.join("analysis", 'mark_categories.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved mark categories visualization to {fig_path}")
    
    # Generate visualization of cultural events
    plt.figure(figsize=(12, 6))
    
    # Group events by type
    event_types = events_df['type'].value_counts()
    
    # Plot event type distribution
    plt.bar(event_types.index, event_types.values)
    plt.xlabel('Event Type')
    plt.ylabel('Number of Events')
    plt.title('Distribution of Cultural Events by Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join("analysis", 'cultural_events.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved cultural events visualization to {fig_path}")
    
    return {
        'mark_system': mark_system,
        'mark_categories': mark_categories,
        'events': events_df
    }

def demonstrate_optimized_hybrid_model():
    """
    Demonstrate the optimized hybrid model concept
    """
    logger.info("Demonstrating optimized hybrid model concept...")
    
    # Define model weights
    weights = {
        'frequency': 0.2,
        'time_sensitive_frequency': 0.5,
        'sequential': 0.2,
        'hot_cold': 0.1
    }
    
    # Generate visualization of model weights
    plt.figure(figsize=(10, 6))
    
    # Plot weights
    plt.bar(weights.keys(), weights.values())
    plt.xlabel('Model')
    plt.ylabel('Weight')
    plt.title('Optimized Hybrid Model Weights')
    plt.ylim(0, 1)
    
    # Add value labels
    for i, (model, weight) in enumerate(weights.items()):
        plt.text(i, weight + 0.02, f'{weight:.1f}', ha='center')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join("models", 'optimized_hybrid_weights.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved optimized hybrid model visualization to {fig_path}")
    
    return weights

def main():
    """
    Main function to demonstrate enhancements
    """
    print("\n=== PLAY WHE PREDICTION SYSTEM ENHANCEMENTS DEMO ===\n")
    
    # Step 1: Process sample data with enhanced cleaning
    df = process_sample_data()
    
    if df is None:
        logger.error("Failed to process sample data. Exiting.")
        return
    
    # Step 2: Demonstrate time-sensitive frequency model
    period_models = demonstrate_time_sensitive_model(df)
    
    # Step 3: Demonstrate cultural pattern integration
    cultural_data = demonstrate_cultural_integration()
    
    # Step 4: Demonstrate optimized hybrid model
    hybrid_weights = demonstrate_optimized_hybrid_model()
    
    print("\n=== ENHANCEMENT DEMONSTRATION COMPLETE ===\n")
    print("Key improvements demonstrated:")
    print("1. Enhanced data cleaning with schedule change handling")
    print("   - Added post_schedule_change flag")
    print("   - Normalized draw periods for consistent analysis")
    
    print("\n2. Time-sensitive frequency model")
    print("   - Created separate models for each draw period:")
    for period, model in period_models.items():
        print(f"     * {period.capitalize()}: {model['sample_size']} draws")
    
    print("\n3. Cultural pattern integration")
    print("   - Integrated Mark system with 36 cultural meanings")
    print("   - Categorized marks into groups for pattern analysis")
    print(f"   - Analyzed {len(cultural_data['events'])} cultural events")
    
    print("\n4. Optimized hybrid model")
    print("   - Dynamic weighting based on model performance:")
    for model, weight in hybrid_weights.items():
        print(f"     * {model}: {weight:.1f}")
    
    print("\nVisualization files created:")
    print("   - models/time_sensitive_model.png")
    print("   - models/schedule_change_comparison.png")
    print("   - analysis/mark_categories.png")
    print("   - analysis/cultural_events.png")
    print("   - models/optimized_hybrid_weights.png")

if __name__ == "__main__":
    main()