#!/usr/bin/env python3
"""
Generate an extended sample dataset for Play Whe testing
"""

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Error: Required packages (pandas, numpy) not installed.")
    print("Please install them using: pip install pandas numpy")
    exit(1)

import random
from datetime import datetime, timedelta
import os
import csv

def generate_extended_sample_data(num_days=90, output_path="data/extended_sample_play_whe_results.csv"):
    """
    Generate an extended sample dataset with realistic patterns
    
    Args:
        num_days (int): Number of days to generate data for
        output_path (str): Path to save the generated data
        
    Returns:
        pandas.DataFrame: Generated sample data
    """
    # Define draw times
    draw_times = ["10:30AM", "1:00PM", "4:00PM", "6:30PM"]
    
    # Start date (90 days before current date)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)
    
    # Create date range
    date_range = [start_date + timedelta(days=i) for i in range(num_days)]
    
    # Initialize data
    data = []
    draw_number = 38000  # Starting draw number
    
    # Create some number patterns
    # More frequent numbers (higher probability)
    hot_numbers = [7, 14, 22, 28, 36]
    # Less frequent numbers (lower probability)
    cold_numbers = [2, 6, 13, 25, 31]
    # Regular numbers (medium probability)
    regular_numbers = [n for n in range(1, 37) if n not in hot_numbers and n not in cold_numbers]
    
    # Generate data
    for date in date_range:
        day_of_week = date.strftime('%A')
        
        for time in draw_times:
            draw_number += 1
            
            # Determine number probabilities based on patterns
            probabilities = []
            for n in range(1, 37):
                if n in hot_numbers:
                    prob = 0.05  # 5% chance each (25% total for hot numbers)
                elif n in cold_numbers:
                    prob = 0.01  # 1% chance each (5% total for cold numbers)
                else:
                    prob = 0.0269  # ~2.7% chance each (70% total for regular numbers)
                probabilities.append(prob)
            
            # Add time-based patterns
            if "10:30AM" in time:
                # Morning draws favor lower numbers slightly
                for i in range(18):  # Numbers 1-18
                    probabilities[i] *= 1.2
            elif "4:00PM" in time:
                # Afternoon draws favor higher numbers slightly
                for i in range(18, 36):  # Numbers 19-36
                    probabilities[i] *= 1.2
            
            # Add day-based patterns
            if day_of_week == "Monday":
                # Mondays favor multiples of 5
                for i in range(36):
                    if (i + 1) % 5 == 0:
                        probabilities[i] *= 1.5
            elif day_of_week == "Friday":
                # Fridays favor prime numbers
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
                for prime in primes:
                    probabilities[prime - 1] *= 1.3
            
            # Normalize probabilities to sum to 1
            total_prob = sum(probabilities)
            probabilities = [p / total_prob for p in probabilities]
            
            # Choose number based on probabilities
            number = np.random.choice(range(1, 37), p=probabilities)
            
            # Create data entry
            entry = {
                "date": date.strftime('%Y-%m-%d'),
                "time": time,
                "number": number,
                "draw_number": draw_number,
                "day_of_week": day_of_week
            }
            
            data.append(entry)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} sample records covering {num_days} days")
    print(f"Saved to {output_path}")
    
    return df

if __name__ == "__main__":
    generate_extended_sample_data()