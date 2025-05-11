#!/usr/bin/env python3
"""
Create extended sample data for Play Whe testing
"""

import csv
from datetime import datetime, timedelta
import os
import random

def create_sample_data(num_days=90, output_path="data/extended_sample_play_whe_results.csv"):
    """
    Create a larger sample dataset for Play Whe testing
    
    Args:
        num_days (int): Number of days of data to generate
        output_path (str): Path to save the CSV file
    """
    # Define draw times
    draw_times = ["10:30AM", "1:00PM", "4:00PM", "6:30PM"]
    
    # Start date (90 days before current date)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)
    
    # Create date range
    date_range = []
    current_date = start_date
    while current_date <= end_date:
        date_range.append(current_date)
        current_date += timedelta(days=1)
    
    # Initialize data
    data = []
    draw_number = 38000  # Starting draw number
    
    # Create some number patterns
    hot_numbers = [7, 14, 22, 28, 36]
    cold_numbers = [2, 6, 13, 25, 31]
    regular_numbers = [n for n in range(1, 37) if n not in hot_numbers and n not in cold_numbers]
    
    # Generate data
    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        day_of_week = date.strftime('%A')
        
        for time in draw_times:
            draw_number += 1
            
            # Choose number with basic patterns
            if random.random() < 0.25:  # 25% chance for hot numbers
                number = random.choice(hot_numbers)
            elif random.random() < 0.05:  # 5% chance for cold numbers
                number = random.choice(cold_numbers)
            else:  # 70% chance for regular numbers
                number = random.choice(regular_numbers)
                
            # Add time and day patterns
            if "10:30AM" in time and random.random() < 0.2:
                number = random.randint(1, 18)  # Morning favors lower numbers
            elif "4:00PM" in time and random.random() < 0.2:
                number = random.randint(19, 36)  # Afternoon favors higher numbers
                
            if day_of_week == "Monday" and random.random() < 0.3:
                # Mondays favor multiples of 5
                multiples_of_5 = [5, 10, 15, 20, 25, 30, 35]
                number = random.choice(multiples_of_5)
            elif day_of_week == "Friday" and random.random() < 0.3:
                # Fridays favor prime numbers
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
                number = random.choice(primes)
            
            # Create data entry
            entry = {
                "date": date_str,
                "time": time,
                "number": number,
                "draw_number": draw_number,
                "day_of_week": day_of_week
            }
            
            data.append(entry)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to CSV
    with open(output_path, 'w', newline='') as f:
        fieldnames = ["date", "time", "number", "draw_number", "day_of_week"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            writer.writerow(entry)
    
    print(f"Generated {len(data)} sample records covering {num_days} days")
    print(f"Saved to {output_path}")
    
    return True

if __name__ == "__main__":
    create_sample_data()