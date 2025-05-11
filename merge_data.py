#!/usr/bin/env python3
"""
Play Whe Lottery Data Merger and Cleaner

This script merges all the individual monthly CSV files into a single dataset,
performs data cleaning, and prepares the data for analysis.
"""

import os
import glob
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/merge_data.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("play_whe_data_merger")

class PlayWheDataMerger:
    """
    A class to merge and clean Play Whe lottery data
    """
    
    def __init__(self, data_dir="data", output_dir="data"):
        """
        Initialize the merger with configuration parameters
        
        Args:
            data_dir (str): Directory containing the data files
            output_dir (str): Directory to save the merged data
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def merge_csv_files(self, pattern="play_whe_results_*.csv"):
        """
        Merge all CSV files matching the pattern
        
        Args:
            pattern (str): Glob pattern to match CSV files
            
        Returns:
            pandas.DataFrame: Merged DataFrame
        """
        logger.info(f"Merging CSV files matching pattern: {pattern}")
        
        # Get list of all CSV files
        csv_files = glob.glob(os.path.join(self.data_dir, pattern))
        
        if not csv_files:
            logger.error(f"No CSV files found matching pattern: {pattern}")
            return None
            
        logger.info(f"Found {len(csv_files)} CSV files to merge")
        
        # Read and merge all CSV files
        dfs = []
        for file in csv_files:
            try:
                logger.debug(f"Reading file: {file}")
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error reading file {file}: {e}")
                
        if not dfs:
            logger.error("No data frames were created")
            return None
            
        # Concatenate all data frames
        merged_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Merged data frame has {len(merged_df)} rows")
        
        return merged_df
        
    def clean_data(self, df):
        """
        Clean the merged data
        
        Args:
            df (pandas.DataFrame): DataFrame to clean
            
        Returns:
            pandas.DataFrame: Cleaned DataFrame
        """
        if df is None or df.empty:
            logger.error("No data to clean")
            return None
            
        logger.info("Cleaning data with enhanced deduplication...")
        
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Standardize date and time formats first
        cleaned_df['date'] = pd.to_datetime(cleaned_df['date'], errors='coerce')
        
        # Normalize time formats
        if 'time' in cleaned_df.columns:
            cleaned_df['time'] = cleaned_df['time'].apply(self._normalize_time_format)
        
        # Create a composite key for deduplication
        cleaned_df['dedup_key'] = cleaned_df['date'].dt.strftime('%Y-%m-%d') + '_' + \
                                 cleaned_df['time'] + '_' + \
                                 cleaned_df['number'].astype(str)
        
        # Remove duplicates based on the composite key
        original_len = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(subset=['dedup_key'])
        logger.info(f"Removed {original_len - len(cleaned_df)} duplicate rows")
        
        # Drop the temporary dedup_key
        cleaned_df = cleaned_df.drop(columns=['dedup_key'])
        
        # Ensure number column is numeric and within valid range (1-36)
        cleaned_df['number'] = pd.to_numeric(cleaned_df['number'], errors='coerce')
        cleaned_df = cleaned_df[cleaned_df['number'].between(1, 36)]
        logger.info(f"Removed {original_len - len(cleaned_df)} rows with invalid numbers")
        
        # Ensure date is in proper format
        cleaned_df = cleaned_df.dropna(subset=['date'])
        logger.info(f"Removed {original_len - len(cleaned_df)} rows with invalid dates")
        
        # Add additional derived columns
        cleaned_df['day_of_week'] = cleaned_df['date'].dt.day_name()
        cleaned_df['month'] = cleaned_df['date'].dt.month_name()
        cleaned_df['year'] = cleaned_df['date'].dt.year
        cleaned_df['day_of_month'] = cleaned_df['date'].dt.day
        cleaned_df['week_of_year'] = cleaned_df['date'].dt.isocalendar().week
        
        # Add schedule change flag (transition from 3 to 4 draws per day)
        # Adjust this date based on when the actual change occurred
        schedule_change_date = pd.to_datetime('2018-06-01')
        cleaned_df['post_schedule_change'] = cleaned_df['date'] > schedule_change_date
        
        # Create draw period identifier that's consistent across schedule changes
        cleaned_df['draw_period'] = cleaned_df['time'].apply(self._normalize_draw_period)
        
        # Sort by date and time
        cleaned_df = cleaned_df.sort_values(by=['date', 'time'])
        
        # Reset index
        cleaned_df = cleaned_df.reset_index(drop=True)
        
        logger.info(f"Cleaned data frame has {len(cleaned_df)} rows")
        
        return cleaned_df
        
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
        
        # Handle common formats
        if ':' in time_str:
            parts = time_str.split(':')
            hour = parts[0].strip()
            
            if len(parts) > 1:
                minute_parts = parts[1].split()
                minute = minute_parts[0].strip()
                
                if len(minute_parts) > 1:
                    ampm = minute_parts[1].strip()
                elif 'AM' in parts[1]:
                    ampm = 'AM'
                    minute = minute.replace('AM', '')
                elif 'PM' in parts[1]:
                    ampm = 'PM'
                    minute = minute.replace('PM', '')
                else:
                    # Default to AM for morning draws, PM for others
                    hour_num = int(hour)
                    ampm = 'AM' if hour_num < 12 else 'PM'
            else:
                minute = '00'
                # Default to AM for morning draws, PM for others
                hour_num = int(hour)
                ampm = 'AM' if hour_num < 12 else 'PM'
        else:
            # Handle formats without colon
            if 'AM' in time_str:
                hour = time_str.replace('AM', '').strip()
                minute = '00'
                ampm = 'AM'
            elif 'PM' in time_str:
                hour = time_str.replace('PM', '').strip()
                minute = '00'
                ampm = 'PM'
            else:
                hour = time_str
                minute = '00'
                hour_num = int(hour)
                ampm = 'AM' if hour_num < 12 else 'PM'
        
        # Format the time consistently
        return f"{hour}:{minute}{ampm}"
        
    def _normalize_draw_period(self, time_str):
        """
        Map draw times to consistent periods across schedule changes
        
        Args:
            time_str (str): Time string
            
        Returns:
            str: Normalized draw period
        """
        if pd.isna(time_str):
            return 'unknown'
            
        time_str = str(time_str).upper()
        
        if '10:30' in time_str or ('10' in time_str and 'AM' in time_str):
            return 'morning'
        elif '1:00' in time_str or '1:30' in time_str or ('1' in time_str and 'PM' in time_str):
            return 'midday'
        elif '4:00' in time_str or '4:30' in time_str or ('4' in time_str and 'PM' in time_str):
            return 'afternoon'
        elif '6:30' in time_str or '7:00' in time_str or ('6' in time_str and 'PM' in time_str) or ('7' in time_str and 'PM' in time_str):
            return 'evening'
        else:
            return 'unknown'
        
    def add_sequential_features(self, df):
        """
        Add sequential features to the data
        
        Args:
            df (pandas.DataFrame): DataFrame to add features to
            
        Returns:
            pandas.DataFrame: DataFrame with added features
        """
        if df is None or df.empty:
            logger.error("No data to add features to")
            return None
            
        logger.info("Adding sequential features...")
        
        # Make a copy to avoid modifying the original
        featured_df = df.copy()
        
        # Sort by date and time to ensure proper sequence
        featured_df = featured_df.sort_values(by=['date', 'time'])
        
        # Add previous draw numbers
        for i in range(1, 6):
            featured_df[f'prev_{i}_number'] = featured_df['number'].shift(i)
            
        # Add next draw number (for training only, would not be available in production)
        featured_df['next_number'] = featured_df['number'].shift(-1)
        
        # Calculate days since last occurrence of each number
        number_indices = {}
        days_since_last = np.zeros(len(featured_df))
        
        for idx, row in featured_df.iterrows():
            num = row['number']
            if num in number_indices:
                days_since_last[idx] = (row['date'] - featured_df.loc[number_indices[num], 'date']).days
            number_indices[num] = idx
            
        featured_df['days_since_last'] = days_since_last
        
        # Calculate rolling frequency of each number
        window_sizes = [10, 30, 50, 100]
        for window in window_sizes:
            featured_df[f'freq_last_{window}'] = featured_df.groupby('number')['number'].transform(
                lambda x: x.rolling(window, min_periods=1).count()
            )
            
        logger.info(f"Added sequential features to data frame")
        
        return featured_df
        
    def process_data(self):
        """
        Process the data: merge, clean, and add features
        
        Returns:
            pandas.DataFrame: Processed DataFrame
        """
        # Merge CSV files
        merged_df = self.merge_csv_files()
        
        if merged_df is None or merged_df.empty:
            logger.error("Failed to merge data")
            return None
            
        # Clean data
        cleaned_df = self.clean_data(merged_df)
        
        if cleaned_df is None or cleaned_df.empty:
            logger.error("Failed to clean data")
            return None
            
        # Add sequential features
        featured_df = self.add_sequential_features(cleaned_df)
        
        if featured_df is None or featured_df.empty:
            logger.error("Failed to add features to data")
            return None
            
        # Save processed data
        csv_path = os.path.join(self.output_dir, 'play_whe_processed.csv')
        featured_df.to_csv(csv_path, index=False)
        logger.info(f"Saved processed data to {csv_path}")
        
        # Save a version without the 'next_number' column for production use
        production_df = featured_df.drop(columns=['next_number'])
        prod_csv_path = os.path.join(self.output_dir, 'play_whe_production.csv')
        production_df.to_csv(prod_csv_path, index=False)
        logger.info(f"Saved production data to {prod_csv_path}")
        
        return featured_df
        
    def generate_summary(self, df):
        """
        Generate a summary of the processed data
        
        Args:
            df (pandas.DataFrame): Processed DataFrame
            
        Returns:
            dict: Summary statistics
        """
        if df is None or df.empty:
            logger.error("No data to generate summary from")
            return None
            
        logger.info("Generating data summary...")
        
        summary = {
            'total_draws': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            },
            'unique_numbers': df['number'].nunique(),
            'number_frequencies': df['number'].value_counts().to_dict(),
            'day_of_week_distribution': df['day_of_week'].value_counts().to_dict(),
            'time_distribution': df['time'].value_counts().to_dict(),
            'most_common_numbers': df['number'].value_counts().head(10).to_dict(),
            'least_common_numbers': df['number'].value_counts().tail(10).to_dict()
        }
        
        # Save summary to JSON
        import json
        json_path = os.path.join(self.output_dir, 'data_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved data summary to {json_path}")
        
        return summary
        
if __name__ == "__main__":
    # Create merger instance
    merger = PlayWheDataMerger()
    
    # Process data
    processed_df = merger.process_data()
    
    # Generate summary
    if processed_df is not None and not processed_df.empty:
        summary = merger.generate_summary(processed_df)
        
        # Print summary
        print(f"Successfully processed {summary['total_draws']} Play Whe draws")
        print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"Number of unique numbers drawn: {summary['unique_numbers']}")
        
        print("\nMost common numbers:")
        for number, count in sorted(summary['most_common_numbers'].items(), key=lambda x: int(x[1]), reverse=True):
            print(f"Number {number}: {count} occurrences")
            
        print("\nLeast common numbers:")
        for number, count in sorted(summary['least_common_numbers'].items(), key=lambda x: int(x[1])):
            print(f"Number {number}: {count} occurrences")
    else:
        print("Failed to process data")
