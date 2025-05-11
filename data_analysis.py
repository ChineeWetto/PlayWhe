#!/usr/bin/env python3
"""
Play Whe Lottery Data Analysis

This script analyzes the collected Play Whe lottery data to identify patterns,
trends, and statistical insights that can be used for prediction models.

Analysis includes:
- Frequency analysis of each number (1-36)
- Time-based analysis (by draw time, day of week)
- Sequential analysis (patterns across consecutive draws)
- Hot/cold number identification
- Statistical significance testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
import matplotlib.dates as mdates
from scipy import stats
import json

# Configure plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class PlayWheAnalyzer:
    """
    A class to analyze Play Whe lottery data
    """
    
    def __init__(self, data_dir="data", output_dir="analysis"):
        """
        Initialize the analyzer with configuration parameters
        
        Args:
            data_dir (str): Directory containing the data files
            output_dir (str): Directory to save analysis results
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data storage
        self.data = None
        self.frequency_data = None
        self.time_analysis = None
        self.day_analysis = None
        self.sequential_patterns = None
        self.hot_cold_numbers = None
        
    def load_data(self):
        """
        Load and combine all CSV data files
        
        Returns:
            pandas.DataFrame: Combined DataFrame of all data
        """
        print("Loading data files...")
        
        # Find all CSV files in the data directory
        csv_files = glob.glob(os.path.join(self.data_dir, "play_whe_results_*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
            
        # Read and combine all CSV files
        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                print(f"Loaded {file} with {len(df)} records")
            except Exception as e:
                print(f"Error loading {file}: {e}")
                
        if not dfs:
            raise ValueError("No data could be loaded from CSV files")
            
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Clean and prepare the data
        combined_df = self._prepare_data(combined_df)
        
        self.data = combined_df
        print(f"Total records loaded: {len(self.data)}")
        
        return self.data
    
    def _prepare_data(self, df):
        """
        Clean and prepare the data for analysis
        
        Args:
            df (pandas.DataFrame): Raw data
            
        Returns:
            pandas.DataFrame: Cleaned and prepared data
        """
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Convert date to datetime
        if 'date' in df.columns:
            # Handle dates with '??' by replacing with '01'
            df['date'] = df['date'].str.replace('-??', '-01')
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        # Ensure number is integer and filter out invalid values
        if 'number' in df.columns:
            # Filter out numbers outside the valid range (1-36)
            df = df[df['number'].between(1, 36)]
            df['number'] = df['number'].astype(int)
            
        # Ensure draw_number is integer
        if 'draw_number' in df.columns:
            df['draw_number'] = df['draw_number'].astype(int)
            
        # Extract day of week if not already present
        if 'date' in df.columns and 'day_of_week' not in df.columns:
            df['day_of_week'] = df['date'].dt.day_name()
            
        # Sort by date and time
        if 'date' in df.columns and 'time' in df.columns:
            df = df.sort_values(by=['date', 'time'])
            
        return df
    
    def analyze_frequency(self):
        """
        Analyze the frequency of each number (1-36)
        
        Returns:
            pandas.DataFrame: Frequency analysis results
        """
        print("Analyzing number frequencies...")
        
        # Count occurrences of each number
        number_counts = self.data['number'].value_counts().sort_index()
        
        # Calculate frequency percentage
        total_draws = len(self.data)
        number_percentage = (number_counts / total_draws * 100).round(2)
        
        # Calculate expected frequency (uniform distribution)
        expected_percentage = 100 / 36  # 36 possible numbers
        
        # Calculate deviation from expected
        deviation = (number_percentage - expected_percentage).round(2)
        
        # Create frequency dataframe
        frequency_df = pd.DataFrame({
            'number': number_counts.index,
            'count': number_counts.values,
            'percentage': number_percentage.values,
            'expected_percentage': expected_percentage,
            'deviation': deviation.values
        })
        
        # Calculate chi-square test for uniformity
        chi2_stat, p_value = stats.chisquare(number_counts.values)
        
        # Add chi-square test results
        frequency_df.attrs['chi2_stat'] = chi2_stat
        frequency_df.attrs['p_value'] = p_value
        frequency_df.attrs['is_uniform'] = p_value > 0.05
        
        self.frequency_data = frequency_df
        
        # Plot frequency distribution
        self._plot_frequency_distribution(frequency_df)
        
        print(f"Frequency analysis complete. Chi-square p-value: {p_value:.4f}")
        return frequency_df
    
    def _plot_frequency_distribution(self, frequency_df):
        """
        Plot the frequency distribution of numbers
        
        Args:
            frequency_df (pandas.DataFrame): Frequency analysis results
        """
        plt.figure(figsize=(14, 8))
        
        # Create bar plot
        ax = sns.barplot(x='number', y='count', data=frequency_df)
        
        # Add horizontal line for expected frequency
        expected_count = len(self.data) / 36
        plt.axhline(y=expected_count, color='r', linestyle='--', label=f'Expected ({expected_count:.1f})')
        
        # Add labels and title
        plt.xlabel('Number')
        plt.ylabel('Frequency')
        plt.title('Play Whe Number Frequency Distribution (2016-2019)')
        plt.legend()
        
        # Add value labels on top of bars
        for i, v in enumerate(frequency_df['count']):
            ax.text(i, v + 5, str(v), ha='center')
            
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'number_frequency.png'))
        plt.close()
    
    def analyze_time_patterns(self):
        """
        Analyze patterns based on draw time
        
        Returns:
            pandas.DataFrame: Time-based analysis results
        """
        print("Analyzing time-based patterns...")
        
        # Group by time and analyze
        time_groups = self.data.groupby('time')
        
        # Initialize results storage
        time_analysis = []
        
        # Analyze each time group
        for time, group in time_groups:
            # Count occurrences of each number for this time
            number_counts = group['number'].value_counts().sort_index()
            
            # Calculate frequency percentage
            total_draws = len(group)
            number_percentage = (number_counts / total_draws * 100).round(2)
            
            # Find most and least frequent numbers
            most_frequent = number_counts.idxmax()
            least_frequent = number_counts.idxmin()
            
            # Calculate chi-square test for uniformity
            chi2_stat, p_value = stats.chisquare(number_counts.values)
            
            # Store results
            time_analysis.append({
                'time': time,
                'total_draws': total_draws,
                'most_frequent_number': most_frequent,
                'most_frequent_count': number_counts[most_frequent],
                'least_frequent_number': least_frequent,
                'least_frequent_count': number_counts[least_frequent],
                'chi2_stat': chi2_stat,
                'p_value': p_value,
                'is_uniform': p_value > 0.05,
                'number_counts': number_counts.to_dict()
            })
            
        # Convert to DataFrame
        time_analysis_df = pd.DataFrame(time_analysis)
        self.time_analysis = time_analysis_df
        
        # Plot time-based patterns
        self._plot_time_patterns(time_analysis_df)
        
        print("Time-based analysis complete.")
        return time_analysis_df
    
    def _plot_time_patterns(self, time_analysis_df):
        """
        Plot time-based patterns
        
        Args:
            time_analysis_df (pandas.DataFrame): Time-based analysis results
        """
        # Plot most frequent numbers by time
        plt.figure(figsize=(10, 6))
        sns.barplot(x='time', y='most_frequent_number', data=time_analysis_df)
        plt.title('Most Frequent Number by Draw Time')
        plt.xlabel('Draw Time')
        plt.ylabel('Number')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'most_frequent_by_time.png'))
        plt.close()
        
        # Create heatmap of number frequencies by time
        plt.figure(figsize=(18, 10))
        
        # Prepare data for heatmap
        heatmap_data = np.zeros((36, len(time_analysis_df)))
        times = time_analysis_df['time'].values
        
        for i, row in enumerate(time_analysis_df.itertuples()):
            counts_dict = row.number_counts
            for num in range(1, 37):
                heatmap_data[num-1, i] = counts_dict.get(num, 0)
                
        # Create heatmap
        ax = sns.heatmap(heatmap_data, cmap="YlGnBu", 
                     xticklabels=times, 
                     yticklabels=range(1, 37),
                     cbar_kws={'label': 'Frequency'})
        
        plt.title('Number Frequency by Draw Time')
        plt.xlabel('Draw Time')
        plt.ylabel('Number')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'number_frequency_by_time.png'))
        plt.close()
    
    def analyze_day_patterns(self):
        """
        Analyze patterns based on day of week
        
        Returns:
            pandas.DataFrame: Day-based analysis results
        """
        print("Analyzing day of week patterns...")
        
        # Ensure day_of_week column exists
        if 'day_of_week' not in self.data.columns:
            print("Warning: day_of_week column not found. Skipping day analysis.")
            return None
            
        # Group by day and analyze
        day_groups = self.data.groupby('day_of_week')
        
        # Initialize results storage
        day_analysis = []
        
        # Define day order for sorting
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Analyze each day group
        for day, group in day_groups:
            if day not in day_order:
                continue
                
            # Count occurrences of each number for this day
            number_counts = group['number'].value_counts().sort_index()
            
            # Calculate frequency percentage
            total_draws = len(group)
            number_percentage = (number_counts / total_draws * 100).round(2)
            
            # Find most and least frequent numbers
            most_frequent = number_counts.idxmax()
            least_frequent = number_counts.idxmin()
            
            # Calculate chi-square test for uniformity
            chi2_stat, p_value = stats.chisquare(number_counts.values)
            
            # Store results
            day_analysis.append({
                'day': day,
                'total_draws': total_draws,
                'most_frequent_number': most_frequent,
                'most_frequent_count': number_counts[most_frequent],
                'least_frequent_number': least_frequent,
                'least_frequent_count': number_counts[least_frequent],
                'chi2_stat': chi2_stat,
                'p_value': p_value,
                'is_uniform': p_value > 0.05,
                'number_counts': number_counts.to_dict()
            })
            
        # Convert to DataFrame and sort by day order
        day_analysis_df = pd.DataFrame(day_analysis)
        if not day_analysis_df.empty and 'day' in day_analysis_df.columns:
            day_analysis_df['day'] = pd.Categorical(day_analysis_df['day'], categories=day_order, ordered=True)
            day_analysis_df = day_analysis_df.sort_values('day')
        
        self.day_analysis = day_analysis_df
        
        # Plot day-based patterns
        self._plot_day_patterns(day_analysis_df)
        
        print("Day-based analysis complete.")
        return day_analysis_df
    
    def _plot_day_patterns(self, day_analysis_df):
        """
        Plot day-based patterns
        
        Args:
            day_analysis_df (pandas.DataFrame): Day-based analysis results
        """
        if day_analysis_df is None or len(day_analysis_df) == 0:
            return
            
        # Plot most frequent numbers by day
        plt.figure(figsize=(12, 6))
        sns.barplot(x='day', y='most_frequent_number', data=day_analysis_df)
        plt.title('Most Frequent Number by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Number')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'most_frequent_by_day.png'))
        plt.close()
        
        # Create heatmap of number frequencies by day
        plt.figure(figsize=(18, 10))
        
        # Prepare data for heatmap
        heatmap_data = np.zeros((36, len(day_analysis_df)))
        days = day_analysis_df['day'].values
        
        for i, row in enumerate(day_analysis_df.itertuples()):
            counts_dict = row.number_counts
            for num in range(1, 37):
                heatmap_data[num-1, i] = counts_dict.get(num, 0)
                
        # Create heatmap
        ax = sns.heatmap(heatmap_data, cmap="YlGnBu", 
                     xticklabels=days, 
                     yticklabels=range(1, 37),
                     cbar_kws={'label': 'Frequency'})
        
        plt.title('Number Frequency by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Number')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'number_frequency_by_day.png'))
        plt.close()
    
    def analyze_sequential_patterns(self):
        """
        Analyze patterns across consecutive draws
        
        Returns:
            dict: Sequential pattern analysis results
        """
        print("Analyzing sequential patterns...")
        
        # Sort data by date and time
        sorted_data = self.data.sort_values(by=['date', 'time'])
        
        # Get the sequence of numbers
        number_sequence = sorted_data['number'].values
        
        # Initialize results
        sequential_patterns = {
            'repeat_analysis': {},
            'transition_matrix': np.zeros((36, 36)),
            'common_pairs': [],
            'common_triplets': []
        }
        
        # Analyze repeats (same number in consecutive draws)
        repeats = 0
        for i in range(1, len(number_sequence)):
    
(Content truncated due to size limit. Use line ranges to read in chunks)