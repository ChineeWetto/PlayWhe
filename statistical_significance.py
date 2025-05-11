#!/usr/bin/env python3
"""
Statistical Significance Testing for Play Whe Prediction

This module implements statistical significance testing to validate that observed
patterns differ significantly from random distribution, quantify confidence levels
for each prediction, and identify which draw periods show the strongest non-random
patterns.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import json
from scipy import stats
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/statistical_significance.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("statistical_significance")

class StatisticalSignificanceTester:
    """
    A class for testing statistical significance of patterns in Play Whe data
    """
    
    def __init__(self, data_file="data/play_whe_processed.csv", output_dir="analysis"):
        """
        Initialize the tester with configuration parameters
        """
        self.data_file = data_file
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.df = self._load_data()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set(font_scale=1.2)
        
    def _load_data(self):
        """
        Load the processed data
        """
        try:
            logger.info(f"Loading data from {self.data_file}")
            df = pd.read_csv(self.data_file)
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"Loaded data with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
            
    def test_frequency_distribution(self, data=None, period=None):
        """
        Test if the frequency distribution of numbers differs from uniform
        """
        # Use provided data or filter by period
        if data is None:
            if period and 'draw_period' in self.df.columns:
                data = self.df[self.df['draw_period'] == period]
            else:
                data = self.df
                
        if data is None or data.empty:
            logger.warning(f"No data available for period {period}")
            return None
            
        logger.info(f"Testing frequency distribution for {period if period else 'all periods'}")
        
        # Count occurrences of each number
        number_counts = data['number'].value_counts().sort_index()
        
        # Ensure all numbers 1-36 are represented
        for num in range(1, 37):
            if num not in number_counts.index:
                number_counts[num] = 0
                
        number_counts = number_counts.sort_index()
        
        # Calculate expected counts (uniform distribution)
        total_draws = len(data)
        expected_counts = np.full(36, total_draws / 36)
        
        # Chi-square test
        chi2_stat, p_value = stats.chisquare(number_counts.values, expected_counts)
        
        # Calculate deviation from expected
        expected_percentage = 100 / 36
        percentage = (number_counts / total_draws * 100).round(2)
        deviation = (percentage - expected_percentage).round(2)
        
        # Find most and least frequent numbers
        most_frequent = number_counts.idxmax()
        least_frequent = number_counts.idxmin()
        
        # Calculate maximum deviations
        max_positive_deviation = deviation.max()
        max_negative_deviation = deviation.min()
        
        # Generate visualization
        plt.figure(figsize=(14, 6))
        
        # Plot frequency distribution
        ax = plt.subplot(111)
        bars = ax.bar(range(1, 37), number_counts.values)
        
        # Color bars based on deviation
        for i, bar in enumerate(bars):
            if deviation.iloc[i] > 0:
                bar.set_color('red')
            else:
                bar.set_color('blue')
                
        # Add expected line
        ax.axhline(y=total_draws/36, color='r', linestyle='--', 
                  label=f'Expected ({total_draws/36:.1f})')
        
        ax.set_xlabel('Number')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Number Frequency Distribution ({period if period else "All Periods"})')
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        period_str = period if period else 'all'
        fig_path = os.path.join(self.output_dir, f'frequency_distribution_{period_str}.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        # Compile results
        results = {
            'chi_square': chi2_stat,
            'p_value': p_value,
            'is_uniform': p_value > 0.05,
            'most_frequent_number': int(most_frequent),
            'most_frequent_count': int(number_counts[most_frequent]),
            'least_frequent_number': int(least_frequent),
            'least_frequent_count': int(number_counts[least_frequent]),
            'max_positive_deviation': float(max_positive_deviation),
            'max_negative_deviation': float(max_negative_deviation),
            'number_counts': number_counts.to_dict(),
            'deviation': deviation.to_dict(),
            'total_draws': total_draws
        }
        
        logger.info(f"Frequency distribution test complete. Chi-square p-value: {p_value:.4f}")
        
        return results
        
    def test_all_periods(self):
        """
        Test frequency distribution for all draw periods
        """
        if self.df is None or self.df.empty:
            logger.error("No data to test")
            return None
            
        logger.info("Testing frequency distribution for all periods")
        
        # Test overall distribution
        overall_results = self.test_frequency_distribution()
        
        period_results = {'overall': overall_results}
        
        # Test each draw period if available
        if 'draw_period' in self.df.columns:
            draw_periods = self.df['draw_period'].unique()
            
            for period in draw_periods:
                if period == 'unknown':
                    continue
                    
                results = self.test_frequency_distribution(period=period)
                if results:
                    period_results[period] = results
                    
        # Generate comparison visualization
        self._visualize_period_comparison(period_results)
        
        return period_results
        
    def _visualize_period_comparison(self, period_results):
        """
        Generate visualization comparing significance across periods
        """
        if not period_results:
            return
            
        # Extract p-values and chi-square statistics
        periods = list(period_results.keys())
        p_values = [period_results[period]['p_value'] for period in periods]
        chi_square = [period_results[period]['chi_square'] for period in periods]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot p-values
        ax1 = plt.subplot(111)
        bars = ax1.bar(periods, p_values, alpha=0.7)
        
        # Color bars based on significance
        for i, bar in enumerate(bars):
            if p_values[i] < 0.05:
                bar.set_color('red')  # Significant
            else:
                bar.set_color('blue')  # Not significant
                
        # Add significance threshold line
        ax1.axhline(y=0.05, color='r', linestyle='--', label='Significance threshold (p=0.05)')
        
        ax1.set_xlabel('Draw Period')
        ax1.set_ylabel('p-value')
        ax1.set_title('Statistical Significance by Draw Period')
        
        # Add chi-square values as text
        for i, (p, chi) in enumerate(zip(p_values, chi_square)):
            ax1.text(i, p + 0.01, f'χ²={chi:.1f}', ha='center')
            
        ax1.legend()
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, 'period_significance_comparison.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
    def test_sequential_dependency(self, lag=1, period=None):
        """
        Test for sequential dependency between draws
        """
        # Use all data or filter by period
        if period and 'draw_period' in self.df.columns:
            data = self.df[self.df['draw_period'] == period]
        else:
            data = self.df
            
        if data is None or data.empty or len(data) <= lag:
            logger.warning(f"Insufficient data for lag {lag} in period {period}")
            return None
            
        logger.info(f"Testing sequential dependency with lag {lag} for {period if period else 'all periods'}")
        
        # Sort data by date and time
        data = data.sort_values(by=['date', 'time'] if 'time' in data.columns else ['date'])
        
        # Get number sequence
        number_sequence = data['number'].values
        
        # Create transition matrix
        transition_counts = np.zeros((36, 36))
        
        for i in range(len(number_sequence) - lag):
            from_num = number_sequence[i]
            to_num = number_sequence[i + lag]
            transition_counts[from_num - 1, to_num - 1] += 1
            
        # Calculate row sums (excluding zeros)
        row_sums = transition_counts.sum(axis=1)
        
        # Calculate expected counts (uniform distribution)
        expected_counts = np.zeros((36, 36))
        for i in range(36):
            if row_sums[i] > 0:
                expected_counts[i, :] = row_sums[i] / 36
                
        # Flatten non-zero elements for chi-square test
        observed_flat = []
        expected_flat = []
        
        for i in range(36):
            for j in range(36):
                if expected_counts[i, j] > 0:
                    observed_flat.append(transition_counts[i, j])
                    expected_flat.append(expected_counts[i, j])
                    
        # Chi-square test
        chi2_stat, p_value = stats.chisquare(observed_flat, expected_flat)
        
        # Generate visualization
        plt.figure(figsize=(12, 10))
        sns.heatmap(transition_counts, cmap='viridis')
        plt.xlabel(f'Number at t+{lag}')
        plt.ylabel('Number at t')
        plt.title(f'Transition Matrix (Lag {lag}, {period if period else "All Periods"})')
        plt.tight_layout()
        
        # Save figure
        period_str = period if period else 'all'
        fig_path = os.path.join(self.output_dir, f'transition_matrix_lag{lag}_{period_str}.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        # Compile results
        results = {
            'lag': lag,
            'chi_square': chi2_stat,
            'p_value': p_value,
            'is_independent': p_value > 0.05,
            'total_transitions': len(observed_flat)
        }
        
        logger.info(f"Sequential dependency test complete. Chi-square p-value: {p_value:.4f}")
        
        return results
        
    def calculate_confidence_levels(self, probabilities):
        """
        Calculate confidence levels for predictions
        """
        if not probabilities:
            return {}
            
        # Calculate expected probability
        expected_prob = 1/36
        
        # Calculate confidence scores based on deviation from expected
        confidence_scores = {}
        for num, prob in probabilities.items():
            if prob > 0:
                # Calculate how much the probability deviates from expected
                deviation = abs(prob - expected_prob) / expected_prob
                
                # Convert to percentage with a cap at 100%
                confidence_scores[num] = min(deviation * 100, 100)
            else:
                confidence_scores[num] = 0
                
        return confidence_scores
        
    def identify_strongest_periods(self):
        """
        Identify which draw periods show the strongest non-random patterns
        """
        if self.df is None or self.df.empty or 'draw_period' not in self.df.columns:
            logger.warning("Cannot identify strongest periods: missing data or draw_period column")
            return None
            
        logger.info("Identifying draw periods with strongest non-random patterns")
        
        draw_periods = self.df['draw_period'].unique()
        period_scores = {}
        
        for period in draw_periods:
            if period == 'unknown':
                continue
                
            # Test frequency distribution
            freq_results = self.test_frequency_distribution(period=period)
            
            # Test sequential dependency
            seq_results = self.test_sequential_dependency(lag=1, period=period)
            
            if freq_results and seq_results:
                # Calculate non-randomness score (lower p-values = higher score)
                freq_score = 1 - freq_results['p_value']
                seq_score = 1 - seq_results['p_value']
                
                # Combined score (weighted average)
                combined_score = (freq_score + seq_score) / 2
                
                period_scores[period] = {
                    'frequency_p_value': freq_results['p_value'],
                    'sequential_p_value': seq_results['p_value'],
                    'combined_score': combined_score,
                    'is_significant': freq_results['p_value'] < 0.05 or seq_results['p_value'] < 0.05
                }
                
        # Sort periods by combined score
        sorted_periods = sorted(period_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        
        # Generate visualization
        plt.figure(figsize=(12, 6))
        
        periods = [p for p, _ in sorted_periods]
        scores = [s['combined_score'] for _, s in sorted_periods]
        
        bars = plt.bar(periods, scores)
        
        # Color bars based on significance
        for i, (_, score) in enumerate(sorted_periods):
            if score['is_significant']:
                bars[i].set_color('red')
            else:
                bars[i].set_color('blue')
                
        plt.xlabel('Draw Period')
        plt.ylabel('Non-Randomness Score')
        plt.title('Draw Periods Ranked by Non-Randomness')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, 'period_nonrandomness_ranking.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        # Compile results
        results = {
            'period_scores': {p: s for p, s in sorted_periods},
            'strongest_period': sorted_periods[0][0] if sorted_periods else None,
            'significant_periods': [p for p, s in sorted_periods if s['is_significant']]
        }
        
        logger.info(f"Identified {len(results['significant_periods'])} periods with significant non-random patterns")
        
        return results
        
    def run_all_tests(self):
        """
        Run all statistical tests
        """
        if self.df is None or self.df.empty:
            logger.error("No data to test")
            return None
            
        logger.info("Running all statistical tests...")
        
        # Test frequency distribution for all periods
        frequency_results = self.test_all_periods()
        
        # Test sequential dependency
        sequential_results = {}
        for lag in [1, 2, 3]:
            sequential_results[f'lag_{lag}'] = self.test_sequential_dependency(lag)
            
        # Identify strongest periods
        period_strength = self.identify_strongest_periods()
        
        # Compile all results
        results = {
            'frequency': frequency_results,
            'sequential': sequential_results,
            'period_strength': period_strength,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save results
        results_path = os.path.join(self.output_dir, 'statistical_significance_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Saved statistical test results to {results_path}")
        
        return results

if __name__ == "__main__":
    # Create tester instance
    tester = StatisticalSignificanceTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Print summary
    if results:
        print("Statistical significance testing complete")
        
        # Frequency distribution
        if 'frequency' in results and 'overall' in results['frequency']:
            overall = results['frequency']['overall']
            print(f"\nFrequency Distribution: p-value = {overall['p_value']:.4f}")
            print(f"  {'Significant deviation from uniform' if overall['p_value'] < 0.05 else 'Consistent with uniform distribution'}")
            
        # Sequential dependency
        if 'sequential' in results and 'lag_1' in results['sequential']:
            lag1 = results['sequential']['lag_1']
            print(f"\nSequential Dependency (Lag 1): p-value = {lag1['p_value']:.4f}")
            print(f"  {'Significant sequential dependency' if lag1['p_value'] < 0.05 else 'No significant sequential dependency'}")
            
        # Strongest periods
        if 'period_strength' in results and 'significant_periods' in results['period_strength']:
            sig_periods = results['period_strength']['significant_periods']
            if sig_periods:
                print(f"\nDraw periods with significant non-random patterns: {', '.join(sig_periods)}")
                print(f"Strongest period: {results['period_strength']['strongest_period']}")
            else:
                print("\nNo draw periods show significant non-random patterns")
    else:
        print("Failed to complete statistical significance testing")
