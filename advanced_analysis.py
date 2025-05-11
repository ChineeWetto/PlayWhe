#!/usr/bin/env python3
"""
Play Whe Lottery Advanced Data Analysis

This script performs comprehensive statistical analysis on the Play Whe lottery data
to identify patterns and insights that can inform prediction models.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import logging
from datetime import datetime
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/advanced_analysis.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("play_whe_advanced_analysis")

class PlayWheAdvancedAnalysis:
    """
    A class to perform advanced analysis on Play Whe lottery data
    """
    
    def __init__(self, data_file="data/play_whe_processed.csv", output_dir="analysis"):
        """
        Initialize the analyzer with configuration parameters
        
        Args:
            data_file (str): Path to the processed data file
            output_dir (str): Directory to save analysis results
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
        
        Returns:
            pandas.DataFrame: Loaded DataFrame
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
            
    def run_all_analyses(self):
        """
        Run all analysis methods and compile results
        
        Returns:
            dict: Compiled analysis results
        """
        if self.df is None or self.df.empty:
            logger.error("No data to analyze")
            return None
            
        logger.info("Running all analyses...")
        
        # Create analysis report file
        report_path = os.path.join(self.output_dir, "analysis_report.md")
        with open(report_path, "w") as f:
            f.write("# Play Whe Lottery Data Analysis Report\n\n")
            f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Data range: {self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}\n")
            f.write(f"Total number of draws analyzed: {len(self.df)}\n\n")
        
        # Run each analysis and append to report
        results = {}
        
        # Basic distribution analysis
        dist_results = self.analyze_number_distribution()
        results['distribution'] = dist_results
        self._append_to_report(report_path, "## Number Distribution Analysis\n\n", dist_results)
        
        # Time-based analysis
        time_results = self.analyze_time_patterns()
        results['time_patterns'] = time_results
        self._append_to_report(report_path, "## Time-Based Pattern Analysis\n\n", time_results)
        
        # Sequential pattern analysis
        seq_results = self.analyze_sequential_patterns()
        results['sequential_patterns'] = seq_results
        self._append_to_report(report_path, "## Sequential Pattern Analysis\n\n", seq_results)
        
        # Hot/cold number analysis
        hot_cold_results = self.analyze_hot_cold_numbers()
        results['hot_cold_numbers'] = hot_cold_results
        self._append_to_report(report_path, "## Hot/Cold Number Analysis\n\n", hot_cold_results)
        
        # Advanced statistical analysis
        adv_results = self.perform_advanced_statistical_analysis()
        results['advanced_statistics'] = adv_results
        self._append_to_report(report_path, "## Advanced Statistical Analysis\n\n", adv_results)
        
        # Correlation analysis
        corr_results = self.analyze_correlations()
        results['correlations'] = corr_results
        self._append_to_report(report_path, "## Correlation Analysis\n\n", corr_results)
        
        # Conclusion and insights
        conclusion = self.generate_insights(results)
        self._append_to_report(report_path, "## Conclusion and Insights\n\n", {"text": conclusion})
        
        logger.info(f"All analyses complete. Report saved to {report_path}")
        
        return results
        
    def _append_to_report(self, report_path, section_header, results):
        """
        Append analysis results to the report file
        
        Args:
            report_path (str): Path to the report file
            section_header (str): Section header text
            results (dict): Analysis results
        """
        with open(report_path, "a") as f:
            f.write(section_header)
            
            # Write text content if available
            if "text" in results:
                f.write(results["text"] + "\n\n")
                
            # Write key findings if available
            if "findings" in results:
                f.write("### Key Findings\n\n")
                for finding in results["findings"]:
                    f.write(f"- {finding}\n")
                f.write("\n")
                
            # Write statistical results if available
            if "statistics" in results:
                f.write("### Statistical Results\n\n")
                for key, value in results["statistics"].items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")
                
            # Write recommendations if available
            if "recommendations" in results:
                f.write("### Recommendations\n\n")
                for rec in results["recommendations"]:
                    f.write(f"- {rec}\n")
                f.write("\n")
                
            # Note image paths if available
            if "images" in results:
                f.write("### Visualizations\n\n")
                for img_path in results["images"]:
                    rel_path = os.path.relpath(img_path, start=os.path.dirname(report_path))
                    f.write(f"![{os.path.basename(img_path)}]({rel_path})\n\n")
                    
    def analyze_number_distribution(self):
        """
        Analyze the distribution of winning numbers
        
        Returns:
            dict: Analysis results
        """
        logger.info("Analyzing number distribution...")
        
        results = {
            "text": "This analysis examines the distribution of winning numbers in the Play Whe lottery to determine if certain numbers appear more frequently than others. A uniform distribution would suggest a fair lottery system, while significant deviations might indicate patterns that could be exploited for predictions.",
            "findings": [],
            "statistics": {},
            "images": []
        }
        
        # Count frequency of each number
        number_counts = self.df['number'].value_counts().sort_index()
        
        # Calculate expected frequency (uniform distribution)
        expected_freq = len(self.df) / 36  # 36 possible numbers (1-36)
        
        # Chi-square test for uniformity
        chi2, p_value = stats.chisquare(number_counts)
        results["statistics"]["chi_square"] = f"{chi2:.4f}"
        results["statistics"]["p_value"] = f"{p_value:.4f}"
        results["statistics"]["most_frequent_number"] = f"{number_counts.idxmax()} ({number_counts.max()} occurrences)"
        results["statistics"]["least_frequent_number"] = f"{number_counts.idxmin()} ({number_counts.min()} occurrences)"
        
        # Calculate deviation from expected
        deviation = (number_counts - expected_freq) / expected_freq * 100
        max_deviation = deviation.max()
        min_deviation = deviation.min()
        results["statistics"]["max_positive_deviation"] = f"{max_deviation:.2f}% (Number {deviation.idxmax()})"
        results["statistics"]["max_negative_deviation"] = f"{min_deviation:.2f}% (Number {deviation.idxmin()})"
        
        # Generate findings
        if p_value < 0.05:
            results["findings"].append(f"The chi-square test (p-value: {p_value:.4f}) suggests that the distribution of winning numbers is not uniform, indicating potential patterns.")
        else:
            results["findings"].append(f"The chi-square test (p-value: {p_value:.4f}) suggests that the distribution of winning numbers is uniform, consistent with a fair lottery system.")
            
        results["findings"].append(f"Number {number_counts.idxmax()} appears most frequently ({number_counts.max()} times), which is {max_deviation:.2f}% above expected frequency.")
        results["findings"].append(f"Number {number_counts.idxmin()} appears least frequently ({number_counts.min()} times), which is {abs(min_deviation):.2f}% below expected frequency.")
        
        # Create visualizations
        plt.figure(figsize=(12, 6))
        bars = plt.bar(number_counts.index, number_counts.values)
        plt.axhline(y=expected_freq, color='r', linestyle='--', label=f'Expected frequency ({expected_freq:.1f})')
        plt.xlabel('Number')
        plt.ylabel('Frequency')
        plt.title('Frequency Distribution of Play Whe Winning Numbers')
        plt.xticks(range(1, 37))
        plt.legend()
        plt.tight_layout()
        
        # Color bars based on deviation
        for i, bar in enumerate(bars):
            if number_counts.iloc[i] > expected_freq:
                bar.set_color('green')
            else:
                bar.set_color('red')
                
        # Save figure
        fig_path = os.path.join(self.output_dir, 'number_distribution.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        results["images"].append(fig_path)
        
        # Create deviation plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(deviation.index, deviation.values)
        plt.axhline(y=0, color='black', linestyle='-')
        plt.xlabel('Number')
        plt.ylabel('Deviation from Expected (%)')
        plt.title('Deviation from Expected Frequency (%)')
        plt.xticks(range(1, 37))
        plt.tight_layout()
        
        # Color bars based on deviation
        for i, bar in enumerate(bars):
            if deviation.iloc[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
                
        # Save figure
        fig_path = os.path.join(self.output_dir, 'number_deviation.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        results["images"].append(fig_path)
        
        # Add recommendations
        if p_value < 0.05:
            results["recommendations"] = [
                "Consider numbers with higher frequencies in prediction models",
                "Implement a weighted probability system based on historical frequencies",
                "Monitor if the deviation pattern persists over time"
            ]
        else:
            results["recommendations"] = [
                "Frequency-based prediction may not be effective due to uniform distribution",
                "Focus on other patterns such as time-based or sequential patterns",
                "Consider a balanced approach that doesn't heavily weight frequency"
            ]
            
        logger.info("Number distribution analysis complete")
        return results
        
    def analyze_time_patterns(self):
        """
        Analyze patterns based on draw times and days
        
        Returns:
            dict: Analysis results
        """
        logger.info("Analyzing time-based patterns...")
        
        results = {
            "text": "This analysis examines whether winning numbers show patterns based on draw times (morning, midday, afternoon, evening) or days of the week. Such patterns could indicate time-dependent biases in the lottery system or drawing process.",
            "findings": [],
            "statistics": {},
            "images": []
        }
        
        # Analyze by draw time
        time_groups = self.df.groupby('time')
        time_stats = {}
        
        for time, group in time_groups:
            # Count frequency of each number for this time
            number_counts = group['number'].value_counts().sort_index()
            
            # Calculate expected frequency (uniform distribution)
            expected_freq = len(group) / 36  # 36 possible numbers (1-36)
            
            # Chi-square test for uniformity
            chi2, p_value = stats.chisquare(number_counts)
            
            time_stats[time] = {
                "count": len(group),
                "chi_square": chi2,
                "p_value": p_value,
                "most_frequent": f"{number_counts.idxmax()} ({number_counts.max()} times)",
                "least_frequent": f"{number_counts.idxmin()} ({number_counts.min()} times)"
            }
            
            # Create visualization for each time
            plt.figure(figsize=(12, 6))
            bars = plt.bar(number_counts.index, number_counts.values)
            plt.axhline(y=expected_freq, color='r', linestyle='--', label=f'Expected frequency ({expected_freq:.1f})')
            plt.xlabel('Number')
            plt.ylabel('Frequency')
            plt.title(f'Number Distribution for {time} Draws')
            plt.xticks(range(1, 37))
            plt.legend()
            plt.tight_layout()
            
            # Save figure
            fig_path = os.path.join(self.output_dir, f'time_distribution_{time.replace(":", "")}.png')
            plt.savefig(fig_path, dpi=300)
            plt.close()
            results["images"].append(fig_path)
            
        # Analyze by day of week
        day_groups = self.df.groupby('day_of_week')
        day_stats = {}
        
        for day, group in day_groups:
            # Count frequency of each number for this day
            number_counts = group['number'].value_counts().sort_index()
            
            # Calculate expected frequency (uniform distribution)
            expected_freq = len(group) / 36  # 36 possible numbers (1-36)
            
            # Chi-square test for uniformity
            chi2, p_value = stats.chisquare(number_counts)
            
            day_stats[day] = {
                "count": len(group),
                "chi_square": chi2,
                "p_value": p_value,
                "most_frequent": f"{number_counts.idxmax()} ({number_counts.max()} times)",
                "least_frequent": f"{number_counts.idxmin()} ({number_counts.min()} times)"
            }
            
            # Create visualization for each day
            plt.figure(figsize=(12, 6))
            bars = plt.bar(number_counts.index, number_counts.values)
            plt.axhline(y=expected_freq, color='r', linestyle='--', label=f'Expected frequency ({expected_freq:.1f})')
            plt.xlabel('Number')
            plt.ylabel('Frequency')
            plt.title(f'Number Distribution for {day} Draws')
            plt.xticks(range(1, 37))
            plt.legend()
            plt.tight_layout()
            
            # Save figure

(Content truncated due to size limit. Use line ranges to read in chunks)