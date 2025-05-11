#!/usr/bin/env python3
"""
Play Whe Lottery Cultural Pattern Analysis

This script implements analysis of cultural patterns in Play Whe lottery data,
including the "Mark" system and correlation with cultural events.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/cultural_patterns.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("play_whe_cultural_patterns")

# Mark system mapping (Play Whe numbers to their cultural meanings)
MARK_SYSTEM = {
    1: "Centipede",
    2: "Old Lady",
    3: "Carriage",
    4: "Dead Man",
    5: "Horse",
    6: "Belly",
    7: "Hog",
    8: "Tiger",
    9: "Cattle",
    10: "Monkey",
    11: "Corbeau",
    12: "King",
    13: "Crapaud",
    14: "Money",
    15: "Fowl",
    16: "Jamette",
    17: "Pigeon",
    18: "Water More Than Flour",
    19: "Horse Boot",
    20: "Dog",
    21: "Mouth",
    22: "Rat",
    23: "House",
    24: "Queen",
    25: "Morocoy",
    26: "Fowl Cock",
    27: "Little Snake",
    28: "Red Fish",
    29: "Opium Man",
    30: "House Cat",
    31: "Parson",
    32: "Shrimps",
    33: "Snake",
    34: "Blind Man",
    35: "Big Snake",
    36: "Donkey"
}

# Mark categories
MARK_CATEGORIES = {
    "Animals": [1, 5, 7, 8, 9, 10, 11, 13, 15, 17, 20, 22, 25, 26, 28, 30, 32, 33, 35, 36],
    "People": [2, 4, 12, 16, 24, 29, 31, 34],
    "Objects": [3, 14, 19, 23],
    "Body Parts": [6, 21],
    "Concepts": [18, 27]
}

class CulturalPatternAnalyzer:
    """
    A class to analyze cultural patterns in Play Whe lottery data
    """
    
    def __init__(self, data_file="data/play_whe_processed.csv", events_file="data/cultural_events.csv", output_dir="analysis"):
        """
        Initialize the analyzer with configuration parameters
        
        Args:
            data_file (str): Path to the processed data file
            events_file (str): Path to the cultural events file
            output_dir (str): Directory to save analysis results
        """
        self.data_file = data_file
        self.events_file = events_file
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.df = self._load_data()
        
        # Load cultural events if file exists
        try:
            self.events_df = pd.read_csv(events_file)
            self.events_df['date'] = pd.to_datetime(self.events_df['date'])
            self.has_events = True
            logger.info(f"Loaded {len(self.events_df)} cultural events")
        except Exception as e:
            logger.warning(f"Could not load events file: {e}")
            self.events_df = None
            self.has_events = False
            
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
            
            # Add mark information
            df['mark'] = df['number'].map(MARK_SYSTEM)
            
            # Add category information
            df['category'] = df['number'].apply(self._get_category)
            
            logger.info(f"Loaded data with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
            
    def _get_category(self, number):
        """
        Get the category for a number
        
        Args:
            number (int): Play Whe number
            
        Returns:
            str: Category name
        """
        for category, numbers in MARK_CATEGORIES.items():
            if number in numbers:
                return category
        return "Unknown"
        
    def analyze_mark_patterns(self):
        """
        Analyze if certain marks appear more frequently than others
        
        Returns:
            dict: Analysis results
        """
        if self.df is None or self.df.empty:
            logger.error("No data to analyze")
            return None
            
        logger.info("Analyzing mark patterns...")
        
        # Count occurrences of each mark
        mark_counts = self.df['mark'].value_counts()
        total_draws = len(self.df)
        
        # Calculate expected count (uniform distribution)
        expected_count = total_draws / len(MARK_SYSTEM)
        
        # Calculate deviation from expected
        mark_deviation = ((mark_counts - expected_count) / expected_count * 100).round(2)
        
        # Test for statistical significance
        chi2_stat, p_value = stats.chisquare(mark_counts.values)
        
        # Generate visualization
        plt.figure(figsize=(14, 8))
        
        # Sort by deviation for better visualization
        sorted_deviation = mark_deviation.sort_values(ascending=False)
        
        # Plot deviation
        ax = sns.barplot(x=sorted_deviation.index, y=sorted_deviation.values)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xticks(rotation=90)
        plt.xlabel('Mark')
        plt.ylabel('Deviation from Expected (%)')
        plt.title('Mark Frequency Deviation from Expected')
        
        # Add value labels
        for i, v in enumerate(sorted_deviation.values):
            ax.text(i, v + (1 if v >= 0 else -3), f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, 'mark_deviation.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        results = {
            'mark_counts': mark_counts.to_dict(),
            'mark_deviation': mark_deviation.to_dict(),
            'chi2_stat': chi2_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'most_frequent_marks': mark_counts.head(5).to_dict(),
            'least_frequent_marks': mark_counts.tail(5).to_dict()
        }
        
        logger.info(f"Mark pattern analysis complete. Chi-square p-value: {p_value:.4f}")
        
        return results
        
    def analyze_category_patterns(self):
        """
        Analyze if certain categories appear more frequently than others
        
        Returns:
            dict: Analysis results
        """
        if self.df is None or self.df.empty:
            logger.error("No data to analyze")
            return None
            
        logger.info("Analyzing category patterns...")
        
        # Count occurrences of each category
        category_counts = self.df['category'].value_counts()
        total_draws = len(self.df)
        
        # Calculate expected counts based on number of marks in each category
        expected_counts = {}
        for category, numbers in MARK_CATEGORIES.items():
            expected_counts[category] = len(numbers) * total_draws / 36
            
        # Create DataFrame for analysis
        category_df = pd.DataFrame({
            'category': category_counts.index,
            'count': category_counts.values,
            'expected': [expected_counts.get(cat, 0) for cat in category_counts.index]
        })
        
        # Calculate deviation from expected
        category_df['deviation'] = ((category_df['count'] - category_df['expected']) / category_df['expected'] * 100).round(2)
        
        # Generate visualization
        plt.figure(figsize=(12, 6))
        
        # Sort by deviation for better visualization
        category_df = category_df.sort_values('deviation', ascending=False)
        
        # Plot deviation
        ax = sns.barplot(x='category', y='deviation', data=category_df)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Category')
        plt.ylabel('Deviation from Expected (%)')
        plt.title('Category Frequency Deviation from Expected')
        
        # Add value labels
        for i, v in enumerate(category_df['deviation']):
            ax.text(i, v + (1 if v >= 0 else -3), f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, 'category_deviation.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        results = {
            'category_counts': category_counts.to_dict(),
            'expected_counts': expected_counts,
            'category_deviation': category_df.set_index('category')['deviation'].to_dict(),
            'is_significant': any(abs(dev) > 10 for dev in category_df['deviation'])
        }
        
        logger.info("Category pattern analysis complete")
        
        return results
        
    def analyze_event_correlation(self):
        """
        Analyze if cultural events correlate with specific numbers
        
        Returns:
            dict: Analysis results
        """
        if self.df is None or self.df.empty:
            logger.error("No data to analyze")
            return None
            
        if not self.has_events:
            logger.warning("No events data available")
            return {'error': 'No events data available'}
            
        logger.info("Analyzing event correlation...")
        
        correlations = {}
        significant_correlations = []
        
        for _, event in self.events_df.iterrows():
            event_date = event['date']
            event_name = event['name']
            event_type = event.get('type', 'Unknown')
            
            # Look at draws within 3 days of the event
            start_date = event_date - timedelta(days=1)
            end_date = event_date + timedelta(days=1)
            
            # Get draws during this period
            event_draws = self.df[(self.df['date'] >= start_date) & (self.df['date'] <= end_date)]
            
            if len(event_draws) > 0:
                # Get most frequent numbers during this event
                event_numbers = event_draws['number'].value_counts().head(3)
                
                # Get corresponding marks
                event_marks = {num: MARK_SYSTEM[num] for num in event_numbers.index}
                
                correlation = {
                    'event_name': event_name,
                    'event_type': event_type,
                    'event_date': event_date.strftime('%Y-%m-%d'),
                    'numbers': event_numbers.to_dict(),
                    'marks': event_marks
                }
                
                correlations[event_name] = correlation
                
                # Check if any number appears more than once
                if any(count > 1 for count in event_numbers.values):
                    significant_correlations.append(correlation)
        
        # Generate visualization if we have significant correlations
        if significant_correlations:
            plt.figure(figsize=(14, 8))
            
            # Prepare data for visualization
            event_names = []
            number_lists = []
            
            for corr in significant_correlations[:10]:  # Limit to top 10 for readability
                event_names.append(corr['event_name'])
                number_lists.append(list(corr['numbers'].keys()))
            
            # Create a heatmap-like visualization
            event_matrix = np.zeros((len(event_names), 36))
            
            for i, numbers in enumerate(number_lists):
                for num in numbers:
                    event_matrix[i, int(num)-1] = 1
            
            # Plot heatmap
            sns.heatmap(event_matrix, cmap='viridis', cbar=False)
            plt.xlabel('Number')
            plt.ylabel('Event')
            plt.title('Numbers Associated with Cultural Events')
            plt.xticks(np.arange(0.5, 36.5), range(1, 37))
            plt.yticks(np.arange(0.5, len(event_names)), event_names, rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save figure
            fig_path = os.path.join(self.output_dir, 'event_correlation.png')
            plt.savefig(fig_path, dpi=300)
            plt.close()
        
        results = {
            'correlations': correlations,
            'significant_correlations': significant_correlations,
            'total_events': len(self.events_df) if self.has_events else 0,
            'events_with_correlations': len(correlations)
        }
        
        logger.info(f"Event correlation analysis complete. Found {len(significant_correlations)} significant correlations")
        
        return results
        
    def run_all_analyses(self):
        """
        Run all cultural pattern analyses
        
        Returns:
            dict: Combined analysis results
        """
        if self.df is None or self.df.empty:
            logger.error("No data to analyze")
            return None
            
        logger.info("Running all cultural pattern analyses...")
        
        # Run all analyses
        mark_results = self.analyze_mark_patterns()
        category_results = self.analyze_category_patterns()
        event_results = self.analyze_event_correlation() if self.has_events else None
        
        # Combine results
        results = {
            'mark_analysis': mark_results,
            'category_analysis': category_results,
            'event_analysis': event_results
        }
        
        # Save results to JSON
        import json
        json_path = os.path.join(self.output_dir, 'cultural_analysis.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved cultural analysis results to {json_path}")
        
        # Generate report
        self._generate_report(results)
        
        logger.info("All cultural pattern analyses complete")
        
        return results
        
    def _generate_report(self, results):
        """
        Generate a markdown report of the cultural pattern analysis
        
        Args:
            results (dict): Analysis results
        """
        report_path = os.path.join(self.output_dir, 'cultural_analysis_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Play Whe Lottery Cultural Pattern Analysis Report\n\n")
            f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if self.df is not None:
                f.write(f"Data range: {self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}\n")
                f.write(f"Total number of draws analyzed: {len(self.df)}\n\n")
            
            # Mark Analysis
            if results['mark_analysis']:
                f.write("## Mark System Analysis\n\n")
                f.write("This analysis examines whether certain 'marks' (cultural meanings assigned to numbers) appear more frequently than others in the Play Whe lottery.\n\n")
                
                f.write("### Key Findings\n\n")
                
                mark_analysis = results['mark_analysis']
                is_significant = mark_analysis['is_significant']
                
                if is_significant:
                    f.write("- The chi-square test (p-value: {:.4f}) suggests that certain marks appear significantly more often than others.\n".format(mark_analysis['p_value']))
                else:
                    f.write("- The chi-square test (p-value: {:.4f}) suggests that the distribution of marks is uniform, consistent with a fair lottery system.\n".format(mark_analysis['p_value']))
                
                # Most frequent marks
                most_frequent = sorted(mark_analysis['mark_counts'].items(), key=lambda x: x[1], reverse=True)[:5]
                f.write("- Most frequent marks:\n")
                for mark, count in most_frequent:
                    deviation = mark_analysis['mark_deviation'].get(mark, 0)
                    f.write(f"  - {mark}: {count} occurrences ({deviation:+.2f}% from expected)\n")
                
                # Least frequent marks
                least_frequent = sorted(mark_analysis['mark_counts'].items(), key=lambda x: x[1])[:5]
                f.write("- Least frequent marks:\n")
                for mark, count in least_frequent:
                    deviation = mark_analysis['mark_deviation'].get(mark, 0)
                    f.write(f"  - {mark}: {count} occurrences ({deviation:+.2f}% from expected)\n")
                
                f.write("\n### Visualizations\n\n")
                f.write("![Mark Deviation](mark_deviation.png)\n\n")
            
            # Category Analysis
            if results['category_analysis']:
                f.write("## Mark Category Analysis\n\n")
                f.write("This analysis examines whether certain categories of marks (e.g., Animals, People) appear more frequently than others.\n\n")
                
                f.write("### Key Findings\n\n")
                
                category_analysis = results['category_analysis']
                is_significant = category_analysis['is_significant']
                
                if is_significant:
                    f.write("- Certain categories of marks appear significantly more often than expected based on their representation in the mark system.\n")
                else:
                    f.write("- No significant deviation was found in the frequency of different mark categories.\n")
                
                # Category deviations
                category_deviations = sorted(category_analysis['category_deviation'].items(), key=lambda x: x[1], reverse=True)
                f.write("- Category frequency deviations from expected:\n")
                for category, deviation in category_deviations:
                    count = category_analysis['category_counts'].get(category, 0)
                    expected = category_analysis['expected_counts'].get(category, 0)
                    f.write(f"  - {category}: {count} occurrences ({deviation:+.2f}% from expected {expected:.1f})\n")
                
                f.write("\n### Visualizations\n\n")
                f.write("![Category Deviation](category_deviation.png)\n\n")
            
            # Event Correlation
            if results['event_analysis'] and 'error' not in results['event_analysis']:
                f.write("## Cultural Event Correlation\n\n")
                f.write("This analysis examines whether cultural events in Trinidad & Tobago correlate with specific Play Whe numbers.\n\n")
                
                f.write("### Key Findings\n\n")
                
                event_analysis = results['event_analysis']
                significant_correlations = event_analysis['significant_correlations']
                
                f.write(f"- Analyzed {event_analysis['total_events']} cultural events\n")
                f.write(f"- Found correlations for {event_analysis['events_with_correlations']} events\n")
                f.write(f"- Identified {len(significant_correlations)} significant correlations where specific numbers appeared multiple times around an event\n\n")
                
                if significant_correlations:
                    f.write("### Significant Event Correlations\n\n")
                    f.write("| Event | Date | Numbers | Marks |\n")
                    f.write("|-------|------|---------|-------|\n")
                    
                    for corr in significant_correlations[:10]:  # Limit to top 10 for readability
                        numbers_str = ", ".join([f"{num} ({count})" for num, count in corr['numbers'].items()])
                        marks_str = ", ".join([f"{num}: {mark}" for num, mark in corr['marks'].items()])
                        f.write(f"| {corr['event_name']} | {corr['event_date']} | {numbers_str} | {marks_str} |\n")
                    
                    f.write("\n### Visualizations\n\n")
                    f.write("![Event Correlation](event_correlation.png)\n\n")
            
            # Recommendations
            f.write("## Recommendations for Prediction Models\n\n")
            
            f.write("Based on the cultural pattern analysis, the following recommendations can be made for enhancing prediction models:\n\n")
            
            if results['mark_analysis'] and results['mark_analysis']['is_significant']:
                f.write("1. **Incorporate Mark System**: The significant variation in mark frequencies suggests that incorporating the mark system into prediction models could improve accuracy.\n")
            else:
                f.write("1. **Limited Mark System Value**: The uniform distribution of marks suggests that the mark system alone may not provide significant predictive power.\n")
            
            if results['category_analysis'] and results['category_analysis']['is_significant']:
                f.write("2. **Consider Mark Categories**: Certain categories of marks appear more frequently than others, suggesting that category-based predictions could be valuable.\n")
            
            if results['event_analysis'] and 'significant_correlations' in results['event_analysis'] and results['event_analysis']['significant_correlations']:
                f.write("3. **Event-Based Predictions**: The correlation between cultural events and specific numbers suggests that incorporating a calendar of upcoming events could enhance prediction accuracy.\n")
            
            f.write("4. **Hybrid Approach**: Combine cultural pattern insights with statistical models for a more comprehensive prediction system.\n")
            
            f.write("\n## Conclusion\n\n")
            
            if (results['mark_analysis'] and results['mark_analysis']['is_significant']) or \
               (results['category_analysis'] and results['category_analysis']['is_significant']) or \
               (results['event_analysis'] and 'significant_correlations' in results['event_analysis'] and results['event_analysis']['significant_correlations']):
                f.write("The analysis reveals that cultural patterns do play a role in the Play Whe lottery outcomes. Incorporating these cultural elements into prediction models could potentially improve their accuracy beyond purely statistical approaches.\n")
            else:
                f.write("The analysis suggests that cultural patterns have limited influence on Play Whe lottery outcomes. Traditional statistical approaches may be more reliable for prediction purposes.\n")
        
        logger.info(f"Generated cultural analysis report at {report_path}")


if __name__ == "__main__":
    # Create analyzer instance
    analyzer = CulturalPatternAnalyzer()
    
    # Run all analyses
    results = analyzer.run_all_analyses()
    
    # Print summary
    if results:
        print("Cultural pattern analysis complete")
        
        if results['mark_analysis'] and results['mark_analysis']['is_significant']:
            print("- Found significant patterns in mark frequencies")
        else:
            print("- No significant patterns found in mark frequencies")
            
        if results['category_analysis'] and results['category_analysis']['is_significant']:
            print("- Found significant patterns in mark categories")
        else:
            print("- No significant patterns found in mark categories")
            
        if results['event_analysis'] and 'significant_correlations' in results['event_analysis']:
            print(f"- Found {len(results['event_analysis']['significant_correlations'])} significant event correlations")
    else:
        print("Failed to complete cultural pattern analysis")