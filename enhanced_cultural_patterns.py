#!/usr/bin/env python3
"""
Enhanced Cultural Pattern Analyzer for Play Whe Prediction

This module implements an advanced cultural pattern analyzer that expands
the Mark system relationships, analyzes correlations between cultural events
and number frequencies, and implements a proximity effect for recent or
upcoming cultural events.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import json
from scipy import stats
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/cultural_patterns.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("enhanced_cultural_patterns")

# Mark system mapping (Play Whe numbers to their cultural meanings)
MARK_SYSTEM = {
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

# Mark categories
MARK_CATEGORIES = {
    "Animals": [1, 5, 7, 8, 9, 10, 11, 13, 15, 17, 20, 22, 25, 26, 28, 30, 32, 33, 35, 36],
    "People": [2, 4, 12, 16, 24, 29, 31, 34],
    "Objects": [3, 14, 19, 23],
    "Body Parts": [6, 21],
    "Concepts": [18, 27]
}

# Enhanced mark relationships (beyond categories)
MARK_RELATIONSHIPS = {
    # Related animals
    "Reptiles": [13, 25, 27, 33, 35],  # Crapaud, Morocoy, Little Snake, Snake, Big Snake
    "Birds": [11, 15, 17, 26],  # Corbeau, Fowl, Pigeon, Fowl Cock
    "Mammals": [5, 7, 8, 9, 10, 20, 30, 36],  # Horse, Hog, Tiger, Cattle, Monkey, Dog, House Cat, Donkey
    "Insects": [1, 22],  # Centipede, Rat
    "Aquatic": [13, 28, 32],  # Crapaud, Red Fish, Shrimps
    
    # Royalty/Authority
    "Authority": [12, 24, 31],  # King, Queen, Parson
    
    # Size relationships
    "Small": [1, 13, 22, 27, 32],  # Centipede, Crapaud, Rat, Little Snake, Shrimps
    "Large": [5, 8, 9, 35, 36],  # Horse, Tiger, Cattle, Big Snake, Donkey
    
    # Domestic vs Wild
    "Domestic": [5, 7, 15, 20, 26, 30, 36],  # Horse, Hog, Fowl, Dog, Fowl Cock, House Cat, Donkey
    "Wild": [1, 8, 9, 10, 11, 13, 17, 22, 25, 27, 28, 32, 33, 35],  # Various wild animals
    
    # Related to money/wealth
    "Wealth": [12, 14, 24],  # King, Money, Queen
    
    # Related to home/shelter
    "Shelter": [23, 30],  # House, House Cat
    
    # Related to transportation
    "Transport": [3, 5, 19, 36],  # Carriage, Horse, Horse Boot, Donkey
    
    # Related to food
    "Food": [7, 9, 13, 15, 26, 28, 32],  # Hog, Cattle, Crapaud, Fowl, Fowl Cock, Red Fish, Shrimps
    
    # Related to death/afterlife
    "Death": [4, 11],  # Dead Man, Corbeau (vulture, associated with death)
    
    # Related to religion/spirituality
    "Spiritual": [4, 31],  # Dead Man, Parson
}

# Cultural event types and their associated marks
EVENT_MARK_ASSOCIATIONS = {
    "Carnival": [10, 16, 18],  # Monkey, Jamette, Water More Than Flour
    "Easter": [4, 31],  # Dead Man, Parson
    "Christmas": [12, 24, 31],  # King, Queen, Parson
    "Divali": [14, 23],  # Money, House
    "Emancipation": [2, 4],  # Old Lady, Dead Man
    "Independence": [12, 24],  # King, Queen
    "Labour Day": [7, 9],  # Hog, Cattle (working animals)
    "Hosay": [3, 31],  # Carriage, Parson
    "Indian Arrival": [3, 5, 36],  # Carriage, Horse, Donkey (transportation)
    "Republic Day": [12, 24],  # King, Queen
}

class EnhancedCulturalPatternAnalyzer:
    """
    A class to analyze enhanced cultural patterns in Play Whe lottery data
    """
    
    def __init__(self, data_file="data/play_whe_processed.csv", events_file="data/cultural_events.csv", output_dir="analysis"):
        """
        Initialize the analyzer with configuration parameters
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
            
            # Add relationship information
            df['relationships'] = df['number'].apply(self._get_relationships)
            
            logger.info(f"Loaded data with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
            
    def _get_category(self, number):
        """Get the category for a number"""
        for category, numbers in MARK_CATEGORIES.items():
            if number in numbers:
                return category
        return "Unknown"
        
    def _get_relationships(self, number):
        """Get all relationships for a number"""
        relationships = []
        for rel_name, rel_numbers in MARK_RELATIONSHIPS.items():
            if number in rel_numbers:
                relationships.append(rel_name)
        return relationships
        
    def analyze_mark_relationships(self):
        """
        Analyze if certain mark relationships appear more frequently than others
        """
        if self.df is None or self.df.empty:
            logger.error("No data to analyze")
            return None
            
        logger.info("Analyzing mark relationship patterns...")
        
        # Explode the relationships column to get one row per relationship
        exploded_df = self.df.explode('relationships')
        
        # Remove rows with no relationships
        exploded_df = exploded_df.dropna(subset=['relationships'])
        
        # Count occurrences of each relationship
        relationship_counts = exploded_df['relationships'].value_counts()
        
        # Calculate expected counts based on number of marks in each relationship
        total_draws = len(self.df)
        expected_counts = {}
        
        for rel_name, rel_numbers in MARK_RELATIONSHIPS.items():
            # Expected probability is the number of marks in the relationship divided by total marks (36)
            expected_prob = len(rel_numbers) / 36
            expected_counts[rel_name] = expected_prob * total_draws
            
        # Calculate deviation from expected
        relationship_deviation = {}
        for rel_name, count in relationship_counts.items():
            expected = expected_counts.get(rel_name, 0)
            if expected > 0:
                deviation = (count - expected) / expected * 100
                relationship_deviation[rel_name] = deviation
                
        # Test for statistical significance
        observed = np.array([relationship_counts.get(rel, 0) for rel in MARK_RELATIONSHIPS.keys()])
        expected = np.array([expected_counts.get(rel, 0.001) for rel in MARK_RELATIONSHIPS.keys()])
        
        # Chi-square test
        chi2_stat, p_value = stats.chisquare(observed, expected)
        
        # Generate visualization
        plt.figure(figsize=(14, 8))
        
        # Sort by deviation for better visualization
        sorted_deviation = pd.Series(relationship_deviation).sort_values(ascending=False)
        
        # Plot deviation
        ax = sns.barplot(x=sorted_deviation.index, y=sorted_deviation.values)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xticks(rotation=90)
        plt.xlabel('Relationship')
        plt.ylabel('Deviation from Expected (%)')
        plt.title('Mark Relationship Frequency Deviation from Expected')
        
        # Add value labels
        for i, v in enumerate(sorted_deviation.values):
            ax.text(i, v + (1 if v >= 0 else -3), f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, 'relationship_deviation.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        results = {
            'relationship_counts': relationship_counts.to_dict(),
            'expected_counts': expected_counts,
            'relationship_deviation': relationship_deviation,
            'chi2_stat': chi2_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'most_frequent_relationships': relationship_counts.head(5).to_dict(),
            'most_overrepresented': sorted_deviation.head(5).to_dict()
        }
        
        logger.info(f"Mark relationship analysis complete. Chi-square p-value: {p_value:.4f}")
        
        return results
        
    def analyze_event_correlation(self, proximity_days=3):
        """
        Analyze correlations between cultural events and number frequencies
        with proximity effect
        """
        if self.df is None or self.df.empty or not self.has_events:
            return {'error': 'No data or events available'}
            
        logger.info(f"Analyzing event correlation with {proximity_days}-day proximity...")
        
        # Group events by type
        event_types = self.events_df['type'].unique()
        
        type_correlations = {}
        all_correlations = {}
        significant_correlations = []
        
        # Analyze each event type
        for event_type in event_types:
            type_events = self.events_df[self.events_df['type'] == event_type]
            
            # Get associated marks for this event type
            associated_marks = EVENT_MARK_ASSOCIATIONS.get(event_type, [])
            associated_numbers = [num for num, mark in MARK_SYSTEM.items() 
                                if num in associated_marks]
            
            # Collect draws around events of this type
            type_draws = []
            
            for _, event in type_events.iterrows():
                event_date = event['date']
                event_name = event['name']
                
                # Define date range around event
                start_date = event_date - timedelta(days=proximity_days)
                end_date = event_date + timedelta(days=proximity_days)
                
                # Get draws during this period
                event_draws = self.df[(self.df['date'] >= start_date) & 
                                     (self.df['date'] <= end_date)]
                
                if len(event_draws) > 0:
                    # Calculate proximity weight based on days from event
                    event_draws['days_from_event'] = abs((event_draws['date'] - event_date).dt.days)
                    event_draws['proximity_weight'] = 1 / (event_draws['days_from_event'] + 1)
                    
                    # Add event information
                    event_draws['event_name'] = event_name
                    event_draws['event_type'] = event_type
                    
                    type_draws.append(event_draws)
                    
                    # Store correlation for this specific event
                    event_numbers = event_draws['number'].value_counts()
                    
                    # Calculate weighted frequency
                    weighted_counts = {}
                    for num in range(1, 37):
                        num_draws = event_draws[event_draws['number'] == num]
                        weighted_counts[num] = num_draws['proximity_weight'].sum() if not num_draws.empty else 0
                        
                    # Get most frequent numbers
                    sorted_counts = sorted(weighted_counts.items(), key=lambda x: x[1], reverse=True)
                    top_numbers = [num for num, _ in sorted_counts[:5]]
                    
                    # Check if any associated number appears in top numbers
                    has_associated = any(num in associated_numbers for num in top_numbers)
                    
                    correlation = {
                        'event_name': event_name,
                        'event_type': event_type,
                        'event_date': event_date.strftime('%Y-%m-%d'),
                        'top_numbers': top_numbers,
                        'associated_numbers': associated_numbers,
                        'has_associated_match': has_associated,
                        'weighted_counts': {str(k): v for k, v in weighted_counts.items() if v > 0}
                    }
                    
                    all_correlations[event_name] = correlation
                    
                    # Check if correlation is significant
                    if has_associated or any(count > 2 for count in event_numbers.values):
                        significant_correlations.append(correlation)
            
            # Combine all draws for this event type
            if type_draws:
                combined_draws = pd.concat(type_draws)
                
                # Calculate weighted frequency for each number
                weighted_counts = {}
                for num in range(1, 37):
                    num_draws = combined_draws[combined_draws['number'] == num]
                    weighted_counts[num] = num_draws['proximity_weight'].sum() if not num_draws.empty else 0
                    
                # Calculate expected weighted count
                total_weight = sum(weighted_counts.values())
                expected_weight = total_weight / 36
                
                # Calculate deviation from expected
                deviation = {num: (count - expected_weight) / expected_weight * 100 
                           if expected_weight > 0 else 0
                           for num, count in weighted_counts.items()}
                
                # Store results for this event type
                type_correlations[event_type] = {
                    'weighted_counts': weighted_counts,
                    'expected_weight': expected_weight,
                    'deviation': deviation,
                    'associated_numbers': associated_numbers,
                    'total_events': len(type_events),
                    'total_draws': len(combined_draws)
                }
        
        # Generate visualization for event type correlations
        if type_correlations:
            # Create figure with subplots for each event type
            fig, axes = plt.subplots(len(type_correlations), 1, figsize=(12, 4 * len(type_correlations)))
            
            if len(type_correlations) == 1:
                axes = [axes]
                
            for i, (event_type, data) in enumerate(type_correlations.items()):
                numbers = list(range(1, 37))
                deviations = [data['deviation'].get(num, 0) for num in numbers]
                
                # Highlight associated numbers
                colors = ['red' if num in data['associated_numbers'] else 'blue' for num in numbers]
                
                axes[i].bar(numbers, deviations, color=colors)
                axes[i].axhline(y=0, color='r', linestyle='--')
                axes[i].set_xlabel('Number')
                axes[i].set_ylabel('Deviation (%)')
                axes[i].set_title(f'{event_type} Events ({data["total_events"]} events, {data["total_draws"]} draws)')
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='red', label='Associated Number'),
                    Patch(facecolor='blue', label='Other Number')
                ]
                axes[i].legend(handles=legend_elements)
                
            plt.tight_layout()
            
            # Save figure
            fig_path = os.path.join(self.output_dir, 'event_type_correlation.png')
            plt.savefig(fig_path, dpi=300)
            plt.close()
        
        results = {
            'type_correlations': type_correlations,
            'all_correlations': all_correlations,
            'significant_correlations': significant_correlations,
            'proximity_days': proximity_days,
            'total_events': len(self.events_df) if self.has_events else 0,
            'events_with_correlations': len(all_correlations)
        }
        
        logger.info(f"Event correlation analysis complete. Found {len(significant_correlations)} significant correlations")
        
        return results
        
    def calculate_proximity_weights(self, target_date=None, proximity_days=7):
        """
        Calculate proximity weights for all numbers based on upcoming and recent events
        """
        if not self.has_events:
            logger.warning("No events data available")
            return {num: 1.0 for num in range(1, 37)}
            
        if target_date is None:
            target_date = datetime.now()
        elif isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
            
        logger.info(f"Calculating proximity weights for {target_date.strftime('%Y-%m-%d')}")
        
        # Define date range around target date
        start_date = target_date - timedelta(days=proximity_days)
        end_date = target_date + timedelta(days=proximity_days)
        
        # Get events in this period
        nearby_events = self.events_df[(self.events_df['date'] >= start_date) & 
                                      (self.events_df['date'] <= end_date)]
        
        if len(nearby_events) == 0:
            logger.info("No events found in proximity period")
            return {num: 1.0 for num in range(1, 37)}
            
        # Initialize weights
        proximity_weights = {num: 1.0 for num in range(1, 37)}
        
        # Calculate weights for each event
        for _, event in nearby_events.iterrows():
            event_date = event['date']
            event_type = event['type']
            
            # Calculate days from target date
            days_diff = abs((event_date - target_date).dt.days[0] 
                          if isinstance((event_date - target_date).dt.days, pd.Series) 
                          else abs((event_date - target_date).days))
            
            # Calculate base weight (higher for closer events)
            base_weight = 1 + (proximity_days - days_diff) / proximity_days
            
            # Get associated numbers for this event type
            associated_numbers = EVENT_MARK_ASSOCIATIONS.get(event_type, [])
            
            # Apply weights to associated numbers
            for num in range(1, 37):
                if num in associated_numbers:
                    # Higher weight for associated numbers
                    proximity_weights[num] *= base_weight * 1.5
                else:
                    # Slight boost for all numbers (events increase overall activity)
                    proximity_weights[num] *= 1.05
                    
        # Normalize weights
        max_weight = max(proximity_weights.values())
        proximity_weights = {num: weight / max_weight for num, weight in proximity_weights.items()}
        
        return proximity_weights
        
    def predict_with_cultural_patterns(self, base_probabilities, target_date=None, proximity_days=7):
        """
        Adjust base probabilities using cultural pattern insights
        """
        if self.df is None or self.df.empty:
            logger.error("No data to analyze")
            return base_probabilities
            
        logger.info("Adjusting probabilities with cultural patterns...")
        
        # Calculate proximity weights
        proximity_weights = self.calculate_proximity_weights(target_date, proximity_days)
        
        # Apply proximity weights to base probabilities
        adjusted_probabilities = {}
        
        for num in range(1, 37):
            base_prob = base_probabilities.get(num, 1/36)
            proximity_weight = proximity_weights.get(num, 1.0)
            
            # Apply weight
            adjusted_prob = base_prob * proximity_weight
            adjusted_probabilities[num] = adjusted_prob
            
        # Normalize to ensure probabilities sum to 1
        total_prob = sum(adjusted_probabilities.values())
        adjusted_probabilities = {num: prob / total_prob 
                                for num, prob in adjusted_probabilities.items()}
        
        return adjusted_probabilities
        
    def run_all_analyses(self):
        """
        Run all cultural pattern analyses
        """
        if self.df is None or self.df.empty:
            logger.error("No data to analyze")
            return None
            
        logger.info("Running all enhanced cultural pattern analyses...")
        
        # Run all analyses
        relationship_results = self.analyze_mark_relationships()
        event_results = self.analyze_event_correlation() if self.has_events else None
        
        # Calculate current proximity weights
        proximity_weights = self.calculate_proximity_weights()
        
        # Combine results
        results = {
            'relationship_analysis': relationship_results,
            'event_analysis': event_results,
            'current_proximity_weights': proximity_weights
        }
        
        # Save results to JSON
        json_path = os.path.join(self.output_dir, 'enhanced_cultural_analysis.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved enhanced cultural analysis results to {json_path}")
        
        logger.info("All enhanced cultural pattern analyses complete")
        
        return results

if __name__ == "__main__":
    # Create analyzer instance
    analyzer = EnhancedCulturalPatternAnalyzer()
    
    # Run all analyses
    results = analyzer.run_all_analyses()
    
    # Print summary
    if results:
        print("Enhanced cultural pattern analysis complete")
        
        if results['relationship_analysis'] and results['relationship_analysis']['is_significant']:
            print("- Found significant patterns in mark relationships")
        else:
            print("- No significant patterns found in mark relationships")
            
        if results['event_analysis'] and 'significant_correlations' in results['event_analysis']:
            print(f"- Found {len(results['event_analysis']['significant_correlations'])} significant event correlations")
        else:
            print("- No significant event correlations found")
    else:
        print("Failed to complete enhanced cultural pattern analysis")
