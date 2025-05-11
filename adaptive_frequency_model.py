#!/usr/bin/env python3
"""
Adaptive Time-Weighted Frequency Model for Play Whe Prediction

This module implements an advanced frequency model that uses multiple
half-life values and automatically determines the optimal weighting
for each draw period based on performance.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/adaptive_model.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("adaptive_frequency_model")

class AdaptiveFrequencyModel:
    """
    A class implementing an adaptive time-weighted frequency model
    """
    
    def __init__(self, data_file="data/play_whe_processed.csv", output_dir="models"):
        """
        Initialize the model with configuration parameters
        
        Args:
            data_file (str): Path to the processed data file
            output_dir (str): Directory to save model results
        """
        self.data_file = data_file
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define half-life values to test (in days)
        self.half_life_values = [7, 30, 90, 180, 365]
        
        # Initialize performance tracking
        self.performance_history = {}
        self.optimal_half_life = {}
        
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
            
    def _calculate_decay_weights(self, dates, half_life):
        """
        Calculate exponential decay weights based on half-life
        
        Args:
            dates (pandas.Series): Series of dates
            half_life (int): Half-life in days
            
        Returns:
            numpy.ndarray: Array of weights
        """
        # Calculate days from most recent date
        max_date = dates.max()
        days_from_max = (max_date - dates).dt.days
        
        # Calculate decay constant
        decay_constant = np.log(2) / half_life
        
        # Calculate weights
        weights = np.exp(-decay_constant * days_from_max)
        
        return weights
        
    def build_model_for_period(self, period, half_life=30):
        """
        Build a time-weighted frequency model for a specific draw period
        
        Args:
            period (str): Draw period ('morning', 'midday', 'afternoon', 'evening')
            half_life (int): Half-life in days for exponential decay
            
        Returns:
            dict: Model details and parameters
        """
        if self.df is None or self.df.empty:
            logger.error("No data to build model")
            return None
            
        # Filter data for the specified period
        period_df = self.df[self.df['draw_period'] == period].copy()
        
        if len(period_df) == 0:
            logger.warning(f"No data for period {period}")
            return None
            
        # Calculate weights based on half-life
        period_df['weight'] = self._calculate_decay_weights(period_df['date'], half_life)
        
        # Calculate weighted frequencies
        weighted_counts = {}
        for num in range(1, 37):
            num_df = period_df[period_df['number'] == num]
            weighted_counts[num] = num_df['weight'].sum() if not num_df.empty else 0
            
        total_weight = period_df['weight'].sum()
        probabilities = {num: count/total_weight if total_weight > 0 else 0 
                       for num, count in weighted_counts.items()}
        
        # Calculate confidence scores based on deviation from expected probability
        expected_prob = 1/36
        confidence_scores = {num: min(abs(prob - expected_prob) / expected_prob * 100, 100) 
                           for num, prob in probabilities.items()}
        
        # Calculate statistical significance
        observed_counts = np.array([weighted_counts.get(num, 0) for num in range(1, 37)])
        expected_counts = np.full(36, total_weight / 36)
        
        # Chi-square test
        chi2_stat, p_value = stats.chisquare(observed_counts, expected_counts)
        
        # Create model
        model = {
            'name': f'Adaptive Frequency Model ({period}, HL={half_life}d)',
            'description': f'Time-weighted frequency model for {period} period with {half_life}-day half-life',
            'period': period,
            'half_life': half_life,
            'probabilities': probabilities,
            'confidence_scores': confidence_scores,
            'chi2_stat': chi2_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'sample_size': len(period_df),
            'effective_sample_size': period_df['weight'].sum()
        }
        
        return model
        
    def evaluate_half_life_values(self):
        """
        Evaluate different half-life values for each draw period
        
        Returns:
            dict: Evaluation results
        """
        if self.df is None or self.df.empty:
            logger.error("No data to evaluate")
            return None
            
        logger.info("Evaluating half-life values...")
        
        # Get unique draw periods
        draw_periods = self.df['draw_period'].unique()
        
        # Remove 'unknown' if present
        if 'unknown' in draw_periods:
            draw_periods = [p for p in draw_periods if p != 'unknown']
            
        results = {}
        
        for period in draw_periods:
            period_results = {}
            
            for half_life in self.half_life_values:
                # Build model with this half-life
                model = self.build_model_for_period(period, half_life)
                
                if model is None:
                    continue
                    
                # Store results
                period_results[half_life] = {
                    'chi2_stat': model['chi2_stat'],
                    'p_value': model['p_value'],
                    'is_significant': model['is_significant'],
                    'effective_sample_size': model['effective_sample_size']
                }
                
            # Find optimal half-life (lowest p-value indicates strongest pattern)
            if period_results:
                optimal_half_life = min(period_results.items(), 
                                      key=lambda x: x[1]['p_value'])[0]
                
                self.optimal_half_life[period] = optimal_half_life
                
                # Store results
                results[period] = {
                    'half_life_results': period_results,
                    'optimal_half_life': optimal_half_life
                }
                
                logger.info(f"Optimal half-life for {period} period: {optimal_half_life} days")
            
        # Generate visualization
        self._visualize_half_life_evaluation(results)
        
        return results
        
    def _visualize_half_life_evaluation(self, results):
        """
        Visualize half-life evaluation results
        
        Args:
            results (dict): Evaluation results
        """
        if not results:
            return
            
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot p-values for each period and half-life
        for i, (period, period_results) in enumerate(results.items()):
            half_lives = list(period_results['half_life_results'].keys())
            p_values = [period_results['half_life_results'][hl]['p_value'] for hl in half_lives]
            
            plt.subplot(2, 2, i+1)
            plt.plot(half_lives, p_values, 'o-')
            plt.axvline(x=period_results['optimal_half_life'], color='r', linestyle='--', 
                       label=f'Optimal: {period_results["optimal_half_life"]} days')
            plt.xlabel('Half-Life (days)')
            plt.ylabel('p-value')
            plt.title(f'{period.capitalize()} Period')
            plt.xscale('log')
            plt.grid(True)
            plt.legend()
            
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, 'half_life_evaluation.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved half-life evaluation visualization to {fig_path}")
        
    def build_adaptive_model(self):
        """
        Build an adaptive model using optimal half-life values for each period
        
        Returns:
            dict: Adaptive model
        """
        if self.df is None or self.df.empty:
            logger.error("No data to build model")
            return None
            
        # If optimal half-life values not determined, evaluate them
        if not self.optimal_half_life:
            self.evaluate_half_life_values()
            
        if not self.optimal_half_life:
            logger.error("Failed to determine optimal half-life values")
            return None
            
        logger.info("Building adaptive frequency model...")
        
        # Build models for each period with optimal half-life
        period_models = {}
        
        for period, half_life in self.optimal_half_life.items():
            model = self.build_model_for_period(period, half_life)
            
            if model is not None:
                period_models[period] = model
                
        # Create adaptive model
        adaptive_model = {
            'name': 'Adaptive Time-Weighted Frequency Model',
            'description': 'Frequency model with optimal half-life values for each draw period',
            'period_models': period_models,
            'optimal_half_life': self.optimal_half_life,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Generate visualization
        self._visualize_adaptive_model(adaptive_model)
        
        # Save model
        model_path = os.path.join(self.output_dir, 'adaptive_frequency_model.json')
        with open(model_path, 'w') as f:
            json.dump(adaptive_model, f, indent=2, default=str)
            
        logger.info(f"Saved adaptive frequency model to {model_path}")
        
        return adaptive_model
        
    def _visualize_adaptive_model(self, model):
        """
        Visualize the adaptive model
        
        Args:
            model (dict): Adaptive model
        """
        if not model or 'period_models' not in model:
            return
            
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Plot probability distribution for each period
        for i, (period, period_model) in enumerate(model['period_models'].items()):
            numbers = list(range(1, 37))
            probs = [period_model['probabilities'].get(num, 0) for num in numbers]
            
            plt.subplot(2, 2, i+1)
            bars = plt.bar(numbers, probs, alpha=0.7)
            
            # Color bars based on significance
            expected_prob = 1/36
            for j, bar in enumerate(bars):
                if probs[j] > expected_prob:
                    bar.set_color('red')  # Above expected
                else:
                    bar.set_color('blue')  # Below expected
                    
            plt.axhline(y=expected_prob, color='r', linestyle='--', 
                       label=f'Expected ({expected_prob:.4f})')
            
            plt.xlabel('Number')
            plt.ylabel('Probability')
            plt.title(f'{period.capitalize()} Period (HL={period_model["half_life"]}d, p={period_model["p_value"]:.4f})')
            plt.legend()
            
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, 'adaptive_model_probabilities.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved adaptive model visualization to {fig_path}")
        
    def predict(self, period, n=5):
        """
        Make predictions using the adaptive model
        
        Args:
            period (str): Draw period ('morning', 'midday', 'afternoon', 'evening')
            n (int): Number of predictions to return
            
        Returns:
            list: List of (number, probability, confidence) tuples
        """
        # Load model if not built
        model_path = os.path.join(self.output_dir, 'adaptive_frequency_model.json')
        
        if os.path.exists(model_path):
            with open(model_path, 'r') as f:
                model = json.load(f)
        else:
            model = self.build_adaptive_model()
            
        if model is None or 'period_models' not in model:
            logger.error("No model available for prediction")
            return None
            
        # Check if period exists in model
        if period not in model['period_models']:
            logger.warning(f"Period {period} not found in model, using morning instead")
            period = 'morning'
            
        # Get period model
        period_model = model['period_models'][period]
        
        # Sort numbers by probability
        sorted_probs = sorted(period_model['probabilities'].items(), 
                             key=lambda x: float(x[1]), reverse=True)
        
        # Get top n predictions with confidence scores
        predictions = [(int(num), float(prob), float(period_model['confidence_scores'][num])) 
                      for num, prob in sorted_probs[:n]]
        
        return predictions
        
    def update_model(self, new_data):
        """
        Update the adaptive model with new data
        
        Args:
            new_data (pandas.DataFrame): New data to incorporate
            
        Returns:
            dict: Updated adaptive model
        """
        if new_data is None or new_data.empty:
            logger.error("No new data to update model")
            return None
            
        logger.info(f"Updating adaptive model with {len(new_data)} new records")
        
        # Combine existing data with new data
        if self.df is not None:
            self.df = pd.concat([self.df, new_data], ignore_index=True)
            self.df = self.df.drop_duplicates()
        else:
            self.df = new_data
            
        # Re-evaluate optimal half-life values
        self.evaluate_half_life_values()
        
        # Rebuild adaptive model
        updated_model = self.build_adaptive_model()
        
        return updated_model
        
    def track_performance(self, period, predictions, actual_number):
        """
        Track prediction performance for a specific period
        
        Args:
            period (str): Draw period
            predictions (list): List of (number, probability, confidence) tuples
            actual_number (int): Actual winning number
            
        Returns:
            dict: Performance metrics
        """
        if not predictions:
            logger.error("No predictions to evaluate")
            return None
            
        logger.info(f"Tracking performance for {period} period")
        
        # Extract predicted numbers
        predicted_numbers = [num for num, _, _ in predictions]
        
        # Check if actual number is in predictions
        is_correct = actual_number in predicted_numbers
        
        # Calculate rank of actual number
        if is_correct:
            rank = predicted_numbers.index(actual_number) + 1
        else:
            rank = None
            
        # Calculate probability assigned to actual number
        for num, prob, _ in predictions:
            if num == actual_number:
                assigned_probability = prob
                break
        else:
            # If actual number not in top predictions, find its probability in the model
            model_path = os.path.join(self.output_dir, 'adaptive_frequency_model.json')
            
            if os.path.exists(model_path):
                with open(model_path, 'r') as f:
                    model = json.load(f)
                    
                if period in model['period_models']:
                    assigned_probability = float(model['period_models'][period]['probabilities'].get(str(actual_number), 0))
                else:
                    assigned_probability = 0
            else:
                assigned_probability = 0
                
        # Create performance metrics
        metrics = {
            'period': period,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'actual_number': actual_number,
            'predictions': predicted_numbers,
            'is_correct': is_correct,
            'rank': rank,
            'assigned_probability': assigned_probability
        }
        
        # Update performance history
        if period not in self.performance_history:
            self.performance_history[period] = []
            
        self.performance_history[period].append(metrics)
        
        # Save performance history
        history_path = os.path.join(self.output_dir, 'performance_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.performance_history, f, indent=2, default=str)
            
        logger.info(f"Updated performance history at {history_path}")
        
        return metrics
        
    def analyze_performance(self):
        """
        Analyze prediction performance across periods
        
        Returns:
            dict: Performance analysis
        """
        # Load performance history if not in memory
        history_path = os.path.join(self.output_dir, 'performance_history.json')
        
        if os.path.exists(history_path) and not self.performance_history:
            with open(history_path, 'r') as f:
                self.performance_history = json.load(f)
                
        if not self.performance_history:
            logger.warning("No performance history available for analysis")
            return None
            
        logger.info("Analyzing prediction performance...")
        
        analysis = {}
        
        for period, history in self.performance_history.items():
            if not history:
                continue
                
            # Calculate accuracy (percentage of correct predictions)
            correct_count = sum(1 for entry in history if entry['is_correct'])
            accuracy = correct_count / len(history) if history else 0
            
            # Calculate average rank of correct predictions
            ranks = [entry['rank'] for entry in history if entry['rank'] is not None]
            avg_rank = sum(ranks) / len(ranks) if ranks else None
            
            # Calculate average assigned probability to actual number
            avg_probability = sum(entry['assigned_probability'] for entry in history) / len(history)
            
            # Calculate expected accuracy based on top-n predictions
            n = len(history[0]['predictions']) if history else 5
            expected_accuracy = n / 36
            
            # Calculate improvement over random guessing
            improvement = (accuracy - expected_accuracy) / expected_accuracy * 100 if expected_accuracy > 0 else 0
            
            # Store analysis
            analysis[period] = {
                'accuracy': accuracy,
                'correct_count': correct_count,
                'total_predictions': len(history),
                'average_rank': avg_rank,
                'average_probability': avg_probability,
                'expected_accuracy': expected_accuracy,
                'improvement': improvement
            }
            
        # Generate visualization
        self._visualize_performance(analysis)
        
        return analysis
        
    def _visualize_performance(self, analysis):
        """
        Visualize prediction performance
        
        Args:
            analysis (dict): Performance analysis
        """
        if not analysis:
            return
            
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot accuracy by period
        periods = list(analysis.keys())
        accuracies = [analysis[period]['accuracy'] * 100 for period in periods]
        expected = [analysis[period]['expected_accuracy'] * 100 for period in periods]
        
        x = np.arange(len(periods))
        width = 0.35
        
        ax = plt.subplot(111)
        bars1 = ax.bar(x - width/2, accuracies, width, label='Actual Accuracy')
        bars2 = ax.bar(x + width/2, expected, width, label='Expected (Random)')
        
        # Add value labels
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom')
            
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        ax.set_xlabel('Draw Period')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Prediction Accuracy by Draw Period')
        ax.set_xticks(x)
        ax.set_xticklabels([p.capitalize() for p in periods])
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, 'prediction_performance.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved performance visualization to {fig_path}")

if __name__ == "__main__":
    # Create model instance
    model = AdaptiveFrequencyModel()
    
    # Evaluate half-life values
    results = model.evaluate_half_life_values()
    
    # Build adaptive model
    adaptive_model = model.build_adaptive_model()
    
    # Make predictions for each period
    for period in ['morning', 'midday', 'afternoon', 'evening']:
        predictions = model.predict(period, n=5)
        
        if predictions:
            print(f"\n{period.capitalize()} Period Predictions:")
            for i, (num, prob, conf) in enumerate(predictions, 1):
                print(f"{i}. Number {num}: Probability {prob:.6f}, Confidence {conf:.2f}%")