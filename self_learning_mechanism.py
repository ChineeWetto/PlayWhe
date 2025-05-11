#!/usr/bin/env python3
"""
Self-Learning Mechanism for Play Whe Prediction

This module implements a self-learning mechanism that tracks prediction performance
for each model component, automatically adjusts model weights based on recent
performance, and implements a reinforcement learning approach that strengthens
successful prediction strategies.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import json
from collections import defaultdict, deque
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/self_learning.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("self_learning_mechanism")

class SelfLearningMechanism:
    """
    A class implementing a self-learning mechanism for prediction models
    """
    
    def __init__(self, output_dir="models", history_length=30, learning_rate=0.1):
        """
        Initialize the self-learning mechanism
        
        Args:
            output_dir (str): Directory to save model results
            history_length (int): Number of predictions to keep in history
            learning_rate (float): Rate at which to adjust weights (0-1)
        """
        self.output_dir = output_dir
        self.history_length = history_length
        self.learning_rate = learning_rate
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model weights
        self.model_weights = {
            'frequency': 0.2,
            'time_sensitive_frequency': 0.5,
            'sequential': 0.2,
            'hot_cold': 0.1,
            'cultural': 0.0  # Start with zero weight, will adjust based on performance
        }
        
        # Initialize performance history
        self.performance_history = {
            'frequency': deque(maxlen=history_length),
            'time_sensitive_frequency': deque(maxlen=history_length),
            'sequential': deque(maxlen=history_length),
            'hot_cold': deque(maxlen=history_length),
            'cultural': deque(maxlen=history_length),
            'combined': deque(maxlen=history_length)
        }
        
        # Initialize success counters
        self.success_counters = {
            'frequency': 0,
            'time_sensitive_frequency': 0,
            'sequential': 0,
            'hot_cold': 0,
            'cultural': 0,
            'combined': 0
        }
        
        # Initialize total prediction counters
        self.total_predictions = {
            'frequency': 0,
            'time_sensitive_frequency': 0,
            'sequential': 0,
            'hot_cold': 0,
            'cultural': 0,
            'combined': 0
        }
        
        # Load existing data if available
        self._load_state()
        
    def _load_state(self):
        """
        Load existing state from files
        """
        # Load model weights
        weights_path = os.path.join(self.output_dir, 'model_weights.json')
        if os.path.exists(weights_path):
            try:
                with open(weights_path, 'r') as f:
                    self.model_weights = json.load(f)
                logger.info(f"Loaded model weights from {weights_path}")
            except Exception as e:
                logger.error(f"Error loading model weights: {e}")
                
        # Load performance history
        history_path = os.path.join(self.output_dir, 'performance_history.pkl')
        if os.path.exists(history_path):
            try:
                with open(history_path, 'rb') as f:
                    history_data = pickle.load(f)
                    self.performance_history = history_data.get('history', self.performance_history)
                    self.success_counters = history_data.get('success', self.success_counters)
                    self.total_predictions = history_data.get('total', self.total_predictions)
                logger.info(f"Loaded performance history from {history_path}")
            except Exception as e:
                logger.error(f"Error loading performance history: {e}")
                
    def _save_state(self):
        """
        Save current state to files
        """
        # Save model weights
        weights_path = os.path.join(self.output_dir, 'model_weights.json')
        try:
            with open(weights_path, 'w') as f:
                json.dump(self.model_weights, f, indent=2)
            logger.info(f"Saved model weights to {weights_path}")
        except Exception as e:
            logger.error(f"Error saving model weights: {e}")
            
        # Save performance history
        history_path = os.path.join(self.output_dir, 'performance_history.pkl')
        try:
            history_data = {
                'history': self.performance_history,
                'success': self.success_counters,
                'total': self.total_predictions
            }
            with open(history_path, 'wb') as f:
                pickle.dump(history_data, f)
            logger.info(f"Saved performance history to {history_path}")
        except Exception as e:
            logger.error(f"Error saving performance history: {e}")
            
    def track_prediction(self, model_name, predictions, actual_number):
        """
        Track the performance of a prediction
        
        Args:
            model_name (str): Name of the model
            predictions (list): List of predicted numbers
            actual_number (int): Actual winning number
            
        Returns:
            bool: Whether the prediction was successful
        """
        if model_name not in self.performance_history:
            logger.warning(f"Unknown model: {model_name}")
            return False
            
        # Check if prediction was successful
        success = actual_number in predictions
        
        # Record result
        self.performance_history[model_name].append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'predictions': predictions,
            'actual': actual_number,
            'success': success
        })
        
        # Update counters
        self.total_predictions[model_name] += 1
        if success:
            self.success_counters[model_name] += 1
            
        logger.info(f"Tracked prediction for {model_name}: {'Success' if success else 'Failure'}")
        
        # Save state
        self._save_state()
        
        return success
        
    def track_combined_prediction(self, combined_predictions, model_predictions, actual_number):
        """
        Track the performance of the combined prediction and all component models
        
        Args:
            combined_predictions (list): List of combined predicted numbers
            model_predictions (dict): Dictionary of model name to predictions
            actual_number (int): Actual winning number
            
        Returns:
            dict: Success results for each model
        """
        results = {}
        
        # Track combined prediction
        combined_success = self.track_prediction('combined', combined_predictions, actual_number)
        results['combined'] = combined_success
        
        # Track individual model predictions
        for model_name, predictions in model_predictions.items():
            if isinstance(predictions, list):
                success = self.track_prediction(model_name, predictions, actual_number)
                results[model_name] = success
                
        return results
        
    def adjust_weights(self):
        """
        Adjust model weights based on recent performance
        
        Returns:
            dict: Updated model weights
        """
        logger.info("Adjusting model weights based on recent performance...")
        
        # Calculate success rates for each model
        success_rates = {}
        
        for model_name in self.model_weights.keys():
            if model_name in self.total_predictions and self.total_predictions[model_name] > 0:
                success_rates[model_name] = self.success_counters[model_name] / self.total_predictions[model_name]
            else:
                success_rates[model_name] = 0
                
        # Calculate total success rate
        total_success_rate = sum(success_rates.values())
        
        if total_success_rate > 0:
            # Calculate new weights based on relative performance
            new_weights = {model: rate / total_success_rate for model, rate in success_rates.items()}
            
            # Apply learning rate to smooth the transition
            for model in self.model_weights:
                if model in new_weights:
                    self.model_weights[model] = (1 - self.learning_rate) * self.model_weights[model] + \
                                              self.learning_rate * new_weights[model]
                    
            # Ensure weights sum to 1
            weight_sum = sum(self.model_weights.values())
            if weight_sum > 0:
                self.model_weights = {model: weight / weight_sum for model, weight in self.model_weights.items()}
                
            logger.info(f"Adjusted weights: {self.model_weights}")
            
            # Save updated weights
            self._save_state()
            
            # Generate visualization
            self._visualize_weights()
            
        return self.model_weights
        
    def _visualize_weights(self):
        """
        Generate visualization of model weights
        """
        plt.figure(figsize=(10, 6))
        
        # Plot weights
        models = list(self.model_weights.keys())
        weights = [self.model_weights[model] for model in models]
        
        bars = plt.bar(models, weights)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
            
        plt.xlabel('Model')
        plt.ylabel('Weight')
        plt.title('Self-Adjusted Model Weights')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, 'self_adjusted_weights.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved weights visualization to {fig_path}")
        
    def visualize_performance(self):
        """
        Generate visualization of model performance
        """
        # Calculate success rates
        success_rates = {}
        
        for model_name in self.model_weights.keys():
            if model_name in self.total_predictions and self.total_predictions[model_name] > 0:
                success_rates[model_name] = self.success_counters[model_name] / self.total_predictions[model_name] * 100
            else:
                success_rates[model_name] = 0
                
        # Add combined model
        if self.total_predictions['combined'] > 0:
            success_rates['combined'] = self.success_counters['combined'] / self.total_predictions['combined'] * 100
            
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot success rates
        models = list(success_rates.keys())
        rates = [success_rates[model] for model in models]
        
        # Expected random success rate (5 predictions out of 36 numbers)
        expected_rate = 5 / 36 * 100
        
        bars = plt.bar(models, rates)
        plt.axhline(y=expected_rate, color='r', linestyle='--', 
                   label=f'Expected random ({expected_rate:.1f}%)')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom')
            
        plt.xlabel('Model')
        plt.ylabel('Success Rate (%)')
        plt.title('Model Performance Comparison')
        plt.ylim(0, max(rates) * 1.2 if rates else 20)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, 'model_performance.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved performance visualization to {fig_path}")
        
        return success_rates
        
    def get_current_weights(self):
        """
        Get the current model weights
        
        Returns:
            dict: Current model weights
        """
        return self.model_weights
        
    def combine_predictions(self, model_predictions, top_n=5):
        """
        Combine predictions from multiple models using current weights
        
        Args:
            model_predictions (dict): Dictionary of model name to predictions
            top_n (int): Number of top predictions to return
            
        Returns:
            list: Combined top predictions
        """
        # Initialize scores for all numbers
        combined_scores = {num: 0.0 for num in range(1, 37)}
        
        # Apply weighted voting
        for model_name, predictions in model_predictions.items():
            if model_name in self.model_weights:
                weight = self.model_weights[model_name]
                
                # Convert predictions to a list of numbers if it's a list of tuples
                if isinstance(predictions, list) and predictions and isinstance(predictions[0], tuple):
                    pred_numbers = [num for num, _, _ in predictions]
                else:
                    pred_numbers = predictions
                    
                # Apply weight to each predicted number
                for i, num in enumerate(pred_numbers):
                    # Higher score for higher ranked predictions
                    position_weight = 1.0 / (i + 1)
                    combined_scores[num] += weight * position_weight
                    
        # Sort by score
        sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N predictions
        return [num for num, _ in sorted_scores[:top_n]]
        
    def analyze_performance_trends(self):
        """
        Analyze trends in model performance over time
        
        Returns:
            dict: Performance trend analysis
        """
        # Check if we have enough history
        if not all(len(history) > 0 for history in self.performance_history.values()):
            logger.warning("Not enough history for trend analysis")
            return None
            
        logger.info("Analyzing performance trends...")
        
        # Calculate rolling success rates
        window_sizes = [5, 10, 20]
        trend_analysis = {}
        
        for model_name, history in self.performance_history.items():
            if not history:
                continue
                
            # Convert history to DataFrame
            df = pd.DataFrame(list(history))
            
            if 'date' in df.columns and 'success' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                
                # Calculate rolling success rates
                model_trends = {}
                
                for window in window_sizes:
                    if len(df) >= window:
                        df[f'rolling_{window}'] = df['success'].rolling(window=window).mean()
                        
                        # Get latest value
                        latest = df[f'rolling_{window}'].iloc[-1]
                        
                        # Calculate trend (positive or negative)
                        if len(df) >= window * 2:
                            previous = df[f'rolling_{window}'].iloc[-window-1]
                            trend = (latest - previous) / previous if previous > 0 else 0
                        else:
                            trend = 0
                            
                        model_trends[window] = {
                            'latest_rate': latest,
                            'trend': trend
                        }
                        
                trend_analysis[model_name] = model_trends
                
        # Generate visualization
        if trend_analysis:
            plt.figure(figsize=(12, 8))
            
            for i, model_name in enumerate(trend_analysis.keys()):
                plt.subplot(len(trend_analysis), 1, i+1)
                
                # Convert history to DataFrame for plotting
                df = pd.DataFrame(list(self.performance_history[model_name]))
                
                if 'date' in df.columns and 'success' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    
                    # Calculate rolling success rates for plotting
                    for window in window_sizes:
                        if len(df) >= window:
                            df[f'rolling_{window}'] = df['success'].rolling(window=window).mean()
                            plt.plot(df['date'], df[f'rolling_{window}'], 
                                   label=f'{window}-draw rolling average')
                            
                    plt.title(f'{model_name.capitalize()} Model Performance Trend')
                    plt.xlabel('Date')
                    plt.ylabel('Success Rate')
                    plt.ylim(0, 1)
                    plt.legend()
                    
            plt.tight_layout()
            
            # Save figure
            fig_path = os.path.join(self.output_dir, 'performance_trends.png')
            plt.savefig(fig_path, dpi=300)
            plt.close()
            
            logger.info(f"Saved performance trends visualization to {fig_path}")
            
        return trend_analysis
        
    def reinforcement_learning_update(self, model_name, success, learning_rate=None):
        """
        Update model weights using reinforcement learning approach
        
        Args:
            model_name (str): Name of the model to update
            success (bool): Whether the prediction was successful
            learning_rate (float, optional): Custom learning rate for this update
            
        Returns:
            dict: Updated model weights
        """
        if model_name not in self.model_weights:
            logger.warning(f"Unknown model: {model_name}")
            return self.model_weights
            
        # Use provided learning rate or default
        lr = learning_rate if learning_rate is not None else self.learning_rate
        
        logger.info(f"Applying reinforcement learning update for {model_name} (success={success})")
        
        # Calculate reward (positive for success, negative for failure)
        reward = 1.0 if success else -0.5
        
        # Update weight for this model
        self.model_weights[model_name] += lr * reward
        
        # Ensure weight is positive
        self.model_weights[model_name] = max(0.01, self.model_weights[model_name])
        
        # Normalize weights to sum to 1
        weight_sum = sum(self.model_weights.values())
        self.model_weights = {model: weight / weight_sum for model, weight in self.model_weights.items()}
        
        # Save updated weights
        self._save_state()
        
        return self.model_weights

if __name__ == "__main__":
    # Create self-learning mechanism
    mechanism = SelfLearningMechanism()
    
    # Visualize current performance
    success_rates = mechanism.visualize_performance()
    
    # Analyze performance trends
    trends = mechanism.analyze_performance_trends()
    
    # Print current weights
    weights = mechanism.get_current_weights()
    print("Current model weights:")
    for model, weight in weights.items():
        print(f"  {model}: {weight:.4f}")
        
    # Print performance if available
    if success_rates:
        print("\nModel success rates:")
        for model, rate in success_rates.items():
            print(f"  {model}: {rate:.2f}%")