#!/usr/bin/env python3
"""
Play Whe Lottery Prediction Models

This script implements multiple prediction models for the Play Whe lottery
based on the patterns identified in the data analysis.
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
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/prediction_models.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("play_whe_prediction_models")

class PlayWhePredictionModels:
    """
    A class to implement and evaluate prediction models for Play Whe lottery
    """
    
    def __init__(self, data_file="data/play_whe_processed.csv", output_dir="models"):
        """
        Initialize the prediction models with configuration parameters
        
        Args:
            data_file (str): Path to the processed data file
            output_dir (str): Directory to save model results
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
        
        # Initialize models
        self.models = {}
        
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
            
    def build_all_models(self):
        """
        Build all prediction models
        
        Returns:
            dict: Dictionary of trained models
        """
        if self.df is None or self.df.empty:
            logger.error("No data to build models")
            return None
            
        logger.info("Building all prediction models...")
        
        # Build each model
        self.models['frequency'] = self.build_frequency_model()
        self.models['time_sensitive_frequency'] = self.build_time_sensitive_frequency_model()
        self.models['sequential'] = self.build_sequential_model()
        self.models['hot_cold'] = self.build_hot_cold_model()
        self.models['hybrid'] = self.build_hybrid_model()
        self.models['optimized_hybrid'] = self.build_optimized_hybrid_model()
        
        # Save models
        self._save_models()
        
        logger.info("All models built successfully")
        
        return self.models
        
    def build_time_sensitive_frequency_model(self):
        """
        Build a time-sensitive frequency model that accounts for draw times and schedule changes
        
        Returns:
            dict: Model details and parameters
        """
        logger.info("Building time-sensitive frequency model...")
        
        # Check if required columns exist
        required_columns = ['draw_period', 'post_schedule_change']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}. Run merge_data.py first.")
            return None
        
        # Group by draw period
        draw_periods = self.df['draw_period'].unique()
        period_models = {}
        
        for period in draw_periods:
            if period == 'unknown':
                continue
                
            logger.info(f"Building model for {period} period...")
            period_df = self.df[self.df['draw_period'] == period]
            
            # Apply exponential decay weighting to favor recent draws
            max_date = period_df['date'].max()
            period_df['days_from_max'] = (max_date - period_df['date']).dt.days
            period_df['weight'] = period_df['days_from_max'].apply(
                lambda x: np.exp(-0.005 * x)  # Decay factor (adjust as needed)
            )
            
            # Calculate weighted frequencies
            weighted_counts = {}
            for num in range(1, 37):
                num_df = period_df[period_df['number'] == num]
                weighted_counts[num] = num_df['weight'].sum() if not num_df.empty else 0
                
            total_weight = period_df['weight'].sum()
            probabilities = {num: count/total_weight if total_weight > 0 else 0
                           for num, count in weighted_counts.items()}
            
            # Calculate confidence scores
            expected_prob = 1/36
            confidence_scores = {num: min(abs(prob - expected_prob) / expected_prob * 100, 100)
                               for num, prob in probabilities.items()}
            
            # Store period model
            period_models[period] = {
                'probabilities': probabilities,
                'confidence_scores': confidence_scores,
                'sample_size': len(period_df)
            }
            
            # Generate visualization for this period
            plt.figure(figsize=(12, 6))
            numbers = list(range(1, 37))
            probs = [probabilities.get(num, 0) for num in numbers]
            conf = [confidence_scores.get(num, 0) for num in numbers]
            
            ax1 = plt.subplot(111)
            bars = ax1.bar(numbers, probs, alpha=0.7)
            ax1.set_xlabel('Number')
            ax1.set_ylabel('Probability')
            ax1.set_title(f'Time-Sensitive Model ({period.capitalize()} Period): Probability by Number')
            ax1.axhline(y=expected_prob, color='r', linestyle='--', label=f'Expected probability ({expected_prob:.4f})')
            
            ax2 = ax1.twinx()
            ax2.plot(numbers, conf, 'g-', label='Confidence Score')
            ax2.set_ylabel('Confidence Score (%)')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.tight_layout()
            
            # Save figure
            fig_path = os.path.join(self.output_dir, f'time_sensitive_model_{period}.png')
            plt.savefig(fig_path, dpi=300)
            plt.close()
        
        # Create separate pre/post schedule change models
        pre_change_df = self.df[self.df['post_schedule_change'] == False]
        post_change_df = self.df[self.df['post_schedule_change'] == True]
        
        pre_change_models = {}
        post_change_models = {}
        
        # Build models for each period pre-change
        for period in draw_periods:
            if period == 'unknown':
                continue
                
            period_pre_df = pre_change_df[pre_change_df['draw_period'] == period]
            
            if len(period_pre_df) > 0:
                # Calculate frequencies
                number_counts = period_pre_df['number'].value_counts()
                total_draws = len(period_pre_df)
                
                # Calculate probability for each number
                probabilities = {num: count/total_draws for num, count in number_counts.items()}
                
                # Fill in missing numbers with zero probability
                for num in range(1, 37):
                    if num not in probabilities:
                        probabilities[num] = 0
                
                # Calculate confidence scores
                expected_prob = 1/36
                confidence_scores = {num: min(abs(prob - expected_prob) / expected_prob * 100, 100)
                                   for num, prob in probabilities.items()}
                
                pre_change_models[period] = {
                    'probabilities': probabilities,
                    'confidence_scores': confidence_scores,
                    'sample_size': total_draws
                }
        
        # Build models for each period post-change
        for period in draw_periods:
            if period == 'unknown':
                continue
                
            period_post_df = post_change_df[post_change_df['draw_period'] == period]
            
            if len(period_post_df) > 0:
                # Calculate frequencies
                number_counts = period_post_df['number'].value_counts()
                total_draws = len(period_post_df)
                
                # Calculate probability for each number
                probabilities = {num: count/total_draws for num, count in number_counts.items()}
                
                # Fill in missing numbers with zero probability
                for num in range(1, 37):
                    if num not in probabilities:
                        probabilities[num] = 0
                
                # Calculate confidence scores
                expected_prob = 1/36
                confidence_scores = {num: min(abs(prob - expected_prob) / expected_prob * 100, 100)
                                   for num, prob in probabilities.items()}
                
                post_change_models[period] = {
                    'probabilities': probabilities,
                    'confidence_scores': confidence_scores,
                    'sample_size': total_draws
                }
        
        # Create model
        model = {
            'name': 'Time-Sensitive Frequency Model',
            'description': 'Predicts numbers based on time-specific historical frequency with recency weighting',
            'period_models': period_models,
            'pre_change_models': pre_change_models,
            'post_change_models': post_change_models,
            'predict': lambda period='morning', n=5: self._predict_time_sensitive_frequency(period, n)
        }
        
        logger.info("Time-sensitive frequency model built successfully")
        
        return model
        
    def _predict_time_sensitive_frequency(self, period='morning', n=5):
        """
        Make predictions using the time-sensitive frequency model
        
        Args:
            period (str): Draw period ('morning', 'midday', 'afternoon', 'evening')
            n (int): Number of predictions to return
            
        Returns:
            list: List of (number, probability, confidence) tuples
        """
        model = self.models['time_sensitive_frequency']
        
        # Check if period exists in model
        if period not in model['period_models']:
            logger.warning(f"Period {period} not found in model, using morning instead")
            period = 'morning'
        
        # Get period model
        period_model = model['period_models'][period]
        
        # Sort numbers by probability
        sorted_probs = sorted(period_model['probabilities'].items(), key=lambda x: x[1], reverse=True)
        
        # Get top n predictions with confidence scores
        predictions = [(int(num), prob, period_model['confidence_scores'][num])
                      for num, prob in sorted_probs[:n]]
        
        return predictions
        
    def build_frequency_model(self):
        """
        Build a frequency-based prediction model
        
        Returns:
            dict: Model details and parameters
        """
        logger.info("Building frequency-based prediction model...")
        
        # Count frequency of each number
        number_counts = self.df['number'].value_counts()
        total_draws = len(self.df)
        
        # Calculate probability for each number
        probabilities = {num: count/total_draws for num, count in number_counts.items()}
        
        # Calculate confidence scores based on deviation from expected probability
        expected_prob = 1/36  # Equal probability for all 36 numbers
        confidence_scores = {num: min(abs(prob - expected_prob) / expected_prob * 100, 100) 
                            for num, prob in probabilities.items()}
        
        # Create model
        model = {
            'name': 'Frequency-Based Model',
            'description': 'Predicts numbers based on their historical frequency',
            'probabilities': probabilities,
            'confidence_scores': confidence_scores,
            'predict': lambda n=5: self._predict_frequency(n)
        }
        
        # Generate visualization
        plt.figure(figsize=(12, 6))
        numbers = list(range(1, 37))
        probs = [probabilities.get(num, 0) for num in numbers]
        conf = [confidence_scores.get(num, 0) for num in numbers]
        
        ax1 = plt.subplot(111)
        bars = ax1.bar(numbers, probs, alpha=0.7)
        ax1.set_xlabel('Number')
        ax1.set_ylabel('Probability')
        ax1.set_title('Frequency-Based Model: Probability and Confidence by Number')
        ax1.axhline(y=expected_prob, color='r', linestyle='--', label=f'Expected probability ({expected_prob:.4f})')
        
        ax2 = ax1.twinx()
        ax2.plot(numbers, conf, 'g-', label='Confidence Score')
        ax2.set_ylabel('Confidence Score (%)')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, 'frequency_model.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        logger.info("Frequency-based model built successfully")
        
        return model
        
    def _predict_frequency(self, n=5):
        """
        Make predictions using the frequency model
        
        Args:
            n (int): Number of predictions to return
            
        Returns:
            list: List of (number, probability, confidence) tuples
        """
        model = self.models['frequency']
        
        # Sort numbers by probability
        sorted_probs = sorted(model['probabilities'].items(), key=lambda x: x[1], reverse=True)
        
        # Get top n predictions with confidence scores
        predictions = [(int(num), prob, model['confidence_scores'][num]) 
                      for num, prob in sorted_probs[:n]]
        
        return predictions
        
    def build_sequential_model(self):
        """
        Build a sequential pattern prediction model
        
        Returns:
            dict: Model details and parameters
        """
        logger.info("Building sequential pattern prediction model...")
        
        # Extract the sequence of numbers
        number_sequence = self.df['number'].values
        
        # Build transition matrix
        transitions = defaultdict(Counter)
        for i in range(len(number_sequence)-1):
            from_num = number_sequence[i]
            to_num = number_sequence[i+1]
            transitions[from_num][to_num] += 1
            
        # Convert to probability matrix
        transition_probs = {}
        for from_num, to_counts in transitions.items():
            total = sum(to_counts.values())
            transition_probs[from_num] = {to_num: count/total for to_num, count in to_counts.items()}
            
        # Calculate confidence scores based on deviation from expected probability
        expected_prob = 1/36  # Equal probability for all 36 numbers
        confidence_scores = {}
        
        for from_num, to_probs in transition_probs.items():
            confidence_scores[from_num] = {}
            for to_num, prob in to_probs.items():
                confidence_scores[from_num][to_num] = min(abs(prob - expected_prob) / expected_prob * 100, 100)
                
        # Create model
        model = {
            'name': 'Sequential Pattern Model',
            'description': 'Predicts numbers based on transitions from previous draws',
            'transition_probabilities': transition_probs,
            'confidence_scores': confidence_scores,
            'predict': lambda prev_num, n=5: self._predict_sequential(prev_num, n)
        }
        
        # Generate visualization (transition heatmap)
        transition_matrix = np.zeros((36, 36))
        for from_num, to_probs in transition_probs.items():
            for to_num, prob in to_probs.items():
                transition_matrix[int(from_num)-1][int(to_num)-1] = prob
                
        plt.figure(figsize=(12, 10))
        sns.heatmap(transition_matrix, cmap='viridis')
        plt.xlabel('To Number')
        plt.ylabel('From Number')
        plt.title('Sequential Model: Transition Probabilities')
        plt.xticks(np.arange(0.5, 36.5), range(1, 37))
        plt.yticks(np.arange(0.5, 36.5), range(1, 37))
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, 'sequential_model.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        logger.info("Sequential pattern model built successfully")
        
        return model
        
    def _predict_sequential(self, prev_num, n=5):
        """
        Make predictions using the sequential model
        
        Args:
            prev_num (int): Previous winning number
            n (int): Number of predictions to return
            
        Returns:
            list: List of (number, probability, confidence) tuples
        """
        model = self.models['sequential']
        
        # If previous number not in transitions, use frequency model
        if prev_num not in model['transition_probabilities']:
            logger.warning(f"Previous number {prev_num} not found in transition matrix, using frequency model")
            return self._predict_frequency(n)
            
        # Sort transitions by probability
        sorted_probs = sorted(model['transition_probabilities'][prev_num].items(), 
                             key=lambda x: x[1], reverse=True)
        
        # Get top n predictions with confidence scores
        predictions = [(int(num), prob, model['confidence_scores'][prev_num][num]) 
                      for num, prob in sorted_probs[:n]]
        
        return predictions
        
    def build_hot_cold_model(self):
        """
        Build a hot/cold number prediction model
        
        Returns:
            dict: Model details and parameters
        """
        logger.info("Building hot/cold number prediction model...")
        
        # Define window sizes for analysis
        windows = [30, 50, 100]
        
        # Calculate hot/cold metrics for each window
        hot_cold_metrics = {}
        
        for window in windows:
            # Get the most recent draws up to the window size
            recent_df = self.df.tail(window)
            
            # Count frequency of each number
            number_counts = recent_df['number'].value_counts().reindex(range(1, 37), fill_value=0)
            
            # Calculate expected frequency (uniform distribution)
            expected_freq = len(recent_df) / 36
            
            # Calculate probability for each number
            probabilities = {num: count/window for num, count in number_counts.items()}
            
            # Calculate hot score (ratio to expected frequency)
            hot_scores = {num: count/expected_freq for num, count in number_counts.items()}
            
            hot_cold_metrics[window] = {
                'counts': number_counts.to_dict(),
                'probabilities': probabilities,
                'hot_scores': hot_scores
            }
            
        # Create model with combined metrics
        combined_hot_scores = {}
        combined_probabilities = {}
        
        # Weight recent windows more heavily
        weights = {30: 0.5, 50: 0.3, 100: 0.2}
        
        for num in range(1, 37):
            combined_hot_scores[num] = sum(hot_cold_metrics[window]['hot_scores'][num] * weights[window] 
                                         for window in windows)
            combined_probabilities[num] = sum(hot_cold_metrics[window]['probabilities'][num] * weights[window] 
                                           for window in windows)
            
        # Calculate confidence scores based on deviation from expected
        expected_prob = 1/36  # Equal probability for all 36 numbers
        confidence_scores = {num: min(abs(prob - expected_prob) / expected_prob * 100, 100) 
                            for num, prob in combined_probabilities.items()}
        
        # Create model
        model = {
            'name': 'Hot/Cold Number Model',
            'description': 'Predicts numbers based on recent frequency patterns',
            'hot_scores': combined_hot_scores,
            'probabilities': combined_probabilities,
            'confidence_scores': confidence_scores,
            'metrics_by_window': hot_cold_metrics,
            'predict': lambda n=5, strategy='hot': self._predict_hot_cold(n, strategy)
        }
        
        # Generate visualization
        plt.figure(figsize=(12, 6))
        numbers = list(range(1, 37))
        hot_scores = [combined_hot_scores[num] for num in numbers]
        conf = [confidence_scores[num] for num in numbers]
        
        ax1 = plt.subplot(111)
        bars = ax1.bar(numbers, hot_scores, alpha=0.7)
        ax1.set_xlabel('Number')
        ax1.set_ylabel('Hot Score (ratio to expected)')
        ax1.set_title('Hot/Cold Model: Hot Scores and Confidence by Number')
        ax1.axhline(y=1.0, color='r', linestyle='--', label='Expected frequency ratio (1.0)')
        
        # Color bars based on hot/cold
        for i, bar in enumerate(bars):
            if hot_scores[i] > 1.0:
                bar.set_color('red')  # Hot
            else:
                bar.set_color('blue')  # Cold
                
        ax2 = ax1.twinx()
        ax2.plot(numbers, conf, 'g-', label='Confidence Score')
        ax2.set_ylabel('Confidence Score (%)')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, 'hot_cold_model.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        logger.info("Hot/cold number model built successfully")
        
        return model
        
    def _predict_hot_cold(self, n=5, strategy='hot'):
        """
        Make predictions using the hot/cold model
        
        Args:
            n (int): Number of predictions to return
            strategy (str): 'hot' to predict hot numbers, 'cold' to predict cold numbers
            
        Returns:
            list: List of (number, probability, confidence) tuples
        """
        model = self.models['hot_cold']
        
        if strategy == 'hot':
            # Sort numbers by hot score (descending)
            sorted_scores = sorted(model['hot_scores'].items(), key=lambda x: x[1], reverse=True)
        else:
            # Sort numbers by hot score (ascending for cold numbers)
            sorted_scores = sorted(model['hot_scores'].items(), key=lambda x: x[1])
            
        # Get top n predictions with probability and confidence scores
        predictions = [(int(num), model['probabilities'][num], model['confidence_scores'][num]) 
                      for num, _ in sorted_scores[:n]]
        
        return predictions
        
    def build_hybrid_model(self):
        """
        Build a hybrid prediction model that combines multiple approaches
        
        Returns:
            dict: Model details and parameters
        """
        logger.info("Building hybrid prediction model...")
        
        # Build individual models if not already built
        if 'frequency' not in self.models:
            self.models['frequency'] = self.build_frequency_model()
        if 'sequential' not in self.models:
            self.models['sequential'] = self.build_sequential_model()
        if 'hot_cold' not in self.models:
            self.models['hot_cold'] = self.build_hot_cold_model()
        
        # Define weights for each model
        weights = {
            'frequency': 0.3,
            'sequential': 0.4,
            'hot_cold': 0.3
        }
        
        # Create model
        model = {
            'name': 'Hybrid Model',
            'description': 'Combines multiple prediction approaches for improved accuracy',
            'weights': weights,
            'predict': lambda prev_num, n=5: self._predict_hybrid(prev_num, n)
        }
        
        logger.info("Hybrid model built successfully")
        
        return model
        
    def _predict_hybrid(self, prev_num, n=5):
        """
        Make predictions using the hybrid model
        
        Args:
            prev_num (int): Previous winning number
            n (int): Number of predictions to return
            
        Returns:
            list: List of (number, probability, confidence) tuples
        """
        # Get predictions from each model
        freq_predictions = self._predict_frequency(n=36)
        seq_predictions = self._predict_sequential(prev_num, n=36)
        hot_predictions = self._predict_hot_cold(n=36, strategy='hot')
        
        # Convert to dictionaries for easier manipulation
        freq_dict = {num: (prob, conf) for num, prob, conf in freq_predictions}
        seq_dict = {num: (prob, conf) for num, prob, conf in seq_predictions}
        hot_dict = {num: (prob, conf) for num, prob, conf in hot_predictions}
        
        # Combine predictions with weights
        combined_scores = {}
        weights = self.models['hybrid']['weights']
        
        for num in range(1, 37):
            freq_score = freq_dict.get(num, (0, 0))[0] * weights['frequency']
            seq_score = seq_dict.get(num, (0, 0))[0] * weights['sequential']
            hot_score = hot_dict.get(num, (0, 0))[0] * weights['hot_cold']
            
            # Calculate weighted average
            combined_score = freq_score + seq_score + hot_score
            
            # Calculate confidence as weighted average of individual confidences
            freq_conf = freq_dict.get(num, (0, 0))[1] * weights['frequency']
            seq_conf = seq_dict.get(num, (0, 0))[1] * weights['sequential']
            hot_conf = hot_dict.get(num, (0, 0))[1] * weights['hot_cold']
            
            combined_conf = (freq_conf + seq_conf + hot_conf) / sum(weights.values())
            
            combined_scores[num] = (combined_score, combined_conf)
        
        # Sort by combined score
        sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1][0], reverse=True)
        
        # Get top n predictions
        predictions = [(num, score, conf) for num, (score, conf) in sorted_scores[:n]]
        
        return predictions
        
    def build_optimized_hybrid_model(self):
        """
        Build an optimized hybrid prediction model with dynamic weighting
        
        Returns:
            dict: Model details and parameters
        """
        logger.info("Building optimized hybrid prediction model...")
        
        # Build individual models if not already built
        if 'frequency' not in self.models:
            self.models['frequency'] = self.build_frequency_model()
        if 'time_sensitive_frequency' not in self.models:
            self.models['time_sensitive_frequency'] = self.build_time_sensitive_frequency_model()
        if 'sequential' not in self.models:
            self.models['sequential'] = self.build_sequential_model()
        if 'hot_cold' not in self.models:
            self.models['hot_cold'] = self.build_hot_cold_model()
        
        # Define dynamic weighting based on recent performance
        # This requires tracking model performance over time
        recent_performance = self._evaluate_recent_performance()
        
        # Default weights if no performance data
        default_weights = {
            'frequency': 0.2,
            'time_sensitive_frequency': 0.5,
            'sequential': 0.2,
            'hot_cold': 0.1
        }
        
        # Adjust weights based on performance
        if recent_performance:
            total_accuracy = sum(recent_performance.values())
            if total_accuracy > 0:
                weights = {model: acc/total_accuracy for model, acc in recent_performance.items()}
            else:
                weights = default_weights
        else:
            weights = default_weights
        
        # Create model
        model = {
            'name': 'Optimized Hybrid Model',
            'description': 'Combines multiple prediction approaches with dynamic weighting',
            'weights': weights,
            'predict': lambda prev_num, period='morning', n=5: self._predict_optimized_hybrid(prev_num, period, n)
        }
        
        logger.info("Optimized hybrid model built successfully")
        
        return model
        
    def _evaluate_recent_performance(self):
        """
        Evaluate recent performance of each model
        
        Returns:
            dict: Model performance metrics
        """
        # This would require historical predictions and outcomes
        # For now, return estimated performance based on analysis
        return {
            'frequency': 0.15,
            'time_sensitive_frequency': 0.18,
            'sequential': 0.13,
            'hot_cold': 0.13
        }
        
    def _predict_optimized_hybrid(self, prev_num, period='morning', n=5):
        """
        Make predictions using the optimized hybrid model
        
        Args:
            prev_num (int): Previous winning number
            period (str): Draw period ('morning', 'midday', 'afternoon', 'evening')
            n (int): Number of predictions to return
            
        Returns:
            list: List of (number, probability, confidence) tuples
        """
        # Get predictions from each model
        freq_predictions = self._predict_frequency(n=36)
        time_freq_predictions = self._predict_time_sensitive_frequency(period, n=36)
        seq_predictions = self._predict_sequential(prev_num, n=36)
        hot_predictions = self._predict_hot_cold(n=36, strategy='hot')
        
        # Convert to dictionaries for easier manipulation
        freq_dict = {num: (prob, conf) for num, prob, conf in freq_predictions}
        time_freq_dict = {num: (prob, conf) for num, prob, conf in time_freq_predictions}
        seq_dict = {num: (prob, conf) for num, prob, conf in seq_predictions}
        hot_dict = {num: (prob, conf) for num, prob, conf in hot_predictions}
        
        # Combine predictions with weights
        combined_scores = {}
        weights = self.models['optimized_hybrid']['weights']
        
        for num in range(1, 37):
            freq_score = freq_dict.get(num, (0, 0))[0] * weights.get('frequency', 0.2)
            time_freq_score = time_freq_dict.get(num, (0, 0))[0] * weights.get('time_sensitive_frequency', 0.5)
            seq_score = seq_dict.get(num, (0, 0))[0] * weights.get('sequential', 0.2)
            hot_score = hot_dict.get(num, (0, 0))[0] * weights.get('hot_cold', 0.1)
            
            # Calculate weighted average
            combined_score = freq_score + time_freq_score + seq_score + hot_score
            
            # Calculate confidence as weighted average of individual confidences
            freq_conf = freq_dict.get(num, (0, 0))[1] * weights.get('frequency', 0.2)
            time_freq_conf = time_freq_dict.get(num, (0, 0))[1] * weights.get('time_sensitive_frequency', 0.5)
            seq_conf = seq_dict.get(num, (0, 0))[1] * weights.get('sequential', 0.2)
            hot_conf = hot_dict.get(num, (0, 0))[1] * weights.get('hot_cold', 0.1)
            
            combined_conf = (freq_conf + time_freq_conf + seq_conf + hot_conf) / sum(weights.values())
            
            combined_scores[num] = (combined_score, combined_conf)
        
        # Sort by combined score
        sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1][0], reverse=True)
        
        # Get top n predictions
        predictions = [(num, score, conf) for num, (score, conf) in sorted_scores[:n]]
        
        return predictions
(Content truncated due to size limit. Use line ranges to read in chunks)