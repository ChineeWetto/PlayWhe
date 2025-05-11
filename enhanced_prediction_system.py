#!/usr/bin/env python3
"""
Enhanced Play Whe Prediction System

This module integrates all the enhanced components of the Play Whe prediction system:
1. Adaptive time-weighted frequency model
2. Enhanced cultural pattern integration
3. Self-learning mechanism
4. Advanced sequential pattern modeling
5. Statistical significance testing

It provides a unified interface for making predictions and continuously improving
the system based on new data and results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import json
from collections import defaultdict

# Import enhanced components
from adaptive_frequency_model import AdaptiveFrequencyModel
from enhanced_cultural_patterns import EnhancedCulturalPatternAnalyzer
from self_learning_mechanism import SelfLearningMechanism
from advanced_sequential_model import AdvancedSequentialModel
from statistical_significance import StatisticalSignificanceTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/enhanced_system.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("enhanced_prediction_system")

class EnhancedPredictionSystem:
    """
    A class that integrates all enhanced components of the Play Whe prediction system
    """
    
    def __init__(self, data_file="data/play_whe_processed.csv", 
                events_file="data/cultural_events.csv", 
                output_dir="predictions"):
        """
        Initialize the enhanced prediction system
        
        Args:
            data_file (str): Path to the processed data file
            events_file (str): Path to the cultural events file
            output_dir (str): Directory to save prediction results
        """
        self.data_file = data_file
        self.events_file = events_file
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        logger.info("Initializing enhanced prediction system components...")
        
        self.adaptive_model = AdaptiveFrequencyModel(data_file=data_file)
        self.cultural_analyzer = EnhancedCulturalPatternAnalyzer(data_file=data_file, events_file=events_file)
        self.sequential_model = AdvancedSequentialModel(data_file=data_file, order=3)
        self.significance_tester = StatisticalSignificanceTester(data_file=data_file)
        self.learning_mechanism = SelfLearningMechanism()
        
        # Load data
        self.df = self._load_data()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set(font_scale=1.2)
        
        logger.info("Enhanced prediction system initialized")
        
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
            dict: Dictionary of built models
        """
        if self.df is None or self.df.empty:
            logger.error("No data to build models")
            return None
            
        logger.info("Building all enhanced prediction models...")
        
        # Build adaptive frequency model
        adaptive_model = self.adaptive_model.build_adaptive_model()
        
        # Build advanced sequential model
        sequential_model = self.sequential_model.build_model()
        
        # Run cultural pattern analysis
        cultural_results = self.cultural_analyzer.run_all_analyses()
        
        # Run statistical significance tests
        significance_results = self.significance_tester.run_all_tests()
        
        # Compile model information
        models = {
            'adaptive_frequency': adaptive_model,
            'advanced_sequential': sequential_model,
            'cultural_patterns': cultural_results,
            'statistical_significance': significance_results,
            'build_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save model information
        models_path = os.path.join(self.output_dir, 'enhanced_models.json')
        with open(models_path, 'w') as f:
            json.dump(models, f, indent=2, default=str)
            
        logger.info(f"Saved enhanced models information to {models_path}")
        
        return models
        
    def predict(self, previous_numbers=None, draw_period=None, target_date=None, top_n=5):
        """
        Make predictions using all enhanced models
        
        Args:
            previous_numbers (list, optional): List of previous winning numbers
            draw_period (str, optional): Draw period ('morning', 'midday', 'afternoon', 'evening')
            target_date (str, optional): Target date for prediction (YYYY-MM-DD)
            top_n (int): Number of top predictions to return
            
        Returns:
            dict: Prediction results
        """
        logger.info(f"Making predictions for {draw_period} period on {target_date}")
        
        # Set default values if not provided
        if previous_numbers is None:
            # Get most recent numbers from data
            if self.df is not None and not self.df.empty:
                recent_data = self.df.sort_values('date', ascending=False).head(3)
                previous_numbers = recent_data['number'].tolist()[::-1]
            else:
                previous_numbers = [14, 7, 22]  # Example default
                
        if draw_period is None:
            draw_period = 'morning'  # Default period
            
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
            
        # Get predictions from each model
        model_predictions = {}
        
        # 1. Adaptive frequency model
        try:
            freq_predictions = self.adaptive_model.predict(period=draw_period, n=top_n)
            model_predictions['adaptive_frequency'] = freq_predictions
        except Exception as e:
            logger.error(f"Error getting adaptive frequency predictions: {e}")
            
        # 2. Advanced sequential model
        try:
            seq_predictions = self.sequential_model.predict(previous_numbers, period=draw_period, n=top_n)
            model_predictions['advanced_sequential'] = seq_predictions
        except Exception as e:
            logger.error(f"Error getting advanced sequential predictions: {e}")
            
        # 3. Hot/cold model (using adaptive model's functionality)
        try:
            # Get base probabilities from adaptive model
            if hasattr(self.adaptive_model, 'df') and self.adaptive_model.df is not None:
                recent_data = self.adaptive_model.df.sort_values('date', ascending=False).head(100)
                number_counts = recent_data['number'].value_counts()
                total = len(recent_data)
                hot_cold_probs = {num: count/total for num, count in number_counts.items()}
                
                # Sort by probability
                sorted_probs = sorted(hot_cold_probs.items(), key=lambda x: x[1], reverse=True)
                
                # Calculate confidence scores
                expected_prob = 1/36
                confidence_scores = {num: min(abs(prob - expected_prob) / expected_prob * 100, 100) 
                                   for num, prob in hot_cold_probs.items()}
                
                # Get top predictions
                hot_cold_predictions = [(int(num), prob, confidence_scores.get(num, 0)) 
                                      for num, prob in sorted_probs[:top_n]]
                                      
                model_predictions['hot_cold'] = hot_cold_predictions
        except Exception as e:
            logger.error(f"Error getting hot/cold predictions: {e}")
            
        # 4. Cultural pattern integration
        try:
            # Get base probabilities (use adaptive frequency as base)
            if 'adaptive_frequency' in model_predictions:
                base_probs = {num: prob for num, prob, _ in model_predictions['adaptive_frequency']}
                
                # Adjust with cultural patterns
                adjusted_probs = self.cultural_analyzer.predict_with_cultural_patterns(
                    base_probs, target_date=target_date)
                
                # Calculate confidence scores
                confidence_scores = self.significance_tester.calculate_confidence_levels(adjusted_probs)
                
                # Sort by probability
                sorted_probs = sorted(adjusted_probs.items(), key=lambda x: x[1], reverse=True)
                
                # Get top predictions
                cultural_predictions = [(int(num), prob, confidence_scores.get(num, 0)) 
                                      for num, prob in sorted_probs[:top_n]]
                                      
                model_predictions['cultural'] = cultural_predictions
        except Exception as e:
            logger.error(f"Error getting cultural predictions: {e}")
            
        # Combine predictions using self-learning mechanism
        try:
            # Convert predictions to lists of numbers
            model_numbers = {}
            for model, predictions in model_predictions.items():
                if predictions:
                    model_numbers[model] = [num for num, _, _ in predictions]
                    
            # Get combined predictions
            combined_numbers = self.learning_mechanism.combine_predictions(model_numbers, top_n)
            
            # Track performance (will be updated later with actual results)
            self.learning_mechanism.track_combined_prediction(combined_numbers, model_numbers, None)
            
            # Format combined predictions
            combined_predictions = []
            for num in combined_numbers:
                # Find highest probability and confidence from component models
                max_prob = 0
                max_conf = 0
                
                for model, predictions in model_predictions.items():
                    for pred_num, prob, conf in predictions:
                        if pred_num == num:
                            max_prob = max(max_prob, prob)
                            max_conf = max(max_conf, conf)
                            
                combined_predictions.append((num, max_prob, max_conf))
                
            model_predictions['combined'] = combined_predictions
        except Exception as e:
            logger.error(f"Error combining predictions: {e}")
            
        # Compile prediction results
        prediction_results = {
            'date': target_date,
            'draw_period': draw_period,
            'previous_numbers': previous_numbers,
            'model_predictions': model_predictions,
            'model_weights': self.learning_mechanism.get_current_weights(),
            'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save prediction results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pred_path = os.path.join(self.output_dir, f'prediction_{timestamp}.json')
        with open(pred_path, 'w') as f:
            json.dump(prediction_results, f, indent=2, default=str)
            
        logger.info(f"Saved prediction results to {pred_path}")
        
        # Generate visualization
        self._visualize_predictions(prediction_results)
        
        return prediction_results
        
    def _visualize_predictions(self, prediction_results):
        """
        Generate visualization of predictions
        
        Args:
            prediction_results (dict): Prediction results
        """
        if not prediction_results or 'model_predictions' not in prediction_results:
            return
            
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Get model predictions
        model_preds = prediction_results['model_predictions']
        models = list(model_preds.keys())
        
        # Create a matrix to visualize predictions
        pred_matrix = np.zeros((len(models), 36))
        
        for i, model in enumerate(models):
            if model in model_preds and model_preds[model]:
                for num, prob, _ in model_preds[model]:
                    pred_matrix[i, num-1] = prob
                    
        # Plot heatmap
        sns.heatmap(pred_matrix, cmap='viridis')
        plt.xlabel('Number')
        plt.ylabel('Model')
        plt.title(f"Prediction Comparison ({prediction_results['draw_period']} period, {prediction_results['date']})")
        plt.yticks(np.arange(len(models)) + 0.5, models)
        plt.xticks(np.arange(36) + 0.5, range(1, 37))
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig_path = os.path.join(self.output_dir, f'prediction_comparison_{timestamp}.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved prediction visualization to {fig_path}")
        
        # Create bar chart of top combined predictions
        plt.figure(figsize=(10, 6))
        
        if 'combined' in model_preds and model_preds['combined']:
            combined = model_preds['combined']
            numbers = [num for num, _, _ in combined]
            probs = [prob for _, prob, _ in combined]
            
            bars = plt.bar(numbers, probs)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{height:.4f}', ha='center', va='bottom')
                
            plt.xlabel('Number')
            plt.ylabel('Probability')
            plt.title(f"Top {len(numbers)} Predictions ({prediction_results['draw_period']} period, {prediction_results['date']})")
            plt.tight_layout()
            
            # Save figure
            fig_path = os.path.join(self.output_dir, f'top_predictions_{timestamp}.png')
            plt.savefig(fig_path, dpi=300)
            plt.close()
            
            logger.info(f"Saved top predictions visualization to {fig_path}")
            
    def update_with_result(self, actual_number, prediction_results=None):
        """
        Update models with actual result
        
        Args:
            actual_number (int): Actual winning number
            prediction_results (dict, optional): Previous prediction results
            
        Returns:
            dict: Update results
        """
        logger.info(f"Updating models with actual result: {actual_number}")
        
        update_results = {
            'actual_number': actual_number,
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Update self-learning mechanism
        if prediction_results and 'model_predictions' in prediction_results:
            model_preds = prediction_results['model_predictions']
            
            # Convert predictions to lists of numbers
            model_numbers = {}
            for model, predictions in model_preds.items():
                if predictions:
                    model_numbers[model] = [num for num, _, _ in predictions]
                    
            # Track performance
            if 'combined' in model_numbers:
                combined_numbers = model_numbers['combined']
                performance = self.learning_mechanism.track_combined_prediction(
                    combined_numbers, model_numbers, actual_number)
                update_results['performance'] = performance
                
            # Apply reinforcement learning update
            for model in model_numbers:
                if model != 'combined':
                    success = actual_number in model_numbers[model]
                    self.learning_mechanism.reinforcement_learning_update(model, success)
                    
            # Adjust weights based on performance
            new_weights = self.learning_mechanism.adjust_weights()
            update_results['new_weights'] = new_weights
            
        # Update sequential model with Bayesian updating
        if prediction_results and 'previous_numbers' in prediction_results:
            previous_numbers = prediction_results['previous_numbers']
            draw_period = prediction_results.get('draw_period')
            
            self.sequential_model.bayesian_update(previous_numbers, actual_number, draw_period)
            update_results['sequential_updated'] = True
            
        # Save update results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        update_path = os.path.join(self.output_dir, f'update_{timestamp}.json')
        with open(update_path, 'w') as f:
            json.dump(update_results, f, indent=2, default=str)
            
        logger.info(f"Saved update results to {update_path}")
        
        return update_results
        
    def analyze_performance(self):
        """
        Analyze prediction performance
        
        Returns:
            dict: Performance analysis
        """
        logger.info("Analyzing prediction performance...")
        
        # Get performance from self-learning mechanism
        success_rates = self.learning_mechanism.visualize_performance()
        
        # Analyze performance trends
        trends = self.learning_mechanism.analyze_performance_trends()
        
        # Compile performance analysis
        performance_analysis = {
            'success_rates': success_rates,
            'trends': trends,
            'current_weights': self.learning_mechanism.get_current_weights(),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save performance analysis
        analysis_path = os.path.join(self.output_dir, 'performance_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(performance_analysis, f, indent=2, default=str)
            
        logger.info(f"Saved performance analysis to {analysis_path}")
        
        return performance_analysis

if __name__ == "__main__":
    # Create enhanced prediction system
    system = EnhancedPredictionSystem()
    
    # Build all models
    models = system.build_all_models()
    
    # Make predictions
    predictions = system.predict(draw_period='morning')
    
    # Print predictions
    if predictions and 'model_predictions' in predictions and 'combined' in predictions['model_predictions']:
        combined = predictions['model_predictions']['combined']
        
        print("\n=== ENHANCED PLAY WHE PREDICTIONS ===")
        print(f"Date: {predictions['date']}")
        print(f"Draw Period: {predictions['draw_period']}")
        print(f"Previous Numbers: {predictions['previous_numbers']}")
        print("===================================")
        
        print("\nTop Predictions:")
        for i, (num, prob, conf) in enumerate(combined, 1):
            print(f"{i}. Number {num}: Probability {prob:.6f}, Confidence {conf:.2f}%")
            
        print("\nModel Weights:")
        for model, weight in predictions['model_weights'].items():
            print(f"  {model}: {weight:.4f}")
    else:
        print("Failed to generate predictions")