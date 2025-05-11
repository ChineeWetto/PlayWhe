#!/usr/bin/env python3
"""
Advanced Sequential Pattern Model for Play Whe Prediction

This module implements an enhanced sequential pattern model with higher-order
Markov chains, draw period-specific transition matrices, and Bayesian updating
of transition probabilities after each new draw.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import json
from collections import defaultdict, Counter
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/advanced_sequential.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("advanced_sequential_model")

class AdvancedSequentialModel:
    """
    A class implementing an advanced sequential pattern model
    """
    
    def __init__(self, data_file="data/play_whe_processed.csv", output_dir="models", order=3):
        """
        Initialize the model with configuration parameters
        
        Args:
            data_file (str): Path to the processed data file
            output_dir (str): Directory to save model results
            order (int): Order of the Markov chain (1, 2, or 3)
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.order = min(max(1, order), 3)  # Ensure order is between 1 and 3
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize transition matrices
        self.transition_matrices = {}
        self.period_matrices = {}
        
        # Initialize prior counts for Bayesian updating
        self.prior_counts = {}
        
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
            
    def _create_state_key(self, previous_numbers):
        """
        Create a state key from previous numbers
        
        Args:
            previous_numbers (list): List of previous numbers
            
        Returns:
            tuple or int: State key
        """
        if len(previous_numbers) == 1:
            return previous_numbers[0]
        else:
            return tuple(previous_numbers)
            
    def _build_transition_matrix(self, number_sequence, order=1):
        """
        Build a transition matrix from a sequence of numbers
        
        Args:
            number_sequence (list): Sequence of numbers
            order (int): Order of the Markov chain
            
        Returns:
            dict: Transition matrix
        """
        transitions = defaultdict(Counter)
        
        # Ensure we have enough numbers for the specified order
        if len(number_sequence) <= order:
            logger.warning(f"Sequence too short for order {order}")
            return transitions
            
        # Build transition counts
        for i in range(len(number_sequence) - order):
            # Create state based on previous numbers
            if order == 1:
                from_state = number_sequence[i]
            else:
                from_state = tuple(number_sequence[i:i+order])
                
            to_num = number_sequence[i+order]
            transitions[from_state][to_num] += 1
            
        return transitions
        
    def _convert_to_probabilities(self, transition_counts):
        """
        Convert transition counts to probabilities
        
        Args:
            transition_counts (dict): Transition count matrix
            
        Returns:
            dict: Transition probability matrix
        """
        transition_probs = {}
        
        for from_state, to_counts in transition_counts.items():
            total = sum(to_counts.values())
            if total > 0:
                transition_probs[from_state] = {to_num: count/total for to_num, count in to_counts.items()}
                
        return transition_probs
        
    def build_model(self):
        """
        Build the advanced sequential model
        
        Returns:
            dict: Model details and parameters
        """
        if self.df is None or self.df.empty:
            logger.error("No data to build model")
            return None
            
        logger.info(f"Building advanced sequential model with order {self.order}...")
        
        # Build overall transition matrix
        number_sequence = self.df['number'].values
        
        # Build matrices for each order
        for order in range(1, self.order + 1):
            transition_counts = self._build_transition_matrix(number_sequence, order)
            transition_probs = self._convert_to_probabilities(transition_counts)
            
            self.transition_matrices[order] = {
                'counts': {str(k): dict(v) for k, v in transition_counts.items()},
                'probabilities': {str(k): v for k, v in transition_probs.items()}
            }
            
            # Save for Bayesian updating
            self.prior_counts[order] = transition_counts
            
        # Build period-specific transition matrices
        if 'draw_period' in self.df.columns:
            draw_periods = self.df['draw_period'].unique()
            
            for period in draw_periods:
                if period == 'unknown':
                    continue
                    
                period_df = self.df[self.df['draw_period'] == period]
                period_sequence = period_df['number'].values
                
                period_matrices = {}
                
                for order in range(1, self.order + 1):
                    transition_counts = self._build_transition_matrix(period_sequence, order)
                    transition_probs = self._convert_to_probabilities(transition_counts)
                    
                    period_matrices[order] = {
                        'counts': {str(k): dict(v) for k, v in transition_counts.items()},
                        'probabilities': {str(k): v for k, v in transition_probs.items()}
                    }
                    
                self.period_matrices[period] = period_matrices
                
        # Generate visualizations
        self._visualize_transition_matrices()
        
        # Create model
        model = {
            'name': f'Advanced Sequential Model (Order {self.order})',
            'description': f'Higher-order Markov chain model with period-specific transitions',
            'order': self.order,
            'transition_matrices': self.transition_matrices,
            'period_matrices': self.period_matrices,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save model
        model_path = os.path.join(self.output_dir, 'advanced_sequential_model.json')
        with open(model_path, 'w') as f:
            json.dump(model, f, indent=2, default=str)
            
        logger.info(f"Saved advanced sequential model to {model_path}")
        
        return model
        
    def _visualize_transition_matrices(self):
        """
        Generate visualizations of transition matrices
        """
        # Visualize first-order transition matrix
        if 1 in self.transition_matrices:
            first_order = self.transition_matrices[1]
            
            # Create transition matrix for visualization
            matrix = np.zeros((36, 36))
            
            for from_num_str, to_probs in first_order['probabilities'].items():
                from_num = int(from_num_str)
                for to_num_str, prob in to_probs.items():
                    to_num = int(to_num_str)
                    matrix[from_num-1, to_num-1] = prob
                    
            plt.figure(figsize=(12, 10))
            sns.heatmap(matrix, cmap='viridis')
            plt.xlabel('To Number')
            plt.ylabel('From Number')
            plt.title('First-Order Transition Probabilities')
            plt.xticks(np.arange(0.5, 36.5), range(1, 37))
            plt.yticks(np.arange(0.5, 36.5), range(1, 37))
            plt.tight_layout()
            
            # Save figure
            fig_path = os.path.join(self.output_dir, 'first_order_transitions.png')
            plt.savefig(fig_path, dpi=300)
            plt.close()
            
        # Visualize period-specific transitions
        if self.period_matrices:
            for period, matrices in self.period_matrices.items():
                if 1 in matrices:
                    first_order = matrices[1]
                    
                    # Create transition matrix for visualization
                    matrix = np.zeros((36, 36))
                    
                    for from_num_str, to_probs in first_order['probabilities'].items():
                        from_num = int(from_num_str)
                        for to_num_str, prob in to_probs.items():
                            to_num = int(to_num_str)
                            matrix[from_num-1, to_num-1] = prob
                            
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(matrix, cmap='viridis')
                    plt.xlabel('To Number')
                    plt.ylabel('From Number')
                    plt.title(f'{period.capitalize()} Period: First-Order Transition Probabilities')
                    plt.xticks(np.arange(0.5, 36.5), range(1, 37))
                    plt.yticks(np.arange(0.5, 36.5), range(1, 37))
                    plt.tight_layout()
                    
                    # Save figure
                    fig_path = os.path.join(self.output_dir, f'{period}_transitions.png')
                    plt.savefig(fig_path, dpi=300)
                    plt.close()
                    
    def predict(self, previous_numbers, period=None, n=5):
        """
        Make predictions using the advanced sequential model
        
        Args:
            previous_numbers (list): List of previous winning numbers
            period (str, optional): Draw period
            n (int): Number of predictions to return
            
        Returns:
            list: List of (number, probability, confidence) tuples
        """
        # Load model if not built
        model_path = os.path.join(self.output_dir, 'advanced_sequential_model.json')
        
        if os.path.exists(model_path) and not self.transition_matrices:
            try:
                with open(model_path, 'r') as f:
                    model = json.load(f)
                    self.transition_matrices = model.get('transition_matrices', {})
                    self.period_matrices = model.get('period_matrices', {})
                    self.order = model.get('order', 1)
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return []
                
        # Ensure we have enough previous numbers
        if len(previous_numbers) < self.order:
            logger.warning(f"Not enough previous numbers for order {self.order}")
            # Pad with the first number if needed
            previous_numbers = [previous_numbers[0]] * (self.order - len(previous_numbers)) + previous_numbers
            
        # Use only the most recent numbers up to the order
        previous_numbers = previous_numbers[-self.order:]
        
        # Try to use period-specific matrix if available
        if period and period in self.period_matrices and self.order in self.period_matrices[period]:
            matrix = self.period_matrices[period][self.order]
            logger.info(f"Using {period} period matrix for prediction")
        elif self.order in self.transition_matrices:
            matrix = self.transition_matrices[self.order]
            logger.info(f"Using overall matrix for prediction")
        else:
            logger.error(f"No transition matrix available for order {self.order}")
            return []
            
        # Create state key
        state_key = self._create_state_key(previous_numbers)
        state_key_str = str(state_key)
        
        # Get transition probabilities for this state
        if state_key_str in matrix['probabilities']:
            probs = matrix['probabilities'][state_key_str]
            
            # Convert string keys to integers
            probs = {int(k): v for k, v in probs.items()}
            
            # Sort by probability
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            
            # Calculate confidence scores
            expected_prob = 1/36
            confidence_scores = {num: min(abs(prob - expected_prob) / expected_prob * 100, 100) 
                               for num, prob in probs.items()}
            
            # Get top n predictions with confidence scores
            predictions = [(num, prob, confidence_scores[num]) 
                          for num, prob in sorted_probs[:n]]
            
            return predictions
        else:
            logger.warning(f"No transitions found for state {state_key}")
            
            # Fall back to lower order if available
            if self.order > 1:
                logger.info(f"Falling back to order {self.order-1}")
                return self.predict(previous_numbers[1:], period, n)
            else:
                logger.warning("No fallback available, returning empty predictions")
                return []
                
    def bayesian_update(self, previous_numbers, actual_number, period=None):
        """
        Update transition probabilities using Bayesian updating
        
        Args:
            previous_numbers (list): List of previous winning numbers
            actual_number (int): Actual winning number
            period (str, optional): Draw period
            
        Returns:
            dict: Updated model
        """
        logger.info(f"Performing Bayesian update with {previous_numbers} -> {actual_number}")
        
        # Ensure we have enough previous numbers
        if len(previous_numbers) < self.order:
            logger.warning(f"Not enough previous numbers for order {self.order}")
            # Pad with the first number if needed
            previous_numbers = [previous_numbers[0]] * (self.order - len(previous_numbers)) + previous_numbers
            
        # Use only the most recent numbers up to the order
        previous_numbers = previous_numbers[-self.order:]
        
        # Update overall transition matrices
        for order in range(1, self.order + 1):
            if order <= len(previous_numbers):
                # Create state key for this order
                state = previous_numbers[-order:]
                state_key = self._create_state_key(state)
                
                # Update prior counts
                if order in self.prior_counts:
                    self.prior_counts[order][state_key][actual_number] += 1
                    
                    # Recalculate probabilities
                    updated_probs = self._convert_to_probabilities(self.prior_counts[order])
                    
                    # Update transition matrices
                    self.transition_matrices[order]['counts'] = {str(k): dict(v) for k, v in self.prior_counts[order].items()}
                    self.transition_matrices[order]['probabilities'] = {str(k): v for k, v in updated_probs.items()}
                    
        # Update period-specific matrices if applicable
        if period and period in self.period_matrices:
            # Create temporary copy of period matrices
            period_priors = {}
            
            # Convert string keys back to proper types
            for order, matrix in self.period_matrices[period].items():
                counts = defaultdict(Counter)
                
                for from_state_str, to_counts in matrix['counts'].items():
                    # Convert from_state back to proper type
                    if ',' in from_state_str:  # It's a tuple
                        from_state = tuple(map(int, from_state_str.strip('()').split(',')))
                    else:  # It's a single number
                        from_state = int(from_state_str)
                        
                    # Convert to_counts
                    for to_num_str, count in to_counts.items():
                        counts[from_state][int(to_num_str)] = count
                        
                period_priors[order] = counts
                
            # Update period-specific counts
            for order in range(1, self.order + 1):
                if order <= len(previous_numbers) and order in period_priors:
                    # Create state key for this order
                    state = previous_numbers[-order:]
                    state_key = self._create_state_key(state)
                    
                    # Update prior counts
                    period_priors[order][state_key][actual_number] += 1
                    
                    # Recalculate probabilities
                    updated_probs = self._convert_to_probabilities(period_priors[order])
                    
                    # Update period matrices
                    self.period_matrices[period][order]['counts'] = {str(k): dict(v) for k, v in period_priors[order].items()}
                    self.period_matrices[period][order]['probabilities'] = {str(k): v for k, v in updated_probs.items()}
                    
        # Save updated model
        model = {
            'name': f'Advanced Sequential Model (Order {self.order})',
            'description': f'Higher-order Markov chain model with period-specific transitions',
            'order': self.order,
            'transition_matrices': self.transition_matrices,
            'period_matrices': self.period_matrices,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        model_path = os.path.join(self.output_dir, 'advanced_sequential_model.json')
        with open(model_path, 'w') as f:
            json.dump(model, f, indent=2, default=str)
            
        logger.info(f"Saved updated model to {model_path}")
        
        return model
        
    def analyze_sequential_patterns(self):
        """
        Analyze sequential patterns in the data
        
        Returns:
            dict: Analysis results
        """
        if self.df is None or self.df.empty:
            logger.error("No data to analyze")
            return None
            
        logger.info("Analyzing sequential patterns...")
        
        # Get number sequence
        number_sequence = self.df['number'].values
        
        # Count consecutive repeats
        consecutive_repeats = 0
        for i in range(1, len(number_sequence)):
            if number_sequence[i] == number_sequence[i-1]:
                consecutive_repeats += 1
                
        repeat_percentage = consecutive_repeats / (len(number_sequence) - 1) * 100
        expected_repeat_percentage = 1 / 36 * 100  # 1/36 chance of repeating
        
        # Find common transitions
        transitions = defaultdict(Counter)
        for i in range(len(number_sequence) - 1):
            from_num = number_sequence[i]
            to_num = number_sequence[i + 1]
            transitions[from_num][to_num] += 1
            
        # Get top transitions
        all_transitions = []
        for from_num, to_counts in transitions.items():
            for to_num, count in to_counts.items():
                all_transitions.append((from_num, to_num, count))
                
        top_transitions = sorted(all_transitions, key=lambda x: x[2], reverse=True)[:10]
        top_transitions_str = [f"{from_num} → {to_num} ({count} times)" 
                             for from_num, to_num, count in top_transitions]
        
        # Calculate autocorrelation
        acf_values = acf(number_sequence, nlags=36)
        
        # Find significant lags
        significant_lags = []
        for lag in range(1, len(acf_values)):
            if abs(acf_values[lag]) > 1.96 / np.sqrt(len(number_sequence)):  # 95% confidence
                significant_lags.append(lag)
                
        # Generate visualization
        plt.figure(figsize=(12, 6))
        plt.stem(range(len(acf_values)), acf_values)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.axhline(y=1.96/np.sqrt(len(number_sequence)), color='g', linestyle='--')
        plt.axhline(y=-1.96/np.sqrt(len(number_sequence)), color='g', linestyle='--')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('Autocorrelation Function')
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, 'autocorrelation.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        # Compile results
        results = {
            'consecutive_repeats': consecutive_repeats,
            'repeat_percentage': repeat_percentage,
            'expected_repeat_percentage': expected_repeat_percentage,
            'top_transitions': top_transitions_str,
            'significant_lags': significant_lags,
            'has_significant_autocorrelation': len(significant_lags) > 0
        }
        
        logger.info(f"Sequential pattern analysis complete. Found {len(significant_lags)} significant lags.")
        
        return results

if __name__ == "__main__":
    # Create model instance
    model = AdvancedSequentialModel(order=3)
    
    # Build model
    advanced_model = model.build_model()
    
    # Analyze sequential patterns
    patterns = model.analyze_sequential_patterns()
    
    # Print summary
    if advanced_model:
        print(f"Advanced Sequential Model (Order {advanced_model['order']}) built successfully")
        
        # Make sample predictions
        previous_numbers = [14, 7, 22]  # Example previous numbers
        predictions = model.predict(previous_numbers, period='morning', n=5)
        
        print(f"\nPredictions based on previous numbers {previous_numbers}:")
        for i, (num, prob, conf) in enumerate(predictions, 1):
            print(f"{i}. Number {num}: Probability {prob:.6f}, Confidence {conf:.2f}%")
            
        # Print pattern analysis
        if patterns:
            print("\nSequential Pattern Analysis:")
            print(f"Consecutive repeats: {patterns['consecutive_repeats']} ({patterns['repeat_percentage']:.2f}%)")
            print(f"Expected repeat percentage: {patterns['expected_repeat_percentage']:.2f}%")
            print(f"Significant autocorrelation lags: {patterns['significant_lags']}")
            print("\nTop transitions:")
            for transition in patterns['top_transitions']:
                print(f"  {transition}")
    else:
        print("Failed to build advanced sequential model")