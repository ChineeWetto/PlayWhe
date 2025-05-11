#!/usr/bin/env python3
"""
Update Models Script for Enhanced Play Whe Prediction System

This script provides a simple command-line interface to update the enhanced
Play Whe prediction system with actual results.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from enhanced_prediction_system import EnhancedPredictionSystem

def parse_arguments():
    """
    Parse command-line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Update Play Whe prediction models with actual results')
    
    parser.add_argument('--number', type=int, required=True,
                      help='Actual winning number')
                      
    parser.add_argument('--prediction-file', type=str,
                      help='Path to prediction file to update (if not provided, uses most recent)')
                      
    parser.add_argument('--output', type=str, default='predictions',
                      help='Directory to save update results (default: predictions)')
                      
    parser.add_argument('--analyze', action='store_true',
                      help='Analyze performance after update')
                      
    parser.add_argument('--verbose', action='store_true',
                      help='Print detailed update information')
                      
    return parser.parse_args()

def find_most_recent_prediction(output_dir):
    """
    Find the most recent prediction file
    
    Args:
        output_dir (str): Directory containing prediction files
        
    Returns:
        str: Path to most recent prediction file
    """
    if not os.path.exists(output_dir):
        return None
        
    prediction_files = [f for f in os.listdir(output_dir) if f.startswith('prediction_') and f.endswith('.json')]
    
    if not prediction_files:
        return None
        
    # Sort by timestamp
    prediction_files.sort(reverse=True)
    
    return os.path.join(output_dir, prediction_files[0])

def load_prediction(prediction_file):
    """
    Load prediction from file
    
    Args:
        prediction_file (str): Path to prediction file
        
    Returns:
        dict: Prediction results
    """
    try:
        with open(prediction_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading prediction file: {e}")
        return None

def format_update_results(update_results, prediction_results):
    """
    Format update results for display
    
    Args:
        update_results (dict): Update results
        prediction_results (dict): Original prediction results
        
    Returns:
        str: Formatted update string
    """
    if not update_results or not prediction_results:
        return "No update results available"
        
    output = "\n=== MODEL UPDATE RESULTS ===\n"
    output += f"Actual Number: {update_results['actual_number']}\n"
    output += f"Update Time: {update_results['update_time']}\n"
    output += "===========================\n\n"
    
    # Check if prediction was successful
    if 'performance' in update_results:
        performance = update_results['performance']
        combined_success = performance.get('combined', False)
        
        if combined_success:
            output += "PREDICTION SUCCESS! The actual number was in the top predictions.\n\n"
        else:
            output += "Prediction missed. The actual number was not in the top predictions.\n\n"
            
        # Show individual model performance
        output += "Model Performance:\n"
        for model, success in performance.items():
            if model != 'combined':
                output += f"  {model}: {'Success' if success else 'Failure'}\n"
                
    # Show updated weights
    if 'new_weights' in update_results:
        output += "\nUpdated Model Weights:\n"
        for model, weight in update_results['new_weights'].items():
            output += f"  {model}: {weight:.4f}\n"
            
    return output

def format_performance_analysis(performance_analysis):
    """
    Format performance analysis for display
    
    Args:
        performance_analysis (dict): Performance analysis
        
    Returns:
        str: Formatted performance analysis string
    """
    if not performance_analysis:
        return "No performance analysis available"
        
    output = "\n=== PERFORMANCE ANALYSIS ===\n"
    
    # Show success rates
    if 'success_rates' in performance_analysis:
        success_rates = performance_analysis['success_rates']
        
        output += "Success Rates:\n"
        for model, rate in success_rates.items():
            output += f"  {model}: {rate:.2f}%\n"
            
        # Calculate expected random success rate (5 predictions out of 36 numbers)
        expected_rate = 5 / 36 * 100
        output += f"\nExpected Random Success Rate: {expected_rate:.2f}%\n"
        
        # Calculate improvement
        if 'combined' in success_rates:
            improvement = (success_rates['combined'] - expected_rate) / expected_rate * 100
            output += f"Improvement over Random: {improvement:+.2f}%\n"
            
    # Show current weights
    if 'current_weights' in performance_analysis:
        output += "\nCurrent Model Weights:\n"
        for model, weight in performance_analysis['current_weights'].items():
            output += f"  {model}: {weight:.4f}\n"
            
    return output

def main():
    """
    Main function
    """
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Find prediction file
    prediction_file = args.prediction_file
    if not prediction_file:
        prediction_file = find_most_recent_prediction(args.output)
        
    if not prediction_file or not os.path.exists(prediction_file):
        print(f"Error: No prediction file found")
        sys.exit(1)
        
    # Load prediction
    prediction_results = load_prediction(prediction_file)
    if not prediction_results:
        print(f"Error: Failed to load prediction from {prediction_file}")
        sys.exit(1)
        
    # Create enhanced prediction system
    system = EnhancedPredictionSystem(output_dir=args.output)
    
    # Update models with actual result
    update_results = system.update_with_result(args.number, prediction_results)
    
    # Print update results
    print(format_update_results(update_results, prediction_results))
    
    # Analyze performance if requested
    if args.analyze:
        performance_analysis = system.analyze_performance()
        print(format_performance_analysis(performance_analysis))
        
    # Save update results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(args.output, f'update_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(update_results, f, indent=2, default=str)
        
    print(f"\nUpdate results saved to: {output_file}")
    
    if args.analyze:
        analysis_file = os.path.join(args.output, 'performance_analysis.json')
        print(f"Performance analysis saved to: {analysis_file}")
        print(f"Visualizations saved to: {args.output}/")

if __name__ == "__main__":
    main()