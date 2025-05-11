#!/usr/bin/env python3
"""
Make Prediction Script for Enhanced Play Whe Prediction System

This script provides a simple command-line interface to make predictions
using the enhanced Play Whe prediction system.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from enhanced_prediction_system import EnhancedPredictionSystem
from config import PROCESSED_DATA_FILE, SAMPLE_DATA_FILE, PREDICTIONS_DIR, get_output_file_path, get_logger

# Get logger for this module
logger = get_logger("make_prediction")

def parse_arguments():
    """
    Parse command-line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Make Play Whe predictions using the enhanced system')
    
    parser.add_argument('--period', type=str, choices=['morning', 'midday', 'afternoon', 'evening'],
                      default='morning', help='Draw period (default: morning)')
                      
    parser.add_argument('--previous', type=int, nargs='+',
                      help='Previous winning numbers (e.g., 14 7 22)')
                      
    parser.add_argument('--date', type=str,
                      help='Target date for prediction (YYYY-MM-DD, default: today)')
                      
    parser.add_argument('--top', type=int, default=5,
                      help='Number of top predictions to return (default: 5)')
                      
    parser.add_argument('--output', type=str, default=PREDICTIONS_DIR,
                      help=f'Directory to save prediction results (default: {PREDICTIONS_DIR})')
                      
    parser.add_argument('--verbose', action='store_true',
                      help='Print detailed prediction information')
                      
    return parser.parse_args()

def format_predictions(predictions):
    """
    Format predictions for display
    
    Args:
        predictions (dict): Prediction results
        
    Returns:
        str: Formatted prediction string
    """
    if not predictions or 'model_predictions' not in predictions or 'combined' not in predictions['model_predictions']:
        return "No predictions available"
        
    combined = predictions['model_predictions']['combined']
    
    output = "\n=== ENHANCED PLAY WHE PREDICTIONS ===\n"
    output += f"Date: {predictions['date']}\n"
    output += f"Draw Period: {predictions['draw_period']}\n"
    output += f"Previous Numbers: {predictions['previous_numbers']}\n"
    output += "===================================\n\n"
    
    output += "Top Predictions:\n"
    for i, (num, prob, conf) in enumerate(combined, 1):
        output += f"{i}. Number {num}: Probability {prob:.6f}, Confidence {conf:.2f}%\n"
        
    output += "\nModel Weights:\n"
    for model, weight in predictions['model_weights'].items():
        output += f"  {model}: {weight:.4f}\n"
        
    return output

def format_detailed_predictions(predictions):
    """
    Format detailed predictions for display
    
    Args:
        predictions (dict): Prediction results
        
    Returns:
        str: Formatted detailed prediction string
    """
    if not predictions or 'model_predictions' not in predictions:
        return "No predictions available"
        
    model_preds = predictions['model_predictions']
    
    output = format_predictions(predictions)
    output += "\n=== DETAILED MODEL PREDICTIONS ===\n"
    
    for model, preds in model_preds.items():
        if model != 'combined' and preds:
            output += f"\n{model.capitalize()} Model:\n"
            for i, (num, prob, conf) in enumerate(preds, 1):
                output += f"{i}. Number {num}: Probability {prob:.6f}, Confidence {conf:.2f}%\n"
                
    return output

def main():
    """
    Main function
    """
    try:
        # Parse arguments
        args = parse_arguments()

        # Validate arguments
        if args.previous and any(num < 1 or num > 36 for num in args.previous):
            logger.error("Previous numbers must be between 1 and 36")
            print("Error: Previous numbers must be between 1 and 36")
            sys.exit(1)

        if args.date:
            try:
                datetime.strptime(args.date, '%Y-%m-%d')
            except ValueError:
                logger.error("Date must be in YYYY-MM-DD format")
                print("Error: Date must be in YYYY-MM-DD format")
                sys.exit(1)

        # Create output directory if it doesn't exist
        try:
            os.makedirs(args.output, exist_ok=True)
        except PermissionError:
            print(f"Error: Cannot create output directory {args.output} (permission denied)")
            sys.exit(1)
        except Exception as e:
            print(f"Error creating output directory: {e}")
            sys.exit(1)

        # Check if required files exist
        data_file = PROCESSED_DATA_FILE
        if not os.path.exists(data_file):
            logger.warning(f"Processed data file {data_file} not found")
            print(f"Warning: Processed data file {data_file} not found")
            print("Run 'python merge_data.py' first or use sample data with 'python test_with_sample_data.py'")

            # Check if sample data exists
            if os.path.exists(SAMPLE_DATA_FILE):
                logger.info(f"Found sample data file {SAMPLE_DATA_FILE}. Using it for predictions.")
                print(f"Found sample data file {SAMPLE_DATA_FILE}. Using it for predictions.")
                data_file = SAMPLE_DATA_FILE
            else:
                logger.error("No data files found. Cannot make predictions.")
                print("No data files found. Cannot make predictions.")
                sys.exit(1)

        try:
            # Create enhanced prediction system
            system = EnhancedPredictionSystem(data_file=data_file, output_dir=args.output)

            # Make predictions
            predictions = system.predict(
                previous_numbers=args.previous,
                draw_period=args.period,
                target_date=args.date,
                top_n=args.top
            )

            # Check if predictions were generated successfully
            if not predictions or 'model_predictions' not in predictions:
                print("Error: Failed to generate predictions")
                sys.exit(1)

            # Print predictions
            if args.verbose:
                print(format_detailed_predictions(predictions))
            else:
                print(format_predictions(predictions))

            # Save predictions to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = get_output_file_path("predictions", f'prediction_{timestamp}.json')

            try:
                with open(output_file, 'w') as f:
                    json.dump(predictions, f, indent=2, default=str)

                logger.info(f"Prediction saved to: {output_file}")
                print(f"\nPrediction saved to: {output_file}")
                print(f"Visualizations saved to: {args.output}/")
            except Exception as e:
                logger.warning(f"Could not save prediction to file: {e}")
                print(f"Warning: Could not save prediction to file: {e}")

        except ImportError as e:
            print(f"Error: Missing required dependencies: {e}")
            print("Install required packages with: pip install -r requirements.txt")
            sys.exit(1)
        except Exception as e:
            print(f"Error making predictions: {e}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nPrediction cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()