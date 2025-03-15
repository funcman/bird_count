#!/usr/bin/env python3
"""
Bird Counter - A program to count birds in images
This script combines both the basic and advanced bird counting methods
"""

import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import sys

# Try to import DBSCAN for advanced method
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import methods from the other modules
from bird_counter import count_birds
try:
    from advanced_bird_counter import count_birds_advanced
except ImportError:
    # If the advanced module isn't available, define a placeholder
    if SKLEARN_AVAILABLE:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from advanced_bird_counter import count_birds_advanced
    else:
        def count_birds_advanced(image_path, save_result=True, debug=False):
            print("Advanced method requires scikit-learn. Please install it with 'pip install scikit-learn'")
            return None, None

def main():
    """Main function to parse arguments and process the image using the selected method."""
    parser = argparse.ArgumentParser(description="Count birds in an image")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--method", choices=["basic", "advanced"], default="basic", 
                        help="Method to use for bird counting (default: basic)")
    parser.add_argument("--debug", action="store_true", help="Show intermediate steps")
    parser.add_argument("--no-save", dest="save", action="store_false", 
                        help="Don't save the result image")
    parser.add_argument("--compare", action="store_true", 
                        help="Run both methods and compare results")
    parser.set_defaults(save=True, debug=False, compare=False)
    
    args = parser.parse_args()
    
    # Check if the image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found.")
        return
    
    try:
        # If compare flag is set, run both methods
        if args.compare:
            if not SKLEARN_AVAILABLE and args.method == "advanced":
                print("Warning: Advanced method requires scikit-learn. Only basic method will be run.")
                print("Install scikit-learn with 'pip install scikit-learn' to use the advanced method.")
                return
            
            print("Running basic method...")
            basic_count, basic_result = count_birds(
                args.image_path, save_result=args.save, debug=args.debug
            )
            
            if SKLEARN_AVAILABLE:
                print("Running advanced method...")
                advanced_count, advanced_result = count_birds_advanced(
                    args.image_path, save_result=args.save, debug=args.debug
                )
                
                # Display comparison
                print("\nComparison Results:")
                print(f"Basic method detected: {basic_count} birds")
                print(f"Advanced method detected: {advanced_count} birds")
                print(f"Difference: {abs(advanced_count - basic_count)} birds")
                
                # Show side-by-side comparison if debug mode
                if args.debug:
                    plt.figure(figsize=(16, 8))
                    plt.subplot(1, 2, 1)
                    plt.title(f"Basic Method: {basic_count} birds")
                    plt.imshow(cv2.cvtColor(basic_result, cv2.COLOR_BGR2RGB))
                    
                    plt.subplot(1, 2, 2)
                    plt.title(f"Advanced Method: {advanced_count} birds")
                    plt.imshow(cv2.cvtColor(advanced_result, cv2.COLOR_BGR2RGB))
                    
                    plt.tight_layout()
                    plt.show()
            else:
                print("Advanced method requires scikit-learn. Only basic method results available.")
                print(f"Basic method detected: {basic_count} birds")
        else:
            # Run the selected method
            if args.method == "basic":
                bird_count, _ = count_birds(
                    args.image_path, save_result=args.save, debug=args.debug
                )
                print(f"Total birds detected (basic method): {bird_count}")
            elif args.method == "advanced":
                if not SKLEARN_AVAILABLE:
                    print("Advanced method requires scikit-learn. Please install with 'pip install scikit-learn'")
                    return
                bird_count, _ = count_birds_advanced(
                    args.image_path, save_result=args.save, debug=args.debug
                )
                print(f"Total birds detected (advanced method): {bird_count}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 