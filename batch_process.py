#!/usr/bin/env python3
"""
Bird Counter - Batch Processing Script
Process multiple bird images in a directory
"""

import os
import sys
import argparse
import time
import csv

# Try importing from the current directory first
try:
    from bird_counter import count_birds
    try:
        from advanced_bird_counter import count_birds_advanced
        ADVANCED_METHOD_AVAILABLE = True
    except ImportError:
        ADVANCED_METHOD_AVAILABLE = False
except ImportError:
    # If imports fail, try adding the script directory to Python path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    
    try:
        from bird_counter import count_birds
        try:
            from advanced_bird_counter import count_birds_advanced
            ADVANCED_METHOD_AVAILABLE = True
        except ImportError:
            ADVANCED_METHOD_AVAILABLE = False
    except ImportError:
        print("Error: Could not import bird counting modules. Make sure you are running from the correct directory.")
        sys.exit(1)

def process_directory(directory, method="basic", save_results=True, debug=False):
    """
    Process all image files in the given directory.
    
    Args:
        directory (str): Path to directory containing bird images
        method (str): Method to use for bird counting ('basic' or 'advanced')
        save_results (bool): Whether to save result images
        debug (bool): Whether to show debug information
    
    Returns:
        dict: Results mapping filename to bird count
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    results = {}
    
    # Validate directory
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        return results
    
    # Check method availability
    if method == "advanced" and not ADVANCED_METHOD_AVAILABLE:
        print("Warning: Advanced method is not available. Falling back to basic method.")
        print("To use the advanced method, install scikit-learn with 'pip install scikit-learn'")
        method = "basic"
    
    # Create output directory
    output_dir = os.path.join(directory, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get list of image files
    image_files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if os.path.isfile(filepath) and file_ext in image_extensions:
            image_files.append(filename)
    
    if not image_files:
        print(f"No image files found in {directory}")
        return results
    
    print(f"Found {len(image_files)} image files to process")
    
    # Process each image
    for i, filename in enumerate(image_files):
        filepath = os.path.join(directory, filename)
        print(f"[{i+1}/{len(image_files)}] Processing {filename}...")
        
        start_time = time.time()
        
        try:
            if method == "basic":
                bird_count, _ = count_birds(filepath, save_result=save_results, debug=debug)
            else:
                bird_count, _ = count_birds_advanced(filepath, save_result=save_results, debug=debug)
            
            process_time = time.time() - start_time
            results[filename] = bird_count
            
            print(f"  Detected {bird_count} birds (processing time: {process_time:.2f}s)")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
    
    # Save results to CSV file
    csv_path = os.path.join(output_dir, "bird_counts.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Bird Count"])
        for filename, count in results.items():
            writer.writerow([filename, count])
    
    print(f"Results saved to {csv_path}")
    
    # Print summary
    if results:
        total_birds = sum(results.values())
        avg_birds = total_birds / len(results)
        print("\nSummary:")
        print(f"Total images processed: {len(results)}")
        print(f"Total birds detected: {total_birds}")
        print(f"Average birds per image: {avg_birds:.2f}")
    
    return results

def main():
    """Parse command line arguments and process directory."""
    parser = argparse.ArgumentParser(description="Process multiple bird images in a directory")
    parser.add_argument("directory", help="Directory containing bird images")
    parser.add_argument("--method", choices=["basic", "advanced"], default="basic", 
                       help="Method to use for bird counting (default: basic)")
    parser.add_argument("--no-save", dest="save", action="store_false", 
                       help="Don't save result images")
    parser.add_argument("--debug", action="store_true", 
                       help="Show debug information (warning: will open many windows!)")
    parser.set_defaults(save=True, debug=False)
    
    args = parser.parse_args()
    
    try:
        process_directory(
            args.directory,
            method=args.method, 
            save_results=args.save,
            debug=args.debug
        )
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 