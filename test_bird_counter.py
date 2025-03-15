#!/usr/bin/env python3
"""
Test script for the Bird Counter program.
This script generates a sample image with birds and runs both counting methods.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from bird_counter import count_birds

try:
    from advanced_bird_counter import count_birds_advanced
    ADVANCED_METHOD_AVAILABLE = True
except ImportError:
    ADVANCED_METHOD_AVAILABLE = False

def generate_sample_image(num_birds=50, image_size=(800, 600), output_path="sample_birds.jpg"):
    """
    Generate a sample image with birds for testing.
    
    Args:
        num_birds (int): Number of birds to generate
        image_size (tuple): Size of the image (width, height)
        output_path (str): Path to save the generated image
    
    Returns:
        str: Path to the generated image
        list: List of bird positions (for validation)
    """
    # Create a gradient sky background
    sky = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    for y in range(image_size[1]):
        # Create blue to light blue gradient for sky
        blue = int(150 + (y / image_size[1]) * 50)
        sky[y, :] = (blue, 200, 240)  # BGR format
    
    # Create ground with some texture at the bottom third of the image
    ground_start = int(image_size[1] * 2/3)
    ground = np.zeros((image_size[1] - ground_start, image_size[0], 3), dtype=np.uint8)
    
    # Create texture on the ground
    for y in range(ground.shape[0]):
        for x in range(ground.shape[1]):
            # Random green/brown ground
            g = random.randint(80, 120)
            b = random.randint(20, 60)
            r = random.randint(30, 80)
            ground[y, x] = (b, g, r)
    
    # Create some ground features (trees, hills)
    for _ in range(30):
        x = random.randint(0, ground.shape[1]-1)
        y = random.randint(0, ground.shape[0]-1)
        radius = random.randint(5, 30)
        color = (20, random.randint(60, 100), 30)  # Dark green
        cv2.circle(ground, (x, y), radius, color, -1)
    
    # Add some clouds in the sky
    for _ in range(10):
        x = random.randint(0, sky.shape[1]-1)
        y = random.randint(0, ground_start//2)
        radius = random.randint(20, 50)
        brightness = random.randint(220, 250)
        color = (brightness, brightness, brightness)  # White-ish
        cv2.circle(sky, (x, y), radius, color, -1)
        
        # Add some texture to clouds
        for _ in range(5):
            offset_x = random.randint(-radius//2, radius//2)
            offset_y = random.randint(-radius//2, radius//2)
            small_radius = random.randint(radius//3, radius//2)
            cv2.circle(sky, (x + offset_x, y + offset_y), small_radius, color, -1)
    
    # Combine sky and ground
    image = sky.copy()
    image[ground_start:] = ground
    
    # Add birds to the sky
    bird_positions = []
    for _ in range(num_birds):
        # Birds only in the sky area
        x = random.randint(10, image_size[0]-10)
        y = random.randint(10, ground_start-10)
        
        # Random bird size (small dark dots)
        size = random.randint(2, 5)
        color = (random.randint(10, 50), random.randint(10, 50), random.randint(10, 50))  # Dark colors
        
        # Draw bird (small ellipse)
        angle = random.randint(0, 180)
        axes = (size, size//2)  # Slightly elongated to look like birds
        cv2.ellipse(image, (x, y), axes, angle, 0, 360, color, -1)
        
        # Sometimes add "wings" to make it look more like a bird
        if random.random() > 0.5:
            wing_size = size // 2
            wing_angle = angle + random.randint(-30, 30)
            cv2.ellipse(image, (x-size//2, y), (wing_size, wing_size//2), wing_angle, 0, 180, color, -1)
            cv2.ellipse(image, (x+size//2, y), (wing_size, wing_size//2), wing_angle, 0, 180, color, -1)
        
        bird_positions.append((x, y))
    
    # Add noise to make detection more challenging
    noise = np.zeros_like(image)
    cv2.randn(noise, 0, 10)
    image = cv2.add(image, noise)
    
    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Generated sample image with {num_birds} birds at {output_path}")
    
    return output_path, bird_positions

def test_bird_counter():
    """Test the bird counter with a generated sample image."""
    # Create a directory for test results if it doesn't exist
    if not os.path.exists("test_results"):
        os.makedirs("test_results")
    
    # Generate sample images with different numbers of birds
    bird_counts = [20, 50, 100]
    
    results = []
    
    for count in bird_counts:
        sample_path, actual_positions = generate_sample_image(
            num_birds=count, 
            output_path=f"test_results/sample_birds_{count}.jpg"
        )
        
        # Run basic method
        print(f"\nTesting basic method with {count} birds...")
        detected_count, result_img = count_birds(
            sample_path, 
            save_result=True, 
            debug=False
        )
        
        # Calculate accuracy
        accuracy = (detected_count / count) * 100
        results.append({
            "actual_count": count,
            "detected_count_basic": detected_count,
            "accuracy_basic": accuracy
        })
        
        print(f"Actual birds: {count}")
        print(f"Detected birds (basic): {detected_count}")
        print(f"Accuracy (basic): {accuracy:.2f}%")
        
        # Run advanced method if available
        if ADVANCED_METHOD_AVAILABLE:
            print(f"\nTesting advanced method with {count} birds...")
            detected_count_adv, result_img_adv = count_birds_advanced(
                sample_path, 
                save_result=True, 
                debug=False
            )
            
            # Calculate accuracy
            accuracy_adv = (detected_count_adv / count) * 100
            results[-1]["detected_count_advanced"] = detected_count_adv
            results[-1]["accuracy_advanced"] = accuracy_adv
            
            print(f"Actual birds: {count}")
            print(f"Detected birds (advanced): {detected_count_adv}")
            print(f"Accuracy (advanced): {accuracy_adv:.2f}%")
    
    # Display summary
    print("\n=== Test Summary ===")
    print("Birds | Basic Method | Advanced Method")
    print("-----|--------------|----------------")
    
    for result in results:
        basic = f"{result['detected_count_basic']} ({result['accuracy_basic']:.2f}%)"
        
        if ADVANCED_METHOD_AVAILABLE and "detected_count_advanced" in result:
            advanced = f"{result['detected_count_advanced']} ({result['accuracy_advanced']:.2f}%)"
        else:
            advanced = "N/A"
            
        print(f"{result['actual_count']} | {basic} | {advanced}")

if __name__ == "__main__":
    test_bird_counter() 