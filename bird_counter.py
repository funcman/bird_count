import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

def count_birds(image_path, save_result=True, debug=False):
    """
    Count the number of birds in an image and mark them with dots.
    
    Args:
        image_path (str): Path to the input image
        save_result (bool): Whether to save the result image
        debug (bool): Whether to show intermediate processing steps
    
    Returns:
        int: Number of birds detected
        numpy.ndarray: Result image with birds marked
    """
    # 1. 读取原始图像
    # Read the original image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    
    # 保存原始图像的副本用于最终标记
    # Save a copy of the original image for final marking
    original = image.copy()
    
    # 2. 将图像转换为灰度图，便于后续处理
    # Convert the image to grayscale for easier processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if debug:
        cv2.imshow("Gray Image", gray)
        cv2.waitKey(0)
    
    # 3. 使用高斯模糊减少噪点，提高处理精度
    # Apply Gaussian blur to reduce noise and improve processing accuracy
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if debug:
        cv2.imshow("Blurred Image", blurred)
        cv2.waitKey(0)
    
    # 4. 分离天空和地面
    # Separate sky and ground
    # 假设图像下部为地面，通过分析图像垂直方向的亮度变化来确定天空与地面的分界线
    # Assume the lower part of the image is ground, determine the boundary by analyzing brightness changes
    
    # 计算每一行的平均亮度
    # Calculate average brightness for each row
    row_means = np.mean(blurred, axis=1)
    
    # 计算亮度梯度（变化率）
    # Calculate brightness gradient (rate of change)
    gradient = np.abs(np.gradient(row_means))
    
    # 找到最大梯度变化点作为天空与地面的分界线
    # Find the point of maximum gradient change as the boundary between sky and ground
    sky_ground_boundary = np.argmax(gradient[len(gradient)//2:]) + len(gradient)//2
    
    # 提取天空区域（剔除地面）
    # Extract the sky region (removing the ground)
    sky_region = blurred[:sky_ground_boundary, :]
    
    if debug:
        cv2.line(image, (0, sky_ground_boundary), (image.shape[1], sky_ground_boundary), (0, 255, 0), 2)
        cv2.imshow("Sky-Ground Boundary", image)
        cv2.waitKey(0)
        cv2.imshow("Sky Region", sky_region)
        cv2.waitKey(0)
    
    # 5. 应用自适应阈值分割，凸显飞鸟
    # Apply adaptive thresholding to highlight birds
    # 鸟通常比天空背景更暗，使用自适应阈值可以更好地适应不同亮度区域
    # Birds are usually darker than the sky background, adaptive thresholding adapts to different brightness regions
    thresh = cv2.adaptiveThreshold(
        sky_region,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    
    if debug:
        cv2.imshow("Thresholded Image", thresh)
        cv2.waitKey(0)
    
    # 6. 应用形态学操作来消除噪点并增强鸟的轮廓
    # Apply morphological operations to remove noise and enhance bird contours
    # 定义一个小的椭圆形结构元素，与鸟的形状相似
    # Define a small elliptical structuring element similar to bird shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # 开运算（先腐蚀后膨胀）去除小噪点
    # Opening operation (erosion followed by dilation) to remove small noise
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    if debug:
        cv2.imshow("After Opening", opening)
        cv2.waitKey(0)
    
    # 闭运算（先膨胀后腐蚀）填充鸟的内部空隙
    # Closing operation (dilation followed by erosion) to fill holes inside birds
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    if debug:
        cv2.imshow("After Closing", closing)
        cv2.waitKey(0)
    
    # 7. 查找轮廓，每个轮廓代表一只可能的鸟
    # Find contours, each contour represents a potential bird
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 8. 过滤轮廓，移除太小或太大的轮廓，这些可能是噪点或云朵
    # Filter contours, remove those that are too small or too large (likely noise or clouds)
    min_contour_area = 5  # 最小轮廓面积 (minimum contour area)
    max_contour_area = 500  # 最大轮廓面积 (maximum contour area)
    
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_contour_area < area < max_contour_area:
            filtered_contours.append(contour)
    
    # 9. 在原始图像上标记每只鸟并计数
    # Mark each bird on the original image and count them
    result_img = original.copy()
    
    # 创建一个全零的掩码，大小与天空区域相同
    # Create a zero mask with the same size as the sky region
    bird_mask = np.zeros_like(sky_region)
    
    # 在掩码上绘制过滤后的轮廓
    # Draw the filtered contours on the mask
    cv2.drawContours(bird_mask, filtered_contours, -1, 255, -1)
    
    # 标记每只鸟的中心位置
    # Mark the center position of each bird
    bird_centers = []
    for contour in filtered_contours:
        # 计算轮廓的中心点
        # Calculate the center of the contour
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # 将天空区域坐标转换为原始图像坐标
            # Convert sky region coordinates to original image coordinates
            bird_centers.append((cX, cY))
            
            # 在结果图像上用红色圆点标记鸟的位置
            # Mark bird positions with red dots on the result image
            cv2.circle(result_img, (cX, cY), 5, (0, 0, 255), -1)
    
    # 在图像上显示鸟的总数
    # Display the total number of birds on the image
    bird_count = len(filtered_contours)
    cv2.putText(result_img, f"Bird Count: {bird_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if debug:
        # 在调试模式下显示检测到的轮廓
        # Display detected contours in debug mode
        contour_img = original.copy()
        cv2.drawContours(contour_img, filtered_contours, -1, (0, 255, 0), 2)
        cv2.imshow("Detected Birds (Contours)", contour_img)
        cv2.waitKey(0)
        
        cv2.imshow("Final Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 保存结果图像
    # Save the result image
    if save_result:
        output_path = image_path.rsplit(".", 1)[0] + "_result.jpg"
        cv2.imwrite(output_path, result_img)
        print(f"Result saved to {output_path}")
    
    return bird_count, result_img

def main():
    """Main function to parse arguments and process the image."""
    parser = argparse.ArgumentParser(description="Count birds in an image")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--debug", action="store_true", help="Show intermediate steps")
    parser.add_argument("--no-save", dest="save", action="store_false", 
                        help="Don't save the result image")
    parser.set_defaults(save=True, debug=False)
    
    args = parser.parse_args()
    
    try:
        bird_count, _ = count_birds(args.image_path, save_result=args.save, debug=args.debug)
        print(f"Total birds detected: {bird_count}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 