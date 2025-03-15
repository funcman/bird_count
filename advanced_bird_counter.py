import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def count_birds_advanced(image_path, save_result=True, debug=False):
    """
    Count the number of birds in an image using advanced techniques and mark them with dots.
    
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
    
    # 2. 将图像转换为不同颜色空间，提供更多检测特征
    # Convert the image to different color spaces for more detection features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    if debug:
        cv2.imshow("Gray Image", gray)
        cv2.imshow("HSV Image", hsv[:, :, 2])  # 显示V通道
        cv2.waitKey(0)
    
    # 3. 使用更大的核进行高斯模糊，更有效地去除噪点
    # Use a larger kernel for Gaussian blur to more effectively remove noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    if debug:
        cv2.imshow("Blurred Image", blurred)
        cv2.waitKey(0)
    
    # 4. 更准确地分离天空和地面
    # More accurate separation of sky and ground
    
    # 4.1 使用Otsu方法自动确定阈值
    # Use Otsu's method to automatically determine threshold
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4.2 使用梯度计算和直方图分析来找到天空/地面边界
    # Use gradient calculation and histogram analysis to find sky/ground boundary
    row_means = np.mean(blurred, axis=1)
    gradient = np.abs(np.gradient(row_means))
    
    # 使用滑动窗口来平滑梯度数据，减少噪声影响
    # Use sliding window to smooth gradient data and reduce noise influence
    window_size = 15
    smoothed_gradient = np.convolve(gradient, np.ones(window_size)/window_size, mode='same')
    
    # 在图像下半部分寻找最大梯度变化
    # Look for maximum gradient change in the lower half of the image
    half_idx = len(smoothed_gradient) // 2
    sky_ground_boundary = np.argmax(smoothed_gradient[half_idx:]) + half_idx
    
    # 提取天空区域（剔除地面）
    # Extract the sky region (removing the ground)
    sky_region = blurred[:sky_ground_boundary, :]
    sky_hsv = hsv[:sky_ground_boundary, :, :]
    
    if debug:
        boundary_img = original.copy()
        cv2.line(boundary_img, (0, sky_ground_boundary), (image.shape[1], sky_ground_boundary), (0, 255, 0), 2)
        cv2.imshow("Sky-Ground Boundary", boundary_img)
        cv2.imshow("Sky Region", sky_region)
        cv2.waitKey(0)
    
    # 5. 多模态检测：结合亮度和颜色特征
    # Multi-modal detection: combining brightness and color features
    
    # 5.1 基于亮度的自适应阈值分割
    # Brightness-based adaptive thresholding
    thresh_brightness = cv2.adaptiveThreshold(
        sky_region,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    
    # 5.2 基于HSV的颜色分割
    # HSV-based color segmentation
    # 提取V通道，鸟通常比天空背景暗
    # Extract V channel, birds are usually darker than the sky background
    v_channel = sky_hsv[:, :, 2]
    
    # 应用阈值分割
    # Apply thresholding
    _, thresh_color = cv2.threshold(
        v_channel,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    # 5.3 组合两种分割结果
    # Combine the two segmentation results
    combined_thresh = cv2.bitwise_or(thresh_brightness, thresh_color)
    
    if debug:
        cv2.imshow("Brightness Threshold", thresh_brightness)
        cv2.imshow("Color Threshold", thresh_color)
        cv2.imshow("Combined Threshold", combined_thresh)
        cv2.waitKey(0)
    
    # 6. 更复杂的形态学操作，提高鸟的检测准确性
    # More complex morphological operations to improve bird detection accuracy
    
    # 6.1 鸟的形状通常是椭圆形，使用椭圆形结构元素
    # Birds are usually elliptical, use an elliptical structuring element
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # 6.2 去除小噪点
    # Remove small noise
    opening = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # 6.3 连接相近的组件（可能是同一只鸟的不同部分）
    # Connect nearby components (possibly different parts of the same bird)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
    
    if debug:
        cv2.imshow("After Opening", opening)
        cv2.imshow("After Closing", closing)
        cv2.waitKey(0)
    
    # 7. 更智能的轮廓提取和过滤
    # More intelligent contour extraction and filtering
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 使用形状特征（面积、周长、长宽比等）来过滤轮廓
    # Use shape features (area, perimeter, aspect ratio, etc.) to filter contours
    min_contour_area = 5  # 最小轮廓面积 (minimum contour area)
    max_contour_area = 500  # 最大轮廓面积 (maximum contour area)
    
    # 存储有效鸟类轮廓和中心点
    # Store valid bird contours and center points
    valid_contours = []
    contour_centers = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 面积检查
        # Area check
        if not (min_contour_area < area < max_contour_area):
            continue
        
        # 计算轮廓形状特征
        # Calculate contour shape features
        perimeter = cv2.arcLength(contour, True)
        
        # 避免除零错误
        # Avoid division by zero
        if perimeter == 0:
            continue
            
        # 计算圆形度（圆度）：4π×面积/周长²，越接近1表示越接近圆形
        # Calculate circularity: 4π×area/perimeter², closer to 1 means closer to circular
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # 鸟类通常具有一定的圆形度范围
        # Birds usually have a certain range of circularity
        if not (0.1 < circularity < 0.9):
            continue
            
        # 获取边界框并计算长宽比
        # Get bounding box and calculate aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        
        # 避免除零错误
        # Avoid division by zero
        if w == 0 or h == 0:
            continue
            
        aspect_ratio = float(w) / h
        
        # 鸟类通常具有合理的长宽比
        # Birds usually have a reasonable aspect ratio
        if not (0.3 < aspect_ratio < 3.0):
            continue
        
        # 通过形状检查的轮廓被认为是有效的鸟类轮廓
        # Contours that pass the shape check are considered valid bird contours
        valid_contours.append(contour)
        
        # 计算轮廓的中心点
        # Calculate the center of the contour
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            contour_centers.append([cX, cY])
    
    # 8. 使用聚类算法处理重叠的鸟
    # Use clustering algorithm to handle overlapping birds
    if len(contour_centers) > 0:
        contour_centers = np.array(contour_centers)
        
        # 使用DBSCAN聚类算法
        # Use DBSCAN clustering algorithm
        # eps: 同一聚类中两个样本间的最大距离
        # min_samples: 形成核心点所需的最小样本数
        # eps: maximum distance between two samples in the same cluster
        # min_samples: minimum number of samples to form a core point
        clustering = DBSCAN(eps=20, min_samples=1).fit(contour_centers)
        
        # 获取聚类标签
        # Get cluster labels
        labels = clustering.labels_
        
        # 聚类数量（去除噪声点，标签为-1）
        # Number of clusters (excluding noise points, label = -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # 创建每个聚类的中心点列表
        # Create a list of center points for each cluster
        cluster_centers = []
        for i in range(n_clusters):
            # 获取属于当前聚类的所有点
            # Get all points belonging to the current cluster
            cluster_points = contour_centers[labels == i]
            
            # 计算该聚类的中心点
            # Calculate the center point of this cluster
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers.append(cluster_center)
    else:
        cluster_centers = []
    
    # 9. 在原始图像上标记每只鸟并计数
    # Mark each bird on the original image and count them
    result_img = original.copy()
    
    # 创建一个掩码用于展示
    # Create a mask for display
    bird_mask = np.zeros((sky_ground_boundary, image.shape[1]), dtype=np.uint8)
    
    # 在掩码上绘制有效的轮廓
    # Draw valid contours on the mask
    cv2.drawContours(bird_mask, valid_contours, -1, 255, -1)
    
    # 在结果图像上标记每个鸟类聚类中心
    # Mark each bird cluster center on the result image
    bird_count = len(cluster_centers)
    for i, center in enumerate(cluster_centers):
        cX, cY = int(center[0]), int(center[1])
        
        # 在图像上用红点标记鸟的位置
        # Mark bird positions with red dots on the image
        cv2.circle(result_img, (cX, cY), 5, (0, 0, 255), -1)
        
        # 如果需要，可以在每个点旁边添加数字标签
        # Optionally add numeric labels next to each point
        # cv2.putText(result_img, str(i+1), (cX+10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 在图像上显示鸟的总数
    # Display the total number of birds on the image
    cv2.putText(result_img, f"Bird Count: {bird_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if debug:
        # 创建轮廓展示图像
        # Create contour display image
        contour_img = original.copy()
        cv2.drawContours(contour_img[:sky_ground_boundary], valid_contours, -1, (0, 255, 0), 2)
        
        # 展示掩码图像
        # Display mask image
        mask_display = cv2.cvtColor(bird_mask, cv2.COLOR_GRAY2BGR)
        mask_display = np.zeros_like(original)
        mask_display[:sky_ground_boundary, :, 1] = bird_mask  # 用绿色通道显示掩码
        
        # 叠加掩码和原始图像
        # Overlay mask and original image
        alpha = 0.5
        overlay = cv2.addWeighted(original, 1, mask_display, alpha, 0)
        
        cv2.imshow("Bird Mask", bird_mask)
        cv2.imshow("Detected Birds (Contours)", contour_img)
        cv2.imshow("Bird Mask Overlay", overlay)
        cv2.imshow("Final Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 保存结果图像
    # Save the result image
    if save_result:
        output_path = image_path.rsplit(".", 1)[0] + "_result_advanced.jpg"
        cv2.imwrite(output_path, result_img)
        print(f"Result saved to {output_path}")
    
    return bird_count, result_img

def main():
    """Main function to parse arguments and process the image."""
    parser = argparse.ArgumentParser(description="Count birds in an image using advanced techniques")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--debug", action="store_true", help="Show intermediate steps")
    parser.add_argument("--no-save", dest="save", action="store_false", 
                        help="Don't save the result image")
    parser.set_defaults(save=True, debug=False)
    
    args = parser.parse_args()
    
    try:
        bird_count, _ = count_birds_advanced(args.image_path, save_result=args.save, debug=args.debug)
        print(f"Total birds detected: {bird_count}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 