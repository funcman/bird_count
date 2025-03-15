# 飞鸟计数器使用示例 (Bird Counter Usage Examples)

本文档提供了使用飞鸟计数器程序的详细示例和最佳实践。

## 基本用法示例

### 1. 处理单张图像

```bash
# 使用基本方法处理图像
python bird_counter.py path/to/your/bird_image.jpg

# 使用高级方法处理图像
python advanced_bird_counter.py path/to/your/bird_image.jpg

# 或使用组合脚本选择方法
python bird_count.py path/to/your/bird_image.jpg --method advanced
```

### 2. 查看中间处理步骤

通过添加`--debug`标志，可以查看图像处理的每个步骤，这对于理解算法工作原理或调试特定图像的处理问题非常有用：

```bash
python bird_count.py path/to/your/bird_image.jpg --debug
```

这会显示以下中间步骤：
- 灰度图转换
- 高斯模糊
- 天空/地面分界
- 阈值分割
- 形态学处理
- 轮廓检测
- 最终结果

### 3. 比较两种方法

要比较基本方法和高级方法的性能，可以使用`--compare`标志：

```bash
python bird_count.py path/to/your/bird_image.jpg --compare
```

这将运行两种方法并输出比较结果，包括每种方法检测到的鸟的数量和差异。使用`--compare --debug`可以同时看到两种方法的结果图像并排显示。

## 批处理多个图像

以下是处理目录中多个图像的示例脚本：

```python
import os
import sys
import argparse
from bird_count import count_birds, count_birds_advanced

def process_directory(directory, method="basic", save_results=True):
    """处理目录中的所有图像文件"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    results = {}
    
    # 确保输出目录存在
    output_dir = os.path.join(directory, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 处理目录中的所有图像
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if os.path.isfile(filepath) and file_ext in image_extensions:
            print(f"Processing {filename}...")
            
            try:
                if method == "basic":
                    bird_count, _ = count_birds(filepath, save_result=save_results)
                else:
                    bird_count, _ = count_birds_advanced(filepath, save_result=save_results)
                
                results[filename] = bird_count
                print(f"  Detected {bird_count} birds")
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
    
    # 保存结果到CSV文件
    with open(os.path.join(output_dir, "bird_counts.csv"), "w") as f:
        f.write("Filename,Bird Count\n")
        for filename, count in results.items():
            f.write(f"{filename},{count}\n")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple bird images in a directory")
    parser.add_argument("directory", help="Directory containing bird images")
    parser.add_argument("--method", choices=["basic", "advanced"], default="basic", 
                      help="Method to use for bird counting")
    parser.add_argument("--no-save", dest="save", action="store_false", 
                      help="Don't save result images")
    
    args = parser.parse_args()
    process_directory(args.directory, method=args.method, save_results=args.save)
```

将此代码保存为`batch_process.py`，然后使用以下命令运行：

```bash
python batch_process.py path/to/your/image/directory --method advanced
```

## 调整参数以优化检测效果

对于特定类型的图像，您可能需要调整一些参数以获得最佳结果。

### 基本方法参数调整

在`bird_counter.py`文件中，您可以修改以下参数：

1. 高斯模糊核大小：增大可减少更多噪点，但可能导致小鸟丢失
   ```python
   blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 调整(5, 5)为(3, 3)或(7, 7)
   ```

2. 轮廓面积过滤范围：调整以匹配您图像中鸟的大小
   ```python
   min_contour_area = 5    # 最小轮廓面积
   max_contour_area = 500  # 最大轮廓面积
   ```

### 高级方法参数调整

在`advanced_bird_counter.py`文件中，还可以调整以下参数：

1. DBSCAN聚类参数：
   ```python
   clustering = DBSCAN(eps=20, min_samples=1).fit(contour_centers)
   ```
   - `eps`：同一聚类中两点间的最大距离，较小的值会创建更多的小聚类
   - `min_samples`：形成核心点所需的最小样本数

2. 形状特征过滤参数：
   ```python
   # 圆形度范围
   if not (0.1 < circularity < 0.9):
       continue
       
   # 长宽比范围
   if not (0.3 < aspect_ratio < 3.0):
       continue
   ```

## 处理挑战性图像的技巧

1. **处理密集鸟群**：
   - 使用高级方法的DBSCAN聚类
   - 调整`eps`参数以更好地分离鸟群

2. **处理复杂背景**：
   - 如果图像下部有复杂地面信息，确保使用debug模式确认天空/地面分界线是否正确
   - 如果分界线不准确，可以手动指定分界线位置

3. **处理低对比度图像**：
   - 可以在处理前对图像进行预处理，如对比度增强
   ```python
   # 增强对比度的预处理代码示例
   def enhance_contrast(image):
       lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
       l, a, b = cv2.split(lab)
       clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
       cl = clahe.apply(l)
       enhanced_lab = cv2.merge((cl, a, b))
       return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
   ```

## 测试程序

可以使用包含的测试脚本生成带有已知鸟数量的样本图像：

```bash
python test_bird_counter.py
```

这会生成三个样本图像（含有20、50和100只鸟），并测试两种方法的准确性。测试结果将显示每种方法检测到的鸟数量以及相对于实际数量的百分比准确度。

## 常见问题解答

1. **Q: 为什么程序计数结果与实际鸟数不符？**  
   A: 可能原因包括：
   - 鸟类重叠过多
   - 图像对比度低
   - 鸟类与背景颜色相似
   - 轮廓面积参数不适合图像中鸟的大小

2. **Q: 程序运行很慢，如何优化？**  
   A: 尝试以下方法：
   - 降低图像分辨率
   - 使用基本方法而非高级方法
   - 减少debug模式的使用
   - 如果批处理多个图像，考虑使用多进程并行处理

3. **Q: 检测到了明显不是鸟的对象，如何解决？**  
   A: 调整轮廓过滤参数，或使用高级方法中的形状特征过滤 