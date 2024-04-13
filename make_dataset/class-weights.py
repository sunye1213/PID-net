from glob import glob
from PIL import Image
import numpy as np
from collections import Counter

def calculate_class_weights(label_dir, num_classes=66):
    # 获取所有标签图片的文件名
    label_files = glob(f"{label_dir}/*.png")
    
    # 初始化计数器，用于统计每个类别的像素数
    # 确保计数器初始化时包含所有类别
    class_counts = Counter({i: 0 for i in range(num_classes)})
    i = 0
    # 遍历每个标签图片
    for label_file in label_files:
        # 读取图片并转换为numpy数组
        label = np.array(Image.open(label_file))
        
        # 更新计数器，计算每个类别的像素数
        class_counts.update(label.flatten())
        i += 1
        print(f"Processing {i}/{len(label_files)}")
        if i == 500:
            break
    # 计算每个类别的权重
    total_counts = sum(class_counts.values())
    class_weights = {class_id: total_counts/count for class_id, count in class_counts.items() if count > 0}
    
    # 归一化权重，使得最大的权重为1
    max_weight = max(class_weights.values())
    normalized_weights = {class_id: weight/max_weight for class_id, weight in class_weights.items()}
    
    return normalized_weights

# 替换为你的标签图片存储路径
label_dir = '../data/mapill/label/train'
weights = calculate_class_weights(label_dir)
# print("Class Weights:", weights)
# 打印值 ，不打印键
print("Class Weights:", list(weights.values()))
#打印key-value的个数
print(len(weights))
