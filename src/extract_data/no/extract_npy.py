"""
手语识别特征提取脚本
该脚本用于从手语图片中批量提取手部特征，并将其保存为NumPy数组格式
主要功能：
1. 使用MediaPipe从图片中提取手部关键点
2. 将关键点坐标转换为特征向量
3. 保存特征向量和对应的标签
"""

import cv2
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm

# 配置TensorFlow日志级别，减少不必要的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 配置文件路径
IMAGE_DIR = 'data/train'  # 训练图片目录
OUTPUT_DIR = 'data/processed'  # 输出目录
FEATURES_FILE = os.path.join(OUTPUT_DIR, 'features.npy')  # 特征文件
LABELS_FILE = os.path.join(OUTPUT_DIR, 'labels.npy')  # 标签文件

def extract_hand_features(image_path, hands_detector):
    """
    从单个图像中提取手部特征
    
    参数:
        image_path: 输入图像的路径
        hands_detector: MediaPipe Hands 检测器实例
    
    返回:
        numpy.ndarray: 63维特征向量（21个关键点 * 3个坐标），如果未检测到手则返回None
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # 转换到RGB颜色空间（MediaPipe要求）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 处理图像，检测手部关键点
    results = hands_detector.process(image_rgb)
    
    # 如果没有检测到手，返回None
    if not results.multi_hand_landmarks:
        return None
    
    # 提取第一只手的所有关键点
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # 将21个关键点的x,y,z坐标展平为一维向量
    # 每个关键点有3个坐标，共63维
    features = np.array([[lm.x, lm.y, lm.z] 
                        for lm in hand_landmarks.landmark]).flatten()
    return features

def main():
    """
    主函数：批量处理图像并保存特征
    """
    print("=== 开始提取手部特征 ===")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 初始化MediaPipe Hands检测器
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=True,    # 静态图片模式
        max_num_hands=1,           # 最多检测一只手
        min_detection_confidence=0.3  # 检测置信度阈值
    )
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(IMAGE_DIR) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("❌ 没有找到图像文件！")
        return
    
    # 存储提取的特征和标签
    features_list = []
    labels_list = []
    
    # 处理每个图像
    print("📸 正在处理图像...")
    with hands_detector:
        for filename in tqdm(image_files, desc="处理进度"):
            filepath = os.path.join(IMAGE_DIR, filename)
            
            # 提取特征
            features = extract_hand_features(filepath, hands_detector)
            
            if features is not None:
                features_list.append(features)
                # 从文件名获取标签（第一个字符）
                label = filename[0].upper()
                labels_list.append(label)
    
    if not features_list:
        print("❌ 没有成功提取到任何特征！")
        return
    
    # 转换为NumPy数组
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    
    # 保存特征和标签
    np.save(FEATURES_FILE, features_array)
    np.save(LABELS_FILE, labels_array)
    
    print("\n=== 提取完成 ===")
    print(f"✅ 成功提取特征: {len(features_list)} 个样本")
    print(f"✅ 特征维度: {features_array.shape}")
    print(f"💾 数据已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()