import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

def extract_hand_landmarks(image_path, hands_detector):
    """从单个图像中提取手部关键点和方向信息"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # 转换颜色空间
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 处理图像
    results = hands_detector.process(image_rgb)
    
    if not results.multi_hand_landmarks or not results.multi_handedness:
        return None
    
    # 获取第一只手的信息
    hand_landmarks = results.multi_hand_landmarks[0]
    handedness = results.multi_handedness[0]
    
    # 获取手部方向和置信度
    hand_type = handedness.classification[0].label  # 'Left' 或 'Right'
    confidence = handedness.classification[0].score
    
    # 提取关键点坐标
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    
    return {
        'landmarks': np.array(landmarks),
        'hand_type': hand_type,
        'confidence': confidence
    }

def process_directory(input_dir, output_dir):
    """处理整个目录的图片并保存结果"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化 MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,  # 只检测一只手
        min_detection_confidence=0.5
    )
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 用于存储所有特征和信息
    features_list = []
    hand_types = []
    confidences = []
    labels = []
    processed_count = 0
    failed_count = 0
    
    # 处理每个图片
    for filename in tqdm(image_files, desc="处理进度"):
        image_path = os.path.join(input_dir, filename)
        
        # 提取特征
        result = extract_hand_landmarks(image_path, hands)
        
        if result is not None:
            features_list.append(result['landmarks'])
            hand_types.append(result['hand_type'])
            confidences.append(result['confidence'])
            # 从文件名获取标签
            label = filename[0].upper()
            labels.append(label)
            processed_count += 1
        else:
            failed_count += 1
            print(f"\n未能从 {filename} 提取手部关键点")
    
    # 转换为 numpy 数组
    if processed_count > 0:
        features_array = np.array(features_list)
        labels_array = np.array(labels)
        hand_types_array = np.array(hand_types)
        confidences_array = np.array(confidences)
        
        # 保存结果
        np.save(os.path.join(output_dir, 'hand_features.npy'), features_array)
        np.save(os.path.join(output_dir, 'labels.npy'), labels_array)
        np.save(os.path.join(output_dir, 'hand_types.npy'), hand_types_array)
        np.save(os.path.join(output_dir, 'confidences.npy'), confidences_array)
        
        print(f"\n处理完成:")
        print(f"✓ 成功处理: {processed_count} 张图片")
        print(f"✗ 处理失败: {failed_count} 张图片")
        print(f"特征维度: {features_array.shape}")
        print(f"检测到的手部类型: {np.unique(hand_types_array)}")
        print(f"平均置信度: {confidences_array.mean():.4f}")
    else:
        print("\n没有成功处理任何图片")

if __name__ == "__main__":
    # 配置路径
    INPUT_DIR = "data/train"
    OUTPUT_DIR = "data/processed"
    
    print("开始批量处理图片...")
    process_directory(INPUT_DIR, OUTPUT_DIR)