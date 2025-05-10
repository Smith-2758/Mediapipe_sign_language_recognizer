"""
手语识别测试评估程序
该程序用于评估手语识别模型在测试集上的性能
主要功能：
1. 处理测试图像集
2. 生成混淆矩阵
3. 输出分类报告
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import re

# 忽略 sklearn 警告信息
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DataConversionWarning, module="sklearn")

# 模型相关文件路径配置
model_path = 'model/svm_model/relative_normalized/new_hand_gesture_recognition_model_partial_normalized.pkl'
scaler_path = 'model/svm_model/relative_normalized/new_hand_gesture_recognition_scaler_partial_normalized.pkl'
label_encoder_path = 'model/svm_model/relative_normalized/new_hand_gesture_recognition_label_encoder_partial_normalized.pkl'

# 检查模型文件是否存在
if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(label_encoder_path):
    print("错误：模型文件、Scaler文件或LabelEncoder文件未找到。请确保它们位于正确的路径。")
    exit()

# 加载训练好的模型和预处理器
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(label_encoder_path)

# 初始化 MediaPipe
mp_hands = mp.solutions.hands

def normalize_inference_landmarks(landmarks):
    """
    使用相对坐标和部分归一化处理手部关键点坐标
    
    参数：
        landmarks: MediaPipe检测到的手部关键点
    
    返回：
        组合的归一化坐标：
        - 归一化的x,y坐标（不含手腕点）
        - 相对于手腕的z坐标
    """
    if not landmarks:
        return None
        
    # 将关键点转换为numpy数组
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    
    # 使用手腕点作为参考点
    wrist = landmarks_array[0]
    # 计算相对于手腕的坐标
    relative_coords = landmarks_array - wrist
    
    # 对除手腕外的其他点进行x,y坐标归一化
    relative_xy_no_wrist = relative_coords[1:, :2]
    # 计算x,y的最小值和最大值
    min_xy = relative_xy_no_wrist.min(axis=0)
    max_xy = relative_xy_no_wrist.max(axis=0)
    # 添加小量避免除零错误
    range_xy = max_xy - min_xy + 1e-6
    # 归一化到[0,1]区间
    normalized_xy_no_wrist = (relative_xy_no_wrist - min_xy) / range_xy
    
    # 提取相对z坐标（不含手腕）
    relative_z_no_wrist = relative_coords[1:, 2]
    
    # 组合归一化的x,y坐标和相对z坐标
    combined = np.hstack((normalized_xy_no_wrist, 
                         relative_z_no_wrist.reshape(-1, 1))).flatten()
    return combined.reshape(1, -1)

def process_test_images():
    """
    处理测试图像并生成混淆矩阵
    """
    test_dir = 'New_data/test'
    true_labels = []
    predicted_labels = []
    
    # 初始化MediaPipe
    test_hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    
    # 获取所有测试图像
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print("开始处理测试图像...")
    for img_name in test_images:
        # 处理特殊类别
        if img_name.lower().startswith('space'):
            true_label = '1'
        elif img_name.lower().startswith('del'):
            true_label = '2'
        else:
            # 从文件名中提取真实标签（第一个字母）
            match = re.match(r'([A-Z])', img_name)
            if match:
                true_label = match.group(1)
            else:
                print(f"警告：无法从文件名 {img_name} 提取标签")
                continue
                
        true_labels.append(true_label)
        
        # 读取和处理图像
        img_path = os.path.join(test_dir, img_name)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = test_hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # 处理手部关键点
            processed_landmarks = normalize_inference_landmarks(results.multi_hand_landmarks[0])
            if processed_landmarks is not None:
                # 特征标准化
                processed_landmarks_scaled = scaler.transform(processed_landmarks)
                # 预测手势
                prediction_encoded = model.predict(processed_landmarks_scaled)[0]
                predicted_label = label_encoder.inverse_transform([prediction_encoded])[0]
                predicted_labels.append(predicted_label)
            else:
                predicted_labels.append("未检测")
        else:
            predicted_labels.append("未检测")
            
        print(f"处理：{img_name}, 真实标签：{true_label}, 预测标签：{predicted_labels[-1]}")
    
    # 计算混淆矩阵
    classes = sorted(list(set(true_labels + predicted_labels)))
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('手语识别测试结果 - 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    # 保存混淆矩阵
    plt.savefig('results/test_confusion_matrix.png')
    print(f"混淆矩阵已保存到：results/test_confusion_matrix.png")
    
    # 打印分类报告
    print("\n分类报告：")
    print(classification_report(true_labels, predicted_labels))
    
    test_hands.close()

if __name__ == "__main__":
    process_test_images()