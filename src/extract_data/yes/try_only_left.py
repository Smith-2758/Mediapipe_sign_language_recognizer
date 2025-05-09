"""
手语识别关键点提取脚本
这个脚本用于从手语图片中提取手部关键点坐标，并将其保存为CSV格式
主要功能：
1. 使用MediaPipe检测手部关键点
2. 统一处理为左手数据（右手图像会被镜像处理）
3. 分别处理训练集和测试集图像
4. 将结果保存为CSV文件，包含21个关键点的x,y,z坐标
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# 初始化 MediaPipe Hands 模型
mp_hands = mp.solutions.hands

def process_and_extract(image, hands):
    """
    处理单张图片并提取手部关键点
    
    参数:
        image: 输入图像
        hands: MediaPipe Hands 模型实例
    
    返回:
        landmarks: 手部关键点数据，包含21个3D坐标点
        handedness: 手的类型（左手/右手）
    """
    # 将BGR图像转换为RGB格式，因为MediaPipe需要RGB输入
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 使用MediaPipe处理图像
    results = hands.process(image_rgb)
    landmarks = None
    handedness = None
    # 如果检测到手和手的类型
    if results.multi_hand_landmarks and results.multi_handedness:
        # 只取第一个检测到的手
        landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].label
    return landmarks, handedness

def extract_and_save_landmarks(image_path, output_csv, hands):
    """
    从图像中提取手部关键点并保存到CSV文件
    
    参数:
        image_path: 输入图像的路径
        output_csv: CSV写入器对象
        hands: MediaPipe Hands 模型实例
    
    返回:
        bool: 处理成功返回True，失败返回False
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法加载图像 {image_path}")
        return False

    # 从文件名获取标签（第一个字符）
    label = os.path.basename(image_path)[0].upper()
    landmarks, handedness = process_and_extract(image, hands)

    if handedness == 'Right':
        # 如果是右手，翻转图像并重新处理
        flipped_image = cv2.flip(image, 1)
        flipped_landmarks, _ = process_and_extract(flipped_image, hands)
        if flipped_landmarks:
            landmarks = flipped_landmarks
            handedness = 'Left'  # 标记为处理后的“左手”
        else:
            print(f"警告：右手图像翻转后未检测到手 {image_path}")
            return False
    elif handedness == 'Left':
        pass # 保持原样
    else:
        print(f"警告：未检测到手或无法识别左右手 {image_path}")
        return False

    if landmarks:
        # 将标签和手的类型写入CSV
        row = [label, handedness]
        # 将21个关键点的x, y, z坐标写入CSV
        for landmark in landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z])
        output_csv.writerow(row)
        return True
    else:
        return False

# 定义输出 CSV 文件名
OUTPUT_CSV_PATH_TRAIN = 'data/processed/new_hand_landmarks_train_left_only.csv'
OUTPUT_CSV_PATH_TEST = 'data/processed/new_hand_landmarks_test_left_only.csv'

# 定义图片所在的目录
TRAIN_IMAGE_DIR = 'data/train'
TEST_IMAGE_DIR = 'data/test'

# 处理训练集
with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
    with open(OUTPUT_CSV_PATH_TRAIN, 'w', newline='', encoding='utf-8') as csvfile_train:
        writer_train = csv.writer(csvfile_train)
        # 写入CSV文件头
        header = ['label', 'handedness']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer_train.writerow(header)

        # 获取训练集图像文件列表
        train_image_files = [os.path.join(TRAIN_IMAGE_DIR, f) for f in os.listdir(TRAIN_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in train_image_files:
            extract_and_save_landmarks(image_file, writer_train, hands)
        print(f"✅ 训练集关键点坐标（仅左手或镜像右手）已保存到 {OUTPUT_CSV_PATH_TRAIN}")

# 处理测试集
with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
    with open(OUTPUT_CSV_PATH_TEST, 'w', newline='', encoding='utf-8') as csvfile_test:
        writer_test = csv.writer(csvfile_test)
        # 写入CSV文件头
        header = ['label', 'handedness']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer_test.writerow(header)

        # 获取测试集图像文件列表
        test_image_files = [os.path.join(TEST_IMAGE_DIR, f) for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in test_image_files:
            extract_and_save_landmarks(image_file, writer_test, hands)
        print(f"✅ 测试集关键点坐标（仅左手或镜像右手）已保存到 {OUTPUT_CSV_PATH_TEST}")

print("\n处理完成。")