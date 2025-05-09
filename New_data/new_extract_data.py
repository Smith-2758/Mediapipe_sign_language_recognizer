"""
手语数据集预处理脚本
该脚本用于从新的ASL手语数据集中提取手部关键点并进行预处理
主要功能：
1. 自动读取和分割训练集/测试集
2. 提取手部关键点坐标
3. 统一处理为左手数据（右手镜像处理）
4. 保存为CSV格式以供训练使用
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import random
from collections import Counter

# 初始化MediaPipe Hands模型
mp_hands = mp.solutions.hands
print("✅ MediaPipe Hands 初始化完成")

def process_and_extract(image, hands):
    """
    处理单张图像并提取手部关键点
    
    参数:
        image: OpenCV格式的图像
        hands: MediaPipe Hands模型实例
    
    返回:
        tuple: (landmarks, handedness)
            - landmarks: 手部关键点数据
            - handedness: 手的类型（左手/右手）
    """
    # 转换为RGB格式（MediaPipe要求）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 处理图像
    results = hands.process(image_rgb)
    landmarks = None
    handedness = None
    # 如果检测到手部和手的类型
    if results.multi_hand_landmarks and results.multi_handedness:
        landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].label
    return landmarks, handedness

def extract_and_save_landmarks(image_path, label, output_csv, hands):
    """
    提取单张图像的手部关键点并保存到CSV
    
    参数:
        image_path: 图像文件路径
        label: 手语字母标签
        output_csv: CSV写入器对象
        hands: MediaPipe Hands模型实例
    
    返回:
        bool: 处理成功返回True，失败返回False
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 错误：无法加载图像 {image_path}")
        return False

    # 提取关键点
    landmarks, handedness = process_and_extract(image, hands)

    # 处理右手数据：镜像转换为左手
    if handedness == 'Right':
        flipped_image = cv2.flip(image, 1)
        flipped_landmarks, _ = process_and_extract(flipped_image, hands)
        if flipped_landmarks:
            landmarks = flipped_landmarks
            handedness = 'Left'  # 标记为处理后的"左手"
        else:
            print(f"⚠️ 警告：右手图像翻转后未检测到手 {image_path}")
            return False
    elif handedness == 'Left':
        pass  # 左手数据保持原样
    else:
        print(f"⚠️ 警告：未检测到手或无法识别左右手 {image_path}")
        return False

    # 保存关键点数据
    if landmarks:
        # 构建数据行：标签 + 手的类型 + 21个关键点的x,y,z坐标
        row = [label, handedness]
        for landmark in landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z])
        output_csv.writerow(row)
        return True
    return False

# 定义输出文件路径
OUTPUT_CSV_PATH_TRAIN = 'data/processed/new_hand_landmarks_train_left_only.csv'
OUTPUT_CSV_PATH_TEST = 'data/processed/new_hand_landmarks_test_left_only.csv'
print(f"💾 训练集输出路径: {OUTPUT_CSV_PATH_TRAIN}")
print(f"💾 测试集输出路径: {OUTPUT_CSV_PATH_TEST}")

# 定义数据目录
NEW_DATA_ROOT = 'New_data/asl_alphabet_train/asl_alphabet_train'
print(f"📂 图片根目录: {NEW_DATA_ROOT}")
all_categories = sorted([d for d in os.listdir(NEW_DATA_ROOT) if os.path.isdir(os.path.join(NEW_DATA_ROOT, d)) and d != 'nothing'])
print(f"🔍 找到以下类别文件夹: {all_categories}")

train_data = []
test_data = []

with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands_init:
    print("⚙️ MediaPipe Hands 对象 (用于收集文件) 已创建")
    for category in all_categories:
        print(f"\nProcessing category: {category}")
        category_path = os.path.join(NEW_DATA_ROOT, category)
        image_files = [os.path.join(category_path, f) for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(image_files)
        print(f"   Found {len(image_files)} images.")

        train_files = image_files[:400]
        test_files = image_files[400:560] # 400 + 160
        print(f"   Selected {len(train_files)} for training and {len(test_files)} for testing.")

        label = category.upper()
        if category == 'space':
            label = '1'
        elif category == 'del':
            label = '2'
        print(f"   Using label: {label}")

        print("   Collecting training image paths...")
        for file in train_files:
            train_data.append({'image_path': file, 'label': label})

        print("   Collecting testing image paths...")
        for file in test_files:
            test_data.append({'image_path': file, 'label': label})

# 保存训练数据到 CSV
train_labels = []
print(f"\n✍️ Writing training data to {OUTPUT_CSV_PATH_TRAIN}...")
with open(OUTPUT_CSV_PATH_TRAIN, 'w', newline='', encoding='utf-8') as csvfile_train:
    writer_train = csv.writer(csvfile_train)
    header = ['label', 'handedness']
    for i in range(21):
        header.extend([f'x{i}', f'y{i}', f'z{i}'])
    writer_train.writerow(header)

    for item in train_data:
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands_process:
            image = cv2.imread(item['image_path'])
            landmarks, handedness_detected = process_and_extract(image, hands_process)
            if handedness_detected == 'Right':
                flipped_image = cv2.flip(image, 1)
                flipped_landmarks, _ = process_and_extract(flipped_image, hands_process)
                if flipped_landmarks:
                    landmarks = flipped_landmarks
                    handedness = 'Left'
                    train_labels.append(item['label'])
                    row_data = [item['label'], handedness]
                    for lm in landmarks.landmark:
                        row_data.extend([lm.x, lm.y, lm.z])
                    writer_train.writerow(row_data)
                # else: print(f"Skipped training {item['image_path']} due to flip fail")
            elif handedness_detected == 'Left':
                handedness = 'Left'
                train_labels.append(item['label'])
                row_data = [item['label'], handedness]
                for lm in landmarks.landmark:
                    row_data.extend([lm.x, lm.y, lm.z])
                writer_train.writerow(row_data)
            # else: print(f"Skipped training {item['image_path']} due to no hand")

print(f"✅ 新的训练集关键点坐标（仅左手或镜像右手）已保存到 {OUTPUT_CSV_PATH_TRAIN}")

# 保存测试数据到 CSV
test_labels = []
print(f"\n✍️ Writing testing data to {OUTPUT_CSV_PATH_TEST}...")
with open(OUTPUT_CSV_PATH_TEST, 'w', newline='', encoding='utf-8') as csvfile_test:
    writer_test = csv.writer(csvfile_test)
    header = ['label', 'handedness']
    for i in range(21):
        header.extend([f'x{i}', f'y{i}', f'z{i}'])
    writer_test.writerow(header)

    for item in test_data:
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands_process:
            image = cv2.imread(item['image_path'])
            landmarks, handedness_detected = process_and_extract(image, hands_process)
            if handedness_detected == 'Right':
                flipped_image = cv2.flip(image, 1)
                flipped_landmarks, _ = process_and_extract(flipped_image, hands_process)
                if flipped_landmarks:
                    landmarks = flipped_landmarks
                    handedness = 'Left'
                    test_labels.append(item['label'])
                    row_data = [item['label'], handedness]
                    for lm in landmarks.landmark:
                        row_data.extend([lm.x, lm.y, lm.z])
                    writer_test.writerow(row_data)
                # else: print(f"Skipped testing {item['image_path']} due to flip fail")
            elif handedness_detected == 'Left':
                handedness = 'Left'
                test_labels.append(item['label'])
                row_data = [item['label'], handedness]
                for lm in landmarks.landmark:
                    row_data.extend([lm.x, lm.y, lm.z])
                writer_test.writerow(row_data)
            # else: print(f"Skipped testing {item['image_path']} due to no hand")

print(f"✅ 新的测试集关键点坐标（仅左手或镜像右手）已保存到 {OUTPUT_CSV_PATH_TEST}")

# 统计训练集类别分布
train_label_counts = Counter(train_labels)
print("\n📊 训练集类别分布:")
for label, count in sorted(train_label_counts.items()):
    print(f"{label}: {count}")

# 统计测试集类别分布
test_label_counts = Counter(test_labels)
print("\n📊 测试集类别分布:")
for label, count in sorted(test_label_counts.items()):
    print(f"{label}: {count}")

print("\n✅ 处理完成。")