import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# 1. 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 统计变量
success_count = 0
fail_count = 0
label_distribution = {}

# 定义输出 CSV 文件名
OUTPUT_CSV_PATH = 'data/raw/hand_landmarks_test.csv'

# 定义图片所在的目录
IMAGE_DIR = 'data/test'  # 请替换为你的图片目录

# 确保输出目录存在
output_dir = os.path.dirname(OUTPUT_CSV_PATH)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# 获取图片文件列表
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 写入 CSV 文件的表头
header = ['label']
for i in range(21):
    header.extend([f'x{i}', f'y{i}', f'z{i}'])

with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(header)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:

        for image_file in image_files:
            image_path = os.path.join(IMAGE_DIR, image_file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"错误：无法加载图像 {image_path}")
                continue

            # 获取标签 (文件名的第一个字符)
            label = image_file[0].upper()

            # 将图像从 BGR 转换为 RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                row = [label]
                for landmark in hand_landmarks.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z])
                csv_writer.writerow(row)
                
                # 更新统计信息
                success_count += 1
                label_distribution[label] = label_distribution.get(label, 0) + 1

                # # 可选：可视化部分 (可以注释掉)
                # annotated_image = image.copy()
                # mp_drawing.draw_landmarks(
                #     annotated_image,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS)
                # cv2.imwrite(f'annotated_{image_file}', annotated_image)
                # cv2.imshow('Annotated Hand', annotated_image)
            else:
                print(f"在图像 {image_file} 中未检测到手部。")
                fail_count += 1

print(f"\n关键点坐标已保存到 {OUTPUT_CSV_PATH}")
print(f"\n统计信息：")
print(f"成功处理图片数：{success_count}")
print(f"处理失败图片数：{fail_count}")
print(f"\n各类别分布情况：")
for label, count in sorted(label_distribution.items()):
    print(f"类别 {label}: {count}张")
print("\n处理完成。")