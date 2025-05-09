"""
手语关键点坐标预处理脚本
该脚本用于处理原始的手部关键点数据，进行以下转换：
1. 将右手数据镜像处理为左手数据
2. 对x,y坐标进行归一化（基于2D边界框）
3. 计算相对于手腕的z坐标
4. 分别处理训练集和测试集数据
"""

import csv
import os
import numpy as np

def calculate_relative_and_partial_normalized_coords(row):
    """
    计算关键点的归一化坐标和相对坐标
    
    参数:
        row: CSV文件中的一行数据，包含标签、手的类型和关键点坐标
        
    返回:
        处理后的坐标数据，包含：
        - 归一化的x,y坐标（不含手腕点）
        - 相对于手腕的z坐标
    """
    # 提取基本信息
    label = row[0]
    handedness = row[1]
    # 将字符串坐标转换为浮点数
    coords = list(map(float, row[2:]))
    # 重塑为(21,3)的数组，每个关键点包含x,y,z坐标
    landmarks = np.array([(coords[i], coords[i+1], coords[i+2]) 
                         for i in range(0, len(coords), 3)])

    # 使用手腕点作为参考点
    wrist = landmarks[0]
    # 计算相对于手腕的坐标
    relative_coords = landmarks - wrist

    # 对除手腕外的其他点进行x,y坐标归一化
    relative_xy_no_wrist = relative_coords[1:, :2]
    # 计算x,y的最小值和最大值
    min_xy = relative_xy_no_wrist.min(axis=0)
    max_xy = relative_xy_no_wrist.max(axis=0)
    # 添加小量避免除零错误
    range_xy = max_xy - min_xy + 1e-6
    # 归一化到[0,1]区间
    normalized_xy_no_wrist = (relative_xy_no_wrist - min_xy) / range_xy

    # 提取相对z坐标（不包含手腕）
    relative_z_no_wrist = relative_coords[1:, 2]

    # 组合归一化的x,y坐标和相对z坐标
    combined = np.hstack((normalized_xy_no_wrist, 
                         relative_z_no_wrist.reshape(-1, 1))).flatten().tolist()
    return [label, handedness] + combined

def convert_file_with_partial_normalization(input_path, output_path):
    """
    转换整个CSV文件的坐标数据
    
    参数:
        input_path: 输入CSV文件路径
        output_path: 输出CSV文件路径
    """
    with open(input_path, 'r', newline='') as infile, \
         open(output_path, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # 读取并跳过输入文件的表头
        input_header = next(reader)

        # 创建新的表头
        output_header = ['label', 'handedness']
        # 为每个非手腕关键点添加nx,ny,rz列（normalized x,y和relative z）
        for i in range(1, 21):  # 跳过手腕点(0)
            output_header.extend([f'nx{i}', f'ny{i}', f'rz{i}'])
        writer.writerow(output_header)

        # 处理每一行数据
        for row in reader:
            if not row:  # 跳过空行
                continue
            # 计算并写入转换后的坐标
            writer.writerow(calculate_relative_and_partial_normalized_coords(row))

# === 主程序 ===
# 处理训练集
input_train_path = 'data/processed/new_hand_landmarks_train_left_only.csv'
output_train_path = 'data/processed/new_hand_landmarks_train_partial_normalized_no_wrist.csv'
convert_file_with_partial_normalization(input_train_path, output_train_path)
print(f"✅ {input_train_path} 的坐标已处理完成：")
print(f"  - x,y坐标已归一化到[0,1]区间（基于2D边界框，不含手腕点）")
print(f"  - z坐标已转换为相对于手腕的值（不含手腕点）")
print(f"  - 结果已保存到 {output_train_path}")

# 处理测试集
input_test_path = 'data/processed/new_hand_landmarks_test_left_only.csv'
output_test_path = 'data/processed/new_hand_landmarks_test_partial_normalized_no_wrist.csv'
convert_file_with_partial_normalization(input_test_path, output_test_path)
print(f"\n✅ {input_test_path} 的坐标已处理完成：")
print(f"  - x,y坐标已归一化到[0,1]区间（基于2D边界框，不含手腕点）")
print(f"  - z坐标已转换为相对于手腕的值（不含手腕点）")
print(f"  - 结果已保存到 {output_test_path}")