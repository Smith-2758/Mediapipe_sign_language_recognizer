import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def calculate_angle(p1, p2, p3):
    """计算三个点之间的角度 (以 p2 为顶点)."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # Handle cases with coincident points

    dot_product = np.dot(v1, v2)
    angle_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def extract_angle_features(df):
    """从关键点 DataFrame 中提取角度特征."""
    angle_features = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        landmarks = row.iloc[1:].values.reshape(21, 3)
        row_angles = [row.iloc[0]]  # Keep the label

        # 手指角度 (使用指根、关节和指尖)
        for i in range(5):  # 遍历五根手指 (拇指: 0, 食指: 1, 中指: 2, 无名指: 3, 小指: 4)
            if i == 0:  # 拇指的特殊连接
                mcp = landmarks[2]
                pip = landmarks[3]
                tip = landmarks[4]
            else:
                mcp = landmarks[4 + (i - 1) * 4 + 1]  # MCP
                pip = landmarks[4 + (i - 1) * 4 + 2]  # PIP
                tip = landmarks[4 + (i - 1) * 4 + 3]  # TIP

            if i > 0: # 角度需要三个点，拇指特殊处理
                dip = landmarks[4 + (i - 1) * 4 + 3 - 1] # DIP

                row_angles.append(calculate_angle(mcp, pip, tip)) # PIP 处的角度
                row_angles.append(calculate_angle(pip, dip, tip)) # DIP 处的角度


        # 掌心角度 (使用手腕和手指根部) - 可以根据需要添加更多
        wrist = landmarks[0]
        thumb_cmc = landmarks[1]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]

        row_angles.append(calculate_angle(wrist, thumb_cmc, index_mcp))
        row_angles.append(calculate_angle(wrist, index_mcp, middle_mcp))
        row_angles.append(calculate_angle(wrist, middle_mcp, ring_mcp))
        row_angles.append(calculate_angle(wrist, ring_mcp, pinky_mcp))

        angle_features.append(row_angles)

    return pd.DataFrame(angle_features)

# 定义输入和输出文件路径
train_landmarks_path = 'data/raw/hand_landmarks_train.csv'
test_landmarks_path = 'data/raw/hand_landmarks_test.csv'
train_angles_path = 'data/processed/hand_landmarks_train_angles.csv'
test_angles_path = 'data/processed/hand_landmarks_test_angles.csv'

# 读取原始关键点数据
df_train_landmarks = pd.read_csv(train_landmarks_path)
df_test_landmarks = pd.read_csv(test_landmarks_path)

# 提取角度特征
df_train_angles = extract_angle_features(df_train_landmarks)
df_test_angles = extract_angle_features(df_test_landmarks)

# 保存角度特征到新的 CSV 文件
df_train_angles.to_csv(train_angles_path, index=False, header=False)
df_test_angles.to_csv(test_angles_path, index=False, header=False)

print(f"\n✅ 训练集角度特征已保存到: {train_angles_path}")
print(f"✅ 测试集角度特征已保存到: {test_angles_path}")