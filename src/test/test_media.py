"""
MediaPipe手部检测测试脚本
用于测试MediaPipe在静态图像上的手部关键点检测功能
主要功能：
1. 读取并处理单张测试图片
2. 使用MediaPipe进行手部关键点检测
3. 可视化检测结果
4. 提供调试信息
"""

import cv2
import mediapipe as mp
import os

# 测试图像路径配置
IMAGE_PATH = 'data/train/B8_jpg.rf.ad0c398b1d9d6a140366dec45163688a.jpg'

# 检查图像文件是否存在
if not os.path.exists(IMAGE_PATH):
    print(f"❌ 图像文件不存在: {IMAGE_PATH}")
    exit()

# 使用OpenCV读取图像
img = cv2.imread(IMAGE_PATH)
if img is None:
    print("❌ OpenCV 无法读取图像")
    exit()

# 转换颜色空间：OpenCV默认使用BGR，而MediaPipe需要RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 初始化 MediaPipe Hands 模型
hands = mp.solutions.hands.Hands(
    static_image_mode=True,      # 静态图像模式
    max_num_hands=1,             # 最多检测一只手
    min_detection_confidence=0.5, # 检测置信度阈值
    min_tracking_confidence=0.5   # 跟踪置信度阈值
)
# 初始化绘图工具
mp_draw = mp.solutions.drawing_utils

# 处理图像：使用RGB格式进行手部检测
result = hands.process(img_rgb)

# 可视化检测结果
if result.multi_hand_landmarks:
    print("✅ 成功检测到手部关键点")
    # 在原图上绘制检测到的手部关键点和连接线
    mp_draw.draw_landmarks(
        img,  # 绘制目标图像
        result.hand_landmarks[0],  # 第一只手的关键点
        mp.solutions.hands.HAND_CONNECTIONS  # 关键点之间的连接关系
    )
else:
    print("⚠️ 未检测到手部关键点")
    print(f"图像尺寸: {img.shape}")  # 输出图像尺寸以帮助调试

# 显示结果
cv2.imshow('Hand Detection', img)
cv2.waitKey(0)  # 等待按键
cv2.destroyAllWindows()  # 关闭所有窗口
