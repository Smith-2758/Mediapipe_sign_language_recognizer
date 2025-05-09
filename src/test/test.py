"""
手语识别测试脚本
使用MediaPipe Tasks进行手语识别的测试代码
主要功能：
1. 加载并初始化MediaPipe手部关键点检测器
2. 处理单张图片进行手部检测
3. 可视化检测结果
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

def init_hand_detector():
    """
    初始化MediaPipe手部关键点检测器
    
    返回:
        HandLandmarker: MediaPipe手部关键点检测器实例
    """
    # 创建基础配置
    base_options = python.BaseOptions(
        model_asset_path='hand_landmarker.task'  # 模型文件路径
    )
    # 创建手部检测器配置
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2  # 最多检测两只手
    )
    # 创建检测器实例
    return vision.HandLandmarker.create_from_options(options)

def draw_landmarks_on_image(image, detection_result):
    """
    在图像上绘制手部关键点和连接线
    
    参数:
        image: 输入图像
        detection_result: 手部检测结果
    
    返回:
        标注后的图像
    """
    annotated_image = image.copy()
    for hand_landmarks in detection_result.hand_landmarks:
        for landmark in hand_landmarks:
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
    return annotated_image

# === 主程序 ===
# 初始化检测器
detector = init_hand_detector()

# 加载测试图像
test_image_path = "data/train/B8_jpg.rf.ad0c398b1d9d6a140366dec45163688a.jpg"
image = mp.Image.create_from_file(test_image_path)

# 进行手部关键点检测
detection_result = detector.detect(image)

# 可视化检测结果
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

# 显示结果
cv2.imshow("Hand Landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()