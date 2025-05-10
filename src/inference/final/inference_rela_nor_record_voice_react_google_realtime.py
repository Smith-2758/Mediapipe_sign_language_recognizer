"""
实时手语识别系统
该系统通过摄像头实时捕获手语动作，并将其转换为文字和语音输出
主要功能：
1. 实时手势检测和识别
2. 文字记录和显示
3. 实时语音反馈
4. 视频录制
5. 支持空格和删除等特殊手势
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import warnings
from sklearn.exceptions import DataConversionWarning
import time
from datetime import datetime
from gtts import gTTS
import pygame  # 用于播放音频
import shutil  # 用于删除临时文件

# 忽略 sklearn 的警告信息
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DataConversionWarning, module="sklearn")

# 模型相关文件路径配置
model_path = 'model/svm_model/relative_normalized/new_hand_gesture_recognition_model_partial_normalized.pkl'
scaler_path = 'model/svm_model/relative_normalized/new_hand_gesture_recognition_scaler_partial_normalized.pkl'
label_encoder_path = 'model/svm_model/relative_normalized/new_hand_gesture_recognition_label_encoder_partial_normalized.pkl'

# 检查模型文件是否存在
if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(label_encoder_path):
    print("错误：模型文件、Scaler 文件或 LabelEncoder 文件未找到。请确保它们位于正确的路径。")
    exit()

# 加载训练好的模型和预处理器
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(label_encoder_path)

# 初始化 MediaPipe 手势检测模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,     # 视频流模式
    max_num_hands=1,            # 最多检测一只手
    min_detection_confidence=0.5,  # 最小检测置信度
    min_tracking_confidence=0.5    # 最小跟踪置信度
)
mp_drawing = mp.solutions.drawing_utils

# 初始化摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 配置视频录制参数
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20  # 视频帧率
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# 生成带时间戳的输出文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_video_filename = f'output/output_gesture_recognition_{timestamp}.avi'
output_video = cv2.VideoWriter(output_video_filename, fourcc, fps, (frame_width, frame_height))
print(f"视频将保存为: {output_video_filename}")

# 配置音频输出
output_audio_filename_base = f'output/spoken_gesture_sequence_{timestamp}'

# 初始化状态变量
last_record_time = time.time()  # 上次记录时间
recorded_labels = []            # 已识别的手势序列
record_interval = 4            # 手势识别间隔（秒）
record_indicator_duration = 0.5  # 绿色提示框显示时间
indicator_active = False        # 提示框状态
indicator_start_time = 0       # 提示框开始时间
exit_flag = False              # 退出标志
hand_detected = False          # 手势检测状态

# 初始化音频播放器
pygame.mixer.init()

# 创建临时语音文件目录
temp_voice_dir = 'temp/voice'
if not os.path.exists(temp_voice_dir):
    os.makedirs(temp_voice_dir)

# 语音播报控制变量
is_speaking = False           # 当前是否正在播报
speaking_start_time = 0       # 播报开始时间
speaking_duration = 1         # 预估的播报持续时间

def normalize_inference_landmarks(landmarks):
    """
    标准化手部关键点坐标
    
    参数:
        landmarks: MediaPipe检测到的手部关键点
    
    返回:
        归一化后的关键点坐标数组，用于模型预测
    """
    if not landmarks:
        return None
    # 提取关键点坐标（跳过手腕点）
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark[1:]])
    # 计算xy坐标的归一化参数
    min_xy = landmarks_array[:, :2].min(axis=0)
    max_xy = landmarks_array[:, :2].max(axis=0)
    range_xy = max_xy - min_xy + 1e-6
    # 归一化xy坐标
    normalized_xy = (landmarks_array[:, :2] - min_xy) / range_xy
    # 计算相对于手腕的z坐标
    relative_z = (landmarks_array[:, 2] - landmarks.landmark[0].z).reshape(-1, 1)
    # 组合归一化后的坐标
    combined = np.hstack((normalized_xy, relative_z)).flatten()
    return combined.reshape(1, -1)

def speak_text(text, filename_base, index=None):
    """
    将文字转换为语音并播放
    
    参数:
        text: 要转换的文字
        filename_base: 音频文件基础名称
        index: 临时文件索引（用于实时播报）
    """
    global is_speaking, speaking_start_time
    if text:
        try:
            is_speaking = True
            speaking_start_time = time.time()
            tts = gTTS(text=text, lang='en')
            # 确定保存路径（临时文件或最终文件）
            if index is not None:
                filename = os.path.join(temp_voice_dir, f"temp_voice_{index}.mp3")
            else:
                filename = f"{filename_base}.mp3"
            # 保存并播放音频
            tts.save(filename)
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            # 等待播放完成
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            print(f"语音已播放: {text}")
            is_speaking = False
        except Exception as e:
            print(f"gTTS 错误: {e}")
            is_speaking = False

# === 主循环 ===
while not exit_flag:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧，退出...")
        break

    # 处理帧
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True
    frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    predicted_label = ' '  # 默认预测标签
    current_hand_detected = False

    current_time = time.time()
    
    # 计算实际倒计时时间（排除语音播报时间）
    if is_speaking:
        last_record_time = current_time
    
    time_elapsed = current_time - last_record_time

    # 显示绿色边框提示
    if time_elapsed >= record_interval - 0.5 and not is_speaking:
        cv2.rectangle(frame, (0, 0), (frame_width, frame_height), (0, 255, 0), 5)

    # 手势检测与识别
    if results.multi_hand_landmarks:
        current_hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            # 处理手部关键点
            processed_landmarks = normalize_inference_landmarks(hand_landmarks)

            if processed_landmarks is not None:
                # 特征标准化
                processed_landmarks_scaled = scaler.transform(processed_landmarks)
                # 预测手势
                prediction_encoded = model.predict(processed_landmarks_scaled)[0]
                predicted_label = label_encoder.inverse_transform([prediction_encoded])[0]
                # 显示预测结果
                cv2.putText(frame, f'Predicted: {predicted_label}', 
                          (frame_width - 250, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # 绘制手部关键点
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        predicted_label = ''  # 未检测到手势

    # 显示倒计时或播报状态
    if not is_speaking:
        time_remaining = max(0, record_interval - int(time_elapsed))
        cv2.putText(frame, f'{time_remaining}s', 
                   (frame_width - 80, frame_height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Speaking...', 
                   (frame_width - 150, frame_height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # 记录手势（非播报状态）
    if time_elapsed >= record_interval and not is_speaking:
        if current_hand_detected:
            current_voice_index = len(recorded_labels)
            # 处理特殊手势
            if predicted_label == '1':  # 空格
                recorded_labels.append(' ')
                print(f"Recorded: Space")
                speak_text("space", None, current_voice_index)
            elif predicted_label == '2' and recorded_labels:  # 删除
                recorded_labels.pop()
                print(f"Recorded: Delete previous")
                speak_text("delete", None, current_voice_index)
            elif predicted_label and predicted_label not in ['1', '2']:  # 普通字母
                recorded_labels.append(predicted_label)
                print(f"Recorded label: {predicted_label}")
                speak_text(predicted_label, None, current_voice_index)
            last_record_time = current_time
        else:
            print("No hand detected, skipping record.")
            last_record_time = current_time

    # 显示已记录的手势序列
    recorded_text = "".join(recorded_labels)
    cv2.putText(frame, f'Recorded: {recorded_text}', 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    # 显示当前帧
    cv2.imshow('Real-time Hand Gesture Recognition', frame)
    output_video.write(frame)

    # 检查退出键
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit_flag = True
        break

# === 清理资源 ===
cap.release()
output_video.release()
cv2.destroyAllWindows()

# 输出最终结果
print("\n识别到的手势序列:")
spoken_text = "".join(recorded_labels)
print(spoken_text)

# 播放完整的识别结果
speak_text(spoken_text, output_audio_filename_base)

# 清理临时文件
try:
    shutil.rmtree(temp_voice_dir)
    os.makedirs(temp_voice_dir)
    print("临时语音文件已清理")
except Exception as e:
    print(f"清理临时语音文件时出错: {e}")