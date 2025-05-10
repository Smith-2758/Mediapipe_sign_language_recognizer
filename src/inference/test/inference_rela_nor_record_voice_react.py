import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import warnings
from sklearn.exceptions import DataConversionWarning
import time
from datetime import datetime

# 导入文本转语音库
import pyttsx3

# 忽略 sklearn 的 UserWarning 和 DataConversionWarning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DataConversionWarning, module="sklearn")

# 加载训练好的模型、Scaler 和 LabelEncoder (使用 _partial_normalized 版本)
model_path = 'model/svm_model/relative_normalized/new_hand_gesture_recognition_model_partial_normalized.pkl'
scaler_path = 'model/svm_model/relative_normalized/new_hand_gesture_recognition_scaler_partial_normalized.pkl'
label_encoder_path = 'model/svm_model/relative_normalized/new_hand_gesture_recognition_label_encoder_partial_normalized.pkl'

if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(label_encoder_path):
    print("错误：模型文件、Scaler 文件或 LabelEncoder 文件未找到。请确保它们位于正确的路径 (使用了 _partial_normalized 版本)。")
    exit()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(label_encoder_path)

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 获取视频的宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20  # 可以调整保存视频的帧率
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# 生成带有时间戳的视频文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_video_filename = f'output/output_gesture_recognition_{timestamp}.avi'
output_video = cv2.VideoWriter(output_video_filename, fourcc, fps, (frame_width, frame_height))
print(f"视频将保存为: {output_video_filename}")

# 生成带有时间戳的音频文件名
output_audio_filename = f'output/spoken_gesture_sequence_{timestamp}.wav'
print(f"语音将保存为: {output_audio_filename}")

# 记录检测结果的时间和标签
last_record_time = time.time()
recorded_labels = []
record_interval = 3
record_indicator_duration = 0.5  # 绿色框显示的持续时间
indicator_active = False
indicator_start_time = 0
exit_flag = False  # 用于标记是否按下 'q' 键
hand_detected = False

# 初始化文本转语音引擎
engine = pyttsx3.init()

# 设置语速
engine.setProperty('rate', 125)

def normalize_inference_landmarks(landmarks):
    if not landmarks:
        return None
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark[1:]]) # Skip wrist for normalization
    min_xy = landmarks_array[:, :2].min(axis=0)
    max_xy = landmarks_array[:, :2].max(axis=0)
    range_xy = max_xy - min_xy + 1e-6
    normalized_xy = (landmarks_array[:, :2] - min_xy) / range_xy
    relative_z = (landmarks_array[:, 2] - landmarks.landmark[0].z).reshape(-1, 1) # Relative z to wrist
    combined = np.hstack((normalized_xy, relative_z)).flatten()
    return combined.reshape(1, -1)

while not exit_flag:
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧 (stream end?). 退出 ...")
        break

    # 将帧转换为 RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True
    frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    predicted_label = ' '  # 默认预测标签为空格
    current_hand_detected = False

    if results.multi_hand_landmarks:
        current_hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            # 提取并处理关键点坐标以匹配训练数据
            processed_landmarks = normalize_inference_landmarks(hand_landmarks)

            if processed_landmarks is not None:
                # 标准化特征
                processed_landmarks_scaled = scaler.transform(processed_landmarks)

                # 进行预测
                prediction_encoded = model.predict(processed_landmarks_scaled)[0]
                predicted_label = label_encoder.inverse_transform([prediction_encoded])[0]

                # 在图像上绘制关键点和预测结果 (右上角显示)
                cv2.putText(frame, f'Predicted: {predicted_label}', (frame_width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        predicted_label = '' # 没有检测到手势时，预测标签为空

    current_time = time.time()
    time_elapsed = current_time - last_record_time

    # 显示倒计时提示
    time_remaining = max(0, record_interval - int(time_elapsed))
    cv2.putText(frame, f'{time_remaining}s', (frame_width - 80, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # 触发记录
    if time_elapsed >= record_interval:
        if current_hand_detected:
            if predicted_label == '1':
                recorded_labels.append(' ')
                print(f"Recorded: Space")
            elif predicted_label == '2' and recorded_labels:
                recorded_labels.pop()
                print(f"Recorded: Delete previous")
            elif predicted_label and predicted_label not in ['1', '2']:
                recorded_labels.append(predicted_label)
                print(f"Recorded label: {predicted_label}")
            last_record_time = current_time
            indicator_active = True
            indicator_start_time = current_time
        else:
            print("No hand detected, skipping record.")
            last_record_time = current_time # Reset timer even if no hand detected

    # 显示记录的标签
    recorded_text = "".join(recorded_labels)
    cv2.putText(frame, f'Recorded: {recorded_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    # 显示绿色边界框提示
    if indicator_active and (current_time - indicator_start_time < record_indicator_duration):
        cv2.rectangle(frame, (0, 0), (frame_width, frame_height), (0, 255, 0), 5)
    elif indicator_active:
        indicator_active = False

    # 显示结果帧
    cv2.imshow('Real-time Hand Gesture Recognition', frame)
    output_video.write(frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit_flag = True
        break

# 释放摄像头和视频写入对象，关闭窗口
cap.release()
output_video.release()
cv2.destroyAllWindows()

# 打印识别到的字符串
print("\n识别到的手势序列:")
spoken_text = "".join(recorded_labels)
print(spoken_text)

# 保存语音结果
engine.save_to_file(spoken_text, output_audio_filename)
engine.runAndWait()
print(f"语音已保存到: {output_audio_filename}")