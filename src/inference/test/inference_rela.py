import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# 加载训练好的模型、Scaler 和 LabelEncoder
model_path = 'model/svm_model/hand_gesture_recognition_model.pkl'
scaler_path = 'model/svm_model/hand_gesture_recognition_scaler.pkl'
label_encoder_path = 'model/svm_model/hand_gesture_recognition_label_encoder.pkl'

if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(label_encoder_path):
    print("错误：模型文件、Scaler 文件或 LabelEncoder 文件未找到。请确保它们位于正确的路径。")
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

while True:
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

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 提取关键点坐标
            landmarks = []
            wrist = hand_landmarks.landmark[0]  # 获取手腕坐标作为参考点
            
            # 跳过手腕坐标，只收集其他关键点相对于手腕的坐标
            for lm in hand_landmarks.landmark[1:]:  # 从索引1开始，跳过手腕
                # 计算相对坐标
                relative_x = lm.x - wrist.x
                relative_y = lm.y - wrist.y
                relative_z = lm.z - wrist.z
                landmarks.extend([relative_x, relative_y, relative_z])
                
            processed_landmarks = np.array(landmarks).reshape(1, -1)  
            
            # 标准化特征
            processed_landmarks_scaled = scaler.transform(processed_landmarks)

            # 进行预测
            prediction_encoded = model.predict(processed_landmarks_scaled)[0]
            predicted_label = label_encoder.inverse_transform([prediction_encoded])[0]

            # 在图像上绘制关键点和预测结果
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 显示结果帧
    cv2.imshow('Real-time Hand Gesture Recognition', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()