import cv2
import mediapipe as mp
import os

IMAGE_FILES = ['New_data/asl_alphabet_test/asl_alphabet_test/F_test.jpg']
os.makedirs('output', exist_ok=True)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
        if not os.path.exists(file):
            print(f"❌ 图像文件不存在: {file}")
            continue

        image = cv2.flip(cv2.imread(file), 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            print("⚠️ 未检测到手部")
            continue

        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        cv2.imwrite(f'output/annotated_image{idx}.png', cv2.flip(annotated_image, 1))
        print(f"✅ 图像已保存为 output/annotated_image{idx}.png")
        # Draw hand world landmarks.