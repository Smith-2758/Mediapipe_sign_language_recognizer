import cv2
import mediapipe as mp
import numpy as np

# 1. 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# 使用 'with' 语句确保资源正确释放
# static_image_mode=True: 适用于静态图片处理
# max_num_hands=2: 最多检测两只手
# min_detection_confidence=0.5: 最低检测置信度阈值
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:

    # 2. 加载图像
    # !!! 请将 'your_hand_image.jpg' 替换为你的手部图像文件路径 !!!
    image_path = 'data/train/A0_jpg.rf.292a080422ba984985192f413101af41.jpg'
    image = cv2.imread(image_path)

    if image is None:
        print(f"错误：无法加载图像 {image_path}")
    else:
        # 获取图像尺寸
        image_height, image_width, _ = image.shape

        # 3. 处理图像
        # MediaPipe 需要 RGB 图像，而 OpenCV 默认加载为 BGR
        # 将图像从 BGR 转换为 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 处理图像并获取结果
        results = hands.process(image_rgb)

        # 用于存储文本坐标的列表
        landmark_coordinates_text = []

        # 4. 提取、转换和格式化坐标
        if results.multi_hand_landmarks:
            print(f"检测到 {len(results.multi_hand_landmarks)} 只手。")
            # 遍历检测到的每一只手
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                print(f"\n--- 手 {hand_index + 1} 的关键点坐标 ---")
                landmark_coordinates_text.append(f"--- 手 {hand_index + 1} ---")

                # 可选：在图像上绘制手部关键点和连接线以供可视化
                annotated_image = image.copy() # 创建副本以绘制
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)
                # 保存或显示带标注的图像 (可选)
                cv2.imwrite(f'annotated_hand_{hand_index + 1}.jpg', annotated_image)
                # cv2.imshow(f'Annotated Hand {hand_index + 1}', annotated_image)

                # 遍历该手的所有关键点 (共 21 个)
                for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                    # landmark.x 和 landmark.y 是归一化的坐标 (0.0 到 1.0)
                    # 将归一化坐标转换为像素坐标
                    pixel_x = int(landmark.x * image_width)
                    pixel_y = int(landmark.y * image_height)

                    # 将坐标格式化为文本
                    coord_text = f"Landmark {landmark_id}: ({pixel_x}, {pixel_y})"
                    print(coord_text)
                    landmark_coordinates_text.append(coord_text)

            # 5. 打印所有提取到的文本坐标 (汇总)
            print("\n--- 所有提取的文本坐标汇总 ---")
            for line in landmark_coordinates_text:
                print(line)

        else:
            print("在图像中未检测到手部。")

# 可选：如果显示了图像窗口，等待按键后关闭
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print("\n处理完成。")