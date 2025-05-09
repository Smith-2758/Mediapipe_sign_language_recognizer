import cv2
import mediapipe as mp
import os
import csv
import numpy as np
from tqdm import tqdm
# landmark_pb2 用于构建绘制所需的数据结构
from mediapipe.framework.formats import landmark_pb2

# === 可视化函数 (保持你的函数结构，稍作健壮性调整) ===
MARGIN = 10  # 文本边距
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # BGR 绿色

def draw_landmarks_on_image(rgb_image, detection_result):
    """在图像上绘制手部关键点和左右手标签"""
    hand_landmarks_list = detection_result.multi_hand_landmarks
    handedness_list = detection_result.multi_handedness
    # 创建图像副本进行绘制，避免修改原始图像
    annotated_image = np.copy(rgb_image)

    # 如果没有检测到手，直接返回原图副本
    if not hand_landmarks_list:
        return annotated_image

    # 遍历检测到的每一只手
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = None
        # 检查是否有对应的左右手信息
        if handedness_list and idx < len(handedness_list):
             handedness = handedness_list[idx]

        # --- 绘制关键点和连接线 ---
        # 需要将 landmark 列表转换为 MediaPipe 定义的 NormalizedLandmarkList 格式
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks.landmark
        ])

        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,                   # 在哪个图像上绘制
            landmark_list=hand_landmarks_proto,      # 关键点数据
            connections=mp.solutions.hands.HAND_CONNECTIONS, # 如何连接关键点
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(), # 关键点样式
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style() # 连接线样式
        )

        # --- 绘制左右手标签 (如果信息可用) ---
        if handedness:
            h, w, _ = annotated_image.shape
            # 获取手部边界用于定位文本
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            # 确保坐标列表不为空再计算 min
            if x_coords and y_coords:
                text_x = int(min(x_coords) * w)
                text_y = int(min(y_coords) * h) - MARGIN # 将文本放在手部上方

                cv2.putText(annotated_image, f"{handedness.classification[0].label}", # 获取 'Left' 或 'Right'
                            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    # 返回带有标注的图像 (RGB格式)
    return annotated_image

# === 主程序 ===
# 设置 TensorFlow 日志级别 (可选，减少无关信息)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 配置路径 ---
IMAGE_DIR = 'data/test'  # 包含训练图片的目录
OUTPUT_CSV = 'data/raw/hand_data.csv'  # 输出 CSV 文件路径

# --- 打印配置信息 ---
print("--- 配置信息 ---")
print(f"当前工作目录: {os.getcwd()}")
print(f"图片目录 (绝对路径): {os.path.abspath(IMAGE_DIR)}")
print(f"输出CSV文件 (绝对路径): {os.path.abspath(OUTPUT_CSV)}")

# --- 检查输入目录 ---
if not os.path.isdir(IMAGE_DIR):
    print(f"\n错误: 图片目录不存在: {os.path.abspath(IMAGE_DIR)}")
    print("请确保路径正确或目录已创建。")
    exit() # 退出程序

# --- 准备输出目录 ---
output_dir = os.path.dirname(OUTPUT_CSV)
# 如果 OUTPUT_CSV 包含目录部分，则创建它
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录已创建: {os.path.abspath(output_dir)}")
elif output_dir:
     print(f"输出目录已存在: {os.path.abspath(output_dir)}")

# --- 获取图片文件列表 ---
try:
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"\n警告: 在目录 '{os.path.abspath(IMAGE_DIR)}' 中未找到任何图片文件 (.png, .jpg, .jpeg)。")
        exit()
    print(f"✅ 在图片目录中找到 {len(image_files)} 张图片。")
except FileNotFoundError:
    print(f"\n错误: 访问图片目录时出错: {os.path.abspath(IMAGE_DIR)}")
    exit()
except Exception as e:
     print(f"\n错误: 读取图片目录时发生未知错误: {e}")
     exit()

# --- 初始化 MediaPipe Hands ---
mp_hands = mp.solutions.hands
# 使用 'with' 语句确保资源正确管理
hands_detector = mp_hands.Hands(
    static_image_mode=True,       # 适用于静态图片
    max_num_hands=1,              # 最多检测一只手 (根据你的设置)
    min_detection_confidence=0.3, # 最低检测置信度 (根据你的设置)
    model_complexity=1            # 模型复杂度 (根据你的设置, 0 或 1)
)

# --- 处理图片并写入 CSV ---
print("\n--- 开始处理图像并提取关键点 ---")
success_count = 0
failed_count = 0

try:
    # 打开 CSV 文件准备写入
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        # 写入 CSV 文件的表头
        header = []
        for i in range(21):  # MediaPipe 提供 21 个手部关键点
            header.extend([f'x{i}', f'y{i}', f'z{i}']) # x, y, z 坐标
        header.append('label') # 添加标签列
        writer.writerow(header)

        # 使用 MediaPipe Hands 对象处理图片
        with hands_detector:
            # 使用 tqdm 显示进度条
            for filename in tqdm(image_files, desc="处理进度", unit="张"):
                filepath = os.path.join(IMAGE_DIR, filename)

                # 1. 读取图片
                image = cv2.imread(filepath)
                if image is None:
                    tqdm.write(f"⚠️ 警告: 无法读取图像文件: {filename}") # 使用 tqdm.write 避免破坏进度条
                    failed_count += 1
                    continue # 跳过这个文件

                # 2. 转换颜色空间 (OpenCV: BGR -> MediaPipe: RGB)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 3. 使用 MediaPipe 处理图像
                try:
                    results = hands_detector.process(image_rgb)
                except Exception as e:
                    tqdm.write(f"❌ 错误: 处理图像 {filename} 时发生异常: {e}")
                    failed_count += 1
                    continue # 跳过这个文件

                # 4. 检查是否检测到手部关键点
                if results.multi_hand_landmarks:
                    # 因为设置了 max_num_hands=1，所以我们只取第一个检测到的手
                    hand_landmarks = results.multi_hand_landmarks[0]

                    # 5. 提取坐标并准备写入 CSV 的行数据
                    row_data = []
                    for lm in hand_landmarks.landmark:
                        # 提取归一化的 x, y, z 坐标
                        row_data.extend([lm.x, lm.y, lm.z])

                    # 6. 从文件名获取标签 (假设标签是文件名的第一个字符)
                    label = filename[0].upper()
                    row_data.append(label)

                    # 7. 写入 CSV 文件
                    writer.writerow(row_data)
                    success_count += 1

                    # --- 可选: 可视化并保存/显示标注图像 ---
                    # annotated_image_rgb = draw_landmarks_on_image(image_rgb, results)
                    # # 转回 BGR 以便 OpenCV 显示或保存
                    # annotated_image_bgr = cv2.cvtColor(annotated_image_rgb, cv2.COLOR_RGB2BGR)
                    #
                    # # 创建用于保存标注图像的目录 (如果需要)
                    # annotated_output_dir = os.path.join(os.path.dirname(OUTPUT_CSV), 'annotated_images')
                    # os.makedirs(annotated_output_dir, exist_ok=True)
                    # annotated_filepath = os.path.join(annotated_output_dir, f"annotated_{filename}")
                    # cv2.imwrite(annotated_filepath, annotated_image_bgr)
                    #
                    # # 或者实时显示 (按任意键继续)
                    # # cv2.imshow("Annotated Image", annotated_image_bgr)
                    # # cv2.waitKey(1) # 显示非常短的时间，或者用 0 等待按键
                    # --- 可视化结束 ---

                else:
                    # 如果没有检测到手部，增加失败计数
                    tqdm.write(f"ℹ️ 信息: 在图像 {filename} 中未检测到手部。") # 可选提示
                    failed_count += 1

    # 可选：如果在循环中使用了 cv2.imshow, 在这里关闭所有窗口
    # cv2.destroyAllWindows()

except IOError as e:
    print(f"\n❌ 错误: 无法打开或写入 CSV 文件 '{OUTPUT_CSV}': {e}")
    print("请检查文件路径和写入权限。")
except Exception as e:
    print(f"\n❌ 错误: 处理过程中发生未预料的错误: {e}")

# --- 输出最终结果 ---
print("\n--- 处理完成 ---")
print(f"✔️ 成功提取并保存关键点: {success_count} 张图像")
print(f"❌ 未检测到手部或处理失败: {failed_count} 张图像")
if success_count > 0:
    print(f"📄 数据已保存到: {os.path.abspath(OUTPUT_CSV)}")
elif failed_count == len(image_files) and len(image_files) > 0:
     print("⚠️ 所有图像均未能检测到手部。请检查：")
     print("   - 图片质量和内容 (手部是否清晰可见？)")
     print(f"   - MediaPipe 参数 (min_detection_confidence={hands_detector.min_detection_confidence}) 是否合适？")
elif not image_files:
     print("目录中未找到符合条件的图片文件。")