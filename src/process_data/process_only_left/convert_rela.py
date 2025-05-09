import csv
import os

def calculate_relative_coords(row):
    label = row[0]
    handedness = row[1]
    coords = list(map(float, row[2:]))  # [x0,y0,z0,x1,y1,z1,...]
    landmarks = [(coords[i], coords[i+1], coords[i+2]) for i in range(0, len(coords), 3)]

    # 手腕坐标作为参考点
    wrist = landmarks[0]
    relative_coords = []

    # 计算其他关键点相对于手腕的坐标差
    for point in landmarks[1:]:
        relative_x = point[0] - wrist[0]
        relative_y = point[1] - wrist[1]
        relative_z = point[2] - wrist[2]
        relative_coords.extend([relative_x, relative_y, relative_z])

    return [label, handedness] + relative_coords  # 标签和左右手放在前两列

def convert_file(input_path, output_path):
    with open(input_path, 'r', newline='') as infile, open(output_path, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        input_header = next(reader)  # 读取输入文件的表头 (并跳过)

        # 创建输出文件的表头
        output_header = ['label', 'handedness']
        for i in range(1, 21):
            output_header.extend([f'rx{i}', f'ry{i}', f'rz{i}'])
        writer.writerow(output_header)

        for row in reader:
            if not row:
                continue
            writer.writerow(calculate_relative_coords(row))

# === 用法 ===
input_train_path = 'data/processed/hand_landmarks_train_left_only.csv'
output_train_path = 'data/processed/hand_landmarks_train_relative_left_only.csv'
convert_file(input_train_path, output_train_path)
print(f"✅ {input_train_path} 的坐标已转换为相对于手腕的相对坐标，并保存到 {output_train_path} (包含表头)")

input_test_path = 'data/processed/hand_landmarks_test_left_only.csv'
output_test_path = 'data/processed/hand_landmarks_test_relative_left_only.csv'
convert_file(input_test_path, output_test_path)
print(f"✅ {input_test_path} 的坐标已转换为相对于手腕的相对坐标，并保存到 {output_test_path} (包含表头)")