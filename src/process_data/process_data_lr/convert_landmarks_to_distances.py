import csv
import math

def extract_distances(row):
    label = row[0]
    coords = list(map(float, row[1:]))  # [x0,y0,z0,x1,y1,z1,...]
    landmarks = [(coords[i], coords[i+1], coords[i+2]) for i in range(0, len(coords), 3)]

    # 计算所有关键点到第0点（手腕）的距离
    base_point = landmarks[0]
    distances = []
    for point in landmarks[1:]:
        d = math.sqrt(sum((bp - p)**2 for bp, p in zip(base_point, point)))
        distances.append(d)

    return distances + [label]  # 将标签放在最后一列

def convert_file(input_path, output_path):
    with open(input_path, 'r', newline='') as infile, open(output_path, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)  # ✅ 跳过表头

        for row in reader:
            if not row:
                continue
            writer.writerow(extract_distances(row))

# === 用法 ===
convert_file('data/raw/hand_landmarks_train.csv', 'data/processed/hand_landmarks_train_dist.csv')
convert_file('data/raw/hand_landmarks_test.csv', 'data/processed/hand_landmarks_test_dist.csv')
