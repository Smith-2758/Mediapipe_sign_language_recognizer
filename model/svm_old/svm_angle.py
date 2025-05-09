import os
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# === 尝试设置 JOBlib 的临时文件夹 ===
temp_folder_path = 'D:\codehub\sign_language_recognizer/temp\joblib_temp'  # 替换为你系统上的有效路径 (例如 'C:\\temp\\joblib_temp' on Windows)
os.environ['JOBLIB_TEMP_FOLDER'] = temp_folder_path

# 确保临时文件夹存在
if not os.path.exists(temp_folder_path):
    try:
        os.makedirs(temp_folder_path)
        print(f"Created temporary folder for joblib: {temp_folder_path}")
    except OSError as e:
        print(f"Error creating temporary folder {temp_folder_path}: {e}")

# === 读取训练/测试数据（基于角度特征） ===
train_path = 'data/processed/hand_landmarks_train_angles.csv'
test_path = 'data/processed/hand_landmarks_test_angles.csv'

df_train = pd.read_csv(train_path, header=None)
df_test = pd.read_csv(test_path, header=None)

# === 特征与标签分离 ===
X_train = df_train.iloc[:, 1:].values  # 角度特征 (第一列是标签)
y_train = df_train.iloc[:, 0].values   # 类别标签

X_test = df_test.iloc[:, 1:].values
y_test = df_test.iloc[:, 0].values

# === 标签编码（A~Z → 0~25） ===
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

# === 特征标准化 ===
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# === 创建并训练SVM模型 ===
clf = svm.SVC(kernel='rbf', C=10, gamma='scale')
try:
    clf.fit(X_train_std, y_train_enc)

    # === 在测试集上进行评估 ===
    y_pred = clf.predict(X_test_std)
    acc = accuracy_score(y_test_enc, y_pred)

    print(f"\n✅ 测试集准确率: {acc * 100:.2f}%")
    print("\n📊 测试集分类报告:")
    print(classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_))

    # === 可视化混淆矩阵 ===
    plt.figure(figsize=(16, 12))
    cm = confusion_matrix(y_test_enc, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("测试集混淆矩阵")
    plt.show()

except Exception as e:
    print(f"发生错误: {e}")