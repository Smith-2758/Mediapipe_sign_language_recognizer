import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # 设置后端为TkAgg
import matplotlib.pyplot as plt
import os
import joblib  # 用于保存模型

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# === 1. 读取数据（使用相对于手腕的相对坐标特征，包含 handedness） ===
train_path = 'data/processed/new_hand_landmarks_train_partial_normalized_no_wrist.csv'
test_path = 'data/processed/new_hand_landmarks_test_partial_normalized_no_wrist.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# === 2. 提取特征和标签 ===
X_train = train_df.drop(['label', 'handedness'], axis=1)
y_train = train_df['label']

X_test = test_df.drop(['label', 'handedness'], axis=1)
y_test = test_df['label']

# === 3. 标签编码 ===
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# === 4. 特征标准化 ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 5. 定义并训练 SVM 模型 ===
model = SVC(kernel='rbf', C=10, gamma='scale')  # 可以调参
model.fit(X_train_scaled, y_train_encoded)

# === 6. 模型预测 ===
y_pred_encoded = model.predict(X_test_scaled)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# === 7. 性能评估 ===
print("✅ 准确率:", accuracy_score(y_test, y_pred))
print("\n📊 分类报告:\n", classification_report(y_test, y_pred))

# 确保results目录存在
os.makedirs('results', exist_ok=True)

# 绘制并保存混淆矩阵
cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y_test.unique()), yticklabels=sorted(y_test.unique()))
plt.title('手语识别混淆矩阵 (使用部分归一化数据)')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.savefig('results/new_confusion_matrix_partial_normalized2.png', bbox_inches='tight', dpi=300)
plt.show()  # 使用 plt.show() 来显示图形并保持打开状态

# === 8. 保存模型、Scaler 和 LabelEncoder ===
model_path = 'model/svm_model/relative_normalized/new_hand_gesture_recognition_model_partial_normalized2.pkl'
joblib.dump(model, model_path)
print(f"✅ SVM 模型已保存到: {model_path}")

scaler_path = 'model/svm_model/relative_normalized/new_hand_gesture_recognition_scaler_partial_normalized2.pkl'
joblib.dump(scaler, scaler_path)
print(f"✅ StandardScaler 已保存到: {scaler_path}")

label_encoder_path = 'model/svm_model/relative_normalized/new_hand_gesture_recognition_label_encoder_partial_normalized2.pkl'
joblib.dump(label_encoder, label_encoder_path)
print(f"✅ LabelEncoder 已保存到: {label_encoder_path}")