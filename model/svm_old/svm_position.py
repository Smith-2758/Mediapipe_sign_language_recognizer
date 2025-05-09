import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. 读取数据 ===
train_df = pd.read_csv('data/raw/hand_landmarks_train.csv')
test_df = pd.read_csv('data/raw/hand_landmarks_test.csv')

# === 2. 提取特征和标签 ===
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']

X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# === 3. 特征归一化 ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 4. 定义并训练 SVM 模型 ===
model = SVC(kernel='rbf', C=10, gamma='scale')  # 可以调参
model.fit(X_train_scaled, y_train)

# === 5. 模型预测 ===
y_pred = model.predict(X_test_scaled)

# === 6. 性能评估 ===
print("✅ 准确率:", accuracy_score(y_test, y_pred))
print("\n📊 分类报告:\n", classification_report(y_test, y_pred))

# === 7. 混淆矩阵可视化 ===
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=sorted(y_test.unique()), yticklabels=sorted(y_test.unique()), cmap='Blues')
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("混淆矩阵")
plt.show()
