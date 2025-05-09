import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# === 读取训练/测试数据（基于距离特征） ===
train_path = 'data/processed/hand_landmarks_train_dist.csv'
test_path  = 'data/processed/hand_landmarks_test_dist.csv'

df_train = pd.read_csv(train_path, header=None)
df_test  = pd.read_csv(test_path, header=None)

# === 特征与标签分离 ===
X_train = df_train.iloc[:, :-1].values  # 距离特征
y_train = df_train.iloc[:, -1].values   # 类别标签

X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values

# === 标签编码（A~Z → 0~25） ===
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

# === 特征标准化（建议用于 SVM） ===
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# === 训练 SVM 模型 ===
clf = svm.SVC(kernel='rbf', C=10, gamma='scale')  # 你也可以尝试 linear、poly 等核
clf.fit(X_train_std, y_train_enc)

# === 测试与评估 ===
y_pred = clf.predict(X_test_std)
acc = accuracy_score(y_test_enc, y_pred)

print(f"\n✅ 测试准确率：{acc * 100:.2f}%")
print("\n📊 分类报告:")
print(classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_))

# === 绘制混淆矩阵 ===
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test_enc, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('手语识别混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.show()


