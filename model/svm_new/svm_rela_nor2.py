"""
手语识别模型训练和评估脚本
该脚本实现了基于SVM的手语识别模型的训练、评估和保存功能。
使用相对坐标特征和标准化处理来提高模型的泛化能力。
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # 设置matplotlib后端为TkAgg，解决可能的显示问题
import matplotlib.pyplot as plt
import os
import joblib  # 用于模型的序列化和反序列化

# === 配置matplotlib以支持中文显示 ===
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# === 1. 数据加载 ===
# 读取预处理后的训练集和测试集数据
# 使用相对于手腕的归一化坐标特征，不包含手腕坐标点
train_path = 'data/processed/new_hand_landmarks_train_partial_normalized_no_wrist.csv'
test_path = 'data/processed/new_hand_landmarks_test_partial_normalized_no_wrist.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# === 2. 特征和标签准备 ===
# 分离特征和标签，排除handedness列
X_train = train_df.drop(['label', 'handedness'], axis=1)  # 特征矩阵
y_train = train_df['label']                               # 训练集标签

X_test = test_df.drop(['label', 'handedness'], axis=1)    # 测试集特征
y_test = test_df['label']                                 # 测试集标签

# === 3. 标签编码 ===
# 将字符串标签转换为数值型标签
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)    # 训练集标签编码
y_test_encoded = label_encoder.transform(y_test)          # 测试集标签编码

# === 4. 特征标准化 ===
# 对特征进行标准化处理，使其均值为0，方差为1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)    # 训练集特征标准化
X_test_scaled = scaler.transform(X_test)          # 测试集特征标准化

# === 5. SVM模型训练 ===
# 使用RBF核函数的SVM分类器
model = SVC(kernel='rbf', C=10, gamma='scale')    # C为惩罚参数，gamma为RBF核参数
model.fit(X_train_scaled, y_train_encoded)        # 训练模型

# === 6. 模型预测 ===
# 对测试集进行预测并将编码的标签转换回原始标签
y_pred_encoded = model.predict(X_test_scaled)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# === 7. 模型评估 ===
# 打印模型性能指标
print("✅ 准确率:", accuracy_score(y_test, y_pred))
print("\n📊 分类报告:\n", classification_report(y_test, y_pred))

# 确保结果保存目录存在
os.makedirs('results', exist_ok=True)

# === 8. 绘制并保存混淆矩阵 ===
# 计算混淆矩阵并转换为百分比形式
cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# 绘制热力图形式的混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=sorted(y_test.unique()), 
            yticklabels=sorted(y_test.unique()))
plt.title('手语识别混淆矩阵 (归一化百分比)')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.savefig('results/new_confusion_matrix_partial_normalized2.png', 
            bbox_inches='tight', dpi=300)
plt.show()

# === 9. 保存模型和预处理器 ===
# 保存训练好的模型
model_path = 'model/svm_model/relative_normalized/new_hand_gesture_recognition_model_partial_normalized2.pkl'
joblib.dump(model, model_path)
print(f"✅ SVM 模型已保存到: {model_path}")

# 保存特征标准化器
scaler_path = 'model/svm_model/relative_normalized/new_hand_gesture_recognition_scaler_partial_normalized2.pkl'
joblib.dump(scaler, scaler_path)
print(f"✅ StandardScaler 已保存到: {scaler_path}")

# 保存标签编码器
label_encoder_path = 'model/svm_model/relative_normalized/new_hand_gesture_recognition_label_encoder_partial_normalized2.pkl'
joblib.dump(label_encoder, label_encoder_path)
print(f"✅ LabelEncoder 已保存到: {label_encoder_path}")