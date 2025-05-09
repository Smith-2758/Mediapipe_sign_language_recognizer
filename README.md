# 手语识别系统

基于计算机视觉的实时手语识别系统，可以将手语手势转换为文字和语音输出。

## 功能特性

- 实时手势识别
- 支持26个英文字母的手语识别
- 支持空格和删除等特殊手势
- 文字转语音输出
- 支持视频录制

## 技术栈

- **计算机视觉框架**：
  - MediaPipe Hands - 用于实时手部关键点检测
  - OpenCV - 用于视频捕获和图像处理

- **机器学习**：
  - Scikit-learn - 使用 SVM 模型进行手势分类
  - 特征工程 - 实现了手部关键点的相对坐标归一化

- **语音合成**：
  - gTTS (Google Text-to-Speech) - 实现文字到语音的转换
  - PyGame - 用于音频播放

## 核心功能实现

### 1. 手势检测与识别
- 使用 MediaPipe 实时检测手部21个关键点
- 通过相对坐标归一化处理消除手部大小和位置差异
- 采用 SVM 分类器实现手势识别

### 2. 实时反馈系统
- 视觉反馈：在画面中显示识别结果和手部关键点
- 语音反馈：实时将识别的字母转换为语音
- 支持特殊手势：空格(手势"1")和删除(手势"2")

## 项目结构

```
├── data/               # 数据目录
├── model/             # 模型目录
│   └── svm_model/     # 训练好的SVM模型
├── src/               # 源代码
│   ├── extract_data/  # 数据提取
│   ├── inference/     # 推理代码
│   ├── process_data/  # 数据处理
│   ├── reference/     # 参考代码
│   └── test/          # 测试代码
├── output/            # 输出目录
└── requirements.txt   # 项目依赖
```

## 环境要求

- Python 3.8+
- OpenCV 4.5+
- MediaPipe 0.8+
- Scikit-learn 0.24+
- PyGame 2.0+
- gTTS 2.2+

## 环境配置

1. 克隆仓库:
```bash
git clone https://github.com/YOUR_USERNAME/sign_language_recognizer.git
cd sign_language_recognizer
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

## 使用方法

1. 实时手语识别:
```bash
python src/inference/inference_rela_nor_record_voice_react_google_realtime.py
```

2. 操作说明:
- 打开程序后，将手掌对准摄像头
- 每4秒进行一次手势识别
- 绿色边框表示即将进行识别
- 使用手势"1"输入空格
- 使用手势"2"删除上一个输入
- 按'q'键退出程序

## 模型训练

模型训练使用了以下数据集：
- yolo训练所用以及初始数据集：数据集来源于 RoboFlow 提供的公开数据集平台 [2]。图像总数约为1500张，每类样本数量基本均衡；该数据集的特点为手语字母图像为不同环境、肤色、光照以及角度等条件下拍摄的。
(https://public.roboflow.com/object-detection/american-sign-language-letters/1/download/yolov5pytorch)
- Mediapipe技术路径最终使用的手语字母数据集：数据集包含87,000张图像，每张图像的尺寸为200×200像素。该数据集共分为29个类别，其中26个类别对应英文字母A–Z（如图3，不包含NOTHING），另外3个类别分别为SPACE（空格）、DELETE（删除）和NOTHING（无动作）,每个类别存在3000张图片。(https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- 特征提取：基于MediaPipe的手部关键点检测
(https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?hl=zh-cn)


## 注意事项

- 确保环境光线充足
- 手掌需要完整出现在摄像头画面中
- 保持手势稳定以提高识别准确率

## 许可证

MIT License

## 致谢

- MediaPipe团队提供的优秀手部检测框架
