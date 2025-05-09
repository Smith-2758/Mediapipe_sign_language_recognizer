# 手语识别系统

基于计算机视觉的实时手语识别系统，可以将手语手势转换为文字和语音输出。

## 功能特性

- 实时手势识别
- 支持26个英文字母的手语识别
- 支持空格和删除等特殊手势
- 文字转语音输出
- 支持视频录制

## 项目结构

```
├── data/               # 数据目录
├── model/             # 模型目录
├── src/               # 源代码
│   ├── extract_data/  # 数据提取
│   ├── inference/     # 推理代码
│   ├── process_data/  # 数据处理
│   ├── reference/     # 参考代码
│   └── test/          # 测试代码
├── output/            # 输出目录
└── requirements.txt   # 项目依赖
```

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

## 许可证

MIT License