# 语音识别字幕生成器

这是一个将音频或视频文件转换为字幕文件的Python工具。它使用OpenAI的Whisper模型来识别音频内容，并生成SRT格式的字幕文件。

## 功能特点

- 支持多种音频格式（mp3, wav, ogg等）
- 支持多种视频格式（mp4, avi, mov, mkv, wmv等）
- 使用Whisper模型进行离线语音识别
- 生成标准SRT格式字幕文件
- 支持多语言识别
- 自动语言检测，智能处理翻译
- 自动分段生成字幕
- 支持多种字幕格式（原文、中文、双字幕）

## 安装要求

1. Python 3.8+
2. FFmpeg
3. CUDA（可选，用于GPU加速）
4. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 使用方法

基本用法：
```bash
python speech_to_subtitle.py input_file.mp4 output_subtitle.srt
```

指定语言（默认为中文）：
```bash
python speech_to_subtitle.py input_file.mp4 output_subtitle.srt --language en
```

选择不同的模型大小（默认为tiny）：
```bash
python speech_to_subtitle.py input_file.mp4 output_subtitle.srt --model base
```

选择字幕类型：
```bash
# 生成原文字幕
python speech_to_subtitle.py input_file.mp4 output_subtitle.srt --subtitle-type original

# 生成中文字幕
python speech_to_subtitle.py input_file.mp4 output_subtitle.srt --subtitle-type chinese

# 生成双字幕（原文+中文）
python speech_to_subtitle.py input_file.mp4 output_subtitle.srt --subtitle-type dual
```

## 字幕类型说明

- original: 生成原始语言字幕
- chinese: 生成中文翻译字幕
- dual: 生成双字幕（原文在上，中文在下）

## 模型大小说明

- tiny: 最小模型，速度最快，准确度较低（默认）
- base: 基础模型，平衡速度和准确度
- small: 小型模型，较好的准确度
- medium: 中型模型，高准确度
- large: 最大模型，最高准确度，但需要更多内存

## 注意事项

1. 首次运行时会自动下载Whisper模型
2. 较大的模型需要更多的内存和计算资源
3. 确保音频/视频质量清晰以获得更好的识别效果
4. 处理视频文件时，会先提取音频再进行识别
5. 程序会自动检测音频语言，对于中文内容会自动跳过翻译步骤
6. 使用GPU可以显著提升处理速度

## 支持的文件格式

支持的音频格式：
- MP3 (.mp3)
- WAV (.wav)
- OGG (.ogg)
- FLAC (.flac)
- M4A (.m4a)

支持的视频格式：
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)

## 支持的语言

Whisper支持多种语言，包括但不限于：
- 中文：zh
- 英语：en
- 日语：ja
- 韩语：ko
- 法语：fr
- 德语：de
- 西班牙语：es

## 常见问题

1. 如果遇到FFmpeg相关错误，请确保正确安装了FFmpeg并添加到系统环境变量
2. 如果遇到CUDA相关错误，可以尝试使用CPU版本
3. 对于较长的音频文件，建议使用较小的模型（tiny或base）以提高处理速度

## 支持的语言代码

- 中文：zh-CN
- 英语：en-US
- 日语：ja-JP
- 韩语：ko-KR
- 更多语言代码请参考Google Speech Recognition API文档 