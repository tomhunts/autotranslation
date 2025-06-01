import os
import whisper
from pydub import AudioSegment
import srt
from datetime import timedelta
import argparse
from moviepy.editor import VideoFileClip
import tempfile
import shutil

class SpeechToSubtitle:
    def __init__(self, model_size="base"):
        """初始化Whisper模型
        model_size: 可选 "tiny", "base", "small", "medium", "large"
        """
        self.model = whisper.load_model(model_size)
        
    def extract_audio_from_video(self, video_file):
        """从视频文件中提取音频"""
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            video = VideoFileClip(video_file)
            if video.audio is None:
                print("警告：视频文件没有音频轨道")
                return None
            video.audio.write_audiofile(temp_audio.name, codec='pcm_s16le')
            video.close()
            return temp_audio.name
        except Exception as e:
            print(f"从视频提取音频时出错: {str(e)}")
            if os.path.exists(temp_audio.name):
                os.unlink(temp_audio.name)
            return None

    def convert_audio_to_wav(self, input_file):
        """将音频文件转换为WAV格式"""
        try:
            audio = AudioSegment.from_file(input_file)
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            audio.export(temp_wav.name, format="wav")
            return temp_wav.name
        except Exception as e:
            print(f"转换音频格式时出错: {str(e)}")
            return None

    def transcribe_audio(self, audio_file, language='zh', task='transcribe'):
        """使用Whisper模型将音频文件转换为文本
        task: 'transcribe' - 转录, 'translate' - 翻译
        """
        try:
            if not os.path.exists(audio_file):
                print(f"错误：音频文件不存在: {audio_file}")
                return None
                
            # 使用Whisper进行识别
            result = self.model.transcribe(
                audio_file,
                language=language,
                task=task
            )
            
            # 提取识别结果
            segments = []
            for segment in result["segments"]:
                segments.append({
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"]
                })
            
            return segments
        except Exception as e:
            print(f"识别过程中出现错误: {str(e)}")
            return None

    def create_subtitles(self, segments, subtitle_type='original'):
        """创建字幕文件
        subtitle_type: 'original' - 原文, 'chinese' - 中文, 'dual' - 双字幕
        """
        subtitles = []
        for i, segment in enumerate(segments, 1):
            if subtitle_type == 'dual':
                # 双字幕模式：原文在上，中文在下
                content = f"{segment['text']}\n{segment.get('chinese_text', '')}"
            else:
                content = segment['text']
                
            subtitle = srt.Subtitle(
                index=i,
                start=timedelta(seconds=segment["start"]),
                end=timedelta(seconds=segment["end"]),
                content=content.strip()
            )
            subtitles.append(subtitle)
        return subtitles

    def process_media(self, input_file, output_file, language='zh', subtitle_type='original'):
        """处理媒体文件（音频或视频）并生成字幕"""
        temp_files = []
        
        try:
            if not os.path.exists(input_file):
                print(f"错误：输入文件不存在: {input_file}")
                return
                
            # 判断文件类型
            file_ext = os.path.splitext(input_file)[1].lower()
            if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                # 处理视频文件
                print("正在从视频中提取音频...")
                audio_file = self.extract_audio_from_video(input_file)
                if not audio_file:
                    return
                temp_files.append(audio_file)
            else:
                # 处理音频文件
                print("正在转换音频格式...")
                audio_file = self.convert_audio_to_wav(input_file)
                if not audio_file:
                    return
                temp_files.append(audio_file)
            
            if subtitle_type == 'chinese':
                # 直接使用translate任务生成中文字幕
                print("正在生成中文字幕...")
                segments = self.transcribe_audio(audio_file, language='zh', task='translate')
            elif subtitle_type == 'dual':
                # 先生成原文，再生成中文翻译
                print("正在生成原文字幕...")
                original_segments = self.transcribe_audio(audio_file, language=language, task='transcribe')
                print("正在生成中文翻译...")
                chinese_segments = self.transcribe_audio(audio_file, language='zh', task='translate')
                
                # 合并原文和翻译
                if original_segments and chinese_segments:
                    for i, segment in enumerate(original_segments):
                        if i < len(chinese_segments):
                            segment['chinese_text'] = chinese_segments[i]['text']
                    segments = original_segments
                else:
                    segments = None
            else:
                # 生成原文字幕
                print("正在生成原文字幕...")
                segments = self.transcribe_audio(audio_file, language=language, task='transcribe')
            
            if segments:
                # 创建字幕
                print("正在生成字幕文件...")
                subtitles = self.create_subtitles(segments, subtitle_type)
                
                # 保存字幕文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(srt.compose(subtitles))
                
                print(f"字幕文件已生成: {output_file}")
            else:
                print("未能识别音频内容")
        
        finally:
            # 清理临时文件
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except Exception as e:
                        print(f"清理临时文件时出错: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='将音频或视频文件转换为字幕文件')
    parser.add_argument('input_file', help='输入音频或视频文件路径')
    parser.add_argument('output_file', help='输出字幕文件路径')
    parser.add_argument('--language', default='zh', help='识别语言 (默认: zh)')
    parser.add_argument('--model', default='base', 
                      choices=['tiny', 'base', 'small', 'medium', 'large'],
                      help='Whisper模型大小 (默认: base)')
    parser.add_argument('--subtitle-type', default='original',
                      choices=['original', 'chinese', 'dual'],
                      help='字幕类型: original(原文), chinese(中文), dual(双字幕)')
    
    args = parser.parse_args()
    
    converter = SpeechToSubtitle(model_size=args.model)
    converter.process_media(args.input_file, args.output_file, args.language, args.subtitle_type)

if __name__ == "__main__":
    main() 