# 端口：7999
# 自动切分，可以加载长音频
import gradio as gr
import tempfile
import os
from pydub import AudioSegment

from app import denoise_audio

def split_audio(audio_path, chunk_length_ms=600000):
    """将音频文件切割为指定长度的片段"""
    audio = AudioSegment.from_file(audio_path)
    # 以固定时间窗进行切分：
    # - chunk_length_ms 单位是毫秒，默认 600000 ms = 10 分钟
    # - range 的步长与切片长度一致，因此是“无重叠分块”
    # - 对超长音频可显著降低单次推理的峰值内存/显存占用
    chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

def merge_audio(chunks, output_path):
    """合并多个音频片段"""
    combined = AudioSegment.empty()
    for chunk in chunks:
        # 顺序累加拼接，保持与原时间轴一致（第 n 段接在第 n-1 段后）。
        # 这里不做交叉淡入淡出，避免额外改变波形幅度与能量分布。
        combined += chunk
    combined.export(output_path, format="wav")

def process_chunk(chunk, index, temp_folder):
    """处理单个音频片段并返回处理后的路径"""
    chunk_path = os.path.join(temp_folder, f"chunk_{index}.wav")
    chunk.export(chunk_path, format="wav")
    output_path = os.path.join(temp_folder, f"processed_{index}.wav")
    # 每个分块独立执行降噪；最终效果由后续“按序合并”重建完整时序。
    denoise_audio(chunk_path, output_path)
    return output_path

def process_audio(input_file: str) -> str:
    """处理上传的音频文件，包括降噪和可能的格式转换"""
    temp_folder = tempfile.mkdtemp()
    output_audio_path = os.path.join(temp_folder, "final_output.wav")

    # 将音频切割成片段并处理每个片段
    chunks = split_audio(input_file)
    # enumerate(chunks) 的索引 i 保证“处理后文件名”和“原始片段顺序”一一对应，
    # 从而在合并阶段不会出现时序错乱。
    processed_chunks = [AudioSegment.from_wav(process_chunk(chunk, i, temp_folder)) for i, chunk in enumerate(chunks)]
    
    # 合并处理后的音频片段
    merge_audio(processed_chunks, output_audio_path)
    
    return output_audio_path

def gradio_interface(input_audio):
    """Gradio接口函数，处理音频并返回结果"""
    processed_audio_path = process_audio(input_audio)
    return processed_audio_path

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>录音降噪——JYD</h1>")
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="上传需要降噪音频文件")
        process_button = gr.Button("降噪", variant="primary")
    with gr.Row():
        gr.Examples(
                [
                    "/opt/jyd01/wangruihua/data/audio/noise1.mp3",
                    "/opt/jyd01/wangruihua/data/audio/noise2.mp3",
                    "/opt/jyd01/wangruihua/data/audio/noise3.mp3",
                    "/opt/jyd01/wangruihua/data/audio/noise4.mp3",
                    "/opt/jyd01/wangruihua/data/audio/noise_class.wav",
                    # "/opt/jyd01/wangruihua/4090copy/synthesis/audio/101数学课.wav",
                    # "/home/jyd01/wangruihua/synthesis/audio/101历史课.wav",
                ],
                [audio_input],label='课堂音频')
    audio_output = gr.Audio(label="降噪后的音频")

    process_button.click(fn=gradio_interface, inputs=audio_input, outputs=audio_output)

if __name__ == "__main__":
    demo.launch(server_port=7999, server_name='0.0.0.0')
