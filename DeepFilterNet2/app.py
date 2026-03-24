# 接口的降噪模块，千万不能删

from __future__ import annotations

import tempfile

from loguru import logger

from audio_backend import get_deepfilter_service, load_audio_file, save_audio_file


def denoise_audio(input_audio_path: str, output_audio_path: str | None = None) -> str:
    try:
        # 读取音频并做基础规范化（在 audio_backend 内部会处理单声道/幅值范围等）。
        samples, sample_rate = load_audio_file(input_audio_path)
        # 送入 DeepFilterNet2 推理：内部若采样率不匹配会先重采样到模型目标采样率，
        # 推理后再重采样回原采样率，保证输出与输入采样率一致。
        enhanced = get_deepfilter_service().enhance_samples(samples, sample_rate)
        if output_audio_path is None:
            # 未指定输出路径时，创建临时 wav 文件并返回该路径。
            output_audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        # 保存增强后的波形；保存前会再次做安全归一化，避免幅值越界导致削波。
        save_audio_file(output_audio_path, enhanced, sample_rate)
        return output_audio_path
    except Exception as exc:
        # 统一记录异常，便于接口层定位失败原因。
        logger.error(f"降噪过程中出错: {exc}")
        raise
