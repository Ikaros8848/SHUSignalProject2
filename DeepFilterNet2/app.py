# 接口的降噪模块，千万不能删

from __future__ import annotations

import tempfile

from loguru import logger

from audio_backend import get_deepfilter_service, load_audio_file, save_audio_file


def denoise_audio(input_audio_path: str, output_audio_path: str | None = None) -> str:
    try:
        samples, sample_rate = load_audio_file(input_audio_path)
        enhanced = get_deepfilter_service().enhance_samples(samples, sample_rate)
        if output_audio_path is None:
            output_audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        save_audio_file(output_audio_path, enhanced, sample_rate)
        return output_audio_path
    except Exception as exc:
        logger.error(f"降噪过程中出错: {exc}")
        raise
