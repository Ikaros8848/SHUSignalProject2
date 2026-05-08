from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import threading

import librosa
import numpy as np
import soundfile as sf
import torch
from pesq import pesq
import scipy.special as sp
from scipy.signal import istft, resample_poly, stft

from df.enhance import enhance, init_df


EPSILON = 1e-8
DEEPFILTER_TARGET_SR = 48000


@dataclass(slots=True)
class QualityMetrics:
    snr: float | None
    seg_snr: float | None
    pesq: float | None


@dataclass(slots=True)
class SpectrogramData:
    freqs: np.ndarray
    times: np.ndarray
    magnitude_db: np.ndarray


@dataclass(slots=True)
class NoiseDiagnosis:
    label: str
    dominant_band: str
    dominant_frequency_hz: float
    spectral_centroid_hz: float
    spectral_flatness: float
    band_energies: dict[str, float]


@dataclass(slots=True)
class AudioAnalysis:
    algorithm: str
    samples: np.ndarray
    sample_rate: int
    spectrogram: SpectrogramData
    metrics: QualityMetrics


@dataclass(slots=True)
class ComparisonResult:
    noisy: AudioAnalysis
    deepfilter: AudioAnalysis
    mmse: AudioAnalysis
    diagnosis: NoiseDiagnosis
    reference_metrics_ready: bool


@dataclass(slots=True)
class MMSEParameters:
    suppression_strength: float = 0.68
    temporal_smoothing: float = 0.60
    speech_protection: float = 0.50
    frame_ms: float = 32.0
    overlap: float = 0.75


def _ensure_mono(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 0:
        return audio.reshape(1)
    if audio.ndim == 2:
        axis = 1 if audio.shape[1] <= audio.shape[0] else 0
        audio = audio.mean(axis=axis)
    return np.squeeze(audio).astype(np.float32)


def normalize_audio(samples: np.ndarray) -> np.ndarray:
    audio = _ensure_mono(samples)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def resample_audio(samples: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    audio = normalize_audio(samples)
    if original_sr == target_sr or audio.size == 0:
        return audio
    factor = math.gcd(int(original_sr), int(target_sr))
    resampled = resample_poly(audio, target_sr // factor, original_sr // factor)
    return normalize_audio(resampled)


def load_audio_file(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    path_str = str(path)
    try:
        samples, sample_rate = sf.read(path_str, dtype="float32", always_2d=False)
    except Exception:
        samples, sample_rate = librosa.load(path_str, sr=None, mono=False, dtype=np.float32)
    sample_rate = int(sample_rate)
    audio = normalize_audio(samples)
    if target_sr is not None and sample_rate != target_sr:
        audio = resample_audio(audio, sample_rate, target_sr)
        sample_rate = target_sr
    return audio, int(sample_rate)


def save_audio_file(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), normalize_audio(samples), sample_rate)


def _align_signals(reference: np.ndarray, degraded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = min(reference.size, degraded.size)
    if length <= 0:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)
    return reference[:length], degraded[:length]


def compute_snr(reference: np.ndarray, degraded: np.ndarray) -> float:
    ref, deg = _align_signals(reference, degraded)
    # 噪声定义为“参考干净信号 - 当前退化信号”，对应经典加性噪声模型：y = s + n。
    noise = ref - deg
    # 全局 SNR（dB）：10 * log10(信号能量 / 噪声能量)。
    # 增加 EPSILON 避免在分母接近 0 时出现数值溢出。
    return float(10.0 * np.log10((np.sum(ref**2) + EPSILON) / (np.sum(noise**2) + EPSILON)))


def compute_segmental_snr(reference: np.ndarray, degraded: np.ndarray, sample_rate: int) -> float:
    ref, deg = _align_signals(reference, degraded)
    # 分段 SNR（SegSNR）按短时帧统计，更接近人耳对局部质量变化的感知。
    # 这里使用 20ms 帧长，50% 重叠（hop = frame_length // 2）。
    frame_length = max(128, int(sample_rate * 0.02))
    hop = max(64, frame_length // 2)
    frame_scores: list[float] = []
    for start in range(0, max(ref.size - frame_length + 1, 1), hop):
        ref_frame = ref[start : start + frame_length]
        deg_frame = deg[start : start + frame_length]
        if ref_frame.size < frame_length:
            break
        error_frame = ref_frame - deg_frame
        # 每一帧都单独计算 SNR，再做均值，防止某些高能量片段“掩盖”其他片段。
        frame_snr = 10.0 * np.log10((np.sum(ref_frame**2) + EPSILON) / (np.sum(error_frame**2) + EPSILON))
        # 经典 SegSNR 常做截断，避免极端值主导均值（-10 dB 到 35 dB）。
        frame_scores.append(float(np.clip(frame_snr, -10.0, 35.0)))
    if not frame_scores:
        return compute_snr(ref, deg)
    return float(np.mean(frame_scores))


def compute_pesq_score(reference: np.ndarray, degraded: np.ndarray, sample_rate: int) -> float | None:
    ref, deg = _align_signals(reference, degraded)
    if ref.size < int(sample_rate * 0.25):
        return None
    target_sr = 16000 if sample_rate >= 16000 else 8000
    mode = "wb" if target_sr == 16000 else "nb"
    ref_resampled = resample_audio(ref, sample_rate, target_sr)
    deg_resampled = resample_audio(deg, sample_rate, target_sr)
    limit = min(ref_resampled.size, deg_resampled.size)
    if limit <= 0:
        return None
    try:
        score = pesq(target_sr, ref_resampled[:limit], deg_resampled[:limit], mode)
    except Exception:
        return None
    return float(score)


def compute_metrics(reference: np.ndarray | None, degraded: np.ndarray, sample_rate: int) -> QualityMetrics:
    if reference is None:
        return QualityMetrics(snr=None, seg_snr=None, pesq=None)
    return QualityMetrics(
        snr=compute_snr(reference, degraded),
        seg_snr=compute_segmental_snr(reference, degraded, sample_rate),
        pesq=compute_pesq_score(reference, degraded, sample_rate),
    )


def compute_spectrogram(samples: np.ndarray, sample_rate: int) -> SpectrogramData:
    audio = normalize_audio(samples)
    if audio.size == 0:
        return SpectrogramData(
            freqs=np.zeros(1, dtype=np.float32),
            times=np.zeros(1, dtype=np.float32),
            magnitude_db=np.zeros((1, 1), dtype=np.float32),
        )
    # STFT 窗长按采样率自适应到约 20ms，并取 2 的幂，平衡频率分辨率与计算效率。
    window = min(max(256, 2 ** math.ceil(math.log2(max(int(sample_rate * 0.02), 32)))), audio.size)
    if window < 16:
        window = min(16, audio.size)
    # 75% 重叠：时间连续性更好，适合语音等非平稳信号。
    overlap = min(window - 1, int(window * 0.75))
    freqs, times, spectrum = stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=window,
        noverlap=overlap,
        boundary="zeros",
        padded=True,
    )
    # 幅度谱转 dB：20*log10(|X|)。
    # 下限 1e-4 是为了避免 log(0) 并抑制极小值带来的数值不稳定。
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(spectrum), 1e-4))
    return SpectrogramData(freqs=freqs, times=times, magnitude_db=magnitude_db.astype(np.float32))


def diagnose_noise(samples: np.ndarray, sample_rate: int) -> NoiseDiagnosis:
    audio = normalize_audio(samples)
    if audio.size == 0:
        return NoiseDiagnosis(
            label="未检测到有效音频",
            dominant_band="未知",
            dominant_frequency_hz=0.0,
            spectral_centroid_hz=0.0,
            spectral_flatness=0.0,
            band_energies={"low": 0.0, "mid": 0.0, "high": 0.0},
        )

    # 先乘 Hann 窗减轻频谱泄漏，再做实数 FFT（rFFT）得到单边频谱。
    spectrum = np.fft.rfft(audio * np.hanning(audio.size))
    # 功率谱 = 幅度平方，表示各频率分量的能量贡献。
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / sample_rate)
    # 总功率用于做归一化，得到各频段“能量占比”。
    total_power = float(np.sum(power) + EPSILON)

    low_mask = freqs < 300.0
    mid_mask = (freqs >= 300.0) & (freqs < 3000.0)
    high_mask = freqs >= 3000.0
    band_energies = {
        "low": float(np.sum(power[low_mask]) / total_power),
        "mid": float(np.sum(power[mid_mask]) / total_power),
        "high": float(np.sum(power[high_mask]) / total_power),
    }

    dominant_band_key = max(band_energies, key=lambda key: band_energies[key])
    dominant_band_names = {
        "low": "低频段 0-300 Hz",
        "mid": "中频段 300-3000 Hz",
        "high": "高频段 3000 Hz 以上",
    }

    # 主导频率：功率谱峰值所在的频点。
    dominant_frequency = float(freqs[int(np.argmax(power))]) if power.size else 0.0
    # 频谱质心：Σ(f*P)/ΣP，可理解为“能量重心频率”。
    centroid = float(np.sum(freqs * power) / total_power)
    # 频谱平坦度：几何均值/算术均值。
    # 值越大越接近“白噪声”，越小越偏“窄带/谐波”结构。
    flatness = float(np.exp(np.mean(np.log(power + EPSILON))) / (np.mean(power) + EPSILON))
    # 峰均比（Crest Factor）：峰值 / RMS。
    # 较高时通常意味着突发脉冲或瞬态成分明显。
    crest_factor = float(np.max(np.abs(audio)) / (np.sqrt(np.mean(audio**2) + EPSILON) + EPSILON))

    if crest_factor > 8.0:
        label = "瞬态冲击噪声占主导"
    elif dominant_band_key == "low" and flatness < 0.2:
        label = "低频机械或电源嗡声占主导"
    elif dominant_band_key == "high" and flatness > 0.35:
        label = "高频宽带嘶声占主导"
    elif dominant_band_key == "mid" and flatness < 0.3:
        label = "中频环境声或人声串扰占主导"
    else:
        label = "复合宽带噪声占主导"

    return NoiseDiagnosis(
        label=label,
        dominant_band=dominant_band_names[dominant_band_key],
        dominant_frequency_hz=dominant_frequency,
        spectral_centroid_hz=centroid,
        spectral_flatness=flatness,
        band_energies=band_energies,
    )


class DeepFilterService:
    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else Path(__file__).resolve().parent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        self._model: torch.nn.Module | None = None
        self._df_state = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._df_state is not None:
            return
        with self._lock:
            if self._model is not None and self._df_state is not None:
                return
            model, df_state, _ = init_df(
                str(self.model_dir),
                config_allow_defaults=True,
                log_file=str(self.model_dir / "enhance.log"),
            )
            self._model = model.to(device=self.device).eval()
            self._df_state = df_state

    def enhance_samples(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = normalize_audio(samples)
        if audio.size == 0:
            return audio
        self._ensure_loaded()
        model = self._model
        df_state = self._df_state
        if model is None or df_state is None:
            raise RuntimeError("DeepFilterNet model is not initialized")
        working_audio = audio
        if sample_rate != DEEPFILTER_TARGET_SR:
            working_audio = resample_audio(audio, sample_rate, DEEPFILTER_TARGET_SR)
        tensor = torch.from_numpy(working_audio).unsqueeze(0).to(self.device)
        with torch.no_grad():
            enhanced = enhance(model, df_state, tensor)
        output = enhanced.squeeze(0).detach().cpu().numpy()
        if sample_rate != DEEPFILTER_TARGET_SR:
            output = resample_audio(output, DEEPFILTER_TARGET_SR, sample_rate)
        return normalize_audio(output)


_deepfilter_service: DeepFilterService | None = None


def get_deepfilter_service(model_dir: str | Path | None = None) -> DeepFilterService:
    global _deepfilter_service
    if _deepfilter_service is None:
        _deepfilter_service = DeepFilterService(model_dir=model_dir)
    return _deepfilter_service


def mmse_decision_directed_denoise(
    samples: np.ndarray,
    sample_rate: int,
    mmse_parameters: MMSEParameters | None = None,
) -> np.ndarray:
    audio = normalize_audio(samples)
    if audio.size == 0:
        return audio

    params = mmse_parameters or MMSEParameters()
    suppression_strength = float(np.clip(params.suppression_strength, 0.0, 1.0))
    temporal_smoothing = float(np.clip(params.temporal_smoothing, 0.0, 1.0))
    speech_protection = float(np.clip(params.speech_protection, 0.0, 1.0))
    frame_ms = float(np.clip(params.frame_ms, 16.0, 64.0))
    overlap = float(np.clip(params.overlap, 0.5, 0.9))

    # 以下是一组“用户参数 -> 算法内部参数”的映射：
    # - suppression_strength 越高，降噪更强（但语音失真风险上升）
    # - temporal_smoothing 越高，时域变化越平滑（但瞬态响应变慢）
    # - speech_protection 越高，语音保真优先（噪声残留可能增加）
    decision_alpha = float(np.clip(0.91 + 0.06 * temporal_smoothing, 0.89, 0.99))
    # 增益下限，避免把语音完全“压没”；数值越大，保留越多底噪与弱语音。
    gain_floor = float(np.clip(0.04 + 0.18 * speech_protection - 0.18 * suppression_strength, 0.015, 0.25))
    # 当前帧增益与上一帧增益的混合系数，用于抑制“音乐噪声”抖动。
    gain_blend = float(np.clip(0.78 + 0.12 * temporal_smoothing, 0.76, 0.95))
    # 语音存在概率的 SNR 阈值，阈值越低越容易判定为“有语音”。
    speech_threshold = float(np.clip(2.7 - 0.7 * suppression_strength + 0.35 * speech_protection, 1.1, 3.6))
    # 用分位数初始化噪声 PSD，分位数越低越保守（更偏向纯噪声底）。
    noise_percentile = float(np.clip(8.0 + 15.0 * suppression_strength - 7.0 * speech_protection, 5.0, 28.0))
    # 过减系数（over-subtraction）：增强时更明显抑制噪声地板。
    over_subtraction = float(np.clip(1.15 + 1.1 * suppression_strength - 0.45 * speech_protection, 1.05, 2.4))
    # 后滤波强度：在低语音存在概率处进一步衰减残余噪声。
    post_filter_strength = float(np.clip(0.12 + 0.4 * suppression_strength - 0.10 * speech_protection, 0.08, 0.50))
    # 语音存在时噪声更新非常慢，避免把语音误吸收到噪声模型中。
    speech_noise_smoothing = float(np.clip(0.994 + 0.004 * speech_protection, 0.993, 0.9995))
    non_speech_noise_smoothing = float(
        np.clip(0.68 + 0.12 * temporal_smoothing + 0.08 * speech_protection - 0.06 * suppression_strength, 0.56, 0.93)
    )
    dry_mix = float(np.clip(0.01 + 0.05 * speech_protection - 0.04 * suppression_strength, 0.0, 0.07))
    rms_limit = float(np.clip(1.30 + 0.25 * speech_protection, 1.1, 1.65))

    # 帧长（STFT 窗长）按毫秒换算后再取 2 的幂，便于 FFT 高效计算。
    frame_length = max(256, 2 ** math.ceil(math.log2(max(int(sample_rate * frame_ms / 1000.0), 32))))
    frame_length = min(frame_length, max(audio.size, 256))
    if frame_length <= 16:
        return audio
    hop = max(64, int(frame_length * (1.0 - overlap)))
    hop = min(hop, frame_length - 1)

    # STFT: x(n) -> X(k, m)，进入频域逐 bin 做噪声抑制。
    _, _, spectrum = stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=frame_length,
        noverlap=frame_length - hop,
        boundary="zeros",
        padded=True,
    )
    # 观测功率谱 |Y(k,m)|^2。
    power = np.abs(spectrum) ** 2 + EPSILON
    if power.shape[1] == 0:
        return audio

    # 前若干帧通常更接近“噪声底”，用于初始化噪声功率谱密度（PSD）。
    initial_frames = max(6, min(power.shape[1] // 8, 20))
    initial_noise_psd = np.mean(power[:, :initial_frames], axis=1) if initial_frames > 0 else np.mean(power, axis=1)
    # 与分位数估计取最小值，减少语音早期出现时的噪声高估。
    noise_psd = np.minimum(initial_noise_psd, np.percentile(power, noise_percentile, axis=1))
    enhanced_spectrum = np.zeros_like(spectrum)
    # 前一帧“净语音功率”作为 Decision-Directed 先验的记忆项。
    previous_clean_power = np.maximum(power[:, 0] - noise_psd, EPSILON)
    previous_gain = np.ones_like(noise_psd)
    # 频率方向 3 点平滑核，抑制孤立频点振铃与颗粒感。
    frequency_smoothing_kernel = np.array([0.15, 0.7, 0.15], dtype=np.float32)

    for frame_index in range(power.shape[1]):
        current_power = power[:, frame_index]
        # 过减后噪声 PSD：等价提高噪声假设，强化抑制。
        adjusted_noise_psd = np.maximum(over_subtraction * noise_psd, EPSILON)
        # 后验 SNR: γ(k,m) = |Y|^2 / λ_n。
        post_snr = current_power / adjusted_noise_psd
        # Decision-Directed 先验 SNR:
        # ξ(k,m) = α * (上一帧净语音功率/噪声PSD) + (1-α)*max(γ-1,0)
        prior_snr = (
            decision_alpha * previous_clean_power / adjusted_noise_psd
            + (1.0 - decision_alpha) * np.maximum(post_snr - 1.0, 0.0)
        )
        prior_snr = np.clip(prior_snr, 1e-4, 1e2)

        # 计算 v 参数 (先验与后验 SNR 的函数)
        v = (prior_snr / (1.0 + prior_snr)) * post_snr
        v_clipped = np.clip(v, 1e-10, 30.0)

        # 1. MMSE-LSA (Log-Spectral Amplitude) 增益估计
        # 使用 scipy.special.expi 计算指数积分 E1(v) = -expi(-v)
        gain_lsa = (prior_snr / (1.0 + prior_snr)) * np.exp(-0.5 * sp.expi(-v_clipped))

        # 2. 计算基于似然比的语音存在概率 (Speech Presence Probability, SPP)
        likelihood = np.exp(v_clipped) / (1.0 + prior_snr)
        # 动态调整先验缺失概率 (q值)，适当降低使得算法更倾向于保留声音
        q_prob = np.clip(0.35 + 0.30 * suppression_strength - 0.20 * speech_protection, 0.08, 0.85)
        theta = q_prob / (1.0 - q_prob)
        # 计算理论 SPP
        spp_smooth = likelihood / (likelihood + theta)
        
        # 使用加权融合让“是否有语音”的判断更保守，避免残余噪声被过度保留
        spp_sigmoid = 1.0 / (1.0 + np.exp(-1.2 * (post_snr - speech_threshold)))
        spp_blend = float(np.clip(0.55 + 0.35 * suppression_strength - 0.25 * speech_protection, 0.35, 0.85))
        speech_presence = spp_blend * spp_smooth + (1.0 - spp_blend) * spp_sigmoid

        # 3. OMLSA (Optimally Modified LSA) 增益融合
        # 充分认定为有语音时使用 LSA 增益，否则柔和降落至底噪增益下限
        gain = (gain_lsa ** speech_presence) * (gain_floor ** (1.0 - speech_presence))
        # 后滤波：对“疑似无语音”的频点再压一层，增强可感知的降噪量
        noise_mask = np.clip(1.0 - speech_presence, 0.0, 1.0)
        gain *= 1.0 - post_filter_strength * (noise_mask ** 1.40)

        # 自适应下限：语音存在概率越低，允许更低的底噪增益
        adaptive_floor = np.clip(gain_floor * (0.25 + 0.75 * speech_presence), 0.01, gain_floor)
        gain = np.clip(gain, adaptive_floor, 1.0)

        # 4. 对频率增益做邻域平滑，降低“音乐噪声”的频点抖动。
        gain = np.pad(gain, (1, 1), mode="edge")
        gain = np.convolve(gain, frequency_smoothing_kernel, mode="valid")
        # 5. 时间方向融合
        gain = gain_blend * previous_gain + (1.0 - gain_blend) * gain

        enhanced_spectrum[:, frame_index] = gain * spectrum[:, frame_index]

        # 自适应噪声跟踪：
        # - 有语音时：平滑系数接近 1，更新慢，保护语音
        # - 无语音时：平滑系数较小，更新快，追踪非平稳噪声
        noise_smoothing = speech_noise_smoothing * speech_presence + non_speech_noise_smoothing * (1.0 - speech_presence)
        noise_psd = noise_smoothing * noise_psd + (1.0 - noise_smoothing) * current_power
        # 估计当前净语音功率，作为下一帧先验 SNR 的历史项。
        previous_clean_power = np.maximum(np.square(gain) * current_power - 0.15 * adjusted_noise_psd, EPSILON)
        previous_gain = gain

    # iSTFT 回到时域信号。
    _, restored = istft(
        enhanced_spectrum,
        fs=sample_rate,
        window="hann",
        nperseg=frame_length,
        noverlap=frame_length - hop,
        input_onesided=True,
        boundary=True,
    )
    restored = restored[: audio.size]
    # RMS 能量补偿：降噪后常出现整体音量下降，限制补偿倍数避免过放大。
    input_rms = float(np.sqrt(np.mean(audio**2) + EPSILON))
    output_rms = float(np.sqrt(np.mean(restored**2) + EPSILON))
    rms_compensation = np.clip(input_rms / output_rms, 1.0, rms_limit)
    restored = rms_compensation * restored
    # 干湿混合（dry/wet）：少量回灌原始信号，减少过处理带来的失真感。
    restored = (1.0 - dry_mix) * restored + dry_mix * audio
    return normalize_audio(restored)


def run_denoise_algorithm(
    algorithm: str,
    samples: np.ndarray,
    sample_rate: int,
    model_dir: str | Path | None = None,
    mmse_parameters: MMSEParameters | None = None,
) -> np.ndarray:
    if algorithm == "deepfilter":
        return get_deepfilter_service(model_dir=model_dir).enhance_samples(samples, sample_rate)
    if algorithm == "mmse":
        return mmse_decision_directed_denoise(samples, sample_rate, mmse_parameters=mmse_parameters)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _prepare_reference(
    reference_samples: np.ndarray | None,
    reference_sr: int | None,
    target_sr: int,
) -> np.ndarray | None:
    if reference_samples is None or reference_sr is None:
        return None
    return resample_audio(reference_samples, reference_sr, target_sr)


def build_audio_analysis(
    algorithm: str,
    samples: np.ndarray,
    sample_rate: int,
    reference_samples: np.ndarray | None,
) -> AudioAnalysis:
    normalized = normalize_audio(samples)
    return AudioAnalysis(
        algorithm=algorithm,
        samples=normalized,
        sample_rate=sample_rate,
        spectrogram=compute_spectrogram(normalized, sample_rate),
        metrics=compute_metrics(reference_samples, normalized, sample_rate),
    )


def compare_denoising_algorithms(
    noisy_samples: np.ndarray,
    sample_rate: int,
    reference_samples: np.ndarray | None = None,
    reference_sr: int | None = None,
    model_dir: str | Path | None = None,
    mmse_parameters: MMSEParameters | None = None,
) -> ComparisonResult:
    noisy = normalize_audio(noisy_samples)
    reference = _prepare_reference(reference_samples, reference_sr, sample_rate)
    deepfilter_output = run_denoise_algorithm("deepfilter", noisy, sample_rate, model_dir=model_dir)
    mmse_output = run_denoise_algorithm("mmse", noisy, sample_rate, model_dir=model_dir, mmse_parameters=mmse_parameters)

    return ComparisonResult(
        noisy=build_audio_analysis("原始输入", noisy, sample_rate, reference),
        deepfilter=build_audio_analysis("DeepFilterNet2", deepfilter_output, sample_rate, reference),
        mmse=build_audio_analysis("MMSE + DD + 自适应噪声", mmse_output, sample_rate, reference),
        diagnosis=diagnose_noise(noisy, sample_rate),
        reference_metrics_ready=reference is not None,
    )