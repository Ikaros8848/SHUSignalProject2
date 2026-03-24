from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import queue
import sys

import numpy as np
import pyqtgraph as pg
import sounddevice as sd
from PyQt6.QtCore import QObject, QRectF, QRunnable, QSize, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from audio_backend import ComparisonResult, MMSEParameters, compare_denoising_algorithms, compute_spectrogram, load_audio_file


pg.setConfigOptions(antialias=True)


STYLE_SHEET = """
QMainWindow {
    background: #f5f5f7;
}
QWidget {
    color: #1d1d1f;
    font-family: 'Segoe UI Variable', 'Segoe UI', 'Microsoft YaHei UI';
    font-size: 13px;
}
QFrame#Card {
    background: #ffffff;
    border: 1px solid #e5e5ea;
    border-radius: 20px;
}
QFrame#HeaderPanel {
    background: #f9fafc;
    border: 1px solid #e2e6ee;
    border-radius: 18px;
}
QLabel#Title {
    font-family: 'Segoe UI Semibold', 'Microsoft YaHei UI';
    font-size: 28px;
    color: #111111;
}
QLabel#Subtitle {
    color: #6e6e73;
    font-size: 13px;
}
QLabel#HeaderCaption {
    color: #6e6e73;
    font-size: 12px;
}
QLabel#SectionTitle {
    font-family: 'Segoe UI Semibold', 'Microsoft YaHei UI';
    font-size: 15px;
    color: #1d1d1f;
}
QLabel#StepLabel {
    color: #0071e3;
    font-family: 'Segoe UI Semibold', 'Microsoft YaHei UI';
    font-size: 12px;
}
QLabel#HintText {
    color: #6e6e73;
}
QLabel#InfoChip {
    padding: 10px 14px;
    border-radius: 12px;
    background: #f2f2f7;
    color: #3a3a3c;
}
QLabel#InputBadge {
    padding: 8px 12px;
    border-radius: 999px;
    background: #e8f2ff;
    color: #005ecb;
    font-family: 'Segoe UI Semibold', 'Microsoft YaHei UI';
}
QLabel#InputBadgeRecording {
    padding: 8px 12px;
    border-radius: 999px;
    background: #ffe9e7;
    color: #c93428;
    font-family: 'Segoe UI Semibold', 'Microsoft YaHei UI';
}
QPushButton {
    background: #0071e3;
    border: 1px solid #0071e3;
    border-radius: 12px;
    color: #ffffff;
    font-family: 'Microsoft YaHei UI';
    font-size: 9px;
    font-weight: 500;
    min-height: 24px;
    padding: 1px 8px;
}
QPushButton:hover {
    background: #0077ed;
    border: 1px solid #0077ed;
}
QPushButton:pressed {
    background: #0068d1;
}
QPushButton:disabled {
    background: #e5e5ea;
    border: 1px solid #e5e5ea;
    color: #8e8e93;
}
QPushButton#SecondaryButton {
    background: #ffffff;
    border: 1px solid #d2d2d7;
    color: #1d1d1f;
}
QPushButton#SecondaryButton:hover {
    background: #f5f5f7;
    border: 1px solid #c7c7cc;
}
QComboBox, QTableWidget {
    background: #ffffff;
    border: 1px solid #d2d2d7;
    border-radius: 10px;
    padding: 6px;
}
QDoubleSpinBox {
    background: #ffffff;
    border: 1px solid #d2d2d7;
    border-radius: 10px;
    padding: 4px 20px 4px 8px;
}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 18px;
    border-left: 1px solid #e5e5ea;
}
QComboBox QAbstractItemView {
    background: #ffffff;
    selection-background-color: #e8f2ff;
}
QTabWidget::pane {
    border: none;
}
QTabBar::tab {
    background: #ececf1;
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
    padding: 8px 14px;
    margin-right: 8px;
    color: #6e6e73;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #1d1d1f;
}
QProgressBar {
    border-radius: 10px;
    background: #ececf1;
    border: 1px solid #d2d2d7;
    text-align: center;
    color: #1d1d1f;
}
QProgressBar::chunk {
    border-radius: 10px;
    background: #0071e3;
}
QHeaderView::section {
    background: #f5f5f7;
    border: none;
    padding: 8px;
    color: #6e6e73;
}
"""


def format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def decimate_signal(samples: np.ndarray, max_points: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    if samples.size == 0:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)
    if samples.size <= max_points:
        indices = np.arange(samples.size, dtype=np.float32)
        return indices, samples
    step = max(1, math.ceil(samples.size / max_points))
    clipped = samples[::step]
    indices = np.arange(0, step * clipped.size, step, dtype=np.float32)
    return indices, clipped


def shared_spectrogram_levels(*spectrograms: np.ndarray) -> tuple[float, float]:
    valid = [spec[np.isfinite(spec)] for spec in spectrograms if spec.size]
    if not valid:
        return (-80.0, 0.0)
    minimum = min(float(np.min(spec)) for spec in valid)
    maximum = max(float(np.max(spec)) for spec in valid)
    if minimum >= maximum:
        return (minimum - 1.0, maximum + 1.0)
    return (minimum, maximum)


class PlotCard(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("Card")
        self.card_layout = QVBoxLayout(self)
        self.card_layout.setContentsMargins(18, 16, 18, 18)
        self.card_layout.setSpacing(10)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("SectionTitle")
        self.card_layout.addWidget(self.title_label)


class WaveformCard(PlotCard):
    def __init__(self, title: str, color: str) -> None:
        super().__init__(title)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#f8fbff")
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.setClipToView(True)
        self.plot.getViewBox().setBorder(pg.mkPen("#d7e0ea", width=1.2))
        self.plot.getAxis("left").setTextPen(pg.mkPen("#8e8e93"))
        self.plot.getAxis("bottom").setTextPen(pg.mkPen("#8e8e93"))
        self.plot.setLabel("left", "Amp")
        self.plot.setLabel("bottom", "Time")
        self.curve = self.plot.plot(pen=pg.mkPen(color, width=2.4))
        self.card_layout.addWidget(self.plot)
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        self.curve.setData(np.array([0.0, 1.0], dtype=np.float32), np.zeros(2, dtype=np.float32))
        view_box = self.plot.getViewBox()
        view_box.setXRange(0.0, 1.0, padding=0.01)
        view_box.setYRange(-0.25, 0.25, padding=0.02)

    def _resolve_amplitude_range(self, samples: np.ndarray) -> float:
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak <= 0.0:
            return 0.25
        if peak < 0.05:
            return 0.06
        return min(1.05, peak * 1.18)

    def set_signal(self, samples: np.ndarray, sample_rate: int) -> None:
        if samples.size == 0:
            self._show_placeholder()
            return
        indices, clipped = decimate_signal(samples)
        time_axis = indices / float(sample_rate)
        self.curve.setData(time_axis, clipped)
        duration = float(time_axis.max()) if time_axis.size > 1 else max(float(samples.size) / float(sample_rate), 0.25)
        amplitude = self._resolve_amplitude_range(clipped)
        view_box = self.plot.getViewBox()
        view_box.setXRange(0.0, max(duration, 0.25), padding=0.01)
        view_box.setYRange(-amplitude, amplitude, padding=0.04)


class ComparisonWaveformCard(PlotCard):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#f8fbff")
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.setClipToView(True)
        self.plot.getViewBox().setBorder(pg.mkPen("#d7e0ea", width=1.2))
        self.plot.getAxis("left").setTextPen(pg.mkPen("#8e8e93"))
        self.plot.getAxis("bottom").setTextPen(pg.mkPen("#8e8e93"))
        self.plot.setLabel("left", "Amp")
        self.plot.setLabel("bottom", "Time")
        self.noisy_curve = self.plot.plot(pen=pg.mkPen("#0a84ff", width=1.8), name="原始")
        self.deepfilter_curve = self.plot.plot(pen=pg.mkPen("#30d158", width=1.8), name="DeepFilter")
        self.mmse_curve = self.plot.plot(pen=pg.mkPen("#ff9f0a", width=1.8), name="MMSE")
        self.card_layout.addWidget(self.plot)
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        x = np.array([0.0, 1.0], dtype=np.float32)
        y = np.zeros(2, dtype=np.float32)
        self.noisy_curve.setData(x, y)
        self.deepfilter_curve.setData(x, y)
        self.mmse_curve.setData(x, y)
        view_box = self.plot.getViewBox()
        view_box.setXRange(0.0, 1.0, padding=0.01)
        view_box.setYRange(-0.25, 0.25, padding=0.02)

    def _resolve_amplitude_range(self, samples: np.ndarray) -> float:
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak <= 0.0:
            return 0.25
        if peak < 0.05:
            return 0.06
        return min(1.05, peak * 1.18)

    def set_signals(self, noisy: np.ndarray, deepfilter: np.ndarray, mmse: np.ndarray, sample_rate: int) -> None:
        if noisy.size == 0 and deepfilter.size == 0 and mmse.size == 0:
            self._show_placeholder()
            return

        noisy_idx, noisy_clip = decimate_signal(noisy)
        deep_idx, deep_clip = decimate_signal(deepfilter)
        mmse_idx, mmse_clip = decimate_signal(mmse)

        noisy_x = noisy_idx / float(sample_rate) if noisy_idx.size else np.zeros(1, dtype=np.float32)
        deep_x = deep_idx / float(sample_rate) if deep_idx.size else np.zeros(1, dtype=np.float32)
        mmse_x = mmse_idx / float(sample_rate) if mmse_idx.size else np.zeros(1, dtype=np.float32)

        self.noisy_curve.setData(noisy_x, noisy_clip)
        self.deepfilter_curve.setData(deep_x, deep_clip)
        self.mmse_curve.setData(mmse_x, mmse_clip)

        max_duration = max(
            float(noisy_x.max()) if noisy_x.size else 0.0,
            float(deep_x.max()) if deep_x.size else 0.0,
            float(mmse_x.max()) if mmse_x.size else 0.0,
            0.25,
        )
        amplitude = self._resolve_amplitude_range(
            np.concatenate([arr for arr in [noisy_clip, deep_clip, mmse_clip] if arr.size])
        )
        view_box = self.plot.getViewBox()
        view_box.setXRange(0.0, max_duration, padding=0.01)
        view_box.setYRange(-amplitude, amplitude, padding=0.04)


class SpectrogramCard(PlotCard):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.plot = pg.PlotWidget()
        self.plot.setBackground((0, 0, 0, 0))
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.getAxis("left").setTextPen(pg.mkPen("#8e8e93"))
        self.plot.getAxis("bottom").setTextPen(pg.mkPen("#8e8e93"))
        self.plot.setLabel("left", "Hz")
        self.plot.setLabel("bottom", "Time")
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        self.image.setColorMap(pg.colormap.get("CET-L4") or "CET-L4")
        self.card_layout.addWidget(self.plot)

    def set_spectrogram(
        self,
        freqs: np.ndarray,
        times: np.ndarray,
        magnitude_db: np.ndarray,
        levels: tuple[float, float] | None = None,
    ) -> None:
        if magnitude_db.size == 0:
            self.image.clear()
            return
        rect = QRectF(
            float(times[0]) if times.size else 0.0,
            float(freqs[0]) if freqs.size else 0.0,
            float(times[-1] - times[0]) if times.size > 1 else 1.0,
            float(freqs[-1] - freqs[0]) if freqs.size > 1 else 1.0,
        )
        image = np.flipud(magnitude_db.T)
        self.image.setImage(image, autoLevels=False)
        if levels is None:
            levels = (float(np.nanmin(magnitude_db)), float(np.nanmax(magnitude_db)))
        self.image.setLevels(levels)
        self.image.setRect(rect)


class AnimatedTitle(QWidget):
    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = text
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._advance_animation)
        self._timer.start()
        self.setMinimumHeight(72)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(920, 76)

    def _advance_animation(self) -> None:
        self._phase = (self._phase + 0.018) % 1.0
        self.update()

    def _title_font(self) -> QFont:
        font = QFont("Segoe UI Variable", 24)
        font.setBold(True)
        width = max(self.width() - 32, 240)
        if width < 760:
            font.setPointSize(20)
        if width < 620:
            font.setPointSize(17)
        if width < 500:
            font.setPointSize(15)
        return font

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        rect = self.rect().adjusted(6, 6, -6, -6)
        font = self._title_font()
        painter.setFont(font)

        text_path = QPainterPath()
        text_path.addText(0.0, 0.0, font, self._text)
        text_bounds = text_path.boundingRect()
        translate_x = rect.left() + 6.0
        translate_y = rect.top() + (rect.height() - text_bounds.height()) / 2.0 - text_bounds.top() - 2.0
        text_path = QPainterPath(text_path)
        text_path.translate(translate_x, translate_y)
        text_bounds = text_path.boundingRect()

        shadow_path = QPainterPath(text_path)
        shadow_path.translate(0.0, 4.0)
        painter.fillPath(shadow_path, QColor(20, 24, 48, 38))

        base_gradient = QLinearGradient(text_bounds.left(), text_bounds.top(), text_bounds.right(), text_bounds.bottom())
        base_gradient.setColorAt(0.0, QColor("#1f6bff"))
        base_gradient.setColorAt(0.35, QColor("#ff4fd8"))
        base_gradient.setColorAt(0.68, QColor("#ff9a3c"))
        base_gradient.setColorAt(1.0, QColor("#00c2ff"))
        painter.fillPath(text_path, base_gradient)

        shimmer_center = text_bounds.left() + (text_bounds.width() * 1.4) * self._phase - text_bounds.width() * 0.2
        shimmer_gradient = QLinearGradient(shimmer_center - 120.0, text_bounds.top(), shimmer_center + 120.0, text_bounds.bottom())
        shimmer_gradient.setColorAt(0.0, QColor(255, 255, 255, 0))
        shimmer_gradient.setColorAt(0.5, QColor(255, 255, 255, 170))
        shimmer_gradient.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.save()
        painter.setClipPath(text_path)
        painter.fillRect(text_bounds.adjusted(-40.0, 0.0, 40.0, 0.0), shimmer_gradient)
        painter.restore()

        painter.strokePath(text_path, QPen(QColor(255, 255, 255, 110), 1.15))

        line_y = min(rect.bottom() - 8.0, text_bounds.bottom() + 10.0)
        line_gradient = QLinearGradient(text_bounds.left(), line_y, text_bounds.right(), line_y)
        line_gradient.setColorAt(0.0, QColor("#1f6bff"))
        line_gradient.setColorAt(0.5, QColor("#ffffff"))
        line_gradient.setColorAt(1.0, QColor("#ff4fd8"))
        painter.setPen(QPen(line_gradient, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(
            int(text_bounds.left()),
            int(line_y),
            int(text_bounds.left() + text_bounds.width() * (0.25 + 0.55 * abs(math.sin(self._phase * math.tau)))),
            int(line_y),
        )

        for index in range(4):
            orbit = (self._phase + index * 0.19) % 1.0
            dot_x = rect.left() + rect.width() * orbit
            dot_y = rect.top() + 10.0 + 10.0 * math.sin((orbit + index * 0.2) * math.tau)
            radius = 2.2 + 1.2 * math.sin((orbit + 0.15) * math.tau)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 115 if index % 2 == 0 else 75))
            painter.drawEllipse(QRectF(dot_x, dot_y, radius * 2.0, radius * 2.0))


class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)


class ComparisonTask(QRunnable):
    def __init__(
        self,
        noisy_samples: np.ndarray,
        sample_rate: int,
        reference_samples: np.ndarray | None,
        reference_sr: int | None,
        model_dir: Path,
        mmse_parameters: MMSEParameters | None,
    ) -> None:
        super().__init__()
        self.noisy_samples = noisy_samples.astype(np.float32, copy=True)
        self.sample_rate = sample_rate
        self.reference_samples = None if reference_samples is None else reference_samples.astype(np.float32, copy=True)
        self.reference_sr = reference_sr
        self.model_dir = model_dir
        self.mmse_parameters = mmse_parameters
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = compare_denoising_algorithms(
                self.noisy_samples,
                self.sample_rate,
                reference_samples=self.reference_samples,
                reference_sr=self.reference_sr,
                model_dir=self.model_dir,
                mmse_parameters=self.mmse_parameters,
            )
        except Exception as exc:
            self.signals.error.emit(str(exc))
            return
        self.signals.finished.emit(result)


@dataclass(slots=True)
class AudioAsset:
    samples: np.ndarray
    sample_rate: int
    name: str


class DenoiseStudio(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.model_dir = Path(__file__).resolve().parent
        self.thread_pool = QThreadPool(self)
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.capture_stream: sd.InputStream | None = None
        self.captured_chunks: list[np.ndarray] = []
        self.live_input_buffer = np.zeros(0, dtype=np.float32)
        self.noisy_asset: AudioAsset | None = None
        self.reference_asset: AudioAsset | None = None
        self.result: ComparisonResult | None = None
        self.is_recording = False
        self.processing_busy = False
        self.total_captured_frames = 0

        self.setWindowTitle("信号与系统2课程项目 by cyx")
        self.resize(1360, 860)
        self.setMinimumSize(1120, 700)
        self._build_ui()
        self._refresh_record_toggle_button()
        self._bind_timers()
        self._clear_reference_views()
        self._show_metrics_as_unavailable()
        self._load_default_reference_audio()

    def _create_shared_controls(self) -> None:
        self.status_chip = QLabel("等待输入音频")
        self.status_chip.setObjectName("InfoChip")
        self.status_chip.setWordWrap(True)
        self.input_source_badge = QLabel("当前输入来源: 未选择")
        self.input_source_badge.setObjectName("InputBadge")
        self.noisy_file_label = QLabel("当前待处理音频: 未选择")
        self.noisy_file_label.setObjectName("HeaderCaption")
        self.noisy_file_label.setWordWrap(True)
        self.reference_file_label = QLabel("参考语音: 未选择")
        self.reference_file_label.setObjectName("HeaderCaption")
        self.reference_file_label.setWordWrap(True)

        self.workflow_hint_label = QLabel("输入音频后可直接在顶部开始降噪，详细结果仍在其他标签页查看。")
        self.workflow_hint_label.setObjectName("HintText")
        self.workflow_hint_label.setWordWrap(True)
        self.workflow_hint_label.setVisible(False)
        self.result_hint_label = QLabel("等待新的双算法对比结果。")
        self.result_hint_label.setObjectName("HintText")
        self.result_hint_label.setWordWrap(True)
        self.result_hint_label.setVisible(False)

        self.load_noisy_button = QPushButton("导入待处理音频")
        self.add_gaussian_noise_button = QPushButton("添加高斯噪声")
        self.add_gaussian_noise_button.setObjectName("SecondaryButton")
        self.load_reference_button = QPushButton("导入参考语音")
        self.clear_reference_button = QPushButton("清空参考语音")
        self.clear_reference_button.setObjectName("SecondaryButton")
        self.clear_reference_button.setEnabled(False)
        self.record_toggle_button = QPushButton("●录")
        self.record_toggle_button.setObjectName("SecondaryButton")
        self.record_toggle_button.setMaximumWidth(64)
        self.record_toggle_button.setMinimumWidth(56)
        self.record_toggle_button.setToolTip("点击开始/停止录音")
        self.process_button = QPushButton("开始降噪")

        self.mmse_suppression_spin = QDoubleSpinBox()
        self.mmse_suppression_spin.setRange(0.0, 1.0)
        self.mmse_suppression_spin.setSingleStep(0.05)
        self.mmse_suppression_spin.setDecimals(2)
        self.mmse_suppression_spin.setValue(0.68)
        self.mmse_suppression_spin.setFixedWidth(96)

        self.mmse_smoothing_spin = QDoubleSpinBox()
        self.mmse_smoothing_spin.setRange(0.0, 1.0)
        self.mmse_smoothing_spin.setSingleStep(0.05)
        self.mmse_smoothing_spin.setDecimals(2)
        self.mmse_smoothing_spin.setValue(0.60)
        self.mmse_smoothing_spin.setFixedWidth(96)

        self.mmse_protection_spin = QDoubleSpinBox()
        self.mmse_protection_spin.setRange(0.0, 1.0)
        self.mmse_protection_spin.setSingleStep(0.05)
        self.mmse_protection_spin.setDecimals(2)
        self.mmse_protection_spin.setValue(0.50)
        self.mmse_protection_spin.setFixedWidth(96)

        self.play_noisy_button = QPushButton("播放原始输入")
        self.play_noisy_button.setObjectName("SecondaryButton")
        self.play_deepfilter_button = QPushButton("播放 DeepFilterNet2")
        self.play_deepfilter_button.setObjectName("SecondaryButton")
        self.play_mmse_button = QPushButton("播放 MMSE")
        self.play_mmse_button.setObjectName("SecondaryButton")
        self.play_noisy_button.setEnabled(False)
        self.play_deepfilter_button.setEnabled(False)
        self.play_mmse_button.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(240)
        self.advanced_button = QPushButton("高级功能")
        self.advanced_button.setObjectName("SecondaryButton")
        self.advanced_button.setMaximumWidth(92)

        self.workflow_hint_label.setVisible(True)
        self.result_hint_label.setVisible(True)

        for button in [
            self.load_noisy_button,
            self.add_gaussian_noise_button,
            self.load_reference_button,
            self.clear_reference_button,
            self.record_toggle_button,
            self.process_button,
            self.advanced_button,
            self.play_noisy_button,
            self.play_deepfilter_button,
            self.play_mmse_button,
        ]:
            button.setMinimumHeight(24)

    def _create_advanced_dialog(self) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle("高级功能")
        dialog.setModal(False)
        dialog.resize(560, 420)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        reference_card = QFrame()
        reference_card.setObjectName("Card")
        reference_layout = QVBoxLayout(reference_card)
        reference_layout.setContentsMargins(12, 10, 12, 10)
        reference_layout.setSpacing(8)
        reference_title = QLabel("参考语音")
        reference_title.setObjectName("SectionTitle")
        reference_row = QHBoxLayout()
        reference_row.setContentsMargins(0, 0, 0, 0)
        reference_row.setSpacing(8)
        reference_row.addWidget(self.load_reference_button)
        reference_row.addWidget(self.clear_reference_button)
        reference_row.addWidget(self.reference_file_label, 1)
        reference_layout.addWidget(reference_title)
        reference_layout.addLayout(reference_row)

        mmse_card = QFrame()
        mmse_card.setObjectName("Card")
        mmse_layout = QVBoxLayout(mmse_card)
        mmse_layout.setContentsMargins(12, 10, 12, 10)
        mmse_layout.setSpacing(8)
        mmse_title = QLabel("MMSE 参数")
        mmse_title.setObjectName("SectionTitle")
        mmse_form = QFormLayout()
        mmse_form.setContentsMargins(0, 0, 0, 0)
        mmse_form.setHorizontalSpacing(12)
        mmse_form.setVerticalSpacing(8)
        mmse_form.addRow("MMSE 抑制", self.mmse_suppression_spin)
        mmse_form.addRow("MMSE 平滑", self.mmse_smoothing_spin)
        mmse_form.addRow("MMSE 保真", self.mmse_protection_spin)
        mmse_layout.addWidget(mmse_title)
        mmse_layout.addLayout(mmse_form)

        hint_card = QFrame()
        hint_card.setObjectName("Card")
        hint_layout = QVBoxLayout(hint_card)
        hint_layout.setContentsMargins(12, 10, 12, 10)
        hint_layout.setSpacing(8)
        hint_title = QLabel("状态提示")
        hint_title.setObjectName("SectionTitle")
        hint_layout.addWidget(hint_title)
        hint_layout.addWidget(self.workflow_hint_label)
        hint_layout.addWidget(self.result_hint_label)
        hint_layout.addWidget(self.progress_bar)

        layout.addWidget(reference_card)
        layout.addWidget(mmse_card)
        layout.addWidget(hint_card)
        return dialog

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        self._create_shared_controls()
        self.advanced_dialog = self._create_advanced_dialog()

        header = QFrame()
        header.setObjectName("Card")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(18, 14, 18, 14)
        header_layout.setSpacing(10)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(12)

        title_panel = QVBoxLayout()
        title_panel.setContentsMargins(0, 0, 0, 0)
        title_panel.setSpacing(6)
        title_label = AnimatedTitle("信号与系统2课程项目 by cyx")
        subtitle_label = QLabel("极简语音降噪界面：主界面聚焦输入波形与关键按钮")
        subtitle_label.setObjectName("Subtitle")
        title_panel.addWidget(title_label)
        title_panel.addWidget(subtitle_label)

        info_panel = QVBoxLayout()
        info_panel.setContentsMargins(0, 0, 0, 0)
        info_panel.setSpacing(8)
        info_panel.addWidget(self.status_chip)
        info_panel.addWidget(self.input_source_badge)

        top_row.addLayout(title_panel, 5)
        top_row.addLayout(info_panel, 3)
        header_layout.addLayout(top_row)

        asset_row = QHBoxLayout()
        asset_row.setContentsMargins(0, 0, 0, 0)
        asset_row.setSpacing(12)
        asset_row.addWidget(self.noisy_file_label, 1)
        header_layout.addLayout(asset_row)

        toolbar_panel = QFrame()
        toolbar_panel.setObjectName("HeaderPanel")
        toolbar_layout = QVBoxLayout(toolbar_panel)
        toolbar_layout.setContentsMargins(12, 10, 12, 10)
        toolbar_layout.setSpacing(8)

        controls_grid = QGridLayout()
        controls_grid.setContentsMargins(0, 0, 0, 0)
        controls_grid.setHorizontalSpacing(10)
        controls_grid.setVerticalSpacing(8)

        quick_panel = QFrame()
        quick_layout = QVBoxLayout(quick_panel)
        quick_layout.setContentsMargins(0, 0, 0, 0)
        quick_layout.setSpacing(6)
        quick_caption = QLabel("关键操作")
        quick_caption.setObjectName("HeaderCaption")
        quick_buttons = QGridLayout()
        quick_buttons.setContentsMargins(0, 0, 0, 0)
        quick_buttons.setHorizontalSpacing(8)
        quick_buttons.setVerticalSpacing(6)
        quick_buttons.addWidget(self.load_noisy_button, 0, 0)
        quick_buttons.addWidget(self.record_toggle_button, 0, 1)
        quick_buttons.addWidget(self.add_gaussian_noise_button, 0, 2)
        quick_buttons.addWidget(self.play_noisy_button, 0, 3)
        quick_buttons.addWidget(self.process_button, 0, 4)
        for index in range(5):
            quick_buttons.setColumnStretch(index, 1)

        self.post_denoise_panel = QWidget()
        post_denoise_layout = QHBoxLayout(self.post_denoise_panel)
        post_denoise_layout.setContentsMargins(0, 0, 0, 0)
        post_denoise_layout.setSpacing(8)
        post_denoise_label = QLabel("降噪后试听")
        post_denoise_label.setObjectName("HeaderCaption")
        post_denoise_layout.addWidget(post_denoise_label)
        post_denoise_layout.addWidget(self.play_deepfilter_button)
        post_denoise_layout.addWidget(self.play_mmse_button)
        post_denoise_layout.addWidget(self.progress_bar, 1)
        self.post_denoise_panel.setVisible(False)

        quick_layout.addWidget(quick_caption)
        quick_layout.addLayout(quick_buttons)
        quick_layout.addWidget(self.post_denoise_panel)

        advanced_row = QHBoxLayout()
        advanced_row.setContentsMargins(0, 0, 0, 0)
        advanced_row.setSpacing(8)
        advanced_row.addWidget(self.advanced_button)
        advanced_row.addStretch(1)

        controls_grid.addWidget(quick_panel, 0, 0)
        controls_grid.addLayout(advanced_row, 1, 0)
        controls_grid.setColumnStretch(0, 1)

        toolbar_layout.addLayout(controls_grid)
        header_layout.addWidget(toolbar_panel)
        outer.addWidget(header)

        self.main_tabs = QTabWidget()
        outer.addWidget(self.main_tabs, 1)

        self.main_tabs.addTab(self._create_workbench_tab(), "开始页")
        self.main_tabs.addTab(self._create_waveform_result_tab(), "波形")
        self.main_tabs.addTab(self._create_spectrogram_result_tab(), "频谱")
        self.main_tabs.addTab(self._create_evaluation_result_tab(), "客观评估")
        self._set_result_tabs_visible(False)

        self.load_noisy_button.clicked.connect(self.load_noisy_audio)
        self.add_gaussian_noise_button.clicked.connect(self.add_gaussian_noise)
        self.load_reference_button.clicked.connect(self.load_reference_audio)
        self.clear_reference_button.clicked.connect(self.clear_reference_audio)
        self.record_toggle_button.clicked.connect(self._toggle_recording)
        self.process_button.clicked.connect(self.run_comparison)
        self.play_noisy_button.clicked.connect(lambda: self.play_variant("noisy"))
        self.play_deepfilter_button.clicked.connect(lambda: self.play_variant("deepfilter"))
        self.play_mmse_button.clicked.connect(lambda: self.play_variant("mmse"))
        self.advanced_button.clicked.connect(self._open_advanced_dialog)
        self.mmse_suppression_spin.valueChanged.connect(self._handle_mmse_parameters_changed)
        self.mmse_smoothing_spin.valueChanged.connect(self._handle_mmse_parameters_changed)
        self.mmse_protection_spin.valueChanged.connect(self._handle_mmse_parameters_changed)

    def _open_advanced_dialog(self) -> None:
        self.advanced_dialog.show()
        self.advanced_dialog.raise_()
        self.advanced_dialog.activateWindow()

    def _set_post_denoise_controls_visible(self, visible: bool) -> None:
        self.post_denoise_panel.setVisible(visible)

    def _set_result_tabs_visible(self, visible: bool) -> None:
        for index in [1, 2, 3]:
            self.main_tabs.setTabVisible(index, visible)

    def _toggle_recording(self) -> None:
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def _refresh_record_toggle_button(self) -> None:
        self.record_toggle_button.setText("■停" if self.is_recording else "●录")
        self.record_toggle_button.setEnabled(not self.processing_busy or self.is_recording)

    def _create_workbench_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        live_card = QFrame()
        live_card.setObjectName("Card")
        live_card_layout = QVBoxLayout(live_card)
        live_card_layout.setContentsMargins(20, 16, 20, 18)
        live_card_layout.setSpacing(10)
        live_title = QLabel("实时输入波形")
        live_title.setObjectName("SectionTitle")
        live_hint = QLabel("主界面仅展示输入波形；开始降噪后自动切换到结果页查看前后对比与效果评价。")
        live_hint.setObjectName("HintText")
        live_hint.setWordWrap(True)
        live_card_layout.addWidget(live_title)
        live_card_layout.addWidget(live_hint)

        self.live_input_card = WaveformCard("输入波形", "#0a84ff")
        self.live_input_card.setMinimumHeight(420)
        self.live_input_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        live_card_layout.addWidget(self.live_input_card, 1)
        layout.addWidget(live_card, 1)
        return page

    def _create_waveform_result_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("降噪前后波形对比")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        card = QFrame()
        card.setObjectName("Card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(12)

        waveform_grid = QGridLayout()
        waveform_grid.setContentsMargins(0, 0, 0, 0)
        waveform_grid.setSpacing(12)
        self.noisy_wave_card = WaveformCard("原始输入", "#0582ff")
        self.deepfilter_wave_card = WaveformCard("DeepFilterNet2", "#30d158")
        self.mmse_wave_card = WaveformCard("MMSE", "#ff9f0a")
        self.comparison_wave_card = ComparisonWaveformCard("降噪效果评价波形（叠加）")
        for widget in [self.noisy_wave_card, self.deepfilter_wave_card, self.mmse_wave_card, self.comparison_wave_card]:
            widget.setMinimumHeight(240)
        waveform_grid.addWidget(self.noisy_wave_card, 0, 0)
        waveform_grid.addWidget(self.deepfilter_wave_card, 0, 1)
        waveform_grid.addWidget(self.mmse_wave_card, 1, 0)
        waveform_grid.addWidget(self.comparison_wave_card, 1, 1)
        card_layout.addLayout(waveform_grid)
        layout.addWidget(card, 1)
        return page

    def _create_spectrogram_result_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("去噪前后频谱图")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        card = QFrame()
        card.setObjectName("Card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(12)

        spectrogram_grid = QGridLayout()
        spectrogram_grid.setContentsMargins(0, 0, 0, 0)
        spectrogram_grid.setSpacing(12)
        self.noisy_spec_card = SpectrogramCard("去噪前（原始输入）")
        self.deepfilter_spec_card = SpectrogramCard("去噪后（DeepFilterNet2）")
        self.mmse_spec_card = SpectrogramCard("去噪后（MMSE）")
        self.reference_spec_card = SpectrogramCard("参考语音频谱图")
        for widget in [self.noisy_spec_card, self.deepfilter_spec_card, self.mmse_spec_card, self.reference_spec_card]:
            widget.setMinimumHeight(240)
        spectrogram_grid.addWidget(self.noisy_spec_card, 0, 0)
        spectrogram_grid.addWidget(self.deepfilter_spec_card, 0, 1)
        spectrogram_grid.addWidget(self.mmse_spec_card, 1, 0)
        spectrogram_grid.addWidget(self.reference_spec_card, 1, 1)
        card_layout.addLayout(spectrogram_grid)
        layout.addWidget(card, 1)
        return page

    def _create_evaluation_result_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        metrics_container = QFrame()
        metrics_container.setObjectName("Card")
        metrics_container_layout = QVBoxLayout(metrics_container)
        metrics_container_layout.setContentsMargins(16, 14, 16, 14)
        metrics_container_layout.setSpacing(12)

        self.reference_wave_card = WaveformCard("参考语音时域", "#bf5af2")
        self.reference_wave_card.setMinimumHeight(220)
        metrics_container_layout.addWidget(self.reference_wave_card)

        metrics_card = QFrame()
        metrics_card.setObjectName("Card")
        metrics_layout = QVBoxLayout(metrics_card)
        metrics_layout.setContentsMargins(14, 12, 14, 12)
        metrics_layout.setSpacing(10)
        metrics_title = QLabel("客观评估")
        metrics_title.setObjectName("SectionTitle")
        self.metrics_hint = QLabel("导入参考语音后，会计算 SNR、SegSNR、PESQ；未导入时仍可完成增强与对比。")
        self.metrics_hint.setWordWrap(True)
        self.metrics_table = QTableWidget(3, 4)
        self.metrics_table.setHorizontalHeaderLabels(["算法", "SNR", "SegSNR", "PESQ"])
        vertical_header = self.metrics_table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
        horizontal_header = self.metrics_table.horizontalHeader()
        if horizontal_header is not None:
            horizontal_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.metrics_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        metrics_layout.addWidget(metrics_title)
        metrics_layout.addWidget(self.metrics_hint)
        metrics_layout.addWidget(self.metrics_table, 1)
        metrics_container_layout.addWidget(metrics_card, 1)

        diagnosis_card = QFrame()
        diagnosis_card.setObjectName("Card")
        diagnosis_layout = QVBoxLayout(diagnosis_card)
        diagnosis_layout.setContentsMargins(16, 14, 16, 14)
        diagnosis_layout.setSpacing(12)
        diagnosis_title = QLabel("噪声诊断")
        diagnosis_title.setObjectName("SectionTitle")
        self.diagnosis_label = QLabel("等待频谱分析")
        self.diagnosis_label.setWordWrap(True)
        self.diagnosis_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        diagnosis_layout.addWidget(diagnosis_title)
        diagnosis_layout.addWidget(self.diagnosis_label, 1)

        layout.addWidget(metrics_container, 3)
        layout.addWidget(diagnosis_card, 2)
        return page

    def _bind_timers(self) -> None:
        self.capture_timer = QTimer(self)
        self.capture_timer.setInterval(70)
        self.capture_timer.timeout.connect(self._drain_audio_queue)

    def _set_status(self, text: str) -> None:
        self.status_chip.setText(text)

    def _apply_badge_state(self, label: QLabel, text: str, recording: bool) -> None:
        label.setText(text)
        label.setObjectName("InputBadgeRecording" if recording else "InputBadge")
        style = label.style()
        if style is not None:
            style.unpolish(label)
            style.polish(label)
        label.update()

    def _set_input_source_badge(self, text: str, recording: bool = False) -> None:
        self._apply_badge_state(self.input_source_badge, text, recording)

    def _sync_header_quick_actions(self) -> None:
        return

    def _workspace_audio_dir(self) -> Path:
        return self.model_dir.parent / "audio"

    def _find_default_reference_audio(self) -> Path | None:
        audio_dir = self._workspace_audio_dir()
        if not audio_dir.exists() or not audio_dir.is_dir():
            return None
        supported_suffixes = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".mp4", ".mkv"}
        candidates = sorted(
            [path for path in audio_dir.iterdir() if path.is_file() and path.suffix.lower() in supported_suffixes],
            key=lambda path: path.name,
        )
        if not candidates:
            return None
        for candidate in candidates:
            if "参考" in candidate.stem or "reference" in candidate.stem.lower():
                return candidate
        return candidates[0]

    def _apply_reference_audio(self, path: str | Path, *, default_loaded: bool = False) -> bool:
        try:
            samples, sample_rate = load_audio_file(path)
        except Exception as exc:
            if default_loaded:
                self._set_status(f"默认参考语音载入失败: {Path(path).name}")
            else:
                QMessageBox.critical(self, "载入失败", str(exc))
            return False
        path_obj = Path(path)
        self.reference_asset = AudioAsset(samples=samples, sample_rate=sample_rate, name=path_obj.name)
        self.reference_file_label.setText(f"参考语音: {self.reference_asset.name}")
        self.reference_wave_card.set_signal(samples, sample_rate)
        reference_spec = compute_spectrogram(samples, sample_rate)
        self.reference_spec_card.set_spectrogram(reference_spec.freqs, reference_spec.times, reference_spec.magnitude_db)
        self.clear_reference_button.setEnabled(True)
        if default_loaded:
            self._set_status(f"已自动载入参考语音: {self.reference_asset.name}")
            self.workflow_hint_label.setText("已从 audio 目录自动载入默认参考语音。导入待处理音频后即可直接做客观评估。")
        else:
            self._set_status(f"已载入参考语音: {self.reference_asset.name}")
            self.workflow_hint_label.setText("参考语音已载入。步骤 3 运行后，评估页会显示 SNR、SegSNR 和 PESQ。")
        return True

    def _load_default_reference_audio(self) -> None:
        default_reference = self._find_default_reference_audio()
        if default_reference is None:
            return
        self._apply_reference_audio(default_reference, default_loaded=True)

    def _current_mmse_parameters(self) -> MMSEParameters:
        return MMSEParameters(
            suppression_strength=float(self.mmse_suppression_spin.value()),
            temporal_smoothing=float(self.mmse_smoothing_spin.value()),
            speech_protection=float(self.mmse_protection_spin.value()),
        )

    def _set_metrics_rows(self, rows: list[tuple[str, tuple[str, str, str]]]) -> None:
        for row_index, (name, values) in enumerate(rows):
            items = [
                QTableWidgetItem(name),
                QTableWidgetItem(values[0]),
                QTableWidgetItem(values[1]),
                QTableWidgetItem(values[2]),
            ]
            for column, item in enumerate(items):
                item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                self.metrics_table.setItem(row_index, column, item)

    def _clear_reference_views(self) -> None:
        self.reference_wave_card.set_signal(np.zeros(0, dtype=np.float32), 16000)
        self.reference_spec_card.set_spectrogram(
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        )

    def _show_metrics_as_unavailable(self) -> None:
        self._set_metrics_rows(
            [
                ("原始输入", ("N/A", "N/A", "N/A")),
                ("DeepFilterNet2", ("N/A", "N/A", "N/A")),
                ("MMSE + DD + 自适应", ("N/A", "N/A", "N/A")),
            ]
        )

    def _handle_mmse_parameters_changed(self) -> None:
        if self.result is not None:
            self._set_status("MMSE 参数已更新，请重新开始降噪以刷新结果")
            self.result_hint_label.setText("MMSE 参数已变更。请重新点击“开始降噪”，频谱图和指标会按新参数刷新。")

    def _set_processing_state(self, busy: bool) -> None:
        self.processing_busy = busy
        self.process_button.setEnabled(not busy)
        self._refresh_record_toggle_button()
        self._sync_header_quick_actions()
        self.progress_bar.setVisible(busy)
        if busy:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)

    def load_noisy_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择待处理音频",
            str(self.model_dir),
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.wma *.mp4 *.mkv)",
        )
        if not path:
            return
        try:
            samples, sample_rate = load_audio_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "载入失败", str(exc))
            return
        self.stop_recording(silent=True)
        self.noisy_asset = AudioAsset(samples=samples, sample_rate=sample_rate, name=Path(path).name)
        self.live_input_buffer = samples[-sample_rate * 3 :]
        self.result = None
        self._set_status(f"已载入待处理音频: {self.noisy_asset.name}")
        self.workflow_hint_label.setText("输入音频已就绪。可在步骤 2 调整 MMSE 参数，或直接在步骤 3 开始降噪。")
        self.result_hint_label.setText("当前结果区等待新的双算法对比。若刚改了 MMSE 参数，需要重新运行才能刷新离线频谱。")
        self.noisy_file_label.setText(f"当前待处理音频: {self.noisy_asset.name}")
        self._set_input_source_badge(f"当前输入来源: 文件导入 · {self.noisy_asset.name}")
        self.live_input_card.set_signal(self.live_input_buffer, sample_rate)
        self.noisy_wave_card.set_signal(samples, sample_rate)
        self.deepfilter_wave_card.set_signal(np.zeros(0, dtype=np.float32), sample_rate)
        self.mmse_wave_card.set_signal(np.zeros(0, dtype=np.float32), sample_rate)
        self.comparison_wave_card.set_signals(
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            sample_rate,
        )
        if self.reference_asset is None:
            self._show_metrics_as_unavailable()
        self.play_noisy_button.setEnabled(True)
        self.play_deepfilter_button.setEnabled(False)
        self.play_mmse_button.setEnabled(False)
        self._set_post_denoise_controls_visible(False)
        self._set_result_tabs_visible(False)
        self.main_tabs.setCurrentIndex(0)
        self._sync_header_quick_actions()

    def add_gaussian_noise(self) -> None:
        if self.processing_busy:
            return
        asset = self._get_current_noisy_asset()
        if asset is None:
            QMessageBox.information(self, "缺少输入", "请先导入待处理音频，或录制一段语音。")
            return
        if self.is_recording:
            self.stop_recording(silent=True)
            asset = self._get_current_noisy_asset()
            if asset is None:
                QMessageBox.information(self, "缺少输入", "录音数据为空，无法添加噪声。")
                return

        samples = asset.samples.astype(np.float32, copy=False)
        if samples.size == 0:
            QMessageBox.information(self, "无有效音频", "当前音频为空，无法添加噪声。")
            return

        signal_rms = float(np.sqrt(np.mean(samples**2)))
        noise_std = max(signal_rms * 0.15, 0.005)
        gaussian_noise = np.random.normal(0.0, noise_std, samples.shape).astype(np.float32)
        noisy_samples = np.clip(samples + gaussian_noise, -1.0, 1.0).astype(np.float32)

        self.noisy_asset = AudioAsset(samples=noisy_samples, sample_rate=asset.sample_rate, name=f"{asset.name} + 高斯噪声")
        self.live_input_buffer = noisy_samples[-asset.sample_rate * 3 :]
        self.result = None

        self.noisy_file_label.setText(f"当前待处理音频: {self.noisy_asset.name}")
        self._set_input_source_badge("当前输入来源: 人工注入高斯噪声")
        self._set_status("已添加高斯噪声，可开始降噪对比")
        self.workflow_hint_label.setText("已向当前音频注入高斯白噪声。点击“开始降噪”可查看新结果。")
        self.result_hint_label.setText("输入已改变，旧结果已失效。请重新点击“开始降噪”刷新双算法对比。")

        self.live_input_card.set_signal(self.live_input_buffer, asset.sample_rate)
        self.noisy_wave_card.set_signal(noisy_samples, asset.sample_rate)
        self.deepfilter_wave_card.set_signal(np.zeros(0, dtype=np.float32), asset.sample_rate)
        self.mmse_wave_card.set_signal(np.zeros(0, dtype=np.float32), asset.sample_rate)
        self.comparison_wave_card.set_signals(
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            asset.sample_rate,
        )
        if self.reference_asset is None:
            self._show_metrics_as_unavailable()

        self.play_noisy_button.setEnabled(True)
        self.play_deepfilter_button.setEnabled(False)
        self.play_mmse_button.setEnabled(False)
        self._set_post_denoise_controls_visible(False)
        self._set_result_tabs_visible(False)
        self.main_tabs.setCurrentIndex(0)
        self._sync_header_quick_actions()

    def load_reference_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择参考语音",
            str(self._workspace_audio_dir() if self._workspace_audio_dir().exists() else self.model_dir),
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.wma *.mp4 *.mkv)",
        )
        if not path:
            return
        self._apply_reference_audio(path)

    def clear_reference_audio(self) -> None:
        if self.reference_asset is None:
            return
        self.reference_asset = None
        self.reference_file_label.setText("参考语音: 未选择")
        self.clear_reference_button.setEnabled(False)
        self._clear_reference_views()
        self._show_metrics_as_unavailable()
        if self.result is not None:
            self.metrics_hint.setText("参考语音已清空，请重新开始降噪以刷新客观指标。")
        else:
            self.metrics_hint.setText("导入参考语音后，会计算 SNR、SegSNR、PESQ；未导入时仍可完成增强与对比。")
        self._set_status("参考语音已清空")
        self.workflow_hint_label.setText("参考语音已清空。仍可进行主观试听与时频对比，但客观指标会显示为 N/A。")
        self._sync_header_quick_actions()

    def _audio_callback(self, indata: np.ndarray, _frames: int, _time_info, status) -> None:
        if status:
            self.audio_queue.put(np.zeros(0, dtype=np.float32))
        self.audio_queue.put(indata[:, 0].astype(np.float32, copy=True))

    def start_recording(self) -> None:
        if self.processing_busy or self.is_recording:
            return
        self.audio_queue = queue.Queue()
        self.captured_chunks = []
        self.live_input_buffer = np.zeros(0, dtype=np.float32)
        self.total_captured_frames = 0
        try:
            self.capture_stream = sd.InputStream(
                samplerate=48000,
                channels=1,
                dtype="float32",
                blocksize=2048,
                callback=self._audio_callback,
            )
            self.capture_stream.start()
        except Exception as exc:
            self.capture_stream = None
            QMessageBox.critical(self, "录音失败", str(exc))
            return
        self.is_recording = True
        self.noisy_asset = None
        self.result = None
        self._refresh_record_toggle_button()
        self.play_noisy_button.setEnabled(False)
        self.play_deepfilter_button.setEnabled(False)
        self.play_mmse_button.setEnabled(False)
        self._set_post_denoise_controls_visible(False)
        self._set_result_tabs_visible(False)
        self.main_tabs.setCurrentIndex(0)
        if self.reference_asset is None:
            self._show_metrics_as_unavailable()
        self.capture_timer.start()
        self._set_status("正在录音，仅更新输入波形")
        self.workflow_hint_label.setText("为节省性能，录音阶段不执行实时降噪。完整结果需点击“开始降噪”离线生成。")
        self.noisy_file_label.setText("当前待处理音频: 麦克风实时录制")
        self._set_input_source_badge("当前输入来源: 麦克风录音中", recording=True)
        self._sync_header_quick_actions()

    def stop_recording(self, silent: bool = False) -> None:
        if self.capture_stream is not None:
            try:
                self.capture_stream.stop()
                self.capture_stream.close()
            except Exception:
                pass
            self.capture_stream = None
        self.capture_timer.stop()
        was_recording = self.is_recording
        self.is_recording = False
        self._refresh_record_toggle_button()

        if self.captured_chunks:
            recorded = np.concatenate(self.captured_chunks).astype(np.float32, copy=False)
            self.noisy_asset = AudioAsset(samples=recorded, sample_rate=48000, name="麦克风录音")
            self.noisy_wave_card.set_signal(recorded, 48000)
            self.play_noisy_button.setEnabled(True)
            self.noisy_file_label.setText("当前待处理音频: 麦克风录音")
            self._set_input_source_badge("当前输入来源: 麦克风录音")
        self._sync_header_quick_actions()
        if was_recording and not silent:
            self._set_status("录音结束，可以开始降噪")
            self.result_hint_label.setText("录音已结束。现在可以点击“开始降噪”生成新的波形、频谱和评估结果。")

    def _update_spectrogram_views(self, result: ComparisonResult) -> None:
        reference_spec = None
        spectrogram_arrays = [
            result.noisy.spectrogram.magnitude_db,
            result.deepfilter.spectrogram.magnitude_db,
            result.mmse.spectrogram.magnitude_db,
        ]
        if self.reference_asset is not None:
            reference_spec = compute_spectrogram(self.reference_asset.samples, self.reference_asset.sample_rate)
            spectrogram_arrays.append(reference_spec.magnitude_db)

        levels = shared_spectrogram_levels(*spectrogram_arrays)
        self.noisy_spec_card.set_spectrogram(
            result.noisy.spectrogram.freqs,
            result.noisy.spectrogram.times,
            result.noisy.spectrogram.magnitude_db,
            levels=levels,
        )
        self.deepfilter_spec_card.set_spectrogram(
            result.deepfilter.spectrogram.freqs,
            result.deepfilter.spectrogram.times,
            result.deepfilter.spectrogram.magnitude_db,
            levels=levels,
        )
        self.mmse_spec_card.set_spectrogram(
            result.mmse.spectrogram.freqs,
            result.mmse.spectrogram.times,
            result.mmse.spectrogram.magnitude_db,
            levels=levels,
        )
        if reference_spec is not None:
            self.reference_spec_card.set_spectrogram(
                reference_spec.freqs,
                reference_spec.times,
                reference_spec.magnitude_db,
                levels=levels,
            )
            self.clear_reference_button.setEnabled(True)
        else:
            self._clear_reference_views()

    def _drain_audio_queue(self) -> None:
        if not self.is_recording:
            return
        drained: list[np.ndarray] = []
        while True:
            try:
                chunk = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            if chunk.size:
                drained.append(chunk)
        if not drained:
            return

        merged = np.concatenate(drained)
        self.captured_chunks.append(merged)
        self.total_captured_frames += merged.size
        self.live_input_buffer = np.concatenate((self.live_input_buffer, merged))[-48000 * 3 :]
        self.live_input_card.set_signal(self.live_input_buffer, 48000)

    def _handle_worker_error(self, message: str) -> None:
        self._set_processing_state(False)
        QMessageBox.critical(self, "处理失败", message)
        self._set_status("处理失败，请检查音频格式、模型权重或环境依赖")

    def _get_current_noisy_asset(self) -> AudioAsset | None:
        if self.is_recording and self.captured_chunks:
            recorded = np.concatenate(self.captured_chunks).astype(np.float32, copy=False)
            return AudioAsset(samples=recorded, sample_rate=48000, name="麦克风录音")
        return self.noisy_asset

    def run_comparison(self) -> None:
        asset = self._get_current_noisy_asset()
        if asset is None:
            QMessageBox.information(self, "缺少输入", "请先导入待处理音频，或录制一段语音。")
            return
        self.stop_recording(silent=True)
        self._set_processing_state(True)
        self._set_status("正在运行 DeepFilterNet2 与 MMSE 双算法，请稍候")
        reference_samples = None if self.reference_asset is None else self.reference_asset.samples
        reference_sr = None if self.reference_asset is None else self.reference_asset.sample_rate
        task = ComparisonTask(
            asset.samples,
            asset.sample_rate,
            reference_samples,
            reference_sr,
            self.model_dir,
            self._current_mmse_parameters(),
        )
        task.signals.finished.connect(self._handle_comparison_ready)
        task.signals.error.connect(self._handle_worker_error)
        self.thread_pool.start(task)

    def _handle_comparison_ready(self, result: ComparisonResult) -> None:
        self.result = result
        self._set_processing_state(False)
        self._set_status("双算法处理完成，可从时域、频谱和客观指标维度对比效果")
        self.result_hint_label.setText("结果已更新。当前频谱图已使用统一色标，便于直接比较原始输入、DeepFilterNet2 和 MMSE 的能量变化。")

        self.noisy_wave_card.set_signal(result.noisy.samples, result.noisy.sample_rate)
        self.deepfilter_wave_card.set_signal(result.deepfilter.samples, result.deepfilter.sample_rate)
        self.mmse_wave_card.set_signal(result.mmse.samples, result.mmse.sample_rate)
        self.comparison_wave_card.set_signals(
            result.noisy.samples,
            result.deepfilter.samples,
            result.mmse.samples,
            result.noisy.sample_rate,
        )
        if self.reference_asset is not None:
            self.reference_wave_card.set_signal(self.reference_asset.samples, self.reference_asset.sample_rate)
        else:
            self.reference_wave_card.set_signal(np.zeros(0, dtype=np.float32), result.noisy.sample_rate)

        self._update_spectrogram_views(result)
        self._update_metrics(result)
        self._update_diagnosis(result)
        self.play_noisy_button.setEnabled(True)
        self.play_deepfilter_button.setEnabled(True)
        self.play_mmse_button.setEnabled(True)
        self._set_post_denoise_controls_visible(True)
        self._set_result_tabs_visible(True)
        self.main_tabs.setCurrentIndex(1)
        self._sync_header_quick_actions()

    def _update_metrics(self, result: ComparisonResult) -> None:
        rows = [
            ("原始输入", (format_metric(result.noisy.metrics.snr), format_metric(result.noisy.metrics.seg_snr), format_metric(result.noisy.metrics.pesq))),
            ("DeepFilterNet2", (format_metric(result.deepfilter.metrics.snr), format_metric(result.deepfilter.metrics.seg_snr), format_metric(result.deepfilter.metrics.pesq))),
            ("MMSE + DD + 自适应", (format_metric(result.mmse.metrics.snr), format_metric(result.mmse.metrics.seg_snr), format_metric(result.mmse.metrics.pesq))),
        ]
        self._set_metrics_rows(rows)
        if result.reference_metrics_ready:
            self.metrics_hint.setText("已载入参考语音，表格展示客观指标对比结果。")
        else:
            self.metrics_hint.setText("未导入参考语音，程序仍可运行，但 SNR、SegSNR、PESQ 会显示为 N/A。")

    def _update_diagnosis(self, result: ComparisonResult) -> None:
        diagnosis = result.diagnosis
        band_energies = diagnosis.band_energies
        self.diagnosis_label.setText(
            "\n".join(
                [
                    f"类型诊断: {diagnosis.label}",
                    f"主导频段: {diagnosis.dominant_band}",
                    f"主导频率: {diagnosis.dominant_frequency_hz:.1f} Hz",
                    f"频谱质心: {diagnosis.spectral_centroid_hz:.1f} Hz",
                    f"频谱平坦度: {diagnosis.spectral_flatness:.4f}",
                    f"低/中/高频能量占比: {band_energies['low']:.2%} / {band_energies['mid']:.2%} / {band_energies['high']:.2%}",
                ]
            )
        )

    def play_variant(self, variant: str) -> None:
        try:
            sd.stop()
            if variant == "noisy":
                asset = self._get_current_noisy_asset()
                if asset is None:
                    return
                sd.play(asset.samples, asset.sample_rate)
                return
            if self.result is None:
                return
            if variant == "deepfilter":
                sd.play(self.result.deepfilter.samples, self.result.deepfilter.sample_rate)
                return
            if variant == "mmse":
                sd.play(self.result.mmse.samples, self.result.mmse.sample_rate)
        except Exception as exc:
            QMessageBox.critical(self, "播放失败", str(exc))

    def closeEvent(self, event) -> None:
        self.stop_recording(silent=True)
        try:
            sd.stop()
        except Exception:
            pass
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)
    app.setFont(QFont("Segoe UI", 10))
    window = DenoiseStudio()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
