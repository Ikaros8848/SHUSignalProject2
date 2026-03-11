
&emsp;&emsp;我们在DNS4数据集上训练DeepFilterNet2，总共使用超过500小时的全波段纯净语音(大约)。150 小时的噪声以及150个真实的和60000个模拟的HRTFs。我们将数据分为训练集、验证集和测试集(70%、15%、15%)。Voicebank集split为speaker-exclusive，与测试集没有重叠。我们在Voicebank+Demand测试集[8]和DNS4盲测试集[9]上评估了我们的方法。我们用AdamW对模型进行了100个epoch的训练，并根据验证损失选择最佳模型。

&emsp;&emsp;在这项工作中，我们使用20毫秒的窗口，50%的重叠，以及两个帧的look-ahead，导致总体算法延迟40毫秒。

## 结果
  我们使用Valentini语音库+需求测试集[8]来评估DeepFilterNet2的语音增强性能。因此，我们选择WB-PESQ [19]， STOI[20]和综合指标CSIG, CBAK, COVL[21]。表1显示了DeepFilterNet2与其他先进(SOTA)方法的比较结果。可以发现，DeepFilterNet2实现了sota级别的结果，同时需要最小的每秒乘法累积运算(MACS)。在DeepFilterNet(第2.5节)上，参数的数量略有增加，但该网络能够以两倍多的速度运行，并获得0.27的高PESQ评分。GaGNet[5]实现了类似的RTF，同时具有良好的SE性能。然而，它只在提供整个音频时运行得很快，由于它使用了大的时间卷积核，需要大的时间缓冲区。FRCRN[3]在大多数指标上都能获得最好的结果，但具有较高的计算复杂度，这在嵌入式设备上是不可实现的。

## 桌面程序

本项目现在提供 PyQt 桌面端工作台，支持以下能力：

- 麦克风实时录音与实时波形显示
- DeepFilterNet2 与 MMSE + 决策导向 + 噪声自适应 两套降噪方案对比
- 时域波形、频谱图、噪声主导频段诊断
- SNR、SegSNR、PESQ 客观指标评估

运行方式：

```bash
python DeepFilterNet2/desktop_app.py
```

说明：

- 桌面端支持导入待处理音频、导入参考语音，或直接用麦克风录音。
- 桌面端提供“清空参考语音”按钮，便于在有参考评估和无参考试听两种模式间切换。
- 控制台内置 MMSE 参数面板，可直接调节抑制强度、时间平滑和语音保真度。
- 参考语音是可选项；如果未导入，增强和对比功能仍可正常使用，但 SNR、SegSNR、PESQ 会显示为 N/A。
- 文件导入支持常见音频容器，界面层已开放 wav、mp3、flac、ogg、m4a、aac、wma，以及常见视频容器中的音轨导入；具体解码能力取决于本机可用解码器。
- 原有的接口入口仍保留在 app.py，Gradio 入口仍保留在 gradio_denoise.py。


this project is based on deepfiliternet by EakAip